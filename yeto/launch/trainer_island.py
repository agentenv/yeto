"""Cloud binding of the trainer scheduler: the trainer island on Modal
(rl-trainer-start-stop L5.3c) or on a sky cloud.

With ``--rl-trainer-scheduler s1`` the launcher holds back the island named by
``--rl-trainer-island``: the fleet controller does not start it. The binding
gives the scheduler loop one trainer island per incarnation:

* Modal (:func:`modal_binding`): :class:`ModalTrainerIslands` spawns the held-back
  function (defined on the run's app) with the incarnation environment
  (``trainer_remote.island_env``), tails its log stream for the island events, and
  cancels the call at release. :class:`ModalVolumeFiles` reads and writes the
  run's ``modal-volume://`` checkpoint store;
* sky (:func:`sky_binding`): :class:`SkyTrainerIslands` launches the held-back
  task as a new cluster per incarnation (``<island>-i<n>``), tails the job log, and
  tears the cluster down at release (verified at the cloud). The head reads and
  writes the run's ``s3://`` checkpoint store through its object API
  (``yeto.cloud.s3_store``); the islands use the same API for the sample buffer,
  the commands and the policy release pointer (not their bucket mount);
* both: the head's buffer view is the sample buffer index (Modal: mirrored from
  the Volume; s3: read by object key), backlog counted from the consumed watermark the trainer committed: the running
  incarnation's reported one, else the newest cut's (``CommittedBacklog``). The
  incarnation ids (Modal call, sky cluster) are kept in the head's run directory,
  so a restarted head can release an orphan. Each incarnation's events are also
  written to ``<events dir>/<island>-i<n>.jsonl``, where the launcher reads the
  start-phase summary (``trainer_start``) for the run manifest.
"""

from __future__ import annotations

import io
import json
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from yeto.rl import event_echo
from yeto.rl.engine.object_store import ObjectStore
from yeto.rl.engine.policy_release import LATEST as RELEASE_LATEST
from yeto.rl.engine.policy_release import RELEASE_DIR
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import (
    BUFFER_DIR,
    INDEX,
    CommittedBacklog,
    FileSampleBuffer,
    UnmergedShards,
    needs_shards,
)
from yeto.rl.engine.start_phases import SUBMITTED_ENV
from yeto.rl.engine.trainer_lifecycle import CutLog, CutRef, DurableSampleBuffer, TrainerLifecycle
from yeto.rl.engine.trainer_remote import (
    EVENT,
    RemoteTrainerLifecycle,
    StoreCutLog,
    StoreFiles,
    consumed_watermark,
    island_env,
)

JOBS_FILE = "trainer-islands.json"  # in the trainer loop's journal dir: incarnation -> call id
MIRROR_DIR = "buffer-mirror"
RELEASE_CHECKS = 12
RELEASE_CHECK_DELAY_S = 5.0
# A release during provisioning cancels the launch request; wait this long for
# the request to end before tearing the cluster down (it may still come up).
PROVISION_CANCEL_WAIT_S = 600.0


@dataclass
class TrainerIsland:
    """The held-back trainer island the launcher hands to the binding."""

    name: str
    cloud: str
    modal_cfg: Any = None  # modal_runner.ModalIslandConfig
    modal_ops: Any = None  # modal_runner.ModalOps (app deployed)
    sky_task: Any = None  # sky.Task of a sky island (not launched by the fleet)
    spec: Any = None  # gpu_spec.ClusterSpec of the island
    events_dir: Path | None = None  # the run's echoed-tape dir (None: no tape files)
    ssh: Any = None  # ssh_trainer_island.SshTrainerSpec of an SSH island (cloud "ssh")


@dataclass
class CloudTrainerBinding:
    """What the loop needs from the cloud layer for one run.

    ``lifecycle(first_incarnation)`` builds the trainer lifecycle (e.g.
    ``DriverLifecycle`` whose ``build_driver`` acquires a trainer island and whose
    ``release`` gives it back). ``release_orphan(incarnation)`` gives back the island
    a dead head started. ``data_ended`` tells the samplers finished.
    ``after_restore()`` runs after the journal is read back and an orphan released
    (e.g. reset the head's buffer to the watermark of the newest cut).
    ``wrap_scheduler(scheduler, record)`` decorates the S1 scheduler (a convertible
    pool holds a start no node may take); ``on_finished()`` runs once the loop
    finished (a convertible pool ends its nodes)."""

    lifecycle: Callable[[int], TrainerLifecycle]
    buffer: DurableSampleBuffer
    cuts: CutLog
    release_orphan: Callable[[int], None] | None = None
    data_ended: Callable[[], bool] = lambda: False
    start_events: Callable[[], Iterable[Mapping[str, Any]]] | None = None
    spec: Mapping[str, Any] | None = None
    after_restore: Callable[[], None] | None = None
    wrap_scheduler: Callable[[Any, Callable[..., None]], Any] | None = None
    on_finished: Callable[[], None] | None = None
    needs_first_policy: Callable[[], bool] | None = None


# ---------------------------------------------------------------------------
# Store access from the head
# ---------------------------------------------------------------------------
class ModalVolumeFiles:
    """:class:`~yeto.rl.engine.trainer_remote.StoreFiles` over a Modal Volume via
    the SDK (``read_file`` / ``batch_upload``). ``prefix``: the store's path in the
    Volume (``modal-volume://NAME/PREFIX``)."""

    def __init__(self, volume_name: str, prefix: str = "", *,
                 volume_factory: Callable[[str], Any] | None = None) -> None:
        self.volume_name = volume_name
        self.prefix = prefix.strip("/")
        self._factory = volume_factory
        self._volume: Any = None

    def _vol(self) -> Any:
        if self._volume is None:
            if self._factory is not None:
                self._volume = self._factory(self.volume_name)
            else:
                import modal

                self._volume = modal.Volume.from_name(self.volume_name)
        return self._volume

    def _path(self, rel: str) -> str:
        return f"{self.prefix}/{rel}" if self.prefix else rel

    def read(self, rel: str) -> bytes | None:
        try:
            return b"".join(self._vol().read_file(self._path(rel)))
        except Exception as error:  # noqa: BLE001 - the SDK names a missing file differently by version
            if isinstance(error, FileNotFoundError) or type(error).__name__ == "NotFoundError":
                return None
            raise

    def write(self, rel: str, data: bytes) -> None:
        with self._vol().batch_upload(force=True) as batch:
            batch.put_file(io.BytesIO(data), self._path(rel))


class MirrorStore(CheckpointStore):
    """A read-only local copy of some store files: ``reload`` copies ``paths`` from
    ``files`` (whole files). Lets :class:`FileSampleBuffer` read an index that
    lives in a store this host does not mount."""

    kind = "mirror"

    def __init__(self, root: str | Path, files: StoreFiles, paths: Iterable[str]) -> None:
        super().__init__(root)
        self.files = files
        self.paths = tuple(paths)

    def reload(self) -> None:
        for rel in self.paths:
            data = self.files.read(rel)
            if data is None:
                continue
            path = Path(self.root) / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)

    def commit(self) -> float:
        raise RuntimeError("the head's mirror of the store is read-only")


# ---------------------------------------------------------------------------
# Islands
# ---------------------------------------------------------------------------
class IslandEvents:
    """Feeds on an island's log stream (``collector.feed`` like ``TapeCollector``);
    keeps the ``rl_trainer_lifecycle`` and start-phase events. A record seen before
    is skipped (a reconnected sky log tail replays the job log from its start).
    With ``tape``, every new event is also appended to that file (one JSON record
    per line)."""

    def __init__(self, tape: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._new: list[dict[str, Any]] = []
        self.all: list[dict[str, Any]] = []
        self.tape = None if tape is None else Path(tape)
        self._seen: set[str] = set()

    def feed(self, chunk: Any) -> None:
        for line in str(chunk).splitlines():
            raw = event_echo.parse_line(line)
            if raw is None or raw is event_echo.INVALID:
                continue
            record = json.loads(raw)
            with self._lock:
                if raw in self._seen:
                    continue
                self._seen.add(raw)
                self.all.append(record)
                if record.get("event") == EVENT:
                    self._new.append(record)
                self._append_to_tape(raw)

    def _append_to_tape(self, raw: str) -> None:
        if self.tape is None:
            return
        try:
            self.tape.parent.mkdir(parents=True, exist_ok=True)
            with open(self.tape, "a", encoding="utf-8") as fh:
                fh.write(raw + "\n")
        except OSError as error:  # the in-memory events still drive the scheduler
            print(f"[launcher] trainer event tape {self.tape}: {error}", flush=True)

    def take(self) -> list[dict[str, Any]]:
        with self._lock:
            out, self._new = self._new, []
            return out


def incarnation_tape(events_dir: Path | None, name: str, incarnation: int) -> Path | None:
    """Where the head writes the events of one trainer incarnation (None: nowhere)."""
    return None if events_dir is None else Path(events_dir) / f"{name}-i{int(incarnation)}.jsonl"


def read_jobs(path: Path) -> dict[str, str]:
    """``{incarnation: cloud id}`` the head recorded (Modal call id, sky cluster)."""
    path = Path(path)
    if not path.is_file():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8")))


def save_job(path: Path, incarnation: int, cloud_id: str) -> None:
    path = Path(path)
    jobs = {**read_jobs(path), str(int(incarnation)): str(cloud_id)}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(jobs, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


@dataclass
class _Job:
    incarnation: int
    call_id: str
    events: IslandEvents


class ModalTrainerIslands:
    """:class:`~yeto.rl.engine.trainer_remote.TrainerIslands` on Modal (module doc).

    ``tail(call_id, label, collector)`` streams the call's logs into ``collector``
    (default: the launcher's ``_tail_modal`` on a daemon thread)."""

    def __init__(self, modal_ops: Any, cfg: Any, *, name: str, jobs_path: Path,
                 poll_s: float | None = None, events_dir: Path | None = None,
                 tail: Callable[[str, str, IslandEvents], None] | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.time) -> None:
        self.modal_ops = modal_ops
        self.cfg = cfg
        self.name = name
        self.jobs_path = Path(jobs_path)
        self.poll_s = poll_s
        self.events_dir = events_dir
        self._tail = tail or self._tail_thread
        self.sleep = sleep
        self.clock = clock
        self.jobs: list[_Job] = []

    # -- TrainerIslands --------------------------------------------------------
    def launch(self, incarnation: int, resume_from: CutRef | None) -> _Job:
        # Phase ``submitted`` of this incarnation is the spawn, not the run's launch:
        # the held-back task was built when the fleet started (T_s, start_phases).
        envs = {**self.cfg.envs, **island_env(incarnation, resume_from, poll_s=self.poll_s),
                SUBMITTED_ENV: f"{self.clock():.3f}"}
        call_id = self.modal_ops.spawn(replace(self.cfg, envs=envs))
        self._save_call(incarnation, call_id)
        job = _Job(int(incarnation), str(call_id),
                   IslandEvents(incarnation_tape(self.events_dir, self.name, incarnation)))
        self.jobs.append(job)
        print(f"[launcher] trainer incarnation {incarnation} on Modal as {self.name} "
              f"(call {call_id}, resume {None if resume_from is None else resume_from.cut_id})",
              flush=True)
        self._tail(job.call_id, f"{self.name}-i{incarnation}", job.events)
        return job

    def events(self, job: _Job) -> list[Mapping[str, Any]]:
        return job.events.take()

    def alive(self, job: _Job) -> bool:
        return self.modal_ops.status(job.call_id) == "RUNNING"

    def release(self, job: _Job) -> bool:
        return self._release_call(job.call_id)

    def release_incarnation(self, incarnation: int) -> bool:
        call_id = self._calls().get(str(int(incarnation)))
        if call_id is None:
            print(f"[launcher] trainer incarnation {incarnation}: no Modal call recorded; "
                  "nothing to release", flush=True)
            return False
        return self._release_call(call_id)

    def start_events(self) -> list[Mapping[str, Any]]:
        """Every event of every incarnation (``trainer_start_summary`` gives T_s)."""
        return [e for job in self.jobs for e in job.events.all]

    # -- internals -------------------------------------------------------------
    def _release_call(self, call_id: str) -> bool:
        """Cancel the call (containers terminated) and check it is no longer
        running: the cards are back when the call ended."""
        try:
            self.modal_ops.cancel(call_id)
        except Exception as error:  # noqa: BLE001 - an ended call may refuse; the status decides
            print(f"[launcher] cancel of trainer call {call_id} failed: {error}", flush=True)
        for attempt in range(RELEASE_CHECKS):
            if self.modal_ops.status(call_id) != "RUNNING":
                return True
            if attempt + 1 < RELEASE_CHECKS:
                self.sleep(RELEASE_CHECK_DELAY_S)
        return False

    def _calls(self) -> dict[str, str]:
        return read_jobs(self.jobs_path)

    def _save_call(self, incarnation: int, call_id: str) -> None:
        save_job(self.jobs_path, incarnation, call_id)

    def _tail_thread(self, call_id: str, label: str, collector: IslandEvents) -> None:
        from .modal_islands import _tail_modal

        threading.Thread(target=_tail_modal, args=(self.modal_ops, call_id, label, collector),
                         name=f"tail-{label}", daemon=True).start()


class SkyIslandOps:
    """The sky calls of :class:`SkyTrainerIslands` (tests inject a fake with the
    same methods). ``submit`` returns a request id at once; ``wait`` blocks until
    the cluster is up and the job submitted and returns the job id."""

    def __init__(self, args: Any, spec: Any) -> None:
        self.args = args
        self.spec = spec

    def submit(self, task: Any, cluster: str) -> Any:
        import sky

        from .sky_ops import submit_island_launch

        return submit_island_launch(sky, self.args, self.spec, task, cluster)

    def wait(self, request: Any, task: Any, cluster: str) -> int:
        import sky

        from .sky_ops import maybe_install_nebius_reclaim_hook

        launched = sky.stream_and_get(request)
        maybe_install_nebius_reclaim_hook(task, launched, cluster)
        return int(launched[0])

    def cancel(self, request: Any) -> None:
        import sky

        sky.get(sky.api_cancel(request))

    def job_alive(self, cluster: str, job_id: int) -> bool:
        from .sky_ops import SkySDKOps

        return SkySDKOps().job_alive(cluster, job_id)

    def down(self, cluster: str) -> bool:
        """Tear the cluster down (diagnostics first) and confirm it at the cloud."""
        import sky

        from .teardown import teardown_island

        return teardown_island(sky, cluster, num_nodes=int(getattr(self.spec, "num_nodes", 1) or 1))

    def tail(self, cluster: str, job_id: int, label: str, collector: IslandEvents) -> None:
        from .sky_ops import _tail

        threading.Thread(target=_tail, args=(cluster, job_id, label, collector),
                         name=f"tail-{label}", daemon=True).start()


@dataclass
class _SkyJob:
    incarnation: int
    cluster: str
    events: IslandEvents
    request: Any = None
    job_id: int | None = None
    error: BaseException | None = None
    provisioning: threading.Thread | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class SkyTrainerIslands:
    """:class:`~yeto.rl.engine.trainer_remote.TrainerIslands` on a sky cloud (module doc).

    Each incarnation is a new cluster ``<island>-i<n>`` launched from the held-back
    task with the incarnation environment. ``launch`` only submits the request:
    provisioning runs on a daemon thread (the island counts as alive meanwhile),
    which starts the log tail once the job is submitted. ``release`` cancels a
    launch still provisioning and tears the cluster down."""

    def __init__(self, task: Any, ops: SkyIslandOps, *, name: str, jobs_path: Path,
                 poll_s: float | None = None, events_dir: Path | None = None,
                 clock: Callable[[], float] = time.time,
                 cancel_wait_s: float = PROVISION_CANCEL_WAIT_S) -> None:
        self.task = task
        self.ops = ops
        self.name = name
        self.jobs_path = Path(jobs_path)
        self.poll_s = poll_s
        self.events_dir = events_dir
        self.clock = clock
        self.cancel_wait_s = float(cancel_wait_s)
        self.jobs: list[_SkyJob] = []

    def cluster_name(self, incarnation: int) -> str:
        from .sky_ops import sky_cluster_name

        return sky_cluster_name(f"{self.name}-i{int(incarnation)}")

    # -- TrainerIslands --------------------------------------------------------
    def launch(self, incarnation: int, resume_from: CutRef | None) -> _SkyJob:
        cluster = self.cluster_name(incarnation)
        # sky snapshots the task at submit: the shared task carries this
        # incarnation's environment only for the submit call below.
        self.task.update_envs({**island_env(incarnation, resume_from, poll_s=self.poll_s),
                               SUBMITTED_ENV: f"{self.clock():.3f}"})
        save_job(self.jobs_path, incarnation, cluster)  # before any spend: a dead head finds it
        job = _SkyJob(int(incarnation), cluster,
                      IslandEvents(incarnation_tape(self.events_dir, self.name, incarnation)))
        job.request = self.ops.submit(self.task, cluster)
        self.jobs.append(job)
        print(f"[launcher] trainer incarnation {incarnation} on sky as {cluster} "
              f"(resume {None if resume_from is None else resume_from.cut_id})", flush=True)
        job.provisioning = threading.Thread(target=self._provision, args=(job,),
                                            name=f"provision-{cluster}", daemon=True)
        job.provisioning.start()
        return job

    def events(self, job: _SkyJob) -> list[Mapping[str, Any]]:
        return job.events.take()

    def alive(self, job: _SkyJob) -> bool:
        with job.lock:
            if job.error is not None:
                return False
            if job.job_id is None:
                return True  # still provisioning
            job_id = job.job_id
        return self.ops.job_alive(job.cluster, job_id)

    def release(self, job: _SkyJob) -> bool:
        provisioning = job.provisioning
        if provisioning is not None and provisioning.is_alive():
            try:
                self.ops.cancel(job.request)
            except Exception as error:  # noqa: BLE001 - the teardown below decides
                print(f"[launcher] cancel of the launch of {job.cluster} failed: {error}",
                      flush=True)
            provisioning.join(self.cancel_wait_s)
        return self._down(job.cluster)

    def release_incarnation(self, incarnation: int) -> bool:
        cluster = read_jobs(self.jobs_path).get(str(int(incarnation)))
        if cluster is None:
            print(f"[launcher] trainer incarnation {incarnation}: no sky cluster recorded; "
                  "nothing to release", flush=True)
            return False
        return self._down(cluster)

    def start_events(self) -> list[Mapping[str, Any]]:
        """Every event of every incarnation (``trainer_start_summary`` gives T_s)."""
        return [e for job in self.jobs for e in job.events.all]

    # -- internals -------------------------------------------------------------
    def _provision(self, job: _SkyJob) -> None:
        try:
            job_id = self.ops.wait(job.request, self.task, job.cluster)
        except BaseException as error:  # noqa: BLE001 - reported as the island ending
            print(f"[launcher] launch of trainer cluster {job.cluster} failed: {error}", flush=True)
            with job.lock:
                job.error = error
            return
        with job.lock:
            job.job_id = job_id
        self.ops.tail(job.cluster, job_id, f"{self.name}-i{job.incarnation}", job.events)

    def _down(self, cluster: str) -> bool:
        try:
            return bool(self.ops.down(cluster))
        except Exception as error:  # noqa: BLE001 - not confirmed: the caller reports it
            print(f"[launcher] teardown of trainer cluster {cluster} failed: {error}", flush=True)
            return False


# ---------------------------------------------------------------------------
# Binding
# ---------------------------------------------------------------------------
def modal_binding(args: Any, island: TrainerIsland, *, journal_dir: Path,
                  files: StoreFiles | None = None, islands: Any = None) -> CloudTrainerBinding:
    """The :class:`CloudTrainerBinding` of a Modal trainer
    island (``files`` / ``islands`` replace the SDK in tests)."""
    from .storage import MODAL_CHECKPOINT_STORE_MOUNT, modal_checkpoint_store

    if files is None:
        volume, mount = modal_checkpoint_store(args)
        files = ModalVolumeFiles(volume, mount[len(MODAL_CHECKPOINT_STORE_MOUNT):])
    if islands is None:
        islands = ModalTrainerIslands(island.modal_ops, island.modal_cfg, name=island.name,
                                      jobs_path=Path(journal_dir) / JOBS_FILE,
                                      poll_s=getattr(args, "rl_trainer_command_poll_seconds", None),
                                      events_dir=island.events_dir)
    return _binding(args, files, islands, journal_dir=journal_dir)


def sky_binding(args: Any, island: TrainerIsland, *, journal_dir: Path,
                files: ObjectStore | None = None, islands: Any = None) -> CloudTrainerBinding:
    """The :class:`CloudTrainerBinding` of a sky trainer island with an ``s3://``
    or ``oss://`` checkpoint store (``files`` / ``islands`` replace the SDKs in
    tests)."""
    if files is None:
        files = head_store_objects(str(args.rl_checkpoint_store))
    if islands is None:
        if island.sky_task is None:
            raise ValueError(f"trainer island {island.name}: no sky task to launch")
        islands = SkyTrainerIslands(island.sky_task, SkyIslandOps(args, island.spec),
                                    name=island.name, jobs_path=Path(journal_dir) / JOBS_FILE,
                                    poll_s=getattr(args, "rl_trainer_command_poll_seconds", None),
                                    events_dir=island.events_dir)
    return object_store_binding(args, files, islands, journal_dir=journal_dir)


def object_store_binding(args: Any, files: ObjectStore, islands: Any, *,
                         journal_dir: Path) -> CloudTrainerBinding:
    """The binding over a store the head reads by object key (sky, SSH islands).
    A sharded buffer (OSS) also counts the shard records the trainer has not
    merged yet: a stopped trainer merges nothing (N3.6)."""
    buffer = FileSampleBuffer(files, batch_size=int(args.rollout_batch_size))  # by object key
    unmerged = UnmergedShards(files) if needs_shards(files) else None
    return _binding(args, files, islands, journal_dir=journal_dir, buffer=buffer, unmerged=unmerged)


def head_store_objects(uri: str) -> ObjectStore:
    """The head's object API of an ``s3://`` or ``oss://`` checkpoint store."""
    from yeto.cloud import oss_store

    if oss_store.is_oss_uri(uri):
        return oss_store.head_objects(uri)
    from yeto.cloud.s3_store import head_objects

    return head_objects(uri)


def _binding(args: Any, files: StoreFiles, islands: Any, *, journal_dir: Path,
             buffer: FileSampleBuffer | None = None,
             unmerged: UnmergedShards | None = None) -> CloudTrainerBinding:
    """``buffer``: the head's view of the sample buffer; default: the index
    mirrored from ``files`` (a store the head reads by whole file only)."""
    cuts = StoreCutLog(files)
    if buffer is None:
        mirror = MirrorStore(Path(journal_dir) / MIRROR_DIR, files, [f"{BUFFER_DIR}/{INDEX}"])
        buffer = FileSampleBuffer(mirror, batch_size=int(args.rollout_batch_size))
    poll_s = float(getattr(args, "rl_trainer_command_poll_seconds", None) or 5.0)
    lifecycles: list[RemoteTrainerLifecycle] = []

    def lifecycle(first: int) -> RemoteTrainerLifecycle:
        lifecycles.append(RemoteTrainerLifecycle(islands, files, first_incarnation=first, poll_s=poll_s))
        return lifecycles[-1]

    def watermark() -> int:
        return consumed_watermark(cuts, lifecycles[-1] if lifecycles else None)

    return CloudTrainerBinding(
        lifecycle=lifecycle, buffer=CommittedBacklog(buffer, watermark, unmerged=unmerged), cuts=cuts,
        release_orphan=islands.release_incarnation, start_events=islands.start_events,
        needs_first_policy=first_policy_probe(files) if samplers_wait_for_a_policy(args) else None)


def samplers_wait_for_a_policy(args: Any) -> bool:
    """A LoRA sampler serves no adapter before the first policy release
    (``sampler_island``: ``require_release``); a full-parameter one serves the base."""
    return getattr(args, "tuning", "lora") == "lora"


def first_policy_probe(files: StoreFiles) -> Callable[[], bool]:
    """True while the store has no policy release (``policy-releases/LATEST.json``);
    once one is there it stays False without reading the store again."""
    seen = False

    def needs_first_policy() -> bool:
        nonlocal seen
        if not seen:
            seen = files.read(f"{RELEASE_DIR}/{RELEASE_LATEST}") is not None
        return not seen

    return needs_first_policy
