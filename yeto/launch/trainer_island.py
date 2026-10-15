"""Cloud binding of the trainer scheduler: the trainer island on Modal
(rl-trainer-start-stop L5.3c).

With ``--rl-trainer-scheduler s1`` the launcher holds back the island named by
``--rl-trainer-island``: its Modal function is defined on the run's app but the
fleet controller does not start it. :func:`modal_binding` gives the scheduler loop:

* :class:`ModalTrainerIslands` (``trainer_remote.TrainerIslands``): each
  incarnation spawns the held-back function with the incarnation environment
  (``trainer_remote.island_env``), tails its log stream for the island events, and
  cancels the call at release. The call ids are kept in the head's run directory,
  so a restarted head can release an orphan;
* :class:`ModalVolumeFiles`: the head's reads and writes in the run's
  ``modal-volume://`` checkpoint store (commands, cut pointers, the buffer index);
* the head's buffer view: the sample buffer index mirrored from the Volume,
  backlog counted from the consumed watermark the trainer committed: the running
  incarnation's reported one, else the newest cut's (``CommittedBacklog``).

Sky trainer islands are not bound yet (``check`` refuses them).
"""

from __future__ import annotations

import io
import json
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from yeto.rl import event_echo
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import BUFFER_DIR, INDEX, CommittedBacklog, FileSampleBuffer
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


@dataclass
class TrainerIsland:
    """The held-back trainer island the launcher hands to the binding."""

    name: str
    cloud: str
    modal_cfg: Any = None  # modal_runner.ModalIslandConfig
    modal_ops: Any = None  # modal_runner.ModalOps (app deployed)


@dataclass
class CloudTrainerBinding:
    """What the loop needs from the cloud layer for one run.

    ``lifecycle(first_incarnation)`` builds the trainer lifecycle (e.g.
    ``DriverLifecycle`` whose ``build_driver`` acquires a trainer island and whose
    ``release`` gives it back). ``release_orphan(incarnation)`` gives back the island
    a dead head started. ``data_ended`` tells the samplers finished.
    ``after_restore()`` runs after the journal is read back and an orphan released
    (e.g. reset the head's buffer to the watermark of the newest cut)."""

    lifecycle: Callable[[int], TrainerLifecycle]
    buffer: DurableSampleBuffer
    cuts: CutLog
    release_orphan: Callable[[int], None] | None = None
    data_ended: Callable[[], bool] = lambda: False
    start_events: Callable[[], Iterable[Mapping[str, Any]]] | None = None
    spec: Mapping[str, Any] | None = None
    after_restore: Callable[[], None] | None = None


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
    keeps the ``rl_trainer_lifecycle`` and start-phase events."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._new: list[dict[str, Any]] = []
        self.all: list[dict[str, Any]] = []

    def feed(self, chunk: Any) -> None:
        for line in str(chunk).splitlines():
            raw = event_echo.parse_line(line)
            if raw is None or raw is event_echo.INVALID:
                continue
            record = json.loads(raw)
            with self._lock:
                self.all.append(record)
                if record.get("event") == EVENT:
                    self._new.append(record)

    def take(self) -> list[dict[str, Any]]:
        with self._lock:
            out, self._new = self._new, []
            return out


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
                 poll_s: float | None = None,
                 tail: Callable[[str, str, IslandEvents], None] | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.time) -> None:
        self.modal_ops = modal_ops
        self.cfg = cfg
        self.name = name
        self.jobs_path = Path(jobs_path)
        self.poll_s = poll_s
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
        job = _Job(int(incarnation), str(call_id), IslandEvents())
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
        if not self.jobs_path.is_file():
            return {}
        return dict(json.loads(self.jobs_path.read_text(encoding="utf-8")))

    def _save_call(self, incarnation: int, call_id: str) -> None:
        calls = {**self._calls(), str(int(incarnation)): str(call_id)}
        self.jobs_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.jobs_path.with_name(self.jobs_path.name + ".tmp")
        tmp.write_text(json.dumps(calls, sort_keys=True), encoding="utf-8")
        tmp.replace(self.jobs_path)

    def _tail_thread(self, call_id: str, label: str, collector: IslandEvents) -> None:
        from .modal_islands import _tail_modal

        threading.Thread(target=_tail_modal, args=(self.modal_ops, call_id, label, collector),
                         name=f"tail-{label}", daemon=True).start()


# ---------------------------------------------------------------------------
# Binding
# ---------------------------------------------------------------------------
def modal_binding(args: Any, island: TrainerIsland, *, journal_dir: Path,
                  files: StoreFiles | None = None, islands: Any = None) -> Any:
    """The :class:`CloudTrainerBinding` of a Modal trainer
    island (``files`` / ``islands`` replace the SDK in tests)."""
    from .storage import MODAL_CHECKPOINT_STORE_MOUNT, modal_checkpoint_store

    if files is None:
        volume, mount = modal_checkpoint_store(args)
        files = ModalVolumeFiles(volume, mount[len(MODAL_CHECKPOINT_STORE_MOUNT):])
    if islands is None:
        islands = ModalTrainerIslands(island.modal_ops, island.modal_cfg, name=island.name,
                                      jobs_path=Path(journal_dir) / JOBS_FILE,
                                      poll_s=getattr(args, "rl_trainer_command_poll_seconds", None))
    cuts = StoreCutLog(files)
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
        lifecycle=lifecycle, buffer=CommittedBacklog(buffer, watermark), cuts=cuts,
        release_orphan=islands.release_incarnation, start_events=islands.start_events)
