"""rl-trainer-start-stop L5.3c on sky (CPU, fakes): the trainer island as one sky
cluster per incarnation, the head on an ``s3://`` checkpoint store.

The sky calls go through a fake :class:`SkyIslandOps`; its "cluster" is the fake
island thread of ``test_rl_trainer_remote`` (a real :class:`IslandTrainerAgent` over
a fresh fake driver). No cloud resource is started."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from test_rl_trainer_remote import (
    BATCH,
    T_S,
    _args,
    _Clock,
    _FakeIslands,
    _loop,
    _ticks,
)

from yeto.launch import trainer_loop as launch_loop
from yeto.launch.trainer_island import (
    JOBS_FILE,
    IslandEvents,
    SkyTrainerIslands,
    TrainerIsland,
    sky_binding,
)
from yeto.rl import event_echo
from yeto.rl.elastic_client import LeaveReason
from yeto.rl.engine import recovery
from yeto.rl.engine.fake_lifecycle import FakeSampler
from yeto.rl.engine.journal import Journal
from yeto.rl.engine.object_store import DirectoryObjectStore
from yeto.rl.engine.resume import FLAT_MANIFEST, CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer
from yeto.rl.engine.start_phases import SUBMITTED_ENV
from yeto.rl.engine.trainer_lifecycle import Hold, LifecycleState, Start, Stop
from yeto.rl.engine.trainer_loop import LOOP_KIND
from yeto.rl.engine.trainer_remote import (
    INCARNATION_ENV,
    RESUME_ENV,
    LocalStoreFiles,
    RemoteTrainerLifecycle,
    cut_ref_from_json,
    island_env,
)


class _Task:
    """The held-back sky task: only its environment matters here."""

    def __init__(self) -> None:
        self.envs = {"LEARNER_ID": "0"}

    def update_envs(self, envs) -> None:
        self.envs.update(envs)


class _SkyOps:
    """Fake sky: ``wait`` provisions the cluster (starts the fake island from the
    environment the task carried at submit); ``tail`` hands its events over."""

    def __init__(self, fake: _FakeIslands | None = None) -> None:
        self.fake = fake
        self.submitted: list[tuple[str, dict]] = []
        self.cancelled: list[str] = []
        self.downed: list[str] = []
        self.jobs: dict[str, dict] = {}
        self.provision_gate: threading.Event | None = None
        self.provision_error: BaseException | None = None

    def submit(self, task, cluster):
        self.submitted.append((cluster, dict(task.envs)))  # sky snapshots the task here
        return f"req-{cluster}"

    def wait(self, request, task, cluster):
        if self.provision_gate is not None:
            self.provision_gate.wait(10)
        if self.provision_error is not None or request in self.cancelled:
            raise self.provision_error or RuntimeError("request cancelled")
        envs = dict(next(e for c, e in self.submitted if c == cluster))
        if self.fake is not None:
            self.jobs[cluster] = self.fake.launch(int(envs[INCARNATION_ENV]),
                                                  cut_ref_from_json(envs[RESUME_ENV]))
        return 1

    def cancel(self, request):
        self.cancelled.append(request)

    def job_alive(self, cluster, job_id):
        return self.fake.alive(self.jobs[cluster])

    def down(self, cluster):
        self.downed.append(cluster)
        job = self.jobs.get(cluster)
        return True if job is None else self.fake.release(job)

    def tail(self, cluster, job_id, label, collector):
        job = self.jobs[cluster]
        early, job["events"] = job["events"], collector  # the log replays from the start
        for record in early.all:
            collector.feed(event_echo.format_record(record))


class _ObjectStoreMount(CheckpointStore):
    """An island's ``S3Store``: the mount plus an object API over the same keys."""

    def __init__(self, root):
        super().__init__(root)
        self.objects = DirectoryObjectStore(root)


def _setup(tmp_path, monkeypatch):
    store_root = tmp_path / "store"
    store_root.mkdir()
    (store_root / FLAT_MANIFEST).write_text("{}")
    # the s3 layout: islands and head use the buffer by object key (a directory here)
    fake = _FakeIslands(store_root, FileSampleBuffer(_ObjectStoreMount(store_root), batch_size=BATCH),
                        store_factory=_ObjectStoreMount)
    ops = _SkyOps(fake)
    args = _args(tmp_path, monkeypatch, ["--rl-trainer-island", "1"])
    args.rl_checkpoint_store = "s3://bucket/runs/a"
    journal_dir = launch_loop.journal_dir(args)
    islands = SkyTrainerIslands(_Task(), ops, name="run-a-l1-eu-north1", jobs_path=journal_dir / JOBS_FILE,
                                events_dir=tmp_path / "events", clock=lambda: 1700000000.5)
    binding = sky_binding(args, TrainerIsland("run-a-l1-eu-north1", "nebius"), journal_dir=journal_dir,
                          files=DirectoryObjectStore(store_root), islands=islands)
    lifecycle = binding.lifecycle(0)
    lifecycle.poll_s, lifecycle.exit_grace_s = 0.005, 0.2
    return fake, ops, islands, binding, lifecycle


def test_sky_trainer_starts_and_stops_per_cluster_and_continues_from_the_cut(tmp_path, monkeypatch):
    fake, ops, islands, binding, lifecycle = _setup(tmp_path, monkeypatch)
    clock = _Clock()
    journal = Journal(tmp_path / "head" / "journal")
    loop = _loop(islands, lifecycle, binding.cuts, binding.buffer, clock, journal)
    sampler = FakeSampler(fake.sampler_buffer, fake.published)

    # incarnation 0 on cluster -i0: train the backlog, idle -> SCHEDULED stop
    sampler.produce(2 * BATCH)
    assert _ticks(loop, 3) == [Start, Hold, Hold]
    assert loop.state is LifecycleState.TRAINING and loop.current_version() == 2
    assert _ticks(loop, 1) == [Hold]  # idle from here
    clock.now += T_S
    assert _ticks(loop, 1) == [Stop]
    cut0 = binding.cuts.latest()
    assert (cut0.trigger, cut0.policy_version, cut0.watermark) == (recovery.SCHEDULED, 2, 2 * BATCH)
    assert ops.downed == ["run-a-l1-eu-north1-i0"] and loop.handle.released is True

    # incarnation 1 on a new cluster -i1 resumes the cut: versions continue, no group twice
    sampler.load()
    sampler.produce(BATCH + 2)
    fake.sampler_buffer.mark_data_end()
    assert _ticks(loop, 3) == [Start, Hold, Stop]
    assert loop.finished and loop.handle.leave_reason is LeaveReason.FINISHED
    assert loop.handle.start_version == 2 and loop.current_version() == 3
    assert fake.launched == [(0, None), (1, cut0.cut_id)]
    assert ops.downed == ["run-a-l1-eu-north1-i0", "run-a-l1-eu-north1-i1"]
    trained = [sid for r in fake.rollouts for batch in r.trained.values() for sid, _ in batch]
    assert len(trained) == 3 * BATCH and len(set(trained)) == len(trained)

    # each submit carried that incarnation's environment and its own submit time
    assert [c for c, _ in ops.submitted] == ["run-a-l1-eu-north1-i0", "run-a-l1-eu-north1-i1"]
    assert ops.submitted[1][1] == {"LEARNER_ID": "0", **island_env(1, cut0),
                                   SUBMITTED_ENV: "1700000000.500"}
    # the clusters are recorded for a restarted head, the events are on per-incarnation tapes
    jobs = json.loads(islands.jobs_path.read_text())
    assert jobs == {"0": "run-a-l1-eu-north1-i0", "1": "run-a-l1-eu-north1-i1"}
    for n in (0, 1):
        tape = [json.loads(line) for line in
                (tmp_path / "events" / f"run-a-l1-eu-north1-i{n}.jsonl").read_text().splitlines()]
        assert [r["phase"] for r in tape][-1] == "stopped" and {r["incarnation"] for r in tape} == {n}
    stops = [r for r in journal.records if r["kind"] == LOOP_KIND and r["event"] == "stopped"]
    assert [(r["reason"], r["trigger"], r["released"]) for r in stops] == [
        ("STOPPED", recovery.SCHEDULED, True), ("FINISHED", recovery.MANUAL, True)]
    journal.close()


def test_provisioning_counts_as_alive_and_a_release_cancels_it(tmp_path):
    ops = _SkyOps()
    ops.provision_gate = threading.Event()
    islands = SkyTrainerIslands(_Task(), ops, name="r-l1", jobs_path=tmp_path / "jobs.json",
                                cancel_wait_s=5.0)
    job = islands.launch(0, None)
    assert islands.alive(job) and islands.events(job) == []
    released = []
    t = threading.Thread(target=lambda: released.append(islands.release(job)))
    t.start()
    while not ops.cancelled:
        threading.Event().wait(0.001)
    ops.provision_gate.set()
    t.join(10)
    assert released == [True] and ops.cancelled == ["req-r-l1-i0"] and ops.downed == ["r-l1-i0"]
    assert not islands.alive(job) and isinstance(job.error, RuntimeError)


def test_failed_provisioning_ends_the_island_and_an_orphan_is_released(tmp_path):
    ops = _SkyOps()
    ops.provision_error = RuntimeError("no capacity")
    islands = SkyTrainerIslands(_Task(), ops, name="r-l1", jobs_path=tmp_path / "jobs.json")
    job = islands.launch(4, None)
    job.provisioning.join(10)
    assert not islands.alive(job) and job.job_id is None

    head2 = SkyTrainerIslands(_Task(), ops, name="r-l1", jobs_path=tmp_path / "jobs.json")
    assert head2.release_incarnation(4) is True and ops.downed == ["r-l1-i4"]
    assert head2.release_incarnation(9) is False


def test_lifecycle_fails_an_incarnation_whose_cluster_never_came_up(tmp_path):
    ops = _SkyOps()
    ops.provision_error = RuntimeError("no capacity")
    islands = SkyTrainerIslands(_Task(), ops, name="r-l1", jobs_path=tmp_path / "jobs.json")
    lifecycle = RemoteTrainerLifecycle(islands, LocalStoreFiles(CheckpointStore(tmp_path)),
                                       poll_s=0.005, exit_grace_s=0.05)
    handle = lifecycle.start(lifecycle.plan_start({}), resume_from=None)
    assert lifecycle.ready(handle, float("inf")) is False
    assert handle.state is LifecycleState.STOPPED and handle.leave_reason is LeaveReason.INFRA_FAILED
    assert ops.downed == ["r-l1-i0"]


def test_island_events_skip_a_replayed_record_and_write_the_tape(tmp_path):
    events = IslandEvents(tmp_path / "t.jsonl")
    line = event_echo.format_record({"island_id": 1, "time_unix": 1.0, "event": "rl_trainer_lifecycle",
                                     "incarnation": 0})
    events.feed(line)
    events.feed("[x] " + line)  # a reconnected tail replays the log
    assert len(events.take()) == 1 and len(events.all) == 1
    assert len((tmp_path / "t.jsonl").read_text().splitlines()) == 1


def test_check_binds_a_sky_trainer_island_to_an_s3_store(tmp_path, monkeypatch):
    args = _args(tmp_path, monkeypatch, ["--rl-trainer-island", "1"])
    with pytest.raises(SystemExit, match=r"trainer island 1 \(nebius\) does not reach"):
        launch_loop.check(args)  # the run's store is a Modal Volume
    args.rl_checkpoint_store = "s3://bucket/runs/a"
    launch_loop.check(args)
    args.gpu = "modal:1xh100,verda:1xh100"
    with pytest.raises(SystemExit, match="Verda"):
        launch_loop.check(args)
    args = _args(tmp_path, monkeypatch, ["--rl-trainer-island", "1", "--rl-trainer-ready-seconds", "0"])
    with pytest.raises(SystemExit, match="ready-seconds"):
        launch_loop.check(args)


def test_default_binding_routes_by_cloud(tmp_path, monkeypatch):
    import yeto.launch.trainer_island as ti

    seen = []
    monkeypatch.setattr(ti, "modal_binding", lambda args, island, **kw: seen.append("modal"))
    monkeypatch.setattr(ti, "sky_binding", lambda args, island, **kw: seen.append("sky"))
    args = _args(tmp_path, monkeypatch, ["--rl-trainer-island", "1"])
    launch_loop.default_binding(args, TrainerIsland("a", "modal"))
    launch_loop.default_binding(args, TrainerIsland("b", "nebius"))
    assert seen == ["modal", "sky"]
    with pytest.raises(SystemExit, match="no trainer island"):
        launch_loop.default_binding(args, None)


def test_sky_binding_needs_the_held_task(tmp_path, monkeypatch):
    args = _args(tmp_path, monkeypatch, ["--rl-trainer-island", "1"])
    args.rl_checkpoint_store = "s3://bucket/runs/a"
    with pytest.raises(ValueError, match="no sky task"):
        sky_binding(args, TrainerIsland("b", "nebius"), journal_dir=Path(tmp_path),
                    files=DirectoryObjectStore(tmp_path))


def test_loop_starts_a_trainer_for_the_samplers_first_policy(tmp_path, monkeypatch):
    """LoRA samplers wait for a first release: the buffer stays empty, S1 alone would
    never start. The loop starts one and keeps it until it trained a round."""
    from yeto.rl.engine.trainer_loop import FIRST_POLICY

    fake, ops, islands, binding, lifecycle = _setup(tmp_path, monkeypatch)
    clock = _Clock()
    journal = Journal(tmp_path / "head" / "journal")
    loop = _loop(islands, lifecycle, binding.cuts, binding.buffer, clock, journal)
    released = {"v": False}
    loop._needs_first_policy = lambda: not released["v"]

    assert _ticks(loop, 1) == [Start] and loop.state is LifecycleState.TRAINING  # S1 alone: Hold
    released["v"] = True  # the warm-up publication reached the store
    assert loop.handle.incarnation == 0 and loop.handle.rounds == 0
    clock.now += 10 * T_S
    assert _ticks(loop, 3) == [Hold, Hold, Hold] and loop.state is LifecycleState.TRAINING  # kept
    sampler = FakeSampler(fake.sampler_buffer, fake.published)
    sampler.produce(BATCH)
    assert _ticks(loop, 1) == [Hold] and loop.handle.rounds == 1
    assert _ticks(loop, 1) == [Hold]  # idle from here
    clock.now += T_S
    assert _ticks(loop, 1) == [Stop]
    assert loop.state is LifecycleState.STOPPED and loop.handle.leave_reason is LeaveReason.STOPPED
    assert [r["incarnation"] for r in journal.records
            if r["kind"] == LOOP_KIND and r["event"] == FIRST_POLICY] == [0]
    journal.close()


def test_first_policy_probe_reads_the_release_store_until_one_is_there():
    from yeto.launch.trainer_island import first_policy_probe

    reads = []
    store = {}

    class _Files:
        def read(self, rel):
            reads.append(rel)
            return store.get(rel)

    probe = first_policy_probe(_Files())
    assert probe() is True
    store["policy-releases/LATEST.json"] = b"{}"
    assert probe() is False and probe() is False
    assert reads == ["policy-releases/LATEST.json"] * 2
