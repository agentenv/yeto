"""rl-trainer-start-stop L5.3c (CPU, fakes): the trainer on its own island.

The head runs the scheduler loop with :class:`RemoteTrainerLifecycle`; each
incarnation is a fake "island" (a thread running :class:`IslandTrainerAgent` over a
fresh fake driver). Head and island share only the store directory (commands, cut
pointer, sample buffer) and the island's event lines (``YETO_RL_EVENT`` echo
format). No cloud resource is started."""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_rl_engine_driver import NAME, PROFILE, _driver

from yeto import runs
from yeto.launch import trainer_loop as launch_loop
from yeto.launch.trainer_island import (
    IslandEvents,
    MirrorStore,
    ModalTrainerIslands,
    ModalVolumeFiles,
    TrainerIsland,
    modal_binding,
)
from yeto.rl import event_echo
from yeto.rl.pool_client import LeaveReason
from yeto.rl.engine import recovery
from yeto.rl.engine.bridges import LocalOnlySync
from yeto.rl.engine.fake_lifecycle import BufferFedFakeRollout, FakeRoundCuts, FakeSampler, cuttable_fake_engine
from yeto.rl.engine.journal import Journal
from yeto.rl.engine.resume import FLAT_MANIFEST, CheckpointStore, write_snapshot
from yeto.rl.engine.sample_buffer import BUFFER_DIR, INDEX, CommittedBacklog, FileSampleBuffer
from yeto.rl.engine.start_phases import SUBMITTED_ENV
from yeto.rl.engine.trainer_lifecycle import BacklogStats, Hold, LifecycleState, Start, Stop
from yeto.rl.engine.trainer_loop import FINISHED, LOOP_KIND, TrainerLoop
from yeto.rl.engine.trainer_remote import (
    CommandReader,
    CommandWriter,
    IslandTrainerAgent,
    LocalStoreFiles,
    RemoteTrainerLifecycle,
    StoreCutLog,
    command_path,
    consumed_watermark,
    island_env,
)
from yeto.rl.engine.trainer_scheduler import S1Config, S1Scheduler

BATCH = 4  # fake engine: 2 groups x 2 samples
T_S = 5.0
POLL = 0.005


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _StoreRoundCuts(FakeRoundCuts):
    """Fake round cuts with the pointer where the real one lands (flat layout)."""

    @property
    def pointer_path(self) -> Path:
        return self.root / "round-cut.json"


class _FakeIslands:
    """The cloud: ``launch`` starts an island thread; its events travel as echo
    lines. ``die_at`` makes one incarnation's island exit without a stopped event."""

    def __init__(self, store_root: Path, sampler_buffer: FileSampleBuffer, *,
                 store_factory=CheckpointStore) -> None:
        self.store_root = store_root
        self.sampler_buffer = sampler_buffer
        self.store_factory = store_factory  # the island's store (``store_for``)
        self.engine = None
        self.rollouts: list = []
        self.launched: list[tuple[int, str | None]] = []
        self.released: list[int] = []
        self.agents: list = []
        self.die_at: int | None = None

    def published(self):
        return None if self.engine is None or self.engine.published is None else self.engine.published[0]

    def launch(self, incarnation, resume_from):
        env = island_env(incarnation, resume_from)  # what the island would read
        self.launched.append((incarnation, resume_from and resume_from.cut_id))
        store = self.store_factory(self.store_root)
        buffer = FileSampleBuffer(store, batch_size=BATCH)  # the trainer's own reader
        cuts = _StoreRoundCuts(self.store_root, watermark=lambda: buffer.watermark)
        buffer.reset_to(0 if resume_from is None else resume_from.watermark)
        engine = cuttable_fake_engine(tensors={NAME: torch.zeros(1, 2)}, step_delta=1.0,
                                      capability_profile=PROFILE)
        engine.rollout = BufferFedFakeRollout(engine, buffer)
        self.engine = engine
        self.rollouts.append(engine.rollout)
        sync = LocalOnlySync(1000)
        sync.round_cuts = cuts
        driver = _driver(engine, sync, self.store_root.parent, name=f"tape-{incarnation}.jsonl")
        job = {"incarnation": incarnation, "events": IslandEvents(), "alive": True, "env": env}

        def emit(event, **fields):
            job["events"].feed(event_echo.format_record(
                {"island_id": 0, "time_unix": 0.0, "event": event, **fields}))

        files = LocalStoreFiles(store)
        agent = IslandTrainerAgent(driver, commands=CommandReader(files, incarnation),
                                   incarnation=int(env["YETO_TRAINER_INCARNATION"]),
                                   resume_from=resume_from, emit=emit, cuts=StoreCutLog(files),
                                   watermark=lambda version: buffer.watermark, poll_s=POLL)
        self.agents.append(agent)

        def island():
            try:
                if self.die_at == incarnation:
                    return  # crashed before any event
                agent.run()
            except Exception:  # noqa: BLE001 - the stopped event carries it
                pass
            finally:
                job["alive"] = False

        job["thread"] = threading.Thread(target=island, daemon=True)
        job["thread"].start()
        return job

    def events(self, job):
        return job["events"].take()

    def alive(self, job):
        return job["alive"]

    def release(self, job):
        job["thread"].join(timeout=30)
        self.released.append(job["incarnation"])
        return not job["thread"].is_alive()

    def release_incarnation(self, incarnation):
        self.released.append(incarnation)
        return True


def _setup(tmp_path):
    store_root = tmp_path / "store"
    (store_root).mkdir()
    (store_root / FLAT_MANIFEST).write_text("{}")  # flat layout: pointer in the store root
    sampler_buffer = FileSampleBuffer(CheckpointStore(store_root), batch_size=BATCH)
    islands = _FakeIslands(store_root, sampler_buffer)
    files = LocalStoreFiles(CheckpointStore(store_root))
    cuts = StoreCutLog(files)
    lifecycle = RemoteTrainerLifecycle(islands, files, poll_s=POLL, exit_grace_s=0.2)
    mirror = MirrorStore(tmp_path / "head" / "mirror", files, [f"{BUFFER_DIR}/{INDEX}"])
    head_buffer = CommittedBacklog(FileSampleBuffer(mirror, batch_size=BATCH),
                                   lambda: consumed_watermark(cuts, lifecycle))
    return islands, lifecycle, cuts, head_buffer


def _loop(islands, lifecycle, cuts, head_buffer, clock, journal):
    scheduler = S1Scheduler(S1Config(a_hard=0, start_seconds=T_S), record=journal.append, clock=clock)
    return TrainerLoop(scheduler=scheduler, lifecycle=lifecycle, buffer=head_buffer, cuts=cuts,
                       record=journal.append, release_orphan=islands.release_incarnation,
                       ready_timeout_s=30.0)


def _ticks(loop, n):
    return [type(loop.tick()) for _ in range(n)]


def test_remote_start_stop_to_the_data_end(tmp_path):
    clock = _Clock()
    islands, lifecycle, cuts, head_buffer = _setup(tmp_path)
    journal = Journal(tmp_path / "head" / "journal")
    loop = _loop(islands, lifecycle, cuts, head_buffer, clock, journal)
    sampler = FakeSampler(islands.sampler_buffer, islands.published)

    # incarnation 0: two batches -> start, train 2 rounds, idle stop (SCHEDULED)
    assert _ticks(loop, 1) == [Hold]
    sampler.produce(2 * BATCH)
    assert head_buffer.backlog(0).batches == 2
    assert _ticks(loop, 3) == [Start, Hold, Hold]
    assert loop.state is LifecycleState.TRAINING and loop.current_version() == 2
    assert head_buffer.backlog(2).samples == 0  # the position the running trainer reported
    assert _ticks(loop, 1) == [Hold]
    clock.now += T_S
    assert _ticks(loop, 1) == [Stop] and loop.state is LifecycleState.STOPPED
    handle = loop.handle
    assert handle.leave_reason is LeaveReason.STOPPED and handle.released is True
    cut0 = cuts.latest()
    assert (cut0.trigger, cut0.policy_version, cut0.watermark) == (recovery.SCHEDULED, 2, 2 * BATCH)
    assert head_buffer.backlog(2).samples == 0  # the trainer's committed watermark

    # incarnation 1: one batch plus a remainder of 2, then the data ends
    sampler.load()
    sampler.produce(BATCH + 2)
    islands.sampler_buffer.mark_data_end()
    assert _ticks(loop, 3) == [Start, Hold, Stop]  # below Q_on, but the data ended
    assert loop.finished and loop.handle.leave_reason is LeaveReason.FINISHED, (
        loop.handle.leave_reason, loop.handle.error, loop.handle.states)
    assert loop.handle.start_version == 2 and loop.current_version() == 3
    assert _ticks(loop, 1) == [Hold]

    assert islands.launched == [(0, None), (1, cut0.cut_id)]
    assert islands.released == [0, 1]
    trained = [sid for r in islands.rollouts for batch in r.trained.values() for sid, _ in batch]
    assert len(trained) == 3 * BATCH and len(set(trained)) == len(trained)
    assert cuts.latest().trigger == recovery.MANUAL and head_buffer.backlog(3).samples == 2

    stops = [r for r in journal.records if r["kind"] == LOOP_KIND and r["event"] == "stopped"]
    assert [(r["reason"], r["trigger"], r["released"]) for r in stops] == [
        ("STOPPED", recovery.SCHEDULED, True), ("FINISHED", recovery.MANUAL, True)]
    # every command reached the island in order
    sent = json.loads((tmp_path / "store" / command_path(0)).read_text().splitlines()[-1])
    assert sent == {"seq": 3, "verb": "stop", "rounds": None, "reason": "STOPPED"}
    journal.close()


def test_island_that_dies_is_failed_and_released(tmp_path):
    clock = _Clock()
    islands, lifecycle, cuts, head_buffer = _setup(tmp_path)
    journal = Journal(tmp_path / "head" / "journal")
    loop = _loop(islands, lifecycle, cuts, head_buffer, clock, journal)
    FakeSampler(islands.sampler_buffer, islands.published).produce(BATCH)
    islands.die_at = 0
    assert _ticks(loop, 1) == [Start]
    assert loop.state is LifecycleState.STOPPED
    assert loop.handle.leave_reason is LeaveReason.INFRA_FAILED and "without a stopped event" in repr(loop.handle.error)
    assert islands.released == [0]
    islands.die_at = None
    assert _ticks(loop, 2) == [Start, Hold]  # the next incarnation starts again
    assert loop.handle.incarnation == 1
    loop.close()
    assert islands.released == [0, 1]
    journal.close()


def test_data_end_below_one_batch_finishes_without_a_trainer(tmp_path):
    """S1 boundary: less than one batch at the data end -> stop, the loop ends."""
    clock = _Clock()
    islands, lifecycle, cuts, head_buffer = _setup(tmp_path)
    journal = Journal(tmp_path / "head" / "journal")
    loop = _loop(islands, lifecycle, cuts, head_buffer, clock, journal)
    FakeSampler(islands.sampler_buffer, islands.published).produce(BATCH - 1)
    islands.sampler_buffer.mark_data_end()
    assert _ticks(loop, 1) == [Stop] and loop.finished and islands.launched == []
    records = list(journal.records)
    assert [(r["event"], r["remainder"]) for r in records if r["kind"] == LOOP_KIND] == [(FINISHED, BATCH - 1)]
    journal.close()

    # a restarted head sees the run finished
    journal = Journal(tmp_path / "head" / "journal")
    loop = _loop(islands, lifecycle, cuts, head_buffer, clock, journal)
    loop.restore(journal.records)
    assert loop.finished and _ticks(loop, 1) == [Hold]
    journal.close()


@pytest.mark.parametrize("state", [LifecycleState.COLD, LifecycleState.STOPPED, LifecycleState.TRAINING])
def test_s1_stops_finished_below_one_batch_at_the_data_end(state):
    s1 = S1Scheduler(S1Config(a_hard=3, start_seconds=T_S), clock=_Clock())
    short = BacklogStats(batches=0.5, samples=2, data_ended=True)
    assert s1.decide(short, state) == Stop(LeaveReason.FINISHED)
    assert s1.decide(short, LifecycleState.WARMING) == Hold()
    # a full batch at the data end starts a cold trainer below Q_on = 4
    full = BacklogStats(batches=1.5, samples=6, data_ended=True)
    assert s1.decide(full, LifecycleState.COLD) == Start()


def test_s1_counts_a_partial_batch_as_idle():
    clock = _Clock()
    s1 = S1Scheduler(S1Config(a_hard=0, start_seconds=T_S), clock=clock)
    partial = BacklogStats(batches=0.5, samples=2)
    assert s1.decide(partial, LifecycleState.TRAINING) == Hold()
    clock.now += T_S
    assert s1.decide(partial, LifecycleState.TRAINING) == Stop(LeaveReason.STOPPED)


# ---------------------------------------------------------------- store pieces
def test_command_channel_round_trip(tmp_path):
    files = LocalStoreFiles(CheckpointStore(tmp_path))
    writer, reader = CommandWriter(files, 7), CommandReader(files, 7)
    assert reader.poll() == []
    writer.send("run", rounds=2)
    assert [(c.seq, c.verb, c.rounds) for c in reader.poll()] == [(1, "run", 2)]
    writer.send("stop", reason=LeaveReason.STOPPED)
    assert [(c.seq, c.reason) for c in reader.poll()] == [(2, "STOPPED")]
    assert reader.poll() == []
    with pytest.raises(ValueError):
        writer.send("stop")


def test_store_cut_log_reads_the_snapshot_layout(tmp_path):
    store = CheckpointStore(tmp_path / "store")
    state = tmp_path / "state"
    state.mkdir()
    cut = recovery.ConsistentCut(checkpoint_id="r000004-i1", trigger=recovery.SCHEDULED, outer_step=None,
                                 island_steps={"local": 4}, buffer_watermark=12)
    (state / "round-cut.json").write_text(json.dumps(
        {"cut_id": "r000004-i1", "next_rollout_id": 4, recovery.CUT_KEY: cut.to_record()}))
    log = StoreCutLog(LocalStoreFiles(store))
    assert log.latest() is None and log.watermark() == 0
    write_snapshot(store, state, {"cut_id": "r000004-i1"})
    ref = log.latest()
    assert (ref.cut_id, ref.trigger, ref.policy_version, ref.watermark) == ("r000004-i1", recovery.SCHEDULED, 4, 12)
    assert log.watermark() == 12


class _Volume:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    def read_file(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        yield self.files[path]

    def batch_upload(self, force=False):
        volume = self

        class _Batch:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def put_file(self, fh, path):
                volume.files[path] = fh.read()

        return _Batch()


def test_modal_volume_files_use_the_store_prefix():
    volume = _Volume()
    files = ModalVolumeFiles("vol", "/runs/a/", volume_factory=lambda name: volume)
    assert files.read("x/y.json") is None
    files.write("x/y.json", b"1")
    assert volume.files == {"runs/a/x/y.json": b"1"} and files.read("x/y.json") == b"1"


class _ModalOps:
    def __init__(self) -> None:
        self.spawned: list = []
        self.cancelled: list[str] = []
        self.stuck: set[str] = set()

    def spawn(self, cfg):
        self.spawned.append(cfg)
        return f"fc-{len(self.spawned)}"

    def status(self, call_id):
        return "RUNNING" if call_id not in self.cancelled or call_id in self.stuck else "FAILED"

    def cancel(self, call_id):
        self.cancelled.append(call_id)


class _Cfg:
    def __init__(self) -> None:
        self.envs = {"BASE": "1"}


def test_modal_islands_spawn_per_incarnation_and_release_an_orphan(tmp_path, monkeypatch):
    import yeto.launch.trainer_island as ti

    monkeypatch.setattr(ti, "replace", lambda cfg, **kw: type("Cfg", (), kw)())
    ops, tails = _ModalOps(), []
    islands = ModalTrainerIslands(ops, _Cfg(), name="run-a-l1", jobs_path=tmp_path / "jobs.json",
                                  tail=lambda call_id, label, events: tails.append((call_id, label)),
                                  sleep=lambda s: None, clock=lambda: 1700000000.25)
    job = islands.launch(3, None)
    # phase ``submitted`` of the incarnation is its spawn, not the build of the held task
    assert ops.spawned[0].envs == {"BASE": "1", **island_env(3, None),
                                   SUBMITTED_ENV: "1700000000.250"}
    assert tails == [("fc-1", "run-a-l1-i3")]
    job.events.feed("[run-a-l1-i3] " + event_echo.format_record(
        {"island_id": 1, "time_unix": 0.0, "event": "rl_trainer_lifecycle", "incarnation": 3}))
    job.events.feed("noise\n")
    assert [e["incarnation"] for e in islands.events(job)] == [3] and islands.events(job) == []
    assert islands.alive(job)

    # a new head (fresh instance) releases incarnation 3 by its recorded call id
    head2 = ModalTrainerIslands(ops, _Cfg(), name="run-a-l1", jobs_path=tmp_path / "jobs.json",
                                tail=lambda *a: None, sleep=lambda s: None)
    assert head2.release_incarnation(3) is True and ops.cancelled == ["fc-1"]
    assert head2.release_incarnation(9) is False
    job2 = islands.launch(4, None)
    ops.stuck.add(job2.call_id)
    assert islands.release(job2) is False  # the cloud never confirmed


# ---------------------------------------------------------------- launcher switch
def _args(tmp_path, monkeypatch, extra=()):
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    parser = argparse.ArgumentParser()
    launch_loop.add_arguments(parser)
    args = parser.parse_args(["--rl-trainer-scheduler", "s1", *extra])
    args.cluster_prefix, args.rl_max_policy_age, args.rollout_batch_size = "run-a", 0, BATCH
    args.gpu, args.rl_checkpoint_store = "modal:1xh100,nebius:1xh100", "modal-volume://vol/runs/a"
    return args


def test_check_refuses_without_a_bound_trainer_island(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="needs --rl-trainer-island"):
        launch_loop.check(_args(tmp_path, monkeypatch))
    with pytest.raises(SystemExit, match=r"trainer island 1 \(nebius\) does not reach"):
        launch_loop.check(_args(tmp_path, monkeypatch, ["--rl-trainer-island", "1"]))
    args = _args(tmp_path, monkeypatch, ["--rl-trainer-island", "0"])
    launch_loop.check(args)
    assert launch_loop.held_island(args) == 0
    args.rl_checkpoint_store = "s3://bucket"
    with pytest.raises(SystemExit, match="modal-volume"):
        launch_loop.check(args)


def test_modal_binding_reads_the_volume_and_backlog(tmp_path, monkeypatch):
    args = _args(tmp_path, monkeypatch, ["--rl-trainer-island", "0"])
    store_root = tmp_path / "store"
    (store_root / FLAT_MANIFEST).parent.mkdir(parents=True)
    (store_root / FLAT_MANIFEST).write_text("{}")
    FakeSampler(FileSampleBuffer(CheckpointStore(store_root), batch_size=BATCH), lambda: None).produce(6)
    islands = ModalTrainerIslands(_ModalOps(), _Cfg(), name="l0", jobs_path=tmp_path / "j.json",
                                  tail=lambda *a: None)
    binding = modal_binding(args, TrainerIsland("l0", "modal"), journal_dir=launch_loop.journal_dir(args),
                            files=LocalStoreFiles(CheckpointStore(store_root)), islands=islands)
    assert binding.buffer.backlog(0).samples == 6 and binding.cuts.latest() is None
    assert binding.release_orphan == islands.release_incarnation
    assert isinstance(binding.lifecycle(5), RemoteTrainerLifecycle)
    assert binding.lifecycle(5).plan_start({}).incarnation == 5


class _SilentIslands:
    """A trainer island that runs but never answers (its round waits for samples
    that no longer come)."""

    def __init__(self):
        self.released = []

    def launch(self, incarnation, resume_from):
        return {"incarnation": incarnation}

    def events(self, job):
        return []

    def alive(self, job):
        return True

    def release(self, job):
        self.released.append(job["incarnation"])
        return True


def test_interrupt_ends_a_run_wait_and_bounds_the_stop_drain(tmp_path, capsys):
    from yeto.rl.engine.trainer_lifecycle import LifecycleError, StopCondition

    islands = _SilentIslands()
    files = LocalStoreFiles(CheckpointStore(tmp_path))
    lifecycle = RemoteTrainerLifecycle(islands, files, poll_s=POLL, interrupted_drain_s=0.05)
    handle = lifecycle.start(lifecycle.plan_start({}), resume_from=None)
    errors = []

    def run():
        try:
            lifecycle.run(handle, None, StopCondition(1))
        except LifecycleError as error:
            errors.append(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(0.1)
    assert thread.is_alive()  # waits for the round
    lifecycle.interrupt()
    thread.join(5)
    assert not thread.is_alive() and "interrupted" in str(errors[0])
    lifecycle.stop(handle, LeaveReason.STOPPED)  # returns: no cut within the bound
    assert handle.state is LifecycleState.STOPPED and handle.cut is None
    assert islands.released == [0]
    assert "no cut within" in capsys.readouterr().out
    verbs = [c.verb for c in CommandReader(files, 0).poll()]
    assert verbs == ["run", "stop"]


def test_loop_thread_stop_interrupts_the_lifecycle_before_the_join():
    from yeto.rl.engine.trainer_loop import TrainerLoopThread

    calls = []
    release = threading.Event()

    class Loop:
        finished = False
        lifecycle = SimpleNamespace(interrupt=lambda: (calls.append("interrupt"), release.set()))

        def tick(self):
            release.wait(5)  # a tick blocked in a run wait until interrupted

        def close(self, reason):
            calls.append(("close", reason))

    runner = TrainerLoopThread(Loop(), 0.01).start()
    threading.Event().wait(0.05)
    runner.stop(timeout=5)
    assert calls == ["interrupt", ("close", LeaveReason.STOPPED)]
