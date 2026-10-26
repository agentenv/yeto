"""Convertible pool: node selection (GOAL §7.1 item 14), buffer shards, and the
remote conversion channel (rl-trainer-start-stop L7.5) with fake nodes."""

from __future__ import annotations

import json
import threading

import pytest

from yeto.rl.elastic_client import LeaveReason
from yeto.rl.engine.convertible import (
    CONVERSION_DECISION_KIND,
    ConvertibleNode,
    NodeClass,
    NodeProfile,
    NodeSpec,
    TrainerNeeds,
    gpu_train_rank,
    rank_nodes,
    select_group,
)
from yeto.rl.engine.convertible_remote import (
    CONVERTING,
    END,
    EXITED,
    ROLLOUT,
    TO_TRAINER,
    TRAINER,
    ConvertibleGate,
    ConvertibleIslands,
    ConvertibleNodeAgent,
    NodeCommandReader,
    NodeCommandWriter,
    RolloutRound,
    append_trainer_command,
    node_path,
)
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import (
    BUFFER_DIR,
    INDEX,
    FileSampleBuffer,
    merge_shards,
    shard_index,
)
from yeto.rl.engine.trainer_lifecycle import (
    BacklogStats,
    CutRef,
    Hold,
    LifecycleState,
    SampleIndexEntry,
    StartPlan,
    Start,
    TrainerHandle,
)
from yeto.rl.engine.trainer_loop import TrainerLoop
from yeto.rl.engine.trainer_remote import (
    EVENT,
    READY,
    RUN,
    STARTED,
    STOP,
    STOPPED,
    TRAINED,
    CommandReader,
    LocalStoreFiles,
    RemoteTrainerLifecycle,
    lifecycle_event,
)
from yeto.rl.engine.trainer_scheduler import S1Config, S1Scheduler

NEEDS = TrainerNeeds(precision="bf16", memory_gb=40, gpus=1, domains=("modal",))


# ---------------------------------------------------------------- selection levels
def _p(name, **kw):
    return NodeProfile(name, **{"gpu": "H100", "gpu_memory_gb": 80, "domain": "modal",
                                "inflight": 0, "capacity": 1.0, **kw})


def test_level1_hard_conditions_refuse_spot_precision_and_memory():
    ranked, refused = rank_nodes([_p("a", spot=True), _p("b", precision="fp8"),
                                  _p("c", gpu_memory_gb=24), _p("d")], NEEDS)
    assert [r.name for r in ranked] == ["d"]
    assert refused == {"a": "spot", "b": "precision fp8 != trainer bf16",
                       "c": "gpu memory 24 GB < trainer state 40 GB"}


def test_level2_same_domain_beats_faster_gpu_and_fewer_inflight():
    ranked, _ = rank_nodes([_p("far", domain="aws/us-east-1", gpu="B200", inflight=0),
                            _p("near", inflight=50)], NEEDS)
    assert [r.name for r in ranked] == ["near", "far"]
    assert ranked[0].levels["near_trainer"] is True and ranked[1].levels["near_trainer"] is False


def test_level3_layout_fit_then_gpu_type():
    needs = TrainerNeeds(precision="bf16", gpus=2, domains=("modal",))
    ranked, _ = rank_nodes([_p("h100x2", gpus=2), _p("b200x1", gpu="B200", gpus=1),
                            _p("h200x2", gpu="H200!", gpus=2)], needs)
    assert [r.name for r in ranked] == ["h200x2", "h100x2", "b200x1"]
    assert gpu_train_rank("H200!") > gpu_train_rank("H100") > gpu_train_rank("unknown") == 0


def test_level4_fewer_inflight_then_level5_smaller_capacity():
    ranked, _ = rank_nodes([_p("busy", inflight=9, capacity=0.1), _p("quiet-big", capacity=3.0),
                            _p("quiet-small", capacity=0.5)], NEEDS)
    assert [r.name for r in ranked] == ["quiet-small", "quiet-big", "busy"]


def test_unknown_values_sort_after_known_ones():
    ranked, _ = rank_nodes([_p("nodomain", domain=None), _p("unmeasured", inflight=None),
                            _p("known")], NEEDS)
    assert [r.name for r in ranked] == ["known", "unmeasured", "nodomain"]


def test_select_group_never_takes_the_last_rollout_node_and_journals_ranking():
    two = [_p("a", capacity=2.0), _p("b", capacity=1.0)]
    sel = select_group(two, NEEDS, group_size=1, rollout_nodes=["a", "b"])
    assert sel.group == ("b",) and sel.lost_capacity == 1.0
    assert [r["node"] for r in sel.record()["ranking"]] == ["b", "a"]
    held = select_group(two, NEEDS, group_size=2, rollout_nodes=["a", "b"])
    assert held.group == () and held.note == "hold:last_rollout_node"
    short = select_group([_p("a", spot=True)], NEEDS, group_size=1, rollout_nodes=["a", "x"])
    assert short.group == () and short.refused == {"a": "spot"}
    assert short.note.startswith("hold:only 0 eligible")


def test_group_of_n_takes_the_n_best():
    nodes = [_p(f"n{i}", capacity=float(i)) for i in range(4)]
    sel = select_group(nodes, NEEDS, group_size=3, rollout_nodes=[n.name for n in nodes])
    assert sel.group == ("n0", "n1", "n2") and sel.lost_capacity == 3.0


# ---------------------------------------------------------------- buffer shards
def _store(tmp_path):
    return CheckpointStore(str(tmp_path))


def _write(buf, sid, version=0):
    ref = buf.write_payload(sid, b"x")
    buf.append(SampleIndexEntry(sample_id=sid, behavior_version=version, payload_ref=ref, tokens=4))


def test_shards_merge_into_one_index_once_each(tmp_path):
    store = _store(tmp_path)
    clock = iter(range(100))
    a = FileSampleBuffer(store, batch_size=1, index=shard_index("island0"), clock=lambda: next(clock))
    b = FileSampleBuffer(store, batch_size=1, index=shard_index("island1"), clock=lambda: next(clock))
    _write(a, "a0")
    _write(b, "b0")
    _write(a, "a1")
    files = LocalStoreFiles(store)
    assert merge_shards(files, ["island0", "island1"]) == 3
    assert merge_shards(files, ["island0", "island1"]) == 0
    _write(b, "b1", version=2)
    assert merge_shards(files, ["island0", "island1"]) == 1
    reader = FileSampleBuffer(store, batch_size=2)
    assert [e.sample_id for e in reader._entries] == ["a0", "b0", "a1", "b1"]
    assert reader.backlog(2).batches == 2
    lines = [json.loads(x) for x in (tmp_path / BUFFER_DIR / INDEX).read_text().splitlines()]
    assert [(r["shard"], r["shard_pos"], r["pos"]) for r in lines] == [
        ("island0", 0, 0), ("island1", 0, 1), ("island0", 1, 2), ("island1", 1, 3)]


def test_merge_skips_a_torn_shard_line(tmp_path):
    store = _store(tmp_path)
    a = FileSampleBuffer(store, batch_size=1, index=shard_index("n"))
    _write(a, "a0")
    with open(tmp_path / BUFFER_DIR / shard_index("n"), "ab") as fh:
        fh.write(b'{"torn": ')
    assert merge_shards(LocalStoreFiles(store), ["n"]) == 1


# ---------------------------------------------------------------- channels
class MemFiles:
    def __init__(self):
        self.data: dict[str, bytes] = {}
        self.lock = threading.Lock()

    def read(self, rel):
        with self.lock:
            return self.data.get(rel)

    def write(self, rel, data):
        with self.lock:
            self.data[rel] = bytes(data)


def test_node_commands_skip_those_of_an_earlier_island():
    files = MemFiles()
    NodeCommandWriter(files, "n").send(TO_TRAINER, incarnation=0)
    reader = NodeCommandReader(files, "n")
    writer = NodeCommandWriter(files, "n")  # a restarted head continues the file
    writer.send(TO_TRAINER, incarnation=1, resume=CutRef("c1", "scheduled", 3, 12))
    writer.send(END)
    got = reader.poll()
    assert [(c.seq, c.verb, c.incarnation) for c in got] == [(2, TO_TRAINER, 1), (3, END, None)]
    assert reader.poll() == []


def test_append_trainer_command_keeps_earlier_commands():
    files = MemFiles()
    append_trainer_command(files, 4, RUN, rounds=1)
    append_trainer_command(files, 4, STOP, reason="STOPPED")
    assert [(c.seq, c.verb) for c in CommandReader(files, 4).poll()] == [(1, RUN), (2, STOP)]


# ---------------------------------------------------------------- fakes
class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class HandoffPort:
    def __init__(self, log):
        self.log = log

    def stop_requests(self):
        self.log.append("stop_requests")
        return []

    def release_rollout(self):
        self.log.append("release_rollout")

    def onload_trainer(self):
        self.log.append("onload_trainer")

    def offload_trainer(self):
        self.log.append("offload_trainer")

    def resume_rollout(self, version):
        self.log.append(("resume_rollout", version))
        return f"sha-{version}"

    def accept_requests(self):
        self.log.append("accept_requests")


class FakeStint:
    """Island side of one trainer incarnation: follows the trainer commands like
    IslandTrainerAgent and reports the same events."""

    def __init__(self, files, start_version=0, fail=False):
        self.files = files
        self.start_version = start_version
        self.fail = fail
        self.runs: list[tuple[int, CutRef | None]] = []

    def __call__(self, incarnation, resume, emit):
        self.runs.append((incarnation, resume))
        h = TrainerHandle(StartPlan(incarnation), resume_from=resume, clock=lambda: 0.0)
        for s in (LifecycleState.ACQUIRING, LifecycleState.JOINING, LifecycleState.RESTORING,
                  LifecycleState.WARMING, LifecycleState.TRAINING):
            h.advance(s)
        h.start_version = resume.policy_version if resume else self.start_version
        h.published_version = h.start_version
        seq = 0
        emit(EVENT, **lifecycle_event(h, STARTED, seq=seq))
        emit(EVENT, **lifecycle_event(h, READY, seq=seq, watermark=0))
        reader = CommandReader(self.files, incarnation)
        while True:
            for c in reader.poll():
                seq = c.seq
                if c.verb == RUN:
                    if self.fail:
                        h.error = RuntimeError("boom")
                        h.advance(LifecycleState.STOPPED)
                        emit(EVENT, **lifecycle_event(h, STOPPED, seq=seq))
                        return h
                    h.rounds += c.rounds
                    h.published_version = h.trained_version
                    emit(EVENT, **lifecycle_event(h, TRAINED, seq=seq, watermark=h.rounds * 4))
                else:
                    h.advance(LifecycleState.DRAINING)
                    h.leave_reason = LeaveReason[c.reason]
                    h.cut = CutRef(f"cut-{h.trained_version}", "scheduled", h.trained_version,
                                   h.rounds * 4)
                    h.advance(LifecycleState.STOPPED)
                    emit(EVENT, **lifecycle_event(h, STOPPED, seq=seq))
                    return h
            threading.Event().wait(0.005)


def _agent(files, name="n0", log=None, stint=None, max_rounds=None, groups=4, seconds=2.0):
    log = [] if log is None else log
    node = ConvertibleNode(NodeSpec(name, NodeClass.CONVERTIBLE), HandoffPort(log))
    return ConvertibleNodeAgent(
        name=name, files=files, node=node,
        generate=lambda: RolloutRound(groups, seconds), run_stint=stint or FakeStint(files),
        published_checksum=lambda: None, inflight=lambda: 0, max_rounds=max_rounds,
        poll_s=0.001, status_s=0.05, idle_exit_s=3600)


# ---------------------------------------------------------------- island side
def test_node_converts_trains_and_comes_back_with_the_trained_version():
    files, log = MemFiles(), []
    agent = _agent(files, log=log, max_rounds=2)
    NodeCommandWriter(files, "n0")  # nothing yet
    thread = threading.Thread(target=agent.run, daemon=True)
    thread.start()
    NodeCommandWriter(files, "n0").send(TO_TRAINER, incarnation=0)
    append_trainer_command(files, 0, RUN, rounds=1)
    _until(lambda: _phase(files, "n0", TRAINED))
    append_trainer_command(files, 0, STOP, reason="STOPPED")
    _until(lambda: agent.role == ROLLOUT and agent.stints == 1)
    NodeCommandWriter(files, "n0").send(END)
    thread.join(5)
    assert not thread.is_alive() and agent.role == EXITED and agent.served_version == 1
    assert log == ["stop_requests", "release_rollout", "onload_trainer",
                   "offload_trainer", ("resume_rollout", 1), "accept_requests"]
    events = _events(files, "n0")
    conv = [e for e in events if e["event"] == "node_conversion"]
    assert [(e["direction"], e["incarnation"]) for e in conv] == [("to_trainer", 0), ("to_rollout", 0)]
    assert conv[1]["version"] == 1 and conv[1]["weight_checksum"] == "sha-1"
    summary = [e for e in events if e["event"] == "trainer_start_summary"]
    assert len(summary) == 1 and summary[0]["kind"] == "conversion" and summary[0]["total_s"] >= 0
    status = json.loads(files.read(node_path("n0", "status.json")))
    assert status["role"] == EXITED and status["generation_done"] and status["capacity"] == 0.0
    assert agent.capacity == 2.0  # measured; reported as 0 once generation is done


def test_node_stint_failure_fails_the_node():
    files = MemFiles()
    agent = _agent(files, stint=FakeStint(files, fail=True))
    errors = []
    thread = threading.Thread(target=lambda: _catch(agent.run, errors), daemon=True)
    thread.start()
    NodeCommandWriter(files, "n0").send(TO_TRAINER, incarnation=0)
    append_trainer_command(files, 0, RUN, rounds=1)
    thread.join(5)
    assert errors and "failed" in str(errors[0])
    assert json.loads(files.read(node_path("n0", "status.json")))["role"] == "failed"


# ---------------------------------------------------------------- head side
def _status(files, name, **kw):
    files.write(node_path(name, "status.json"), json.dumps(
        {"node": name, "role": ROLLOUT, "at": 1000.0, "inflight": 0, "capacity": 1.0,
         "generation_done": False, **kw}).encode())


def _islands(files, names=("n0", "n1"), records=None, clock=None, **kw):
    rec = (lambda kind, tx_id=None, **f: records.append({"kind": kind, **f})) \
        if records is not None else None
    return ConvertibleIslands(files, [_p(n) for n in names], NEEDS, record=rec,
                              clock=clock or Clock(), sleep=lambda s: None, poll_s=0.0, **kw)


def test_plan_uses_measured_status_and_skips_stale_or_busy_nodes():
    files, clock = MemFiles(), Clock()
    _status(files, "n0", inflight=5)
    _status(files, "n1", inflight=1)
    _status(files, "n2", role=TRAINER)
    isl = _islands(files, ("n0", "n1", "n2"), clock=clock)
    assert isl.plan().group == ("n1",)
    _status(files, "n1", inflight=1, at=1000.0 - 301)  # stale
    sel = isl.plan()
    assert sel.group == () and sel.note == "hold:last_rollout_node"


def test_gate_holds_without_an_eligible_node_and_journals_it():
    files, records = MemFiles(), []
    _status(files, "n0")
    isl = _islands(files, ("n0",), records=records)
    s1 = S1Scheduler(S1Config(a_hard=1, start_seconds=5.0))
    gate = ConvertibleGate(s1, isl)
    assert isinstance(gate.decide(BacklogStats(batches=9, samples=36), LifecycleState.COLD), Hold)
    assert records[-1]["kind"] == CONVERSION_DECISION_KIND and records[-1]["action"] == "hold"
    _status(files, "n1")
    isl2 = _islands(files, ("n0", "n1"), records=records)
    gate2 = ConvertibleGate(s1, isl2)
    assert isinstance(gate2.decide(BacklogStats(batches=9, samples=36), LifecycleState.COLD), Start)
    assert isl2.pending is not None and isl2.pending.group == ("n0",)


def test_group_size_above_one_is_refused_on_a_pool():
    with pytest.raises(ValueError, match="one node"):
        ConvertibleIslands(MemFiles(), [_p("a")], NEEDS, group_size=2)


def test_data_ended_when_every_node_finished_generating():
    files = MemFiles()
    isl = _islands(files)
    _status(files, "n0", generation_done=True)
    assert not isl.data_ended()
    _status(files, "n1", role=EXITED)
    assert isl.data_ended()


# ---------------------------------------------------------------- end to end (fakes)
class Backlog:
    """The head's buffer view: a settable backlog."""

    def __init__(self):
        self.batches = 0

    def backlog(self, current_version):
        return BacklogStats(batches=self.batches, samples=self.batches * 4)


class Cuts:
    def __init__(self):
        self.cut = None

    def latest(self):
        return self.cut


def test_scheduler_converts_the_selected_node_trains_and_brings_it_back():
    """S1 decides on its own from the backlog; the gate picks the node by the rule
    (n1: fewer in-flight); the node trains two rounds, is stopped by ski rental and
    comes back to rollout with the trained version; n0 keeps generating."""
    files, records, clock = MemFiles(), [], Clock()
    agents = {n: _agent(files, name=n) for n in ("n0", "n1")}
    agents["n0"].inflight = lambda: 6
    agents["n1"].inflight = lambda: 2
    for a in agents.values():
        a.clock = clock
        a.generate = lambda: (threading.Event().wait(0.002), RolloutRound(4, 2.0))[1]
    threads = [threading.Thread(target=a.run, daemon=True) for a in agents.values()]
    for t in threads:
        t.start()
    _until(lambda: all(files.read(node_path(n, "status.json")) for n in agents))
    rec = lambda kind, tx_id=None, **f: records.append({"kind": kind, **f})  # noqa: E731
    isl = ConvertibleIslands(files, [_p(n, inflight=None, capacity=None) for n in agents], NEEDS,
                             record=rec, clock=clock, sleep=lambda s: threading.Event().wait(0.002),
                             poll_s=0.0)
    s1 = S1Scheduler(S1Config(a_hard=1, start_seconds=5.0), start_events=isl.start_events,
                     record=rec, clock=clock)
    lifecycle = RemoteTrainerLifecycle(isl, files, poll_s=0.0, clock=clock,
                                       sleep=lambda s: threading.Event().wait(0.002))
    cuts, backlog = Cuts(), Backlog()
    loop = TrainerLoop(scheduler=ConvertibleGate(s1, isl), lifecycle=lifecycle, buffer=backlog,
                       cuts=cuts, record=rec, clock=clock)
    backlog.batches = 1
    assert isinstance(loop.tick(), Hold)  # below Q_on = 2: keep sampling
    backlog.batches = 3
    assert isinstance(loop.tick(), Start)  # autonomous start: n1 converts
    convert = [r for r in records if r["kind"] == CONVERSION_DECISION_KIND][-1]
    assert convert["node"] == "n1" and convert["selection"]["group"] == ["n1"]
    assert [r["node"] for r in convert["selection"]["ranking"]] == ["n1", "n0"]
    assert loop.state is LifecycleState.TRAINING and agents["n1"].role == TRAINER
    loop.tick()  # trains one round
    loop.tick()  # and another
    backlog.batches = 0
    loop.tick()  # idle begins
    clock.t += 10  # ski rental: idle >= T_s
    loop.tick()  # stop: SCHEDULED cut, node converts back
    assert loop.state is LifecycleState.STOPPED and loop.handle.released
    _until(lambda: agents["n1"].role == ROLLOUT)
    assert agents["n1"].served_version == 2 and agents["n0"].role == ROLLOUT
    assert agents["n0"].rounds > 0  # the other node kept generating
    stops = [r for r in records if r["kind"] == "trainer_loop" and r["event"] == "stopped"]
    assert stops[-1]["reason"] == "STOPPED" and stops[-1]["trigger"] == "scheduled"
    assert s1.measured_start_seconds() is not None  # T_s is the conversion time
    isl.finish()
    for t in threads:
        t.join(5)
    assert all(a.role == EXITED for a in agents.values())


# ---------------------------------------------------------------- helpers
def _events(files, node):
    return [json.loads(x) for x in (files.read(node_path(node, "events.jsonl")) or b"").decode().splitlines()]


def _phase(files, node, phase):
    return any(e.get("phase") == phase for e in _events(files, node))


def _until(cond, timeout=5.0):
    done = threading.Event()
    deadline = 0.0
    while not cond():
        done.wait(0.002)
        deadline += 0.002
        assert deadline < timeout, "condition not reached"


def _catch(fn, errors):
    try:
        fn()
    except Exception as error:  # noqa: BLE001
        errors.append(error)


_ = CONVERTING  # imported for the role names


# ---------------------------------------------------------------- Miles node wiring (fakes)
def test_miles_convertible_node_boots_generates_converts_and_ends(tmp_path, monkeypatch):
    """run_convertible_node over fake Miles components: boot (version 0 weights + KV),
    a rollout round into the shard, one conversion with a stubbed trainer
    incarnation (published version 2), the way back (load_version + checksum),
    then the end command."""
    import types

    from yeto.rl.adapters.miles import buffer_rollout as br
    from yeto.rl.adapters.miles import convertible as mconv
    from yeto.rl.adapters.miles import publish, rollout
    from yeto.rl.engine import trainer_remote
    from yeto.rl.engine.sample_roles import SampleBufferConfig

    log = []

    class Controller:
        async def prepare_rollout(self, rid):
            log.append(("prepare_rollout", rid))

        async def abort_all(self):
            log.append("abort_all")

        async def onload_kv(self):
            log.append("onload_kv")

        async def offload(self, tags=None):
            log.append(("offload", tuple(tags)))

    class Executor:
        async def get(self, rid):
            log.append(("get", rid))
            return "pack"

    class Actor:
        async def run_plugin(self, name, kw):
            log.append(("plugin", kw))

        async def onload(self):
            log.append("wake_trainer")

    class Runner:
        def run(self, coro):
            import asyncio

            return asyncio.run(coro)

    class Pool:
        def __init__(self, **kw):
            pass

        def load_sample(self):
            return {"running_requests": 3, "queued_requests": 1}

        def set_admission(self, open_):
            log.append(("admission", open_))

        def members(self):
            return frozenset()

        def _in_executor(self, fn, *a):
            return fn(None, *a)

    class Sink:
        def discard(self):
            return 0

    class Publisher:
        published_checksum = "sha-2"

        def load_version(self, v):
            log.append(("load_version", v))

        def served_checksum(self):
            return "sha-2"

    driver = types.SimpleNamespace(publisher=Publisher(), sync=types.SimpleNamespace(
        round_cuts=types.SimpleNamespace(controller=types.SimpleNamespace(store=object()),
                                         buffer_watermark=lambda v: 0)))

    class Agent:
        def __init__(self, drv, *, commands, incarnation, resume_from, emit, **kw):
            self.handle = None
            self.emit, self.incarnation = emit, incarnation

        def run(self):
            assert br.convertible_mode() == br.TRAINER  # reading the buffer while training
            h = TrainerHandle(StartPlan(self.incarnation), resume_from=None, clock=lambda: 0.0)
            h.start_version, h.published_version = 0, 2
            self.emit(EVENT, **lifecycle_event(h, READY, seq=0))
            return h

    async def update_weights(*a):
        log.append("update_weights")

    monkeypatch.setattr(rollout, "MilesRolloutPool", Pool)
    monkeypatch.setattr(rollout, "RayMetadataSink", Sink)
    monkeypatch.setattr(rollout, "_release", lambda pack: log.append(("release", pack)))
    monkeypatch.setattr(publish, "_default_update_weights", lambda: update_weights)
    monkeypatch.setattr(trainer_remote, "IslandTrainerAgent", Agent)
    monkeypatch.setattr(mconv, "_kv_tags", lambda: ["kv_cache", "cuda_graph"])
    monkeypatch.setenv(br.SHARD_ENV, "island0")
    monkeypatch.setenv(trainer_remote.POLL_ENV, "0.01")
    store = tmp_path / "store"
    files = LocalStoreFiles(CheckpointStore(str(store)))
    miles_args = types.SimpleNamespace(
        yeto_rl_sample_buffer=SampleBufferConfig(role="convertible", store=str(store)),
        num_rollout=1, rollout_batch_size=4, offload_rollout=True, offload_train=True,
        yeto_rl_event_tape=str(tmp_path / "tape.jsonl"), yeto_rl_learner_id=0)
    NodeCommandWriter(files, "island0")
    import threading as _t

    def drive():
        _until(lambda: files.read(node_path("island0", "status.json")) and json.loads(
            files.read(node_path("island0", "status.json")))["generation_done"])
        NodeCommandWriter(files, "island0").send(TO_TRAINER, incarnation=0)
        _until(lambda: any(e.get("direction") == "to_rollout" for e in _events(files, "island0")))
        NodeCommandWriter(files, "island0").send(END)

    t = _t.Thread(target=drive, daemon=True)
    t.start()
    stints = mconv.run_convertible_node(miles_args, build_driver=lambda: driver,
                                        controller=Controller(), executor=Executor(),
                                        actor=Actor(), runner=Runner())
    t.join(5)
    assert stints == 1 and br.convertible_mode() == br.SAMPLER
    assert log[:3] == [("plugin", {"next_version": 0}), "update_weights", "onload_kv"]
    assert ("get", 0) in log and ("release", "pack") in log
    i = log.index("abort_all")
    assert log[i - 1] == ("admission", False)
    # the driver's first publication needs a fresh-boot engine: weights in, KV out
    assert log[i + 1:i + 3] == [("offload", ("kv_cache", "cuda_graph")), "wake_trainer"]
    assert ("load_version", 2) in log and log[-1] == ("admission", True)
    back = [e for e in _events(files, "island0") if e.get("direction") == "to_rollout"][0]
    assert back["version"] == 2 and back["weight_checksum"] == "sha-2"
    status = json.loads(files.read(node_path("island0", "status.json")))
    assert status["role"] == EXITED and status["served_version"] == 2


def test_store_writes_are_retried_on_os_errors():
    from yeto.rl.engine.convertible_remote import write_retrying

    class Flaky(MemFiles):
        def __init__(self, fails):
            super().__init__()
            self.fails = fails

        def write(self, rel, data):
            if self.fails:
                self.fails -= 1
                raise PermissionError(1, "Operation not permitted")
            super().write(rel, data)

    files = Flaky(2)
    write_retrying(files, "a", b"x", wait=lambda s: None)
    assert files.read("a") == b"x"
    with pytest.raises(PermissionError):
        write_retrying(Flaky(9), "a", b"x", wait=lambda s: None)


# ---------------------------------------------------------------- follow + backlog cap
def test_pool_backlog_gate_holds_at_the_cap_and_ignores_a_missing_or_stale_count():
    from yeto.rl.engine.convertible_remote import PoolBacklogGate, write_pool_backlog

    files, clock = MemFiles(), Clock()
    gate = PoolBacklogGate(files, 8, stale_s=60.0, clock=clock)
    assert gate() and gate.last is None  # the head has not counted yet
    write_pool_backlog(files, groups=7, version=2, at=clock.t, shards={"n0": 0, "n1": 0})
    assert gate() and gate.backlog == 7 and gate.last["shards"] == {"n0": 0, "n1": 0}
    write_pool_backlog(files, groups=8, version=2, at=clock.t, shards={"n0": 0, "n1": 0})
    assert not gate()
    clock.t += 61.0  # the head stopped writing: do not stall the node
    assert gate() and gate.backlog is None
    with pytest.raises(ValueError):
        PoolBacklogGate(files, 0)


def test_pool_backlog_gate_adds_the_groups_appended_since_the_heads_count(tmp_path):
    """The head counts between trainer rounds only; groups the nodes wrote since
    then count too (GPU run s24-final-conv-20261019d ran past the cap)."""
    from yeto.rl.engine.convertible_remote import PoolBacklogGate, write_pool_backlog

    files, clock = LocalStoreFiles(CheckpointStore(str(tmp_path))), Clock()
    n0 = FileSampleBuffer(CheckpointStore(str(tmp_path)), batch_size=4, index=shard_index("n0"))
    for i in range(2):
        _write(n0, f"a{i}")
    write_pool_backlog(files, groups=5, version=1, at=clock.t, shards={"n0": 2, "n1": 0})
    gate = PoolBacklogGate(files, 8, clock=clock)
    assert gate() and gate.backlog == 5
    for i in range(3):
        _write(n0, f"b{i}")
    assert not gate() and gate.backlog == 8  # 5 counted + 3 appended since


def test_node_follows_releases_and_waits_while_the_pool_backlog_is_full():
    files, clock = MemFiles(), Clock()
    admit = iter([True, False, False, True, True])
    follows = iter([None, 3, None, None, None])
    calls = []
    node = ConvertibleNode(NodeSpec("n0", NodeClass.CONVERTIBLE), HandoffPort([]))

    def generate():
        calls.append("round")
        return RolloutRound(4, 2.0)

    def follow():
        calls.append("follow")
        return next(follows)

    def wait(seconds):
        clock.t += 1.0

    agent = ConvertibleNodeAgent(
        name="n0", files=files, node=node, generate=generate, run_stint=FakeStint(files),
        published_checksum=lambda: None, follow=follow, admit=lambda: next(admit),
        max_rounds=3, poll_s=0.001, status_s=3600, idle_exit_s=0.0, clock=clock, wait=wait)
    agent.run()
    # follow runs before every admission check; a held round is not generated
    assert calls == ["follow", "round", "follow", "follow", "follow", "round", "follow", "round"]
    assert agent.rounds == 3 and agent.served_version == 3
    events = [e["event"] for e in _events(files, "n0")]
    assert events.count("node_backlog_full") == 1  # once per wait, not per poll
    admitted = [e for e in _events(files, "n0") if e["event"] == "node_backlog_admitted"]
    assert len(admitted) == 1 and admitted[0]["waited_s"] == 2.0


def test_after_stint_runs_once_the_node_is_back_to_rollout():
    files, log = MemFiles(), []
    order = []
    node = ConvertibleNode(NodeSpec("n0", NodeClass.CONVERTIBLE), HandoffPort(log))
    agent = ConvertibleNodeAgent(
        name="n0", files=files, node=node, generate=lambda: RolloutRound(4, 2.0),
        run_stint=FakeStint(files), published_checksum=lambda: None,
        after_stint=lambda: order.append(("after_stint", list(log))), max_rounds=0,
        poll_s=0.001, status_s=0.05, idle_exit_s=3600)
    thread = threading.Thread(target=agent.run, daemon=True)
    thread.start()
    NodeCommandWriter(files, "n0").send(TO_TRAINER, incarnation=0)
    append_trainer_command(files, 0, RUN, rounds=1)
    _until(lambda: _phase(files, "n0", TRAINED))
    append_trainer_command(files, 0, STOP, reason="STOPPED")
    _until(lambda: agent.role == ROLLOUT and agent.stints == 1)
    NodeCommandWriter(files, "n0").send(END)
    thread.join(5)
    # once the engines serve the trained version, before the next rollout round
    assert order == [("after_stint", ["stop_requests", "release_rollout", "onload_trainer",
                                      "offload_trainer", ("resume_rollout", 1),
                                      "accept_requests"])]


def test_merged_backlog_writes_the_trainable_count_for_the_nodes(tmp_path):
    from yeto.launch.convertible_pool import MergedBacklog
    from yeto.rl.engine.convertible_remote import POOL_BACKLOG_PATH

    class Inner:
        buffer = type("B", (), {"batch_size": 4})()

        def backlog(self, current_version):
            return BacklogStats(batches=3.0, samples=12, oldest_version=0,
                                version_histogram={0: 4, 2: 8})

    files = MemFiles()
    merged = MergedBacklog(files, ("n0",), Inner(), max_policy_age=1, clock=lambda: 50.0)
    stats = merged.backlog(3)
    assert stats.samples == 8 and stats.batches == 2.0  # the v0 groups are too old to train
    assert json.loads(files.read(POOL_BACKLOG_PATH)) == {"groups": 8, "version": 3, "at": 50.0,
                                                         "shards": {"n0": 0}}


def test_miles_convertible_node_follows_another_nodes_release_but_not_its_own(tmp_path, monkeypatch):
    """A node in rollout loads the newest release (another node's trainer released
    it); after its own stint it serves its own release and does not load it again.
    Converted a second time, it rewinds its trainer's LR scheduler before the
    restore."""
    import types

    from yeto.rl.adapters.miles import buffer_rollout as br
    from yeto.rl.adapters.miles import convertible as mconv
    from yeto.rl.adapters.miles import publish, rollout
    from yeto.rl.adapters.miles import sampler_island as si
    from yeto.rl.engine import trainer_remote
    from yeto.rl.engine.convertible_remote import write_pool_backlog
    from yeto.rl.engine.policy_release import LORA, PolicyReleaseStore
    from yeto.rl.engine.sample_roles import SampleBufferConfig

    store = tmp_path / "store"
    releases = PolicyReleaseStore(CheckpointStore(str(store)))

    def adapter(directory):
        (directory / "adapter_model.safetensors").write_text("a")

    releases.release(1, "h1", adapter, weights=LORA, engine_checksum="sha-1")  # node n1 trained v1
    loads, built = [], []

    class Loader:
        def __init__(self, controller, executor, run, *, lora=False, adapter_loaded=False):
            built.append((lora, adapter_loaded))
            self.last_checksum, self.last_probe = None, None

        def __call__(self, release, directory):
            loads.append(release.version)
            self.last_checksum = release.engine_checksum

    class Controller:
        async def prepare_rollout(self, rid):
            pass

        async def abort_all(self):
            pass

        async def onload_kv(self):
            pass

        async def offload(self, tags=None):
            pass

    class Executor:
        async def get(self, rid):
            return "pack"

    plugins = []

    class Actor:
        async def run_plugin(self, name, kw):
            plugins.append(name.rsplit(".", 1)[-1])
            return [{"rewound_from": 3}]

        async def onload(self):
            pass

    class Runner:
        def run(self, coro):
            import asyncio

            return asyncio.run(coro)

    class Pool:
        def __init__(self, **kw):
            pass

        def load_sample(self):
            return None

        def set_admission(self, open_):
            pass

        def _in_executor(self, fn, *a):
            return fn(None, *a)

    class Sink:
        def discard(self):
            return 0

    class Publisher:
        published_checksum = "sha-2"

        def load_version(self, v):
            pass

        def served_checksum(self):
            return "sha-2"

    driver = types.SimpleNamespace(publisher=Publisher(), sync=types.SimpleNamespace(
        round_cuts=types.SimpleNamespace(controller=types.SimpleNamespace(store=object()),
                                         buffer_watermark=lambda v: 0)))

    class Agent:
        def __init__(self, drv, *, commands, incarnation, resume_from, emit, **kw):
            self.handle = None
            self.emit, self.incarnation = emit, incarnation

        def run(self):
            # this node's trainer publishes v2 and releases it (ReleasingPublisher)
            releases.release(2, "h2", adapter, weights=LORA, engine_checksum="sha-2")
            h = TrainerHandle(StartPlan(self.incarnation), resume_from=None, clock=lambda: 0.0)
            h.start_version, h.published_version = 1, 2
            self.emit(EVENT, **lifecycle_event(h, READY, seq=0))
            return h

    async def update_weights(*a):
        pass

    monkeypatch.setattr(rollout, "MilesRolloutPool", Pool)
    monkeypatch.setattr(rollout, "RayMetadataSink", Sink)
    monkeypatch.setattr(rollout, "_release", lambda pack: None)
    monkeypatch.setattr(publish, "_default_update_weights", lambda: update_weights)
    monkeypatch.setattr(trainer_remote, "IslandTrainerAgent", Agent)
    monkeypatch.setattr(si, "EngineReleaseLoader", Loader)
    monkeypatch.setattr(mconv, "_kv_tags", lambda: ["kv_cache", "cuda_graph"])
    monkeypatch.setenv(br.SHARD_ENV, "island0")
    monkeypatch.setenv(trainer_remote.POLL_ENV, "0.01")
    files = LocalStoreFiles(CheckpointStore(str(store)))
    write_pool_backlog(files, groups=0, version=1, at=__import__("time").time(), shards={})
    tape = tmp_path / "tape.jsonl"
    miles_args = types.SimpleNamespace(
        yeto_rl_sample_buffer=SampleBufferConfig(role="convertible", store=str(store),
                                                 max_backlog_groups=16),
        num_rollout=2, rollout_batch_size=4, offload_rollout=True, offload_train=True,
        lora_rank=16, yeto_rl_event_tape=str(tape), yeto_rl_learner_id=0)
    NodeCommandWriter(files, "island0")

    def status():
        raw = files.read(node_path("island0", "status.json"))
        return json.loads(raw) if raw else {}

    def back(incarnation):
        return any(e.get("direction") == "to_rollout" and e.get("incarnation") == incarnation
                   for e in _events(files, "island0"))

    def drive():
        _until(lambda: status().get("rounds", 0) >= 1)
        NodeCommandWriter(files, "island0").send(TO_TRAINER, incarnation=0)
        _until(lambda: back(0))
        NodeCommandWriter(files, "island0").send(TO_TRAINER, incarnation=1)  # the same node again
        _until(lambda: back(1))
        _until(lambda: status().get("generation_done"))
        NodeCommandWriter(files, "island0").send(END)

    t = threading.Thread(target=drive, daemon=True)
    t.start()
    assert mconv.run_convertible_node(miles_args, build_driver=lambda: driver,
                                      controller=Controller(), executor=Executor(),
                                      actor=Actor(), runner=Runner()) == 2
    t.join(5)
    # the second stint restores into the trainer the first one stepped: scheduler rewound once
    assert plugins == ["align_weight_version", "rewind_scheduler"]
    assert built == [(True, True)]  # LoRA, and the engines already hold the boot adapter
    assert loads == [1]  # n1's release once; its own v2 release is not loaded again
    assert status()["served_version"] == 2
    tape_events = [json.loads(line) for line in tape.read_text().splitlines()]
    loaded = [e for e in tape_events if e["event"] == "rl_convertible_policy_loaded"]
    assert len(loaded) == 1 and loaded[0]["policy_version"] == 1
    assert loaded[0]["served_checksum"] == "sha-1" == loaded[0]["engine_checksum"]
    start = [e for e in tape_events if e["event"] == "rl_convertible_node_start"][0]
    assert start["max_backlog_groups"] == 16
    rewound = [e for e in tape_events if e["event"] == "rl_convertible_scheduler_rewound"]
    assert [(e["incarnation"], e["ranks"]) for e in rewound] == [(1, [{"rewound_from": 3}])]
