"""Convertible nodes (rl-trainer-start-stop L7.1–L7.3) with fake engines."""

from __future__ import annotations

import argparse

import pytest

from yeto.launch import trainer_loop as launch_loop
from yeto.rl.adapters.miles import LoopRunner
from yeto.rl.adapters.miles.convertible import (
    MilesConversionPort,
    engine_requests_inflight,
    miles_conversion_port,
)
from yeto.rl.adapters.miles.publish import MilesPublisher, PublicationError, engine_checksum_digest
from yeto.rl.engine.convertible import (
    CONVERSION_DECISION_KIND,
    ConversionError,
    ConversionScheduler,
    ConvertibleNode,
    InflightTrajectory,
    LastRolloutNodeError,
    NodeClass,
    NodeClassError,
    NodePool,
    NodeSpec,
    Role,
    apply_cut_rule,
    parse_node_spec,
    validate_nodes,
)
from yeto.rl.engine.trainer_lifecycle import BacklogStats, Hold, SampleIndexEntry
from yeto.rl.engine.trainer_scheduler import S1Config, S1Scheduler


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class FakeGpu:
    """One GPU that holds either the rollout engine or the trainer, never both."""

    def __init__(self, clock: Clock, inflight=(), checksums=None) -> None:
        self.clock = clock
        self.holder = "rollout"
        self.accepting = True
        self.calls: list[str] = []
        self.inflight = list(inflight)
        self.checksums = checksums or {}

    def _step(self, name: str, seconds: float = 1.0) -> None:
        self.calls.append(name)
        self.clock.t += seconds

    def stop_requests(self):
        self._step("stop_requests")
        self.accepting = False
        out, self.inflight = self.inflight, []
        return out

    def release_rollout(self):
        assert self.holder == "rollout" and not self.accepting
        self._step("release_rollout")
        self.holder = None

    def onload_trainer(self):
        assert self.holder is None, "trainer onload before rollout released its memory"
        self._step("onload_trainer", 2.0)
        self.holder = "trainer"

    def offload_trainer(self):
        assert self.holder == "trainer"
        self._step("offload_trainer")
        self.holder = None

    def resume_rollout(self, version):
        assert self.holder is None, "rollout resume before trainer offload"
        self._step("resume_rollout", 3.0)
        self.holder = "rollout"
        return self.checksums.get(version, f"sha-{version}")

    def accept_requests(self):
        self._step("accept_requests")
        self.accepting = True


class Buffer:
    def __init__(self) -> None:
        self.entries: list[SampleIndexEntry] = []

    def append(self, entry):
        self.entries.append(entry)


def _entry(sid, version=3):
    return SampleIndexEntry(sample_id=sid, behavior_version=version, payload_ref=f"p/{sid}")


def _node(name="c0", clock=None, **gpu):
    clock = clock or Clock()
    return ConvertibleNode(NodeSpec(name, NodeClass.CONVERTIBLE), FakeGpu(clock, **gpu), clock=clock)


# ---------------------------------------------------------------- L7.1 classes
def test_convertible_spot_or_low_precision_is_refused_by_name():
    with pytest.raises(NodeClassError, match="node c1: a spot node cannot be convertible"):
        validate_nodes([NodeSpec("c1", NodeClass.CONVERTIBLE, spot=True)], trainer_precision="bf16")
    with pytest.raises(NodeClassError, match="node c2: rollout precision fp8"):
        validate_nodes([NodeSpec("c2", NodeClass.CONVERTIBLE, precision="fp8")], trainer_precision="bf16")
    validate_nodes([NodeSpec("r", NodeClass.ROLLOUT, spot=True, precision="fp8"),
                    NodeSpec("c", NodeClass.CONVERTIBLE, precision="bf16")], trainer_precision="bf16")


def test_parse_node_spec():
    assert parse_node_spec("island2=rollout:spot:fp8:capacity=2") == NodeSpec(
        "island2", NodeClass.ROLLOUT, spot=True, precision="fp8", capacity=2.0)
    with pytest.raises(NodeClassError, match="class 'gpu'"):
        parse_node_spec("x=gpu")


def _launch_args(argv):
    parser = argparse.ArgumentParser()
    launch_loop.add_arguments(parser)
    parser.add_argument("--on-demand", dest="spot", action="store_false", default=True)
    return parser.parse_args(argv)


def test_launcher_refuses_convertible_on_a_spot_run():
    args = _launch_args(["--rl-node-class", "island1=convertible"])  # --spot is the default
    with pytest.raises(SystemExit, match="node island1: a spot node cannot be convertible"):
        launch_loop.check(args)
    ok = _launch_args(["--on-demand", "--rl-node-class", "island1=convertible",
                       "--rl-node-class", "island2=rollout:spot:fp8"])
    launch_loop.check(ok)
    assert [s.spot for s in launch_loop.node_specs(ok)] == [False, True]


# ---------------------------------------------------------------- L7.2 cut rule
def test_cut_rule_returns_finished_segments_and_drops_unfinished():
    buf = Buffer()
    out = apply_cut_rule([
        InflightTrajectory("t0", finished=(_entry("t0.s0"), _entry("t0.s1")), unfinished=True),
        InflightTrajectory("t1", finished=(_entry("t1.s0"),), unfinished=False),
        InflightTrajectory("t2"),
    ], buf)
    assert out.returned == ("t0.s0", "t0.s1", "t1.s0")
    assert out.dropped == ("t0", "t2")
    assert [e.sample_id for e in buf.entries] == list(out.returned)
    assert {e.behavior_version for e in buf.entries} == {3}  # behavior version kept


# ---------------------------------------------------------------- L7.2 both directions
def test_both_directions_in_order_with_timing_and_mid_conversion_trajectories():
    inflight = [InflightTrajectory("t0", finished=(_entry("t0.s0"),))]
    node = _node(inflight=inflight)
    gpu, buf = node.port, Buffer()

    r1 = node.to_trainer(buf)
    assert gpu.calls == ["stop_requests", "release_rollout", "onload_trainer"]
    assert node.role is Role.TRAINER and gpu.holder == "trainer"
    assert r1.seconds == 4.0 and r1.phases == {"stop_requests": 1.0, "release_rollout": 1.0,
                                                "onload_trainer": 2.0}
    assert r1.cut.returned == ("t0.s0",) and r1.cut.dropped == ("t0",)
    assert [e.sample_id for e in buf.entries] == ["t0.s0"]

    r2 = node.to_rollout(7, expected_checksum="sha-7")
    assert gpu.calls[3:] == ["offload_trainer", "resume_rollout", "accept_requests"]
    assert node.role is Role.ROLLOUT and gpu.holder == "rollout" and gpu.accepting
    assert r2.seconds == 5.0 and r2.weight_checksum == "sha-7"
    ev = r2.event()
    assert ev["event"] == "node_conversion" and ev["direction"] == "to_rollout" and ev["version"] == 7


def test_checksum_mismatch_fails_and_engine_stays_closed():
    node = _node(checksums={7: "sha-other"})
    node.to_trainer(Buffer())
    with pytest.raises(ConversionError, match="weight checksum sha-other != published sha-7"):
        node.to_rollout(7, expected_checksum="sha-7")
    assert node.role is Role.FAILED and not node.port.accepting


def test_wrong_role_and_port_failure():
    node = _node()
    with pytest.raises(ConversionError, match="needs role trainer"):
        node.to_rollout(1)

    def boom():
        raise RuntimeError("cuda oom")
    node.port.onload_trainer = boom
    with pytest.raises(ConversionError, match="rollout -> trainer failed: cuda oom"):
        node.to_trainer(Buffer())
    assert node.role is Role.FAILED


def test_only_convertible_class_gets_a_node():
    with pytest.raises(NodeClassError, match="not convertible"):
        ConvertibleNode(NodeSpec("r", NodeClass.ROLLOUT), FakeGpu(Clock()))


# ---------------------------------------------------------------- L7.3 capacity and guard
def _pool(clock, *, rollout=1, convertible=2, capacity=1.0):
    specs = [NodeSpec(f"r{i}", NodeClass.ROLLOUT, spot=True) for i in range(rollout)]
    specs.append(NodeSpec("t0", NodeClass.TRAINER))
    nodes = [ConvertibleNode(NodeSpec(f"c{i}", NodeClass.CONVERTIBLE, capacity=capacity),
                             FakeGpu(clock), clock=clock) for i in range(convertible)]
    return NodePool(specs, nodes)


def test_last_rollout_node_is_never_converted():
    clock = Clock()
    pool = _pool(clock, rollout=0, convertible=2)
    assert pool.check_conversion(["c0"]) == 1.0
    with pytest.raises(LastRolloutNodeError):
        pool.check_conversion(["c0", "c1"])
    pool.convertible["c0"].to_trainer(Buffer())
    with pytest.raises(LastRolloutNodeError):
        pool.check_conversion(["c1"])


def test_capacity_counts_conversion_and_trainer_class_does_no_rollout():
    clock = Clock()
    pool = _pool(clock, rollout=2, convertible=1, capacity=0.5)
    assert pool.total_capacity() == 2.5  # t0 (trainer class) is not counted
    pool.convertible["c0"].to_trainer(Buffer())
    assert pool.capacity() == 2.0 and pool.doing_rollout() == ["r0", "r1"]


def test_idle_counts_only_time_with_neither_rollout_nor_training():
    clock = Clock()
    node = _node(clock=clock)
    clock.t += 100  # rollout: not idle
    node.to_trainer(Buffer())
    node.set_training(True)
    clock.t += 50  # training: not idle
    node.set_training(False)
    clock.t += 7  # neither
    assert node.idle_seconds() == 7
    node.to_rollout(1)  # the conversion time is not idle either
    clock.t += 30
    assert node.idle_seconds() == 7


def _sched(clock, pool, records=None, group_size=1):
    s1 = S1Scheduler(S1Config(a_hard=1, start_seconds=10.0), clock=clock)
    rec = (lambda kind, tx_id=None, **f: records.append({"kind": kind, **f})) if records is not None else None
    return ConversionScheduler(s1, pool, group_size=group_size, record=rec)


def test_scheduler_converts_on_backlog_and_back_by_ski_rental():
    clock, records, buf = Clock(), [], Buffer()
    pool = _pool(clock, rollout=1, convertible=2)
    sched = _sched(clock, pool, records)

    assert sched.decide(BacklogStats(batches=1, samples=4)).to_trainer == ()
    d = sched.decide(BacklogStats(batches=2, samples=8))  # Q_on = a_hard + 1 = 2
    assert d.to_trainer == ("c0",) and d.lost_capacity == 1.0 and d.capacity_after == 2.0
    sched.apply(d, buf)
    assert pool.capacity() == 2.0

    clock.t += 1
    assert sched.decide(BacklogStats(batches=0, samples=0)).to_rollout == ()  # idle starts
    clock.t += 10  # T_idle = T_s with an empty backlog
    back = sched.decide(BacklogStats(batches=0, samples=0))
    assert back.to_rollout == ("c0",) and back.lost_capacity == -1.0 and back.note == "stop:STOPPED"
    receipts = sched.apply(back, buf, version=4, expected_checksum="sha-4")
    assert receipts[0].weight_checksum == "sha-4" and pool.capacity() == 3.0
    assert [e["direction"] for e in sched.events] == ["to_trainer", "to_rollout"]
    assert [r["kind"] for r in records] == [CONVERSION_DECISION_KIND] * 2


def test_scheduler_holds_instead_of_converting_the_last_rollout_node():
    clock, records = Clock(), []
    pool = _pool(clock, rollout=0, convertible=2)
    sched = _sched(clock, pool, records, group_size=2)
    d = sched.decide(BacklogStats(batches=5, samples=20))
    assert d.to_trainer == () and d.note == "hold:last_rollout_node"
    assert records[-1]["decision"]["note"] == "hold:last_rollout_node"
    sched.group_size = 1
    assert sched.decide(BacklogStats(batches=5, samples=20)).to_trainer == ("c0",)


def test_scheduler_finishes_at_data_end():
    clock = Clock()
    pool = _pool(clock)
    sched = _sched(clock, pool)
    sched.apply(sched.decide(BacklogStats(batches=3, samples=12)), Buffer())
    d = sched.decide(BacklogStats(batches=0, samples=0, data_ended=True))
    assert d.to_rollout == ("c0",) and d.note == "stop:FINISHED"


def test_inner_decision_types_are_the_lifecycle_ones():
    # A transition state holds: the inner scheduler is asked with a non-startable state.
    clock = Clock()
    pool = _pool(clock)
    pool.convertible["c0"].role = Role.TO_TRAINER
    sched = _sched(clock, pool)
    assert sched.decide(BacklogStats(batches=9, samples=36)).to_trainer == ()
    assert isinstance(sched.inner.decide(BacklogStats(batches=0, samples=0),
                                         sched.lifecycle_state()), Hold)


# ---------------------------------------------------------------- Miles port
class FakeController:
    def __init__(self, log):
        self.log = log

    async def abort_all(self):
        self.log.append("abort_all")

    async def offload(self, tags=None):
        self.log.append(("offload", tuple(tags)))

    async def onload(self, tags=None):
        self.log.append(("onload", tuple(tags)))

    async def onload_kv(self):
        self.log.append("onload_kv")


class FakeTrainer:
    def __init__(self, log):
        self.log = log

    def onload(self):
        self.log.append("trainer.onload")

    def offload(self):
        self.log.append("trainer.offload")


def _run(coro):
    try:
        coro.send(None)
    except StopIteration as stop:
        return stop.value
    raise AssertionError("fake coroutine did not finish")


def test_miles_port_call_order_both_directions():
    log: list = []
    port = MilesConversionPort(
        controller=FakeController(log), trainer=FakeTrainer(log), run=_run,
        load_weights=lambda v: (log.append(("load_weights", v)), f"sha-{v}")[1],
        collect_inflight=lambda: (log.append("collect"), [InflightTrajectory("t9")])[1],
        set_admission=lambda on: log.append(("admission", on)),
        memory_tags=lambda: ["weights", "kv_cache", "cuda_graph"], weight_tags=lambda: ["weights"])
    clock = Clock()
    node = ConvertibleNode(NodeSpec("c0", NodeClass.CONVERTIBLE), port, clock=clock)
    r = node.to_trainer(Buffer())
    assert r.cut.dropped == ("t9",)
    node.to_rollout(5, expected_checksum="sha-5")
    assert log == [("admission", False), "collect", "abort_all",
                   ("offload", ("weights", "kv_cache", "cuda_graph")), "trainer.onload",
                   "trainer.offload", ("onload", ("weights",)), ("load_weights", 5), "onload_kv",
                   ("admission", True)]


class ChecksumController(FakeController):
    """Engines whose ``check_weights(checksum)`` body is the last loaded version."""

    def __init__(self, log, engines=1):
        super().__init__(log)
        self.loaded = None
        self.engines = engines
        self.skew = False

    async def check_weights(self, action):
        assert action == "checksum"
        bodies = [{"w": f"v{self.loaded}", "e": i if self.skew else 0} for i in range(self.engines)]
        return bodies


def _publisher(log, controller):
    async def update_weights(args, actor, executor, ctrl):
        log.append("update_weights")
        ctrl.loaded = 7

    return MilesPublisher(args=argparse.Namespace(offload_rollout=True), actor_model=None,
                          rollout_executor=None, inference_controller=controller,
                          update_weights=update_weights, flatten_checksums=list,
                          runner=LoopRunner())


def test_publisher_load_version_returns_the_engine_checksum_digest():
    log: list = []
    controller = ChecksumController(log, engines=2)
    publisher = _publisher(log, controller)
    assert publisher.published_checksum is None
    digest = publisher.load_version(7)
    assert digest == engine_checksum_digest({"e": 0, "w": "v7"})
    assert log == ["update_weights"]  # no onload, no KV: the port does those
    publisher._reference = ("token", {"w": "v7", "e": 0})
    assert publisher.published_checksum == digest


def test_publisher_load_version_fails_when_engines_differ():
    log: list = []
    controller = ChecksumController(log, engines=2)
    controller.skew = True
    with pytest.raises(PublicationError, match="2 different weight checksums"):
        _publisher(log, controller).load_version(7)


class FakePool:
    def __init__(self, log, sample):
        self.log = log
        self.sample = sample

    def load_sample(self):
        return self.sample

    def set_admission(self, open_):
        self.log.append(("admission", open_))


def test_engine_requests_inflight_counts_running_and_queued_or_nothing_when_unknown():
    assert [t.trajectory_id for t in engine_requests_inflight(
        FakePool([], {"running_requests": 2, "queued_requests": 1}))] == [
        "engine-request-0", "engine-request-1", "engine-request-2"]
    assert engine_requests_inflight(FakePool([], None)) == []
    assert engine_requests_inflight(FakePool([], {"running_requests": None})) == []


def test_miles_conversion_port_wires_publisher_pool_and_trainer():
    log: list = []
    controller = ChecksumController(log)
    publisher = _publisher(log, controller)
    port = miles_conversion_port(controller=controller,
                                 rollout=FakePool(log, {"running_requests": 1, "queued_requests": 0}),
                                 trainer=FakeTrainer(log), publisher=publisher, run=_run)
    port._memory_tags = lambda: ["weights", "kv_cache", "cuda_graph"]
    port._weight_tags = lambda: ["weights"]
    node = ConvertibleNode(NodeSpec("c0", NodeClass.CONVERTIBLE), port, clock=Clock())
    assert node.to_trainer(Buffer()).cut.dropped == ("engine-request-0",)
    expected = engine_checksum_digest({"e": 0, "w": "v7"})
    receipt = node.to_rollout(7, expected_checksum=expected)
    assert receipt.weight_checksum == expected
    assert log == [("admission", False), "abort_all",
                   ("offload", ("weights", "kv_cache", "cuda_graph")), "trainer.onload",
                   "trainer.offload", ("onload", ("weights",)), "update_weights", "onload_kv",
                   ("admission", True)]
