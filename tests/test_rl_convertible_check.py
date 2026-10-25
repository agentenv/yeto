"""Correctness check of the convertible nodes (rl-trainer-start-stop D7 / L7), CPU.

* the node core (:class:`ConvertibleNode`) over several rollout -> trainer ->
  rollout round trips: call order, the engine checksum against the published one
  on every way back, the P0-S cut rule;
* a convertible pool end to end with the real head (S1, the convertible gate,
  the remote lifecycle, the merged backlog), the real node agents, the real
  island trainer agent and driver lifecycle over a fake engine, and the real
  file sample buffer with node shards: three round trips, a hand-over of the
  trainer state between nodes, every sample trained once;
* the agentic path at a conversion with a fake boat client (no sandbox is created).
"""

from __future__ import annotations

import enum
import json
import threading

import pytest
import torch

from test_rl_algorithm_provenance import _fake_sky, _no_modal_listing  # noqa: F401 - fixtures
from test_rl_engine_driver import NAME, PROFILE, _driver

from yeto.rl.agentic.types import OUTCOME_KEY
from yeto.rl.engine.bridges import LocalOnlySync
from yeto.rl.engine.convertible import (
    ConversionError,
    ConvertibleNode,
    InflightTrajectory,
    NodeClass,
    NodePool,
    NodeProfile,
    NodeSpec,
    Role,
    TrainerNeeds,
)
from yeto.rl.engine.convertible_remote import (
    EXITED,
    ROLLOUT,
    TRAINER,
    ConvertibleGate,
    ConvertibleIslands,
    ConvertibleNodeAgent,
    RolloutRound,
    node_path,
)
from yeto.rl.engine.fake_lifecycle import BufferFedFakeRollout, FakeRoundCuts, cuttable_fake_engine
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer, shard_index
from yeto.rl.engine.trainer_lifecycle import LifecycleState, SampleIndexEntry, Start, Stop
from yeto.rl.engine.trainer_loop import TrainerLoop
from yeto.rl.engine.trainer_remote import (
    CommandReader,
    IslandTrainerAgent,
    LocalStoreFiles,
    RemoteTrainerLifecycle,
)
from yeto.rl.engine.trainer_scheduler import S1Config, S1Scheduler


# ---------------------------------------------------------------- node core: round trips
class EnginePort:
    """A :class:`ConversionPort` over a fake engine: it serves one weight version
    and takes requests only while admission is open. ``releases`` maps a published
    version to its checksum; ``stale`` versions load the previous weights (a lost
    weight update)."""

    def __init__(self, releases, inflight=()):
        self.releases = releases
        self.inflight = list(inflight)
        self.version = 0
        self.admission = True
        self.memory = "rollout"
        self.stale: set[int] = set()
        self.calls: list = []

    def stop_requests(self):
        self.calls.append("stop_requests")
        self.admission = False
        out, self.inflight = self.inflight, []
        return out

    def release_rollout(self):
        self.calls.append("release_rollout")
        assert self.memory == "rollout"
        self.memory = "free"

    def onload_trainer(self):
        self.calls.append("onload_trainer")
        assert self.memory == "free"
        self.memory = "trainer"

    def offload_trainer(self):
        self.calls.append("offload_trainer")
        assert self.memory == "trainer"
        self.memory = "free"

    def resume_rollout(self, version):
        self.calls.append(("resume_rollout", version))
        assert self.memory == "free"
        self.memory = "rollout"
        if version not in self.stale:
            self.version = version
        return self.releases[self.version]

    def accept_requests(self):
        self.calls.append("accept_requests")
        self.admission = True


class Buffer:
    def __init__(self):
        self.entries: list[SampleIndexEntry] = []

    def append(self, entry):
        self.entries.append(entry)


ONE_WAY = ["stop_requests", "release_rollout", "onload_trainer"]


def test_three_round_trips_reload_the_published_checksum_every_time():
    releases = {0: "sha-0", 2: "sha-2", 5: "sha-5", 7: "sha-7"}
    port = EnginePort(releases)
    node = ConvertibleNode(NodeSpec("n0", NodeClass.CONVERTIBLE), port)
    buffer = Buffer()
    for version in (2, 5, 7):
        port.calls.clear()
        out = node.to_trainer(buffer)
        assert node.role is Role.TRAINER and not port.admission and port.memory == "trainer"
        back = node.to_rollout(version, expected_checksum=releases[version])
        assert port.calls == [*ONE_WAY, "offload_trainer", ("resume_rollout", version),
                              "accept_requests"]
        assert node.role is Role.ROLLOUT and port.admission and port.version == version
        assert (back.version, back.weight_checksum) == (version, releases[version])
        assert set(out.phases) == set(ONE_WAY)
        assert set(back.phases) == {"offload_trainer", "resume_rollout", "accept_requests"}
    assert buffer.entries == []


def test_a_stale_engine_on_a_later_trip_fails_closed():
    """Trip 1 loads version 2. On trip 2 the engine keeps version 2 instead of
    loading 5: the checksum differs, the node fails, admission stays closed and
    the node refuses another conversion."""
    releases = {0: "sha-0", 2: "sha-2", 5: "sha-5"}
    port = EnginePort(releases)
    node = ConvertibleNode(NodeSpec("n0", NodeClass.CONVERTIBLE), port)
    node.to_trainer(Buffer())
    node.to_rollout(2, expected_checksum="sha-2")
    node.to_trainer(Buffer())
    port.stale.add(5)
    with pytest.raises(ConversionError, match="sha-2 != published sha-5"):
        node.to_rollout(5, expected_checksum="sha-5")
    assert node.role is Role.FAILED and not port.admission
    assert port.calls[-1] == ("resume_rollout", 5)  # accept_requests never ran
    with pytest.raises(ConversionError, match="needs role rollout"):
        node.to_trainer(Buffer())


def _segment(trajectory, k, version):
    return SampleIndexEntry(f"{trajectory}-seg{k}", version, f"payloads/{trajectory}-{k}.bin", tokens=8)


class SegmentedSampler:
    """Long trajectories of ``segments`` parts each. A conversion cuts the ones in
    flight (P0-S): their finished segments are returned, the unfinished rest is
    dropped and the trajectory is generated again from its start, under a new
    attempt id, so no sample id repeats."""

    def __init__(self, segments=3):
        self.segments = segments
        self.progress: dict[str, int] = {}  # trajectory attempt -> finished segments
        self.attempts: dict[str, int] = {}
        self.done: list[SampleIndexEntry] = []

    def start(self, trajectory):
        attempt = f"{trajectory}.a{self.attempts.get(trajectory, 0)}"
        self.attempts[trajectory] = self.attempts.get(trajectory, 0) + 1
        self.progress[attempt] = 0
        return attempt

    def step(self, version):
        for attempt in list(self.progress):
            k = self.progress[attempt]
            if k < self.segments:
                self.progress[attempt] = k + 1
            if self.progress[attempt] == self.segments:
                self.done.extend(_segment(attempt, i, version) for i in range(self.segments))
                del self.progress[attempt]

    def inflight(self, version):
        return [InflightTrajectory(attempt, tuple(_segment(attempt, i, version) for i in range(k)),
                                   unfinished=True)
                for attempt, k in self.progress.items()]

    def restart(self, dropped):
        for attempt in dropped:
            del self.progress[attempt]
            self.start(attempt.rsplit(".a", 1)[0])


def test_cut_rule_over_round_trips_returns_each_finished_segment_once():
    """Three conversions in the middle of long trajectories: every finished
    segment reaches the buffer once, every cut trajectory is generated again,
    and no sample id is appended twice."""
    sampler = SegmentedSampler(segments=3)
    releases = {v: f"sha-{v}" for v in range(10)}
    port = EnginePort(releases)
    node = ConvertibleNode(NodeSpec("n0", NodeClass.CONVERTIBLE), port)
    buffer = Buffer()
    for t in ("t0", "t1", "t2", "t3"):
        sampler.start(t)
    version, cut = 0, 0
    for trip in range(3):
        sampler.step(version)  # every trajectory one segment further
        sampler.start(f"late{trip}")  # a new trajectory starts just before the cut
        inflight = port.inflight = sampler.inflight(version)
        out = node.to_trainer(buffer)
        assert list(out.cut.returned) == [e.sample_id for t in inflight for e in t.finished]
        assert set(out.cut.dropped) == {t.trajectory_id for t in inflight}
        cut += len(out.cut.dropped)
        sampler.restart(out.cut.dropped)
        version += 2
        node.to_rollout(version, expected_checksum=releases[version])
    for _ in range(3):
        sampler.step(version)  # no more conversions: everything finishes
    buffer.entries.extend(sampler.done)
    ids = [e.sample_id for e in buffer.entries]
    assert len(ids) == len(set(ids))  # nothing appended twice
    finished = {e.sample_id.split(".a")[0] for e in sampler.done}
    assert finished == {"t0", "t1", "t2", "t3", "late0", "late1", "late2"}  # nothing lost
    assert cut == sum(n - 1 for n in sampler.attempts.values())  # each cut regenerated once


def test_pool_never_converts_its_last_rollout_node_over_round_trips():
    ports = {n: EnginePort({0: "sha-0", 1: "sha-1"}) for n in ("a", "b")}
    nodes = [ConvertibleNode(NodeSpec(n, NodeClass.CONVERTIBLE), ports[n]) for n in ("a", "b")]
    pool = NodePool([n.spec for n in nodes], nodes)
    for name in ("a", "b", "a"):
        assert pool.check_conversion([name]) == 1.0
        pool.convertible[name].to_trainer(Buffer())
        other = "b" if name == "a" else "a"
        with pytest.raises(ConversionError, match="no node doing rollout|not a convertible"):
            pool.check_conversion([other])
        pool.convertible[name].to_rollout(1, expected_checksum="sha-1")
    with pytest.raises(ConversionError, match="no node doing rollout"):
        pool.check_conversion(["a", "b"])


# ---------------------------------------------------------------- pool end to end
BATCH = 4  # fake engine: 2 groups x 2 samples; one buffer entry is one sample
NEEDS = TrainerNeeds(precision="bf16", gpus=1, domains=("modal",))


class ManualClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class PoolCuts(FakeRoundCuts):
    """The fake round cuts of the store, with the buffer hooks of a real
    ``RoundCutCheckpoint``: the cut records the consumed watermark of the
    trainer's reader, and a restore resets the reader to the newest cut's
    watermark (0 without a cut). The trainer node sets both hooks."""

    def __init__(self, root):
        super().__init__(root)
        self.buffer_resume = None
        self.restores: list[tuple[str | None, int]] = []

    def resume(self, driver):
        info = super().resume(driver)
        watermark = 0 if info is None else int(self.latest().watermark)
        self.buffer_resume(watermark)
        self.restores.append((None if info is None else info["cut_id"], watermark))
        return info


class HandoffPort:
    """The node port of a pool node (``DriverHandoffPort`` shape): the engines load
    the version the trainer of this node published and report its checksum."""

    def __init__(self, node):
        self.node = node
        self.calls: list = []

    def stop_requests(self):
        self.calls.append("stop_requests")
        return []

    def release_rollout(self):
        self.calls.append("release_rollout")

    def onload_trainer(self):
        self.calls.append("onload_trainer")

    def offload_trainer(self):
        self.calls.append("offload_trainer")

    def resume_rollout(self, version):
        self.calls.append(("resume_rollout", version))
        published = self.node.engine.published
        return published[1] if published is not None and published[0] == version else None

    def accept_requests(self):
        self.calls.append("accept_requests")


class PoolNode:
    """One convertible island: its shard writer, the buffer reader of its rollout
    process (lives as long as the island, like the Miles rollout executor), and the
    trainer incarnations it runs (fresh driver each, as ``run_convertible_node``)."""

    def __init__(self, name, store, files, cuts, tmp_path, clock):
        self.name = name
        self.shard = FileSampleBuffer(store, batch_size=BATCH, index=shard_index(name))
        self.reader = FileSampleBuffer(store, batch_size=BATCH, create_index=False)
        self.cuts = cuts
        self.files = files
        self.tmp_path = tmp_path
        self.permits = 0
        self.generated: list[str] = []
        self.rollouts: list[BufferFedFakeRollout] = []
        self.published: list[tuple[int, str]] = []
        self.engine = None
        self.lock = threading.Lock()
        self.inflight = 0
        port = self.port = HandoffPort(self)
        self.agent = ConvertibleNodeAgent(
            name=name, files=files, node=ConvertibleNode(NodeSpec(name, NodeClass.CONVERTIBLE), port),
            generate=self.generate, run_stint=self.stint,
            published_checksum=lambda: None if self.engine is None else self.engine.published[1],
            inflight=lambda: self.inflight, admit=self.admit, poll_s=0.001, status_s=0.01,
            idle_exit_s=3600, clock=clock)

    def admit(self):
        with self.lock:
            return self.permits > 0

    def generate(self):
        with self.lock:
            self.permits -= 1
        for _ in range(BATCH):
            sid = f"{self.name}-{len(self.generated):04d}-v{self.agent.served_version}"
            ref = self.shard.write_payload(sid, sid.encode())
            self.shard.append(SampleIndexEntry(sid, self.agent.served_version, ref, tokens=3))
            self.generated.append(sid)
        return RolloutRound(BATCH, 1.0)

    def stint(self, incarnation, resume_from, emit):
        engine = self.engine = cuttable_fake_engine(tensors={NAME: torch.zeros(1, 2)}, step_delta=1.0,
                                                    capability_profile=PROFILE)
        engine.rollout = BufferFedFakeRollout(engine, self.reader)
        self.rollouts.append(engine.rollout)
        self.cuts.watermark = lambda: self.reader.watermark
        self.cuts.buffer_resume = self.reader.reset_to
        sync = LocalOnlySync(1000)
        sync.round_cuts = self.cuts
        driver = _driver(engine, sync, self.tmp_path, name=f"events-{self.name}-{incarnation}.jsonl")
        handle = IslandTrainerAgent(
            driver, commands=CommandReader(self.files, incarnation), incarnation=incarnation,
            resume_from=resume_from, emit=emit, cuts=self.cuts,
            watermark=lambda _version: self.reader.watermark, poll_s=0.001).run()
        self.published.append(engine.published)
        return handle

    def trained(self):
        return [sid for r in self.rollouts for rid in sorted(r.trained) for sid, _ in r.trained[rid]]


def _wait(cond, timeout=10.0):
    event = threading.Event()
    waited = 0.0
    while not cond():
        event.wait(0.002)
        waited += 0.002
        assert waited < timeout, "condition not reached"


def _node_events(files, name):
    raw = files.read(node_path(name, "events.jsonl")) or b""
    return [json.loads(line) for line in raw.decode().splitlines()]


def test_pool_three_round_trips_hand_the_trainer_over_and_train_each_sample_once(tmp_path):
    """Two convertible nodes. S1 starts a trainer three times on its own; the gate
    converts n1, then n0 (restores n1's cut: hand-over between nodes), then n1
    again (its rollout process already read the buffer in its first stint).
    Each time the node comes back with the trained version and the engine
    checksum equals the published one. The other node generates while one trains.
    At the end every generated sample was trained exactly once."""
    from yeto.launch.convertible_pool import MergedBacklog
    from yeto.rl.engine.sample_buffer import CommittedBacklog

    store = CheckpointStore(str(tmp_path / "store"))
    files = LocalStoreFiles(store)
    cuts = PoolCuts(tmp_path / "cuts")
    clock = ManualClock()
    nodes = {n: PoolNode(n, store, files, cuts, tmp_path, clock) for n in ("n0", "n1")}
    records: list[dict] = []
    rec = lambda kind, tx_id=None, **f: records.append({"kind": kind, **f})  # noqa: E731
    islands = ConvertibleIslands(
        files, [NodeProfile(n, gpu="H200", gpus=1, gpu_memory_gb=140, domain="modal") for n in nodes],
        NEEDS, record=rec, clock=clock, sleep=lambda s: threading.Event().wait(0.002), poll_s=0.0)
    s1 = S1Scheduler(S1Config(a_hard=1, start_seconds=5.0, fixed_start_seconds=True),
                     start_events=islands.start_events, record=rec, clock=clock)
    lifecycle = RemoteTrainerLifecycle(islands, files, poll_s=0.0, clock=clock,
                                       sleep=lambda s: threading.Event().wait(0.002))

    def watermark():
        cut = cuts.latest()
        return max(0 if cut is None or cut.watermark is None else cut.watermark,
                   lifecycle.running_watermark() or 0)

    head_buffer = FileSampleBuffer(store, batch_size=BATCH)
    backlog = MergedBacklog(files, tuple(nodes), CommittedBacklog(head_buffer, watermark),
                            max_policy_age=100)
    loop = TrainerLoop(scheduler=ConvertibleGate(s1, islands), lifecycle=lifecycle, buffer=backlog,
                       cuts=cuts, record=rec, clock=clock)
    threads = [threading.Thread(target=n.agent.run, daemon=True) for n in nodes.values()]
    for t in threads:
        t.start()
    _wait(lambda: all(files.read(node_path(n, "status.json")) for n in nodes))

    def give(node, rounds):
        target = node.agent.rounds + rounds
        with node.lock:
            node.permits += rounds
        _wait(lambda: node.agent.rounds == target)

    def steer(prefer):
        """Fewer in-flight requests rank first (selection level 4)."""
        for name, node in nodes.items():
            node.inflight = 0 if name == prefer else 5
        _wait(lambda: all(json.loads(files.read(node_path(n, "status.json")))["inflight"]
                          == nodes[n].inflight for n in nodes))

    order = ["n1", "n0", "n1"]
    for wave, chosen in enumerate(order):
        steer(chosen)
        for node in nodes.values():
            give(node, 1)  # 2 batches in all: Q_on = A_hard + 1 = 2
        decision = loop.tick()
        assert isinstance(decision, Start), (wave, decision)
        assert nodes[chosen].agent.role == TRAINER
        other = nodes["n0" if chosen == "n1" else "n1"]
        assert other.agent.role == ROLLOUT  # never both: the last rollout node stays
        give(other, 1)  # the other node generates while this one trains
        for _ in range(10):
            if backlog.backlog(loop.current_version()).batches < 1:
                break
            loop.tick()
        assert backlog.backlog(loop.current_version()).batches < 1
        assert other.agent.role == ROLLOUT
        loop.tick()  # idle begins
        clock.t += 10.0  # idle >= T_s = 5 s
        assert isinstance(loop.tick(), Stop)
        _wait(lambda: nodes[chosen].agent.role == ROLLOUT)
    islands.finish()
    for t in threads:
        t.join(10)
    assert all(n.agent.role == EXITED for n in nodes.values())

    # three round trips, in the steered order, each back with the published checksum
    trips = [(e["node"], e["direction"], e.get("version"), e.get("weight_checksum"))
             for n in nodes for e in _node_events(files, n) if e["event"] == "node_conversion"]
    backs = sorted((e for e in trips if e[1] == "to_rollout"), key=lambda e: e[2])
    assert [b[0] for b in backs] == order
    published = sorted(p for n in nodes.values() for p in n.published)
    assert [(b[2], b[3]) for b in backs] == published and all(b[3] for b in backs)
    assert sum(1 for t in trips if t[1] == "to_trainer") == 3
    # hand-over: every start restored the newest cut, the reader reset to its watermark
    assert [r[0] is None for r in cuts.restores] == [True, False, False]
    assert [w for _, w in cuts.restores] == sorted(w for _, w in cuts.restores)
    # every generated sample trained exactly once, none lost
    generated = [sid for n in nodes.values() for sid in n.generated]
    trained = [sid for n in nodes.values() for sid in n.trained()]
    assert len(trained) == len(set(trained)), "a sample was trained twice"
    assert sorted(trained) == sorted(generated), "a sample was lost"
    assert len(generated) == 3 * 3 * BATCH
    assert cuts.latest().watermark == len(generated)


# ---------------------------------------------------------------- agentic path at a conversion
class _Status(enum.Enum):
    COMPLETED = "completed"
    ABORTED = "aborted"


class _Sample:
    """Stand-in for Miles' ``Sample``: the fields the reward hook and the writer read."""

    Status = _Status

    def __init__(self, data, outcome):
        self.data = data
        self.group_index = data["group_index"]
        self.status = _Status.COMPLETED
        self.metadata = {OUTCOME_KEY: outcome, "reward": 1.0}

    def to_dict(self):
        return dict(self.data)


def _plugin_with_stub_miles(monkeypatch, events):
    """``buffer_rollout_miles`` against stand-ins for the Miles modules it imports
    (the base rollout function has Miles' async ``dispose``)."""
    import importlib
    import sys
    import types

    class _Base:
        def __init__(self, input):
            self.constructor_input = input

        async def dispose(self):
            events.append("miles_dispose")

    stubs = {
        "miles.rollout.base_types": dict(BaseRolloutFn=_Base, RolloutFnConstructorInput=object,
                                         RolloutFnInput=object, RolloutFnTrainOutput=object),
        "miles.rollout.inference_rollout.inference_rollout_common": dict(InferenceRolloutFn=_Base),
        "miles.utils.function_registry": dict(load_function=lambda path: None),
        "miles.utils.types": dict(Sample=_Sample),
    }
    for name, attrs in stubs.items():
        monkeypatch.setitem(sys.modules, name, types.SimpleNamespace(**attrs))
    plugin = "yeto.rl.adapters.miles.buffer_rollout_miles"
    monkeypatch.setitem(sys.modules, plugin, None)  # teardown restores the state before the test
    del sys.modules[plugin]
    return importlib.import_module(plugin)


def test_agentic_trial_cut_at_a_conversion_is_never_trained_and_its_vm_is_deleted(tmp_path, monkeypatch):
    """A convertible node's agentic round ends with two groups: one whose trial
    finished (its VM went back to the boat pool) and one whose trial was cut
    (outcome ABORTED, e.g. its VM was lost while it waited through the trainer
    stint), plus one trial still suspended (VM leased). The reward hook marks the
    cut group ABORTED and the writer leaves it out of the shard, so the trainer
    never sees it. The node converts and comes back; at the island's end the
    rollout function's dispose deletes every VM of the pool (fake boat client:
    no sandbox is created)."""
    import asyncio
    import weakref

    from buffer_rollout_procs import sample
    from test_boat_pool import SPEC_A, SPEC_B, VmFake, _pool

    from yeto.rl.adapters.miles import buffer_rollout as br
    from yeto.rl.adapters.miles.harbor_agent import reward_func
    from yeto.rl.agentic.types import Outcome
    from yeto.rl.harness.harbor import boat_pool

    events: list[str] = []
    plugin = _plugin_with_stub_miles(monkeypatch, events)
    monkeypatch.setattr(boat_pool, "_exit_pools", weakref.WeakSet())
    fake = VmFake()

    async def trials():
        pool = _pool(fake)
        boat_pool._exit_pools.add(pool)  # what process_pool registers (no signal hooks here)
        done = await pool.acquire(SPEC_A, name="trial-ok", machine_type="small")
        cut = await pool.acquire(SPEC_B, name="trial-cut", machine_type="small")
        suspended = await pool.acquire(SPEC_A, name="trial-suspended", machine_type="small")
        vms = (done.sandbox_id, cut.sandbox_id, suspended.sandbox_id)
        await pool.release(done)  # the finished trial gave its VM back (idle, reusable)
        await pool.release(cut, reusable=False)  # its sandbox failed: deleted at once
        return pool, vms

    pool, (vm_ok, vm_cut, vm_suspended) = asyncio.run(trials())
    assert vm_cut in fake.deleted and {vm_ok, vm_suspended} <= set(fake.live)

    ok = [_Sample(sample(0, 0, 3), Outcome.OK.value), _Sample(sample(1, 0, 3), Outcome.OK.value)]
    cut = [_Sample(sample(2, 1, 3), Outcome.OK.value), _Sample(sample(3, 1, 3), Outcome.ABORTED.value)]
    rewards = asyncio.run(reward_func(None, ok + cut))
    assert rewards == [1.0, 1.0, 1.0, 0.0] and cut[1].status is _Status.ABORTED

    store = CheckpointStore(str(tmp_path / "store"))
    fn = plugin.ConvertibleRolloutFn.__new__(plugin.ConvertibleRolloutFn)
    shard = FileSampleBuffer(store, batch_size=1, index=shard_index("island0"))
    fn._rounds = br.SamplerRoundWriter(shard, "island0", incarnation="i0", resume=False)
    fn._write(5, [ok, cut], cursor=None)
    head = FileSampleBuffer(store, batch_size=1)
    from yeto.rl.engine.sample_buffer import merge_shards

    assert merge_shards(LocalStoreFiles(store), ["island0"]) == 1
    head.refresh()
    assert [e.sample_id for e in head.entries()] == ["island0-i0-r5-g0"]  # the cut group is not there

    # the conversion and the way back do not touch the trials: the suspended one keeps its VM
    br.executor_set_mode(None, br.TRAINER)
    br.executor_set_mode(None, br.SAMPLER)
    assert vm_suspended in fake.live

    asyncio.run(fn.dispose())  # the island ends: Miles' RolloutExecutor.dispose()
    assert events == ["miles_dispose"]
    assert {vm_ok, vm_cut, vm_suspended} <= set(fake.deleted) and not fake.live
    assert pool.close_now() == 0


def test_buffer_rollout_dispose_without_a_boat_pool_imports_nothing(monkeypatch):
    """A math island never imported the boat pool: dispose runs Miles' own and
    does not import the Harbor package."""
    import asyncio
    import sys

    events: list[str] = []
    plugin = _plugin_with_stub_miles(monkeypatch, events)
    monkeypatch.delitem(sys.modules, plugin.BOAT_POOL_MODULE, raising=False)
    fn = plugin.ConvertibleRolloutFn.__new__(plugin.ConvertibleRolloutFn)
    asyncio.run(fn.dispose())
    assert events == ["miles_dispose"] and plugin.BOAT_POOL_MODULE not in sys.modules


# ---------------------------------------------------------------- prompts: one data slot per node
def test_pool_nodes_draw_disjoint_data_slots(_no_modal_listing, _fake_sky):
    """Without a slot every node of a pool reads the same rows in the same order
    (Miles' data source starts at row 0 and does not shuffle by default), so the
    trainer trains each prompt once per node. Each node now draws its own slot,
    as sampler islands do: the island command carries it, the island exports it
    to the rollout process, and the slots split the rows."""
    import types

    from test_rl_convertible_pool_launch import _args, _flag, _island_tokens

    from yeto.launcher import dry_run_plan
    from yeto.rl.adapters.miles import island_entry
    from yeto.rl.adapters.miles.buffer_rollout import DATA_SLOT_ENV
    from yeto.rl.engine.data_split import DataSlot

    plan = dry_run_plan(_args())
    slots = []
    for m, island in enumerate(plan["island_requests"]):
        tokens = _island_tokens(island["learner_command"])
        assert _flag(tokens, "--rl-sampler-data-slot") == f"{m}/2"
        env = {}
        island_entry.apply_sample_role(island_entry.parse_args(tokens), types.SimpleNamespace(), env)
        slots.append(DataSlot.parse(env[DATA_SLOT_ENV]))
    rows = list(range(10))
    drawn = [set(slot.rows(rows)) for slot in slots]
    assert not drawn[0] & drawn[1] and drawn[0] | drawn[1] == set(rows)


# ---------------------------------------------------------------- parked trainer sees its commands
class _Volume:
    """What an island sees of a Modal Volume: reads come from its mount as of the
    last reload; a reload is skipped while a writer guard is held in the container
    (``ModalVolumeStore.reload``)."""

    def __init__(self):
        self.remote: dict[str, bytes] = {}
        self.mount: dict[str, bytes] = {}
        self.guards = 0

    def hold_writer(self):
        self.guards += 1
        volume = self

        class Guard:
            def release(self):
                volume.guards -= 1

        return Guard()

    def read(self, rel):
        if self.guards == 0:
            self.mount = dict(self.remote)
        return self.mount.get(rel)


class _CutSession:
    """The driver's sync session with the E1 cut preparation: a prepared cut holds
    the writer guard until its safe point."""

    def __init__(self, volume):
        self.volume = volume
        self.prepared = None

    def prepare_cut(self, driver, *, rollout_id, state):
        self.prepared = (int(rollout_id), self.volume.hold_writer())

    def published(self, driver, *, rollout_id, policy_hash):
        pass


class _Driver:
    def __init__(self):
        self.events = []

    def emit(self, event, **fields):
        self.events.append((event, fields))


def test_a_trainer_parked_after_its_publication_holds_no_store_writer_guard():
    """GPU run s27-eg-a1: the convertible node's trainer trained its round, prepared
    the next cut (E1: before the colocated trainer offloads) and parked after the
    publication, waiting for the head's next ``run``. The prepared cut held the
    Volume writer guard, so every reload was skipped and the command never became
    visible: both sides waited. The lifecycle now skips the preparation when the
    driver will park; when it trains on, the preparation runs as before."""
    from yeto.rl.engine.trainer_lifecycle import StartPlan, TrainerHandle, _LifecycleSync

    volume, driver = _Volume(), _Driver()
    command = "trainer-control/inc-000000.jsonl"

    # before: the inner session alone keeps the guard through the park
    inner = _CutSession(volume)
    inner.prepare_cut(driver, rollout_id=1, state=None)
    volume.remote[command] = b'{"seq": 2, "verb": "run"}\n'
    assert volume.read(command) is None  # the parked trainer cannot see its command
    inner.prepared[1].release()

    # the lifecycle: the round reached its target, the driver will park -> no preparation
    volume.remote.clear()
    volume.mount.clear()
    inner = _CutSession(volume)
    handle = TrainerHandle(StartPlan(0), resume_from=None, clock=lambda: 0.0)
    handle._target_rounds, handle.rounds = 1, 1
    sync = _LifecycleSync(inner, handle, membership=None)
    sync.prepare_cut(driver, rollout_id=1, state=None)
    assert inner.prepared is None and volume.guards == 0
    assert driver.events == [("rl_cut_prepare_skipped", {"rollout_id": 1, "reason": "park"})]
    volume.remote[command] = b'{"seq": 2, "verb": "run"}\n'
    assert volume.read(command) == b'{"seq": 2, "verb": "run"}\n'

    # more rounds commanded: the driver trains on without a park, so it prepares
    handle._target_rounds = 3
    sync.prepare_cut(driver, rollout_id=2, state=None)
    assert inner.prepared is not None and inner.prepared[0] == 2
