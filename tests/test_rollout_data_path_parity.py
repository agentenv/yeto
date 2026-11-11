"""The rollout runtime's fixes hold on both data paths (``data_path`` r2 and syncer).

Every test runs twice: once on the r2 data path (``R2Release`` over a local
store, ``R2Buffer``) and once on the syncer data path (``SyncerBroadcast`` over
a stand-in subscriber, ``SyncerRelay`` over a stand-in relay producer). Real
pieces: the policy follower, the post-load checks, ``build_engine_sampler``,
``AgenticRound``, ``EngineRound``, ``GroupWriter``, the engine sampler entry.
The live syncer end to end run is ``test_spot_rollout_e2e``.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from yeto.rl.adapters.sampler_engines.fake import FakeEngine
from yeto.rl.agentic import TokenTurn, TrialVerdict
from yeto.rl.engine import engine_sampler_entry as entry
from yeto.rl.engine.agentic_round import DRAIN_EVENT, AgenticRound
from yeto.rl.engine.buffer_payload import decode_group
from yeto.rl.engine.engine_sampler import EngineSamplerConfig, PromptItem, build_engine_sampler
from yeto.rl.engine.policy_release import PolicyFollower
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.rollout_data_path import R2, SYNCER, DataPath, SampleSink, WeightSource, r2_classes
from yeto.rl.engine.sample_buffer import FileSampleBuffer
from yeto.rl.engine.sampler_checks import PostLoadChecks
from yeto.rl.engine.sampler_engine import SamplingParams
from yeto.rl.engine.task_dispense import DISPENSE_POSITION_KEY
from yeto.rl.spot_rollout.relay import PUT_STORED, Depth, PutAck
from yeto.rl.spot_rollout.relay_buffer import decode_item
from yeto.rl.spot_rollout.subscriber import WeightCut
from yeto.rl.spot_rollout.syncer_path import SyncerBroadcast, SyncerRelay

PATHS = (R2, SYNCER)
TASK = {"instance_id": "fix-git", "agent_name": "codex"}
N = 2
CHECKS = dict(expected_expert_matrices=None, require_fixed_answers=False)


# ----------------------------------------------------------------- the two paths

class StubFollower:
    """The ``WeightFollower`` surface: the newest cut of one fragment."""

    def __init__(self) -> None:
        self._cut: WeightCut | None = None

    @property
    def versions(self) -> tuple[int, ...]:
        return () if self._cut is None else self._cut.versions

    def drain(self) -> None:
        pass

    def cut(self) -> WeightCut | None:
        return self._cut


class StubProducer:
    """The ``RelayProducer`` surface; the trainer takes every item at once."""

    def __init__(self) -> None:
        self.items: list[bytes] = []
        self.should_stop = lambda: False
        self._lock = threading.Lock()

    def _depth(self) -> Depth:
        return Depth(items=0, bytes=0, acked_through=len(self.items), next_id=len(self.items) + 1)

    def put_blocking(self, data: bytes, *, should_stop=None, wait_s: float = 0.0) -> PutAck:
        with self._lock:
            self.items.append(bytes(data))
            return PutAck(seq=len(self.items), status=PUT_STORED, item_id=len(self.items), depth=self._depth())

    def depth(self) -> Depth:
        return self._depth()

    def close(self) -> None:
        pass


class Path_:
    """One data path under test: its weights, samples, and how a test publishes and reads."""

    def __init__(self, kind: str, tmp_path: Path) -> None:
        self.kind = kind
        self.root = tmp_path / kind
        self.store = CheckpointStore(self.root / "store")
        if kind == R2:
            R2Release, R2Buffer = r2_classes()
            self.weights = R2Release(self.store)
            self.samples = R2Buffer(self.store, batch_size=1)
        else:
            base = self.root / "base"
            base.mkdir(parents=True)
            (base / "model.safetensors").write_bytes(b"base weights")
            self.follower = StubFollower()
            self.producer = StubProducer()

            def materialize(cut: WeightCut) -> Path:
                out = self.root / "weights" / f"v{cut.merged_version}"
                out.mkdir(parents=True, exist_ok=True)
                (out / "model.safetensors").write_bytes(cut.fragments[0])
                return out

            self.weights = SyncerBroadcast(self.follower, materialize, base_dir=base)
            self.samples = SyncerRelay(self.producer, fragment_versions=self.weights.fragment_versions,
                                       cut_of=self.weights.cut_identity)

    def publish(self, version: int) -> None:
        if self.kind == R2:
            self.weights.release(version, f"h{version}", lambda d: (d / "model.safetensors").write_text("w"))
        else:
            self.follower._cut = WeightCut(versions=(int(version),), fragments=(b"w%d" % version,))

    def published(self) -> bool:
        return self.weights.latest() is not None

    def groups(self) -> dict[str, tuple[int, dict | None, list[dict]]]:
        """sample id -> (behavior version, provenance, samples) of every group written."""
        out = {}
        if self.kind == R2:
            reader = FileSampleBuffer(self.store, batch_size=1)
            reader.refresh()
            for s in reader.select("t", 1000):
                e = s.entry
                out[e.sample_id] = (e.behavior_version, e.provenance, decode_group(reader.read_payload(e.payload_ref)))
        else:
            for item in self.producer.items:
                header, payload = decode_item(item)
                out[header["sample_id"]] = (header["behavior_version"], header["provenance"], decode_group(payload))
        return out


@pytest.fixture(params=PATHS)
def path(request, tmp_path) -> Path_:
    return Path_(request.param, tmp_path)


# ------------------------------------------------------------------ fakes

class Recorder:
    def __init__(self, engine: FakeEngine) -> None:
        self.engine = engine
        self.opened: dict[str, int | None] = {}

    async def open(self, trajectory_id, sampling_params):
        assert trajectory_id not in self.opened, f"{trajectory_id} already open"
        self.opened[trajectory_id] = self.engine.served_version()
        return f"http://unused/{trajectory_id}"

    async def close(self, trajectory_id):
        first, last = self.opened.pop(trajectory_id), self.engine.served_version()
        return [TokenTurn(prompt_ids=(1, 2), completion_ids=(3, 4), completion_logprobs=(-0.1, -0.2),
                          policy_version=first),
                TokenTurn(prompt_ids=(1, 2, 3, 4, 5), completion_ids=(6, 7), completion_logprobs=(-0.3, -0.4),
                          policy_version=last)]


class Harness:
    def __init__(self, delays: dict[str, float], on_start=lambda prompt: None) -> None:
        self.delays, self.on_start = dict(delays), on_start
        self.started: list[str] = []

    async def run(self, task, endpoint):
        prompt = next(p for p in self.delays if f"-{p}-" in task.trajectory_id)
        self.started.append(task.trajectory_id)
        self.on_start(prompt)
        await asyncio.sleep(self.delays[prompt])
        return TrialVerdict(rewards={"reward": 1.0})


class Prompts:
    def __init__(self, rounds: dict[int, list[str]], wait_for=None) -> None:
        self.rounds, self.wait_for = rounds, wait_for

    def __call__(self, rollout_id):
        if rollout_id and self.wait_for is not None:
            _wait(self.wait_for)
        return [PromptItem(p, f"task {p}", metadata={"agent_task": dict(TASK)}) for p in self.rounds.get(rollout_id, [])]

    def cursor(self, rollout_id):
        return {"row_offset": int(rollout_id) + 1}


def _wait(predicate, timeout=30.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.005)


def _island(path: Path_, prompts, *, harness=None, engine=None, num_rollouts=3, live=False, **config):
    engine = engine or FakeEngine()
    if live:
        engine.updates_during_generation = True
    events: list[tuple[str, dict]] = []
    cfg = EngineSamplerConfig(sampler_id="s", num_rollouts=num_rollouts, max_backlog_groups=100,
                              sampling=SamplingParams(max_tokens=8, n=N), max_prompt_tokens=1024, poll_s=0.0,
                              **config)
    make = None
    if harness is not None:
        def make(engine_, writer, **kw):
            return AgenticRound(engine_, writer, recorder=Recorder(engine_), harness=harness, max_in_flight=4,
                                max_response_len=64, min_samples=1, task_defaults={"provider": "docker"},
                                release_poll_s=0.0, wait_s=0.005, weight_switch="live" if live else "drain", **kw)
    island = build_engine_sampler(cfg, engine=engine, releases=path.weights, buffer=path.samples,
                                  emit=lambda n, **f: events.append((n, f)), prompts=prompts,
                                  reward=lambda item, c: 0.5, sleep=lambda s: None, make_round=make,
                                  checks=PostLoadChecks(**CHECKS))
    return island, events


# ------------------------------------------------------------------ interfaces

def test_both_paths_implement_the_interfaces(path):
    assert isinstance(path.weights, WeightSource) and isinstance(path.samples, SampleSink)
    assert path.weights.kind == path.samples.kind == path.kind
    assert path.weights.behind(None) is False and path.samples.shard_entries("other") == []
    assert path.weights.latest() is None
    path.publish(2)
    release = path.weights.latest()
    assert release.version == 2 and release.seq >= 1
    assert (Path(path.weights.resolve(release)) / "model.safetensors").exists()


# ------------------------------------------------------ forced reload bound

class _AlwaysInFlight:
    def holds_weights(self) -> bool:
        return True


def test_a_deferred_release_is_force_loaded_after_reload_max_defer_s(path):
    from yeto.rl.engine.sampler_loop import SamplerLoop

    loaded, now, events = [], {"t": 0.0}, []
    follower = PolicyFollower(path.weights, lambda release, directory: loaded.append(release.version),
                              timer=lambda: now["t"], max_defer_s=600.0)
    follower.hold = _AlwaysInFlight().holds_weights
    loop = SamplerLoop(follower=follower, buffer=None, generate=lambda rid: None, num_rollouts=1,
                       max_backlog_groups=1, emit=lambda n, **f: events.append((n, f)), sleep=lambda s: None)
    path.publish(1)
    for t in (0.0, 599.9):
        now["t"] = t
        loop._load_newest()
    assert not loaded and follower.counters.deferred_in_flight == 2
    now["t"] = 600.0
    loop._load_newest()
    assert loaded == [1] and follower.counters.forced_loads == 1
    (event,) = [f for n, f in events if n == "rl_sampler_policy_loaded"]
    assert event["forced"] is True and event["deferred_s"] == 600.0


def test_the_island_follower_gets_the_spec_bound(path):
    island, _ = _island(path, Prompts({}), num_rollouts=0, reload_max_defer_s=42.0)
    assert island.follower.max_defer_s == 42.0 and island.follower.releases is path.weights


# ------------------------------------------------------ drain / live switch

def test_drain_holds_the_release_until_the_running_groups_end(path):
    def publish(prompt):
        if prompt == "a" and not path.published():
            path.publish(1)

    harness = Harness({"a": 0.2, "b": 0.0, "c": 0.0}, on_start=publish)
    island, events = _island(path, Prompts({0: ["a"], 1: ["b"], 2: ["c"]}, wait_for=path.published),
                             harness=harness)
    assert island.follower.hold == island.round.holds_weights and island.follower.live_update is False
    try:
        assert island.run() == 3
    finally:
        island.round.close()
    versions = {k.rsplit("-", 2)[-2] + "-" + k.rsplit("-", 1)[-1]: v for k, (v, _, _) in path.groups().items()}
    assert versions == {"r0-a": 0, "r1-b": 1, "r2-c": 1}
    names = [n for n, _ in events]
    assert names.index(DRAIN_EVENT) < names.index("rl_sampler_policy_loaded")


def test_live_loads_during_a_running_trial_and_keeps_its_oldest_version(path):
    def publish(prompt):
        if prompt == "a" and not path.published():
            path.publish(1)

    harness = Harness({"a": 0.3}, on_start=publish)
    island, events = _island(path, Prompts({0: ["a"]}), harness=harness, num_rollouts=1, live=True,
                             live_poll_s=0.01)
    assert island.follower.live_update is True
    try:
        island.run()
    finally:
        island.round.close()
    loaded = [f for n, f in events if n == "rl_sampler_policy_loaded"]
    assert loaded and loaded[0]["policy_version"] == 1 and loaded[0].get("in_flight") is True
    ((version, _, _),) = path.groups().values()
    assert version == 0  # the trial opened on v0: the group's behavior version is its oldest token's


# ------------------------------------------- recorder sessions / group keys

def test_a_row_dispensed_twice_in_a_round_gets_its_own_sessions_and_group(path):
    class Twice(Prompts):
        def __call__(self, rollout_id):
            return [PromptItem("row7", "task row7", metadata={"agent_task": dict(TASK), DISPENSE_POSITION_KEY: s})
                    for s in (74, 75)]

    harness = Harness({"row7": 0.0})
    island, _ = _island(path, Twice({}), harness=harness, num_rollouts=1)
    island.round.incarnation = "132f9179"
    try:
        island.run()
    finally:
        island.round.close()
    assert len(set(harness.started)) == len(harness.started) == 2 * N  # Recorder asserts no session reopens
    assert sorted(path.groups()) == ["s-132f9179-r0-row7", "s-132f9179-r0-row7-p75"]


# -------------------------------------------- xhw source (group provenance)

def test_every_group_names_its_source_for_the_xhw_guard(path):
    from yeto.rl.engine.xhw_guard import UNKNOWN, source_key

    island, _ = _island(path, lambda rid: [PromptItem("p0", "question 0"), PromptItem("p1", "question 1")],
                        num_rollouts=1)
    assert island.run() == 1
    groups = path.groups()
    assert len(groups) == 2
    keys = {source_key(provenance, "s") for _, provenance, _ in groups.values()}
    assert len(keys) == 1 and UNKNOWN not in next(iter(keys)).split("|")[:2]
    assert all(provenance["sampler_id"] == "s" for _, provenance, _ in groups.values())


# ------------------------------------------- predecessor handoff and orphan sweep

class _Gate:
    def __init__(self) -> None:
        self.swept: list[str] = []

    def sweep_shard(self, shard: str) -> int:
        self.swept.append(shard)
        return 3


class _Prompts:
    def __init__(self) -> None:
        self.gate = _Gate()


def _data_path(path: Path_) -> DataPath:
    dp = DataPath(path.weights, path.samples, path.store)
    dp.kind = path.kind
    dp.shares_member_store = path.kind == R2
    return dp


def test_a_replacement_sweeps_a_gone_predecessor_s_orphans(path):
    from yeto.rl.engine.sampler_resume import SamplerMembership, resume_point

    dp, prompts, events = _data_path(path), _Prompts(), []
    spec: dict[str, Any] = {"handoff_wait_s": 0.01}
    if path.kind == R2:  # the predecessor's LEAVE is in the shared member store
        member = SamplerMembership(path.store, "old", "inc0", resume_point([], "old"), lambda *a, **k: None)
        member.join(tape="t")
        member.leave("FENCED")
    else:  # the pool retired it on the syncer and says so in the spec
        spec["predecessor_retired"] = True
    started = time.monotonic()
    entry.continue_predecessor(dp, spec, prompts, "old", "new", lambda n, **f: events.append((n, f)))
    assert time.monotonic() - started < 5.0
    assert prompts.gate.swept == ["old"]
    assert ("rl_sandbox_orphans_removed", "old") in [(n, f.get("shard")) for n, f in events]


def test_a_predecessor_not_known_gone_is_left_alone(path):
    dp, prompts, events = _data_path(path), _Prompts(), []
    entry.continue_predecessor(dp, {"handoff_wait_s": 0.01}, prompts, "old", "new",
                               lambda n, **f: events.append((n, f)))
    assert prompts.gate.swept == []
    (handoff,) = [f for n, f in events if n == "rl_sampler_handoff"]
    assert handoff["predecessor"] == "old" and handoff["predecessor_left"] is False


def test_the_sweep_needs_no_store_dispenser(path, monkeypatch):
    """The syncer path's task dispenser has no sandbox gate: the spec's agentic.envs gives one."""
    gate = _Gate()
    monkeypatch.setattr(entry, "sweep_gate", lambda spec, environ=None: gate)
    spec = {"agentic": {"envs": {"hosts": []}}, "predecessor_retired": True, "handoff_wait_s": 0.01}
    entry.continue_predecessor(_data_path(path), spec, object(), "old", "new", lambda n, **f: None)
    assert gate.swept == ["old"]


# ------------------------------------------------------- the entry on both paths

def _spec(tmp_path: Path, kind: str, **over) -> dict:
    from tests.test_stage1_gaps import _sampler_spec

    spec = _sampler_spec(tmp_path, num_rollouts=2, data_path=kind)
    if kind == SYNCER:
        spec.update(syncer="127.0.0.1:1", channel=0, work_dir=str(tmp_path / "work"))
        spec.pop("store")
    return {**spec, **over}


class StubSyncerPath(DataPath):
    kind = SYNCER
    shares_member_store = False

    def __init__(self, path: Path_, spec) -> None:
        super().__init__(path.weights, path.samples, CheckpointStore(Path(spec["store"])))
        self.fenced = False
        self.closed = False

    def stopping(self) -> bool:
        return self.fenced or len(self.samples) >= 4

    def stop_reason(self):
        from yeto.rl.pool_client import LeaveReason

        return (LeaveReason.FENCED, {"fenced": "superseded"}) if self.fenced else None

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def entry_path(path, monkeypatch):
    opened: list[StubSyncerPath] = []
    if path.kind == SYNCER:
        from yeto.rl.spot_rollout import syncer_path

        def open_syncer(spec, environ=None, *, incarnation, emit):
            opened.append(StubSyncerPath(path, spec))
            return opened[-1]

        monkeypatch.setattr(syncer_path, "open_syncer", open_syncer)
    return path, opened


def _tape(tmp_path: Path) -> list[dict]:
    [tape] = (tmp_path / "tape").glob("events.*.jsonl")
    return [json.loads(line) for line in tape.read_text().splitlines()]


def test_the_entry_runs_the_same_island_on_both_paths(entry_path, tmp_path):
    path, opened = entry_path
    seen: list[Any] = []
    entry.run(_spec(tmp_path, path.kind, reload_max_defer_s=33), resource_sampler=lambda emit: None,
              incarnation="aaaa0001", on_data_path=seen.append)
    (dp,) = seen
    assert dp.kind == path.kind
    groups = (path.groups() if path.kind == SYNCER
              else Path_.groups(type("P", (), {"kind": R2, "store": CheckpointStore(tmp_path / "store")})()))
    assert len(groups) == 4 and all("-aaaa0001-" in k for k in groups)
    names = [e["event"] for e in _tape(tmp_path)]
    assert "rl_sampler_join" in names and names[-1] == "rl_learner_finalized"
    (leave,) = [e for e in _tape(tmp_path) if e["event"] == "rl_sampler_leave"]
    assert leave["reason"] == "FINISHED"
    if path.kind == SYNCER:
        assert opened[0].closed  # the runtime closes its data path


def test_a_fence_of_the_data_path_leaves_fenced_on_both_paths(entry_path, tmp_path):
    from yeto.rl.engine.incarnation_lease import IncarnationLease

    path, opened = entry_path
    spec = _spec(tmp_path, path.kind)
    seen: list[Any] = []

    def fence(event, **fields):
        if event != "rl_sampler_round":
            return
        if path.kind == SYNCER:
            opened[0].fenced = True  # the syncer member table superseded this incarnation
        else:
            IncarnationLease(seen[0].member_store, "ppu4", "bbbb0002").claim([])  # a newer incarnation

    entry.run({**spec, "num_rollouts": 50, "incarnation_lease": {"poll_s": 0.01, "write_check_s": 0.0}},
              resource_sampler=lambda emit: None, incarnation="aaaa0001", on_event=fence,
              on_data_path=seen.append)
    (leave,) = [e for e in _tape(tmp_path) if e["event"] == "rl_sampler_leave"]
    assert leave["reason"] == "FENCED"


def test_the_spot_entry_is_a_shell_over_the_engine_entry(monkeypatch):
    from yeto.rl.spot_rollout import sampler

    calls = {}

    def fake_run(spec, **kw):
        calls.update(spec=spec, **kw)
        return 0

    monkeypatch.setattr(entry, "run", fake_run)
    rounds, _ = sampler.run_spec({"sampler_id": "x"})
    assert rounds == 0 and calls["spec"]["data_path"] == SYNCER


def test_the_spot_main_uses_the_entry_s_reclaim_deadline(tmp_path, monkeypatch):
    from yeto.rl.spot_rollout import sampler

    calls = {}

    def fake_run(spec, **kw):
        calls.update(spec=spec, **kw)
        return 1

    monkeypatch.setattr(entry, "run", fake_run)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"sampler_id": "x"}))
    assert sampler.main(["--spec", str(spec)]) == 0
    assert calls["spec"]["data_path"] == SYNCER
    assert isinstance(calls["reclaim_deadline"], entry.ReclaimDeadline)
    spec.write_text(json.dumps({"sampler_id": "x", "data_path": R2}))
    sampler.main(["--spec", str(spec)])
    assert calls["spec"]["data_path"] == R2  # an explicit data path is kept


@pytest.mark.parametrize("spec, kind", [
    ({"store": "s"}, R2), ({"syncer": "h:1", "channel": 0, "work_dir": "w"}, SYNCER),
    ({"data_path": "r2", "store": "s", "syncer": "h:1"}, R2),
])
def test_the_data_path_comes_from_the_spec(spec, kind):
    from yeto.rl.engine.rollout_data_path import normalize_spec, resolve_data_path

    assert resolve_data_path(spec) == kind
    normal = normalize_spec(spec)
    assert normal["data_path"] == kind
    if kind == SYNCER:
        assert normal["store"].endswith("store") and normal["num_rollouts"] == 1 << 40


@pytest.mark.parametrize("spec", [{"data_path": "s3", "store": "s"}, {"data_path": "syncer", "store": "s"},
                                  {"data_path": "r2"}])
def test_a_bad_data_path_is_refused(spec):
    from yeto.rl.engine.rollout_data_path import resolve_data_path

    with pytest.raises(entry.SamplerSpecError):
        resolve_data_path(spec)


# ------------------------------------------- scale-down drain / pause --hold-tasks

@pytest.mark.parametrize("lift", [False, True], ids=["drain", "hold_then_resume"])
def test_a_drained_sampler_claims_nothing_new_writes_its_groups_and_answers(path, tmp_path, lift):
    """The pool's drain request on the store dispenser (a scale down; ``pool-pause
    --hold-tasks`` writes the same request): the island writes the groups it holds to
    its data path's sink, claims nothing more and answers ``drained``; once the
    request is lifted (``pool-resume``) it claims again."""
    from tests.test_store_dispense import Clock, FakeObjects, _open, _rows, _write
    from yeto.rl.engine import store_dispense as sd

    store = FakeObjects()
    rows = _write(tmp_path, _rows(6))
    clock = Clock(store)
    prompts = _open(rows, store, holder="nc-0", groups=2, clock=clock)
    other = _open(rows, store, holder="nc-1", groups=2)
    state: dict[str, Any] = {"answered": False}

    class Drained:
        written = staticmethod(prompts.written)

        def cursor(self, rollout_id):
            return {"row_offset": int(rollout_id)}

        def __call__(self, rollout_id):
            items = prompts(rollout_id)
            if "seq" not in state:  # the pool asks for a drain while round 0's rows are in flight
                state["seq"] = sd.request_drain(store, "run/dispense", "nc-0")
                for _ in range(1 if lift else 2):  # the other sampler takes the rest (all, or two)
                    spots = [it.metadata[DISPENSE_POSITION_KEY] for it in other(0)]
                    other.written(0, spots)
            return items

    def sleep(s):  # the pool reads the answer; hold_then_resume then lifts the request
        if not state["answered"] and sd.drain_answered(store, "run/dispense", "nc-0", state["seq"]):
            state["answered"] = True
            if lift:
                sd.request_drain(store, "run/dispense", "nc-0", on=False)
            else:  # a drained sampler waits; the pool then stops it (a fence / RETIRE)
                from yeto.rl.engine.task_dispense import DispenseEnded

                raise DispenseEnded("stopped by the pool after its drain answer")
        clock.sleep(s)

    prompts._sleep = sleep
    island, _ = _island(path, Drained(), num_rollouts=20)
    island.run()
    assert state["answered"] is True
    tasks = {t["slot"]: t for t in sd.status(store, "run/dispense")["tasks"]}
    assert len(tasks) == 6 and all(t["state"] == "trained" for t in tasks.values())
    mine = sorted(s for s, t in tasks.items() if t["holder"] == "nc-0")
    assert mine == ([0, 1, 4, 5] if lift else [0, 1])
    assert len(path.groups()) == len(mine)  # every group it held reached its sink, nothing else


# ------------------------------------------------ WorkContract on both paths

def test_a_group_the_dispenser_no_longer_holds_is_not_written(path):
    """``still_holds`` is read before every group write (WorkContract)."""
    class Held:
        def __call__(self, rollout_id):
            return [PromptItem(f"row{s}", f"question {s}", metadata={DISPENSE_POSITION_KEY: s}) for s in (0, 1)]

        def still_holds(self, slot):
            return slot != 1  # slot 1 was given out again (expired / retired)

        def written(self, rollout_id, positions, **kw):
            self.reported = list(positions)
            return {}

    prompts = Held()
    island, _ = _island(path, prompts, num_rollouts=1)
    island.run()
    assert [k.rsplit("-", 1)[-1] for k in path.groups()] == ["row0"] and prompts.reported == [0]
    assert island.round.dropped_groups == 1


@pytest.mark.parametrize("why", ["retired", "superseded", "store"])
def test_a_dispenser_fence_ends_the_incarnation_fenced_without_a_restart(entry_path, tmp_path, monkeypatch, why):
    """A member retired by the controller or superseded stops for good: the island
    leaves FENCED and ``run`` returns (a crash would make a supervisor restart it
    under a new incarnation, which the contract forbids)."""
    from yeto.rl.engine import task_dispense as td

    path, _ = entry_path
    error = {"retired": td.DispenseFenced(f"fenced {td.RETIRED_MARK}"),
             "superseded": td.DispenseFenced(f"fenced {td.SUPERSEDED_MARK}"),
             "store": td.DispenseFencedEnd("fenced: this incarnation takes no more tasks")}[why]
    real = entry.build_prompts

    def build_prompts(spec, environ=None):
        rows = real(spec, environ)

        def prompts(rollout_id):
            if rollout_id >= 1:
                raise error
            return rows(rollout_id)

        prompts.cursor = rows.cursor
        return prompts

    monkeypatch.setattr(entry, "build_prompts", build_prompts)
    rounds = entry.run(_spec(tmp_path, path.kind, num_rollouts=5), resource_sampler=lambda emit: None,
                       incarnation="aaaa0001")
    assert rounds == 1
    tape = _tape(tmp_path)
    (leave,) = [e for e in tape if e["event"] == "rl_sampler_leave"]
    assert leave["reason"] == "FENCED" and leave["fenced"] == {"store": "dispenser"}.get(why, why)
    assert "rl_sampler_dispense_fenced" in [e["event"] for e in tape]


def test_another_dispenser_error_still_fails_the_island(entry_path, tmp_path, monkeypatch):
    from yeto.rl.engine import task_dispense as td

    path, _ = entry_path

    def build_prompts(spec, environ=None):
        def prompts(rollout_id):
            raise td.DispenseFenced("lease expired")  # not final: the task source rejoins

        return prompts

    monkeypatch.setattr(entry, "build_prompts", build_prompts)
    with pytest.raises(td.DispenseFenced):
        entry.run(_spec(tmp_path, path.kind), resource_sampler=lambda emit: None, incarnation="aaaa0001")
    (leave,) = [e for e in _tape(tmp_path) if e["event"] == "rl_sampler_leave"]
    assert leave["reason"] == "INFRA_FAILED"
