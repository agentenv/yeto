"""S29: the agentic round streams groups instead of waiting for the round (CPU).

Real pieces: ``AgenticRound`` (the outer layer ``run_agent_task`` and
``finish_sample``, ``GroupWriter``, ``FileSampleBuffer``, ``RoundGuard`` /
``resume_point``), ``build_engine_sampler`` with ``SamplerLoop``,
``PolicyFollower`` and a ``PolicyReleaseStore``. Fakes: the engine
(``FakeEngine``), a token recorder that stamps each turn with the served
version at the trial's start and end, a harness that sleeps per task, and a
task dispenser that records its ``written`` calls.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from yeto.rl.adapters.sampler_engines.fake import FakeEngine
from yeto.rl.agentic import TokenTurn, TrialVerdict
from yeto.rl.engine.agentic_round import DRAIN_EVENT, GROUP_EVENT, LOST_DRAIN, AgenticRound
from yeto.rl.engine.buffer_payload import GroupWriter, decode_group
from yeto.rl.engine.engine_sampler import ROUND_EVENT, EngineSamplerConfig, PromptItem, build_engine_sampler
from yeto.rl.engine.policy_release import PolicyFollower, PolicyReleaseStore
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer
from yeto.rl.engine.sampler_checks import PostLoadChecks
from yeto.rl.engine.sampler_engine import Provenance, SamplingParams
from yeto.rl.engine.sampler_resume import SamplerResumeError, resume_point
from yeto.rl.engine.task_dispense import DISPENSE_POSITION_KEY, DispensedRows

TASK = {"instance_id": "fix-git", "agent_name": "codex"}
N = 2


class Recorder:
    """Two turns per trial: the served version when the trial opened and when it closed."""

    def __init__(self, engine: FakeEngine) -> None:
        self.engine = engine
        self.opened: dict[str, int | None] = {}

    async def open(self, trajectory_id, sampling_params):
        self.opened[trajectory_id] = self.engine.served_version()
        return f"http://unused/{trajectory_id}"

    async def close(self, trajectory_id):
        first, last = self.opened.pop(trajectory_id), self.engine.served_version()
        return [TokenTurn(prompt_ids=(1, 2), completion_ids=(3, 4), completion_logprobs=(-0.1, -0.2),
                          policy_version=first),
                TokenTurn(prompt_ids=(1, 2, 3, 4, 5), completion_ids=(6, 7), completion_logprobs=(-0.3, -0.4),
                          policy_version=last)]


class Harness:
    """Sleeps ``delays[prompt id]`` seconds (or until ``gates[prompt id]`` is set)."""

    def __init__(self, delays: dict[str, float] | None = None, *, on_start=None) -> None:
        self.delays = dict(delays or {})
        self.gates: dict[str, threading.Event] = {}
        self.started: list[str] = []
        self.on_start = on_start or (lambda prompt: None)

    async def run(self, task, endpoint):
        prompt = next(p for p in [*self.delays, *self.gates] if f"-{p}-" in task.trajectory_id)
        self.started.append(task.trajectory_id)
        self.on_start(prompt)
        gate = self.gates.get(prompt)
        if gate is not None:
            while not gate.is_set():
                await asyncio.sleep(0.005)
        else:
            await asyncio.sleep(self.delays.get(prompt, 0.0))
        return TrialVerdict(rewards={"reward": 1.0})


class Prompts:
    """``prompts(rollout_id)`` from a per-round list of prompt ids; with
    ``dispensed`` every row has a dispenser position and ``written`` is recorded."""

    def __init__(self, rounds: dict[int, list[str]], *, dispensed: bool = False) -> None:
        self.rounds = rounds
        self.positions: dict[str, int] = {}
        self.calls: list[tuple[int, list[int], str, list[int]]] = []
        if dispensed:
            self.written = self._written

    def __call__(self, rollout_id):
        items = []
        for prompt in self.rounds.get(rollout_id, []):
            meta: dict[str, Any] = {"agent_task": dict(TASK)}
            if hasattr(self, "written"):
                meta[DISPENSE_POSITION_KEY] = self.positions.setdefault(prompt, 100 + len(self.positions))
            items.append(PromptItem(prompt, f"task {prompt}", metadata=meta))
        return items

    def cursor(self, rollout_id):
        return {"row_offset": int(rollout_id) + 1}

    def _written(self, rollout_id, positions, *, unwritten="drop", keep=()):
        self.calls.append((rollout_id, list(positions), unwritten, sorted(keep)))
        return {"rollout_id": rollout_id, "trained": len(positions)}


def _round(tmp_path: Path, harness: Harness, prompts: Prompts, *, max_in_flight: int = 4, live: bool = False,
           incarnation: str | None = None, resume=None, reclaimed=lambda: False, engine=None, **kw):
    engine = engine or FakeEngine()
    if engine.served_version() is None:
        engine.start(None, 0)
    if live:
        engine.updates_during_generation = True
    events: list[tuple[str, dict]] = []
    sampling = SamplingParams(max_tokens=8, n=N)
    buffer = FileSampleBuffer(CheckpointStore(tmp_path), batch_size=1)
    rnd = AgenticRound(engine, GroupWriter(buffer, "s"), recorder=Recorder(engine), harness=harness,
                       max_in_flight=max_in_flight, max_response_len=64, min_samples=1,
                       task_defaults={"provider": "docker"}, prompts=prompts, sampling=sampling,
                       provenance=Provenance.of(engine.identity, sampler_id="s", sampling=sampling),
                       emit=lambda n, **f: events.append((n, f)), incarnation=incarnation, resume=resume,
                       data_cursor=prompts.cursor, reclaimed=reclaimed,
                       weight_switch="live" if live else "drain", release_poll_s=0.0, wait_s=0.005, **kw)
    return rnd, engine, events


def _entries(tmp_path: Path):
    reader = FileSampleBuffer(CheckpointStore(tmp_path), batch_size=1)
    reader.refresh()
    entries = [s.entry for s in reader.select("t", 100)]
    return entries, {e.sample_id: decode_group(reader.read_payload(e.payload_ref)) for e in entries}


def _wait(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.005)


# ---------------------------------------------------------------------- streaming
def test_a_fast_group_is_written_while_a_slow_group_of_the_same_round_still_runs(tmp_path):
    harness = Harness({"fast": 0.0})
    harness.gates["slow"] = threading.Event()
    rnd, engine, events = _round(tmp_path, harness, Prompts({0: ["slow", "fast"]}))
    try:
        rnd(0)  # returns once the fast group is written and a slot is free
        entries, _ = _entries(tmp_path)
        assert [e.sample_id for e in entries] == ["s-r0-fast"]
        (first,) = [f for n, f in events if n == ROUND_EVENT]
        assert first["groups"] == 1 and first["agentic"]["in_flight"] == N and rnd.holds_weights()
        harness.gates["slow"].set()
        rnd.flush()
    finally:
        rnd.close()
    entries, payloads = _entries(tmp_path)
    assert [e.sample_id for e in entries] == ["s-r0-fast", "s-r0-slow"]
    assert all(len(p) == N for p in payloads.values())
    groups = [f for n, f in events if n == GROUP_EVENT]
    assert [g["prompt_id"] for g in groups] == ["fast", "slow"]
    slow = groups[1]
    assert slow["written"] and len(slow["trial_s"]) == N and slow["group_s"] >= max(slow["trial_s"])
    assert 0 < slow["in_flight_utilization"] <= 1 and slow["version_lag"] == 0
    flushed = [f for n, f in events if n == ROUND_EVENT][-1]
    assert flushed["flush"] and flushed["groups"] == 1
    # the round event keeps the agentic fields the readers use
    agentic = flushed["agentic"]
    assert {"trials", "kept", "lost", "lost_rate", "groups_below_min", "outcomes", "reward_mean",
            "policy_tokens_mean", "lost_total", "trials_total"} <= set(agentic)
    assert agentic["trials_total"] == 2 * N


def test_the_next_round_is_drawn_while_an_earlier_one_runs(tmp_path):
    harness = Harness({"fast": 0.0})
    harness.gates["slow"] = threading.Event()
    rnd, engine, events = _round(tmp_path, harness, Prompts({0: ["slow"], 1: ["fast"]}))
    try:
        rnd(0)  # 2 of 4 slots taken: returns at once for the next round's tasks
        assert not _entries(tmp_path)[0]
        rnd(1)
        assert [e.sample_id for e in _entries(tmp_path)[0]] == ["s-r1-fast"]
        harness.gates["slow"].set()
        rnd.flush()
    finally:
        rnd.close()
    assert [e.sample_id for e in _entries(tmp_path)[0]] == ["s-r1-fast", "s-r0-slow"]


# ---------------------------------------------------------------------- drain
def test_drain_stops_new_groups_until_the_running_ones_end_and_keeps_one_version_per_group(tmp_path):
    harness = Harness({"b": 0.0, "c": 0.0})
    harness.gates["a"] = threading.Event()
    pending = {"seq": None}
    rnd, engine, events = _round(tmp_path, harness, Prompts({0: ["a"], 1: ["b"], 2: ["c"]}))
    rnd.release_pending = lambda: pending["seq"]
    try:
        rnd(0)
        pending["seq"] = 1  # a release appears while group a runs
        done = threading.Thread(target=rnd, args=(1,))
        done.start()
        time.sleep(0.05)
        assert done.is_alive() and not any("-b-" in s for s in harness.started)  # b waits for the drain
        harness.gates["a"].set()
        done.join(5)
        assert not done.is_alive()
        engine.load(Path("r1"), 1)  # what the sampler loop does between the calls
        pending["seq"] = None
        rnd(2)
        rnd.flush()
    finally:
        rnd.close()
    entries, payloads = _entries(tmp_path)
    versions = {e.sample_id: e.behavior_version for e in entries}
    assert versions == {"s-r0-a": 0, "s-r1-b": 1, "s-r2-c": 1}
    for samples in payloads.values():
        assert len({s["metadata"]["weight_version"] for s in samples}) == 1
        assert all("final_weight_version" not in s["metadata"] for s in samples)
    (drain,) = [f for n, f in events if n == DRAIN_EVENT]
    assert drain["seq"] == 1 and drain["waited_s"] > 0 and not drain["timed_out"]
    assert sum(f["agentic"]["drain_wait_s"] for n, f in events if n == ROUND_EVENT) > 0


def test_a_drain_past_its_timeout_cancels_the_running_trials(tmp_path):
    harness = Harness({"b": 0.0})
    harness.gates["a"] = threading.Event()  # never set: only the timeout ends the drain
    pending = {"seq": None}
    rnd, engine, events = _round(tmp_path, harness, Prompts({0: ["a"], 1: ["b"]}), drain_timeout_s=0.05)
    rnd.release_pending = lambda: pending["seq"]
    try:
        rnd(0)
        pending["seq"] = 3
        rnd(1)
        assert not rnd.holds_weights()
    finally:
        rnd.close()
    (drain,) = [f for n, f in events if n == DRAIN_EVENT]
    assert drain["timed_out"] and drain["abandoned_trials"] == N
    (group,) = [f for n, f in events if n == GROUP_EVENT]
    assert group["prompt_id"] == "a" and not group["written"] and group["lost"] == {LOST_DRAIN: N}
    assert not _entries(tmp_path)[0]


# ---------------------------------------------------------------------- live
def test_live_keeps_a_trial_that_spans_a_reload_at_its_oldest_version(tmp_path):
    harness = Harness({"b": 0.0})
    harness.gates["a"] = threading.Event()
    rnd, engine, events = _round(tmp_path, harness, Prompts({0: ["a"], 1: ["b"]}), live=True)
    rnd.release_pending = lambda: 9  # never drained for in live mode
    try:
        rnd(0)
        engine.load(Path("r1"), 1)  # the live loader updates the running engine
        rnd(1)
        harness.gates["a"].set()
        rnd.flush()
    finally:
        rnd.close()
    entries, payloads = _entries(tmp_path)
    versions = {e.sample_id: e.behavior_version for e in entries}
    assert versions == {"s-r1-b": 1, "s-r0-a": 0}
    for sample in payloads["s-r0-a"]:
        assert sample["metadata"]["weight_version"] == "0" and sample["metadata"]["final_weight_version"] == "1"
    assert not [n for n, _ in events if n == DRAIN_EVENT]
    group_a = [f for n, f in events if n == GROUP_EVENT and f["prompt_id"] == "a"][0]
    assert group_a["kept"] == N and group_a["versions"] == [0, 1] and group_a["version_lag"] == 1


def test_live_needs_an_engine_that_updates_during_a_generation(tmp_path):
    with pytest.raises(ValueError, match="updates its weights during a generation"):
        engine = FakeEngine()
        engine.start(None, 0)
        sampling = SamplingParams(max_tokens=8, n=N)
        AgenticRound(engine, GroupWriter(FileSampleBuffer(CheckpointStore(tmp_path), batch_size=1), "s"),
                     recorder=None, harness=None, max_in_flight=1, max_response_len=8, min_samples=1,
                     prompts=Prompts({}), sampling=sampling, weight_switch="live",
                     provenance=Provenance.of(engine.identity, sampler_id="s", sampling=sampling),
                     emit=lambda n, **f: None, loop=_NoLoop())


class _NoLoop:
    def close(self):
        pass


# ---------------------------------------------------------------------- resume, reclaim, dispenser
def test_a_new_incarnation_never_writes_a_round_the_old_one_wrote_or_had_in_flight(tmp_path):
    harness = Harness({"fast": 0.0, "next": 0.0})
    harness.gates["slow"] = threading.Event()
    prompts = Prompts({0: ["slow"], 1: ["fast"], 2: ["next"]})
    leaving = {"now": False}
    old, _, _ = _round(tmp_path, harness, prompts, incarnation="a", reclaimed=lambda: leaving["now"])
    try:
        old(0)
        old(1)  # round 1 written, round 0 still running
        leaving["now"] = True
        harness.gates["slow"].set()
        _wait(lambda: not old.holds_weights())
        old(2)  # reclaimed: draws, starts nothing new, writes nothing
    finally:
        old.close()
    entries, _ = _entries(tmp_path)
    assert [e.sample_id for e in entries] == ["s-a-r1-fast"]
    assert entries[0].sampler_round["last"] and entries[0].sampler_round["data_cursor"] == {"row_offset": 2}
    assert old.dropped_groups == 1  # the slow group ended after the reclaim marker
    assert not any("-next-" in s for s in harness.started)
    resume = resume_point(entries, "s")
    assert resume.next_rollout_id == 2 and resume.written_rounds == (1,)
    new, _, _ = _round(tmp_path, harness, prompts, incarnation="b", resume=resume)
    try:
        for rid in (0, 1):
            with pytest.raises(SamplerResumeError):
                new(rid)
        new(2)
        new.flush()
    finally:
        new.close()
    entries, _ = _entries(tmp_path)
    assert [e.sample_id for e in entries] == ["s-a-r1-fast", "s-b-r2-next"]
    assert len({e.sample_id for e in entries}) == len(entries)
    resume_point(entries, "s")  # one cursor and at most one last group per round


def test_each_settled_group_reports_its_own_dispenser_row_and_keeps_the_rows_in_flight(tmp_path):
    harness = Harness({"fast": 0.0, "bad": 0.0})
    harness.gates["slow"] = threading.Event()
    prompts = Prompts({0: ["slow", "fast"], 1: ["bad"]}, dispensed=True)

    class FailingRecorder(Recorder):
        async def close(self, trajectory_id):
            turns = await super().close(trajectory_id)
            if "-bad-" in trajectory_id:  # misaligned tokens: the trial is ABORTED
                return [turns[0], TokenTurn(prompt_ids=(9,), completion_ids=(8,), completion_logprobs=(-0.1,),
                                            policy_version=turns[1].policy_version)]
            return turns

    rnd, engine, events = _round(tmp_path, harness, prompts, max_in_flight=6)
    rnd.recorder = FailingRecorder(engine)
    try:
        rnd(0)
        rnd(1)
        _wait(lambda: len(prompts.calls) >= 2)
        harness.gates["slow"].set()
        rnd.flush()
    finally:
        rnd.close()
    slow, fast, bad = (prompts.positions[p] for p in ("slow", "fast", "bad"))
    assert len(prompts.calls) == 3 and all(c[2] == "release" for c in prompts.calls)
    (fast_call,) = [c for c in prompts.calls if c[1] == [fast]]
    (bad_call,) = [c for c in prompts.calls if c[0] == 1]
    assert fast_call[0] == 0 and slow in fast_call[3] and fast not in fast_call[3]  # trained; slow kept
    assert bad_call[1] == [] and slow in bad_call[3] and bad not in bad_call[3]  # released; slow kept
    assert prompts.calls[-1] == (0, [slow], "release", [])
    (bad_group,) = [f for n, f in events if n == GROUP_EVENT and f["prompt_id"] == "bad"]
    assert not bad_group["written"] and bad_group["dropped_reason"] == "below_min_samples"


def test_dispensed_rows_written_keeps_the_rows_still_in_flight():
    class Tasks:
        def __init__(self):
            self.held = {1, 2, 3, 4}
            self.reports = []

        def taken(self):
            return set(self.held)

        def report(self, **kw):
            self.reports.append({k: list(v) for k, v in kw.items() if v})
            for v in kw.values():
                self.held -= set(v)

        def flush(self, timeout_s):
            return True

    tasks = Tasks()
    rows = DispensedRows(tasks, positions=None)
    out = rows.written(0, [1], unwritten="release", keep=[3, 4])
    assert out["trained"] == 1 and out["released"] == 1
    assert tasks.reports == [{"trained": [1]}, {"release": [2]}] and tasks.held == {3, 4}


# ---------------------------------------------------------------------- the island
def test_the_island_holds_releases_while_groups_run_and_drains_before_loading(tmp_path):
    store = CheckpointStore(tmp_path)
    releases = PolicyReleaseStore(store)
    buffer = FileSampleBuffer(store, batch_size=1)
    engine = FakeEngine()
    events: list[tuple[str, dict]] = []

    def publish(prompt):
        if prompt == "a" and releases.latest() is None:
            releases.release(1, "h1", lambda d: (d / "model.safetensors").write_text("w"))

    harness = Harness({"a": 0.2, "b": 0.0, "c": 0.0}, on_start=publish)
    prompts = Prompts({0: ["a"], 1: ["b"], 2: ["c"]})
    config = EngineSamplerConfig(sampler_id="s", num_rollouts=3, max_backlog_groups=100,
                                 sampling=SamplingParams(max_tokens=8, n=N), max_prompt_tokens=1024, poll_s=0.0)

    def make(engine_, writer, **kw):
        return AgenticRound(engine_, writer, recorder=Recorder(engine_), harness=harness, max_in_flight=4,
                            max_response_len=64, min_samples=1, task_defaults={"provider": "docker"},
                            release_poll_s=0.0, wait_s=0.005, **kw)

    island = build_engine_sampler(config, engine=engine, releases=releases, buffer=buffer,
                                  emit=lambda n, **f: events.append((n, f)), prompts=prompts,
                                  reward=lambda item, c: 0.0, sleep=lambda s: None, make_round=make,
                                  checks=PostLoadChecks(expected_expert_matrices=None, require_fixed_answers=False))
    assert isinstance(island.follower, PolicyFollower) and island.follower.hold == island.round.holds_weights
    assert island.follower.live_update is False
    try:
        assert island.run() == 3
    finally:
        island.round.close()
    entries, _ = _entries(tmp_path)
    assert {e.sample_id: e.behavior_version for e in entries} == {"s-r0-a": 0, "s-r1-b": 1, "s-r2-c": 1}
    names = [n for n, _ in events]
    assert names.index(DRAIN_EVENT) < names.index("rl_sampler_policy_loaded")
    reader = FileSampleBuffer(store, batch_size=1)
    reader.refresh()
    assert reader.data_ended


def test_the_follower_loads_nothing_while_the_round_holds_the_weights(tmp_path):
    releases = PolicyReleaseStore(CheckpointStore(tmp_path))
    loaded = []
    follower = PolicyFollower(releases, lambda release, directory: loaded.append(release.version))
    holding = {"now": True}
    follower.hold = lambda: holding["now"]
    releases.release(1, "h1", lambda d: (d / "model.safetensors").write_text("w"))
    assert follower.pending_seq() is not None
    assert follower.refresh() is None and follower.counters.deferred_in_flight == 1 and not loaded
    holding["now"] = False
    assert follower.refresh().version == 1 and loaded == [1] and follower.pending_seq() is None
    follower.live_update, holding["now"] = True, True  # live: the hold does not apply
    releases.release(2, "h2", lambda d: (d / "model.safetensors").write_text("w"))
    assert follower.refresh().version == 2


def test_auto_weight_switch_resolves_by_engine_capability():
    from yeto.rl.engine.agentic_round import resolve_weight_switch

    class Live:
        updates_during_generation = True

    class Static:
        pass

    assert resolve_weight_switch("auto", Live()) == "live"
    assert resolve_weight_switch("auto", Static()) == "drain"
    assert resolve_weight_switch("drain", Live()) == "drain"


@pytest.mark.parametrize("seg_truncated,last,status", [
    (False, "stop", "completed"), (True, "stop", "truncated"), (False, "length", "truncated")])
def test_a_trial_whose_last_turn_hit_the_engine_length_limit_is_truncated(seg_truncated, last, status):
    from types import SimpleNamespace as NS

    from yeto.rl.engine.agentic_round import agentic_sample
    from yeto.rl.engine.engine_sampler import PromptItem

    meta = {k: None for k in ("yeto_agent_outcome", "outcome_reason", "eval_report", "agent_metrics", "trial_dir")}
    finished = NS(
        trajectory=NS(reward=0.0, to_metadata=lambda: meta, task=NS(task_id="t", trajectory_id="t-0")),
        segments=NS(prompt_ids=(1, 2), response_ids=(3, 4), response_logprobs=(0.0, 0.0),
                    response_mask=(1, 1), truncated=seg_truncated),
        turns=[NS(finish_reason=last)])
    item = PromptItem(prompt_id="p", prompt="x", label=None, metadata={})
    out = agentic_sample(item, finished, version=1, provenance={}, group_index=0, index=0)
    assert out["status"] == status
    assert (out["metadata"]["finish_reason"] == "length") == (status == "truncated")


# ---------------------------------------------------------------------- drain timeout vs retry
def test_a_trial_that_ended_just_before_a_drain_gave_up_is_not_retried(tmp_path, monkeypatch):
    """The trial finished (its done callback is queued) when the drain timed out:
    ``cancel()`` returns False, the callback sees an infrastructure failure, and
    the slot must not go back in the queue to start a fresh trial."""
    from yeto.rl.engine import agentic_round as ar

    monkeypatch.setattr(ar, "infra_failure", lambda result: "RuntimeError" if result == "INFRA" else None)
    harness = Harness({})
    rnd, engine, events = _round(tmp_path, harness, Prompts({}), infra_retries=2)
    try:
        def race():
            async def finished():
                return "INFRA"

            group = ar._Group(rollout_id=7, position=0, item=PromptItem("z", "task z",
                                                                       metadata={"agent_task": dict(TASK)}),
                              sessions=["s-r7-z-0", "s-r7-z-1"])
            group.started, group.started_at = 2, rnd.clock()
            task = rnd.loop.loop.create_task(finished())
            return group, task

        group, task = rnd._on_loop(race)
        _wait(task.done)

        def abandon_then_callback():
            rnd._running[task] = (group, 0)
            task.add_done_callback(rnd._trial_done)  # already done: runs on the next loop pass
            assert task.cancel() is False
            return rnd._abandon()

        assert rnd._on_loop(abandon_then_callback) == 1
        _wait(lambda: task not in rnd._running)
        assert group.abandoned and group not in rnd._queue and not group.retry_slots
        assert group.results[0] == "INFRA" and group.retries == [0, 0] and harness.started == []
    finally:
        rnd.close()


def test_no_slot_is_retried_while_a_drain_runs(tmp_path, monkeypatch):
    from yeto.rl.engine import agentic_round as ar

    monkeypatch.setattr(ar, "infra_failure", lambda result: "RuntimeError")
    rnd, engine, events = _round(tmp_path, Harness({}), Prompts({}), infra_retries=2)
    try:
        group = ar._Group(rollout_id=1, position=0, item=PromptItem("z", "task z"), sessions=["a", "b"])
        assert rnd._on_loop(rnd._retry, group, 0, object()) is True
        rnd._paused = True
        assert rnd._on_loop(rnd._retry, group, 1, object()) is False
        rnd._paused = False
        group.abandoned = True
        assert rnd._on_loop(rnd._retry, group, 1, object()) is False
    finally:
        rnd._queue.clear()
        rnd.close()


def test_live_loses_a_trial_without_any_known_version(tmp_path):
    """Live mode: a trial whose start and every turn carry no version is lost
    (``LOST_VERSION``) instead of failing on ``min([])``."""
    class Unversioned(Recorder):
        async def close(self, trajectory_id):
            turns = await super().close(trajectory_id)
            return [TokenTurn(prompt_ids=t.prompt_ids, completion_ids=t.completion_ids,
                              completion_logprobs=t.completion_logprobs, policy_version=None) for t in turns]

    from yeto.rl.engine.agentic_round import LOST_VERSION

    engine = FakeEngine()
    engine.start(None, 0)
    rnd, engine, events = _round(tmp_path, Harness({"a": 0.0}), Prompts({0: ["a"]}), live=True, engine=engine)
    rnd.recorder = Unversioned(engine)
    calls = iter([0])  # the round's own check sees a version; every trial start sees none
    rnd.engine.served_version = lambda: next(calls, None)
    try:
        rnd(0)
        rnd.flush()
    finally:
        rnd.close()
    (group,) = [f for n, f in events if n == GROUP_EVENT]
    assert group["lost"] == {LOST_VERSION: N} and not group["written"]
