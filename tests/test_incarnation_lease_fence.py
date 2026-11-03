"""The incarnation lease fences a superseded
engine sampler and the buffer feed refuses its late groups; the verl round reports
nonzero advantages and per-group uniform rewards; infra failures before the policy
ran are classified for a retry."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_stage1_gaps import _sampler_spec
from yeto.rl.engine import engine_sampler_entry as entry
from yeto.rl.engine.incarnation_lease import (
    STALE_REASON,
    IncarnationGate,
    IncarnationLease,
    lease_key,
    read_lease,
)
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer
from yeto.rl.engine.trainer_lifecycle import SampleIndexEntry

STATE_ENV = "YETO_TEST_FENCE_STATE"


def fencing_reward(item, completion) -> float:
    """At call ``fence_at`` another incarnation claims the island's lease."""
    path = Path(os.environ[STATE_ENV])
    state = json.loads(path.read_text())
    state["n"] += 1
    path.write_text(json.dumps(state))
    if state["n"] == state["fence_at"]:
        store = CheckpointStore(state["store"])
        rival = IncarnationLease(store, "ppu4", "zzzz9999", clock=lambda: 1e12)
        buffer = FileSampleBuffer(store, batch_size=1)
        rival.claim(buffer.entries())
    return 0.5


def _entry(sampler: str, inc: str, appended_at: float = 0.0) -> SampleIndexEntry:
    return SampleIndexEntry(sample_id=f"{inc}-x", behavior_version=0, payload_ref="p", appended_at=appended_at,
                            sampler_round={"sampler": sampler, "incarnation": inc, "rollout_id": 0})


# --------------------------------------------------------------------------- c1: sampler side

def test_lease_claim_lists_the_superseded_and_poll_fences(tmp_path):
    store = CheckpointStore(tmp_path)
    a = IncarnationLease(store, "spot-0", "aaaa", clock=lambda: 100.0)
    a.claim()
    assert a.poll() is False
    b = IncarnationLease(store, "spot-0", "bbbb", clock=lambda: 200.0)
    record = b.claim([_entry("spot-0", "aaaa")] * 3 + [_entry("other", "cccc")], earlier=["old0"])
    assert record["superseded"] == ["aaaa", "old0"] and record["sealed"] == {"aaaa": 3}
    assert record["generation"] == 2
    assert a.poll() is True and a.fenced_by == "bbbb"
    assert a.check_before_write() is True  # sticky
    assert b.poll() is False
    c = IncarnationLease(store, "spot-0", "cccc")
    assert c.claim()["superseded"] == ["aaaa", "old0", "bbbb"]


def test_check_before_write_rereads_after_the_interval(tmp_path):
    store = CheckpointStore(tmp_path)
    now = [0.0]
    a = IncarnationLease(store, "s", "aaaa", write_check_s=10.0, monotonic=lambda: now[0])
    a.claim()
    assert a.check_before_write() is False
    IncarnationLease(store, "s", "bbbb").claim()
    now[0] = 5.0
    assert a.check_before_write() is False  # trusted the read 5 s ago
    now[0] = 11.0
    assert a.check_before_write() is True


def test_a_missing_or_broken_lease_never_fences(tmp_path):
    store = CheckpointStore(tmp_path)
    a = IncarnationLease(store, "s", "aaaa")
    assert a.poll() is False and a.read_failures == 1
    path = tmp_path / lease_key("s")
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    assert a.poll() is False and a.read_failures == 2


def test_a_superseded_engine_sampler_writes_nothing_more_and_leaves_fenced(tmp_path, monkeypatch):
    state = tmp_path / "state.json"
    monkeypatch.setenv(STATE_ENV, str(state))
    state.write_text(json.dumps({"n": 0, "fence_at": 5, "store": str(tmp_path / "store")}))
    spec = _sampler_spec(tmp_path, num_rollouts=4, reward="tests.test_incarnation_lease_fence:fencing_reward",
                         incarnation_lease={"write_check_s": 0, "poll_s": 0.05})
    rounds = entry.run(spec, resource_sampler=lambda emit: None, incarnation="aaaa0001")
    assert rounds <= 2
    buffer = FileSampleBuffer(CheckpointStore(tmp_path / "store"), batch_size=1)
    # round 0 (reward calls 1-4) wrote its 2 groups; the claim came during round 1's scoring
    assert [e.sampler_round["rollout_id"] for e in buffer.entries()] == [0, 0]
    [tape] = (tmp_path / "tape").glob("events.*.jsonl")
    events = [json.loads(line) for line in tape.read_text().splitlines()]
    names = [e["event"] for e in events]
    assert "rl_sampler_lease" in names
    (leave,) = [e for e in events if e["event"] == "rl_sampler_leave"]
    assert leave["reason"] == "FENCED" and leave["superseded_by"] == "zzzz9999"
    lease = read_lease(CheckpointStore(tmp_path / "store"), "ppu4")
    assert lease["incarnation"] == "zzzz9999" and lease["sealed"] == {"aaaa0001": 2}


def test_the_lease_can_be_turned_off(tmp_path):
    spec = _sampler_spec(tmp_path, num_rollouts=1, incarnation_lease=False)
    assert entry.run(spec, resource_sampler=lambda emit: None, incarnation="aaaa0001") == 1
    assert read_lease(CheckpointStore(tmp_path / "store"), "ppu4") is None
    other = tmp_path / "on"
    other.mkdir()
    spec = _sampler_spec(other, num_rollouts=1)
    assert entry.run(spec, resource_sampler=lambda emit: None, incarnation="bbbb0002") == 1
    assert read_lease(CheckpointStore(other / "store"), "ppu4")["incarnation"] == "bbbb0002"


def test_streaming_round_drops_a_settled_group_once_fenced(tmp_path):
    """The streaming agentic round reads the fence right before each group write."""
    from tests.test_agentic_round_streaming import Harness, Prompts, _entries, _round
    from yeto.rl.engine.agentic_round import GROUP_EVENT

    fenced = [False]
    rnd, engine, events = _round(tmp_path, Harness({"a": 0.0, "b": 0.0}), Prompts({0: ["a"], 1: ["b"]}),
                                 incarnation="aaaa")
    rnd.write_fenced = lambda: fenced[0]
    try:
        rnd(0)
        rnd.flush()
        fenced[0] = True
        rnd(1)
        rnd.flush()
    finally:
        rnd.close()
    groups = [f for n, f in events if n == GROUP_EVENT]
    assert [(g["prompt_id"], g["written"], g["dropped_reason"]) for g in groups] == [
        ("a", True, None), ("b", False, "fenced")]
    entries, _ = _entries(tmp_path)
    assert len(entries) == 1


# --------------------------------------------------------------------------- c1: buffer side

def test_gate_refuses_late_groups_of_a_superseded_incarnation(tmp_path):
    store = CheckpointStore(tmp_path)
    entries = [_entry("s", "aaaa", 10.0), _entry("s", "aaaa", 20.0), _entry("s", "bbbb", 30.0),
               _entry("s", "aaaa", 40.0), _entry("s", "cccc", 50.0), _entry("t", "xxxx", 60.0),
               SampleIndexEntry(sample_id="old", behavior_version=0, payload_ref="p")]
    gate = IncarnationGate(store, lambda: entries, clock_margin_s=5.0)
    assert [gate.verdict(e, i) for i, e in enumerate(entries)] == [None] * 7  # no lease: all pass
    IncarnationLease(store, "s", "aaaa", clock=lambda: 0.0).claim()
    IncarnationLease(store, "s", "bbbb", clock=lambda: 35.0).claim(entries[:1])  # saw 1 of aaaa
    gate = IncarnationGate(store, lambda: entries, clock_margin_s=5.0)
    verdicts = [gate.verdict(e, i) for i, e in enumerate(entries)]
    # aaaa #1 sealed; aaaa #2 written at 20 < 35 - 5 (before the claim); bbbb holds;
    # aaaa #3 at 40 after the claim: stale; cccc unknown (a newer claim not seen yet); other sampler
    assert verdicts[:3] == [None, None, None] and verdicts[3] and "superseded by bbbb" in verdicts[3]
    assert verdicts[4:] == [None, None, None] and gate.stale == 1
    # read again from an earlier cursor: the ordinals are counted again
    assert gate.verdict(entries[3], 3) == verdicts[3]


def test_buffer_feed_drops_stale_incarnation_groups(tmp_path):
    from yeto.rl.adapters.verl.buffer_feed import BufferFeed, RowShape
    from yeto.rl.engine.buffer_payload import GroupWriter
    from yeto.rl.engine.policy_versions import engine_weight_version

    store = CheckpointStore(tmp_path)
    writer_buffer = FileSampleBuffer(store, batch_size=1)
    IncarnationLease(store, "s", "aaaa").claim()

    def write(inc: str, rid: int) -> None:
        sample = {"tokens": [1, 2, 3], "response_length": 1, "rollout_log_probs": [-0.1], "loss_mask": [1],
                  "reward": 1.0, "metadata": {"weight_version": engine_weight_version(0)}}
        GroupWriter(writer_buffer, "s").write([dict(sample, index=0), dict(sample, index=1)],
                                              group_key=f"{inc}-r{rid}-p",
                                              sampler_round={"sampler": "s", "incarnation": inc,
                                                             "rollout_id": rid, "last": True})

    write("aaaa", 0)
    IncarnationLease(store, "s", "bbbb", clock=lambda: 0.0).claim(writer_buffer.entries())
    write("aaaa", 1)  # the superseded process still writes
    write("bbbb", 2)
    reader = FileSampleBuffer(store, batch_size=1)
    records, emitted = [], []
    feed = BufferFeed(reader, vmap=lambda: None, shape=RowShape(16, 16, 0), add_group=lambda rows: None,
                      record=records.append, emit=lambda name, **f: emitted.append((name, f)), wait_s=0.0)
    feed._local = lambda version: 0  # noqa: SLF001 - versions are not under test
    first = feed.pull()
    feed.drop()
    second = feed.pull()
    feed.drop()
    assert first["uid"] == "buf0" and second["uid"] == "buf2"
    stale = [r for r in records if r.get("dropped") == STALE_REASON]
    assert [r["position"] for r in stale] == [1]
    assert feed.dropped == {STALE_REASON: 1}
    assert emitted[0][0] == "rl_buffer_group_dropped" and emitted[0][1]["reason"] == STALE_REASON


def test_buffer_feed_gate_off_by_env(tmp_path, monkeypatch):
    from yeto.rl.adapters.verl.buffer_feed import INCARNATION_FENCE_ENV, default_incarnation_gate

    reader = FileSampleBuffer(CheckpointStore(tmp_path), batch_size=1)
    assert default_incarnation_gate(reader) is not None
    assert default_incarnation_gate(reader, {INCARNATION_FENCE_ENV: "0"}) is None
    assert default_incarnation_gate(SimpleNamespace()) is None


# --------------------------------------------------------------------------- c2

def test_advantage_stats_counts_nonzero_samples_and_uniform_groups():
    from yeto.rl.adapters.verl.fully_async_round import advantage_stats

    def kept(*rewards):
        return SimpleNamespace(meta=SimpleNamespace(rewards=tuple(rewards)))

    collector = SimpleNamespace(kept=[kept(1.0, 1.0, 1.0, 1.0), kept(1.0, 0.0, 0.0, 0.0), kept(0.0, 0.0),
                                      kept(0.5, 1.0)])
    stats = advantage_stats(collector)
    assert stats == {"group_reward_uniform": [True, False, True, False], "uniform_reward_groups": 2,
                     "nonzero_advantages": 6}
    nan = advantage_stats(SimpleNamespace(kept=[kept(float("nan"), 1.0)]))
    assert nan["nonzero_advantages"] is None


def test_driver_stats_carry_the_nonzero_advantage_count():
    from yeto.rl.engine.driver import _nonzero_advantage_count

    groups = [SimpleNamespace(sample_ids=("a", "b")), SimpleNamespace(sample_ids=("c", "d"))]
    assert _nonzero_advantage_count(SimpleNamespace(groups=groups, nonzero_advantages=3)) == 3
    assert _nonzero_advantage_count(SimpleNamespace(groups=groups, nonzero_advantages=None)) is None
    assert _nonzero_advantage_count(SimpleNamespace(groups=groups, nonzero_advantages=9)) is None
    assert _nonzero_advantage_count(SimpleNamespace(groups=groups)) is None


# --------------------------------------------------------------------------- infra failure classification

@pytest.mark.parametrize("reason, verdict_exc, turns, expected", [
    ("RuntimeError", None, (), "RuntimeError"),  # the harness raised before the agent ran
    ("EnvironmentStartTimeoutError", "EnvironmentStartTimeoutError", (), "EnvironmentStartTimeoutError"),
    ("wall_clock_cap", None, (), None),  # a timeout: never retried
    ("token_alignment: x", None, (), None),
    ("BoatQuotaError", "BoatQuotaError", (), None),  # the next sandbox fails the same way
    ("NonZeroAgentExitCodeError", "NonZeroAgentExitCodeError", ("turn",), None),  # the policy ran
])
def test_infra_failure_classification(reason, verdict_exc, turns, expected):
    from yeto.rl.agentic import Outcome
    from yeto.rl.engine.agentic_round import infra_failure

    verdict = None if verdict_exc is None else SimpleNamespace(exception_type=verdict_exc)
    trajectory = SimpleNamespace(outcome=Outcome.ABORTED, reason=reason, verdict=verdict)
    assert infra_failure(SimpleNamespace(trajectory=trajectory, turns=turns)) == expected
    ok = SimpleNamespace(outcome=Outcome.OK, reason="verified", verdict=None)
    assert infra_failure(SimpleNamespace(trajectory=ok, turns=())) is None
