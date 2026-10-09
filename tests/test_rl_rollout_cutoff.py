"""agentic-rollout-utilization 1.1/1.2: over-sample, cut off at the target, discard."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from yeto.rl.engine.rollout_cutoff import (
    CUTOFF_EVENT,
    CUTOFF_FIELDS,
    Completion,
    CutoffReport,
    cutoff_at_target,
    cutoff_report,
    discard_stats_fields,
)


def test_cutoff_keeps_first_target_and_counts_filter_and_cutoff_separately():
    order = [Completion("a", 4, 10), Completion("b", 4, 12, keep=False),
             Completion("c", 4, 14), Completion("d", 4, 30), Completion("e", 4, 40)]
    out = cutoff_at_target(order, 2)
    assert [c.group_id for c in out.kept] == ["a", "c"]
    assert [c.group_id for c in out.filtered] == ["b"]
    assert [c.group_id for c in out.discarded] == ["d", "e"]
    r = CutoffReport.from_outcome(out)
    assert (r.submitted_groups, r.filtered_groups, r.discarded_groups,
            r.discarded_trajectories, r.discarded_tokens) == (5, 1, 2, 8, 70)
    assert tuple(r.fields()) == CUTOFF_FIELDS
    with pytest.raises(ValueError, match="did not fill"):
        cutoff_at_target(order[:2], 2)


def test_discard_stats_translation_is_all_or_nothing():
    stats = {"groups": 3, "samples": 12, "response_tokens": 900, "unknown_groups": 0}
    assert discard_stats_fields(stats) == {
        "aborted_in_flight_groups": 3, "aborted_in_flight_trajectories": 12,
        "aborted_in_flight_tokens": 900, "aborted_in_flight_unknown_groups": 0}
    assert discard_stats_fields(None) == {}
    assert discard_stats_fields({**stats, "response_tokens": None}) == {}


def test_no_cutoff_reported_emits_nothing():
    assert cutoff_report(SimpleNamespace(aborted_in_flight_groups=0, groups=())) is None
    assert cutoff_report(SimpleNamespace(aborted_in_flight_groups=None, groups=())) is None


def test_fake_engine_target_24_over_sampled_48_trains_exactly_24_current(tmp_path):
    """Spec 多发请求与截止: target 24 trajectories (6 groups x 4), over-sampled 48
    (12 groups), limit 0 -> exactly 24 trained, all current-version; the cut-off
    event reports the discarded trajectories and their generated tokens."""
    import torch

    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.bridges import LocalOnlySync
    from yeto.rl.engine.driver import EventTape, IslandDriver
    from yeto.rl.engine.fake import FakeEngine, fake_capabilities
    from yeto.rl.engine.driver import policy_token

    engine = FakeEngine(tensors={"base_model.model.layer.lora_A.weight": torch.zeros(1, 2)},
                        step_delta=1.0, groups=6, samples_per_group=4, over_sampling_groups=12)
    handles = []
    original = engine.rollout.generate

    def capture(rid, **kw):
        handles.append(original(rid, **kw))
        return handles[-1]

    engine.rollout.generate = capture
    IslandDriver(learner_id=0, rollout=engine.rollout, trainer=engine.trainer,
                 policy_state=engine.policy_state, publisher=engine.publisher,
                 placement=engine.placement, capabilities=fake_capabilities(),
                 algorithm=AlgorithmSpec(), sync=LocalOnlySync(2),
                 events=EventTape(tmp_path / "e.jsonl", 0)).run()
    assert len(handles) == 2
    for h in handles:
        assert sum(len(g.sample_ids) for g in h.groups) == 24
        assert {g.policy_token for g in h.groups} == {policy_token(h.policy_version, h.policy_hash)}
    events = [json.loads(l) for l in (tmp_path / "e.jsonl").read_text().splitlines()]
    cut = [e for e in events if e["event"] == CUTOFF_EVENT]
    assert len(cut) == 2
    for e in cut:
        assert (e["submitted_groups"], e["target_groups"], e["filtered_groups"],
                e["discarded_groups"], e["discarded_trajectories"]) == (12, 6, 0, 6, 24)
        assert e["discarded_tokens"] > 0 and e["mechanism"] == "fake_cutoff"
    trained = [e for e in events if e["event"] == "rl_round_trained"]
    assert [e["aborted_in_flight_groups"] for e in trained] == [6, 6]


def test_fake_engine_event_stream_splits_three_phases(tmp_path):
    """1.3: from the tape alone, each trajectory's generation + tool seconds add up
    to its worker time, and judging is reported separately."""
    import torch

    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.bridges import LocalOnlySync
    from yeto.rl.engine.driver import EventTape, IslandDriver
    from yeto.rl.engine.fake import FakeEngine, fake_capabilities
    from yeto.rl.engine.timeline import validate_trajectory_reward
    from yeto.rl.engine.trajectory_timing import phase_totals

    engine = FakeEngine(tensors={"base_model.model.layer.lora_A.weight": torch.zeros(1, 2)},
                        step_delta=1.0, groups=6, samples_per_group=4, over_sampling_groups=12)
    IslandDriver(learner_id=0, rollout=engine.rollout, trainer=engine.trainer,
                 policy_state=engine.policy_state, publisher=engine.publisher,
                 placement=engine.placement, capabilities=fake_capabilities(),
                 algorithm=AlgorithmSpec(), sync=LocalOnlySync(1), observe=True,
                 events=EventTape(tmp_path / "e.jsonl", 0)).run()
    events = [json.loads(l) for l in (tmp_path / "e.jsonl").read_text().splitlines()]
    trajectories = [e for e in events if e["event"] == "rl_trajectory_reward"]
    assert len(trajectories) == 24
    for e in trajectories:
        assert validate_trajectory_reward(e) == []
        t = phase_totals(e)
        assert t["generation_seconds"] + t["tool_seconds"] == pytest.approx(e["worker_seconds"])
        assert t["judge_seconds"] == 0.5
