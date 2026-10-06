"""rl-infra-spec task 1.7 (pure accounting part)."""

from __future__ import annotations

import pytest

from yeto.rl.engine.timeline import LoadSample, Span, classify_load, summarize


def _s(task, role, kind, a, b, epoch=0):
    return Span(task, role, kind, a, b, "sha256:p", epoch)


def test_overlapping_spans_are_not_billed_twice():
    serial = summarize([_s("generate", "rollout", "compute", 0, 10),
                        _s("train", "trainer", "compute", 10, 15)])
    assert serial["wall_s"] == 15 and serial["overlap_s"] == 0
    overlap = summarize([_s("generate", "rollout", "compute", 0, 10),
                         _s("checkpoint", "trainer", "compute", 5, 12),
                         _s("reward", "cpu", "tool-wait", 2, 8)])
    assert overlap["wall_s"] == 12
    assert overlap["overlap_s"] == pytest.approx(23 - 12)
    assert overlap["by_role"]["rollout"] == 10 and overlap["by_kind"]["tool-wait"] == 6


def test_mixed_epochs_are_refused():
    with pytest.raises(ValueError, match="one summary"):
        summarize([_s("train", "trainer", "compute", 0, 1, 0),
                   _s("train", "trainer", "compute", 1, 2, 1)])


def test_tool_wait_is_not_gpu_saturation():
    assert classify_load(LoadSample(0, 0, 5, 0, 8)) == "tool-wait"
    assert classify_load(LoadSample(20, 8, 0, 0, 8)) == "rollout-saturated"
    assert classify_load(LoadSample(0, 2, 0, 3, 8)) == "long-tail"
    assert classify_load(LoadSample(0, 0, 0, 3, 8)) == "rollout-idle"


def test_transition_cost_distribution_and_predeclared_bottleneck_rule():
    from yeto.rl.engine.timeline import (
        MIN_SAMPLES_PER_EDGE,
        select_bottleneck,
        transition_cost_distribution,
    )

    rows = [("T4R2S2", "T4R4S0", {"drain": 2.0, "init": 10.0 + i, "publish": 3.0,
                                  "background_restore": 50.0})
            for i in range(MIN_SAMPLES_PER_EDGE)]
    rows += [("T4R4S0", "T4R2S2", {"drain": 8.0, "init": 1.0, "publish": 1.0})
             for _ in range(MIN_SAMPLES_PER_EDGE)]
    dist = transition_cost_distribution(rows)
    fwd = dist[("T4R2S2", "T4R4S0")]
    assert fwd["n"] == 3 and fwd["phases"]["init"]["p50"] == 11.0
    assert fwd["phases"]["blocking_total"]["max"] == 17.0  # background not billed as blocking
    choice = select_bottleneck(dist)
    # init share: (11/16 + 1/10)/2 = 0.39; drain: (2/16 + 8/10)/2 = 0.46
    assert choice["status"] == "selected" and choice["phase"] == "drain"
    assert select_bottleneck(transition_cost_distribution(rows[:2]))["status"] == "insufficient"
    assert select_bottleneck({})["status"] == "insufficient"
    with pytest.raises(ValueError, match="unknown"):
        transition_cost_distribution([("a", "b", {"magic": 1.0})])


# -- 1.7 load_windows / LoadSummary ------------------------------------------
from yeto.rl.engine.timeline import LoadSummary, load_windows, validate_load_sample  # noqa: E402


def _span(task, role, kind, a, b, epoch=0):
    return {"event": "rl_timeline_span", "task": task, "role": role, "kind": kind,
            "start": a, "end": b, "profile_hash": "sha256:p", "epoch": epoch,
            "weight_transport": "nccl"}


def _sample(t, q, a, tw, cap=8, epoch=0):
    return {"event": "rl_load_sample", "t": t, "queued_requests": q, "running_requests": a,
            "tool_wait_trajectories": tw, "engine_capacity": cap, "ready_groups": None,
            "profile_hash": "sha256:p", "epoch": epoch, "weight_transport": "nccl"}


def test_load_windows_separates_tool_wait_from_gpu_saturation():
    evs = [_span("generate", "rollout", "compute", 0, 10),
           _span("train", "trainer", "compute", 5, 10),   # overlapped: not double billed
           _span("publish", "trainer+rollout", "transfer", 10, 12),
           *[_sample(t, 20, 8, 0) for t in (1, 2, 3)],     # saturated
           *[_sample(t, 0, 0, 4) for t in (11, 12, 13)],   # tool wait
           {"event": "rl_readiness", "t": 12, "ready_groups": 2, "policy_age": 1,
            "profile_hash": "sha256:p", "epoch": 0},
           {"event": "rl_round_labels", "t": 15, "rl/groups": 4, "profile_hash": "sha256:p",
            "config_epoch": 0, "rl/masked_fraction": 0.1}]
    w0, w1 = load_windows(evs, 10.0)
    assert isinstance(w0, LoadSummary) and (w0.window_start, w0.window_end) == (0, 10)
    assert w0.gpu_busy_fraction == 1.0 and w0.tool_wait_fraction == 0.0
    assert w0.queued == 20 and w0.active == 8 and w0.weight_transport == "nccl"
    assert w1.gpu_busy_fraction == 0.0 and w1.tool_wait_fraction == 1.0
    assert w1.publish_block_fraction == pytest.approx(0.2)
    assert w1.policy_age == 1 and w1.ready_groups == 2 and w1.consume_rate == pytest.approx(0.4)
    assert w0.policy_age is None and w0.ready_groups is None


def test_load_windows_split_by_epoch_and_empty_when_unobserved():
    evs = [_span("train", "trainer", "compute", 0, 4, 0), _span("train", "trainer", "compute", 0, 2, 1)]
    ws = load_windows(evs, 4.0)
    assert [(w.epoch, w.gpu_busy_fraction) for w in ws] == [(0, 1.0), (1, 0.5)]
    assert load_windows([{"event": "rl_phase", "phase": "train"}], 1.0) == []
    with pytest.raises(ValueError):
        load_windows([], 0)


def test_resource_peaks_in_load_schema():
    assert validate_load_sample({"peak_gpu_mem_bytes": 10, "peak_cpu_rss_bytes": None}) == []
