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
