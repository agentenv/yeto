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
