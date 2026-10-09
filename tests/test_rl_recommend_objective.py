"""S19 G2: wall-time and GPU-seconds measures; removing an engine can be
recommended only under objective="gpu_seconds" (default stays "wall")."""
from types import SimpleNamespace as NS

import pytest

from yeto.rl.engine.recommend import (
    GPU_SECONDS, HOLD, WALL, CandidateEdge, EdgeCost, LoadWindow, Recommender, RecommendMode,
    candidate_edges_from_attestation, predict_gain, SERIAL)

PH = "b" * 64
# T2R2 -> T2R1: 2 engines -> 1, 4 GPUs -> 3
SHRINK = CandidateEdge("T2R2", "T2R1", 2, 1, 4, 3)
GROW = CandidateEdge("T2R1", "T2R2", 1, 2, 3, 4)
COSTS = {(PH, "T2R2", "T2R1"): EdgeCost(PH, "T2R2", "T2R1", 3.2, 6.4, 0.0),
         (PH, "T2R1", "T2R2"): EdgeCost(PH, "T2R1", "T2R2", 146.5, 179.7, 2.7)}


class Ctl:
    def __init__(self, config):
        self.journal = NS(epochs=NS(config_epoch=1, config_id=config))
        self.profile = NS(contract_hash=PH)


def wins(busy, n=4, tool=0.4):
    return [LoadWindow(i * 60.0, (i + 1) * 60.0, PH, 1, busy, tool, 0.1, 0.0,
                       rollout_busy_fraction=busy) for i in range(n)]


def rec(objective, **kw):
    return Recommender(mode=RecommendMode.RECOMMEND, clock=lambda: 1000.0,
                       objective=objective, **kw)


def test_default_objective_is_wall_and_unknown_rejected():
    assert Recommender().objective == WALL
    with pytest.raises(ValueError):
        Recommender(objective="dollars")


def test_shrink_never_recommended_under_wall():
    r = rec(WALL).recommend(Ctl("T2R2"), wins(0.05), [SHRINK], COSTS)
    assert r.target == "T2R2" and r.reason == HOLD
    assert "no net gain" in r.rejection_reason


def test_shrink_recommended_under_gpu_seconds_when_f_small():
    r = rec(GPU_SECONDS).recommend(Ctl("T2R2"), wins(0.05), [SHRINK], COSTS)
    assert r.actionable and r.target == "T2R1"
    m = r.evidence["measures"]
    assert r.evidence["objective"] == GPU_SECONDS
    assert m["wall_net_s"] < 0 < m["gpu_net_s"]  # both measures reported
    # pessimistic slowdown 2/0.7: wall gain = -0.05*(2/0.7-1)
    g = -0.05 * (2 / 0.7 - 1)
    assert m["gpu_saving_fraction"] == pytest.approx(1 - 0.75 * (1 - g))
    assert m["gpu_need_s"] == pytest.approx((6.4 + 60.0) * 4)


def test_shrink_held_under_gpu_seconds_when_f_large():
    # f=0.5: slowdown 0.5*(2/0.7-1)=0.93 -> 3 GPUs * 1.93 > 4 GPUs
    r = rec(GPU_SECONDS).recommend(Ctl("T2R2"), wins(0.5), [SHRINK], COSTS)
    assert not r.actionable and "gpu_seconds" in r.rejection_reason


def test_grow_with_large_f_wins_wall_but_not_gpu_seconds():
    w = wins(0.6, tool=0.05)
    assert rec(WALL).recommend(Ctl("T2R1"), w, [GROW], COSTS).target == "T2R2"
    r = rec(GPU_SECONDS).recommend(Ctl("T2R1"), w, [GROW], COSTS)
    assert not r.actionable  # one more GPU never pays at efficiency 0.7, f=0.6


def test_gpu_objective_needs_gpu_counts():
    bare = CandidateEdge("T2R2", "T2R1", 2, 1)
    r = rec(GPU_SECONDS).recommend(Ctl("T2R2"), wins(0.05), [bare], COSTS)
    assert "GPU counts" in r.rejection_reason
    gain = predict_gain(wins(0.05), bare, SERIAL)
    m = rec(WALL).measures(gain, COSTS[(PH, "T2R2", "T2R1")], bare)
    assert m["gpu_net_s"] is None and m["wall_net_s"] < 0


def test_attestation_edges_carry_gpu_counts():
    att = NS(certified_edges=[("T2R2", "T2R1", "rollout-only")])
    cfg = {"T2R2": NS(trainer=2, rollout=2, rollout_engine_gpus=1),
           "T2R1": NS(trainer=2, rollout=1, rollout_engine_gpus=1)}
    (e,) = candidate_edges_from_attestation(att, cfg)
    assert (e.source_gpus, e.target_gpus) == (4, 3)
