"""rl-infra-spec 6.6 CPU replay: four scenarios x groups (no GPU)."""
import json

import pytest

from yeto.rl.engine.auto import AutoPolicy
from yeto.rl.engine.recommend import OVERLAP, SERIAL, EdgeCost
from yeto.rl.engine.replay import (GROUPS, RESULT_FIELDS, SCENARIOS, SimConfig, TimeModel,
                                   compare, payback_rounds, synth_trace, trace_from_events)

PH = "replay-profile"
CFGS = {"T2R1S1": SimConfig("T2R1S1", 1, 2), "T2R2S0": SimConfig("T2R2S0", 2, 2),
        "T1R3": SimConfig("T1R3", 3, 1)}
POL = AutoPolicy(k_windows=3, safety_margin_s=10, horizon_s=3600, min_dwell_s=100,
                 cooldown_s=100, max_switches=2, switch_window_s=3600)


def costs(up=30.0, down=5.0):
    out = {}
    for a in CFGS:
        for b in CFGS:
            if a != b:
                c = up if CFGS[b].rollout_engines > CFGS[a].rollout_engines else down
                out[(PH, a, b)] = EdgeCost(PH, a, b, c * 0.8, c, 1.0)
    return out


def trace(s, **kw):
    base = dict(rounds=200, seed=7, reference="T2R1S1", gen_s=20.0, train_s=5.0,
                tail_s=60.0, tool_s=60.0, period=50)
    base.update(kw)
    return synth_trace(s, **base)


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("mode", [SERIAL, OVERLAP])
def test_four_scenarios_all_groups(scenario, mode):
    r = compare(trace(scenario), TimeModel(CFGS, "T2R1S1", mode=mode), "T2R1S1",
                pool_gpus=4, costs=costs(), policy=POL)
    json.loads(json.dumps(r))
    assert set(r["groups"]) == set(GROUPS)
    for g in GROUPS:
        assert set(RESULT_FIELDS) <= set(r["groups"][g])
        assert r["groups"][g]["rounds"] == 200
    assert r["groups"]["best_fixed"]["wall_s"] <= r["groups"]["fixed_default"]["wall_s"]
    assert r["groups"]["dynamic"]["switches"] <= POL.max_switches * 3
    if mode == SERIAL:
        assert r["mode_gain_s"] == 0
    assert r["resize_gain_dynamic_s"] == pytest.approx(
        r["groups"]["fixed_default"]["wall_s"] - r["groups"]["dynamic"]["wall_s"])


def test_dynamic_switches_on_stable_gen_heavy_load():
    r = compare(trace("stable"), TimeModel(CFGS, "T2R1S1"), "T2R1S1", pool_gpus=4,
                costs=costs(), policy=POL)
    d = r["groups"]["dynamic"]
    assert d["switches"] >= 1 and d["switch_block_s"] > 0
    assert r["resize_gain_dynamic_s"] > 0


def test_oscillating_load_bounded_switches():
    r = compare(trace("changing", period=1, step_factor=0.05), TimeModel(CFGS, "T2R1S1"),
                "T2R2S0", pool_gpus=4, costs=costs(), policy=POL)
    assert r["groups"]["dynamic"]["switches"] <= POL.max_switches


def test_no_cost_table_dynamic_equals_default():
    for s in SCENARIOS:
        r = compare(trace(s), TimeModel(CFGS, "T2R1S1"), "T2R1S1", pool_gpus=4, costs={},
                    policy=POL)
        assert r["groups"]["dynamic"]["switches"] == 0
        assert r["groups"]["dynamic"]["wall_s"] == pytest.approx(r["groups"]["fixed_default"]["wall_s"])


def test_tool_wait_holds():
    r = compare(trace("tool_wait", gen_s=2.0), TimeModel(CFGS, "T2R1S1"), "T2R1S1",
                pool_gpus=4, costs=costs(), policy=POL)
    d = r["groups"]["dynamic"]
    assert d["switches"] == 0
    assert any("tool-heavy" in x for x in d["decisions"])


def test_long_tail_no_switch_from_tail():
    # gain only scales GPU-busy generation; tail is not billed as a resize gain
    r = compare(trace("long_tail", gen_s=0.05, tail_prob=0.5), TimeModel(CFGS, "T2R1S1"),
                "T2R1S1", pool_gpus=4, costs=costs(), policy=POL)
    assert r["groups"]["dynamic"]["switches"] == 0


def test_payback_and_baseline_scale():
    # 5.1-BASELINE n=1: R1->R2 saves 0.15 s/round, up edge 97 s + 2.5 s recovery
    tr = synth_trace("stable", rounds=50, seed=1, reference="T2R1S1", gen_s=0.3,
                     train_s=10.57, jitter=0.0)
    m = TimeModel(CFGS, "T2R1S1")
    pb = payback_rounds(tr, m, "T2R1S1", "T2R2S0", EdgeCost(PH, "T2R1S1", "T2R2S0", 97, 97, 2.5))
    assert pb == pytest.approx(99.5 / 0.15)


def test_replay_from_events():
    ev = []
    for i in range(6):
        t = i * 10.0
        ev.append({"event": "rl_timeline_span", "kind": "compute", "role": "rollout",
                   "start": t, "end": t + 4, "profile_hash": PH, "epoch": 1})
        ev.append({"event": "rl_timeline_span", "kind": "compute", "role": "trainer",
                   "start": t + 4, "end": t + 10, "profile_hash": PH, "epoch": 1})
    tr = trace_from_events(ev, 10.0, reference="T2R1S1")
    assert len(tr.rounds) == 6
    assert tr.rounds[0].gen_s == pytest.approx(4) and tr.rounds[0].train_s == pytest.approx(6)
