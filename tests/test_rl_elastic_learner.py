"""D2 learner wiring: --rl-recommend-mode / --rl-edge-costs-path / --rl-elastic-window-s
from the launcher to the learner, ElasticHook built by the adapter, in-memory event
mirror, eval excluded from rollout busy, and CPU end-to-end runs (no GPU)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from rl_e2e_launch import island_run, learner_from_run  # noqa: E402
from test_rl_elastic_wire import (BIG, PH, SMALL, W, Clock, _profile,  # noqa: E402
                                  cost_table, setup, synth)
from test_rl_launch_e2e_b1 import BASE, _elastic  # noqa: E402
from test_rl_reconfig_e1 import _events  # noqa: E402
from yeto.rl.engine.auto import AutoPolicy  # noqa: E402
from yeto.rl.engine.controller import SUCCEEDED, read_journal  # noqa: E402
from yeto.rl.engine.miles_adapter.elastic_hook import (TUNING,  # noqa: E402
                                                       check_recommend_flags,
                                                       elastic_hook_for,
                                                       recommend_flags)
from yeto.rl.engine.timeline import load_windows  # noqa: E402

FLAGS = ("--rl-observe-timeline", "--rl-recommend-mode", "recommend",
         "--rl-edge-costs-path", "/tmp/costs.json", "--rl-elastic-window-s", "50")


# ---------------------------------------------------------------- CLI -> learner
def test_flags_reach_the_learner_and_miles_args(tmp_path, monkeypatch):
    from yeto.rl import learner

    run = island_run(BASE + _elastic(tmp_path) + FLAGS, monkeypatch)
    args, _ = learner_from_run(run, tmp_path / "home")
    assert (args.rl_recommend_mode, args.rl_edge_costs_path, args.rl_elastic_window_s) == (
        "recommend", "/tmp/costs.json", 50.0)
    ma = SimpleNamespace()
    learner.apply_ports_infra_switches(args, ma, {})
    assert (ma.yeto_rl_recommend_mode, ma.yeto_rl_edge_costs_path,
            ma.yeto_rl_elastic_window_s) == ("recommend", "/tmp/costs.json", 50.0)


def test_default_run_carries_no_recommend_flags(tmp_path, monkeypatch):
    run = island_run(BASE + _elastic(tmp_path) + ("--rl-observe-timeline",), monkeypatch)
    assert "--rl-recommend-mode" not in run and "--rl-edge-costs-path" not in run
    args, _ = learner_from_run(run, tmp_path / "home")
    assert args.rl_recommend_mode is None
    from yeto.rl import learner

    ma = SimpleNamespace()
    learner.apply_ports_infra_switches(args, ma, {})
    assert not hasattr(ma, "yeto_rl_recommend_mode")


@pytest.mark.parametrize("given,missing", [
    (dict(rl_recommend_mode="recommend", rl_elastic=True), "--rl-observe-timeline"),
    (dict(rl_recommend_mode="auto", rl_observe_timeline=True), "--rl-elastic"),
    (dict(rl_elastic_window_s=10.0), "--rl-elastic"),
    (dict(rl_elastic_window_s=0.0, rl_elastic=True, rl_observe_timeline=True), "positive"),
])
def test_flags_refused_without_their_preconditions(given, missing):
    with pytest.raises(ValueError, match=missing):
        check_recommend_flags(SimpleNamespace(**given))
    check_recommend_flags(SimpleNamespace())  # nothing given: fine


def test_launcher_refuses_before_provisioning(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="--rl-observe-timeline"):
        island_run(BASE + _elastic(tmp_path) + ("--rl-recommend-mode", "recommend"), monkeypatch)


# ---------------------------------------------------------------- events mirror
class _NoPathTape:
    """Like Miles' tape via ``_append_rl_event``: the hook has no local file to read."""

    def __init__(self):
        self.records = []

    def append(self, ev):
        self.records.append(dict(ev))


def _hooked(tmp_path, mode, *, auto_cap=True, rounds=4, num_rollout=1000, costs=None, observe=True):
    driver, ctl, _, fork = setup(tmp_path, hook=False, rounds=rounds, auto_cap=auto_cap,
                                 observe=observe, wall=Clock(10_000.0, 0.0))
    ma = SimpleNamespace(num_rollout=num_rollout, yeto_rl_elastic_window_s=W,
                         **({"yeto_rl_recommend_mode": mode} if mode else {}),
                         **({"yeto_rl_edge_costs_path": costs} if costs else {}))
    driver.elastic_hook = elastic_hook_for(ma, controller=ctl, profile=_profile(),
                                           observe=observe, clock=Clock(10_000.0, 0.0))
    return driver, ctl, fork


def test_hook_reads_the_drivers_own_events_without_a_tape_path(tmp_path):
    driver, ctl, _ = _hooked(tmp_path, "recommend")
    tape = _NoPathTape()
    driver.events = tape
    assert getattr(tape, "path", None) is None
    driver.run()
    seen = {e["event"] for e in driver.elastic_hook._buf}
    assert {"rl_timeline_span", "rl_readiness"} <= seen
    recs = [e for e in tape.records if e["event"] == "rl_elastic_recommendation"]
    assert recs and recs[-1]["windows"] >= 1


# ---------------------------------------------------------------- eval exclusion
def test_eval_compute_is_not_rollout_busy():
    lab = dict(profile_hash=PH, epoch=0)
    base = [{"event": "rl_timeline_span", "task": "generate", "role": "rollout",
             "kind": "compute", "start": 0.0, "end": 20.0, **lab},
            {"event": "rl_load_sample", "t": 99.0, **lab}]
    ev = [{"event": "rl_timeline_span", "task": "eval", "role": "rollout", "kind": "compute",
           "start": 20.0, "end": 80.0, **lab}]
    (w0,), (w1,) = load_windows(base, 100.0), load_windows(base + ev, 100.0)
    assert w0.rollout_busy_fraction == pytest.approx(0.2)
    assert w1.rollout_busy_fraction == pytest.approx(0.2)


# ---------------------------------------------------------------- end to end
def test_recommend_mode_end_to_end(tmp_path):
    driver, ctl, _ = _hooked(tmp_path, "recommend", costs=cost_table(tmp_path))
    assert ctl.recommend_mode == "recommend"
    assert driver.elastic_hook.declared_edges == {(SMALL, BIG), (BIG, SMALL)}
    driver.run()
    recs = [e for e in _events(tmp_path) if e["event"] == "rl_elastic_recommendation"]
    assert recs and all(e["profile_hash"] == PH for e in recs)
    assert ctl.journal.epochs.config_epoch == 0  # recommend never executes


def test_auto_mode_end_to_end_requests_once(tmp_path):
    driver, ctl, _ = _hooked(tmp_path, "auto", costs=cost_table(tmp_path), rounds=5)
    assert ctl.recommend_mode == "auto"
    hook = driver.elastic_hook
    hook.auto.policy = AutoPolicy(k_windows=3, safety_margin_s=60.0, min_dwell_s=0.0,
                                  cooldown_s=600.0)
    hook.events_source = lambda: synth("saturated", 4, epoch=int(ctl.journal.epochs.config_epoch))
    calls = []
    real = ctl.request
    ctl.request = lambda *a, **k: (calls.append(a), real(*a, **k))[1]
    driver.run()
    assert len(calls) == 1 and calls[0][0].startswith("auto-") and calls[0][1] == BIG
    assert ctl.status(calls[0][0])["phase"] == SUCCEEDED
    assert ctl.journal.epochs.config_id == BIG


def test_auto_without_capability_falls_back_to_recommend(tmp_path):
    _, ctl, _ = _hooked(tmp_path, "auto", auto_cap=False)
    assert ctl.recommend_mode == "recommend"
    modes = [r for r in read_journal(tmp_path / "state/reconfig") if r["kind"] == "recommend_mode"]
    assert modes[-1]["mode"] == "recommend" and "auto refused" in modes[-1]["reason"]


def _shape(tmp_path):
    # values carry wall-clock/run-specific fields; the sequence and schema must match
    return [(e["event"], sorted(e)) for e in _events(tmp_path)]


@pytest.mark.parametrize("mode", [None, "disabled"])
def test_disabled_is_event_identical_to_no_flags(tmp_path, mode):
    a, _, _ = _hooked(tmp_path / "a", mode)
    assert a.elastic_hook is None
    a.run()
    b, _, _ = setup(tmp_path / "b", hook=False, wall=Clock(10_000.0, 0.0))[:3]
    b.run()
    assert _shape(tmp_path / "a") == _shape(tmp_path / "b")


# ---------------------------------------------------------------- tuning flags
_GOOD = {"auto-k-windows": 4, "auto-safety-margin-s": 30.0, "auto-horizon-s": 900.0,
         "auto-min-dwell-s": 60.0, "auto-cooldown-s": 90.0, "auto-max-switches": 3,
         "auto-switch-window-s": 600.0, "recommend-ttl-s": 120.0, "recommend-min-windows": 2,
         "recommend-efficiency-lower": 0.5}
_ON = dict(rl_recommend_mode="auto", rl_elastic=True, rl_observe_timeline=True)


def _ns(flag, v, **kw):
    return SimpleNamespace(**{**_ON, **kw, "rl_" + flag.replace("-", "_"): v})


@pytest.mark.parametrize("flag", [t[0] for t in TUNING])
def test_tuning_flag_passthrough(flag):
    args = _ns(flag, _GOOD[flag])
    check_recommend_flags(args)
    assert f" --rl-{flag} {_GOOD[flag]!r}" in recommend_flags(args)
    assert f"--rl-{flag}" not in recommend_flags(SimpleNamespace(**_ON))  # default: absent


@pytest.mark.parametrize("flag,bad", [
    ("auto-k-windows", 0), ("auto-safety-margin-s", 0.0), ("auto-horizon-s", -1.0),
    ("auto-min-dwell-s", 0.0), ("auto-cooldown-s", -5.0), ("auto-max-switches", 0),
    ("auto-switch-window-s", 0.0), ("recommend-ttl-s", 0.0), ("recommend-min-windows", 0),
    ("recommend-efficiency-lower", 0.0), ("recommend-efficiency-lower", 1.5),
])
def test_tuning_flag_rejects_bad_values(flag, bad):
    with pytest.raises(ValueError, match=f"--rl-{flag}"):
        check_recommend_flags(_ns(flag, bad))


def test_tuning_flag_needs_recommend_mode():
    with pytest.raises(ValueError, match="needs --rl-recommend-mode"):
        check_recommend_flags(SimpleNamespace(rl_auto_k_windows=3, rl_elastic=True,
                                              rl_observe_timeline=True))
    check_recommend_flags(_ns("recommend-efficiency-lower", 1.0))  # upper bound inclusive


def test_tuning_reaches_policy_and_recommender(tmp_path):
    ma = SimpleNamespace(yeto_rl_recommend_mode="recommend", num_rollout=100)
    for flag, v in _GOOD.items():
        setattr(ma, "yeto_rl_" + flag.replace("-", "_"), v)
    ctl = SimpleNamespace(configs={}, set_recommend_mode=lambda *a, **k: None)
    hook = elastic_hook_for(ma, controller=ctl, profile=None, observe=True)
    pol = hook.auto.policy
    assert (pol.k_windows, pol.safety_margin_s, pol.horizon_s, pol.min_dwell_s, pol.cooldown_s,
            pol.max_switches, pol.switch_window_s) == (4, 30.0, 900.0, 60.0, 90.0, 3, 600.0)
    for r in (hook.recommender, hook.auto.recommender):
        assert (r.ttl_s, r.min_windows, r.efficiency_lower) == (120.0, 2, 0.5)
