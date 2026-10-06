"""fleet-dashboard 3.3/5.3: every alert rule fires and stays quiet."""

import pytest

from yeto.dashboard.reducer import Reducer

from dashboard_helpers import T0, four_island_reducer, local_round, merge


def rules(r, now=None):
    return [(a["rule"], a["sev"], a["island"]) for a in r.overview(now=now)["alerts"]]


def test_heartbeat_thresholds():
    r = Reducer()
    r.feed({"event": "rl_heartbeat", "island_id": 0, "time_unix": T0, "phase": "train"})
    assert rules(r, now=T0 + 30) == []
    assert rules(r, now=T0 + 61) == [("heartbeat", 1, "0")]
    assert rules(r, now=T0 + 301) == [("heartbeat", 0, "0")]
    assert r.overview(now=T0 + 301)["islands"][0]["status"] == "stale"


def test_finalized_island_has_no_heartbeat_alert():
    r = Reducer()
    r.feed({"event": "rl_learner_finalized", "island_id": 0, "time_unix": T0})
    assert rules(r, now=T0 + 3600) == []


def test_consecutive_missed():
    r = Reducer()
    for step in range(1, 7):
        r.feed(merge(step, [0, 1], [0] if step >= 3 else [0, 1], t=T0 + step))
    r.feed({"event": "rl_heartbeat", "island_id": 1, "time_unix": T0 + 6})
    r.feed({"event": "rl_heartbeat", "island_id": 0, "time_unix": T0 + 6})
    got = [a for a in r.overview()["alerts"] if a["rule"] == "missed"]
    assert [(a["sev"], a["island"], a["round"]) for a in got] == [(1, "1", 3)]
    r.feed(merge(7, [0, 1], [0], t=T0 + 7))
    got = [a for a in r.overview()["alerts"] if a["rule"] == "missed"]
    assert got[0]["sev"] == 0
    single = Reducer()
    single.feed(merge(1, [0, 1], [0], t=T0))
    assert [a for a in single.overview()["alerts"] if a["rule"] == "missed"] == []


def test_quorum_latency_regression():
    r = Reducer()
    for step in range(1, 11):
        r.feed(merge(step, [0], [0], quorum_ms=1000, t=T0 + step))
    assert "quorum" not in [a[0] for a in rules(r)]
    for step in range(11, 18):
        r.feed(merge(step, [0], [0], quorum_ms=16000, t=T0 + step))
    (a,) = [a for a in r.overview()["alerts"] if a["rule"] == "quorum"]
    assert a["sev"] == 1 and a["round"] == 17


def test_grad_norm_spike_and_nan():
    r = Reducer()
    for x in range(1, 12):
        r.feed(local_round(0, x, T0 + x, grad_norm=6.8 if x == 10 else 0.8))
    got = [(a["rule"], a["round"], a["metric"]) for a in r.overview()["alerts"]]
    assert ("grad_norm_spike", 10, "grad_norm") in got and ("grad_norm_nan", 10, "grad_norm") not in got
    r.feed(local_round(0, 12, T0 + 12, grad_norm=float("nan")))
    assert any(a["rule"] == "grad_norm_nan" and a["sev"] == 0 and a["round"] == 12
               for a in r.overview()["alerts"])
    quiet = Reducer()
    for x in range(1, 12):
        quiet.feed(local_round(0, x, T0 + x))
    assert not [a for a in quiet.overview()["alerts"] if a["rule"].startswith("grad_norm")]


def test_clipfrac_and_kl_thresholds_are_configurable():
    r = Reducer(thresholds={"kl_warn": 0.01})
    r.feed(local_round(0, 1, T0, clip_fraction=0.25, mean_kl=0.02))
    got = {a["rule"]: a["sev"] for a in r.overview()["alerts"]}
    assert got == {"clipfrac": 2, "kl": 1}
    r2 = Reducer()
    r2.feed(local_round(0, 1, T0, clip_fraction=0.1, mean_kl=0.02))
    assert r2.overview()["alerts"] == []
    with pytest.raises(ValueError):
        Reducer(thresholds={"nope": 1})


@pytest.mark.parametrize("hours,sev", [(4.0, None), (5.5, 1), (7.0, 0)])
def test_budget_thresholds(hours, sev):
    r = Reducer(budget_usd=100.0, prices={"prices": {"nebius:H200": 4.0}})
    r.feed({"event": "island_ready", "island": "x-l0-eu", "island_id": 0, "cloud": "nebius",
            "gpu": "H200", "gpus": 3, "time_unix": T0})
    got = [a for a in r.overview(now=T0 + hours * 3600)["alerts"] if a["rule"] == "budget"]
    assert [a["sev"] for a in got] == ([] if sev is None else [sev])


def test_alerts_sorted_by_severity_and_locatable():
    r = four_island_reducer()
    r.feed({"event": "rl_reconfiguration", "island_id": 3, "result": "RECOVERY_REQUIRED",
            "rollout_id": 2, "time_unix": T0 + 40, "error": "node_lost"})
    alerts = r.overview(now=T0 + 41)["alerts"]
    assert [a["sev"] for a in alerts] == sorted(a["sev"] for a in alerts)
    assert alerts[0]["rule"] == "recovery_required" and alerts[0]["island"] == "3"
    assert all({"island", "round", "metric", "id"} <= set(a) for a in alerts)
