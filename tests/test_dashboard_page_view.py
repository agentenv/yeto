"""fleet-dashboard 9.3-9.5: page_view round records, run_kind, per-node series."""

from yeto.dashboard.reducer import Reducer

from dashboard_helpers import T0


def _span(task, rid, a, b, mono0=1000.0):
    # driver monotonic seconds; time_unix written at span end
    return {"event": "rl_timeline_span", "island_id": 0, "task": task, "rollout_id": rid,
            "start": mono0 + a, "end": mono0 + b, "time_unix": T0 + b}


def _single():
    r = Reducer()
    r.feed({"event": "rl_driver_start", "island_id": 0, "time_unix": T0})
    for rec in (_span("publish", None, 0, 240), _span("generate", 0, 240, 411), _span("train", 0, 411, 1027),
                _span("outer_sync", 0, 1027, 1243), _span("publish", None, 1243, 1486),
                _span("generate", 1, 1486, 1637)):
        r.feed(rec)
    r.feed({"event": "rl_publication", "island_id": 0, "policy_version": 1, "time_unix": T0 + 1486})
    r.feed({"event": "rl_round_trained", "island_id": 0, "rollout_id": 0, "time_unix": T0 + 1027,
            "truncated_frac": 0.5625, "reward_p10": 0.0, "reward_p50": 1.0, "reward_p90": 1.0,
            "resp_len_mean": 5462.25, "resp_len_p95": 8192.0,
            "train_metrics": {"train_rollout_logprob_abs_diff": 0.0232}})
    r.feed({"event": "rl_local_round", "island_id": 0, "local_round_id": 1, "reward_mean": 0.546875,
            "time_unix": T0 + 1243})
    return r


def test_round_records_phases_and_metrics():
    v = _single().page_view(now=T0 + 2000)
    assert v["overview"]["run_kind"] == "single_island"
    rr = v["islands"]["0"]["rounds"]
    r0 = rr[0]
    assert r0["dur"] == {"R": 171.0, "T": 616.0, "S": 216.0, "P": 243.0}  # publish after sync -> v1
    assert r0["published_version"] == 1 and r0["trunc"] == 0.5625 and r0["logprob_diff"] == 0.0232
    assert r0["p50"] == 1.0 and r0["resp_p95"] == 8192.0 and r0["reward"] == 0.546875
    assert rr[1]["dur"] == {"R": 151.0} and rr[1]["published_version"] is None and rr[1]["trunc"] is None


def test_unused_panels_and_node_series():
    r = _single()
    for i in range(3):
        r.feed({"event": "modal_host_sample", "time_unix": T0 + i, "gpu_mem_used_mib": [2048, 4096],
                "gpu_util_pct": [10, 30]}, island="0", node="1")
    v = r.page_view(now=T0 + 10)
    assert v["usage"] == {"syncer": False, "journal": False, "cells": False, "transactions": False}
    assert v["islands"]["0"]["node_series"]["1"][0] == [T0, 3.0, 20.0]


def test_multi_island_run_kind():
    r = _single()
    r.feed({"event": "rl_driver_start", "island_id": 1, "time_unix": T0})
    assert r.page_view(now=T0)["overview"]["run_kind"] == "multi_island"
