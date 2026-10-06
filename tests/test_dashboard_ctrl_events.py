"""dashboard 5.1/5.2: reducer prefers controller events, falls back to journal records."""
from __future__ import annotations

from yeto.dashboard.reducer import Reducer

T0 = 1_790_000_000.0


def _j(seq, kind, **kw):
    return {"seq": seq, "kind": kind, "wall_time": T0 + seq, **kw}


def _ev(seq, tx, phase, result=None, reason=None):
    return _j(seq, "rl_reconfig_phase", tx_id=tx, txn_id=tx, request_id="q", source="A",
              target="B", phase=phase, result=result, t=T0 + seq, expected_epoch=3,
              config_epoch=3, reason=reason)


def test_journaled_events_own_the_transaction_no_duplicates():
    r = Reducer()
    recs = [_j(1, "request", tx_id="rc-1", request_id="q", body={"kind": "rollout-only"}),
            _j(2, "phase", tx_id="rc-1", phase="VALIDATING"), _ev(3, "rc-1", "VALIDATING"),
            _j(4, "phase", tx_id="rc-1", phase="COMMITTED"), _ev(5, "rc-1", "COMMITTED", "COMMITTED"),
            _j(6, "phase", tx_id="rc-1", phase="RECOVERY_REQUIRED", error="boom"),
            _ev(7, "rc-1", "RECOVERY_REQUIRED", "RECOVERY_REQUIRED", "boom"),
            _j(8, "gpu_pool", roles={"GPU-a": "trainer"}, accepted=True),
            _j(9, "rl_cell_snapshot", txn_id="rc-1", cells=[
                {"cell_id": "GPU-a", "role": "trainer", "node": 0, "gpu_uuid": "GPU-a", "gpus": 1,
                 "state": "recovery_required", "config": "B", "epoch": 3}])]
    for x in recs:
        r.feed(x, island="1")
    v = r.island_view("1")
    (tx,) = v["transactions"]
    assert tx["phases"] == ["VALIDATING", "COMMITTED", "RECOVERY_REQUIRED"]
    assert tx["result"] == "RECOVERY_REQUIRED" and tx["error"] == "boom"
    assert tx["source_config"] == "A" and tx["target_config"] == "B" and tx["expected_epoch"] == 3
    assert len(v["recovery_required"]) == 1
    assert any(a["rule"] == "recovery_required" and a["sev"] == 0 for a in r.overview()["alerts"])
    assert v["cells_source"] == "rl_cell_snapshot" and v["cells"][0]["gpu_uuid"] == "GPU-a"


def test_tape_events_and_legacy_fallback():
    r = Reducer()
    r.feed({"event": "rl_reconfig_phase", "island_id": 0, "time_unix": T0, "txn_id": "t9",
            "phase": "SUCCEEDED", "result": "SUCCEEDED"})
    assert r.island_view("0")["transactions"][0]["result"] == "SUCCEEDED"
    old = Reducer()  # old tape: no new events -> journal-derived path
    old.feed(_j(1, "phase", tx_id="rc-1", phase="COMMITTED"), island="2")
    old.feed(_j(2, "gpu_pool", roles={"GPU-a": "trainer"}, accepted=True), island="2")
    v = old.island_view("2")
    assert v["transactions"][0]["phases"] == ["COMMITTED"]
    assert v["cells_source"].startswith("journal gpu_pool")
