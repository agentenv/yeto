"""fleet-dashboard 9.9: our own stop of the app shows "已停止", not RECOVERY_REQUIRED."""

import json
import os

from yeto.dashboard.reducer import Reducer
from yeto.dashboard.sources import load_all, operator_stops_near, stop_records_in

from dashboard_helpers import T0

STAMP = "2026-10-08T08:42:04Z"
STOP_TS = 1791448924.0  # == STAMP


def _run(tmp_path, rr_ts, stop_line=f"{STAMP} gate\n", legacy=None):
    run = tmp_path / "s1" / "run-x"
    tape = run / "tape-direct" / "yeto-run-x" / "l0" / "rank0"
    tape.mkdir(parents=True)
    rows = [{"event": "rl_driver_start", "island_id": 0, "time_unix": STOP_TS - 3000},
            {"event": "rl_heartbeat", "island_id": 0, "time_unix": STOP_TS - 30},
            {"event": "rl_reconfiguration", "island_id": 0, "time_unix": rr_ts, "result": "RECOVERY_REQUIRED",
             "cause": "node_lost: 1 of 2 island nodes alive"}]
    (tape / "rl-island-0.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    if stop_line is not None:
        (run / "STOP_ISSUED_utc.txt").write_text(stop_line)
    if legacy:
        (run / legacy).write_text("")
        os.utime(run / legacy, (STOP_TS, STOP_TS))
    return run, tape


def _ov(paths):
    r = Reducer()
    load_all(r, paths)
    return r.overview(now=STOP_TS + 600)


def test_failure_after_our_stop_is_stopped_not_recovery(tmp_path):
    run, tape = _run(tmp_path, STOP_TS + 4)
    ov = _ov([str(tape)])
    assert ov["islands"][0]["status"] == "stopped" and ov["islands"][0]["stopped_by_us"]
    assert not [a for a in ov["alerts"] if a["rule"] in ("recovery_required", "heartbeat")]
    assert ov["global_status"]["text"].startswith("已停止") and ov["operator_stop"]["cause"] == "gate"


def test_failure_before_our_stop_stays_recovery(tmp_path):
    run, tape = _run(tmp_path, STOP_TS - 60)
    ov = _ov([str(tape)])
    assert ov["islands"][0]["status"] == "recovery"
    assert [a for a in ov["alerts"] if a["rule"] == "recovery_required"]


def test_no_stop_marker_keeps_recovery(tmp_path):
    run, tape = _run(tmp_path, STOP_TS + 4, stop_line=None)
    assert _ov([str(tape)])["islands"][0]["status"] == "recovery"


def test_legacy_marker_uses_mtime_upper_bound(tmp_path):
    run, tape = _run(tmp_path, STOP_TS + 4, stop_line=None, legacy="GATE_STOPPED")
    recs = stop_records_in(run)
    assert recs[0]["time_unix"] == STOP_TS and recs[0]["marker"] == "GATE_STOPPED (mtime)"
    assert _ov([str(tape)])["islands"][0]["status"] == "stopped"


def test_stop_issued_file_wins_over_legacy(tmp_path):
    run, _ = _run(tmp_path, STOP_TS + 4, stop_line="2026-10-08T08:41:00Z watchdog\n", legacy="GATE_STOPPED")
    assert [r["cause"] for r in stop_records_in(run)] == ["watchdog"]


def test_operator_stop_records_deduplicated(tmp_path):
    run, tape = _run(tmp_path, STOP_TS + 4)
    r = Reducer()
    for rec in operator_stops_near([str(tape)]) + operator_stops_near([str(run)]):
        r.feed_operator_stop(rec)
    assert len(r.operator_stops) == 1
