"""fleet-dashboard 3.1/3.2/5.3: reducer views over real and constructed tapes."""

import json

from yeto.dashboard.reducer import Reducer
from yeto.dashboard.sources import TapeSource, load_all

from dashboard_helpers import FX, T0, four_island_reducer, local_round


def real_reducer():
    r = Reducer(run="s9-m4x1")
    load_all(r, [str(FX / "s9-m4x1")])
    return r


def test_real_ports_tape_views():
    r = real_reducer()
    o = r.overview()
    assert r.counts == {"learner": 43, "syncer": 0, "journal": 4, "fleet": 0, "other": 0}
    (card,) = o["islands"]
    assert card["id"] == "0" and card["round"] == 2 and card["policy_version"] == 2
    assert card["status"] == "recovery"
    s = o["series"]["0"]
    assert [p[0] for p in s["reward"]] == [1, 2] and len(s["grad_norm"]) == 2
    assert s["loss"] == [] and s["pg_loss"] == []  # None on the ports path -> no points (无数据)
    assert s["tok_s"]  # derived from action_tokens / (rollout+train seconds)
    assert card["heartbeat_seen"] is False and card["gpu_util_pct"] is None
    assert card["staleness"] is None  # no syncer tape


def test_real_journal_feeds_cells_and_e1_panel_and_severe_alert():
    r = real_reducer()
    v = r.island_view("0")
    assert [c["role"] for c in v["cells"]] == ["trainer", "rollout"]
    assert v["cells_source"].startswith("journal gpu_pool")
    (tx,) = v["transactions"]
    assert tx["result"] == "RECOVERY_REQUIRED" and tx["scope"] == "island"
    alerts = r.overview()["alerts"]
    assert alerts[0]["sev"] == 0 and alerts[0]["rule"] == "recovery_required"
    assert alerts[0]["island"] == "0"


def test_e1_transaction_lifecycle_from_journal_records():
    r = Reducer()
    j = [
        {"seq": 1, "kind": "request", "tx_id": "rc-1", "request_id": "q1", "body": {"kind": "rollout-only"},
         "wall_time": T0},
        {"seq": 2, "kind": "phase", "tx_id": "rc-1", "phase": "QUIESCING", "wall_time": T0 + 1},
        {"seq": 3, "kind": "phase", "tx_id": "rc-1", "phase": "COMMITTED", "wall_time": T0 + 2},
        {"seq": 4, "kind": "phase", "tx_id": "rc-1", "phase": "SUCCEEDED", "wall_time": T0 + 3},
        {"seq": 5, "kind": "request", "tx_id": "rc-2", "request_id": "q2", "body": {"kind": "trainer-dp"},
         "wall_time": T0 + 4},
        {"seq": 6, "kind": "phase", "tx_id": "rc-2", "phase": "RECOVERY_REQUIRED", "scope": "request",
         "error": "drain timeout", "wall_time": T0 + 5},
    ]
    for rec in j:
        r.feed(rec, island="2")
    txs = {t["tx_id"]: t for t in r.island_view("2")["transactions"]}
    assert txs["rc-1"]["phases"] == ["QUIESCING", "COMMITTED", "SUCCEEDED"]
    assert txs["rc-1"]["result"] == "SUCCEEDED" and txs["rc-1"]["kind"] == "rollout-only"
    assert txs["rc-2"]["result"] == "RECOVERY_REQUIRED" and txs["rc-2"]["error"] == "drain timeout"
    assert any(a["rule"] == "recovery_required" and a["sev"] == 0 for a in r.overview()["alerts"])


def test_cell_snapshot_overrides_derived_cells():
    r = Reducer()
    r.feed({"seq": 1, "kind": "gpu_pool", "roles": {"GPU-a": "trainer"}, "accepted": True}, island="0")
    r.feed({"event": "rl_cell_snapshot", "island_id": 0, "time_unix": T0,
            "cells": [{"cell_id": "c0", "role": "trainer", "gpus": 4, "state": "running"}]})
    v = r.island_view("0")
    assert v["cells"] == [{"cell": "c0", "role": "trainer", "gpus": 4, "state": "running"}]
    assert v["cells_source"] == "rl_cell_snapshot"


def test_refeeding_the_same_offsets_does_not_double_count(tmp_path):
    r = Reducer()
    tape = FX / "s9-m4x1" / "rl-island-0.jsonl"
    TapeSource(tape).pump(r)
    before = (r.event_seq, json.dumps(r.overview(now=T0), sort_keys=True, default=str))
    TapeSource(tape).pump(r)  # a fresh source restarts at offset 0
    assert (r.event_seq, json.dumps(r.overview(now=T0), sort_keys=True, default=str)) == before


def test_incremental_tail_only_consumes_complete_lines(tmp_path):
    tape = tmp_path / "rl-island-5.jsonl"
    r = Reducer()
    src = TapeSource(tape)
    tape.write_text(json.dumps(local_round(5, 1, T0)) + "\n" + '{"event": "rl_local_ro')
    assert src.pump(r) == 1
    with open(tape, "a") as fh:
        fh.write('und", "island_id": 5, "local_round_id": 2, "time_unix": %s}\n' % (T0 + 1))
    assert src.pump(r) == 1
    assert r.islands["5"]["round"] == 2


def test_unknown_events_pass_through_and_missing_fields_are_none():
    r = Reducer()
    r.feed({"event": "rl_something_new", "island_id": 0, "time_unix": T0, "x": 1})
    r.feed({"weird": True})
    r.feed({"event": "rl_local_round", "island_id": 0, "local_round_id": 1, "time_unix": T0})
    ev = r.events_view()["events"]
    assert [e["type"] for e in ev] == ["rl_something_new", "unknown", "rl_local_round"]
    card = r.overview()["islands"][0]
    assert card["reward"] is None and card["tok_s"] is None and card["heartbeat_age_s"] is None
    assert all(v == [] for v in r.overview()["series"]["0"].values())


def test_round_view_one_island_did_not_push_and_resend():
    rows = four_island_reducer().rounds()
    assert [(x["round"], x["responded"], x["expected"]) for x in rows] == [(1, 4, 4), (2, 3, 4), (3, 4, 4)]
    assert rows[1]["missed"] == ["3"] and rows[1]["bad"] and rows[1]["pushed"] == ["0", "1", "2"]
    assert rows[2]["resend"] == 1 and rows[2]["bad"] and rows[2]["missed"] == []
    assert not rows[0]["bad"] and all(x["derived"] for x in rows)
    assert rows[0]["merge_ms"] == 120.0 and rows[0]["sync_ms"] == 300


def test_round_view_without_expected_falls_back_to_pushes():
    r = Reducer()
    for i in (0, 1):
        r.feed({"event": "rl_fragment_push", "island_id": i, "global_step": 1, "time_unix": T0})
    r.feed({"event": "rl_local_round", "island_id": 2, "local_round_id": 1, "time_unix": T0})
    r.feed({"step": 1, "fragment": 0, "responders": [], "sync/responders": 2, "time_unix": T0 + 1})
    (row,) = r.rounds()
    assert row["expected"] == 3 and row["responded"] == 2 and row["missed"] == ["2"]
    assert row["quorum_ms"] is None  # old syncer record: 无数据, not 0


def test_events_endpoint_filters_and_cursor():
    r = four_island_reducer()
    page = r.events_view(island="1", type="rl_fragment_push", limit=2)
    assert len(page["events"]) == 2 and page["more"]
    nxt = r.events_view(island="1", type="rl_fragment_push", after=page["cursor"], limit=2)
    assert len(nxt["events"]) == 1 and not nxt["more"]


def test_ray_embed_per_island_kind():
    r = Reducer()
    for iid, name, cloud in ((0, "run-l0-modal", "modal"), (1, "run-l1-eu", "nebius"),
                             (2, "lab", "local")):
        r.feed({"event": "island_ready", "island": name, "island_id": iid, "cloud": cloud,
                "gpu": "H100", "gpus": 8, "time_unix": T0})
    r.feed({"event": "rl_local_round", "island_id": 3, "local_round_id": 1, "time_unix": T0})
    r.feed({"event": "rl_resource_sample", "island_id": 0, "time_unix": T0,
            "gpus": [{"index": 0, "util_pct": 80, "mem_used_mb": 40, "mem_total_mb": 80}]})
    assert r.island_view("0")["ray_embed"]["mode"] == "none"
    assert r.island_view("0")["card"]["gpu_util_pct"] == 80 and r.island_view("0")["card"]["mem_pct"] == 50
    tun = r.island_view("1")["ray_embed"]
    assert tun == {"mode": "tunnel", "port": 18266, "command": "ssh -L 18266:localhost:8265 run-l1-eu"}
    assert r.island_view("2")["ray_embed"]["mode"] == "direct"
    assert r.island_view("3")["ray_embed"]["mode"] == "unknown"
    r.feed({"event": "rl_resource_sample", "island_id": 1, "time_unix": T0, "available": False})
    assert r.island_view("1")["resource"]["available"] is False
    assert r.island_view("1")["card"]["gpu_util_pct"] is None
