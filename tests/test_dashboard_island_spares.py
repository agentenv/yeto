"""A resident ssh: trainer island with spare nodes on the dashboard (own-cluster 4.5): its nodes'
health reasons, its ``waiting_spare`` state, its replacements (count, last time) and the spare-node
events (``island_node_lost`` / ``island_waiting_spare`` / ``island_node_replaced`` /
``island_replacement_limit``) in the event stream. The island's record (``spares`` / ``health`` in
``fleet/trainers/<k>/state.json``) wins over its fleet.jsonl events; an island without spare nodes or
health checks (every customer run) carries no new field at all."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from test_dashboard_diloco import SMOKE
from test_dashboard_fleet_islands import T0, fleet_reducer, plan_doc

from yeto.dashboard import fleet_islands
from yeto.dashboard.export import export_html
from yeto.dashboard.reducer import Reducer
from yeto.dashboard.sources import load_all
from yeto.launch import island_spares as isp
from yeto.rl.engine.object_store import DirectoryObjectStore

LEGEND = 'legend: mk("topoLg").innerHTML'
# the topology smoke, also printing the event stream, trainer 1's tip (a hover on its card) and the dashed boxes
# (each colour reads as its CSS variable's name: --warn, --bad, ...)
EV_SMOKE = "globalThis.getComputedStyle = () => ({ getPropertyValue: (n) => n });\n" + SMOKE.replace(LEGEND, LEGEND + ''', events: mk("evs").innerHTML,
  tip: (function () { const h = Object.values(els).find((n) => n.listeners && n.listeners.mousemove);
    if (!h) return null; h.listeners.mousemove[0]({ target: { closest: () => ({ getAttribute: () => "trn:1" }) },
      clientX: 0, clientY: 0 }); return mk("topoTip").innerHTML; })(),
  dashed: svg.filter((n) => n.style && n.style.strokeDasharray === "5 4").length,
  roster: (svg.find((n) => /^roster /.test(String(n.textContent || ""))) || { style: {} }).style.fill || null''')
assert EV_SMOKE != SMOKE
FX = Path(__file__).parent / "fixtures" / "dashboard"
NEW_KEYS = ("spares", "node_health")
WAIT = "fabric group rack-b has no free healthy spare node (gpu-e: lease held by run other)"
LOST = "rank 1 (gpu-d) node health bad in 2 checks in a row: GPU 3 XID 79"


def ev(kind: str, at: float, k: int = 1, **fields) -> dict:
    return {"event": kind, "time_unix": at, "island": f"fleet-qa-l{k}-modal", "island_id": k, **fields}


def lost(at: float) -> dict:
    return ev(isp.NODE_LOST, at, node="gpu-d", rank=1, incarnation=0, reason=LOST)


def waiting(at: float, elapsed: float = 0, **kw) -> dict:
    return ev(isp.WAITING_SPARE, at, lost=["gpu-d"], reason=WAIT, group="rack-b", next_check_s=60,
              elapsed_s=elapsed, **kw)


def replaced(at: float) -> dict:
    return ev(isp.NODE_REPLACED, at, rank=1, lost="gpu-d", spare="gpu-e", incarnation=1, reason=LOST,
              resume_step=6, group="rack-b")


def reducer(*extra: dict) -> Reducer:
    r = fleet_reducer()
    for rec in extra:
        r.feed(rec)
    return r


def cards(r: Reducer, now: float | None = None) -> dict:
    return {c["id"]: c for c in r.overview(now=r.max_ts if now is None else now)["islands"]}


def spares_doc(**kw) -> dict:
    return {"nodes": ["gpu-c", "gpu-d"], "spare_nodes": ["gpu-e"], "lost": [], "waiting_spare": False,
            "wait_reason": None, "waiting_since": None, "replacements": 0, "replacements_window": 0,
            "max_replacements": 3, "last_replaced_at": None, **kw}


def node(verdict: str, at: float, reasons=(), bad=0) -> dict:
    return {"verdict": verdict, "ok": verdict == "ok", "reasons": list(reasons), "bad_in_row": bad, "at": at,
            "every_s": 60.0, "multi": True}


def store_state(tmp_path: Path, life: str, **fields) -> dict:
    """The fleet state of a run store whose island 1 record is ``life`` with ``fields``."""
    for k, doc in {0: {"state": "running"}, 1: {"state": life, "relaunches": 1, "incarnation": 1, **fields}}.items():
        p = tmp_path / "store" / f"fleet/trainers/{k}/state.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"since": T0, "handle": {"cluster": f"fleet-qa-l{k}-modal"}, **doc}))
    return fleet_islands.read_fleet_state(DirectoryObjectStore(str(tmp_path / "store")))


def smoke(tmp: Path, r: Reducer, width: int = 1200) -> dict:
    exe = shutil.which("node")
    if exe is None:
        pytest.skip("node not installed")
    tmp.mkdir(parents=True, exist_ok=True)
    page = export_html(r, tmp / "p.html", generated_at=r.max_ts)
    (tmp / "smoke.js").write_text(EV_SMOKE)
    res = subprocess.run([exe, str(tmp / "smoke.js"), str(page), str(width)], capture_output=True, text=True,
                         timeout=60, check=False)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


# -- the reducer ---------------------------------------------------------------------------------
def test_spare_and_health_events_are_no_lifecycle_state_nor_a_sign_of_life():
    end = reducer().max_ts
    r = reducer(ev("node_health", end + 4000, node="gpu-d", rank=1, verdict="lost", reasons=["GPU 3 XID 79"]),
                lost(end + 4001), waiting(end + 4002))
    assert r.islands["1"]["last_event_ts"] == reducer().islands["1"]["last_event_ts"]  # not a sign of life
    assert r.islands["1"]["fleet_state"] == reducer().islands["1"]["fleet_state"]  # not a lifecycle state
    plain = cards(reducer(), now=end + 4002)["1"]["status"]
    assert plain in ("stale", "lost") and cards(r, now=end + 4002)["1"]["status"] == plain
    marks = ("node_health", isp.NODE_LOST, isp.WAITING_SPARE)
    assert all(e["stream"] == "fleet" for e in r.events if e["type"] in marks)


def test_the_events_mark_the_card_without_a_record():
    r = reducer(lost(T0 + 400), waiting(T0 + 460, elapsed=30))
    sp = cards(r)["1"]["spares"]
    assert sp["waiting"] and sp["since"] == T0 + 430 and sp["reason"] == WAIT and sp["source"] == "events"
    assert sp["lost"] == ["gpu-d"] and sp["replacements"] == 0
    r.feed(waiting(T0 + 520, elapsed=90))  # a later report keeps the wait's start
    assert cards(r)["1"]["spares"]["since"] == T0 + 430
    r.feed(waiting(T0 + 560))  # a report without its elapsed time (it would start the wait at T0 + 560)
    assert cards(r)["1"]["spares"]["since"] == T0 + 430  # the earliest start stays
    r.feed(replaced(T0 + 600))
    sp = cards(r)["1"]["spares"]
    assert not sp["waiting"] and sp["since"] is None and sp["replacements"] == 1 and sp["last_at"] == T0 + 600
    assert sp["last"]["lost"] == "gpu-d" and sp["last"]["spare"] == "gpu-e" and sp["lost"] == []
    assert "spares" not in cards(r)["0"]  # the other island: no field


def limit(at: float) -> dict:
    return ev(isp.REPLACEMENT_LIMIT, at, replaced=[], lost=["gpu-d"], reason="3 node replacement(s)")


@pytest.mark.parametrize("final", [limit(T0 + 700), waiting(T0 + 700, elapsed=86400, given_up=True)])
def test_a_limit_or_a_given_up_wait_ends_the_wait(final):
    sp = cards(reducer(lost(T0 + 400), waiting(T0 + 401), final))["1"]["spares"]
    assert not sp["waiting"] and sp["stopped"]


# -- replay order: by time across sources, in write order within one source ----------------------------
def _spares_of(rows) -> dict:
    """The events-only ``spares`` mark of island 1 after feeding ``rows`` ([(source, record)])."""
    r = reducer()
    for source, rec in rows:
        assert r.feed(rec, source=source)
    return cards(r)["1"]["spares"]


HOST_A = [("host-a", lost(T0 + 400)), ("host-a", waiting(T0 + 401, elapsed=1))]  # the lost host's launcher
HOST_B = [("host-b", replaced(T0 + 900))]  # the new host's launcher: a spare took gpu-d's place


@pytest.mark.parametrize("rows", [HOST_A + HOST_B, HOST_B + HOST_A, [HOST_A[0], HOST_B[0], HOST_A[1]]],
                         ids=["in-order", "new-host-first", "interleaved"])
def test_a_lost_host_s_records_read_late_replay_in_place(rows):
    sp = _spares_of(rows)
    assert sp == _spares_of(HOST_A + HOST_B)
    assert not sp["waiting"] and sp["since"] is None and sp["lost"] == [] and sp["replacements"] == 1
    assert sp["last"]["spare"] == "gpu-e" and sp["last_at"] == T0 + 900


def test_one_source_keeps_its_write_order_over_an_older_time():
    """A launcher whose clock stepped back: its replacement line (written after the wait) has an
    older time; within its own source the write order decides: the island is not waiting."""
    sp = _spares_of([("host-a", lost(T0 + 400)), ("host-a", waiting(T0 + 500)), ("host-a", replaced(T0 + 450))])
    assert not sp["waiting"] and sp["replacements"] == 1 and sp["last_at"] == T0 + 450  # the record's own time
    # another source's later line still goes after it by time
    sp = _spares_of([("host-a", lost(T0 + 400)), ("host-a", waiting(T0 + 500)), ("host-a", replaced(T0 + 450)),
                     ("host-b", waiting(T0 + 480))])
    assert sp["waiting"] is False  # host-a's replacement replays at 500 (after its wait), host-b's wait at 480


def test_an_untimed_line_keeps_the_time_of_the_line_before_it():
    untimed = {k: v for k, v in replaced(0).items() if k != "time_unix"}
    sp = _spares_of([("host-a", lost(T0 + 400)), ("host-a", untimed), ("host-b", waiting(T0 + 300))])
    assert not sp["waiting"] and sp["replacements"] == 1 and sp["last_at"] is None  # host-b's wait is older


def test_a_spare_event_sets_no_shape_field_nor_state():
    base = reducer().islands["1"]
    r = reducer(dict(lost(T0 + 10), island="other-name", gpu="H100", cloud="nowhere"),
                dict(replaced(T0 + 5), gpu="H100"))  # the second arrives older: a replay, still no lifecycle
    isl = r.islands["1"]
    for key in ("name", "gpu", "cloud", "fleet_state", "ready_ts", "lost_ts", "stop_ts", "open_ts", "closed_s"):
        assert isl[key] == base[key], key
    assert r.fleet_spares_log["1"] and "1" in r.fleet_spares_src_t


def test_a_health_event_is_no_spare_record():
    r = reducer(ev("node_health", T0 + 10, node="gpu-d", verdict="lost", reasons=["GPU 3 XID 79"]))
    assert "spares" not in r.islands["1"] and "1" not in r.fleet_spares_log
    assert any(e["type"] == "node_health" for e in r.events)


def test_the_record_wins_over_the_events(tmp_path):
    r = reducer(lost(T0 + 400), waiting(T0 + 401), replaced(T0 + 450))
    st = store_state(tmp_path, "recovering", spares=spares_doc(lost=["gpu-d"], waiting_spare=True,
                                                                wait_reason=WAIT, waiting_since=T0 + 500))
    assert r.load_fleet_state(st)
    c = cards(r)["1"]
    sp = c["spares"]
    assert sp["source"] == "record" and sp["waiting"] and sp["since"] == T0 + 500 and sp["replacements"] == 0
    assert sp["last"] is None  # no replacement on the record: no names from the events either
    assert c["status"] == "starting" and c["island_now"]["reason"] == "waiting for a spare node"
    assert c["island_now"]["detail"] == WAIT


def test_the_record_s_replacements_take_the_names_of_the_last_event(tmp_path):
    r = reducer(lost(T0 + 400), replaced(T0 + 450))
    st = store_state(tmp_path, "running", spares=spares_doc(nodes=["gpu-c", "gpu-e"], spare_nodes=["gpu-d"],
                                                             replacements=1, replacements_window=1,
                                                             last_replaced_at=T0 + 449))
    assert r.load_fleet_state(st)
    sp = cards(r)["1"]["spares"]
    assert (sp["replacements"], sp["window"], sp["max"], sp["last_at"]) == (1, 1, 3, T0 + 449)
    assert sp["last"]["spare"] == "gpu-e"


def test_island_state_of_an_island_that_lost_a_node():
    rec = {"life_state": "recovering", "relaunches": 1, "incarnation": 1,
           "spares": fleet_islands.spares_record(spares_doc(lost=["gpu-d"]))}
    assert fleet_islands.island_state(rec) == {"state": "starting", "reason": "replacing lost node gpu-d",
                                               "detail": "new launch recovering · incarnation 1"}
    rec["spares"]["waiting_spare"], rec["spares"]["wait_reason"] = True, WAIT
    assert fleet_islands.island_state(rec)["reason"] == "waiting for a spare node"
    rec["life_state"] = "running"  # replaced and running: the spares say nothing of its state
    assert fleet_islands.island_state(rec) == {"state": "running", "reason": None, "detail": None}


def test_the_unhealthy_nodes_of_the_record(tmp_path):
    r = reducer()
    now = r.max_ts
    health = {"verdict": "lost", "nodes": {"gpu-c": node("ok", now - 900),
                                           "gpu-d": node("lost", now - 900, ["GPU 3 XID 79"], 2)}}
    # the record is not written again while no verdict changes: a check as old as the record is current
    assert r.load_fleet_state(store_state(tmp_path, "recovering", health=health, updated=now - 890))
    assert cards(r)["1"]["node_health"] == [{"node": "gpu-d", "verdict": "lost", "reasons": ["GPU 3 XID 79"],
                                             "bad_in_row": 2}]
    assert "spares" not in cards(r)["1"]  # health checks without spare nodes


def test_a_check_older_than_its_record_reads_stale(tmp_path):
    r = reducer()
    now = r.max_ts
    health = {"verdict": "ok", "nodes": {"gpu-c": node("ok", now - 10), "gpu-d": node("ok", now - 400)}}
    assert r.load_fleet_state(store_state(tmp_path, "running", health=health, updated=now - 10))
    (bad,) = cards(r)["1"]["node_health"]  # gpu-d's checks stopped 390 s before the record: 3 periods are 180 s
    assert bad["node"] == "gpu-d" and bad["verdict"] == "read_failed" and bad["stale"] is True
    assert fleet_islands.stale_health(health["nodes"]["gpu-d"], now - 300) is health["nodes"]["gpu-d"]  # 100 s


@pytest.mark.parametrize("life", ["done", "failed"])
def test_an_ended_island_shows_no_health(tmp_path, life):
    now = reducer().max_ts
    health = {"verdict": "lost", "nodes": {"gpu-d": node("lost", now - 10, ["GPU 3 XID 79"], 2)}}
    r = reducer()
    assert r.load_fleet_state(store_state(tmp_path, life, health=health, updated=now))
    assert "node_health" not in cards(r)["1"]


def test_the_record_fields_read_defensively():
    assert fleet_islands.spares_record(None) is None and fleet_islands.health_record("x") is None
    sp = fleet_islands.spares_record({"waiting_spare": "yes", "replacements": True, "lost": ["a", 3, None]})
    assert sp["waiting_spare"] is False and sp["replacements"] == 0 and sp["lost"] == ["a"]
    many = {"verdict": "lost", "reasons": [f"r{i}" for i in range(9)]}
    h = fleet_islands.health_record({"nodes": {"a": many, "b": 1}})
    assert list(h["nodes"]) == ["a"] and len(h["nodes"]["a"]["reasons"]) == fleet_islands.HEALTH_REASONS_MAX


# -- no new field without spare nodes (every customer run) ---------------------------------------------
def test_no_new_field_without_spare_nodes(tmp_path):
    r = reducer()
    assert r.load_fleet_state(store_state(tmp_path, "running"))
    view = json.dumps(r.page_view(now=r.max_ts), default=str)
    assert not any(f'"{k}"' in view for k in NEW_KEYS)
    st = store_state(tmp_path / "b", "running")
    assert not any(k in (row["record"] or {}) for row in st["trainers"].values() for k in ("spares", "health"))


@pytest.mark.parametrize("fixture", sorted(p.name for p in FX.iterdir() if p.is_dir()))
def test_customer_fixtures_carry_no_new_field(fixture):
    r = Reducer(run="cust")
    load_all(r, [str(p) for p in sorted((FX / fixture).iterdir())])
    view = json.dumps(r.page_view(now=r.max_ts), default=str)
    for key in NEW_KEYS + tuple(isp.__dict__[n] for n in ("NODE_LOST", "NODE_REPLACED", "WAITING_SPARE")):
        assert f'"{key}"' not in view


# -- the page ------------------------------------------------------------------------------------------
def _with_record(tmp_path, life, *extra, **fields):
    r = reducer(*extra)
    assert r.load_fleet_state(store_state(tmp_path / "s", life, **fields))
    return r


def test_a_waiting_island_s_card_badge_phase_and_why(tmp_path):
    end = reducer().max_ts
    r = _with_record(tmp_path, "recovering", lost(end - 100), waiting(end - 95),
                     spares=spares_doc(lost=["gpu-d"], waiting_spare=True, wait_reason=WAIT, waiting_since=end - 95))
    out = smoke(tmp_path, r)
    assert out["dashed"] == 0  # warned, not drawn as a lost trainer
    assert "waiting spare" not in out["texts"]  # the phase line says it: no badge saying it again
    assert "waiting for a spare node · 1m35s" in out["texts"]  # how long, not the record's incarnation
    assert any(t.startswith("gpu-e: lease held by run other · no free spare") for t in out["texts"])  # what to act on first
    # Islands now: the mark in place of the status (no "waiting for a spare node" twice), the reason in the last cell
    assert '<span class="wn" style="white-space:nowrap">waiting spare 1m35s</span>' in out["nodes"]
    assert "waiting for a spare node" not in out["nodes"]
    short = "gpu-e: lease held by run other · no free spare in rack-b"
    assert f'<td class="spn"><div class="ns wt" title="{WAIT}">{short}</div></td>' in out["nodes"]
    assert '<th class="spn">Nodes</th><th>Shard · incarnation</th>' in out["nodes"]  # its own column
    assert "waiting for a spare</span><span>1m35s · " + WAIT in out["tip"]  # the whole reason on a touch page
    assert 'title="wait"' in out["events"] and "<b>island_waiting_spare</b>" in out["events"]


def test_a_wait_known_from_its_events_only(tmp_path):
    end = reducer().max_ts
    r = reducer(lost(end + 10), waiting(end + 20, elapsed=40))
    out = smoke(tmp_path, r)
    assert "waiting for a spare node · 40s" in out["texts"]
    assert "lost" not in out["texts"] and out["dashed"] == 0  # the wait, not a lost trainer
    view = r.page_view(now=r.max_ts)["topology"]["diloco"]
    assert view["roster"] == 1  # out of the syncer's roster while it waits


def test_a_silent_trainer_waiting_for_a_spare_is_not_drawn_lost(tmp_path):
    end = reducer().max_ts  # its tape silent for an hour: lost, unless it waits for a spare node
    gone = smoke(tmp_path / "a", reducer(lost(end + 3590), ev("island_lost", end + 3600, reason=LOST)))
    assert "lost" in gone["texts"] and gone["dashed"] >= 1
    out = smoke(tmp_path / "b", reducer(lost(end + 3590), ev("island_lost", end + 3595, reason=LOST),
                                        waiting(end + 3600, elapsed=10)))
    assert any(t.startswith("waiting for a spare node") for t in out["texts"])
    assert "lost" not in out["texts"] and out["dashed"] == 0


@pytest.mark.parametrize("life", ["done", "failed"])
def test_an_island_that_ended_is_no_longer_waiting(tmp_path, life):
    end = reducer().max_ts
    r = _with_record(tmp_path, life, lost(end - 100), waiting(end - 95),
                     spares=spares_doc(lost=["gpu-d"], waiting_spare=True, wait_reason=WAIT, waiting_since=end - 95))
    out = smoke(tmp_path, r)
    assert "waiting spare" not in out["texts"] and not any("waiting for a spare" in t for t in out["texts"])
    assert "waiting spare" not in out["nodes"]


HELPERS = r"""
const fs = require("fs"), src = fs.readFileSync(process.argv[2], "utf8");
const a = src.indexOf("// a resident ssh: island with spare nodes (tr.spares"), b = src.indexOf("function col(k){");
var D = JSON.parse(process.argv[3]), C = {warn: "W", bad: "B", ink2: "M"};
function fin(x) { return typeof x === "number" && isFinite(x); }
function dur(s) { return Math.round(s) + "s"; }
function spNow() { return 1000; }
function row(k, v) { return k + "=" + v + ";"; }
function col(k) { return k; }
function rich() {}
eval(src.slice(a, b));
const tr = JSON.parse(process.argv[4]);
console.log(JSON.stringify({badge: spareBadge(tr), waiting: waitingSpare(tr), note: spareNote(tr, 1000)}));
"""


def helpers(tmp_path, d: dict, tr: dict) -> dict:
    exe = shutil.which("node")
    if exe is None:
        pytest.skip("node not installed")
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "h.js").write_text(HELPERS)
    src = Path(__file__).parents[1] / "yeto/dashboard/static/topology_diloco.js"
    res = subprocess.run([exe, str(tmp_path / "h.js"), str(src), json.dumps(d), json.dumps(tr)], capture_output=True,
                         text=True, timeout=60, check=False)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


@pytest.mark.parametrize("d, now", [({"done": True}, None), ({"done": False}, {"state": "stopped"}),
                                    ({"done": False}, {"state": "failed"})])
def test_a_finished_run_or_island_has_no_spare_badge(tmp_path, d, now):
    tr = {"island_now": now, "spares": {"waiting": True, "since": 900, "reason": WAIT, "replacements": 0},
          "node_health": [{"node": "gpu-d", "verdict": "lost", "reasons": ["GPU 3 XID 79"]}]}
    out = helpers(tmp_path, d, tr)
    assert out["badge"] is None and out["waiting"] is False and out["note"] is None
    live = helpers(tmp_path / "b", {"done": False}, {**tr, "island_now": {"state": "starting"}})
    assert live["badge"] == ["node ✕", "B"] and live["waiting"] is True  # the wait is no badge: its phase line


def test_a_lost_node_s_health_reason_and_badge(tmp_path):
    now = reducer().max_ts
    health = {"verdict": "lost",
              "nodes": {"gpu-c": node("ok", now - 5), "gpu-d": node("lost", now - 5, ["GPU 3 XID 79"], 2)}}
    r = _with_record(tmp_path, "recovering",
                     ev("node_health", now - 4, node="gpu-d", rank=1, verdict="lost", reasons=["GPU 3 XID 79"]),
                     lost(now - 3), spares=spares_doc(lost=["gpu-d"]), health=health, updated=now - 3)
    out = smoke(tmp_path, r)
    assert "node ✕" in out["texts"] and "gpu-d: GPU 3 XID 79" in out["texts"]
    assert any(t.startswith("replacing lost node gpu-d") for t in out["texts"])
    assert '<span class="bd" style="white-space:nowrap">node ✕</span>' in out["nodes"]
    assert '<td class="spn"><div class="ns bd" title="gpu-d: lost · GPU 3 XID 79">gpu-d: GPU 3 XID 79</div></td>' in out["nodes"]
    assert "node gpu-d</span><span>lost · GPU 3 XID 79" in out["tip"]
    assert "<b>node_health</b>" in out["events"] and "reasons=GPU 3 XID 79" in out["events"]
    assert r.page_view(now=r.max_ts)["topology"]["diloco"]["roster"] == 1  # stopped while a spare takes its place


def test_the_reason_styles_are_on_the_page(tmp_path):
    page = export_html(reducer(), tmp_path / "p.html").read_text()
    assert ".ns.bd{color:var(--bad)}" in page  # a lost node's reason in red, not in the .ns grey
    # the Nodes column: cut by the column's width (no fixed character count), the whole on hover
    assert "table.nt th.spn,table.nt td.spn{text-align:left;width:14%;max-width:0}" in page
    assert "table.nt td.spn .ns{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}" in page
    assert "28ch" not in page


def test_one_bad_check_is_a_warning_without_badge(tmp_path):
    now = reducer().max_ts
    health = {"verdict": "bad_once", "nodes": {"gpu-c": node("bad_once", now - 5, ["GPU 0 uncorrectable ECC +1"], 1)}}
    out = smoke(tmp_path, _with_record(tmp_path, "running", health=health, updated=now - 5))
    assert "gpu-c: GPU 0 uncorrectable ECC +1 · 1 bad check" in out["texts"] and "node ✕" not in out["texts"]
    assert '<td class="spn"><div class="ns wt"' in out["nodes"] and "node ✕" not in out["nodes"]


@pytest.mark.parametrize("verdict, words", [("bad_once", "health bad in 1 check"),
                                            ("lost", "health bad in 2 checks in a row"),
                                            ("read_failed", "health unread")])
def test_a_check_without_reasons_reads_in_words(tmp_path, verdict, words):
    now = reducer().max_ts
    health = {"verdict": verdict, "nodes": {"gpu-c": node(verdict, now - 5)}}
    out = smoke(tmp_path, _with_record(tmp_path, "running", health=health, updated=now - 5))
    assert "gpu-c: " + words in out["texts"] and "gpu-c: " + words in out["nodes"]
    assert verdict not in " ".join(out["texts"])


def test_a_stale_check_reads_unread(tmp_path):
    now = reducer().max_ts
    health = {"verdict": "ok", "nodes": {"gpu-c": node("ok", now - 400)}}
    out = smoke(tmp_path, _with_record(tmp_path, "running", health=health, updated=now - 5))
    assert "gpu-c: health unread (stale)" in out["texts"] and "gpu-c: health unread (stale)" in out["nodes"]


def test_a_replaced_island_s_count_and_last_time(tmp_path):
    end = reducer().max_ts
    r = _with_record(tmp_path, "running", lost(end - 200), replaced(end - 150),
                     spares=spares_doc(nodes=["gpu-c", "gpu-e"], spare_nodes=["gpu-d"], replacements=1,
                                       replacements_window=1, last_replaced_at=end - 150))
    out = smoke(tmp_path, r)
    assert "replaced 1× · gpu-d → gpu-e · 2m30s ago" in out["texts"]
    assert "waiting spare" not in out["texts"] and "node ✕" not in out["texts"]
    assert "replaced 1× · 2m30s ago" in out["nodes"] and "1 of at most 3 replacements in 24 h" in out["nodes"]
    assert "gpu-d → gpu-e · resumed at step 6 · 2m30s ago" in out["tip"] and "1 · 1 of at most 3 in 24 h" in out["tip"]


def test_the_reason_it_gave_up_is_on_its_card(tmp_path):
    end = reducer().max_ts
    out = smoke(tmp_path, reducer(lost(end + 10), waiting(end + 20), limit(end + 30)))
    assert "3 node replacement(s)" in out["texts"] and "waiting spare" not in out["texts"]
    assert '<td class="spn"><div class="ns bd" title="3 node replacement(s)">3 node replacement(s)</div></td>' in out["nodes"]


def test_the_reason_a_trainer_left_is_kept(tmp_path):
    end = reducer().max_ts
    r = _with_record(tmp_path, "failed", lost(end - 100), waiting(end - 95), limit(end - 50),
                     exit="3 node replacement(s) within 24 h: the island stops")
    c = cards(r)["1"]
    assert c["left"]["reason"] and c["spares"]["stopped"]
    out = smoke(tmp_path, r)
    assert '<div class="ns">' + c["left"]["reason"] + "</div></td></tr>" in out["nodes"]  # its last cell, as on main


@pytest.mark.parametrize("why, short", [
    ("fabric group rack-b has no free healthy spare node (gpu-e: quarantined (XID 79 (GPU 3)); gpu-f: held)",
     "gpu-e: quarantined (XID 79 (GPU 3)); gpu-f: held · no free spare in rack-b"),
    ("the inventory has no free healthy spare node (gpu-e: unhealthy); placement: no group (of 2) has room",
     "gpu-e: unhealthy · no free spare · placement: no group (of 2) has room"),
    ("fabric group rack-b has no free healthy spare node", "no free spare in rack-b"),
    ("spare refused: no rendezvous address", "spare refused: no rendezvous address"),
])
def test_why_no_spare_in_short(tmp_path, why, short):
    end = reducer().max_ts
    r = _with_record(tmp_path, "recovering", lost(end - 100),
                     spares=spares_doc(lost=["gpu-d"], waiting_spare=True, wait_reason=why, waiting_since=end - 95))
    out = smoke(tmp_path, r, width=2000)
    assert '">' + short + "</div>" in out["nodes"]


def test_the_event_stream_shows_the_spare_node_events(tmp_path):
    end = reducer().max_ts  # the newest events: the stream shows the latest 15
    out = smoke(tmp_path, reducer(lost(end + 10), waiting(end + 20), replaced(end + 30),
                                  limit(end + 40),
                                  waiting(end + 50, elapsed=86400, given_up=True)))["events"]
    assert 'title="lost"' in out and "<b>island_node_lost</b>" in out and "node=gpu-d" in out
    assert 'title="wait"' in out and "elapsed_s=0" in out and "next_check_s=60" in out
    assert 'title="swap"' in out and "lost=gpu-d spare=gpu-e resume_step=6" in out
    assert 'title="max"' in out and 'title="stop"' in out and "given_up=true" in out


def test_a_long_field_is_cut_with_an_ellipsis_the_whole_on_hover(tmp_path):
    end = reducer().max_ts
    why = WAIT[:-1] + "; gpu-f: quarantined (XID 79 (GPU 3)))"  # 120 characters
    out = smoke(tmp_path, reducer(lost(end + 10), ev(isp.WAITING_SPARE, end + 20, lost=["gpu-d"], reason=why)))["events"]
    assert "reason=" + why[:79] + "…" in out and "reason=" + why + " " not in out.split("title=")[0]
    assert 'title="lost=gpu-d reason=' + why + '"' in out
    assert "title=\"node=gpu-d" not in out  # a line with no cut field has no hover of its own


def test_a_card_without_spare_nodes_draws_nothing_new(tmp_path):
    out = smoke(tmp_path, reducer())
    for word in ("waiting spare", "node ✕", "replaced", "no free spare"):
        assert word not in json.dumps(out)


# -- review fixes: health over a relaunch / a wait, replaced nodes, give-ups, long waits, the roster ---------
def test_a_waiting_island_keeps_its_lost_node_s_reason_and_no_stale_check(tmp_path):
    """Its checks stop on purpose while it recovers (the record is written by the hourly wait report):
    the lost node still reads lost with its reason, a healthy node is not shown, nothing reads stale."""
    end = reducer().max_ts
    health = {"verdict": "lost", "nodes": {"gpu-c": node("ok", end - 3700),
                                           "gpu-d": node("lost", end - 3700, ["GPU 3 XID 79"], 2)}}
    r = _with_record(tmp_path, "recovering", lost(end - 3700), waiting(end - 3690),
                     spares=spares_doc(lost=["gpu-d"], waiting_spare=True, wait_reason=WAIT, waiting_since=end - 3690),
                     health=health, updated=end - 10)
    assert cards(r)["1"]["node_health"] == [{"node": "gpu-d", "verdict": "lost", "reasons": ["GPU 3 XID 79"],
                                             "bad_in_row": 2}]
    out = smoke(tmp_path / "p", r)
    assert "node gpu-d</span><span>lost · GPU 3 XID 79" in out["tip"] and "stale" not in out["tip"]


def test_a_node_a_spare_replaced_is_no_node_of_the_island(tmp_path):
    end = reducer().max_ts
    health = {"verdict": "lost", "nodes": {"gpu-c": node("ok", end - 30),
                                           "gpu-d": node("lost", end - 30, ["GPU 3 XID 79"], 2)}}
    r = _with_record(tmp_path, "running", lost(end - 100), replaced(end - 20),
                     spares=spares_doc(nodes=["gpu-c", "gpu-e"], spare_nodes=["gpu-d"], replacements=1,
                                       replacements_window=1, last_replaced_at=end - 20),
                     health=health, updated=end - 5)
    assert "node_health" not in cards(r)["1"]
    out = smoke(tmp_path / "p", r)
    assert "node ✕" not in out["texts"] and "node gpu-d" not in (out["tip"] or "")


def test_a_later_wait_or_replacement_ends_a_give_up():
    sp = _spares_of([("a", lost(T0 + 400)), ("a", limit(T0 + 500)),
                     ("b", lost(T0 + 9000)), ("b", waiting(T0 + 9001))])
    assert sp["stopped"] is None and sp["waiting"]  # waiting again: not given up
    sp = _spares_of([("a", lost(T0 + 400)), ("a", limit(T0 + 500)), ("b", lost(T0 + 9000)), ("b", replaced(T0 + 9100))])
    assert sp["stopped"] is None and sp["replacements"] == 1


def test_a_running_record_shows_no_give_up(tmp_path):
    end = reducer().max_ts
    r = _with_record(tmp_path, "running", lost(end - 9000), limit(end - 8000),
                     spares=spares_doc(replacements=1, replacements_window=1, last_replaced_at=end - 100))
    assert cards(r)["1"]["spares"]["stopped"] is None
    out = smoke(tmp_path / "p", r)
    assert "3 node replacement(s)" not in out["nodes"] and "3 node replacement(s)" not in out["texts"]
    events_only = cards(reducer(lost(end + 10), limit(end + 20)))["1"]["spares"]
    assert events_only["stopped"] == "3 node replacement(s)"  # no record: the events' give-up stays
    # a running record without spares of its own: the events' give-up is over too
    r = _with_record(tmp_path / "b", "running", lost(end - 9000), limit(end - 8000))
    sp = cards(r)["1"]["spares"]
    assert sp["source"] == "events" and sp["stopped"] is None


HOUR = 3600.0


def _long_wait(tmp_path, waited: float, *, report_age: float = 60.0, plan_absent: float | None = None,
               max_wait: float | None = None, poll: float | None = None):
    """Island 1 waiting ``waited`` seconds for a spare node at the page's time, its record written
    ``report_age`` seconds before it; its tape silent for an hour."""
    end = reducer().max_ts
    now = end + HOUR
    r = reducer(lost(now - 10), ev("island_lost", now - 5, reason=LOST), waiting(now, elapsed=waited))
    roster = {**({"waiting_absent_after_s": plan_absent} if plan_absent is not None else {}),
              **({"controller_poll_s": poll} if poll is not None else {})}
    if roster:
        assert r.load_fleet_plan({**plan_doc(), "roster": roster}, "test")
    doc = spares_doc(lost=["gpu-d"], waiting_spare=True, wait_reason=WAIT, waiting_since=now - waited,
                     **({"max_wait_s": max_wait} if max_wait is not None else {}))
    assert r.load_fleet_state(store_state(tmp_path / "s", "recovering", spares=doc, updated=now - report_age))
    assert r.max_ts == now
    return r


def test_a_wait_past_waiting_absent_after_s_counts_as_absent(tmp_path):
    r = _long_wait(tmp_path, 2 * HOUR)
    sp = cards(r)["1"]["spares"]
    assert sp["waiting"] and sp["wait_state"] == {"state": "absent"}
    out = smoke(tmp_path / "p", r)
    assert "waiting for a spare node · counts as absent" in out["texts"] and out["dashed"] == 0
    assert "counts as absent" in out["nodes"] and "counts as absent for the fleet" in out["tip"]
    assert out["roster"] == "--warn"  # out of the roster as a warning, not a loss
    # the plan's fleet.waiting_absent_after_s (3 h): two hours is an ordinary wait
    r = _long_wait(tmp_path / "b", 2 * HOUR, plan_absent=3 * HOUR)
    assert "wait_state" not in cards(r)["1"]["spares"]
    assert "counts as absent" not in json.dumps(smoke(tmp_path / "q", r))


@pytest.mark.parametrize("max_wait", [None, 6 * HOUR])
def test_a_wait_past_spare_max_wait_s_reads_lost(tmp_path, max_wait):
    waited = 30 * HOUR if max_wait is None else 7 * HOUR  # past the default 24 h, or the record's own limit
    r = _long_wait(tmp_path, waited, max_wait=max_wait)
    c = cards(r)["1"]
    assert not c["spares"]["waiting"] and c["spares"]["wait_state"]["state"] == "expired"
    assert c["island_now"]["state"] == "lost" and c["status"] in ("lost", "stale")
    out = smoke(tmp_path / "p", r)
    assert out["dashed"] >= 1 and "lost" in out["texts"] and "unknown" not in out["texts"]  # a lost trainer again
    assert any(t.startswith("lost · no event for 1h0m") for t in out["texts"])  # the phase, against the same time
    assert not any(t.startswith("waiting for a spare node ·") for t in out["texts"])  # no wait phase
    assert any("past --spare-max-wait-s" in t for t in out["texts"]) and "past --spare-max-wait-s" in out["nodes"]
    assert out["roster"] == "--bad"  # a loss, not a wait


def test_a_wait_no_record_reports_any_more_reads_lost(tmp_path):
    r = _long_wait(tmp_path, 2 * HOUR, report_age=2 * HOUR)  # the launcher stopped writing it two hours ago
    c = cards(r)["1"]
    assert not c["spares"]["waiting"] and c["spares"]["wait_state"]["state"] == "frozen"
    assert c["island_now"]["state"] == "unknown"  # nobody says what it is
    out = smoke(tmp_path / "p", r)
    # unknown (nobody reports the wait), not lost as past the longest wait: badge, phase and Fragments agree
    assert out["dashed"] >= 1 and "unknown" in out["texts"] and "lost" not in out["texts"]
    assert any(t.startswith("unknown · no event for 1h0m") for t in out["texts"])
    why = "launcher may be gone · no wait report for 2h0m"  # what to act on first, the same duration format
    assert any(t.startswith(why) for t in out["texts"]) and why in out["nodes"] and why in out["tip"]
    assert "trainer 1 unknown" in out["nodes"] and "trainer 1 lost" not in out["nodes"]  # the ended islands' line
    # an hourly report one hour and ten minutes old is still a wait
    ok = _long_wait(tmp_path / "b", 30 * 60, report_age=70 * 60)
    assert cards(ok)["1"]["spares"]["waiting"] and "wait_state" not in cards(ok)["1"]["spares"]


def test_an_events_only_wait_no_event_reports_any_more_reads_lost(tmp_path):
    end = reducer().max_ts
    r = reducer(lost(end + 10), waiting(end + 20), ev("island_ready", end + 3 * HOUR, k=0))
    sp = cards(r)["1"]["spares"]
    assert sp["source"] == "events" and not sp["waiting"] and sp["wait_state"]["state"] == "frozen"


def test_a_fresh_failure_waiting_for_a_spare_is_out_of_the_roster(tmp_path):
    """Right after the failure its tape is still live: the wait alone (no lost-node event yet) takes
    it out of the roster, and the layout line counts it the same way."""
    end = reducer().max_ts
    base = reducer().page_view(now=end)["topology"]["diloco"]
    assert base["roster"] == 2 and "roster_spare" not in base
    r = reducer(waiting(end + 5))
    view = r.page_view(now=r.max_ts)["topology"]["diloco"]
    assert cards(r)["1"]["spares"]["lost"] == [] and view["roster"] == 1 and view["roster_spare"] == 1
    out = smoke(tmp_path, r)
    assert "1/2 trainer(s) live (1 replacing a lost node)" in out["note"]
    assert out["roster"] == "--warn"


def test_the_wait_reads_the_same_time_on_card_tip_and_table(tmp_path):
    end = reducer().max_ts
    r = _with_record(tmp_path, "recovering", lost(end - 100), waiting(end - 95),
                     spares=spares_doc(lost=["gpu-d"], waiting_spare=True, wait_reason=WAIT, waiting_since=end - 95))
    out = smoke(tmp_path, r)
    assert "waiting for a spare node · 1m35s" in out["texts"]  # card
    assert "waiting for a spare</span><span>1m35s · " in out["tip"]  # its tip
    assert "waiting spare 1m35s" in out["nodes"]  # Islands now


def test_a_slow_controller_poll_is_no_unreported_wait(tmp_path):
    """--controller-poll 1500 s (from the plan): a report lands up to two polls late (3600 + 2 x 1500 + 600 s)."""
    ok = _long_wait(tmp_path, 2 * HOUR, report_age=HOUR + 2 * 1500 + 500, poll=1500.0)
    assert cards(ok)["1"]["spares"]["waiting"] and cards(ok)["1"]["spares"].get("wait_state") == {"state": "absent"}
    late = _long_wait(tmp_path / "b", 2 * HOUR, report_age=HOUR + 2 * 1500 + 700, poll=1500.0)
    assert cards(late)["1"]["spares"]["wait_state"]["state"] == "frozen"


@pytest.mark.parametrize("age, frozen", [(fleet_islands.WAIT_FROZEN_S - 1, False), (fleet_islands.WAIT_FROZEN_S + 1, True)])
def test_the_unreported_wait_boundary(age, frozen):
    assert fleet_islands.WAIT_FROZEN_S == 3600 + 2 * 30 + 600  # the hourly report, two default polls, 10 min
    ws = fleet_islands.spare_wait_state(1000.0, 10_000.0, 10_000.0 + age, absent_after_s=1e9)
    assert (ws is not None and ws["state"] == "frozen") is frozen


def test_the_dashboard_constants_are_the_runtime_s():
    from yeto.cli import build_parser  # noqa: PLC0415
    from yeto.rl import fleet as rl_fleet  # noqa: PLC0415

    assert fleet_islands.SPARE_MAX_WAIT_S == isp.MAX_WAIT_S
    assert fleet_islands.SPARE_WAIT_REPORT_S == isp.WAIT_REPORT_S
    assert fleet_islands.WAITING_ABSENT_AFTER_S == rl_fleet.DEFAULT_WAITING_ABSENT_AFTER_S
    sub = next(a for a in build_parser()._actions if isinstance(a, argparse._SubParsersAction))  # noqa: SLF001
    poll = next(a for a in sub.choices["launch"]._actions if "--controller-poll" in a.option_strings)  # noqa: SLF001
    assert fleet_islands.CONTROLLER_POLL_S == poll.default


def test_the_plan_carries_the_controller_poll():
    from yeto.rl import fleet as rl_fleet  # noqa: PLC0415

    assert rl_fleet.controller_poll_s(["--gpu", "x", "--controller-poll", "1500"]) == 1500.0
    assert rl_fleet.controller_poll_s(["--controller-poll=90"]) == 90.0 and rl_fleet.controller_poll_s(["--gpu", "x"]) is None
    plan = fleet_islands.load_plan({**plan_doc(), "roster": {"controller_poll_s": 1500}})
    assert plan["controller_poll_s"] == 1500.0 and "controller_poll_s" not in fleet_islands.load_plan(plan_doc())


def test_an_events_only_wait_takes_its_max_wait_from_its_events(tmp_path):
    end = reducer().max_ts
    r = reducer(lost(end + 10), waiting(end + 20, elapsed=3 * HOUR, max_wait_s=2 * HOUR))
    sp = cards(r)["1"]["spares"]
    assert sp["source"] == "events" and sp["wait_state"]["state"] == "expired" and "(2 h)" in sp["wait_state"]["why"]
    assert cards(reducer(lost(end + 10), waiting(end + 20, elapsed=3 * HOUR)))["1"]["spares"]["wait_state"] == {
        "state": "absent"}  # without it: the default 24 h


def test_an_empty_nodes_column_is_not_drawn(tmp_path):
    r = _long_wait(tmp_path, 30 * HOUR)  # the trainer reads lost: in the ended table, the live one has no reason
    out = smoke(tmp_path / "p", r)
    live, _, ended = out["nodes"].partition("<details")
    assert '<th class="spn">Nodes</th>' not in live and '<th class="spn">Nodes</th>' in ended
