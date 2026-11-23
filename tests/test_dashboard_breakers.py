"""Breakers panel (design 9): resolved settings, status snapshots, trips, alerts, run status."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from yeto.dashboard import alerts, sources
from yeto.dashboard.breakers import (
    RESOLVED_FILE,
    STORE_RESOLVED_KEY,
    BreakerState,
    ResolvedFile,
    resolved_candidates,
    summary,
)
from yeto.dashboard.export import export_html, render_page
from yeto.dashboard.reducer import Reducer
from yeto.dashboard.serve import DashboardState
from yeto.rl.engine.object_store import DirectoryObjectStore

from dashboard_helpers import FX, T0

BRK = FX / "breakers"
NOW = T0 + 3500


def fixture_reducer(**kw) -> Reducer:
    r = Reducer(run="brk-demo", **kw)
    sources.load_all(r, [str(BRK)])
    ResolvedFile(resolved_candidates([str(BRK)])).pump(r)
    return r


def rows(view) -> dict:
    return {row["name"]: row for layer in view["layers"] for row in layer["rows"]}


def brk(name, state, t, *, action="stop", value=1.0, limit=2.0, scope=None, **kw):
    return {"event": "rl_breaker", "schema": "yeto-breaker-v1", "name": name, "layer": name.split(".")[0],
            "kind": "breaker", "state": state, "action": action, "value": value, "limit": limit, "unit": "s",
            "direction": "max", "scope": scope or {}, "count": 1, "time_unix": t, **kw}


def status(items, t, scope=None, layer="sampler"):
    return {"event": "rl_breaker_status", "schema": "yeto-breaker-v1", "layer": layer, "scope": scope or {},
            "items": items, "time_unix": t}


def test_fixture_panel_layers_states_sources():
    v = fixture_reducer().page_view(now=NOW)["breakers"]
    assert [x["layer"] for x in v["layers"]] == ["run", "launch", "trainer", "sampler", "harness", "pool", "monitor"]
    assert v["counts"] == {"armed": 13, "implicit": 1, "off": 2, "invariant": 1}
    r = rows(v)
    assert r["run.wall_clock"]["state"] == "implicit"
    assert r["run.wall_clock"]["source"] == {"kind": "default", "location": "default (implicit)"}
    assert r["launch.idle"]["state"] == "off" and r["launch.idle"]["source"]["kind"] == "switch"
    assert r["trainer.token_alignment"]["state"] == "invariant"
    # the most specific source of an entry: the alias over the default of its enabled flag
    assert r["trainer.sample_wait"]["source"]["location"] == "trainer.argv --rl-sample-wait-s"
    assert r["sampler.dispense_redo"]["redundant"] == ["spec prompts.dispense.max_redo"]
    assert v["switches"]["cost_guard"]["on"] is False
    assert v["resolved_from"].endswith(RESOLVED_FILE)


def test_fixture_headroom_from_status_snapshots():
    r = rows(fixture_reducer().page_view(now=NOW)["breakers"])
    g = r["trainer.sample_wait"]["gauge"]
    assert (g["value"], g["limit"], g["ratio"], g["scope"]) == (3300.0, 3600.0, 0.9167, "island 0")  # newest
    assert r["sampler.dispense_redo"]["gauge"]["ratio"] == 0.75
    assert r["sampler.min_samples"]["gauge"]["limit"] is None and r["sampler.min_samples"]["gauge"]["value"] == 5
    # the cost guard's own cost_tick is the launch budget gauge
    lb = r["launch.budget"]["gauge"]
    assert (lb["value"], lb["limit"], lb["detail"]) == (12.0, 450.0, "cost guard spend")
    assert r["run.starvation"]["gauge"] is None


def test_fixture_trips_with_stop_reason_and_findings():
    v = fixture_reducer().page_view(now=NOW)["breakers"]
    names = [(t["name"], t["state"]) for t in v["trips"]]
    assert names[0] == ("trainer.sample_wait", "trip")  # newest first
    assert ("launch.budget", "warn") in names and ("run.wall_clock", "warn") in names  # cost guard / run up tapes
    first = v["trips"][0]
    assert first["stop_reason"] == "breaker:trainer.sample_wait" and first["scope"] == "island 0"
    r = rows(v)
    assert r["trainer.sample_wait"]["tripped"] and r["trainer.sample_wait"]["last"]["state"] == "trip"
    assert not r["sampler.drain"]["tripped"] and r["sampler.drain"]["last"]["state"] == "warn"
    assert [f["rule"] for f in v["findings"]] == ["D1", "D8", "D9"]


def test_no_panel_on_tapes_without_breakers():
    r = Reducer(run="old")
    r.feed({"event": "rl_local_round", "island_id": 0, "local_round_id": 1, "time_unix": T0, "reward_mean": 0.1})
    assert "breakers" not in r.page_view(now=T0 + 10)


def test_panel_from_events_only():
    r = Reducer()
    r.feed(brk("pool.engine_stall", "trip", T0, action="replace", scope={"pool": "nc"}))
    v = r.page_view(now=T0 + 1)["breakers"]
    row = rows(v)["pool.engine_stall"]
    assert row["from"] == "events" and row["enabled"] is None and row["state"] == "armed" and row["tripped"]
    assert v["trips"][0]["scope"] == "nc" and v["trips"][0]["stop_reason"] is None  # replace is not a stop


def test_stop_trip_without_reason_gets_one():
    st = BreakerState()
    st.fold(brk("run.budget", "trip", T0), None, T0)
    assert st.view()["trips"][0]["stop_reason"] == "breaker:run.budget"


def test_status_older_snapshot_does_not_replace_newer():
    r = Reducer()
    item = {"name": "sampler.drain", "value": 500.0, "limit": 600, "unit": "s", "ratio": 0.8333}
    r.feed(status([item], T0 + 100, {"sampler_id": "s-0"}))
    r.feed(status([dict(item, value=60.0, ratio=0.1)], T0 + 50, {"sampler_id": "s-0"}))
    g = rows(r.page_view(now=T0 + 200)["breakers"])["sampler.drain"]["gauge"]
    assert g["value"] == 500.0


def test_status_worst_scope_wins():
    r = Reducer()
    item = {"name": "sampler.drain", "value": 60.0, "limit": 600, "unit": "s", "ratio": 0.1}
    r.feed(status([item], T0, {"sampler_id": "s-0"}))
    r.feed(status([dict(item, value=570.0, ratio=0.95)], T0, {"sampler_id": "s-1"}))
    g = rows(r.page_view(now=T0 + 1)["breakers"])["sampler.drain"]["gauge"]
    assert (g["scope"], g["scopes"], g["ratio"]) == ("s-1", 2, 0.95)


def test_process_settings_differ_from_resolved():
    r = fixture_reducer()
    r.feed(status([{"name": "sampler.drain", "value": 1.0, "limit": 900, "unit": "s", "ratio": 0.0011,
                    "enabled": True, "action": "drop", "params": {"timeout_s": 900}}], T0 + 3400, {"sampler_id": "s-1"}))
    assert rows(r.page_view(now=NOW)["breakers"])["sampler.drain"]["runtime_params"] == {"timeout_s": 900}


def test_alerts_trip_critical_warn_warning_clear_ends():
    r = Reducer()
    r.feed(brk("trainer.sample_wait", "trip", T0, scope={"island": "0"}, detail="rollout 3"))
    r.feed(brk("sampler.drain", "warn", T0, action="drop", scope={"sampler_id": "s-0"}))
    r.feed(brk("launch.stall", "config_mismatch", T0))
    r.feed(brk("trainer.nan", "shadow", T0, action="warn"))
    got = {a["title"]: a for a in r.overview(now=T0 + 10)["alerts"] if a["rule"] == "breaker"}
    assert got["breaker trainer.sample_wait tripped (island 0)"]["sev"] == 0
    assert got["breaker trainer.sample_wait tripped (island 0)"]["island"] == "0"
    assert "breaker:trainer.sample_wait" in got["breaker trainer.sample_wait tripped (island 0)"]["detail"]
    assert got["breaker sampler.drain near its limit (s-0)"]["sev"] == 1
    assert got["breaker launch.stall runs with other settings"]["sev"] == 1
    assert got["breaker trainer.nan would have tripped"]["sev"] == 2
    r.feed(brk("sampler.drain", "clear", T0 + 20, action="drop", scope={"sampler_id": "s-0"}))
    titles = [a["title"] for a in r.overview(now=T0 + 30)["alerts"]]
    assert "breaker sampler.drain near its limit (s-0)" not in titles


def test_alert_for_a_drop_trip_ages_out_a_stop_trip_stays():
    r = Reducer()
    r.feed(brk("sampler.min_samples", "trip", T0, action="drop"))
    r.feed(brk("run.budget", "trip", T0, action="stop"))
    later = T0 + 2 * 3600
    titles = [a["title"] for a in r.overview(now=later)["alerts"] if a["rule"] == "breaker"]
    assert titles == ["breaker run.budget tripped"]


def test_alert_of_a_disabled_monitor_entry_is_dropped():
    r = Reducer()
    r.load_resolved({"entries": [{"name": "monitor.dashboard_x", "layer": "monitor", "kind": "alert",
                                  "enabled": False, "params": {}, "sources": {}}]}, "t")
    th = r.thresholds
    keep = alerts._alert(1, "x", "kept", "")
    drop = dict(alerts._alert(1, "x", "dropped", ""), breaker="monitor.dashboard_x")
    orig = alerts.heartbeat
    try:
        alerts.heartbeat = lambda cards, th: [keep, drop]
        out = alerts.evaluate(r, now=T0, rounds=[], cost={}, cards=[])
    finally:
        alerts.heartbeat = orig
    assert [a["title"] for a in out] == ["kept"] and th is r.thresholds


def resolved_budget(kind, usd=300.0, warn=0.7):
    src = {"kind": kind, "location": "x"}
    return {"entries": [{"name": "run.budget", "layer": "run", "kind": "breaker", "enabled": True,
                         "params": {"usd": usd, "warn_ratio": warn},
                         "sources": {"usd": src, "warn_ratio": src}}]}


def test_budget_from_the_breakers_section_only():
    r = Reducer()
    r.load_resolved(resolved_budget("legacy"), "t")  # an alias: the page keeps its defaults
    assert r.budget_usd is None and r.thresholds["budget_warn_pct"] == 60.0
    r = Reducer()
    r.load_resolved(resolved_budget("breakers"), "t")
    assert r.budget_usd == 300.0 and r.thresholds["budget_warn_pct"] == 70.0


def test_cli_budget_and_threshold_win_and_mismatch_is_shown():
    r = Reducer(budget_usd=500.0, thresholds={"budget_warn_pct": 50})
    r.load_resolved(resolved_budget("breakers"), "t")
    assert r.budget_usd == 500.0 and r.thresholds["budget_warn_pct"] == 50.0
    assert r.page_view(now=T0)["breakers"]["budget_mismatch"] == {"cli": 500.0, "resolved": 300.0}


def test_cost_guard_tape_feeds_only_the_panel(tmp_path):
    (tmp_path / "cost-events.jsonl").write_text(
        json.dumps({"event": "cost_tick", "time_unix": T0, "total_usd": 5.0, "budget_usd": 10.0}) + "\n"
        + json.dumps({"event": "idle_warn", "time_unix": T0 + 1}) + "\n")
    r = Reducer()
    sources.load_all(r, [str(tmp_path)])
    assert len(r.cost_ticks) == 0 and r.counts["fleet"] == 0  # not the fleet's cost tick
    g = rows(r.page_view(now=T0 + 5)["breakers"])["launch.budget"]["gauge"]
    assert g["ratio"] == 0.5
    sources.load_all(r, [str(tmp_path)])  # read again from offset 0 by a new source: not folded twice
    assert sum(1 for e in r.events if e["source"].endswith("cost-events.jsonl")) == 1


def test_run_up_tape_needs_its_state_file(tmp_path):
    up = tmp_path / "up"
    up.mkdir()
    (up / "events.jsonl").write_text(json.dumps(brk("run.first_group", "trip", T0)) + "\n")
    assert not sources.breakers_only(up / "events.jsonl")
    (up / "state.json").write_text("{}")
    assert sources.breakers_only(up / "events.jsonl")


def test_run_sources_include_cost_guard_and_run_up_tapes(tmp_path, monkeypatch):
    monkeypatch.setattr("yeto.runs.run_dir", lambda run: tmp_path / "runs" / run)
    monkeypatch.setenv("YETO_STATE_DIR", str(tmp_path / "up"))
    paths = sources.run_sources("r1")
    assert str(tmp_path / "runs" / "r1" / "cost-events.jsonl") in paths
    assert str(tmp_path / "up" / "r1" / "events.jsonl") in paths


def test_resolved_candidates_and_reload_on_change(tmp_path, monkeypatch):
    monkeypatch.setenv("YETO_STATE_DIR", str(tmp_path / "up"))
    cands = resolved_candidates([str(tmp_path / "tapes" / "a.jsonl")], run="r1", up_state=str(tmp_path / "s" / "state.json"))
    assert cands[:2] == [str(tmp_path / "s" / RESOLVED_FILE), str(tmp_path / "up" / "r1" / RESOLVED_FILE)]
    f = tmp_path / "s" / RESOLVED_FILE
    f.parent.mkdir()
    f.write_text(json.dumps(resolved_budget("breakers", usd=100.0)))
    r = Reducer()
    rf = ResolvedFile(cands)
    assert rf.pump(r) and r.budget_usd == 100.0
    assert not rf.pump(r)  # unchanged
    f.write_text(json.dumps(resolved_budget("breakers", usd=200.0)))
    import os
    os.utime(f, (T0 + 99, T0 + 99))
    assert rf.pump(r) and r.budget_usd == 200.0


def test_resolved_file_from_the_store(tmp_path):
    store = DirectoryObjectStore(str(tmp_path))
    store.write(STORE_RESOLVED_KEY, json.dumps(resolved_budget("breakers", usd=42.0)).encode())
    clock = [0.0]
    rf = ResolvedFile([], store, clock=lambda: clock[0], interval_s=300)
    r = Reducer()
    assert rf.pump(r) and r.breakers.resolved_from == "store " + STORE_RESOLVED_KEY and r.budget_usd == 42.0
    store.write(STORE_RESOLVED_KEY, json.dumps(resolved_budget("breakers", usd=43.0)).encode())
    assert not rf.pump(r)  # not polled again before the interval
    clock[0] = 301.0
    assert rf.pump(r) and r.budget_usd == 43.0


def test_serve_poll_reads_the_resolved_file():
    r = Reducer(run="brk-demo")
    state = DashboardState(r, [str(BRK)], resolved=ResolvedFile(resolved_candidates([str(BRK)])))
    state.poll()
    assert r.breakers.resolved is not None and r.page_view(now=NOW)["breakers"]["trip_total"] >= 5


def test_legacy_events_map_to_their_entry(monkeypatch):
    monkeypatch.setattr("yeto.dashboard.breakers.legacy_event_names", lambda: {"nan_skip_breaker": "trainer.nan"})
    r = Reducer()
    r.feed({"event": "nan_skip_breaker", "island_id": 0, "time_unix": T0, "detail": "2 NaN rounds in a row"})
    v = r.page_view(now=T0 + 1)["breakers"]
    t = v["trips"][0]
    assert (t["name"], t["legacy"], t["legacy_event"], t["scope"]) == ("trainer.nan", True, "nan_skip_breaker", "island 0")
    # an entry with an rl_breaker of its own does not list its legacy events again
    r.feed(brk("trainer.nan", "trip", T0 + 2))
    assert [x["legacy"] for x in r.page_view(now=T0 + 3)["breakers"]["trips"]] == [False]


def test_run_status_summary():
    doc = json.loads((BRK / RESOLVED_FILE).read_text())
    recs = [json.loads(line) for name in ("trainer.jsonl", "events.jsonl") for line in (BRK / name).read_text().splitlines()]
    out = summary(doc, recs)
    assert out["resolved"] and out["has_section"] and out["entries"]["implicit"] == 1
    assert out["implicit"] == ["run.wall_clock"] and set(out["off"]) == {"launch.idle", "harness.reach_check"}
    assert out["trips"][0]["stop_reason"] == "breaker:trainer.sample_wait"
    assert {"name": "trainer.sample_wait", "ratio": 0.9167, "scope": "island 0"} in out["near_limit"]
    assert [f["rule"] for f in out["findings"]] == ["D1", "D8"]  # info left out
    assert summary(None, []) == {"resolved": False, "entries": None, "trips": []}


def test_run_status_reads_the_state_dir_and_trainer_log(tmp_path):
    from yeto.rl.event_echo import PREFIX
    from yeto.rl.up import Run

    log = tmp_path / "trainer.log"
    log.write_text("noise\n" + PREFIX + json.dumps(brk("trainer.sample_wait", "trip", T0)) + "\n")
    (tmp_path / RESOLVED_FILE).write_text((BRK / RESOLVED_FILE).read_text())
    (tmp_path / "events.jsonl").write_text(json.dumps(brk("run.wall_clock", "warn", T0 + 5)) + "\n")
    run = Run.__new__(Run)
    run.dir = tmp_path
    run.state = {"procs": {"trainer": {"log": str(log)}}}
    out = run.breaker_status()
    assert [t["name"] for t in out["trips"]] == ["run.wall_clock", "trainer.sample_wait"]
    assert out["resolved"] is True


def test_page_ships_the_panel_script():
    html = render_page(None)
    assert "var YetoBrk=" in html and 'id="brkBox"' in html and 'id="navBrk"' in html
    assert html.index("var YetoBrk=") < html.index("YetoBrk.render(V)")
    js = (Path(sources.__file__).parent / "static" / "breakers.js").read_text()
    assert "://" not in js  # the page makes no request


SMOKE = r"""
const fs = require("fs");
const html = fs.readFileSync(process.argv[2], "utf8");
const data = html.match(/<script type="application\/json" id="yeto-data">([\s\S]*?)<\/script>/)[1];
const app = html.match(/<script>\n([\s\S]*?)\n<\/script>/)[1];
const els = {};
function node(id) {
  return { id, innerHTML: "", textContent: "", hidden: false, style: {}, dataset: {}, children: [], attrs: {},
    listeners: {}, offsetWidth: 200, offsetHeight: 120, clientWidth: 420,
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    setAttribute(k, v) { this.attrs[k] = String(v); }, removeAttribute(k) { delete this.attrs[k]; },
    appendChild(c) { this.children.push(c); return c; },
    addEventListener(t, fn) { (this.listeners[t] = this.listeners[t] || []).push(fn); },
    getBoundingClientRect() { return { left: 0, top: 0, width: 420, height: 400 }; },
    querySelectorAll() { return []; } };
}
function mk(id) { if (!els[id]) { els[id] = node(id); if (id === "yeto-data") els[id].textContent = data; } return els[id]; }
global.document = { documentElement: node("html"), getElementById: mk, querySelectorAll: () => [], title: "",
  createElementNS: () => node(null), createElement: () => node(null) };
global.navigator = {}; global.localStorage = { getItem() { return null; }, setItem() {} };
global.setInterval = () => 0; global.setTimeout = () => 0;
eval(app);
console.log(JSON.stringify({ hidden: els.brkBox.hidden, nav: els.navBrk.style.display, note: els.brkNote.textContent,
  html: els.brk.innerHTML }));
"""


def test_exported_page_draws_the_panel(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    page = export_html(fixture_reducer(), tmp_path / "p.html", generated_at=NOW)
    script = tmp_path / "smoke.js"
    script.write_text(SMOKE)
    res = subprocess.run([node, str(script), str(page)], capture_output=True, text=True, timeout=60, check=False)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout)
    assert out["hidden"] is False and out["nav"] == ""
    assert "13 armed" in out["note"] and "6 trips" in out["note"]
    html = out["html"]
    for want in ("trainer.sample_wait", "stop_reason=breaker:trainer.sample_wait", "Dependency checks", "D8",
                 "default (implicit)", "show 1 off / invariant", "55 min / 60 min (92%)", "cost_guard"):
        assert want in html, want
    assert "undefined" not in html and "NaN" not in html
