"""Breakers panel (design 9): resolved settings, status snapshots, trips, alerts, run status."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from yeto.dashboard import alerts, sources
from yeto.dashboard.breakers import (
    LEGACY_TRIPS,
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
    # guards that drop data did their job: counted on their card, not listed as trips
    assert ("trainer.policy_age", "trip") not in names and ("sampler.min_samples", "trip") not in names
    assert not r["trainer.policy_age"]["tripped"] and not r["sampler.min_samples"]["tripped"]
    assert r["trainer.policy_age"]["acted"]["acts"] == 1 and r["trainer.policy_age"]["acted"]["dropped"] == 3
    assert r["sampler.min_samples"]["acted"]["dropped"] == 4 and v["acted_total"] == 2


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
    folded = sum(1 for e in r.events if e["source"].endswith("cost-events.jsonl"))
    assert folded >= 1  # the tick (idle_warn too once the launch layer maps it)
    sources.load_all(r, [str(tmp_path)])  # read again from offset 0 by a new source: not folded twice
    assert sum(1 for e in r.events if e["source"].endswith("cost-events.jsonl")) == folded


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
    assert r.breakers.resolved is not None and r.page_view(now=NOW)["breakers"]["trip_total"] >= 4


R2_HEALTHY = FX / "breakers_r2_healthy"


def legacy(event, t=T0, **kw):
    return {"event": event, "sampler_id": "s-0", "time_unix": t, **kw}


def test_legacy_table_names_entries_that_declare_the_event():
    from yeto.breakers.registry import default_registry

    reg = default_registry()
    for event, rule in LEGACY_TRIPS.items():
        probe = {"event": event, "written": False, "dropped_reason": "below_min_samples", "lost": {"wall_clock_cap": 1},
                 "abandoned_trials": 1, "state": "exhausted", "stale_groups": 1}
        hits = rule(probe)
        assert hits, event
        for name, state, count in hits:
            assert count > 0
            assert name in reg and event in reg.get(name).legacy_events, (event, name)
            assert state in ("trip", "warn", "clear")


def test_healthy_replay_shows_no_trip():
    """A trimmed slice of a healthy run's tapes (every group written, drains clean, tasks
    trained, no stale group, lease expiries recovered): routine telemetry, not trips."""
    r = Reducer()
    sources.load_all(r, [str(R2_HEALTHY)])
    assert r.counts["learner"] == 79
    v = r.page_view(now=T0)
    assert "breakers" not in v  # nothing to show: no trip, no setting, no snapshot
    assert not [a for a in v["overview"]["alerts"] if a["rule"] == "breaker"]


@pytest.mark.parametrize("record,entry,dropped", [
    (legacy("rl_agentic_group", written=False, dropped_reason="below_min_samples", kept=1, trials=8),
     "sampler.min_samples", 1),
    (legacy("rl_agentic_group", written=True, dropped_reason=None, lost={"wall_clock_cap": 2}), "sampler.trial_wall_clock", 2),
    (legacy("rl_agentic_drain", abandoned_trials=0, detached_trials=3), "sampler.drain", 3),
    (legacy("rl_agentic_drain", abandoned_trials=1, detached_trials=0), "sampler.drain", 1),
    ({"event": "rl_buffer_stale_skipped", "island_id": 0, "time_unix": T0, "stale_groups": 3}, "trainer.policy_age", 3),
])
def test_legacy_record_of_a_dropping_guard_is_counted_on_its_card(record, entry, dropped):
    r = Reducer()
    r.feed(record)
    v = r.page_view(now=T0 + 1)["breakers"]
    assert v["trips"] == [] and v["acted_total"] == 1  # the guard acted as designed: no red trip row
    row = rows(v)[entry]
    assert not row["tripped"] and row["last"] is None
    assert (row["acted"]["acts"], row["acted"]["dropped"], row["acted"]["last_ts"]) == (1, dropped, T0)
    assert row["acted"]["legacy_event"] == record["event"]
    assert not [a for a in r.overview(now=T0 + 1)["alerts"] if a["rule"] in ("breaker", "guard_drops")]


@pytest.mark.parametrize("record,entry", [
    (legacy("rl_dispense_settle", state="exhausted", slot=4), "sampler.dispense_redo"),
    (legacy("rl_dispense_settle", state="expired", slot=4), "sampler.dispense_redo"),
    ({"event": "trainer_starvation_stop", "time_unix": T0, "island_id": 0}, "run.starvation"),
])
def test_legacy_record_of_a_breaker_is_a_trip(record, entry):
    r = Reducer()
    r.feed(record)
    v = r.page_view(now=T0 + 1)["breakers"]
    t = v["trips"][0]
    assert (t["name"], t["state"], t["legacy"], t["legacy_event"]) == (entry, "trip", True, record["event"])
    assert rows(v)[entry]["tripped"]


@pytest.mark.parametrize("record", [
    legacy("rl_agentic_group", written=True, dropped_reason=None, lost={}),
    legacy("rl_agentic_group", written=False, dropped_reason="fenced"),  # not the minimum: another entry's
    legacy("rl_agentic_drain", abandoned_trials=0, detached_trials=0, detached_in_flight=0, timed_out=False),
    legacy("rl_dispense_settle", state="trained"),
    legacy("rl_dispense_settle", state="dropped"),
    {"event": "rl_buffer_stale_skipped", "island_id": 0, "time_unix": T0, "stale_groups": 0},
    legacy("rl_dispense_expired", slot=3),  # the lease noticed; the settle tells whether it cost a redo
    legacy("rl_dispense_lease", slot=3),
    legacy("rl_sampler_vram_ok"),
])
def test_routine_legacy_record_is_not_a_trip(record):
    r = Reducer()
    r.feed(record)
    assert "breakers" not in r.page_view(now=T0 + 1)


def test_legacy_warn_and_clear_states():
    r = Reducer()
    r.feed({"event": "trainer_starvation_alert", "time_unix": T0, "island_id": 0})
    r.feed({"event": "trainer_starvation_recovered", "time_unix": T0 + 5, "island_id": 0})
    v = r.page_view(now=T0 + 10)["breakers"]
    assert [t["state"] for t in v["trips"]] == ["clear", "warn"]
    assert not rows(v)["run.starvation"]["tripped"]


def test_layer_with_rl_breaker_ignores_its_legacy_events():
    r = Reducer()
    drop = legacy("rl_agentic_group", written=False, dropped_reason="below_min_samples")
    r.feed(drop)
    r.feed({"event": "rl_buffer_stale_skipped", "island_id": 0, "time_unix": T0, "stale_groups": 2})
    r.feed(legacy("rl_dispense_settle", state="exhausted", slot=1))
    r.feed({"event": "trainer_starvation_stop", "time_unix": T0, "island_id": 0})
    r.feed(brk("sampler.drain", "warn", T0 + 1, action="drop", scope={"sampler_id": "s-0"}))
    v = r.page_view(now=T0 + 2)["breakers"]
    # the sampler layer writes rl_breaker: its legacy rows and acts go; the other layers' stay
    assert sorted((t["name"], t["legacy"]) for t in v["trips"]) == [("run.starvation", True),
                                                                      ("sampler.drain", False)]
    assert "sampler.min_samples" not in rows(v) and rows(v)["trainer.policy_age"]["acted"]["dropped"] == 2


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
    assert "13 armed" in out["note"] and "4 trips" in out["note"] and "2 guard acts" in out["note"]
    html = out["html"]
    for want in ("trainer.sample_wait", "stop_reason=breaker:trainer.sample_wait", "Dependency checks", "D8",
                 "acted 1 time · 3 groups dropped", "acted 1 time · 4 groups dropped",
                 "default (implicit)", "show 1 off / invariant", "55 min / 60 min (92%)", "cost_guard"):
        assert want in html, want
    assert "undefined" not in html and "NaN" not in html


def test_run_up_writes_the_resolved_file(tmp_path):
    from yeto.rl.up import Run

    run = Run.__new__(Run)
    run.dir = tmp_path
    run.cfg = {"run": "r", "trainer": {"argv": ["--rl-stall-timeout", "900"]}, "budget": {"usd": 40}}
    run.write_breakers_resolved()
    doc = json.loads((tmp_path / RESOLVED_FILE).read_text())
    assert doc["schema"] == "yeto-breakers-resolved-v1" and doc["has_section"] is False
    budget = next(e for e in doc["entries"] if e["name"] == "run.budget")
    assert budget["params"]["usd"] == 40 and budget["sources"]["usd"]["kind"] == "legacy"
    r = Reducer()
    assert ResolvedFile([tmp_path / RESOLVED_FILE]).pump(r)
    assert r.budget_usd is None  # a legacy budget does not change the page


def test_cost_guard_tape_name_matches_the_guard():
    from yeto.cost.guard import EVENTS_FILE

    assert sources.COST_EVENTS_FILE == EVENTS_FILE  # launch trips go there too (yeto.launch.trips)


def trainer_rounds(r, n, *, trained=8, stale=0, island=0, t0=T0, every_stale=1):
    """``n`` trainer rounds: the reader's per-round record, then the trained round."""
    for i in range(n):
        t = t0 + 60 * i
        s = stale if i % every_stale == 0 else 0
        r.feed({"event": "rl_buffer_stale_skipped", "island_id": island, "time_unix": t, "rollout_id": i,
                "stale_groups": s, "max_policy_age": 2})
        r.feed({"event": "rl_round_trained", "island_id": island, "time_unix": t + 30, "rollout_id": i,
                "trained_groups": trained})


def guard_alerts(r, now):
    return [(a["sev"], a["title"]) for a in r.overview(now=now)["alerts"] if a["rule"] in ("breaker", "guard_drops")]


def test_occasional_stale_drops_stay_amber():
    """An async run trims a group now and then at a release boundary: the guard's card counts
    it (amber), no trip row, no alert."""
    r = Reducer()
    trainer_rounds(r, 12, stale=1, every_stale=6)  # 2 groups of 98 over 12 rounds
    v = r.page_view(now=T0 + 3600)["breakers"]
    row = rows(v)["trainer.policy_age"]
    assert v["trips"] == [] and not row["tripped"]
    assert row["acted"]["acts"] == 2 and row["acted"]["dropped"] == 2
    assert row["acted"]["rate"]["rounds"] == 10 and row["acted"]["rate"]["ratio"] < 0.1
    assert guard_alerts(r, T0 + 3600) == []


def test_large_stale_drops_go_red():
    r = Reducer()
    trainer_rounds(r, 12, trained=6, stale=4)  # 40 % of the groups aged out, every round
    v = r.page_view(now=T0 + 3600)["breakers"]
    row = rows(v)["trainer.policy_age"]
    assert row["tripped"] and row["acted"]["rate"]["ratio"] == 0.4 and row["acted"]["rate"]["scope"] == "island 0"
    assert guard_alerts(r, T0 + 3600) == [(0, "breaker trainer.policy_age dropped 40% of the groups")]


def test_drop_rate_warning_and_window():
    r = Reducer()
    trainer_rounds(r, 10, trained=8, stale=1)  # 1 of 9: 11 %
    assert guard_alerts(r, T0 + 3600) == [(1, "breaker trainer.policy_age dropped 11% of the groups")]
    trainer_rounds(r, 10, trained=8, stale=0, t0=T0 + 1000)  # 10 clean rounds push them out of the window
    assert guard_alerts(r, T0 + 3600) == []


def test_one_off_burst_stays_on_the_card():
    """Shape of a real replay: 25 stale groups in one round (a backlog aging out at once), none
    before or after; over 20 % of the window, but in one round only: counted on the card, no alert."""
    r = Reducer()
    for i in range(12):
        t = T0 + 60 * i
        r.feed({"event": "rl_buffer_stale_skipped", "island_id": 0, "time_unix": t, "rollout_id": i,
                "stale_groups": 25 if i == 8 else 0, "max_policy_age": 2})
        r.feed({"event": "rl_round_trained", "island_id": 0, "time_unix": t + 30, "rollout_id": i, "trained_groups": 8})
    row = rows(r.page_view(now=T0 + 900)["breakers"])["trainer.policy_age"]
    rate = row["acted"]["rate"]
    assert rate["drop_rounds"] == 1 and rate["ratio"] > 0.2 and not row["tripped"]
    assert row["acted"]["dropped"] == 25
    assert guard_alerts(r, T0 + 900) == []


def test_too_few_groups_are_not_judged():
    r = Reducer()
    trainer_rounds(r, 2, trained=2, stale=3)  # 6 of 10 groups, but only 10 groups
    assert guard_alerts(r, T0 + 600) == [] and not rows(r.page_view(now=T0 + 600)["breakers"])["trainer.policy_age"]["tripped"]


def test_thresholds_come_from_the_monitor_entry():
    from yeto.breakers.layers.monitor import DASHBOARD_RULES

    assert DASHBOARD_RULES["guard_drops"] == "monitor.dashboard_guard_drops"
    assert {k: alerts.DEFAULT_THRESHOLDS[k] for k in ("guard_drop_window_rounds", "guard_drop_min_groups",
                                                       "guard_drop_min_rounds", "guard_drop_warn_ratio",
                                                       "guard_drop_alert_ratio")} == {
        "guard_drop_window_rounds": 10, "guard_drop_min_groups": 20, "guard_drop_min_rounds": 3,
        "guard_drop_warn_ratio": 0.1, "guard_drop_alert_ratio": 0.25}
    r = Reducer(thresholds={"guard_drop_alert_ratio": 0.5})
    trainer_rounds(r, 12, trained=6, stale=4)
    assert guard_alerts(r, T0 + 3600) == [(1, "breaker trainer.policy_age dropped 40% of the groups")]


def sampler_round(r, rid, groups, dropped, sid="s-0", t0=T0):
    for g in range(groups):
        bad = g < dropped
        r.feed({"event": "rl_agentic_group", "sampler_id": sid, "time_unix": t0 + rid * 60 + g, "rollout_id": rid,
                "written": not bad, "dropped_reason": "below_min_samples" if bad else None, "kept": 1 if bad else 8})


def test_sampler_groups_below_minimum_rate():
    r = Reducer()
    for rid in range(4):
        sampler_round(r, rid, 8, 1)  # 1 of 8 per round: 12.5 %
    v = r.page_view(now=T0 + 600)["breakers"]
    row = rows(v)["sampler.min_samples"]
    assert row["acted"]["acts"] == 4 and row["acted"]["rate"]["ratio"] == 0.125 and not row["tripped"]
    assert guard_alerts(r, T0 + 600) == [(1, "breaker sampler.min_samples dropped 12% of the groups")]
    for rid in range(4, 8):
        sampler_round(r, rid, 8, 5)
    assert (0, "breaker sampler.min_samples dropped 38% of the groups") in guard_alerts(r, T0 + 900)


def test_rl_breaker_acts_of_a_dropping_guard():
    r = Reducer()
    for i in range(3):
        r.feed({**brk("trainer.policy_age", "trip", T0 + i, action="drop", value=None, limit=2, scope={"island": "0"},
                      detail=f"rollout {i}: 2 groups older than 2 versions"), "kind": "guard", "count": 2})
    v = r.page_view(now=T0 + 10)["breakers"]
    row = rows(v)["trainer.policy_age"]
    assert v["trips"] == [] and (row["acted"]["acts"], row["acted"]["dropped"]) == (3, 6)
    assert row["acted"]["last_detail"] == "rollout 2: 2 groups older than 2 versions"
    assert guard_alerts(r, T0 + 10) == []


def _f1_reducer(tmp_path, stale=None) -> Reducer:
    """The f1 follow-trainer replay (22 rl_buffer_stale_skipped records, all 0); with ``stale``,
    every one of them is rewritten to drop that many groups."""
    src = FX / "diloco" / "f1-follow"
    if stale is not None:
        dst = tmp_path / "f1"
        shutil.copytree(src, dst)
        for f in (dst / "events").glob("*.jsonl"):
            out = []
            for line in f.read_text().splitlines():
                d = json.loads(line) if line.strip() else None
                if d and d.get("event") == "rl_buffer_stale_skipped":
                    d["stale_groups"] = stale
                out.append(json.dumps(d) if d else line)
            f.write_text("\n".join(out) + "\n")
        src = dst
    r = Reducer()
    sources.load_all(r, [str(src)])
    return r


def test_f1_replay_is_clean(tmp_path):
    r = _f1_reducer(tmp_path)
    v = r.page_view()["breakers"]
    assert v["trips"] == [] and v["acted_total"] == 0
    assert not any(x["tripped"] for L in v["layers"] for x in L["rows"])
    assert [a for a in r.overview()["alerts"] if a["rule"] in ("breaker", "guard_drops")] == []


def test_f1_replay_with_real_stale_drops_goes_red(tmp_path):
    r = _f1_reducer(tmp_path, stale=4)  # 4 groups aged out next to 8 trained, every round
    v = r.page_view()["breakers"]
    row = rows(v)["trainer.policy_age"]
    assert v["trips"] == [] and row["tripped"] and row["acted"]["acts"] == 22
    assert [(a["sev"], a["rule"]) for a in r.overview()["alerts"] if a["rule"] in ("breaker", "guard_drops")] == [
        (0, "guard_drops")]
