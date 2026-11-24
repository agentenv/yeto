"""How a run that ended reads on the page, and which policy version a trainer card shows.

Fixtures (``fixtures/dashboard/run_end``) are trimmed tapes of two customer-path async GRPO
runs: ``c1-main`` finished on its own (``rl_run_end`` data_ended, the launcher's
``teardown.json`` after it), ``c1-dl`` releases every 6 rounds (its trainer runs ahead of its
last release).
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess

import pytest

from yeto.dashboard.export import export_html
from yeto.dashboard.reducer import Reducer
from yeto.dashboard.sources import load_all, operator_stops_near

from dashboard_helpers import FX

RUN_END = FX / "run_end"


def load(*paths, up_state=None) -> Reducer:
    r = Reducer()
    if up_state is not None:
        r.up_state = str(up_state)
    paths = [str(p) for p in paths]
    load_all(r, paths)
    for rec in operator_stops_near(paths):
        r.feed_operator_stop(rec)
    return r


def copy_run(tmp_path, name):
    dst = tmp_path / name
    shutil.copytree(RUN_END / name, dst)
    return dst


# -- the run's end ----------------------------------------------------------------------------
def test_normal_finish_reads_finished_not_stopped_by_us():
    r = load(RUN_END / "c1-main")
    ov = r.overview()
    assert ov["global_status"] == {"level": "ok", "text": "finished (data_ended, buffer_exhausted)"}
    assert ov["run_end"]["kind"] == "finished" and ov["run_end"]["finish"]["reason"] == "data_ended"
    assert ov["operator_stop"]["cause"] == "teardown"  # still recorded, as the launcher's cleanup
    assert [(c["status"], c["stopped_by_us"]) for c in ov["islands"]] == [("done", False)]


def test_teardown_without_a_normal_end_is_our_stop(tmp_path):
    run = copy_run(tmp_path, "c1-main")
    tape = run / "events" / "c1-main-l0-modal.jsonl"
    tape.write_text("".join(line + "\n" for line in tape.read_text().splitlines()
                            if json.loads(line)["event"] != "rl_run_end"))
    ov = load(run).overview()
    assert ov["global_status"]["text"] == "stopped (stopped by us: teardown)"
    assert ov["run_end"]["kind"] == "stopped"


def test_stop_marker_before_the_end_is_our_stop(tmp_path):
    """A gate / watchdog stop issued before the trainer's own end: the run was stopped."""
    run = copy_run(tmp_path, "c1-main")
    end = next(json.loads(x) for x in (run / "events" / "c1-main-l0-modal.jsonl").read_text().splitlines()
               if json.loads(x)["event"] == "rl_run_end")["time_unix"]
    stamp = dt.datetime.fromtimestamp(end - 120, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    (run / "STOP_ISSUED_utc.txt").write_text(f"{stamp} watchdog\n")
    ov = load(run).overview()
    assert ov["global_status"]["text"] == "stopped (stopped by us: watchdog)"


def test_run_up_breaker_stop_is_our_stop(tmp_path):
    """``run up`` recorded a breaker stop (``stop_reason`` in its state file) before the end."""
    run = copy_run(tmp_path, "c1-main")
    end = next(json.loads(x) for x in (run / "events" / "c1-main-l0-modal.jsonl").read_text().splitlines()
               if json.loads(x)["event"] == "rl_run_end")["time_unix"]
    at = dt.datetime.fromtimestamp(end - 60, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    state = tmp_path / "up" / "state.json"
    state.parent.mkdir()
    state.write_text(json.dumps({"breaker": {"at": at, "why": "spend over budget", "stop_reason": "breaker:run.budget"}}))
    ov = load(run, up_state=state).overview()
    assert ov["global_status"]["text"] == "stopped (stopped by us: breaker:run.budget)"
    # the same state file without a stop: the normal finish
    state.write_text(json.dumps({"finished_banner": at}))
    assert load(run, up_state=state).overview()["global_status"]["text"].startswith("finished (data_ended")


def test_a_stop_after_the_finish_is_cleanup(tmp_path):
    run = copy_run(tmp_path, "c1-main")
    end = next(json.loads(x) for x in (run / "events" / "c1-main-l0-modal.jsonl").read_text().splitlines()
               if json.loads(x)["event"] == "rl_run_end")["time_unix"]
    stamp = dt.datetime.fromtimestamp(end + 300, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    (run / "STOP_ISSUED_utc.txt").write_text(f"{stamp} early\n")
    assert load(run).overview()["global_status"]["text"] == "finished (data_ended, buffer_exhausted)"


def test_a_running_run_has_no_end():
    ov = load(RUN_END / "c1-dl").overview()
    assert ov["run_end"] is None and ov["global_status"]["text"] == "all healthy"


# -- local vs released policy version -----------------------------------------------------------
def test_card_carries_the_last_release(tmp_path):
    c = load(RUN_END / "c1-dl").overview()["islands"][0]
    assert (c["policy_version"], c["released_version"]) == (17, 12)
    t = load(with_sampler(tmp_path)).page_view()["topology"]["trainers"][0]
    assert (t["policy_version"], t["released_version"]) == (17, 12)


SMOKE = r"""
const fs = require("fs");
const html = fs.readFileSync(process.argv[2], "utf8");
const data = html.match(/<script type="application\/json" id="yeto-data">([\s\S]*?)<\/script>/)[1];
const app = html.match(/<script>\n([\s\S]*?)\n<\/script>/)[1];
const els = {}, svg = [];
function node(id) {
  return { id, innerHTML: "", textContent: "", hidden: false, style: {}, dataset: {}, children: [], attrs: {},
    listeners: {}, offsetWidth: 200, offsetHeight: 120, clientWidth: 1200,
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    setAttribute(k, v) { this.attrs[k] = String(v); }, removeAttribute(k) { delete this.attrs[k]; },
    appendChild(c) { this.children.push(c); return c; },
    addEventListener(t, fn) { (this.listeners[t] = this.listeners[t] || []).push(fn); },
    getBoundingClientRect() { return { left: 0, top: 0, width: 560, height: 400 }; },
    querySelectorAll() { return []; } };
}
function mk(id) { if (!els[id]) { els[id] = node(id); if (id === "yeto-data") els[id].textContent = data; } return els[id]; }
global.document = { documentElement: node("html"), getElementById: mk, querySelectorAll: () => [], title: "",
  createElementNS: () => { const n = node(null); svg.push(n); return n; }, createElement: () => node(null) };
global.navigator = {}; global.localStorage = { getItem() { return null; }, setItem() {} };
global.setInterval = () => 0; global.setTimeout = () => 0;
eval(app);
console.log(JSON.stringify({ status: mk("navSub").textContent, sub: mk("sub").textContent, wall: mk("aHead").innerHTML, kpis: mk("rKpis").innerHTML,
  nodes: mk("rNodes").innerHTML, texts: svg.map((n) => String(n.textContent || "")).filter(Boolean) }));
"""


def smoke(tmp_path, reducer):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    page = export_html(reducer, tmp_path / "p.html", generated_at=reducer.max_ts)
    script = tmp_path / "smoke.js"
    script.write_text(SMOKE)
    res = subprocess.run([node, str(script), str(page)], capture_output=True, text=True, timeout=60, check=False)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def with_sampler(tmp_path):
    """c1-dl plus a sampler tape generating with the last release: an async page (the trainer's
    versions are in its island block header), as the live page with the sampler tapes from the run store."""
    run = copy_run(tmp_path, "c1-dl")
    rows = [json.loads(x) for x in (run / "events" / "c1-dl-l0-modal.jsonl").read_text().splitlines()]
    t0 = rows[0]["time_unix"]
    rel = [x for x in rows if x["event"] == "rl_policy_released"]
    recs = [{"event": "rl_sampler_start", "sampler_id": "c1-dl-s0", "time_unix": t0 + 5}]
    for i, x in enumerate(rel):
        recs.append({"event": "rl_sampler_policy_loaded", "sampler_id": "c1-dl-s0", "policy_version": x["policy_version"],
                     "time_unix": x["time_unix"] + 20})
    for i in range(8):
        recs.append({"event": "rl_sampler_round", "sampler_id": "c1-dl-s0", "rollout_id": i, "groups": 4,
                     "policy_version": rel[-1]["policy_version"], "time_unix": rows[-1]["time_unix"] - 400 + 50 * i})
    (run / "events" / "c1-dl-s0.jsonl").write_text("".join(json.dumps(x) + "\n" for x in recs))
    return run


def test_page_names_local_and_released_versions(tmp_path):
    out = smoke(tmp_path, load(with_sampler(tmp_path)))
    assert "local v17 · released v12" in out["wall"]
    assert ">v17<" not in out["wall"]  # never the bare local version beside the release
    assert "local v17 / 2" in out["kpis"] and "last release v12" in out["kpis"]
    assert "local v17" in out["nodes"] and "released v12" in out["nodes"]
    assert any("local v17 · released v12" in t for t in out["texts"]), out["texts"]


def test_page_reads_finished(tmp_path):
    out = smoke(tmp_path, load(RUN_END / "c1-main"))
    assert out["status"] == "finished (data_ended, buffer_exhausted)"
    assert "stopped by us" not in out["status"]
    assert out["sub"].endswith(" · run finished (data_ended, buffer_exhausted)")


def test_page_reads_stopped_by_us(tmp_path):
    run = copy_run(tmp_path, "c1-main")
    tape = run / "events" / "c1-main-l0-modal.jsonl"
    tape.write_text("".join(line + "\n" for line in tape.read_text().splitlines()
                            if json.loads(line)["event"] != "rl_run_end"))
    out = smoke(tmp_path, load(run))
    assert out["sub"].endswith(" · run stopped (stopped by us: teardown)")


def test_running_page_has_no_end(tmp_path):
    assert " · run " not in smoke(tmp_path, load(RUN_END / "c1-dl"))["sub"]


# -- the run up monitor's tape and the cost bar -------------------------------------------------
def dashboard_args(*argv):
    import argparse

    from yeto.dashboard import cli

    p = argparse.ArgumentParser()
    cli.add_parser(p.add_subparsers(dest="command"))
    return p.parse_args(["dashboard", "export", "-o", "x.html", *argv])


def from_args(monkeypatch, name, *argv, state_dir=None):
    """A dashboard over the ``run_end/<name>`` fixture: its run directory, its ``run up`` state."""
    from yeto.dashboard import cli

    base = RUN_END / name
    monkeypatch.setattr("yeto.runs.run_dir", lambda run: base / "run-dirs" / run)
    monkeypatch.setattr("yeto.launcher.RL_SYNCER_EVENT_TAPE", str(base / "no-syncer.jsonl"), raising=False)
    if state_dir is not None:
        monkeypatch.setenv("YETO_STATE_DIR", str(state_dir))
    args = dashboard_args("--run", name, *argv)
    r = cli._reducer(args)
    paths = cli._paths(args)
    load_all(r, paths)
    return r, paths


def test_up_state_dir_feeds_the_run_layer_breakers(monkeypatch, tmp_path):
    """Only --up-state names the run up state (YETO_STATE_DIR points elsewhere, or is unset):
    the run-layer trip of its events.jsonl reaches the Breakers panel."""
    state = RUN_END / "c1-wc" / "rl-up" / "c1-wc" / "state.json"
    r, paths = from_args(monkeypatch, "c1-wc", "--up-state", str(state), state_dir=tmp_path / "elsewhere")
    assert str(state.parent / "events.jsonl") in paths
    assert not any(str(tmp_path) in p for p in paths)
    assert [(t["name"], t["state"]) for t in r.page_view()["breakers"]["trips"]] == [("run.wall_clock", "trip")]
    assert r.overview()["global_status"]["text"] == "stopped (stopped by us: breaker:run.wall_clock)"


def test_state_dir_is_the_fallback_without_up_state(monkeypatch):
    r, paths = from_args(monkeypatch, "c1-wc", state_dir=RUN_END / "c1-wc" / "rl-up")
    assert str(RUN_END / "c1-wc" / "rl-up" / "c1-wc" / "events.jsonl") in paths
    assert [t["name"] for t in r.page_view()["breakers"]["trips"]] == ["run.wall_clock"]


def test_up_state_without_run(monkeypatch):
    from yeto.dashboard import cli

    state = RUN_END / "c1-bk" / "rl-up" / "c1-bk" / "state.json"
    args = dashboard_args("--tapes", str(RUN_END / "c1-bk" / "run-dirs" / "c1-bk"), "--up-state", str(state))
    assert cli._paths(args)[-1] == str(state.parent / "events.jsonl")


def test_over_budget_reads_over_without_an_eta(monkeypatch):
    state = RUN_END / "c1-bk" / "rl-up" / "c1-bk" / "state.json"
    r, _ = from_args(monkeypatch, "c1-bk", "--up-state", str(state), "--budget", "0.5")
    ov = r.overview()
    budget = [a for a in ov["alerts"] if a["rule"] == "budget"]
    assert len(budget) == 1 and budget[0]["sev"] == 0 and budget[0]["title"] == "over budget"
    assert budget[0]["detail"].startswith("over budget: $0.") and "/ $0.50 (1" in budget[0]["detail"]
    assert "reached in" not in budget[0]["detail"]
    assert [t["name"] for t in r.page_view()["breakers"]["trips"]] == ["run.budget"]
    assert ov["global_status"]["text"] == "stopped (stopped by us: breaker:run.budget)"


def test_budget_below_the_cap_keeps_its_eta():
    from yeto.dashboard import alerts

    th = alerts.DEFAULT_THRESHOLDS
    cost = {"budget_pct": 85.0, "total_usd": 0.85, "budget_usd": 1.0, "hours_to_cap": 0.4, "burn_usd_h": 0.375}
    assert alerts.budget(cost, th)[0]["detail"].endswith("reached in about 0.4 h at $0.38/h")
    cost.update(hours_to_cap=0.0)  # at the cap this instant: no "about 0.0 h"
    assert "reached in" not in alerts.budget(cost, th)[0]["detail"]
