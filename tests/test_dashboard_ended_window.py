"""A run that ended: every time window of the RL page ends at its last event (plus a little room), not at now,
so the data spans the charts; the window's wording follows ("over the whole run (N min)"). A live run keeps
the window that ends at now."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from test_dashboard_run_end import RUN_END, load

from yeto.dashboard.export import render_page
from yeto.dashboard.reducer import Reducer

T0 = 1_790_000_000.0

SMOKE = r"""
const fs=require('fs');const html=fs.readFileSync(process.argv[2],'utf8');
const dataM=html.match(/<script type="application\/json" id="yeto-data">([\s\S]*?)<\/script>/);
const appM=html.match(/<script>\n([\s\S]*?)\n<\/script>/);
const els={};
function node(id){return {id,innerHTML:"",textContent:"",hidden:false,style:{},dataset:{},children:[],attrs:{},className:"",
  classList:{add(){},remove(){},toggle(){},contains(){return false}},setAttribute(k,v){this.attrs[k]=String(v)},
  removeAttribute(){},appendChild(c){this.children.push(c);return c},addEventListener(){},clientWidth:1200,
  getBoundingClientRect(){return {left:0,top:0,width:1200,height:400}},querySelectorAll(){return []},querySelector(){return null}}}
function mk(id){if(!els[id]){els[id]=node(id);if(id==="yeto-data"&&dataM)els[id].textContent=dataM[1]}return els[id]}
global.document={documentElement:node("html"),getElementById:(id)=>mk(id),createElementNS:()=>node(null),
  createElement:()=>node(null),querySelectorAll:()=>[],title:""};
global.navigator={};global.localStorage={getItem(){return null},setItem(){}};
global.setInterval=()=>0;global.setTimeout=()=>0;global.location={hash:"",search:""};
eval(appM[1]);
process.stdout.write(JSON.stringify({win:YetoExt.window(),fleetH:mk("rFleetH").textContent,costH:mk("rCostH").textContent,
  distH:mk("rlDistH").textContent,cost:mk("rCostK").innerHTML,none:mk("rFleetNone").innerHTML}));
"""


def _page(tmp_path: Path, r: Reducer, now: float) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    page = tmp_path / "p.html"  # the page as the server renders it at ``now`` (statuses and NOW at that time)
    page.write_text(render_page(r.full_view(now=now)), encoding="utf-8")
    script = tmp_path / "smoke.js"
    script.write_text(SMOKE)
    res = subprocess.run([node, str(script), str(page)], capture_output=True, text=True, timeout=60, check=False)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def _span_txt(s: float) -> str:
    m = s / 60
    return f"{round(m)} min" if m < 120 else f"{s / 3600:.1f} h"


def test_an_ended_run_window_ends_at_its_last_event(tmp_path):
    """A finished single-trainer run looked at 15 hours later: the window ends at the last event plus 2% of the
    run (at least a minute), and the headers name the whole run's span."""
    r = load(RUN_END / "c1-main")
    ov = r.overview()
    assert ov["run_end"]["kind"] == "finished"
    last, first = ov["data_ts"], ov["first_ts"]
    out = _page(tmp_path, r, now=last + 15 * 3600)
    win, span = out["win"], last - first
    assert win["ended"] is True and win["now"] == pytest.approx(last + 15 * 3600)
    assert win["t1"] == pytest.approx(last + max(60, 0.02 * span))
    assert win["t0"] <= first < win["t1"] and win["t1"] - win["t0"] < 1.2 * span + 300  # the run fills the window
    assert out["fleetH"] == f"Fleet over the whole run ({_span_txt(span)})"
    assert out["costH"] == f"Cost / efficiency over the whole run ({_span_txt(span)})"
    assert "nothing before the run" in out["cost"] and "no previous" not in out["cost"]


def _live(now_offset: float) -> Reducer:
    r = Reducer(run="live")
    for i in range(6):
        r.feed({"event": "rl_round_trained", "learner_id": "0", "rollout_id": i, "time_unix": T0 + 60 * i,
                "trained_groups": 8, "reward_mean": 0.5})
        r.feed({"event": "rl_sampler_round", "sampler_id": "s0", "rollout_id": i, "groups": 8, "policy_version": i,
                "time_unix": T0 + 30 + 60 * i})
    return r


def test_a_live_run_window_still_ends_at_now(tmp_path):
    r = _live(0)
    now = r.max_ts + 20  # the trainer reported 20 s ago: running
    out = _page(tmp_path, r, now=now)
    assert out["win"]["ended"] is False and out["win"]["t1"] == pytest.approx(now)
    assert out["fleetH"].startswith("Fleet over the last ") and out["fleetH"].endswith("(whole run)")


def test_a_run_whose_islands_are_gone_and_quiet_counts_as_ended(tmp_path):
    """No run end on the tapes, but every island is gone (lost / stale) and the tapes are quiet for 10 min."""
    r = _live(0)
    last = r.max_ts
    out = _page(tmp_path, r, now=last + 3 * 3600)
    assert out["win"]["ended"] is True and out["win"]["t1"] == pytest.approx(last + 60)
    # quiet for less than 10 min: still live, the window runs to now
    out = _page(tmp_path, r, now=last + 300)
    assert out["win"]["ended"] is False and out["win"]["t1"] == pytest.approx(last + 300)


def test_an_open_run_up_keeps_a_quiet_run_live(tmp_path):
    r = _live(0)
    last = r.max_ts
    state = tmp_path / "state.json"
    # an up with no end whose process is alive (this test's own pid): the launch is still running
    state.write_text(json.dumps({"ups": [{"start": last - 600, "pid": os.getpid()}]}))
    r.up_state = str(state)
    out = _page(tmp_path, r, now=last + 3 * 3600)
    assert out["win"]["ended"] is False
