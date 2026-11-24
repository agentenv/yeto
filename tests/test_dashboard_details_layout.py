"""The Details area of the RL page (app.js): an island switch drives one block per island (trainer
rounds, the island's releases with one load tag per sampler, Samplers & pools, backlog, release
timeline, per-round drill-down); "Fleet" holds the run-wide ledgers; empty panels are folded."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from test_dashboard_fleet_islands import T0, fleet_reducer

from yeto.dashboard.export import export_html
from yeto.dashboard.reducer import Reducer
from yeto.dashboard.sources import load_all

FX = Path(__file__).parent / "fixtures" / "dashboard"

# the page script under a DOM stub; PICK is what localStorage holds for the island switch
SMOKE = r"""
const fs=require('fs');const html=fs.readFileSync(process.argv[2],'utf8');const PICK=process.argv[3]||null;
const dataM=html.match(/<script type="application\/json" id="yeto-data">([\s\S]*?)<\/script>/);
const appM=html.match(/<script>\n([\s\S]*?)\n<\/script>/);
const els={},svg=[];
function node(id){return {id,innerHTML:"",textContent:"",hidden:false,style:{},dataset:{},children:[],attrs:{},className:"",
  classList:{add(){},remove(){},toggle(){},contains(){return false}},setAttribute(k,v){this.attrs[k]=String(v)},
  removeAttribute(){},appendChild(c){this.children.push(c);c.parent=this;return c},addEventListener(){},clientWidth:900,
  getBoundingClientRect(){return {left:0,top:0,width:900,height:400}},querySelectorAll(){return []},querySelector(){return null}}}
function mk(id){if(!els[id]){els[id]=node(id);if(id==="yeto-data"&&dataM)els[id].textContent=dataM[1]}return els[id]}
global.document={documentElement:node("html"),getElementById:(id)=>mk(id),createElementNS:()=>{const n=node(null);svg.push(n);return n},
  createElement:()=>node(null),querySelectorAll:()=>[],title:""};
global.navigator={};global.localStorage={getItem(k){return /^yeto-pick/.test(k)?PICK:null},setItem(){}};
global.setInterval=()=>0;global.setTimeout=()=>0;global.location={hash:"",search:""};
eval(appM[1]);
const ids=["islSw","async","aHead","aTrainer","aRelTable","aSamplers","legBacklog","fleetBox","fleetTbl","rounds","roundsIsl","legMetric",
  "rDurBox","util","nodes","runWide","syncerBox","aDispBox","aDisp","foldBox","detailsCs"];
const out={};ids.forEach(i=>{const e=mk(i);out[i]={html:e.innerHTML,text:e.textContent,hidden:e.hidden,cls:e.className}});
// release timeline: every version label (text "vN") with its position
const rel=mk("aRelChart");const inRel=(n)=>{for(let p=n;p;p=p.parent)if(p===rel)return true;return false};
out.vlabels=svg.filter(n=>inRel(n)&&/^v\d+$/.test(String(n.textContent||""))&&n.attrs.x!==undefined).map(n=>[n.textContent,+n.attrs.x,+n.attrs.y]);
process.stdout.write(JSON.stringify(out));
"""


def _page(tmp_path: Path, r: Reducer, pick: str | None = None) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    page = export_html(r, tmp_path / "p.html", generated_at=r.max_ts)
    script = tmp_path / "smoke.js"
    script.write_text(SMOKE)
    res = subprocess.run([node, str(script), str(page), pick or ""], capture_output=True, text=True, timeout=60, check=False)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def _fleet() -> Reducer:
    r = fleet_reducer()
    # a task ledger: one recover (listed) and two unwritten slots handed back on a reload (counted only)
    for i, rec in enumerate([
            {"event": "rl_dispense_claim", "sampler_id": "qa-v-i0-0", "slot": 1, "attempt": 0, "time_unix": T0 + 30},
            {"event": "rl_dispense_settle", "sampler_id": "qa-v-i0-0", "slot": 1, "attempt": 0, "state": "trained", "time_unix": T0 + 60},
            {"event": "rl_dispense_release", "sampler_id": "qa-v-i0-0", "slot": 2, "attempt": 0, "time_unix": T0 + 70},
            {"event": "rl_dispense_release", "sampler_id": "qa-s-i0-0", "slot": 3, "attempt": 0, "time_unix": T0 + 71},
            {"event": "rl_dispense_recover", "sampler_id": "qa-a-i1-0", "time_unix": T0 + 80}]):
        r.feed(rec, source="ledger", offset=i)
    return r


def test_fleet_switch_lists_islands_and_fleet(tmp_path):
    out = _page(tmp_path, _fleet())
    sw = out["islSw"]
    assert sw["hidden"] is False
    assert ">Island 0<" in sw["html"] and ">Island 1<" in sw["html"] and ">Fleet<" in sw["html"]
    assert 'aria-selected="true" class="on" data-k="i0"' in sw["html"]  # first island by default
    assert "island wall" not in json.dumps(out)


def test_island_1_shows_its_own_trainer_samplers_and_releases(tmp_path):
    out = _page(tmp_path, _fleet(), "i1")
    assert out["async"]["hidden"] is False and out["fleetBox"]["hidden"] is True
    assert "Island 1 · west-h100" in out["aHead"]["html"] and "trainer 1" in out["aHead"]["html"]
    assert "pick another trainer" not in out["aTrainer"]["html"] and "trained groups" in out["aTrainer"]["html"]
    sp = out["aSamplers"]["html"]
    assert ">sampler qa-a-i1-0<" in sp and ">sampler qa-b-i1-0<" in sp and "qa-v-i0-0" not in sp and "qa-s-i0-0" not in sp
    rel = out["aRelTable"]["html"]
    # two SGLang pools on island 1: numbered tags, each named once in the legend; nothing of island 0
    assert '<span class="lt">SGL 1</span>sampler qa-a-i1-0' in rel and '<span class="lt">SGL 2</span>sampler qa-b-i1-0' in rel
    assert "qa-v-i0-0" not in rel and "vLLM" not in rel
    assert rel.count("<tr>") == 6  # header + v1..v5 of island 1 only
    # per-round drill-down follows the island: the trainer alone in the legend, no sampler
    assert out["roundsIsl"]["text"] == "· trainer 1"
    assert "sampler" not in out["legMetric"]["html"] and "trainer 1" in out["legMetric"]["html"]
    # run-wide ledgers are under "Fleet"
    assert out["runWide"]["hidden"] is True and out["aDispBox"]["hidden"] is True


def test_release_tags_mark_unloaded_samplers(tmp_path):
    r = _fleet()
    # island 0 releases v6, only its vLLM pool loads it
    r.feed({"event": "rl_policy_released", "island_id": 0, "time_unix": T0 + 5000, "policy_version": 6, "local_version": 6,
            "released_at": T0 + 4999, "release_timings": {"upload_s": 9.0}, "origin": {"kind": "local"}}, source="x", offset=0)
    r.feed({"event": "rl_sampler_policy_loaded", "sampler_id": "qa-v-i0-0", "time_unix": T0 + 5100, "policy_version": 6,
            "released_at": T0 + 4999}, source="y", offset=0)
    rel = _page(tmp_path, r, "i0")["aRelTable"]["html"]
    row = rel.split("<td class=\"l num\">v6</td>", 1)[1].split("</tr>", 1)[0]
    assert '<span class="lt" title="sampler qa-v-i0-0 loaded v6' in row and ">vLLM 101s<" in row
    assert '<span class="lt no" title="sampler qa-s-i0-0 never loaded v6">SGL</span>' in row


def test_fleet_view_holds_the_island_summary_and_the_ledgers(tmp_path):
    out = _page(tmp_path, _fleet(), "fleet")
    assert out["async"]["hidden"] is True and out["fleetBox"]["hidden"] is False
    ft = out["fleetTbl"]["html"]
    assert "Island 0" in ft and "Island 1" in ft and "i0-vllm-h100" in ft and "i1-sgl-b" in ft
    # trainer metrics of every island side by side, nothing else of the drill-down
    assert out["roundsIsl"]["text"] == "· every trainer of the fleet" and out["rDurBox"]["hidden"] is True
    assert "trainer 0" in out["legMetric"]["html"] and "trainer 1" in out["legMetric"]["html"]
    assert out["util"]["hidden"] is True
    # the task ledger lists the recover, counts the releases
    d = out["aDisp"]["html"]
    assert out["aDispBox"]["hidden"] is False and out["runWide"]["hidden"] is False
    assert ">recover<" in d and ">release<" not in d and "2 unwritten slots were handed back on a reload" in d
    assert out["syncerBox"]["hidden"] is False and out["runWide"]["cls"] == "dg2"  # two ledgers: two to the row


def test_release_timeline_labels_never_overlap(tmp_path):
    r = fleet_reducer()
    # many releases a few seconds apart on island 0: labels that do not fit are left out
    for v in range(6, 30):
        t = T0 + 4000 + 3 * v
        r.feed({"event": "rl_policy_released", "island_id": 0, "time_unix": t, "policy_version": v, "local_version": v,
                "released_at": t, "origin": {"kind": "local"}}, source="x", offset=v)
    labels = _page(tmp_path, r, "i0")["vlabels"]
    assert labels and len(labels) < 29
    by_row: dict[float, list] = {}
    for name, x, y in labels:
        by_row.setdefault(y, []).append((x, 7.3 * len(name)))
    assert len(by_row) <= 2
    for row in by_row.values():
        row.sort()
        for (x0, w0), (x1, w1) in zip(row, row[1:]):
            assert x1 - w1 / 2 >= x0 + w0 / 2, row


def test_single_trainer_async_has_no_switch_and_the_same_block(tmp_path):
    from yeto.dashboard import sources

    r = Reducer(run="async-compaction")
    d = FX / "diloco" / "async-compaction"
    load_all(r, [str(p) for p in sources.discover([str(d), str(d / "events")])])
    out = _page(tmp_path, r)
    assert out["islSw"]["hidden"] is True
    assert out["async"]["hidden"] is False and out["aHead"]["html"].startswith("Async GRPO")
    assert "trained groups" in out["aTrainer"]["html"] and "<tr>" in out["aRelTable"]["html"]
    assert out["aSamplers"]["html"].count('class="nn nc"') == 2
    assert out["runWide"]["hidden"] is True  # no syncer, no ledger on this run
    assert out["nodes"]["hidden"] is True  # island health lives in the block header and the sampler table
    # utilization charts without data are folded, each named with what it needs
    assert out["util"]["hidden"] is True
    for want in ("Cutoff drop / carry-over", "KV usage and queueing", "no rl_load_sample"):
        assert want in out["foldBox"]["html"], want


def test_synchronous_multi_island_run_keeps_island_health_and_overlay(tmp_path):
    from test_dashboard_diloco import load

    out = _page(tmp_path, load("run-b"))
    assert out["islSw"]["hidden"] is False and ">trainer 1<" in out["islSw"]["html"]
    assert out["async"]["hidden"] is True and out["nodes"]["hidden"] is False
    assert "trainer 0" in out["legMetric"]["html"] and "trainer 1" in out["legMetric"]["html"]  # one round clock: overlaid
    assert out["syncerBox"]["hidden"] is False
