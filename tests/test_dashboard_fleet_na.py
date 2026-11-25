"""A fleet of async islands on the RL page shows no bare "n/a": the buffer backlog of every island lane,
zero-variance groups a pool keeps, the finish from the syncer's total steps, no MFU card (no FLOP counts on
the tapes), a dash for a count one sampler does not report, the release cadence a trainer's tape names; the
event stream in time order; a wide table marks that it scrolls."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from test_dashboard_fleet_islands import ROUNDS, T0, fleet_reducer, plan_doc

from yeto.dashboard.export import export_html
from yeto.dashboard.reducer import Reducer

STATIC = Path(__file__).parent.parent / "yeto" / "dashboard" / "static"
LAST = T0 + 100 + 60 * (ROUNDS - 1)

SMOKE = r"""
const fs=require('fs');const html=fs.readFileSync(process.argv[2],'utf8');const PICK=process.argv[3]||null;
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
global.navigator={};global.localStorage={getItem(k){return /^yeto-pick/.test(k)?PICK:null},setItem(){}};
global.setInterval=()=>0;global.setTimeout=()=>0;global.location={hash:"",search:""};
eval(appM[1]);
const out={};["rKpis","rCostK","rFleet","aSamplers","aTrainer","fleetTbl","costBox","evs"].forEach(i=>{out[i]=mk(i).innerHTML});
process.stdout.write(JSON.stringify(out));
"""


def _node() -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    return node


def _page(tmp_path: Path, r: Reducer, pick: str | None = None) -> dict:
    page = export_html(r, tmp_path / "p.html", generated_at=r.max_ts)
    script = tmp_path / "smoke.js"
    script.write_text(SMOKE)
    res = subprocess.run([_node(), str(script), str(page), pick or ""], capture_output=True, text=True, timeout=60,
                         check=False)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def _syncer_config(total: int) -> dict:
    return {"kind": "syncer_config", "fragments": 2, "pipeline": 1, "quorum": 2, "learners": 2, "total_steps": total,
            "last_step": total, "sync_interval_steps": 1.0, "shard_index": 0, "shard_count": 1, "time_unix": T0 + 50}


def _fleet(total: int = 12, *, engine_rounds: bool = True) -> Reducer:
    """The fleet fixture with a syncer config; island 0's vLLM pool reports its engine summary
    (zero-variance groups seen, not dropped: the pool keeps them), island 0's SGLang pool does not;
    island 0's samplers write 40 groups after the last read."""
    r = fleet_reducer()
    r.feed(_syncer_config(total))
    if engine_rounds:
        for i in range(ROUNDS):
            r.feed({"event": "rl_sampler_engine_round", "sampler_id": "qa-v-i0-0", "rollout_id": i, "samples": 8,
                    "response_tokens": 800, "zero_variance_seen": 1, "finish": {"length": 2}, "reward_mean": 0.5,
                    "time_unix": T0 + 70 + 60 * i})
    for j, sid in enumerate(("qa-v-i0-0", "qa-s-i0-0")):
        r.feed({"event": "rl_sampler_round", "sampler_id": sid, "time_unix": LAST + 40 + j, "rollout_id": ROUNDS,
                "groups": 20, "policy_version": 5, "version_lag": 0, "generate_s": 28.0})
    return r


def test_fleet_kpis_name_every_island_backlog_and_kept_zero_variance(tmp_path):
    r = _fleet()
    view = r.page_view(now=r.max_ts)
    assert view["async"]["backlog"] is None  # one buffer per island lane
    out = _page(tmp_path, r)
    kp = out["rKpis"]
    assert "n/a" not in kp
    last = [x["backlog"]["points"][-1][3] for x in view["async"]["islands"]]
    assert last[0] > 0
    assert ">Buffer backlog<" in kp and f">{last[0] + last[1]:.0f}<" in kp
    assert f"island 0: {last[0]:.0f} · island 1: {last[1]:.0f}" in kp
    # the rejected counts: stale / zero-var (kept by the pool: 0 dropped) / truncated / infra
    assert "zero-var: 10 of" in kp and "kept by the pools" in kp
    assert "qa-s-i0-0" in kp.split("kept by the pools", 1)[1]  # the pool without a zero-variance count is named


def test_fleet_cost_has_no_mfu_card_and_paces_the_finish_by_the_syncer(tmp_path):
    out = _page(tmp_path, _fleet(total=12))
    ck = out["rCostK"]
    assert "n/a" not in ck and ">MFU<" not in ck
    assert ">Projected finish<" in ck and "syncer step 10 of 12 at its mean merge pace" in ck


def test_fleet_finished_by_the_syncer_total(tmp_path):
    out = _page(tmp_path, _fleet(total=10))
    ck = out["rCostK"]
    assert ">Finished<" in ck and "syncer reached step 10 of 10 (final cut)" in ck and "n/a" not in ck


def test_fleet_without_a_syncer_total_says_so(tmp_path):
    r = fleet_reducer()
    out = _page(tmp_path, r)
    ck = out["rCostK"]
    assert "n/a" not in ck and "no total step count yet" in ck


def test_single_trainer_cost_cards_are_unchanged(tmp_path):
    """No fleet: the MFU and projected-finish cards keep their text."""
    r = Reducer(run="one")
    for i in range(3):
        r.feed({"event": "rl_round_trained", "learner_id": "0", "rollout_id": i, "time_unix": T0 + 60 * i,
                "trained_groups": 8, "reward_mean": 0.5})
        r.feed({"event": "rl_sampler_round", "sampler_id": "s0", "rollout_id": i, "groups": 8, "policy_version": i,
                "time_unix": T0 + 30 + 60 * i})
    ck = _page(tmp_path, r)["rCostK"]
    assert ">MFU<" in ck and "the RL tapes carry no FLOP counts" in ck and "the tapes carry no total round count" in ck


def test_island_samplers_table_dashes_counts_a_sampler_does_not_report(tmp_path):
    out = _page(tmp_path, _fleet(), pick="i0")
    sp = out["aSamplers"]
    assert "n/a" not in sp
    assert ">10 / <span title=\"the pool keeps zero-variance groups" in sp  # seen 10, dropped 0
    assert "not on this sampler&#39;s tape\">—</span>" in sp or "not on this sampler's tape\">—</span>" in sp
    assert "reports no zero-variance, truncated, dropped or reward counts" in sp


def test_fleet_table_names_the_release_cadence_from_the_trainer_tape(tmp_path):
    doc = plan_doc()
    doc["islands"][1]["release_every"] = None
    doc["islands"][1]["max_policy_age"] = None
    r = Reducer(run="fleet-qa")
    from test_dashboard_fleet_islands import fleet_records  # noqa: PLC0415

    for rec in fleet_records():
        r.feed(rec)
    assert r.load_fleet_plan(doc, "test")
    view = r.page_view(now=r.max_ts)
    assert [x["release_every"] for x in view["async"]["islands"]] == [1, None]
    r.feed({"event": "rl_policy_release_skipped", "island_id": 1, "time_unix": LAST + 10, "policy_version": 6,
            "reason": "upload_in_flight", "release_every": 3})
    view = r.page_view(now=r.max_ts)
    assert [x["release_every"] for x in view["async"]["islands"]] == [1, 3]
    tbl = _page(tmp_path, r, pick="fleet")["fleetTbl"]
    assert "n/a" not in tbl
    # max policy age: the island's trainer rounds name it (2); nothing names it: "shared"
    assert '<td class="num">3</td>' in tbl


def test_fleet_table_without_any_cadence_says_shared(tmp_path):
    doc = plan_doc()
    for isl in doc["islands"]:
        isl["release_every"] = None
    r = Reducer(run="fleet-qa")
    from test_dashboard_fleet_islands import fleet_records  # noqa: PLC0415

    for rec in fleet_records():
        r.feed(rec)
    assert r.load_fleet_plan(doc, "test")
    tbl = _page(tmp_path, r, pick="fleet")["fleetTbl"]
    assert "n/a" not in tbl and ">shared</span>" in tbl


def test_fleet_cost_rows_without_a_price_say_unpriced(tmp_path):
    out = _page(tmp_path, _fleet())  # the fixture plan has no syncer price
    cost = out["costBox"]
    syncer = cost.split(">Syncer<", 1)[1].split("</tr>", 1)[0]
    assert "unpriced" in syncer and "n/a" not in syncer


def test_event_stream_is_newest_first_by_event_time(tmp_path):
    """Tapes are read one after another: the arrival order is not time order."""
    r = fleet_reducer()
    for i, t in enumerate((LAST + 300, LAST + 100, LAST + 200)):  # read out of order
        r.feed({"event": "rl_learner_finalized", "island_id": i % 2, "time_unix": t})
    evs = _page(tmp_path, r)["evs"]
    import re  # noqa: PLC0415

    times = re.findall(r'<span class="t">([0-9:]+)</span>', evs)
    assert times == sorted(times, reverse=True) and len(times) >= 3


HINT = r"""
eval(require('fs').readFileSync(process.argv[1],'utf8'));
function box(sw,cw,sl){const c=new Set();return {scrollWidth:sw,clientWidth:cw,scrollLeft:sl,classList:{contains:(k)=>c.has(k),toggle:(k,on)=>{on?c.add(k):c.delete(k)}},c}}
const wide=box(900,300,0),end=box(900,300,600),fits=box(300,300,0);
YetoTbl.hint({querySelectorAll:()=>[wide,end,fits]});
YetoTbl.hint(null);YetoTbl.hint({});
process.stdout.write(JSON.stringify([wide.c.has("more"),end.c.has("more"),fits.c.has("more")]));
"""


def test_a_wide_table_fades_its_right_edge_until_scrolled_to_the_end():
    res = subprocess.run([_node(), "-e", HINT, str(STATIC / "tables.js")], capture_output=True, text=True, timeout=60,
                         check=False)
    assert res.returncode == 0, res.stderr
    assert json.loads(res.stdout) == [True, False, False]
    css = (STATIC / "index.html").read_text()
    assert ".tw.more{" in css and "mask-image:linear-gradient(to right" in css
