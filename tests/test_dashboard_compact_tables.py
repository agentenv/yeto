"""Long tables stay short: every table that grows with the run shows its newest rows.

The page (tables.js, YetoTbl) shows the newest 10 rows of such a table with "+50" / "all"
buttons and keeps each table's choice in localStorage; cells stay on one line. The page view
sends the newest ``VIEW_TABLE_MAX`` rows of lists drawn only as tables, with their totals."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.test_dashboard_async_grpo import sampler_records, trainer_records
from yeto.dashboard import reducer as reducer_mod
from yeto.dashboard.export import export_html
from yeto.dashboard.reducer import Reducer

T0 = 1_790_000_000.0
STATIC = Path(__file__).parent.parent / "yeto" / "dashboard" / "static"


def _node() -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    return node


COMPONENT_JS = """
const fs=require('fs');const store={};let broken=process.argv[2]==='broken';
global.localStorage={getItem(k){if(broken)throw new Error('denied');return k in store?store[k]:null},
  setItem(k,v){if(broken)throw new Error('denied');store[k]=String(v)}};
const src=fs.readFileSync(process.argv[1],'utf8')+';global.T=YetoTbl;';
eval(src);
const rows=[];for(let i=0;i<200;i++)rows.push(i);
const out={};let renders=0;T.setRender(function(){renders++});
function btn(key,def,a){return {getAttribute(n){return {'data-k':key,'data-d':String(def),'data-a':a}[n]}}}
let c=T.cap('t',rows.slice().reverse(),{def:10,unit:'rounds'});out.def=[c.rows.length,c.rows[0],c.bar];
T.click(btn('t',10,'more'));c=T.cap('t',rows.slice().reverse(),{def:10});out.more=[c.rows.length,c.rows[59]];
T.click(btn('t',10,'more'));c=T.cap('t',rows.slice().reverse(),{def:10});out.more2=c.rows.length;
T.click(btn('t',10,'all'));c=T.cap('t',rows.slice().reverse(),{def:10});out.all=[c.rows.length,c.bar];
out.stored=store['yeto-tables-v1']||null;
// a second page load reads the stored choice back
eval(src);c=T.cap('t',rows.slice().reverse(),{def:10});out.reloaded=c.rows.length;
T.setRender(function(){renders++});
T.click(btn('t',10,'less'));c=T.cap('t',rows.slice().reverse(),{def:10});out.less=c.rows.length;
// newest last (tail): the last rows are kept in their order
c=T.cap('tail',rows,{def:10,tail:true});out.tail=[c.rows[0],c.rows[9]];
// the server sent the newest 500 of 1200
c=T.cap('srv',rows.slice().reverse(),{def:10,total:1200,unit:'releases'});out.srv=c.bar;
out.small=T.cap('small',[1,2,3],{def:10,unit:'rows'}).bar;out.none=T.cap('none',[],{}).bar;
out.spark=T.spark([{label:'reward',vals:[0.1,0.2,null,0.4]},{label:'empty',vals:[null,1]}]);
out.renders=renders;
process.stdout.write(JSON.stringify(out));
"""


@pytest.mark.parametrize("storage", ["ok", "broken"])
def test_component_shows_newest_ten_and_expands(storage):
    res = subprocess.run([_node(), "-e", COMPONENT_JS, str(STATIC / "tables.js"), storage],
                         capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout)
    n, newest, bar = out["def"]
    assert n == 10 and newest == 199 and "latest 10 of 200 rounds" in bar
    assert 'data-a="more"' in bar and 'data-a="all"' in bar and 'data-a="less"' not in bar
    assert out["more"] == [60, 140] and out["more2"] == 110
    assert out["all"][0] == 200 and "200 rows" in out["all"][1] and 'data-a="less"' in out["all"][1]
    assert out["tail"] == [190, 199]
    assert "latest 10 of 1200 releases (the page holds the newest 200)" in out["srv"]
    assert out["small"] == '<div class="tcap"><span>3 rows</span></div>' and out["none"] == ""
    assert out["spark"].count('class="si"') == 1 and "reward" in out["spark"]  # one point: no line
    assert out["renders"] == 4
    if storage == "ok":  # the choice survives a reload, "latest 10" forgets it
        assert json.loads(out["stored"]) == {"t": "all"} and out["reloaded"] == 200 and out["less"] == 10
    else:  # storage throws: the page still works, nothing is remembered
        assert out["stored"] is None and out["reloaded"] == 10 and out["less"] == 10


PAGE_JS = r"""
const fs=require('fs');const html=fs.readFileSync(process.argv[1],'utf8');
const dataM=html.match(/<script type="application\/json" id="yeto-data">([\s\S]*?)<\/script>/);
const appM=html.match(/<script>\n([\s\S]*?)\n<\/script>/);
const els={};
function node(id){return {id,innerHTML:"",textContent:"",hidden:false,style:{},dataset:{},children:[],attrs:{},listeners:{},
  offsetWidth:200,offsetHeight:120,classList:{add(){},remove(){},toggle(){},contains(){return false}},
  setAttribute(k,v){this.attrs[k]=String(v)},removeAttribute(k){delete this.attrs[k]},getAttribute(k){return this.attrs[k]},
  appendChild(c){this.children.push(c);return c},addEventListener(t,fn){(this.listeners[t]=this.listeners[t]||[]).push(fn)},
  getBoundingClientRect(){return {left:0,top:0,width:560,height:400}},querySelectorAll(){return []}}}
function mk(id){if(!els[id]){els[id]=node(id);if(id==="yeto-data")els[id].textContent=dataM[1]}return els[id]}
global.document={documentElement:node("html"),getElementById:(id)=>mk(id),createElementNS:()=>node(null),
  createElement:()=>node(null),querySelectorAll:()=>[],title:""};
global.navigator={};const store={};
global.localStorage={getItem(k){return k in store?store[k]:null},setItem(k,v){store[k]=String(v)}};
global.location={hash:"",reload(){}};global.setInterval=()=>0;global.setTimeout=()=>0;
eval(appM[1]+";global.T=YetoTbl;");
function rows(id){const h=mk(id).innerHTML,m=h.match(/<table[\s\S]*?<\/table>/);return m?(m[0].match(/<tr>/g)||[]).length-1:0}
const out={def:{trainer:rows("aTrainer"),release:rows("aRelTable"),html:mk("aTrainer").innerHTML}};
function btn(key,a){return {getAttribute(n){return {"data-k":key,"data-d":"10","data-a":a}[n]}}}
T.click(btn("trainer-rounds","more"));T.click(btn("releases","all"));
out.more={trainer:rows("aTrainer"),release:rows("aRelTable")};
T.click(btn("trainer-rounds","all"));out.all={trainer:rows("aTrainer")};
process.stdout.write(JSON.stringify(out));
"""


def _long_run(rounds: int) -> Reducer:
    r = Reducer(run="long")
    for rec in trainer_records(0, rounds=rounds) + sampler_records("s0", rounds=12):
        r.feed(rec)
    return r


def test_page_renders_ten_of_200_trainer_rounds_and_expands(tmp_path):
    node = _node()
    out = export_html(_long_run(200), tmp_path / "p.html", generated_at=T0 + 20000)
    res = subprocess.run([node, "-e", PAGE_JS, str(out)], capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr
    page = json.loads(res.stdout)
    html = page["def"]["html"]
    assert page["def"]["trainer"] == 10 and "latest 10 of 200 rounds" in html
    assert page["def"]["release"] == 10
    assert page["more"] == {"trainer": 60, "release": 200} and page["all"] == {"trainer": 200}
    # newest round first; quantiles, token and age cells on one line
    first = re.search(r"<tr><td class=\"num\">(\d+)</td>", html)
    assert first and first.group(1) == "199"
    assert "0 · 0.25 · 1" in html and "<div" not in html.split("<table", 1)[1].split("</table>")[0]
    assert html.count('class="si"') >= 5  # reward, KL, logprob diff, grad_norm, TIS, tok/s, policy age


def test_page_view_caps_table_only_lists(monkeypatch):
    monkeypatch.setattr(reducer_mod, "VIEW_TABLE_MAX", 50)
    r = _long_run(120)
    for i in range(80):  # syncer merges
        r.feed({"step": i, "fragment": 0, "responders": [], "sync/responders": 1, "time_unix": T0 + i})
    v = r.page_view(table_limit=reducer_mod.VIEW_TABLE_MAX)
    a = v["async"]
    tt = v["table_totals"]
    assert len(a["releases"]) == 50 and tt["releases"] == 120 and tt["releases_trainer"] == 120
    assert a["releases"][-1]["policy_version"] == 120  # the newest kept
    assert len(v["islands"]["0"]["rounds"]) == 120  # chart source: every round
    assert len(a["trainers"]["0"]["rounds"]) == 120
    assert v["table_limit"] == 50
    full = r.page_view(table_limit=None)
    assert len(full["async"]["releases"]) == 120 and full["table_totals"]["releases"] == 120
    syn = r.rounds()
    assert len(syn) == 80 and tt["rounds_syncer"] == 80
    assert [x["round"] for x in v["rounds_syncer"]] == list(range(30, 80))
    assert len(full["rounds_syncer"]) == 80
    assert len(r.full_view()["rounds"]) == 80  # the export's own syncer rounds list is not cut


def test_utilization_rounds_and_carry_are_capped():
    util = {"rounds": [{"round": i} for i in range(30)], "carry": [{"id": i} for i in range(7)]}
    reducer_mod._cap_util(util, 5)
    assert [x["round"] for x in util["rounds"]] == list(range(25, 30)) and util["rounds_total"] == 30
    assert [x["id"] for x in util["carry"]] == [2, 3, 4, 5, 6] and util["carry_total"] == 7
    assert reducer_mod._cap_util(None, 5) is None
    keep = {"rounds": [{"round": 0}], "carry": None}
    assert reducer_mod._cap_util(keep, None) == {"rounds": [{"round": 0}], "carry": None, "rounds_total": 1}


def test_api_view_limit_parameter(tmp_path):
    import threading
    import urllib.error
    import urllib.request

    from yeto.dashboard.serve import DashboardState, make_server

    state = DashboardState(_long_run(30), [])
    srv = make_server(state, "127.0.0.1", 0)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}/api/view"

        def get(q: str) -> dict:
            with urllib.request.urlopen(base + q, timeout=30) as resp:
                return json.loads(resp.read())

        assert len(get("")["async"]["releases"]) == 30
        v = get("?limit=5")
        assert len(v["async"]["releases"]) == 5 and v["table_totals"]["releases"] == 30 and v["table_limit"] == 5
        assert get("?limit=all")["table_limit"] is None
        with pytest.raises(urllib.error.HTTPError) as err:
            get("?limit=x")
        assert err.value.code == 400
    finally:
        srv.shutdown()
        srv.server_close()


def test_fleet_island_release_tables_are_capped():
    from tests.test_dashboard_fleet_islands import fleet_reducer

    v = fleet_reducer().page_view(now=T0 + 1000, table_limit=2)
    tt = v["table_totals"]
    assert tt["island_releases"] == {"0": 5, "1": 5} and tt["releases"] == 10
    for b in v["async"]["islands"]:
        assert [row["policy_version"] for row in b["releases"]] == [4, 5]
    assert len(v["async"]["releases"]) == 2
