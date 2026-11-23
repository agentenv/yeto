"""Fleet charts of the RL page: sampler incarnations attributed by tape, marker strip,
line breaks, sampler card time, trainer policy age and the round still waiting."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from yeto.dashboard.reducer import Reducer

T0 = 1_790_000_000.0
JS = Path(__file__).parent.parent / "yeto" / "dashboard" / "static" / "sft.js"


def _sampler_tape(inc: str, t: float, rounds: int, rid0: int) -> list[dict]:
    """One incarnation's tape: engine start (no incarnation yet), join, rounds, leave."""
    recs = [{"event": "rl_sampler_start", "sampler_id": "s0", "max_backlog_groups": 16, "time_unix": t},
            {"event": "rl_sampler_join", "sampler_id": "s0", "incarnation": inc, "time_unix": t + 90}]
    for i in range(rounds):
        recs.append({"event": "rl_sampler_round", "sampler_id": "s0", "rollout_id": rid0 + i, "groups": 2,
                     "policy_version": 0, "time_unix": t + 120 + 60 * i})
    recs.append({"event": "rl_sampler_leave", "sampler_id": "s0", "incarnation": inc,
                 "time_unix": t + 120 + 60 * rounds})
    return recs


def test_interleaved_incarnation_tapes_keep_their_own_spans():
    # a run store lists tape folders by name: the newer incarnation "a..." is read before "b..."
    old, new = _sampler_tape("b-old", T0, 3, 0), _sampler_tape("a-new", T0 + 1000, 2, 3)
    r = Reducer(run="r")
    for i, rec in enumerate(new):
        r.feed(rec, source="store:s0/a-new", offset=i)
    for i, rec in enumerate(old):
        r.feed(rec, source="store:s0/b-old", offset=i)
    s = r.page_view()["async"]["samplers"]["s0"]
    spans = {x["incarnation"]: (x["first_ts"] - T0, x["last_ts"] - T0) for x in s["incarnations"]}
    # each incarnation starts at its engine start (boot before the join) and ends at its leave
    assert spans == {"b-old": (0, 300), "a-new": (1000, 1240)}
    assert s["incarnation"] == "a-new"  # the newest by start time, not the last one read
    assert {(x["incarnation"], x["rollout_id"]) for x in s["rounds"]} == {
        ("b-old", 0), ("b-old", 1), ("b-old", 2), ("a-new", 3), ("a-new", 4)}
    assert [x["incarnation"] for x in s["restarts"]] == ["a-new"]


def test_late_stop_of_an_older_incarnation_does_not_end_the_newest():
    new = _sampler_tape("a-new", T0 + 1000, 2, 3)[:-1]  # still running
    old = _sampler_tape("b-old", T0, 3, 0)
    r = Reducer(run="r")
    for i, rec in enumerate(new):
        r.feed(rec, source="store:s0/a-new", offset=i)
    for i, rec in enumerate(old):
        r.feed(rec, source="store:s0/b-old", offset=i)
    assert r.page_view()["async"]["samplers"]["s0"]["ended"] is False


def test_node_series_is_time_ordered():
    r = Reducer(run="r")
    for t in (T0 + 120, T0, T0 + 60):
        r.feed({"event": "rl_resource_sample", "sampler_id": "s0", "time_unix": t, "available": True,
                "gpus": [{"mem_used_mb": 1024.0, "mem_total_mb": 2048.0}]}, source=f"x{t}", offset=0)
    ns = r.page_view()["islands"]["s0"]["node_series"]["nvml"]
    assert [p[0] for p in ns] == [T0, T0 + 60, T0 + 120]


HELPERS_JS = """
const fs=require('fs');global.localStorage={getItem(){return null},setItem(){}};global.location={hash:''};
global.document={getElementById(){return null},body:null};
eval(fs.readFileSync(process.argv[1],'utf8')+';global.YE=YetoExt;');
const out={};
// marks 60 s apart chain into one cluster; one 10 min later stays apart; outside the range is dropped
out.mk=YE.markClusters([{t:100,color:'a'},{t:160,color:'b'},{t:700,color:'a'},{t:5000}],0,1000,1)
  .map(function(g){return [g.t,g.items.length,g.colors]});
// regular 60 s spacing: a 10 min hole breaks the line, so does a new incarnation
var P=[[0,1,'x'],[60,1,'x'],[120,1,'x'],[720,1,'x'],[780,1,'y'],[840,1,'y']];
out.seg=YE.segments(P,P).map(function(g){return g.map(function(p){return p[0]})});
out.two=YE.segments([[0,1],[900,2]],[[0,1],[900,2]]).length; // too few gaps for a median: never split
out.union=YE.spanUnion([[10,20],[15,30],[40,50],[null,5]]);
out.ov=YE.spanOverlap([[10,20],[40,50]],15,45);
process.stdout.write(JSON.stringify(out));
"""


def _node(script: str, *args: str) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    res = subprocess.run([node, "-e", script, str(JS), *args], capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def test_chart_helpers():
    out = _node(HELPERS_JS)
    assert out["mk"] == [[100, 2, ["a", "b"]], [700, 1, ["a"]]]
    assert out["seg"] == [[0, 60, 120], [720], [780, 840]]
    assert out["two"] == 1
    assert out["union"] == [[10, 30], [40, 50]]
    assert out["ov"] == 10


SMOKE_JS = Path(__file__).parent / "js" / "dashboard_smoke.js"


def _rl_view_page(tmp_path, recs, now):
    from yeto.dashboard.export import render_page

    r = Reducer(run="r")
    for i, (src, rec) in enumerate(recs):
        r.feed(rec, source=src, offset=i)
    view = r.full_view(now=now)
    out = tmp_path / "page.html"
    out.write_text(render_page(view))
    return out


def test_rl_page_marks_waiting_round_and_card_time(tmp_path):
    """The rendered page: restart marks sit in the strip (no full-height dashed line), the sampler
    reward is a rolling mean named as such, sampler card time covers both incarnations, the trainer
    policy age comes from policy_age_mean, and the round still waiting is shown."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    recs = [("store:s0/b-old", x) for x in _sampler_tape("b-old", T0, 3, 0)]
    recs += [("store:s0/a-new", x) for x in _sampler_tape("a-new", T0 + 1000, 2, 3)[:-1]]
    for rec in recs:
        if rec[1]["event"] == "rl_sampler_round":
            rec[1]["reward_mean"] = 0.5
    trainer = [{"event": "rl_round_trained", "learner_id": "0", "rollout_id": 0, "time_unix": T0 + 400,
                "trained_groups": 8, "policy_age_mean": 2.0, "policy_age_max": 2, "reward_mean": 0.25},
               {"event": "rl_round_phases", "learner_id": "0", "rollout_id": 0, "time_unix": T0 + 410,
                "wall_seconds": 300.0, "phases": {"generate": 280.0, "train": 20.0}}]
    page = _rl_view_page(tmp_path, recs + [("t", x) for x in trainer], T0 + 1300)
    res = subprocess.run([node, "-e", RL_SMOKE, str(page)], capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout)
    assert 'class="mks"' in out["rew"] and 'stroke-dasharray="2 3"' not in out["rew"]
    assert "mean of last 8 groups" in out["rew"]
    assert "trainer 0 (no data)" not in out["age"]
    assert "union of their incarnations" in out["cost"]
    # Async GRPO trainer table: a column no round reports (PPO KL here) is hidden and named
    hidden = out["atr"].split("hidden: ", 1)[1]
    assert "PPO KL" in hidden and ">PPO KL</th>" not in out["atr"] and ">policy ages</th>" in out["atr"]
    assert "samplers do not train" in out["nodes"]


RL_SMOKE = r"""
const fs=require('fs');const html=fs.readFileSync(process.argv[1],'utf8');
const dataM=html.match(/<script type="application\/json" id="yeto-data">([\s\S]*?)<\/script>/);
const appM=html.match(/<script>\n([\s\S]*?)\n<\/script>/);
const els={};
function node(id){return {id,innerHTML:"",textContent:"",hidden:false,style:{},dataset:{},children:[],attrs:{},
  classList:{add(){},remove(){},toggle(){},contains(){return false}},setAttribute(k,v){this.attrs[k]=String(v)},
  removeAttribute(){},appendChild(c){this.children.push(c);return c},addEventListener(){},clientWidth:1200,
  getBoundingClientRect(){return {left:0,top:0,width:1200,height:400}},querySelectorAll(){return []},querySelector(){return null}}}
function mk(id){if(!els[id]){els[id]=node(id);if(id==="yeto-data"&&dataM)els[id].textContent=dataM[1]}return els[id]}
global.document={documentElement:node("html"),getElementById:(id)=>mk(id),createElementNS:()=>node(null),
  createElement:()=>node(null),querySelectorAll:()=>[],title:""};
global.navigator={};global.localStorage={getItem(){return null},setItem(){}};global.setInterval=()=>0;global.setTimeout=()=>0;
global.location={hash:"",search:""};
eval(appM[1]);
process.stdout.write(JSON.stringify({rew:mk("rfRew").innerHTML,age:mk("rfAge").innerHTML,cost:mk("rCostK").innerHTML,
  nodes:mk("rNodes").innerHTML,atr:mk("aTrainer").innerHTML,asp:mk("aSamplers").innerHTML,wall:mk("wall").innerHTML}));
"""


def _trainer_round(rid: int, t: float, *, ages: dict, zvr: float, sampler_inc: str) -> list[tuple[str, dict]]:
    src = "store:trainer/l0-modal-20261123T0000Z"
    recs = [{"event": "rl_round_trained", "island_id": 0, "rollout_id": rid, "time_unix": t, "trained_groups": 8,
             "trained_samples": 16, "reward_mean": 0.5, "policy_age_groups": ages, "policy_age_mean": 1.0},
            {"event": "rl_local_round", "island_id": 0, "local_round_id": rid + 1, "time_unix": t + 1,
             "zero_variance_group_ratio": zvr}]
    for g in range(2):  # two of the round's groups came from the sampler, generated with v(rid-1)
        recs.append({"event": "rl_trajectory_reward", "island_id": 0, "rollout_id": rid, "time_unix": t,
                     "trajectory_id": f"{sampler_inc}-r0-row{g}-0", "group_index": g, "reward": float(g),
                     "policy_version": rid, "policy_versions": [[rid - 1, 0, 100]]})
    return [(src, x) for x in recs]


def test_trainer_round_ages_zero_variance_and_sampler_share():
    recs = [("store:s0/a-new", x) for x in _sampler_tape("a-new", T0, 3, 0)]
    recs += _trainer_round(4, T0 + 400, ages={"1": 6, "2": 2}, zvr=0.625, sampler_inc="a-new")
    r = Reducer(run="r")
    for i, (src, rec) in enumerate(recs):
        r.feed(rec, source=src, offset=i)
    v = r.page_view()
    t = v["async"]["trainers"]["0"]
    assert t["age_hist"] == {"1": 6, "2": 2}  # rl_round_trained.policy_age_groups, not the empty read histogram
    (row,) = [x for x in v["islands"]["0"]["rounds"] if x["round"] == 4]
    assert row["zero_var_ratio"] == 0.625
    assert row["by_sampler"] == {"s0": {"groups": 2, "samples": 2, "versions": [3], "age_mean": 1.0, "reward_mean": 0.5}}
    # samplers keep their own round numbers: the synchronous "missed rounds" alert never names them
    assert not [a for a in r.overview(now=T0 + 500)["alerts"] if "missed" in a["rule"]]


def test_backlog_cap_is_the_newest_starts():
    new = _sampler_tape("a-new", T0 + 1000, 1, 3)
    new[0]["max_backlog_groups"] = 64
    r = Reducer(run="r")
    for i, rec in enumerate(new):
        r.feed(rec, source="store:s0/a-new", offset=i)
    for i, rec in enumerate(_sampler_tape("b-old", T0, 1, 0)):  # cap 16, read later
        r.feed(rec, source="store:s0/b-old", offset=i)
    assert r.page_view()["async"]["samplers"]["s0"]["max_backlog_groups"] == 64


def test_many_samplers_draw_one_aggregate_with_a_toggle(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    recs = []
    for n in range(4):
        tape = _sampler_tape(f"i{n}", T0 + 10 * n, 4, 0)
        for x in tape:
            x["sampler_id"] = f"s{n}"
            if x["event"] == "rl_sampler_round":
                x["reward_mean"], x["generate_s"] = 0.25 * n, 30.0 + n
        recs += [(f"store:s{n}/i{n}", x) for x in tape]
    page = _rl_view_page(tmp_path, recs, T0 + 600)
    res = subprocess.run([node, "-e", RL_SMOKE, str(page)], capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout)
    assert "all 4 samplers (mean of last 32 groups)" in out["rew"] and "show each sampler (4)" in out["rew"]
    assert "sampler s1 (mean of last 8 groups)" not in out["rew"]  # each sampler only behind the toggle


def test_hot_reload_names_the_drain_it_waited_for():
    recs = [{"event": "rl_policy_released", "island_id": 0, "policy_version": 3, "time_unix": T0, "released_at": T0},
            {"event": "rl_agentic_drain", "sampler_id": "s0", "policy_version": 2, "time_unix": T0 - 300, "waited_s": 9.0},
            {"event": "rl_agentic_drain", "sampler_id": "s0", "policy_version": 2, "time_unix": T0 + 60, "waited_s": 44.5,
             "in_flight": 12, "abandoned_trials": 0, "detached_trials": 1},
            {"event": "rl_sampler_policy_loaded", "sampler_id": "s0", "policy_version": 3, "time_unix": T0 + 140,
             "released_at": T0, "seen_at": T0 + 4, "pause_s": 1.1,
             "follower_timings": {"resolve_s": 37.9, "load_s": 31.8}}]
    r = Reducer(run="r")
    for i, rec in enumerate(recs):
        r.feed(rec, source="t", offset=i)
    (rel,) = [x for x in r.page_view()["async"]["releases"] if x["policy_version"] == 3]
    ld = rel["loads"]["s0"]
    assert (ld["latency_s"], ld["drain_s"], ld["drain_in_flight"], ld["drain_cut"], ld["pause_s"]) == (140.0, 44.5, 12, 1, 1.1)
