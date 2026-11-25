"""A fleet of async islands on the RL page shows no bare "n/a": the buffer backlog of every island lane,
zero-variance groups a pool keeps, the finish from the syncer's total steps, no MFU card (no FLOP counts on
the tapes), a dash for a count one sampler does not report, the release cadence a trainer's tape names; the
event stream in time order; a wide table marks that it scrolls."""

from __future__ import annotations

import json
import re
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
const out={};["rKpis","rCostK","rFleet","aSamplers","aTrainer","fleetTbl","costBox","evs","rNodes"].forEach(i=>{out[i]=mk(i).innerHTML});
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
    assert "—: not on the tapes" in kp  # step times are not on the fixture's tapes: a dash, and why


def test_fleet_kpis_show_kept_zero_variance_beside_a_pool_that_drops(tmp_path):
    r = _fleet()
    for i in range(3):
        r.feed({"event": "rl_sampler_engine_round", "sampler_id": "qa-a-i1-0", "rollout_id": i, "samples": 8,
                "zero_variance_seen": 2, "zero_variance_groups": 2, "finish": {"length": 0},
                "time_unix": T0 + 72 + 60 * i})
    kp = _page(tmp_path, r)["rKpis"]
    assert "n/a" not in kp and " / 6 / " in kp  # 6 dropped by the island 1 pool
    assert "zero-var kept (not dropped) by sampler qa-v-i0-0: 10 of" in kp


def test_fleet_backlog_wording_without_a_read(tmp_path):
    """No trainer read yet: the card and the Fleet table say the same thing."""
    r = Reducer(run="fleet-qa")
    from test_dashboard_fleet_islands import fleet_records  # noqa: PLC0415

    for rec in fleet_records():
        if rec.get("event") not in ("rl_policy_released", "rl_sampler_policy_loaded"):
            r.feed(rec)
    assert r.load_fleet_plan(plan_doc(), "test")
    out = _page(tmp_path, r)
    assert ">not read yet<" in out["rKpis"] and "written, not read yet" in out["rKpis"]
    tbl = _page(tmp_path, r, pick="fleet")["fleetTbl"]
    assert "written, not read yet</span>" in tbl and "n/a" not in tbl


def test_fleet_cost_names_why_mfu_is_missing_and_paces_the_finish_by_the_syncer(tmp_path):
    out = _page(tmp_path, _fleet(total=12))
    ck = out["rCostK"]
    assert "n/a" not in ck
    assert ">MFU<" in ck and "not on the tapes (no FLOP counts)" in ck
    assert ">Projected finish<" in ck and "syncer step 10 of 12 at its mean merge pace" in ck


def test_a_resumed_syncer_paces_from_the_step_it_resumed_at(tmp_path):
    """syncer_config of a resume: global_step 6; the pace counts the 4 steps since, not all 10."""
    r = fleet_reducer()
    r.feed({**_syncer_config(12), "resumed": True, "global_step": 6, "time_unix": T0 + 400})
    lm = r.page_view(now=r.max_ts)["topology"]["diloco"]["last_merge_ts_any"]
    ck = _page(tmp_path, r)["rCostK"]
    assert "since its resume at step 6" in ck
    pace = (lm - (T0 + 400)) / 4
    assert f"mean merge pace {pace:.1f} s since its resume" in ck  # not (lm - t0) / 10
    r2 = fleet_reducer()
    r2.feed({**_syncer_config(12), "resumed": True, "time_unix": T0 + 400})  # no global_step: no pace
    ck2 = _page(tmp_path, r2)["rCostK"]
    assert "resumed at a step not on its tape" in ck2 and "n/a" not in ck2
    r3 = fleet_reducer()  # a sharded syncer resumes each shard at its own step: no pace
    r3.feed({**_syncer_config(12), "resumed": True, "global_step": 6, "shard_count": 2, "time_unix": T0 + 400})
    assert "a sharded syncer resumed per shard, so no pace" in _page(tmp_path, r3)["rCostK"]


def test_a_finish_past_its_pace_says_overdue(tmp_path):
    r = _fleet(total=12)
    r.feed({"event": "rl_heartbeat", "island_id": 0, "time_unix": LAST + 5000})  # the page's now, long after
    ck = _page(tmp_path, r)["rCostK"]
    assert "overdue by" in ck and "last merge" in ck and "in 0 s" not in ck


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


def test_single_trainer_kpis_and_sampler_table_keep_their_rendering(tmp_path):
    """No fleet: the KPI cards and the Samplers table keep "n/a" where a count is missing; none of the
    fleet wording (island backlogs, kept zero-variance groups, dashes) leaks in."""
    r = Reducer(run="one")
    for i in range(3):
        r.feed({"event": "rl_round_trained", "learner_id": "0", "rollout_id": i, "time_unix": T0 + 60 * i,
                "trained_groups": 8, "reward_mean": 0.5})
        for sid in ("s0", "s1"):
            r.feed({"event": "rl_sampler_round", "sampler_id": sid, "rollout_id": i, "groups": 8,
                    "policy_version": i, "time_unix": T0 + 30 + 60 * i})
        r.feed({"event": "rl_sampler_engine_round", "sampler_id": "s0", "rollout_id": i, "samples": 8,
                "zero_variance_seen": 1, "finish": {"length": 2}, "reward_mean": 0.5, "time_unix": T0 + 31 + 60 * i})
    out = _page(tmp_path, r)
    kp, sp = out["rKpis"], out["aSamplers"]
    assert '>n/a / n/a / 6 / n/a<' in kp and "kept by the pools" not in kp and "—: not on the tapes" not in kp
    assert ">Buffer backlog<" in kp and "written 48 · read n/a" in kp and "island 0:" not in kp
    assert ">3 / n/a<" in sp and "the pool keeps zero-variance groups" not in sp and "—" not in sp
    assert "zero-variance then comes from the trainer" in sp


def test_fleet_islands_now_table_has_no_bare_na(tmp_path):
    """The fixture trainers carry no train-rollout KL, tokens/s, step or round time: each is a dash that
    says so (tags and attributes untouched), and no "round" line without a round time."""
    nodes = _page(tmp_path, _fleet())["rNodes"]
    text = re.sub(r"<[^>]*>", " ", nodes)
    assert ">Island<" in nodes and "n/a" not in text and "round —" not in text
    assert "not on this island&#39;s tapes" in nodes or "not on this island's tapes" in nodes
    assert all("n/a" not in tag or 'title="' in tag for tag in re.findall(r"<[^>]*>", nodes))


def test_single_trainer_islands_now_table_keeps_its_rendering(tmp_path):
    """No fleet: a missing trainer value stays "n/a" (and the round line stays) as before."""
    r = Reducer(run="one")
    for i in range(3):
        r.feed({"event": "rl_round_trained", "learner_id": "0", "rollout_id": i, "time_unix": T0 + 60 * i,
                "trained_groups": 8, "reward_mean": 0.5})
    nodes = _page(tmp_path, r)["rNodes"]
    assert ">n/a<" in nodes and "round n/a" in nodes and "not on this island" not in nodes


def test_island_samplers_table_dashes_counts_a_sampler_does_not_report(tmp_path):
    out = _page(tmp_path, _fleet(), pick="i0")
    sp = out["aSamplers"]
    assert "n/a" not in sp
    assert ">10 / <span title=\"the pool keeps zero-variance groups" in sp  # seen 10, dropped 0
    assert "not on this sampler&#39;s tape\">—</span>" in sp or "not on this sampler's tape\">—</span>" in sp
    assert "reports no zero-variance, truncated, dropped or reward counts" in sp


def test_a_dash_never_lands_inside_a_cell_attribute(tmp_path):
    """A sampler without version lags: its age cell's title holds "n/a" text; only the cell text becomes a
    dash, the title attribute stays whole."""
    r = Reducer(run="fleet-qa")
    from test_dashboard_fleet_islands import fleet_records  # noqa: PLC0415

    for rec in fleet_records():
        if rec.get("event") == "rl_sampler_round" and rec.get("sampler_id") == "qa-s-i0-0":
            rec = {k: v for k, v in rec.items() if k != "version_lag"}
        r.feed(rec)
    assert r.load_fleet_plan(plan_doc(), "test")
    sp = _page(tmp_path, r, pick="i0")["aSamplers"]
    assert '—">' not in sp and not re.search(r'title="[^"]*<', sp)
    assert re.search(r'<span title="rounds by policy age: n/a">', sp)  # the attribute is intact
    assert '<span class="note" title="not on this sampler&#39;s tape">—</span> / ' in sp or \
        '<span class="note" title="not on this sampler\'s tape">—</span> / ' in sp


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


def test_fleet_table_without_a_plan_does_not_claim_the_shared_value(tmp_path):
    r = fleet_reducer(plan=False)
    view = r.page_view(now=r.max_ts)
    assert view["async"]["islands"] and all(x["release_every"] is None for x in view["async"]["islands"])
    tbl = _page(tmp_path, r, pick="fleet")["fleetTbl"]
    assert "n/a" not in tbl and "shared" not in tbl and "no fleet plan and no tape names it" in tbl


def test_release_cadence_takes_the_newest_record_whatever_the_read_order():
    r = fleet_reducer()
    for t, every in ((LAST + 50, 4), (LAST + 10, 3)):  # the newer record is read first
        r.feed({"event": "rl_policy_release_skipped", "island_id": 1, "time_unix": t, "policy_version": 6,
                "reason": "upload_in_flight", "release_every": every})
    assert r.async_.trainer["1"]["release_every"] == 4


def test_fleet_cost_rows_without_a_price_say_unpriced(tmp_path):
    out = _page(tmp_path, _fleet())  # the fixture plan has no syncer price
    cost = out["costBox"]
    syncer = cost.split(">Syncer<", 1)[1].split("</tr>", 1)[0]
    assert "unpriced" in syncer and "n/a" not in syncer


def _stream(tmp_path: Path, times: list) -> list[int]:
    """The event stream of a single-trainer run whose events arrive with ``times`` (None: no time),
    as the arrival indexes in display order."""
    r = Reducer(run="one")
    for i, t in enumerate(times):
        rec = {"event": "rl_learner_finalized", "learner_id": "0", "rollout_id": i}
        if t is not None:
            rec["time_unix"] = T0 + t
        r.feed(rec)
    return [int(x) for x in re.findall(r"rollout_id=(\d+)", _page(tmp_path, r)["evs"])]


def test_event_stream_is_newest_first_by_event_time(tmp_path):
    """Tapes are read one after another: the arrival order is not time order."""
    assert _stream(tmp_path, [300, 100, 200]) == [0, 2, 1]


def test_event_stream_order_is_total_with_events_without_a_time(tmp_path):
    """An event without a time sorts at the time of the last timed event before it (its arrival position):
    a total order, so timed events stay in time order around it whatever their arrival order (a comparator
    that skipped untimed pairs put old events first)."""
    times = [None, 300, None, 100, 200, None] + [400 + i for i in range(40)] + [None] + [10 + i for i in range(40)]
    got = _stream(tmp_path, times)
    timed = [times[i] for i in got if times[i] is not None]
    assert timed == sorted(timed, reverse=True) and timed[0] == 439 and len(got) == 15
    shuffled = [200, None, 100, 300]
    order = _stream(tmp_path, shuffled)
    assert [shuffled[i] for i in order if shuffled[i] is not None] == [300, 200, 100]
    assert order.index(1) == order.index(0) - 1  # the untimed event sits just above the event before it


HINT = r"""
eval(require('fs').readFileSync(process.argv[1],'utf8'));
function box(sw,cw,sl){const c=new Set(),st={};return {scrollWidth:sw,clientWidth:cw,scrollLeft:sl,offsetHeight:215,clientHeight:200,
  style:{setProperty(k,v){st[k]=v}},classList:{contains:(k)=>c.has(k),toggle:(k,on)=>{on?c.add(k):c.delete(k)}},c,st}}
const wide=box(900,300,0),end=box(900,300,600),fits=box(300,300,0);
YetoTbl.hint({querySelectorAll:()=>[wide,end,fits]});
YetoTbl.hint(null);YetoTbl.hint({});
process.stdout.write(JSON.stringify([wide.c.has("more"),end.c.has("more"),fits.c.has("more"),wide.st["--sb"]||null,fits.st["--sb"]||null]));
"""


def test_a_wide_table_fades_its_right_edge_until_scrolled_to_the_end():
    res = subprocess.run([_node(), "-e", HINT, str(STATIC / "tables.js")], capture_output=True, text=True, timeout=60,
                         check=False)
    assert res.returncode == 0, res.stderr
    # the scrollbar's own height (offsetHeight - clientHeight) stays unmasked; a box that fits is untouched
    assert json.loads(res.stdout) == [True, False, False, "15px", None]
    css = (STATIC / "index.html").read_text()
    assert ".tw.more{" in css and "mask-image:linear-gradient(to right" in css and "var(--sb,0px)" in css
