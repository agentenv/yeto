"""A fleet of async islands as a real prerun wrote it (two Modal islands, island 0 one vLLM L40S
pool, island 1 one SGLang L40S pool whose first shard was fenced and replaced): the fields of the
real records, laid out like the run store (trainer tapes under ``tapes/trainer/l<k>-modal-*``, pool
tapes under ``trainer<k>/tapes/<shard>/<incarnation>/``). Covers the island binding without a
plan while ``rl_sampler_start.store`` is the sampler's local staging path, the fleet spend
(pools + each island's trainer + the syncer), the per-pool mismatch of a single-pool island, a
ratio breaker gauge, the island id of the reader's echoed events and the Breakers panel's place."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from yeto.dashboard import fleet_islands
from yeto.dashboard.cost import load_prices
from yeto.dashboard.export import export_html, render_page
from yeto.dashboard.reducer import Reducer
from yeto.dashboard.store_tapes import StoreTapes
from yeto.rl.engine.object_store import DirectoryObjectStore

T0 = 1_795_515_600.0
H100_MB = 81559.0  # NVML mem_total_mb of an H100 80GB (the trainer's GPU model comes from it)
# shard -> (island, engine, engine version, incarnation, first event offset s, rounds)
SHARDS = {"fpv-i0-0": (0, "vllm", "vllm-kimi-k31@0.1", "9462286c", 23, 8),
          "fps-i1-0": (1, "sglang", "sglang-main@842b", "9d8a2d47", 36, 3),
          "fps-i1-1": (1, "sglang", "sglang-main@842b", "47516430", 900, 3)}
ROUNDS = 6  # trained rounds per trainer


def _trainer(k: int) -> list[dict]:
    t = T0 + 380 + 20 * k
    recs = [{"event": "rl_driver_start", "island_id": k, "time_unix": t, "placement": "partitioned-serial"},
            {"event": "rl_engine_selected", "island_id": k, "time_unix": t + 1, "rl/outer_sync": True},
            {"event": "rl_resource_sample", "island_id": k, "time_unix": t + 2, "source": "nvml", "phase": "startup",
             "gpus": [{"index": 0, "mem_total_mb": H100_MB, "mem_used_mb": 491.0, "util_pct": 0}]}]
    for r in range(ROUNDS):
        tr = t + 120 + 60 * r
        recs.append({"event": "rl_round_trained", "island_id": k, "time_unix": tr, "rollout_id": r, "train_step": r + 1,
                     "trained_groups": 4, "trained_samples": 16, "policy_age_groups": {"2": 4}, "policy_age_mean": 2.0,
                     "policy_age_max": 2, "reward_mean": 0.5, "grad_norm": 3.0, "tok_per_s": 9000.0,
                     "mismatch": {"tis_clipfrac": 0.002 * (k + 1)},
                     "train_metrics": {"train_rollout_logprob_abs_diff": 0.02 + 0.01 * k, "tis_clipfrac": 0.002 * (k + 1)}})
        # the trainer's own trajectory records: ids empty, as a Miles trainer reading external pools writes them
        recs += [{"event": "rl_trajectory_reward", "island_id": k, "time_unix": tr - 1, "rollout_id": r,
                  "trajectory_id": "", "task_id": "", "group_index": g, "sample_index": 0, "reward": 1.0,
                  "policy_version": r, "policy_versions": [[max(0, r - 2), 0, 60]]} for g in range(4)]
        recs.append({"event": "rl_policy_released", "island_id": k, "time_unix": tr + 15, "policy_version": r,
                     "local_version": r, "released_at": tr + 15, "upload_s": 13.0, "release_bytes": 1519187274})
    recs.append({"event": "rl_learner_finalized", "island_id": k, "time_unix": t + 120 + 60 * ROUNDS})
    return recs


def _sampler(sid: str) -> list[dict]:
    k, engine, version, inc, off, rounds = SHARDS[sid]
    t = T0 + off
    # store: the sampler's local staging copy of trainer<k>/ (not the run store's URI)
    recs = [{"event": "rl_sampler_start", "sampler_id": sid, "time_unix": t, "island": k,
             "store": f"/tmp/stage/b/runs/r/trainer{k}", "max_backlog_groups": 12,
             "provenance": {"card_model": "L40S", "device_family": "cuda", "engine": engine, "engine_version": version,
                            "precision": "bf16", "quant_scheme": "bf16", "sampler_id": sid, "topology": "tp1"}},
            {"event": "rl_sampler_join", "sampler_id": sid, "sampler": sid, "time_unix": t + 120, "incarnation": inc},
            {"event": "rl_breaker_status", "sampler_id": sid, "time_unix": t + 121, "layer": "sampler",
             # written before status.primary_limit read the gauge's unit: the ratio gauge carries realert_s
             "items": [{"name": "sampler.vram_guard", "layer": "sampler", "kind": "alert", "enabled": True,
                        "action": "warn", "gauge": "ratio", "unit": "ratio", "value": None, "limit": 600.0,
                        "ratio": None, "params": {"high_frac_ratio": 0.97, "interval_s": None, "realert_s": 600.0},
                        "source": {"kind": "default", "location": "default"}}]}]
    for r in range(rounds):
        recs.append({"event": "rl_sampler_round", "sampler_id": sid, "time_unix": t + 150 + 30 * r, "rollout_id": r,
                     "groups": 4, "policy_version": r, "version_lag": 1, "generate_s": 24.0, "buffer_end": 4 * (r + 1)})
    recs.append({"event": "rl_sampler_leave", "sampler_id": sid, "sampler": sid, "incarnation": inc,
                 "time_unix": t + 150 + 30 * rounds, "reason": "FINISHED", "rounds": rounds})
    return recs


def write_store(root: Path, plan: dict | None = None) -> Path:
    for k in (0, 1):
        d = root / "tapes" / "trainer" / f"l{k}-modal-20261124T10250{8 + k}Z"
        d.mkdir(parents=True, exist_ok=True)
        (d / "00000000.jsonl").write_text("".join(json.dumps(x) + "\n" for x in _trainer(k)))
    for sid, (k, _e, _v, inc, *_rest) in SHARDS.items():
        d = root / f"trainer{k}" / "tapes" / sid / inc
        d.mkdir(parents=True, exist_ok=True)
        (d / "00000000.jsonl").write_text("".join(json.dumps(x) + "\n" for x in _sampler(sid)))
    if plan is not None:
        (root / "fleet").mkdir(exist_ok=True)
        (root / "fleet" / "plan.json").write_text(json.dumps(plan))
    return root


def plan_doc(*, prices: bool) -> dict:
    """The plan ``run up`` writes for this fleet; ``prices``: with the pool and syncer prices
    (a plan written before them has neither)."""
    islands = []
    for k, (name, engine, base) in enumerate((("i0-vllm", "vllm", "fpv-i0"), ("i1-sglang", "sglang", "fps-i1"))):
        pool = {"name": f"{name}-l40s", "source_name": f"{engine}-l40s", "source": "external", "provider": "modal",
                "shard_base": base, "shard_limit": 2, "run_prefix": f"r/trainer{k}", "target": 1, "engine": engine,
                "rollout_quant": None}
        if prices:
            pool["usd_per_h"] = 2.4
        islands.append({"index": k, "name": name, "learner": k, "substore": f"trainer{k}", "store": f"s3://b/runs/r/trainer{k}",
                        "gpu": "modal:1x1xh100@us", "usd_per_h": 5.06, "release_every": 1 + k, "max_policy_age": None,
                        "buffer_shards": [f"{base}-0", f"{base}-1"], "pools": [pool]})
    doc = {"version": 1, "run": "r", "store": "s3://b/runs/r", "dispense": "fleet", "islands": islands}
    if prices:
        doc["syncer"] = {"usd_per_h": 0.26}
    return doc


def reduce(root: Path) -> Reducer:
    r = Reducer(run="r", prices=load_prices(None), budget_usd=12.0)
    objs = DirectoryObjectStore(str(root))
    StoreTapes(objs, label="store").pump(r)
    pf = fleet_islands.PlanFile([], objs)
    pf.pump(r)
    for k in (0, 1):  # without a plan the lanes are read the same way (the coordinator lists them)
        StoreTapes(objs, label=f"store/trainer{k}", prefix=f"trainer{k}/tapes/").pump(r)
    return r


@pytest.fixture
def no_plan(tmp_path):
    return reduce(write_store(tmp_path / "store"))


# -- binding -------------------------------------------------------------------------------
def test_without_a_plan_and_with_a_local_store_path_samplers_bind_by_their_island(no_plan):
    r = no_plan
    assert r.fleet_plan is None
    fleet = fleet_islands.resolve(r)
    assert fleet["source"] == "events" and fleet["unbound"] == []
    assert fleet["island_of"] == {"fpv-i0-0": 0, "fps-i1-0": 1, "fps-i1-1": 1, "0": 0, "1": 1}
    assert {p["name"]: p["samplers"] for isl in fleet["islands"] for p in isl["pools"]} == {
        "fpv-i0": ["fpv-i0-0"], "fps-i1": ["fps-i1-0", "fps-i1-1"]}  # one pool per shard base
    page = r.page_view(live=False)
    topo = page["topology"]["diloco"]["fleet"]["islands"]
    assert [(x["island"], x["index"], x["trainer"]) for x in topo] == [(0, 0, "0"), (1, 1, "1")]
    assert [(p["engine"], p["card"]) for x in topo for p in x["pools"]] == [("vLLM", "L40S"), ("SGLang", "L40S")]
    assert [(b["island"], b["index"]) for b in page["async"]["islands"]] == [(0, 0), (1, 1)]


def test_a_store_path_alone_still_binds_by_its_trainer_lane():
    assert fleet_islands.sampler_learner("x-0", {"store": "/tmp/stage/b/runs/r/trainer1"}, None) == 1
    assert fleet_islands.sampler_learner("x-0", {"store": "s3://b/runs/r/trainer0/", "island": 1}, None) == 1
    assert fleet_islands.sampler_learner("x-0", {"store": "/tmp/stage/b/runs/r"}, None) is None


# -- spend -----------------------------------------------------------------------------------
def _rows(r):
    cost = r.cost(r.max_ts)
    return cost, {row["id"]: row for row in cost["islands"]}


def test_fleet_spend_without_a_plan_counts_pools_trainers_and_an_unpriced_syncer(no_plan):
    cost, rows = _rows(no_plan)
    assert [rows[i]["kind"] for i in ("0", "1", "fpv-i0-0", "fps-i1-0", "fps-i1-1", "syncer")] == [
        "trainer", "trainer", "pool sampler", "pool sampler", "pool sampler", "syncer"]
    l40s = load_prices(None)["prices"]["modal:L40S"]
    for sid, (_k, _e, _v, _i, _off, rounds) in SHARDS.items():
        row = rows[sid]
        # the pool's provider is not on the tapes without a plan: the trainers' cloud (modal)
        assert row["rate_usd_h"] == l40s and "the trainers' cloud" in row["rate_source"]
        assert row["hours"] == pytest.approx((150 + 30 * rounds) / 3600)  # its first to last event
    assert rows["syncer"]["priced"] is False and rows["syncer"]["cost_usd"] is None
    f = cost["fleet"]
    assert f["pools_usd"] == pytest.approx(sum(rows[s]["cost_usd"] for s in SHARDS))
    assert f["trainers_usd"] == pytest.approx(rows["0"]["cost_usd"] + rows["1"]["cost_usd"])
    assert cost["total_usd"] == pytest.approx(f["pools_usd"] + f["trainers_usd"])
    assert "syncer" in cost["unpriced"]


def test_fleet_spend_with_a_plan_bills_its_prices(tmp_path):
    cost, rows = _rows(reduce(write_store(tmp_path / "s", plan_doc(prices=True))))
    assert rows["0"]["rate_usd_h"] == rows["1"]["rate_usd_h"] == 5.06  # the run config's, not the price table's
    assert all(rows[s]["rate_usd_h"] == 2.4 and rows[s]["pool"] in ("i0-vllm-l40s", "i1-sglang-l40s") for s in SHARDS)
    syn = rows["syncer"]
    assert syn["rate_usd_h"] == 0.26 and syn["hours"] == pytest.approx(
        (max(_trainer(1)[-1]["time_unix"], _trainer(0)[-1]["time_unix"]) - T0 - 380) / 3600)
    f = cost["fleet"]
    assert cost["total_usd"] == pytest.approx(f["pools_usd"] + f["trainers_usd"] + f["syncer_usd"])
    assert f["syncer_usd"] == pytest.approx(0.26 * syn["hours"]) and cost["unpriced"] == []


def test_a_plan_written_before_the_prices_keeps_the_syncer_unpriced(tmp_path):
    cost, rows = _rows(reduce(write_store(tmp_path / "s", plan_doc(prices=False))))
    assert rows["0"]["rate_usd_h"] == 5.06 and rows["fpv-i0-0"]["rate_source"].startswith("price table modal:L40S")
    assert rows["syncer"]["cost_usd"] is None and "written before it" in rows["syncer"]["rate_source"]
    unset = plan_doc(prices=False) | {"syncer": {"usd_per_h": None}}  # unset, and no placement on the plan
    _cost, rows = _rows(reduce(write_store(tmp_path / "u", unset)))
    assert rows["syncer"]["cost_usd"] is None and "does not say where the syncer runs" in rows["syncer"]["rate_source"]


@pytest.mark.parametrize("placement, rate, says", [
    # --controller head: the controller head VM it shares with the fleet controller, billed by its row
    ({"where": "head", "provider": "aws"}, 0.384, "the controller head VM on aws (it hosts the syncer and the fleet "
                                                  "controller): price table syncer@aws 0.384 USD/h"),
    ({"where": "vm", "provider": "aws"}, 0.384, "its own syncer VM on aws: price table syncer@aws 0.384 USD/h"),
    ({"where": "vm", "provider": "modal"}, 2.4, "its own syncer VM on modal: price table syncer@modal 2.4 USD/h"),
])
def test_an_unset_syncer_price_follows_where_the_plan_says_the_syncer_runs(tmp_path, placement, rate, says):
    doc = plan_doc(prices=False) | {"syncer": {"usd_per_h": None, "placement": placement}}
    cost, rows = _rows(reduce(write_store(tmp_path / "s", doc)))
    syn = rows["syncer"]
    assert syn["rate_usd_h"] == rate and syn["cost_usd"] == pytest.approx(rate * syn["hours"])
    assert syn["rate_source"].startswith("syncer.usd_per_h unset: " + says) and "syncer" not in cost["unpriced"]
    # syncer.usd_per_h set wins wherever it runs
    doc["syncer"]["usd_per_h"] = 0.26
    _cost, rows = _rows(reduce(write_store(tmp_path / "p", doc)))
    assert rows["syncer"]["rate_usd_h"] == 0.26 and rows["syncer"]["rate_source"] == "fleet plan syncer.usd_per_h"


def test_compile_writes_the_pool_and_syncer_prices_into_the_plan():
    from yeto.rl.fleet import FleetPlan, compile_fleet

    cfg = json.loads((Path(__file__).parents[1] / "examples" / "fleet" / "run.two-islands.json").read_text())
    cfg["syncer"] = {"usd_per_h": 0.26}
    _out, plan = compile_fleet(cfg)
    doc = plan.to_json()
    # the example launches with the default --controller head: the syncer runs on the controller head VM (AWS)
    assert doc["syncer"] == {"usd_per_h": 0.26, "placement": {"where": "head", "provider": "aws"}}
    assert [[p["usd_per_h"] for p in i["pools"]] for i in doc["islands"]] == [[4.0, 2.0], [2.0]]
    assert FleetPlan.from_json(doc).to_json() == doc
    loaded = fleet_islands.load_plan(doc)
    assert loaded["syncer"] == doc["syncer"] and loaded["islands"][0]["pools"][0]["usd_per_h"] == 4.0
    cfg["fleet"]["trainer_argv"] += ["--controller", "local", "--syncer-region", "modal/us-east"]
    assert compile_fleet(cfg)[1].to_json()["syncer"]["placement"] == {"where": "vm", "provider": "modal"}


def test_a_run_that_is_no_fleet_keeps_its_cost_view():
    r = Reducer(prices=load_prices(None))
    r.feed({"event": "island_ready", "island_id": 0, "time_unix": T0, "cloud": "modal", "gpu": "H100", "gpus": 1})
    r.feed({"event": "rl_round_trained", "island_id": 0, "time_unix": T0 + 60, "rollout_id": 0})
    cost = r.cost(T0 + 120)
    assert "fleet" not in cost and all("kind" not in row for row in cost["islands"])


# -- per pool mismatch, breakers ---------------------------------------------------------------
def test_a_single_pool_island_gets_its_trainer_mismatch(no_plan):
    blocks = {b["island"]: b for b in no_plan.page_view(live=False)["async"]["islands"]}
    for k, want_diff in ((0, 0.02), (1, 0.03)):
        m = blocks[k]["pools"][0]["mismatch"]
        assert m["logprob_diff"]["value"] == pytest.approx(want_diff)
        assert m["tis_clip"]["value"] == pytest.approx(0.002 * (k + 1))
        assert m["logprob_diff"]["method"] == "single-pool rounds" and m["logprob_diff"]["rounds"] == ROUNDS


def test_a_ratio_gauge_is_compared_with_its_ratio(no_plan):
    row = next(x for L in no_plan.breakers_view()["layers"] for x in L["rows"] if x["name"] == "sampler.vram_guard")
    assert row["gauge"]["limit"] == 0.97 and row["gauge"]["unit"] == "ratio"


def test_status_snapshot_limit_takes_the_gauge_unit():
    from yeto.breakers.registry import default_registry
    from yeto.breakers.status import primary_limit

    reg = {b.name: b for b in default_registry()}
    params = {"high_frac_ratio": 0.97, "interval_s": None, "realert_s": 600.0}
    assert primary_limit(params) == 600.0  # without the entry: the first non-ratio number, as before
    assert primary_limit(params, reg["sampler.vram_guard"]) == 0.97
    assert primary_limit({"timeout_s": 3600.0}, reg["trainer.sample_wait"]) == 3600.0


# -- the trainer's echoed reader events carry its island -----------------------------------------
def test_reader_events_carry_the_trainer_island(monkeypatch, capsys):
    from yeto.rl.adapters.miles import buffer_rollout as br

    env = br.island_env("trainer", store_root="/s", max_policy_age=4, island_id=1)
    assert env[br.ISLAND_ID_ENV] == "1" and br.ISLAND_ID_ENV in br.ISLAND_ENVS
    assert br.ISLAND_ID_ENV not in br.island_env("sampler", store_root="/s", sampler_id="a", island_id=1)
    monkeypatch.setenv(br.ISLAND_ID_ENV, "1")
    br._echo_event(br.WAIT_EVENT, rollout_id=3, waited_s=225.0, groups=0, needed=4)
    br._echo_event("rl_other", island_id=0)  # a record naming its island keeps it
    monkeypatch.delenv(br.ISLAND_ID_ENV)
    br._echo_event(br.WAIT_EVENT, rollout_id=4)
    recs = [json.loads(line.split(" ", 1)[1]) for line in capsys.readouterr().out.splitlines()]
    assert [(x["event"], x["island_id"]) for x in recs] == [(br.WAIT_EVENT, 1), ("rl_other", 0), (br.WAIT_EVENT, None)]


def test_apply_sample_role_exports_the_learner_id(monkeypatch):
    from types import SimpleNamespace

    from yeto.rl.adapters.miles import buffer_rollout as br
    from yeto.rl.adapters.miles import island_entry
    from yeto.rl.engine import sample_roles

    seen = {}
    monkeypatch.setattr(sample_roles, "sample_role", lambda args: sample_roles.TRAINER)
    monkeypatch.setattr(sample_roles, "resolve", lambda args, groups_per_round: SimpleNamespace(
        role=sample_roles.TRAINER, store="/s", wait_s=None))
    monkeypatch.setattr(sample_roles, "buffer_lane", lambda args: None)
    monkeypatch.setattr(br, "island_env", lambda role, **kw: seen.update(kw) or {})
    args = SimpleNamespace(learner_id=1, rl_max_policy_age=4, groups_per_round=4)
    island_entry.apply_sample_role(args, SimpleNamespace(), environ={})
    assert seen["island_id"] == 1


# -- the page --------------------------------------------------------------------------------------
def test_breakers_panel_is_last_and_the_topology_is_followed_by_the_metrics():
    html = render_page(None)
    order = [html.index(f'id="{i}"') for i in ("alertBox", "topoBox", "rFleet", "rCost", "detailsH", "islSw", "fleetBox",
                                                "async", "rounds", "util", "nodes", "runWide", "cost", "events", "cmds",
                                                "unused", "brkBox")]
    assert order == sorted(order)
    assert html.index('id="navBrk"') > html.index('href="#unused"')


SMOKE = r"""
const fs = require("fs");
const html = fs.readFileSync(process.argv[2], "utf8");
const data = html.match(/<script type="application\/json" id="yeto-data">([\s\S]*?)<\/script>/)[1];
const app = html.match(/<script>\n([\s\S]*?)\n<\/script>/)[1];
const els = {};
function node(id) {
  return { id, innerHTML: "", textContent: "", hidden: false, style: {}, dataset: {}, children: [], attrs: {},
    listeners: {}, offsetWidth: 200, offsetHeight: 120, clientWidth: 1200,
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    setAttribute(k, v) { this.attrs[k] = String(v); }, getAttribute(k) { return this.attrs[k]; },
    removeAttribute(k) { delete this.attrs[k]; }, appendChild(c) { this.children.push(c); return c; },
    addEventListener(t, fn) { (this.listeners[t] = this.listeners[t] || []).push(fn); },
    getBoundingClientRect() { return { left: 0, top: 0, width: 1200, height: 400 }; },
    querySelectorAll() { return []; } };
}
function mk(id) { if (!els[id]) { els[id] = node(id); if (id === "yeto-data") els[id].textContent = data; } return els[id]; }
global.document = { documentElement: node("html"), getElementById: mk, querySelectorAll: () => [], title: "",
  createElementNS: () => node(null), createElement: () => node(null) };
global.navigator = {}; global.localStorage = { getItem() { return null; }, setItem() {} };
global.setInterval = () => 0; global.setTimeout = () => 0;
eval(app);
console.log(JSON.stringify({ brk: els.brk.innerHTML, alerts: els.alertBox.innerHTML, cost: els.costBox.innerHTML }));
"""


def _page(tmp_path, r):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    page = export_html(r, tmp_path / "p.html", generated_at=r.max_ts)
    script = tmp_path / "smoke.js"
    script.write_text(SMOKE)
    res = subprocess.run([node, str(script), str(page)], capture_output=True, text=True, timeout=60, check=False)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def test_breakers_collapsed_to_one_line_and_linked_from_the_alerts(tmp_path):
    from test_dashboard_breakers import fixture_reducer

    out = _page(tmp_path, fixture_reducer())
    brk = out["brk"]
    assert brk.startswith('<details class="bdet" id="brkDet"><summary class="bsum">')  # collapsed
    summary = brk[:brk.index("</summary>")]
    assert "13 armed" in summary and "4 trips" in summary and "newest " in summary
    assert "dependency check" in summary
    assert "trainer.sample_wait" in brk and "Dependency checks" in brk  # the details are still drawn
    assert 'href="#brkBox"' in out["alerts"] and "breaker trainer.sample_wait tripped" in out["alerts"]


def test_fleet_cost_table_names_trainers_pools_and_the_syncer(tmp_path):
    cost = _page(tmp_path, reduce(write_store(tmp_path / "s", plan_doc(prices=True))))["cost"]
    for want in ("Island 0 trainer", "Island 1 pool i1-sglang-l40s", "fps-i1-1", "Syncer", "$0.26",
                 "fleet spend = pools $"):
        assert want in cost, want
    assert "Specs: ." not in cost and "undefined" not in cost and "NaN" not in cost


def test_a_per_island_fleet_s_trainers_cost_what_run_budget_counts(tmp_path):
    """fleet.launch per_island: the cost panel bills each trainer island from its launch
    records with the function run.budget counts them with (trainer_worker.island_spend of
    billed): an island moved from $8/h to $100/h, and one that ran 1 h without a price
    (its declared $5.06/h), not the plan price times the tape hours."""
    from yeto.rl import trainer_worker as tw

    plan = dict(plan_doc(prices=True), launch="per_island", max_islands=2)
    root = write_store(tmp_path / "s", plan)
    objs = DirectoryObjectStore(str(root))
    now = T0 + 3 * 3600
    tw.save_record(objs, 0, {"prefix": "r-i0", "prefix_now": "r-i0-m1", "incarnation": 1, "state": tw.RUNNING,
                             "gpu": "aws:2xh100@us-east-2", "usd_per_h": 100.0, "billed_s": 3600.0, "billed_usd": 8.0,
                             "started_at": T0 + 3600, "closed_at": None}, T0)
    tw.save_record(objs, 1, {"prefix": "r-i1", "prefix_now": "r-i1", "incarnation": 0, "state": tw.RUNNING,
                             "gpu": "aws:2xh100@us-east-2", "usd_per_h": None, "started_at": T0 + 2 * 3600,
                             "closed_at": None}, T0)
    r = reduce(root)
    pf = fleet_islands.PlanFile([], objs)
    pf.pump(r)
    assert pf.pump_trainer_spend(r, now=now)
    cost = r.cost(r.max_ts)
    rows = {row["id"]: row for row in cost["islands"]}
    expected = tw.island_spend(tw.billed(objs, now), {0: 5.06, 1: 5.06})
    assert expected[0][1] == pytest.approx(8.0 + 2 * 100.0) and expected[1][1] == pytest.approx(5.06)
    for k in (0, 1):
        assert rows[str(k)]["cost_usd"] == pytest.approx(expected[k][1])
        assert rows[str(k)]["hours"] == pytest.approx(expected[k][0])
    assert cost["fleet"]["trainers_usd"] == pytest.approx(expected[0][1] + expected[1][1])
    # a synced group's plan: the plan price times the tape hours, as before
    assert reduce(write_store(tmp_path / "g", plan_doc(prices=True))).cost(T0)["fleet"]["trainers_usd"] != \
        pytest.approx(expected[0][1] + expected[1][1])
