"""A fleet of async GRPO islands on the dashboard (yeto/dashboard/fleet_islands.py, the fleet
part of topology_diloco.py and async_rl.py): two trainers of one synced group, each with its
own external sampler pools on its store lane trainer<k>/, mixed engines in island 0."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from dashboard_helpers import FX

from yeto.dashboard import fleet_islands, sources
from yeto.dashboard.async_rl import AsyncState
from yeto.dashboard.export import export_view
from yeto.dashboard.reducer import Reducer
from yeto.dashboard.sources import load_all
from yeto.dashboard.store_tapes import StoreTapes, tape_of
from yeto.rl.engine.object_store import DirectoryObjectStore

T0 = 1_790_000_000.0
STORE = "s3://bucket/runs/fleet-qa"
ROUNDS = 10
# island -> pools: (name, source, shard base, engine, card, true logprob |diff|, true TIS clipfrac)
POOLS = {
    0: [("i0-vllm-h100", "vllm-h100", "qa-v-i0", "vllm", "H100", 0.08, 0.05),
        ("i0-sgl-l40s", "sgl-l40s", "qa-s-i0", "sglang", "L40S", 0.02, 0.005)],
    1: [("i1-sgl-a", "sgl-a", "qa-a-i1", "sglang", "L40S", 0.02, 0.004),
        ("i1-sgl-b", "sgl-b", "qa-b-i1", "sglang", "L40S", 0.03, 0.006)],
}
NAMES = {0: "east-h100", 1: "west-h100"}


def plan_doc() -> dict:
    islands = []
    for k, pools in POOLS.items():
        islands.append({"index": k, "name": NAMES[k], "learner": k, "substore": f"trainer{k}", "store": f"{STORE}/trainer{k}",
                        "gpu": "modal:1x2xh100", "usd_per_h": 8.0, "release_every": 1, "max_policy_age": 2,
                        "buffer_shards": [f"{p[2]}-0" for p in pools],
                        "pools": [{"name": p[0], "source_name": p[1], "provider": "modal", "shard_base": p[2],
                                   "engine": p[3], "rollout_quant": None} for p in pools]})
    return {"version": 1, "run": "fleet-qa", "store": STORE, "dispense": "fleet", "islands": islands}


def _samples(k: int, r: int) -> list[int]:
    """Samples each pool of island ``k`` gave round ``r`` (shares vary, so the pools separate;
    island 1's round 0 comes from its first pool alone)."""
    if k == 0:
        return [2 + 2 * (r % 3), 8 - 2 * (r % 3)]
    return [8, 0] if r == 0 else [4 + (r % 2) * 2, 4]


def fleet_records(*, island_field: bool = True, store_field: bool = True) -> list[dict]:
    """Two islands, 10 trained rounds each, both islands releasing v1..v5 on their own lanes."""
    recs: list[dict] = []
    for k in POOLS:
        recs.append({"event": "island_ready", "time_unix": T0, "island": f"fleet-qa-l{k}-modal", "island_id": k,
                     "cloud": "modal", "gpu": "H100", "gpus": 2, "initial": True})
        recs += [{"event": "rl_driver_start", "island_id": k, "time_unix": T0 + 10, "placement": "colocated"},
                 {"event": "rl_engine_selected", "island_id": k, "time_unix": T0 + 11, "rl/outer_sync": True,
                  "placement": "colocated"}]
        for j, (name, _src, base, engine, card, _d, _c) in enumerate(POOLS[k]):
            sid = f"{base}-0"
            start = {"event": "rl_sampler_start", "sampler_id": sid, "time_unix": T0 + 20 + j, "max_backlog_groups": 32,
                     "num_rollouts": None, "until_trainers_end": 1,
                     "provenance": {"card_model": card, "engine": engine, "precision": "bf16", "topology": "tp1",
                                    "device_family": "cuda"}}
            if island_field:
                start["island"] = k
            if store_field:
                start["store"] = f"{STORE}/trainer{k}"
            recs.append(start)
            recs.append({"event": "rl_sampler_join", "sampler_id": sid, "time_unix": T0 + 21 + j, "sampler": sid,
                         "incarnation": f"{sid}-x1"})
    for r in range(ROUNDS):
        t = T0 + 100 + 60 * r
        for k, pools in POOLS.items():
            n = _samples(k, r)
            for (name, _src, base, _eng, _card, _d, _c), m in zip(pools, n):
                sid = f"{base}-0"
                recs.append({"event": "rl_sampler_round", "sampler_id": sid, "time_unix": t - 30, "rollout_id": r,
                             "groups": max(1, m // 2), "policy_version": r // 2, "version_lag": r % 2, "generate_s": 28.0})
                for s in range(m):
                    recs.append({"event": "rl_trajectory_reward", "island_id": k, "time_unix": t - 2, "rollout_id": r,
                                 "trajectory_id": f"{sid}-x1-r{r}-row{s // 2}-{s % 2}", "reward": 0.5, "group_index": s // 2})
            diff = sum(m * p[5] for p, m in zip(pools, n)) / sum(n)
            clip = sum(m * p[6] for p, m in zip(pools, n)) / sum(n)
            recs.append({"event": "rl_round_trained", "island_id": k, "time_unix": t, "rollout_id": r, "local_round_id": r + 1,
                         "trained_groups": 8, "trained_samples": 16, "policy_age_mean": 0.5, "policy_age_max": 1,
                         "policy_age_groups": {"0": 4, "1": 4}, "reward_mean": 0.4 + 0.01 * r, "grad_norm": 0.3,
                         "train_metrics": {"train_rollout_logprob_abs_diff": diff, "tis_clipfrac": clip}})
            recs.append({"event": "rl_buffer_stale_skipped", "island_id": k, "time_unix": t - 5, "rollout_id": r,
                         "stale_groups": 0, "max_policy_age": 2})
            recs.append({"event": "rl_driver_phase", "island_id": k, "time_unix": t + 1, "phase": "train"})
            recs.append({"event": "rl_policy_snapshot", "island_id": k, "time_unix": t + 2, "rl/policy_hash": f"{k}abc{r:04d}ef",
                         "rl/fragment_versions": [r // 2 * 2 + 1, r // 2 * 2 + 2]})
            if r % 2 == 1:
                v = r // 2 + 1
                recs.append({"event": "rl_policy_released", "island_id": k, "time_unix": t + 5, "policy_version": v,
                             "local_version": v, "released_at": t + 4, "buffer_selected": 8 * (r + 1), "release_bytes": 8e9,
                             "release_timings": {"upload_s": 12.0}, "origin": {"kind": "local"}})
                for j, p in enumerate(pools):
                    recs.append({"event": "rl_sampler_policy_loaded", "sampler_id": f"{p[2]}-0", "time_unix": t + 20 + j,
                                 "policy_version": v, "released_at": t + 4, "seen_at": t + 8,
                                 "follower_timings": {"resolve_s": 2.0, "load_s": 6.0}})
            recs.append({"event": "rl_fragment_push", "island_id": k, "time_unix": t + 3, "global_step": r + 1,
                         "fragment_id": r % 2, "base_version": r, "payload_bytes": 4e8, "realized_h": 1,
                         "pull_to_push_seconds": 1.5})
        recs.append({"step": r + 1, "fragment": r % 2, "attempt": 1, "expected": [0, 1], "responded": [0, 1],
                     "responders": [{"id": i, "arrived_unix": t + 4, "recv_seconds": 0.5} for i in (0, 1)],
                     "quorum": 2, "quorum_ms": 900, "sync/merge_seconds": 0.1, "ms": 1100, "time_unix": t + 6})
        for k in POOLS:
            recs.append({"event": "rl_fragment_bcast", "island_id": k, "time_unix": t + 9, "fragment_id": r % 2,
                         "version": r + 1, "payload_bytes": 4e8, "queue_seconds": 0.4})
    return sorted(recs, key=lambda x: x.get("time_unix", 0.0))


def fleet_reducer(*, plan: bool = True, **kw) -> Reducer:
    r = Reducer(run="fleet-qa")
    for rec in fleet_records(**kw):
        r.feed(rec)
    if plan:
        assert r.load_fleet_plan(plan_doc(), "test")
    return r


def _diloco(r: Reducer) -> dict:
    return r.page_view(now=r.max_ts)["topology"]["diloco"]


# -- 4.1 pools bind to their island ------------------------------------------------------------
@pytest.mark.parametrize("kw", [{"plan": True}, {"plan": False},
                                {"plan": True, "island_field": False, "store_field": False},
                                {"plan": False, "island_field": False}])
def test_two_islands_mixed_engines_no_pool_on_the_wrong_island(kw):
    """By rl_sampler_start.island, by the plan's shard bases, or by the trainer<k> store."""
    r = fleet_reducer(**kw)
    if not kw["plan"] and not kw.get("island_field", True):
        assert fleet_islands.resolve(r) is None  # no plan, no island field: the follow lanes of a synced group
        d = _diloco(r)
        assert {k: sorted(v["samplers"]) for k, v in d["lanes"].items()} == {"0": ["qa-s-i0-0", "qa-v-i0-0"],
                                                                         "1": ["qa-a-i1-0", "qa-b-i1-0"]}
        return
    d = _diloco(r)
    assert d["kind"] == "synced_group" and d["follow"] and d["publisher"] is None
    fl = d["fleet"]
    assert [x["island"] for x in fl["islands"]] == [0, 1] and fl["unbound"] == []
    for x in fl["islands"]:
        assert x["trainer"] == str(x["island"]) and x["substore"] == f"trainer{x['island']}"
        for p in x["pools"]:  # every sampler sits in the pool and island its shard was written for
            assert p["samplers"] and all(s.endswith(f"-i{x['island']}-0") for s in p["samplers"])
    engines = {x["island"]: sorted(p["engine"] for p in x["pools"]) for x in fl["islands"]}
    assert engines == {0: ["SGLang", "vLLM"], 1: ["SGLang", "SGLang"]}
    for n in d["sampler_nodes"]:
        assert n["trainer"] == str(n["island"]) and n["id"].endswith(f"-i{n['island']}-0")
    assert {k: sorted(v["samplers"]) for k, v in d["lanes"].items()} == {"0": ["qa-s-i0-0", "qa-v-i0-0"],
                                                                     "1": ["qa-a-i1-0", "qa-b-i1-0"]}
    if kw["plan"]:
        assert [x["name"] for x in fl["islands"]] == ["east-h100", "west-h100"] and fl["source"] == "plan"
        pools = {p["name"]: p for x in fl["islands"] for p in x["pools"]}
        assert pools["i0-vllm-h100"]["card"] == "H100" and pools["i0-vllm-h100"]["precision"] == "BF16"
        assert pools["i0-sgl-l40s"]["provider"] == "modal" and pools["i0-sgl-l40s"]["source_name"] == "sgl-l40s"


def test_lane_follows_the_learner_id_not_the_island_id():
    """A synced group with sampler islands 0, 1 and trainer islands 2, 3: the trainers are
    learners 0 and 1, so the samplers on trainer0/ and trainer1/ feed islands 2 and 3."""
    r = Reducer(run="mixed")
    for k, tid in ((0, 2), (1, 3)):
        r.feed({"event": "rl_engine_selected", "island_id": tid, "time_unix": T0, "rl/outer_sync": True})
        r.feed({"event": "rl_round_trained", "island_id": tid, "time_unix": T0 + 10, "rollout_id": 0, "trained_groups": 4})
        r.feed({"event": "rl_sampler_start", "island_id": k, "time_unix": T0 + 1, "store": f"{STORE}/trainer{k}"})
        r.feed({"event": "rl_sampler_round", "island_id": k, "time_unix": T0 + 5, "rollout_id": 0, "groups": 4})
    d = _diloco(r)
    assert "fleet" not in d and d["follow"]
    assert {k: v["samplers"] for k, v in d["lanes"].items()} == {"2": ["0"], "3": ["1"]}
    assert d["lanes"]["2"]["substore"] == "trainer0" and d["lanes"]["3"]["substore"] == "trainer1"


def test_plan_draws_the_fleet_before_any_pool_writes():
    r = Reducer(run="fleet-qa")
    for rec in fleet_records():
        if "sampler_id" not in rec:
            r.feed(rec)
    r.load_fleet_plan(plan_doc(), "test")
    d = _diloco(r)
    assert d["kind"] == "synced_group" and "fleet plan" in r.page_view(now=r.max_ts)["layout"]["reason"]
    assert [len(x["pools"]) for x in d["fleet"]["islands"]] == [2, 2]
    assert all(p["samplers"] == [] for x in d["fleet"]["islands"] for p in x["pools"])


def test_load_plan_rejects_other_documents():
    assert fleet_islands.load_plan({"islands": []}) is None  # no version
    assert fleet_islands.load_plan({"version": 1, "islands": "x"}) is None
    assert fleet_islands.load_plan([1]) is None
    plan = fleet_islands.load_plan({"version": 1, "islands": [{"pools": [{"name": "a"}], "extra": 1}], "new_key": 2})
    assert plan["islands"][0]["learner"] == 0 and plan["islands"][0]["substore"] == "trainer0"
    r = Reducer()
    assert r.load_fleet_plan(plan_doc(), "a") and not r.load_fleet_plan(plan_doc(), "a")  # unchanged: no reload


# -- 4.3 per-island async GRPO panel --------------------------------------------------------
def test_two_islands_both_at_v4_make_two_release_rows():
    asy = fleet_reducer().page_view(now=T0 + 1000)["async"]
    v4 = [row for row in asy["releases"] if row["policy_version"] == 4]
    assert [row["island"] for row in v4] == [0, 1] and [row["trainer"] for row in v4] == ["0", "1"]
    assert sorted(v4[0]["loads"]) == ["qa-s-i0-0", "qa-v-i0-0"] and sorted(v4[1]["loads"]) == ["qa-a-i1-0", "qa-b-i1-0"]
    assert asy["backlog"] is None  # one read position per lane: no shared backlog
    blocks = {b["island"]: b for b in asy["islands"]}
    for k, b in blocks.items():
        assert [row["policy_version"] for row in b["releases"]] == [1, 2, 3, 4, 5] and b["latest"] == 5
        assert {row["island"] for row in b["releases"]} == {k} and b["trainer"] == str(k)
        assert b["backlog"]["has_reads"] and b["policy_age"]["hist"] == {"0": 40.0, "1": 40.0}
        assert b["policy_age"]["mean_last"] == 0.5 and b["max_policy_age"] == 2 and b["name"] == NAMES[k]
        # island 0 wrote 2+3+4+... groups per round, its trainer read 8 per round: its own lane's numbers
    assert blocks[0]["backlog"]["written"] != blocks[1]["backlog"]["written"]
    assert blocks[0]["backlog"]["read"] == blocks[1]["backlog"]["read"] == 80.0
    assert [p["name"] for p in blocks[0]["pools"]] == ["i0-vllm-h100", "i0-sgl-l40s"]
    assert blocks[0]["pools"][0]["lag_max"] == 1.0 and blocks[0]["pools"][0]["policy_version"] == 4


def test_release_rows_without_a_fleet_keep_their_keys():
    st = AsyncState()
    for iid, v in (("0", 4), ("1", 4)):
        st.fold({"event": "rl_policy_released", "policy_version": v, "released_at": T0}, iid, T0)
    assert [r["policy_version"] for r in st.release_rows()] == [4]  # one row per version, as before
    rows = st.release_rows(island_of={"0": 0, "1": 1})
    assert [(r["island"], r["policy_version"]) for r in rows] == [(0, 4), (1, 4)]


# -- 4.4 per-pool mismatch ---------------------------------------------------------------------
def test_per_pool_logprob_diff_and_tis_clipfrac_are_split():
    blocks = {b["island"]: b for b in fleet_reducer().page_view(now=T0 + 1000)["async"]["islands"]}
    mm = {p["name"]: p["mismatch"] for b in blocks.values() for p in b["pools"]}
    for k, pools in POOLS.items():
        for name, _src, _base, _eng, _card, diff, clip in pools:
            assert mm[name]["logprob_diff"]["value"] == pytest.approx(diff, abs=1e-6), name
            assert mm[name]["tis_clip"]["value"] == pytest.approx(clip, abs=1e-6), name
    # island 0's vLLM pool is the one whose numerics drift: it reads 4x its SGLang neighbour
    assert mm["i0-vllm-h100"]["logprob_diff"]["value"] > 3 * mm["i0-sgl-l40s"]["logprob_diff"]["value"]
    assert mm["i0-vllm-h100"]["logprob_diff"]["method"] == "least squares over mixed rounds"
    assert mm["i1-sgl-a"]["logprob_diff"]["method"] == "single-pool rounds"  # round 0 came from it alone
    assert mm["i1-sgl-a"]["logprob_diff"]["alone"] == 1 and mm["i1-sgl-b"]["logprob_diff"]["rounds"] == ROUNDS - 1


def test_per_pool_split_prefers_xhw_token_shares_and_marks_inseparable_pools():
    pool_of = {"a-0": "A", "b-0": "B"}
    rounds = [{"round": r, "logprob_diff": 0.05, "tis_clip": None,
               "by_sampler": {"a-0": {"samples": 5}, "b-0": {"samples": 5}}} for r in range(6)]
    out = fleet_islands.pool_mismatch(rounds, pool_of)
    assert out["A"]["logprob_diff"]["method"] == "share-weighted (pools not separable)"
    assert out["A"]["logprob_diff"]["value"] == pytest.approx(0.05) and "tis_clip" not in out["A"]
    xhw = {r: {"a-0": 0.2 + 0.1 * r, "b-0": 0.8 - 0.1 * r} for r in range(6)}
    rounds = [dict(x, logprob_diff=xhw[x["round"]]["a-0"] * 0.1 + xhw[x["round"]]["b-0"] * 0.01) for x in rounds]
    out = fleet_islands.pool_mismatch(rounds, pool_of, xhw)
    assert out["A"]["logprob_diff"]["value"] == pytest.approx(0.1) and out["B"]["logprob_diff"]["value"] == pytest.approx(0.01)


# -- 4.5 a run that is no fleet keeps its view -------------------------------------------------
@pytest.mark.parametrize("name", ["p7", "async-e2e", "f1-follow"])
def test_no_fleet_keys_without_a_fleet(name):
    paths = ([str(FX / "async_grpo_replay" / "p7")] if name == "p7"
             else [str(FX / "diloco" / name), str(FX / "diloco" / name / "events")])
    r = Reducer(run=name)
    load_all(r, [str(p) for p in sources.discover(paths)])
    assert fleet_islands.resolve(r) is None
    view = export_view(r, generated_at=0.0)["page"]
    asy = view["async"]
    assert "islands" not in asy and all("island" not in row for row in asy["releases"])
    assert "fleet" not in ((view.get("topology") or {}).get("diloco") or {})
    assert asy == copy.deepcopy(r.async_.view())


# -- reading the plan and the islands' lane tapes -------------------------------------------------
def test_plan_file_reads_a_local_copy_and_the_store(tmp_path):
    store = tmp_path / "store"
    (store / "fleet").mkdir(parents=True)
    (store / "fleet" / "plan.json").write_text(json.dumps(plan_doc()))
    r = Reducer()
    pf = fleet_islands.PlanFile(fleet_islands.plan_candidates([str(store)]))
    assert pf.pump(r) and r.fleet_plan["islands"][1]["name"] == "west-h100" and not pf.pump(r)
    r2 = Reducer()
    pf2 = fleet_islands.PlanFile([], DirectoryObjectStore(str(store)))
    assert pf2.pump(r2) and r2.fleet_plan_source == "store fleet/plan.json"
    assert pf2.substores == ["trainer0", "trainer1"]


def test_plan_file_reads_the_pools_tapes_under_their_island_lane(tmp_path):
    """External samplers upload their tapes under <store>/trainer<k>/tapes/ (their store is
    the island's lane): the plan's lanes are read too."""
    store = tmp_path / "store"
    (store / "fleet").mkdir(parents=True)
    (store / "fleet" / "plan.json").write_text(json.dumps(plan_doc()))
    for rec in fleet_records():
        sid = rec.get("sampler_id")
        if sid:
            k = 0 if "-i0-" in sid else 1
            path = store / f"trainer{k}" / "tapes" / sid / "x1" / "000001.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
    r = Reducer()
    pf = fleet_islands.PlanFile([], DirectoryObjectStore(str(store)))
    pf.pump(r)
    assert pf.pump_tapes(r) > 0
    assert sorted(i for i in r.islands if r.async_.roles.get(i) == "sampler") == [
        "qa-a-i1-0", "qa-b-i1-0", "qa-s-i0-0", "qa-v-i0-0"]
    assert sorted(x for t in pf.tapes() for x in t.sources) == [
        "store/trainer0:qa-s-i0-0/x1", "store/trainer0:qa-v-i0-0/x1",
        "store/trainer1:qa-a-i1-0/x1", "store/trainer1:qa-b-i1-0/x1"]
    assert pf.pump_tapes(r) == 0  # read once


def test_store_tapes_prefix():
    assert tape_of("tapes/s/i/000001.jsonl") == "tapes/s/i/"
    assert tape_of("trainer0/tapes/s/i/000001.jsonl") is None
    assert tape_of("trainer0/tapes/s/i/000001.jsonl", "trainer0/tapes/") == "trainer0/tapes/s/i/"
    assert StoreTapes(None).prefix == "tapes/"


# -- the page ----------------------------------------------------------------------------------
@pytest.mark.parametrize("width", [1200, 400])
def test_page_draws_one_box_per_island(tmp_path, width):
    from test_dashboard_diloco import _smoke

    r = fleet_reducer()
    out = _smoke(tmp_path, r, width=width)
    joined = " | ".join(out["texts"])
    assert out["viewBox"].startswith("0 0 1480 " if width > 640 else "0 0 400 ")
    for want in ("Island 0", "Island 1", "trainer0/", "trainer1/", "i0-vllm-h100 · 1 sampler", "i1-sgl-b · 1 sampler",
                 "H100 · BF16 · vLLM", "L40S · BF16 · SGLang", "Syncer", "p0 · vllm-h100", "p1 · sgl-l40s"):
        assert want in joined, want
    assert "east-h100" in joined and "west-h100" in joined
    assert "n/a" not in joined, [t for t in out["texts"] if "n/a" in t]
    assert "in 2 island(s)" in out["note"]


def test_screenshot_fixture_store(tmp_path):
    """The records above, written as a local run store (the screenshot tool reads it)."""
    write_store(tmp_path / "store")
    r = Reducer(run="fleet-qa")
    load_all(r, [str(tmp_path / "store" / "tapes")])
    pf = fleet_islands.PlanFile(fleet_islands.plan_candidates([str(tmp_path / "store" / "tapes")]))
    assert pf.pump(r)
    assert len(_diloco(r)["fleet"]["islands"]) == 2


def write_store(root: Path) -> Path:
    """A local run store holding the fleet's plan and one tape per island."""
    (root / "fleet").mkdir(parents=True, exist_ok=True)
    (root / "fleet" / "plan.json").write_text(json.dumps(plan_doc()))
    (root / "tapes").mkdir(exist_ok=True)
    by: dict[str, list[dict]] = {}
    for rec in fleet_records():
        name = rec.get("sampler_id") or (f"l{rec['island_id']}" if "island_id" in rec else
                                          "fleet" if rec.get("event") else "yeto-tape")
        by.setdefault(name, []).append(rec)
    for name, recs in by.items():
        (root / "tapes" / f"{name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in recs))
    return root


def test_plan_written_by_run_up_binds_shards_to_their_pool():
    """The plan ``yeto run up`` compiles from the example fleet run.json: shards bind to their
    pool by its shard base; a pool the plan has no engine for takes its samplers' provenance."""
    from yeto.rl.fleet import compile_fleet

    cfg = json.loads((Path(__file__).parents[1] / "examples" / "fleet" / "run.two-islands.json").read_text())
    _, plan = compile_fleet(cfg)
    r = Reducer()
    for k in (0, 1):
        r.feed({"event": "rl_engine_selected", "island_id": k, "time_unix": T0, "rl/outer_sync": True})
        r.feed({"event": "rl_round_trained", "island_id": k, "time_unix": T0 + 9, "rollout_id": 0, "trained_groups": 4})
    for sid, engine in (("<run>-vllm-i0-2", "vllm"), ("<run>-sglang-i0-0", "sglang"), ("<run>-sglang-i1-5", "sglang")):
        r.feed({"event": "rl_sampler_start", "sampler_id": sid, "time_unix": T0 + 1,
                "provenance": {"engine": engine, "card_model": "H100", "precision": "bf16"}})
        r.feed({"event": "rl_sampler_round", "sampler_id": sid, "time_unix": T0 + 5, "rollout_id": 0, "groups": 2})
    assert r.load_fleet_plan(json.loads(json.dumps(plan.to_json())), "test")
    fl = _diloco(r)["fleet"]
    pools = {p["name"]: (x["island"], p["engine"], p["samplers"]) for x in fl["islands"] for p in x["pools"]}
    assert pools == {"i0-vllm-h100": (0, "vLLM", ["<run>-vllm-i0-2"]),
                     "i0-sglang-l40s": (0, "SGLang", ["<run>-sglang-i0-0"]),
                     "i1-sglang-l40s": (1, "SGLang", ["<run>-sglang-i1-5"])}
