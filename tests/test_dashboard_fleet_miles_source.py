"""A fleet island with both kinds of sampler on the dashboard (yeto/dashboard/fleet_islands.py,
topology_diloco.py, async_rl.py): island 1 = its trainer + a Miles sampler island of the
launch (launch island 2, events on the launch's tape) + an external SGLang pool (its tapes
under <store>/trainer1/tapes/). Both bind to island 1 and each names its source."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from yeto.dashboard import fleet_islands
from yeto.dashboard.export import export_html, export_view
from yeto.dashboard.reducer import Reducer

T0 = 1_790_000_000.0
STORE = "s3://bucket/runs/mixed-qa"
ROUNDS = 10
MILES_SID = "2"  # the Miles sampler island: launch island 2 (after trainers 0 and 1)
MILES_INC = "isl2-x1"
# island -> samplers: (kind, name, sid, engine, card, true logprob |diff|, true TIS clipfrac)
SAMPLERS = {
    0: [("external", "i0-vllm-h100", "qa-v-i0-0", "vllm", "H100", 0.08, 0.05)],
    1: [("miles", "i1-miles-l40s", MILES_SID, "sglang", "L40S", 0.01, 0.002),
        ("external", "i1-sgl-l40s", "qa-s-i1-0", "sglang", "L40S", 0.03, 0.006)],
}


def plan_doc() -> dict:
    """The plan.json ``yeto run up`` writes for this fleet (#715 form)."""
    islands = []
    for k, rows in SAMPLERS.items():
        pools = [r for r in rows if r[0] == "external"]
        miles = [r for r in rows if r[0] == "miles"]
        islands.append({"index": k, "name": ["vllm", "mixed"][k], "learner": k, "substore": f"trainer{k}",
                        "store": f"{STORE}/trainer{k}", "gpu": "modal:1x2xh100",
                        "usd_per_h": 8.0 + 2.0 * len(miles), "trainer_usd_per_h": 8.0,
                        "release_every": 1, "max_policy_age": 2,
                        "buffer_shards": [f"island{r[2]}" for r in miles] + [r[2] for r in pools],
                        "pools": [{"name": r[1], "source_name": r[1].split("-", 1)[1], "source": "external",
                                   "provider": "modal", "shard_base": r[2][:-2], "engine": r[3],
                                   "rollout_quant": None} for r in pools],
                        "miles_samplers": [{"name": r[1], "source_name": r[1].split("-", 1)[1], "source": "miles",
                                            "gpu": "modal:1x1xl40s", "usd_per_h": 2.0, "launch_island": int(r[2]),
                                            "shard": f"island{r[2]}", "engine": "sglang"} for r in miles]})
    return {"version": 1, "run": "mixed-qa", "store": STORE, "dispense": "fleet", "islands": islands}


def _samples(k: int, r: int) -> list[int]:
    """Samples each sampler of island ``k`` gave round ``r`` (island 1's shares vary so the
    two sources separate; its round 0 comes from the Miles sampler alone)."""
    if k == 0:
        return [8]
    return [8, 0] if r == 0 else [2 + 2 * (r % 3), 8 - 2 * (r % 3)]


def _sampler_id(kind: str, sid: str) -> dict:
    """How a sampler's records name it: a Miles sampler island by the launch tape's island_id,
    an external pool by its sampler_id."""
    return {"island_id": int(sid)} if kind == "miles" else {"sampler_id": sid}


def records(*, island_field: bool = True) -> list[dict]:
    recs: list[dict] = []
    for k in SAMPLERS:
        recs.append({"event": "island_ready", "time_unix": T0, "island": f"mixed-qa-l{k}-modal", "island_id": k,
                     "cloud": "modal", "gpu": "H100", "gpus": 2, "initial": True})
        recs += [{"event": "rl_driver_start", "island_id": k, "time_unix": T0 + 10, "placement": "colocated"},
                 {"event": "rl_engine_selected", "island_id": k, "time_unix": T0 + 11, "rl/outer_sync": True,
                  "placement": "colocated"}]
    recs.append({"event": "island_ready", "time_unix": T0, "island": "mixed-qa-l2-modal", "island_id": 2,
                 "cloud": "modal", "gpu": "L40S", "gpus": 1, "initial": True})
    for k, rows in SAMPLERS.items():
        for j, (kind, _name, sid, engine, card, _d, _c) in enumerate(rows):
            ident = _sampler_id(kind, sid)
            inc = MILES_INC if kind == "miles" else f"{sid}-x1"
            if kind == "miles":  # Miles' own start record: its store (its trainer's sub-store), no island
                start = {"event": "rl_sampler_start", **ident, "time_unix": T0 + 20 + j,
                         "store": f"/mnt/yeto-store/runs/mixed-qa/trainer{k}", "num_rollouts": None,
                         "until_trainers_end": 1, "max_backlog_groups": 32}
            else:
                start = {"event": "rl_sampler_start", **ident, "time_unix": T0 + 20 + j, "max_backlog_groups": 32,
                         "num_rollouts": None, "until_trainers_end": 1, "store": f"{STORE}/trainer{k}",
                         "provenance": {"card_model": card, "engine": engine, "precision": "bf16", "topology": "tp1",
                                        "device_family": "cuda"}}
                if island_field:
                    start["island"] = k
            recs += [start, {"event": "rl_sampler_join", **ident, "time_unix": T0 + 21 + j, "incarnation": inc}]
    for r in range(ROUNDS):
        t = T0 + 100 + 60 * r
        for k, rows in SAMPLERS.items():
            n = _samples(k, r)
            for (kind, _name, sid, _eng, _card, _d, _c), m in zip(rows, n):
                ident = _sampler_id(kind, sid)
                inc = MILES_INC if kind == "miles" else f"{sid}-x1"
                if m:
                    recs.append({"event": "rl_sampler_round", **ident, "time_unix": t - 30, "rollout_id": r,
                                 "groups": m // 2, "policy_version": r // 2, "version_lag": r % 2, "generate_s": 28.0})
                for s in range(m):
                    recs.append({"event": "rl_trajectory_reward", "island_id": k, "time_unix": t - 2, "rollout_id": r,
                                 "trajectory_id": f"{inc}-r{r}-row{s // 2}-{s % 2}", "reward": 0.5,
                                 "group_index": s // 2})
            diff = sum(m * x[5] for x, m in zip(rows, n)) / sum(n)
            clip = sum(m * x[6] for x, m in zip(rows, n)) / sum(n)
            recs.append({"event": "rl_round_trained", "island_id": k, "time_unix": t, "rollout_id": r,
                         "local_round_id": r + 1, "trained_groups": 4, "trained_samples": 8, "policy_age_mean": 0.5,
                         "policy_age_max": 1, "policy_age_groups": {"0": 2, "1": 2}, "reward_mean": 0.4,
                         "grad_norm": 0.3, "train_metrics": {"train_rollout_logprob_abs_diff": diff,
                                                             "tis_clipfrac": clip}})
            recs.append({"event": "rl_buffer_stale_skipped", "island_id": k, "time_unix": t - 5, "rollout_id": r,
                         "stale_groups": 0, "max_policy_age": 2})
            recs.append({"event": "rl_driver_phase", "island_id": k, "time_unix": t + 1, "phase": "train"})
            recs.append({"event": "rl_policy_snapshot", "island_id": k, "time_unix": t + 2,
                         "rl/policy_hash": f"{k}abc{r:04d}ef", "rl/fragment_versions": [r + 1, r + 1]})
            if r % 2 == 1:
                v = r // 2 + 1
                recs.append({"event": "rl_policy_released", "island_id": k, "time_unix": t + 5, "policy_version": v,
                             "local_version": v, "released_at": t + 4, "buffer_selected": 4 * (r + 1),
                             "release_bytes": 8e9, "release_timings": {"upload_s": 12.0}, "origin": {"kind": "local"}})
                for j, (kind, _name, sid, *_rest) in enumerate(rows):
                    recs.append({"event": "rl_sampler_policy_loaded", **_sampler_id(kind, sid), "time_unix": t + 20 + j,
                                 "policy_version": v, "released_at": t + 4, "seen_at": t + 8,
                                 "follower_timings": {"resolve_s": 2.0, "load_s": 6.0}})
            recs.append({"event": "rl_fragment_push", "island_id": k, "time_unix": t + 3, "global_step": r + 1,
                         "fragment_id": r % 2, "base_version": r, "payload_bytes": 4e8, "realized_h": 1,
                         "pull_to_push_seconds": 1.5})
        recs.append({"step": r + 1, "fragment": r % 2, "attempt": 1, "expected": [0, 1], "responded": [0, 1],
                     "responders": [{"id": i, "arrived_unix": t + 4, "recv_seconds": 0.5} for i in (0, 1)],
                     "quorum": 2, "quorum_ms": 900, "sync/merge_seconds": 0.1, "ms": 1100, "time_unix": t + 6})
        for k in SAMPLERS:
            recs.append({"event": "rl_fragment_bcast", "island_id": k, "time_unix": t + 9, "fragment_id": r % 2,
                         "version": r + 1, "payload_bytes": 4e8, "queue_seconds": 0.4})
    return sorted(recs, key=lambda x: x.get("time_unix", 0.0))


def reducer(*, plan: bool = True, **kw) -> Reducer:
    r = Reducer(run="mixed-qa")
    for rec in records(**kw):
        r.feed(rec)
    if plan:
        assert r.load_fleet_plan(plan_doc(), "test")
    return r


def _diloco(r: Reducer) -> dict:
    return r.page_view(now=r.max_ts)["topology"]["diloco"]


# -- binding: both kinds in island 1, each with its source -------------------------------------
@pytest.mark.parametrize("plan", [True, False])
def test_island1_holds_the_miles_sampler_and_the_external_pool(plan):
    r = reducer(plan=plan)
    fl = fleet_islands.resolve(r)
    assert fl["island_of"][MILES_SID] == 1 and fl["island_of"]["qa-s-i1-0"] == 1 and fl["island_of"]["qa-v-i0-0"] == 0
    assert fl["unbound"] == []
    # without a plan: the Miles sampler is named by its buffer shard, a pool by its shard base
    miles, v0, s1 = ("i1-miles-l40s", "i0-vllm-h100", "i1-sgl-l40s") if plan else ("island2", "qa-v-i0", "qa-s-i1")
    assert fl["pool_of"][MILES_SID] == miles
    rows = {x["learner"]: [(p["name"], p["source"], p["samplers"]) for p in x["pools"]] for x in fl["islands"]}
    assert rows == {0: [(v0, "external", ["qa-v-i0-0"])],
                    1: [(miles, "miles", [MILES_SID]), (s1, "external", ["qa-s-i1-0"])]}

    d = _diloco(r)
    assert d["kind"] == "synced_group" and d["follow"] and d["publisher"] is None
    assert sorted(d["lanes"]["1"]["samplers"]) == [MILES_SID, "qa-s-i1-0"] and d["lanes"]["1"]["substore"] == "trainer1"
    assert d["lanes"]["0"]["samplers"] == ["qa-v-i0-0"]
    node = next(n for n in d["sampler_nodes"] if n["id"] == MILES_SID)
    assert node["trainer"] == "1" and node["island"] == 1 and node["pool"] == miles
    box = {x["island"]: x for x in d["fleet"]["islands"]}
    m, ext = box[1]["pools"]
    assert (m["source"], m["engine"], m["card"], m["launch_island"], m["shard"]) == \
        ("miles", "SGLang", "L40S", 2, "island2")  # Miles rolls out with SGLang, plan or not
    assert m["provider"] == "modal" and m["gpus"] == 1 and m["arch"] is not None
    assert (ext["source"], ext["engine"], ext["card"]) == ("external", "SGLang", "L40S")
    assert box[0]["pools"][0]["source"] == "external" and "launch_island" not in box[0]["pools"][0]


@pytest.mark.parametrize("old_plan", [False, True])
def test_a_miles_start_naming_its_island_is_still_a_miles_sampler(old_plan):
    """A Miles sampler that follows its trainer names its island (learner id) and its store URI
    in rl_sampler_start, as an external pool does. Without a plan (or with one written before
    Miles samplers) it is still a Miles sampler (its start has no sampler_id) in island 1."""
    recs = records()
    for rec in recs:
        if rec.get("event") == "rl_sampler_start" and rec.get("island_id") == int(MILES_SID):
            rec.update(island=1, store=f"{STORE}/trainer1")
    r = Reducer(run="mixed-qa")
    for rec in recs:
        r.feed(rec)
    if old_plan:
        doc = plan_doc()
        for isl in doc["islands"]:
            isl.pop("miles_samplers")
        assert r.load_fleet_plan(doc, "old")
    fl = fleet_islands.resolve(r)
    assert fleet_islands.is_miles_sampler(MILES_SID, {"island": 1, "store": f"{STORE}/trainer1"}, None)
    assert not fleet_islands.is_miles_sampler("qa-s-i1-0", {"sampler_id": "qa-s-i1-0", "island": 1}, None)
    assert fl["island_of"][MILES_SID] == 1 and fl["island_of"]["qa-s-i1-0"] == 1
    assert fl["pool_of"][MILES_SID] == "island2"
    rows = {x["learner"]: [(p["name"], p["source"], p["samplers"]) for p in x["pools"]] for x in fl["islands"]}
    assert rows[1][0] == ("island2", "miles", [MILES_SID])
    assert [row[1] for row in rows[1][1:]] == ["external"] and rows[0][0][1] == "external"


def test_miles_sampler_does_not_need_the_plan_to_draw_before_it_writes():
    """The plan lists the Miles sampler: its row is drawn (not started) before its tape arrives."""
    r = Reducer(run="mixed-qa")
    for rec in records():
        if rec.get("island_id") != 2:
            r.feed(rec)
    r.load_fleet_plan(plan_doc(), "test")
    box = {x["island"]: x for x in _diloco(r)["fleet"]["islands"]}
    m = box[1]["pools"][0]
    assert (m["name"], m["source"], m["samplers"], m["card"], m["engine"]) == \
        ("i1-miles-l40s", "miles", [], "L40S", "SGLang")


def test_an_old_plan_without_miles_samplers_binds_by_the_store():
    """A plan written before Miles samplers (no miles_samplers rows): the Miles sampler island
    still lands in island 1 by the trainer<k> its store ends in, as its own Miles row."""
    doc = plan_doc()
    for isl in doc["islands"]:
        isl.pop("miles_samplers")
        isl.pop("trainer_usd_per_h")
    r = reducer(plan=False)
    assert r.load_fleet_plan(doc, "old")
    fl = fleet_islands.resolve(r)
    assert fl["island_of"][MILES_SID] == 1
    assert [(p["name"], p["source"]) for p in fl["islands"][1]["pools"]] == [("island2", "miles"),
                                                                            ("i1-sgl-l40s", "external")]


# -- the per-island async GRPO panel ------------------------------------------------------------
def test_island_panel_lists_the_miles_sampler_its_loads_and_its_own_mismatch():
    asy = reducer().page_view(now=T0 + 1000)["async"]
    blocks = {b["island"]: b for b in asy["islands"]}
    m, ext = blocks[1]["pools"]
    assert (m["name"], m["source"], m["samplers"], m["card"], m["gpus"], m["provider"], m["launch_island"]) == \
        ("i1-miles-l40s", "miles", [MILES_SID], "L40S", 1, "modal", 2)
    assert m["groups"] and m["policy_version"] == 4 and ext["source"] == "external"
    assert blocks[0]["pools"][0]["source"] == "external"
    # its loads sit in island 1's release rows, next to the pool's, never in island 0's
    for row in blocks[1]["releases"]:
        assert sorted(row["loads"]) == [MILES_SID, "qa-s-i1-0"]
    assert all(MILES_SID not in row["loads"] for row in blocks[0]["releases"])
    # island 1's backlog counts what both of its sources wrote
    written = sum(_samples(1, r)[j] // 2 for r in range(ROUNDS) for j in (0, 1))
    assert blocks[1]["backlog"]["written"] == written
    # the Miles sampler is a source of its own in the per-pool mismatch split
    for k, rows in SAMPLERS.items():
        mm = {p["name"]: p["mismatch"] for p in blocks[k]["pools"]}
        for _kind, name, _sid, _eng, _card, diff, clip in rows:
            assert mm[name]["logprob_diff"]["value"] == pytest.approx(diff, abs=1e-6), name
            assert mm[name]["tis_clip"]["value"] == pytest.approx(clip, abs=1e-6), name
    assert blocks[1]["pools"][0]["mismatch"]["logprob_diff"]["method"] == "single-pool rounds"


# -- the plan run up compiles from the mixed example ---------------------------------------------
def test_plan_compiled_from_the_mixed_example_binds_both_kinds():
    from yeto.rl.fleet import compile_fleet

    cfg = json.loads((Path(__file__).parents[1] / "examples" / "fleet" / "run.mixed-miles.json").read_text())
    _, plan = compile_fleet(cfg)
    doc = json.loads(plan.dumps())
    r = Reducer()
    for k in (0, 1):
        r.feed({"event": "rl_engine_selected", "island_id": k, "time_unix": T0, "rl/outer_sync": True})
        r.feed({"event": "rl_round_trained", "island_id": k, "time_unix": T0 + 9, "rollout_id": 0, "trained_groups": 4})
    for k, sid in ((0, "<run>-vllm-i0-1"), (1, "<run>-sglang-i1-0")):
        r.feed({"event": "rl_sampler_start", "sampler_id": sid, "time_unix": T0 + 1, "island": k,
                "store": f"s3://<bucket>/runs/<run>/trainer{k}", "provenance": {"engine": "x", "card_model": "H100"}})
        r.feed({"event": "rl_sampler_round", "sampler_id": sid, "time_unix": T0 + 5, "rollout_id": 0, "groups": 2})
    launch = doc["islands"][1]["miles_samplers"][0]["launch_island"]
    r.feed({"event": "rl_sampler_start", "island_id": launch, "time_unix": T0 + 2,
            "store": "/mnt/yeto/runs/<run>/trainer1", "num_rollouts": None, "until_trainers_end": 1})
    r.feed({"event": "rl_sampler_round", "island_id": launch, "time_unix": T0 + 6, "rollout_id": 0, "groups": 2})
    assert r.load_fleet_plan(doc, "test")
    fl = _diloco(r)["fleet"]
    rows = {p["name"]: (x["island"], p["source"], p["samplers"]) for x in fl["islands"] for p in x["pools"]}
    assert rows == {"i0-vllm-h100": (0, "external", ["<run>-vllm-i0-1"]),
                    "i1-miles-l40s": (1, "miles", [str(launch)]),
                    "i1-sglang-l40s": (1, "external", ["<run>-sglang-i1-0"])}
    m = fl["islands"][1]["pools"][0]
    assert (m["engine"], m["card"], m["provider"], m["shard"]) == ("SGLang", "L40S", "modal", f"island{launch}")
    assert doc["islands"][1]["buffer_shards"][0] == m["shard"]  # the shard the trainer reads


# -- a fleet without Miles samplers keeps its view -------------------------------------------------
def test_a_fleet_of_pools_alone_has_no_source_keys():
    from test_dashboard_fleet_islands import fleet_reducer

    view = export_view(fleet_reducer(), generated_at=0.0)["page"]
    text = json.dumps(view)
    for key in ('"source": "external"', '"source": "miles"', '"launch_island"'):
        assert key not in text, key
    fl = view["topology"]["diloco"]["fleet"]
    assert all(set(p) == {"name", "source_name", "provider", "engine", "precision", "card", "arch", "topology",
                          "samplers"} for x in fl["islands"] for p in x["pools"])


# -- the page ---------------------------------------------------------------------------------------
def _page(tmp_path, r: Reducer, width: int) -> dict:
    from test_dashboard_diloco import SMOKE

    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    page = export_html(r, tmp_path / "p.html", generated_at=r.max_ts)
    script = tmp_path / "smoke.js"
    marker = "legend: mk(\"topoLg\").innerHTML"
    assert marker in SMOKE
    script.write_text(SMOKE.replace(marker, marker + ", fleet: mk(\"aFleet\").innerHTML"))
    res = subprocess.run([node, str(script), str(page), str(width)], capture_output=True, text=True, timeout=60,
                         check=False)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


@pytest.mark.parametrize("width", [1200, 400])
def test_page_tags_each_sampler_with_its_source(tmp_path, width):
    out = _page(tmp_path, reducer(), width)
    joined = " | ".join(out["texts"])
    assert out["viewBox"].startswith("0 0 1480 " if width > 640 else "0 0 400 ")
    for want in ("Island 0", "Island 1", "trainer1/", "MILES NATIVE", "EXTERNAL POOL", "i1-miles-l40s",
                 "i1-sgl-l40s · modal", "L40S · SGLang", "L40S · BF16 · SGLang",
                 "1 Miles" if width < 640 else "1 Miles sampler(s) + 1 external pool(s)"):
        assert want in joined, want
    assert joined.count("MILES NATIVE") == 1 and joined.count("EXTERNAL POOL") == 2
    # a fleet box has no machine labels: its pool and trainer cards carry the machine
    assert "m0 · miles-l40s" not in joined and "p1 · sgl-l40s" not in joined
    assert "n/a" not in joined, [t for t in out["texts"] if "n/a" in t]
    # the island panel's sampler table: both sources tagged, the Miles row with its cards
    assert "Miles native</span>" in out["fleet"] and out["fleet"].count("external pool</span>") == 2
    assert "L40S × 1 · modal" in out["fleet"] and "launch island 2" in out["fleet"]
    assert "Miles sampler island 2 loaded (latency)" in out["fleet"] and "sampler qa-s-i1-0 loaded" in out["fleet"]
    assert "n/a" not in out["fleet"]


def test_page_of_a_fleet_of_pools_alone_has_no_source_tags(tmp_path):
    from test_dashboard_fleet_islands import fleet_reducer

    out = _page(tmp_path, fleet_reducer(), 1200)
    joined = " | ".join(out["texts"])
    assert "MILES NATIVE" not in joined and "EXTERNAL POOL" not in joined and "vllm-h100" in joined
    assert 'class="tag"' not in out["fleet"]


def test_a_miles_sampler_tape_under_its_trainer_lane_binds_to_its_launch_island():
    """A Miles sampler island runs as learner 0 of 1 and uploads its tape under its trainer's
    lane (store/trainer1:trainer/l0-…) with island_id 0: its records are launch island 2's,
    never trainer 0's (trainer 0 stays a trainer, island 2 a sampler)."""
    r = Reducer()
    assert r.load_fleet_plan(plan_doc(), "plan")
    r.feed({"event": "rl_round_trained", "island_id": 0, "time_unix": T0 + 20, "round": 0, "groups": 4},
           source="store:trainer/l0-modal-20261124T000000Z")
    lane = "store/trainer1:trainer/l0-modal-20261124T000001Z"
    r.feed({"event": "rl_sampler_start", "island_id": 0, "time_unix": T0 + 21, "weights": "full",
            "store": "~/yeto-checkpoint-store/runs/mixed-qa/trainer1", "until_trainers_end": 1}, source=lane)
    r.feed({"event": "rl_sampler_round", "island_id": 0, "time_unix": T0 + 60, "groups": 4, "policy_version": 1,
            "generate_s": 30.0}, source=lane)
    assert r.async_.role("0") == "trainer"
    assert r.async_.role(MILES_SID) == "sampler"


def test_a_lane_tape_without_a_plan_keeps_its_records_island():
    r = Reducer()
    r.feed({"event": "rl_sampler_round", "island_id": 0, "time_unix": T0 + 60, "groups": 4, "policy_version": 1,
            "generate_s": 30.0}, source="store/trainer1:trainer/l0-modal-x")
    assert r.async_.role("0") == "sampler"
