"""A fleet whose trainers never trained (each kept launching or relaunching) and a pool whose
machines were all released: the step, the local steps, the policy ages and the read position say
"not trained yet" / "not read yet", never a bare "n/a" or a lone dash, and the pool card reads
released, as the pool state and the cost panel do. A run with a syncer but no fleet and a run with
one trainer render as before."""

from __future__ import annotations

import json
import re

import test_dashboard_fleet_na as fleet_na
from test_dashboard_diloco import _smoke
from test_dashboard_failed_island_na import never_trained  # noqa: F401 (fixture)
from test_dashboard_fleet_islands import fleet_records, fleet_reducer, plan_doc
from test_dashboard_trainer_islands import _store

from yeto.dashboard import fleet_islands
from yeto.dashboard.reducer import Reducer
from yeto.dashboard.topology_diloco import _fleet_part
from yeto.rl.engine.object_store import DirectoryObjectStore

# what a trainer writes once it trains (and what follows from it: releases, loads, merges)
TRAINED = {"rl_round_trained", "rl_buffer_stale_skipped", "rl_driver_phase", "rl_policy_snapshot", "rl_policy_released",
           "rl_sampler_policy_loaded", "rl_fragment_push", "rl_fragment_bcast", "rl_trajectory_reward"}


def _never_trained() -> Reducer:
    """The two-island fleet fixture with every trainer's training cut: the samplers wrote their
    rounds, no trainer stepped (no syncer merge either)."""
    r = Reducer(run="fleet-qa")
    for rec in fleet_records():
        if rec.get("event") in TRAINED or "event" not in rec:
            continue
        r.feed(rec)
    assert r.load_fleet_plan(plan_doc(), "test")
    return r


def _text(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]*>", " ", html))


def _page(tmp_path, r: Reducer, pick: str | None = None) -> dict:
    """The fleet n/a smoke page, with the backlog legend (``legBacklog``) and the release
    timeline's texts (``relTexts``) as well."""
    smoke = fleet_na.SMOKE.replace('"evs","rNodes"]', '"evs","rNodes","legBacklog"]').replace(
        "process.stdout.write(", "const walk=(n)=>[n.textContent].concat(...n.children.map(walk));"
        "out.relTexts=walk(mk('aRelChart')).filter(Boolean);process.stdout.write(")
    assert smoke != fleet_na.SMOKE
    orig, fleet_na.SMOKE = fleet_na.SMOKE, smoke
    try:
        return fleet_na._page(tmp_path, r, pick=pick)
    finally:
        fleet_na.SMOKE = orig


def test_trainers_that_never_trained_are_drawn_not_trained_yet(tmp_path):
    out = _smoke(tmp_path, _never_trained())
    texts = out["texts"]
    assert texts.count("not trained yet") == 2  # one big line per trainer card
    assert "no snapshot yet" in texts and not any(re.search(r"\bn/a\b", t) for t in texts)
    matrix = _text(out["matrix"])
    assert len(re.findall(r"Trainer [01]\s+H100 · not trained yet", matrix)) == 2 and "step —" not in matrix
    assert "trainer 0 not trained yet · trainer 1 not trained yet" in out["kpis"] and "local —" not in out["kpis"]


def test_a_syncer_run_without_a_fleet_keeps_its_dashes(tmp_path, never_trained):
    """Trainer islands without a fleet (a synced group launched as one run): a trainer that never
    stepped keeps "step —" on its card, its Fragments row and the local-steps KPI."""
    out = _smoke(tmp_path, never_trained)
    assert "step —" in out["texts"] and "not trained yet" not in out["texts"]
    assert "trainer 1 local —" in out["kpis"] and "not trained yet" not in out["kpis"]
    assert "not trained yet" not in out["matrix"]


def test_an_island_that_never_trained_names_why_its_counts_and_read_position_are_missing(tmp_path):
    for pick in ("i0", "i1"):
        (tmp_path / pick).mkdir()
        out = _page(tmp_path / pick, _never_trained(), pick=pick)
        chips = _text(out["aTrainer"])
        assert "not trained yet: trained groups, policy ages, dropped as stale" in chips, pick
        assert "policy ages n/a" not in chips and "not reported:" not in chips
        legend = out["legBacklog"]
        assert re.search(r"backlog \(written \d+ · not read yet\)", legend) and "read position n/a" not in legend
        for key in ("aTrainer", "legBacklog", "rKpis"):
            assert not re.search(r"\bn/a\b", _text(out[key])), (pick, key)
    kp = _page(tmp_path, _never_trained())["rKpis"]
    assert re.search(r'<div class="kt">[^<]*step p50</div><div class="kv[^"]*"[^>]*>not trained yet</div>', kp)


def test_an_island_that_trained_keeps_its_policy_ages_and_read_position(tmp_path):
    out = _page(tmp_path, fleet_reducer(), pick="i0")
    assert "policy ages <b>0:" in out["aTrainer"] and "not trained yet" not in out["aTrainer"]
    assert re.search(r"read position \d+", out["legBacklog"]) and "not read yet" not in out["legBacklog"]


def test_a_single_trainer_that_never_trained_keeps_na(tmp_path):
    """No fleet: the trainer chips, the backlog legend and a one-instant release timeline render
    as before ("n/a", three time labels)."""
    t0 = 1_000_000.0
    r = Reducer(run="one")
    r.feed({"event": "rl_driver_start", "island_id": 0, "time_unix": t0, "placement": "colocated"})
    for i in range(3):
        r.feed({"event": "rl_sampler_round", "sampler_id": "s0", "rollout_id": i, "groups": 8, "policy_version": 0,
                "time_unix": t0 + 30 + 60 * i})
    r.feed({"event": "rl_policy_released", "island_id": 0, "time_unix": t0 + 200, "policy_version": 1, "local_version": 1,
            "released_at": t0 + 200, "origin": {"kind": "local"}})
    out = _page(tmp_path, r)
    assert "no trained rounds" in out["aTrainer"] and "policy ages <b>n/a</b>" in out["aTrainer"]
    assert "not trained yet" not in out["aTrainer"] and "n/a: trained groups" in out["aTrainer"]
    assert "read position n/a" in out["legBacklog"] and "not read yet" not in out["legBacklog"]
    assert len([x for x in out["relTexts"] if re.fullmatch(r"\d\d:\d\d:\d\d", x)]) == 3
    # the run KPIs: a step time no trainer reported is not called "not trained yet" without a fleet
    assert not re.search(r'step p50</div><div class="kv[^"]*"[^>]*>not trained yet<', out["rKpis"])


def _drawn_pools(tmp_path, members: list[dict], *, stopped: bool = True) -> tuple[dict, Reducer]:
    root = _store(tmp_path)
    doc = {"members": members, **({"stopped": "pool-stop requested"} if stopped else {})}
    (root / "fleet/pools/i1-a/state.json").write_text(json.dumps(doc))
    r = fleet_reducer()
    r.load_fleet_state(fleet_islands.read_fleet_state(DirectoryObjectStore(str(root))))
    drawn = {i["island"]: i for i in _fleet_part(fleet_islands.resolve(r), r.async_)["islands"]}
    return {p["name"]: p for p in drawn[1]["pools"]}, r


GONE = {"shard": "qa-a-i1-0", "state": "gone", "granted": 1.0, "charged_to": 2.0}


def test_a_stopped_pool_whose_machines_are_all_released_reads_released(tmp_path):
    pools, r = _drawn_pools(tmp_path, [GONE])
    assert pools["i1-a"]["released"] is True and pools["i1-a"]["stopped"] == "pool-stop requested"
    texts = _smoke(tmp_path, r)["texts"]
    assert "released" in texts and "stopped" not in texts  # its card (a pool fleet apply added, no sampler of its own)


def test_a_stopped_pool_still_holding_a_machine_is_not_released(tmp_path):
    pools, r = _drawn_pools(tmp_path, [{"shard": "qa-a-i1-0", "state": "ready", "granted": 1.0, "charged_to": None}])
    assert "released" not in pools["i1-a"] and "machines" not in pools["i1-a"]
    texts = _smoke(tmp_path, r)["texts"]
    assert "stopped" in texts and "released" not in texts
    # a pool state without members (older stores): no claim either way
    pools, _ = _drawn_pools(tmp_path / "b", [])
    assert "released" not in pools["i1-a"] and "machines" not in pools["i1-a"]


def test_a_running_pool_that_lost_its_machine_reads_replacing_not_released(tmp_path):
    """Its only spot machine preempted (gone), the pool not stopped: the controller orders a new
    one on its next tick, so the card reads replacing, never released."""
    pools, r = _drawn_pools(tmp_path, [GONE], stopped=False)
    assert "released" not in pools["i1-a"] and pools["i1-a"]["machines"] == "replacing"
    texts = _smoke(tmp_path, r)["texts"]
    assert "replacing" in texts and "released" not in texts and "stopped" not in texts


def test_a_running_pool_whose_replacement_is_not_granted_waits_for_capacity(tmp_path):
    ordered = {"shard": "qa-a-i1-1", "state": "provisioning", "granted": None, "charged_to": None}
    pools, r = _drawn_pools(tmp_path, [GONE, ordered], stopped=False)
    assert "released" not in pools["i1-a"] and pools["i1-a"]["machines"] == "waiting for capacity"
    texts = _smoke(tmp_path, r)["texts"]
    assert "waiting for capacity" in texts and "released" not in texts
    # the replacement granted: its samplers' tapes say how it runs
    granted = dict(ordered, state="starting", granted=3.0)
    pools, _ = _drawn_pools(tmp_path / "b", [GONE, granted], stopped=False)
    assert "machines" not in pools["i1-a"] and "released" not in pools["i1-a"]
    # a pool that never lost a machine, its first one still ordered: nothing claimed either
    pools, _ = _drawn_pools(tmp_path / "c", [ordered], stopped=False)
    assert "machines" not in pools["i1-a"] and "released" not in pools["i1-a"]


def test_a_release_timeline_of_one_instant_has_one_time_label(tmp_path):
    """A fleet island whose timeline holds one instant (its single release, as a trainer that
    failed right after releasing the base weights): one time label, not three drawn over each other."""
    r = _never_trained()
    t = r.max_ts
    r.feed({"event": "rl_policy_released", "island_id": 0, "time_unix": t, "policy_version": 1, "local_version": 1,
            "released_at": t, "origin": {"kind": "local"}})
    texts = _page(tmp_path, r, pick="i0")["relTexts"]
    clocks = [x for x in texts if re.fullmatch(r"\d\d:\d\d:\d\d", x)]
    assert len(clocks) == 1, texts
    # two instants: the three labels as before
    r.feed({"event": "rl_policy_released", "island_id": 0, "time_unix": t + 120, "policy_version": 2,
            "local_version": 2, "released_at": t + 120, "origin": {"kind": "local"}})
    (tmp_path / "b").mkdir()
    assert len([x for x in _page(tmp_path / "b", r, pick="i0")["relTexts"] if re.fullmatch(r"\d\d:\d\d:\d\d", x)]) == 3


def _trained_without(*fields: str) -> Reducer:
    """The two-island fleet fixture, its trainer rounds without ``fields``."""
    r = Reducer(run="fleet-qa")
    for rec in fleet_records():
        if rec.get("event") == "rl_round_trained":
            rec = {k: v for k, v in rec.items() if k not in fields}
        r.feed(rec)
    assert r.load_fleet_plan(plan_doc(), "test")
    return r


def test_an_island_that_trained_without_policy_ages_says_they_are_not_on_the_tapes(tmp_path):
    out = _page(tmp_path, _trained_without("policy_age_groups"), pick="i0")
    chips = _text(out["aTrainer"])
    assert "policy ages not on the tapes" in chips and "not trained yet" not in chips


def test_trainer_rounds_without_a_trained_group_count_are_not_called_not_trained(tmp_path):
    """Rounds on the table below but no trained_groups or policy ages on them (an older tape):
    the chips name what is not reported, never "not trained yet" above the trained rounds."""
    out = _page(tmp_path, _trained_without("policy_age_groups", "trained_groups"), pick="i0")
    chips = _text(out["aTrainer"])
    assert "not reported: trained groups" in chips and "policy ages not on the tapes" in chips
    assert "not trained yet" not in chips
