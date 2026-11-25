"""A fleet launched island by island (``fleet.launch: per_island``) on the dashboard: the trainer
islands of the desired state (added, moved, removed while the fleet trains) and what their records
under ``fleet/trainers/<k>/`` say they are (stopped, removed, failed, abandoned, moving), and a
syncer whose tape this host does not see."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_dashboard_diloco import D as DILOCO
from test_dashboard_diloco import _smoke, diloco, lines, load
from test_dashboard_fleet_islands import T0, fleet_reducer, plan_doc

from yeto.dashboard import cost, fleet_islands
from yeto.dashboard.reducer import Reducer
from yeto.dashboard.topology_diloco import _fleet_part
from yeto.rl.engine.object_store import DirectoryObjectStore

MODAL, AWS = "modal:1x1xh100@us", "aws:1xh100@us-west-2"


def _trainer(name: str, k: int, gpu: str, usd: float) -> dict:
    return {"kind": "trainer", "name": name, "learner_id": k, "gpu": gpu, "usd_per_h": usd,
            "prefix": f"fleet-qa-{name}", "pools": [], "argv": []}


def _pool(name: str, k: int, base: str) -> dict:
    return {"kind": "sampler_pool", "name": name,
            "pool": {"provider": "modal", "shard_base": base, "usd_per_h": 1.2, "run_prefix": f"fleet-qa/trainer{k}",
                     "sampler": {"island": k}}}


def _store(tmp_path: Path) -> Path:
    """Seeded with islands i0, i1; ``fleet island add i2`` (with its pool), ``move i0`` to AWS,
    ``remove i2``. i0 was given up on AWS, i1 stopped by the starvation breaker (its pool stopped,
    its machine released), i2 removed."""
    root = tmp_path / "store"

    def w(rel: str, doc: dict) -> None:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(json.dumps(doc))

    base = [_pool("i0-v", 0, "qa-v-i0"), _pool("i1-a", 1, "qa-a-i1")]
    gens = [base + [_trainer("i0", 0, MODAL, 11.7), _trainer("i1", 1, MODAL, 11.7)],
            base + [_trainer("i0", 0, MODAL, 11.7), _trainer("i1", 1, MODAL, 11.7), _trainer("i2", 2, MODAL, 11.7),
                    _pool("i2-x", 2, "qa-x-i2")],
            base + [_trainer("i0", 0, AWS, 6.88), _trainer("i1", 1, MODAL, 11.7), _trainer("i2", 2, MODAL, 11.7),
                    _pool("i2-x", 2, "qa-x-i2")],
            base + [_trainer("i0", 0, AWS, 6.88), _trainer("i1", 1, MODAL, 11.7)]]
    for n, members in enumerate(gens, 1):
        w(f"fleet/desired/{n:08d}.json", {"generation": n, "updated_at": T0 + 100 * n, "members": members})
        w(f"fleet/applied/{n:08d}.json", {"seq": n, "at": T0 + 100 * n, "members": members})
    w("fleet/trainers/0/launch.json", {"state": "abandoned", "gpu": AWS, "usd_per_h": 6.88, "incarnation": 4,
                                       "relaunches": 3, "prefix_now": "fleet-qa-i0-m1", "why": "moved",
                                       "decision": {"action": "relaunch", "cause": "infra", "detail": "FAILED"},
                                       "ended": {"detail": "FAILED", "exit_code": 1}})
    w("fleet/trainers/0/state.json", {"state": "ordering", "reason": "ordering", "since": T0 + 310, "incarnation": 4})
    w("fleet/trainers/1/launch.json", {"state": "done", "gpu": MODAL, "usd_per_h": 11.7, "incarnation": 0})
    w("fleet/trainers/1/state.json", {"state": "done", "since": T0 + 450, "exit": "STOPPED (island STOP: breaker:run.starvation)",
                                      "decision": {"action": "stop", "cause": "stopped",
                                                   "detail": "breaker:run.starvation", "state": "done"}})
    w("fleet/trainers/1/STOP", {"at": T0 + 440, "by": "yeto run up", "reason": "breaker:run.starvation"})
    w("fleet/trainers/2/launch.json", {"state": "down", "gpu": MODAL, "usd_per_h": 11.7, "why": "added"})
    w("fleet/trainers/2/state.json", {"state": "done", "since": T0 + 420, "decision": {
        "action": "stop", "cause": "stopped", "detail": "removed from the fleet's desired state", "state": "done"}})
    w("fleet/trainers/2/STOP", {"at": T0 + 410, "by": "trainer worker", "reason": "removed from the fleet's desired state"})
    w("fleet/pools/i1-a/state.json", {"stopped": "pool-stop requested",
                                      "members": [{"shard": "qa-a-i1-0", "state": "gone", "charged_to": T0 + 5000}]})
    return root


def _state(tmp_path: Path) -> dict:
    return fleet_islands.read_fleet_state(DirectoryObjectStore(str(_store(tmp_path))))


def test_the_desired_state_and_the_records_name_each_islands_state(tmp_path):
    st = _state(tmp_path)
    tr = st["trainers"]
    assert sorted(tr) == [0, 1, 2]
    assert tr[0]["moves"] == [{"at": T0 + 300, "gpu": AWS, "from": MODAL}] and tr[0]["usd_per_h"] == 6.88
    assert tr[2]["removed_at"] == T0 + 400 and tr[2]["name"] == "i2"
    assert tr[0]["now"]["state"] == "abandoned" and "after 3 relaunch" in tr[0]["now"]["reason"]
    assert "last exit 1: FAILED" in tr[0]["now"]["detail"] and "still billed" in tr[0]["now"]["detail"]
    assert tr[1]["now"] == {"state": "stopped", "reason": "breaker:run.starvation",
                            "detail": "STOPPED (island STOP: breaker:run.starvation)"}
    assert tr[2]["now"]["state"] == "removed"
    # the stopped pool's released machine: its sampler ended
    assert st["shards"]["qa-a-i1-0"]["stopped"] == "pool-stop requested"


@pytest.mark.parametrize("rec, desired, state", [
    ({"launch_state": "running", "life_state": "recovering", "relaunches": 1, "incarnation": 1}, {}, "starting"),
    ({"launch_state": "moving", "life_state": "ordering", "gpu": AWS}, {}, "moving"),
    ({"launch_state": "running", "life_state": "ordering"}, {"moves": [{"gpu": AWS}], "gpu": AWS}, "moving"),
    ({"launch_state": "running", "life_state": "running"}, {}, "running"),
    ({"launch_state": "failed", "life_state": "failed", "exit": "strict"}, {}, "failed"),
    ({"launch_state": "down", "life_state": "done"}, {}, "down"),
    ({"launch_state": "removed", "life_state": "draining"}, {}, "removed"),
])
def test_island_state(rec, desired, state):
    assert fleet_islands.island_state(rec, desired)["state"] == state
    assert fleet_islands.island_state(None) is None


def test_trainer_cards_follow_their_island_records(tmp_path):
    r = fleet_reducer()
    now = r.max_ts + 30
    before = {c["id"]: c["status"] for c in r.overview(now=now)["islands"]}
    assert before["0"] not in ("lost", "stopped") and before["1"] not in ("lost", "stopped")
    assert r.load_fleet_state(_state(tmp_path))
    cards = {c["id"]: c for c in r.overview(now=now)["islands"]}
    assert cards["0"]["status"] == "lost" and cards["0"]["left"]["reason"].startswith("abandoned")
    assert cards["0"]["island_now"]["gpu"] == AWS and cards["0"]["island_now"]["moves"][0]["gpu"] == AWS
    assert cards["1"]["status"] == "stopped" and cards["1"]["island_now"]["reason"] == "breaker:run.starvation"
    # the stopped pool's sampler: left (its machine released), not stale
    assert cards["qa-a-i1-0"]["status"] == "left" and cards["qa-a-i1-0"]["left"]["reason"] == "pool stopped"


def test_an_added_and_removed_island_is_drawn_with_its_pool(tmp_path):
    r = fleet_reducer()
    r.load_fleet_state(_state(tmp_path))
    fleet = fleet_islands.resolve(r)
    by = {i["learner"]: i for i in fleet["islands"]}
    assert by[2]["name"] == "i2" and by[2]["added"] and by[2]["now"]["state"] == "removed"
    assert [p["name"] for p in by[2]["pools"]] == ["i2-x"] and by[2]["pools"][0]["removed"]["at"] == T0 + 400
    assert by[0]["desired"]["moves"][-1]["gpu"] == AWS and by[0]["now"]["state"] == "abandoned"
    assert next(p for p in by[1]["pools"] if p["name"] == "i1-a")["stopped"] == "pool-stop requested"
    drawn = {i["island"]: i for i in _fleet_part(fleet, r.async_)["islands"]}
    assert drawn[2]["now"]["state"] == "removed" and drawn[2]["added"]
    assert drawn[0]["desired"]["moves"][-1]["gpu"] == AWS
    assert next(p for p in drawn[1]["pools"] if p["name"] == "i1-a")["stopped"] == "pool-stop requested"


def test_without_island_records_the_tapes_decide(tmp_path):
    (tmp_path / "s" / "fleet" / "applied").mkdir(parents=True)
    (tmp_path / "s" / "fleet" / "applied" / "00000001.json").write_text(json.dumps(
        {"seq": 1, "at": T0, "members": [_pool("i0-v", 0, "qa-v-i0")]}))
    st = fleet_islands.read_fleet_state(DirectoryObjectStore(str(tmp_path / "s")))
    assert st["trainers"] == {}
    r = fleet_reducer()
    now = r.max_ts + 30
    want = {c["id"]: c["status"] for c in r.overview(now=now)["islands"]}
    r.load_fleet_state(st)
    assert {c["id"]: c["status"] for c in r.overview(now=now)["islands"]} == want


def test_a_syncer_tape_not_seen_here_reads_the_islands_versions():
    """Run B's islands before their syncer tape arrives: no syncer record here, so each fragment's
    version is at least the newest an island applied (not drawn as 0); once the tape arrives, its."""
    r = load("run-b", skip=lambda p: p.name == "yeto-tape.jsonl")
    d = diloco(r)
    assert d["syncer_tape"] is False and d["versions"] == [0, 0, 0, 0]
    want = [max(t["versions"][f] for t in d["trainers"] if t["versions"]) for f in range(4)]
    assert d["inferred_versions"] == want and max(want) > 0
    for off, rec in enumerate(lines(DILOCO / "run-b" / "yeto-tape.jsonl")):
        r.feed(rec, source="late/yeto-tape.jsonl", offset=off)
    d = diloco(r)
    assert d["syncer_tape"] is True and d["inferred_versions"] is None and d["versions"] == [21, 22, 23, 24]


def _open_at_the_end(r):
    """Island 0 pushes step 25 (fragment 0) 30 s before the trainers' last event; no merge follows."""
    end = max(x["last_event_ts"] for x in r.islands.values() if x.get("last_event_ts") is not None)
    r.feed({"event": "rl_fragment_push", "island_id": 0, "global_step": 25, "fragment_id": 0, "base_version": 21,
            "round_attempt": 1, "payload_bytes": 1, "time_unix": end - 30}, source="late/push.jsonl", offset=0)
    return end


def test_a_round_open_when_the_run_ended_stops_its_clock(tmp_path):
    r = load("run-b")
    end = _open_at_the_end(r)
    d = diloco(r, now=end + 7200)
    assert d["done"]
    (rd,) = [x for x in d["rounds"] if x["step"] == 25]
    assert rd["ended"] == "run ended" and rd["age_s"] == pytest.approx(30.0, abs=1.0) and not rd["blocked"]
    out = _smoke(tmp_path, r)
    joined = " | ".join(out["texts"])
    assert "1 unmerged at the end" in joined and "in flight 1" not in joined
    assert "open 30s when the run ended" in joined and "PUSH 1/2 t25 · unmerged at the end" in out["matrix"]
    assert "Fragment rounds in flight / BCAST pending" in out["kpis"] and "1 unmerged at the end" in out["kpis"]


def test_a_round_open_while_no_trainer_is_up_is_stale(tmp_path):
    """Run B as a live page saw it halfway, read two hours later: no trainer is up and none
    finished; the open rounds are stale (their clock stopped at the last event), not waiting."""
    from test_dashboard_diloco import _replayed

    full = load("run-b")
    ts = [x for x in full.islands.values() if x.get("first_ts") is not None]
    mid = (min(x["first_ts"] for x in ts) + max(x["last_event_ts"] for x in ts)) / 2
    r = _replayed("run-b", mid)
    live = diloco(r, now=mid)
    assert live["rounds"] and not any(x.get("ended") for x in live["rounds"]) and live["shards"][0]["state"] != "stale"
    d = diloco(r, now=mid + 7200)
    assert not d["done"] and d["shards"][0]["state"] == "stale"
    for x, y in zip(d["rounds"], live["rounds"]):
        assert x["ended"] == "no trainer up" and not x["blocked"] and x["age_s"] <= y["age_s"] + 1


def test_without_the_syncer_tape_a_push_an_island_applied_is_not_open():
    """No syncer record here: every PUSH looks unmerged; one whose step an island already applied
    (its fragment's inferred version) was merged, so it is no round in flight."""
    r = load("run-b", skip=lambda p: p.name == "yeto-tape.jsonl")
    d = diloco(r)
    assert d["syncer_tape"] is False and d["inferred_versions"] == [21, 22, 23, 24]
    assert d["rounds"] == []
    end = _open_at_the_end(r)
    (rd,) = diloco(r, now=end)["rounds"]
    assert rd["step"] == 25 and rd["ended"] == "run ended"


def _cost(tmp_path: Path, *, up: dict | None = None, pools: dict | None = None, spend: dict | None = None,
          later: float = 30.0, extra: tuple = ()):
    root = _store(tmp_path)
    for name, members in (pools or {}).items():
        (root / f"fleet/pools/{name}").mkdir(parents=True, exist_ok=True)
        (root / f"fleet/pools/{name}/state.json").write_text(json.dumps({"members": members}))
    r = fleet_reducer()
    for rec in extra:
        r.feed(rec)
    now = r.max_ts + later
    if up is not None:
        (tmp_path / "up.json").write_text(json.dumps(up))
        r.up_state = str(tmp_path / "up.json")
    r.load_fleet_state(fleet_islands.read_fleet_state(DirectoryObjectStore(str(root))))
    if spend is not None:
        r.trainer_spend = spend
    view = cost.cost_view(r, now)
    return {row["id"]: row for row in view["islands"]}, view, now


def test_an_added_islands_pool_bills_its_machines_from_the_pool_state(tmp_path):
    """i2's pool wrote no sampler tape here: its machines are rows, billed from their grant to
    their release as run.budget counts them (never granted: nothing), at the pool's price."""
    rows, view, _ = _cost(tmp_path, pools={"i2-x": [
        {"shard": "qa-x-i2-0", "state": "gone", "granted": T0 + 150, "charged_to": T0 + 3750},
        {"shard": "qa-x-i2-1", "state": "gone", "granted": None, "charged_to": None}]})
    got = rows["qa-x-i2-0"]
    assert got["island"] == 2 and got["pool"] == "i2-x" and got["hours"] == pytest.approx(1.0)
    assert got["cost_usd"] == pytest.approx(1.2) and not got["running"] and "pool state" in got["billed_from"]
    never = rows["qa-x-i2-1"]
    assert never["cost_usd"] == 0.0 and "never granted" in never["billed_from"]
    assert view["fleet"]["pools_usd"] >= 1.2


def test_a_pool_machine_billed_longer_than_its_tapes_bills_the_grant(tmp_path):
    """A sampler whose tape starts after its machine's grant (boot) and a machine still held:
    billed from the grant to now, never less than run.budget."""
    base, _, now = _cost(tmp_path)
    tape_h = base["qa-a-i1-0"]["hours"]
    rows, _, _ = _cost(tmp_path / "b", pools={"i1-a": [
        {"shard": "qa-a-i1-0", "state": "ready", "granted": T0 - 7200, "charged_to": None}]})
    got = rows["qa-a-i1-0"]
    assert got["hours"] == pytest.approx((now - (T0 - 7200)) / 3600) and got["hours"] > tape_h
    assert got["running"] and "more than its sampler tapes" in got["billed_from"]
    assert cost.member_hours({"granted": "bad", "state": "gone", "charged_to": "bad"}, 100.0, 40.0) == pytest.approx(60 / 3600)
    assert cost.member_hours(None, 1.0, 0.0) is None


def test_the_syncer_bills_the_head_vm_until_a_down_confirms_it_released(tmp_path):
    rows, _, now = _cost(tmp_path, up={"trainer_t0": T0 - 600, "ups": [
        {"start": T0 - 900, "end": T0 - 300}, {"start": T0 + 60, "end": None, "pid": 1}]})
    s = rows["syncer"]
    assert s["hours"] == pytest.approx((300 + now - (T0 + 60)) / 3600)
    assert s["running"] and "until yeto run down confirms" in s["billed_from"]
    closed, _, _ = _cost(tmp_path / "b", up={"trainer_t0": T0, "ups": [{"start": T0, "end": T0 + 99999}]})
    assert closed["syncer"]["hours"] == pytest.approx(99999 / 3600)
    assert "until yeto run down" not in closed["syncer"]["billed_from"]


def test_the_total_is_never_below_run_budgets_own_count(tmp_path):
    _, view, _ = _cost(tmp_path)
    assert "run_budget_floor_usd" not in view["fleet"]
    _, raised, _ = _cost(tmp_path / "b", up={"spend_usd_estimate": (view["total_usd"] or 0) + 50.0})
    assert raised["total_usd"] == pytest.approx((view["total_usd"] or 0) + 50.0)
    assert raised["fleet"]["run_budget_floor_usd"] == raised["total_usd"] and "run.budget" in raised["note"]


def test_an_added_islands_unpriced_hours_bill_its_desired_price(tmp_path):
    """The trainer spend reads every island's declared price: an added island's is the
    desired state's (run.budget bills it nothing; the panel never less)."""
    from types import SimpleNamespace

    from yeto.rl import trainer_worker

    r = fleet_reducer()
    r.fleet_plan = dict(r.fleet_plan, launch="per_island")
    r.load_fleet_state(_state(tmp_path))
    reader = SimpleNamespace(objects=object(), clock=lambda: 0.0, interval_s=60.0)
    rows = [(2, 2.0, 0.0, 2.0), (0, 1.0, 6.88, 0.0)]
    orig = trainer_worker.billed
    trainer_worker.billed = lambda _objects, _now: rows
    try:
        assert fleet_islands.PlanFile.pump_trainer_spend(reader, r, now=T0)
    finally:
        trainer_worker.billed = orig
    assert r.trainer_spend["islands"][2] == (2.0, pytest.approx(23.4), True)
    assert r.trainer_spend["islands"][0] == (1.0, pytest.approx(6.88), False)


def test_an_added_islands_lane_tapes_are_read(tmp_path):
    """The plan names the seeded islands; the samplers of i2 (added while the fleet trains)
    upload under trainer2/tapes/, read once the desired state names i2."""
    from yeto.dashboard.reducer import Reducer

    root = _store(tmp_path)
    (root / "fleet" / "plan.json").write_text(json.dumps(plan_doc()))
    tape = root / "trainer2" / "tapes" / "qa-x-i2-0" / "x1" / "000001.jsonl"
    tape.parent.mkdir(parents=True)
    tape.write_text(json.dumps({"event": "rl_sampler_start", "sampler_id": "qa-x-i2-0", "time_unix": T0 + 200,
                                "island": 2, "max_backlog_groups": 8}) + "\n")
    r = Reducer(run="fleet-qa")
    pf = fleet_islands.PlanFile([], DirectoryObjectStore(str(root)))
    pf.pump(r)
    assert pf.pump_tapes(r) > 0
    assert "store/trainer2:qa-x-i2-0/x1" in {x for t in pf.tapes() for x in t.sources}
    assert r.async_.roles.get("qa-x-i2-0") == "sampler"


def test_plan_doc_is_unchanged_by_the_fixture():
    assert [i["name"] for i in plan_doc()["islands"]]  # the shared fleet fixture still loads


# -- REV #838: a single-trainer run we stopped stays "stopped" ------------------------------
def _stopped_single_trainer(tmp_path: Path, *, recovery: bool):
    """One trainer, no fleet (test_dashboard_operator_stop's run): we stop it (STOP_ISSUED),
    it writes RECOVERY_REQUIRED (optionally) and its island_leave STOPPED on the way out."""
    from test_dashboard_operator_stop import STOP_TS, _run

    from yeto.dashboard.sources import load_all

    run, tape = _run(tmp_path, STOP_TS + 4)
    rows = [json.loads(x) for x in (tape / "rl-island-0.jsonl").read_text().splitlines()]
    if not recovery:
        rows = [x for x in rows if x.get("event") != "rl_reconfiguration"]
    rows.append({"event": "island_leave", "island_id": 0, "time_unix": STOP_TS + 8, "reason_name": "STOPPED"})
    (tape / "rl-island-0.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    r = Reducer()
    load_all(r, [str(tape)])
    return r.overview(now=STOP_TS + 600)


@pytest.mark.parametrize("recovery", [False, True])
def test_a_trainer_we_stopped_that_wrote_its_leave_stays_stopped(tmp_path, recovery):
    ov = _stopped_single_trainer(tmp_path, recovery=recovery)
    card = ov["islands"][0]
    assert card["status"] == "stopped" and card["stopped_by_us"] and card["island_now"] is None
    assert card["left"] is not None  # its island_leave is read, and does not override our stop
    assert not [a for a in ov["alerts"] if a["rule"] in ("recovery_required", "heartbeat")]
    assert ov["global_status"]["text"].startswith("stopped")


def test_a_fleet_record_is_above_our_stop(tmp_path):
    """A per-island fleet's trainer we stopped whose record says abandoned reads lost (the
    record), one whose record says stopped reads stopped."""
    r = fleet_reducer()
    r.load_fleet_state(_state(tmp_path))
    now = r.max_ts + 30
    for iid in ("0", "1"):
        r.islands[iid]["stop_ts"] = r.max_ts
    r.operator_stop = lambda: {"time_unix": r.max_ts, "cause": "gate"}
    r.stopped_by_us = lambda isl: True
    cards = {c["id"]: c for c in r.overview(now=now)["islands"]}
    assert cards["0"]["status"] == "lost" and cards["0"]["left"]["source"] == "trainer island record"
    assert cards["1"]["status"] == "stopped"


def test_a_given_up_island_whose_bill_closed_says_so():
    rec = {"launch_state": "abandoned", "life_state": "abandoned", "relaunches": 2, "gpu": AWS}
    assert "still billed" in fleet_islands.island_state(rec)["detail"]
    down = fleet_islands.island_state({**rec, "down_at": T0, "closed_at": T0})
    assert down["state"] == "abandoned" and "still billed" not in down["detail"] and "run went down" in down["detail"]
    assert "its bill closed" in fleet_islands.island_state({**rec, "closed_at": T0})["detail"]
    assert fleet_islands.island_state({**rec, "launch_state": "removed"})["state"] == "removed"


# -- REV #838: mutations the first round did not catch ----------------------------------------
def test_a_never_granted_machine_with_a_tape_keeps_its_tape_hours(tmp_path):
    """The pool state says never granted (run.budget bills 0): the sampler's tape hours stand,
    the grant rule never lowers them."""
    base, _, _ = _cost(tmp_path)
    tape_h = base["qa-a-i1-0"]["hours"]
    rows, _, _ = _cost(tmp_path / "b", pools={"i1-a": [
        {"shard": "qa-a-i1-0", "state": "gone", "granted": None, "charged_to": None}]})
    assert tape_h > 0 and rows["qa-a-i1-0"]["hours"] == pytest.approx(tape_h)
    assert "sampler tapes" in rows["qa-a-i1-0"]["billed_from"]


def test_the_syncer_runs_while_the_up_is_open_after_the_trainers_stopped(tmp_path):
    rows, _, now = _cost(tmp_path, up={"trainer_t0": T0, "ups": [{"start": T0, "end": None, "pid": 1}]},
                         later=7200.0, extra=tuple({"event": "island_stop", "island_id": k, "time_unix": T0 + 900,
                                                    "island": f"fleet-qa-l{k}-modal"} for k in (0, 1)))
    assert not any(r.get("running") for r in rows.values() if r.get("kind") == "trainer")
    assert rows["syncer"]["running"]


def test_a_relaunching_or_moving_island_reads_starting(tmp_path):
    r = fleet_reducer()
    st = _state(tmp_path)
    st["trainers"][1]["record"] = {**st["trainers"][1]["record"], "launch_state": "running", "life_state": "recovering",
                                   "relaunches": 1}
    st["trainers"][1]["now"] = fleet_islands.island_state(st["trainers"][1]["record"], st["trainers"][1])
    st["trainers"][0]["now"] = {"state": "moving", "reason": f"moving to {AWS}", "detail": None}
    r.load_fleet_state(st)
    cards = {c["id"]: c for c in r.overview(now=r.max_ts + 3600)["islands"]}
    assert cards["1"]["status"] == "starting" and cards["1"]["starting"]
    assert cards["0"]["status"] == "starting" and cards["0"]["island_now"]["state"] == "moving"


def test_a_restart_of_every_island_keeps_its_open_rounds_until_they_go_stale():
    """No trainer up, but the last trainer event is less than ROUNDS_STALE_S old (every island
    restarting): the open rounds stay in flight, their clock runs and their quorum warning
    stands; past it they are stale."""
    from test_dashboard_diloco import _replayed

    from yeto.dashboard.topology_diloco import ROUNDS_STALE_S

    full = load("run-b")
    ts = [x for x in full.islands.values() if x.get("first_ts") is not None]
    mid = (min(x["first_ts"] for x in ts) + max(x["last_event_ts"] for x in ts)) / 2
    r = _replayed("run-b", mid)
    end = max(x["last_event_ts"] for x in r.islands.values() if x.get("last_event_ts") is not None)
    d = diloco(r, now=end + ROUNDS_STALE_S / 2)
    assert d["roster"] == 0 and not d["done"] and d["rounds"]
    now = end + ROUNDS_STALE_S / 2
    assert not any(x.get("ended") for x in d["rounds"]) and d["shards"][0]["state"] != "stale"
    assert all(x["age_s"] == pytest.approx(now - x["since"]) for x in d["rounds"])  # its clock runs
    stale = diloco(r, now=end + ROUNDS_STALE_S + 60)
    assert stale["rounds"] and all(x["ended"] == "no trainer up" for x in stale["rounds"])
    assert stale["shards"][0]["state"] == "stale"


# -- an island moving: its new launch is starting, not lost ----------------------------------
@pytest.mark.parametrize("rec, state, reason", [
    ({"launch_state": "moving", "life_state": "running", "moving_to": AWS, "gpu": MODAL}, "moving", f"moving to {AWS}"),
    ({"launch_state": "starting", "life_state": "ordering", "why": f"moved to {AWS}", "gpu": AWS, "incarnation": 1},
     "moving", f"moving to {AWS}"),
    ({"launch_state": "starting", "life_state": "ordering", "why": "relaunch 2/3: infra FAILED", "gpu": AWS,
      "relaunches": 2}, "starting", "relaunch 2/3: infra FAILED"),
])
def test_a_moving_or_relaunching_island_reads_so(rec, state, reason):
    got = fleet_islands.island_state(rec)
    assert got["state"] == state and got["reason"] == reason
    if rec["life_state"] == "ordering":
        assert "ordering machines" in got["detail"]


def _moving_state(tmp_path: Path, *, since: float | None) -> dict:
    """The fixture store's fleet state with island 0's next launch, after a move, ordering its
    machines since ``since`` (None: no time on its record)."""
    st = _state(tmp_path)
    st["trainers"][0]["record"] = {**st["trainers"][0]["record"], "launch_state": "starting", "life_state": "ordering",
                                   "why": f"moved to {AWS}", "relaunches": 0, "incarnation": 1, "since": since,
                                   "started_at": None, "stop": None}
    st["trainers"][0]["now"] = fleet_islands.island_state(st["trainers"][0]["record"], st["trainers"][0])
    return st


def _blocked_fleet():
    """The fleet fixture where island 1 pushed the next step and island 0 (moving) never will."""
    r = fleet_reducer()
    r.feed({"event": "island_leave", "island_id": 0, "time_unix": r.max_ts + 5, "reason_name": "STOPPED"})
    r.feed({"event": "rl_fragment_push", "island_id": 1, "time_unix": r.max_ts + 6, "global_step": 11,
            "fragment_id": 0, "base_version": 10, "payload_bytes": 1, "round_attempt": 1})
    return r


@pytest.mark.parametrize("ago", [None, 1800 + 600])
def test_an_island_moving_too_long_raises_the_stuck_and_blocked_alerts_again(tmp_path, ago):
    """Moving or relaunching for longer than MOVE_QUIET_S (or with no time on its record): its
    card still reads starting, but the rounds it holds up are blocked and alerted again."""
    from yeto.dashboard.topology_diloco import MOVE_QUIET_S

    assert MOVE_QUIET_S == 1800
    r = _blocked_fleet()
    now = r.max_ts + 900
    r.load_fleet_state(_moving_state(tmp_path, since=None if ago is None else now - ago))
    view = r.page_view(now=now)
    card = next(c for c in view["overview"]["islands"] if c["id"] == "0")
    assert card["status"] == "starting"
    d = view["topology"]["diloco"]
    assert any(x["blocked"] for x in d["rounds"])
    assert [a for a in view["overview"]["alerts"] if a["rule"] == "diloco_round_open"]


def test_a_moving_islands_old_leave_does_not_make_it_lost(tmp_path):
    """The old launch wrote island_leave STOPPED (its STOP: moving); the new launch orders its
    machines on the target cloud: the card reads starting (moving to <target>), not left / lost."""
    r = fleet_reducer()
    r.feed({"event": "island_leave", "island_id": 0, "time_unix": r.max_ts + 5, "reason_name": "STOPPED"})
    # island 1 pushed the next step; island 0 (moving) never will: without the move, a blocked round
    r.feed({"event": "rl_fragment_push", "island_id": 1, "time_unix": r.max_ts + 6, "global_step": 11,
            "fragment_id": 0, "base_version": 10, "payload_bytes": 1, "round_attempt": 1})
    now = r.max_ts + 900
    assert any(x["blocked"] for x in diloco(r, now=now)["rounds"])
    before = {c["id"]: c for c in r.overview(now=now)["islands"]}
    assert before["0"]["status"] in ("left", "lost")
    r.load_fleet_state(_moving_state(tmp_path, since=now - 600))
    view = r.page_view(now=now)
    card = next(c for c in view["overview"]["islands"] if c["id"] == "0")
    assert card["status"] == "starting" and card["left"] is None and card["waiting_reason"] == f"moving to {AWS}"
    assert not [a for a in view["overview"]["alerts"] if a.get("island") == "0" and a["rule"] in ("island_lost", "heartbeat")]
    d = view["topology"]["diloco"]
    assert d["stuck"] == [] and d["rounds"] and not any(x["blocked"] for x in d["rounds"])


def test_a_moving_islands_quiet_time_starts_at_the_newest_of_its_marks(tmp_path):
    """The STOP that began a move (or the new launch's start) is newer than the old launch's
    lifecycle state: the quiet time counts from it."""
    r = fleet_reducer()
    st = _moving_state(tmp_path, since=T0)
    st["trainers"][0]["record"].update(started_at=T0 + 50, stop={"at": T0 + 90, "by": "trainer worker",
                                                                 "reason": f"moving to {AWS}"})
    r.load_fleet_state(st)
    assert r._trainer_island_now(r.islands["0"])["state_since"] == T0 + 90
