"""A fleet launched island by island (``fleet.launch: per_island``) on the dashboard: the trainer
islands of the desired state (added, moved, removed while the fleet trains) and what their records
under ``fleet/trainers/<k>/`` say they are (stopped, removed, failed, abandoned, moving), and a
syncer whose tape this host does not see."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_dashboard_diloco import D as DILOCO
from test_dashboard_diloco import diloco, lines, load
from test_dashboard_fleet_islands import T0, fleet_reducer, plan_doc

from yeto.dashboard import cost, fleet_islands
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


def _cost(tmp_path: Path, *, up: dict | None = None, pools: dict | None = None, spend: dict | None = None):
    root = _store(tmp_path)
    for name, members in (pools or {}).items():
        (root / f"fleet/pools/{name}").mkdir(parents=True, exist_ok=True)
        (root / f"fleet/pools/{name}/state.json").write_text(json.dumps({"members": members}))
    r = fleet_reducer()
    now = r.max_ts + 30
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
