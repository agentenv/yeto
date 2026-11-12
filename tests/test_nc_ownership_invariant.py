"""The ownership invariant of the National Compute client under injected faults
(tests/nc_fault_sim.py): no changing call ever hits a ticket outside the ledger, and
every ticket the tool's orders placed ends released or reported for an operator."""

from __future__ import annotations

import pytest

import nc_fault_sim as sim

SEEDS = range(60)


def _untouched_foreign(sc):
    """Foreign tickets the client ended, outside the residual window."""
    residual = {w["ticket"] for w in sc.nc.writes if w.get("residual")}
    return [t["id"] for t in sc.nc.tickets.values()
            if t["by"] == "foreign" and t.get("ended_by") == "client" and t["id"] not in residual]


@pytest.mark.parametrize("seed", SEEDS)
def test_under_random_faults_the_client_never_changes_a_ticket_outside_its_ledger(seed, tmp_path):
    sc = sim.scenario(tmp_path, seed)
    sim.run_scenario(sc, steps=150)
    assert sim.violations(sc.nc) == []

    sim.settle(sc, operator=False)
    assert sim.violations(sc.nc) == [] and _untouched_foreign(sc) == []
    # a delete that took someone else's ticket in the residual window never goes unnoticed
    assert sim.residual_outcomes(sc)["undetected"] == [] and sc.nc.creates_after_residual == 0
    assert sim.leaks(sc) == []  # what is still open of ours was reported (order_unrecorded, or a halt)
    foreign = sim.still_open(sc, "foreign")

    sim.settle(sc)  # the operator lifts halts, adopts or releases what was reported; the tool releases the rest
    assert sim.violations(sc.nc) == [] and _untouched_foreign(sc) == []
    assert sim.still_open(sc, "client") == [] and sim.still_open(sc, "foreign") == foreign


def test_the_scenarios_reach_the_cases_the_invariant_is_about(tmp_path):
    """The random runs are not vacuous: orders, releases, lost answers that left a
    ticket, reports, other people's tickets, the residual window and each way the
    client accounts for it all occur across the seeds."""
    seen = {"created": 0, "deleted": 0, "reported": 0, "stale_reports": 0, "foreign": 0, "recorded": 0,
            "residual_detected": 0, "residual_unconfirmed": 0, "residual_pending": 0, "residual_undetected": 0,
            "false_halts": 0}
    for seed in SEEDS:
        sc = sim.scenario(tmp_path / str(seed), seed)
        sim.run_scenario(sc, steps=150)
        sim.settle(sc, operator=False)
        seen["created"] += sum(w["op"] == "create" for w in sc.nc.writes)
        seen["deleted"] += sum(w["op"] == "delete" and w["ticket"] is not None for w in sc.nc.writes)
        seen["reported"] += len(sc.reported())
        seen["stale_reports"] += sum(1 for name, d in sc.events if name == "order_unrecorded" and d.get("stale"))
        seen["foreign"] += sum(t["by"] == "foreign" for t in sc.nc.tickets.values())
        seen["recorded"] += sum(1 for r in sc.ledger.load() if r.get("ticket"))
        for kind, hits in sim.residual_outcomes(sc).items():
            seen[f"residual_{kind}"] += len(hits)
        seen["false_halts"] += sim.false_halts(sc)
    print("scenario coverage:", seen)
    for key in ("created", "deleted", "reported", "foreign", "recorded"):
        assert seen[key] >= 30, seen
    assert seen["stale_reports"] >= 1 and seen["residual_detected"] >= 1, seen
    assert seen["residual_undetected"] == 0, seen


def test_the_check_catches_a_delete_of_a_ticket_that_is_not_in_the_ledger(tmp_path):
    sc = sim.scenario(tmp_path, 0, fault_rate=0.0, race_rate=0.0)
    sc.nc._place(1, "foreign")
    sc.nc(["nc-manage", "delete", "--account", "1", "--yes"])  # what the client must never do
    assert len(sim.violations(sc.nc)) == 1
    sc = sim.scenario(tmp_path / "b", 0, fault_rate=0.0, race_rate=0.0)
    sc.nc._place(1, "client")  # placed by an order whose answer was lost: in no ledger
    sc.nc(["nc-manage", "delete", "--account", "1", "--yes"])
    assert len(sim.violations(sc.nc)) == 1
    sim.settle(sim.scenario(tmp_path / "c", 0, fault_rate=0.0))
    sc = sim.scenario(tmp_path / "d", 0, fault_rate=0.0, race_rate=0.0)
    sc.nc._place(4, "client")
    assert len(sim.leaks(sc)) == 1  # open, ours, and nobody was told


def test_a_delete_after_another_look_at_the_account_is_a_violation_not_a_residual_hit(tmp_path):
    """The residual window is only "the status answer, then the delete": a client that
    looked again after the race and still deleted has no excuse."""
    sc = sim.scenario(tmp_path, 0, fault_rate=0.0, race_rate=0.0)
    c = sc.clients[1]
    c.create(run="run-a")
    assert sc.nc.race(1)
    sc.nc(["nc-query", "status", "--account", "1"])  # the client could see the foreign ticket here
    sc.nc(["nc-manage", "delete", "--account", "1", "--yes"])
    assert len(sim.violations(sc.nc)) == 1
    assert sim.residual_outcomes(sc) == {"detected": [], "unconfirmed": [], "pending": [], "undetected": []}


def test_the_check_catches_a_client_that_goes_on_after_a_residual_hit_it_did_not_notice(tmp_path):
    sc = sim.scenario(tmp_path, 0, fault_rate=0.0, race_rate=0.0)
    c = sc.clients[1]
    c.create(run="run-a")
    assert sc.nc.race(1)
    sc.nc(["nc-manage", "delete", "--account", "1", "--yes"])  # a client without the pending-delete record
    assert len(sim.residual_outcomes(sc)["undetected"]) == 1
    sc.nc.current[1] = None
    sc.nc(["nc-manage", "create", "--account", "1", "--model", "mi355x", "--yes"])
    assert sc.nc.creates_after_residual == 1
