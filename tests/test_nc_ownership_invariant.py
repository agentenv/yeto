"""The ownership invariant of the National Compute client under injected faults
(tests/nc_fault_sim.py): no changing call ever hits a ticket outside the ledger, and
every ticket the tool's orders placed ends released or reported for an operator."""

from __future__ import annotations

import pytest

import nc_fault_sim as sim

SEEDS = range(60)


@pytest.mark.parametrize("seed", SEEDS)
def test_under_random_faults_the_client_never_changes_a_ticket_outside_its_ledger(seed, tmp_path):
    sc = sim.scenario(tmp_path, seed)
    sim.run_scenario(sc, steps=150)
    assert sim.violations(sc.nc) == []
    foreign_before = sim.still_open(sc, "foreign")

    sim.settle(sc, operator=False)
    assert sim.violations(sc.nc) == []
    assert sim.leaks(sc) == []  # what is still open of ours was reported (event order_unrecorded)
    assert sc.ledger.open_rows() == [] or all(  # an open row is a ticket that ended behind another's order
        sc.nc.tickets[r["ticket"]]["state"] == "ended" for r in sc.ledger.open_rows())
    assert sim.still_open(sc, "foreign") == foreign_before  # nobody else's ticket was touched

    sim.settle(sc)  # the operator adopts what was reported and is ours; the tool releases it
    assert sim.violations(sc.nc) == []
    assert sim.still_open(sc, "client") == [] and sim.still_open(sc, "foreign") == foreign_before


def test_the_scenarios_reach_the_cases_the_invariant_is_about(tmp_path):
    """The random runs are not vacuous: orders, releases, lost answers that left a
    ticket, reports, and other people's tickets all occur across the seeds."""
    seen = {"created": 0, "deleted": 0, "reported": 0, "foreign": 0, "recorded": 0}
    for seed in SEEDS:
        sc = sim.scenario(tmp_path / str(seed), seed)
        sim.run_scenario(sc, steps=150)
        seen["created"] += sum(w["op"] == "create" for w in sc.nc.writes)
        seen["deleted"] += sum(w["op"] == "delete" and w["ticket"] is not None for w in sc.nc.writes)
        seen["reported"] += len(sc.reported())
        seen["foreign"] += sum(t["by"] == "foreign" for t in sc.nc.tickets.values())
        seen["recorded"] += sum(1 for r in sc.ledger.load() if r.get("ticket"))
    assert all(count >= 30 for count in seen.values()), seen


def test_the_check_catches_a_delete_of_a_ticket_that_is_not_in_the_ledger(tmp_path):
    sc = sim.scenario(tmp_path, 0, fault_rate=0.0)
    sc.nc._place(1, "foreign")
    sc.nc(["nc-manage", "delete", "--account", "1", "--yes"])  # what the client must never do
    assert len(sim.violations(sc.nc)) == 2
    sc = sim.scenario(tmp_path / "b", 0, fault_rate=0.0)
    sc.nc._place(1, "client")  # placed by an order whose answer was lost: in no ledger
    sc.nc(["nc-manage", "delete", "--account", "1", "--yes"])
    assert len(sim.violations(sc.nc)) == 1
    sim.settle(sim.scenario(tmp_path / "c", 0, fault_rate=0.0))
    sc = sim.scenario(tmp_path / "d", 0, fault_rate=0.0)
    sc.nc._place(4, "client")
    assert len(sim.leaks(sc)) == 1  # open, ours, and nobody was told
