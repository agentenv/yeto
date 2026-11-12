"""Invariants of other suites, plugged into the simulation (``SimConfig.checks``,
``--check tests.store_sim.checks:<name>``).

* :data:`nc_ownership`: the National Compute client changes only tickets its
  ledger holds. The fake order tool and its checks live in ``tests/nc_fault_sim.py``;
  a tree without that module skips the check (``nc_ownership_not_available``).
* :data:`read_outage`: while the store refuses reads for no longer than the
  bound, no sampler exits and no claim in flight is given back.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any


def _nc_module() -> Any:
    for name in ("nc_fault_sim", "tests.nc_fault_sim"):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    return None


class NcOwnership:
    """One random sequence of NC client calls per simulation, on the simulation's
    seed, next to the store run: every changing call must hit a ticket of the ledger,
    and after the faults end no ticket the tool placed stays open unreported."""

    steps = 150

    def on_check(self, sim: Any) -> None:
        nc = _nc_module()
        if nc is None:
            sim.metric("nc_ownership_not_available")
            return
        scenario = nc.scenario(Path(sim.tmp) / "nc", sim.config.seed)
        nc.run_scenario(scenario, self.steps)
        for line in nc.violations(scenario.nc):
            sim.violation("nc_change_outside_the_ledger", detail=line)
        nc.settle(scenario)
        for line in nc.violations(scenario.nc):
            sim.violation("nc_change_outside_the_ledger", detail=line, phase="settle")
        for line in nc.leaks(scenario):
            sim.violation("nc_ticket_left_open_unreported", detail=line)
        sim.metric("nc_changing_calls", len(scenario.nc.writes))


class ReadOutage:
    """Reads fail for at most ``FaultConfig.read_outage_max_s``; the dispenser lease
    outlasts it. A sampler that exits on a failed read, or gives a claim back while
    it closes, breaks the invariant. Other faults must be off (a crash is an exit)."""

    def on_check(self, sim: Any) -> None:
        faults = sim.config.faults
        if not faults.read_outage or faults.crash or faults.partition or faults.outage or faults.control:
            sim.metric("read_outage_check_not_applicable")
            return
        if sim.config.lease_s <= faults.read_outage_max_s:
            raise ValueError("the read-outage check needs a dispenser lease longer than the outage")
        exits = sim.metrics.get("sampler_exits_on_store_error", 0) + sim.metrics.get("sampler_exits_on_error", 0)
        if exits:
            sim.violation("sampler_exited_during_a_read_outage", exits=exits,
                          errors=dict(list(sim.errors.items())[:3]))
        released = sum(n for name, n in sim.metrics.items() if name.startswith("claims_released_"))
        if released:
            sim.violation("claim_released_during_a_read_outage", released=released)


nc_ownership = NcOwnership()
read_outage = ReadOutage()

__all__ = ["NcOwnership", "ReadOutage", "nc_ownership", "read_outage"]
