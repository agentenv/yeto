"""Ledger time faults must never shrink the bill or disable the budget guard: a
corrupt ``ts`` field, or a release stamped before its create (wall clock stepped back)."""

from __future__ import annotations

import json

import pytest

from tests.test_cost_guard import H, T0, Clock, events, ledger_at, make_guard
from yeto.cost.billing import Rates, resource_costs
from yeto.cost.ledger import ResourceLedger


def _append(path, **rec):
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"attrs": {}, "name": None, **rec}) + "\n")


def test_a_corrupt_ts_does_not_make_every_guard_tick_raise(tmp_path):
    clock = Clock(T0 + 3 * H)
    g, led, calls = make_guard(tmp_path, clock, budget=50.0)
    ledger_at(led.path, T0).created("nc_lease", "t-1", "nc", account=4, usd_per_hour=24.0)
    _append(led.path, op="create", kind="nc_lease", id="t-2", name="nc2", ts="garbage",
            attrs={"account": 4, "usd_per_hour": 24.0})
    s = g.tick()
    # t-2 bills from the newest good stamp before it (T0): never less than its true span
    assert s.total_usd == pytest.approx(2 * 72.0)
    assert "cost_budget_exceeded" in events(tmp_path) and calls["graceful"]


def test_a_corrupt_release_ts_keeps_at_least_the_span_up_to_the_last_good_stamp(tmp_path):
    p = tmp_path / "r.jsonl"
    ledger_at(p, T0).created("nc_lease", "t-1", "nc", account=4, usd_per_hour=10.0)
    ledger_at(p, T0 + 2 * H).created("nc_lease", "t-2", "nc2", account=4, usd_per_hour=10.0)
    _append(p, op="release", kind="nc_lease", id="t-1", name="nc", ts=["bad"])
    res = {r.id: r for r in ResourceLedger(p).resources()}
    assert res["t-1"].released_at == T0 + 2 * H


def test_a_release_stamped_before_its_create_bills_the_span_to_the_newest_stamp_seen(tmp_path):
    """The clock stepped back 3 h between create and release: append order is the true
    order, so the release is no earlier than the newest stamp before it (a monotonic floor)."""
    p = tmp_path / "r.jsonl"
    ledger_at(p, T0).created("nc_lease", "t-1", "nc", account=4, usd_per_hour=10.0)
    ledger_at(p, T0 + 2 * H).created("nc_lease", "t-2", "nc2", account=4, usd_per_hour=10.0)
    ledger_at(p, T0 - 1 * H).released("nc_lease", "t-1", "nc")
    s = resource_costs(ResourceLedger(p).resources(), T0 + 2 * H, rates=Rates())
    by = {i.key: i for i in s.items}
    t1 = next(v for k, v in by.items() if "t-1" in k)
    assert t1.usd == pytest.approx(20.0)  # not 0


def test_span_cost_warns_when_end_is_before_start(tmp_path, caplog):
    p = tmp_path / "r.jsonl"
    ledger_at(p, T0).created("nc_lease", "t-1", "nc", account=4, usd_per_hour=10.0)
    r = ResourceLedger(p).resources()[0]
    r.released_at = T0 - H
    with caplog.at_level("WARNING"):
        resource_costs([r], T0 + H, rates=Rates())
    assert "before its start" in caplog.text
