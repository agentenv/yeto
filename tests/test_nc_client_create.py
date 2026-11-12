"""An order National Compute placed is in the ticket ledger even when the order call
did not return its ticket (yeto.launch.nc.client): CPU tests over a fake order tool."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from yeto.launch.nc import client as ncc


def answer(doc=None, *, code=0, stdout=None, stderr=""):
    return SimpleNamespace(returncode=code, stdout=json.dumps(doc) if stdout is None else stdout, stderr=stderr)


class Nc:
    """One account. ``on_create(nc)`` runs after the order is (or is not) placed and
    returns / raises what ``nc-manage create`` does; ``places``: the order is placed."""

    def __init__(self, ticket=None, *, on_create=None, places=True, status_down_after_order=False):
        self.ticket, self.on_create, self.places = ticket, on_create, places
        self.status_down_after_order = status_down_after_order
        self.creates = 0
        self.deletes = 0

    def __call__(self, argv, **kw):
        if argv[0] == "nc-query":
            if self.status_down_after_order and self.creates:
                return answer(code=1, stdout="", stderr='{"ok":false,"error":"502"}')
            return answer({"ok": True, "account": 1, "ticket": self.ticket and dict(self.ticket, others=[])})
        if argv[1] == "delete":
            self.deletes += 1
            self.ticket = dict(self.ticket, state="ended", end_reason="released")
            return answer({"ok": True, "released": True})
        self.creates += 1
        if self.places:
            self.ticket = {"id": f"new{self.creates}", "state": "queued"}
        if self.on_create is not None:
            return self.on_create(self)
        return answer({"ok": True, "created": True, "ticket": dict(self.ticket)})


@pytest.fixture(autouse=True)
def _tools(monkeypatch):
    monkeypatch.setenv("NC_MANAGE", "nc-manage")
    monkeypatch.setenv("NC_QUERY", "nc-query")


def client(run, tmp_path):
    return ncc.NcClient(1, ncc.Ledger(tmp_path / "tickets.json"), run=run, sleep=lambda s: None)


def timed_out(nc):
    raise subprocess.TimeoutExpired(["nc-manage", "create"], 300)


LOST_ANSWERS = {
    "the call timed out": timed_out,
    "the output cannot be parsed": lambda nc: answer(stdout='{"ok": true, "ticket": {"id": "ne'),
    "the tool exited 1 after the order": lambda nc: answer(code=1, stdout="", stderr='{"ok":false,"error":"ws"}'),
    "the answer has no ticket": lambda nc: answer({"ok": True}),
}


@pytest.mark.parametrize("name", sorted(LOST_ANSWERS))
def test_an_order_that_was_placed_but_whose_answer_was_lost_is_recorded_and_can_be_released(name, tmp_path):
    """status: no order | nc-manage create: the order is placed, the answer is lost |
    the ticket bills, the ledger does not know it: ``delete`` refuses it, no patrol sees it."""
    nc = Nc(on_create=LOST_ANSWERS[name])
    c = client(nc, tmp_path)
    assert c.create(run="run-a") == "new1"
    assert [(r["account"], r["ticket"]) for r in c.ledger.run_rows("run-a")] == [(1, "new1")]
    assert c.delete("new1")["state"] == "ended" and nc.deletes == 1
    assert c.ledger.open_rows() == []


def test_a_failed_order_that_placed_nothing_raises_and_does_not_block_the_next_order(tmp_path):
    refused = answer(code=1, stdout="", stderr='{"ok":false,"error":"余额不足"}')
    nc = Nc(places=False, on_create=lambda nc: refused)
    c = client(nc, tmp_path)
    with pytest.raises(ncc.NcError, match="余额不足"):
        c.create(run="run-a")
    assert c.ledger.open_rows() == [] and c.ledger.run_rows("run-a") == []
    nc.places, nc.on_create = True, None
    assert c.create(run="run-a") == "new2"
    assert [r["ticket"] for r in c.ledger.open_rows()] == ["new2"]


def test_a_ticket_the_tool_found_instead_of_ordering_is_not_taken_into_the_ledger(tmp_path):
    """Someone else ordered on the account between our status and our order: the tool
    answers ``created: false`` with their ticket. It is not ours to record or delete."""
    def found(nc):
        nc.ticket = {"id": "theirs", "state": "active"}
        return answer({"ok": True, "created": False, "reason": "账号已有未结束订单", "ticket": dict(nc.ticket)})

    c = client(Nc(places=False, on_create=found), tmp_path)
    with pytest.raises(ncc.NcRefused, match="theirs"):
        c.create(run="run-a")
    assert not c.ledger.owns(1, "theirs") and c.ledger.open_rows() == []
    with pytest.raises(ncc.NcRefused, match="not created by this tool"):
        c.delete("theirs")
    with pytest.raises(ncc.NcRefused) as refused:  # and no later order calls it our lost order
        c.create(run="run-a")
    assert "not recorded" not in str(refused.value) and c.ledger.pending_orders(1) == []


def test_a_failed_order_never_adopts_the_ticket_the_account_had_before(tmp_path):
    nc = Nc({"id": "old", "state": "ended", "end_reason": "released"}, places=False, on_create=timed_out)
    c = client(nc, tmp_path)
    with pytest.raises(subprocess.TimeoutExpired):
        c.create(run="run-a")
    assert not c.ledger.owns(1, "old") and c.ledger.open_rows() == []


def test_an_order_cut_off_with_the_process_is_named_by_the_next_order_on_the_account(tmp_path):
    """The process dies inside the order call: the ledger keeps the pending order, which
    is no ticket to anyone, and the next order on the account says the open ticket may be
    that unrecorded order."""
    def killed(nc):
        raise KeyboardInterrupt

    nc = Nc(on_create=killed)
    c = client(nc, tmp_path)
    with pytest.raises(KeyboardInterrupt):
        c.create(run="run-a")
    assert c.ledger.open_rows() == [] and c.ledger.run_rows("run-a") == []
    assert [(r["account"], r.get("run")) for r in c.ledger.pending_orders(1)] == [(1, "run-a")]
    with pytest.raises(ncc.NcRefused, match="new1.*not recorded"):
        client(nc, tmp_path).create(run="run-a")
    assert not c.ledger.owns(1, "new1")  # never adopted later: nothing ties the ticket to that order


def test_a_failed_order_followed_by_an_unreadable_status_stays_pending_and_raises_the_order_error(tmp_path):
    nc = Nc(on_create=timed_out, status_down_after_order=True)
    c = client(nc, tmp_path)
    with pytest.raises(subprocess.TimeoutExpired):
        c.create(run="run-a")
    assert len(c.ledger.pending_orders(1)) == 1 and c.ledger.open_rows() == []
    nc.status_down_after_order = False
    with pytest.raises(ncc.NcRefused, match="new1.*not recorded"):
        c.create(run="run-a")


def test_a_pending_order_is_dropped_once_the_account_reads_free(tmp_path):
    def killed(nc):
        raise KeyboardInterrupt

    nc = Nc(places=False, on_create=killed)
    c = client(nc, tmp_path)
    with pytest.raises(KeyboardInterrupt):
        c.create(run="run-a")
    assert len(c.ledger.pending_orders(1)) == 1
    c.ledger.mark_pending(1, created_at=0.0)  # long ago: no order call is in flight
    nc.places, nc.on_create = True, None
    assert c.create(run="run-a") == "new2"
    assert c.ledger.pending_orders(1) == [] and [r["ticket"] for r in c.ledger.load()] == ["new2"]


def test_an_order_in_flight_on_the_account_refuses_a_second_order(tmp_path):
    """Two processes on one state dir order on the same account: the second would read
    the first one's ticket as its own lost order."""
    nc = Nc()
    c = client(nc, tmp_path)
    c.ledger.begin_order(1, before=None, run="run-a")  # another process, inside nc-manage create
    with pytest.raises(ncc.NcRefused, match="in flight"):
        c.create(run="run-a")
    assert nc.creates == 0


def test_an_order_whose_pending_row_was_dropped_meanwhile_is_still_recorded(tmp_path):
    def slow(nc):  # another command on the state dir clears the account's pending rows during the call
        ledger = ncc.Ledger(tmp_path / "tickets.json")
        ledger._write([r for r in ledger.load() if r.get("ticket")])
        return answer({"ok": True, "created": True, "ticket": dict(nc.ticket)})

    c = client(Nc(on_create=slow), tmp_path)
    assert c.create(run="run-a") == "new1" and c.ledger.owns(1, "new1")
    assert [r["ticket"] for r in c.ledger.run_rows("run-a")] == ["new1"]
