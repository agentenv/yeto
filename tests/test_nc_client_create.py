"""Orders and the ticket ledger (yeto.launch.nc.client): a ticket is this tool's only
when its id came in the answer of an order this tool wrote down first, or an operator
adopted it. An order whose answer is lost is reported, never recorded by the tool.
CPU tests over a fake order tool."""

from __future__ import annotations

import json
import subprocess
import time
from types import SimpleNamespace

import pytest

from yeto.launch.nc import cli
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


def client(run, tmp_path, events=None):
    event = None if events is None else (lambda name, **detail: events.append((name, detail)))
    return ncc.NcClient(1, ncc.Ledger(tmp_path / "tickets.json"), run=run, sleep=lambda s: None, event=event)


def timed_out(nc):
    raise subprocess.TimeoutExpired(["nc-manage", "create"], 300)


def killed(nc):
    raise KeyboardInterrupt


LOST_ANSWERS = {
    "the call timed out": timed_out,
    "the output cannot be parsed": lambda nc: answer(stdout='{"ok": true, "ticket": {"id": "ne'),
    "the tool exited 1 after the order": lambda nc: answer(code=1, stdout="", stderr='{"ok":false,"error":"ws"}'),
    "the answer has no ticket": lambda nc: answer({"ok": True}),
}


def test_an_order_is_recorded_from_its_own_answer_with_the_order_it_answers(tmp_path):
    c = client(Nc(), tmp_path)
    assert c.create(run="run-a") == "new1"
    (row,) = c.ledger.load()
    assert (row["account"], row["ticket"], row["run"], row["model"]) == (1, "new1", "run-a", "mi355x")
    assert row["order"] and "adopted_by" not in row and c.ledger.pending_orders(1) == []


@pytest.mark.parametrize("name", sorted(LOST_ANSWERS))
def test_an_order_whose_answer_was_lost_is_reported_and_never_recorded_or_deleted_by_the_tool(name, tmp_path):
    """status: no order | nc-manage create: the order is placed, the answer is lost | the
    account shows a ticket whose id no answer gave us: maybe ours, maybe not. It is
    reported for an operator, the account takes no further order, nothing touches it."""
    nc, events = Nc(on_create=LOST_ANSWERS[name]), []
    c = client(nc, tmp_path, events)
    with pytest.raises(ncc.NcOrderUnrecorded, match="new1.*not recorded"):
        c.create(run="run-a")
    assert not c.ledger.owns(1, "new1") and c.ledger.open_rows() == [] and c.ledger.run_rows("run-a") == []
    (pending,) = c.ledger.pending_orders(1)
    assert pending["observed_ticket"] == "new1" and pending["run"] == "run-a" and pending["call_ended"]
    assert [(n, d["account"], d["ticket"]) for n, d in events] == [("order_unrecorded", 1, "new1")]
    with pytest.raises(ncc.NcRefused, match="not created by this tool"):
        c.delete("new1")
    assert c.account_free() is False
    with pytest.raises(ncc.NcRefused, match="new1.*not recorded"):
        c.create(run="run-a")
    assert (nc.creates, nc.deletes, len(events)) == (1, 0, 1)  # reported once, no second order, no delete


def test_an_operator_adopts_the_unrecorded_ticket_and_the_tool_can_then_release_it(tmp_path):
    nc = Nc(on_create=timed_out)
    c = client(nc, tmp_path, [])
    with pytest.raises(ncc.NcOrderUnrecorded):
        c.create(run="run-a")
    row = c.adopt("new1")
    assert (row["ticket"], row["adopted_by"], row["run"]) == ("new1", "operator", "run-a") and row["adopted_at"]
    assert [(r["account"], r["ticket"]) for r in c.ledger.run_rows("run-a")] == [(1, "new1")]
    assert c.ledger.pending_orders(1) == []
    assert c.delete("new1")["state"] == "ended" and nc.deletes == 1 and c.ledger.open_rows() == []


def test_adopt_is_refused_without_an_unrecorded_order_or_for_a_ticket_that_cannot_be_it(tmp_path):
    foreign = {"id": "theirs", "state": "active", "granted_at": "2026-11-10T08:00:00Z"}
    c = client(Nc(dict(foreign)), tmp_path)
    with pytest.raises(ncc.NcRefused, match="no order of this tool"):
        c.adopt("theirs")  # an open ticket, but this tool sent no order that could be it
    assert c.ledger.load() == []

    nc = Nc(on_create=timed_out)
    c = client(nc, tmp_path / "b", [])
    with pytest.raises(ncc.NcOrderUnrecorded):
        c.create(run="run-a")
    with pytest.raises(ncc.NcRefused, match="not the account's current open ticket"):
        c.adopt("other")
    nc.ticket = dict(nc.ticket, state="active", granted_at="2020-01-01T00:00:00Z")
    with pytest.raises(ncc.NcRefused, match="older than the unrecorded order"):
        c.adopt("new1")  # granted long before the order was sent
    nc.ticket = {"id": "new1", "state": "ended", "end_reason": "released"}
    with pytest.raises(ncc.NcRefused, match="not the account's current open ticket"):
        c.adopt("new1")  # no longer open: nothing to release
    assert not c.ledger.owns(1, "new1") and len(c.ledger.pending_orders(1)) == 1

    nc.ticket = {"id": "new1", "state": "queued"}
    c.ledger.mark_pending(1, call_ended=None, created_at=time.time())  # an order call still in flight
    with pytest.raises(ncc.NcRefused, match="no order of this tool"):
        c.adopt("new1")

    c = client(Nc(), tmp_path / "c")
    assert c.create() == "new1"
    with pytest.raises(ncc.NcRefused, match="already in the ledger"):
        c.adopt("new1")


def test_the_adopt_command_needs_yes_and_records_the_operator(tmp_path, monkeypatch, capsys):
    nc = Nc(on_create=timed_out)
    c = client(nc, tmp_path, [])
    with pytest.raises(ncc.NcOrderUnrecorded):
        c.create(run="run-a")
    monkeypatch.setattr(cli, "client", lambda account, **kw: c)
    assert cli.main(["adopt", "--account", "1", "--ticket", "new1"]) == 2 and not c.ledger.owns(1, "new1")
    assert cli.main(["adopt", "--account", "1", "--ticket", "nope", "--yes"]) == 1
    assert "adopt refused" in capsys.readouterr().err
    assert cli.main(["adopt", "--account", "1", "--ticket", "new1", "--yes"]) == 0
    assert c.ledger.load()[0]["adopted_by"] == "operator"


def test_a_failed_order_that_placed_nothing_raises_and_does_not_block_the_next_order(tmp_path):
    refused = answer(code=1, stdout="", stderr='{"ok":false,"error":"insufficient balance"}')
    nc, events = Nc(places=False, on_create=lambda nc: refused), []
    c = client(nc, tmp_path, events)
    with pytest.raises(ncc.NcError, match="insufficient balance"):
        c.create(run="run-a")
    assert c.ledger.open_rows() == [] and c.ledger.run_rows("run-a") == [] and events == []
    assert c.account_free() is True
    nc.places, nc.on_create = True, None
    assert c.create(run="run-a") == "new2"
    assert [r["ticket"] for r in c.ledger.load()] == ["new2"]


def test_a_ticket_the_tool_found_instead_of_ordering_is_not_taken_into_the_ledger(tmp_path):
    """Someone else ordered on the account between our status and our order: the tool
    answers ``created: false`` with their ticket. It is not ours to record or delete."""
    def found(nc):
        nc.ticket = {"id": "theirs", "state": "active"}
        return answer({"ok": True, "created": False, "reason": "an open order exists", "ticket": dict(nc.ticket)})

    events = []
    c = client(Nc(places=False, on_create=found), tmp_path, events)
    with pytest.raises(ncc.NcRefused, match="theirs"):
        c.create(run="run-a")
    assert c.ledger.load() == []
    with pytest.raises(ncc.NcRefused, match="not created by this tool"):
        c.delete("theirs")
    with pytest.raises(ncc.NcRefused) as refused:  # and no later look calls it our lost order
        c.create(run="run-a")
    assert "not recorded" not in str(refused.value) and events == []
    with pytest.raises(ncc.NcRefused):
        c.adopt("theirs")


def test_a_failed_order_never_reports_the_ticket_the_account_had_before(tmp_path):
    nc, events = Nc({"id": "old", "state": "ended", "end_reason": "released"}, places=False, on_create=timed_out), []
    c = client(nc, tmp_path, events)
    with pytest.raises(subprocess.TimeoutExpired):
        c.create(run="run-a")
    assert not c.ledger.owns(1, "old") and c.ledger.open_rows() == [] and events == []


def test_an_order_cut_off_with_the_process_is_reported_at_the_next_look_at_the_account(tmp_path):
    """The process dies inside the order call: the ledger keeps the pending order. Once
    its call can no longer be running, the next look at the account reports the ticket."""
    nc, events = Nc(on_create=killed), []
    c = client(nc, tmp_path, events)
    with pytest.raises(KeyboardInterrupt):
        c.create(run="run-a")
    assert c.ledger.open_rows() == [] and len(c.ledger.pending_orders(1)) == 1
    assert c.account_free() is False and events == []  # the call may still run (another process)
    c.ledger.mark_pending(1, created_at=time.time() - 2 * ncc.ORDER_IN_FLIGHT_S)
    assert c.account_free() is False
    assert [(n, d["ticket"]) for n, d in events] == [("order_unrecorded", "new1")]
    with pytest.raises(ncc.NcRefused, match="new1.*not recorded"):
        client(nc, tmp_path, events).create(run="run-a")
    assert not c.ledger.owns(1, "new1") and len(events) == 1 and nc.creates == 1


def test_a_failed_order_followed_by_an_unreadable_status_raises_the_order_error_and_reports_later(tmp_path):
    nc, events = Nc(on_create=timed_out, status_down_after_order=True), []
    c = client(nc, tmp_path, events)
    with pytest.raises(subprocess.TimeoutExpired):
        c.create(run="run-a")
    assert len(c.ledger.pending_orders(1)) == 1 and c.ledger.open_rows() == [] and events == []
    nc.status_down_after_order = False
    assert c.account_free() is False
    assert [(n, d["ticket"]) for n, d in events] == [("order_unrecorded", "new1")]


def test_a_pending_order_is_kept_through_failed_orders_and_closed_by_the_next_order_that_succeeds(tmp_path):
    nc = Nc(places=False, on_create=killed)
    c = client(nc, tmp_path)
    with pytest.raises(KeyboardInterrupt):
        c.create(run="run-a")
    c.ledger.mark_pending(1, created_at=1000.0)  # long ago: no order call is in flight
    nc.on_create = timed_out
    with pytest.raises(subprocess.TimeoutExpired):
        c.create(run="run-a")
    (pending,) = c.ledger.pending_orders(1)  # one row per account, dated from the first unrecorded order
    assert pending["since"] == 1000.0 and pending["created_at"] > 1000.0
    nc.places, nc.on_create = True, None
    assert c.create(run="run-a") == "new3"
    (row,) = c.ledger.load()
    assert row["ticket"] == "new3" and not set(row) & set(ncc.PENDING_ONLY_FIELDS)


def test_one_wrong_no_order_answer_does_not_make_the_tool_forget_an_unrecorded_order(tmp_path):
    """unrecorded ticket new1 | status wrongly says "no order" once | the order tool
    itself finds new1 and answers ``created: false`` -> new1 must still be reported."""
    nc, events = Nc(on_create=timed_out), []
    c = client(nc, tmp_path, events)
    with pytest.raises(ncc.NcOrderUnrecorded):
        c.create(run="run-a")
    real, flap = nc.ticket, [True]

    def run(argv, **kw):
        if argv[0] == "nc-query" and flap and flap.pop():
            return answer({"ok": True, "ticket": None})
        if argv[0] != "nc-query":
            return answer({"ok": True, "created": False, "ticket": dict(real)})
        return nc(argv, **kw)

    c.run = run
    with pytest.raises(ncc.NcRefused, match="not placed"):
        c.create(run="run-a")
    (pending,) = c.ledger.pending_orders(1)
    assert pending["observed_ticket"] == "new1" and c.account_free() is False
    assert c.adopt("new1")["adopted_by"] == "operator"


def test_an_order_in_flight_on_the_account_refuses_a_second_order(tmp_path):
    """Two processes on one state dir order on the same account at once."""
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


def test_a_report_that_cannot_be_delivered_does_not_fail_the_call(tmp_path, capsys):
    def broken(name, **detail):
        raise RuntimeError("pager down")

    c = ncc.NcClient(1, ncc.Ledger(tmp_path / "tickets.json"), run=Nc(on_create=timed_out), event=broken)
    with pytest.raises(ncc.NcOrderUnrecorded):
        c.create()
    assert "order_unrecorded" in capsys.readouterr().err


def test_the_pools_provider_reports_an_unrecorded_order_in_the_pools_event_file(tmp_path, monkeypatch):
    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("YETO_NC_ACCOUNTS", "1")
    cfg = {"target": 1, "accounts": [1], "shard_base": "nc-x", "run": "run-x", "spec": "s.json",
           "run_prefix": "run-x", "r2_env_file": "store.env"}
    provider = cli.nc_provider(cfg, "run-x")
    provider._run, provider.before_order, provider.pubkey = Nc(on_create=timed_out), None, None
    with pytest.raises(ncc.NcOrderUnrecorded):
        provider.order(1)
    assert provider.account_free(1) is False and provider.run_tickets() == []
    (line,) = cli.pool_events_path(cfg).read_text().splitlines()
    doc = json.loads(line)
    assert (doc["event"], doc["account"], doc["ticket"], doc["run"]) == (
        "nc_elastic_order_unrecorded", 1, "new1", "run-x")
