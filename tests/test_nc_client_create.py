"""Orders and the ticket ledger (yeto.launch.nc.client): a ticket is this tool's only
when its id came in the answer of an order this tool wrote down first, or an operator
adopted it. An order whose answer is lost is reported, never recorded by the tool.
CPU tests over a fake order tool."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

import nc_fault_sim as sim
from yeto.launch.nc import cli
from yeto.launch.nc import client as ncc
from yeto.launch.nc.provider import NCProvider


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
    with pytest.raises(ncc.NcRefused, match="does not fit the unrecorded order"):
        c.adopt("new1")  # granted long before the order was sent
    late = datetime.fromtimestamp(time.time() + ncc.ADOPT_QUEUE_MAX_S + 3600, UTC).isoformat()
    nc.ticket = dict(nc.ticket, granted_at=late)
    with pytest.raises(ncc.NcRefused, match="does not fit the unrecorded order"):
        c.adopt("new1")  # granted long after the order: someone else's later order
    nc.ticket = {"id": "new1", "state": "ended", "end_reason": "released"}
    with pytest.raises(ncc.NcRefused, match="not the account's current open ticket"):
        c.adopt("new1")  # no longer open: nothing to release
    assert not c.ledger.owns(1, "new1") and len(c.ledger.pending_orders(1)) == 1

    nc.ticket = {"id": "new1", "state": "queued"}
    c.ledger.mark_pending(1, call_ended=None, created_at=time.time())  # an order call still in flight
    with pytest.raises(ncc.NcRefused, match="no order of this tool"):
        c.adopt("new1")
    c.ledger.mark_pending(1, call_ended=time.time())
    nc.ticket = {"id": "later", "state": "queued"}  # not the ticket this tool saw and reported for the order
    c.ledger.mark_pending(1, call_ended=time.time() - 2 * ncc.ORDER_PENDING_MAX_S)
    with pytest.raises(ncc.NcRefused, match="no order of this tool"):
        c.adopt("later")

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


# -- only tickets with evidence are touched (over the fault simulator's fake, without faults) -----

def _quiet(tmp_path):
    sc = sim.scenario(tmp_path, 0, fault_rate=0.0, race_rate=0.0)
    return sc, sc.nc, sc.clients[1]


def test_a_failed_order_that_placed_nothing_does_not_make_a_later_foreign_ticket_adoptable(tmp_path):
    """Day 1: an order call fails before the request (nothing placed); the account reads
    free afterwards. Day 3: someone else orders on the account. The pending order of day
    1 is closed: their ticket is not reported as ours, cannot be adopted, is not deleted."""
    sc, nc, c = _quiet(tmp_path)
    real = nc._create
    nc._create = lambda argv, account: nc._broken("error_exit", argv)  # failed before the request
    with pytest.raises(ncc.NcError):
        c.create(run="run-a")
    nc._create = real
    (pending,) = c.ledger.pending_orders(1)
    assert pending["free_reads"] == 1  # the look right after the failed call: one free read, not closed yet
    assert c.account_free() and c.ledger.pending_orders(1) == []  # the second in a row closes it
    sc.clock.advance(2 * 86400)
    foreign = nc._place(1, "foreign")
    nc._grant(foreign)
    assert not c.account_free() and sc.events == []
    with pytest.raises(ncc.NcRefused, match="no order of this tool"):
        c.adopt(foreign["id"])
    with pytest.raises(ncc.NcRefused, match="not created by this tool"):
        c.delete(foreign["id"])
    assert sim.violations(nc) == [] and sim.still_open(sc, "foreign") == [foreign["id"]]


def test_a_pending_order_that_never_read_free_goes_stale_and_a_late_ticket_cannot_be_adopted(tmp_path):
    """The order call is cut off and nobody looks at the account for an hour. A ticket
    first seen then is reported once, without the adopt command, and adopt refuses it."""
    sc, nc, c = _quiet(tmp_path)
    real = nc._create
    nc._create = lambda argv, account: nc._broken("timeout", argv)
    with pytest.raises(subprocess.TimeoutExpired):
        c.create(run="run-a")
    nc._create = real
    sc.clock.advance(ncc.ORDER_PENDING_MAX_S + 60)
    foreign = nc._place(1, "foreign")
    assert not c.account_free() and not c.account_free()
    ((name, detail),) = sc.events
    assert name == "order_unrecorded" and detail["stale"] is True and "adopt --account" not in detail["action"]
    with pytest.raises(ncc.NcRefused, match="no order of this tool"):
        c.adopt(foreign["id"])
    with pytest.raises(ncc.NcRefused) as refused:
        c.create(run="run-a")
    assert "not recorded" not in str(refused.value) and sim.violations(nc) == []


def test_a_free_read_between_two_looks_at_an_unrecorded_ticket_does_not_close_the_order(tmp_path):
    """One wrong "no order" answer counts one free read; the next answer shows the
    ticket again and the count starts over."""
    nc, events = Nc(on_create=timed_out), []
    c = client(nc, tmp_path, events)
    with pytest.raises(ncc.NcOrderUnrecorded):
        c.create(run="run-a")
    shown = nc.ticket
    for _ in range(3):
        nc.ticket = None
        assert c.account_free()
        nc.ticket = shown
        assert not c.account_free()
    assert c.ledger.pending_orders(1)[0]["observed_ticket"] == "new1"
    assert c.adopt("new1")["adopted_by"] == "operator"


def test_the_node_of_a_ticket_outside_the_ledger_is_not_handed_out(tmp_path):
    sc, nc, c = _quiet(tmp_path)
    foreign = nc._place(1, "foreign")
    nc._grant(foreign)
    with pytest.raises(ncc.NcRefused, match="not created by this tool"):
        c.wait_active(foreign["id"], timeout_s=0, log=lambda line: None)
    assert c.ledger.load() == []


def test_an_answer_without_the_created_field_is_not_recorded(tmp_path):
    """The order tool answers ok with the account's ticket and no ``created`` field,
    after a status read that missed the ticket: it does not say this order placed it."""
    sc, nc, c = _quiet(tmp_path)
    foreign = nc._place(1, "foreign")
    real_status, calls = nc._status, []

    def status(argv, account):  # the status read before the order misses the ticket once
        calls.append(1)
        return sim._answer({"ok": True, "account": account, "ticket": None}) if len(calls) == 1 \
            else real_status(argv, account)

    nc._status = status
    nc._create = lambda argv, account: sim._answer({"ok": True, "account": account, "ticket": nc._view(account)})
    with pytest.raises(ncc.NcError):
        c.create(run="run-a")
    assert not c.ledger.owns(1, foreign["id"]) and c.ledger.open_rows() == []


# -- the delete has no ticket id: what guards it ------------------------------------------------

def test_a_delete_is_not_sent_when_the_last_look_before_it_shows_another_ticket(tmp_path):
    """status: ours, open | ours ends, someone else orders | status again (right before
    the call): theirs -> no delete."""
    sc, nc, c = _quiet(tmp_path)
    mine = c.create(run="run-a")
    real_status, calls = nc._status, []

    def status(argv, account):
        out = real_status(argv, account)
        calls.append(1)
        if len(calls) == 1:
            nc.race(account)
        return out

    nc._status = status
    with pytest.raises(ncc.NcError, match="changed between two reads.*no delete sent"):
        c.delete(mine)
    assert [w for w in nc.writes if w["op"] == "delete"] == [] and sim.still_open(sc, "foreign") != []


@pytest.mark.parametrize("field", ["lease_until", "preempt_at"])
def test_no_delete_is_sent_while_the_ticket_is_about_to_end_on_its_own(field, tmp_path):
    sc, nc, c = _quiet(tmp_path)
    mine = c.create(run="run-a")
    nc._grant(nc.tickets[mine])
    nc.tickets[mine][field] = sc.clock() + 3600
    sc.clock.advance(3600 - ncc.DELETE_GUARD_S + 10)  # 110 s before it
    with pytest.raises(ncc.NcReleaseDeferred, match=field):
        c.delete(mine)
    assert [w for w in nc.writes if w["op"] == "delete"] == [] and c.ledger.owns(1, mine)
    if field == "lease_until":  # a lease does not end by itself: past the guard the delete goes out
        sc.clock.advance(2 * ncc.DELETE_GUARD_S)
        assert c.delete(mine)["state"] == "ended"
    else:  # preempted: nothing to delete, the ticket reads ended
        sc.clock.advance(2 * ncc.DELETE_GUARD_S)
        nc._due()
        assert c.delete(mine)["already"] is True
    assert sim.violations(nc) == []


@pytest.mark.parametrize("names_ticket", [True, False])
def test_a_delete_that_released_someone_elses_ticket_is_reported_and_halts_the_account(names_ticket, tmp_path):
    """The residual window: between the last status read and nc-manage delete our ticket
    ends on its own and someone else orders; the delete takes their ticket. The client
    cannot prevent it; it must notice, report it at once, and stop writing on the account."""
    sc, nc, c = _quiet(tmp_path)
    mine = c.create(run="run-a")
    real = nc._delete
    nc.rng.random = lambda: 0.01 if names_ticket else 0.99  # whether the delete answer names the ticket

    def raced(argv, account):
        nc.race(account)
        return real(argv, account)

    nc._delete = raced
    with pytest.raises(ncc.NcForeignTicketReleased, match="halted"):
        c.delete(mine)
    nc._delete = real
    ((name, detail),) = sc.events
    theirs = nc.current[1]
    assert (name, detail["ticket"], detail["expected"], detail["severity"]) == (
        "foreign_ticket_released", theirs, mine, "critical")
    assert sim.violations(nc) == [] and sim.residual_hits(sc) == {"detected": [theirs], "undetected": []}
    for call in (lambda: c.create(run="run-a"), lambda: c.delete(mine), lambda: c.adopt("x")):
        with pytest.raises(ncc.NcHalted, match="clear-halt"):
            call()
    assert c.ledger.clear_halt(1) and c.ledger.halted(1) is None
    assert c.delete(mine)["already"] is True  # ours had ended on its own


def test_a_foreign_release_behind_a_lost_delete_answer_is_noticed_at_the_next_release_attempt(tmp_path):
    sc, nc, c = _quiet(tmp_path)
    mine = c.create(run="run-a")
    real = nc._delete

    def raced_and_lost(argv, account):
        nc.race(account)
        real(argv, account)
        raise subprocess.TimeoutExpired(argv, 300)

    nc._delete = raced_and_lost
    with pytest.raises(subprocess.TimeoutExpired):
        c.delete(mine)
    nc._delete = real
    with pytest.raises(ncc.NcForeignTicketReleased):
        c.delete(mine)  # the pool's retry
    assert c.ledger.halted(1)["released_ticket"] == nc.current[1]


def test_the_clear_halt_command_shows_the_halt_and_needs_yes(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path))
    assert cli.main(["clear-halt", "--account", "1"]) == 0 and "not halted" in capsys.readouterr().out
    cli.ledger().halt(1, "a delete meant for ticket a released ticket b", released_ticket="b")
    assert cli.main(["clear-halt", "--account", "1"]) == 2 and cli.ledger().halted(1) is not None
    assert '"released_ticket": "b"' in capsys.readouterr().out
    assert cli.main(["clear-halt", "--account", "1", "--yes"]) == 0 and cli.ledger().halted(1) is None
    assert cli.ledger().open_rows() == []  # a halt row is no ticket


def test_open_rows_of_an_earlier_version_are_listed_once_for_the_operator(tmp_path):
    events = []
    ledger = ncc.Ledger(tmp_path / "tickets.json")
    ledger.add(1, "old-open", run="run-x")  # as the earlier version wrote it: no order
    ledger.add(1, "old-ended", run="run-x", ended=True)
    provider = NCProvider([1], "run-x", ledger=ledger, allowed={1}, run=Nc(),
                          event=lambda name, **detail: events.append((name, detail)))
    assert provider.order(1) == "new1"
    assert sorted(provider.run_tickets()) == [(1, "new1"), (1, "old-open")]
    assert provider.run_tickets() and [(n, d["tickets"]) for n, d in events] == [("legacy_ledger_rows", ["1:old-open"])]
