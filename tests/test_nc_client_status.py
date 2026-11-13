"""The National Compute client acts only on a status answer that reports the ticket
(yeto.launch.nc.client): CPU tests over a fake ``nc-query`` / ``nc-manage``."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from yeto.launch.nc import client as ncc
from yeto.launch.pool import status as el

NODE = {"public_ip": "192.0.2.10", "ssh_user": "ubuntu", "host_key": "k"}
ACTIVE = {"id": "abc", "state": "active", "node": NODE, "lease_until": "2026-11-10T09:35:03Z",
          "preempt_at": None, "gpus": 8, "rate_ucr_per_gpu_hour": 3_000_000, "others": []}


def answer(doc=None, *, code=0, stdout=None, stderr=""):
    return SimpleNamespace(returncode=code, stdout=json.dumps(doc) if stdout is None else stdout, stderr=stderr)


def status_of(ticket):
    """What ``nc-query status`` prints on success (``ticket`` None: the account has no order)."""
    return answer({"ok": True, "account": 1, "balance": {"balance_ucredits": 10**9}, "ticket": ticket,
                   "hasTicket": bool(ticket and ticket["state"] in ncc.OPEN_STATES)})


class Script:
    """``nc-query status`` answers in order (the last one repeats); ``nc-manage`` is recorded."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.manage = []

    def __call__(self, argv, **kw):
        if argv[0] == "nc-query":
            return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        self.manage.append(list(argv))
        return answer({"ok": True, "released": True})


@pytest.fixture(autouse=True)
def _tools(monkeypatch):
    monkeypatch.setenv("NC_MANAGE", "nc-manage")
    monkeypatch.setenv("NC_QUERY", "nc-query")


def client(run, tmp_path):
    return ncc.NcClient(1, ncc.Ledger(tmp_path / "tickets.json"), run=run, sleep=lambda s: None)


UNTRUSTED = {
    "no output, exit 0": answer(stdout=""),
    "an empty document": answer({}),
    "an error document on stdout, exit 1": answer({"ok": False, "error": "reading the current order failed: 502"}, code=1),
    "an error document, exit 0": answer({"ok": False, "error": "gateway"}),
    "a success without a ticket field": answer({"ok": True, "account": 1}),
    "a ticket that is an error body": answer({"ok": True, "ticket": {"detail": "<html>502</html>"}}),
}


@pytest.mark.parametrize("name", sorted(UNTRUSTED))
def test_a_status_answer_that_does_not_report_the_ticket_is_never_read_as_an_ended_lease(name, tmp_path):
    c = client(Script(UNTRUSTED[name]), tmp_path)
    with pytest.raises(ncc.NcError):
        c.ticket_status("abc")
    with pytest.raises(ncc.NcError):  # nor as "the account is free" (an order would follow)
        c.status()


def test_one_untrusted_status_answer_does_not_evict_a_healthy_node(tmp_path):
    """What the pool does with the answer: a raise is a control-plane blip (nothing is
    decided); ``ended`` without a reason is an eviction (fence the shard, drop the node)."""
    c = client(Script(answer(stdout=""), status_of(ACTIVE)), tmp_path)
    cfg = el.ElasticConfig()

    def verdict():
        try:
            status = c.ticket_status("abc")
        except ncc.NcError:
            status = el.LeaseStatus("unknown")  # IslandPool._status
        return el.classify(status, None, now=0.0, unreachable_since=None, cfg=cfg, joined=True).kind

    assert verdict() == el.CONTROL_BLIP
    assert verdict() == el.HEALTHY


def test_a_single_no_order_answer_is_read_again_before_the_lease_counts_as_ended(tmp_path):
    c = client(Script(status_of(None), status_of(ACTIVE)), tmp_path)
    assert c.ticket_status("abc").state == "active"


def test_an_account_without_an_order_reads_ended_and_another_order_reads_ended(tmp_path):
    assert client(Script(status_of(None)), tmp_path).ticket_status("abc").state == "ended"
    other = dict(ACTIVE, id="other")
    assert client(Script(status_of(other)), tmp_path).ticket_status("abc").state == "ended"
    ended = {"id": "abc", "state": "ended", "end_reason": "preempted", "others": []}
    s = client(Script(status_of(ended)), tmp_path).ticket_status("abc")
    assert (s.state, s.end_reason) == ("ended", "preempted")


@pytest.mark.parametrize("name", sorted(UNTRUSTED))
def test_a_release_that_reads_an_untrusted_status_keeps_the_ticket_open_in_the_ledger(name, tmp_path):
    """The ticket is active and billing; the status answer at release time says nothing.
    The release fails (the caller retries); the ledger still lists the ticket as open."""
    run = Script(UNTRUSTED[name])
    c = client(run, tmp_path)
    c.ledger.add(1, "abc", run="run-a")
    with pytest.raises(ncc.NcError):
        c.delete("abc")
    assert [r["ticket"] for r in c.ledger.run_rows("run-a")] == ["abc"]
    assert run.manage == []


def test_a_release_reads_a_no_order_answer_again_and_then_deletes_the_active_ticket(tmp_path):
    released = {"id": "abc", "state": "ended", "end_reason": "released", "others": []}
    run = Script(status_of(None), status_of(ACTIVE), status_of(ACTIVE), status_of(released))
    c = client(run, tmp_path)
    c.ledger.add(1, "abc", run="run-a")
    result = c.delete("abc")
    assert result["state"] == "ended" and not result.get("already")
    assert run.manage == [["nc-manage", "delete", "--account", "1", "--yes"]]
    assert c.ledger.run_rows("run-a") == []


def test_a_release_of_a_ticket_the_account_no_longer_holds_is_recorded_as_already_ended(tmp_path):
    run = Script(status_of(None))
    c = client(run, tmp_path)
    c.ledger.add(1, "abc", run="run-a")
    assert c.delete("abc")["already"] is True
    assert c.ledger.run_rows("run-a") == [] and run.manage == []


# The three release tests below were written for "the delete is confirmed when the
# ticket reads ended, or when two answers in a row show no order". The second half is
# gone: nc-manage delete names no ticket, so "no order" after it does not say which
# ticket it released. A release is now confirmed only when the ticket itself reads
# ended, or when the delete's answer named it. Each script has one more status answer
# than before: the status is read again right before the delete is sent.

def test_one_no_order_answer_after_the_delete_call_does_not_confirm_the_release(tmp_path):
    """delete sent, without effect | status: no order (one wrong answer) | status: active
    -> the ticket still bills; the ledger must not say it ended. (Unchanged expectation;
    the delete now also stays on record as unconfirmed.)"""
    run = Script(status_of(ACTIVE), status_of(ACTIVE), status_of(None), status_of(ACTIVE))
    c = client(run, tmp_path)
    c.ledger.add(1, "abc", run="run-a")
    with pytest.raises(ncc.NcError, match="does not read ended"):
        c.delete("abc")
    assert [r["ticket"] for r in c.ledger.run_rows("run-a")] == ["abc"]
    assert c.ledger.pending_delete(1)["pending_delete"] == "abc" and c.ledger.halted(1) is None


def test_no_order_after_a_delete_confirms_the_release_only_when_the_answer_named_the_ticket(tmp_path):
    """Was ``test_two_no_order_answers_after_the_delete_call_confirm_the_release``: two
    "no order" answers closed the ticket in the ledger. Changed on purpose. When the
    delete's answer names the ticket, they still do. When it does not, what the delete
    released is unknown: the ticket stays open in the ledger and the account is halted."""
    run = Script(status_of(ACTIVE), status_of(ACTIVE), status_of(None))
    named = lambda argv, **kw: (  # noqa: E731
        answer({"ok": True, "released": True, "ticket": {"id": "abc", "state": "ended"}})
        if argv[0] == "nc-manage" else run(argv, **kw))
    c = client(named, tmp_path)
    c.ledger.add(1, "abc", run="run-a")
    assert c.delete("abc")["state"] == "ended" and c.ledger.run_rows("run-a") == []
    assert c.ledger.pending_delete(1) is None and c.ledger.halted(1) is None

    c = client(Script(status_of(ACTIVE), status_of(ACTIVE), status_of(None)), tmp_path / "b")
    c.ledger.add(1, "abc", run="run-a")
    with pytest.raises(ncc.NcDeleteUnconfirmed, match="no order"):
        c.delete("abc")
    assert c.ledger.halted(1)["expected_ticket"] == "abc" and c.ledger.pending_delete(1) is None
    assert [r["ticket"] for r in c.ledger.run_rows("run-a")] == ["abc"]  # not closed on a guess


def test_the_usual_release_is_confirmed_by_the_ticket_reading_ended(tmp_path):
    """The release as National Compute answers it in practice: the ended ticket stays
    the account's ticket. No record is left, nothing is halted."""
    released = {"id": "abc", "state": "ended", "end_reason": "released", "others": []}
    c = client(Script(status_of(ACTIVE), status_of(ACTIVE), status_of(released)), tmp_path)
    c.ledger.add(1, "abc", run="run-a")
    result = c.delete("abc")
    assert (result["state"], result["end_reason"]) == ("ended", "released")
    assert c.ledger.run_rows("run-a") == [] and c.ledger.pending_delete(1) is None and c.ledger.halted(1) is None


def test_a_delete_call_the_tool_refused_is_a_failed_release(tmp_path):
    """(Unchanged expectation; the status is read after the failed call, and the
    ticket, still active, keeps the delete on record.)"""
    run = Script(status_of(ACTIVE))
    run_manage = lambda argv, **kw: (  # noqa: E731
        answer({"ok": False, "error": "gateway"}) if argv[0] == "nc-manage" else run(argv, **kw))
    c = client(run_manage, tmp_path)
    c.ledger.add(1, "abc", run="run-a")
    with pytest.raises(ncc.NcError, match="gateway"):
        c.delete("abc")
    assert [r["ticket"] for r in c.ledger.run_rows("run-a")] == ["abc"]


def test_nc_error_redacts_credentials_in_nc_manage_stderr(tmp_path):
    """A failing ``nc-manage`` call's stderr reaches the error text (and the pool's
    event log): without credentials, with the diagnostics."""
    sig = "Zm9vYmFy" + "c2lnbmF0dXJl"
    secret = "wJalr" + "XUtnFEMI" + "K7MDENG"
    userinfo = "admin:" + "hunt" + "er2pass"
    stderr = (f"upload https://bucket.r2.example/o?X-Amz-Signature={sig} failed\n"
              f"aws_secret_access_key={secret}\nproxy https://{userinfo}@proxy.example refused\n")

    def run(argv, **kw):
        return answer(code=2, stdout="", stderr=stderr)

    with pytest.raises(ncc.NcError) as err:
        client(run, tmp_path)._call(["nc-manage", "delete"])
    msg = str(err.value)
    for value in (sig, secret, userinfo):
        assert value not in msg
    assert "nc-manage exit 2" in msg and "https://bucket.r2.example/o?" in msg and "proxy.example refused" in msg
