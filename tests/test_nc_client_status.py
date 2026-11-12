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
    "an error document on stdout, exit 1": answer({"ok": False, "error": "读取当前订单失败：502"}, code=1),
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
    run = Script(status_of(None), status_of(ACTIVE), status_of(released))
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
