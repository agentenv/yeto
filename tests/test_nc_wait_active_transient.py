"""``wait_active`` polls a queued ticket for up to an hour; one failed status read
(a 502, a rate limit, a tool timeout) must not end the wait, because ``up``
deletes the ticket on any wait error and the queue position is lost."""

from __future__ import annotations

import subprocess

import pytest

from yeto.launch.nc import client as ncc
from tests.test_nc_client_status import ACTIVE, Script, answer, status_of  # noqa: F401
from tests.test_nc_client_status import _tools  # noqa: F401  (autouse fixture)

QUEUED = {"id": "abc", "state": "queued", "position": 3, "others": []}
BAD = answer({"ok": False, "error": "429 too many requests"}, code=1)


def _client(run, tmp_path):
    c = ncc.NcClient(1, ncc.Ledger(tmp_path / "tickets.json"), run=run, sleep=lambda s: None)
    c.ledger.add(1, "abc")
    return c


def test_one_failed_status_read_does_not_end_the_wait(tmp_path):
    run = Script(status_of(QUEUED), BAD, status_of(QUEUED), status_of(ACTIVE))
    lease = _client(run, tmp_path).wait_active("abc", log=lambda m: None)
    assert lease.ticket == "abc" and run.manage == []


def test_a_tool_timeout_does_not_end_the_wait(tmp_path):
    seq = iter(["queued", "timeout", "active"])

    def run(argv, **kw):
        step = next(seq)
        if step == "timeout":
            raise subprocess.TimeoutExpired(argv, 300)
        return status_of(QUEUED if step == "queued" else ACTIVE)

    assert _client(run, tmp_path).wait_active("abc", log=lambda m: None).ticket == "abc"


def test_persistent_status_failure_still_raises(tmp_path):
    run = Script(status_of(QUEUED), BAD)
    with pytest.raises(ncc.NcError):
        _client(run, tmp_path).wait_active("abc", log=lambda m: None)


def test_a_foreign_current_ticket_is_not_retried(tmp_path):
    run = Script(status_of({**QUEUED, "id": "other"}))
    with pytest.raises(ncc.NcError, match="expected abc"):
        _client(run, tmp_path).wait_active("abc", log=lambda m: None)
