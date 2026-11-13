"""``delete-ticket``: one ticket of the pool's own run, never another run's, another
account's or a protected (production) account's; a dry run unless ``--yes``. A fake
client stands in for National Compute: no tool is run, no socket opened."""

from __future__ import annotations

import json
import subprocess

import pytest

from tests.test_nc_pool_run import _cfg
from yeto.launch.nc import cli, pool_run


class FakeClient:
    def __init__(self, account, calls, result):
        self.account, self.calls, self.result = account, calls, result

    def delete(self, ticket):
        self.calls.append((self.account, ticket))
        return dict(self.result, ticket=ticket)


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("YETO_NC_ACCOUNTS", "1,4,5")

    def no_subprocess(*a, **k):
        raise AssertionError(f"no subprocess in these tests: {a}")

    monkeypatch.setattr(subprocess, "run", no_subprocess)


@pytest.fixture
def calls(monkeypatch):
    made = []
    result = {"state": "ended", "end_reason": "released"}
    monkeypatch.setattr(cli, "client", lambda account, **k: FakeClient(account, made, result))
    return made



def _ledger(account, ticket, run="run-a"):
    cli.ledger().add(account, ticket, run=run)


def _events(cfg):
    path = cli.pool_events_path(cfg)
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _main(path, ticket, *extra):
    return cli.main(["delete-ticket", "--config", path, "--ticket", ticket, *extra])


def test_dry_run_prints_and_sends_nothing(tmp_path, calls, capsys):
    cfg, path = _cfg(tmp_path)
    _ledger(4, "t-own")
    assert _main(path, "t-own") == 0
    out = capsys.readouterr().out
    assert "account 4, ticket t-own" in out and "dry run" in out
    assert calls == [] and _events(cfg) == []


def test_yes_deletes_once_confirms_ended_and_writes_the_event(tmp_path, calls, capsys):
    cfg, path = _cfg(tmp_path)
    _ledger(4, "t-own")
    assert _main(path, "t-own", "--yes") == 0
    assert calls == [(4, "t-own")]
    ev = [e for e in _events(cfg) if "ticket_deleted" in json.dumps(e)]
    assert len(ev) == 1 and ev[0]["ticket"] == "t-own" and ev[0]["account"] == 4


def test_a_delete_that_does_not_read_ended_fails_and_is_recorded(tmp_path, monkeypatch, capsys):
    cfg, path = _cfg(tmp_path)
    _ledger(4, "t-own")
    made = []
    monkeypatch.setattr(cli, "client", lambda a, **k: FakeClient(a, made, {"state": "active"}))
    assert _main(path, "t-own", "--yes") == 1
    assert made == [(4, "t-own")]
    assert any("ticket_delete_unconfirmed" in json.dumps(e) for e in _events(cfg))


def test_a_pool_state_member_counts_as_this_runs_ticket(tmp_path, calls):
    cfg, path = _cfg(tmp_path)
    cli.pool_state_path(cfg).write_text(json.dumps({"run": "run-a", "members": [
        {"account": 1, "ticket": "t-member", "shard": "nc-0", "state": "ready"}]}))
    assert _main(path, "t-member", "--yes") == 0
    assert calls == [(1, "t-member")]


@pytest.mark.parametrize("setup", ["other_run", "unknown"])
def test_a_ticket_not_of_this_run_is_refused(tmp_path, calls, capsys, setup):
    _cfg_doc, path = _cfg(tmp_path)
    if setup == "other_run":
        _ledger(4, "t-x", run="run-b")
    assert _main(path, "t-x", "--yes") == 2
    assert "this run did not order it" in capsys.readouterr().err and calls == []


def test_an_account_outside_the_pool_config_is_refused(tmp_path, calls, capsys):
    _c, path = _cfg(tmp_path)  # accounts [1, 4]
    _ledger(5, "t-five")
    assert _main(path, "t-five", "--yes") == 2
    assert "not one of the pool's accounts" in capsys.readouterr().err and calls == []


def test_records_that_disagree_on_the_account_are_refused(tmp_path, calls, capsys):
    cfg, path = _cfg(tmp_path)
    _ledger(4, "t-own")
    cli.pool_state_path(cfg).write_text(json.dumps({"members": [{"account": 1, "ticket": "t-own"}]}))
    assert _main(path, "t-own", "--yes") == 2
    assert "disagree" in capsys.readouterr().err and calls == []


@pytest.mark.parametrize("account", [2, 3])
def test_production_accounts_are_always_refused(tmp_path, calls, capsys, account):
    _c, path = _cfg(tmp_path, accounts=[1, account], protected_accounts=[])  # the config cannot lift it
    _ledger(account, "t-prod")
    assert _main(path, "t-prod", "--yes") == 2
    assert "protected" in capsys.readouterr().err and calls == []


def test_protected_accounts_of_the_config_are_refused(tmp_path, calls, capsys):
    _c, path = _cfg(tmp_path, protected_accounts=[4])
    _ledger(4, "t-own")
    assert _main(path, "t-own", "--yes") == 2
    assert "protected" in capsys.readouterr().err and calls == []


@pytest.mark.parametrize("ticket", ["*", "t-*", "all", "t-a,t-b", "t-a t-b", "t-[ab]", "--all"])
def test_no_pattern_list_or_all(tmp_path, calls, ticket):
    _c, path = _cfg(tmp_path)
    _ledger(4, ticket)  # even a ledger row with that literal text is not a way in
    try:
        rc = _main(path, ticket, "--yes")
    except SystemExit as exit_:  # argparse refuses "--all" as a value
        rc = exit_.code
    assert rc == 2 and calls == []


def test_the_parser_has_no_bulk_option():
    with pytest.raises(SystemExit):
        cli.main(["delete-ticket", "--config", "x", "--all", "--yes"])
    with pytest.raises(SystemExit):
        cli.main(["delete-ticket", "--config", "x", "--yes"])  # --ticket is required


def test_refused_while_the_pool_loop_runs(tmp_path, calls, capsys):
    cfg, path = _cfg(tmp_path)
    _ledger(4, "t-own")
    pool_run.pid_path(cfg).parent.mkdir(parents=True, exist_ok=True)
    pool_run.pid_path(cfg).write_text(json.dumps({"pid": 1, "host": "h", "config": path}))
    lock = pool_run.acquire_lock(pool_run.lock_path(cfg), run="run-a")
    try:
        assert _main(path, "t-own", "--yes") == 2
    finally:
        lock.close()
    assert "pool-scale --remove" in capsys.readouterr().err and calls == []
