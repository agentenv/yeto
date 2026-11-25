"""A pool state with an unsettled move to another pool (capacity broker phase 2): the
machine is no member of the pool any more, yet the pool may still be billed for it. Every
command that reads the state file sees it: pool-stop does not report the pool drained,
another run does not archive the state, pool-status lists it, and the billing banner says
why and what to do. Fake National Compute of test_nc_elastic; nothing here orders."""

from __future__ import annotations

import json
import os
import socket

import pytest

from tests.test_nc_elastic import Clock, Cloud
from tests.test_nc_pool_run import _cfg
from tests.test_nc_pool_run import _env  # noqa: F401 - the autouse fixture (state dir, accounts, no subprocess)
from tests.test_nc_pool_state_owner import _run_b
from yeto.launch.nc import cli, pool_run

MOVE = {"move": "nc-t9.1000", "state": "handed_over", "to": "other-i1", "since": 1.0, "at": 2.0, "account": 4,
        "shard": "nc-7", "box": "a box this pool cannot open", "timeout_s": 1800.0}


def _stopped_with_a_move(tmp_path):
    cfg, path = _cfg(tmp_path)
    state = cli.pool_state_path(cfg)
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps({
        "format": 2, "stopped": "signal SIGTERM", "owner": pool_run.state_owner(cfg), "target": 1,
        "members": [{"account": 4, "ticket": "nc-t9", "shard": "nc-7", "state": "gone"}],
        "moves": {"out": {"nc-t9": MOVE}}}))
    return cfg, path


def test_billed_members_include_a_machine_handed_over_and_not_settled(tmp_path):
    cfg, _path = _stopped_with_a_move(tmp_path)
    doc = json.loads(cli.pool_state_path(cfg).read_text())
    billed = pool_run.billed_members(doc)
    assert [m["ticket"] for m in billed] == ["nc-t9"] and "moving to pool other-i1" in billed[0]["moving"]
    settled = dict(doc, moves={"out": {"nc-t9": dict(MOVE, state="taken")}})
    assert pool_run.billed_members(settled) == [], "a machine its new pool took is not this pool's"


def test_pool_stop_does_not_report_a_pool_with_an_unsettled_move_drained(tmp_path, capsys):
    cfg, path = _stopped_with_a_move(tmp_path)
    cloud = Cloud(Clock())
    assert pool_run.stop_pool(cfg, path, ops=cloud, sleep=lambda s: None) == 1
    out = capsys.readouterr().out
    assert "drained every node" not in out and "NODES STILL BILLING" in out
    assert "moving to pool other-i1" in out and "pool-stop cannot release it" in out
    assert not cloud.kinds("drain"), "nothing released blind: the machine may be its new pool's"


def test_pool_stop_after_the_loop_exited_does_not_report_a_pool_with_an_unsettled_move_drained(tmp_path, capsys):
    """pool-stop asks the running loop to stop; the loop exits and leaves a state whose
    members are all gone but whose move is not settled: not "drained every node"."""
    cfg, path = _stopped_with_a_move(tmp_path)
    state = cli.pool_state_path(cfg)
    left = state.read_text()
    pool_run.pid_path(cfg).write_text(json.dumps({"pid": os.getpid(), "host": socket.gethostname()}))

    def loop_exits(_s):  # the running loop reads the stop file, stops, leaves its state
        state.write_text(left)
        pool_run.pid_path(cfg).unlink(missing_ok=True)

    cloud = Cloud(Clock())
    assert pool_run.stop_pool(cfg, path, ops=cloud, sleep=loop_exits) == 1
    out = capsys.readouterr().out
    assert "drained every node" not in out and "moving to pool other-i1" in out
    assert not cloud.kinds("drain")


def test_another_run_does_not_archive_a_state_with_an_unsettled_move(tmp_path):
    _stopped_with_a_move(tmp_path)
    cfg_b, _ = _run_b(tmp_path)
    with pytest.raises(pool_run.PoolStateOwnerError, match="billed member") as raised:
        pool_run.claim_state(cfg_b)
    assert "nc-t9: moving to pool other-i1" in str(raised.value), "why pool-stop alone does not clear it"


def test_pool_status_lists_the_moving_machine_apart_from_the_nodes(tmp_path, capsys):
    cfg, path = _stopped_with_a_move(tmp_path)
    pool_run.pool_status(cfg, path)
    out = capsys.readouterr().out
    doc = json.loads(out[:out.index("\n}") + 2])
    assert doc["nodes"] == 0 and [m["ticket"] for m in doc["moving"]] == ["nc-t9"]


def test_a_state_without_moves_reads_as_before(tmp_path, capsys):
    cfg, path = _stopped_with_a_move(tmp_path)
    state = cli.pool_state_path(cfg)
    doc = json.loads(state.read_text())
    doc.pop("moves")
    state.write_text(json.dumps(doc))
    assert pool_run.billed_members(doc) == []
    pool_run.pool_status(cfg, path)
    out = capsys.readouterr().out
    assert '"moving"' not in out
