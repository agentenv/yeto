"""NC pool: the state file carries a format number. A pool refuses a file of a newer
format with a clear error, and ``pool-state-downgrade`` writes the file back in the older
format before a rollback of yeto (the older code fails on unknown member fields)."""

from __future__ import annotations

import dataclasses
import json

import pytest

from yeto.launch.nc import cli, pool_run
from yeto.launch.nc.lease_pool import PoolConfig
from yeto.launch.pool import island_pool
from yeto.launch.pool import status as el
from yeto.launch.pool.island_pool import FENCING, READY, IslandPool, Member

from test_nc_elastic import Clock, _names, _pool, _ready_member, _run, _state  # noqa: F401
from test_nc_pool_fence_retry import FlakyFence, _lose
from test_nc_pool_run import _cfg

FORMAT_2_MEMBER_FIELDS = {"unreachable_seen", "absent_seen", "fenced_for_restart", "fence_tokens", "fence_pending",
                          "fence_then", "fence_lost", "fence_failures", "fence_since", "continues_from",
                          "fence_wait_since"}


def _saved_pool(tmp_path, **cfg_extra):
    cfg, config_path = _cfg(tmp_path, **cfg_extra)
    clock = Clock()
    cloud = FlakyFence(clock, ready_after=300)
    pool = _pool(clock, cloud, accounts=(1, 4))
    m = _ready_member(pool, clock, cloud)
    state = cli.pool_state_path(cfg)
    state.parent.mkdir(parents=True, exist_ok=True)
    return cfg, config_path, clock, cloud, pool, m, state


def _restore(state, cloud, clock):
    return IslandPool.restore(state, PoolConfig(target=1, accounts=[1, 4], shard_base="nc"), cloud, clock=clock,
                              log=lambda x: None, events=el.EventLog(log=lambda x: None, clock=clock, run="run-a"))


def test_every_member_field_belongs_to_a_state_format():
    """A new Member field is a state format decision: add it to the newest format's list
    here (and bump STATE_FORMAT when an older pool must not read it)."""
    names = {f.name for f in dataclasses.fields(Member)}
    assert names == set(island_pool.FORMAT_1_MEMBER_FIELDS) | FORMAT_2_MEMBER_FIELDS
    assert not set(island_pool.FORMAT_1_MEMBER_FIELDS) & FORMAT_2_MEMBER_FIELDS


def test_the_state_file_carries_its_format_and_a_file_without_one_is_format_1(tmp_path):
    cfg, config_path, clock, cloud, pool, m, state = _saved_pool(tmp_path)
    pool.save(state)
    doc = json.loads(state.read_text())
    assert doc["format"] == island_pool.STATE_FORMAT == 2
    old = {k: v for k, v in doc.items() if k != "format"}
    old["members"] = [{k: v for k, v in x.items() if k in island_pool.FORMAT_1_MEMBER_FIELDS} for x in doc["members"]]
    state.write_text(json.dumps(old))
    again = _restore(state, cloud, clock)  # an upgrade: the older file reads as it is
    assert [(x.ticket, x.state) for x in again.members] == [(m.ticket, READY)]


def test_a_state_file_of_a_newer_format_is_refused_with_the_way_out(tmp_path):
    cfg, config_path, clock, cloud, pool, m, state = _saved_pool(tmp_path)
    doc = pool.snapshot()
    doc["format"] = island_pool.STATE_FORMAT + 1
    doc["members"][0]["a_field_of_the_future"] = 1
    state.write_text(json.dumps(doc))
    with pytest.raises(RuntimeError, match="format 3.*newer yeto.*pool-state-downgrade"):
        _restore(state, cloud, clock)


def test_pool_state_downgrade_writes_the_older_format_and_keeps_the_file_as_it_was(tmp_path, capsys):
    cfg, config_path, clock, cloud, pool, m, state = _saved_pool(tmp_path)
    m.unreachable_seen, m.absent_seen, m.fenced_for_restart = clock.t, True, True
    pool.save(state)
    before = state.read_text()
    assert cli.main(["pool-state-downgrade", "--config", config_path]) == 0
    doc = json.loads(state.read_text())
    assert "format" not in doc and set(doc["members"][0]) == set(island_pool.FORMAT_1_MEMBER_FIELDS)
    assert doc["members"][0]["ticket"] == m.ticket and doc["next_shard"] == pool.next_shard
    backup = state.with_name(state.name + ".format2.bak")
    assert backup.read_text() == before and "written as format 1" in capsys.readouterr().out
    assert cli.main(["pool-state-downgrade", "--config", config_path]) == 0  # again: nothing to do
    assert "format 1 already" in capsys.readouterr().out
    assert _restore(state, cloud, clock).members[0].state == READY  # and this code still reads it


def test_pool_state_downgrade_is_refused_while_a_member_is_fencing(tmp_path, capsys):
    cfg, config_path, clock, cloud, pool, m, state = _saved_pool(tmp_path)
    cloud.fence_errors = 10**6
    _lose(pool, clock, cloud, m)
    assert m.state == FENCING
    pool.save(state)
    before = state.read_text()
    assert cli.main(["pool-state-downgrade", "--config", config_path, "--drop-owed-fences"]) == 2
    out = capsys.readouterr().out
    assert "refused, nothing changed" in out and "`fencing`" in out and m.ticket in out
    assert state.read_text() == before


def test_pool_state_downgrade_gives_up_an_owed_fence_only_when_told_to(tmp_path, capsys):
    cfg, config_path, clock, cloud, pool, m, state = _saved_pool(tmp_path)
    cloud.fence_errors = 10**6
    cloud.tickets[m.ticket].update(state="ended", reason="preempted")  # released, its fence owed
    pool.tick()
    assert m.fence_pending == "evicted"
    pool.save(state)
    before = state.read_text()
    assert cli.main(["pool-state-downgrade", "--config", config_path]) == 2
    assert "still owed" in capsys.readouterr().out and state.read_text() == before
    assert cli.main(["pool-state-downgrade", "--config", config_path, "--drop-owed-fences"]) == 0
    assert f"shard {m.shard} of {m.ticket} is NOT fenced" in capsys.readouterr().out
    assert "format" not in json.loads(state.read_text())


def test_pool_state_downgrade_is_refused_while_a_pool_loop_holds_the_lock(tmp_path, capsys):
    cfg, config_path, clock, cloud, pool, m, state = _saved_pool(tmp_path)
    pool.save(state)
    before = state.read_text()
    lock = pool_run.acquire_lock(pool_run.lock_path(cfg), run="run-a")
    try:
        assert cli.main(["pool-state-downgrade", "--config", config_path]) == 2
    finally:
        lock.close()
    assert "a pool loop is alive" in capsys.readouterr().out and state.read_text() == before
