"""Manual operations take priority over the NC pool's automation: a node a person
releases (``nc down`` or the NC console) is not replaced, a paused node gets no
automatic action, a manual scale holds off an automatic one, and all of it is in the
pool's control file (it outlives a restart). Over the fake National Compute of
test_nc_elastic; nothing here orders, calls a real tool or opens a socket."""

from __future__ import annotations

import json
import subprocess
import threading

import pytest

from tests.test_nc_elastic import _names, _run
from tests.test_nc_pool_scale_drain import NS, _elastic, _events, _ready, _save
from yeto.dashboard.async_rl import AsyncState
from yeto.launch.nc import cli, pool_run
from yeto.launch.nc.lease_pool import GONE, READY, STARTING, LeasePool
from yeto.rl import up
from yeto.rl.engine import store_dispense as sd


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("YETO_NC_ACCOUNTS", "1,4,5")

    def no_subprocess(*a, **k):
        raise AssertionError(f"no subprocess in these tests: {a}")

    monkeypatch.setattr(subprocess, "run", no_subprocess)


def _restart(cfg, pool, cloud, clock):
    again = LeasePool.restore(pool.state_path, cli.pool_config(cfg, allowed=None), cloud, clock=clock,
                              log=lambda m: None, events=pool.events)
    again.scale_path, again.scale_run = pool.scale_path, pool.scale_run
    return again


def _alive(cfg, path):
    """A pool loop of ``cfg`` holds its lock and wrote its pid file (it names the config)."""
    pool_run.pid_path(cfg).parent.mkdir(parents=True, exist_ok=True)
    pool_run.pid_path(cfg).write_text(json.dumps({"pid": 1, "host": "h", "config": path}))
    return pool_run.acquire_lock(pool_run.lock_path(cfg), run="run-a")


# -- a release by a person ---------------------------------------------------------

def test_a_console_release_lowers_the_target_and_orders_no_replacement(tmp_path):
    cfg, path, clock, cloud, pool = _elastic(tmp_path, target=2)
    gone = _ready(pool)[0]
    orders = len(cloud.kinds("order"))
    cloud.tickets[gone.ticket].update(state="ended", reason="released")  # deleted in the NC console
    _run(pool, clock, clock.t + 1800)
    assert gone.state == GONE and len(cloud.kinds("order")) == orders and pool.config.target == 1
    ev = _events(pool, "manual_release_detected")
    assert len(ev) == 1 and ev[0]["ticket"] == gone.ticket and (ev[0]["old"], ev[0]["new"]) == (2, 1)
    assert ev[0]["by"] == "console"
    assert not pool.pending  # nothing handed on
    doc = pool_run.read_scale(cfg)
    assert doc["target"] == 1 and doc["source"] == "manual_release"
    _save(pool)
    again = _restart(cfg, pool, cloud, clock)  # the lowered target outlives a restart
    _run(again, clock, clock.t + 1800)
    assert again.config.target == 1 and len(cloud.kinds("order")) == orders


def test_a_platform_eviction_is_still_replaced(tmp_path):
    cfg, path, clock, cloud, pool = _elastic(tmp_path, target=2)
    lost = _ready(pool)[0]
    orders = len(cloud.kinds("order"))
    cloud.tickets[lost.ticket].update(state="ended", reason="preempted")
    _run(pool, clock, clock.t + 1800)
    assert len(cloud.kinds("order")) == orders + 1 and pool.config.target == 2 and len(_ready(pool)) == 2
    assert not _events(pool, "manual_release_detected")


def test_external_release_replace_keeps_the_old_behaviour(tmp_path):
    cfg, path, clock, cloud, pool = _elastic(tmp_path, target=2, pool_kw={"external_release": "replace"})
    gone = _ready(pool)[0]
    orders = len(cloud.kinds("order"))
    cloud.tickets[gone.ticket].update(state="ended", reason="released")
    _run(pool, clock, clock.t + 1800)
    assert len(cloud.kinds("order")) == orders + 1 and pool.config.target == 2
    assert "released_externally" in _names(pool)


def test_nc_down_of_a_running_pools_member_becomes_a_pool_removal(tmp_path, monkeypatch, capsys):
    cfg, path, clock, cloud, pool = _elastic(tmp_path, target=2)
    victim = _ready(pool)[1]
    monkeypatch.setattr(cli, "down", lambda *a: pytest.fail("deleted behind the pool's back"))
    held = _alive(cfg, path)
    try:
        assert cli.main(["down", "--account", str(victim.account), "--ticket", victim.ticket]) == 0
    finally:
        held.close()
    out = capsys.readouterr().out
    assert "asking the pool to remove it" in out
    doc = pool_run.read_scale(cfg)
    assert doc["target"] == 1 and doc["retire"] == [victim.ticket] and doc["mode"] == "fence"
    assert doc["source"] == "manual"
    orders = len(cloud.kinds("order"))
    _run(pool, clock, clock.t + 1800)
    assert victim.state == GONE and len(cloud.kinds("order")) == orders
    assert _events(pool, "manual_scale")[0]["retire"] == [victim.ticket]


def test_nc_down_without_a_pool_loop_deletes_and_a_restart_does_not_lower_twice(tmp_path, monkeypatch, capsys):
    cfg, path, clock, cloud, pool = _elastic(tmp_path, target=2)
    victim = _ready(pool)[1]
    pool_run.pid_path(cfg).write_text(json.dumps({"pid": 1, "host": "h", "config": path}))
    pool_run.lock_path(cfg).write_text("")  # the lock is free: no loop runs
    deleted = []

    def down(account, ticket):
        deleted.append(ticket)
        cloud.tickets[ticket].update(state="ended", reason="released")

    monkeypatch.setattr(cli, "down", down)
    assert cli.main(["down", "--account", str(victim.account), "--ticket", victim.ticket]) == 0
    assert deleted == [victim.ticket] and "is not running: deleting it here" in capsys.readouterr().out
    doc = pool_run.read_scale(cfg)
    assert doc["target"] == 1 and victim.ticket in doc["manual_released"]
    cloud.released_by_us = lambda t: True  # the ledger says this tool deleted it
    again = _restart(cfg, pool, cloud, clock)
    orders = len(cloud.kinds("order"))
    _run(again, clock, clock.t + 1800)
    assert again.config.target == 1 and len(cloud.kinds("order")) == orders
    assert _events(again, "manual_release_detected")[-1]["by"] == "nc down"


def test_nc_down_of_a_ticket_no_pool_owns_deletes_it(tmp_path, monkeypatch):
    deleted = []
    monkeypatch.setattr(cli, "down", lambda account, ticket: deleted.append(ticket))
    assert cli.main(["down", "--account", "1", "--ticket", "other"]) == 0
    assert deleted == ["other"]


# -- pause / resume -------------------------------------------------------------------

def test_a_paused_node_is_left_alone_and_not_replaced(tmp_path, capsys):
    cfg, path, clock, cloud, pool = _elastic(tmp_path, target=2)
    node = _ready(pool)[0]
    assert cli.main(["pool-pause", "--config", path, "--ticket", node.ticket]) == 0
    pool.tick()
    assert node.paused and _events(pool, "manual_pause")
    orders, restarts = len(cloud.kinds("order")), len(cloud.kinds("restart"))
    cloud.tickets[node.ticket].update(sampler="exited", exit=1)  # its sampler crashes
    _run(pool, clock, clock.t + 3600)
    cloud.tickets[node.ticket]["reachable"] = False  # and the node stops answering
    _run(pool, clock, clock.t + 3600)
    assert len(cloud.kinds("restart")) == restarts and node.state == READY  # no restart, no retire
    assert len(cloud.kinds("order")) == orders  # its absence is not filled
    _save(pool)
    cloud.tickets[node.ticket].update(reachable=True, sampler="exited", exit=1)
    assert cli.main(["pool-resume", "--config", path, "--ticket", node.ticket]) == 0
    pool.tick()
    assert not node.paused and _events(pool, "manual_resume")
    assert len(cloud.kinds("restart")) == restarts + 1  # watched again: the crash is handled


def test_a_paused_node_past_its_lease_end_is_warned_about_not_drained(tmp_path):
    cfg, path, clock, cloud, pool = _elastic(tmp_path, target=2)
    node = _ready(pool)[0]
    assert pool_run.pause_pool(cfg, path, [node.ticket]) == 0
    pool.tick()
    _run(pool, clock, node.lease_until + 60)
    assert node.state == READY and _events(pool, "manual_paused_past_lease_end")
    assert all(m.replaces != node.ticket for m in pool.members)  # no replacement for it
    assert not [d for d in cloud.kinds("drain") if d[1] == node.ticket]


def test_a_pause_survives_a_pool_restart_and_shows_in_pool_status(tmp_path, capsys):
    cfg, path, clock, cloud, pool = _elastic(tmp_path, target=2)
    node = _ready(pool)[0]
    assert pool_run.pause_pool(cfg, path, [node.ticket], hold_tasks=True) == 0
    pool.tick()
    assert node.paused and node.drain_seq  # its sampler claims no new task meanwhile
    _save(pool)
    again = _restart(cfg, pool, cloud, clock)
    restarts = len(cloud.kinds("restart"))
    cloud.tickets[node.ticket].update(sampler="exited", exit=1)
    _run(again, clock, clock.t + 1800)
    v = next(m for m in again.members if m.ticket == node.ticket)
    assert v.paused and len(cloud.kinds("restart")) == restarts
    _save(again)
    capsys.readouterr()
    pool_run.pool_status(cfg, path)
    doc = json.JSONDecoder().raw_decode(capsys.readouterr().out)[0]
    assert doc["manual"]["paused"] == [node.ticket] and doc["manual"]["hold_tasks"] == [node.ticket]
    assert next(b for b in doc["billed"] if b["ticket"] == node.ticket)["paused"] is True
    assert pool_run.pause_pool(cfg, path, [node.ticket], resume=True) == 0
    again.tick()
    assert not v.paused
    assert json.loads(cloud.ledger_objects.read(sd.drain_key(NS, v.shard)))["drain"] is False


def test_pausing_an_unknown_ticket_is_refused(tmp_path, capsys):
    cfg, path, clock, cloud, pool = _elastic(tmp_path, target=1)
    assert pool_run.pause_pool(cfg, path, ["nope"]) == 2
    assert "not a live member" in capsys.readouterr().out


def test_a_paused_node_is_never_an_automatic_scale_down_pick(tmp_path):
    cfg, path, clock, cloud, pool = _elastic(tmp_path, target=2, pool_kw={"scale_down": "fence"})
    newest = max(_ready(pool), key=lambda m: m.created)
    other = next(m for m in _ready(pool) if m is not newest)
    assert pool_run.pause_pool(cfg, path, [newest.ticket]) == 0
    pool.tick()
    _save(pool)
    assert pool_run.scale_pool(cfg, path, 1) == 0
    pool.tick()
    assert newest.state == READY and other.state == GONE


def test_a_pause_does_not_drop_an_unapplied_removal(tmp_path):
    cfg, path, clock, cloud, pool = _elastic(tmp_path, target=2, pool_kw={"scale_down": "fence"})
    a, b = _ready(pool)
    assert pool_run.scale_pool(cfg, path, remove=[a.ticket]) == 0
    assert pool_run.pause_pool(cfg, path, [b.ticket]) == 0  # before the loop's next tick
    pool.tick()
    assert a.state == GONE and b.paused and b.state == READY


# -- manual before automatic -----------------------------------------------------------

def test_an_automatic_scale_waits_for_the_manual_hold(tmp_path, monkeypatch, capsys):
    cfg, path, clock, cloud, pool = _elastic(tmp_path, target=1, elastic={"manual_hold_s": 600})
    assert pool_run.scale_pool(cfg, path, add=1) == 0  # a person
    assert pool_run.scale_pool(cfg, path, 1, source="auto") == 2
    assert "takes priority over automatic scaling" in capsys.readouterr().out
    assert pool_run.read_scale(cfg)["target"] == 2
    now = pool_run.time.time()
    monkeypatch.setattr(pool_run.time, "time", lambda: now + 601)
    assert pool_run.scale_pool(cfg, path, 3, source="auto") == 0
    doc = pool_run.read_scale(cfg)
    assert doc["target"] == 3 and doc["source"] == "auto"
    pool.tick()
    assert pool.config.target == 3 and not _events(pool, "manual_scale")[1:]


def test_concurrent_manual_and_automatic_scales_never_let_auto_override_manual(tmp_path):
    cfg, path, clock, cloud, pool = _elastic(tmp_path, target=1, elastic={"manual_hold_s": 600})
    rcs = {}
    threads = [threading.Thread(target=lambda: rcs.__setitem__("manual", pool_run.scale_pool(cfg, path, 3))),
               threading.Thread(target=lambda: rcs.__setitem__("auto", pool_run.scale_pool(cfg, path, 2,
                                                                                         source="auto")))]
    [t.start() for t in threads]
    [t.join(10) for t in threads]
    assert rcs["manual"] == 0
    doc = pool_run.read_scale(cfg)
    if rcs["auto"] == 0:  # the automatic one came first; the manual one is the last word
        assert doc["seq"] == 2
    assert doc["target"] == 3 and doc["source"] == "manual"


def test_yeto_run_pause_and_scale_source_pass_through(tmp_path, monkeypatch):
    import argparse

    calls = []
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "rl-up")
    monkeypatch.setattr(up.subprocess, "run",
                        lambda argv, **k: calls.append(argv) or subprocess.CompletedProcess(argv, 0))
    (tmp_path / "pool.json").write_text(json.dumps({"target": 1, "accounts": [1], "shard_base": "nc",
                                                    "run": "run-a"}))
    (tmp_path / "run.json").write_text(json.dumps({"run": "r", "store": "s3://b/runs/r", "trainer": {"argv": []},
                                                   "pool": {"config": str(tmp_path / "pool.json")}}))
    p = argparse.ArgumentParser()
    up.add_parser(p.add_subparsers(dest="cmd"))
    args = p.parse_args(["run", "pause", "--config", str(tmp_path / "run.json"), "--ticket", "t1", "--hold-tasks"])
    assert up.main(args) == 0
    assert calls[-1][3:] == ["pool-pause", "--config", str(tmp_path / "pool.json"), "--ticket", "t1",
                             "--hold-tasks"]
    args = p.parse_args(["run", "scale", "--config", str(tmp_path / "run.json"), "--samplers", "0",
                         "--source", "auto"])
    assert up.main(args) == 0 and calls[-1][-2:] == ["--source", "auto"]


# -- the dashboard ----------------------------------------------------------------------

def test_the_dashboard_marks_manual_operations(tmp_path):
    cfg, path, clock, cloud, pool = _elastic(tmp_path, target=2)
    a, b = _ready(pool)
    assert pool_run.pause_pool(cfg, path, [a.ticket]) == 0
    pool.tick()
    cloud.tickets[b.ticket].update(state="ended", reason="released")
    pool.tick()
    state = AsyncState()
    for e in pool.events.events:
        state.fold_pool(e)
    sc = state.view()["pool_scale"]
    assert sc["paused"] == [a.shard] and sc["target"] == 1 and sc["manual"] is True
    assert sc["target_source"] == "manual_release"
    assert {e["event"] for e in sc["manual_events"]} == {"nc_elastic_manual_pause",
                                                          "nc_elastic_manual_release_detected"}
    assert state.pool_exit(b.shard)["event"] == "nc_elastic_scaled_down"
    assert state.pool_exit(a.shard) is None  # paused, not left


def test_a_paused_node_that_the_platform_reclaims_is_replaced(tmp_path):
    cfg, path, clock, cloud, pool = _elastic(tmp_path, target=2)
    node = _ready(pool)[0]
    assert pool_run.pause_pool(cfg, path, [node.ticket]) == 0
    pool.tick()
    orders = len(cloud.kinds("order"))
    cloud.tickets[node.ticket].update(state="ended", reason="preempted")
    _run(pool, clock, clock.t + 1800)
    assert node.state == GONE and not node.paused and len(cloud.kinds("order")) == orders + 1
    assert _events(pool, "manual_pause_ended")
    assert any(m.state in (READY, STARTING) and m.ticket != node.ticket for m in pool.members)
