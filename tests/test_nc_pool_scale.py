"""``pool-scale`` / ``yeto run scale``: the sampler target of a running NC pool
(yeto.launch.nc.lease_pool / pool_run), over the fake National Compute of
test_nc_elastic. Nothing here orders, calls a real tool or opens a socket."""

from __future__ import annotations

import json
import subprocess

import pytest

from tests.test_nc_elastic import Clock, Cloud, _names, _pool, _run
from tests.test_nc_pool_run import Signals, _cfg
from yeto.launch.nc import cli, pool_run
from yeto.launch.pool.island_pool import GONE, READY, IslandPool


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("YETO_NC_ACCOUNTS", "1,2,3,4")

    def no_subprocess(*a, **k):
        raise AssertionError(f"no subprocess in these tests: {a}")

    monkeypatch.setattr(subprocess, "run", no_subprocess)


DISPENSE = {"backend": "store", "namespace": "run-a/dispense"}


def _scaled_pool(tmp_path, *, target=2, accounts=(1, 2, 3, 4), **extra):
    if "data_slots" not in extra:  # an elastic pool takes its tasks from the store dispenser
        extra.setdefault("sampler", {"dispense": DISPENSE})
    cfg, path = _cfg(tmp_path, target=target, accounts=list(accounts), **extra)
    clock = Clock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud, target=target, accounts=accounts)
    pool.scale_path = pool_run.scale_path(cfg)
    pool.state_path = cli.pool_state_path(cfg)
    pool.state_path.parent.mkdir(parents=True, exist_ok=True)
    _run(pool, clock, clock.t + 1800)
    assert len([m for m in pool.members if m.state == READY]) == target
    pool.save(pool.state_path)
    return cfg, path, clock, cloud, pool


def _ready(pool):
    return [m for m in pool.members if m.state == READY]


def test_a_higher_target_orders_at_the_next_tick(tmp_path, capsys):
    cfg, path, clock, cloud, pool = _scaled_pool(tmp_path)
    orders = len(cloud.kinds("order"))
    assert cli.main(["pool-scale", "--config", path, "--target", "3"]) == 0
    assert "target 2 -> 3 (scale up)" in capsys.readouterr().out
    pool.tick()
    assert len(cloud.kinds("order")) == orders + 1 and pool.config.target == 3
    assert "target_changed" in _names(pool)
    _run(pool, clock, clock.t + 1800)
    assert len(_ready(pool)) == 3


def test_a_lower_target_fences_drains_releases_the_newest_and_orders_nothing(tmp_path):
    cfg, path, clock, cloud, pool = _scaled_pool(tmp_path, target=3)
    newest = max(_ready(pool), key=lambda m: m.created)
    others = [m for m in _ready(pool) if m is not newest]
    orders = len(cloud.kinds("order"))
    assert pool_run.scale_pool(cfg, path, 2) == 0
    _run(pool, clock, clock.t + 3600)
    assert newest.state == GONE and newest.end_reason == "scaled down"
    assert [e[1] for e in cloud.kinds("fence")] == [newest.ticket]
    assert [e[1] for e in cloud.kinds("drain")] == [newest.ticket]
    assert cloud.tickets[newest.ticket]["state"] == "ended"
    assert all(m.state == READY for m in others)  # the others are untouched
    assert len(cloud.kinds("order")) == orders and not pool.pending  # no refill, no handoff
    ev = [e for e in pool.events.events if e["event"] == "nc_elastic_scaled_down"]
    assert len(ev) == 1 and ev[0]["ticket"] == newest.ticket and ev[0]["ended"] is True


def test_retire_picks_the_named_node(tmp_path):
    cfg, path, clock, cloud, pool = _scaled_pool(tmp_path, target=3)
    oldest = min(_ready(pool), key=lambda m: m.created)
    assert pool_run.scale_pool(cfg, path, 2, retire=[oldest.ticket]) == 0
    pool.tick()
    assert oldest.state == GONE and len(_ready(pool)) == 2
    # the retire list is applied once: a later tick does not touch anything
    pool.tick()
    assert len(_ready(pool)) == 2 and len(cloud.kinds("drain")) == 1


def test_scale_to_zero_releases_every_node_and_below_zero_is_refused(tmp_path, capsys):
    cfg, path, clock, cloud, pool = _scaled_pool(tmp_path)
    assert cli.main(["pool-scale", "--config", path, "--target", "-1"]) == 2
    assert "refused" in capsys.readouterr().out and not pool_run.scale_path(cfg).exists()
    assert pool_run.scale_pool(cfg, path, 0) == 0
    _run(pool, clock, clock.t + 600)
    assert not pool.live() and not pool.stopped
    assert len(cloud.kinds("order")) == 2


@pytest.mark.parametrize("extra,target,match", [
    ({}, 5, "accounts"),
    ({"elastic": {"max_nodes": 2}}, 3, "max_nodes"),
    ({"elastic": {"max_orders": 2}}, 3, "max_orders"),
    ({"elastic": {"max_run_spend_ucr": 1}}, 3, "max_run_spend_ucr"),
    ({"shard_limit": 2}, 3, "shard_limit"),
    ({"data_slots": 2}, 3, "data_slots"),
])
def test_scale_up_past_a_limit_is_refused_before_any_order(tmp_path, capsys, extra, target, match):
    cfg, path, clock, cloud, pool = _scaled_pool(tmp_path, **extra)
    orders = len(cloud.kinds("order"))
    assert cli.main(["pool-scale", "--config", path, "--target", str(target)]) == 2
    out = capsys.readouterr().out
    assert "refused, nothing changed" in out and match in out
    assert not pool_run.scale_path(cfg).exists()
    pool.tick()
    assert len(cloud.kinds("order")) == orders and pool.config.target == 2


def test_a_syncer_path_pool_scales_up_on_the_syncers_dispenser(tmp_path):
    """data_path syncer: the samplers claim from the syncer's task dispenser
    (prompts.dispense, tcp), so a scale up repeats no task and is accepted; without
    a dispenser it is refused as on the r2 path."""
    from tests.test_nc_pool_run import SPEC  # noqa: PLC0415

    syncer = {"data_path": "syncer", "syncer": {"address": "203.0.113.5:7000"}}
    doc = {"run": "run-a", "target": 1, "members": []}
    spec = dict(SPEC, prompts=dict(SPEC["prompts"], dispense={"seed": 1}))
    cfg, _ = _cfg(tmp_path, spec_doc=spec, accounts=[1, 2, 3], **syncer)
    assert cli.pool_has_dispenser(cfg)
    assert pool_run.scale_refusal(cfg, doc, 2, []) is None
    cfg, _ = _cfg(tmp_path, spec_doc=SPEC, accounts=[1, 2, 3], **syncer)
    assert not cli.pool_has_dispenser(cfg)
    assert "dispenser" in pool_run.scale_refusal(cfg, doc, 2, [])
    store = dict(SPEC, prompts=dict(SPEC["prompts"], dispense={"seed": 1}))
    cfg, _ = _cfg(tmp_path, spec_doc=store, accounts=[1, 2, 3])  # r2: a tcp section is no store ledger
    assert not cli.pool_has_dispenser(cfg)


def test_retire_of_an_unknown_ticket_or_without_a_lower_target_is_refused(tmp_path, capsys):
    cfg, path, clock, cloud, pool = _scaled_pool(tmp_path)
    assert pool_run.scale_pool(cfg, path, 1, retire=["nope"]) == 2
    assert pool_run.scale_pool(cfg, path, 2, retire=[pool.members[0].ticket]) == 2
    assert not pool_run.scale_path(cfg).exists()


def test_the_target_outlives_a_pool_restart(tmp_path):
    cfg, path, clock, cloud, pool = _scaled_pool(tmp_path, target=3)
    assert pool_run.scale_pool(cfg, path, 2) == 0
    _run(pool, clock, clock.t + 600)
    pool.save(pool.state_path)
    drains = len(cloud.kinds("drain"))
    # a new pool process: config target 3, the scale file says 2
    again = IslandPool.restore(pool.state_path, cli.pool_config(cfg, allowed=None), cloud, clock=clock,
                               log=lambda m: None, events=pool.events)
    again.scale_path = pool_run.scale_path(cfg)
    orders = len(cloud.kinds("order"))
    _run(again, clock, clock.t + 1800)
    assert again.config.target == 2 and len(cloud.kinds("order")) == orders
    assert len(cloud.kinds("drain")) == drains  # the applied retire is not replayed


def test_scale_runs_while_the_loop_holds_the_pool_lock_and_never_takes_it(tmp_path, capsys):
    cfg, path, clock, cloud, pool = _scaled_pool(tmp_path)
    seen = []

    class ScaleOnWait(Signals):
        def _wait(self, timeout):
            clock.t += timeout
            if not seen:
                assert pool_run.scale_pool(cfg, path, 3) == 0
                seen.append(pool_run.lock_holder(pool_run.lock_path(cfg)))
            elif len(seen) < 40:
                seen.append(None)
            else:
                self.name = "SIGTERM"
            return False

    pool_run.run_pool(cfg, path, interval=60, pool=pool, signals=ScaleOnWait())
    out = capsys.readouterr().out
    assert seen[0]["role"] == "pool"  # the loop kept its lock while pool-scale ran
    assert "applies it at its next tick" in out
    assert pool.config.target == 3 and len(cloud.kinds("order")) == 3


def test_concurrent_scale_commands_serialize_on_their_own_lock(tmp_path):
    cfg, path, clock, cloud, pool = _scaled_pool(tmp_path)
    held = pool_run.acquire_lock(pool_run.lock_path(cfg), run="run-a")  # a pool loop holds it
    try:
        assert pool_run.scale_pool(cfg, path, 3) == 0
        assert pool_run.scale_pool(cfg, path, 1) == 0
    finally:
        held.close()
    doc = pool_run.read_scale(cfg)
    assert doc["target"] == 1 and doc["seq"] == 2 and doc["previous"] == 3


def test_status_shows_target_and_nodes(tmp_path, capsys):
    cfg, path, clock, cloud, pool = _scaled_pool(tmp_path)
    pool_run.scale_pool(cfg, path, 3)
    capsys.readouterr()
    pool_run.pool_status(cfg, path)
    doc = json.JSONDecoder().raw_decode(capsys.readouterr().out)[0]
    assert doc["target"] == 3 and doc["nodes"] == 2 and doc["samplers"] == "2/3"


def test_yeto_run_scale_wraps_pool_scale(tmp_path, monkeypatch):
    from yeto.rl import up  # noqa: PLC0415

    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "rl-up")
    calls = []
    monkeypatch.setattr(up.subprocess, "run",
                        lambda argv, **k: calls.append(argv) or subprocess.CompletedProcess(argv, 0))
    run = up.Run({"run": "r", "store": "s3://b/runs/r", "pool": {"config": str(tmp_path / "pool.json")}})
    assert run.scale(3, ["t1"]) == 0
    assert calls[0][-6:] == ["--config", str(tmp_path / "pool.json"), "--target", "3", "--retire", "t1"]
    assert calls[0][1:4] == ["-m", "yeto.launch.nc", "pool-scale"]


def test_the_scale_file_carries_the_run_and_another_runs_file_is_ignored(tmp_path):
    cfg, path, clock, cloud, pool = _scaled_pool(tmp_path)
    pool.scale_run = cli.run_id(cfg)
    assert pool_run.scale_pool(cfg, path, 3) == 0
    assert json.loads(pool_run.scale_path(cfg).read_text())["run"] == cli.run_id(cfg)
    assert pool_run.read_scale(cfg)["target"] == 3
    # a file a previous run on the same shard_base left behind: ignored by both readers
    pool_run.scale_path(cfg).write_text(json.dumps({"run": "old-run", "target": 7, "seq": 9}))
    assert pool_run.read_scale(cfg) is None
    pool.apply_scale()
    assert pool.config.target == 2 and pool.scale_seq == 0
    pool_run.scale_path(cfg).write_text(json.dumps({"target": 7, "seq": 9}))  # no run id
    assert pool_run.read_scale(cfg) is None
    pool.apply_scale()
    assert pool.config.target == 2


def test_pool_stop_removes_the_scale_file(tmp_path):
    cfg, path, clock, cloud, pool = _scaled_pool(tmp_path)
    assert pool_run.scale_pool(cfg, path, 2) == 0
    assert pool_run.scale_path(cfg).exists()
    assert pool_run.stop_pool(cfg, path, ops=cloud, sleep=lambda s: None) == 0
    assert not pool_run.scale_path(cfg).exists()
