"""A pool whose target is 0 (``yeto run scale --samplers 0``, a desired target 0 on the
store control plane): its config is still a valid config of a running pool. ``pool-stop``,
``pool-retry``, ``pool-scale`` back up and a ``run up`` that adopts it take it; only a
pool loop's start keeps refusing a hand-written config with target 0."""

from __future__ import annotations

import json

import pytest

from tests.test_fleet_desired_state import _store_run
from tests.test_nc_pool_run import Cloud, _cfg, _crashed_pool, _pool, _ready_member
from tests.test_rollout_island_pool import Clock, FakeFencer, FakeProvider, FakeRoute
from yeto.launch.nc import cli, pool_run
from yeto.launch.pool import desired as dz
from yeto.launch.pool.provider import ProviderPoolOps
from yeto.rl.engine.object_store import DirectoryObjectStore


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("YETO_NC_ACCOUNTS", "1,2,3,4")


def _at_zero(cfg, path):
    cfg = dict(cfg, target=0)
    with open(path, "w") as f:
        json.dump(cfg, f)
    return cfg


def test_a_pool_stop_of_a_pool_scaled_to_zero_releases_its_members(tmp_path):
    """INT2 r2: `run scale --samplers 0 --fence` left target 0 in the pool config; run down's
    pool-stop refused it (target must be >= 1) and reported the run NOT verified."""
    cfg, path, cloud, m = _crashed_pool(tmp_path)
    cfg = _at_zero(cfg, path)
    assert pool_run.stop_pool(cfg, path, ops=cloud, sleep=lambda s: None) == 0
    assert m.ticket in [e[1] for e in cloud.kinds("drain")]
    doc = json.loads(cli.pool_state_path(cfg).read_text())
    assert doc["stopped"] == "pool-stop" and not pool_run.billed_members(doc)


def test_a_pool_retry_of_a_pool_scaled_to_zero_restarts_its_halted_member(tmp_path, capsys):
    """REV #826: pool-retry (and the loop's retry) check the sampler spec with the config
    they read, target 0 included; it refused it again (target must be >= 1)."""
    cfg, path = _cfg(tmp_path, target=0)
    clock = Clock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud, accounts=(1, 4))
    m = _ready_member(pool, clock, cloud)
    cloud.tickets[m.ticket].update(sampler="exited", exit=2)  # spec rejected
    for _ in range(5):
        pool.tick()
        clock.t += 60
    assert pool.halted and m.config_error
    pool.save(cli.pool_state_path(cfg))
    assert pool_run.retry_pool(cfg, path) == 0
    assert "start it again" in capsys.readouterr().out
    pool_run._retry_in_loop(pool, path)  # noqa: SLF001
    assert [e[1] for e in cloud.kinds("restart")] == [m.ticket] and not pool.halted


def test_a_pool_config_at_target_zero_still_refuses_a_start(tmp_path):
    cfg, _path = _cfg(tmp_path, target=0)
    with pytest.raises(ValueError, match="target must be >= 1"):
        cli.pool_config(cfg)
    assert cli.pool_config(cfg, min_target=0).target == 0
    with pytest.raises(ValueError, match="target must be >= 0"):
        cli.pool_config(dict(cfg, target=-1), min_target=0)


def test_run_up_adopts_a_pool_the_desired_state_scaled_to_zero(tmp_path, monkeypatch):
    from tests.test_rollout_entry import _NoThread  # noqa: PLC0415

    objects = DirectoryObjectStore(tmp_path / "store")
    up, cfg, _ = _store_run(tmp_path, monkeypatch, objects)
    run = up.Run(cfg)
    assert run.scale(0, pool="nc") == 0  # the desired state: nc at target 0
    assert dz.pool_target(dz.sampler_pools(dz.DesiredStore(objects).read())["nc"]) == 0
    clock = Clock()
    ops = {name: ProviderPoolOps(FakeProvider(name, clock), FakeRoute(), FakeFencer()) for name in ("nc", "modal")}
    c = up.Run(cfg).start_controller(ops=ops, thread_cls=_NoThread)
    try:
        assert set(c.slots) == {"nc", "modal"} and c.slots["nc"].pool.config.target == 0
        c.tick()
        assert not c.slots["nc"].pool.live() and c.slots["modal"].pool.live()
    finally:
        c.stop("test end")


def test_run_down_of_a_run_with_a_pool_scaled_to_zero_is_verified(tmp_path, monkeypatch, capsys):
    """INT2 bug 19: after `yeto run scale --samplers 0` the pool config holds target 0; run
    down's pool-stop and pool-status of that pool succeed and the verdict is verified."""
    from tests.test_rollout_entry import _two_pools  # noqa: PLC0415
    from yeto.rl import up  # noqa: PLC0415

    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "rl-up")
    run_cfg, _run_path, nc_path = _two_pools(tmp_path)
    nc_cfg = json.loads(open(nc_path).read())
    _at_zero(nc_cfg, nc_path)
    run = up.Run(dict(run_cfg, controller="in_process"))
    assert nc_path in [str(p) for p in run.pool_configs]
    pool_calls = []

    def cmd(argv, **kw):
        a = [str(x) for x in argv]
        if "-m" in a and a[a.index("-m") + 1] == "yeto.launch.nc":  # pool-stop / pool-status, in this process
            args = a[a.index("-m") + 2:]
            if "--config" in args and args[args.index("--config") + 1] == nc_path:
                pool_calls.append(args[0])
            return cli.main(args)
        return 0  # the trainer's `yeto down`, the dashboard export

    monkeypatch.setattr(run, "_cmd", cmd)
    monkeypatch.setattr(run, "lock_for_down", lambda *a, **kw: True)
    monkeypatch.setattr(run, "sandbox_hosts", lambda: [])
    assert run.down() is True
    assert sorted(pool_calls) == ["pool-status", "pool-stop"]
    assert "down: verified" in capsys.readouterr().out
