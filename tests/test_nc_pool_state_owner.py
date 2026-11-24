"""The pool state file ``pool-<shard_base>.json`` records the run it belongs to: a later
run that keeps the shard_base in the same state dir archives a stopped state of the
earlier run and starts, refuses one that still runs or bills, and a state with no owner
(an older file) behaves as before. Fake National Compute of test_nc_elastic; nothing
here orders, calls a real tool or opens a socket."""

from __future__ import annotations

import json
import types

import pytest

from tests.test_nc_elastic import Clock, Cloud, _pool, _ready_member
from tests.test_nc_pool_run import Signals, _cfg
from yeto.launch.nc import cli, pool_run
from yeto.launch.pool import controller
from yeto.launch.pool.island_pool import GONE, IslandPool, downgrade_state
from yeto.rl import up

# the autouse fixture of test_nc_pool_run (state dir, accounts, no subprocess)
from tests.test_nc_pool_run import _env  # noqa: F401


def _run_b(tmp_path, **extra):
    """A second run: new run, run_prefix and store, the same shard_base and state dir."""
    other = tmp_path / "b"
    other.mkdir(exist_ok=True)
    return _cfg(other, run="run-b", run_prefix="run-b", **extra)


def _stopped_by_sigterm(tmp_path):
    """Run a's pool loop, stopped by SIGTERM: its state file says stopped, no member left."""
    cfg, path = _cfg(tmp_path)
    clock = Clock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud, accounts=(1, 4))
    _ready_member(pool, clock, cloud)
    assert pool_run.run_pool(cfg, path, interval=60, pool=pool, signals=Signals("SIGTERM", clock=clock)) == 0
    doc = json.loads(cli.pool_state_path(cfg).read_text())
    assert doc["stopped"] == "signal SIGTERM" and not pool_run.billed_members(doc)
    assert doc["owner"] == {"run": "run-a", "run_prefix": "run-a"}
    return cfg, path


def _live_state(tmp_path, *, owner=True):
    """Run a's pool loop died (SIGKILL): its state holds a billed member, not stopped."""
    cfg, path = _cfg(tmp_path)
    clock = Clock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud, accounts=(1, 4))
    m = _ready_member(pool, clock, cloud)
    if owner:
        pool.owner = pool_run.state_owner(cfg)
    state = cli.pool_state_path(cfg)
    state.parent.mkdir(parents=True, exist_ok=True)
    pool.save(state)
    return cfg, path, cloud, m


def _fresh_loop(cfg, path, monkeypatch, cloud=None):
    """``pool --config F`` with the pool restored from the state file (ops: the fake cloud)."""
    clock = Clock()
    cloud = cloud or Cloud(clock)
    monkeypatch.setattr(cli, "pool_ops", lambda c: cloud)
    rc = pool_run.run_pool(cfg, path, interval=60, signals=Signals("SIGTERM"))
    return rc, cloud


# -- another run, same shard_base -------------------------------------------------------

def test_a_new_run_archives_the_stopped_state_of_another_run_and_starts(tmp_path, monkeypatch, capsys):
    cfg_a, _ = _stopped_by_sigterm(tmp_path)
    old = json.loads(cli.pool_state_path(cfg_a).read_text())
    cfg_b, path_b = _run_b(tmp_path)
    assert cli.pool_state_path(cfg_b) == cli.pool_state_path(cfg_a)  # the B6 collision
    capsys.readouterr()
    rc, cloud = _fresh_loop(cfg_b, path_b, monkeypatch)
    out = capsys.readouterr().out
    assert rc == 0 and cloud.kinds("order"), out  # the new pool ordered a node
    assert "this loop starts nothing" not in out and "archived to" in out and "run 'run-a'" in out
    archived = list((cli.pool_state_path(cfg_b).parent / "archive").glob("pool-nc.run-a.*.json"))
    assert len(archived) == 1 and json.loads(archived[0].read_text()) == old  # renamed, never deleted
    doc = json.loads(cli.pool_state_path(cfg_b).read_text())
    assert doc["owner"] == {"run": "run-b", "run_prefix": "run-b"}


def test_the_same_run_name_with_another_run_prefix_is_another_run(tmp_path):
    cfg_a, _ = _stopped_by_sigterm(tmp_path)
    (tmp_path / "c").mkdir()
    cfg_b, _ = _cfg(tmp_path / "c", run_prefix="run-a2")  # run "run-a" again, a new store
    assert pool_run.claim_state(cfg_b) is not None
    assert not cli.pool_state_path(cfg_a).exists()


def test_a_live_state_of_another_run_is_refused_and_left_as_it_is(tmp_path, monkeypatch):
    cfg_a, _, cloud, m = _live_state(tmp_path)
    before = cli.pool_state_path(cfg_a).read_text()
    cfg_b, path_b = _run_b(tmp_path)
    with pytest.raises(SystemExit, match=r"belongs to run 'run-a'.*not stopped.*billed member") as err:
        _fresh_loop(cfg_b, path_b, monkeypatch, cloud=cloud)
    assert m.ticket in str(err.value) and "nothing changed" in str(err.value)
    assert cli.pool_state_path(cfg_a).read_text() == before
    assert not cloud.kinds("drain") and not (cli.pool_state_path(cfg_a).parent / "archive").exists()


def test_a_stopped_state_of_another_run_with_members_left_is_refused(tmp_path):
    cfg_a, _, _cloud, _m = _live_state(tmp_path)
    doc = json.loads(cli.pool_state_path(cfg_a).read_text())
    cli.pool_state_path(cfg_a).write_text(json.dumps(dict(doc, stopped="signal SIGTERM")))
    cfg_b, _ = _run_b(tmp_path)
    with pytest.raises(pool_run.PoolStateOwnerError, match=r"stopped \(signal SIGTERM\) with billed member"):
        pool_run.claim_state(cfg_b)
    assert cli.pool_state_path(cfg_a).exists()


@pytest.mark.parametrize("cmd", ["pool-stop", "pool-retry", "pool-state-downgrade"])
def test_commands_of_another_run_refuse_its_state(tmp_path, capsys, cmd):
    cfg_a, _, cloud, m = _live_state(tmp_path)
    before = cli.pool_state_path(cfg_a).read_text()
    _cfg_b, path_b = _run_b(tmp_path)
    assert cli.main([cmd, "--config", path_b]) == 2
    err = capsys.readouterr().err
    assert f"{cmd}: refused, nothing changed" in err and "belongs to run 'run-a'" in err
    assert cli.pool_state_path(cfg_a).read_text() == before and not cloud.kinds("drain") and m.state != GONE


def test_status_of_another_run_says_whose_state_it_shows(tmp_path, capsys):
    _live_state(tmp_path)
    cfg_b, path_b = _run_b(tmp_path)
    pool_run.pool_status(cfg_b, path_b)
    assert "belongs to run 'run-a'" in capsys.readouterr().err


def test_delete_ticket_of_another_run_does_not_count_its_state_members(tmp_path):
    _cfg_a, _, _cloud, m = _live_state(tmp_path)
    cfg_b, _ = _run_b(tmp_path)
    assert pool_run._pool_state_doc(cfg_b) == {}  # noqa: SLF001
    assert pool_run._pool_state_doc(_cfg_a)["members"][0]["ticket"] == m.ticket  # noqa: SLF001


# -- the same run -----------------------------------------------------------------------

def test_the_same_run_restarting_on_its_stopped_state_keeps_it(tmp_path, monkeypatch, capsys):
    cfg_a, path_a = _stopped_by_sigterm(tmp_path)
    before = cli.pool_state_path(cfg_a).read_text()
    capsys.readouterr()
    rc, cloud = _fresh_loop(cfg_a, path_a, monkeypatch)
    out = capsys.readouterr().out
    assert rc == 0 and not cloud.kinds("order")
    assert cli.pool_state_path(cfg_a).read_text() == before
    assert "says stopped (signal SIGTERM) by an earlier stop of run 'run-a'" in out
    assert "yeto run up" in out and "this one used: nc-0" in out
    assert "move it away to start a new pool" not in out


def test_the_same_run_resumes_its_live_state(tmp_path, monkeypatch):
    cfg_a, path_a, cloud, m = _live_state(tmp_path)
    assert pool_run.claim_state(cfg_a) is None
    rc, _ = _fresh_loop(cfg_a, path_a, monkeypatch, cloud=cloud)
    assert rc == 0 and m.ticket in [e[1] for e in cloud.kinds("drain")]  # its own member: adopted, drained at SIGTERM


# -- a state file with no owner (older yeto) ------------------------------------------

def test_an_old_state_file_without_owner_keeps_todays_behavior(tmp_path, monkeypatch, capsys):
    cfg_a, _ = _stopped_by_sigterm(tmp_path)
    doc = json.loads(cli.pool_state_path(cfg_a).read_text())
    del doc["owner"]
    cli.pool_state_path(cfg_a).write_text(json.dumps(doc))
    cfg_b, path_b = _run_b(tmp_path)
    assert pool_run.foreign_owner(cfg_b, doc) is None and pool_run.claim_state(cfg_b) is None
    capsys.readouterr()
    rc, cloud = _fresh_loop(cfg_b, path_b, monkeypatch)
    out = capsys.readouterr().out
    assert rc == 0 and not cloud.kinds("order") and cli.pool_state_path(cfg_a).exists()
    assert "says stopped (signal SIGTERM) by an earlier stop of a run it does not record" in out


def test_a_running_pool_of_an_old_state_file_is_not_cut_off(tmp_path, monkeypatch, capsys):
    cfg_a, path_a, cloud, m = _live_state(tmp_path, owner=False)
    assert "owner" not in json.loads(cli.pool_state_path(cfg_a).read_text())
    for cmd in ("pool-retry", "pool-state-downgrade"):  # not refused for want of an owner
        cli.main([cmd, "--config", path_a])
        assert "belongs to" not in capsys.readouterr().err
    rc, _ = _fresh_loop(cfg_a, path_a, monkeypatch, cloud=cloud)
    assert rc == 0 and m.ticket in [e[1] for e in cloud.kinds("drain")]
    assert json.loads(cli.pool_state_path(cfg_a).read_text())["owner"] == {"run": "run-a", "run_prefix": "run-a"}


def test_the_owner_survives_restore_and_a_format_1_downgrade(tmp_path):
    cfg_a, _ = _stopped_by_sigterm(tmp_path)
    state = cli.pool_state_path(cfg_a)
    pool = IslandPool.restore(state, cli.pool_config(cfg_a, allowed=None), None)
    assert pool.owner == {"run": "run-a", "run_prefix": "run-a"}
    old, _ = downgrade_state(json.loads(state.read_text()))
    assert old["owner"] == pool.owner and "format" not in old


def test_a_run_without_run_tag_is_told_apart_by_its_run_prefix(tmp_path):
    cfg, _ = _cfg(tmp_path)
    cfg = {k: v for k, v in cfg.items() if k != "run"}
    assert pool_run.state_owner(cfg) == {"run": "nc", "run_prefix": "run-a"}
    doc = {"owner": {"run": "nc", "run_prefix": "run-old"}}
    assert pool_run.foreign_owner(cfg, doc) == doc["owner"]
    assert pool_run.foreign_owner(cfg, {"owner": {"run": "nc"}}) is None


# -- the rollout controller and run up --------------------------------------------------

def test_the_rollout_controller_archives_a_stopped_state_of_another_run(tmp_path):
    _stopped_by_sigterm(tmp_path)
    cfg_b, path_b = _run_b(tmp_path)
    slot = controller.nc_pool(cfg_b, path_b, ops=Cloud(Clock()))
    try:
        assert not slot.pool.stopped and slot.pool.owner == pool_run.state_owner(cfg_b)
    finally:
        slot.lock.close()
    assert list((cli.pool_state_path(cfg_b).parent / "archive").glob("pool-nc.run-a.*.json"))


def test_the_rollout_controller_refuses_a_live_state_of_another_run(tmp_path):
    _live_state(tmp_path)
    cfg_b, path_b = _run_b(tmp_path)
    with pytest.raises(SystemExit, match="belongs to run 'run-a'"):
        controller.nc_pool(cfg_b, path_b, ops=Cloud(Clock()))
    assert pool_run.lock_holder(pool_run.lock_path(cfg_b)) is None  # the lock was let go


def _ref(cfg, path):
    return types.SimpleNamespace(name="nc", config=path, load=lambda: cfg)


def test_run_up_archives_another_runs_stopped_state_instead_of_restarting_it(tmp_path):
    _stopped_by_sigterm(tmp_path)
    cfg_b, path_b = _run_b(tmp_path)
    restarted = []
    fake = types.SimpleNamespace(_restart_stopped_pool=lambda name, pool: restarted.append(name))
    up.Run._restart_stopped_pool_states(fake, [_ref(cfg_b, path_b)])
    assert not restarted and not cli.pool_state_path(cfg_b).exists()
    assert list((cli.pool_state_path(cfg_b).parent / "archive").glob("pool-nc.run-a.*.json"))


def test_run_up_refuses_a_live_state_of_another_run(tmp_path):
    _live_state(tmp_path)
    cfg_b, path_b = _run_b(tmp_path)
    fake = types.SimpleNamespace(_restart_stopped_pool=lambda name, pool: None)
    with pytest.raises(up.RunUpRefused, match="belongs to run 'run-a'"):
        up.Run._restart_stopped_pool_states(fake, [_ref(cfg_b, path_b)])
    assert cli.pool_state_path(cfg_b).exists()


def test_run_up_does_not_read_another_runs_state_as_its_own(tmp_path, monkeypatch):
    cfg_a, _, _cloud, _m = _live_state(tmp_path)
    cfg_b, path_b = _run_b(tmp_path)
    run = types.SimpleNamespace(pools=[_ref(cfg_b, path_b)], env={"YETO_NC_STATE_DIR": str(tmp_path / "state")})
    run._nc_dir = lambda: up.Run._nc_dir(run)
    assert up.Run.pool_doc(run) is None
    run.pools = [_ref(cfg_a, "a.json")]
    assert up.Run.pool_doc(run)["members"]
