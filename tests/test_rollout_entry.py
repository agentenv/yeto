"""``yeto rollout scale / status`` and the run config's sampler pools
(yeto.launch.rollout.entry / pools): parsing, old and new run.json formats, pool
selection, and the old entries (``yeto run scale``, ``nc pool-scale``) going
through the same implementation. No subprocess, no provider call."""

from __future__ import annotations

import argparse
import json
import subprocess

import pytest

from tests.test_nc_pool_run import _cfg as _base_cfg
from tests.test_nc_pool_scale import DISPENSE
from yeto.launch.nc import cli as nc_cli, pool_run


def _cfg(tmp_path, **extra):
    extra.setdefault("sampler", {"dispense": DISPENSE})  # an elastic pool takes tasks from the dispenser
    return _base_cfg(tmp_path, **extra)
from yeto.launch.rollout import entry, pools
from yeto.rl import up


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("YETO_NC_ACCOUNTS", "1,2,3,4")
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "rl-up")

    def no_subprocess(*a, **k):
        raise AssertionError(f"no subprocess in these tests: {a}")

    monkeypatch.setattr(subprocess, "run", no_subprocess)


def _parser():
    p = argparse.ArgumentParser(prog="yeto")
    sub = p.add_subparsers(dest="command")
    entry.add_parser(sub)
    up.add_parser(sub)
    return p


def _two_pools(tmp_path):
    nc_cfg, nc_path = _cfg(tmp_path, accounts=[1, 4])
    modal_dir = tmp_path / "m"
    modal_dir.mkdir()
    mcfg, _ = _cfg(modal_dir, accounts=[1, 2, 3], shard_base="mod", run="run-a",
                   provider={"kind": "modal", "image": "i", "secret": "s", "bucket": "b"})
    run_cfg = {"run": "run-a", "store": "s3://b/runs/run-a", "trainer": {"argv": []},
               "samplers": [{"name": "nc", "provider": "nc", "config": nc_path},
                            {"name": "modal", **{k: v for k, v in mcfg.items()}, "usd_per_h": 50}]}
    path = tmp_path / "run.json"
    path.write_text(json.dumps(run_cfg))
    return run_cfg, str(path), nc_path


# -- CLI parsing ------------------------------------------------------------------------

@pytest.fixture
def in_process(monkeypatch):
    """``python -m yeto.launch.nc ...`` (what run scale spawns) runs in this process."""
    calls = []

    def run(argv, **kw):
        assert argv[1:3] == ["-m", "yeto.launch.nc"], argv
        calls.append(argv[3:])
        return subprocess.CompletedProcess(argv, nc_cli.main(argv[3:]))

    monkeypatch.setattr(subprocess, "run", run)
    return calls


def test_rollout_scale_takes_the_run_scale_flags():
    p = _parser()
    a = p.parse_args(["rollout", "scale", "3", "--config", "run.json", "--pool", "nc", "--retire", "t1",
                      "--retire", "t2", "--fence", "--dry-run"])
    assert (a.command, a.rollout_command, a.target, a.pool, a.retire, a.mode, a.dry_run) == \
        ("rollout", "scale", 3, "nc", ["t1", "t2"], "fence", True)
    a = p.parse_args(["rollout", "scale", "--add", "2", "--config", "run.json", "--drain", "--source", "auto"])
    assert (a.target, a.add, a.mode, a.source) == (None, 2, "drain", "auto")
    a = p.parse_args(["rollout", "scale", "--remove", "t9", "--pool-config", "pool.json", "--provider", "modal"])
    assert a.remove == ["t9"] and a.pool_config == "pool.json" and a.provider == "modal" and a.config is None
    a = p.parse_args(["rollout", "status", "--config", "run.json", "--detail"])
    assert a.rollout_command == "status" and a.detail
    for bad in (["rollout", "scale", "1"], ["rollout", "scale", "1", "--config", "a", "--pool-config", "b"],
                ["rollout", "scale", "1", "--config", "a", "--provider", "aws"], ["rollout", "scale", "--config", "a"],
                ["rollout", "scale", "1", "--add", "1", "--config", "a"],
                ["rollout", "scale", "1", "--drain", "--fence", "--config", "a"]):
        with pytest.raises(SystemExit):
            p.parse_args(bad)


def test_run_scale_keeps_its_flags_and_gains_pool_selectors():
    p = _parser()
    a = p.parse_args(["run", "scale", "--config", "run.json", "--samplers", "2", "--retire", "t"])
    assert (a.target, a.retire, a.pool, a.provider, a.dry_run) == (2, ["t"], None, None, False)
    a = p.parse_args(["run", "scale", "--config", "run.json", "--add", "1", "--pool", "modal", "--dry-run"])
    assert a.pool == "modal" and a.dry_run and a.add == 1
    a = p.parse_args(["run", "pause", "--config", "run.json", "--ticket", "t", "--pool", "nc", "--hold-tasks"])
    assert a.pool == "nc" and a.hold_tasks


def test_the_yeto_cli_registers_rollout():
    from yeto import cli  # noqa: PLC0415

    with pytest.raises(SystemExit) as exc:
        cli.main(["rollout", "--help"])
    assert exc.value.code == 0


# -- run.json formats -------------------------------------------------------------------

def test_the_old_pool_key_is_one_pool(tmp_path):
    refs = pools.resolve({"pool": {"config": "~/pool.json", "interval_s": 45}}, tmp_path)
    assert len(refs) == 1 and refs[0].name == "pool" and refs[0].interval_s == 45
    assert refs[0].config.endswith("/pool.json") and "~" not in refs[0].config
    with pytest.raises(ValueError, match="not both"):
        pools.resolve({"pool": {"config": "a"}, "samplers": [{"config": "b"}]}, tmp_path)
    with pytest.raises(ValueError, match="pool.config"):
        pools.resolve({"pool": {}}, tmp_path)


def test_samplers_entries_file_inline_and_overrides(tmp_path):
    run_cfg, _, nc_path = _two_pools(tmp_path)
    refs = pools.resolve(run_cfg, tmp_path / "state-dir")
    assert [r.name for r in refs] == ["nc", "modal"]
    assert refs[0].config == nc_path  # an entry that changes nothing uses its file as it is
    assert refs[1].config == str(tmp_path / "state-dir" / "pools" / "modal.json") and refs[1].usd_per_h == 50
    assert refs[1].kind() == "modal" and refs[1].load()["shard_base"] == "mod"
    # an override of a file is written next to the run state; the file is not touched
    over = pools.resolve({"samplers": [{"config": nc_path, "target": 2}]}, tmp_path / "s2")
    assert over[0].load()["target"] == 2 and json.loads(open(nc_path).read())["target"] == 1
    assert over[0].name == "nc"


def test_samplers_declarations_are_checked(tmp_path):
    _, nc_path = _cfg(tmp_path)
    with pytest.raises(ValueError, match="non-empty list"):
        pools.resolve({"samplers": []}, tmp_path)
    with pytest.raises(ValueError, match="names repeat"):
        pools.resolve({"samplers": [{"config": nc_path}, {"config": nc_path, "target": 2}]}, tmp_path)
    with pytest.raises(ValueError, match="share shard_base"):
        pools.resolve({"samplers": [{"config": nc_path}, {"name": "b", "config": nc_path, "target": 2}]}, tmp_path)
    with pytest.raises(ValueError, match="data_path"):
        pools.resolve({"samplers": [{"config": nc_path, "data_path": "s3"}]}, tmp_path)
    with pytest.raises(ValueError, match="one data path per run"):
        pools.resolve({"syncer": {"address": "198.51.100.7:7000"},
                       "samplers": [{"config": nc_path}, {"name": "b", "config": nc_path, "shard_base": "x",
                                                          "data_path": "syncer"}]}, tmp_path)
    with pytest.raises(ValueError, match="pool config's provider is 'nc'"):
        pools.resolve({"samplers": [{"provider": "sky", "config": nc_path}]}, tmp_path)
    with pytest.raises(ValueError, match="not one of"):
        pools.resolve({"samplers": [{"provider": "aws", "shard_base": "x"}]}, tmp_path)


def test_a_syncer_pool_fences_and_drains_through_member_retire(tmp_path, monkeypatch):
    from yeto.launch.pool.data_path import R2DataPath, SyncerDataPath  # noqa: PLC0415

    base, _ = _cfg(tmp_path)
    ref = pools.resolve({"samplers": [{**base, "provider": "sky", "data_path": "syncer",
                                       "syncer": {"address": "203.0.113.5:7000"}, "accounts": [1, 2]}]},
                        tmp_path / "s")[0]
    cfg = ref.load()
    assert cfg["data_path"] == "syncer" and cfg["provider"] == {"kind": "sky"} and ref.data_path == "syncer"
    assert ref.name == "sky"
    nc_cli.pool_config(dict(cfg, provider={"kind": "sky", "resources": {"cloud": "aws"}}))
    with pytest.raises(ValueError, match="data_path"):
        nc_cli.pool_config(dict(cfg, data_path="relay"))
    with pytest.raises(ValueError, match="syncer.address"):
        nc_cli.pool_config(dict(cfg, syncer={}))
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "k" * 8)
    dp = nc_cli.pool_data_path(cfg, None, None)
    assert isinstance(dp, SyncerDataPath) and dp.address == ("203.0.113.5", 7000) and dp.key == b"k" * 8
    assert isinstance(nc_cli.pool_data_path({}, None, None), R2DataPath)
    calls = []
    retire = lambda addr, holder, inc, **kw: calls.append((addr, holder, kw["mode"])) or {"status": "drained"}  # noqa: E731
    monkeypatch.setattr("yeto.rl.engine.task_dispense.retire_member", retire)
    ops = nc_cli.NcSshOps(cfg, run=lambda *a, **k: None, provider=object())
    member = type("M", (), {"ticket": "t", "shard": "nc-0", "drain_seq": None})()
    ops.fence(member)
    ops.request_drain(member)
    assert ops.drained(member)
    assert [c[2] for c in calls] == ["retire", "drain", "drain"] and calls[0][0] == ("203.0.113.5", 7000)
    from yeto.launch.pool.island_pool import Member  # noqa: PLC0415

    fresh = nc_cli.render_spec(cfg, Member(account=1, ticket="t2", shard="nc-1"), ["nc-0", "nc-1"])
    assert fresh["data_path"] == "syncer" and fresh["syncer"] == "203.0.113.5:7000"
    assert "predecessor_retired" not in fresh
    heir = Member(account=1, ticket="t3", shard="nc-2", replaces_shard="nc-0", predecessor_retired=True)
    assert nc_cli.render_spec(cfg, heir, ["nc-1", "nc-2"])["predecessor_retired"] is True


def test_the_pool_marks_a_successor_whose_predecessor_it_already_retired():
    from tests.test_rollout_island_pool import Clock, FakeProvider, make_pool, ticks  # noqa: PLC0415

    clock = Clock()
    p = FakeProvider("x", clock)
    pool, *_ = make_pool(p, p.slots() if hasattr(p, "slots") else [1, 2], target=1, clock=clock)
    ticks(pool, 3, clock, step=60)
    first = pool.live()[0]
    p.evict(first.ticket)
    ticks(pool, 6, clock, step=60)
    heir = next(m for m in pool.members if m.replaces_shard == first.shard)
    assert heir.predecessor_retired


def test_run_up_reads_both_formats(tmp_path):
    run_cfg, path, nc_path = _two_pools(tmp_path)
    assert up.load_config(path)["samplers"]
    run = up.Run(run_cfg)
    assert run.pool_config == nc_path and len(run.pool_configs) == 2
    assert run._proc_name(0) == "pool" and run._proc_name(1) == "pool-modal"
    old = up.Run({"run": "r2", "store": "s3://b/runs/r2", "trainer": {"argv": []}, "pool": {"config": nc_path}})
    assert old.pool_configs == [nc_path]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"run": "r", "store": "s", "trainer": {}}))
    with pytest.raises(ValueError, match="samplers"):
        up.load_config(str(bad))


def test_run_up_runs_every_pool_command_on_every_pool(tmp_path, monkeypatch):
    run_cfg, _, nc_path = _two_pools(tmp_path)
    run = up.Run(run_cfg)
    calls = []
    run._cmd = lambda argv, *, log, timeout=None: calls.append(argv) or 0
    spawned = []
    run._spawn = lambda name, argv: spawned.append((name, argv)) or 1
    monkeypatch.setattr(up.Run, "budget_problem", lambda self: None)
    monkeypatch.setattr(up.Run, "stall_timeout_s", lambda self: 900.0)
    run.preflight()
    dry = [a for a in calls if "--dry-run" in a and "pool" in a]
    assert [a[a.index("--config") + 1] for a in dry] == run.pool_configs
    run.pool()
    assert [n for n, _ in spawned] == ["pool", "pool-modal"]
    assert spawned[1][1][spawned[1][1].index("--config") + 1] == run.pool_configs[1]
    calls.clear()
    monkeypatch.setattr(up, "_alive", lambda pid: False)
    run.down()
    stops = [a[a.index("--config") + 1] for a in calls if "pool-stop" in a]
    assert stops == run.pool_configs


def test_run_spend_prices_each_pool(tmp_path, monkeypatch):
    run_cfg, _, _ = _two_pools(tmp_path)
    run_cfg["budget"] = {"nc_usd_per_h": 10}
    run_cfg["trainer"]["usd_per_h"] = 0
    run = up.Run(run_cfg)
    now = 100_000.0
    monkeypatch.setattr(up.time, "time", lambda: now)
    run.state["t0"] = now - 3600
    # no pool state yet: each pool's target, every node billed (nc 1 x $10 + modal 1 x $50)
    assert run.spend_usd() == pytest.approx(60.0)
    assert run.nc_node_hours(now) == pytest.approx(2.0)


# -- scale / status ---------------------------------------------------------------------

def _scale(argv):
    return entry.main(_parser().parse_args(["rollout", "scale", *argv]))


def test_scale_picks_the_named_pool_and_refuses_an_ambiguous_one(tmp_path, capsys, in_process):
    run_cfg, path, nc_path = _two_pools(tmp_path)
    modal_cfg = up.Run(run_cfg).pools[1].load()
    assert _scale(["2", "--config", path]) == 2 and not in_process  # two pools, none named
    assert _scale(["2", "--config", path, "--pool", "nope"]) == 2 and not in_process
    assert _scale(["2", "--config", path, "--pool", "nc"]) == 0
    assert in_process[-1][:3] == ["pool-scale", "--config", nc_path]
    assert pool_run.read_scale(json.loads(open(nc_path).read()))["target"] == 2
    assert _scale(["1", "--config", path, "--provider", "modal", "--fence"]) == 0
    assert in_process[-1][-2:] == ["1", "--fence"]
    assert pool_run.read_scale(modal_cfg)["target"] == 1
    assert _scale(["--add", "1", "--config", path, "--pool", "modal"]) == 0
    assert pool_run.read_scale(modal_cfg)["target"] == 2
    assert _scale(["9", "--config", path, "--pool", "nc"]) == 2  # past the accounts: refused by the pool
    assert pool_run.read_scale(json.loads(open(nc_path).read()))["target"] == 2
    scales = up.Run(run_cfg).state["scales"]
    assert [s["pool"] for s in scales] == ["nc", "modal", "modal"]


def test_scale_keeps_the_run_level_budget_check(tmp_path, in_process, capsys):
    run_cfg, path, nc_path = _two_pools(tmp_path)
    run_cfg["budget"] = {"usd": 40, "nc_usd_per_h": 10}
    open(path, "w").write(json.dumps(run_cfg))
    # a new modal node hour is $50 (its usd_per_h), past the $40 budget; an nc one ($10) fits
    assert _scale(["--add", "1", "--config", path, "--pool", "modal"]) == 2 and not in_process
    assert _scale(["--add", "1", "--config", path, "--pool", "nc"]) == 0


def test_scale_dry_run_and_a_pool_config(tmp_path, in_process, capsys):
    cfg, path = _cfg(tmp_path, accounts=[1, 4])
    assert _scale(["2", "--pool-config", path, "--dry-run"]) == 0
    assert pool_run.read_scale(cfg) is None and not in_process  # straight to pool-scale, nothing written
    assert _scale(["2", "--pool-config", path]) == 0 and pool_run.read_scale(cfg)["target"] == 2
    assert _scale(["1", "--pool-config", path, "--provider", "modal"]) == 2
    assert _scale(["1", "--pool-config", path, "--pool", "nc"]) == 2
    assert _scale(["1", "--config", str(tmp_path / "missing.json")]) == 2


def test_status_through_the_main_entry(tmp_path, capsys, in_process):
    run_cfg, path, nc_path = _two_pools(tmp_path)
    assert _scale(["2", "--config", path, "--pool", "nc"]) == 0
    capsys.readouterr()
    assert entry.main(_parser().parse_args(["rollout", "status", "--config", path])) == 0
    doc = json.loads(capsys.readouterr().out)
    assert [r["target"] for r in doc["pools"]] == [2, 1]


def test_run_pause_goes_to_the_pool_holding_the_ticket(tmp_path, in_process, monkeypatch):
    run_cfg, path, nc_path = _two_pools(tmp_path)
    run = up.Run(run_cfg)
    modal_cfg = run.pools[1].load()
    state = nc_cli.pool_state_path(modal_cfg)
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps({"target": 1, "members": [{"ticket": "m-t1", "shard": "mod-0", "state": "ready"}]}))
    assert run.pool_index(ticket="m-t1") == 1 and run.pool_index() is None and run.pool_index(pool="nc") == 0
    seen = []
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: seen.append(argv) or
                        subprocess.CompletedProcess(argv, 0))
    assert up.pause_resume(run, "pause", ["m-t1"], hold_tasks=True) == 0
    assert seen[-1][3:6] == ["pool-pause", "--config", run.pool_configs[1]] and seen[-1][-1] == "--hold-tasks"
    assert up.pause_resume(run, "resume", ["unknown"]) == 2


def test_status_summarises_every_pool(tmp_path, capsys):
    run_cfg, _, nc_path = _two_pools(tmp_path)
    refs = pools.resolve(run_cfg, tmp_path / "s")
    state = nc_cli.pool_state_path(json.loads(open(nc_path).read()))
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps({"target": 1, "members": [{"account": 1, "ticket": "t1", "shard": "nc-0",
                                                                  "state": "ready"}]}))
    assert entry.status(refs) == 1  # an island bills with no loop alive: warned
    doc = json.loads(capsys.readouterr().out)
    by = {r["pool"]: r for r in doc["pools"]}
    assert by["nc"]["provider"] == "nc" and by["nc"]["islands"] == 1 and by["nc"]["ready"] == 1
    assert "pool-stop" in by["nc"]["warning"]
    assert by["modal"]["provider"] == "modal" and by["modal"]["islands"] == 0
    assert doc["total"] == {"target": 2, "islands": 1}
    ctl = by["nc"]["controller"]  # RolloutController.status() over the restored state, read-only
    assert [m["ticket"] for m in ctl["members"]] == ["t1"] and ctl["stopped"] is None and "controller" in by["modal"]
    assert entry.status(refs, provider="sky") == 2


def test_pool_scale_argv_carries_every_flag():
    args = _parser().parse_args(["rollout", "scale", "--remove", "a", "--remove", "b", "--pool-config", "p",
                                 "--drain", "--force-static", "--dry-run", "--source", "auto"])
    assert entry.pool_scale_argv("p", args) == ["pool-scale", "--config", "p", "--remove", "a", "--remove", "b",
                                                "--drain", "--force-static", "--dry-run", "--source", "auto"]


class _NoThread:
    def __init__(self, target, name=None, daemon=None):
        self.target = target

    def start(self):
        pass

    def join(self, timeout=None):
        pass

    def is_alive(self):
        return False


def test_run_up_in_process_drives_every_pool_in_one_controller(tmp_path, monkeypatch):
    from tests.test_rollout_island_pool import Clock, FakeFencer, FakeProvider, FakeRoute  # noqa: PLC0415
    from yeto.launch.pool import controller as ctl  # noqa: PLC0415
    from yeto.launch.pool.provider import ProviderPoolOps  # noqa: PLC0415

    run_cfg, _, _ = _two_pools(tmp_path)
    run = up.Run(dict(run_cfg, controller="in_process"))
    assert run.in_process and not up.Run(run_cfg).in_process
    clock = Clock()
    ops = {name: ProviderPoolOps(FakeProvider(name, clock), FakeRoute(), FakeFencer()) for name in ("nc", "modal")}
    c = run.start_controller(ops=ops, thread_cls=_NoThread)
    try:
        assert set(c.slots) == {"nc", "modal"} and c in ctl.active()
        c.tick()
        st = c.status()["pools"]
        assert st["nc"]["target"] == 1 and len(st["nc"]["members"]) == 1 and len(st["modal"]["members"]) == 1
        assert run.breach() is None or "controller" not in run.breach()
        c.error = RuntimeError("boom")
        assert "controller thread died" in run.breach()
        c.error = None
    finally:
        c.stop("test")
    assert c not in ctl.active()
    with pytest.raises(ValueError, match="controller"):
        _ = up.Run(dict(run_cfg, controller="threads")).in_process


def test_run_up_in_process_refuses_no_monitor(tmp_path, monkeypatch):
    run_cfg, _, _ = _two_pools(tmp_path)
    path = tmp_path / "run-ip.json"
    path.write_text(json.dumps(dict(run_cfg, controller="in_process")))
    args = _parser().parse_args(["run", "up", "--config", str(path), "--no-monitor"])
    assert up.main(args) == 2


def test_yeto_launch_rollout_pool_builds_one_controller_for_the_fleet(tmp_path):
    from tests.test_rollout_island_pool import Clock, FakeFencer, FakeProvider, FakeRoute  # noqa: PLC0415
    from yeto import launcher  # noqa: PLC0415
    from yeto.launch.pool.provider import ProviderPoolOps  # noqa: PLC0415

    run_cfg, _, nc_path = _two_pools(tmp_path)
    modal_path = str(pools.resolve(run_cfg, tmp_path / "s")[1].config)
    args = argparse.Namespace(rollout_pool=[nc_path, modal_path], controller_poll=15)
    assert launcher._rollout_controller(argparse.Namespace(rollout_pool=[])) is None
    clock = Clock()
    names = [json.loads(open(p).read())["shard_base"] for p in (nc_path, modal_path)]
    ops = {n: ProviderPoolOps(FakeProvider(n, clock), FakeRoute(), FakeFencer()) for n in names}
    c = launcher._rollout_controller(args, ops=ops, thread_cls=_NoThread)
    try:
        assert sorted(c.slots) == sorted(names) and c.interval == 15
        c.tick()
        assert all(len(p["members"]) == 1 for p in c.status()["pools"].values())
    finally:
        c.stop("test")


def test_a_run_level_syncer_reaches_every_syncer_pool(tmp_path):
    base, _ = _cfg(tmp_path)
    run_cfg = {"run": "run-k", "syncer": {"address": "198.51.100.7:7000", "key_env": "K"},
               "samplers": [{**base, "provider": "sky", "data_path": "syncer"}]}
    cfg = pools.resolve(run_cfg, tmp_path / "s")[0].load()
    assert cfg["syncer"] == {"address": "198.51.100.7:7000", "key_env": "K"}
    with pytest.raises(ValueError, match="syncer.address"):
        pools.resolve({"samplers": [{**base, "provider": "sky", "data_path": "syncer"}]}, tmp_path / "t")
