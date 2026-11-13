"""``yeto rollout scale / status`` and the run config's sampler pools
(yeto.launch.rollout.entry / pools): parsing, old and new run.json formats, pool
selection, and the old entries (``yeto run scale``, ``nc pool-scale``) going
through the same implementation. No subprocess, no provider call."""

from __future__ import annotations

import argparse
import json
import subprocess
import types

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


def test_a_syncer_pool_member_claims_from_the_dispenser_as_its_own_island(tmp_path):
    """data_path syncer with the syncer's task dispenser (prompts.dispense, tcp): every
    member's spec names its own holder id, the one the pool retires / drains it by, and
    the dispenser address defaults to the syncer's."""
    from yeto.launch.pool.island_pool import Member  # noqa: PLC0415
    from yeto.rl.engine.task_dispense import sampler_holder_id  # noqa: PLC0415

    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"sampler_id": "x", "prompts": {
        "path": "p.jsonl", "input_key": "q", "groups_per_round": 1, "dispense": {"seed": 3, "learner_id": 7}}}))
    cfg = {"spec": str(spec), "run_prefix": "r", "data_path": "syncer", "syncer": {"address": "203.0.113.5:7000"}}
    a = nc_cli.render_spec(cfg, Member(account=1, ticket="t1", shard="m-0"), ["m-0", "m-1"])
    b = nc_cli.render_spec(cfg, Member(account=2, ticket="t2", shard="m-1"), ["m-0", "m-1"])
    assert a["prompts"]["dispense"] == {"seed": 3, "learner_id": sampler_holder_id("m-0"),
                                        "address": "203.0.113.5:7000"}
    assert b["prompts"]["dispense"]["learner_id"] == sampler_holder_id("m-1") != a["prompts"]["dispense"]["learner_id"]
    store = dict(json.loads(spec.read_text()), prompts={"path": "p", "input_key": "q", "groups_per_round": 1,
                                                       "dispense": {"backend": "store", "namespace": "d"}})
    spec.write_text(json.dumps(store))
    assert nc_cli.render_spec(cfg, Member(account=1, ticket="t1", shard="m-0"), ["m-0"])["prompts"]["dispense"] == {
        "backend": "store", "namespace": "d"}


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


def test_run_up_in_process_logs_a_rollout_stall_once(tmp_path, monkeypatch, capsys):
    run_cfg, _, _ = _two_pools(tmp_path)
    run = up.Run(dict(run_cfg, controller="in_process", rollout={"stall_alert_s": 600}))
    views = iter(["no sampler wrote a group or a heartbeat for 700s", "same", None])
    run.controller = type("C", (), {"stalled": lambda self: next(views)})()
    assert run.rollout_stall_alert() and run.rollout_stall_alert() and run.rollout_stall_alert() is None
    out = capsys.readouterr().out
    assert out.count("rollout stalled") == 1 and "progress resumed" in out
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(dict(run_cfg, rollout={"stall_s": 1})))
    with pytest.raises(ValueError, match="unknown rollout keys"):
        up.load_config(str(path))


def test_in_process_drain_on_exit_false_leaves_the_pools_for_the_next_run_up(tmp_path, monkeypatch, capsys):
    """rollout.drain_on_exit false: a process exit drains nothing (the machines bill,
    undriven, with a banner), and the next run up of the config adopts the members."""
    from tests.test_rollout_island_pool import Clock, FakeFencer, FakeProvider, FakeRoute  # noqa: PLC0415
    from yeto.launch.pool import controller as ctl  # noqa: PLC0415
    from yeto.launch.pool.provider import ProviderPoolOps  # noqa: PLC0415

    run_cfg, _, _ = _two_pools(tmp_path)
    cfg = dict(run_cfg, controller="in_process", rollout={"drain_on_exit": False})
    clock = Clock()
    providers = {name: FakeProvider(name, clock) for name in ("nc", "modal")}
    ops = {name: ProviderPoolOps(p, FakeRoute(), FakeFencer()) for name, p in providers.items()}
    banners = []
    monkeypatch.setattr(up.atexit, "register", lambda fn: banners.append(fn))
    run = up.Run(cfg)
    c = run.start_controller(ops=ops, thread_cls=_NoThread)
    for _ in range(3):
        c.tick()
        clock.t += 60
    tickets = {name: [m.ticket for m in pool.live()] for name, pool in c.pools.items()}
    assert all(tickets.values()) and not c.drain_on_exit
    assert ctl.drain_active("process exit") == {}  # the exit drains nothing
    assert not any(p.released for p in providers.values())
    banners[0]()
    assert "STILL BILLING, undriven" in capsys.readouterr().out
    for slot in c.slots.values():  # the process is gone: its pool locks with it
        slot.lock.close()
        slot.lock = None
    with ctl._ACTIVE_LOCK:  # noqa: SLF001
        ctl._ACTIVE.remove(c)  # noqa: SLF001
    again = up.Run(cfg).start_controller(ops=ops, thread_cls=_NoThread)
    try:
        assert {name: [m.ticket for m in pool.live()] for name, pool in again.pools.items()} == tickets
        again.tick()
        assert not any(p.released for p in providers.values()) and \
            sum(len(p.orders) for p in providers.values()) == 2  # adopted, nothing ordered again
    finally:
        again.stop("test")
    with pytest.raises(ValueError, match="drain_on_exit"):
        up.rollout_section({"rollout": {"drain_on_exit": "no"}})


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


def test_a_finished_trainer_names_and_releases_every_pool(tmp_path, monkeypatch, capsys):
    """SUCCEEDED with two pools: the banner names each pool with its provider, and
    release_nc_on_finish stops every pool."""
    run_cfg, _, _ = _two_pools(tmp_path)
    run = up.Run(run_cfg)
    run.save = lambda: None
    assert run.on_trainer_finished() is False
    out = capsys.readouterr().out
    assert "the sampler pools nc (nc), modal (modal) and the sandboxes still bill" in out
    run = up.Run(dict(run_cfg, budget={"release_nc_on_finish": True}))
    calls = []
    run._cmd = lambda argv, *, log, timeout=None: calls.append(argv) or 0
    monkeypatch.setattr(up, "_alive", lambda pid: False)
    assert run.on_trainer_finished() is True
    assert [a[a.index("--config") + 1] for a in calls if "pool-stop" in a] == run.pool_configs


def test_run_up_drives_a_pool_an_earlier_stop_left_stopped(tmp_path, monkeypatch):
    """run down (or a process exit) drains the pools and saves them stopped; the next
    run up must order samplers again instead of skipping the stopped pool forever."""
    from tests.test_rollout_island_pool import Clock, FakeFencer, FakeProvider, FakeRoute  # noqa: PLC0415
    from yeto.launch.pool.provider import ProviderPoolOps  # noqa: PLC0415

    run_cfg, _, _ = _two_pools(tmp_path)
    cfg = dict(run_cfg, controller="in_process")
    clock = Clock()
    providers = {name: FakeProvider(name, clock) for name in ("nc", "modal")}
    ops = {name: ProviderPoolOps(p, FakeRoute(), FakeFencer()) for name, p in providers.items()}
    c = up.Run(cfg).start_controller(ops=ops, thread_cls=_NoThread)
    c.tick()
    assert all(pool.live() for pool in c.pools.values())
    c.stop("process exit")
    assert all(pool.stopped and not pool.live() for pool in c.pools.values())
    again = up.Run(cfg).start_controller(ops=ops, thread_cls=_NoThread)
    try:
        assert not any(pool.stopped for pool in again.pools.values())
        again.tick()
        assert all(pool.live() for pool in again.pools.values())
        assert sum(len(p.orders) for p in providers.values()) == 4
    finally:
        again.stop("test")


def test_run_up_drains_the_in_process_pools_before_main_returns(tmp_path, monkeypatch):
    """The exit drain runs in main (a trainer breaker returns from monitor), not only in
    the atexit hook: at interpreter shutdown the Modal client is already closed."""
    from tests.test_rollout_island_pool import Clock, FakeFencer, FakeProvider, FakeRoute  # noqa: PLC0415
    from yeto.launch.pool import controller as ctl  # noqa: PLC0415
    from yeto.launch.pool.provider import ProviderPoolOps  # noqa: PLC0415

    run_cfg, _, _ = _two_pools(tmp_path)
    run = up.Run(dict(run_cfg, controller="in_process"))
    clock = Clock()
    providers = {name: FakeProvider(name, clock) for name in ("nc", "modal")}
    ops = {name: ProviderPoolOps(p, FakeRoute(), FakeFencer()) for name, p in providers.items()}
    monkeypatch.setattr(up, "Run", lambda cfg: run)
    monkeypatch.setattr(up, "load_config", lambda path: {})
    monkeypatch.setattr(up.signal, "signal", lambda *a: None)
    monkeypatch.setattr(run, "sandbox_prepare", lambda: True)
    monkeypatch.setattr(run, "pool", lambda: run.start_controller(ops=ops, thread_cls=_NoThread).tick())
    monkeypatch.setattr(run, "trainer", lambda: None)
    monkeypatch.setattr(run, "dashboard", lambda: None)
    seen = {}
    monkeypatch.setattr(run, "monitor", lambda: seen.update(live=sum(len(p.live()) for p in run.controller.pools.values())))
    args = types.SimpleNamespace(config="x", rl_command="up", skip_preflight=True, dry_run=False, no_monitor=False,
                                 trainer_now=True)
    assert up.main(args) == 0
    assert seen["live"] == 2
    assert run.controller.stopped == "process exit" and run.controller not in ctl.active()
    assert all(not pool.live() for pool in run.controller.pools.values())


# -- run up exit drain and refusals ----------------------------------------------------

def _main_harness(tmp_path, monkeypatch, *, pool=None, cfg_extra=None):
    """``up.main`` over an in-process run of two fake pools; returns (run, providers, args)."""
    from tests.test_rollout_island_pool import Clock, FakeFencer, FakeProvider, FakeRoute  # noqa: PLC0415
    from yeto.launch.pool.provider import ProviderPoolOps  # noqa: PLC0415

    run_cfg, _, _ = _two_pools(tmp_path)
    run = up.Run(dict(run_cfg, controller="in_process", **(cfg_extra or {})))
    clock = Clock()
    providers = {name: FakeProvider(name, clock) for name in ("nc", "modal")}
    ops = {name: ProviderPoolOps(p, FakeRoute(), FakeFencer()) for name, p in providers.items()}
    monkeypatch.setattr(up, "Run", lambda cfg: run)
    monkeypatch.setattr(up, "load_config", lambda path: {})
    monkeypatch.setattr(run, "sandbox_prepare", lambda: True)
    start = lambda: run.start_controller(ops=ops, thread_cls=_NoThread).tick()  # noqa: E731
    monkeypatch.setattr(run, "pool", pool(start) if pool else start)
    monkeypatch.setattr(run, "trainer", lambda: None)
    monkeypatch.setattr(run, "dashboard", lambda: None)
    monkeypatch.setattr(run, "monitor", lambda: None)
    args = types.SimpleNamespace(config="x", rl_command="up", skip_preflight=True, dry_run=False, no_monitor=False,
                                 trainer_now=True, allow_after_forced_stop=False)
    return run, providers, args


@pytest.fixture
def keep_sigterm():
    import signal  # noqa: PLC0415

    old = signal.getsignal(signal.SIGTERM)
    yield
    signal.signal(signal.SIGTERM, old)


def test_a_failed_exit_drain_is_reported_once_and_not_retried_at_exit(tmp_path, monkeypatch, capsys, keep_sigterm):
    from yeto.launch.pool import controller as ctl  # noqa: PLC0415

    run, _, args = _main_harness(tmp_path, monkeypatch)
    calls = []

    def broken_stop(why, **kw):
        calls.append(why)
        raise RuntimeError("ClientClosed")

    real_start = run.start_controller

    def start(**kw):
        c = real_start(**kw)
        monkeypatch.setattr(c, "stop", broken_stop)
        return c

    monkeypatch.setattr(run, "start_controller", start)
    assert up.main(args) == 1  # the drive returned 0; the failed drain makes it 1, it does not raise
    out = capsys.readouterr().out
    assert "CRITICAL" in out and "STILL BE BILLING" in out and "yeto run down" in out
    assert run.controller not in ctl.active()
    ctl.drain_active("process exit")  # what the atexit hook runs
    assert calls == ["process exit"]


def test_a_sigterm_during_the_exit_drain_does_not_cut_it_short(tmp_path, monkeypatch, capsys, keep_sigterm):
    import os  # noqa: PLC0415
    import signal  # noqa: PLC0415

    run, _, args = _main_harness(tmp_path, monkeypatch)
    real_start = run.start_controller

    def start(**kw):
        c = real_start(**kw)
        real_stop = c.stop

        def stop(why, **kw2):
            os.kill(os.getpid(), signal.SIGTERM)  # systemd stop while the drain runs
            return real_stop(why, **kw2)

        monkeypatch.setattr(c, "stop", stop)
        return c

    monkeypatch.setattr(run, "start_controller", start)
    assert up.main(args) == 0
    assert all(not pool.live() for pool in run.controller.pools.values())
    assert "SIGTERM during the exit drain" in capsys.readouterr().out
    assert signal.getsignal(signal.SIGTERM) is up._exit_on_sigterm  # restored after the drain


def test_a_sigterm_while_the_pools_start_still_drains_them(tmp_path, monkeypatch, keep_sigterm):
    def pool(start):
        def go():
            start()
            raise SystemExit(143)  # SIGTERM through _exit_on_sigterm, inside run.pool()
        return go

    run, _, args = _main_harness(tmp_path, monkeypatch, pool=pool)
    with pytest.raises(SystemExit):
        up.main(args)
    assert run.controller.stopped == "process exit"
    assert all(not p.live() for p in run.controller.pools.values())


def test_run_up_refuses_a_stopped_pool_that_still_has_live_members(tmp_path, capsys):
    from tests.test_rollout_island_pool import Clock, FakeFencer, FakeProvider, FakeRoute  # noqa: PLC0415
    from yeto.launch.pool.provider import ProviderPoolOps  # noqa: PLC0415

    run_cfg, _, _ = _two_pools(tmp_path)
    cfg = dict(run_cfg, controller="in_process")
    clock = Clock()
    ops = {n: ProviderPoolOps(FakeProvider(n, clock), FakeRoute(), FakeFencer()) for n in ("nc", "modal")}
    c = up.Run(cfg).start_controller(ops=ops, thread_cls=_NoThread)
    c.tick()
    for name, slot in c.slots.items():  # a stop that has not released its members yet
        slot.pool.stopped = "run down"
        c._save(slot)
        c._close_slot(name, slot)
    with pytest.raises(up.RunUpRefused, match="yeto run down"):
        up.Run(cfg).start_controller(ops=ops, thread_cls=_NoThread)


def test_run_up_refuses_a_restart_that_the_pool_budget_would_refuse(tmp_path):
    from tests.test_rollout_island_pool import Clock, FakeFencer, FakeProvider, FakeRoute  # noqa: PLC0415
    from yeto.launch.pool.provider import ProviderPoolOps  # noqa: PLC0415

    run_cfg, _, nc_path = _two_pools(tmp_path)
    doc = json.loads(open(nc_path).read())
    doc["elastic"] = {**(doc.get("elastic") or {}), "max_orders": 1}
    open(nc_path, "w").write(json.dumps(doc))
    cfg = dict(run_cfg, controller="in_process")
    clock = Clock()
    ops = {n: ProviderPoolOps(FakeProvider(n, clock), FakeRoute(), FakeFencer()) for n in ("nc", "modal")}
    c = up.Run(cfg).start_controller(ops=ops, thread_cls=_NoThread)
    c.tick()
    c.stop("process exit")
    with pytest.raises(up.RunUpRefused, match="max_orders.*WHOLE run"):
        up.Run(cfg).start_controller(ops=ops, thread_cls=_NoThread)


def test_a_budget_refused_order_is_logged_on_the_console():
    from tests.test_rollout_island_pool import Clock, FakeProvider, make_pool, quiet_elastic  # noqa: PLC0415

    clock = Clock()
    pool, _, _ = make_pool(FakeProvider("p", clock), [1, 2], target=1, clock=clock,
                           elastic=quiet_elastic(max_orders=0))
    logs = []
    pool.log = logs.append
    pool.tick()
    pool.tick()
    refused = [line for line in logs if "max_orders 0" in line and "not ordering" in line]
    assert len(refused) == 1  # once per change, not every tick


def test_run_up_refuses_after_a_forced_stop_until_acknowledged(tmp_path, monkeypatch, capsys, keep_sigterm):
    run, providers, args = _main_harness(tmp_path, monkeypatch)
    run.mark("nc_forced_stop", why="spend $600 > $540", rc=0)
    assert up.main(args) == 2
    assert "FORCED" in capsys.readouterr().out and not any(p.orders for p in providers.values())
    args.allow_after_forced_stop = True
    assert up.main(args) == 0
    assert any(p.orders for p in providers.values())
    args.allow_after_forced_stop = False  # acknowledged: later run ups of the run go on
    assert up.forced_stop_refusal(run, False) is None
    assert _parser().parse_args(["run", "up", "--config", "c", "--allow-after-forced-stop"]).allow_after_forced_stop


@pytest.mark.parametrize("how", ["dry_run", "failed_preflight"])
def test_an_allowed_run_up_that_orders_nothing_does_not_acknowledge_the_forced_stop(
        tmp_path, monkeypatch, capsys, keep_sigterm, how):
    """``--allow-after-forced-stop`` with ``--dry-run`` (or a preflight that fails)
    orders nothing, so the forced stop stays unacknowledged: a later plain run up is
    still refused."""
    run, providers, args = _main_harness(tmp_path, monkeypatch)
    run.mark("nc_forced_stop", why="spend $600 > $540", rc=0)
    args.allow_after_forced_stop = True
    args.skip_preflight = False
    monkeypatch.setattr(run, "preflight", lambda: how != "failed_preflight")
    args.dry_run = how == "dry_run"
    assert up.main(args) == (0 if how == "dry_run" else 2)
    assert not any(p.orders for p in providers.values())
    assert "nc_forced_stop_ack" not in run.state["steps"]
    assert up.forced_stop_refusal(run, False) is not None


def test_subprocess_run_up_drives_a_pool_an_earlier_stop_left_stopped(tmp_path, monkeypatch):
    """controller subprocess (the default): run down saves each pool stopped; the next
    run up spawns `nc pool`, which exits at once on a stopped state ("move it away"), so
    no sampler is ordered. run up must clear the stop first, as in_process does."""
    from tests.test_rollout_island_pool import Clock, FakeFencer, FakeProvider, FakeRoute  # noqa: PLC0415
    from yeto.launch.pool.provider import ProviderPoolOps  # noqa: PLC0415

    run_cfg, _, _ = _two_pools(tmp_path)
    clock = Clock()
    providers = {name: FakeProvider(name, clock) for name in ("nc", "modal")}
    ops = {name: ProviderPoolOps(p, FakeRoute(), FakeFencer()) for name, p in providers.items()}
    c = up.Run(dict(run_cfg, controller="in_process")).start_controller(ops=ops, thread_cls=_NoThread)
    c.tick()
    c.stop("run down")
    run = up.Run(dict(run_cfg))
    assert all(run.pool_doc(i)["stopped"] for i in range(2))
    spawned = []
    monkeypatch.setattr(run, "_spawn", lambda name, argv: spawned.append(name) or 1)
    run.pool()
    assert len(spawned) == 2
    assert not any(run.pool_doc(i)["stopped"] for i in range(2))


def test_subprocess_run_up_refuses_a_stopped_pool_with_live_members(tmp_path, monkeypatch):
    from tests.test_rollout_island_pool import Clock, FakeFencer, FakeProvider, FakeRoute  # noqa: PLC0415
    from yeto.launch.pool.provider import ProviderPoolOps  # noqa: PLC0415

    run_cfg, _, _ = _two_pools(tmp_path)
    clock = Clock()
    providers = {name: FakeProvider(name, clock) for name in ("nc", "modal")}
    ops = {name: ProviderPoolOps(p, FakeRoute(), FakeFencer()) for name, p in providers.items()}
    c = up.Run(dict(run_cfg, controller="in_process")).start_controller(ops=ops, thread_cls=_NoThread)
    c.tick()
    c.stop("run down")
    run = up.Run(dict(run_cfg))
    path = nc_cli.pool_state_path(run.pools[0].load())
    doc = json.loads(path.read_text())
    doc["members"][0]["state"] = "ready"  # stopped, but a release has not finished
    path.write_text(json.dumps(doc))
    monkeypatch.setattr(run, "_spawn", lambda name, argv: 1)
    with pytest.raises(up.RunUpRefused):
        run.pool()
