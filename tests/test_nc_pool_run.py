"""``python -m yeto.launch.nc pool --dry-run`` / ``pool-stop`` / ``pool-status`` and
the pool loop's exits (yeto.launch.nc.pool_run), over the fake National Compute
of test_nc_elastic. Nothing here orders, calls a real tool or opens a socket."""

from __future__ import annotations

import json
import os
import socket
import subprocess
from pathlib import Path

import pytest

from tests.test_nc_elastic import Clock, Cloud, _pool, _ready_member
from yeto.launch.nc import cli, pool_run
from yeto.launch.nc.lease_pool import GONE


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("YETO_NC_ACCOUNTS", "1,4")
    monkeypatch.setenv("NC_MANAGE", "true")
    monkeypatch.setenv("NC_QUERY", "true")

    def no_subprocess(*a, **k):
        raise AssertionError(f"no subprocess in these tests: {a}")

    monkeypatch.setattr(subprocess, "run", no_subprocess)


SPEC = {"event_tape": "/work/sampler/events.jsonl", "prompts": {"groups_per_round": 4}, "num_rollouts": 8,
        "max_backlog_groups": 64, "max_prompt_tokens": 512, "sampling": {"max_tokens": 64, "temperature": 1.0},
        "engine": {"factory": "yeto.rl.adapters.sampler_engines.from_spec:build", "adapter": "fake", "kwargs": {}},
        "reward": "yeto.rl.numeric_answer_reward:score", "reward_kind": "miles"}
EXAMPLE_SPEC = Path(__file__).resolve().parents[1] / "examples/k31/sampler-spec.nc-mi355x-vllm.json"


def _cfg(tmp_path, spec_doc=None, **extra):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(spec_doc if spec_doc is not None else SPEC))
    env = tmp_path / "store.env"
    env.write_text("R2_BUCKET=<bucket>\nAWS_ACCESS_KEY_ID=x\nAWS_SECRET_ACCESS_KEY=y\n")
    env.chmod(0o600)
    cfg = {"target": 1, "accounts": [1, 4], "shard_base": "nc", "run": "run-a", "spec": str(spec),
           "run_prefix": "run-a", "r2_env_file": str(env), **extra}
    path = tmp_path / "pool.json"
    path.write_text(json.dumps(cfg))
    return cfg, str(path)


class Signals(pool_run._Signals):
    """Delivers ``name`` after ``after`` waits (no real signal)."""

    def __init__(self, name=None, after=1, clock=None):
        super().__init__()
        self.deliver, self.after, self.clock = name, after, clock
        self.wake.wait = self._wait

    def install(self):
        pass

    def restore(self):
        pass

    def _wait(self, timeout):
        if self.clock is not None:
            self.clock.t += timeout
        self.after -= 1
        if self.after <= 0 and self.deliver:
            self.name = self.deliver
        return False


# -- dry run --------------------------------------------------------------------

def test_dry_run_prints_orders_and_spend_and_orders_nothing(tmp_path, capsys):
    _doc, path = _cfg(tmp_path, target=2, elastic={"max_run_spend_ucr": 480_000_000})
    assert cli.main(["pool", "--config", path, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "account 1: 1 x mi355x node, shard nc-0" in out and "account 4: 1 x mi355x node, shard nc-1" in out
    assert "steady 48000000 ucr/h" in out
    assert "does NOT end a lease at lease_until" in out and f"pool-stop --config {path}" in out
    assert "nothing ordered" in out
    assert not (tmp_path / "state" / "tickets.json").exists()


def test_dry_run_reports_invalid_configs(tmp_path, capsys):
    _doc, path = _cfg(tmp_path, target=3)
    assert cli.main(["pool", "--config", path, "--dry-run"]) == 2
    assert "one node at a time" in capsys.readouterr().out
    _doc, path = _cfg(tmp_path, model="mi355x")
    (tmp_path / "store.env").chmod(0o644)
    assert cli.main(["pool", "--config", path, "--dry-run"]) == 1
    out = capsys.readouterr().out
    assert "deprecated" in out and "machine_type" in out and "mode 0600" in out
    assert "max_run_spend_ucr is unset" in out


def test_dry_run_refuses_a_model_pull_whose_model_is_not_in_the_bucket(tmp_path, capsys, monkeypatch):
    seen = []
    monkeypatch.setattr(pool_run, "model_problems",
                        lambda uri, env: seen.append(uri) or [f"no {uri}/config.json"])
    _doc, path = _cfg(tmp_path, model_pull={"src_key": "/models/Qwen3-0.6B/"},
                      elastic={"max_run_spend_ucr": 1})
    assert cli.main(["pool", "--config", path, "--dry-run"]) == 1
    out = capsys.readouterr().out
    assert seen == ["s3://<bucket>/models/Qwen3-0.6B"]
    assert "no s3://<bucket>/models/Qwen3-0.6B/config.json: upload the model" in out
    monkeypatch.setattr(pool_run, "model_problems", lambda uri, env: [])
    assert cli.main(["pool", "--config", path, "--dry-run"]) == 0


def test_dry_run_refuses_a_run_prefix_that_repeats_runs(tmp_path, capsys):
    _doc, path = _cfg(tmp_path, run_prefix="runs/run-a", elastic={"max_run_spend_ucr": 1})
    assert cli.main(["pool", "--config", path, "--dry-run"]) == 1
    assert "lands in s3://<bucket>/runs/runs/run-a; drop the leading 'runs/'" in capsys.readouterr().out


def test_dry_run_reports_an_unreadable_model_prefix_and_a_dest_the_sampler_does_not_mount(
        tmp_path, capsys, monkeypatch):
    def denied(uri, env):
        raise PermissionError("AccessDenied")

    monkeypatch.setattr(pool_run, "model_problems", denied)
    _doc, path = _cfg(tmp_path, model_pull={"src_key": "models/m", "dest": "/mnt/nvme/models/m"})
    assert cli.main(["pool", "--config", path, "--dry-run"]) == 1
    out = capsys.readouterr().out
    assert "cannot list s3://<bucket>/models/m" in out and "AccessDenied" in out
    assert "model_pull.dest '/mnt/nvme/models/m'" in out and "leave dest unset" in out


# -- the loop's exits --------------------------------------------------------------

def test_sigterm_drains_every_node_before_the_loop_exits(tmp_path, capsys):
    cfg, path = _cfg(tmp_path)
    clock = Clock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud, accounts=(1, 4))
    m = _ready_member(pool, clock, cloud)
    rc = pool_run.run_pool(cfg, path, interval=60, pool=pool, signals=Signals("SIGTERM", clock=clock))
    assert rc == 0 and m.state == GONE and cloud.kinds("drain")
    assert "signal SIGTERM" in pool.stopped
    assert not pool_run.pid_path(cfg).exists()
    assert "NODES STILL BILLING" not in capsys.readouterr().out


def test_a_drain_that_keeps_failing_prints_the_billing_banner(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(pool_run.time, "sleep", lambda s: None)
    cfg, path = _cfg(tmp_path)
    clock = Clock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud, accounts=(1, 4))
    m = _ready_member(pool, clock, cloud)
    cloud.drain_error = 100
    rc = pool_run.run_pool(cfg, path, interval=60, pool=pool, signals=Signals("SIGINT", clock=clock))
    out = capsys.readouterr().out
    assert rc == 1 and m.state != GONE
    assert "NODES STILL BILLING" in out and m.ticket in out and pool_run.stop_command(path) in out


def test_the_stop_file_drains_a_running_loop(tmp_path):
    cfg, path = _cfg(tmp_path)
    clock = Clock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud, accounts=(1, 4))
    _ready_member(pool, clock, cloud)

    class StopOnWait(Signals):
        def _wait(self, timeout):
            pool_run.stop_path(cfg).write_text("{}")
            return False

    assert pool_run.run_pool(cfg, path, interval=60, pool=pool, signals=StopOnWait()) == 0
    assert pool.stopped == "pool-stop requested" and not pool.live()
    assert not pool_run.stop_path(cfg).exists()


def test_a_second_loop_on_the_same_pool_is_refused(tmp_path):
    cfg, path = _cfg(tmp_path)
    pool_run.pid_path(cfg).parent.mkdir(parents=True, exist_ok=True)
    pool_run.pid_path(cfg).write_text(json.dumps({"pid": os.getppid(), "host": socket.gethostname()}))
    with pytest.raises(SystemExit, match="already runs"):
        pool_run.run_pool(cfg, path, interval=60)


# -- stop and status after a crash ---------------------------------------------------

def _crashed_pool(tmp_path):
    """A pool whose loop died (SIGKILL, host gone): state file with a live member,
    a stale pid file."""
    cfg, path = _cfg(tmp_path)
    clock = Clock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud, accounts=(1, 4))
    m = _ready_member(pool, clock, cloud)
    state = cli.pool_state_path(cfg)
    state.parent.mkdir(parents=True, exist_ok=True)
    pool.save(state)
    pool_run.pid_path(cfg).write_text(json.dumps({"pid": 2**22 + 12345, "host": socket.gethostname()}))
    return cfg, path, cloud, m


def test_status_after_a_crash_shows_the_billing_banner(tmp_path, capsys):
    cfg, path, _cloud, m = _crashed_pool(tmp_path)
    assert pool_run.pool_status(cfg, path) == 1
    out = capsys.readouterr().out
    doc = json.loads(out[:out.index("\n!!!")])
    assert doc["process"]["alive"] is False and doc["billed"][0]["ticket"] == m.ticket
    assert "NODES STILL BILLING" in out and pool_run.stop_command(path) in out


def test_stop_after_a_crash_releases_the_pools_own_tickets_only(tmp_path, capsys):
    cfg, path, cloud, m = _crashed_pool(tmp_path)
    cloud.ledger.append(["run-a", 4, "t-orphan", True])  # this run's, no member owns it
    cloud.tickets["t-orphan"] = dict(cloud.tickets[m.ticket], account=4)
    cloud.ledger.append(["run-b", 4, "t-other", True])  # another run's: never touched
    assert pool_run.stop_pool(cfg, path, ops=cloud, sleep=lambda s: None) == 0
    drained = [e[1] for e in cloud.kinds("drain")]
    assert m.ticket in drained and "t-orphan" in drained and "t-other" not in drained
    doc = json.loads(cli.pool_state_path(cfg).read_text())
    assert doc["stopped"] == "pool-stop" and not pool_run.billed_members(doc)
    assert not pool_run.pid_path(cfg).exists()
    assert "every node of the pool is released" in capsys.readouterr().out


def test_stop_asks_a_running_loop_and_waits_for_it(tmp_path, capsys):
    cfg, path = _cfg(tmp_path)
    state = cli.pool_state_path(cfg)
    state.parent.mkdir(parents=True, exist_ok=True)
    pool_run.pid_path(cfg).write_text(json.dumps({"pid": os.getpid(), "host": socket.gethostname()}))

    def loop_drains(_s):  # the running loop reads the stop file, drains, exits
        assert pool_run.stop_path(cfg).exists()
        state.write_text(json.dumps({"stopped": "pool-stop requested", "members": [
            {"account": 1, "ticket": "t1", "shard": "nc-0", "state": "gone"}]}))
        pool_run.pid_path(cfg).unlink()

    assert pool_run.stop_pool(cfg, path, sleep=loop_drains) == 0
    assert "drained every node" in capsys.readouterr().out


def test_stop_reports_a_running_loop_that_does_not_stop(tmp_path, capsys):
    cfg, path = _cfg(tmp_path)
    state = cli.pool_state_path(cfg)
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps({"members": [{"account": 1, "ticket": "t1", "shard": "nc-0", "state": "ready",
                                              "lease_until": 0.0}]}))
    pool_run.pid_path(cfg).write_text(json.dumps({"pid": os.getpid(), "host": socket.gethostname()}))
    assert pool_run.stop_pool(cfg, path, wait_s=0.0, sleep=lambda s: None) == 1
    out = capsys.readouterr().out
    assert "NODES STILL BILLING" in out and "PAST lease_until, STILL BILLED" in out


# -- configuration errors ---------------------------------------------------------

def test_a_spec_the_sampler_refuses_is_caught_before_any_order(tmp_path, capsys):
    bad = {k: v for k, v in SPEC.items() if k != "max_backlog_groups"}
    _doc, path = _cfg(tmp_path, spec_doc=bad)
    assert cli.main(["pool", "--config", path, "--dry-run"]) == 1
    assert "max_backlog_groups" in capsys.readouterr().out
    cfg = json.loads(Path(path).read_text())
    clock = Clock()
    cloud = Cloud(clock)
    with pytest.raises(SystemExit, match="CONFIGURATION ERROR.*max_backlog_groups"):
        pool_run.run_pool(cfg, path, interval=60, pool=_pool(clock, cloud), signals=Signals("SIGTERM"))
    assert not cloud.kinds("order")


def test_a_config_error_halts_keeps_the_node_and_pool_retry_restarts_it_in_place(tmp_path, capsys):
    cfg, path = _cfg(tmp_path)
    clock = Clock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud, accounts=(1, 4))
    m = _ready_member(pool, clock, cloud)
    cloud.tickets[m.ticket].update(sampler="exited", exit=2)  # spec rejected
    for _ in range(5):
        pool.tick()
        clock.t += 60
    assert pool.halted and m.config_error and m.state != GONE
    assert not cloud.kinds("restart") and not cloud.kinds("fence") and not cloud.kinds("drain")
    assert len(cloud.kinds("order")) == 1  # no replacement node
    state = cli.pool_state_path(cfg)
    pool.save(state)
    assert pool_run.retry_pool(cfg, path) == 0
    assert "start it again" in capsys.readouterr().out and pool_run.retry_path(cfg).exists()
    pool_run._retry_in_loop(pool, path)
    assert [e[1] for e in cloud.kinds("restart")] == [m.ticket] and not pool.halted and m.config_error is None
    assert not pool_run.retry_path(cfg).exists()


def test_pool_retry_refuses_a_config_that_is_still_wrong(tmp_path, capsys):
    _doc, path = _cfg(tmp_path, spec_doc={k: v for k, v in SPEC.items() if k != "sampling"})
    assert cli.main(["pool-retry", "--config", path]) == 2
    assert "still wrong" in capsys.readouterr().out


def test_a_halted_node_still_goes_at_its_lease_end(tmp_path):
    clock = Clock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud, accounts=(1, 4))
    m = _ready_member(pool, clock, cloud)
    cloud.tickets[m.ticket].update(sampler="exited", exit=2)
    while clock.t < m.lease_until:
        pool.tick()
        clock.t += 60
    assert m.state == GONE and len(cloud.kinds("order")) == 1


@pytest.mark.parametrize("break_it, match", [
    (lambda d: d["engine"]["options"].update(dsa_stable_topk=True), "dsa_stable_topk"),
    (lambda d: d["engine"].update(extra_section=1), "unknown keys"),
    (lambda d: d["engine"]["server"].update(no_such_field=1), "no_such_field"),
    (lambda d: d["rollout_fields"].update(no_such_field=1), "rollout_fields"),
    (lambda d: d.update(reward="yeto.rl.no_such_reward:score"), "no_such_reward"),
    (lambda d: d.update(reward_kind="other"), "reward_kind"),
])
def test_the_full_local_build_refuses_what_the_node_would_refuse(tmp_path, capsys, break_it, match):
    doc = json.loads(EXAMPLE_SPEC.read_text())
    doc["reward"], doc["reward_kind"] = "yeto.rl.numeric_answer_reward:score", "miles"
    cfg, path = _cfg(tmp_path, spec_doc=doc)
    pool_run.check_sampler_spec(cfg)  # the example itself passes
    break_it(doc)
    cfg, path = _cfg(tmp_path, spec_doc=doc)
    with pytest.raises(ValueError, match=match):
        pool_run.check_sampler_spec(cfg)
    assert cli.main(["pool", "--config", path, "--dry-run"]) == 1
    assert "refused by the sampler" in capsys.readouterr().out


# -- H2: one pool instance (flock), shared with pool-stop and yeto cleanup --include-nc --------

def test_a_second_instance_is_refused_by_the_pool_lock(tmp_path):
    cfg, path = _cfg(tmp_path)
    held = pool_run.acquire_lock(pool_run.lock_path(cfg), run="run-a")
    assert held is not None
    try:
        assert pool_run.acquire_lock(pool_run.lock_path(cfg), run="run-a") is None
        with pytest.raises(SystemExit, match="another instance holds"):
            pool_run.run_pool(cfg, path, interval=60)
        proc = pool_run.pool_process(cfg)
        assert proc["alive"] is True and proc["lock"]["run"] == "run-a"
    finally:
        held.close()
    assert pool_run.lock_holder(pool_run.lock_path(cfg)) is None


def test_a_stale_pid_of_another_host_does_not_block_a_free_lock(tmp_path):
    cfg, path = _cfg(tmp_path)
    pool_run.acquire_lock(pool_run.lock_path(cfg), run="run-a").close()  # an earlier loop's lock file
    pool_run.pid_path(cfg).write_text(json.dumps({"pid": 1, "host": "another-name-of-this-mac"}))
    assert pool_run.pool_process(cfg)["alive"] is False


def test_the_loop_holds_the_lock_while_it_runs(tmp_path):
    cfg, path = _cfg(tmp_path)
    clock = Clock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud, accounts=(1, 4))
    _ready_member(pool, clock, cloud)
    seen = []

    class Probe(Signals):
        def _wait(self, timeout):
            seen.append(pool_run.lock_holder(pool_run.lock_path(cfg)))
            self.name = "SIGTERM"
            return False

    pool_run.run_pool(cfg, path, interval=60, pool=pool, signals=Probe())
    assert seen and seen[0]["role"] == "pool" and seen[0]["pid"] == os.getpid()
    assert pool_run.lock_holder(pool_run.lock_path(cfg)) is None


def test_pool_stop_from_the_state_file_refuses_while_a_loop_holds_the_lock(tmp_path, capsys):
    cfg, path, cloud, m = _crashed_pool(tmp_path)
    held = pool_run.acquire_lock(pool_run.lock_path(cfg), run="run-a")
    try:
        # the loop holds the lock (alive): pool-stop asks it and waits; it does not release itself
        assert pool_run.stop_pool(cfg, path, ops=cloud, wait_s=0.0, sleep=lambda s: None) == 1
        assert not cloud.kinds("drain")
    finally:
        held.close()


def test_cleanup_include_nc_leaves_the_nodes_of_a_running_pool(tmp_path, monkeypatch):
    from yeto.cost import cleanup as cleanup_mod
    from yeto.cost.ledger import ResourceLedger

    monkeypatch.setenv("HOME", str(tmp_path))
    cfg, _path = _cfg(tmp_path)
    led = ResourceLedger.for_run("run-a")
    led.created("nc_lease", "t1", "run-a", account=1)
    calls = []

    class Ops:
        def nc_release(self, account, ticket):
            calls.append(ticket)
            return {"ok": True}

    held = pool_run.acquire_lock(pool_run.lock_path(cfg), run="run-a")
    try:
        assert not cleanup_mod.cleanup_run("run-a", Ops(), include_nc=True, log=lambda m: None)
        assert calls == [] and led.open_resources()
    finally:
        held.close()
    assert cleanup_mod.cleanup_run("run-a", Ops(), include_nc=True, log=lambda m: None)
    assert calls == ["t1"]
