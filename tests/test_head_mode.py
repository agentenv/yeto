"""Unit tests for head controller mode: no network, and `sky` is never
really imported — a fake module is injected into sys.modules. Covers the
launch-args JSON round-trip through `_head`, head-mode submission recording,
`yeto down` on a head-mode entry, the local-subprocess syncer supervision in
FleetController, and LocalSyncer's probe/restart against a real (tiny)
subprocess."""

import argparse
import json
import os
import sys
import types

import pytest

import yeto.cli as cli
import yeto.launcher as launcher
import yeto.runs as runs
from yeto.launcher import FleetController, LocalSyncer


@pytest.fixture(autouse=True)
def tmp_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")


LAUNCH_ARGS = [
    "--gpu", "aws:8xa100@us-east-2,aws:8xa100@us-west-2",
    "--model", "gemma4",
    "--model-revision", "a" * 40,
    "--data", "org/ds",
    "--data-revision", "b" * 40,
]


# ---------------------------------------------------------------------------
# fake sky module (injected into sys.modules so `import sky` never happens)


class FakeHandle:
    head_ip = "203.0.113.7"


class FakeTask:
    def __init__(self, name=None, setup=None, run=None, envs=None,
                 num_nodes=1, workdir=None, file_mounts=None):
        self.name = name
        self.setup = setup
        self.run = run
        self.envs = envs
        self.num_nodes = num_nodes
        self.workdir = workdir
        self.file_mounts = file_mounts
        self.resources = None

    def set_resources(self, resources):
        self.resources = resources


class FakeResources:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def make_fake_sky(record):
    sky = types.ModuleType("sky")
    sky.Task = FakeTask
    sky.Resources = FakeResources

    def launch(task, cluster_name=None, retry_until_up=False):
        record.setdefault("launches", []).append((cluster_name, task))
        return ("launch", task)

    def sky_exec(task, cluster_name=None):
        record.setdefault("execs", []).append((cluster_name, task))
        return ("exec", task)

    def stream_and_get(rid):
        kind, _task = rid
        if kind == "launch":
            return (None, FakeHandle())
        return (record.get("next_job_id", 7), FakeHandle())

    def tail_logs(cluster, job_id, follow=True, preload_content=False):
        record.setdefault("tails", []).append((cluster, job_id, follow))
        return iter(record.get("log_lines", []))

    def down(cluster):
        record.setdefault("downs", []).append(cluster)

    sky.launch = launch
    sky.exec = sky_exec
    sky.stream_and_get = stream_and_get
    sky.tail_logs = tail_logs
    sky.down = down
    sky.get = lambda x: x
    return sky


@pytest.fixture
def fake_sky(monkeypatch):
    record = {}
    monkeypatch.setitem(sys.modules, "sky", make_fake_sky(record))
    return record


# ---------------------------------------------------------------------------
# args JSON round-trip through the `_head` serialization


def test_launch_args_json_roundtrip():
    ns = cli.parse_args(LAUNCH_ARGS + ["--cluster-prefix", "rt", "--max-rows", "100"])
    payload = json.dumps(cli._serializable_args(ns))
    rebuilt = argparse.Namespace(**json.loads(payload))
    for key, value in vars(ns).items():
        assert getattr(rebuilt, key) == value, key
        assert type(getattr(rebuilt, key)) is type(value), key  # int/float/bool survive
    assert rebuilt.controller == "head"


def test_serializable_args_drops_unserializable_fields():
    ns = cli.parse_args(LAUNCH_ARGS)
    ns.command = "launch"
    ns.opaque = object()
    out = cli._serializable_args(ns)
    assert "command" not in out and "opaque" not in out
    json.dumps(out)  # everything left is JSON-serializable


def test_cmd_head_reconstructs_args_and_starts_syncer(monkeypatch):
    seen = {}

    class FakeLocalSyncer:
        def __init__(self, args, num_learners):
            seen["num_learners"] = num_learners
            self.started = False

        def start(self):
            seen["started"] = True

        def start_log_forwarder(self):
            seen["forwarder"] = True

        def stop(self):
            seen["stopped"] = True

    def fake_run(args, on_clusters=None, local_syncer=None):
        seen["args"] = args
        seen["local_syncer"] = local_syncer
        return 0

    monkeypatch.setattr(launcher, "LocalSyncer", FakeLocalSyncer)
    monkeypatch.setattr(launcher, "run", fake_run)

    ns = cli.parse_args(LAUNCH_ARGS + ["--cluster-prefix", "hh"])
    rc = cli.main(["_head", json.dumps(cli._serializable_args(ns))])

    assert rc == 0
    assert seen["num_learners"] == 2
    assert seen["started"] and seen["forwarder"] and seen["stopped"]
    assert isinstance(seen["local_syncer"], FakeLocalSyncer)
    args = seen["args"]
    assert args.gpu == ns.gpu
    assert args.cluster_prefix == "hh"
    assert args.total_steps == ns.total_steps
    assert args.controller == "head"


# ---------------------------------------------------------------------------
# cmd_launch in head mode: registry recording with sky stubbed


def test_launch_head_records_registry(fake_sky, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_spawn_worker", lambda name: pytest.fail("head mode must not spawn a local worker"))
    fake_sky["next_job_id"] = 42

    rc = cli.main(["launch", *LAUNCH_ARGS, "--cluster-prefix", "h1"])  # default: head
    assert rc == 0

    meta = runs.load_run("h1")
    assert meta["controller"] == "head"
    assert meta["head_cluster"] == "h1-head"
    assert meta["head_job_id"] == 42
    assert meta["state"] == runs.SUBMITTED
    assert meta["clusters"] == ["h1-head", "h1-l0-us-east-2", "h1-l1-us-west-2"]
    assert meta["args"]["controller"] == "head"

    # One provisioning launch of the head, then the controller job exec'd
    # with the head's IP (from the handle) and the serialized args.
    (cluster, head_task), = fake_sky["launches"]
    assert cluster == "h1-head"
    assert "~/yeto-syncer" not in head_task.file_mounts
    assert head_task.resources.kwargs["use_spot"] is False
    assert head_task.resources.kwargs["infra"] == "aws/us-west-2"
    assert 'pip install -q "skypilot[aws,gcp]>=0.12"' in head_task.setup
    assert "cargo build --release --quiet" in head_task.setup
    assert "touch ~/.yeto_head_ready" in head_task.setup
    (exec_cluster, job_task), = fake_sky["execs"]
    assert exec_cluster == "h1-head"
    assert job_task.envs["SYNCER_PUBLIC_IP"] == "203.0.113.7"
    assert "python3 -m yeto.cli _head " in job_task.run
    assert fake_sky["tails"] == [("h1-head", 42, True)]  # log stream attached

    out = capsys.readouterr().out
    assert "yeto down h1" in out


def test_rl_head_forwards_cybergym_secret_without_serializing_it(
    fake_sky, monkeypatch
):
    monkeypatch.setenv("CYBERGYM_API_KEY", "test-secret")
    monkeypatch.setattr(launcher, "prepare_launch_args", lambda args: None)
    args = cli.parse_args(
        LAUNCH_ARGS
        + [
            "--cluster-prefix",
            "rlh",
            "--training-mode",
            "rl",
            "--reward-function",
            "yeto_miles_cybergym.reward:score",
        ]
    )

    assert cli.cmd_launch_head(args) == 0

    (_, head_task), = fake_sky["launches"]
    (_, job_task), = fake_sky["execs"]
    assert head_task.envs is None
    assert job_task.envs["CYBERGYM_API_KEY"] == "test-secret"
    assert "test-secret" not in job_task.run
    assert "test-secret" not in json.dumps(runs.load_run("rlh")["args"])


def test_rl_head_stages_the_initial_adapter_for_learner_mounts(
    fake_sky, monkeypatch, tmp_path
):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")

    def prepare(args):
        args.rl_initial_adapter_sha256 = "a" * 64

    monkeypatch.setattr(launcher, "prepare_launch_args", prepare)
    args = cli.parse_args(
        LAUNCH_ARGS
        + [
            "--cluster-prefix",
            "rl-parent",
            "--training-mode",
            "rl",
            "--reward-function",
            "yeto_miles_cybergym.reward:score",
            "--rl-sync-preset",
            "decoupled",
            "--rl-initial-adapter",
            str(adapter),
        ]
    )

    assert cli.cmd_launch_head(args) == 0

    (_, head_task), = fake_sky["launches"]
    (_, job_task), = fake_sky["execs"]
    assert (
        head_task.file_mounts["~/yeto-rl-initial-adapter-src"]
        == str(adapter)
    )
    assert '"rl_initial_adapter": "~/yeto-rl-initial-adapter-src"' in job_task.run
    assert '"rl_initial_adapter_sha256": "' + "a" * 64 + '"' in job_task.run


def test_launch_head_mounts_aws_credentials_when_present(
    fake_sky, monkeypatch, tmp_path
):
    fake_home = tmp_path / "home"
    (fake_home / ".aws").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(fake_home))

    assert cli.main(["launch", *LAUNCH_ARGS, "--cluster-prefix", "h2"]) == 0
    (_, head_task), = fake_sky["launches"]
    assert head_task.file_mounts["~/.aws"] == str(fake_home / ".aws")


def test_launch_head_warns_when_aws_credentials_missing(
    fake_sky, monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("HOME", str(tmp_path / "empty-home"))

    assert cli.main(["launch", *LAUNCH_ARGS, "--cluster-prefix", "h3"]) == 0
    (_, head_task), = fake_sky["launches"]
    assert "~/.aws" not in head_task.file_mounts
    assert "~/.aws not found" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# yeto down / logs on a head-mode registry entry


def make_head_meta(name):
    ns = cli.parse_args(LAUNCH_ARGS + ["--cluster-prefix", name])
    runs.create_run(name, cli._serializable_args(ns))
    runs.update_run(
        name,
        state=runs.SUBMITTED,
        controller="head",
        head_cluster=f"{name}-head",
        head_job_id=9,
        clusters=[f"{name}-head", f"{name}-l0-us-east-2", f"{name}-l1-us-west-2"],
    )
    return runs.load_run(name)


def test_down_head_run_downs_all_recorded_clusters(monkeypatch):
    make_head_meta("hd")
    downed = []
    monkeypatch.setattr(cli, "_sky_down_cluster", downed.append)

    assert cli.main(["down", "hd"]) == 0
    assert sorted(downed) == ["hd-head", "hd-l0-us-east-2", "hd-l1-us-west-2"]
    assert runs.load_run("hd")["state"] == runs.DOWN


def test_logs_head_run_streams_from_head(fake_sky, capsys):
    make_head_meta("hl")
    fake_sky["log_lines"] = ["[launcher] hello from the head\n"]

    assert cli.main(["logs", "hl", "--no-follow"]) == 0
    assert fake_sky["tails"] == [("hl-head", 9, False)]
    assert "hello from the head" in capsys.readouterr().out


def test_status_hints_head_runs(capsys):
    make_head_meta("hs")
    assert cli.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "SUBMITTED" in out
    assert "controlled from hs-head" in out
    assert "yeto logs hs" in out


# ---------------------------------------------------------------------------
# FleetController with a subprocess syncer (probe/restart callables)


class FakeStatus:
    def __init__(self, name, terminal):
        self._name = name
        self._terminal = terminal

    def is_terminal(self):
        return self._terminal

    def __str__(self):
        return f"JobStatus.{self._name}"


RUNNING = FakeStatus("RUNNING", False)
SUCCEEDED = FakeStatus("SUCCEEDED", True)


class ImmediateThread:
    def __init__(self, target=None, args=(), kwargs=None, daemon=None):
        self._target, self._args, self._kwargs = target, args, kwargs or {}

    def start(self):
        self._target(*self._args, **self._kwargs)


class FakeOps:
    def __init__(self):
        self.t = 0.0
        self.sleeps = 0
        self.status_seq = {}
        self.down_calls = []

    def job_status(self, cluster, job_id):
        seq = self.status_seq[cluster]
        return seq.pop(0) if len(seq) > 1 else seq[0]

    def cluster_up(self, cluster):
        return True

    def relaunch(self, task, cluster):
        return None

    def down(self, cluster):
        self.down_calls.append(cluster)

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds
        self.sleeps += 1
        assert self.sleeps < 500, "controller poll loop did not terminate"


def make_head_controller(ops, learners, probe, restart):
    return FleetController(
        learners={name: (f"task-{name}", job) for name, job in learners.items()},
        syncer=None,
        sky_ops=ops,
        poll_interval=30,
        recover_timeout=100,
        thread_cls=ImmediateThread,
        syncer_probe=probe,
        syncer_restart=restart,
    )


def test_dead_syncer_subprocess_is_restarted():
    ops = FakeOps()
    ops.status_seq["l0"] = [RUNNING] * 4 + [SUCCEEDED]
    calls = {"restarts": 0}

    def probe():
        # Dead until the first restart, healthy afterwards.
        return None if calls["restarts"] else "syncer subprocess exited with code 1"

    def restart():
        calls["restarts"] += 1

    ctl = make_head_controller(ops, {"l0": 1}, probe, restart)
    exit_codes = ctl.run()

    assert exit_codes == {"l0": "JobStatus.SUCCEEDED"}
    assert calls["restarts"] == 1
    assert ops.down_calls == []  # no syncer cluster to tear down, ever


def test_syncer_subprocess_never_abandoned(capsys):
    # The subprocess keeps dying: restart is attempted every poll, the
    # controller never gives up on it, and the run still completes.
    ops = FakeOps()
    ops.status_seq["l0"] = [RUNNING] * 8 + [SUCCEEDED]
    restarts = []

    ctl = make_head_controller(
        ops,
        {"l0": 1},
        probe=lambda: "syncer subprocess exited with code 1",
        restart=lambda: restarts.append(1),
    )
    exit_codes = ctl.run()

    assert exit_codes == {"l0": "JobStatus.SUCCEEDED"}
    assert len(restarts) >= 8  # once per poll, for the whole run
    assert ops.down_calls == []
    assert "restarting the local syncer" in capsys.readouterr().err


def test_restart_failure_is_retried_not_fatal(capsys):
    ops = FakeOps()
    ops.status_seq["l0"] = [RUNNING, RUNNING, SUCCEEDED]

    def restart():
        raise OSError("no such binary")

    ctl = make_head_controller(
        ops, {"l0": 1}, probe=lambda: "syncer subprocess exited", restart=restart
    )
    assert ctl.run() == {"l0": "JobStatus.SUCCEEDED"}
    assert "syncer restart failed" in capsys.readouterr().err


def test_probe_requires_restart_callable():
    with pytest.raises(ValueError):
        FleetController(
            learners={},
            syncer=None,
            sky_ops=FakeOps(),
            poll_interval=30,
            recover_timeout=100,
            syncer_probe=lambda: None,
        )


# ---------------------------------------------------------------------------
# LocalSyncer against a real (tiny) subprocess


def test_local_syncer_probe_and_restart(tmp_path):
    binary = tmp_path / "yeto-syncer"
    binary.write_text("#!/bin/sh\necho syncer up\nsleep 30\n")
    args = cli.parse_args(LAUNCH_ARGS)
    syncer = LocalSyncer(
        args, 2, binary=str(binary), log_file=str(tmp_path / "syncer.log")
    )
    assert syncer.probe() == "syncer subprocess was never started"

    syncer.start()
    try:
        assert os.access(str(binary), os.X_OK)  # chmod +x applied
        assert syncer.probe() is None  # healthy
        first_pid = syncer.proc.pid

        syncer.proc.kill()
        syncer.proc.wait()
        assert "exited with code" in syncer.probe()

        syncer.restart()
        assert syncer.probe() is None
        assert syncer.proc.pid != first_pid
    finally:
        syncer.stop()
    assert syncer.probe() is not None  # stopped


def test_local_syncer_command_matches_cluster_syncer_flags():
    args = cli.parse_args(LAUNCH_ARGS + ["--quorum", "2", "--total-steps", "17"])
    cmd = launcher.syncer_command(args, 3)
    assert "--learners 3" in cmd
    assert "--quorum 2" in cmd
    assert "--total-steps 17" in cmd
    assert "--resume" in cmd
    assert "--mark-final-checkpoint" in cmd
    assert f"--port {launcher.SYNCER_PORT}" in cmd
