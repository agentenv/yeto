"""Unit tests for the run registry and the subcommand CLI: no network, no
SkyPilot. The registry is redirected to a temp dir; the launcher and the
sky teardown call are stubbed."""

import argparse
import json
import os
import subprocess
import time

import pytest

import yeto.cli as cli
import yeto.runs as runs


@pytest.fixture(autouse=True)
def tmp_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")


# These tests exercise the local controller mode (a detached worker on this
# machine); head mode is covered by test_head_mode.py.
LAUNCH_ARGS = [
    "--gpu", "aws:8xa100@us-east-2", "--model", "gemma4",
    "--model-revision", "a" * 40, "--data", "org/ds",
    "--data-revision", "b" * 40,
    "--controller", "local",
]


def make_args_dict(name="run1"):
    ns = cli.parse_args(LAUNCH_ARGS + ["--cluster-prefix", name])
    return vars(ns)


# ---------------------------------------------------------------------------
# registry


def test_registry_create_load_list_update_roundtrip():
    meta = runs.create_run("r1", make_args_dict("r1"))
    assert meta["state"] == "PENDING"
    assert meta["pid"] is None and meta["exit_code"] is None
    assert runs.log_path("r1").exists()

    loaded = runs.load_run("r1")
    assert loaded == meta
    assert loaded["args"]["gpu"] == "aws:8xa100@us-east-2"

    runs.update_run("r1", pid=4242, state="RUNNING")
    runs.update_run("r1", clusters=["r1-syncer", "r1-l0-us-east-2"])
    loaded = runs.load_run("r1")
    assert loaded["pid"] == 4242
    assert loaded["state"] == "RUNNING"  # earlier fields survive later updates
    assert loaded["clusters"] == ["r1-syncer", "r1-l0-us-east-2"]

    time.sleep(0.01)
    runs.create_run("r2", make_args_dict("r2"))
    assert [m["name"] for m in runs.list_runs()] == ["r2", "r1"]  # newest first


def test_load_unknown_run_returns_none():
    assert runs.load_run("nope") is None
    assert runs.list_runs() == []


def test_create_run_resets_log_of_reused_name():
    runs.create_run("r1", make_args_dict("r1"))
    runs.log_path("r1").write_text("old output\n")
    runs.create_run("r1", make_args_dict("r1"))
    assert runs.log_path("r1").read_text() == ""


def test_is_alive():
    assert runs.is_alive(os.getpid())
    assert not runs.is_alive(None)
    assert not runs.is_alive(0)
    proc = subprocess.Popen(["sleep", "0"])
    proc.wait()
    assert not runs.is_alive(proc.pid)


# ---------------------------------------------------------------------------
# launch: duplicate-name refusal + detach path


def test_launch_refused_while_worker_alive(monkeypatch, capsys):
    runs.create_run("busy", make_args_dict("busy"))
    proc = subprocess.Popen(["sleep", "30"])
    try:
        runs.update_run("busy", pid=proc.pid, state="RUNNING")
        monkeypatch.setattr(
            cli, "_spawn_worker", lambda name: pytest.fail("must not spawn a worker")
        )
        rc = cli.main(["launch", *LAUNCH_ARGS, "--cluster-prefix", "busy"])
        assert rc == 1
        assert "already has a live worker" in capsys.readouterr().err
    finally:
        proc.kill()
        proc.wait()


def test_launch_allows_reuse_of_dead_run(monkeypatch, capsys):
    runs.create_run("old", make_args_dict("old"))
    runs.update_run("old", pid=None, state="FAILED", exit_code=1)

    class FakeProc:
        pid = 5555

        def poll(self):
            return 0  # exited immediately

    def fake_spawn(name):
        # Simulate a worker that ran to completion instantly.
        runs.update_run(name, state="SUCCEEDED", exit_code=0, finished_at=time.time())
        return FakeProc()

    monkeypatch.setattr(cli, "_spawn_worker", fake_spawn)
    rc = cli.main(["launch", *LAUNCH_ARGS, "--cluster-prefix", "old"])
    out = capsys.readouterr().out
    assert "submitted; worker pid 5555" in out
    assert rc == 0


def test_launch_ctrl_c_detaches_with_exit_code_zero(monkeypatch, capsys):
    class FakeProc:
        pid = 7777

        def poll(self):
            return None

    monkeypatch.setattr(cli, "_spawn_worker", lambda name: FakeProc())

    def fake_stream(name, follow=True, alive=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_stream_log", fake_stream)
    rc = cli.main(["launch", *LAUNCH_ARGS, "--cluster-prefix", "bg"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "yeto logs bg" in out and "yeto down bg" in out
    assert runs.load_run("bg")["pid"] == 7777


# ---------------------------------------------------------------------------
# _worker: state recording around yeto.launcher.run


def test_worker_records_success_and_clusters(monkeypatch):
    import yeto.launcher

    runs.create_run("w1", make_args_dict("w1"))

    def fake_run(args, on_clusters=None):
        assert isinstance(args, argparse.Namespace)
        assert args.gpu == "aws:8xa100@us-east-2"
        if on_clusters is not None:
            on_clusters(["w1-syncer", "w1-l0-us-east-2"])
        return 0

    monkeypatch.setattr(yeto.launcher, "run", fake_run)
    rc = cli.main(["_worker", "w1"])
    assert rc == 0
    meta = runs.load_run("w1")
    assert meta["state"] == "SUCCEEDED"
    assert meta["exit_code"] == 0
    assert meta["finished_at"] is not None
    assert meta["pid"] == os.getpid()
    assert meta["clusters"] == ["w1-syncer", "w1-l0-us-east-2"]


def test_worker_records_nonzero_exit_as_failed(monkeypatch):
    import yeto.launcher

    runs.create_run("w2", make_args_dict("w2"))
    monkeypatch.setattr(yeto.launcher, "run", lambda args, on_clusters=None: 3)
    rc = cli.main(["_worker", "w2"])
    assert rc == 3
    meta = runs.load_run("w2")
    assert meta["state"] == "FAILED"
    assert meta["exit_code"] == 3


def test_worker_records_failure_on_exception(monkeypatch, capsys):
    import yeto.launcher

    runs.create_run("w3", make_args_dict("w3"))

    def boom(args, on_clusters=None):
        raise RuntimeError("all learners abandoned")

    monkeypatch.setattr(yeto.launcher, "run", boom)
    rc = cli.main(["_worker", "w3"])
    assert rc == 1
    meta = runs.load_run("w3")
    assert meta["state"] == "FAILED"
    assert meta["exit_code"] == 1
    assert meta["finished_at"] is not None
    assert "all learners abandoned" in capsys.readouterr().err  # traceback in log


def test_worker_unknown_run(capsys):
    assert cli.main(["_worker", "ghost"]) == 1
    assert "no recorded args" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# diffusion sampling command


def test_sample_diffusion_requires_one_prompt_source(capsys):
    base = ["sample-diffusion", "--gpu", "aws:1xt4", "--adapter-dir", "adapter"]

    assert cli.main(base) == 1
    assert "exactly one" in capsys.readouterr().err
    assert cli.main(base + ["--prompt", "p", "--data", "org/ds"]) == 1
    assert "exactly one" in capsys.readouterr().err


def test_sample_diffusion_dispatches_to_launcher(monkeypatch):
    import yeto.launcher

    seen = {}

    def fake_run(args):
        seen["args"] = args
        return 0

    monkeypatch.setattr(yeto.launcher, "run_diffusion_sample", fake_run)

    rc = cli.main(
        [
            "sample-diffusion",
            "--gpu",
            "aws:1xt4@us-west-2",
            "--adapter-dir",
            "s3://bucket/adapter",
            "--prompt",
            "a cat",
            "--output",
            "samples",
        ]
    )

    assert rc == 0
    assert seen["args"].prompt == "a cat"
    assert seen["args"].adapter_dir == "s3://bucket/adapter"
    assert seen["args"].output == "samples"


# ---------------------------------------------------------------------------
# down: dead worker, recorded clusters torn down via stubbed sky


def test_down_dead_worker_tears_down_clusters(monkeypatch, capsys):
    runs.create_run("d1", make_args_dict("d1"))
    proc = subprocess.Popen(["sleep", "0"])
    proc.wait()
    runs.update_run(
        "d1", pid=proc.pid, state="RUNNING", clusters=["d1-syncer", "d1-l0-us-east-2"]
    )

    downed = []
    monkeypatch.setattr(cli, "_sky_down_cluster", downed.append)
    rc = cli.main(["down", "d1"])
    assert rc == 0
    assert sorted(downed) == ["d1-l0-us-east-2", "d1-syncer"]
    meta = runs.load_run("d1")
    assert meta["state"] == "DOWN"
    assert meta["finished_at"] is not None
    assert "worker is not running" in capsys.readouterr().out


def test_down_survives_sky_errors(monkeypatch, capsys):
    runs.create_run("d2", make_args_dict("d2"))
    runs.update_run("d2", pid=None, clusters=["d2-syncer"])

    def explode(cluster):
        raise RuntimeError("cluster already gone")

    monkeypatch.setattr(cli, "_sky_down_cluster", explode)
    rc = cli.main(["down", "d2"])
    assert rc == 0
    assert runs.load_run("d2")["state"] == "DOWN"
    assert "teardown failed" in capsys.readouterr().err


def test_down_unknown_run(capsys):
    assert cli.main(["down", "ghost"]) == 1
    assert "unknown run" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# status + logs (registry only, instant)


def test_status_lists_runs_without_sky(capsys):
    runs.create_run("s1", make_args_dict("s1"))
    runs.update_run(
        "s1", pid=None, state="SUCCEEDED", exit_code=0, clusters=["s1-syncer"]
    )
    with open(runs.log_path("s1"), "a") as f:
        f.write("[launcher] fine-tuned model saved on s1-l0:~/yeto-output\n\n")
    rc = cli.main(["status"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "NAME" in out and "STATE" in out
    line = next(ln for ln in out.splitlines() if ln.startswith("s1"))
    assert "SUCCEEDED" in line
    assert "s1-syncer" in line
    assert "fine-tuned model saved" in line


def test_status_shows_running_for_live_pid(capsys):
    runs.create_run("s2", make_args_dict("s2"))
    runs.update_run("s2", pid=os.getpid(), state="PENDING")
    cli.main(["status"])
    line = next(
        ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("s2")
    )
    assert "RUNNING" in line


def test_status_summarizes_event_tape(tmp_path, capsys):
    tape = tmp_path / "events.jsonl"
    records = [
        {
            "step": 1,
            "fragment": 0,
            "expected": [0, 1],
            "responded": [0, 1],
            "missed_grace": [],
            "responders": [
                {"id": 0, "c_steps": 2, "c_tokens": 20, "weight": 200.0},
                {"id": 1, "c_steps": 2, "c_tokens": 20, "weight": 200.0},
            ],
        },
        {
            "step": 2,
            "fragment": 1,
            "expected": [0, 1],
            "responded": [0],
            "missed_grace": [1],
            "responders": [
                {"id": 0, "c_steps": 2, "c_tokens": 20, "weight": 200.0},
            ],
        },
    ]
    tape.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    assert cli.main(["status", "--tape", str(tape)]) == 0
    out = capsys.readouterr().out
    assert "TAPE" in out
    assert "ROUNDS 2" in out
    assert "MISSED 1 across 1 rounds" in out
    assert "NODE  RESPONSES" in out
    assert "0     2" in out
    assert "1     1" in out
    assert "step=2/frag=1: [1]" in out


def test_logs_no_follow_dumps_log(capsys):
    runs.create_run("lg", make_args_dict("lg"))
    runs.update_run("lg", pid=None, state="SUCCEEDED", exit_code=0)
    runs.log_path("lg").write_text("hello from the worker\n")
    rc = cli.main(["logs", "lg", "--no-follow"])
    assert rc == 0
    assert "hello from the worker" in capsys.readouterr().out


def test_logs_follow_ends_when_worker_dead(capsys):
    runs.create_run("lf", make_args_dict("lf"))
    runs.update_run("lf", pid=None, state="FAILED", exit_code=1)
    runs.log_path("lf").write_text("boom\n")
    rc = cli.main(["logs", "lf"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "boom" in out
    assert "not running: FAILED (exit code 1)" in out


def test_logs_unknown_run(capsys):
    assert cli.main(["logs", "ghost"]) == 1
    assert "unknown run" in capsys.readouterr().err
