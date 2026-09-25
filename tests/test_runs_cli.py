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
from yeto.status_metrics import load_tape, summarize_tape


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


def test_down_stops_the_modal_app_and_skips_sky_for_modal_islands(monkeypatch, capsys):
    runs.create_run("m1", make_args_dict("m1"))
    runs.update_run("m1", pid=None, clusters=["m1-syncer", "m1-l0-us-east-1", "m1-l1-modal"])
    downed, stopped = [], []
    monkeypatch.setattr(cli, "_sky_down_cluster", downed.append)
    monkeypatch.setattr(cli, "_modal_stop_app", stopped.append)
    monkeypatch.setattr(cli, "_modal_app_stopped", lambda run: (True, "stopped, 0 tasks"))
    assert cli.main(["down", "m1"]) == 0
    assert sorted(downed) == ["m1-l0-us-east-1", "m1-syncer"]  # never sky.down a Modal island
    assert stopped == ["m1"]  # one app stop covers every Modal island of the run
    assert runs.load_run("m1")["state"] == "DOWN"
    assert "m1-l1-modal: stopped with the Modal app" in capsys.readouterr().out


def test_down_treats_a_cluster_sky_never_had_as_gone(monkeypatch, capsys):
    """Local mode, no cloud probe: sky saying it never had the cluster is
    the one down error that still counts as gone (a rerun after a clean
    down must stay green)."""
    runs.create_run("d2", make_args_dict("d2"))
    runs.update_run("d2", pid=None, clusters=["d2-syncer"])

    def vanished(cluster):
        raise ValueError(f"Cluster '{cluster}' does not exist.")

    monkeypatch.setattr(cli, "_sky_down_cluster", vanished)
    monkeypatch.setattr(cli, "_cloud_probe", lambda cluster: None)
    assert cli.main(["down", "d2"]) == 0
    assert runs.load_run("d2")["state"] == "DOWN"
    assert "not cloud-verifiable here; trusting sky" in capsys.readouterr().err


def test_down_no_longer_claims_success_on_other_sky_errors(monkeypatch, capsys):
    """Any other down error used to print "teardown failed" and then
    "run is down" with exit 0; now it is unconfirmed and the run is left
    for a rerun."""
    runs.create_run("d3", make_args_dict("d3"))
    runs.update_run("d3", pid=None, clusters=["d3-syncer"])

    def explode(cluster):
        raise RuntimeError("sky API server unreachable")

    monkeypatch.setattr(cli, "_sky_down_cluster", explode)
    monkeypatch.setattr(cli, "_cloud_probe", lambda cluster: None)
    assert cli.main(["down", "d3"]) == 1
    meta = runs.load_run("d3")
    assert meta["state"] == runs.TEARDOWN_INCOMPLETE
    assert meta["teardown_unconfirmed"] == ["d3-syncer"]
    out, err = capsys.readouterr()
    assert "run 'd3' is down" not in out
    assert "d3-syncer: not confirmed down" in err
    assert "NOT fully down; unconfirmed: d3-syncer" in err


def test_down_cloud_verifies_and_retries_until_the_cloud_is_empty(monkeypatch, capsys):
    runs.create_run("d4", make_args_dict("d4"))
    runs.update_run("d4", pid=None, clusters=["d4-syncer"])
    downed = []
    monkeypatch.setattr(cli, "_sky_down_cluster", downed.append)
    seen = {"n": 0}

    def probe():
        seen["n"] += 1
        return ["i-zombie"] if seen["n"] < 3 else []

    monkeypatch.setattr(cli, "_cloud_probe", lambda cluster: probe)
    monkeypatch.setattr(cli, "DOWN_VERIFY_SLEEP", lambda s: None)
    assert cli.main(["down", "d4"]) == 0
    assert len(downed) == 3  # initial + one retry per live report
    assert runs.load_run("d4")["state"] == "DOWN"
    assert "d4-syncer: down" in capsys.readouterr().out


def test_down_fails_when_the_cloud_still_has_an_instance(monkeypatch, capsys):
    runs.create_run("d5", make_args_dict("d5"))
    runs.update_run("d5", pid=None, clusters=["d5-syncer"])
    monkeypatch.setattr(cli, "_sky_down_cluster", lambda c: None)
    monkeypatch.setattr(cli, "_cloud_probe", lambda cluster: (lambda: ["i-zombie"]))
    monkeypatch.setattr(cli, "DOWN_VERIFY_SLEEP", lambda s: None)
    assert cli.main(["down", "d5"]) == 1
    assert runs.load_run("d5")["state"] == runs.TEARDOWN_INCOMPLETE
    out, err = capsys.readouterr()
    assert "i-zombie" in err  # the surviving instance is named so it can be deleted
    assert "run 'd5' is down" not in out


def test_down_fails_when_the_modal_app_is_still_running(monkeypatch, capsys):
    runs.create_run("m2", make_args_dict("m2"))
    runs.update_run("m2", pid=None, clusters=["m2-syncer", "m2-l0-modal"])
    monkeypatch.setattr(cli, "_sky_down_cluster", lambda c: None)
    monkeypatch.setattr(cli, "_cloud_probe", lambda cluster: (lambda: []))
    monkeypatch.setattr(cli, "_modal_stop_app", lambda run: None)
    monkeypatch.setattr(
        cli, "_modal_app_stopped", lambda run: (False, "Modal app yeto-m2: still running with 1 task(s)")
    )
    assert cli.main(["down", "m2"]) == 1
    assert runs.load_run("m2")["teardown_unconfirmed"] == ["m2-l0-modal"]
    assert "m2-l0-modal: not confirmed stopped" in capsys.readouterr().err


def test_modal_app_stopped_reads_state_and_tasks(monkeypatch):
    from yeto import modal_runner

    class Ops:
        def __init__(self, app):
            self.app = app

        def app_status(self):
            return {"yeto-ok": ("stopped", 0), "yeto-busy": ("running", 2)}.get(self.app)

    monkeypatch.setattr(modal_runner, "ModalOps", Ops)
    assert cli._modal_app_stopped("ok") == (True, "Modal app yeto-ok: stopped, 0 tasks")
    ok, detail = cli._modal_app_stopped("busy")
    assert not ok and "still running with 2 task(s)" in detail
    ok, detail = cli._modal_app_stopped("gone")
    assert ok and "not listed by Modal" in detail

    class Broken(Ops):
        def app_status(self):
            raise RuntimeError("modal app list failed")

    monkeypatch.setattr(modal_runner, "ModalOps", Broken)
    ok, detail = cli._modal_app_stopped("x")
    assert not ok and "status unverified" in detail


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


def test_status_prefers_sweep_accounting_and_preserves_legacy_counters():
    records = [
        {
            "step": 1,
            "responders": [
                {
                    "id": 0,
                    "c_steps": 1,
                    "c_tokens": 17,
                    "accounted_c_steps": 0,
                    "accounted_c_tokens": 0,
                }
            ],
        },
        {
            "step": 2,
            "responders": [
                {
                    "id": 0,
                    "c_steps": 1,
                    "c_tokens": 17,
                    "accounted_c_steps": 1,
                    "accounted_c_tokens": 17,
                },
                {"id": 1, "c_steps": 2, "c_tokens": 23},
            ],
        },
    ]
    rows = {row["id"]: row for row in summarize_tape(records)["contributions"]}
    assert (rows[0]["responses"], rows[0]["steps"], rows[0]["tokens"]) == (
        2,
        1,
        17,
    )
    assert (rows[1]["responses"], rows[1]["steps"], rows[1]["tokens"]) == (
        1,
        2,
        23,
    )


def test_status_uses_latest_ledger_snapshot_without_counting_it_as_a_round():
    records = [
        {
            "step": 1,
            "fragment": 0,
            "responders": [
                {
                    "id": 0,
                    "c_steps": 1,
                    "c_tokens": 17,
                    "accounted_c_steps": 0,
                    "accounted_c_tokens": 0,
                }
            ],
        },
        {
            "event": "policy_sweep_ledger",
            "global_step": 2,
            "policy_round": 1,
            "sweep_fragments": 2,
            "sweep_complete": True,
            "ledger": [{"id": 0, "merges": 2, "steps": 1, "tokens": 17}],
        },
    ]
    summary = summarize_tape(records)
    assert (summary["rounds"], summary["latest_step"], summary["latest_fragment"]) == (
        1,
        2,
        1,
    )
    assert summary["contributions"][0] == {
        "id": 0,
        "responses": 2,
        "missed": 0,
        "tokens": 17,
        "steps": 1,
        "weight": 0.0,
        "contribution": 0.0,
    }


def test_status_loads_snapshot_after_a_torn_diagnostic_line(tmp_path):
    tape = tmp_path / "events.jsonl"
    snapshot = {
        "event": "policy_sweep_ledger",
        "global_step": 2,
        "sweep_fragments": 2,
        "ledger": [{"id": 0, "merges": 2, "steps": 1, "tokens": 17}],
    }
    tape.write_text('{"step":1,"responders":[]}\n{"step":2\n')
    with tape.open("a") as stream:
        stream.write(json.dumps(snapshot) + "\n")

    records = load_tape(tape)
    assert [record.get("event") for record in records] == [
        None,
        "policy_sweep_ledger",
    ]
    assert summarize_tape(records)["latest_step"] == 2


def test_status_adds_merge_accounting_after_a_resume_ledger_snapshot():
    records = [
        {
            "event": "policy_sweep_ledger",
            "phase": "resume",
            "global_step": 2,
            "policy_round": 1,
            "sweep_fragments": 2,
            "sweep_complete": True,
            "ledger": [{"id": 0, "merges": 2, "steps": 1, "tokens": 17}],
        },
        {
            "step": 3,
            "fragment": 0,
            "responders": [
                {
                    "id": 0,
                    "accounted_c_steps": 0,
                    "accounted_c_tokens": 0,
                }
            ],
        },
        {
            "step": 4,
            "fragment": 1,
            "responders": [
                {
                    "id": 0,
                    "accounted_c_steps": 1,
                    "accounted_c_tokens": 19,
                }
            ],
        },
    ]
    summary = summarize_tape(records)
    assert (summary["rounds"], summary["latest_step"], summary["latest_fragment"]) == (
        2,
        4,
        1,
    )
    assert summary["contributions"][0]["responses"] == 4
    assert summary["contributions"][0]["steps"] == 2
    assert summary["contributions"][0]["tokens"] == 36


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


def test_modal_ops_app_status_parses_modal_app_list_json(monkeypatch):
    import json
    import subprocess
    from types import SimpleNamespace

    from yeto import modal_runner

    rows = [
        {"app_id": "ap-1", "description": "yeto-other", "state": "running", "tasks": "2"},
        {"app_id": "ap-2", "description": "yeto-td2", "state": "stopped", "tasks": "0"},
    ]
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=json.dumps(rows), stderr="")
    )
    assert modal_runner.ModalOps("yeto-td2").app_status() == ("stopped", 0)
    assert modal_runner.ModalOps("yeto-other").app_status() == ("running", 2)
    assert modal_runner.ModalOps("yeto-none").app_status() is None
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="token expired")
    )
    with pytest.raises(RuntimeError, match="modal app list failed"):
        modal_runner.ModalOps("yeto-td2").app_status()
