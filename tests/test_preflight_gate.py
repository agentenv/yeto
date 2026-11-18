"""The cluster preflight a launch of an ``ssh:`` trainer island runs first
(``yeto launch`` and ``yeto run up``): FAIL refuses, WARN prints, ``--skip-preflight``
skips, a dry run only prints the command."""

from __future__ import annotations

import io
import json
import time
import types
from pathlib import Path

import pytest

from tests.test_cluster_preflight import Behavior, FakeCluster, _inventory, b300
from yeto import cli, runs
from yeto.launch.preflight import cluster as cp
from yeto.launch.preflight import gate
from yeto.rl import up


def test_ssh_entries_and_commands_from_a_launch_argv():
    assert gate.ssh_entries("modal:2x8xb300, ssh:6x8xb300") == ["ssh:6x8xb300"]
    assert gate.ssh_entries(None) == [] and gate.ssh_entries("aws:8xh100") == []
    assert gate.argv_value(["--gpu", "a", "--gpu=b"], "--gpu") == "b"
    argv = ["--gpu=ssh:6x8xb300", "--rl-node-inventory", "inv.yaml", "--cluster-prefix", "k31"]
    assert gate.commands_for_launch_argv(argv) == [
        ["preflight", "--gpu", "ssh:6x8xb300", "--rl-node-inventory", "inv.yaml", "--no-bench",
         "--cluster-prefix", "k31"]]
    assert gate.commands_for_launch_argv(["--gpu", "modal:2x8xb300"]) == []


def _args(tmp_path, **kw):
    inv = _inventory(tmp_path)
    base = dict(gpu="ssh:2x8xb300", rl_node_inventory=str(inv), cluster_prefix="k31-bm", skip_preflight=False,
                dry_run=False)
    return types.SimpleNamespace(**{**base, **kw})


def _fake(**per_node):
    ops = FakeCluster([b300(0), b300(1)], **per_node)
    ops.NOW = time.time()
    return ops


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    return tmp_path / "runs"


def test_no_ssh_island_no_gate(tmp_path, runs_dir):
    assert gate.launch_gate(_args(tmp_path, gpu="modal:2x8xb300"), ops=_fake()) is None


def test_skip_and_dry_run_run_nothing(tmp_path, runs_dir):
    ops, out = _fake(), io.StringIO()
    rec = gate.launch_gate(_args(tmp_path, skip_preflight=True), ops=ops, out=out)
    assert rec["skipped"] and "skipped (--skip-preflight)" in out.getvalue() and not ops.scripts
    out = io.StringIO()
    rec = gate.launch_gate(_args(tmp_path, dry_run=True), ops=ops, out=out)
    assert rec["dry_run"] and not ops.scripts
    assert "would first run the cluster preflight: yeto preflight --gpu ssh:2x8xb300" in out.getvalue()
    assert "--no-bench" in out.getvalue() and not runs_dir.exists()


def test_healthy_nodes_pass_with_node_checks_and_ports_only(tmp_path, runs_dir):
    ops, out = _fake(), io.StringIO()
    rec = gate.launch_gate(_args(tmp_path), ops=ops, out=out)
    (r,) = rec["reports"]
    assert r["ok"] and r["counts"]["FAIL"] == 0
    report = json.loads((Path(r["report"]) / "preflight.json").read_text())
    names = {c["name"] for c in report["checks"]}
    assert {"driver", "memlock_container", "tcp_ports", "image_pull"} <= names
    assert not names & {"ib_write_bw", "nccl_transport", "nccl_all_reduce"}
    assert Path(r["report"]).parent == runs_dir / "k31-bm" / "preflight"
    assert not any("ib_write_bw -d" in s or "torch.distributed.run --nnodes" in s for _, s in ops.scripts)


def test_warn_prints_and_goes_on(tmp_path, runs_dir):
    out = io.StringIO()
    rec = gate.launch_gate(_args(tmp_path), ops=_fake(**{"b300-1": Behavior(host={"clock": (0, "ntp=no")})}),
                           out=out)
    assert rec["reports"][0]["ok"]
    assert "[WARN] b300-1 clock_sync: NTPSynchronized=no -> enable chrony" in out.getvalue()


def test_fail_refuses_the_launch_with_the_escape_hatch(tmp_path, runs_dir):
    ops = _fake(**{"b300-1": Behavior(container={"memlock": (0, "64")})})
    out = io.StringIO()
    with pytest.raises(ValueError, match=r"b300-1 memlock_container.*preflight.txt.*--skip-preflight"):
        gate.launch_gate(_args(tmp_path), ops=ops, out=out)
    assert "[FAIL] b300-1 memlock_container" in out.getvalue()


def test_inventory_that_does_not_match_the_island_is_refused(tmp_path, runs_dir):
    with pytest.raises(ValueError, match="trainer node"):
        gate.launch_gate(_args(tmp_path, gpu="ssh:6x8xb300"), ops=_fake())


def test_launch_cli_has_the_flag_and_refuses_on_a_failed_gate(monkeypatch, capsys):
    args = cli.build_parser().parse_args(["launch", "--model", "m", "--data", "d", "--gpu", "ssh:2x8xb300", "--skip-preflight"])
    assert args.skip_preflight
    assert not cli.build_parser().parse_args(["launch", "--model", "m", "--data", "d", "--gpu", "ssh:2x8xb300"]).skip_preflight

    def refuse(a):
        raise ValueError("cluster preflight of ssh:2x8xb300 failed (b300-1 driver)")

    monkeypatch.setattr(gate, "launch_gate", refuse)
    assert cli._cluster_preflight(args) == 1
    assert "cluster preflight refused the launch: cluster preflight of ssh:2x8xb300 failed" in capsys.readouterr().err
    monkeypatch.setattr(gate, "launch_gate", lambda a: None)
    assert cli._cluster_preflight(args) is None


def test_cmd_launch_runs_the_gate_after_the_launch_guards_and_before_the_worker():
    import inspect

    src = inspect.getsource(cli.cmd_launch)
    assert src.index("pre_cloud_checks(args)") < src.index("_cluster_preflight(args)") < src.index("_spawn_worker")
    assert src.index("launch_gate(args)") < src.index("dry_run_plan(args)")
    head = inspect.getsource(cli.cmd_launch_head)
    assert head.index("pre_cloud_checks(args)") < head.index("_cluster_preflight(args)")


def test_preflight_cli_no_bench_keeps_the_tcp_ports(tmp_path, capsys):
    inv = _inventory(tmp_path)
    args = cli.build_parser().parse_args(["preflight", "--gpu", f"ssh:{inv}", "--run-dir", str(tmp_path / "r"),
                                          "--no-bench"])
    ops = _fake()
    assert cp.main(args, ops=ops) == 0
    assert any(" ports --ports " in s for _, s in ops.scripts)
    assert not any("ib_write_bw -d" in s or "torch.distributed.run --nnodes" in s for _, s in ops.scripts)
    args = cli.build_parser().parse_args(["preflight", "--gpu", f"ssh:{inv}", "--run-dir", str(tmp_path / "r2"),
                                          "--no-bench", "--no-ports"])
    ops = _fake()
    assert cp.main(args, ops=ops) == 0 and not any(" ports --ports " in s for _, s in ops.scripts)


# ------------------------------------------------------------------- yeto run up
@pytest.fixture
def make_up(tmp_path, monkeypatch):
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "rl-up")

    def build(argv, rc=0):
        cfg = {"run": "r1", "store": "s3://b/runs/r1", "pool": {"config": str(tmp_path / "pool.json")},
               "trainer": {"argv": argv, "usd_per_h": 100}, "first_group": {"timeout_s": 0, "poll_s": 0}}
        (tmp_path / "pool.json").write_text(json.dumps({"shard_base": "s"}))
        run = up.Run(cfg)
        run.calls = []

        def cmd(a, *, log, timeout=None):
            run.calls.append(a)
            return rc if "preflight" in a else 0

        run._cmd = cmd
        return run

    return build


SSH_ARGV = ["--gpu", "ssh:6x8xb300", "--rl-node-inventory", "nodes.yaml", "--cluster-prefix", "k31-bm"]


def test_run_up_runs_the_cluster_preflight_then_the_trainer_skips_it(make_up):
    run = make_up(SSH_ARGV)
    assert run.cluster_preflight()
    (call,) = run.calls
    assert call[-8:] == ["preflight", "--gpu", "ssh:6x8xb300", "--rl-node-inventory", "nodes.yaml", "--no-bench",
                         "--cluster-prefix", "k31-bm"]
    assert run.state["steps"]["cluster_preflight"]["ok"]
    assert run.trainer_argv()[-1] == "--skip-preflight"


def test_run_up_cluster_preflight_failure_and_dry_run(make_up, capsys):
    run = make_up(SSH_ARGV, rc=1)
    assert not run.cluster_preflight()
    assert "--skip-preflight" not in run.trainer_argv()
    assert "cluster preflight failed" in "".join(capsys.readouterr())
    run = make_up(SSH_ARGV)
    assert run.cluster_preflight(dry_run=True) and not run.calls
    assert "would run the trainer's cluster preflight: yeto preflight --gpu ssh:6x8xb300" in "".join(
        capsys.readouterr())


def test_run_up_without_an_ssh_trainer_runs_no_cluster_preflight(make_up):
    run = make_up(["--gpu", "modal:2x8xb300", "--cluster-prefix", "r1-trainer"])
    assert run.cluster_preflight() and not run.calls
    run.cluster_preflight_passed = True
    assert "--skip-preflight" not in run.trainer_argv()


def _up_args(**kw):
    base = dict(config="x", rl_command="up", skip_preflight=False, dry_run=False, no_monitor=True,
                trainer_now=False, allow_after_forced_stop=False)
    return types.SimpleNamespace(**{**base, **kw})


def test_run_up_main_refuses_before_ordering_and_skip_reaches_the_trainer(make_up, monkeypatch):
    run = make_up(SSH_ARGV, rc=1)
    order = []
    monkeypatch.setattr(up, "Run", lambda cfg: run)
    monkeypatch.setattr(up, "load_config", lambda path: {})
    monkeypatch.setattr(run, "preflight", lambda: True)
    monkeypatch.setattr(run, "sandbox_prepare", lambda: order.append("sandbox") or True)
    monkeypatch.setattr(run, "pool", lambda: order.append("pool"))
    monkeypatch.setattr(run, "wait_first_group", lambda: None)
    monkeypatch.setattr(run, "trainer", lambda: order.append(run.trainer_argv()))
    monkeypatch.setattr(run, "dashboard", lambda: None)
    assert up.main(_up_args()) == 2 and order == []
    assert up.main(_up_args(dry_run=True)) == 0 and order == []
    assert up.main(_up_args(skip_preflight=True)) == 0
    assert order[:2] == ["sandbox", "pool"] and order[-1][-1] == "--skip-preflight"


def test_the_gate_checks_the_run_image(tmp_path, runs_dir):
    """The run's --rl-image decides the trainer image: the gate's preflight pulls and
    probes it (not only the inventory's), and refuses an inventory that names another."""
    from tests.test_cluster_preflight import IMAGE

    argv = ["--gpu", "ssh:6x8xb300", "--rl-node-inventory", "inv.yaml", "--rl-image", f"docker:{IMAGE}"]
    (cmd,) = gate.commands_for_launch_argv(argv)
    assert cmd[cmd.index("--image") + 1] == IMAGE
    ops = _fake()
    rec = gate.launch_gate(_args(tmp_path, rl_image=f"docker:{IMAGE}"), ops=ops, out=io.StringIO())
    assert rec["reports"][0]["ok"]
    assert any("docker pull" in s and IMAGE in s for _, s in ops.scripts)
    with pytest.raises(ValueError, match="is not the run's --rl-image"):
        gate.launch_gate(_args(tmp_path, rl_image="docker:ghcr.io/o/other@sha256:" + "c" * 64), ops=_fake(),
                         out=io.StringIO())
