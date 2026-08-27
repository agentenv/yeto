"""Auto-fleet planning in `yeto launch` (--gpu omitted)."""

from __future__ import annotations

import sys

import pytest

from yeto import cli
from yeto.shape.ilp import Candidate, Plan
from yeto.shape.plan import ShapeResult

BASE = ["launch", "--model", "gemma4", "--data", "org/data"]


def _args(extra):
    return cli.build_parser().parse_args(BASE + extra)


def test_launch_cli_has_deterministic_lm_seed_by_default():
    assert _args([]).seed == 0
    assert _args(["--seed", "29"]).seed == 29


def _result(counts):
    cands = [
        Candidate(
            key=k, region="us-east-2", gpu="H100", instance_type="p5.48xlarge",
            nodes=1, gpus_per_node=8, vcpus_per_island=192, price_per_hour=13.0,
            eff_tflops=2630.0, quota_bucket=("us-east-2", "L-7212CCBC"), score=9,
        )
        for k in counts
    ]
    plan = Plan(counts=dict(counts), total_tflops=2630.0, total_cost=13.4, binding=[])
    return ShapeResult(
        plan=plan, candidates=cands, rejections=[], warnings=[], weights_gb=66.0,
        shard="fsdp", est_cost=13.4, price_margin=0.15, head_cost=0.4, fetch_seconds=0.1,
    )


def test_launch_cli_defaults_to_native_mask_and_accepts_explicit_legacy():
    assert _args([]).assistant_mask_mode == "native"
    assert _args(["--assistant-mask-mode", "legacy"]).assistant_mask_mode == "legacy"


def test_launch_cli_accepts_data_normalization_and_qlora_options():
    assert _args([]).data_format == "auto"
    assert _args(["--data-format", "alpaca"]).data_format == "alpaca"
    assert _args([]).base_quantization == "none"
    assert _args(["--base-quantization", "nf4"]).base_quantization == "nf4"


def test_launch_cli_accepts_one_parent_adapter_mode():
    resumed = _args(["--resume-from", "/tmp/adapter"])
    assert resumed.resume_from == "/tmp/adapter"
    assert resumed.branch_from is None
    branched = _args(["--branch-from", "s3://bucket/adapter"])
    assert branched.branch_from == "s3://bucket/adapter"
    with pytest.raises(SystemExit):
        _args(
            [
                "--resume-from",
                "/tmp/adapter-a",
                "--branch-from",
                "/tmp/adapter-b",
            ]
        )


def test_merge_cli_parses_safe_shard_options():
    args = cli.build_parser().parse_args(
        [
            "merge",
            "--adapter-dir",
            "/tmp/adapter",
            "--output-dir",
            "/tmp/merged",
            "--device",
            "cuda",
            "--dtype",
            "bf16",
            "--max-shard-size",
            "2GB",
        ]
    )
    assert args.command == "merge"
    assert args.max_shard_size == "2GB"
    assert args.dtype == "bf16"


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--branch-from", "/tmp/adapter", "--tuning", "full"], "--tuning lora"),
        (
            ["--branch-from", "/tmp/adapter", "--island-backend", "megatron"],
            "--island-backend torch",
        ),
        (
            ["--branch-from", "/tmp/adapter", "--external-learners", "1"],
            "external MLX",
        ),
        (
            ["--branch-from", "/tmp/adapter", "--training-mode", "rl"],
            "not supported with --training-mode rl",
        ),
    ],
)
def test_parent_adapter_invalid_profiles_fail_before_launch(extra, message):
    assert message in cli._fleet_args_error(_args(["--gpu", "aws:8xa100", *extra]))


def test_gpu_with_budget_rejected(capsys):
    assert cli.main(BASE + ["--gpu", "aws:8xa100", "--budget", "10"]) == 1
    assert "drop them or drop --gpu" in capsys.readouterr().err


def test_neither_gpu_nor_objective_rejected(capsys):
    assert cli.main(list(BASE)) == 1
    assert "auto-planned fleet" in capsys.readouterr().err


def test_qlora_auto_fleet_is_rejected_until_memory_model_is_calibrated(capsys):
    assert cli.main(BASE + ["--budget", "40", "--base-quantization", "nf4"]) == 1
    assert "not calibrated" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--gpu", "aws:8xa100", "--base-quantization", "nf4", "--shard", "fsdp"], "--shard ddp"),
        (["--gpu", "aws:8xa100", "--base-quantization", "nf4", "--tuning", "full"], "--tuning lora"),
        (["--gpu", "aws:8xa100", "--base-quantization", "nf4", "--external-learners", "1"], "external MLX"),
    ],
)
def test_qlora_invalid_profiles_fail_before_launch(extra, message):
    assert message in cli._fleet_args_error(_args(extra))


def test_auto_fleet_fills_args_and_skips_prompt_with_confirm(monkeypatch):
    captured = {}

    def fake_build_shape(**kw):
        captured.update(kw)
        return _result({"aws:8xh100@us-east-2": 2})

    monkeypatch.setattr("yeto.shape.plan.build_shape", fake_build_shape)
    args = _args(["--budget", "40", "--confirm"])
    assert cli._resolve_auto_fleet(args) == 0
    assert args.gpu == "aws:8xh100@us-east-2,aws:8xh100@us-east-2"
    assert args.shard == "fsdp"
    assert args.disk_size >= 199  # 66 GB * 1.5 + 100
    assert captured["budget"] == 40.0 and captured["target_tflops"] is None


def test_auto_fleet_flops_objective_passthrough(monkeypatch):
    captured = {}

    def fake_build_shape(**kw):
        captured.update(kw)
        return _result({"aws:8xh100@us-east-2": 1})

    monkeypatch.setattr("yeto.shape.plan.build_shape", fake_build_shape)
    args = _args(["--flops", "5000", "--confirm"])
    assert cli._resolve_auto_fleet(args) == 0
    assert captured["target_tflops"] == 5000.0 and captured["budget"] is None


def test_auto_fleet_requires_confirmation(monkeypatch, capsys):
    monkeypatch.setattr(
        "yeto.shape.plan.build_shape", lambda **kw: _result({"aws:8xh100@us-east-2": 1})
    )
    args = _args(["--budget", "40"])
    # Non-interactive without --confirm: refuse.
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    assert cli._resolve_auto_fleet(args) == 1
    assert "--confirm" in capsys.readouterr().err
    # Interactive: 'n' aborts with the distinct code, 'y' proceeds.
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert cli._resolve_auto_fleet(_args(["--budget", "40"])) == 2
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    assert cli._resolve_auto_fleet(_args(["--budget", "40"])) == 0


def test_empty_plan_stops_launch(monkeypatch):
    empty = _result({})
    empty.plan.counts = {}
    monkeypatch.setattr("yeto.shape.plan.build_shape", lambda **kw: empty)
    assert cli._resolve_auto_fleet(_args(["--budget", "1"])) == 1
