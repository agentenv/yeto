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


# --- yeto shape --training-mode rl: the island the RL launcher will build -----


def _shape_args(extra):
    return cli.build_parser().parse_args(["shape", "--model", "gemma4", "--budget", "60"] + extra)


def test_rl_island_shape_from_shape_flags():
    assert cli.rl_island_shape(_shape_args([])) is None  # sft: memory-model sizing
    split = cli.rl_island_shape(_shape_args([
        "--training-mode", "rl", "--parameter-mode", "full", "--rollout-num-gpus", "4", "--actor-gpus", "4",
    ]))
    assert (split.gpus_per_node, split.num_nodes, split.single_node_only) == (8, 1, True)
    assert split.needs_container_image and split.spot_needs_storage and split.label == "rl"
    assert split.note == "actor 4 + rollout 4, disjoint"
    colocated = cli.rl_island_shape(_shape_args(["--training-mode", "rl", "--actor-gpus", "8", "--actor-nodes", "2"]))
    assert (colocated.gpus_per_node, colocated.num_nodes, colocated.single_node_only) == (8, 2, False)
    with pytest.raises(ValueError, match="single-node"):
        cli.rl_island_shape(_shape_args([
            "--training-mode", "rl", "--parameter-mode", "full", "--rollout-num-gpus", "4", "--actor-nodes", "2",
        ]))
    with pytest.raises(ValueError, match="--rollout-num-gpus >= 1"):
        cli.rl_island_shape(_shape_args(["--training-mode", "rl", "--parameter-mode", "full"]))


# --- Modal islands: routing decided before any resource is touched ------------


def _specs(gpu: str):
    from yeto.gpu_spec import parse_gpu_spec

    return parse_gpu_spec(gpu)


def test_modal_entries_get_modal_names_and_mix_with_sky_entries():
    from yeto.launcher import learner_cluster_names

    names = learner_cluster_names("run", _specs("aws:8xh100@us-east-1,modal:8xh100,modal:8xh100@us"))
    assert names == ["run-l0-us-east-1", "run-l1-modal", "run-l2-modal"]


def test_modal_prerequisites_fail_before_launch():
    from yeto.launcher import check_cloud_prerequisites

    args = _args(["--gpu", "modal:2x4xh100"])
    with pytest.raises(ValueError, match="whole nodes: H100:8 per container, not H100:4"):
        check_cloud_prerequisites(_specs(args.gpu), args=args, modal_ok=True)
    args = _args(["--gpu", "modal:8xh100"])
    with pytest.raises(ValueError, match="need a Modal token"):
        check_cloud_prerequisites(_specs(args.gpu), args=args, modal_ok=False)
    args = _args(["--gpu", "modal:8xh100", "--syncer-public-addr", "no-port"])
    with pytest.raises(ValueError, match="HOST:PORT"):
        check_cloud_prerequisites(_specs(args.gpu), args=args, modal_ok=True)
    args = cli.build_parser().parse_args(["launch", "--model", "gemma4", "--data", "s3://bucket/data", "--gpu", "modal:8xh100"])
    with pytest.raises(ValueError, match="cannot mount object-store data"):
        check_cloud_prerequisites(_specs(args.gpu), args=args, modal_ok=True)
    # A valid mixed fleet passes without touching Modal or sky.
    args = _args(["--gpu", "aws:8xh100@us-east-1,modal:8xh100@us", "--syncer-public-addr", "1.2.3.4:5000"])
    check_cloud_prerequisites(_specs(args.gpu), args=args, modal_ok=True)


def test_modal_rl_island_needs_a_digest_pinned_image():
    from yeto.launcher import check_cloud_prerequisites

    args = _args(["--gpu", "modal:8xh100", "--training-mode", "rl", "--rl-image", "docker:ghcr.io/x/miles:latest"])
    with pytest.raises(ValueError, match="must pin a digest"):
        check_cloud_prerequisites(_specs(args.gpu), args=args, modal_ok=True)


def test_modal_island_config_reuses_the_sky_task_script(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from yeto.launcher import _rl_checkpoint_storage_name, build_modal_island_config

    monkeypatch.setenv("HOME", str(tmp_path))  # no real HF token leaks in
    (tmp_path / ".cache" / "huggingface").mkdir(parents=True)
    (tmp_path / ".cache" / "huggingface" / "token").write_text("hf_local\n")
    task = SimpleNamespace(
        run="MASTER_ADDR=$(echo \"$SKYPILOT_NODE_IPS\" | head -n1)\ntorchrun -m yeto.learner",
        envs={"SYNCER_ADDR": "10.0.0.5:5000", "LEARNER_ID": "1", "HF_HUB_ENABLE_HF_TRANSFER": "1"},
        setup="pip install -q -r requirements.txt",
    )
    args = _args(["--gpu", "aws:8xh100@us-east-1,modal:8xh100@us", "--cluster-prefix", "run"])
    (_, spec) = _specs(args.gpu)
    cfg = build_modal_island_config(args, spec, 1, task, "1.2.3.4:5000")
    assert cfg.app_name == "yeto-run" and cfg.learner_id == 1 and cfg.training_mode == "sft"
    assert cfg.run_script == task.run and cfg.setup_script is None
    assert cfg.envs["SYNCER_ADDR"] == "1.2.3.4:5000"  # the Modal-reachable address wins
    assert cfg.envs["LEARNER_ID"] == "1" and cfg.envs["HF_TOKEN"] == "hf_local"
    assert cfg.region == "us" and cfg.gpu_request == "H100:8"
    assert "torch" in cfg.pip_requirements and any(r.startswith("transformers") for r in cfg.pip_requirements)
    assert cfg.volume_name is None
    assert "YETO_RL_EXTERNAL_ROUTER" not in cfg.envs
    # RL + spot: the digest image, the sky setup baked into it, and a
    # checkpoint volume at the sky island's mount path.
    digest = "docker:ghcr.io/x/miles@sha256:" + "d" * 64
    args = _args([
        "--gpu", "modal:8xh100", "--cluster-prefix", "run", "--training-mode", "rl",
        "--rl-image", digest, "--rl-completed-groups-path", "~/yeto-rl/groups.jsonl",
    ])
    (spec,) = _specs(args.gpu)
    cfg = build_modal_island_config(args, spec, 0, task, "1.2.3.4:5000")
    assert cfg.training_mode == "rl" and cfg.image_ref == digest[len("docker:"):]
    assert cfg.setup_script == task.setup and cfg.pip_requirements == ()
    assert cfg.volume_name == _rl_checkpoint_storage_name("run", 0)
    assert cfg.volume_mount == "/root/yeto-rl"
    # Modal RL islands start the SGLang router themselves (Miles' 30 s
    # router deadline is missed on Modal); SFT islands do not.
    assert cfg.envs["YETO_RL_EXTERNAL_ROUTER"] == "1"
    cfg.validate()


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
