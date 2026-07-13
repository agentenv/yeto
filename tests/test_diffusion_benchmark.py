"""Pure-logic tests for the independent diffusion benchmark harness."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "benchmark_diffusion_diloco",
    ROOT / "scripts" / "benchmark_diffusion_diloco.py",
)
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def _args(**overrides):
    values = dict(
        model="ltx-video",
        shard="fsdp",
        lora_r=16,
        lora_alpha=32,
        lora_targets="auto",
        micro_batch_size=1,
        grad_accum=2,
        inner_lr=3e-4,
        weight_decay=0.01,
        warmup_steps=10,
        stream_workers=0,
        wan_streams=4,
        device="cuda",
        diffusion_adapter=None,
        cache_latents=False,
        cache_text_embeds=False,
        bucket_by_shape=True,
        image_column="image",
        video_column="video",
        prompt_column="prompt",
        latent_column="latents",
        text_embeds_column="prompt_embeds",
        text_attention_mask_column="prompt_attention_mask",
        pooled_text_embeds_column="pooled_prompt_embeds",
        diffusion_loss_weighting="none",
        diffusion_min_snr_gamma=5.0,
        height=256,
        width=256,
        resize_mode="center-crop",
        num_frames=49,
        fps=9.3,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_presets_cover_quality_and_system_axes():
    arms = benchmark.select_arms("all", fragments=6)
    names = {arm.name for arm in arms}
    assert names == {
        "m2",
        "m4",
        "alpha0",
        "q4",
        "serial",
        "noheloco",
        "strided",
        "direct-rda",
        "unthrottled",
    }
    assert all(arm.fragments == 6 for arm in arms)
    assert benchmark.PRESETS["m2"].sync_interval_steps == 24.0
    assert benchmark.PRESETS["unthrottled"].sync_interval_steps == 0.0
    direct = benchmark.PRESETS["direct-rda"]
    assert direct.outer_lr == 1.0
    assert direct.outer_momentum == 0.0
    assert direct.merge_alpha == 0.0


def test_sample_budget_is_equal_for_sync_and_diloco():
    budget = 10_000
    micro = 1
    accum = 2
    m = 4
    ranks_per_island = 2
    total_ranks = m * ranks_per_island

    baseline_steps = benchmark.steps_for_samples(budget, micro, accum, total_ranks)
    learner_steps = benchmark.steps_for_samples(budget, micro, accum, total_ranks)

    assert baseline_steps == learner_steps == 625
    assert benchmark.processed_samples(
        baseline_steps, micro, accum, total_ranks
    ) == benchmark.processed_samples(learner_steps, micro, accum, total_ranks)
    assert benchmark.effective_grad_accum(2, 4) == 2
    assert benchmark.effective_grad_accum(4, 1) == 1


def test_learner_commands_hold_recipe_fixed_and_only_arm_gets_sync_flags():
    args = _args()
    train = Path("/tmp/diffusion/train")
    arm = benchmark.PRESETS["q4"]
    diloco = benchmark.learner_command(
        args,
        train,
        Path("/tmp/diffusion/arm"),
        nproc=2,
        learner_id=1,
        num_learners=2,
        syncer="127.0.0.1:1",
        max_steps=20,
        seed=17,
        arm=arm,
    )
    baseline = benchmark.learner_command(
        args,
        train,
        Path("/tmp/diffusion/base"),
        nproc=4,
        learner_id=0,
        num_learners=1,
        syncer="none",
        max_steps=20,
        seed=17,
        arm=None,
    )

    assert "torch.distributed.run" in diloco
    assert "--nproc_per_node=2" in diloco
    assert "--nproc_per_node=4" in baseline
    for flag in (
        "--model",
        "--data",
        "--shard",
        "--lora-r",
        "--lora-alpha",
        "--lora-targets",
        "--micro-batch-size",
        "--grad-accum",
        "--inner-lr",
        "--weight-decay",
        "--warmup-steps",
        "--seed",
        "--height",
        "--width",
        "--resize-mode",
        "--num-frames",
    ):
        assert diloco[diloco.index(flag) + 1] == baseline[baseline.index(flag) + 1]
    assert diloco[diloco.index("--wire-dtype") + 1] == "q4"
    assert "--wire-dtype" not in baseline


def test_single_rank_learner_still_uses_torchrun_environment():
    command = benchmark._distributed_prefix(1)

    assert "torch.distributed.run" in command
    assert "--nproc_per_node=1" in command


def test_syncer_command_uses_all_learner_quorum_and_explicit_h(tmp_path):
    args = SimpleNamespace(grace_ms=1000)
    arm = benchmark.PRESETS["m4"]
    command = benchmark.syncer_command(args, arm, 1234, tmp_path, total_steps=99)

    assert command[command.index("--learners") + 1] == "4"
    assert command[command.index("--quorum") + 1] == "4"
    assert command[command.index("--sync-interval-steps") + 1] == "24.0"
    assert command[command.index("--checkpoint-every") + 1] == "1"


def test_seed_parser_rejects_empty_invalid_and_duplicate_values():
    assert benchmark.parse_seeds("17, 29,43") == [17, 29, 43]
    for value in ("", "17,x", "17,17"):
        with pytest.raises(ValueError):
            benchmark.parse_seeds(value)


def test_cuda_env_respects_parent_visible_device_mapping(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,7,8,9")
    env = benchmark.cuda_env(1, 2, "cuda")
    assert env["CUDA_VISIBLE_DEVICES"] == "7,8"
    with pytest.raises(ValueError):
        benchmark.cuda_env(3, 2, "cuda")
    assert benchmark.cuda_env(0, 1, "cpu") is None


def test_wait_for_free_gpus_ignores_processes_on_hidden_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    outputs = iter(
        [
            "0, GPU-visible\n1, GPU-hidden\n",
            "GPU-hidden, 123, python, 3000\nGPU-visible, 456, python, 0\n",
        ]
    )

    def run(*_args, **_kwargs):
        return SimpleNamespace(stdout=next(outputs))

    monkeypatch.setattr(benchmark.subprocess, "run", run)
    benchmark.wait_for_free_gpus("cuda", timeout_s=0)


def test_materialize_data_source_syncs_s3_prefix(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(benchmark.shutil, "which", lambda name: "/usr/bin/aws")

    def run(command, check):
        calls.append((command, check))
        destination = Path(command[4])
        (destination / "train.jsonl").write_text("{}\n")

    monkeypatch.setattr(benchmark.subprocess, "run", run)
    destination = tmp_path / "source-data"

    got = benchmark.materialize_data_source("s3://bucket/dataset", destination)

    assert got == str(destination.resolve())
    assert calls == [
        (
            [
                "aws",
                "s3",
                "sync",
                "s3://bucket/dataset",
                str(destination),
                "--only-show-errors",
            ],
            True,
        )
    ]
    assert benchmark.materialize_data_source("org/dataset", tmp_path / "unused") == "org/dataset"


def test_materialize_data_source_requires_aws_cli(monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="AWS CLI"):
        benchmark.materialize_data_source("s3://bucket/dataset", tmp_path / "source-data")


def test_partial_results_round_trip_atomically(tmp_path):
    records = [
        {"kind": "base", "arm": "base", "seed": None},
        {"kind": "diloco", "arm": "m2", "seed": 17},
    ]

    benchmark.write_partial_results(tmp_path, records)

    assert benchmark.load_partial_results(tmp_path) == records
    assert not (tmp_path / "results.jsonl.tmp").exists()
    assert benchmark._record_key(records[1]) == ("diloco", "m2", 17)


def test_tape_summary_reports_measured_h_participation_and_staleness(tmp_path):
    tape = tmp_path / "tape.jsonl"
    records = [
        {
            "step": 1,
            "fragment": 0,
            "ms": 10,
            "responders": [
                {"id": 0, "base_version": 0, "c_steps": 20},
                {"id": 1, "base_version": 0, "c_steps": 24},
            ],
        },
        {
            "step": 9,
            "fragment": 0,
            "ms": 14,
            "responders": [{"id": 0, "base_version": 0, "c_steps": 28}],
        },
    ]
    tape.write_text("".join(json.dumps(record) + "\n" for record in records))

    summary = benchmark.summarize_tape(tape, learners=2)

    assert summary["merges"] == 2
    assert summary["mean_h"] == 24.0
    assert summary["participation_rate"] == 0.75
    assert summary["mean_sync_ms"] == 12.0
    assert summary["mean_staleness"] == pytest.approx(1 / 3)
    assert summary["max_staleness"] == 1


def test_aggregate_pairs_each_arm_with_same_seed_and_m_baseline():
    records = []
    for seed, baseline_loss, arm_loss in ((17, 2.0, 2.2), (29, 4.0, 4.2)):
        records.append(
            {
                "kind": "baseline",
                "arm": "baseline-m2",
                "seed": seed,
                "learners": 2,
                "total_gpus": 2,
                "processed_samples": 100,
                "wall_s": 10.0,
                "samples_per_s": 10.0,
                "gpu_hours": 0.0,
                "eval": {"loss_per_element": baseline_loss},
                "tape": None,
            }
        )
        records.append(
            {
                "kind": "diloco",
                "arm": "m2",
                "seed": seed,
                "learners": 2,
                "total_gpus": 2,
                "processed_samples": 100,
                "wall_s": 11.0,
                "samples_per_s": 9.0,
                "gpu_hours": 0.0,
                "eval": {"loss_per_element": arm_loss},
                "tape": {"mean_h": 24.0, "participation_rate": 1.0},
            }
        )

    aggregates = benchmark.aggregate_records(records)
    arm = next(item for item in aggregates if item["arm"] == "m2")

    assert arm["loss_mean"] == 3.2
    assert arm["delta_mean_pct"] == pytest.approx(7.5)
    assert arm["mean_h"] == 24.0
    assert arm["participation_rate"] == 1.0
