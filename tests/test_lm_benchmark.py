"""Pure-logic tests for the LM DiLoCo benchmark harness."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "benchmark_lm_diloco", ROOT / "scripts" / "compare_diloco.py"
)
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def _args(**overrides):
    values = {
        "model": "gemma4",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_targets": "auto",
        "base_quantization": "none",
        "seq_len": 512,
        "micro_batch_size": 1,
        "grad_accum": 2,
        "inner_lr": 3e-4,
        "weight_decay": 0.01,
        "warmup_steps": 10,
        "train_on": "assistant",
        "assistant_mask_mode": "native",
        "gradient_checkpointing": "auto",
        "wan_streams": 4,
        "device": "cuda",
        "shard": "fsdp",
        "learner_gpus": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_presets_match_production_and_explicit_stress_arm():
    arms = benchmark.select_arms("all", fragments=6)
    assert [arm.name for arm in arms] == [
        "m2",
        "m4",
        "alpha0",
        "q4",
        "serial",
        "noheloco",
        "strided",
        "iso",
        "direct-rda",
        "unthrottled",
    ]
    assert all(arm.fragments == 6 for arm in arms)
    assert benchmark.PRESETS["m2"].sync_interval_steps == 24.0
    assert benchmark.PRESETS["unthrottled"].sync_interval_steps == 0.0
    assert benchmark.PRESETS["iso"].matrix_merge == "iso"
    direct = benchmark.PRESETS["direct-rda"]
    assert (direct.outer_lr, direct.outer_momentum, direct.merge_alpha) == (1.0, 0.0, 0.0)


def test_settings_must_be_known_and_nonempty():
    with pytest.raises(ValueError, match="at least one"):
        benchmark.select_arms("")
    with pytest.raises(ValueError, match="unknown"):
        benchmark.select_arms("m2,not-an-arm")


def test_matching_baseline_has_same_steps_and_raw_tokens():
    budget = 500_000
    baseline_steps = benchmark.steps_for(
        budget, 1, 512, learners=1, world=4, grad_accum=2
    )
    diloco_steps = benchmark.steps_for(
        budget, 1, 512, learners=2, world=2, grad_accum=2
    )
    assert baseline_steps == diloco_steps
    assert benchmark.processed_tokens(baseline_steps, 1, 512, 4, 2) == (
        benchmark.processed_tokens(diloco_steps, 1, 512, 4, 2)
    )


def test_commands_encode_matching_topologies_and_seed():
    args = _args()
    train = Path("/tmp/train.jsonl")
    baseline = benchmark.learner_command(
        args,
        Path("/tmp/baseline-m2"),
        nproc=4,
        learner_id=0,
        num_learners=1,
        syncer="none",
        max_steps=100,
        seed=29,
        train_data=train,
        arm=None,
    )
    diloco = benchmark.learner_command(
        args,
        Path("/tmp/m2"),
        nproc=2,
        learner_id=1,
        num_learners=2,
        syncer="127.0.0.1:1234",
        max_steps=100,
        seed=29,
        train_data=train,
        arm=benchmark.PRESETS["q4"],
    )
    assert "--nproc_per_node=4" in baseline
    assert "--nproc_per_node=2" in diloco
    assert baseline[baseline.index("--seed") + 1] == "29"
    assert diloco[diloco.index("--grad-accum") + 1] == "2"
    assert diloco[diloco.index("--tokenize") + 1] == "stream"
    assert diloco[diloco.index("--stream-workers") + 1] == "0"
    assert diloco[diloco.index("--assistant-mask-mode") + 1] == "native"
    assert diloco[diloco.index("--wire-dtype") + 1] == "q4"
    assert diloco[diloco.index("--base-quantization") + 1] == "none"
    assert "--wire-dtype" not in baseline


def test_benchmark_mask_mode_is_fixed_for_training_and_subprocess_eval(monkeypatch, tmp_path):
    args = _args(
        assistant_mask_mode="legacy",
        eval_device="cpu",
        base_quantization="none",
    )
    command = benchmark.learner_command(
        args,
        tmp_path / "arm",
        learner_id=0,
        num_learners=1,
        syncer="none",
        max_steps=1,
        arm=None,
    )
    assert command[command.index("--assistant-mask-mode") + 1] == "legacy"

    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return subprocess.CompletedProcess(
            command,
            0,
            stdout='EVAL_JSON {"loss_per_token": 1.0}\n',
            stderr="",
        )

    monkeypatch.setattr(benchmark, "wait_for_free_gpus", lambda *args, **kwargs: None)
    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    result = benchmark.eval_in_subprocess(args, None, tmp_path / "eval.jsonl")

    assert result == {"loss_per_token": 1.0}
    eval_command = captured["command"]
    assert eval_command[eval_command.index("--assistant-mask-mode") + 1] == "legacy"


def test_benchmark_parser_defaults_to_native_mask_and_accepts_legacy():
    parser = benchmark.build_parser()
    assert parser.parse_args(["--data", "rows.jsonl"]).assistant_mask_mode == "native"
    assert (
        parser.parse_args(
            ["--data", "rows.jsonl", "--assistant-mask-mode", "legacy"]
        ).assistant_mask_mode
        == "legacy"
    )


@pytest.mark.parametrize("value", ["", "1,1", "1,nope"])
def test_parse_seeds_rejects_invalid_lists(value):
    with pytest.raises(ValueError):
        benchmark.parse_seeds(value)


def test_cuda_env_respects_existing_visibility(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,6,7,9")
    env = benchmark.cuda_env(1, 2, "cuda")
    assert env["CUDA_VISIBLE_DEVICES"] == "6,7"
    with pytest.raises(ValueError):
        benchmark.cuda_env(3, 2, "cuda")
    assert benchmark.cuda_env(0, 1, "cpu") is None


def test_materialize_s3_uses_read_only_sync(monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark.shutil, "which", lambda name: "/usr/bin/aws")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        (tmp_path / "source" / "train.jsonl").write_text("{}\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(benchmark.subprocess, "run", run)
    got = benchmark.materialize_data_source("s3://bucket/data", tmp_path / "source")
    assert got == str((tmp_path / "source").resolve())
    assert calls[0][0] == [
        "aws",
        "s3",
        "sync",
        "s3://bucket/data",
        str(tmp_path / "source"),
        "--only-show-errors",
    ]
    assert calls[0][1]["check"] is True


def test_split_data_requires_messages_rows(tmp_path):
    with pytest.raises(SystemExit, match="messages-format"):
        benchmark.split_data(
            [{"prompt": "not a conversation"}, {"prompt": "held out"}],
            tmp_path,
            eval_rows=1,
            max_rows=None,
        )


def test_partial_results_round_trip(tmp_path):
    records = [
        {"kind": "base", "arm": "base", "seed": None},
        {"kind": "diloco", "arm": "m2", "seed": 17},
    ]
    benchmark.write_partial_results(tmp_path, records)
    assert benchmark.load_partial_results(tmp_path) == records
    assert benchmark._record_key(records[1]) == ("diloco", "m2", 17)


def test_resume_reuses_verified_splits_and_rejects_recipe_changes(tmp_path):
    work = tmp_path / "work"
    report = tmp_path / "report"
    work.mkdir()
    train = work / "train.jsonl"
    evaluation = work / "eval.jsonl"
    train.write_text('{"messages": []}\n', encoding="utf-8")
    evaluation.write_text('{"messages": []}\n', encoding="utf-8")
    args = _args(
        seeds="17,29,43",
        work_dir=work,
        report_dir=report,
        resume=False,
        overwrite=False,
        dry_run=False,
        eval_only=False,
        adapter_dir=None,
    )
    arms = benchmark.select_arms("m2")
    from yeto.benchmark_resume import build_data_manifest

    manifest = build_data_manifest(
        work,
        train,
        evaluation,
        train_rows=1,
        eval_rows=1,
    )
    benchmark.write_config(args, arms, manifest)

    args.resume = True
    assert benchmark.load_resume_data(args, arms) == (train, evaluation, 1)
    args.inner_lr = 1e-4
    with pytest.raises(ValueError, match="arguments.inner_lr"):
        benchmark.load_resume_data(args, arms)


def test_tape_summary_and_wire_estimate(tmp_path):
    tape = tmp_path / "tape.jsonl"
    rows = [
        {
            "fragment": 0,
            "step": 1,
            "ms": 10,
            "responders": [
                {"c_steps": 20, "c_tokens": 10_240, "base_version": 0},
                {"c_steps": 24, "c_tokens": 12_288, "base_version": 0},
            ],
        },
        {
            "fragment": 0,
            "step": 2,
            "ms": 14,
            "responders": [
                {"c_steps": 26, "c_tokens": 13_312, "base_version": 0}
            ],
        },
    ]
    tape.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    summary = benchmark.summarize_tape(tape, learners=2)
    assert summary["merges"] == 2
    assert summary["mean_h"] == pytest.approx(70 / 3)
    assert summary["participation_rate"] == 0.75
    assert summary["mean_staleness"] == pytest.approx(1 / 3)
    assert summary["mean_tokens_per_response"] == pytest.approx(35_840 / 3)
    bf16 = benchmark.estimate_tensor_bytes([256], tape, "bf16", learners=2)
    q4 = benchmark.estimate_tensor_bytes([256], tape, "q4", learners=2)
    assert q4 < bf16


def test_aggregate_uses_seed_matched_baseline():
    records = []
    for seed, baseline_loss, diloco_loss in [(17, 2.0, 2.1), (29, 4.0, 4.2)]:
        common = {
            "seed": seed,
            "learners": 2,
            "total_gpus": 4,
            "processed_tokens": 1000,
            "processed_target_tokens": 400,
            "target_density": 0.4,
            "wall_s": 10.0,
            "tokens_per_s": 100.0,
            "target_tokens_per_s": 40.0,
            "gpu_hours": 4 / 360,
            "estimated_cost": None,
            "tape": None,
        }
        records.append(
            {
                **common,
                "kind": "baseline",
                "arm": "baseline-m2",
                "eval": {
                    "loss_per_token": baseline_loss,
                    "perplexity": 2.718281828**baseline_loss,
                },
            }
        )
        records.append(
            {
                **common,
                "kind": "diloco",
                "arm": "m2",
                "eval": {
                    "loss_per_token": diloco_loss,
                    "perplexity": 2.718281828**diloco_loss,
                },
            }
        )
    summary = benchmark.aggregate_records(records)
    m2 = next(item for item in summary if item["arm"] == "m2")
    assert m2["delta_mean_pct"] == pytest.approx(5.0)
    assert m2["runs"] == 2
    assert m2["target_density_mean"] == 0.4


def test_training_log_summary_tracks_lm_target_tokens(tmp_path):
    first = tmp_path / "learner-0.log"
    second = tmp_path / "learner-1.log"
    first.write_text(
        "inner loop done at local_step=10 global_step=2 "
        "raw_tokens=5120 target_tokens=2048\n",
        encoding="utf-8",
    )
    second.write_text(
        "prefix inner loop done at local_step=10 global_step=2 "
        "raw_tokens=5120 target_tokens=1536\n",
        encoding="utf-8",
    )
    summary = benchmark.summarize_training_logs([first, second])
    assert summary == {
        "reported_ranks": 2,
        "processed_tokens": 10_240,
        "processed_target_tokens": 3_584,
        "target_density": 0.35,
    }


def test_lm_fairness_rejects_target_token_mismatch():
    arm = benchmark.PRESETS["m2"]
    records = [
        {
            "kind": "baseline",
            "seed": 17,
            "learners": 2,
            "processed_target_tokens": 100,
        }
    ]
    benchmark.validate_target_token_match(
        records, arm, 17, {"processed_target_tokens": 100}
    )
    with pytest.raises(RuntimeError, match="target-token mismatch"):
        benchmark.validate_target_token_match(
            records, arm, 17, {"processed_target_tokens": 99}
        )


def test_eval_fairness_requires_same_held_out_target_count():
    records = [{"kind": "base", "eval": {"trained_tokens": 1234}}]
    benchmark.validate_eval_token_match(records, {"trained_tokens": 1234})
    with pytest.raises(RuntimeError, match="held-out target-token mismatch"):
        benchmark.validate_eval_token_match(records, {"trained_tokens": 1233})
