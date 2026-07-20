"""CPU-only logic tests for the standalone A100 kernel benchmark."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_a100_kernels", ROOT / "scripts" / "benchmark_a100_kernels.py"
)
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def test_variant_plan_has_a_stable_reference_and_all_combinations():
    assert [variant.name for variant in benchmark.select_variants("all")] == [
        "native-sdpa",
        "native-flash-attn-2",
        "liger-sdpa",
        "liger-flash-attn-2",
    ]
    selected = benchmark.select_variants("liger-sdpa")
    assert [variant.name for variant in selected] == ["native-sdpa", "liger-sdpa"]
    with pytest.raises(ValueError, match="unknown variants"):
        benchmark.select_variants("unknown")


def test_percentile_interpolates_and_validates_input():
    assert benchmark.percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert benchmark.percentile([1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.85)
    with pytest.raises(ValueError):
        benchmark.percentile([], 0.5)


def test_gradient_parity_gate_checks_loss_keys_and_values():
    reference = {"layer.weight": torch.tensor([2.0, 0.5, 3.0, -1.0])}
    passed = benchmark.compare_parity(
        12.0,
        {"layer.weight": reference["layer.weight"].clone()},
        12.0,
        reference,
        rtol=1e-5,
        atol=1e-6,
        parameter_deltas={"layer.weight": torch.tensor([0.1, 0.2])},
        reference_parameter_deltas={
            "layer.weight": torch.tensor([0.1, 0.2])
        },
    )
    assert passed["passed"] and passed["checked_parameter_delta_tensors"] == 1

    bad = benchmark.compare_parity(
        12.0,
        {"layer.weight": torch.tensor([2.0, 0.5, 9.0, -1.0])},
        12.0,
        reference,
        rtol=1e-5,
        atol=1e-6,
    )
    assert not bad["passed"] and "gradient parity" in bad["reason"]

    missing = benchmark.compare_parity(
        12.0, {}, 12.0, reference, rtol=1e-5, atol=1e-6
    )
    assert not missing["passed"] and "key mismatch" in missing["reason"]

    bad_delta = benchmark.compare_parity(
        12.0,
        {"layer.weight": reference["layer.weight"].clone()},
        12.0,
        reference,
        rtol=1e-5,
        atol=1e-6,
        parameter_deltas={"layer.weight": torch.tensor([0.1, 0.9])},
        reference_parameter_deltas={
            "layer.weight": torch.tensor([0.1, 0.2])
        },
    )
    assert not bad_delta["passed"] and "parameter-delta parity" in bad_delta["reason"]


def test_benchmark_argument_validation_is_cpu_safe():
    args = benchmark.build_parser().parse_args([])
    benchmark.validate_args(args)
    assert args.dtype == "bf16"
    assert args.model == "Qwen/Qwen2.5-1.5B-Instruct"
    assert args.trial_index == 1
    assert args.steps > 0 and args.warmup_steps > 0

    broken = SimpleNamespace(**vars(args))
    broken.target_fraction = 0
    with pytest.raises(ValueError, match="target-fraction"):
        benchmark.validate_args(broken)


def test_model_revision_is_resolved_to_an_immutable_hub_commit(monkeypatch):
    calls = []

    class API:
        def model_info(self, model_id, revision):
            calls.append((model_id, revision))
            return SimpleNamespace(sha="a" * 40)

    monkeypatch.setattr(huggingface_hub, "HfApi", API)
    assert benchmark.resolve_model_revision("Qwen/model", "release") == "a" * 40
    assert calls == [("Qwen/model", "release")]


def test_explicit_commit_remains_usable_when_hub_metadata_is_offline(monkeypatch):
    class API:
        def model_info(self, model_id, revision):
            raise OSError("offline")

    monkeypatch.setattr(huggingface_hub, "HfApi", API)
    commit = "ABCDEF0123456789ABCDEF0123456789ABCDEF01"
    assert benchmark.resolve_model_revision("Qwen/model", commit) == commit.lower()
    with pytest.raises(RuntimeError, match="could not resolve"):
        benchmark.resolve_model_revision("Qwen/model", "main")
