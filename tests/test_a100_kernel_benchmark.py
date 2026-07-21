"""CPU-only logic tests for the standalone A100 kernel benchmark."""

from __future__ import annotations

import gc
import importlib.util
import json
import sys
import weakref
from pathlib import Path
from types import MethodType, SimpleNamespace

import huggingface_hub
import peft
import pytest
import torch
import transformers

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_a100_kernels", ROOT / "scripts" / "benchmark_a100_kernels.py"
)
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def test_variant_plan_has_a_stable_reference_and_component_isolated_candidates():
    assert [variant.name for variant in benchmark.select_variants("all")] == [
        "native-sdpa",
        "native-flash-attn-2",
        "fused-linear-ce-sdpa",
    ]
    selected = benchmark.select_variants("fused-linear-ce-sdpa")
    assert [variant.name for variant in selected] == [
        "native-sdpa",
        "fused-linear-ce-sdpa",
    ]
    fused_loss = selected[-1]
    assert fused_loss.layer_backend == benchmark.NATIVE_LAYER_BACKEND
    assert fused_loss.loss_backend == "liger"
    assert (
        fused_loss.loss_implementation
        == benchmark.FUSED_LINEAR_CE_IMPLEMENTATION
    )
    assert not any("liger-flash" in variant.name for variant in benchmark.VARIANTS)
    with pytest.raises(ValueError, match="unknown variants"):
        benchmark.select_variants("unknown")


def test_fused_loss_variant_applies_instance_patch_before_peft(monkeypatch):
    events = []
    config = SimpleNamespace(model_type="qwen2", use_cache=True)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.weight = torch.nn.Parameter(torch.ones(1))

        def to(self, device):
            events.append(("to", device.type))
            return self

    model = Model()
    monkeypatch.setattr(
        benchmark, "validate_kernel_request", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(benchmark, "attention_load_kwargs", lambda *args: {})
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: events.append("config") or config,
    )
    monkeypatch.setattr(
        benchmark,
        "require_liger_model_support",
        lambda actual_config: events.append("support") or actual_config.model_type,
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: events.append("model") or model,
    )
    application = {
        "layer_backend": benchmark.NATIVE_LAYER_BACKEND,
        "loss_implementation": benchmark.FUSED_LINEAR_CE_IMPLEMENTATION,
    }
    monkeypatch.setattr(
        benchmark,
        "apply_liger_fused_linear_ce",
        lambda actual_model: events.append("apply") or application,
    )
    monkeypatch.setattr(benchmark, "resolve_lora_targets", lambda *args: "all-linear")
    monkeypatch.setattr(peft, "LoraConfig", lambda **kwargs: kwargs)

    def fake_get_peft_model(actual_model, _config):
        assert actual_model is model
        assert "apply" in events
        events.append("peft")
        return actual_model

    monkeypatch.setattr(peft, "get_peft_model", fake_get_peft_model)
    monkeypatch.setattr(
        benchmark,
        "validate_lora_production_envelope",
        lambda actual_model: {
            "output_head": {"frozen": True, "adapted": False},
            "trainable_dtype_counts": {"float32": 1},
        },
    )
    monkeypatch.setattr(
        benchmark,
        "resolved_attention_backend",
        lambda actual_model, requested: events.append("attention") or requested,
    )

    loaded, tuning, kernel_application = benchmark.load_raw_model(
        "Qwen/Qwen2.5-1.5B-Instruct",
        "a" * 40,
        benchmark.VARIANTS_BY_NAME["fused-linear-ce-sdpa"],
        torch.bfloat16,
        torch.device("cpu"),
        "lora",
        16,
        32,
        "auto",
        1234,
    )

    assert loaded is model
    assert tuning["mode"] == "lora"
    assert kernel_application == application
    assert events == [
        "config",
        "support",
        "model",
        "apply",
        "peft",
        ("to", "cpu"),
        "attention",
    ]


def test_every_kernel_isolation_failure_is_a_fatal_benchmark_load_error():
    complete = benchmark.KernelIsolationError(
        "restored",
        failed_invariants=["class_binding"],
        rollback_report={"complete": True},
    )
    poisoned = benchmark.KernelIsolationError(
        "poisoned",
        failed_invariants=["parameter_contents"],
        rollback_report={"complete": False},
    )
    signature_failure = benchmark.KernelIsolationError(
        "signature inspection failed after irreversible mutation",
        failed_invariants=["post_apply_validation_completed"],
        rollback_report={"complete": False},
    )

    assert benchmark.is_fatal_model_load_error(complete)
    assert benchmark.is_fatal_model_load_error(poisoned)
    assert benchmark.is_fatal_model_load_error(signature_failure)
    assert benchmark.is_fatal_model_load_error(RuntimeError("CUDA out of memory"))
    assert not benchmark.is_fatal_model_load_error(RuntimeError("missing optional package"))

    try:
        raise RuntimeError("wrapped") from complete
    except RuntimeError as wrapped:
        assert benchmark.is_fatal_model_load_error(wrapped)

    try:
        raise poisoned
    except benchmark.KernelIsolationError:
        try:
            raise RuntimeError("implicit context")
        except RuntimeError as contextual:
            assert benchmark.is_fatal_model_load_error(contextual)


def test_failed_model_load_unconditionally_collects_instance_forward_cycle(
    monkeypatch,
    tmp_path,
):
    references = []
    cleanup_observations = []

    class CyclicModel:
        pass

    def forward(self):
        return self

    def failing_load(*args, **kwargs):
        del args, kwargs
        model = CyclicModel()
        model.forward = MethodType(forward, model)
        references.append(weakref.ref(model))
        raise RuntimeError("load failed after instance patch")

    def observed_cleanup():
        gc.collect()
        cleanup_observations.append(references[-1]() is None)

    monkeypatch.setattr(
        benchmark,
        "setup_distributed",
        lambda: (0, 1, torch.device("cpu")),
    )
    monkeypatch.setattr(
        benchmark,
        "resolve_model_revision",
        lambda *args: "a" * 40,
    )
    monkeypatch.setattr(benchmark, "environment_report", lambda *args: {})
    monkeypatch.setattr(benchmark, "load_raw_model", failing_load)
    monkeypatch.setattr(benchmark, "cleanup_cuda", observed_cleanup)

    output = tmp_path / "failed.json"
    result = benchmark.main(
        [
            "--variants",
            "native-sdpa",
            "--output",
            str(output),
        ]
    )

    assert result == 1
    assert cleanup_observations == [True]
    assert references[0]() is None
    report = json.loads(output.read_text())
    assert report["supported_evidence_scope"]["fused-linear-ce-sdpa"][
        "peft_version"
    ] == benchmark.PEFT_VERSION


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
    assert not bad["passed"]
    assert bad["gradient_status"] == "failed"
    assert bad["first_failing_gradient_tensor"] == "layer.weight"

    missing = benchmark.compare_parity(
        12.0, {}, 12.0, reference, rtol=1e-5, atol=1e-6
    )
    assert not missing["passed"]
    assert missing["checked_gradient_tensors"] == 0
    assert "key mismatch" in missing["reason"]

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
    assert not bad_delta["passed"]
    assert bad_delta["parameter_delta_status"] == "failed"
    assert bad_delta["first_failing_parameter_delta_tensor"] == "layer.weight"


def test_parity_scans_all_gradients_and_deltas_after_early_failures():
    reference_gradients = {
        "first": torch.tensor([1.0, 2.0]),
        "worst": torch.tensor([3.0, 4.0]),
    }
    actual_gradients = {
        "first": torch.tensor([1.0, 2.1]),
        "worst": torch.tensor([30.0, 4.0]),
    }
    reference_deltas = {
        "first": torch.tensor([0.1]),
        "worst": torch.tensor([0.2]),
    }
    actual_deltas = {
        "first": torch.tensor([0.3]),
        "worst": torch.tensor([2.0]),
    }

    result = benchmark.compare_parity(
        1.0,
        actual_gradients,
        1.0,
        reference_gradients,
        rtol=0,
        atol=1e-6,
        parameter_deltas=actual_deltas,
        reference_parameter_deltas=reference_deltas,
    )

    assert not result["passed"]
    assert result["checked_gradient_tensors"] == 2
    assert result["checked_parameter_delta_tensors"] == 2
    assert result["first_failing_gradient_tensor"] == "first"
    assert result["worst_failing_gradient_tensor"] == "worst"
    assert result["first_failing_parameter_delta_tensor"] == "first"
    assert result["worst_failing_parameter_delta_tensor"] == "worst"
    assert result["max_gradient_abs_error"] == pytest.approx(27.0)
    assert result["max_parameter_delta_abs_error"] == pytest.approx(1.8)


def test_whole_model_metrics_are_scale_aware_and_count_violations():
    summary = benchmark.compare_tensor_maps(
        {
            "a": torch.tensor([3.0, 0.0]),
            "b": torch.tensor([0.0, 4.0]),
        },
        {
            "a": torch.tensor([0.0, 0.0]),
            "b": torch.tensor([0.0, 4.0]),
        },
        rtol=0,
        atol=0,
    )

    assert summary["status"] == "failed"
    assert summary["checked_tensors"] == 2
    assert summary["element_count"] == 4
    assert summary["allclose_violation_count"] == 1
    assert summary["allclose_violation_fraction"] == pytest.approx(0.25)
    assert summary["actual_l2_norm"] == pytest.approx(5.0)
    assert summary["reference_l2_norm"] == pytest.approx(4.0)
    assert summary["difference_l2_norm"] == pytest.approx(3.0)
    assert summary["relative_l2_error"] == pytest.approx(0.75)
    assert summary["cosine_similarity"] == pytest.approx(0.8)
    assert summary["max_actual_absolute"] == pytest.approx(4.0)
    assert summary["max_reference_absolute"] == pytest.approx(4.0)


def test_nonfinite_and_structural_failures_are_explicit():
    summary = benchmark.compare_tensor_maps(
        {
            "finite": torch.tensor([float("nan")]),
            "numeric": torch.tensor([2.0]),
            "shape": torch.ones(2),
            "extra": torch.ones(1),
        },
        {
            "finite": torch.ones(1),
            "numeric": torch.tensor([1.0]),
            "shape": torch.ones(3),
            "missing": torch.ones(1),
        },
        rtol=1e-5,
        atol=1e-6,
    )

    assert summary["status"] == "failed"
    assert summary["structural_status"] == "failed"
    assert summary["finiteness_status"] == "failed"
    assert summary["numeric_status"] == "partial"
    assert summary["numeric_scope"] == "compatible_finite_elements"
    assert summary["checked_tensors"] == 3
    assert summary["nonfinite_actual_elements"] == 1
    assert summary["numeric_element_count"] == 1
    assert summary["max_absolute_error"] == pytest.approx(1.0)
    assert summary["relative_l2_error"] == pytest.approx(1.0)
    assert summary["missing_tensors"] == ["missing"]
    assert summary["extra_tensors"] == ["extra"]
    assert summary["shape_mismatches"][0]["tensor"] == "shape"
    json.dumps(summary, allow_nan=False)


def test_nonfinite_loss_and_unevaluable_tensors_serialize_as_strict_json():
    result = benchmark.compare_parity(
        float("nan"),
        {"shape": torch.ones(2), "nonfinite": torch.tensor([float("inf")])},
        1.0,
        {"shape": torch.ones(3), "nonfinite": torch.ones(1)},
        rtol=1e-5,
        atol=1e-6,
    )

    assert result["loss_status"] == "nonfinite"
    assert result["loss"] is None
    assert result["loss_abs_error"] is None
    assert result["gradient_parity"]["max_absolute_error"] is None
    assert result["gradient_parity"]["relative_l2_error"] is None
    assert result["gradient_parity"]["nonfinite_actual_elements"] == 1
    json.dumps(result, allow_nan=False)


def test_nonzero_fractions_use_only_jointly_finite_elements():
    summary = benchmark.compare_tensor_maps(
        {"weight": torch.tensor([1.0, float("nan"), 2.0])},
        {"weight": torch.tensor([float("nan"), 3.0, 2.0])},
        rtol=0,
        atol=0,
    )

    assert summary["finite_element_count"] == 1
    assert summary["actual_nonzero_elements"] == 1
    assert summary["reference_nonzero_elements"] == 1
    assert summary["actual_nonzero_fraction"] == pytest.approx(1.0)
    assert summary["reference_nonzero_fraction"] == pytest.approx(1.0)
    assert 0 <= summary["actual_nonzero_fraction"] <= 1
    assert 0 <= summary["reference_nonzero_fraction"] <= 1


def test_unmeasured_and_insensitive_parameter_deltas_are_not_zero_errors():
    gradients = {"weight": torch.tensor([1.0])}
    not_evaluated = benchmark.compare_parity(
        1.0, gradients, 1.0, gradients, rtol=0, atol=0
    )
    assert not_evaluated["passed"]
    assert not_evaluated["parameter_delta_status"] == "not_evaluated"
    assert not_evaluated["max_parameter_delta_abs_error"] is None
    assert not_evaluated["checked_parameter_delta_tensors"] == 0

    rounded_away = benchmark.compare_parity(
        1.0,
        gradients,
        1.0,
        gradients,
        rtol=0,
        atol=0,
        parameter_deltas={"weight": torch.zeros(2)},
        reference_parameter_deltas={"weight": torch.zeros(2)},
    )
    assert not rounded_away["passed"]
    assert rounded_away["parameter_delta_status"] == "not_meaningful"
    assert rounded_away["max_parameter_delta_abs_error"] == 0
    assert (
        rounded_away["parameter_delta_actual_sensitivity"]["status"]
        == "rounded_away"
    )

    strict_delta = benchmark.compare_parity(
        1.0,
        gradients,
        1.0,
        gradients,
        rtol=0,
        atol=0.1,
        parameter_deltas={"weight": torch.tensor([2e-5])},
        reference_parameter_deltas={"weight": torch.tensor([1e-5])},
        parameter_delta_rtol=0,
        parameter_delta_atol=1e-8,
    )
    assert strict_delta["parameter_delta_status"] == "failed"


def test_bf16_adam_update_rounding_is_detected(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    model = torch.nn.Module()
    model.register_parameter(
        "weight", torch.nn.Parameter(torch.tensor([0.02], dtype=torch.bfloat16))
    )
    model.weight.grad = torch.ones_like(model.weight)
    optimizer = benchmark.make_optimizer(model, 1e-5, 0.01)

    deltas, _ = benchmark.parameter_delta_witness(
        model, optimizer, torch.device("cpu")
    )
    result = benchmark.compare_parity(
        1.0,
        {"weight": torch.ones(1)},
        1.0,
        {"weight": torch.ones(1)},
        rtol=0,
        atol=0,
        parameter_deltas=deltas,
        reference_parameter_deltas={"weight": torch.zeros(1)},
    )

    assert torch.count_nonzero(deltas["weight"]).item() == 0
    assert result["parameter_delta_status"] == "not_meaningful"
    assert not result["passed"]


def test_distributed_parity_aggregation_promotes_every_failing_rank():
    reference = {"weight": torch.tensor([1.0])}
    deltas = {"weight": torch.tensor([1e-5])}
    rank_zero = benchmark.compare_parity(
        1.0,
        reference,
        1.0,
        reference,
        rtol=0,
        atol=1e-6,
        parameter_deltas=deltas,
        reference_parameter_deltas=deltas,
        parameter_delta_rtol=0,
        parameter_delta_atol=1e-8,
    )
    rank_one = benchmark.compare_parity(
        1.0,
        {"weight": torch.tensor([10.0])},
        1.0,
        reference,
        rtol=0,
        atol=1e-6,
        parameter_deltas=deltas,
        reference_parameter_deltas=deltas,
        parameter_delta_rtol=0,
        parameter_delta_atol=1e-8,
    )
    rank_two = benchmark.compare_parity(
        float("nan"),
        reference,
        1.0,
        reference,
        rtol=0,
        atol=1e-6,
        parameter_deltas=deltas,
        reference_parameter_deltas=deltas,
        parameter_delta_rtol=0,
        parameter_delta_atol=1e-8,
    )
    diagnostics = [
        benchmark.compact_parity_diagnostic(rank_zero, 0),
        benchmark.compact_parity_diagnostic(rank_one, 1),
        benchmark.compact_parity_diagnostic(rank_two, 2),
    ]

    aggregate = benchmark.aggregate_parity_diagnostics(diagnostics)
    assert not aggregate["passed"]
    assert aggregate["failing_ranks"] == [1, 2]
    assert aggregate["worst_failing_rank"] == 2
    assert aggregate["gradient_max_absolute_error_max"] == pytest.approx(9.0)
    assert aggregate["gradient_nonfinite_actual_elements_total"] == 0
    assert aggregate["nonfinite_loss_ranks"] == [2]
    assert aggregate["loss_nonfinite_actual_ranks"] == [2]
    assert aggregate["loss_nonfinite_reference_ranks"] == []

    promoted = benchmark.apply_distributed_parity(rank_zero, diagnostics, rank=0)
    assert not promoted["passed"]
    assert promoted["reason"] == rank_two["reason"]
    assert promoted["max_gradient_abs_error"] == pytest.approx(9.0)
    assert promoted["loss_nonfinite_actual"] is True
    assert promoted["nonfinite_loss_ranks"] == [2]
    assert promoted["distributed_parity"]["failing_ranks"] == [1, 2]
    assert len(promoted["rank_diagnostics"]) == 3
    json.dumps(promoted, allow_nan=False)


def test_distributed_parity_aggregation_rejects_duplicate_ranks():
    parity = benchmark.compare_parity(
        1.0,
        {"weight": torch.ones(1)},
        1.0,
        {"weight": torch.ones(1)},
        rtol=0,
        atol=0,
    )
    diagnostic = benchmark.compact_parity_diagnostic(parity, 0)
    with pytest.raises(ValueError, match="duplicate"):
        benchmark.aggregate_parity_diagnostics([diagnostic, diagnostic])


def test_state_digest_aggregation_identifies_remote_mismatch():
    reference = "a" * 64
    different = "b" * 64
    report = benchmark.aggregate_state_digest_diagnostics(
        [
            {"rank": 0, "digest": reference},
            {"rank": 3, "digest": different},
        ]
    )

    assert report["reference_digest"] == reference
    assert report["passed"] is False
    assert report["failing_ranks"] == [3]
    assert report["unique_digest_count"] == 2
    assert report["rank_digests"] == [
        {"rank": 0, "digest": reference, "matches_reference": True},
        {"rank": 3, "digest": different, "matches_reference": False},
    ]
    json.dumps(report, allow_nan=False)


def test_benchmark_argument_validation_is_cpu_safe():
    args = benchmark.build_parser().parse_args([])
    benchmark.validate_args(args)
    assert args.dtype == "bf16"
    assert args.model == "Qwen/Qwen2.5-1.5B-Instruct"
    assert args.tuning == "lora"
    assert args.lora_r == 16
    assert args.lora_alpha == 32
    assert args.lora_targets == "auto"
    assert args.learning_rate == pytest.approx(3e-4)
    assert args.parameter_delta_atol == 1e-8
    assert args.trial_index == 1
    assert args.steps > 0 and args.warmup_steps > 0

    broken = SimpleNamespace(**vars(args))
    broken.target_fraction = 0
    with pytest.raises(ValueError, match="target-fraction"):
        benchmark.validate_args(broken)

    broken = SimpleNamespace(**vars(args))
    broken.lora_r = 0
    with pytest.raises(ValueError, match="lora-r"):
        benchmark.validate_args(broken)

    broken = SimpleNamespace(**vars(args))
    broken.tuning = "full"
    with pytest.raises(ValueError, match="approved only for --tuning lora"):
        benchmark.validate_args(broken)

    native_full = SimpleNamespace(**vars(args))
    native_full.tuning = "full"
    native_full.variants = "native-sdpa"
    benchmark.validate_args(native_full)


def test_trainable_state_digest_covers_values_names_and_dtypes():
    first = torch.nn.Module()
    first.register_parameter("weight", torch.nn.Parameter(torch.tensor([1.0])))
    first.register_parameter(
        "frozen", torch.nn.Parameter(torch.tensor([2.0]), requires_grad=False)
    )
    second = torch.nn.Module()
    second.register_parameter("weight", torch.nn.Parameter(torch.tensor([1.0])))

    digest = benchmark.trainable_state_digest(first)
    assert digest == benchmark.trainable_state_digest(second)
    second.weight.data.add_(1)
    assert digest != benchmark.trainable_state_digest(second)

    second.weight = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    assert digest != benchmark.trainable_state_digest(second)


def test_trainable_anchor_restore_and_lora_factor_validation():
    model = torch.nn.Module()
    model.lora_A = torch.nn.Linear(2, 2, bias=False)
    model.lora_B = torch.nn.Linear(2, 2, bias=False)
    model.lora_B.weight.data.zero_()
    with pytest.raises(RuntimeError, match="lora_B"):
        benchmark.lora_factor_nonzero_report(
            benchmark.snapshot_trainable_state(model)
        )

    model.lora_B.weight.data.fill_(0.25)
    anchor = benchmark.snapshot_trainable_state(model)
    anchor_digest = benchmark.trainable_state_digest(model)
    report = benchmark.lora_factor_nonzero_report(anchor)
    assert report["lora_A"]["nonzero_elements"] > 0
    assert report["lora_B"]["nonzero_elements"] == 4

    model.lora_A.weight.data.zero_()
    benchmark.restore_trainable_state(model, anchor)
    assert benchmark.trainable_state_digest(model) == anchor_digest


def test_exact_global_target_backward_clips_trainable_gradients():
    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.tensor([1.0])))
    report = benchmark.backward_and_clip(
        model,
        model.weight.sum() * 10.0,
        torch.tensor(2),
        tuning="lora",
        world=1,
        device=torch.device("cpu"),
    )

    assert report["global_target_tokens"] == 2
    assert report["pre_clip_grad_norm"] == pytest.approx(5.0)
    assert model.weight.grad.item() == pytest.approx(1.0)
    optimizer = benchmark.make_optimizer(model, 3e-4, 0.01)
    assert optimizer.defaults["lr"] == pytest.approx(3e-4)
    assert optimizer.defaults["foreach"] is None
    assert optimizer.defaults["fused"] is None


def test_lora_output_head_must_be_frozen_and_unadapted():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(2, 2, bias=False)
            self.adapter = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))

        def get_output_embeddings(self):
            return self.lm_head

    model = Model()
    model.lm_head.requires_grad_(False)
    assert benchmark.validate_lora_production_envelope(model) == {
        "output_head": {
            "frozen": True,
            "adapted": False,
            "parameter_count": 4,
        },
        "trainable_dtype_counts": {"float32": 2},
    }
    model.lm_head.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="frozen, unadapted"):
        benchmark.validate_lora_production_envelope(model)

    model.lm_head.weight.requires_grad_(False)
    model.adapter = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    with pytest.raises(RuntimeError, match="FP32 trainable adapters"):
        benchmark.validate_lora_production_envelope(model)


def test_source_provenance_environment_overrides_are_strict(monkeypatch):
    sha = "a" * 40
    monkeypatch.setenv("YETO_GIT_SHA", sha.upper())
    monkeypatch.setenv("YETO_GIT_DIRTY", "true")
    provenance = benchmark.source_provenance()
    assert provenance["git_sha"] == sha
    assert provenance["git_dirty"] is True
    assert provenance["clean_commit_exact"] is False
    assert provenance["provenance_source"] == "environment_override"
    assert len(provenance["benchmark_script_sha256"]) == 64

    monkeypatch.setenv("YETO_GIT_SHA", "not-a-sha")
    with pytest.raises(ValueError, match="YETO_GIT_SHA"):
        benchmark.source_provenance()
    monkeypatch.setenv("YETO_GIT_SHA", sha)
    monkeypatch.setenv("YETO_GIT_DIRTY", "maybe")
    with pytest.raises(ValueError, match="YETO_GIT_DIRTY"):
        benchmark.source_provenance()


def test_source_provenance_requires_complete_metadata(monkeypatch):
    monkeypatch.delenv("YETO_GIT_SHA", raising=False)
    monkeypatch.delenv("YETO_GIT_DIRTY", raising=False)
    monkeypatch.setattr(benchmark, "_git_output", lambda _arguments: None)
    with pytest.raises(RuntimeError, match="source provenance is unavailable"):
        benchmark.source_provenance()

    monkeypatch.setenv("YETO_GIT_SHA", "b" * 40)
    with pytest.raises(ValueError, match="supplied together"):
        benchmark.source_provenance()


def test_source_provenance_git_fallback_marks_clean_commit(monkeypatch):
    monkeypatch.delenv("YETO_GIT_SHA", raising=False)
    monkeypatch.delenv("YETO_GIT_DIRTY", raising=False)

    def git_output(arguments):
        return "c" * 40 if arguments[0] == "rev-parse" else ""

    monkeypatch.setattr(benchmark, "_git_output", git_output)
    provenance = benchmark.source_provenance()
    assert provenance["git_sha"] == "c" * 40
    assert provenance["git_dirty"] is False
    assert provenance["clean_commit_exact"] is True
    assert provenance["provenance_source"] == "git_worktree"


@pytest.mark.parametrize(
    ("variants", "failed", "expected_status"),
    [
        (["a", "b"], False, "passed"),
        (["a"], False, "incomplete"),
        (["a"], True, "failed"),
    ],
)
def test_report_status_distinguishes_pass_failure_and_incomplete(
    variants, failed, expected_status
):
    report = {
        "status": "incomplete",
        "planned_variants": ["a", "b"],
        "completed_variants": [],
        "fatal": None,
        "variants": [
            {"variant": {"name": name}, "status": "passed"} for name in variants
        ],
    }
    benchmark.finalize_report_status(
        report,
        failed=failed,
        fatal_phase="parity" if failed else None,
        fatal_reason="mismatch" if failed else None,
    )
    assert report["status"] == expected_status
    assert report["completed_variants"] == variants
    if failed:
        assert report["fatal"] == {"phase": "parity", "reason": "mismatch"}
    else:
        assert report["fatal"] is None

    if expected_status == "passed":
        report["variants"][1]["status"] = "skipped"
        benchmark.finalize_report_status(report, False, None, None)
        assert report["status"] == "incomplete"


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
