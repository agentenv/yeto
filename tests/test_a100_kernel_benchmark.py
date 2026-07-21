"""CPU-only logic tests for the standalone A100 kernel benchmark."""

from __future__ import annotations

import importlib.util
import json
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


def test_variant_plan_has_a_stable_reference_and_one_candidate_contract():
    assert [variant.name for variant in benchmark.VARIANTS] == [
        "native-sdpa",
        "native-sdpa-flash",
        "native-sdpa-math",
        "native-sdpa-efficient",
        "native-sdpa-cudnn",
        "native-flash-attn-2",
        "liger-sdpa",
        "liger-flash-attn-2",
    ]
    assert benchmark.select_variants("") == [benchmark.REFERENCE_VARIANT]
    assert benchmark.REFERENCE_VARIANT.internal_sdpa_backend == "auto"
    assert {
        variant.internal_sdpa_backend
        for variant in benchmark.VARIANTS
        if variant.attention_backend == "sdpa"
    } == {"auto", "flash", "math", "efficient", "cudnn"}
    selected = benchmark.select_variants("liger-sdpa")
    assert [variant.name for variant in selected] == ["native-sdpa", "liger-sdpa"]
    selected = benchmark.select_variants("native-sdpa-math,native-sdpa")
    assert [variant.name for variant in selected] == [
        "native-sdpa",
        "native-sdpa-math",
    ]
    forced_reference = benchmark.select_variants("", reference_name="native-sdpa-cudnn")
    assert [variant.name for variant in forced_reference] == ["native-sdpa-cudnn"]
    forced_with_candidate = benchmark.select_variants(
        "native-sdpa-math", reference_name="native-sdpa-cudnn"
    )
    assert [variant.name for variant in forced_with_candidate] == [
        "native-sdpa-cudnn",
        "native-sdpa-math",
    ]
    with pytest.raises(ValueError, match="at most one candidate"):
        benchmark.select_variants("all")
    with pytest.raises(ValueError, match="at most one candidate"):
        benchmark.select_variants("native-sdpa-math,native-sdpa-flash")
    with pytest.raises(ValueError, match="unknown variants"):
        benchmark.select_variants("unknown")
    with pytest.raises(ValueError, match="unknown reference variant"):
        benchmark.select_variants("", reference_name="unknown")


def test_public_sdpa_backend_mapping_and_context_restore_exact_flags(monkeypatch):
    from torch.nn.attention import SDPBackend

    assert benchmark.sdpa_backend_objects("flash") == [SDPBackend.FLASH_ATTENTION]
    assert benchmark.sdpa_backend_objects("math") == [SDPBackend.MATH]
    assert benchmark.sdpa_backend_objects("efficient") == [
        SDPBackend.EFFICIENT_ATTENTION
    ]
    assert benchmark.sdpa_backend_objects("cudnn") == [SDPBackend.CUDNN_ATTENTION]
    assert set(benchmark.sdpa_backend_objects("auto")) == {
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.MATH,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.CUDNN_ATTENTION,
    }
    with pytest.raises(ValueError, match="unknown internal SDPA backend"):
        benchmark.sdpa_backend_objects("unknown")

    before = benchmark.snapshot_sdpa_backend_flags()
    with benchmark.sdpa_backend_context("math") as report:
        assert benchmark.snapshot_sdpa_backend_flags() == {
            "flash": False,
            "math": True,
            "efficient": False,
            "cudnn": False,
        }
        assert report["active"] == report["expected_active"]
    assert benchmark.snapshot_sdpa_backend_flags() == before
    assert report["after"] == before
    assert report["restored_exactly"] is True

    controller = benchmark.SDPAArmController()
    math_variant = benchmark.VARIANTS_BY_NAME["native-sdpa-math"]
    flash_variant = benchmark.VARIANTS_BY_NAME["native-sdpa-flash"]
    math_control = controller.activate(math_variant)
    assert benchmark.snapshot_sdpa_backend_flags() == math_control["expected_active"]
    with pytest.raises(RuntimeError, match="already active"):
        controller.activate(flash_variant)
    controller.close()
    assert math_control["restored_exactly"] is True
    flash_control = controller.activate(flash_variant)
    assert benchmark.snapshot_sdpa_backend_flags() == flash_control["expected_active"]
    controller.close()
    assert flash_control["restored_exactly"] is True
    assert benchmark.snapshot_sdpa_backend_flags() == before

    snapshots = iter(
        [
            {"flash": True, "math": True, "efficient": True, "cudnn": True},
            {"flash": False, "math": True, "efficient": False, "cudnn": False},
            {"flash": False, "math": True, "efficient": False, "cudnn": False},
            {"flash": True, "math": True, "efficient": True, "cudnn": True},
        ]
    )
    repairs = []
    monkeypatch.setattr(
        benchmark, "snapshot_sdpa_backend_flags", lambda: next(snapshots)
    )
    monkeypatch.setattr(
        benchmark, "restore_sdpa_backend_flags", lambda flags: repairs.append(flags)
    )
    monkeypatch.setattr(
        torch.nn.attention,
        "sdpa_kernel",
        lambda _backends: benchmark.contextlib.nullcontext(),
    )
    with pytest.raises(RuntimeError, match="did not restore"):
        with benchmark.sdpa_backend_context("math"):
            pass
    assert repairs == [{"flash": True, "math": True, "efficient": True, "cudnn": True}]


def test_sdpa_arm_lifecycle_requires_explicit_all_rank_restore():
    before = benchmark.snapshot_sdpa_backend_flags()
    controller = benchmark.SDPAArmController()
    variant = benchmark.VARIANTS_BY_NAME["native-sdpa-math"]

    lifecycle = benchmark.begin_sdpa_arm(controller, variant, rank=0, world=1)
    assert lifecycle["activation"]["passed"] is True
    assert lifecycle["restoration"] is None
    assert controller.active is True

    restoration = benchmark.finish_sdpa_arm(controller, lifecycle, rank=0, world=1)
    assert restoration["passed"] is True
    assert lifecycle["passed"] is True
    assert controller.active is False
    assert benchmark.snapshot_sdpa_backend_flags() == before

    audit = benchmark.final_sdpa_controller_audit(controller, rank=0, world=1)
    assert audit["passed"] is True
    json.dumps(lifecycle, allow_nan=False)
    json.dumps(audit, allow_nan=False)


def test_sdpa_control_aggregation_promotes_remote_failures():
    expected = {"flash": False, "math": True, "efficient": False, "cudnn": False}
    control = {
        "requested": "math",
        "before": {name: True for name in benchmark.SDPA_FLAG_GETTERS},
        "expected_active": expected,
        "active": expected,
        "after": None,
        "restored_exactly": False,
    }
    activation = benchmark.aggregate_sdpa_control_phase(
        [
            {
                "rank": 0,
                "phase": "activation",
                "variant": "native-sdpa-math",
                "active_after_phase": True,
                "error": None,
                "control": dict(control),
            },
            {
                "rank": 1,
                "phase": "activation",
                "variant": "native-sdpa-math",
                "active_after_phase": False,
                "error": "RuntimeError: selector mismatch",
                "control": dict(control),
            },
        ],
        expected_world=2,
        variant_name="native-sdpa-math",
        phase="activation",
    )
    assert activation["passed"] is False
    assert activation["failing_ranks"] == [1]


def test_mid_arm_selector_mutation_invalidates_the_arm():
    before = benchmark.snapshot_sdpa_backend_flags()
    controller = benchmark.SDPAArmController()
    variant = benchmark.VARIANTS_BY_NAME["native-sdpa-math"]
    lifecycle = benchmark.begin_sdpa_arm(controller, variant, rank=0, world=1)
    torch.backends.cuda.enable_flash_sdp(True)
    record = {
        "variant": {"name": variant.name},
        "status": "passed",
        "metrics": {"mean_step_seconds": 1.0},
    }
    reason = benchmark.restore_sdpa_arm_for_record(
        record, controller, lifecycle, rank=0, world=1
    )
    assert reason is not None
    assert record["status"] == "failed"
    assert "metrics" not in record
    assert (
        lifecycle["restoration"]["rank_reports"][0]["flags_before_close"]["flash"]
        is True
    )
    assert benchmark.snapshot_sdpa_backend_flags() == before


def test_failed_sdpa_activation_is_cleaned_up_before_arm_work():
    class ActivationFailureController:
        def __init__(self):
            self.active = False
            self.control = benchmark._empty_sdpa_control("math")
            self.close_calls = 0

        def activate(self, _variant):
            raise RuntimeError("activation failed")

        def close(self, _exc_info=(None, None, None)):
            self.close_calls += 1
            return self.control

    controller = ActivationFailureController()
    variant = benchmark.VARIANTS_BY_NAME["native-sdpa-math"]
    lifecycle = benchmark.begin_sdpa_arm(controller, variant, rank=0, world=1)
    assert lifecycle["activation"]["passed"] is False
    assert lifecycle["activation"]["failing_ranks"] == [0]
    assert lifecycle["restoration"]["passed"] is True
    assert lifecycle["passed"] is False
    assert controller.close_calls == 1
    assert controller.active is False


def test_failed_selector_restore_invalidates_metrics_and_final_audit():
    class FailingController:
        def __init__(self):
            self.active = True
            self.control = {
                "requested": "math",
                "before": {name: True for name in benchmark.SDPA_FLAG_GETTERS},
                "expected_active": {
                    "flash": False,
                    "math": True,
                    "efficient": False,
                    "cudnn": False,
                },
                "active": {
                    "flash": False,
                    "math": True,
                    "efficient": False,
                    "cudnn": False,
                },
                "after": {name: True for name in benchmark.SDPA_FLAG_GETTERS},
                "restored_exactly": False,
            }

        def close(self, _exc_info=(None, None, None)):
            self.active = False
            raise RuntimeError("restore failed")

    controller = FailingController()
    lifecycle = {
        "variant": "native-sdpa-math",
        "activation": {"passed": True},
        "restoration": None,
        "passed": False,
    }
    record = {
        "variant": {"name": "native-sdpa-math"},
        "status": "passed",
        "metrics": {"mean_step_seconds": 1.0},
    }
    reason = benchmark.restore_sdpa_arm_for_record(
        record, controller, lifecycle, rank=0, world=1
    )
    assert reason is not None
    assert record["status"] == "failed"
    assert "metrics" not in record
    assert lifecycle["restoration"]["failing_ranks"] == [0]

    active_controller = benchmark.SDPAArmController()
    variant = benchmark.VARIANTS_BY_NAME["native-sdpa-math"]
    active_controller.activate(variant)
    final = benchmark.final_sdpa_controller_audit(active_controller, rank=0, world=1)
    assert final["passed"] is False
    assert final["failing_ranks"] == [0]
    assert active_controller.active is False

    report = {
        "status": "incomplete",
        "planned_variants": ["native-sdpa-math"],
        "completed_variants": [],
        "fatal": None,
        "variants": [record],
        "sdpa_selector_finalization": final,
    }
    benchmark.finalize_report_status(
        report,
        failed=True,
        fatal_phase="sdpa_selector_restoration",
        fatal_reason=reason,
    )
    serialized = json.dumps(report, allow_nan=False)
    assert json.loads(serialized)["status"] == "failed"
    assert json.loads(serialized)["variants"][0]["status"] == "failed"


def test_rank_one_restore_failure_invalidates_rank_zero_record_and_json(monkeypatch):
    expected = {"flash": False, "math": True, "efficient": False, "cudnn": False}
    before = {name: True for name in benchmark.SDPA_FLAG_GETTERS}

    class Controller:
        active = True
        control = {
            "api": "torch.nn.attention.sdpa_kernel",
            "requested": "math",
            "before": before,
            "expected_active": expected,
            "active": expected,
            "after": before,
            "restored_exactly": True,
        }

        def close(self, _exc_info=(None, None, None)):
            self.active = False
            return self.control

    def all_gather_object(output, local):
        output[0] = local
        remote = json.loads(json.dumps(local))
        remote["rank"] = 1
        remote["error"] = "RuntimeError: rank-one restore failed"
        remote["control"]["restored_exactly"] = False
        output[1] = remote

    monkeypatch.setattr(benchmark.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(benchmark.dist, "all_gather_object", all_gather_object)
    monkeypatch.setattr(benchmark, "snapshot_sdpa_backend_flags", lambda: expected)
    controller = Controller()
    lifecycle = {
        "variant": "native-sdpa-math",
        "requested_internal_backend": "math",
        "activation": {"passed": True},
        "restoration": None,
        "passed": False,
    }
    record = {
        "variant": {"name": "native-sdpa-math"},
        "status": "passed",
        "metrics": {"mean_step_seconds": 1.0},
    }
    reason = benchmark.restore_sdpa_arm_for_record(
        record, controller, lifecycle, rank=0, world=2
    )

    assert lifecycle["restoration"]["failing_ranks"] == [1]
    assert record["status"] == "failed"
    assert "metrics" not in record
    report = {
        "status": "incomplete",
        "planned_variants": ["native-sdpa-math"],
        "completed_variants": [],
        "fatal": None,
        "variants": [record],
    }
    benchmark.finalize_report_status(report, True, "sdpa_selector_restoration", reason)
    serialized = json.dumps(report, allow_nan=False)
    assert json.loads(serialized)["status"] == "failed"
    assert json.loads(serialized)["variants"][0]["status"] == "failed"


def test_arm_transitions_are_activation_then_explicit_restoration(monkeypatch):
    phases = []
    original_gather = benchmark.gather_rank_records

    def gather(local, rank, world):
        phases.append((local["variant"], local["phase"]))
        return original_gather(local, rank, world)

    monkeypatch.setattr(benchmark, "gather_rank_records", gather)
    controller = benchmark.SDPAArmController()
    math_variant = benchmark.VARIANTS_BY_NAME["native-sdpa-math"]
    flash_variant = benchmark.VARIANTS_BY_NAME["native-sdpa-flash"]
    math = benchmark.begin_sdpa_arm(controller, math_variant, rank=0, world=1)
    benchmark.finish_sdpa_arm(controller, math, rank=0, world=1)
    flash = benchmark.begin_sdpa_arm(controller, flash_variant, rank=0, world=1)
    benchmark.finish_sdpa_arm(controller, flash, rank=0, world=1)

    assert phases == [
        (math_variant.name, "activation"),
        (math_variant.name, "restoration"),
        (flash_variant.name, "activation"),
        (flash_variant.name, "restoration"),
    ]
    assert math["passed"] is True and flash["passed"] is True
    assert controller.active is False


def test_sdpa_input_recorder_restores_callable_and_aggregates_full_signature():
    functional = torch.nn.functional
    original = functional.scaled_dot_product_attention
    query = torch.randn(2, 3, 4, 8, requires_grad=True)
    recorder = benchmark.SDPAInputRecorder()
    with recorder:
        functional.scaled_dot_product_attention(
            query,
            query,
            query,
            dropout_p=0.0,
            is_causal=True,
            scale=0.25,
        ).sum().backward()

    assert functional.scaled_dot_product_attention is original
    report = recorder.report()
    assert report["total_calls"] == 1
    assert report["unique_signature_count"] == 1
    signature = report["unique_signatures"][0]["signature"]
    assert signature["query"]["shape"] == [2, 3, 4, 8]
    assert signature["query"]["stride"] == list(query.stride())
    assert signature["query"]["dtype"] == "float32"
    assert signature["query"]["device_type"] == "cpu"
    assert signature["attn_mask"] is None
    assert signature["dropout_p"] == 0.0
    assert signature["is_causal"] is True
    assert signature["scale"] == pytest.approx(0.25)
    assert signature["enable_gqa"] is False
    assert signature["selector_eligibility"]["math"] is True
    json.dumps(report, allow_nan=False)


def test_attribution_probe_is_forward_only_rank_local_and_restores_state(monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.trainable = torch.nn.Parameter(torch.tensor([1.0]))
            self.frozen = torch.nn.Parameter(torch.tensor([3.0]), requires_grad=False)
            self.register_buffer("cache", torch.tensor([2.0]))

    class Wrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

    class Recorder:
        def __enter__(self):
            return self

        def __exit__(self, *_exc_info):
            return False

        def report(self):
            return {
                "total_calls": 1,
                "unique_signature_count": 1,
                "ordered_signature_sha256": "a" * 64,
                "unique_signatures": [
                    {
                        "sha256": "b" * 64,
                        "first_call_index": 0,
                        "call_count": 1,
                        "signature": {
                            "query": None,
                            "key": None,
                            "value": None,
                            "attn_mask": None,
                            "selector_eligibility": {
                                "flash": False,
                                "math": True,
                                "efficient": False,
                                "cudnn": False,
                            },
                        },
                    }
                ],
            }

    class Profiler:
        def __enter__(self):
            return self

        def __exit__(self, *_exc_info):
            return False

        def key_averages(self):
            return [
                {"key": "aten::scaled_dot_product_attention", "count": 1},
                {"key": "aten::_scaled_dot_product_attention_math", "count": 1},
            ]

    base = Model()
    wrapped = Wrapper(base)
    anchor = benchmark.snapshot_trainable_state(wrapped)
    anchor_digest = benchmark.trainable_state_digest(wrapped)
    cpu_rng_before = torch.random.get_rng_state().clone()
    cuda_rng = {"state": torch.tensor([17], dtype=torch.uint8)}
    events = []

    def set_cuda_seed(seed):
        cuda_rng["state"] = torch.tensor([seed % 251], dtype=torch.uint8)

    def set_cuda_state(state, _device):
        cuda_rng["state"] = state.clone()

    def forward_sum(model, _variant, _input_ids, _weights):
        events.append("forward")
        assert model is base
        assert torch.is_grad_enabled()
        with torch.no_grad():
            model.cache.add_(5)
        torch.rand(1)
        cuda_rng["state"].add_(1)
        return model.trainable.sum(), torch.tensor(1)

    original_gather = benchmark.gather_sdpa_attribution

    def gather(*args, **kwargs):
        events.append("gather")
        return original_gather(*args, **kwargs)

    monkeypatch.setattr(benchmark, "SDPAInputRecorder", Recorder)
    monkeypatch.setattr(torch.profiler, "profile", lambda **_kwargs: Profiler())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(
        torch.cuda, "get_rng_state", lambda _device: cuda_rng["state"].clone()
    )
    monkeypatch.setattr(torch.cuda, "set_rng_state", set_cuda_state)
    monkeypatch.setattr(torch.cuda, "manual_seed", set_cuda_seed)
    monkeypatch.setattr(benchmark, "forward_sum", forward_sum)
    monkeypatch.setattr(benchmark, "gather_sdpa_attribution", gather)
    monkeypatch.setattr(
        benchmark,
        "training_step",
        lambda *_args, **_kwargs: pytest.fail("attribution called training_step"),
    )
    monkeypatch.setattr(
        benchmark,
        "make_optimizer",
        lambda *_args, **_kwargs: pytest.fail("attribution created an optimizer"),
    )

    result = benchmark.profile_sdpa_attribution_probe(
        wrapped,
        benchmark.VARIANTS_BY_NAME["native-sdpa-math"],
        torch.ones((1, 2), dtype=torch.long),
        torch.ones((1, 2)),
        rank=0,
        world=1,
        device=torch.device("cpu"),
        anchor_state=anchor,
        anchor_digest=anchor_digest,
        seed=123,
        shape_name="parity",
    )

    assert result["passed"] is True
    assert events == ["forward", "gather"]
    assert base.cache.item() == pytest.approx(2.0)
    assert torch.equal(torch.random.get_rng_state(), cpu_rng_before)
    assert torch.equal(cuda_rng["state"], torch.tensor([17], dtype=torch.uint8))

    events.clear()
    frozen_before = base.frozen.detach().clone()

    def mutate_frozen(model, _variant, _input_ids, _weights):
        events.append("forward")
        with torch.no_grad():
            model.frozen.add_(1)
        return model.trainable.sum(), torch.tensor(1)

    monkeypatch.setattr(benchmark, "forward_sum", mutate_frozen)
    mutated = benchmark.profile_sdpa_attribution_probe(
        wrapped,
        benchmark.VARIANTS_BY_NAME["native-sdpa-math"],
        torch.ones((1, 2), dtype=torch.long),
        torch.ones((1, 2)),
        rank=0,
        world=1,
        device=torch.device("cpu"),
        anchor_state=anchor,
        anchor_digest=anchor_digest,
        seed=789,
        shape_name="timing",
    )
    assert mutated["passed"] is False
    assert events == ["forward", "gather"]
    assert any(
        "mutated a frozen parameter" in error
        for error in mutated["rank_reports"][0]["errors"]
    )
    with torch.no_grad():
        base.frozen.copy_(frozen_before)
    state = result["rank_reports"][0]["relevant_state_restore"]
    assert state["named_buffers_mutated_during_probe"] is True
    assert state["named_buffers_restored_exactly"] is True
    assert state["frozen_parameters_unchanged"] is True
    assert state["rng_restored_exactly"] is True
    json.dumps(result, allow_nan=False)

    events.clear()

    def failing_forward(*_args, **_kwargs):
        events.append("forward")
        raise RuntimeError("rank-local probe failure")

    monkeypatch.setattr(benchmark, "forward_sum", failing_forward)
    failed = benchmark.profile_sdpa_attribution_probe(
        wrapped,
        benchmark.VARIANTS_BY_NAME["native-sdpa-math"],
        torch.ones((1, 2), dtype=torch.long),
        torch.ones((1, 2)),
        rank=0,
        world=1,
        device=torch.device("cpu"),
        anchor_state=anchor,
        anchor_digest=anchor_digest,
        seed=456,
        shape_name="timing",
    )
    assert failed["passed"] is False
    assert events == ["forward", "gather"]
    assert any(
        "rank-local probe failure" in error
        for error in failed["rank_reports"][0]["errors"]
    )
    assert base.cache.item() == pytest.approx(2.0)
    assert torch.equal(torch.random.get_rng_state(), cpu_rng_before)
    assert torch.equal(cuda_rng["state"], torch.tensor([17], dtype=torch.uint8))


def test_rank_one_profiler_failure_reaches_exactly_one_common_gather(monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.trainable = torch.nn.Parameter(torch.tensor([1.0]))
            self.frozen = torch.nn.Parameter(torch.tensor([2.0]), requires_grad=False)
            self.register_buffer("cache", torch.tensor([3.0]))

    class Recorder:
        def __enter__(self):
            return self

        def __exit__(self, *_exc_info):
            return False

        def report(self):
            events.append("report")
            raise RuntimeError("rank-local recorder failure")

    class Profiler:
        def __enter__(self):
            return self

        def __exit__(self, *_exc_info):
            return False

        def key_averages(self):
            return []

    model = Model()
    anchor = benchmark.snapshot_trainable_state(model)
    anchor_digest = benchmark.trainable_state_digest(model)
    cuda_rng = {"state": torch.tensor([31], dtype=torch.uint8)}
    events = []

    def failing_forward(*_args, **_kwargs):
        events.append("forward")
        raise RuntimeError("rank-one forward failure")

    def gather(local, rank, world, selector_backend, shape_name):
        events.append("gather")
        assert (rank, world, selector_backend, shape_name) == (1, 2, "math", "timing")
        assert local["passed"] is False
        assert any("rank-one forward failure" in error for error in local["errors"])
        assert any("rank-local recorder failure" in error for error in local["errors"])
        return {"passed": False, "rank_reports": [local]}

    monkeypatch.setattr(benchmark, "SDPAInputRecorder", Recorder)
    monkeypatch.setattr(torch.profiler, "profile", lambda **_kwargs: Profiler())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(
        torch.cuda, "get_rng_state", lambda _device: cuda_rng["state"].clone()
    )
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        lambda state, _device: cuda_rng.__setitem__("state", state.clone()),
    )
    monkeypatch.setattr(
        torch.cuda,
        "manual_seed",
        lambda seed: cuda_rng.__setitem__(
            "state", torch.tensor([seed % 251], dtype=torch.uint8)
        ),
    )
    monkeypatch.setattr(benchmark, "forward_sum", failing_forward)
    monkeypatch.setattr(benchmark, "gather_sdpa_attribution", gather)

    result = benchmark.profile_sdpa_attribution_probe(
        model,
        benchmark.VARIANTS_BY_NAME["native-sdpa-math"],
        torch.ones((1, 2), dtype=torch.long),
        torch.ones((1, 2)),
        rank=1,
        world=2,
        device=torch.device("cpu"),
        anchor_state=anchor,
        anchor_digest=anchor_digest,
        seed=900,
        shape_name="timing",
    )

    assert result["passed"] is False
    assert events == ["forward", "report", "gather"]
    assert cuda_rng["state"].item() == 31


def test_profiler_parser_and_selector_operator_agreement_are_fail_closed():
    events = [
        SimpleNamespace(key="aten::scaled_dot_product_attention", count=3),
        SimpleNamespace(key="aten::_scaled_dot_product_flash_attention", count=3),
        SimpleNamespace(key="aten::unrelated", count=99),
    ]
    parsed = benchmark.parse_sdpa_profiler_events(events)
    assert parsed["generic_sdpa_call_count"] == 3
    assert parsed["primary_backend_call_count"] == 3
    assert parsed["observed_backends"] == ["flash"]
    assert parsed["backend_call_counts"]["flash"] == 3
    assert "aten::unrelated" not in parsed["operator_counts"]
    assert parsed["unexpected_aten_attention_operator_counts"] == {}

    signature = {
        "selector_eligibility": {
            "flash": True,
            "math": True,
            "efficient": False,
            "cudnn": False,
        }
    }
    recorder = {
        "total_calls": 3,
        "unique_signature_count": 1,
        "ordered_signature_sha256": "a" * 64,
        "unique_signatures": [
            {
                "sha256": "b" * 64,
                "first_call_index": 0,
                "call_count": 3,
                "signature": signature,
            }
        ],
    }
    passed = benchmark.evaluate_local_sdpa_attribution("flash", recorder, parsed)
    assert passed["passed"] is True
    assert passed["selector_operator_agreement"] is True

    for unexpected_operator in (
        "aten::_scaled_dot_product_flash_attention_backward",
        "aten::_scaled_dot_product_future_attention",
        "aten::unrelated_attention_extension",
    ):
        unexpected = benchmark.parse_sdpa_profiler_events(
            [
                {"key": "aten::scaled_dot_product_attention", "count": 3},
                {"key": "aten::_scaled_dot_product_flash_attention", "count": 3},
                {"key": unexpected_operator, "count": 1},
            ]
        )
        rejected = benchmark.evaluate_local_sdpa_attribution(
            "auto", recorder, unexpected
        )
        assert rejected["passed"] is False
        assert (
            unexpected_operator
            in unexpected["unexpected_aten_attention_operator_counts"]
        )
        assert any("unexpected ATen" in error for error in rejected["errors"])

    wrong = benchmark.evaluate_local_sdpa_attribution("math", recorder, parsed)
    assert wrong["passed"] is False
    assert wrong["selector_operator_agreement"] is False
    assert any("disagreed" in error for error in wrong["errors"])

    incomplete = dict(parsed)
    incomplete["generic_sdpa_call_count"] = 2
    incomplete_result = benchmark.evaluate_local_sdpa_attribution(
        "flash", recorder, incomplete
    )
    assert incomplete_result["passed"] is False
    assert any("coverage differed" in error for error in incomplete_result["errors"])

    mixed = benchmark.parse_sdpa_profiler_events(
        [
            {"key": "aten::scaled_dot_product_attention", "count": 3},
            {"key": "aten::_scaled_dot_product_flash_attention", "count": 1},
            {"key": "aten::_scaled_dot_product_attention_math", "count": 2},
        ]
    )
    mixed_recorder = json.loads(json.dumps(recorder))
    mixed_recorder["unique_signatures"][0]["signature"]["selector_eligibility"][
        "flash"
    ] = False
    mixed_auto = benchmark.evaluate_local_sdpa_attribution(
        "auto", mixed_recorder, mixed
    )
    assert mixed_auto["passed"] is False
    assert mixed_auto["selector_eligibility_all_calls"] is False
    assert any(
        "exactly one observed backend" in error for error in mixed_auto["errors"]
    )


def test_percentile_interpolates_and_validates_input():
    assert benchmark.percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert benchmark.percentile([1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.85)
    with pytest.raises(ValueError):
        benchmark.percentile([], 0.5)


def test_timing_schema_labels_the_first_step_as_post_attribution(monkeypatch):
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 10)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 20)
    monkeypatch.setattr(benchmark, "distributed_max", lambda value, _device: value)
    monkeypatch.setattr(benchmark, "distributed_sum", lambda value, _device: value)
    monkeypatch.setattr(
        benchmark, "training_step", lambda *_args, **_kwargs: {"passed": True}
    )

    metrics = benchmark.benchmark_variant(
        model=object(),
        optimizer=object(),
        variant=benchmark.REFERENCE_VARIANT,
        input_ids=torch.ones((2, 3), dtype=torch.long),
        weights=torch.ones((2, 3)),
        warmup_steps=0,
        measured_steps=1,
        tuning="lora",
        world=1,
        device=torch.device("cpu"),
    )
    assert metrics["first_post_attribution_training_step_seconds"] >= 0
    assert "first_post_attribution_optimizer_step_seconds" not in metrics
    assert "first_optimizer_step_seconds" not in metrics
    json.dumps(metrics, allow_nan=False)


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
        reference_parameter_deltas={"layer.weight": torch.tensor([0.1, 0.2])},
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

    missing = benchmark.compare_parity(12.0, {}, 12.0, reference, rtol=1e-5, atol=1e-6)
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
        reference_parameter_deltas={"layer.weight": torch.tensor([0.1, 0.2])},
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
        rounded_away["parameter_delta_actual_sensitivity"]["status"] == "rounded_away"
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

    deltas, _ = benchmark.parameter_delta_witness(model, optimizer, torch.device("cpu"))
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


def test_distributed_sdpa_attribution_aggregates_signatures_and_rank_agreement():
    def rank_report(rank: int) -> dict:
        tensor = {
            "shape": [1, 4, 128, 64],
            "stride": [32768, 64, 256, 1],
            "dtype": "bfloat16",
            "device_type": "cuda",
            "device_index": rank,
            "layout": "strided",
            "requires_grad": True,
            "is_contiguous": False,
            "is_nested": False,
            "storage_offset": 0,
        }
        signature = {
            "query": dict(tensor),
            "key": dict(tensor),
            "value": dict(tensor),
            "attn_mask": None,
            "dropout_p": 0.0,
            "is_causal": True,
            "scale": None,
            "enable_gqa": False,
            "grad_enabled": True,
            "autocast_enabled": False,
            "selector_eligibility": {
                "flash": False,
                "math": True,
                "efficient": False,
                "cudnn": False,
            },
        }
        recorder = {
            "total_calls": 2,
            "unique_signature_count": 1,
            "ordered_signature_sha256": str(rank) * 64,
            "unique_signatures": [
                {
                    "sha256": str(rank) * 64,
                    "first_call_index": 0,
                    "call_count": 2,
                    "signature": signature,
                }
            ],
        }
        profiler = benchmark.parse_sdpa_profiler_events(
            [
                {"key": "aten::scaled_dot_product_attention", "count": 2},
                {
                    "key": "aten::_scaled_dot_product_attention_math",
                    "count": 2,
                },
            ]
        )
        report = benchmark.evaluate_local_sdpa_attribution("math", recorder, profiler)
        report["relevant_state_restore"] = {
            "passed": True,
            "before_trainable_state_sha256": "a" * 64,
            "after_trainable_state_sha256": "a" * 64,
            "expected_trainable_state_sha256": "a" * 64,
            "frozen_parameters_before": {
                "sha256": "b" * 64,
                "normalized_sha256": "b" * 64,
            },
            "frozen_parameters_after": {
                "sha256": "b" * 64,
                "normalized_sha256": "b" * 64,
            },
            "named_buffers_before": {
                "sha256": "c" * 64,
                "normalized_sha256": "c" * 64,
            },
            "named_buffers_after_restore": {
                "sha256": "c" * 64,
                "normalized_sha256": "c" * 64,
            },
        }
        report.update(rank=rank, shape="timing")
        return report

    reports = [rank_report(0), rank_report(1)]
    aggregate = benchmark.aggregate_sdpa_attribution(
        reports,
        expected_world=2,
        selector_backend="math",
        shape_name="timing",
    )
    assert aggregate["passed"] is True
    assert all(aggregate["all_rank_agreement"].values())
    assert aggregate["failing_ranks"] == []
    assert len(aggregate["full_input_signature_aggregation"]) == 1
    signature = aggregate["full_input_signature_aggregation"][0]
    assert signature["total_call_count"] == 4
    assert signature["per_rank_call_count"] == {"0": 2, "1": 2}
    assert signature["signature"]["query"]["device_index"] is None
    json.dumps(aggregate, allow_nan=False)

    reports[1]["profiler"] = benchmark.parse_sdpa_profiler_events(
        [
            {"key": "aten::scaled_dot_product_attention", "count": 2},
            {"key": "aten::_scaled_dot_product_flash_attention", "count": 2},
        ]
    )
    disagreed = benchmark.aggregate_sdpa_attribution(
        reports,
        expected_world=2,
        selector_backend="math",
        shape_name="timing",
    )
    assert disagreed["passed"] is False
    assert disagreed["all_rank_agreement"]["observed_backends"] is False

    with pytest.raises(ValueError, match="duplicate"):
        benchmark.aggregate_sdpa_attribution(
            [rank_report(0), rank_report(0)],
            expected_world=2,
            selector_backend="math",
            shape_name="timing",
        )


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
    assert args.variants == ""
    assert args.reference_variant == "native-sdpa"
    assert args.steps > 0 and args.warmup_steps > 0

    forced = benchmark.build_parser().parse_args(
        [
            "--reference-variant",
            "native-sdpa-cudnn",
            "--micro-batch-size",
            "2",
            "--seq-len",
            "1024",
            "--parity-micro-batch-size",
            "2",
            "--parity-seq-len",
            "1024",
        ]
    )
    benchmark.validate_args(forced)
    assert [
        variant.name
        for variant in benchmark.select_variants(
            forced.variants, forced.reference_variant
        )
    ] == ["native-sdpa-cudnn"]
    assert (forced.parity_micro_batch_size, forced.parity_seq_len) == (
        forced.micro_batch_size,
        forced.seq_len,
    )

    broken = SimpleNamespace(**vars(args))
    broken.target_fraction = 0
    with pytest.raises(ValueError, match="target-fraction"):
        benchmark.validate_args(broken)

    broken = SimpleNamespace(**vars(args))
    broken.lora_r = 0
    with pytest.raises(ValueError, match="lora-r"):
        benchmark.validate_args(broken)


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


def test_registered_buffer_restore_repairs_identity_aliases_and_persistence():
    model = torch.nn.Module()
    original = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    model.register_buffer("cache", original, persistent=True)
    model.register_buffer("cache_alias", original, persistent=False)
    model.register_buffer("empty", None, persistent=False)
    anchor = benchmark.snapshot_named_buffers(model)
    before = benchmark.buffer_state_report(anchor)

    model.cache = model.cache.t()
    model.cache_alias = model.cache_alias.clone()
    model._non_persistent_buffers_set.add("cache")
    model._non_persistent_buffers_set.discard("cache_alias")
    model._non_persistent_buffers_set.add("ghost")
    model.empty = torch.ones(1)
    model.register_buffer("extra", torch.zeros(1), persistent=False)
    model._buffers["cache"] = model._buffers.pop("cache")
    mutated = benchmark.snapshot_named_buffers(model)
    mutation = benchmark.compare_registered_tensor_states(anchor, mutated)

    assert mutation["passed"] is False
    assert mutation["extra_registrations"] == ["extra"]
    assert "cache" in mutation["object_identity_changed"]
    assert "cache" in mutation["metadata_changed"]
    assert "cache" in mutation["persistence_changed"]
    assert "cache_alias" in mutation["aliasing_changed"]
    assert mutation["registration_order_changed"] == ["<root>"]
    assert mutation["module_persistence_set_changed"] == ["<root>"]

    benchmark.restore_named_buffers(model, anchor)
    restored = benchmark.snapshot_named_buffers(model)
    comparison = benchmark.compare_registered_tensor_states(anchor, restored)
    after = benchmark.buffer_state_report(restored)
    assert comparison["passed"] is True
    assert model.cache is original and model.cache_alias is original
    assert model.empty is None and "extra" not in model._buffers
    assert "cache" not in model._non_persistent_buffers_set
    assert "cache_alias" in model._non_persistent_buffers_set
    assert "ghost" not in model._non_persistent_buffers_set
    assert list(model._buffers) == ["cache", "cache_alias", "empty"]
    assert before == after


def test_frozen_state_detects_equal_value_identity_metadata_and_alias_changes():
    model = torch.nn.Module()
    original = torch.nn.Parameter(
        torch.tensor([[1.0, 2.0], [2.0, 1.0]]), requires_grad=False
    )
    model.register_parameter("frozen", original)
    model.register_parameter("frozen_alias", original)
    model.register_parameter("empty", None)
    anchor = benchmark.snapshot_frozen_parameter_state(model)
    before = benchmark.frozen_parameter_state_report(anchor)

    model.frozen = torch.nn.Parameter(original.detach().t(), requires_grad=False)
    current = benchmark.snapshot_frozen_parameter_state(model)
    comparison = benchmark.compare_registered_tensor_states(anchor, current)
    after = benchmark.frozen_parameter_state_report(current)

    assert comparison["passed"] is False
    assert "frozen" in comparison["object_identity_changed"]
    assert "frozen" in comparison["metadata_changed"]
    assert comparison["aliasing_changed"] == ["frozen_alias"]
    assert before["normalized_sha256"] != after["normalized_sha256"]


def test_same_object_storage_swap_is_detected_for_parameters_and_buffers():
    model = torch.nn.Module()
    parameter = torch.nn.Parameter(torch.ones(4), requires_grad=False)
    buffer = torch.arange(4.0)
    model.register_parameter("frozen", parameter)
    model.register_buffer("cache", buffer)
    frozen_anchor = benchmark.snapshot_frozen_parameter_state(model)
    buffer_anchor = benchmark.snapshot_named_buffers(model)
    parameter_pointer = parameter.data_ptr()
    buffer_pointer = buffer.data_ptr()

    parameter.data = parameter.detach().clone()
    buffer.data = buffer.detach().clone()
    assert parameter.data_ptr() != parameter_pointer
    assert buffer.data_ptr() != buffer_pointer
    frozen_current = benchmark.snapshot_frozen_parameter_state(model)
    buffer_current = benchmark.snapshot_named_buffers(model)
    frozen_comparison = benchmark.compare_registered_tensor_states(
        frozen_anchor, frozen_current
    )
    buffer_comparison = benchmark.compare_registered_tensor_states(
        buffer_anchor, buffer_current
    )

    assert frozen_comparison["object_identity_changed"] == []
    assert buffer_comparison["object_identity_changed"] == []
    assert frozen_comparison["metadata_changed"] == ["frozen"]
    assert buffer_comparison["metadata_changed"] == ["cache"]
    frozen_metadata = frozen_anchor["entries"]["frozen"]["metadata"]
    buffer_metadata = buffer_anchor["entries"]["cache"]["metadata"]
    for metadata in (frozen_metadata, buffer_metadata):
        assert metadata["object_type"].startswith("torch.")
        assert metadata["data_ptr"] > 0
        assert metadata["storage_data_ptr"] > 0
        assert metadata["storage_identity"] > 0
        assert metadata["storage_nbytes"] == 16


def test_registered_state_digest_normalizes_only_device_index():
    model = torch.nn.Module()
    model.register_buffer("cache", torch.ones(2))
    rank_zero = benchmark.snapshot_named_buffers(model)
    rank_one = {
        "kind": rank_zero["kind"],
        "modules": rank_zero["modules"],
        "entries": {
            name: {
                **entry,
                "metadata": {**entry["metadata"], "device_index": 1},
            }
            for name, entry in rank_zero["entries"].items()
        },
    }
    rank_zero["entries"]["cache"]["metadata"]["device_index"] = 0
    first = benchmark.buffer_state_report(rank_zero)
    second = benchmark.buffer_state_report(rank_one)
    assert first["sha256"] != second["sha256"]
    assert first["normalized_sha256"] == second["normalized_sha256"]


def test_trainable_anchor_restore_and_lora_factor_validation():
    model = torch.nn.Module()
    model.lora_A = torch.nn.Linear(2, 2, bias=False)
    model.lora_B = torch.nn.Linear(2, 2, bias=False)
    model.lora_B.weight.data.zero_()
    with pytest.raises(RuntimeError, match="lora_B"):
        benchmark.lora_factor_nonzero_report(benchmark.snapshot_trainable_state(model))

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

        def get_output_embeddings(self):
            return self.lm_head

    model = Model()
    model.lm_head.requires_grad_(False)
    assert benchmark.validate_lora_output_head(model) == {
        "frozen": True,
        "adapted": False,
        "parameter_count": 4,
    }
    model.lm_head.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="frozen, unadapted"):
        benchmark.validate_lora_output_head(model)


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


def test_report_status_requires_exact_unique_planned_variant_order():
    for completed in (["a", "a"], ["b", "a"], ["a", "c"]):
        report = {
            "status": "incomplete",
            "planned_variants": ["a", "b"],
            "completed_variants": [],
            "fatal": None,
            "variants": [
                {"variant": {"name": name}, "status": "passed"} for name in completed
            ],
        }
        benchmark.finalize_report_status(report, False, None, None)
        assert report["status"] == "incomplete"


def test_atomic_report_write_and_collective_write_failure(tmp_path, monkeypatch):
    output = tmp_path / "reports" / "result.json"
    benchmark.write_report_atomic(output, '{"status":"passed"}')
    assert json.loads(output.read_text()) == {"status": "passed"}
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))

    published = []

    def broadcast(value, rank):
        if rank == 0:
            published.append(value)
            return value
        assert value is None
        return published[-1]

    def fail_write(_output, _serialized):
        raise OSError("disk full")

    monkeypatch.setattr(benchmark, "broadcast_object", broadcast)
    monkeypatch.setattr(benchmark, "write_report_atomic", fail_write)
    rank_zero_report = {
        "status": "incomplete",
        "planned_variants": ["a"],
        "completed_variants": [],
        "fatal": None,
        "variants": [{"variant": {"name": "a"}, "status": "passed"}],
    }
    rank_zero = benchmark.publish_report_collectively(
        rank_zero_report,
        output,
        rank=0,
        failed=False,
        fatal_phase=None,
        fatal_reason=None,
    )
    rank_one = benchmark.publish_report_collectively(
        {},
        output,
        rank=1,
        failed=False,
        fatal_phase=None,
        fatal_reason=None,
    )
    assert rank_zero == rank_one
    assert rank_zero["passed"] is False
    assert "disk full" in rank_zero["error"]


def test_main_uses_guarded_destroy_without_a_final_barrier(monkeypatch):
    events = []
    monkeypatch.setattr(benchmark, "_main", lambda _argv, _controller: 0)
    monkeypatch.setattr(benchmark.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        benchmark.dist,
        "destroy_process_group",
        lambda: events.append("destroy"),
    )
    monkeypatch.setattr(
        benchmark.dist,
        "barrier",
        lambda: pytest.fail("main must not use a fallible final barrier"),
    )
    assert benchmark.main([]) == 0
    assert events == ["destroy"]


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
