"""Correctness and fail-closed tests for the optional causal kernel lane."""

from __future__ import annotations

import sys
from pathlib import Path
from types import MethodType
from types import ModuleType, SimpleNamespace

import pytest
import torch

from yeto import causal_kernels


ROOT = Path(__file__).resolve().parents[1]


def test_optional_dependency_pins_match_runtime_contract():
    project = (ROOT / "pyproject.toml").read_text()
    assert (
        f'a100-liger = ["liger-kernel=={causal_kernels.LIGER_KERNEL_VERSION}"]'
        in project
    )
    assert (
        f'a100-flash-attn = ["flash-attn=={causal_kernels.FLASH_ATTN_VERSION}"]'
        in project
    )


def test_attention_load_kwargs_map_only_explicit_requests():
    device = SimpleNamespace(type="cuda")
    assert causal_kernels.attention_load_kwargs("auto", device, torch.bfloat16) == {}
    assert causal_kernels.attention_load_kwargs("sdpa", device, torch.float32) == {
        "attn_implementation": "sdpa"
    }


def test_flash_attention_requires_cuda_bf16_and_the_exact_pin(monkeypatch):
    with pytest.raises(RuntimeError, match="requires CUDA"):
        causal_kernels.attention_load_kwargs(
            "flash-attn-2", SimpleNamespace(type="cpu"), torch.bfloat16
        )
    with pytest.raises(RuntimeError, match="requires BF16"):
        causal_kernels.attention_load_kwargs(
            "flash-attn-2", SimpleNamespace(type="cuda"), torch.float32
        )

    monkeypatch.setattr(causal_kernels, "_require_a100", lambda device: None)
    monkeypatch.setattr(
        causal_kernels.metadata,
        "version",
        lambda name: causal_kernels.FLASH_ATTN_VERSION,
    )
    monkeypatch.setattr(causal_kernels.importlib.util, "find_spec", lambda name: object())
    assert causal_kernels.attention_load_kwargs(
        "flash-attn-2", SimpleNamespace(type="cuda"), torch.bfloat16
    ) == {"attn_implementation": "flash_attention_2"}


def test_flash_attention_rejects_a_different_installed_version(monkeypatch):
    monkeypatch.setattr(causal_kernels, "_require_a100", lambda device: None)
    monkeypatch.setattr(causal_kernels.metadata, "version", lambda name: "2.7.0")
    with pytest.raises(RuntimeError, match="2.8.3 is required"):
        causal_kernels.attention_load_kwargs(
            "flash-attn-2", SimpleNamespace(type="cuda"), torch.bfloat16
        )


def test_explicit_attention_resolution_must_match():
    model = SimpleNamespace(config=SimpleNamespace(_attn_implementation="sdpa"))
    assert causal_kernels.resolved_attention_backend(model, "sdpa") == "sdpa"
    with pytest.raises(RuntimeError, match="resolved 'sdpa'"):
        causal_kernels.resolved_attention_backend(model, "flash-attn-2")
    composite = SimpleNamespace(
        config=SimpleNamespace(
            _attn_implementation={"text": "sdpa", "vision": "sdpa"}
        )
    )
    assert "'text': 'sdpa'" in causal_kernels.resolved_attention_backend(
        composite, "sdpa"
    )


def test_optimized_lane_rejects_non_a100_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "NVIDIA H100 80GB")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    with pytest.raises(RuntimeError, match="scoped to A100/SM80"):
        causal_kernels.validate_kernel_request(
            "liger", "cross_entropy", torch.device("cuda", 0), torch.bfloat16
        )


def test_native_kernel_request_preserves_all_loss_types():
    causal_kernels.validate_kernel_request(
        "native",
        "custom:loss.py",
        SimpleNamespace(type="cpu"),
        torch.float32,
    )


def test_liger_request_fails_closed_for_device_loss_quantization_and_pin(monkeypatch):
    with pytest.raises(RuntimeError, match="requires CUDA"):
        causal_kernels.validate_kernel_request(
            "liger", "cross_entropy", SimpleNamespace(type="cpu"), torch.float32
        )
    monkeypatch.setattr(causal_kernels, "_require_a100", lambda device: None)
    with pytest.raises(ValueError, match="only the built-in cross_entropy"):
        causal_kernels.validate_kernel_request(
            "liger", "pickle:loss.pkl", SimpleNamespace(type="cuda"), torch.bfloat16
        )
    with pytest.raises(ValueError, match="does not support a quantized base"):
        causal_kernels.validate_kernel_request(
            "liger",
            "cross_entropy",
            SimpleNamespace(type="cuda"),
            torch.bfloat16,
            "nf4",
        )
    with pytest.raises(ValueError, match="only for --tuning lora"):
        causal_kernels.validate_kernel_request(
            "liger",
            "cross_entropy",
            SimpleNamespace(type="cuda"),
            torch.bfloat16,
            tuning="full",
        )
    with pytest.raises(ValueError, match="only for --tuning lora"):
        causal_kernels.validate_kernel_request(
            "liger",
            "cross_entropy",
            SimpleNamespace(type="cuda"),
            torch.bfloat16,
        )
    with pytest.raises(ValueError, match="only for --shard ddp"):
        causal_kernels.validate_kernel_request(
            "liger",
            "cross_entropy",
            SimpleNamespace(type="cuda"),
            torch.bfloat16,
            tuning="lora",
            shard="fsdp",
        )

    monkeypatch.setattr(causal_kernels, "_require_exact_package", lambda *args: None)
    causal_kernels.validate_kernel_request(
        "liger",
        "cross_entropy",
        SimpleNamespace(type="cuda"),
        torch.bfloat16,
        tuning="lora",
        shard="ddp",
    )


def _install_fake_qwen2_apply(monkeypatch, apply_fn):
    root = ModuleType("liger_kernel")
    root.__path__ = []
    transformers = ModuleType("liger_kernel.transformers")
    transformers.__path__ = []
    transformers.apply_liger_kernel_to_qwen2 = apply_fn
    functional = ModuleType("liger_kernel.transformers.functional")

    def fused_linear_cross_entropy(input, weight, target, accum_dtype=None):
        del input, weight, target, accum_dtype

    functional.liger_fused_linear_cross_entropy = fused_linear_cross_entropy
    transformers.functional = functional
    monkeypatch.setitem(sys.modules, "liger_kernel", root)
    monkeypatch.setitem(sys.modules, "liger_kernel.transformers", transformers)
    monkeypatch.setitem(
        sys.modules, "liger_kernel.transformers.functional", functional
    )


def _supported_qwen2_apply(
    rope=True,
    cross_entropy=False,
    fused_linear_cross_entropy=True,
    rms_norm=True,
    swiglu=True,
    model=None,
):
    del rope, cross_entropy, fused_linear_cross_entropy, rms_norm, swiglu, model


def _tiny_qwen2_model():
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2Config,
        Qwen2ForCausalLM,
    )

    return Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    )


def _replacement_forward(
    self,
    input_ids=None,
    labels=None,
    use_cache=None,
    skip_logits=None,
    **kwargs,
):
    del self, input_ids, labels, use_cache, skip_logits, kwargs


def test_liger_model_support_requires_qwen2_and_the_exact_public_controls(monkeypatch):
    _install_fake_qwen2_apply(monkeypatch, _supported_qwen2_apply)
    assert causal_kernels.require_liger_model_support(
        SimpleNamespace(model_type="qwen2")
    ) == "qwen2"
    with pytest.raises(RuntimeError, match="supports only Hugging Face Qwen2"):
        causal_kernels.require_liger_model_support(
            SimpleNamespace(model_type="unknown")
        )

    def missing_controls(fused_linear_cross_entropy=True):
        del fused_linear_cross_entropy

    _install_fake_qwen2_apply(monkeypatch, missing_controls)
    with pytest.raises(RuntimeError, match="exact controls"):
        causal_kernels.require_liger_model_support(
            SimpleNamespace(model_type="qwen2")
        )

    _install_fake_qwen2_apply(monkeypatch, _supported_qwen2_apply)

    def missing_accum_dtype(input, weight, target):
        del input, weight, target

    sys.modules[
        "liger_kernel.transformers.functional"
    ].liger_fused_linear_cross_entropy = missing_accum_dtype
    with pytest.raises(RuntimeError, match="silently filtered"):
        causal_kernels.require_liger_model_support(
            SimpleNamespace(model_type="qwen2")
        )


def test_fused_linear_ce_application_changes_only_one_qwen2_instance(monkeypatch):
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2Config,
        Qwen2ForCausalLM,
    )

    calls = []

    def replacement_forward(
        self,
        input_ids=None,
        labels=None,
        use_cache=None,
        skip_logits=None,
        **kwargs,
    ):
        del self, input_ids, labels, use_cache, skip_logits, kwargs

    def apply(
        rope=True,
        cross_entropy=False,
        fused_linear_cross_entropy=True,
        rms_norm=True,
        swiglu=True,
        model=None,
    ):
        calls.append(
            {
                "rope": rope,
                "cross_entropy": cross_entropy,
                "fused_linear_cross_entropy": fused_linear_cross_entropy,
                "rms_norm": rms_norm,
                "swiglu": swiglu,
                "model": model,
            }
        )
        model.forward = MethodType(replacement_forward, model)

    _install_fake_qwen2_apply(monkeypatch, apply)
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    model = Qwen2ForCausalLM(config)
    untouched_model = Qwen2ForCausalLM(config)
    class_forward = Qwen2ForCausalLM.forward
    untouched_forward = untouched_model.forward.__func__

    report = causal_kernels.apply_liger_fused_linear_ce(model)

    assert calls == [
        {
            "rope": False,
            "cross_entropy": False,
            "fused_linear_cross_entropy": True,
            "rms_norm": False,
            "swiglu": False,
            "model": model,
        }
    ]
    assert model.forward.__func__ is replacement_forward
    assert Qwen2ForCausalLM.forward is class_forward
    assert untouched_model.forward.__func__ is untouched_forward
    assert report["layer_backend"] == causal_kernels.NATIVE_LAYER_BACKEND
    assert (
        report["loss_implementation"]
        == causal_kernels.FUSED_LINEAR_CE_IMPLEMENTATION
    )
    assert report["patch_scope"] == "model-instance-forward-only"
    assert all(report["invariants"].values())
    assert report["state_attestation"] == {
        "tensor_content_digest": "sha256",
        "tensor_digest_chunk_bytes": 4 * 1024 * 1024,
        "retains_duplicate_model_tensors": False,
        "config_and_builtin_containers": "structural-sha256",
        "failure_contract": "verified-rollback-or-fatal-poisoned-process",
    }


def test_fused_linear_ce_application_rejects_global_class_mutation(monkeypatch):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

    from transformers.models.qwen2 import modeling_qwen2

    original_forward = Qwen2ForCausalLM.forward
    original_norm = modeling_qwen2.Qwen2RMSNorm
    original_cross_entropy = torch.nn.functional.cross_entropy

    def replacement_cross_entropy(*args, **kwargs):
        del args, kwargs

    def bad_apply(
        rope=True,
        cross_entropy=False,
        fused_linear_cross_entropy=True,
        rms_norm=True,
        swiglu=True,
        model=None,
    ):
        del rope, cross_entropy, fused_linear_cross_entropy, rms_norm, swiglu
        Qwen2ForCausalLM.forward = _replacement_forward
        modeling_qwen2.Qwen2RMSNorm = object
        torch.nn.functional.cross_entropy = replacement_cross_entropy
        model.forward = MethodType(_replacement_forward, model)

    _install_fake_qwen2_apply(monkeypatch, bad_apply)
    model = _tiny_qwen2_model()

    with pytest.raises(causal_kernels.KernelIsolationError) as caught:
        causal_kernels.apply_liger_fused_linear_ce(model)

    error = caught.value
    assert error.rollback_complete
    assert not error.process_state_poisoned
    assert not error.fatal
    assert Qwen2ForCausalLM.forward is original_forward
    assert modeling_qwen2.Qwen2RMSNorm is original_norm
    assert torch.nn.functional.cross_entropy is original_cross_entropy
    assert "forward" not in vars(model)
    assert all(error.rollback_report["process_invariants"].values())
    assert all(error.rollback_report["model_invariants"].values())


def test_fused_linear_ce_rolls_back_a_partially_applied_exception(monkeypatch):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

    original_forward = Qwen2ForCausalLM.forward

    def bad_apply(
        rope=True,
        cross_entropy=False,
        fused_linear_cross_entropy=True,
        rms_norm=True,
        swiglu=True,
        model=None,
    ):
        del rope, cross_entropy, fused_linear_cross_entropy, rms_norm, swiglu
        Qwen2ForCausalLM.forward = _replacement_forward
        model.forward = MethodType(_replacement_forward, model)
        raise ValueError("partial apply")

    _install_fake_qwen2_apply(monkeypatch, bad_apply)
    model = _tiny_qwen2_model()
    with pytest.raises(causal_kernels.KernelIsolationError) as caught:
        causal_kernels.apply_liger_fused_linear_ce(model)

    error = caught.value
    assert isinstance(error.__cause__, ValueError)
    assert error.failed_invariants == ("third_party_apply_completed",)
    assert error.rollback_complete and not error.process_state_poisoned
    assert Qwen2ForCausalLM.forward is original_forward
    assert "forward" not in vars(model)


def test_fused_linear_ce_detects_in_place_parameter_content_mutation(monkeypatch):
    model = _tiny_qwen2_model()
    parameter = model.model.embed_tokens.weight
    before = parameter.detach().clone()
    before_version = parameter._version

    def bad_apply(
        rope=True,
        cross_entropy=False,
        fused_linear_cross_entropy=True,
        rms_norm=True,
        swiglu=True,
        model=None,
    ):
        del rope, cross_entropy, fused_linear_cross_entropy, rms_norm, swiglu
        # `.data` deliberately bypasses autograd's normal version bump; the
        # streaming content digest must still detect it.
        model.model.embed_tokens.weight.data.add_(1)
        model.forward = MethodType(_replacement_forward, model)

    _install_fake_qwen2_apply(monkeypatch, bad_apply)
    with pytest.raises(causal_kernels.KernelIsolationError) as caught:
        causal_kernels.apply_liger_fused_linear_ce(model)

    error = caught.value
    assert parameter._version == before_version
    assert not torch.equal(parameter, before)
    assert "parameter_contents_unchanged" in error.failed_invariants
    assert error.process_state_poisoned and error.fatal
    assert not error.rollback_complete
    assert not error.rollback_report["model_invariants"][
        "parameter_contents_unchanged"
    ]
    assert "forward" not in vars(model)


def test_fused_linear_ce_detects_in_place_buffer_content_mutation(monkeypatch):
    model = _tiny_qwen2_model()
    layer = model.model.layers[0]
    layer.register_buffer("isolation_probe", torch.tensor([1.0, 2.0]))
    before = layer.isolation_probe.detach().clone()

    def bad_apply(
        rope=True,
        cross_entropy=False,
        fused_linear_cross_entropy=True,
        rms_norm=True,
        swiglu=True,
        model=None,
    ):
        del rope, cross_entropy, fused_linear_cross_entropy, rms_norm, swiglu
        model.model.layers[0].isolation_probe.data.mul_(3)
        model.forward = MethodType(_replacement_forward, model)

    _install_fake_qwen2_apply(monkeypatch, bad_apply)
    with pytest.raises(causal_kernels.KernelIsolationError) as caught:
        causal_kernels.apply_liger_fused_linear_ce(model)

    error = caught.value
    assert not torch.equal(layer.isolation_probe, before)
    assert "buffer_contents_unchanged" in error.failed_invariants
    assert error.process_state_poisoned and error.fatal
    assert not error.rollback_report["model_invariants"][
        "buffer_contents_unchanged"
    ]


def test_fused_linear_ce_restores_exact_nested_module_class(monkeypatch):
    model = _tiny_qwen2_model()
    target = model.model.layers[0].mlp
    original_type = type(target)
    mutated_type = type("MutatedQwen2MLP", (original_type,), {})

    def bad_apply(
        rope=True,
        cross_entropy=False,
        fused_linear_cross_entropy=True,
        rms_norm=True,
        swiglu=True,
        model=None,
    ):
        del rope, cross_entropy, fused_linear_cross_entropy, rms_norm, swiglu
        target.__class__ = mutated_type
        model.forward = MethodType(_replacement_forward, model)

    _install_fake_qwen2_apply(monkeypatch, bad_apply)
    with pytest.raises(causal_kernels.KernelIsolationError) as caught:
        causal_kernels.apply_liger_fused_linear_ce(model)

    error = caught.value
    assert "module_types_unchanged" in error.failed_invariants
    assert error.rollback_complete and not error.process_state_poisoned
    assert type(target) is original_type
    assert "forward" not in vars(model)


def test_fused_linear_ce_detects_in_place_module_container_mutation(monkeypatch):
    model = _tiny_qwen2_model()
    target = model.model.layers[0]
    target.isolation_probe = [{"value": 1}]

    def bad_apply(
        rope=True,
        cross_entropy=False,
        fused_linear_cross_entropy=True,
        rms_norm=True,
        swiglu=True,
        model=None,
    ):
        del rope, cross_entropy, fused_linear_cross_entropy, rms_norm, swiglu
        target.isolation_probe[0]["value"] = 2
        model.forward = MethodType(_replacement_forward, model)

    _install_fake_qwen2_apply(monkeypatch, bad_apply)
    with pytest.raises(causal_kernels.KernelIsolationError) as caught:
        causal_kernels.apply_liger_fused_linear_ce(model)

    error = caught.value
    assert "nested_container_state_unchanged" in error.failed_invariants
    assert error.process_state_poisoned and not error.rollback_complete
    assert target.isolation_probe == [{"value": 2}]
    assert "forward" not in vars(model)


def test_fused_linear_ce_detects_in_place_config_container_mutation(monkeypatch):
    model = _tiny_qwen2_model()
    model.config.isolation_probe = {"values": [1]}

    def bad_apply(
        rope=True,
        cross_entropy=False,
        fused_linear_cross_entropy=True,
        rms_norm=True,
        swiglu=True,
        model=None,
    ):
        del rope, cross_entropy, fused_linear_cross_entropy, rms_norm, swiglu
        model.config.isolation_probe["values"].append(2)
        model.forward = MethodType(_replacement_forward, model)

    _install_fake_qwen2_apply(monkeypatch, bad_apply)
    with pytest.raises(causal_kernels.KernelIsolationError) as caught:
        causal_kernels.apply_liger_fused_linear_ce(model)

    error = caught.value
    assert (
        "model_config_identity_type_and_structure_unchanged"
        in error.failed_invariants
    )
    assert error.process_state_poisoned and not error.rollback_complete
    assert model.config.isolation_probe == {"values": [1, 2]}


def test_binary_mask_labels_shift_count_and_reject_fractional_weights():
    ids = torch.tensor([[2, 3, 4, 5]])
    labels, count = causal_kernels.binary_mask_labels(
        ids, torch.tensor([[1.0, 0.0, 1.0, 1.0]])
    )
    assert labels.tolist() == [[2, -100, 4, 5]]
    assert count == 2

    with pytest.raises(ValueError, match="fractional weights"):
        causal_kernels.binary_mask_labels(
            ids, torch.tensor([[1.0, 0.5, 1.0, 0.0]])
        )


def test_liger_forward_requests_a_sum_and_never_accepts_logits():
    class Model:
        def __init__(self, logits=None):
            self.logits = logits
            self.kwargs = None

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(loss=torch.tensor(7.5), logits=self.logits)

    ids = torch.tensor([[1, 2, 3]])
    weights = torch.tensor([[0.0, 1.0, 1.0]])
    model = Model()
    loss, count = causal_kernels.liger_sft_forward(model, ids, weights)
    assert loss == 7.5 and count == 2
    assert model.kwargs["num_items_in_batch"] == 1
    assert model.kwargs["accum_dtype"] is torch.float32
    assert model.kwargs["skip_logits"] is True
    assert model.kwargs["use_cache"] is False
    assert model.kwargs["labels"].tolist() == [[-100, 2, 3]]

    with pytest.raises(RuntimeError, match="materialized logits"):
        causal_kernels.liger_sft_forward(
            Model(logits=torch.empty(1, 3, 5)), ids, weights
        )
