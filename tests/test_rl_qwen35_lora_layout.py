"""A-T3-3: PEFT-predicted LoRA layout must equal Megatron-Bridge's export for Qwen3.5.

Fixture ``qwen35_08b_lora_export_smoke5.json`` is the real CanonicalLoRA export
captured from ``codex-smoke-20261003-5`` (Qwen/Qwen3.5-0.8B, rank 16,
``--lora-targets attention``).  Megatron exported ``q_proj.lora_B[2048, 16]``
while PEFT predicts ``[4096, 16]``: the HF ``q_proj`` carries the attention
output gate (``attn_output_gate``), CanonicalLoRA's split ``linear_q`` adapter
does not.  Gated attention therefore trains only ``o_proj`` + GDN ``out_proj``.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from yeto.rl import learner as rl_learner
from yeto.rl.core import CanonicalTensorSpec, canonical_layout_hash
from yeto.rl.export import adapter_targets, attention_output_gated, derive_peft_lora_specs, target_modules

FIXTURE = Path(__file__).parent / "fixtures" / "qwen35_08b_lora_export_smoke5.json"

# Qwen/Qwen3.5-0.8B@2fc06364 config.json, minus tokens/vision irrelevant to layout.
QWEN35_08B_CONFIG = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "tie_word_embeddings": True,
    "text_config": {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attn_output_gate": True,
        "full_attention_interval": 4,
        "head_dim": 256,
        "hidden_act": "silu",
        "hidden_size": 1024,
        "intermediate_size": 3584,
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 6,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 16,
        "linear_value_head_dim": 128,
        "max_position_embeddings": 262144,
        "mlp_only_layers": [],
        "model_type": "qwen3_5_text",
        "num_attention_heads": 8,
        "num_hidden_layers": 24,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-06,
        "tie_word_embeddings": True,
        "vocab_size": 248320,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
        },
    },
    "vision_config": {
        "depth": 12,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 768,
        "in_channels": 3,
        "intermediate_size": 3072,
        "model_type": "qwen3_5",
        "num_heads": 12,
        "num_position_embeddings": 2304,
        "out_hidden_size": 1024,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
}


def _exported_specs(predicate=lambda name: True):
    tensors = json.loads(FIXTURE.read_text())["tensors"]
    return tuple(
        sorted(
            CanonicalTensorSpec(name, tuple(shape), "float32", shape[0] * shape[1])
            for name, shape in tensors.items()
            if predicate(name)
        )
    )


@pytest.fixture(scope="module")
def qwen35_model_dir(tmp_path_factory):
    pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3_5ForConditionalGeneration"):
        pytest.skip("transformers lacks Qwen3.5")
    model = tmp_path_factory.mktemp("qwen35")
    (model / "config.json").write_text(json.dumps(QWEN35_08B_CONFIG))
    return str(model)


def test_attention_output_gate_detected_on_nested_text_config():
    gated = types.SimpleNamespace(text_config=types.SimpleNamespace(attn_output_gate=True))
    assert attention_output_gated(gated)
    assert not attention_output_gated(types.SimpleNamespace(text_config=None, attn_output_gate=False))
    assert target_modules("attention", gated) == r".*\.(o_proj|out_proj)$"
    # Ungated attention keeps the full q/k/v/o regex.
    plain = types.SimpleNamespace(model_type="qwen3")
    assert "q_proj" in target_modules("attention", plain)


def test_fixture_documents_megatron_gate_unaware_q_export(qwen35_model_dir):
    from yeto.learner import _ATTENTION_TARGETS

    exported = _exported_specs()
    assert len(exported) == 84
    predicted = derive_peft_lora_specs(qwen35_model_dir, None, rank=16, targets=_ATTENTION_TARGETS)
    assert [s.name for s in predicted] == [s.name for s in exported]
    diff = [(p, e) for p, e in zip(predicted, exported) if p.shape != e.shape]
    assert diff and all(p.name.endswith("self_attn.q_proj.lora_B.weight") for p, _ in diff)
    assert {(p.shape, e.shape) for p, e in diff} == {((4096, 16), (2048, 16))}  # 2 * 8 * 256 vs 8 * 256
    assert canonical_layout_hash(predicted) != canonical_layout_hash(exported)


def test_gated_attention_prediction_matches_megatron_export(qwen35_model_dir):
    predicted = derive_peft_lora_specs(qwen35_model_dir, None, rank=16, targets="attention")
    exported = _exported_specs(lambda name: ".o_proj." in name or ".out_proj." in name)
    assert len(predicted) == 48  # 6 full-attention o_proj + 18 GDN out_proj, A and B
    assert predicted == exported
    assert canonical_layout_hash(predicted) == canonical_layout_hash(exported)
    assert adapter_targets(predicted) == ["o_proj", "out_proj"]


def test_megatron_adapter_targets_fail_closed_on_gated_qkv():
    qkv = types.SimpleNamespace(
        megatron_param="language_model.decoder.layers.3.self_attention.linear_qkv.weight",
        hf_param={
            "q": "model.language_model.layers.3.self_attn.q_proj.weight",
            "k": "model.language_model.layers.3.self_attn.k_proj.weight",
            "v": "model.language_model.layers.3.self_attn.v_proj.weight",
        },
    )
    out_proj = types.SimpleNamespace(
        megatron_param="language_model.decoder.layers.0.self_attention.out_proj.weight",
        hf_param="model.language_model.layers.0.linear_attn.out_proj.weight",
    )
    mappings = {value: qkv for value in qkv.hf_param.values()} | {out_proj.hf_param: out_proj}
    bridge = types.SimpleNamespace(
        _model_bridge=types.SimpleNamespace(
            mapping_registry=lambda: types.SimpleNamespace(hf_to_megatron_lookup=mappings.get)
        )
    )

    def specs(*modules):
        return tuple(
            types.SimpleNamespace(name=f"base_model.model.{m}.lora_{side}.weight")
            for m in modules
            for side in ("A", "B")
        )

    gdn = "model.language_model.layers.0.linear_attn.out_proj"
    q = "model.language_model.layers.3.self_attn.q_proj"
    assert rl_learner.megatron_adapter_targets(specs(gdn), bridge, attention_output_gate=True) == [
        "language_model.decoder.layers.0.self_attention.out_proj"
    ]
    assert "linear_q" in rl_learner.megatron_adapter_targets(specs(q), bridge)[0]
    with pytest.raises(ValueError, match="attention_output_gate"):
        rl_learner.megatron_adapter_targets(specs(q, gdn), bridge, attention_output_gate=True)
