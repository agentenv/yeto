from __future__ import annotations

import struct

import pytest
import torch

from yeto.export import CKPT_MAGIC
from yeto.rl.core import (
    canonical_layout_hash,
    canonical_lora_config_hash,
    canonical_state,
    flat_tensor,
    tensors_from_flat,
)
from yeto.rl.export import derive_peft_lora_specs, export_rl_checkpoint


def _model(tmp_path):
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    path = tmp_path / "model"
    config.save_pretrained(path)
    return path, config


def test_attention_regex_is_resolved_before_peft_moe_conversion(tmp_path):
    transformers = pytest.importorskip("transformers")
    config = transformers.OlmoeConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_experts=2,
        num_experts_per_tok=1,
    )
    path = tmp_path / "olmoe"
    config.save_pretrained(path)

    specs = derive_peft_lora_specs(str(path), None, rank=2, targets="attention")

    assert specs
    assert all("self_attn" in spec.name for spec in specs)


MODEL_REVISION = "a" * 40


def _write_checkpoint(
    path, values: torch.Tensor, layout_hash: str | None, version=2, ledger_size=2
):
    body = bytearray()
    body += struct.pack("<IQI", CKPT_MAGIC, version, 1)
    body += struct.pack("<QQ", version, values.numel())
    body += values.float().numpy().tobytes()
    body += torch.zeros_like(values).float().numpy().tobytes()
    body += struct.pack("<I", ledger_size if version else 0)
    for learner_id in range(ledger_size if version else 0):
        body += struct.pack("<IQQQ", learner_id, version, version, version)
    if layout_hash is not None:
        body += bytes.fromhex(layout_hash)
    path.write_bytes(body)


def test_committed_checkpoint_exports_and_standard_peft_reloads(tmp_path):
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    model_path, config = _model(tmp_path)
    specs = derive_peft_lora_specs(
        str(model_path), None, rank=2, targets="all-linear"
    )
    values = torch.arange(sum(spec.numel for spec in specs), dtype=torch.float32) / 10
    checkpoint = tmp_path / "state.ckpt"
    layout_hash = canonical_layout_hash(specs)
    _write_checkpoint(checkpoint, values, layout_hash)

    output = tmp_path / "adapter"
    state = export_rl_checkpoint(
        checkpoint,
        output,
        model=str(model_path),
        model_revision=MODEL_REVISION,
        rank=2,
    )
    assert state.policy_version == 2
    assert (output / "adapter_model.safetensors").is_file()
    assert (output / "adapter_config.json").is_file()

    base = transformers.AutoModelForCausalLM.from_config(config)
    loaded = peft.PeftModel.from_pretrained(base, output)
    loaded_state = peft.get_peft_model_state_dict(loaded)
    expected = tensors_from_flat(values, specs)
    assert set(loaded_state) == set(expected)
    assert all(torch.equal(loaded_state[name], expected[name]) for name in expected)


def test_export_ignores_ledger_and_rejects_tensor_mismatch(tmp_path):
    model_path, _ = _model(tmp_path)
    specs = derive_peft_lora_specs(
        str(model_path), None, rank=2, targets="all-linear"
    )
    layout_hash = canonical_layout_hash(specs)
    state = canonical_state(
        2,
        tensors_from_flat(
            torch.zeros(sum(spec.numel for spec in specs)),
            specs,
        ),
        base_model_revision=MODEL_REVISION,
        lora_config_hash=canonical_lora_config_hash(
            rank=2,
            target_modules=[
                spec.name.rsplit(".lora_", 1)[0].rsplit(".", 1)[-1]
                for spec in specs
            ],
        ),
        layout_hash=layout_hash,
    )
    checkpoint = tmp_path / "state.ckpt"
    values = flat_tensor(state.tensors)
    _write_checkpoint(checkpoint, values, layout_hash, ledger_size=0)
    exported = export_rl_checkpoint(
        checkpoint,
        tmp_path / "out",
        model=str(model_path),
        model_revision=MODEL_REVISION,
        rank=2,
    )
    assert all(
        torch.equal(exported.tensors[name], state.tensors[name])
        for name in state.tensors
    )

    _write_checkpoint(checkpoint, values, None)
    with pytest.raises(ValueError, match="does not contain a canonical layout hash"):
        export_rl_checkpoint(
            checkpoint,
            tmp_path / "out",
            model=str(model_path),
            model_revision=MODEL_REVISION,
            rank=2,
        )

    _write_checkpoint(checkpoint, values, "f" * 64)
    with pytest.raises(ValueError, match="layout hash does not match"):
        export_rl_checkpoint(
            checkpoint,
            tmp_path / "out",
            model=str(model_path),
            model_revision=MODEL_REVISION,
            rank=2,
        )

    _write_checkpoint(checkpoint, values[:-1], layout_hash)
    with pytest.raises(ValueError, match="values, expected"):
        export_rl_checkpoint(
            checkpoint,
            tmp_path / "out",
            model=str(model_path),
            model_revision=MODEL_REVISION,
            rank=2,
        )
