from __future__ import annotations

import json
import struct
from types import SimpleNamespace

import pytest
import torch

from yeto.export import CKPT_MAGIC
from yeto.protocol import layout_fingerprint
from yeto.rl import export as rl_export
from yeto.rl.core import (
    build_rl_fragment_layout,
    canonical_layout_hash,
    canonical_lora_config_hash,
    canonical_state,
    flat_tensor,
    policy_tensor_hash,
    tensors_from_flat,
)
from yeto.rl.export import derive_peft_lora_specs, export_rl_checkpoint
from yeto.rl.deepseek_v4_expert_clone import ExpertCloneContract, NUM_LAYERS
from yeto.tensor_io import fragment_flat


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


def test_rl_model_factory_preserves_the_checkpoint_architecture(monkeypatch):
    transformers = pytest.importorskip("transformers")

    class DeclaredConditionalGeneration:
        @classmethod
        def from_config(cls, config, **kwargs):
            return cls()

    monkeypatch.setattr(
        transformers,
        "DeclaredConditionalGeneration",
        DeclaredConditionalGeneration,
        raising=False,
    )

    config = type(
        "Config",
        (),
        {"architectures": ["DeclaredConditionalGeneration"]},
    )()
    assert rl_export._rl_model_factory(config) is DeclaredConditionalGeneration

    config.architectures = None
    assert rl_export._rl_model_factory(config) is transformers.AutoModelForCausalLM


def test_declared_rl_architecture_does_not_receive_auto_factory_kwargs(monkeypatch):
    transformers = pytest.importorskip("transformers")

    class DeclaredConditionalGeneration:
        @classmethod
        def _from_config(cls, config):
            return cls()

    monkeypatch.setattr(
        transformers,
        "DeclaredConditionalGeneration",
        DeclaredConditionalGeneration,
        raising=False,
    )
    config = type(
        "Config",
        (),
        {"architectures": ["DeclaredConditionalGeneration"]},
    )()

    assert isinstance(
        rl_export._rl_model_from_config(config, trust_remote_code=True),
        DeclaredConditionalGeneration,
    )


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


def _expanded_deepseek_v4_config():
    contract = ExpertCloneContract(
        tuple(tuple(range(32)) for _ in range(NUM_LAYERS)),
        "a" * 64,
        "b" * 64,
    )
    return SimpleNamespace(
        model_type="deepseek_v4",
        architectures=["DeepseekV4ForCausalLM"],
        hidden_size=4096,
        moe_intermediate_size=2048,
        num_hidden_layers=43,
        n_routed_experts=288,
        num_experts_per_tok=6,
        num_nextn_predict_layers=0,
        yeto_routed_expert_clone=contract.config_value(),
    )


def test_e288_fused_expert_specs_are_complete_and_have_canonical_shapes():
    specs = rl_export._deepseek_v4_clone_expert_lora_specs(
        _expanded_deepseek_v4_config(),
        rank=8,
    )

    assert specs == tuple(sorted(specs))
    assert len(specs) == 43 * 288 * 3 * 2 == 74_304
    assert sum(spec.numel for spec in specs) == 1_826_095_104
    by_name = {spec.name: spec for spec in specs}
    prefix = "base_model.model.model.layers.42.mlp.experts.287"
    assert by_name[f"{prefix}.gate_proj.lora_A.weight"].shape == (8, 4096)
    assert by_name[f"{prefix}.gate_proj.lora_B.weight"].shape == (2048, 8)
    assert by_name[f"{prefix}.up_proj.lora_A.weight"].shape == (8, 4096)
    assert by_name[f"{prefix}.up_proj.lora_B.weight"].shape == (2048, 8)
    assert by_name[f"{prefix}.down_proj.lora_A.weight"].shape == (8, 2048)
    assert by_name[f"{prefix}.down_proj.lora_B.weight"].shape == (4096, 8)


def test_e288_fused_expert_specs_fail_closed_without_the_clone_contract():
    config = _expanded_deepseek_v4_config()
    config.yeto_routed_expert_clone = None
    config.n_routed_experts = 256

    with pytest.raises(ValueError, match="expanded E288 clone contract"):
        rl_export._deepseek_v4_clone_expert_lora_specs(config, rank=8)


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


def _write_fragment_checkpoint(path, fragments, layout_hash, global_step):
    body = bytearray(struct.pack("<IQI", CKPT_MAGIC, global_step, len(fragments)))
    for version, values in fragments:
        body += struct.pack("<QQ", version, values.numel())
        body += values.float().numpy().tobytes()
        body += torch.zeros_like(values).float().numpy().tobytes()
    body += struct.pack("<I", 0)
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


def test_decoupled_checkpoint_exports_all_fragments_and_provenance(tmp_path):
    model_path, _ = _model(tmp_path)
    specs = derive_peft_lora_specs(
        str(model_path), None, rank=2, targets="all-linear"
    )
    layout = build_rl_fragment_layout(specs, 2)
    expected = tensors_from_flat(
        torch.arange(sum(spec.numel for spec in specs), dtype=torch.float32),
        specs,
    )
    checkpoint = tmp_path / "decoupled.ckpt"
    _write_fragment_checkpoint(
        checkpoint,
        [
            (version, fragment_flat(fragment, expected))
            for version, fragment in zip((3, 4), layout.fragments)
        ],
        layout_fingerprint(layout).hex(),
        global_step=4,
    )

    output = tmp_path / "adapter"
    state = export_rl_checkpoint(
        checkpoint,
        output,
        model=str(model_path),
        model_revision=MODEL_REVISION,
        rank=2,
        sync_preset="decoupled",
        fragments=2,
        pipeline=2,
        local_horizon=2,
    )

    assert all(
        torch.equal(state.tensors[name], expected[name]) for name in expected
    )
    provenance = json.loads((output / "yeto_rl_provenance.json").read_text())
    assert provenance["sync_preset"] == "decoupled"
    assert provenance["fragments"] == 2
    assert provenance["pipeline"] == 2
    assert provenance["local_horizon"] == 2
    assert provenance["total_sweeps"] == 2
    assert provenance["total_fragment_steps"] == 4
    assert provenance["final_fragment_versions"] == [3, 4]
    assert provenance["policy_hash"] == policy_tensor_hash(state)
    assert provenance["sync_layout_fingerprint"] == layout_fingerprint(layout).hex()
    assert provenance["checkpoint_sha256"]


def test_benchmark_consolidation_accepts_only_a_rotated_terminal_sweep(tmp_path):
    model_path, _ = _model(tmp_path)
    specs = derive_peft_lora_specs(
        str(model_path), None, rank=2, targets="all-linear"
    )
    layout = build_rl_fragment_layout(specs, 2)
    expected = tensors_from_flat(
        torch.arange(sum(spec.numel for spec in specs), dtype=torch.float32),
        specs,
    )
    checkpoint = tmp_path / "budget.ckpt"
    _write_fragment_checkpoint(
        checkpoint,
        [
            (version, fragment_flat(fragment, expected))
            for version, fragment in zip((7, 6), layout.fragments)
        ],
        layout_fingerprint(layout).hex(),
        global_step=7,
    )

    with pytest.raises(ValueError, match="complete fragment sweep"):
        export_rl_checkpoint(
            checkpoint,
            tmp_path / "production",
            model=str(model_path),
            model_revision=MODEL_REVISION,
            rank=2,
            sync_preset="decoupled",
            fragments=2,
            pipeline=2,
            local_horizon=2,
        )

    output = tmp_path / "benchmark"
    state = export_rl_checkpoint(
        checkpoint,
        output,
        model=str(model_path),
        model_revision=MODEL_REVISION,
        rank=2,
        sync_preset="decoupled",
        fragments=2,
        pipeline=2,
        local_horizon=2,
        benchmark_learner_budget_steps=6,
    )

    assert state.policy_version == 7
    assert all(torch.equal(state.tensors[name], expected[name]) for name in expected)
    provenance = json.loads((output / "yeto_rl_provenance.json").read_text())
    assert provenance["benchmark_learner_budget_steps"] == 6
    assert provenance["final_fragment_versions"] == [7, 6]

    _write_fragment_checkpoint(
        checkpoint,
        [
            (version, fragment_flat(fragment, expected))
            for version, fragment in zip((6, 7), layout.fragments)
        ],
        layout_fingerprint(layout).hex(),
        global_step=7,
    )
    with pytest.raises(ValueError, match="fragment versions"):
        export_rl_checkpoint(
            checkpoint,
            tmp_path / "invalid",
            model=str(model_path),
            model_revision=MODEL_REVISION,
            rank=2,
            sync_preset="decoupled",
            fragments=2,
            pipeline=2,
            local_horizon=2,
            benchmark_learner_budget_steps=6,
        )


def test_decoupled_export_cli_reports_an_outer_fragment_step(monkeypatch, capsys):
    monkeypatch.setattr(
        rl_export,
        "export_rl_checkpoint",
        lambda *_args, **_kwargs: type("State", (), {"policy_version": 8})(),
    )

    rl_export.main(
        [
            "--checkpoint",
            "state.ckpt",
            "--model",
            "org/model",
            "--model-revision",
            MODEL_REVISION,
            "--lora-r",
            "2",
            "--sync-preset",
            "decoupled",
            "--fragments",
            "2",
            "--pipeline",
            "2",
            "--local-horizon",
            "2",
            "--output-dir",
            "adapter",
        ]
    )

    assert "outer fragment step 8" in capsys.readouterr().out
