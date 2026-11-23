"""verl builds its PEFT model on the modules the island's canonical specs name.

Before ``lora_targets``, ``fsdp_island_run`` never set ``VerlRunConfig.target_modules``,
so verl always adapted ``all-linear`` while the island derived its specs from
``--lora-targets`` (the launcher's default ``auto`` is attention-only on a MoE model):
the first export failed the spec check. CPU only (models on the meta device).
"""
from __future__ import annotations

import pytest

from yeto.rl.adapters.verl import config as vconf
from yeto.rl.adapters.verl import lora_targets
from yeto.rl.export import derive_peft_lora_specs

transformers = pytest.importorskip("transformers")


def _save(tmp_path, config, name):
    path = tmp_path / name
    config.save_pretrained(path)
    return path


@pytest.fixture
def moe_model(tmp_path):
    return _save(tmp_path, transformers.Qwen3MoeConfig(
        vocab_size=32, hidden_size=8, intermediate_size=16, moe_intermediate_size=8, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4, num_experts=4, num_experts_per_tok=2,
        decoder_sparse_step=1, mlp_only_layers=[]), "qwen3-moe")


@pytest.fixture
def dense_model(tmp_path):
    return _save(tmp_path, transformers.LlamaConfig(
        vocab_size=32, hidden_size=8, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2), "llama")


def _verl_specs(path, choice):
    """The LoRA tensors of the PEFT model verl builds from its target_modules override."""
    value = lora_targets.island_target_modules(choice, str(path))
    return value, derive_peft_lora_specs(str(path), None, rank=2, targets=lora_targets.target_list(value))


@pytest.mark.parametrize("choice", ["auto", "attention"])
def test_a_moe_model_gets_the_attention_modules_the_island_specs_name(moe_model, choice):
    value, verl = _verl_specs(moe_model, choice)
    assert value.startswith("[") and "q_proj" in value and "experts" not in value
    island = derive_peft_lora_specs(str(moe_model), None, rank=2, targets=choice)
    assert verl == island and island
    assert all("self_attn" in spec.name for spec in island)


def test_all_linear_on_a_moe_model_stays_all_linear(moe_model):
    # verl and the island hand PEFT the same "all-linear"
    assert lora_targets.island_target_modules("all-linear", str(moe_model)) == "all-linear"


@pytest.mark.parametrize("choice, expected", [("auto", "all-linear"), ("all-linear", "all-linear")])
def test_a_dense_model_keeps_all_linear(dense_model, choice, expected):
    assert lora_targets.island_target_modules(choice, str(dense_model)) == expected


def test_attention_on_a_dense_model_matches_the_island_specs(dense_model):
    value, verl = _verl_specs(dense_model, "attention")
    assert verl == derive_peft_lora_specs(str(dense_model), None, rank=2, targets="attention")
    assert all("self_attn" in spec.name for spec in verl) and value != "all-linear"


def test_the_override_carries_the_list():
    run = vconf.VerlRunConfig(model_path="m", train_file="t", val_file="v", out_dir="o", lora_rank=2,
                              lora_alpha=2, target_modules="[q_proj,k_proj]")
    assert "actor_rollout_ref.model.target_modules=[q_proj,k_proj]" in vconf.build_overrides(run)


def test_unmapped_choices_are_refused(dense_model, monkeypatch):
    with pytest.raises(vconf.VerlConfigError, match="attention-routed-experts"):
        lora_targets.island_target_modules("attention-routed-experts", str(dense_model))
    import yeto.rl.export as export

    # a regex that is not a plain module-name alternation has no list form
    monkeypatch.setattr(export, "target_modules", lambda choice, config: r"model\.layers\.0\..*")
    with pytest.raises(vconf.VerlConfigError, match="no verl target_modules list form"):
        lora_targets.verl_target_modules("attention", None)
