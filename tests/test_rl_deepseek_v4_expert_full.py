from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from yeto.rl.deepseek_v4_expert_clone import (
    CLONES_PER_LAYER,
    NUM_LAYERS,
    ORIGINAL_EXPERTS,
    TOTAL_EXPERTS,
    ExpertCloneContract,
)
from yeto.rl.deepseek_v4_expert_full import (
    configure_clone_expert_full,
    expert_full_specs,
    wrap_attention_lora_with_expert_full,
)


def _contract() -> ExpertCloneContract:
    return ExpertCloneContract(
        tuple(
            tuple((layer + index) % ORIGINAL_EXPERTS for index in range(CLONES_PER_LAYER))
            for layer in range(NUM_LAYERS)
        ),
        "a" * 64,
        "b" * 64,
    )


def _config():
    contract = _contract()
    return SimpleNamespace(
        model_type="deepseek_v4",
        architectures=["DeepseekV4ForCausalLM"],
        hidden_size=4096,
        moe_intermediate_size=2048,
        n_routed_experts=TOTAL_EXPERTS,
        num_hidden_layers=NUM_LAYERS,
        num_experts_per_tok=6,
        num_nextn_predict_layers=0,
        yeto_routed_expert_clone=contract.config_value(),
    )


class _PackedExperts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_fc1 = torch.nn.Linear(3, 4, bias=False)
        self.linear_fc2 = torch.nn.Linear(2, 3, bias=False)
        self.linear_fc1.weight = torch.nn.Parameter(torch.ones(36, 4, 3))
        self.linear_fc2.weight = torch.nn.Parameter(torch.ones(36, 3, 2))


class _Mlp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = _PackedExperts()


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _Mlp()


class _Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(_Layer() for _ in range(NUM_LAYERS))


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = _Decoder()
        self.attention_lora = torch.nn.Parameter(torch.ones(1))


def test_expert_full_specs_bind_only_clone_ids_and_the_attested_map():
    specs = expert_full_specs(
        _config(),
        expert_count=16,
        expected_selection_sha256="a" * 64,
        expected_selection_contract_sha256="b" * 64,
    )

    assert len(specs) == NUM_LAYERS * 16 * 3
    assert sum(spec.numel for spec in specs) == 17_314_086_912
    assert all(
        f".experts.{expert}." not in spec.name
        for expert in range(ORIGINAL_EXPERTS)
        for spec in specs
    )
    assert any(".experts.256.gate_proj.weight" in spec.name for spec in specs)
    assert any(".experts.271.down_proj.weight" in spec.name for spec in specs)
    assert all(".experts.272." not in spec.name for spec in specs)

    with pytest.raises(ValueError, match="selection SHA256"):
        expert_full_specs(
            _config(),
            expert_count=16,
            expected_selection_sha256="c" * 64,
            expected_selection_contract_sha256="b" * 64,
        )


class _IndividualExperts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for branch, shape in (("linear_fc1", (4, 3)), ("linear_fc2", (3, 2))):
            module = torch.nn.Module()
            for local_id in range(36):
                module.register_parameter(
                    f"weight{local_id}",
                    torch.nn.Parameter(torch.ones(shape)),
                )
            setattr(self, branch, module)


class _IndividualMlp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = _IndividualExperts()


class _IndividualLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _IndividualMlp()


class _IndividualDecoder(torch.nn.Module):
    def __init__(self, num_layers=NUM_LAYERS):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            _IndividualLayer() for _ in range(num_layers)
        )


class _IndividualModel(torch.nn.Module):
    def __init__(self, num_layers=NUM_LAYERS):
        super().__init__()
        self.decoder = _IndividualDecoder(num_layers)
        self.attention_lora = torch.nn.Parameter(torch.ones(1))


def test_individual_grouped_weights_train_exactly_sixteen_clone_experts():
    model = _IndividualModel()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.attention_lora.requires_grad_(True)

    records = configure_clone_expert_full(
        model,
        expert_count=16,
        expert_parallel_rank=7,
        expert_parallel_size=8,
    )

    assert len(records) == NUM_LAYERS * 2 * 36
    trainable = {
        record.local_expert_ids[0]
        for record in records
        if record.trainable_clone_ids
    }
    assert trainable == set(range(256, 272))
    for name, parameter in model.named_parameters():
        if ".mlp.experts.linear_fc" not in name:
            continue
        local_id = int(name.rsplit("weight", 1)[1])
        assert parameter.requires_grad == (4 <= local_id < 20)
        if parameter.requires_grad:
            assert parameter._yeto_expert_id == 252 + local_id
            assert parameter._yeto_expert_full


def test_packed_expert_weights_are_rejected_by_the_pinned_individual_layout():
    model = _Model()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.attention_lora.requires_grad_(True)

    with pytest.raises(RuntimeError, match="individual expert weights"):
        configure_clone_expert_full(
            model,
            expert_count=32,
            expert_parallel_rank=7,
            expert_parallel_size=8,
        )


def test_pipeline_stage_validates_only_its_local_expert_layers():
    model = _IndividualModel(num_layers=22)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    records = configure_clone_expert_full(
        model,
        expert_count=32,
        expert_parallel_rank=7,
        expert_parallel_size=8,
    )

    assert len(records) == 22 * 2 * 36


def test_non_owner_ep_ranks_keep_every_expert_parameter_frozen():
    model = _IndividualModel()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.attention_lora.requires_grad_(True)

    records = configure_clone_expert_full(
        model,
        expert_count=32,
        expert_parallel_rank=0,
        expert_parallel_size=8,
    )

    assert len(records) == NUM_LAYERS * 2 * 36
    assert all(not record.trainable_clone_ids for record in records)
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if ".mlp.experts.linear_fc" in name
    )
    assert model.attention_lora.requires_grad


def test_attention_lora_proxy_enables_experts_only_after_lora_freeze():
    model = _IndividualModel()

    class Inner:
        marker = "attention"

        def __call__(self, value, *args, **kwargs):
            return value

        def set_params_to_save(self, value):
            for parameter in value.parameters():
                parameter.requires_grad_(False)
            value.attention_lora.requires_grad_(True)

    proxy = wrap_attention_lora_with_expert_full(
        Inner(),
        configure_kwargs={
            "expert_count": 32,
            "expert_parallel_rank": 7,
            "expert_parallel_size": 8,
        },
    )
    transformed = proxy(model)
    proxy.set_params_to_save(transformed)

    assert proxy.marker == "attention"
    assert len(proxy.expert_records) == NUM_LAYERS * 2 * 36
    assert model.attention_lora.requires_grad
    for name, parameter in model.named_parameters():
        if ".mlp.experts.linear_fc" in name:
            local_id = int(name.rsplit("weight", 1)[1])
            assert parameter.requires_grad == (local_id >= 4)
