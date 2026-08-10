from __future__ import annotations

import json
from collections import namedtuple
from types import SimpleNamespace

import torch

from yeto.rl.deepseek_v4_expert_clone import (
    CLONES_PER_LAYER,
    NUM_LAYERS,
    ORIGINAL_EXPERTS,
    TOTAL_EXPERTS,
    ExpertCloneContract,
)
from yeto.rl.sglang_deepseek_v4_clone import (
    _RUNTIME_INDEXER_Q_B_TARGET,
    _RUNTIME_Q_B_TARGET,
    _configure_moe,
    _runtime_lora_targets,
    install_on_lora_adapter,
    install_on_lora_manager,
    install_on_lora_pool,
)


TopKOutput = namedtuple("TopKOutput", "topk_weights topk_ids router_logits")


class FakeTopK(torch.nn.Module):
    def __init__(self, correction_bias):
        super().__init__()
        self.topk_config = SimpleNamespace(
            correction_bias=correction_bias,
            output_format=None,
        )

    def forward(self, hidden_states, router_logits, **_kwargs):
        ids = torch.topk(router_logits, 6, dim=1).indices.to(torch.int32)
        weights = torch.ones_like(ids, dtype=torch.float32) / 6
        return TopKOutput(weights, ids, router_logits)


def _config():
    sources = tuple(
        tuple((layer + rank) % ORIGINAL_EXPERTS for rank in range(CLONES_PER_LAYER))
        for layer in range(NUM_LAYERS)
    )
    contract = ExpertCloneContract(sources, "1" * 64, "2" * 64)
    return SimpleNamespace(
        hidden_size=16,
        n_routed_experts=TOTAL_EXPERTS,
        num_hidden_layers=NUM_LAYERS,
        num_experts_per_tok=6,
        num_nextn_predict_layers=0,
        q_lora_rank=4,
        qk_nope_head_dim=6,
        qk_rope_head_dim=2,
        head_dim=8,
        num_attention_heads=2,
        index_n_heads=3,
        index_head_dim=5,
        yeto_routed_expert_clone=contract.config_value(),
    )


def test_sglang_adapter_keeps_288_experts_but_uses_256_way_gate(monkeypatch):
    # The production branch imports this enum lazily.  Supply the minimum fake
    # module tree needed by the pure adapter test.
    import sys

    topk_module = SimpleNamespace(TopKOutputFormat=SimpleNamespace(STANDARD=7))
    monkeypatch.setitem(sys.modules, "sglang", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "sglang.srt", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "sglang.srt.layers", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.moe", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.moe.topk", topk_module)

    config = _config()
    gate = SimpleNamespace(
        weight=torch.nn.Parameter(torch.empty(TOTAL_EXPERTS, config.hidden_size)),
        e_score_correction_bias=torch.nn.Parameter(torch.empty(TOTAL_EXPERTS)),
    )
    topk = FakeTopK(gate.e_score_correction_bias)
    module = SimpleNamespace(
        layer_id=7,
        is_nextn=False,
        is_hash=False,
        num_fused_shared_experts=0,
        gate=gate,
        topk=topk,
        correction_bias=gate.e_score_correction_bias.data,
    )

    _configure_moe(module, config, require_runtime_flags=False)

    assert module.gate.weight.shape == (ORIGINAL_EXPERTS, config.hidden_size)
    assert module.gate.e_score_correction_bias.shape == (ORIGINAL_EXPERTS,)
    assert module.topk.topk_config.correction_bias.shape == (ORIGINAL_EXPERTS,)
    assert module.topk.topk_config.output_format == 7
    assert module.topk._yeto_clone_source_expert_ids.shape == (CLONES_PER_LAYER,)
    assert module.topk._yeto_clone_source_expert_ids.dtype == torch.int32
    assert module.is_hash is True
    assert module._yeto_clone_selection_sha256 == "1" * 64

    logits = torch.full((64, ORIGINAL_EXPERTS), -100.0)
    source = config.yeto_routed_expert_clone["source_experts_by_layer"][7][0]
    logits[:, source] = 100.0
    logits[:, 40:45] = 50.0
    output = module.topk(
        torch.zeros(64, config.hidden_size),
        logits,
        input_ids=torch.arange(64),
    )
    assert output.topk_ids.shape == (64, 6)
    assert torch.any(output.topk_ids == ORIGINAL_EXPERTS)
    assert torch.any(output.topk_ids == source)
    assert not torch.any(
        (output.topk_ids == ORIGINAL_EXPERTS).any(dim=1)
        & (output.topk_ids == source).any(dim=1)
    )


def test_clone_lora_scan_excludes_non_routed_mlp_linears(monkeypatch):
    import sys

    class FakeFusedMoE:
        is_shared_fused_moe = False

    layer_module = SimpleNamespace(FusedMoE=FakeFusedMoE)
    monkeypatch.setitem(sys.modules, "sglang", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "sglang.srt", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "sglang.srt.layers", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.moe", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.layers.moe.fused_moe_triton",
        SimpleNamespace(),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.layers.moe.fused_moe_triton.layer",
        layer_module,
    )
    class FakePool:
        def get_lora_B_shape(self, *args, **kwargs):
            return get_hidden_dim(*args, **kwargs)

    pool_module = SimpleNamespace(
        get_hidden_dim=lambda *_args, **_kwargs: (99, 101),
        get_normalized_target_modules=lambda targets: set(targets),
        REPLICATED_LINEAR_LORA_NAMES=[],
        LoRAMemoryPool=FakePool,
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.lora.mem_pool",
        pool_module,
    )

    dense_gate = object()
    dense_down = object()
    routed = FakeFusedMoE()
    attention = object()

    class FakeBase:
        def named_modules(self):
            yield "model.layers.0.mlp.gate_up_proj", dense_gate
            yield "model.layers.0.mlp.down_proj", dense_down
            yield "model.layers.0.mlp.experts", routed
            yield "model.layers.0.self_attn.q_b_proj", attention

    class FakeManager:
        def init_lora_shapes(self, max_lora_rank=None, target_modules=None):
            self.shape_args = (max_lora_rank, target_modules)
            self.target_modules = manager_module.get_normalized_target_modules(
                target_modules
            )

        def validate_new_adapter(self, lora_config, lora_ref):
            self.validated = (lora_config, lora_ref)

        def init_lora_modules(self):
            self.seen = tuple(self.base_model.named_modules())
            self.lora_modules = [dict(self.seen)]

    def normalize_targets(targets):
        mapping = {
            "gate_proj": "gate_up_proj",
            "up_proj": "gate_up_proj",
            "wq_b": "indexer.wq_b",
        }
        return {mapping.get(name.split(".")[-1], name) for name in targets}

    manager_module = SimpleNamespace(
        LoRAManager=FakeManager,
        LoRAMemoryPool=FakePool,
        get_normalized_target_modules=normalize_targets,
        DSA_INDEXER_LORA_NAMES=frozenset(
            {
                _RUNTIME_INDEXER_Q_B_TARGET,
                "indexer.wk",
                "indexer.weights_proj",
            }
        ),
    )
    install_on_lora_manager(manager_module)
    manager = FakeManager()
    manager.base_hf_config = _config()
    manager.base_model = FakeBase()
    manager.configs = {}
    manager.target_modules = {
        "gate_up_proj",
        "down_proj",
        "fused_qkv_a_proj_with_mqa",
        "q_b_proj",
    }

    manager.init_lora_modules()

    assert manager.seen == (
        ("model.layers.0.mlp.experts", routed),
        ("model.layers.0.self_attn.q_b_proj", attention),
    )
    assert manager._yeto_clone_excluded_lora_modules == (
        "model.layers.0.mlp.gate_up_proj",
        "model.layers.0.mlp.down_proj",
    )
    assert tuple(manager.base_model.named_modules()) == (
        ("model.layers.0.mlp.gate_up_proj", dense_gate),
        ("model.layers.0.mlp.down_proj", dense_down),
        ("model.layers.0.mlp.experts", routed),
        ("model.layers.0.self_attn.q_b_proj", attention),
    )

    manager.init_lora_shapes(
        max_lora_rank=8,
        target_modules=["q_a_proj", "q_b_proj", "gate_proj", "down_proj"],
    )
    assert manager.shape_args == (
        8,
        [
            "wq_a",
            _RUNTIME_Q_B_TARGET,
            _RUNTIME_INDEXER_Q_B_TARGET,
            "gate_proj",
            "down_proj",
        ],
    )
    assert manager._yeto_clone_runtime_attention_targets == (
        _RUNTIME_INDEXER_Q_B_TARGET,
        _RUNTIME_Q_B_TARGET,
        "wq_a",
    )
    manager.init_lora_shapes(
        max_lora_rank=8,
        target_modules={"wq_a", "wq_b", "gate_proj", "down_proj"},
    )
    assert manager.shape_args == (
        8,
        {
            "wq_a",
            _RUNTIME_Q_B_TARGET,
            _RUNTIME_INDEXER_Q_B_TARGET,
            "gate_proj",
            "down_proj",
        },
    )
    assert manager._yeto_clone_runtime_attention_targets == (
        _RUNTIME_INDEXER_Q_B_TARGET,
        _RUNTIME_Q_B_TARGET,
        "wq_a",
    )
    adapter_config = SimpleNamespace(target_modules=["q_a_proj", "q_b_proj"])
    manager.validate_new_adapter(adapter_config, "fixture")
    assert adapter_config.target_modules == [
        "wq_a",
        _RUNTIME_Q_B_TARGET,
        _RUNTIME_INDEXER_Q_B_TARGET,
    ]
    assert manager_module.get_normalized_target_modules(
        ["wq_a", _RUNTIME_Q_B_TARGET]
    ) == {"wq_a", _RUNTIME_Q_B_TARGET}
    assert manager_module.get_normalized_target_modules(
        ["indexer.wq_b"]
    ) == {"indexer.wq_b"}


def test_clone_attention_lora_names_and_buffer_geometry_are_runtime_native():
    assert _runtime_lora_targets(
        ["q_a_proj", "q_b_proj", "gate_proj"]
    ) == [
        "wq_a",
        _RUNTIME_Q_B_TARGET,
        _RUNTIME_INDEXER_Q_B_TARGET,
        "gate_proj",
    ]
    assert _runtime_lora_targets(
        ["fused_qkv_a_proj_with_mqa", "q_b_proj"]
    ) == [
        "wq_a",
        _RUNTIME_Q_B_TARGET,
        _RUNTIME_INDEXER_Q_B_TARGET,
    ]
    assert _runtime_lora_targets(
        {"wq_a", "wq_b", "gate_proj", "down_proj"}
    ) == {
        "wq_a",
        _RUNTIME_Q_B_TARGET,
        _RUNTIME_INDEXER_Q_B_TARGET,
        "gate_proj",
        "down_proj",
    }

    class FakeAdapter:
        def normalize_fused_qkv_a_proj(self, _weight_names, _weights):
            raise AssertionError("ordinary fused normalization must not run")

    adapter_module = SimpleNamespace(LoRAAdapter=FakeAdapter)
    install_on_lora_adapter(adapter_module)
    adapter = FakeAdapter()
    adapter.base_hf_config = _config()
    q_a = torch.ones(2, 3)
    q_b = torch.ones(5, 2)
    weights = {
        "model.layers.0.self_attn.q_a_proj.lora_A.weight": q_a,
        "model.layers.0.self_attn.q_b_proj.lora_B.weight": q_b,
    }
    adapter.normalize_fused_qkv_a_proj(list(weights), weights)
    assert set(weights) == {
        "model.layers.0.self_attn.wq_a.lora_A.weight",
        "model.layers.0.self_attn.wq_b.lora_B.weight",
    }
    assert weights["model.layers.0.self_attn.wq_a.lora_A.weight"] is q_a
    assert weights["model.layers.0.self_attn.wq_b.lora_B.weight"] is q_b

    def ordinary_hidden_dim(*_args, **_kwargs):
        return (99, 101)

    pool_globals = {
        "get_hidden_dim": ordinary_hidden_dim,
        "REPLICATED_LINEAR_LORA_NAMES": [],
    }
    exec(
        "def get_lora_B_shape(self, *args, **kwargs):\n"
        "    return get_hidden_dim(*args, **kwargs)\n",
        pool_globals,
    )
    fake_pool_cls = type(
        "FakeLoRAMemoryPool",
        (),
        {"get_lora_B_shape": pool_globals["get_lora_B_shape"]},
    )
    pool_module = SimpleNamespace(
        get_hidden_dim=ordinary_hidden_dim,
        get_normalized_target_modules=lambda targets: {
            "indexer.wq_b" if name.split(".")[-1] == "wq_b" else name
            for name in targets
        },
        REPLICATED_LINEAR_LORA_NAMES=pool_globals[
            "REPLICATED_LINEAR_LORA_NAMES"
        ],
        LoRAMemoryPool=fake_pool_cls,
    )
    install_on_lora_pool(pool_module)
    config = _config()
    # A worker-side compatibility update may attach a stale V2 no-PE field;
    # V4's checkpoint-native head_dim remains authoritative.
    config.qk_nope_head_dim = 1
    assert pool_module.get_hidden_dim("wq_a", config) == (16, 4)
    assert pool_module.get_hidden_dim(_RUNTIME_Q_B_TARGET, config) == (4, 16)
    assert pool_module.get_hidden_dim(
        _RUNTIME_INDEXER_Q_B_TARGET, config
    ) == (4, 15)
    assert pool_module.get_hidden_dim("other", config) == (99, 101)
    assert pool_module.REPLICATED_LINEAR_LORA_NAMES == ["wq_a"]
    assert (
        fake_pool_cls.get_lora_B_shape.__globals__["get_hidden_dim"]
        is pool_module.get_hidden_dim
    )
    assert pool_module.get_normalized_target_modules(
        [_RUNTIME_Q_B_TARGET]
    ) == {_RUNTIME_Q_B_TARGET}
