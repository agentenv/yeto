"""S25 N7.2: FSDP2 full-parameter plan; one step on 2 gloo ranks on CPU."""

from __future__ import annotations

import json
import math
import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from yeto.rl.fullparam.fsdp2 import (
    FullParamConfig,
    FullParamConfigError,
    check_ulysses,
    decoder_layers,
    init_distributed,
    shard_model,
    train_step,
    verl_overrides,
)

transformers = pytest.importorskip("transformers")


def tiny_llama_config() -> dict:
    return {"model_type": "llama", "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
            "max_position_embeddings": 64}


def tiny_glm5_next_config() -> dict:
    """4 layers: 3 KDA + 1 DSA (with mHC), layers 2-3 MoE; same as glm_rl/model/testing.py."""
    return {
        "model_type": "glm5_next",
        "text_config": {
            "model_type": "glm5_next_text",
            "vocab_size": 128, "hidden_size": 64, "intermediate_size": 96, "moe_intermediate_size": 32,
            "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 4,
            "n_routed_experts": 4, "num_experts_per_tok": 2, "n_shared_experts": 1,
            "kv_lora_rank": 16, "q_lora_rank": 24, "qk_nope_head_dim": 16, "qk_rope_head_dim": 0, "v_head_dim": 16,
            "index_topk": 8, "index_head_dim": 8, "index_n_heads": 2, "index_kpool": 4,
            "linear_attn_config": {"num_heads": 4, "head_dim": 8, "short_conv_kernel_size": 4},
            "mlp_layer_types": ["dense", "dense", "sparse", "sparse"],
            "layer_types": ["linear_attention"] * 3 + ["deepseek_sparse_attention"],
            "max_position_embeddings": 4096, "pad_token_id": 0,
        },
        "vision_config": {"depth": 1, "hidden_size": 32, "num_heads": 2, "intermediate_size": 32,
                          "out_hidden_size": 64, "projection_intermediate_size": 32},
    }  # fmt: skip


def has_model_type(name: str) -> bool:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    return name in CONFIG_MAPPING


def build(config: dict) -> torch.nn.Module:
    hf = transformers.AutoConfig.for_model(**config)
    torch.manual_seed(0)
    try:
        return transformers.AutoModelForCausalLM.from_config(hf)
    except ValueError:
        return transformers.AutoModelForImageTextToText.from_config(hf)


def _worker(rank: int, world: int, init_file: str, config: dict, offload: str, out: str) -> None:
    torch.set_num_threads(1)
    init_distributed("cpu", init_method=f"file://{init_file}", rank=rank, world_size=world)
    try:
        torch.manual_seed(0)
        model = build(config)
        cfg = FullParamConfig(offload=offload, param_dtype=torch.float32)  # CPU: no BF16 matmul speed needed
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        shard_model(model, cfg)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
        vocab = model.config.get_text_config().vocab_size
        g = torch.Generator().manual_seed(1 + rank)
        ids = torch.randint(1, vocab, (1, 32), generator=g)
        res = train_step(model, optim, ids)
        unchanged = []
        for n, p in model.named_parameters():
            full = p.full_tensor() if hasattr(p, "full_tensor") else p
            if torch.equal(full.detach().cpu(), before[n]):
                unchanged.append(n)
        if rank == 0:
            with open(out, "w") as fh:
                json.dump({"loss": res.loss, "grad_norm": res.grad_norm, "unchanged": unchanged}, fh)
    finally:
        dist.destroy_process_group()


def run_two_ranks(config: dict, offload: str = "none") -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")  # spawned ranks inherit it; keeps the thread count low
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "result.json")
        mp.start_processes(_worker, args=(2, os.path.join(tmp, "init"), config, offload, out),
                           nprocs=2, join=True, start_method="spawn")
        with open(out) as fh:
            return json.load(fh)


def test_one_step_two_gloo_ranks_tiny_llama():
    res = run_two_ranks(tiny_llama_config())
    assert math.isfinite(res["loss"]) and math.isfinite(res["grad_norm"]) and res["grad_norm"] > 0
    assert res["unchanged"] == []  # full parameter: every tensor moved


def test_full_offload_selects_cpu_offload_policy(monkeypatch):
    """CPU cannot run CPUOffloadPolicy (no accelerator); check the policy each module gets."""
    import torch.distributed.fsdp as fsdp

    seen = []
    monkeypatch.setattr(fsdp, "fully_shard", lambda module, **kw: seen.append((module, kw)))
    model = build(tiny_llama_config())
    shard_model(model, FullParamConfig(offload="full", gradient_checkpointing=False))
    assert len(seen) == 3 and seen[-1][0] is model  # 2 layers, then the root
    assert all(isinstance(kw["offload_policy"], fsdp.CPUOffloadPolicy) for _, kw in seen)


@pytest.mark.skipif(not has_model_type("glm5_next"), reason="transformers without glm5_next")
def test_one_step_two_gloo_ranks_tiny_glm5_next():
    res = run_two_ranks(tiny_glm5_next_config())
    assert math.isfinite(res["loss"]) and math.isfinite(res["grad_norm"]) and res["grad_norm"] > 0
    # Not trained: the DSA indexer (no gradient on the HF path, N7.0 T5; seen here on CPU)
    # and the vision tower (text-only input). Every other tensor moved.
    assert res["unchanged"], "expected the indexer and vision tower to stay put"
    assert all(".indexer." in n or n.startswith("model.visual.") for n in res["unchanged"])


def test_optimizer_offload_is_refused():
    with pytest.raises(FullParamConfigError, match="not implemented"):
        FullParamConfig(offload="optimizer")


def test_ulysses_refused_without_flash_attention_path():
    model = build(tiny_llama_config())
    model._supports_flash_attn = False
    with pytest.raises(FullParamConfigError, match="flash-attention"):
        check_ulysses(model, FullParamConfig(ulysses=2))
    model._supports_flash_attn = True
    check_ulysses(model, FullParamConfig(ulysses=2))
    with pytest.raises(FullParamConfigError, match="does not divide"):
        check_ulysses(model, FullParamConfig(ulysses=4))  # 2 KV heads


def test_decoder_layers_follow_no_split_modules():
    model = build(tiny_llama_config())
    assert len(decoder_layers(model)) == 2


def test_verl_overrides_full_parameter():
    ov = verl_overrides(FullParamConfig(offload="full", fsdp_size=8))
    assert "actor_rollout_ref.model.lora_rank=0" in ov
    assert "actor_rollout_ref.actor.strategy=fsdp2" in ov
    assert "actor_rollout_ref.actor.fsdp_config.offload_policy=true" in ov
    assert "actor_rollout_ref.actor.fsdp_config.fsdp_size=8" in ov
    assert "actor_rollout_ref.actor.fsdp_config.optimizer_offload=false" in ov
    assert "actor_rollout_ref.actor.fsdp_config.fsdp_size=-1" in verl_overrides(FullParamConfig())


def test_backends_follow_the_device_family():
    from yeto.rl.fullparam.fsdp2 import BACKENDS

    assert BACKENDS == {"cpu": "gloo", "cuda": "nccl", "ppu": "nccl", "npu": "hccl"}


def test_init_distributed_passes_the_family_backend_and_card(monkeypatch):
    import yeto.rl.fullparam.fsdp2 as fsdp2

    calls = []
    monkeypatch.setattr(fsdp2.accel, "bind_family_device", lambda family: torch.device("cuda", 1))
    monkeypatch.setattr(dist, "init_process_group", lambda **kw: calls.append(kw))
    device = fsdp2.init_distributed("ppu", rank=0, world_size=1)
    assert device == torch.device("cuda", 1)
    assert calls == [{"backend": "nccl", "device_id": torch.device("cuda", 1), "rank": 0, "world_size": 1}]
