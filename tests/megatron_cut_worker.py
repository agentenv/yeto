"""Write real Megatron cut shards and a reference HF state for test_rl_eval_cut_to_hf (S27 R-5).

Run with a Python that has megatron-core and a Miles checkout on ``PYTHONPATH``
(CPU, gloo; no GPU)::

    python tests/megatron_cut_worker.py OUT --tp 2 --pp 2 --ep 1 [--moe]

Every rank builds a tiny Qwen3-like (dense or MoE) ``GPTModel`` through
Megatron's own sharded construction, then writes what the trainer's cut writes
for a full-parameter cut: the local model copy (``adapter``) and the layout
recorded by ``MilesCutBackend.param_layout`` (Miles global naming and TP
partition rules). Rank 0 also writes the reference: the whole model rebuilt
from Megatron's ``sharded_state_dict`` (global shapes and offsets), named by
Miles' torch_dist->HF tool and converted by Miles' ``megatron_to_hf``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

VOCAB, HIDDEN, FFN, MOE_FFN, LAYERS, HEADS, GROUPS, HEAD_DIM, EXPERTS, TOPK = 64, 32, 48, 16, 4, 4, 2, 8, 4, 2


def _stub_sglang_bridge() -> None:
    """``miles.backends.megatron_utils.sglang`` only re-exports SGLang helpers (FP8 quantizers,
    torch reductions) that the gather/convert code paths used here never call."""
    name = "miles.backends.megatron_utils.sglang"
    module = types.ModuleType(name)

    def missing(attr: str):
        if attr.startswith("__"):
            raise AttributeError(attr)
        return lambda *a, **k: (_ for _ in ()).throw(RuntimeError(f"sglang stub {attr} called"))

    module.__getattr__ = missing
    sys.modules[name] = module


def _miles_root() -> Path:
    import miles

    return Path(miles.__file__).resolve().parents[1]


def _load_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Wrap(torch.nn.Module):
    """Stands in for DDP(Float16Module(model)): parameter names gain ``module.module.``."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.module = inner
        self.config = getattr(inner, "config", None)


def _build(tp: int, pp: int, ep: int, moe: bool):
    from megatron.core import parallel_state as mps
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.transformer import TransformerConfig

    mps.initialize_model_parallel(tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp,
                                  expert_model_parallel_size=ep)
    torch.manual_seed(1234)
    cfg = TransformerConfig(
        num_layers=LAYERS, hidden_size=HIDDEN, ffn_hidden_size=FFN, num_attention_heads=HEADS,
        num_query_groups=GROUPS, kv_channels=HEAD_DIM, gated_linear_unit=True,
        activation_func=torch.nn.functional.silu, add_bias_linear=False, normalization="RMSNorm",
        qk_layernorm=True, use_cpu_initialization=True, params_dtype=torch.float32,
        pipeline_dtype=torch.float32, tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp,
        expert_model_parallel_size=ep, num_moe_experts=EXPERTS if moe else None,
        moe_ffn_hidden_size=MOE_FFN if moe else None, moe_router_topk=TOPK)
    spec = get_gpt_layer_local_spec(num_experts=EXPERTS if moe else None, qk_layernorm=True, normalization="RMSNorm")
    model = GPTModel(config=cfg, transformer_layer_spec=spec, vocab_size=VOCAB, max_sequence_length=32,
                     pre_process=mps.is_pipeline_first_stage(), post_process=mps.is_pipeline_last_stage(),
                     share_embeddings_and_output_weights=False, position_embedding_type="rope")
    return model, mps


def _parallel_state(mps) -> SimpleNamespace:
    def group(rank, size):
        return SimpleNamespace(rank=rank, size=size)

    return SimpleNamespace(
        tp=group(mps.get_tensor_model_parallel_rank(), mps.get_tensor_model_parallel_world_size()),
        pp=group(mps.get_pipeline_model_parallel_rank(), mps.get_pipeline_model_parallel_world_size()),
        ep=group(mps.get_expert_model_parallel_rank(), mps.get_expert_model_parallel_world_size()),
        etp=group(mps.get_expert_tensor_parallel_rank(), mps.get_expert_tensor_parallel_world_size()))


def _fill(named, layout) -> None:
    """Distinct values per tensor; replicas of one slice (over DP/EP) get the same values."""
    for local, param in named:
        meta = layout["params"][local]
        slot = layout["tp_rank"] if meta["tensor_parallel"] and not meta["expert"] else 0
        seed = int(hashlib.sha256(f"{meta['global_name']}/{slot}".encode()).hexdigest()[:8], 16)
        gen = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            param.copy_(torch.randn(param.shape, generator=gen) * 0.1)


def _reference(model, moe: bool, out: Path) -> None:
    """Whole tensors from Megatron's sharded_state_dict, gathered to rank 0, then Miles' tool + converter."""
    from megatron.core.dist_checkpointing.dict_utils import nested_values
    from megatron.core.dist_checkpointing.mapping import ShardedTensor, apply_factories

    sd = model.sharded_state_dict(prefix="")
    apply_factories(sd)
    pieces = []
    for v in nested_values(sd):
        if isinstance(v, ShardedTensor) and all(r == 0 for r in v.replica_id) and v.flattened_range is None:
            pieces.append((v.key, tuple(v.global_shape), tuple(v.global_offset), v.prepend_axis_num,
                           v.data.detach().clone()))
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, pieces)
    if dist.get_rank() != 0:
        return
    full: dict[str, torch.Tensor] = {}
    for key, shape, offset, prepend, data in (p for rank in gathered for p in rank):
        target = full.setdefault(key, torch.full(shape, float("nan")))
        view = data.reshape((1,) * prepend + tuple(data.shape))
        index = tuple(slice(o, o + s) for o, s in zip(offset, view.shape))
        target[index] = view
    for key, t in full.items():
        assert not torch.isnan(t).any(), f"reference {key} has unfilled slices"

    root = _miles_root()
    tool = _load_file(root / "tools" / "convert_torch_dist_to_hf.py", "_miles_torch_dist_to_hf")
    from miles.backends.megatron_utils.megatron_to_hf import _convert_to_hf_core, remove_padding

    args = SimpleNamespace(num_layers=LAYERS, num_experts=EXPERTS if moe else None, hidden_size=HIDDEN,
                           num_attention_heads=HEADS, num_query_groups=GROUPS, kv_channels=HEAD_DIM,
                           vocab_size=VOCAB)
    family = "qwen3moe" if moe else "qwen3"
    hf = {}
    for name, param in tool.get_named_params(args, full):
        for hf_name, tensor in _convert_to_hf_core(args, family, name, remove_padding(name, param, VOCAB)):
            hf[hf_name] = tensor.contiguous().clone()
    torch.save(hf, out / "reference_hf.pt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--ep", type=int, default=1)
    ap.add_argument("--moe", action="store_true")
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--rank", type=int, required=True)
    a = ap.parse_args()
    out = Path(a.out)
    dist.init_process_group("gloo", rank=a.rank, world_size=a.world)
    _stub_sglang_bridge()
    model, mps = _build(a.tp, a.pp, a.ep, a.moe)

    import miles.backends.megatron_utils.named_weights as named_weights
    import miles.backends.megatron_utils.update_weight.hf_weight_iterator_direct as direct
    import miles.backends.training_utils.parallel as miles_parallel

    from yeto.rl.adapters.miles.cut_plugin import MilesCutBackend

    state = _parallel_state(mps)
    for module in (named_weights, direct, miles_parallel):  # Miles reads ranks through this accessor
        module.get_parallel_state = lambda: state
    backend = MilesCutBackend()
    wrapped = [_Wrap(_Wrap(model))]
    named = [(n, p) for n, p in backend.named_parameters(wrapped) if p.requires_grad]
    actor = SimpleNamespace(args=SimpleNamespace(num_experts=EXPERTS if a.moe else None, swiglu=True), model=wrapped)
    layout = backend.param_layout(actor, named)
    _fill(named, layout)

    coord = {"tp": state.tp.rank, "pp": state.pp.rank, "dp": mps.get_data_parallel_rank(), "global_rank": a.rank}
    if a.ep > 1:
        coord["ep"] = state.ep.rank
    shard = {"schema": "yeto.cut_shard/v1", "coord": coord, "param_layout": layout,
             "adapter": {n: p.detach().clone() for n, p in named}}
    name = f"trainer_tp{coord['tp']}_pp{coord['pp']}_dp{coord['dp']}.pt"
    torch.save(shard, out / name)
    (out / f"coord-{a.rank}.json").write_text(json.dumps({"path": name, "coord": coord}))
    _reference(model, a.moe, out)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    main()
