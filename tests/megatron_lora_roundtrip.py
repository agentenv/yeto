"""Real Megatron-Core + Megatron-Bridge canonical LoRA round trip on CPU.

Run by ``tests/test_rl_lora_parallel.py`` in a separate interpreter that has
Megatron-Core and Megatron-Bridge (``YETO_MEGATRON_PYTHON``,
``YETO_MEGATRON_LM_PATH``, ``YETO_MEGATRON_BRIDGE_PATH``); not collected by
pytest. Takes ``tp pp ep dp out.json [flags]`` and spawns ``tp * pp * dp`` gloo
ranks (``ep`` 0 builds a dense Qwen3-like model, otherwise a Qwen3-MoE-like
one; Megatron needs ``ep * tp`` to divide ``tp * dp``). With ``pp`` 2 the
stages are uneven (1 + 2 layers), so they hold different adapter counts. It
injects Megatron-Bridge ``CanonicalLoRA`` as Miles does
(``share_expert_adapters=False``) and runs the Yeto state plugin
(``yeto.rl.adapters.miles.state_plugin``) export and apply against Bridge's
Qwen3 / Qwen3-MoE mapping:

1. the export (rank 0) carries every adapter under its PEFT name; the caller
   compares names and shapes with a single-rank run;
2. apply of a random FP32 policy (values bf16 cannot hold), then export,
   gives back exactly that policy: the FP32 masters got the canonical values,
   not a bf16-rounded copy;
3. after apply every bf16 adapter parameter equals its master cast to bf16.

Prints one JSON line ``{"ok": true, ...}`` from rank 0.
"""

from __future__ import annotations

import contextlib
import importlib.abc
import importlib.machinery
import json
import os
import sys
import types
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
for _path in (os.environ.get("YETO_MEGATRON_LM_PATH"), os.environ.get("YETO_MEGATRON_BRIDGE_PATH"), str(REPO),
              str(REPO / "tests")):
    if _path and _path not in sys.path:
        sys.path.insert(0, _path)

from megatron_bridge_roundtrip import _StubFinder  # noqa: E402  (kernel stand-ins for Bridge)


class _TritonStub(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Newer Megatron-Core imports triton kernels at module level; the CPU run
    never calls them, so a permissive stand-in is enough."""

    def find_spec(self, name, path, target=None):
        if name.split(".")[0] == "triton":
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        module = mock.MagicMock()
        module.__path__, module.__spec__, module.__name__, module.__version__ = [], spec, spec.name, "3.4.0"
        return module

    def exec_module(self, module):
        return None


HF = dict(vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=3, num_attention_heads=4,
          num_key_value_heads=2, head_dim=16, tie_word_embeddings=False)
NUM_EXPERTS = 4
MOE_FFN = 32
RANK = 4
TARGETS = ["linear_q", "linear_k", "linear_v", "linear_proj", "linear_fc1_gate", "linear_fc1_up", "linear_fc2"]


def _grouped_linears():
    """CPU stand-ins for Transformer Engine's column/row grouped linears.

    ``TEGroupedMLP`` (Miles' MoE experts, ``--moe-grouped-gemm``) builds its fc1/fc2
    from TE's grouped linear, which has no CPU build. Bridge only reads its
    ``num_gemms``, ``in_features``/``out_features``, ``config`` and the per-expert
    ``weight<i>`` parameters when it injects and converts the grouped-expert LoRA;
    nothing here runs a forward pass. Expert tensor parallelism is 1.
    """

    import torch

    class GroupedLinear(torch.nn.Module):
        def __init__(self, num_gemms, input_size, output_size, *, config, **_kwargs):
            super().__init__()
            self.num_gemms, self.config = num_gemms, config
            self.in_features, self.out_features = input_size, output_size
            for index in range(num_gemms):
                weight = torch.nn.Parameter(torch.randn(output_size, input_size).to(config.params_dtype))
                self.register_parameter(f"weight{index}", weight)

    class ColumnGroupedLinear(GroupedLinear):
        pass

    class RowGroupedLinear(GroupedLinear):
        pass

    return ColumnGroupedLinear, RowGroupedLinear


def _model(tp: int, pp: int, ep: int, *, grouped: bool, glm: bool):
    import functools

    import torch
    from megatron.core import parallel_state as mpu
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec, get_gpt_layer_local_spec
    from megatron.core.transformer import TransformerConfig
    from megatron.core.transformer.moe.experts import GroupedMLPSubmodules, TEGroupedMLP

    moe = ep > 0
    config = TransformerConfig(
        num_layers=HF["num_hidden_layers"], hidden_size=HF["hidden_size"],
        num_attention_heads=HF["num_attention_heads"], num_query_groups=HF["num_key_value_heads"],
        ffn_hidden_size=HF["intermediate_size"], kv_channels=HF["head_dim"], use_cpu_initialization=True,
        gated_linear_unit=True, add_bias_linear=False, qk_layernorm=not glm, normalization="RMSNorm",
        params_dtype=torch.bfloat16, bf16=True, tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp,
        pipeline_dtype=torch.bfloat16 if pp > 1 else None,
        # uneven stages (1 + 2 layers): the stages hold different adapter counts
        **(dict(num_layers_in_first_pipeline_stage=1) if pp == 2 else {}),
        **(dict(num_moe_experts=NUM_EXPERTS, moe_ffn_hidden_size=MOE_FFN, moe_router_topk=2,
                expert_model_parallel_size=ep, expert_tensor_parallel_size=1,
                moe_token_dispatcher_type="allgather")
           if moe else {}),
        # GLM-4.5 / DeepSeek-V3 layout: a dense first layer, then MoE layers with a shared expert
        **(dict(moe_layer_freq=[0] + [1] * (HF["num_hidden_layers"] - 1),
                moe_shared_expert_intermediate_size=MOE_FFN) if glm else {}))
    config.share_embeddings_and_output_weights = False
    if glm:  # one spec per layer, dense or MoE by moe_layer_freq
        spec = get_gpt_decoder_block_spec(config, use_transformer_engine=False, normalization="RMSNorm")
        layer_specs = spec.layer_specs
    else:
        spec = get_gpt_layer_local_spec(num_experts=NUM_EXPERTS if moe else None, moe_grouped_gemm=False,
                                        qk_layernorm=True, normalization="RMSNorm")
        layer_specs = [spec]
    if moe and grouped:
        column, row = _grouped_linears()
        for layer_spec in layer_specs:
            mlp = layer_spec.submodules.mlp  # MoE: partial(MoELayer, submodules=MoESubmodules(...))
            if isinstance(mlp, functools.partial) and hasattr(mlp.keywords.get("submodules"), "experts"):
                mlp.keywords["submodules"].experts = functools.partial(
                    TEGroupedMLP, submodules=GroupedMLPSubmodules(linear_fc1=column, linear_fc2=row))
    torch.manual_seed(0)
    return GPTModel(config=config, transformer_layer_spec=spec,
                    vocab_size=HF["vocab_size"], max_sequence_length=32,
                    share_embeddings_and_output_weights=False, position_embedding_type="rope",
                    pre_process=mpu.is_pipeline_first_stage(), post_process=mpu.is_pipeline_last_stage())


def _bridge_sees_grouped_linears(model) -> None:
    """Point Bridge's TE class checks at the stand-ins (and nothing else at TE)."""

    from megatron.bridge.peft import utils as peft_utils

    grouped = {type(m) for m in model.modules() if hasattr(m, "num_gemms")}
    column = tuple(t for t in grouped if t.__name__.startswith("Column"))
    row = tuple(t for t in grouped if t.__name__.startswith("Row"))
    unused = type("NotTE", (), {})
    peft_utils.HAVE_TE = bool(grouped)
    peft_utils.TECL, peft_utils.TERL = column, row
    peft_utils.TELayerNormColumnParallelLinear = peft_utils.TELinear = unused


class _Bridge:
    """The two ``AutoBridge`` members the state plugin uses, over Bridge's own
    Qwen3 (dense) or Qwen3-MoE mapping plus the local layer spec's norms."""

    def __init__(self, moe: bool, glm: bool = False):
        from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
        from megatron.bridge.models.conversion.param_mapping import AutoMapping

        if glm:
            from megatron.bridge.models.glm.glm45_bridge import GLM45Bridge as Base
        elif moe:
            from megatron.bridge.models.qwen.qwen3_moe_bridge import Qwen3MoEBridge as Base
        else:
            from megatron.bridge.models.qwen.qwen3_bridge import Qwen3Bridge as Base

        class LocalSpecBridge(Base):
            def mapping_registry(self):
                base = super().mapping_registry()
                return MegatronMappingRegistry(
                    *base.mappings,
                    AutoMapping("decoder.layers.*.input_layernorm.weight", "model.layers.*.input_layernorm.weight"),
                    AutoMapping("decoder.layers.*.pre_mlp_layernorm.weight",
                                "model.layers.*.post_attention_layernorm.weight"))

        self._model_bridge = LocalSpecBridge()
        if glm:
            # GLM45Bridge reads the HF config (MTP layers) and, without an HF state
            # source, maps one checkpoint tensor per expert (the published layout)
            from transformers import Glm4MoeConfig

            config = Glm4MoeConfig(num_hidden_layers=HF["num_hidden_layers"], num_nextn_predict_layers=0)
            self._model_bridge.hf_config = config
            self._model_bridge.hf_pretrained = types.SimpleNamespace(config=config, state=None)

    def export_adapter_weights(self, model, cpu=True, show_progress=True, **kwargs):
        return self._model_bridge.stream_adapter_weights_megatron_to_hf(
            model, cpu=cpu, show_progress=show_progress, **kwargs)


class _Leaf:
    """A Float16 optimizer leaf: complete FP32 masters on ``main_param``."""

    def __init__(self, params):
        self.params = list(params)
        for p in self.params:
            p.main_param = p.detach().float().clone()

    def _copy_main_params_to_model_params(self):
        import torch

        with torch.no_grad():
            for p in self.params:
                p.copy_(p.main_param.view_as(p).to(p.dtype))


def _miles_package(name: str) -> types.ModuleType:
    for parent in ("miles", "miles.backends", "miles.backends.megatron_utils", "miles.utils"):
        sys.modules.setdefault(parent, types.ModuleType(parent))
    module = sys.modules[name] = types.ModuleType(name)
    parent, _, leaf = name.rpartition(".")
    setattr(sys.modules[parent], leaf, module)
    return module


def _install_replay_stub() -> None:
    """The pinned (Miles-patched) Megatron-Core registers every MoE router with
    Miles' routing replay manager; outside Miles it is a no-op."""

    _miles_package("miles.utils.replay_base").routing_replay_manager = mock.MagicMock()


def _install_miles_stubs(bridge) -> None:
    """``state_plugin`` reaches Bridge through two Miles helpers; provide them."""

    _miles_package("miles.backends.megatron_utils.hf_export")._get_hf_bridge = lambda _checkpoint: bridge
    _miles_package("miles.utils.megatron_bridge_utils").patch_megatron_model = (
        lambda _model: contextlib.nullcontext())


def _perturb(name: str, shape, scale: float):
    import hashlib

    import torch

    seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    return scale * torch.randn(tuple(shape), generator=torch.Generator().manual_seed(seed))


def _rank_main(rank: int, world: int, tp: int, pp: int, ep: int, flags: tuple[str, ...], port: int,
               out_path: str) -> None:
    grouped, glm = "sequential" not in flags, "glm45" in flags
    import torch
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    sys.meta_path.insert(0, _TritonStub())
    from megatron.core import parallel_state as mpu

    mpu.initialize_model_parallel(tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp,
                                  expert_model_parallel_size=max(ep, 1), expert_tensor_parallel_size=1)
    _install_replay_stub()
    model = _model(tp, pp, ep, grouped=grouped, glm=glm)
    sys.meta_path.insert(0, _StubFinder())  # after the model: Megatron-Core must not see a TE stand-in
    torch.cuda.current_device = lambda: None

    from megatron.bridge.peft.canonical_lora import CanonicalLoRA

    _bridge_sees_grouped_linears(model)

    lora = CanonicalLoRA(target_modules=TARGETS, dim=RANK, alpha=RANK, dropout=0.0,
                         lora_A_init_method="xavier", lora_B_init_method="xavier", share_expert_adapters=False)
    modules = lora([model], training=True)
    adapters = [p for chunk in modules for p in chunk.parameters() if p.requires_grad]
    # grouped experts: Bridge's GroupedExpertLinearAdapter, one [local experts, ...] parameter
    counted = torch.tensor([sum(1 for p in adapters if p.ndim == 3)])
    dist.all_reduce(counted)  # a dense-only pipeline stage holds none
    grouped_adapters = int(counted.item())
    assert (grouped_adapters > 0) == (ep > 0 and grouped), grouped_adapters
    leaf = _Leaf(adapters)
    # FP32 detail the bf16 copy cannot hold: proves the export reads the masters.
    for p in adapters:
        p.main_param.add_(1e-5 * _perturb(str(id(p)), p.shape, 1.0))
    bridge = _Bridge(moe=ep > 0, glm=glm)
    _install_miles_stubs(bridge)

    from yeto.rl.adapters.miles import state_plugin

    args = types.SimpleNamespace(hf_checkpoint="unused", tensor_model_parallel_size=tp,
                                 pipeline_model_parallel_size=pp, expert_model_parallel_size=max(ep, 1),
                                 global_batch_size=1, offload_train=False)
    actor = types.SimpleNamespace(args=args, model=modules, optimizer=types.SimpleNamespace(chained_optimizers=[leaf]),
                                  opt_param_scheduler=types.SimpleNamespace(num_steps=0, step=lambda increment: None),
                                  weights_backuper=None, _is_first_replica_megatron_main_rank=rank == 0)

    first = state_plugin._export_state(actor, policy_version=0)
    shapes = None
    if rank == 0:
        shapes = {name: tuple(t.shape) for name, t in first["tensors"].items()}
    box = [shapes]
    dist.broadcast_object_list(box, src=0)
    shapes = box[0]
    result = {"names": len(shapes), "grouped_adapters": grouped_adapters}

    # 2. apply a perturbed policy on every rank, export it back
    target = {name: (_perturb(name, shape, 0.05)).float() for name, shape in shapes.items()}
    summary = state_plugin._apply_state(actor, tensors=target, policy_version=1, local_step=0, optimizer="preserve")
    second = state_plugin._export_state(actor, policy_version=1)
    if rank == 0:
        back = second["tensors"]
        bad = sorted(n for n in target if not torch.equal(back[n], target[n]))
        assert not bad and set(back) == set(target), (bad[:4], sorted(set(back) ^ set(target))[:4])
    # 3. every bf16 adapter copy equals its master in bf16
    for p in adapters:
        assert torch.equal(p.detach(), p.main_param.view_as(p).to(p.dtype)), "model copy differs from its master"
    result["applied"] = summary["tensors"]
    if rank == 0:
        result.update(ok=True, ranks=world, shapes=sorted(shapes.items()))
        Path(out_path).write_text(json.dumps(result, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()


def main(argv: list[str]) -> int:
    import socket

    import torch.multiprocessing as mp

    tp, pp, ep, dp, out_path = int(argv[0]), int(argv[1]), int(argv[2]), int(argv[3]), argv[4]
    flags = tuple(argv[5:])  # "sequential" (SequentialMLP experts), "glm45" (dense first layer + shared expert)
    world = tp * pp * dp
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    mp.spawn(_rank_main, args=(world, tp, pp, ep, flags, port, out_path), nprocs=world)
    print(Path(out_path).read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
