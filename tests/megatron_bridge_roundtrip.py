"""Real Megatron-Core + Megatron-Bridge round trip of the HF boundary on CPU (S27 G-FG).

Run by ``tests/test_s27_verl_megatron_sync.py`` in a separate interpreter that
has Megatron-Core and Megatron-Bridge (paths in ``YETO_MEGATRON_LM_PATH`` and
``YETO_MEGATRON_BRIDGE_PATH``); not collected by pytest.  Takes ``tp pp dp
out.json``, spawns ``tp * pp * dp`` gloo ranks, builds a small Qwen3-like
``GPTModel`` (local layer spec, tied embeddings) and runs
:mod:`yeto.rl.megatron_hf_boundary` against Bridge's Qwen3 mapping:

* ``tp``: tensor parallel size (column/row parallel, fused QKV and gate/up);
* ``pp``: pipeline parallel size (S27 G-FL); each stage owns part of the tasks
  and receives the rest over the PP group, and the last stage owns the tied
  embedding's second copy;
* ``dp``: data parallel size; the optimizer is a DistributedOptimizer stand-in
  whose FP32 master shards the ranks own by range (``megatron_masters``), so
  ``dp > 1`` exercises the master gather (gloo all-reduce) and range writes.

Checks, every one bit-exact on every rank:

1. the export reads the FP32 masters, not the bf16 model copy;
2. apply of every tensor, then export, gives back the target;
3. after an apply each bf16 model parameter equals its master cast to bf16;
4. a partial apply (one fused input, one replicated norm) changes exactly
   those tensors (the other QKV inputs are read back from the masters);
5. the task index covers exactly the HF model's parameters (the FSDP2 layout);
6. after a full apply the last stage's tied output layer holds the applied
   embedding (master and bf16 copy);
7. a partial apply of the embedding alone moves that copy with it.

``YETO_PP_MUTATE`` breaks one piece of the cross-stage work on purpose
(:func:`_mutate`); the run must then fail.

Prints one JSON line ``{"ok": true, ...}`` from rank 0.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import json
import os
import sys
import types
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
# Module level, so the spawned ranks (which re-import this file) see the same paths.
for _path in (os.environ.get("YETO_MEGATRON_LM_PATH"), os.environ.get("YETO_MEGATRON_BRIDGE_PATH"), str(REPO)):
    if _path and _path not in sys.path:
        sys.path.insert(0, _path)
# Kernels and quantization packages the CPU run does not have (Megatron-Bridge
# imports them at module level); stand-ins answer every attribute.
_STUBBED = ("omegaconf", "transformer_engine", "modelopt", "nvidia_resiliency_ext", "apex", "flash_attn",
            "mamba_ssm", "causal_conv1d", "nvidia_modelopt")


class _AnyModule(types.ModuleType):
    def __getattr__(self, name):
        if name == "__version__":
            return "99.0.0"
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        value = type(name, (), {"__module__": self.__name__}) if name[:1].isupper() else mock.MagicMock()
        setattr(self, name, value)
        return value


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in _STUBBED:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        module = _AnyModule(spec.name)
        module.__path__ = []
        return module

    def exec_module(self, module):
        return None


HF = dict(vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4,
          num_key_value_heads=2, head_dim=16, tie_word_embeddings=True)


def _model(tp: int, pp: int = 1):
    import torch
    from megatron.core import parallel_state as mpu
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.transformer import TransformerConfig

    config = TransformerConfig(num_layers=HF["num_hidden_layers"], hidden_size=HF["hidden_size"],
                               num_attention_heads=HF["num_attention_heads"],
                               num_query_groups=HF["num_key_value_heads"], ffn_hidden_size=HF["intermediate_size"],
                               kv_channels=HF["head_dim"], use_cpu_initialization=True, gated_linear_unit=True,
                               add_bias_linear=False, qk_layernorm=True, normalization="RMSNorm",
                               params_dtype=torch.bfloat16, bf16=True, tensor_model_parallel_size=tp,
                               pipeline_model_parallel_size=pp,
                               pipeline_dtype=torch.bfloat16 if pp > 1 else None)
    # Bridge reads the tie off the *config* (``_share_embeddings_and_output_weights``),
    # as Bridge's own GPT provider sets it; without it Bridge would map the last
    # stage's output layer to ``lm_head.weight``, which a tied HF model does not have.
    config.share_embeddings_and_output_weights = True
    torch.manual_seed(0)
    # PP > 1 and tied embeddings: Megatron copies the first stage's embedding into the
    # last stage's output layer over the embedding group, moving it to CUDA first
    # (``LanguageModule.setup_embeddings_and_output_layer``). This CPU run keeps it
    # where it is; the gloo all-reduce itself runs for real.
    cuda = torch.Tensor.cuda
    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    try:
        return GPTModel(config=config, transformer_layer_spec=get_gpt_layer_local_spec(
                            qk_layernorm=True, normalization="RMSNorm"),
                        vocab_size=HF["vocab_size"], max_sequence_length=32,
                        share_embeddings_and_output_weights=True, position_embedding_type="rope",
                        pre_process=mpu.is_pipeline_first_stage(), post_process=mpu.is_pipeline_last_stage())
    finally:
        torch.Tensor.cuda = cuda


class _Range:
    def __init__(self, start: int, end: int):
        self.start, self.end = start, end


class _DistributedLeaf:
    """The parts of Megatron's ``DistributedOptimizer`` that ``megatron_masters``
    reads: per-rank FP32 master shards of the bf16 parameters, one contiguous
    range of the flattened parameters per data-parallel rank."""

    def __init__(self, params, *, dp_rank: int, dp_size: int, group, seed: int):
        import torch

        sizes = [p.numel() for p in params]
        total = sum(sizes)
        shard = -(-total // dp_size)
        lo, hi = dp_rank * shard, min((dp_rank + 1) * shard, total)
        param_map, self.mains, offset = {}, {}, 0
        generator = torch.Generator().manual_seed(seed)
        for param, size in zip(params, sizes, strict=True):
            # the master is the bf16 value plus FP32 detail the model copy cannot hold
            detail = 1e-4 * torch.randn(size, generator=generator)
            start, end = max(lo, offset) - offset, min(hi, offset + size) - offset
            if end > start:
                param_map[param] = {"param": _Range(start, end)}
                main = (param.detach().float().reshape(-1) + detail)[start:end].clone()
                self.mains[id(param)] = main
                param.main_param, param.main_param_sharded = main, True
            else:
                param.main_param, param.main_param_sharded = None, True
            offset += size
        self.gbuf_ranges = [{(torch.bfloat16, torch.float32): [{"param_map": param_map}]}]
        self.model_param_group_index_map = {p: (0, i) for i, p in enumerate(param_map)}
        self.buffers = [types.SimpleNamespace(param_index_map={p: None for p in params})]
        self.data_parallel_group = group
        self.config = types.SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=False)

    def _get_main_param_and_optimizer_states(self, param):
        return {"param": self.mains[id(param)]}


class _Bridge:
    """``AutoBridge``'s two calls the boundary makes, over Bridge's own Qwen3
    mapping with two extra names for the local layer spec's separate norms."""

    def __init__(self):
        from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
        from megatron.bridge.models.conversion.param_mapping import AutoMapping
        from megatron.bridge.models.qwen.qwen3_bridge import Qwen3Bridge
        from transformers import Qwen3Config

        class LocalSpecQwen3Bridge(Qwen3Bridge):
            def mapping_registry(self):
                base = super().mapping_registry()
                return MegatronMappingRegistry(
                    *base.mappings,
                    AutoMapping("decoder.layers.*.input_layernorm.weight", "model.layers.*.input_layernorm.weight"),
                    AutoMapping("decoder.layers.*.pre_mlp_layernorm.weight",
                                "model.layers.*.post_attention_layernorm.weight"))

        self.config = Qwen3Config(**HF)
        self.model_bridge = LocalSpecQwen3Bridge()

    def get_conversion_tasks(self, modules):
        return self.model_bridge.build_conversion_tasks(self.config, modules)

    def export_hf_weights(self, modules, cpu=False, show_progress=True, conversion_tasks=None,
                          merge_adapter_weights=True):
        return self.model_bridge.stream_weights_megatron_to_hf(
            modules, self.config, cpu=cpu, show_progress=show_progress, conversion_tasks=conversion_tasks,
            merge_adapter_weights=merge_adapter_weights)


def _delta(name: str, shape, scale: float):
    import hashlib

    import torch

    seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    return scale * torch.randn(tuple(shape), generator=torch.Generator().manual_seed(seed))


MUTATIONS = ("", "no-tied", "stage-local-export")


def _mutate(boundary) -> None:
    """``YETO_PP_MUTATE`` breaks one piece of the cross-stage work on purpose, so the
    checks below can be shown to catch it (S27 G-FL):

    * ``no-tied``: the apply stops writing the tied embedding's second copy, so the
      last stage keeps a stale output layer (check 6 and 7 must fail);
    * ``stage-local-export``: the export hands the sink only the names of the tasks
      this rank owns, i.e. it drops what the PP broadcast delivered (check 1 must
      fail on every stage that does not own every name).
    """

    mutation = os.environ.get("YETO_PP_MUTATE", "")
    if mutation not in MUTATIONS:
        raise SystemExit(f"YETO_PP_MUTATE must be one of {MUTATIONS}, got {mutation!r}")
    if mutation == "no-tied":
        boundary.tied_output_copies = lambda modules, tasks: []
    elif mutation == "stage-local-export":
        original = boundary._export_swapped

        def stage_local(bridge, modules, swapped, indices, sink):
            mine = set()
            for index in indices:
                if swapped[index] is not None and swapped[index].param_weight is not None:
                    mine.update(boundary.task_hf_names(swapped[index]))
            return original(bridge, modules, swapped, indices,
                            lambda name, tensor: sink(name, tensor) if name in mine else None)

        boundary._export_swapped = stage_local


def _rank_main(rank: int, world: int, tp: int, pp: int, dp: int, port: int, out_path: str) -> None:
    import torch
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from megatron.core import parallel_state as mpu

    mpu.initialize_model_parallel(tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp)
    model = _model(tp, pp)
    sys.meta_path.insert(0, _StubFinder())  # after the model: Megatron-Core must not see a TE stand-in
    torch.cuda.current_device = lambda: None  # Bridge's replicated load compares devices with the CUDA one

    from yeto.rl import megatron_hf_boundary as boundary
    from yeto.rl import megatron_masters

    _mutate(boundary)

    bf16 = [p for p in model.parameters() if p.dtype == torch.bfloat16]
    leaf = _DistributedLeaf(bf16, dp_rank=mpu.get_data_parallel_rank(), dp_size=dp,
                            group=mpu.get_data_parallel_group(), seed=100 + mpu.get_tensor_model_parallel_rank())
    optimizer = types.SimpleNamespace(chained_optimizers=[leaf])
    bridge, modules = _Bridge(), [model]

    def export(indices) -> dict:
        out = {}
        boundary.export_hf(bridge, modules, optimizer, indices, lambda n, t: out.__setitem__(n, t.clone()))
        return out

    record = boundary.manifest(bridge, modules, optimizer)
    records = [None] * world
    dist.all_gather_object(records, record)
    from transformers import AutoModelForCausalLM

    with torch.device("meta"):
        hf_model = AutoModelForCausalLM.from_config(bridge.config)
    shapes = {name: tuple(p.shape) for name, p in hf_model.named_parameters()}
    index = boundary.HfTaskIndex.from_manifests(records, shapes)
    every = index.tasks_for(shapes)
    result = {"tasks": len(every), "names": len(shapes)}

    # 1. the export reads the masters (FP32 detail), not the bf16 model copy
    with torch.no_grad():
        h0 = export(every)
        model_copy = {n: t.float() for n, t in bridge.export_hf_weights(modules, cpu=False, show_progress=False,
                                                                       conversion_tasks=[
                                                                           bridge.get_conversion_tasks(modules)[i]
                                                                           for i in every],
                                                                       merge_adapter_weights=False)}
    assert set(h0) >= set(shapes), sorted(set(shapes) - set(h0))
    assert all(h0[n].dtype == torch.float32 and tuple(h0[n].shape) == shapes[n] for n in shapes)
    differs = sorted(n for n in shapes if not torch.equal(h0[n], model_copy[n]))
    assert differs, "the export read the bf16 model copy"
    result["names_read_from_masters"] = len(differs)

    # 2./3. apply every tensor, export, compare; model copy == master in bf16
    target = {n: h0[n] + _delta(n, shapes[n], 1e-3) for n in shapes}
    write, read = index.apply_plan(target)
    assert read == [], read
    with torch.no_grad():
        counts = boundary.apply_hf(bridge, modules, optimizer, target, write, read)
        h1 = export(every)
    bad = sorted(n for n in shapes if not torch.equal(h1[n], target[n]))
    assert not bad, bad
    result["apply_all"] = counts

    # 6. the tied embedding's second copy: the last pipeline stage owns its own
    # output-layer weight, which no conversion task carries (PP > 1 only). Ground
    # truth read from the model, not from the production helper, so the mutation
    # cannot hide the check. Checked before 3, so that a stale copy is named here
    # rather than by the general master/model comparison.
    tied = None
    if getattr(model, "post_process", False) and not getattr(model, "pre_process", False):
        tied = getattr(getattr(model, "output_layer", None), "weight", None)
    result["tied_copy"] = tied is not None
    embed = "model.embed_tokens.weight"

    def shard_of(whole):
        return whole if tp == 1 else whole.chunk(tp, dim=0)[mpu.get_tensor_model_parallel_rank()]

    def master_by_id():  # collective over each DP group: every rank calls it
        return {id(param): master for param, master
                in zip(model.parameters(), megatron_masters.full_masters(
                    optimizer, list(model.parameters())), strict=True)}

    by_id = master_by_id()
    if tied is not None:
        master = by_id[id(tied)]
        assert torch.equal(master, shard_of(target[embed])), "the tied output layer kept a stale master"
        assert torch.equal(tied.detach(), master.to(tied.dtype)), "the tied output layer kept a stale copy"

    # 3. every model copy equals its master cast to the model dtype
    for param in model.parameters():
        assert torch.equal(param.detach(), by_id[id(param)].to(param.dtype)), \
            "model copy differs from its master"

    # 4. partial apply: q_proj of layer 0 (QKV needs k, v read back) and one replicated norm
    partial = {"model.layers.0.self_attn.q_proj.weight": target["model.layers.0.self_attn.q_proj.weight"] + 1.0,
               "model.layers.1.input_layernorm.weight": target["model.layers.1.input_layernorm.weight"] - 0.5}
    write, read = index.apply_plan(partial)
    assert read and set(read) <= set(write), (write, read)
    with torch.no_grad():
        boundary.apply_hf(bridge, modules, optimizer, partial, write, read)
        h2 = export(every)
    changed = sorted(n for n in shapes if not torch.equal(h2[n], target[n]))
    assert changed == sorted(partial), changed
    assert all(torch.equal(h2[n], partial[n]) for n in partial)
    result["partial_changed"] = changed
    result["partial_tasks"] = {"write": len(write), "read": len(read)}

    # 5. per-fragment export of a subset: only the tasks of the asked names run
    subset = index.tasks_for(["model.layers.1.mlp.up_proj.weight"])
    got = export(subset)
    assert set(got) == {"model.layers.1.mlp.gate_proj.weight", "model.layers.1.mlp.up_proj.weight"}, sorted(got)
    assert torch.equal(got["model.layers.1.mlp.up_proj.weight"], h2["model.layers.1.mlp.up_proj.weight"])

    # 7. partial apply of the embedding alone: the tied copy on the last stage follows
    bump = {embed: target[embed] + 2.0}
    write, read = index.apply_plan(bump)
    assert read == [] and len(write) == 1, (write, read)
    with torch.no_grad():
        tied_counts = boundary.apply_hf(bridge, modules, optimizer, bump, write, read)
        h3 = export(every)
    assert torch.equal(h3[embed], bump[embed]), "the embedding apply did not land"
    if tied is not None:
        master = master_by_id()[id(tied)]
        assert torch.equal(master, shard_of(bump[embed])), "the tied output layer did not follow the embedding"
        assert torch.equal(tied.detach(), master.to(tied.dtype)), "the tied copy differs from its master"
    else:
        master_by_id()
    result["tied_outputs"] = tied_counts["tied_outputs"]
    result["pp_rank"] = mpu.get_pipeline_model_parallel_rank()
    # Rank 0 sits on the first stage, so its own numbers say nothing about the last
    # stage: the totals below prove the pipeline checks ran somewhere.
    every_rank = [None] * world
    dist.all_gather_object(every_rank, result)
    if rank == 0:
        result["ok"] = True
        result["ranks"] = world
        result["tied_copies"] = sum(1 for row in every_rank if row["tied_copy"])
        result["tied_outputs_written"] = sum(int(row["tied_outputs"]) for row in every_rank)
        result["parameters_written"] = sum(int(row["apply_all"]["parameters_written"]) for row in every_rank)
        Path(out_path).write_text(json.dumps(result, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()


def main(argv: list[str]) -> int:
    import socket

    import torch.multiprocessing as mp

    tp, pp, dp, out_path = int(argv[0]), int(argv[1]), int(argv[2]), argv[3]
    world = tp * pp * dp
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    mp.spawn(_rank_main, args=(world, tp, pp, dp, port, out_path), nprocs=world)
    print(Path(out_path).read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
