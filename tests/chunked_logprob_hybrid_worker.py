"""Real Megatron-Core HybridModel + real verl ``model_forward_fused`` on gloo (S28 FCE).

Run by ``tests/test_s28_chunked_logprob.py`` in an interpreter that has Megatron-Core
(``YETO_MEGATRON_LM_PATH``) and verl's import closure (``YETO_VERL_FORK_PATH``, plus
``YETO_VERL_EXTRA_SITE`` for packages that interpreter lacks); not collected by pytest.

``python chunked_logprob_hybrid_worker.py <tp> <dtype> <out.json>`` spawns ``tp`` gloo
ranks. Each builds the same tiny ``HybridModel`` (two MLP layers, ``--``; RMSNorm,
untied output layer, sequence parallel when tp > 1) and runs one batch twice:

1. the current path: ``model(...)`` returns the vocab-parallel logits, then verl's
   ``vocab_parallel_log_probs_from_logits`` and ``vocab_parallel_entropy``
   (``verl/utils/megatron/tensor_parallel.py``), as the non-fused engine does;
2. the fused path: :func:`chunked_logprob.install`, verl's own
   ``patch_fused_forward`` and :func:`chunked_logprob.check_installed` (what the
   ``patch_verl.CHUNKED_LOGPROB`` hook runs), then the call verl's
   ``fused_forward_model_engine`` makes in hook mode
   (``output_processor=fused_output_processor``).

Before installing, it also records what happens without the fix: verl leaves the
HybridModel unpatched and the hook-mode call fails.

The decoder layers are MLP-only on purpose: what this checks is the seam between the
decoder output and the cross entropy (embedding, sequence-parallel scatter and gather,
final norm, output-layer weight, gradients back through all of it); Mamba and
attention layers sit upstream of it. TE is not on CPU, so ``HybridStack``'s final
``TENorm`` is swapped for a plain ``torch.nn.RMSNorm`` before the model is built.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _key in ("YETO_VERL_EXTRA_SITE", "YETO_MEGATRON_LM_PATH", "YETO_VERL_FORK_PATH"):
    _path = os.environ.get(_key)
    if _path and _path not in sys.path:
        sys.path.insert(0, _path)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")  # verl's entropy wraps a helper in torch.compile

S, H, V = 64, 64, 256


def _model(tp: int, dtype):
    from dataclasses import replace
    from functools import partial

    import torch
    from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
    from megatron.core.models.hybrid import hybrid_block
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from megatron.core.ssm.mlp_layer import MLPLayer
    from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
    from megatron.core.transformer import TransformerConfig
    from megatron.core.transformer.mlp import MLP, MLPSubmodules
    from megatron.core.transformer.spec_utils import ModuleSpec
    from megatron.core.transformer.torch_norm import WrappedTorchNorm
    from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules

    class _Norm:
        """``WrappedTorchNorm`` refuses sequence parallelism; a plain RMSNorm does not care
        (each rank normalizes its own tokens). Its weight gradient stays a per-rank partial
        here, in both paths alike (Megatron's grad finalize would sum it over TP)."""

        def __new__(cls, config, hidden_size, eps=1e-5, **_ignored):
            return torch.nn.RMSNorm(normalized_shape=hidden_size, eps=eps)

    del WrappedTorchNorm
    hybrid_block.TENorm = _Norm
    mlp_layer = ModuleSpec(module=MLPLayer, submodules=TransformerLayerSubmodules(
        pre_mlp_layernorm=_Norm,
        mlp=partial(MLP.as_mlp_submodule, submodules=MLPSubmodules(linear_fc1=ColumnParallelLinear,
                                                                   linear_fc2=RowParallelLinear)),
        mlp_bda=get_bias_dropout_add))
    spec = ModuleSpec(module=hybrid_stack_spec.module,
                      submodules=replace(hybrid_stack_spec.submodules, mlp_layer=mlp_layer))
    config = TransformerConfig(num_layers=2, hidden_size=H, num_attention_heads=4, ffn_hidden_size=96,
                               use_cpu_initialization=True, add_bias_linear=False, normalization="RMSNorm",
                               params_dtype=dtype, bf16=dtype == torch.bfloat16, tensor_model_parallel_size=tp,
                               sequence_parallel=tp > 1, hidden_dropout=0.0, attention_dropout=0.0)
    torch.manual_seed(0)
    return HybridModel(config=config, hybrid_stack_spec=spec, vocab_size=V, max_sequence_length=S,
                       hybrid_layer_pattern="--", position_embedding_type="none",
                       share_embeddings_and_output_weights=False)


def _rank_main(rank: int, tp: int, port: int, dtype_name: str, out_path: str) -> None:
    import torch
    import torch.distributed as dist

    torch.set_num_threads(1)
    # Megatron's sequence-parallel scatter/gather allocate on torch.cuda.current_device()
    # (tensor_parallel/mappings.py); None puts them on the CPU (as megatron_bridge_roundtrip)
    torch.cuda.current_device = lambda: None
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=tp)
    from megatron.core import parallel_state as mpu

    mpu.initialize_model_parallel(tensor_model_parallel_size=tp)
    import contextlib

    from megatron.core.tensor_parallel import random as tp_random

    # sequence parallel enters the CUDA RNG tracker around the embedding dropout (0 here)
    tp_random.get_cuda_rng_tracker().fork = lambda *args, **kwargs: contextlib.nullcontext()
    from verl.models.mcore import model_forward_fused as mff
    from verl.utils.megatron.tensor_parallel import vocab_parallel_entropy, vocab_parallel_log_probs_from_logits

    from yeto.rl.adapters.verl import chunked_logprob as cl

    dtype = getattr(torch, dtype_name)
    model = _model(tp, dtype)
    gen = torch.Generator().manual_seed(42)
    input_ids = torch.randint(0, V, (1, S), generator=gen)
    labels = torch.randint(0, V, (1, S), generator=gen)
    position_ids = torch.arange(S).unsqueeze(0)
    a = torch.randn(1, S, generator=gen, dtype=torch.float64)
    b = torch.randn(1, S, generator=gen, dtype=torch.float64)

    def grads():
        out = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        return out

    # 1. the current (non-fused) path
    # in the order verl's engine uses (megatron/transformer_impl.py logits_processor):
    # Megatron's cross entropy rewrites fp32 logits in place, so the log-probs take a clone
    logits = model(input_ids=input_ids, position_ids=position_ids, attention_mask=None)
    logits_seen = logits.detach().clone()
    logits_bak = logits.clone()
    ref_ent = vocab_parallel_entropy(logits)
    ref_lp = vocab_parallel_log_probs_from_logits(logits_bak, labels)
    (ref_lp.double() * a).sum().add((ref_ent.double() * b).sum()).backward()
    ref_grads = grads()

    # without the fix: verl skips the HybridModel, so fused_forward_model_engine takes the
    # legacy call (model(temperature=..., ...)), which HybridModel.forward does not accept
    mff.patch_fused_forward(model)
    unpatched_mode = getattr(model, mff._FUSED_FORWARD_MODE_ATTR, None)
    legacy = not mff._use_output_processor_hook(model)
    try:
        model(temperature=1.0, input_ids=input_ids, position_ids=position_ids, attention_mask=None,
              labels=labels)
        before_error = None
    except TypeError as error:
        before_error = f"TypeError: {error}"

    # 2. the fused path, as the patch_verl hook wires it
    installed = cl.install()
    for module in [model]:
        mff.patch_fused_forward(module)
    checked = cl.check_installed([model])
    assert mff._use_output_processor_hook(model)
    out = model(input_ids=input_ids, position_ids=position_ids, attention_mask=None, labels=labels,
                output_processor=mff.fused_output_processor,
                output_processor_context=mff.FusedOutputProcessorContext(temperature=1.0))
    lp = out.log_probs.view(1, S)
    ent = out.entropy.view(1, S)
    (lp.double() * a).sum().add((ent.double() * b).sum()).backward()
    fused_grads = grads()
    # the plain call is untouched after install
    again = model(input_ids=input_ids, position_ids=position_ids, attention_mask=None)
    plain_unchanged = bool(torch.equal(again.detach(), logits_seen))
    model.zero_grad(set_to_none=True)

    def gap(x, y):
        return float((x.detach().double() - y.detach().double()).abs().max())

    grad_gaps = {name: [gap(fused_grads[name], ref_grads[name]), float(ref_grads[name].abs().max())]
                 for name in ref_grads}
    local = {"logprob": [gap(lp, ref_lp), float(ref_lp.abs().max())],
             "entropy": [gap(ent, ref_ent), float(ref_ent.abs().max())], "grads": grad_gaps}
    gathered = [None] * tp
    dist.all_gather_object(gathered, local)
    if rank == 0:
        Path(out_path).write_text(json.dumps({
            "tp": tp, "dtype": dtype_name, "unpatched_mode": unpatched_mode, "legacy_call_before": legacy,
            "before_error": before_error,
            "installed": installed, "checked": checked, "plain_unchanged": plain_unchanged,
            "logits_shape": list(logits.shape), "ranks": gathered}))
    dist.barrier()
    dist.destroy_process_group()


def main(argv: list[str]) -> int:
    import socket

    import torch.multiprocessing as mp

    tp, dtype_name, out_path = int(argv[0]), argv[1], argv[2]
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    mp.spawn(_rank_main, args=(tp, port, dtype_name, out_path), nprocs=tp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
