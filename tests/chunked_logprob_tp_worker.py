"""Gloo TP ranks for ``tests/test_s28_chunked_logprob.py`` (S28 FCE); not collected by pytest.

``python chunked_logprob_tp_worker.py <tp> <dtype> <out.json>`` spawns ``tp`` gloo ranks.
Every rank builds the same full ``[V, H]`` weight and ``[T, H]`` hidden states (same
seed), keeps its vocabulary shard (rows ``rank * V/tp ...``, Megatron's layout) and runs
:class:`ChunkedVocabParallelLogprob` over the TP group. Rank 0 compares with the
whole-vocabulary reference (full logits, ``log_softmax``):

* log-probs and entropy (every rank's must also be bit-identical to rank 0's);
* the hidden-state gradient: the sum over ranks of each rank's partial gradient (what
  the sequence-parallel gather's reduce-scatter forms) against the full gradient;
* the weight gradient: each rank's shard against the matching rows of the full one.

A fourth argument picks the implementation: ``torch`` (the default,
:class:`ChunkedVocabParallelLogprob`) or ``npu`` (:class:`ChunkedNpuLogprob`). The Ascend
class runs here because its temperature path is plain torch -- only its temperature-1.0
forward calls a CANN operator -- so this worker covers its shard combination and its
backward on the CPU, where an fp64 reference is available. (S28 G-FCK.)

Prints nothing; writes ``{"ok": ..., "max_abs": {...}}`` from rank 0.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

T, H, V, CHUNK, TEMPERATURE = 333, 48, 1024, 64, 0.8


def _inputs(dtype):
    import torch

    gen = torch.Generator().manual_seed(1234)
    hidden = torch.randn(T, H, generator=gen, dtype=torch.float64)
    weight = torch.randn(V, H, generator=gen, dtype=torch.float64) * (2.0 / H ** 0.5)
    labels = torch.randint(0, V, (T,), generator=gen)
    d_lp = torch.randn(T, generator=gen, dtype=torch.float64)
    d_ent = torch.randn(T, generator=gen, dtype=torch.float64)
    return hidden.to(dtype), weight.to(dtype), labels, d_lp, d_ent


def _reference(hidden, weight, labels, d_lp, d_ent):
    import torch

    h = hidden.detach().clone().requires_grad_()
    w = weight.detach().clone().requires_grad_()
    logp = torch.log_softmax((h @ w.t()).to(torch.promote_types(h.dtype, torch.float32)) / TEMPERATURE, -1)
    lp = logp.gather(1, labels.unsqueeze(-1)).squeeze(-1)
    ent = -(logp.exp() * logp).sum(-1)
    (lp * d_lp.to(lp.dtype)).sum().add((ent * d_ent.to(ent.dtype)).sum()).backward()
    return lp.detach(), ent.detach(), h.grad, w.grad


def _rank_main(rank: int, tp: int, port: int, dtype_name: str, out_path: str,
               impl: str = "torch") -> None:
    import torch
    import torch.distributed as dist

    from yeto.rl.adapters.verl.chunked_logprob import ChunkedNpuLogprob, ChunkedVocabParallelLogprob

    function = ChunkedNpuLogprob if impl.startswith("npu") else ChunkedVocabParallelLogprob
    want_entropy = impl != "npu_noent"

    torch.set_num_threads(1)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=tp)
    group = dist.new_group(list(range(tp)))
    dtype = getattr(torch, dtype_name)
    hidden, weight, labels, d_lp, d_ent = _inputs(dtype)
    shard = V // tp
    h = hidden.clone().requires_grad_()
    w = weight[rank * shard:(rank + 1) * shard].clone().requires_grad_()
    if want_entropy:
        lp, ent = function.apply(h, w, labels, TEMPERATURE, CHUNK, group)
        (lp * d_lp.to(lp.dtype)).sum().add((ent * d_ent.to(ent.dtype)).sum()).backward()
    else:
        # verl's calculate_entropy=False: the entropy output is NaN and untouched, so
        # only the log-prob and its gradients can be compared
        lp, ent = function.apply(h, w, labels, TEMPERATURE, CHUNK, group, False)
        (lp * d_lp.to(lp.dtype)).sum().backward()
    partial_dh = h.grad.clone()
    dist.all_reduce(h.grad, group=group)
    outs = [torch.empty_like(lp) for _ in range(tp)]
    dist.all_gather(outs, lp, group=group)
    ents = [torch.empty_like(ent) for _ in range(tp)]
    dist.all_gather(ents, ent, group=group)
    dws = [torch.empty_like(w.grad) for _ in range(tp)]
    dist.all_gather(dws, w.grad, group=group)
    if rank == 0:
        ref_lp, ref_ent, ref_dh, ref_dw = _reference(
            hidden, weight, labels, d_lp, d_ent if want_entropy else torch.zeros_like(d_ent))

        def gap(a, b):
            return float((a.detach().double() - b.detach().double()).abs().max())

        result = {
            "tp": tp, "dtype": dtype_name, "impl": impl, "T": T, "H": H, "V": V, "chunk": CHUNK, "temperature": TEMPERATURE,
            "max_abs": {"logprob": gap(lp, ref_lp), "d_hidden": gap(h.grad, ref_dh),
                        "d_weight": gap(torch.cat(dws), ref_dw),
                        **({"entropy": gap(ent, ref_ent)} if want_entropy else {})},
            "ref_max_abs": {"logprob": float(ref_lp.abs().max()), "entropy": float(ref_ent.abs().max()),
                            "d_hidden": float(ref_dh.abs().max()), "d_weight": float(ref_dw.abs().max())},
            "entropy_is_nan": bool(torch.isnan(ent).all()) if not want_entropy else None,
            "ranks_identical": all(torch.equal(o, lp) for o in outs)
            and (all(torch.equal(e, ent) for e in ents) if want_entropy else True),
            "partial_d_hidden_differs_from_full": gap(partial_dh, ref_dh) > 0 if tp > 1 else True,
        }
        Path(out_path).write_text(json.dumps(result))
    dist.barrier(group=group)
    dist.destroy_process_group()


def main(argv: list[str]) -> int:
    import socket

    import torch.multiprocessing as mp

    tp, dtype_name, out_path = int(argv[0]), argv[1], argv[2]
    impl = argv[3] if len(argv) > 3 else "torch"
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    mp.spawn(_rank_main, args=(tp, port, dtype_name, out_path, impl), nprocs=tp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
