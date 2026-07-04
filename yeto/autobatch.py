"""Auto micro-batch sizing: probe the largest per-GPU batch that fits.

Static micro-batch defaults leave most of a big card idle (a B200 running
an FSDP-sharded base has plenty free at micro-batch 1). No offline formula
gets activation memory right — it depends on the attention implementation,
MoE routing spikes, checkpointing policy, and allocator behavior — so we
probe at startup: double the batch until a real fwd/bwd/step OOMs and keep
the largest size that passed. Packed blocks give constant shapes, so a
passing probe is representative of every future step.

The probe runs after model wrap + optimizer construction (memory-accurate)
and before the syncer handshake (counters never see it). Probing must not
change the model: gradients are zeroed before the optimizer step, which
still materializes optimizer state (part of the footprint) while Adam's
update is exactly zero.
"""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist

log = logging.getLogger("learner")

_MAX_MICRO_BATCH = 256
_PROBE_ITERATIONS = 2  # second pass sees allocator/fragmentation steady state


def int_or_auto(value: str):
    """argparse type for --micro-batch-size."""
    return value if value == "auto" else int(value)


def _probe_once(model, params, opt, seq_len: int, vocab: int, device, micro_batch: int) -> None:
    for _ in range(_PROBE_ITERATIONS):
        ids = torch.randint(0, vocab, (micro_batch, seq_len), device=device)
        out = model(input_ids=ids)
        # Mirror sft_loss's memory peak: the shifted f32 logits copy
        # dominates the loss computation.
        loss = out.logits[:, :-1].float().sum()
        loss.backward()
        with torch.no_grad():
            for p in params.values():
                if p.grad is not None:
                    p.grad.zero_()
        opt.step()
        opt.zero_grad(set_to_none=True)


def resolve_micro_batch_size(args, model, params, opt, tokenizer, device, world: int) -> int:
    """The micro batch to train with; probes when --micro-batch-size=auto.

    All ranks probe each size in lockstep and agree via a MIN all-reduce
    after every attempt, so one rank's OOM cannot silently desynchronize
    the group (ranks are memory-symmetric in practice; the flag exchange
    is the backstop). CPU runs skip probing entirely.
    """
    if args.micro_batch_size != "auto":
        return int(args.micro_batch_size)
    if device.type != "cuda":
        return 1
    vocab = max(2, len(tokenizer) - 1)
    best, size = 0, 1
    while size <= _MAX_MICRO_BATCH:
        ok = True
        try:
            _probe_once(model, params, opt, args.seq_len, vocab, device, size)
        except torch.cuda.OutOfMemoryError:
            ok = False
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        if world > 1:
            flag = torch.tensor([1.0 if ok else 0.0], device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            ok = flag.item() > 0
        if not ok:
            break
        best = size
        size *= 2
    if best == 0:
        raise RuntimeError(
            "model does not fit even micro-batch 1 at this --seq-len; "
            "reduce --seq-len or use a bigger island"
        )
    return best


def rebalance_grad_accum(grad_accum: int, micro_batch: int) -> int:
    """Keep the effective batch (micro_batch x grad_accum) no larger than
    the flags implied at micro-batch 1: bigger probed batches shrink the
    accumulation instead of multiplying tokens per optimizer step."""
    return max(1, -(-grad_accum // micro_batch))
