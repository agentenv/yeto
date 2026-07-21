"""Auto micro-batch sizing: probe the largest exact batch recipe that fits.

Static micro-batch defaults leave most of a big card idle (a B200 running
an FSDP-sharded base has plenty free at micro-batch 1). No offline formula
gets activation memory right — it depends on the attention implementation,
MoE routing spikes, checkpointing policy, and allocator behavior — so we
probe at startup with real forward/backward passes and keep the largest
micro-batch that both fits and divides the requested effective batch.
Packed blocks give constant shapes, so a passing probe is representative of
every future step.

The probe runs after model wrap + optimizer construction (memory-accurate)
and before the syncer handshake (counters never see it). Probing must not
change training state. Temporary AdamW buffers account for the steady-state
optimizer footprint without stepping the real parameters or optimizer; CPU
and CUDA RNG and any pre-existing gradients are restored on every exit.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

import torch
import torch.distributed as dist

log = logging.getLogger("learner")

_MAX_MICRO_BATCH = 256
_PROBE_ITERATIONS = 2  # second pass sees allocator/fragmentation steady state


def int_or_auto(value: str):
    """argparse type for --micro-batch-size."""
    return value if value == "auto" else int(value)


def _parameters_once(params) -> list[torch.nn.Parameter]:
    """Return optimizer parameters once, preserving their input order."""
    unique: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for param in params.values():
        if id(param) not in seen:
            unique.append(param)
            seen.add(id(param))
    return unique


@contextmanager
def _preserve_probe_observables(params, device) -> Iterator[list[torch.nn.Parameter]]:
    """Restore RNG and gradients even when a probe OOMs.

    Parameters themselves are never stepped, so no full-model backup is
    needed. Gradient backups are allocated only when the caller already had
    gradients; at the startup call site all gradients are normally ``None``.
    ``fork_rng`` restores the CPU generator and the CUDA generator used by
    this rank.
    """
    parameters = _parameters_once(params)
    saved_grads = []
    for param in parameters:
        original = param.grad
        saved_grads.append(
            (param, original, None if original is None else original.detach().clone())
        )

    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices.append(
            device.index if device.index is not None else torch.cuda.current_device()
        )

    try:
        with torch.random.fork_rng(devices=cuda_devices):
            yield parameters
    finally:
        for param, original, saved in saved_grads:
            if original is None:
                param.grad = None
            else:
                original.copy_(saved)
                param.grad = original


def _temporary_adamw_footprint(opt) -> list[torch.Tensor]:
    """Model single-tensor AdamW's persistent and step-time GPU footprint.

    The learner always constructs ``torch.optim.AdamW``. A real first
    ``step`` would create step/first-moment/second-moment tensors and, with
    weight decay, would also change parameters even for zero gradients.
    Holding equivalent scratch buffers during the forward/backward probe
    measures the steady-state footprint without either side effect.

    The training optimizer sets ``foreach=False``: single-tensor AdamW holds
    one parameter-sized denominator at a time during ``step`` rather than
    foreach's parameter-list-sized intermediates. One scratch tensor matching
    the largest optimizer parameter conservatively reserves that transient.
    """
    if not isinstance(opt, torch.optim.AdamW):
        raise TypeError("causal-LM auto-batch probing requires torch.optim.AdamW")

    scratch: list[torch.Tensor] = []
    largest_param = None
    for group in opt.param_groups:
        if group.get("foreach") is not False or group.get("fused") is True:
            raise ValueError(
                "causal-LM auto-batch probing requires AdamW(foreach=False, fused=False)"
            )
        for param in group["params"]:
            if largest_param is None or param.numel() > largest_param.numel():
                largest_param = param
            if opt.state.get(param):
                # Existing state already occupies memory and must remain
                # completely untouched by the probe.
                continue
            state_device = (
                param.device
                if group.get("capturable", False) or group.get("fused", False)
                else torch.device("cpu")
            )
            scratch.append(torch.zeros((), device=state_device))  # Adam step
            scratch.append(torch.zeros_like(param))  # first moment
            scratch.append(torch.zeros_like(param))  # second moment
            if group.get("amsgrad", False):
                scratch.append(torch.zeros_like(param))
    if largest_param is not None:
        scratch.append(torch.empty_like(largest_param))  # step-time denominator
    return scratch


def _probe_once(
    model,
    params,
    opt,
    seq_len: int,
    vocab: int,
    device,
    micro_batch: int,
    *,
    loss_forward=None,
) -> None:
    with _preserve_probe_observables(params, device) as parameters:
        # Keep these live for both passes so the probe sees the same memory
        # available after AdamW has initialized its state on the first real
        # training step. This is intentionally scratch state, not opt.state.
        optimizer_footprint = _temporary_adamw_footprint(opt)
        for _ in range(_PROBE_ITERATIONS):
            ids = torch.randint(0, vocab, (micro_batch, seq_len), device=device)
            if loss_forward is None:
                out = model(input_ids=ids)
                # Mirror native sft_loss's memory peak: the shifted f32
                # logits copy dominates the loss computation.
                loss = out.logits[:, :-1].float().sum()
            else:
                # Optimized backends can have a materially different peak
                # (notably fused linear CE, which never materializes logits).
                # Probe the exact selected loss path instead of a conservative
                # native-logits surrogate.
                loss = loss_forward(model, ids)
            loss.backward()
            for param in parameters:
                param.grad = None
        # Keep an explicit reference through the last backward above.
        del optimizer_footprint


def _exact_micro_batch_candidates(effective_batch: int) -> list[int]:
    """Ascending micro-batches that preserve ``effective_batch`` exactly."""
    if effective_batch < 1:
        raise ValueError("--grad-accum must be at least 1")
    limit = min(_MAX_MICRO_BATCH, effective_batch)
    return [size for size in range(1, limit + 1) if effective_batch % size == 0]


def exact_grad_accum(effective_batch: int, micro_batch: int) -> int:
    """Gradient accumulation for an exact integer batch recipe."""
    if effective_batch < 1 or micro_batch < 1:
        raise ValueError("effective batch and micro batch must be at least 1")
    quotient, remainder = divmod(effective_batch, micro_batch)
    if remainder:
        raise ValueError(
            f"micro batch {micro_batch} does not divide effective batch {effective_batch}"
        )
    return quotient


def resolve_micro_batch_size(
    args,
    model,
    params,
    opt,
    tokenizer,
    device,
    world: int,
    *,
    loss_forward=None,
) -> int:
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
    requested_effective_batch = int(args.grad_accum)
    best = 0
    for size in _exact_micro_batch_candidates(requested_effective_batch):
        ok = True
        try:
            _probe_once(
                model,
                params,
                opt,
                args.seq_len,
                vocab,
                device,
                size,
                loss_forward=loss_forward,
            )
        except torch.cuda.OutOfMemoryError:
            ok = False
        torch.cuda.empty_cache()
        if world > 1:
            flag = torch.tensor([1.0 if ok else 0.0], device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            ok = flag.item() > 0
        if not ok:
            break
        best = size
    if best == 0:
        raise RuntimeError(
            "model does not fit even micro-batch 1 at this --seq-len; "
            "reduce --seq-len or use a bigger island"
        )
    log.info(
        "auto-batch probe: largest fitting exact micro-batch=%d for requested "
        "per-rank effective batch=%d",
        best,
        requested_effective_batch,
    )
    return best


def rebalance_grad_accum(grad_accum: int, micro_batch: int) -> int:
    """Keep the effective batch (micro_batch x grad_accum) no larger than
    the flags implied at micro-batch 1: bigger probed batches shrink the
    accumulation instead of multiplying tokens per optimizer step."""
    return max(1, -(-grad_accum // micro_batch))
