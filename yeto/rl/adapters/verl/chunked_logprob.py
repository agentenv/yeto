"""Chunked vocab-parallel log-probs for verl's Megatron fused-kernel path (S28 FCE).

verl's ``use_fused_kernels`` path (``verl/models/mcore/model_forward_fused.py``) asks
for per-token log-probs and entropy straight from the last hidden states and the
output-layer weight, so the ``[T, V/TP]`` logits and their gradient never exist as a
whole. Its implementation (``verl/utils/kernel/linear_cross_entropy.py``) cannot run
on Ascend: it wraps everything in ``torch.cuda.nvtx.range`` (a RuntimeError on a
``+cpu`` torch build), asserts ``.is_cuda`` on its inputs, and is a Triton kernel.
And the path only knows ``GPTModel``: NemotronH is built by Megatron-Bridge as
``MambaModel`` (a ``HybridModel``), whose ``forward`` has no ``output_processor``
hook, so ``patch_fused_forward`` skips it and the fused call then fails with a
TypeError. (Investigation: ``infra-drafts/SESSION28/report/FCE-PLAN.md``.)

This module replaces both pieces at run time, in the worker process, without
changing a verl file on disk:

* :func:`linear_cross_entropy` -- same signature and contract as verl's (1-D fp32
  log-probs and entropy; the hidden-state gradient is this rank's partial sum over
  its vocabulary shard, which the sequence-parallel gather's reduce-scatter sums, as
  with verl's kernel). Plain torch: the tokens go through in chunks of
  :func:`chunk_tokens`; each chunk's logits live only inside the chunk and are
  computed again in the backward. Two TP all-reduces per chunk in the forward (MAX,
  then one SUM of three rows), none in the backward.
* :func:`install` -- points ``model_forward_fused.linear_cross_entropy`` here, lets
  ``_get_patching_model`` accept a ``HybridModel``, and gives ``HybridModel.forward``
  Megatron Core 0.18's ``output_processor`` / ``output_processor_context`` keywords
  (the contract ``GPTModel.forward`` already has), so verl's hook mode drives it.
* :func:`check_installed` -- run after ``patch_fused_forward``; raises unless every
  model chunk ended up on a path that reaches :func:`linear_cross_entropy`.

``patch_verl.CHUNKED_LOGPROB`` calls the last two inside verl's
``_maybe_enable_fused_kernels``, i.e. only when ``use_fused_kernels`` is True.

On Ascend, :func:`linear_cross_entropy` dispatches to :class:`ChunkedNpuLogprob`, which
keeps the same contract and numbers but uses CANN's fused vocab-parallel operators. It is
both faster than the current non-chunked path and far smaller in HBM; the measurements and
the operator semantics are in the "Ascend path" section below. (S28 G-FCK.)
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

CHUNK_ENV = "YETO_CHUNKED_LOGPROB_CHUNK"
NPU_ENV = "YETO_CHUNKED_LOGPROB_NPU"   # "0" keeps the plain torch path on Ascend too
NPU_CHUNK_ENV = "YETO_CHUNKED_LOGPROB_NPU_CHUNK"
# "0" forces the log-prob-only path, "1" forces the entropy; unset follows verl's
# calculate_entropy, which :func:`install` reads off the fused forward engine.
ENTROPY_ENV = "YETO_CHUNKED_LOGPROB_ENTROPY"
# On Ascend 910B3 the Ascend path is fastest at 8192 tokens per chunk: fewer and larger
# GEMMs, and the peak is still 1.5-2.6 GiB against the 20-40 GiB of the non-chunked path
# (measured; see the "Ascend path" section below).
DEFAULT_NPU_CHUNK_TOKENS = 8192
# Tokens per chunk. With V/TP = 32768 (131072 vocabulary, TP4) one chunk holds two
# fp32 [2048, 32768] buffers (256 MiB each) at a time; the whole-sequence logits the
# chunking removes are T * 32768 * 2 bytes in bf16, plus as much again for the gradient.
DEFAULT_CHUNK_TOKENS = 2048
# HybridModel.forward must hand back the decoder output at this line when post_process
# is off; the hook below relies on it (checked against the installed Megatron source).
_SEAM = "if not self.post_process:\n            return hidden_states\n"


def chunk_tokens() -> int:
    raw = os.environ.get(CHUNK_ENV, "")
    value = int(raw) if raw.strip() else DEFAULT_CHUNK_TOKENS
    if value <= 0:
        raise ValueError(f"{CHUNK_ENV} must be a positive token count, got {raw!r}")
    return value


# verl's calculate_entropy for the fused forward that is running, set by the wrapper
# :func:`install` puts around ``fused_forward_model_engine``. True unless verl says
# otherwise, so a missed wiring computes the entropy rather than dropping it.
_WANT_ENTROPY = True


def entropy_wanted() -> bool:
    """Whether this forward has to produce the entropy (see :data:`ENTROPY_ENV`)."""
    override = os.environ.get(ENTROPY_ENV, "")
    if override.strip():
        return override.strip() != "0"
    return _WANT_ENTROPY


def npu_chunk_tokens() -> int:
    raw = os.environ.get(NPU_CHUNK_ENV, "")
    value = int(raw) if raw.strip() else DEFAULT_NPU_CHUNK_TOKENS
    if value <= 0:
        raise ValueError(f"{NPU_CHUNK_ENV} must be a positive token count, got {raw!r}")
    return value


def _world(group) -> tuple[int, int]:
    if group is None or not dist.is_initialized():
        return 0, 1
    return dist.get_rank(group), dist.get_world_size(group)


def _accum_dtype(dtype: torch.dtype) -> torch.dtype:
    """fp32 for bf16/fp16/fp32 inputs (verl's contract), fp64 for fp64 (gradcheck)."""
    return torch.promote_types(dtype, torch.float32)


def _logits(hidden: torch.Tensor, weight: torch.Tensor, temperature: float) -> torch.Tensor:
    # the GEMM in the input dtype (as the output layer does it), then fp32 from here on
    z = torch.matmul(hidden, weight.t()).to(_accum_dtype(hidden.dtype))
    if temperature != 1.0:
        z.div_(temperature)
    return z


class ChunkedVocabParallelLogprob(torch.autograd.Function):
    """Per-token log-prob of ``labels`` and entropy of ``softmax(hidden @ weight.T / t)``.

    ``weight`` is this rank's ``[V/TP, H]`` shard of a vocab-parallel output layer
    (rows ``rank * V/TP ...``); ``group`` is the tensor-parallel group, or None."""

    @staticmethod
    def forward(ctx, hidden, weight, labels, temperature, chunk, group, want_entropy=True):
        ctx.set_materialize_grads(False)
        rank, world = _world(group)
        vocab_local = weight.shape[0]
        vocab_start = rank * vocab_local
        num_tokens = hidden.shape[0]
        acc = _accum_dtype(hidden.dtype)
        logprobs = torch.empty(num_tokens, device=hidden.device, dtype=acc)
        entropy = torch.empty(num_tokens, device=hidden.device, dtype=acc)
        lse = torch.empty(num_tokens, device=hidden.device, dtype=acc)
        mean_z = torch.empty(num_tokens, device=hidden.device, dtype=acc)
        with torch.no_grad():
            for start in range(0, num_tokens, chunk):
                end = min(start + chunk, num_tokens)
                z = _logits(hidden[start:end], weight, temperature)
                z_max = z.max(dim=-1).values
                if world > 1:
                    dist.all_reduce(z_max, op=dist.ReduceOp.MAX, group=group)
                e = torch.exp(z - z_max.unsqueeze(-1))
                sum_e = e.sum(dim=-1)
                sum_ez = e.mul_(z).sum(dim=-1)
                del e
                local = labels[start:end] - vocab_start
                in_shard = (local >= 0) & (local < vocab_local)
                target = z.gather(1, local.clamp(0, vocab_local - 1).unsqueeze(-1)).squeeze(-1)
                target = torch.where(in_shard, target, torch.zeros_like(target))
                del z
                sums = torch.stack([sum_e, sum_ez, target])
                if world > 1:
                    dist.all_reduce(sums, op=dist.ReduceOp.SUM, group=group)
                sum_e, sum_ez, target = sums[0], sums[1], sums[2]
                chunk_lse = z_max + torch.log(sum_e)
                chunk_mean = sum_ez / sum_e
                lse[start:end] = chunk_lse
                mean_z[start:end] = chunk_mean
                logprobs[start:end] = target - chunk_lse
                entropy[start:end] = chunk_lse - chunk_mean
        ctx.save_for_backward(hidden, weight, labels, lse, mean_z)
        ctx.temperature = temperature
        ctx.chunk = chunk
        ctx.vocab_start = vocab_start
        return logprobs, entropy

    @staticmethod
    def backward(ctx, d_logprobs, d_entropy):
        hidden, weight, labels, lse, mean_z = ctx.saved_tensors
        temperature, chunk, vocab_start = ctx.temperature, ctx.chunk, ctx.vocab_start
        need_hidden, need_weight = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        if (d_logprobs is None and d_entropy is None) or not (need_hidden or need_weight):
            return None, None, None, None, None, None
        vocab_local = weight.shape[0]
        num_tokens = hidden.shape[0]
        d_hidden = torch.empty_like(hidden) if need_hidden else None
        d_weight = (torch.zeros(weight.shape, device=weight.device, dtype=_accum_dtype(weight.dtype))
                    if need_weight else None)
        for start in range(0, num_tokens, chunk):
            end = min(start + chunk, num_tokens)
            h = hidden[start:end]
            z = _logits(h, weight, temperature)
            p = torch.exp(z - lse[start:end].unsqueeze(-1))
            # dH/dz_j = -p_j (z_j - E_p[z]);  dlogp_y/dz_j = [j == y] - p_j
            if d_entropy is not None:
                g = z.sub_(mean_z[start:end].unsqueeze(-1)).mul_(p).mul_(-d_entropy[start:end].unsqueeze(-1))
                if d_logprobs is not None:
                    g.sub_(p.mul_(d_logprobs[start:end].unsqueeze(-1)))
            else:
                g = p.mul_(-d_logprobs[start:end].unsqueeze(-1))
            del p
            if d_logprobs is not None:
                local = labels[start:end] - vocab_start
                in_shard = (local >= 0) & (local < vocab_local)
                rows = torch.arange(end - start, device=g.device)
                cols = local.clamp(0, vocab_local - 1)
                vals = torch.where(in_shard, d_logprobs[start:end], torch.zeros_like(d_logprobs[start:end]))
                g.index_put_((rows, cols), vals.to(g.dtype), accumulate=True)
            if temperature != 1.0:
                g.div_(temperature)
            g = g.to(hidden.dtype)
            if need_hidden:
                d_hidden[start:end] = torch.matmul(g, weight)
            if need_weight:
                part = torch.matmul(g.t(), h)
                d_weight.add_(part)
                del part
            del g
        if d_weight is not None:
            d_weight = d_weight.to(weight.dtype)
        return d_hidden, d_weight, None, None, None, None


# --------------------------------------------------------------------------- Ascend path
# Measured on one Ascend 910B3 card at the 120B island's output-layer shape (H=4096,
# V/TP=32768, bf16), forward+backward, 8192 tokens per chunk, against the current
# non-chunked path (one [T, V/TP] logits tensor plus autograd), `s1-runs/s28-fck`:
#
#   tokens | entropy | current path         | this path          | gain  | peak HBM
#   -------|---------|----------------------|--------------------|-------|-----------
#    8192  | yes     |  74.4 ms,  4800 MiB  |  59.3 ms, 1216 MiB | 1.25x | 3.9x less
#    8192  | no      |  49.2 ms,  4800 MiB  |  47.0 ms,  896 MiB | 1.05x | 5.4x less
#   32768  | yes     | 281.9 ms, 19969 MiB  | 228.1 ms, 2561 MiB | 1.24x | 7.8x less
#   32768  | no      | 182.4 ms, 19969 MiB  | 179.4 ms, 1537 MiB | 1.02x | 13x less
#   65536  | yes     | 563.7 ms, 40193 MiB  | 454.6 ms, 2306 MiB | 1.24x | 17x less
#   65536  | no      | 364.9 ms, 40193 MiB  | 357.9 ms, 1537 MiB | 1.02x | 26x less
#
# At temperature 0.7 the entropy rows are 1.27x, 1.23x and 1.24x and the 8192-token
# log-prob-only row 1.04x, but the two large log-prob-only rows are 0.987x (32768) and
# 0.996x (65536): ten timed repetitions, three runs, spread under 0.5 ms, so that is a
# real 1.3% and 0.4%, not noise. That forward cannot use the CANN operator (it has no
# temperature argument) and has to cast the chunk to fp32 to keep the log-prob exact to
# 1.5e-07, so only the recompute GEMM is left to pay for. Closing it needs a fused
# reduction kernel over the bf16 logits (triton-ascend), which is not written yet. Every
# temperature in this repo's configs is 1.0, where all six rows are >= 1.02x.
#
# Three measurements drove the implementation:
#  * ``npu_fused_linear_online_max_sum`` does the GEMM, the row max, ``sum exp(z - max)``
#    and the target logit in one 2.69 ms pass per [2048, 32768] chunk, with no logits
#    tensor at all; the equivalent torch sequence costs 2.12 ms of GEMM plus five fp32
#    passes. It has no temperature argument, so it is used only at temperature 1.0.
#  * the backward's elementwise work stays in bf16 (one ``softmax`` plus per-row scalars),
#    which pays for the extra GEMM the recompute costs.
#  * MindSpeed's ``npu_matmul_add_fp32`` folds the weight-gradient GEMM and its fp32
#    accumulation into one kernel (4.91 + 2.06 ms -> 5.66 ms per 4096-token chunk).
#
# ``npu_fused_linear_cross_entropy_loss_with_max_sum_backward`` is NOT used: its
# memory-saving mode measured 45 ms per 2048-token chunk, 5x the torch backward.

try:  # MindSpeed's "C += A.T @ B" with bf16 operands and an fp32 accumulator
    from mindspeed.ops.npu_matmul_add import npu_matmul_add_fp32 as _npu_matmul_add_fp32
except Exception:  # pragma: no cover - MindSpeed is absent off the Ascend hosts
    _npu_matmul_add_fp32 = None


def npu_fast_path_usable(hidden: torch.Tensor, weight: torch.Tensor) -> bool:
    """True when :class:`ChunkedNpuLogprob` can take this call.

    bf16/fp16 on an Ascend device, and a vocabulary shard of at least 128 rows (the CANN
    operator refuses a shorter one). ``YETO_CHUNKED_LOGPROB_NPU=0`` turns it off."""
    if os.environ.get(NPU_ENV, "1") == "0":
        return False
    if hidden.device.type != "npu" or weight.device.type != "npu":
        return False
    if hidden.dtype not in (torch.bfloat16, torch.float16) or weight.dtype != hidden.dtype:
        return False
    return weight.shape[0] >= 128


def _combine_shards(lm, se, pl, sum_ez, in_shard, group, world):
    """Reduce one shard's (local-shift) sums over the tensor-parallel group.

    ``lm`` is the shift each of ``se``/``pl``/``sum_ez`` was taken against -- the row max
    from the CANN operator, or the local log-sum-exp on the temperature path. A group
    cannot all-reduce these directly (each shard used its own shift: that was wrong by
    0.18 in the probe); every shard is rescaled to the group shift first. Returns the
    group shift, ``sum exp(z - shift)``, the target logit minus the shift, and
    ``sum exp(z - shift) * z``."""
    if world > 1:
        lm_g = lm.clone()
        dist.all_reduce(lm_g, op=dist.ReduceOp.MAX, group=group)
    else:
        lm_g = lm
    shift = lm - lm_g
    scale = torch.exp(shift)
    rows = [se * scale, torch.where(in_shard, pl + shift, torch.zeros_like(pl))]
    if sum_ez is not None:
        rows.append(sum_ez * scale)
    if world > 1:
        stacked = torch.stack(rows)
        dist.all_reduce(stacked, op=dist.ReduceOp.SUM, group=group)
        rows = list(stacked)
    return lm_g, rows[0], rows[1], (rows[2] if sum_ez is not None else None)


def _npu_forward_chunk_t1(h, weight, labels, vocab_start, vocab_local, want_entropy=True):
    """temperature 1.0: the CANN fused operator, logits materialised only for the entropy.

    Without the entropy (``return_logits=False``) the ``[chunk, V/TP]`` logits are never
    written at all: one 2.69 ms pass per [2048, 32768] chunk and nothing else."""
    import torch_npu

    lm, se, pl, _mask, _mtgt, logits = torch_npu.npu_fused_linear_online_max_sum(
        h, weight, labels, vocab_start, vocab_start + vocab_local, want_entropy)
    if not want_entropy:
        return lm, se, pl, None
    z = logits.float()
    del logits
    z.sub_(lm.unsqueeze(-1))
    # sum exp(z-m) z = sum exp(z-m) (z-m) + m sum exp(z-m)
    sum_ez = (torch.exp(z) * z).sum(-1) + lm * se
    del z
    return lm, se, pl, sum_ez


def _npu_forward_chunk_t(h, weight, labels, temperature, vocab_start, vocab_local,
                         want_entropy=True):
    """temperature != 1.0: one fp32 chunk, four reductions on it.

    The shift returned is the local log-sum-exp, so ``sum_exp`` is 1 and
    :func:`_combine_shards` still holds (``se * exp(lm - lm_g) == exp(lse - lm_g)``)."""
    z = torch.matmul(h, weight.t())
    # the cast comes first on purpose: ``torch.mul(bf16, scalar, out=fp32)`` rounds the
    # product to bf16 before writing it on this device, which cost 1.3e-4 of log-prob
    # accuracy (the criterion is 1e-5)
    zf = z.to(_accum_dtype(z.dtype))
    zf.div_(temperature)
    del z
    lse_loc = torch.logsumexp(zf, dim=-1)
    local = labels - vocab_start
    in_shard = (local >= 0) & (local < vocab_local)
    pl = zf.gather(1, local.clamp(0, vocab_local - 1).unsqueeze(-1)).squeeze(-1) - lse_loc
    pl = torch.where(in_shard, pl, torch.zeros_like(pl))
    sum_ez = None
    if want_entropy:
        p = torch.softmax(zf, dim=-1)
        sum_ez = (p * zf).sum(-1)
        del p
    del zf
    return lse_loc, torch.ones_like(lse_loc), pl, sum_ez


class ChunkedNpuLogprob(torch.autograd.Function):
    """Ascend version of :class:`ChunkedVocabParallelLogprob`: same contract, CANN ops.

    The forward keeps no ``[T, V/TP]`` tensor; the backward recomputes the chunk's logits
    and does its elementwise work in bf16, with the per-row scalars folded together."""

    @staticmethod
    def forward(ctx, hidden, weight, labels, temperature, chunk, group, want_entropy=True):
        ctx.set_materialize_grads(False)
        rank, world = _world(group)
        vocab_local = weight.shape[0]
        vocab_start = rank * vocab_local
        n = hidden.shape[0]
        dev = hidden.device
        acc = _accum_dtype(hidden.dtype)  # fp32, or fp64 for an fp64 gradcheck
        logprobs = torch.empty(n, device=dev, dtype=acc)
        lse_group = torch.empty(n, device=dev, dtype=acc)
        lse_local = torch.empty(n, device=dev, dtype=acc)
        if want_entropy:
            entropy = torch.empty(n, device=dev, dtype=acc)
            mean_z = torch.empty(n, device=dev, dtype=acc)
        else:
            # verl ignores this output when calculate_entropy is False. It is NaN, not
            # zero, so that a caller who reads it anyway sees the mistake at once.
            entropy = torch.full((n,), float("nan"), device=dev, dtype=acc)
            mean_z = torch.empty(0, device=dev, dtype=acc)
        with torch.no_grad():
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                h = hidden[start:end].contiguous()
                lab = labels[start:end]
                if temperature == 1.0:
                    lm, se, pl, sum_ez = _npu_forward_chunk_t1(
                        h, weight, lab, vocab_start, vocab_local, want_entropy)
                else:
                    lm, se, pl, sum_ez = _npu_forward_chunk_t(
                        h, weight, lab, temperature, vocab_start, vocab_local, want_entropy)
                in_shard = (lab >= vocab_start) & (lab < vocab_start + vocab_local)
                lm_g, se_g, pl_g, ez_g = _combine_shards(
                    lm, se, pl, sum_ez, in_shard, group, world)
                log_se = torch.log(se_g)
                # pl_g is the target logit *relative to lm_g*, so the log-prob subtracts
                # log(sum_exp) alone; subtracting the whole lse would remove lm_g twice
                logprobs[start:end] = pl_g - log_se
                lse_group[start:end] = lm_g + log_se
                lse_local[start:end] = lm + torch.log(se)
                if want_entropy:
                    chunk_mean = ez_g / se_g
                    mean_z[start:end] = chunk_mean
                    entropy[start:end] = lm_g + log_se - chunk_mean
        ctx.save_for_backward(hidden, weight, labels, lse_group, lse_local, mean_z)
        ctx.temperature, ctx.chunk, ctx.vocab_start = temperature, chunk, vocab_start
        ctx.want_entropy = want_entropy
        return logprobs, entropy

    @staticmethod
    def backward(ctx, d_logprobs, d_entropy):
        hidden, weight, labels, lse_g, lse_loc, mean_z = ctx.saved_tensors
        temperature, chunk, vocab_start = ctx.temperature, ctx.chunk, ctx.vocab_start
        need_hidden, need_weight = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        if (d_logprobs is None and d_entropy is None) or not (need_hidden or need_weight):
            return None, None, None, None, None, None, None
        if d_entropy is not None and not ctx.want_entropy:
            raise RuntimeError("chunked_logprob: this forward ran without the entropy "
                               "(verl's calculate_entropy was False), so there is no "
                               "entropy to differentiate")
        n, vocab_local = hidden.shape[0], weight.shape[0]
        dtype = hidden.dtype
        d_hidden = torch.empty_like(hidden) if need_hidden else None
        d_weight = (torch.zeros(weight.shape, device=weight.device,
                                dtype=_accum_dtype(weight.dtype)) if need_weight else None)
        inv_t = 1.0 / temperature
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            h = hidden[start:end].contiguous()
            if temperature == 1.0:
                z = torch.matmul(h, weight.t())
            else:
                # scale the [chunk, H] hidden slice (32 MiB) instead of the
                # [chunk, V/TP] logits (three fp32 passes over 256 MiB). Only the
                # softmax is affected, so the extra bf16 rounding lands on the gradient
                # (already bf16) and not on the log-prob the forward returned.
                z = torch.matmul((h.to(_accum_dtype(dtype)) / temperature).to(dtype),
                                 weight.t())
            # exp(z - lse_group) = softmax_local(z) * exp(lse_local - lse_group), so one
            # per-row factor turns the local softmax into the group's
            p = torch.softmax(z, dim=-1)
            k = torch.exp(lse_loc[start:end] - lse_g[start:end]) * inv_t
            local = labels[start:end] - vocab_start
            in_shard = (local >= 0) & (local < vocab_local)
            target_row = local.clamp(0, vocab_local - 1)
            if d_entropy is None:
                # Log-prob only: the whole chunk gradient is softmax times one per-row
                # scalar plus a one-hot column, and both factor out of the two GEMMs:
                #   d_hidden = (p @ W) * c + dlp/t * W[y]
                #   d_weight = p.T @ (h * c) + one-hot rows of dlp/t * h
                # so no [chunk, V/TP] tensor is written at all (that saved the 0.61 ms
                # row multiply and the 0.22 ms index_put per 4096-token chunk).
                c = (-d_logprobs[start:end] * k).to(dtype).unsqueeze(-1)
                vals = torch.where(in_shard, d_logprobs[start:end] * inv_t,
                                   torch.zeros_like(d_logprobs[start:end]))
                if need_hidden:
                    part = torch.matmul(p, weight).mul_(c)
                    part.add_(weight.index_select(0, target_row)
                              .mul(vals.to(dtype).unsqueeze(-1)))
                    d_hidden[start:end] = part
                    del part
                if need_weight:
                    scaled = h * c
                    if _npu_matmul_add_fp32 is not None:
                        _npu_matmul_add_fp32(scaled, p, d_weight)
                    else:
                        d_weight.add_(torch.matmul(p.t(), scaled).to(d_weight.dtype))
                    d_weight.index_add_(0, target_row,
                                        (h * vals.to(dtype).unsqueeze(-1)).to(d_weight.dtype))
                    del scaled
                del p, z
                continue
            # dH/dz_j = -p_j (z_j - E_p[z]);  dlogp_y/dz_j = [j == y] - p_j
            u = z.sub_(mean_z[start:end].to(dtype).unsqueeze(-1))
            u.mul_((-d_entropy[start:end] * k).to(dtype).unsqueeze(-1))
            if d_logprobs is not None:
                u.sub_((d_logprobs[start:end] * k).to(dtype).unsqueeze(-1))
            g = u.mul_(p)
            del u, p
            if d_logprobs is not None:
                vals = torch.where(in_shard, d_logprobs[start:end] * inv_t,
                                   torch.zeros_like(d_logprobs[start:end]))
                g.index_put_((torch.arange(end - start, device=g.device), target_row),
                             vals.to(dtype), accumulate=True)
            if need_hidden:
                d_hidden[start:end] = torch.matmul(g, weight)
            if need_weight:
                if _npu_matmul_add_fp32 is not None:
                    _npu_matmul_add_fp32(h, g, d_weight)
                else:
                    d_weight.add_(torch.matmul(g.t(), h).to(d_weight.dtype))
            del g, z
        if d_weight is not None:
            d_weight = d_weight.to(weight.dtype)
        return d_hidden, d_weight, None, None, None, None, None

def linear_cross_entropy(hidden, weight, labels, temperature=1.0, reduction="none", dist_process_group=None):
    """Drop-in for ``verl.utils.kernel.linear_cross_entropy.linear_cross_entropy``.

    ``hidden`` is ``[T, H]`` or Megatron's ``[S, 1, H]``; ``labels`` ``[T]`` or ``[1, S]``.
    Returns 1-D fp32 ``(logprobs, entropy)`` of length T."""
    if not isinstance(temperature, float):
        raise TypeError(f"temperature must be a float, got {type(temperature)}")
    if reduction.lower() != "none":
        raise ValueError(f"chunked_logprob: only reduction='none' is implemented, got {reduction!r}")
    if hidden.dim() == 3:
        # Megatron hands [S, B, H] while verl's labels are [B, S]: the flattened orders
        # agree only for B == 1, which is what verl's THD packing produces
        if hidden.shape[1] != 1:
            raise ValueError(f"chunked_logprob: hidden {tuple(hidden.shape)} has batch != 1")
        hidden = hidden.reshape(-1, hidden.shape[-1])
    elif hidden.dim() != 2:
        raise ValueError(f"chunked_logprob: hidden must be 2-D or 3-D, got {tuple(hidden.shape)}")
    labels = labels.reshape(-1)
    if labels.numel() != hidden.shape[0]:
        raise ValueError(f"chunked_logprob: {labels.numel()} labels for {hidden.shape[0]} tokens")
    if npu_fast_path_usable(hidden, weight):
        return ChunkedNpuLogprob.apply(hidden, weight, labels, temperature,
                                       npu_chunk_tokens(), dist_process_group,
                                       entropy_wanted())
    return ChunkedVocabParallelLogprob.apply(hidden, weight, labels, temperature, chunk_tokens(),
                                             dist_process_group)


# ----------------------------------------------------------------------------- wiring into verl
def _hybrid_class():
    try:
        from megatron.core.models.hybrid.hybrid_model import HybridModel
    except ImportError:  # Megatron Core before HybridModel: MambaModel carries the forward itself
        from megatron.core.models.mamba.mamba_model import MambaModel as HybridModel
    return HybridModel


def _wrap_fused_forward_engine(mff) -> bool:
    """Make verl's ``calculate_entropy`` visible to :func:`entropy_wanted`.

    verl puts only the temperature in ``FusedOutputProcessorContext``, so the output
    processor -- and therefore :func:`linear_cross_entropy` -- cannot see whether the
    caller asked for the entropy. ``fused_forward_model_engine`` does receive it, and
    verl calls that factory on every fused forward
    (``registry.get_mcore_forward_fused_model_engine_fn``), so this wrapper records the
    flag for the duration of the call. Returns True when it wired the wrapper.

    ``verl.models.mcore.registry`` imported the factory by name, so both module
    attributes are replaced."""
    import functools

    factory = mff.fused_forward_model_engine
    if getattr(factory, "_yeto_chunked_logprob", False):
        return False

    def fused_forward_model_engine(*factory_args, **factory_kwargs):
        inner = factory(*factory_args, **factory_kwargs)

        @functools.wraps(inner)
        def wrapper(*args, **kwargs):
            global _WANT_ENTROPY
            previous = _WANT_ENTROPY
            # absent means "compute it": a renamed argument must not silently drop the
            # entropy, it must only cost speed
            _WANT_ENTROPY = bool(kwargs.get("calculate_entropy", True))
            try:
                return inner(*args, **kwargs)
            finally:
                _WANT_ENTROPY = previous

        wrapper._yeto_chunked_logprob = True
        return wrapper

    fused_forward_model_engine._yeto_chunked_logprob = True
    mff.fused_forward_model_engine = fused_forward_model_engine
    try:
        from verl.models.mcore import registry
    except ImportError:  # pragma: no cover - verl layouts without the registry
        registry = None
    if registry is not None and hasattr(registry, "fused_forward_model_engine"):
        registry.fused_forward_model_engine = fused_forward_model_engine
    return True


def _patch_hybrid_forward(cls) -> None:
    import inspect

    if getattr(cls.forward, "_yeto_chunked_logprob", False):
        return
    if _SEAM not in inspect.getsource(cls.forward):
        raise RuntimeError(f"chunked_logprob: {cls.__module__}.{cls.__name__}.forward no longer returns the "
                           "decoder output at 'if not self.post_process'; the output_processor hook needs it")
    original = cls.forward

    def forward(self, *args, output_processor=None, output_processor_context=None, **kwargs):
        if output_processor is None or not self.post_process:
            return original(self, *args, **kwargs)
        if len(args) > 3:
            raise TypeError("chunked_logprob: pass decoder_input/labels to HybridModel.forward by keyword")
        if getattr(self, "mtp_process", False) or self.config.use_mup:
            raise RuntimeError("chunked_logprob: the output_processor hook does not cover MTP or MuP logits")
        labels = kwargs.pop("labels", None)
        # the decoder output, final norm applied (HybridStack owns it), from Megatron's own
        # forward: post_process off makes it return right before the output layer
        self.post_process = False
        try:
            hidden_states = original(self, *args, **kwargs)
        finally:
            self.post_process = True
        output_weight = self.shared_embedding_or_output_weight() if self.share_embeddings_and_output_weights \
            else None
        named = dict(zip(("input_ids", "position_ids", "attention_mask"), args))
        return output_processor(
            hidden_states=hidden_states, output_layer=self.output_layer, output_weight=output_weight,
            labels=labels, loss_mask=kwargs.get("loss_mask"),
            input_ids=named.get("input_ids", kwargs.get("input_ids")),
            position_ids=named.get("position_ids", kwargs.get("position_ids")),
            attention_mask=named.get("attention_mask", kwargs.get("attention_mask")),
            decoder_input=kwargs.get("decoder_input"), inference_context=kwargs.get("inference_context"),
            packed_seq_params=kwargs.get("packed_seq_params"),
            runtime_gather_output=kwargs.get("runtime_gather_output"), context=output_processor_context,
            compute_language_model_loss=self.compute_language_model_loss, scale_logits=self._scale_logits,
            config=self.config)

    forward.__doc__ = original.__doc__
    forward._yeto_chunked_logprob = True
    forward._yeto_original = original
    cls.forward = forward


def install() -> str:
    """Wire :func:`linear_cross_entropy` and the HybridModel hook into verl (idempotent)."""
    from verl.models.mcore import model_forward_fused as mff

    hybrid = _hybrid_class()
    mff.linear_cross_entropy = linear_cross_entropy
    wrapped_engine = _wrap_fused_forward_engine(mff)
    if not getattr(mff._get_patching_model, "_yeto_chunked_logprob", False):
        original = mff._get_patching_model

        def _get_patching_model(model):
            inner = mff.unwrap_model(model)
            if isinstance(inner, hybrid):
                return inner
            return original(model)

        _get_patching_model._yeto_chunked_logprob = True
        mff._get_patching_model = _get_patching_model
    _patch_hybrid_forward(hybrid)
    return (f"linear_cross_entropy -> yeto chunked ({chunk_tokens()} tokens/chunk, "
            f"{npu_chunk_tokens()} on Ascend); {hybrid.__name__}.forward takes "
            f"output_processor; calculate_entropy "
            f"{'wired' if wrapped_engine else 'already wired'}")


def check_installed(modules) -> str:
    """Raise unless every model chunk now reaches :func:`linear_cross_entropy`."""
    from megatron.core import parallel_state
    from verl.models.mcore import model_forward_fused as mff

    hybrid = _hybrid_class()
    if mff.linear_cross_entropy is not linear_cross_entropy:
        raise RuntimeError("chunked_logprob: model_forward_fused.linear_cross_entropy is not yeto's")
    if not getattr(mff.fused_forward_model_engine, "_yeto_chunked_logprob", False):
        raise RuntimeError("chunked_logprob: fused_forward_model_engine is not wrapped, so "
                           "verl's calculate_entropy cannot reach the Ascend path")
    tp = parallel_state.get_tensor_model_parallel_world_size()
    seen = []
    for chunk in modules:
        inner = mff._get_patching_model(chunk)
        if inner is None:
            raise RuntimeError(f"chunked_logprob: {type(mff.unwrap_model(chunk)).__name__} has no fused forward")
        mode = mff._get_fused_forward_mode(chunk)
        if isinstance(inner, hybrid) and mode != "hook":
            raise RuntimeError(f"chunked_logprob: {type(inner).__name__} is in fused mode {mode!r}, not 'hook'")
        if inner.post_process:
            config = inner.config
            if config.use_mup or getattr(inner, "mtp_process", False):
                raise RuntimeError("chunked_logprob: MTP or MuP on the last stage is not covered")
            # the hidden gradient is a per-shard partial sum; only the sequence-parallel
            # gather's reduce-scatter adds the shards up (verl's kernel has the same contract)
            if tp > 1 and not config.sequence_parallel:
                raise RuntimeError("chunked_logprob: TP > 1 needs sequence_parallel=True")
        seen.append(f"{type(inner).__name__}:{mode}:{'last' if inner.post_process else 'mid'}")
    return (f"tp={tp} chunk={chunk_tokens()} npu_chunk={npu_chunk_tokens()} "
            f"entropy={entropy_wanted()} " + ",".join(seen))
