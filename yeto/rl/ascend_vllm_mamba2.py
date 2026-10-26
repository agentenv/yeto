"""Mamba2 (NemotronH) generation on Ascend NPU with vLLM 0.18.0 + vllm_ascend 0.17.0rc2.

vllm_ascend 0.17.0rc2 has no Mamba2 SSD kernels, and its causal_conv1d prefill wrapper does
not take the keyword arguments vLLM's MambaMixer2 passes. Its unquantized MoE MLP also always
applies SwiGLU, while NemotronH experts use relu2 without a gate. This module replaces those
pieces with plain torch code:

- ``ssd_prefill``: chunked SSD scan per sequence (fp32), the semantics of
  ``mamba_chunk_scan_combined_varlen`` without intermediate states.
- ``ssd_decode``: one-token state update, the semantics of ``selective_state_update``.
- ``conv_prefill`` / ``conv_decode``: depthwise causal conv1d with the conv state cache.
- ``relu2_unquant_apply_mlp``: vllm_ascend's unquantized grouped MoE MLP with relu2 activation.

Limits (asserted): no prefix caching (mamba_cache_mode "all"), no speculative decoding.
The patches target eager mode; graph capture is untested.

Activation: ``apply()`` rebinds module globals in vLLM / vllm_ascend. In vLLM worker processes
it runs through the ``vllm.general_plugins`` entry point ``register`` (see
``tools/ascend_vllm_mamba2/``). All reports use ``print(..., flush=True)``.
"""

from __future__ import annotations

import os

PAD_SLOT_ID = -1
_APPLIED: list[str] = []


def _say(msg: str) -> None:
    print(f"[yeto-mamba2] {msg}", flush=True)


# ----------------------------------------------------------------------------- SSD prefill
def ssd_scan_one(x, dt, A, B, C, chunk_size, init_state=None):
    """One sequence. x (L,H,P), dt (L,H) after softplus/bias, A (H,), B/C (L,H,N) per head.

    Returns y (L,H,P) without the D term and the final state (H,P,N), all fp32.
    """
    import torch

    L, H, P = x.shape
    N = B.shape[-1]
    Q = chunk_size
    c = (L + Q - 1) // Q
    pad = c * Q - L
    xdt = x * dt[..., None]
    dA = dt * A[None, :]
    if pad:
        xdt = torch.nn.functional.pad(xdt, (0, 0, 0, 0, 0, pad))
        dA = torch.nn.functional.pad(dA, (0, 0, 0, pad))
        B = torch.nn.functional.pad(B, (0, 0, 0, 0, 0, pad))
        C = torch.nn.functional.pad(C, (0, 0, 0, 0, 0, pad))
    xdt = xdt.view(c, Q, H, P)
    Bc = B.view(c, Q, H, N)
    Cc = C.view(c, Q, H, N)
    acs = torch.cumsum(dA.view(c, Q, H), dim=1)  # (c,Q,H)
    seg = acs[:, :, None, :] - acs[:, None, :, :]  # (c,l,s,H) = acs[l]-acs[s]
    causal = torch.ones(Q, Q, dtype=torch.bool, device=x.device).tril()
    seg = seg.masked_fill(~causal[None, :, :, None], float("-inf"))
    scores = torch.einsum("clhn,cshn->clsh", Cc, Bc) * torch.exp(seg)
    y = torch.einsum("clsh,cshp->clhp", scores, xdt)
    decay_states = torch.exp(acs[:, -1:, :] - acs)  # (c,Q,H)
    states = torch.einsum("clhn,clhp->chpn", Bc * decay_states[..., None], xdt)  # (c,H,P,N)
    chunk_decay = torch.exp(acs[:, -1, :])  # (c,H)
    s = torch.zeros(H, P, N, dtype=x.dtype, device=x.device) if init_state is None else init_state
    states_in = []
    for k in range(c):
        states_in.append(s)
        s = s * chunk_decay[k][:, None, None] + states[k]
    states_in = torch.stack(states_in, 0)  # (c,H,P,N)
    y = y + torch.einsum("clhn,chpn->clhp", Cc * torch.exp(acs)[..., None], states_in)
    return y.reshape(c * Q, H, P)[:L], s


def ssd_prefill(x, dt, A, B, C, chunk_size, cu_seqlens, cu_chunk_seqlens=None, last_chunk_indices=None,
                seq_idx=None, out=None, D=None, z=None, dt_bias=None, initial_states=None, dt_softplus=False,
                dt_limit=(0.0, float("inf")), return_intermediate_states=False, state_dtype=None):
    """Drop-in for vLLM's ``mamba_chunk_scan_combined_varlen`` (no intermediate states, no z)."""
    import torch

    assert z is None, "z is not supported"
    assert not return_intermediate_states, "prefix caching (mamba_cache_mode='all') is not supported"
    T, H, P = x.shape
    G = B.shape[1]
    rep = H // G
    f32 = torch.float32
    dtf = dt.to(f32)
    if dt_bias is not None:
        dtf = dtf + dt_bias.to(f32)[None, :]
    if dt_softplus:
        dtf = torch.nn.functional.softplus(dtf)
    lo, hi = dt_limit
    if lo > 0.0 or hi != float("inf"):
        dtf = dtf.clamp(min=lo, max=hi)
    Af = A.to(f32)
    xf = x.to(f32)
    Bh = B.to(f32).repeat_interleave(rep, dim=1)
    Ch = C.to(f32).repeat_interleave(rep, dim=1)
    bounds = cu_seqlens.tolist()
    finals = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        init = None if initial_states is None else initial_states[i].to(f32)
        y, s = ssd_scan_one(xf[a:b], dtf[a:b], Af, Bh[a:b], Ch[a:b], chunk_size, init)
        if D is not None:
            Df = D.to(f32)
            y = y + (Df[:, None] if Df.dim() == 1 else Df) * xf[a:b]
        out[a:b] = y.to(out.dtype)
        finals.append(s)
    st = torch.stack(finals, 0)
    return st.to(state_dtype) if state_dtype is not None else st


# ----------------------------------------------------------------------------- SSD decode
def ssd_decode(state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False, state_batch_indices=None,
               dst_state_batch_indices=None, pad_slot_id=PAD_SLOT_ID, out=None, num_accepted_tokens=None,
               cu_seqlens=None, is_blackwell=False):
    """Drop-in for vLLM's ``selective_state_update`` (one token per sequence, 4-D state)."""
    import torch

    assert z is None, "z is not supported"
    assert num_accepted_tokens is None and cu_seqlens is None, "speculative decoding is not supported"
    assert state.dim() == 4 and x.dim() == 3
    f32 = torch.float32
    b, H, P = x.shape
    G = B.shape[1]
    rep = H // G
    if state_batch_indices is None:
        src = torch.arange(b, device=x.device)
    else:
        src = state_batch_indices.reshape(-1)[:b].long()
    dst = src if dst_state_batch_indices is None else dst_state_batch_indices.reshape(-1)[:b].long()
    valid = src != pad_slot_id
    rows = None
    if not bool(valid.all()):
        rows = valid.nonzero().squeeze(1)
        src, dst = src[rows], dst[rows]
        x, dt, B, C = x[rows], dt[rows], B[rows], C[rows]
    dtf = dt.to(f32)
    if dt_bias is not None:
        dtf = dtf + dt_bias.to(f32)
    if dt_softplus:
        dtf = torch.nn.functional.softplus(dtf)
    s = state[src].to(f32)  # (b,H,P,N)
    dA = torch.exp(dtf[..., None] * A.to(f32)[None])  # (b,H,P,N)
    Bh = B.to(f32).repeat_interleave(rep, dim=1)  # (b,H,N)
    Ch = C.to(f32).repeat_interleave(rep, dim=1)
    xf = x.to(f32)
    s = s * dA + (dtf * xf)[..., None] * Bh[:, :, None, :]
    y = (s * Ch[:, :, None, :]).sum(-1)
    if D is not None:
        y = y + D.to(f32) * xf
    state[dst] = s.to(state.dtype)
    if rows is None:
        out.copy_(y.to(out.dtype))
    else:
        out[rows] = y.to(out.dtype)
    return out


# ----------------------------------------------------------------------------- causal conv1d
def conv_prefill(x, weight, bias=None, activation="silu", conv_states=None, has_initial_state=None,
                 cache_indices=None, query_start_loc=None, block_idx_first_scheduled_token=None,
                 block_idx_last_scheduled_token=None, initial_state_idx=None, num_computed_tokens=None,
                 block_size_to_align=0, metadata=None, pad_slot_id=PAD_SLOT_ID, validate_data=False):
    """Drop-in for vLLM's ``causal_conv1d_fn``. x (dim, T); conv_states (slots, dim, width-1) view."""
    import torch

    assert block_idx_first_scheduled_token is None and block_idx_last_scheduled_token is None, (
        "prefix caching (mamba_cache_mode='all') is not supported")
    assert activation in (None, "silu", "swish")
    dim, _ = x.shape
    width = weight.shape[1]
    out = torch.zeros_like(x)
    bounds = query_start_loc.tolist()
    idx = cache_indices.tolist()
    has_init = has_initial_state.tolist() if has_initial_state is not None else [False] * len(idx)
    w = weight[:, None, :]
    for i in range(len(bounds) - 1):
        if idx[i] == pad_slot_id:
            continue
        a, b = bounds[i], bounds[i + 1]
        xs = x[:, a:b]
        if has_init[i]:
            init = conv_states[idx[i]][:, : width - 1].to(xs.dtype)
        else:
            init = torch.zeros(dim, width - 1, dtype=xs.dtype, device=xs.device)
        xx = torch.cat([init, xs], dim=-1)  # (dim, width-1+L)
        y = torch.nn.functional.conv1d(xx[None].float(), w.float(), None if bias is None else bias.float(),
                                       groups=dim)[0]
        if activation is not None:
            y = torch.nn.functional.silu(y)
        out[:, a:b] = y.to(x.dtype)
        conv_states[idx[i]][:, : width - 1] = xx[:, -(width - 1):].to(conv_states.dtype)
    return out


def conv_decode(x, conv_state, weight, bias=None, activation=None, conv_state_indices=None,
                num_accepted_tokens=None, query_start_loc=None, max_query_len=-1, pad_slot_id=PAD_SLOT_ID,
                block_idx_last_scheduled_token=None, initial_state_idx=None, validate_data=False):
    """Drop-in for vLLM's ``causal_conv1d_update`` with x (batch, dim), one token per sequence."""
    import torch

    assert num_accepted_tokens is None and query_start_loc is None, "speculative decoding is not supported"
    assert block_idx_last_scheduled_token is None, "prefix caching is not supported"
    assert x.dim() == 2
    width = weight.shape[1]
    b = x.shape[0]
    idx = (torch.arange(b, device=x.device) if conv_state_indices is None
           else conv_state_indices.reshape(-1)[:b].long())
    valid = idx != pad_slot_id
    rows = None
    xs = x
    if not bool(valid.all()):
        rows = valid.nonzero().squeeze(1)
        idx, xs = idx[rows], x[rows]
    st = conv_state[idx][:, :, : width - 1]  # (b, dim, width-1)
    xx = torch.cat([st.to(xs.dtype), xs[:, :, None]], dim=-1)  # (b, dim, width)
    y = (xx.float() * weight.float()[None]).sum(-1)
    if bias is not None:
        y = y + bias.float()
    if activation in ("silu", "swish", True):
        y = torch.nn.functional.silu(y)
    conv_state[idx, :, : width - 1] = xx[:, :, 1:].to(conv_state.dtype)
    if rows is None:
        return y.to(x.dtype)
    out = torch.zeros_like(x)
    out[rows] = y.to(x.dtype)
    return out


# ----------------------------------------------------------------------------- MoE relu2
def _is_relu2_no_mul(activation) -> bool:
    return getattr(activation, "value", activation) == "relu2_no_mul"


def make_relu2_unquant_apply_mlp(orig):
    def relu2_unquant_apply_mlp(hidden_states, w1, w2, group_list, w1_bias=None, w2_bias=None, activation=None,
                                group_list_type=1, topk_scales=None, need_trans=True):
        if not _is_relu2_no_mul(activation):
            return orig(hidden_states=hidden_states, w1=w1, w2=w2, group_list=group_list, w1_bias=w1_bias,
                        w2_bias=w2_bias, activation=activation, group_list_type=group_list_type,
                        topk_scales=topk_scales, need_trans=need_trans)
        import torch
        import torch_npu

        if need_trans:
            w1 = w1.transpose(1, 2)
            w2 = w2.transpose(1, 2)
        h = torch_npu.npu_grouped_matmul(
            x=[hidden_states], weight=[w1],
            bias=[w1_bias.to(dtype=torch.float32)] if w1_bias is not None else None,
            split_item=2, group_list_type=group_list_type, group_type=0, group_list=group_list)[0]
        h = torch.square(torch.relu(h))
        if topk_scales is not None:
            h *= topk_scales
        return torch_npu.npu_grouped_matmul(
            x=[h], weight=[w2],
            bias=[w2_bias.to(dtype=torch.float32)] if w2_bias is not None else None,
            split_item=2, group_list_type=group_list_type, group_type=0, group_list=group_list)[0]

    relu2_unquant_apply_mlp._yeto_orig = orig
    return relu2_unquant_apply_mlp


class _GatelessMoEPlatform:
    """Proxy for ``fused_moe.layer.current_platform``.

    ``FusedMoE.__init__`` rejects ``is_act_and_mul=False`` (relu2 experts) unless the platform is
    CUDA-alike. Its body has exactly one ``current_platform`` call (vLLM 0.18.0 layer.py:596), so the
    proxy answers ``is_cuda_alike() -> True`` only when the direct caller is that function. All other
    attributes and callers see the real platform.
    """

    def __init__(self, real, init_code):
        self._real = real
        self._init_code = init_code

    def is_cuda_alike(self):
        import sys

        if sys._getframe(1).f_code is self._init_code:
            return True
        return self._real.is_cuda_alike()

    def __getattr__(self, name):
        return getattr(self._real, name)


def make_shared_part1(orig):
    """vllm_ascend's split shared-expert path calls ``gate_up_proj``; NemotronH's ungated MLP has
    ``up_proj`` (then relu2 in ``act_fn`` and ``down_proj``, which part2 already calls)."""

    def _shared_experts_part1(self, hidden_states):
        se = self._shared_experts
        if not hasattr(se, "gate_up_proj") and hasattr(se, "up_proj"):
            out, _ = se.up_proj(hidden_states)
            return out
        return orig(self, hidden_states)

    _shared_experts_part1._yeto_orig = orig
    return _shared_experts_part1


# ----------------------------------------------------------------------------- apply
def apply() -> list[str]:
    """Rebind the vLLM / vllm_ascend globals. Idempotent. Returns the list of applied patches."""
    if _APPLIED:
        return list(_APPLIED)
    import vllm.model_executor.layers.mamba.mamba_mixer2 as mm2

    mm2.mamba_chunk_scan_combined_varlen = ssd_prefill
    mm2.selective_state_update = ssd_decode
    mm2.causal_conv1d_fn = conv_prefill
    mm2.causal_conv1d_update = conv_decode
    _APPLIED.extend(["mixer2.ssd_prefill", "mixer2.ssd_decode", "mixer2.conv_prefill", "mixer2.conv_decode"])
    import vllm.model_executor.layers.fused_moe.layer as moe_layer

    if not isinstance(moe_layer.current_platform, _GatelessMoEPlatform):
        moe_layer.current_platform = _GatelessMoEPlatform(moe_layer.current_platform,
                                                          moe_layer.FusedMoE.__init__.__code__)
    _APPLIED.append("fused_moe.allow_gateless")
    try:
        import vllm_ascend.ops.fused_moe.moe_mlp as moe_mlp
    except Exception as e:  # noqa: BLE001 - report and continue: dense models do not need it
        _say(f"FAILED to import vllm_ascend moe_mlp, relu2 MoE not patched: {type(e).__name__}: {e}")
    else:
        if not hasattr(moe_mlp.unquant_apply_mlp, "_yeto_orig"):
            moe_mlp.unquant_apply_mlp = make_relu2_unquant_apply_mlp(moe_mlp.unquant_apply_mlp)
        _APPLIED.append("moe_mlp.relu2")
    try:
        from vllm_ascend.ops.fused_moe.fused_moe import AscendSharedFusedMoE
    except Exception as e:  # noqa: BLE001
        _say(f"FAILED to import AscendSharedFusedMoE, ungated shared expert not patched: {type(e).__name__}: {e}")
    else:
        if not hasattr(AscendSharedFusedMoE._shared_experts_part1, "_yeto_orig"):
            AscendSharedFusedMoE._shared_experts_part1 = make_shared_part1(AscendSharedFusedMoE._shared_experts_part1)
        _APPLIED.append("shared_moe.up_proj")
    _say(f"applied {_APPLIED} pid={os.getpid()}")
    return list(_APPLIED)


def register() -> None:
    """``vllm.general_plugins`` entry point. Set YETO_ASCEND_MAMBA2=0 to skip."""
    if os.environ.get("YETO_ASCEND_MAMBA2", "1") == "0":
        _say("skipped (YETO_ASCEND_MAMBA2=0)")
        return
    try:
        apply()
    except Exception as e:  # noqa: BLE001
        _say(f"FAILED: {type(e).__name__}: {e}")
        raise
