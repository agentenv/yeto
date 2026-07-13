"""Sidecar-side CTTN step: compute the outer direction from real curvature.

Runs inside the action-probe sidecar (yeto/action_probe_server.py), the only
merge-time place with a torch model + held-out data + autograd. Given the merged
pseudo-gradient g and the pre-step Nesterov buffer b (flat, in the syncer's
fragment order) plus the trainable LoRA params and held-out panels, it computes
the CTTN direction d = g + mu^2 z and the updated buffer b_new via:

  1. a retained-graph HVP of the (panel-averaged) held-out loss — ONE forward,
     ONE create_graph first-backward, then one second-backward per Lanczos
     column (<=8), all fp32, eager attention (flash/SDPA has no double-backward);
  2. block_lanczos_torch -> (V, T) curvature sketch seeded with {q, r/||r||};
  3. cttn_step_torch -> (d, b_new, diagnostics), where q^T d == ||g|| exactly.

The heavy p-dim work is fp32 on the model device; the <=8-dim solve is the
shared numpy core. Everything here is validated piecewise in scripts/test_cttn.py
(math), scripts/test_hvp_lora.py (HVP), and scripts/cttn_qwen_hvp.py (9B scale);
this module is exercised end-to-end by test_cttn_sidecar (tiny local model).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from yeto.cttn import orth as _np_orth
from yeto.cttn_torch import block_lanczos_torch, cttn_step_torch
from yeto.losses import sft_loss

_WORK_DTYPE = torch.float32


def flatten_params(params) -> torch.Tensor:
    """Concatenate a param tuple into a flat fp32 vector (fragment order)."""
    return torch.cat([p.reshape(-1).to(_WORK_DTYPE) for p in params])


def _unflatten_like(vec: torch.Tensor, params):
    """Split a flat [p] vector into pieces shaped like each param."""
    parts = []
    off = 0
    for p in params:
        n = p.numel()
        parts.append(vec[off:off + n].view_as(p))
        off += n
    return parts


def make_hvp(model, params, panels, *, loss_function: str = "cross_entropy"):
    """Build a retained-graph HVP closure for the panel-averaged held-out loss.

    ``params``: tuple of trainable tensors (in the syncer's fragment order),
    requires_grad=True. ``panels``: sequence of (input_ids [B,T], weights [B,T]).
    The Hessian is of the MEAN per-token loss averaged over panels — E[H] over
    held-out microbatches, matching the design's averaged curvature.

    Returns an ``hvp`` callable mapping [p, m] fp32 -> [p, m] fp32 (columnwise
    H v). ONE forward+first-backward is done here; each hvp() column is a single
    retained-graph second-backward. Do NOT mutate params between calls; call
    ``release()`` (returned) when done to free the graph.
    """
    # Panel-averaged mean-per-token loss, one graph retained for all HVPs.
    total = None
    for input_ids, weights in panels:
        out = model(input_ids=input_ids, use_cache=False)
        loss_sum, ntok = sft_loss(out.logits, input_ids, loss_function, weights)
        lpt = loss_sum / torch.clamp(ntok.to(_WORK_DTYPE), min=1.0)
        total = lpt if total is None else total + lpt
    loss = total / float(len(panels))

    first_grads = torch.autograd.grad(loss, params, create_graph=True)

    def hvp(X: torch.Tensor) -> torch.Tensor:
        X = X.to(dtype=_WORK_DTYPE)
        cols = []
        for j in range(X.shape[1]):
            v_parts = _unflatten_like(X[:, j], params)
            hv = torch.autograd.grad(
                first_grads, params, grad_outputs=v_parts,
                retain_graph=True, allow_unused=False,
            )
            cols.append(torch.cat([h.reshape(-1).to(_WORK_DTYPE) for h in hv]))
        return torch.stack(cols, dim=1)

    def release():
        # Drop references so the retained graph can be freed.
        nonlocal first_grads, loss, total
        del first_grads, loss, total

    return hvp, float(loss.detach()), release


@dataclass
class CttnSidecarResult:
    d: torch.Tensor          # applied direction (theta -= outer_lr * d), fp32 [p]
    b_new: torch.Tensor      # updated Nesterov buffer, fp32 [p]
    diag: object             # CttnResult (bind, tau, retention, energies, ...)
    loss: float              # panel-averaged held-out loss at the eval point


def cttn_sidecar_step(
    model,
    params,
    panels,
    g: torch.Tensor,
    b: torch.Tensor,
    *,
    mu: float,
    rho: float,
    block_steps: int = 4,
    loss_function: str = "cross_entropy",
) -> CttnSidecarResult:
    """Compute the CTTN outer step from real HVP curvature.

    g, b: flat fp32 tensors on the model device, in the SAME order as ``params``
    (the syncer's fragment order). Returns d = g + mu^2 z and b_new. The caller
    (Rust, via the wire) applies params -= outer_lr * d and stores b_new — it must
    NOT feed b_new through the plain-Nesterov materialization.
    """
    dev = g.device
    g = g.to(dtype=_WORK_DTYPE)
    b = b.to(dtype=_WORK_DTYPE)

    hvp, loss_val, release = make_hvp(model, params, panels, loss_function=loss_function)
    try:
        gn = float(torch.linalg.vector_norm(g))
        if gn == 0.0:
            # No merged signal: plain Nesterov fallback (matches cttn_step_torch).
            d, b_new, res = cttn_step_torch(
                g, b, torch.zeros(g.numel(), 1, device=dev, dtype=_WORK_DTYPE),
                torch.zeros(1, 1, device=dev, dtype=_WORK_DTYPE), mu=mu, rho=rho)
            return CttnSidecarResult(d, b_new, res, loss_val)

        q = g / gn
        r = b - q * torch.dot(q, b)
        rnorm = float(torch.linalg.vector_norm(r))
        if rnorm == 0.0:
            # No transverse momentum to damp: d is pure SGD in the parallel dir.
            d, b_new, res = cttn_step_torch(
                g, b, torch.zeros(g.numel(), 1, device=dev, dtype=_WORK_DTYPE),
                torch.zeros(1, 1, device=dev, dtype=_WORK_DTYPE), mu=mu, rho=rho)
            return CttnSidecarResult(d, b_new, res, loss_val)

        # Seed the Krylov space with {q, r/||r||}; block_lanczos in torch (fp32).
        seed_np = _np_orth([q.detach().cpu().numpy(),
                            (r / rnorm).detach().cpu().numpy()])
        Q0 = torch.tensor(seed_np, device=dev, dtype=_WORK_DTYPE)
        V, T = block_lanczos_torch(hvp, Q0, block_steps)
        d, b_new, res = cttn_step_torch(g, b, V, T, mu=mu, rho=rho)
        return CttnSidecarResult(d, b_new, res, loss_val)
    finally:
        release()
