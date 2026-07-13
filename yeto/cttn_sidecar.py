"""Sidecar-side CTTN step: compute the outer direction from real curvature.

Runs inside the action-probe sidecar (yeto/action_probe_server.py), the only
merge-time place with a torch model + held-out data + autograd. Given the merged
pseudo-gradient g and the pre-step Nesterov buffer b (flat, in the syncer's
fragment order) plus the trainable LoRA params and held-out panels, it computes
the CTTN direction d = g + mu^2 z and the updated buffer b_new via:

  1. a panel-streamed HVP of the panel-averaged held-out loss. Only one panel's
     autograd graph is live at a time; within that panel one forward and one
     create_graph first-backward are reused across every requested HVP column;
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
    """Build a panel-streamed HVP closure for the averaged held-out loss.

    ``params``: tuple of trainable tensors (in the syncer's fragment order),
    requires_grad=True. ``panels``: sequence of (input_ids [B,T], weights [B,T]).
    The Hessian is of the MEAN per-token loss averaged over panels — E[H] over
    held-out microbatches, matching the design's averaged curvature.

    Returns an ``hvp`` callable mapping [p, m] fp32 -> [p, m] fp32. Each call
    visits the panels one at a time and accumulates their Hessian-vector
    products, so peak activation memory is one panel graph rather than all
    panels. A panel's forward and first backward are retained across every
    column in that call. Do NOT mutate params between calls; call ``release()``
    when done.
    """
    panels = tuple(panels)
    if not panels:
        raise ValueError("make_hvp requires at least one held-out panel")

    # Loss reporting does not need a graph. Evaluate panels independently so
    # this pass also keeps peak memory at one panel.
    loss_total = 0.0
    with torch.no_grad():
        for input_ids, weights in panels:
            out = model(input_ids=input_ids, use_cache=False)
            loss_sum, ntok = sft_loss(out.logits, input_ids, loss_function, weights)
            lpt = loss_sum / torch.clamp(ntok.to(_WORK_DTYPE), min=1.0)
            loss_total += float(lpt)
    loss_value = loss_total / float(len(panels))
    released = False

    def hvp(X: torch.Tensor) -> torch.Tensor:
        if released:
            raise RuntimeError("HVP closure has been released")
        if X.ndim != 2 or X.shape[1] == 0:
            raise ValueError("HVP input must have shape [p, m] with m > 0")
        X = X.to(dtype=_WORK_DTYPE)
        accumulated = torch.zeros_like(X)
        panel_scale = 1.0 / float(len(panels))
        for input_ids, weights in panels:
            out = model(input_ids=input_ids, use_cache=False)
            loss_sum, ntok = sft_loss(out.logits, input_ids, loss_function, weights)
            panel_loss = loss_sum / torch.clamp(ntok.to(_WORK_DTYPE), min=1.0)
            first_grads = torch.autograd.grad(panel_loss, params, create_graph=True)

            for j in range(X.shape[1]):
                v_parts = _unflatten_like(X[:, j], params)
                hv = torch.autograd.grad(
                    first_grads,
                    params,
                    grad_outputs=v_parts,
                    retain_graph=j + 1 < X.shape[1],
                    allow_unused=False,
                )
                accumulated[:, j].add_(
                    torch.cat([h.reshape(-1).to(_WORK_DTYPE) for h in hv]),
                    alpha=panel_scale,
                )
            del first_grads, panel_loss, loss_sum, out
        return accumulated

    def release():
        nonlocal released
        released = True

    return hvp, loss_value, release


@dataclass
class CttnSidecarResult:
    d: torch.Tensor          # applied direction (theta -= outer_lr * d), fp32 [p]
    b_new: torch.Tensor      # updated Nesterov buffer, fp32 [p]
    diag: object             # CttnResult (bind, tau, retention, energies, ...)
    loss: float              # panel-averaged held-out loss at the eval point


@dataclass
class CttnShadowSidecarResult:
    z_matrix: torch.Tensor   # matrix-damped transverse momentum, fp32 [p]
    z_scalar: torch.Tensor   # scalar-control transverse momentum, fp32 [p]
    matrix_diag: object      # matrix CttnResult diagnostics
    scalar_diag: object      # scalar-control CttnResult diagnostics
    loss: float


def _z_from_direction(g: torch.Tensor, d: torch.Tensor, mu: float) -> torch.Tensor:
    if float(torch.linalg.vector_norm(g)) == 0.0 or mu == 0.0:
        return torch.zeros_like(g, dtype=_WORK_DTYPE)
    return ((d.to(_WORK_DTYPE) - g.to(_WORK_DTYPE)) / (mu * mu)).to(_WORK_DTYPE)


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
    scalar_control: bool = False,
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
                torch.zeros(1, 1, device=dev, dtype=_WORK_DTYPE), mu=mu, rho=rho,
                scalar_control=scalar_control)
            return CttnSidecarResult(d, b_new, res, loss_val)

        q = g / gn
        r = b - q * torch.dot(q, b)
        rnorm = float(torch.linalg.vector_norm(r))
        if rnorm == 0.0:
            # No transverse momentum to damp: d is pure SGD in the parallel dir.
            d, b_new, res = cttn_step_torch(
                g, b, torch.zeros(g.numel(), 1, device=dev, dtype=_WORK_DTYPE),
                torch.zeros(1, 1, device=dev, dtype=_WORK_DTYPE), mu=mu, rho=rho,
                scalar_control=scalar_control)
            return CttnSidecarResult(d, b_new, res, loss_val)

        # Seed the Krylov space with {q, r/||r||}; block_lanczos in torch (fp32).
        seed_np = _np_orth([q.detach().cpu().numpy(),
                            (r / rnorm).detach().cpu().numpy()])
        Q0 = torch.tensor(seed_np, device=dev, dtype=_WORK_DTYPE)
        V, T = block_lanczos_torch(hvp, Q0, block_steps)
        d, b_new, res = cttn_step_torch(
            g, b, V, T, mu=mu, rho=rho, scalar_control=scalar_control
        )
        return CttnSidecarResult(d, b_new, res, loss_val)
    finally:
        release()


def cttn_sidecar_shadow_step(
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
) -> CttnShadowSidecarResult:
    """Compute matrix and scalar would-be z from one shared HVP sketch."""
    dev = g.device
    g = g.to(dtype=_WORK_DTYPE)
    b = b.to(dtype=_WORK_DTYPE)
    hvp, loss_val, release = make_hvp(
        model, params, panels, loss_function=loss_function
    )
    try:
        gn = float(torch.linalg.vector_norm(g))
        if gn == 0.0:
            V = torch.zeros(g.numel(), 1, device=dev, dtype=_WORK_DTYPE)
            T = torch.zeros(1, 1, device=dev, dtype=_WORK_DTYPE)
        else:
            q = g / gn
            r = b - q * torch.dot(q, b)
            rnorm = float(torch.linalg.vector_norm(r))
            if rnorm == 0.0:
                V = torch.zeros(g.numel(), 1, device=dev, dtype=_WORK_DTYPE)
                T = torch.zeros(1, 1, device=dev, dtype=_WORK_DTYPE)
            else:
                seed_np = _np_orth(
                    [q.detach().cpu().numpy(), (r / rnorm).detach().cpu().numpy()]
                )
                Q0 = torch.tensor(seed_np, device=dev, dtype=_WORK_DTYPE)
                V, T = block_lanczos_torch(hvp, Q0, block_steps)

        d_matrix, _, matrix_diag = cttn_step_torch(
            g, b, V, T, mu=mu, rho=rho, scalar_control=False
        )
        d_scalar, _, scalar_diag = cttn_step_torch(
            g, b, V, T, mu=mu, rho=rho, scalar_control=True
        )
        return CttnShadowSidecarResult(
            z_matrix=_z_from_direction(g, d_matrix, mu),
            z_scalar=_z_from_direction(g, d_scalar, mu),
            matrix_diag=matrix_diag,
            scalar_diag=scalar_diag,
            loss=loss_val,
        )
    finally:
        release()
