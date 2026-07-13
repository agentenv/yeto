"""Torch-native CTTN step for the action-probe sidecar.

Same math as yeto.cttn (validated in scripts/test_cttn.py) but keeps the p-dim
vectors as torch tensors on the model's device (V is [p, k] with k<=8, so V^T r
and V @ z_coords are the only p-sized ops and belong on-GPU). The tiny <=8-dim
trust-region solve is done in float64 on CPU via the shared numpy helper so the
result is bit-parity with the numpy core.

Used by yeto/action_probe_server.py: it supplies the HVP (double-backward, eager
attention) and the merged g / buffer b; this returns the applied direction d and
the updated buffer b_new.
"""

from __future__ import annotations

import numpy as np
import torch

from yeto.cttn import RHO_DEFAULT, CttnResult, _solve_trustregion_kdim


def block_lanczos_torch(hvp, Q0: torch.Tensor, block_steps: int):
    """Block-Lanczos on the local Hessian, all torch. hvp: [p,m]->[p,m].

    Q0: [p, b0] orthonormal (float32). Returns (V [p,k], T [k,k]) with full
    reorthogonalization (k<=8). Mirrors yeto.cttn.block_lanczos."""
    basis = [Q0[:, j].contiguous() for j in range(Q0.shape[1])]
    cur = Q0
    for _ in range(max(0, block_steps - 1)):
        W = hvp(cur)                                    # [p, bcur]
        for u in basis:
            W = W - torch.outer(u, u @ W)
        # orthonormalize W's columns against nothing new (already reorth'd)
        Qn_cols = []
        for j in range(W.shape[1]):
            w = W[:, j].clone()
            for u in basis + Qn_cols:
                w = w - u * (u @ w)
            n = torch.linalg.norm(w)
            if n > 1e-6 * (torch.linalg.norm(W[:, j]) + 1e-30):
                Qn_cols.append(w / n)
        if not Qn_cols:
            break
        basis.extend(Qn_cols)
        cur = torch.stack(Qn_cols, dim=1)
    V = torch.stack(basis, dim=1)                       # [p, k]
    T = V.T @ hvp(V)                                    # [k, k]
    T = 0.5 * (T + T.T)
    return V, T


def cttn_step_torch(
    g: torch.Tensor,
    b: torch.Tensor,
    V: torch.Tensor,
    T: torch.Tensor,
    *,
    mu: float,
    rho: float = RHO_DEFAULT,
) -> tuple[torch.Tensor, torch.Tensor, CttnResult]:
    """One CTTN step in torch. Returns (d, b_new, diagnostics).

    d, b_new are torch tensors on g's device/dtype; the CttnResult carries the
    scalar diagnostics (bind, tau, retention, energies, ritz, n90) plus numpy
    z for logging. theta -= eta * d ;  store b_new as the new Nesterov buffer."""
    dev, dt = g.device, g.dtype
    gn = float(torch.linalg.norm(g))
    if gn <= 1e-30:
        b_new = mu * b + g
        d = g + mu * b_new
        z0 = torch.zeros_like(g)
        return d, b_new, CttnResult(z0.cpu().numpy(), (mu * b + g).cpu().numpy(),
                                    z0.cpu().numpy(), 0.0, False, 0.0, 0.0, 1.0,
                                    0.0, 0.0, 0.0, np.zeros(0), 0)

    q = g / gn
    r = b - q * (q @ b)                        # transverse buffer
    b_parallel = b - r

    # Project the three p-vectors into the k-dim V basis (tiny outputs).
    qc = (V.T @ q).double().cpu().numpy()
    rc = (V.T @ r).double().cpu().numpy()
    gc = (V.T @ g).double().cpu().numpy()
    Tk = T.double().cpu().numpy()

    z_coords_np, diag = _solve_trustregion_kdim(qc, rc, gc, Tk, mu=mu, rho=rho)

    z_coords = torch.tensor(z_coords_np, device=dev, dtype=dt)
    z = V @ z_coords                            # [p]; q^T z == 0 by construction
    b_new = mu * (b_parallel + z) + g
    d = g + (mu * mu) * z

    r_norm = float(torch.linalg.norm(r))
    z_norm = float(torch.linalg.norm(z))
    res = CttnResult(
        d=d.detach().cpu().numpy(), b_new=b_new.detach().cpu().numpy(),
        z=z.detach().cpu().numpy(), tau=diag["tau"], bind=diag["bind"],
        r_norm=r_norm, z_norm=z_norm,
        norm_retention=(z_norm / r_norm) if r_norm > 1e-30 else 1.0,
        e_before=diag["e_before"], e_after=diag["e_after"], budget=diag["budget"],
        ritz=diag["ritz"], n_modes_90=diag["n_modes_90"],
    )
    return d, b_new, res
