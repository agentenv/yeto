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

from yeto.cttn import (
    RHO_DEFAULT,
    CttnResult,
    _assert_trust_postcondition,
    _solve_scalar_trustregion_kdim,
    _solve_trustregion_kdim,
    _transverse_energy_kdim,
)


_F32_EPS = torch.finfo(torch.float32).eps


def _stable_norm_float32(v: torch.Tensor) -> torch.Tensor:
    """Float32 Euclidean norm without overflow/underflow from squaring v."""
    if v.numel() == 0:
        return torch.zeros((), device=v.device, dtype=torch.float32)
    scale = torch.max(torch.abs(v))
    if float(scale) == 0.0:
        return torch.zeros((), device=v.device, dtype=torch.float32)
    return scale * torch.linalg.vector_norm(v / scale)


def block_lanczos_torch(hvp, Q0: torch.Tensor, block_steps: int):
    """Block-Lanczos on the local Hessian, all torch. hvp: [p,m]->[p,m].

    Q0: [p, b0] orthonormal. The p-dimensional work is always float32.
    Returns (V [p,k], T [k,k]) with two-pass full reorthogonalization (k<=8).
    H@Q blocks are cached, so there is exactly one HVP per returned column.
    Mirrors yeto.cttn.block_lanczos."""
    assert Q0.ndim == 2 and Q0.shape[1] > 0
    work_dtype = torch.float32
    Q0 = Q0.to(dtype=work_dtype)
    assert bool(torch.all(torch.isfinite(Q0))), "Q0 must contain only finite columns"

    basis_blocks = [Q0]
    basis = [Q0[:, j].contiguous() for j in range(Q0.shape[1])]
    hv_blocks = []
    cur = Q0
    for _ in range(max(0, block_steps - 1)):
        Hcur = hvp(cur).to(device=Q0.device, dtype=work_dtype)
        assert Hcur.shape == cur.shape
        assert bool(torch.all(torch.isfinite(Hcur))), "HVP returned non-finite values"
        hv_blocks.append(Hcur)

        Qn_cols = []
        for j in range(Hcur.shape[1]):
            original = Hcur[:, j]
            original_norm = float(_stable_norm_float32(original))
            w = original.clone()
            for _ in range(2):
                for u in basis + Qn_cols:
                    w = w - u * torch.dot(u, w)
            residual_norm_t = _stable_norm_float32(w)
            residual_norm = float(residual_norm_t)
            rel_tol = 32.0 * _F32_EPS * max(1, len(basis) + len(Qn_cols))
            if np.isfinite(residual_norm) and residual_norm > rel_tol * original_norm:
                Qn_cols.append(w / residual_norm_t)
        if not Qn_cols:
            break
        Qn = torch.stack(Qn_cols, dim=1)
        basis_blocks.append(Qn)
        basis.extend(Qn_cols)
        cur = Qn

    if len(hv_blocks) < len(basis_blocks):
        Hcur = hvp(basis_blocks[-1]).to(device=Q0.device, dtype=work_dtype)
        assert Hcur.shape == basis_blocks[-1].shape
        assert bool(torch.all(torch.isfinite(Hcur))), "HVP returned non-finite values"
        hv_blocks.append(Hcur)

    V = torch.cat(basis_blocks, dim=1)                  # [p, k]
    HV = torch.cat(hv_blocks, dim=1)                    # [p, k]
    T = V.T @ HV                                        # [k, k]
    T = 0.5 * (T + T.T)
    gram = V.T @ V - torch.eye(V.shape[1], device=V.device, dtype=work_dtype)
    gram_error = float(torch.linalg.matrix_norm(gram))
    orth_tol = max(5e-5, 128.0 * _F32_EPS * max(1, V.shape[1]))
    assert bool(torch.all(torch.isfinite(V))), "block-Lanczos basis is non-finite"
    assert bool(torch.all(torch.isfinite(T))), "block-Lanczos Rayleigh block is non-finite"
    assert gram_error <= orth_tol, (
        f"block-Lanczos basis lost orthogonality: {gram_error} > {orth_tol}"
    )
    return V, T


def cttn_step_torch(
    g: torch.Tensor,
    b: torch.Tensor,
    V: torch.Tensor,
    T: torch.Tensor,
    *,
    mu: float,
    rho: float = RHO_DEFAULT,
    scalar_control: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, CttnResult]:
    """One CTTN step in torch. Returns (d, b_new, diagnostics).

    d, b_new are torch tensors on g's device/dtype; the CttnResult carries the
    scalar diagnostics (bind, tau, retention, energies, ritz, n90) plus numpy
    z for logging. theta -= eta * d ;  store b_new as the new Nesterov buffer."""
    assert g.device == b.device == V.device == T.device, (
        "g, b, V, and T must share a device"
    )
    dev, original_dtype = g.device, g.dtype
    work_dtype = torch.float32
    gw = g.to(dtype=work_dtype)
    bw = b.to(dtype=work_dtype)
    Vw = V.to(dtype=work_dtype)
    Tw = T.to(dtype=work_dtype)

    gn_t = _stable_norm_float32(gw)
    gn = float(gn_t)
    if gn == 0.0:
        b_new_work = mu * bw + gw
        d_work = gw + mu * b_new_work
        z0 = torch.zeros_like(gw)
        d = d_work.to(dtype=original_dtype)
        b_new = b_new_work.to(dtype=original_dtype)
        return d, b_new, CttnResult(
            d.detach().float().cpu().numpy(),
            b_new.detach().float().cpu().numpy(),
            z0.detach().float().cpu().numpy(),
            0.0, False, 0.0, 0.0, 1.0,
            0.0, 0.0, 0.0, np.zeros(0), 0,
        )

    q = gw / gn_t
    r = bw - q * torch.dot(q, bw)               # transverse buffer
    b_parallel = bw - r

    # Project the three p-vectors into the k-dim V basis (tiny outputs).
    qc = (Vw.T @ q).detach().double().cpu().numpy()
    rc = (Vw.T @ r).detach().double().cpu().numpy()
    gc = (Vw.T @ gw).detach().double().cpu().numpy()
    Tk = Tw.detach().double().cpu().numpy()

    solve = (
        _solve_scalar_trustregion_kdim
        if scalar_control
        else _solve_trustregion_kdim
    )
    z_coords_np, diag = solve(
        qc,
        rc,
        gc,
        Tk,
        mu=mu,
        rho=rho,
        working_eps=_F32_EPS,
    )

    z_coords = torch.tensor(z_coords_np, device=dev, dtype=work_dtype)
    z = Vw @ z_coords
    z = z - q * torch.dot(q, z)                 # final full-space projection
    r_norm = float(_stable_norm_float32(r))
    z_norm = float(_stable_norm_float32(z))
    z_coords_final = (Vw.T @ z).detach().double().cpu().numpy()
    diag["e_after"] = _transverse_energy_kdim(qc, z_coords_final, Tk)
    _assert_trust_postcondition(
        mu,
        diag["e_after"],
        diag["budget"],
        working_eps=_F32_EPS,
    )

    b_new_work = mu * (b_parallel + z) + gw
    d_work = gw + (mu * mu) * z
    qtd = float(torch.dot(q, d_work))
    qtd_tol = 512.0 * _F32_EPS * max(abs(qtd), abs(gn)) * max(1, g.numel())
    assert abs(qtd - gn) <= qtd_tol, (
        f"parallel-step invariant violated: q^T d={qtd}, ||g||={gn}"
    )

    # The public tensors retain the caller's dtype; every p-dimensional
    # computation above, including reconstruction and dot products, was fp32.
    d = d_work.to(dtype=original_dtype)
    b_new = b_new_work.to(dtype=original_dtype)
    res = CttnResult(
        d=d.detach().float().cpu().numpy(),
        b_new=b_new.detach().float().cpu().numpy(),
        z=z.detach().float().cpu().numpy(),
        tau=diag["tau"], bind=diag["bind"],
        r_norm=r_norm, z_norm=z_norm,
        norm_retention=(z_norm / r_norm) if r_norm > 0.0 else 1.0,
        e_before=diag["e_before"], e_after=diag["e_after"], budget=diag["budget"],
        ritz=diag["ritz"], n_modes_90=diag["n_modes_90"],
    )
    return d, b_new, res
