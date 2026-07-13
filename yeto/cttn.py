"""Curvature-Trust Transverse Nesterov (CTTN) — outer-optimizer core.

The theoretically-correct anisotropic curvature-damped outer optimizer for the
DiLoCo momentum poison. Unlike the scalar caps (capped-nesterov, curv) and the
worker-disagreement variant (wsub), CTTN:

  * keeps the parallel step exactly equal to memoryless SGD-0.28
    (q^T d == ||g||), so any improvement over SGD CANNOT be a hidden
    effective-LR change;
  * damps ONLY the transverse component r = P(b) of the momentum buffer,
    per-eigendirection, via a matrix trust region z = (I + tau*A)^-1 r, where
    A = P Hplus P is the PSD-projected local curvature restricted to the
    transverse subspace (real Hessian, via HVPs — not worker disagreement);
  * uses ONE dimensionless constant rho (fraction of the current memoryless
    step's positive-curvature budget the transverse momentum may consume).
    rho is homogeneous of degree two: scaling deltas / inner-LR / loss / eta
    does not change whether the cap binds. The dual variable tau adapts.

This module is HVP-agnostic: the curvature enters only through a small block
of Hessian-vector products, supplied by the caller as a Krylov basis (V) and
its projected tridiagonal/Rayleigh block (T = V^T H V). The heavy autograd
(double-backward) lives on the learner side; this core does only tiny dense
linear algebra on the <=8-dim Ritz block.

See docs/CTTN_DESIGN.md (Codex gpt-5.6-sol design pass, 2026-07-13) for the
derivation and the pre-registered experiment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The one dimensionless constant. rho=0.10: the transverse momentum displacement
# may consume at most 10% of the current memoryless step's positive-curvature
# budget. Used everywhere, never per-H/rank/inner-LR tuned.
RHO_DEFAULT = 0.10
_F64_EPS = np.finfo(np.float64).eps


def _stable_norm(v: np.ndarray) -> float:
    """Euclidean norm without squaring the original scale."""
    if v.size == 0:
        return 0.0
    scale = float(np.max(np.abs(v)))
    if scale == 0.0:
        return 0.0
    return scale * float(np.linalg.norm(v / scale))


def _scale_relative_tol(*values: float, factor: float = 256.0) -> float:
    """Roundoff allowance with no absolute floor (important near 1e-40)."""
    scale = max((abs(float(value)) for value in values), default=0.0)
    return factor * _F64_EPS * scale


def project_out(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """P(v) = v - q (q^T v), with q a unit vector. Removes the parallel part."""
    return v - q * float(q @ v)


def orth(cols: list[np.ndarray]) -> np.ndarray:
    """Modified Gram-Schmidt orthonormal basis of the given columns.

    Drops columns whose residual norm is ~0 (rank-deficient seed). Returns a
    [p, k] matrix with orthonormal columns (k <= len(cols))."""
    basis: list[np.ndarray] = []
    for c in cols:
        w = c.astype(np.float64, copy=True)
        original_norm = _stable_norm(w)
        for _ in range(2):
            for b in basis:
                w = w - b * float(b @ w)
        n = _stable_norm(w)
        rel_tol = 32.0 * _F64_EPS * max(1, len(basis))
        if np.isfinite(n) and n > rel_tol * original_norm:
            basis.append(w / n)
    if not basis:
        return np.zeros((cols[0].shape[0], 0), dtype=np.float64)
    return np.stack(basis, axis=1)


def block_lanczos(hvp, Q0: np.ndarray, block_steps: int):
    """Block-Lanczos Krylov basis for the local Hessian, seeded with Q0.

    ``hvp`` maps [p, m] -> [p, m] (columnwise Hessian-vector products). ``Q0``
    is a [p, b0] orthonormal seed (here {q, r/||r||}). Returns (V, T) with V a
    [p, k] orthonormal basis (k <= b0 * block_steps) and T = V^T H V the [k, k]
    Rayleigh block. Full reorthogonalization (k is tiny, <=8) for numerical
    stability in the near-flat LoRA spectrum.

    Every computed H@Q block is cached and reused to assemble T, so the routine
    performs exactly one HVP per returned basis column. With b0=2 and
    block_steps=4 this is at most 8 columnwise HVPs, including T construction.
    Exhausted or invariant Krylov directions are dropped rather than normalized
    from roundoff."""
    Q0 = np.asarray(Q0, dtype=np.float64)
    assert Q0.ndim == 2 and Q0.shape[1] > 0
    assert np.all(np.isfinite(Q0)), "Q0 must contain only finite columns"

    basis_blocks = [Q0]
    basis = [Q0[:, j].copy() for j in range(Q0.shape[1])]
    hv_blocks: list[np.ndarray] = []
    cur = Q0
    for _ in range(max(0, block_steps - 1)):
        Hcur = np.asarray(hvp(cur), dtype=np.float64)  # [p, bcur]
        assert Hcur.shape == cur.shape
        assert np.all(np.isfinite(Hcur)), "HVP returned non-finite values"
        hv_blocks.append(Hcur)

        Qn_cols: list[np.ndarray] = []
        for j in range(Hcur.shape[1]):
            original = Hcur[:, j]
            original_norm = _stable_norm(original)
            w = original.copy()
            # Two-pass modified Gram-Schmidt against every accumulated column,
            # including significant columns accepted earlier in this block.
            for _ in range(2):
                for u in basis + Qn_cols:
                    w = w - u * float(u @ w)
            residual_norm = _stable_norm(w)
            rel_tol = 32.0 * _F64_EPS * max(1, len(basis) + len(Qn_cols))
            if (np.isfinite(residual_norm)
                    and residual_norm > rel_tol * original_norm):
                Qn_cols.append(w / residual_norm)

        if not Qn_cols:
            break
        Qn = np.stack(Qn_cols, axis=1)
        basis_blocks.append(Qn)
        basis.extend(Qn_cols)
        cur = Qn

    # If growth reached its requested depth, the last accepted block has not
    # yet been multiplied by H. Compute it once; all earlier products are cached.
    if len(hv_blocks) < len(basis_blocks):
        Hcur = np.asarray(hvp(basis_blocks[-1]), dtype=np.float64)
        assert Hcur.shape == basis_blocks[-1].shape
        assert np.all(np.isfinite(Hcur)), "HVP returned non-finite values"
        hv_blocks.append(Hcur)

    V = np.concatenate(basis_blocks, axis=1)          # [p, k]
    HV = np.concatenate(hv_blocks, axis=1)            # [p, k]
    T = V.T @ HV                                      # [k, k]
    T = 0.5 * (T + T.T)
    gram_error = float(np.linalg.norm(V.T @ V - np.eye(V.shape[1])))
    orth_tol = max(1e-12, 128.0 * _F64_EPS * max(1, V.shape[1]))
    assert np.all(np.isfinite(V)), "block-Lanczos basis contains non-finite values"
    assert np.all(np.isfinite(T)), "block-Lanczos Rayleigh block is non-finite"
    assert gram_error <= orth_tol, (
        f"block-Lanczos basis lost orthogonality: {gram_error} > {orth_tol}"
    )
    return V, T


def _projected_psd_operator(qc: np.ndarray, T: np.ndarray):
    """Return (P, Tplus, P Tplus P) in the small Krylov coordinates."""
    T = np.asarray(T, dtype=np.float64)
    T = 0.5 * (T + T.T)
    evals, evecs = np.linalg.eigh(T)
    Tplus = (evecs * np.clip(evals, 0.0, None)) @ evecs.T

    qc = np.asarray(qc, dtype=np.float64)
    qnorm = _stable_norm(qc)
    qn = qc / qnorm if qnorm > 0.0 else np.zeros_like(qc)
    Pc = np.eye(qc.shape[0]) - np.outer(qn, qn)
    Ac = Pc @ Tplus @ Pc
    Ac = 0.5 * (Ac + Ac.T)
    if Ac.size:
        aevals, aevecs = np.linalg.eigh(Ac)
        ascale = float(np.max(np.abs(aevals)))
        atol = _scale_relative_tol(ascale, factor=256.0)
        aevals = np.where(aevals > atol, aevals, 0.0)
        Ac = (aevecs * aevals) @ aevecs.T
        Ac = 0.5 * (Ac + Ac.T)
    return Pc, Tplus, Ac


def _transverse_energy_kdim(qc: np.ndarray, zc: np.ndarray, T: np.ndarray) -> float:
    """Curvature energy after explicitly projecting small coordinates."""
    Pc, _, Ac = _projected_psd_operator(qc, T)
    zc = Pc @ np.asarray(zc, dtype=np.float64)
    energy = float(zc @ (Ac @ zc))
    scale = float(np.max(np.abs(Ac))) * _stable_norm(zc) ** 2 if Ac.size else 0.0
    tol = _scale_relative_tol(energy, scale)
    assert energy >= -tol, f"PSD transverse energy became negative: {energy}"
    if abs(energy) <= tol:
        return 0.0
    return max(energy, 0.0)


def _assert_trust_postcondition(
    mu: float,
    e_after: float,
    budget: float,
    *,
    working_eps: float = _F64_EPS,
) -> None:
    """Check the trust cap at the precision used for p-dimensional work."""
    lhs = float(mu) ** 4 * e_after
    if not np.isfinite(working_eps) or working_eps <= 0.0:
        raise ValueError("working_eps must be finite and positive")
    scale = max(abs(lhs), abs(float(budget)))
    tol = 64.0 * float(working_eps) * scale
    assert lhs <= budget + tol, (
        f"trust-region postcondition violated: {lhs} > {budget} + {tol}"
    )


def _solve_trustregion_kdim(
    qc,
    rc,
    gc,
    T,
    *,
    mu,
    rho,
    working_eps: float = _F64_EPS,
):
    """Shared <=k-dim CTTN trust-region solve (numpy; k<=8).

    qc, rc, gc : [k] coordinates of q, r, g in the orthonormal V basis
                 (q = g/||g|| parallel dir, r = P(b) transverse buffer).
    T          : [k, k] Rayleigh block V^T H V.
    Returns (z_coords [k], diag) where z = V @ z_coords is the damped transverse
    momentum and diag carries tau/bind/energies/budget/ritz/n_modes_90.

    A = P Hplus P (q-direction projected out on both sides) keeps z transverse so
    q^T z == 0 exactly; the budget uses the full current-step curvature g^T Hplus g.
    """
    if rho < 0.0:
        raise ValueError("rho must be non-negative")

    qc = np.asarray(qc, dtype=np.float64)
    rc = np.asarray(rc, dtype=np.float64)
    gc = np.asarray(gc, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    assert qc.ndim == rc.ndim == gc.ndim == 1
    assert qc.shape == rc.shape == gc.shape
    assert T.shape == (qc.shape[0], qc.shape[0])
    assert all(np.all(np.isfinite(x)) for x in (qc, rc, gc, T))

    Pc, Tplus, Ac = _projected_psd_operator(qc, T)
    rc = Pc @ rc                               # structural transversality
    muA, W = np.linalg.eigh(Ac)
    eig_scale = float(np.max(np.abs(muA))) if muA.size else 0.0
    eig_tol = _scale_relative_tol(eig_scale, factor=256.0)
    muA = np.where(muA > eig_tol, muA, 0.0)

    rj = W.T @ rc                              # r in the A-eigenbasis (rc perp qc)
    e_before = float(np.sum(muA * rj * rj))    # r^T A r
    g_curv_raw = float(gc @ (Tplus @ gc))      # g^T Hplus g
    g_scale = (float(np.max(np.abs(Tplus))) * _stable_norm(gc) ** 2
               if Tplus.size else 0.0)
    g_tol = _scale_relative_tol(g_curv_raw, g_scale)
    assert g_curv_raw >= -g_tol, f"PSD gradient curvature became negative: {g_curv_raw}"
    g_curv = max(g_curv_raw, 0.0)
    budget = rho * g_curv
    mu4 = float(mu) ** 4

    active = muA > 0.0
    lambda_scale = float(np.max(muA)) if np.any(active) else 0.0
    lambda_scaled = muA[active] / lambda_scale if lambda_scale > 0.0 else muA[active]

    def energy_scaled(sigma: float) -> float:
        denom = 1.0 + sigma * lambda_scaled
        z_active = rj[active] / denom
        return float(np.sum(muA[active] * z_active * z_active))

    lhs_before = mu4 * e_before
    gate_tol = _scale_relative_tol(lhs_before, budget)
    if mu4 == 0.0 or e_before == 0.0 or lhs_before <= budget + gate_tol:
        tau = 0.0
        bind = False
        z_eig = rj.copy()
    elif budget == 0.0:
        # The finite-tau limit for a zero target: remove every positive-
        # curvature component and retain A's nullspace exactly.
        tau = np.inf
        bind = True
        z_eig = rj.copy()
        z_eig[active] = 0.0
    else:
        bind = True
        target = budget / mu4
        target_tol = _scale_relative_tol(e_before, target)
        assert target > 0.0, "positive-budget branch requires a positive target"
        assert target < e_before + target_tol
        # Land just inside the cap so eigenspace reconstruction and the final
        # full-space projection cannot round an equality to the wrong side.
        solve_target = target * (1.0 - 32768.0 * _F64_EPS)
        assert solve_target > 0.0

        # Solve in sigma=tau*lambda_max so 1e-40 and 1e+20 curvature have the
        # same well-scaled bracket. A positive target must be explicitly bracketed.
        lo, hi = 0.0, 1.0
        bracketed = False
        for _ in range(2048):
            if energy_scaled(hi) <= solve_target:
                bracketed = True
                break
            hi *= 2.0
        assert bracketed, "failed to bracket positive trust-region target"
        for _ in range(256):
            mid = 0.5 * (lo + hi)
            if energy_scaled(mid) > solve_target:
                lo = mid
            else:
                hi = mid
        tau = hi / lambda_scale
        z_eig = rj.copy()
        z_eig[active] = rj[active] / (1.0 + hi * lambda_scaled)

    z_coords = Pc @ (W @ z_eig)
    e_after = float(z_coords @ (Ac @ z_coords))
    e_after_scale = (float(np.max(np.abs(Ac))) * _stable_norm(z_coords) ** 2
                     if Ac.size else 0.0)
    e_after_tol = _scale_relative_tol(e_after, e_after_scale)
    assert e_after >= -e_after_tol, f"PSD transverse energy became negative: {e_after}"
    e_after = 0.0 if abs(e_after) <= e_after_tol else max(e_after, 0.0)
    _assert_trust_postcondition(mu, e_after, budget, working_eps=working_eps)

    contrib = muA * rj * rj                     # Ritz-mode concentration of e_before
    order = np.argsort(contrib)[::-1]
    csum = np.cumsum(contrib[order])
    total = csum[-1] if csum.size else 0.0
    n90 = int(np.searchsorted(csum, 0.9 * total) + 1) if total > 0.0 else 0

    diag = {"tau": tau, "bind": bind, "e_before": e_before, "e_after": e_after,
            "budget": budget, "ritz": muA, "n_modes_90": n90}
    return z_coords, diag


def _solve_scalar_trustregion_kdim(
    qc,
    rc,
    gc,
    T,
    *,
    mu,
    rho,
    working_eps: float = _F64_EPS,
):
    """Isotropic scalar-HVP control using the same curvature budget as CTTN."""
    if rho < 0.0:
        raise ValueError("rho must be non-negative")

    qc = np.asarray(qc, dtype=np.float64)
    rc = np.asarray(rc, dtype=np.float64)
    gc = np.asarray(gc, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    assert qc.ndim == rc.ndim == gc.ndim == 1
    assert qc.shape == rc.shape == gc.shape
    assert T.shape == (qc.shape[0], qc.shape[0])
    assert all(np.all(np.isfinite(x)) for x in (qc, rc, gc, T))

    Pc, Tplus, Ac = _projected_psd_operator(qc, T)
    rc = Pc @ rc
    muA, W = np.linalg.eigh(Ac)
    eig_scale = float(np.max(np.abs(muA))) if muA.size else 0.0
    eig_tol = _scale_relative_tol(eig_scale, factor=256.0)
    muA = np.where(muA > eig_tol, muA, 0.0)

    e_before = float(rc @ (Ac @ rc))
    e_before_scale = (
        float(np.max(np.abs(Ac))) * _stable_norm(rc) ** 2 if Ac.size else 0.0
    )
    e_before_tol = _scale_relative_tol(e_before, e_before_scale)
    assert e_before >= -e_before_tol, (
        f"PSD transverse energy became negative: {e_before}"
    )
    e_before = 0.0 if abs(e_before) <= e_before_tol else max(e_before, 0.0)

    g_curv_raw = float(gc @ (Tplus @ gc))
    g_scale = (
        float(np.max(np.abs(Tplus))) * _stable_norm(gc) ** 2
        if Tplus.size
        else 0.0
    )
    g_tol = _scale_relative_tol(g_curv_raw, g_scale)
    assert g_curv_raw >= -g_tol, (
        f"PSD gradient curvature became negative: {g_curv_raw}"
    )
    budget = rho * max(g_curv_raw, 0.0)
    mu4 = float(mu) ** 4
    lhs_before = mu4 * e_before
    gate_tol = _scale_relative_tol(lhs_before, budget)

    if mu4 == 0.0 or e_before == 0.0 or lhs_before <= budget + gate_tol:
        alpha = 1.0
        bind = False
        tau = 0.0
    elif budget == 0.0:
        alpha = 0.0
        bind = True
        tau = np.inf
    else:
        alpha = float(np.sqrt(budget / lhs_before))
        bind = True
        tau = (1.0 / alpha) - 1.0

    z_coords = Pc @ (alpha * rc)
    e_after = float(z_coords @ (Ac @ z_coords))
    e_after_scale = (
        float(np.max(np.abs(Ac))) * _stable_norm(z_coords) ** 2
        if Ac.size
        else 0.0
    )
    e_after_tol = _scale_relative_tol(e_after, e_after_scale)
    assert e_after >= -e_after_tol, f"PSD transverse energy became negative: {e_after}"
    e_after = 0.0 if abs(e_after) <= e_after_tol else max(e_after, 0.0)
    _assert_trust_postcondition(mu, e_after, budget, working_eps=working_eps)

    rj = W.T @ rc
    contrib = muA * rj * rj
    order = np.argsort(contrib)[::-1]
    csum = np.cumsum(contrib[order])
    total = csum[-1] if csum.size else 0.0
    n90 = int(np.searchsorted(csum, 0.9 * total) + 1) if total > 0.0 else 0
    diag = {
        "tau": tau,
        "bind": bind,
        "e_before": e_before,
        "e_after": e_after,
        "budget": budget,
        "ritz": muA,
        "n_modes_90": n90,
        "alpha": alpha,
    }
    return z_coords, diag


@dataclass
class CttnResult:
    d: np.ndarray            # the applied outer direction; theta -= eta * d
    b_new: np.ndarray        # updated Nesterov buffer
    z: np.ndarray            # damped transverse momentum (in param space)
    tau: float               # trust-region dual variable (0.0 => cap inactive)
    bind: bool               # did the trust region bind?
    # diagnostics (for the pre-registered instrumentation)
    r_norm: float            # ||r|| = ||P(b)|| before damping
    z_norm: float            # ||z|| after damping
    norm_retention: float    # ||z|| / ||r||
    e_before: float          # r^T Hplus r  (transverse curvature energy in)
    e_after: float           # z^T Hplus z  (after damping) — bounded by budget/mu^4
    budget: float            # B = rho * g^T Hplus g
    ritz: np.ndarray         # PSD Ritz values (curvature eigenvalues in-basis)
    n_modes_90: int          # # Ritz modes explaining 90% of e_before


def cttn_step(
    g: np.ndarray,
    b: np.ndarray,
    V: np.ndarray,
    T: np.ndarray,
    *,
    mu: float,
    rho: float = RHO_DEFAULT,
) -> CttnResult:
    """One CTTN outer step (dense/numpy core).

    Parameters
    ----------
    g : [p] merged pseudo-gradient (mean of worker deltas), gradient-sign
        convention (theta -= eta * d).
    b : [p] incoming Nesterov buffer (transverse part will be curvature-damped).
    V : [p, k] orthonormal Krylov basis from block-Lanczos on the local Hessian,
        seeded with {q, r/||r||}. Must contain q and the r direction in its span.
    T : [k, k] symmetric Rayleigh block V^T H V (H the held-out Hessian at the
        current global). PSD-projected internally (negative curvature -> 0).
    mu : Nesterov momentum coefficient (e.g. 0.9).
    rho : dimensionless curvature-budget fraction (default 0.10).

    Returns CttnResult with the applied direction d = g + mu^2 z and diagnostics.

    Guarantees (verified in test_cttn.py against a synthetic Hessian):
      * q^T d == ||g||  (parallel step is exact SGD; no effective-LR change);
      * mu^4 z^T A z <= rho g^T Hplus g  (transverse curvature trust region);
      * flat Ritz modes of r are preserved, sharp modes shrunk 1/(1+tau*lam).
    """
    g = g.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    V = V.astype(np.float64, copy=False)
    T = T.astype(np.float64, copy=False)
    gn = _stable_norm(g)
    if gn == 0.0:
        # Degenerate: no merged signal. Fall back to plain Nesterov.
        b_new = mu * b + g
        d = g + mu * b_new
        return CttnResult(d, b_new, np.zeros_like(g), 0.0, False,
                          0.0, 0.0, 1.0, 0.0, 0.0, 0.0, np.zeros(0), 0)

    q = g / gn
    r = project_out(b, q)              # transverse buffer
    b_parallel = b - r
    r_norm = _stable_norm(r)

    # Project into the k-dim V basis (q, r, g all lie in span(V)) and solve the
    # <=k-dim trust region with the shared core (identical for the torch path).
    qc = V.T @ q
    rc = V.T @ r
    gc = V.T @ g
    z_coords, diag = _solve_trustregion_kdim(qc, rc, gc, T, mu=mu, rho=rho)

    z = V @ z_coords
    z = project_out(z, q)              # final full-space transversality guard
    z_norm = _stable_norm(z)
    z_coords_final = V.T @ z
    diag["e_after"] = _transverse_energy_kdim(qc, z_coords_final, T)
    _assert_trust_postcondition(mu, diag["e_after"], diag["budget"])
    b_new = mu * (b_parallel + z) + g
    d = g + mu * mu * z                 # == g + mu * P(b_new); q^T d == ||g||
    qtd = float(q @ d)
    qtd_tol = _scale_relative_tol(qtd, gn, factor=1024.0 * max(1, g.size))
    assert abs(qtd - gn) <= qtd_tol, (
        f"parallel-step invariant violated: q^T d={qtd}, ||g||={gn}"
    )

    return CttnResult(
        d=d, b_new=b_new, z=z, tau=diag["tau"], bind=diag["bind"],
        r_norm=r_norm, z_norm=z_norm,
        norm_retention=(z_norm / r_norm) if r_norm > 0.0 else 1.0,
        e_before=diag["e_before"], e_after=diag["e_after"], budget=diag["budget"],
        ritz=diag["ritz"], n_modes_90=diag["n_modes_90"],
    )


def cttn_scalar_step(
    g: np.ndarray,
    b: np.ndarray,
    V: np.ndarray,
    T: np.ndarray,
    *,
    mu: float,
    rho: float = RHO_DEFAULT,
) -> CttnResult:
    """Scalar-HVP control: apply one isotropic shrink ``z = alpha * P(b)``.

    The scalar control uses the same HVP sketch and the same curvature budget
    as CTTN. Its only difference is that every transverse direction shares one
    shrink factor instead of receiving per-eigendirection damping.
    """
    g = g.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    V = V.astype(np.float64, copy=False)
    T = T.astype(np.float64, copy=False)
    gn = _stable_norm(g)
    if gn == 0.0:
        b_new = mu * b + g
        d = g + mu * b_new
        return CttnResult(
            d,
            b_new,
            np.zeros_like(g),
            0.0,
            False,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            np.zeros(0),
            0,
        )

    q = g / gn
    r = project_out(b, q)
    b_parallel = b - r
    r_norm = _stable_norm(r)
    qc = V.T @ q
    rc = V.T @ r
    gc = V.T @ g
    z_coords, diag = _solve_scalar_trustregion_kdim(
        qc, rc, gc, T, mu=mu, rho=rho
    )

    z = project_out(V @ z_coords, q)
    z_norm = _stable_norm(z)
    z_coords_final = V.T @ z
    diag["e_after"] = _transverse_energy_kdim(qc, z_coords_final, T)
    _assert_trust_postcondition(mu, diag["e_after"], diag["budget"])
    b_new = mu * (b_parallel + z) + g
    d = g + mu * mu * z
    qtd = float(q @ d)
    qtd_tol = _scale_relative_tol(qtd, gn, factor=1024.0 * max(1, g.size))
    assert abs(qtd - gn) <= qtd_tol, (
        f"parallel-step invariant violated: q^T d={qtd}, ||g||={gn}"
    )

    return CttnResult(
        d=d,
        b_new=b_new,
        z=z,
        tau=diag["tau"],
        bind=diag["bind"],
        r_norm=r_norm,
        z_norm=z_norm,
        norm_retention=(z_norm / r_norm) if r_norm > 0.0 else 1.0,
        e_before=diag["e_before"],
        e_after=diag["e_after"],
        budget=diag["budget"],
        ritz=diag["ritz"],
        n_modes_90=diag["n_modes_90"],
    )
