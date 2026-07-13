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
_EPS = 1e-30


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
        for b in basis:
            w = w - b * float(b @ w)
        n = float(np.linalg.norm(w))
        if n > 1e-12 * (float(np.linalg.norm(c)) + _EPS):
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

    HVP budget: b0*(block_steps-1) to grow the basis, then one T = V^T (H V)
    block of k more — i.e. ~b0*block_steps total. Seed with b0=2, block_steps=4
    for the design's 8-HVP-per-round point (the +k for T can reuse the last
    block's products in a Lanczos recurrence; kept explicit here for clarity)."""
    basis = [Q0[:, j].astype(np.float64) for j in range(Q0.shape[1])]
    cur = Q0.astype(np.float64)
    for _ in range(max(0, block_steps - 1)):
        W = hvp(cur)                                  # [p, bcur]
        # full reorthogonalization against the accumulated basis
        for u in basis:
            W = W - np.outer(u, u @ W)
        Qn = orth([W[:, j] for j in range(W.shape[1])])
        if Qn.shape[1] == 0:
            break
        for j in range(Qn.shape[1]):
            basis.append(Qn[:, j])
        cur = Qn
    V = np.stack(basis, axis=1)                       # [p, k]
    T = V.T @ hvp(V)                                  # [k, k]
    T = 0.5 * (T + T.T)
    return V, T


def _solve_trustregion_kdim(qc, rc, gc, T, *, mu, rho):
    """Shared <=k-dim CTTN trust-region solve (numpy; k<=8).

    qc, rc, gc : [k] coordinates of q, r, g in the orthonormal V basis
                 (q = g/||g|| parallel dir, r = P(b) transverse buffer).
    T          : [k, k] Rayleigh block V^T H V.
    Returns (z_coords [k], diag) where z = V @ z_coords is the damped transverse
    momentum and diag carries tau/bind/energies/budget/ritz/n_modes_90.

    A = P Hplus P (q-direction projected out on both sides) keeps z transverse so
    q^T z == 0 exactly; the budget uses the full current-step curvature g^T Hplus g.
    """
    T = 0.5 * (T + T.T)
    evals, evecs = np.linalg.eigh(T)
    Tplus = (evecs * np.clip(evals, 0.0, None)) @ evecs.T          # [k,k] PSD

    qn = qc / (float(np.linalg.norm(qc)) + _EPS)
    Pc = np.eye(qc.shape[0]) - np.outer(qn, qn)
    Ac = Pc @ Tplus @ Pc
    Ac = 0.5 * (Ac + Ac.T)
    muA, W = np.linalg.eigh(Ac)                # muA >= 0 (PSD, q-dir -> 0)
    muA = np.clip(muA, 0.0, None)

    rj = W.T @ rc                              # r in the A-eigenbasis (rc ⊥ qc)
    e_before = float(np.sum(muA * rj * rj))    # r^T A r
    g_curv = float(gc @ (Tplus @ gc))          # g^T Hplus g
    budget = rho * g_curv
    mu4 = mu ** 4

    def energy(tau: float) -> float:
        zc = rj / (1.0 + tau * muA)
        return float(np.sum(muA * zc * zc))

    if mu4 * e_before <= budget or e_before <= _EPS:
        tau = 0.0
        bind = False
        z_eig = rj.copy()
    else:
        bind = True
        target = budget / mu4
        lo, hi = 0.0, 1.0
        for _ in range(200):                   # grow hi until energy(hi) <= target
            if energy(hi) <= target:
                break
            hi *= 2.0
        for _ in range(200):                   # bisection (energy monotone dec)
            mid = 0.5 * (lo + hi)
            if energy(mid) > target:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-14 * (hi + 1.0):
                break
        tau = hi
        z_eig = rj / (1.0 + tau * muA)

    z_coords = W @ z_eig
    e_after = float(np.sum(muA * z_eig * z_eig))

    contrib = muA * rj * rj                     # Ritz-mode concentration of e_before
    order = np.argsort(contrib)[::-1]
    csum = np.cumsum(contrib[order])
    total = csum[-1] if csum.size else 0.0
    n90 = int(np.searchsorted(csum, 0.9 * total) + 1) if total > _EPS else 0

    diag = {"tau": tau, "bind": bind, "e_before": e_before, "e_after": e_after,
            "budget": budget, "ritz": muA, "n_modes_90": n90}
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
    gn = float(np.linalg.norm(g))
    if gn <= _EPS:
        # Degenerate: no merged signal. Fall back to plain Nesterov.
        b_new = mu * b + g
        d = g + mu * b_new
        return CttnResult(d, b_new, np.zeros_like(g), 0.0, False,
                          0.0, 0.0, 1.0, 0.0, 0.0, 0.0, np.zeros(0), 0)

    q = g / gn
    r = project_out(b, q)              # transverse buffer
    b_parallel = b - r
    r_norm = float(np.linalg.norm(r))

    # Project into the k-dim V basis (q, r, g all lie in span(V)) and solve the
    # <=k-dim trust region with the shared core (identical for the torch path).
    qc = V.T @ q
    rc = V.T @ r
    gc = V.T @ g
    z_coords, diag = _solve_trustregion_kdim(qc, rc, gc, T, mu=mu, rho=rho)

    z = V @ z_coords                   # parameter space; q^T z == 0 by construction
    z_norm = float(np.linalg.norm(z))
    b_new = mu * (b_parallel + z) + g
    d = g + mu * mu * z                 # == g + mu * P(b_new); q^T d == ||g||

    return CttnResult(
        d=d, b_new=b_new, z=z, tau=diag["tau"], bind=diag["bind"],
        r_norm=r_norm, z_norm=z_norm,
        norm_retention=(z_norm / r_norm) if r_norm > _EPS else 1.0,
        e_before=diag["e_before"], e_after=diag["e_after"], budget=diag["budget"],
        ritz=diag["ritz"], n_modes_90=diag["n_modes_90"],
    )
