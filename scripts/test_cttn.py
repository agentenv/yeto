"""Golden-trace validation of the CTTN core on a synthetic Hessian.

Verifies the three guarantees the design rests on, so the GPU campaign tests a
correct optimizer rather than a buggy one:
  1. parallel step is exact SGD:            q^T d == ||g||
  2. transverse curvature trust region:     mu^4 z^T A z <= rho g^T Hplus g
  3. anisotropic damping:                   sharp Ritz modes of r shrunk,
                                            flat modes preserved
plus dimensionless-rho scale invariance (the bug that killed wsub) and the
non-binding / degenerate fallbacks.

Run:  python scripts/test_cttn.py
"""

from __future__ import annotations

import numpy as np

from yeto.cttn import block_lanczos, cttn_step, orth, project_out

RNG = np.random.default_rng(20260713)


def make_H(p: int, sharp: list[float], flat_scale: float) -> np.ndarray:
    """Symmetric PSD-ish H: a few sharp eigen-directions + a flat bulk.
    Returns the dense [p,p] matrix (test-only; real runs never form H)."""
    # random orthonormal eigenbasis
    A = RNG.standard_normal((p, p))
    U, _ = np.linalg.qr(A)
    lam = flat_scale * np.abs(RNG.standard_normal(p)) * 0.01
    for i, s in enumerate(sharp):
        lam[i] = s
    return (U * lam) @ U.T, U, lam


def hvp_of(H):
    return lambda X: H @ X


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    return cond


def main() -> int:
    p = 300
    mu = 0.9
    rho = 0.10
    H, U, lam = make_H(p, sharp=[120.0, 60.0, 30.0], flat_scale=1.0)
    hvp = hvp_of(H)

    g = RNG.standard_normal(p)
    b = RNG.standard_normal(p) * 2.0        # buffer with real transverse content

    q = g / np.linalg.norm(g)
    r = project_out(b, q)
    Q0 = orth([q, r / np.linalg.norm(r)])
    V, T = block_lanczos(hvp, Q0, block_steps=4)

    res = cttn_step(g, b, V, T, mu=mu, rho=rho)

    ok = True
    print("CTTN core golden-trace:")

    # 1. parallel step is exact SGD
    qtd = float(q @ res.d)
    ok &= check(f"q^T d == ||g||  ({qtd:.10f} vs {np.linalg.norm(g):.10f})",
                abs(qtd - np.linalg.norm(g)) < 1e-8)

    # 2. transverse curvature trust region (A = P Hplus P; z transverse)
    #    compute z^T H z directly in full space using the true H (PSD part).
    #    Hplus true: clip eigenvalues of H at 0.
    Hplus_true = (U * np.clip(lam, 0, None)) @ U.T
    zAz = float(res.z @ (Hplus_true @ res.z))
    budget_true = rho * float(g @ (Hplus_true @ g))
    ok &= check(f"mu^4 z^T Hplus z <= rho g^T Hplus g  "
                f"({mu**4 * zAz:.6g} <= {budget_true:.6g})",
                mu**4 * zAz <= budget_true * (1 + 1e-6) + 1e-9)
    ok &= check(f"cap bound (in-basis)  ({mu**4*res.e_after:.6g} <= {res.budget:.6g})",
                mu**4 * res.e_after <= res.budget * (1 + 1e-6) + 1e-12)

    # 3. anisotropic damping: project r and z onto the true sharp eigenvectors;
    #    sharp components must be shrunk MORE than flat ones.
    def comp(vec, k):
        return float(U[:, k] @ vec)
    sharp_ret = abs(comp(res.z, 0)) / (abs(comp(r, 0)) + 1e-30)   # lam=120
    flat_k = int(np.argmin(lam))
    flat_ret = abs(comp(res.z, flat_k)) / (abs(comp(r, flat_k)) + 1e-30)
    ok &= check(f"sharp mode shrunk below flat mode  "
                f"(sharp_ret={sharp_ret:.4f} < flat_ret={flat_ret:.4f})",
                sharp_ret < flat_ret)
    ok &= check(f"flat mode ~preserved (ret={flat_ret:.4f} > 0.98)", flat_ret > 0.98)
    ok &= check(f"cap binds on this poisoned buffer (bind={res.bind})", res.bind)
    print(f"    diag: ||r||={res.r_norm:.4f} ||z||={res.z_norm:.4f} "
          f"retention={res.norm_retention:.4f} tau={res.tau:.4g} "
          f"n_modes_90={res.n_modes_90} ritz_max={res.ritz.max():.3g}")

    # 4. dimensionless-rho scale invariance (the wsub-killer bug).
    #    Scale g,b by alpha and H by beta: bind decision unchanged, z scales ~alpha,
    #    retention identical.
    alpha, beta = 7.0, 0.3
    Hs = beta * H
    Vs, Ts = block_lanczos(hvp_of(Hs), orth([q, r / np.linalg.norm(r)]), block_steps=4)
    res_s = cttn_step(alpha * g, alpha * b, Vs, Ts, mu=mu, rho=rho)
    ok &= check(f"scale-invariant bind decision (bind {res.bind} == {res_s.bind})",
                res.bind == res_s.bind)
    ok &= check(f"scale-invariant retention  "
                f"({res.norm_retention:.6f} ~= {res_s.norm_retention:.6f})",
                abs(res.norm_retention - res_s.norm_retention) < 1e-3)
    ok &= check(f"z scales ~alpha  (||z_s||/||z|| = {res_s.z_norm/res.z_norm:.4f} ~ {alpha})",
                abs(res_s.z_norm / res.z_norm - alpha) < 1e-2 * alpha)

    # 5. non-binding case: tiny flat-only buffer => no cap, CTTN == plain Nesterov.
    H_flat, _, _ = make_H(p, sharp=[], flat_scale=1e-6)
    hv = hvp_of(H_flat)
    b_small = project_out(RNG.standard_normal(p), q) * 1e-3 + q * 0.5
    Vf, Tf = block_lanczos(hv, orth([q, project_out(b_small, q) /
                                     (np.linalg.norm(project_out(b_small, q)) + 1e-30)]),
                           block_steps=4)
    res_f = cttn_step(g, b_small, Vf, Tf, mu=mu, rho=rho)
    # No cap => z == r (full transverse momentum retained). CTTN is SGD on the
    # parallel axis + full transverse momentum: d == g + mu^2 * P(b). (NOT plain
    # Nesterov: the parallel momentum component is dropped by design so that
    # q^T d == ||g|| always — a win cannot be a hidden effective-LR change.)
    r_small = project_out(b_small, q)
    d_expected = g + mu * mu * r_small
    ok &= check(f"flat spectrum => no bind (bind={res_f.bind})", not res_f.bind)
    ok &= check("flat spectrum => z == r (d = g + mu^2 P(b))",
                np.allclose(res_f.d, d_expected, atol=1e-7))

    # 6. degenerate g==0 fallback doesn't crash and returns plain Nesterov.
    res0 = cttn_step(np.zeros(p), b, V, T, mu=mu, rho=rho)
    ok &= check("g==0 fallback finite", np.all(np.isfinite(res0.d)))

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
