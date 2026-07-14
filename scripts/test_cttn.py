"""Golden-trace validation of the CTTN core on a synthetic Hessian.

Verifies the three guarantees the design rests on, so the GPU campaign tests a
correct optimizer rather than a buggy one:
  1. parallel step is exact SGD:            q^T d == ||g||
  2. transverse curvature trust region:     mu^4 z^T A z <= rho g^T Hplus g
  3. anisotropic damping:                   sharp Ritz modes of r shrunk,
                                            flat modes preserved
plus dimensionless-rho scale invariance (the bug that killed wsub) and the
non-binding / degenerate fallbacks. Regression cases cover exhausted Krylov
spaces, cached-HVP accounting, indefinite curvature, zero budgets, extreme
scales, and the bf16 torch path.

Run:  python scripts/test_cttn.py
"""

from __future__ import annotations

# This is a directly executable golden-trace program whose helper functions
# take explicit numerical arguments. Prevent repository-wide pytest discovery
# from misinterpreting those arguments as fixture names.
__test__ = False

import numpy as np

from yeto.cttn import (
    _solve_trustregion_kdim,
    block_lanczos,
    cttn_step,
    orth,
    project_out,
)

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


def scaled_bound_holds(lhs: float, rhs: float, rtol: float = 1e-11) -> bool:
    return lhs <= rhs + rtol * max(abs(lhs), abs(rhs))


def counted_hvp(H: np.ndarray):
    stats = {"calls": 0, "columns": 0}

    def hvp(X):
        stats["calls"] += 1
        stats["columns"] += X.shape[1]
        return H @ X

    return hvp, stats


def test_identity_rank_collapse(mu: float, rho: float) -> bool:
    print("\nIdentity-Hessian rank-collapse regression:")
    p = 24
    H = np.eye(p)
    g = RNG.standard_normal(p)
    q = g / np.linalg.norm(g)
    transverse = project_out(RNG.standard_normal(p), q)
    transverse /= np.linalg.norm(transverse)
    b = 4.0 * np.linalg.norm(g) * transverse + 0.25 * q
    r = project_out(b, q)
    Q0 = orth([q, r / np.linalg.norm(r)])
    hvp, stats = counted_hvp(H)
    V, T = block_lanczos(hvp, Q0, block_steps=4)
    res = cttn_step(g, b, V, T, mu=mu, rho=rho)

    gram_error = float(np.linalg.norm(V.T @ V - np.eye(V.shape[1])))
    lhs = mu**4 * res.e_after
    ok = True
    ok &= check(f"H=I Krylov stops at seed rank (k={V.shape[1]} == 2)", V.shape[1] == 2)
    ok &= check(
        f"H=I basis remains orthonormal (error={gram_error:.3g})", gram_error < 1e-12
    )
    ok &= check(
        f"cached HVP count equals returned rank ({stats['columns']} == 2)",
        stats["columns"] == V.shape[1] == 2,
    )
    ok &= check(
        "identity parallel invariant",
        np.isclose(q @ res.d, np.linalg.norm(g), rtol=1e-12, atol=0.0),
    )
    ok &= check(
        f"identity trust bound ({lhs:.6g} <= {res.budget:.6g})",
        scaled_bound_holds(lhs, res.budget),
    )
    return ok


def test_invariant_subspace(mu: float, rho: float) -> bool:
    print("\nLow-rank invariant-subspace regression:")
    H = np.diag([7.0, 3.0, 1.0, 0.0, 0.0, 0.0])
    g = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    q = g / np.linalg.norm(g)
    b = np.array([3.0, -2.0, 4.0, 0.0, 0.0, 0.0])
    r = project_out(b, q)
    Q0 = orth([q, r / np.linalg.norm(r)])
    hvp, stats = counted_hvp(H)
    V, T = block_lanczos(hvp, Q0, block_steps=4)
    res = cttn_step(g, b, V, T, mu=mu, rho=rho)

    gram_error = float(np.linalg.norm(V.T @ V - np.eye(V.shape[1])))
    ok = True
    ok &= check(
        f"invariant Krylov space stops at rank 3 (k={V.shape[1]})", V.shape[1] == 3
    )
    ok &= check(
        f"invariant basis orthonormal (error={gram_error:.3g})", gram_error < 1e-12
    )
    ok &= check(
        f"invariant-space HVPs cached ({stats['columns']} == {V.shape[1]})",
        stats["columns"] == V.shape[1],
    )
    ok &= check(
        "invariant-space result finite",
        np.all(np.isfinite(res.d)) and np.all(np.isfinite(res.z)),
    )
    return ok


def test_indefinite_and_zero_budget(mu: float) -> bool:
    print("\nIndefinite/zero-budget regressions:")
    H = np.diag([-5.0, 0.0, 3.0, 8.0])
    g = np.array([0.0, 0.0, 0.0, 1.0])
    b = np.array([1.0, 0.0, 1.0, 0.0])
    q = g.copy()
    r = project_out(b, q)
    V, T = block_lanczos(hvp_of(H), orth([q, r / np.linalg.norm(r)]), 4)
    res = cttn_step(g, b, V, T, mu=mu, rho=0.0)

    # Negative curvature is zeroed by Hplus, while the positive transverse
    # direction must be removed to meet a truly zero target.
    ok = True
    ok &= check("indefinite spectrum is PSD-projected", np.all(res.ritz >= 0.0))
    ok &= check(
        f"negative-curvature component retained ({res.z[0]:.6g} ~= 1)",
        np.isclose(res.z[0], 1.0, atol=1e-12),
    )
    ok &= check(
        f"positive-curvature component removed ({res.z[2]:.3g} ~= 0)",
        abs(res.z[2]) < 1e-12,
    )
    ok &= check(
        "rho=0 binds with positive transverse energy",
        res.bind and np.isinf(res.tau) and res.e_after == 0.0,
    )

    # g lies in H's nullspace, so the budget is zero even at rho>0. The
    # positive transverse mode is removed and A's nullspace is retained.
    H0 = np.diag([0.0, 2.0, 0.0])
    g0 = np.array([1.0, 0.0, 0.0])
    b0 = np.array([0.0, 1.0, 1.0])
    r0 = project_out(b0, g0)
    V0, T0 = block_lanczos(hvp_of(H0), orth([g0, r0 / np.linalg.norm(r0)]), 4)
    res0 = cttn_step(g0, b0, V0, T0, mu=mu, rho=0.1)
    ok &= check(
        "g_curv=0 removes positive transverse curvature",
        res0.bind and abs(res0.z[1]) < 1e-12,
    )
    ok &= check(
        "g_curv=0 retains transverse nullspace", np.isclose(res0.z[2], 1.0, atol=1e-12)
    )
    ok &= check(
        "zero-budget result remains exactly transverse",
        abs(g0 @ res0.z) < 1e-12 and np.isclose(g0 @ res0.d, 1.0),
    )
    return ok


def test_trustregion_edge_cases(mu: float, rho: float) -> bool:
    print("\nTrust-region scale/degeneracy regressions:")
    qc = np.array([1.0, 0.0])
    rc = np.array([0.0, 1.0])
    gc = np.array([1.0, 0.0])

    # Exact former counterexample: 0.9^4 * 1e-40 > 0.1 * 1e-40.
    _, tiny_diag = _solve_trustregion_kdim(
        qc,
        rc,
        gc,
        np.eye(2) * 1e-40,
        mu=mu,
        rho=rho,
    )
    tiny_lhs = mu**4 * tiny_diag["e_after"]
    ok = True
    ok &= check("T=1e-40 cap binds (no fixed absolute epsilon)", tiny_diag["bind"])
    ok &= check(
        f"T=1e-40 postcondition ({tiny_lhs:.3g} <= {tiny_diag['budget']:.3g})",
        scaled_bound_holds(tiny_lhs, tiny_diag["budget"]),
    )

    # Full-step scale invariance across 80 orders of curvature energy.
    scaled_results = []
    for label, alpha, beta in (
        ("tiny", 1e-20, 1e-40),
        ("huge", 1e20, 1e20),
    ):
        H = np.eye(2) * beta
        g = np.array([alpha, 0.0])
        b = np.array([0.0, alpha])
        V, T = block_lanczos(hvp_of(H), np.eye(2), block_steps=4)
        res = cttn_step(g, b, V, T, mu=mu, rho=rho)
        lhs = mu**4 * res.e_after
        scaled_results.append(res)
        ok &= check(f"{label}-scale cap binds", res.bind)
        ok &= check(
            f"{label}-scale postcondition ({lhs:.3g} <= {res.budget:.3g})",
            scaled_bound_holds(lhs, res.budget),
        )
        ok &= check(
            f"{label}-scale parallel invariant",
            np.isclose(res.d[0], alpha, rtol=1e-12, atol=0.0),
        )
    ok &= check(
        "tiny/huge retention is scale-invariant",
        np.isclose(
            scaled_results[0].norm_retention,
            scaled_results[1].norm_retention,
            rtol=1e-11,
            atol=0.0,
        ),
    )

    z_mu0, diag_mu0 = _solve_trustregion_kdim(
        qc,
        rc,
        gc,
        np.eye(2),
        mu=0.0,
        rho=0.0,
    )
    ok &= check(
        "mu=0 retains transverse momentum and cannot bind",
        not diag_mu0["bind"] and np.allclose(z_mu0, rc),
    )

    leaking_rc = np.array([3.0, 4.0])
    z_zero, diag_zero = _solve_trustregion_kdim(
        qc,
        leaking_rc,
        gc,
        np.zeros((2, 2)),
        mu=mu,
        rho=0.0,
    )
    ok &= check(
        "all-zero A retains only structurally transverse rc",
        not diag_zero["bind"] and np.allclose(z_zero, [0.0, 4.0]),
    )
    return ok


def test_torch_bf16_parity() -> bool:
    print("\nTorch bf16/fp32-work regression:")
    try:
        import torch
        from yeto.cttn_torch import block_lanczos_torch, cttn_step_torch
    except (ImportError, OSError) as exc:
        return check(f"torch import available ({exc})", False)

    Q0 = torch.tensor(
        [[0.5, 0.5], [0.5, -0.5], [0.5, 0.5], [0.5, -0.5]],
        dtype=torch.bfloat16,
    )
    stats = {"columns": 0}

    def identity_hvp(X):
        stats["columns"] += X.shape[1]
        return X

    V32, T32 = block_lanczos_torch(identity_hvp, Q0, block_steps=4)
    g = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.bfloat16)
    b = torch.tensor([4.0, -4.0, 4.0, -4.0], dtype=torch.bfloat16)
    V_bf16 = V32.to(torch.bfloat16)
    T_bf16 = T32.to(torch.bfloat16)
    d_t, b_new_t, diag_t = cttn_step_torch(
        g,
        b,
        V_bf16,
        T_bf16,
        mu=0.5,
        rho=0.25,
    )

    g_np = g.float().numpy().astype(np.float64)
    b_np = b.float().numpy().astype(np.float64)
    V_np = V_bf16.float().numpy().astype(np.float64)
    T_np = T_bf16.float().numpy().astype(np.float64)
    res_np = cttn_step(g_np, b_np, V_np, T_np, mu=0.5, rho=0.25)
    d_np_bf16 = torch.from_numpy(res_np.d).to(torch.bfloat16).float().numpy()
    b_np_bf16 = torch.from_numpy(res_np.b_new).to(torch.bfloat16).float().numpy()

    q = g.float() / torch.linalg.vector_norm(g.float())
    qtd = float(torch.dot(q, d_t.float()))
    gram_error = float(torch.linalg.matrix_norm(V32.T @ V32 - torch.eye(2)))
    ok = True
    ok &= check(
        f"torch H=I Krylov stops at 2 columns (k={V32.shape[1]})",
        V32.shape[1] == 2 and stats["columns"] == 2,
    )
    ok &= check(
        f"torch Krylov basis orthonormal (error={gram_error:.3g})", gram_error < 1e-6
    )
    ok &= check(
        "bf16 public output dtype preserved",
        d_t.dtype == torch.bfloat16 and b_new_t.dtype == torch.bfloat16,
    )
    ok &= check(
        "bf16 path matches numpy after the required output cast",
        np.allclose(d_t.float().numpy(), d_np_bf16, rtol=1e-6, atol=1e-6)
        and np.allclose(b_new_t.float().numpy(), b_np_bf16, rtol=1e-6, atol=1e-6),
    )
    ok &= check(
        f"bf16 torch parallel invariant ({qtd:.6g} == 2)",
        np.isclose(qtd, 2.0, rtol=1e-6, atol=1e-6),
    )
    ok &= check(
        "bf16 diagnostics are finite fp32-compatible numpy arrays",
        diag_t.d.dtype == np.float32
        and np.all(np.isfinite(diag_t.d))
        and diag_t.z.dtype == np.float32,
    )
    return ok


def main() -> int:
    p = 300
    mu = 0.9
    rho = 0.10
    H, U, lam = make_H(p, sharp=[120.0, 60.0, 30.0], flat_scale=1.0)
    hvp, golden_hvp_stats = counted_hvp(H)

    g = RNG.standard_normal(p)
    b = RNG.standard_normal(p) * 2.0  # buffer with real transverse content

    q = g / np.linalg.norm(g)
    r = project_out(b, q)
    Q0 = orth([q, r / np.linalg.norm(r)])
    V, T = block_lanczos(hvp, Q0, block_steps=4)

    res = cttn_step(g, b, V, T, mu=mu, rho=rho)

    ok = True
    print("CTTN core golden-trace:")

    ok &= check(
        f"full-rank Krylov uses one HVP per column "
        f"({golden_hvp_stats['columns']} == {V.shape[1]} <= 8)",
        golden_hvp_stats["columns"] == V.shape[1] <= 8,
    )

    # 1. parallel step is exact SGD
    qtd = float(q @ res.d)
    ok &= check(
        f"q^T d == ||g||  ({qtd:.10f} vs {np.linalg.norm(g):.10f})",
        abs(qtd - np.linalg.norm(g)) < 1e-8,
    )

    # 2. transverse curvature trust region (A = P Hplus P; z transverse)
    #    compute z^T H z directly in full space using the true H (PSD part).
    #    Hplus true: clip eigenvalues of H at 0.
    Hplus_true = (U * np.clip(lam, 0, None)) @ U.T
    zAz = float(res.z @ (Hplus_true @ res.z))
    budget_true = rho * float(g @ (Hplus_true @ g))
    ok &= check(
        f"mu^4 z^T Hplus z <= rho g^T Hplus g  "
        f"({mu**4 * zAz:.6g} <= {budget_true:.6g})",
        mu**4 * zAz <= budget_true * (1 + 1e-6) + 1e-9,
    )
    ok &= check(
        f"cap bound (in-basis)  ({mu**4 * res.e_after:.6g} <= {res.budget:.6g})",
        mu**4 * res.e_after <= res.budget * (1 + 1e-6) + 1e-12,
    )

    # 3. anisotropic damping: project r and z onto the true sharp eigenvectors;
    #    sharp components must be shrunk MORE than flat ones.
    def comp(vec, k):
        return float(U[:, k] @ vec)

    sharp_ret = abs(comp(res.z, 0)) / (abs(comp(r, 0)) + 1e-30)  # lam=120
    flat_k = int(np.argmin(lam))
    flat_ret = abs(comp(res.z, flat_k)) / (abs(comp(r, flat_k)) + 1e-30)
    ok &= check(
        f"sharp mode shrunk below flat mode  "
        f"(sharp_ret={sharp_ret:.4f} < flat_ret={flat_ret:.4f})",
        sharp_ret < flat_ret,
    )
    ok &= check(f"flat mode ~preserved (ret={flat_ret:.4f} > 0.98)", flat_ret > 0.98)
    ok &= check(f"cap binds on this poisoned buffer (bind={res.bind})", res.bind)
    print(
        f"    diag: ||r||={res.r_norm:.4f} ||z||={res.z_norm:.4f} "
        f"retention={res.norm_retention:.4f} tau={res.tau:.4g} "
        f"n_modes_90={res.n_modes_90} ritz_max={res.ritz.max():.3g}"
    )

    # 4. dimensionless-rho scale invariance (the wsub-killer bug).
    #    Scale g,b by alpha and H by beta: bind decision unchanged, z scales ~alpha,
    #    retention identical.
    alpha, beta = 7.0, 0.3
    Hs = beta * H
    Vs, Ts = block_lanczos(hvp_of(Hs), orth([q, r / np.linalg.norm(r)]), block_steps=4)
    res_s = cttn_step(alpha * g, alpha * b, Vs, Ts, mu=mu, rho=rho)
    ok &= check(
        f"scale-invariant bind decision (bind {res.bind} == {res_s.bind})",
        res.bind == res_s.bind,
    )
    ok &= check(
        f"scale-invariant retention  "
        f"({res.norm_retention:.6f} ~= {res_s.norm_retention:.6f})",
        abs(res.norm_retention - res_s.norm_retention) < 1e-3,
    )
    ok &= check(
        f"z scales ~alpha  (||z_s||/||z|| = {res_s.z_norm / res.z_norm:.4f} ~ {alpha})",
        abs(res_s.z_norm / res.z_norm - alpha) < 1e-2 * alpha,
    )

    # 5. non-binding case: tiny flat-only buffer => no cap, CTTN == plain Nesterov.
    H_flat, _, _ = make_H(p, sharp=[], flat_scale=1e-6)
    hv = hvp_of(H_flat)
    b_small = project_out(RNG.standard_normal(p), q) * 1e-3 + q * 0.5
    Vf, Tf = block_lanczos(
        hv,
        orth(
            [
                q,
                project_out(b_small, q)
                / (np.linalg.norm(project_out(b_small, q)) + 1e-30),
            ]
        ),
        block_steps=4,
    )
    res_f = cttn_step(g, b_small, Vf, Tf, mu=mu, rho=rho)
    # No cap => z == r (full transverse momentum retained). CTTN is SGD on the
    # parallel axis + full transverse momentum: d == g + mu^2 * P(b). (NOT plain
    # Nesterov: the parallel momentum component is dropped by design so that
    # q^T d == ||g|| always — a win cannot be a hidden effective-LR change.)
    r_small = project_out(b_small, q)
    d_expected = g + mu * mu * r_small
    ok &= check(f"flat spectrum => no bind (bind={res_f.bind})", not res_f.bind)
    ok &= check(
        "flat spectrum => z == r (d = g + mu^2 P(b))",
        np.allclose(res_f.d, d_expected, atol=1e-7),
    )

    # 6. degenerate g==0 fallback doesn't crash and returns plain Nesterov.
    res0 = cttn_step(np.zeros(p), b, V, T, mu=mu, rho=rho)
    ok &= check("g==0 fallback finite", np.all(np.isfinite(res0.d)))

    ok &= test_identity_rank_collapse(mu, rho)
    ok &= test_invariant_subspace(mu, rho)
    ok &= test_indefinite_and_zero_budget(mu)
    ok &= test_trustregion_edge_cases(mu, rho)
    ok &= test_torch_bf16_parity()

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
