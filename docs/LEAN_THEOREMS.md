# Machine-checked theorems: the transverse-momentum mechanism (2D quadratic)

Status: 2026-07-12. Lean 4 + Mathlib, sorry-free. Source:
`lean-mechanism/LeanMechanism/Basic.lean` (builds with `lake build`; Lean
toolchain `leanprover/lean4:v4.31.0`). Every theorem below compiles and
`#print axioms` reports only the three standard Mathlib axioms
(`propext`, `Classical.choice`, `Quot.sound`) — no `sorryAx`, no extra
assumptions smuggled in.

These compress the DiLoCo-poison empirical findings (docs/THEORY.md Result A/B,
the EXP2.37 matched-η_eff triads) onto a **minimal** anisotropic 2D quadratic —
deliberately *not* a full Local-SGD formalization. They isolate the one
mechanism the experiments kept surfacing: **outer directional (first-moment)
memory injects a transverse displacement whose harm is curvature-gated, and no
geometry-blind scalar controller can undo it.**

## Model (exact, no smoothness slack)

Loss `L(p) = ½ pᵀ H p`, `H` symmetric 2×2 with entries `a=H₁₁, b=H₂₂, c=H₁₂`.
Gradient `∇L(θ) = Hθ`. For a quadratic the one-step change of the update
`θ ↦ θ − u` is the exact identity (`dL` in the source):

    ΔL(u) = L(θ−u) − L(θ) = − uᵀ(Hθ) + ½ uᵀ H u.

Pseudo-gradient `g_t = ∇L(θ) = Hθ` (minimal model: even when the merged delta
equals the true gradient, the **buffer** `b_{t-1}` carries a foreign direction).
- SGD applies `u_sgd = η_s·g_t`.
- Nesterov applies `u_mom = η_m·d_t`, `d_t = (1+μ)g_t + μ²b_{t-1} = A·g_t + d⊥`
  with aligned gain `A = 1+μ+μ²c_t` and transverse part `d⊥ = μ²b⊥ ⟂ g_t`
  (docs/THEORY.md Prop A.1).
- "Matched aligned gain": choose `η_s = η_m·A` so the SGD step equals the
  parallel part of the momentum step, `u_sgd = Dpar = η_m A g_t`.

---

## T1 — Matched aligned gain does not imply matched descent  ✅ proved

`T1_descent_gap`, `T1_matched_sgd`.

Exact decomposition of the descent-gap: with equal aligned part `Dpar` and the
extra momentum part `Dperp ⟂ g` (`⟨Dperp,g⟩ = 0`),

    ΔL_mom − ΔL_sgd = Dparᵀ H Dperp + ½ Dperpᵀ H Dperp
                    = ½ ( Dperpᵀ H Dperp + 2 Dparᵀ H Dperp ).

This is the exact form the task asked for. `T1_matched_sgd` instantiates the full
Nesterov step with a transverse buffer (`c_t = 0`, `A = 1+μ`) and SGD at the
matched LR `η_s = η_m(1+μ)`, and proves *both* that the two steps share the
aligned part exactly (`u_mom = u_sgd + Dperp`, `Dperp = η_m μ² buf ⟂ g`) *and*
that their loss changes nonetheless differ by the transverse form above. So
matching the aligned projection provably does **not** equalize descent; the gap
is exactly the sharp-direction contamination the aligned match cannot see.

Product implication: an outer optimizer tuned to match SGD's *aligned* effective
step (the EXP2.25/2.37 η_eff matching) is not thereby matched in loss — the
residual is `½ Dperpᵀ H Dperp + Dparᵀ H Dperp`, a curvature quantity.

## T2 — Transverse momentum: benign under isotropy, harmful under anisotropy  ✅ proved

`T2_sgd_descends`, `T2_aniso_mom_ascends(_pos)`, `T2_iso_mom_descends(_neg)`,
`T2_threshold`.

Concrete verified instance (rationals; matched aligned step η_s = 57/200 ≈ 0.28,
the tuned-SGD LR; μ = 9/10; buffer purely in the sharp direction):

| quantity | value |
|---|---|
| `H` | `diag(1, ℓ_y)` |
| `θ = g` | `(1, 0)` (gradient in the flat direction) |
| buffer `b` | `(0, 1)` (transverse, sharp direction) |
| `Dpar = u_sgd` | `(57/200, 0) = η_m·A·g`, `η_m = 3/20`, `A = 19/10` |
| `Dperp` | `(0, 243/2000) = η_m·μ²·b` |

- Matched SGD descends by `19551/80000 ≈ 0.2444`, **independent of ℓ_y**
  (`T2_sgd_descends`).
- **Anisotropic ℓ_y = 100**: `ΔL_mom = 1974900/4000000 = +0.4937 > 0` — momentum
  turns a descent into an **ascent** (`T2_aniso_mom_ascends_pos`).
- **Isotropic ℓ_y = 1**: `ΔL_mom = −0.2370 < 0` — momentum still descends, the
  transverse harm is a negligible `0.0074` (`T2_iso_mom_descends_neg`).
- **Exact threshold** (`T2_threshold`): `ΔL_mom(ℓ_y) > 0 ⇔ ℓ_y > 3910200/118098
  ≈ 33.11`. Below it momentum is benign, above it harmful — the threshold-like
  behavior the matched-pairs experiment reported (negligible at mild curvature,
  severe at sharp).

Same buffer, same aligned step: only the curvature ratio flips momentum from
benign to catastrophic. This is the exact-arithmetic mechanism behind the
empirical "harm is threshold-like, severe under anisotropy" finding.

## T3 — No geometry-blind scalar cap dominates tuned SGD over all quadratics  ✅ proved

`bestReduction_is_scalar_optimum`, `T3_strict_gap`, `T3_no_uniform_bound`.

A controller applying a scalar step-scale `s(g,b)` to `d = g + μb` is stuck with
the **direction** `d`; only its length is free. `bestReduction_is_scalar_optimum`
proves that `bestReduction lx ly v g = ⟨v,g⟩²/(2 vᵀHv)` really is the supremum
over scalars `s` of the *actual* one-step loss reduction `−dL(s·v)`, attained at
`s⋆ = ⟨v,g⟩/vᵀHv` — so the impossibility below is about the true loss, not a
chosen formula.

- `T3_strict_gap`: for diagonal `H`, gradient `g=(gx,0)`, buffer `b=(0,by)`, the
  controller's best reduction along `d=(gx, μby)` is **strictly less** than tuned
  SGD's along `g`, whenever the transverse sharp mass is nonzero. The exact ratio
  is `ℓx gx² / (ℓx gx² + μ² ℓy by²) < 1`.
- `T3_no_uniform_bound` (the impossibility): for **every** target ratio
  `κ ∈ (0,1)` there is a sharp curvature `ℓy > 0` on which the controller's best
  reduction is `< κ ·` tuned SGD's. Since `g, b, μ` (hence any scalar `s`) are
  held fixed while `ℓy` varies, no fixed geometry-blind scalar controller stays
  within any constant factor of tuned SGD across the quadratic family.

This is the formal core of why three controller generations (rho-adaptive v1/v2,
capped-Nesterov, capped-Nesterov-gc/-r) failed: they scale `d_t` by a scalar
computed from gain/cosine (`g, b`) only, and that is provably insufficient.

## T4 — A curvature-aware transverse cap preserves SGD stability  ✅ proved (conditional, honest)

`T4_curvature_aware_cap`.

If (i) `Dperp ⟂ g`, (ii) the directions are eigen-aligned so the cross curvature
term vanishes (`Dparᵀ H Dperp = 0`), (iii) SGD's aligned part strictly descends,
and (iv) the controller uses **curvature information** to cap the transverse
energy below the aligned descent it achieved (`½ Dperpᵀ H Dperp ≤ −ΔL_sgd`),
then the full momentum step also descends (`ΔL_mom ≤ 0`).

Assumptions stated honestly: the cap needs `Dperpᵀ H Dperp`, a Hessian
(curvature) quantity — exactly the information a geometry-blind cap (T3) lacks;
and eigen-alignment is what makes the cross term drop. The theorem is a
conditional safety criterion, not an unconditional win: it says the *right* extra
signal (curvature) is sufficient to restore the descent guarantee that a scalar
cap cannot.

---

## Proof status summary

| Theorem | Lean name | Status |
|---|---|---|
| T1 exact descent-gap identity | `T1_descent_gap` | proved, sorry-free |
| T1 matched-aligned instantiation | `T1_matched_sgd` | proved, sorry-free |
| T2 SGD descent (matched, ℓ_y-free) | `T2_sgd_descends` | proved, sorry-free |
| T2 anisotropic ascent | `T2_aniso_mom_ascends_pos` | proved, sorry-free |
| T2 isotropic descent (benign) | `T2_iso_mom_descends_neg` | proved, sorry-free |
| T2 exact threshold ℓ_y⋆ = 33.11 | `T2_threshold` | proved, sorry-free |
| T3 scalar-optimality of bestReduction | `bestReduction_is_scalar_optimum` | proved, sorry-free |
| T3 strict gap vs tuned SGD | `T3_strict_gap` | proved, sorry-free |
| T3 no uniform bound (impossibility) | `T3_no_uniform_bound` | proved, sorry-free |
| T4 curvature-aware cap preserves descent | `T4_curvature_aware_cap` | proved, sorry-free |

All ten depend only on `[propext, Classical.choice, Quot.sound]`.

## Concrete counterexample instances (verified in Lean as exact rationals)

- **T1/T2 poison instance** (numerically pre-searched in `.venv` python, then
  formalized): `H = diag(1,100)`, `θ = g = (1,0)`, buffer `b = (0,1)`,
  `μ = 9/10`, `η_m = 3/20` (so `η_s = 57/200 ≈ 0.28`). Matched SGD descends
  `0.2444`; momentum ascends `+0.4937`. The same buffer/step at `H = diag(1,1)`
  descends `−0.2370` (harm `0.0074`). Threshold `ℓ_y⋆ = 3910200/118098 ≈ 33.11`.
- **T3 impossibility witness**: at `lx=gx=by=1, μ=9/10`, `ℓ_y=100` gives ratio
  `1/82 ≈ 0.0122` — the geometry-blind controller, even at its own optimal
  scalar, gets ~1.2% of tuned SGD's one-step descent; `→ 0` as `ℓ_y → ∞`.

## Product implication (per the user's framing)

The next-generation outer optimizer must **either drop directional (first-moment)
memory entirely OR acquire curvature/covariance information**. A controller that
sees only gain/cosine of `(g_t, b_{t-1})` and outputs a scalar step-scale is
provably insufficient: it cannot cancel the transverse contamination `Dperp`
(T1), that contamination is catastrophic under the anisotropic curvature this
regime exhibits (T2), and no such scalar dominates tuned SGD across the quadratic
family (T3). Restoring the descent guarantee provably requires curvature input
(T4). This is the machine-checked backbone of docs/NEXT_OPTIMIZER_PLAN.md's
decision to build on the memoryless SGD base and add only current-round
spatial/curvature/consensus signals — never a directional buffer.

Scope caveats (honest): 2D and per-step (no multi-step/convergence claim);
exact-arithmetic (no f32 rounding, cf. THEORY.md §0); `g_t = ∇L` in the model
(the production pseudo-gradient-vs-gradient gap is THEORY.md Conjecture C.5); T4
is a conditional (eigen-aligned, curvature-cap) sufficiency statement.

---

# Anchor-drift modules: two SEPARABLE mechanisms + the exact correction

Status: 2026-07-13. Four new sorry-free modules
(`lean-mechanism/LeanMechanism/{MergeSemantics,AnchorDrift,Counterexamples,Correction}.lean`),
same toolchain, all built by `lake build` and independently
`#print axioms`-checked to depend only on `[propext, Classical.choice,
Quot.sound]` (R5 only `[propext]`). These machine-verify the docs/ANCHOR_DRIFT_CONTROL.md
thesis that "native momentum poison" and "current-anchor contamination" are two
distinct mechanisms, and prove the base-version correction exactly removes the
second. Every merge definition was validated against the real Rust syncer by a
golden trace (below); the Lean model is the update `merge.rs`/`server.rs`
actually apply.

## Semantics (both sign conventions, kept separate by name)
Per-worker syncer **pseudo-gradient** is `δ = anchor − upload` (anchor MINUS
learner; confirmed in `merge_avg`/`build_aggregate_from_subset`). Two anchors:
current-anchor `δ_ca = current − upload` (PRODUCTION), version-matched
`δ_vm = base − upload`; `anchorDrift = current − base`, so
`δ_ca = δ_vm + anchorDrift`. The **local-delta** convention (learner movement
`= upload − anchor`) gives `localCa = localVm − anchorDrift` — the exact form of
docs/ANCHOR_DRIFT_CONTROL.md's `current_anchor_delta = true_local_delta − anchor_drift`.

## MergeSemantics.lean — three state machines, equivalence condition ✅ sorry-free
`MergeEvent {base, current, upload}`; `caDelta`, `vmDelta`, `anchorDrift`,
`Barrier e := current = base`.
| theorem | statement |
|---|---|
| `caDelta_eq_vmDelta_add_drift` | `caDelta = vmDelta + anchorDrift` (componentwise) |
| `barrier_drift_zero` | `Barrier e → anchorDrift e = 0` |
| `barrier_ca_eq_vm` | `Barrier e → caDelta e = vmDelta e` |
| `ca_eq_vm_iff_no_drift` | `caDelta e = vmDelta e ↔ anchorDrift e = 0` (THE equivalence: the two streaming semantics agree exactly iff drift = 0) |

## AnchorDrift.lean — the core identity + the 5 numbered results ✅ sorry-free
`localVm`, `localCa`; `localCa_eq_localVm_sub_drift` is the core identity
`localCa = localVm − anchorDrift`.
| theorem | statement (turns prose into a theorem) |
|---|---|
| `R1_barrier_implies_no_drift` | barrier ⇒ `anchorDrift = 0` |
| `R2_no_drift_ca_eq_vm` | `anchorDrift = 0` ⇒ current-anchor ≡ version-matched exactly |
| `R3_quorum_not_no_drift` | ∃ event: full quorum yet `base_version < current_version` and `anchorDrift ≠ 0` (strict quorum does NOT imply drift 0) |
| `R4_zero_delay_not_no_drift` | ∃ event: `injected_delay = 0` yet `anchorDrift ≠ 0` (zero injected delay does NOT imply drift 0) |
| `R5_all_commit_not_same_version` | ∃ two committing workers with different `base_version` (all-commit does NOT imply all-same-version) |

R3/R4/R5 are existence-of-counterexample-state theorems: quorum and
zero-injected-delay both constrain PARTICIPATION, not VERSION agreement, so
"strict-quorum but not barrier" carries nonzero anchor drift.

## Counterexamples.lean — the two separable mechanisms ✅ sorry-free
**(A) NATIVE momentum poison, compatible with barrier (drift = 0).** Reuses the
T2 instance `H = diag(1,100)`, `θ = g = (1,0)`, transverse buffer `(0,1)`,
`μ = 9/10`, matched SGD LR `57/200`. `cexA_barrier_no_drift` (this event has zero
drift), `cexA_sgd_descends` (`−19551/80000 < 0`), `cexA_mom_ascends` (`+19749/40000
> 0`). Shows the momentum mechanism can exist even under ORIGINAL barrier DiLoCo
(zero staleness, zero overlap, zero drift) — a possibility, requiring `μ > 0` and
a transverse buffer.
**(B) ANCHOR-DRIFT poison, ZERO momentum (`μ = 0`).** `H = diag(1,4)`, `θ = (2,0)`,
`δ_vm = (2,0)`, anchor drift `d = (0, 3/2)` (sharp direction), `δ_ca = (2, 3/2)`.
`cexB_vm_descends` (`= −2`) / `cexB_vm_descends_neg`; `cexB_ca_ascends` (`= +5/2`)
/ `cexB_ca_ascends_pos`; `cexB_only_drift_differs` (`δ_ca = δ_vm + (0,3/2)`).
Identical worker updates / quorum / data / optimizer — ONLY swapping version-
matched → current-anchor flips descent (`−2`) to ascent (`+5/2`), with **no
momentum buffer at all**. This is why the two mechanisms are distinct: (A) needs
momentum + transverse buffer + curvature even at zero drift; (B) needs only
nonzero anchor drift, no momentum.

**Parametric danger-threshold** (general `H`, `μ = 0`, `η = 1`):
`drift_gap_identity`:
`dL(δ_vm + d) − dL(δ_vm) = Bil H d (δ_vm − θ) + ½ Q H d`.
(Sign note — genuine correction: the interaction is `dᵀH(δ_vm − θ)`, NOT the
`dᵀH(θ − δ_vm)` first drafted; forced by the `θ ↦ θ − u` convention and
`δ_ca = δ_vm + d`. Caught during the compile loop; the wrong sign is false, e.g.
`H = I, θ = (1,0), δ_vm = 0, d = (1,0)`.) `drift_flip_threshold`: given the
version-matched step descends, current-anchor ASCENDS iff
`Bil H d (δ_vm − θ) + ½ Q H d > −dL(δ_vm)` (the descent margin). The flip depends
on drift **direction and curvature**, not norm: a large drift in a flat direction
is harmless; modest sharp-direction drift is catastrophic. → the GPU experiment
should log this curvature-weighted interaction, not just `‖anchorDrift‖`.

**Upgraded T3 (system-faithful indistinguishability).** `T3_upgrade_indistinguishable`,
`T3_upgrade_controller_blind`. Existence of an INDISTINGUISHABLE PAIR: same merged
delta `δ = (2, 3/2)` and same buffer `(0,0)`, but a version-matched instance
(`H = I, θ = (2,3/2)`) descends `−25/8` while a current-anchor-contaminated
instance (`H = diag(1,4), θ = (2,0)`) ascends `+5/2`. Any controller whose entire
input is `(δ, buffer)` — hence gain `‖δ‖` and `cos(δ,buf)`, both functions of
`(δ,buf)` — returns the same action on both, so gain/cosine alone cannot separate
a safe true-local update from current-anchor contamination.
Exact quantifiers (audited, deliberately NOT over-informing the comparator):
(i) EXISTENCE of one bad pair, not an all-instances / worst-case lower bound;
(ii) the controller sees ONLY `(δ, buffer)` — it is NOT given `H`, `θ`,
`anchorDrift`, or `base_version`; (iii) fixed unit step `η = 1`, `μ = 0`, same
applied step on both problems (no per-problem oracle LR tuning of the
controller); (iv) ONE-step (no convergence/long-horizon claim); (v) gain and
cosine are functions of `(δ,buf)`, so equal observable ⇒ equal gain/cosine. This
strengthens Basic.lean's T3 from "scalar-cap suboptimality across a quadratic
family" to "merged-delta gain/cosine cannot even distinguish current-anchor
contamination from a safe update."

## Correction.lean — base-version correction exactly recovers the true delta ✅ sorry-free
| theorem | statement |
|---|---|
| `correction_recovers_vm` | `localCa + anchorDrift = localVm` (current_anchor_delta + anchor_drift = version_matched_local_delta — the task's exact identity) |
| `correction_recovers_vm_pg` | `caDelta − anchorDrift = vmDelta` (pseudo-gradient form) |
| `corrected_step_eq_vm_step` | `dL(θ, (δ_vm + d) − d) = dL(θ, δ_vm)` (correcting by −d gives exactly the version-matched loss change) |
| `correction_safe` | the corrected step, being the version-matched step, inherits its descent |

Product implication: if the GPU 3-arm control (docs/ANCHOR_DRIFT_CONTROL.md) finds
current-anchor is the main source, the fix is NOT a fancier optimizer — it is
learners uploading `base_version` so the syncer differences against the true base
and recovers the exact true local delta. This is precisely the algebra of the
in-progress Rust `--version-matched-anchor` flag (`server.rs`: re-anchor
`upload' = upload + (current − base)`, so `current − upload' = base − upload`).

## Lean ↔ Rust golden-trace consistency check (more load-bearing than another theorem)
`scripts/lean_rust_golden_trace.py` (exact-rational reference) +
`syncer/src/merge.rs::tests::anchor_drift_golden_trace_matches_lean_model` (drives
the REAL production `merge_avg` + `nesterov_step`). A 2-dim / 2-worker / 3-commit
trace (`η = μ = 1/2`, base `(0,0)`, `b_0 = 0`), all dyadic so f32 == exact and
assertions are bit-exact `==`. **Verified against `merge.rs`/`state.rs` reality:**
delta SIGN = anchor − learner; merge is weighted MEAN not sum (equal weights c1,
1:3 weights c2); the exact Nesterov form `b_t = μ b_{t-1} + δ_t`,
`d_t = δ_t + μ b_t`, `θ −= η d_t`, `b_0 = 0`; and WHERE current-anchor bites — at
commit 3 a lagging worker (base `θ_1` while the server is at `θ_2`) makes the
production current-anchor delta `(−7/64, 1/32)` differ from the version-matched
delta `(−15/128, −3/64)` by exactly that worker's merge-weighted anchor drift
(`½ · (1/64, 5/32)`). The Lean model, the Python reference, and BOTH Rust paths
(production current-anchor; the WIP `--version-matched-anchor`) agree. **No
discrepancy found** — the Lean theorems are about the update the syncer really
applies. (Ground-truth check: `Δ = self.params[fid] − upload` with
`self.params[fid]` the syncer's CURRENT global is the current-anchor locus;
`nesterov_step` matches OPTIMIZER_SEMANTICS.md's recursion bit-for-bit; buffer
init `b_0 = 0`; push-vs-commit versions gate admission but not the arithmetic.)

## Proof status summary (anchor-drift modules)
All 20 theorems below are proved, sorry-free, `#print axioms` = the three
standard axioms (R5: `propext` only).

| module | theorems |
|---|---|
| MergeSemantics | `caDelta_eq_vmDelta_add_drift`, `barrier_drift_zero`, `barrier_ca_eq_vm`, `ca_eq_vm_iff_no_drift` |
| AnchorDrift | `localCa_eq_localVm_sub_drift`, `R1_barrier_implies_no_drift`, `R2_no_drift_ca_eq_vm`, `R3_quorum_not_no_drift`, `R4_zero_delay_not_no_drift`, `R5_all_commit_not_same_version` |
| Counterexamples | `cexA_barrier_no_drift`, `cexA_sgd_descends`, `cexA_mom_ascends`, `cexB_vm_descends(_neg)`, `cexB_ca_ascends(_pos)`, `cexB_only_drift_differs`, `drift_gap_identity`, `drift_flip_threshold`, `T3_upgrade_indistinguishable`, `T3_upgrade_controller_blind` |
| Correction | `correction_recovers_vm`, `correction_recovers_vm_pg`, `corrected_step_eq_vm_step`, `correction_safe` |

Scope caveats (honest, same discipline as T1–T4): 2D and per-step (no
multi-step/convergence); exact rational arithmetic (no f32 rounding); the
memoryless (`μ = 0`) cases isolate the anchor-drift mechanism but the production
optimizer carries momentum, which can only compound accumulated drift (THEORY.md
§0, A.3); `dL` uses the exact quadratic loss change, and `g = Hθ` where a true
gradient is invoked. Lean does NOT and must NOT adjudicate which mechanism
dominates real training — that is the GPU 3-arm control's job
(docs/ANCHOR_DRIFT_CONTROL.md). These modules prove the mechanisms are
SEPARABLE and that the correction is EXACT; they do not claim current-anchor is
the empirical culprit.
