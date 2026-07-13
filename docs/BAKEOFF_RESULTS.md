# Outer-Optimizer Bake-off — Exact Results

Every alternative outer optimizer tested against memoryless SGD-0.28 on the
DiLoCo momentum poison (Qwen3.5-9B, LoRA, strict-quorum streaming). Δ =
candidate − SGD-0.28 (negative = better than SGD). Noise floor ≈ 0.009.
**Product gate:** paired win **> 0.018** on ≥2 workloads AND never worse than 1
noise floor on any workload; no per-H/rank/inner-LR tuning.

## Δ vs SGD-0.28 (per horizon)

| Candidate | H16 | H64 | H256 | other cell | pattern |
|---|---|---|---|---|---|
| worker-SNR (consensus)      | **−0.0019** (win) | +0.0016      | **+0.0157 ✗** | —                     | short-H helps, long-H breaks |
| block-RMS (2nd-moment)      | **+0.0093 ✗**     | +0.0026      | **−0.0095** (win) | —                  | opposite tilt: long-H helps |
| block-Yogi (2nd-moment)     | +0.0086           | +0.0012      | n/a¹          | —                     | ~flat |
| curvature-aware momentum    | **+0.0303 ✗**     | +0.0055 ✗    | **−0.0103** (win) | innerlr-hi **+0.0283 ✗** | H256-only; loses where predicted to help |
| Iso-C (spectral)            | **−0.0078** (win) | +0.0001 (tie)| **+0.0183 ✗** | —                     | short-H helps, long-H breaks |
| capnest v2.1 (scalar cap)   | +0.0236 ✗         | +0.0179 ✗    | −0.0040 (tie) | —                     | H-stable, never a real win |
| dynamic-H                   | —                 | —            | —             | best controller **+0.011 behind** tuned SGD | SGD wins |

**Gate result: NONE passes.** No candidate reaches a >0.018 win on ≥2 workloads,
and every one is worse than SGD by ≥1 noise floor at some horizon → **ship
memoryless SGD-0.28.** The unifying finding: each candidate wins in exactly the
horizon regime the dynamics diagnostic predicts (spatial/consensus → short-H;
second-moment/spectral/curvature → long-H), and none is uniformly better — there
is no free lunch in the outer optimizer at this cadence.

## Absolute eval-loss (raw numbers)

| Candidate | H16 | H64 | H256 | innerlr-hi-H64 |
|---|---|---|---|---|
| SGD-0.28 ref            | 1.351855 | 1.357837 | 1.380456 | 1.378648² |
| worker-SNR              | 1.349991 | 1.359412 | 1.396235 | — |
| block-RMS               | 1.361138 | 1.360397 | 1.371002 | — |
| block-Yogi              | 1.360425 | 1.359039 | —¹       | — |
| curvature-aware momentum| 1.382600 | 1.362800 | 1.369068 | 1.406944 |
| Iso-C                   | 1.351354 | 1.360359 | 1.398720 | — |
| capnest v2.1            | 1.375451 | 1.375779 | 1.376484 | — |

**dynamic-H:** fixed Nesterov-μ0.9 destabilizes at horizon switches (post-switch
direction cosine **0.374** vs SGD **1.00**, capnest 0.873); tuned SGD wins, best
controller trails by **+0.011**.

## capped-nesterov-wsub (EXP2.43, directional successor to curv)

The directional worker-subspace variant, judged head-to-head against curv:
- wsub-h16 = **1.3787** vs SGD-0.28 1.351855 → **+0.027** (loses in the poison corner).
- Codex validated the implementation as faithful (correct E_b, b_perp, per-worker
  s_i, ¼-power cap) and confirmed the run used `--outer-lr 0.147`, so the result
  is genuine. Its scale-flawed disagreement cap barely engages at H16 (E_b is
  quartic in delta scale → inverted binding), and the residual aligned
  capped-Nesterov still loses to SGD. Remaining arms in progress.

## New candidates (Codex brainstorm/lit-research; docs/OTHER_OPTIMIZERS.md)
Reviewed → fixed → re-reviewed before GPU. Results as they land.

### Tail-time primal averaging (exp2-tailavg) — FAIL/KILL
Token-weighted Polyak-Ruppert average of the final 25% of committed models,
θ_out = 0.5·θ_T + 0.5·θ_tail. Reconstructed from EXISTING SGD-0.28 captures
(H16 ← exp2-46 C-h16mu0, H64 ← vanilla-sync sgd028, H256 ← exp2-46 C-h256mu0);
no new training.
| cell | θ_out | θ_T (SGD-0.28) | Δ win |
|------|-------|----------------|-------|
| H16  | 1.352363 | 1.358285 | +0.0059 |
| H64  | 1.357980 | 1.359852 | +0.0019 |
| H256 | 1.371549 | 1.371557 | +0.00001 |
Gate FAIL (need >0.018 on ≥2 cells; H16 <0.009 kill fires). The gain is GENUINE
off-trajectory averaging, NOT shrinkage — survives the effective-LR control
(beats the norm-matched-LR checkpoint at every cell; pullback cosine ≈ −0.40, not
−1). Horizon tilt as predicted (grows as H shrinks). An order of magnitude below
the product bar. Matches the spec's P≈14% + honest-negative note.

### SCAFFOLD-lite inner control variates (exp2-51) — FAIL/KILL
Δ = amount SCAFFOLD-lite's loss is BELOW the matched SGD-0.28 control (paired
within each workload's own eval set; higher Δ = better). Correctness gate PASSED
(control-arm zero-sum exactly 0.0/0.0 → implementation provably unbiased).
| cell | SCAFFOLD | control | Δ (better by) |
|------|----------|---------|---------------|
| **H16-IID** | 1.47322 | 1.54547 | **+0.0722** |
| H64-IID | 1.53250 | 1.54859 | +0.0161 |
| H256-IID | 1.54791 | 1.54865 | +0.0007 (≈null) |
| HET-H64 | 1.43882 | 1.45705 | +0.0182 (clears) |
Gate needs a >0.018 win on H256 AND heterogeneous. H256 is ~null (+0.0007) → the
gate's long-H requirement is unmet → **KILL as an outer-momentum fix** (no change
to "ship SGD-0.28"). The implementation is clean (control-arm zero-sum exactly
0.0/0.0 → provably unbiased), so the null is real, not a bug.

**But the horizon tilt is the OPPOSITE of the design hypothesis, and real.** With
the H16 cell now in, the IID gain is cleanly MONOTONIC-DECREASING in H:
+0.0722 (H16) → +0.0161 (H64) → +0.0007 (H256). A 3-point monotone trend, H16 at
~8× the noise floor, is not single-seed noise (this supersedes the earlier
"H64 +0.016 was noise" read, made before the H16 point existed). The control
arms are flat (~1.545–1.549 across all H) while the scaffold arms improve as H
shrinks — a genuine short-horizon effect. This makes SCAFFOLD-lite the STRONGEST
short-H helper in the campaign (cf. worker-SNR −0.0019, Iso-C −0.0078 at H16) and
places it squarely in the unifying no-free-lunch pattern: consensus/spatial
corrections help short-H, break/vanish at long-H. The gate hunted the payoff at
long-H (null here) and used short-H only as a do-no-harm guard — where the actual
+0.072 sat. Flagged for a short-H effective-LR control arm (NOT just a
replication seed) before it is claimed as a mechanism result: the H16 gain is
large AND opposite to the design's grow-with-H prediction — the signature of a
merge-frequency (∝1/H) effective-LR/variance confound rather than the intended
drift correction. The zero-sum gate already excludes a merge-level bias. No extra
compute spent yet (gate already decided).

### Guarded Chebyshev-SGD (exp2-48) — FAIL/wash
Secant-gated Chebyshev acceleration on the outer step. Δ vs SGD-0.28
(candidate − SGD; positive = worse). Raw eval loss/token, with a lag-shuffled
control column:
| cell | SGD-0.28 | cheb_guard | cheb_guard_shuffled |
|------|----------|-----------|---------------------|
| H16  | 1.3725   | 1.3733 (+0.0008) | 1.3713 |
| H64  | 1.3559   | 1.3584 (+0.0025) | 1.3544 |
| H256 | 1.3572   | n/a¹             | n/a¹  |
cheb_guard is within ±0.0025 of SGD (worse at H16/H64), and the lag-shuffled
control is marginally BETTER than the real arm → no real acceleration signal.
Wash; never beats SGD. Confirmed again as the outer arm of the Muon factorial
(exp2-50) under both inner optimizers.

### Trust-Krylov secant trust-region (exp2-49) — no win at H16; H64/H256 in progress
Low-rank secant (delayed-plant) trust region on the transverse component; norm
PINNED to SGD so it cannot buy an effective-LR change. Raw eval loss/token:
| cell | SGD-0.28 | trust_krylov | trust_krylov_shuffled |
|------|----------|--------------|-----------------------|
| H16  | 1.36643  | 1.36688 (+0.00045) | 1.36511 |
| H64  | in progress | | |
| H256 | in progress | | |
No win at H16 (trust_krylov +0.00045 WORSE; real-vs-shuffled −0.00177, needs
≥+0.009). Norm diagnostic: realized ‖d‖/‖g‖ = 0.2800 for all three arms (= the
0.28 outer-LR); trust_krylov mean ‖d‖ 1.3387 is only +0.5% over SGD's 1.3322 —
norm-pinning works, no step-size artifact, no norm-matched control warranted.

### Muon inner-optimizer 2×2 factorial (exp2-50) — double-negative
inner ∈ {AdamW, Muon} × outer ∈ {SGD-0.28, guarded-Chebyshev}. Raw eval
loss/token (lower = better):
| cell | AdamW+SGD | Muon+SGD | AdamW+Cheb | Muon+Cheb |
|------|-----------|----------|------------|-----------|
| H16  | 1.3936 | 1.4218 | 1.3963 | 1.4284 |
| H64  | 1.3917 | 1.4306 | 1.3923 | 1.4350 |
| H256 | 1.3979 | 1.4368 | in progress | in progress |
| H64 inner-lr-hi (2.4e-3) | 1.3931 | 1.5417 | in progress | in progress |
Muon-inner strictly worse than AdamW-inner at every horizon (+0.028/+0.039/+0.039;
+0.149 at high inner-LR where Muon nearly fails to train). Guarded-Chebyshev outer
≈ SGD-0.28 outer under BOTH inner optimizers (worst gap +0.0066). MuLoCo's
"inner optimizer changes the pseudo-gradient enough to flip the outer verdict"
does not hold here; the outer-Chebyshev wash is confirmed under a 2nd inner opt.

## Notes
¹ block-Yogi H256 lost to node preemption mid-run; conclusion robust without it.
² curvature-aware momentum and Iso-C are compared against their **same-node** SGD
  anchors (curv: 1.352342 / 1.357306 / 1.379344 / 1.378648; Iso-C: 1.359198 /
  1.360265 / —) because they ran at a different outer-LR/config than the H-sweep
  refs. All other Δ are vs the H-sweep SGD-0.28 refs above.

Sources: exp2-41 (worker-SNR/block-RMS/block-Yogi), exp2-42 (curvature-aware),
exp2-40 (Iso-C), exp2-36 (capnest v2.1), exp2-43 (wsub), dynamic-H screen.
Raw data: s3://yeto-exp-artifacts-533462777468-us-west-2/probecommit-resume-20260710/exp2-*.
