# Morning Summary — COMPLETE (all overnight verdicts landed; fleet down, $0)

## ONE-LINE STATUS (~13:56 UTC / 06:56 PDT)
All queued science finished. Fleet fully wound down (0 AWS P-nodes any region,
0 Verda nodes, $0/hr). Every headline verdict is secured and durable in S3.
The paper's spine is complete: mechanism proven + causally closed, confounder
ruled out, DiLoCo-intrinsic under true barrier, product = memoryless SGD-0.28.

### THE FIVE THINGS THAT MATTER (read these first)
1. **Poison is DiLoCo-INTRINSIC.** Barrier arm A reproduces the +0.100 H16
   poison with correct version-matched deltas (A≈B≈C, all ~1.460 vs ~1.360
   baseline). Staleness, asynchrony, AND current-anchor differencing all ruled
   out as necessary causes. [exp2-46 + exp2-45]
2. **Confounder dead.** Current-anchor (C) ≡ version-matched (B) at every corner
   (drift ≡ 0 under strict quorum); poison anchor-independent. [exp2-46]
3. **Mechanism causally closed.** Damage is pseudo-gradient CORRELATION geometry,
   not norm: at fixed applied merged norm the inner-LR spread is +0.163; momentum
   dose-response monotone (+0.243 mu0→mu0.9 at high inner-LR). [exp2-39]
4. **Horizon crossover quantified & clean.** Poison collapses ~10× from +0.100
   (H16) to ~+0.01 (H256). Same-sign, confounder-free. [exp2-46]
5. **Product settled: ship memoryless SGD-0.28.** Every outer-optimizer family
   (worker-SNR, block-RMS/Yogi, curvature-aware, Iso-C, capnest) FAILS the
   product gate; each wins only in the horizon regime the dynamics diagnostic
   predicts, none uniformly. "Remove momentum" > "engineer a cleverer optimizer."
   One untested lead: directional worker-subspace (wsub). [exp2-40/41/42]

## OPS FINAL
- Fleet DOWN: terminated stranded curv p4de (i-014868a0, had crashed at
  rank16-h16 ~04:05, billing idle since 06:33) and the anchor p4d (i-0c5b0f32,
  arm-A H16 corner captured, past 5h backstop). Zero nodes running now.
- Data protection: both surviving nodes had dead sidecar S3 sync; I manually
  pushed results + ran detached sync loops, so all arm-A + H256 results are
  durable in s3://…/exp2-46-anchorctl before teardown.
- Verda still blocked by an account storage-quota wall (needs a quota raise to
  finish iso-C's last 5 sensitivity arms — non-critical; verdict already firm).

---
# (detailed verdicts below)
# Morning Summary (details — updated as overnight verdicts land)

## ★ CAUSAL-CLOSURE + CONFOUNDER VERDICTS (landed ~10:30 UTC / 03:30 PDT)

### exp2-39 MEDIATION — COMPLETE 7/7 (paper-critical: closes the mechanism)
delta_norm_ref R = 10.4436 (all norm-matched arms apply this fixed merged norm).
1. **NORM-MATCH:** at fixed applied merged norm = R, spread across inner-LR =
   normmatch-hi(1.531921) − normmatch-lo(1.368771) = **+0.16315 ≫ 0.009**.
   → the inner-LR-dependent momentum damage is POISON (pseudo-gradient geometry
   / correlation), NOT mere norm amplification. Matching the applied norm does
   NOT remove it. (normmatch-base 1.418508 ≈ base-ref 1.418466 → norm-ref
   control inert at reference inner-LR, as designed → clean control.)
2. **MOMENTUM DOSE-RESPONSE (inner-LR 2e-3):** mu0 1.382111 < mu0.5 1.430336 <
   mu0.9 1.625482 — monotone. Damage is momentum-induced, grows with mu,
   +0.243 from mu0→mu0.9 at high inner-LR.
   → Momentum damage is mediated by pseudo-gradient CORRELATION geometry set by
   the inner optimizer's step size, not by delta magnitude. This is the
   single-Z-falsified, two-channel causal closure the paper needed.

### exp2-46 3-ARM ANCHOR CONTROL — confounder RULED OUT (poison corner done)
**H16 corner COMPLETE — all 4 arms (drift ≡ 0 on every arm):**
| arm | mu0 (baseline) | mu0.9 (poison) | poison Δ |
|-----|----------------|----------------|----------|
| C (current-anchor)  | 1.358285 | 1.457329 | **+0.099** |
| B (version-matched) | 1.360637 | 1.460016 | **+0.099** |
- **Airtight causal control:** C vs B is statistically identical at BOTH momentum
  levels (Δ = 0.0024 at mu0, 0.0027 at mu0.9; both < noise floor 0.009), and the
  +0.099 short-horizon momentum poison appears IDENTICALLY in the version-matched
  arm B. mu0 baseline (~1.359) sits right at the SGD-0.28 ref (1.3519). The
  poison is fully anchor-independent at H16. Under strict quorum the anchor-drift
  term is
  EMPIRICALLY zero (every push's base_version == fragment's current version;
  median local-delta norm ~4.63 so the zero is real, not a dead signal).
  → The observed poison is NOT caused by current-anchor differencing.
- Claim discipline (Codex adversarial check): drift==0 is EMPIRICAL to THIS
  scheduler (fixed-window, apply-broadcast-before-pull), NOT a hard invariant of
  strict quorum. Do not over-generalize. The native-vs-anchor separation is also
  machine-proven in Lean (commit b10d69d). Poison attributed to native outer
  momentum; barrier reference = arm A (pending) + exp2-45.
**H256 corner COMPLETE — momentum poison collapses at long horizon (C≡B again):**
| arm | mu0 | mu05 | Δ(mu05−mu0) |
|-----|-----|------|-------------|
| C (current-anchor)  | 1.371557 | 1.381074 | +0.0095 |
| B (version-matched) | 1.370819 | 1.380580 | +0.0098 |
- C vs B identical at H256 too (Δ 0.0007 at mu0, 0.0005 at mu05 — both ≪ 0.009).
- **Horizon-dependence quantified:** the momentum penalty collapses ~10× from
  +0.099 (H16) to ~+0.01 (H256, ≈1 noise floor). Same-sign, confounder-free
  crossover: momentum is catastrophic at short horizon, ~neutral at long horizon.
**★ BARRIER ARM A — poison corner IN: the pathology is DiLoCo-INTRINSIC.**
A-h16mu09 (true lockstep barrier + version-matched delta) = **1.461825**.
Compare the H16-mu0.9 poison corner across all three arms:
| arm | semantics | H16-mu0.9 loss |
|-----|-----------|----------------|
| A | barrier + version-matched (true vanilla DiLoCo) | 1.461825 |
| B | non-barrier + version-matched | 1.460016 |
| C | non-barrier + current-anchor (our impl) | 1.457329 |
- **A ≈ B ≈ C (all within ~0.005).** Per the pre-registered interpretation table
  (docs/ANCHOR_DRIFT_CONTROL.md), A,B,C all showing the poison ⇒ **outer-momentum
  pathology is DiLoCo-family-INTRINSIC** — NOT caused by non-barrier overlap, NOT
  by current-anchor differencing. Directly corroborates exp2-45.
- **A-h16mu0 baseline CAPTURED = 1.361698** → barrier poison Δ = 1.461825 −
  1.361698 = **+0.1001**. Complete 3×2 H16 corner (all baselines ~1.360, all
  poison ~1.460, all Δ ≈ +0.10):
  | arm | mu0 | mu0.9 | Δ |
  |-----|-----|-------|---|
  | A barrier+ver-matched | 1.361698 | 1.461825 | +0.100 |
  | B non-barrier+ver-matched | 1.360637 | 1.460016 | +0.099 |
  | C non-barrier+current-anchor | 1.358285 | 1.457329 | +0.099 |
- ⇒ We CAN now write the strongest form: the short-horizon outer-momentum poison
  appears under true lockstep barrier DiLoCo with correct version-matched delta
  semantics, at zero injected delay. Staleness, asynchrony, and current-anchor
  differencing are all ruled out as necessary causes.

### exp2-42 CURVATURE-AWARE MOMENTUM (T4) — beats SGD ONLY at H256, fails gate
| cell            | curv     | co-run SGD | Δ(curv−sgd) |
|-----------------|----------|-----------|-------------|
| H16             | 1.382600 | 1.352342  | +0.0303 ✗   |
| H64             | 1.362800 | 1.357306  | +0.0055 ✗   |
| H256            | 1.369068 | 1.379344  | **−0.0103 win** |
| innerlr-hi-h64  | 1.406944 | 1.378648  | +0.0283 ✗   |
- Curvature cap contains the momentum blow-up (plain Nesterov mu0.9 → 1.625 in
  the inner-lr-hi cell) but SGD still wins everywhere except H256. Critically it
  LOSES in the inner-lr-hi stress cell — the exact cell T2/T4 predicted
  curvature-awareness should help MOST. Diagnosis: scalar λ̂ is anisotropy-blind.
  → FAILS product gate; points directly at DIRECTIONAL worker-subspace (wsub) as
  the one untested candidate. (rank16-h16 stress cell still pending.)

## HEADLINE VERDICTS (as of ~00:15 PDT / 07:15 UTC)

### CONFIRMED / STRENGTHENED
1. **LR-gate PASSES (full-parameter SmolLM2, 27 cells).** The momentum penalty
   is NOT rescued by the eta sweep at ANY (H,mu) — best-eta mu>0 always worse
   than the mu0 best (penalties +0.058 to +0.406). This kills the advisor's
   biggest kill-criterion ("it's just LR tuning") on a full-parameter model.
   The crossover is real, not an LR artifact.
2. **Two-term law: mu09 loses +0.0435 at MATCHED effective-LR** (aligned-only
   predicted ~0). Aligned-only refuted; variance/transverse term is real.
3. **Crossover replicated on 3 seeds** (223/251/283): mu0 wins short H, mu05
   wins long H, all same sign.
4. **Rank-16: poison 6-7x LARGER** (+0.528/+0.435 vs rank-2 0.086/0.062) and
   rho higher (0.356 vs 0.25). Rank-2 "artifact" objection refuted from two
   angles (loss AND dynamics-geometry-invariant per the diagnostic).
5. **Dynamic-H systems result:** fixed Nesterov mu09 DESTABILIZES at horizon
   switches (post-switch direction cosine 0.374 vs sgd 1.00, capnest 0.873);
   capnest absorbs switches untuned (0.011 behind sgd, 0.056 ahead of nesterov).
6. **Lean: 12 sorry-free theorems** — T1-T4; poison is curvature x transverse,
   proven; geometry-blind scalar controllers provably can't beat SGD.
7. **Dynamics diagnostic:** field is stiff (cond 7.7-20), mildly rotational
   (0.3-0.6), spread modes -> Chebyshev/Krylov recommended. Reframing confirmed.

### IMPORTANT HONESTY CAVEATS (must shape paper claims)
1. **Inner-LR poison is NOT mediated by rho.** Loss swings 1.362->1.625 across
   inner-LR (0.0005/0.001/0.002) while rho stays FLAT (0.234 vs 0.232). So rho
   is NOT the sufficient/universal state variable — the inner-LR axis decouples
   from rho. This is exactly the "H is proxy but which geometry variable is THE
   state variable is unclosed" gap. Likely mediator = delta norm / curvature,
   not rho. The EXP2.39 geometry panel + collapse test addresses this.
2. **Noise floor: single-worker neg-merge rate 0.3375** — below the merged
   band (0.3875-0.4125), far above the 0.10-0.25 "interference is real" window.
   Read: per-step negative-merge is largely small-eval MEASUREMENT NOISE, not
   merge interference. BUT merged sync still wins final loss (1.3605 < 1.3781).
   So the "~35% harmful merges" framing must be retired; the measurement-wall
   thesis is reinforced.
3. **Current-anchor confounder** (docs/ANCHOR_DRIFT_CONTROL.md): our variant is
   strict-quorum NON-BARRIER CURRENT-ANCHOR streaming DiLoCo. Cannot claim
   "vanilla DiLoCo has this." The 3-arm control (exp2-46) + Lean anchor-drift
   modules are running to separate native-poison vs anchor-contamination.
4. **Staleness dose-response:** k4 = +0.005 vs k0 — small but first monotone
   commit-lag signal. "Staleness second-order" now has direct version-lag data.

## PRODUCT DIRECTION (bake-off) — FINAL: keep SGD-0.28; NO candidate passes gate
exp2-41 memoryless preconditioner bake-off (8/9 arms; byogi-h256 lost to node
death but conclusion robust). Product gate = paired win >0.018 on ≥2 workloads
AND never worse than 1 noise floor (0.009) anywhere; no per-H tuning.
| candidate | H16 | H64 | H256 | verdict |
|---|---|---|---|---|
| worker-SNR (consensus) | −0.0019 win | +0.0016 | **+0.0157 ✗** | helps short, breaks long |
| block-RMS (2nd-moment) | **+0.0092 ✗** | +0.0026 | −0.0095 win | opposite tilt: helps long |
| block-Yogi (2nd-moment)| +0.0085 | +0.0012 | (incomplete) | ~tie |
| curvature-aware momentum | +0.0303 ✗ | +0.0055 ✗ | −0.0103 win | H256-only |
| Iso-C (spectral, exp2-40)| −0.0078 win | +0.0001 tie | +0.018 ✗ | helps short, breaks long |
- **NO candidate clears the gate.** Every one is worse than SGD by ≥1 noise
  floor at some horizon; best single win (brms-h256 −0.0095) is under threshold.
- Scientifically clean finding: candidates have OPPOSITE horizon tilts
  (consensus helps short-H / hurts long-H; second-moment + curvature hurt
  short-H / help long-H) — each wins exactly where the dynamics diagnostic
  predicted, none uniformly. Worker-consensus (the most original idea) has the
  WORST horizon-robustness (over-damps long-H: inner drift inflates cross-worker
  variance → q_l collapse). Needs H-aware calibration to productize.
- **PRODUCT CALL: ship memoryless SGD-0.28.** No outer optimizer — momentum,
  spatial-consensus, second-moment, or scalar-curvature — beats it across
  horizons. Strong clean paper message: "remove momentum" > "engineer a cleverer
  outer optimizer." One untested best-bet remains: DIRECTIONAL wsub (worker-
  subspace curvature), predicted by T2/T4 + the anisotropy-blind curv failure.
- capnest v2.1 (curvature-blind, stable mu_par): H-invariant (range 0.001) but
  never beats plain SGD (+0.024 H16). Safety device, not a winner.
- cheb-sgd (Chebyshev, memoryless spectral): implemented, staged, not run.
- iso-C (exp2-40): DONE 5/10 (halted by Verda storage-quota wall, not a science
  problem; fleet empty, $0). Iso 1.351354 vs same-node SGD-0.28 anchor 1.359198
  at H16 (−0.0078, still < 2× floor); tie at H64; +0.018 worse at H256. Same-node
  eta0.28 anchors confirmed the historical-ref confound (hist refs are eta0.175;
  eta0.28 SGD lands +0.0025–0.0074 above them). Spectrum-flattening alone does
  NOT robustly beat SGD → drops below the consensus/spatial priorities. Remaining
  5 sensitivity arms resume-ready (acquire_40iso.sh) if storage is restored.

## RUNNING (as of ~11:12 UTC / 04:12 PDT)
- exp2-46 anchor control (AWS p4d spot i-0c5b0f32, 44.248.39.254): ALIVE,
  GPUs 28-35%, tmux exp246 up. H16 corner DONE (4/4, now in S3). Currently on
  H256 corners → then barrier arm A (the cleanest lockstep-DiLoCo reference).
  NOTE: node's sidecar S3 sync was dead (S3 frozen at 04:03); I manually pushed
  the H16 corner + installed a fresh detached 300s sync loop (pid 20466) so
  H256/armA are protected against spot preemption. cost_guard backstop ~13:52.
DONE overnight: exp2-39 mediation (7/7 causal closure), exp2-41 bakeoff (8/9,
verdict firm), exp2-42 curv T4 (8/8 core+stress; rank16-h16 crashed, non-crit,
verdict firm), exp2-40 iso-C (5/10, verdict firm, halted by Verda quota),
exp2-45 barrier crossover (4 arms — reproduces), Lean anchor-drift separation
(b10d69d), Lean T1–T4.

## OPS ACTIONS (this cycle, ~11:10 UTC)
- TERMINATED stranded curv node i-014868a0 (p4de): all 8 GPUs 0%/0 MiB, no tmux,
  crashed at rank16-h16 ~04:05, had been billing idle since 06:33 UTC launch.
  exp2-42 verdict already complete without it → no science lost.
- Verified only 1 AWS node now billing (anchor p4d). Verda 0 nodes.

## OPS
- Storage: earlier babysitter freed ~3.9TB detached Verda volumes (unblocks
  provisioning). Balance ~$60, autopay ON. Cost guard: straggler>5h kill,
  global 8h wind-down, node ceiling 7.

---
## FOLLOW-ON (post-morning, in progress as of ~15:45 UTC 2026-07-13)
User greenlit the CTTN oracle test (the one remaining "can anything beat SGD?"
experiment). Status:
- CTTN math core DONE + golden-trace validated (yeto/cttn.py, scripts/test_cttn.py,
  commit aea1f67). Design of record: docs/CTTN_DESIGN.md (commit 6ff354f).
- Architecture mapped: CTTN rides the action-probe sidecar (only merge-time
  torch+autograd+held-out-data channel). Moderate change, not a refactor.
- NEXT: HVP autograd de-risk in the sidecar -> block-Lanczos wiring -> Rust
  plumbing -> local parity -> 24-run pre-registered GPU campaign.
- IN FLIGHT: exp2-43 wsub bake-off on 1x p4d (i-05fc6ba7, us-west-2) — the
  "even the directional variant fails" data point + inverted-binding diagnostic.
  Predicted-dead by the Codex pre-mortem. Runner agent owns it + self-teardown.
