# Paper Idea: Staleness Is Not the Poison

**Working title:** *Staleness Is Not the Poison: The Measurement Wall and
Memory Failures in Local-Update LLM Training*

**Alt titles:** *The DiLoCo Poison* / *You Can't Probe Your Way Out: Limits
of Merge-Time Control in Local SGD*

**One-line thesis:** In DiLoCo-style training, the two things everyone
engineers against (staleness) and for (outer momentum, smart merge
selection) are exactly backwards: staleness is nearly free, momentum is the
poison, and merge-time selection sits behind a quantifiable
signal-to-noise wall that no affordable probe can cross — except early in
training, where the wall predictably lifts.

---

## The reframe (why this could be a best paper, not a good paper)

The async-distributed-training literature is organized around a folk
assumption: *the enemy is staleness*, so we buffer (FedBuff), correct
(delta rules), weight by age, and gate merges. This paper inverts the
picture with three claims, each already partially evidenced:

1. **Staleness is nearly free.** Injecting 0–2400 ms of delay into a
   strict-quorum Qwen3.5-9B run changes final held-out loss by ~0.0003 and
   leaves the harmful-merge rate statistically unchanged (0.3625 async vs
   0.3875 sync). The phenomena attributed to asynchrony are properties of
   merging independent local trajectories at all.

2. **The real poison is optimizer memory at the wrong horizon.** Outer
   Nesterov μ=0.9 — the standard DiLoCo recipe — loses ~0.04–0.06 loss at
   short sync horizons, and the damage decomposes cleanly as
   *(directional memory rotation) × (realized step scale)*: rotated steps
   (cosine ≈ 0.77 to current evidence) are harmless when small (DC-matched
   μ=0.9 ties the best arm) and poisonous at working scale (norm-matched
   μ=0.9 loses 0.04 where a 58% scale change alone moves SGD by 0.002).

3. **Merge-time selection is behind a measurement wall.** There is real
   oracle headroom (~0.002/step), but the per-decision gap between merge
   actions falls below the noise floor of any affordable evaluation probe.
   Measured on 240 replay groups: median action gap 0.00095 vs median
   probe SE 0.00138. Early in training the gap is 0.00294 — above the
   floor — which is precisely the one regime where a frozen selector
   captured 56–84% of headroom. Ten independent selection strategies
   (scores, kNN, consensus, robust aggregation, LOO probes) all landed at
   chance-level within-group ranking; the wall explains every one of them
   at once, and predicts when selection *can* work.

The best-paper shape: a unifying quantitative principle (the wall), a
mechanism with a falsifiable interaction law (rotation × scale), a folk
assumption overturned (staleness), and a constructive escape hatch (below).

## The constructive turn: adapt from free signals, not probes

The wall says eval-based merge control cannot pay for itself: resolving a
gap δ with per-block noise σ needs ≈ (zσ/δ)² eval blocks; at late-training
δ ≈ 8×10⁻⁴ and σ_block ≈ 0.005–0.01 that is 100–600 blocks (13k–80k eval
tokens) per action per decision — of the same order as, or more than, the
8,192 training tokens the decision governs, multiplied by the action count.
The probe must out-spend the step it is trying to improve.

But the *tape* is free. Realized step geometry — round-to-round merged
update autocorrelation, history/current norm ratio, direction cosine —
costs nothing and directly measures the quantity momentum bets on:
persistence of the merged direction. Proposed controller (**AutoDiLoCo**):
set outer momentum each round from the measured direction autocorrelation
(μ̂_t ≈ clip(cos(ḡ_t, ḡ_{t−1}), 0, μ_max)) and rescale to hold realized
step norm at the memoryless-SGD reference. Prediction from the mechanism:
this recovers memoryless SGD at short H (measured cosine is low), recovers
classic DiLoCo at long H (persistent direction), and never pays the
rotation × scale penalty. Zero eval data, zero extra communication.

## Contributions as they would appear in the paper

1. **The measurement wall.** A sample-complexity argument for merge-time
   action selection, with an empirical crossover validated on 600+ decision
   groups across two model scales: selection works iff the action gap
   exceeds the affordable-probe noise floor, which happens only in early
   training. Retro-explains a 10-experiment negative-result series.
2. **Memory poison law.** Damage ≈ rotation × scale, established by a
   6-point sync decomposition grid and an H-sweep showing the
   helpful↔harmful crossover of outer momentum as sync horizon grows, with
   tape autocorrelation as the predictive statistic.
3. **Staleness is (nearly) free** at practical delay scales, isolated from
   merge interference by matched sync/async runs and a single-worker noise
   floor.
4. **AutoDiLoCo**, a probe-free adaptive outer optimizer derived from (1)+(2):
   matches or beats the best fixed recipe across the entire H-sweep without
   per-configuration tuning.

## Status update (2026-07-12): E2 and E3 executed — both predictions confirmed

EXP2.25 ran the H-sweep and wall curve (see `docs/EXP2_25.md`):

- **Crossover confirmed** between H=64 and H=256: μ=0 wins at H∈{16,64},
  μ=0.5 wins at H=256 by 0.0131; the μ=0.9 penalty decays monotonically
  (0.086 → 0.062 → 0.030) toward the DiLoCo design regime. Tape geometry
  tracks it: μ=0.9's applied-step cosine heals 0.752 → 0.872 as H grows.
- **Wall saturation confirmed:** anchor probe 2→8 panels lifts captured
  headroom 44.2% → 53.8%; 8→16 panels adds 0.2 points. Measurement stops
  buying selection quality exactly as the gap-below-noise-floor analysis
  predicts.
- **Mechanism upgraded (and simplified) by the autocorrelation data:**
  merged-delta persistence ρ *falls* with H (0.982 / 0.925 / 0.734 at
  H=16/64/256) — inverting the naive horizon story — and the single law
  η_eff = η/(1−μρ) retrodicts all nine grid cells plus EXP2.24's
  DC-matched control (7/7 quantitative checks; see EXP2_25.md). The
  poison is μρ→1 amplification; tape rotation/oscillation is its symptom,
  with a smaller residual oscillation penalty at matched mean scale.
  AutoDiLoCo v2 becomes a *calibrated* controller — hold η/(1−μρ̂) at the
  tuned effective step — rather than a heuristic. This tightens the
  paper: one measured, free statistic (ρ) explains the crossover,
  prescribes the fix, and prices DiLoCo's μ=0.9 recipe as a bet that
  ρ(H) is small.

Remaining for the paper: E1 noise floor, autocorrelation extraction from
the retained captures (the AutoDiLoCo control signal), norm-matched H=256
control, second seed on crossover corners, E4 generality, E5 controller.

## Experiment matrix (beyond what exists)

| Block | What | Why | Cost (spot) |
|---|---|---|---|
| E1 noise floor | Single-learner one-step negative rate on same evals; decompose interference vs stochastic | Load-bearing control for claim 3 | ~$3, computable partly from retained captures |
| E2 H-sweep | H ∈ {16, 64, 256, 1024} inner steps × μ ∈ {0, 0.5, 0.9, μ̂} × 2 seeds, sync | The crossover; core of claims 2+4 | ~32 arms ≈ $40–80 |
| E3 wall curve | Probe-size sweep (2–64 blocks) × training phase; plot selection AUROC vs gap/σ ratio | Turns the wall into a measured curve | replay-only, ~$10 |
| E4 generality | One full-finetune (no LoRA) config + one second model/dataset, key corners only | Kills the "LoRA rank-2 artifact" review | ~$60–120 |
| E5 AutoDiLoCo | Controller vs best-fixed per H, fresh seeds, predeclared gates | The constructive win | ~$40 |
| E6 scale probe (stretch) | One 8-learner run at the best/worst corners | Breadth | ~$50 |

Total ≈ $200–300 on current spot infra; every protocol already exists in
`compare_diloco.py` + replay tooling. E1+E2 alone are enough for a credible
submission; E5 is what pushes it toward award territory.

## What kills it (pre-registered threats)

- E1 shows single-worker negative rate ≈ merge negative rate → claim 3
  collapses to "small-eval noise", and the wall becomes the *only* story
  (still a paper, weaker one).
- The H-sweep shows no crossover (momentum bad everywhere) → DiLoCo's own
  ablations are contradicted; check inner-optimizer / full-finetune
  interaction before believing it.
- AutoDiLoCo ties fixed-best everywhere → demote to diagnostic; the wall +
  poison mechanism still carry the paper.
- Effects vanish at full finetune → paper narrows to PEFT-DiLoCo; scope
  honestly.

## Positioning (checked against arXiv/OpenReview through July 2026)

Four groups circle the same latent variable — memory-horizon matching —
without the causal decomposition, the schedule-independence result, the
wall, or the measured-autocorrelation controller:

- **DiLoCo / async-DiLoCo (Douillard et al., 2311.08105):** establish
  outer Nesterov 0.9 at H≈50–500 full-finetune; we show the recipe's value
  is a horizon bet and give the statistic that prices the bet.
- **DeepMind Async Local-SGD (2401.09135):** found the "momentum
  challenge" but attributed it to staleness and fixed it with Delayed
  Nesterov + DyLU. Our sync replication shows the attribution is wrong:
  the poison needs no staleness. Their fix does not touch rotation×scale.
- **Outer-Momentum Restarting (2605.28585, Jun 2026):** *closest work.*
  Periodic outer-momentum restarts widen stable (LR, μ) ranges; their
  stated future work is adaptive restart rules vs communication period —
  i.e., AutoDiLoCo. Our EXP2.19 already shows conflict-triggered restart
  is only a partial repair that loses to memoryless SGD at short H.
  Urgency signal: adjacent groups are one step from the controller.
- **MT-DAO (2510.05361):** diagnoses time-scale mismatch from the opposite
  corner (momentum too short for long H; adds slow momenta). Combined with
  our short-H result, evidence for one unified matching law — neither side
  has it.
- **DES-LOC (2505.22549):** matches momenta half-lives to sync intervals,
  but for communication savings, not the quality poison.
- **SparseLoCo (2508.15706):** outer momentum detrimental under high
  sparsification (conflict with error feedback) — another unexplained
  instance our law should cover.
- **FedBuff / staleness-weighted merging / Pseudo-Async Local SGD
  (2504.18454) / Decoupled DiLoCo (2604.21428):** engineered against or
  around staleness; consistent with but never stating our claim that
  staleness is second-order and memory is the operative poison.
- **Merge-selection prior art (2604.08056 held-out outlier gating,
  FedGSNR):** heuristic probe-based gating with no cost analysis. The
  measurement wall — gap below the noise floor of any probe cheaper than
  the step it governs, with chance-level ranking on 600+ groups —
  appears unclaimed.
- **Gradient-conflict (PCGrad etc.) / Byzantine-robust aggregation:**
  conflict is known; the decision-theoretic unmeasurability result and the
  m≈4–12 within-group noise explanation for robust-rule ties are new.

## Assets already in hand

Six-point sync decomposition grid; sync-vs-async matched pair with
identical tooling; 240-group gap/SE measurements; 600+ group chance-level
selection results across SmolLM2-135M and Qwen3.5-9B; early-window 83.5%
headroom capture; complete capture tapes for retro-computing
autocorrelation and E1/E3 without new training. Docs: EXP2.14, EXP2.19–24.
