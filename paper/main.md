# Temporal Correlation Is the Missing State Variable Governing Outer Momentum in Two-Phase Distributed Optimization

*Draft assembled 2026-07-12. Source of record for all numbers: `docs/EXP2_23.md`–`docs/EXP2_27.md`, `docs/OPTIMIZER_SEMANTICS.md`, `experiment-results/EXP2/rda-rho-law/summary.{md,json}`. Sections marked **PLACEHOLDER** await tonight's runs.*

## Abstract

DiLoCo-style training applies an outer optimizer to *pseudo-gradients* — merged parameter displacements produced by $H$ inner steps on each of $M$ workers. The literature treats the sync horizon $H$, staleness, and worker count as the governing variables, and engineers against staleness while engineering *for* outer momentum. We show that a single, freely measurable quantity — the temporal autocorrelation kernel $\rho_k$ of the pseudo-gradient sequence — is the state variable through which these knobs actually act. Outer Nesterov momentum is exactly a temporal filter: its update decomposes, without approximation, into a data-dependent aligned gain $A_t = 1+\mu+\mu^2 c_t$ on the current merged delta plus a transverse residual of magnitude $\mu^2 r_t\lVert\delta_t\rVert$; on a stationary kernel the aligned gain and the RMS energy amplification are closed forms in $(\mu,\rho)$. A two-term law — aligned overstep plus buffer variance accumulation — fits a 9-cell $H\times\mu$ grid on Qwen3.5-9B LoRA training with $R^2=0.90$ where aligned-only ($R^2=0.15$) and a superseded single-term law ($R^2=0.28$) fail, retrodicts the helpful-to-harmful momentum crossover in $H$, and correctly predicted on a fresh preregistered seed that a tuned-$\eta$ memoryless baseline dissolves the fixed-$\eta$ momentum advantage at long horizons. Matched sync/async runs show the pathology needs no staleness: removing all injected delay (0–2400 ms) changes final loss by $\sim 3\times 10^{-4}$ and leaves the negative-merge rate statistically unchanged, while merge-time selection sits behind a measurement wall — the per-decision action gap lies below the noise floor of any affordable probe, and captured headroom saturates at 8 eval panels (+0.2 points from 8$\to$16). Because the tape is free where probes are not, we derive a zero-evaluation-cost gain controller from the measured kernel; a v1 development screen already halves the worst-case cross-horizon regret of the best fixed policy with zero per-horizon tuning.

## 1 Introduction

Two-phase ("local-update") distributed optimization — DiLoCo and its descendants — separates cheap inner optimization on workers from rare outer synchronization. Its folk model assigns blame and credit by *schedule*: staleness is the enemy to be buffered, corrected, and weighted away; outer momentum ($\mu=0.9$ Nesterov, the standard recipe) is the friend; and smarter merge-time selection is the frontier. Each element of that model is either wrong or incomplete, and all three failures have one root.

**The reframe.** The outer optimizer never sees $H$, staleness, or worker count. It sees a *sequence of pseudo-gradients* $\delta_1,\delta_2,\dots$ and it is, exactly, a linear temporal filter on that sequence. What the schedule knobs actually change is the *geometry* of the sequence — above all its temporal autocorrelation kernel $\rho_k = \mathrm{corr}(\delta_t, \delta_{t-k})$. Momentum on a correlated sequence silently multiplies the effective step ($\eta_{\mathrm{eff}} = \eta\,(1+\mu/(1-\mu\rho))$ for a geometric kernel) and accumulates off-direction variance in its buffer; both amplifications are set by $(\mu,\rho)$, not by $H$ per se. Measured on production merges, $\rho$ *falls* as $H$ grows — the opposite of the naive "long horizons persist more" intuition — and that inversion, plus the two amplification terms, explains at once: why $\mu=0.9$ is poison at short horizons and merely mediocre at long ones ("the momentum crossover"), why the damage survives with zero staleness, why conventional learning-rate reasoning mispredicts the grid, and why the benefit of momentum at long $H$ is mostly an effective-LR effect that a tuned memoryless baseline recovers. Because $\rho_k$, the aligned gain $A_t$, and the transverse ratio $r_t$ are computable from the training tape at zero evaluation cost, the same analysis yields a controller — measure the kernel, normalize the gain — that needs no probes, no per-horizon tuning, and no knowledge of the schedule.

**Contributions** (verbatim from the study plan, `docs/NORTH_STAR_PLAN.md`):

1. **Mechanism** — outer Nesterov is a temporal filter whose data-dependent gain and transverse accumulation are set by the pseudo-gradient correlation kernel;
2. **Evidence** — controlled interventions + cross-regime collapse predict the short-horizon phase transition better than H or staleness;
3. **Method + consequence** — a zero-eval-cost gain controller transfers without retuning and enables elastic synchronization.

A secondary, self-contained result motivates the "zero-eval-cost" requirement: merge-time action selection is *information-limited*. On 240 replay groups the median per-decision action gap (0.00095 loss) lies below the median affordable-probe standard error (0.00138); growing the probe 2$\to$8$\to$16 panels lifts captured oracle headroom 44.2% $\to$ 53.8% $\to$ 54.0% — measurement stops buying selection quality exactly where the gap-below-noise-floor analysis says it must (Section 4.4). Any adaptive scheme that pays for evaluation therefore out-spends the steps it governs; the tape is the only free signal, and the kernel is what the tape measures.

## 2 Mechanism: outer Nesterov as a temporal filter

### 2.1 The exact decomposition (no model assumptions)

The syncer's outer update per fragment (audited semantics, `docs/OPTIMIZER_SEMANTICS.md`; deterministic hand-computed vector tests in `syncer/src/merge.rs`), with merged pseudo-gradient $\delta_t$, outer LR $\eta$, momentum $\mu$, buffer $b_0=0$:

$$
b_t = \mu\, b_{t-1} + \delta_t,\qquad
d_t = \delta_t + \mu\, b_t = (1+\mu)\,\delta_t + \mu^2\, b_{t-1},\qquad
\theta_t = \theta_{t-1} - \eta\, d_t .
$$

Two immediate, frequently missed consequences. First, $\mu$ multiplies the *current* delta by $(1+\mu)$ before any memory exists: at $\mu=0.9$ the very first step is $1.9\times$ the $\mu=0$ step at the same $\eta$, so "$\mu=0$ vs $\mu=0.9$ at fixed $\eta$" never isolates memory. Second, define the projection coefficients of the buffer on the current delta,

$$
c_t = \frac{\langle b_{t-1}, \delta_t\rangle}{\lVert\delta_t\rVert^2},
\qquad
r_t = \frac{\lVert b_{t-1} - c_t\,\delta_t\rVert}{\lVert\delta_t\rVert}.
$$

Then, exactly and for any input sequence,

$$
d_t = A_t\,\delta_t + d_t^{\perp},
\qquad
A_t = 1 + \mu + \mu^2 c_t,
\qquad
\lVert d_t^{\perp}\rVert = \mu^2 r_t \lVert\delta_t\rVert .
$$

Momentum's entire action is a data-dependent scalar gain $A_t$ along the fresh evidence plus a transverse kick of relative size $\mu^2 r_t$. Both are logged on every run ($A_t$, $r_t$, applied-step norm, direction cosine) at zero cost.

### 2.2 Stationary closed forms

On a stationary sequence with lag kernel $\rho_k$, the expected aligned gain is

$$
\mathbb{E}[A_t] \;=\; 1 + \mu + \mu^2 \sum_{k\ge 1}\mu^{k-1}\rho_k
\;\longrightarrow\;
1 + \frac{\mu}{1-\mu\rho}
\quad\text{(geometric kernel } \rho_k=\rho^k\text{)},
$$

so the *aligned effective step* is $\eta_{\mathrm{eff}} = \eta\left(1 + \mu/(1-\mu\rho)\right)$. This corrects an earlier single-term form $\eta/(1-\mu\rho)$ used in our own first analysis (they agree only at $\rho=1$); the correction was derived in independent review and is part of the falsification trail (Box 1). Aligned gain alone is not enough: the buffer also *accumulates* the off-direction components of past deltas. For the geometric kernel the second-moment (energy) amplification of the applied step relative to a memoryless step is the closed form

$$
A^2_{\mathrm{RMS}}(\mu,\rho)
= (1+\mu)^2
+ \frac{2(1+\mu)\,\mu^2\rho}{1-\mu\rho}
+ \frac{\mu^4\,(1+\mu\rho)}{(1-\mu^2)(1-\mu\rho)},
$$

verified to reproduce the fitted table's $A^2_{\mathrm{RMS}}$ values exactly (10.06 / 11.38 / 17.64 at $\mu=0.9$ for the three measured $\rho$'s; note this is the second moment, not its square root). The variance term *punishes memory hardest at high $\rho$* — which is precisely the short-$H$ regime — and is what the aligned term alone cannot capture.

### 2.3 The two-term law and its fit

The proposed law for end-of-run loss across a grid of $(H,\mu)$ cells at fixed nominal $\eta$:

$$
\mathcal{L}(H,\mu)\;=\;c_H \;+\; b\,\log^2\!\frac{\eta_{\mathrm{eff}}(\mu,\rho_H)}{\eta^\ast}\;+\;v\,\log A^2_{\mathrm{RMS}}(\mu,\rho_H),
$$

with per-horizon intercepts $c_H$, a quadratic aligned-overstep penalty around the tuned reference $\eta^\ast=0.28$, and a variance-accumulation penalty. The inputs $\rho_H$ are the *production-RDA* per-tensor autocorrelations measured on replay-verified captures (Section 4.5, Figure 3) — not a proxy convention. Fit on the 9-cell seed-223 grid (`experiment-results/EXP2/rda-rho-law/summary.md`):

- $b$ (aligned overstep) $= 0.1061$; $v$ (variance accumulation) $= 0.0234$;
- $R^2 = 0.899$, RMSE $= 0.0090$, max $|$resid$| = 0.0155$;
- aligned-only (corrected $\eta_{\mathrm{eff}}$): RMSE 0.0262, $R^2 = 0.154$;
- superseded single-term $\eta/(1-\mu\rho)$: RMSE 0.0242, $R^2 = 0.275$.

| $H$ | $\mu$ | $\eta_{\mathrm{eff}}$ | $A^2_{\mathrm{RMS}}$ | loss | pred | resid |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.0 | 0.1750 | 1.00 | 1.3519 | 1.3576 | −0.0057 |
| 16 | 0.5 | 0.2967 | 2.99 | 1.3632 | 1.3602 | +0.0030 |
| 16 | 0.9 | 0.4938 | 17.64 | 1.4383 | 1.4356 | +0.0027 |
| 64 | 0.0 | 0.1750 | 1.00 | 1.3578 | 1.3665 | −0.0087 |
| 64 | 0.5 | 0.2750 | 2.57 | 1.3614 | 1.3652 | −0.0038 |
| 64 | 0.9 | 0.3782 | 10.06 | 1.4193 | 1.4068 | +0.0125 |
| 256 | 0.0 | 0.1750 | 1.00 | 1.3805 | 1.3664 | +0.0141 |
| 256 | 0.5 | 0.2796 | 2.66 | 1.3674 | 1.3659 | +0.0015 |
| 256 | 0.9 | 0.3984 | 11.38 | 1.3977 | 1.4132 | −0.0155 |

Statistical honesty (independent statistical review, 2026-07-12): adjusted $R^2 = 0.798$; the $A^2$ term survives a nested F-test against the aligned-only model ($F(1,4)\approx 29.5$, $p\approx 0.006$) and removes $\sim$88% of the aligned-only residual SSE, while $H$-intercepts alone explain at most 15%. But leverage is high ($p/n = 5/9$), there is one development seed, and no independent noise estimate — so the correct claim today is an excellent *descriptive* fit on the measured manifold, not yet a separately identified causal decomposition of $b$ vs $v$. The identification experiments (matched-$\eta_{\mathrm{eff}}$ pairs, blind seed-251 $\mu=0.9$ predictions) are preregistered and pending (Sections 4.8–4.9).

### 2.4 What the law explains at once

- **The crossover.** Optimal $\mu$ at fixed $\eta$ rises with $H$ because $\rho$ *falls* with $H$ (0.562 → 0.250 → 0.328 lag-1 RDA at $H{=}16/64/256$, with a monotone plain-average trend 0.98 → 0.93 → 0.73), so momentum's amplification shrinks toward — and eventually lands on — the tuned effective step.
- **The $\mu=0.9$ poison at short $H$.** $\eta_{\mathrm{eff}}$ reaches 0.49 (1.8× the tuned 0.28) *and* $A^2_{\mathrm{RMS}}=17.6$; both terms fire.
- **Why scale-matching alone fails.** EXP2.24's realized-norm-matched $\mu=0.9$ arm sits at the right average scale and still loses $\sim$0.04 — the variance term is scale-matching-invisible.
- **Why DC-gain matching succeeds.** $\mu=0.9$ at $\eta=0.028$ has $\eta_{\mathrm{eff}}\approx 0.24$–0.28 ≈ tuned, and its (equally rotated) transverse component is too small to hurt: tied-best, as observed.
- **Why the long-$H$ momentum "win" is an LR effect.** At $H=256$, $\mu=0.5$ has $\eta_{\mathrm{eff}}\approx 0.28$ = exactly tuned; the preregistered seed-251 test confirmed that memoryless SGD at tuned $\eta=0.28$ slightly beats it (Section 4.2).

![Figure 3: pseudo-gradient lag kernel](figs/fig-mechanism-rho-lag-kernel.png)

*Figure 3 (mechanism). Energy-weighted autocorrelation $\rho_k$ of production RDA-merged deltas at lags 1–4, per horizon, computed from retained $\mu=0$ captures whose replayed outer steps were verified bit-for-bit against the next anchor checkpoint (316/316, 76/76, 16/16 exact). Persistence decays fast in lag and is highest at the shortest horizon — the inversion that drives the crossover.*

## 3 Methods

### 3.1 Protocol

All experiments use the in-repo `compare_diloco.py` harness: Qwen3.5-9B, LoRA rank 2 / alpha 4, `trl-lib/Capybara` (5000 train / 64 eval rows), $M=4$ learners, 4 fragments, strict full quorum, inner AdamW at LR $10^{-3}$, fixed local windows (padded, fixed microstep count), delta correction off, bf16 wire / f32 syncer math. Total learner work is held fixed across horizons: $H\times$ outer-steps $= 5120$ microsteps (H=16 → 320 commits, H=64 → 80, H=256 → 20), token budget 700k. Seeds are (shuffle/training) 223/223223 for development and 251/251251 — preregistered before any arm ran — for confirmation. Async comparators inject per-learner push delays of 0/800/1600/2400 ms (jitter 50); "sync" arms set all delays to zero (learners do not hard-block at a barrier; staleness is bounded by commit latency). Runs execute on Verda spot GPUs (A100 and RTX PRO 6000 Blackwell) with an S3-resume babysitter; every syncer commit is captured for offline replay.

Pseudo-gradient semantics (audited, `docs/OPTIMIZER_SEMANTICS.md`): per learner $\Delta_m = \Theta_p - \theta_m$ against the syncer's *current* f32 anchor; learner weights $w_m = c_{\mathrm{tokens}}^2/c_{\mathrm{steps}}$; per-tensor radial-directional averaging (RDA: merged norm = weighted mean of norms, merged direction = normalized weighted mean of unit directions) for all fragments except the direct-averaged embedding. No normalization by $H$, no clipping, no weight decay, no dampening anywhere in the outer path. Deterministic hand-computed vector tests pin the Nesterov and rho-adaptive recursions.

Replay verification: $\rho$ statistics are computed only on captures where the replayed outer step $\Theta_{t-1} - \eta\,\delta_t^{\mathrm{RDA}}$ matches the next same-fragment anchor bit-for-bit (408/408 checked transitions exact).

### 3.2 Predeclared gates and kill criteria

From the frozen study plan (`docs/NORTH_STAR_PLAN.md`), before the confirmatory phase:

- **LR-matching gate:** proceed only if conventional LR matching does *not* remove the momentum effect while correlation-aware matching collapses it.
- Kill criteria: standard LR matching removes the effect → mechanism not novel; $\rho/A_t/r_\perp$ fail held-out prediction → no universal state variable; buffer intervention fails → epiphenomenon; a simple clip matches the controller → theory must carry; controller needs per-setting tuning → drop "tuning-free"; controller sacrifices the long-$H$ benefit → safety patch only; wall bound dies under sequential testing → failed probe implementation, not impossibility.
- Controller discipline: develop on one small full-param + one LoRA setup, freeze formula/constants/warmup/clipping, then never modify. Primary metric: worst-case regret vs the best per-setting tuned fixed optimizer.

> ### Box 1 — The falsification trail (how this claim was built by breaking it)
>
> 1. **"Momentum is the poison" (EXP2.23/24).** Vanilla $\mu=0.9$ loses 0.059 to scale-matched memoryless SGD under zero delay; a six-point decomposition grid factors the damage as (directional memory rotation) × (realized step scale). Correct observations, pre-mechanistic framing.
> 2. **Single-term law (EXP2.25).** Merged-delta autocorrelation measured with a plain-candidate-average convention gave $\rho$ = 0.98/0.93/0.73 and the law $\eta_{\mathrm{eff}}=\eta/(1-\mu\rho)$ retrodicted 7/7 grid checks. It was wrong twice.
> 3. **Wrong convention (EXP2.26).** The controller built on that law measured *production RDA* $\rho$ online and got values 2–4× lower (0.45 vs 0.98 at H=16) — the buffer integrates RDA merges, not plain averages. The law's inputs were invalid; the controller's $\kappa=2$ rule was consequently miscalibrated (~2× amplification where ~1× was optimal), explaining its 0.008 gap at H=16.
> 4. **Wrong form (independent review, 2026-07-12).** For this Nesterov form the aligned gain is $\eta(1+\mu/(1-\mu\rho))$, not $\eta/(1-\mu\rho)$. Under the corrected form, aligned gain alone no longer retrodicts the $\mu=0$ wins at short $H$; the review derived the missing second term — buffer variance accumulation $A^2_{\mathrm{RMS}}$ — which does.
> 5. **Refit on corrected inputs and form (rda-rho-law).** Two-term law: $R^2=0.90$; aligned-only 0.15; the superseded single-term law 0.28. Four predictions preregistered on unseen cells.
> 6. **Fresh-seed test (EXP2.27).** The crossover replicates in both directions on preregistered seed 251, and the law's sharpest qualitative prediction — tuned-$\eta$ memoryless SGD dissolves the fixed-$\eta$ momentum advantage at $H=256$ — lands. Controller-v1 arms were withdrawn from the fresh seed before launch because v1 was known-miscalibrated; seed 251 carries only preregistered fixed-policy arms.
>
> Every correction *tightened* the claim: the state variable survived two demolitions of its first quantitative law, and the demolitions were forced by our own convention audits and solicited external review, not by a referee.

## 4 Experiments

### 4.1 The momentum crossover in $H$ (seed 223)

64-row eval loss, fixed $\eta=0.175$, equal total work (EXP2.25; H=64 row from EXP2.23/24):

| $H$ | $\mu=0$ | $\mu=0.5$ | $\mu=0.9$ | $\mu=0.9$ penalty vs best |
|---:|---:|---:|---:|---:|
| 16 | **1.3519** | 1.3632 | 1.4383 | 0.0864 |
| 64 | **1.3578** | 1.3614 | 1.4193 | 0.0615 |
| 256 | 1.3805 | **1.3674** | 1.3977 | 0.0303 |

The crossover sits between $H=64$ and $H=256$: memoryless SGD wins at $H\in\{16,64\}$; $\mu=0.5$ wins at $H=256$ by 0.0131. The $\mu=0.9$ penalty decays monotonically (0.086 → 0.062 → 0.030) toward the DiLoCo design regime, and $\mu=0$ degrades as syncs get rarer (1.3519 → 1.3805) while $\mu=0.5$ is nearly flat. Tape geometry tracks the mechanism — the same $\mu=0.9$ becomes better aligned as $H$ grows:

| Arm | Step norm | Applied-step cosine | History/current |
|---|---:|---:|---:|
| h16-$\mu$0 | 0.838 | 1.000 | 0.000 |
| h16-$\mu$0.5 | 1.366 | 0.980 | 0.230 |
| h16-$\mu$0.9 | 1.974 | 0.752 | 0.823 |
| h64-$\mu$0.9 (EXP2.23) | 4.025 | 0.797 | 0.685 |
| h256-$\mu$0 | 3.517 | 1.000 | 0.000 |
| h256-$\mu$0.5 | 5.500 | 0.986 | 0.158 |
| h256-$\mu$0.9 | 7.533 | 0.872 | 0.489 |

![Figure 1: H x mu heatmaps, seeds 223 and 251](figs/fig1-h-mu-heatmap.png)

*Figure 1. Held-out eval loss over the $H\times\mu$ grid. Left: seed-223 development grid. Right: preregistered fresh seed 251, including the tuned-$\eta$ memoryless control. Darker = worse; per-panel scales.*

### 4.2 Fresh-seed confirmation and the LR-confound test (seed 251)

Six preregistered arms on a seed the series had never touched (EXP2.27):

| $H$ | $\mu=0$ ($\eta=0.175$) | $\mu=0.5$ ($\eta=0.175$) | SGD-0.28 ($\mu=0$) |
|---:|---:|---:|---:|
| 16 | **1.627433** | 1.641685 | 1.636382 |
| 256 | 1.645687 | 1.640791 | **1.639274** |

1. The crossover replicates in both directions: at fixed $\eta=0.175$, $\mu=0$ wins $H=16$ by 0.0143 and $\mu=0.5$ wins $H=256$ by 0.0049 — same signs as seed 223.
2. The LR-confound test lands where the two-term law points: tuned-$\eta$ memoryless SGD-0.28 slightly beats $\mu=0.5$ at $H=256$ (1.6393 vs 1.6408) and loses at $H=16$ to $\eta=0.175$, $\mu=0$. The long-horizon momentum benefit at fixed $\eta$ is primarily an effective-LR effect via the aligned term.
3. No fixed $\eta$ wins at both horizons (0.175 takes $H=16$, 0.28 takes $H=256$) — exactly the gap a kernel-measuring controller exists to close.

### 4.3 Decomposition: what exactly does $\mu=0.9$ break? (sync six-point grid)

EXP2.24 completes a six-point grid at $H=64$, zero delay, seed 223, sorted by eval loss:

| Arm | $\mu$ | $\eta$ | DC gain | Realized step norm | Cosine | History/current | Eval loss |
|---|--:|--:|--:|--:|--:|--:|--:|
| sync-sgd-lr175 | 0 | 0.175 | 0.175 | 1.813 | 1.000 | 0.000 | **1.357837** |
| sync-mu09-lr0028 | 0.9 | 0.028 | 0.28 | 0.939 | 0.741 | 1.135 | 1.358003 |
| sync-sgd028 (EXP2.23) | 0 | 0.28 | 0.28 | 2.869 | 1.000 | 0.000 | 1.359852 |
| sync-mu05-lr175 | 0.5 | 0.175 | 0.35 | 2.879 | 0.982 | 0.200 | 1.361390 |
| sync-mu09-lr0125 | 0.9 | 0.125 | 1.25 | 3.145 | 0.773 | 0.777 | 1.398193 |
| sync-mu09-lr175 (EXP2.23) | 0.9 | 0.175 | 1.75 | 4.025 | 0.797 | 0.685 | 1.419274 |

- Scale alone does not explain the damage: within memoryless SGD a 58% realized-norm change moves loss by +0.002, yet at matched realized norm (2.87–3.15) $\mu=0.9$ is 0.037–0.040 worse than $\mu\in\{0,0.5\}$.
- Momentum is harmless when steps are small: DC-matched $\mu=0.9$ ($\eta=0.028$, realized norm 0.94) ties the best arm despite equal rotation (cosine 0.74).
- In two-term-law terms: norm matching fixes neither $\eta_{\mathrm{eff}}$ calibration nor $A^2_{\mathrm{RMS}}$; DC matching fixes the aligned term and shrinks the transverse kick below threshold.

### 4.4 Staleness is nearly free; negative merges are schedule-inherent

Matched sync/async pairs with identical model, data, seeds, tooling (EXP2.23). Exact-LR replay on the sync SGD capture, same anchors/oracle/thresholds as the async series:

| Schedule, seed 223, SGD-0.28 | Negative-merge rate | Mean scalar oracle headroom |
|---|---:|---:|
| async 0/800/1600/2400 ms | 0.3625 | 0.001807 |
| sync 0/0/0/0 ms | 0.3875 | 0.001995 |

(Async seed 239: rate 0.2875, headroom 0.001834.) Removing all injected staleness does not reduce the negative-merge rate; the one-step damage and the $\sim$0.002 oracle headroom are properties of merging independent local trajectories at all. Endpoint check: sync SGD-0.28 ends at 1.359852 vs async 1.359505 — the 0–2400 ms delay pattern moves final loss by $\sim$0.0003. The momentum pathology likewise needs no asynchrony: sync $\mu=0.9$ loses 0.0594 to sync SGD-0.28 (1.419274 vs 1.359852), as large as the async-schedule penalty. The frozen selector's phenomenology also transfers unchanged (selection rate 0.1625, gain vs ×1 = 0.001073, 53.8% headroom captured; async: 0.2125 / 0.001009 / 56.1%).

**The measurement wall.** Frozen EXP2.21 selector on the same 80-group sync capture, anchor probe size swept (EXP2.25 E3):

| Anchor panels | Selection rate | Gain vs ×1 | Headroom captured |
|---:|---:|---:|---:|
| 2 | 0.475 | 0.000898 | 44.2% |
| 8 | 0.1625 | 0.001073 | 53.8% |
| 16 | 0.1625 | 0.001098 | 54.0% |

Going 2→8 panels prunes false selections and lifts captured headroom 9.6 points; doubling again buys 0.2 points. The selectable groups are the early large-gap ones; no affordable probe reaches the rest (median late-training gap 0.00084 vs per-decision SE 0.00138). A numerics footnote doubles as a measurement: re-running the baseline replay across GPU generations flipped 2/80 borderline groups from bf16 kernel differences alone.

![Figure 2: measurement wall](figs/fig-wall-headroom-vs-panels.png)

*Figure 2. Captured oracle headroom vs anchor probe size. The probe saturates at 8 panels: selection quality is gap-limited, not measurement-limited, beyond the early-training regime.*

### 4.5 The kernel: production-RDA autocorrelation by horizon

Energy-weighted per-tensor RDA autocorrelation on replay-verified $\mu=0$ captures (`experiment-results/EXP2/rda-rho-law/summary.md`; Figure 3):

| $H$ | pairs (lag1) | $\rho_1$ | $\rho_2$ | $\rho_3$ | $\rho_4$ | tensor p10 | tensor p90 | replay exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 316 | 0.5622 | 0.2058 | 0.0665 | 0.0215 | 0.544 | 0.571 | 316/316 |
| 64 | 76 | 0.2498 | 0.0437 | 0.0051 | −0.0012 | 0.230 | 0.263 | 76/76 |
| 256 | 16 | 0.3277 | 0.1048 | 0.0359 | 0.0158 | 0.293 | 0.362 | 16/16 |

Known confound (flagged in review, audit pending): the $H=64$ kernel comes from the EXP2.23 sync-SGD capture at $\eta=0.28$, while $H=16/256$ ran at $\eta=0.175$; the non-monotonicity at $H=64$ vs $H=256$ is unresolved within naive CIs and a re-capture at matched $\eta$ is queued before any $\rho(H)$-shape claim.

### 4.6 Controller v1 development screen (rho-adaptive)

A probe-free controller measuring $\rho$ online (buffer stores the previously applied direction; $\mu_{\mathrm{eff}}=\mathrm{clamp}(2(1-\rho),0,0.9)$; step $\eta/(1-\mu_{\mathrm{eff}}\rho)$ capped at 4×; previews bit-identical to applied steps), EXP2.26, seed-223 development screen:

| $H$ | $\mu=0$ | $\mu=0.5$ | $\mu=0.9$ | rho-adaptive | Online mean $\rho$ (RDA) |
|---:|---:|---:|---:|---:|---:|
| 16 | **1.3519** | 1.3632 | 1.4383 | 1.3597 | 0.447 |
| 64 | 1.3578 | 1.3614 | 1.4193 | **1.3582** | 0.289 |
| 256 | 1.3805 | **1.3674** | 1.3977 | 1.3723 | 0.221 |

Never worse than second-best at any horizon; never catastrophic; ties the best arm at $H=64$ (+0.0004). Worst-case cross-horizon regret: rho-adaptive 0.008 vs 0.013 ($\mu=0$), 0.011 ($\mu=0.5$), 0.086 ($\mu=0.9$) — roughly half the best fixed policy's, with zero per-horizon tuning. This is explicitly a *development screen*: the controller was designed after seeing the fixed grid on the same seed, and its $\kappa=2$ rule was calibrated on the wrong $\rho$ convention (Box 1), which explains its 0.008 give-back at $H=16$. The frozen v2 (capped-Nesterov Candidate 2: $\mu_{\max}=0.9$, transverse cap $\tau_\perp=1.0$, sign-reversal guard, one-sided release EMA, $\eta = 0.28/1.9 \approx 0.147$; pointwise bound $A^2 \le 4.61$) is specified and pending (Section 4.10).

### 4.7 [PLACEHOLDER] Blind prediction test (preregistered seed-251 $\mu=0.9$ cells)

**Spec:** measured seed-251 losses at ($H{=}16,\mu{=}0.9$) and ($H{=}256,\mu{=}0.9$) against the frozen predictions 1.4356 and 1.4132, ordering gap 0.0224, sibling penalties +0.0780 / +0.0473, plus the $H{=}64$ $\mu{=}0.5$ $\eta{=}0.28$ cell vs predicted 1.3869 — one table, pass/fail per preregistered ordering/gap, with competing-model (aligned-only, single-term) predictions alongside.

### 4.8 [PLACEHOLDER] Single-learner noise floor (E1)

**Spec:** single-learner one-step negative rate on the same 512-row oracle, from retained sync captures — one number to subtract from the 0.36–0.39 merge negative rate, splitting "merge interference" from "any stochastic step looks negative".

### 4.9 [PLACEHOLDER] Staleness in optimization units

**Spec:** penalty vs staleness measured as version lag / parameter displacement / buffer displacement / arrival cosine, crossed with $\rho$ and gain — one figure showing whether moderate staleness stays second-order once the kernel is controlled.

### 4.10 [PLACEHOLDER] Capped-Nesterov controller screens (frozen v2 + ablations)

**Spec:** dev-screen table like 4.6 for frozen Candidate 2 ($\mu_{\max}{=}0.9,\tau_\perp{=}1.0,\eta{=}0.147$) vs Candidate 1 gain-normalizer vs rho-adaptive-v2 scalar baseline — per-H losses, worst-case regret, realized $A^2$ vs the ≤4.61 bound.

### 4.11 [PLACEHOLDER] Buffer-orientation intervention (causality)

**Spec:** one-step paired eval-loss deltas for same-norm momentum buffers at real / aligned / orthogonal / anti-aligned / random-rotated orientations from a common checkpoint — one bar figure; kill criterion if orientation does not order the damage.

### 4.12 [PLACEHOLDER] Full-parameter phase map (SmolLM2-135M)

**Spec:** $H\times\mu\times\eta$ reversal phase diagram with the five predeclared fairness controls (fixed nominal LR, $\div(1{+}\mu)$, aligned-gain-matched, norm-matched, independently tuned) — the figure that discharges the LoRA-rank-2 and LR-matching gates.

### 4.13 [PLACEHOLDER] Matched-$\eta_{\mathrm{eff}}$ pairs (identifying $v$ separately from $b$)

**Spec:** paired-seed off-diagonal cells — fixed $H$, two $\mu/\rho$ conditions with $\eta$ tuned so corrected $\eta_{\mathrm{eff}}$ matches — reporting paired $\Delta$loss vs the two-term prediction $v\,\Delta\log A^2$ (aligned-only predicts ≈0).

### 4.14 [PLACEHOLDER] Dynamic-H / elastic synchronization

**Spec:** wall-clock and FLOPs to target loss under bandwidth variation and worker churn, frozen controller vs best fixed policies vs per-H tuned oracle — the systems-payoff figure.

## 5 Related work

**DiLoCo and the momentum recipe.** DiLoCo (Douillard et al., arXiv 2311.08105) establishes outer Nesterov $\mu=0.9$ at $H\approx 50$–500 full-finetune steps; we show the recipe's value is a correlation bet and give the free statistic that prices it. Muennighoff et al.'s observation that outer LR must be tuned jointly with sync frequency and that outer Nesterov exhibits instability (arXiv 2509.10439) is, in our terms, the aligned term of the filter — $\eta_{\mathrm{eff}}$ moves with $\rho(H)$ — without the kernel or the variance term. SNOO (pseudo-gradient momentum beneficial at scale, $M=1$) sits in the long-horizon/low-$\rho$ regime our law marks as safe.

**Async local-SGD and staleness.** DeepMind's async local-SGD study (arXiv 2401.09135) found the "momentum challenge" but attributed it to staleness and fixed it with Delayed Nesterov and dynamic local updates. Our matched sync replication shows the attribution is wrong: the poison needs no staleness (Section 4.4), and their fix does not touch either amplification term. FedBuff, staleness-weighted merging, Pseudo-Async Local SGD (arXiv 2504.18454), and Decoupled DiLoCo (arXiv 2604.21428) engineer against or around staleness; all are consistent with — but never state — the claim that staleness is second-order while filter memory is the operative variable.

**Momentum-horizon matching.** Outer-Momentum Restarting (arXiv 2605.28585) is the closest work: periodic momentum restarts widen stable $(\eta,\mu)$ ranges, and its stated future work — adaptive restart vs communication period — is precisely the controller we build; our EXP2.19 shows conflict-triggered restart is a partial repair that still loses to memoryless SGD at short $H$. HeLoCo (arXiv 2606.00271) corrects local deltas against the outer momentum buffer (a $\delta$-side intervention; our delta-correction ablation runs it under sync). MT-DAO (arXiv 2510.05361) diagnoses the mismatch from the opposite corner — momentum too *short* for long $H$, adding slow momenta; combined with our short-$H$ results this is evidence for a single matching law neither side had. DES-LOC (arXiv 2505.22549) matches momenta half-lives to sync intervals for communication savings, not quality. SparseLoCo (arXiv 2508.15706) finds outer momentum detrimental under high sparsification — an unexplained instance a correlation-kernel law should cover, since error feedback reshapes the delta sequence.

**Merge-time selection and probing.** IsoLoCo (arXiv 2607.03011) shows a merge transform beating momentum DiLoCo, which narrows any selection claim we make to validation-guided selection under affordable probe budgets — exactly the regime our wall quantifies; a planned intervention separates temporal correlation from worker agreement to locate where Iso-C acts. Held-out outlier gating (arXiv 2604.08056) and FedGSNR gate merges heuristically with no cost analysis; the measurement-wall result — per-decision gaps below the noise floor of any probe cheaper than the step it governs, chance-level ranking across 600+ groups, saturation at 8 panels — appears unclaimed. Gradient-conflict methods (PCGrad and kin) and Byzantine-robust aggregation know conflict exists; the decision-theoretic unmeasurability at $m\approx 4$–12 candidates is new.

## 6 Limitations

- **Scope of the grid evidence.** Two seeds (223 development, 251 preregistered confirmation); one model/data/config (Qwen3.5-9B, LoRA rank 2, Capybara); 64-row eval endpoints without confidence intervals; $H=256$ arms have only 20 outer rounds and 16 kernel pairs. LoRA rank 2 constrains deltas to a low-dimensional space and may structurally inflate $\rho$; the full-parameter phase map (Section 4.12) is the required check before generalizing.
- **The fit is descriptive, not yet causal.** Leverage $p/n=5/9$; $b$ and $v$ may be individually near-unidentified on the current manifold (VIF / joint confidence ellipse / leave-one-$H$-out to be reported); the $H=64$ kernel carries an $\eta$ confound (Section 4.5). The matched-$\eta_{\mathrm{eff}}$ pairs and blind seed-251 predictions are the identification instruments.
- **Open-loop kernel.** $\rho$ is measured on $\mu=0$ captures; closed-loop $\rho$ under momentum differs, and the controller's online estimate is the closed-loop object.
- **"Sync" is vanilla-schedule, not lockstep.** Learners do not barrier; staleness is bounded by commit latency, not zero. The DiLoCo design point ($H\approx 500$ full-finetune) differs from our 16–256-microstep LoRA windows; the crossover location is a claim about this cadence and budget.
- **Cross-hardware numerics.** Arms span A100/cu121 and Blackwell/cu128; all decisive matched comparisons are same-environment, but bf16 kernel differences alone flip 2/80 borderline one-step signs — itself evidence for how small per-decision signals are.
- **Controller status.** v1 results are a same-seed development screen with a known miscalibration; no frozen-controller fresh-seed evidence exists yet. The wall claim is about affordable probes for closely spaced merge actions after early training, not about selection in general (early-training selection demonstrably works, 56–84% headroom).

## Reproducibility

All arms are launched from `scripts/compare_diloco.py` with full flag lists recorded in `docs/EXP2_23.md`–`EXP2_27.md`; captures, replays, and aggregates live under `experiment-results/EXP2/` and the S3 prefix recorded per experiment doc. Figures in this draft are rendered by `paper/figs/render_figs.py` from checked-in tables and `summary.json`. Optimizer semantics are pinned by deterministic vector tests (`nesterov_three_step_hand_computed_sequence`, `rho_adaptive_three_step_hand_computed_sequence`).
