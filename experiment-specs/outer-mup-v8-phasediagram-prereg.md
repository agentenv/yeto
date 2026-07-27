# Outer-muP v8 tuned-loss phase-diagram preregistration

**Program ID:** `outer-mup-v8-phasediagram`

**Status:** `PREREGISTERED_PENDING_FLEET` — design and analysis are frozen; no v8 GPU process, result root, or launch authority may exist yet.

**Registered:** 2026-07-26 (prospective for every v8 outcome).

**Authoritative machine contract:** `experiment-specs/outer-mup-v8-phasediagram-prereg.json`.

## 1. Question and deliverable

At SmolLM2-135M, `M=4`, and fixed local window `H=512`, v8 asks the question the preceding LR-ratio studies could not answer: after giving every method its own fair LR tuning curve, does outer momentum improve, harm, or practically tie the no-momentum optimum?

The deliverable is two registered phase diagrams—raw production Nesterov and the opt-in bias-corrected Nesterov arm—over `T in {2,5,10,20,40}` and `mu in {.5,.8,.9,.95}`. Each plotted cell reports the fitted tuned-optimum loss difference

```text
Delta(T,mu,arm) = L*_arm(T,mu) - L*_mu0(T)
```

with a paired pointwise interval, a familywise simultaneous interval across all 40 comparisons, and the frozen phase label below. This is a best-loss map, not the earlier `D` map of whether a conventional LR rule understeps or oversteps.

## 2. Design

- `H=512` and `S=512T`, hence `S={1024,2560,5120,10240,20480}`.
- Five new paired scientific seeds `{801,809,811,821,823}`; `training_seed=int(str(seed)+str(seed))`.
- One common correction-OFF `mu=0` baseline curve per T, four raw curves per T, and four corrected curves per T.
- Four LR values and five seeds per curve: `45 curves * 4 etas * 5 seeds = 900 fresh cells`.
- Every LR ladder is symmetric in `log2(eta)` at offsets `{-1.5,-.5,+.5,+1.5}` bits. Adjacent rungs differ by 2x; endpoints span 8x. No outcome-aware extension or point removal is allowed.
- Primary outcome is finite NLL/token on the same fixed 1,024-row development set used by v3/v6.

### Registered centers

The table gives centers; each exact four-number ladder is enumerated in the JSON contract. `r50...r95` are raw centers and `c50...c95` corrected centers.

| T | S | mu0 | r50 | r80 | r90 | r95 | c50 | c80 | c90 | c95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1024 | 0.0710513974 | 0.0453520189 | 0.0347601849 | 0.0319974245 | 0.0307372398 | 0.0346790481 | 0.0136723096 | 0.00680325571 | 0.0033934328 |
| 5 | 2560 | 0.0470177128 | 0.0258988194 | 0.0145095634 | 0.0116111032 | 0.0103522183 | 0.0217378111 | 0.00829595496 | 0.0040835065 | 0.00202582388 |
| 10 | 5120 | 0.0344047075 | 0.017765905 | 0.00791978897 | 0.00530873561 | 0.00423745583 | 0.0145324373 | 0.00525350079 | 0.00253962108 | 0.00124857313 |
| 20 | 10240 | 0.025175276 | 0.0117730349 | 0.00456603096 | 0.00250610911 | 0.00168098405 | 0.00887620629 | 0.00287910341 | 0.00134240604 | 0.000648159985 |
| 40 | 20480 | 0.0506602231 | 0.0194488685 | 0.00663979786 | 0.00319114271 | 0.00174652236 | 0.0124447266 | 0.00324978258 | 0.00140959374 | 0.000656445675 |

### Center provenance and the T=40 hedge

The `mu=0` anchor is the already-registered v6 absolute-rate surface fitted from the v3 135M anchors, fixed-T disambiguation, and v4 1.7B center pilot. Momentum transfer uses the frozen Theory Lane C surface. For raw Nesterov it retains the exact code-true transient `1/(1-mu^(T+1))` and fits only its measured residual; corrected transfer uses the frozen measured drift surface. In both cases `eta_center=eta0_center*(1-mu)*D_arm`. At `mu=0` both formulas return `D=1` exactly.

The sole prospective exception is disclosed rather than hidden: v3 T=40 `mu=0` was still improving at its highest registered eta `0.03582218728980824`, while the absolute-rate surface predicts `0.01842173837783116`. v8 therefore applies the pre-outcome rule `max(surface, sqrt(2)*v3_high_edge)` and centers the T=40 baseline at `0.05066022309911592`. The wide symmetric ladder then covers both the surface prediction and the v3 boundary evidence.

## 3. Exact-reuse audit

No prior cell is reused. Exact identity includes the scientific/training seed, shuffled input bytes, eta, all protocol flags, source commit, and optimizer semantics—not merely `(T,S,H,mu)`.

- v3 has all five structural `mu=0` coordinates, but its seeds are `{301,311,313,317,331}` and its shuffled training inputs/source commit differ.
- v6 has structural `mu=0` matches only at T=5/10/20, but its seeds are `{601,607,613,617,619}`; T=2 and T=40 are absent, and its input bytes/source commit differ everywhere.
- Therefore the registered reuse count is `0` and the fresh count is `900`. v3/v4/v6 data inform centers, noise, and runtime only; they never enter the v8 outcome estimator.

The single v8 `mu=0` curve at each T is legitimately shared within v8: the production regression proves correction ON at `mu=0` bit-identical to correction OFF, so duplicating it would be an A/A expenditure rather than an independent control.

## 4. Frozen analysis and phase labels

For each curve, fit `loss=a*x^2+b*x+c` in `x=log2(eta)` to the four five-seed means. A fit is usable only when `a>0` and its vertex is strictly inside the exact ladder (1e-12-bit boundary tolerance). Then `eta*=2^(-b/(2a))` and `L*=c-b^2/(4a)`. Missing/nonfinite work, nonpositive curvature, or a boundary/outside vertex is `UNBRACKETED`; no extrapolated minimum is accepted.

The frozen analyzer performs 10,000 paired training-seed bootstrap draws with RNG seed `20260728`. One five-index draw is shared across all 45 curves. The complete diagram requires at least 9,500 draws in which all 45 refits remain interior. It reports ordinary paired 95% intervals and a primary simultaneous 95% interval whose common radius is the 95th percentile of the valid-draw maximum absolute deviation across all 40 Delta estimates.

With the practical margin `epsilon=0.01` loss, the primary familywise label is:

- `HELPS` if the simultaneous upper endpoint is below `-0.01`.
- `HURTS` if the simultaneous lower endpoint is above `+0.01`.
- `NEUTRAL` if the entire simultaneous interval lies inside `[-0.01,+0.01]`.
- `UNCERTAIN` if the evidence is evaluable but supports none of those three statements.
- `NOT_EVALUABLE` for incomplete/invalid evidence, any unbracketed point curve, or fewer than 9,500 valid complete refits.

`UNCERTAIN` is deliberately retained: forcing every noisy coordinate into helps/hurts/neutral would turn absence of resolution into a scientific sign claim. The requested diagram still displays all cells, with uncertainty visibly distinct.

Frozen analyzer: `scripts/analyze_v8.py`, SHA-256 `23f0bf723d911bae4fcccbc6f5c8e759bb78a482ef1bd4173915a78921159f44`.

## 5. Mandatory gate-feasibility simulation

The pre-outcome CPU simulation is frozen at `experiment-specs/outer-mup-v8-phasediagram-gatesim.json` (SHA-256 `54070c480c1b5e373adba4c6cea5a58321de276cc853631886695c74d46c9e70`). It transports the sealed v3 five-seed per-eta standard deviations by T/arm/rung, transports v3 quadratic curvature, generates independent Gaussian eta-cell noise, and runs the exact all-45 point gate plus the exact registered 10,000-draw shared bootstrap (compressed to the 126 distinct count vectors; a literal-loop spot check matches 10,000/10,000). The sealed inputs are g3 readout `d4a3cde...44c8` and v3 launch manifest `8fae6137...54fc`.

Primary result: **`P_eval=1.000` (500/500; Wilson 95% `[0.9924,1.0000]`)**. Every primary simulation retained all 45 point vertices and all 10,000 bootstrap refits. A sensitivity that shifts true vertices by the pre-existing v3 fitted-vertex offsets also gives `P_eval=1.000` (500/500). This clears the prospectively fixed `P_eval>=0.8` readiness threshold.

The calculation is a measured-noise transport model, not a guarantee: it assumes the v3 seed-noise scale and local curvature transport to the new mu values. Independence discards favorable cross-curve seed covariance and is conservative for paired differences, but model transport remains the dominant limitation.

## 6. Cost and the registered T=40 rule

Observed successful v3 attempt-1 runtimes (80 cells at each S) give ideal mean cost `215.872 GPU-hours / 16 = 13.492 fleet-hours`. The registered planning estimate uses each S stratum’s measured p90 runtime plus 10% controller/eval/seal overhead: **15.255 fleet-hours**.

The pre-outcome rule is: only if that planning estimate exceeds 18.0 fleet-hours, omit all 160 T=40 momentum cells (both arms, four mus, four etas, five seeds), retain the 20-cell T=40 baseline anchor, and label the eight missing comparisons `NOT_RUN_COST_RULE`. Equality retains them. Because `15.255 < 18`, the rule does not fire: all 900 cells, including the complete T=40 momentum subset, remain registered. The post-authority wall ceiling is 18 hours; breach stops only v8 process groups and never authorizes an extension.

## 7. Inputs, evidence, retry, and fleet gate

New no-wrap bundles use the frozen Day-1 partition and fixed development/audit bytes under `/root/yeto-data/outer-mup-v8`, with deterministic prep/verification scripts for seeds 801/809/811/821/823. CPU-only preparation and verification passed on both nodes with byte-identical combined manifest SHA-256 `5f4235e56be5fc968227e02a6c9a6ebe57277d2736fb2947da14f7bd7f15a20b` and capacity-report SHA-256 `de532769475ef116001748ffedc3824ab4292b93664cb0b2c5b27bdd48294d94`; the minimum is 26,573 complete 128-token blocks per learner versus 20,480 required. Both hashes must be rebound in the eventual launch manifest.

A cell counts only with its exact registered command hash, pushed source commit, exact S steps on all four learners, `4T` strict-quorum tape rows with `c_steps=512`/`c_tokens=65536`, one rho row per commit, a finite 1,024-row endpoint loss, successful exits, and all registered artifact hashes. Retry authority is loss-blind, limited to the full 20-cell paired curve, an enumerated infrastructure cause, and one retry.

The registered execution stack is `build_v8_launch_manifest.py`, `check_v8_gates.py`, `authorize_v8_launch.py`, `authorize_v8_retry.py`, and `run_slot_v8.py`; the authoritative JSON binds every raw-file hash. The manifest builder materializes and hashes all 900 attempt-1 commands and all registered attempt-2 commands, then balances deterministic longest-first queues with the measured v3 p90 runtime weights.

v8 is **not launched by this registration**. The v6 factorial owns the fleet until every v6 slot queue is drained. A future v8 authority requires a fresh loss-blind proof on both nodes, including empty bracketed checks

```bash
pgrep -af '[r]un_slot_v6.py|[c]ompare_diloco.py.*yeto-results-v6|[a]nalyze_v6.py'
```

plus drained v6 slot registries, >=1 TB free per node, identical clean pushed registration/manifest/input hashes, and proof that no v8 result root or process predates authority. Execution then uses 16 deterministic longest-S-first queues (one cell/GPU, four learners packed), 30-second heartbeats, registered watchdogs, and process-group-scoped termination only. Broad/unbracketed pgrep kill patterns are forbidden.

## 8. Claim boundary

v8 may claim only the registered tuned-loss phase labels at these exact 135M/M4/H512/T/mu/protocol coordinates. It may not convert a `D` tuning-ratio boundary into a performance boundary, treat `UNCERTAIN` as neutral, pool raw and corrected arms, extrapolate beyond the grid, or silently reuse any earlier seed. All 900 statuses, all fitted/unbracketed curves, and all 40 simultaneous intervals must be reported.
