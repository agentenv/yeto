# Outer-muP v2 confirmatory preregistration

**Program ID:** `outer-mup-v2-confirm`

**Status:** `PREREGISTERED`, prospective for all scientific outcomes described below.

**Registered:** 2026-07-25.

**Machine-readable companion:** `experiment-specs/outer-mup-v2-confirm-prereg.json`.

**Pilot:** program `outer-mup-2day` (v1), contract `outer-mup-2day-prereg.{json,md}`
(json sha256 `2094d59a505956b329eefce65bce0e14c05b4d0fb97632f5bcf9e1a619b0e2f5`,
md sha256 `dca46f83dc3a58764361b4bd070ed38242987efa7ecee269d795393230ecd012`), closed as
`STOP_G1_NOT_EVALUABLE_AUDIT_PAPER` — see `experiment-specs/outer-mup-v1-closure.md`.

This is the confirmatory re-registration of the outer-muP program with corrected
instruments. **Every change from v1 is pilot-informed and disclosed here before any v2
outcome exists. v1 is treated as pilot/calibration input only; no v1 result is reused
as v2 evidence. No v1 result, evidence file, or sealed record may be modified.**
v2 uses fresh seeds throughout.

The pilot established (v1 closure record):

1. An H-dependent breakdown of the `(1-mu)` equivalence with seed CIs at H16
   (D_obs ≈ 1.03) and H512 (D_obs ≈ 2.51, CI excluding 1).
2. **Instrument defect A:** mu=.9 eta ladders at intermediate H must be centered on an
   H-dependent prior; the v1 first-order-prior windows left H64/H256 mu.9 unbracketed.
3. **Instrument defect B:** the raw lag-1 projected pseudo-gradient cosine is
   noise-compressed and is not a valid estimator of effective persistence.
4. **Analyzer defect:** the v1 analyzer only computed rho at an H when the mu=.9 fit
   was also valid, coupling rho availability to mu=.9 bracketing.

The v2 registered corrections are: recentered/widened mu=.9 ladders (registered as
explicit numeric grids below), a noise-corrected effective-persistence estimator
`rho_eff` built from cross-worker cosines, an analyzer whose rho computation is
unconditional on mu=.9 curve validity, and a gate in which quantitative D_pred/D_obs
ratios are descriptive only. There are no mu=.5 curves in v2.

Except where this document registers a change, every rail, vocabulary, evidence rule,
estimator, bootstrap, retry rule, and stop rule is carried over verbatim from the v1
contract; the closed vocabularies, common numerical/work contract, evidence contract,
and downstream stage structure are restated in the JSON companion, which is
authoritative.

## 1. Frozen analyzer

The registered E1' analysis is implemented in `scripts/analyze_e1v2.py`, frozen before
launch with raw-file sha256

```text
41f7dd17421edfaa2d86746a4cc5eb6b273931502eefdb9b7287e1bd79b721dd
```

recorded here and in the JSON companion (`frozen_analyzer`). Identical bytes are
distributed to both node control planes. Outcome-aware edits are forbidden; any
pre-outcome amendment follows the v1 amendment rules (new commit + raw sha256,
superseded bytes retained).

## 2. Execution and registration commits

The science executes on `h200-n1`/`h200-n2` at repository commit
`a886a3996905913d37ec56cc14914878f636283d` (fast syncer build; an ancestor of this
registration commit). The launch manifest binds `a886a39` as `source.git_commit`
together with the raw sha256 of both contract files and this registration commit.
No GCP/AWS resources are used.

## 3. E1' — 135M confirmatory discriminator grid

SmolLM2-135M, `M=4`, `H in {16,64,256,512}`, full eta curves for `mu in {0,.9}`.

**Fresh primary seeds:** `{211,223,227,229,233}`.
**Fresh contested top-up seeds:** `{239,241,251}` run every eta at the three contested
curves `(H64,mu=.9)`, `(H256,mu=.9)`, `(H512,mu=.9)` — the curves the pilot left
unbracketed or contested. Top-ups are a secondary robustness report only: they cannot
enter the five-seed G1' confidence intervals or the Spearman input, and cannot change
G1'.

Cell count: `8*4*5 + 3*4*3 = 196` training runs, every one with telemetry.

### 3.1 Registered four-point ladders

Explicit numeric grids are authoritative (JSON `stages.E1V2.eta_grid_by_curve`).
Provenance: "v1 reused" = the exact v1 E1 ladder; others are pilot-informed
replacements registered before any v2 outcome.

| H | mu | Four registered eta values | Ladder | Provenance |
|---:|---:|---|---|---|
| 16 | 0 | `.01300695282034226, .018394609083675004, .02601390564068452, .03678921816735001` | sqrt2 | v1 reused |
| 16 | .9 | `.0016258691025427825, .0022993261354593755, .003251738205085565, .004598652270918751` | sqrt2 | v1 reused |
| 64 | 0 | `.01300695282034226, .018394609083675004, .02601390564068452, .03678921816735001` | sqrt2 | v1 reused |
| 64 | .9 | `.0008129346, .0016258691, .0032517382, .0065034764` | **2x (log2 step 1.0)** | pilot defect A: widened window |
| 256 | 0 | `.0183946766, .02601390564068452, .03678921816735001, .05202781128136904` | sqrt2 | v1 ladder shifted one rung down (top rung dropped, bottom rung `.0183946766` added) |
| 256 | .9 | `.0032517382, .0045987253, .0065034764, .0091974507` | sqrt2 | pilot defect A: recentered up |
| 512 | 0 | `.02601390564068452, .03678921816735001, .05202781128136904, .07357843633470001` | sqrt2 | v1 reused |
| 512 | .9 | `.00650347641017113, .009197304541837502, .01300695282034226, .018394609083675004` | sqrt2 | v1 reused |

The `eta_star` fit is the v1 quadratic in `x=log2(eta)` on the four seed-mean losses,
evaluated at the exact registered eta values. **The H64 mu.9 curve's quadratic fit in
log2-eta therefore uses the registered 2x spacing; this is registered here explicitly.**
Interior/UNBRACKETED rules, the seed-curve pairing unit, the 10,000-replicate paired
bootstrap with RNG seed `20260724`, and the no-extension rule are unchanged from v1.

### 3.2 Work

Identical to v1 E1: 1,310,720 tokens per arm, 2,560 inner steps per learner, global
outer commits `{H16:640, H64:160, H256:40, H512:20}`, no terminal partial windows,
strict quorum, fixed windows, zero injected delay/jitter, `--rho-telemetry` on every
cell.

### 3.3 Registered rho estimator (pilot-selected)

For each H, let `c` be the **fragment-balanced cross-worker cosine** from `mu=0`
telemetry at the eta grid point nearest the pooled `eta_star(mu=0)` (lower eta wins the
tie), using exactly the v1 aggregation: mean within fragment, then mean across the four
fragments, then mean across the five primary seeds. The per-row statistic is the exact
pairwise cross-worker mean cosine (`cross_worker.mean_cosine`, 4 workers, all 6 pairs
defined). The registered effective persistence and gate quantities are

```text
rho_eff(H) = 1 - 4c/(1+3c)
A          = 1 + .9/(1 - .9*rho_eff(H))
D_pred(H)  = 10/A
D_obs(H)   = [eta_star(.9,H)/eta_star(0,H)] / .1
```

The raw lag-1 (and lag-2..4) projected cosines are reported as diagnostics only and are
never gated (pilot defect B). **The rho_eff computation must be, and in the frozen
analyzer is, independent of the validity of the mu=.9 curve at the same H (v1 analyzer
defect).** rho_eff is undefined (fail-closed, G1' NOT_EVALUABLE) if the mu=0 optimum is
unbracketed, any required telemetry is missing/invalid, `1+3c<=0`, or `1-.9*rho_eff<=0`.

### 3.4 G1' — gate and stop rule

Using only the five primary seeds:

- **NOT_EVALUABLE preconditions (a):** every required eta optimum (all 8 curves)
  interior; every E1' primary work record valid; `rho_eff` defined at all four H.
  If any fails, G1' is `NOT_EVALUABLE`.
- **PASS requirements (b), (c):**
  1. At H256 the paired five-seed 95% CI for `D_obs` excludes `1.0`.
  2. At H512 the paired five-seed 95% CI for `D_obs` excludes `1.0`.
  3. Spearman(D_pred, D_obs) across the four H values is `>= .8` (ordinary Spearman
     with midranks, n=4 exact test, exactly as v1).
- Quantitative `D_pred/D_obs` ratios are reported descriptively and are **not** gated.

A scientifically evaluable false condition is `FAIL`. `FAIL` and `NOT_EVALUABLE` carry
exactly the v1 STOP semantics: the experimental program stops, no downstream scientific
cell launches, and the fallback is the (v2) audit paper. The eight-seed contested
summaries at H64/H256/H512 are secondary and cannot change G1'.

## 4. Downstream stages

E2 (M-axis), E3 (loss-blind probes, sealed 360M/1.7B predictions, verification, G2 at
75%-within-CI of 48 sealed cells), E4 (mu=.95 boundary + buffer surgery), E5 (two-node
real-interconnect 1.7B demo), and optional E6/E7 carry over **structurally unchanged
from v1** with these registered substitutions (JSON authoritative):

- every rho quantity is `rho_eff` (E3 target scales: `rho_eff = 1 - M*c/(1+(M-1)*c)`
  for `M in {4,8}` from the target-scale mu=0 probe; `M=1` cells, which have no
  cross-worker pairs, take the same-scale M=4 `rho_eff` at that H — a disclosed
  pilot-informed instrument limitation);
- fresh v2 seeds everywhere (`{211,223,227,229,233}`; probe/E5 seed `211`);
- E2 ladders center on sealed E1' pooled `eta_star`; E4 mu=.95 centers use sealed E1'
  `rho_eff`;
- the sealed prediction artifact is `experiment-specs/outer-mup-v2-sealed-predictions.json`.

## 5. Wall-clock ceilings

| Stage | Hard wall ceiling |
|---|---:|
| **E1'** | **6 h** |
| E2 | 8 h |
| E3 loss-blind probes | 6 h |
| E3 prediction seal | 1 h |
| E3 verification | 14 h |
| E4 | 8 h |
| E5 | 6 h |
| E6, if authorized | 6 h |

Only the E1' ceiling changes from v1 (14 h -> 6 h, pilot-informed by observed v1 E1
throughput on the same 16 H200 slots). Measurement and breach rules are v1's verbatim.
The 48-hour program clock starts at the first E1' scientific process.

## 6. Fallback paper

Identical trigger set and reporting duties as v1, plus mandatory disclosure of the v1
pilot record and every pilot-informed change registered here. Negative, null,
reversed, unbracketed, boundary, and failed-transfer outcomes are publishable.
