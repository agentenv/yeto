# C3-vs-C4 final-checkpoint curvature discriminator protocol

Status: **FROZEN BEFORE DISCRIMINATOR HESSIAN COMPUTATION**

Freeze date: 2026-07-28 (America/Los_Angeles)

This protocol implements the zero-training discriminator specified in
`mech/law-v2/theory/CANDIDATES.md`.  It fixes the inventory rule, checkpoint
selection, Hessian estimator, stability seeds, uncertainty summary, and
closed verdict bands before any discriminator spectrum is computed.  Existing
Round-3C spectrum JSONs were produced for a different kappa estimator and are
not discriminator outcomes; their effective-curvature values are not inputs
to this analysis.

## 1. Estimand and hypotheses

For a matched momentum/control final-checkpoint pair, the estimand is

```text
R = lambda_max(momentum checkpoint) / lambda_max(mu=0 checkpoint),
```

where `lambda_max` is the largest algebraic Ritz value returned by the fixed
held-out-loss Hessian block-Lanczos probe below.

The bands are frozen as follows:

- **C3 parity band:** `0.80 <= R <= 1.25`.  This is symmetric on the log
  scale (`1/1.25` to `1.25`) and allows 25% multiplicative estimator/checkpoint
  variation around the C3 prediction `R = 1`.
- **C4 curvature-ratchet band:** `1.80 <= R <= 3.40`, exactly the range in the
  theory discrimination matrix and task specification.
- Values between or outside those disjoint bands are not rounded or coerced
  into either mechanism.

## 2. Frozen checkpoint inventory and selection rule

The archive inventory is performed before extraction by streaming a complete
table of contents from
`/home/c/h200-evac/evac2-heuristic-shockley.tar.zst` on `dev16`.  Extraction
may begin only after that listing completes successfully.  Only regular
terminal `attempt-1/work/m4/state.ckpt` members needed by the rules below may
be extracted.  The capped `n1`/`n2` trees are metadata evidence only; they do
not contain admissible weights.

For G6 and G8, candidate records are frozen readout records with 135M scale,
`T=20`, attempt 1.  Within each `(campaign,T,H,S,convention,mu)` curve, use
the sampled eta rung having the lowest frozen pooled `seed_mean_losses` value.
The frozen readouts select `e2` except for G6 raw
`(T=20,H=256,S=5120)`, which selects `e1`.  A matched checkpoint pair must
have the same campaign, T, H, S, and training seed; each arm uses its own
independently frozen minimum-loss sampled rung.  One arm is momentum and the
other is `mu0`.  The pair is probeable only when
both runs are recorded as `h200-n1` and both exact terminal checkpoint members
exist in the completed full-fidelity archive listing.  Any pair with either
member resident on `h200-n2` is recorded as lost, not replaced by a different
seed/rung/campaign, and not cross-seed paired.

G6 has the exact target contrast (`mu=0.9` versus `mu=0`).  G8 has no
`mu=0.9` arm: its `mu=0.8` and `mu=0.95` pairs are bracketing diagnostics and
cannot decide the exact-target overall verdict.

The post-hoc Round-3 panel under `/home/c/yeto-mechR3-20260727` is a separate,
fully retained exact `mu=0.9` corrected-versus-`mu0` trajectory pair.  Probe
all matched ages 5, 10, 15, and 20.  Ages 5--15 are trajectory diagnostics;
only final age 20 enters the exact-target overall verdict.

Every exact archive member path, byte size, SHA-256, readout record, and
probeability/loss reason will be recorded in `inventory.csv` and the report.

## 3. Frozen Hessian probe

Use the validated Lane-E checkpoint/data/model adapter from commit `c7650ef`:

```text
mech/lane-e/checkpoint_spectrum_probe.py
SHA-256 857c88c2a227c32f983c5d206c48d43f49792cdda2f797db691df1386e46d8bd
```

The unmodified adapter's `--seed` affects its start block only in the
degenerate case where the checkpoint buffer has no transverse component.  It
therefore cannot, in general, supply the requested independent-seed stability
check.  A discriminator-local extension will change only the second Lanczos
start vector from the normalized transverse buffer to a seed-controlled
Gaussian vector orthogonalized against the held-out gradient.  Checkpoint
parsing, model loading, deterministic panel construction, fp32 HVP function,
float64 two-pass orthogonalization, and NumPy block-Lanczos/Rayleigh code
remain the validated Lane-E implementations.  The extension must be committed
and hashed before its first spectrum run, and its default (non-randomized)
mode must reproduce the original adapter output on a validation checkpoint.

Fixed probe settings:

```text
host                 dev16 (CPU only)
model                /home/c/yeto-mechR3-20260727/inputs/model
held-out data        byte-identical Round-3 eval stream
device               cpu
threads              80 per process
fragments            4
fragment pattern     binpack
sequence length      128
train_on             assistant
loss                  cross_entropy
panels                4 (the first four deterministic packed panels)
batch size            1
max rows              128
block steps           4
Krylov rank           8 (two start vectors times four block steps)
HVP dtype             fp32
orthogonalization     float64, Lane-E NumPy implementation
probe seeds           20260727 and 20260728
```

For each checkpoint and probe seed, `lambda_max` is `max(ritz_values)` with no
mode dropping, absolute-value transform, effective-curvature substitution, or
post-hoc block-depth choice.  A nonfinite/nonpositive top Ritz value, wrong
rank, input/hash mismatch, or incomplete result invalidates that seed.
Momentum and control arms of a pair must use the same held-out data bytes,
probe seed, adapter hash, model bytes, and all estimator arguments.  At most
two 80-thread processes may run concurrently, keeping total requested probe
threads at 160.

## 4. Frozen stability, uncertainty, and verdict rules

For valid seed-specific ratios `R_1` and `R_2`, report

```text
point ratio       exp((log R_1 + log R_2) / 2)
seed uncertainty  [min(R_1,R_2), max(R_1,R_2)]
log half-spread   abs(log R_1 - log R_2) / 2.
```

This two-seed envelope is a numerical/Krylov stability interval, not a
population confidence interval.  A pair fails the stability gate when
`max(R_1,R_2)/min(R_1,R_2) > 1.25`; a failed gate is `AMBIGUOUS` regardless of
the point ratio.

Closed pair-level labels:

- **C4_SUPPORTED:** both seeds are valid, the stability gate passes, and the
  entire seed uncertainty interval lies in `[1.80,3.40]`.
- **C3_SUPPORTED:** both seeds are valid, the stability gate passes, and the
  entire seed uncertainty interval lies in `[0.80,1.25]`.
- **AMBIGUOUS:** every other numerical or validity outcome.

The exact-target overall label uses all probeable G6 exact-target final pairs
plus the Round-3 age-20 exact-target pair.  It is `C4_SUPPORTED` only if every
included exact-target pair is C4-supported, `C3_SUPPORTED` only if every one
is C3-supported, and `AMBIGUOUS` for mixed labels, any included invalid pair,
or no included pair.  G8 bracketing pairs and Round-3 ages 5--15 are reported
but cannot upgrade or override this label.  Archive loss is an inventory
limitation, not itself a numerical failure for pairs that cannot be included.

## 5. Frozen angle-1 tape-rho check

For every row in `mech/law-unification/paired_cancellation.csv` for which a
same-cell momentum telemetry set can be mapped without substitution, compute
the C1-normalized residual

```text
phi = observed_to_law_ratio /
      ((T / C_convention(T,mu)) / ((1-mu) * M_convention(T,mu))),
```

using the exact `C` and `M` definitions in
`mech/law-v2/theory/c_normalization_check.py`.  The response is `ln(phi)`.

For a telemetry run, reconstruct the energy-weighted same-fragment lag-1
correlation from every finite JSONL lag-1 row:

```text
rho_run = sum(cos_t * norm_t * norm_previous_same_fragment)
          / sum(norm_t * norm_previous_same_fragment).
```

Map each ledger momentum point to the frozen sampled rung with minimum pooled
seed-mean loss, then pool all available completed training seeds at that exact
cell/rung by summing the run numerators and denominators.  Do not use endpoint
loss or rho to choose a rung, do not use control-arm rho as a substitute, and
do not impute missing telemetry.  Record exclusions and reasons.

The microscopic test is OLS after demeaning both `ln(phi)` and Fisher
`atanh(rho)` within `(campaign,scale,convention,mu)` strata having at least two
mapped rows.  Report the through-origin slope, ordinary and HC1 standard
errors, two-sided normal-approximation p-values, the within-stratum Pearson
correlation, row/stratum counts, and a scatter table.  Also report the raw
unstratified fit as a labeled diagnostic.  This regression is descriptive and
cannot change the closed Hessian verdict.

## 6. Required artifacts

All committed outputs live under `mech/law-v2/discriminator/` and include the
protocol, inventory, adapter/runner and hashes, raw spectrum JSON manifests,
machine-readable summary, rho regression inputs/results, and a human-readable
report.  Large checkpoints and raw probe JSONs may remain on `dev16`; the
repository records their absolute paths, byte sizes, and SHA-256s.  The final
external note is `/private/tmp/h200-mech-discrim-note.md` with the exact form:

```text
DISCRIMINATOR: <verdict, ratios>
```
