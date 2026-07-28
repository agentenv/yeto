# Law-unification evidence policy (frozen before fitting)

Frozen on 2026-07-28 (America/Los_Angeles), before the law-fit code or any
law-fit result was produced.  This audit is banked-data-only: it launches no
training and treats missing science as missing rather than imputing it.

## Measurement universe

The inventory is outcome-blind at the curve level: every banked campaign that
tuned the production outer learning-rate variable and records enough metadata
to identify `(model scale, T, mu, optimizer convention)` is enumerated.  The
requested universe is:

1. the 135M two-parameter master table (pilot, v1, v2, v3), including the
   fixed-`T=5` exploratory disambiguation curves;
2. the complete 135M v6 `T x S` factorial and the fresh v8 phase-diagram
   curves (the latter supplies additional `mu=.8/.95` measurements);
3. the canonical final v4-family 1.7B fits (G4C), plus the banked prospective
   1.7B `T=10` raw/corrected fits;
4. the final G9 7B relative fits, with their singleton-seed uncertainty
   limitation preserved;
5. the G12 135M heavy-ball scan; and
6. the canonical combined G5B SNOO fits.

G13/G13B uses a different model family *and* a different corpus.  It is not a
point on the requested SmolLM scale axis; it is listed as an out-of-scope
external-transfer exclusion and may be used only as a disclosed sensitivity,
never silently mixed into the primary scale fit.

## Canonical point and duplicate rules

- The unit is one pooled tuned optimum for one scientific curve, not an eta
  rung, a transferred prediction, or a tuned-loss contrast.
- A curve enters the numerical fit when its canonical analyzer reports a
  finite accepted optimum and a seed-based 95% interval.  Strictly interior
  fits enter.  A campaign's prospectively accepted near-bracketed rule may
  enter, but the status remains visible.  Unbracketed, nonconvex, missing, and
  extrapolated-only curves are listed in `excluded_points.csv` and never
  converted into pseudo-optima.
- Exploratory curves are not discarded: eligible ones enter with an
  `exploratory` flag, and a confirmatory-only sensitivity is mandatory.
- Successive cumulative analyses of the same cells are not independent
  measurements.  G4 and G4B are provenance rows, while the final five-seed
  G4C estimator is the sole v4-family fit input.  Likewise, the final combined
  G5B estimator supersedes the earlier unbracketed SNOO analyses.
- Repeated campaigns with fresh cells remain repeated measurements.  Exact
  A/A-equivalent controls are retained in the assembly table and identified by
  a dependence/duplication field; no claim relies on treating them as
  independent.
- A reported interval with invalid conditional bootstrap refits is retained as
  a visibly qualified interval and point.  It is excluded from the
  seed-noise chi-square calibration but retained in the equal-point global fit
  and collapse plot.  A zero-width singleton-seed G9 interval is handled the
  same way: the point is retained, but it gets no infinite statistical weight.

## Law encoded without reinterpretation

The audited equation is

```text
eta_star = eta0[scale] * (1 - mu) * M(T, mu, convention) * q**T
```

with one fitted intercept `eta0[scale]` for each requested model scale and one
candidate shared `q`.  No `S`, `H`, campaign, or outcome-dependent offset is
allowed in the keystone fit.

The code-true finite-age factors are fixed before fitting:

- raw Nesterov: `M=(1-mu**(T+1))/(1-mu)`;
- heavy-ball (G12): `M=(1-mu**T)/(1-mu)`;
- bias-corrected Nesterov: `M=1`, because the production correction removes
  the raw finite-age factor while retaining the equation's leading
  `(1-mu)` normalization; and
- `mu=0`: `M=1` under every convention.

SNOO condition `b` uses raw Nesterov; its `mu=0` controls use `M=1`.

The linearized fit is therefore

```text
log(eta_star / ((1-mu)*M)) = log(eta0[scale]) + T*log(q).
```

The collapse plot additionally divides the left side by the fitted
`eta0[scale]`, so its common curve is exactly `q**T`.  Its horizontal axis is
the per-fragment effective update age `T`; the `T+1` Nesterov and `T`
heavy-ball age conventions remain in `M` rather than being hidden in the
horizontal coordinate.

## Fit, sharing, and verdict rules

- The primary parameter estimate is equal-point ordinary least squares in
  log space.  This prevents narrow, correlated, or singleton intervals from
  dominating.  A fixed-effect inverse-variance fit over intervals eligible
  for noise calibration is a mandatory sensitivity.
- Sharing is tested with prespecified nested log-linear models: one `q` versus
  `q` by exact `mu`, by model scale, and by optimizer convention.  Report
  likelihood/F tests, information criteria, and stratum estimates; do not
  select a sharing pattern and then call it prospective.
- Seed-noise consistency is evaluated only on nondegenerate, formally usable
  intervals, using log-CI-derived standard errors.  Coverage, standardized
  residuals, fixed-effect heterogeneity, and leave-one-campaign/confirmatory
  sensitivities are all reported.
- `COLLAPSES` requires the shared law's residuals to be consistent with the
  reported seed uncertainty: fixed-effect heterogeneity `p >= .05`, at least
  90% marginal 95% interval coverage, and no rejected (`p < .05`) requested
  sharing test.  `PARTIAL` requires the global rule to fail but at least one
  prespecified optimizer-convention stratum with at least eight points, two
  independent campaigns, and three distinct ages to meet the same
  heterogeneity and coverage thresholds; the breaking stratum(s) must be
  named.  A three-point scan cannot earn `PARTIAL` by fitting itself.
  Otherwise the verdict is `FAILS`.
  Visual attractiveness or a fitted `q` near 0.9703 cannot override these
  rules.
