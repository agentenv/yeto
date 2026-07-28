# Law unification verdict: FAILS

**Decision.** The global law fails and no prespecified optimizer-convention stratum clears the frozen PARTIAL threshold. This is not a cosmetic rejection: the best-fitting shared curve has fixed-effect heterogeneity $\chi^2=2811548.1$ on 103 degrees of freedom (p<1e-300) and covers only 2/106 (1.9%) of the noise-eligible marginal 95% intervals, versus the frozen 90% requirement. The honest v2 keystone label is therefore **FAILS**, not a law-level collapse.

## Evidence and estimand

The ledger contains all 112 accepted banked tuned optima: DISAMBIG=6, G12=6, G4C=4, G5B=3, G6=36, G8=15, G9A=2, G9B=2, TP-pilot=6, TP-v1=8, TP-v2=7, TP-v3=17. It excludes no accepted point because it disagrees with the hypothesis. The separate exclusion ledger has 24 rows: 12 non-interior/nonconvex/extrapolated two-parameter fits, 11 superseded cumulative estimators (G4/G4B and G5), and one out-of-scope G13/G13B external transfer. Exact A/A-equivalent controls remain visible with dependence labels. The two final 7B points use the sole retained seed and point-mass empirical intervals; they enter equal-point OLS and the figure, but not WLS or seed-noise calibration. The final SHA-bound scale artifact is used because the raw joint G9 readout was not found under its expected evacuation-archive path.

For each point the audited response is $z=\log\eta^*-\log[(1-\mu)M]$. The primary equal-point model is $z=\log\eta_0(\mathrm{scale})+T\log q$. Raw Nesterov uses $M=(1-\mu^{T+1})/(1-\mu)$, G12 heavy-ball uses $M=(1-\mu^T)/(1-\mu)$, corrected Nesterov uses $M=1$, and $\mu=0$ uses $M=1$. No campaign, $S$, or $H$ term is admitted to the keystone fit.

## Global fit and sharing tests

Equal-point OLS gives **q=0.987576** (95% CI 0.982264-0.992916); fitted $\eta_0$ values are 135M 0.03135 [0.025625, 0.038355], 1.7B 0.0051841 [0.002446, 0.010987], 7B 0.0062621 [0.0017123, 0.022902]. The fixed-effect inverse-variance sensitivity gives q=0.987116 (0.987088-0.987143), showing that narrow seed intervals do not merely sharpen the same descriptive fit. Leave-one-campaign-out OLS spans q=0.981551-0.988925; confirmatory-only OLS gives q=0.987762; and one-per-dependence-group OLS gives q=0.987819.

Requested slope-sharing tests: mu: OLS p=4.99e-05, WLS p<1e-300 (rejected); scale: OLS p=0.195, WLS p<1e-300 (rejected); convention: OLS p=7.11e-04, WLS p<1e-300 (rejected). The scale test is restricted to 135M and 1.7B because two 7B points at one age cannot identify a 7B slope. A restriction can be descriptively stable in OLS yet still fail under the seed-precision audit; the report calls sharing successful only when both prespecified tests survive.

| Convention | points | campaigns | ages | WLS q | heterogeneity p | 95% coverage | PARTIAL bar? |
|---|---:|---:|---:|---:|---:|---:|:---:|
| mu0 | 42 | 11 | 6 | 0.99513 | <1e-300 | 2.6% | no |
| nesterov_raw | 42 | 11 | 6 | 0.98995 | <1e-300 | 0.0% | no |
| nesterov_corrected | 25 | 5 | 4 | 0.92812 | <1e-300 | 4.2% | no |
| heavy_ball | 3 | 1 | 3 | 0.83071 | <1e-300 | 33.3% | no |

## Residual mechanism target

The decisive residual structure is convention by age. In matched G6 cells, dividing a momentum optimum by its same-$T$/same-$S$ $\mu=0$ control and by $(1-\mu)M$ cancels both $\eta_0(\mathrm{scale})$ and $q^T$; the law therefore requires exactly zero log2 bits at every age. Observed median bits are raw (T=2: +0.583, T=5: -0.938, T=10: -2.143, T=20: -3.077); corrected (T=2: -0.100, T=5: -0.334, T=10: -0.506, T=20: -0.634). The raw arm reverses across age and misses by nearly three bits at $T=20$, so neither a different shared q nor a scale intercept can repair it.

By contrast, within the balanced G6 factorial and after $T$ fixed effects, each doubling of local-work scale $S$ shifts the residual by only **+0.0976 bits** (95% CI -0.2819 to +0.4772, p=0.604). Thus local work is not the first-order break in this audit. The largest absolute global residual is TP-v2:135M:T40:mu0.9:raw:H64:S2560 at -3.261 bits. Residual summaries by campaign, scale, exact $\mu$, and convention are in `residual_summary.csv`; the full matched cancellation ledger, correlations, and 15 largest misses are in `residual_structure.json` and `paired_cancellation.csv`.

The next mechanism target is the convention-specific age/history transform itself - especially why the raw and heavy-ball tuned-rate ratio decays with age while the proposed multiplier grows toward one - not another refit of one universal scalar q. A law-paper rebrand is not supported by these banked data unless a revised transform is prospectively specified and independently tested.
