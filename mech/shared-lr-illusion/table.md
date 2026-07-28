# Shared-LR illusion audit

> **Post-hoc descriptive analysis of already-sealed data.** No result in this table was preregistered as a test of the shared-LR illusion; no gate was run or changed, and no new training was performed.

Gain is endpoint evaluation loss for `mu=0` minus endpoint evaluation loss for standard raw Nesterov `mu=.9`; positive values favor momentum. Brackets are pointwise paired-training-seed bootstrap 95% intervals (10,000 refits). `HELP`, `NULL`, and `HURT` mean that the interval is respectively above, contains, or is below zero; `NOT_EVALUABLE` means the frozen minimum-valid-refit threshold was not met. `a1` evaluates both fitted quadratics at the no-momentum fitted optimum; `a2` evaluates both at eta=.7 only when both observed ladders contain .7; `b` evaluates each arm at its own fitted optimum. The position in parentheses after `a1` is the shared eta relative to the momentum ladder; `above 1.30x`, for example, means 30% above its highest rung.

## Result

Under the requested no-momentum-optimum shared-LR rule, momentum appears to help in **0 of 28** bootstrap-evaluable shared-rate comparisons. Among the **26** cells that also have a per-arm-tuned estimate, **0** are HELP-to-NULL/HURT flips after retuning, and **0** shared-LR HELP calls survive retuning. However, only **0** of those a1 evaluations lie inside the momentum ladder (28 require quadratic extrapolation). The DiLoCo-style eta=.7 rule has **0** eligible comparisons because none of the exact-mu paired ladders straddles .7. With honest per-arm retuning, momentum is HELP/NULL/HURT in **1/5/20** of 26 evaluable comparisons.

## Full table

| campaign | scale | T | S | H | a1: shared at eta0* gain [95% CI] | a2: shared at .7 | b: per-arm-tuned gain [95% CI] | c: flip a1 / a2 |
|---|---:|---:|---:|---:|---|---|---|:---:|
| 135M pilot S-scan | 135M | 2 | 1024 | 512 | −0.0372 [−0.0645, −0.0269] HURT (above 2.01x) | NA | +0.0035 [−0.0067, +0.0079] NULL | N / NA |
| 135M pilot S-scan | 135M | 10 | 5120 | 512 | −0.3805 [−0.4074, −0.3542] HURT (above 2.70x) | NA | −0.0115 [−0.0127, −0.0103] HURT | N / NA |
| 135M pilot S-scan | 135M | 20 | 10240 | 512 | −0.2281 [−0.2306, −0.2246] HURT (above 2.85x) | NA | +0.0017 [−0.0028, +0.0078] NULL | N / NA |
| 135M v1 | 135M | 160 | 2560 | 16 | −0.1065 [−0.1087, −0.1043] HURT (above 5.06x) | NA | −0.0017 [−0.0018, −0.0016] HURT | N / NA |
| 135M v1 | 135M | 40 | 2560 | 64 | −0.0967 [−0.1082, −0.0869] HURT (above 4.08x) | NA | NA | NA / NA |
| 135M v1 | 135M | 10 | 2560 | 256 | −0.2767 [−0.3415, −0.2642] NOT_EVALUABLE (above 1.47x) | NA | NA | NA / NA |
| 135M v1 | 135M | 5 | 2560 | 512 | −0.1874 [−0.2252, −0.1657] HURT (above 2.41x) | NA | −0.0045 [−0.0059, −0.0029] HURT | N / NA |
| 135M v2 | 135M | 160 | 2560 | 16 | −0.1083 [−0.1105, −0.1053] HURT (above 5.01x) | NA | −0.0013 [−0.0015, −0.0011] HURT | N / NA |
| 135M v2 | 135M | 40 | 2560 | 64 | −0.1121 [−0.1178, −0.1012] HURT (above 3.02x) | NA | −0.0160 [−0.0169, −0.0154] HURT | N / NA |
| 135M v2 | 135M | 10 | 2560 | 256 | −0.0973 [−0.3012, +0.0861] NULL (above 2.94x) | NA | NA | NA / NA |
| 135M v2 | 135M | 5 | 2560 | 512 | −0.2198 [−0.3426, −0.1439] HURT (above 2.41x) | NA | −0.0034 [−0.0111, +0.0061] NULL | N / NA |
| 135M v3 T-scan | 135M | 2 | 1024 | 512 | −0.0642 [−0.0697, −0.0586] HURT (above 1.43x) | NA | −0.0008 [−0.0028, +0.0016] NULL | N / NA |
| 135M v3 T-scan | 135M | 5 | 2560 | 512 | −0.1639 [−0.1868, −0.1412] HURT (above 2.40x) | NA | −0.0057 [−0.0081, −0.0033] HURT | N / NA |
| 135M v3 T-scan | 135M | 10 | 5120 | 512 | −0.3537 [−0.3748, −0.3328] HURT (above 3.69x) | NA | −0.0129 [−0.0139, −0.0119] HURT | N / NA |
| 135M v3 T-scan | 135M | 20 | 10240 | 512 | −0.2975 [−0.3023, −0.2910] HURT (above 5.37x) | NA | −0.0280 [−0.0290, −0.0270] HURT | N / NA |
| 135M v3 T-scan | 135M | 40 | 20480 | 512 | NA | NA | NA | NA / NA |
| 135M v6 factorial | 135M | 2 | 2560 | 1280 | −0.0572 [−0.0603, −0.0548] HURT (above 1.31x) | NA | −0.0026 [−0.0038, −0.0009] HURT | N / NA |
| 135M v6 factorial | 135M | 2 | 5120 | 2560 | −0.0667 [−0.0692, −0.0640] HURT (above 1.32x) | NA | −0.0058 [−0.0071, −0.0049] HURT | N / NA |
| 135M v6 factorial | 135M | 2 | 10240 | 5120 | −0.0618 [−0.0653, −0.0574] HURT (above 1.48x) | NA | −0.0058 [−0.0073, −0.0049] HURT | N / NA |
| 135M v6 factorial | 135M | 5 | 2560 | 512 | −0.1376 [−0.1519, −0.1261] HURT (above 2.16x) | NA | −0.0091 [−0.0110, −0.0062] HURT | N / NA |
| 135M v6 factorial | 135M | 5 | 5120 | 1024 | −0.1675 [−0.1723, −0.1635] HURT (above 2.21x) | NA | −0.0097 [−0.0113, −0.0085] HURT | N / NA |
| 135M v6 factorial | 135M | 5 | 10240 | 2048 | −0.1708 [−0.1737, −0.1653] HURT (above 2.47x) | NA | −0.0146 [−0.0158, −0.0133] HURT | N / NA |
| 135M v6 factorial | 135M | 10 | 2560 | 256 | −0.2660 [−0.2885, −0.2423] HURT (above 3.13x) | NA | −0.0114 [−0.0165, −0.0082] HURT | N / NA |
| 135M v6 factorial | 135M | 10 | 5120 | 512 | −0.3166 [−0.3219, −0.3085] HURT (above 3.26x) | NA | −0.0128 [−0.0133, −0.0124] HURT | N / NA |
| 135M v6 factorial | 135M | 10 | 10240 | 1024 | −0.3416 [−0.3460, −0.3389] HURT (above 3.72x) | NA | −0.0190 [−0.0202, −0.0178] HURT | N / NA |
| 135M v6 factorial | 135M | 20 | 2560 | 128 | −0.3403 [−0.3504, −0.3290] HURT (above 4.09x) | NA | −0.0100 [−0.0108, −0.0093] HURT | N / NA |
| 135M v6 factorial | 135M | 20 | 5120 | 256 | −0.3732 [−0.4172, −0.3279] HURT (above 4.35x) | NA | −0.0150 [−0.0192, −0.0116] HURT | N / NA |
| 135M v6 factorial | 135M | 20 | 10240 | 512 | −0.4457 [−0.4580, −0.4378] HURT (above 5.31x) | NA | −0.0212 [−0.0220, −0.0200] HURT | N / NA |
| 1.7B v4-family combined | 1.7B | 5 | 2560 | 512 | −0.3871 [−0.4163, −0.3623] HURT (above 1.69x) | NA | +0.0240 [+0.0161, +0.0323] HELP | N / NA |
| 1.7B v4-family combined | 1.7B | 20 | 10240 | 512 | −1.0306 [−1.3014, −0.7918] HURT (above 2.06x) | NA | +0.0060 [−0.0021, +0.0145] NULL | N / NA |

## Eligibility and exclusions

The main table contains all 30 banked exact-mu pairs found in the sealed two-parameter, factorial, and 1.7B v4-family artifacts, including rows whose fitted optimum is unbracketed. The v4, v4b, and v4c stages are nested; only the final five-seed G4C combined grid is counted, avoiding duplicate outcomes. The standard raw Nesterov arm is used; the separately bias-corrected arm is not silently pooled with it. `NA` in a flip column means that the shared or tuned comparison is unavailable; it is not counted as a no-flip result.

The requested v8 audit is present but cannot enter an exact `mu=.9` table:

| campaign | scale | T | S | H | banked raw momentum mu values | disposition |
|---|---:|---:|---:|---:|---|---|
| 135M v8 phase diagram | 135M | 2 | 1024 | 512 | 0.8, 0.95 | excluded from the exact-mu table: v8 banked raw-Nesterov sweeps are mu=.8 and .95, not mu=.9; no interpolation in mu |
| 135M v8 phase diagram | 135M | 5 | 2560 | 512 | 0.8, 0.95 | excluded from the exact-mu table: v8 banked raw-Nesterov sweeps are mu=.8 and .95, not mu=.9; no interpolation in mu |
| 135M v8 phase diagram | 135M | 20 | 10240 | 512 | 0.8, 0.95 | excluded from the exact-mu table: v8 banked raw-Nesterov sweeps are mu=.8 and .95, not mu=.9; no interpolation in mu |

## Honest caveats

- **The a1 comparison is structurally conservative for momentum wherever both optima are accepted.** For convex quadratics, the per-arm-tuned momentum loss cannot exceed its loss at eta0*, so the tuned gain is algebraically at least the a1 gain. A HELP-to-NULL/HURT point-estimate reversal is impossible under this exact estimand; only an interval-label reversal could occur because refitting changes uncertainty.
- **a1 is outside the sampled momentum curve wherever marked `above` or `below`.** Those numerical values are quadratic extrapolations and are descriptive stress tests, not in-grid evidence. The table reports them because the requested eta0* rule otherwise has no estimate, but the count of supported in-grid comparisons is stated separately.
- **The .7 default is not extrapolated.** None of the exact mu=0/.9 paired grids contains eta=.7. The data therefore cannot support a DiLoCo-default illusion claim; reporting a fitted value at .7 would be a long-range extrapolation.
- **Grid resolution is coarse.** Curves have four to six eta rungs and two to five training seeds. Quadratic minima and percentile intervals inherit that model and resolution; pilot extrapolations retain the already-disclosed pilot convention and are labeled as such.
- **Repeated coordinates are not independent replications.** Pilot, v1, v2, v3, and v6 differ in seeds, shuffled inputs, grids, or campaign source. The row count is descriptive and is not a meta-analytic sample size. Pointwise intervals are not multiplicity-adjusted.
- **This question was posed after outcomes were known.** The frozen bootstrap mechanics are reused, but the shared-LR estimand, inclusion summary, and flip count are post-hoc. No causal claim about optimizer age follows from this table alone.

## Reproduction and provenance

Run `analyze.py` with explicit paths to the sealed two-parameter root and the canonical G4C, G6, and G8 readouts. `results.json` contains every eta grid, fit status, bootstrap-valid count, source hash, and unrounded value; `table.csv` is the flat machine-readable table. The reproducer also fails closed unless all 72 pilot and 460 v3 CSV rows match their raw snapshots and all 1,557 hashes in the recorded raw-source lists verify.

| source | SHA-256 |
|---|---|
| two-parameter pilot cells | `26b5e4860fa7aaafe83b70bb1155331985a77a68d7105348560253817bd28bca` |
| two-parameter v1 readout | `c2dcd6b9ab7dce0dc28d1e2473a72c7e0bdb8d6221728f09503a6354a39cae2b` |
| two-parameter v2 readout | `5d4eed9685f25fd1db3135319908a045a389300a008a67bc009cd178cabe2fc8` |
| two-parameter v3 cells | `29970559ff513ec19a9d5d0ab9f802bb6fba1898984877854c9b3485b89e320c` |
| two-parameter v3 manifest | `8fae6137d673d4c57861b37de09a42c5c462b0dff692cf10bde49e73caa554fc` |
| two-parameter pilot raw hash list | `32f5b06698dbf05c9154f98d60e9965d67a4e3d8acc9bba49611df08ca618dd4` |
| two-parameter v3 raw hash list | `ed1433898b61824b485e043a751d7b998c2cadbd875a9509725e6d1685f4fbb5` |
| 135M frozen G3 readout | `d4a3cde6aa47580dff255c7a66030ab997a95f4072b1883bf71aa54d7da744c8` |
| 1.7B canonical G4C | `16bab85dce3a83f42c8b1c91be8d91eb1ec1372cfba572f2ffd9b491044b04aa` |
| 135M canonical G6 | `7e7e9cd8901520cc8f0ca822151af92c4c6183ee99bf02199bc1452a2763af8c` |
| 135M canonical G8 | `9884f0775b30a35964e7df878bd0569b62f24af23619d90b4ff346d1afae596c` |
