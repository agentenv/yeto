# C3-vs-C4 mechanism discriminator

## Verdict

**Formal exact-target verdict: `AMBIGUOUS`.**

The primary exact `mu=0.9` versus `mu=0` final-checkpoint result is the
Round-3 age-20 pair:

```text
probe seed 20260727: lambda_mom = 42.8296923
                     lambda_mu0 = 59.5659199
                     ratio      = 0.7190301

probe seed 20260728: lambda_mom = 42.7478996
                     lambda_mu0 = 59.6106644
                     ratio      = 0.7171183

geometric-mean ratio = 0.7180736
seed envelope         = [0.7171183, 0.7190301]
stability quotient    = 1.002666 (passes the <=1.25 gate)
```

The complete envelope is far below C4's preregistered `[1.8,3.4]` range, so
the decisive exact pair does **not** support the curvature-ratchet prediction.
It is also below the preregistered C3 parity band `[0.80,1.25]`.  The protocol
therefore forbids promoting this stable but sub-parity result to
`C3_SUPPORTED`; the required label is `AMBIGUOUS`.  Directionally, the
momentum checkpoint is about 28% *flatter* than the control, the opposite of
C4's required 80--240% sharpness excess, but the result is not parity.

The bands and full-envelope rule were committed before any discriminator HVP
in commit `c862b3b184ada0687fab661b2fc1d01d9827ef3a`; the inventory-only rung
clarification was committed in `d45e96eae50e9103b890cd6ce1e754f9bc5a6303`,
also before any HVP.  The protocol SHA-256 used by the probes is
`cd85592d2594bd513b72ac4ad1d043c795d3f612a3089fdd6cb53e45997fd9c2`.

## Spectrum results

The uncertainty interval is the frozen two-probe-seed envelope.  It measures
randomized Krylov-start stability, not population/checkpoint-seed uncertainty.
Every pair passed the stability gate.

| pair | role | lambda_mom (seeds 27 / 28) | lambda_mu0 (seeds 27 / 28) | seed ratios | geometric mean [envelope] | label |
|---|---|---:|---:|---:|---:|---|
| G8 corrected, mu=.8, train seed 811 | contextual | 55.8776 / 55.8560 | 70.1370 / 72.2980 | .79669 / .77258 | .78454 [.77258,.79669] | `AMBIGUOUS` |
| G8 corrected, mu=.95, train seed 801 | contextual | 45.8943 / 43.4500 | 37.4946 / 38.1039 | 1.22402 / 1.14030 | 1.18142 [1.14030,1.22402] | `C3_SUPPORTED` |
| G8 raw, mu=.8, train seed 801 | contextual | 81.4725 / 82.0297 | 37.4946 / 38.1039 | 2.17291 / 2.15279 | 2.16283 [2.15279,2.17291] | `C4_SUPPORTED` |
| G8 raw, mu=.95, train seed 801 | contextual | 41.8561 / 40.2278 | 37.4946 / 38.1039 | 1.11632 / 1.05574 | 1.08561 [1.05574,1.11632] | `C3_SUPPORTED` |
| G8 raw, mu=.95, train seed 811 | contextual | 272.1062 / 272.1063 | 70.1370 / 72.2980 | 3.87964 / 3.76368 | 3.82122 [3.76368,3.87964] | `AMBIGUOUS` |
| Round-3 age 5 | trajectory | 122.1875 / 122.1588 | 154.2157 / 155.0536 | .79232 / .78785 | .79008 [.78785,.79232] | `AMBIGUOUS` |
| Round-3 age 10 | trajectory | 62.6347 / 64.2476 | 74.7817 / 76.7526 | .83757 / .83707 | .83732 [.83707,.83757] | `C3_SUPPORTED` |
| Round-3 age 15 | trajectory | 49.0537 / 49.1016 | 63.4559 / 63.8702 | .77304 / .76877 | .77090 [.76877,.77304] | `AMBIGUOUS` |
| **Round-3 age 20** | **primary exact target** | **42.8297 / 42.7479** | **59.5659 / 59.6107** | **.71903 / .71712** | **.71807 [.71712,.71903]** | **`AMBIGUOUS`** |

The G8 neighbors are deliberately non-adjudicating because G8 contains
`mu=0.8` and `mu=0.95`, not the requested `mu=0.9`.  They are also genuinely
heterogeneous: two parity labels, one C4-band label, and two ambiguous labels;
the two retained raw-mu=.95 training seeds alone range from approximately
1.09 to 3.82.  This makes a pooled G8 mechanism call unjustified.

The exact Round-3 trajectory is consistently below one: geometric ratios
`.790, .837, .771, .718` at ages 5, 10, 15, and 20.  Only age 10 lies inside
the frozen parity band, and no age shows the C4-required momentum sharpness
excess.

## Checkpoint inventory

The 225,427,003,969-byte full-fidelity archive was streamed to a complete
8,892-line table of contents before extraction.  The listing completed at
2026-07-29 01:09:09 UTC and has SHA-256
`7960593425e5ac2ada860bcc112c463a574c957b85111d658a0b96abaeb8ccb7`.
Extraction began only after that marker.  Exactly seven preregistered G8
checkpoint members were extracted; the sorted extracted path set is identical
to `archive-members.txt`, and no other member was written.  See
`extraction-manifest.md` for sizes and hashes.

The frozen inventory unit is a same-campaign/T/H/S/training-seed pair at each
arm's independently selected minimum-pooled-loss sampled rung.  This selects
`e2` except G6 raw `(H=256,S=5120)`, whose momentum arm selects `e1`.

### G6 (exact mu=.9 target)

There are 18 selected-rung same-seed candidate pairs: 3 local-work cells x 2
conventions x 3 training seeds.

- **Probeable from the archive: 0/18.**  The complete archive listing contains
  no `yeto-results-v6/` member at all.
- **n2-incomplete: 16/18.**  At least one member of each pair was recorded as
  `h200-n2`, whose capped evacuation tree has no weights.
- **n1 metadata but absent from the full archive: 2/18.**  These are corrected
  `(H=128,S=2560,seed=607,e2)` and corrected
  `(H=512,S=10240,seed=613,e2)`.  Both arms are recorded as `h200-n1`, but
  neither terminal checkpoint member occurs in the complete archive listing.

Thus no G6 final pair can honestly be probed, including every raw G6 pair.
The exact loss reasons and both expected member paths are in `inventory.csv`.

### G8 (bracketing diagnostics, not exact mu=.9)

Of 12 selected-rung same-seed pairs, 5 are probeable and 7 are n2-incomplete.
The probeable pairs are:

1. corrected mu=.8, seed 811;
2. corrected mu=.95, seed 801;
3. raw mu=.8, seed 801;
4. raw mu=.95, seed 801; and
5. raw mu=.95, seed 811.

These five pairs use seven unique checkpoint files because controls are reused.
No n2 weight was substituted or cross-seed paired.

### Round-3 panel (exact mu=.9)

All four corrected-vs-mu0 checkpoint pairs are retained and hash-verified at
per-fragment ages 5, 10, 15, and 20 (global steps 20, 40, 60, 80).  Each file
is 1,076,137,912 bytes.  Ages 5--15 are trajectory diagnostics; age 20 is the
primary exact-target final checkpoint.  Exact paths, sizes, and SHA-256s are
in `round3-inventory.csv`.

## Probe implementation and validation

The base adapter is Lane-E commit `c7650ef` file
`mech/lane-e/checkpoint_spectrum_probe.py`, SHA-256
`857c88c2a227c32f983c5d206c48d43f49792cdda2f797db691df1386e46d8bd`.
Its ordinary `--seed` is inert whenever a nonzero transverse checkpoint buffer
supplies the second starting vector, so merely rerunning it under two seed
numbers would not be an independent-seed check.  The preregistered local
extension changes only that second vector to a seed-controlled Gaussian vector
orthogonal to the held-out gradient.  Checkpoint parsing, deterministic data
panels, fp32 HVP, float64 orthogonalization, and NumPy block Lanczos are the
validated Lane-E implementations.  Extension SHA-256:
`fd75968eb091b3282e79eb8ad911d82c07b14938c119411e14ccead308f907b1`.

Before randomized probing, default mode was rerun on the Round-3 age-20 mu0
checkpoint and compared recursively with its original Lane-E JSON after
excluding runtime.  It passed with zero differences at `1e-6` absolute and
relative tolerance (`adapter-equivalence.json`).

Every arm used:

```text
device/host        CPU, dev16
threads            80 per process; at most two processes (160 total)
model              byte-frozen SmolLM2-135M copy
held-out data      eval.jsonl SHA-256 533838a0564b13519956a044d23ed8db6705ddc7ae5f0ddb96538f49460bcebc
panels/batch        first 4 deterministic packed panels, batch size 1
sequence/max rows  128 / 128
fragments/layout   4 / binpack
loss/train_on      cross_entropy / assistant
block steps/rank   4 / 8
probe seeds        20260727, 20260728
lambda_max         largest algebraic Ritz value, no mode filtering
```

All 36 randomized arm probes completed with the fixed hashes/settings.  Raw
JSONs and logs remain under
`/mnt/nvme1/yeto-mech-discriminator-20260728/`; committed
`spectrum-files.sha256` authenticates the raw JSON set.  The exact seed-level
values, raw paths, and result hashes are in `discriminator-seed-results.csv`
and `discriminator-summary.json`.

## Angle-1 microscopic check: phi versus tape rho

Twelve of 62 ledger rows have a complete frozen readout-to-evacuated-telemetry
mapping: all G8 pairs, pooling 36 selected-rung training runs.  G6 trees are
absent; G9B is incomplete/not evaluable; the other ledger campaigns have no
unambiguous evacuated telemetry mapping.  No missing rho was imputed.

For each run, the energy-weighted same-fragment lag-1 rho is
`sum(cos*n_t*n_prev)/sum(n_t*n_prev)`.  Runs were pooled at the frozen
minimum-pooled-loss rung, `phi` was recomputed with the exact C1 normalization,
and both `ln(phi)` and Fisher `atanh(rho)` were demeaned within
`(campaign,scale,convention,mu)` strata before OLS.

```text
mapped rows / strata       12 / 4
within-stratum slope       -4.066185 ln(phi) per Fisher-z(rho)
ordinary SE / p            2.116632 / 0.05472
HC1 SE / p                 1.669434 / 0.01486
within-stratum Pearson r   -0.587548

raw unstratified slope     -4.064004
raw ordinary SE / p        1.874757 / 0.03018
raw Pearson r              -0.565410
```

The available G8 panel therefore has a statistically visible *negative*
association: higher measured rho accompanies a deeper (more negative)
`ln(phi)` after broad-stratum demeaning.  That is not the natural sign for a
simple "more decorrelation -> more drag" microscopic story, and the narrow
G8-only coverage plus age variation makes it descriptive rather than causal.
It does not alter the closed Hessian verdict.  Inputs, telemetry hashes,
exclusions, and both regressions are committed in the `rho-regression-*`
artifacts.

## Bottom line

The free exact-target curvature probe rules out the specific C4 signature it
was designed to find: the final momentum Hessian is not 1.8--3.4 times sharper
than control; it is approximately 0.718 times as sharp.  Because the result is
also outside the preregistered C3 parity band, the defensible matrix label is
`AMBIGUOUS`, not `C3_SUPPORTED`.  The G8 neighbors are too convention- and
training-seed-sensitive to repair that label, and the tape-rho sign does not
provide an independent rescue for the simple interference-origin story.
