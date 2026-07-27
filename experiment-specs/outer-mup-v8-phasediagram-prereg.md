# Outer-muP v8 MINI tuned-loss phase-diagram preregistration

**Program ID:** `outer-mup-v8-phasediagram`

**Status:** `PREREGISTERED_PENDING_PRIORITY_FLEET` — no v8 launch authority, GPU process, scientific attempt, or outcome exists.

**Registered:** 2026-07-26; prospectively amended to MINI before any v8 launch/outcome.

**Authoritative contract:** `experiment-specs/outer-mup-v8-phasediagram-prereg.json`.

## Pre-outcome MINI amendment

This contract prospectively supersedes the unlaunched 45-curve/900-cell design frozen at commit `4175b71b72251aa279345e3ed858caa7ce5b39f9`. The operator re-scoped the Tuesday-deadline deliverable in `/private/tmp/h200-phasediag-note.md` before any v8 authority, process, result root, attempt, or outcome: retain only `T={2,5,20}`, `mu={.8,.95}`, raw and corrected arms, and three paired seeds `{801,809,811}`. The superseded full manifest is forbidden to launch. v6 drain and the separate seal-verification cells have strict scheduling priority.

## Design and deliverable

At SmolLM2-135M, `M=4`, fixed `H=512`, and `S=512T`, v8 MINI produces separate raw- and bias-corrected maps of the independently tuned optimum relative to a same-T no-momentum optimum:

```text
Delta(T,mu,arm) = L*_arm(T,mu) - L*_mu0(T)
```

There are three T values, one shared mu0 curve plus two raw and two corrected curves per T, four LR points per curve, and three seeds: `15 * 4 * 3 = 180` fresh cells and 12 momentum comparisons.

Every ladder is symmetric in `log2(eta)` at offsets `{-1.5,-.5,+.5,+1.5}` bits (2x adjacent, 8x endpoint span). Exact numbers are authoritative in the JSON. Centers are:

| T | S | mu0 | raw .8 | raw .95 | corrected .8 | corrected .95 |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1024 | 0.07105139738 | 0.03476018487 | 0.03073723984 | 0.01367230955 | 0.003393432799 |
| 5 | 2560 | 0.04701771281 | 0.01450956336 | 0.01035221835 | 0.008295954957 | 0.00202582388 |
| 20 | 10240 | 0.02517527596 | 0.004566030958 | 0.001680984048 | 0.00287910341 | 0.0006481599845 |

Centers use the already-registered v3/v4-informed absolute-rate surface and frozen Theory Lane C momentum transfer. Raw transfer includes the exact code-true transient `1/(1-mu^(T+1))`; corrected transfer uses the frozen measured drift surface. No result-aware recentering, ladder extension, or point removal is permitted.

## Exact reuse

Reuse count is zero. v3 and v6 structural mu0 coordinates fail exact identity because scientific/training seeds, shuffled input bytes, and source commits differ. The already verified v8 input pool contains two inert shards for seeds 821/823 from the prospectively superseded full design; no MINI command or analysis key references them. Only seeds 801/809/811 are scientific.

## Frozen estimator and labels

For each curve fit `loss=a*x^2+b*x+c`, `x=log2(eta)`, to the four three-seed means. Accept only `a>0` with the vertex strictly inside the registered ladder; then `eta*=2^(-b/(2a))` and `L*=c-b^2/(4a)`. Any missing/nonfinite cell, nonpositive curvature, or boundary/outside vertex is unbracketed and never extrapolated.

The analyzer runs 10,000 paired three-seed bootstrap draws (`seed=20260728`), sharing one three-index draw across all 15 curves. MINI is evaluable only with 180/180 valid cells, 15/15 interior point fits, and at least 9,500 complete interior refits. It reports pointwise intervals and a primary simultaneous 95% interval using the 95th percentile of the valid-draw maximum absolute deviation across all 12 Delta values. At practical margin `0.01`:

- `HELPS`: simultaneous high < -0.01.
- `HURTS`: simultaneous low > +0.01.
- `NEUTRAL`: the full simultaneous interval is inside [-0.01,+0.01].
- `UNCERTAIN`: evaluable but none of those three.
- `NOT_EVALUABLE`: incomplete/invalid/unbracketed or <9,500 valid refits.

Frozen analyzer SHA-256: `00166147bae8566e2d2980c50cb94a282b3ecfb302a58025f98c88cdb870b699`.

## Mandatory MINI feasibility simulation

The CPU-only artifact `outer-mup-v8-phasediagram-gatesim.json` has SHA-256 `a5bf834db795dfbbfd2413ceda628a4bd6df0f2fb569b6bfbffb4aa81ae5bf84`. It transports sealed v3 five-seed per-rung SDs and T/arm curvature, then applies the exact MINI point gate and exact 10,000-draw shared bootstrap (10 unique multinomial count vectors; literal spot check exact).

Primary **`P_eval=1.000`** (500/500; Wilson 95% `[0.9924,1.0000]`); all primary datasets retain all 15 point fits and 10,000/10,000 refits. The pre-existing v3 fitted-vertex-shift sensitivity is also 500/500. This clears the fixed 0.8 readiness threshold but remains a measured-noise transport model, not a guarantee.

## Cost and scheduling priority

Measured v3 p90 runtimes imply 28.529 GPU-hours. On the registered bounded eight-GPU subfleet (GPUs 0-3 on each node), plus 10% controller/eval/seal overhead, planning cost is **3.923 fleet-hours**; ideal mean is 3.479 hours. The post-authority wall ceiling is 6 hours.

Launch is forbidden until both conditions hold: (1) all v6 slot queues are `DRAINED`; and (2) `/private/tmp/h200-seal-note.md` contains the exact line `SEAL VERIFICATION CELLS DONE`. A fresh gate proof also requires an empty bracketed priority-process check:

```bash
pgrep -af '[r]un_slot_v6.py|[c]ompare_diloco.py.*yeto-results-v6|[a]nalyze_v6.py|[r]un_slot_v9.py|[s]moke_v9_qwen.py|[f]reeze_v6_selection.py'
```

v8 MINI may use only GPUs 0-3 per node and only after those higher-priority conditions. Apparent idle GPUs alone never authorize launch.

## Input/evidence/rails

Dual-node CPU preparation already sealed input manifest `5f4235e56be5fc968227e02a6c9a6ebe57277d2736fb2947da14f7bd7f15a20b` and capacity report `de532769475ef116001748ffedc3824ab4292b93664cb0b2c5b27bdd48294d94` (minimum 26,573 blocks/learner versus 20,480 required). The model files, input bytes, every command, and execution scripts are hash-bound.

A cell counts only with exact command/source hashes, exact S steps on four learners, `4T` strict-quorum tape/rho rows, fixed `c_steps=512`/`c_tokens=65536`, finite 1,024-row evaluation, and complete artifact evidence. Retry is loss-blind, whole-curve (12 cells), one attempt, and only for an enumerated infrastructure cause. Execution uses eight deterministic longest-S-first queues, one cell/GPU, four learners packed, 30-second heartbeats, watchdogs, and process-group-scoped termination.

## Claim boundary

Claims are limited to the 12 registered tuned-loss contrasts at these exact coordinates. `D` cannot substitute for a performance map; `UNCERTAIN` cannot be called neutral; prior outcomes cannot be pooled into MINI; and all statuses, fits, intervals, and labels must be reported.
