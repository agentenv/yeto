# Outer-muP v18 FedAdam finite-age preregistration

**Program ID:** `outer-mup-v18-fedadam-finite-age`

**Status:** `REGISTERED_PRE_OUTCOME`; math, Lean, and CPU-only gatesim exist,
and no v18 training cell has run.

**Registered:** 2026-07-28.

**Registration object:** the Git commit containing this file. That commit must
be present on the configured GitHub remote before any v18 manifest, authority
file, result directory, or training process is created. The external operator
note `/private/tmp/h200-fedadam-note.md` records the pushed commit SHA. A local
commit, an unpushed commit, or a note without a remotely reachable commit does
not authorize a run.

## 1. Prospective claim

With raw Reddi-style FedAdam server moments, `m_0=v_0=0`, `beta1=0.9`,
`beta2=0.99`, and exactly `T` calls to one moment state, the constant-input
terminal multiplier predicts

\[
D_{\rm Adam}(T)
=\frac{\sqrt{1-0.99^T}}{1-0.9^T}.
\]

`D_adam` is the predicted tuned FedAdam-rate deviation in the registered scalar
constant-input normalization relative to the same-age SGD control. The primary
experiment uses `tau=0`, with a zero-safe coordinate rule, so this target is
scale-free. The derivation, the `tau>0` generalization, and the bias-correction
variants are frozen in `mech/fedadam-prediction/derivation.md`.

### Registered numeric prediction

| `T` | predicted `D_adam(T)` | registered log2 band | equivalent numeric band |
|---:|---:|---:|---:|
| 2 | **0.742459788403** | `log2(D_obs/D_pred) in [-0.35,+0.35]` | `[0.582522143309, 0.946310013667]` |
| 5 | **0.540601963459** | `log2(D_obs/D_pred) in [-0.35,+0.35]` | `[0.424147703821, 0.689029977676]` |
| 20 | **0.485783579482** | `log2(D_obs/D_pred) in [-0.35,+0.35]` | `[0.381138071481, 0.619160623805]` |
| 40 | **0.583982312067** | `log2(D_obs/D_pred) in [-0.35,+0.35]` | `[0.458183235501, 0.744320862012]` |

This is a registered U-shape:

`D(2) > D(5) > D(20) < D(40)`, with every listed value below one.

It differs qualitatively from the prior raw Nesterov prediction, which is
above one and decreases toward one. No v18 result, pilot, implementation trace,
or model-specific fit supplied these four values.

The primary target is the no-floor raw-FedAdam curve. The already-frozen
unit-coordinate sensitivity at `tau/|g|=0.001` is
`{0.746975970328, 0.542501404863, 0.486435547194, 0.584412901800}`; it lies
only `{0.00875,0.00506,0.00193,0.00106}` log2 bits above the primary curve and
cannot replace it in the verdict. Fully PyTorch-style bias-corrected Adam has
the distinct prediction `D(T)=1` at every age and is not an eligible FedAdam
arm for this registration.

## 2. Fixed 135M design

The future scan has two arms, four horizons, four eta levels, and three fresh
training seeds: `2 * 4 * 4 * 3 = 96` cells.

- Model: `HuggingFaceTB/SmolLM2-135M`, revision
  `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.
- Data: the frozen no-wrap v3 135M Capybara train/eval bundle. Train file
  `/root/yeto-data/outer-mup-v3/scale-s2560/raw/train.jsonl`, SHA-256
  `e680a29ea8c8fc7c99efdceb4f62e485d3eed1ac2afd15bab43b506cb3f4ecaf`;
  eval file `/root/yeto-data/outer-mup-v3/scale-s2560/raw/eval.jsonl`, SHA-256
  `533838a0564b13519956a044d23ed8db6705ddc7ae5f0ddb96538f49460bcebc`.
- Four learners/fragments (`M=4`), `S=2560` inner steps, sequence length 128,
  inner AdamW learning rate `0.001`, and otherwise the frozen v6 135M work
  contract.
- `T in {2,5,20,40}` is calls seen by each fragment's own server-optimizer
  state. `H=S/T`, hence `H in {1280,512,128,64}`. Round-robin four-fragment
  execution therefore makes `4*T` global fragment commits while each moment
  state has exactly the registered age.
- Seeds `{1009,1013,1019}` with training seeds
  `{10091009,10131013,10191019}`. The same training seed is paired across every
  eta, arm, and horizon.
- Endpoint metric: the one frozen eval loss after exactly `S` inner steps per
  learner. No early-stop or best-checkpoint selection.

### Arm S: memoryless SGD outer control

The outer direction is the merged pseudo-gradient itself: repository
Nesterov with `outer_momentum=0`, no bias correction, no age controller, and no
other outer state. The four grid centers are banked 135M SGD optima used only
for placement, not as v18 scientific outcomes.

### Arm A: raw FedAdam outer

The required future implementation is coordinatewise FedAdam/FedOpt in the
Reddi et al. convention:

```text
m <- 0.9*m + 0.1*delta
v <- 0.99*v + 0.01*(delta*delta)
u[j] <- 0 if v[j] == 0 else m[j]/sqrt(v[j])
theta <- theta - eta*u
```

Both `m` and `v` are all-zero before a fragment's first call, persist for that
fragment without restart through its `T` calls, and are disjoint across
fragments. There is no first- or second-moment bias correction, no Nesterov
lookahead, no global norm match, no nonzero moment initializer, no server
weight decay, and no epsilon in the primary arm. The sign of `delta` may be
adapted once to the repository convention, but the recorded recurrence and
parameter displacement must be algebraically equivalent to the display.

The current block-RMS/Yogi paths are not substitutes for this arm. Before a
future launch, the new FedAdam path must have CPU unit tests for the first five
constant-input calls, zero-coordinate behavior, per-fragment state separation,
state persistence, and an off-path proof that Arm S is bit-identical to the
existing `mu=0` SGD path. Implementation, tests, frozen analyzer, and manifest
builder may be added after this registration, but they may not change any
scientific constant, grid, band, seed, estimator, or verdict rule here.

## 3. Frozen four-eta grids

Every grid uses the same four log2 offsets from its geometric center:

`{-1, -1/3, +1/3, +1}`.

For each `T`, the Arm A center is exactly the Arm S placement center times the
registered `D_adam(T)`. The even four-rung grid does not contain its geometric
center; "centered" means symmetric in geometric mean.

| `T` | `H` | arm | geometric center | four eta values, ascending |
|---:|---:|:---|---:|:---|
| 2 | 1280 | SGD | 0.071064626670 | `{0.035532313335, 0.056404031567, 0.089535819044, 0.142129253340}` |
| 2 | 1280 | FedAdam | 0.052762627680 | `{0.026381313840, 0.041877725342, 0.066476745262, 0.105525255360}` |
| 5 | 512 | SGD | 0.043419181140 | `{0.021709590570, 0.034461826909, 0.054704740288, 0.086838362281}` |
| 5 | 512 | FedAdam | 0.023472494576 | `{0.011736247288, 0.018630131291, 0.029573490010, 0.046944989153}` |
| 20 | 128 | SGD | 0.021926218662 | `{0.010963109331, 0.017402851285, 0.027625304437, 0.043852437324}` |
| 20 | 128 | FedAdam | 0.010651396986 | `{0.005325698493, 0.008454019390, 0.013419919274, 0.021302793972}` |
| 40 | 64 | SGD | 0.019635295360 | `{0.009817647680, 0.015584544255, 0.024738921945, 0.039270590720}` |
| 40 | 64 | FedAdam | 0.011466665183 | `{0.005733332591, 0.009101098187, 0.014447092836, 0.022933330365}` |

Placement provenance is frozen: the SGD centers for `T={2,5,20}` are the
banked v6 135M `S=2560` accepted optima; the `T=40` center is the accepted v2
135M `H=64,S=2560` optimum. Reusing those locations spends no v18 outcome.
There is no pilot, recenter, fifth rung, or outcome-triggered extension.

## 4. Prospective gatesim from banked noise

The gatesim is CPU-only feasibility evidence, not evidence that FedAdam obeys
the prediction. It used these immutable prior readouts:

| role | source | bytes | SHA-256 |
|:---|:---|---:|:---|
| `T={2,5,20}` curvature and paired seed residual profiles | `/private/tmp/g6-readout-tonight85.json` | 370022 | `7e7e9cd8901520cc8f0ca822151af92c4c6183ee99bf02199bc1452a2763af8c` |
| `T=40` / `H=64` curvature and paired seed residual profiles | `/private/tmp/number-audit.HmdmkU/readouts/g1v2-readout.json` | 127859 | `5d4eed9685f25fd1db3135319908a045a389300a008a67bc009cd178cabe2fc8` |

For each of 2,000 synthetic experiments (RNG seed `2026072818`), the simulator
placed the true vertices at the eight registered centers, retained each donor
curve's fitted curvature, sampled three paired banked seed residual profiles,
linearly interpolated residuals by relative log2-eta position, fit the exact
registered four-rung quadratic estimator, and evaluated all `3^3=27` unique
ordered three-seed bootstrap draws per curve. Arm A borrows the corresponding
raw-momentum residual profile only as a noise/curvature surrogate; its prior
optimum and D value are never used.

The simulated evaluability rule required every pooled curve to be accepted
within the registered 0.5-bit near-bracket allowance and at least 21 of 27
unique bootstrap draws to be valid, the exact finite analogue of at least
7,500 of 10,000 draws. The simulation then applied the v18 `>=3 of 4` hit rule.

| quantity | result |
|:---|---:|
| synthetic experiments | 2000 |
| evaluable | 2000 |
| `SHAPE_CONFIRMED` | 2000 |
| `SHAPE_WRONG` | 0 |
| four band hits | 1986 |
| exactly three band hits | 14 |
| minimum valid unique bootstrap draws seen | 27/27 |
| `P(evaluable)` | 1.0000 |
| `P(SHAPE_CONFIRMED | evaluable, registered truth)` | 1.0000 |
| Wilson 95% interval for either 2000/2000 frequency | `[0.9981,1.0000]` |

The no-floor and normalized `q=0.001` placements give the same simulated
counts because the simulator transports residuals by relative log2 offsets;
the primary no-floor values remain the only gate target. The gatesim supports
the adequacy of four eta levels and three seeds under transported banked noise.
It does not license a claim of scientific power against every nearby wrong
shape, and it cannot turn a future `NOT_EVALUABLE` result into a confirmation.

## 5. Frozen estimator

For each of the eight `(T, arm)` curves:

1. Compute the arithmetic mean endpoint loss across the three paired training
   seeds at each of the four registered eta values.
2. Fit ordinary least squares
   `loss = a*(log2 eta)^2 + b*log2 eta + c` using all four means.
3. Accept the fitted optimum only if `a>0`, all inputs are finite, and the
   vertex lies between the lowest and highest registered log2 eta or no more
   than 0.5 bits beyond either edge. Label the latter case `NEAR_BRACKETED`.
   No new cell may be added because of that label.
4. Set `eta_star=2^(-b/(2a))` for accepted fits.
5. Form
   `D_obs(T)=eta_star_FedAdam(T)/eta_star_SGD(T)`.

The analyzer must additionally run 10,000 paired training-seed bootstrap
resamples, shared across all arms and horizons, with RNG seed `2026072818`.
Each resample refits all eight curves. At least 7,500 joint resamples must have
all eight accepted fits; otherwise the whole program is `NOT_EVALUABLE`.
Percentile 95% intervals for every `eta_star`, `log2 D`, and prediction error
must be reported, but the registered band hit is determined by the pooled
three-seed point estimate, as gatesimmed.

For each `T`, define one and only one hit bit:

```text
hit(T) := abs(log2(D_obs(T) / D_pred(T))) <= 0.35
```

Equality at either boundary is a hit. No rounding is applied before the test.

## 6. Closed verdict vocabulary and precedence

The only allowed top-level scientific verdict strings are:

- `SHAPE_CONFIRMED`
- `SHAPE_WRONG`
- `NOT_EVALUABLE`

They are assigned in this order:

1. **`NOT_EVALUABLE`** if any required cell is absent; any artifact, command,
   seed, model, data, work, age, optimizer-state, or registration-order check
   fails; any required endpoint is missing or nonfinite; any of the eight
   pooled fits is unaccepted; fewer than 7,500 joint bootstrap refits are
   valid; the FedAdam trace does not prove zero initialization and the exact
   uncorrected recurrence; an eta grid differs; or the pushed registration
   commit cannot be verified. This verdict takes precedence and no partial
   hit count may be promoted.
2. Otherwise, **`SHAPE_CONFIRMED`** iff `sum_T hit(T) >= 3` for
   `T={2,5,20,40}`.
3. Otherwise, **`SHAPE_WRONG`**.

Finite high losses and finite divergences remain scientific observations; they
are not infrastructure retry reasons. If they prevent an accepted optimum on
the fixed grid, the result is `NOT_EVALUABLE`, not a license to recenter.
`SHAPE_WRONG` is reserved for a complete, valid, evaluable experiment whose
point-estimate curve misses at least two registered bands.

No synonyms (`PASS`, `FAIL`, `PARTIAL`, `TREND`, `INCONCLUSIVE`, or a
bias-corrected verdict) may appear as the top-level outcome. Descriptive
ordering, the path-matching sensitivity, a posteriori vector projections, the
`q=0.001` curve, bias-corrected Adam, alternative bands, and per-seed fits must
be reported if computed but cannot change the verdict.

## 7. Integrity, retry, and advance-registration rules

- This contract and its numeric curve must be committed and pushed before any
  v18 run. A future implementation commit does not erase or replace this
  earlier prediction commit; the manifest must bind both.
- The launch manifest must record the prediction-commit SHA, raw SHA-256 of
  this file and the derivation, implementation/test/analyzer hashes, all 96
  commands, exact model/data hashes, and proof that no v18 result directory
  predates the prediction commit's remote visibility.
- Scheduling and retry decisions are loss-blind. One retry is allowed only for
  host/GPU failure, framework/driver failure, storage/network failure, or a
  registered process timeout without a valid endpoint. The retry unit is all
  four eta cells sharing `(T, arm, training_seed)`. A finite endpoint, bad
  loss, edge optimum, wrong shape, or surprising optimizer trace is not
  retryable.
- Attempts and failures are append-only. A retry never overwrites attempt 1.
- No GPU process is authorized by this document now. This deliverable is
  math, Lean, gatesim, and prospective specification only.

The mathematical assumptions are deliberately falsifiable. Rotation and
non-constant pseudo-gradients, coordinate sparsity, curvature, and the
directional difference between Adam and SGD can make the 135M tuned-rate ratio
depart from the scalar terminal law. A valid departure is `SHAPE_WRONG`; those
effects may be analyzed after the frozen verdict, not used to rewrite it.
