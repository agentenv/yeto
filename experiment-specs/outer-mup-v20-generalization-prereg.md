# Outer-muP V20: optimizer-reset generalization hop

**Program ID:** `outer-mup-v20-generalization`

**Status:** `REGISTERED_PRE_OUTCOME`; prospective specification and CPU-only
gatesim only. This document authorizes no V20 training, input preparation,
implementation, manifest, result directory, launch authority, GPU process, or
outcome read.

**Registration date:** 2026-07-29.

**Registration object:** the Git commit containing this file. It is a valid
registration only after that exact commit is reachable from the configured
GitHub remote. The external note `/private/tmp/h200-v20-note.md` must contain
`V20 REGISTERED sha=<40-hex pushed commit>` before any V20 artifact other than
this specification may be created. A local commit, an unpushed commit, a note
without a remotely reachable commit, or a commit made after any V20 training
started is not a registration.

Immediately before this file was created, a local pathname scan, process-table
scan, all-reachable-Git-history scan, and GitHub-head scan found no V20 result,
manifest, implementation, launch authority, result root, attempt directory, or
training process. The only process-table matches were the active Codex process
whose command line contained the text of the registration request and the
read-only audit commands themselves. No claim is made about an unreachable
machine. Any later launcher must prove from append-only timestamps and hashes
that the pushed registration preceded every V20 input, implementation,
manifest, authority, result, attempt, and training process.

## 1. Prospective claim, transfer bridge, and estimand

The registered claim is deliberately a generalization hop:

> The finite-age law family fitted only to the already-banked outer-loop data
> predicts the learning-rate re-warmup curve of ordinary, single-worker,
> full-parameter 135M AdamW after its optimizer state is reset at a mature
> checkpoint.

No V20 observation, pilot, trunk checkpoint, local line search, gradient norm,
or target-domain fit supplied any candidate formula, parameter, clock, grid,
band, or decision threshold below.

The banked outer-loop response was

\[
r(T)=\frac{\eta^*_{\rm raw}(T)/\eta^*_{\mu0}(T)}{1-\mu},
\qquad \mu=0.9,
\]

which is normalized so that mature finite-age behavior has `r(T) -> 1`. The
target-domain paired response is

\[
D_{\rm obs}(k)
=\frac{\eta^*_{\rm RESET}(k)}{\eta^*_{\rm PERSIST}(k)}.
\]

`RESET` and `PERSIST` start from byte-identical model weights at step 10,240,
see the same continuation examples in the same order, and differ only in the
AdamW state described in section 2. The cross-domain bridge being tested is

\[
\boxed{D_{\rm obs}(k)=r_j(T_{\rm clock}(k))}
\]

for at least one frozen candidate `j`. Mapping `r`, rather than the unnormalized
outer ratio `(1-mu)r`, is mandatory: a reset AdamW state and its persistent
control become the same optimizer as reset age grows, so their no-ratchet ratio
has limiting value one. The factor `1-mu` is an outer-convention normalization,
not an AdamW coefficient.

The scientific unit is one independently trained seed trunk. The LR branches
within a trunk are paired repeated measures, not additional independent seeds.
The primary scientific object is the four-point ratio curve `D_obs`; absolute
LR optima for both arms are mandatory outputs but do not silently introduce a
target-domain scale fit.

## 2. Fixed ordinary single-worker AdamW experiment

### 2.1 Model, data, seeds, and mature trunks

- Model: `HuggingFaceTB/SmolLM2-135M`, revision
  `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, full-parameter tuning on one
  worker and one GPU. There is no learner federation, fragmenting, syncer,
  outer optimizer, parameter averaging, pseudo-gradient, or outer LR.
- Source data: pinned `trl-lib/Capybara`, revision
  `e235e846458bff3398a88aed812347f7f0756520`, source-parquet SHA-256
  `970f88b3f2fa6758f3b5f94052f4e91b872541a2ba530223b44a779168c51409`.
  A future builder must create one no-wrap stream of at least 10,280 training
  sequences per seed and record the exact prepared-file SHA-256 before launch.
- Held-out endpoint stream: the fixed 1,024-row
  `confirmation-audit.jsonl`, SHA-256
  `d71b90040a57731f25c78a2d191017ce90a12c1bb79f55a1cd2f3d085a706d7b`.
  It is disjoint from every training stream and from the development endpoint
  stream used by the outer-loop bank. It is never used for training, early
  stopping, trunk selection, grid placement, retry selection, or refitting.
- Shuffle seeds: `{1301,1303,1307}`. Training seeds:
  `{13011301,13031303,13071307}`, respectively. These seeds are absent from the
  committed law-v2 bank. The same per-seed CPU, CUDA, dropout, data-loader, and
  sampler RNG snapshot is restored for every branch from that seed's trunk.
- Sequence length 128, microbatch 1, gradient accumulation 1, causal-LM token
  NLL objective, gradient-norm clipping at 1.0, native bf16 single-GPU full
  parameters, and no checkpoint selection.
- Optimizer: PyTorch 2.8.0 `torch.optim.AdamW` with explicit
  `betas=(0.9,0.999)`, `eps=1e-8`, `weight_decay=0.01`, `amsgrad=False`,
  `maximize=False`, `capturable=False`, and `differentiable=False`.
  `foreach` and `fused` must be recorded at their PyTorch-2.8.0 resolved
  values and held identical in both arms; a future implementation may instead
  set both explicitly only if a prelaunch equivalence test proves bit identity
  to the recorded ordinary path.
- Each seed is trained once for exactly `S1=10240` optimizer steps. The trunk
  LR is `0.001 * n/10` on optimizer step `n=1,...,10` and exactly `0.001`
  thereafter. At step 10,240, model bytes, AdamW state, data position, and all
  RNG states are checkpointed. Since `0.999^10240 =
  3.5530345780919746e-05` and `0.9^10240` is negligible, the persistent
  control is operationally mature without using any V20 outcome to declare it
  so.

Model bytes, tokenizer bytes, prepared data, held-out data, package lock,
PyTorch/CUDA build, command, environment, and each trunk checkpoint must be
hashed in the future manifest. A trunk is a required ancestor, not a pilot:
its loss cannot move a grid, remove a seed, or change a candidate.

### 2.2 The two paired state arms

Every `(seed,k,eta)` branch begins from the same saved step-10,240 model
weights, data cursor, and RNG snapshot.

1. **`PERSIST`:** retain `exp_avg`, `exp_avg_sq`, and the scalar/tensor AdamW
   `step=10240` exactly. Change only the param-group LR to the registered rung.
2. **`RESET`:** retain model weights exactly, then set every trainable
   parameter's `exp_avg` and `exp_avg_sq` tensor to exact zeros and set its
   AdamW `step` to exact zero. No parameter, gradient, data cursor, RNG state,
   weight-decay setting, or scheduler state is reset. `amsgrad=False`, so no
   `max_exp_avg_sq` exists. The first continuation update must use Adam bias
   corrections at age one.

The pretraining warmup scheduler is not restarted. During the continuation,
the registered rung is a constant LR and no scheduler step is taken. This
prevents an LR-scheduler warmup from being mislabeled optimizer-state
re-warmup.

For each `k in {2,5,20,40}`, each branch performs exactly the first `k`
post-trunk training steps at its constant registered LR and then evaluates one
finite per-token NLL on the held-out stream. Thus "the LR at reset+k" means a
constant continuation LR applied on all steps `10241,...,10240+k`, with the
endpoint immediately after step `10240+k`; it does not mean a burn-in followed
by a one-step probe. Branches at different `k` restart from the trunk and are
not nested checkpoints selected from a longer outcome-bearing branch.

The complete future design contains two state arms, four ages, five rungs, and
three seed trunks: `2 * 4 * 5 * 3 = 120` required scientific endpoints, plus
the three shared trunks. No pilot, sixth rung, regrid, recenter, extra seed,
outcome-triggered horizon, or alternative reset is permitted.

## 3. Adam-to-outer clock mapping, derived before outcomes

Outer law age `T` counts calls into one zero-started outer state. On a constant
coordinate, a zero-started Adam state after `k` ordinary optimizer calls has

\[
m_k=(1-\beta_1^k)g,\qquad
v_k=(1-\beta_2^k)g^2.
\]

The outer momentum deficit is exponential in `mu^T`. Matching fractional
relaxation, rather than raw step labels, gives the only two optimizer-state
clock mappings registered here:

\[
\mu^{T_i(k)}=\beta_i^k
\quad\Longrightarrow\quad
\boxed{T_i(k)=k\frac{\log\beta_i}{\log\mu}}.
\]

The second moment enters an Adam direction through a square root, but
`sqrt(1-beta2^k) = 1 - beta2^k/2 + o(beta2^k)`; the factor one-half changes
amplitude, not the exponential e-fold clock. There is no unique
amplitude-matched nonlinear age, so none may be selected after seeing data.

The registered mappings are:

| target `k` | `B1_UPDATE`: `T1=k log(.9)/log(.9)` | `B2_RELAX`: `T2=k log(.999)/log(.9)` |
|---:|---:|---:|
| 2 | **2** | **0.01899194071587132** |
| 5 | **5** | **0.0474798517896783** |
| 20 | **20** | **0.1899194071587132** |
| 40 | **40** | **0.3798388143174264** |

The shared second-moment scale factor is
`log(0.999)/log(0.9) = 0.00949597035793566`.

`B1_UPDATE` is the **primary clock** because the banked law is a momentum-age
law, AdamW `beta1` exactly equals the bank's `mu=0.9`, and one target optimizer
step is one call into the first-moment state. Update-count and first-moment
matching are therefore the same mapping, not two researcher degrees of
freedom. `B2_RELAX` is a mandatory, separately sealed sensitivity because
ordinary AdamW also has a much slower adaptive-preconditioner state. It
extrapolates the outer formulas below the bank's minimum observed `T=2`; that
out-of-support fact must accompany every B2 result. A B2-only hit cannot
promote the primary family verdict.

PyTorch bias correction makes the constant-gradient, zero-epsilon Adam
direction age-invariant. Accordingly, `D(k)=1` at all four ages is a mandatory
flat null diagnostic. It is not a fitted law-v2 candidate and cannot receive
or trigger `LAW_TRANSFERS`.

For C1+C3, one ordinary AdamW gradient/update is both one state call and all
the work between calls, so the target analogue is fixed at `H_A=1`. The mature
10,240-step trunk is an initial condition, not a hidden `H=10240/k`. Importing
`H=512`, fitting an effective H, or treating pre-reset work as local work is
forbidden.

## 4. Immutable provenance and complete parameter ledger

All candidate parameters predate V20 and are reachable in Git before this
registration. SHA-256, not a mutable pathname alone, is authoritative.

| frozen artifact | Git commit introducing current bytes | SHA-256 | role |
|:---|:---|:---|:---|
| `mech/law-v2/verifier/signed_align_results.json` | `c8ac48b5f41f77d5d9a5c4c649f50f34e2f6c8e0` | `03e2671cfc0a84fd47b44026c8c5ba1f94c8741478e37876879a1267ce8b647f` | signed-align fit |
| `mech/law-v2/verifier/signed_align_probe.py` | `c8ac48b5f41f77d5d9a5c4c649f50f34e2f6c8e0` | `2e9124a006a5111595cd1aa113fc48b769b97a24f1a2a4d4a40e4701c8baca96` | signed-align equation |
| `mech/law-v2/forensics/shape_fits.csv` | `25eb2e18e9f62e3a4d1f529740302e6192747342` | `314efb55f9bf31d30d02444332e6144558956810f987a06dd803b8862de9f717` | displayed sat-exp fit |
| `mech/law-v2/forensics/r_of_T_pairs.csv` | `25eb2e18e9f62e3a4d1f529740302e6192747342` | `e76c711f725db4cbf7c99ef2bf1f043200d0680c78bd0b441652feb3ee11d857` | sat-exp pair ledger |
| `mech/law-v2/forensics/forensics.py` | `25eb2e18e9f62e3a4d1f529740302e6192747342` | `267eb0584180453169c4386af39f2942ab71cb96dbe486516a6f0b0a1e658747` | 240-point tau grid and WLS equation |
| `mech/law-v2/theory/interference_fit.py` | `0dde35e4a90d52a4f5647adb9a148ae737b25c7f` | `1092e501fe8aaabde3afe6a3fde6706b278e8f20a7b9372a37a30f23d7457a4d` | C1+C3 constants and fit |
| `mech/law-v2/theory/CANDIDATES.md` | `0dde35e4a90d52a4f5647adb9a148ae737b25c7f` | `f7e2e6dd9a0048edf6e6bc2bc731da073e6cdb97482741b801ac604d786f193c` | C1/C3/C4 definitions and C4 knee |
| `mech/law-v2/zoo/fit_zoo.py` | `24e228195fb9fd284fe25f8ff682aea7bd0af0e4` | `2e2ebc6f250fe9157218c55e8c4090b72188a638122304eb5b909da19dcdf0bf` | code-true `C`, `M`, and profiling rules |
| `lean-mechanism/LeanMechanism/FiniteHorizonOuter.lean` | `1cbba1ed7b155d966df746e16a7bfc069d67c2c1` | `e820e78adb74a7a00d79cb4f2fad4baf554e7ad45d43c5ca7185f97f898b593b` | C1 accumulated-coefficient identity |
| `mech/law-unification/paired_cancellation.csv` | `c44ae3ccc16c3ca8a89d559fb27ffed7d53e88b2` | `15759bec558807f1ed3834f4465f18fdabaa1a10e678f6fd47f836ad870c3566` | C3 fit input and C4 frozen cells |
| `experiment-specs/outer-mup-v19-discrimination-prereg.md` | `1314f9eec3c392c78c5bedebe3648145328ebb95` | `0f7d4e46ffab5e112bc35c3ae471f632c161205dec0725b7f0a27db36a04eac9` | pre-V20 full-precision replay and C4 freeze |

The exact numeric parameter ledger is:

| candidate | parameter | frozen value | source bytes above |
|:---|:---|---:|:---|
| all outer candidates | `mu` | `0.9` | `fit_zoo.py`, `CANDIDATES.md` |
| signed-align | `kappa` | `-0.0914859135453169` | `signed_align_results.json` |
| signed-align | `d` | `0.9837346435938291` | `signed_align_results.json` |
| signed-align nuisance clock | `alpha` | `1.1114924647313817` | `signed_align_results.json` |
| signed-align nuisance floor | `f` | `0.1625532413326748` | `signed_align_results.json` |
| signed-align nuisance work tilt | `sigma` | `0.09430894830710623` | `signed_align_results.json` |
| signed-align corrected-only tilt | `beta_c` | `0.15023763054364514` | `signed_align_results.json` |
| signed-align profiled 135M intercept | `a_135M` | `-3.0222685237372704` log2-LR | full-precision replay in V19 preregistration |
| sat-exp | `A` | `6.708816513977305` | full-precision replay in V19 from `shape_fits.csv`, `r_of_T_pairs.csv`, `forensics.py` |
| sat-exp | `tau` | `6.739447073713096` | same three sat-exp artifacts and V19 replay |
| C1 | fitted transform parameters | **none** | `fit_zoo.py`, Lean file |
| C1+C3 | `eta0_135M` | `0.031350405899186334` | `interference_fit.py` |
| C1+C3 | `q` | `0.9875756970034051` | `interference_fit.py` |
| C1+C3 | `beta` | `0.005949532405537653` | 62-pair full-precision replay in V19 from `interference_fit.py` and `paired_cancellation.csv` |
| C4 | `T_knee=2/(1-mu)` | `20` | `CANDIDATES.md` |
| C4 | `phi20_frozen` | `0.6164021201189884` | full-precision V19 geometric pool from `paired_cancellation.csv` via `interference_fit.py` residual convention |

In the signed-align paired ratio, `a_135M`, `alpha`, `f`, and `sigma` cancel
exactly because RESET and PERSIST are evaluated at the same target age, work,
and scale; `beta_c` is zero because the transferred source curve is the raw,
not corrected, stratum. They are listed to make the nuisance cancellation
auditable rather than silently omitted. Only `kappa` and `d` remain in its
registered ratio prediction.

The source CSV displays sat-exp parameters to six decimals. V20 uses the
already-committed V19 full-precision replay above, not a future choice between
rounded and unrounded values. No parameter may be re-estimated from V20,
including an intercept, vertical renormalization, horizontal age rescaling,
effective H, C4 smoothing constant, or clock mixture.

## 5. Frozen candidate equations

For positive real `T`, extend the committed integer-age formulas analytically:

\[
C(T)=\frac{T}{1-\mu}
-\frac{\mu^2(1-\mu^T)}{(1-\mu)^2},\qquad
M(T)=\frac{1-\mu^{T+1}}{1-\mu}.
\]

This real-age extension is needed only for the explicitly out-of-support B2
sensitivity. The five eligible candidate curves are:

1. **`signed-align`**

   \[
   D_{\rm SA}(T)
   =\frac{T/C(T)}{1-\mu}\,
     2^{\kappa d^T(M(T)-1)}.
   \]

2. **`sat-exp`**

   \[
   D_{\rm SE}(T)
   =\exp\!\left[\log(A)\exp(-T/\tau)\right].
   \]

3. **`C1`**

   \[
   D_{\rm C1}(T)=\frac{T/C(T)}{1-\mu}.
   \]

4. **`C1+C3`**, with the target-domain work analogue sealed as `H_A=1`

   \[
   D_{\rm C1+C3}(T)
   =D_{\rm C1}(T)\exp\!\left[
   -\beta\sqrt{1}\,\eta_{0,135M}q^T(C(T)-T)\right].
   \]

5. **`C4`** (`frozen-knee` operationalization)

   The committed C4 text proposed a knee near `2/(1-mu)` but did not identify
   its two-parameter curvature-growth closure. Fitting those missing parameters
   now would violate zero refitting. V20 therefore uses the operational C4
   prediction already frozen by V19: no extra deficit before the theoretical
   knee, then the geometrically pooled T=20/H=512 raw deficit carried forward
   without release:

   \[
   D_{\rm C4}(T)=D_{\rm C1}(T)
   \begin{cases}
   1,&T<20,\\
   0.6164021201189884,&T\ge 20.
   \end{cases}
   \]

   This hard-knee closure is intentionally testable and intentionally not
   smoothed. A post-outcome sigmoid, interpolation, C4 gamma fit, or choice of a
   constituent T=20 cell is ineligible.

The standalone, all-62-pair C1+C3 fit operationalized at full precision by the
pre-V20 V19 registration is authoritative. The zoo's alternative
`C1q+C3`/`C1sat+C3` joint fits are not eligible substitutes. Likewise, the
only eligible C4 curve is the frozen-knee equation above; `C4` is the canonical
candidate name in the outcome file, and `frozen-knee` is a mandatory method
qualifier, not a sixth candidate.

## 6. Sealed predictions for every candidate and clock

### 6.1 Scientific ratio predictions

Primary `B1_UPDATE` predictions, in target order `k={2,5,20,40}`:

| candidate | `D_pred(2)` | `D_pred(5)` | `D_pred(20)` | `D_pred(40)` |
|:---|---:|---:|---:|---:|
| `signed-align` | **3.90620639018876** | **2.39542890675678** | **1.08170452982685** | **0.933079509483444** |
| `sat-exp` | **4.11511121543085** | **2.47545688082606** | **1.10283736250253** | **1.00504663616333** |
| `C1` | **4.33839479392625** | **2.97094004702404** | **1.55222007064251** | **1.24922996091737** |
| `C1+C3` | **4.33633542919211** | **2.96478834621305** | **1.52787126147047** | **1.21025519774771** |
| `C4` (`frozen-knee`) | **4.33839479392625** | **2.97094004702404** | **0.956791742435289** | **0.770027996425627** |

Mandatory secondary `B2_RELAX` predictions:

| candidate | `D_pred(2)` | `D_pred(5)` | `D_pred(20)` | `D_pred(40)` |
|:---|---:|---:|---:|---:|
| `signed-align` | **6.77500595820464** | **6.70534885085433** | **6.37676693667302** | **5.98414153324768** |
| `sat-exp` | **6.67297784879688** | **6.61976549938746** | **6.36320598755802** | **6.04427636883478** |
| `C1` | **6.7827373395747** | **6.72445727693066** | **6.44904880192505** | **6.11878191172377** |
| `C1+C3` | **6.78272594551781** | **6.72442828615119** | **6.4489233132196** | **6.11850824543454** |
| `C4` (`frozen-knee`) | **6.7827373395747** | **6.72445727693066** | **6.44904880192505** | **6.11878191172377** |

C4 equals C1 under B2 because all four mapped ages are prospectively below
the frozen knee. This equality is a prediction, not a transcription error.

### 6.2 Optimal continuation-LR curves and scale convention

The exact candidate prediction for the RESET LR is

\[
\eta^*_{{\rm RESET},j}(k)
=D_j(T_{\rm clock}(k))\,\eta^*_{\rm PERSIST}(k).
\]

That conditional equation is the scale-free scientific prediction. For an
auditable absolute placement curve, the mature pretraining LR is frozen at
`eta_ref=0.001`; multiplying every ratio above by `0.001` gives:

| primary candidate | `eta_RESET,pred(2)` | `eta_RESET,pred(5)` | `eta_RESET,pred(20)` | `eta_RESET,pred(40)` |
|:---|---:|---:|---:|---:|
| `signed-align` | `0.00390620639018876` | `0.00239542890675678` | `0.00108170452982685` | `0.000933079509483444` |
| `sat-exp` | `0.00411511121543085` | `0.00247545688082606` | `0.00110283736250253` | `0.00100504663616333` |
| `C1` | `0.00433839479392625` | `0.00297094004702404` | `0.00155222007064251` | `0.00124922996091737` |
| `C1+C3` | `0.00433633542919211` | `0.00296478834621305` | `0.00152787126147047` | `0.00121025519774771` |
| `C4` (`frozen-knee`) | `0.00433839479392625` | `0.00297094004702404` | `0.000956791742435289` | `0.000770027996425627` |

| B2 sensitivity candidate | `eta_RESET,pred(2)` | `eta_RESET,pred(5)` | `eta_RESET,pred(20)` | `eta_RESET,pred(40)` |
|:---|---:|---:|---:|---:|
| `signed-align` | `0.00677500595820464` | `0.00670534885085433` | `0.00637676693667302` | `0.00598414153324768` |
| `sat-exp` | `0.00667297784879688` | `0.00661976549938747` | `0.00636320598755802` | `0.00604427636883478` |
| `C1` | `0.0067827373395747` | `0.00672445727693066` | `0.00644904880192505` | `0.00611878191172377` |
| `C1+C3` | `0.00678272594551781` | `0.00672442828615119` | `0.0064489233132196` | `0.00611850824543454` |
| `C4` (`frozen-knee`) | `0.0067827373395747` | `0.00672445727693066` | `0.00644904880192505` | `0.00611878191172377` |

The absolute `0.001`-anchored table is a mandatory calibration diagnostic. The
closed transfer verdict uses `D_obs`, so a common target-domain LR-scale shift
cannot masquerade as failure of the finite-age shape and no V20 intercept is
fit. For every candidate and clock, the report must nevertheless emit the
four-bit absolute-curve diagnostic

```text
absolute_hit(j,c,k) :=
    abs(log2(eta_star_RESET(k) / (0.001 * D_pred(j,c,k)))) <= 0.35
```

and the corresponding PERSIST-anchor errors
`log2(eta_star_PERSIST(k)/0.001)`. Thus a ratio transfer accompanied by a poor
absolute LR prediction remains visible and cannot be presented as an
unqualified absolute-rate hit.

### 6.3 Registered band

For every candidate, clock, and age, the inclusive registered band is

```text
abs(log2(D_obs(k) / D_pred(candidate,clock,k))) <= 0.35
```

Equivalently, `D_obs` must lie in
`[0.7845840978967507 * D_pred, 1.2745606273192622 * D_pred]`. The `0.35`-bit
width is frozen from the pre-V20 finite-age precedent in
`experiment-specs/outer-mup-v18-fedadam-prereg.md`, SHA-256
`eb1c3f17f6566a8f4341cea98f60989b97d1294b1bb6bdbe4e391681b977cc11`.
It is not estimated from V20. Point estimates, not rounded displays or a
bootstrap median, determine hits.

## 7. Prospectively fixed five-rung grids

Every PERSIST curve uses the same five one-octave-spaced LRs centered on the
mature reference:

```text
{0.00025, 0.0005, 0.001, 0.002, 0.004}
```

For RESET, each age's center was fixed to `0.001 * sqrt(D_min(k)*D_max(k))`,
where min and max range over all ten frozen candidate-by-clock predictions in
section 6. The five offsets are exactly `{-2,-1,0,+1,+2}` bits:

| `k` | frozen geometric center | five RESET LRs, ascending |
|---:|---:|:---|
| 2 | `0.005147307251254637` | `{0.0012868268128136593, 0.0025736536256273186, 0.005147307251254637, 0.010294614502509274, 0.02058922900501855}` |
| 5 | `0.004013472230302672` | `{0.001003368057575668, 0.002006736115151336, 0.004013472230302672, 0.008026944460605343, 0.016053888921210686}` |
| 20 | `0.0024840283090665624` | `{0.0006210070772666406, 0.0012420141545332812, 0.0024840283090665624, 0.004968056618133125, 0.00993611323626625}` |
| 40 | `0.0021706297187797886` | `{0.0005426574296949472, 0.0010853148593898943, 0.0021706297187797886, 0.004341259437559577, 0.008682518875119154}` |

At the `0.001` placement anchor, the nearest registered candidate vertex is at
least `1.60`, `1.26`, `0.62`, and `0.50` bits inside the RESET grid at
`k=2,5,20,40`, respectively. The grids test both clock choices without
candidate-specific data collection. A finite boundary minimum, nonpositive
curvature, surprising divergence, or candidate miss cannot add, delete, or
move a rung.

## 8. Prospective gatesim from banked 135M seed noise

The gatesim is CPU-only feasibility evidence. It is not a V20 training run,
does not use a V20 trunk or endpoint, and supplies no evidence that any
candidate transfers.

The sole donor is the committed G6 135M readout
`mech/law-unification/sources/g6-readout.json`, introduced at
`c44ae3ccc16c3ca8a89d559fb27ffed7d53e88b2`, 370,022 bytes, SHA-256
`7e7e9cd8901520cc8f0ca822151af92c4c6183ee99bf02199bc1452a2763af8c`.
It contributes only the three banked seed residual profiles `{601,607,613}`
and fitted log2-LR curvature:

| target age | PERSIST donor / curvature `a` | RESET donor / curvature `a` |
|---:|:---|:---|
| 2 | G6 `T=2,S=10240,mu0` / `0.03633699218588449` | G6 `T=2,S=10240,raw` / `0.03277457313050147` |
| 5 | G6 `T=5,S=10240,mu0` / `0.03485242217844575` | G6 `T=5,S=10240,raw` / `0.03550624085374759` |
| 20 | G6 `T=20,S=10240,mu0` / `0.017384641505974296` | G6 `T=20,S=10240,raw` / `0.035986270481111665` |
| 40 | same T=20 mu0 donor / `0.017384641505974296` | same T=20 raw donor / `0.035986270481111665` |

No committed T=40 donor with full three-seed profiles exists, so reusing T=20
at k=40 is disclosed prospectively rather than substituting aggregate T=40
intervals. For each donor, the simulator subtracts the committed pooled mean
loss at each donor LR from each seed loss, expresses that five-value residual
profile against log2 LR relative to the donor vertex, and linearly
interpolates/extrapolates between the two adjacent residual values.

For each of the ten exact candidate-by-clock truths and 2,000 synthetic
experiments, CPython 3.14.6 uses one continuous
`random.Random(2026072920)` stream in clock order B1 then B2 and candidate
order `signed-align`, `sat-exp`, `C1`, `C1+C3`, `C4`:

1. draws one independent `Normal(0,0.25 bits)` common placement shift per age
   and applies it to both state arms, so the true ratio remains the registered
   candidate;
2. samples three paired donor-profile indices with replacement, using the same
   ordered indices for all ages and arms;
3. adds the translated seed residuals to a quadratic with the donor curvature
   and the shifted true vertex on the exact five-rung target grid;
4. applies the pooled estimator in section 9; and
5. enumerates all `3^3=27` ordered three-seed resamples. A synthetic experiment
   is evaluable only if all eight pooled curves are strict-interior and at least
   21 of 27 joint resamples accept all eight curves.

The frozen results are:

| clock | generating candidate | evaluable / 2000 | `P_eval` | `LAW_TRANSFERS` among evaluable |
|:---|:---|---:|---:|---:|
| B1 | `signed-align` | 1999 | 0.9995 | 1999/1999 |
| B1 | `sat-exp` | 1995 | 0.9975 | 1995/1995 |
| B1 | `C1` | 2000 | 1.0000 | 2000/2000 |
| B1 | `C1+C3` | 2000 | 1.0000 | 2000/2000 |
| B1 | `C4` (`frozen-knee`) | 1947 | 0.9735 | 1947/1947 |
| B2 | `signed-align` | 1946 | 0.9730 | 1946/1946 |
| B2 | `sat-exp` | 1936 | 0.9680 | 1936/1936 |
| B2 | `C1` | 1936 | 0.9680 | 1936/1936 |
| B2 | `C1+C3` | 1936 | 0.9680 | 1936/1936 |
| B2 | `C4` (`frozen-knee`) | 1927 | 0.9635 | 1927/1927 |

The worst-case evaluability frequency is `1927/2000=0.9635`, Wilson 95%
interval `[0.954353,0.970870]`. Every evaluable draw at every registered truth
was `LAW_TRANSFERS`; no evaluable draw was `PARTIAL` or `NO_TRANSFER` for its
generating curve.

This only shows that three seeds and five rungs can recover the registered
truths under transported G6 curvature/noise and the declared placement stress.
The future two-step ordinary-AdamW loss curve may be flatter than an outer-loop
donor, and T=40 reuses T=20 noise. Those are real extrapolation risks. The
gatesim cannot turn a future invalid curve into a transfer, authorize a regrid,
or be cited as target-domain evidence.

## 9. Frozen estimator and uncertainty

For each of the eight `(state_arm,k)` curves independently:

1. Require all 15 endpoints: the exact five LRs for each of the three seeds,
   each finite and backed by a valid trunk and state trace.
2. At each exact LR, take the arithmetic mean held-out per-token NLL across the
   three seeds.
3. With `x=log2(eta)`, fit ordinary least squares
   `loss = a*x^2 + b*x + c` to all five rung means with equal rung weight.
4. Accept the pooled curve iff `a>0` and the unrounded vertex `-b/(2a)` lies
   strictly between the lowest and highest registered `x`. There is no
   near-bracket allowance, rung deletion, robust replacement, or fit-window
   choice.
5. For an accepted curve, set `eta_star=2^(-b/(2a))` and form
   `D_obs(k)=eta_star_RESET(k)/eta_star_PERSIST(k)` without intermediate
   rounding.

Run 10,000 paired nonparametric seed-bootstrap draws with RNG seed
`2026072920`. One common ordered three-index resample is applied to every rung,
arm, and age, and all eight curves are refit inside every draw. At least 7,500
joint draws must accept all eight curves. Report equal-tailed 95% percentile
intervals in log2 space for every `eta_star`, `D_obs`, and candidate error,
then exponentiate for display. The pooled three-seed point estimate, not a
bootstrap median, decides registered band hits.

The absolute `0.001` calibration errors and flat bias-corrected null are
mandatory outputs. Per-seed vertices, token-weighted alternative pooling,
cubic/spline fits, leave-one-rung-out fits, a shared-curvature fit,
gradient-based line searches, and any post-hoc clock mixture are
mandatory-to-label if reported and cannot change the registered result.

## 10. Closed scientific vocabulary and precedence

For each named candidate `j` and each separately labeled clock `c`, define

```text
hit(j,c,k) := abs(log2(D_obs(k) / D_pred(j,c,k))) <= 0.35
hits(j,c)  := sum over k in {2,5,20,40} of hit(j,c,k)
```

Equality at a band edge is a hit. Full-precision values decide; displayed
rounding does not.

Each candidate-by-clock classification uses exactly the requested vocabulary:

- **`LAW_TRANSFERS`** iff `hits(j,c) >= 3`;
- **`PARTIAL`** iff `hits(j,c) == 2`;
- **`NO_TRANSFER`** iff `hits(j,c) <= 1`.

The one top-level V20 family verdict is based only on the primary
`B1_UPDATE` clock:

1. **`LAW_TRANSFERS`** if at least one named B1 candidate has at least three
   hits. The report must include a lexicographically sorted
   `transferring_candidates` list; multiple qualifying candidates are not
   forced into a false unique winner.
2. Otherwise **`PARTIAL`** if at least one named B1 candidate has exactly two
   hits.
3. Otherwise **`NO_TRANSFER`**.

All five B2 classifications must be reported beside the primary result, but a
B2-only transfer is described as `second-moment-clock-only` and cannot promote
the top-level verdict. The flat `D=1` null gets a descriptive four-bit hit
vector only. Every candidate-by-clock row must also show its separately named
`absolute_hit` vector from section 6.2; it cannot rescue or veto the scale-free
classification. Heavy overlap among candidate bands is allowed: V20 tests
cross-domain law-family transfer, not unique within-family model selection.

The closed vocabulary applies only after the validity gate. If registration
ordering or remote reachability fails; any required model, data, seed, trunk,
state, command, grid, endpoint, or trace is missing or mismatched; any endpoint
is nonfinite; any pooled curve is unaccepted; or fewer than 7,500 joint
bootstrap draws are valid, the analyzer must stop with an integrity error and
emit **no scientific verdict string**. A validation failure is not a fourth
outcome and must not be rewritten as `PARTIAL`, `NO_TRANSFER`, `INCONCLUSIVE`,
or a favorable result from the available ages.

No synonyms such as `PASS`, `FAIL`, `TREND`, `CONFIRMED`, `LIKELY_TRANSFER`, or
`SHAPE_WRONG` may replace the three scientific strings.

## 11. Integrity, leakage, retry, and reporting rules

- The future manifest must bind this pushed registration commit and raw file
  hash, every frozen source hash in section 4, exact prediction tables, model
  and tokenizer bytes, prepared train/eval bytes, package and CUDA builds,
  three trunk hashes, 120 commands, optimizer-state traces, and the analyzer
  hash.
- Before launch, CPU tests must prove: ordinary PERSIST parity; exact RESET
  zeros and `step=0`; weight/RNG/data-cursor identity at the fork; first five
  constant-gradient PyTorch AdamW updates; constant continuation LR with no
  scheduler restart; and off-path identity of the trunk training path.
- Held-out endpoints stay blinded until all 120 required cells have a terminal
  evidence state. Scheduling is loss-blind. Trunk training loss may be used
  only to detect nonfinite/infrastructure failure, never to alter science.
- At most one attempt-2 retry may later be authorized for a host/GPU,
  framework/driver, storage/network, or registered-timeout failure that
  produced no valid endpoint. A branch retry unit is all five LRs sharing
  `(seed,k,state_arm)`; a trunk retry repeats the whole seed trunk and
  invalidates all descendants of the failed attempt. Attempts are append-only
  and never overwrite one another.
- A finite high loss, finite divergence, flat curve, boundary vertex,
  nonpositive curvature, wide interval, clock rejection, candidate miss, or
  surprising reset response is scientific behavior, not a retry reason and
  not authority to regrid.
- No V20 result may be pooled with the outer bank, used to update a candidate,
  or used to choose B1 versus B2 before the frozen V20 verdict is emitted. Any
  later target-domain model is explicitly post-V20 and cannot retroactively
  change this registration.
- The report must show all ten candidate-by-clock hit vectors, both absolute
  LR curves, all eight fitted curvatures/vertices, bootstrap intervals and
  valid-draw count, the flat-null diagnostic, state-reset checks, and every
  missing/failed endpoint. Selective reporting of only the best candidate or
  clock is prohibited.

## 12. Scope and hostile-review disclosures

This is intentionally harder than interpolation within the outer bank.
Ordinary AdamW is bias-corrected, its second moment relaxes far more slowly
than the bank's momentum buffer, and its first two post-reset steps may not
produce the donor curvature assumed by the gatesim. B2 evaluates all
candidates below their fitted age support. C1+C3's original mechanism involved
an inner optimizer beneath an outer call, which the target lacks; V20 resolves
that mismatch prospectively with `H_A=1`. C4 had no identified numerical
growth-rate fit, so its already-frozen hard-knee operationalization is tested
without pretending otherwise. The three branches per rung are paired through
three trunks and therefore provide three, not fifteen, independent replicates.

These limitations are reasons the experiment can return `PARTIAL`,
`NO_TRANSFER`, or no verdict after a validation failure. They are not licenses to
change the clock, normalize at a favorable age, fit a target intercept, smooth
C4, widen the band, ignore k=2, add seeds, or run a second grid after outcomes.
The generalization claim survives only by the prospective rules above.
