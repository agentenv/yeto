# Outer-muP V19: advance-registered model discrimination

**Program ID:** `outer-mup-v19-discrimination`

**Status:** `REGISTERED_PRE_OUTCOME`; specification and CPU-only gatesim only.
No V19 training, pilot, implementation, manifest, launch authority, result root,
or GPU process is authorized by this document.

**Registered:** 2026-07-28.

**Registration object:** the Git commit containing this file. That commit must
be reachable on the configured public GitHub remote before any V19 manifest,
authority, result directory, attempt directory, or training process is created.
The external note `/private/tmp/h200-v19-note.md` records the pushed commit SHA.
A local-only commit or an unpushed note is not a registration.

This is the verifier's single cheapest decisive new-data experiment, fixed at
135M, `T=40`, `mu=0.9`, `H=512`, `S=20480`, with independently tuned `mu0`,
`nesterov_raw`, and `heavy_ball` arms. The coordinate and rationale are the
verifier's committed recommendation at
`mech/law-v2/verifier/VERDICT.md:349-377`. The fleet was deliberately left
offline while this document and its read-only CPU gatesim were prepared. A
local pathname/process-name check found no pre-existing V19 artifact; no claim
about unreachable fleet state is inferred from that local check.

**Pre-registration bank disclosure:** immediately before the V19 registration,
commit `fbefc07b16e53328bafaaad891c23fd3b18fee20` recorded the already-frozen,
zero-training checkpoint curvature discriminator. Its exact-target formal
verdict is `AMBIGUOUS`; directionally it rejects C4's preregistered sharpness
excess but also falls outside C3's parity band
(`mech/law-v2/discriminator/REPORT.md:1-26`). It is not a V19 tuning run, does
not measure `r` at the V19 coordinate, and changes none of the four numerical
predictions below. This disclosure preserves the chronology while the new
`T=40,H=512` rate-tuning cell tests the kinematic tail that a checkpoint probe
cannot test (`mech/law-v2/verifier/VERDICT.md:383-389`).

## 1. Fixed estimands and experimental unit

The primary estimand is

\[
r_{\rm obs}
=\frac{\eta^*_{\rm raw}/\eta^*_{\mu0}}{1-0.9}
=10\frac{\eta^*_{\rm raw}}{\eta^*_{\mu0}}.
\]

The mandatory secondary estimand is

\[
h_{\rm obs}=\frac{\eta^*_{\rm hb}}{\eta^*_{\rm raw}}.
\]

Each `eta*` is fitted independently from that arm's own six-rung curve. No eta,
loss, curvature, or fitted vertex is shared between arms. Seeds are paired
across all arms and rungs only to reduce comparison noise.

The complete future design is one coordinate, three arms, six eta values per
arm, and three fresh training seeds: `1 * 3 * 6 * 3 = 54` required scientific
cells. There is no pilot, adaptive allocation, shared-LR comparison, recenter,
seventh rung, or outcome-triggered extension.

## 2. Frozen 135M work and held-out evaluation contract

- Model: `HuggingFaceTB/SmolLM2-135M`, revision
  `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.
- Source data: pinned local `trl-lib/Capybara`, revision
  `e235e846458bff3398a88aed812347f7f0756520`, source-parquet SHA-256
  `970f88b3f2fa6758f3b5f94052f4e91b872541a2ba530223b44a779168c51409`.
  These bindings are enforced by `scripts/prepare_inputs_v3.py:114-123` and
  `scripts/prepare_inputs_v3.py:220-235`.
- Fresh shuffle seeds: `{1201,1213,1217}`. Training seeds are respectively
  `{12011201,12131213,12171217}`. A future input builder must deterministically
  prepare no-wrap 13,758-row bundles from the same frozen source/pool rule and
  record every resulting file hash before launch. None of these seed outcomes
  is in the law-v2 model bank.
- Primary evaluation stream: the fixed, disjoint 1,024-row
  `confirmation-audit.jsonl`, SHA-256
  `d71b90040a57731f25c78a2d191017ce90a12c1bb79f55a1cd2f3d085a706d7b`.
  It is disjoint from training and from the development stream on which the
  banked tuning curves were scored. The hash check is committed at
  `scripts/prepare_inputs_v3.py:118-123`. V19 consumes this stream as its one
  held-out endpoint stream; it is never used for training, early stopping,
  checkpoint selection, grid movement, or retry selection.
- Four full-parameter learners/fragments, sequence length 128, microbatch 1,
  inner AdamW learning rate `0.001`, RDA matrix merge, bf16 wire/f32 syncer,
  strict full quorum, true barrier, version-matched anchor, no delta
  correction, and zero injected delay/jitter.
- Each learner executes exactly `S=20480` inner steps. Each fragment's own
  outer state receives exactly `T=40` calls, with exactly `H=S/T=512` local
  steps per call. Round-robin four-fragment execution therefore makes exactly
  `4*T=160` global fragment commits.
- The endpoint is one finite per-token NLL evaluation on the held-out stream
  after exactly `S` steps per learner. There is no intermediate/best-checkpoint
  selection.

The future manifest must bind the model bytes, source and prepared-data hashes,
held-out stream hash, registration commit, implementation commit, exact 54
commands, and the three optimizer recurrences below. Preparation and validation
may happen only after this registration is public and cannot change any
scientific constant here.

## 3. Three independently tuned optimizer arms

Let `p_t` be the merged pseudo-gradient for one fragment and initialize every
arm's outer buffer to zero before its first call.

1. `mu0`: memoryless control, implemented by the repository Nesterov path with
   `outer_momentum=0`; `u_t=p_t`. It has no bias correction or other outer
   state.
2. `nesterov_raw`: `mu=0.9`, `b_t=0.9*b_(t-1)+p_t`,
   `u_t=p_t+0.9*b_t`. There is no bias correction, age controller, restart,
   normalization, or state reset.
3. `heavy_ball`: `mu=0.9`, `b_t=0.9*b_(t-1)+p_t`, `u_t=b_t`. There is no
   Nesterov lookahead, bias correction, age controller, restart, normalization,
   or state reset.

The current branch does not itself authorize a heavy-ball run. A later
implementation/manifest commit must provide CPU constant-input tests for the
first 40 calls, zero initialization, per-fragment state separation and
persistence, and off-path bit identity for `mu0` and raw Nesterov. The
scientific design cannot be amended to fit the implementation.

## 4. Prospectively fixed six-rung grids

### 4.1 Centering model and provenance

Grid placement uses the current best descriptive model,
`Vsigned-align-M`, not any of the four candidates being tested. Its committed
LOCO RMSE is `0.27945504035086804` bits
(`mech/law-v2/verifier/signed_align_results.json:2-15`). Its fitted correction
and clock are

```text
g = log2(T/C)
    + kappa*d^T*(M-1)
    + log2(T^(-alpha)+f)
    + sigma*log2(S/2560)
```

for these uncorrected arms. The formula is committed at
`mech/law-v2/verifier/signed_align_probe.py:45-58`; the exact fitted parameters
are at `mech/law-v2/verifier/signed_align_results.json:3-9`. Profiling the 135M
intercept exactly as specified at `mech/law-v2/zoo/fit_zoo.py:170-194` gives

```text
a_135M = -3.0222685237372704  (log2 eta units)
```

and therefore the following per-arm descriptive centers at the one registered
coordinate:

| arm | `C` | `M` | signed-align center |
|:---|---:|---:|---:|
| `mu0` | `40` | `1` | **0.026824270191168845** |
| `nesterov_raw` | `320.1972515182563` | `9.86697205352709` | **0.0025029176872227174** |
| `heavy_ball` | `311.3302794647292` | `9.852191170585655` | **0.002575455491535066** |

These are placement values only. In particular, the implied center
`r=0.9330795094834432` is not a fifth registered scientific prediction and can
never win the V19 adjudication.

### 4.2 Paranoid width fixed before outcomes

The G13/G13B chronicle showed that apparently adequate four-rung transported
grids can put every discrete minimum on a boundary and can remain unbracketed
after a regrid. V19 therefore uses six one-octave-spaced rungs with symmetric
log2 offsets

```text
{-2.5, -1.5, -0.5, +0.5, +1.5, +2.5}.
```

The even grid intentionally does not contain its geometric center. It spans
five bits, a 32-fold endpoint ratio, and every arm has its own center. Exact eta
values, ascending, are:

| arm | six registered eta values |
|:---|:---|
| `mu0` | `{0.004741905838138914, 0.009483811676277829, 0.018967623352555658, 0.037935246705111315, 0.07587049341022263, 0.15174098682044526}` |
| `nesterov_raw` | `{0.00044245751734673343, 0.0008849150346934669, 0.0017698300693869337, 0.0035396601387738674, 0.007079320277547735, 0.01415864055509547}` |
| `heavy_ball` | `{0.0004552805106771446, 0.0009105610213542892, 0.0018211220427085783, 0.0036422440854171566, 0.007284488170834313, 0.014568976341668627}` |

No finite outcome, boundary minimum, nonconvex fit, surprising arm ratio, or
candidate preference permits moving or adding a rung.

## 5. Four primary predictions sealed before V19 data

All numeric calculations below use only artifacts committed in verifier commit
`c8ac48b5f41f77d5d9a5c4c649f50f34e2f6c8e0`. Source hashes used by the
recomputation are:

| artifact | SHA-256 |
|:---|:---|
| `mech/law-v2/verifier/signed_align_results.json` | `03e2671cfc0a84fd47b44026c8c5ba1f94c8741478e37876879a1267ce8b647f` |
| `mech/law-v2/forensics/shape_fits.csv` | `314efb55f9bf31d30d02444332e6144558956810f987a06dd803b8862de9f717` |
| `mech/law-v2/forensics/r_of_T_pairs.csv` | `e76c711f725db4cbf7c99ef2bf1f043200d0680c78bd0b441652feb3ee11d857` |
| `mech/law-v2/theory/interference_fit.py` | `1092e501fe8aaabde3afe6a3fde6706b278e8f20a7b9372a37a30f23d7457a4d` |
| `mech/law-unification/paired_cancellation.csv` | `15759bec558807f1ed3834f4465f18fdabaa1a10e678f6fd47f836ad870c3566` |

### 5.1 Forensics saturating exponential: `r_F = 1.0050466361633321`

The committed raw-`mu=0.9` fit has `A=6.708817` and `tau=6.739447`
(`mech/law-v2/forensics/shape_fits.csv:60`). The fitted form is
`log r = log(A)*exp(-T/tau)`
(`mech/law-v2/forensics/forensics.py:287-298` and
`mech/law-v2/forensics/forensics.py:469-470`). Replaying that script's fixed
240-point tau grid and WLS against its committed pair ledger, before the CSV's
six-decimal display rounding, gives `A=6.708816513977305` and
`tau=6.739447073713096`. Therefore

```text
r_F = exp(log(6.708816513977305) * exp(-40/6.739447073713096))
    = 1.0050466361633321.
```

This is the verifier's displayed `1.00` prediction.

### 5.2 C1-pure kinematics: `r_C1 = 1.2492299609173685`

The code/Lean-matched raw accumulated coefficient is

```text
C_raw(T,mu) = T/(1-mu) - mu^2*(1-mu^T)/(1-mu)^2,
```

committed at `mech/law-v2/zoo/fit_zoo.py:93-105` and
`mech/law-v2/theory/interference_fit.py:22-27`. At `(T,mu)=(40,0.9)`:

```text
C_raw = 320.1972515182563
T/C_raw = 0.12492299609173682
r_C1 = (T/C_raw)/(1-mu)
      = 1.2492299609173685.
```

This is the verifier's displayed `1.25` prediction.

### 5.3 C1+C3 interference: `r_C3 = 0.6097701376463005`

The committed interference script fixes
`eta0_135M=0.031350405899186334`, `q=0.9875756970034051`, defines

```text
x = sqrt(H)*eta0_135M*q^T*(C_raw-T),
phi_C3 = exp(-beta*x),
```

and fits `beta` through the origin over all 62 committed pairs
(`mech/law-v2/theory/interference_fit.py:19-20`,
`mech/law-v2/theory/interference_fit.py:29-42`, and
`mech/law-v2/theory/interference_fit.py:52-56`). Direct recomputation gives

```text
beta = 0.005949532405537653
x(40,0.9,512) = 120.54738048263995
phi_C3 = 0.4881168053306355
r_C3 = r_C1*phi_C3
     = 0.6097701376463005.
```

This is the verifier's displayed `~0.61` prediction. The full-precision fitted
`beta`, rather than a post-outcome choice between rounded `0.006` and a zoo
clock-dependent beta, is sealed here.

### 5.4 C4-frozen curvature ratchet: `r_C4 = 0.770027996425627`

C4 is qualitative in the candidate script, so the verifier made it testable by
freezing the raw deficit at its banked `(T=20,H=512)` level and carrying that
deficit to `T=40` (`mech/law-v2/verifier/VERDICT.md:357-364`). V19 removes any
later freedom in the phrase "T=20 level" as follows.

The two committed 135M raw pairs at exactly `(T,H,S,mu)=(20,512,10240,0.9)`
have `observed_to_law_ratio` values `0.10384689238282543` (G6) and
`0.11114616199505725` (TP-v3), at
`mech/law-unification/paired_cancellation.csv:30` and
`mech/law-unification/paired_cancellation.csv:60`. Applying the committed C1
residual calculation from `mech/law-v2/theory/interference_fit.py:34-38` gives

```text
phi_G6   = 0.5958180295630751
phi_TPv3 = 0.6376973418642764.
```

Because the response and model fits are in log-rate space, the preregistered
pool is their geometric mean:

```text
phi_C4_frozen = sqrt(phi_G6*phi_TPv3)
              = 0.6164021201189884
r_C4 = r_C1(T=40)*phi_C4_frozen
     = 0.770027996425627.
```

This is the exact operational value underlying the verifier's displayed
`~0.78`; neither constituent cell may be selected after seeing V19.

### 5.5 Sealed ordering and minimum separations

The only four eligible predictions, in increasing order, are:

| winner label | sealed `r` | `log2 r` separation from next prediction |
|:---|---:|---:|
| `C1_C3_INTERFERENCE` | **0.6097701376463005** | `0.3366454010123803` bits |
| `C4_FROZEN_RATCHET` | **0.770027996425627** | `0.3842796419699818` bits |
| `FORENSICS_SAT_EXP` | **1.0050466361633321** | `0.313776628897436` bits |
| `C1_PURE` | **1.2492299609173685** | — |

Nearest-prediction boundaries in `r` are the geometric midpoints
`{0.6852299448885458, 0.8797238473289029, 1.120506300747324}`. These values
are descriptive conveniences; the no-rounding distance calculation below is
authoritative.

## 6. Heavy-ball secondary signature sealed

The committed theory/verifier text describes the parameter-free signature as
`C_raw/C_hb = 1.026` and calls it near-equality
(`mech/law-v2/theory/CANDIDATES.md:65-72` and
`mech/law-v2/verifier/VERDICT.md:369-374`). Re-evaluating those files' own
closed forms at exactly `T=40`, however, gives

```text
C_raw(40,0.9) = 320.1972515182563
C_hb(40,0.9)  = 311.3302794647292
h_C1 = C_raw/C_hb
     = 1.0284809176568759  (0.04051502647618015 bits from equality).
```

The discrepancy between the committed display `1.026` and direct evaluation is
`0.0024809176568759` in ratio, far below expected experimental resolution, but
it is disclosed before outcomes. V19 seals **`1.026 near-equality` as the
artifact label and `1.0284809176568759` as the analyzer's full-precision numeric
target**. No analyzer may substitute the display value after V19 data exist.

The analyzer must report `h_obs`, its paired-bootstrap 95% CI, and the boolean

```text
hb_signature_hit := 1.0284809176568759 is inside the inclusive 95% CI.
```

This secondary boolean cannot change the primary winner. It directly tests the
shared/convention-blind residual premise: a material raw-vs-heavy-ball residual
difference remains a mandatory falsification diagnostic even if one primary
`r` prediction wins.

## 7. Prospective gatesim from banked cell noise

The gatesim is zero-GPU, read-only feasibility evidence. It does not count as a
V19 run and does not assume that one of the four scientific models is true.

### 7.1 Frozen donors

| target arm | banked donor | curvature `a` | donor seeds | source SHA-256 |
|:---|:---|---:|:---|:---|
| `mu0` | G6 135M `T=20,H=512,S=10240,mu0` | `0.017384641505974296` | `{601,607,613}` | `7e7e9cd8901520cc8f0ca822151af92c4c6183ee99bf02199bc1452a2763af8c` |
| `nesterov_raw` | G6 135M `T=20,H=512,S=10240,raw` | `0.035986270481111665` | `{601,607,613}` | same G6 hash |
| `heavy_ball` | G12 135M `T=20,H=128,S=2560,heavy_ball` | `0.03769428486377985` | `{981,983,991}` | `0162dbdd2a78492c0b167e2dd516396ab7ea0d1bcb34ac0293abf6abdfc67ae5` |

The committed sources are `mech/law-unification/sources/g6-readout.json` and
`mech/law-unification/sources/g12-readout.json`. The first two donors match the
target's 135M/H=512 coordinate class exactly in scale and local work. No
heavy-ball H=512 bank exists; the closest convention-matched G12 donor is
therefore disclosed rather than silently replacing heavy-ball noise with raw
noise.

For each donor and seed, the simulator subtracts the committed pooled loss at
each donor eta to obtain a seed residual profile. It retains the donor's fitted
curvature, translates the profile by log2 eta relative to the simulated true
vertex, and linearly interpolates/extrapolates the two adjacent donor residuals.
Each synthetic experiment samples three paired profile indices with replacement
and evaluates the exact six registered offsets.

The prospective placement stress uses one shared
`Normal(0,0.35 bits)` shift for all three true vertices and independent
`Normal(0,0.10 bits)` additional shifts for raw and heavy-ball. This reproduces
the registered G13-family stress while V19's wider ladder addresses the lesson
that the stress is not a scientific transport guarantee.

For each synthetic experiment the simulator applies the exact pooled quadratic
fit in section 8 and enumerates all `3^3=27` ordered three-seed bootstrap draws.
Evaluability requires all three pooled vertices to be strict interior optima and
at least 21/27 joint draws to accept all three curves. With RNG seed
`2026072819` and 2,000 synthetic experiments:

| gatesim quantity | result |
|:---|---:|
| synthetic experiments | `2000` |
| evaluable | `2000` |
| not evaluable | `0` |
| **`P_eval`** | **`1.0000`** |
| Wilson 95% interval | `[0.9981,1.0000]` |

As a discrimination diagnostic, 2,000 additional synthetic experiments at each
of the four exact sealed truths were all evaluable and all selected their
generating nearest prediction; none triggered `AMBIGUOUS`. These `8000/8000`
results show adequacy only under transported banked curvature/noise. They are
not empirical support for any candidate, do not license narrower grids or fewer
seeds, and cannot convert a future `NOT_EVALUABLE` or `AMBIGUOUS` outcome into a
winner.

## 8. Frozen estimator and uncertainty

For each arm independently:

1. At each exact eta, take the arithmetic mean held-out endpoint loss over the
   three registered seeds.
2. With `x=log2(eta)`, fit ordinary least squares
   `loss = a*x^2 + b*x + c` to all six rung means.
3. Accept the curve iff all 18 underlying endpoints are present and finite,
   `a>0`, and the vertex `-b/(2a)` lies strictly between the lowest and highest
   registered `x`. There is no near-bracket allowance because the ladder was
   made prospectively wide.
4. For an accepted curve set `eta*=2^(-b/(2a))`.
5. Form `r_obs=10*eta*_raw/eta*_mu0` and
   `h_obs=eta*_hb/eta*_raw` without intermediate rounding.

Run 10,000 paired nonparametric seed-bootstrap draws with RNG seed
`2026072819`. One common ordered three-index resample is applied to every eta
and all three arms; all three curves are refit inside each draw. At least 7,500
joint draws must accept all three curves. The reported 95% CIs are the
linear-interpolated 2.5th and 97.5th percentiles of `log2 r` and `log2 h`, then
exponentiated for display. Inclusive unrounded log-space endpoints determine
prediction coverage and `hb_signature_hit`.

The three-seed pooled fit, not a bootstrap median, is the point estimate used
by the winner rule. Per-seed fits, alternative regressions, robust fits,
development-stream losses, rung deletion, curvature pooling, and Bayesian
model weights are descriptive only and cannot alter the registered outcome.

## 9. Closed winner vocabulary and precedence

The only allowed top-level V19 scientific outcomes are:

- `FORENSICS_SAT_EXP`
- `C1_PURE`
- `C1_C3_INTERFERENCE`
- `C4_FROZEN_RATCHET`
- `AMBIGUOUS`
- `NOT_EVALUABLE`

Assign exactly one in this order:

1. **`NOT_EVALUABLE`** if registration-order or remote-reachability proof
   fails; any required command, model, data, held-out-stream, work, seed,
   optimizer recurrence, grid, endpoint, or artifact check fails; any of the 54
   cells is missing/nonfinite; any pooled curve is unaccepted; or fewer than
   7,500 joint bootstrap refits are valid. This takes precedence over every
   partial ratio or apparent model match.
2. Otherwise count, inclusively and without rounding, how many of the four
   sealed predictions lie inside the paired-bootstrap 95% CI for `r_obs`. If
   the CI contains **two or more**, assign **`AMBIGUOUS`** and declare no
   winner. This is the minimum-separation clause; a point estimate alone cannot
   break an interval that covers multiple registered models.
3. Otherwise compute, for every sealed prediction `r_j`,
   `distance_j = abs(log2(r_obs/r_j))`. Assign the label of the unique smallest
   distance. An exact tie within `1e-12` bits is **`AMBIGUOUS`**. The rule still
   applies when the CI contains zero or one prediction; no unregistered
   acceptance band is introduced.

No synonyms such as `PASS`, `FAIL`, `TREND`, `PARTIAL`, `LIKELY`, or
`INCONCLUSIVE` may replace these outcomes. The heavy-ball result is a mandatory
secondary signature, not a fifth primary model and not a tiebreaker.

## 10. Integrity, retry, and advance-registration rules

- The public commit containing this file precedes every V19 implementation,
  input manifest, launch manifest, authority, result root, attempt directory,
  and training process. Their timestamps and hashes must prove that ordering.
- This registration is `spec + gatesim only`. It spends no GPU time and does
  not open a fleet gate. Fleet inspection, input preparation, implementation,
  analyzer code, and launch authorization are separate future acts.
- No V19 outcome may be read until all 54 required cells have reached a
  terminal evidence state. Scheduling is loss-blind and longest-work ties are
  deterministic.
- At most one attempt-2 retry may later be authorized, and only for host/GPU,
  framework/driver, storage/network, or registered-timeout failure without a
  valid endpoint. The retry unit is all six eta cells sharing `(arm,seed)`.
  Attempts are append-only and attempt 2 never overwrites attempt 1.
- A finite high loss, finite scientific divergence, surprising ordering,
  boundary/disallowed vertex, nonpositive curvature, wide CI, model loss, or
  heavy-ball mismatch is not retryable. It yields the registered closed-vocabulary
  outcome, including `NOT_EVALUABLE` where required.
- Earlier law-v2 artifacts and results are immutable. They may supply the
  already-sealed centers, predictions, and gatesim noise only; no earlier tuned
  optimum is pooled into V19's estimator.

The four predictions are now sealed before the experiment. A future valid
measurement can select one, remain `AMBIGUOUS`, or be `NOT_EVALUABLE`; it cannot
rewrite the candidates, C4 pooling rule, heavy-ball target, grids, seeds,
estimator, interval, or winner rule.
