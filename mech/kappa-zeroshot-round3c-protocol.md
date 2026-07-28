# Kappa zero-shot Round 3C: execution-environment amendment to the frozen trajectory-instrumentation protocol

**Status:** FROZEN BEFORE ANY ROUND-3C PROBE

**Freeze date:** 2026-07-28 (America/Los_Angeles)

**Target used only for final adjudication:** `[0.9932, 0.9938]`

**Remaining free choices:** **EMPTY**

## Round-3C pre-measurement execution amendment

This is an execution-environment-only amendment to the scientific protocol
frozen in commit `2fa97103b6c9552ca52d2ee9eb912bada080f792`
(`mech/kappa-zeroshot-round3-protocol.md`, SHA-256
`650a465546ca0d69a1bfa8ca0f32cd55a12b0d58836e8b870c417fe402f2f977`).
Except for the Round-3C metadata, this amendment, and the exact CPU environment
registered below, the text following this section is carried forward from
`2fa9710`. Every scientific choice remains unchanged: the checkpoint panel,
adapter and arguments, deterministic panels, estimator chain, target interval,
closed adjudication vocabulary, fold rule, and empty free-choice list.

Round 3C performs no new producer run. Its inputs are the same eight retained
checkpoint files used by Round 3B, reverified byte-for-byte on `dev16` before
this freeze:

| arm | global steps | checkpoint SHA-256s in step order |
|---|---|---|
| `mu0` | `20,40,60,80` | `80abf3a6528f12c2c0f96ebc9c8f0492c6c149ca2d17a3497e81b5f3c6034d1a`, `f1ce70021c42c5ef6825a4a56a9051b17b02760a95d50d63f0cb89d95cb519af`, `ee93ffec6499367ca9851b435ddcc14869622cbffd0ba5fe883cf8f2bea4d4cd`, `1d7d3a3ee4471e8ae55bfd51d2165823397f1627b2f41186ce7b7625d0fde5b0` |
| `corrected` | `20,40,60,80` | `b81d16335ceba79e71dd5ed2131c2015476c00e4540bd4238be3317fd50a07eb`, `7a27f4c5235ba387919b16e6e53e9b7680f75be22c7338943f0c9cba4c6981e2`, `7cd0656fe5f269039bf09de398af31c7163977a432f74009097cca511a6e5d74`, `9dc8c5e084aa6ed197375cc0cffe4f0b65ad5026ce7834a18a34794fa0e46697` |

The corresponding `mu0` and `corrected` trajectory-manifest SHA-256s are
`08171f726e9585e06f162e1bcb6d8b4a3873c1da911aa391c1b820dc40927d2d`
and `9758ed608bd3e3d436e0adc4ca4e5b65eb490593db04aa29f69fe58d59e43a5c`.

This amendment is pre-measurement. At amendment time there are zero complete
probe JSONs, zero temporary probe JSONs, and no live adapter or measurement
process: probes are `0/8`. The cited Round-3B FAIL line in
`/private/tmp/h200-mechR3-note.md` is:

> `curvature_probes:FAIL(0/8; first input mu0 age=5 global_step=20 exited before JSON)`

The Round-3B adapter log (SHA-256
`7fc246a47e40c6225c387c4b897b25de510f666e1309720842aa3d7821aa1869`)
shows that its sole invocation stopped during deterministic panel construction
at `PermissionError: [Errno 13] Permission denied: '/data'`; it emitted no
spectrum, HVP, Ritz value, curvature, or estimator outcome. Round 3C therefore
registers exactly one fresh invocation per checkpoint, with no within-Round-3C
retry or substitution. For the inherited timing gate, scientific/acquisition
freeze remains the pre-producer commit `2fa9710`, while this amendment must
precede every Round-3C adapter invocation.

The original Round-3 execution-rail violation is neither erased nor cured:
these eight inputs were produced on `h200-n2:7` (`mu0`) and `h200-n2:6`
(`corrected`) while v10 compute remained elsewhere in the reserved
`h200-n2:4-7` block. This provenance caveat must accompany every Round-3C
result and any later write-up.

This is the last mechanism measurement before the ship decision. The two 135M
reruns below are **POST-HOC INSTRUMENTATION RUNS, NOT GATED SCIENCE**. They do
not reopen, replace, top up, or enter any registered v3/v6/v8 result. Their
endpoint losses are not an outcome of Round 3 and may not enter the estimator.
Their only scientific products are four retained trajectory checkpoints per
arm, the fixed Lane-E probes of those checkpoints, and the schedule audit.

Round 3 is necessarily retrospective. The target and the Round-1/Round-2
failures are already known. The protection against postdiction is therefore a
complete freeze of cells, checkpoint ages, probes, estimator, gates, coordinate
map, and labels before acquisition. Any unlisted choice discovered after this
commit makes the result `VOID`; execution stops and the choice is documented
without repairing the chain.

## 1. Why a rerun is required

Lane E found no retained intermediate trajectory in any specified live or
evacuated result tree. The conventional `work/m4/state.ckpt` files are rolling
terminal checkpoints: `--syncer-checkpoint-every` changes overwrite cadence
but does not retain generations. No banked run enabled numbered probe capture.
Consequently, terminal checkpoints cannot be relabeled as trajectory cuts and
no neighboring banked cell may substitute.

Lane E validated a read-only full-parameter checkpoint spectrum adapter on one
real 135M checkpoint. The accepted source is commit `c7650ef` and the fixed
files are:

```text
mech/lane-e/archive_checkpoint_steps.py
sha256 0aaa86715cc6484ab7860e369a3803202ce62deacc16d2670538faa2dd06f374

mech/lane-e/checkpoint_spectrum_probe.py
sha256 857c88c2a227c32f983c5d206c48d43f49792cdda2f797db691df1386e46d8bd
```

The adapter uses fp32 HVP evaluations and float64 NumPy
Lanczos/orthogonalization. The previously rejected fp32 full-vector Gram path
is forbidden.

## 2. Exact rerun cells

Exactly two cells are acquired. They reproduce an actual banked v3 S-scan
bias-correction configuration and its same-index, same-seed `mu=0` twin:

| Round-3 name | banked source cell | `mu` | correction flag | outer LR |
|---|---|---:|---|---:|
| `corrected` | `v3b-135m-s10240-mu0p9-e1-s313` | 0.9 | on | `0.002034969324913989` |
| `mu0` | `v3b-135m-s10240-mu0-e1-s313` | 0.0 | on (code-identity no-op) | `0.02034969324913989` |

The selection rule is target-independent and now closed: use the lowest
registered v3 seed (`313`) and the lower-index member (`e1`) of the two central
S=10240 rungs. No banked loss, fitted optimum, endpoint, kappa value, probe, or
GPU availability selected the seed or rung. No second rung, raw-momentum cell,
extra seed, replacement cell, or terminal canary enters Round 3.

The source launch manifest is
`/root/yeto-day1-control-v3/v3-launch-manifest.json`, SHA-256
`8fae6137d673d4c57861b37de09a42c5c462b0dff692cf10bde49e73caa554fc`.
The two original command hashes are, in table order,
`1e2b49b0d47bd3c18632d43862088b76dcc8da32a4eeda007a3ce3720a675848`
and
`3aca437eec731d85857febcf3551bc94b131d04fdcf0deea3ad48ee7f163c666`.
The producer source is the manifest-recorded commit
`6ea517425f149f10938e05b4487f217066acb8d7` in an isolated checkout.

Both cells retain all banked scientific settings:

- `HuggingFaceTB/SmolLM2-135M`, revision
  `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`;
- model-files manifest SHA-256
  `43f9494fad3335a9237f7a3093ae1401b7c4d3164c7486542070e2cc04837132`;
- train input `/root/yeto-data/outer-mup-v3/seed-313/train.jsonl`, SHA-256
  `0649212565ac84945e24ba4c449757171cbfe4c5254a8ec2d4a067c4e42ebaa6`;
- data seed `313`, training seed `313313`, development split seed `331`;
- `M=4`, four bin-packed full-parameter RDA fragments, `H=512`, `S=10240`,
  and `T=20` per-fragment outer updates (`80` global syncer steps);
- sequence length 128, microbatch 1, inner AdamW LR `0.001`, fixed window 512
  microsteps / 65,536 tokens, padded to the fixed token window;
- strict quorum, pipeline depth 4, barrier synchronization,
  version-matched anchor, no delay/jitter, no delta correction, and production
  Nesterov semantics;
- `HF_DATASETS_CACHE=/data/hf-datasets-cache`.

Only four instrumentation changes to the banked commands are allowed: an
isolated `/data` work/report root, the eligible GPU index, checkpoint overwrite
cadence `20` instead of `80` global steps, and
`--train-only-sealed-checkpoint` so no endpoint evaluation is produced. These
changes do not alter optimizer/data dynamics. No other command token changes.

There is no scientific retry or replacement rule. A preemption, producer
failure, archiver failure, missed cut, nonzero exit, or invalid artifact makes
Round 3 `VOID`. Partial attempts and checkpoints from different attempts may
not be combined.

## 3. Exact checkpoint panel

The sidecar archives the rolling checkpoint at exactly:

```text
global syncer step = 20, 40, 60, 80
per-fragment age   =  5, 10, 15, 20
```

This is every five per-fragment outer updates. At a retained age `a`, success
requires:

```text
checkpoint.global_step = 4*a
checkpoint.fragment_versions = [4*a-3, 4*a-2, 4*a-1, 4*a]
```

Each arm must have exactly four immutable
`state_after_step_XXXXXXXX.ckpt` files and one complete
`trajectory-manifest.json`. Every manifest size and SHA-256 is rechecked after
transfer. A later or terminal checkpoint cannot replace a missed earlier cut.
There are exactly eight primary spectrum inputs.

## 4. Probe run at every checkpoint

Run `mech/lane-e/checkpoint_spectrum_probe.py` once on each of the eight
checkpoint inputs, with no cache of a different checkpoint and these exact
arguments:

```text
model             /root/yeto-data/model (or byte-identical CPU copy)
data              /root/yeto-data/splits/seed-337/eval.jsonl
data sha256        533838a0564b13519956a044d23ed8db6705ddc7ae5f0ddb96538f49460bcebc
fragments         4
fragment pattern  binpack
sequence length   128
train_on          assistant
loss              cross_entropy
panels            4
batch size        1
max rows          128
block steps       4
Krylov rank       8 (two seed vectors times four block steps)
probe seed        20260727
dtype             fp32 HVP; float64 Lanczos/Rayleigh algebra
CPU threads       80 when run on the CPU server
```

The first four deterministic packed panels are used. Every emitted Ritz mode
and held-out-gradient coordinate enters the primary estimator. No panel,
fragment, layer, block, mode, sign, age, or arm may be selected or dropped
after inspection. The manifest/header/version/hash checks are also run for
every checkpoint. Sparse five-age cuts are not passed to Lane E's adjacent-age
recurrence helper, because that helper assumes every age is present; inventing
the unobserved intermediate buffers would be a new choice.

## 5. Frozen estimator chain

Round 3 retains Lane A's Round-2 identified scalar local-quadratic model and
changes only the source of the missing age-resolved curvature input. It does
not fit any coefficient to the target or to a corrected-arm tuned optimum.

### 5.1 NLL floor and loss exponent from banked `mu=0`

Use exactly the same six independent v8 `mu=0`, correction-off loss cells and
the same fit as Round 2:

```text
v8-t20-mu0-mu0-e{1,2}-seed{801,809,811}
```

Only the evacuated CPU copies under
`/home/c/h200-evac/{n1,n2}/yeto-results-v8`, attempt 1, are admissible. Use all
four learners and every logged local step `10,20,...,10240`. At a
`(seed,step)`, arithmetically average both rungs and four learners; at a step,
arithmetically average the three seed means. Nonfinite loss is a hard failure;
nonpositive no-target rows are excluded exactly as in Round 2.

For every candidate `F` in
`0 <= F < min_step(mean_loss(step))`, fit by unweighted OLS

```text
log(mean_loss(step) - F) = A - beta * log(step).
```

Choose `F` by the Round-2 4,097-point inclusive profile followed, only for an
interior grid minimum, by 256 golden-section iterations between adjacent grid
points; endpoints are evaluated explicitly. This is the byte-frozen floor
routine in `mech/kappa_round2_measure.py`, SHA-256
`4902dfff95b74bb089eb7fa10a483d9a3082d529c7b39966cda718310a9f6b16`.
The prior full Round-2 record, used only as a deterministic cross-check, is
`mech/kappa-round2-results.json`, SHA-256
`28358ada20742685128003d19cf63b8fb807a0e3851ccc5b602b712cd617b0e7`.
The Round-3 point inputs are the recomputed `ell_inf=F_hat` and `beta=beta_hat`;
rerun training losses and endpoint reports are forbidden inputs.

### 5.2 Effective curvature from all eight trajectory probes

For arm `r in {mu0, corrected}` and age `a in {5,10,15,20}`, let
`lambda_j` be the eight returned Ritz values and `z_j` the eight coordinates
of the unit held-out gradient. Define, unchanged from Round 2,

\[
\widehat\lambda_{r,a}
=\frac{\sum_{j=1}^{8}z_j^2}
       {\sum_{j=1}^{8}z_j^2/\lambda_j}.
\tag{1}
\]

This is the local-quadratic identity
`||g||^2/(g^T H^{-1}g)`. A result with the wrong rank, a zero/nonfinite Ritz
value, a nonfinite coordinate, zero gradient mass, a nonpositive/nonfinite
inverse quadratic form, or a nonpositive/nonfinite effective curvature makes
Round 3 `VOID`. Negative Ritz modes are never silently removed.

Fit one common curvature exponent with an arm fixed effect:

\[
\log \widehat\lambda_{r,a}=A_r-\gamma\log a+\epsilon_{r,a}
\tag{2}
\]

by unweighted OLS over all eight values. Since both arms have the same four
ages, this is equivalently OLS of

\[
\bar y_a=\tfrac12(\log\widehat\lambda_{\mathrm{mu0},a}
                 +\log\widehat\lambda_{\mathrm{corrected},a})
\]

on `log(a)` with an intercept, and `gamma` is the negative fitted slope. Arm
levels do not affect the common slope. Both arm-specific slopes and residuals
are reported as diagnostics but cannot replace, reweight, or gate the common
slope.

The Lane-A scale exponent is then

\[
\alpha=(\beta+\gamma)/2,
\qquad s_t=t^{-\alpha}.
\tag{3}
\]

`alpha` must be finite and strictly positive; otherwise the result is `VOID`.

### 5.3 Kernel fixed by the production schedule

No kernel is estimated from the probes. For fixed `mu=0.9`, per-fragment age
`t`, and `b_0=0`, the production schedule and optimizer source give

\[
b_t=\mu b_{t-1}+s_t,\qquad
d_t=s_t+\mu b_t.
\]

After the code's finite-history divisor and normalization by the intended
steady gain `1/(1-mu)`, the corrected scalar direction is

\[
c_t=\frac{(1-\mu)d_t}{1-\mu^{t+1}}
=\frac{1-\mu}{1-\mu^{t+1}}
\left[(1+\mu)s_t+\sum_{k=1}^{t-1}\mu^{k+1}s_{t-k}\right].
\tag{4}
\]

The server schedule is
`fragment=(global_step-1) mod 4`, with per-fragment age
`floor((global_step-1)/4)+1`. Thus the same-fragment kernel has global support
only at lags `4k`; the other three global commits have coefficient zero. The
two rerun tapes must contain exactly 80 rows and pass this schedule identity.
The audit can invalidate the result but cannot change the kernel.

### 5.4 Lane-A path estimate and mapped kappa

For the already registered scoring ages `T in {2,5,10,20}`, compute

\[
D_{\rm pred}(T)=
\frac{\sum_{t=1}^{T}s_t}{\sum_{t=1}^{T}c_t}.
\tag{5}
\]

Fit the literal per-age multiplier through the origin with the unchanged
sealed target-coordinate weights:

\[
\log q_{\rm pred}=
\frac{\sum_T w_T T\log D_{\rm pred}(T)}
     {\sum_T w_T T^2},
\tag{6}
\]

where, in `(2,5,10,20)` order,

```text
w = (4403.205555698397,
     3714.8199765100494,
     9071.170574218335,
     1896.2276107325906)
```

Finally map like-for-like into the supplied legacy coordinate:

\[
U=q_{\rm pred}^{2}\frac{1+\mu}{1-\mu},\qquad
\kappa_{\rm pred}=\frac{U-1}{\mu(U+1)}.
\tag{7}
\]

Equation (7) is only the frozen scoring-coordinate map. It does not validate
the previously rejected scalar-correlation interpretation. Every intermediate
`ell_inf`, `beta`, eight effective curvatures, `gamma`, `alpha`, four
`D_pred`, `q_pred`, and `kappa_pred` is reported. The point prediction, not a
diagnostic sensitivity, receives the label. With one fixed trajectory pair,
Round 3 does not invent a seed-bootstrap confidence interval; the regression
residuals and the already frozen Round-2 loss bootstrap are descriptive only
and cannot change the label.

## 6. Input, execution, and model gates

All gates are conjunctive. Failure of any gate is `VOID`:

1. this protocol commit predates every Round-3 producer and probe process;
2. both exact cells complete once, from the fixed producer commit and hashed
   inputs, without preemption or command drift;
3. both manifests contain all four exact cuts and every post-transfer hash,
   size, header, fragment count, and version check passes;
4. all eight spectrum probes complete with the fixed adapter/input hashes and
   rank eight;
5. the banked loss inputs and Round-2 floor fit pass, with finite
   `0 <= ell_inf < min(mean loss)` and `beta>0`;
6. all eight calculations in (1), the common-slope fit (2), and (3) pass;
7. the two 80-row schedules pass and equations (4)--(7) produce finite,
   strictly positive `D_pred`, `q_pred`, and `kappa_pred`;
8. the measurement program never opens a rerun endpoint report or uses a
   banked corrected-arm endpoint/tuned optimum as an estimator input; and
9. no unregistered cell, age, mode treatment, pooling rule, retry,
   substitution, response functional, weight, or coefficient is introduced.

Discovery of any new analytical or operational choice after the freeze is not
an invitation to decide it. Write `KAPPA ROUND3: <VOID, ...>` to the progress
note, state the choice, stop the chain, and preserve the partial artifacts.

## 7. Closed adjudication vocabulary

Let `I=[0.9932,0.9938]` and `W=0.0006`.

- **HIT:** all gates pass and `kappa_pred` lies in the closed interval `I`.
- **NEAR:** all gates pass, the result is not a HIT, and its distance to `I`
  is at most `2W=0.0012`; equivalently it lies in `[0.9920,0.9950]` outside
  `I`.
- **MISS:** all gates pass and it lies outside `[0.9920,0.9950]`.
- **VOID:** a protocol/input/model/execution gate fails, or any free choice is
  discovered after this freeze. `VOID` precedes and overrides numerical
  comparison.

No alternate vocabulary, rounding-to-hit rule, interval-overlap rule, or
uncertainty-based upgrade/downgrade is allowed.

## 8. Fleet and storage rails

The artifacts live under an isolated `/data/yeto-mech-round3` root on LVM;
the registered v3/v10/P1 result trees and checkouts are never modified.

- Never use `h200-n1:0-7` or `h200-n2:0-3` while any P1 island lives.
- Never use `h200-n2:4-7` while v10 compute is present.
- Before an early opportunistic claim, both the v10 slot controllers and P1
  islands must leave the candidate device unclaimed continuously for at least
  ten minutes. Eligibility is established by polling at most every 30 seconds
  for at least 600 seconds, followed by an immediate final compute-PID check.
- After the expected v10 drain, use the first two eligible devices from
  `h200-n2:4-7` in ascending index order; bind `corrected` to the lower and
  `mu0` to the higher. GPU identity is operational provenance and never an
  estimator input.
- Every process search uses bracketed patterns such as
  `pgrep -af '[r]un_slot_v10.py'` and
  `pgrep -af '[c]ompare_diloco.py'`; an unbracketed self-matching `pgrep` is
  forbidden.
- A higher-priority claim after launch is obeyed immediately. Killing either
  Round-3 producer makes the frozen result `VOID`; no replacement is launched.
- Require at least 30 GB free in the isolated acquisition root before launch.

CPU probing uses the validated CPU server or, only if that host is unavailable,
an eligible post-drain GPU with the same adapter arguments. Host choice cannot
change the panel or estimator. Round 3C fixes the CPU host to
`c@65.19.161.135` (`dev16`), uses 80 threads, and exports exactly:

```text
HF_DATASETS_CACHE=/home/c/yeto-mechR3-20260727/scratch/round3c/hf-datasets-cache
TMPDIR=/home/c/yeto-mechR3-20260727/scratch/round3c/tmp
```

Both directories were created as user `c`, resolve under `/home/c`, and were
verified by `test -d`, `test -w`, and successful sentinel create/remove checks
before this freeze. The recorded check at `2026-07-28T10:02:34Z` was:

```text
path=/home/c/yeto-mechR3-20260727/scratch/round3c/hf-datasets-cache owner=c:c mode=700 type=directory
path=/home/c/yeto-mechR3-20260727/scratch/round3c/tmp owner=c:c mode=700 type=directory
writable=PASS host=dev16 user=c utc=2026-07-28T10:02:34Z json_count=0
```

The `/data/hf-datasets-cache` setting above remains the historical producer
setting; it must not be exported by the Round-3C CPU wrapper. No other probe
environment or argument changes. Results and the verdict are continuously
recorded at `/private/tmp/h200-mechR3-note.md` in the form:

```text
KAPPA ROUND3: <verdict, ell_inf=..., beta=..., curvature=8/8,
gamma=..., alpha=..., D_pred=..., q_pred=..., kappa_pred=...,
schedule=..., free_choices=EMPTY>
```

Pending stages use `verdict=PENDING`; a failed gate uses `verdict=VOID` and
names the failed input.

## 9. Fold rule

A numerical `HIT` does **not** authorize an automatic paper edit. It may fold
into the shipping paper only after both:

1. an independent umpire countersigns protocol compliance and the empty
   free-choice declaration; and
2. the orchestrator reviews and explicitly approves the fold.

Without both approvals, even a HIT remains an external Round-3 result. Every
`NEAR`, `MISS`, or `VOID` goes to the v2/ICLR mechanism file and is not folded
into the shipping paper. The Round-3 executor does not choose or edit that
destination file; the orchestrator owns the handoff.

## 10. Remaining free choices

**EMPTY.**

Changing a cell, seed, rung, producer source, command token beyond the four
listed instrumentation substitutions, checkpoint cadence, age, retry rule,
probe input, panel, Krylov rank, Ritz treatment, curvature functional, arm
pooling, loss curve, floor objective, kernel, path aggregation, scoring weight,
coordinate map, label boundary, or fold rule after this commit makes Round 3
`VOID`.
