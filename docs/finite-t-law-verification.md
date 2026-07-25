# Finite-\(T\) outer-learning-rate law: code derivation and full v1/v2 check

**Lane A verdict: REFUTED.** The proposed

\[
D_{\mathrm{HB}}(T,\mu)=\frac{1}{1-\mu^T}
\]

is the final-multiplier formula for heavy-ball momentum, but it is not the
formula implemented by the syncer. The production Nesterov look-ahead uses the
newly updated buffer in the same commit, giving

\[
c_{\mathrm{code}}(T,\mu)
  =\frac{1-\mu^{T+1}}{1-\mu},\qquad
D_{\mathrm{code}}(T,\mu)
  =\frac{1}{1-\mu^{T+1}}.
\]

That one-power shift is not a rounding detail. Across the two readouts there
are 18 registered curve families, 15 interior optima, eight \(\mu=0\)
calibration identities, and seven genuine nonzero-\(\mu\) primary
comparisons. The code-true final-multiplier formula has 6.96% mean absolute
relative error and lands inside only one of five registered-valid paired 95%
intervals. Two other numerical intervals contain it but are formally
non-evaluable because some paired bootstrap replicates are unbracketed. In
particular, at \(H=512,\mu=.9,T=5\), the code-true prediction is
\(2.1342\), not \(2.4420\), and is below both the v1 and v2 intervals.

## Scope and evidence

This audit used CPU-only local calculations and read-only SSH inspection of
`h200-n1`; it launched no training or GPU work.

| item | value |
|---|---|
| repository branch | `experiment/best-paper-phase-map` |
| audited local commit before this report | `44359f557376e9e3d13f5bf9de152e7e3317f80d` |
| v1 readout | `h200-n1:/root/g1-readout.json` |
| v1 SHA-256 | `c2dcd6b9ab7dce0dc28d1e2473a72c7e0bdb8d6221728f09503a6354a39cae2b` |
| v1 source commit recorded by readout | `ace897824b396018e0090bec0f1f3a9b707aa709` |
| v2 readout | `h200-n1:/root/g1v2-readout.json` |
| v2 SHA-256 | `5d4eed9685f25fd1db3135319908a045a389300a008a67bc009cd178cabe2fc8` |
| v2 source commit recorded by readout | `a886a3996905913d37ec56cc14914878f636283d` |

Both recorded experiment commits are ancestors of the audited local commit.
The extracted `nesterov_step` source has the same SHA-256,
`65368157a34e8c01b2556972f5054b3ea24d799180f1bcc88ff66e16c44dbb45`,
at the v1 commit, the v2 commit, and the audited local commit. The
`GlobalState::new` initialization slice is likewise unchanged across the three
versions.

The read-only launch manifests on `h200-n1` show that both campaigns explicitly
used `--outer-optimizer nesterov`, the treatment value of
`--outer-momentum`, one global `--outer-lr`, `--delta-correction none`,
`--learner-max-steps 2560`, strict fixed windows, and four fragments. The work
contract and observed tape lengths agree:

| \(H\) | learner steps \(S\) | full model rounds \(T=S/H\) | syncer commits \(4T\) | commits seen by each fragment buffer |
|---:|---:|---:|---:|---:|
| 16 | 2560 | 160 | 640 | 160 |
| 64 | 2560 | 40 | 160 | 40 |
| 256 | 2560 | 10 | 40 | 10 |
| 512 | 2560 | 5 | 20 | 5 |

For example, the audited v1 H512 tape has exactly 20 rows. Fragment 0 is
updated at global commits 1, 5, 9, 13, 17; the other three fragments have the
same five-update cadence at offsets 2, 3, and 4. Thus \(T=5\), rather than
20, is the correct exponent-count horizon for a particular momentum buffer.

## What the production caller path actually executes

The relevant path is short and has no hidden bias correction:

1. [`syncer/src/main.rs`](../syncer/src/main.rs) parses `outer_lr`,
   `outer_momentum`, and `outer_optimizer`, with Nesterov as the default, and
   places the exact values in the server configuration (lines 60–79 and
   216–234 at the audited commit).
2. [`syncer/src/server.rs`](../syncer/src/server.rs) constructs
   `GlobalState` with that learning rate and momentum and installs the selected
   optimizer (lines 2103–2122). The E1/E1v2 telemetry runs use the
   `TokenWeighted` production branch, which builds the aggregate and calls
   `apply_aggregate_step` (lines 1574–1608).
3. [`syncer/src/state.rs`](../syncer/src/state.rs) allocates parameter-shaped
   zero vectors and clones them into the per-fragment momentum buffers (lines
   577–583). Loading initial model parameters changes `params`, not those zero
   buffers (lines 687–704). `preview_aggregate_inner` clones the current buffer
   and passes the raw configured LR and momentum to `apply_outer_step` (lines
   915–959). The resulting buffer is installed as the next persistent buffer
   (lines 1885–1907).
4. [`syncer/src/merge.rs`](../syncer/src/merge.rs) dispatches plain Nesterov
   directly to `nesterov_step` (lines 138–158). That function performs, in this
   order, `b = mu*b + d`, `direction = d + mu*b`, `step = lr*direction`, and
   `p -= step` (lines 1573–1593).

Consequently, in real-arithmetic notation for one fragment,

\[
\begin{aligned}
b_0 &= 0, \\
b_t &= \mu b_{t-1}+\delta_t, \\
d_t &= \delta_t+\mu b_t, \\
\theta_t &= \theta_{t-1}-\eta d_t.
\end{aligned}
\]

There is no division by \(1-\mu^t\), no bias correction, no
dampening, no weight decay, and no placement of \(\eta\) inside the
buffer recurrence. The LR multiplies the completed look-ahead direction once.
The production implementation is f32, so the identities below describe its
real-arithmetic law up to ordinary f32 rounding; the exponent shift is exact at
the algorithmic level.

The existing deterministic Rust audit also confirms this ordering. Running
`cargo test nesterov_three_step -- --nocapture` in `syncer/` passes the
hand-computed three-step Nesterov sequence. As an independent result-tape
check, the first H512, \(\mu=.9\) v1 commit has
`gnorm=104.4138150095`, `eta=.01300695282034226`, and
`outer_step_norm=2.5804004908`, hence

\[
\frac{\|\mathrm{step}_1\|}{\eta\|\delta_1\|}
=1.89999994\approx 1+\mu,
\]

which rules out a heavy-ball first multiplier of 1 for the actual run.

## Step-by-step derivation

### 1. Exact filter for arbitrary round deltas

Starting from \(b_0=0\):

\[
\begin{array}{lll}
b_1=\delta_1,
&d_1=(1+\mu)\delta_1, \\[3pt]
b_2=\mu\delta_1+\delta_2,
&d_2=\mu^2\delta_1+(1+\mu)\delta_2, \\[3pt]
b_3=\mu^2\delta_1+\mu\delta_2+\delta_3,
&d_3=\mu^3\delta_1+\mu^2\delta_2+(1+\mu)\delta_3.
\end{array}
\]

Induction gives

\[
b_t=\sum_{s=1}^{t}\mu^{t-s}\delta_s
\]

and therefore

\[
\begin{aligned}
d_t
  &=\delta_t+\mu\sum_{s=1}^{t}\mu^{t-s}\delta_s \\
  &=(1+\mu)\delta_t
    +\sum_{s=1}^{t-1}\mu^{t-s+1}\delta_s \\
  &=(1+\mu)\delta_t
    +\mu^2\delta_{t-1}+\mu^3\delta_{t-2}
    +\cdots+\mu^t\delta_1.
\end{aligned}
\]

This vector convolution is the exact code law. For general training deltas
there is no scalar \(c(T,\mu)\): prior deltas can change norm, rotate,
or oppose the current delta. The exact scalar projection onto the current
delta is

\[
a_t
=\frac{\langle d_t,\delta_t\rangle}{\|\delta_t\|^2}
=1+\mu+\sum_{k=1}^{t-1}\mu^{k+1}
  \rho_{t,k}
  \frac{\|\delta_{t-k}\|}{\|\delta_t\|},
\]

where \(\rho_{t,k}\) is the cosine between
\(\delta_{t-k}\) and \(\delta_t\). Even this projection
does not represent the transverse part of \(d_t\). A law depending only
on \(T\) and \(\mu\) therefore requires an extra
constant-direction/equal-norm approximation.

### 2. Equal, perfectly aligned deltas

Set every \(\delta_s=\delta\). Then the first round multipliers
are

\[
c_1=1+\mu,\quad
c_2=1+\mu+\mu^2,\quad
c_3=1+\mu+\mu^2+\mu^3,\quad\dots
\]

and the effective direction in round \(t\) is

\[
d_t=c_t\delta,\qquad
c_t=\sum_{j=0}^{t}\mu^j
    =\frac{1-\mu^{t+1}}{1-\mu}.
\]

The proposed heavy-ball expression instead ends at
\(\mu^{t-1}\):

\[
c_t^{\mathrm{HB}}
=\sum_{j=0}^{t-1}\mu^j
=\frac{1-\mu^t}{1-\mu}.
\]

It would be correct for an applied heavy-ball direction \(d_t=b_t\),
not for this code's \(d_t=\delta_t+\mu b_t\). Put another way,
the hypothesized five-round H512 multiplier \(4.0951\) is the code's
*fourth*-round equal-delta multiplier; the actual fifth-round multiplier is
\(4.68559\):

\[
1.9,\;2.71,\;3.439,\;4.0951,\;4.68559.
\]

For completeness, the total parameter-path multiplier through \(T\)
rounds is a different quantity:

\[
C_T=\sum_{t=1}^{T}c_t
=\frac{T}{1-\mu}
 -\frac{\mu^2(1-\mu^T)}{(1-\mu)^2}.
\]

At H512, \(\mu=.9\), this is \(C_5=16.82969\). The
question's hypothesis is explicitly a *final-multiplier* heuristic, so
\(c_T\), not \(C_T\), is used for the primary predictions below.
A path-matching sensitivity check is reported later and does not fit better.

### 3. Code-true final-multiplier prediction

Matching the final equal-delta effective step to the \(\mu=0\)
optimum gives

\[
\eta^*_{\mathrm{pred}}(\mu)c_T(\mu)
=\eta^*_{\mathrm{obs}}(0)c_T(0)
=\eta^*_{\mathrm{obs}}(0).
\]

Thus

\[
\begin{aligned}
\eta^*_{\mathrm{pred}}(\mu)
  &=\eta^*_{\mathrm{obs}}(0)
    \frac{1-\mu}{1-\mu^{T+1}}, \\
D_{\mathrm{pred}}(T,\mu)
  &=\frac{\eta^*_{\mathrm{pred}}(\mu)/\eta^*_{\mathrm{obs}}(0)}
          {1-\mu}
    =\frac{1}{1-\mu^{T+1}}.
\end{aligned}
\]

The \(\mu=0\) rows below are calibration identities, not validation
points: their observed optimum supplies the baseline used to predict the
nonzero-\(\mu\) optimum at the same H and in the same readout.

## Complete primary curve-family comparison

`CI` below means the paired training-seed 95% interval for \(D_{\mathrm{obs}}\),
which is the interval cited in the hypothesis. `inside`/`outside` compares the
code-true \(D_{\mathrm{pred}}\) with that interval. The readouts contain
the registered 10,000-draw paired intervals for the \(\mu=.9\) rows.
Because v1's registered G1 did not request \(\mu=.5\) ratios, those two
intervals were reconstructed from v1 `cell_evidence` using the identical
frozen procedure: `random.Random(20260724)`, paired five-seed resampling,
quadratic refitting in \(\log_2\eta\), transformation in
\(\log_2 D\), and the analyzer's linearly interpolated quantiles.
The reconstruction reproduces the stored \(\mu=.9\) intervals to the
displayed digits.

A dagger means the displayed quantiles are descriptive only: at least one of
the 10,000 resamples was unbracketed, so the frozen analyzer's status is
`NOT_EVALUABLE`. `NE` means the pooled optimum itself is unbracketed. The
\(\mu=0\) \([1,1]\) interval is an algebraic identity for the
self-ratio; the separate \(\eta^*\) uncertainty remains in the source
readout.

| readout | H | \(\mu\) | \(T\) | \(\eta^*_{\mathrm{obs}}\) | \(\eta^*_{\mathrm{pred}}\) | \(D_{\mathrm{obs}}\) | \(D_{\mathrm{pred}}\) | paired 95% CI for \(D_{\mathrm{obs}}\) | result |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| v1 | 16 | 0 | 160 | 0.02324899 | 0.02324899 | 1.0000 | 1.0000 | [1, 1] identity | calibration |
| v1 | 16 | .5 | 160 | 0.01166595 | 0.01162450 | 1.0036 | 1.0000 | [0.9944, 1.0147] | **inside** |
| v1 | 16 | .9 | 160 | 0.00238669 | 0.00232490 | 1.0266 | 1.0000 | [1.0133, 1.0416] | **outside** |
| v1 | 64 | 0 | 40 | 0.01874129 | 0.01874129 | 1.0000 | 1.0000 | [1, 1] identity | calibration |
| v1 | 64 | .9 | 40 | — (`UNBRACKETED`) | 0.00189940 | — | 1.0135 | — | NE |
| v1 | 256 | 0 | 10 | 0.02703341 | 0.02703341 | 1.0000 | 1.0000 | [1, 1] identity | calibration |
| v1 | 256 | .5 | 10 | 0.01465196 | 0.01352331 | 1.0840 | 1.0005 | [0.9500, 1.1149]†; 3,639 invalid | inside† |
| v1 | 256 | .9 | 10 | — (`UNBRACKETED`) | 0.00393964 | — | 1.4573 | — | NE |
| v1 | 512 | 0 | 5 | 0.04440766 | 0.04440766 | 1.0000 | 1.0000 | [1, 1] identity | calibration |
| v1 | 512 | .9 | 5 | 0.01115341 | 0.00947750 | 2.5116 | 2.1342 | [2.4550, 2.5685] | **outside** |
| v2 | 16 | 0 | 160 | 0.02306166 | 0.02306166 | 1.0000 | 1.0000 | [1, 1] identity | calibration |
| v2 | 16 | .9 | 160 | 0.00239010 | 0.00230617 | 1.0364 | 1.0000 | [1.0281, 1.0469] | **outside** |
| v2 | 64 | 0 | 40 | 0.01963530 | 0.01963530 | 1.0000 | 1.0000 | [1, 1] identity | calibration |
| v2 | 64 | .9 | 40 | 0.00195657 | 0.00199000 | 0.9965 | 1.0135 | [0.9319, 1.1618]†; 4 invalid | inside† |
| v2 | 256 | 0 | 10 | 0.02705318 | 0.02705318 | 1.0000 | 1.0000 | [1, 1] identity | calibration |
| v2 | 256 | .9 | 10 | — (`UNBRACKETED`) | 0.00394252 | — | 1.4573 | — | NE |
| v2 | 512 | 0 | 5 | 0.04429579 | 0.04429579 | 1.0000 | 1.0000 | [1, 1] identity | calibration |
| v2 | 512 | .9 | 5 | 0.01081410 | 0.00945362 | 2.4413 | 2.1342 | [2.2920, 2.6090] | **outside** |

This table includes every registered primary curve family, including the three
families for which no pooled optimum can be reported. It also makes the sample
count explicit: there are 15 numerical optima, but only seven nonzero-momentum
predictions. Counting the eight \(\mu=0\) self-calibrations as successful
predictions would artificially inflate fit quality.

## Secondary eight-seed top-up fits

The readouts also contain five secondary contested-cell records. They overlap
the primary seeds, use the five-primary-seed \(\mu=0\) denominator, and
were preregistered as robustness-only, so they are shown for completeness but
are not pooled into the primary error metrics.

| readout | role | H | \(\mu\) | \(T\) | all-eight \(\eta^*_{\mathrm{obs}}\) | \(\eta^*_{\mathrm{pred}}\) | \(D_{\mathrm{obs}}\) | \(D_{\mathrm{pred}}\) | secondary 95% CI | result |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| v1 | secondary | 256 | .9 | 10 | — (`UNBRACKETED`) | 0.00393964 | — | 1.4573 | [2.1180, 2.7443]†; 8,463 invalid | NE |
| v1 | secondary | 512 | .9 | 5 | 0.01116699 | 0.00947750 | 2.5147 | 2.1342 | [2.4378, 2.5855] | **outside** |
| v2 | secondary | 64 | .9 | 40 | 0.00194626 | 0.00199000 | 0.9912 | 1.0135 | [0.9239, 1.1704]†; 4 invalid | inside† |
| v2 | secondary | 256 | .9 | 10 | 0.00438562 | 0.00394252 | 1.6211 | 1.4573 | [1.3415, 2.3512]†; 2,736 invalid | inside† |
| v2 | secondary | 512 | .9 | 5 | 0.01065649 | 0.00945362 | 2.4058 | 2.1342 | [2.2974, 2.5744] | **outside** |

Among the two registered-valid secondary intervals, the code-true prediction
is outside both.

## Honest fit assessment across all predictive primary points

Define the signed relative residual as

\[
r=\frac{D_{\mathrm{obs}}}{D_{\mathrm{pred}}}-1.
\]

| readout | H | \(\mu\) | \(T\) | \(D_{\mathrm{obs}}\) | \(D_{\mathrm{pred}}\) | relative residual |
|---|---:|---:|---:|---:|---:|---:|
| v1 | 16 | .5 | 160 | 1.0036 | 1.0000 | +0.36% |
| v1 | 16 | .9 | 160 | 1.0266 | 1.0000 | +2.66% |
| v1 | 256 | .5 | 10 | 1.0840 | 1.0005 | +8.35% |
| v1 | 512 | .9 | 5 | 2.5116 | 2.1342 | +17.68% |
| v2 | 16 | .9 | 160 | 1.0364 | 1.0000 | +3.64% |
| v2 | 64 | .9 | 40 | 0.9965 | 1.0135 | −1.68% |
| v2 | 512 | .9 | 5 | 2.4413 | 2.1342 | +14.39% |

Across these seven points:

- mean signed relative residual: **+6.48%**;
- mean absolute relative error: **6.96%**;
- RMSE in \(\log_2 D\): **0.1259 bits**, equivalent to a 1.091x
  multiplicative error factor;
- maximum absolute relative error: **17.68%**;
- registered-valid paired-CI coverage: **1/5**; if the two formally invalid
  descriptive intervals are nevertheless counted, numerical coverage is
  **3/7**.

The residual structure is not random-looking white noise:

1. **Short H / long filter history:** at H16, where \(T=160\) makes every
   finite-\(T\) power negligible, the residual is only +0.36% at
   \(\mu=.5\) but +2.66% in v1 and +3.64% in v2 at
   \(\mu=.9\). That is consistent with an omitted nonlinear-in-
   \(\mu\) or geometry term, but two momentum levels are insufficient
   to identify a \(\mu^2\) law.
2. **The effect is not only a high-\(\mu\) correction:** at H256,
   \(\mu=.5\) is already +8.35% above the code-true prediction. Its
   paired bootstrap interval is formally invalid because 3,639 resamples are
   unbracketed, so this is descriptive rather than confirmatory, but the point
   estimate shows a strong H/T interaction.
3. **H64 is the exception, not a clean trend:** v2 H64 \(\mu=.9\) is
   −1.68%, but its interval is broad and formally non-evaluable because four
   resamples are unbracketed. It prevents claiming a monotone residual curve.
4. **H512 is the replicated large miss:** both independent readouts are well
   above the code formula, by +17.68% and +14.39%, and both valid paired
   intervals exclude \(2.1342\). The secondary eight-seed summaries have
   the same sign and also exclude it when their intervals are valid.

This pattern is compatible with the exact arbitrary-delta filter: lagged
directions and their norms change with H, fragment, optimizer feedback, and
training time. It is not compatible with treating every prior delta as one
unchanging collinear vector and expecting \(T\) and \(\mu\) alone
to determine the tuned optimum.

An illustrative tape check makes that limitation concrete. In the same v1
H512 seed-101 arm, all four first-update norm gains are 1.9 as the code
requires. By each fragment's fifth update, the realized norm gains
\(\|d_t\|/\|\delta_t\|\) are 5.27 for fragment 0 and
approximately 3.47–3.48 for fragments 1–3, while the equal-delta scalar is
4.68559. Their direction/current-delta cosines are approximately .91 and .81,
respectively. The source code is the same, but the vector histories are not.

## The H512 v1–v2 tension

At \(T=5,\mu=.9\):

| quantity | value |
|---|---:|
| proposed heavy-ball \(D_{\mathrm{HB}}=1/(1-.9^5)\) | 2.441943 |
| code-true Nesterov \(D_{\mathrm{code}}=1/(1-.9^6)\) | 2.134203 |
| v1 primary \(D_{\mathrm{obs}}\) | 2.511596 [2.454970, 2.568480] |
| v2 primary \(D_{\mathrm{obs}}\) | 2.441338 [2.292028, 2.609023] |

The code derivation does **not** resolve the tension:

- v1 excludes the proposed 2.441943 narrowly: it is 0.013027 below the CI's
  lower endpoint. The code-true 2.134203 is 0.320767 below that endpoint and
  is therefore a substantially worse fit.
- v2 contains the proposed value, which differs from the point estimate by
  only 0.025% relative. The code-true value is below the v2 interval by
  0.157825.
- The v1 and v2 point estimates differ by about 2.8%, while their intervals
  overlap broadly. The “tension” is mainly that the narrow v1 interval happens
  to put 2.441943 just outside while the wider v2 interval contains it; it is
  not evidence that the off-by-one code formula reconciles the datasets.
- The secondary eight-seed H512 intervals contain 2.441943 in both versions
  but exclude 2.134203 in both versions.

The v2 H512 match is therefore a real numerical observation but a flattering
single coordinate, not verification of a code-derived finite-\(T\) law.

## Sensitivity to the scalarization choice

For transparency, three different collinear scalarizations were scored on the
same seven nonzero primary point estimates:

| scalarization | \(D_{\mathrm{pred}}\) | MAPE | RMSE in \(\log_2D\) | valid-CI hits |
|---|---|---:|---:|---:|
| **code final multiplier** | \(1/(1-\mu^{T+1})\) | 6.96% | 0.1259 | 1/5 |
| proposed heavy-ball final multiplier | \(1/(1-\mu^T)\) | 2.81% | 0.0530 | 2/5 |
| code total-path matching | \(T/[(1-\mu)C_T]\) | 8.67% | 0.1886 | 1/5 |

The proposed expression happens to fit these point estimates better than the
code-true final multiplier, driven heavily by v2 H512. That empirical ranking
does not repair its derivation, and even the proposed expression hits only two
of five valid intervals. The total-path interpretation also fails to rescue a
T-only law; at H512 it predicts 2.97094, overshooting both observed optima.

## Conclusion

The hypothesis is refuted on two independent grounds:

1. **Implementation:** the actual Nesterov look-ahead has a same-round
   \(1+\mu\) first multiplier, shifting the equal-delta final gain from
   \((1-\mu^T)/(1-\mu)\) to
   \((1-\mu^{T+1})/(1-\mu)\).
2. **Data:** the resulting code-true heuristic misses four of five valid
   primary paired intervals, including both H512 readouts, with systematic
   H- and \(\mu\)-dependent residuals.

The only defensible “modified formula” from source inspection is the
equal-delta heuristic \(D=1/(1-\mu^{T+1})\), but the full readout does
not support it as an empirical law of tuned LR. A viable next model would need
the lagged cosine and norm-ratio terms in the exact projection \(a_t\), or
another explicitly registered summary of the realized vector filter; no
scalar function of \(T\) and \(\mu\) alone is established here.
