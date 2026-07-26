# Theory Lane A: per-mode linear response

Status: **PARTIAL** (2026-07-26).

This lane derives a code-true closed-loop linear response model for DiLoCo
outer Nesterov, fits one low-parameter spectrum jointly to the raw tuning
ratios, corrected tuning ratios, and measured buffer gains, and tests the
specific claim that early bias-correction amplification against sharp modes
explains the corrected-arm drift.

The useful result is the recurrence and the resulting help-versus-hurt
criterion. The requested one-spectrum closure does **not** pass. A static
power-law spectrum explains the raw shape to 9.4% pointwise and the two buffer
gains to 10.4%, and it produces corrected drift in the right direction through
`T=10`. It then predicts flattening: `D_corr(T=20)=0.712` instead of `0.531`,
and at fixed `T=5` it predicts `D_corr(H=2048)=0.808` instead of `0.747`.
Those misses are much larger than the reported uncertainty. A two-point
spectrum fails in the same places. The discriminator therefore rejects a
single time-invariant spectral density, not the per-mode state formulation.

## 1. Empirical contract

Throughout, `M=4`, `mu=0.9`, and the model size is 135M. A learner performs
`H` inner steps per outer round, `T=S/H` rounds per fragment, and the syncer
uses the production post-buffer Nesterov rule. The normalized tuning ratio is

$$
D_{a}(T,H,S)
=\frac{\eta^*_{a}(\mu,T,H,S)}{\eta^*_0(T,H,S)(1-\mu)},
\qquad a\in\{\mathrm{raw},\mathrm{corr}\}.
$$

The observations this lane treats as mandatory are:

| family | `(H,T,S)` | observed quantity |
|---|---:|---:|
| raw `T` scan | `(512,2,1024)` | `D_raw=4.16746` |
| raw `T` scan | `(512,5,2560)` | `D_raw=2.56682` |
| raw `T` scan | `(512,10,5120)` | `D_raw=1.58834` |
| raw `T` scan | `(512,20,10240)` | `D_raw=0.989847` |
| fixed `S` | `(64,40,2560)` | `D_raw=0.996453` |
| fixed `S` | `(16,160,2560)` | `D_raw=1.03639` |
| corrected `T` scan | `(512,2,1024)` | `D_corr=0.955266` |
| corrected `T` scan | `(512,5,2560)` | `D_corr=0.861517` |
| corrected `T` scan | `(512,10,5120)` | `D_corr=0.748001` |
| corrected `T` scan | `(512,20,10240)` | `D_corr=0.530626` |
| corrected fixed `T=5` | `(1024,5,5120)` | `D_corr=0.809865` |
| corrected fixed `T=5` | `(2048,5,10240)` | `D_corr=0.747421` |
| raw fixed `T=5` diagnostic | `(1024,5,5120)` | `D_raw=2.55320` |
| raw fixed `T=5` diagnostic | `(2048,5,10240)` | `D_raw=2.37478` |
| buffer norm | `(16,160,2560)` | `G_B=4.27980` |
| buffer norm | `(512,5,2560)` | `G_B=3.51446` |

For the fixed-`H` raw scan, the registered open-loop ramp estimate is

$$
D_{\rm ramp}(T)=\frac1{1-\mu^{T+1}}.
$$

It gives `3.6900, 2.1342, 1.4573, 1.1229` at
`T=(2,5,10,20)`. The observed relative residuals are respectively
`+12.94%, +20.27%, +8.99%, -11.85%`, reproducing the stated
`+13/+20/+9/-12%` comparison. This law is a useful finite-ramp baseline, not
the closed-loop terminal-loss theory derived below.

The corrected `T` scan is nearly log-linear:

$$
\log_2 D_{\rm corr}
=(-0.0660,-0.2150,-0.4189,-0.9142),
$$

with a through-origin slope of `-0.04349` bits per round. The fixed-`T` row
is age-dominant in the empirical decomposition: age-only log-RMSE `0.108`
versus duration-only log-RMSE `0.249`.

The scalar AR(1) closure is already rejected by independent measurements.
Tuning residuals require an effective `rho` around `0.994`, direct buffer
norms give a common steady inversion around `0.519` (and incompatible
per-`H` inversions), while direct telemetry is around `0.69`. The measured
buffer gains are `4.28` and `3.51`, not the asymptotic coherent limit `10`.
For context, the finite coherent buffer gain at `T=5` is `4.0951`, so the
`H512,T5` observation is `0.858` of even that finite ramp.

Finally, correction at `mu=0` must be an exact A/A identity. This is both an
algebraic constraint and an implementation constraint, not another fitted
point.

## 2. Code-true per-mode state

The mode decomposition assumes that the linearized inner response is
diagonalizable with real nonnegative eigenvalues in the evaluation-loss metric,
that the evaluation quadratic shares that local eigenbasis, and that forcing
is uncorrelated across modes. The fitted closure also uses one underlying
`kappa` basis across `H`, with `H` changing eigenvalues but not eigenvectors.
These assumptions are what make a scalar spectral density sufficient. They are
not established for AdamW training; cross-mode rotation or nonnormal coupling
is one possible explanation for the failed fit.

Let `e_t` be the error along one local eigenmode just after outer round `t`.
Linearize the `H`-step inner map around the current trajectory:

$$
\delta_t=\lambda_H e_{t-1}+\xi_t.
$$

Here `delta_t` is the merged pseudo-gradient, `lambda_H` is an eigenvalue of
the linearized inner-training response `A_H`, and `xi_t` contains minibatch,
worker, merge, and linearization error. This is a closed loop because the next
pseudo-gradient depends on the parameter error produced by the previous outer
updates.

The production syncer performs

$$
b_t=\mu b_{t-1}+\delta_t,
\qquad
d_t=\delta_t+\mu b_t=(1+\mu)\delta_t+\mu^2b_{t-1},
$$

$$
e_t=e_{t-1}-\eta q_t d_t,
\qquad b_0=0.
$$

The arm-specific scalar is

$$
q_t=\begin{cases}
1,&\text{raw},\\
(1-\mu^{t+1})^{-1},&\text{corrected},
\end{cases}
\qquad t=1,\ldots,T.
$$

The indexing matters. Production calls the first per-fragment commit `t=1`,
so at `mu=0.9` the first factors are

$$
q_1=5.2632,\quad q_2=3.6900,\quad q_3=2.9078,
\quad q_4=2.4419,\quad q_5=2.1342.
$$

The factor is folded into the learning rate. It does not alter `b_t`.

With `x_t=(e_t,b_t)^T` and `alpha_t=eta*q_t`, the deterministic state matrix
for one mode is exactly

$$
x_t=M_t(\lambda_H)x_{t-1}+n_t\xi_t,
$$

$$
M_t(\lambda)=
\begin{bmatrix}
1-\alpha_t(1+\mu)\lambda&-\alpha_t\mu^2\\
\lambda&\mu
\end{bmatrix},
\qquad
n_t=\begin{bmatrix}-\alpha_t(1+\mu)\\1\end{bmatrix}.
$$

This is the requested 2x2 block iteration. For the raw arm the matrix is
constant in time. Its trace and determinant are

$$
\operatorname{tr}M=1+\mu-\eta(1+\mu)\lambda,
\qquad
\det M=\mu(1-\eta\lambda).
$$

The scalar Jury conditions give the exact raw per-mode stability interval

$$
0<\eta\lambda<\frac{2(1+\mu)}{1+2\mu}.
$$

At `mu=0.9`, the upper bound is `1.35714`. For a corrected round, replace
`eta` by `eta*q_t` in this one-matrix condition. This is only a local
per-round check: the corrected matrices vary and need not commute, so their
product can have substantial transient amplification even when each matrix
separately passes its Jury test. Stability is necessary for useful momentum,
but it is not a benefit criterion.

### Constant-gradient identity and its limit

If `delta_t=delta` is literally constant, then

$$
d_t=\frac{1-\mu^{t+1}}{1-\mu}\delta.
$$

The correction cancels this scalar exactly, so every corrected displacement
is `eta*delta/(1-mu)`. That identity does **not** survive by substitution when
`delta_t=lambda_H e_{t-1}+xi_t`: changing the step changes future deltas and
therefore changes the buffer. The 2x2 matrix is the missing feedback term in
the open-loop finite-ramp argument.

## 3. Finite-horizon terminal loss

For arbitrary zero-mean forcing, define the state transition product

$$
\Phi_{T:s}(\lambda)=M_T(\lambda)M_{T-1}(\lambda)\cdots M_s(\lambda),
\qquad \Phi_{T:T+1}=I.
$$

Then

$$
x_T=\Phi_{T:1}x_0+
\sum_{s=1}^{T}\Phi_{T:s+1}n_s\xi_s.
$$

Let `r=(1,0)^T`. For initial covariance `P_0` and a forcing covariance kernel
`K_xi(s,u)`, the exact expected squared terminal error of this mode is

$$
\mathbb E[e_T^2]
=r^T\Phi_{T:1}P_0\Phi_{T:1}^Tr
+\sum_{s,u=1}^{T}
r^T\Phi_{T:s+1}n_s K_\xi(s,u)n_u^T\Phi_{T:u+1}^Tr.
$$

Let `nu_H(d lambda)` be the evaluation-loss-weighted spectral measure of the
inner response. The finite-horizon terminal loss is

$$
\mathcal L_{a,T}(\eta;H)
=\mathcal L_{\rm floor}
+\frac12\int \mathbb E[e_T(\lambda)^2]\,\nu_H(d\lambda),
$$

and the theory-defined tuned rate is

$$
\eta^*_{a}(H,T)=\arg\min_{\eta>0}\mathcal L_{a,T}(\eta;H).
$$

Substitution into the definition in Section 1 produces `D_raw` and `D_corr`.
This is a prediction of the optimum of the whole terminal loss, not a match to
the final direction norm or to a guessed scalar gain.

## 4. Checkable stochastic closure used for the fit

The general covariance formula above does not require AR(1). For a small
numerical model, this lane uses an AR coordinate only for the exogenous
forcing, while retaining the closed-loop `(e,b)` state. Set

$$
\xi_t=\rho_\xi\xi_{t-1}+\epsilon_t,
\qquad \rho_\xi=0.69\quad\text{(fixed from telemetry, not fitted)}.
$$

For `y_t=(e_t,b_t,xi_t)^T`, one mode obeys

$$
y_t=F_t(\lambda)y_{t-1}+v_t\epsilon_t,
$$

$$
F_t(\lambda)=
\begin{bmatrix}
1-\alpha_t(1+\mu)\lambda&-\alpha_t\mu^2&-\alpha_t(1+\mu)\rho_\xi\\
\lambda&\mu&\rho_\xi\\
0&0&\rho_\xi
\end{bmatrix},
\qquad
v_t=\begin{bmatrix}-\alpha_t(1+\mu)\\1\\1\end{bmatrix}.
$$

Thus the covariance recursion evaluated by the fitter is simply

$$
P_t=F_tP_{t-1}F_t^T+\tau_H^2v_tv_t^T.
$$

This use of `rho_xi` does not resurrect the rejected scalar AR(1) closure.
The observable pseudo-gradient is `lambda*e+xi`; its correlation is an output
of the feedback system. Likewise, the buffer gain is computed from `P_t`, not
set to the scalar stationary formula. In particular, no tuning-derived
`rho=0.994` is inserted anywhere.

Telemetry observes `delta`, not the latent `xi`, so assigning its approximate
lag to `rho_xi` is a favorable closure assumption rather than identification.
A post-fit output check gives terminal pseudo-gradient lag-1 values
`0.718, 0.696, 0.703, 0.707, 0.699, 0.693` for
`(H,T)=(512,2),(512,5),(512,10),(512,20),(64,40),(16,160)`. This reproduces
the coarse `about 0.69` fact, but not the full per-cell telemetry profile
(whose fitted values span roughly `0.555` to `0.713`). A stricter closure
would infer `rho_xi` by matching every output lag rather than copying a common
summary into the forcing state.

### Inner-map and spectral parameterization

Use an underlying relaxation rate `kappa` per inner step and

$$
\lambda_H(\kappa)=c_A(1-e^{-H\kappa}).
$$

The same scale `c_A` multiplies the forcing. It cancels from every `D` ratio
and buffer ratio, and is calibrated only when absolute `eta` is displayed.

The primary spectral family has three parameters:

$$
p(\kappa)=C\kappa^{\alpha},
\qquad \kappa\in[\kappa_{\min},\kappa_{\max}].
$$

Coordinate mass uses `p(kappa) d kappa`; terminal quadratic loss uses the
additional curvature factor `kappa`. The inner-map-consistent forcing scale is

$$
\operatorname{Var}(\xi_t\mid\kappa,H)
=\sigma^2(1-e^{-2H\kappa}),
$$

with stationary AR innovation variance equal to this quantity times
`1-rho_xi^2`. The initial conditions are `Var(e_0)=1`, `b_0=0`, and stationary
`xi_0`.

The fitted parameters are:

| quantity | fitted value | role |
|---|---:|---|
| `kappa_min` | `1.49537e-5` | lower edge of spectrum |
| `kappa_max` | `8.70819e-2` | upper edge of spectrum |
| `alpha` | `-1.96595` | power-law exponent |
| `sigma^2` | `0.398591` | forcing variance scale, outside the 3-parameter spectrum |
| `rho_xi` | `0.69` | measured telemetry anchor, fixed |
| `c_A` | `6.75324` | post-fit scale matching `eta0` at `H512,T5` |

`c_A` and `sigma^2` are nuisance scales, not extra shape parameters in the
spectral density. There is still only one spectrum shared by every raw,
corrected, fixed-`S`, and fixed-`T` cell.

### Fit objective and numerical audit

The fit used 24 log-spaced quadrature nodes and exact covariance propagation.
It minimized the equal-weight log error over all 16 quantities in Section 1:

$$
J(\theta)=\frac1{16}\sum_{i=1}^{16}
\left[\log\frac{\widehat y_i(\theta)}{y_i}\right]^2.
$$

Multiple bounded starts were used. The best retained solution has
`J=0.0088303`, or log-RMSE `0.09452` (`0.13637` bits). This is a local
numerical optimum, not a proof that no other power-law parameters exist.
All calculations were NumPy/SciPy CPU calculations; no GPU library or job was
used.

Two ablations are important:

1. A deterministic two-atom model can tune terminal cancellation, but then
   `delta_T` approaches zero while history remains. Its fitted buffer gains
   were about `12` and `196`, so it is structurally unusable.
2. With iid forcing, the long-age buffer gain lands on the independent-input
   floor `1/sqrt(1-mu^2)=2.294`, far below `4.28`. Telemetry-scale temporal
   forcing is required, but adding it still does not repair corrected drift.

## 5. Joint predictions

The following are in-sample predictions of the single primary spectrum.
They are shown pointwise because aggregate RMSE hides the decisive misses.

| arm/observable | `(H,T)` | observed | predicted | relative error |
|---|---:|---:|---:|---:|
| raw `D` | `(512,2)` | `4.16746` | `3.90831` | `-6.22%` |
| raw `D` | `(512,5)` | `2.56682` | `2.32742` | `-9.33%` |
| raw `D` | `(512,10)` | `1.58834` | `1.47566` | `-7.09%` |
| raw `D` | `(512,20)` | `0.989847` | `0.976922` | `-1.31%` |
| raw `D` | `(64,40)` | `0.996453` | `1.03908` | `+4.28%` |
| raw `D` | `(16,160)` | `1.03639` | `0.988734` | `-4.60%` |
| raw `D` | `(1024,5)` | `2.55320` | `2.32402` | `-8.98%` |
| raw `D` | `(2048,5)` | `2.37478` | `2.32395` | `-2.14%` |
| corrected `D` | `(512,2)` | `0.955266` | `0.906260` | `-5.13%` |
| corrected `D` | `(512,5)` | `0.861517` | `0.809085` | `-6.09%` |
| corrected `D` | `(512,10)` | `0.748001` | `0.744756` | `-0.43%` |
| corrected `D` | `(512,20)` | `0.530626` | `0.711510` | `+34.09%` |
| corrected `D` | `(1024,5)` | `0.809865` | `0.808174` | `-0.21%` |
| corrected `D` | `(2048,5)` | `0.747421` | `0.808260` | `+8.14%` |
| buffer gain | `(16,160)` | `4.27980` | `4.72336` | `+10.36%` |
| buffer gain | `(512,5)` | `3.51446` | `3.48436` | `-0.86%` |

Subset log-RMSE is `0.0859` bits for raw `D`, `0.1907` bits for corrected
`D`, and `0.1005` bits for buffer gains. The maximum error is the corrected
`T=20` point.

Using the reported 95% intervals as approximate log-normal standard errors
gives a diagnostic `chi^2=1183` for 12 nominal degrees of freedom. That is not
a formal likelihood calculation (bootstrap intervals are not independent
Gaussian errors, and two points have invalid-bootstrap qualifications), but
it makes the scale clear. The corrected `T=20` and fixed-`T` `H=2048` misses
are about 18 and 19 such standard errors. Most smaller percentage residuals
also fall outside the very narrow intervals. This fit is descriptive, not
statistical closure.

### The key discriminator

At fixed `H=512`, the observed and predicted corrected log ratios are

| `T` | observed `log2 D_corr` | predicted `log2 D_corr` |
|---:|---:|---:|
| 2 | `-0.0660` | `-0.1420` |
| 5 | `-0.2150` | `-0.3056` |
| 10 | `-0.4189` | `-0.4252` |
| 20 | `-0.9142` | `-0.4910` |

The model does generate downward drift. Within this closure the mechanism is
the proposed one: `q_t` changes the sharp-mode feedback and forcing transfer,
so correction is not a harmless DC rescaling. But the static model flattens
after `T=10` while the data continue almost log-linearly.

At fixed `T=5`, the model predicts `(0.8091,0.8082,0.8083)` over
`H=(512,1024,2048)`, whereas the observations are
`(0.8615,0.8099,0.7474)`. Increasing spectral resolution therefore does not
produce the observed age dependence.

### Spectrum-family sensitivity

A two-point spectral density was also fitted jointly, with three spectral
parameters plus one noise scale. Its log-RMSE was `0.146` bits. It predicted
`D_corr(H512,T20)=0.721`, `D_corr(H2048,T5)=0.778`, and buffer gains
`4.76/3.84`. The power-law band improves the short buffer prediction but does
not move the long corrected point. The failure is not just two-atom
discretization.

## 6. Absolute-rate check

The `D` and buffer objectives cannot identify a common pseudo-gradient scale.
Calibrating `c_A=6.75324` to the measured `eta0` at `(H,T)=(512,5)` gives:

| `(H,T)` | observed `eta0*` | predicted `eta0*` |
|---:|---:|---:|
| `(512,2)` | `0.077665` | `0.077939` |
| `(512,5)` | `0.043685` | `0.043685` (calibration) |
| `(512,10)` | `0.032252` | `0.028745` |
| `(512,20)` | `0.027578` | `0.020042` |
| `(64,40)` | `0.019635` | `0.018104` |
| `(16,160)` | `0.023062` | `0.014302` |
| `(1024,5)` | `0.044386` | `0.043077` |
| `(2048,5)` | `0.051691` | `0.042526` |

The same scale works at short and moderate horizons but underpredicts the
long-age `eta0` values by up to 38%. This is independent evidence against a
time-invariant inner spectrum/noise closure.

## 7. Where momentum helps and where it hurts

The theory gives a precise finite-horizon criterion. Let

$$
\mathcal L^*_{a}(H,T,\mu)=\min_{\eta>0}\mathcal L_{a,T}(\eta;H).
$$

Then momentum helps exactly when

$$
\Delta_a(H,T,\mu)
=\mathcal L^*_{a}(H,T,\mu)-\mathcal L^*_0(H,T)<0,
$$

and hurts when `Delta_a>0`. In transfer-function form, the comparison splits
into a signal term and a forcing term:

$$
p_{a,T}(\kappa,\eta)=r^T\Phi^{a}_{T:1}r,
\qquad
r_{a,T,s}(\kappa,\eta)=r^T\Phi^{a}_{T:s+1}n_s,
\qquad r=(1,0)^T.
$$

For a **common** candidate learning rate define
`delta_a(eta)=L_a(eta)-L_0(eta)`. Then

$$
2\delta_a(\eta)
=\int \kappa\left[
(p_{a,T}^2-p_{0,T}^2)\operatorname{Var}(e_0)
+\sum_{s,u}(r_{a,T,s}r_{a,T,u}-r_{0,T,s}r_{0,T,u})K_\xi(s,u)
\right]p(\kappa)d\kappa.
$$

The tuned `Delta_a` is obtained by evaluating the two full integrals at their
own minimizers and subtracting; it is not the common-`eta` expression above at
one shared rate.

The first term is the acceleration opportunity on persistent low modes. The
second is the history/noise cost, weighted by curvature. Bias correction
changes both terms through the noncommuting early matrices. A large buffer norm
alone is neither help nor harm; its phase and curvature weighting decide the
sign.

For the fitted spectrum, the first low-noise sign crossing in `sigma^2` is:

| `(H,T)` | raw crossing | corrected crossing |
|---:|---:|---:|
| `(512,2)` | `0.0276` | `0.0382` |
| `(512,5)` | `0.0183` | `0.0278` |
| `(512,10)` | `0.0119` | `0.0189` |
| `(512,20)` | `0.00959` | `0.0152` |
| `(64,40)` | `0.0123` | `0.0167` |
| `(16,160)` | `0.00798` | `0.00954` |
| `(1024,5)` | `0.0162` | `0.0261` |
| `(2048,5)` | `0.0121` | `0.0222` |

Below these first crossings, deterministic signal acceleration dominates in
this fitted family. Above them, forcing/history initially dominates. The loss
ratio is not globally monotone in noise scale, so these are local regime
boundaries, not universal iff thresholds. The fitted `sigma^2=0.399` is far
inside the history/noise-dominated regime.

At the fitted parameters, the predicted tuned terminal-loss excess of raw
momentum over `mu=0` is `+0.13%, +1.02%, +2.49%, +4.37%` at
`H512,T=(2,5,10,20)`. The corresponding corrected excess is
`+0.25%, +1.28%, +2.50%, +3.20%`. Thus the model predicts near-neutrality at
`T=2` and increasing harm thereafter.

The fitted quadratic minima in the actual v3 loss curves have raw
`mom-minus-SGD` gaps `+0.00085, +0.00570, +0.01289, +0.02797` and corrected
gaps `-0.00044, +0.00555, +0.01789, +0.02952` at `T=(2,5,10,20)`.
The `T=2` signs are practically neutral; from `T=5` onward both arms hurt,
with the harm growing with age. The model gets this sign pattern, but its
corrected tuning ratio still fails quantitatively.

## 8. Why scalar AR(1) remains rejected

The distinction among three objects must remain explicit:

1. `rho_xi=0.69` is a fixed forcing coordinate used inside a state-feedback
   model.
2. `||b_T||/||delta_T||` is an output of that model. The fit predicts
   `4.72/3.48`, close to but not equal to the measured `4.28/3.51`.
3. The `rho` obtained by inverting `D` through a scalar displacement law is
   not a process parameter. Its value near `0.994` is evidence that the scalar
   mapping is misspecified.

Equating these three quantities recreates the rejected closure. Keeping them
separate improves the buffer and raw predictions, but the corrected `T=20`
failure shows that separation alone is insufficient.

## 9. A/A identity

At `mu=0`,

$$
q_t=\frac1{1-0^{t+1}}=1,
\qquad b_t=\delta_t,
\qquad d_t=\delta_t.
$$

Therefore `M_t`, `n_t`, every covariance, every terminal loss, and every tuned
rate are identical in raw and corrected arms. The production branch multiplies
the f32 learning rate by exactly `1`, and the Rust regression test
`bias_correction_with_zero_momentum_is_bit_identical_to_off` checks bit
identity. Thus correction at `mu=0` is bit-identical to raw. Independently, all
100 matched v3 `(S,eta,seed)` A/A loss pairs in the local master snapshot
compare exactly equal.

## 10. What the model does not fit

The failed residual pattern is specific enough to constrain the next theory.

1. A single `A_H` spectrum is too stationary. The corrected data require an
   additional evolution with total training age, outer duration, or both.
2. The missing term is not a scalar buffer correlation: buffer gains and
   telemetry are already present at their measured scale.
3. The missing term is not simply more spectral support: two atoms and a
   continuous power-law band fail at the same corrected points.
4. Candidates include `A_{H,t}` changing along the training trajectory,
   eta-dependent inner responses, multiplicative/non-Gaussian forcing,
   cross-mode rotation, and nonnormal coupling between modes. None is
   identified by the present data, so none is silently added.
5. The fixed-`T` row is particularly discriminating. Any extension must lower
   `D_corr` from about `0.81` to `0.75` as `H` doubles from 1024 to 2048 while
   retaining raw `D` near `2.4` and the measured buffer norms.

Consequently this lane supplies a closed-loop calculation, not a closed
empirical law. A new fit with an age-dependent operator would be a new model
with at least one additional parameter and must be validated out of sample.

## 11. Lean and implementation boundary

Existing formalizations cover the algebra used as inputs:

- `LeanMechanism/FiniteHorizonOuter.lean` selects post-buffer textbook
  Nesterov and proves the finite constant-input multiplier.
- `LeanMechanism/QuadraticAlignment.lean` defines the correction scale and
  proves constant-input correction.
- `LeanMechanism/CorrectionCosts.lean` proves correction variance
  amplification in its stated zero-indexed scalar model. Its generic
  zero-indexed "first step" convention is not the production `t=1` ledger
  indexing used above.
- `LeanMechanism/StochasticBuffer.lean` proves the scalar correlation-limited
  buffer formula and its independent/coherent endpoints.

The new 2x2 recurrence follows by direct substitution, and the 3x3 covariance
recursion is standard matrix algebra. They have **not** been added as new Lean
theorems. No claim in this document upgrades the numerical spectral fit to a
machine-checked result.

## 12. Provenance and reproduction

The literal `/root/...` bundle was not mounted at those paths in this shell.
The byte-addressed local mirror used here is under
`/private/tmp/diloco-twoparam-inspect/data/`, with `g3-readout.json` under
`/private/tmp/theory-B-inputs/`. Key SHA-256 values are:

| input | SHA-256 |
|---|---|
| `master_D.csv` | `f1b132a5b4580a396da344a959f195c747d3b759d6db54243d303316eed77427` |
| `master_eta.csv` | `610f8e302d3a3ba7e888e268ad832d1be7e5015c0b93ab000208c57d0cf5d649` |
| `telemetry_rho.csv` | `f8520c7bde54fcddc63947b775ba76f4dffe911b71497c73e1de11c8e640de99` |
| `key_numbers.json` | `c4f6c89bceeafb4e2394e7b81d1840bff8af6ff39f72088e8ae8ef5f8777386b` |
| `g1-readout.json` | `c2dcd6b9ab7dce0dc28d1e2473a72c7e0bdb8d6221728f09503a6354a39cae2b` |
| `g1v2-readout.json` | `5d4eed9685f25fd1db3135319908a045a389300a008a67bc009cd178cabe2fc8` |
| `g3-readout.json` | `d4a3cde6aa47580dff255c7a66030ab997a95f4072b1883bf71aa54d7da744c8` |
| `h200-disambig-note.md` | `4be85f66125472fca267f284912442d5bf4ce6e25dbaae9dbca8f381e4af4835` |

The disposable CPU evaluator is
`/private/tmp/theory-A/fit_lane_a.py`, SHA-256
`9aecc5aa66e557db3a2cc920c41fdb5ea3b566221b96ee79bd94fb9dc9c5313b`.
The primary fit command is:

```bash
CUDA_VISIBLE_DEVICES="" /private/tmp/theory-A/.venv/bin/python \
  /private/tmp/theory-A/fit_lane_a.py \
  --spectrum power-law --include-buffer \
  --start -8 -3 -2.5 -1.5
```

The evaluator searches each `eta*` on a log grid, refines the selected basin,
propagates the covariance recursion above, and prints every observed/predicted
row. Reproduction should compare the prediction table, not only the scalar
objective.

## Closure verdict

The per-mode feedback theory is the right algebraic level: it explains why
constant-gradient correction is not trajectory correction, separates forcing
correlation from buffer and tuning inversions, predicts the raw horizon trend,
and gives a checkable finite-horizon help/hurt condition. The requested stronger
claim, that one static low-parameter spectrum explains raw `D`, corrected drift,
the fixed-`T` age row, and buffer gains jointly, is unsupported by both
spectral families attempted. The residual pattern points to an additional
age-dependent, cross-mode, or nonlinear state variable that these measurements
do not yet identify.

THEORY A PARTIAL Closed-loop spectral response explains the raw trend and buffer scale, but one static spectrum cannot reproduce the corrected long-age drift.
