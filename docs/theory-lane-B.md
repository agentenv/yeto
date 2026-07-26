# Theory lane B — finite-time underdamped/SDE closure

**Status: PARTIAL.** There is a checkable three-parameter reduced closure for
the measured learning-rate optima, but not a parameter-free microscopic
closure. The model gets the raw finite-age law at the accuracy actually seen in
the data, fits the corrected drift to about 1.3% in log scale, preserves the
exact `mu = 0` A/A identity, and gives a finite-risk criterion for when momentum
helps. It does **not** derive the two kinetic-clock coefficients from the
available lag telemetry, and the stationary scalar quadratic SDE from which one
would most naturally try to derive them fails at corrected `T = 20`.

The compact prediction is as follows. Let

\[
T=S/H,\qquad \epsilon=1-\mu,\qquad
m(\mu)=\frac{\mu}{1-\mu},\qquad H_0=512,
\]

and let \(\eta_0^*(H,S)\) be the independently tuned `mu = 0` optimum. Then

\[
\boxed{
\begin{aligned}
D_{\rm raw}(\mu,T,H,S)
  &=\frac{1}{1-\mu^{T+1}},\\[2mm]
\Lambda(\mu,T,H)
  &=\exp\!\left\{m(\mu)
      \left[\chi T+\psi\log(H/H_0)\right]\right\},\\[2mm]
D_{\rm corr}(\mu,T,H,S)
  &=\Lambda(\mu,T,H)^{-\alpha},\\[2mm]
\eta_{A}^{*}(\mu,H,S)
  &=\epsilon\,\eta_0^*(H,S)D_A(\mu,T,H,S),
  \qquad A\in\{\mathrm{raw},\mathrm{corr}\}.
\end{aligned}}
\tag{1}
\]

Only three scalar quantities are calibrated:

| quantity | value | calibration source | role |
|---|---:|---|---|
| \(\alpha\) | 0.454804 | log-log slope of the four interior `mu=0`, `H=512` v3 optima | local LR-versus-length elasticity |
| \(\chi\) | 0.00756933 | corrected-arm log fit | underdamped effective-time dilation per outer update and per unit \(m(\mu)\) |
| \(\psi\) | 0.0231492 | corrected fixed-`T` disambiguation fit | local-window/duration contribution to the same clock |

At the measured \(\mu=0.9\), (1) is equivalently

\[
D_{\rm corr}=\exp(-0.0309830T)(H/512)^{-0.0947549}.
\tag{2}
\]

The six corrected observations have natural-log RMSE 0.01285 under (2),
equivalent to a 1.29% multiplicative error. The raw formula has no fitted
coefficient. Its largest discrepancy among the four valid fixed-`H` v3 points
is the already reported +20.3% observation/prediction residual at `T=5`.

## 1. Evidence boundary and provenance

This analysis used CPU-only arithmetic. It launched no training and no GPU
work. The measurement files were read from `h200-n1` over read-only SSH; small
copies were used for local inspection. The relevant source hashes are:

| artifact on `h200-n1` | SHA-256 |
|---|---|
| `/root/two-param-analysis/REPORT.md` | `765c9ce0104c9bb8f82dcb3a1fb8cf8393f80152bfc2eea8f7b9c0c3b11f9f77` |
| `/root/two-param-analysis/data/master_D.csv` | `f1b132a5b4580a396da344a959f195c747d3b759d6db54243d303316eed77427` |
| `/root/two-param-analysis/data/master_eta.csv` | `610f8e302d3a3ba7e888e268ad832d1be7e5015c0b93ab000208c57d0cf5d649` |
| `/root/two-param-analysis/data/telemetry_rho.csv` | `f8520c7bde54fcddc63947b775ba76f4dffe911b71497c73e1de11c8e640de99` |
| `/root/g3-readout.json` | `d4a3cde6aa47580dff255c7a66030ab997a95f4072b1883bf71aa54d7da744c8` |
| `/root/g1-readout.json` | `c2dcd6b9ab7dce0dc28d1e2473a72c7e0bdb8d6221728f09503a6354a39cae2b` |
| `/root/g1v2-readout.json` | `5d4eed9685f25fd1db3135319908a045a389300a008a67bc009cd178cabe2fc8` |
| `/private/tmp/h200-disambig-note.md` | `4be85f66125472fca267f284912442d5bf4ce6e25dbaae9dbca8f381e4af4835` |

The empirical definition throughout is

\[
D=\frac{\eta^*(\mu)/\eta^*(0)}{1-\mu}
  =\frac{a^*(\mu)}{\eta^*(0)},
\qquad a\equiv\frac{\eta}{1-\mu}.
\tag{3}
\]

Thus \(D=1\) means that the conventional steady-state rule
\(\eta^*(\mu)=(1-\mu)\eta^*(0)\) is right. It does not, by itself, say that the
best momentum loss is better than the best memoryless loss.

## 2. Exact discrete filter before taking a continuum limit

For one fragment, production Nesterov is

\[
b_t=\mu b_{t-1}+g_t,\qquad
q_t=g_t+\mu b_t=(1+\mu)g_t+\mu^2b_{t-1},
\qquad b_0=0.
\tag{4}
\]

The parameter update is

\[
\theta_t=\theta_{t-1}-\eta c_tq_t,
\qquad
c_t=\begin{cases}
1,&\text{raw},\\
(1-\mu^{t+1})^{-1},&\text{corrected},
\end{cases}
\tag{5}
\]

where `t=1` is the human first update. This is the code-true `T+1`
Nesterov indexing, not the heavy-ball `T` indexing.

For a frozen pseudo-gradient \(g_t=g\),

\[
q_t=\frac{1-\mu^{t+1}}{1-\mu}g.
\tag{6}
\]

In normalized coordinates \(a=\eta/(1-\mu)\), the raw applied step is
\(-a r_tg\), with \(r_t=1-\mu^{t+1}\), while the corrected step is exactly
\(-ag\) at every age. Terminal-step matching to the `mu=0` optimum therefore
gives

\[
a^*_{\rm raw}r_T\simeq\eta_0^*,
\qquad
D_{\rm raw}^{(0)}=\frac{a^*_{\rm raw}}{\eta_0^*}
 =\frac{1}{r_T}.
\tag{7}
\]

Equation (7) is a mean-side, terminal-response approximation. It is not a
claim that the real buffer norm equals the aligned scalar coefficient, nor is
it an average-path matching claim. The distinction is essential for the
buffer evidence below.

The exact identities used here already have sorry-free Lean counterparts:

- [`FiniteHorizonOuter.lean`](../lean-mechanism/LeanMechanism/FiniteHorizonOuter.lean)
  proves the code-true terminal multiplier and accumulated coefficient.
- [`QuadraticAlignment.lean`](../lean-mechanism/LeanMechanism/QuadraticAlignment.lean)
  proves that correction makes the frozen-gradient multiplier exactly steady
  at every age; it also explicitly separates the exact evolving quadratic from
  the frozen-gradient approximation.
- [`CorrectionCosts.lean`](../lean-mechanism/LeanMechanism/CorrectionCosts.lean)
  proves the correction's finite-age variance multiplier and its decay to one.
- [`NadamEquivalence.lean`](../lean-mechanism/LeanMechanism/NadamEquivalence.lean)
  proves the constant-momentum product-correction identity.

## 3. SDE interpretation and the exact finite-risk quadrature

Write the pseudo-gradient as a state-dependent drift plus centered noise,

\[
g_t=m(\theta_{t-1},s_t)+\xi_t,qquad
\mathbb E\xi_t=0,qquad
K_{su}=\mathbb E[\xi_s\xi_u^\top].
\tag{8}
\]

No geometric or stationary form is imposed on \(K\). This is the point at
which the present model differs from the rejected scalar AR(1) closure.

Let \(v_t=(1-\mu)b_t\). Then

\[
v_t=\mu v_{t-1}+(1-\mu)g_t,
\qquad
\theta_t-\theta_{t-1}
  =-a c_t\{(1-\mu)g_t+\mu v_t\}.
\tag{9}
\]

With \(\gamma=-\log\mu\), (9) is the unit-step discretization of a Kramers-like
or underdamped Langevin system in which \(v\) relaxes toward the stochastic
pseudo-gradient:

\[
dv=-\gamma\{v-m(\theta,s)\}\,dt+G(\theta,s)\,dW_t,
\qquad
d\theta=-a c(t)\{(1-\mu)m+\mu v\}\,dt+\text{look-ahead noise}.
\tag{10}
\]

This is an analogy, not an assertion that DiLoCo samples a Gibbs law. The
look-ahead injection means it is not literally textbook underdamped Langevin.
What survives exactly is the two-state linear response and its finite-time
controllability Gramian.

For a scalar quadratic mode with curvature \(\lambda\), put
\(g_t=\lambda x_{t-1}+\xi_t\) and \(y_t=(x_t,b_t)^\top\). Equations (4)-(5)
give

\[
y_t=M_t(\eta,\lambda)y_{t-1}+u_t(\eta)\xi_t,
\tag{11}
\]

with

\[
M_t=
\begin{bmatrix}
1-\eta c_t(1+\mu)\lambda&-\eta c_t\mu^2\\
\lambda&\mu
\end{bmatrix},
\qquad
u_t=\begin{bmatrix}-\eta c_t(1+\mu)\\1\end{bmatrix}.
\tag{12}
\]

Define \(\Phi_{T:s+1}=M_T\cdots M_{s+1}\), with the empty product equal to
the identity. The terminal covariance is the finite double quadrature

\[
\Sigma_T(\eta)=
\sum_{s=1}^T\sum_{u=1}^T
 \Phi_{T:s+1}u_s K_{su}u_u^\top\Phi_{T:u+1}^\top.
\tag{13}
\]

For a curvature spectrum \(\{(w_j,\lambda_j)\}\), the predicted terminal risk
and optimal step are

\[
J_A(\eta)=\frac12\sum_jw_j\lambda_j
 \left[\{e_1^\top\Phi^{(j)}_{T:1}y_0^{(j)}\}^2
       +e_1^\top\Sigma_T^{(j)}(\eta)e_1\right],
\qquad
\eta_A^*=\arg\min_{\eta\ge0}J_A(\eta).
\tag{14}
\]

Equations (11)-(14) are the microscopic closed form requested by this lane:
only a one-dimensional minimization remains once the drift/Hessian spectrum
and measured covariance kernel are supplied. They also expose the finite-time
bias-temperature tradeoff. In a one-mode coarse approximation,

\[
J(a)\simeq B e^{-2\lambda a\tau}+a\mathcal T,
\qquad
a^*=\frac{[\log(2\lambda\tau B/\mathcal T)]_+}
            {2\lambda\tau},
\tag{15}
\]

where \(\tau\) is the drift clock and \(\mathcal T\) is the
curvature-weighted effective temperature obtained from (13). Momentum may
increase \(\tau\), \(\mathcal T\), or both. A scalar direction gain alone is
therefore insufficient to determine the optimum.

The available telemetry does not identify the full joint Hessian/noise
spectrum required by (14). The next section gives the reduced closure used for
actual predictions.

## 4. From the `mu=0` LR-length law to corrected drift

Over the four interior v3 `H=512`, `mu=0` points,

\[
\eta_0^*(512,S)\simeq A_{512}(S/2560)^{-\alpha},
\qquad \alpha=0.454804.
\tag{16}
\]

This is a local elasticity, not a proposed universal scaling exponent. Its own
four-point natural-log RMSE is 0.0796; individual fitted values differ from
the measured baseline optima by 7.1%, 8.9%, 7.6%, and 8.2%. The amplitude
\(A_H\) is not counted as a theory parameter because it cancels from every
\(D\) prediction; absolute predictions use the measured or separately modeled
\(\eta_0^*(H,S)\).

For a linear underdamped semigroup, phase-space bias contracts exponentially at
its hypocoercive rate. The minimal reduced closure is to express that contraction
as a *baseline-equivalent length dilation*. Define \(S_{\rm eff}=S\Lambda\)
and take the first-order separable kinetic clock

\[
\frac{d\log S_{\rm eff}}{dt}=m(\mu)\chi,
\qquad
\frac{d\log S_{\rm eff}}{d\log H}=m(\mu)\psi.
\tag{17}
\]

Integrating (17) gives the \(\Lambda\) in (1). Substituting the ordinary
baseline LR-length law (16) then yields

\[
\frac{a_{\rm corr}^*(H,S)}{\eta_0^*(H,S)}
\simeq
\frac{\eta_0^*(H,S\Lambda)}{\eta_0^*(H,S)}
=\Lambda^{-\alpha}=D_{\rm corr}.
\tag{18}
\]

This is how the duration drift enters: it is not an independent exponential
penalty pasted onto \(D\); it is the measured `mu=0` LR-versus-length
elasticity evaluated at an underdamped, baseline-equivalent training length.
At fixed `T`, increasing `H` also increases total duration `S=HT`, producing
the power-law factor in (2). At fixed `H`, optimizer age supplies the dominant
exponential factor.

The memory load \(m(\mu)=\mu/(1-\mu)\) is fixed by the relaxation time of the
momentum filter. It is not fitted. It makes \(\Lambda(0,T,H)=1\), which is
required by the exact A/A control. The form should not be extrapolated close to
`mu=1`: neither the finite-step stability boundary nor the local power law
(16) remains uniform in that limit.

Raw momentum does not receive (18) at leading order. Its uncorrected ramp
co-scales drift and stochastic forcing during warm-up; terminal mean matching
leaves (7), with the residual assigned to the finite curvature/temperature
terms of (14). The data support that ordering—the parameter-free ramp explains
the primary raw trend—but they also show that the residual is not zero.

## 5. Required numerical checks

### 5.1 Raw fixed-`H` scan

For `mu=0.9`, `H=512`, and `S=512T`:

| `T` | observed \(D_{\rm raw}\) | (7) | observation / prediction - 1 |
|---:|---:|---:|---:|
| 2 | 4.16746 | 3.69004 | +12.94% |
| 5 | 2.56682 | 2.13420 | +20.27% |
| 10 | 1.58834 | 1.45732 | +8.99% |
| 20 | 0.989847 | 1.12286 | -11.85% |

This reproduces the requested `+13/+20/+9/-12%` residual pattern after
rounding. It also states the failure plainly: all four narrow v3 confidence
intervals exclude the point prediction, and `T=5` misses by 20%.
The law is a useful leading term, not a statistically adequate full model.

The independent pilot `T=5` momentum sweep is a useful out-of-fit stress test:

| `mu` | observed raw `D` | (7) | observation / prediction - 1 |
|---:|---:|---:|---:|
| 0.5 | 1.01899 | 1.01587 | +0.31% |
| 0.8 | 1.52386 | 1.35528 | +12.44% |
| 0.95 | 4.61025 | 3.77489 | +22.13% |

The deterioration at `mu=0.95` is a warning against extrapolating the
relaxation-time closure toward zero friction.

### 5.2 Raw fixed-`S` long-horizon points

At `S=2560`, the model uses only `T=S/H`:

| cell | observed \(D_{\rm raw}\) | (7) | observation / prediction - 1 |
|---|---:|---:|---:|
| `H=64`, `T=40` | 0.996453 | 1.013482 | -1.68% |
| `H=16`, `T=160` | 1.036395 | 1.00000004 | +3.64% |

The `H=64` registered interval is formally not evaluable because four
bootstrap replicates are unbracketed; the finite point is retained only as the
requested descriptive check. Both points are near one because
\(T\gg(1-\mu)^{-1}\).

At fixed `T=5`, (7) has no leading-order `H` dependence. The additional raw
disambiguation points are 2.5532 (`H=1024`) and 2.37478 (`H=2048`) versus the
common prediction 2.1342. Those -16.4% and -10.1% prediction errors are further
evidence that the unmodeled term in (14) depends weakly on duration/window
geometry.

### 5.3 Corrected fixed-`H` drift and fixed-`T` disambiguation

The fit of (2) uses no intercept, so `mu=0` remains an exact identity:

| `H` | `S` | `T` | observed \(D_{\rm corr}\) | prediction | prediction / observation - 1 |
|---:|---:|---:|---:|---:|---:|
| 512 | 1024 | 2 | 0.955266 | 0.939915 | -1.61% |
| 512 | 2560 | 5 | 0.861517 | 0.856488 | -0.58% |
| 512 | 5120 | 10 | 0.748001 | 0.733571 | -1.93% |
| 512 | 10240 | 20 | 0.530626 | 0.538127 | +1.41% |
| 1024 | 5120 | 5 | 0.809865 | 0.802042 | -0.97% |
| 2048 | 10240 | 5 | 0.747421 | 0.751057 | +0.49% |

For the four fixed-`H` points, the fitted per-update multiplier is
\(e^{-0.030983}=0.969492\), close to the independently reported through-origin
value 0.970304. The observed free-intercept log-linear fit has \(R^2=0.9979\);
the closure deliberately forgoes that intercept to preserve exact A/A
behavior.

For the two new fixed-`T` points, the three competing predictions have
natural-log RMSE:

| decomposition | predictions at `H=1024,2048` | log-RMSE |
|---|---|---:|
| age only | 0.8615, 0.8615 | 0.1079 |
| duration only | 0.7480, 0.5306 | 0.2491 |
| mixed kinetic clock (2) | 0.8020, 0.7511 | 0.00767 |

The fitted log contributions at `T=5` are 0.1549 from optimizer age and
0.0657/0.1313 from doubling/quadrupling `H` and `S`. Thus age is the larger
term at both new points, while duration is measurably nonzero—exactly the
qualitative outcome of the disambiguation lane.

### 5.4 Scalar AR(1) is rejected, not repaired

If one imposes

\[
\frac{\mathbb E[g_t\cdot g_{t-k}]}{\mathbb E\|g_t\|^2}=\rho^k,
\tag{19}
\]

the same scalar \(\rho\) is forced to control current-gradient projection,
buffer norm, diffusion temperature, and curvature-weighted terminal response.
The measured inversions are mutually incompatible:

| probe | equivalent \(\rho\) or direct statistic |
|---|---:|
| corrected log slope under the rejected multiplicative mapping | 0.9935 `[0.9932, 0.9938]` |
| conditionally fitted raw residual | 0.9944 `[0.9928, 0.9963]`, but \(\chi^2=1036.5/10\) |
| common steady buffer-norm inversion | 0.5191 `[0.5098, 0.5285]` |
| lag-1--4 telemetry summary | 0.6973 `[0.6122, 0.7002]` |

The common-correlation test is \(\chi^2=10132.2\) on three degrees of freedom.
Calling the tuning inversion a physical correlation would therefore be a
category error.

The direct norm facts are also retained rather than replaced by the ideal
coefficient:

| cell | aligned ideal at terminal age | measured \(\|b_T\|/\|g_T\|\) | measured applied-direction factor |
|---|---:|---:|---:|
| `H=512`, `T=5` | 4.0951 for the buffer | 3.514 `[3.434,3.548]` | 3.917 |
| `H=16`, `T=160` | 10.0000 for the buffer | 4.280 `[4.256,4.368]` | 4.453 |

`H=16` is coefficient-age-saturated but not norm-saturated: its measured gain
peaks near 5.08 at round 20 and declines to 4.28 by round 160 as old directions
rotate. The ratio of measured buffer gains is 0.821, not the aligned
`4.0951/10 = 0.4095` ratio.

The non-AR(1) SDE reproduces this distinction structurally. For a general
nonstationary full-vector covariance \(C_{ij}=\mathbb E[g_i g_j^\top]\),

\[
\mathbb E\|b_T\|^2
=\sum_{i=0}^{T-1}\sum_{j=0}^{T-1}
 \mu^{i+j}\operatorname{tr}C_{T-i,T-j}.
\tag{20}
\]

Equation (20), with the actual lag-dependent norms and cosines, is a
quadrature rather than a scalar persistence. The retained H512 projected-lag
reconstruction agrees with the exact tape gain to 0.153% median relative error.
That is a telemetry-conditioned reconstruction, not an independent prediction.
At H16, the available short-lag summary is insufficient to predict the full
160-step tail; the exact measured buffer norm must remain an input to (13).

[`StochasticBuffer.lean`](../lean-mechanism/LeanMechanism/StochasticBuffer.lean)
proves the geometric-kernel formula and its bounds. Its module header correctly
states that it is a scalar surrogate and does not assert a geometric covariance
kernel for arbitrary training deltas. The empirical rejection concerns that
extra modeling assumption, not the Lean algebra.

### 5.5 `mu=0` A/A identity

At \(\mu=0\),

\[
b_t=g_t,\qquad q_t=g_t,qquad
1-\mu^{t+1}=1,qquad m(0)=0,qquad \Lambda=1.
\tag{21}
\]

Therefore raw and corrected recurrences, their noise kernels, and both
predictions are identical: \(D_{\rm raw}=D_{\rm corr}=1\). This is stronger
than equality in distribution. The production regression test
`bias_correction_with_zero_momentum_is_bit_identical_to_off` verifies bit
identity, and the g3 readout contains exactly equal raw/corrected `mu=0` loss
curves and eta optima at every `S`.

## 6. Where momentum helps and where it hurts

There are two different questions that should not be merged.

### 6.1 Is the conventional momentum LR scaling conservative or aggressive?

This question is answered by \(D\):

- \(D>1\): `eta=(1-mu) eta0` is conservative; the tuned momentum LR is larger.
- \(D<1\): that rule is aggressive; the tuned momentum LR must be smaller.
- \(D\simeq1\): the steady scaling is adequate.

For raw momentum, (7) is above one for every finite `T` and approaches one on
the relaxation scale \(T\sim(1-\mu)^{-1}\). Empirically it is conservative at
`T=2,5,10` and effectively neutral by `T=20,40,160`; the slight valid
`T=20` value below one is one of the leading-law residuals.

For corrected momentum, the calibrated boundary \(D_{\rm corr}=1\) is

\[
\chi T+\psi\log(H/512)=0,
\qquad
H_{\rm boundary}(T)=512\exp(-\chi T/\psi).
\tag{22}
\]

At `mu=0.9`, (22) gives approximate boundary windows 266, 100, 19, and 0.74
for `T=2,5,10,20`. Every measured corrected cell (`H>=512`) lies on the
aggressive side, and its distance from the boundary grows with age. These
small-`H` boundary values are extrapolations, not validated corrected-arm
predictions.

### 6.2 Does momentum improve the best attainable loss?

`D` cannot answer this. For a local quadratic step with normalized direction
\(z\),

\[
\mathbb E[\Delta L]
\simeq-aP+\frac{a^2}{2}Q,
\quad
P=\nabla L^\top\mathbb E z,
\quad
Q=\mathbb E[z^\top(\nabla^2L)z].
\tag{23}
\]

The optimal local reduction is \(P^2/(2Q)\). Momentum helps locally iff

\[
\boxed{\frac{P_\mu^2}{Q_\mu}>\frac{P_0^2}{Q_0}},
\tag{24}
\]

and hurts when the inequality reverses. The finite-horizon version is the
comparison of the two minima in (14). In the coarse bias-temperature model
(15), with \(z_A=2\lambda\tau_AB/\mathcal T_A>1\),

\[
J_A^*=\frac{\mathcal T_A}{2\lambda\tau_A}
       [1+\log z_A].
\tag{25}
\]

Thus momentum helps when its drift-clock gain beats its effective-temperature
cost, including the slowly varying logarithmic term. Correction sets the mean
clock to its steady value immediately, but the fitted \(\Lambda>1\) says that
its curvature-weighted finite-time temperature/path cost grows with age. Raw
momentum suppresses that early cost through the ramp, at the price of a smaller
short-horizon drift clock.

The current telemetry measures Euclidean norms and a few lag cosines, not the
curvature-weighted \(Q\) in (23). Consequently (24) is a real, predictive phase
criterion once a Hessian-vector or loss-response instrument supplies \(Q\),
but the existing fact list does not support a numerical best-loss phase map.
Claiming one from `D` or from buffer norms alone would overstate the evidence.

## 7. What failed and what remains unidentified

The following limitations are part of the verdict:

1. **The raw formula is only leading order.** It reproduces the registered
   finite-age pattern but misses narrow point intervals and has a 20.3% residual
   at `T=5`. It has no leading `H` dependence at fixed `T`, while the new raw
   disambiguation row shows a modest decline with `H`.

2. **A stationary scalar quadratic SDE does not explain corrected `T=20`.** A
   CPU-only exact covariance recursion using (11)-(14), one scalar curvature,
   white noise, and a single stationary noise-to-signal ratio had best joint
   fixed-`H` fit \(\nu=0.4083\), natural-log RMSE 0.1024. Its
   raw predictions were 4.110/2.347/1.437/0.963 and corrected predictions were
   0.958/0.830/0.736/0.682 at `T=2/5/10/20`. The last value is 28.5% above the
   measured 0.531. Fixing an OU color near the measured 0.69 does not remove
   this failure.

3. **Allowing temperature drift still does not microscopically close.** A
   three-parameter scalar extension
   \(\nu(H,S)=\nu_0(H/512)^{-\beta}(S/2560)^{-\delta}\) reduced joint log-RMSE
   to 0.0669, but still predicted corrected `T=20` as 0.626 (+18.0%) and missed
   the two held-out fixed-`T` raw points by -8.7% and -14.4%. This is why that
   model is not presented as the answer.

4. **The successful clock coefficients are calibrated, not telemetry-derived.**
   Equations (17)-(18) compress curvature spectrum, state-dependent noise, and
   nonnormal transient response into \(\chi\) and \(\psi\). Deriving them from
   (13) requires the joint Hessian/noise spectrum that the current telemetry
   does not contain.

5. **The corrected fit is in-sample.** `alpha` is fit only from the `mu=0`
   baseline, but `chi` and `psi` use all six corrected observations. The model
   therefore needs a new corrected cell—preferably `mu=0.8` or `0.95`, or
   fixed `H` with a new `T`—for genuine prospective validation.

6. **The model is local to the 135M, M=4 regime.** Model scale, worker count,
   asynchronous delay, and non-version-matched anchoring are not arguments of
   (1). No invariance to them is claimed.

These failures are why the lane is `PARTIAL`, despite the accurate reduced
corrected law.

## 8. Dependency-free numerical check

The following Python uses only the standard library. It refits all three
reported parameters from the point values above and prints the two prediction
tables. It intentionally uses no confidence-interval weighting; this matches
the reduced closure's stated unweighted log-RMSE estimand.

```python
from math import exp, log, sqrt

# Independent mu=0 baseline calibration at H=512.
S = [1024.0, 2560.0, 5120.0, 10240.0]
eta0 = [0.07766526318797157, 0.04368465066903625,
        0.03225158464590506, 0.027577613353225165]
x = [log(s / 2560.0) for s in S]
y = [log(e) for e in eta0]
xbar, ybar = sum(x) / len(x), sum(y) / len(y)
slope = sum((a-xbar)*(b-ybar) for a, b in zip(x, y)) / \
        sum((a-xbar)**2 for a in x)
alpha = -slope

# Corrected observations: solve -log(D) = kappa*T + p*log(H/512).
corr = [(512.0, 2.0, 0.9552656777062855),
        (512.0, 5.0, 0.8615170030442565),
        (512.0, 10.0, 0.7480014394580564),
        (512.0, 20.0, 0.5306260060759194),
        (1024.0, 5.0, 0.809865),
        (2048.0, 5.0, 0.747421)]
a11 = sum(t*t for h, t, d in corr)
a12 = sum(t*log(h/512.0) for h, t, d in corr)
a22 = sum(log(h/512.0)**2 for h, t, d in corr)
b1 = sum(t*(-log(d)) for h, t, d in corr)
b2 = sum(log(h/512.0)*(-log(d)) for h, t, d in corr)
det = a11*a22 - a12*a12
kappa = (b1*a22 - b2*a12) / det
p = (a11*b2 - a12*b1) / det
chi = kappa / (9.0*alpha)       # m(0.9) = 9
psi = p / (9.0*alpha)

print("alpha chi psi =", alpha, chi, psi)
print("kappa p q =", kappa, p, exp(-kappa))

corr_err = []
for h, t, obs in corr:
    pred = exp(-kappa*t) * (h/512.0)**(-p)
    corr_err.append(log(pred/obs))
    print("corr", int(h), int(t), "obs", obs, "pred", pred)
print("corrected log-RMSE =",
      sqrt(sum(e*e for e in corr_err)/len(corr_err)))

raw = [(2, 4.167462537063047), (5, 2.566821721877774),
       (10, 1.5883366898315632), (20, 0.989846613037108),
       (40, 0.9964531442050868), (160, 1.0363947372772233)]
for t, obs in raw:
    pred = 1.0 / (1.0 - 0.9**(t+1))
    print("raw", t, "obs", obs, "pred", pred,
          "obs/pred-1", obs/pred-1.0)
```

Expected leading output is

```text
alpha chi psi = 0.4548038032 0.00756932970 0.0231491511
kappa p q = 0.0309830394 0.0947548977 0.9694920161
corrected log-RMSE = 0.0128517756
```

## 9. Checkable claims versus interpretation

The status of each layer is:

| layer | status |
|---|---|
| code-true Nesterov `T+1` multiplier | exact, source-traced, Lean-formalized |
| correction mean multiplier and `mu=0` identity | exact, source-tested, Lean-formalized |
| arbitrary-kernel finite-risk quadrature (11)-(14) | exact for the stated linearized SDE |
| raw terminal law (7) | parameter-free leading approximation; 9--20% short-horizon residuals |
| baseline LR elasticity (16) | measured local fit; 0.0796 log-RMSE |
| corrected kinetic clock (17)-(18) | three-parameter reduced closure; 0.0129 log-RMSE, in-sample |
| scalar AR(1) interpretation | rejected |
| numerical best-loss help/hurt phase map | not identified without curvature-weighted direction statistics |

The strongest defensible claim is therefore narrower than a universal theory:
finite-age mean response explains the raw `D` trend; a nonstationary
underdamped effective clock, passed through the ordinary `mu=0`
LR-versus-length elasticity, explains the corrected log drift and its weaker
duration component. The present data do not reduce that clock to a stationary
scalar correlation or to a single measured buffer gain.

THEORY B PARTIAL parameter-free finite-age response closes the raw trend, while a three-parameter non-AR(1) kinetic clock fits corrected drift but is not yet derived from telemetry.
