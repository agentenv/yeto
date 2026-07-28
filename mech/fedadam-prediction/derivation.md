# FedAdam finite-age prediction

## Result

For the uncorrected FedAdam server recursion of Reddi et al. (2021), with both
moments initialized to zero, a constant nonzero scalar pseudo-gradient `g`, and
one-indexed outer age `t`, define

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g^2,
\]

with `m_0=v_0=0`. The exact unrolling is

\[
m_t=(1-\beta_1^t)g,\qquad
v_t=(1-\beta_2^t)g^2.
\]

Let

\[
a_t=1-\beta_1^t,\qquad
s_t=\sqrt{1-\beta_2^t},\qquad
q=\frac{\tau}{|g|},
\]

where `tau` is the additive denominator floor. Ignoring the sign convention for
the model delta, which does not affect a multiplier, the applied direction is

\[
u_t=\frac{m_t}{\sqrt{v_t}+\tau}
    =\operatorname{sign}(g)\frac{a_t}{s_t+q}.
\]

Its steady-age value is

\[
u_\infty=\frac{\operatorname{sign}(g)}{1+q}.
\]

Consequently the finite-age gain relative to the steady-age gain and the
inverse tuned-rate deviation are

\[
R_{\rm raw}(t;q)
  =\frac{u_t}{u_\infty}
  =\frac{a_t(1+q)}{s_t+q},
\]

\[
\boxed{
D_{\rm Adam}(t;q)
  =R_{\rm raw}(t;q)^{-1}
  =\frac{\sqrt{1-\beta_2^t}+q}
         {(1-\beta_1^t)(1+q)}.}
\]

This is the terminal-call matching heuristic: a tuned rate is predicted to
move inversely to the direction multiplier at age `T`. It is the adaptive
analogue of the terminal-multiplier prediction in
`FiniteHorizonOuter.lean`. It is not an assertion that a nonlinear training
trajectory has a globally optimal rate determined by its last call alone.

With no floor,

\[
\boxed{
D_{\rm Adam}(T;0)
  =\frac{\sqrt{1-\beta_2^T}}{1-\beta_1^T}.}
\]

The square root is the qualitative difference from a momentum-only geometric
startup law.

## Registered constants and numerical curve

The preregistered v18 prediction uses the FedAdam constants fixed by Reddi et
al. for their experiments,

\[
\beta_1=0.9,\qquad \beta_2=0.99,
\]

and the no-floor, zero-safe convention described below. The registered curve
is therefore

| `T` | `1-beta1^T` | `sqrt(1-beta2^T)` | registered `D_adam(T;0)` | normalized age gain `1/D` |
|---:|---:|---:|---:|---:|
| 2 | 0.190000000000 | 0.141067359797 | **0.742459788403** | 1.346874289516 |
| 5 | 0.409510000000 | 0.221381910056 | **0.540601963459** | 1.849789803948 |
| 20 | 0.878423345409 | 0.426723637033 | **0.485783579482** | 2.058529852053 |
| 40 | 0.985219117059 | 0.575350537873 | **0.583982312067** | 1.712380630948 |

The values are evaluated directly from the boxed formula, without a fit or an
outcome-dependent constant.

For comparison, the prior code-true Nesterov terminal heuristic at `mu=0.9`
is `1/(1-mu^(T+1))`, giving `3.6900, 2.1342, 1.1229, 1.0135` at the same four
ages. That curve is above one and decreases toward one. Raw FedAdam instead is
below one at all four registered ages and is U-shaped:

\[
D(2)>D(5)>D(20)<D(40)<1.
\]

Thus the registered prediction is not merely a different decay constant for
the Nesterov shape. It reverses the side of one and reverses direction between
`T=20` and `T=40`. Lean proves the exact rational inequalities after squaring
the nonnegative values, and then proves the corresponding unsquared U-order in
`AdaptiveFiniteAge.lean`.

## Why the curve is non-monotone

Write `A=-log(beta1)>0` and `B=-log(beta2)>0` and extend age continuously. For
zero floor,

\[
\frac{d}{dt}\log D(t;0)
=\frac{B\beta_2^t}{2(1-\beta_2^t)}
 -\frac{A\beta_1^t}{1-\beta_1^t}.
\]

The first-moment denominator initially catches up faster than the square-root
second moment, so the tuned deviation falls. At large age,

\[
D(t;q)-1
  =\beta_1^t-
    \frac{\beta_2^t}{2(1+q)}
    +o(\beta_1^t+\beta_2^t).
\]

Here `beta2=0.99 > beta1=0.9`; the negative, more slowly decaying second-moment
term eventually dominates. The curve therefore approaches one from below and
must turn upward. For the registered points the turn is already visible by
`T=40`.

This conclusion is parameter-dependent. It is not a theorem that every pair
of Adam betas produces the same four-point order. The exact formula, not the
word "Adam", is the prediction.

## Additive epsilon floor

The floor cannot be represented by `tau` alone in a scale-free prediction: its
dimensionless strength is `q=tau/|g|`, coordinate by coordinate. At a fixed
age,

\[
D(t;q)-D(t;0)
=\frac{q(1-s_t)}{a_t(1+q)}\ge 0.
\]

The floor therefore raises `D` at every finite age. It may move early points
above one and can hide the upturn over a finite scan. For the registered betas,

\[
D(40;q)>D(20;q)\quad\Longleftrightarrow\quad
q<0.795771761775.
\]

The other registered adjacent inequalities, `D(2)>D(5)>D(20)`, hold for every
`q>=0`. For very large fixed `q` and these finite ages,
`D(t;q)` is close to `1/(1-beta1^t)`, an above-one decreasing curve. For every
finite `q`, however, the `beta2^t` asymptotic term eventually makes the curve
approach one from below because `beta2>beta1`; the late upturn can simply occur
after the scanned horizon.

If the scalar coordinate is normalized to `|g|=1` and the Reddi experimental
choice `tau=1e-3` is used, then `q=1e-3` and

| `T` | `D_adam(T; q=0)` | `D_adam(T; q=0.001)` | floor shift in log2 bits |
|---:|---:|---:|---:|
| 2 | 0.742459788403 | 0.746975970328 | 0.008748942 |
| 5 | 0.540601963459 | 0.542501404863 | 0.005060122 |
| 20 | 0.485783579482 | 0.486435547194 | 0.001934936 |
| 40 | 0.583982312067 | 0.584412901800 | 0.001063355 |

The shape is unchanged in this normalized small-floor example, but a single
numeric floor curve is not scale-free in a real vector. This is why the v18
primary registration uses `tau=0` and defines a zero-over-zero coordinate to
produce zero. The `q=0.001` table is a frozen sensitivity prediction, not an
alternative gate target.

## Bias correction: three distinct conventions

### Reddi FedAdam: no bias correction

Algorithm 2 of Reddi et al., *Adaptive Federated Optimization* (ICLR 2021,
arXiv:2003.00295), uses the raw moments in the server update. It does not divide
by `1-beta1^t` or `1-beta2^t`. This is the convention that produces the boxed
U-shaped prediction.

The paper states `v_{-1} >= tau^2`; the present question explicitly instead
requires a zero-initialized second moment. If an implementation initializes
`v` to `tau^2`, then

\[
v_t=\beta_2^t\tau^2+(1-\beta_2^t)g^2,
\]

and it is a different mechanism. Such a run is ineligible for the v18 verdict.

### Corrected moments (PyTorch mathematical convention)

If both moments are corrected before epsilon is added,

\[
\widehat m_t=\frac{m_t}{1-\beta_1^t}=g,
\qquad
\widehat v_t=\frac{v_t}{1-\beta_2^t}=g^2,
\]

then

\[
u_t^{\rm corrected}
=\frac{\widehat m_t}{\sqrt{\widehat v_t}+\tau}
=\frac{g}{|g|+\tau}.
\]

Both startup factors cancel exactly, with or without the floor. Hence

\[
\boxed{D_{\rm PyTorch-corrected}(T;q)=1}
\]

for every positive age. A fully bias-corrected implementation cannot test the
registered raw-FedAdam curve.

### Original-Adam scalar step-size correction

The Algorithm-1 presentation of Adam is also commonly written with a scalar
step-size prefactor

\[
\alpha_t=\alpha\frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t}
\]

multiplying `m_t/(sqrt(v_t)+tau)`. With `tau=0`, this is equivalent to corrected
moments and gives `D=1`. If the same unscaled `tau` remains inside the raw
denominator, however,

\[
u_t^{\rm scalar}=\operatorname{sign}(g)\frac{s_t}{s_t+q},
\]

and

\[
\boxed{
D_{\rm scalar-corrected}(T;q)
=\frac{s_T+q}{s_T(1+q)}
=1+\frac{q(1-s_T)}{s_T(1+q)}.}
\]

It is above one and decreases monotonically to one for `q>0`. At `q=0.001`,
its values at `T={2,5,20,40}` are
`{1.006082729, 1.003513567, 1.001342095, 1.000737334}`. Thus the two fully
equivalent no-floor bias-correction forms differ slightly when they place a
fixed epsilon in different algebraic locations. Implementations must record
which denominator was used.

## Terminal versus path matching

The terminal registered multiplier is not the total path coefficient. The
normalized accumulated gain over `T` calls is

\[
\sum_{t=1}^T R_{\rm raw}(t;q),
\]

so a steady-total-displacement matching sensitivity would instead use

\[
D_{\rm path}(T;q)
=\frac{T}{\sum_{t=1}^T R_{\rm raw}(t;q)}.
\]

For `q=0.001`, this gives
`{0.858435, 0.669918, 0.517465, 0.525995}` at the four registered horizons.
It retains the down-then-up qualitative shape but is not the v18 target. The
Lean file defines the accumulated sum so a terminal and an accumulated object
cannot be silently interchanged.

## Assumptions and exclusions

The closed form is exact under these assumptions:

1. age is the number of calls seen by one persistent server-moment state, with
   the first call at age one;
2. first and second moments are both exactly zero before that first call;
3. the same scalar `g` is supplied on every call;
4. powers, square roots, and division are real arithmetic; implementation
   rounding is outside the theorem;
5. the terminal multiplier is used as a tuned-rate matching heuristic; and
6. for the scale-free registered curve, `tau=0` with an explicit zero-safe
   coordinate rule.

For vectors, the moment unrolling is coordinatewise. With `tau=0`, every
nonzero constant coordinate has the same normalized age gain, so the finite-age
factor is common across the vector even though FedAdam changes its direction
relative to SGD. With `tau>0`, `q_j=tau/|g_j|` varies by coordinate and there is
generally no single exact scalar curve. Changing or rotating pseudo-gradients,
sparse coordinates, client sampling, curvature, stochasticity, optimizer-state
resets, and nonlinear learning-rate optima are deliberate possible reasons for
the registered prediction to be wrong; they are not algebraic exceptions added
after observing the experiment.

The machine-checked statements are in
`lean-mechanism/LeanMechanism/AdaptiveFiniteAge.lean`.
