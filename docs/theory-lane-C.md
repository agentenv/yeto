# Theory Lane C: empirical outer-momentum response surface

## Verdict and scope

**CLOSED as an empirical tuning law.** This lane gives a frozen, CPU-reproducible response surface for the tuned outer-learning-rate ratio $D$. It is not a mechanism, it does not revive the rejected scalar AR(1) closure, and it does not infer an optimizer-performance advantage from a tuning ratio.

The fit uses every finite measured $D$ point supplied for the 135M study: 19 raw observations and 6 corrected observations across the pilot, v1, v2, v3, and fixed-$T$ disambiguation campaigns. The two v3 $T=40$ rows whose required optima were unbracketed have no finite $D$, so there is no response to fit. The pilot $T=2$ and $T=20$ extrapolations and the v2 `NOT_EVALUABLE` point are retained with their original labels. No GPU work was run.

The result in one line is:

> Factor out the exact $1/(1-\mu^{T+1})$ raw-Nesterov transient, then use a four-coefficient robust surface for the remaining raw residual and a three-coefficient linear surface for corrected drift; typical held-out error is small, but the disclosed pilot $T=20$ extrapolation is a real 2.14-fold failure and makes the honest raw 95% interval wide.

## Frozen formulas

For a legal experimental cell, $S=TH$. Define

\[
u={\mu\over0.9},\qquad
q={T-5\over10},\qquad
h=\log_2{H\over512}=\log_2{S\over512T},
\]

and define the code-true final-step raw transient

\[
K(T,\mu)={1\over1-\mu^{T+1}}.
\]

The raw response was fitted after transforming each observation to

\[
z_{\rm raw}=\log_2{D_{\rm raw}\over K}
=\log_2\!\left[D_{\rm raw}(1-\mu^{T+1})\right].
\]

The frozen raw formula is

\[
\boxed{
\log_2 \widehat D_{\rm raw}
=-\log_2(1-\mu^{T+1})
+u\left(
0.2105314103
-0.2561806810q
-0.0407132320h
-0.0464485728qh
\right).}
\tag{R}
\]

The frozen corrected formula is

\[
\boxed{
\log_2 \widehat D_{\rm corr}
=u\left(
-0.2033958365
-0.4691937641q
-0.1068146276h
\right).}
\tag{C}
\]

Given a no-momentum optimum $\eta_0^*(T,H,S)$, either response predicts the momentum-arm optimum without a momentum sweep:

\[
\boxed{
\widehat\eta^*_{\rm arm}(T,H,S,\mu)
=\eta_0^*(T,H,S)(1-\mu)\widehat D_{\rm arm}(T,H,S,\mu).}
\tag{LR}
\]

This is a zero-shot **momentum transfer** formula, conditional on one no-momentum anchor at the target training configuration. The present data do not support a checkable absolute-$\eta_0^*$ surface, so a claim that (LR) predicts an absolute LR without an $\eta_0^*$ anchor would be stronger than what was fitted.

At $\mu=0$, $u=0$, $K=1$, and both formulas return $D=1$ exactly. Equation (LR) therefore returns $\eta^*=\eta_0^*$, reproducing the correction-on versus correction-off bit-identical A/A control by construction, rather than approximately.

## Why there are only two design coordinates

The requested notation $a(T)+b(S)+c(H)$ contains a structural alias: every observed and every legal future cell obeys $S=TH$, hence

\[
\log_2 S=\log_2 T+\log_2 H.
\]

For example, adding $\lambda\log_2T$ to $a$, subtracting $\lambda\log_2S$ from $b$, and adding $\lambda\log_2H$ to $c$ changes no fitted value. Separate linear age, duration, and window effects therefore cannot be identified from this design. Any three named coefficients would be a gauge choice, not three empirical effects.

Equations (R) and (C) use the canonical gauge $b(S)=0$, with $T$ and $h=\log_2(S/(512T))$ as the two identifiable coordinates. The raw fit adds the one interaction that improved literal pointwise LOO prediction. The corrected observations lie only on the two axes $h=0$ and $q=0$, so a corrected interaction is exactly unidentifiable and is frozen to zero. The fixed-$T$ disambiguation row establishes a scale-axis association, but cannot distinguish a causal $H$ effect from a causal $S$ effect.

## Data boundary and provenance

| campaign | raw finite (D) | corrected finite (D) | treatment in this fit |
|---|---:|---:|---|
| pilot | 6 | 0 | all retained; (T=2,20) retain `EXTRAPOLATED` |
| v1 | 4 | 0 | all retained |
| v2 | 3 | 0 | all retained; (H=64,T=40) retains `NOT_EVALUABLE` |
| v3 | 4 | 4 | all finite points retained; empty (T=40) ratios are not observations |
| disambiguation | 2 | 2 | the two new (H=1024,2048, T=5) points per arm |
| **total** | **19** | **6** | **25 finite responses** |

The immutable inputs and generated audit are bound by these SHA-256 values:

| artifact | SHA-256 |
|---|---|
| `/root/two-param-analysis/data/master_D.csv` | `f1b132a5b4580a396da344a959f195c747d3b759d6db54243d303316eed77427` |
| `/root/two-param-analysis/data/master_eta.csv` | `610f8e302d3a3ba7e888e268ad832d1be7e5015c0b93ab000208c57d0cf5d649` |
| `/root/two-param-analysis/REPORT.md` | `765c9ce0104c9bb8f82dcb3a1fb8cf8393f80152bfc2eea8f7b9c0c3b11f9f77` |
| `/private/tmp/h200-disambig-note.md` | `4be85f66125472fca267f284912442d5bf4ce6e25dbaae9dbca8f381e4af4835` |
| `/root/g3-readout.json` | `d4a3cde6aa47580dff255c7a66030ab997a95f4072b1883bf71aa54d7da744c8` |
| `/root/g1-readout.json` | `c2dcd6b9ab7dce0dc28d1e2473a72c7e0bdb8d6221728f09503a6354a39cae2b` |
| `/root/g1v2-readout.json` | `5d4eed9685f25fd1db3135319908a045a389300a008a67bc009cd178cabe2fc8` |
| `scripts/fit_theory_lane_c.py` | `26b4ce92a83a6db06899b5067262d6ba7220bc7d6e51675712b8797a31b8f896` |
| `/root/theory-C/fit.json` | `45a80ba01819016f296719db1eaaa87aaccc001ec9410b841d66e34867dd4378` |

The code-true multiplier is independently formalized by `codeTrue_terminalMultiplier_closed_form` in `lean-mechanism/LeanMechanism/FiniteHorizonOuter.lean`. The implementation divides by $1-\mu^{t+1}$ before the applied outer step in `syncer/src/state.rs`; its regression `bias_correction_with_zero_momentum_is_bit_identical_to_off` checks the $\mu=0$ A/A fact.

## Estimator and model selection

All selection scores below are literal leave-one-observation-out predictions. Each held-out prediction refits the estimator on the other (n-1) observations. There is no in-sample substitution for LOO.

The raw response uses a fixed Huber threshold of 0.15 bits, the pre-existing residual tolerance used by the two-parameter closure audit. This threshold was not optimized against the candidate table. The full fit retains all 19 observations and downweights only the already-labelled pilot (H=512,T=20,mu=.9) extrapolation, to weight 0.1491. Equal-weight OLS on the selected raw basis has LOO RMSE 0.3231 bits and MAE 0.1981 bits, versus 0.2642 and 0.1209 for the frozen robust estimator. Thus the robust choice materially affects the raw coefficients and is part of the frozen specification.

The corrected response has no comparable outlier and uses ordinary least squares.

### Raw candidate audit

| candidate residual surface | parameters | LOO RMSE (bits) | LOO MAE (bits) | maximum absolute error (bits) |
|---|---:|---:|---:|---:|
| constant | 1 | 0.3267 | 0.1589 | 1.3305 |
| linear $T$ | 2 | 0.3255 | 0.1409 | 1.3421 |
| linear $\log_2T$ | 2 | 0.3117 | 0.1407 | 1.2922 |
| linear $T+h$ | 3 | 0.3310 | 0.1502 | 1.3532 |
| linear $\log_2T+h$ | 3 | 0.3138 | 0.1549 | 1.2785 |
| **linear $T+h$ plus $T\times h$** | **4** | **0.2642** | **0.1209** | **1.0947** |
| selected surface plus $u(u-1)$ momentum bend | 5 | 0.2647 | 0.1231 | 1.0936 |
| linear $T+h$ plus $(\log_2S)^2$ | 4 | 0.2650 | 0.1156 | 1.1059 |
| linear $T+h$ plus separate $T^2+h^2$ | 5 | 0.2751 | 0.1401 | 1.0958 |

The interaction and duration-curvature candidates are effectively tied in aggregate error; their RMSEs differ by only 0.0009 bits. The interaction is frozen because RMSE was the primary metric and it is linear on the directly measured fixed-$T$ scale row. This tie is a model-form uncertainty that the numerical prediction interval does not erase.

The fifth momentum-curvature coefficient slightly worsens both RMSE and MAE, so the minimal $u=\mu/.9$ gate is retained. It enforces the exact $\mu=0$ control. Only the raw arm has nonzero-$\mu$ observations at more than one momentum value, so the same linear gate in the corrected formula is a structural interpolation, not a separately validated corrected-arm $\mu$ law.

### Corrected candidate audit

| candidate surface | parameters | LOO RMSE (bits) | LOO MAE (bits) | maximum absolute error (bits) |
|---|---:|---:|---:|---:|
| constant | 1 | 0.3173 | 0.2336 | 0.6294 |
| linear $T$ | 2 | 0.1027 | 0.0896 | 0.1899 |
| linear $\log_2T$ | 2 | 0.1752 | 0.1525 | 0.3089 |
| linear $h$ | 2 | 0.3572 | 0.2839 | 0.6837 |
| **linear $T+h$** | **3** | **0.0264** | **0.0211** | **0.0537** |
| linear $\log_2T+h$ | 3 | 0.2314 | 0.1961 | 0.3697 |
| linear $T+h$ plus $(\log_2S)^2$ | 4 | 0.0343 | 0.0274 | 0.0634 |
| linear $T+h$ plus separate $T^2+h^2$ | 5 | 0.1138 | 0.0801 | 0.2547 |

At $H=512,\mu=.9$, equation (C) is

\[
\log_2\widehat D_{\rm corr}=0.0312010-0.0469194T.
\]

That reproduces the measured log-linear drift and is numerically consistent with the independently reported free-intercept slope $-0.04614$ bits per update. At fixed $T=5$, each doubling of the scale coordinate multiplies predicted $D_{\rm corr}$ by $2^{-0.1068146}=0.92864$.

## Required-fact checks against the full fit

These are fitted predictions, not in-sample interpolation constraints.

### Fixed $H=512,\mu=.9$ scan

| arm | $T$ | code-only $K(T,.9)$ | measured $D$ | full-fit $\widehat D$ | fit error |
|---|---:|---:|---:|---:|---:|
| raw | 2 | 3.6900 | 4.1675 | 4.5034 | +8.1% |
| raw | 5 | 2.1342 | 2.5668 | 2.4695 | -3.8% |
| raw | 10 | 1.4573 | 1.5883 | 1.5430 | -2.9% |
| raw | 20 | 1.1229 | 0.9898 | 0.9955 | +0.6% |
| corrected | 2 | — | 0.9553 | 0.9575 | +0.2% |
| corrected | 5 | — | 0.8615 | 0.8685 | +0.8% |
| corrected | 10 | — | 0.7480 | 0.7382 | -1.3% |
| corrected | 20 | — | 0.5306 | 0.5332 | +0.5% |

The raw observations themselves are $+12.9\%,+20.3\%,+9.0\%,-11.8\%$ from the factored code law at $T=2,5,10,20$, respectively. Equation (R) preserves the code law as the dominant factor while fitting the systematic residual rather than declaring the residual zero.

### Fixed-$S$ and fixed-$T$ checks

| arm and cell | measured $D$ | full-fit $\widehat D$ | fit error |
|---|---:|---:|---:|
| raw v2 (T=40,H=64,S=2560) | 0.9965 | 0.9613 | -3.5% |
| raw v2 (T=160,H=16,S=2560) | 1.0364 | 1.0303 | -0.6% |
| raw disambiguation (T=5,H=1024,S=5120) | 2.5532 | 2.4008 | -6.0% |
| raw disambiguation (T=5,H=2048,S=10240) | 2.3748 | 2.3340 | -1.7% |
| corrected disambiguation (T=5,H=1024,S=5120) | 0.8099 | 0.8065 | -0.4% |
| corrected disambiguation (T=5,H=2048,S=10240) | 0.7474 | 0.7490 | +0.2% |

The corrected fixed-$T$ predictions are $0.8685,0.8065,0.7490$ for $H=512,1024,2048$, reproducing the observed $0.8615,0.8099,0.7474$ row. This is compatible with the reported age-only log-RMSE 0.108 beating the duration-only 0.249, while still retaining a smaller scale-axis drift.

### The rejected scalar closure stays rejected

This surface deliberately has no $\rho$ parameter. The incompatible measurements remain:

| source | reported scalar-equivalent value |
|---|---:|
| corrected-slope multiplicative mapping | $\rho\approx0.9935$ |
| conditional raw residual fit | $\rho\approx0.9944$, but $\chi^2=1036.5/10$ and rejected |
| measured buffer-norm pooled inversion | $\rho\approx0.5191$ |
| direct lag telemetry | $\rho\approx0.6973$ |
| measured buffer gain (T=160,H=16) | 4.2798 versus ideal 10 |
| measured buffer gain (T=5,H=512) | 3.5145 versus ideal 10 |

Using $\rho\approx.994$ as if it were a mechanism would contradict both direct norm gains and telemetry. Lane C predicts the response without making that identification.

## Every held-out prediction

The signed bit error is

\[
e_i=\log_2D^{\rm LOO}_i-\log_2D_i;
\]

positive means overprediction. The original status is carried through and never upgraded.

### Raw arm: 19/19

| held-out point | original status | observed $D$ | LOO prediction | error (bits) | relative error |
|---|---|---:|---:|---:|---:|
| pilot (H512,T2,mu=.9) | `EXTRAPOLATED` | 4.112718 | 4.594980 | +0.1600 | +11.7% |
| pilot (H512,T5,mu=.5) | `VALID_WITH_INDEPENDENT_V2_REFERENCE` | 1.018995 | 1.104363 | +0.1161 | +8.4% |
| pilot (H512,T5,mu=.8) | `VALID_WITH_INDEPENDENT_V2_REFERENCE` | 1.523857 | 1.544616 | +0.0195 | +1.4% |
| pilot (H512,T5,mu=.95) | `VALID_WITH_INDEPENDENT_V2_REFERENCE` | 4.610253 | 4.378681 | -0.0743 | -5.0% |
| pilot (H512,T10,mu=.9) | `VALID` | 1.684740 | 1.528872 | -0.1401 | -9.3% |
| pilot (H512,T20,mu=.9) | `EXTRAPOLATED` | 0.495702 | 1.058679 | **+1.0947** | **+113.6%** |
| v1 (H16,T160,mu=.5) | `CONSERVATIVE_CURVE_CI_ENVELOPE` | 1.003566 | 1.018759 | +0.0217 | +1.5% |
| v1 (H16,T160,mu=.9) | `VALID` | 1.026579 | 1.033134 | +0.0092 | +0.6% |
| v1 (H256,T10,mu=.5) | `CONSERVATIVE_CURVE_CI_ENVELOPE` | 1.083989 | 1.056992 | -0.0364 | -2.5% |
| v1 (H512,T5,mu=.9) | `VALID` | 2.511596 | 2.464957 | -0.0270 | -1.9% |
| v2 (H16,T160,mu=.9) | `VALID` | 1.036395 | 1.025686 | -0.0150 | -1.0% |
| v2 (H64,T40,mu=.9) | `NOT_EVALUABLE` | 0.996453 | 0.902284 | -0.1432 | -9.5% |
| v2 (H512,T5,mu=.9) | `VALID` | 2.441338 | 2.472618 | +0.0184 | +1.3% |
| v3 (H512,T2,mu=.9) | `VALID` | 4.167463 | 4.586718 | +0.1383 | +10.1% |
| v3 (H512,T5,mu=.9) | `VALID` | 2.566822 | 2.459100 | -0.0619 | -4.2% |
| v3 (H512,T10,mu=.9) | `VALID` | 1.588337 | 1.538349 | -0.0461 | -3.1% |
| v3 (H512,T20,mu=.9) | `VALID` | 0.989847 | 1.003678 | +0.0200 | +1.4% |
| disambiguation (H1024,T5,mu=.9) | `INTERIOR` | 2.553200 | 2.372493 | -0.1059 | -7.1% |
| disambiguation (H2048,T5,mu=.9) | `INTERIOR` | 2.374780 | 2.295884 | -0.0487 | -3.3% |

Raw LOO summary: RMSE 0.2642 bits, MAE 0.1209 bits, median absolute error 0.0487 bits, maximum 1.0947 bits. The RMSE is dominated by the disclosed pilot (T=20) point. Removing it would be scientifically cleaner for prediction but would violate the instruction to fit every measured point, so it remains in both fitting and error accounting.

### Corrected arm: 6/6

| held-out point | original status | observed $D$ | LOO prediction | error (bits) | relative error |
|---|---|---:|---:|---:|---:|
| v3 (H512,T2,mu=.9) | `VALID` | 0.955266 | 0.959757 | +0.0068 | +0.5% |
| v3 (H512,T5,mu=.9) | `VALID` | 0.861517 | 0.871876 | +0.0172 | +1.2% |
| v3 (H512,T10,mu=.9) | `VALID` | 0.748001 | 0.735032 | -0.0252 | -1.7% |
| v3 (H512,T20,mu=.9) | `POINT_WITH_INVALID_BOOTSTRAPS` | 0.530626 | 0.550752 | +0.0537 | +3.8% |
| disambiguation (H1024,T5,mu=.9) | `INTERIOR` | 0.809865 | 0.805402 | -0.0080 | -0.6% |
| disambiguation (H2048,T5,mu=.9) | `INTERIOR` | 0.747421 | 0.755727 | +0.0159 | +1.1% |

Corrected LOO summary: RMSE 0.0264 bits, MAE 0.0211 bits, median absolute error 0.0166 bits, maximum 0.0537 bits. This is a strong interpolation result on only six points, not evidence for arbitrary extrapolation.

## Frozen prediction intervals

For each arm, sort the absolute LOO errors and take rank

\[
k_p=\min\{n,\lceil(n+1)p\rceil\}.
\]

Let $q_p$ be that ranked absolute error in bits and $F_p=2^{q_p}$. The frozen response interval is

\[
\boxed{
\operatorname{PI}_p(D)=
[\widehat D/F_p,\ \widehat D F_p],}
\qquad
\operatorname{PI}_p(\eta^*)=
\eta_0^*(1-\mu)\operatorname{PI}_p(D).
\tag{PI}
\]

| arm | $n$ | nominal $p$ | rank | radius (bits) | multiplicative factor $F_p$ |
|---|---:|---:|---:|---:|---:|
| raw | 19 | 50% | 10 | 0.04874 | 1.0344 |
| raw | 19 | 80% | 16 | 0.14006 | 1.1019 |
| raw | 19 | 90% | 18 | 0.15997 | 1.1173 |
| raw | 19 | 95% | 19 | 1.09472 | **2.1357** |
| corrected | 6 | 50% | 4 | 0.01724 | 1.0120 |
| corrected | 6 | 80% | 6 | 0.05371 | 1.0379 |
| corrected | 6 | 90% | 6 | 0.05371 | 1.0379 |
| corrected | 6 | 95% | 6 | 0.05371 | 1.0379 |

These are empirical LOO-residual intervals, not Gaussian coefficient intervals and not a theorem of external coverage. The campaigns are heterogeneous rather than exchangeable, and the raw 95% interval is correctly forced to include the pilot failure. At $n=6$, corrected 80–95% all use the maximum residual; the apparent narrowness comes from the six points lying close to one plane.

The calibrated design supports are the convex hulls in $(q,h)$:

- raw: $[(-0.3,0),(0.5,-1),(3.5,-3),(15.5,-5),(0,2)]$, with measured $\mu\in\{.5,.8,.9,.95\}$ distributed sparsely inside it;
- corrected: the triangle $[(-0.3,0),(1.5,0),(0,2)]$, at $\mu=.9$, plus the exact $\mu=0$ A/A identity.

Equation (PI) is the reported interval only for interpolation in those supports. For $(q,h)$ outside the relevant hull, report the point formula as `EXTRAPOLATION` and the interval as **not calibrated**. In particular, applying (C) to $H<512,T\gg20$ produces extreme numbers unsupported by any corrected observation. For $0<\mu<.9$, corrected predictions are conditional on the linear $u$ gate; there is no corrected-arm intermediate-$\mu$ validation. Uncertainty in the supplied $\eta_0^*$ must be propagated separately.

## Reference calculator

The committed audit script regenerates all coefficients, candidate scores, held-out rows, interval factors, and checksums. The frozen forward calculation itself is:

```python
from math import log2

RAW_BETA = (0.2105314103, -0.2561806810, -0.0407132320, -0.0464485728)
CORR_BETA = (-0.2033958365, -0.4691937641, -0.1068146276)
PI_FACTOR = {
    "raw": {0.80: 1.1019497307, 0.90: 1.1172612031, 0.95: 2.1357189183},
    "corrected": {0.80: 1.0379286834, 0.90: 1.0379286834, 0.95: 1.0379286834},
}

def lane_c_lr(T, H, S, mu, eta0, arm="raw", coverage=0.80):
    if T <= 0 or H <= 0 or S != T * H:
        raise ValueError("Lane C requires positive values and S == T*H")
    if not (0.0 <= mu < 1.0) or eta0 <= 0:
        raise ValueError("require 0 <= mu < 1 and eta0 > 0")
    u = mu / 0.9
    q = (T - 5.0) / 10.0
    h = log2(H / 512.0)
    if arm == "raw":
        b0, bq, bh, bqh = RAW_BETA
        log2_D = -log2(1.0 - mu ** (T + 1)) + u * (
            b0 + bq * q + bh * h + bqh * q * h
        )
    elif arm == "corrected":
        b0, bq, bh = CORR_BETA
        log2_D = u * (b0 + bq * q + bh * h)
    else:
        raise ValueError("arm must be 'raw' or 'corrected'")
    D = 2.0 ** log2_D
    eta = eta0 * (1.0 - mu) * D
    if mu == 0.0:  # code-exact A/A, not an empirical interval
        return {"D": 1.0, "eta": eta0, "D_interval": (1.0, 1.0),
                "eta_interval": (eta0, eta0)}
    factor = PI_FACTOR[arm][coverage]
    return {
        "D": D,
        "eta": eta,
        "D_interval": (D / factor, D * factor),
        "eta_interval": (eta / factor, eta * factor),
    }
```

Hull membership and momentum-support labels must be checked before attaching the calibrated interpretation to the returned interval.

## Where the steady momentum rule understeps or oversteps

Suppose the deployed heuristic is the usual steady scaling

\[
\eta_{\rm steady}=(1-\mu)\eta_0^*.
\]

Then

\[
{\eta_{\rm steady}\over\widehat\eta^*}={1\over\widehat D}.
\]

This gives a checkable tuning phase map:

- $\widehat D>1$: the steady rule is below the predicted optimum, i.e. it **understeps**;
- $\widehat D<1$: the steady rule is above the predicted optimum, i.e. it **oversteps**;
- $\widehat D\approx1$: the steady rule is calibrated.

At $H=512,\mu=.9$:

| $T$ | raw $\widehat D$ | raw steady/optimal | raw diagnosis | corrected $\widehat D$ | corrected steady/optimal | corrected diagnosis |
|---:|---:|---:|---|---:|---:|---|
| 2 | 4.503 | 0.222 | severe understep | 0.958 | 1.044 | mild overstep |
| 5 | 2.470 | 0.405 | understep | 0.869 | 1.151 | overstep |
| 10 | 1.543 | 0.648 | understep | 0.738 | 1.355 | overstep |
| 20 | 0.995 | 1.005 | calibrated | 0.533 | 1.875 | severe overstep |

The raw point surface crosses $\widehat D=1$ at $T\approx19.85$ on the measured $H=512,\mu=.9$ line. This is a point-estimate boundary; the honest raw 95% interval is much too broad to promote 19.85 into a sharp universal phase transition.

At fixed $T=5,\mu=.9$, the predicted raw $D$ values for $H=512,1024,2048$ are $2.470,2.401,2.334$: the steady rule understeps throughout. The corresponding corrected values are $0.869,0.807,0.749$: the steady rule increasingly oversteps.

This phase map says where a **learning-rate rule** helps or hurts calibration. It does not say where momentum improves the minimum achievable loss. $D$ locates a curve's optimum; it contains no sign for the loss difference between the momentum and no-momentum minima. The only direct matched performance result in the supplied evidence is the SNOO cell at $M=1,H=512,T=5$, where Nesterov is worse than plain by 0.1132 NLL $[0.0753,0.1463]$ and worse than its effective-LR-matched control by 0.0686 $[0.0107,0.1194]$. Momentum hurts in that measured cell. A multi-cell performance-benefit boundary is not identified by these data, and Lane C does not invent one.

## What the surface fails to fit or justify

1. **The pilot $T=20$ point is not predicted.** Its held-out prediction is 1.0587 versus 0.4957, an error of 1.0947 bits or +113.6%. It was an extrapolated optimum on a wrapped pilot dataset; those provenance facts explain why robust fitting is reasonable, but do not erase the observation. This single miss expands the raw 95% factor to 2.1357.

2. **The raw form is not unique.** The selected interaction and an additive duration-curvature model have almost identical LOO scores. Equation (R) is a frozen predictive convention, not discovery of a unique functional law.

3. **$S$ versus $H$ is not causally separated.** The algebraic constraint $S=TH$ makes a three-way decomposition non-identifiable. The reported scale coefficient predicts legal cells but must not be narrated as an independently measured $H$-only effect.

4. **Corrected momentum transfer away from $\mu=.9$ is unvalidated.** The exact $\mu=0$ endpoint is known from code, but no corrected observations lie strictly between 0 and .9. The linear $u$ interpolation is the minimal frozen choice, not a measured $\mu$-response curve.

5. **The corrected hull is small.** The excellent corrected LOO score covers the $H=512,T=2\ldots20$ line and the $T=5,H=512\ldots2048$ line. It does not validate low-$H$, long-$T$ combinations.

6. **Intervals are conditional and small-sample.** They summarize held-out response error within heterogeneous historical campaigns. They do not include uncertainty in $\eta_0^*$, future model scale, model architecture, learner count, data order, or a changed optimizer implementation.

7. **No mechanism closes.** The buffer-norm, telemetry, raw-tuning, and corrected-drift mappings disagree. Equations (R) and (C) are empirical response surfaces and should not be back-translated into a universal scalar correlation.

8. **No general loss-benefit map closes.** The fit predicts the LR optimum ratio and the direction of steady-rule miscalibration. Apart from the measured SNOO reversal, it cannot say whether optimally tuned momentum beats optimally tuned $\mu=0$.

## Reproduction

On `h200-n1`, using the existing CPU environment:

```bash
CUDA_VISIBLE_DEVICES="" /root/yeto-venv/bin/python \
  scripts/fit_theory_lane_c.py \
  --master-d /root/two-param-analysis/data/master_D.csv \
  --disambig-note /root/theory-C/h200-disambig-note.md \
  --output /root/theory-C/fit.json
```

The executed working copy and note are under `/root/theory-C/`; the immutable source analysis tree was read only. The script fails closed on either input checksum, on any illegal (S\ne TH) row, or if the robust fit does not converge.

THEORY C CLOSED The code-transient-factored surfaces predict corrected drift to 0.026 bits LOO and raw tuning to 0.264 bits LOO, with an honest 2.14-fold raw 95% band forced by the failed pilot T=20 point.
