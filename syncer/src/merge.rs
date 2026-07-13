//! Merge math for asynchronous multi-learner training.
//!
//! Per-learner outer gradient for fragment p: Δ_m = Θ_p(prev) − θ_m,p, i.e.
//! anchored at the syncer's own previous global fragment.
//! Learner weights w_m = c_tokens · (c_tokens / c_steps) — quantity ×
//! quality. Merging is either weighted direct averaging (embedding
//! fragment) or weighted radial-directional averaging, RDA (everything else):
//!
//!   RDA({v_m, w_m}) = (Σ w_m ‖v_m‖ / Σ w_m) · φ(Σ w_m φ(v_m) / Σ w_m)
//!
//! with φ(x) = x/‖x‖ and φ(0) := 0. RDA keeps the merged norm invariant to the
//! number of learners: near-orthogonal same-norm deltas would otherwise
//! shrink as R/√M and force outer-lr retuning. Applied per tensor within a
//! fragment. Degenerate mean direction falls back to direct averaging.
//!
//! A third opt-in mode, Iso-C-style isotropic aggregation ("iso", IsoLoCo,
//! arXiv 2607.03011), direct-averages the per-tensor deltas and then
//! flattens the singular-value spectrum of the averaged matrix to its mean:
//! Δ ← σ̄·U·Vᵀ with SVD(Δ) = U·diag(σ)·Vᵀ and σ̄ = mean(σ). See `merge_iso`.
//!
//! Outer optimizer state is held on the syncer. Nesterov remains the default;
//! normalized EMA variants are available for gain-controlled experiments.
//!
//! Outer optimizer state is held on the syncer. Nesterov remains the default;
//! normalized EMA variants are available for gain-controlled experiments.

use std::fmt;
use std::str::FromStr;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum OuterOptimizer {
    #[default]
    Nesterov,
    NormalizedEma,
    RestartedEma,
    RhoAdaptive,
    CappedNesterov,
    CappedNesterovGc,
    CappedNesterovR,
    CappedNesterovCurv,
    CappedNesterovWsub,
    BlockRms,
    BlockYogi,
    ChebSgd,
}

impl OuterOptimizer {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Nesterov => "nesterov",
            Self::NormalizedEma => "normalized-ema",
            Self::RestartedEma => "restarted-ema",
            Self::RhoAdaptive => "rho-adaptive",
            Self::CappedNesterov => "capped-nesterov",
            Self::CappedNesterovGc => "capped-nesterov-gc",
            Self::CappedNesterovR => "capped-nesterov-r",
            Self::CappedNesterovCurv => "capped-nesterov-curv",
            Self::CappedNesterovWsub => "capped-nesterov-wsub",
            Self::BlockRms => "block-rms",
            Self::BlockYogi => "block-yogi",
            Self::ChebSgd => "cheb-sgd",
        }
    }

    pub const fn uses_normalized_ema(self) -> bool {
        matches!(self, Self::NormalizedEma | Self::RestartedEma)
    }
}

impl fmt::Display for OuterOptimizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for OuterOptimizer {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "nesterov" => Ok(Self::Nesterov),
            "normalized-ema" => Ok(Self::NormalizedEma),
            "restarted-ema" => Ok(Self::RestartedEma),
            "rho-adaptive" => Ok(Self::RhoAdaptive),
            "capped-nesterov" => Ok(Self::CappedNesterov),
            "capped-nesterov-gc" => Ok(Self::CappedNesterovGc),
            "capped-nesterov-r" => Ok(Self::CappedNesterovR),
            "capped-nesterov-curv" => Ok(Self::CappedNesterovCurv),
            "capped-nesterov-wsub" => Ok(Self::CappedNesterovWsub),
            "block-rms" => Ok(Self::BlockRms),
            "block-yogi" => Ok(Self::BlockYogi),
            "cheb-sgd" => Ok(Self::ChebSgd),
            other => Err(format!(
                "outer optimizer must be one of nesterov, normalized-ema, restarted-ema, rho-adaptive, capped-nesterov, capped-nesterov-gc, capped-nesterov-r, capped-nesterov-curv, capped-nesterov-wsub, block-rms, block-yogi, cheb-sgd; got {other:?}"
            )),
        }
    }
}

/// Diagnostics for one applied outer-optimizer step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OuterStepStats {
    /// L2 norm of the parameter displacement, `lr * optimizer_direction`.
    pub applied_step_norm: f64,
    /// Cosine between the optimizer direction and the current merged delta.
    /// Undefined when either vector has zero norm.
    pub direction_delta_cosine: Option<f64>,
    /// L2 norm of the optimizer's history contribution divided by the L2
    /// norm of its current-delta contribution. Undefined when the current
    /// contribution has zero norm.
    pub history_current_norm_ratio: Option<f64>,
    /// Whether restarted EMA discarded nonzero history on this commit.
    pub restarted: bool,
}

/// Apply one configured outer-optimizer step.
///
/// Keeping the dispatch next to the optimizer implementations gives the
/// state layer a single production path for both mutating commits and pure
/// previews made from cloned parameter and buffer slices. `rho_ema` is the
/// rho-adaptive controller's persistent scalar state (see
/// `rho_adaptive_step`) and `capped_mu` is the shared capped-Nesterov-family
/// persistent effective momentum (see `capped_nesterov_step`); `capped_gain`
/// is the gain-compensated variant's applied step-scale (see
/// `capped_nesterov_gc_step`). The other optimizers leave them untouched,
/// exactly as they leave each other's buffer conventions alone.
/// `tensor_numels` gives the per-tensor block boundaries of `delta` (their sum
/// is `delta.len()`) and `block_v` is the per-tensor scalar second-moment state
/// (one entry per tensor); both are consumed only by the block-second-moment
/// optimizers (`block-rms`/`block-yogi`) and otherwise ignored, exactly as the
/// scalar `rho_ema`/`capped_mu`/`capped_gain` states are left untouched by the
/// optimizers that do not use them. `disagreement_energy` is the worker-
/// disagreement transverse curvature-energy proxy `b_perp^T C b_perp`
/// (`disagreement_transverse_energy`, computed at merge time where the per-worker
/// deltas exist); it is consumed only by `capped-nesterov-wsub` and ignored (0.0)
/// by every other optimizer, exactly as the block/curv states are.
#[allow(clippy::too_many_arguments)]
pub fn apply_outer_step(
    optimizer: OuterOptimizer,
    params: &mut [f32],
    buf: &mut [f32],
    delta: &[f32],
    lr: f32,
    momentum: f32,
    restart_cos_threshold: f32,
    rho_ema: &mut f32,
    capped_mu: &mut f32,
    capped_gain: &mut f32,
    curv_prev_delta: &mut [f32],
    curv_prev_dtheta: &mut [f32],
    tensor_numels: &[usize],
    block_v: &mut [f32],
    disagreement_energy: f64,
    cheb_phase: &mut f32,
) -> OuterStepStats {
    match optimizer {
        OuterOptimizer::Nesterov => nesterov_step(params, buf, delta, lr, momentum),
        OuterOptimizer::NormalizedEma => normalized_ema_step(params, buf, delta, lr, momentum),
        OuterOptimizer::RestartedEma => {
            restarted_ema_step(params, buf, delta, lr, momentum, restart_cos_threshold)
        }
        OuterOptimizer::RhoAdaptive => rho_adaptive_step(params, buf, delta, lr, rho_ema),
        OuterOptimizer::CappedNesterov => {
            capped_nesterov_step(params, buf, delta, lr, capped_mu)
        }
        OuterOptimizer::CappedNesterovGc => {
            capped_nesterov_gc_step(params, buf, delta, lr, capped_mu, capped_gain)
        }
        OuterOptimizer::CappedNesterovR => {
            capped_nesterov_r_step(params, buf, delta, lr, capped_mu)
        }
        OuterOptimizer::CappedNesterovCurv => capped_nesterov_curv_step(
            params,
            buf,
            delta,
            lr,
            capped_mu,
            curv_prev_delta,
            curv_prev_dtheta,
        ),
        OuterOptimizer::CappedNesterovWsub => {
            capped_nesterov_wsub_step(params, buf, delta, lr, capped_mu, disagreement_energy)
        }
        OuterOptimizer::BlockRms => {
            block_second_moment_step(params, buf, delta, lr, tensor_numels, block_v, false)
        }
        OuterOptimizer::BlockYogi => {
            block_second_moment_step(params, buf, delta, lr, tensor_numels, block_v, true)
        }
        OuterOptimizer::ChebSgd => cheb_sgd_step(params, buf, delta, lr, cheb_phase),
    }
}

/// Materialize the nominal f32 parameter displacement produced by an outer
/// step after its optimizer buffer has been updated. This is the same vector
/// whose norm is reported by `OuterStepStats::applied_step_norm`.
///
/// For `CappedNesterov`, `CappedNesterovR`, `CappedNesterovCurv`, and
/// `CappedNesterovWsub` the
/// caller must pass the EFFECTIVE per-commit momentum written back by the step
/// (the updated persistent scalar), not the CLI momentum; with that value the
/// Nesterov branch is bit-identical to the applied step. For `CappedNesterovGc` the
/// caller must ALSO pass the applied gain written back by the step; the
/// gc branch then reproduces `lr * (gain * (delta + mu * buf))`
/// bit-for-bit. `gain` is ignored by every other optimizer.
pub fn materialize_applied_step(
    optimizer: OuterOptimizer,
    updated_buf: &[f32],
    delta: &[f32],
    lr: f32,
    momentum: f32,
    gain: f32,
) -> Vec<f32> {
    debug_assert_eq!(updated_buf.len(), delta.len());
    match optimizer {
        OuterOptimizer::Nesterov
        | OuterOptimizer::CappedNesterov
        | OuterOptimizer::CappedNesterovR
        | OuterOptimizer::CappedNesterovCurv
        | OuterOptimizer::CappedNesterovWsub => updated_buf
            .iter()
            .zip(delta)
            .map(|(buf, value)| lr * (*value + momentum * *buf))
            .collect(),
        OuterOptimizer::CappedNesterovGc => updated_buf
            .iter()
            .zip(delta)
            .map(|(buf, value)| lr * (gain * (*value + momentum * *buf)))
            .collect(),
        OuterOptimizer::NormalizedEma
        | OuterOptimizer::RestartedEma
        | OuterOptimizer::RhoAdaptive
        | OuterOptimizer::BlockRms
        | OuterOptimizer::BlockYogi
        | OuterOptimizer::ChebSgd => {
            updated_buf.iter().map(|buf| lr * *buf).collect()
        }
    }
}

/// Reference momentum whose tuned behavior the v2 controller reproduces at
/// the reference autocorrelation.
pub const RHO_ADAPTIVE_MU_STAR: f64 = 0.5;
/// Compile-time reference autocorrelation: at rho_hat = RHO_ADAPTIVE_RHO_REF
/// the gain is exactly 1 and the step matches the tuned mu_star baseline.
pub const RHO_ADAPTIVE_RHO_REF: f64 = 0.5;
/// EMA smoothing of the per-commit rho measurement (half-life one commit).
pub const RHO_ADAPTIVE_EMA_BETA: f64 = 0.5;
/// Hard bounds on the applied step-scale gain.
pub const RHO_ADAPTIVE_GAIN_MIN: f64 = 0.5;
pub const RHO_ADAPTIVE_GAIN_MAX: f64 = 2.0;
/// Fresh controller state starts at the reference autocorrelation so the
/// first commit (and the first commit after a checkpoint restore, which does
/// not persist this scalar) applies exactly the tuned baseline step.
pub const RHO_ADAPTIVE_INITIAL_RHO_EMA: f32 = RHO_ADAPTIVE_RHO_REF as f32;

/// v2 gain from the corrected aligned-amplification law (docs/EXP2_25.md
/// Correction): a Nesterov buffer with momentum mu on a rho-persistent
/// direction amplifies the aligned step by a(rho) = 1 + mu/(1 - mu*rho).
/// The controller targets the effective step a well-calibrated mu_star
/// setting would produce at the measured persistence, normalized so the
/// reference point is gain 1:
///
///   g(rho_hat) = a(rho_hat) / a(rho_ref),  clamped to [GAIN_MIN, GAIN_MAX].
pub fn rho_adaptive_gain(rho_hat: f64) -> f64 {
    let amplification = |rho: f64| 1.0 + RHO_ADAPTIVE_MU_STAR / (1.0 - RHO_ADAPTIVE_MU_STAR * rho);
    (amplification(rho_hat) / amplification(RHO_ADAPTIVE_RHO_REF))
        .clamp(RHO_ADAPTIVE_GAIN_MIN, RHO_ADAPTIVE_GAIN_MAX)
}

/// Rho-adaptive step, v2. `buf` stores the previously APPLIED direction,
/// `scale * delta` (not momentum); since the gain is always positive, the
/// stored direction has the same orientation as the previous merged delta
/// and the measured autocorrelation is unaffected. Each commit measures the
/// round-to-round direction autocorrelation rho = cos(delta, buf), folds it
/// into the persistent EMA `rho_ema` (beta = RHO_ADAPTIVE_EMA_BETA; commits
/// without a measurement — zero buffer or zero delta — leave the EMA
/// unchanged), and applies
///
///   theta <- theta - lr * g(rho_ema) * delta
///
/// with g from `rho_adaptive_gain`. Unlike v1's
/// mu_eff = clamp(2(1-rho), 0, mu_max) heuristic (calibrated on the wrong
/// rho convention; see docs/EXP2_26.md), v2 directly targets the effective
/// step of a tuned mu_star baseline under the corrected law: g == 1 at
/// rho_ema = rho_ref, mild amplification for persistent directions, mild
/// damping for anti-persistent ones. `--outer-momentum` is not consumed.
/// `OuterStepStats` reuse for diagnostics: `history_current_norm_ratio`
/// reports the per-commit RHO (not a norm ratio) for this optimizer.
pub fn rho_adaptive_step(
    params: &mut [f32],
    buf: &mut [f32],
    delta: &[f32],
    lr: f32,
    rho_ema: &mut f32,
) -> OuterStepStats {
    let mut dot = 0.0f64;
    let mut buf_norm_sq = 0.0f64;
    let mut delta_norm_sq = 0.0f64;
    for (b, d) in buf.iter().zip(delta) {
        dot += *b as f64 * *d as f64;
        buf_norm_sq += (*b as f64).powi(2);
        delta_norm_sq += (*d as f64).powi(2);
    }
    let measured = buf_norm_sq > 0.0 && delta_norm_sq > 0.0;
    let rho = if measured {
        (dot / (buf_norm_sq.sqrt() * delta_norm_sq.sqrt())).clamp(-1.0, 1.0)
    } else {
        0.0
    };
    if measured {
        *rho_ema = (RHO_ADAPTIVE_EMA_BETA * *rho_ema as f64
            + (1.0 - RHO_ADAPTIVE_EMA_BETA) * rho) as f32;
    }
    let scale = rho_adaptive_gain(*rho_ema as f64) as f32;
    let mut step_norm_sq = 0.0f64;
    for ((p, b), d) in params.iter_mut().zip(buf.iter_mut()).zip(delta) {
        // Write the buffer first and derive the step from it so this path
        // is bit-identical to materialize_applied_step's lr * buf.
        *b = scale * *d;
        let step = lr * *b;
        *p -= step;
        step_norm_sq += (step as f64).powi(2);
    }
    OuterStepStats {
        applied_step_norm: step_norm_sq.sqrt(),
        direction_delta_cosine: Some(1.0),
        history_current_norm_ratio: Some(rho),
        restarted: false,
    }
}

/// Capped-Nesterov (frozen controller spec, 2026-07-12): design-point
/// momentum, also the pointwise ceiling on the effective momentum.
pub const CAPPED_NESTEROV_MU_MAX: f64 = 0.9;
/// Transverse budget: mu_t^2 * r_t <= TAU_PERP caps the norm of the
/// history's delta-orthogonal contribution at TAU_PERP * |delta_t|.
pub const CAPPED_NESTEROV_TAU_PERP: f64 = 1.0;
/// Floor on the relative transverse residual r_t before inverting it; keeps
/// mu_perp finite (and inert, ~1e6) when the buffer is parallel to delta.
pub const CAPPED_NESTEROV_R_EPS: f64 = 1e-12;
/// One-sided release EMA weight on the previous effective momentum. Caps
/// bind instantly (min with the cap); release toward a loosened cap is
/// smoothed with this beta.
pub const CAPPED_NESTEROV_EMA_BETA: f64 = 0.9;
/// Fresh controller state starts at the design-point momentum. Like the
/// rho-adaptive EMA this scalar is NOT part of the checkpoint format; a
/// restore behaves like tuned Nesterov until the caps re-bind.
pub const CAPPED_NESTEROV_INITIAL_MU: f32 = CAPPED_NESTEROV_MU_MAX as f32;

/// Gain-compensated variant ("capped-nesterov-gc", EXP2.31 follow-up): the
/// EXP2.31 matched pairs showed the frozen controller understeps whenever a
/// cap binds — mu_t < mu_max makes the aligned gain A_t = 1 + mu_t +
/// mu_t^2 c_t fall below the (1 + mu_max) design gain, so a run tuned for
/// eta_eff = (1 + mu_max) * eta never reaches it. The gc variant keeps the
/// caps (they exist to bound the transverse-variance harm) and restores the
/// aligned effective LR by rescaling the applied step by
///
///   g_t = (1 + mu_max) / max(A_t, eps),  clamped to [GAIN_MIN, GAIN_MAX].
///
/// Since mu_t never exceeds the aligned cap, A_t <= 1 + mu_max and the raw
/// gain is >= 1; the clamp bounds the boost when A_t collapses toward 0
/// (strongly opposing history). Only the parameter displacement is scaled;
/// the momentum buffer keeps the plain capped-Nesterov recursion, so the
/// cap geometry (c_t, r_t) is identical to the frozen controller's.
pub const CAPPED_NESTEROV_GC_GAIN_MIN: f64 = 0.5;
pub const CAPPED_NESTEROV_GC_GAIN_MAX: f64 = 2.5;
/// Floor on A_t before inverting it; with the clamp above any A_t below
/// (1 + mu_max)/GAIN_MAX already saturates the gain, so this only guards
/// the division.
pub const CAPPED_NESTEROV_GC_A_EPS: f64 = 1e-12;
/// Fresh gc gain state (a fragment that has never committed under gc has
/// applied no rescale). Like `capped_mu` it is NOT checkpointed.
pub const CAPPED_NESTEROV_GC_INITIAL_GAIN: f32 = 1.0;

/// Relaxed transverse budget for the "capped-nesterov-r" variant. EXP2.31
/// showed the transverse harm is threshold-like: negligible at energy
/// amplification A2_RMS ~ 2.6, severe at ~ 10. With the A2 bound
/// ~ (1 + mu_max)^2 + tau^2, tau = sqrt(6 - 1.9^2) = 1.55 targets a bound
/// of ~6, comfortably below the harmful regime while releasing useful
/// transverse history the tau = 1 budget discards.
pub const CAPPED_NESTEROV_R_TAU_PERP: f64 = 1.55;

/// Transverse curvature-energy budget for the "capped-nesterov-curv" variant
/// (the T4-sanctioned controller). Unlike the geometry-blind budget
/// `CAPPED_NESTEROV_TAU_PERP` — which bounds only the transverse *norm*
/// (`mu^2 r_t |g| <= tau |g|`, geometry-blind, provably insufficient by Lean
/// T3) — this variant bounds the transverse *curvature energy proxy*
///
///   lambda_hat_t * ||d_perp||^2 = lambda_hat_t * (mu^2 * r_t * ||g_t||)^2 <= E_PERP
///
/// where `lambda_hat_t` is the online Hessian-free secant curvature proxy
/// (EXP2.39; see `capped_nesterov_curv_lambda_hat`). This is exactly the
/// curvature information Lean T4 requires to restore the SGD descent guarantee
/// that no scalar cap can (T3). Solving for the largest admissible mu gives
///
///   mu_curv = (E_PERP / (lambda_hat_t * r_t^2 * ||g_t||^2))^{1/4}
///
/// clamped to [0, mu_max]; lambda_hat_t -> 0 (or r_t/||g_t|| -> 0, or the first
/// commit) leaves the transverse cap inactive (mu_curv = mu_max).
///
/// Calibration (documented; capture is ON to validate post-hoc): E_PERP is set
/// so the curvature cap reproduces the frozen geometry-blind tau_perp = 1
/// behavior at the *median* lambda_hat at the H16 operating point. Equating the
/// two caps' transverse-norm bounds — geometry-blind binds `mu^2 r_t <= tau`,
/// curvature binds `mu^2 r_t <= sqrt(E_PERP / (lambda_hat ||g||^2))` — gives
///
///   E_PERP = tau_perp^2 * lambda_hat_med * ||g_t||_med^2.
///
/// With tau_perp = 1, a representative median per-fragment merged-delta norm
/// ||g_t||_med ~ 4.7 (||g_t||^2 ~ 22, from the EXP2.32-family capped-Nesterov
/// syncer tapes, per-fragment `gnorm`), and an order-unity secant curvature
/// proxy lambda_hat_med ~ 1.0 at the operating point, E_PERP ~ 22. This is a
/// starting value: it makes the cap bind comparably to the frozen tau_perp
/// controller at median curvature, tightening it where curvature is above
/// median (the T2 anisotropic-harm regime) and relaxing toward mu_max where it
/// is below.
pub const CAPPED_NESTEROV_CURV_E_PERP: f64 = 22.0;
/// Floor added to the previous applied step's squared norm in the secant
/// curvature proxy denominator (keeps lambda_hat finite and zeroes it on the
/// first commit / after a restore, where the stored previous step is zero).
pub const CAPPED_NESTEROV_CURV_LAMBDA_EPS: f64 = 1e-12;

/// Per-commit momentum cap from the realized buffer/delta geometry
/// (c_t = <b_{t-1}, delta_t> / |delta_t|^2, r_t = |b_{t-1} - c_t delta_t| /
/// |delta_t|), before the release EMA:
///
///   mu_par : largest mu in [0, mu_max] with mu + mu^2 * [c_t]_+ <= mu_max,
///            i.e. the positive root of [c_t]_+ mu^2 + mu - mu_max = 0,
///            computed in the rationalized (numerically stable) form
///            mu_par = 2 mu_max / (1 + sqrt(1 + 4 [c_t]_+ mu_max))
///            (THEORY.md F1; equals mu_max at [c_t]_+ = 0 with no branch,
///            and is continuous as c -> 0+ where the textbook root form
///            (sqrt(1+4c mu_max)-1)/(2c) cancels catastrophically) — caps
///            the aligned gain A_t = 1 + mu + mu^2 c_t at 1 + mu_max for
///            amplifying history;
///   mu_perp: sqrt(tau_perp / max(r_t, eps)) — caps the transverse
///            contribution mu^2 r_t |delta| at tau_perp |delta|;
///   cap    = min(mu_max, mu_par, mu_perp).
///
/// Sign-reversal guard: if A_t(cap) < 0 (possible only for strongly negative
/// c_t, where A_t is a downward parabola in mu with A_t(0) = 1), the cap is
/// zeroed — since A_t(0) = 1 > 0 and A_t(cap) >= 0 imply A_t >= 0 on all of
/// [0, cap], the guard keeps every admissible mu on the descent side.
pub fn capped_nesterov_cap(c_t: f64, r_t: f64) -> f64 {
    capped_nesterov_cap_with_tau(c_t, r_t, CAPPED_NESTEROV_TAU_PERP)
}

/// `capped_nesterov_cap` with an explicit transverse budget; the frozen
/// controller uses `CAPPED_NESTEROV_TAU_PERP`, the relaxed "-r" variant
/// `CAPPED_NESTEROV_R_TAU_PERP`. Everything else is shared.
pub fn capped_nesterov_cap_with_tau(c_t: f64, r_t: f64, tau_perp: f64) -> f64 {
    let mu_par = capped_nesterov_mu_par(c_t);
    let mu_perp = (tau_perp / r_t.max(CAPPED_NESTEROV_R_EPS)).sqrt();
    let cap = CAPPED_NESTEROV_MU_MAX.min(mu_par).min(mu_perp);
    if 1.0 + cap + cap * cap * c_t < 0.0 {
        0.0
    } else {
        cap
    }
}

/// Aligned momentum cap shared by the capped-Nesterov family: the largest mu
/// in [0, mu_max] with `mu + mu^2 * [c_t]_+ <= mu_max`, i.e. the positive root
/// of `[c_t]_+ mu^2 + mu - mu_max = 0`, in the rationalized (cancellation-free)
/// form `2 mu_max / (1 + sqrt(1 + 4 [c_t]_+ mu_max))`. Equals mu_max at
/// `[c_t]_+ = 0` with no separate branch and is continuous as c -> 0+. This
/// bounds the aligned gain `A_t = 1 + mu + mu^2 c_t` at `1 + mu_max` for
/// amplifying history.
pub fn capped_nesterov_mu_par(c_t: f64) -> f64 {
    let c_plus = c_t.max(0.0);
    2.0 * CAPPED_NESTEROV_MU_MAX / (1.0 + (1.0 + 4.0 * c_plus * CAPPED_NESTEROV_MU_MAX).sqrt())
}

/// Realized buffer/delta geometry shared by the capped-Nesterov family:
/// c_t = <b_{t-1}, delta_t> / |delta_t|^2 and the relative transverse
/// residual r_t = |b_{t-1} - c_t delta_t| / |delta_t| (both 0 when
/// delta = 0, leaving the caps inactive).
fn capped_nesterov_geometry(buf: &[f32], delta: &[f32]) -> (f64, f64) {
    let mut dot = 0.0f64;
    let mut buf_norm_sq = 0.0f64;
    let mut delta_norm_sq = 0.0f64;
    for (b, d) in buf.iter().zip(delta) {
        dot += *b as f64 * *d as f64;
        buf_norm_sq += (*b as f64).powi(2);
        delta_norm_sq += (*d as f64).powi(2);
    }
    if delta_norm_sq > 0.0 {
        let c = dot / delta_norm_sq;
        // |b - c*delta|^2 expands to |b|^2 - c*<b, delta> at the projection
        // coefficient; clamp at 0 against cancellation when b ~ c*delta.
        let r = ((buf_norm_sq - c * dot).max(0.0) / delta_norm_sq).sqrt();
        (c, r)
    } else {
        // No measurable geometry (zero delta): caps stay inactive and the
        // EMA relaxes toward mu_max, mirroring rho-adaptive's unmeasured
        // commits.
        (0.0, 0.0)
    }
}

/// Shared per-commit effective momentum of the capped-Nesterov family: the
/// one-sided release EMA of the given cap against the persistent `mu_prev`
/// (updated in place and returned).
fn capped_nesterov_effective_mu(cap: f64, mu_prev: &mut f32) -> f32 {
    let released =
        CAPPED_NESTEROV_EMA_BETA * *mu_prev as f64 + (1.0 - CAPPED_NESTEROV_EMA_BETA) * cap;
    *mu_prev = cap.min(released) as f32;
    *mu_prev
}

/// Capped-Nesterov step. Standard Nesterov recursion, but the momentum is
/// chosen per commit from the realized geometry of the buffer against the
/// merged delta instead of being a fixed CLI constant (`--outer-momentum` is
/// not consumed; the constants above are compile-time). Per commit:
///
///   c_t = <b_{t-1}, delta_t> / |delta_t|^2,
///   r_t = |b_{t-1} - c_t delta_t| / |delta_t|      (both 0 when delta = 0),
///   cap = capped_nesterov_cap(c_t, r_t),
///   mu_t = min(cap, beta * mu_{t-1} + (1 - beta) * cap)   (one-sided EMA),
///   b_t = mu_t b_{t-1} + delta_t;  d_t = delta_t + mu_t b_t;
///   theta -= lr * d_t.
///
/// `mu_prev` is the persistent effective momentum (per fragment, init
/// CAPPED_NESTEROV_INITIAL_MU, threaded like the rho-adaptive `rho_ema`); it
/// is updated to mu_t before the step so callers can hand the exact applied
/// momentum to `materialize_applied_step`. mu_t is rounded to f32 once and
/// the update is delegated to `nesterov_step`, so the applied step and all
/// `OuterStepStats` conventions are bit-for-bit those of plain Nesterov at
/// momentum mu_t.
pub fn capped_nesterov_step(
    params: &mut [f32],
    buf: &mut [f32],
    delta: &[f32],
    lr: f32,
    mu_prev: &mut f32,
) -> OuterStepStats {
    let (c_t, r_t) = capped_nesterov_geometry(buf, delta);
    let mu = capped_nesterov_effective_mu(capped_nesterov_cap(c_t, r_t), mu_prev);
    nesterov_step(params, buf, delta, lr, mu)
}

/// Capped-Nesterov with the relaxed transverse budget
/// `CAPPED_NESTEROV_R_TAU_PERP` (see the constant's rationale). Identical to
/// `capped_nesterov_step` in every other respect, including the `mu_prev`
/// threading and the plain-Nesterov stats conventions.
pub fn capped_nesterov_r_step(
    params: &mut [f32],
    buf: &mut [f32],
    delta: &[f32],
    lr: f32,
    mu_prev: &mut f32,
) -> OuterStepStats {
    let (c_t, r_t) = capped_nesterov_geometry(buf, delta);
    let cap = capped_nesterov_cap_with_tau(c_t, r_t, CAPPED_NESTEROV_R_TAU_PERP);
    let mu = capped_nesterov_effective_mu(cap, mu_prev);
    nesterov_step(params, buf, delta, lr, mu)
}

/// Online Hessian-free secant curvature proxy shared with EXP2.39's geometry
/// panel (`scripts/analyze_geometry_panel.py`): between two consecutive
/// same-fragment commits,
///
///   lambda_hat = [<g_t - g_{t-1}, dtheta_{t-1}>]_+ / (||dtheta_{t-1}||^2 + eps)
///
/// where `g_t` is the current merged delta, `g_{t-1}` the previous merged
/// delta, and `dtheta_{t-1}` the previous APPLIED outer step measured as the
/// parameter displacement `theta_t - theta_{t-1}` (= -(subtracted step); this
/// sign convention is what makes the secant `<Delta g, Delta theta>` a positive
/// curvature under local convexity, matching the panel's `anchor - panchor`).
/// The clamp `[.]_+` keeps the proxy nonnegative; the first commit / a restore
/// (previous step zero) yields lambda_hat = 0 (cap inactive). `prev_delta` and
/// `prev_dtheta` may be shorter than `delta` only when uninitialized, in which
/// case the zip stops early and lambda_hat is 0 — but the production caller
/// keeps them full-length (zero-filled) so the panel form is exact.
fn capped_nesterov_curv_lambda_hat(
    delta: &[f32],
    prev_delta: &[f32],
    prev_dtheta: &[f32],
) -> f64 {
    let mut dot = 0.0f64;
    let mut dtheta_norm_sq = 0.0f64;
    for ((d, pd), pt) in delta.iter().zip(prev_delta).zip(prev_dtheta) {
        let dgrad = *d as f64 - *pd as f64;
        dot += dgrad * *pt as f64;
        dtheta_norm_sq += (*pt as f64).powi(2);
    }
    dot.max(0.0) / (dtheta_norm_sq + CAPPED_NESTEROV_CURV_LAMBDA_EPS)
}

/// Curvature-aware per-commit momentum cap for "capped-nesterov-curv". Shares
/// the aligned cap `mu_par` and the mu_max ceiling with the frozen controller,
/// but REPLACES the geometry-blind transverse cap `mu_perp = sqrt(tau/r_t)`
/// with the curvature-aware `mu_curv` derived from the transverse
/// curvature-energy budget `CAPPED_NESTEROV_CURV_E_PERP`:
///
///   mu_curv = (E_PERP / (lambda_hat * r_t^2 * ||g_t||^2))^{1/4},
///   cap     = min(mu_max, mu_par, mu_curv).
///
/// `||g_t||^2 = delta_norm_sq`. When `lambda_hat`, `r_t`, or `||g_t||` is zero
/// the transverse curvature energy is zero for any mu, so mu_curv = mu_max
/// (inactive). The same sign-reversal guard as `capped_nesterov_cap` applies:
/// if `A_t(cap) = 1 + cap + cap^2 c_t < 0` the cap is zeroed (keeps every
/// admissible mu on the descent side).
pub fn capped_nesterov_curv_cap(
    c_t: f64,
    r_t: f64,
    delta_norm_sq: f64,
    lambda_hat: f64,
) -> f64 {
    let mu_par = capped_nesterov_mu_par(c_t);
    // Transverse curvature-energy denominator lambda_hat * r_t^2 * ||g_t||^2.
    let denom = lambda_hat * r_t * r_t * delta_norm_sq;
    let mu_curv = if denom > 0.0 {
        (CAPPED_NESTEROV_CURV_E_PERP / denom).powf(0.25)
    } else {
        // No transverse curvature energy at any mu: cap inactive.
        CAPPED_NESTEROV_MU_MAX
    };
    let cap = CAPPED_NESTEROV_MU_MAX.min(mu_par).min(mu_curv);
    if 1.0 + cap + cap * cap * c_t < 0.0 {
        0.0
    } else {
        cap
    }
}

/// Curvature-aware capped-Nesterov step ("capped-nesterov-curv"): the
/// T4-sanctioned controller. Identical to `capped_nesterov_step` except the
/// transverse cap is curvature-aware (`capped_nesterov_curv_cap`) using the
/// online secant curvature proxy `lambda_hat_t` instead of the geometry-blind
/// norm budget `tau_perp`. Per commit:
///
///   c_t, r_t   = capped_nesterov_geometry(buf, delta),      (as capped-nesterov)
///   lambda_hat = [<g_t - g_{t-1}, dtheta_{t-1}>]_+ / (|dtheta_{t-1}|^2 + eps),
///   cap        = min(mu_max, mu_par(c_t), mu_curv(r_t, |g_t|, lambda_hat)),
///   mu_t       = min(cap, beta mu_{t-1} + (1 - beta) cap)   (one-sided EMA),
///   theta     -= lr * (delta + mu_t * (mu_t b_{t-1} + delta)).
///
/// `mu_prev` is the persistent effective momentum threaded exactly like the
/// frozen controller's. `prev_delta` (g_{t-1}) and `prev_dtheta` (dtheta_{t-1},
/// the previous applied step as a parameter displacement) are the two extra
/// per-fragment vector states; both are updated in place to the current merged
/// delta and the current applied step (as `theta_t - theta_{t-1}` =
/// `-(lr * (delta + mu_t * b_t))`, bit-identical to the negated
/// `materialize_applied_step` displacement). The step itself is delegated to
/// `nesterov_step` at the f32-rounded mu_t, so the applied step and every
/// `OuterStepStats` field are bit-for-bit plain Nesterov's at momentum mu_t.
pub fn capped_nesterov_curv_step(
    params: &mut [f32],
    buf: &mut [f32],
    delta: &[f32],
    lr: f32,
    mu_prev: &mut f32,
    prev_delta: &mut [f32],
    prev_dtheta: &mut [f32],
) -> OuterStepStats {
    let (c_t, r_t) = capped_nesterov_geometry(buf, delta);
    let delta_norm_sq = norm_sq(delta);
    let lambda_hat = capped_nesterov_curv_lambda_hat(delta, prev_delta, prev_dtheta);
    let cap = capped_nesterov_curv_cap(c_t, r_t, delta_norm_sq, lambda_hat);
    let mu = capped_nesterov_effective_mu(cap, mu_prev);
    let stats = nesterov_step(params, buf, delta, lr, mu);
    // Update the history states for the next commit. `buf` now holds
    // b_t = mu b_{t-1} + delta, so lr * (delta + mu * buf) is exactly the
    // applied step (materialize_applied_step's CappedNesterov branch); the
    // parameter displacement dtheta_t = theta_t - theta_{t-1} is its negation.
    for ((pd, pt), (d, b)) in prev_delta
        .iter_mut()
        .zip(prev_dtheta.iter_mut())
        .zip(delta.iter().zip(buf.iter()))
    {
        *pd = *d;
        *pt = -(lr * (*d + mu * *b));
    }
    stats
}

/// Transverse worker-disagreement curvature-energy budget for the
/// "capped-nesterov-wsub" variant (the DiLoCo-native directional controller).
/// Where `capped-nesterov-curv` uses an ONLINE SCALAR secant curvature proxy
/// `lambda_hat` (isotropic, cross-round memory), this variant uses a FREE,
/// DIRECTIONAL, MEMORY-FREE proxy that a single-worker optimizer cannot see:
/// the M worker deltas `g_i` disagree, and their spread `s_i = g_i - gbar`
/// spans the rank-(M-1) subspace that IS the high-noise / high-curvature
/// directional structure this round (Lean T2: the poison is
/// `d_perp^T H d_perp`, curvature*transverse and ANISOTROPIC). Proxying the
/// (unknown) Hessian `H` by the unnormalized cross-worker disagreement
/// covariance `C = sum_i s_i s_i^T`, the transverse-momentum harm per commit is
///
///   d_perp^T C d_perp = mu^4 * (b_perp^T C b_perp) = mu^4 * E_b
///
/// where `d_perp = mu^2 b_perp` is the Nesterov direction's component
/// orthogonal to the merged delta `g_t` (THEORY.md Prop A.1) and
/// `E_b = b_perp^T C b_perp = sum_i <s_i, b_perp>^2` is the disagreement energy
/// carried by the transverse buffer (`disagreement_transverse_energy`). Note
/// `E_b` is EXACTLY the eigenvalue-weighted energy of `b_perp` in the
/// rank-(M-1) disagreement subspace: with `C = sum_k lambda_k u_k u_k^T` its
/// eigendecomposition, `sum_k lambda_k <u_k, b_perp>^2 = b_perp^T C b_perp` —
/// weighting the subspace projection by the disagreement eigenvalues cancels
/// the basis normalization, so no explicit Gram eigendecomposition is needed to
/// evaluate the curvature-energy proxy T2/T4 call for. Capping the harm at the
/// budget `E_SUB` and solving for the largest admissible mu gives
///
///   mu_wsub = (E_SUB / E_b)^{1/4},
///
/// clamped to [0, mu_max]; `E_b -> 0` (workers agree, or `M = 1`, or the buffer
/// is aligned with `g_t`) leaves the transverse cap inactive (mu_wsub = mu_max)
/// and the aligned cap alone governs.
///
/// Calibration (documented; capture is ON to validate post-hoc): `E_SUB` is set
/// to the curv variant's transverse-energy scale `CAPPED_NESTEROV_CURV_E_PERP`
/// (22.0) so the two curvature-aware caps bind comparably at a representative
/// operating point — there the geometry-blind transverse budget `tau_perp = 1`
/// binds `mu^2 ||b_perp|| <= tau |g_t|`, i.e. `mu^2 <= tau |g_t| / ||b_perp||`,
/// while wsub binds `mu^2 <= sqrt(E_SUB / E_b)`; modeling `C ~ lambda_dis I` on
/// the transverse subspace gives `E_b ~ lambda_dis ||b_perp||^2` and
/// `E_SUB = tau_perp^2 |g_t|^2 lambda_dis`, so with `tau_perp = 1`,
/// `|g_t|^2 ~ 22`, and an order-unity disagreement eigenvalue `lambda_dis ~ 1`,
/// `E_SUB ~ 22`. This is a starting value: it tightens the cap where the
/// transverse buffer concentrates in high-disagreement directions (the T2
/// anisotropic-harm regime) and relaxes toward mu_max where it does not.
pub const CAPPED_NESTEROV_WSUB_E_SUB: f64 = 22.0;

/// Disagreement transverse curvature-energy proxy `E_b = b_perp^T C b_perp =
/// sum_i <s_i, b_perp>^2` for "capped-nesterov-wsub", where `C = sum_i s_i s_i^T`
/// is the unnormalized cross-worker disagreement covariance of the M worker
/// deltas. Computed at merge time (the only place the per-worker deltas exist),
/// stored on the aggregate, and threaded to the step as a single scalar so the
/// preview stays bit-exact.
///
/// - `d_i = anchor - learner_i` is worker i's full-fragment delta;
///   `gbar = (1/M) sum_i d_i` is the UNWEIGHTED mean; `s_i = d_i - gbar`.
/// - `b_perp = buffer - (<buffer, g> / <g, g>) g` is the momentum buffer's
///   component orthogonal to the merged delta `g` (= the delta fed to the
///   Nesterov step). `b_perp`'s DIRECTION — hence `E_b` — depends only on the
///   direction of `g`, so it is invariant to any positive rescale of the merged
///   delta (e.g. the delta-norm-ref renormalization applied later).
/// - `<s_i, b_perp> = <d_i, b_perp> - <gbar, b_perp>`, so with
///   `t_i = <d_i, b_perp>` and `tbar = (1/M) sum_i t_i`,
///   `E_b = sum_i (t_i - tbar)^2` — the spread of how the workers' deltas align
///   with the transverse buffer, needing only M dot products and no d*d matrix.
///
/// Returns 0.0 when there is no measurable disagreement (fewer than two workers,
/// zero merged delta, or an aligned buffer), leaving the transverse cap inactive.
pub fn disagreement_transverse_energy(
    anchor: &[f32],
    learners: &[&[f32]],
    buffer: &[f32],
    merged_delta: &[f32],
) -> f64 {
    let m = learners.len();
    if m < 2 {
        return 0.0;
    }
    // b_perp = buffer - (<buffer, g>/<g, g>) g, the transverse momentum buffer.
    let mut bg = 0.0f64;
    let mut gg = 0.0f64;
    for (b, g) in buffer.iter().zip(merged_delta) {
        bg += *b as f64 * *g as f64;
        gg += (*g as f64).powi(2);
    }
    let coeff = if gg > 0.0 { bg / gg } else { 0.0 };
    let b_perp: Vec<f64> = buffer
        .iter()
        .zip(merged_delta)
        .map(|(b, g)| *b as f64 - coeff * *g as f64)
        .collect();
    // t_i = <d_i, b_perp> = <anchor - learner_i, b_perp>.
    let t: Vec<f64> = learners
        .iter()
        .map(|learner| {
            anchor
                .iter()
                .zip(*learner)
                .zip(&b_perp)
                .map(|((a, l), bp)| (*a as f64 - *l as f64) * *bp)
                .sum::<f64>()
        })
        .collect();
    let tbar = t.iter().sum::<f64>() / m as f64;
    // E_b = sum_i (t_i - tbar)^2 = b_perp^T C b_perp (C the disagreement cov).
    t.iter().map(|ti| (ti - tbar).powi(2)).sum()
}

/// Worker-disagreement transverse cap for "capped-nesterov-wsub". Shares the
/// aligned cap `mu_par` and the `mu_max` ceiling with the frozen controller, but
/// REPLACES the geometry-blind transverse cap `mu_perp = sqrt(tau/r_t)` with the
/// directional, curvature-aware `mu_wsub` from the disagreement-energy budget
/// `CAPPED_NESTEROV_WSUB_E_SUB` (see that constant):
///
///   mu_wsub = (E_SUB / E_b)^{1/4},   cap = min(mu_max, mu_par, mu_wsub).
///
/// `E_b = disagreement_energy` is the transverse buffer's energy in the
/// disagreement covariance. When `E_b <= 0` (workers agree / M = 1 / aligned
/// buffer) the transverse curvature energy is zero for any mu, so mu_wsub =
/// mu_max (inactive). The same sign-reversal guard as `capped_nesterov_cap`
/// applies: if `A_t(cap) = 1 + cap + cap^2 c_t < 0` the cap is zeroed.
pub fn capped_nesterov_wsub_cap(c_t: f64, disagreement_energy: f64) -> f64 {
    let mu_par = capped_nesterov_mu_par(c_t);
    let mu_wsub = if disagreement_energy > 0.0 {
        (CAPPED_NESTEROV_WSUB_E_SUB / disagreement_energy).powf(0.25)
    } else {
        // No transverse disagreement energy at any mu: cap inactive.
        CAPPED_NESTEROV_MU_MAX
    };
    let cap = CAPPED_NESTEROV_MU_MAX.min(mu_par).min(mu_wsub);
    if 1.0 + cap + cap * cap * c_t < 0.0 {
        0.0
    } else {
        cap
    }
}

/// Worker-disagreement-subspace capped-Nesterov step ("capped-nesterov-wsub"):
/// the DiLoCo-native directional controller. Identical to `capped_nesterov_step`
/// except the transverse cap is the worker-disagreement curvature-energy cap
/// `capped_nesterov_wsub_cap` (using the merge-time `disagreement_energy`
/// `E_b = b_perp^T C b_perp`) instead of the geometry-blind norm budget
/// `tau_perp`. Per commit:
///
///   c_t        = <b_{t-1}, delta_t> / |delta_t|^2,   (as capped-nesterov)
///   cap        = min(mu_max, mu_par(c_t), (E_SUB / E_b)^{1/4}),
///   mu_t       = min(cap, beta mu_{t-1} + (1 - beta) cap)   (one-sided EMA),
///   theta     -= lr * (delta + mu_t * (mu_t b_{t-1} + delta)).
///
/// `mu_prev` is the persistent effective momentum threaded exactly like the
/// frozen controller's. The disagreement subspace is recomputed fresh each round
/// from the per-worker deltas (no cross-round DIRECTIONAL memory beyond the
/// Nesterov buffer itself). The step is delegated to `nesterov_step` at the
/// f32-rounded mu_t, so the applied step and every `OuterStepStats` field are
/// bit-for-bit plain Nesterov's at momentum mu_t.
pub fn capped_nesterov_wsub_step(
    params: &mut [f32],
    buf: &mut [f32],
    delta: &[f32],
    lr: f32,
    mu_prev: &mut f32,
    disagreement_energy: f64,
) -> OuterStepStats {
    let (c_t, _r_t) = capped_nesterov_geometry(buf, delta);
    let cap = capped_nesterov_wsub_cap(c_t, disagreement_energy);
    let mu = capped_nesterov_effective_mu(cap, mu_prev);
    nesterov_step(params, buf, delta, lr, mu)
}

/// Gain-compensated capped-Nesterov (see `CAPPED_NESTEROV_GC_GAIN_MIN` for
/// the rationale). The buffer recursion, cap geometry, and `mu_prev`
/// threading are exactly `capped_nesterov_step`'s (transverse budget
/// `CAPPED_NESTEROV_TAU_PERP`); only the applied parameter displacement is
/// rescaled:
///
///   A_t = 1 + mu_t + mu_t^2 c_t,
///   g_t = clamp((1 + mu_max) / max(A_t, eps), GAIN_MIN, GAIN_MAX),
///   b_t = mu_t b_{t-1} + delta_t;  d_t = delta_t + mu_t b_t;
///   theta -= lr * (g_t * d_t).
///
/// `gain` is the persistent applied-gain scalar, threaded like `mu_prev`
/// (init `CAPPED_NESTEROV_GC_INITIAL_GAIN`); it is updated to g_t BEFORE
/// the step so callers can hand the exact applied (f32-rounded) gain to
/// `materialize_applied_step`, whose gc branch reproduces the applied step
/// bit-for-bit. Stats keep the plain-Nesterov conventions with the step and
/// direction scaled by g_t (the cosine and history/current ratio are
/// invariant under the common positive scale).
pub fn capped_nesterov_gc_step(
    params: &mut [f32],
    buf: &mut [f32],
    delta: &[f32],
    lr: f32,
    mu_prev: &mut f32,
    gain: &mut f32,
) -> OuterStepStats {
    let (c_t, r_t) = capped_nesterov_geometry(buf, delta);
    let mu = capped_nesterov_effective_mu(capped_nesterov_cap(c_t, r_t), mu_prev);
    // A_t from the f32-rounded effective momentum actually applied below.
    let mu_f64 = mu as f64;
    let a_t = 1.0 + mu_f64 + mu_f64 * mu_f64 * c_t;
    *gain = (((1.0 + CAPPED_NESTEROV_MU_MAX) / a_t.max(CAPPED_NESTEROV_GC_A_EPS))
        .clamp(CAPPED_NESTEROV_GC_GAIN_MIN, CAPPED_NESTEROV_GC_GAIN_MAX)) as f32;
    let g = *gain;
    let mut step_norm_sq = 0.0;
    let mut direction_norm_sq = 0.0;
    let mut delta_norm_sq = 0.0;
    let mut direction_delta_dot = 0.0;
    let mut history_norm_sq = 0.0;
    let mut current_norm_sq = 0.0;
    for ((p, b), d) in params.iter_mut().zip(buf.iter_mut()).zip(delta) {
        let previous_buffer = *b;
        *b = mu * *b + *d;
        // Write the buffer first and derive the step from it in exactly the
        // form of materialize_applied_step's gc branch so this path is
        // bit-identical to lr * (gain * (delta + mu * buf)).
        let direction = g * (*d + mu * *b);
        let step = lr * direction;
        *p -= step;

        let direction = direction as f64;
        let delta = *d as f64;
        let step = step as f64;
        let history = (g * (mu * (mu * previous_buffer))) as f64;
        let current = (g * (*d + mu * *d)) as f64;
        step_norm_sq += step * step;
        direction_norm_sq += direction * direction;
        delta_norm_sq += delta * delta;
        direction_delta_dot += direction * delta;
        history_norm_sq += history * history;
        current_norm_sq += current * current;
    }
    finish_outer_step_stats(
        step_norm_sq,
        direction_norm_sq,
        delta_norm_sq,
        direction_delta_dot,
        history_norm_sq,
        current_norm_sq,
        false,
    )
}

/// Block second-moment (beta1 = 0) outer optimizers, shared kernel for
/// `block-rms` and `block-yogi`. No first-moment / directional memory: the
/// only cross-round state is one scalar per tensor block, the second-moment
/// EMA of the block's per-coordinate mean squared pseudo-gradient magnitude.
pub const BLOCK_ADAPTIVE_BETA2: f64 = 0.95;
/// Denominator floor for the per-block whitening `g / (sqrt(v) + eps)`.
pub const BLOCK_ADAPTIVE_EPS: f64 = 1e-8;
/// Denominator floor for the global norm-match back to the plain-SGD step norm.
pub const BLOCK_ADAPTIVE_NORM_EPS: f64 = 1e-12;

/// One block second-moment step (`yogi = false` -> RMS, `true` -> Yogi).
///
/// For each tensor block `l` (delimited by `tensor_numels`, summing to
/// `delta.len()`), with `d_l` = block size and `s_l = ||g_l||^2 / d_l` the
/// current mean squared magnitude:
///
///   RMS  : v_l <- beta2 v_l + (1 - beta2) s_l
///   Yogi : v_l <- v_l - (1 - beta2) sign(v_l - s_l) s_l
///   u_l  = g_l / (sqrt(v_l) + eps)
///
/// then a single GLOBAL norm-match restores the plain-SGD step norm,
/// `u <- u * ||g|| / (||u|| + eps)`, and `theta <- theta - lr * u`. `v_l` is a
/// scalar with NO direction memory; `block_v` (one entry per tensor) is updated
/// in place. The whitened, norm-matched direction `u` is written into `buf` and
/// the step is derived as `lr * buf`, so this path is bit-identical to
/// `materialize_applied_step`'s `lr * buf` branch (as with the normalized-EMA
/// and rho-adaptive optimizers). `--outer-momentum` is not consumed.
pub fn block_second_moment_step(
    params: &mut [f32],
    buf: &mut [f32],
    delta: &[f32],
    lr: f32,
    tensor_numels: &[usize],
    block_v: &mut [f32],
    yogi: bool,
) -> OuterStepStats {
    debug_assert_eq!(tensor_numels.iter().sum::<usize>(), delta.len());
    debug_assert_eq!(tensor_numels.len(), block_v.len());
    // Pass 1: per-block second-moment update and unnormalized whitened
    // direction into `buf`; accumulate ||g|| and ||u|| for the global match.
    let mut offset = 0usize;
    let mut g_norm_sq = 0.0f64;
    let mut u_norm_sq = 0.0f64;
    for (block_index, &numel) in tensor_numels.iter().enumerate() {
        let block = &delta[offset..offset + numel];
        let block_norm_sq: f64 = block.iter().map(|value| (*value as f64).powi(2)).sum();
        let s_l = if numel > 0 {
            block_norm_sq / numel as f64
        } else {
            0.0
        };
        let v_prev = block_v[block_index] as f64;
        let v_new = if yogi {
            // v -= (1 - beta2) * sign(v - s) * s; equivalently move v toward s
            // by (1-beta2)*s, with the additive (not multiplicative) Yogi
            // update that resists variance blow-ups from large s.
            let sign = (v_prev - s_l).signum();
            (v_prev - (1.0 - BLOCK_ADAPTIVE_BETA2) * sign * s_l).max(0.0)
        } else {
            BLOCK_ADAPTIVE_BETA2 * v_prev + (1.0 - BLOCK_ADAPTIVE_BETA2) * s_l
        };
        block_v[block_index] = v_new as f32;
        let denom = (v_new.sqrt() + BLOCK_ADAPTIVE_EPS) as f32;
        for (b, d) in buf[offset..offset + numel].iter_mut().zip(block) {
            let u = *d / denom;
            *b = u;
            u_norm_sq += (u as f64).powi(2);
        }
        g_norm_sq += block_norm_sq;
        offset += numel;
    }
    // Global norm-match: rescale u back to the plain-SGD step norm ||g||.
    let g_norm = g_norm_sq.sqrt();
    let u_norm = u_norm_sq.sqrt();
    let scale = (g_norm / (u_norm + BLOCK_ADAPTIVE_NORM_EPS)) as f32;
    // Pass 2: apply u * scale via the buffer so `lr * buf` reproduces the step.
    let mut step_norm_sq = 0.0f64;
    let mut direction_norm_sq = 0.0f64;
    let mut delta_norm_sq = 0.0f64;
    let mut direction_delta_dot = 0.0f64;
    for ((p, b), d) in params.iter_mut().zip(buf.iter_mut()).zip(delta) {
        *b *= scale;
        let step = lr * *b;
        *p -= step;
        let direction = *b as f64;
        step_norm_sq += (step as f64).powi(2);
        direction_norm_sq += direction * direction;
        delta_norm_sq += (*d as f64).powi(2);
        direction_delta_dot += direction * *d as f64;
    }
    // No history contribution (beta1 = 0): history/current ratio is 0.
    finish_outer_step_stats(
        step_norm_sq,
        direction_norm_sq,
        delta_norm_sq,
        direction_delta_dot,
        0.0,
        delta_norm_sq,
        false,
    )
}

// ---------------------------------------------------------------------------
// Chebyshev-SGD: memoryless spectral-acceleration outer optimizer.
//
// The escape from the transverse-momentum poison (docs/LEAN_THEOREMS.md T1-T3,
// docs/POLYNOMIAL_OUTER_OPTIMIZER.md family 1): instead of a directional
// first-moment buffer, apply a short cyclical learning-rate schedule around the
// tuned base LR whose K-step product is the shifted-Chebyshev residual
// polynomial for a symmetric operator with eigenvalues in [1, kappa]. This is
// pure scaled SGD each commit -- the update direction is the current merged
// pseudo-gradient `delta`, scaled by a SCALAR multiplier m_k -- so it carries
// no old velocity into newly-steep directions. The only cross-round state is a
// scalar cycle-phase counter per fragment (`cheb_phase`); the optimizer buffer
// `buf` is used ONLY as read-only scratch to detect a geometry change (the
// cosine between the current delta and the previous applied direction) and is
// overwritten every commit with the current scaled delta -- it is never added
// to the step, so no directional memory enters the update. Because the step
// ACQUIRES a curvature estimate (the [1, kappa] range comes from the measured
// spectral width, EXP2 outer-dynamics diagnostic) rather than a gain/cosine
// scalar of (g, buffer), Lean T3's geometry-blind impossibility does not
// directly apply.

/// Cycle length K (small -> large -> mid -> small).
pub const CHEB_SGD_CYCLE: usize = 4;
/// Estimated symmetric-part condition number kappa = lambda_max / lambda_min of
/// the local pseudo-gradient dynamics operator. Calibrated to the measured
/// short-horizon spectral width (rank2/rank16 H16 condition number ~= 19-20 in
/// the Krylov subspace, EXP2 outer-dynamics diagnostic). kappa = 1 makes every
/// multiplier exactly 1 (degenerate to plain SGD at the base LR).
pub const CHEB_SGD_KAPPA: f64 = 20.0;
/// Hard multiplier bounds (robust-control safety): the applied step is always
/// within [m_min, m_max] * base LR of tuned SGD, and the clamped schedule still
/// averages to 1 over the cycle (see `cheb_sgd_multipliers`).
pub const CHEB_SGD_M_MIN: f64 = 0.5;
pub const CHEB_SGD_M_MAX: f64 = 2.0;
/// Restart the cycle (reset phase to 0) when the merged delta's cosine against
/// the previous applied direction drops below this: a large geometry change
/// invalidates the [1, kappa] schedule, so damp with the smallest step first.
pub const CHEB_SGD_RESTART_COS: f32 = 0.0;

/// The K ordered, arithmetic-mean-normalized, hard-bounded Chebyshev step-size
/// multipliers for condition number `kappa` (>= 1), in the Leja "small -> large
/// -> mid -> small" order. Returns `[1.0; K]` exactly when `kappa == 1`.
///
/// Construction (docs/POLYNOMIAL_OUTER_OPTIMIZER.md; codex gpt-5.6-sol derivation):
/// with `a = 1`, `b = kappa`, center `c = (a+b)/2`, half-width `d = (b-a)/2`,
/// the shifted-Chebyshev nodes give raw steps `s_j = 1 / (c +/- d * cos(theta))`
/// for `theta in {pi/8, 3pi/8}`. Ordered small->large->mid->small:
///   s = [ 1/(c+d q), 1/(c-d q), 1/(c-d r), 1/(c+d r) ],  q=cos(pi/8), r=cos(3pi/8).
/// Arithmetic-mean normalization `m_j = tau * s_j` anchors the cycle-average
/// multiplier to 1 (so the average LR stays the tuned base) and preserves
/// `m_j = 1` at kappa = 1; `tau` is chosen by a monotone bisection so the
/// hard-clamped multipliers still average exactly 1 (feasible since
/// m_min <= 1 <= m_max).
pub fn cheb_sgd_multipliers(kappa: f64) -> [f64; CHEB_SGD_CYCLE] {
    if !(kappa > 1.0) {
        return [1.0; CHEB_SGD_CYCLE];
    }
    let a = 1.0;
    let b = kappa;
    let c = 0.5 * (a + b);
    let d = 0.5 * (b - a);
    let q = (std::f64::consts::PI / 8.0).cos();
    let r = (3.0 * std::f64::consts::PI / 8.0).cos();
    // small -> large -> mid-large -> mid-small
    let s = [
        1.0 / (c + d * q),
        1.0 / (c - d * q),
        1.0 / (c - d * r),
        1.0 / (c + d * r),
    ];
    // Monotone bisection for tau with mean(clip(tau*s, m_min, m_max)) == 1.
    let mean_clamped = |tau: f64| -> f64 {
        s.iter()
            .map(|sj| (tau * sj).clamp(CHEB_SGD_M_MIN, CHEB_SGD_M_MAX))
            .sum::<f64>()
            / CHEB_SGD_CYCLE as f64
    };
    let mut lo = 0.0f64;
    let mut hi = CHEB_SGD_M_MAX / s.iter().cloned().fold(f64::INFINITY, f64::min);
    // Ensure the bracket contains the root (mean is nondecreasing in tau).
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if mean_clamped(mid) < 1.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let tau = 0.5 * (lo + hi);
    let mut m = [0.0f64; CHEB_SGD_CYCLE];
    for j in 0..CHEB_SGD_CYCLE {
        m[j] = (tau * s[j]).clamp(CHEB_SGD_M_MIN, CHEB_SGD_M_MAX);
    }
    m
}

/// One Chebyshev-SGD commit. `phase` is the persistent scalar cycle counter in
/// `0..CHEB_SGD_CYCLE` (as f32; reset to 0 on a checkpoint restore, like the
/// other outer-optimizer scalar states). The applied step is
/// `lr * m_k * delta`; `buf` is overwritten with `m_k * delta` so the
/// `materialize_applied_step` `lr * buf` branch is bit-identical, and its
/// PRIOR contents (last commit's applied direction) are read first to detect a
/// geometry change. No first-moment memory enters the direction.
pub fn cheb_sgd_step(
    params: &mut [f32],
    buf: &mut [f32],
    delta: &[f32],
    lr: f32,
    phase: &mut f32,
) -> OuterStepStats {
    debug_assert_eq!(params.len(), buf.len());
    debug_assert_eq!(buf.len(), delta.len());
    // Geometry-change restart: cosine of the current delta against the previous
    // applied direction held in `buf` (read-only; not injected into the step).
    let buf_norm_sq = norm_sq(buf);
    let delta_norm_sq = norm_sq(delta);
    if buf_norm_sq > 0.0 && delta_norm_sq > 0.0 {
        let dot: f64 = buf
            .iter()
            .zip(delta)
            .map(|(b, d)| *b as f64 * *d as f64)
            .sum();
        let cosine = (dot / (buf_norm_sq * delta_norm_sq).sqrt()).clamp(-1.0, 1.0);
        if cosine < CHEB_SGD_RESTART_COS as f64 {
            *phase = 0.0;
        }
    }
    let restarted = *phase == 0.0 && buf_norm_sq > 0.0;
    let k = ((*phase as usize) % CHEB_SGD_CYCLE).min(CHEB_SGD_CYCLE - 1);
    let mult = cheb_sgd_multipliers(CHEB_SGD_KAPPA)[k] as f32;

    let mut step_norm_sq = 0.0;
    let mut direction_norm_sq = 0.0;
    let mut direction_delta_dot = 0.0;
    for ((p, b), d) in params.iter_mut().zip(buf.iter_mut()).zip(delta) {
        let direction = mult * *d;
        *b = direction; // scratch: current scaled delta, for materialize + next-commit probe
        let step = lr * direction;
        *p -= step;
        let direction = direction as f64;
        let step = step as f64;
        step_norm_sq += step * step;
        direction_norm_sq += direction * direction;
        direction_delta_dot += direction * *d as f64;
    }
    *phase = (((k + 1) % CHEB_SGD_CYCLE) as f32).floor();
    // Direction is a positive scalar multiple of delta, so cosine is +1 and the
    // history contribution is zero (memoryless): report current == direction.
    finish_outer_step_stats(
        step_norm_sq,
        direction_norm_sq,
        delta_norm_sq,
        direction_delta_dot,
        0.0,
        direction_norm_sq,
        restarted,
    )
}

/// Purely scale a nominal applied-step vector and apply it once to the same
/// f32 base parameters used by the production outer step. Returning both the
/// scaled vector and resulting parameters keeps action previews consistent on
/// the f32 lattice instead of reconstructing a step from rounded parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct ScaledAppliedStep {
    pub params: Vec<f32>,
    pub applied_step: Vec<f32>,
    pub applied_step_norm: f64,
}

pub fn scale_applied_step(
    base_params: &[f32],
    applied_step: &[f32],
    scalar: f64,
) -> Option<ScaledAppliedStep> {
    if base_params.len() != applied_step.len() || !scalar.is_finite() || scalar < 0.0 {
        return None;
    }
    let mut params = Vec::with_capacity(base_params.len());
    let mut scaled_step = Vec::with_capacity(applied_step.len());
    let mut norm_sq = 0.0f64;
    for (&base, &step) in base_params.iter().zip(applied_step) {
        let scaled = (scalar * step as f64) as f32;
        let param = base - scaled;
        if !scaled.is_finite() || !param.is_finite() {
            return None;
        }
        params.push(param);
        scaled_step.push(scaled);
        norm_sq += (scaled as f64).powi(2);
    }
    if !norm_sq.is_finite() {
        return None;
    }
    Some(ScaledAppliedStep {
        params,
        applied_step: scaled_step,
        applied_step_norm: norm_sq.sqrt(),
    })
}

/// w_m = c_tokens² / c_steps ("quantity × quality").
pub fn learner_weight(c_tokens: u64, c_steps: u32) -> f64 {
    if c_steps == 0 {
        0.0
    } else {
        let t = c_tokens as f64;
        t * t / c_steps as f64
    }
}

fn l2_norm(anchor: &[f32], learner: &[f32]) -> f64 {
    anchor
        .iter()
        .zip(learner)
        .map(|(a, l)| {
            let d = (*a - *l) as f64;
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

/// Weighted direct averaging: out[i] = Σ_m w_m (anchor[i] − learner_m[i]) / Σ w.
pub fn merge_avg(anchor: &[f32], learners: &[&[f32]], weights: &[f64], out: &mut [f32]) {
    let wsum: f64 = weights.iter().sum();
    if wsum <= 0.0 {
        out.fill(0.0);
        return;
    }
    out.fill(0.0);
    for (learner, &w) in learners.iter().zip(weights) {
        let w = (w / wsum) as f32;
        for ((o, a), l) in out.iter_mut().zip(anchor).zip(*learner) {
            *o += w * (*a - *l);
        }
    }
}

/// Weighted radial-directional averaging over one tensor slice.
pub fn merge_rda(anchor: &[f32], learners: &[&[f32]], weights: &[f64], out: &mut [f32]) {
    let wsum: f64 = weights.iter().sum();
    if wsum <= 0.0 {
        out.fill(0.0);
        return;
    }
    let norms: Vec<f64> = learners.iter().map(|l| l2_norm(anchor, l)).collect();
    let radial: f64 = norms
        .iter()
        .zip(weights)
        .map(|(n, w)| n * w)
        .sum::<f64>()
        / wsum;

    // Weighted mean of unit directions, φ(0) := 0.
    out.fill(0.0);
    for ((learner, &w), &n) in learners.iter().zip(weights).zip(&norms) {
        if n == 0.0 {
            continue;
        }
        let coef = (w / wsum / n) as f32;
        for ((o, a), l) in out.iter_mut().zip(anchor).zip(*learner) {
            *o += coef * (*a - *l);
        }
    }
    let mean_dir_norm = out.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt();
    if mean_dir_norm < 1e-12 {
        // Degenerate (all-zero or cancelling directions): fall back to Avg.
        merge_avg(anchor, learners, weights, out);
        return;
    }
    let scale = (radial / mean_dir_norm) as f32;
    for o in out.iter_mut() {
        *o *= scale;
    }
}

/// Denominator floor for the worker-SNR confidence and its global norm-match
/// (see `merge_worker_snr`).
pub const WORKER_SNR_EPS: f64 = 1e-12;

/// Worker-SNR consensus merge over a whole fragment (the original
/// contribution). Unlike avg/RDA/iso this needs the per-worker deltas AND all
/// tensor blocks at once, because it shrinks each block by its cross-worker
/// signal-to-noise ratio and then restores the plain-mean step norm GLOBALLY.
///
/// For each tensor block `l` (delimited by `tensor_numels`, summing to
/// `anchor.len()`), with `M` workers, `d_l` = block size, unweighted mean
/// delta `gbar_l = (1/M) sum_i (anchor - learner_i)` over the block, and
/// unbiased cross-worker variance `sigma_l^2 = (1/(M-1)) sum_i ||delta_i -
/// gbar_l||^2 / d_l` (0 when `M = 1`):
///
///   q_l = (||gbar_l||^2 / d_l) / (||gbar_l||^2 / d_l + sigma_l^2 / M + eps)
///   u_l = q_l * gbar_l
///
/// High-consensus blocks (small sigma) keep `q_l ~ 1`; high-disagreement
/// blocks are shrunk. A single global norm-match `u <- u * ||gbar|| /
/// (||u|| + eps)` then restores the plain-mean merged-delta norm before the
/// outer optimizer (plain SGD) runs. `M = 1` is the identity (`u = gbar`).
/// `weights` are accepted for signature parity with the other merges but the
/// consensus statistics are deliberately unweighted (every worker is one vote).
pub fn merge_worker_snr(
    anchor: &[f32],
    learners: &[&[f32]],
    _weights: &[f64],
    tensor_numels: &[usize],
    out: &mut [f32],
) {
    let m = learners.len();
    if m == 0 {
        out.fill(0.0);
        return;
    }
    let inv_m = 1.0 / m as f64;
    let mut offset = 0usize;
    let mut gbar_norm_sq_total = 0.0f64;
    let mut u_norm_sq_total = 0.0f64;
    for &numel in tensor_numels {
        let block = offset..offset + numel;
        // gbar_l = unweighted mean delta over the block, written into out.
        for j in block.clone() {
            let mut sum = 0.0f64;
            for learner in learners {
                sum += (anchor[j] - learner[j]) as f64;
            }
            out[j] = (sum * inv_m) as f32;
        }
        let gbar_norm_sq: f64 = out[block.clone()]
            .iter()
            .map(|value| (*value as f64).powi(2))
            .sum();
        // Cross-worker variance: sigma^2 = (1/(M-1)) sum_i ||delta_i - gbar||^2 / d.
        let sigma_sq = if m > 1 && numel > 0 {
            let mut sum_sq_dev = 0.0f64;
            for learner in learners {
                for j in block.clone() {
                    let dev = (anchor[j] - learner[j]) as f64 - out[j] as f64;
                    sum_sq_dev += dev * dev;
                }
            }
            sum_sq_dev / ((m - 1) as f64) / numel as f64
        } else {
            0.0
        };
        let mean_sq_gbar = if numel > 0 {
            gbar_norm_sq / numel as f64
        } else {
            0.0
        };
        let q_l = mean_sq_gbar / (mean_sq_gbar + sigma_sq * inv_m + WORKER_SNR_EPS);
        let q_l_f32 = q_l as f32;
        for j in block.clone() {
            let u = q_l_f32 * out[j];
            out[j] = u;
            u_norm_sq_total += (u as f64).powi(2);
        }
        gbar_norm_sq_total += gbar_norm_sq;
        offset += numel;
    }
    // Global norm-match back to the plain-mean merged-delta norm ||gbar||.
    let scale = (gbar_norm_sq_total.sqrt() / (u_norm_sq_total.sqrt() + WORKER_SNR_EPS)) as f32;
    for value in out.iter_mut() {
        *value *= scale;
    }
}

/// Relative cutoff below which a singular value of the averaged delta is
/// treated as zero by the iso transform. Directions in the numerical null
/// space are dropped rather than inflated to σ̄ (their singular vectors are
/// arbitrary); the paper's ablation attributes the gain to clipping HIGH
/// singular values down to the mean, which this preserves exactly.
const ISO_SINGULAR_VALUE_RTOL: f64 = 1e-9;
/// Cyclic-Jacobi sweep budget for the Gram eigendecomposition. Convergence
/// is quadratic; small LoRA-side Gram matrices settle in a handful of
/// sweeps, and the loop exits early on a converged off-diagonal.
const ISO_JACOBI_MAX_SWEEPS: usize = 64;

/// Iso-C-style isotropic aggregation over one tensor slice (IsoLoCo,
/// arXiv 2607.03011, Alg. 2; Iso-C from the isotropic model-merging line).
/// The learner deltas are direct-averaged (weighted, like `merge_avg`) and
/// the averaged pseudo-gradient, viewed as a `rows`×`cols` row-major
/// matrix Δ with SVD Δ = U·diag(σ)·Vᵀ, is replaced by the
/// spectrally-flattened
///
///   Δ_iso = σ̄·U·Vᵀ,  σ̄ = (1/min(rows,cols))·Σ_k σ_k,
///
/// equivalently the whitening transform σ̄·(ΔΔᵀ)^(-1/2)·Δ (pseudo-inverse
/// square root). Computed exactly on the Gram matrix of the SMALLER side in
/// f64 — O(min² ·max) plus an O(min³) Jacobi eigendecomposition — so cost
/// stays proportional to the LoRA rank, not the full matrix. A shape whose
/// product does not match the slice keeps the plain weighted average
/// (callers validate shapes at HELLO decode; this is a deterministic
/// safety net). The outer optimizer (Nesterov by default) is applied by the
/// caller, which is exactly the paper's IsoLoCo composition.
pub fn merge_iso(
    anchor: &[f32],
    learners: &[&[f32]],
    weights: &[f64],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    merge_avg(anchor, learners, weights, out);
    if rows == 0 || cols == 0 || rows.saturating_mul(cols) != out.len() {
        return;
    }
    iso_flatten_spectrum(out, rows, cols);
}

/// Replace the `rows`×`cols` row-major matrix `m` (in place) by σ̄·U·Vᵀ,
/// dropping numerically-null directions (see `ISO_SINGULAR_VALUE_RTOL`).
/// σ̄ averages over all min(rows, cols) singular values, zeros included,
/// matching the paper's σ̄ = (1/r)Σσ_k with r = min(m, n). A zero matrix is
/// left untouched.
pub fn iso_flatten_spectrum(m: &mut [f32], rows: usize, cols: usize) {
    debug_assert_eq!(rows * cols, m.len());
    let k = rows.min(cols);
    if k == 0 {
        return;
    }
    let values: Vec<f64> = m.iter().map(|value| *value as f64).collect();
    // Gram matrix of the smaller side: G = ΔΔᵀ when rows ≤ cols (k = rows),
    // else G = ΔᵀΔ (k = cols). Eigenvectors of G are the corresponding
    // singular vectors; eigenvalues are σ².
    let gram_on_rows = rows <= cols;
    let mut gram = vec![0.0f64; k * k];
    for i in 0..k {
        for j in i..k {
            let mut acc = 0.0f64;
            if gram_on_rows {
                for c in 0..cols {
                    acc += values[i * cols + c] * values[j * cols + c];
                }
            } else {
                for r in 0..rows {
                    acc += values[r * cols + i] * values[r * cols + j];
                }
            }
            gram[i * k + j] = acc;
            gram[j * k + i] = acc;
        }
    }
    let mut basis = vec![0.0f64; k * k];
    jacobi_eigh(&mut gram, &mut basis, k);
    let sigmas: Vec<f64> = (0..k).map(|i| gram[i * k + i].max(0.0).sqrt()).collect();
    let sigma_max = sigmas.iter().cloned().fold(0.0f64, f64::max);
    if sigma_max <= 0.0 {
        return; // zero delta: nothing to whiten
    }
    let sigma_bar = sigmas.iter().sum::<f64>() / k as f64;
    let cutoff = sigma_max * ISO_SINGULAR_VALUE_RTOL;
    // W = Q·diag(σ̄/σ_k or 0)·Qᵀ, the scaled (pseudo-)whitening matrix.
    let mut whiten = vec![0.0f64; k * k];
    for j in 0..k {
        if sigmas[j] <= cutoff {
            continue;
        }
        let gain = sigma_bar / sigmas[j];
        for a in 0..k {
            let qa = basis[a * k + j] * gain;
            if qa == 0.0 {
                continue;
            }
            for b in 0..k {
                whiten[a * k + b] += qa * basis[b * k + j];
            }
        }
    }
    // Δ_iso = W·Δ (rows ≤ cols) or Δ·W (cols < rows).
    if gram_on_rows {
        for c in 0..cols {
            for r in 0..k {
                let mut acc = 0.0f64;
                for t in 0..k {
                    acc += whiten[r * k + t] * values[t * cols + c];
                }
                m[r * cols + c] = acc as f32;
            }
        }
    } else {
        for r in 0..rows {
            for c in 0..k {
                let mut acc = 0.0f64;
                for t in 0..k {
                    acc += values[r * cols + t] * whiten[t * k + c];
                }
                m[r * cols + c] = acc as f32;
            }
        }
    }
}

/// Deterministic cyclic-Jacobi eigendecomposition of the symmetric matrix
/// `g` (k×k, row-major, overwritten; eigenvalues end up on its diagonal).
/// `q` receives the orthonormal eigenvectors as COLUMNS: g_in = Q·Λ·Qᵀ.
fn jacobi_eigh(g: &mut [f64], q: &mut [f64], k: usize) {
    for i in 0..k {
        for j in 0..k {
            q[i * k + j] = if i == j { 1.0 } else { 0.0 };
        }
    }
    for _ in 0..ISO_JACOBI_MAX_SWEEPS {
        let mut off_sq = 0.0f64;
        let mut diag_sq = 0.0f64;
        for i in 0..k {
            diag_sq += g[i * k + i] * g[i * k + i];
            for j in i + 1..k {
                off_sq += g[i * k + j] * g[i * k + j];
            }
        }
        if off_sq <= diag_sq.max(f64::MIN_POSITIVE) * 1e-30 {
            break;
        }
        for p in 0..k {
            for r in p + 1..k {
                let gpr = g[p * k + r];
                if gpr == 0.0 {
                    continue;
                }
                // Rotation zeroing g[p][r] (Golub & Van Loan 8.4).
                let tau = (g[r * k + r] - g[p * k + p]) / (2.0 * gpr);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    1.0 / (tau - (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                for i in 0..k {
                    let gip = g[i * k + p];
                    let gir = g[i * k + r];
                    g[i * k + p] = c * gip - s * gir;
                    g[i * k + r] = s * gip + c * gir;
                }
                for i in 0..k {
                    let gpi = g[p * k + i];
                    let gri = g[r * k + i];
                    g[p * k + i] = c * gpi - s * gri;
                    g[r * k + i] = s * gpi + c * gri;
                }
                for i in 0..k {
                    let qip = q[i * k + p];
                    let qir = q[i * k + r];
                    q[i * k + p] = c * qip - s * qir;
                    q[i * k + r] = s * qip + c * qir;
                }
            }
        }
    }
}

/// SGD + Nesterov momentum treating `delta` as the gradient:
/// buf ← μ·buf + Δ;  θ ← θ − lr·(Δ + μ·buf).
pub fn nesterov_step(
    params: &mut [f32],
    buf: &mut [f32],
    delta: &[f32],
    lr: f32,
    mu: f32,
) -> OuterStepStats {
    let mut step_norm_sq = 0.0;
    let mut direction_norm_sq = 0.0;
    let mut delta_norm_sq = 0.0;
    let mut direction_delta_dot = 0.0;
    let mut history_norm_sq = 0.0;
    let mut current_norm_sq = 0.0;
    for ((p, b), d) in params.iter_mut().zip(buf.iter_mut()).zip(delta) {
        let previous_buffer = *b;
        *b = mu * *b + *d;
        let direction = *d + mu * *b;
        let step = lr * direction;
        *p -= step;

        let direction = direction as f64;
        let delta = *d as f64;
        let step = step as f64;
        let history = (mu * (mu * previous_buffer)) as f64;
        let current = (*d + mu * *d) as f64;
        step_norm_sq += step * step;
        direction_norm_sq += direction * direction;
        delta_norm_sq += delta * delta;
        direction_delta_dot += direction * delta;
        history_norm_sq += history * history;
        current_norm_sq += current * current;
    }
    finish_outer_step_stats(
        step_norm_sq,
        direction_norm_sq,
        delta_norm_sq,
        direction_delta_dot,
        history_norm_sq,
        current_norm_sq,
        false,
    )
}

fn norm_sq(values: &[f32]) -> f64 {
    values.iter().map(|value| (*value as f64).powi(2)).sum()
}

fn is_zero(values: &[f32]) -> bool {
    values.iter().all(|value| *value == 0.0)
}

fn update_normalized_ema(
    buf: &mut [f32],
    delta: &[f32],
    beta: f32,
    buf_is_zero: bool,
    delta_is_zero: bool,
) {
    if buf_is_zero && !delta_is_zero {
        buf.copy_from_slice(delta);
        return;
    }
    for (b, d) in buf.iter_mut().zip(delta) {
        *b = beta * *b + (1.0 - beta) * *d;
    }
}

fn apply_buffer(params: &mut [f32], buf: &[f32], delta: &[f32], lr: f32) -> (f64, f64, f64) {
    let mut step_norm_sq = 0.0;
    let mut direction_norm_sq = 0.0;
    let mut direction_delta_dot = 0.0;
    for ((p, b), d) in params.iter_mut().zip(buf).zip(delta) {
        let step = lr * *b;
        *p -= step;
        let step = step as f64;
        let direction = *b as f64;
        step_norm_sq += step * step;
        direction_norm_sq += direction * direction;
        direction_delta_dot += direction * *d as f64;
    }
    (step_norm_sq, direction_norm_sq, direction_delta_dot)
}

fn finish_outer_step_stats(
    step_norm_sq: f64,
    direction_norm_sq: f64,
    delta_norm_sq: f64,
    direction_delta_dot: f64,
    history_norm_sq: f64,
    current_norm_sq: f64,
    restarted: bool,
) -> OuterStepStats {
    let direction_delta_cosine = if direction_norm_sq > 0.0
        && delta_norm_sq > 0.0
        && direction_norm_sq.is_finite()
        && delta_norm_sq.is_finite()
        && direction_delta_dot.is_finite()
    {
        let cosine = direction_delta_dot / (direction_norm_sq * delta_norm_sq).sqrt();
        cosine.is_finite().then(|| cosine.clamp(-1.0, 1.0))
    } else {
        None
    };
    let history_current_norm_ratio =
        if current_norm_sq > 0.0 && current_norm_sq.is_finite() && history_norm_sq.is_finite() {
            let ratio = (history_norm_sq / current_norm_sq).sqrt();
            ratio.is_finite().then_some(ratio)
        } else {
            None
        };
    OuterStepStats {
        applied_step_norm: step_norm_sq.sqrt(),
        direction_delta_cosine,
        history_current_norm_ratio,
        restarted,
    }
}

fn ema_contribution_norms(
    previous_buffer_norm_sq: f64,
    delta_norm_sq: f64,
    beta: f32,
    initialized_from_delta: bool,
) -> (f64, f64) {
    if initialized_from_delta {
        return (0.0, delta_norm_sq);
    }
    let history_scale = beta as f64;
    let current_scale = (1.0f32 - beta) as f64;
    (
        history_scale * history_scale * previous_buffer_norm_sq,
        current_scale * current_scale * delta_norm_sq,
    )
}

/// Unit-gain exponential moving average of the merged pseudo-gradient.
/// A zero buffer initializes from the first nonzero delta, avoiding the
/// usual EMA warmup attenuation.
pub fn normalized_ema_step(
    params: &mut [f32],
    buf: &mut [f32],
    delta: &[f32],
    lr: f32,
    beta: f32,
) -> OuterStepStats {
    debug_assert_eq!(params.len(), buf.len());
    debug_assert_eq!(buf.len(), delta.len());
    let buf_is_zero = is_zero(buf);
    let delta_is_zero = is_zero(delta);
    let previous_buffer_norm_sq = norm_sq(buf);
    let delta_norm_sq = norm_sq(delta);
    let initialized_from_delta = buf_is_zero && !delta_is_zero;
    update_normalized_ema(buf, delta, beta, buf_is_zero, delta_is_zero);
    let (step_norm_sq, direction_norm_sq, direction_delta_dot) =
        apply_buffer(params, buf, delta, lr);
    let (history_norm_sq, current_norm_sq) = ema_contribution_norms(
        previous_buffer_norm_sq,
        delta_norm_sq,
        beta,
        initialized_from_delta,
    );
    finish_outer_step_stats(
        step_norm_sq,
        direction_norm_sq,
        delta_norm_sq,
        direction_delta_dot,
        history_norm_sq,
        current_norm_sq,
        false,
    )
}

/// Normalized EMA with a gradient-restart criterion. When both vectors are
/// nonzero and their cosine is at or below `threshold`, history is discarded
/// and the current delta becomes the full buffer. A zero delta follows the
/// ordinary EMA decay; a zero buffer initializes exactly as above.
pub fn restarted_ema_step(
    params: &mut [f32],
    buf: &mut [f32],
    delta: &[f32],
    lr: f32,
    beta: f32,
    threshold: f32,
) -> OuterStepStats {
    debug_assert_eq!(params.len(), buf.len());
    debug_assert_eq!(buf.len(), delta.len());
    let buf_norm_sq = norm_sq(buf);
    let delta_norm_sq = norm_sq(delta);
    let restart = if buf_norm_sq > 0.0 && delta_norm_sq > 0.0 {
        let dot: f64 = buf
            .iter()
            .zip(delta)
            .map(|(b, d)| *b as f64 * *d as f64)
            .sum();
        let cosine = (dot / (buf_norm_sq * delta_norm_sq).sqrt()).clamp(-1.0, 1.0);
        cosine <= threshold as f64
    } else {
        false
    };
    if restart {
        buf.copy_from_slice(delta);
    } else {
        update_normalized_ema(buf, delta, beta, buf_norm_sq == 0.0, delta_norm_sq == 0.0);
    }
    let initialized_from_delta = restart || (buf_norm_sq == 0.0 && delta_norm_sq > 0.0);
    let (step_norm_sq, direction_norm_sq, direction_delta_dot) =
        apply_buffer(params, buf, delta, lr);
    let (history_norm_sq, current_norm_sq) =
        ema_contribution_norms(buf_norm_sq, delta_norm_sq, beta, initialized_from_delta);
    finish_outer_step_stats(
        step_norm_sq,
        direction_norm_sq,
        delta_norm_sq,
        direction_delta_dot,
        history_norm_sq,
        current_norm_sq,
        restart,
    )
}

/// HeLoCo per-tensor directional correction (arXiv 2606.00271, Alg. 1).
///
/// Applied server-side to each learner's outer delta, per tensor, before
/// merging: a stale delta can carry components that oppose the current
/// global trajectory. With û = Δ/‖Δ‖, v̂ = m/‖m‖, c = û·v̂ and
/// conf = ‖Δ‖ / (‖Δ‖ + κ‖m‖ + ε):
///
/// * c ≥ c_ok       — well aligned, pass through;
/// * c < 0          — shrink the opposing component:
///                    Δ ← Δ − β·c·‖Δ‖·v̂ with β = min(k_s·(−c)·conf, β_max);
/// * 0 ≤ c < c_ok   — rotate toward the momentum, preserving magnitude:
///                    ũ = (1−λ)û + λv̂, Δ ← ‖Δ‖·ũ/max(‖ũ‖, ε)
///                    with λ = min(k_d·(1−c)·conf, 1).
///
/// Near-zero Δ or momentum (< ε norm) skips the correction — early rounds
/// with an empty momentum buffer pass through untouched.
#[derive(Clone, Copy, Debug)]
pub struct Heloco {
    pub c_ok: f64,
    pub k_s: f64,
    pub k_d: f64,
    pub beta_max: f64,
    pub kappa: f64,
    pub eps: f64,
}

impl Default for Heloco {
    fn default() -> Self {
        // Table 3 of the paper.
        Self { c_ok: 0.2, k_s: 0.5, k_d: 1.0, beta_max: 0.5, kappa: 3.0, eps: 1e-8 }
    }
}

pub fn heloco_correct(delta: &mut [f32], momentum: &[f32], h: &Heloco) {
    debug_assert_eq!(delta.len(), momentum.len());
    let du = delta.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    let dm = momentum.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    if du < h.eps || dm < h.eps {
        return;
    }
    let dot: f64 = delta.iter().zip(momentum).map(|(d, m)| *d as f64 * *m as f64).sum();
    let c = dot / (du * dm);
    if c >= h.c_ok {
        return;
    }
    let conf = du / (du + h.kappa * dm + h.eps);
    if c < 0.0 {
        let beta = (h.k_s * (-c) * conf).min(h.beta_max);
        // Δ − β·c·‖Δ‖·v̂ (c < 0, so this adds a positive momentum component).
        let coef = (-beta * c * du / dm) as f32;
        for (d, m) in delta.iter_mut().zip(momentum) {
            *d += coef * *m;
        }
    } else {
        let lambda = (h.k_d * (1.0 - c) * conf).min(1.0);
        // ũ = (1−λ)û + λv̂, then rescale to the original magnitude.
        let (wu, wv) = ((1.0 - lambda) / du, lambda / dm);
        let mut norm_sq = 0.0f64;
        for (d, m) in delta.iter_mut().zip(momentum) {
            let t = wu * *d as f64 + wv * *m as f64;
            *d = t as f32;
            norm_sq += t * t;
        }
        let scale = (du / norm_sq.sqrt().max(h.eps)) as f32;
        for d in delta.iter_mut() {
            *d *= scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn norm(v: &[f32]) -> f64 {
        v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt()
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-6,
            "got {actual}, expected {expected}"
        );
    }

    #[test]
    fn weight_formula() {
        assert_eq!(learner_weight(100, 10), 1000.0);
        assert_eq!(learner_weight(0, 10), 0.0);
        assert_eq!(learner_weight(100, 0), 0.0);
    }

    #[test]
    fn applied_step_scaling_is_pure_and_f32_consistent() {
        let scaled = scale_applied_step(&[100_000_000.0], &[-16.0], 1.0 / 16.0).unwrap();
        assert_eq!(scaled.applied_step, vec![-1.0]);
        assert_eq!(scaled.params, vec![100_000_000.0]);
        assert_eq!(scaled.applied_step_norm, 1.0);

        let zero = scale_applied_step(&[3.0, -2.0], &[1.0, -4.0], 0.0).unwrap();
        assert_eq!(zero.applied_step, vec![0.0, -0.0]);
        assert_eq!(zero.params, vec![3.0, -2.0]);
        assert_eq!(zero.applied_step_norm, 0.0);
        assert!(scale_applied_step(&[1.0], &[1.0], -1.0).is_none());
        assert!(scale_applied_step(&[1.0], &[1.0], f64::NAN).is_none());
        assert!(scale_applied_step(&[1.0], &[1.0, 2.0], 1.0).is_none());
    }

    #[test]
    fn avg_equal_weights_is_mean_delta() {
        let anchor = [1.0f32, 1.0];
        let l0 = [0.0f32, 1.0]; // delta (1, 0)
        let l1 = [1.0f32, 0.0]; // delta (0, 1)
        let mut out = [0.0f32; 2];
        merge_avg(&anchor, &[&l0, &l1], &[1.0, 1.0], &mut out);
        assert_eq!(out, [0.5, 0.5]);
    }

    #[test]
    fn avg_respects_weights() {
        let anchor = [0.0f32];
        let l0 = [-1.0f32]; // delta 1
        let l1 = [3.0f32]; // delta -3
        let mut out = [0.0f32; 1];
        merge_avg(&anchor, &[&l0, &l1], &[3.0, 1.0], &mut out);
        assert!((out[0] - 0.0).abs() < 1e-6); // (3*1 + 1*(-3))/4
    }

    #[test]
    fn rda_preserves_norm_of_orthogonal_inputs() {
        // Two orthogonal deltas of norm 2: direct avg gives norm 2/√2 ≈ 1.41,
        // RDA must give 2.
        let anchor = [0.0f32, 0.0];
        let l0 = [-2.0f32, 0.0];
        let l1 = [0.0f32, -2.0];
        let mut out = [0.0f32; 2];
        merge_rda(&anchor, &[&l0, &l1], &[1.0, 1.0], &mut out);
        assert!((norm(&out) - 2.0).abs() < 1e-5, "norm was {}", norm(&out));
        // Direction is the diagonal.
        assert!((out[0] - out[1]).abs() < 1e-6 && out[0] > 0.0);
    }

    #[test]
    fn rda_single_learner_is_identity_delta() {
        let anchor = [1.0f32, 2.0, 3.0];
        let l0 = [0.5f32, 2.5, 3.0];
        let mut out = [0.0f32; 3];
        merge_rda(&anchor, &[&l0], &[7.0], &mut out);
        for (o, e) in out.iter().zip([0.5f32, -0.5, 0.0]) {
            assert!((o - e).abs() < 1e-6);
        }
    }

    #[test]
    fn rda_zero_deltas_give_zero() {
        let anchor = [1.0f32, 2.0];
        let l0 = anchor;
        let mut out = [9.0f32; 2];
        merge_rda(&anchor, &[&l0[..]], &[1.0], &mut out);
        assert_eq!(out, [0.0, 0.0]);
    }

    #[test]
    fn rda_cancelling_directions_fall_back_to_avg() {
        let anchor = [0.0f32];
        let l0 = [-1.0f32]; // delta +1
        let l1 = [1.0f32]; // delta -1: unit dirs cancel exactly
        let mut out = [0.0f32; 1];
        merge_rda(&anchor, &[&l0, &l1], &[1.0, 1.0], &mut out);
        assert!(out[0].abs() < 1e-6);
    }

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
        dot / (norm(a) * norm(b))
    }

    /// A·Aᵀ of a rows×cols row-major matrix, in f64.
    fn gram(a: &[f32], rows: usize, cols: usize) -> Vec<f64> {
        let mut g = vec![0.0f64; rows * rows];
        for i in 0..rows {
            for j in 0..rows {
                g[i * rows + j] = (0..cols)
                    .map(|c| a[i * cols + c] as f64 * a[j * cols + c] as f64)
                    .sum();
            }
        }
        g
    }

    #[test]
    fn iso_flattens_diagonal_spectrum_to_mean_singular_value() {
        // Alg. 2 step 2 on an axis-aligned 2x2 delta diag(3, 1): U = V = I,
        // sigma = {3, 1}, sigma_bar = 2 -> Delta_iso = diag(2, 2). Learner
        // values are anchor - delta so the merged delta is exactly diag(3, 1).
        let anchor = [0.0f32; 4];
        let learner = [-3.0f32, 0.0, 0.0, -1.0];
        let mut out = [0.0f32; 4];
        merge_iso(&anchor, &[&learner], &[1.0], 2, 2, &mut out);
        for (o, e) in out.iter().zip([2.0f32, 0.0, 0.0, 2.0]) {
            assert!((o - e).abs() < 1e-6, "got {out:?}");
        }
    }

    #[test]
    fn iso_matches_sigma_bar_u_vt_on_rectangular_matrices() {
        // 2x3 (Gram on rows): Delta = [[1,0,0],[0,2,0]], sigma = {2, 1},
        // sigma_bar = 1.5 -> 1.5*U*V^T = [[1.5,0,0],[0,1.5,0]].
        let anchor = [0.0f32; 6];
        let learner = [-1.0f32, 0.0, 0.0, 0.0, -2.0, 0.0];
        let mut out = [0.0f32; 6];
        merge_iso(&anchor, &[&learner], &[2.5], 2, 3, &mut out);
        for (o, e) in out.iter().zip([1.5f32, 0.0, 0.0, 0.0, 1.5, 0.0]) {
            assert!((o - e).abs() < 1e-6, "got {out:?}");
        }
        // 3x2 (Gram on cols, the transposed branch): same spectrum.
        let learner_t = [-1.0f32, 0.0, 0.0, -2.0, 0.0, 0.0];
        let mut out_t = [0.0f32; 6];
        merge_iso(&anchor, &[&learner_t], &[1.0], 3, 2, &mut out_t);
        for (o, e) in out_t.iter().zip([1.5f32, 0.0, 0.0, 1.5, 0.0, 0.0]) {
            assert!((o - e).abs() < 1e-6, "got {out_t:?}");
        }
    }

    #[test]
    fn iso_rotated_svd_is_reconstructed_exactly() {
        // Delta = R(30 deg) * diag(3, 1): U = R, V = I, sigma_bar = 2, so
        // Delta_iso = 2*R = [[sqrt(3), -1], [1, sqrt(3)]].
        let h = 3.0f64.sqrt() / 2.0;
        let delta = [3.0 * h, -0.5, 3.0 * 0.5, h];
        let anchor = [0.0f32; 4];
        let learner: Vec<f32> = delta.iter().map(|d| -*d as f32).collect();
        let mut out = [0.0f32; 4];
        merge_iso(&anchor, &[&learner[..]], &[1.0], 2, 2, &mut out);
        let expected = [3.0f64.sqrt(), -1.0, 1.0, 3.0f64.sqrt()];
        for (o, e) in out.iter().zip(expected) {
            assert!((*o as f64 - e).abs() < 1e-6, "got {out:?}");
        }
    }

    #[test]
    fn iso_output_spectrum_is_isotropic_and_aligned() {
        // Full-rank deterministic 3x4 input: the output must satisfy
        // M' M'^T = sigma_bar^2 * I (all singular values equal) and
        // <M', M>_F = ||M'||_F^2 (M' = sigma_bar*U*V^T aligns with M's
        // singular basis). Both checks are basis-free, so they hold no
        // matter how the SVD is computed internally.
        let anchor = [0.0f32; 12];
        let delta = [
            2.0f32, -1.0, 0.5, 3.0, //
            0.25, 4.0, -2.0, 1.0, //
            -1.5, 0.75, 3.5, -0.5,
        ];
        let learner: Vec<f32> = delta.iter().map(|d| -d).collect();
        let mut out = [0.0f32; 12];
        merge_iso(&anchor, &[&learner[..]], &[1.0], 3, 4, &mut out);
        let g = gram(&out, 3, 4);
        let sigma_bar_sq = (g[0] + g[4] + g[8]) / 3.0;
        assert!(sigma_bar_sq > 0.0);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { sigma_bar_sq } else { 0.0 };
                assert!(
                    (g[i * 3 + j] - expected).abs() < 1e-4 * sigma_bar_sq,
                    "gram {g:?}"
                );
            }
        }
        let inner: f64 = out
            .iter()
            .zip(delta)
            .map(|(o, d)| *o as f64 * d as f64)
            .sum();
        let out_norm_sq: f64 = out.iter().map(|o| (*o as f64).powi(2)).sum();
        assert!((inner - out_norm_sq).abs() < 1e-4 * out_norm_sq);
        // Idempotence: an already-isotropic matrix is a fixed point.
        let mut again = out;
        iso_flatten_spectrum(&mut again, 3, 4);
        for (a, o) in again.iter().zip(out) {
            assert!((a - o).abs() < 1e-5 * sigma_bar_sq.sqrt() as f32);
        }
    }

    #[test]
    fn iso_weighted_average_feeds_the_transform() {
        // Weighted direct average first (house weighting, a documented
        // deviation from the paper's uniform 1/R mean): deltas diag(4, 0)
        // and diag(0, 4) at weights (3, 1) average to diag(3, 1), which
        // flattens to diag(2, 2) as in the diagonal test.
        let anchor = [0.0f32; 4];
        let l0 = [-4.0f32, 0.0, 0.0, 0.0];
        let l1 = [0.0f32, 0.0, 0.0, -4.0];
        let mut out = [0.0f32; 4];
        merge_iso(&anchor, &[&l0, &l1], &[3.0, 1.0], 2, 2, &mut out);
        for (o, e) in out.iter().zip([2.0f32, 0.0, 0.0, 2.0]) {
            assert!((o - e).abs() < 1e-6, "got {out:?}");
        }
    }

    #[test]
    fn iso_rank_deficient_null_directions_are_dropped_not_inflated() {
        // Rank-1 2x2 delta diag(2, 0): sigma = {2, 0}, sigma_bar = 1 (zeros
        // count toward the mean, per the paper). The null direction has an
        // arbitrary singular vector, so it is dropped rather than filled:
        // Delta_iso = diag(1, 0). Documented deviation.
        let anchor = [0.0f32; 4];
        let learner = [-2.0f32, 0.0, 0.0, 0.0];
        let mut out = [0.0f32; 4];
        merge_iso(&anchor, &[&learner], &[1.0], 2, 2, &mut out);
        for (o, e) in out.iter().zip([1.0f32, 0.0, 0.0, 0.0]) {
            assert!((o - e).abs() < 1e-6, "got {out:?}");
        }
    }

    #[test]
    fn iso_vector_and_zero_deltas_are_stable() {
        // A 1xN delta has a single singular value, so sigma_bar equals it
        // and the transform is the identity (the paper leaves non-matrix
        // parameters untouched; a degenerate matrix shape agrees).
        let anchor = [0.0f32, 0.0];
        let learner = [-3.0f32, -4.0];
        let mut out = [0.0f32; 2];
        merge_iso(&anchor, &[&learner], &[1.0], 1, 2, &mut out);
        assert!((out[0] - 3.0).abs() < 1e-6 && (out[1] - 4.0).abs() < 1e-6);
        // Zero delta stays exactly zero.
        let zero_learner = [0.0f32, 0.0];
        merge_iso(&anchor, &[&zero_learner], &[1.0], 2, 1, &mut out);
        assert_eq!(out, [0.0, 0.0]);
        // Shape/product mismatch keeps the plain weighted average.
        merge_iso(&anchor, &[&learner], &[1.0], 3, 5, &mut out);
        assert_eq!(out, [3.0, 4.0]);
    }


    #[test]
    fn heloco_aligned_passes_through() {
        let h = Heloco::default();
        let mut d = [1.0f32, 0.1];
        let orig = d;
        heloco_correct(&mut d, &[1.0, 0.0], &h); // cos ≈ 0.995 ≥ c_ok
        assert_eq!(d, orig);
    }

    #[test]
    fn rho_adaptive_first_step_is_plain_sgd_and_stores_delta() {
        let mut params = vec![1.0, 1.0];
        let mut buf = vec![0.0, 0.0];
        let mut rho_ema = RHO_ADAPTIVE_INITIAL_RHO_EMA;
        let stats = rho_adaptive_step(&mut params, &mut buf, &[0.5, -0.5], 0.1, &mut rho_ema);
        // zero buffer -> no measurement -> rho_ema stays at rho_ref -> gain
        // exactly 1, i.e. the tuned-baseline (plain SGD) step
        assert!((params[0] - (1.0 - 0.05)).abs() < 1e-7);
        assert!((params[1] - (1.0 + 0.05)).abs() < 1e-7);
        assert_eq!(buf, vec![0.5, -0.5]);
        assert_eq!(rho_ema, RHO_ADAPTIVE_INITIAL_RHO_EMA);
        assert_eq!(stats.history_current_norm_ratio, Some(0.0));
    }

    #[test]
    fn rho_adaptive_persistent_direction_amplifies_per_corrected_law() {
        let mut params = vec![0.0, 0.0];
        let mut buf = vec![1.0, 0.0];
        let mut rho_ema = RHO_ADAPTIVE_INITIAL_RHO_EMA;
        // identical direction: rho = 1 -> rho_ema = 0.5*0.5 + 0.5*1 = 0.75
        // g(0.75) = a(0.75)/a(0.5) = 1.8/(5/3) = 1.08
        let stats = rho_adaptive_step(&mut params, &mut buf, &[1.0, 0.0], 0.1, &mut rho_ema);
        assert!((rho_ema - 0.75).abs() < 1e-7);
        assert!((params[0] + 0.108).abs() < 1e-6);
        // buffer stores the applied (scaled) direction
        assert!((buf[0] - 1.08).abs() < 1e-6);
        assert_eq!(stats.history_current_norm_ratio, Some(1.0));
    }

    #[test]
    fn rho_adaptive_anticorrelated_direction_dampens() {
        let mut params = vec![0.0];
        let mut buf = vec![1.0];
        let mut rho_ema = RHO_ADAPTIVE_INITIAL_RHO_EMA;
        // rho = -1 -> rho_ema = 0.5*0.5 - 0.5 = -0.25
        // g(-0.25) = a(-0.25)/a(0.5) = (13/9)/(5/3) = 13/15 < 1
        let stats = rho_adaptive_step(&mut params, &mut buf, &[-1.0], 0.1, &mut rho_ema);
        let gain = 13.0 / 15.0;
        assert!((rho_ema + 0.25).abs() < 1e-7);
        assert!((params[0] - 0.1 * gain).abs() < 1e-6);
        assert_eq!(stats.history_current_norm_ratio, Some(-1.0));
        // buffer stores the applied (scaled) direction
        assert!((buf[0] + gain as f32).abs() < 1e-6);
    }

    #[test]
    fn rho_adaptive_gain_matches_corrected_law_and_is_bounded() {
        // Reference point reproduces the tuned baseline exactly.
        assert_eq!(rho_adaptive_gain(RHO_ADAPTIVE_RHO_REF), 1.0);
        // Closed forms: a(1) = 2, a(-1) = 4/3, a(0.5) = 5/3.
        assert!((rho_adaptive_gain(1.0) - 1.2).abs() < 1e-12);
        assert!((rho_adaptive_gain(-1.0) - 0.8).abs() < 1e-12);
        // Monotone increasing in persistence over the valid range.
        assert!(rho_adaptive_gain(1.0) > rho_adaptive_gain(0.5));
        assert!(rho_adaptive_gain(0.5) > rho_adaptive_gain(-1.0));
        // Hard bounds hold even for degenerate inputs (a(2) diverges).
        assert_eq!(rho_adaptive_gain(2.0), RHO_ADAPTIVE_GAIN_MAX);
        assert!(rho_adaptive_gain(100.0) >= RHO_ADAPTIVE_GAIN_MIN);
    }

    #[test]
    fn rho_adaptive_ema_converges_and_gain_saturates() {
        let mut params = vec![0.0];
        let mut buf = vec![1.0];
        let mut rho_ema = RHO_ADAPTIVE_INITIAL_RHO_EMA;
        for _ in 0..20 {
            let stats = rho_adaptive_step(&mut params, &mut buf, &[1.0], 0.1, &mut rho_ema);
            assert_eq!(stats.history_current_norm_ratio, Some(1.0));
        }
        // Persistent direction: rho_ema -> 1 and the gain saturates at
        // a(1)/a(0.5) = 1.2, well inside the [0.5, 2.0] hard bounds.
        assert!((rho_ema - 1.0).abs() < 1e-4);
        assert!((buf[0] - 1.2).abs() < 1e-4);
    }

    #[test]
    fn heloco_zero_momentum_is_noop() {
        let h = Heloco::default();
        let mut d = [-3.0f32, 4.0];
        let orig = d;
        heloco_correct(&mut d, &[0.0, 0.0], &h);
        assert_eq!(d, orig);
    }

    #[test]
    fn heloco_anti_aligned_reduces_opposition() {
        let h = Heloco::default();
        let m = [1.0f32, 0.0];
        let mut d = [-2.0f32, 0.5];
        let before = cosine(&d, &m);
        heloco_correct(&mut d, &m, &h);
        let after = cosine(&d, &m);
        assert!(after > before, "cosine {before} -> {after} did not improve");
        // Shrinkage is bounded: the delta cannot flip past the momentum.
        assert!(d[0] < 0.0 || d[0].abs() < 2.0);
    }

    #[test]
    fn heloco_weakly_aligned_preserves_magnitude() {
        let h = Heloco::default();
        let m = [1.0f32, 0.0];
        let mut d = [0.1f32, 1.0]; // cos ≈ 0.0995, in [0, c_ok)
        let mag = norm(&d);
        let before = cosine(&d, &m);
        heloco_correct(&mut d, &m, &h);
        assert!((norm(&d) - mag).abs() < 1e-5, "magnitude changed: {mag} -> {}", norm(&d));
        assert!(cosine(&d, &m) > before);
    }

    #[test]
    fn heloco_confidence_damps_correction_under_large_momentum() {
        let h = Heloco::default();
        let mut small_m = [-1.0f32, 0.2];
        let mut large_m = small_m;
        heloco_correct(&mut small_m, &[0.1, 0.0], &h);
        heloco_correct(&mut large_m, &[100.0, 0.0], &h);
        // Same directions, but huge momentum norm → low confidence → weaker
        // correction (closer to the original delta).
        let orig = [-1.0f32, 0.2];
        let moved_small: f64 = small_m.iter().zip(&orig).map(|(a, b)| (a - b).abs() as f64).sum();
        let moved_large: f64 = large_m.iter().zip(&orig).map(|(a, b)| (a - b).abs() as f64).sum();
        assert!(moved_large < moved_small);
    }

    #[test]
    fn nesterov_matches_reference() {
        // One step from zero state: buf = Δ; θ -= lr(Δ + μΔ) = lr(1+μ)Δ.
        let mut p = [1.0f32];
        let mut buf = [0.0f32];
        let stats = nesterov_step(&mut p, &mut buf, &[0.5], 0.7, 0.9);
        assert!((p[0] - (1.0 - 0.7 * (0.5 + 0.9 * 0.5))).abs() < 1e-6);
        assert!((buf[0] - 0.5).abs() < 1e-6);
        assert_close(stats.applied_step_norm, (0.7f32 * 0.95) as f64);
        assert_close(stats.direction_delta_cosine.unwrap(), 1.0);
        assert_eq!(stats.history_current_norm_ratio, Some(0.0));
        assert!(!stats.restarted);
    }

    #[test]
    fn nesterov_stats_separate_history_and_current_contributions() {
        let mut p = [0.0f32];
        let mut buf = [2.0f32];
        let stats = nesterov_step(&mut p, &mut buf, &[1.0], 0.25, 0.5);
        assert_eq!(buf, [2.0]);
        assert_eq!(p, [-0.5]);
        assert_close(stats.applied_step_norm, 0.5);
        assert_close(stats.direction_delta_cosine.unwrap(), 1.0);
        // direction = (1 + mu) * delta + mu^2 * previous_buffer.
        assert_close(stats.history_current_norm_ratio.unwrap(), 0.5 / 1.5);
        assert!(!stats.restarted);
    }

    #[test]
    fn nesterov_three_step_hand_computed_sequence() {
        // Deterministic 3-step audit at mu = 0.9, lr = 0.1, theta_0 = 0,
        // b_0 = 0 (production initialization). Recursion under test:
        //   b_t = mu*b_{t-1} + delta_t
        //   d_t = delta_t + mu*b_t = (1+mu)*delta_t + mu^2*b_{t-1}
        //   theta_t = theta_{t-1} - lr*d_t
        //
        // t=1, delta_1 = [1, 2]:
        //   b_1 = [1, 2]
        //   d_1 = (1+mu)*delta_1 = [1.9, 3.8]      (zero history)
        //   step_1 = [0.19, 0.38], theta_1 = [-0.19, -0.38]
        //   |step_1| = (1+mu)*lr*|delta_1| = 0.19*sqrt(5)
        // t=2, delta_2 = [0.5, -1]:
        //   b_2 = 0.9*[1, 2] + [0.5, -1] = [1.4, 0.8]
        //   d_2 = [0.5, -1] + 0.9*[1.4, 0.8] = [1.76, -0.28]
        //   cross-check via the two-term form:
        //     (1+mu)*delta_2 + mu^2*b_1 = [0.95+0.81, -1.9+1.62] = [1.76, -0.28]
        //   step_2 = [0.176, -0.028], theta_2 = [-0.366, -0.352]
        // t=3, delta_3 = [2, 0]:
        //   b_3 = 0.9*[1.4, 0.8] + [2, 0] = [3.26, 0.72]
        //   d_3 = [2, 0] + 0.9*[3.26, 0.72] = [4.934, 0.648]
        //   step_3 = [0.4934, 0.0648], theta_3 = [-0.8594, -0.4168]
        let mu = 0.9f32;
        let lr = 0.1f32;
        let mut p = [0.0f32, 0.0];
        let mut buf = [0.0f32, 0.0];
        let tol = 1e-5f64;

        let s1 = nesterov_step(&mut p, &mut buf, &[1.0, 2.0], lr, mu);
        assert!((buf[0] - 1.0).abs() < tol as f32 && (buf[1] - 2.0).abs() < tol as f32);
        assert!((p[0] as f64 + 0.19).abs() < tol && (p[1] as f64 + 0.38).abs() < tol);
        // Explicit first-step factor: |step_1| == (1+mu)*lr*|delta_1|.
        let delta1_norm = 5.0f64.sqrt();
        assert!((s1.applied_step_norm - 1.9 * 0.1 * delta1_norm).abs() < tol);
        assert_eq!(s1.history_current_norm_ratio, Some(0.0));

        let s2 = nesterov_step(&mut p, &mut buf, &[0.5, -1.0], lr, mu);
        assert!((buf[0] as f64 - 1.4).abs() < tol && (buf[1] as f64 - 0.8).abs() < tol);
        assert!((p[0] as f64 + 0.366).abs() < tol && (p[1] as f64 + 0.352).abs() < tol);
        // Exact identity check at t=2 (documented in OPTIMIZER_SEMANTICS.md):
        //   c_2 = <b_1, delta_2>/|delta_2|^2 = (0.5 - 2)/1.25 = -1.2
        //   A_2 = 1 + mu + mu^2*c_2 = 1.9 - 0.972 = 0.928
        //   b_1 - c_2*delta_2 = [1.6, 0.8], r_2 = sqrt(3.2/1.25) = 1.6
        //   d_2 = A_2*delta_2 + mu^2*[1.6, 0.8]
        //       = [0.464, -0.928] + [1.296, 0.648] = [1.76, -0.28]  (matches)
        let d2_norm = (1.76f64 * 1.76 + 0.28 * 0.28).sqrt();
        assert!((s2.applied_step_norm - 0.1 * d2_norm).abs() < tol);
        // history/current ratio = mu^2*|b_1| / ((1+mu)*|delta_2|)
        let expected_ratio = 0.81 * 5.0f64.sqrt() / (1.9 * 1.25f64.sqrt());
        assert!((s2.history_current_norm_ratio.unwrap() - expected_ratio).abs() < tol);

        let s3 = nesterov_step(&mut p, &mut buf, &[2.0, 0.0], lr, mu);
        assert!((buf[0] as f64 - 3.26).abs() < tol && (buf[1] as f64 - 0.72).abs() < tol);
        assert!((p[0] as f64 + 0.8594).abs() < tol && (p[1] as f64 + 0.4168).abs() < tol);
        let d3_norm = (4.934f64 * 4.934 + 0.648 * 0.648).sqrt();
        assert!((s3.applied_step_norm - 0.1 * d3_norm).abs() < tol);
    }

    #[test]
    fn rho_adaptive_three_step_hand_computed_sequence() {
        // Deterministic 3-step audit of the implemented rho-adaptive
        // semantics (v2; v1's mu_eff heuristic was retired in EXP2_26) at
        // lr = 0.1, theta_0 = 0, b_0 = 0, rho_ema_0 = rho_ref = 0.5.
        // Per commit: rho_t = cos(delta_t, b_{t-1}) (unmeasured when either
        // is zero), rho_ema <- 0.5*rho_ema + 0.5*rho_t (only when measured),
        // s_t = clamp(a(rho_ema)/a(0.5), 0.5, 2) with a(r) = 1 + 0.5/(1-0.5r),
        // theta -= lr*s_t*delta_t, b <- s_t*delta_t.
        //
        // t=1, delta_1 = [1, 2]: b_0 = 0 -> unmeasured, rho_ema stays 0.5,
        //   s_1 = 1. step = [0.1, 0.2], theta_1 = [-0.1, -0.2], b_1 = [1, 2].
        // t=2, delta_2 = [2, 4] (parallel, rho = 1):
        //   rho_ema = 0.5*0.5 + 0.5*1 = 0.75
        //   s_2 = a(0.75)/a(0.5) = 1.8/(5/3) = 1.08
        //   b_2 = 1.08*[2, 4] = [2.16, 4.32], step = [0.216, 0.432]
        //   theta_2 = [-0.316, -0.632]
        // t=3, delta_3 = [-1, -2] (anti-parallel, rho = -1):
        //   rho_ema = 0.5*0.75 - 0.5 = -0.125
        //   s_3 = a(-0.125)/a(0.5) = (1 + 8/17)/(5/3) = 15/17
        //   b_3 = -(15/17)*[1, 2], step = [-1.5/17, -3/17]
        //   theta_3 = [-0.316 + 1.5/17, -0.632 + 3/17]
        let lr = 0.1f32;
        let mut p = [0.0f32, 0.0];
        let mut buf = [0.0f32, 0.0];
        let mut rho_ema = RHO_ADAPTIVE_INITIAL_RHO_EMA;
        let tol = 1e-6f64;

        let s1 = rho_adaptive_step(&mut p, &mut buf, &[1.0, 2.0], lr, &mut rho_ema);
        assert_eq!(rho_ema, 0.5);
        assert!((p[0] as f64 + 0.1).abs() < tol && (p[1] as f64 + 0.2).abs() < tol);
        assert_eq!(buf, [1.0, 2.0]);
        assert_eq!(s1.history_current_norm_ratio, Some(0.0));
        assert!((s1.applied_step_norm - 0.1 * 5.0f64.sqrt()).abs() < tol);

        let s2 = rho_adaptive_step(&mut p, &mut buf, &[2.0, 4.0], lr, &mut rho_ema);
        assert!((rho_ema as f64 - 0.75).abs() < tol);
        assert!((buf[0] as f64 - 2.16).abs() < tol && (buf[1] as f64 - 4.32).abs() < tol);
        assert!((p[0] as f64 + 0.316).abs() < tol && (p[1] as f64 + 0.632).abs() < tol);
        assert!((s2.history_current_norm_ratio.unwrap() - 1.0).abs() < tol);
        assert!((s2.applied_step_norm - 0.1 * 1.08 * 20.0f64.sqrt()).abs() < 1e-5);

        let s3 = rho_adaptive_step(&mut p, &mut buf, &[-1.0, -2.0], lr, &mut rho_ema);
        let gain3 = 15.0 / 17.0;
        assert!((rho_ema as f64 + 0.125).abs() < tol);
        assert!(
            (buf[0] as f64 + gain3).abs() < tol && (buf[1] as f64 + 2.0 * gain3).abs() < tol
        );
        assert!(
            (p[0] as f64 + (0.316 - 0.1 * gain3)).abs() < tol
                && (p[1] as f64 + (0.632 - 0.2 * gain3)).abs() < tol
        );
        assert!((s3.history_current_norm_ratio.unwrap() + 1.0).abs() < tol);
        assert!((s3.applied_step_norm - 0.1 * gain3 * 5.0f64.sqrt()).abs() < tol);
    }

    #[test]
    fn capped_nesterov_zero_buffer_first_step_runs_at_mu_max() {
        // b_0 = 0 gives no measurable geometry: dot = 0 so c_1 = 0 and
        // r_1 = 0. Then mu_par = mu_max (c_plus = 0), mu_perp =
        // sqrt(tau_perp/eps) = 1e6 (inactive), cap = mu_max = 0.9,
        // A(0.9) = 1.9 > 0 (guard idle). EMA path from mu_prev = 0.9:
        // released = 0.9*0.9 + 0.1*0.9 = 0.9, mu_1 = min(0.9, 0.9) = 0.9.
        // The commit is exactly plain Nesterov at mu_max from zero state:
        // b_1 = delta, step = lr*(1 + 0.9)*delta = 0.19*delta.
        let mut p = [1.0f32, -1.0];
        let mut buf = [0.0f32, 0.0];
        let mut mu = CAPPED_NESTEROV_INITIAL_MU;
        let stats = capped_nesterov_step(&mut p, &mut buf, &[0.5, 2.0], 0.1, &mut mu);
        assert!((mu as f64 - 0.9).abs() < 1e-7);
        assert_eq!(buf, [0.5, 2.0]);
        assert!((p[0] as f64 - (1.0 - 0.19 * 0.5)).abs() < 1e-6);
        assert!((p[1] as f64 - (-1.0 - 0.19 * 2.0)).abs() < 1e-6);
        // Stats keep the plain-Nesterov conventions.
        let delta_norm = (0.5f64 * 0.5 + 4.0).sqrt();
        assert!((stats.applied_step_norm - 0.19 * delta_norm).abs() < 1e-6);
        assert_close(stats.direction_delta_cosine.unwrap(), 1.0);
        assert_eq!(stats.history_current_norm_ratio, Some(0.0));
        assert!(!stats.restarted);
    }

    #[test]
    fn capped_nesterov_three_step_hand_computed_sequence() {
        // Deterministic 3-step audit at lr = 0.1, theta_0 = 0, b_0 = 0,
        // mu_prev = 0.9 (production initialization); mu_max = 0.9,
        // tau_perp = 1, release beta = 0.9.
        //
        // t=1, delta_1 = [1, 0]: zero buffer -> c_1 = 0, r_1 = 0, all caps
        //   inactive, cap = 0.9; released = 0.9*0.9 + 0.1*0.9 = 0.9;
        //   mu_1 = 0.9. Nesterov: b_1 = [1, 0], d_1 = 1.9*[1, 0],
        //   step_1 = [0.19, 0], theta_1 = [-0.19, 0].
        // t=2, delta_2 = [1, 0] (fully aligned history):
        //   c_2 = <[1,0],[1,0]>/1 = 1, r_2 = 0. Aligned cap binds:
        //   mu_par solves mu + mu^2 = 0.9, mu_par = (sqrt(1 + 4*0.9) - 1)/2
        //         = (sqrt(4.6) - 1)/2 = 0.57238053...
        //   (check: 0.57238053 + 0.57238053^2 = 0.9). cap = mu_par;
        //   released = 0.9*0.9 + 0.1*0.57238053 = 0.86723805 > cap, so the
        //   cap binds instantly (one-sided): mu_2 = 0.57238053.
        //   By construction A_2 = 1 + mu_2 + mu_2^2*c_2 = 1 + 0.9 = 1.9, so
        //   d_2 = A_2*delta_2 = [1.9, 0]: the aligned cap holds the applied
        //   step at exactly the (1+mu_max) design gain.
        //   b_2 = mu_2*[1,0] + [1,0] = [1.57238053, 0],
        //   step_2 = [0.19, 0], theta_2 = [-0.38, 0].
        // t=3, delta_3 = [0, 1] (orthogonal history):
        //   c_3 = 0, r_3 = |b_2|/|delta_3| = 1.57238053. Transverse cap:
        //   mu_perp = sqrt(1/1.57238053) = 0.79748276...; cap = 0.79748276.
        //   released = 0.9*0.57238053 + 0.1*0.79748276 = 0.59489075 < cap,
        //   so the release EMA binds (smooth recovery): mu_3 = 0.59489075.
        //   b_3 = mu_3*[1.57238053, 0] + [0, 1] = [0.93539497, 1],
        //   d_3 = [0, 1] + mu_3*b_3 = [0.55645437, 1.59489075],
        //   step_3 = [0.05564544, 0.15948908],
        //   theta_3 = [-0.43564544, -0.15948908].
        let lr = 0.1f32;
        let mut p = [0.0f32, 0.0];
        let mut buf = [0.0f32, 0.0];
        let mut mu = CAPPED_NESTEROV_INITIAL_MU;
        let tol = 1e-5f64;

        let s1 = capped_nesterov_step(&mut p, &mut buf, &[1.0, 0.0], lr, &mut mu);
        assert!((mu as f64 - 0.9).abs() < 1e-7);
        assert_eq!(buf, [1.0, 0.0]);
        assert!((p[0] as f64 + 0.19).abs() < tol && p[1] == 0.0);
        assert!((s1.applied_step_norm - 0.19).abs() < tol);
        assert_eq!(s1.history_current_norm_ratio, Some(0.0));

        let s2 = capped_nesterov_step(&mut p, &mut buf, &[1.0, 0.0], lr, &mut mu);
        let mu2 = ((1.0f64 + 4.0 * 0.9).sqrt() - 1.0) / 2.0;
        assert!((mu as f64 - mu2).abs() < 1e-7);
        // The one-sided min bound the EMA path from above at t=2.
        assert!(0.9 * 0.9 + 0.1 * mu2 > mu2);
        assert!((buf[0] as f64 - (1.0 + mu2)).abs() < tol && buf[1] == 0.0);
        assert!((p[0] as f64 + 0.38).abs() < tol && p[1] == 0.0);
        assert!((s2.applied_step_norm - 0.19).abs() < tol);
        // history/current ratio keeps the Nesterov convention:
        // mu^2*|b_1| / ((1+mu)*|delta_2|).
        let expected_ratio = mu2 * mu2 / (1.0 + mu2);
        assert!((s2.history_current_norm_ratio.unwrap() - expected_ratio).abs() < tol);

        let s3 = capped_nesterov_step(&mut p, &mut buf, &[0.0, 1.0], lr, &mut mu);
        let cap3 = (1.0f64 / (1.0 + mu2)).sqrt();
        let mu3 = 0.9 * mu2 + (1.0 - 0.9) * cap3;
        // The release EMA binds from below at t=3.
        assert!(mu3 < cap3);
        assert!((mu as f64 - mu3).abs() < 1e-7);
        let b3 = [mu3 * (1.0 + mu2), 1.0];
        assert!((buf[0] as f64 - b3[0]).abs() < tol && (buf[1] as f64 - 1.0).abs() < tol);
        let d3 = [mu3 * b3[0], 1.0 + mu3 * b3[1]];
        assert!((p[0] as f64 + (0.38 + 0.1 * d3[0])).abs() < tol);
        assert!((p[1] as f64 + 0.1 * d3[1]).abs() < tol);
        let d3_norm = (d3[0] * d3[0] + d3[1] * d3[1]).sqrt();
        assert!((s3.applied_step_norm - 0.1 * d3_norm).abs() < tol);
    }

    #[test]
    fn capped_nesterov_high_transverse_residual_binds_mu_perp() {
        // b = [0, 10], delta = [1, 0]: c = 0 (orthogonal), r = |b|/|delta|
        // = 10. mu_par = mu_max = 0.9; mu_perp = sqrt(tau_perp/10)
        // = sqrt(0.1) = 0.31622777; cap = 0.31622777, far below the EMA path
        // (0.9*0.9 + 0.1*cap = 0.84162278), so the transverse cap binds:
        // mu = sqrt(0.1). The step's delta-orthogonal component is then
        // mu^2*b = 0.1*[0, 10] with norm exactly tau_perp*|delta| = 1.
        let mut p = [0.0f32, 0.0];
        let mut buf = [0.0f32, 10.0];
        let mut mu = CAPPED_NESTEROV_INITIAL_MU;
        let stats = capped_nesterov_step(&mut p, &mut buf, &[1.0, 0.0], 1.0, &mut mu);
        let expected_mu = 0.1f64.sqrt();
        assert!((mu as f64 - expected_mu).abs() < 1e-7);
        // b_new = mu*[0, 10] + [1, 0]; d = delta + mu*b_new
        //       = [1 + mu, 10*mu^2] = [1.31622777, 1.0].
        assert!((buf[0] as f64 - 1.0).abs() < 1e-6);
        assert!((buf[1] as f64 - 10.0 * expected_mu).abs() < 1e-5);
        assert!((p[0] as f64 + (1.0 + expected_mu)).abs() < 1e-6);
        assert!((p[1] as f64 + 1.0).abs() < 1e-5);
        // history/current ratio, Nesterov convention:
        // mu^2*|b_prev| / ((1+mu)*|delta|) = 1 / (1 + mu).
        let expected_ratio = 1.0 / (1.0 + expected_mu);
        assert!((stats.history_current_norm_ratio.unwrap() - expected_ratio).abs() < 1e-6);
    }

    #[test]
    fn capped_nesterov_negative_c_guard_and_smooth_release() {
        // Mildly negative c keeps full momentum: b = [-0.5, 0],
        // delta = [1, 0]: c = -0.5, c_plus = 0 -> mu_par = 0.9; b = c*delta
        // exactly, so r = 0 -> mu_perp inactive; cap = 0.9;
        // A(0.9) = 1 + 0.9 + 0.81*(-0.5) = 1.495 > 0 -> no guard; mu = 0.9.
        let mut p = [0.0f32, 0.0];
        let mut buf = [-0.5f32, 0.0];
        let mut mu = CAPPED_NESTEROV_INITIAL_MU;
        capped_nesterov_step(&mut p, &mut buf, &[1.0, 0.0], 0.1, &mut mu);
        assert!((mu as f64 - 0.9).abs() < 1e-7);

        // Strongly negative c flips the aligned gain sign: b = [-10, 0],
        // delta = [1, 0]: c = -10, cap before the guard = 0.9,
        // A(0.9) = 1.9 + 0.81*(-10) = -6.2 < 0 -> guard zeroes the cap;
        // mu = min(0, 0.9*0.9 + 0.1*0) = 0. The commit degrades to plain
        // SGD and the buffer restarts from the delta (b = 0*b + delta).
        let mut p = [0.0f32, 0.0];
        let mut buf = [-10.0f32, 0.0];
        let mut mu = CAPPED_NESTEROV_INITIAL_MU;
        let stats = capped_nesterov_step(&mut p, &mut buf, &[1.0, 0.0], 0.1, &mut mu);
        assert_eq!(mu, 0.0);
        assert_eq!(buf, [1.0, 0.0]);
        assert!((p[0] as f64 + 0.1).abs() < 1e-7 && p[1] == 0.0);
        assert_close(stats.applied_step_norm, 0.1);
        assert_eq!(stats.history_current_norm_ratio, Some(0.0));

        // Release is smooth, not a jump back to mu_max: next commit with
        // delta = [0, 1] sees c = 0, r = |[1,0]|/1 = 1 -> mu_perp = 1,
        // cap = 0.9; mu = min(0.9, 0.9*0 + 0.1*0.9) = 0.09.
        let _ = capped_nesterov_step(&mut p, &mut buf, &[0.0, 1.0], 0.1, &mut mu);
        assert!((mu as f64 - 0.09).abs() < 1e-7);
    }

    #[test]
    fn capped_nesterov_step_is_bit_identical_to_materialized_step() {
        // Same requirement the state layer enforces on every preview commit:
        // materialize_applied_step with the updated buffer and the effective
        // momentum written by the step must reproduce the applied step
        // bit-for-bit on the f32 lattice.
        let base = [0.25f32, -1.5, 3.0];
        let mut p = base;
        let mut buf = [0.5f32, 1.0, -2.0];
        let delta = [1.0f32, -0.5, 0.25];
        let mut mu = CAPPED_NESTEROV_INITIAL_MU;
        capped_nesterov_step(&mut p, &mut buf, &delta, 0.3, &mut mu);
        let step =
            materialize_applied_step(OuterOptimizer::CappedNesterov, &buf, &delta, 0.3, mu, 1.0);
        for ((b, s), after) in base.iter().zip(&step).zip(&p) {
            assert_eq!((b - s).to_bits(), after.to_bits());
        }
    }

    // Raw-arithmetic reference for the curvature-aware controller, independent
    // of the module internals: given the pre-step state it returns the
    // effective momentum the step should apply. Mirrors the documented spec
    // (lambda_hat -> mu_curv -> min(mu_max, mu_par, mu_curv) -> release EMA).
    fn curv_reference_mu(
        buf: &[f32],
        delta: &[f32],
        prev_delta: &[f32],
        prev_dtheta: &[f32],
        mu_prev: f64,
    ) -> f64 {
        let mut dot = 0.0f64;
        let mut buf_norm_sq = 0.0f64;
        let mut delta_norm_sq = 0.0f64;
        for (b, d) in buf.iter().zip(delta) {
            dot += *b as f64 * *d as f64;
            buf_norm_sq += (*b as f64).powi(2);
            delta_norm_sq += (*d as f64).powi(2);
        }
        let (c_t, r_t) = if delta_norm_sq > 0.0 {
            let c = dot / delta_norm_sq;
            let r = ((buf_norm_sq - c * dot).max(0.0) / delta_norm_sq).sqrt();
            (c, r)
        } else {
            (0.0, 0.0)
        };
        let mut lam_dot = 0.0f64;
        let mut dtheta_norm_sq = 0.0f64;
        for ((d, pd), pt) in delta.iter().zip(prev_delta).zip(prev_dtheta) {
            lam_dot += (*d as f64 - *pd as f64) * *pt as f64;
            dtheta_norm_sq += (*pt as f64).powi(2);
        }
        let lambda_hat = lam_dot.max(0.0) / (dtheta_norm_sq + CAPPED_NESTEROV_CURV_LAMBDA_EPS);
        let mu_par =
            2.0 * CAPPED_NESTEROV_MU_MAX / (1.0 + (1.0 + 4.0 * c_t.max(0.0) * CAPPED_NESTEROV_MU_MAX).sqrt());
        let denom = lambda_hat * r_t * r_t * delta_norm_sq;
        let mu_curv = if denom > 0.0 {
            (CAPPED_NESTEROV_CURV_E_PERP / denom).powf(0.25)
        } else {
            CAPPED_NESTEROV_MU_MAX
        };
        let mut cap = CAPPED_NESTEROV_MU_MAX.min(mu_par).min(mu_curv);
        if 1.0 + cap + cap * cap * c_t < 0.0 {
            cap = 0.0;
        }
        let released =
            CAPPED_NESTEROV_EMA_BETA * mu_prev + (1.0 - CAPPED_NESTEROV_EMA_BETA) * cap;
        cap.min(released)
    }

    #[test]
    fn capped_nesterov_curv_first_commit_is_inactive() {
        // Fresh state: b_0 = 0, prev_delta = prev_dtheta = 0. The secant
        // proxy has dtheta_norm_sq = 0 so lambda_hat = 0 -> mu_curv = mu_max;
        // c = r = 0 -> mu_par = mu_max. cap = 0.9, EMA from 0.9 -> 0.9. The
        // commit is exactly plain Nesterov at mu_max, and the history states
        // are seeded: g_t = delta, dtheta_t = -(applied step).
        let lr = 0.1f32;
        let mut p = [0.0f32, 0.0];
        let mut buf = [0.0f32, 0.0];
        let mut mu = CAPPED_NESTEROV_INITIAL_MU;
        let mut prev_delta = [0.0f32, 0.0];
        let mut prev_dtheta = [0.0f32, 0.0];
        let delta = [1.0f32, 0.0];
        let stats = capped_nesterov_curv_step(
            &mut p,
            &mut buf,
            &delta,
            lr,
            &mut mu,
            &mut prev_delta,
            &mut prev_dtheta,
        );
        assert!((mu as f64 - 0.9).abs() < 1e-7);
        assert_eq!(buf, [1.0, 0.0]);
        assert!((p[0] as f64 + 0.19).abs() < 1e-6 && p[1] == 0.0);
        // history seeded for the next commit: g_1 = delta, dtheta_1 = -step_1.
        assert_eq!(prev_delta, [1.0, 0.0]);
        assert!((prev_dtheta[0] as f64 + 0.19).abs() < 1e-6 && prev_dtheta[1] == 0.0);
        assert_close(stats.direction_delta_cosine.unwrap(), 1.0);
        assert_eq!(stats.history_current_norm_ratio, Some(0.0));
    }

    #[test]
    fn capped_nesterov_curv_three_step_hand_computed_sequence() {
        // lr = 0.1, theta_0 = 0, b_0 = 0, mu_prev = 0.9, E_PERP = 22, tau not
        // used (curvature cap replaces mu_perp). All expected momenta are
        // recomputed by the independent raw-arithmetic reference above and the
        // params are checked against a plain Nesterov step at that momentum.
        //
        // t=1, delta_1 = [10, 0]: zero buffer + zero history -> lambda_hat = 0,
        //   c = r = 0, cap = 0.9, mu_1 = 0.9. b_1 = [10, 0],
        //   d_1 = 1.9*[10, 0], step_1 = [1.9, 0], theta_1 = [-1.9, 0].
        //   history: g_1 = [10, 0], dtheta_1 = -step_1 = [-1.9, 0].
        // t=2, delta_2 = [0, 10] (orthogonal, curvature turns on):
        //   Delta g = delta_2 - g_1 = [-10, 10]; <Delta g, dtheta_1> =
        //   (-10)(-1.9) = 19; |dtheta_1|^2 = 3.61; lambda_hat = 19/3.61 =
        //   5.2631579. Geometry: c = 0, r = |[10,0]|/10 = 1, |g|^2 = 100.
        //   mu_curv = (22 / (5.2631579 * 1 * 100))^{1/4} =
        //   (0.0418)^{1/4} = 0.4521605 < mu_max -> the CURVATURE cap binds.
        //   released = 0.9*0.9 + 0.1*0.4521605 > cap, so mu_2 = 0.4521605.
        // t=3, delta_3 = [5, 5] (curvature stays on, different direction):
        //   lambda_hat, geometry, and cap all recomputed by the reference.
        let lr = 0.1f32;
        let mut p = [0.0f32, 0.0];
        let mut buf = [0.0f32, 0.0];
        let mut mu = CAPPED_NESTEROV_INITIAL_MU;
        let mut prev_delta = [0.0f32, 0.0];
        let mut prev_dtheta = [0.0f32, 0.0];
        let tol = 1e-5f64;

        for delta in [[10.0f32, 0.0], [0.0, 10.0], [5.0, 5.0]] {
            // Expected momentum from the pre-step state (independent formula).
            let expected_mu =
                curv_reference_mu(&buf, &delta, &prev_delta, &prev_dtheta, mu as f64);
            // Expected plain-Nesterov transition at that (f32-rounded) mu.
            let m = expected_mu as f32;
            let expected_buf = [m * buf[0] + delta[0], m * buf[1] + delta[1]];
            let dir = [
                delta[0] + m * expected_buf[0],
                delta[1] + m * expected_buf[1],
            ];
            let expected_p = [p[0] - lr * dir[0], p[1] - lr * dir[1]];
            let expected_dtheta = [-(lr * dir[0]), -(lr * dir[1])];

            capped_nesterov_curv_step(
                &mut p,
                &mut buf,
                &delta,
                lr,
                &mut mu,
                &mut prev_delta,
                &mut prev_dtheta,
            );

            assert!((mu as f64 - expected_mu).abs() < 1e-6, "mu {mu} vs {expected_mu}");
            assert!((buf[0] as f64 - expected_buf[0] as f64).abs() < tol);
            assert!((buf[1] as f64 - expected_buf[1] as f64).abs() < tol);
            assert!((p[0] as f64 - expected_p[0] as f64).abs() < tol);
            assert!((p[1] as f64 - expected_p[1] as f64).abs() < tol);
            // History states are seeded exactly for the next commit.
            assert_eq!(prev_delta, delta);
            assert!((prev_dtheta[0] as f64 - expected_dtheta[0] as f64).abs() < tol);
            assert!((prev_dtheta[1] as f64 - expected_dtheta[1] as f64).abs() < tol);
        }

        // The t=2 commit must have bound the curvature cap strictly below
        // mu_max (the whole point of the controller): re-derive it here.
        // lambda_hat = 19/3.61, mu_curv = (22/(lambda*1*100))^{1/4}.
        let lambda2 = 19.0f64 / 3.61;
        let mu_curv2 = (CAPPED_NESTEROV_CURV_E_PERP / (lambda2 * 100.0)).powf(0.25);
        assert!(mu_curv2 < CAPPED_NESTEROV_MU_MAX);
        assert!((mu_curv2 - 0.4521605).abs() < 1e-5);
    }

    #[test]
    fn capped_nesterov_curv_high_curvature_tightens_cap() {
        // Fixed geometry (c = 0 so mu_par = mu_max, r = 1, |g|^2 = 1): the cap
        // is a strictly decreasing function of the secant curvature lambda_hat.
        // Low curvature -> mu_curv > mu_max -> cap = mu_max (momentum kept);
        // high curvature -> mu_curv < mu_max -> cap tightens toward 0.
        let cap_lo = capped_nesterov_curv_cap(0.0, 1.0, 1.0, 0.01);
        let cap_mid = capped_nesterov_curv_cap(0.0, 1.0, 1.0, 100.0);
        let cap_hi = capped_nesterov_curv_cap(0.0, 1.0, 1.0, 1000.0);
        assert!((cap_lo - CAPPED_NESTEROV_MU_MAX).abs() < 1e-12); // inactive
        assert!(cap_mid < cap_lo);
        assert!(cap_hi < cap_mid);
        // Exact values: mu_curv = (22 / lambda)^{1/4}.
        assert!((cap_mid - (22.0f64 / 100.0).powf(0.25)).abs() < 1e-12);
        assert!((cap_hi - (22.0f64 / 1000.0).powf(0.25)).abs() < 1e-12);
        // Larger merged-delta norm ||g|| also tightens the cap at fixed
        // curvature (||g||^2 enters the transverse energy budget).
        let cap_big_g = capped_nesterov_curv_cap(0.0, 1.0, 100.0, 100.0);
        assert!(cap_big_g < cap_mid);
    }

    #[test]
    fn capped_nesterov_curv_isotropic_vs_anisotropic_cap() {
        // T2/T4 sanity: SAME buffer/delta geometry (identical c, r, ||g||, so
        // identical aligned step and transverse residual), the ONLY difference
        // is the curvature the secant proxy reports. Under (near-)isotropy the
        // transverse motion is benign (T2 iso), so the curvature cap stays at
        // mu_max and momentum is preserved; under anisotropy the same
        // transverse residual costs sharp-direction curvature (T2 aniso), so
        // the cap tightens and momentum is suppressed. This is exactly the T4
        // mechanism: curvature — not the geometry (c, r) a scalar cap sees —
        // gates the cap.
        let c_t = 0.0;
        let r_t = 1.0;
        let g_sq = 25.0; // ||g||^2, a representative operating-point magnitude
        let lambda_iso = 0.05; // benign, low curvature ratio
        let lambda_aniso = 50.0; // sharp, T2-poison curvature ratio
        let cap_iso = capped_nesterov_curv_cap(c_t, r_t, g_sq, lambda_iso);
        let cap_aniso = capped_nesterov_curv_cap(c_t, r_t, g_sq, lambda_aniso);
        // Isotropy keeps full momentum; anisotropy strictly tightens it.
        assert!((cap_iso - CAPPED_NESTEROV_MU_MAX).abs() < 1e-12);
        assert!(cap_aniso < cap_iso);
        // The geometry-blind cap CANNOT tell these apart: with identical c, r
        // it returns one value regardless of curvature (the T3 blind spot).
        let blind = capped_nesterov_cap(c_t, r_t);
        assert!((capped_nesterov_cap(c_t, r_t) - blind).abs() < 1e-12);
        assert!(cap_aniso < blind); // curvature-awareness caps tighter than blind here
    }

    #[test]
    fn capped_nesterov_curv_step_is_bit_identical_to_materialized_step() {
        // Preview bit-exactness at the merge level: materialize_applied_step
        // with the updated buffer and the effective momentum written by the
        // step reproduces the applied displacement bit-for-bit, and the stored
        // dtheta equals its negation.
        let base = [0.25f32, -1.5, 3.0];
        let mut p = base;
        let mut buf = [0.5f32, 1.0, -2.0];
        let delta = [1.0f32, -0.5, 0.25];
        let mut mu = CAPPED_NESTEROV_INITIAL_MU;
        // Non-trivial history so lambda_hat is nonzero (cap may engage).
        let mut prev_delta = [0.8f32, 0.1, 0.4];
        let mut prev_dtheta = [-0.2f32, 0.05, -0.1];
        capped_nesterov_curv_step(
            &mut p,
            &mut buf,
            &delta,
            0.3,
            &mut mu,
            &mut prev_delta,
            &mut prev_dtheta,
        );
        let step =
            materialize_applied_step(OuterOptimizer::CappedNesterovCurv, &buf, &delta, 0.3, mu, 1.0);
        for ((b, s), after) in base.iter().zip(&step).zip(&p) {
            assert_eq!((b - s).to_bits(), after.to_bits());
        }
        // dtheta_t stored by the step is exactly -(applied step).
        for (pt, s) in prev_dtheta.iter().zip(&step) {
            assert_eq!(pt.to_bits(), (-s).to_bits());
        }
        // g_t stored is exactly the merged delta.
        assert_eq!(prev_delta, delta);
    }

    // ---- capped-nesterov-wsub (worker-disagreement subspace) ----

    #[test]
    fn disagreement_transverse_energy_hand_computed_three_workers() {
        // anchor = 0 so worker delta d_i = -learner_i:
        //   d_0 = [1, 0], d_1 = [0, 1], d_2 = [1, 2].
        // gbar = [2/3, 1]; s_i = d_i - gbar. Merged delta g = [1, 0]; buffer
        // b = [0, 5] is already transverse (<b, g> = 0) so b_perp = [0, 5].
        //   t_i = <d_i, b_perp> = 5 * d_i[1] = [0, 5, 10], tbar = 5,
        //   E_b = sum_i (t_i - tbar)^2 = 25 + 0 + 25 = 50.
        // Equivalently sum_i <s_i, b_perp>^2 = (-5)^2 + 0 + 5^2 = 50.
        let anchor = [0.0f32, 0.0];
        let learners_owned = [[-1.0f32, 0.0], [0.0f32, -1.0], [-1.0f32, -2.0]];
        let learners: Vec<&[f32]> = learners_owned.iter().map(|l| l.as_slice()).collect();
        let e = disagreement_transverse_energy(&anchor, &learners, &[0.0, 5.0], &[1.0, 0.0]);
        assert!((e - 50.0).abs() < 1e-9, "E_b = {e}, expected 50");

        // Buffer aligned with g -> b_perp = 0 -> no transverse energy.
        let e_aligned =
            disagreement_transverse_energy(&anchor, &learners, &[3.0, 0.0], &[1.0, 0.0]);
        assert!(e_aligned.abs() < 1e-9, "aligned buffer must give E_b = 0");

        // Zero disagreement (identical workers) -> every s_i = 0 -> E_b = 0.
        let same = [[-1.0f32, -0.5], [-1.0f32, -0.5], [-1.0f32, -0.5]];
        let same_l: Vec<&[f32]> = same.iter().map(|l| l.as_slice()).collect();
        let e_same = disagreement_transverse_energy(&anchor, &same_l, &[0.0, 5.0], &[1.0, 0.0]);
        assert!(e_same.abs() < 1e-9, "agreeing workers must give E_b = 0");

        // M = 1 degenerate: no disagreement subspace exists.
        let one: Vec<&[f32]> = vec![learners_owned[0].as_slice()];
        let e_one = disagreement_transverse_energy(&anchor, &one, &[0.0, 5.0], &[1.0, 0.0]);
        assert!(e_one == 0.0, "single worker must give E_b = 0");
    }

    #[test]
    fn capped_nesterov_wsub_cap_is_directional_and_guarded() {
        // Zero disagreement energy leaves the transverse cap inactive: cap is
        // the aligned cap alone (mu_max at c = 0).
        assert_eq!(capped_nesterov_wsub_cap(0.0, 0.0), CAPPED_NESTEROV_MU_MAX);
        // mu_wsub = (E_SUB / E_b)^{1/4}: large disagreement energy tightens the
        // cap below mu_max; the cap is a strictly decreasing function of E_b.
        let cap_lo = capped_nesterov_wsub_cap(0.0, 1.0);
        let cap_hi = capped_nesterov_wsub_cap(0.0, 1000.0);
        assert!(cap_hi < cap_lo);
        assert!(cap_lo >= cap_hi);
        // Exact value at E_b = 50: (22/50)^{1/4}.
        let cap50 = capped_nesterov_wsub_cap(0.0, 50.0);
        let expected = (CAPPED_NESTEROV_WSUB_E_SUB / 50.0).powf(0.25);
        assert!((cap50 - expected).abs() < 1e-12);
        assert!(cap50 < CAPPED_NESTEROV_MU_MAX);
        // Sign-reversal guard: strongly opposing aligned history with cap that
        // would drive A_t < 0 is zeroed (same guard as the frozen controller).
        assert_eq!(capped_nesterov_wsub_cap(-10.0, 0.0), 0.0);
    }

    #[test]
    fn capped_nesterov_wsub_transverse_cap_binds_at_budget() {
        // Worker disagreement is entirely in the transverse (buffer) direction:
        // E_b = 50 (from the hand-computed scenario). buf = [0, 5], delta =
        // [1, 0], lr = 1: c = 0 so mu_par = mu_max; mu_wsub = (22/50)^{1/4} =
        // 0.81444682, far below the release-EMA path (0.9*0.9 + 0.1*mu_wsub =
        // 0.89144468), so the disagreement cap binds hard: mu = mu_wsub.
        let mut p = [0.0f32, 0.0];
        let mut buf = [0.0f32, 5.0];
        let mut mu = CAPPED_NESTEROV_INITIAL_MU;
        let e_b = 50.0;
        let stats = capped_nesterov_wsub_step(&mut p, &mut buf, &[1.0, 0.0], 1.0, &mut mu, e_b);
        let expected_mu = (CAPPED_NESTEROV_WSUB_E_SUB / e_b).powf(0.25);
        assert!((mu as f64 - expected_mu).abs() < 1e-7);
        assert!((mu as f64) < CAPPED_NESTEROV_MU_MAX);
        // At the binding cap the transverse curvature energy sits exactly on
        // the budget: mu^4 * E_b = E_SUB (the defining invariant of the cap).
        let applied_energy = (mu as f64).powi(4) * e_b;
        assert!((applied_energy - CAPPED_NESTEROV_WSUB_E_SUB).abs() < 1e-3);
        // Nesterov recursion at the capped mu: b_new = mu*[0,5] + [1,0], and
        // the applied transverse displacement is mu^2 * b (delta is along x).
        assert!((buf[0] as f64 - 1.0).abs() < 1e-6);
        assert!((buf[1] as f64 - 5.0 * expected_mu).abs() < 1e-5);
        assert!((p[0] as f64 + (1.0 + expected_mu)).abs() < 1e-6);
        assert!((p[1] as f64 + expected_mu * expected_mu * 5.0).abs() < 1e-5);
        assert!(stats.applied_step_norm.is_finite());
    }

    #[test]
    fn capped_nesterov_wsub_zero_disagreement_runs_at_mu_max() {
        // No worker disagreement (E_b = 0): the transverse cap is inactive, so
        // the first commit runs at exactly mu_max, identical to tuned Nesterov.
        // A parallel plain-Nesterov run at mu_max must be bit-identical.
        let mut p = [0.0f32, 0.0, 0.0];
        let mut buf = [0.3f32, -0.7, 1.1];
        let mut mu = CAPPED_NESTEROV_INITIAL_MU;
        let delta = [1.0f32, 0.5, -0.25];
        capped_nesterov_wsub_step(&mut p, &mut buf, &delta, 0.2, &mut mu, 0.0);
        assert_eq!(mu, CAPPED_NESTEROV_MU_MAX as f32);
        let mut p_ref = [0.0f32, 0.0, 0.0];
        let mut buf_ref = [0.3f32, -0.7, 1.1];
        nesterov_step(&mut p_ref, &mut buf_ref, &delta, 0.2, CAPPED_NESTEROV_MU_MAX as f32);
        for (a, b) in p.iter().zip(&p_ref) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn capped_nesterov_wsub_m1_falls_back_to_aligned_cap_only() {
        // With one worker the disagreement energy the merge threads is 0, so
        // the transverse cap is inert and only the aligned cap governs. Aligned
        // history (c = 1) then caps mu at the aligned root, NOT mu_max.
        let mut p = [0.0f32, 0.0];
        let mut buf = [2.0f32, 0.0];
        let mut mu = CAPPED_NESTEROV_INITIAL_MU;
        // delta = [1, 0] -> c = 2 (aligned, amplifying); aligned cap binds.
        let stats = capped_nesterov_wsub_step(&mut p, &mut buf, &[1.0, 0.0], 1.0, &mut mu, 0.0);
        let mu_par = capped_nesterov_mu_par(2.0);
        // released EMA from mu_max stays above the aligned cap, so it binds.
        assert!((mu as f64 - mu_par).abs() < 1e-7);
        assert!((mu as f64) < CAPPED_NESTEROV_MU_MAX);
        // Aligned gain pinned at the design ceiling 1 + mu_max.
        let a_t = 1.0 + mu as f64 + (mu as f64).powi(2) * 2.0;
        assert!((a_t - (1.0 + CAPPED_NESTEROV_MU_MAX)).abs() < 1e-6);
        assert!(stats.applied_step_norm.is_finite());
    }

    #[test]
    fn capped_nesterov_wsub_step_is_bit_identical_to_materialized_step() {
        let base = [0.25f32, -1.5, 3.0];
        let mut p = base;
        let mut buf = [0.5f32, 1.0, -2.0];
        let delta = [1.0f32, -0.5, 0.25];
        let mut mu = CAPPED_NESTEROV_INITIAL_MU;
        // A disagreement energy large enough that the transverse cap engages.
        capped_nesterov_wsub_step(&mut p, &mut buf, &delta, 0.3, &mut mu, 500.0);
        assert!((mu as f64) < CAPPED_NESTEROV_MU_MAX, "cap must engage");
        let step = materialize_applied_step(
            OuterOptimizer::CappedNesterovWsub,
            &buf,
            &delta,
            0.3,
            mu,
            1.0,
        );
        for ((b, s), after) in base.iter().zip(&step).zip(&p) {
            assert_eq!((b - s).to_bits(), after.to_bits());
        }
    }

    #[test]
    fn capped_nesterov_r_three_step_hand_computed_sequence() {
        // Same deterministic scenario as the base three-step audit (lr = 0.1,
        // theta_0 = 0, b_0 = 0, mu_prev = 0.9) but with the relaxed
        // transverse budget tau_perp = 1.55. t=1 and t=2 involve no
        // transverse residual (r_t = 0), so they are IDENTICAL to the frozen
        // controller; the variants only part ways at t=3.
        //
        // t=1, delta_1 = [1, 0]: caps inactive, mu_1 = 0.9, b_1 = [1, 0],
        //   step_1 = [0.19, 0], theta_1 = [-0.19, 0].
        // t=2, delta_2 = [1, 0]: c_2 = 1, r_2 = 0; aligned cap binds at
        //   mu_2 = (sqrt(4.6) - 1)/2 = 0.57238053; step_2 = [0.19, 0],
        //   theta_2 = [-0.38, 0], b_2 = [1 + mu_2, 0].
        // t=3, delta_3 = [0, 1]: c_3 = 0, r_3 = 1 + mu_2 = 1.57238053.
        //   mu_perp = sqrt(1.55/1.57238053) = 0.99286744 > mu_max, so unlike
        //   the tau = 1 budget (which capped at 0.79748276) the relaxed
        //   transverse cap does NOT bind: cap = mu_max = 0.9. The release
        //   EMA still binds from below:
        //   mu_3 = 0.9*mu_2 + 0.1*0.9 = 0.60514248 < 0.9.
        //   b_3 = mu_3*[1.57238053, 0] + [0, 1] = [0.95151846, 1],
        //   d_3 = [0, 1] + mu_3*b_3 = [0.57580557, 1.60514248],
        //   step_3 = [0.05758056, 0.16051425].
        let lr = 0.1f32;
        let mut p = [0.0f32, 0.0];
        let mut buf = [0.0f32, 0.0];
        let mut mu = CAPPED_NESTEROV_INITIAL_MU;
        let tol = 1e-5f64;

        let s1 = capped_nesterov_r_step(&mut p, &mut buf, &[1.0, 0.0], lr, &mut mu);
        assert!((mu as f64 - 0.9).abs() < 1e-7);
        assert_eq!(buf, [1.0, 0.0]);
        assert!((p[0] as f64 + 0.19).abs() < tol && p[1] == 0.0);
        assert!((s1.applied_step_norm - 0.19).abs() < tol);

        let s2 = capped_nesterov_r_step(&mut p, &mut buf, &[1.0, 0.0], lr, &mut mu);
        let mu2 = ((1.0f64 + 4.0 * 0.9).sqrt() - 1.0) / 2.0;
        assert!((mu as f64 - mu2).abs() < 1e-7);
        assert!((buf[0] as f64 - (1.0 + mu2)).abs() < tol && buf[1] == 0.0);
        assert!((p[0] as f64 + 0.38).abs() < tol && p[1] == 0.0);
        assert!((s2.applied_step_norm - 0.19).abs() < tol);

        let s3 = capped_nesterov_r_step(&mut p, &mut buf, &[0.0, 1.0], lr, &mut mu);
        // Relaxed budget: mu_perp > mu_max, transverse cap inactive.
        assert!((CAPPED_NESTEROV_R_TAU_PERP / (1.0 + mu2)).sqrt() > CAPPED_NESTEROV_MU_MAX);
        let mu3 = 0.9 * mu2 + 0.1 * CAPPED_NESTEROV_MU_MAX;
        assert!(mu3 < CAPPED_NESTEROV_MU_MAX);
        assert!((mu as f64 - mu3).abs() < 1e-7);
        let b3 = [mu3 * (1.0 + mu2), 1.0];
        assert!((buf[0] as f64 - b3[0]).abs() < tol && (buf[1] as f64 - 1.0).abs() < tol);
        let d3 = [mu3 * b3[0], 1.0 + mu3 * b3[1]];
        assert!((p[0] as f64 + (0.38 + 0.1 * d3[0])).abs() < tol);
        assert!((p[1] as f64 + 0.1 * d3[1]).abs() < tol);
        let d3_norm = (d3[0] * d3[0] + d3[1] * d3[1]).sqrt();
        assert!((s3.applied_step_norm - 0.1 * d3_norm).abs() < tol);
        // Strictly more momentum than the frozen controller at this commit.
        let frozen_mu3 = 0.9 * mu2 + 0.1 * (1.0f64 / (1.0 + mu2)).sqrt();
        assert!(mu3 > frozen_mu3);
    }

    #[test]
    fn capped_nesterov_r_transverse_cap_binds_at_relaxed_budget() {
        // b = [0, 10], delta = [1, 0]: c = 0, r = 10. mu_perp =
        // sqrt(1.55/10) = 0.39370039 (vs sqrt(0.1) = 0.31622777 at tau = 1),
        // far below the EMA path (0.9*0.9 + 0.1*cap = 0.84937), so the
        // relaxed transverse cap binds: mu = sqrt(0.155). The applied step's
        // delta-orthogonal component is mu^2*b = 0.155*[0, 10] with norm
        // exactly tau_perp*|delta| = 1.55.
        let mut p = [0.0f32, 0.0];
        let mut buf = [0.0f32, 10.0];
        let mut mu = CAPPED_NESTEROV_INITIAL_MU;
        capped_nesterov_r_step(&mut p, &mut buf, &[1.0, 0.0], 1.0, &mut mu);
        let expected_mu = (CAPPED_NESTEROV_R_TAU_PERP / 10.0).sqrt();
        assert!((mu as f64 - expected_mu).abs() < 1e-7);
        assert!(expected_mu > (CAPPED_NESTEROV_TAU_PERP / 10.0).sqrt());
        // b_new = mu*[0, 10] + [1, 0]; d = delta + mu*b_new
        //       = [1 + mu, 10*mu^2] = [1.39370039, 1.55].
        assert!((buf[0] as f64 - 1.0).abs() < 1e-6);
        assert!((buf[1] as f64 - 10.0 * expected_mu).abs() < 1e-5);
        assert!((p[0] as f64 + (1.0 + expected_mu)).abs() < 1e-6);
        assert!((p[1] as f64 + CAPPED_NESTEROV_R_TAU_PERP).abs() < 1e-5);
    }

    #[test]
    fn capped_nesterov_r_step_is_bit_identical_to_materialized_step() {
        let base = [0.25f32, -1.5, 3.0];
        let mut p = base;
        let mut buf = [0.5f32, 1.0, -2.0];
        let delta = [1.0f32, -0.5, 0.25];
        let mut mu = CAPPED_NESTEROV_INITIAL_MU;
        capped_nesterov_r_step(&mut p, &mut buf, &delta, 0.3, &mut mu);
        let step = materialize_applied_step(
            OuterOptimizer::CappedNesterovR,
            &buf,
            &delta,
            0.3,
            mu,
            1.0,
        );
        for ((b, s), after) in base.iter().zip(&step).zip(&p) {
            assert_eq!((b - s).to_bits(), after.to_bits());
        }
    }

    #[test]
    fn capped_nesterov_gc_three_step_hand_computed_sequence() {
        // Same deterministic scenario as the base three-step audit (lr = 0.1,
        // theta_0 = 0, b_0 = 0, mu_prev = 0.9, gain_0 = 1); caps are the
        // frozen controller's (tau_perp = 1), so mu_t matches the base
        // variant at every commit and only the applied displacement changes.
        //
        // t=1, delta_1 = [1, 0]: mu_1 = 0.9, A_1 = 1 + 0.9 = 1.9,
        //   g_1 = 1.9/1.9 = 1: exactly the base step. b_1 = [1, 0],
        //   step_1 = [0.19, 0], theta_1 = [-0.19, 0].
        // t=2, delta_2 = [1, 0]: c_2 = 1, aligned cap binds at
        //   mu_2 = 0.57238053 and by construction A_2 = 1 + mu_2 + mu_2^2
        //   = 1.9, so g_2 = 1 again: when the aligned cap binds the frozen
        //   controller already delivers the design gain and gc changes
        //   nothing. step_2 = [0.19, 0], theta_2 = [-0.38, 0].
        // t=3, delta_3 = [0, 1]: c_3 = 0, r_3 = 1.57238053; transverse cap
        //   plus release EMA give mu_3 = 0.59489075 (base variant), and the
        //   understeer appears: A_3 = 1 + mu_3 = 1.59489075 < 1.9. Gain
        //   compensation restores it: g_3 = 1.9/1.59489075 = 1.19130347.
        //   b_3 = mu_3*[1.57238053, 0] + [0, 1] = [0.93539497, 1],
        //   d_3 = [0.55645437, 1.59489075],
        //   applied direction g_3*d_3 = [0.66289245, 1.9] — the aligned
        //   component is pinned at exactly (1 + mu_max) = 1.9.
        //   step_3 = [0.06628925, 0.19].
        let lr = 0.1f32;
        let mut p = [0.0f32, 0.0];
        let mut buf = [0.0f32, 0.0];
        let mut mu = CAPPED_NESTEROV_INITIAL_MU;
        let mut gain = CAPPED_NESTEROV_GC_INITIAL_GAIN;
        let tol = 1e-5f64;

        let s1 = capped_nesterov_gc_step(&mut p, &mut buf, &[1.0, 0.0], lr, &mut mu, &mut gain);
        assert!((mu as f64 - 0.9).abs() < 1e-7);
        assert!((gain as f64 - 1.0).abs() < 1e-6);
        assert_eq!(buf, [1.0, 0.0]);
        assert!((p[0] as f64 + 0.19).abs() < tol && p[1] == 0.0);
        assert!((s1.applied_step_norm - 0.19).abs() < tol);
        assert_eq!(s1.history_current_norm_ratio, Some(0.0));

        let s2 = capped_nesterov_gc_step(&mut p, &mut buf, &[1.0, 0.0], lr, &mut mu, &mut gain);
        let mu2 = ((1.0f64 + 4.0 * 0.9).sqrt() - 1.0) / 2.0;
        assert!((mu as f64 - mu2).abs() < 1e-7);
        assert!((gain as f64 - 1.0).abs() < 1e-6);
        assert!((buf[0] as f64 - (1.0 + mu2)).abs() < tol && buf[1] == 0.0);
        assert!((p[0] as f64 + 0.38).abs() < tol && p[1] == 0.0);
        assert!((s2.applied_step_norm - 0.19).abs() < tol);

        let s3 = capped_nesterov_gc_step(&mut p, &mut buf, &[0.0, 1.0], lr, &mut mu, &mut gain);
        let cap3 = (1.0f64 / (1.0 + mu2)).sqrt();
        let mu3 = 0.9 * mu2 + 0.1 * cap3;
        assert!((mu as f64 - mu3).abs() < 1e-7);
        let g3 = 1.9 / (1.0 + mu3);
        assert!((gain as f64 - g3).abs() < 1e-6);
        let b3 = [mu3 * (1.0 + mu2), 1.0];
        assert!((buf[0] as f64 - b3[0]).abs() < tol && (buf[1] as f64 - 1.0).abs() < tol);
        let d3 = [g3 * mu3 * b3[0], g3 * (1.0 + mu3)];
        // Aligned effective gain pinned at the design point.
        assert!((d3[1] - 1.9).abs() < 1e-12);
        assert!((p[0] as f64 + (0.38 + 0.1 * d3[0])).abs() < tol);
        // theta_3[1] = -0.1 * 1.9 = -0.19: same aligned displacement as t=1.
        assert!((p[1] as f64 + 0.19).abs() < tol);
        let d3_norm = (d3[0] * d3[0] + d3[1] * d3[1]).sqrt();
        assert!((s3.applied_step_norm - 0.1 * d3_norm).abs() < tol);
    }

    #[test]
    fn capped_nesterov_gc_transverse_cap_binds_and_gain_compensates() {
        // b = [0, 10], delta = [1, 0], lr = 1: transverse cap binds at
        // mu = sqrt(0.1) exactly as in the base variant, which would leave
        // the aligned gain at A = 1 + mu = 1.31622777 (understeer). The gc
        // rescale g = 1.9/1.31622777 = 1.44351811 restores the aligned
        // component to exactly 1.9; the transverse component becomes
        // g*mu^2*|b| = 1.44351811 (scalar rescale scales both).
        let mut p = [0.0f32, 0.0];
        let mut buf = [0.0f32, 10.0];
        let mut mu = CAPPED_NESTEROV_INITIAL_MU;
        let mut gain = CAPPED_NESTEROV_GC_INITIAL_GAIN;
        capped_nesterov_gc_step(&mut p, &mut buf, &[1.0, 0.0], 1.0, &mut mu, &mut gain);
        let expected_mu = 0.1f64.sqrt();
        let expected_g = 1.9 / (1.0 + expected_mu);
        assert!((mu as f64 - expected_mu).abs() < 1e-7);
        assert!((gain as f64 - expected_g).abs() < 1e-6);
        // Buffer keeps the plain capped-Nesterov recursion (no rescale).
        assert!((buf[0] as f64 - 1.0).abs() < 1e-6);
        assert!((buf[1] as f64 - 10.0 * expected_mu).abs() < 1e-5);
        // Applied step: aligned component pinned at 1.9, transverse g*1.
        assert!((p[0] as f64 + 1.9).abs() < 1e-5);
        assert!((p[1] as f64 + expected_g).abs() < 1e-5);
    }

    #[test]
    fn capped_nesterov_gc_gain_is_clamped_and_never_dampens() {
        // Strongly opposing but guard-admissible history: b = [-1.45, 0.1],
        // delta = [1, 0]: c = -1.45, r = 0.1. c_plus = 0 -> mu_par = 0.9;
        // mu_perp = sqrt(1/0.1) = 3.16 (inactive); cap = 0.9;
        // A(0.9) = 1.9 + 0.81*(-1.45) = 0.7255 > 0 (guard idle); EMA path
        // stays at mu = 0.9. Raw gain 1.9/0.7255 = 2.61888 exceeds GAIN_MAX
        // and must clamp to 2.5.
        let mut p = [0.0f32, 0.0];
        let mut buf = [-1.45f32, 0.1];
        let mut mu = CAPPED_NESTEROV_INITIAL_MU;
        let mut gain = CAPPED_NESTEROV_GC_INITIAL_GAIN;
        capped_nesterov_gc_step(&mut p, &mut buf, &[1.0, 0.0], 0.1, &mut mu, &mut gain);
        assert!((mu as f64 - 0.9).abs() < 1e-7);
        assert_eq!(gain as f64, CAPPED_NESTEROV_GC_GAIN_MAX);

        // mu_t <= cap implies A_t <= 1 + mu_max on the admissible set, so
        // the raw gain is always >= 1: gc only compensates understeer, it
        // never dampens below the design step.
        for (b, d) in [
            ([2.0f32, 0.0], [1.0f32, 0.0]),   // aligned cap binds
            ([0.0f32, 10.0], [1.0f32, 0.0]),  // transverse cap binds
            ([0.0f32, 0.0], [1.0f32, 1.0]),   // fresh state
            ([-10.0f32, 0.0], [1.0f32, 0.0]), // sign guard zeroes the cap
        ] {
            let mut p = [0.0f32, 0.0];
            let mut buf = b;
            let mut mu = CAPPED_NESTEROV_INITIAL_MU;
            let mut gain = CAPPED_NESTEROV_GC_INITIAL_GAIN;
            capped_nesterov_gc_step(&mut p, &mut buf, &d, 0.1, &mut mu, &mut gain);
            assert!(
                gain as f64 >= 1.0 && gain as f64 <= CAPPED_NESTEROV_GC_GAIN_MAX,
                "gain {gain} out of range for buffer {b:?}"
            );
        }
    }

    #[test]
    fn capped_nesterov_gc_step_is_bit_identical_to_materialized_step() {
        // The gc branch of materialize_applied_step must reproduce the
        // applied step bit-for-bit from the updated buffer, the effective
        // momentum, AND the applied gain written back by the step.
        let base = [0.25f32, -1.5, 3.0];
        let mut p = base;
        let mut buf = [0.5f32, 1.0, -2.0];
        let delta = [1.0f32, -0.5, 0.25];
        let mut mu = CAPPED_NESTEROV_INITIAL_MU;
        let mut gain = CAPPED_NESTEROV_GC_INITIAL_GAIN;
        capped_nesterov_gc_step(&mut p, &mut buf, &delta, 0.3, &mut mu, &mut gain);
        assert!(gain > 1.0, "scenario must exercise a non-trivial gain");
        let step = materialize_applied_step(
            OuterOptimizer::CappedNesterovGc,
            &buf,
            &delta,
            0.3,
            mu,
            gain,
        );
        for ((b, s), after) in base.iter().zip(&step).zip(&p) {
            assert_eq!((b - s).to_bits(), after.to_bits());
        }
    }

    #[test]
    fn capped_nesterov_cap_mu_par_is_stable_and_continuous_at_small_c() {
        // THEORY.md F1 regression: the textbook root form
        // (sqrt(1+4c*mu_max)-1)/(2c) cancels catastrophically for small
        // positive c (~0.555 at c = 2e-16; exactly 0 at c = 1e-20). The
        // rationalized form must return mu_max in that regime, restoring
        // the "largest admissible mu" spec and continuity at c -> 0+.
        // r_t = 0 keeps mu_perp inert (~1e6).
        assert_eq!(capped_nesterov_cap(0.0, 0.0), CAPPED_NESTEROV_MU_MAX);
        for c in [1e-20f64, 2e-16, 1e-12, 1e-9] {
            let cap = capped_nesterov_cap(c, 0.0);
            assert!(
                (cap - CAPPED_NESTEROV_MU_MAX).abs() < 1e-8,
                "cap({c:e}) = {cap} should be ~mu_max"
            );
            // Still the exact root: never exceeds mu_max admissibility.
            assert!(cap + cap * cap * c <= CAPPED_NESTEROV_MU_MAX + 1e-12);
        }
    }

    #[test]
    fn capped_nesterov_cap_mu_par_matches_root_at_moderate_c() {
        // At c where the old form was accurate the rationalized form is
        // algebraically identical: c = 1 gives the three-step-audit value
        // mu_par = (sqrt(4.6) - 1)/2 = 0.57238053..., and the defining
        // quadratic mu + mu^2 c = mu_max holds to f64 precision.
        for c in [0.25f64, 0.5, 1.0, 2.0, 10.0] {
            let cap = capped_nesterov_cap(c, 0.0);
            assert!(
                (cap + cap * cap * c - CAPPED_NESTEROV_MU_MAX).abs() < 1e-12,
                "root property violated at c = {c}"
            );
        }
        let audit = ((1.0f64 + 4.0 * CAPPED_NESTEROV_MU_MAX).sqrt() - 1.0) / 2.0;
        assert!((capped_nesterov_cap(1.0, 0.0) - audit).abs() < 1e-15);
        // Monotone nonincreasing in c (larger aligned gain -> tighter cap).
        let mut prev = capped_nesterov_cap(0.0, 0.0);
        for c in [1e-6f64, 1e-3, 0.1, 1.0, 10.0, 1e6] {
            let cap = capped_nesterov_cap(c, 0.0);
            assert!(cap <= prev + 1e-15);
            prev = cap;
        }
    }

    #[test]
    fn nesterov_zero_delta_leaves_direction_ratios_undefined() {
        let mut p = [1.0f32];
        let mut buf = [2.0f32];
        let stats = nesterov_step(&mut p, &mut buf, &[0.0], 0.25, 0.5);
        assert_eq!(buf, [1.0]);
        assert_eq!(p, [0.875]);
        assert_close(stats.applied_step_norm, 0.125);
        assert_eq!(stats.direction_delta_cosine, None);
        assert_eq!(stats.history_current_norm_ratio, None);
        assert!(!stats.restarted);
    }

    #[test]
    fn normalized_ema_matches_reference() {
        let mut p = [1.0f32, -1.0];
        let mut buf = [0.0f32, 0.0];
        let first = normalized_ema_step(&mut p, &mut buf, &[0.5, -0.25], 0.2, 0.8);
        assert_eq!(buf, [0.5, -0.25]);
        assert!((p[0] - 0.9).abs() < 1e-6);
        assert!((p[1] + 0.95).abs() < 1e-6);
        assert_close(first.applied_step_norm, norm(&[0.1, -0.05]));
        assert_close(first.direction_delta_cosine.unwrap(), 1.0);
        assert_eq!(first.history_current_norm_ratio, Some(0.0));
        assert!(!first.restarted);

        let second = normalized_ema_step(&mut p, &mut buf, &[1.0, 0.75], 0.2, 0.8);
        assert!((buf[0] - 0.6).abs() < 1e-6);
        assert!((buf[1] + 0.05).abs() < 1e-6);
        assert!((p[0] - 0.78).abs() < 1e-6);
        assert!((p[1] + 0.94).abs() < 1e-6);
        assert_close(second.applied_step_norm, norm(&[0.12, -0.01]));
        assert_close(
            second.direction_delta_cosine.unwrap(),
            cosine(&buf, &[1.0, 0.75]),
        );
        let expected_ratio =
            (0.8f32 as f64 * norm(&[0.5, -0.25])) / ((1.0f32 - 0.8) as f64 * norm(&[1.0, 0.75]));
        assert_close(second.history_current_norm_ratio.unwrap(), expected_ratio);
        assert!(!second.restarted);
    }

    #[test]
    fn normalized_ema_has_unit_gain_from_first_constant_gradient() {
        let mut p = [2.0f32];
        let mut buf = [0.0f32];
        for step in 1..=5 {
            normalized_ema_step(&mut p, &mut buf, &[0.5], 0.2, 0.9);
            assert!((buf[0] - 0.5).abs() < 1e-6);
            assert!((p[0] - (2.0 - step as f32 * 0.1)).abs() < 1e-6);
        }
    }

    #[test]
    fn normalized_ema_zero_delta_leaves_direction_ratios_undefined() {
        let mut p = [1.0f32];
        let mut buf = [1.0f32];
        let stats = normalized_ema_step(&mut p, &mut buf, &[0.0], 0.5, 0.5);
        assert_eq!(buf, [0.5]);
        assert_eq!(p, [0.75]);
        assert_close(stats.applied_step_norm, 0.25);
        assert_eq!(stats.direction_delta_cosine, None);
        assert_eq!(stats.history_current_norm_ratio, None);
        assert!(!stats.restarted);
    }

    #[test]
    fn restarted_ema_discards_conflicting_history() {
        let mut p = [0.0f32, 0.0];
        let mut buf = [1.0f32, 0.0];
        let stats = restarted_ema_step(&mut p, &mut buf, &[-2.0, 0.0], 0.25, 0.9, 0.0);
        assert_eq!(buf, [-2.0, 0.0]);
        assert_eq!(p, [0.5, 0.0]);
        assert_close(stats.applied_step_norm, 0.5);
        assert_close(stats.direction_delta_cosine.unwrap(), 1.0);
        assert_eq!(stats.history_current_norm_ratio, Some(0.0));
        assert!(stats.restarted);
    }

    #[test]
    fn restarted_ema_handles_zero_norms_deterministically() {
        let mut p = [1.0f32];
        let mut buf = [0.0f32];
        let empty = restarted_ema_step(&mut p, &mut buf, &[0.0], 1.0, 0.5, 0.0);
        assert_eq!(buf, [0.0]);
        assert_eq!(p, [1.0]);
        assert_eq!(empty.applied_step_norm, 0.0);
        assert_eq!(empty.direction_delta_cosine, None);
        assert_eq!(empty.history_current_norm_ratio, None);
        assert!(!empty.restarted);

        buf[0] = 1.0;
        let decayed = restarted_ema_step(&mut p, &mut buf, &[0.0], 1.0, 0.5, 0.0);
        assert_eq!(buf, [0.5]);
        assert_eq!(p, [0.5]);
        assert_close(decayed.applied_step_norm, 0.5);
        assert_eq!(decayed.direction_delta_cosine, None);
        assert_eq!(decayed.history_current_norm_ratio, None);
        assert!(!decayed.restarted);
    }

    #[test]
    fn outer_optimizer_names_are_strict() {
        assert_eq!("nesterov".parse(), Ok(OuterOptimizer::Nesterov));
        assert_eq!("normalized-ema".parse(), Ok(OuterOptimizer::NormalizedEma));
        assert_eq!("restarted-ema".parse(), Ok(OuterOptimizer::RestartedEma));
        assert_eq!("capped-nesterov".parse(), Ok(OuterOptimizer::CappedNesterov));
        assert_eq!(
            "capped-nesterov-gc".parse(),
            Ok(OuterOptimizer::CappedNesterovGc)
        );
        assert_eq!(
            "capped-nesterov-r".parse(),
            Ok(OuterOptimizer::CappedNesterovR)
        );
        assert_eq!(
            "capped-nesterov-curv".parse(),
            Ok(OuterOptimizer::CappedNesterovCurv)
        );
        assert_eq!(
            "capped-nesterov-wsub".parse(),
            Ok(OuterOptimizer::CappedNesterovWsub)
        );
        assert_eq!(
            OuterOptimizer::CappedNesterovWsub.to_string(),
            "capped-nesterov-wsub"
        );
        assert!("ema".parse::<OuterOptimizer>().is_err());
        assert!("capped-nesterov-x".parse::<OuterOptimizer>().is_err());
        assert_eq!("block-rms".parse(), Ok(OuterOptimizer::BlockRms));
        assert_eq!("block-yogi".parse(), Ok(OuterOptimizer::BlockYogi));
        assert_eq!(OuterOptimizer::BlockRms.to_string(), "block-rms");
        assert_eq!(OuterOptimizer::BlockYogi.to_string(), "block-yogi");
    }

    // ---- worker-SNR merge mode ----

    /// Build M per-worker learner vectors from a shared anchor and per-worker
    /// deltas (learner_i = anchor - delta_i), the layout the merge consumes.
    fn learners_from_deltas(anchor: &[f32], deltas: &[Vec<f32>]) -> Vec<Vec<f32>> {
        deltas
            .iter()
            .map(|d| anchor.iter().zip(d).map(|(a, di)| a - di).collect())
            .collect()
    }

    #[test]
    fn worker_snr_preserves_high_consensus_and_shrinks_high_disagreement() {
        // Two tensor blocks of size 2. Block 0: all workers agree exactly
        // (zero variance -> q ~ 1, kept). Block 1: same mean magnitude but the
        // workers strongly disagree (large variance -> q < 1, shrunk). After
        // the global norm-match, block 0 must retain (essentially) all of the
        // global step energy and block 1 almost none.
        let anchor = [0.0f32, 0.0, 0.0, 0.0];
        let deltas = vec![
            vec![1.0f32, 1.0, 3.0, -3.0],
            vec![1.0f32, 1.0, -3.0, 3.0],
            vec![1.0f32, 1.0, 3.0, -3.0],
            vec![1.0f32, 1.0, -3.0, 3.0],
        ];
        let learners = learners_from_deltas(&anchor, &deltas);
        let refs: Vec<&[f32]> = learners.iter().map(Vec::as_slice).collect();
        let weights = vec![1.0; refs.len()];
        let mut out = vec![0.0f32; 4];
        merge_worker_snr(&anchor, &refs, &weights, &[2, 2], &mut out);

        let consensus_energy = (out[0] as f64).powi(2) + (out[1] as f64).powi(2);
        let disagree_energy = (out[2] as f64).powi(2) + (out[3] as f64).powi(2);
        // gbar of block 1 is ~0 (workers cancel), so it is shrunk to nothing.
        assert!(
            disagree_energy < 1e-6,
            "disagreement block not shrunk: {disagree_energy}"
        );
        assert!(consensus_energy > 1e-3, "consensus block lost energy");
        // Norm-match restores the plain-mean merged-delta norm globally.
        let gbar: Vec<f32> = (0..4)
            .map(|j| deltas.iter().map(|d| d[j]).sum::<f32>() / deltas.len() as f32)
            .collect();
        assert_close(norm(&out), norm(&gbar));
    }

    #[test]
    fn worker_snr_norm_match_is_exact() {
        // Arbitrary asymmetric deltas across three tensor blocks; the merged
        // output norm must equal the unweighted-mean delta norm exactly.
        let anchor = [0.5f32, -1.0, 2.0, 0.0, -0.5, 1.5];
        let deltas = vec![
            vec![0.2f32, -0.4, 1.0, 0.3, -0.1, 0.7],
            vec![0.6f32, 0.1, -0.2, 0.9, 0.4, -0.3],
            vec![-0.1f32, 0.5, 0.3, -0.6, 0.2, 0.8],
        ];
        let learners = learners_from_deltas(&anchor, &deltas);
        let refs: Vec<&[f32]> = learners.iter().map(Vec::as_slice).collect();
        let weights = vec![1.0; refs.len()];
        let mut out = vec![0.0f32; 6];
        merge_worker_snr(&anchor, &refs, &weights, &[1, 2, 3], &mut out);
        let gbar: Vec<f32> = (0..6)
            .map(|j| deltas.iter().map(|d| d[j]).sum::<f32>() / deltas.len() as f32)
            .collect();
        assert_close(norm(&out), norm(&gbar));
    }

    #[test]
    fn worker_snr_single_worker_is_identity() {
        // M = 1: variance is 0, q ~ 1, norm-match cancels -> the single delta.
        let anchor = [1.0f32, -2.0, 0.5, 3.0];
        let delta = vec![0.3f32, -0.7, 0.2, 1.1];
        let learners = learners_from_deltas(&anchor, &[delta.clone()]);
        let refs: Vec<&[f32]> = learners.iter().map(Vec::as_slice).collect();
        let mut out = vec![0.0f32; 4];
        merge_worker_snr(&anchor, &refs, &[1.0], &[2, 2], &mut out);
        for (got, want) in out.iter().zip(&delta) {
            assert_close(*got as f64, *want as f64);
        }
    }

    // ---- block second-moment optimizers ----

    #[test]
    fn block_rms_first_step_v_init_and_norm_match() {
        // Fresh v = 0. Step 1 -> v_l = (1 - beta2) * ||g_l||^2 / d_l, and the
        // global norm-matched applied step has norm lr * ||g||.
        let lr = 0.28f32;
        let delta = [3.0f32, 4.0, 1.0, 1.0]; // block 0 rms bigger than block 1
        let numels = [2usize, 2];
        let mut params = [0.0f32; 4];
        let mut buf = [0.0f32; 4];
        let mut block_v = [0.0f32; 2];
        let stats =
            block_second_moment_step(&mut params, &mut buf, &delta, lr, &numels, &mut block_v, false);
        // v_0 = 0.05 * (9 + 16)/2 = 0.625 ; v_1 = 0.05 * (1 + 1)/2 = 0.05.
        assert_close(block_v[0] as f64, 0.05 * 25.0 / 2.0);
        assert_close(block_v[1] as f64, 0.05 * 2.0 / 2.0);
        // Applied step norm == lr * ||delta|| (global norm-match).
        let g_norm = norm(&delta);
        assert_close(stats.applied_step_norm, lr as f64 * g_norm);
        // beta1 = 0 -> no history contribution.
        assert_eq!(stats.history_current_norm_ratio, Some(0.0));
        // The relative allocation shifts energy toward the low-second-moment
        // block: block 1 gets a larger share of the step than plain SGD.
        let step0 = (params[0].powi(2) + params[1].powi(2)).sqrt();
        let step1 = (params[2].powi(2) + params[3].powi(2)).sqrt();
        let sgd0 = lr * (delta[0].powi(2) + delta[1].powi(2)).sqrt();
        let sgd1 = lr * (delta[2].powi(2) + delta[3].powi(2)).sqrt();
        assert!(step1 / step0 > sgd1 / sgd0, "block reallocation absent");
    }

    #[test]
    fn block_rms_three_step_deterministic() {
        // Deterministic 3-step trajectory of the per-block v EMA, checked
        // against the closed-form recursion; the applied step norm stays
        // lr*||g|| every step (global norm-match).
        let lr = 0.5f32;
        let numels = [2usize, 1];
        let mut params = [0.0f32; 3];
        let mut buf = [0.0f32; 3];
        let mut block_v = [0.0f32; 2];
        let deltas = [[1.0f32, 1.0, 2.0], [2.0f32, 0.0, 1.0], [0.0f32, 3.0, 4.0]];
        let mut v0 = 0.0f64;
        let mut v1 = 0.0f64;
        for d in &deltas {
            let s0 = (d[0].powi(2) + d[1].powi(2)) as f64 / 2.0;
            let s1 = (d[2].powi(2)) as f64 / 1.0;
            v0 = BLOCK_ADAPTIVE_BETA2 * v0 + (1.0 - BLOCK_ADAPTIVE_BETA2) * s0;
            v1 = BLOCK_ADAPTIVE_BETA2 * v1 + (1.0 - BLOCK_ADAPTIVE_BETA2) * s1;
            let stats = block_second_moment_step(
                &mut params, &mut buf, d, lr, &numels, &mut block_v, false,
            );
            assert_close(block_v[0] as f64, v0);
            assert_close(block_v[1] as f64, v1);
            assert_close(stats.applied_step_norm, lr as f64 * norm(d));
        }
    }

    #[test]
    fn block_yogi_additive_update_and_norm_match() {
        // Yogi additive update: with v_prev < s the sign is negative and v
        // increases by (1 - beta2) * s (same magnitude as RMS on the first
        // step from 0, since v_prev - s < 0). Norm-match holds.
        let lr = 0.28f32;
        let delta = [2.0f32, 0.0, 0.0, 6.0];
        let numels = [2usize, 2];
        let mut params = [0.0f32; 4];
        let mut buf = [0.0f32; 4];
        let mut block_v = [0.0f32; 2];
        let stats =
            block_second_moment_step(&mut params, &mut buf, &delta, lr, &numels, &mut block_v, true);
        // First step from 0: sign(0 - s) = -1, v = (1 - beta2) * s (>= 0).
        assert_close(block_v[0] as f64, 0.05 * 4.0 / 2.0);
        assert_close(block_v[1] as f64, 0.05 * 36.0 / 2.0);
        assert_close(stats.applied_step_norm, lr as f64 * norm(&delta));

        // A second step with a SMALLER magnitude drives v DOWN (v_prev > s ->
        // sign +1, additive decrease), the Yogi robustness property.
        let small = [0.1f32, 0.0, 0.0, 0.1];
        let v1_prev = block_v[1] as f64;
        block_second_moment_step(
            &mut params, &mut buf, &small, lr, &numels, &mut block_v, true,
        );
        assert!((block_v[1] as f64) < v1_prev, "yogi did not decrease v");
    }

    // ---- cheb-sgd -----------------------------------------------------------

    #[test]
    fn cheb_sgd_multipliers_degenerate_isotropic_is_plain_sgd() {
        // kappa = 1 (or <= 1): every multiplier is exactly 1, so a cheb-sgd
        // commit is byte-identical to plain SGD at the base LR.
        assert_eq!(cheb_sgd_multipliers(1.0), [1.0; CHEB_SGD_CYCLE]);
        assert_eq!(cheb_sgd_multipliers(0.5), [1.0; CHEB_SGD_CYCLE]);
    }

    #[test]
    fn cheb_sgd_multipliers_are_ordered_bounded_and_mean_one() {
        // Anisotropic kappa: arithmetic-mean-anchored, hard-bounded, and in the
        // small -> large -> mid -> small Leja order.
        let m = cheb_sgd_multipliers(CHEB_SGD_KAPPA);
        // hard safety bounds
        for &mj in &m {
            assert!(mj >= CHEB_SGD_M_MIN - 1e-12 && mj <= CHEB_SGD_M_MAX + 1e-12,
                    "multiplier {mj} out of [{CHEB_SGD_M_MIN}, {CHEB_SGD_M_MAX}]");
        }
        // cycle-average multiplier is 1 (average LR stays the tuned base)
        let mean = m.iter().sum::<f64>() / CHEB_SGD_CYCLE as f64;
        assert_close(mean, 1.0);
        // ordering (small -> large -> mid-large -> mid-small): phase 1 is the
        // largest step, phase 0 the smallest; the reordered schedule is
        // monotone m[0] <= m[3] <= m[2] <= m[1] (small steps may tie at m_min
        // for large kappa, cf. codex kappa=30 -> [0.5, 2.0, 1.0, 0.5]).
        assert!(m[1] > m[2] && m[2] > m[0], "phase 1 not the largest: {m:?}");
        assert!(m[0] <= m[3] && m[3] <= m[2] && m[2] <= m[1],
                "schedule not monotone in Leja order: {m:?}");
        assert!(m[0] < 1.0 && m[1] > 1.0, "cycle should straddle base LR");
        // deterministic
        assert_eq!(m, cheb_sgd_multipliers(CHEB_SGD_KAPPA));

        // Moderate kappa stays inside the bounds -> strictly ordered, no ties.
        let m4 = cheb_sgd_multipliers(4.0);
        assert_close(m4.iter().sum::<f64>() / CHEB_SGD_CYCLE as f64, 1.0);
        assert!(m4[0] < m4[3] && m4[3] < m4[2] && m4[2] < m4[1],
                "moderate kappa should be strictly small<mid-small<mid-large<large: {m4:?}");
    }

    #[test]
    fn cheb_sgd_cycle_sequence_is_deterministic() {
        // Over CHEB_SGD_CYCLE aligned commits the phase advances 0->1->2->3->0
        // and each commit applies lr * m_k * delta (pure scaled SGD).
        let lr = 0.28f32;
        let delta = [1.0f32, -2.0, 0.5];
        let m = cheb_sgd_multipliers(CHEB_SGD_KAPPA);
        let mut params = [0.0f32; 3];
        let mut buf = [0.0f32; 3];
        let mut phase = 0.0f32;
        for k in 0..CHEB_SGD_CYCLE {
            let before = params;
            let stats = cheb_sgd_step(&mut params, &mut buf, &delta, lr, &mut phase);
            // applied displacement is exactly lr * m_k * delta
            for i in 0..3 {
                let expected = lr * m[k] as f32 * delta[i];
                assert_close((before[i] - params[i]) as f64, expected as f64);
                // buf holds the (unscaled-by-lr) applied direction m_k * delta
                assert_close(buf[i] as f64, (m[k] as f32 * delta[i]) as f64);
            }
            // direction is a positive multiple of delta -> cosine +1, no history
            assert_close(stats.direction_delta_cosine.unwrap(), 1.0);
            assert_eq!(stats.history_current_norm_ratio, Some(0.0));
            assert_eq!(phase, ((k + 1) % CHEB_SGD_CYCLE) as f32);
        }
        assert_eq!(phase, 0.0); // wrapped back to the start of the cycle
    }

    #[test]
    fn cheb_sgd_restart_resets_phase_on_geometry_change() {
        let lr = 0.1f32;
        let delta = [1.0f32, 0.0];
        let mut params = [0.0f32; 2];
        let mut buf = [0.0f32; 2];
        let mut phase = 0.0f32;
        // advance two aligned commits: phase -> 2, buf holds a +x direction.
        cheb_sgd_step(&mut params, &mut buf, &delta, lr, &mut phase);
        cheb_sgd_step(&mut params, &mut buf, &delta, lr, &mut phase);
        assert_eq!(phase, 2.0);
        // an anti-aligned delta (cosine -1 < CHEB_SGD_RESTART_COS) restarts the
        // cycle: phase resets to 0 and the smallest (phase-0) multiplier applies.
        let flip = [-1.0f32, 0.0];
        let m0 = cheb_sgd_multipliers(CHEB_SGD_KAPPA)[0] as f32;
        let before = params;
        let stats = cheb_sgd_step(&mut params, &mut buf, &flip, lr, &mut phase);
        assert!(stats.restarted, "geometry-change restart not flagged");
        assert_close((before[0] - params[0]) as f64, (lr * m0 * flip[0]) as f64);
        assert_eq!(phase, 1.0); // 0 (restart) applied, then advanced to 1
    }

    #[test]
    fn cheb_sgd_step_is_bit_identical_to_materialized_step() {
        // The lr * buf materialize branch must reproduce the applied step
        // bit-for-bit (as for the block/ema optimizers).
        let lr = 0.28f32;
        let delta = [0.7f32, -1.3, 2.1, 0.0, -0.4];
        let mut params = [0.0f32; 5];
        let mut buf = [0.0f32; 5];
        let mut phase = 1.0f32; // mid-cycle, multiplier != 1
        let base = params;
        cheb_sgd_step(&mut params, &mut buf, &delta, lr, &mut phase);
        let materialized =
            materialize_applied_step(OuterOptimizer::ChebSgd, &buf, &delta, lr, 0.9, 1.0);
        for i in 0..5 {
            assert_eq!(base[i] - params[i], materialized[i]);
        }
    }

    /// Golden-trace Lean<->Rust consistency check for the anchor-drift
    /// formalization (lean-mechanism/LeanMechanism/*.lean, generated by
    /// scripts/lean_rust_golden_trace.py). Drives the REAL production
    /// functions `merge_avg` (current-anchor delta = anchor - upload, weighted
    /// MEAN) and `nesterov_step` (b_t = mu b_{t-1} + delta; d_t = delta + mu
    /// b_t; theta -= lr d_t; b_0 = 0) over a 2-dim / 2-worker / 3-commit trace.
    /// Every quantity is a dyadic rational (f32-exact), so assertions are exact
    /// `==` and must match the Lean model and the Python reference bit-for-bit.
    ///
    /// It verifies, against merge.rs reality: delta SIGN (anchor MINUS learner),
    /// mean-not-sum (equal weights c1) and WEIGHTING (1:3 at c2), the exact
    /// Nesterov form + buffer init, and WHERE current-anchor bites: at c3 a
    /// lagging worker B (base = theta_1 while the server is at theta_2) makes
    /// the production current-anchor delta differ from the version-matched
    /// delta by exactly worker B's anchor_drift contribution.
    #[test]
    fn anchor_drift_golden_trace_matches_lean_model() {
        let lr = 0.5f32;
        let mu = 0.5f32;
        // --- current-anchor (PRODUCTION) trajectory ---
        let mut theta = [0.0f32, 0.0];
        let mut buf = [0.0f32, 0.0];

        // Commit 1: anchor = current global [0,0]; A=[1,0], B=[0,1]; equal weights.
        let mut d1 = [0.0f32, 0.0];
        merge_avg(&theta, &[&[1.0, 0.0], &[0.0, 1.0]], &[1.0, 1.0], &mut d1);
        assert_eq!(d1, [-0.5, -0.5]); // mean of (anchor - upload); sign = anchor MINUS learner
        nesterov_step(&mut theta, &mut buf, &d1, lr, mu);
        assert_eq!(buf, [-0.5, -0.5]); // b_1 = mu*0 + delta_1
        assert_eq!(theta, [0.375, 0.375]); // theta_1

        // Commit 2: anchor = theta_1; A=[.5,.5], B=[.25,.5]; UNEQUAL weights 1:3.
        let mut d2 = [0.0f32, 0.0];
        merge_avg(&theta, &[&[0.5, 0.5], &[0.25, 0.5]], &[1.0, 3.0], &mut d2);
        assert_eq!(d2, [0.0625, -0.125]); // 1/4*[-1/8,-1/8] + 3/4*[1/8,-1/8]
        nesterov_step(&mut theta, &mut buf, &d2, lr, mu);
        assert_eq!(buf, [-0.1875, -0.375]); // b_2 = 0.5*[-0.5,-0.5] + d2
        assert_eq!(theta, [0.390625, 0.53125]); // theta_2 = [25/64, 17/32]

        // Commit 3: anchor = theta_2; A=[.5,.5], B=[.5,.5]; equal weights.
        // Worker B LAGS: it trained from theta_1, but current-anchor differences
        // it against theta_2 anyway -> this is where current-anchor bites.
        let theta1 = [0.375f32, 0.375];
        let theta2 = theta; // [0.390625, 0.53125]
        let mut d3_current = [0.0f32, 0.0];
        merge_avg(&theta2, &[&[0.5, 0.5], &[0.5, 0.5]], &[1.0, 1.0], &mut d3_current);
        assert_eq!(d3_current, [-0.109375, 0.03125]); // = theta_2 - [0.5,0.5]

        // Version-matched control (NO such flag exists in the Rust syncer today,
        // git log has no --version-matched-anchor; computed here by hand):
        // each worker's delta vs its OWN declared base. Worker A base = theta_2,
        // worker B base = theta_1.
        let d3_vmatched = [
            0.5 * (theta2[0] - 0.5) + 0.5 * (theta1[0] - 0.5),
            0.5 * (theta2[1] - 0.5) + 0.5 * (theta1[1] - 0.5),
        ];
        assert_eq!(d3_vmatched, [-0.1171875, -0.046875]); // [-15/128, -3/64]

        // The divergence is exactly worker B's anchor_drift contribution.
        // anchor_drift_B = current(theta_2) - base(theta_1); its weighted share
        // (weight 1/2) is what separates current-anchor from version-matched.
        let anchor_drift_b = [theta2[0] - theta1[0], theta2[1] - theta1[1]];
        assert_eq!(anchor_drift_b, [0.015625, 0.15625]); // [1/64, 5/32]
        // current - vmatched == 0.5 * anchor_drift_B  (worker B's merge weight)
        assert_eq!(d3_current[0] - d3_vmatched[0], 0.5 * anchor_drift_b[0]);
        assert_eq!(d3_current[1] - d3_vmatched[1], 0.5 * anchor_drift_b[1]);

        // Complete the production (current-anchor) trajectory.
        nesterov_step(&mut theta, &mut buf, &d3_current, lr, mu);
        assert_eq!(buf, [-0.203125, -0.15625]);
        assert_eq!(theta, [0.49609375, 0.5546875]); // theta_3 = [127/256, 71/128]
    }
}
