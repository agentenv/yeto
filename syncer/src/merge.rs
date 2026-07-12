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
}

impl OuterOptimizer {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Nesterov => "nesterov",
            Self::NormalizedEma => "normalized-ema",
            Self::RestartedEma => "restarted-ema",
            Self::RhoAdaptive => "rho-adaptive",
            Self::CappedNesterov => "capped-nesterov",
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
            other => Err(format!(
                "outer optimizer must be one of nesterov, normalized-ema, restarted-ema, rho-adaptive, capped-nesterov; got {other:?}"
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
/// `rho_adaptive_step`) and `capped_mu` is the capped-Nesterov controller's
/// persistent effective momentum (see `capped_nesterov_step`); the other
/// optimizers leave them untouched, exactly as they leave each other's
/// buffer conventions alone.
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
    }
}

/// Materialize the nominal f32 parameter displacement produced by an outer
/// step after its optimizer buffer has been updated. This is the same vector
/// whose norm is reported by `OuterStepStats::applied_step_norm`.
///
/// For `CappedNesterov` the caller must pass the EFFECTIVE per-commit
/// momentum written back by `capped_nesterov_step` (the updated persistent
/// scalar), not the CLI momentum; with that value the Nesterov branch is
/// bit-identical to the applied step.
pub fn materialize_applied_step(
    optimizer: OuterOptimizer,
    updated_buf: &[f32],
    delta: &[f32],
    lr: f32,
    momentum: f32,
) -> Vec<f32> {
    debug_assert_eq!(updated_buf.len(), delta.len());
    match optimizer {
        OuterOptimizer::Nesterov | OuterOptimizer::CappedNesterov => updated_buf
            .iter()
            .zip(delta)
            .map(|(buf, value)| lr * (*value + momentum * *buf))
            .collect(),
        OuterOptimizer::NormalizedEma
        | OuterOptimizer::RestartedEma
        | OuterOptimizer::RhoAdaptive => {
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

/// Per-commit momentum cap from the realized buffer/delta geometry
/// (c_t = <b_{t-1}, delta_t> / |delta_t|^2, r_t = |b_{t-1} - c_t delta_t| /
/// |delta_t|), before the release EMA:
///
///   mu_par : largest mu in [0, mu_max] with mu + mu^2 * [c_t]_+ <= mu_max,
///            i.e. the positive root of [c_t]_+ mu^2 + mu - mu_max = 0,
///            mu_par = (sqrt(1 + 4 [c_t]_+ mu_max) - 1) / (2 [c_t]_+)
///            (mu_max itself when [c_t]_+ = 0) — caps the aligned gain
///            A_t = 1 + mu + mu^2 c_t at 1 + mu_max for amplifying history;
///   mu_perp: sqrt(tau_perp / max(r_t, eps)) — caps the transverse
///            contribution mu^2 r_t |delta| at tau_perp |delta|;
///   cap    = min(mu_max, mu_par, mu_perp).
///
/// Sign-reversal guard: if A_t(cap) < 0 (possible only for strongly negative
/// c_t, where A_t is a downward parabola in mu with A_t(0) = 1), the cap is
/// zeroed — since A_t(0) = 1 > 0 and A_t(cap) >= 0 imply A_t >= 0 on all of
/// [0, cap], the guard keeps every admissible mu on the descent side.
pub fn capped_nesterov_cap(c_t: f64, r_t: f64) -> f64 {
    let c_plus = c_t.max(0.0);
    let mu_par = if c_plus > 0.0 {
        ((1.0 + 4.0 * c_plus * CAPPED_NESTEROV_MU_MAX).sqrt() - 1.0) / (2.0 * c_plus)
    } else {
        CAPPED_NESTEROV_MU_MAX
    };
    let mu_perp = (CAPPED_NESTEROV_TAU_PERP / r_t.max(CAPPED_NESTEROV_R_EPS)).sqrt();
    let cap = CAPPED_NESTEROV_MU_MAX.min(mu_par).min(mu_perp);
    if 1.0 + cap + cap * cap * c_t < 0.0 {
        0.0
    } else {
        cap
    }
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
        // |b - c*delta|^2 expands to |b|^2 - c*<b, delta> at the projection
        // coefficient; clamp at 0 against cancellation when b ~ c*delta.
        let r = ((buf_norm_sq - c * dot).max(0.0) / delta_norm_sq).sqrt();
        (c, r)
    } else {
        // No measurable geometry (zero delta): caps stay inactive and the
        // EMA relaxes toward mu_max, mirroring rho-adaptive's unmeasured
        // commits.
        (0.0, 0.0)
    };
    let cap = capped_nesterov_cap(c_t, r_t);
    let released =
        CAPPED_NESTEROV_EMA_BETA * *mu_prev as f64 + (1.0 - CAPPED_NESTEROV_EMA_BETA) * cap;
    *mu_prev = cap.min(released) as f32;
    nesterov_step(params, buf, delta, lr, *mu_prev)
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
            materialize_applied_step(OuterOptimizer::CappedNesterov, &buf, &delta, 0.3, mu);
        for ((b, s), after) in base.iter().zip(&step).zip(&p) {
            assert_eq!((b - s).to_bits(), after.to_bits());
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
        assert!("ema".parse::<OuterOptimizer>().is_err());
    }
}
