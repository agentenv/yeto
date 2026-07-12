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
}

impl OuterOptimizer {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Nesterov => "nesterov",
            Self::NormalizedEma => "normalized-ema",
            Self::RestartedEma => "restarted-ema",
            Self::RhoAdaptive => "rho-adaptive",
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
            other => Err(format!(
                "outer optimizer must be one of nesterov, normalized-ema, restarted-ema, rho-adaptive; got {other:?}"
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
/// previews made from cloned parameter and buffer slices.
pub fn apply_outer_step(
    optimizer: OuterOptimizer,
    params: &mut [f32],
    buf: &mut [f32],
    delta: &[f32],
    lr: f32,
    momentum: f32,
    restart_cos_threshold: f32,
) -> OuterStepStats {
    match optimizer {
        OuterOptimizer::Nesterov => nesterov_step(params, buf, delta, lr, momentum),
        OuterOptimizer::NormalizedEma => normalized_ema_step(params, buf, delta, lr, momentum),
        OuterOptimizer::RestartedEma => {
            restarted_ema_step(params, buf, delta, lr, momentum, restart_cos_threshold)
        }
        OuterOptimizer::RhoAdaptive => rho_adaptive_step(params, buf, delta, lr, momentum),
    }
}

/// Materialize the nominal f32 parameter displacement produced by an outer
/// step after its optimizer buffer has been updated. This is the same vector
/// whose norm is reported by `OuterStepStats::applied_step_norm`.
pub fn materialize_applied_step(
    optimizer: OuterOptimizer,
    updated_buf: &[f32],
    delta: &[f32],
    lr: f32,
    momentum: f32,
) -> Vec<f32> {
    debug_assert_eq!(updated_buf.len(), delta.len());
    match optimizer {
        OuterOptimizer::Nesterov => updated_buf
            .iter()
            .zip(delta)
            .map(|(buf, value)| lr * (*value + momentum * *buf))
            .collect(),
        OuterOptimizer::NormalizedEma | OuterOptimizer::RestartedEma => {
            updated_buf.iter().map(|buf| lr * *buf).collect()
        }
        // The amplification scale depends on the pre-update buffer (the
        // previous merged delta), which is gone after the step. Action-probe
        // previews are unsupported for rho-adaptive; the unscaled SGD step
        // is returned so callers see a well-defined vector.
        OuterOptimizer::RhoAdaptive => delta.iter().map(|value| lr * *value).collect(),
    }
}

/// Rho-adaptive memoryless step. `buf` stores the PREVIOUS merged delta,
/// not momentum. Each commit measures the round-to-round direction
/// autocorrelation rho = cos(delta, buf) and applies the outer step that a
/// Nesterov buffer with mu_eff = clamp(2*(1 - rho), 0, mu_max) would have
/// produced in steady state on a rho-correlated direction:
///
///   theta <- theta - lr / (1 - mu_eff * rho) * delta
///
/// High persistence (rho -> 1) yields mu_eff -> 0 and recovers plain SGD;
/// decorrelated rounds earn a bounded step-scale boost. `mu_max` arrives via
/// the `--outer-momentum` argument. `OuterStepStats` reuse for diagnostics:
/// `history_current_norm_ratio` reports RHO (not a norm ratio) for this
/// optimizer.
pub fn rho_adaptive_step(
    params: &mut [f32],
    buf: &mut [f32],
    delta: &[f32],
    lr: f32,
    mu_max: f32,
) -> OuterStepStats {
    let mut dot = 0.0f64;
    let mut buf_norm_sq = 0.0f64;
    let mut delta_norm_sq = 0.0f64;
    for (b, d) in buf.iter().zip(delta) {
        dot += *b as f64 * *d as f64;
        buf_norm_sq += (*b as f64).powi(2);
        delta_norm_sq += (*d as f64).powi(2);
    }
    let rho = if buf_norm_sq > 0.0 && delta_norm_sq > 0.0 {
        (dot / (buf_norm_sq.sqrt() * delta_norm_sq.sqrt())).clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let mu_eff = (2.0 * (1.0 - rho)).clamp(0.0, mu_max as f64);
    // Denominator floor bounds the amplification at 4x even if mu_max * rho
    // approaches 1; negative rho dampens (denominator > 1).
    let scale = (1.0 / (1.0 - mu_eff * rho).max(0.25)) as f32;
    let mut step_norm_sq = 0.0f64;
    for ((p, b), d) in params.iter_mut().zip(buf.iter_mut()).zip(delta) {
        let step = lr * scale * *d;
        *p -= step;
        *b = *d;
        step_norm_sq += (step as f64).powi(2);
    }
    OuterStepStats {
        applied_step_norm: step_norm_sq.sqrt(),
        direction_delta_cosine: Some(1.0),
        history_current_norm_ratio: Some(rho),
        restarted: false,
    }
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
        let stats = rho_adaptive_step(&mut params, &mut buf, &[0.5, -0.5], 0.1, 0.9);
        // zero buffer -> rho treated as 0 -> scale exactly 1
        assert!((params[0] - (1.0 - 0.05)).abs() < 1e-7);
        assert!((params[1] - (1.0 + 0.05)).abs() < 1e-7);
        assert_eq!(buf, vec![0.5, -0.5]);
        assert_eq!(stats.history_current_norm_ratio, Some(0.0));
    }

    #[test]
    fn rho_adaptive_persistent_direction_recovers_sgd_and_reports_rho() {
        let mut params = vec![0.0, 0.0];
        let mut buf = vec![1.0, 0.0];
        // identical direction: rho = 1 -> mu_eff = 0 -> plain SGD step
        let stats = rho_adaptive_step(&mut params, &mut buf, &[1.0, 0.0], 0.1, 0.9);
        assert!((params[0] + 0.1).abs() < 1e-7);
        assert_eq!(stats.history_current_norm_ratio, Some(1.0));
    }

    #[test]
    fn rho_adaptive_anticorrelated_direction_dampens() {
        let mut params = vec![0.0];
        let mut buf = vec![1.0];
        // rho = -1 -> mu_eff clamps to mu_max -> scale = 1/(1+mu_max) < 1
        let stats = rho_adaptive_step(&mut params, &mut buf, &[-1.0], 0.1, 0.9);
        let expected = 0.1 / (1.0 + 0.9);
        assert!((params[0] - expected).abs() < 1e-6);
        assert_eq!(stats.history_current_norm_ratio, Some(-1.0));
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
        assert!("ema".parse::<OuterOptimizer>().is_err());
    }
}
