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
//! Outer optimizer: SGD with Nesterov momentum, state held here on the
//! syncer, with defaults lr=0.7 and μ=0.9.

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

fn norm(values: &[f32]) -> f64 {
    values
        .iter()
        .map(|v| (*v as f64) * (*v as f64))
        .sum::<f64>()
        .sqrt()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConsensusScale {
    Sqrt,
    Linear,
    Affine50,
    Floor50,
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
    let radial: f64 = norms.iter().zip(weights).map(|(n, w)| n * w).sum::<f64>() / wsum;

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
    let mean_dir_norm = out
        .iter()
        .map(|v| (*v as f64) * (*v as f64))
        .sum::<f64>()
        .sqrt();
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

/// RDA with a consensus-dependent radial scale. Consensus is the norm of the
/// weighted mean unit direction; low agreement damps that tensor's update.
pub fn merge_consensus_rda(
    anchor: &[f32],
    learners: &[&[f32]],
    weights: &[f64],
    scale_mode: ConsensusScale,
    out: &mut [f32],
) -> f64 {
    let wsum: f64 = weights.iter().sum();
    if wsum <= 0.0 {
        out.fill(0.0);
        return 0.0;
    }
    let norms: Vec<f64> = learners.iter().map(|l| l2_norm(anchor, l)).collect();
    let radial: f64 = norms.iter().zip(weights).map(|(n, w)| n * w).sum::<f64>() / wsum;

    out.fill(0.0);
    for ((learner, &w), &n) in learners.iter().zip(weights).zip(&norms) {
        if n <= 1e-12 {
            continue;
        }
        let coef = (w / wsum / n) as f32;
        for ((o, a), l) in out.iter_mut().zip(anchor).zip(*learner) {
            *o += coef * (*a - *l);
        }
    }
    let consensus = norm(out).clamp(0.0, 1.0);
    if consensus < 1e-12 {
        merge_avg(anchor, learners, weights, out);
        return 0.0;
    }
    let scale = match scale_mode {
        ConsensusScale::Sqrt => consensus.sqrt(),
        ConsensusScale::Linear => consensus,
        ConsensusScale::Affine50 => 0.5 + 0.5 * consensus,
        ConsensusScale::Floor50 => consensus.max(0.5),
    };
    let factor = (radial * scale / consensus) as f32;
    for o in out.iter_mut() {
        *o *= factor;
    }
    scale
}

/// Coordinate-wise midpoint/median over learner deltas, with the resulting
/// robust direction rescaled to the production merge norm for this tensor.
pub fn merge_coord_midpoint_normmatch(
    anchor: &[f32],
    learners: &[&[f32]],
    target_delta: &[f32],
    out: &mut [f32],
) {
    if learners.is_empty() {
        out.fill(0.0);
        return;
    }
    debug_assert_eq!(anchor.len(), target_delta.len());
    debug_assert_eq!(anchor.len(), out.len());
    let mut coord = Vec::with_capacity(learners.len());
    for i in 0..anchor.len() {
        coord.clear();
        for learner in learners {
            coord.push(anchor[i] - learner[i]);
        }
        coord.sort_by(|a, b| a.total_cmp(b));
        let mid = coord.len() / 2;
        out[i] = if coord.len() % 2 == 0 {
            0.5 * (coord[mid - 1] + coord[mid])
        } else {
            coord[mid]
        };
    }
    let source_norm = norm(out);
    if source_norm < 1e-12 {
        return;
    }
    let target_norm = norm(target_delta);
    let scale = (target_norm / source_norm) as f32;
    for v in out.iter_mut() {
        *v *= scale;
    }
}

/// SGD + Nesterov momentum treating `delta` as the gradient:
/// buf ← μ·buf + Δ;  θ ← θ − lr·(Δ + μ·buf).
pub fn nesterov_step(params: &mut [f32], buf: &mut [f32], delta: &[f32], lr: f32, mu: f32) {
    for ((p, b), d) in params.iter_mut().zip(buf.iter_mut()).zip(delta) {
        *b = mu * *b + *d;
        *p -= lr * (*d + mu * *b);
    }
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
        Self {
            c_ok: 0.2,
            k_s: 0.5,
            k_d: 1.0,
            beta_max: 0.5,
            kappa: 3.0,
            eps: 1e-8,
        }
    }
}

pub fn heloco_correct(delta: &mut [f32], momentum: &[f32], h: &Heloco) {
    debug_assert_eq!(delta.len(), momentum.len());
    let du = delta
        .iter()
        .map(|v| (*v as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let dm = momentum
        .iter()
        .map(|v| (*v as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    if du < h.eps || dm < h.eps {
        return;
    }
    let dot: f64 = delta
        .iter()
        .zip(momentum)
        .map(|(d, m)| *d as f64 * *m as f64)
        .sum();
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

    #[test]
    fn weight_formula() {
        assert_eq!(learner_weight(100, 10), 1000.0);
        assert_eq!(learner_weight(0, 10), 0.0);
        assert_eq!(learner_weight(100, 0), 0.0);
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

    #[test]
    fn consensus_rda_floor50_damps_low_agreement() {
        let anchor = [0.0f32, 0.0];
        let l0 = [-2.0f32, 0.0]; // delta (2, 0)
        let l1 = [0.0f32, -2.0]; // delta (0, 2)
        let mut rda = [0.0f32; 2];
        let mut out = [0.0f32; 2];

        merge_rda(&anchor, &[&l0, &l1], &[1.0, 1.0], &mut rda);
        let scale = merge_consensus_rda(
            &anchor,
            &[&l0, &l1],
            &[1.0, 1.0],
            ConsensusScale::Floor50,
            &mut out,
        );

        assert!(scale > 0.5 && scale < 1.0);
        assert!((norm(&out) - norm(&rda) * scale).abs() < 1e-5);
        assert!((out[0] - out[1]).abs() < 1e-6);
    }

    #[test]
    fn consensus_rda_affine50_is_gentler_than_linear() {
        let anchor = [0.0f32, 0.0];
        let l0 = [-2.0f32, 0.0];
        let l1 = [0.0f32, -2.0];
        let mut linear = [0.0f32; 2];
        let mut affine = [0.0f32; 2];

        let linear_scale = merge_consensus_rda(
            &anchor,
            &[&l0, &l1],
            &[1.0, 1.0],
            ConsensusScale::Linear,
            &mut linear,
        );
        let affine_scale = merge_consensus_rda(
            &anchor,
            &[&l0, &l1],
            &[1.0, 1.0],
            ConsensusScale::Affine50,
            &mut affine,
        );

        assert!(affine_scale > linear_scale);
        assert!(norm(&affine) > norm(&linear));
    }

    #[test]
    fn coord_midpoint_normmatch_uses_robust_direction_and_target_norm() {
        let anchor = [0.0f32, 0.0];
        let l0 = [-10.0f32, 0.0]; // delta (10, 0)
        let l1 = [0.0f32, -10.0]; // delta (0, 10)
        let l2 = [-1.0f32, -1.0]; // median delta (1, 1)
        let target = [3.0f32, 4.0]; // norm 5
        let mut out = [0.0f32; 2];

        merge_coord_midpoint_normmatch(&anchor, &[&l0, &l1, &l2], &target, &mut out);

        assert!((norm(&out) - 5.0).abs() < 1e-5);
        assert!((out[0] - out[1]).abs() < 1e-6);
        assert!(out[0] > 0.0);
    }

    #[test]
    fn coord_midpoint_normmatch_midpoints_even_groups() {
        let anchor = [0.0f32];
        let l0 = [-2.0f32]; // delta 2
        let l1 = [0.0f32]; // delta 0
        let target = [3.0f32];
        let mut out = [0.0f32; 1];

        merge_coord_midpoint_normmatch(&anchor, &[&l0, &l1], &target, &mut out);

        assert!((out[0] - 3.0).abs() < 1e-6);
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
        assert!(
            (norm(&d) - mag).abs() < 1e-5,
            "magnitude changed: {mag} -> {}",
            norm(&d)
        );
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
        let moved_small: f64 = small_m
            .iter()
            .zip(&orig)
            .map(|(a, b)| (a - b).abs() as f64)
            .sum();
        let moved_large: f64 = large_m
            .iter()
            .zip(&orig)
            .map(|(a, b)| (a - b).abs() as f64)
            .sum();
        assert!(moved_large < moved_small);
    }

    #[test]
    fn nesterov_matches_reference() {
        // One step from zero state: buf = Δ; θ -= lr(Δ + μΔ) = lr(1+μ)Δ.
        let mut p = [1.0f32];
        let mut buf = [0.0f32];
        nesterov_step(&mut p, &mut buf, &[0.5], 0.7, 0.9);
        assert!((p[0] - (1.0 - 0.7 * (0.5 + 0.9 * 0.5))).abs() < 1e-6);
        assert!((buf[0] - 0.5).abs() < 1e-6);
    }
}
