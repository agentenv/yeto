//! Merge math from "Decoupled DiLoCo" (arXiv 2604.21428).
//!
//! Per-learner outer gradient for fragment p: Δ_m = Θ_p(prev) − θ_m,p, i.e.
//! anchored at the syncer's own previous global fragment (Appendix D.2).
//! Learner weights w_m = c_tokens · (c_tokens / c_steps) — quantity × quality
//! (Section 3.2). Merging is either weighted direct averaging (embedding
//! fragment) or weighted radial-directional averaging, RDA (everything else):
//!
//!   RDA({v_m, w_m}) = (Σ w_m ‖v_m‖ / Σ w_m) · φ(Σ w_m φ(v_m) / Σ w_m)
//!
//! with φ(x) = x/‖x‖ and φ(0) := 0. (The paper prints the radial factor as
//! `Σ w v / Σ w`, a vector — dimensionally inconsistent; the weighted mean of
//! norms is the intended reading.) RDA keeps the merged norm invariant to the
//! number of learners: near-orthogonal same-norm deltas would otherwise
//! shrink as R/√M and force outer-lr retuning. Applied per tensor within a
//! fragment. Degenerate mean direction falls back to direct averaging.
//!
//! Outer optimizer: SGD with Nesterov momentum, state held here on the
//! syncer. Values lr=0.7, μ=0.9 follow the DiLoCo lineage defaults (the
//! decoupled paper publishes no numbers).

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
pub fn nesterov_step(params: &mut [f32], buf: &mut [f32], delta: &[f32], lr: f32, mu: f32) {
    for ((p, b), d) in params.iter_mut().zip(buf.iter_mut()).zip(delta) {
        *b = mu * *b + *d;
        *p -= lr * (*d + mu * *b);
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
    fn nesterov_matches_reference() {
        // One step from zero state: buf = Δ; θ -= lr(Δ + μΔ) = lr(1+μ)Δ.
        let mut p = [1.0f32];
        let mut buf = [0.0f32];
        nesterov_step(&mut p, &mut buf, &[0.5], 0.7, 0.9);
        assert!((p[0] - (1.0 - 0.7 * (0.5 + 0.9 * 0.5))).abs() < 1e-6);
        assert!((buf[0] - 0.5).abs() < 1e-6);
    }
}
