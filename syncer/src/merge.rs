//! Merge math for asynchronous multi-learner training.
//!
//! Per-learner outer gradient for fragment p: Δ_m = anchor_m,p − θ_m,p.
//! Learners transmit the opposite signed, base-relative update
//! `θ_m,p − anchor_m,p`; the protocol decoder negates it exactly once before
//! this module sees it. Consequently every function here consumes outer
//! gradients directly and never needs a parameter anchor.
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
//! flattens the singular-value spectrum of the averaged matrix to its mean.
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

fn l2_norm(delta: &[f32]) -> f64 {
    delta
        .iter()
        .map(|d| (*d as f64) * (*d as f64))
        .sum::<f64>()
        .sqrt()
}

/// Weighted direct averaging: out[i] = Σ_m w_m Δ_m[i] / Σ_m w_m.
pub fn merge_avg(deltas: &[&[f32]], weights: &[f64], out: &mut [f32]) {
    let wsum: f64 = weights.iter().sum();
    if wsum <= 0.0 {
        out.fill(0.0);
        return;
    }
    out.fill(0.0);
    for (delta, &w) in deltas.iter().zip(weights) {
        let w = (w / wsum) as f32;
        for (o, d) in out.iter_mut().zip(*delta) {
            *o += w * *d;
        }
    }
}

/// Weighted radial-directional averaging over one tensor slice.
pub fn merge_rda(deltas: &[&[f32]], weights: &[f64], out: &mut [f32]) {
    let wsum: f64 = weights.iter().sum();
    if wsum <= 0.0 {
        out.fill(0.0);
        return;
    }
    let norms: Vec<f64> = deltas.iter().map(|d| l2_norm(d)).collect();
    let radial: f64 = norms.iter().zip(weights).map(|(n, w)| n * w).sum::<f64>() / wsum;

    // Weighted mean of unit directions, φ(0) := 0.
    out.fill(0.0);
    for ((delta, &w), &n) in deltas.iter().zip(weights).zip(&norms) {
        if n == 0.0 {
            continue;
        }
        let coef = (w / wsum / n) as f32;
        for (o, d) in out.iter_mut().zip(*delta) {
            *o += coef * *d;
        }
    }
    let mean_dir_norm = out
        .iter()
        .map(|v| (*v as f64) * (*v as f64))
        .sum::<f64>()
        .sqrt();
    if mean_dir_norm < 1e-12 {
        // Degenerate (all-zero or cancelling directions): fall back to Avg.
        merge_avg(deltas, weights, out);
        return;
    }
    let scale = (radial / mean_dir_norm) as f32;
    for o in out.iter_mut() {
        *o *= scale;
    }
}

// Newton-Schulz polar iteration bounds. 40 iterations mean a singular value
// below ~1.5^-40 of the Frobenius norm never flattens up to sigma_bar, so
// the numerical rank cannot increase; the tolerance is the early-exit floor
// once the retained spectrum has converged (internal math is f64).
const ISO_NS_MAX_ITERS: usize = 40;
const ISO_NS_REL_TOL: f64 = 1e-9;

/// Iso-C-style isotropic aggregation over one tensor slice (IsoLoCo,
/// arXiv 2607.03011, Alg. 2). Weighted direct average is applied first;
/// then the averaged pseudo-gradient, viewed as a row-major `rows` x `cols`
/// matrix, has its singular-value spectrum flattened to the mean singular
/// value. Shape/product mismatches leave the direct average untouched.
pub fn merge_iso(deltas: &[&[f32]], weights: &[f64], rows: usize, cols: usize, out: &mut [f32]) {
    merge_avg(deltas, weights, out);
    if rows == 0 || cols == 0 || rows.saturating_mul(cols) != out.len() {
        return;
    }
    iso_flatten_spectrum(out, rows, cols);
}

/// Replace the `rows` x `cols` row-major matrix `m` in place by sigma_bar*U*V^T.
///
/// sigma_bar*U*V^T is the polar factor of the matrix scaled by the mean
/// singular value, computed without an SVD: after Frobenius normalization
/// every singular value lies in (0, 1], where the cubic Newton-Schulz map
/// X <- 1.5 X - 0.5 (X X^T) X converges monotonically to 1 and keeps exact
/// zeros at zero. sigma_bar = trace(Q^T A) / k because trace(Q^T A) is the
/// nuclear norm. This scalar path is the small-model/test fallback; the
/// torch-svd backend runs the same iteration on a GPU.
pub fn iso_flatten_spectrum(m: &mut [f32], rows: usize, cols: usize) {
    debug_assert_eq!(rows * cols, m.len());
    let k = rows.min(cols);
    if k == 0 {
        return;
    }
    // Iterate on the thin side so the Gram product is k x k: `a` holds the
    // input as a k x n row-major matrix, transposed when rows > cols.
    let n = rows.max(cols);
    let a: Vec<f64> = if rows <= cols {
        m.iter().map(|v| *v as f64).collect()
    } else {
        let mut t = vec![0.0f64; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                t[c * rows + r] = m[r * cols + c] as f64;
            }
        }
        t
    };
    let norm = a.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm == 0.0 {
        return;
    }
    let mut x: Vec<f64> = a.iter().map(|v| v / norm).collect();
    let mut gram = vec![0.0f64; k * k];
    let mut nxt = vec![0.0f64; k * n];
    for _ in 0..ISO_NS_MAX_ITERS {
        for i in 0..k {
            for j in i..k {
                let acc: f64 = (0..n).map(|c| x[i * n + c] * x[j * n + c]).sum();
                gram[i * k + j] = acc;
                gram[j * k + i] = acc;
            }
        }
        for r in 0..k {
            for c in 0..n {
                let acc: f64 = (0..k).map(|t| gram[r * k + t] * x[t * n + c]).sum();
                nxt[r * n + c] = 1.5 * x[r * n + c] - 0.5 * acc;
            }
        }
        let (mut diff_sq, mut next_sq) = (0.0f64, 0.0f64);
        for (nx, xv) in nxt.iter().zip(x.iter_mut()) {
            let d = nx - *xv;
            diff_sq += d * d;
            next_sq += nx * nx;
            *xv = *nx;
        }
        if diff_sq.sqrt() <= ISO_NS_REL_TOL * next_sq.sqrt() {
            break;
        }
    }
    let sigma_bar = x.iter().zip(&a).map(|(q, v)| q * v).sum::<f64>() / k as f64;
    if rows <= cols {
        for (mv, q) in m.iter_mut().zip(&x) {
            *mv = (sigma_bar * q) as f32;
        }
    } else {
        for r in 0..rows {
            for c in 0..cols {
                m[r * cols + c] = (sigma_bar * x[c * rows + r]) as f32;
            }
        }
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
    fn avg_preserves_signed_outer_gradient() {
        let d0 = [1.0f32, 0.0];
        let d1 = [0.0f32, -2.0];
        let mut out = [0.0f32; 2];
        merge_avg(&[&d0, &d1], &[1.0, 1.0], &mut out);
        assert_eq!(out, [0.5, -1.0]);
    }

    #[test]
    fn avg_respects_weights() {
        let d0 = [1.0f32];
        let d1 = [-3.0f32];
        let mut out = [0.0f32; 1];
        merge_avg(&[&d0, &d1], &[3.0, 1.0], &mut out);
        assert!((out[0] - 0.0).abs() < 1e-6); // (3*1 + 1*(-3))/4
    }

    #[test]
    fn rda_preserves_norm_of_orthogonal_inputs() {
        // Two orthogonal deltas of norm 2: direct avg gives norm 2/√2 ≈ 1.41,
        // RDA must give 2.
        let d0 = [2.0f32, 0.0];
        let d1 = [0.0f32, 2.0];
        let mut out = [0.0f32; 2];
        merge_rda(&[&d0, &d1], &[1.0, 1.0], &mut out);
        assert!((norm(&out) - 2.0).abs() < 1e-5, "norm was {}", norm(&out));
        // Direction is the diagonal.
        assert!((out[0] - out[1]).abs() < 1e-6 && out[0] > 0.0);
    }

    #[test]
    fn rda_single_learner_is_identity_delta() {
        let d0 = [0.5f32, -0.5, 0.0];
        let mut out = [0.0f32; 3];
        merge_rda(&[&d0], &[7.0], &mut out);
        for (o, e) in out.iter().zip([0.5f32, -0.5, 0.0]) {
            assert!((o - e).abs() < 1e-6);
        }
    }

    #[test]
    fn rda_zero_deltas_give_zero() {
        let d0 = [0.0f32, 0.0];
        let mut out = [9.0f32; 2];
        merge_rda(&[&d0[..]], &[1.0], &mut out);
        assert_eq!(out, [0.0, 0.0]);
    }

    #[test]
    fn rda_cancelling_directions_fall_back_to_avg() {
        let d0 = [1.0f32];
        let d1 = [-1.0f32];
        let mut out = [0.0f32; 1];
        merge_rda(&[&d0, &d1], &[1.0, 1.0], &mut out);
        assert!(out[0].abs() < 1e-6);
    }

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
        let delta = [3.0f32, 0.0, 0.0, 1.0];
        let mut out = [0.0f32; 4];
        merge_iso(&[&delta], &[1.0], 2, 2, &mut out);
        for (o, e) in out.iter().zip([2.0f32, 0.0, 0.0, 2.0]) {
            assert!((o - e).abs() < 1e-6, "got {out:?}");
        }
    }

    #[test]
    fn iso_matches_sigma_bar_u_vt_on_rectangular_matrices() {
        let delta = [1.0f32, 0.0, 0.0, 0.0, 2.0, 0.0];
        let mut out = [0.0f32; 6];
        merge_iso(&[&delta], &[2.5], 2, 3, &mut out);
        for (o, e) in out.iter().zip([1.5f32, 0.0, 0.0, 0.0, 1.5, 0.0]) {
            assert!((o - e).abs() < 1e-6, "got {out:?}");
        }
        let delta_t = [1.0f32, 0.0, 0.0, 2.0, 0.0, 0.0];
        let mut out_t = [0.0f32; 6];
        merge_iso(&[&delta_t], &[1.0], 3, 2, &mut out_t);
        for (o, e) in out_t.iter().zip([1.5f32, 0.0, 0.0, 1.5, 0.0, 0.0]) {
            assert!((o - e).abs() < 1e-6, "got {out_t:?}");
        }
    }

    #[test]
    fn iso_output_spectrum_is_isotropic_and_aligned() {
        let delta = [
            2.0f32, -1.0, 0.5, 3.0, 0.25, 4.0, -2.0, 1.0, -1.5, 0.75, 3.5, -0.5,
        ];
        let mut out = [0.0f32; 12];
        merge_iso(&[&delta[..]], &[1.0], 3, 4, &mut out);
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
        let mut again = out;
        iso_flatten_spectrum(&mut again, 3, 4);
        for (a, o) in again.iter().zip(out) {
            assert!((a - o).abs() < 1e-5 * sigma_bar_sq.sqrt() as f32);
        }
    }

    #[test]
    fn iso_weighted_average_feeds_the_transform() {
        let d0 = [4.0f32, 0.0, 0.0, 0.0];
        let d1 = [0.0f32, 0.0, 0.0, 4.0];
        let mut out = [0.0f32; 4];
        merge_iso(&[&d0, &d1], &[3.0, 1.0], 2, 2, &mut out);
        for (o, e) in out.iter().zip([2.0f32, 0.0, 0.0, 2.0]) {
            assert!((o - e).abs() < 1e-6, "got {out:?}");
        }
    }

    #[test]
    fn iso_vector_and_zero_deltas_are_stable() {
        let delta = [3.0f32, 4.0];
        let mut out = [0.0f32; 2];
        merge_iso(&[&delta], &[1.0], 1, 2, &mut out);
        assert!((out[0] - 3.0).abs() < 1e-6 && (out[1] - 4.0).abs() < 1e-6);
        let zero_delta = [0.0f32, 0.0];
        merge_iso(&[&zero_delta], &[1.0], 2, 1, &mut out);
        assert_eq!(out, [0.0, 0.0]);
        merge_iso(&[&delta], &[1.0], 3, 5, &mut out);
        assert_eq!(out, [3.0, 4.0]);
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
