//! Streaming reduction of one fragment's learner pushes (S27 P4, P4b).
//!
//! The buffered merge (`GlobalState::prepare_merge`) keeps every learner's
//! decoded f32 outer gradient for a fragment until the round completes, so
//! the head holds M fragment-sized copies per in-flight round.  A
//! `FragmentFold` instead folds each push into running accumulators as soon
//! as it arrives and drops the push buffer, so a round holds one accumulator
//! (two for RDA) plus only the pushes that arrived ahead of their turn.
//!
//! Bit identity with the buffered merge is a hard requirement.  f32 addition
//! is not associative, so the fold reproduces the exact operation sequence of
//! `merge::merge_avg` / `merge::merge_rda`:
//!
//! * Learners are folded in the canonical reduction order (ascending learner
//!   id, which is the `Member` sort order when ids are unique).  A push that
//!   arrives before a lower-id learner is held unchanged until its turn.
//! * The normalised coefficient `(w / wsum) as f32` must be known before the
//!   first add, so the fold is used only when the final responder set and
//!   weights are fixed when the round starts: every expected learner must
//!   answer (quorum == roster) and learner weights are equal.  Token-weighted
//!   or partial-quorum rounds keep the buffered path.
//! * RDA needs each push's per-tensor L2 norm (kept as a few f64 scalars) and
//!   the weighted unit-direction sum.  Its degenerate fallback is the direct
//!   average, which cannot be recomputed after pushes are dropped, so RDA
//!   folds both accumulators.
//! * HeLoCo corrects each push against momentum on a copy; it keeps the
//!   buffered path (not needed by the strict full-parameter preset).
//!
//! Rounds outside those conditions (quorum below the roster, adaptive grace,
//! token weights) can use `StreamingFold` instead (S27 P4b,
//! `--weighted-fold streaming`): pushes fold in arrival order into
//! unnormalised sums that are divided by the weight sum at close.  It trades
//! bit identity for bounded memory; see its documentation for the bound.

use anyhow::{bail, Result};

use crate::merge;
use crate::state::{FragmentInfo, MERGE_AVG, MERGE_ISO, MERGE_RDA};

/// Running, order-exact reduction of one fragment for one round attempt.
#[derive(Debug)]
pub struct FragmentFold {
    /// Canonical fold order: ascending, unique learner ids of the roster.
    order: Vec<u32>,
    /// Number of learners in `order` already folded.
    folded: usize,
    weights: Vec<f64>,
    wsum: f64,
    tensor_numels: Vec<usize>,
    merge_mode: u8,
    /// Σ (w/wsum)·Δ — the direct average (AVG, ISO, RDA fallback).
    avg: Vec<f32>,
    /// Σ (w/wsum/‖Δ_t‖)·Δ per tensor — RDA direction sum.
    direction: Vec<f32>,
    /// RDA: `norms[t][m]` is the L2 norm of learner m's tensor t.
    norms: Vec<Vec<f64>>,
    /// First fold failure; reported by `finish` (pushes are already gone).
    failure: Option<anyhow::Error>,
}

impl FragmentFold {
    /// A fold for a round whose responders are exactly `roster` with equal
    /// weights, or `None` when the round must keep the buffered merge.
    pub fn for_equal_full_roster(fragment: &FragmentInfo, roster: &[u32]) -> Option<Self> {
        let mut order = roster.to_vec();
        order.sort_unstable();
        order.dedup();
        if order.is_empty() || order.len() != roster.len() {
            return None;
        }
        if !matches!(fragment.merge_mode, MERGE_AVG | MERGE_RDA | MERGE_ISO) {
            return None;
        }
        let tensor_numels = fragment
            .tensor_numels
            .iter()
            .map(|&n| usize::try_from(n).ok())
            .collect::<Option<Vec<_>>>()?;
        let weights = vec![1.0f64; order.len()];
        // Same expression as merge_avg / merge_rda so wsum is bit-identical.
        let wsum: f64 = weights.iter().sum();
        let norms = if fragment.merge_mode == MERGE_RDA {
            vec![Vec::with_capacity(order.len()); tensor_numels.len()]
        } else {
            Vec::new()
        };
        Some(Self {
            order,
            folded: 0,
            weights,
            wsum,
            tensor_numels,
            merge_mode: fragment.merge_mode,
            avg: Vec::new(),
            direction: Vec::new(),
            norms,
            failure: None,
        })
    }

    /// The learner id whose push must be folded next, if any.  A poisoned
    /// fold accepts nothing more.
    pub fn next_learner(&self) -> Option<u32> {
        if self.failure.is_some() {
            return None;
        }
        self.order.get(self.folded).copied()
    }

    /// Record a failure discovered while folding; `finish` returns it.
    pub fn poison(&mut self, error: anyhow::Error) {
        self.failure.get_or_insert(error);
    }

    pub fn is_complete(&self) -> bool {
        self.folded == self.order.len()
    }

    /// Learners folded so far, in fold order.
    pub fn folded_learners(&self) -> &[u32] {
        &self.order[..self.folded]
    }

    fn numel(&self) -> usize {
        self.tensor_numels.iter().sum()
    }

    /// Fold the next learner's outer gradient.  The buffer is consumed: the
    /// first push becomes the average accumulator in place.
    pub fn fold_next(&mut self, learner_id: u32, mut gradient: Vec<f32>) -> Result<()> {
        let Some(expected) = self.next_learner() else {
            bail!("fold already complete; unexpected learner {learner_id}");
        };
        if learner_id != expected {
            bail!("fold order violation: got learner {learner_id}, expected {expected}");
        }
        let numel = self.numel();
        if gradient.len() != numel {
            bail!(
                "push from learner {learner_id} has {} values, expected {numel}",
                gradient.len()
            );
        }
        let m = self.folded;
        let w = self.weights[m];
        let first = m == 0;

        if self.merge_mode == MERGE_RDA {
            if first {
                self.direction = zeroed(numel)?;
            }
            let mut off = 0;
            for (t, &tn) in self.tensor_numels.iter().enumerate() {
                let slice = &gradient[off..off + tn];
                let n = merge::l2_norm(slice);
                self.norms[t].push(n);
                if n != 0.0 {
                    let coef = (w / self.wsum / n) as f32;
                    for (o, d) in self.direction[off..off + tn].iter_mut().zip(slice) {
                        *o += coef * *d;
                    }
                }
                off += tn;
            }
        }

        let coef = (w / self.wsum) as f32;
        if first {
            // merge_avg starts from out.fill(0.0); `0.0 + x` keeps that
            // exact rounding (and its -0.0 -> +0.0 behaviour) in place.
            for d in gradient.iter_mut() {
                *d = 0.0f32 + coef * *d;
            }
            self.avg = gradient;
        } else {
            for (o, d) in self.avg.iter_mut().zip(&gradient) {
                *o += coef * *d;
            }
        }
        self.folded += 1;
        Ok(())
    }

    /// The reduced fragment, before any ISO spectrum flattening — the same
    /// values the buffered merge has in `delta` after its reduction loop.
    pub fn finish(self) -> Result<Vec<f32>> {
        if let Some(failure) = self.failure {
            return Err(failure.context("streaming fold failed"));
        }
        if !self.is_complete() {
            bail!(
                "fold incomplete: {} of {} learners",
                self.folded,
                self.order.len()
            );
        }
        if self.merge_mode != MERGE_RDA {
            return Ok(self.avg);
        }
        let Self {
            weights,
            wsum,
            tensor_numels,
            mut avg,
            mut direction,
            norms,
            ..
        } = self;
        let mut off = 0;
        for (t, &tn) in tensor_numels.iter().enumerate() {
            let out = &mut direction[off..off + tn];
            let radial = merge::rda_radial(&norms[t], &weights, wsum);
            if !merge::rda_rescale(out, radial) {
                // Degenerate mean direction: merge_rda falls back to Avg.
                out.copy_from_slice(&avg[off..off + tn]);
            }
            off += tn;
        }
        avg.clear();
        avg.shrink_to_fit();
        Ok(direction)
    }
}

/// How a round that the exact fold cannot reproduce reduces its pushes.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum WeightedFoldMode {
    /// Keep every push until the round closes (historical, bit-exact).
    #[default]
    Buffered,
    /// Fold each push on arrival (`StreamingFold`).
    Streaming,
}

impl std::str::FromStr for WeightedFoldMode {
    type Err = anyhow::Error;

    fn from_str(text: &str) -> Result<Self> {
        match text {
            "buffered" => Ok(Self::Buffered),
            "streaming" => Ok(Self::Streaming),
            other => bail!("--weighted-fold must be buffered or streaming, got {other:?}"),
        }
    }
}

/// Arrival-order streaming reduction for rounds whose responder set or
/// weights are only known when the round closes (quorum K below the roster,
/// adaptive grace, tokens²/steps weights).  S27 P4b.
///
/// Each push is folded on arrival into unnormalised accumulators and its
/// buffer is dropped.  `finish` normalises by the final weight sum.  To keep
/// the f32 accumulators well scaled whatever the magnitude of tokens²/steps
/// weights, every coefficient is taken relative to `w_ref`, the first
/// positive weight seen:
///
/// * `avg`       = Σ (w_m/w_ref)·Δ_m             (Avg, Iso, RDA fallback)
/// * `direction` = Σ (w_m/w_ref)·Δ_m,t/‖Δ_m,t‖   (RDA, per tensor t)
/// * `radial[t]` = Σ w_m·‖Δ_m,t‖                  (RDA, f64 scalars)
///
/// and `finish` multiplies by `w_ref/Σw`.  The result is not bit-identical
/// to the buffered merge (different rounding order); it is deterministic for
/// a given arrival sequence and agrees with the buffered merge to within a
/// few f32 ulps per accumulated push (see tests).
#[derive(Debug)]
pub struct StreamingFold {
    /// Merge weight of a push from its (c_tokens, c_steps).
    weight_of: fn(u64, u64) -> f64,
    tensor_numels: Vec<usize>,
    merge_mode: u8,
    /// Learners folded so far, in arrival order.
    folded: Vec<u32>,
    wsum: f64,
    /// First positive weight; 0 until one arrives.
    w_ref: f64,
    avg: Vec<f32>,
    direction: Vec<f32>,
    radial: Vec<f64>,
    failure: Option<anyhow::Error>,
}

impl StreamingFold {
    /// A streaming fold for `fragment`, or `None` for a merge mode it does
    /// not reproduce.
    pub fn new(fragment: &FragmentInfo, weight_of: fn(u64, u64) -> f64) -> Option<Self> {
        if !matches!(fragment.merge_mode, MERGE_AVG | MERGE_RDA | MERGE_ISO) {
            return None;
        }
        let tensor_numels = fragment
            .tensor_numels
            .iter()
            .map(|&n| usize::try_from(n).ok())
            .collect::<Option<Vec<_>>>()?;
        let radial = if fragment.merge_mode == MERGE_RDA {
            vec![0.0; tensor_numels.len()]
        } else {
            Vec::new()
        };
        Some(Self {
            weight_of,
            tensor_numels,
            merge_mode: fragment.merge_mode,
            folded: Vec::new(),
            wsum: 0.0,
            w_ref: 0.0,
            avg: Vec::new(),
            direction: Vec::new(),
            radial,
            failure: None,
        })
    }

    /// Learners folded so far, in arrival order.
    pub fn folded_learners(&self) -> &[u32] {
        &self.folded
    }

    /// The merge weight of a push with these counters.
    pub fn weight_of(&self, c_tokens: u64, c_steps: u64) -> f64 {
        (self.weight_of)(c_tokens, c_steps)
    }

    pub fn is_poisoned(&self) -> bool {
        self.failure.is_some()
    }

    /// Record a failure discovered while folding; `finish` returns it.
    pub fn poison(&mut self, error: anyhow::Error) {
        self.failure.get_or_insert(error);
    }

    fn numel(&self) -> usize {
        self.tensor_numels.iter().sum()
    }

    /// Fold one learner's outer gradient with merge weight `weight` and
    /// release its buffer.
    pub fn fold(&mut self, learner_id: u32, gradient: Vec<f32>, weight: f64) -> Result<()> {
        if self.failure.is_some() {
            bail!("fold is poisoned; learner {learner_id} not folded");
        }
        if !(weight.is_finite() && weight >= 0.0) {
            bail!("learner {learner_id} has invalid merge weight {weight}");
        }
        let numel = self.numel();
        if gradient.len() != numel {
            bail!(
                "push from learner {learner_id} has {} values, expected {numel}",
                gradient.len()
            );
        }
        if self.folded.is_empty() {
            self.avg = zeroed(numel)?;
            if self.merge_mode == MERGE_RDA {
                self.direction = zeroed(numel)?;
            }
        }
        self.folded.push(learner_id);
        self.wsum += weight;
        if weight == 0.0 {
            // Contributes nothing to any accumulator, as in merge_avg/rda.
            return Ok(());
        }
        if self.w_ref == 0.0 {
            self.w_ref = weight;
        }
        let rel = weight / self.w_ref;
        if self.merge_mode == MERGE_RDA {
            let mut off = 0;
            for (t, &tn) in self.tensor_numels.iter().enumerate() {
                let slice = &gradient[off..off + tn];
                let n = merge::l2_norm(slice);
                self.radial[t] += weight * n;
                if n != 0.0 {
                    let coef = (rel / n) as f32;
                    for (o, d) in self.direction[off..off + tn].iter_mut().zip(slice) {
                        *o += coef * *d;
                    }
                }
                off += tn;
            }
        }
        let coef = rel as f32;
        for (o, d) in self.avg.iter_mut().zip(&gradient) {
            *o += coef * *d;
        }
        Ok(())
    }

    /// The normalised reduced fragment (before ISO spectrum flattening).
    pub fn finish(self) -> Result<Vec<f32>> {
        if let Some(failure) = self.failure {
            return Err(failure.context("streaming fold failed"));
        }
        if self.folded.is_empty() {
            bail!("streaming fold finished with no pushes");
        }
        let Self {
            tensor_numels,
            merge_mode,
            wsum,
            w_ref,
            mut avg,
            mut direction,
            radial,
            ..
        } = self;
        if wsum <= 0.0 {
            // merge_avg / merge_rda return zeros for a zero weight sum.
            avg.fill(0.0);
            return Ok(avg);
        }
        let scale = (w_ref / wsum) as f32;
        for v in avg.iter_mut() {
            *v *= scale;
        }
        if merge_mode != MERGE_RDA {
            return Ok(avg);
        }
        for v in direction.iter_mut() {
            *v *= scale;
        }
        let mut off = 0;
        for (t, &tn) in tensor_numels.iter().enumerate() {
            let out = &mut direction[off..off + tn];
            if !merge::rda_rescale(out, radial[t] / wsum) {
                out.copy_from_slice(&avg[off..off + tn]);
            }
            off += tn;
        }
        Ok(direction)
    }
}

/// The streaming reduction a round uses, if any (see `server::round_fold`).
#[derive(Debug)]
pub enum RoundFold {
    /// Canonical-order, bit-identical to the buffered merge.
    Exact(FragmentFold),
    /// Arrival-order, normalised at close.
    Streaming(StreamingFold),
}

impl RoundFold {
    pub fn folded_learners(&self) -> &[u32] {
        match self {
            Self::Exact(fold) => fold.folded_learners(),
            Self::Streaming(fold) => fold.folded_learners(),
        }
    }

    pub fn finish(self) -> Result<Vec<f32>> {
        match self {
            Self::Exact(fold) => fold.finish(),
            Self::Streaming(fold) => fold.finish(),
        }
    }
}

fn zeroed(numel: usize) -> Result<Vec<f32>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(numel)
        .map_err(|_| anyhow::anyhow!("cannot allocate {numel}-value fold accumulator"))?;
    values.resize(numel, 0.0);
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fragment(mode: u8, tensors: &[u64]) -> FragmentInfo {
        FragmentInfo {
            merge_mode: mode,
            tensor_numels: tensors.to_vec(),
            tensor_shapes: (mode == MERGE_ISO)
                .then(|| tensors.iter().map(|&n| (1, n)).collect()),
        }
    }

    /// Deterministic pseudo-random values with mixed signs, zeros, -0.0 and
    /// wide exponents so rounding differences would show.
    fn values(seed: u64, n: usize) -> Vec<f32> {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|i| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                match i % 11 {
                    0 => 0.0,
                    1 => -0.0,
                    _ => {
                        let mantissa = (x >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                        let exponent = ((x >> 8) % 24) as i32 - 12;
                        mantissa * 2f32.powi(exponent)
                    }
                }
            })
            .collect()
    }

    fn buffered(fragment: &FragmentInfo, grads: &[Vec<f32>]) -> Vec<f32> {
        let weights = vec![1.0; grads.len()];
        let numel: usize = fragment.tensor_numels.iter().sum::<u64>() as usize;
        let mut out = vec![0.0f32; numel];
        let mut off = 0;
        for &tn in &fragment.tensor_numels {
            let tn = tn as usize;
            let slices: Vec<&[f32]> = grads.iter().map(|g| &g[off..off + tn]).collect();
            let o = &mut out[off..off + tn];
            match fragment.merge_mode {
                MERGE_RDA => merge::merge_rda(&slices, &weights, o),
                _ => merge::merge_avg(&slices, &weights, o),
            }
            off += tn;
        }
        out
    }

    fn folded(fragment: &FragmentInfo, ids: &[u32], grads: &[Vec<f32>]) -> Vec<f32> {
        let mut fold = FragmentFold::for_equal_full_roster(fragment, ids).unwrap();
        for (&id, g) in ids.iter().zip(grads) {
            fold.fold_next(id, g.clone()).unwrap();
        }
        fold.finish().unwrap()
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    #[test]
    fn fold_is_bit_identical_to_buffered_merge_for_every_mode() {
        for mode in [MERGE_AVG, MERGE_RDA, MERGE_ISO] {
            for learners in 1..=5u32 {
                let frag = fragment(mode, &[37, 1, 256, 3]);
                let ids: Vec<u32> = (0..learners).collect();
                let grads: Vec<Vec<f32>> =
                    ids.iter().map(|&id| values(u64::from(id) + 7, 297)).collect();
                assert_eq!(
                    bits(&folded(&frag, &ids, &grads)),
                    bits(&buffered(&frag, &grads)),
                    "mode {mode} learners {learners}"
                );
            }
        }
    }

    #[test]
    fn rda_fold_matches_zero_and_cancelling_tensors() {
        let frag = fragment(MERGE_RDA, &[4, 4, 4]);
        let a = vec![1.0, -2.0, 3.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 1.0, -1.0, 0.25, 0.0, 0.0, 0.0, 0.0, -1.0, -2.0, -3.0, -4.0];
        let grads = vec![a, b];
        assert_eq!(
            bits(&folded(&frag, &[0, 1], &grads)),
            bits(&buffered(&frag, &grads))
        );
    }

    #[test]
    fn fold_rejects_wrong_order_length_and_incomplete_finish() {
        let frag = fragment(MERGE_AVG, &[2]);
        let mut fold = FragmentFold::for_equal_full_roster(&frag, &[3, 1]).unwrap();
        assert_eq!(fold.next_learner(), Some(1));
        assert!(fold.fold_next(3, vec![1.0, 2.0]).is_err());
        assert!(fold.fold_next(1, vec![1.0]).is_err());
        fold.fold_next(1, vec![1.0, 2.0]).unwrap();
        assert_eq!(fold.folded_learners(), &[1]);
        let partial = FragmentFold::for_equal_full_roster(&frag, &[1, 3]).unwrap();
        assert!(partial.finish().is_err());
        assert!(FragmentFold::for_equal_full_roster(&frag, &[1, 1]).is_none());
        assert!(FragmentFold::for_equal_full_roster(&frag, &[]).is_none());
    }

    // ---- StreamingFold (arrival order, normalised at close) ----

    fn weighted_buffered(fragment: &FragmentInfo, grads: &[Vec<f32>], weights: &[f64]) -> Vec<f32> {
        let numel: usize = fragment.tensor_numels.iter().sum::<u64>() as usize;
        let mut out = vec![0.0f32; numel];
        let mut off = 0;
        for &tn in &fragment.tensor_numels {
            let tn = tn as usize;
            let slices: Vec<&[f32]> = grads.iter().map(|g| &g[off..off + tn]).collect();
            let o = &mut out[off..off + tn];
            match fragment.merge_mode {
                MERGE_RDA => merge::merge_rda(&slices, weights, o),
                _ => merge::merge_avg(&slices, weights, o),
            }
            off += tn;
        }
        out
    }

    /// f64 reference of the same reduction (what numpy computes in f64).
    fn reference(fragment: &FragmentInfo, grads: &[Vec<f32>], weights: &[f64]) -> Vec<f64> {
        let wsum: f64 = weights.iter().sum();
        let numel: usize = fragment.tensor_numels.iter().sum::<u64>() as usize;
        let mut out = vec![0.0f64; numel];
        if wsum <= 0.0 {
            return out;
        }
        let mut off = 0;
        for &tn in &fragment.tensor_numels {
            let tn = tn as usize;
            let avg: Vec<f64> = (off..off + tn)
                .map(|i| grads.iter().zip(weights).map(|(g, w)| w * g[i] as f64).sum::<f64>() / wsum)
                .collect();
            let result = if fragment.merge_mode == MERGE_RDA {
                let norms: Vec<f64> = grads
                    .iter()
                    .map(|g| g[off..off + tn].iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt())
                    .collect();
                let radial = norms.iter().zip(weights).map(|(n, w)| n * w).sum::<f64>() / wsum;
                let dir: Vec<f64> = (off..off + tn)
                    .map(|i| {
                        grads
                            .iter()
                            .zip(weights)
                            .zip(&norms)
                            .filter(|(_, &n)| n != 0.0)
                            .map(|((g, w), n)| w * g[i] as f64 / n)
                            .sum::<f64>()
                            / wsum
                    })
                    .collect();
                let dn = dir.iter().map(|v| v * v).sum::<f64>().sqrt();
                if dn < 1e-12 {
                    avg
                } else {
                    dir.iter().map(|v| v * radial / dn).collect()
                }
            } else {
                avg
            };
            out[off..off + tn].copy_from_slice(&result);
            off += tn;
        }
        out
    }

    fn streamed(fragment: &FragmentInfo, order: &[usize], grads: &[Vec<f32>], weights: &[f64]) -> Vec<f32> {
        let mut fold = StreamingFold::new(fragment, merge::learner_weight).unwrap();
        for &m in order {
            fold.fold(m as u32, grads[m].clone(), weights[m]).unwrap();
        }
        fold.finish().unwrap()
    }

    /// Largest |a−b| relative to the fragment's L∞ scale of the reference.
    fn max_rel_error(a: &[f32], b: &[f64]) -> f64 {
        let scale = b.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(f64::MIN_POSITIVE);
        a.iter().zip(b).map(|(x, y)| (*x as f64 - y).abs()).fold(0.0, f64::max) / scale
    }

    /// tokens²/steps weights spanning several orders of magnitude.
    fn token_weights(n: usize) -> Vec<f64> {
        (0..n)
            .map(|m| merge::learner_weight(1000 * (m as u64 + 1) * (m as u64 + 3), 1 + m as u64 % 3))
            .collect()
    }

    fn permutations(n: usize) -> Vec<Vec<usize>> {
        if n == 0 {
            return vec![Vec::new()];
        }
        let mut out = Vec::new();
        for rest in permutations(n - 1) {
            for at in 0..=rest.len() {
                let mut order = rest.clone();
                order.insert(at, n - 1);
                out.push(order);
            }
        }
        out
    }

    // Bound: each of M f32 accumulations plus the final rescale adds at most
    // ~1 ulp relative to the L∞ scale; values here have wide exponents, so
    // allow 4·M ulps of headroom.
    fn tolerance(learners: usize) -> f64 {
        4.0 * (learners as f64 + 2.0) * f32::EPSILON as f64
    }

    #[test]
    fn streaming_fold_matches_buffered_and_f64_reference_in_every_order() {
        for mode in [MERGE_AVG, MERGE_RDA, MERGE_ISO] {
            for learners in 1..=5usize {
                let frag = fragment(mode, &[37, 1, 256, 3]);
                let grads: Vec<Vec<f32>> = (0..learners).map(|m| values(m as u64 + 11, 297)).collect();
                let weights = token_weights(learners);
                let exact = reference(&frag, &grads, &weights);
                let buffered = weighted_buffered(&frag, &grads, &weights);
                let tol = tolerance(learners);
                assert!(max_rel_error(&buffered, &exact) <= tol, "buffered mode {mode}");
                let first = streamed(&frag, &(0..learners).collect::<Vec<_>>(), &grads, &weights);
                for order in permutations(learners) {
                    let out = streamed(&frag, &order, &grads, &weights);
                    let err = max_rel_error(&out, &exact);
                    assert!(err <= tol, "mode {mode} learners {learners} order {order:?}: {err:e} > {tol:e}");
                    let buffered64: Vec<f64> = buffered.iter().map(|v| *v as f64).collect();
                    assert!(max_rel_error(&out, &buffered64) <= 2.0 * tol);
                    if order.iter().copied().eq(0..learners) {
                        assert_eq!(bits(&out), bits(&first), "same arrival order must repeat bit-exactly");
                    }
                }
            }
        }
    }

    #[test]
    fn streaming_fold_handles_zero_weights_and_degenerate_rda() {
        let frag = fragment(MERGE_RDA, &[4, 4]);
        let a = vec![1.0, -2.0, 3.0, 0.5, 1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 1.0, -1.0, 0.25, -1.0, -2.0, -3.0, -4.0];
        let grads = vec![a, b];
        // Equal weights: second tensor cancels and falls back to Avg (zeros).
        let w = [3.0, 3.0];
        let out = streamed(&frag, &[1, 0], &grads, &w);
        let exact = reference(&frag, &grads, &w);
        assert!(max_rel_error(&out, &exact) <= tolerance(2));
        assert!(out[4..].iter().all(|v| *v == 0.0));
        // A zero-weight push contributes nothing, even when it arrives first.
        let w = [0.0, 5.0];
        let out = streamed(&frag, &[0, 1], &grads, &w);
        let buffered = weighted_buffered(&frag, &grads, &w);
        assert!(max_rel_error(&out, &buffered.iter().map(|v| *v as f64).collect::<Vec<_>>()) <= tolerance(2));
        // All-zero weights reduce to zeros like merge_avg / merge_rda.
        let out = streamed(&frag, &[0, 1], &grads, &[0.0, 0.0]);
        assert!(out.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn streaming_fold_rejects_bad_input() {
        let frag = fragment(MERGE_AVG, &[2]);
        let mut fold = StreamingFold::new(&frag, merge::learner_weight).unwrap();
        assert!(fold.fold(1, vec![1.0], 1.0).is_err());
        assert!(fold.fold(1, vec![1.0, 2.0], f64::NAN).is_err());
        assert!(fold.fold(1, vec![1.0, 2.0], -1.0).is_err());
        fold.fold(1, vec![1.0, 2.0], 1.0).unwrap();
        assert_eq!(fold.weight_of(10, 2), 50.0);
        let empty = StreamingFold::new(&frag, merge::learner_weight).unwrap();
        assert!(empty.finish().is_err());
        let mut poisoned = StreamingFold::new(&frag, merge::learner_weight).unwrap();
        poisoned.poison(anyhow::anyhow!("boom"));
        assert!(poisoned.finish().is_err());
    }
}
