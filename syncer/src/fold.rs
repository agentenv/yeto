//! Streaming reduction of one fragment's learner pushes (S27 P4).
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
}
