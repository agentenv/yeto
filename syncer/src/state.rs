//! Global model state held by the syncer: fragment layout, parameters Θ,
//! and outer-optimizer momentum, all in f32.

use anyhow::{bail, Result};

use crate::merge;
use crate::protocol::Reader;

pub const MERGE_AVG: u8 = 0;
pub const MERGE_RDA: u8 = 1;

#[derive(Clone, Debug, PartialEq)]
pub struct FragmentInfo {
    pub merge_mode: u8,
    pub tensor_numels: Vec<u64>,
}

impl FragmentInfo {
    pub fn numel(&self) -> usize {
        self.tensor_numels.iter().sum::<u64>() as usize
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Layout {
    pub fragments: Vec<FragmentInfo>,
}

impl Layout {
    /// Decode the layout section of a HELLO payload (cursor positioned after
    /// learner_id/dtype/num_fragments have been consumed except the count).
    pub fn decode(r: &mut Reader<'_>, num_fragments: u32) -> Result<Self> {
        let mut fragments = Vec::with_capacity(num_fragments as usize);
        for _ in 0..num_fragments {
            let merge_mode = r.u8()?;
            if merge_mode > MERGE_RDA {
                bail!("bad merge mode {merge_mode}");
            }
            let num_tensors = r.u32()?;
            let mut tensor_numels = Vec::with_capacity(num_tensors as usize);
            for _ in 0..num_tensors {
                tensor_numels.push(r.u64()?);
            }
            fragments.push(FragmentInfo { merge_mode, tensor_numels });
        }
        Ok(Layout { fragments })
    }
}

/// Cumulative per-learner merge accounting (the "event-tape ledger").
#[derive(Clone, Copy, Default)]
pub struct LearnerLedger {
    pub merges: u64,
    pub steps: u64,
    pub tokens: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MergeStats {
    /// L2 norm of the merged pseudo-gradient before the outer optimizer.
    pub gnorm: f64,
    pub outer: merge::OuterStepStats,
}

pub struct GlobalState {
    pub layout: Layout,
    pub layout_meta: Option<String>,
    /// Θ_p, flat f32 per fragment (concatenated tensors in layout order).
    pub params: Vec<Vec<f32>>,
    /// Outer-optimizer buffers, same shape as params.
    momentum: Vec<Vec<f32>>,
    pub initialized: Vec<bool>,
    /// Global step at which each fragment was last merged (its version).
    pub versions: Vec<u64>,
    /// Last completed global step t (checkpoint cut point).
    pub global_step: u64,
    pub ledger: std::collections::BTreeMap<u32, LearnerLedger>,
    pub outer_lr: f32,
    pub outer_lr_by_fragment: Option<Vec<f32>>,
    pub outer_momentum: f32,
    pub outer_optimizer: merge::OuterOptimizer,
    pub outer_restart_cos_threshold: f32,
    /// Dtype used on the wire (from HELLO); merge math stays f32.
    pub wire_dtype: u8,
    /// HeLoCo per-tensor directional correction of learner deltas against
    /// the outer momentum before merging (None disables).
    pub delta_correction: Option<merge::Heloco>,
}

impl GlobalState {
    pub fn new(
        layout: Layout,
        layout_meta: Option<String>,
        outer_lr: f32,
        outer_momentum: f32,
        wire_dtype: u8,
    ) -> Self {
        let params: Vec<Vec<f32>> = layout.fragments.iter().map(|f| vec![0.0; f.numel()]).collect();
        let momentum = params.clone();
        let initialized = vec![false; layout.fragments.len()];
        let versions = vec![0; layout.fragments.len()];
        Self {
            layout,
            layout_meta,
            params,
            momentum,
            initialized,
            versions,
            global_step: 0,
            ledger: Default::default(),
            outer_lr,
            outer_lr_by_fragment: None,
            outer_momentum,
            outer_optimizer: merge::OuterOptimizer::Nesterov,
            outer_restart_cos_threshold: 0.0,
            wire_dtype,
            delta_correction: None,
        }
    }

    pub fn all_initialized(&self) -> bool {
        self.initialized.iter().all(|&b| b)
    }

    pub fn init_fragment(&mut self, fid: usize, values: Vec<f32>) -> Result<()> {
        if values.len() != self.params[fid].len() {
            bail!("init fragment {fid}: got {} values, expected {}", values.len(), self.params[fid].len());
        }
        if !self.initialized[fid] {
            self.params[fid] = values;
            self.initialized[fid] = true;
        }
        Ok(())
    }

    pub fn record_merge(&mut self, learner_id: u32, c_steps: u32, c_tokens: u64) {
        let e = self.ledger.entry(learner_id).or_default();
        e.merges += 1;
        e.steps += c_steps as u64;
        e.tokens += c_tokens;
    }

    /// Merge learner copies of fragment `fid` and apply the outer step.
    /// The returned `gnorm` remains the pre-optimizer merged-delta norm.
    pub fn merge_and_step(
        &mut self,
        fid: usize,
        learners: &[&[f32]],
        weights: &[f64],
    ) -> Result<MergeStats> {
        let frag = &self.layout.fragments[fid];
        let numel = frag.numel();
        for (i, l) in learners.iter().enumerate() {
            if l.len() != numel {
                bail!("push for fragment {fid} from entry {i} has {} values, expected {numel}", l.len());
            }
        }
        // HeLoCo: correct each learner's outer delta against the outer
        // momentum, per tensor, before merging (stale deltas can oppose the
        // current global trajectory). Materializes corrected copies so the
        // anchor-based merge below stays unchanged.
        let corrected: Vec<Vec<f32>>;
        let learners: Vec<&[f32]> = if let Some(h) = self.delta_correction {
            let anchor = &self.params[fid];
            let momentum = &self.momentum[fid];
            corrected = learners
                .iter()
                .map(|l| {
                    let mut vals = l.to_vec();
                    let mut off = 0usize;
                    for &tn in &frag.tensor_numels {
                        let tn = tn as usize;
                        let mut d: Vec<f32> = anchor[off..off + tn]
                            .iter()
                            .zip(&vals[off..off + tn])
                            .map(|(a, v)| a - v)
                            .collect();
                        merge::heloco_correct(&mut d, &momentum[off..off + tn], &h);
                        for (i, di) in d.iter().enumerate() {
                            vals[off + i] = anchor[off + i] - di;
                        }
                        off += tn;
                    }
                    vals
                })
                .collect();
            corrected.iter().map(|v| v.as_slice()).collect()
        } else {
            learners.to_vec()
        };
        let learners = learners.as_slice();
        let anchor = &self.params[fid];
        let mut delta = vec![0.0f32; numel];
        // Merge per tensor slice within the fragment.
        let mut off = 0usize;
        for &tn in &frag.tensor_numels {
            let tn = tn as usize;
            let slice_learners: Vec<&[f32]> = learners.iter().map(|l| &l[off..off + tn]).collect();
            let out = &mut delta[off..off + tn];
            match frag.merge_mode {
                MERGE_AVG => merge::merge_avg(&anchor[off..off + tn], &slice_learners, weights, out),
                _ => merge::merge_rda(&anchor[off..off + tn], &slice_learners, weights, out),
            }
            off += tn;
        }
        let gnorm = delta.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
        let outer_lr = self
            .outer_lr_by_fragment
            .as_ref()
            .map(|rates| rates[fid])
            .unwrap_or(self.outer_lr);
        let outer = match self.outer_optimizer {
            merge::OuterOptimizer::Nesterov => merge::nesterov_step(
                &mut self.params[fid],
                &mut self.momentum[fid],
                &delta,
                outer_lr,
                self.outer_momentum,
            ),
            merge::OuterOptimizer::NormalizedEma => merge::normalized_ema_step(
                &mut self.params[fid],
                &mut self.momentum[fid],
                &delta,
                outer_lr,
                self.outer_momentum,
            ),
            merge::OuterOptimizer::RestartedEma => merge::restarted_ema_step(
                &mut self.params[fid],
                &mut self.momentum[fid],
                &delta,
                outer_lr,
                self.outer_momentum,
                self.outer_restart_cos_threshold,
            ),
        };
        Ok(MergeStats { gnorm, outer })
    }
}

const CKPT_MAGIC: u32 = 0xD170_5A7E;

impl GlobalState {
    /// Persist a consistent snapshot. Called only at the quiescent cut
    /// between rounds (see docs/PROTOCOL.md "Consistent snapshots").
    /// Written to `<path>.tmp` then renamed, so a crash mid-write never
    /// corrupts the previous checkpoint.
    pub fn save_checkpoint(&self, path: &std::path::Path) -> Result<()> {
        use std::io::Write;
        let tmp = path.with_extension("tmp");
        {
            let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
            f.write_all(&CKPT_MAGIC.to_le_bytes())?;
            f.write_all(&self.global_step.to_le_bytes())?;
            f.write_all(&(self.params.len() as u32).to_le_bytes())?;
            for p in 0..self.params.len() {
                f.write_all(&self.versions[p].to_le_bytes())?;
                f.write_all(&(self.params[p].len() as u64).to_le_bytes())?;
                for v in &self.params[p] {
                    f.write_all(&v.to_le_bytes())?;
                }
                for v in &self.momentum[p] {
                    f.write_all(&v.to_le_bytes())?;
                }
            }
            f.write_all(&(self.ledger.len() as u32).to_le_bytes())?;
            for (id, l) in &self.ledger {
                f.write_all(&id.to_le_bytes())?;
                f.write_all(&l.merges.to_le_bytes())?;
                f.write_all(&l.steps.to_le_bytes())?;
                f.write_all(&l.tokens.to_le_bytes())?;
            }
            if let Some(meta) = &self.layout_meta {
                let bytes = meta.as_bytes();
                f.write_all(&(bytes.len() as u32).to_le_bytes())?;
                f.write_all(bytes)?;
            }
            f.flush()?;
        }
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Restore params/momentum/versions/step/ledger from a snapshot.
    /// The layout (from HELLO) must match the checkpointed fragment shapes.
    pub fn load_checkpoint(&mut self, path: &std::path::Path) -> Result<()> {
        use std::io::Read;
        let mut buf = Vec::new();
        std::fs::File::open(path)?.read_to_end(&mut buf)?;
        let mut r = Reader(&buf);
        if r.u32()? != CKPT_MAGIC {
            bail!("bad checkpoint magic");
        }
        self.global_step = r.u64()?;
        let np = r.u32()? as usize;
        if np != self.params.len() {
            bail!("checkpoint has {np} fragments, layout has {}", self.params.len());
        }
        for p in 0..np {
            self.versions[p] = r.u64()?;
            let numel = r.u64()? as usize;
            if numel != self.params[p].len() {
                bail!("checkpoint fragment {p} numel {numel} != layout {}", self.params[p].len());
            }
            for slot in [&mut self.params[p], &mut self.momentum[p]] {
                for v in slot.iter_mut() {
                    *v = f32::from_le_bytes(r.take(4)?.try_into()?);
                }
            }
            self.initialized[p] = true;
        }
        let nl = r.u32()? as usize;
        self.ledger.clear();
        for _ in 0..nl {
            let id = r.u32()?;
            let l = LearnerLedger { merges: r.u64()?, steps: r.u64()?, tokens: r.u64()? };
            self.ledger.insert(id, l);
        }
        if !r.0.is_empty() {
            let n = r.u32()? as usize;
            let bytes = r.take(n)?;
            if !r.0.is_empty() {
                bail!("checkpoint has trailing bytes after layout metadata");
            }
            let meta = String::from_utf8(bytes.to_vec())?;
            if let Some(current) = &self.layout_meta {
                if current != &meta {
                    bail!("checkpoint layout metadata does not match HELLO metadata");
                }
            }
            self.layout_meta = Some(meta);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout2() -> Layout {
        Layout {
            fragments: vec![
                FragmentInfo { merge_mode: MERGE_AVG, tensor_numels: vec![4] },
                FragmentInfo { merge_mode: MERGE_RDA, tensor_numels: vec![2, 2] },
            ],
        }
    }

    #[test]
    fn init_once() {
        let mut st = GlobalState::new(layout2(), None, 0.7, 0.9, crate::protocol::DTYPE_F32);
        st.init_fragment(0, vec![1.0; 4]).unwrap();
        st.init_fragment(0, vec![2.0; 4]).unwrap(); // ignored
        assert_eq!(st.params[0], vec![1.0; 4]);
        assert!(!st.all_initialized());
        st.init_fragment(1, vec![0.0; 4]).unwrap();
        assert!(st.all_initialized());
    }

    #[test]
    fn merge_moves_toward_learners() {
        let mut st = GlobalState::new(layout2(), None, 1.0, 0.0, crate::protocol::DTYPE_F32); // plain SGD lr=1 = weight averaging
        st.init_fragment(0, vec![1.0; 4]).unwrap();
        st.init_fragment(1, vec![1.0; 4]).unwrap();
        let learner = vec![0.0f32; 4];
        let stats = st.merge_and_step(0, &[&learner], &[1.0]).unwrap();
        assert!(stats.gnorm > 0.0);
        // Θ − 1.0·(Θ − θ) = θ
        assert_eq!(st.params[0], vec![0.0; 4]);
    }

    #[test]
    fn default_outer_optimizer_regresses_to_nesterov() {
        let mut st = GlobalState::new(layout2(), None, 0.7, 0.9, crate::protocol::DTYPE_F32);
        assert_eq!(st.outer_optimizer, merge::OuterOptimizer::Nesterov);
        st.init_fragment(0, vec![1.0; 4]).unwrap();
        st.init_fragment(1, vec![0.0; 4]).unwrap();
        let stats = st.merge_and_step(0, &[&[0.5f32; 4]], &[1.0]).unwrap();
        let expected = 1.0 - 0.7 * (0.5 + 0.9 * 0.5);
        for value in &st.params[0] {
            assert!((*value - expected).abs() < 1e-6);
        }
        assert_eq!(st.momentum[0], vec![0.5; 4]);
        assert!((stats.gnorm - 1.0).abs() < 1e-12);
        let expected_step_norm = 2.0 * (0.7f32 * (0.5 + 0.9 * 0.5)) as f64;
        assert!((stats.outer.applied_step_norm - expected_step_norm).abs() < 1e-6);
        assert_eq!(stats.outer.direction_delta_cosine, Some(1.0));
        assert_eq!(stats.outer.history_current_norm_ratio, Some(0.0));
        assert!(!stats.outer.restarted);
    }

    #[test]
    fn fragment_outer_lr_overrides_global_rate() {
        let mut st = GlobalState::new(layout2(), None, 1.0, 0.0, crate::protocol::DTYPE_F32);
        st.outer_lr_by_fragment = Some(vec![0.5, 1.0]);
        st.init_fragment(0, vec![1.0; 4]).unwrap();
        st.init_fragment(1, vec![1.0; 4]).unwrap();
        let learner = vec![0.0f32; 4];
        st.merge_and_step(0, &[&learner], &[1.0]).unwrap();
        assert_eq!(st.params[0], vec![0.5; 4]);
    }

    #[test]
    fn checkpoint_roundtrip() {
        let dir = std::env::temp_dir().join("yeto-ckpt-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("state.ckpt");
        let mut st = GlobalState::new(layout2(), Some("{\"task\":\"nava\"}".to_string()), 0.7, 0.9, crate::protocol::DTYPE_F32);
        st.init_fragment(0, vec![1.5; 4]).unwrap();
        st.init_fragment(1, vec![-2.0; 4]).unwrap();
        let learner = vec![0.0f32; 4];
        st.merge_and_step(0, &[&learner], &[1.0]).unwrap();
        st.global_step = 7;
        st.versions[0] = 7;
        st.record_merge(3, 12, 4096);
        st.save_checkpoint(&path).unwrap();

        let mut st2 = GlobalState::new(layout2(), Some("{\"task\":\"nava\"}".to_string()), 0.7, 0.9, crate::protocol::DTYPE_F32);
        st2.load_checkpoint(&path).unwrap();
        assert_eq!(st2.global_step, 7);
        assert_eq!(st2.versions, vec![7, 0]);
        assert_eq!(st2.params, st.params);
        assert!(st2.all_initialized());
        assert_eq!(st2.ledger.get(&3).unwrap().tokens, 4096);
        assert_eq!(st2.layout_meta.as_deref(), Some("{\"task\":\"nava\"}"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn legacy_checkpoint_load_preserves_runtime_outer_policy() {
        let dir = std::env::temp_dir().join("yeto-ckpt-outer-policy-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("state.ckpt");

        let mut legacy = GlobalState::new(layout2(), None, 0.7, 0.9, crate::protocol::DTYPE_F32);
        legacy.init_fragment(0, vec![1.0; 4]).unwrap();
        legacy.init_fragment(1, vec![0.0; 4]).unwrap();
        legacy.merge_and_step(0, &[&[0.0f32; 4]], &[1.0]).unwrap();
        legacy.save_checkpoint(&path).unwrap();

        let mut resumed = GlobalState::new(layout2(), None, 0.2, 0.8, crate::protocol::DTYPE_F32);
        resumed.outer_optimizer = merge::OuterOptimizer::RestartedEma;
        resumed.outer_restart_cos_threshold = -0.25;
        resumed.load_checkpoint(&path).unwrap();

        assert_eq!(resumed.params, legacy.params);
        assert_eq!(resumed.momentum, legacy.momentum);
        assert_eq!(resumed.outer_optimizer, merge::OuterOptimizer::RestartedEma);
        assert_eq!(resumed.outer_restart_cos_threshold, -0.25);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn heloco_correction_damps_anti_aligned_learner() {
        // Two identical states with warm momentum; the learner's delta
        // opposes it. With correction the outer step must move less far
        // in the opposing direction than without.
        let mk = || {
            let mut st = GlobalState::new(layout2(), None, 1.0, 0.0, crate::protocol::DTYPE_F32);
            st.init_fragment(0, vec![0.0; 4]).unwrap();
            st.init_fragment(1, vec![0.0; 4]).unwrap();
            // Warm the momentum: a learner pulling params down (delta +1).
            st.merge_and_step(0, &[&[-1.0f32; 4][..]], &[1.0]).unwrap();
            st
        };
        let mut plain = mk();
        let mut corrected = mk();
        corrected.delta_correction = Some(merge::Heloco::default());
        // Now a learner pulling the opposite way (delta anchored at current params).
        let opposing: Vec<f32> = plain.params[0].iter().map(|p| p + 3.0).collect();
        plain.merge_and_step(0, &[&opposing], &[1.0]).unwrap();
        let opposing2: Vec<f32> = corrected.params[0].iter().map(|p| p + 3.0).collect();
        corrected.merge_and_step(0, &[&opposing2], &[1.0]).unwrap();
        // The opposing learner drags params up; the correction shrinks the
        // anti-aligned delta, so the corrected state moves up less.
        assert!(
            corrected.params[0][0] < plain.params[0][0],
            "corrected {} !< plain {}",
            corrected.params[0][0],
            plain.params[0][0]
        );
    }

    #[test]
    fn size_mismatch_rejected() {
        let mut st = GlobalState::new(layout2(), None, 0.7, 0.9, crate::protocol::DTYPE_F32);
        assert!(st.init_fragment(0, vec![1.0; 3]).is_err());
        let learner = vec![0.0f32; 3];
        assert!(st.merge_and_step(0, &[&learner], &[1.0]).is_err());
    }
}
