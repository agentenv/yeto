//! Global model state held by the syncer: fragment layout, parameters Θ,
//! and outer-optimizer momentum, all in f32.

use anyhow::{bail, Context, Result};

use crate::merge;
use crate::protocol::Reader;

pub const MERGE_AVG: u8 = 0;
pub const MERGE_RDA: u8 = 1;
pub const MERGE_ISO: u8 = 2;

#[derive(Clone, Debug, PartialEq)]
pub struct FragmentInfo {
    pub merge_mode: u8,
    pub tensor_numels: Vec<u64>,
    /// Per-tensor (rows, cols) matrix shapes, present exactly for MERGE_ISO.
    pub tensor_shapes: Option<Vec<(u64, u64)>>,
}

impl FragmentInfo {
    pub fn numel(&self) -> Result<usize> {
        let total = self
            .tensor_numels
            .iter()
            .try_fold(0u64, |sum, value| sum.checked_add(*value))
            .ok_or_else(|| anyhow::anyhow!("fragment numel overflow"))?;
        if total == 0 {
            bail!("fragment numel must be positive");
        }
        usize::try_from(total).map_err(|_| anyhow::anyhow!("fragment numel does not fit usize"))
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
        if num_fragments == 0 {
            bail!("layout must contain at least one fragment");
        }
        if num_fragments as usize > r.remaining() / 5 {
            bail!("fragment count exceeds HELLO payload bounds");
        }
        let fragment_count = num_fragments as usize;
        let mut fragments = Vec::new();
        fragments
            .try_reserve_exact(fragment_count)
            .context("cannot allocate fragment layout")?;
        for _ in 0..num_fragments {
            let merge_mode = r.u8()?;
            if merge_mode > MERGE_ISO {
                bail!("bad merge mode {merge_mode}");
            }
            let num_tensors = r.u32()?;
            if num_tensors == 0 || num_tensors as usize > r.remaining() / 8 {
                bail!("invalid tensor count {num_tensors} in fragment");
            }
            let tensor_count = num_tensors as usize;
            let mut tensor_numels = Vec::new();
            tensor_numels
                .try_reserve_exact(tensor_count)
                .context("cannot allocate tensor layout")?;
            for _ in 0..num_tensors {
                let numel = r.u64()?;
                if numel == 0 {
                    bail!("tensor numel must be positive");
                }
                tensor_numels.push(numel);
            }
            let tensor_shapes = if merge_mode == MERGE_ISO {
                let mut shapes = Vec::new();
                shapes
                    .try_reserve_exact(tensor_count)
                    .context("cannot allocate iso tensor shapes")?;
                for &numel in &tensor_numels {
                    let rows = r.u64()?;
                    let cols = r.u64()?;
                    if rows == 0 || cols == 0 || rows.checked_mul(cols) != Some(numel) {
                        bail!("bad iso tensor shape {rows}x{cols} for numel {numel}");
                    }
                    shapes.push((rows, cols));
                }
                Some(shapes)
            } else {
                None
            };
            fragments.push(FragmentInfo {
                merge_mode,
                tensor_numels,
                tensor_shapes,
            });
        }
        let layout = Layout { fragments };
        for fragment in &layout.fragments {
            fragment.numel()?;
        }
        Ok(layout)
    }
}

/// Cumulative per-learner merge accounting (the "event-tape ledger").
#[derive(Clone, Copy, Default)]
pub struct LearnerLedger {
    pub merges: u64,
    pub steps: u64,
    pub tokens: u64,
}

pub struct GlobalState {
    pub layout: Layout,
    /// Θ_p, flat f32 per fragment (concatenated tensors in layout order).
    pub params: Vec<Vec<f32>>,
    /// Nesterov momentum buffers, same shape as params.
    momentum: Vec<Vec<f32>>,
    pub initialized: Vec<bool>,
    /// Global step at which each fragment was last merged (its version).
    pub versions: Vec<u64>,
    /// Last completed global step t (checkpoint cut point).
    pub global_step: u64,
    pub ledger: std::collections::BTreeMap<u32, LearnerLedger>,
    pub outer_lr: f32,
    pub outer_momentum: f32,
    /// Dtype used on the wire (from HELLO); merge math stays f32.
    pub wire_dtype: u8,
    /// HeLoCo per-tensor directional correction of learner deltas against
    /// the outer momentum before merging (None disables).
    pub delta_correction: Option<merge::Heloco>,
}

impl GlobalState {
    pub fn new(layout: Layout, outer_lr: f32, outer_momentum: f32, wire_dtype: u8) -> Result<Self> {
        // Validate declared sizes now, but do not allocate model-sized state
        // from an unauthenticated HELLO. Params arrive in INIT_PARAMS; the
        // matching momentum is allocated only after that exact-sized payload
        // has passed negotiated bounds and decoded successfully.
        for fragment in &layout.fragments {
            fragment.numel()?;
        }
        let fragment_count = layout.fragments.len();
        let mut params = Vec::new();
        params
            .try_reserve_exact(fragment_count)
            .context("cannot allocate parameter fragment table")?;
        params.resize_with(fragment_count, Vec::new);
        let mut momentum = Vec::new();
        momentum
            .try_reserve_exact(fragment_count)
            .context("cannot allocate momentum fragment table")?;
        momentum.resize_with(fragment_count, Vec::new);
        let mut initialized = Vec::new();
        initialized
            .try_reserve_exact(fragment_count)
            .context("cannot allocate initialization flags")?;
        initialized.resize(fragment_count, false);
        let mut versions = Vec::new();
        versions
            .try_reserve_exact(fragment_count)
            .context("cannot allocate fragment versions")?;
        versions.resize(fragment_count, 0);
        Ok(Self {
            layout,
            params,
            momentum,
            initialized,
            versions,
            global_step: 0,
            ledger: Default::default(),
            outer_lr,
            outer_momentum,
            wire_dtype,
            delta_correction: None,
        })
    }

    pub fn all_initialized(&self) -> bool {
        self.initialized.iter().all(|&b| b)
    }

    pub fn init_fragment(&mut self, fid: usize, values: Vec<f32>) -> Result<()> {
        let expected = self
            .layout
            .fragments
            .get(fid)
            .with_context(|| format!("init for unknown fragment {fid}"))?
            .numel()?;
        if values.len() != expected {
            bail!(
                "init fragment {fid}: got {} values, expected {expected}",
                values.len()
            );
        }
        if !self.initialized[fid] {
            let mut momentum = Vec::new();
            momentum
                .try_reserve_exact(expected)
                .context("cannot allocate fragment momentum")?;
            momentum.resize(expected, 0.0);
            self.params[fid] = values;
            self.momentum[fid] = momentum;
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

    /// Merge learner outer gradients for fragment `fid` and apply the outer step.
    /// Returns the l2 norm of the merged outer gradient (for logging).
    pub fn merge_and_step(
        &mut self,
        fid: usize,
        outer_gradients: &[&[f32]],
        weights: &[f64],
    ) -> Result<f64> {
        let frag = self
            .layout
            .fragments
            .get(fid)
            .with_context(|| format!("merge for unknown fragment {fid}"))?;
        let numel = frag.numel()?;
        for (i, gradient) in outer_gradients.iter().enumerate() {
            if gradient.len() != numel {
                bail!(
                    "push for fragment {fid} from entry {i} has {} values, expected {numel}",
                    gradient.len()
                );
            }
        }
        // HeLoCo: correct each learner's outer delta against the outer
        // momentum, per tensor, before merging (stale deltas can oppose the
        // current global trajectory). Inputs already have outer-gradient
        // sign, so correction operates on copies without reconstructing
        // learner parameters or consulting a parameter anchor.
        let corrected: Vec<Vec<f32>>;
        let outer_gradients: Vec<&[f32]> = if let Some(h) = self.delta_correction {
            let momentum = &self.momentum[fid];
            corrected = outer_gradients
                .iter()
                .map(|gradient| {
                    let mut values = gradient.to_vec();
                    let mut off = 0usize;
                    for &tn in &frag.tensor_numels {
                        let tn = tn as usize;
                        merge::heloco_correct(
                            &mut values[off..off + tn],
                            &momentum[off..off + tn],
                            &h,
                        );
                        off += tn;
                    }
                    values
                })
                .collect();
            corrected.iter().map(|v| v.as_slice()).collect()
        } else {
            outer_gradients.to_vec()
        };
        let outer_gradients = outer_gradients.as_slice();
        let mut delta = vec![0.0f32; numel];
        // Merge per tensor slice within the fragment.
        let mut off = 0usize;
        for (tensor_index, &tn) in frag.tensor_numels.iter().enumerate() {
            let tn = tn as usize;
            let slices: Vec<&[f32]> = outer_gradients
                .iter()
                .map(|gradient| &gradient[off..off + tn])
                .collect();
            let out = &mut delta[off..off + tn];
            match frag.merge_mode {
                MERGE_AVG => merge::merge_avg(&slices, weights, out),
                MERGE_RDA => merge::merge_rda(&slices, weights, out),
                MERGE_ISO => {
                    let (rows, cols) = frag
                        .tensor_shapes
                        .as_ref()
                        .and_then(|shapes| shapes.get(tensor_index))
                        .copied()
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "fragment {fid}: iso merge missing shape for tensor {tensor_index}"
                            )
                        })?;
                    merge::merge_iso(&slices, weights, rows as usize, cols as usize, out);
                }
                mode => bail!("fragment {fid}: unsupported merge mode {mode}"),
            }
            off += tn;
        }
        let gnorm = delta
            .iter()
            .map(|v| (*v as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        if !gnorm.is_finite() || delta.iter().any(|value| !value.is_finite()) {
            bail!("fragment {fid}: merged outer gradient is non-finite");
        }
        merge::nesterov_step(
            &mut self.params[fid],
            &mut self.momentum[fid],
            &delta,
            self.outer_lr,
            self.outer_momentum,
        );
        if self.params[fid].iter().any(|value| !value.is_finite())
            || self.momentum[fid].iter().any(|value| !value.is_finite())
        {
            bail!("fragment {fid}: outer optimizer produced non-finite state");
        }
        Ok(gnorm)
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
            f.flush()?;
            f.get_ref().sync_all()?;
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
            bail!(
                "checkpoint has {np} fragments, layout has {}",
                self.params.len()
            );
        }
        for p in 0..np {
            self.versions[p] = r.u64()?;
            let numel = usize::try_from(r.u64()?)
                .context("checkpoint fragment numel does not fit usize")?;
            let expected = self.layout.fragments[p].numel()?;
            if numel != expected {
                bail!("checkpoint fragment {p} numel {numel} != layout {expected}");
            }
            for slot in [&mut self.params[p], &mut self.momentum[p]] {
                slot.try_reserve_exact(numel)
                    .context("cannot allocate checkpoint fragment")?;
                slot.resize(numel, 0.0);
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
            let l = LearnerLedger {
                merges: r.u64()?,
                steps: r.u64()?,
                tokens: r.u64()?,
            };
            self.ledger.insert(id, l);
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
                FragmentInfo {
                    merge_mode: MERGE_AVG,
                    tensor_numels: vec![4],
                    tensor_shapes: None,
                },
                FragmentInfo {
                    merge_mode: MERGE_RDA,
                    tensor_numels: vec![2, 2],
                    tensor_shapes: None,
                },
            ],
        }
    }

    #[test]
    fn init_once() {
        let mut st = GlobalState::new(layout2(), 0.7, 0.9, crate::protocol::DTYPE_F32).unwrap();
        st.init_fragment(0, vec![1.0; 4]).unwrap();
        st.init_fragment(0, vec![2.0; 4]).unwrap(); // ignored
        assert_eq!(st.params[0], vec![1.0; 4]);
        assert!(!st.all_initialized());
        st.init_fragment(1, vec![0.0; 4]).unwrap();
        assert!(st.all_initialized());
    }

    #[test]
    fn hello_layout_does_not_allocate_model_state_before_init() {
        let layout = Layout {
            fragments: vec![FragmentInfo {
                merge_mode: MERGE_AVG,
                tensor_numels: vec![1_000_000],
                tensor_shapes: None,
            }],
        };
        let st = GlobalState::new(layout, 0.7, 0.9, crate::protocol::DTYPE_F32).unwrap();
        assert!(st.params[0].is_empty());
        assert!(st.momentum[0].is_empty());
    }

    #[test]
    fn merge_moves_toward_learners() {
        let mut st = GlobalState::new(layout2(), 1.0, 0.0, crate::protocol::DTYPE_F32).unwrap();
        st.init_fragment(0, vec![1.0; 4]).unwrap();
        st.init_fragment(1, vec![1.0; 4]).unwrap();
        let outer_gradient = vec![1.0f32; 4];
        let g = st.merge_and_step(0, &[&outer_gradient], &[1.0]).unwrap();
        assert!(g > 0.0);
        // Θ − 1.0·(Θ − θ) = θ
        assert_eq!(st.params[0], vec![0.0; 4]);
    }

    #[test]
    fn layout_decode_reads_iso_shapes_and_validates_them() {
        let mut bytes = Vec::new();
        bytes.push(MERGE_RDA);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&4u64.to_le_bytes());
        bytes.push(MERGE_ISO);
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&4u64.to_le_bytes());
        bytes.extend_from_slice(&6u64.to_le_bytes());
        for dim in [2u64, 2, 2, 3] {
            bytes.extend_from_slice(&dim.to_le_bytes());
        }
        let mut r = crate::protocol::Reader(&bytes);
        let layout = Layout::decode(&mut r, 2).unwrap();
        assert_eq!(layout.fragments[0].merge_mode, MERGE_RDA);
        assert_eq!(layout.fragments[0].tensor_shapes, None);
        assert_eq!(layout.fragments[1].merge_mode, MERGE_ISO);
        assert_eq!(
            layout.fragments[1].tensor_shapes,
            Some(vec![(2, 2), (2, 3)])
        );

        let mut bad = Vec::new();
        bad.push(MERGE_ISO);
        bad.extend_from_slice(&1u32.to_le_bytes());
        bad.extend_from_slice(&4u64.to_le_bytes());
        bad.extend_from_slice(&2u64.to_le_bytes());
        bad.extend_from_slice(&3u64.to_le_bytes());
        let mut r = crate::protocol::Reader(&bad);
        assert!(Layout::decode(&mut r, 1).is_err());

        let mut short = Vec::new();
        short.push(MERGE_ISO);
        short.extend_from_slice(&1u32.to_le_bytes());
        short.extend_from_slice(&4u64.to_le_bytes());
        short.extend_from_slice(&2u64.to_le_bytes());
        let mut r = crate::protocol::Reader(&short);
        assert!(Layout::decode(&mut r, 1).is_err());
    }

    #[test]
    fn iso_fragment_merges_with_flattened_spectrum() {
        let layout = Layout {
            fragments: vec![FragmentInfo {
                merge_mode: MERGE_ISO,
                tensor_numels: vec![4],
                tensor_shapes: Some(vec![(2, 2)]),
            }],
        };
        let mut st = GlobalState::new(layout, 1.0, 0.0, crate::protocol::DTYPE_F32).unwrap();
        st.init_fragment(0, vec![0.0; 4]).unwrap();
        let outer_gradient = [3.0f32, 0.0, 0.0, 1.0];
        let g = st.merge_and_step(0, &[&outer_gradient], &[1.0]).unwrap();
        for (value, expected) in st.params[0].iter().zip([-2.0f32, 0.0, 0.0, -2.0]) {
            assert!((value - expected).abs() < 1e-6, "params {:?}", st.params[0]);
        }
        assert!((g - 2.0 * 2.0f64.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn checkpoint_roundtrip() {
        let dir = std::env::temp_dir().join("yeto-ckpt-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("state.ckpt");
        let mut st = GlobalState::new(layout2(), 0.7, 0.9, crate::protocol::DTYPE_F32).unwrap();
        st.init_fragment(0, vec![1.5; 4]).unwrap();
        st.init_fragment(1, vec![-2.0; 4]).unwrap();
        let outer_gradient = vec![1.5f32; 4];
        st.merge_and_step(0, &[&outer_gradient], &[1.0]).unwrap();
        st.global_step = 7;
        st.versions[0] = 7;
        st.record_merge(3, 12, 4096);
        st.save_checkpoint(&path).unwrap();

        let mut st2 = GlobalState::new(layout2(), 0.7, 0.9, crate::protocol::DTYPE_F32).unwrap();
        st2.load_checkpoint(&path).unwrap();
        assert_eq!(st2.global_step, 7);
        assert_eq!(st2.versions, vec![7, 0]);
        assert_eq!(st2.params, st.params);
        assert!(st2.all_initialized());
        assert_eq!(st2.ledger.get(&3).unwrap().tokens, 4096);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn checkpoint_atomically_replaces_previous_file() {
        let dir = std::env::temp_dir().join(format!("yeto-atomic-ckpt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("state.ckpt");
        std::fs::write(&path, b"old checkpoint bytes").unwrap();

        let mut st = GlobalState::new(layout2(), 0.7, 0.9, crate::protocol::DTYPE_F32).unwrap();
        st.init_fragment(0, vec![3.0; 4]).unwrap();
        st.init_fragment(1, vec![-4.0; 4]).unwrap();
        st.global_step = 13;
        st.versions = vec![12, 13];
        st.save_checkpoint(&path).unwrap();

        let mut restored =
            GlobalState::new(layout2(), 0.7, 0.9, crate::protocol::DTYPE_F32).unwrap();
        restored.load_checkpoint(&path).unwrap();
        assert_eq!(restored.global_step, 13);
        assert_eq!(restored.versions, vec![12, 13]);
        assert!(!path.with_extension("tmp").exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn heloco_correction_damps_anti_aligned_learner() {
        // Two identical states with warm momentum; the learner's delta
        // opposes it. With correction the outer step must move less far
        // in the opposing direction than without.
        let mk = || {
            let mut st = GlobalState::new(layout2(), 1.0, 0.0, crate::protocol::DTYPE_F32).unwrap();
            st.init_fragment(0, vec![0.0; 4]).unwrap();
            st.init_fragment(1, vec![0.0; 4]).unwrap();
            // Warm the momentum with a positive outer gradient.
            st.merge_and_step(0, &[&[1.0f32; 4][..]], &[1.0]).unwrap();
            st
        };
        let mut plain = mk();
        let mut corrected = mk();
        corrected.delta_correction = Some(merge::Heloco::default());
        // Now apply an outer gradient that opposes the warm momentum.
        let opposing = vec![-3.0f32; 4];
        plain.merge_and_step(0, &[&opposing], &[1.0]).unwrap();
        corrected.merge_and_step(0, &[&opposing], &[1.0]).unwrap();
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
        let mut st = GlobalState::new(layout2(), 0.7, 0.9, crate::protocol::DTYPE_F32).unwrap();
        assert!(st.init_fragment(0, vec![1.0; 3]).is_err());
        let learner = vec![0.0f32; 3];
        assert!(st.merge_and_step(0, &[&learner], &[1.0]).is_err());
    }

    #[test]
    fn outer_optimizer_rejects_non_finite_state() {
        let layout = Layout {
            fragments: vec![FragmentInfo {
                merge_mode: MERGE_AVG,
                tensor_numels: vec![1],
                tensor_shapes: None,
            }],
        };
        let mut st = GlobalState::new(layout, f32::MAX, 0.0, crate::protocol::DTYPE_F32).unwrap();
        st.init_fragment(0, vec![1.0]).unwrap();
        let error = st.merge_and_step(0, &[&[f32::MAX]], &[1.0]).unwrap_err();
        assert!(error.to_string().contains("non-finite state"));
    }
}
