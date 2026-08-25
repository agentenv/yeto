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
    /// Semantic tensor identity supplied in HELLO and persisted in checkpoints.
    pub layout_fingerprint: [u8; 32],
    /// Whether a loaded checkpoint carried and matched the fingerprint.
    pub checkpoint_layout_verified: bool,
    /// Strict dense-policy sweep profile carried by the checkpoint trailer.
    /// `None` is the byte-for-byte legacy checkpoint profile.
    pub policy_sweep_fragments: Option<u32>,
    /// Opaque semantic/profile/roster identity supplied by every learner.
    /// Required and checkpointed only for strict dense-policy sweeps.
    pub session_contract_hash: Option<[u8; 32]>,
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
    #[cfg(test)]
    pub fn new(layout: Layout, outer_lr: f32, outer_momentum: f32, wire_dtype: u8) -> Result<Self> {
        Self::new_with_layout_fingerprint(layout, outer_lr, outer_momentum, wire_dtype, [0; 32])
    }

    pub fn new_with_layout_fingerprint(
        layout: Layout,
        outer_lr: f32,
        outer_momentum: f32,
        wire_dtype: u8,
        layout_fingerprint: [u8; 32],
    ) -> Result<Self> {
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
            layout_fingerprint,
            checkpoint_layout_verified: false,
            policy_sweep_fragments: None,
            session_contract_hash: None,
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
        self.record_fragment_merge(learner_id, c_steps, c_tokens, true);
    }

    /// Record one fragment merge, charging local optimizer progress only at
    /// the atomic accounting boundary selected by the scheduler.
    pub fn record_fragment_merge(
        &mut self,
        learner_id: u32,
        c_steps: u32,
        c_tokens: u64,
        account_local_progress: bool,
    ) {
        let e = self.ledger.entry(learner_id).or_default();
        e.merges += 1;
        if account_local_progress {
            e.steps += c_steps as u64;
            e.tokens += c_tokens;
        }
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
/// Policy-sweep checkpoint marker, followed by the configured fragment count.
const POLICY_SWEEP_CKPT_MAGIC: u32 = 0x5053_5750;
const FINAL_MARKER_MAGIC: &str = "YETO_FINAL_V1";

pub fn final_marker_path(path: &std::path::Path) -> std::path::PathBuf {
    let mut value = path.as_os_str().to_os_string();
    value.push(".final");
    value.into()
}

pub fn remove_final_marker(path: &std::path::Path) -> Result<()> {
    let marker = final_marker_path(path);
    match std::fs::remove_file(&marker) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(|| format!("remove {}", marker.display())),
    }
}

pub fn write_final_marker(path: &std::path::Path, global_step: u64) -> Result<()> {
    use std::io::Write;

    let marker = final_marker_path(path);
    let mut tmp_value = marker.as_os_str().to_os_string();
    tmp_value.push(".tmp");
    let tmp = std::path::PathBuf::from(tmp_value);
    {
        let mut file = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
        write!(file, "{FINAL_MARKER_MAGIC}\nglobal_step={global_step}\n")?;
        file.flush()?;
        file.get_ref().sync_all()?;
    }
    std::fs::rename(&tmp, &marker)?;
    Ok(())
}

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
            f.write_all(&self.layout_fingerprint)?;
            if let Some(fragments) = self.policy_sweep_fragments {
                f.write_all(&POLICY_SWEEP_CKPT_MAGIC.to_le_bytes())?;
                f.write_all(&fragments.to_le_bytes())?;
                f.write_all(
                    &self
                        .session_contract_hash
                        .context("policy-sweep checkpoint requires a session contract hash")?,
                )?;
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
        let expected_policy_sweep_fragments = self.policy_sweep_fragments;
        let (checkpoint_policy_sweep_fragments, checkpoint_session_contract_hash) =
            match r.remaining() {
                0 => {
                    self.checkpoint_layout_verified = false;
                    (None, None)
                }
                32 => {
                    let checkpoint_fingerprint: [u8; 32] = r.take(32)?.try_into()?;
                    if checkpoint_fingerprint != self.layout_fingerprint {
                        bail!("checkpoint layout fingerprint does not match HELLO");
                    }
                    self.checkpoint_layout_verified = true;
                    (None, None)
                }
                40 => {
                    let checkpoint_fingerprint: [u8; 32] = r.take(32)?.try_into()?;
                    if checkpoint_fingerprint != self.layout_fingerprint {
                        bail!("checkpoint layout fingerprint does not match HELLO");
                    }
                    if r.u32()? != POLICY_SWEEP_CKPT_MAGIC {
                        bail!("checkpoint has an invalid policy-sweep trailer");
                    }
                    let fragments = r.u32()?;
                    if fragments == 0 {
                        bail!("checkpoint policy-sweep fragment count must be positive");
                    }
                    bail!("policy-sweep checkpoint is missing its session contract hash");
                }
                72 => {
                    let checkpoint_fingerprint: [u8; 32] = r.take(32)?.try_into()?;
                    if checkpoint_fingerprint != self.layout_fingerprint {
                        bail!("checkpoint layout fingerprint does not match HELLO");
                    }
                    if r.u32()? != POLICY_SWEEP_CKPT_MAGIC {
                        bail!("checkpoint has an invalid policy-sweep trailer");
                    }
                    let fragments = r.u32()?;
                    if fragments == 0 {
                        bail!("checkpoint policy-sweep fragment count must be positive");
                    }
                    let contract_hash = r.take(32)?.try_into()?;
                    self.checkpoint_layout_verified = true;
                    (Some(fragments), Some(contract_hash))
                }
                remaining => bail!("checkpoint has {remaining} trailing bytes"),
            };
        if checkpoint_policy_sweep_fragments != expected_policy_sweep_fragments {
            bail!(
                "checkpoint policy-sweep profile {:?} does not match configured profile {:?}",
                checkpoint_policy_sweep_fragments,
                expected_policy_sweep_fragments
            );
        }
        if checkpoint_session_contract_hash != self.session_contract_hash {
            bail!("checkpoint session contract hash does not match HELLO");
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
        assert!(st2.checkpoint_layout_verified);
        assert_eq!(st2.ledger.get(&3).unwrap().tokens, 4096);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn policy_sweep_checkpoint_trailer_is_explicit_and_legacy_bytes_are_unchanged() {
        let dir = std::env::temp_dir().join(format!(
            "yeto-policy-sweep-ckpt-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let legacy_path = dir.join("legacy.ckpt");
        let sweep_path = dir.join("sweep.ckpt");

        let mut state = GlobalState::new_with_layout_fingerprint(
            layout2(),
            0.7,
            0.9,
            crate::protocol::DTYPE_F32,
            [9; 32],
        )
        .unwrap();
        state.init_fragment(0, vec![1.0; 4]).unwrap();
        state.init_fragment(1, vec![2.0; 4]).unwrap();
        state.save_checkpoint(&legacy_path).unwrap();
        let legacy = std::fs::read(&legacy_path).unwrap();

        state.policy_sweep_fragments = Some(2);
        state.session_contract_hash = Some([7; 32]);
        state.save_checkpoint(&sweep_path).unwrap();
        let sweep = std::fs::read(&sweep_path).unwrap();
        assert_eq!(&sweep[..legacy.len()], legacy.as_slice());
        assert_eq!(sweep.len(), legacy.len() + 40);
        assert_eq!(
            &sweep[legacy.len()..legacy.len() + 4],
            &POLICY_SWEEP_CKPT_MAGIC.to_le_bytes()
        );
        assert_eq!(
            &sweep[legacy.len() + 4..legacy.len() + 8],
            &2u32.to_le_bytes()
        );
        assert_eq!(&sweep[legacy.len() + 8..], &[7; 32]);

        let mut restored = GlobalState::new_with_layout_fingerprint(
            layout2(),
            0.7,
            0.9,
            crate::protocol::DTYPE_F32,
            [9; 32],
        )
        .unwrap();
        restored.policy_sweep_fragments = Some(2);
        restored.session_contract_hash = Some([7; 32]);
        restored.load_checkpoint(&sweep_path).unwrap();
        assert_eq!(restored.policy_sweep_fragments, Some(2));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn checkpoint_resume_rejects_legacy_sweep_and_fragment_profile_mismatches() {
        let dir = std::env::temp_dir().join(format!(
            "yeto-policy-sweep-mismatch-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let legacy_path = dir.join("legacy.ckpt");
        let sweep_path = dir.join("sweep.ckpt");

        let make_state = || {
            let mut state = GlobalState::new_with_layout_fingerprint(
                layout2(),
                0.7,
                0.9,
                crate::protocol::DTYPE_F32,
                [5; 32],
            )
            .unwrap();
            state.init_fragment(0, vec![1.0; 4]).unwrap();
            state.init_fragment(1, vec![2.0; 4]).unwrap();
            state
        };
        let legacy = make_state();
        legacy.save_checkpoint(&legacy_path).unwrap();
        let mut sweep = make_state();
        sweep.policy_sweep_fragments = Some(2);
        sweep.session_contract_hash = Some([5; 32]);
        sweep.save_checkpoint(&sweep_path).unwrap();

        let mut configured_sweep = make_state();
        configured_sweep.policy_sweep_fragments = Some(2);
        configured_sweep.session_contract_hash = Some([5; 32]);
        let error = configured_sweep.load_checkpoint(&legacy_path).unwrap_err();
        assert!(format!("{error:#}").contains("does not match configured profile"));

        let mut configured_legacy = make_state();
        let error = configured_legacy.load_checkpoint(&sweep_path).unwrap_err();
        assert!(format!("{error:#}").contains("does not match configured profile"));

        let mut configured_three = make_state();
        configured_three.policy_sweep_fragments = Some(3);
        configured_three.session_contract_hash = Some([5; 32]);
        let error = configured_three.load_checkpoint(&sweep_path).unwrap_err();
        assert!(format!("{error:#}").contains("does not match configured profile"));

        let mut malformed = std::fs::read(&sweep_path).unwrap();
        let magic_offset = malformed.len() - 40;
        malformed[magic_offset..magic_offset + 4].copy_from_slice(b"NOPE");
        let malformed_path = dir.join("malformed.ckpt");
        std::fs::write(&malformed_path, malformed).unwrap();
        let mut configured_sweep = make_state();
        configured_sweep.policy_sweep_fragments = Some(2);
        configured_sweep.session_contract_hash = Some([5; 32]);
        let error = configured_sweep
            .load_checkpoint(&malformed_path)
            .unwrap_err();
        assert!(format!("{error:#}").contains("invalid policy-sweep trailer"));

        let mut wrong_semantics = make_state();
        wrong_semantics.policy_sweep_fragments = Some(2);
        wrong_semantics.session_contract_hash = Some([6; 32]);
        let error = wrong_semantics.load_checkpoint(&sweep_path).unwrap_err();
        assert!(format!("{error:#}").contains("session contract hash"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn policy_sweep_accounts_every_merge_and_local_progress_only_at_sweep_end() {
        let mut state = GlobalState::new(layout2(), 1.0, 0.0, crate::protocol::DTYPE_F32).unwrap();
        state.record_fragment_merge(7, 1, 123, false);
        let partial = state.ledger.get(&7).unwrap();
        assert_eq!(partial.merges, 1);
        assert_eq!(partial.steps, 0);
        assert_eq!(partial.tokens, 0);

        state.record_fragment_merge(7, 1, 123, true);
        let complete = state.ledger.get(&7).unwrap();
        assert_eq!(complete.merges, 2);
        assert_eq!(complete.steps, 1);
        assert_eq!(complete.tokens, 123);

        // The legacy API remains one merge + one progress charge.
        state.record_merge(8, 2, 456);
        let legacy = state.ledger.get(&8).unwrap();
        assert_eq!(legacy.merges, 1);
        assert_eq!(legacy.steps, 2);
        assert_eq!(legacy.tokens, 456);
    }

    #[test]
    fn checkpoint_layout_fingerprint_is_verified_and_legacy_is_readable() {
        let dir = std::env::temp_dir().join(format!("yeto-layout-ckpt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("state.ckpt");
        let mut state = GlobalState::new_with_layout_fingerprint(
            layout2(),
            0.7,
            0.9,
            crate::protocol::DTYPE_F32,
            [1; 32],
        )
        .unwrap();
        state.init_fragment(0, vec![1.0; 4]).unwrap();
        state.init_fragment(1, vec![2.0; 4]).unwrap();
        state.save_checkpoint(&path).unwrap();

        let mut mismatched = GlobalState::new_with_layout_fingerprint(
            layout2(),
            0.7,
            0.9,
            crate::protocol::DTYPE_F32,
            [2; 32],
        )
        .unwrap();
        assert!(mismatched.load_checkpoint(&path).is_err());

        let mut bytes = std::fs::read(&path).unwrap();
        bytes.truncate(bytes.len() - 32);
        std::fs::write(&path, bytes).unwrap();
        let mut legacy = GlobalState::new_with_layout_fingerprint(
            layout2(),
            0.7,
            0.9,
            crate::protocol::DTYPE_F32,
            [2; 32],
        )
        .unwrap();
        legacy.load_checkpoint(&path).unwrap();
        assert!(!legacy.checkpoint_layout_verified);
        std::fs::remove_dir_all(&dir).ok();
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
    fn final_marker_uses_adjacent_atomic_file_and_exact_content() {
        let dir = std::env::temp_dir().join(format!("yeto-final-marker-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let checkpoint = dir.join("state.ckpt");
        let marker = final_marker_path(&checkpoint);
        write_final_marker(&checkpoint, 23).unwrap();
        assert_eq!(
            std::fs::read_to_string(&marker).unwrap(),
            "YETO_FINAL_V1\nglobal_step=23\n"
        );
        assert!(!std::path::PathBuf::from(format!("{}.tmp", marker.display())).exists());
        remove_final_marker(&checkpoint).unwrap();
        assert!(!marker.exists());
        remove_final_marker(&checkpoint).unwrap();
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn failed_marker_write_never_publishes_the_final_path() {
        let dir =
            std::env::temp_dir().join(format!("yeto-final-marker-failure-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let checkpoint = dir.join("state.ckpt");
        let marker = final_marker_path(&checkpoint);
        std::fs::create_dir(format!("{}.tmp", marker.display())).unwrap();

        assert!(write_final_marker(&checkpoint, 23).is_err());
        assert!(!marker.exists());
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
