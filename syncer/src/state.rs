//! Global model state held by the syncer: fragment layout, parameters Θ,
//! and outer-optimizer momentum, all in f32.

use anyhow::{bail, Context, Result};
use std::io::{BufReader, Read};

use crate::iso_worker::{IsoBackend, IsoBackendConfig, IsoBackendKind};
use crate::merge;
use crate::protocol::Reader;

// Checkpoints can be hundreds of gigabytes. Keep deserialization scratch
// bounded instead of reading the complete file into a second model-sized
// allocation before constructing the restored parameter and momentum state.
const CHECKPOINT_READ_CHUNK_BYTES: usize = 8 * 1024 * 1024;

struct CheckpointReader {
    inner: BufReader<std::fs::File>,
    remaining: u64,
}

impl CheckpointReader {
    fn open(path: &std::path::Path) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let remaining = file.metadata()?.len();
        Ok(Self {
            inner: BufReader::new(file),
            remaining,
        })
    }

    fn read_exact(&mut self, bytes: &mut [u8]) -> Result<()> {
        let count = u64::try_from(bytes.len()).context("checkpoint read size does not fit u64")?;
        if count > self.remaining {
            bail!("payload truncated");
        }
        self.inner.read_exact(bytes)?;
        self.remaining -= count;
        Ok(())
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut bytes = [0; N];
        self.read_exact(&mut bytes)?;
        Ok(bytes)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.array::<1>()?[0])
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn remaining(&self) -> u64 {
        self.remaining
    }

    fn require_bytes(&self, count: usize) -> Result<()> {
        if u64::try_from(count).context("checkpoint payload size does not fit u64")?
            > self.remaining
        {
            bail!("payload truncated");
        }
        Ok(())
    }

    fn f32s(&mut self, count: usize, scratch: &mut [u8]) -> Result<Vec<f32>> {
        if scratch.len() < 4 || scratch.len() % 4 != 0 {
            bail!("checkpoint f32 scratch must be a non-empty multiple of four bytes");
        }
        let byte_count = count
            .checked_mul(4)
            .context("checkpoint f32 payload size overflow")?;
        self.require_bytes(byte_count)?;

        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .context("cannot allocate checkpoint fragment")?;
        while values.len() < count {
            let chunk_values = (count - values.len()).min(scratch.len() / 4);
            let chunk_bytes = chunk_values * 4;
            self.read_exact(&mut scratch[..chunk_bytes])?;
            values.extend(
                scratch[..chunk_bytes]
                    .chunks_exact(4)
                    .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap())),
            );
        }
        Ok(values)
    }
}

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
    /// `None` means the generic streaming profile; its session contract is
    /// persisted independently below.
    pub policy_sweep_fragments: Option<u32>,
    /// Opaque semantic/profile/roster identity supplied by every learner.
    /// Server-created states always carry and checkpoint this identity. Direct
    /// callers may leave it unset only to read or write the legacy format.
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
    /// Spectrum flattener. Torch mode owns one persistent child process.
    iso_backend: IsoBackend,
}

/// A merge whose deterministic learner reduction has completed but whose
/// Torch Iso polar matrices may still be executing.  It owns every input
/// buffer used by the workers, so no matrix can alias coordinator state.
pub struct PreparedMerge {
    fid: usize,
    base_version: u64,
    delta: Vec<f32>,
    iso_jobs: Vec<IsoJob>,
    iso_backend: IsoBackend,
}

struct IsoJob {
    tensor_index: usize,
    offset: usize,
    rows: usize,
    cols: usize,
    matrix: Vec<f32>,
}

/// Immutable worker output waiting for the single coordinator to commit it.
/// Nesterov state, versions, the event tape, and broadcasts are deliberately
/// absent: those are changed only by `GlobalState::commit_merge`, in t order.
pub struct ComputedMerge {
    fid: usize,
    base_version: u64,
    delta: Vec<f32>,
}

impl ComputedMerge {
    pub fn fid(&self) -> usize {
        self.fid
    }

    pub fn base_version(&self) -> u64 {
        self.base_version
    }
}

impl PreparedMerge {
    pub fn fid(&self) -> usize {
        self.fid
    }

    pub fn base_version(&self) -> u64 {
        self.base_version
    }

    pub async fn compute(mut self) -> Result<ComputedMerge> {
        let mut tasks = tokio::task::JoinSet::new();
        let fid = self.fid;
        for job in self.iso_jobs.drain(..) {
            let backend = self.iso_backend.clone();
            tasks.spawn(async move {
                let matrix = backend
                    .flatten_owned(job.matrix, job.rows, job.cols)
                    .await
                    .with_context(|| {
                        format!(
                            "fragment {}: iso backend failed for tensor {} ({}x{})",
                            fid, job.tensor_index, job.rows, job.cols
                        )
                    })?;
                Ok::<_, anyhow::Error>((job.offset, matrix))
            });
        }
        while let Some(result) = tasks.join_next().await {
            let (offset, matrix) = result.context("iso matrix task panicked")??;
            if self.delta.is_empty() && offset == 0 {
                self.delta = matrix;
                continue;
            }
            let end = offset
                .checked_add(matrix.len())
                .context("iso result offset overflow")?;
            let out = self
                .delta
                .get_mut(offset..end)
                .context("iso result lies outside prepared fragment")?;
            out.copy_from_slice(&matrix);
        }
        if self.delta.iter().any(|value| !value.is_finite()) {
            bail!("fragment {}: merged outer gradient is non-finite", self.fid);
        }
        Ok(ComputedMerge {
            fid: self.fid,
            base_version: self.base_version,
            delta: self.delta,
        })
    }

    fn finish_inline(self) -> Result<ComputedMerge> {
        if !self.iso_jobs.is_empty() {
            bail!("torch-svd prepared merge requires asynchronous compute");
        }
        Ok(ComputedMerge {
            fid: self.fid,
            base_version: self.base_version,
            delta: self.delta,
        })
    }
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
        Self::new_inner(
            layout,
            outer_lr,
            outer_momentum,
            wire_dtype,
            layout_fingerprint,
            &IsoBackendConfig::default(),
        )
    }

    pub fn new_with_iso_backend(
        layout: Layout,
        outer_lr: f32,
        outer_momentum: f32,
        wire_dtype: u8,
        layout_fingerprint: [u8; 32],
        iso_backend_config: &IsoBackendConfig,
    ) -> Result<Self> {
        Self::new_inner(
            layout,
            outer_lr,
            outer_momentum,
            wire_dtype,
            layout_fingerprint,
            iso_backend_config,
        )
    }

    fn new_inner(
        layout: Layout,
        outer_lr: f32,
        outer_momentum: f32,
        wire_dtype: u8,
        layout_fingerprint: [u8; 32],
        iso_backend_config: &IsoBackendConfig,
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
            iso_backend: IsoBackend::start(iso_backend_config)?,
        })
    }

    pub fn iso_backend_kind(&self) -> IsoBackendKind {
        self.iso_backend.kind()
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

    /// Deterministically reduce learner gradients and enqueue no mutable
    /// coordinator state.  Torch Iso work is represented as owned matrices
    /// inside the returned value and may execute out of order.
    pub fn prepare_merge(
        &self,
        fid: usize,
        base_version: u64,
        outer_gradients: &[&[f32]],
        weights: &[f64],
    ) -> Result<PreparedMerge> {
        let current_version = *self
            .versions
            .get(fid)
            .with_context(|| format!("merge for unknown fragment {fid}"))?;
        if current_version != base_version {
            bail!("fragment {fid}: prepare version {base_version} != current {current_version}");
        }
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
        let mut iso_jobs = Vec::new();
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
                    // The syncer owns the deterministic weighted reduction;
                    // the selected backend receives exactly one complete
                    // canonical matrix, never learner or TP shards.
                    merge::merge_avg(&slices, weights, out);
                    match self.iso_backend.kind() {
                        IsoBackendKind::Scalar => {
                            merge::iso_flatten_spectrum(out, rows as usize, cols as usize)
                        }
                        IsoBackendKind::TorchSvd => iso_jobs.push(IsoJob {
                            tensor_index,
                            offset: off,
                            rows: rows as usize,
                            cols: cols as usize,
                            matrix: out.to_vec(),
                        }),
                    }
                }
                mode => bail!("fragment {fid}: unsupported merge mode {mode}"),
            }
            off += tn;
        }
        // Miles uses one canonical tensor per fragment. Move that complete
        // averaged matrix into its worker job instead of retaining a second
        // model-sized host copy while the Iso worker runs.
        if iso_jobs.len() == 1 && iso_jobs[0].offset == 0 && iso_jobs[0].matrix.len() == delta.len()
        {
            iso_jobs[0].matrix = std::mem::take(&mut delta);
        }
        if delta.iter().any(|value| !value.is_finite()) {
            bail!("fragment {fid}: merged outer gradient is non-finite");
        }
        Ok(PreparedMerge {
            fid,
            base_version,
            delta,
            iso_jobs,
            iso_backend: self.iso_backend.clone(),
        })
    }

    /// Commit one fully computed merge.  This is the only merge path that
    /// mutates Nesterov state and it rejects stale/out-of-order fragment work.
    pub fn commit_merge(&mut self, computed: ComputedMerge) -> Result<f64> {
        let ComputedMerge {
            fid,
            base_version,
            delta,
        } = computed;
        let current_version = *self
            .versions
            .get(fid)
            .with_context(|| format!("commit for unknown fragment {fid}"))?;
        if current_version != base_version {
            bail!(
                "fragment {fid}: computed base version {base_version} != current {current_version}"
            );
        }
        let expected = self.layout.fragments[fid].numel()?;
        if delta.len() != expected {
            bail!(
                "fragment {fid}: computed merge has {} values, expected {expected}",
                delta.len()
            );
        }
        let gnorm = delta
            .iter()
            .map(|v| (*v as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        if !gnorm.is_finite() || delta.iter().any(|value| !value.is_finite()) {
            bail!("fragment {fid}: merged outer gradient is non-finite");
        }
        // Validate the complete optimizer result before mutating either
        // buffer. This keeps commit all-or-nothing without cloning a
        // model-sized fragment.
        for ((param, momentum), gradient) in
            self.params[fid].iter().zip(&self.momentum[fid]).zip(&delta)
        {
            let next_momentum = self.outer_momentum * *momentum + *gradient;
            let next_param =
                *param - self.outer_lr * (*gradient + self.outer_momentum * next_momentum);
            if !next_param.is_finite() || !next_momentum.is_finite() {
                bail!("fragment {fid}: outer optimizer produced non-finite state");
            }
        }
        merge::nesterov_step(
            &mut self.params[fid],
            &mut self.momentum[fid],
            &delta,
            self.outer_lr,
            self.outer_momentum,
        );
        debug_assert!(self.params[fid].iter().all(|value| value.is_finite()));
        debug_assert!(self.momentum[fid].iter().all(|value| value.is_finite()));
        Ok(gnorm)
    }

    /// Compatibility helper for scalar unit tests and non-Torch callers.
    pub fn merge_and_step(
        &mut self,
        fid: usize,
        outer_gradients: &[&[f32]],
        weights: &[f64],
    ) -> Result<f64> {
        let prepared = self.prepare_merge(fid, self.versions[fid], outer_gradients, weights)?;
        self.commit_merge(prepared.finish_inline()?)
    }

    /// Explicit barrier used before any terminal checkpoint/marker is made
    /// publishable.  The worker pool reports poison instead of silently
    /// publishing a cut after partial Iso worker failure.
    pub async fn drain_iso_backend(&self) -> Result<()> {
        self.iso_backend.drain().await
    }
}

const CKPT_MAGIC_V1: u32 = 0xD170_5A7E;
const CKPT_MAGIC_V2: u32 = 0xD170_5A7F;
const CKPT_MAGIC_V3: u32 = 0xD170_5A80;
/// Generic streaming checkpoint marker, followed by the session contract hash.
const SESSION_CONTRACT_CKPT_MAGIC: u32 = 0x5254_4353;
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
        Ok(()) => sync_parent_directory(&marker),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(|| format!("remove {}", marker.display())),
    }
}

fn sync_parent_directory(path: &std::path::Path) -> Result<()> {
    let parent = path.parent().unwrap_or_else(|| std::path::Path::new("."));
    std::fs::File::open(parent)
        .with_context(|| format!("open checkpoint directory {}", parent.display()))?
        .sync_all()
        .with_context(|| format!("sync checkpoint directory {}", parent.display()))
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
    sync_parent_directory(&marker)?;
    Ok(())
}

impl GlobalState {
    /// Persist a consistent snapshot. Called only at the quiescent cut
    /// between rounds (see docs/PROTOCOL.md "Consistent snapshots").
    /// Written to `<path>.tmp` then renamed, so a crash mid-write never
    /// corrupts the previous checkpoint.
    pub fn save_checkpoint(&self, path: &std::path::Path) -> Result<()> {
        use std::io::Write;
        let layout_fingerprint = self.layout_fingerprint;
        let tmp = path.with_extension("tmp");
        {
            let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
            f.write_all(&CKPT_MAGIC_V3.to_le_bytes())?;
            f.write_all(&[self.iso_backend_kind() as u8])?;
            f.write_all(&layout_fingerprint)?;
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
            } else if let Some(contract_hash) = self.session_contract_hash {
                f.write_all(&SESSION_CONTRACT_CKPT_MAGIC.to_le_bytes())?;
                f.write_all(&contract_hash)?;
            }
            f.flush()?;
            f.get_ref().sync_all()?;
        }
        std::fs::rename(&tmp, path)?;
        sync_parent_directory(path)?;
        Ok(())
    }

    /// Restore params/momentum/versions/step/ledger from a snapshot.
    ///
    /// V3 binds the snapshot to both the ISO backend and exact semantic HELLO
    /// layout. V2 carries only the backend identity. V1 is rejected because
    /// historical scalar and Torch checkpoints are indistinguishable. The
    /// complete checkpoint is parsed and validated before any live state is
    /// mutated.
    pub fn load_checkpoint(&mut self, path: &std::path::Path) -> Result<()> {
        let mut r = CheckpointReader::open(path)?;
        let mut scratch = vec![0; CHECKPOINT_READ_CHUNK_BYTES];
        let magic = r.u32()?;
        let (checkpoint_iso_backend, checkpoint_layout_fingerprint) = match magic {
            CKPT_MAGIC_V1 => bail!(
                "checkpoint V1 cannot be resumed safely: it records neither the ISO backend nor the semantic layout fingerprint"
            ),
            CKPT_MAGIC_V2 => (Some(IsoBackendKind::from_checkpoint(r.u8()?)?), None),
            CKPT_MAGIC_V3 => {
                let backend = IsoBackendKind::from_checkpoint(r.u8()?)?;
                let fingerprint = r.array::<32>()?;
                (Some(backend), Some(fingerprint))
            }
            _ => bail!("bad checkpoint magic"),
        };
        if let Some(checkpoint_iso_backend) = checkpoint_iso_backend {
            if checkpoint_iso_backend != self.iso_backend_kind() {
                bail!(
                    "checkpoint iso backend {} does not match configured backend {}",
                    checkpoint_iso_backend,
                    self.iso_backend_kind()
                );
            }
        }
        if let Some(checkpoint_fingerprint) = checkpoint_layout_fingerprint {
            if checkpoint_fingerprint != self.layout_fingerprint {
                bail!(
                    "checkpoint semantic layout fingerprint does not match the accepted HELLO layout"
                );
            }
        }

        let global_step = r.u64()?;
        let np = r.u32()? as usize;
        if np != self.params.len() {
            bail!(
                "checkpoint has {np} fragments, layout has {}",
                self.params.len()
            );
        }
        let mut versions = Vec::with_capacity(np);
        let mut params = Vec::with_capacity(np);
        let mut momentum = Vec::with_capacity(np);
        for p in 0..np {
            versions.push(r.u64()?);
            let numel = usize::try_from(r.u64()?)
                .context("checkpoint fragment numel does not fit usize")?;
            let expected = self.layout.fragments[p].numel()?;
            if numel != expected {
                bail!("checkpoint fragment {p} numel {numel} != layout {expected}");
            }
            // Preflight both model-sized arrays before allocating either one.
            // A truncated/corrupt checkpoint therefore cannot strand a full
            // parameter fragment allocation while discovering that its
            // matching momentum payload is absent.
            let fragment_state_bytes = numel
                .checked_mul(8)
                .context("checkpoint fragment state size overflow")?;
            r.require_bytes(fragment_state_bytes)?;
            let restored_params = r.f32s(numel, &mut scratch)?;
            let restored_momentum = r.f32s(numel, &mut scratch)?;
            params.push(restored_params);
            momentum.push(restored_momentum);
        }
        let nl = r.u32()? as usize;
        let mut ledger = std::collections::BTreeMap::new();
        for _ in 0..nl {
            let id = r.u32()?;
            let l = LearnerLedger {
                merges: r.u64()?,
                steps: r.u64()?,
                tokens: r.u64()?,
            };
            if ledger.insert(id, l).is_some() {
                bail!("checkpoint contains duplicate ledger entry for learner {id}");
            }
        }
        let expected_policy_sweep_fragments = self.policy_sweep_fragments;
        let header_layout_verified = checkpoint_layout_fingerprint.is_some();
        let restored_layout_verified;
        let (checkpoint_policy_sweep_fragments, checkpoint_session_contract_hash) =
            match r.remaining() {
                0 => {
                    restored_layout_verified = header_layout_verified;
                    (None, None)
                }
                32 => {
                    let checkpoint_fingerprint = r.array::<32>()?;
                    if checkpoint_fingerprint != self.layout_fingerprint {
                        bail!("checkpoint layout fingerprint does not match HELLO");
                    }
                    restored_layout_verified = true;
                    (None, None)
                }
                40 => {
                    let checkpoint_fingerprint = r.array::<32>()?;
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
                68 => {
                    let checkpoint_fingerprint = r.array::<32>()?;
                    if checkpoint_fingerprint != self.layout_fingerprint {
                        bail!("checkpoint layout fingerprint does not match HELLO");
                    }
                    if r.u32()? != SESSION_CONTRACT_CKPT_MAGIC {
                        bail!("checkpoint has an invalid session-contract trailer");
                    }
                    let contract_hash = r.array::<32>()?;
                    restored_layout_verified = true;
                    (None, Some(contract_hash))
                }
                72 => {
                    let checkpoint_fingerprint = r.array::<32>()?;
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
                    let contract_hash = r.array::<32>()?;
                    restored_layout_verified = true;
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
        match (checkpoint_session_contract_hash, self.session_contract_hash) {
            (actual, expected) if actual == expected => {}
            // A layout-only generic checkpoint predates explicit contract
            // persistence. Its only safe implied contract is exactly the
            // verified layout fingerprint used by legacy clients.
            (None, Some(expected))
                if restored_layout_verified
                    && expected == self.layout_fingerprint
                    && expected_policy_sweep_fragments.is_none() => {}
            (None, Some(_)) => {
                bail!("checkpoint is missing the session contract hash required by HELLO");
            }
            _ => bail!("checkpoint session contract hash does not match HELLO"),
        }
        if !restored_layout_verified
            && self
                .layout
                .fragments
                .iter()
                .any(|fragment| fragment.tensor_numels.len() != 1)
        {
            bail!(
                "checkpoint has no semantic layout fingerprint and cannot resume a grouped multi-tensor layout"
            );
        }

        // A strict-order checkpoint's global_step is the contiguous commit
        // cursor, so round-robin fragment versions are fully determined by
        // it. Reject legacy completion-order snapshots with holes rather than
        // silently skipping an uncommitted low t on resume.
        let fragments = versions.len() as u64;
        for (fid, version) in versions.iter().copied().enumerate() {
            let first = fid as u64 + 1;
            let expected = if global_step < first {
                0
            } else {
                first + ((global_step - first) / fragments) * fragments
            };
            if version != expected {
                bail!(
                    "checkpoint is not a contiguous strict-order cut: fragment {fid} version {version}, expected {expected} at global_step {}",
                    global_step
                );
            }
        }

        self.global_step = global_step;
        self.versions = versions;
        self.params = params;
        self.momentum = momentum;
        self.initialized.fill(true);
        self.ledger = ledger;
        self.checkpoint_layout_verified = restored_layout_verified;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fingerprint(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    fn state_with_fingerprint(layout: Layout, byte: u8) -> GlobalState {
        GlobalState::new_with_layout_fingerprint(
            layout,
            0.7,
            0.9,
            crate::protocol::DTYPE_F32,
            fingerprint(byte),
        )
        .unwrap()
    }

    fn write_v2_checkpoint(path: &std::path::Path, st: &GlobalState) {
        use std::io::Write;

        let mut f = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
        f.write_all(&CKPT_MAGIC_V2.to_le_bytes()).unwrap();
        f.write_all(&[st.iso_backend_kind() as u8]).unwrap();
        f.write_all(&st.global_step.to_le_bytes()).unwrap();
        f.write_all(&(st.params.len() as u32).to_le_bytes())
            .unwrap();
        for p in 0..st.params.len() {
            f.write_all(&st.versions[p].to_le_bytes()).unwrap();
            f.write_all(&(st.params[p].len() as u64).to_le_bytes())
                .unwrap();
            for value in &st.params[p] {
                f.write_all(&value.to_le_bytes()).unwrap();
            }
            for value in &st.momentum[p] {
                f.write_all(&value.to_le_bytes()).unwrap();
            }
        }
        f.write_all(&(st.ledger.len() as u32).to_le_bytes())
            .unwrap();
        for (id, ledger) in &st.ledger {
            f.write_all(&id.to_le_bytes()).unwrap();
            f.write_all(&ledger.merges.to_le_bytes()).unwrap();
            f.write_all(&ledger.steps.to_le_bytes()).unwrap();
            f.write_all(&ledger.tokens.to_le_bytes()).unwrap();
        }
        f.flush().unwrap();
    }

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
    fn prepared_merge_checks_version_at_prepare_and_commit() {
        let mut st = GlobalState::new(layout2(), 1.0, 0.0, crate::protocol::DTYPE_F32).unwrap();
        st.init_fragment(0, vec![1.0; 4]).unwrap();
        st.init_fragment(1, vec![1.0; 4]).unwrap();
        let gradient = [1.0f32; 4];

        assert!(st.prepare_merge(0, 9, &[&gradient], &[1.0]).is_err());
        let computed = st
            .prepare_merge(0, 0, &[&gradient], &[1.0])
            .unwrap()
            .finish_inline()
            .unwrap();
        st.versions = vec![7, 6];
        assert!(st.commit_merge(computed).is_err());
        assert_eq!(st.params[0], vec![1.0; 4]);
    }

    #[test]
    fn nonfinite_nesterov_result_does_not_partially_mutate_state() {
        let mut st =
            GlobalState::new(layout2(), f32::MAX, 0.0, crate::protocol::DTYPE_F32).unwrap();
        st.init_fragment(0, vec![1.0; 4]).unwrap();
        st.init_fragment(1, vec![1.0; 4]).unwrap();
        let before_params = st.params[0].clone();
        let before_momentum = st.momentum[0].clone();
        let gradient = [f32::MAX; 4];
        let computed = st
            .prepare_merge(0, 0, &[&gradient], &[1.0])
            .unwrap()
            .finish_inline()
            .unwrap();

        assert!(st.commit_merge(computed).is_err());
        assert_eq!(st.params[0], before_params);
        assert_eq!(st.momentum[0], before_momentum);
        assert_eq!(st.versions[0], 0);
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
        let mut st = state_with_fingerprint(layout2(), 0x11);
        st.init_fragment(0, vec![1.5; 4]).unwrap();
        st.init_fragment(1, vec![-2.0; 4]).unwrap();
        let outer_gradient = vec![1.5f32; 4];
        st.merge_and_step(0, &[&outer_gradient], &[1.0]).unwrap();
        st.global_step = 7;
        st.versions = vec![7, 6];
        st.record_merge(3, 12, 4096);
        st.save_checkpoint(&path).unwrap();
        let encoded = std::fs::read(&path).unwrap();
        assert_eq!(&encoded[0..4], &CKPT_MAGIC_V3.to_le_bytes());
        assert_eq!(encoded[4], IsoBackendKind::Scalar as u8);
        assert_eq!(&encoded[5..37], &fingerprint(0x11));

        let mut st2 = state_with_fingerprint(layout2(), 0x11);
        st2.load_checkpoint(&path).unwrap();
        assert_eq!(st2.global_step, 7);
        assert_eq!(st2.versions, vec![7, 6]);
        assert_eq!(st2.params, st.params);
        assert!(st2.all_initialized());
        assert!(st2.checkpoint_layout_verified);
        assert_eq!(st2.ledger.get(&3).unwrap().tokens, 4096);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn checkpoint_reader_streams_f32_payloads_across_bounded_chunks() {
        use std::io::Write;

        let dir = std::env::temp_dir().join(format!(
            "yeto-streaming-ckpt-reader-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("values.bin");
        let count = CHECKPOINT_READ_CHUNK_BYTES / 4 + 17;
        {
            let mut writer = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
            for index in 0..count {
                writer
                    .write_all(&(index as f32 * 0.25).to_le_bytes())
                    .unwrap();
            }
            writer.flush().unwrap();
        }

        let mut reader = CheckpointReader::open(&path).unwrap();
        // Deliberately use a tiny scratch buffer so the test exercises many
        // chunk boundaries without changing the production 8 MiB bound.
        let mut scratch = [0u8; 12];
        let values = reader.f32s(count, &mut scratch).unwrap();
        assert_eq!(reader.remaining(), 0);
        assert_eq!(values.len(), count);
        for index in [0, 1, 2, count / 2, count - 2, count - 1] {
            assert_eq!(values[index], index as f32 * 0.25);
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn truncated_streaming_checkpoint_does_not_mutate_live_state() {
        let dir = std::env::temp_dir().join(format!(
            "yeto-truncated-streaming-ckpt-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("state.ckpt");

        let mut saved = state_with_fingerprint(layout2(), 0x19);
        saved.session_contract_hash = Some([0x19; 32]);
        saved.init_fragment(0, vec![1.0; 4]).unwrap();
        saved.init_fragment(1, vec![2.0; 4]).unwrap();
        saved.global_step = 2;
        saved.versions = vec![1, 2];
        saved.save_checkpoint(&path).unwrap();
        let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.set_len(file.metadata().unwrap().len() - 1).unwrap();

        let mut restored = state_with_fingerprint(layout2(), 0x19);
        restored.session_contract_hash = Some([0x19; 32]);
        restored.init_fragment(0, vec![8.0; 4]).unwrap();
        restored.init_fragment(1, vec![9.0; 4]).unwrap();
        let before_params = restored.params.clone();
        let before_momentum = restored.momentum.clone();
        let before_versions = restored.versions.clone();
        assert!(restored.load_checkpoint(&path).is_err());
        assert!(!restored.checkpoint_layout_verified);
        assert_eq!(restored.global_step, 0);
        assert_eq!(restored.params, before_params);
        assert_eq!(restored.momentum, before_momentum);
        assert_eq!(restored.versions, before_versions);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn fragment_payload_preflight_rejects_missing_momentum_before_restore() {
        let dir = std::env::temp_dir().join(format!(
            "yeto-truncated-fragment-state-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("state.ckpt");

        let mut saved = state_with_fingerprint(layout2(), 0x29);
        saved.init_fragment(0, vec![1.0; 4]).unwrap();
        saved.init_fragment(1, vec![2.0; 4]).unwrap();
        saved.save_checkpoint(&path).unwrap();
        // V3 header + fragment-0 header + its complete parameter array, but
        // only two bytes of the required momentum array.
        let fragment_zero_payload = 4 + 1 + 32 + 8 + 4 + 8 + 8;
        let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.set_len((fragment_zero_payload + 4 * 4 + 2) as u64)
            .unwrap();

        let mut restored = state_with_fingerprint(layout2(), 0x29);
        restored.init_fragment(0, vec![8.0; 4]).unwrap();
        restored.init_fragment(1, vec![9.0; 4]).unwrap();
        let before_params = restored.params.clone();
        let before_momentum = restored.momentum.clone();
        assert!(restored.load_checkpoint(&path).is_err());
        assert_eq!(restored.params, before_params);
        assert_eq!(restored.momentum, before_momentum);
        assert!(!restored.checkpoint_layout_verified);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn generic_checkpoint_persists_and_verifies_the_session_contract() {
        let dir = std::env::temp_dir().join(format!(
            "yeto-generic-contract-ckpt-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let legacy_path = dir.join("legacy.ckpt");
        let contract_path = dir.join("contract.ckpt");

        let make_state = || {
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
            state.global_step = 3;
            state.versions = vec![3, 2];
            state
        };

        let legacy = make_state();
        legacy.save_checkpoint(&legacy_path).unwrap();
        let legacy_bytes = std::fs::read(&legacy_path).unwrap();

        let mut contract_bound = make_state();
        contract_bound.session_contract_hash = Some([7; 32]);
        contract_bound.save_checkpoint(&contract_path).unwrap();
        let contract_bytes = std::fs::read(&contract_path).unwrap();
        assert_eq!(
            &contract_bytes[..legacy_bytes.len()],
            legacy_bytes.as_slice()
        );
        assert_eq!(contract_bytes.len(), legacy_bytes.len() + 36);
        assert_eq!(
            &contract_bytes[legacy_bytes.len()..legacy_bytes.len() + 4],
            &SESSION_CONTRACT_CKPT_MAGIC.to_le_bytes()
        );
        assert_eq!(&contract_bytes[legacy_bytes.len() + 4..], &[7; 32]);

        let mut matching = make_state();
        matching.session_contract_hash = Some([7; 32]);
        matching.load_checkpoint(&contract_path).unwrap();
        assert_eq!(matching.params, contract_bound.params);
        assert_eq!(matching.versions, contract_bound.versions);

        let mut mismatched = make_state();
        mismatched.session_contract_hash = Some([8; 32]);
        let error = mismatched.load_checkpoint(&contract_path).unwrap_err();
        assert!(format!("{error:#}").contains("session contract hash does not match"));

        // The old layout-only format has one safe implied contract: the
        // layout fingerprint itself. A stronger SAO/profile contract must not
        // silently inherit such a checkpoint.
        let mut legacy_default = make_state();
        legacy_default.session_contract_hash = Some([9; 32]);
        legacy_default.load_checkpoint(&legacy_path).unwrap();
        let mut legacy_custom = make_state();
        legacy_custom.session_contract_hash = Some([7; 32]);
        let error = legacy_custom.load_checkpoint(&legacy_path).unwrap_err();
        assert!(format!("{error:#}").contains("missing the session contract hash"));

        std::fs::remove_dir_all(&dir).ok();
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
        let before_params = wrong_semantics.params.clone();
        let before_momentum = wrong_semantics.momentum.clone();
        let before_versions = wrong_semantics.versions.clone();
        let before_ledger = wrong_semantics.ledger.clone();
        let error = wrong_semantics.load_checkpoint(&sweep_path).unwrap_err();
        assert!(format!("{error:#}").contains("session contract hash"));
        assert!(!wrong_semantics.checkpoint_layout_verified);
        assert_eq!(wrong_semantics.params, before_params);
        assert_eq!(wrong_semantics.momentum, before_momentum);
        assert_eq!(wrong_semantics.versions, before_versions);
        assert_eq!(wrong_semantics.ledger.len(), before_ledger.len());
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
    fn checkpoint_layout_fingerprint_is_verified_even_without_redundant_trailer() {
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
        let error = legacy.load_checkpoint(&path).unwrap_err();
        assert!(format!("{error:#}").contains("semantic layout fingerprint does not match"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn checkpoint_atomically_replaces_previous_file() {
        let dir = std::env::temp_dir().join(format!("yeto-atomic-ckpt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("state.ckpt");
        std::fs::write(&path, b"old checkpoint bytes").unwrap();

        let mut st = state_with_fingerprint(layout2(), 0x22);
        st.init_fragment(0, vec![3.0; 4]).unwrap();
        st.init_fragment(1, vec![-4.0; 4]).unwrap();
        st.global_step = 13;
        st.versions = vec![13, 12];
        st.save_checkpoint(&path).unwrap();

        let mut restored = state_with_fingerprint(layout2(), 0x22);
        restored.load_checkpoint(&path).unwrap();
        assert_eq!(restored.global_step, 13);
        assert_eq!(restored.versions, vec![13, 12]);
        assert!(!path.with_extension("tmp").exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn v3_rejects_same_sizes_with_different_semantic_layout_before_mutation() {
        let dir =
            std::env::temp_dir().join(format!("yeto-v3-layout-mismatch-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("state.ckpt");

        let mut saved = state_with_fingerprint(layout2(), 0x31);
        saved.init_fragment(0, vec![1.0; 4]).unwrap();
        saved.init_fragment(1, vec![2.0; 4]).unwrap();
        saved.global_step = 2;
        saved.versions = vec![1, 2];
        saved.save_checkpoint(&path).unwrap();

        // Same numerical Layout, fragment count, and per-fragment numel. The
        // different fingerprint represents changed tensor names/order/shapes.
        let mut restored = state_with_fingerprint(layout2(), 0x32);
        restored.init_fragment(0, vec![8.0; 4]).unwrap();
        restored.init_fragment(1, vec![9.0; 4]).unwrap();
        restored.global_step = 0;
        let before_params = restored.params.clone();
        let before_versions = restored.versions.clone();

        let error = restored.load_checkpoint(&path).unwrap_err().to_string();
        assert!(error.contains("semantic layout fingerprint"), "{error}");
        assert_eq!(restored.global_step, 0);
        assert_eq!(restored.versions, before_versions);
        assert_eq!(restored.params, before_params);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn v3_rejects_backend_mismatch_before_mutation() {
        let dir =
            std::env::temp_dir().join(format!("yeto-v3-backend-mismatch-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("state.ckpt");

        let mut saved = state_with_fingerprint(layout2(), 0x35);
        saved.init_fragment(0, vec![1.0; 4]).unwrap();
        saved.init_fragment(1, vec![2.0; 4]).unwrap();
        saved.global_step = 2;
        saved.versions = vec![1, 2];
        saved.save_checkpoint(&path).unwrap();
        let mut encoded = std::fs::read(&path).unwrap();
        encoded[4] = IsoBackendKind::TorchSvd as u8;
        std::fs::write(&path, encoded).unwrap();

        let mut restored = state_with_fingerprint(layout2(), 0x35);
        restored.init_fragment(0, vec![8.0; 4]).unwrap();
        restored.init_fragment(1, vec![9.0; 4]).unwrap();
        let before_params = restored.params.clone();
        let error = restored.load_checkpoint(&path).unwrap_err().to_string();
        assert!(
            error.contains("does not match configured backend"),
            "{error}"
        );
        assert_eq!(restored.global_step, 0);
        assert_eq!(restored.params, before_params);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn v2_resume_is_allowed_only_for_ungrouped_layouts() {
        let dir = std::env::temp_dir().join(format!("yeto-v2-compat-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("state.ckpt");

        let ungrouped = Layout {
            fragments: vec![FragmentInfo {
                merge_mode: MERGE_AVG,
                tensor_numels: vec![4],
                tensor_shapes: None,
            }],
        };
        let mut saved = state_with_fingerprint(ungrouped.clone(), 0x41);
        saved.init_fragment(0, vec![3.0; 4]).unwrap();
        saved.global_step = 1;
        saved.versions = vec![1];
        write_v2_checkpoint(&path, &saved);

        let mut restored = state_with_fingerprint(ungrouped, 0x99);
        restored.load_checkpoint(&path).unwrap();
        assert_eq!(restored.global_step, 1);
        assert_eq!(restored.params, saved.params);

        let grouped_path = dir.join("grouped.ckpt");
        let mut grouped_saved = state_with_fingerprint(layout2(), 0x41);
        grouped_saved.init_fragment(0, vec![4.0; 4]).unwrap();
        grouped_saved.init_fragment(1, vec![5.0; 4]).unwrap();
        grouped_saved.global_step = 2;
        grouped_saved.versions = vec![1, 2];
        write_v2_checkpoint(&grouped_path, &grouped_saved);
        let mut grouped = state_with_fingerprint(layout2(), 0x41);
        let error = grouped
            .load_checkpoint(&grouped_path)
            .unwrap_err()
            .to_string();
        assert!(error.contains("cannot resume a grouped"), "{error}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn v1_resume_never_guesses_the_missing_backend() {
        let dir = std::env::temp_dir().join(format!("yeto-v1-reject-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("state.ckpt");
        std::fs::write(&path, CKPT_MAGIC_V1.to_le_bytes()).unwrap();
        let mut restored = state_with_fingerprint(layout2(), 0x51);
        let error = restored.load_checkpoint(&path).unwrap_err().to_string();
        assert!(error.contains("V1 cannot be resumed safely"), "{error}");
        assert!(error.contains("neither the ISO backend"), "{error}");
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
