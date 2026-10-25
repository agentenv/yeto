//! Global model state held by the syncer: fragment layout, parameters Θ,
//! and outer-optimizer momentum, all in f32.

use anyhow::{bail, Context, Result};

use crate::iso_worker::{IsoBackend, IsoBackendConfig, IsoBackendKind};
use crate::merge;
use crate::protocol::Reader;
use crate::shard::{ShardPlan, ShardSchedule};
use std::sync::Arc;

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
    /// Shared copy-on-write so a checkpoint snapshot costs no copy: a commit
    /// clones only a fragment a pending snapshot still references.
    pub params: Vec<Arc<Vec<f32>>>,
    /// Nesterov momentum buffers, same shape as params.
    momentum: Vec<Arc<Vec<f32>>>,
    /// Delayed Nesterov period N (>= 2); None keeps plain Nesterov. Set it
    /// with `set_delayed_nesterov_period` before INIT_PARAMS.
    delayed_nesterov_period: Option<u32>,
    /// Per-fragment Delayed Nesterov step count and accumulator; empty
    /// unless the period is set.
    delayed_nesterov: Vec<merge::DelayedNesterovState>,
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
    /// bf16 downlink anchors (S26 E3); in memory only, never checkpointed.
    pub downlink: Option<std::sync::Mutex<crate::downlink::DownlinkAnchors>>,
    /// Head shard of this process and the fragments it owns (S27 P4b).
    /// Unowned fragments are never allocated, initialized or checkpointed.
    shard: ShardPlan,
    schedule: ShardSchedule,
    /// Small records written next to every checkpoint at the same cut
    /// (the task dispenser, S27 P2). Empty: none.
    pub checkpoint_sidecars: Vec<Arc<dyn CheckpointSidecar>>,
}

/// A small record captured at every checkpoint cut and written next to the
/// checkpoint as `<checkpoint><suffix>`, before the checkpoint itself (a
/// crash between the two leaves the newer record with the older checkpoint).
pub trait CheckpointSidecar: Send + Sync {
    fn suffix(&self) -> &'static str;
    fn snapshot(&self) -> Vec<u8>;
}

/// A merge whose deterministic learner reduction has completed but whose
/// exact Torch SVD matrices may still be executing.  It owns every input
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

    pub(crate) fn finish_inline(self) -> Result<ComputedMerge> {
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
        params.resize_with(fragment_count, Default::default);
        let mut momentum = Vec::new();
        momentum
            .try_reserve_exact(fragment_count)
            .context("cannot allocate momentum fragment table")?;
        momentum.resize_with(fragment_count, Default::default);
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
            delayed_nesterov_period: None,
            delayed_nesterov: Vec::new(),
            initialized,
            versions,
            global_step: 0,
            ledger: Default::default(),
            checkpoint_sidecars: Vec::new(),
            outer_lr,
            outer_momentum,
            wire_dtype,
            delta_correction: None,
            iso_backend: IsoBackend::start(iso_backend_config)?,
            downlink: None,
            shard: ShardPlan::default(),
            schedule: ShardSchedule::unsharded(fragment_count),
        })
    }

    /// Restrict this state to `plan`'s fragment range.  Call before any
    /// fragment is initialized or a checkpoint is loaded.
    pub fn set_shard(&mut self, plan: ShardPlan) -> Result<()> {
        if self.initialized.iter().any(|&b| b) {
            bail!("head shard must be set before fragments are initialized");
        }
        self.schedule = plan.schedule(self.layout.fragments.len())?;
        self.shard = plan;
        Ok(())
    }

    /// The global steps this process runs (every step unsharded).
    pub fn schedule(&self) -> &ShardSchedule {
        &self.schedule
    }

    /// Fragments this process owns (all of them unsharded).
    pub fn owned_fragments(&self) -> std::ops::Range<usize> {
        self.schedule.owned()
    }

    pub fn owns_fragment(&self, fid: usize) -> bool {
        self.schedule.owned().contains(&fid)
    }

    pub fn iso_backend_kind(&self) -> IsoBackendKind {
        self.iso_backend.kind()
    }

    /// Enable Delayed Nesterov with period `period` (>= 2), or disable it.
    /// Must be called before any fragment is initialized.
    pub fn set_delayed_nesterov_period(&mut self, period: Option<u32>) -> Result<()> {
        if period.is_some_and(|n| n < 2) {
            bail!("delayed Nesterov period must be >= 2 (1 is plain Nesterov)");
        }
        if self.initialized.iter().any(|&b| b) {
            bail!("delayed Nesterov period must be set before fragment initialization");
        }
        self.delayed_nesterov_period = period;
        self.delayed_nesterov = match period {
            Some(_) => vec![Default::default(); self.params.len()],
            None => Vec::new(),
        };
        Ok(())
    }

    pub fn all_initialized(&self) -> bool {
        self.initialized[self.schedule.owned()].iter().all(|&b| b)
    }

    pub fn init_fragment(&mut self, fid: usize, values: Vec<f32>) -> Result<()> {
        let expected = self
            .layout
            .fragments
            .get(fid)
            .with_context(|| format!("init for unknown fragment {fid}"))?
            .numel()?;
        if !self.owns_fragment(fid) {
            bail!(
                "init for fragment {fid} not owned by syncer shard {}/{} (owns {:?})",
                self.shard.index,
                self.shard.count,
                self.schedule.owned()
            );
        }
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
            if self.delayed_nesterov_period.is_some() {
                let mut accum = Vec::new();
                accum
                    .try_reserve_exact(expected)
                    .context("cannot allocate fragment delayed Nesterov buffer")?;
                accum.resize(expected, 0.0);
                self.delayed_nesterov[fid] =
                    merge::DelayedNesterovState { step: 0, accum: Arc::new(accum) };
            }
            self.params[fid] = Arc::new(values);
            self.momentum[fid] = Arc::new(momentum);
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
    /// coordinator state.  Torch SVD work is represented as owned matrices
    /// inside the returned value and may execute out of order.
    pub fn prepare_merge(
        &self,
        fid: usize,
        base_version: u64,
        outer_gradients: &[&[f32]],
        weights: &[f64],
    ) -> Result<PreparedMerge> {
        self.prepare_merge_scaled(fid, base_version, outer_gradients, weights, None)
    }

    /// `prepare_merge` with an optional per-tensor scale of every learner's
    /// weight (expert token weights): tensor `t` of learner `m` merges with
    /// weight `weights[m] * tensor_scales[m][t]`. `None` is `prepare_merge`.
    pub fn prepare_merge_scaled(
        &self,
        fid: usize,
        base_version: u64,
        outer_gradients: &[&[f32]],
        weights: &[f64],
        tensor_scales: Option<&[&[f64]]>,
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
        if let Some(scales) = tensor_scales {
            if scales.len() != weights.len()
                || scales.iter().any(|s| s.len() != frag.tensor_numels.len())
            {
                bail!("fragment {fid}: tensor scales do not match the learners and tensors");
            }
        }
        let mut tensor_weights = weights.to_vec();
        let mut delta = vec![0.0f32; numel];
        // Merge per tensor slice within the fragment.
        let mut off = 0usize;
        for (tensor_index, &tn) in frag.tensor_numels.iter().enumerate() {
            let tn = tn as usize;
            let slices: Vec<&[f32]> = outer_gradients
                .iter()
                .map(|gradient| &gradient[off..off + tn])
                .collect();
            let weights = match tensor_scales {
                None => weights,
                Some(scales) => {
                    for ((w, &base), s) in tensor_weights.iter_mut().zip(weights).zip(scales) {
                        *w = base * s[tensor_index];
                    }
                    tensor_weights.as_slice()
                }
            };
            let out = &mut delta[off..off + tn];
            match frag.merge_mode {
                // ISO reduces by direct average; the spectrum is flattened
                // by `prepare_reduced_merge`.
                MERGE_AVG | MERGE_ISO => merge::merge_avg(&slices, weights, out),
                MERGE_RDA => merge::merge_rda(&slices, weights, out),
                mode => bail!("fragment {fid}: unsupported merge mode {mode}"),
            }
            off += tn;
        }
        self.prepare_reduced_merge(fid, base_version, delta)
    }

    /// Prepare a merge from an already reduced fragment: the output of the
    /// learner reduction in `prepare_merge`, or of a streaming
    /// `crate::fold::FragmentFold` that produced the same values in place.
    pub fn prepare_reduced_merge(
        &self,
        fid: usize,
        base_version: u64,
        mut delta: Vec<f32>,
    ) -> Result<PreparedMerge> {
        let current_version = *self
            .versions
            .get(fid)
            .with_context(|| format!("merge for unknown fragment {fid}"))?;
        if current_version != base_version {
            bail!("fragment {fid}: prepare version {base_version} != current {current_version}");
        }
        let frag = &self.layout.fragments[fid];
        let numel = frag.numel()?;
        if delta.len() != numel {
            bail!(
                "fragment {fid}: reduced merge has {} values, expected {numel}",
                delta.len()
            );
        }
        let mut iso_jobs = Vec::new();
        if frag.merge_mode == MERGE_ISO {
            let mut off = 0usize;
            for (tensor_index, &tn) in frag.tensor_numels.iter().enumerate() {
                let tn = tn as usize;
                let out = &mut delta[off..off + tn];
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
                off += tn;
            }
        }
        // Miles uses one canonical tensor per fragment. Move that complete
        // averaged matrix into its worker job instead of retaining a second
        // model-sized host copy while SVD runs.
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
        if let Some(period) = self.delayed_nesterov_period {
            self.commit_delayed_nesterov(fid, &delta, period)?;
            return Ok(gnorm);
        }
        // Validate the complete optimizer result before mutating either
        // buffer. This keeps commit all-or-nothing without cloning a
        // model-sized fragment.
        for ((param, momentum), gradient) in
            self.params[fid].iter().zip(self.momentum[fid].iter()).zip(&delta)
        {
            let next_momentum = self.outer_momentum * *momentum + *gradient;
            let next_param =
                *param - self.outer_lr * (*gradient + self.outer_momentum * next_momentum);
            if !next_param.is_finite() || !next_momentum.is_finite() {
                bail!("fragment {fid}: outer optimizer produced non-finite state");
            }
        }
        merge::nesterov_step(
            Arc::make_mut(&mut self.params[fid]).as_mut_slice(),
            Arc::make_mut(&mut self.momentum[fid]).as_mut_slice(),
            &delta,
            self.outer_lr,
            self.outer_momentum,
        );
        debug_assert!(self.params[fid].iter().all(|value| value.is_finite()));
        debug_assert!(self.momentum[fid].iter().all(|value| value.is_finite()));
        Ok(gnorm)
    }

    /// The Delayed Nesterov variant of the outer step in `commit_merge`,
    /// with the same validate-then-mutate contract.
    fn commit_delayed_nesterov(&mut self, fid: usize, delta: &[f32], period: u32) -> Result<()> {
        let state = &mut self.delayed_nesterov[fid];
        let update = merge::delayed_nesterov_update(state.step, period);
        for (((param, momentum), accum), gradient) in self.params[fid]
            .iter()
            .zip(self.momentum[fid].iter())
            .zip(state.accum.iter())
            .zip(delta)
        {
            let (p, m, a) = merge::delayed_nesterov_elem(
                *param,
                *momentum,
                *accum,
                *gradient,
                self.outer_lr,
                self.outer_momentum,
                period,
                update,
            );
            if !p.is_finite() || !m.is_finite() || !a.is_finite() {
                bail!("fragment {fid}: outer optimizer produced non-finite state");
            }
        }
        merge::delayed_nesterov_step(
            Arc::make_mut(&mut self.params[fid]).as_mut_slice(),
            Arc::make_mut(&mut self.momentum[fid]).as_mut_slice(),
            state,
            delta,
            self.outer_lr,
            self.outer_momentum,
            period,
        );
        Ok(())
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
    /// publishing a cut after partial SVD failure.
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
/// Last 4 bytes of a Delayed Nesterov checkpoint. The block before it is,
/// per fragment, `step u64 || accum f32 × numel`, then `period u32`. It is
/// appended after every other trailer, only when the period is set, so
/// plain-Nesterov checkpoints keep their exact bytes.
const DELAYED_NESTEROV_CKPT_MAGIC: u32 = 0x444E_5354;
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

/// Little-endian f32 values in 64 KiB blocks (same bytes as per-value
/// `to_le_bytes`, without one writer call per element).
fn write_f32s<W: std::io::Write>(f: &mut W, values: &[f32]) -> Result<()> {
    let mut block = Vec::with_capacity(16 * 1024 * 4);
    for chunk in values.chunks(16 * 1024) {
        block.clear();
        for v in chunk {
            block.extend_from_slice(&v.to_le_bytes());
        }
        f.write_all(&block)?;
    }
    Ok(())
}

/// Buffered little-endian reader over a checkpoint file.
struct CheckpointReader {
    inner: std::io::BufReader<std::fs::File>,
    len: u64,
    pos: u64,
    /// Trailing bytes hidden by `reserve_tail`.
    reserved: u64,
}

impl CheckpointReader {
    fn open(path: &std::path::Path) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let len = file.metadata()?.len();
        Ok(Self {
            inner: std::io::BufReader::with_capacity(1 << 20, file),
            len,
            pos: 0,
            reserved: 0,
        })
    }

    fn fill(&mut self, out: &mut [u8]) -> Result<()> {
        use std::io::Read;
        if self.len - self.pos < out.len() as u64 {
            bail!("payload truncated");
        }
        self.inner.read_exact(out)?;
        self.pos += out.len() as u64;
        Ok(())
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut out = [0u8; N];
        self.fill(&mut out)?;
        Ok(out)
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

    fn array32(&mut self) -> Result<[u8; 32]> {
        self.array()
    }

    /// Decode `out.len()` little-endian f32 values in 64 KiB blocks.
    fn f32s(&mut self, out: &mut [f32]) -> Result<()> {
        let mut block = vec![0u8; 16 * 1024 * 4];
        for chunk in out.chunks_mut(16 * 1024) {
            let bytes = &mut block[..chunk.len() * 4];
            self.fill(bytes)?;
            for (v, b) in chunk.iter_mut().zip(bytes.chunks_exact(4)) {
                *v = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
        }
        Ok(())
    }

    /// Hide the last `bytes` of the file from reads until `release_tail`.
    fn reserve_tail(&mut self, bytes: u64) -> Result<()> {
        if self.len - self.pos < bytes {
            bail!("checkpoint is shorter than its {bytes}-byte trailing block");
        }
        self.len -= bytes;
        self.reserved = bytes;
        Ok(())
    }

    /// Make the bytes hidden by `reserve_tail` readable again.
    fn release_tail(&mut self) {
        self.len += std::mem::take(&mut self.reserved);
    }

    /// The rest of the file, refusing more than `max` bytes.
    fn tail(&mut self, max: u64) -> Result<Vec<u8>> {
        let remaining = self.len - self.pos;
        if remaining > max {
            bail!("checkpoint has {remaining} trailing bytes");
        }
        let mut out = vec![0u8; remaining as usize];
        self.fill(&mut out)?;
        Ok(out)
    }
}

enum CheckpointTrailer {
    LayoutOnly,
    SessionContract([u8; 32]),
    PolicySweep {
        fragments: u32,
        contract_hash: [u8; 32],
    },
}

/// Owned, immutable checkpoint cut; see `GlobalState::checkpoint_snapshot`.
pub struct CheckpointSnapshot {
    iso_backend: u8,
    layout_fingerprint: [u8; 32],
    global_step: u64,
    /// (version, params, momentum) per fragment, shared with the live state.
    fragments: Vec<(u64, Arc<Vec<f32>>, Arc<Vec<f32>>)>,
    ledger: std::collections::BTreeMap<u32, LearnerLedger>,
    trailer: CheckpointTrailer,
    /// Delayed Nesterov period and per-fragment state, shared like params.
    delayed_nesterov: Option<(u32, Vec<merge::DelayedNesterovState>)>,
    /// (suffix, bytes) of every checkpoint sidecar at this cut.
    sidecars: Vec<(&'static str, Vec<u8>)>,
}

impl CheckpointSnapshot {
    /// Stream the V3 checkpoint encoding, one fragment at a time.
    ///
    /// The snapshot drops its share of each fragment's buffers as soon as
    /// they are written, so a later commit to a written fragment no longer
    /// copies it. Returns how many fragments the live state had already
    /// copied (committed to while this snapshot still held them).
    fn write_to<W: std::io::Write>(self, f: &mut W) -> Result<usize> {
        let CheckpointSnapshot {
            iso_backend,
            layout_fingerprint,
            global_step,
            fragments,
            ledger,
            trailer,
            delayed_nesterov,
            // `save` writes the sidecars before the checkpoint itself.
            sidecars: _,
        } = self;
        let mut copied = vec![false; fragments.len()];
        f.write_all(&CKPT_MAGIC_V3.to_le_bytes())?;
        f.write_all(&[iso_backend])?;
        f.write_all(&layout_fingerprint)?;
        f.write_all(&global_step.to_le_bytes())?;
        f.write_all(&(fragments.len() as u32).to_le_bytes())?;
        for (fid, (version, params, momentum)) in fragments.into_iter().enumerate() {
            f.write_all(&version.to_le_bytes())?;
            f.write_all(&(params.len() as u64).to_le_bytes())?;
            write_f32s(f, &params)?;
            write_f32s(f, &momentum)?;
            // Both buffers are released; either may be the copied one.
            let params_copied = release_shared(params);
            copied[fid] = release_shared(momentum) || params_copied;
        }
        f.write_all(&(ledger.len() as u32).to_le_bytes())?;
        for (id, l) in &ledger {
            f.write_all(&id.to_le_bytes())?;
            f.write_all(&l.merges.to_le_bytes())?;
            f.write_all(&l.steps.to_le_bytes())?;
            f.write_all(&l.tokens.to_le_bytes())?;
        }
        f.write_all(&layout_fingerprint)?;
        match &trailer {
            CheckpointTrailer::LayoutOnly => {}
            CheckpointTrailer::SessionContract(contract_hash) => {
                f.write_all(&SESSION_CONTRACT_CKPT_MAGIC.to_le_bytes())?;
                f.write_all(contract_hash)?;
            }
            CheckpointTrailer::PolicySweep {
                fragments,
                contract_hash,
            } => {
                f.write_all(&POLICY_SWEEP_CKPT_MAGIC.to_le_bytes())?;
                f.write_all(&fragments.to_le_bytes())?;
                f.write_all(contract_hash)?;
            }
        }
        // Appended after every other trailer, only when the period is set,
        // so plain-Nesterov checkpoints keep their exact bytes.
        if let Some((period, states)) = delayed_nesterov {
            for (fid, state) in states.into_iter().enumerate() {
                f.write_all(&state.step.to_le_bytes())?;
                write_f32s(f, &state.accum)?;
                if release_shared(state.accum) {
                    if let Some(copied) = copied.get_mut(fid) {
                        *copied = true;
                    }
                }
            }
            f.write_all(&period.to_le_bytes())?;
            f.write_all(&DELAYED_NESTEROV_CKPT_MAGIC.to_le_bytes())?;
        }
        Ok(copied.into_iter().filter(|&copied| copied).count())
    }

    /// Atomically persist the snapshot at `path` through the sibling
    /// temporary `<path without extension>.tmp`; see `persist`.  Every
    /// checkpoint sidecar (the task dispenser, S27 P2) is written first as
    /// `<path><suffix>` (temporary file, rename), so a crash between the two
    /// leaves the newer record with the older checkpoint.  Evaluation
    /// checkpoints (`persist` only) carry no sidecars.
    pub fn save(mut self, path: &std::path::Path) -> Result<()> {
        for (suffix, bytes) in std::mem::take(&mut self.sidecars) {
            let mut name = path.as_os_str().to_owned();
            name.push(suffix);
            let sidecar = std::path::PathBuf::from(name);
            let mut partial = sidecar.as_os_str().to_owned();
            partial.push(".tmp");
            let partial = std::path::PathBuf::from(partial);
            std::fs::write(&partial, &bytes).with_context(|| format!("write {}", partial.display()))?;
            std::fs::rename(&partial, &sidecar).with_context(|| format!("write {}", sidecar.display()))?;
        }
        self.persist(path, &path.with_extension("tmp")).map(drop)
    }

    /// Atomically persist the snapshot: stream it into the temporary `tmp`
    /// (same directory as `path`), fsync, rename over `path`, fsync the
    /// directory.  A failure at any point leaves the previous file at `path`
    /// intact; `tmp` may remain.
    pub fn persist(self, path: &std::path::Path, tmp: &std::path::Path) -> Result<PersistedSnapshot> {
        let persisted = {
            let file = std::fs::File::create(tmp)
                .with_context(|| format!("create checkpoint temporary {}", tmp.display()))?;
            let mut f = std::io::BufWriter::with_capacity(1 << 20, file);
            let copied_fragments = self.write_to(&mut f)?;
            std::io::Write::flush(&mut f)?;
            let file = f.get_ref();
            file.sync_all()?;
            PersistedSnapshot {
                bytes: file.metadata()?.len(),
                copied_fragments,
            }
        };
        std::fs::rename(tmp, path)?;
        sync_parent_directory(path)?;
        Ok(persisted)
    }
}

/// What `CheckpointSnapshot::persist` wrote.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PersistedSnapshot {
    /// Size of the written file.
    pub bytes: u64,
    /// Fragments the live state copied (copy-on-write) because a commit
    /// reached them before the writer did.
    pub copied_fragments: usize,
}

/// Drop a snapshot's share of a fragment buffer.  True when the snapshot
/// held the last reference: the live state had already replaced the buffer
/// with its own copy (`Arc::make_mut` in a commit).  `Arc::into_inner` is
/// atomic, so a commit that races this drop is counted on exactly one side.
fn release_shared(buffer: Arc<Vec<f32>>) -> bool {
    Arc::into_inner(buffer).is_some()
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
    /// Byte length of the trailing Delayed Nesterov checkpoint block for
    /// this layout (0 when the period is unset): per fragment
    /// `step u64 || accum f32 × numel`, then `period u32 || magic u32`.
    fn delayed_nesterov_block_bytes(&self) -> Result<u64> {
        if self.delayed_nesterov_period.is_none() {
            return Ok(0);
        }
        let mut len = 8u64;
        for p in 0..self.layout.fragments.len() {
            len = u64::try_from(self.delayed_nesterov_numel(p)?)
                .ok()
                .and_then(|numel| numel.checked_mul(4))
                .and_then(|bytes| len.checked_add(8 + bytes))
                .context("delayed Nesterov checkpoint block size overflows")?;
        }
        Ok(len)
    }

    /// Accumulator length of fragment `p` in the Delayed Nesterov block: a
    /// sharded head stores only its own fragments, the others are empty
    /// (like their params and momentum).
    fn delayed_nesterov_numel(&self, p: usize) -> Result<usize> {
        if self.owns_fragment(p) {
            self.layout.fragments[p].numel()
        } else {
            Ok(0)
        }
    }

    /// Parse the Delayed Nesterov block that `r` is positioned at.
    fn read_delayed_nesterov_block(
        &self,
        r: &mut CheckpointReader,
    ) -> Result<Vec<merge::DelayedNesterovState>> {
        let Some(period) = self.delayed_nesterov_period else {
            return Ok(Vec::new());
        };
        let mut states = Vec::with_capacity(self.layout.fragments.len());
        for p in 0..self.layout.fragments.len() {
            let numel = self.delayed_nesterov_numel(p)?;
            let step = r.u64()?;
            let mut accum = Vec::new();
            accum
                .try_reserve_exact(numel)
                .context("cannot allocate checkpoint delayed Nesterov buffer")?;
            accum.resize(numel, 0.0);
            r.f32s(&mut accum)?;
            states.push(merge::DelayedNesterovState { step, accum: Arc::new(accum) });
        }
        let checkpoint_period = r.u32()?;
        if r.u32()? != DELAYED_NESTEROV_CKPT_MAGIC {
            bail!("checkpoint has no delayed Nesterov state");
        }
        if checkpoint_period != period {
            bail!("checkpoint delayed Nesterov period {checkpoint_period} != configured {period}");
        }
        Ok(states)
    }

    /// Persist a consistent snapshot. Called only at the quiescent cut
    /// between rounds (see docs/PROTOCOL.md "Consistent snapshots").
    /// Written to `<path>.tmp` then renamed, so a crash mid-write never
    /// corrupts the previous checkpoint.
    pub fn save_checkpoint(&self, path: &std::path::Path) -> Result<()> {
        self.checkpoint_snapshot()?.save(path)
    }

    /// The exact bytes `save_checkpoint` writes (tests and small states).
    pub fn encode_checkpoint(&self) -> Result<Vec<u8>> {
        let state_floats: usize = self.params.iter().map(|p| p.len()).sum::<usize>() * 2;
        let mut bytes = Vec::with_capacity(state_floats * 4 + 4096);
        self.write_checkpoint_to(&mut bytes)?;
        Ok(bytes)
    }

    /// A consistent cut that a background checkpoint writer persists
    /// fragment by fragment while later commits mutate `self`.  It shares
    /// the fragment buffers instead of encoding them: taking it copies no
    /// model data, and a later commit clones only a fragment the writer has
    /// not written yet (the writer releases each fragment once written).
    pub fn checkpoint_snapshot(&self) -> Result<CheckpointSnapshot> {
        let trailer = if let Some(fragments) = self.policy_sweep_fragments {
            CheckpointTrailer::PolicySweep {
                fragments,
                contract_hash: self
                    .session_contract_hash
                    .context("policy-sweep checkpoint requires a session contract hash")?,
            }
        } else if let Some(contract_hash) = self.session_contract_hash {
            CheckpointTrailer::SessionContract(contract_hash)
        } else {
            CheckpointTrailer::LayoutOnly
        };
        Ok(CheckpointSnapshot {
            iso_backend: self.iso_backend_kind() as u8,
            layout_fingerprint: self.layout_fingerprint,
            global_step: self.global_step,
            fragments: self
                .versions
                .iter()
                .zip(&self.params)
                .zip(&self.momentum)
                .map(|((&version, params), momentum)| {
                    (version, Arc::clone(params), Arc::clone(momentum))
                })
                .collect(),
            ledger: self.ledger.clone(),
            trailer,
            delayed_nesterov: self
                .delayed_nesterov_period
                .map(|period| (period, self.delayed_nesterov.clone())),
            sidecars: self.checkpoint_sidecars.iter().map(|s| (s.suffix(), s.snapshot())).collect(),
        })
    }

    fn write_checkpoint_to<W: std::io::Write>(&self, f: &mut W) -> Result<()> {
        self.checkpoint_snapshot()?.write_to(f).map(drop)
    }

    /// Restore params/momentum/versions/step/ledger from a snapshot.
    ///
    /// V3 binds the snapshot to both the ISO backend and exact semantic HELLO
    /// layout. V2 carries only the backend identity. V1 is rejected because
    /// historical scalar and Torch checkpoints are indistinguishable. The
    /// complete checkpoint is parsed and validated before any live state is
    /// mutated.
    pub fn load_checkpoint(&mut self, path: &std::path::Path) -> Result<()> {
        // Stream the file: only one fragment-sized decode block is held
        // besides the restored state (no whole-file buffer). A Delayed
        // Nesterov block sits after every other trailer; reserve it so the
        // base parse ends where it begins.
        let mut r = CheckpointReader::open(path)?;
        let delayed_nesterov_bytes = self.delayed_nesterov_block_bytes()?;
        r.reserve_tail(delayed_nesterov_bytes)
            .context("checkpoint has no delayed Nesterov state")?;
        let magic = r.u32()?;
        let (checkpoint_iso_backend, checkpoint_layout_fingerprint) = match magic {
            CKPT_MAGIC_V1 => bail!(
                "checkpoint V1 cannot be resumed safely: it records neither the ISO backend nor the semantic layout fingerprint"
            ),
            CKPT_MAGIC_V2 => (Some(IsoBackendKind::from_checkpoint(r.u8()?)?), None),
            CKPT_MAGIC_V3 => {
                let backend = IsoBackendKind::from_checkpoint(r.u8()?)?;
                (Some(backend), Some(r.array32()?))
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
            // A sharded head checkpoints only its own fragments; the others
            // are stored empty (numel 0) at version 0.
            let expected = if self.owns_fragment(p) {
                self.layout.fragments[p].numel()?
            } else {
                0
            };
            if numel != expected {
                bail!("checkpoint fragment {p} numel {numel} != layout {expected}");
            }
            let mut restored_params = Vec::new();
            let mut restored_momentum = Vec::new();
            for slot in [&mut restored_params, &mut restored_momentum] {
                slot.try_reserve_exact(numel)
                    .context("cannot allocate checkpoint fragment")?;
                slot.resize(numel, 0.0);
            }
            for slot in [&mut restored_params, &mut restored_momentum] {
                r.f32s(slot)?;
            }
            params.push(Arc::new(restored_params));
            momentum.push(Arc::new(restored_momentum));
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
        // The trailer is at most 72 bytes; parse it from memory as before.
        let tail = r.tail(72)?;
        r.release_tail();
        let delayed_nesterov = self.read_delayed_nesterov_block(&mut r)?;
        let mut r = Reader(&tail);
        let expected_policy_sweep_fragments = self.policy_sweep_fragments;
        let header_layout_verified = checkpoint_layout_fingerprint.is_some();
        let (checkpoint_policy_sweep_fragments, checkpoint_session_contract_hash) =
            match r.remaining() {
                0 => {
                    self.checkpoint_layout_verified = header_layout_verified;
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
                68 => {
                    let checkpoint_fingerprint: [u8; 32] = r.take(32)?.try_into()?;
                    if checkpoint_fingerprint != self.layout_fingerprint {
                        bail!("checkpoint layout fingerprint does not match HELLO");
                    }
                    if r.u32()? != SESSION_CONTRACT_CKPT_MAGIC {
                        bail!("checkpoint has an invalid session-contract trailer");
                    }
                    let contract_hash = r.take(32)?.try_into()?;
                    self.checkpoint_layout_verified = true;
                    (None, Some(contract_hash))
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
        match (checkpoint_session_contract_hash, self.session_contract_hash) {
            (actual, expected) if actual == expected => {}
            // A layout-only generic checkpoint predates explicit contract
            // persistence. Its only safe implied contract is exactly the
            // verified layout fingerprint used by legacy clients.
            (None, Some(expected))
                if self.checkpoint_layout_verified
                    && expected == self.layout_fingerprint
                    && expected_policy_sweep_fragments.is_none() => {}
            (None, Some(_)) => {
                bail!("checkpoint is missing the session contract hash required by HELLO");
            }
            _ => bail!("checkpoint session contract hash does not match HELLO"),
        }
        if !self.checkpoint_layout_verified
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
        if global_step != 0 && !self.schedule.owns_step(global_step) {
            bail!(
                "checkpoint global_step {global_step} is not a step of syncer shard {}/{}",
                self.shard.index,
                self.shard.count
            );
        }
        for (fid, version) in versions.iter().copied().enumerate() {
            let expected = if self.owns_fragment(fid) {
                self.schedule.expected_version(fid, global_step)
            } else {
                0
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
        self.delayed_nesterov = delayed_nesterov;
        self.initialized[self.schedule.owned()].fill(true);
        self.ledger = ledger;
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
            for value in st.params[p].iter() {
                f.write_all(&value.to_le_bytes()).unwrap();
            }
            for value in st.momentum[p].iter() {
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
        assert_eq!(*st.params[0], vec![1.0; 4]);
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
        assert_eq!(*st.params[0], vec![0.0; 4]);
    }

    #[test]
    fn tensor_scales_weight_each_tensor_separately() {
        let merged = |scales: Option<&[&[f64]]>| {
            let mut st = GlobalState::new(layout2(), 1.0, 0.0, crate::protocol::DTYPE_F32).unwrap();
            st.init_fragment(0, vec![0.0; 4]).unwrap();
            st.init_fragment(1, vec![0.0; 4]).unwrap();
            let a = [1.0f32, 0.0, 3.0, 0.0];
            let b = [0.0f32, 2.0, 0.0, 5.0];
            let computed = st
                .prepare_merge_scaled(1, 0, &[&a, &b], &[1.0, 3.0], scales)
                .unwrap()
                .finish_inline()
                .unwrap();
            st.commit_merge(computed).unwrap();
            st.params[1].iter().map(|v| -v).collect::<Vec<f32>>()
        };
        let plain = merged(None);
        assert_eq!(merged(Some(&[&[1.0, 1.0], &[1.0, 1.0]])), plain);
        // Tensor 1: learner a routed nothing, so only b's update remains
        // (RDA of a single learner is that learner's update).
        let scaled = merged(Some(&[&[1.0, 0.0], &[1.0, 2.0]]));
        assert_eq!(scaled[..2], plain[..2]);
        assert_eq!(scaled[2..], [0.0, 5.0]);
        let mut st = GlobalState::new(layout2(), 1.0, 0.0, crate::protocol::DTYPE_F32).unwrap();
        st.init_fragment(1, vec![0.0; 4]).unwrap();
        let g = [0.0f32; 4];
        assert!(st.prepare_merge_scaled(1, 0, &[&g, &g], &[1.0, 1.0], Some(&[&[1.0]])).is_err());
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
        assert_eq!(*st.params[0], vec![1.0; 4]);
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
    fn checkpoint_snapshot_is_a_copy_free_consistent_cut() {
        let dir = std::env::temp_dir().join(format!("yeto-ckpt-snapshot-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("state.ckpt");
        let mut st = state_with_fingerprint(layout2(), 0x22);
        st.session_contract_hash = Some([5; 32]);
        st.init_fragment(0, vec![1.5; 4]).unwrap();
        st.init_fragment(1, vec![-2.0; 4]).unwrap();
        st.record_merge(1, 2, 30);
        let cut = st.encode_checkpoint().unwrap();

        let snapshot = st.checkpoint_snapshot().unwrap();
        assert!(Arc::ptr_eq(&snapshot.fragments[0].1, &st.params[0]));
        // A commit while the snapshot is pending clones only its fragment.
        st.merge_and_step(0, &[&[1.0f32; 4][..]], &[1.0]).unwrap();
        st.record_merge(1, 2, 30);
        assert!(!Arc::ptr_eq(&snapshot.fragments[0].1, &st.params[0]));
        assert!(Arc::ptr_eq(&snapshot.fragments[1].1, &st.params[1]));
        snapshot.save(&path).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), cut);
        assert_ne!(st.encode_checkpoint().unwrap(), cut);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn snapshot_write_counts_copied_fragments_and_releases_every_buffer() {
        let dir = std::env::temp_dir().join(format!("yeto-ckpt-persist-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("state.ckpt.step-1");
        let tmp = dir.join("state.ckpt.step-1.tmp");
        let mut st = state_with_fingerprint(layout2(), 0x23);
        st.init_fragment(0, vec![1.5; 4]).unwrap();
        st.init_fragment(1, vec![-2.0; 4]).unwrap();
        let cut = st.encode_checkpoint().unwrap();
        let size = cut.len() as u64;
        // Nothing commits during this write: nothing is copied.
        let quiet = st.checkpoint_snapshot().unwrap().persist(&path, &tmp).unwrap();
        assert_eq!(quiet, PersistedSnapshot { bytes: size, copied_fragments: 0 });
        // A commit while the snapshot holds fragment 1 copies that fragment only.
        let snapshot = st.checkpoint_snapshot().unwrap();
        st.merge_and_step(1, &[&[1.0f32; 4][..]], &[1.0]).unwrap();
        let persisted = snapshot.persist(&path, &tmp).unwrap();
        assert_eq!(persisted, PersistedSnapshot { bytes: size, copied_fragments: 1 });
        assert_eq!(std::fs::read(&path).unwrap(), cut);
        assert!(!tmp.exists());
        // The written snapshot holds no buffer, so later commits copy nothing.
        for fid in 0..2 {
            assert_eq!(Arc::strong_count(&st.params[fid]), 1);
            assert_eq!(Arc::strong_count(&st.momentum[fid]), 1);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn streamed_load_rejects_truncated_and_oversized_files() {
        let dir = std::env::temp_dir().join(format!("yeto-ckpt-stream-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("state.ckpt");
        let mut st = state_with_fingerprint(layout2(), 0x33);
        st.init_fragment(0, vec![1.0; 4]).unwrap();
        st.init_fragment(1, vec![2.0; 4]).unwrap();
        let bytes = st.encode_checkpoint().unwrap();
        for (bad, message) in [
            (bytes[..bytes.len() - 40].to_vec(), "truncated"),
            ([bytes.as_slice(), &[0u8; 100]].concat(), "trailing bytes"),
        ] {
            std::fs::write(&path, &bad).unwrap();
            let mut fresh = state_with_fingerprint(layout2(), 0x33);
            let error = format!("{:#}", fresh.load_checkpoint(&path).unwrap_err());
            assert!(error.contains(message), "{error}");
            assert!(!fresh.all_initialized());
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn sharded_delayed_nesterov_checkpoints_only_owned_fragments() {
        let dir = std::env::temp_dir().join(format!("yeto-dn-shard-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dn-shard1.ckpt");
        let fresh = || {
            let mut st = state_with_fingerprint(layout2(), 0x11);
            st.set_shard(ShardPlan::new(1, 2).unwrap()).unwrap();
            st.set_delayed_nesterov_period(Some(2)).unwrap();
            st
        };
        let mut st = fresh();
        assert!(st.init_fragment(0, vec![1.0; 4]).is_err());
        st.init_fragment(1, vec![-2.0; 4]).unwrap();
        // Fragment 1 is merged at step 2 (round robin over 2 fragments).
        st.merge_and_step(1, &[&[2.0f32; 4]], &[1.0]).unwrap();
        st.global_step = 2;
        st.versions = vec![0, 2];
        st.save_checkpoint(&path).unwrap();
        let mut resumed = fresh();
        resumed.load_checkpoint(&path).unwrap();
        assert_eq!(resumed.delayed_nesterov, st.delayed_nesterov);
        assert!(resumed.delayed_nesterov[0].accum.is_empty());
        assert_eq!(resumed.delayed_nesterov[1].accum.len(), 4);
        for state in [&mut st, &mut resumed] {
            state.merge_and_step(1, &[&[2.0f32; 4]], &[1.0]).unwrap();
        }
        assert_eq!(resumed.params, st.params);
        assert_eq!(resumed.momentum, st.momentum);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn delayed_nesterov_commits_and_checkpoints_its_state() {
        let dir = std::env::temp_dir().join(format!("yeto-dn-ckpt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (dn_path, plain_path) = (dir.join("dn.ckpt"), dir.join("plain.ckpt"));
        let fresh = |period| {
            let mut st = state_with_fingerprint(layout2(), 0x11);
            st.set_delayed_nesterov_period(period).unwrap();
            st
        };
        let mut st = fresh(Some(2));
        st.init_fragment(0, vec![1.0; 4]).unwrap();
        st.init_fragment(1, vec![-2.0; 4]).unwrap();
        // Step t=0 of N=2: θ -= lr·g/N, momentum untouched, Δ holds g.
        st.merge_and_step(0, &[&[2.0f32; 4]], &[1.0]).unwrap();
        assert_eq!(*st.params[0], vec![1.0 - 0.7 * 1.0; 4]);
        assert_eq!(*st.momentum[0], vec![0.0; 4]);
        assert_eq!(*st.delayed_nesterov[0].accum, vec![2.0; 4]);
        st.global_step = 1;
        st.versions = vec![1, 0];
        st.save_checkpoint(&dn_path).unwrap();

        let mut resumed = fresh(Some(2));
        resumed.load_checkpoint(&dn_path).unwrap();
        assert_eq!(resumed.delayed_nesterov, st.delayed_nesterov);
        // Step t=1 fires: m = Δ/N = 2, Δ cleared, on both states alike.
        for state in [&mut st, &mut resumed] {
            state.merge_and_step(0, &[&[2.0f32; 4]], &[1.0]).unwrap();
            assert_eq!(*state.momentum[0], vec![2.0; 4]);
            assert_eq!(*state.delayed_nesterov[0].accum, vec![0.0; 4]);
        }
        assert_eq!(resumed.params, st.params);

        // A plain checkpoint keeps its bytes and cannot resume a DN run;
        // a DN checkpoint cannot resume a plain or other-period run.
        let mut plain = fresh(None);
        plain.init_fragment(0, vec![1.0; 4]).unwrap();
        plain.init_fragment(1, vec![-2.0; 4]).unwrap();
        plain.save_checkpoint(&plain_path).unwrap();
        let mut reference = state_with_fingerprint(layout2(), 0x11);
        reference.init_fragment(0, vec![1.0; 4]).unwrap();
        reference.init_fragment(1, vec![-2.0; 4]).unwrap();
        assert_eq!(std::fs::read(&plain_path).unwrap(), reference.encode_checkpoint().unwrap());
        assert!(fresh(Some(2)).load_checkpoint(&plain_path).is_err());
        assert!(fresh(None).load_checkpoint(&dn_path).is_err());
        assert!(fresh(Some(3)).load_checkpoint(&dn_path).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn delayed_nesterov_period_must_be_at_least_two_and_set_before_init() {
        let mut st = state_with_fingerprint(layout2(), 0x11);
        assert!(st.set_delayed_nesterov_period(Some(1)).is_err());
        st.init_fragment(0, vec![1.0; 4]).unwrap();
        assert!(st.set_delayed_nesterov_period(Some(2)).is_err());
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
