//! Global model state held by the syncer: fragment layout, parameters Θ,
//! and outer-optimizer momentum, all in f32.

use anyhow::{bail, Result};

use crate::merge;
use crate::protocol::Reader;

pub const MERGE_AVG: u8 = 0;
pub const MERGE_RDA: u8 = 1;
pub const MERGE_ISO: u8 = 2;
/// Worker-SNR consensus merge (see `merge::merge_worker_snr`). Like avg/RDA it
/// carries no per-tensor shapes on the wire; unlike them it consumes all
/// tensor blocks and the per-worker deltas at once.
pub const MERGE_WORKER_SNR: u8 = 3;

#[derive(Clone, Debug, PartialEq)]
pub struct FragmentInfo {
    pub merge_mode: u8,
    pub tensor_numels: Vec<u64>,
    /// Per-tensor (rows, cols) matrix shapes, present exactly for
    /// `MERGE_ISO` fragments (the Iso-C transform needs the 2D view; avg
    /// and RDA operate on flat slices and carry no shapes on the wire).
    pub tensor_shapes: Option<Vec<(u64, u64)>>,
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
            if merge_mode > MERGE_WORKER_SNR {
                bail!("bad merge mode {merge_mode}");
            }
            let num_tensors = r.u32()?;
            let mut tensor_numels = Vec::with_capacity(num_tensors as usize);
            for _ in 0..num_tensors {
                tensor_numels.push(r.u64()?);
            }
            // Iso fragments carry an extra (rows, cols) pair per tensor;
            // avg/RDA fragments keep the original wire format untouched.
            let tensor_shapes = if merge_mode == MERGE_ISO {
                let mut shapes = Vec::with_capacity(num_tensors as usize);
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

/// One validated learner candidate presented to the deterministic merge API.
/// Candidate slices must be ordered by strictly increasing `responder_id`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MergeCandidate<'a> {
    pub responder_id: u32,
    pub values: &'a [f32],
    pub weight: f64,
}

#[allow(dead_code)] // Public action API; server wiring lands separately.
impl<'a> MergeCandidate<'a> {
    pub const fn new(responder_id: u32, values: &'a [f32], weight: f64) -> Self {
        Self {
            responder_id,
            values,
            weight,
        }
    }
}

/// Pure result of the production per-tensor merge, before the outer optimizer.
#[derive(Clone, Debug, PartialEq)]
pub struct AggregateDelta {
    fragment_id: usize,
    base_version: u64,
    base_state_epoch: u64,
    base_state_fingerprint: [u64; 2],
    responder_ids: Vec<u32>,
    selected_weight: f64,
    selected_weight_mass: f64,
    delta: Vec<f32>,
    gnorm: f64,
    /// Worker-disagreement transverse curvature-energy proxy
    /// `E_b = b_perp^T C b_perp` (`merge::disagreement_transverse_energy`),
    /// computed here where the per-worker deltas exist and threaded to the
    /// `capped-nesterov-wsub` outer step. `0.0` for every other optimizer
    /// (the energy is not computed) and whenever there is no measurable
    /// disagreement. Deterministic from the fingerprinted base state and
    /// candidates, so it needs no separate seal.
    disagreement_energy: f64,
}

#[allow(dead_code)] // Public action API; server wiring lands separately.
impl AggregateDelta {
    pub const fn fragment_id(&self) -> usize {
        self.fragment_id
    }

    pub const fn base_version(&self) -> u64 {
        self.base_version
    }

    pub const fn base_state_epoch(&self) -> u64 {
        self.base_state_epoch
    }

    pub fn responder_ids(&self) -> &[u32] {
        &self.responder_ids
    }

    pub const fn selected_weight(&self) -> f64 {
        self.selected_weight
    }

    pub const fn selected_weight_mass(&self) -> f64 {
        self.selected_weight_mass
    }

    pub fn delta(&self) -> &[f32] {
        &self.delta
    }

    pub const fn gnorm(&self) -> f64 {
        self.gnorm
    }

    pub const fn disagreement_energy(&self) -> f64 {
        self.disagreement_energy
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CttnInputs {
    pub g: Vec<f32>,
    pub b: Vec<f32>,
    pub outer_lr: f32,
    pub mu: f32,
}

/// Caller-configurable admissible range for one adaptive step multiplier.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepScaleBounds {
    min: f64,
    max: f64,
}

#[allow(dead_code)] // Public action API; server wiring lands separately.
impl StepScaleBounds {
    pub fn new(min: f64, max: f64) -> Result<Self> {
        if !min.is_finite() || !max.is_finite() || min <= 0.0 || max < min {
            bail!("step-scale bounds must be finite and satisfy 0 < min <= max");
        }
        Ok(Self { min, max })
    }

    pub const fn min(self) -> f64 {
        self.min
    }

    pub const fn max(self) -> f64 {
        self.max
    }

    pub fn contains(self, scalar: f64) -> bool {
        scalar >= self.min && scalar <= self.max
    }
}

/// Action-level statistics that augment the legacy merge/optimizer metrics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActionPreviewStats {
    pub merge: MergeStats,
    /// Multiplier relative to the original unscaled action step.
    pub step_scale: f64,
    /// Nominal norm before any action-specific step scaling.
    pub unscaled_applied_step_norm: f64,
    /// Seal over the action metadata, tensors, optimizer buffer, and stats.
    pub action_fingerprint: [u64; 2],
}

/// A complete, immutable outer-step action computed from one state snapshot.
/// Committing consumes the preview so the exact parameters and optimizer
/// buffer evaluated by a caller are the values installed in `GlobalState`.
#[derive(Clone, Debug, PartialEq)]
pub struct ActionPreview {
    fragment_id: usize,
    base_version: u64,
    base_state_epoch: u64,
    base_state_fingerprint: [u64; 2],
    target_version: u64,
    responder_ids: Vec<u32>,
    selected_weight: f64,
    selected_weight_mass: f64,
    resulting_params: Vec<f32>,
    resulting_optimizer_buffer: Vec<f32>,
    /// Rho-adaptive EMA value after this step; unchanged for the other
    /// optimizers. Installed alongside the buffer on commit.
    resulting_rho_ema: f32,
    /// Capped-Nesterov-family effective momentum after this step; unchanged
    /// for the other optimizers. Installed alongside the buffer on commit.
    resulting_capped_mu: f32,
    /// Capped-Nesterov-gc applied step gain for this step; unchanged for the
    /// other optimizers. Installed alongside the buffer on commit.
    resulting_capped_gain: f32,
    /// Block second-moment (`block-rms`/`block-yogi`) per-tensor state after
    /// this step; unchanged for the other optimizers. One entry per tensor in
    /// the fragment. Installed alongside the buffer on commit.
    resulting_block_v: Vec<f32>,
    /// Chebyshev-SGD cycle-phase counter after this step; unchanged for the
    /// other optimizers. Installed alongside the buffer on commit.
    resulting_cheb_phase: f32,
    /// Curvature-aware controller (`capped-nesterov-curv`) history after this
    /// step: the merged delta `g_t` and applied step `dtheta_t` this commit
    /// stores for the next commit's lambda_hat. Unchanged (carried through) for
    /// the other optimizers. Installed alongside the buffer on commit.
    resulting_curv_prev_delta: Vec<f32>,
    resulting_curv_prev_dtheta: Vec<f32>,
    applied_step: Vec<f32>,
    stats: MergeStats,
    step_scale: f64,
    unscaled_applied_step_norm: f64,
    action_fingerprint: [u64; 2],
}

#[allow(dead_code)] // Public action API; server wiring lands separately.
impl ActionPreview {
    pub const fn fragment_id(&self) -> usize {
        self.fragment_id
    }

    pub const fn base_version(&self) -> u64 {
        self.base_version
    }

    pub const fn base_state_epoch(&self) -> u64 {
        self.base_state_epoch
    }

    pub const fn target_version(&self) -> u64 {
        self.target_version
    }

    pub fn responder_ids(&self) -> &[u32] {
        &self.responder_ids
    }

    pub const fn selected_weight(&self) -> f64 {
        self.selected_weight
    }

    pub const fn selected_weight_mass(&self) -> f64 {
        self.selected_weight_mass
    }

    pub fn resulting_params(&self) -> &[f32] {
        &self.resulting_params
    }

    pub fn resulting_optimizer_buffer(&self) -> &[f32] {
        &self.resulting_optimizer_buffer
    }

    /// Nominal f32 displacement applied to the base parameters. It is kept
    /// explicitly so norm matching never mixes nominal optimizer statistics
    /// with a rounded `base - resulting_params` reconstruction.
    pub fn applied_step(&self) -> &[f32] {
        &self.applied_step
    }

    pub const fn stats(&self) -> MergeStats {
        self.stats
    }

    pub const fn action_stats(&self) -> ActionPreviewStats {
        ActionPreviewStats {
            merge: self.stats,
            step_scale: self.step_scale,
            unscaled_applied_step_norm: self.unscaled_applied_step_norm,
            action_fingerprint: self.action_fingerprint,
        }
    }

    pub const fn action_fingerprint(&self) -> [u64; 2] {
        self.action_fingerprint
    }

    /// Multiplicative change to this action's original applied outer step.
    /// It is `1.0` for an ordinary preview.
    pub const fn step_scale(&self) -> f64 {
        self.step_scale
    }

    /// Backward-compatible name for callers that only use LOO norm matching.
    pub const fn norm_match_scale(&self) -> f64 {
        self.step_scale
    }
}

#[allow(dead_code)] // Used by the not-yet-wired leave-one-out action API.
fn is_ordered_subset(subset: &[u32], full: &[u32]) -> bool {
    let mut full_index = 0usize;
    for &id in subset {
        while full_index < full.len() && full[full_index] < id {
            full_index += 1;
        }
        if full_index == full.len() || full[full_index] != id {
            return false;
        }
        full_index += 1;
    }
    true
}

#[allow(dead_code)] // Used by the not-yet-wired leave-one-out action API.
fn flat_l2_norm(values: &[f32]) -> f64 {
    values
        .iter()
        .map(|value| (*value as f64).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn mix_action_fingerprint(hash: &mut [u64; 2], word: u64) {
    hash[0] ^= word;
    hash[0] = hash[0].wrapping_mul(0x0000_0100_0000_01b3);
    hash[1] ^= word.wrapping_add(0x517c_c1b7_2722_0a95);
    hash[1] = hash[1].rotate_left(27).wrapping_mul(0x94d0_49bb_1331_11eb);
}

fn mix_optional_f64(hash: &mut [u64; 2], value: Option<f64>) {
    match value {
        None => mix_action_fingerprint(hash, 0),
        Some(value) => {
            mix_action_fingerprint(hash, 1);
            mix_action_fingerprint(hash, value.to_bits());
        }
    }
}

fn compute_action_fingerprint(preview: &ActionPreview) -> [u64; 2] {
    let mut hash = [0x2f6e_2b1d_58a4_9c73u64, 0x7c15_9e37_79b9_4a7cu64];
    for word in [
        preview.fragment_id as u64,
        preview.base_version,
        preview.base_state_epoch,
        preview.base_state_fingerprint[0],
        preview.base_state_fingerprint[1],
        preview.target_version,
        preview.responder_ids.len() as u64,
    ] {
        mix_action_fingerprint(&mut hash, word);
    }
    for &responder_id in &preview.responder_ids {
        mix_action_fingerprint(&mut hash, responder_id as u64);
    }
    for value in [
        preview.selected_weight,
        preview.selected_weight_mass,
        preview.stats.gnorm,
        preview.stats.outer.applied_step_norm,
        preview.step_scale,
        preview.unscaled_applied_step_norm,
    ] {
        mix_action_fingerprint(&mut hash, value.to_bits());
    }
    mix_optional_f64(&mut hash, preview.stats.outer.direction_delta_cosine);
    mix_optional_f64(&mut hash, preview.stats.outer.history_current_norm_ratio);
    mix_action_fingerprint(&mut hash, preview.stats.outer.restarted as u64);
    mix_action_fingerprint(&mut hash, preview.resulting_rho_ema.to_bits() as u64);
    mix_action_fingerprint(&mut hash, preview.resulting_capped_mu.to_bits() as u64);
    mix_action_fingerprint(&mut hash, preview.resulting_capped_gain.to_bits() as u64);
    mix_action_fingerprint(&mut hash, preview.resulting_cheb_phase.to_bits() as u64);
    mix_action_fingerprint(&mut hash, preview.resulting_block_v.len() as u64);
    for value in &preview.resulting_block_v {
        mix_action_fingerprint(&mut hash, value.to_bits() as u64);
    }
    for values in [
        preview.resulting_params.as_slice(),
        preview.resulting_optimizer_buffer.as_slice(),
        preview.resulting_curv_prev_delta.as_slice(),
        preview.resulting_curv_prev_dtheta.as_slice(),
        preview.applied_step.as_slice(),
    ] {
        mix_action_fingerprint(&mut hash, values.len() as u64);
        for value in values {
            mix_action_fingerprint(&mut hash, value.to_bits() as u64);
        }
    }
    hash
}

pub struct GlobalState {
    pub layout: Layout,
    pub layout_meta: Option<String>,
    /// Θ_p, flat f32 per fragment (concatenated tensors in layout order).
    pub params: Vec<Vec<f32>>,
    /// Outer-optimizer buffers, same shape as params.
    momentum: Vec<Vec<f32>>,
    /// Counterfactual mu=0.9 Nesterov buffer used only by CTTN shadow
    /// diagnostics. It is updated from observed SGD pseudo-gradients and is
    /// never consulted by the applied optimizer.
    cttn_shadow_momentum: Vec<Vec<f32>>,
    /// Per-fragment rho-adaptive EMA of the measured round-to-round delta
    /// autocorrelation. Only the rho-adaptive optimizer reads or writes it.
    /// Deliberately NOT checkpointed (the on-disk format is shared with
    /// yeto/export.py): with beta 0.5 its half-life is one commit, so a
    /// restore behaves like the tuned baseline for one commit and then
    /// re-converges.
    rho_ema: Vec<f32>,
    /// Per-fragment capped-Nesterov-family effective momentum after the most
    /// recent commit (the one-sided release EMA state). Only the
    /// capped-nesterov family reads or writes it. Like `rho_ema` it is
    /// deliberately NOT checkpointed: a restore re-initializes to the
    /// design-point mu_max and the caps re-bind on the first measured commit.
    capped_mu: Vec<f32>,
    /// Per-fragment capped-Nesterov-gc applied step gain after the most
    /// recent commit. Only the gc variant writes it (it is recomputed per
    /// commit, not consumed); kept per fragment so materialized previews and
    /// diagnostics see the exact applied value. Not checkpointed, like
    /// `capped_mu`.
    capped_gain: Vec<f32>,
    /// Per-fragment, per-tensor scalar second-moment state for the block
    /// second-moment optimizers (`block-rms`/`block-yogi`). `block_v[fid]` has
    /// one entry per tensor in fragment `fid`. Only those optimizers read or
    /// write it; it holds no directional memory (a scalar per block). Like the
    /// other outer-optimizer scalar states it is NOT part of the checkpoint
    /// format; a restore re-initializes to zero and the EMAs re-warm on the
    /// first measured commit.
    block_v: Vec<Vec<f32>>,
    /// Per-fragment previous merged delta `g_{t-1}` and previous applied outer
    /// step `dtheta_{t-1}` (as a parameter displacement), the two extra vector
    /// states of the curvature-aware controller (`capped-nesterov-curv`). Only
    /// that optimizer reads or writes them. Same shape as `params`; the online
    /// secant curvature proxy `lambda_hat` is computed from them each commit.
    /// Like the other outer-optimizer states above they are NOT part of the
    /// checkpoint format: a restore re-initializes both to zero, so the
    /// curvature transverse cap is inactive for one commit (lambda_hat = 0) and
    /// then re-warms — exactly the fresh-start behavior of `capped_mu`.
    curv_prev_delta: Vec<Vec<f32>>,
    curv_prev_dtheta: Vec<Vec<f32>>,
    /// Per-fragment Chebyshev-SGD cycle-phase counter (0..CHEB_SGD_CYCLE, as
    /// f32). Only `cheb-sgd` reads or writes it; it holds no directional memory
    /// (a scalar phase). Like the other outer-optimizer scalar states it is NOT
    /// part of the checkpoint format: a restore re-initializes to 0, restarting
    /// the cycle from its smallest (safest) step.
    cheb_phase: Vec<f32>,
    pub initialized: Vec<bool>,
    /// Global step at which each fragment was last merged (its version).
    pub versions: Vec<u64>,
    /// Last completed global step t (checkpoint cut point).
    pub global_step: u64,
    /// Per-fragment in-process mutation token used to invalidate previews.
    /// This is intentionally independent from the externally visible version:
    /// legacy callers still update `versions` after `merge_and_step` returns.
    state_epochs: Vec<u64>,
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
    /// Post-merge renormalization for mediation-control experiments
    /// (EXP2.39): when > 0, the merged delta of every commit is rescaled to
    /// this L2 norm (per fragment, after the production merge, before the
    /// outer-optimizer step). The scale is deterministic from the delta
    /// (R / ‖δ‖, computed once and shared by the step and its materialized
    /// preview, so previews stay bit-exact). Zero-norm deltas are left
    /// untouched. 0 = off (the default; byte-identical to the pre-flag
    /// production path). `stats.gnorm` still reports the PRE-rescale merged
    /// delta norm.
    pub delta_norm_ref: f32,
    /// EXP2.46 3-arm current-anchor causal control. When true, each learner's
    /// delta is differenced against the RETAINED global fragment value at the
    /// learner's pushed base_version (version-matched anchoring) rather than
    /// the current global (current-anchor). Default false = byte-identical
    /// current-anchor production path. See docs/ANCHOR_DRIFT_CONTROL.md.
    pub version_matched_anchor: bool,
    /// EXP2.46: retain prior global snapshots and compute per-push anchor-drift
    /// instrumentation even when NOT version-matching, so the current-anchor
    /// arm still reports the drift it injects into every delta. Implied by
    /// `version_matched_anchor`.
    pub anchor_drift_instrument: bool,
    /// Bounded ring of prior committed global fragment snapshots per fragment,
    /// keyed by the version they represent, newest at the back. Populated only
    /// while anchor retention is enabled (else always empty -> zero cost and a
    /// byte-identical path). Holds at most `ANCHOR_HISTORY_DEPTH` entries,
    /// which covers the non-barrier overlap window (pipeline depth x quorum
    /// grace); a base_version older than the window is reported unresolved and
    /// falls back to the current anchor.
    anchor_history: Vec<std::collections::VecDeque<(u64, Vec<f32>)>>,
}

/// EXP2.46: retained prior-global depth per fragment. The non-barrier overlap
/// window (pipeline depth x in-flight rounds) never approaches this, so a
/// learner's base version is always still resident when its push lands.
const ANCHOR_HISTORY_DEPTH: usize = 8;

impl GlobalState {
    pub fn new(
        layout: Layout,
        layout_meta: Option<String>,
        outer_lr: f32,
        outer_momentum: f32,
        wire_dtype: u8,
    ) -> Self {
        let params: Vec<Vec<f32>> = layout
            .fragments
            .iter()
            .map(|f| vec![0.0; f.numel()])
            .collect();
        let momentum = params.clone();
        let cttn_shadow_momentum = params.clone();
        let rho_ema = vec![merge::RHO_ADAPTIVE_INITIAL_RHO_EMA; layout.fragments.len()];
        let capped_mu = vec![merge::CAPPED_NESTEROV_INITIAL_MU; layout.fragments.len()];
        let capped_gain =
            vec![merge::CAPPED_NESTEROV_GC_INITIAL_GAIN; layout.fragments.len()];
        let block_v = layout
            .fragments
            .iter()
            .map(|f| vec![0.0f32; f.tensor_numels.len()])
            .collect();
        // Curvature-aware controller history: zero-filled per fragment (same
        // shape as params), so lambda_hat = 0 on the first measured commit.
        let curv_prev_delta = params.clone();
        let curv_prev_dtheta = params.clone();
        let cheb_phase = vec![0.0f32; layout.fragments.len()];
        let initialized = vec![false; layout.fragments.len()];
        let versions = vec![0; layout.fragments.len()];
        let state_epochs = vec![0; layout.fragments.len()];
        let anchor_history = (0..layout.fragments.len())
            .map(|_| std::collections::VecDeque::new())
            .collect();
        Self {
            layout,
            layout_meta,
            params,
            momentum,
            cttn_shadow_momentum,
            rho_ema,
            capped_mu,
            capped_gain,
            block_v,
            curv_prev_delta,
            curv_prev_dtheta,
            cheb_phase,
            initialized,
            versions,
            global_step: 0,
            state_epochs,
            ledger: Default::default(),
            outer_lr,
            outer_lr_by_fragment: None,
            outer_momentum,
            outer_optimizer: merge::OuterOptimizer::Nesterov,
            outer_restart_cos_threshold: 0.0,
            wire_dtype,
            delta_correction: None,
            delta_norm_ref: 0.0,
            version_matched_anchor: false,
            anchor_drift_instrument: false,
            anchor_history,
        }
    }

    /// EXP2.46: is prior-global retention active? Retention feeds both
    /// version-matched anchoring and current-anchor drift instrumentation.
    pub fn anchor_retention_enabled(&self) -> bool {
        self.version_matched_anchor || self.anchor_drift_instrument
    }

    /// EXP2.46: the global fragment value at `version` — the current params if
    /// `version` is the current version, else a retained prior snapshot, else
    /// None once the version has aged out of the bounded history.
    pub fn anchor_at(&self, fid: usize, version: u64) -> Option<&[f32]> {
        if version == self.versions[fid] {
            return Some(&self.params[fid]);
        }
        self.anchor_history[fid]
            .iter()
            .rev()
            .find(|(v, _)| *v == version)
            .map(|(_, params)| params.as_slice())
    }

    /// EXP2.46: outer-momentum buffer of a fragment, for the anchor-drift /
    /// momentum cosine diagnostic. Read-only.
    pub fn momentum_fragment(&self, fid: usize) -> &[f32] {
        &self.momentum[fid]
    }

    /// EXP2.46: retain the CURRENT global fragment snapshot (under its current
    /// version) before an outer step overwrites it, so a push tagged at that
    /// version can still be differenced against it. No-op unless retention is
    /// enabled; the ring is bounded to `ANCHOR_HISTORY_DEPTH` (oldest evicted).
    fn retain_anchor_snapshot(&mut self, fid: usize) {
        if !self.anchor_retention_enabled() {
            return;
        }
        let version = self.versions[fid];
        let ring = &mut self.anchor_history[fid];
        // A version is recorded once; a re-entrant install at the same version
        // (which should not occur) refreshes rather than duplicates it.
        if ring.back().map(|(v, _)| *v) == Some(version) {
            return;
        }
        ring.push_back((version, self.params[fid].clone()));
        while ring.len() > ANCHOR_HISTORY_DEPTH {
            ring.pop_front();
        }
    }

    pub fn all_initialized(&self) -> bool {
        self.initialized.iter().all(|&b| b)
    }

    pub fn init_fragment(&mut self, fid: usize, values: Vec<f32>) -> Result<()> {
        if fid >= self.params.len() {
            bail!("init fragment {fid}: fragment id out of range");
        }
        if values.len() != self.params[fid].len() {
            bail!(
                "init fragment {fid}: got {} values, expected {}",
                values.len(),
                self.params[fid].len()
            );
        }
        if !self.initialized[fid] {
            let next_epoch = self.next_state_epoch(fid)?;
            self.params[fid] = values;
            self.initialized[fid] = true;
            self.state_epochs[fid] = next_epoch;
        }
        Ok(())
    }

    pub fn record_merge(&mut self, learner_id: u32, c_steps: u32, c_tokens: u64) {
        let e = self.ledger.entry(learner_id).or_default();
        e.merges += 1;
        e.steps += c_steps as u64;
        e.tokens += c_tokens;
    }

    /// Build the exact production aggregate for all candidates in a group.
    /// Candidates must be sorted by strictly increasing responder ID.
    pub fn build_full_aggregate(
        &self,
        fid: usize,
        candidates: &[MergeCandidate<'_>],
    ) -> Result<AggregateDelta> {
        let full_weight = self.validate_candidate_group(fid, candidates)?;
        self.build_aggregate_from_subset(fid, candidates, full_weight)
    }

    /// Build the exact production aggregate for a sorted responder subset.
    ///
    /// `candidates` is the complete sorted group. `selected_responder_ids`
    /// must also be strictly sorted and contain only IDs from that group. The
    /// resulting weight mass is measured against the complete group.
    #[allow(dead_code)] // Public action API; server wiring lands separately.
    pub fn build_selected_aggregate(
        &self,
        fid: usize,
        candidates: &[MergeCandidate<'_>],
        selected_responder_ids: &[u32],
    ) -> Result<AggregateDelta> {
        let full_weight = self.validate_candidate_group(fid, candidates)?;
        if selected_responder_ids.is_empty() {
            bail!("fragment {fid}: selected responder subset is empty");
        }
        for pair in selected_responder_ids.windows(2) {
            if pair[0] >= pair[1] {
                bail!("fragment {fid}: selected responder IDs must be strictly increasing");
            }
        }

        let mut selected = Vec::with_capacity(selected_responder_ids.len());
        let mut candidate_index = 0usize;
        for &selected_id in selected_responder_ids {
            while candidate_index < candidates.len()
                && candidates[candidate_index].responder_id < selected_id
            {
                candidate_index += 1;
            }
            if candidate_index == candidates.len()
                || candidates[candidate_index].responder_id != selected_id
            {
                bail!("fragment {fid}: selected responder {selected_id} is not in the candidate group");
            }
            selected.push(candidates[candidate_index]);
            candidate_index += 1;
        }
        self.build_aggregate_from_subset(fid, &selected, full_weight)
    }

    /// Preview the exact production outer step for `target_version` without
    /// mutating global state. The target is bound into the returned action so
    /// an abandoned round cannot later be relabeled at commit time.
    #[allow(dead_code)] // Public action API; server wiring lands separately.
    pub fn preview_aggregate(
        &self,
        aggregate: &AggregateDelta,
        target_version: u64,
    ) -> Result<ActionPreview> {
        if target_version <= aggregate.base_version {
            bail!(
                "fragment {}: preview target version {target_version} must be newer than base version {}",
                aggregate.fragment_id,
                aggregate.base_version
            );
        }
        self.preview_aggregate_inner(aggregate, target_version)
    }

    fn outer_lr_for_fragment(&self, fid: usize) -> Result<f32> {
        let outer_lr = if let Some(rates) = &self.outer_lr_by_fragment {
            *rates.get(fid).ok_or_else(|| {
                anyhow::anyhow!(
                    "outer-lr-by-fragment has {} entries, missing fragment {fid}",
                    rates.len()
                )
            })?
        } else {
            self.outer_lr
        };
        if !outer_lr.is_finite() || outer_lr < 0.0 {
            bail!("fragment {fid}: outer learning rate must be finite and non-negative");
        }
        Ok(outer_lr)
    }

    fn post_renormalized_delta(&self, fid: usize, delta: &[f32]) -> Result<Vec<f32>> {
        if !self.delta_norm_ref.is_finite() || self.delta_norm_ref < 0.0 {
            bail!("fragment {fid}: delta-norm-ref must be finite and non-negative");
        }
        if self.delta_norm_ref <= 0.0 {
            return Ok(delta.to_vec());
        }
        let raw_norm = flat_l2_norm(delta);
        if raw_norm == 0.0 {
            return Ok(delta.to_vec());
        }
        let scale = (self.delta_norm_ref as f64 / raw_norm) as f32;
        if !scale.is_finite() {
            bail!("fragment {fid}: delta-norm-ref rescale is not finite");
        }
        Ok(delta.iter().map(|value| scale * *value).collect())
    }

    /// Freeze the exact CTTN inputs from one aggregate. `g` is the same
    /// post-renormalization, anchor-minus-upload delta that the fallback outer
    /// optimizer consumes; `b` is the incoming fragment momentum buffer.
    pub fn cttn_inputs(&self, aggregate: &AggregateDelta, mu: f32) -> Result<CttnInputs> {
        self.ensure_current_base(
            aggregate.fragment_id,
            aggregate.base_version,
            aggregate.base_state_epoch,
            aggregate.base_state_fingerprint,
        )?;
        let fid = aggregate.fragment_id;
        if !mu.is_finite() || !(0.0..1.0).contains(&mu) {
            bail!("fragment {fid}: CTTN mu must be finite and in [0, 1)");
        }
        Ok(CttnInputs {
            g: self.post_renormalized_delta(fid, &aggregate.delta)?,
            b: self.momentum[fid].clone(),
            outer_lr: self.outer_lr_for_fragment(fid)?,
            mu,
        })
    }

    /// Freeze shadow inputs using the independent counterfactual Nesterov
    /// buffer rather than the applied optimizer's buffer.
    pub fn cttn_shadow_inputs(&self, aggregate: &AggregateDelta, mu: f32) -> Result<CttnInputs> {
        self.ensure_current_base(
            aggregate.fragment_id,
            aggregate.base_version,
            aggregate.base_state_epoch,
            aggregate.base_state_fingerprint,
        )?;
        let fid = aggregate.fragment_id;
        if !mu.is_finite() || !(0.0..1.0).contains(&mu) {
            bail!("fragment {fid}: CTTN shadow mu must be finite and in [0, 1)");
        }
        Ok(CttnInputs {
            g: self.post_renormalized_delta(fid, &aggregate.delta)?,
            b: self.cttn_shadow_momentum[fid].clone(),
            outer_lr: self.outer_lr_for_fragment(fid)?,
            mu,
        })
    }

    fn preview_aggregate_inner(
        &self,
        aggregate: &AggregateDelta,
        target_version: u64,
    ) -> Result<ActionPreview> {
        self.ensure_current_base(
            aggregate.fragment_id,
            aggregate.base_version,
            aggregate.base_state_epoch,
            aggregate.base_state_fingerprint,
        )?;
        let fid = aggregate.fragment_id;
        let outer_lr = self.outer_lr_for_fragment(fid)?;
        if !self.outer_momentum.is_finite() || !self.outer_restart_cos_threshold.is_finite() {
            bail!("fragment {fid}: outer optimizer configuration is not finite");
        }
        // Post-merge renormalization (mediation-control experiments): rescale
        // the merged delta to L2 norm `delta_norm_ref` before the outer step.
        // The scale is a single f32 deterministic from the delta; the SAME
        // rescaled slice feeds both the applied step and the materialized
        // preview below, so preview bit-exactness is preserved by
        // construction. gnorm (aggregate.gnorm) keeps the pre-rescale norm.
        let renormalized_delta = self.post_renormalized_delta(fid, &aggregate.delta)?;
        let delta = renormalized_delta.as_slice();
        let mut resulting_params = self.params[fid].clone();
        let mut resulting_optimizer_buffer = self.momentum[fid].clone();
        let mut resulting_rho_ema = self.rho_ema[fid];
        let mut resulting_capped_mu = self.capped_mu[fid];
        let mut resulting_capped_gain = self.capped_gain[fid];
        let mut resulting_block_v = self.block_v[fid].clone();
        let mut resulting_curv_prev_delta = self.curv_prev_delta[fid].clone();
        let mut resulting_curv_prev_dtheta = self.curv_prev_dtheta[fid].clone();
        let mut resulting_cheb_phase = self.cheb_phase[fid];
        let tensor_numels: Vec<usize> = self.layout.fragments[fid]
            .tensor_numels
            .iter()
            .map(|&n| n as usize)
            .collect();
        let outer = merge::apply_outer_step(
            self.outer_optimizer,
            &mut resulting_params,
            &mut resulting_optimizer_buffer,
            delta,
            outer_lr,
            self.outer_momentum,
            self.outer_restart_cos_threshold,
            &mut resulting_rho_ema,
            &mut resulting_capped_mu,
            &mut resulting_capped_gain,
            &mut resulting_curv_prev_delta,
            &mut resulting_curv_prev_dtheta,
            &tensor_numels,
            &mut resulting_block_v,
            aggregate.disagreement_energy,
            &mut resulting_cheb_phase,
        );
        // The capped-Nesterov family chooses its momentum (and, for gc, its
        // gain) per commit; materializing its step must use the effective
        // values the step just wrote, not the CLI momentum, to stay
        // bit-identical to the applied displacement.
        let materialize_momentum = if matches!(
            self.outer_optimizer,
            merge::OuterOptimizer::CappedNesterov
                | merge::OuterOptimizer::CappedNesterovGc
                | merge::OuterOptimizer::CappedNesterovR
                | merge::OuterOptimizer::CappedNesterovCurv
                | merge::OuterOptimizer::CappedNesterovWsub
        ) {
            resulting_capped_mu
        } else {
            self.outer_momentum
        };
        let applied_step = merge::materialize_applied_step(
            self.outer_optimizer,
            &resulting_optimizer_buffer,
            delta,
            outer_lr,
            materialize_momentum,
            resulting_capped_gain,
        );
        let materialized_step_norm = flat_l2_norm(&applied_step);
        let norm_tolerance = 1e-6 * outer.applied_step_norm.abs().max(1.0);
        if resulting_params.iter().any(|value| !value.is_finite())
            || resulting_optimizer_buffer
                .iter()
                .any(|value| !value.is_finite())
            || !resulting_rho_ema.is_finite()
            || !resulting_capped_mu.is_finite()
            || !resulting_capped_gain.is_finite()
            || !resulting_cheb_phase.is_finite()
            || resulting_block_v.iter().any(|value| !value.is_finite())
            || resulting_curv_prev_delta
                .iter()
                .any(|value| !value.is_finite())
            || resulting_curv_prev_dtheta
                .iter()
                .any(|value| !value.is_finite())
            || applied_step.iter().any(|value| !value.is_finite())
            || !outer.applied_step_norm.is_finite()
            || outer
                .direction_delta_cosine
                .is_some_and(|value| !value.is_finite())
            || outer
                .history_current_norm_ratio
                .is_some_and(|value| !value.is_finite())
            || (materialized_step_norm - outer.applied_step_norm).abs() > norm_tolerance
        {
            bail!("fragment {fid}: outer-step preview produced a non-finite result");
        }
        let mut preview = ActionPreview {
            fragment_id: fid,
            base_version: aggregate.base_version,
            base_state_epoch: aggregate.base_state_epoch,
            base_state_fingerprint: aggregate.base_state_fingerprint,
            target_version,
            responder_ids: aggregate.responder_ids.clone(),
            selected_weight: aggregate.selected_weight,
            selected_weight_mass: aggregate.selected_weight_mass,
            resulting_params,
            resulting_optimizer_buffer,
            resulting_rho_ema,
            resulting_capped_mu,
            resulting_capped_gain,
            resulting_block_v,
            resulting_curv_prev_delta,
            resulting_curv_prev_dtheta,
            resulting_cheb_phase,
            applied_step,
            stats: MergeStats {
                gnorm: aggregate.gnorm,
                outer,
            },
            step_scale: 1.0,
            unscaled_applied_step_norm: materialized_step_norm,
            action_fingerprint: [0; 2],
        };
        preview.action_fingerprint = compute_action_fingerprint(&preview);
        Ok(preview)
    }

    /// Commit the sidecar-computed CTTN direction without routing it through
    /// `apply_outer_step` or `materialize_applied_step`. Both the sealed
    /// displacement and resulting parameters are built from this same `d`.
    pub fn commit_cttn_step(
        &mut self,
        aggregate: &AggregateDelta,
        target_version: u64,
        d: &[f32],
        b_new: &[f32],
        outer_lr: f32,
    ) -> Result<MergeStats> {
        self.ensure_current_base(
            aggregate.fragment_id,
            aggregate.base_version,
            aggregate.base_state_epoch,
            aggregate.base_state_fingerprint,
        )?;
        if target_version <= aggregate.base_version {
            bail!(
                "fragment {}: CTTN target version {target_version} must be newer than base version {}",
                aggregate.fragment_id,
                aggregate.base_version
            );
        }
        let fid = aggregate.fragment_id;
        let expected_lr = self.outer_lr_for_fragment(fid)?;
        if outer_lr.to_bits() != expected_lr.to_bits() {
            bail!("fragment {fid}: CTTN outer learning rate changed before commit");
        }
        let numel = self.params[fid].len();
        if d.len() != numel
            || b_new.len() != numel
            || d.iter().chain(b_new).any(|value| !value.is_finite())
        {
            bail!("fragment {fid}: CTTN returned malformed or non-finite vectors");
        }

        let g = self.post_renormalized_delta(fid, &aggregate.delta)?;
        let applied_step: Vec<f32> = d.iter().map(|value| outer_lr * *value).collect();
        let resulting_params: Vec<f32> = self.params[fid]
            .iter()
            .zip(&applied_step)
            .map(|(param, step)| *param - *step)
            .collect();
        if applied_step.iter().any(|value| !value.is_finite())
            || resulting_params.iter().any(|value| !value.is_finite())
        {
            bail!("fragment {fid}: CTTN parameter materialization is non-finite");
        }

        let mut direction_norm_sq = 0.0f64;
        let mut delta_norm_sq = 0.0f64;
        let mut direction_delta_dot = 0.0f64;
        let mut history_norm_sq = 0.0f64;
        for (&direction, &delta) in d.iter().zip(&g) {
            let direction = direction as f64;
            let delta = delta as f64;
            let history = direction - delta;
            direction_norm_sq += direction * direction;
            delta_norm_sq += delta * delta;
            direction_delta_dot += direction * delta;
            history_norm_sq += history * history;
        }
        let direction_delta_cosine = if direction_norm_sq > 0.0 && delta_norm_sq > 0.0 {
            Some(
                (direction_delta_dot / (direction_norm_sq * delta_norm_sq).sqrt()).clamp(-1.0, 1.0),
            )
        } else {
            None
        };
        let history_current_norm_ratio = if delta_norm_sq > 0.0 {
            Some((history_norm_sq / delta_norm_sq).sqrt())
        } else {
            None
        };
        let step_norm = flat_l2_norm(&applied_step);
        let mut preview = ActionPreview {
            fragment_id: fid,
            base_version: aggregate.base_version,
            base_state_epoch: aggregate.base_state_epoch,
            base_state_fingerprint: aggregate.base_state_fingerprint,
            target_version,
            responder_ids: aggregate.responder_ids.clone(),
            selected_weight: aggregate.selected_weight,
            selected_weight_mass: aggregate.selected_weight_mass,
            resulting_params,
            resulting_optimizer_buffer: b_new.to_vec(),
            resulting_rho_ema: self.rho_ema[fid],
            resulting_capped_mu: self.capped_mu[fid],
            resulting_capped_gain: self.capped_gain[fid],
            resulting_block_v: self.block_v[fid].clone(),
            resulting_curv_prev_delta: self.curv_prev_delta[fid].clone(),
            resulting_curv_prev_dtheta: self.curv_prev_dtheta[fid].clone(),
            resulting_cheb_phase: self.cheb_phase[fid],
            applied_step,
            stats: MergeStats {
                gnorm: aggregate.gnorm,
                outer: merge::OuterStepStats {
                    applied_step_norm: step_norm,
                    direction_delta_cosine,
                    history_current_norm_ratio,
                    restarted: false,
                },
            },
            step_scale: 1.0,
            unscaled_applied_step_norm: step_norm,
            action_fingerprint: [0; 2],
        };
        preview.action_fingerprint = compute_action_fingerprint(&preview);
        self.commit_preview(preview)
    }

    /// Commit the CTTN shadow policy's applied action: exact plain SGD
    /// (`d == g`) while independently advancing the counterfactual momentum
    /// buffer `b <- mu*b + g`. No CTTN vector enters params or optimizer state.
    pub fn commit_cttn_shadow_sgd(
        &mut self,
        aggregate: &AggregateDelta,
        target_version: u64,
        mu: f32,
    ) -> Result<MergeStats> {
        let inputs = self.cttn_shadow_inputs(aggregate, mu)?;
        let fid = aggregate.fragment_id;
        let next_shadow: Vec<f32> = inputs
            .b
            .iter()
            .zip(&inputs.g)
            .map(|(buffer, gradient)| mu * *buffer + *gradient)
            .collect();
        // Plain SGD's ordinary Nesterov(mu=0) storage is exactly g. Passing
        // it here keeps the checkpointed applied-optimizer buffer conventional
        // while the independent shadow buffer carries mu=0.9 history.
        let stats = self.commit_cttn_step(
            aggregate,
            target_version,
            &inputs.g,
            &inputs.g,
            inputs.outer_lr,
        )?;
        self.cttn_shadow_momentum[fid] = next_shadow;
        Ok(stats)
    }

    /// Purely scale the applied parameter step of a complete full-group
    /// action. The aggregate's optimizer-buffer transition is deliberately
    /// retained unchanged; only this action's effective outer step changes.
    ///
    /// Zero is rejected: a zero parameter step with a nonzero buffer update is
    /// a distinct momentum-only action and should be represented explicitly,
    /// not smuggled in as adaptive step-size shrinkage.
    #[allow(dead_code)] // Public action API; server wiring lands separately.
    pub fn scale_full_group_preview(
        &self,
        preview: &ActionPreview,
        scalar: f64,
        bounds: StepScaleBounds,
    ) -> Result<ActionPreview> {
        self.ensure_current_preview(preview)?;
        if !scalar.is_finite() || scalar <= 0.0 {
            bail!("step-scale scalar must be finite and positive; got {scalar}");
        }
        if !bounds.contains(scalar) {
            bail!(
                "step-scale scalar {scalar} is outside [{}, {}]",
                bounds.min(),
                bounds.max()
            );
        }
        if preview.selected_weight_mass != 1.0 || preview.responder_ids.is_empty() {
            bail!("adaptive step scaling requires a nonempty full-group preview");
        }
        if preview.step_scale != 1.0 {
            bail!("adaptive step scaling requires an unscaled full-group preview");
        }

        let scaled_step = merge::scale_applied_step(
            &self.params[preview.fragment_id],
            &preview.applied_step,
            scalar,
        )
        .ok_or_else(|| anyhow::anyhow!("scaled action produced a malformed or non-finite step"))?;
        let mut scaled = preview.clone();
        scaled.resulting_params = scaled_step.params;
        scaled.applied_step = scaled_step.applied_step;
        scaled.stats.outer.applied_step_norm = scaled_step.applied_step_norm;
        scaled.step_scale = scalar;
        scaled.action_fingerprint = compute_action_fingerprint(&scaled);
        Ok(scaled)
    }

    /// Return a pure copy of an exact leave-one-out `preview` whose nominal
    /// applied parameter displacement matches its full-group `reference` in
    /// L2 norm. The candidate aggregate and resulting optimizer buffer are
    /// unchanged; scaling the displacement is equivalent to changing only the
    /// effective outer learning rate for this action.
    #[allow(dead_code)] // Public action API; server wiring lands separately.
    pub fn norm_match_leave_one_out(
        &self,
        preview: &ActionPreview,
        reference: &ActionPreview,
    ) -> Result<ActionPreview> {
        self.ensure_current_preview(preview)?;
        self.ensure_current_preview(reference)?;
        if preview.fragment_id != reference.fragment_id
            || preview.base_version != reference.base_version
            || preview.base_state_epoch != reference.base_state_epoch
            || preview.base_state_fingerprint != reference.base_state_fingerprint
            || preview.target_version != reference.target_version
        {
            bail!("cannot norm-match previews from different fragment states");
        }

        if reference.selected_weight_mass != 1.0
            || reference.responder_ids.len() != preview.responder_ids.len() + 1
            || !is_ordered_subset(&preview.responder_ids, &reference.responder_ids)
        {
            bail!(
                "norm matching requires an exact leave-one-out action and its full-group reference"
            );
        }

        let source_norm = flat_l2_norm(&preview.applied_step);
        let target_norm = flat_l2_norm(&reference.applied_step);
        if !source_norm.is_finite() || source_norm < 0.0 {
            bail!("preview has invalid applied-step norm {source_norm}");
        }
        if !target_norm.is_finite() || target_norm < 0.0 {
            bail!("reference has invalid applied-step norm {target_norm}");
        }
        if source_norm <= 1e-12 {
            if target_norm <= 1e-12 {
                return Ok(preview.clone());
            }
            bail!("cannot norm-match an effectively zero source step to target norm {target_norm}");
        }
        let scale = target_norm / source_norm;
        if !scale.is_finite() {
            bail!("norm-match scale is not finite");
        }

        let scaled_step = merge::scale_applied_step(
            &self.params[preview.fragment_id],
            &preview.applied_step,
            scale,
        )
        .ok_or_else(|| anyhow::anyhow!("norm-matched action produced a non-finite step"))?;
        let mut matched = preview.clone();
        matched.resulting_params = scaled_step.params;
        matched.applied_step = scaled_step.applied_step;
        matched.stats.outer.applied_step_norm = scaled_step.applied_step_norm;
        matched.step_scale *= scale;
        if !matched.step_scale.is_finite() {
            bail!("norm-matched action has a non-finite cumulative step scale");
        }
        matched.action_fingerprint = compute_action_fingerprint(&matched);
        Ok(matched)
    }

    /// Commit the exact values held by a preview and advance its fragment to
    /// the target version bound when the preview was created. The visible
    /// fragment version, state epoch, state fingerprint, and outer policy must
    /// still match the preview's base snapshot.
    #[allow(dead_code)] // Public action API; server wiring lands separately.
    pub fn commit_preview(&mut self, preview: ActionPreview) -> Result<MergeStats> {
        self.ensure_current_preview(&preview)?;
        let fid = preview.fragment_id;
        if preview.target_version <= preview.base_version {
            bail!(
                "fragment {fid}: commit version {} must be newer than base version {}",
                preview.target_version,
                preview.base_version
            );
        }
        let target_version = preview.target_version;
        let stats = self.install_preview(preview)?;
        self.versions[fid] = target_version;
        self.global_step = self.global_step.max(target_version);
        Ok(stats)
    }

    /// Merge learner copies of fragment `fid` and apply the outer step.
    /// The returned `gnorm` remains the pre-optimizer merged-delta norm.
    ///
    /// This compatibility entry point intentionally leaves version/global
    /// step advancement to its caller, exactly as before. Its math now flows
    /// through the same aggregate and preview APIs used by action policies.
    pub fn merge_and_step(
        &mut self,
        fid: usize,
        learners: &[&[f32]],
        weights: &[f64],
    ) -> Result<MergeStats> {
        if learners.len() != weights.len() {
            bail!(
                "fragment {fid}: got {} learner values but {} weights",
                learners.len(),
                weights.len()
            );
        }
        let candidates: Vec<MergeCandidate<'_>> = learners
            .iter()
            .zip(weights)
            .enumerate()
            .map(|(responder_id, (values, &weight))| MergeCandidate {
                responder_id: responder_id as u32,
                values,
                weight,
            })
            .collect();
        let aggregate = self.build_full_aggregate(fid, &candidates)?;
        self.apply_aggregate_step(aggregate)
    }

    /// Apply an already-built production aggregate through the same legacy
    /// outer-step path as `merge_and_step`. Rho telemetry uses this entry
    /// point after sketching the aggregate in place, avoiding a second merge
    /// or a full-size pseudo-gradient clone. Version/global-step advancement
    /// remains the server caller's responsibility, as in `merge_and_step`.
    pub(crate) fn apply_aggregate_step(
        &mut self,
        aggregate: AggregateDelta,
    ) -> Result<MergeStats> {
        let preview = self.preview_aggregate_inner(&aggregate, aggregate.base_version)?;
        self.install_preview(preview)
    }

    fn validate_candidate_group(
        &self,
        fid: usize,
        candidates: &[MergeCandidate<'_>],
    ) -> Result<f64> {
        if fid >= self.layout.fragments.len() {
            bail!("fragment id {fid} out of range");
        }
        if !self.initialized[fid] {
            bail!("fragment {fid}: global parameters are not initialized");
        }
        if candidates.is_empty() {
            bail!("fragment {fid}: candidate group is empty");
        }
        let numel = self.layout.fragments[fid].numel();
        let mut previous_id = None;
        let mut total_weight = 0.0f64;
        for candidate in candidates {
            if previous_id.is_some_and(|id| candidate.responder_id <= id) {
                bail!("fragment {fid}: candidate responder IDs must be strictly increasing");
            }
            previous_id = Some(candidate.responder_id);
            if candidate.values.len() != numel {
                bail!(
                    "push for fragment {fid} from responder {} has {} values, expected {numel}",
                    candidate.responder_id,
                    candidate.values.len()
                );
            }
            if !candidate.weight.is_finite() || candidate.weight <= 0.0 {
                bail!(
                    "fragment {fid}: responder {} has invalid weight {}",
                    candidate.responder_id,
                    candidate.weight
                );
            }
            if candidate.values.iter().any(|value| !value.is_finite()) {
                bail!(
                    "fragment {fid}: responder {} has non-finite values",
                    candidate.responder_id
                );
            }
            total_weight += candidate.weight;
            if !total_weight.is_finite() {
                bail!("fragment {fid}: total candidate weight is not finite");
            }
        }
        if total_weight <= 0.0 {
            bail!("fragment {fid}: total candidate weight must be positive");
        }
        Ok(total_weight)
    }

    fn build_aggregate_from_subset(
        &self,
        fid: usize,
        candidates: &[MergeCandidate<'_>],
        full_weight: f64,
    ) -> Result<AggregateDelta> {
        let frag = &self.layout.fragments[fid];
        let anchor = &self.params[fid];
        let selected_weight: f64 = candidates.iter().map(|candidate| candidate.weight).sum();
        if !selected_weight.is_finite() || selected_weight <= 0.0 {
            bail!("fragment {fid}: selected candidate weight must be positive and finite");
        }

        // HeLoCo correction is candidate-local and per tensor, exactly as in
        // the original production merge path.
        let corrected: Vec<Vec<f32>>;
        let learners: Vec<&[f32]> = if let Some(h) = self.delta_correction {
            let momentum = &self.momentum[fid];
            corrected = candidates
                .iter()
                .map(|candidate| {
                    let mut values = candidate.values.to_vec();
                    let mut offset = 0usize;
                    for &tensor_numel in &frag.tensor_numels {
                        let tensor_numel = tensor_numel as usize;
                        let mut delta: Vec<f32> = anchor[offset..offset + tensor_numel]
                            .iter()
                            .zip(&values[offset..offset + tensor_numel])
                            .map(|(a, value)| a - value)
                            .collect();
                        merge::heloco_correct(
                            &mut delta,
                            &momentum[offset..offset + tensor_numel],
                            &h,
                        );
                        for (index, value) in delta.iter().enumerate() {
                            values[offset + index] = anchor[offset + index] - value;
                        }
                        offset += tensor_numel;
                    }
                    values
                })
                .collect();
            corrected.iter().map(Vec::as_slice).collect()
        } else {
            candidates
                .iter()
                .map(|candidate| candidate.values)
                .collect()
        };
        let weights: Vec<f64> = candidates
            .iter()
            .map(|candidate| candidate.weight)
            .collect();
        let mut delta = vec![0.0f32; frag.numel()];
        // Worker-SNR consumes the per-worker deltas and every tensor block at
        // once (its confidence shrink is per block, its norm-match is global),
        // so it runs on the whole fragment instead of the per-tensor loop.
        if frag.merge_mode == MERGE_WORKER_SNR {
            let tensor_numels: Vec<usize> =
                frag.tensor_numels.iter().map(|&n| n as usize).collect();
            merge::merge_worker_snr(anchor, &learners, &weights, &tensor_numels, &mut delta);
            let gnorm = delta
                .iter()
                .map(|value| (*value as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            if !gnorm.is_finite() || delta.iter().any(|value| !value.is_finite()) {
                bail!("fragment {fid}: aggregate delta is not finite");
            }
            let disagreement_energy =
                self.disagreement_energy_for(anchor, &learners, fid, &delta);
            return Ok(AggregateDelta {
                fragment_id: fid,
                base_version: self.versions[fid],
                base_state_epoch: self.state_epochs[fid],
                base_state_fingerprint: self.fragment_state_fingerprint(fid)?,
                responder_ids: candidates
                    .iter()
                    .map(|candidate| candidate.responder_id)
                    .collect(),
                selected_weight,
                selected_weight_mass: selected_weight / full_weight,
                delta,
                gnorm,
                disagreement_energy,
            });
        }
        let mut offset = 0usize;
        for (tensor_index, &tensor_numel) in frag.tensor_numels.iter().enumerate() {
            let tensor_numel = tensor_numel as usize;
            let tensor_learners: Vec<&[f32]> = learners
                .iter()
                .map(|values| &values[offset..offset + tensor_numel])
                .collect();
            let out = &mut delta[offset..offset + tensor_numel];
            match frag.merge_mode {
                MERGE_AVG => merge::merge_avg(
                    &anchor[offset..offset + tensor_numel],
                    &tensor_learners,
                    &weights,
                    out,
                ),
                MERGE_RDA => merge::merge_rda(
                    &anchor[offset..offset + tensor_numel],
                    &tensor_learners,
                    &weights,
                    out,
                ),
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
                    merge::merge_iso(
                        &anchor[offset..offset + tensor_numel],
                        &tensor_learners,
                        &weights,
                        rows as usize,
                        cols as usize,
                        out,
                    );
                }
                mode => bail!("fragment {fid}: unsupported merge mode {mode}"),
            }
            offset += tensor_numel;
        }
        let gnorm = delta
            .iter()
            .map(|value| (*value as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        if !gnorm.is_finite() || delta.iter().any(|value| !value.is_finite()) {
            bail!("fragment {fid}: aggregate delta is not finite");
        }
        let disagreement_energy = self.disagreement_energy_for(anchor, &learners, fid, &delta);
        Ok(AggregateDelta {
            fragment_id: fid,
            base_version: self.versions[fid],
            base_state_epoch: self.state_epochs[fid],
            base_state_fingerprint: self.fragment_state_fingerprint(fid)?,
            responder_ids: candidates
                .iter()
                .map(|candidate| candidate.responder_id)
                .collect(),
            selected_weight,
            selected_weight_mass: selected_weight / full_weight,
            delta,
            gnorm,
            disagreement_energy,
        })
    }

    /// Worker-disagreement transverse curvature-energy proxy for the aggregate,
    /// computed here where the per-worker deltas exist. Only the
    /// `capped-nesterov-wsub` outer step consumes it; every other optimizer
    /// gets `0.0` and skips the O(M*d) work. The momentum buffer is the
    /// pre-step `self.momentum[fid]` — exactly the `b_{t-1}` the outer step
    /// reads — so the threaded scalar matches the step's own geometry, keeping
    /// the preview bit-exact.
    fn disagreement_energy_for(
        &self,
        anchor: &[f32],
        learners: &[&[f32]],
        fid: usize,
        delta: &[f32],
    ) -> f64 {
        if self.outer_optimizer == merge::OuterOptimizer::CappedNesterovWsub {
            merge::disagreement_transverse_energy(anchor, learners, &self.momentum[fid], delta)
        } else {
            0.0
        }
    }

    fn ensure_current_preview(&self, preview: &ActionPreview) -> Result<()> {
        if preview.fragment_id >= self.params.len() {
            bail!("malformed action preview: fragment id out of range");
        }
        if preview.action_fingerprint != compute_action_fingerprint(preview) {
            bail!("malformed action preview: action fingerprint mismatch");
        }
        self.ensure_current_base(
            preview.fragment_id,
            preview.base_version,
            preview.base_state_epoch,
            preview.base_state_fingerprint,
        )?;
        let fid = preview.fragment_id;
        let numel = self.params[fid].len();
        if preview.responder_ids.is_empty()
            || preview
                .responder_ids
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || !preview.selected_weight.is_finite()
            || preview.selected_weight <= 0.0
            || !preview.selected_weight_mass.is_finite()
            || preview.selected_weight_mass <= 0.0
            || preview.selected_weight_mass > 1.0
            || !preview.step_scale.is_finite()
            || preview.step_scale < 0.0
            || !preview.unscaled_applied_step_norm.is_finite()
            || preview.unscaled_applied_step_norm < 0.0
            || preview.resulting_params.len() != numel
            || preview.resulting_optimizer_buffer.len() != numel
            || preview.applied_step.len() != numel
            || preview
                .resulting_params
                .iter()
                .any(|value| !value.is_finite())
            || preview
                .resulting_optimizer_buffer
                .iter()
                .any(|value| !value.is_finite())
            || preview.applied_step.iter().any(|value| !value.is_finite())
            || !preview.resulting_rho_ema.is_finite()
            || !preview.resulting_capped_mu.is_finite()
            || !preview.resulting_capped_gain.is_finite()
            || !preview.resulting_cheb_phase.is_finite()
            || preview.resulting_block_v.len() != self.layout.fragments[fid].tensor_numels.len()
            || preview
                .resulting_block_v
                .iter()
                .any(|value| !value.is_finite())
            || preview.resulting_curv_prev_delta.len() != numel
            || preview.resulting_curv_prev_dtheta.len() != numel
            || preview
                .resulting_curv_prev_delta
                .iter()
                .any(|value| !value.is_finite())
            || preview
                .resulting_curv_prev_dtheta
                .iter()
                .any(|value| !value.is_finite())
            || !preview.stats.gnorm.is_finite()
            || preview.stats.gnorm < 0.0
            || !preview.stats.outer.applied_step_norm.is_finite()
            || preview.stats.outer.applied_step_norm < 0.0
            || preview
                .stats
                .outer
                .direction_delta_cosine
                .is_some_and(|value| !value.is_finite())
            || preview
                .stats
                .outer
                .history_current_norm_ratio
                .is_some_and(|value| !value.is_finite())
        {
            bail!("malformed action preview: invalid metadata or tensor shape");
        }
        for ((&base, &step), &result) in self.params[fid]
            .iter()
            .zip(&preview.applied_step)
            .zip(&preview.resulting_params)
        {
            if (base - step).to_bits() != result.to_bits() {
                bail!("malformed action preview: parameters do not match the sealed step");
            }
        }
        let step_norm = flat_l2_norm(&preview.applied_step);
        let tolerance = 1e-6 * preview.stats.outer.applied_step_norm.abs().max(1.0);
        if (step_norm - preview.stats.outer.applied_step_norm).abs() > tolerance {
            bail!("malformed action preview: applied-step statistics do not match the step");
        }
        Ok(())
    }

    fn ensure_current_base(
        &self,
        fid: usize,
        base_version: u64,
        base_state_epoch: u64,
        base_state_fingerprint: [u64; 2],
    ) -> Result<()> {
        if fid >= self.params.len() {
            bail!("fragment id {fid} out of range");
        }
        if self.versions[fid] != base_version {
            bail!(
                "stale fragment {fid} preview: base version {base_version}, current version {}",
                self.versions[fid]
            );
        }
        if self.state_epochs[fid] != base_state_epoch {
            bail!(
                "stale fragment {fid} preview: base state epoch {base_state_epoch}, current epoch {}",
                self.state_epochs[fid]
            );
        }
        if self.fragment_state_fingerprint(fid)? != base_state_fingerprint {
            bail!("stale fragment {fid} preview: state or outer policy changed");
        }
        Ok(())
    }

    fn fragment_state_fingerprint(&self, fid: usize) -> Result<[u64; 2]> {
        if fid >= self.params.len() || fid >= self.layout.fragments.len() {
            bail!("fragment id {fid} out of range");
        }
        let mut hash = [0xcbf2_9ce4_8422_2325u64, 0x9e37_79b9_7f4a_7c15u64];
        let mut mix = |word: u64| {
            hash[0] ^= word;
            hash[0] = hash[0].wrapping_mul(0x0000_0100_0000_01b3);
            hash[1] ^= word.wrapping_add(0x517c_c1b7_2722_0a95);
            hash[1] = hash[1].rotate_left(27).wrapping_mul(0x94d0_49bb_1331_11eb);
        };
        let fragment = &self.layout.fragments[fid];
        mix(fid as u64);
        mix(fragment.merge_mode as u64);
        mix(fragment.tensor_numels.len() as u64);
        for &numel in &fragment.tensor_numels {
            mix(numel);
        }
        // Iso shapes change the merged values, so they are part of the
        // fragment identity; avg/RDA fragments have no shapes and keep
        // their fingerprints bit-identical to the pre-iso format.
        if let Some(shapes) = &fragment.tensor_shapes {
            for &(rows, cols) in shapes {
                mix(rows);
                mix(cols);
            }
        }
        mix(self.initialized[fid] as u64);
        mix(self.params[fid].len() as u64);
        for value in &self.params[fid] {
            mix(value.to_bits() as u64);
        }
        mix(self.momentum[fid].len() as u64);
        for value in &self.momentum[fid] {
            mix(value.to_bits() as u64);
        }
        mix(self.rho_ema[fid].to_bits() as u64);
        mix(self.capped_mu[fid].to_bits() as u64);
        mix(self.capped_gain[fid].to_bits() as u64);
        mix(self.cheb_phase[fid].to_bits() as u64);
        mix(self.block_v[fid].len() as u64);
        for value in &self.block_v[fid] {
            mix(value.to_bits() as u64);
        }
        mix(self.curv_prev_delta[fid].len() as u64);
        for value in &self.curv_prev_delta[fid] {
            mix(value.to_bits() as u64);
        }
        mix(self.curv_prev_dtheta[fid].len() as u64);
        for value in &self.curv_prev_dtheta[fid] {
            mix(value.to_bits() as u64);
        }
        match &self.outer_lr_by_fragment {
            None => {
                mix(0);
                mix(self.outer_lr.to_bits() as u64);
            }
            Some(rates) => {
                mix(1);
                mix(rates.len() as u64);
                mix(rates
                    .get(fid)
                    .map(|rate| rate.to_bits() as u64)
                    .unwrap_or(u64::MAX));
            }
        }
        mix(self.outer_momentum.to_bits() as u64);
        mix(self.outer_restart_cos_threshold.to_bits() as u64);
        mix(match self.outer_optimizer {
            merge::OuterOptimizer::Nesterov => 0,
            merge::OuterOptimizer::NormalizedEma => 1,
            merge::OuterOptimizer::RestartedEma => 2,
            merge::OuterOptimizer::RhoAdaptive => 3,
            merge::OuterOptimizer::CappedNesterov => 4,
            merge::OuterOptimizer::CappedNesterovGc => 5,
            merge::OuterOptimizer::CappedNesterovR => 6,
            merge::OuterOptimizer::BlockRms => 7,
            merge::OuterOptimizer::BlockYogi => 8,
            merge::OuterOptimizer::CappedNesterovCurv => 9,
            merge::OuterOptimizer::CappedNesterovWsub => 10,
            merge::OuterOptimizer::ChebSgd => 11,
        });
        match self.delta_correction {
            None => mix(0),
            Some(config) => {
                mix(1);
                for value in [
                    config.c_ok,
                    config.k_s,
                    config.k_d,
                    config.beta_max,
                    config.kappa,
                    config.eps,
                ] {
                    mix(value.to_bits());
                }
            }
        }
        Ok(hash)
    }

    fn install_preview(&mut self, preview: ActionPreview) -> Result<MergeStats> {
        self.ensure_current_preview(&preview)?;
        let fid = preview.fragment_id;
        let ActionPreview {
            resulting_params,
            resulting_optimizer_buffer,
            resulting_rho_ema,
            resulting_capped_mu,
            resulting_capped_gain,
            resulting_block_v,
            resulting_curv_prev_delta,
            resulting_curv_prev_dtheta,
            resulting_cheb_phase,
            stats,
            ..
        } = preview;
        let next_epoch = self.next_state_epoch(fid)?;
        // EXP2.46: snapshot the outgoing global under its current version before
        // it is overwritten, so a later push tagged at that version can still be
        // version-matched. No-op unless anchor retention is enabled.
        self.retain_anchor_snapshot(fid);
        self.params[fid] = resulting_params;
        self.momentum[fid] = resulting_optimizer_buffer;
        self.rho_ema[fid] = resulting_rho_ema;
        self.capped_mu[fid] = resulting_capped_mu;
        self.capped_gain[fid] = resulting_capped_gain;
        self.block_v[fid] = resulting_block_v;
        self.curv_prev_delta[fid] = resulting_curv_prev_delta;
        self.curv_prev_dtheta[fid] = resulting_curv_prev_dtheta;
        self.cheb_phase[fid] = resulting_cheb_phase;
        self.state_epochs[fid] = next_epoch;
        Ok(stats)
    }

    fn next_state_epoch(&self, fid: usize) -> Result<u64> {
        let Some(next) = self.state_epochs[fid].checked_add(1) else {
            bail!("fragment {fid}: state epoch overflow");
        };
        Ok(next)
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
            bail!(
                "checkpoint has {np} fragments, layout has {}",
                self.params.len()
            );
        }
        for p in 0..np {
            let next_epoch = self.next_state_epoch(p)?;
            self.versions[p] = r.u64()?;
            let numel = r.u64()? as usize;
            if numel != self.params[p].len() {
                bail!(
                    "checkpoint fragment {p} numel {numel} != layout {}",
                    self.params[p].len()
                );
            }
            for slot in [&mut self.params[p], &mut self.momentum[p]] {
                for v in slot.iter_mut() {
                    *v = f32::from_le_bytes(r.take(4)?.try_into()?);
                }
            }
            self.cttn_shadow_momentum[p].fill(0.0);
            // The rho-adaptive EMA and the capped-Nesterov-family effective
            // momentum and gc gain are not part of the checkpoint format;
            // restore to their references so resumed runs are deterministic.
            self.rho_ema[p] = merge::RHO_ADAPTIVE_INITIAL_RHO_EMA;
            self.capped_mu[p] = merge::CAPPED_NESTEROV_INITIAL_MU;
            self.capped_gain[p] = merge::CAPPED_NESTEROV_GC_INITIAL_GAIN;
            // The Chebyshev-SGD cycle phase is likewise not checkpointed; reset
            // to 0 so a resumed run restarts the cycle from its smallest step.
            self.cheb_phase[p] = 0.0;
            // The curvature-aware controller history is not checkpointed
            // either; reset to zero so lambda_hat is inactive for one commit.
            for slot in [
                &mut self.curv_prev_delta[p],
                &mut self.curv_prev_dtheta[p],
            ] {
                for v in slot.iter_mut() {
                    *v = 0.0;
                }
            }
            self.initialized[p] = true;
            self.state_epochs[p] = next_epoch;
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

    fn vector_norm(values: &[f32]) -> f64 {
        values
            .iter()
            .map(|value| (*value as f64).powi(2))
            .sum::<f64>()
            .sqrt()
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

    // ---- EXP2.46 version-matched anchoring / anchor history ---------------

    /// One f32 fragment of 4 params, lr 1 mu 0 (plain SGD = weight averaging),
    /// used to exercise the anchor-history retention independent of the outer
    /// optimizer.
    fn anchor_state(instrument: bool) -> GlobalState {
        let layout = Layout {
            fragments: vec![FragmentInfo {
                merge_mode: MERGE_AVG,
                tensor_numels: vec![4],
                tensor_shapes: None,
            }],
        };
        let mut st = GlobalState::new(layout, None, 1.0, 0.0, crate::protocol::DTYPE_F32);
        st.anchor_drift_instrument = instrument;
        st
    }

    #[test]
    fn anchor_history_retains_prior_global_when_enabled() {
        let mut st = anchor_state(true);
        st.init_fragment(0, vec![1.0; 4]).unwrap();
        // Merge to a fresh global; install snapshots (version 0, [1;4]) first.
        st.merge_and_step(0, &[&vec![0.0f32; 4]], &[1.0]).unwrap();
        st.versions[0] = 5; // the server bumps the version after merge_and_step
        // Current version resolves to the live params.
        assert_eq!(st.anchor_at(0, 5).unwrap(), &[0.0f32; 4]);
        // The learner's base version resolves to the retained prior global.
        assert_eq!(st.anchor_at(0, 0).unwrap(), &[1.0f32; 4]);
        // An unknown intermediate version is unresolved.
        assert!(st.anchor_at(0, 3).is_none());
    }

    #[test]
    fn anchor_history_stays_empty_when_disabled() {
        let mut st = anchor_state(false);
        assert!(!st.anchor_retention_enabled());
        st.init_fragment(0, vec![1.0; 4]).unwrap();
        st.merge_and_step(0, &[&vec![0.0f32; 4]], &[1.0]).unwrap();
        st.versions[0] = 5;
        assert!(st.anchor_history[0].is_empty());
        // Current anchor still works; prior versions are simply unresolved.
        assert_eq!(st.anchor_at(0, 5).unwrap(), &[0.0f32; 4]);
        assert!(st.anchor_at(0, 0).is_none());
    }

    #[test]
    fn anchor_history_is_bounded_and_evicts_oldest() {
        let mut st = anchor_state(true);
        st.init_fragment(0, vec![0.0; 4]).unwrap();
        let total = ANCHOR_HISTORY_DEPTH as u64 + 3;
        for v in 1..=total {
            st.merge_and_step(0, &[&vec![0.0f32; 4]], &[1.0]).unwrap();
            st.versions[0] = v;
        }
        // Recorded one snapshot per merge (versions 0..total-1), capped.
        assert_eq!(st.anchor_history[0].len(), ANCHOR_HISTORY_DEPTH);
        let oldest_kept = total - ANCHOR_HISTORY_DEPTH as u64;
        assert!(st.anchor_at(0, oldest_kept).is_some());
        assert!(st.anchor_at(0, oldest_kept - 1).is_none());
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
    fn layout_decode_reads_iso_shapes_and_validates_them() {
        // fragment 0: rda, one tensor of 4 (legacy wire format, no shapes);
        // fragment 1: iso, tensors of 4 and 6 with shapes (2,2) and (2,3).
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

        // rows*cols must equal the tensor numel.
        let mut bad = Vec::new();
        bad.push(MERGE_ISO);
        bad.extend_from_slice(&1u32.to_le_bytes());
        bad.extend_from_slice(&4u64.to_le_bytes());
        bad.extend_from_slice(&2u64.to_le_bytes());
        bad.extend_from_slice(&3u64.to_le_bytes());
        let mut r = crate::protocol::Reader(&bad);
        assert!(Layout::decode(&mut r, 1).is_err());

        // Truncated shape block is rejected.
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
        // One 2x2 tensor merged in iso mode at lr 1, mu 0: the learner delta
        // diag(3, 1) flattens to diag(2, 2) (sigma_bar = 2) and is applied
        // as a plain SGD step.
        let layout = Layout {
            fragments: vec![FragmentInfo {
                merge_mode: MERGE_ISO,
                tensor_numels: vec![4],
                tensor_shapes: Some(vec![(2, 2)]),
            }],
        };
        let mut st = GlobalState::new(layout, None, 1.0, 0.0, crate::protocol::DTYPE_F32);
        st.init_fragment(0, vec![0.0; 4]).unwrap();
        let learner = [-3.0f32, 0.0, 0.0, -1.0];
        let stats = st.merge_and_step(0, &[&learner], &[1.0]).unwrap();
        for (value, expected) in st.params[0].iter().zip([-2.0f32, 0.0, 0.0, -2.0]) {
            assert!((value - expected).abs() < 1e-6, "params {:?}", st.params[0]);
        }
        // gnorm reports the flattened delta: |diag(2, 2)| = 2*sqrt(2).
        assert!((stats.gnorm - 2.0 * 2.0f64.sqrt()).abs() < 1e-6);
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
    fn delta_norm_ref_rescales_merged_delta_before_outer_step() {
        // Raw merged delta [1,1,1,1] has norm 2; ref 1.0 halves it. lr 1,
        // mu 0: params move by exactly the renormalized delta, gnorm keeps
        // the PRE-rescale norm, and the momentum buffer receives the
        // rescaled delta.
        let mut st = GlobalState::new(layout2(), None, 1.0, 0.0, crate::protocol::DTYPE_F32);
        st.delta_norm_ref = 1.0;
        st.init_fragment(0, vec![1.0; 4]).unwrap();
        st.init_fragment(1, vec![1.0; 4]).unwrap();
        let learner = vec![0.0f32; 4];
        let stats = st.merge_and_step(0, &[&learner], &[1.0]).unwrap();
        assert!((stats.gnorm - 2.0).abs() < 1e-12, "gnorm is pre-rescale");
        assert!((stats.outer.applied_step_norm - 1.0).abs() < 1e-7);
        assert_eq!(st.params[0], vec![0.5; 4]);
        assert_eq!(st.momentum[0], vec![0.5; 4]);
    }

    #[test]
    fn delta_norm_ref_zero_is_identical_to_default_and_zero_delta_is_untouched() {
        let mut plain = GlobalState::new(layout2(), None, 0.7, 0.9, crate::protocol::DTYPE_F32);
        let mut explicit = GlobalState::new(layout2(), None, 0.7, 0.9, crate::protocol::DTYPE_F32);
        explicit.delta_norm_ref = 0.0;
        for st in [&mut plain, &mut explicit] {
            st.init_fragment(0, vec![1.0; 4]).unwrap();
            st.init_fragment(1, vec![1.0; 4]).unwrap();
            st.merge_and_step(0, &[&[0.25f32, -0.5, 0.75, -1.0][..]], &[1.0])
                .unwrap();
            st.merge_and_step(0, &[&[0.5f32, 0.5, -0.25, 0.125][..]], &[2.0])
                .unwrap();
        }
        assert_eq!(plain.params[0], explicit.params[0]);
        assert_eq!(plain.momentum[0], explicit.momentum[0]);

        // Zero merged delta with a positive ref: no rescale, no NaN, no step.
        let mut st = GlobalState::new(layout2(), None, 0.7, 0.0, crate::protocol::DTYPE_F32);
        st.delta_norm_ref = 3.0;
        st.init_fragment(0, vec![1.0; 4]).unwrap();
        st.init_fragment(1, vec![1.0; 4]).unwrap();
        let stats = st.merge_and_step(0, &[&[1.0f32; 4][..]], &[1.0]).unwrap();
        assert_eq!(stats.gnorm, 0.0);
        assert_eq!(stats.outer.applied_step_norm, 0.0);
        assert_eq!(st.params[0], vec![1.0; 4]);
    }

    #[test]
    fn delta_norm_ref_momentum_buffer_compounds_on_rescaled_deltas() {
        // Two commits under Nesterov mu 0.9, lr 1, both renormalized to norm
        // 2 (their raw norms are 2 and 4, so scales are 1 and 0.5). Buffer
        // and params must match the hand-computed recursion on the RESCALED
        // deltas d1 = [1;4], d2 = [1;4]:
        //   b1 = d1, step1 = d1 + 0.9 b1 = 1.9
        //   b2 = 0.9 b1 + d2 = 1.9, step2 = d2 + 0.9 b2 = 2.71
        let mut st = GlobalState::new(layout2(), None, 1.0, 0.9, crate::protocol::DTYPE_F32);
        st.delta_norm_ref = 2.0;
        st.init_fragment(0, vec![10.0; 4]).unwrap();
        st.init_fragment(1, vec![10.0; 4]).unwrap();
        st.merge_and_step(0, &[&[9.0f32; 4][..]], &[1.0]).unwrap();
        for value in &st.params[0] {
            assert!((*value - (10.0 - 1.9)).abs() < 1e-5, "{:?}", st.params[0]);
        }
        assert_eq!(st.momentum[0], vec![1.0; 4]);
        // Learner sits 2.0 below the new anchor: raw delta [2;4], norm 4.
        let learner2: Vec<f32> = st.params[0].iter().map(|value| value - 2.0).collect();
        st.merge_and_step(0, &[learner2.as_slice()], &[1.0]).unwrap();
        for value in &st.params[0] {
            assert!(
                (*value - (10.0 - 1.9 - 2.71)).abs() < 1e-4,
                "{:?}",
                st.params[0]
            );
        }
        for value in &st.momentum[0] {
            assert!((*value - 1.9).abs() < 1e-6);
        }
    }

    #[test]
    fn delta_norm_ref_preview_is_bit_exact_and_commits() {
        // The rescaled delta feeds both the applied step and its
        // materialization from the SAME slice, so a norm-matched preview is
        // bit-identical to re-materializing from the updated buffer with the
        // identically-recomputed scale (deterministic from the raw delta).
        let mut st = GlobalState::new(layout2(), None, 0.7, 0.9, crate::protocol::DTYPE_F32);
        st.delta_norm_ref = 1.5;
        st.init_fragment(0, vec![1.0; 4]).unwrap();
        st.init_fragment(1, vec![1.0; 4]).unwrap();
        let learner = [0.25f32, -0.5, 0.75, 2.0];
        let candidates = [MergeCandidate::new(0, &learner, 1.0)];
        let aggregate = st.build_full_aggregate(0, &candidates).unwrap();
        let preview = st.preview_aggregate(&aggregate, 1).unwrap();

        let raw_norm = vector_norm(aggregate.delta());
        let scale = (st.delta_norm_ref as f64 / raw_norm) as f32;
        let scaled: Vec<f32> = aggregate.delta().iter().map(|value| scale * *value).collect();
        assert!((vector_norm(&scaled) - 1.5).abs() < 1e-6);
        let rematerialized = merge::materialize_applied_step(
            merge::OuterOptimizer::Nesterov,
            &preview.resulting_optimizer_buffer,
            &scaled,
            0.7,
            0.9,
            1.0,
        );
        assert_eq!(preview.applied_step, rematerialized, "bit-exact preview");
        assert_eq!(preview.resulting_optimizer_buffer, scaled, "b1 = d1");

        let params_before = st.params[0].clone();
        let stats = st.commit_preview(preview.clone()).unwrap();
        assert!((stats.gnorm - raw_norm).abs() < 1e-12);
        for ((committed, before), step) in st.params[0]
            .iter()
            .zip(&params_before)
            .zip(&preview.applied_step)
        {
            assert_eq!(*committed, before - step);
        }
    }

    #[test]
    fn cttn_inputs_and_commit_share_post_renorm_direction_exactly() {
        let mut st = GlobalState::new(layout2(), None, 0.25, 0.0, crate::protocol::DTYPE_F32);
        st.delta_norm_ref = 2.0;
        st.init_fragment(0, vec![1.0; 4]).unwrap();
        st.init_fragment(1, vec![1.0; 4]).unwrap();
        st.momentum[0] = vec![0.5, -0.5, 1.0, -1.0];
        let learner = [0.0f32, 2.0, -1.0, 3.0];
        let aggregate = st
            .build_full_aggregate(0, &[MergeCandidate::new(0, &learner, 1.0)])
            .unwrap();
        let inputs = st.cttn_inputs(&aggregate, 0.9).unwrap();
        assert!((vector_norm(&inputs.g) - 2.0).abs() < 1e-6);
        assert_eq!(inputs.b, vec![0.5, -0.5, 1.0, -1.0]);
        assert_eq!(inputs.outer_lr, 0.25);
        assert_eq!(inputs.mu, 0.9);

        let d = inputs.g.clone();
        let b_new = vec![1.5, 2.5, 3.5, 4.5];
        let before = st.params[0].clone();
        let stats = st
            .commit_cttn_step(&aggregate, 1, &d, &b_new, inputs.outer_lr)
            .unwrap();
        for index in 0..d.len() {
            let applied_step = 0.25f32 * d[index];
            assert_eq!(
                st.params[0][index].to_bits(),
                (before[index] - applied_step).to_bits()
            );
        }
        assert_eq!(st.momentum[0], b_new);
        assert_eq!(st.versions[0], 1);
        assert!((stats.outer.applied_step_norm - vector_norm(&d) * 0.25).abs() < 1e-7);
    }

    #[test]
    fn cttn_shadow_commits_exact_sgd_and_keeps_independent_mu09_buffer() {
        let mut st = GlobalState::new(layout2(), None, 0.28, 0.0, crate::protocol::DTYPE_F32);
        st.init_fragment(0, vec![1.0; 4]).unwrap();
        st.init_fragment(1, vec![1.0; 4]).unwrap();
        let learner = [0.0f32, 2.0, -1.0, 3.0];
        let aggregate = st
            .build_full_aggregate(0, &[MergeCandidate::new(0, &learner, 1.0)])
            .unwrap();
        let inputs = st.cttn_shadow_inputs(&aggregate, 0.9).unwrap();
        let before = st.params[0].clone();
        st.commit_cttn_shadow_sgd(&aggregate, 1, 0.9).unwrap();
        for index in 0..inputs.g.len() {
            assert_eq!(
                st.params[0][index].to_bits(),
                (before[index] - 0.28 * inputs.g[index]).to_bits(),
                "shadow d must equal g exactly"
            );
        }
        assert_eq!(st.momentum[0], inputs.g);
        assert_eq!(st.cttn_shadow_momentum[0], inputs.g);

        let learner2 = [0.5f32, 1.5, -0.5, 2.5];
        let aggregate2 = st
            .build_full_aggregate(0, &[MergeCandidate::new(0, &learner2, 1.0)])
            .unwrap();
        let inputs2 = st.cttn_shadow_inputs(&aggregate2, 0.9).unwrap();
        assert_eq!(inputs2.b, inputs.g);
        st.commit_cttn_shadow_sgd(&aggregate2, 2, 0.9).unwrap();
        let expected_shadow: Vec<f32> = inputs2
            .b
            .iter()
            .zip(&inputs2.g)
            .map(|(b, g)| 0.9 * *b + *g)
            .collect();
        assert_eq!(st.cttn_shadow_momentum[0], expected_shadow);
        assert_eq!(st.momentum[0], inputs2.g);
    }

    #[test]
    fn selected_aggregate_is_sorted_exact_and_reports_weight_mass() {
        let mut st = GlobalState::new(layout2(), None, 0.7, 0.0, crate::protocol::DTYPE_F32);
        st.init_fragment(1, vec![1.0; 4]).unwrap();
        let learner_2 = [0.0f32, 1.0, 1.0, 0.0];
        let learner_5 = [1.0f32, 0.0, 0.0, 1.0];
        let learner_9 = [2.0f32, 1.0, 1.0, 2.0];
        let candidates = [
            MergeCandidate::new(2, &learner_2, 1.0),
            MergeCandidate::new(5, &learner_5, 3.0),
            MergeCandidate::new(9, &learner_9, 2.0),
        ];

        let aggregate = st
            .build_selected_aggregate(1, &candidates, &[2, 5])
            .unwrap();
        assert_eq!(
            aggregate,
            st.build_selected_aggregate(1, &candidates, &[2, 5])
                .unwrap()
        );
        let mut expected = vec![0.0f32; 4];
        merge::merge_rda(
            &[1.0, 1.0],
            &[&learner_2[..2], &learner_5[..2]],
            &[1.0, 3.0],
            &mut expected[..2],
        );
        merge::merge_rda(
            &[1.0, 1.0],
            &[&learner_2[2..], &learner_5[2..]],
            &[1.0, 3.0],
            &mut expected[2..],
        );

        assert_eq!(aggregate.fragment_id(), 1);
        assert_eq!(aggregate.base_version(), 0);
        assert_eq!(aggregate.base_state_epoch(), st.state_epochs[1]);
        assert_eq!(aggregate.responder_ids(), &[2, 5]);
        assert_eq!(aggregate.selected_weight(), 4.0);
        assert!((aggregate.selected_weight_mass() - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(aggregate.delta(), expected);
        assert!((aggregate.gnorm() - vector_norm(&expected)).abs() < 1e-12);

        let unsorted = [candidates[1], candidates[0], candidates[2]];
        assert!(st.build_full_aggregate(1, &unsorted).is_err());
        assert!(st
            .build_selected_aggregate(1, &candidates, &[5, 2])
            .is_err());
        assert!(st
            .build_selected_aggregate(1, &candidates, &[2, 7])
            .is_err());
    }

    #[test]
    fn preview_is_pure_and_commit_installs_exact_values() {
        let mut st = GlobalState::new(layout2(), None, 0.4, 0.5, crate::protocol::DTYPE_F32);
        st.init_fragment(0, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let learner_3 = [0.5f32, 1.5, 2.5, 3.5];
        let learner_8 = [1.5f32, 1.0, 3.5, 2.0];
        let candidates = [
            MergeCandidate::new(3, &learner_3, 2.0),
            MergeCandidate::new(8, &learner_8, 1.0),
        ];
        let params_before = st.params.clone();
        let momentum_before = st.momentum.clone();
        let versions_before = st.versions.clone();
        let global_step_before = st.global_step;
        let epochs_before = st.state_epochs.clone();

        let aggregate = st.build_full_aggregate(0, &candidates).unwrap();
        let preview = st.preview_aggregate(&aggregate, 7).unwrap();
        assert_eq!(st.params, params_before);
        assert_eq!(st.momentum, momentum_before);
        assert_eq!(st.versions, versions_before);
        assert_eq!(st.global_step, global_step_before);
        assert_eq!(st.state_epochs, epochs_before);

        assert_eq!(preview.fragment_id(), 0);
        assert_eq!(preview.base_version(), 0);
        assert_eq!(preview.base_state_epoch(), epochs_before[0]);
        assert_eq!(preview.target_version(), 7);
        assert_eq!(preview.responder_ids(), &[3, 8]);
        assert_eq!(preview.selected_weight(), 3.0);
        assert_eq!(preview.selected_weight_mass(), 1.0);
        assert_eq!(preview.norm_match_scale(), 1.0);
        let expected_params = preview.resulting_params().to_vec();
        let expected_buffer = preview.resulting_optimizer_buffer().to_vec();
        let expected_stats = preview.stats();
        let stale_copy = preview.clone();

        let committed_stats = st.commit_preview(preview).unwrap();
        assert_eq!(committed_stats, expected_stats);
        assert_eq!(st.params[0], expected_params);
        assert_eq!(st.momentum[0], expected_buffer);
        assert_eq!(st.versions[0], 7);
        assert_eq!(st.global_step, 7);
        assert_eq!(st.state_epochs[0], epochs_before[0] + 1);
        assert!(st.commit_preview(stale_copy).is_err());
    }

    #[test]
    fn preview_commit_rejects_stale_epoch_and_version_without_mutation() {
        let make_preview = |st: &GlobalState, learner: &[f32], target_version: u64| {
            let candidates = [MergeCandidate::new(1, learner, 1.0)];
            let aggregate = st.build_full_aggregate(0, &candidates).unwrap();
            st.preview_aggregate(&aggregate, target_version).unwrap()
        };

        let mut epoch_stale =
            GlobalState::new(layout2(), None, 0.5, 0.0, crate::protocol::DTYPE_F32);
        epoch_stale.init_fragment(0, vec![1.0; 4]).unwrap();
        let learner = [0.0f32; 4];
        let stale_preview = make_preview(&epoch_stale, &learner, 1);
        epoch_stale
            .merge_and_step(0, &[&[0.5f32; 4]], &[1.0])
            .unwrap();
        let params_after_newer_step = epoch_stale.params.clone();
        let momentum_after_newer_step = epoch_stale.momentum.clone();
        let error = epoch_stale.commit_preview(stale_preview).unwrap_err();
        assert!(error.to_string().contains("state epoch"));
        assert_eq!(epoch_stale.params, params_after_newer_step);
        assert_eq!(epoch_stale.momentum, momentum_after_newer_step);

        let mut version_stale =
            GlobalState::new(layout2(), None, 0.5, 0.0, crate::protocol::DTYPE_F32);
        version_stale.init_fragment(0, vec![1.0; 4]).unwrap();
        let stale_preview = make_preview(&version_stale, &learner, 1);
        version_stale.versions[0] = 3;
        let params_before = version_stale.params.clone();
        let error = version_stale.commit_preview(stale_preview).unwrap_err();
        assert!(error.to_string().contains("base version"));
        assert_eq!(version_stale.params, params_before);

        let fresh_aggregate = version_stale
            .build_full_aggregate(0, &[MergeCandidate::new(1, &learner, 1.0)])
            .unwrap();
        assert!(version_stale
            .preview_aggregate(&fresh_aggregate, 3)
            .is_err());
    }

    #[test]
    fn norm_matching_is_pure_and_preserves_candidate_optimizer_state() {
        for optimizer in [
            merge::OuterOptimizer::Nesterov,
            merge::OuterOptimizer::NormalizedEma,
            merge::OuterOptimizer::RestartedEma,
        ] {
            let mut st = GlobalState::new(layout2(), None, 0.3, 0.5, crate::protocol::DTYPE_F32);
            st.outer_optimizer = optimizer;
            st.outer_restart_cos_threshold = -0.25;
            st.init_fragment(0, vec![2.0, 1.0, -1.0, 0.5]).unwrap();
            st.merge_and_step(0, &[&[1.0f32, 0.5, -1.5, 0.0]], &[1.0])
                .unwrap();

            let base = st.params[0].clone();
            let learner_4 = [base[0] - 1.0, base[1], base[2] - 0.5, base[3]];
            let learner_7 = [base[0], base[1] - 2.0, base[2], base[3] + 0.25];
            let candidates = [
                MergeCandidate::new(4, &learner_4, 1.0),
                MergeCandidate::new(7, &learner_7, 1.0),
            ];
            let full = st.build_full_aggregate(0, &candidates).unwrap();
            let leave_one_out = st.build_selected_aggregate(0, &candidates, &[4]).unwrap();
            let target_version = st.versions[0] + 1;
            let reference = st.preview_aggregate(&full, target_version).unwrap();
            let candidate = st
                .preview_aggregate(&leave_one_out, target_version)
                .unwrap();
            let params_before = st.params.clone();
            let momentum_before = st.momentum.clone();

            let matched = st.norm_match_leave_one_out(&candidate, &reference).unwrap();
            let target = reference.stats().outer.applied_step_norm;
            assert!(
                (matched.stats().outer.applied_step_norm - target).abs() < 1e-6,
                "{optimizer}: matched {} vs target {target}",
                matched.stats().outer.applied_step_norm
            );
            assert_eq!(
                matched.resulting_optimizer_buffer(),
                candidate.resulting_optimizer_buffer()
            );
            assert_eq!(matched.applied_step().len(), candidate.applied_step().len());
            assert_eq!(matched.responder_ids(), candidate.responder_ids());
            assert_eq!(
                matched.selected_weight_mass(),
                candidate.selected_weight_mass()
            );
            assert_eq!(matched.stats().gnorm, candidate.stats().gnorm);
            assert_eq!(
                matched.stats().outer.direction_delta_cosine,
                candidate.stats().outer.direction_delta_cosine
            );
            assert_eq!(
                matched.stats().outer.history_current_norm_ratio,
                candidate.stats().outer.history_current_norm_ratio
            );
            assert_eq!(
                matched.stats().outer.restarted,
                candidate.stats().outer.restarted
            );
            assert!(matched.norm_match_scale().is_finite());
            assert_eq!(st.params, params_before);
            assert_eq!(st.momentum, momentum_before);
        }
    }

    #[test]
    fn norm_matching_rejects_zero_step_to_nonzero_target() {
        let mut st = GlobalState::new(layout2(), None, 1.0, 0.0, crate::protocol::DTYPE_F32);
        st.init_fragment(0, vec![1.0; 4]).unwrap();
        let unchanged = [1.0f32; 4];
        let moved = [0.0f32; 4];
        let candidates = [
            MergeCandidate::new(1, &unchanged, 1.0),
            MergeCandidate::new(2, &moved, 1.0),
        ];
        let zero = st
            .preview_aggregate(
                &st.build_selected_aggregate(0, &candidates, &[1]).unwrap(),
                1,
            )
            .unwrap();
        let nonzero = st
            .preview_aggregate(&st.build_full_aggregate(0, &candidates).unwrap(), 1)
            .unwrap();
        assert!(st.norm_match_leave_one_out(&zero, &nonzero).is_err());

        let positive = [0.0f32; 4];
        let negative = [2.0f32; 4];
        let cancelling = [
            MergeCandidate::new(1, &positive, 1.0),
            MergeCandidate::new(2, &negative, 1.0),
        ];
        let nonzero_source = st
            .preview_aggregate(
                &st.build_selected_aggregate(0, &cancelling, &[1]).unwrap(),
                1,
            )
            .unwrap();
        let zero_target = st
            .preview_aggregate(&st.build_full_aggregate(0, &cancelling).unwrap(), 1)
            .unwrap();
        let expected_buffer = nonzero_source.resulting_optimizer_buffer().to_vec();
        let matched_zero = st
            .norm_match_leave_one_out(&nonzero_source, &zero_target)
            .unwrap();
        assert_eq!(matched_zero.resulting_params(), st.params[0]);
        assert_eq!(matched_zero.resulting_optimizer_buffer(), expected_buffer);
        assert_eq!(matched_zero.applied_step(), &[0.0; 4]);
        assert_eq!(matched_zero.stats().outer.applied_step_norm, 0.0);
        assert_eq!(matched_zero.norm_match_scale(), 0.0);
    }

    #[test]
    fn norm_matching_uses_nominal_step_vector_on_the_f32_lattice() {
        let layout = Layout {
            fragments: vec![FragmentInfo {
                merge_mode: MERGE_AVG,
                tensor_numels: vec![1],
                tensor_shapes: None,
            }],
        };
        let mut st = GlobalState::new(layout, None, 0.125, 0.0, crate::protocol::DTYPE_F32);
        st.init_fragment(0, vec![100_000_000.0]).unwrap();
        let large = [100_000_128.0f32]; // delta -128, nominal step -16
        let compensating = [99_999_888.0f32]; // delta +112; full mean delta -8
        let candidates = [
            MergeCandidate::new(1, &large, 1.0),
            MergeCandidate::new(2, &compensating, 1.0),
        ];
        let reference = st
            .preview_aggregate(&st.build_full_aggregate(0, &candidates).unwrap(), 1)
            .unwrap();
        let leave_one_out = st
            .preview_aggregate(
                &st.build_selected_aggregate(0, &candidates, &[1]).unwrap(),
                1,
            )
            .unwrap();
        assert_eq!(reference.applied_step(), &[-1.0]);
        assert_eq!(leave_one_out.applied_step(), &[-16.0]);
        assert_eq!(reference.resulting_params(), &[100_000_000.0]);

        let matched = st
            .norm_match_leave_one_out(&leave_one_out, &reference)
            .unwrap();
        assert_eq!(matched.applied_step(), &[-1.0]);
        assert_eq!(matched.stats().outer.applied_step_norm, 1.0);
        assert_eq!(matched.resulting_params(), reference.resulting_params());
        assert_eq!(matched.norm_match_scale(), 1.0 / 16.0);
    }

    #[test]
    fn adaptive_step_scaling_preserves_optimizer_transition_across_modes() {
        let cases = [
            (merge::OuterOptimizer::Nesterov, 0.0f32, "sgd"),
            (merge::OuterOptimizer::Nesterov, 0.5f32, "nesterov"),
            (
                merge::OuterOptimizer::NormalizedEma,
                0.6f32,
                "normalized-ema",
            ),
            (merge::OuterOptimizer::RestartedEma, 0.6f32, "restarted-ema"),
            (
                merge::OuterOptimizer::CappedNesterov,
                0.0f32,
                "capped-nesterov",
            ),
            (
                merge::OuterOptimizer::CappedNesterovGc,
                0.0f32,
                "capped-nesterov-gc",
            ),
            (
                merge::OuterOptimizer::CappedNesterovR,
                0.0f32,
                "capped-nesterov-r",
            ),
            (
                merge::OuterOptimizer::CappedNesterovCurv,
                0.0f32,
                "capped-nesterov-curv",
            ),
        ];
        for (optimizer, momentum, label) in cases {
            let mut st = GlobalState::new(
                Layout {
                    fragments: vec![FragmentInfo {
                        merge_mode: MERGE_AVG,
                        tensor_numels: vec![2, 2],
                        tensor_shapes: None,
                    }],
                },
                None,
                0.4,
                momentum,
                crate::protocol::DTYPE_F32,
            );
            st.outer_optimizer = optimizer;
            st.outer_restart_cos_threshold = -0.2;
            st.init_fragment(0, vec![2.0, -1.0, 3.0, 0.5]).unwrap();
            st.merge_and_step(0, &[&[1.5f32, -1.5, 2.0, 0.0]], &[1.0])
                .unwrap();

            let base = st.params[0].clone();
            let learner_2 = [base[0] - 1.0, base[1] + 0.5, base[2], base[3] - 0.25];
            let learner_7 = [base[0] + 0.5, base[1] - 1.5, base[2] - 0.75, base[3]];
            let candidates = [
                MergeCandidate::new(2, &learner_2, 1.0),
                MergeCandidate::new(7, &learner_7, 3.0),
            ];
            let full = st.build_full_aggregate(0, &candidates).unwrap();
            let preview = st.preview_aggregate(&full, 1).unwrap();
            let params_before = st.params.clone();
            let momentum_before = st.momentum.clone();
            let bounds = StepScaleBounds::new(0.25, 2.0).unwrap();

            let scaled = st.scale_full_group_preview(&preview, 0.5, bounds).unwrap();
            assert_eq!(st.params, params_before, "{label}");
            assert_eq!(st.momentum, momentum_before, "{label}");
            assert_eq!(
                scaled.resulting_optimizer_buffer(),
                preview.resulting_optimizer_buffer(),
                "{label}"
            );
            let expected_step: Vec<f32> = preview
                .applied_step()
                .iter()
                .map(|step| (0.5f64 * *step as f64) as f32)
                .collect();
            assert_eq!(scaled.applied_step(), expected_step, "{label}");
            let expected_params: Vec<f32> = base
                .iter()
                .zip(&expected_step)
                .map(|(param, step)| *param - *step)
                .collect();
            assert_eq!(scaled.resulting_params(), expected_params, "{label}");
            assert_eq!(scaled.stats().gnorm, preview.stats().gnorm, "{label}");
            assert_eq!(
                scaled.stats().outer.direction_delta_cosine,
                preview.stats().outer.direction_delta_cosine,
                "{label}"
            );
            assert_eq!(
                scaled.stats().outer.history_current_norm_ratio,
                preview.stats().outer.history_current_norm_ratio,
                "{label}"
            );
            assert_eq!(
                scaled.stats().outer.restarted,
                preview.stats().outer.restarted,
                "{label}"
            );
            assert_eq!(scaled.step_scale(), 0.5, "{label}");
            assert_eq!(scaled.action_stats().step_scale, 0.5, "{label}");
            assert_eq!(
                scaled.action_stats().unscaled_applied_step_norm,
                preview.stats().outer.applied_step_norm,
                "{label}"
            );
            assert_eq!(
                scaled.action_stats().action_fingerprint,
                scaled.action_fingerprint(),
                "{label}"
            );
            assert_ne!(
                scaled.action_fingerprint(),
                preview.action_fingerprint(),
                "{label}"
            );

            let expected_params = scaled.resulting_params().to_vec();
            let expected_buffer = scaled.resulting_optimizer_buffer().to_vec();
            let expected_stats = scaled.stats();
            assert_eq!(
                st.commit_preview(scaled).unwrap(),
                expected_stats,
                "{label}"
            );
            assert_eq!(st.params[0], expected_params, "{label}");
            assert_eq!(st.momentum[0], expected_buffer, "{label}");
            assert_eq!(st.versions[0], 1, "{label}");
        }
    }

    #[test]
    fn adaptive_step_scaling_is_consistent_on_the_f32_lattice() {
        let mut st = GlobalState::new(
            Layout {
                fragments: vec![FragmentInfo {
                    merge_mode: MERGE_AVG,
                    tensor_numels: vec![1],
                    tensor_shapes: None,
                }],
            },
            None,
            0.125,
            0.0,
            crate::protocol::DTYPE_F32,
        );
        st.init_fragment(0, vec![100_000_000.0]).unwrap();
        let learner = [100_000_128.0f32];
        let aggregate = st
            .build_full_aggregate(0, &[MergeCandidate::new(4, &learner, 1.0)])
            .unwrap();
        let preview = st.preview_aggregate(&aggregate, 1).unwrap();
        assert_eq!(preview.applied_step(), &[-16.0]);
        assert_eq!(preview.resulting_optimizer_buffer(), &[-128.0]);

        let scaled = st
            .scale_full_group_preview(
                &preview,
                1.0 / 16.0,
                StepScaleBounds::new(1.0 / 32.0, 2.0).unwrap(),
            )
            .unwrap();
        assert_eq!(scaled.applied_step(), &[-1.0]);
        assert_eq!(scaled.stats().outer.applied_step_norm, 1.0);
        assert_eq!(scaled.resulting_params(), &[100_000_000.0]);
        assert_eq!(scaled.resulting_optimizer_buffer(), &[-128.0]);
        assert_eq!(scaled.step_scale(), 1.0 / 16.0);

        st.commit_preview(scaled).unwrap();
        assert_eq!(st.params[0], vec![100_000_000.0]);
        assert_eq!(st.momentum[0], vec![-128.0]);
    }

    #[test]
    fn adaptive_step_scaling_rejects_bad_scalars_subsets_and_broken_seals() {
        assert!(StepScaleBounds::new(0.0, 1.0).is_err());
        assert!(StepScaleBounds::new(-1.0, 1.0).is_err());
        assert!(StepScaleBounds::new(2.0, 1.0).is_err());
        assert!(StepScaleBounds::new(f64::NAN, 1.0).is_err());
        assert!(StepScaleBounds::new(0.5, f64::INFINITY).is_err());

        let mut st = GlobalState::new(layout2(), None, 0.5, 0.0, crate::protocol::DTYPE_F32);
        st.init_fragment(0, vec![1.0; 4]).unwrap();
        let learner_1 = [0.0f32; 4];
        let learner_2 = [0.5f32; 4];
        let candidates = [
            MergeCandidate::new(1, &learner_1, 1.0),
            MergeCandidate::new(2, &learner_2, 1.0),
        ];
        let full = st.build_full_aggregate(0, &candidates).unwrap();
        let full_preview = st.preview_aggregate(&full, 1).unwrap();
        let bounds = StepScaleBounds::new(0.25, 2.0).unwrap();
        for bad_scalar in [0.0, -0.5, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(st
                .scale_full_group_preview(&full_preview, bad_scalar, bounds)
                .is_err());
        }
        assert!(st
            .scale_full_group_preview(&full_preview, 0.1, bounds)
            .is_err());
        assert!(st
            .scale_full_group_preview(&full_preview, 2.5, bounds)
            .is_err());

        let subset = st.build_selected_aggregate(0, &candidates, &[1]).unwrap();
        let subset_preview = st.preview_aggregate(&subset, 1).unwrap();
        assert!(st
            .scale_full_group_preview(&subset_preview, 0.5, bounds)
            .is_err());
        let scaled_once = st
            .scale_full_group_preview(&full_preview, 0.5, bounds)
            .unwrap();
        assert!(st
            .scale_full_group_preview(&scaled_once, 0.5, bounds)
            .is_err());

        let params_before = st.params.clone();
        let momentum_before = st.momentum.clone();
        let mut malformed = full_preview.clone();
        malformed.resulting_params[0] += 1.0;
        assert!(st
            .scale_full_group_preview(&malformed, 0.5, bounds)
            .is_err());
        assert!(st.commit_preview(malformed).is_err());
        let mut scalar_tampered = full_preview.clone();
        scalar_tampered.step_scale = 0.5;
        assert!(st.commit_preview(scalar_tampered).is_err());
        assert_eq!(st.params, params_before);
        assert_eq!(st.momentum, momentum_before);
    }

    #[test]
    fn preview_rejects_invalid_candidates_and_malformed_outer_policy() {
        let mut st = GlobalState::new(layout2(), None, 0.5, 0.0, crate::protocol::DTYPE_F32);
        let finite = [0.0f32; 4];
        let valid = [MergeCandidate::new(1, &finite, 1.0)];
        assert!(st.build_full_aggregate(0, &valid).is_err());

        st.init_fragment(0, vec![1.0; 4]).unwrap();
        let zero_weight = [MergeCandidate::new(1, &finite, 0.0)];
        assert!(st.build_full_aggregate(0, &zero_weight).is_err());
        let nonfinite = [0.0f32, f32::NAN, 0.0, 0.0];
        assert!(st
            .build_full_aggregate(0, &[MergeCandidate::new(1, &nonfinite, 1.0)])
            .is_err());

        st.outer_lr_by_fragment = Some(Vec::new());
        let aggregate = st.build_full_aggregate(0, &valid).unwrap();
        assert!(st.preview_aggregate(&aggregate, 1).is_err());
    }

    #[test]
    fn preview_fingerprint_rejects_direct_state_and_policy_changes() {
        let mut params_changed =
            GlobalState::new(layout2(), None, 0.5, 0.0, crate::protocol::DTYPE_F32);
        params_changed.init_fragment(0, vec![1.0; 4]).unwrap();
        let learner = [0.0f32; 4];
        let aggregate = params_changed
            .build_full_aggregate(0, &[MergeCandidate::new(1, &learner, 1.0)])
            .unwrap();
        let preview = params_changed.preview_aggregate(&aggregate, 1).unwrap();
        params_changed.params[0][0] = 7.0;
        let changed_params = params_changed.params.clone();
        assert!(params_changed.commit_preview(preview).is_err());
        assert_eq!(params_changed.params, changed_params);

        let mut policy_changed =
            GlobalState::new(layout2(), None, 0.5, 0.0, crate::protocol::DTYPE_F32);
        policy_changed.init_fragment(0, vec![1.0; 4]).unwrap();
        let aggregate = policy_changed
            .build_full_aggregate(0, &[MergeCandidate::new(1, &learner, 1.0)])
            .unwrap();
        let preview = policy_changed.preview_aggregate(&aggregate, 1).unwrap();
        policy_changed.outer_lr = 0.25;
        assert!(policy_changed.commit_preview(preview).is_err());
    }

    #[test]
    fn unrelated_fragment_commit_does_not_invalidate_preview() {
        let mut st = GlobalState::new(layout2(), None, 0.5, 0.0, crate::protocol::DTYPE_F32);
        st.init_fragment(0, vec![1.0; 4]).unwrap();
        st.init_fragment(1, vec![1.0; 4]).unwrap();
        let learner = [0.0f32; 4];
        let aggregate = st
            .build_full_aggregate(0, &[MergeCandidate::new(1, &learner, 1.0)])
            .unwrap();
        let preview = st.preview_aggregate(&aggregate, 2).unwrap();
        st.merge_and_step(1, &[&learner], &[1.0]).unwrap();
        st.versions[1] = 1;
        st.global_step = 1;

        st.commit_preview(preview).unwrap();
        assert_eq!(st.versions, vec![2, 1]);
        assert_eq!(st.global_step, 2);
    }

    #[test]
    fn compatibility_merge_matches_preview_commit_across_production_modes() {
        for (merge_mode, tensor_shapes) in [
            (MERGE_AVG, None),
            (MERGE_RDA, None),
            (MERGE_ISO, Some(vec![(1u64, 2u64), (2, 1)])),
            (MERGE_WORKER_SNR, None),
        ] {
            for optimizer in [
                merge::OuterOptimizer::Nesterov,
                merge::OuterOptimizer::NormalizedEma,
                merge::OuterOptimizer::RestartedEma,
                merge::OuterOptimizer::CappedNesterov,
                merge::OuterOptimizer::CappedNesterovGc,
                merge::OuterOptimizer::CappedNesterovR,
                merge::OuterOptimizer::CappedNesterovCurv,
                merge::OuterOptimizer::BlockRms,
                merge::OuterOptimizer::BlockYogi,
            ] {
                for use_heloco in [false, true] {
                    let make_state = || {
                        let layout = Layout {
                            fragments: vec![FragmentInfo {
                                merge_mode,
                                tensor_numels: vec![2, 2],
                                tensor_shapes: tensor_shapes.clone(),
                            }],
                        };
                        let mut st =
                            GlobalState::new(layout, None, 0.35, 0.6, crate::protocol::DTYPE_F32);
                        st.outer_optimizer = optimizer;
                        st.outer_restart_cos_threshold = -0.1;
                        st.init_fragment(0, vec![1.0, -2.0, 0.5, 3.0]).unwrap();
                        st.merge_and_step(0, &[&[0.5f32, -2.5, 0.0, 2.5]], &[2.0])
                            .unwrap();
                        if use_heloco {
                            st.delta_correction = Some(merge::Heloco::default());
                        }
                        st
                    };
                    let mut compatibility = make_state();
                    let mut preview_path = make_state();
                    let base = compatibility.params[0].clone();
                    let learner_2 = [base[0] - 0.5, base[1] + 1.0, base[2], base[3] - 0.75];
                    let learner_4 = [base[0] + 1.25, base[1], base[2] - 0.5, base[3] + 0.25];
                    let learner_9 = [base[0] - 0.1, base[1] - 0.2, base[2] + 1.0, base[3]];
                    let weights = [1.0, 3.0, 2.0];

                    let compatibility_stats = compatibility
                        .merge_and_step(0, &[&learner_2, &learner_4, &learner_9], &weights)
                        .unwrap();
                    let candidates = [
                        MergeCandidate::new(2, &learner_2, weights[0]),
                        MergeCandidate::new(4, &learner_4, weights[1]),
                        MergeCandidate::new(9, &learner_9, weights[2]),
                    ];
                    let aggregate = preview_path.build_full_aggregate(0, &candidates).unwrap();
                    let preview = preview_path.preview_aggregate(&aggregate, 1).unwrap();
                    let preview_stats = preview_path.commit_preview(preview).unwrap();

                    assert_eq!(compatibility_stats, preview_stats);
                    assert_eq!(compatibility.params, preview_path.params);
                    assert_eq!(compatibility.momentum, preview_path.momentum);
                    assert_eq!(compatibility.capped_mu, preview_path.capped_mu);
                    assert_eq!(compatibility.capped_gain, preview_path.capped_gain);
                    assert_eq!(compatibility.block_v, preview_path.block_v);
                    assert_eq!(compatibility.state_epochs, preview_path.state_epochs);
                    assert_eq!(compatibility.versions, vec![0]);
                    assert_eq!(compatibility.global_step, 0);
                    assert_eq!(preview_path.versions, vec![1]);
                    assert_eq!(preview_path.global_step, 1);
                }
            }
        }
    }

    #[test]
    fn checkpoint_roundtrip() {
        let dir = std::env::temp_dir().join("yeto-ckpt-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("state.ckpt");
        let mut st = GlobalState::new(
            layout2(),
            Some("{\"task\":\"nava\"}".to_string()),
            0.7,
            0.9,
            crate::protocol::DTYPE_F32,
        );
        st.init_fragment(0, vec![1.5; 4]).unwrap();
        st.init_fragment(1, vec![-2.0; 4]).unwrap();
        let learner = vec![0.0f32; 4];
        st.merge_and_step(0, &[&learner], &[1.0]).unwrap();
        st.global_step = 7;
        st.versions[0] = 7;
        st.record_merge(3, 12, 4096);
        st.save_checkpoint(&path).unwrap();

        let mut st2 = GlobalState::new(
            layout2(),
            Some("{\"task\":\"nava\"}".to_string()),
            0.7,
            0.9,
            crate::protocol::DTYPE_F32,
        );
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
