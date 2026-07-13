//! Exact online retained-preview action probing.
//!
//! Rust owns all merge and optimizer math. This module only serializes the
//! resulting complete LoRA state plus five already-materialized LOO or scalar
//! step-scale trials, talks to the persistent loopback sidecar, and verifies
//! that the returned decision is bound to those exact bytes before a preview
//! can be committed.

use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::net::SocketAddr;
use std::path::Path;
use std::str::FromStr;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use crate::state::{ActionPreview, AggregateDelta, GlobalState, MergeCandidate, StepScaleBounds};

pub const PROTOCOL: &str = "yeto-action-probe-v1";
const FRAME_MAGIC: &[u8; 8] = b"YETOAP01";
const FRAME_PREFIX_BYTES: usize = 20;
const MAX_HEADER_BYTES: usize = 1024 * 1024;
const MAX_PAYLOAD_BYTES: usize = 2 * 1024 * 1024 * 1024;
pub const ACTION_NAMES: [&str; 5] = ["A0", "A1", "A2", "A3", "A4"];
pub const LR_PREVIEW_MULTIPLIERS: [f64; 5] = [1.0, 0.75, 1.125, 1.25, 1.5];
const LEAVE_ONE_OUT_ACTION_FAMILY: &str = "leave_one_out";
const STEP_SCALE_ACTION_FAMILY: &str = "step_scale";
const MIN_SELECTED_MASS: f64 = 0.70;
const MIN_NORM_MULTIPLIER: f64 = 0.5;
const MAX_NORM_MULTIPLIER: f64 = 2.0;
const MAX_STEP_NORM_RELATIVE_ERROR: f64 = 0.01;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommitPolicy {
    TokenWeighted,
    ProbeShadow,
    ProbeLooV1,
    ProbeLrShadow,
    ProbeLrV1,
    ProbeCttnV1,
}

impl CommitPolicy {
    pub const fn requires_probe(self) -> bool {
        !matches!(self, Self::TokenWeighted)
    }

    pub const fn is_shadow(self) -> bool {
        matches!(self, Self::ProbeShadow | Self::ProbeLrShadow)
    }

    pub const fn is_leave_one_out(self) -> bool {
        matches!(self, Self::ProbeShadow | Self::ProbeLooV1)
    }

    pub const fn step_scale_multipliers(self) -> Option<&'static [f64; 5]> {
        match self {
            Self::ProbeLrShadow | Self::ProbeLrV1 => Some(&LR_PREVIEW_MULTIPLIERS),
            _ => None,
        }
    }
}

impl fmt::Display for CommitPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::TokenWeighted => "token_weighted",
            Self::ProbeShadow => "probe_shadow",
            Self::ProbeLooV1 => "probe_loo_v1",
            Self::ProbeLrShadow => "probe_lr_shadow",
            Self::ProbeLrV1 => "probe_lr_v1",
            Self::ProbeCttnV1 => "cttn_v1",
        })
    }
}

impl FromStr for CommitPolicy {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "token_weighted" | "token-weighted" => Ok(Self::TokenWeighted),
            "probe_shadow" | "probe-shadow" => Ok(Self::ProbeShadow),
            "probe_loo_v1" | "probe-loo-v1" => Ok(Self::ProbeLooV1),
            "probe_lr_shadow" | "probe-lr-shadow" => Ok(Self::ProbeLrShadow),
            "probe_lr_v1" | "probe-lr-v1" => Ok(Self::ProbeLrV1),
            "cttn_v1" | "cttn-v1" => Ok(Self::ProbeCttnV1),
            _ => Err(format!(
                "commit policy must be token_weighted, probe_shadow, probe_loo_v1, probe_lr_shadow, probe_lr_v1, or cttn_v1; got {value:?}"
            )),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ClientConfig {
    endpoint: SocketAddr,
    timeout: Duration,
    run_uuid: String,
    expected: ExpectedProbeConfig,
}

impl ClientConfig {
    pub fn from_expected_file(
        endpoint: &str,
        timeout: Duration,
        run_uuid: String,
        expected_path: &Path,
    ) -> Result<Self> {
        let endpoint: SocketAddr = endpoint
            .parse()
            .with_context(|| format!("invalid action-probe endpoint {endpoint:?}"))?;
        if !endpoint.ip().is_loopback() || endpoint.port() == 0 {
            bail!("action-probe endpoint must be a numeric loopback address with a nonzero port");
        }
        if timeout.is_zero() {
            bail!("action-probe timeout must be positive");
        }
        if run_uuid.is_empty() || run_uuid.len() > 256 {
            bail!("action-probe run UUID must contain 1..=256 bytes");
        }
        let raw = std::fs::read(expected_path).with_context(|| {
            format!(
                "read expected action-probe config {}",
                expected_path.display()
            )
        })?;
        let expected = ExpectedProbeConfig::parse(&raw).with_context(|| {
            format!(
                "parse expected action-probe config {}",
                expected_path.display()
            )
        })?;
        Ok(Self {
            endpoint,
            timeout,
            run_uuid,
            expected,
        })
    }
}

#[derive(Clone, Debug)]
struct ExpectedProbeConfig {
    anchor_manifest_sha256: String,
    anchor_tensors_sha256: String,
    probe_config_sha256: String,
    layout_hash: String,
    fragment_pattern: String,
    lora_r: usize,
    fragment_names: Vec<Vec<String>>,
    tensor_shapes: BTreeMap<String, Vec<usize>>,
}

impl ExpectedProbeConfig {
    fn parse(raw: &[u8]) -> Result<Self> {
        let root = strict_json(raw)?;
        let object = root
            .as_object()
            .context("expected probe config must be a JSON object")?;
        let protocol = string_field(object, "protocol")?;
        if protocol != PROTOCOL {
            bail!("expected probe config protocol is {protocol:?}, expected {PROTOCOL:?}");
        }
        let anchor_manifest_sha256 = digest_field(object, "anchor_manifest_sha256")?;
        let anchor_tensors_sha256 = digest_field(object, "anchor_tensors_sha256")?;
        let probe_config_sha256 = digest_field(object, "probe_config_sha256")?;
        let layout_hash = digest_field(object, "layout_hash")?;
        let fragment_pattern = string_field(object, "fragment_pattern")?.to_owned();
        if fragment_pattern.is_empty() || fragment_pattern.len() > 128 {
            bail!("expected fragment_pattern must contain 1..=128 bytes");
        }
        let lora_r = usize_field(object, "lora_r")?;
        if lora_r == 0 {
            bail!("expected lora_r must be positive");
        }

        let fragment_value = object
            .get("fragment_layout")
            .context("expected probe config is missing fragment_layout")?;
        let fragment_object = fragment_value
            .as_object()
            .context("fragment_layout must be an object keyed by fragment id")?;
        if fragment_object.is_empty() {
            bail!("fragment_layout must not be empty");
        }
        let mut by_id = BTreeMap::new();
        for (id_text, value) in fragment_object {
            let id: usize = id_text
                .parse()
                .with_context(|| format!("fragment_layout key {id_text:?} is not an integer"))?;
            let names = value
                .as_array()
                .with_context(|| format!("fragment_layout[{id}] must be a list"))?;
            if names.is_empty() {
                bail!("fragment_layout[{id}] must not be empty");
            }
            let names: Vec<String> = names
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    let name = value.as_str().with_context(|| {
                        format!("fragment_layout[{id}][{index}] must be a string")
                    })?;
                    if name.is_empty() || name.len() > 1024 {
                        bail!("fragment tensor names must contain 1..=1024 bytes");
                    }
                    Ok(name.to_owned())
                })
                .collect::<Result<_>>()?;
            if by_id.insert(id, names).is_some() {
                bail!("duplicate fragment_layout id {id}");
            }
        }
        let expected_ids: Vec<usize> = (0..by_id.len()).collect();
        if by_id.keys().copied().collect::<Vec<_>>() != expected_ids {
            bail!("fragment_layout ids must be contiguous from zero");
        }
        let fragment_names = by_id.into_values().collect::<Vec<_>>();
        let mut all_names = HashSet::new();
        for name in fragment_names.iter().flatten() {
            if !all_names.insert(name.clone()) {
                bail!("fragment_layout contains duplicate tensor name {name:?}");
            }
        }

        let mut tensor_shapes = BTreeMap::new();
        if let Some(value) = object.get("tensor_shapes") {
            let shapes = value
                .as_object()
                .context("tensor_shapes must be an object when present")?;
            for (name, value) in shapes {
                if !all_names.contains(name) {
                    bail!("tensor_shapes contains unknown tensor {name:?}");
                }
                let dims = value
                    .as_array()
                    .with_context(|| format!("tensor_shapes[{name:?}] must be a list"))?;
                if dims.is_empty() {
                    bail!("tensor_shapes[{name:?}] must not be empty");
                }
                let dims = dims
                    .iter()
                    .enumerate()
                    .map(|(index, dim)| {
                        dim.as_u64()
                            .and_then(|value| usize::try_from(value).ok())
                            .filter(|value| *value > 0)
                            .with_context(|| {
                                format!("tensor_shapes[{name:?}][{index}] must be positive")
                            })
                    })
                    .collect::<Result<Vec<_>>>()?;
                tensor_shapes.insert(name.clone(), dims);
            }
        }

        Ok(Self {
            anchor_manifest_sha256,
            anchor_tensors_sha256,
            probe_config_sha256,
            layout_hash,
            fragment_pattern,
            lora_r,
            fragment_names,
            tensor_shapes,
        })
    }
}

#[derive(Clone, Debug)]
struct TensorBinding {
    name: String,
    shape: Vec<usize>,
    fragment_id: usize,
    offset: usize,
    numel: usize,
}

#[derive(Clone, Debug)]
struct BoundLayout {
    state_tensors: Vec<TensorBinding>,
    fragment_names: Vec<Vec<String>>,
}

impl BoundLayout {
    fn bind(expected: &ExpectedProbeConfig, state: &GlobalState) -> Result<Self> {
        if expected.fragment_names.len() != state.layout.fragments.len() {
            bail!(
                "expected probe config has {} fragments, syncer layout has {}",
                expected.fragment_names.len(),
                state.layout.fragments.len()
            );
        }
        let mut state_tensors = Vec::new();
        let mut seen = HashSet::new();
        for (fragment_id, (names, fragment)) in expected
            .fragment_names
            .iter()
            .zip(&state.layout.fragments)
            .enumerate()
        {
            if names.len() != fragment.tensor_numels.len() {
                bail!(
                    "fragment {fragment_id}: expected config has {} tensors, syncer layout has {}",
                    names.len(),
                    fragment.tensor_numels.len()
                );
            }
            let mut offset = 0usize;
            for (name, &numel_u64) in names.iter().zip(&fragment.tensor_numels) {
                if !seen.insert(name.clone()) {
                    bail!("duplicate tensor name {name:?} in bound probe layout");
                }
                let numel = usize::try_from(numel_u64)
                    .with_context(|| format!("tensor {name:?} numel does not fit usize"))?;
                let shape = if let Some(shape) = expected.tensor_shapes.get(name) {
                    shape.clone()
                } else {
                    infer_lora_shape(name, numel, expected.lora_r)?
                };
                let shape_numel = shape.iter().try_fold(1usize, |product, &dim| {
                    product
                        .checked_mul(dim)
                        .context("tensor shape product overflow")
                })?;
                if shape_numel != numel {
                    bail!(
                        "tensor {name:?} shape {shape:?} has {shape_numel} values, syncer layout has {numel}"
                    );
                }
                state_tensors.push(TensorBinding {
                    name: name.clone(),
                    shape,
                    fragment_id,
                    offset,
                    numel,
                });
                offset = offset
                    .checked_add(numel)
                    .context("fragment tensor offset overflow")?;
            }
            if offset != fragment.numel() {
                bail!(
                    "fragment {fragment_id}: tensor bindings cover {offset} values, layout has {}",
                    fragment.numel()
                );
            }
        }

        let layout_hash =
            layout_contract_digest(&expected.fragment_pattern, &state.layout, &state_tensors)?;
        if layout_hash != expected.layout_hash {
            bail!(
                "bound syncer layout SHA-256 {layout_hash} does not match expected sidecar layout {}",
                expected.layout_hash
            );
        }
        state_tensors.sort_by(|left, right| left.name.cmp(&right.name));
        Ok(Self {
            state_tensors,
            fragment_names: expected.fragment_names.clone(),
        })
    }
}

fn infer_lora_shape(name: &str, numel: usize, rank: usize) -> Result<Vec<usize>> {
    if numel == 0 || numel % rank != 0 {
        bail!("LoRA tensor {name:?} numel {numel} is not divisible by rank {rank}");
    }
    let other = numel / rank;
    if name.contains("lora_A") || name.contains("lora_embedding_A") {
        Ok(vec![rank, other])
    } else if name.contains("lora_B") || name.contains("lora_embedding_B") {
        Ok(vec![other, rank])
    } else {
        bail!(
            "cannot infer shape for non-LoRA tensor {name:?}; add it to tensor_shapes in the expected probe config"
        )
    }
}

fn layout_contract_digest(
    pattern: &str,
    layout: &crate::state::Layout,
    tensors: &[TensorBinding],
) -> Result<String> {
    let mut fragments = Vec::with_capacity(layout.fragments.len());
    for (fragment_id, fragment) in layout.fragments.iter().enumerate() {
        let fragment_tensors = tensors
            .iter()
            .filter(|tensor| tensor.fragment_id == fragment_id)
            .map(|tensor| {
                json!({
                    "name": tensor.name,
                    "shape": tensor.shape,
                    "numel": tensor.numel,
                })
            })
            .collect::<Vec<_>>();
        fragments.push(json!({
            "id": fragment_id,
            "merge_mode": fragment.merge_mode,
            "tensors": fragment_tensors,
        }));
    }
    let contract = json!({"pattern": pattern, "fragments": fragments});
    Ok(sha256_hex(&canonical_json(&contract)?))
}

#[derive(Clone, Debug, PartialEq)]
pub enum RetainedActionKind {
    Baseline,
    LeaveOneOut { omitted_responder_id: u32 },
    ScaledFullGroup { multiplier: f64 },
}

#[derive(Clone, Debug)]
pub struct ActionMetadata {
    pub kind: RetainedActionKind,
    pub eligible: bool,
    pub selected_mass: f64,
    pub norm_multiplier: f64,
    pub step_norm_ratio: f64,
    pub step_scale: Option<f64>,
    pub ineligible_reason: Option<String>,
}

impl ActionMetadata {
    fn baseline() -> Self {
        Self {
            kind: RetainedActionKind::Baseline,
            eligible: true,
            selected_mass: 1.0,
            norm_multiplier: 1.0,
            step_norm_ratio: 1.0,
            step_scale: None,
            ineligible_reason: None,
        }
    }

    fn omitted_responder_id(&self) -> Option<u32> {
        match self.kind {
            RetainedActionKind::LeaveOneOut {
                omitted_responder_id,
            } => Some(omitted_responder_id),
            _ => None,
        }
    }
}

struct RetainedAction {
    name: String,
    preview: Option<ActionPreview>,
    metadata: ActionMetadata,
}

/// An owned set of sealed previews. Construction remains independent from the
/// family-specific wire validation used by LOO and scalar step-scale probes.
pub struct RetainedPreviews {
    actions: Vec<RetainedAction>,
}

impl RetainedPreviews {
    pub fn loo_v1(
        baseline: ActionPreview,
        alternatives: Vec<(ActionPreview, ActionMetadata)>,
    ) -> Result<Self> {
        if alternatives.len() != 4 {
            bail!("action-probe preview set requires exactly four alternatives");
        }
        let mut actions = Vec::with_capacity(5);
        actions.push(RetainedAction {
            name: ACTION_NAMES[0].to_owned(),
            preview: Some(baseline),
            metadata: ActionMetadata::baseline(),
        });
        for (index, (preview, metadata)) in alternatives.into_iter().enumerate() {
            actions.push(RetainedAction {
                name: ACTION_NAMES[index + 1].to_owned(),
                preview: Some(preview),
                metadata,
            });
        }
        let retained = Self { actions };
        retained.validate_loo_v1()?;
        Ok(retained)
    }

    pub fn generic(actions: Vec<(String, ActionPreview, ActionMetadata)>) -> Result<Self> {
        if actions.is_empty() {
            bail!("retained preview set must not be empty");
        }
        let mut names = HashSet::new();
        let identity = {
            let preview = &actions[0].1;
            (
                preview.fragment_id(),
                preview.base_version(),
                preview.base_state_epoch(),
                preview.target_version(),
            )
        };
        let actions = actions
            .into_iter()
            .map(|(name, preview, metadata)| {
                if name.is_empty() || !names.insert(name.clone()) {
                    bail!("retained action names must be nonempty and unique");
                }
                if (
                    preview.fragment_id(),
                    preview.base_version(),
                    preview.base_state_epoch(),
                    preview.target_version(),
                ) != identity
                {
                    bail!("retained actions must target one identical fragment state");
                }
                if !metadata.selected_mass.is_finite()
                    || !(0.0..=1.0).contains(&metadata.selected_mass)
                    || !metadata.norm_multiplier.is_finite()
                    || metadata.norm_multiplier <= 0.0
                    || !metadata.step_norm_ratio.is_finite()
                    || metadata.step_norm_ratio < 0.0
                    || metadata
                        .step_scale
                        .is_some_and(|value| !value.is_finite() || value <= 0.0)
                {
                    bail!("retained action metadata must be finite and nonnegative");
                }
                Ok(RetainedAction {
                    name,
                    preview: Some(preview),
                    metadata,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { actions })
    }

    pub fn len(&self) -> usize {
        self.actions.len()
    }

    pub fn name(&self, index: usize) -> &str {
        &self.actions[index].name
    }

    pub fn preview(&self, index: usize) -> &ActionPreview {
        self.actions[index]
            .preview
            .as_ref()
            .expect("action preview was consumed before request completion")
    }

    pub fn metadata(&self, index: usize) -> &ActionMetadata {
        &self.actions[index].metadata
    }

    pub fn take(&mut self, index: usize) -> ActionPreview {
        self.actions[index]
            .preview
            .take()
            .expect("selected action preview was already consumed")
    }

    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.actions.iter().position(|action| action.name == name)
    }

    fn validate_loo_v1(&self) -> Result<()> {
        if self.actions.len() != ACTION_NAMES.len() {
            bail!("LOO v1 requires exactly A0-A4");
        }
        let mut omitted = HashSet::new();
        for (index, expected_name) in ACTION_NAMES.iter().enumerate() {
            let action = &self.actions[index];
            if action.name != *expected_name {
                bail!("LOO v1 action {index} must be named {expected_name}");
            }
            match (&action.metadata.kind, index) {
                (RetainedActionKind::Baseline, 0) if action.metadata.step_scale.is_none() => {}
                (
                    RetainedActionKind::LeaveOneOut {
                        omitted_responder_id,
                    },
                    1..=4,
                ) if action.metadata.step_scale.is_none()
                    && omitted.insert(*omitted_responder_id) => {}
                _ => bail!("LOO v1 requires A0 baseline plus four distinct omissions"),
            }
        }
        Ok(())
    }

    fn validate_step_scale_v1(&self) -> Result<()> {
        if self.actions.len() != ACTION_NAMES.len() {
            bail!("step-scale v1 requires exactly A0-A4");
        }
        for (index, (&expected_name, &expected_scale)) in ACTION_NAMES
            .iter()
            .zip(LR_PREVIEW_MULTIPLIERS.iter())
            .enumerate()
        {
            let action = &self.actions[index];
            if action.name != expected_name {
                bail!("step-scale v1 action {index} must be named {expected_name}");
            }
            let RetainedActionKind::ScaledFullGroup { multiplier } = &action.metadata.kind else {
                bail!("step-scale v1 actions must all be scaled full-group previews");
            };
            let preview = action
                .preview
                .as_ref()
                .expect("action preview was consumed before request completion");
            if multiplier.to_bits() != expected_scale.to_bits()
                || action.metadata.step_scale.map(f64::to_bits) != Some(expected_scale.to_bits())
                || preview.step_scale().to_bits() != expected_scale.to_bits()
                || !action.metadata.eligible
                || action.metadata.selected_mass.to_bits() != 1.0f64.to_bits()
                || action.metadata.ineligible_reason.is_some()
            {
                bail!("step-scale v1 action {expected_name} does not match the frozen grid");
            }
        }
        Ok(())
    }

    fn wire_action_family(&self) -> Result<&'static str> {
        if self.validate_loo_v1().is_ok() {
            return Ok(LEAVE_ONE_OUT_ACTION_FAMILY);
        }
        self.validate_step_scale_v1()?;
        Ok(STEP_SCALE_ACTION_FAMILY)
    }
}

pub fn build_baseline_preview(
    state: &GlobalState,
    fragment_id: usize,
    target_version: u64,
    candidates: &[MergeCandidate<'_>],
) -> Result<ActionPreview> {
    let aggregate = state.build_full_aggregate(fragment_id, candidates)?;
    state.preview_aggregate(&aggregate, target_version)
}

pub fn build_leave_one_out_previews(
    state: &GlobalState,
    fragment_id: usize,
    target_version: u64,
    candidates: &[MergeCandidate<'_>],
    baseline: &ActionPreview,
) -> Result<Vec<(ActionPreview, ActionMetadata)>> {
    if candidates.len() != 4 {
        bail!(
            "probe_loo_v1 requires exactly four responders, got {}",
            candidates.len()
        );
    }
    let ids = candidates
        .iter()
        .map(|candidate| candidate.responder_id)
        .collect::<Vec<_>>();
    let baseline_norm = baseline.stats().outer.applied_step_norm;
    let mut alternatives = Vec::with_capacity(4);
    for omitted_index in 0..4 {
        let selected_ids = ids
            .iter()
            .enumerate()
            .filter_map(|(index, &id)| (index != omitted_index).then_some(id))
            .collect::<Vec<_>>();
        let aggregate = state.build_selected_aggregate(fragment_id, candidates, &selected_ids)?;
        let preview = state.preview_aggregate(&aggregate, target_version)?;
        let matched = state.norm_match_leave_one_out(&preview, baseline)?;
        let norm_multiplier = matched.norm_match_scale();
        if !norm_multiplier.is_finite() || norm_multiplier <= 0.0 {
            bail!(
                "A{} has non-positive norm multiplier {norm_multiplier}",
                omitted_index + 1
            );
        }
        let matched_norm = matched.stats().outer.applied_step_norm;
        let step_norm_ratio = if baseline_norm <= 1e-12 && matched_norm <= 1e-12 {
            1.0
        } else if baseline_norm > 0.0 {
            matched_norm / baseline_norm
        } else {
            bail!("A{} cannot match a zero baseline step", omitted_index + 1);
        };
        if !step_norm_ratio.is_finite() || step_norm_ratio <= 0.0 {
            bail!(
                "A{} has invalid step-norm ratio {step_norm_ratio}",
                omitted_index + 1
            );
        }
        let selected_mass = matched.selected_weight_mass();
        let mass_safe = selected_mass >= MIN_SELECTED_MASS;
        let norm_safe = (MIN_NORM_MULTIPLIER..=MAX_NORM_MULTIPLIER).contains(&norm_multiplier);
        let step_safe = (step_norm_ratio - 1.0).abs() <= MAX_STEP_NORM_RELATIVE_ERROR + 1e-12;
        let ineligible_reason = if !mass_safe {
            Some("selected_mass_below_0.70".to_owned())
        } else if !norm_safe {
            Some("norm_multiplier_outside_0.5_2.0".to_owned())
        } else if !step_safe {
            Some("step_norm_ratio_outside_1pct".to_owned())
        } else {
            None
        };
        alternatives.push((
            matched,
            ActionMetadata {
                kind: RetainedActionKind::LeaveOneOut {
                    omitted_responder_id: ids[omitted_index],
                },
                eligible: mass_safe && norm_safe && step_safe,
                selected_mass,
                norm_multiplier,
                step_norm_ratio,
                step_scale: None,
                ineligible_reason,
            },
        ));
    }
    Ok(alternatives)
}

/// Construct the five frozen scalar actions by scaling only the full-group
/// parameter displacement. A0 is the exact x1 production fallback; A1-A4 are
/// the predeclared selector alternatives.
pub fn build_scaled_full_group_previews(
    state: &GlobalState,
    baseline: &ActionPreview,
    multipliers: &[f64],
) -> Result<RetainedPreviews> {
    if multipliers.len() != ACTION_NAMES.len() {
        bail!("scaled full-group action grid must contain exactly A0-A4");
    }
    if multipliers
        .iter()
        .any(|multiplier| !multiplier.is_finite() || *multiplier <= 0.0)
    {
        bail!("scaled full-group multipliers must be finite and positive");
    }
    let min = multipliers.iter().copied().fold(f64::INFINITY, f64::min);
    let max = multipliers
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let bounds = StepScaleBounds::new(min, max)?;
    let baseline_norm = baseline.stats().outer.applied_step_norm;
    let mut actions = Vec::with_capacity(multipliers.len());
    for (index, &multiplier) in multipliers.iter().enumerate() {
        let preview = if multiplier.to_bits() == 1.0f64.to_bits() {
            baseline.clone()
        } else {
            state.scale_full_group_preview(baseline, multiplier, bounds)?
        };
        let scaled_norm = preview.stats().outer.applied_step_norm;
        let step_norm_ratio = if baseline_norm <= 1e-12 && scaled_norm <= 1e-12 {
            1.0
        } else if baseline_norm > 0.0 {
            scaled_norm / baseline_norm
        } else {
            bail!("scaled action cannot be compared with an invalid zero baseline norm");
        };
        if !step_norm_ratio.is_finite() || step_norm_ratio < 0.0 {
            bail!("scaled action has invalid step-norm ratio {step_norm_ratio}");
        }
        actions.push((
            ACTION_NAMES[index].to_owned(),
            preview,
            ActionMetadata {
                kind: RetainedActionKind::ScaledFullGroup { multiplier },
                eligible: true,
                selected_mass: 1.0,
                norm_multiplier: multiplier,
                step_norm_ratio,
                step_scale: Some(multiplier),
                ineligible_reason: None,
            },
        ));
    }
    let previews = RetainedPreviews::generic(actions)?;
    previews.validate_step_scale_v1()?;
    Ok(previews)
}

#[derive(Clone, Debug)]
pub struct VerifiedSelection {
    pub action_index: usize,
    pub action_name: String,
    pub fallback_reason: Option<String>,
    pub request_digest: String,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CttnDiagnostics {
    pub bind: bool,
    pub tau: f64,
    pub retention: f64,
    pub e_before: f64,
    pub e_after: f64,
    pub budget: f64,
    pub n_modes_90: u64,
    pub ritz_max: f64,
    pub loss: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VerifiedCttn {
    pub d: Vec<f32>,
    pub b_new: Vec<f32>,
    pub diagnostics: CttnDiagnostics,
    pub request_digest: String,
}

#[derive(Debug)]
pub enum ProbeError {
    Timeout,
    Io(String),
    Protocol(String),
    UnsafeResponse(String),
    Remote(String),
}

impl ProbeError {
    pub const fn code(&self) -> &'static str {
        match self {
            Self::Timeout => "probe_timeout",
            Self::Io(_) => "probe_io_error",
            Self::Protocol(_) => "probe_protocol_error",
            Self::UnsafeResponse(_) => "unsafe_probe_response",
            Self::Remote(_) => "probe_remote_error",
        }
    }
}

impl fmt::Display for ProbeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Timeout => f.write_str("action-probe request timed out"),
            Self::Io(message) => write!(f, "action-probe I/O error: {message}"),
            Self::Protocol(message) => write!(f, "action-probe protocol error: {message}"),
            Self::UnsafeResponse(message) => {
                write!(f, "unsafe action-probe response: {message}")
            }
            Self::Remote(message) => write!(f, "action-probe service failed closed: {message}"),
        }
    }
}

impl std::error::Error for ProbeError {}

pub struct ActionProbeClient {
    config: ClientConfig,
    layout: BoundLayout,
    stream: Option<TcpStream>,
    last_request_digest: Option<String>,
}

fn append_state_block(
    state: &GlobalState,
    layout: &BoundLayout,
    payload: &mut Vec<u8>,
) -> std::result::Result<(Vec<Value>, String), ProbeError> {
    let mut state_specs = Vec::with_capacity(layout.state_tensors.len());
    let mut state_hasher = Sha256::new();
    for tensor in &layout.state_tensors {
        let values = state
            .params
            .get(tensor.fragment_id)
            .and_then(|fragment| {
                fragment.get(tensor.offset..tensor.offset.saturating_add(tensor.numel))
            })
            .ok_or_else(|| {
                ProbeError::Protocol(format!(
                    "state tensor {:?} is outside fragment {}",
                    tensor.name, tensor.fragment_id
                ))
            })?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err(ProbeError::Protocol(format!(
                "state tensor {:?} contains NaN or Inf",
                tensor.name
            )));
        }
        let raw = f32_bytes(values);
        let offset = payload.len();
        payload.extend_from_slice(&raw);
        state_specs.push(json!({
            "name": tensor.name,
            "shape": tensor.shape,
            "offset": offset,
            "nbytes": raw.len(),
            "sha256": sha256_hex(&raw),
        }));
        state_hasher.update(tensor.name.as_bytes());
        state_hasher.update([0]);
        let shape_json = canonical_json(&json!({"shape": tensor.shape})).map_err(|error| {
            ProbeError::Protocol(format!("serialize state tensor shape: {error:#}"))
        })?;
        state_hasher.update(shape_json);
        state_hasher.update([0]);
        state_hasher.update(raw);
    }
    Ok((
        state_specs,
        digest_to_hex(state_hasher.finalize().as_slice()),
    ))
}

impl ActionProbeClient {
    pub fn bind(config: ClientConfig, state: &GlobalState) -> Result<Self> {
        let layout = BoundLayout::bind(&config.expected, state)?;
        Ok(Self {
            config,
            layout,
            stream: None,
            last_request_digest: None,
        })
    }

    pub fn last_request_digest(&self) -> Option<&str> {
        self.last_request_digest.as_deref()
    }

    pub async fn select(
        &mut self,
        state: &GlobalState,
        previews: &RetainedPreviews,
        step: u64,
        fragment_id: usize,
    ) -> std::result::Result<VerifiedSelection, ProbeError> {
        self.last_request_digest = None;
        let request = self.build_request(state, previews, step, fragment_id)?;
        self.last_request_digest = Some(request.request_digest.clone());
        let timeout = self.config.timeout;
        let response = match tokio::time::timeout(timeout, self.roundtrip(&request.frame)).await {
            Err(_) => {
                self.stream = None;
                return Err(ProbeError::Timeout);
            }
            Ok(Err(error)) => {
                self.stream = None;
                return Err(error);
            }
            Ok(Ok(response)) => response,
        };
        match verify_response(&response, &request, &self.config, previews) {
            Ok(selection) => Ok(selection),
            Err(error) => {
                self.stream = None;
                Err(error)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn cttn_step(
        &mut self,
        state: &GlobalState,
        aggregate: &AggregateDelta,
        step: u64,
        g: &[f32],
        b: &[f32],
        mu: f32,
        rho: f32,
        block_steps: u32,
    ) -> std::result::Result<VerifiedCttn, ProbeError> {
        self.last_request_digest = None;
        let request =
            self.build_cttn_request(state, aggregate, step, g, b, mu, rho, block_steps)?;
        self.last_request_digest = Some(request.request_digest.clone());
        let timeout = self.config.timeout;
        let response = match tokio::time::timeout(timeout, self.roundtrip(&request.frame)).await {
            Err(_) => {
                self.stream = None;
                return Err(ProbeError::Timeout);
            }
            Ok(Err(error)) => {
                self.stream = None;
                return Err(error);
            }
            Ok(Ok(response)) => response,
        };
        match verify_cttn_response(&response, &request, &self.config) {
            Ok(result) => Ok(result),
            Err(error) => {
                self.stream = None;
                Err(error)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_cttn_request(
        &self,
        state: &GlobalState,
        aggregate: &AggregateDelta,
        step: u64,
        g: &[f32],
        b: &[f32],
        mu: f32,
        rho: f32,
        block_steps: u32,
    ) -> std::result::Result<WireCttnRequest, ProbeError> {
        let fragment_id = aggregate.fragment_id();
        if fragment_id >= self.layout.fragment_names.len() {
            return Err(ProbeError::Protocol(format!(
                "fragment {fragment_id} is outside bound probe layout"
            )));
        }
        if aggregate.base_version() != state.versions[fragment_id] {
            return Err(ProbeError::Protocol(
                "CTTN aggregate is stale before serialization".to_owned(),
            ));
        }
        let expected_numel = state.layout.fragments[fragment_id].numel();
        if g.len() != expected_numel || b.len() != expected_numel {
            return Err(ProbeError::Protocol(format!(
                "CTTN vectors have lengths g={} b={}, expected {expected_numel}",
                g.len(),
                b.len()
            )));
        }
        if g.iter().chain(b).any(|value| !value.is_finite()) {
            return Err(ProbeError::Protocol(
                "CTTN g or b contains NaN or Inf".to_owned(),
            ));
        }
        if !mu.is_finite() || !(0.0..1.0).contains(&mu) {
            return Err(ProbeError::Protocol(
                "CTTN mu must be finite and in [0, 1)".to_owned(),
            ));
        }
        if !rho.is_finite() || rho < 0.0 || block_steps == 0 {
            return Err(ProbeError::Protocol(
                "CTTN rho must be finite and non-negative and block_steps must be positive"
                    .to_owned(),
            ));
        }

        let request_id = format!(
            "cttn-step-{step}-fragment-{fragment_id}-base-{}-epoch-{}",
            aggregate.base_version(),
            aggregate.base_state_epoch()
        );
        let mut payload = Vec::new();
        let (state_specs, state_digest) = append_state_block(state, &self.layout, &mut payload)?;
        let mut vector_specs = Map::new();
        let mut vector_digests = Vec::with_capacity(2);
        for (name, values) in [("g", g), ("b", b)] {
            let raw = f32_bytes(values);
            let digest = sha256_hex(&raw);
            let offset = payload.len();
            payload.extend_from_slice(&raw);
            vector_specs.insert(
                name.to_owned(),
                json!({
                    "offset": offset,
                    "nbytes": raw.len(),
                    "sha256": digest,
                }),
            );
            vector_digests.push(digest);
        }
        if payload.len() > MAX_PAYLOAD_BYTES {
            return Err(ProbeError::Protocol(format!(
                "action-probe payload has {} bytes, limit is {MAX_PAYLOAD_BYTES}",
                payload.len()
            )));
        }
        vector_specs.insert("mu".to_owned(), json!(mu));
        vector_specs.insert("rho".to_owned(), json!(rho));
        vector_specs.insert("block_steps".to_owned(), json!(block_steps));

        let header = json!({
            "protocol": PROTOCOL,
            "type": "cttn_step",
            "request_id": request_id,
            "run_uuid": self.config.run_uuid,
            "step": step,
            "fragment_id": fragment_id,
            "base_version": aggregate.base_version(),
            "state_epoch": aggregate.base_state_epoch(),
            "fragment_versions": state.versions,
            "layout_hash": self.config.expected.layout_hash,
            "anchor_manifest_sha256": self.config.expected.anchor_manifest_sha256,
            "probe_config_sha256": self.config.expected.probe_config_sha256,
            "dtype": "f32le",
            "state": {
                "tensors": state_specs,
                "sha256": state_digest,
            },
            "fragment": {
                "tensor_names": self.layout.fragment_names[fragment_id],
                "numel": expected_numel,
            },
            "cttn": Value::Object(vector_specs),
        });
        let header_bytes = canonical_json(&header)
            .map_err(|error| ProbeError::Protocol(format!("serialize CTTN request: {error:#}")))?;
        if header_bytes.is_empty() || header_bytes.len() > MAX_HEADER_BYTES {
            return Err(ProbeError::Protocol(format!(
                "action-probe header has {} bytes, limit is {MAX_HEADER_BYTES}",
                header_bytes.len()
            )));
        }
        let frame = encode_frame_parts(&header_bytes, &payload)
            .map_err(|error| ProbeError::Protocol(format!("encode CTTN frame: {error:#}")))?;
        let request_digest = sha256_hex(&frame);
        Ok(WireCttnRequest {
            frame,
            request_id,
            request_digest,
            state_digest,
            g_digest: vector_digests[0].clone(),
            b_digest: vector_digests[1].clone(),
            g: g.to_vec(),
            step,
            fragment_id,
            base_version: aggregate.base_version(),
            state_epoch: aggregate.base_state_epoch(),
            fragment_versions: state.versions.clone(),
        })
    }

    fn build_request(
        &self,
        state: &GlobalState,
        previews: &RetainedPreviews,
        step: u64,
        fragment_id: usize,
    ) -> std::result::Result<WireRequest, ProbeError> {
        let action_family = previews.wire_action_family().map_err(|error| {
            ProbeError::Protocol(format!("YETOAP01 v1 action contract: {error:#}"))
        })?;
        if fragment_id >= self.layout.fragment_names.len() {
            return Err(ProbeError::Protocol(format!(
                "fragment {fragment_id} is outside bound probe layout"
            )));
        }
        let baseline = previews.preview(0);
        if baseline.fragment_id() != fragment_id || baseline.target_version() != step {
            return Err(ProbeError::Protocol(
                "baseline preview identity does not match the request".to_owned(),
            ));
        }
        if baseline.base_version() != state.versions[fragment_id] {
            return Err(ProbeError::Protocol(
                "baseline preview is stale before serialization".to_owned(),
            ));
        }
        let request_id = format!(
            "step-{step}-fragment-{fragment_id}-base-{}-epoch-{}",
            baseline.base_version(),
            baseline.base_state_epoch()
        );

        let mut payload = Vec::new();
        let (state_specs, state_digest) = append_state_block(state, &self.layout, &mut payload)?;

        let expected_numel = state.layout.fragments[fragment_id].numel();
        let mut action_specs = Vec::with_capacity(5);
        let mut action_digests = Vec::with_capacity(5);
        for index in 0..previews.len() {
            let action_name = previews.name(index);
            let preview = previews.preview(index);
            if preview.fragment_id() != fragment_id
                || preview.base_version() != baseline.base_version()
                || preview.base_state_epoch() != baseline.base_state_epoch()
                || preview.target_version() != step
                || preview.resulting_params().len() != expected_numel
                || preview
                    .resulting_params()
                    .iter()
                    .any(|value| !value.is_finite())
            {
                return Err(ProbeError::Protocol(format!(
                    "{action_name} preview is stale, malformed, or targets a different state"
                )));
            }
            let raw = f32_bytes(preview.resulting_params());
            let digest = sha256_hex(&raw);
            let offset = payload.len();
            payload.extend_from_slice(&raw);
            let metadata = previews.metadata(index);
            let mut action_spec = json!({
                "name": action_name,
                "offset": offset,
                "nbytes": raw.len(),
                "sha256": digest,
                "eligible": metadata.eligible,
                "omitted_responder_id": metadata.omitted_responder_id(),
                "selected_mass": metadata.selected_mass,
                "norm_multiplier": metadata.norm_multiplier,
                "step_norm_ratio": metadata.step_norm_ratio,
                "ineligible_reason": metadata.ineligible_reason,
            });
            if action_family == STEP_SCALE_ACTION_FAMILY {
                action_spec["step_scale"] = json!(metadata.step_scale.ok_or_else(|| {
                    ProbeError::Protocol(format!(
                        "{action_name} step-scale action lacks explicit step_scale metadata"
                    ))
                })?);
            }
            action_specs.push(action_spec);
            action_digests.push(digest);
        }
        if payload.len() > MAX_PAYLOAD_BYTES {
            return Err(ProbeError::Protocol(format!(
                "action-probe payload has {} bytes, limit is {MAX_PAYLOAD_BYTES}",
                payload.len()
            )));
        }

        let header = json!({
            "protocol": PROTOCOL,
            "type": "evaluate",
            "request_id": request_id,
            "run_uuid": self.config.run_uuid,
            "step": step,
            "fragment_id": fragment_id,
            "base_version": baseline.base_version(),
            "state_epoch": baseline.base_state_epoch(),
            "fragment_versions": state.versions,
            "layout_hash": self.config.expected.layout_hash,
            "anchor_manifest_sha256": self.config.expected.anchor_manifest_sha256,
            "probe_config_sha256": self.config.expected.probe_config_sha256,
            "dtype": "f32le",
            "state": {
                "tensors": state_specs,
                "sha256": state_digest,
            },
            "fragment": {
                "action_family": action_family,
                "tensor_names": self.layout.fragment_names[fragment_id],
                "numel": expected_numel,
                "actions": action_specs,
            },
        });
        let header_bytes = canonical_json(&header).map_err(|error| {
            ProbeError::Protocol(format!("serialize action-probe request: {error:#}"))
        })?;
        if header_bytes.is_empty() || header_bytes.len() > MAX_HEADER_BYTES {
            return Err(ProbeError::Protocol(format!(
                "action-probe header has {} bytes, limit is {MAX_HEADER_BYTES}",
                header_bytes.len()
            )));
        }
        let frame = encode_frame_parts(&header_bytes, &payload).map_err(|error| {
            ProbeError::Protocol(format!("encode action-probe frame: {error:#}"))
        })?;
        let request_digest = sha256_hex(&frame);
        Ok(WireRequest {
            frame,
            request_id,
            request_digest,
            state_digest,
            action_digests,
            step,
            fragment_id,
            base_version: baseline.base_version(),
            state_epoch: baseline.base_state_epoch(),
            fragment_versions: state.versions.clone(),
            action_family,
        })
    }

    async fn roundtrip(&mut self, frame: &[u8]) -> std::result::Result<WireResponse, ProbeError> {
        if self.stream.is_none() {
            let stream = TcpStream::connect(self.config.endpoint)
                .await
                .map_err(|error| ProbeError::Io(error.to_string()))?;
            stream
                .set_nodelay(true)
                .map_err(|error| ProbeError::Io(error.to_string()))?;
            self.stream = Some(stream);
        }
        let stream = self.stream.as_mut().expect("stream initialized above");
        stream
            .write_all(frame)
            .await
            .map_err(|error| ProbeError::Io(error.to_string()))?;

        let mut prefix = [0u8; FRAME_PREFIX_BYTES];
        stream
            .read_exact(&mut prefix)
            .await
            .map_err(|error| ProbeError::Io(error.to_string()))?;
        if &prefix[..8] != FRAME_MAGIC {
            return Err(ProbeError::Protocol("bad response frame magic".to_owned()));
        }
        let header_len = u32::from_be_bytes(prefix[8..12].try_into().unwrap()) as usize;
        let payload_len = u64::from_be_bytes(prefix[12..20].try_into().unwrap());
        if header_len == 0 || header_len > MAX_HEADER_BYTES {
            return Err(ProbeError::Protocol(format!(
                "invalid response header length {header_len}"
            )));
        }
        if payload_len > MAX_PAYLOAD_BYTES as u64 {
            return Err(ProbeError::Protocol(format!(
                "invalid response payload length {payload_len}"
            )));
        }
        let payload_len = usize::try_from(payload_len)
            .map_err(|_| ProbeError::Protocol("response payload length overflow".to_owned()))?;
        let mut header = vec![0u8; header_len];
        stream
            .read_exact(&mut header)
            .await
            .map_err(|error| ProbeError::Io(error.to_string()))?;
        let mut payload = vec![0u8; payload_len];
        if payload_len > 0 {
            stream
                .read_exact(&mut payload)
                .await
                .map_err(|error| ProbeError::Io(error.to_string()))?;
        }
        Ok(WireResponse { header, payload })
    }
}

struct WireRequest {
    frame: Vec<u8>,
    request_id: String,
    request_digest: String,
    state_digest: String,
    action_digests: Vec<String>,
    step: u64,
    fragment_id: usize,
    base_version: u64,
    state_epoch: u64,
    fragment_versions: Vec<u64>,
    action_family: &'static str,
}

struct WireCttnRequest {
    frame: Vec<u8>,
    request_id: String,
    request_digest: String,
    state_digest: String,
    g_digest: String,
    b_digest: String,
    g: Vec<f32>,
    step: u64,
    fragment_id: usize,
    base_version: u64,
    state_epoch: u64,
    fragment_versions: Vec<u64>,
}

struct WireResponse {
    header: Vec<u8>,
    payload: Vec<u8>,
}

fn verify_cttn_response(
    response: &WireResponse,
    request: &WireCttnRequest,
    config: &ClientConfig,
) -> std::result::Result<VerifiedCttn, ProbeError> {
    let root = strict_json(&response.header)
        .map_err(|error| ProbeError::Protocol(format!("invalid response JSON: {error:#}")))?;
    let object = root
        .as_object()
        .ok_or_else(|| ProbeError::Protocol("response JSON must be an object".to_owned()))?;
    require_response_string(object, "protocol", PROTOCOL)?;
    require_response_string(object, "type", "cttn_result")?;
    require_response_string(object, "request_id", &request.request_id)?;
    require_response_string(object, "run_uuid", &config.run_uuid)?;
    require_response_string(object, "request_digest", &request.request_digest)?;
    let ok = object
        .get("ok")
        .and_then(Value::as_bool)
        .ok_or_else(|| ProbeError::Protocol("response ok must be boolean".to_owned()))?;
    if !ok {
        let reason = object
            .get("fallback_reason")
            .and_then(Value::as_str)
            .unwrap_or("remote_fail_closed");
        let error = object.get("error").and_then(Value::as_str).unwrap_or("");
        return Err(ProbeError::Remote(format!("{reason}: {error}")));
    }

    require_response_string(object, "dtype", "f32le")?;
    require_u64(object, "step", request.step)?;
    require_u64(object, "fragment_id", request.fragment_id as u64)?;
    require_u64(object, "base_version", request.base_version)?;
    require_u64(object, "state_epoch", request.state_epoch)?;
    let versions = object
        .get("fragment_versions")
        .and_then(Value::as_array)
        .ok_or_else(|| ProbeError::Protocol("fragment_versions must be a list".to_owned()))?;
    if versions.len() != request.fragment_versions.len()
        || versions
            .iter()
            .zip(&request.fragment_versions)
            .any(|(actual, expected)| actual.as_u64() != Some(*expected))
    {
        return Err(ProbeError::UnsafeResponse(
            "response fragment_versions do not match the request".to_owned(),
        ));
    }

    let (d, next_offset) =
        parse_response_f32(&response.payload, object.get("d"), 0, request.g.len(), "d")?;
    let (b_new, final_offset) = parse_response_f32(
        &response.payload,
        object.get("b_new"),
        next_offset,
        request.g.len(),
        "b_new",
    )?;
    if final_offset != response.payload.len() {
        return Err(ProbeError::UnsafeResponse(format!(
            "CTTN response has {} unclaimed payload bytes",
            response.payload.len() - final_offset
        )));
    }

    let digests = object
        .get("digests")
        .and_then(Value::as_object)
        .ok_or_else(|| ProbeError::UnsafeResponse("missing response digests".to_owned()))?;
    require_response_string(digests, "state_sha256", &request.state_digest)?;
    require_response_string(digests, "g_sha256", &request.g_digest)?;
    require_response_string(digests, "b_sha256", &request.b_digest)?;
    require_response_string(
        digests,
        "anchor_manifest_sha256",
        &config.expected.anchor_manifest_sha256,
    )?;
    require_response_string(
        digests,
        "anchor_tensors_sha256",
        &config.expected.anchor_tensors_sha256,
    )?;
    require_response_string(
        digests,
        "probe_config_sha256",
        &config.expected.probe_config_sha256,
    )?;
    require_response_string(digests, "layout_hash", &config.expected.layout_hash)?;

    let diagnostics_object = object
        .get("diagnostics")
        .and_then(Value::as_object)
        .ok_or_else(|| ProbeError::Protocol("diagnostics must be an object".to_owned()))?;
    let finite = |field: &str| -> std::result::Result<f64, ProbeError> {
        diagnostics_object
            .get(field)
            .and_then(Value::as_f64)
            .filter(|value| value.is_finite())
            .ok_or_else(|| ProbeError::Protocol(format!("diagnostics.{field} must be finite")))
    };
    let diagnostics = CttnDiagnostics {
        bind: diagnostics_object
            .get("bind")
            .and_then(Value::as_bool)
            .ok_or_else(|| ProbeError::Protocol("diagnostics.bind must be boolean".to_owned()))?,
        tau: finite("tau")?,
        retention: finite("retention")?,
        e_before: finite("e_before")?,
        e_after: finite("e_after")?,
        budget: finite("budget")?,
        n_modes_90: diagnostics_object
            .get("n_modes_90")
            .and_then(Value::as_u64)
            .ok_or_else(|| {
                ProbeError::Protocol("diagnostics.n_modes_90 must be non-negative".to_owned())
            })?,
        ritz_max: finite("ritz_max")?,
        loss: finite("loss")?,
    };
    if diagnostics.tau < 0.0
        || diagnostics.retention < 0.0
        || diagnostics.e_before < 0.0
        || diagnostics.e_after < 0.0
        || diagnostics.budget < 0.0
        || diagnostics.ritz_max < 0.0
    {
        return Err(ProbeError::UnsafeResponse(
            "CTTN diagnostics contain a negative norm, energy, budget, or curvature".to_owned(),
        ));
    }

    let gnorm_sq = request
        .g
        .iter()
        .map(|value| {
            let value = *value as f64;
            value * value
        })
        .sum::<f64>();
    if gnorm_sq > 0.0 {
        let gnorm = gnorm_sq.sqrt();
        let q_dot_d = request
            .g
            .iter()
            .zip(&d)
            .map(|(g, d)| *g as f64 * *d as f64)
            .sum::<f64>()
            / gnorm;
        let tolerance = 1e-4 * gnorm.max(1e-12);
        if !q_dot_d.is_finite() || (q_dot_d - gnorm).abs() > tolerance {
            return Err(ProbeError::UnsafeResponse(format!(
                "CTTN parallel-step invariant failed: q.d={q_dot_d}, ||g||={gnorm}, tolerance={tolerance}"
            )));
        }
    }

    Ok(VerifiedCttn {
        d,
        b_new,
        diagnostics,
        request_digest: request.request_digest.clone(),
    })
}

fn parse_response_f32(
    payload: &[u8],
    spec: Option<&Value>,
    expected_offset: usize,
    expected_numel: usize,
    name: &str,
) -> std::result::Result<(Vec<f32>, usize), ProbeError> {
    let spec = spec.and_then(Value::as_object).ok_or_else(|| {
        ProbeError::Protocol(format!("response {name} descriptor must be an object"))
    })?;
    let offset = spec
        .get("offset")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| ProbeError::Protocol(format!("response {name}.offset is invalid")))?;
    let nbytes = spec
        .get("nbytes")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| *value > 0 && *value % 4 == 0)
        .ok_or_else(|| ProbeError::Protocol(format!("response {name}.nbytes is invalid")))?;
    if offset != expected_offset {
        return Err(ProbeError::UnsafeResponse(format!(
            "response {name} starts at {offset}, expected {expected_offset}"
        )));
    }
    let end = offset
        .checked_add(nbytes)
        .ok_or_else(|| ProbeError::Protocol(format!("response {name} payload range overflows")))?;
    if end > payload.len() || nbytes / 4 != expected_numel {
        return Err(ProbeError::UnsafeResponse(format!(
            "response {name} payload length does not match the fragment"
        )));
    }
    let raw = &payload[offset..end];
    let expected_digest = spec
        .get("sha256")
        .and_then(Value::as_str)
        .filter(|value| value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .ok_or_else(|| ProbeError::Protocol(format!("response {name}.sha256 is invalid")))?;
    if sha256_hex(raw) != expected_digest.to_ascii_lowercase() {
        return Err(ProbeError::UnsafeResponse(format!(
            "response {name} SHA-256 mismatch"
        )));
    }
    let mut values = Vec::with_capacity(expected_numel);
    for chunk in raw.chunks_exact(4) {
        let value = f32::from_le_bytes(chunk.try_into().expect("four-byte f32 chunk"));
        if !value.is_finite() {
            return Err(ProbeError::UnsafeResponse(format!(
                "response {name} contains NaN or Inf"
            )));
        }
        values.push(value);
    }
    Ok((values, end))
}

fn verify_response(
    response: &WireResponse,
    request: &WireRequest,
    config: &ClientConfig,
    previews: &RetainedPreviews,
) -> std::result::Result<VerifiedSelection, ProbeError> {
    if !response.payload.is_empty() {
        return Err(ProbeError::UnsafeResponse(
            "evaluate_result must have an empty binary payload".to_owned(),
        ));
    }
    let root = strict_json(&response.header)
        .map_err(|error| ProbeError::Protocol(format!("invalid response JSON: {error:#}")))?;
    let object = root
        .as_object()
        .ok_or_else(|| ProbeError::Protocol("response JSON must be an object".to_owned()))?;
    require_response_string(object, "protocol", PROTOCOL)?;
    require_response_string(object, "type", "evaluate_result")?;
    require_response_string(object, "request_id", &request.request_id)?;
    require_response_string(object, "run_uuid", &config.run_uuid)?;
    require_response_string(object, "request_digest", &request.request_digest)?;
    require_response_string(object, "action_family", request.action_family)?;

    let ok = object
        .get("ok")
        .and_then(Value::as_bool)
        .ok_or_else(|| ProbeError::Protocol("response ok must be boolean".to_owned()))?;
    if !ok {
        let reason = object
            .get("fallback_reason")
            .and_then(Value::as_str)
            .unwrap_or("remote_fail_closed");
        let error = object.get("error").and_then(Value::as_str).unwrap_or("");
        return Err(ProbeError::Remote(format!("{reason}: {error}")));
    }

    require_u64(object, "step", request.step)?;
    require_u64(object, "fragment_id", request.fragment_id as u64)?;
    require_u64(object, "base_version", request.base_version)?;
    require_u64(object, "state_epoch", request.state_epoch)?;
    let versions = object
        .get("fragment_versions")
        .and_then(Value::as_array)
        .ok_or_else(|| ProbeError::Protocol("fragment_versions must be a list".to_owned()))?;
    if versions.len() != request.fragment_versions.len()
        || versions
            .iter()
            .zip(&request.fragment_versions)
            .any(|(actual, expected)| actual.as_u64() != Some(*expected))
    {
        return Err(ProbeError::UnsafeResponse(
            "response fragment_versions do not match the request".to_owned(),
        ));
    }

    let selected_action = object
        .get("selected_action")
        .and_then(Value::as_str)
        .ok_or_else(|| ProbeError::Protocol("selected_action must be a string".to_owned()))?;
    let action_index = ACTION_NAMES
        .iter()
        .position(|name| *name == selected_action)
        .ok_or_else(|| ProbeError::UnsafeResponse("unknown selected_action".to_owned()))?;
    let selected_digest = object
        .get("selected_action_sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            ProbeError::UnsafeResponse(
                "successful response lacks selected action SHA-256".to_owned(),
            )
        })?;
    if selected_digest != request.action_digests[action_index] {
        return Err(ProbeError::UnsafeResponse(
            "selected action SHA-256 does not match local trial bytes".to_owned(),
        ));
    }
    if action_index > 0 && !previews.metadata(action_index).eligible {
        return Err(ProbeError::UnsafeResponse(format!(
            "service selected locally ineligible {selected_action}"
        )));
    }

    let fail_closed = object
        .get("fail_closed")
        .and_then(Value::as_bool)
        .ok_or_else(|| ProbeError::Protocol("fail_closed must be boolean".to_owned()))?;
    if fail_closed != (action_index == 0) {
        return Err(ProbeError::UnsafeResponse(
            "fail_closed is inconsistent with selected_action".to_owned(),
        ));
    }
    verify_selected_metadata(
        object
            .get("selected_action_metadata")
            .and_then(Value::as_object)
            .ok_or_else(|| {
                ProbeError::UnsafeResponse("missing selected_action_metadata".to_owned())
            })?,
        previews.metadata(action_index),
        request.action_family,
    )?;

    let digests = object
        .get("digests")
        .and_then(Value::as_object)
        .ok_or_else(|| ProbeError::UnsafeResponse("missing response digests".to_owned()))?;
    require_response_string(digests, "state_sha256", &request.state_digest)?;
    require_response_string(
        digests,
        "anchor_manifest_sha256",
        &config.expected.anchor_manifest_sha256,
    )?;
    require_response_string(
        digests,
        "anchor_tensors_sha256",
        &config.expected.anchor_tensors_sha256,
    )?;
    require_response_string(
        digests,
        "probe_config_sha256",
        &config.expected.probe_config_sha256,
    )?;
    require_response_string(digests, "layout_hash", &config.expected.layout_hash)?;
    let action_digests = digests
        .get("action_sha256")
        .and_then(Value::as_object)
        .ok_or_else(|| ProbeError::UnsafeResponse("missing action digest map".to_owned()))?;
    for (index, name) in ACTION_NAMES.iter().enumerate() {
        require_response_string(action_digests, name, &request.action_digests[index])?;
    }

    let fallback_reason = match object.get("fallback_reason") {
        None | Some(Value::Null) => None,
        Some(Value::String(value)) if value.len() <= 256 => Some(value.clone()),
        _ => {
            return Err(ProbeError::Protocol(
                "fallback_reason must be null or a short string".to_owned(),
            ))
        }
    };
    if action_index > 0 && fallback_reason.is_some() {
        return Err(ProbeError::UnsafeResponse(
            "alternative action carries a fallback reason".to_owned(),
        ));
    }
    Ok(VerifiedSelection {
        action_index,
        action_name: selected_action.to_owned(),
        fallback_reason,
        request_digest: request.request_digest.clone(),
    })
}

fn verify_selected_metadata(
    actual: &Map<String, Value>,
    expected: &ActionMetadata,
    action_family: &str,
) -> std::result::Result<(), ProbeError> {
    if actual.get("eligible").and_then(Value::as_bool) != Some(expected.eligible) {
        return Err(ProbeError::UnsafeResponse(
            "selected eligibility changed in the response".to_owned(),
        ));
    }
    let omitted = match actual.get("omitted_responder_id") {
        Some(Value::Null) => None,
        Some(value) => value.as_u64().and_then(|value| u32::try_from(value).ok()),
        None => None,
    };
    if omitted != expected.omitted_responder_id() {
        return Err(ProbeError::UnsafeResponse(
            "selected omitted responder changed in the response".to_owned(),
        ));
    }
    for (field, expected_value) in [
        ("selected_mass", expected.selected_mass),
        ("norm_multiplier", expected.norm_multiplier),
        ("step_norm_ratio", expected.step_norm_ratio),
    ] {
        let actual_value = actual
            .get(field)
            .and_then(Value::as_f64)
            .filter(|value| value.is_finite())
            .ok_or_else(|| ProbeError::Protocol(format!("{field} must be finite")))?;
        let tolerance = 1e-12 * expected_value.abs().max(1.0);
        if (actual_value - expected_value).abs() > tolerance {
            return Err(ProbeError::UnsafeResponse(format!(
                "selected {field} changed in the response"
            )));
        }
    }
    let reason = match actual.get("ineligible_reason") {
        None | Some(Value::Null) => None,
        Some(Value::String(value)) => Some(value.as_str()),
        _ => {
            return Err(ProbeError::Protocol(
                "ineligible_reason must be null or a string".to_owned(),
            ))
        }
    };
    if reason != expected.ineligible_reason.as_deref() {
        return Err(ProbeError::UnsafeResponse(
            "selected ineligible reason changed in the response".to_owned(),
        ));
    }
    match (action_family, expected.step_scale) {
        (STEP_SCALE_ACTION_FAMILY, Some(expected_value)) => {
            let actual_value = actual
                .get("step_scale")
                .and_then(Value::as_f64)
                .filter(|value| value.is_finite() && *value > 0.0)
                .ok_or_else(|| {
                    ProbeError::Protocol(
                        "selected step_scale must be positive and finite".to_owned(),
                    )
                })?;
            if actual_value.to_bits() != expected_value.to_bits() {
                return Err(ProbeError::UnsafeResponse(
                    "selected step_scale changed in the response".to_owned(),
                ));
            }
        }
        (LEAVE_ONE_OUT_ACTION_FAMILY, None) if !actual.contains_key("step_scale") => {}
        _ => {
            return Err(ProbeError::UnsafeResponse(
                "selected action metadata does not match its action family".to_owned(),
            ))
        }
    }
    Ok(())
}

fn require_response_string(
    object: &Map<String, Value>,
    field: &str,
    expected: &str,
) -> std::result::Result<(), ProbeError> {
    let actual = object
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| ProbeError::Protocol(format!("response {field} must be a string")))?;
    if actual != expected {
        return Err(ProbeError::UnsafeResponse(format!(
            "response {field} does not match the request"
        )));
    }
    Ok(())
}

fn require_u64(
    object: &Map<String, Value>,
    field: &str,
    expected: u64,
) -> std::result::Result<(), ProbeError> {
    if object.get(field).and_then(Value::as_u64) != Some(expected) {
        return Err(ProbeError::UnsafeResponse(format!(
            "response {field} does not match the request"
        )));
    }
    Ok(())
}

fn string_field<'a>(object: &'a Map<String, Value>, field: &str) -> Result<&'a str> {
    object
        .get(field)
        .and_then(Value::as_str)
        .with_context(|| format!("{field} must be a string"))
}

fn usize_field(object: &Map<String, Value>, field: &str) -> Result<usize> {
    object
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .with_context(|| format!("{field} must be a nonnegative integer fitting usize"))
}

fn digest_field(object: &Map<String, Value>, field: &str) -> Result<String> {
    let value = string_field(object, field)?;
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{field} must be a 64-character hexadecimal SHA-256");
    }
    Ok(value.to_ascii_lowercase())
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut raw = Vec::with_capacity(values.len() * 4);
    for value in values {
        raw.extend_from_slice(&value.to_le_bytes());
    }
    raw
}

fn encode_frame_parts(header: &[u8], payload: &[u8]) -> Result<Vec<u8>> {
    let header_len = u32::try_from(header.len()).context("action-probe header exceeds u32")?;
    let payload_len = u64::try_from(payload.len()).context("action-probe payload exceeds u64")?;
    let mut frame = Vec::with_capacity(FRAME_PREFIX_BYTES + header.len() + payload.len());
    frame.extend_from_slice(FRAME_MAGIC);
    frame.extend_from_slice(&header_len.to_be_bytes());
    frame.extend_from_slice(&payload_len.to_be_bytes());
    frame.extend_from_slice(header);
    frame.extend_from_slice(payload);
    Ok(frame)
}

fn canonical_json(value: &Value) -> Result<Vec<u8>> {
    // serde_json::Map is key-sorted unless its preserve_order feature is
    // enabled. Rebuilding recursively through BTreeMap keeps this canonical
    // even if that feature is enabled elsewhere in the dependency graph.
    fn sorted(value: &Value) -> Value {
        match value {
            Value::Object(object) => {
                let ordered = object
                    .iter()
                    .map(|(key, value)| (key.clone(), sorted(value)))
                    .collect::<BTreeMap<_, _>>();
                Value::Object(ordered.into_iter().collect())
            }
            Value::Array(values) => Value::Array(values.iter().map(sorted).collect()),
            _ => value.clone(),
        }
    }
    Ok(serde_json::to_vec(&sorted(value))?)
}

fn sha256_hex(bytes: &[u8]) -> String {
    digest_to_hex(&Sha256::digest(bytes))
}

fn digest_to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

struct StrictValue(Value);

impl<'de> Deserialize<'de> for StrictValue {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(StrictValueVisitor)
    }
}

struct StrictValueVisitor;

impl<'de> Visitor<'de> for StrictValueVisitor {
    type Value = StrictValue;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a finite JSON value without duplicate object keys")
    }

    fn visit_bool<E>(self, value: bool) -> std::result::Result<Self::Value, E> {
        Ok(StrictValue(Value::Bool(value)))
    }

    fn visit_i64<E>(self, value: i64) -> std::result::Result<Self::Value, E> {
        Ok(StrictValue(Value::Number(value.into())))
    }

    fn visit_u64<E>(self, value: u64) -> std::result::Result<Self::Value, E> {
        Ok(StrictValue(Value::Number(value.into())))
    }

    fn visit_f64<E>(self, value: f64) -> std::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        let number = serde_json::Number::from_f64(value)
            .ok_or_else(|| E::custom("non-finite JSON number"))?;
        Ok(StrictValue(Value::Number(number)))
    }

    fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(StrictValue(Value::String(value.to_owned())))
    }

    fn visit_string<E>(self, value: String) -> std::result::Result<Self::Value, E> {
        Ok(StrictValue(Value::String(value)))
    }

    fn visit_none<E>(self) -> std::result::Result<Self::Value, E> {
        Ok(StrictValue(Value::Null))
    }

    fn visit_unit<E>(self) -> std::result::Result<Self::Value, E> {
        Ok(StrictValue(Value::Null))
    }

    fn visit_some<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        StrictValue::deserialize(deserializer)
    }

    fn visit_seq<A>(self, mut sequence: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(StrictValue(value)) = sequence.next_element::<StrictValue>()? {
            values.push(value);
        }
        Ok(StrictValue(Value::Array(values)))
    }

    fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut object = Map::new();
        while let Some(key) = map.next_key::<String>()? {
            if object.contains_key(&key) {
                return Err(de::Error::custom(format!("duplicate JSON key {key:?}")));
            }
            let StrictValue(value) = map.next_value::<StrictValue>()?;
            object.insert(key, value);
        }
        Ok(StrictValue(Value::Object(object)))
    }
}

fn strict_json(raw: &[u8]) -> Result<Value> {
    let mut deserializer = serde_json::Deserializer::from_slice(raw);
    let StrictValue(value) = StrictValue::deserialize(&mut deserializer)?;
    deserializer.end()?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{FragmentInfo, Layout, MERGE_AVG};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use tokio::net::TcpListener;

    fn state_and_candidates() -> (GlobalState, Vec<Vec<f32>>) {
        let layout = Layout {
            fragments: vec![
                FragmentInfo {
                    merge_mode: MERGE_AVG,
                    tensor_numels: vec![4],
                    tensor_shapes: None,
                },
                FragmentInfo {
                    merge_mode: MERGE_AVG,
                    tensor_numels: vec![4],
                    tensor_shapes: None,
                },
            ],
        };
        let mut state = GlobalState::new(layout, None, 0.2, 0.0, 1);
        state.init_fragment(0, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        state.init_fragment(1, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        state.versions = vec![3, 4];
        let candidates = vec![
            vec![0.8, 1.8, 2.8, 3.8],
            vec![0.7, 1.7, 2.7, 3.7],
            vec![0.9, 1.9, 2.9, 3.9],
            vec![0.6, 1.6, 2.6, 3.6],
        ];
        (state, candidates)
    }

    fn expected_for(state: &GlobalState) -> ExpectedProbeConfig {
        let mut expected = ExpectedProbeConfig {
            anchor_manifest_sha256: "a".repeat(64),
            anchor_tensors_sha256: "d".repeat(64),
            probe_config_sha256: "b".repeat(64),
            layout_hash: String::new(),
            fragment_pattern: "binpack".to_owned(),
            lora_r: 2,
            fragment_names: vec![
                vec!["model.q_proj.lora_A.default.weight".to_owned()],
                vec!["model.q_proj.lora_B.default.weight".to_owned()],
            ],
            tensor_shapes: BTreeMap::new(),
        };
        let mut bindings = Vec::new();
        for (fragment_id, name) in expected.fragment_names.iter().flatten().enumerate() {
            bindings.push(TensorBinding {
                name: name.clone(),
                shape: vec![2, 2],
                fragment_id,
                offset: 0,
                numel: 4,
            });
        }
        expected.layout_hash = layout_contract_digest("binpack", &state.layout, &bindings).unwrap();
        expected
    }

    fn previews(state: &GlobalState, values: &[Vec<f32>]) -> RetainedPreviews {
        let candidates = values
            .iter()
            .enumerate()
            .map(|(index, values)| MergeCandidate::new(index as u32, values, 1.0))
            .collect::<Vec<_>>();
        let baseline = build_baseline_preview(state, 0, 5, &candidates).unwrap();
        let alternatives =
            build_leave_one_out_previews(state, 0, 5, &candidates, &baseline).unwrap();
        RetainedPreviews::loo_v1(baseline, alternatives).unwrap()
    }

    fn scaled_previews(state: &GlobalState, values: &[Vec<f32>]) -> RetainedPreviews {
        let candidates = values
            .iter()
            .enumerate()
            .map(|(index, values)| MergeCandidate::new(index as u32, values, 1.0))
            .collect::<Vec<_>>();
        let baseline = build_baseline_preview(state, 0, 5, &candidates).unwrap();
        build_scaled_full_group_previews(state, &baseline, &LR_PREVIEW_MULTIPLIERS).unwrap()
    }

    async fn read_test_frame(stream: &mut TcpStream) -> Vec<u8> {
        let mut prefix = [0u8; FRAME_PREFIX_BYTES];
        stream.read_exact(&mut prefix).await.unwrap();
        let header_len = u32::from_be_bytes(prefix[8..12].try_into().unwrap()) as usize;
        let payload_len = u64::from_be_bytes(prefix[12..20].try_into().unwrap()) as usize;
        let mut body = vec![0u8; header_len + payload_len];
        stream.read_exact(&mut body).await.unwrap();
        let mut frame = prefix.to_vec();
        frame.extend_from_slice(&body);
        frame
    }

    fn response_for_request(frame: &[u8], selected_index: usize) -> Vec<u8> {
        let header_len = u32::from_be_bytes(frame[8..12].try_into().unwrap()) as usize;
        let request = strict_json(&frame[20..20 + header_len]).unwrap();
        let actions = request["fragment"]["actions"].as_array().unwrap();
        let selected = actions[selected_index].as_object().unwrap();
        let mut selected_metadata = json!({
            "eligible": selected["eligible"],
            "omitted_responder_id": selected["omitted_responder_id"],
            "selected_mass": selected["selected_mass"],
            "norm_multiplier": selected["norm_multiplier"],
            "step_norm_ratio": selected["step_norm_ratio"],
            "ineligible_reason": selected["ineligible_reason"],
        });
        if let Some(step_scale) = selected.get("step_scale") {
            selected_metadata["step_scale"] = step_scale.clone();
        }
        let action_sha256 = actions
            .iter()
            .map(|action| {
                (
                    action["name"].as_str().unwrap().to_owned(),
                    action["sha256"].clone(),
                )
            })
            .collect::<Map<_, _>>();
        let response = json!({
            "protocol": PROTOCOL,
            "type": "evaluate_result",
            "request_id": request["request_id"],
            "run_uuid": request["run_uuid"],
            "step": request["step"],
            "fragment_id": request["fragment_id"],
            "action_family": request["fragment"]["action_family"],
            "base_version": request["base_version"],
            "state_epoch": request["state_epoch"],
            "fragment_versions": request["fragment_versions"],
            "request_digest": sha256_hex(frame),
            "ok": true,
            "fail_closed": selected_index == 0,
            "selected_action": selected["name"],
            "selected_action_sha256": selected["sha256"],
            "selected_action_metadata": selected_metadata,
            "fallback_reason": if selected_index == 0 {
                Value::String("no_action_passed".to_owned())
            } else {
                Value::Null
            },
            "digests": {
                "state_sha256": request["state"]["sha256"],
                "action_sha256": Value::Object(action_sha256),
                "anchor_manifest_sha256": request["anchor_manifest_sha256"],
                "anchor_tensors_sha256": "d".repeat(64),
                "probe_config_sha256": request["probe_config_sha256"],
                "layout_hash": request["layout_hash"],
            },
        });
        let header = canonical_json(&response).unwrap();
        encode_frame_parts(&header, &[]).unwrap()
    }

    #[test]
    fn commit_policy_names_accept_exact_and_cli_friendly_spellings() {
        assert_eq!("token_weighted".parse(), Ok(CommitPolicy::TokenWeighted));
        assert_eq!("probe-shadow".parse(), Ok(CommitPolicy::ProbeShadow));
        assert_eq!(CommitPolicy::ProbeLooV1.to_string(), "probe_loo_v1");
        assert_eq!("probe-lr-shadow".parse(), Ok(CommitPolicy::ProbeLrShadow));
        assert_eq!("probe_lr_v1".parse(), Ok(CommitPolicy::ProbeLrV1));
        assert_eq!("cttn-v1".parse(), Ok(CommitPolicy::ProbeCttnV1));
        assert!(CommitPolicy::ProbeLrShadow.is_shadow());
        assert!(!CommitPolicy::ProbeCttnV1.is_shadow());
        assert!(!CommitPolicy::ProbeCttnV1.is_leave_one_out());
        assert_eq!(
            CommitPolicy::ProbeLrV1.step_scale_multipliers(),
            Some(&LR_PREVIEW_MULTIPLIERS)
        );
        assert!("probe".parse::<CommitPolicy>().is_err());
    }

    #[test]
    fn strict_json_rejects_duplicate_keys() {
        assert!(strict_json(br#"{"ok":true,"ok":false}"#).is_err());
    }

    #[test]
    fn layout_binding_infers_lora_shapes_and_verifies_sidecar_hash() {
        let (state, _) = state_and_candidates();
        let expected = expected_for(&state);
        assert_eq!(
            expected.layout_hash,
            "b74af0ab4b118be75e536fccf374de367814e3064c4b6f3e3b56b7e5eaaa50c2"
        );
        let bound = BoundLayout::bind(&expected, &state).unwrap();
        assert_eq!(bound.state_tensors[0].shape, vec![2, 2]);
        assert_eq!(
            bound.fragment_names[1][0],
            "model.q_proj.lora_B.default.weight"
        );

        let mut stale = expected;
        stale.layout_hash = "0".repeat(64);
        assert!(BoundLayout::bind(&stale, &state).is_err());
    }

    #[test]
    fn loo_previews_are_deterministic_norm_matched_and_mass_checked() {
        let (state, values) = state_and_candidates();
        let set = previews(&state, &values);
        let baseline_norm = set.preview(0).stats().outer.applied_step_norm;
        for index in 1..5 {
            let metadata = set.metadata(index);
            assert_eq!(metadata.omitted_responder_id(), Some((index - 1) as u32));
            assert_eq!(metadata.selected_mass, 0.75);
            assert!((metadata.step_norm_ratio - 1.0).abs() <= 0.01);
            assert!(
                (set.preview(index).stats().outer.applied_step_norm - baseline_norm).abs() < 1e-6
            );
        }
    }

    #[test]
    fn generic_retained_previews_use_the_exact_predeclared_lr_action_order() {
        let (state, values) = state_and_candidates();
        let mut grid = scaled_previews(&state, &values);

        assert_eq!(grid.len(), LR_PREVIEW_MULTIPLIERS.len());
        assert_eq!(
            (0..grid.len())
                .map(|index| grid.name(index))
                .collect::<Vec<_>>(),
            ACTION_NAMES
        );
        for (index, multiplier) in LR_PREVIEW_MULTIPLIERS.iter().copied().enumerate() {
            assert_eq!(
                grid.metadata(index).kind,
                RetainedActionKind::ScaledFullGroup { multiplier }
            );
            assert_eq!(grid.metadata(index).selected_mass, 1.0);
            assert_eq!(grid.metadata(index).step_scale, Some(multiplier));
            assert_eq!(grid.preview(index).step_scale(), multiplier);
        }
        let retained = grid.take(4);
        assert_eq!(retained.step_scale(), 1.5);
    }

    #[test]
    fn request_frame_uses_yetoap01_and_exact_local_digests() {
        let (state, values) = state_and_candidates();
        let expected = expected_for(&state);
        let config = ClientConfig {
            endpoint: "127.0.0.1:1".parse().unwrap(),
            timeout: Duration::from_secs(1),
            run_uuid: "run-test".to_owned(),
            expected,
        };
        let client = ActionProbeClient::bind(config, &state).unwrap();
        let set = previews(&state, &values);
        let request = client.build_request(&state, &set, 5, 0).unwrap();
        assert_eq!(&request.frame[..8], FRAME_MAGIC);
        assert_eq!(request.request_digest, sha256_hex(&request.frame));
        assert_eq!(request.action_digests.len(), 5);
        let header_len = u32::from_be_bytes(request.frame[8..12].try_into().unwrap()) as usize;
        let header = strict_json(&request.frame[20..20 + header_len]).unwrap();
        assert_eq!(
            header["fragment"]["action_family"],
            LEAVE_ONE_OUT_ACTION_FAMILY
        );
        assert_eq!(header["fragment"]["actions"][1]["name"], "A1");
        assert_eq!(header["state"]["sha256"], request.state_digest);
    }

    #[test]
    fn cttn_request_uses_fragment_order_and_state_then_g_then_b_payload() {
        let (state, values) = state_and_candidates();
        let config = ClientConfig {
            endpoint: "127.0.0.1:1".parse().unwrap(),
            timeout: Duration::from_secs(1),
            run_uuid: "run-cttn".to_owned(),
            expected: expected_for(&state),
        };
        let client = ActionProbeClient::bind(config, &state).unwrap();
        let candidates = values
            .iter()
            .enumerate()
            .map(|(index, values)| MergeCandidate::new(index as u32, values, 1.0))
            .collect::<Vec<_>>();
        let aggregate = state.build_full_aggregate(0, &candidates).unwrap();
        let inputs = state.cttn_inputs(&aggregate, 0.9).unwrap();
        let request = client
            .build_cttn_request(
                &state,
                &aggregate,
                5,
                &inputs.g,
                &inputs.b,
                inputs.mu,
                0.1,
                4,
            )
            .unwrap();
        let header_len = u32::from_be_bytes(request.frame[8..12].try_into().unwrap()) as usize;
        let header = strict_json(&request.frame[20..20 + header_len]).unwrap();
        let state_bytes: usize = header["state"]["tensors"]
            .as_array()
            .unwrap()
            .iter()
            .map(|spec| spec["nbytes"].as_u64().unwrap() as usize)
            .sum();
        assert_eq!(header["type"], "cttn_step");
        assert_eq!(header["fragment"]["tensor_names"], json!(client.layout.fragment_names[0]));
        assert_eq!(header["cttn"]["g"]["offset"], state_bytes);
        assert_eq!(
            header["cttn"]["b"]["offset"],
            state_bytes + inputs.g.len() * 4
        );
        assert_eq!(header["cttn"]["mu"], json!(inputs.mu));
        assert_eq!(header["cttn"]["rho"], json!(0.1f32));
        assert_eq!(header["cttn"]["block_steps"], 4);
    }

    #[test]
    fn scalar_request_frame_carries_exact_step_scale_contract() {
        let (state, values) = state_and_candidates();
        let config = ClientConfig {
            endpoint: "127.0.0.1:1".parse().unwrap(),
            timeout: Duration::from_secs(1),
            run_uuid: "run-scalars".to_owned(),
            expected: expected_for(&state),
        };
        let client = ActionProbeClient::bind(config, &state).unwrap();
        let set = scaled_previews(&state, &values);
        let request = client.build_request(&state, &set, 5, 0).unwrap();
        let header_len = u32::from_be_bytes(request.frame[8..12].try_into().unwrap()) as usize;
        let header = strict_json(&request.frame[20..20 + header_len]).unwrap();
        let fragment = header["fragment"].as_object().unwrap();
        assert_eq!(fragment["action_family"], STEP_SCALE_ACTION_FAMILY);
        let actions = fragment["actions"].as_array().unwrap();
        for (index, ((action, expected_name), expected_scale)) in actions
            .iter()
            .zip(ACTION_NAMES)
            .zip(LR_PREVIEW_MULTIPLIERS)
            .enumerate()
        {
            assert_eq!(action["name"], expected_name, "action {index}");
            assert_eq!(action["step_scale"], expected_scale, "action {index}");
            assert_eq!(action["omitted_responder_id"], Value::Null);
            assert_eq!(action["selected_mass"], 1.0);
        }
    }

    #[test]
    fn scalar_response_selects_the_exact_retained_preview_and_scale() {
        let (state, values) = state_and_candidates();
        let config = ClientConfig {
            endpoint: "127.0.0.1:1".parse().unwrap(),
            timeout: Duration::from_secs(1),
            run_uuid: "run-scalars".to_owned(),
            expected: expected_for(&state),
        };
        let client = ActionProbeClient::bind(config.clone(), &state).unwrap();
        let set = scaled_previews(&state, &values);
        let request = client.build_request(&state, &set, 5, 0).unwrap();
        let response_frame = response_for_request(&request.frame, 4);
        let header_len = u32::from_be_bytes(response_frame[8..12].try_into().unwrap()) as usize;
        let response = WireResponse {
            header: response_frame[20..20 + header_len].to_vec(),
            payload: Vec::new(),
        };

        let selected = verify_response(&response, &request, &config, &set).unwrap();
        assert_eq!(selected.action_index, 4);
        assert_eq!(selected.action_name, "A4");
        assert_eq!(set.metadata(selected.action_index).step_scale, Some(1.5));
        assert_eq!(set.preview(selected.action_index).step_scale(), 1.5);
    }

    #[tokio::test]
    async fn client_reuses_one_connection_and_verifies_exact_selected_bytes() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = listener.local_addr().unwrap();
        let accepts = Arc::new(AtomicUsize::new(0));
        let server_accepts = accepts.clone();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            server_accepts.fetch_add(1, Ordering::SeqCst);
            for _ in 0..2 {
                let request = read_test_frame(&mut stream).await;
                let response = response_for_request(&request, 1);
                stream.write_all(&response).await.unwrap();
            }
        });

        let (state, values) = state_and_candidates();
        let config = ClientConfig {
            endpoint,
            timeout: Duration::from_secs(1),
            run_uuid: "run-persistent".to_owned(),
            expected: expected_for(&state),
        };
        let mut client = ActionProbeClient::bind(config, &state).unwrap();
        let set = previews(&state, &values);
        for _ in 0..2 {
            let selection = client.select(&state, &set, 5, 0).await.unwrap();
            assert_eq!(selection.action_name, "A1");
            assert_eq!(
                selection.request_digest,
                client.last_request_digest().unwrap()
            );
        }
        server.await.unwrap();
        assert_eq!(accepts.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn hard_timeout_closes_connection_and_keeps_request_digest_for_tape() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let _request = read_test_frame(&mut stream).await;
            tokio::time::sleep(Duration::from_millis(100)).await;
        });

        let (state, values) = state_and_candidates();
        let config = ClientConfig {
            endpoint,
            timeout: Duration::from_millis(10),
            run_uuid: "run-timeout".to_owned(),
            expected: expected_for(&state),
        };
        let mut client = ActionProbeClient::bind(config, &state).unwrap();
        let set = previews(&state, &values);
        let error = client.select(&state, &set, 5, 0).await.unwrap_err();
        assert!(matches!(error, ProbeError::Timeout));
        assert!(client.last_request_digest().is_some());
        assert!(client.stream.is_none());
        server.await.unwrap();
    }

    #[test]
    fn selected_action_digest_mismatch_is_unsafe() {
        let (state, values) = state_and_candidates();
        let config = ClientConfig {
            endpoint: "127.0.0.1:1".parse().unwrap(),
            timeout: Duration::from_secs(1),
            run_uuid: "run-unsafe".to_owned(),
            expected: expected_for(&state),
        };
        let client = ActionProbeClient::bind(config.clone(), &state).unwrap();
        let set = previews(&state, &values);
        let request = client.build_request(&state, &set, 5, 0).unwrap();
        let response_frame = response_for_request(&request.frame, 1);
        let header_len = u32::from_be_bytes(response_frame[8..12].try_into().unwrap()) as usize;
        let mut response = strict_json(&response_frame[20..20 + header_len]).unwrap();
        response["selected_action_sha256"] = Value::String("0".repeat(64));
        let response = WireResponse {
            header: canonical_json(&response).unwrap(),
            payload: Vec::new(),
        };
        let error = verify_response(&response, &request, &config, &set).unwrap_err();
        assert!(matches!(error, ProbeError::UnsafeResponse(_)));
    }
}
