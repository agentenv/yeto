//! Opt-in pseudo-gradient autocorrelation telemetry.
//!
//! The production merge can contain hundreds of millions of f32 values, so
//! retaining four exact pseudo-gradients per fragment is not viable.  This
//! module instead applies a deterministic CountSketch random projection to
//! each tensor group and retains only the four most recent projected
//! pseudo-gradients for that fragment.  Exact L2 norms and exact pairwise
//! worker cosines are computed while the current vectors are resident.

use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::Serialize;

pub const RHO_TELEMETRY_SCHEMA: &str = "yeto_rho_telemetry_v1";
pub const PROJECTION_DIMENSION: usize = 4096;
pub const PROJECTION_SEED: u64 = 0x5945_544f_5248_4f31;
const MAX_LAG: usize = 4;

#[derive(Clone, Debug, PartialEq)]
struct GroupedSketch {
    groups: Vec<Vec<f32>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ProjectionSketcher {
    dimension: usize,
    seed: u64,
}

impl ProjectionSketcher {
    fn new(dimension: usize, seed: u64) -> Result<Self> {
        if dimension == 0 {
            bail!("rho telemetry projection dimension must be positive");
        }
        Ok(Self { dimension, seed })
    }

    /// Sparse signed random projection, independently keyed by tensor group.
    /// Each input coordinate contributes to one of `dimension` buckets with
    /// a deterministic random sign.  This is a CountSketch matrix without
    /// ever materializing the matrix itself.
    fn sketch(&self, tensor_numels: &[u64], values: &[f32]) -> Result<GroupedSketch> {
        let expected = tensor_numels.iter().try_fold(0usize, |total, &numel| {
            let numel =
                usize::try_from(numel).context("rho telemetry tensor size does not fit usize")?;
            total
                .checked_add(numel)
                .context("rho telemetry tensor sizes overflow usize")
        })?;
        if expected != values.len() {
            bail!(
                "rho telemetry got {} pseudo-gradient values for tensor groups totaling {expected}",
                values.len()
            );
        }
        let mut groups = Vec::with_capacity(tensor_numels.len());
        let mut offset = 0usize;
        for (group_index, &numel) in tensor_numels.iter().enumerate() {
            let numel = numel as usize;
            let mut projected = vec![0.0f32; self.dimension];
            for (coordinate, &value) in values[offset..offset + numel].iter().enumerate() {
                let hash = projection_hash(self.seed, group_index, coordinate);
                let bucket = (hash as usize) % self.dimension;
                let sign = if splitmix64(hash) & (1u64 << 63) == 0 {
                    1.0f32
                } else {
                    -1.0f32
                };
                projected[bucket] += sign * value;
            }
            groups.push(projected);
            offset += numel;
        }
        Ok(GroupedSketch { groups })
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn projection_hash(seed: u64, group_index: usize, coordinate: usize) -> u64 {
    splitmix64(
        seed ^ (group_index as u64).wrapping_mul(0xd6e8_feb8_6659_fd93)
            ^ (coordinate as u64).wrapping_mul(0xa076_1d64_78bd_642f),
    )
}

fn grouped_cosine(left: &GroupedSketch, right: &GroupedSketch) -> Result<Option<f64>> {
    if left.groups.len() != right.groups.len() {
        bail!("rho telemetry sketch tensor-group counts differ");
    }
    let mut dot = 0.0f64;
    let mut left_sq = 0.0f64;
    let mut right_sq = 0.0f64;
    for (left_group, right_group) in left.groups.iter().zip(&right.groups) {
        if left_group.len() != right_group.len() {
            bail!("rho telemetry sketch dimensions differ");
        }
        for (&left_value, &right_value) in left_group.iter().zip(right_group) {
            let left_value = left_value as f64;
            let right_value = right_value as f64;
            dot += left_value * right_value;
            left_sq += left_value * left_value;
            right_sq += right_value * right_value;
        }
    }
    if left_sq == 0.0 || right_sq == 0.0 {
        return Ok(None);
    }
    let cosine = dot / (left_sq.sqrt() * right_sq.sqrt());
    if !cosine.is_finite() {
        bail!("rho telemetry projected cosine is not finite");
    }
    Ok(Some(cosine.clamp(-1.0, 1.0)))
}

fn grouped_l2_norm(sketch: &GroupedSketch) -> f64 {
    sketch
        .groups
        .iter()
        .flat_map(|group| group.iter())
        .map(|value| (*value as f64).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn exact_l2_norm(values: &[f32]) -> Result<f64> {
    let squared = values
        .iter()
        .map(|value| (*value as f64).powi(2))
        .sum::<f64>();
    let norm = squared.sqrt();
    if !norm.is_finite() {
        bail!("rho telemetry pseudo-gradient norm is not finite");
    }
    Ok(norm)
}

#[derive(Clone, Debug, Serialize)]
struct PseudoGradientStats {
    definition: &'static str,
    l2_norm: f64,
    projected_l2_norm: f64,
}

#[derive(Clone, Debug, Serialize)]
struct AutocorrelationStats {
    estimator: &'static str,
    lag_1: Option<f64>,
    lag_2: Option<f64>,
    lag_3: Option<f64>,
    lag_4: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
struct SketchMetadata {
    method: &'static str,
    dimension_per_tensor_group: usize,
    seed: String,
    tensor_group_count: usize,
    retained_lags: usize,
}

#[derive(Clone, Debug, Serialize)]
struct WorkerNorm {
    learner_id: u32,
    l2_norm: f64,
}

#[derive(Clone, Debug, Serialize)]
struct WorkerPairCosine {
    learner_a: u32,
    learner_b: u32,
    cosine: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
struct CrossWorkerStats {
    definition: &'static str,
    estimator: &'static str,
    worker_count: usize,
    pair_count: usize,
    defined_pair_count: usize,
    mean_cosine: Option<f64>,
    min_cosine: Option<f64>,
    max_cosine: Option<f64>,
    workers: Vec<WorkerNorm>,
    pairs: Vec<WorkerPairCosine>,
}

#[derive(Clone, Debug, Serialize)]
struct RhoTelemetryRecord {
    schema: &'static str,
    event: &'static str,
    outer_step: u64,
    fragment: usize,
    fragment_round: u64,
    pseudo_gradient: PseudoGradientStats,
    autocorrelation: AutocorrelationStats,
    cross_worker: CrossWorkerStats,
    sketch: SketchMetadata,
}

#[derive(Debug)]
pub struct PreparedRhoTelemetry {
    fragment: usize,
    fragment_round: u64,
    sketch: GroupedSketch,
    record: RhoTelemetryRecord,
}

/// Stateful writer with four independent history slots per fragment.
pub struct RhoTelemetry {
    path: PathBuf,
    projection: ProjectionSketcher,
    histories: Vec<VecDeque<GroupedSketch>>,
    fragment_rounds: Vec<u64>,
}

impl RhoTelemetry {
    pub fn new(path: impl Into<PathBuf>, fragment_count: usize) -> Result<Self> {
        Self::with_projection(path, fragment_count, PROJECTION_DIMENSION, PROJECTION_SEED)
    }

    fn with_projection(
        path: impl Into<PathBuf>,
        fragment_count: usize,
        dimension: usize,
        seed: u64,
    ) -> Result<Self> {
        if fragment_count == 0 {
            bail!("rho telemetry requires at least one fragment");
        }
        let path = path.into();
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("create rho telemetry output directory {}", parent.display())
            })?;
        }
        // Fail before training if the configured artifact cannot be opened.
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("open rho telemetry JSONL {}", path.display()))?;
        Ok(Self {
            path,
            projection: ProjectionSketcher::new(dimension, seed)?,
            histories: (0..fragment_count)
                .map(|_| VecDeque::with_capacity(MAX_LAG))
                .collect(),
            fragment_rounds: vec![0; fragment_count],
        })
    }

    /// Prepare one record without mutating history.  Call `append` only after
    /// the corresponding outer step has committed successfully.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &self,
        outer_step: u64,
        fragment: usize,
        tensor_numels: &[u64],
        merged_pseudo_gradient: &[f32],
        current_anchor: &[f32],
        worker_candidates: &[(u32, &[f32])],
    ) -> Result<PreparedRhoTelemetry> {
        let history = self
            .histories
            .get(fragment)
            .with_context(|| format!("rho telemetry fragment {fragment} is out of range"))?;
        let fragment_round = self.fragment_rounds[fragment] + 1;
        let sketch = self
            .projection
            .sketch(tensor_numels, merged_pseudo_gradient)?;
        let lag = |distance: usize| -> Result<Option<f64>> {
            let Some(previous) = history.iter().rev().nth(distance - 1) else {
                return Ok(None);
            };
            grouped_cosine(&sketch, previous)
        };
        let cross_worker = cross_worker_stats(current_anchor, worker_candidates)?;
        let record = RhoTelemetryRecord {
            schema: RHO_TELEMETRY_SCHEMA,
            event: "outer_round_fragment",
            outer_step,
            fragment,
            fragment_round,
            pseudo_gradient: PseudoGradientStats {
                definition: "production_merged_anchor_minus_candidate_before_outer_step",
                l2_norm: exact_l2_norm(merged_pseudo_gradient)?,
                projected_l2_norm: grouped_l2_norm(&sketch),
            },
            autocorrelation: AutocorrelationStats {
                estimator: "cosine_of_count_sketch_random_projections",
                lag_1: lag(1)?,
                lag_2: lag(2)?,
                lag_3: lag(3)?,
                lag_4: lag(4)?,
            },
            cross_worker,
            sketch: SketchMetadata {
                method: "count_sketch_v1",
                dimension_per_tensor_group: self.projection.dimension,
                seed: format!("0x{:016x}", self.projection.seed),
                tensor_group_count: tensor_numels.len(),
                retained_lags: MAX_LAG,
            },
        };
        Ok(PreparedRhoTelemetry {
            fragment,
            fragment_round,
            sketch,
            record,
        })
    }

    pub fn append(&mut self, prepared: PreparedRhoTelemetry) -> Result<()> {
        let expected_round = self.fragment_rounds[prepared.fragment] + 1;
        if prepared.fragment_round != expected_round {
            bail!(
                "rho telemetry fragment {} expected round {expected_round}, got {}",
                prepared.fragment,
                prepared.fragment_round
            );
        }
        append_jsonl(&self.path, &prepared.record)?;
        let history = &mut self.histories[prepared.fragment];
        if history.len() == MAX_LAG {
            history.pop_front();
        }
        history.push_back(prepared.sketch);
        self.fragment_rounds[prepared.fragment] = prepared.fragment_round;
        Ok(())
    }
}

fn cross_worker_stats(
    anchor: &[f32],
    worker_candidates: &[(u32, &[f32])],
) -> Result<CrossWorkerStats> {
    for pair in worker_candidates.windows(2) {
        if pair[0].0 >= pair[1].0 {
            bail!("rho telemetry worker IDs must be strictly increasing");
        }
    }
    let mut workers = Vec::with_capacity(worker_candidates.len());
    let mut squared_norms = Vec::with_capacity(worker_candidates.len());
    for &(learner_id, values) in worker_candidates {
        if values.len() != anchor.len() {
            bail!(
                "rho telemetry worker {learner_id} candidate has {} values, expected {}",
                values.len(),
                anchor.len()
            );
        }
        let squared = anchor
            .iter()
            .zip(values)
            .map(|(&anchor_value, &candidate_value)| {
                // Match the production merge's f32 subtraction exactly,
                // then accumulate norms/dots in f64.
                let delta = (anchor_value - candidate_value) as f64;
                delta * delta
            })
            .sum::<f64>();
        let norm = squared.sqrt();
        if !norm.is_finite() {
            bail!("rho telemetry worker {learner_id} norm is not finite");
        }
        squared_norms.push(squared);
        workers.push(WorkerNorm {
            learner_id,
            l2_norm: norm,
        });
    }

    let pair_count = worker_candidates
        .len()
        .saturating_mul(worker_candidates.len().saturating_sub(1))
        / 2;
    let mut pairs = Vec::with_capacity(pair_count);
    let mut defined = Vec::with_capacity(pair_count);
    for left in 0..worker_candidates.len() {
        for right in left + 1..worker_candidates.len() {
            let (learner_a, left_values) = worker_candidates[left];
            let (learner_b, right_values) = worker_candidates[right];
            let cosine = if squared_norms[left] == 0.0 || squared_norms[right] == 0.0 {
                None
            } else {
                let dot = anchor
                    .iter()
                    .zip(left_values)
                    .zip(right_values)
                    .map(|((&anchor_value, &left_value), &right_value)| {
                        ((anchor_value - left_value) as f64) * ((anchor_value - right_value) as f64)
                    })
                    .sum::<f64>();
                let value = dot / (squared_norms[left].sqrt() * squared_norms[right].sqrt());
                if !value.is_finite() {
                    bail!("rho telemetry worker cosine for {learner_a}/{learner_b} is not finite");
                }
                Some(value.clamp(-1.0, 1.0))
            };
            if let Some(value) = cosine {
                defined.push(value);
            }
            pairs.push(WorkerPairCosine {
                learner_a,
                learner_b,
                cosine,
            });
        }
    }
    let mean_cosine =
        (!defined.is_empty()).then(|| defined.iter().sum::<f64>() / defined.len() as f64);
    let min_cosine = defined.iter().copied().reduce(f64::min);
    let max_cosine = defined.iter().copied().reduce(f64::max);
    Ok(CrossWorkerStats {
        definition:
            "current_syncer_anchor_minus_admitted_candidate_before_optional_delta_correction",
        estimator: "exact_cosine",
        worker_count: worker_candidates.len(),
        pair_count,
        defined_pair_count: defined.len(),
        mean_cosine,
        min_cosine,
        max_cosine,
        workers,
        pairs,
    })
}

fn append_jsonl(path: &Path, record: &RhoTelemetryRecord) -> Result<()> {
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("open rho telemetry JSONL {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, record)
        .with_context(|| format!("serialize rho telemetry JSONL {}", path.display()))?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_jsonl(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "yeto-rho-{label}-{}-{nonce}.jsonl",
            std::process::id()
        ))
    }

    #[test]
    fn sketch_is_deterministic_for_fixed_seed() {
        let values = (0..97)
            .map(|index| ((index as f32) * 0.37).sin())
            .collect::<Vec<_>>();
        let first = ProjectionSketcher::new(64, PROJECTION_SEED)
            .unwrap()
            .sketch(&[31, 66], &values)
            .unwrap();
        let second = ProjectionSketcher::new(64, PROJECTION_SEED)
            .unwrap()
            .sketch(&[31, 66], &values)
            .unwrap();
        let other_seed = ProjectionSketcher::new(64, PROJECTION_SEED + 1)
            .unwrap()
            .sketch(&[31, 66], &values)
            .unwrap();
        assert_eq!(first, second);
        assert_ne!(first, other_seed);
    }

    #[test]
    fn recovers_identical_and_orthogonal_known_signals() {
        let path = temp_jsonl("signals");
        let mut telemetry = RhoTelemetry::with_projection(&path, 1, 128, PROJECTION_SEED).unwrap();
        let projection = telemetry.projection;
        let first_coordinate = 0usize;
        let first_bucket =
            (projection_hash(projection.seed, 0, first_coordinate) as usize) % projection.dimension;
        let orthogonal_coordinate = (1..256)
            .find(|&coordinate| {
                (projection_hash(projection.seed, 0, coordinate) as usize) % projection.dimension
                    != first_bucket
            })
            .unwrap();
        let mut signal = vec![0.0f32; 256];
        signal[first_coordinate] = 3.0;
        let mut orthogonal = vec![0.0f32; 256];
        orthogonal[orthogonal_coordinate] = -7.0;
        let anchor = vec![0.0f32; 256];
        let candidate = vec![0.0f32; 256];
        let workers = [(0u32, candidate.as_slice())];

        let first = telemetry
            .prepare(1, 0, &[256], &signal, &anchor, &workers)
            .unwrap();
        assert_eq!(first.record.autocorrelation.lag_1, None);
        telemetry.append(first).unwrap();
        let identical = telemetry
            .prepare(2, 0, &[256], &signal, &anchor, &workers)
            .unwrap();
        assert!((identical.record.autocorrelation.lag_1.unwrap() - 1.0).abs() < 1e-12);
        telemetry.append(identical).unwrap();
        let orthogonal_record = telemetry
            .prepare(3, 0, &[256], &orthogonal, &anchor, &workers)
            .unwrap();
        assert!(
            orthogonal_record
                .record
                .autocorrelation
                .lag_1
                .unwrap()
                .abs()
                < 1e-12
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn jsonl_record_has_registered_schema_and_cross_worker_fields() {
        let path = temp_jsonl("schema");
        let mut telemetry = RhoTelemetry::with_projection(&path, 2, 32, PROJECTION_SEED).unwrap();
        let anchor = [0.0f32, 0.0, 0.0, 0.0];
        let candidate_a = [-1.0f32, 0.0, 0.0, 0.0];
        let candidate_b = [-1.0f32, 0.0, 0.0, 0.0];
        let workers = [
            (3u32, candidate_a.as_slice()),
            (9u32, candidate_b.as_slice()),
        ];
        let prepared = telemetry
            .prepare(17, 1, &[2, 2], &[1.0, 0.0, 0.0, 0.0], &anchor, &workers)
            .unwrap();
        telemetry.append(prepared).unwrap();

        let contents = std::fs::read_to_string(&path).unwrap();
        assert_eq!(contents.lines().count(), 1);
        let record: serde_json::Value = serde_json::from_str(contents.trim()).unwrap();
        assert_eq!(record["schema"], RHO_TELEMETRY_SCHEMA);
        assert_eq!(record["event"], "outer_round_fragment");
        assert_eq!(record["outer_step"], 17);
        assert_eq!(record["fragment"], 1);
        assert_eq!(record["fragment_round"], 1);
        assert_eq!(record["autocorrelation"]["lag_1"], serde_json::Value::Null);
        assert_eq!(record["autocorrelation"]["lag_4"], serde_json::Value::Null);
        assert_eq!(record["pseudo_gradient"]["l2_norm"], 1.0);
        assert_eq!(record["sketch"]["dimension_per_tensor_group"], 32);
        assert_eq!(record["sketch"]["tensor_group_count"], 2);
        assert_eq!(record["cross_worker"]["pair_count"], 1);
        assert_eq!(record["cross_worker"]["defined_pair_count"], 1);
        assert_eq!(record["cross_worker"]["mean_cosine"], 1.0);
        assert_eq!(record["cross_worker"]["pairs"][0]["cosine"], 1.0);
        std::fs::remove_file(path).unwrap();
    }
}
