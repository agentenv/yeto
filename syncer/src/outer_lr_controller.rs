//! Age-aware outer-learning-rate scaling.
//!
//! The controller is deliberately scalar and side-effect free at evaluation
//! time. The syncer derives a fragment's one-indexed age and planned horizon,
//! then folds the returned scale into the configured outer learning rate before
//! materializing an optimizer step. JSON inputs are parsed and bound to the
//! fragment layout once at startup.

use anyhow::{bail, Context, Result};
use clap::ValueEnum;
use serde::de::DeserializeOwned;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

pub const DRIFT_SURFACE_SCHEMA: &str = "yeto.outer_lr_drift_surface.v1";
pub const SPECTRAL_SKETCH_SCHEMA: &str = "yeto.outer_lr_spectral_sketch.v1";
pub const ORACLE_SCHEDULE_SCHEMA: &str = "yeto.outer_lr_oracle_schedule.v1";
const DRIFT_OUTPUT: &str = "log2_lr_multiplier";
const MAX_FACTOR_POWER: u32 = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum ControllerMode {
    /// Normalize the finite-age code-true Nesterov direction multiplier.
    #[value(name = "transient")]
    Transient,
    /// Apply transient normalization and a measured factorial drift surface.
    #[value(name = "measured-drift")]
    MeasuredDrift,
    /// Use an externally supplied, exact per-fragment scale schedule.
    #[value(name = "oracle")]
    Oracle,
}

impl ControllerMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Transient => "transient",
            Self::MeasuredDrift => "measured-drift",
            Self::Oracle => "oracle",
        }
    }

    pub const fn uses_transient_normalization(self) -> bool {
        matches!(self, Self::Transient | Self::MeasuredDrift)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScaleOutput {
    /// Complete multiplier folded into the configured outer learning rate.
    pub scale: f64,
    /// Direction-multiplier normalization, absent only in oracle mode.
    pub transient_scale: Option<f64>,
    /// Measured surface multiplier, present only in measured-drift mode.
    pub drift_scale: Option<f64>,
}

pub struct ScaleInput<'a> {
    pub mu: f64,
    /// One-indexed per-fragment outer-update age.
    pub t: u64,
    /// Planned number of outer updates for this fragment.
    pub t_planned: u64,
    /// Probe-produced named scalars used by a measured-drift surface.
    pub spectral_sketch: Option<&'a BTreeMap<String, f64>>,
    /// Exact scale supplied by an oracle schedule.
    pub oracle_scale: Option<f64>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SurfaceFeature {
    name: String,
    source: String,
    #[serde(default)]
    transform: FeatureTransform,
    #[serde(default)]
    center: f64,
    #[serde(default = "one")]
    scale: f64,
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum FeatureTransform {
    #[default]
    Identity,
    Log2,
    Ln,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SurfaceTerm {
    coefficient: f64,
    /// Product of named normalized features raised to small integer powers.
    /// An empty map is the intercept term.
    #[serde(default)]
    powers: BTreeMap<String, u32>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScaleBounds {
    min: f64,
    max: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FactorialSurface {
    schema: String,
    output: String,
    features: Vec<SurfaceFeature>,
    terms: Vec<SurfaceTerm>,
    drift_scale_bounds: ScaleBounds,
}

impl FactorialSurface {
    pub fn from_json_str(contents: &str) -> Result<Self> {
        let surface: Self =
            serde_json::from_str(contents).context("parse outer-LR measured-drift surface JSON")?;
        surface.validate()?;
        Ok(surface)
    }

    fn validate(&self) -> Result<()> {
        if self.schema != DRIFT_SURFACE_SCHEMA {
            bail!(
                "outer-LR drift surface schema must be {DRIFT_SURFACE_SCHEMA:?}, got {:?}",
                self.schema
            );
        }
        if self.output != DRIFT_OUTPUT {
            bail!(
                "outer-LR drift surface output must be {DRIFT_OUTPUT:?}, got {:?}",
                self.output
            );
        }
        if !self.drift_scale_bounds.min.is_finite()
            || !self.drift_scale_bounds.max.is_finite()
            || self.drift_scale_bounds.min <= 0.0
            || self.drift_scale_bounds.min > 1.0
            || self.drift_scale_bounds.max < 1.0
            || self.drift_scale_bounds.min > self.drift_scale_bounds.max
        {
            bail!("outer-LR drift_scale_bounds must be finite, positive, ordered, and contain 1");
        }

        let mut names = BTreeSet::new();
        for feature in &self.features {
            validate_name(&feature.name, "surface feature")?;
            if !names.insert(feature.name.clone()) {
                bail!("duplicate outer-LR surface feature {:?}", feature.name);
            }
            validate_source(&feature.source)?;
            if !feature.center.is_finite() || !feature.scale.is_finite() || feature.scale == 0.0 {
                bail!(
                    "outer-LR surface feature {:?} has a non-finite center or zero/non-finite scale",
                    feature.name
                );
            }
        }

        for term in &self.terms {
            if !term.coefficient.is_finite() {
                bail!("outer-LR surface term has a non-finite coefficient");
            }
            for (name, &power) in &term.powers {
                if !names.contains(name) {
                    bail!("outer-LR surface term references unknown feature {name:?}");
                }
                if power == 0 || power > MAX_FACTOR_POWER {
                    bail!("outer-LR surface power for {name:?} must be in 1..={MAX_FACTOR_POWER}");
                }
            }
        }
        Ok(())
    }

    fn required_spectral_sources(&self) -> BTreeSet<&str> {
        let used_features: BTreeSet<&str> = self
            .terms
            .iter()
            .filter(|term| term.coefficient != 0.0)
            .flat_map(|term| term.powers.keys().map(String::as_str))
            .collect();
        self.features
            .iter()
            .filter(|feature| used_features.contains(feature.name.as_str()))
            .filter_map(|feature| feature.source.strip_prefix("spectral."))
            .collect()
    }

    fn drift_scale(&self, input: &ScaleInput<'_>) -> Result<f64> {
        let mut values = BTreeMap::new();
        let mut log2_multiplier = 0.0f64;
        for term in &self.terms {
            // Besides being cheaper, this makes an all-zero fitted surface
            // independent of optional spectral inputs and exactly reducible to
            // transient mode.
            if term.coefficient == 0.0 {
                continue;
            }
            let mut product = 1.0f64;
            for (name, &power) in &term.powers {
                let value = if let Some(value) = values.get(name) {
                    *value
                } else {
                    let feature = self
                        .features
                        .iter()
                        .find(|feature| feature.name == *name)
                        .expect("validated surface term must reference a feature");
                    let value = feature.value(input)?;
                    values.insert(name.clone(), value);
                    value
                };
                product *= value.powi(power as i32);
            }
            log2_multiplier += term.coefficient * product;
        }
        if !log2_multiplier.is_finite() {
            bail!("outer-LR drift surface produced a non-finite log2 multiplier");
        }
        if log2_multiplier == 0.0 {
            return Ok(1.0);
        }
        let min_log2 = self.drift_scale_bounds.min.log2();
        let max_log2 = self.drift_scale_bounds.max.log2();
        let scale = if log2_multiplier <= min_log2 {
            self.drift_scale_bounds.min
        } else if log2_multiplier >= max_log2 {
            self.drift_scale_bounds.max
        } else {
            log2_multiplier.exp2()
        };
        if !scale.is_finite() || scale <= 0.0 {
            bail!("outer-LR drift surface produced a non-finite multiplier");
        }
        Ok(scale)
    }
}

impl SurfaceFeature {
    fn value(&self, input: &ScaleInput<'_>) -> Result<f64> {
        let raw = match self.source.as_str() {
            "mu" => input.mu,
            "t" => input.t as f64,
            "T_planned" => input.t_planned as f64,
            "age_fraction" => input.t as f64 / input.t_planned as f64,
            "remaining_steps" => (input.t_planned - input.t) as f64,
            "remaining_fraction" => (input.t_planned - input.t) as f64 / input.t_planned as f64,
            source => {
                let key = source
                    .strip_prefix("spectral.")
                    .expect("validated source must be built-in or spectral");
                *input
                    .spectral_sketch
                    .and_then(|sketch| sketch.get(key))
                    .with_context(|| {
                        format!("outer-LR drift surface requires spectral feature {key:?}")
                    })?
            }
        };
        if !raw.is_finite() {
            bail!("outer-LR surface source {:?} is not finite", self.source);
        }
        let transformed = match self.transform {
            FeatureTransform::Identity => raw,
            FeatureTransform::Log2 if raw > 0.0 => raw.log2(),
            FeatureTransform::Ln if raw > 0.0 => raw.ln(),
            FeatureTransform::Log2 | FeatureTransform::Ln => {
                bail!(
                    "outer-LR surface source {:?} must be positive for its transform",
                    self.source
                )
            }
        };
        let value = (transformed - self.center) / self.scale;
        if !value.is_finite() {
            bail!(
                "outer-LR surface feature {:?} normalized to a non-finite value",
                self.name
            );
        }
        Ok(value)
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SpectralSketch {
    schema: String,
    #[serde(default)]
    global_features: BTreeMap<String, f64>,
    #[serde(default)]
    fragment_features: BTreeMap<usize, BTreeMap<String, f64>>,
}

impl SpectralSketch {
    fn validate(&self) -> Result<()> {
        if self.schema != SPECTRAL_SKETCH_SCHEMA {
            bail!(
                "outer-LR spectral sketch schema must be {SPECTRAL_SKETCH_SCHEMA:?}, got {:?}",
                self.schema
            );
        }
        validate_feature_map(&self.global_features)?;
        for features in self.fragment_features.values() {
            validate_feature_map(features)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OracleSchedule {
    schema: String,
    /// Default one-indexed schedule, addressed as `scales[t - 1]`.
    scales: Option<Vec<f64>>,
    /// Per-fragment overrides keyed by the decimal fragment id.
    #[serde(default)]
    fragment_scales: BTreeMap<usize, Vec<f64>>,
}

impl OracleSchedule {
    fn validate(&self) -> Result<()> {
        if self.schema != ORACLE_SCHEDULE_SCHEMA {
            bail!(
                "outer-LR oracle schedule schema must be {ORACLE_SCHEDULE_SCHEMA:?}, got {:?}",
                self.schema
            );
        }
        if self.scales.is_none() && self.fragment_scales.is_empty() {
            bail!("outer-LR oracle schedule must provide scales or fragment_scales");
        }
        if let Some(scales) = &self.scales {
            validate_scales(scales)?;
        }
        for scales in self.fragment_scales.values() {
            validate_scales(scales)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct ControllerConfig {
    mode: ControllerMode,
    surface: Option<FactorialSurface>,
    sketch: Option<SpectralSketch>,
    oracle: Option<OracleSchedule>,
}

impl ControllerConfig {
    pub fn load(
        mode: ControllerMode,
        drift_surface_path: Option<&Path>,
        spectral_sketch_path: Option<&Path>,
        oracle_schedule_path: Option<&Path>,
    ) -> Result<Self> {
        match mode {
            ControllerMode::Transient => {
                reject_path(drift_surface_path, "--outer-lr-drift-surface", mode)?;
                reject_path(spectral_sketch_path, "--outer-lr-spectral-sketch", mode)?;
                reject_path(oracle_schedule_path, "--outer-lr-oracle-schedule", mode)?;
                Ok(Self::transient())
            }
            ControllerMode::MeasuredDrift => {
                let surface_path = drift_surface_path.context(
                    "--outer-lr-controller measured-drift requires --outer-lr-drift-surface",
                )?;
                reject_path(oracle_schedule_path, "--outer-lr-oracle-schedule", mode)?;
                let surface = FactorialSurface::from_json_str(
                    &std::fs::read_to_string(surface_path).with_context(|| {
                        format!("read outer-LR drift surface {}", surface_path.display())
                    })?,
                )?;
                let sketch = spectral_sketch_path
                    .map(read_json::<SpectralSketch>)
                    .transpose()?;
                if let Some(sketch) = &sketch {
                    sketch.validate()?;
                }
                Ok(Self {
                    mode,
                    surface: Some(surface),
                    sketch,
                    oracle: None,
                })
            }
            ControllerMode::Oracle => {
                reject_path(drift_surface_path, "--outer-lr-drift-surface", mode)?;
                reject_path(spectral_sketch_path, "--outer-lr-spectral-sketch", mode)?;
                let oracle_path = oracle_schedule_path
                    .context("--outer-lr-controller oracle requires --outer-lr-oracle-schedule")?;
                let oracle: OracleSchedule = read_json(oracle_path)?;
                oracle.validate()?;
                Ok(Self {
                    mode,
                    surface: None,
                    sketch: None,
                    oracle: Some(oracle),
                })
            }
        }
    }

    pub const fn transient() -> Self {
        Self {
            mode: ControllerMode::Transient,
            surface: None,
            sketch: None,
            oracle: None,
        }
    }

    pub fn bind(&self, total_steps: u64, fragments: usize) -> Result<BoundController> {
        if fragments == 0 {
            bail!("outer-LR controller cannot bind to an empty fragment layout");
        }
        let planned_steps: Vec<u64> = (0..fragments)
            .map(|fid| planned_fragment_steps(total_steps, fragments, fid))
            .collect::<Result<_>>()?;

        let mut spectral_features = vec![BTreeMap::new(); fragments];
        if let Some(sketch) = &self.sketch {
            for &fid in sketch.fragment_features.keys() {
                if fid >= fragments {
                    bail!(
                        "outer-LR spectral sketch contains fragment {fid}, but layout has {fragments} fragments"
                    );
                }
            }
            for (fid, features) in spectral_features.iter_mut().enumerate() {
                features.extend(sketch.global_features.clone());
                if let Some(overrides) = sketch.fragment_features.get(&fid) {
                    features.extend(overrides.clone());
                }
            }
        }

        if let Some(surface) = &self.surface {
            for source in surface.required_spectral_sources() {
                for (fid, (&planned, features)) in
                    planned_steps.iter().zip(&spectral_features).enumerate()
                {
                    if planned > 0 && !features.contains_key(source) {
                        bail!(
                            "outer-LR drift surface requires spectral feature {source:?} for fragment {fid}"
                        );
                    }
                }
            }
        }

        let mut oracle_scales = vec![Vec::new(); fragments];
        if let Some(oracle) = &self.oracle {
            for &fid in oracle.fragment_scales.keys() {
                if fid >= fragments {
                    bail!(
                        "outer-LR oracle schedule contains fragment {fid}, but layout has {fragments} fragments"
                    );
                }
            }
            for fid in 0..fragments {
                if planned_steps[fid] == 0 {
                    continue;
                }
                let scales = oracle
                    .fragment_scales
                    .get(&fid)
                    .or(oracle.scales.as_ref())
                    .with_context(|| {
                        format!("outer-LR oracle schedule has no scales for fragment {fid}")
                    })?;
                if scales.len() as u64 != planned_steps[fid] {
                    bail!(
                        "outer-LR oracle schedule has {} scales for fragment {fid}, planned horizon is {}",
                        scales.len(),
                        planned_steps[fid]
                    );
                }
                oracle_scales[fid] = scales.clone();
            }
        }

        Ok(BoundController {
            mode: self.mode,
            surface: self.surface.clone(),
            planned_steps,
            spectral_features,
            oracle_scales,
        })
    }
}

#[derive(Clone, Debug)]
pub struct BoundController {
    mode: ControllerMode,
    surface: Option<FactorialSurface>,
    planned_steps: Vec<u64>,
    spectral_features: Vec<BTreeMap<String, f64>>,
    oracle_scales: Vec<Vec<f64>>,
}

impl BoundController {
    pub const fn mode(&self) -> ControllerMode {
        self.mode
    }

    pub fn scale_for_fragment(&self, mu: f32, fid: usize, t: u64) -> Result<ScaleOutput> {
        let &t_planned = self
            .planned_steps
            .get(fid)
            .with_context(|| format!("outer-LR controller fragment {fid} is out of range"))?;
        let spectral_sketch = self
            .spectral_features
            .get(fid)
            .filter(|map| !map.is_empty());
        let oracle_scale = if self.mode == ControllerMode::Oracle && t > 0 {
            usize::try_from(t - 1)
                .ok()
                .and_then(|index| self.oracle_scales.get(fid)?.get(index))
                .copied()
        } else {
            None
        };
        evaluate_scale(
            self.mode,
            self.surface.as_ref(),
            ScaleInput {
                mu: f64::from(mu),
                t,
                t_planned,
                spectral_sketch,
                oracle_scale,
            },
        )
    }
}

/// Evaluate one controller step. This is the cross-language reference boundary.
pub fn evaluate_scale(
    mode: ControllerMode,
    surface: Option<&FactorialSurface>,
    input: ScaleInput<'_>,
) -> Result<ScaleOutput> {
    if input.t == 0 || input.t_planned == 0 || input.t > input.t_planned {
        bail!(
            "outer-LR controller requires 1 <= t <= T_planned, got t={} T_planned={}",
            input.t,
            input.t_planned
        );
    }
    match mode {
        ControllerMode::Transient => {
            let transient = transient_normalization_scale(input.mu, input.t)?;
            Ok(ScaleOutput {
                scale: transient,
                transient_scale: Some(transient),
                drift_scale: None,
            })
        }
        ControllerMode::MeasuredDrift => {
            let transient = transient_normalization_scale(input.mu, input.t)?;
            let drift = surface
                .context("measured-drift outer-LR controller has no factorial surface")?
                .drift_scale(&input)?;
            let scale = if drift == 1.0 {
                transient
            } else {
                transient * drift
            };
            if !scale.is_finite() || scale <= 0.0 {
                bail!("outer-LR controller produced a non-finite or non-positive scale");
            }
            Ok(ScaleOutput {
                scale,
                transient_scale: Some(transient),
                drift_scale: Some(drift),
            })
        }
        ControllerMode::Oracle => {
            let scale = input
                .oracle_scale
                .context("oracle outer-LR controller has no scale for this fragment and age")?;
            if !scale.is_finite() || scale <= 0.0 {
                bail!("oracle outer-LR scale must be finite and positive, got {scale}");
            }
            Ok(ScaleOutput {
                scale,
                transient_scale: None,
                drift_scale: None,
            })
        }
    }
}

/// Exact direction-multiplier rule used by the legacy
/// `--outer-bias-correction` path.
pub fn transient_normalization_scale(mu: f64, t: u64) -> Result<f64> {
    if !(0.0..1.0).contains(&mu) {
        bail!("transient outer-LR normalization requires mu in [0, 1), got {mu}");
    }
    if t == 0 {
        bail!("transient outer-LR normalization requires one-indexed t >= 1");
    }
    let exponent = t.saturating_add(1).min(i32::MAX as u64) as i32;
    let divisor = 1.0 - mu.powi(exponent);
    if !divisor.is_finite() || divisor <= 0.0 {
        bail!("transient outer-LR normalization produced a non-positive divisor");
    }
    Ok(1.0 / divisor)
}

pub fn planned_fragment_steps(total_steps: u64, fragments: usize, fid: usize) -> Result<u64> {
    if fragments == 0 || fid >= fragments {
        bail!("cannot plan outer-LR horizon for fragment {fid} of {fragments}");
    }
    let first_step = (fid as u64).saturating_add(1);
    if first_step > total_steps {
        return Ok(0);
    }
    Ok(1 + (total_steps - first_step) / fragments as u64)
}

fn validate_source(source: &str) -> Result<()> {
    match source {
        "mu" | "t" | "T_planned" | "age_fraction" | "remaining_steps" | "remaining_fraction" => {
            Ok(())
        }
        source if source.starts_with("spectral.") => {
            validate_name(&source["spectral.".len()..], "spectral source")
        }
        _ => bail!("unknown outer-LR surface source {source:?}"),
    }
}

fn validate_name(name: &str, kind: &str) -> Result<()> {
    if name.is_empty()
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
    {
        bail!("{kind} name {name:?} must contain only ASCII letters, digits, '.', '-', or '_'");
    }
    Ok(())
}

fn validate_feature_map(features: &BTreeMap<String, f64>) -> Result<()> {
    for (name, value) in features {
        validate_name(name, "spectral feature")?;
        if !value.is_finite() {
            bail!("spectral feature {name:?} must be finite");
        }
    }
    Ok(())
}

fn validate_scales(scales: &[f64]) -> Result<()> {
    if scales
        .iter()
        .any(|scale| !scale.is_finite() || *scale <= 0.0)
    {
        bail!("outer-LR oracle scales must all be finite and positive");
    }
    Ok(())
}

fn reject_path(path: Option<&Path>, flag: &str, mode: ControllerMode) -> Result<()> {
    if path.is_some() {
        bail!(
            "{flag} is not valid with --outer-lr-controller {}",
            mode.as_str()
        );
    }
    Ok(())
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("read outer-LR controller JSON {}", path.display()))?;
    serde_json::from_str(&contents)
        .with_context(|| format!("parse outer-LR controller JSON {}", path.display()))
}

const fn one() -> f64 {
    1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_surface() -> FactorialSurface {
        FactorialSurface::from_json_str(
            r#"{
                "schema":"yeto.outer_lr_drift_surface.v1",
                "output":"log2_lr_multiplier",
                "features":[
                    {"name":"u","source":"mu","scale":0.9},
                    {"name":"q","source":"T_planned","center":5.0,"scale":10.0},
                    {"name":"sharp","source":"spectral.lambda_max","transform":"log2"}
                ],
                "terms":[
                    {"coefficient":0.0,"powers":{}},
                    {"coefficient":0.0,"powers":{"u":1,"q":1,"sharp":1}}
                ],
                "drift_scale_bounds":{"min":0.125,"max":8.0}
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn transient_property_bit_matches_legacy_direction_rule() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..20_000 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let mu = ((state >> 32) as u32 % 999_999) as f32 / 1_000_000.0;
            let t = 1 + (state % 100_000);
            let exponent = (t + 1).min(i32::MAX as u64) as i32;
            let legacy = 1.0 / (1.0 - f64::from(mu).powi(exponent));
            let actual = transient_normalization_scale(f64::from(mu), t).unwrap();
            assert_eq!(actual.to_bits(), legacy.to_bits(), "mu={mu} t={t}");
        }
    }

    #[test]
    fn zero_drift_surface_property_bit_reduces_to_transient_mode() {
        let surface = zero_surface();
        for &mu in &[0.0f64, 0.25, 0.5, 0.9, 0.99, 0.999_999] {
            for t_planned in 1u64..=64 {
                for t in 1..=t_planned {
                    let transient = evaluate_scale(
                        ControllerMode::Transient,
                        None,
                        ScaleInput {
                            mu,
                            t,
                            t_planned,
                            spectral_sketch: None,
                            oracle_scale: None,
                        },
                    )
                    .unwrap();
                    let drift = evaluate_scale(
                        ControllerMode::MeasuredDrift,
                        Some(&surface),
                        ScaleInput {
                            mu,
                            t,
                            t_planned,
                            spectral_sketch: None,
                            oracle_scale: None,
                        },
                    )
                    .unwrap();
                    assert_eq!(drift.scale.to_bits(), transient.scale.to_bits());
                    assert_eq!(drift.drift_scale, Some(1.0));
                }
            }
        }
    }

    #[test]
    fn measured_surface_uses_centered_factorial_and_spectral_features() {
        let surface = FactorialSurface::from_json_str(
            r#"{
                "schema":"yeto.outer_lr_drift_surface.v1",
                "output":"log2_lr_multiplier",
                "features":[
                    {"name":"u","source":"mu","scale":0.9},
                    {"name":"q","source":"T_planned","center":5.0,"scale":10.0},
                    {"name":"h","source":"spectral.window","transform":"log2","center":9.0}
                ],
                "terms":[
                    {"coefficient":-0.2,"powers":{"u":1}},
                    {"coefficient":-0.4,"powers":{"u":1,"q":1}},
                    {"coefficient":-0.1,"powers":{"u":1,"h":1}}
                ],
                "drift_scale_bounds":{"min":0.125,"max":8.0}
            }"#,
        )
        .unwrap();
        let sketch = BTreeMap::from([("window".to_owned(), 1024.0)]);
        let output = evaluate_scale(
            ControllerMode::MeasuredDrift,
            Some(&surface),
            ScaleInput {
                mu: 0.9,
                t: 3,
                t_planned: 10,
                spectral_sketch: Some(&sketch),
                oracle_scale: None,
            },
        )
        .unwrap();
        let expected_drift = 2.0f64.powf(-0.2 - 0.4 * 0.5 - 0.1);
        assert!((output.drift_scale.unwrap() - expected_drift).abs() < 1e-15);
        assert_eq!(output.transient_scale, Some(1.0 / (1.0 - 0.9f64.powi(4))));
    }

    #[test]
    fn drift_bounds_apply_before_exponentiation() {
        let surface = FactorialSurface::from_json_str(
            r#"{
                "schema":"yeto.outer_lr_drift_surface.v1",
                "output":"log2_lr_multiplier",
                "features":[],
                "terms":[{"coefficient":1e300,"powers":{}}],
                "drift_scale_bounds":{"min":0.125,"max":8.0}
            }"#,
        )
        .unwrap();
        let output = evaluate_scale(
            ControllerMode::MeasuredDrift,
            Some(&surface),
            ScaleInput {
                mu: 0.0,
                t: 1,
                t_planned: 1,
                spectral_sketch: None,
                oracle_scale: None,
            },
        )
        .unwrap();
        assert_eq!(output.drift_scale, Some(8.0));
        assert_eq!(output.scale, 8.0);
    }

    #[test]
    fn oracle_is_an_exact_passthrough() {
        for &scale in &[f64::MIN_POSITIVE, 0.125, 1.0, 3.75, f64::MAX / 2.0] {
            let output = evaluate_scale(
                ControllerMode::Oracle,
                None,
                ScaleInput {
                    mu: f64::NAN,
                    t: 2,
                    t_planned: 3,
                    spectral_sketch: None,
                    oracle_scale: Some(scale),
                },
            )
            .unwrap();
            assert_eq!(output.scale.to_bits(), scale.to_bits());
            assert_eq!(output.transient_scale, None);
            assert_eq!(output.drift_scale, None);
        }
    }

    #[test]
    fn planned_horizons_follow_round_robin_total_steps() {
        assert_eq!(
            (0..4)
                .map(|fid| planned_fragment_steps(10, 4, fid).unwrap())
                .collect::<Vec<_>>(),
            vec![3, 3, 2, 2]
        );
        assert_eq!(planned_fragment_steps(2, 4, 2).unwrap(), 0);
    }

    #[test]
    fn bound_oracle_uses_exact_fragment_schedule_and_planned_horizon() {
        let config = ControllerConfig {
            mode: ControllerMode::Oracle,
            surface: None,
            sketch: None,
            oracle: Some(OracleSchedule {
                schema: ORACLE_SCHEDULE_SCHEMA.to_owned(),
                scales: Some(vec![0.75, 0.5]),
                fragment_scales: BTreeMap::from([(0, vec![1.25, 1.0, 0.875])]),
            }),
        };
        let controller = config.bind(5, 2).unwrap();
        let first = controller.scale_for_fragment(0.9, 0, 1).unwrap();
        let last = controller.scale_for_fragment(0.9, 1, 2).unwrap();
        assert_eq!(first.scale.to_bits(), 1.25f64.to_bits());
        assert_eq!(last.scale.to_bits(), 0.5f64.to_bits());

        let short = ControllerConfig {
            mode: ControllerMode::Oracle,
            surface: None,
            sketch: None,
            oracle: Some(OracleSchedule {
                schema: ORACLE_SCHEDULE_SCHEMA.to_owned(),
                scales: Some(vec![1.0]),
                fragment_scales: BTreeMap::new(),
            }),
        };
        assert!(short.bind(5, 2).is_err());
    }

    #[test]
    fn active_missing_spectral_feature_fails_at_bind_time() {
        let mut surface = zero_surface();
        surface.terms[1].coefficient = 0.5;
        let config = ControllerConfig {
            mode: ControllerMode::MeasuredDrift,
            surface: Some(surface),
            sketch: None,
            oracle: None,
        };
        let error = config.bind(8, 4).unwrap_err();
        assert!(error.to_string().contains("lambda_max"), "{error}");
    }
}
