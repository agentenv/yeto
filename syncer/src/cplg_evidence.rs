//! Frozen active-E1 evidence and receipt production for CPLG-SGD.
//!
//! This module is deliberately separate from the selector implementation: it
//! observes already-produced boundary diagnostics, enforces their closed
//! action-hash contract, and writes canonical receipts without participating
//! in any CPLG arithmetic or selection decision.

use std::collections::BTreeMap;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::merge::OuterOptimizer;
use crate::state::{CplgReason, CplgStepStats, GlobalState};

pub const EXPECTED_COMMITS: u64 = 32;
pub const EXPECTED_FRAGMENTS: usize = 4;
const TERMINAL_LOCAL_STEPS: u64 = 34;
const RAW_TRAINING_TOKENS: u64 = 4_352;
const ZERO_SHA256: &str = "0000000000000000000000000000000000000000000000000000000000000000";

const LEDGER_FIELDS: [&str; 32] = [
    "schema_version",
    "row_index",
    "run_id",
    "run_config_sha256",
    "source_commit",
    "commit_sequence",
    "fragment",
    "fragment_version",
    "responder_step",
    "responder_tokens",
    "weight_identity_sha256",
    "layout_sha256",
    "initial_state_sha256",
    "cplg_rho",
    "cplg_theta",
    "cplg_previous_theta",
    "cplg_coherence",
    "cplg_phi",
    "cplg_shadow_score",
    "cplg_score_count",
    "cplg_interlock_open",
    "cplg_used_nonstock",
    "cplg_state_cleared",
    "cplg_reason",
    "cplg_stock_sha256",
    "cplg_previous_stock_sha256",
    "cplg_previous_tangent_sha256",
    "cplg_transported_tangent_sha256",
    "cplg_candidate_sha256",
    "cplg_action_sha256",
    "previous_row_sha256",
    "row_sha256",
];

#[derive(Clone, Debug)]
pub struct CplgOnlineConfig {
    pub run_id: String,
    pub run_config_sha256: String,
    pub source_commit: String,
    pub initial_state_manifest: PathBuf,
    pub completion_manifest: PathBuf,
    pub action_ledger: Option<PathBuf>,
    pub action_ledger_manifest: Option<PathBuf>,
}

impl CplgOnlineConfig {
    pub fn validate(&self) -> Result<()> {
        validate_config(self)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Arm {
    Stock,
    Candidate,
}

impl Arm {
    fn for_optimizer(optimizer: OuterOptimizer) -> Result<Self> {
        match optimizer {
            OuterOptimizer::Nesterov => Ok(Self::Stock),
            OuterOptimizer::CplgSgd => Ok(Self::Candidate),
            _ => bail!("CPLG online evidence requires outer optimizer nesterov or cplg-sgd"),
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Stock => "cplg_m1_stock",
            Self::Candidate => "cplg_m1_candidate",
        }
    }

    const fn optimizer_name(self) -> &'static str {
        match self {
            Self::Stock => "nesterov",
            Self::Candidate => "cplg-sgd",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BoundaryIdentity {
    pub commit_sequence: u64,
    pub fragment: usize,
    pub fragment_version: u64,
    pub responder_id: u32,
    pub responder_step: u64,
    pub responder_c_steps: u32,
    pub responder_tokens: u64,
    pub weight_f64_bits: u64,
    pub cplg: CplgStepStats,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct LedgerSeal {
    pub rows: u64,
    pub head: String,
    pub sha256: String,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct WriterAccounting {
    pub dropped: u64,
    pub abandoned: u64,
    pub pending: u64,
    pub errors: u64,
}

#[derive(Clone, Debug)]
pub struct Completion {
    pub interval_start_ns: u64,
    pub interval_end_ns: u64,
    pub interval_ns: u64,
    pub event_tape_sha256: String,
    pub final_checkpoint_sha256: String,
    pub ledger_head: Option<String>,
    pub ledger_rows: Option<u64>,
    pub writer: WriterAccounting,
}

pub struct CplgOnlineEvidence {
    config: CplgOnlineConfig,
    arm: Arm,
    layout_sha256: String,
    initial_state_sha256: String,
    clock_origin: Instant,
    interval_start_ns: Option<u64>,
    commits_observed: u64,
    commits_per_fragment: [u64; EXPECTED_FRAGMENTS],
    ledger: Option<CplgActionLedgerWriter>,
}

impl CplgOnlineEvidence {
    /// Bind the receipts to producer-derived identities from the initialized
    /// live HELLO layout and exact f32 global state. A supplied layout digest
    /// is only an expectation and can never replace the producer derivation.
    pub fn prepare(
        config: CplgOnlineConfig,
        optimizer: OuterOptimizer,
        state: &GlobalState,
        expected_layout_sha256: Option<&str>,
        clock_origin: Instant,
    ) -> Result<Self> {
        validate_config(&config)?;
        let arm = Arm::for_optimizer(optimizer)?;
        match (
            arm,
            config.action_ledger.as_ref(),
            config.action_ledger_manifest.as_ref(),
        ) {
            (Arm::Stock, None, None) | (Arm::Candidate, Some(_), Some(_)) => {}
            (Arm::Stock, _, _) => bail!("stock arm must not configure a CPLG action ledger"),
            (Arm::Candidate, _, _) => {
                bail!("candidate arm requires both CPLG action-ledger paths")
            }
        }
        if state.wire_dtype != crate::protocol::DTYPE_F32 {
            bail!("CPLG online evidence requires the live HELLO wire dtype f32");
        }
        if state.layout.fragments.len() != EXPECTED_FRAGMENTS {
            bail!(
                "CPLG online evidence requires {EXPECTED_FRAGMENTS} live HELLO fragments, got {}",
                state.layout.fragments.len()
            );
        }
        let layout_sha256 = state.canonical_layout_sha256();
        validate_sha256("producer-derived layout_sha256", &layout_sha256)?;
        if let Some(expected) = expected_layout_sha256 {
            validate_sha256("supplied layout_sha256 expectation", expected)?;
            if expected != layout_sha256 {
                bail!(
                    "supplied layout_sha256 does not match the producer-derived live HELLO layout"
                );
            }
        }
        let initial_state_sha256 = state.canonical_initial_state_sha256()?;
        validate_sha256(
            "producer-derived initial_state_sha256",
            &initial_state_sha256,
        )?;
        let initial = value_from_pairs([
            ("arm", Value::String(arm.name().to_owned())),
            ("expected_commits", Value::from(EXPECTED_COMMITS)),
            ("fragments", Value::from(EXPECTED_FRAGMENTS as u64)),
            (
                "initial_state_sha256",
                Value::String(initial_state_sha256.clone()),
            ),
            ("layout_sha256", Value::String(layout_sha256.clone())),
            (
                "outer_optimizer",
                Value::String(arm.optimizer_name().to_owned()),
            ),
            (
                "run_config_sha256",
                Value::String(config.run_config_sha256.clone()),
            ),
            ("run_id", Value::String(config.run_id.clone())),
            ("schema_version", Value::from(1)),
            ("source_commit", Value::String(config.source_commit.clone())),
        ]);
        write_json_with_sidecar(&config.initial_state_manifest, &initial)?;
        Ok(Self {
            config,
            arm,
            layout_sha256,
            initial_state_sha256,
            clock_origin,
            interval_start_ns: None,
            commits_observed: 0,
            commits_per_fragment: [0; EXPECTED_FRAGMENTS],
            ledger: None,
        })
    }

    /// Open the measured interval immediately before the scheduler loop. The
    /// candidate ledger is created after taking the timestamp, so all live
    /// evidence-writing cost is inside the candidate interval.
    pub fn open_interval(&mut self) -> Result<()> {
        if self.interval_start_ns.is_some() {
            bail!("CPLG online evidence interval is already open");
        }
        self.interval_start_ns = Some(monotonic_ns(&self.clock_origin)?);
        if self.arm == Arm::Candidate {
            let path = self
                .config
                .action_ledger
                .as_deref()
                .context("candidate action-ledger path is absent")?;
            self.ledger = Some(CplgActionLedgerWriter::create(path)?);
        }
        Ok(())
    }

    pub fn record_boundary(
        &mut self,
        state: &mut GlobalState,
        boundary: BoundaryIdentity,
    ) -> Result<()> {
        if self.interval_start_ns.is_none() {
            bail!("CPLG online evidence boundary arrived before the interval opened");
        }
        let expected_sequence = self
            .commits_observed
            .checked_add(1)
            .context("CPLG online commit counter overflow")?;
        if boundary.commit_sequence != expected_sequence {
            bail!(
                "CPLG online commit sequence expected {expected_sequence}, got {}",
                boundary.commit_sequence
            );
        }
        let expected_fragment = self.commits_observed as usize % EXPECTED_FRAGMENTS;
        if boundary.fragment != expected_fragment {
            bail!(
                "CPLG online fragment order expected {expected_fragment}, got {}",
                boundary.fragment
            );
        }
        if boundary.fragment_version != boundary.commit_sequence {
            bail!(
                "CPLG online fragment version {} does not equal commit sequence {}",
                boundary.fragment_version,
                boundary.commit_sequence
            );
        }
        if self.arm == Arm::Candidate {
            enforce_action_hash_contract(&boundary.cplg)?;
            let row = build_ledger_row(
                &self.config,
                &self.layout_sha256,
                &self.initial_state_sha256,
                self.commits_observed,
                &boundary,
            )?;
            let ledger = self
                .ledger
                .as_mut()
                .context("candidate action-ledger writer is absent")?;
            let seal = ledger.submit(row)?;
            state.set_cplg_action_ledger_evidence(&seal.head, seal.rows)?;
        }
        self.commits_observed = expected_sequence;
        self.commits_per_fragment[boundary.fragment] = self.commits_per_fragment[boundary.fragment]
            .checked_add(1)
            .context("CPLG online fragment commit counter overflow")?;
        Ok(())
    }

    /// Close event/action evidence durably, write the candidate ledger
    /// manifest, take the exact monotonic endpoint, then publish completion.
    /// The completion receipt itself is intentionally outside its measured
    /// interval because the endpoint it records must already exist.
    pub fn finish(
        mut self,
        state: &GlobalState,
        event_tape_sha256: String,
        final_checkpoint_sha256: String,
    ) -> Result<Completion> {
        validate_sha256("event_tape_sha256", &event_tape_sha256)?;
        validate_sha256("final_checkpoint_sha256", &final_checkpoint_sha256)?;
        if self.commits_observed != EXPECTED_COMMITS {
            bail!(
                "CPLG online completion expected {EXPECTED_COMMITS} commits, got {}",
                self.commits_observed
            );
        }
        if self.commits_per_fragment != [8, 8, 8, 8] {
            bail!(
                "CPLG online completion requires commits_per_fragment [8,8,8,8], got {:?}",
                self.commits_per_fragment
            );
        }
        if state.global_step != EXPECTED_COMMITS {
            bail!(
                "CPLG online completion expected final global step {EXPECTED_COMMITS}, got {}",
                state.global_step
            );
        }

        let mut ledger_head = None;
        let mut ledger_rows = None;
        let writer = if self.arm == Arm::Candidate {
            let seal = self
                .ledger
                .take()
                .context("candidate action-ledger writer is absent")?
                .finish(EXPECTED_COMMITS)?;
            let (checkpoint_head, checkpoint_rows) = state.cplg_action_ledger_evidence();
            if checkpoint_head != seal.head || checkpoint_rows != seal.rows {
                bail!("final checkpoint CPLG ledger evidence differs from the closed ledger");
            }
            let accounting = WriterAccounting::default();
            let manifest = value_from_pairs([
                ("arm", Value::String(self.arm.name().to_owned())),
                (
                    "event_tape_sha256",
                    Value::String(event_tape_sha256.clone()),
                ),
                ("expected_commits", Value::from(EXPECTED_COMMITS)),
                (
                    "final_checkpoint_sha256",
                    Value::String(final_checkpoint_sha256.clone()),
                ),
                ("fragments", Value::from(EXPECTED_FRAGMENTS as u64)),
                (
                    "initial_state_sha256",
                    Value::String(self.initial_state_sha256.clone()),
                ),
                ("layout_sha256", Value::String(self.layout_sha256.clone())),
                ("ledger_head", Value::String(seal.head.clone())),
                ("ledger_rows", Value::from(seal.rows)),
                (
                    "outer_optimizer",
                    Value::String(self.arm.optimizer_name().to_owned()),
                ),
                (
                    "run_config_sha256",
                    Value::String(self.config.run_config_sha256.clone()),
                ),
                ("run_id", Value::String(self.config.run_id.clone())),
                ("schema_version", Value::from(1)),
                (
                    "source_commit",
                    Value::String(self.config.source_commit.clone()),
                ),
                (
                    "unresolved_tail",
                    Value::from(state.cplg_unresolved_tail_count() as u64),
                ),
                ("writer_abandoned", Value::from(accounting.abandoned)),
                ("writer_dropped", Value::from(accounting.dropped)),
                ("writer_errors", Value::from(accounting.errors)),
                ("writer_pending", Value::from(accounting.pending)),
            ]);
            write_json_with_sidecar(
                self.config
                    .action_ledger_manifest
                    .as_deref()
                    .context("candidate action-ledger manifest path is absent")?,
                &manifest,
            )?;
            ledger_head = Some(seal.head);
            ledger_rows = Some(seal.rows);
            accounting
        } else {
            WriterAccounting::default()
        };

        let interval_start_ns = self
            .interval_start_ns
            .context("CPLG online completion requested before interval open")?;
        let interval_end_ns = monotonic_ns(&self.clock_origin)?;
        let interval_ns = interval_end_ns
            .checked_sub(interval_start_ns)
            .context("monotonic CPLG online interval moved backwards")?;
        let completion = value_from_pairs([
            ("arm", Value::String(self.arm.name().to_owned())),
            ("commits_observed", Value::from(self.commits_observed)),
            (
                "commits_per_fragment",
                Value::Array(
                    self.commits_per_fragment
                        .iter()
                        .copied()
                        .map(Value::from)
                        .collect(),
                ),
            ),
            (
                "event_tape_sha256",
                Value::String(event_tape_sha256.clone()),
            ),
            (
                "final_checkpoint_sha256",
                Value::String(final_checkpoint_sha256.clone()),
            ),
            ("final_global_step", Value::from(state.global_step)),
            ("interval_end_ns", Value::from(interval_end_ns)),
            ("interval_ns", Value::from(interval_ns)),
            ("interval_start_ns", Value::from(interval_start_ns)),
            (
                "ledger_head",
                ledger_head
                    .clone()
                    .map(Value::String)
                    .unwrap_or(Value::Null),
            ),
            (
                "ledger_rows",
                ledger_rows.map(Value::from).unwrap_or(Value::Null),
            ),
            ("raw_training_tokens", Value::from(RAW_TRAINING_TOKENS)),
            ("run_id", Value::String(self.config.run_id.clone())),
            ("schema_version", Value::from(1)),
            ("terminal_local_steps", Value::from(TERMINAL_LOCAL_STEPS)),
            ("writer_abandoned", Value::from(writer.abandoned)),
            ("writer_dropped", Value::from(writer.dropped)),
            ("writer_errors", Value::from(writer.errors)),
            ("writer_pending", Value::from(writer.pending)),
        ]);
        write_json_with_sidecar(&self.config.completion_manifest, &completion)?;
        Ok(Completion {
            interval_start_ns,
            interval_end_ns,
            interval_ns,
            event_tape_sha256,
            final_checkpoint_sha256,
            ledger_head,
            ledger_rows,
            writer,
        })
    }
}

fn validate_config(config: &CplgOnlineConfig) -> Result<()> {
    if config.run_id.trim().is_empty() {
        bail!("CPLG online run_id must not be empty");
    }
    validate_sha256("run_config_sha256", &config.run_config_sha256)?;
    validate_lower_hex("source_commit", &config.source_commit, 40)?;
    for (name, path) in [
        (
            "initial-state manifest",
            Some(&config.initial_state_manifest),
        ),
        ("completion manifest", Some(&config.completion_manifest)),
        ("action ledger", config.action_ledger.as_ref()),
        (
            "action-ledger manifest",
            config.action_ledger_manifest.as_ref(),
        ),
    ] {
        if let Some(path) = path {
            if !path.is_absolute() {
                bail!(
                    "CPLG online {name} path must be absolute: {}",
                    path.display()
                );
            }
        }
    }
    if config.action_ledger.is_some() != config.action_ledger_manifest.is_some() {
        bail!("CPLG action-ledger paths must be configured together");
    }
    let mut paths = vec![&config.initial_state_manifest, &config.completion_manifest];
    if let Some(path) = &config.action_ledger {
        paths.push(path);
    }
    if let Some(path) = &config.action_ledger_manifest {
        paths.push(path);
    }
    for (index, path) in paths.iter().enumerate() {
        if paths[..index].iter().any(|prior| *prior == *path) {
            bail!("CPLG online evidence paths must be distinct");
        }
    }
    Ok(())
}

fn validate_lower_hex(name: &str, value: &str, length: usize) -> Result<()> {
    if value.len() != length
        || !value
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
    {
        bail!("{name} must be exactly {length} lowercase hexadecimal characters");
    }
    Ok(())
}

fn validate_sha256(name: &str, value: &str) -> Result<()> {
    validate_lower_hex(name, value, 64)
}

fn monotonic_ns(origin: &Instant) -> Result<u64> {
    u64::try_from(origin.elapsed().as_nanos())
        .context("monotonic clock reading exceeds u64 nanoseconds")
}

fn value_from_pairs<const N: usize>(pairs: [(&str, Value); N]) -> Value {
    let object: Map<String, Value> = pairs
        .into_iter()
        .map(|(name, value)| (name.to_owned(), value))
        .collect();
    Value::Object(object)
}

fn exact_f32_value(value: Option<f64>, absent: Option<Value>) -> Result<Value> {
    match value {
        None => match absent {
            Some(value) => Ok(value),
            None => Ok(serde_json::to_value(0.0f32)?),
        },
        Some(value) => {
            let rounded = value as f32;
            if !value.is_finite() || !rounded.is_finite() {
                bail!("CPLG ledger contains a non-finite f32 diagnostic");
            }
            Ok(serde_json::to_value(rounded)?)
        }
    }
}

fn digest_string(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn digest_value(bytes: &[u8; 32], optional: bool) -> Value {
    if optional && bytes.iter().all(|byte| *byte == 0) {
        Value::Null
    } else {
        Value::String(hex_bytes(bytes))
    }
}

fn hex_bytes(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        result.push(HEX[(byte >> 4) as usize] as char);
        result.push(HEX[(byte & 0x0f) as usize] as char);
    }
    result
}

fn build_ledger_row(
    config: &CplgOnlineConfig,
    layout_sha256: &str,
    initial_state_sha256: &str,
    row_index: u64,
    boundary: &BoundaryIdentity,
) -> Result<BTreeMap<String, Value>> {
    let weight_identity = value_from_pairs([
        ("c_steps", Value::from(boundary.responder_c_steps)),
        ("c_tokens", Value::from(boundary.responder_tokens)),
        ("responder_id", Value::from(boundary.responder_id)),
        (
            "weight_f64_bits",
            Value::String(format!("{:016x}", boundary.weight_f64_bits)),
        ),
    ]);
    let weight_identity_sha256 = digest_string(&serde_json::to_vec(&weight_identity)?);
    let stats = boundary.cplg;
    let mut row = BTreeMap::new();
    row.insert(
        "commit_sequence".to_owned(),
        Value::from(boundary.commit_sequence),
    );
    row.insert(
        "cplg_action_sha256".to_owned(),
        digest_value(&stats.action_sha256, false),
    );
    row.insert(
        "cplg_candidate_sha256".to_owned(),
        digest_value(&stats.candidate_sha256, true),
    );
    row.insert(
        "cplg_coherence".to_owned(),
        exact_f32_value(stats.coherence, None)?,
    );
    row.insert(
        "cplg_interlock_open".to_owned(),
        Value::Bool(stats.interlock_open),
    );
    row.insert("cplg_phi".to_owned(), exact_f32_value(stats.phi, None)?);
    row.insert(
        "cplg_previous_stock_sha256".to_owned(),
        digest_value(&stats.previous_stock_sha256, true),
    );
    row.insert(
        "cplg_previous_tangent_sha256".to_owned(),
        digest_value(&stats.previous_tangent_sha256, true),
    );
    row.insert(
        "cplg_previous_theta".to_owned(),
        exact_f32_value(stats.previous_theta, None)?,
    );
    row.insert(
        "cplg_reason".to_owned(),
        Value::String(stats.reason.as_str().to_owned()),
    );
    row.insert("cplg_rho".to_owned(), exact_f32_value(stats.rho, None)?);
    row.insert(
        "cplg_score_count".to_owned(),
        Value::from(stats.interlock_score_count),
    );
    row.insert(
        "cplg_shadow_score".to_owned(),
        exact_f32_value(stats.resolved_shadow_score, Some(Value::Null))?,
    );
    row.insert(
        "cplg_state_cleared".to_owned(),
        Value::Bool(stats.state_cleared),
    );
    row.insert(
        "cplg_stock_sha256".to_owned(),
        digest_value(&stats.stock_sha256, false),
    );
    row.insert("cplg_theta".to_owned(), exact_f32_value(stats.theta, None)?);
    row.insert(
        "cplg_transported_tangent_sha256".to_owned(),
        digest_value(&stats.transported_tangent_sha256, true),
    );
    row.insert(
        "cplg_used_nonstock".to_owned(),
        Value::Bool(stats.used_nonstock),
    );
    row.insert("fragment".to_owned(), Value::from(boundary.fragment));
    row.insert(
        "fragment_version".to_owned(),
        Value::from(boundary.fragment_version),
    );
    row.insert(
        "initial_state_sha256".to_owned(),
        Value::String(initial_state_sha256.to_owned()),
    );
    row.insert(
        "layout_sha256".to_owned(),
        Value::String(layout_sha256.to_owned()),
    );
    row.insert(
        "responder_step".to_owned(),
        Value::from(boundary.responder_step),
    );
    row.insert(
        "responder_tokens".to_owned(),
        Value::from(boundary.responder_tokens),
    );
    row.insert("row_index".to_owned(), Value::from(row_index));
    row.insert("run_id".to_owned(), Value::String(config.run_id.clone()));
    row.insert(
        "run_config_sha256".to_owned(),
        Value::String(config.run_config_sha256.clone()),
    );
    row.insert("schema_version".to_owned(), Value::from(1));
    row.insert(
        "source_commit".to_owned(),
        Value::String(config.source_commit.clone()),
    );
    row.insert(
        "weight_identity_sha256".to_owned(),
        Value::String(weight_identity_sha256),
    );
    Ok(row)
}

fn is_f32_field(name: &str) -> bool {
    matches!(
        name,
        "cplg_rho"
            | "cplg_theta"
            | "cplg_previous_theta"
            | "cplg_coherence"
            | "cplg_phi"
            | "cplg_shadow_score"
    )
}

fn canonical_row_bytes<'a, I>(pairs: I) -> Result<Vec<u8>>
where
    I: IntoIterator<Item = (&'a str, &'a Value)>,
{
    let mut pairs: Vec<(&str, &Value)> = pairs.into_iter().collect();
    pairs.sort_by_key(|(name, _)| *name);
    let mut bytes = Vec::new();
    bytes.push(b'{');
    for (index, (name, value)) in pairs.into_iter().enumerate() {
        if index > 0 {
            bytes.push(b',');
        }
        serde_json::to_writer(&mut bytes, name)?;
        bytes.push(b':');
        if is_f32_field(name) && !value.is_null() {
            let as_f64 = value
                .as_f64()
                .with_context(|| format!("CPLG ledger {name} must be numeric"))?;
            let as_f32 = as_f64 as f32;
            if !as_f64.is_finite() || !as_f32.is_finite() {
                bail!("CPLG ledger {name} is not a finite f32");
            }
            serde_json::to_writer(&mut bytes, &as_f32)?;
        } else {
            serde_json::to_writer(&mut bytes, value)?;
        }
    }
    bytes.push(b'}');
    Ok(bytes)
}

fn enforce_action_hash_contract(stats: &CplgStepStats) -> Result<()> {
    let stock = hex_bytes(&stats.stock_sha256);
    let action = hex_bytes(&stats.action_sha256);
    let candidate = hex_bytes(&stats.candidate_sha256);
    if stock == ZERO_SHA256 || action == ZERO_SHA256 {
        bail!("CPLG boundary is missing stock/action identity");
    }
    if stats.reason == CplgReason::CandidateSelected {
        if !stats.used_nonstock
            || candidate == ZERO_SHA256
            || action != candidate
            || action == stock
        {
            bail!("candidate_selected violates the frozen CPLG action-hash contract");
        }
    } else if stats.used_nonstock || action != stock {
        bail!(
            "non-candidate CPLG reason {} violates exact stock fallback",
            stats.reason.as_str()
        );
    }
    Ok(())
}

fn enforce_json_action_hash_contract(row: &Map<String, Value>) -> Result<()> {
    let reason = required_string(row, "cplg_reason")?;
    if ![
        "not_active",
        "stock_warmup",
        "phase_warmup",
        "interlock_closed",
        "candidate_selected",
        "degenerate_stock",
        "nonacute_turn",
        "invalid_geometry",
        "invalid_shadow_score",
        "zero_or_rounded_phase",
    ]
    .contains(&reason)
    {
        bail!("CPLG ledger contains unknown reason {reason:?}");
    }
    let used = row
        .get("cplg_used_nonstock")
        .and_then(Value::as_bool)
        .context("CPLG ledger cplg_used_nonstock must be boolean")?;
    let stock = required_digest(row, "cplg_stock_sha256")?;
    let action = required_digest(row, "cplg_action_sha256")?;
    if reason == "candidate_selected" {
        let candidate = required_digest(row, "cplg_candidate_sha256")?;
        if !used || action != candidate || action == stock {
            bail!("candidate_selected ledger row violates the action-hash contract");
        }
    } else if used || action != stock {
        bail!("fallback ledger row violates the exact-stock action-hash contract");
    }
    Ok(())
}

fn required_string<'a>(row: &'a Map<String, Value>, name: &str) -> Result<&'a str> {
    row.get(name)
        .and_then(Value::as_str)
        .with_context(|| format!("CPLG ledger {name} must be a string"))
}

fn required_digest<'a>(row: &'a Map<String, Value>, name: &str) -> Result<&'a str> {
    let value = required_string(row, name)?;
    validate_sha256(name, value)?;
    Ok(value)
}

struct CplgActionLedgerWriter {
    path: PathBuf,
    writer: BufWriter<std::fs::File>,
    rows: u64,
    head: String,
    accepted: u64,
    completed: u64,
    errors: u64,
}

impl CplgActionLedgerWriter {
    fn create(path: &Path) -> Result<Self> {
        require_parent(path)?;
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .with_context(|| format!("create fresh CPLG action ledger {}", path.display()))?;
        Ok(Self {
            path: path.to_owned(),
            writer: BufWriter::new(file),
            rows: 0,
            head: ZERO_SHA256.to_owned(),
            accepted: 0,
            completed: 0,
            errors: 0,
        })
    }

    #[cfg(test)]
    fn resume(path: &Path, expected_head: &str, expected_rows: u64) -> Result<Self> {
        let seal = verify_ledger(path, None)?;
        if seal.head != expected_head || seal.rows != expected_rows {
            bail!("resumed CPLG ledger differs from checkpoint head/count");
        }
        let file = std::fs::OpenOptions::new().append(true).open(path)?;
        Ok(Self {
            path: path.to_owned(),
            writer: BufWriter::new(file),
            rows: seal.rows,
            head: seal.head,
            accepted: seal.rows,
            completed: seal.rows,
            errors: 0,
        })
    }

    fn submit(&mut self, mut row: BTreeMap<String, Value>) -> Result<LedgerSeal> {
        self.accepted = self
            .accepted
            .checked_add(1)
            .context("CPLG action-ledger accepted counter overflow")?;
        row.insert(
            "previous_row_sha256".to_owned(),
            Value::String(self.head.clone()),
        );
        let row_without_hash =
            canonical_row_bytes(row.iter().map(|(name, value)| (name.as_str(), value)))?;
        let row_sha256 = digest_string(&row_without_hash);
        row.insert("row_sha256".to_owned(), Value::String(row_sha256.clone()));
        let bytes = canonical_row_bytes(row.iter().map(|(name, value)| (name.as_str(), value)))?;
        let result = (|| -> Result<()> {
            self.writer.write_all(&bytes)?;
            self.writer.write_all(b"\n")?;
            // Make the ledger no less recoverable than the checkpoint head
            // that is saved after this method returns.
            self.writer.flush()?;
            self.writer.get_ref().sync_data()?;
            Ok(())
        })();
        if let Err(error) = result {
            self.errors = self.errors.saturating_add(1);
            return Err(error).context("write durable CPLG action-ledger row");
        }
        self.completed = self
            .completed
            .checked_add(1)
            .context("CPLG action-ledger completed counter overflow")?;
        self.rows = self
            .rows
            .checked_add(1)
            .context("CPLG action-ledger row counter overflow")?;
        self.head = row_sha256;
        Ok(LedgerSeal {
            rows: self.rows,
            head: self.head.clone(),
            sha256: String::new(),
        })
    }

    fn finish(mut self, expected_rows: u64) -> Result<LedgerSeal> {
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;
        let pending = self.accepted.saturating_sub(self.completed);
        if self.rows != expected_rows
            || self.accepted != expected_rows
            || self.completed != expected_rows
            || pending != 0
            || self.errors != 0
        {
            bail!(
                "CPLG action-ledger writer did not close exactly: rows={} accepted={} completed={} pending={} errors={}",
                self.rows,
                self.accepted,
                self.completed,
                pending,
                self.errors
            );
        }
        drop(self.writer);
        let seal = verify_ledger(&self.path, Some(expected_rows))?;
        if seal.head != self.head {
            bail!("closed CPLG action-ledger head differs from writer head");
        }
        Ok(seal)
    }
}

pub(crate) fn verify_ledger(path: &Path, expected_rows: Option<u64>) -> Result<LedgerSeal> {
    let mut raw = Vec::new();
    std::fs::File::open(path)
        .with_context(|| format!("open CPLG action ledger {}", path.display()))?
        .read_to_end(&mut raw)?;
    if !raw.is_empty() && !raw.ends_with(b"\n") {
        bail!("CPLG action ledger is not newline-terminated");
    }
    let text = std::str::from_utf8(&raw).context("CPLG action ledger is not UTF-8")?;
    let mut head = ZERO_SHA256.to_owned();
    let mut rows = 0u64;
    for (index, line) in text.lines().enumerate() {
        if line.is_empty() || line.as_bytes().contains(&b'\r') {
            bail!("CPLG action ledger contains a noncanonical empty/CRLF row");
        }
        let value: Value = serde_json::from_str(line)
            .with_context(|| format!("parse CPLG action-ledger row {index}"))?;
        let object = value
            .as_object()
            .context("CPLG action-ledger row must be an object")?;
        let expected_fields: std::collections::BTreeSet<&str> =
            LEDGER_FIELDS.iter().copied().collect();
        let actual_fields: std::collections::BTreeSet<&str> =
            object.keys().map(String::as_str).collect();
        if actual_fields != expected_fields {
            bail!("CPLG action-ledger row {index} fields differ from the closed schema");
        }
        let canonical =
            canonical_row_bytes(object.iter().map(|(name, value)| (name.as_str(), value)))?;
        if canonical != line.as_bytes() {
            bail!("CPLG action-ledger row {index} is not canonical sorted-key JSON");
        }
        let row_index = object
            .get("row_index")
            .and_then(Value::as_u64)
            .context("CPLG action-ledger row_index must be a nonnegative integer")?;
        if row_index != index as u64 {
            bail!("CPLG action-ledger row index is not contiguous");
        }
        let commit_sequence = object
            .get("commit_sequence")
            .and_then(Value::as_u64)
            .context("CPLG action-ledger commit_sequence must be an integer")?;
        if commit_sequence != row_index + 1 {
            bail!("CPLG action-ledger commit sequence is not contiguous and one-based");
        }
        let fragment = object
            .get("fragment")
            .and_then(Value::as_u64)
            .context("CPLG action-ledger fragment must be an integer")?;
        if fragment != row_index % EXPECTED_FRAGMENTS as u64 {
            bail!("CPLG action-ledger fragment order differs from [0,1,2,3] repeated");
        }
        let predecessor = required_digest(object, "previous_row_sha256")?;
        if predecessor != head {
            bail!("CPLG action-ledger predecessor chain is corrupt at row {index}");
        }
        enforce_json_action_hash_contract(object)?;
        for name in [
            "cplg_rho",
            "cplg_theta",
            "cplg_previous_theta",
            "cplg_coherence",
            "cplg_phi",
        ] {
            validate_exact_f32_json(object.get(name), name, false)?;
        }
        validate_exact_f32_json(object.get("cplg_shadow_score"), "cplg_shadow_score", true)?;
        let declared = required_digest(object, "row_sha256")?;
        let mut unhashed = object.clone();
        unhashed.remove("row_sha256");
        let computed = digest_string(&canonical_row_bytes(
            unhashed.iter().map(|(name, value)| (name.as_str(), value)),
        )?);
        if declared != computed {
            bail!("CPLG action-ledger row hash is corrupt at row {index}");
        }
        head = declared.to_owned();
        rows += 1;
    }
    if let Some(expected) = expected_rows {
        if rows != expected {
            bail!("CPLG action ledger expected {expected} rows, got {rows}");
        }
    }
    Ok(LedgerSeal {
        rows,
        head,
        sha256: digest_string(&raw),
    })
}

fn validate_exact_f32_json(value: Option<&Value>, name: &str, nullable: bool) -> Result<()> {
    let Some(value) = value else {
        bail!("CPLG ledger is missing {name}");
    };
    if nullable && value.is_null() {
        return Ok(());
    }
    let number = value
        .as_number()
        .with_context(|| format!("CPLG ledger {name} must be numeric"))?;
    let as_f64 = number
        .as_f64()
        .with_context(|| format!("CPLG ledger {name} is not finite"))?;
    let as_f32 = as_f64 as f32;
    if !as_f64.is_finite() || !as_f32.is_finite() {
        bail!("CPLG ledger {name} is not a finite f32");
    }
    if serde_json::to_string(&as_f32)? != number.to_string() {
        bail!("CPLG ledger {name} is not an exact f32-round-tripped decimal");
    }
    Ok(())
}

pub fn durably_close_and_sha256(path: &Path) -> Result<String> {
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .with_context(|| format!("open evidence file for durable closure {}", path.display()))?;
    file.sync_all()?;
    sha256_file(path)
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn require_parent(path: &Path) -> Result<&Path> {
    let parent = path.parent().context("evidence path has no parent")?;
    if !parent.is_dir() {
        bail!(
            "evidence parent directory does not exist: {}",
            parent.display()
        );
    }
    Ok(parent)
}

fn sidecar_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".sha256");
    PathBuf::from(name)
}

fn write_json_with_sidecar(path: &Path, value: &Value) -> Result<String> {
    let sidecar_path = sidecar_path(path);
    if path.exists() || sidecar_path.exists() {
        bail!(
            "evidence JSON and checksum sidecar must both be fresh: {}",
            path.display()
        );
    }
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    atomic_write_new(path, &bytes)?;
    let digest = digest_string(&bytes);
    let basename = path
        .file_name()
        .and_then(|name| name.to_str())
        .context("evidence JSON basename is not UTF-8")?;
    let sidecar = format!("{digest}  {basename}\n");
    atomic_write_new(&sidecar_path, sidecar.as_bytes())?;
    Ok(digest)
}

fn atomic_write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = require_parent(path)?;
    if path.exists() {
        bail!("evidence output path is not fresh: {}", path.display());
    }
    let basename = path
        .file_name()
        .and_then(|name| name.to_str())
        .context("evidence output basename is not UTF-8")?;
    let temporary = parent.join(format!(".{basename}.tmp"));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .with_context(|| format!("create atomic evidence temporary {}", temporary.display()))?;
    file.write_all(bytes)?;
    file.flush()?;
    file.sync_all()?;
    drop(file);
    std::fs::rename(&temporary, path)?;
    std::fs::File::open(parent)?.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{FragmentInfo, Layout, MERGE_RDA};

    fn test_root(label: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "yeto-cplg-online-{label}-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::remove_dir_all(&root).ok();
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    fn config(root: &Path) -> CplgOnlineConfig {
        CplgOnlineConfig {
            run_id: "exp2-cplg-active-e1-m1-r1-test".to_owned(),
            run_config_sha256: "a".repeat(64),
            source_commit: "b".repeat(40),
            initial_state_manifest: root.join("cplg_online_initial_state.json"),
            completion_manifest: root.join("cplg_online_completion.json"),
            action_ledger: Some(root.join("cplg_action_ledger.jsonl")),
            action_ledger_manifest: Some(root.join("cplg_action_ledger_manifest.json")),
        }
    }

    fn state() -> GlobalState {
        let layout = Layout {
            fragments: (0..EXPECTED_FRAGMENTS)
                .map(|_| FragmentInfo {
                    merge_mode: MERGE_RDA,
                    tensor_numels: vec![2],
                    tensor_shapes: None,
                })
                .collect(),
        };
        let mut state = GlobalState::new(
            layout,
            Some("{\"fragment_pattern\":\"binpack\"}".to_owned()),
            f32::from_bits(0x3e8f_5c29),
            0.0,
            crate::protocol::DTYPE_F32,
        );
        state.outer_optimizer = OuterOptimizer::CplgSgd;
        for fragment in 0..EXPECTED_FRAGMENTS {
            state
                .init_fragment(fragment, vec![fragment as f32, -0.0])
                .unwrap();
        }
        state
    }

    fn hash(label: &str, index: u64) -> [u8; 32] {
        Sha256::digest(format!("{label}-{index}").as_bytes()).into()
    }

    fn stats(index: u64, selected: bool) -> CplgStepStats {
        let stock = hash("stock", index);
        let candidate = hash("candidate", index);
        CplgStepStats {
            rho: Some(0.984_807_7f32 as f64),
            theta: Some(0.174_532_92f32 as f64),
            previous_theta: Some(0.174_532_92f32 as f64),
            coherence: Some(1.0),
            phi: Some(0.174_532_92f32 as f64),
            resolved_shadow_score: (index >= 4).then_some(0.015_192_3f32 as f64),
            interlock_score_count: (index.min(3)) as u8,
            interlock_open: selected,
            used_nonstock: selected,
            state_cleared: false,
            reason: if selected {
                CplgReason::CandidateSelected
            } else {
                CplgReason::InterlockClosed
            },
            stock_sha256: stock,
            previous_stock_sha256: (index > 0)
                .then(|| hash("stock", index - 1))
                .unwrap_or([0; 32]),
            previous_tangent_sha256: (index > 1)
                .then(|| hash("tangent", index - 1))
                .unwrap_or([0; 32]),
            transported_tangent_sha256: (index > 1)
                .then(|| hash("transported", index))
                .unwrap_or([0; 32]),
            candidate_sha256: candidate,
            action_sha256: if selected { candidate } else { stock },
        }
    }

    fn boundary(index: u64) -> BoundaryIdentity {
        BoundaryIdentity {
            commit_sequence: index + 1,
            fragment: index as usize % EXPECTED_FRAGMENTS,
            fragment_version: index + 1,
            responder_id: 0,
            responder_step: (index + 3).min(34),
            responder_c_steps: 4,
            responder_tokens: 512,
            weight_f64_bits: 128.0f64.to_bits(),
            cplg: stats(index, index % 4 == 3),
        }
    }

    fn write_rows(
        writer: &mut CplgActionLedgerWriter,
        config: &CplgOnlineConfig,
        start: u64,
        end: u64,
    ) -> LedgerSeal {
        let mut seal = None;
        for index in start..end {
            seal = Some(submit_row(writer, config, index));
        }
        seal.unwrap()
    }

    fn submit_row(
        writer: &mut CplgActionLedgerWriter,
        config: &CplgOnlineConfig,
        index: u64,
    ) -> LedgerSeal {
        let row = build_ledger_row(
            config,
            &"c".repeat(64),
            &"d".repeat(64),
            index,
            &boundary(index),
        )
        .unwrap();
        writer.submit(row).unwrap()
    }

    fn rewrite_rows(path: &Path, rows: &[Value]) {
        let mut bytes = Vec::new();
        for row in rows {
            bytes.extend_from_slice(serde_json::to_string(row).unwrap().as_bytes());
            bytes.push(b'\n');
        }
        std::fs::write(path, bytes).unwrap();
    }

    fn rehash_row(row: &mut Value) {
        let object = row.as_object_mut().unwrap();
        object.remove("row_sha256");
        let digest = digest_string(&serde_json::to_vec(object).unwrap());
        object.insert("row_sha256".to_owned(), Value::String(digest));
    }

    #[test]
    fn producer_derived_initial_receipt_rejects_supplied_layout_mismatch() {
        let root = test_root("initial");
        let state = state();
        let actual = state.canonical_layout_sha256();
        let error = CplgOnlineEvidence::prepare(
            config(&root),
            OuterOptimizer::CplgSgd,
            &state,
            Some(&"f".repeat(64)),
            Instant::now(),
        )
        .err()
        .unwrap();
        assert!(error.to_string().contains("producer-derived live HELLO"));
        assert!(!root.join("cplg_online_initial_state.json").exists());

        let _evidence = CplgOnlineEvidence::prepare(
            config(&root),
            OuterOptimizer::CplgSgd,
            &state,
            Some(&actual),
            Instant::now(),
        )
        .unwrap();
        let path = root.join("cplg_online_initial_state.json");
        let value: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(value["layout_sha256"], actual);
        assert_eq!(
            value["initial_state_sha256"],
            state.canonical_initial_state_sha256().unwrap()
        );
        assert_eq!(value["fragments"], 4);
        assert_eq!(value["expected_commits"], 32);
        let digest = sha256_file(&path).unwrap();
        assert_eq!(
            std::fs::read_to_string(sidecar_path(&path)).unwrap(),
            format!("{digest}  cplg_online_initial_state.json\n")
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn stock_and_candidate_publish_equal_producer_derived_initial_identities() {
        let stock_root = test_root("matched-stock-initial");
        let candidate_root = test_root("matched-candidate-initial");
        let mut stock_config = config(&stock_root);
        stock_config.action_ledger = None;
        stock_config.action_ledger_manifest = None;
        let candidate_config = config(&candidate_root);
        let mut stock_state = state();
        stock_state.outer_optimizer = OuterOptimizer::Nesterov;
        let candidate_state = state();
        let _stock = CplgOnlineEvidence::prepare(
            stock_config.clone(),
            OuterOptimizer::Nesterov,
            &stock_state,
            None,
            Instant::now(),
        )
        .unwrap();
        let _candidate = CplgOnlineEvidence::prepare(
            candidate_config.clone(),
            OuterOptimizer::CplgSgd,
            &candidate_state,
            None,
            Instant::now(),
        )
        .unwrap();
        let stock: Value =
            serde_json::from_slice(&std::fs::read(stock_config.initial_state_manifest).unwrap())
                .unwrap();
        let candidate: Value = serde_json::from_slice(
            &std::fs::read(candidate_config.initial_state_manifest).unwrap(),
        )
        .unwrap();
        assert_eq!(stock["layout_sha256"], candidate["layout_sha256"]);
        assert_eq!(
            stock["initial_state_sha256"],
            candidate["initial_state_sha256"]
        );
        std::fs::remove_dir_all(stock_root).ok();
        std::fs::remove_dir_all(candidate_root).ok();
    }

    #[test]
    fn uninterrupted_and_resumed_ledger_are_byte_identical() {
        let uninterrupted_root = test_root("ledger-uninterrupted");
        let resumed_root = test_root("ledger-resumed");
        let uninterrupted_config = config(&uninterrupted_root);
        let resumed_config = config(&resumed_root);
        let uninterrupted_path = uninterrupted_config.action_ledger.as_deref().unwrap();
        let resumed_path = resumed_config.action_ledger.as_deref().unwrap();

        let mut uninterrupted = CplgActionLedgerWriter::create(uninterrupted_path).unwrap();
        let mut uninterrupted_state = state();
        for index in 0..EXPECTED_COMMITS {
            let seal = submit_row(&mut uninterrupted, &uninterrupted_config, index);
            uninterrupted_state
                .set_cplg_action_ledger_evidence(&seal.head, seal.rows)
                .unwrap();
        }
        let uninterrupted_seal = uninterrupted.finish(EXPECTED_COMMITS).unwrap();

        let mut first_process = CplgActionLedgerWriter::create(resumed_path).unwrap();
        let mut checkpoint_state = state();
        let mut checkpoint_seal = None;
        for index in 0..13 {
            let seal = submit_row(&mut first_process, &resumed_config, index);
            checkpoint_state
                .set_cplg_action_ledger_evidence(&seal.head, seal.rows)
                .unwrap();
            checkpoint_seal = Some(seal);
        }
        let checkpoint_seal = checkpoint_seal.unwrap();
        checkpoint_state.global_step = 13;
        let checkpoint_path = resumed_root.join("resume.ckpt");
        checkpoint_state.save_checkpoint(&checkpoint_path).unwrap();
        drop(first_process);

        let mut resumed_state = state();
        resumed_state.load_checkpoint(&checkpoint_path).unwrap();
        assert_eq!(
            resumed_state.cplg_action_ledger_evidence(),
            (checkpoint_seal.head.clone(), checkpoint_seal.rows)
        );
        let (checkpoint_head, checkpoint_rows) = resumed_state.cplg_action_ledger_evidence();
        let mut resumed =
            CplgActionLedgerWriter::resume(resumed_path, &checkpoint_head, checkpoint_rows)
                .unwrap();
        for index in 13..EXPECTED_COMMITS {
            let seal = submit_row(&mut resumed, &resumed_config, index);
            resumed_state
                .set_cplg_action_ledger_evidence(&seal.head, seal.rows)
                .unwrap();
        }
        let resumed_seal = resumed.finish(EXPECTED_COMMITS).unwrap();

        assert_eq!(resumed_seal, uninterrupted_seal);
        assert_eq!(
            std::fs::read(resumed_path).unwrap(),
            std::fs::read(uninterrupted_path).unwrap()
        );
        assert_eq!(resumed_seal.rows, 32);
        assert_ne!(resumed_seal.head, ZERO_SHA256);
        uninterrupted_state.global_step = EXPECTED_COMMITS;
        resumed_state.global_step = EXPECTED_COMMITS;
        let uninterrupted_checkpoint = uninterrupted_root.join("final.ckpt");
        let resumed_checkpoint = resumed_root.join("final.ckpt");
        uninterrupted_state
            .save_checkpoint(&uninterrupted_checkpoint)
            .unwrap();
        resumed_state.save_checkpoint(&resumed_checkpoint).unwrap();
        assert_eq!(
            std::fs::read(uninterrupted_checkpoint).unwrap(),
            std::fs::read(resumed_checkpoint).unwrap(),
            "uninterrupted and resumed checkpoint evidence must be byte-identical"
        );
        std::fs::remove_dir_all(uninterrupted_root).ok();
        std::fs::remove_dir_all(resumed_root).ok();
    }

    #[test]
    fn ledger_verifier_rejects_row_hash_chain_and_action_corruption() {
        let root = test_root("ledger-corruption");
        let config = config(&root);
        let original = config.action_ledger.as_deref().unwrap();
        let mut writer = CplgActionLedgerWriter::create(original).unwrap();
        write_rows(&mut writer, &config, 0, EXPECTED_COMMITS);
        writer.finish(EXPECTED_COMMITS).unwrap();
        let original_rows: Vec<Value> = std::fs::read_to_string(original)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();

        let row_hash_path = root.join("row-hash.jsonl");
        let mut row_hash_rows = original_rows.clone();
        row_hash_rows[0]["responder_tokens"] = Value::from(513);
        rewrite_rows(&row_hash_path, &row_hash_rows);
        assert!(verify_ledger(&row_hash_path, Some(32)).is_err());

        let chain_path = root.join("chain.jsonl");
        let mut chain_rows = original_rows.clone();
        chain_rows[1]["previous_row_sha256"] = Value::String(ZERO_SHA256.to_owned());
        rehash_row(&mut chain_rows[1]);
        rewrite_rows(&chain_path, &chain_rows);
        assert!(verify_ledger(&chain_path, Some(32))
            .unwrap_err()
            .to_string()
            .contains("predecessor chain"));

        let action_path = root.join("action.jsonl");
        let mut action_rows = original_rows;
        action_rows[0]["cplg_used_nonstock"] = Value::Bool(true);
        rehash_row(&mut action_rows[0]);
        rewrite_rows(&action_path, &action_rows);
        assert!(verify_ledger(&action_path, Some(32))
            .unwrap_err()
            .to_string()
            .contains("action-hash contract"));
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn closure_writes_exact_receipts_and_zero_writer_accounting() {
        let root = test_root("closure");
        let config = config(&root);
        let mut state = state();
        let clock = Instant::now();
        let mut evidence = CplgOnlineEvidence::prepare(
            config.clone(),
            OuterOptimizer::CplgSgd,
            &state,
            None,
            clock,
        )
        .unwrap();
        evidence.open_interval().unwrap();
        for index in 0..EXPECTED_COMMITS {
            evidence
                .record_boundary(&mut state, boundary(index))
                .unwrap();
        }
        state.global_step = EXPECTED_COMMITS;
        let checkpoint = root.join("state.ckpt");
        state.save_checkpoint(&checkpoint).unwrap();
        let checkpoint_sha256 = sha256_file(&checkpoint).unwrap();
        let event_tape = root.join("tape.jsonl");
        std::fs::write(&event_tape, b"event\n").unwrap();
        let event_tape_sha256 = durably_close_and_sha256(&event_tape).unwrap();
        let completion = evidence
            .finish(&state, event_tape_sha256.clone(), checkpoint_sha256.clone())
            .unwrap();
        assert_eq!(completion.writer, WriterAccounting::default());
        assert_eq!(completion.ledger_rows, Some(32));
        assert!(completion.interval_end_ns >= completion.interval_start_ns);
        assert_eq!(
            completion.interval_ns,
            completion.interval_end_ns - completion.interval_start_ns
        );

        let completion_path = root.join("cplg_online_completion.json");
        let receipt: Value =
            serde_json::from_slice(&std::fs::read(&completion_path).unwrap()).unwrap();
        assert_eq!(receipt["terminal_local_steps"], 34);
        assert_eq!(receipt["raw_training_tokens"], 4_352);
        assert_eq!(receipt["commits_observed"], 32);
        assert_eq!(
            receipt["commits_per_fragment"],
            serde_json::json!([8, 8, 8, 8])
        );
        assert_eq!(receipt["event_tape_sha256"], event_tape_sha256);
        assert_eq!(receipt["final_checkpoint_sha256"], checkpoint_sha256);
        for field in [
            "writer_dropped",
            "writer_abandoned",
            "writer_pending",
            "writer_errors",
        ] {
            assert_eq!(receipt[field], 0);
        }
        let manifest_path = root.join("cplg_action_ledger_manifest.json");
        let manifest: Value =
            serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
        assert_eq!(manifest["ledger_rows"], 32);
        assert_eq!(manifest["ledger_head"], completion.ledger_head.unwrap());
        assert_eq!(manifest["unresolved_tail"], 0);
        assert_eq!(
            verify_ledger(config.action_ledger.as_deref().unwrap(), Some(32))
                .unwrap()
                .rows,
            32
        );
        for path in [completion_path, manifest_path] {
            let digest = sha256_file(&path).unwrap();
            assert_eq!(
                std::fs::read_to_string(sidecar_path(&path)).unwrap(),
                format!(
                    "{digest}  {}\n",
                    path.file_name().unwrap().to_str().unwrap()
                )
            );
        }
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn ledger_close_rejects_a_non_32_row_denominator() {
        let root = test_root("short-close");
        let config = config(&root);
        let path = config.action_ledger.as_deref().unwrap();
        let mut writer = CplgActionLedgerWriter::create(path).unwrap();
        write_rows(&mut writer, &config, 0, 31);
        assert!(writer.finish(EXPECTED_COMMITS).is_err());
        std::fs::remove_dir_all(root).ok();
    }
}
