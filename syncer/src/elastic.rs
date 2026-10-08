//! Inter-island scheduling, `elastic` mode only (change rl-inter-island-scheduling,
//! design D-S1..S4, D-S6 parts). The legacy (default) mode never touches this
//! module: no new message type is registered by the legacy server, and the
//! legacy merge/quorum/grace/strict path in `server.rs` is unchanged.
//!
//! Reference implementation mirrored here: Python
//! `yeto/rl/engine/island_ledger.CrossIslandLedger` (elastic branch).
//!
//! Status: frames, HMAC, fencing and the coordinator state machine are
//! implemented and unit-tested; wiring them into the live TCP server loop is
//! not done yet (the server refuses to start in elastic mode).

use std::collections::BTreeMap;

use anyhow::{bail, ensure, Context, Result};
use sha2::{Digest, Sha256};

// ---------------------------------------------------------------------------
// Mode and parameters
// ---------------------------------------------------------------------------

pub const MODE_LEGACY: &str = "legacy";
pub const MODE_ELASTIC: &str = "elastic";

/// Defaults: design.md P4, marked [待真机校准] (not yet hardware-calibrated).
pub const DEFAULT_QUORUM_THETA: f64 = 0.75;
pub const DEFAULT_CARRY_GAMMA: f64 = 0.5;
pub const DEFAULT_Q_MIN: u32 = 1;
pub const DEFAULT_MAX_CARRY_LAG: u32 = 2;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElasticParams {
    pub quorum_theta: f64,
    pub carry_gamma: f64,
    pub soft_deadline_s: u64,
    pub q_min: u32,
    pub max_carry_lag: u32,
}

impl ElasticParams {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.quorum_theta > 0.0 && self.quorum_theta <= 1.0,
            "quorum_theta must be in (0, 1], got {}",
            self.quorum_theta
        );
        ensure!(
            (0.0..=1.0).contains(&self.carry_gamma),
            "carry_gamma must be in [0, 1], got {}",
            self.carry_gamma
        );
        ensure!(self.q_min >= 1, "q_min must be >= 1");
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum IslandSchedulingMode {
    /// Existing strict syncer behavior; contributes nothing to the contract
    /// hash so every pre-existing semantic profile hash is unchanged.
    #[default]
    Legacy,
    Elastic(ElasticParams),
}

const ISLAND_SCHEDULING_DOMAIN: &[u8] = b"yeto-syncer-island-scheduling-v1\0";

impl IslandSchedulingMode {
    pub fn parse(
        mode: &str,
        quorum_theta: f64,
        carry_gamma: f64,
        soft_deadline_s: u64,
        q_min: u32,
        max_carry_lag: u32,
    ) -> Result<Self> {
        match mode {
            MODE_LEGACY => Ok(Self::Legacy),
            MODE_ELASTIC => {
                let p = ElasticParams {
                    quorum_theta,
                    carry_gamma,
                    soft_deadline_s,
                    q_min,
                    max_carry_lag,
                };
                p.validate()?;
                Ok(Self::Elastic(p))
            }
            other => bail!("--island-scheduling-mode must be legacy|elastic, got {other:?}"),
        }
    }

    /// Bytes appended to the semantic profile encoding. Legacy appends
    /// nothing (byte-identical legacy hash). Elastic appends the domain, the
    /// field name `island_scheduling_mode`, the mode string and all P4
    /// parameters, so a legacy HELLO can never match an elastic server and
    /// vice versa. Layout (little endian):
    /// domain | u8 len | "island_scheduling_mode" | u8 len | "elastic" |
    /// f64 quorum_theta | f64 carry_gamma | u64 soft_deadline_s | u32 q_min |
    /// u32 max_carry_lag
    pub fn encode_contract(&self, encoded: &mut Vec<u8>) {
        let Self::Elastic(p) = self else { return };
        encoded.extend_from_slice(ISLAND_SCHEDULING_DOMAIN);
        let field = b"island_scheduling_mode";
        encoded.push(field.len() as u8);
        encoded.extend_from_slice(field);
        encoded.push(MODE_ELASTIC.len() as u8);
        encoded.extend_from_slice(MODE_ELASTIC.as_bytes());
        encoded.extend_from_slice(&p.quorum_theta.to_bits().to_le_bytes());
        encoded.extend_from_slice(&p.carry_gamma.to_bits().to_le_bytes());
        encoded.extend_from_slice(&p.soft_deadline_s.to_le_bytes());
        encoded.extend_from_slice(&p.q_min.to_le_bytes());
        encoded.extend_from_slice(&p.max_carry_lag.to_le_bytes());
    }
}

// ---------------------------------------------------------------------------
// HMAC-SHA256 (RFC 2104), only for the elastic message types
// ---------------------------------------------------------------------------

pub const HMAC_LEN: usize = 32;

pub fn hmac_sha256(key: &[u8], msg: &[u8]) -> [u8; 32] {
    const B: usize = 64;
    let mut k = [0u8; B];
    if key.len() > B {
        k[..32].copy_from_slice(&Sha256::digest(key));
    } else {
        k[..key.len()].copy_from_slice(key);
    }
    let mut inner = Sha256::new();
    inner.update(k.map(|b| b ^ 0x36));
    inner.update(msg);
    let ih = inner.finalize();
    let mut outer = Sha256::new();
    outer.update(k.map(|b| b ^ 0x5c));
    outer.update(ih);
    outer.finalize().into()
}

fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// Append the MAC over `msg_type || body`.
pub fn seal(key: &[u8], msg_type: u8, mut body: Vec<u8>) -> Vec<u8> {
    let mut mac_input = Vec::with_capacity(body.len() + 1);
    mac_input.push(msg_type);
    mac_input.extend_from_slice(&body);
    body.extend_from_slice(&hmac_sha256(key, &mac_input));
    body
}

/// Verify and strip the trailing MAC; returns the authenticated body.
pub fn open<'a>(key: &[u8], msg_type: u8, payload: &'a [u8]) -> Result<&'a [u8]> {
    ensure!(payload.len() >= HMAC_LEN, "elastic frame shorter than its HMAC");
    let (body, mac) = payload.split_at(payload.len() - HMAC_LEN);
    let mut mac_input = Vec::with_capacity(body.len() + 1);
    mac_input.push(msg_type);
    mac_input.extend_from_slice(body);
    ensure!(
        ct_eq(&hmac_sha256(key, &mac_input), mac),
        "elastic frame HMAC mismatch"
    );
    Ok(body)
}

// ---------------------------------------------------------------------------
// Frames (D-S1, D-S2, D-S4). Every frame starts with syncer_epoch (u64).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum ElasticMsg {
    Join {
        syncer_epoch: u64,
        island_id: u32,
        incarnation: u64,
        capacity: f64,
    },
    JoinAck {
        syncer_epoch: u64,
        learner_slot: u32,
        membership_epoch: u64,
        base_version: u64,
        policy_hash: [u8; 32],
        catch_up: bool,
    },
    Leave {
        syncer_epoch: u64,
        island_id: u32,
        reason: u8,
    },
    LeaseHeartbeat {
        syncer_epoch: u64,
        island_id: u32,
        membership_epoch: u64,
        inner_step: u64,
        round_wall_s: f64,
    },
    /// P3: the coordinator validates, deduplicates and judges the index
    /// entry; samples stay in object storage and are never forwarded.
    SampleIndex {
        syncer_epoch: u64,
        entry: SampleIndexEntry,
    },
    /// Coordinator: the run reached total_steps; islands should exit 0.
    Finished {
        syncer_epoch: u64,
        outer_version: u64,
        policy_hash: [u8; 32],
    },
    /// Coordinator reply to SAMPLE_INDEX.
    SampleVerdict {
        syncer_epoch: u64,
        verdict: u8,
        reason: String,
        outer_lag: u64,
    },
    DeltaReady {
        syncer_epoch: u64,
        island_id: u32,
        base_version: u64,
        c_tokens: u64,
        c_steps: u64,
    },
    ElasticInit {
        syncer_epoch: u64,
        island_id: u32,
        params: Vec<f32>,
    },
    /// `update` is θ − base as sent by the island (same sign convention as
    /// PUSH_FRAGMENT); the server negates it into an outer gradient.
    DeltaTensor {
        syncer_epoch: u64,
        island_id: u32,
        base_version: u64,
        c_tokens: u64,
        c_steps: u64,
        update: Vec<f32>,
    },
    ElasticBase {
        syncer_epoch: u64,
        outer_version: u64,
        params: Vec<f32>,
    },
}

fn put_f32s(b: &mut Vec<u8>, v: &[f32]) {
    b.extend_from_slice(&(v.len() as u64).to_le_bytes());
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
}

fn get_f32s(r: &mut crate::protocol::Reader) -> Result<Vec<f32>> {
    let n = usize::try_from(r.u64()?).context("tensor length")?;
    let bytes = r.take(n.checked_mul(4).context("tensor length overflow")?)?;
    let mut out = Vec::new();
    crate::protocol::decode_tensor(crate::protocol::DTYPE_F32, bytes, &mut out)?;
    Ok(out)
}

pub const LEAVE_REASON_REQUESTED: u8 = 0;
pub const LEAVE_REASON_LEASE_EXPIRED: u8 = 1;

impl ElasticMsg {
    pub fn msg_type(&self) -> u8 {
        use crate::protocol::*;
        match self {
            Self::Join { .. } => MSG_JOIN,
            Self::JoinAck { .. } => MSG_JOIN_ACK,
            Self::Leave { .. } => MSG_LEAVE,
            Self::LeaseHeartbeat { .. } => MSG_LEASE_HEARTBEAT,
            Self::SampleIndex { .. } => MSG_SAMPLE_INDEX,
            Self::SampleVerdict { .. } => MSG_SAMPLE_VERDICT,
            Self::Finished { .. } => MSG_FINISHED,
            Self::DeltaReady { .. } => MSG_DELTA_READY,
            Self::ElasticInit { .. } => MSG_ELASTIC_INIT,
            Self::DeltaTensor { .. } => MSG_DELTA_TENSOR,
            Self::ElasticBase { .. } => MSG_ELASTIC_BASE,
        }
    }

    pub fn syncer_epoch(&self) -> u64 {
        match self {
            Self::Join { syncer_epoch, .. }
            | Self::JoinAck { syncer_epoch, .. }
            | Self::Leave { syncer_epoch, .. }
            | Self::LeaseHeartbeat { syncer_epoch, .. }
            | Self::SampleIndex { syncer_epoch, .. }
            | Self::SampleVerdict { syncer_epoch, .. }
            | Self::Finished { syncer_epoch, .. }
            | Self::DeltaReady { syncer_epoch, .. }
            | Self::ElasticInit { syncer_epoch, .. }
            | Self::DeltaTensor { syncer_epoch, .. }
            | Self::ElasticBase { syncer_epoch, .. } => *syncer_epoch,
        }
    }

    /// Encode body and append the HMAC. Returns (msg_type, payload).
    pub fn encode(&self, key: &[u8]) -> (u8, Vec<u8>) {
        let mut b = Vec::new();
        b.extend_from_slice(&self.syncer_epoch().to_le_bytes());
        match self {
            Self::Join { island_id, incarnation, capacity, .. } => {
                b.extend_from_slice(&island_id.to_le_bytes());
                b.extend_from_slice(&incarnation.to_le_bytes());
                b.extend_from_slice(&capacity.to_bits().to_le_bytes());
            }
            Self::JoinAck {
                learner_slot,
                membership_epoch,
                base_version,
                policy_hash,
                catch_up,
                ..
            } => {
                b.extend_from_slice(&learner_slot.to_le_bytes());
                b.extend_from_slice(&membership_epoch.to_le_bytes());
                b.extend_from_slice(&base_version.to_le_bytes());
                b.extend_from_slice(policy_hash);
                b.push(u8::from(*catch_up));
            }
            Self::Leave { island_id, reason, .. } => {
                b.extend_from_slice(&island_id.to_le_bytes());
                b.push(*reason);
            }
            Self::LeaseHeartbeat { island_id, membership_epoch, inner_step, round_wall_s, .. } => {
                b.extend_from_slice(&island_id.to_le_bytes());
                b.extend_from_slice(&membership_epoch.to_le_bytes());
                b.extend_from_slice(&inner_step.to_le_bytes());
                b.extend_from_slice(&round_wall_s.to_bits().to_le_bytes());
            }
            Self::SampleIndex { entry, .. } => entry.encode(&mut b),
            Self::Finished { outer_version, policy_hash, .. } => {
                b.extend_from_slice(&outer_version.to_le_bytes());
                b.extend_from_slice(policy_hash);
            }
            Self::SampleVerdict { verdict, reason, outer_lag, .. } => {
                b.push(*verdict);
                put_str(&mut b, reason);
                b.extend_from_slice(&outer_lag.to_le_bytes());
            }
            Self::DeltaReady { island_id, base_version, c_tokens, c_steps, .. } => {
                b.extend_from_slice(&island_id.to_le_bytes());
                b.extend_from_slice(&base_version.to_le_bytes());
                b.extend_from_slice(&c_tokens.to_le_bytes());
                b.extend_from_slice(&c_steps.to_le_bytes());
            }
            Self::ElasticInit { island_id, params, .. } => {
                b.extend_from_slice(&island_id.to_le_bytes());
                put_f32s(&mut b, params);
            }
            Self::DeltaTensor { island_id, base_version, c_tokens, c_steps, update, .. } => {
                b.extend_from_slice(&island_id.to_le_bytes());
                b.extend_from_slice(&base_version.to_le_bytes());
                b.extend_from_slice(&c_tokens.to_le_bytes());
                b.extend_from_slice(&c_steps.to_le_bytes());
                put_f32s(&mut b, update);
            }
            Self::ElasticBase { outer_version, params, .. } => {
                b.extend_from_slice(&outer_version.to_le_bytes());
                put_f32s(&mut b, params);
            }
        }
        let t = self.msg_type();
        (t, seal(key, t, b))
    }

    /// Authenticate and decode. Unknown types are an error.
    pub fn decode(key: &[u8], msg_type: u8, payload: &[u8]) -> Result<Self> {
        use crate::protocol::*;
        let body = open(key, msg_type, payload)?;
        let mut r = Reader(body);
        let syncer_epoch = r.u64()?;
        let f64_ = |r: &mut Reader| -> Result<f64> { Ok(f64::from_bits(r.u64()?)) };
        let hash = |r: &mut Reader| -> Result<[u8; 32]> {
            Ok(r.take(32)?.try_into().context("policy hash")?)
        };
        let flag = |r: &mut Reader| -> Result<bool> {
            match r.u8()? {
                0 => Ok(false),
                1 => Ok(true),
                v => bail!("invalid bool byte {v}"),
            }
        };
        let msg = match msg_type {
            MSG_JOIN => Self::Join {
                syncer_epoch,
                island_id: r.u32()?,
                incarnation: r.u64()?,
                capacity: f64_(&mut r)?,
            },
            MSG_JOIN_ACK => Self::JoinAck {
                syncer_epoch,
                learner_slot: r.u32()?,
                membership_epoch: r.u64()?,
                base_version: r.u64()?,
                policy_hash: hash(&mut r)?,
                catch_up: flag(&mut r)?,
            },
            MSG_LEAVE => Self::Leave { syncer_epoch, island_id: r.u32()?, reason: r.u8()? },
            MSG_LEASE_HEARTBEAT => Self::LeaseHeartbeat {
                syncer_epoch,
                island_id: r.u32()?,
                membership_epoch: r.u64()?,
                inner_step: r.u64()?,
                round_wall_s: f64_(&mut r)?,
            },
            MSG_SAMPLE_INDEX => {
                Self::SampleIndex { syncer_epoch, entry: SampleIndexEntry::decode(&mut r)? }
            }
            MSG_FINISHED => Self::Finished {
                syncer_epoch,
                outer_version: r.u64()?,
                policy_hash: hash(&mut r)?,
            },
            MSG_SAMPLE_VERDICT => Self::SampleVerdict {
                syncer_epoch,
                verdict: r.u8()?,
                reason: get_str(&mut r)?,
                outer_lag: r.u64()?,
            },
            MSG_DELTA_READY => Self::DeltaReady {
                syncer_epoch,
                island_id: r.u32()?,
                base_version: r.u64()?,
                c_tokens: r.u64()?,
                c_steps: r.u64()?,
            },
            MSG_ELASTIC_INIT => {
                Self::ElasticInit { syncer_epoch, island_id: r.u32()?, params: get_f32s(&mut r)? }
            }
            MSG_DELTA_TENSOR => Self::DeltaTensor {
                syncer_epoch,
                island_id: r.u32()?,
                base_version: r.u64()?,
                c_tokens: r.u64()?,
                c_steps: r.u64()?,
                update: get_f32s(&mut r)?,
            },
            MSG_ELASTIC_BASE => Self::ElasticBase {
                syncer_epoch,
                outer_version: r.u64()?,
                params: get_f32s(&mut r)?,
            },
            other => bail!("message type {other} is not an elastic message"),
        };
        ensure!(r.remaining() == 0, "trailing bytes in elastic frame type {msg_type}");
        if let Self::Join { capacity, .. } = &msg {
            ensure!(capacity.is_finite() && *capacity > 0.0, "JOIN capacity must be > 0");
        }
        Ok(msg)
    }
}

// ---------------------------------------------------------------------------
// Coordinator state machine (mirrors CrossIslandLedger, elastic branch)
// ---------------------------------------------------------------------------

/// Syncer merge weight, same formula as merge.rs (tokens^2 / steps).
pub fn merge_weight(c_tokens: u64, c_steps: u64) -> f64 {
    if c_tokens == 0 || c_steps == 0 {
        return 0.0;
    }
    (c_tokens as f64).powi(2) / c_steps as f64
}

#[derive(Debug, Clone, PartialEq)]
struct Member {
    joined_at: u64,
    catch_up: bool,
    last_heartbeat: f64,
    capacity: f64,
    /// D-S5: EMA of the island-reported round wall time (LEASE_HEARTBEAT).
    round_wall_ema_s: Option<f64>,
    /// D-S5: arrived-on-time flags of the most recent steps (newest last).
    arrivals: std::collections::VecDeque<bool>,
}

impl Member {
    fn new(joined_at: u64, catch_up: bool, now: f64, capacity: f64) -> Self {
        Self {
            joined_at,
            catch_up,
            last_heartbeat: now,
            capacity,
            round_wall_ema_s: None,
            arrivals: Default::default(),
        }
    }
}

/// D-S5 EMA smoothing factor and arrival history length.
pub const ROUND_WALL_EMA_ALPHA: f64 = 0.2;
pub const ARRIVAL_HISTORY: usize = 8;

/// Sample staleness bound (design: user ruling 2026-10-07, at most 2 outer
/// steps; same as Python StalenessPolicy.max_outer_lag). Not a CLI flag yet.
pub const MAX_SAMPLE_OUTER_LAG: u64 = 2;

pub const SAMPLE_INDEX_SCHEMA: &str = "yeto.rl.sample-index/v1";
pub const SAMPLE_URI_SCHEMES: [&str; 3] = ["s3://", "modal-volume://", "nebius-os://"];
pub const VERDICT_ACCEPT: u8 = 0;
pub const VERDICT_ACCEPT_IS: u8 = 1;
pub const VERDICT_REJECT: u8 = 2;

pub fn verdict_name(v: u8) -> &'static str {
    match v {
        VERDICT_ACCEPT => "ACCEPT",
        VERDICT_ACCEPT_IS => "ACCEPT_IS",
        _ => "REJECT",
    }
}

fn put_str(b: &mut Vec<u8>, v: &str) {
    b.extend_from_slice(&(v.len() as u32).to_le_bytes());
    b.extend_from_slice(v.as_bytes());
}

fn get_str(r: &mut crate::protocol::Reader) -> Result<String> {
    let n = r.u32()? as usize;
    String::from_utf8(r.take(n)?.to_vec()).context("string is not utf-8")
}

/// Mirrors Python `yeto/rl/engine/sample_pool.SampleIndexEntry`
/// (schema yeto.rl.sample-index/v1). Wire order (little endian; str = u32
/// length + utf-8): island_id u32 | outer_version u64 | inner_step u64 |
/// policy_hash [32] (raw sha256; Python hex string decoded) | group_id str |
/// prompt_id str | n u64 | uri str | size_bytes u64 | sha256 str (64 hex) |
/// has_behavior_logprob u8 | advantage_included u8 | created_at f64 | schema str
#[derive(Debug, Clone, PartialEq)]
pub struct SampleIndexEntry {
    pub island_id: u32,
    pub outer_version: u64,
    pub inner_step: u64,
    pub policy_hash: [u8; 32],
    pub group_id: String,
    pub prompt_id: String,
    pub n: u64,
    pub uri: String,
    pub size_bytes: u64,
    pub sha256: String,
    pub has_behavior_logprob: bool,
    pub advantage_included: bool,
    pub created_at: f64,
    pub schema: String,
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn json_str(v: &str) -> String {
    let mut o = String::from("\"");
    for c in v.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

impl SampleIndexEntry {
    pub fn encode(&self, b: &mut Vec<u8>) {
        b.extend_from_slice(&self.island_id.to_le_bytes());
        b.extend_from_slice(&self.outer_version.to_le_bytes());
        b.extend_from_slice(&self.inner_step.to_le_bytes());
        b.extend_from_slice(&self.policy_hash);
        put_str(b, &self.group_id);
        put_str(b, &self.prompt_id);
        b.extend_from_slice(&self.n.to_le_bytes());
        put_str(b, &self.uri);
        b.extend_from_slice(&self.size_bytes.to_le_bytes());
        put_str(b, &self.sha256);
        b.push(u8::from(self.has_behavior_logprob));
        b.push(u8::from(self.advantage_included));
        b.extend_from_slice(&self.created_at.to_bits().to_le_bytes());
        put_str(b, &self.schema);
    }

    pub fn decode(r: &mut crate::protocol::Reader) -> Result<Self> {
        let flag = |r: &mut crate::protocol::Reader| -> Result<bool> {
            match r.u8()? {
                0 => Ok(false),
                1 => Ok(true),
                v => bail!("invalid bool byte {v}"),
            }
        };
        Ok(Self {
            island_id: r.u32()?,
            outer_version: r.u64()?,
            inner_step: r.u64()?,
            policy_hash: r.take(32)?.try_into()?,
            group_id: get_str(r)?,
            prompt_id: get_str(r)?,
            n: r.u64()?,
            uri: get_str(r)?,
            size_bytes: r.u64()?,
            sha256: get_str(r)?,
            has_behavior_logprob: flag(r)?,
            advantage_included: flag(r)?,
            created_at: f64::from_bits(r.u64()?),
            schema: get_str(r)?,
        })
    }

    /// Same checks as the Python `__post_init__` (sha256 additionally must
    /// be lowercase hex).
    pub fn validate(&self) -> Result<()> {
        ensure!(self.schema == SAMPLE_INDEX_SCHEMA, "schema {:?} != {SAMPLE_INDEX_SCHEMA}", self.schema);
        ensure!(
            SAMPLE_URI_SCHEMES.iter().any(|p| self.uri.starts_with(p)),
            "uri {:?} is not one of {SAMPLE_URI_SCHEMES:?}",
            self.uri
        );
        ensure!(self.n >= 1, "n >= 1 required");
        ensure!(
            self.sha256.len() == 64 && self.sha256.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)),
            "sha256 must be 64 hex chars"
        );
        ensure!(self.advantage_included, "group rule: the producer must include the group advantage");
        ensure!(!self.group_id.is_empty(), "group_id must not be empty");
        Ok(())
    }

    /// JSON with Python's field names (island_id as the decimal string).
    pub fn to_json(&self) -> String {
        format!(
            "{{\"advantage_included\":{},\"created_at\":{},\"group_id\":{},\"has_behavior_logprob\":{},\"inner_step\":{},\"island_id\":{},\"n\":{},\"outer_version\":{},\"policy_hash\":\"{}\",\"prompt_id\":{},\"schema\":{},\"sha256\":{},\"size_bytes\":{},\"uri\":{}}}",
            self.advantage_included,
            f64_json(self.created_at),
            json_str(&self.group_id),
            self.has_behavior_logprob,
            self.inner_step,
            json_str(&self.island_id.to_string()),
            self.n,
            self.outer_version,
            hex(&self.policy_hash),
            json_str(&self.prompt_id),
            json_str(&self.schema),
            json_str(&self.sha256),
            self.size_bytes,
            json_str(&self.uri),
        )
    }
}

/// One accepted-into-index record (verdict ACCEPT or ACCEPT_IS).
#[derive(Debug, Clone, PartialEq)]
pub struct IndexedSample {
    pub entry: SampleIndexEntry,
    pub verdict: u8,
    pub reason: &'static str,
    pub outer_lag: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Delta {
    pub island_id: u32,
    pub base_version: u64,
    pub c_tokens: u64,
    pub c_steps: u64,
}

/// Tape records (JSONL kinds match the Python ledger's `kind` strings).
#[derive(Debug, Clone, PartialEq)]
pub enum TapeEvent {
    PoolJoin { island_id: u32, catch_up: bool, base_version: u64, membership_epoch: u64 },
    PoolLeave {
        island_id: u32,
        reason: u8,
        dropped_uncommitted: Option<Delta>,
        membership_epoch: u64,
    },
    DeltaAccepted { island_id: u32 },
    DeltaCarriedOver { island_id: u32, base: u64, lag: u64, discount: f64 },
    DeltaRejected { island_id: u32, reason: &'static str },
    RoundIdle { arrived: usize, cap_arrived: f64, cap_total: f64 },
    SampleIndexed { island_id: u32, group_id: String, verdict: u8, reason: &'static str, outer_lag: u64, duplicate: bool },
    OuterStep {
        base_version: u64,
        arrived: usize,
        cap_arrived: f64,
        cap_total: f64,
        timed_out: bool,
        /// (island, base_version, weight before normalization); carried
        /// entries have base_version < this step's base_version.
        raw_weights: Vec<(u32, u64, f64)>,
        /// Same entries normalized to sum 1 (what merge_avg applies).
        weights: Vec<(u32, u64, f64)>,
        absent: Vec<u32>,
    },
}

fn delta_json(d: &Delta) -> String {
    format!(
        "{{\"island_id\":{},\"outer_version\":{},\"c_tokens\":{},\"c_steps\":{}}}",
        d.island_id, d.base_version, d.c_tokens, d.c_steps
    )
}

fn f64_json(v: f64) -> String {
    if v.is_finite() { format!("{v:?}") } else { "null".into() }
}

impl TapeEvent {
    /// One JSONL record; `kind` strings match the Python ledger.
    pub fn to_json(&self, syncer_epoch: u64) -> String {
        let body = match self {
            Self::PoolJoin { island_id, catch_up, base_version, membership_epoch } => format!(
                "\"island_id\":{island_id},\"catch_up\":{catch_up},\"base_version\":{base_version},\"membership_epoch\":{membership_epoch}"
            ),
            Self::PoolLeave { island_id, reason, dropped_uncommitted, membership_epoch } => format!(
                "\"island_id\":{island_id},\"reason\":\"{}\",\"dropped_uncommitted\":{},\"membership_epoch\":{membership_epoch}",
                if *reason == LEAVE_REASON_LEASE_EXPIRED { "lease_expired" } else { "requested" },
                dropped_uncommitted.as_ref().map(delta_json).unwrap_or_else(|| "null".into())
            ),
            Self::DeltaAccepted { island_id } => format!("\"island_id\":{island_id}"),
            Self::SampleIndexed { island_id, group_id, verdict, reason, outer_lag, duplicate } => format!(
                "\"island_id\":{island_id},\"group_id\":{},\"verdict\":\"{}\",\"reason\":\"{reason}\",\"outer_lag\":{outer_lag},\"duplicate\":{duplicate}",
                json_str(group_id),
                verdict_name(*verdict)
            ),
            Self::DeltaCarriedOver { island_id, base, lag, discount } => format!(
                "\"island_id\":{island_id},\"base\":{base},\"lag\":{lag},\"discount\":{}",
                f64_json(*discount)
            ),
            Self::DeltaRejected { island_id, reason } => {
                format!("\"island_id\":{island_id},\"reason\":\"{reason}\"")
            }
            Self::RoundIdle { arrived, cap_arrived, cap_total } => format!(
                "\"arrived\":{arrived},\"cap_arrived\":{},\"cap_total\":{}",
                f64_json(*cap_arrived),
                f64_json(*cap_total)
            ),
            Self::OuterStep { base_version, arrived, cap_arrived, cap_total, timed_out, raw_weights, weights, absent } => {
                let fmt = |v: &Vec<(u32, u64, f64)>| -> String {
                    v.iter().map(|(i, b, w)| format!("[{i},{b},{}]", f64_json(*w))).collect::<Vec<_>>().join(",")
                };
                let w = fmt(raw_weights);
                let nw = fmt(weights);
                let a: Vec<String> = absent.iter().map(u32::to_string).collect();
                format!(
                    "\"base_version\":{base_version},\"arrived\":{arrived},\"cap_arrived\":{},\"cap_total\":{},\"timed_out\":{timed_out},\"raw_weights\":[{}],\"weights\":[{}],\"absent\":[{}]",
                    f64_json(*cap_arrived),
                    f64_json(*cap_total),
                    w,
                    nw,
                    a.join(",")
                )
            }
        };
        format!("{{\"kind\":\"{}\",\"syncer_epoch\":{syncer_epoch},{body}}}", self.kind())
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Self::PoolJoin { .. } => "pool_join",
            Self::PoolLeave { .. } => "pool_leave",
            Self::DeltaAccepted { .. } => "delta_accepted",
            Self::DeltaCarriedOver { .. } => "delta_carried_over",
            Self::DeltaRejected { .. } => "delta_rejected",
            Self::RoundIdle { .. } => "round_idle",
            Self::SampleIndexed { .. } => "sample_indexed",
            Self::OuterStep { .. } => "outer_step",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StepPlan {
    pub base_version: u64,
    /// Normalized merge weights over current + carried deltas.
    pub weights: Vec<(u32, u64, f64)>,
}

pub struct ElasticCoordinator {
    pub params: ElasticParams,
    pub lease_s: f64,
    pub syncer_epoch: u64,
    pub membership_epoch: u64,
    pub outer_version: u64,
    members: BTreeMap<u32, Member>,
    pending: BTreeMap<u32, Delta>,
    carried: BTreeMap<u32, (Delta, u64)>,
    pub tape: Vec<TapeEvent>,
    /// Ledger of uncommitted deltas dropped by LEAVE / lease expiry
    /// (persisted by the D-S6 checkpoint).
    pub dropped: Vec<Delta>,
    /// Published policy hash per outer version (sha256 of the f32 base);
    /// set by the server after each step. Empty in control-plane-only runs.
    pub published: BTreeMap<u64, [u8; 32]>,
    /// Sample index keyed by (island, group_id); never forwarded.
    pub sample_index: BTreeMap<(u32, String), IndexedSample>,
}

const COORD_STATE_MAGIC: &[u8; 8] = b"YELCST1\0";

fn put_delta(b: &mut Vec<u8>, d: &Delta) {
    b.extend_from_slice(&d.island_id.to_le_bytes());
    b.extend_from_slice(&d.base_version.to_le_bytes());
    b.extend_from_slice(&d.c_tokens.to_le_bytes());
    b.extend_from_slice(&d.c_steps.to_le_bytes());
}

fn get_delta(r: &mut crate::protocol::Reader) -> Result<Delta> {
    Ok(Delta { island_id: r.u32()?, base_version: r.u64()?, c_tokens: r.u64()?, c_steps: r.u64()? })
}

impl ElasticCoordinator {
    fn observe_round_wall(&mut self, island_id: u32, round_wall_s: f64) {
        if let Some(m) = self.members.get_mut(&island_id) {
            if round_wall_s.is_finite() && round_wall_s > 0.0 {
                m.round_wall_ema_s = Some(match m.round_wall_ema_s {
                    None => round_wall_s,
                    Some(e) => e + ROUND_WALL_EMA_ALPHA * (round_wall_s - e),
                });
            }
        }
    }

    /// P3 index judgement relative to the current outer version (the
    /// consumer is unknown at index time). Mirrors the ledger's `judge`
    /// minus consumer-specific rules. Returns (verdict, reason, outer_lag);
    /// errors only for malformed entries. Accepted entries are indexed;
    /// a byte-identical resubmission is idempotent.
    pub fn index_sample(&mut self, e: &SampleIndexEntry) -> Result<(u8, &'static str, u64)> {
        e.validate()?;
        let key = (e.island_id, e.group_id.clone());
        if let Some(prev) = self.sample_index.get(&key) {
            ensure!(prev.entry == *e, "group {} of island {} already indexed with different content", e.group_id, e.island_id);
            let (v, r, l) = (prev.verdict, prev.reason, prev.outer_lag);
            self.tape.push(TapeEvent::SampleIndexed { island_id: e.island_id, group_id: e.group_id.clone(), verdict: v, reason: r, outer_lag: l, duplicate: true });
            return Ok((v, r, l));
        }
        let lag = self.outer_version.checked_sub(e.outer_version);
        let (verdict, reason) = if !self.is_member(e.island_id) {
            (VERDICT_REJECT, "producer_not_member")
        } else if lag.is_none() || !self.published.contains_key(&e.outer_version) {
            (VERDICT_REJECT, "unknown_version")
        } else if self.published[&e.outer_version] != e.policy_hash {
            (VERDICT_REJECT, "policy_hash_mismatch")
        } else if lag == Some(0) {
            (VERDICT_ACCEPT, "same_version")
        } else if lag.unwrap() > MAX_SAMPLE_OUTER_LAG {
            (VERDICT_REJECT, "outer_lag_exceeded")
        } else if !e.has_behavior_logprob {
            (VERDICT_REJECT, "missing_behavior_logprob")
        } else {
            (VERDICT_ACCEPT_IS, "stale_within_bound")
        };
        let outer_lag = lag.unwrap_or(0);
        if verdict != VERDICT_REJECT {
            self.sample_index.insert(key, IndexedSample { entry: e.clone(), verdict, reason, outer_lag });
        }
        self.tape.push(TapeEvent::SampleIndexed { island_id: e.island_id, group_id: e.group_id.clone(), verdict, reason, outer_lag, duplicate: false });
        Ok((verdict, reason, outer_lag))
    }

    /// D-S5 status snapshot (JSON, schema yeto.syncer.elastic-status/v1).
    pub fn status_json(&self, now: f64, soft_deadline_remaining_s: f64, state: &str) -> String {
        let (cap_arrived, cap_total) = self.capacity_arrived();
        let islands: Vec<String> = self
            .members
            .iter()
            .map(|(id, m)| {
                let arrivals: Vec<&str> = m.arrivals.iter().map(|a| if *a { "true" } else { "false" }).collect();
                format!(
                    "{}:{{\"capacity\":{},\"joined_at\":{},\"catch_up\":{},\"round_wall_ema_s\":{},\"lease_remaining_s\":{},\"arrival_history\":[{}],\"pending\":{},\"carried_over_lag\":{}}}",
                    json_str(&id.to_string()),
                    f64_json(m.capacity),
                    m.joined_at,
                    m.catch_up && m.joined_at == self.outer_version,
                    m.round_wall_ema_s.map(f64_json).unwrap_or_else(|| "null".into()),
                    f64_json((self.lease_s - (now - m.last_heartbeat)).max(0.0)),
                    arrivals.join(","),
                    self.pending.contains_key(id),
                    self.carried.get(id).map(|(_, l)| l.to_string()).unwrap_or_else(|| "null".into()),
                )
            })
            .collect();
        format!(
            "{{\"schema\":\"yeto.syncer.elastic-status/v1\",\"state\":{},\"syncer_epoch\":{},\"membership_epoch\":{},\"outer_version\":{},\"policy_hash\":{},\"cap_arrived\":{},\"cap_total\":{},\"soft_deadline_remaining_s\":{},\"pending_count\":{},\"carried_over_count\":{},\"dropped_uncommitted_count\":{},\"sample_index_count\":{},\"islands\":{{{}}}}}",
            json_str(state),
            self.syncer_epoch,
            self.membership_epoch,
            self.outer_version,
            self.published.get(&self.outer_version).map(|h| format!("\"{}\"", hex(h))).unwrap_or_else(|| "null".into()),
            f64_json(cap_arrived),
            f64_json(cap_total),
            f64_json(soft_deadline_remaining_s.max(0.0)),
            self.pending.len(),
            self.carried.len(),
            self.dropped.len(),
            self.sample_index.len(),
            islands.join(","),
        )
    }

    /// D-S6: serialize membership, epochs, carried-over and dropped ledger.
    /// Pending current-round deltas are not persisted (islands resend);
    /// the elastic contract encoding is embedded and checked on restore.
    pub fn encode_state(&self) -> Vec<u8> {
        let mut b = COORD_STATE_MAGIC.to_vec();
        let mut contract = Vec::new();
        IslandSchedulingMode::Elastic(self.params).encode_contract(&mut contract);
        b.extend_from_slice(&(contract.len() as u32).to_le_bytes());
        b.extend_from_slice(&contract);
        b.extend_from_slice(&self.syncer_epoch.to_le_bytes());
        b.extend_from_slice(&self.membership_epoch.to_le_bytes());
        b.extend_from_slice(&self.outer_version.to_le_bytes());
        b.extend_from_slice(&(self.members.len() as u32).to_le_bytes());
        for (id, m) in &self.members {
            b.extend_from_slice(&id.to_le_bytes());
            b.extend_from_slice(&m.joined_at.to_le_bytes());
            b.push(u8::from(m.catch_up));
            b.extend_from_slice(&m.capacity.to_bits().to_le_bytes());
        }
        b.extend_from_slice(&(self.carried.len() as u32).to_le_bytes());
        for (d, lag) in self.carried.values() {
            put_delta(&mut b, d);
            b.extend_from_slice(&lag.to_le_bytes());
        }
        b.extend_from_slice(&(self.dropped.len() as u32).to_le_bytes());
        for d in &self.dropped {
            put_delta(&mut b, d);
        }
        b
    }

    /// Restore from `encode_state`. Members get a fresh lease starting at
    /// `now`; the caller decides the new syncer_epoch (saved + 1).
    pub fn decode_state(params: ElasticParams, lease_s: f64, bytes: &[u8], now: f64) -> Result<Self> {
        let mut r = crate::protocol::Reader(bytes);
        ensure!(r.take(8)? == COORD_STATE_MAGIC, "not an elastic coordinator state");
        let n = r.u32()? as usize;
        let mut expected = Vec::new();
        IslandSchedulingMode::Elastic(params).encode_contract(&mut expected);
        ensure!(r.take(n)? == expected.as_slice(), "checkpoint elastic parameters differ from this launch");
        let mut c = Self::new(params, lease_s, r.u64()?)?;
        c.membership_epoch = r.u64()?;
        c.outer_version = r.u64()?;
        for _ in 0..r.u32()? {
            let id = r.u32()?;
            let joined_at = r.u64()?;
            let catch_up = r.u8()? != 0;
            let capacity = f64::from_bits(r.u64()?);
            c.members.insert(id, Member::new(joined_at, catch_up, now, capacity));
        }
        for _ in 0..r.u32()? {
            let d = get_delta(&mut r)?;
            let lag = r.u64()?;
            c.carried.insert(d.island_id, (d, lag));
        }
        for _ in 0..r.u32()? {
            c.dropped.push(get_delta(&mut r)?);
        }
        ensure!(r.remaining() == 0, "trailing bytes in elastic coordinator state");
        Ok(c)
    }

    pub fn new(params: ElasticParams, lease_s: f64, syncer_epoch: u64) -> Result<Self> {
        params.validate()?;
        Ok(Self {
            params,
            lease_s,
            syncer_epoch,
            membership_epoch: 0,
            outer_version: 0,
            members: BTreeMap::new(),
            pending: BTreeMap::new(),
            carried: BTreeMap::new(),
            tape: Vec::new(),
            dropped: Vec::new(),
            published: BTreeMap::new(),
            sample_index: BTreeMap::new(),
        })
    }

    /// P9 fencing: refuse anything stamped by an older coordinator incarnation.
    pub fn check_fence(&self, msg_epoch: u64) -> Result<()> {
        ensure!(
            msg_epoch >= self.syncer_epoch,
            "fenced: syncer_epoch {msg_epoch} < {}",
            self.syncer_epoch
        );
        Ok(())
    }

    pub fn member_count(&self) -> usize {
        self.members.len()
    }

    pub fn is_member(&self, island_id: u32) -> bool {
        self.members.contains_key(&island_id)
    }

    pub fn join(&mut self, island_id: u32, capacity: f64, now: f64) -> Result<bool> {
        ensure!(!self.members.contains_key(&island_id), "{island_id} is already a member");
        ensure!(capacity.is_finite() && capacity > 0.0, "capacity must be > 0");
        // Founding round (nothing published yet) is not catch-up.
        let catch_up = self.outer_version > 0;
        self.members.insert(
            island_id,
            Member::new(self.outer_version, catch_up, now, capacity),
        );
        self.membership_epoch += 1;
        self.tape.push(TapeEvent::PoolJoin {
            island_id,
            catch_up,
            base_version: self.outer_version,
            membership_epoch: self.membership_epoch,
        });
        Ok(catch_up)
    }

    pub fn leave(&mut self, island_id: u32, reason: u8) -> Result<()> {
        ensure!(self.members.remove(&island_id).is_some(), "{island_id} is not a member");
        let mut dropped = self.pending.remove(&island_id);
        let carried = self.carried.remove(&island_id);
        if dropped.is_none() {
            dropped = carried.map(|(d, _)| d);
        }
        self.dropped.extend(dropped);
        self.membership_epoch += 1;
        self.tape.push(TapeEvent::PoolLeave {
            island_id,
            reason,
            dropped_uncommitted: dropped,
            membership_epoch: self.membership_epoch,
        });
        Ok(())
    }

    pub fn heartbeat(&mut self, island_id: u32, now: f64) -> Result<()> {
        let m = self
            .members
            .get_mut(&island_id)
            .with_context(|| format!("{island_id} is not a member (rejoin required)"))?;
        m.last_heartbeat = now;
        Ok(())
    }

    pub fn expire_leases(&mut self, now: f64) -> Vec<u32> {
        let gone: Vec<u32> = self
            .members
            .iter()
            .filter(|(_, m)| now - m.last_heartbeat > self.lease_s)
            .map(|(&i, _)| i)
            .collect();
        for &i in &gone {
            self.leave(i, LEAVE_REASON_LEASE_EXPIRED).expect("member present");
        }
        gone
    }

    /// Apply one authenticated elastic message (fenced first).
    pub fn apply(&mut self, msg: &ElasticMsg, now: f64) -> Result<Option<ElasticMsg>> {
        self.check_fence(msg.syncer_epoch())?;
        match msg {
            ElasticMsg::Join { island_id, capacity, .. } => {
                let catch_up = self.join(*island_id, *capacity, now)?;
                Ok(Some(ElasticMsg::JoinAck {
                    syncer_epoch: self.syncer_epoch,
                    learner_slot: *island_id,
                    membership_epoch: self.membership_epoch,
                    base_version: self.outer_version,
                    policy_hash: self.published.get(&self.outer_version).copied().unwrap_or([0; 32]),
                    catch_up,
                }))
            }
            ElasticMsg::Leave { island_id, reason, .. } => {
                self.leave(*island_id, *reason)?;
                Ok(None)
            }
            ElasticMsg::LeaseHeartbeat { island_id, round_wall_s, .. } => {
                self.heartbeat(*island_id, now)?;
                self.observe_round_wall(*island_id, *round_wall_s);
                Ok(None)
            }
            ElasticMsg::SampleIndex { entry, .. } => {
                let (verdict, reason, outer_lag) = self.index_sample(entry)?;
                Ok(Some(ElasticMsg::SampleVerdict {
                    syncer_epoch: self.syncer_epoch,
                    verdict,
                    reason: reason.to_string(),
                    outer_lag,
                }))
            }
            ElasticMsg::DeltaReady { island_id, base_version, c_tokens, c_steps, .. }
            | ElasticMsg::DeltaTensor { island_id, base_version, c_tokens, c_steps, .. } => {
                // A delivered delta also proves liveness (renews the lease).
                if self.is_member(*island_id) {
                    self.heartbeat(*island_id, now)?;
                }
                let ev = self.submit(Delta {
                    island_id: *island_id,
                    base_version: *base_version,
                    c_tokens: *c_tokens,
                    c_steps: *c_steps,
                });
                if let TapeEvent::DeltaRejected { reason, .. } = ev {
                    bail!("delta rejected: {reason}");
                }
                Ok(None)
            }
            ElasticMsg::JoinAck { .. }
            | ElasticMsg::ElasticBase { .. }
            | ElasticMsg::SampleVerdict { .. }
            | ElasticMsg::Finished { .. } => {
                bail!("coordinator-to-island message received by the coordinator")
            }
            ElasticMsg::ElasticInit { .. } => bail!("ELASTIC_INIT is handled by the server"),
        }
    }

    pub fn capacity_arrived(&self) -> (f64, f64) {
        let total = self.members.values().map(|m| m.capacity).sum();
        let arrived = self.pending.keys().map(|i| self.members[i].capacity).sum();
        (arrived, total)
    }

    pub fn submit(&mut self, d: Delta) -> &TapeEvent {
        let ev = if !self.members.contains_key(&d.island_id) {
            TapeEvent::DeltaRejected { island_id: d.island_id, reason: "not_member" }
        } else if d.base_version > self.outer_version {
            TapeEvent::DeltaRejected { island_id: d.island_id, reason: "unknown_version" }
        } else {
            let lag = self.outer_version - d.base_version;
            if lag == 0 {
                self.pending.insert(d.island_id, d);
                TapeEvent::DeltaAccepted { island_id: d.island_id }
            } else if lag > u64::from(self.params.max_carry_lag) {
                TapeEvent::DeltaRejected { island_id: d.island_id, reason: "carry_lag_exceeded" }
            } else {
                self.carried.insert(d.island_id, (d, lag));
                TapeEvent::DeltaCarriedOver {
                    island_id: d.island_id,
                    base: d.base_version,
                    lag,
                    discount: self.params.carry_gamma.powi(lag as i32),
                }
            }
        };
        self.tape.push(ev);
        self.tape.last().unwrap()
    }

    fn weight_of(&self, d: &Delta) -> f64 {
        let m = &self.members[&d.island_id];
        if m.catch_up && m.joined_at == self.outer_version {
            return 0.0;
        }
        merge_weight(d.c_tokens, d.c_steps)
    }

    /// P4 step. `timed_out` means the soft deadline (soft_deadline_s) passed.
    pub fn try_advance(&mut self, timed_out: bool) -> Option<StepPlan> {
        let (cap_arrived, cap_total) = self.capacity_arrived();
        let arrived = self.pending.len();
        let reached = cap_total > 0.0 && cap_arrived >= self.params.quorum_theta * cap_total;
        if !reached && !timed_out {
            return None;
        }
        if arrived < self.params.q_min as usize {
            if timed_out {
                self.tape.push(TapeEvent::RoundIdle { arrived, cap_arrived, cap_total });
            }
            return None;
        }
        let base_version = self.outer_version;
        let mut raw: Vec<(u32, u64, f64)> =
            self.pending.values().map(|d| (d.island_id, d.base_version, self.weight_of(d))).collect();
        for (i, (d, lag)) in &self.carried {
            if self.members.contains_key(i) {
                let w = merge_weight(d.c_tokens, d.c_steps)
                    * self.params.carry_gamma.powi(*lag as i32);
                raw.push((*i, d.base_version, w));
            }
        }
        self.carried.clear();
        let total: f64 = raw.iter().map(|x| x.2).sum();
        let weights: Vec<(u32, u64, f64)> = raw
            .iter()
            .map(|&(i, b, w)| (i, b, if total > 0.0 { w / total } else { 0.0 }))
            .collect();
        let absent = self.members.keys().filter(|i| !self.pending.contains_key(i)).copied().collect();
        let arrived_ids: Vec<u32> = self.pending.keys().copied().collect();
        for (id, m) in self.members.iter_mut() {
            m.arrivals.push_back(arrived_ids.contains(id));
            while m.arrivals.len() > ARRIVAL_HISTORY {
                m.arrivals.pop_front();
            }
        }
        self.outer_version += 1;
        self.pending.clear();
        self.tape.push(TapeEvent::OuterStep {
            base_version,
            arrived,
            cap_arrived,
            cap_total,
            timed_out,
            raw_weights: raw,
            weights: weights.clone(),
            absent,
        });
        Some(StepPlan { base_version, weights })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::*;

    fn params() -> ElasticParams {
        ElasticParams {
            quorum_theta: DEFAULT_QUORUM_THETA,
            carry_gamma: DEFAULT_CARRY_GAMMA,
            soft_deadline_s: 900,
            q_min: DEFAULT_Q_MIN,
            max_carry_lag: DEFAULT_MAX_CARRY_LAG,
        }
    }

    fn delta(island_id: u32, base_version: u64) -> Delta {
        Delta { island_id, base_version, c_tokens: 10, c_steps: 1 }
    }

    fn entry(island_id: u32, outer_version: u64, policy_hash: [u8; 32], group: &str) -> SampleIndexEntry {
        SampleIndexEntry {
            island_id,
            outer_version,
            inner_step: 3,
            policy_hash,
            group_id: group.into(),
            prompt_id: "p".into(),
            n: 8,
            uri: "s3://bucket/g.parquet".into(),
            size_bytes: 1024,
            sha256: "a".repeat(64),
            has_behavior_logprob: true,
            advantage_included: true,
            created_at: 1.5,
            schema: SAMPLE_INDEX_SCHEMA.into(),
        }
    }

    #[test]
    fn sample_index_validates_dedups_and_judges_three_ways() {
        let mut c = ElasticCoordinator::new(params(), 30.0, 0).unwrap();
        c.join(1, 1.0, 0.0).unwrap();
        c.published.insert(0, [0; 32]);
        for v in 1..=3u64 {
            c.submit(delta(1, v - 1));
            c.try_advance(false).unwrap();
            c.published.insert(v, [v as u8; 32]);
        }
        // now outer_version 3
        assert_eq!(c.index_sample(&entry(1, 3, [3; 32], "a")).unwrap(), (VERDICT_ACCEPT, "same_version", 0));
        assert_eq!(c.index_sample(&entry(1, 2, [2; 32], "b")).unwrap(), (VERDICT_ACCEPT_IS, "stale_within_bound", 1));
        assert_eq!(c.index_sample(&entry(1, 0, [0; 32], "c")).unwrap().1, "outer_lag_exceeded");
        assert_eq!(c.index_sample(&entry(1, 2, [9; 32], "d")).unwrap().1, "policy_hash_mismatch");
        assert_eq!(c.index_sample(&entry(1, 7, [3; 32], "e")).unwrap().1, "unknown_version");
        assert_eq!(c.index_sample(&entry(2, 3, [3; 32], "f")).unwrap().1, "producer_not_member");
        let mut no_lp = entry(1, 2, [2; 32], "g");
        no_lp.has_behavior_logprob = false;
        assert_eq!(c.index_sample(&no_lp).unwrap().1, "missing_behavior_logprob");
        // Duplicate is idempotent; conflicting content is an error.
        assert_eq!(c.index_sample(&entry(1, 2, [2; 32], "b")).unwrap().0, VERDICT_ACCEPT_IS);
        let mut conflict = entry(1, 2, [2; 32], "b");
        conflict.n = 9;
        assert!(c.index_sample(&conflict).is_err());
        assert_eq!(c.sample_index.len(), 2);
        // Validation mirrors Python __post_init__.
        for bad in [
            SampleIndexEntry { schema: "x".into(), ..entry(1, 3, [3; 32], "h") },
            SampleIndexEntry { uri: "http://x".into(), ..entry(1, 3, [3; 32], "h") },
            SampleIndexEntry { n: 0, ..entry(1, 3, [3; 32], "h") },
            SampleIndexEntry { sha256: "z".repeat(64), ..entry(1, 3, [3; 32], "h") },
            SampleIndexEntry { advantage_included: false, ..entry(1, 3, [3; 32], "h") },
        ] {
            assert!(c.index_sample(&bad).is_err(), "{bad:?}");
        }
        assert!(c.tape.iter().any(|e| e.kind() == "sample_indexed"));
        assert!(entry(1, 3, [3; 32], "a").to_json().starts_with("{\"advantage_included\":true,"));
    }

    #[test]
    fn status_snapshot_exports_scheduling_fields() {
        let mut c = ElasticCoordinator::new(params(), 30.0, 2).unwrap();
        c.join(1, 3.0, 0.0).unwrap();
        c.join(2, 1.0, 0.0).unwrap();
        c.observe_round_wall(1, 10.0);
        c.observe_round_wall(1, 20.0);
        c.submit(delta(1, 0));
        c.try_advance(false).unwrap();
        c.submit(delta(2, 0)); // carried, lag 1
        let s = c.status_json(5.0, 12.0, "running");
        assert!(s.contains("\"state\":\"running\""));
        for needle in [
            "\"syncer_epoch\":2",
            "\"outer_version\":1",
            "\"carried_over_count\":1",
            "\"1\":{\"capacity\":3.0",
            "\"round_wall_ema_s\":12.0",
            "\"lease_remaining_s\":25.0",
            "\"arrival_history\":[true]",
            "\"arrival_history\":[false],\"pending\":false,\"carried_over_lag\":1",
            "\"soft_deadline_remaining_s\":12.0",
        ] {
            assert!(s.contains(needle), "{needle} not in {s}");
        }
    }

    #[test]
    fn hmac_matches_rfc4231_case_2() {
        let mac = hmac_sha256(b"Jefe", b"what do ya want for nothing?");
        let hex: String = mac.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");
    }

    #[test]
    fn mode_parsing_and_validation() {
        assert_eq!(IslandSchedulingMode::parse("legacy", 9.0, 9.0, 0, 0, 0).unwrap(), IslandSchedulingMode::Legacy);
        assert!(IslandSchedulingMode::parse("elastic", 0.75, 0.5, 900, 1, 2).is_ok());
        assert!(IslandSchedulingMode::parse("elastic", 0.0, 0.5, 900, 1, 2).is_err());
        assert!(IslandSchedulingMode::parse("elastic", 0.75, 1.5, 900, 1, 2).is_err());
        assert!(IslandSchedulingMode::parse("elastic", 0.75, 0.5, 900, 0, 2).is_err());
        assert!(IslandSchedulingMode::parse("Elastic", 0.75, 0.5, 900, 1, 2).is_err());
        let mut v = Vec::new();
        IslandSchedulingMode::Legacy.encode_contract(&mut v);
        assert!(v.is_empty(), "legacy must not change the contract encoding");
    }

    #[test]
    fn all_frames_roundtrip_and_reject_tampering_and_wrong_key() {
        let msgs = vec![
            ElasticMsg::Join { syncer_epoch: 3, island_id: 1, incarnation: 7, capacity: 2.5 },
            ElasticMsg::JoinAck {
                syncer_epoch: 3,
                learner_slot: 1,
                membership_epoch: 4,
                base_version: 9,
                policy_hash: [7; 32],
                catch_up: true,
            },
            ElasticMsg::Leave { syncer_epoch: 3, island_id: 1, reason: LEAVE_REASON_REQUESTED },
            ElasticMsg::LeaseHeartbeat {
                syncer_epoch: 3,
                island_id: 1,
                membership_epoch: 4,
                inner_step: 11,
                round_wall_s: 1.25,
            },
            ElasticMsg::SampleIndex { syncer_epoch: 3, entry: entry(1, 9, [1; 32], "g1") },
            ElasticMsg::SampleVerdict {
                syncer_epoch: 3,
                verdict: VERDICT_ACCEPT_IS,
                reason: "stale_within_bound".into(),
                outer_lag: 1,
            },
            ElasticMsg::Finished { syncer_epoch: 3, outer_version: 8, policy_hash: [5; 32] },
            ElasticMsg::DeltaReady {
                syncer_epoch: 3,
                island_id: 1,
                base_version: 9,
                c_tokens: 100,
                c_steps: 4,
            },
            ElasticMsg::ElasticInit { syncer_epoch: 3, island_id: 1, params: vec![1.0, -2.0] },
            ElasticMsg::DeltaTensor {
                syncer_epoch: 3,
                island_id: 1,
                base_version: 2,
                c_tokens: 5,
                c_steps: 1,
                update: vec![0.5, 0.25, -1.0],
            },
            ElasticMsg::ElasticBase { syncer_epoch: 3, outer_version: 4, params: vec![3.0] },
        ];
        for m in msgs {
            let (t, p) = m.encode(b"k1");
            assert_eq!(ElasticMsg::decode(b"k1", t, &p).unwrap(), m);
            assert!(ElasticMsg::decode(b"k2", t, &p).is_err(), "wrong key accepted");
            let mut bad = p.clone();
            bad[0] ^= 1;
            assert!(ElasticMsg::decode(b"k1", t, &bad).is_err(), "tampered body accepted");
            // MAC binds the type byte too.
            let other = if t == MSG_LEAVE { MSG_JOIN } else { MSG_LEAVE };
            assert!(ElasticMsg::decode(b"k1", other, &p).is_err());
        }
        assert!(ElasticMsg::decode(b"k1", MSG_HEARTBEAT, &seal(b"k1", MSG_HEARTBEAT, vec![0; 8])).is_err());
    }

    #[test]
    fn stale_syncer_epoch_is_fenced() {
        let mut c = ElasticCoordinator::new(params(), 30.0, 5).unwrap();
        let old = ElasticMsg::Join { syncer_epoch: 4, island_id: 1, incarnation: 0, capacity: 1.0 };
        assert!(format!("{:#}", c.apply(&old, 0.0).unwrap_err()).contains("fenced"));
        assert!(!c.is_member(1));
        let cur = ElasticMsg::Join { syncer_epoch: 5, island_id: 1, incarnation: 0, capacity: 1.0 };
        assert!(matches!(c.apply(&cur, 0.0).unwrap(), Some(ElasticMsg::JoinAck { syncer_epoch: 5, .. })));
        let hb = ElasticMsg::LeaseHeartbeat { syncer_epoch: 4, island_id: 1, membership_epoch: 1, inner_step: 0, round_wall_s: 0.0 };
        assert!(c.apply(&hb, 1.0).is_err());
    }

    #[test]
    fn capacity_weighted_quorum_steps_without_waiting_for_small_island() {
        let mut c = ElasticCoordinator::new(params(), 30.0, 0).unwrap();
        c.join(1, 3.0, 0.0).unwrap();
        c.join(2, 1.0, 0.0).unwrap();
        // Small island alone: 1/4 < 0.75 -> wait.
        c.submit(delta(2, 0));
        assert!(c.try_advance(false).is_none());
        c.pending.clear();
        // Big island alone: 3/4 >= 0.75 -> step, small island absent.
        c.submit(delta(1, 0));
        let plan = c.try_advance(false).unwrap();
        assert_eq!(plan.base_version, 0);
        assert_eq!(plan.weights, vec![(1, 0, 1.0)]);
        assert_eq!(c.outer_version, 1);
        match c.tape.last().unwrap() {
            TapeEvent::OuterStep { absent, .. } => assert_eq!(absent, &vec![2]),
            e => panic!("{e:?}"),
        }
    }

    #[test]
    fn soft_deadline_steps_or_idles_by_q_min() {
        let mut p = params();
        p.q_min = 2;
        let mut c = ElasticCoordinator::new(p, 30.0, 0).unwrap();
        for i in 1..=3 {
            c.join(i, 1.0, 0.0).unwrap();
        }
        c.submit(delta(1, 0));
        assert!(c.try_advance(true).is_none());
        assert_eq!(c.tape.last().unwrap().kind(), "round_idle");
        c.submit(delta(2, 0));
        assert!(c.try_advance(false).is_none(), "2/3 < 0.75 before deadline");
        assert!(c.try_advance(true).is_some(), "deadline with arrived >= q_min");
    }

    #[test]
    fn late_delta_is_discounted_by_gamma_pow_lag_and_rejected_past_max_lag() {
        let mut c = ElasticCoordinator::new(params(), 30.0, 0).unwrap();
        c.join(1, 1.0, 0.0).unwrap();
        c.join(2, 1.0, 0.0).unwrap();
        c.submit(delta(1, 0));
        c.try_advance(true).unwrap(); // v1
        c.submit(delta(1, 1));
        c.try_advance(true).unwrap(); // v2
        // Island 2 finally delivers on base 0: lag 2 -> carried with 0.25.
        match c.submit(delta(2, 0)) {
            TapeEvent::DeltaCarriedOver { lag, discount, .. } => {
                assert_eq!((*lag, *discount), (2, 0.25))
            }
            e => panic!("{e:?}"),
        }
        c.submit(delta(1, 2));
        let plan = c.try_advance(true).unwrap();
        // raw: island1 100, island2 100*0.25=25 -> 0.8 / 0.2
        assert_eq!(plan.weights, vec![(1, 2, 0.8), (2, 0, 0.2)]);
        // lag 3 > max_carry_lag 2 -> rejected.
        match c.submit(delta(2, 0)) {
            TapeEvent::DeltaRejected { reason, .. } => assert_eq!(*reason, "carry_lag_exceeded"),
            e => panic!("{e:?}"),
        }
    }

    #[test]
    fn leave_and_lease_expiry_drop_uncommitted_delta() {
        let mut c = ElasticCoordinator::new(params(), 10.0, 0).unwrap();
        c.join(1, 1.0, 0.0).unwrap();
        c.join(2, 1.0, 0.0).unwrap();
        c.join(3, 1.0, 0.0).unwrap();
        c.submit(delta(2, 0));
        c.leave(2, LEAVE_REASON_REQUESTED).unwrap();
        match c.tape.last().unwrap() {
            TapeEvent::PoolLeave { dropped_uncommitted: Some(d), .. } => assert_eq!(d.island_id, 2),
            e => panic!("{e:?}"),
        }
        c.submit(delta(3, 0));
        c.heartbeat(1, 15.0).unwrap();
        assert_eq!(c.expire_leases(15.0), vec![3]);
        match c.tape.last().unwrap() {
            TapeEvent::PoolLeave { reason, dropped_uncommitted: Some(d), .. } => {
                assert_eq!((*reason, d.island_id), (LEAVE_REASON_LEASE_EXPIRED, 3))
            }
            e => panic!("{e:?}"),
        }
        assert_eq!(c.capacity_arrived(), (0.0, 1.0));
        assert_eq!(c.membership_epoch, 5);
    }

    #[test]
    fn coordinator_state_roundtrips_membership_carry_and_dropped_ledger() {
        let mut c = ElasticCoordinator::new(params(), 30.0, 4).unwrap();
        c.join(1, 2.0, 0.0).unwrap();
        c.join(2, 1.0, 0.0).unwrap();
        c.join(3, 1.0, 0.0).unwrap();
        c.submit(delta(1, 0));
        c.try_advance(true).unwrap();
        c.submit(delta(2, 0)); // carried, lag 1
        c.submit(delta(3, 1));
        c.leave(3, LEAVE_REASON_REQUESTED).unwrap(); // dropped
        let bytes = c.encode_state();
        let r = ElasticCoordinator::decode_state(params(), 30.0, &bytes, 100.0).unwrap();
        assert_eq!((r.syncer_epoch, r.membership_epoch, r.outer_version), (4, 4, 1));
        assert_eq!(r.members.keys().copied().collect::<Vec<_>>(), vec![1, 2]);
        assert_eq!(r.members[&1].capacity, 2.0);
        assert_eq!(r.carried[&2], (delta(2, 0), 1));
        assert_eq!(r.dropped, vec![delta(3, 1)]);
        assert_eq!(r.members[&2].last_heartbeat, 100.0);
        let mut other = params();
        other.carry_gamma = 0.25;
        assert!(ElasticCoordinator::decode_state(other, 30.0, &bytes, 0.0).is_err());
    }

    #[test]
    fn joiner_has_zero_weight_in_its_first_round() {
        let mut c = ElasticCoordinator::new(params(), 30.0, 0).unwrap();
        assert!(!c.join(1, 1.0, 0.0).unwrap(), "founding member is not catch-up");
        c.submit(delta(1, 0));
        c.try_advance(false).unwrap();
        assert!(c.join(2, 1.0, 0.0).unwrap());
        c.submit(delta(1, 1));
        c.submit(delta(2, 1));
        let plan = c.try_advance(false).unwrap();
        assert_eq!(plan.weights, vec![(1, 1, 1.0), (2, 1, 0.0)]);
        c.submit(delta(1, 2));
        c.submit(delta(2, 2));
        let plan = c.try_advance(false).unwrap();
        assert_eq!(plan.weights, vec![(1, 2, 0.5), (2, 2, 0.5)]);
    }
}
