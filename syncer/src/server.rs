//! Async TCP server implementing the syncer side of docs/PROTOCOL.md:
//! per-learner connection groups (control stream + striped data streams),
//! chunk reassembly, and the pull-driven quorum/grace merge scheduler
//! at the core of the training loop. Rounds are pipelined: up to
//! `Config::pipeline` fragments are in flight at once (two by default), so
//! a slow quorum on one fragment never delays pulling the next.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::future::Future;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;
use tokio::net::tcp::OwnedWriteHalf;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::local_init::{InitDigests, LocalInit};
use crate::recovery::{self, HelloExt, PeerRecovery};
use crate::membership::{Admission, AdmitError, CapacityQuorum, Member, Membership};
use crate::protocol::*;
use crate::fold::{FragmentFold, RoundFold, StreamingFold, WeightedFoldMode};
use crate::state::{
    remove_final_marker, write_final_marker, ComputedMerge, GlobalState, Layout, PreparedMerge,
};

const CHUNK_SIZE: usize = 4 * 1024 * 1024;
const CHUNK_HEADER_SIZE: u64 = 24;
const MAX_PARTIAL_MESSAGES: usize = 64;
const WRITE_TIMEOUT: Duration = Duration::from_secs(180);
const WRITER_QUEUE: usize = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LearnerWeight {
    Tokens2OverSteps,
    Equal,
}

impl LearnerWeight {
    /// The merge weight as a function of a push's (c_tokens, c_steps).
    pub fn function(self) -> fn(u64, u64) -> f64 {
        match self {
            Self::Tokens2OverSteps => crate::merge::learner_weight,
            Self::Equal => |_, _| 1.0,
        }
    }
}

#[derive(Clone)]
pub struct Config {
    pub port: u16,
    pub learners: u32,
    pub quorum: u32,
    /// Upper bound on the grace window; the actual wait adapts per round
    /// (see `adaptive_grace`).
    pub grace_ms: u64,
    /// Safety margin γ < 1 on the computed slack.
    pub grace_gamma: f64,
    /// Compute-overlap budget τ, in learner inner steps.
    pub grace_tau: f64,
    /// Lower bound on the grace window of a round that reached quorum, in ms
    /// (still capped by `grace_ms`). 0 = off: the adaptive window alone.
    pub grace_min_ms: u64,
    /// Arrival-gap floor on the grace window: at least this quantile of the
    /// recent gaps between a round's first push and each later push of the
    /// same round, late pushes included (`ArrivalGaps`). Still capped by
    /// `grace_ms`. 0 = off.
    pub grace_arrival_quantile: f64,
    /// Terminal grace, in ms: the last round of each fragment waits up to
    /// this long for every expected member (not capped by `grace_ms`). No
    /// later round of that fragment exists to carry a late push, so without
    /// it a late delta of a last round never reaches the final parameters.
    /// 0 = off.
    pub terminal_grace_ms: u64,
    /// Fragment rounds in flight at once (two by default, matching the
    /// default overlap budget τ=2). While one round sits in its quorum/grace window,
    /// the next fragment's pull is already out — sync latency overlaps
    /// learner compute instead of serializing with it. 1 = serial rounds.
    /// Clamped to the fragment count so concurrent rounds always target
    /// distinct fragments.
    pub pipeline: u32,
    /// Lower bound on the time between consecutive round LAUNCHES (ms).
    /// On a WAN, round latency naturally spaces merges many inner steps
    /// apart (the sync interval H the algorithm is tuned for); on a LAN or
    /// localhost, rounds complete as fast as learners answer and H
    /// collapses to a step or two, over-driving the outer optimizer.
    /// 0 = unthrottled.
    pub min_round_interval_ms: u64,
    /// Target sync interval H, in inner steps per fragment. The launch
    /// floor adapts to the measured learner step time:
    /// interval = H·ξ_step/P, so each fragment re-merges after ~H steps of
    /// the slowest learner. Inactive until a step-time estimate exists and
    /// wherever natural round latency already exceeds it (any real WAN);
    /// measured on gemma4/Lean: H≈2 costs ~+9% eval loss vs synchronous,
    /// H≈24 (the default) matches it. 0 disables; the manual
    /// min_round_interval_ms floor applies on top.
    pub sync_interval_steps: f64,
    /// HeLoCo per-tensor delta correction before merging.
    pub delta_correction: bool,
    pub quorum_timeout_s: u64,
    /// Bounded wait for learners to apply and acknowledge the terminal cut.
    pub final_ack_timeout_s: u64,
    pub total_steps: u64,
    /// Strict dense-policy mode: one logical local step contributes every
    /// fragment in a serial sweep.  None preserves the legacy independent
    /// fragment-round behavior.
    pub policy_sweep_fragments: Option<u32>,
    pub outer_lr: f32,
    pub outer_momentum: f32,
    /// Delayed Nesterov period N >= 2 (momentum applied every N outer
    /// steps); None keeps plain Nesterov and the old profile hash.
    pub outer_delayed_nesterov_period: Option<u32>,
    /// Spectrum-flattening backend for MERGE_ISO tensors.
    pub iso_backend: crate::iso_worker::IsoBackendConfig,
    pub final_state: Option<std::path::PathBuf>,
    /// Consistent-snapshot file; written every `checkpoint_every` rounds at
    /// the quiescent cut between rounds, resumed from when `resume` is set.
    pub checkpoint_path: Option<std::path::PathBuf>,
    pub checkpoint_every: u64,
    /// Global steps whose cut is also written to `<checkpoint_path>.step-<t>`
    /// off the scheduler (`crate::eval_checkpoint`); strictly increasing,
    /// below `total_steps`. Empty: none.
    pub checkpoint_at_steps: Vec<u64>,
    /// Policy-sweep only: persist each per-round checkpoint on a blocking
    /// writer instead of the scheduler, and broadcast that round only after
    /// the write is durable. Merge results, versions, and checkpoint bytes
    /// are unchanged, so this is deliberately not part of the semantic
    /// profile hash.
    pub async_checkpoint: bool,
    /// S26 E3: full-f32 refresh period of bf16 downlink deltas (sessions
    /// with `DTYPE_FLAG_BCAST_DELTA`). Wire encoding only: merges, versions
    /// and checkpoints are unchanged, so it is not in the profile hash.
    pub bcast_delta_full_every: Option<u64>,
    pub resume: bool,
    /// Create the adjacent final marker only for a completed terminal cut.
    pub mark_final_checkpoint: bool,
    /// `--resume` refuses a checkpoint whose `.profile` record is missing or
    /// differs from this syncer's semantic profile. Not in the profile hash.
    pub require_profile_record: bool,
    /// Benchmark-only exact local optimizer-step target for every learner.
    pub learner_budget_steps: Option<u64>,
    /// JSONL event tape: merge records plus sweep ledger reconciliation cuts.
    pub event_tape: Option<std::path::PathBuf>,
    /// Maximum admitted learner base-version lag; None preserves the
    /// existing unbounded behavior.
    pub max_base_lag: Option<u64>,
    pub learner_weight: LearnerWeight,
    /// Fail closed unless HELLO carries and matches this Config's canonical
    /// semantic profile hash. Disabled by default for legacy/non-SAO clients.
    pub require_profile_binding: bool,
    /// HMAC-SHA256 key (never hashed into the profile): every HELLO must
    /// carry a trailing HMAC under this key.
    pub island_hmac_key: Option<Vec<u8>>,
    /// secret-handling-hardening: island contract (backend identity bound to
    /// the LR schedule, 32 bytes) pinned by the head's launch config.
    /// A HELLO whose session_contract_hash is not
    /// sha256(domain || layout_fingerprint || pinned) is refused, so the first
    /// island to connect no longer decides the contract. None keeps first-wins.
    pub expected_island_contract: Option<[u8; 32]>,
    /// Fencing token of this syncer incarnation: a HELLO that carries a
    /// lower epoch is refused; the epoch is persisted next to the checkpoint
    /// and never lowered on resume.
    pub syncer_epoch: u64,
    /// Dynamic membership of the fragment-round server (leases, re-JOIN,
    /// fencing, capacity quorum, soft deadline, carry discount). The default
    /// keeps the fixed-membership behavior.
    pub membership: crate::membership::MembershipConfig,
    /// Peer recovery of an island that lost its trainer state (HELLO with
    /// the recover flag). Bound, donor tries and donor timeout only: it never
    /// changes a merge, so it is not in the semantic profile hash.
    pub recovery: crate::recovery::RecoveryConfig,
    /// S27 P4b: how rounds that the exact fold cannot reproduce (partial
    /// quorum, token weights) reduce pushes.  Numerics only (the result is
    /// within f32 rounding of the buffered merge), so not in the profile
    /// hash; Buffered keeps the historical bits.
    pub weighted_fold: crate::fold::WeightedFoldMode,
    /// S27 P4b: this process's slice of a fragment-sharded head.  Default is
    /// unsharded and changes nothing.
    pub shard: crate::shard::ShardPlan,
    /// MoE: weight each expert tensor of a push by the tokens the island
    /// routed to that expert during the outer interval (see
    /// `protocol::decode_expert_tags`). Every HELLO must declare the matching
    /// session flag. Off keeps the wire bytes and the merge unchanged.
    pub expert_token_weights: bool,
    /// The final cut carries the version manifest only (no f32 fragments):
    /// an island acknowledges once it has the final versions, and the final
    /// parameters stay in this syncer's checkpoint / final state. Islands
    /// must expect it, so it is in the semantic profile hash (appended only
    /// when on). Off sends every final fragment to every island.
    pub final_cut_manifest_only: bool,
    /// S27 P2 task dispenser (`dispense.rs`): islands claim the rows they
    /// train over TASK_* connections. Hands out data only, never changes a
    /// merge, so it is not in the semantic profile hash. None: off.
    pub dispense: Option<crate::dispense::DispenseConfig>,
    /// JSONL log of every dispenser event (`task-dispense.jsonl`).
    pub dispense_log: Option<std::path::PathBuf>,
}

// v2 drops `resume`: it is how a run (re)starts, not its semantics, so a
// restarted syncer keeps the hash every island already carries in HELLO.
const SEMANTIC_PROFILE_DOMAIN: &[u8] = b"yeto-syncer-semantic-profile-v2\0";
const DELAYED_NESTEROV_PROFILE_DOMAIN: &[u8] = b"yeto-syncer-delayed-nesterov-v1\0";
const EXPERT_TOKEN_WEIGHTS_PROFILE_DOMAIN: &[u8] = b"yeto-syncer-expert-token-weights-v1\0";
const FINAL_CUT_MANIFEST_ONLY_PROFILE_DOMAIN: &[u8] = b"yeto-syncer-final-cut-manifest-only-v1\0";
const GRACE_FLOOR_PROFILE_DOMAIN: &[u8] = b"yeto-syncer-grace-floor-v1\0";
const TERMINAL_GRACE_PROFILE_DOMAIN: &[u8] = b"yeto-syncer-terminal-grace-v1\0";
const EVAL_CHECKPOINTS_PROFILE_DOMAIN: &[u8] = b"yeto-syncer-eval-checkpoints-v1\0";

impl Config {
    /// Canonical identity of every launch field that can affect merging,
    /// scheduling, or recovery. Listener ports and concrete output paths are
    /// deliberately excluded; checkpoint *enablement* remains included.
    pub fn semantic_profile_hash(&self) -> [u8; 32] {
        let mut encoded = Vec::with_capacity(160);
        encoded.extend_from_slice(SEMANTIC_PROFILE_DOMAIN);
        encoded.extend_from_slice(&self.learners.to_le_bytes());
        encoded.extend_from_slice(&self.quorum.to_le_bytes());
        encoded.extend_from_slice(&self.grace_ms.to_le_bytes());
        encoded.extend_from_slice(&self.grace_gamma.to_bits().to_le_bytes());
        encoded.extend_from_slice(&self.grace_tau.to_bits().to_le_bytes());
        encoded.extend_from_slice(&self.pipeline.to_le_bytes());
        encoded.extend_from_slice(&self.min_round_interval_ms.to_le_bytes());
        encoded.extend_from_slice(&self.sync_interval_steps.to_bits().to_le_bytes());
        encoded.push(u8::from(self.delta_correction));
        encoded.extend_from_slice(&self.quorum_timeout_s.to_le_bytes());
        encoded.extend_from_slice(&self.final_ack_timeout_s.to_le_bytes());
        encoded.extend_from_slice(&self.total_steps.to_le_bytes());
        encode_optional_u32(&mut encoded, self.policy_sweep_fragments);
        encoded.extend_from_slice(&self.outer_lr.to_bits().to_le_bytes());
        encoded.extend_from_slice(&self.outer_momentum.to_bits().to_le_bytes());
        encoded.push(u8::from(self.checkpoint_path.is_some()));
        encoded.extend_from_slice(&self.checkpoint_every.to_le_bytes());
        encoded.push(u8::from(self.mark_final_checkpoint));
        encode_optional_u64(&mut encoded, self.learner_budget_steps);
        encode_optional_u64(&mut encoded, self.max_base_lag);
        encoded.push(match self.learner_weight {
            LearnerWeight::Tokens2OverSteps => 0,
            LearnerWeight::Equal => 1,
        });
        encoded.push(u8::from(self.require_profile_binding));
        // Appended only when set, so plain-Nesterov hashes stay unchanged.
        if let Some(period) = self.outer_delayed_nesterov_period {
            encoded.extend_from_slice(DELAYED_NESTEROV_PROFILE_DOMAIN);
            encoded.extend_from_slice(&period.to_le_bytes());
        }
        // Appended only when on, so every existing hash stays unchanged.
        if self.expert_token_weights {
            encoded.extend_from_slice(EXPERT_TOKEN_WEIGHTS_PROFILE_DOMAIN);
        }
        // Appended only when on, so every existing hash stays unchanged.
        if self.final_cut_manifest_only {
            encoded.extend_from_slice(FINAL_CUT_MANIFEST_ONLY_PROFILE_DOMAIN);
        }
        // Appended only when a floor is on, so every existing hash stays unchanged.
        if self.grace_min_ms > 0 || self.grace_arrival_quantile > 0.0 {
            encoded.extend_from_slice(GRACE_FLOOR_PROFILE_DOMAIN);
            encoded.extend_from_slice(&self.grace_min_ms.to_le_bytes());
            encoded.extend_from_slice(&self.grace_arrival_quantile.to_bits().to_le_bytes());
        }
        // Appended only when on, so every existing hash stays unchanged.
        if self.terminal_grace_ms > 0 {
            encoded.extend_from_slice(TERMINAL_GRACE_PROFILE_DOMAIN);
            encoded.extend_from_slice(&self.terminal_grace_ms.to_le_bytes());
        }
        // Evaluation checkpoints change no merge, but a step that comes due
        // during the previous write stalls the scheduler, and the list sits
        // beside the checkpoint settings above. Appended only when set, so
        // every existing hash stays unchanged.
        if !self.checkpoint_at_steps.is_empty() {
            encoded.extend_from_slice(EVAL_CHECKPOINTS_PROFILE_DOMAIN);
            encoded.extend_from_slice(&(self.checkpoint_at_steps.len() as u64).to_le_bytes());
            for step in &self.checkpoint_at_steps {
                encoded.extend_from_slice(&step.to_le_bytes());
            }
        }
        Sha256::digest(encoded).into()
    }
}

fn encode_optional_u32(encoded: &mut Vec<u8>, value: Option<u32>) {
    match value {
        None => encoded.push(0),
        Some(value) => {
            encoded.push(1);
            encoded.extend_from_slice(&value.to_le_bytes());
        }
    }
}

fn encode_optional_u64(encoded: &mut Vec<u8>, value: Option<u64>) {
    match value {
        None => encoded.push(0),
        Some(value) => {
            encoded.push(1);
            encoded.extend_from_slice(&value.to_le_bytes());
        }
    }
}

struct OutFrame {
    msg_type: u8,
    parts: Vec<bytes::Bytes>,
}

#[derive(Debug)]
struct OutboundStreamClosed {
    member: Member,
}

impl std::fmt::Display for OutboundStreamClosed {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "learner {} generation {} outbound stream closed",
            self.member.learner_id, self.member.generation
        )
    }
}

impl std::error::Error for OutboundStreamClosed {}

struct PartialMsg {
    /// First chunk of this message (upload start, for PushTransfer).
    started: Instant,
    buf: Vec<u8>,
    filled: usize,
    ranges: Vec<(usize, usize)>,
}


struct Group {
    member: Member,
    dtype: u8,
    /// PUSH_FRAGMENT carries the routed-token tag block (HELLO flag).
    expert_tokens: bool,
    layout: Layout,
    layout_fingerprint: [u8; 32],
    session_contract_hash: [u8; 32],
    num_streams: u16,
    max_init_payload: u64,
    max_push_payload: u64,
    max_chunked_inner: u64,
    control: mpsc::Sender<OutFrame>,
    data: Mutex<HashMap<u16, mpsc::Sender<OutFrame>>>,
    msg_id: AtomicU64,
    rr: AtomicUsize,
    reasm: Mutex<HashMap<u64, PartialMsg>>,
}

impl Group {
    async fn send_small(&self, msg_type: u8, payload: bytes::Bytes) -> Result<()> {
        self.control
            .send(OutFrame {
                msg_type,
                parts: vec![payload],
            })
            .await
            .map_err(|_| {
                anyhow::anyhow!(
                    "learner {} generation {} control stream closed",
                    self.member.learner_id,
                    self.member.generation
                )
            })
    }

    fn chunk_streams(&self) -> Vec<mpsc::Sender<OutFrame>> {
        let mut streams: Vec<(u16, mpsc::Sender<OutFrame>)> = self
            .data
            .lock()
            .unwrap()
            .iter()
            .map(|(index, sender)| (*index, sender.clone()))
            .collect();
        streams.sort_unstable_by_key(|(index, _)| *index);
        let mut streams: Vec<mpsc::Sender<OutFrame>> =
            streams.into_iter().map(|(_, sender)| sender).collect();
        if streams.is_empty() {
            streams.push(self.control.clone());
        }
        streams
    }

    /// Stream one model-sized tensor as CHUNK envelopes without ever
    /// materializing a complete encoded tensor or inner frame. The receiver
    /// observes exactly the same inner frame as `header + prefix + tensor`.
    async fn send_tensor_large(
        &self,
        msg_type: u8,
        prefix: &[u8],
        dtype: u8,
        values: &[f32],
    ) -> Result<()> {
        self.send_tensor_large_chunked(msg_type, prefix, dtype, values, CHUNK_SIZE)
            .await
    }

    async fn send_tensor_large_chunked(
        &self,
        msg_type: u8,
        prefix: &[u8],
        dtype: u8,
        values: &[f32],
        chunk_size: usize,
    ) -> Result<()> {
        if chunk_size == 0 {
            bail!("tensor chunk size must be positive");
        }
        let tensor_len = tensor_nbytes(dtype, values.len())?;
        let payload_len = prefix
            .len()
            .checked_add(tensor_len)
            .context("tensor frame payload length overflow")?;
        let total = 13usize
            .checked_add(payload_len)
            .context("tensor inner-frame length overflow")?;
        let payload_len_u64 = u64::try_from(payload_len).context("tensor payload is too large")?;
        let total_u64 = u64::try_from(total).context("tensor inner frame is too large")?;

        let mut fixed = Vec::new();
        fixed
            .try_reserve_exact(13 + prefix.len())
            .context("cannot allocate tensor frame prefix")?;
        fixed.extend_from_slice(&MAGIC.to_le_bytes());
        fixed.push(msg_type);
        fixed.extend_from_slice(&payload_len_u64.to_le_bytes());
        fixed.extend_from_slice(prefix);

        let streams = self.chunk_streams();

        let msg_id = self.msg_id.fetch_add(1, Ordering::Relaxed);
        let mut offset = 0usize;
        while offset < total {
            let end = offset.saturating_add(chunk_size).min(total);
            let mut chunk = Vec::new();
            chunk
                .try_reserve_exact(end - offset)
                .context("cannot allocate tensor wire chunk")?;
            if offset < fixed.len() {
                chunk.extend_from_slice(&fixed[offset..end.min(fixed.len())]);
            }
            if end > fixed.len() {
                let tensor_start = offset.saturating_sub(fixed.len());
                let tensor_end = end - fixed.len();
                append_tensor_bytes(
                    dtype,
                    values,
                    tensor_start,
                    tensor_end - tensor_start,
                    &mut chunk,
                )?;
            }
            debug_assert_eq!(chunk.len(), end - offset);

            let mut head = Vec::with_capacity(24);
            head.extend_from_slice(&msg_id.to_le_bytes());
            head.extend_from_slice(&total_u64.to_le_bytes());
            head.extend_from_slice(&(offset as u64).to_le_bytes());
            let idx = self.rr.fetch_add(1, Ordering::Relaxed) % streams.len();
            streams[idx]
                .send(OutFrame {
                    msg_type: MSG_CHUNK,
                    parts: vec![bytes::Bytes::from(head), bytes::Bytes::from(chunk)],
                })
                .await
                .map_err(|_| OutboundStreamClosed {
                    member: self.member,
                })?;
            offset = end;
        }
        Ok(())
    }
}

fn append_tensor_bytes(
    dtype: u8,
    values: &[f32],
    byte_offset: usize,
    byte_len: usize,
    out: &mut Vec<u8>,
) -> Result<()> {
    let width = match dtype {
        DTYPE_F32 => 4,
        DTYPE_BF16 => 2,
        _ => bail!("unknown dtype {dtype}"),
    };
    let tensor_len = tensor_nbytes(dtype, values.len())?;
    let byte_end = byte_offset
        .checked_add(byte_len)
        .context("tensor byte range overflow")?;
    if byte_end > tensor_len {
        bail!("tensor byte range exceeds encoded length");
    }
    out.try_reserve_exact(byte_len)
        .context("cannot allocate encoded tensor chunk")?;
    if byte_len == 0 {
        return Ok(());
    }
    let first = byte_offset / width;
    let last = byte_end.div_ceil(width);
    for (index, value) in values[first..last].iter().enumerate() {
        let value_index = first + index;
        let value_start = value_index * width;
        let start = byte_offset.saturating_sub(value_start);
        let end = (byte_end - value_start).min(width);
        match dtype {
            DTYPE_F32 => out.extend_from_slice(&value.to_le_bytes()[start..end]),
            DTYPE_BF16 => {
                let bytes = half::bf16::from_f32(*value).to_bits().to_le_bytes();
                out.extend_from_slice(&bytes[start..end]);
            }
            _ => unreachable!(),
        }
    }
    Ok(())
}

enum Event {
    Fatal {
        metric: &'static str,
        message: String,
    },
    Hello {
        group: Arc<Group>,
    },
    Init {
        member: Member,
        fragment_id: u32,
        values: Vec<f32>,
        /// SHA-256 of the INIT_PARAMS tensor bytes as received.
        sha256: [u8; 32],
    },
    /// Local-init startup: the island's per-fragment INIT digests.
    InitDigest {
        member: Member,
        digests: InitDigests,
    },
    Push {
        member: Member,
        push: Push,
    },
    Heartbeat {
        member: Member,
        local_step: u64,
    },
    FinalAck {
        member: Member,
        global_step: u64,
    },
    BudgetDone {
        member: Member,
        local_steps: u64,
    },
    Disconnected {
        member: Member,
    },
    /// The island announced why it leaves; its disconnect follows.
    Leave {
        member: Member,
        reason: u8,
    },
    /// A peer-recovery control message (`recovery.rs`).
    Recovery {
        member: Member,
        msg: RecoveryMsg,
    },
}

enum RecoveryMsg {
    /// Donor -> syncer.
    SnapshotReady(recovery::SnapshotReady),
    /// Recovering island -> syncer: versions held after the restore.
    Restored(Option<Vec<u64>>),
    /// Recovering island -> syncer: the named source did not work.
    Failed(String),
}

struct Push {
    learner_id: u32,
    fragment_id: u32,
    global_step: u64,
    round_attempt: u32,
    base_version: u64,
    local_step: u64,
    c_steps: u32,
    c_tokens: u64,
    /// Signed outer gradient `raw_anchor - local`, converted exactly once
    /// from the learner's wire delta `local - raw_anchor`.
    outer_gradient: Vec<f32>,
    /// Expert token weights: the per-tensor merge scale of this push (see
    /// `protocol::decode_expert_tags`); `None` when the option is off.
    tensor_scales: Option<Vec<f64>>,
    transfer: PushTransfer,
}

/// When and how fast one PUSH_FRAGMENT reached the syncer (D4 full-parameter
/// timing): `recv_seconds` runs from the first byte of the push (its first
/// chunk, or the frame header) to its last byte; `arrived_unix` is the wall
/// clock at the last byte. Written per responder on the event tape so a round
/// splits into upload, waiting for the slowest island, and merge.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct PushTransfer {
    recv_seconds: f64,
    arrived_unix: f64,
}

#[derive(Clone, Debug, PartialEq)]
struct SessionSpec {
    dtype: u8,
    layout: Layout,
    layout_fingerprint: [u8; 32],
    session_contract_hash: [u8; 32],
    syncer_profile_hash: Option<[u8; 32]>,
    /// yeto-framework-decoupling 7.7c: "<vendor>-<card>" from the optional
    /// HELLO trailer; empty = not declared (older or generic clients).
    compat_group: String,
}

/// Magic of the optional HELLO compat_group trailer (after num_streams):
/// b"YCG1" | u32 length | utf-8. Older syncers refuse it (trailing bytes).
const HELLO_COMPAT_MAGIC: &[u8; 4] = b"YCG1";

struct ParsedHello {
    learner_id: u32,
    generation: u64,
    dtype: u8,
    layout: Layout,
    layout_fingerprint: [u8; 32],
    session_contract_hash: [u8; 32],
    syncer_profile_hash: Option<[u8; 32]>,
    num_streams: u16,
    max_init_payload: u64,
    max_push_payload: u64,
    compat_group: String,
    /// Optional trailer: syncer-epoch fence and the recover flag.
    ext: HelloExt,
}

#[derive(Default)]
struct RegistryState {
    /// The generation eligible for future rounds for each logical learner.
    current: HashMap<u32, Member>,
    /// Every still-connected generation, including one retained solely for
    /// a round that captured it before a reconnect or duplicate HELLO.
    groups: HashMap<Member, Arc<Group>>,
    /// Leases and fencing; admission runs under this registry lock.
    membership: Membership,
    /// Islands under peer recovery: registered (in `groups`) but kept out of
    /// `current`, so they get no pulls and no broadcasts until READY.
    recovering: HashMap<u32, Member>,
    /// Where the membership record (syncer epoch, retired generations) is
    /// persisted next to the checkpoint; None: not persisted.
    membership_record: Option<std::path::PathBuf>,
    /// The task dispenser (S27 P2); its record is a checkpoint sidecar.
    dispense: Option<Arc<crate::dispense_service::DispenseService>>,
}

#[derive(Debug, Eq, PartialEq)]
enum RegisterError {
    /// This exact generation is still connected.
    Duplicate,
    Fenced(AdmitError),
}

impl RegistryState {
    fn new(membership: Membership) -> Self {
        Self { membership, ..Self::default() }
    }

    /// Atomically register one control connection generation. Returning an
    /// error leaves every map untouched, so an identical concurrent HELLO
    /// cannot replace a live group and later remove it on disconnect, and a
    /// retired generation cannot come back.
    #[cfg(test)]
    fn register_group(
        &mut self,
        group: Arc<Group>,
        now: Instant,
    ) -> std::result::Result<(Option<Member>, Admission), RegisterError> {
        self.register_group_with(group, now, HelloExt::default())
    }

    /// `register_group` for a HELLO with its extension: the syncer-epoch
    /// fence runs first, and a recovering island is registered without
    /// joining `current` (the previous generation still leaves it).
    fn register_group_with(
        &mut self,
        group: Arc<Group>,
        now: Instant,
        ext: HelloExt,
    ) -> std::result::Result<(Option<Member>, Admission), RegisterError> {
        let member = group.member;
        if self.groups.contains_key(&member) {
            return Err(RegisterError::Duplicate);
        }
        self.membership.check_syncer_epoch(ext.syncer_epoch).map_err(RegisterError::Fenced)?;
        let admission = self.membership.admit(member, now).map_err(RegisterError::Fenced)?;
        self.groups.insert(member, group);
        let replaced = if ext.recover {
            self.recovering.insert(member.learner_id, member);
            self.current.remove(&member.learner_id)
        } else {
            self.recovering.remove(&member.learner_id);
            self.current.insert(member.learner_id, member)
        };
        self.persist_membership();
        Ok((replaced, admission))
    }

    /// The control connection of `member` closed.
    fn unregister(&mut self, member: Member) {
        self.groups.remove(&member);
        if self.current.get(&member.learner_id) == Some(&member) {
            self.current.remove(&member.learner_id);
        }
        if self.recovering.get(&member.learner_id) == Some(&member) {
            self.recovering.remove(&member.learner_id);
        }
        self.membership.depart(member);
        self.persist_membership();
    }

    fn is_recovering(&self, member: Member) -> bool {
        self.recovering.get(&member.learner_id) == Some(&member)
    }

    /// Move a recovered island onto the roster. False if `member` is no
    /// longer the recovering generation.
    fn promote_recovered(&mut self, member: Member) -> bool {
        if !self.is_recovering(member) {
            return false;
        }
        self.recovering.remove(&member.learner_id);
        self.current.insert(member.learner_id, member);
        true
    }

    /// Write the membership record if its persisted part changed. A failed
    /// write only weakens fencing across a restart, so it is logged.
    fn persist_membership(&mut self) {
        if !self.membership.take_dirty() {
            return;
        }
        if let Some(path) = &self.membership_record {
            if let Err(error) = write_membership_record(path, &self.membership.encode_persisted()) {
                warn!(path = %path.display(), "membership record write failed: {error:#}");
            }
        }
    }

    /// Remove members whose lease expired from future rounds. Their
    /// connections stay registered until they close, so rounds that already
    /// captured them keep a valid group, but they can no longer renew.
    fn expire_leases(&mut self, now: Instant) -> Vec<(Member, Option<Arc<Group>>)> {
        let expired = self.membership.expire(now);
        let expired = expired
            .into_iter()
            .map(|member| {
                if self.current.get(&member.learner_id) == Some(&member) {
                    self.current.remove(&member.learner_id);
                }
                if self.recovering.get(&member.learner_id) == Some(&member) {
                    self.recovering.remove(&member.learner_id);
                }
                (member, self.groups.get(&member).cloned())
            })
            .collect();
        self.persist_membership();
        expired
    }
}

/// The membership record lives next to the checkpoint.
fn membership_record_path(checkpoint: &std::path::Path) -> std::path::PathBuf {
    let mut name = checkpoint.as_os_str().to_owned();
    name.push(".membership");
    std::path::PathBuf::from(name)
}

fn write_membership_record(path: &std::path::Path, bytes: &[u8]) -> Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = std::path::PathBuf::from(tmp);
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// The task dispenser of `cfg` (S27 P2): restored from the record next to
/// the checkpoint on `--resume`, which retires every island process of the
/// record and puts their unreported positions back first.
fn initial_dispense(cfg: &Config) -> Result<Option<Arc<crate::dispense_service::DispenseService>>> {
    let Some(dispense) = cfg.dispense.clone() else { return Ok(None) };
    if cfg.shard.index != 0 {
        bail!("--dispense runs on shard 0 of a sharded head only");
    }
    let record = cfg.checkpoint_path.as_deref().map(crate::dispense_service::record_path);
    let restore = record.as_deref().filter(|p| cfg.resume && p.exists());
    let service = crate::dispense_service::DispenseService::new(
        dispense,
        cfg.dispense_log.clone(),
        cfg.syncer_epoch,
        cfg.island_hmac_key.clone(),
        restore,
    )?;
    info!(restored = restore.is_some(), "task dispenser on");
    Ok(Some(Arc::new(service)))
}

/// Registry membership for `cfg`: on resume the persisted record restores
/// the syncer epoch and the retired generations; the configured epoch only
/// raises the restored one.
fn initial_membership(cfg: &Config, id_limit: u32) -> Result<(Membership, Option<std::path::PathBuf>)> {
    let mut membership = Membership::new(cfg.membership.clone(), id_limit);
    let record = cfg.checkpoint_path.as_deref().map(membership_record_path);
    if let Some(path) = record.as_deref().filter(|p| cfg.resume && p.exists()) {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        membership
            .restore_persisted(&bytes)
            .with_context(|| format!("restore membership record {}", path.display()))?;
        info!(path = %path.display(), syncer_epoch = membership.syncer_epoch(), "membership record restored");
    }
    membership.raise_syncer_epoch(cfg.syncer_epoch);
    Ok((membership, record))
}

/// Adaptive grace window: after quorum, wait for stragglers only within the
/// slack the learners' compute
/// overlap leaves free — γ · (τ·ξ_step − ξ_quorum − ξ_sync), clamped to
/// [0, cap]. With no step-time estimate yet, fall back to the full cap.
fn adaptive_grace(
    tau: f64,
    gamma: f64,
    step_secs: Option<f64>,
    quorum_secs: f64,
    sync_secs: f64,
    cap: Duration,
) -> Duration {
    let Some(step) = step_secs else { return cap };
    let slack = tau * step - quorum_secs - sync_secs;
    Duration::from_secs_f64((gamma * slack).max(0.0)).min(cap)
}

/// Grace window of a round that just reached quorum and the floor in it:
/// the larger of the adaptive window and the floor, capped by `grace_ms`.
/// A `terminal` round (the last round of its fragment) waits at least
/// `terminal_grace_ms`, which the cap does not bound.
fn round_grace(
    cfg: &Config,
    terminal: bool,
    step_secs: Option<f64>,
    quorum_secs: f64,
    sync_secs: f64,
    gaps: &ArrivalGaps,
) -> (Duration, Duration) {
    let cap = Duration::from_millis(cfg.grace_ms);
    let adaptive =
        adaptive_grace(cfg.grace_tau, cfg.grace_gamma, step_secs, quorum_secs, sync_secs, cap);
    let mut floor = grace_floor(cfg.grace_min_ms, cfg.grace_arrival_quantile, gaps);
    let mut grace = adaptive.max(floor).min(cap);
    if terminal {
        let terminal_grace = Duration::from_millis(cfg.terminal_grace_ms);
        floor = floor.max(terminal_grace);
        grace = grace.max(terminal_grace);
    }
    (grace, floor)
}

/// Whether step `t` is the last round of its fragment within `total_steps`.
fn is_terminal_round(schedule: &crate::shard::ShardSchedule, t: u64, total_steps: u64) -> bool {
    schedule.next_merge_of_fragment(t) > total_steps
}

/// Floor of the grace window (`--grace-min-ms`, `--grace-arrival-quantile`).
/// The adaptive window sizes the wait from the learners' step time minus the
/// time the round took to reach quorum; with a deep pipeline a round opens
/// long before any push, so that slack is near zero even when the islands
/// deliver the same fragment seconds apart. The floor waits for that gap.
fn grace_floor(min_ms: u64, quantile: f64, gaps: &ArrivalGaps) -> Duration {
    let measured = (quantile > 0.0)
        .then(|| gaps.quantile(quantile))
        .flatten()
        .map_or(Duration::ZERO, Duration::from_secs_f64);
    Duration::from_millis(min_ms).max(measured)
}

/// Gaps kept for the arrival quantile (about P/H rounds of history at P=64).
const ARRIVAL_GAP_WINDOW: usize = 64;
/// First arrivals remembered for late pushes, oldest dropped first.
const ARRIVAL_FIRSTS_KEPT: usize = 512;

/// Recent gaps between the first accepted push of a round `(step, fragment)`
/// and each later push of that round, whether it joined the round or came
/// after the round closed (a late push). Only the grace floor reads them.
#[derive(Default)]
struct ArrivalGaps {
    gaps: std::collections::VecDeque<f64>,
    firsts: BTreeMap<(u64, usize), Instant>,
}

impl ArrivalGaps {
    /// An accepted push of an open round: the first one starts the clock,
    /// each later one is a gap.
    fn note_accepted(&mut self, step: u64, fragment: usize, now: Instant) {
        match self.firsts.get(&(step, fragment)) {
            Some(first) => self.note(now.saturating_duration_since(*first)),
            None => {
                self.firsts.insert((step, fragment), now);
                while self.firsts.len() > ARRIVAL_FIRSTS_KEPT {
                    self.firsts.pop_first();
                }
            }
        }
    }

    /// A push for a round that is no longer open.
    fn note_late(&mut self, step: u64, fragment: usize, now: Instant) {
        if let Some(first) = self.firsts.get(&(step, fragment)) {
            self.note(now.saturating_duration_since(*first));
        }
    }

    /// A relaunched attempt starts its own clock.
    fn forget(&mut self, step: u64, fragment: usize) {
        self.firsts.remove(&(step, fragment));
    }

    fn note(&mut self, gap: Duration) {
        if self.gaps.len() == ARRIVAL_GAP_WINDOW {
            self.gaps.pop_front();
        }
        self.gaps.push_back(gap.as_secs_f64());
    }

    /// Nearest-rank quantile of the kept gaps; None before the first gap.
    fn quantile(&self, q: f64) -> Option<f64> {
        if self.gaps.is_empty() {
            return None;
        }
        let mut sorted: Vec<f64> = self.gaps.iter().copied().collect();
        sorted.sort_by(f64::total_cmp);
        let rank = (q.clamp(0.0, 1.0) * sorted.len() as f64).ceil() as usize;
        Some(sorted[rank.clamp(1, sorted.len()) - 1])
    }
}

/// Round-launch floor: the manual ms floor, raised to H·ξ_step/P once a
/// learner step-time estimate exists (each fragment then re-merges after
/// ~H inner steps of the slowest learner). Anywhere natural round latency
/// exceeds this — any real WAN — the floor never binds.
fn launch_interval(
    manual_floor: Duration,
    h_target: f64,
    num_fragments: usize,
    step_secs: Option<f64>,
) -> Duration {
    let adaptive = match step_secs {
        Some(step) if h_target > 0.0 => {
            Duration::from_secs_f64(h_target * step / num_fragments.max(1) as f64)
        }
        _ => Duration::ZERO,
    };
    manual_floor.max(adaptive)
}

/// Per-learner inner-step duration estimated from consecutive pushes
/// (each push carries the learner's local_step), smoothed with an EMA like
/// the other grace-window inputs (ξ_quorum, ξ_sync) — a single push
/// interval is too noisy to size the grace window on its own.
#[derive(Default)]
struct StepRates(HashMap<Member, (Option<Instant>, u64, Option<f64>)>);

/// EMA smoothing: new estimate = α·sample + (1−α)·previous.
const STEP_EMA_ALPHA: f64 = 0.5;

impl StepRates {
    fn note(&mut self, member: Member, local_step: u64, now: Instant) {
        let entry = self.0.entry(member).or_insert((None, 0, None));
        // One local step produces a push per fragment (and heartbeats): only
        // the first report of a new step marks its start. Re-stamping on
        // every report of the same step would measure only the gap between
        // the last report of one step and the first of the next.
        match entry.0 {
            Some(prev) if local_step > entry.1 => {
                let secs = now.duration_since(prev).as_secs_f64() / (local_step - entry.1) as f64;
                entry.2 = Some(match entry.2 {
                    Some(ema) => STEP_EMA_ALPHA * secs + (1.0 - STEP_EMA_ALPHA) * ema,
                    None => secs, // first sample seeds the EMA
                });
                *entry = (Some(now), local_step, entry.2);
            }
            // A step counter that went backwards (a restarted learner)
            // starts a new baseline; a repeat of the current step keeps it.
            Some(_) if local_step < entry.1 => *entry = (Some(now), local_step, entry.2),
            Some(_) => {}
            None => *entry = (Some(now), local_step, None),
        }
    }

    /// Slowest estimated step time among one frozen/current member set.
    /// Historical or superseded generations must not pace future rounds.
    fn max_step_secs_for(&self, members: &[Member]) -> Option<f64> {
        members
            .iter()
            .filter_map(|member| self.0.get(member).and_then(|(_, _, estimate)| *estimate))
            .fold(None, |acc, v| Some(acc.map_or(v, |a: f64| a.max(v))))
    }

    fn remove(&mut self, member: Member) {
        self.0.remove(&member);
    }
}

type Registry = Arc<Mutex<RegistryState>>;
type Session = Arc<Mutex<Option<SessionSpec>>>;

/// Orders every cutoff-sensitive scheduler action against BUDGET_DONE.
///
/// `request` closes the gate before the decoded report is queued.  A
/// successful `try_linearize_work` is the linearization point for exactly one
/// subsequent commit, pull/retry, or compute submission.  Both operations use
/// the same mutex, so work that linearizes after a queued report is impossible;
/// work that won the mutex first is part of the authoritative pre-cutoff cut
/// and is allowed to finish before the scheduler quiesces accepted compute.
struct BudgetCutoff {
    enabled: bool,
    timeout: Duration,
    deadline: Mutex<Option<tokio::time::Instant>>,
    notify: tokio::sync::Notify,
}

impl BudgetCutoff {
    fn new(enabled: bool, timeout: Duration) -> Self {
        Self {
            enabled,
            timeout,
            deadline: Mutex::new(None),
            notify: tokio::sync::Notify::new(),
        }
    }

    /// Close the gate exactly once and establish the one absolute deadline
    /// shared by report enqueue, report collection, and compute draining.
    fn request(&self) -> Option<tokio::time::Instant> {
        if !self.enabled {
            return None;
        }
        let (deadline, first) = {
            let mut deadline = self.deadline.lock().unwrap();
            let first = deadline.is_none();
            if first {
                *deadline = Some(tokio::time::Instant::now() + self.timeout);
            }
            (*deadline, first)
        };
        if first {
            self.notify.notify_waiters();
        }
        deadline
    }

    fn deadline(&self) -> Option<tokio::time::Instant> {
        *self.deadline.lock().unwrap()
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Claim one cutoff-sensitive action in the total order.  The scheduler
    /// is the sole action executor, so it may release the mutex after this
    /// decision; a report that closes the gate afterwards is ordered after
    /// this already-claimed action and before every later action.
    fn try_linearize_work(&self) -> bool {
        !self.enabled || self.deadline.lock().unwrap().is_none()
    }

    async fn wait_requested(&self) {
        if !self.enabled {
            std::future::pending::<()>().await;
        }
        loop {
            // Register before checking the state so a request between those
            // operations cannot be missed by Notify's edge-triggered wakeup.
            let notified = self.notify.notified();
            if self.deadline().is_some() {
                return;
            }
            notified.await;
        }
    }
}

fn validate_config(cfg: &Config) -> Result<()> {
    if cfg.learners == 0 {
        bail!("--learners must be positive");
    }
    if cfg.quorum == 0 {
        bail!("--quorum must be positive");
    }
    if cfg.quorum > cfg.learners {
        bail!("--quorum must not exceed --learners");
    }
    if cfg.quorum_timeout_s == 0 {
        bail!("--quorum-timeout-s must be positive");
    }
    if cfg.mark_final_checkpoint && cfg.checkpoint_path.is_none() {
        bail!("--mark-final-checkpoint requires --checkpoint-path");
    }
    cfg.membership.validate(cfg.learners, cfg.quorum)?;
    if !cfg.membership.is_static() {
        // Dynamic membership needs rounds that may merge without some island.
        let fixed_roster =
            cfg.max_base_lag == Some(0) && cfg.quorum == cfg.learners && cfg.grace_ms == 0;
        if fixed_roster
            || cfg.policy_sweep_fragments.is_some()
            || cfg.learner_budget_steps.is_some()
        {
            bail!(
                "dynamic membership options (lease, soft deadline, capacity quorum, carry \
                 discount, max learners) need non-fixed rounds: not with a strict fixed roster, \
                 --policy-sweep-fragments, or --learner-budget-steps"
            );
        }
    }
    if let Some(fragments) = cfg.policy_sweep_fragments {
        if fragments == 0 {
            bail!("--policy-sweep-fragments must be positive");
        }
        if cfg.total_steps == 0 {
            bail!("--policy-sweep-fragments requires positive --total-steps");
        }
        // Fragment rounds of one sweep are independent (H=1, plain average,
        // no momentum), so up to P of them may gather concurrently; commits
        // stay in strict step order.
        if cfg.pipeline == 0 || cfg.pipeline > fragments {
            bail!("--policy-sweep-fragments requires --pipeline between 1 and the fragment count");
        }
        if !cfg.total_steps.is_multiple_of(u64::from(fragments)) {
            bail!("--total-steps must be divisible by --policy-sweep-fragments");
        }
        if cfg.max_base_lag != Some(0) || cfg.quorum != cfg.learners || cfg.grace_ms != 0 {
            bail!(
                "--policy-sweep-fragments requires a strict fixed roster: \
                 --max-base-lag 0, --quorum=--learners, and --grace-ms 0"
            );
        }
        if cfg.delta_correction {
            bail!("--policy-sweep-fragments requires --delta-correction none");
        }
        if cfg.outer_lr != 1.0 || cfg.outer_momentum != 0.0 {
            bail!("--policy-sweep-fragments requires --outer-lr 1 and --outer-momentum 0");
        }
        if cfg.outer_delayed_nesterov_period.is_some() {
            bail!("--policy-sweep-fragments does not support --outer-delayed-nesterov-period");
        }
        if cfg.learner_weight != LearnerWeight::Equal {
            bail!("--policy-sweep-fragments requires --learner-weight equal");
        }
        if cfg.expert_token_weights {
            bail!("--policy-sweep-fragments does not support --expert-token-weights");
        }
        // --resume is not required here: a fresh sweep has no checkpoint yet and
        // --resume refuses a missing file (see check_resume_checkpoint). The
        // launcher adds --resume only when the checkpoint exists.
        if cfg.checkpoint_path.is_none()
            || cfg.checkpoint_every == 0
            || (cfg.checkpoint_every != 1
                && !cfg.checkpoint_every.is_multiple_of(u64::from(fragments)))
        {
            bail!(
                "--policy-sweep-fragments requires --checkpoint-path and \
                 --checkpoint-every 1 or a multiple of the fragment count"
            );
        }
    }
    validate_shard_config(cfg)?;
    if matches!(cfg.bcast_delta_full_every, Some(n) if n < 2) {
        bail!("--bcast-delta-full-every must be at least 2");
    }
    if cfg.async_checkpoint {
        if cfg.policy_sweep_fragments.is_none() {
            bail!("--async-checkpoint requires --policy-sweep-fragments");
        }
        if cfg.learner_budget_steps.is_some() {
            bail!("--async-checkpoint does not support --learner-budget-steps");
        }
    }
    validate_eval_checkpoints(cfg)
}

/// `--checkpoint-at-steps` is for fragment rounds that write their
/// checkpoint at the quiescent cut after a round (`publish_round`).
fn validate_eval_checkpoints(cfg: &Config) -> Result<()> {
    if cfg.checkpoint_at_steps.is_empty() {
        return Ok(());
    }
    if cfg.checkpoint_path.is_none() {
        bail!("--checkpoint-at-steps requires --checkpoint-path (it writes <checkpoint-path>.step-<t>)");
    }
    crate::eval_checkpoint::validate_steps(&cfg.checkpoint_at_steps, cfg.total_steps)?;
    // A sweep checkpoint is the commit record of its broadcast; with
    // --async-checkpoint its own snapshot writer is already in flight.
    if cfg.policy_sweep_fragments.is_some() {
        bail!(
            "--checkpoint-at-steps does not support --policy-sweep-fragments: sweep checkpoints \
             are commit records written before their broadcast"
        );
    }
    if checkpoint_before_broadcast(cfg) {
        bail!(
            "--checkpoint-at-steps does not support --checkpoint-every 1: every step is already \
             written to --checkpoint-path before its broadcast, which stalls the scheduler anyway"
        );
    }
    if cfg.shard.is_sharded() {
        bail!(
            "--checkpoint-at-steps does not support --shard-count > 1: shards advance on their \
             own, so no shard holds the whole model at one step"
        );
    }
    Ok(())
}

/// A sharded head runs independent generic fragment rounds per shard.  Modes
/// whose semantics span all fragments (strict policy sweeps) or all shards (learner-budget cutoff) are refused.
fn validate_shard_config(cfg: &Config) -> Result<()> {
    if !cfg.shard.is_sharded() {
        return Ok(());
    }
    if cfg.policy_sweep_fragments.is_some() {
        bail!("--shard-count > 1 does not support --policy-sweep-fragments");
    }
    if cfg.learner_budget_steps.is_some() {
        bail!("--shard-count > 1 does not support --learner-budget-steps");
    }
    // Membership (leases, re-JOIN, capacity quorum, soft deadline, carry
    // discount) is per shard: every shard keeps its own view of the same
    // islands, each island holds one connection per shard and re-JOINs each
    // shard on its own.  No decision needs another shard's view, because a
    // round touches one fragment and that fragment has one owner.
    Ok(())
}

/// A per-round checkpoint is the commit record for an externally visible
/// fragment version. Persist it before BCAST regardless of the staleness
/// profile; otherwise a process crash can expose version t and resume at t-1.
fn checkpoint_before_broadcast(cfg: &Config) -> bool {
    cfg.checkpoint_path.is_some() && cfg.checkpoint_every == 1
}

/// Whether step `t` must be durable before its BCAST. Per-round checkpoints
/// always are; a policy sweep with `--checkpoint-every N` (N a multiple of
/// the fragment count) persists only every N-th step, at sweep boundaries,
/// and broadcasts the other steps without a durable record (a crash resumes
/// at the newest checkpoint; islands ahead of it fail closed on their next
/// push's local-step check).
fn persist_before_broadcast(cfg: &Config, t: u64) -> bool {
    checkpoint_before_broadcast(cfg)
        || (cfg.policy_sweep_fragments.is_some()
            && cfg.checkpoint_path.is_some()
            && cfg.checkpoint_every > 0
            && t.is_multiple_of(cfg.checkpoint_every))
}

fn validate_policy_sweep_push(push: &Push, fragments: u32) -> Result<()> {
    if fragments == 0 {
        bail!("policy sweep fragment count must be positive");
    }
    if push.global_step == 0 {
        bail!("policy-sweep push global_step must be positive");
    }
    let policy_round = push.global_step.div_ceil(u64::from(fragments));
    if push.local_step != policy_round {
        bail!(
            "policy-sweep push at global step {} requires local_step {}, got {}",
            push.global_step,
            policy_round,
            push.local_step
        );
    }
    if push.c_steps != 1 {
        bail!(
            "policy-sweep push at global step {} requires c_steps=1, got {}",
            push.global_step,
            push.c_steps
        );
    }
    Ok(())
}

fn expected_sweep_fragment_version(global_step: u64, fragment: u32, fragments: u32) -> u64 {
    let first = u64::from(fragment) + 1;
    if global_step < first {
        0
    } else {
        first + ((global_step - first) / u64::from(fragments)) * u64::from(fragments)
    }
}

fn validate_resumed_policy_sweep(cfg: &Config, st: &GlobalState) -> Result<()> {
    let Some(fragments) = cfg.policy_sweep_fragments else {
        return Ok(());
    };
    if st.policy_sweep_fragments != Some(fragments) {
        bail!(
            "policy-sweep checkpoint profile {:?} does not match configured profile {fragments}",
            st.policy_sweep_fragments
        );
    }
    if st.global_step > cfg.total_steps {
        bail!(
            "policy-sweep checkpoint step {} exceeds configured total {}",
            st.global_step,
            cfg.total_steps
        );
    }
    for (fragment, actual) in st.versions.iter().copied().enumerate() {
        let expected = expected_sweep_fragment_version(
            st.global_step,
            u32::try_from(fragment).context("fragment index does not fit u32")?,
            fragments,
        );
        if actual != expected {
            bail!(
                "policy-sweep checkpoint fragment {fragment} has version {actual}, expected {expected}"
            );
        }
    }
    let expected_steps = st.global_step / u64::from(fragments);
    if st.global_step == 0 {
        if !st.ledger.is_empty() {
            bail!("policy-sweep checkpoint at step zero has learner accounting");
        }
        return Ok(());
    }
    if st.ledger.len() != cfg.learners as usize {
        bail!(
            "policy-sweep checkpoint has accounting for {} learners, expected {}",
            st.ledger.len(),
            cfg.learners
        );
    }
    for learner_id in 0..cfg.learners {
        let ledger = st
            .ledger
            .get(&learner_id)
            .with_context(|| format!("policy-sweep checkpoint is missing learner {learner_id}"))?;
        if ledger.merges != st.global_step || ledger.steps != expected_steps {
            bail!(
                "policy-sweep checkpoint learner {learner_id} has merges={} steps={}, expected merges={} steps={expected_steps}",
                ledger.merges,
                ledger.steps,
                st.global_step
            );
        }
    }
    Ok(())
}

/// `--resume` must point at an existing checkpoint file.
fn check_resume_checkpoint(cfg: &Config) -> Result<()> {
    if cfg.resume {
        let path = cfg
            .checkpoint_path
            .as_ref()
            .context("--resume requires --checkpoint-path")?;
        if !path.is_file() {
            bail!(
                "--resume checkpoint does not exist or is not a file: {}",
                path.display()
            );
        }
    }
    Ok(())
}

fn profile_record_path(checkpoint: &std::path::Path) -> std::path::PathBuf {
    let mut name = checkpoint.as_os_str().to_owned();
    name.push(".profile");
    std::path::PathBuf::from(name)
}

/// Every start with a checkpoint path records this syncer's semantic profile
/// hash next to the checkpoint. With `--require-profile-record`, `--resume`
/// refuses a checkpoint whose record differs (the merge settings of a session
/// may not change under it) or is missing (written before the record
/// existed). Without the flag nothing is checked: some restarts change the
/// profile on purpose (the learner-budget consolidation restart).
fn check_and_record_profile(cfg: &Config, profile: &[u8; 32]) -> Result<()> {
    let Some(checkpoint) = cfg.checkpoint_path.as_ref() else {
        return Ok(());
    };
    let record = profile_record_path(checkpoint);
    let ours = hex32(profile);
    if cfg.resume {
        match std::fs::read_to_string(&record) {
            Ok(recorded) if recorded.trim() == ours || !cfg.require_profile_record => {}
            Ok(recorded) => bail!(
                "cannot resume {}: the session was started with other sync settings \
                 (semantic profile {}, this syncer {ours}); relaunch with the session's \
                 original sync options (a session started before the async defaults: \
                 --rl-sync-mode strict)",
                checkpoint.display(),
                recorded.trim()
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                if cfg.require_profile_record {
                    bail!(
                        "cannot resume {}: the checkpoint has no semantic profile record, so it \
                         was written by an older syncer under the strict defaults; relaunch \
                         with --rl-sync-mode strict (or the session's original explicit sync \
                         options)",
                        checkpoint.display()
                    );
                }
            }
            Err(error) => {
                return Err(error).with_context(|| format!("read {}", record.display()));
            }
        }
    }
    let partial = profile_record_path(&checkpoint.with_extension("profile-tmp"));
    std::fs::write(&partial, format!("{ours}\n"))
        .with_context(|| format!("write {}", partial.display()))?;
    std::fs::rename(&partial, &record).with_context(|| format!("write {}", record.display()))?;
    Ok(())
}

pub async fn run(cfg: Config) -> Result<()> {
    validate_config(&cfg)?;
    check_resume_checkpoint(&cfg)?;
    let semantic_profile_hash = cfg.semantic_profile_hash();
    check_and_record_profile(&cfg, &semantic_profile_hash)?;
    if let Some(steps) = cfg.learner_budget_steps {
        if !(1..=u32::MAX as u64).contains(&steps) {
            bail!("--learner-budget-steps must fit a positive u32");
        }
        if cfg.checkpoint_path.is_none() || cfg.mark_final_checkpoint || cfg.resume {
            bail!("--learner-budget-steps requires a fresh unmarked checkpoint");
        }
        // Budget runs are always fresh. Remove an old publishable marker
        // before accepting any learner so every early failure stays recovery-only.
        remove_final_marker(cfg.checkpoint_path.as_ref().unwrap())?;
    }
    let listener = TcpListener::bind(("0.0.0.0", cfg.port))
        .await
        .with_context(|| format!("bind port {}", cfg.port))?;
    let semantic_profile_sha256: String = semantic_profile_hash
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    info!(
        port = cfg.port,
        semantic_profile_sha256,
        require_profile_binding = cfg.require_profile_binding,
        "syncer listening"
    );
    let (event_tx, event_rx) = mpsc::channel::<Event>(1024);
    let expected_learners = cfg.membership.id_limit(cfg.learners);
    let (membership, membership_record) = initial_membership(&cfg, expected_learners)?;
    let dispense = initial_dispense(&cfg)?;
    let registry: Registry = Arc::new(Mutex::new(RegistryState {
        membership_record,
        dispense: dispense.clone(),
        ..RegistryState::new(membership)
    }));
    registry.lock().unwrap().persist_membership();
    let session: Session = Arc::new(Mutex::new(None));
    let budget_cutoff = Arc::new(BudgetCutoff::new(
        cfg.learner_budget_steps.is_some(),
        Duration::from_secs(cfg.quorum_timeout_s),
    ));

    let accept_registry = registry.clone();
    let accept_session = session.clone();
    let accept_budget_cutoff = budget_cutoff.clone();
    let require_profile_binding = cfg.require_profile_binding;
    let expert_token_weights = cfg.expert_token_weights;
    let hello_auth = Arc::new(HelloAuth {
        key: cfg.island_hmac_key.clone(),
        expected_contract: cfg.expected_island_contract,
    });
    if hello_auth.key.is_none() {
        warn!("syncer runs WITHOUT island authentication (HELLO is not HMAC-checked)");
    }
    tokio::spawn(async move {
        loop {
            match listener.accept().await {
                Ok((stream, peer)) => {
                    let reg = accept_registry.clone();
                    let session = accept_session.clone();
                    let budget_cutoff = accept_budget_cutoff.clone();
                    let tx = event_tx.clone();
                    let hello_auth = hello_auth.clone();
                    let dispense = dispense.clone();
                    tokio::spawn(async move {
                        if let Err(e) = handle_connection(
                            stream,
                            reg,
                            session,
                            expected_learners,
                            semantic_profile_hash,
                            require_profile_binding,
                            expert_token_weights,
                            tx,
                            budget_cutoff,
                            hello_auth,
                            dispense,
                        )
                        .await
                        {
                            warn!(%peer, "connection ended: {e:#}");
                        }
                    });
                }
                Err(e) => warn!("accept failed: {e}"),
            }
        }
    });

    scheduler(cfg, event_rx, registry, budget_cutoff).await
}

fn negotiated_payload_limits(layout: &Layout, dtype: u8, expert_tags: bool) -> Result<(u64, u64)> {
    let mut max_init = 0u64;
    let mut max_push = 0u64;
    for fragment in &layout.fragments {
        let numel = fragment.numel()?;
        let init_bytes = tensor_nbytes(bulk_dtype(dtype), numel)? as u64;
        let segments = push_segments(dtype, numel, &fragment.tensor_numels)?;
        let mut delta_bytes = max_push_nbytes_segments(dtype, &segments)? as u64;
        if expert_tags {
            delta_bytes += expert_tags_nbytes(fragment.tensor_numels.len())? as u64;
        }
        max_init = max_init.max(
            4u64.checked_add(init_bytes)
                .context("INIT_PARAMS payload limit overflow")?,
        );
        max_push = max_push.max(
            48u64
                .checked_add(delta_bytes)
                .context("PUSH_FRAGMENT payload limit overflow")?,
        );
    }
    Ok((max_init, max_push))
}

fn group_label(g: &str) -> &str {
    if g.is_empty() { "未声明" } else { g }
}

fn hex32(h: &[u8; 32]) -> String {
    h.iter().map(|b| format!("{b:02x}")).collect()
}

/// Legacy HELLO authentication and the head-pinned island contract
/// (secret-handling-hardening).
pub struct HelloAuth {
    pub key: Option<Vec<u8>>,
    pub expected_contract: Option<[u8; 32]>,
}

/// Domain of the HELLO MAC: HMAC-SHA256(key, domain || hello_body).
pub const HELLO_MAC_DOMAIN: &[u8] = b"yeto-hello-mac-v1\0";
/// Domain of the RL session contract (yeto.rl.engine.backend_identity).
pub const SESSION_CONTRACT_DOMAIN: &[u8] = b"yeto-rl-session-contract-v2\0";

/// With a key: verify and strip the trailing 32-byte HMAC of a HELLO.
/// Without a key: the payload is the HELLO body (unauthenticated mode).
fn open_hello<'a>(key: Option<&[u8]>, payload: &'a [u8]) -> Result<&'a [u8]> {
    let Some(key) = key else { return Ok(payload) };
    let mut framed = Vec::with_capacity(payload.len() + 1);
    if payload.len() < crate::hmac::HMAC_LEN {
        bail!("HELLO shorter than its HMAC (island has no YETO_ISLAND_HMAC_KEY?)");
    }
    let (body, mac) = payload.split_at(payload.len() - crate::hmac::HMAC_LEN);
    framed.extend_from_slice(HELLO_MAC_DOMAIN);
    framed.extend_from_slice(body);
    let expected = crate::hmac::hmac_sha256(key, &framed);
    let diff = expected.iter().zip(mac).fold(0u8, |acc, (x, y)| acc | (x ^ y));
    if diff != 0 {
        bail!("HELLO HMAC mismatch (wrong or missing island HMAC key)");
    }
    Ok(body)
}

/// The session contract an island with the pinned contract must offer.
fn pinned_session_contract(layout_fingerprint: &[u8; 32], pinned: &[u8; 32]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(SESSION_CONTRACT_DOMAIN);
    h.update(layout_fingerprint);
    h.update(pinned);
    h.finalize().into()
}

fn check_pinned_contract(pinned: Option<[u8; 32]>, offered: &SessionSpec) -> Option<String> {
    let pinned = pinned?;
    let want = pinned_session_contract(&offered.layout_fingerprint, &pinned);
    (want != offered.session_contract_hash).then(|| {
        format!(
            "session mismatch (HELLO refused, session keeps running): the head pins \
             island contract {}, expected session_contract_hash={}, got {}",
            hex32(&pinned),
            hex32(&want),
            hex32(&offered.session_contract_hash),
        )
    })
}

/// Admit `offered` into the session: the first HELLO establishes the session
/// contract; later HELLOs must match it exactly. Returns the refusal message
/// (sent to that connection only) on mismatch. Never fatal for the session.
fn admit_session(session: &Session, offered: SessionSpec) -> Option<String> {
    let mut guard = session.lock().unwrap();
    match guard.as_ref() {
        None => {
            *guard = Some(offered);
            None
        }
        Some(expected) if expected == &offered => None,
        Some(expected) if expected.compat_group != offered.compat_group => Some(format!(
            "session mismatch (HELLO refused, session keeps running): \
             兼容组不同：{} 对 {}，容差未标定 (compat_group differs: expected {}, got {}; \
             tolerance not calibrated)",
            group_label(&expected.compat_group),
            group_label(&offered.compat_group),
            group_label(&expected.compat_group),
            group_label(&offered.compat_group),
        )),
        Some(expected) => Some(format!(
            "session mismatch (HELLO refused, session keeps running): \
             expected session_contract_hash={} layout_fingerprint={} dtype={} fragments={}, \
             got session_contract_hash={} layout_fingerprint={} dtype={} fragments={}",
            hex32(&expected.session_contract_hash),
            hex32(&expected.layout_fingerprint),
            expected.dtype,
            expected.layout.fragments.len(),
            hex32(&offered.session_contract_hash),
            hex32(&offered.layout_fingerprint),
            offered.dtype,
            offered.layout.fragments.len(),
        )),
    }
}

/// Length of the optional compat_group trailer at the end of `rest` (the
/// HELLO bytes after the session contract hash), 0 when there is none. The
/// part before the trailer must be 2 (num_streams) or 34 (profile +
/// num_streams) bytes; the trailer is b"YCG1" | u32 n | n bytes, 1 <= n <= 64.
fn hello_compat_trailer_len(rest: &[u8]) -> usize {
    for base in [2usize, 34] {
        if rest.len() < base + 8 || &rest[base..base + 4] != HELLO_COMPAT_MAGIC {
            continue;
        }
        let n = u32::from_le_bytes(rest[base + 4..base + 8].try_into().unwrap()) as usize;
        if (1..=64).contains(&n) && rest.len() == base + 8 + n {
            return 8 + n;
        }
    }
    0
}

/// Parse a HELLO with its optional extension trailer. A body that only
/// looks like it ends in a trailer is parsed whole as a legacy HELLO.
fn parse_hello(payload: &[u8], expected_learners: u32) -> Result<ParsedHello> {
    let (body, ext) = HelloExt::split(payload);
    let Some(ext) = ext else { return parse_hello_body(payload, expected_learners) };
    match parse_hello_body(body, expected_learners) {
        Ok(parsed) if ext.expert_tokens => {
            let (_, max_push_payload) = negotiated_payload_limits(&parsed.layout, parsed.dtype, true)?;
            Ok(ParsedHello { ext, max_push_payload, ..parsed })
        }
        Ok(parsed) => Ok(ParsedHello { ext, ..parsed }),
        Err(error) => parse_hello_body(payload, expected_learners).map_err(|_| error),
    }
}

fn parse_hello_body(payload: &[u8], expected_learners: u32) -> Result<ParsedHello> {
    let mut r = Reader(payload);
    let version = r.u16()?;
    if version != PROTOCOL_VERSION {
        bail!("wire protocol version mismatch: server={PROTOCOL_VERSION}, client={version}");
    }
    let learner_id = r.u32()?;
    if learner_id >= expected_learners {
        bail!("learner id {learner_id} is outside configured range 0..{expected_learners}");
    }
    let generation = r.u64()?;
    if generation == 0 {
        bail!("connection generation must be nonzero");
    }
    let dtype = r.u8()?;
    if !valid_session_dtype(dtype) {
        bail!("unsupported session dtype {dtype}");
    }
    let num_fragments = r.u32()?;
    let layout = Layout::decode(&mut r, num_fragments)?;
    let layout_fingerprint = r
        .take(32)?
        .try_into()
        .context("layout fingerprint must be 32 bytes")?;
    let session_contract_hash = r
        .take(32)?
        .try_into()
        .context("session contract hash must be 32 bytes")?;
    // Optional compat_group trailer (7.7c) after num_streams: find where the
    // trailer starts so the profile extension is still decided by length.
    let trailer_len = hello_compat_trailer_len(r.0);
    let ext_len = r.remaining() - trailer_len;
    let syncer_profile_hash = match ext_len {
        // Legacy/non-SAO HELLO.
        2 => None,
        // Profile-bound HELLO. Keeping the extension before num_streams lets
        // protocol-v4 generic clients retain their exact wire bytes.
        34 => Some(
            r.take(32)?
                .try_into()
                .context("syncer profile hash must be 32 bytes")?,
        ),
        remaining => bail!("invalid HELLO profile extension length {remaining}"),
    };
    let num_streams = r.u16()?;
    let compat_group = if trailer_len == 0 {
        String::new()
    } else {
        let _magic = r.take(4)?;
        let n = r.u32()? as usize;
        String::from_utf8(r.take(n)?.to_vec()).context("HELLO compat_group is not utf-8")?
    };
    if r.remaining() != 0 {
        bail!("trailing bytes in HELLO");
    }
    if num_streams > 256 {
        bail!("num_streams {num_streams} exceeds limit 256");
    }
    let (max_init_payload, max_push_payload) = negotiated_payload_limits(&layout, dtype, false)?;
    Ok(ParsedHello {
        learner_id,
        generation,
        dtype,
        layout,
        layout_fingerprint,
        session_contract_hash,
        syncer_profile_hash,
        num_streams,
        max_init_payload,
        max_push_payload,
        compat_group,
        ext: HelloExt::default(),
    })
}

fn parse_data_hello(payload: &[u8]) -> Result<(Member, u16)> {
    let mut r = Reader(payload);
    let version = r.u16()?;
    if version != PROTOCOL_VERSION {
        bail!("wire protocol version mismatch: server={PROTOCOL_VERSION}, client={version}");
    }
    let member = Member {
        learner_id: r.u32()?,
        generation: r.u64()?,
    };
    if member.generation == 0 {
        bail!("connection generation must be nonzero");
    }
    let stream_idx = r.u16()?;
    if r.remaining() != 0 {
        bail!("trailing bytes in DATA_HELLO");
    }
    Ok((member, stream_idx))
}

async fn handle_connection(
    stream: TcpStream,
    registry: Registry,
    session: Session,
    expected_learners: u32,
    semantic_profile_hash: [u8; 32],
    require_profile_binding: bool,
    expert_token_weights: bool,
    event_tx: mpsc::Sender<Event>,
    budget_cutoff: Arc<BudgetCutoff>,
    hello_auth: Arc<HelloAuth>,
    dispense: Option<Arc<crate::dispense_service::DispenseService>>,
) -> Result<()> {
    stream.set_nodelay(true)?;
    let (mut rd, mut wr) = stream.into_split();
    let first = match read_frame_limited(&mut rd, |msg_type| match msg_type {
        MSG_HELLO | MSG_DATA_HELLO | MSG_BW_PROBE => Ok(MAX_HELLO_FRAME),
        MSG_TASK_HELLO => Ok(4096),
        other => bail!("first frame must be HELLO/DATA_HELLO/BW_PROBE/TASK_HELLO, got {other}"),
    })
    .await
    {
        Ok(frame) => frame,
        Err(error) => {
            if error.downcast_ref::<std::io::Error>().is_none() {
                let _ = send_direct(
                    &mut wr,
                    MSG_ERROR,
                    format!("invalid first frame: {error:#}").as_bytes(),
                )
                .await;
            }
            return Err(error);
        }
    };
    match first.msg_type {
        MSG_HELLO => {
            let body = match open_hello(hello_auth.key.as_deref(), &first.payload) {
                Ok(body) => body,
                Err(error) => {
                    let message = format!("HELLO authentication failed: {error:#}");
                    let _ = send_direct(&mut wr, MSG_ERROR, message.as_bytes()).await;
                    return Err(error);
                }
            };
            let parsed = match parse_hello(body, expected_learners) {
                Ok(parsed) => parsed,
                Err(error) => {
                    let message = format!("invalid HELLO: {error:#}");
                    let _ = send_direct(&mut wr, MSG_ERROR, message.as_bytes()).await;
                    return Err(error);
                }
            };
            let ParsedHello {
                learner_id,
                generation,
                dtype,
                layout,
                layout_fingerprint,
                session_contract_hash,
                syncer_profile_hash,
                num_streams,
                max_init_payload,
                max_push_payload,
                compat_group,
                ext,
            } = parsed;
            let profile_error = match syncer_profile_hash {
                Some(offered) if offered != semantic_profile_hash => {
                    Some("HELLO syncer semantic profile does not match the running server")
                }
                None if require_profile_binding => {
                    Some("HELLO is missing the required syncer semantic profile binding")
                }
                _ if ext.expert_tokens != expert_token_weights => Some(if expert_token_weights {
                    "the syncer weights experts by routed tokens; this HELLO does not send expert token counts"
                } else {
                    "this HELLO sends expert token counts; the syncer runs without expert token weights"
                }),
                _ => None,
            };
            if let Some(message) = profile_error {
                // Refuse this connection only; the running session is unaffected.
                send_direct(&mut wr, MSG_ERROR, message.as_bytes()).await?;
                warn!(learner_id, generation, "rejected HELLO, session keeps running: {message}");
                return Ok(());
            }
            let num_fragments = layout.fragments.len();
            let offered = SessionSpec {
                dtype,
                layout: layout.clone(),
                layout_fingerprint,
                session_contract_hash,
                syncer_profile_hash,
                compat_group,
            };
            // A HELLO whose contract differs from the
            // established session is refused on this connection only (MSG_ERROR
            // naming both hashes). The syncer and the islands already admitted
            // keep running; one mis-wired island (e.g. a Miles island in a verl
            // session) must not stop the whole run. The first HELLO still
            // establishes the session contract.
            if let Some(message) = check_pinned_contract(hello_auth.expected_contract, &offered) {
                send_direct(&mut wr, MSG_ERROR, message.as_bytes()).await?;
                warn!(learner_id, generation, "rejected HELLO, session keeps running: {message}");
                return Ok(());
            }
            if let Some(message) = admit_session(&session, offered) {
                send_direct(&mut wr, MSG_ERROR, message.as_bytes()).await?;
                warn!(learner_id, generation, "rejected HELLO, session keeps running: {message}");
                return Ok(());
            }
            let member = Member {
                learner_id,
                generation,
            };
            let max_chunked_inner = 13u64
                .checked_add(max_init_payload.max(max_push_payload))
                .context("chunked inner-frame limit overflow")?;
            let (tx, rx) = mpsc::channel::<OutFrame>(WRITER_QUEUE);
            let group = Arc::new(Group {
                member,
                dtype,
                expert_tokens: ext.expert_tokens,
                layout,
                layout_fingerprint,
                session_contract_hash,
                num_streams,
                max_init_payload,
                max_push_payload,
                max_chunked_inner,
                control: tx,
                data: Mutex::new(HashMap::new()),
                msg_id: AtomicU64::new(0),
                rr: AtomicUsize::new(0),
                reasm: Mutex::new(HashMap::new()),
            });
            let registration =
                { registry.lock().unwrap().register_group_with(group.clone(), Instant::now(), ext) };
            let replaced = match registration {
                Ok((replaced, admission)) => {
                    if admission == Admission::Rejoin {
                        info!(learner_id, generation, "island re-joined after lease expiry");
                    }
                    if ext.recover {
                        info!(learner_id, generation, "island asks for peer recovery");
                    }
                    replaced
                }
                Err(error) => {
                    let message = match error {
                        RegisterError::Duplicate => format!(
                            "duplicate connection generation {generation} for learner {learner_id}"
                        ),
                        // Starts with the fence prefix so the client stops
                        // instead of redialing the same stale epoch.
                        RegisterError::Fenced(reason @ AdmitError::StaleSyncerEpoch { .. }) => format!(
                            "{reason} (learner {learner_id} generation {generation}); relaunch the island for this syncer"
                        ),
                        RegisterError::Fenced(reason) => format!(
                            "refused generation {generation} for learner {learner_id}: {reason}"
                        ),
                    };
                    send_direct(&mut wr, MSG_ERROR, message.as_bytes()).await?;
                    bail!(message);
                }
            };
            tokio::spawn(writer_task(wr, rx));
            info!(
                learner_id,
                generation, num_streams, "learner connected (layout: {} fragments)", num_fragments
            );
            if let Some(previous) = replaced {
                warn!(
                    learner_id,
                    old_generation = previous.generation,
                    generation,
                    "new connection generation supersedes learner for future rounds"
                );
            }
            event_tx
                .send(Event::Hello {
                    group: group.clone(),
                })
                .await
                .ok();
            let res = read_loop(&mut rd, &group, &event_tx, &budget_cutoff).await;
            registry.lock().unwrap().unregister(member);
            event_tx.send(Event::Disconnected { member }).await.ok();
            res
        }
        MSG_DATA_HELLO => {
            let (member, stream_idx) = match parse_data_hello(&first.payload) {
                Ok(parsed) => parsed,
                Err(error) => {
                    let message = format!("invalid DATA_HELLO: {error:#}");
                    let _ = send_direct(&mut wr, MSG_ERROR, message.as_bytes()).await;
                    return Err(error);
                }
            };
            let learner_id = member.learner_id;
            let generation = member.generation;
            // The control socket's HELLO may still be in flight; wait for it.
            let mut group = None;
            for _ in 0..200 {
                group = registry.lock().unwrap().groups.get(&member).cloned();
                if group.is_some() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            let Some(group) = group else {
                let message =
                    format!("DATA_HELLO for unknown learner {learner_id} generation {generation}");
                send_direct(&mut wr, MSG_ERROR, message.as_bytes()).await?;
                bail!(message);
            };
            if stream_idx >= group.num_streams {
                let message = format!(
                    "data stream index {stream_idx} out of bounds for {} streams",
                    group.num_streams
                );
                send_direct(&mut wr, MSG_ERROR, message.as_bytes()).await?;
                bail!(message);
            }
            let (tx, rx) = mpsc::channel::<OutFrame>(WRITER_QUEUE);
            let duplicate = {
                let mut data = group.data.lock().unwrap();
                if data.contains_key(&stream_idx) {
                    true
                } else {
                    data.insert(stream_idx, tx);
                    false
                }
            };
            if duplicate {
                let message = format!("duplicate data stream index {stream_idx}");
                send_direct(&mut wr, MSG_ERROR, message.as_bytes()).await?;
                bail!(message);
            }
            tokio::spawn(writer_task(wr, rx));
            read_loop(&mut rd, &group, &event_tx, &budget_cutoff).await
        }
        MSG_BW_PROBE => {
            let request = open_hello(hello_auth.key.as_deref(), &first.payload)
                .and_then(parse_bw_probe);
            let (upload, download) = match request {
                Ok(sizes) => sizes,
                Err(error) => {
                    let message = format!("invalid BW_PROBE: {error:#}");
                    let _ = send_direct(&mut wr, MSG_ERROR, message.as_bytes()).await;
                    return Err(error);
                }
            };
            let Some(_slot) = BwProbeSlot::acquire() else {
                let message = "too many concurrent BW_PROBE connections";
                send_direct(&mut wr, MSG_ERROR, message.as_bytes()).await?;
                bail!(message);
            };
            tokio::time::timeout(BW_PROBE_TIMEOUT, serve_bw_probe(&mut rd, &mut wr, upload, download))
                .await
                .context("BW_PROBE timed out")?
        }
        MSG_TASK_HELLO => match dispense {
            Some(service) => crate::dispense_service::serve(rd, wr, &first.payload, service).await,
            None => {
                let message = "dispense refused: this syncer has no task dispenser (--dispense)";
                let _ = send_direct(&mut wr, MSG_ERROR, message.as_bytes()).await;
                bail!(message)
            }
        },
        t => bail!("first frame must be HELLO/DATA_HELLO/BW_PROBE/TASK_HELLO, got {t}"),
    }
}

/// Largest upload or download one BW_PROBE connection may ask for.
const BW_PROBE_MAX_BYTES: u64 = 1 << 30;
const BW_PROBE_MAX_CONNECTIONS: usize = 64;
const BW_PROBE_TIMEOUT: Duration = Duration::from_secs(300);
static BW_PROBE_ACTIVE: AtomicUsize = AtomicUsize::new(0);

/// One of at most BW_PROBE_MAX_CONNECTIONS concurrent probe connections.
struct BwProbeSlot;

impl BwProbeSlot {
    fn acquire() -> Option<Self> {
        let previous = BW_PROBE_ACTIVE.fetch_add(1, Ordering::AcqRel);
        if previous >= BW_PROBE_MAX_CONNECTIONS {
            BW_PROBE_ACTIVE.fetch_sub(1, Ordering::AcqRel);
            return None;
        }
        Some(BwProbeSlot)
    }
}

impl Drop for BwProbeSlot {
    fn drop(&mut self) {
        BW_PROBE_ACTIVE.fetch_sub(1, Ordering::AcqRel);
    }
}

/// BW_PROBE body (after the HELLO-style HMAC trailer is removed):
/// u64 upload bytes, u64 download bytes, both at most BW_PROBE_MAX_BYTES.
fn parse_bw_probe(body: &[u8]) -> Result<(u64, u64)> {
    if body.len() != 16 {
        bail!("BW_PROBE body has {} bytes, expected 16", body.len());
    }
    let upload = u64::from_le_bytes(body[0..8].try_into().unwrap());
    let download = u64::from_le_bytes(body[8..16].try_into().unwrap());
    if upload > BW_PROBE_MAX_BYTES || download > BW_PROBE_MAX_BYTES {
        bail!("BW_PROBE asks for more than {BW_PROBE_MAX_BYTES} bytes per direction");
    }
    Ok((upload, download))
}

/// Read and discard `upload` raw bytes, acknowledge with their count, then
/// write `download` zero bytes and close. Nothing touches session state.
async fn serve_bw_probe(
    rd: &mut (impl tokio::io::AsyncReadExt + Unpin),
    wr: &mut OwnedWriteHalf,
    upload: u64,
    download: u64,
) -> Result<()> {
    let mut buf = vec![0u8; 1 << 20];
    let mut remaining = upload;
    while remaining > 0 {
        let want = remaining.min(buf.len() as u64) as usize;
        let got = rd.read(&mut buf[..want]).await?;
        if got == 0 {
            bail!("BW_PROBE upload ended after {} of {upload} bytes", upload - remaining);
        }
        remaining -= got as u64;
    }
    wr.write_all(&upload.to_le_bytes()).await?;
    buf.fill(0);
    let mut remaining = download;
    while remaining > 0 {
        let take = remaining.min(buf.len() as u64) as usize;
        wr.write_all(&buf[..take]).await?;
        remaining -= take as u64;
    }
    wr.shutdown().await?;
    Ok(())
}

async fn send_direct(wr: &mut OwnedWriteHalf, msg_type: u8, payload: &[u8]) -> Result<()> {
    let mut header = [0u8; 13];
    header[0..4].copy_from_slice(&MAGIC.to_le_bytes());
    header[4] = msg_type;
    header[5..13].copy_from_slice(&(payload.len() as u64).to_le_bytes());
    wr.write_all(&header).await?;
    wr.write_all(payload).await?;
    Ok(())
}

async fn writer_task(mut wr: OwnedWriteHalf, mut rx: mpsc::Receiver<OutFrame>) {
    while let Some(frame) = rx.recv().await {
        let len: usize = frame.parts.iter().map(|p| p.len()).sum();
        let mut header = [0u8; 13];
        header[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        header[4] = frame.msg_type;
        header[5..13].copy_from_slice(&(len as u64).to_le_bytes());
        let write = async {
            wr.write_all(&header).await?;
            for part in &frame.parts {
                wr.write_all(part).await?;
            }
            std::io::Result::Ok(())
        };
        match tokio::time::timeout(WRITE_TIMEOUT, write).await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                warn!("write failed: {e}");
                return;
            }
            Err(_) => {
                warn!("write timed out; dropping connection");
                return;
            }
        }
    }
}

async fn read_loop(
    rd: &mut (impl tokio::io::AsyncReadExt + Unpin),
    group: &Arc<Group>,
    event_tx: &mpsc::Sender<Event>,
    budget_cutoff: &BudgetCutoff,
) -> Result<()> {
    loop {
        let mut header_at: Option<Instant> = None;
        let frame = match read_frame_limited(rd, |msg_type| {
            header_at = Some(Instant::now());
            match msg_type {
                MSG_INIT_PARAMS => Ok(group.max_init_payload),
                MSG_INIT_DIGEST => Ok(init_digest_payload_len(group.layout.fragments.len())),
                MSG_PUSH_FRAGMENT => Ok(group.max_push_payload),
                MSG_HEARTBEAT => Ok(12),
                MSG_LEAVE => Ok(5),
                MSG_FINAL_ACK => Ok(10),
                MSG_BUDGET_DONE => Ok(8),
                MSG_CHUNK => Ok(CHUNK_HEADER_SIZE + CHUNK_SIZE as u64),
                recovery::MSG_PEER_SNAPSHOT_READY
                | recovery::MSG_RECOVERY_RESTORED
                | recovery::MSG_RECOVERY_FAILED => Ok(recovery::MAX_RECOVERY_FRAME),
                other => bail!(
                    "unexpected direct message type {other} from learner {} generation {}",
                    group.member.learner_id,
                    group.member.generation
                ),
            }
        })
        .await
        {
            Ok(frame) => frame,
            Err(error) => {
                if error.downcast_ref::<std::io::Error>().is_none() {
                    report_protocol_error(group, &error).await;
                }
                return Err(error);
            }
        };
        let dispatched = match frame.msg_type {
            MSG_CHUNK => match reassemble(group, &frame.payload) {
                Ok(Some((inner, recv_seconds))) => {
                    dispatch_inner(
                        group,
                        inner.msg_type,
                        &inner.payload,
                        recv_seconds,
                        event_tx,
                        budget_cutoff,
                    )
                    .await
                }
                Ok(None) => Ok(()),
                Err(error) => Err(error),
            },
            t => {
                let recv_seconds = header_at.map_or(0.0, |at| at.elapsed().as_secs_f64());
                dispatch_inner(group, t, &frame.payload, recv_seconds, event_tx, budget_cutoff).await
            }
        };
        if let Err(error) = dispatched {
            report_protocol_error(group, &error).await;
            return Err(error);
        }
    }
}

async fn report_protocol_error(group: &Arc<Group>, error: &anyhow::Error) {
    let _ = group
        .send_small(MSG_ERROR, bytes::Bytes::from(format!("{error:#}")))
        .await;
}

/// Adds one chunk; a completed message comes back with its receive time
/// (first chunk to last chunk).
fn reassemble(group: &Arc<Group>, payload: &[u8]) -> Result<Option<(Frame, f64)>> {
    let mut r = Reader(payload);
    let msg_id = r.u64()?;
    let total_u64 = r.u64()?;
    if total_u64 > group.max_chunked_inner {
        bail!(
            "chunked inner frame length {total_u64} exceeds negotiated limit {}",
            group.max_chunked_inner
        );
    }
    let total = usize::try_from(total_u64).context("chunk total does not fit usize")?;
    let offset = usize::try_from(r.u64()?).context("chunk offset does not fit usize")?;
    let data = r.rest();
    if data.is_empty() {
        bail!("empty chunk");
    }
    let end = offset
        .checked_add(data.len())
        .context("chunk offset overflow")?;
    if end > total {
        bail!("chunk overflow");
    }
    let mut reasm = group.reasm.lock().unwrap();
    if !reasm.contains_key(&msg_id) {
        if reasm.len() >= MAX_PARTIAL_MESSAGES {
            bail!("too many partial chunked messages");
        }
        let mut buf = Vec::new();
        buf.try_reserve_exact(total)
            .map_err(|error| anyhow::anyhow!("cannot allocate chunked frame: {error}"))?;
        buf.resize(total, 0);
        reasm.insert(
            msg_id,
            PartialMsg {
                started: Instant::now(),
                buf,
                filled: 0,
                ranges: Vec::new(),
            },
        );
    }
    let entry = reasm
        .get_mut(&msg_id)
        .context("chunk reassembly state disappeared")?;
    if entry.buf.len() != total {
        bail!("chunk total changed within one message");
    }
    if entry
        .ranges
        .iter()
        .any(|(old_start, old_end)| offset < *old_end && *old_start < end)
    {
        bail!("overlapping chunk");
    }
    entry.buf[offset..end].copy_from_slice(data);
    entry.filled += data.len();
    entry.ranges.push((offset, end));
    if entry.filled < total {
        return Ok(None);
    }
    if entry.filled != total {
        bail!("chunk byte count exceeds frame length");
    }
    let msg = reasm.remove(&msg_id).unwrap();
    drop(reasm);
    let recv_seconds = msg.started.elapsed().as_secs_f64();
    // The reassembled buffer is a complete inner frame.
    let buf = msg.buf;
    if buf.len() < 13 {
        bail!("inner frame too short");
    }
    let mut r = Reader(&buf);
    let magic = r.u32()?;
    if magic != MAGIC {
        bail!("bad inner magic");
    }
    let msg_type = r.u8()?;
    let len = r.u64()? as usize;
    let payload = r.rest();
    if payload.len() != len {
        bail!("inner frame length mismatch");
    }
    Ok(Some((
        Frame {
            msg_type,
            payload: payload.to_vec(),
        },
        recv_seconds,
    )))
}

async fn dispatch_inner(
    group: &Arc<Group>,
    msg_type: u8,
    payload: &[u8],
    recv_seconds: f64,
    event_tx: &mpsc::Sender<Event>,
    budget_cutoff: &BudgetCutoff,
) -> Result<()> {
    match msg_type {
        MSG_INIT_PARAMS => {
            if group.member.learner_id != 0 {
                bail!("only learner 0 may send INIT_PARAMS");
            }
            let mut r = Reader(payload);
            let fragment_id = r.u32()?;
            let fragment = group
                .layout
                .fragments
                .get(fragment_id as usize)
                .with_context(|| format!("INIT_PARAMS for unknown fragment {fragment_id}"))?;
            let numel = fragment.numel()?;
            let bytes = r.rest();
            let expected = tensor_nbytes(bulk_dtype(group.dtype), numel)?;
            if bytes.len() != expected {
                bail!(
                    "INIT_PARAMS fragment {fragment_id} has {} tensor bytes, expected {expected}",
                    bytes.len()
                );
            }
            let sha256: [u8; 32] = Sha256::digest(bytes).into();
            let mut values = Vec::new();
            decode_tensor(bulk_dtype(group.dtype), bytes, &mut values)?;
            event_tx
                .send(Event::Init {
                    member: group.member,
                    fragment_id,
                    values,
                    sha256,
                })
                .await
                .ok();
        }
        MSG_INIT_DIGEST => {
            if bulk_dtype(group.dtype) != DTYPE_F32 {
                bail!("INIT_DIGEST needs an f32-bulk session (dtype {})", group.dtype);
            }
            let digests = InitDigests::decode(payload, group.member.learner_id)?;
            if digests.first as usize + digests.digests.len() > group.layout.fragments.len() {
                bail!(
                    "INIT_DIGEST covers fragments {}..{}, layout has {}",
                    digests.first,
                    digests.first as usize + digests.digests.len(),
                    group.layout.fragments.len()
                );
            }
            event_tx
                .send(Event::InitDigest {
                    member: group.member,
                    digests,
                })
                .await
                .ok();
        }
        MSG_PUSH_FRAGMENT => {
            let mut r = Reader(payload);
            let learner_id = r.u32()?;
            if learner_id != group.member.learner_id {
                bail!(
                    "PUSH_FRAGMENT learner id {learner_id} does not match connected group {}",
                    group.member.learner_id
                );
            }
            let fragment_id = r.u32()?;
            let global_step = r.u64()?;
            let round_attempt = r.u32()?;
            if round_attempt == 0 {
                bail!("PUSH_FRAGMENT round_attempt must be positive");
            }
            let base_version = r.u64()?;
            let local_step = r.u64()?;
            let c_steps = r.u32()?;
            let c_tokens = r.u64()?;
            if c_steps == 0 {
                bail!("PUSH_FRAGMENT c_steps must be positive");
            }
            let fragment = group
                .layout
                .fragments
                .get(fragment_id as usize)
                .with_context(|| format!("PUSH_FRAGMENT for unknown fragment {fragment_id}"))?;
            let numel = fragment.numel()?;
            let tensor_scales = if group.expert_tokens {
                Some(
                    decode_expert_tags(&mut r, fragment.tensor_numels.len())
                        .with_context(|| format!("PUSH_FRAGMENT fragment {fragment_id}"))?,
                )
            } else {
                None
            };
            let bytes = r.rest();
            let mut outer_gradient = Vec::new();
            let segments = push_segments(group.dtype, numel, &fragment.tensor_numels)?;
            decode_push_segments(group.dtype, bytes, &segments, &mut outer_gradient)
                .with_context(|| format!("PUSH_FRAGMENT fragment {fragment_id}"))?;
            // Every wire dtype carries learner_delta = local - raw_anchor.
            // Merge math consumes the corresponding outer gradient with the
            // opposite sign and never reconstructs a full learner parameter.
            for value in &mut outer_gradient {
                *value = -*value;
            }
            event_tx
                .send(Event::Push {
                    member: group.member,
                    push: Push {
                        learner_id,
                        fragment_id,
                        global_step,
                        round_attempt,
                        base_version,
                        local_step,
                        c_steps,
                        c_tokens,
                        outer_gradient,
                        tensor_scales,
                        transfer: PushTransfer {
                            recv_seconds,
                            arrived_unix: unix_now(),
                        },
                    },
                })
                .await
                .ok();
        }
        MSG_HEARTBEAT => {
            let mut r = Reader(payload);
            let learner_id = r.u32()?;
            let local_step = r.u64()?;
            if learner_id != group.member.learner_id {
                bail!(
                    "HEARTBEAT learner id {learner_id} does not match connected group {}",
                    group.member.learner_id
                );
            }
            if r.remaining() != 0 {
                bail!("trailing bytes in HEARTBEAT");
            }
            event_tx
                .send(Event::Heartbeat {
                    member: group.member,
                    local_step,
                })
                .await
                .ok();
        }
        MSG_LEAVE => {
            let mut r = Reader(payload);
            let learner_id = r.u32()?;
            let reason = r.u8()?;
            if learner_id != group.member.learner_id {
                bail!(
                    "LEAVE learner id {learner_id} does not match connected group {}",
                    group.member.learner_id
                );
            }
            if r.remaining() != 0 {
                bail!("trailing bytes in LEAVE");
            }
            event_tx
                .send(Event::Leave {
                    member: group.member,
                    reason,
                })
                .await
                .ok();
        }
        MSG_FINAL_ACK => {
            let global_step = decode_final_ack(payload)?;
            event_tx
                .send(Event::FinalAck {
                    member: group.member,
                    global_step,
                })
                .await
                .ok();
        }
        MSG_BUDGET_DONE => {
            let local_steps = decode_budget_done(payload)?;
            // This is the cutoff linearization point.  It must precede event
            // queueing so a report waiting behind ordinary traffic still
            // closes the scheduler gate immediately.
            budget_cutoff.request();
            // The first report closes the scheduler gate, but it must not
            // make a later learner's otherwise-valid report unsendable. Each
            // queue operation gets its own bounded timeout; report collection
            // separately enforces a progress lease per logical learner.
            tokio::time::timeout(
                budget_cutoff.timeout(),
                event_tx.send(Event::BudgetDone {
                    member: group.member,
                    local_steps,
                }),
            )
            .await
            .context("timed out queueing BUDGET_DONE")?
            .context("event channel closed while queueing BUDGET_DONE")?;
        }
        recovery::MSG_PEER_SNAPSHOT_READY | recovery::MSG_RECOVERY_RESTORED | recovery::MSG_RECOVERY_FAILED => {
            let fragments = group.layout.fragments.len();
            let msg = match msg_type {
                recovery::MSG_PEER_SNAPSHOT_READY => {
                    RecoveryMsg::SnapshotReady(recovery::SnapshotReady::decode(payload, fragments)?)
                }
                recovery::MSG_RECOVERY_RESTORED => {
                    RecoveryMsg::Restored(recovery::decode_restored(payload, fragments)?)
                }
                _ => RecoveryMsg::Failed(recovery::decode_failed(payload)?),
            };
            event_tx.send(Event::Recovery { member: group.member, msg }).await.ok();
        }
        t => bail!(
            "unexpected message type {t} from learner {} generation {}",
            group.member.learner_id,
            group.member.generation
        ),
    }
    Ok(())
}

// --- scheduler -------------------------------------------------------------

fn encode_pull(fragment_id: usize, global_step: u64, round_attempt: u32) -> bytes::Bytes {
    let mut payload = Vec::with_capacity(16);
    payload.extend_from_slice(&(fragment_id as u32).to_le_bytes());
    payload.extend_from_slice(&global_step.to_le_bytes());
    payload.extend_from_slice(&round_attempt.to_le_bytes());
    bytes::Bytes::from(payload)
}

async fn send_small_until_cutoff(
    group: &Arc<Group>,
    msg_type: u8,
    payload: bytes::Bytes,
    budget_cutoff: &BudgetCutoff,
) -> bool {
    if !budget_cutoff.enabled {
        let _ = group.send_small(msg_type, payload).await;
        return true;
    }
    tokio::select! {
        biased;
        _ = budget_cutoff.wait_requested() => false,
        _ = group.send_small(msg_type, payload) => true,
    }
}

async fn send_tensor_until_cutoff(
    group: &Arc<Group>,
    msg_type: u8,
    prefix: &[u8],
    dtype: u8,
    values: &[f32],
    budget_cutoff: &BudgetCutoff,
) -> Result<bool> {
    if !budget_cutoff.enabled {
        group
            .send_tensor_large(msg_type, prefix, dtype, values)
            .await?;
        return Ok(true);
    }
    tokio::select! {
        biased;
        _ = budget_cutoff.wait_requested() => Ok(false),
        result = group.send_tensor_large(msg_type, prefix, dtype, values) => result.map(|()| true),
    }
}

async fn send_fragment_until_cutoff(
    st: &GlobalState,
    group: &Arc<Group>,
    p: usize,
    msg_type: u8,
    dtype: u8,
    budget_cutoff: &BudgetCutoff,
) -> Result<bool> {
    if let Some(encoded) = recovery_bcast(st, p, msg_type) {
        return send_tensor_until_cutoff(
            group,
            msg_type,
            &encoded.prefix,
            encoded.dtype,
            &encoded.values,
            budget_cutoff,
        )
        .await;
    }
    let mut prefix = [0u8; 12];
    prefix[..4].copy_from_slice(&(p as u32).to_le_bytes());
    prefix[4..].copy_from_slice(&st.versions[p].to_le_bytes());
    send_tensor_until_cutoff(
        group,
        msg_type,
        &prefix,
        dtype,
        &st.params[p],
        budget_cutoff,
    )
    .await
}

/// A BCAST outside publication (initial cut, HELLO/reconnect recovery) on a
/// bf16-downlink session: always FULL, carrying the shared anchor.
fn recovery_bcast(st: &GlobalState, p: usize, msg_type: u8) -> Option<crate::downlink::EncodedBcast> {
    if msg_type != MSG_BCAST_FRAGMENT {
        return None;
    }
    st.downlink
        .as_ref()
        .map(|anchors| anchors.lock().unwrap().resend(p, st.versions[p], &st.params[p]))
}

async fn send_pull_until_cutoff(
    groups: &[Arc<Group>],
    pull: &bytes::Bytes,
    budget_cutoff: &BudgetCutoff,
) -> bool {
    for group in groups {
        if !send_small_until_cutoff(group, MSG_PULL_REQ, pull.clone(), budget_cutoff).await {
            return false;
        }
    }
    true
}

async fn recv_init_event(
    events: &mut mpsc::Receiver<Event>,
    budget_cutoff: &BudgetCutoff,
    reports_received: usize,
    learners: u32,
) -> Result<Event> {
    let event = match budget_cutoff.deadline() {
        Some(deadline) => tokio::time::timeout_at(deadline, events.recv())
            .await
            .map_err(|_| {
                anyhow::anyhow!(
                    "learner-budget cutoff timed out during initialization/report collection: \
                     received {reports_received}/{learners} BUDGET_DONE reports"
                )
            })?,
        None => events.recv().await,
    };
    event.context("event channel closed")
}

/// The islands did not start from the same weights: record the strict
/// failure, tell every connected island why, and stop the syncer.
async fn refuse_local_init(cfg: &Config, registry: &Registry, error: anyhow::Error) -> Result<()> {
    let message = format!("{error:#}");
    append_strict_failure(cfg.event_tape.as_deref(), "init_digest_mismatch", &message);
    for group in current_groups(registry) {
        let _ = group.send_small(MSG_ERROR, message.clone().into()).await;
    }
    bail!("RL strict failure init_digest_mismatch: {message}");
}

/// Connected learners the scheduler waits for before the first round: every
/// learner with static membership, the launch quorum with dynamic membership.
fn startup_members(cfg: &Config) -> u32 {
    if cfg.membership.is_static() {
        cfg.learners
    } else {
        (cfg.membership.min_launch_members(cfg.quorum) as u32).clamp(1, cfg.learners)
    }
}

async fn scheduler(
    cfg: Config,
    mut events: mpsc::Receiver<Event>,
    registry: Registry,
    budget_cutoff: Arc<BudgetCutoff>,
) -> Result<()> {
    let mut state: Option<GlobalState> = None;
    let mut budget_reports: HashSet<u32> = HashSet::new();
    let strict_exact = cfg.max_base_lag == Some(0);
    let fixed_roster = strict_exact && cfg.quorum == cfg.learners && cfg.grace_ms == 0;
    // Dynamic membership writes island-pool records next to the merge records.
    let dynamic = !cfg.membership.is_static();
    let mut lease_expired: HashSet<Member> = HashSet::new();
    // LEAVE reasons of islands whose disconnect has not arrived yet.
    let mut leave_reasons: HashMap<Member, u8> = HashMap::new();
    let checkpoint_each_round = checkpoint_before_broadcast(&cfg);
    // Local-init digests (`local_init`); created with the state.
    let mut local_init: Option<LocalInit> = None;
    let mut resumed = false;

    // Phase 1: wait until every fragment is initialized (via INIT_PARAMS or
    // a resumed checkpoint) and the expected learners have connected (late
    // joiners are still served afterwards). Static membership waits for all
    // learners; dynamic membership starts once a round could launch.
    let start_members = startup_members(&cfg);
    info!(
        expected = cfg.learners,
        start_members, "waiting for learners and INIT_PARAMS"
    );
    loop {
        let connected = registry.lock().unwrap().current.len() as u32;
        if let Some(st) = &state {
            let budget_ready =
                budget_reports.is_empty() || budget_reports.len() == cfg.learners as usize;
            if st.all_initialized() && connected >= start_members && budget_ready {
                break;
            }
        }
        match recv_init_event(
            &mut events,
            &budget_cutoff,
            budget_reports.len(),
            cfg.learners,
        )
        .await?
        {
            Event::Fatal { metric, message } => {
                append_strict_failure(cfg.event_tape.as_deref(), metric, &message);
                bail!("RL strict failure {metric}: {message}");
            }
            Event::Hello { group } => {
                admit_recovering_outside_training(&registry, &group, "initialization").await;
                if dynamic {
                    append_pool_record(
                        cfg.event_tape.as_deref(),
                        "pool_join",
                        cfg.syncer_epoch,
                        &format!(
                            "\"island_id\":{},\"generation\":{},\"catch_up\":false,\"base_version\":0",
                            group.member.learner_id, group.member.generation
                        ),
                    );
                }
                if state.is_none() {
                    // Layout comes from the HELLO of the first learner.
                    // (All learners must build identical layouts.)
                    let mut st = new_state_for(&group, &cfg)?;
                    if let Some(service) = registry.lock().unwrap().dispense.clone() {
                        st.checkpoint_sidecars.push(service);
                    }
                    if cfg.resume {
                        if let Some(path) = cfg.checkpoint_path.as_ref().filter(|p| p.exists()) {
                            if let Err(error) = st.load_checkpoint(path) {
                                if strict_exact {
                                    let message = format!("cannot resume RL checkpoint: {error:#}");
                                    append_strict_failure(
                                        cfg.event_tape.as_deref(),
                                        "layout_hash_mismatch",
                                        &message,
                                    );
                                    bail!("RL strict failure layout_hash_mismatch: {message}");
                                }
                                return Err(error);
                            }
                            if strict_exact && !st.checkpoint_layout_verified {
                                let message = "RL checkpoint is missing its canonical layout hash";
                                append_strict_failure(
                                    cfg.event_tape.as_deref(),
                                    "layout_hash_mismatch",
                                    message,
                                );
                                bail!("RL strict failure layout_hash_mismatch: {message}");
                            }
                            if let Err(error) = validate_resumed_policy_sweep(&cfg, &st) {
                                let message =
                                    format!("cannot resume policy-sweep checkpoint: {error:#}");
                                append_strict_failure(
                                    cfg.event_tape.as_deref(),
                                    "policy_sweep_checkpoint_mismatch",
                                    &message,
                                );
                                bail!(
                                    "RL strict failure policy_sweep_checkpoint_mismatch: {message}"
                                );
                            }
                            if cfg.policy_sweep_fragments.is_some() {
                                if let Some(tape) = cfg.event_tape.as_deref() {
                                    append_policy_sweep_ledger_snapshot(tape, &st, "resume")?;
                                }
                            }
                            info!(step = st.global_step, "resumed from checkpoint");
                            resumed = true;
                        }
                    }
                    local_init = Some(LocalInit::new(
                        st.owned_fragments(),
                        st.layout.fragments.len(),
                    ));
                    state = Some(st);
                }
            }
            Event::InitDigest { member, digests } => {
                if !is_current_member(&registry, member) {
                    continue;
                }
                let checked = local_init
                    .as_mut()
                    .context("INIT_DIGEST before HELLO")?
                    .on_digests(digests);
                if let Err(error) = checked {
                    refuse_local_init(&cfg, &registry, error).await?;
                }
            }
            Event::Init {
                member,
                fragment_id,
                values,
                sha256,
            } => {
                if !is_current_member(&registry, member) {
                    warn!(
                        learner_id = member.learner_id,
                        generation = member.generation,
                        "INIT_PARAMS from superseded connection dropped"
                    );
                    continue;
                }
                let st = state.as_mut().context("INIT before HELLO")?;
                if !st.initialized[fragment_id as usize] {
                    let checked = local_init.as_mut().unwrap().on_init(
                        fragment_id as usize,
                        member.learner_id,
                        sha256,
                    );
                    if let Err(error) = checked {
                        refuse_local_init(&cfg, &registry, error).await?;
                    }
                }
                st.init_fragment(fragment_id as usize, values)?;
                if st.all_initialized() {
                    info!("global parameters initialized");
                }
            }
            Event::Push { .. } => warn!("push before initialization; dropped"),
            // The lease runs from HELLO: an island that waits for the other
            // islands' INIT (or warms up its engine) keeps it by heartbeat.
            Event::Heartbeat { member, .. } => {
                registry.lock().unwrap().membership.renew(member, Instant::now());
            }
            Event::Leave { .. } => {}
            Event::FinalAck { member, .. } => warn!(
                learner_id = member.learner_id,
                generation = member.generation,
                "premature final acknowledgement dropped"
            ),
            Event::BudgetDone {
                member,
                local_steps,
            } => {
                let target = cfg
                    .learner_budget_steps
                    .context("received BUDGET_DONE outside learner-budget mode")?;
                if !is_current_member(&registry, member) {
                    bail!("BUDGET_DONE from a superseded learner");
                }
                record_budget_report(&mut budget_reports, target, member, local_steps)?;
            }
            Event::Disconnected { member } => warn!(
                learner_id = member.learner_id,
                generation = member.generation,
                "disconnected during init"
            ),
            Event::Recovery { member, .. } => ignore_recovery_message(member, "initialization"),
        }
    }
    let mut st = state.unwrap();
    if budget_cutoff.deadline().is_some() || !budget_reports.is_empty() {
        let deadline = budget_cutoff
            .deadline()
            .context("learner-budget gate closed without a deadline")?;
        let target = cfg
            .learner_budget_steps
            .context("learner-budget gate closed outside learner-budget mode")?;
        let (drain_result, reports_result) = tokio::join!(
            drain_iso_before(&st, deadline),
            collect_budget_reports(
                target,
                cfg.learners,
                &mut budget_reports,
                &mut events,
                &registry,
                budget_cutoff.timeout(),
                cfg.event_tape.as_deref(),
                None,
            )
        );
        // Always finish the bounded backend drain even when a report is
        // malformed; a fatal report must not abandon accepted worker state.
        drain_result?;
        reports_result?;
        save_budget_checkpoint(&cfg, &st)?;
        return Ok(());
    }
    // A sharded head schedules only its own fragments and global steps
    // (crate::shard); unsharded, `owned` is every fragment and the steps
    // are 1..=total_steps.
    let schedule = st.schedule().clone();
    let num_fragments = schedule.owned().len() as u64;
    let last_step = schedule.last_step(cfg.total_steps);
    let mut step_rates = StepRates::default();
    let mut arrival_gaps = ArrivalGaps::default();
    let mut last_sync_secs = 0.0f64; // previous round's merge+broadcast time

    // Once a marked run is actually going to make more progress, the old
    // marker must stop being publishable before any new round can commit.
    if cfg.mark_final_checkpoint && st.global_step < last_step {
        remove_final_marker(cfg.checkpoint_path.as_ref().unwrap())?;
    }

    // Persist a fresh version zero before its first BCAST. A resumed cut is
    // already committed and only needs to be rebroadcast.
    if checkpoint_each_round || cfg.policy_sweep_fragments.is_some() {
        if let Some(path) = cfg.checkpoint_path.as_ref() {
            if !cfg.resume || !path.exists() {
                st.save_checkpoint(path)?;
                info!(step = st.global_step, path = %path.display(), "checkpoint committed");
            }
        }
    }
    // Send everyone the initial (or resumed) global parameters so all
    // learners start bit-identical (also serves recovery for late joiners).
    // A checkpoint already at the terminal cut goes straight to lossless
    // FINAL delivery; a redundant full BCAST would only inflate receiver RSS.
    // Local init: islands whose digests agree with the applied INIT already
    // hold version 0 and only get INIT_CONFIRM. A resumed cut is not the
    // base checkpoint, so it is always broadcast.
    if st.global_step < last_step {
        let confirm = local_init
            .as_ref()
            .filter(|_| !resumed)
            .and_then(|init| init.confirm().map(|digest| (init, digest)));
        let groups = current_groups(&registry);
        let (confirmed, rest): (Vec<_>, Vec<_>) = groups.into_iter().partition(|group| {
            confirm.is_some_and(|(init, _)| init.holds(group.member.learner_id))
        });
        if let Some((_, digest)) = confirm {
            for group in &confirmed {
                let payload = bytes::Bytes::copy_from_slice(&digest);
                if !send_small_until_cutoff(group, MSG_INIT_CONFIRM, payload, &budget_cutoff).await {
                    break;
                }
            }
            info!(
                confirmed = confirmed.len(),
                broadcast = rest.len(),
                "local init: islands kept their own version-0 weights"
            );
        }
        broadcast_all_fragments(&st, &rest, &budget_cutoff).await;
    }

    // Phase 2: gather, compute, and commit are separate stages.  Torch SVD
    // matrices execute concurrently and may finish out of order, but only
    // this scheduler mutates coordinator state, strictly in fragment-step t
    // order.  A fragment remains busy across all three stages, preventing a
    // second round from observing an uncommitted version/momentum state. The
    // launch pipeline still overlaps learner compute and communication.
    let depth = (cfg.pipeline.max(1) as u64).min(num_fragments) as usize;
    let manual_floor = Duration::from_millis(cfg.min_round_interval_ms);
    let mut next_launch = Instant::now(); // earliest allowed next round launch
    let mut next_t = schedule.step_after(st.global_step);
    let mut next_commit_t = schedule.step_after(st.global_step);
    let mut inflight: Vec<Round> = Vec::new();
    let mut computing = tokio::task::JoinSet::<Result<ComputedRound>>::new();
    let mut ready: BTreeMap<u64, ComputedRound> = BTreeMap::new();
    let mut busy_fragments: HashSet<usize> = HashSet::new();
    let mut late_carry = LateCarry::new(cfg.max_base_lag, fixed_roster);
    // --async-checkpoint: the committed round whose checkpoint is in flight.
    let mut publishing: Option<PendingPublish> = None;
    // --checkpoint-at-steps: evaluation copies written off the scheduler.
    let mut eval_checkpoints = crate::eval_checkpoint::EvalCheckpoints::new(
        &cfg.checkpoint_at_steps,
        cfg.checkpoint_path.as_deref(),
        st.global_step,
    )?;
    let mut cutoff_established = false;
    let mut first_budget_report: Option<(Member, u64)> = None;
    let mut recovery = PeerRecovery::new(cfg.recovery.clone());
    let mut catch_ups: CatchUps<Arc<Group>> = CatchUps::new();
    if dynamic {
        write_pool_status(&cfg, &registry, st.global_step, "running");
    }
    'outer: while next_t <= last_step
        || !inflight.is_empty()
        || !computing.is_empty()
        || !ready.is_empty()
        || publishing.is_some()
    {
        if budget_cutoff.deadline().is_some() {
            cutoff_established = true;
            break 'outer;
        }
        let expired = expire_member_leases(&registry, &mut inflight, &mut step_rates).await;
        for member in &expired {
            lease_expired.insert(*member);
            append_pool_record(
                cfg.event_tape.as_deref(),
                "pool_leave",
                cfg.syncer_epoch,
                &format!(
                    "\"island_id\":{},\"generation\":{},\"reason\":\"lease_expired\"",
                    member.learner_id, member.generation
                ),
            );
        }
        if !recovery.is_empty() {
            let now = Instant::now();
            let candidates = recovery_candidates(&registry, &inflight);
            let mut out = Vec::new();
            for member in expired {
                out.extend(recovery.on_gone(member, &candidates, now));
            }
            out.extend(recovery.on_tick(&candidates, now));
            send_recovery(&registry, out).await;
        }
        // Commit only the contiguous prefix.  A faster cuda:N worker can put
        // t+1 in `ready` first, but it cannot update Nesterov/version/tape or
        // broadcast until t has committed.
        while ready.contains_key(&next_commit_t) {
            if !budget_cutoff.try_linearize_work() {
                cutoff_established = true;
                break 'outer;
            }
            let completed = take_next_commit(&mut ready, next_commit_t)
                .context("ready commit disappeared after linearization")?;
            let p = completed.round.p;
            if cfg.async_checkpoint {
                // At most one checkpoint write is outstanding: the snapshot
                // of t must be durable before t+1 is snapshotted, which also
                // bounds host memory to one extra copy of the state.
                if let Some(pending) = publishing.take() {
                    settle_pending_publish(
                        &cfg,
                        &st,
                        &registry,
                        &mut last_sync_secs,
                        pending,
                        &budget_cutoff,
                        &mut busy_fragments,
                    )
                    .await?;
                }
                let committed = apply_round_commit(&cfg, &mut st, completed)?;
                if persist_before_broadcast(&cfg, committed.t) {
                    publishing = Some(start_async_checkpoint(&cfg, &st, committed)?);
                } else {
                    publish_round(
                        &cfg,
                        &st,
                        &registry,
                        &mut last_sync_secs,
                        committed,
                        None,
                        &budget_cutoff,
                    )
                    .await?;
                    if !busy_fragments.remove(&p) {
                        bail!("fragment {p} lost its busy ownership at commit");
                    }
                }
            } else {
                commit_round(
                    &cfg,
                    &mut st,
                    &registry,
                    &mut last_sync_secs,
                    completed,
                    &budget_cutoff,
                )
                .await?;
                if !busy_fragments.remove(&p) {
                    bail!("fragment {p} lost its busy ownership at commit");
                }
            }
            // The cut at the step just committed; later rounds keep merging
            // while an evaluation copy of it is written.
            eval_checkpoints.after_commit(&st).await?;
            next_commit_t = schedule.step_after(next_commit_t);
            if dynamic {
                write_pool_status(&cfg, &registry, st.global_step, "running");
            }
        }

        // Keep the pipeline full (throttled by min_round_interval_ms).
        while inflight.len() + computing.len() + ready.len() < depth
            && next_t <= last_step
            && Instant::now() >= next_launch
        {
            if !budget_cutoff.try_linearize_work() {
                cutoff_established = true;
                break 'outer;
            }
            let p = schedule.fragment_for_step(next_t);
            if busy_fragments.contains(&p) {
                break;
            }
            let mut groups = current_groups(&registry);
            groups.retain(|group| !catch_ups.holds(group.member));
            if groups.is_empty() || (fixed_roster && groups.len() < cfg.quorum as usize) {
                next_launch = Instant::now() + Duration::from_millis(100);
                break;
            }
            let t = next_t;
            next_t = schedule.step_after(t);
            let pull = encode_pull(p, t, 1);
            let mut expected_members: Vec<Member> =
                groups.iter().map(|group| group.member).collect();
            expected_members.sort_unstable();
            next_launch = Instant::now()
                + launch_interval(
                    manual_floor,
                    cfg.sync_interval_steps,
                    num_fragments as usize,
                    step_rates.max_step_secs_for(&expected_members),
                );
            let launch_quorum = cfg.quorum as usize;
            if !send_pull_until_cutoff(&groups, &pull, &budget_cutoff).await {
                cutoff_established = true;
                break 'outer;
            }
            if !busy_fragments.insert(p) {
                bail!("fragment {p} acquired twice");
            }
            let fold = round_fold(&cfg, &st, p, &expected_members, launch_quorum);
            inflight.push(Round {
                t,
                p,
                base_version: st.versions[p],
                attempt: 1,
                pull,
                started: Instant::now(),
                expected_members,
                quorum_deadline: Instant::now() + Duration::from_secs(cfg.quorum_timeout_s),
                grace_deadline: None,
                quorum_size: launch_quorum,
                capacity_quorum: cfg.membership.round_quorum(),
                soft_deadline: cfg.membership.soft_deadline.map(|soft| Instant::now() + soft),
                soft_min: cfg.membership.soft_min_arrivals(),
                quorum_ms: None,
                grace_ms: None,
                fold,
                pushes: HashMap::new(),
                carried: Vec::new(),
            });
            if let Some(carry) = late_carry.as_mut() {
                let round = inflight.last_mut().expect("round just launched");
                for (member, missed, outcome) in carry.attach_queued(round) {
                    log_carry(&cfg, dynamic, member, missed, t, outcome);
                }
            }
        }

        // Arm the grace window of any round that just reached quorum.
        for r in inflight.iter_mut() {
            if r.grace_deadline.is_none() && r.quorum_reached() {
                let step_secs = step_rates.max_step_secs_for(&r.expected_members);
                let quorum_secs = r.started.elapsed().as_secs_f64();
                let terminal = is_terminal_round(&schedule, r.t, cfg.total_steps);
                let (grace, floor) =
                    round_grace(&cfg, terminal, step_secs, quorum_secs, last_sync_secs, &arrival_gaps);
                if r.pushes.len() < r.expected_members.len() {
                    info!(
                        step = r.t,
                        fragment = r.p,
                        step_secs = ?step_secs,
                        quorum_secs,
                        sync_secs = last_sync_secs,
                        grace_ms = grace.as_millis() as u64,
                        grace_floor_ms = floor.as_millis() as u64,
                        terminal,
                        "grace window armed"
                    );
                }
                // The soft deadline is an upper bound on the grace window too.
                let now = Instant::now();
                let grace = r
                    .soft_deadline
                    .map_or(grace, |soft| grace.min(soft.saturating_duration_since(now)));
                r.grace_deadline = Some(now + grace);
                r.quorum_ms = Some(r.started.elapsed().as_millis() as u64);
                r.grace_ms = Some(grace.as_millis() as u64);
            }
        }

        // A ready gather is reduced deterministically and its complete ISO
        // matrices are submitted to the bounded worker pool.  The scheduler
        // immediately continues serving learner traffic while they run.
        let now = Instant::now();
        let mut submitted_any = false;
        let mut i = 0;
        while i < inflight.len() {
            match round_action(&inflight[i], now) {
                RoundAction::Complete => {
                    let mut round = inflight.remove(i);
                    let prepared = prepare_round_compute(&cfg, &st, &mut round)?;
                    if !budget_cutoff.try_linearize_work() {
                        cutoff_established = true;
                        break 'outer;
                    }
                    computing.spawn(async move {
                        let compute_start = Instant::now();
                        let merge = prepared.compute().await?;
                        Ok(ComputedRound {
                            round,
                            merge,
                            compute_secs: compute_start.elapsed().as_secs_f64(),
                        })
                    });
                    submitted_any = true;
                    continue;
                }
                RoundAction::Restart => {
                    if !budget_cutoff.try_linearize_work() {
                        cutoff_established = true;
                        break 'outer;
                    }
                    let r = &mut inflight[i];
                    let groups = current_groups(&registry);
                    if fixed_roster {
                        if groups.len() < cfg.quorum as usize {
                            r.quorum_deadline =
                                Instant::now() + Duration::from_secs(cfg.quorum_timeout_s);
                            i += 1;
                            continue;
                        }
                        warn!(
                            step = r.t,
                            responses = r.pushes.len(),
                            roster = cfg.learners,
                            "fixed roster incomplete; waiting for missing logical learners"
                        );
                        for g in groups {
                            if !r
                                .pushes
                                .keys()
                                .any(|member| member.learner_id == g.member.learner_id)
                            {
                                let _ = g.send_small(MSG_PULL_REQ, r.pull.clone()).await;
                            }
                        }
                        r.quorum_deadline =
                            Instant::now() + Duration::from_secs(cfg.quorum_timeout_s);
                        i += 1;
                        continue;
                    }
                    warn!(
                        step = r.t,
                        attempt = r.attempt,
                        responses = r.pushes.len(),
                        quorum = r.quorum_size,
                        "quorum timeout below K; launching a new frozen-membership attempt"
                    );
                    let required = cfg.membership.min_launch_members(cfg.quorum);
                    if groups.len() < required {
                        warn!(
                            step = r.t,
                            connected = groups.len(),
                            required,
                            "fixed quorum unavailable; deferring retry"
                        );
                        r.quorum_deadline = Instant::now() + Duration::from_secs(1);
                        i += 1;
                        continue;
                    }
                    r.expected_members = groups.iter().map(|group| group.member).collect();
                    r.expected_members.sort_unstable();
                    r.quorum_size = cfg.quorum as usize;
                    r.base_version = st.versions[r.p];
                    r.attempt += 1;
                    r.pull = encode_pull(r.p, r.t, r.attempt);
                    if let Some(carry) = late_carry.as_mut() {
                        for (member, push) in &r.pushes {
                            carry.release_anchor(*member, push);
                        }
                    }
                    r.pushes.clear();
                    arrival_gaps.forget(r.t, r.p);
                    r.fold = round_fold(&cfg, &st, r.p, &r.expected_members, r.quorum_size);
                    r.started = Instant::now();
                    r.soft_deadline = cfg.membership.soft_deadline.map(|soft| r.started + soft);
                    r.grace_deadline = None;
                    r.quorum_ms = None;
                    r.grace_ms = None;
                    if !send_pull_until_cutoff(&groups, &r.pull, &budget_cutoff).await {
                        cutoff_established = true;
                        break 'outer;
                    }
                    r.quorum_deadline = Instant::now() + Duration::from_secs(cfg.quorum_timeout_s);
                }
                RoundAction::Wait => {}
            }
            i += 1;
        }
        // Advance one rejoin catch-up by one fragment (or run its tail).
        // The round deadlines above were served first, so a long catch-up
        // never holds a grace window open.
        for member in catch_ups.drop_superseded(|member| is_current_member(&registry, member)) {
            warn!(
                learner_id = member.learner_id,
                generation = member.generation,
                "superseded learner catch-up dropped"
            );
        }
        match catch_ups.next_step() {
            None => {}
            Some(CatchUpStep::Send(member, group, p)) => {
                // Catch-up must never expose a round whose checkpoint is not
                // yet durable.
                if let Some(pending) = publishing.take() {
                    settle_pending_publish(
                        &cfg,
                        &st,
                        &registry,
                        &mut last_sync_secs,
                        pending,
                        &budget_cutoff,
                        &mut busy_fragments,
                    )
                    .await?;
                }
                let dtype = bulk_dtype(st.wire_dtype);
                match send_fragment_until_cutoff(&st, &group, p, MSG_BCAST_FRAGMENT, dtype, &budget_cutoff)
                    .await
                {
                    Ok(true) => {}
                    Ok(false) => {
                        cutoff_established = true;
                        break 'outer;
                    }
                    Err(error) => {
                        warn!(
                            learner_id = member.learner_id,
                            generation = member.generation,
                            fragment = p,
                            %error,
                            "recovery fragment encode failed"
                        );
                        catch_ups.abandon(member);
                    }
                }
            }
            Some(CatchUpStep::Done(done)) => {
                let group = done.group;
                if done.tail.rejoin {
                    registry
                        .lock()
                        .unwrap()
                        .membership
                        .record_catch_up(group.member, &st.versions);
                    if dynamic {
                        append_pool_record(
                            cfg.event_tape.as_deref(),
                            "pool_join",
                            cfg.syncer_epoch,
                            &format!(
                                "\"island_id\":{},\"generation\":{},\"catch_up\":true,\"base_version\":{}",
                                group.member.learner_id, group.member.generation, st.global_step
                            ),
                        );
                        write_pool_status(&cfg, &registry, st.global_step, "running");
                    }
                    if !repull_rounds_for(&group, &registry, &mut inflight, fixed_roster, &budget_cutoff).await {
                        cutoff_established = true;
                        break 'outer;
                    }
                }
                if done.tail.rebind {
                    let repulls = rebind_current_unanswered_rounds(&registry, &mut inflight, group.member);
                    for repull in repulls {
                        if !send_small_until_cutoff(&group, MSG_PULL_REQ, repull.pull, &budget_cutoff).await {
                            cutoff_established = true;
                            break 'outer;
                        }
                        info!(
                            learner_id = group.member.learner_id,
                            old_generation = repull.old_generation,
                            generation = group.member.generation,
                            step = repull.t,
                            fragment = repull.p,
                            attempt = repull.attempt,
                            base_version = repull.base_version,
                            "rebound outstanding pull after captured generation disconnected"
                        );
                    }
                }
            }
        }
        if submitted_any {
            continue; // refill the pipeline before waiting again
        }
        // The commit above may have drained the last round. Re-check the loop
        // condition now: waiting here would sleep until the next lease expiry,
        // and the leases of idle learners then lapse before the final cut.
        if next_t > last_step
            && inflight.is_empty()
            && computing.is_empty()
            && ready.is_empty()
            && publishing.is_none()
        {
            continue;
        }
        // Wait for the next event, the earliest in-flight deadline, or the
        // launch throttle opening (whichever comes first). Without the
        // throttle term an empty pipeline would spin; without in-flight
        // deadlines a throttled launch would oversleep.
        let deadline_now = Instant::now();
        let mut earliest = inflight
            .iter()
            .filter_map(|r| {
                (round_action(r, deadline_now) != RoundAction::Complete).then_some(
                    r.soft_deadline.map_or(r.grace_deadline.unwrap_or(r.quorum_deadline), |soft| {
                        soft.min(r.grace_deadline.unwrap_or(r.quorum_deadline))
                    }),
                )
            })
            .chain(registry.lock().unwrap().membership.next_expiry())
            .chain(recovery.next_deadline())
            .min();
        let next_fragment_available = inflight.len() + computing.len() + ready.len() < depth
            && next_t <= last_step
            && !busy_fragments.contains(&schedule.fragment_for_step(next_t));
        if next_fragment_available {
            earliest = Some(earliest.map_or(next_launch, |d| d.min(next_launch)));
        }
        if !catch_ups.is_empty() {
            // Poll events without sleeping, then send the next fragment.
            earliest = Some(deadline_now);
        }
        let wake = if let Some(pending) = publishing.as_mut() {
            // Same event-first bias; a durable checkpoint outranks compute
            // so the oldest committed round is exposed first.
            let sleep_until = earliest.map(tokio::time::Instant::from_std);
            tokio::select! {
                biased;
                event = events.recv() => SchedulerWake::Event(event),
                written = &mut pending.write => SchedulerWake::Durable(written),
                result = computing.join_next(), if !computing.is_empty() => SchedulerWake::Computed(result),
                _ = sleep_until_opt(sleep_until) => SchedulerWake::Deadline,
            }
        } else {
            match earliest {
            Some(deadline) if !computing.is_empty() => tokio::select! {
                biased;
                wake = prefer_event_over_compute(events.recv(), computing.join_next()) => wake,
                _ = tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)) => SchedulerWake::Deadline,
            },
            Some(deadline) => tokio::select! {
                biased;
                event = events.recv() => SchedulerWake::Event(event),
                _ = tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)) => SchedulerWake::Deadline,
            },
            None if !computing.is_empty() => {
                prefer_event_over_compute(events.recv(), computing.join_next()).await
            }
            None => continue,
            }
        };
        match wake {
            SchedulerWake::Deadline => continue,
            SchedulerWake::Durable(written) => {
                let pending = publishing
                    .take()
                    .context("durable checkpoint without a pending publish")?;
                finish_pending_publish(
                    &cfg,
                    &st,
                    &registry,
                    &mut last_sync_secs,
                    pending.committed,
                    pending.write_started,
                    written,
                    &budget_cutoff,
                    &mut busy_fragments,
                )
                .await?;
            }
            SchedulerWake::Computed(Some(result)) => {
                let completed = result.context("iso compute task panicked")??;
                let t = completed.round.t;
                if completed.merge.fid() != completed.round.p
                    || completed.merge.base_version() != completed.round.base_version
                {
                    bail!("computed merge metadata does not match round t={t}");
                }
                if st.versions.get(completed.round.p).copied() != Some(completed.round.base_version)
                {
                    bail!(
                        "computed result t={t} fragment {} base version {} no longer matches state",
                        completed.round.p,
                        completed.round.base_version
                    );
                }
                if ready.insert(t, completed).is_some() {
                    bail!("duplicate computed round t={t}");
                }
            }
            SchedulerWake::Computed(None) => bail!("iso compute task set ended unexpectedly"),
            SchedulerWake::Event(None) => bail!("event channel closed"),
            SchedulerWake::Event(Some(ev)) => match ev {
                Event::Fatal { metric, message } => {
                    append_strict_failure(cfg.event_tape.as_deref(), metric, &message);
                    bail!("RL strict failure {metric}: {message}");
                }
                Event::Push { member, push } => {
                    if let Some(fragments) = cfg.policy_sweep_fragments {
                        if let Err(error) = validate_policy_sweep_push(&push, fragments) {
                            let message = format!(
                                "learner {} generation {}: {error:#}",
                                member.learner_id, member.generation
                            );
                            append_strict_failure(
                                cfg.event_tape.as_deref(),
                                "policy_sweep_push_mismatch",
                                &message,
                            );
                            bail!("RL strict failure policy_sweep_push_mismatch: {message}");
                        }
                    }
                    let learner_id = member.learner_id;
                    let generation = member.generation;
                    let local_step = push.local_step;
                    let global_step = push.global_step;
                    let fragment_id = push.fragment_id;
                    let fenced = {
                        let mut registry = registry.lock().unwrap();
                        registry.membership.renew(member, Instant::now());
                        registry.membership.check_push_base(
                            member,
                            fragment_id as usize,
                            push.base_version,
                        )
                    };
                    let pool = |kind: &str, fields: String| {
                        if dynamic {
                            append_pool_record(cfg.event_tape.as_deref(), kind, cfg.syncer_epoch, &fields);
                        }
                    };
                    let rejected = |reason: &str| {
                        pool("delta_rejected", format!("\"island_id\":{learner_id},\"reason\":\"{reason}\""));
                    };
                    if let Err(reason) = fenced {
                        warn!(
                            learner_id,
                            generation,
                            step = global_step,
                            fragment = fragment_id,
                            "push fenced: {reason:#}"
                        );
                        rejected("fenced");
                        continue;
                    }
                    // Pace from every unfenced push, refused ones included: a
                    // peer whose pushes keep missing their round is exactly
                    // the one the grace window must be sized for.
                    step_rates.note(member, local_step, Instant::now());
                    if !inflight
                        .iter()
                        .any(|round| round.t == global_step && round.p == fragment_id as usize)
                    {
                        arrival_gaps.note_late(global_step, fragment_id as usize, Instant::now());
                    }
                    let base_version = push.base_version;
                    if let Some(carry) = late_carry.as_mut() {
                        if carry.reuses_anchor(member, &push) {
                            warn!(
                                learner_id,
                                generation,
                                step = global_step,
                                fragment = fragment_id,
                                base = base_version,
                                "push reuses an anchor whose delta was already taken"
                            );
                            rejected("anchor_reused");
                            continue;
                        }
                        let open = inflight.iter().any(|round| {
                            round.t == global_step
                                && round.p == fragment_id as usize
                                && round.attempt == push.round_attempt
                        });
                        if !open {
                            let current = st.versions[fragment_id as usize];
                            let outcome = carry.carry(&mut inflight, member, push, next_t, current);
                            if !matches!(outcome, CarryOutcome::Rejected(_)) {
                                carry.spend_anchor(member, fragment_id as usize, base_version);
                            }
                            match outcome {
                                CarryOutcome::Attached { step, lag } => {
                                    log_carry(&cfg, dynamic, member, global_step, step, Ok(lag))
                                }
                                CarryOutcome::Queued => info!(
                                    learner_id,
                                    generation,
                                    step = global_step,
                                    fragment = fragment_id,
                                    base = base_version,
                                    "late push queued for the fragment's next round"
                                ),
                                CarryOutcome::Rejected(disposition) => log_carry(
                                    &cfg,
                                    dynamic,
                                    member,
                                    global_step,
                                    global_step,
                                    Err(disposition),
                                ),
                            }
                            continue;
                        }
                    }
                    let round_base = inflight
                        .iter()
                        .find(|round| round.t == global_step && round.p == fragment_id as usize)
                        .map(|round| round.base_version);
                    let disposition =
                        route_push(&mut inflight, member, push, cfg.max_base_lag, fixed_roster);
                    match disposition {
                        PushDisposition::Accepted => {
                            arrival_gaps.note_accepted(global_step, fragment_id as usize, Instant::now());
                            if let Some(carry) = late_carry.as_mut() {
                                carry.note_gathered(member, fragment_id as usize, base_version);
                            }
                            let lag = round_base.map_or(0, |base| base.saturating_sub(base_version));
                            if lag == 0 {
                                pool("delta_accepted", format!("\"island_id\":{learner_id}"));
                            } else {
                                pool(
                                    "delta_carried_over",
                                    format!(
                                        "\"island_id\":{learner_id},\"base\":{base_version},\"lag\":{lag},\"discount\":{:?}",
                                        cfg.membership.carry_discount(lag)
                                    ),
                                );
                            }
                        }
                        PushDisposition::Duplicate => warn!(
                            learner_id,
                            generation,
                            step = global_step,
                            fragment = fragment_id,
                            "duplicate push rejected"
                        ),
                        // Fixed roster: a stale base is a strict failure. In an island
                        // pool a late island is only refused (delta_rejected record).
                        PushDisposition::StaleBase if strict_exact && !dynamic => {
                            let metric = "rejected_stale_updates";
                            let message = format!(
                                "learner {learner_id} generation {generation} step {global_step} fragment {fragment_id}: {disposition:?}"
                            );
                            append_strict_failure(cfg.event_tape.as_deref(), metric, &message);
                            bail!("RL strict failure {metric}: {message}");
                        }
                        disposition => {
                            warn!(
                                learner_id,
                                generation,
                                step = global_step,
                                fragment = fragment_id,
                                reason = ?disposition,
                                "push rejected"
                            );
                            rejected(match disposition {
                                PushDisposition::StaleBase => "carry_lag_exceeded",
                                PushDisposition::OutOfRound => "out_of_round",
                                PushDisposition::FutureBase => "future_base",
                                _ => "unexpected_member",
                            });
                        }
                    }
                }
                Event::Leave { member, reason } => {
                    info!(
                        learner_id = member.learner_id,
                        generation = member.generation,
                        reason = %leave_reason_name(reason),
                        "island leaves the pool"
                    );
                    leave_reasons.insert(member, reason);
                }
                Event::Heartbeat { member, local_step } => {
                    registry.lock().unwrap().membership.renew(member, Instant::now());
                    if is_current_member(&registry, member) {
                        step_rates.note(member, local_step, Instant::now());
                    }
                }
                Event::Hello { group } => {
                    if registry.lock().unwrap().is_recovering(group.member) {
                        // Peer recovery reads the clock only from durable rounds.
                        if let Some(pending) = publishing.take() {
                            settle_pending_publish(
                                &cfg,
                                &st,
                                &registry,
                                &mut last_sync_secs,
                                pending,
                                &budget_cutoff,
                                &mut busy_fragments,
                            )
                            .await?;
                        }
                        let candidates = recovery_candidates(&registry, &inflight);
                        let out = recovery.begin(group.member, st.global_step, &st.versions, &candidates, Instant::now());
                        info!(
                            learner_id = group.member.learner_id,
                            generation = group.member.generation,
                            t_s = st.global_step,
                            "peer recovery started"
                        );
                        send_recovery(&registry, out).await;
                        continue;
                    }
                    // A queued HELLO may already have been superseded by a
                    // newer connection generation. Never catch up or rebind
                    // rounds to anything except the registry's current
                    // generation for this logical learner.
                    if !is_current_member(&registry, group.member) {
                        warn!(
                            learner_id = group.member.learner_id,
                            generation = group.member.generation,
                            "superseded learner reconnect ignored"
                        );
                        continue;
                    }

                    // Rejoining learner: catch it up to the current
                    // parameters, then atomically rebind only its unanswered
                    // slots in existing rounds. The scheduler sends the
                    // catch-up one fragment per pass (see `CatchUps`) and
                    // revalidates the generation before each send, because a
                    // newer HELLO may supersede this one meanwhile.
                    let fragments = if st.global_step < last_step { st.owned_fragments() } else { 0..0 };
                    catch_ups.enqueue(group.member, group, fragments, CatchUpTail::REJOIN);
                }
                Event::Recovery { member, msg } => {
                    let restored = match msg {
                        RecoveryMsg::SnapshotReady(ready) => {
                            let candidates = recovery_candidates(&registry, &inflight);
                            let out = recovery.on_snapshot_ready(member, ready, &candidates, Instant::now());
                            send_recovery(&registry, out).await;
                            continue;
                        }
                        RecoveryMsg::Failed(reason) => {
                            warn!(learner_id = member.learner_id, generation = member.generation, %reason, "recovery source refused by island");
                            let candidates = recovery_candidates(&registry, &inflight);
                            let out = recovery.on_failed(member, reason, &candidates, Instant::now());
                            send_recovery(&registry, out).await;
                            continue;
                        }
                        RecoveryMsg::Restored(held) => held,
                    };
                    if !recovery.is_pending(member) {
                        // E.g. an island recovered on the degraded path during
                        // initialization reports its restore late.
                        ignore_recovery_message(member, "training");
                        continue;
                    }
                    // Replay against durable versions only.
                    if let Some(pending) = publishing.take() {
                        settle_pending_publish(
                            &cfg,
                            &st,
                            &registry,
                            &mut last_sync_secs,
                            pending,
                            &budget_cutoff,
                            &mut busy_fragments,
                        )
                        .await?;
                    }
                    let group = registry.lock().unwrap().groups.get(&member).cloned();
                    let Some(group) = group else { continue };
                    let replay = match recovery.on_restored(member, restored.as_deref(), &st.versions) {
                        Ok(replay) => replay,
                        Err(error) => {
                            warn!(learner_id = member.learner_id, generation = member.generation, "recovery refused: {error:#}");
                            let _ = group.send_small(MSG_ERROR, format!("recovery refused: {error:#}").into()).await;
                            continue;
                        }
                    };
                    // Resend the newest value of each newer fragment; no
                    // broadcast can interleave until the island is promoted.
                    for &p in &replay {
                        match send_fragment_until_cutoff(&st, &group, p, MSG_BCAST_FRAGMENT, bulk_dtype(st.wire_dtype), &budget_cutoff).await {
                            Ok(true) => {}
                            Ok(false) => {
                                cutoff_established = true;
                                break 'outer;
                            }
                            Err(error) => warn!(learner_id = member.learner_id, fragment = p, %error, "recovery replay encode failed"),
                        }
                    }
                    let promoted = {
                        let mut registry = registry.lock().unwrap();
                        let promoted = registry.promote_recovered(member);
                        if promoted {
                            registry.membership.record_catch_up(member, &st.versions);
                        }
                        promoted
                    };
                    let Some((record, done)) = recovery.finish(member, st.global_step, &st.versions, replay.len(), Instant::now()) else {
                        continue;
                    };
                    if !promoted {
                        continue;
                    }
                    append_recovery_record(cfg.event_tape.as_deref(), &record);
                    if record.missed {
                        warn!(learner_id = member.learner_id, steps = record.steps, bound = ?record.bound_steps, "peer recovery missed its bound");
                    } else {
                        info!(learner_id = member.learner_id, steps = record.steps, path = ?record.source, "peer recovery done");
                    }
                    send_recovery(&registry, vec![done]).await;
                    if dynamic {
                        // The pool took the island back with recovered state; its
                        // pushes are fenced at the versions it now holds.
                        append_pool_record(
                            cfg.event_tape.as_deref(),
                            "pool_join",
                            cfg.syncer_epoch,
                            &format!(
                                "\"island_id\":{},\"generation\":{},\"catch_up\":true,\"recovered\":true,\"base_version\":{}",
                                member.learner_id, member.generation, st.global_step
                            ),
                        );
                        write_pool_status(&cfg, &registry, st.global_step, "running");
                    }
                    if !repull_rounds_for(&group, &registry, &mut inflight, fixed_roster, &budget_cutoff).await {
                        cutoff_established = true;
                        break 'outer;
                    }
                }
                Event::Init { .. } | Event::InitDigest { .. } => {} // already initialized; ignore
                Event::FinalAck { member, .. } => {
                    warn!(
                        learner_id = member.learner_id,
                        generation = member.generation,
                        "premature final acknowledgement dropped"
                    )
                }
                Event::BudgetDone {
                    member,
                    local_steps,
                } => {
                    // Validation is deliberately deferred until after the
                    // bounded quiesce has started.  Even a malformed or
                    // superseded report closes the gate fail-closed and must
                    // not strand already-accepted worker jobs.
                    first_budget_report = Some((member, local_steps));
                    cutoff_established = true;
                    break 'outer;
                }
                Event::Disconnected { member } => {
                    // A disconnect never erases accepted work. An unanswered
                    // slot may move to the already-current replacement only
                    // after the captured generation is gone. This also covers
                    // the ordering where the replacement's HELLO was processed
                    // while the old generation was still alive and therefore
                    // could not alter the frozen round.
                    warn!(
                        learner_id = member.learner_id,
                        generation = member.generation,
                        "learner connection generation disconnected"
                    );
                    if dynamic && !lease_expired.contains(&member) {
                        // Only a departure without a newer generation leaves the
                        // pool; an expired lease already wrote its pool_leave.
                        let replaced = registry.lock().unwrap().current.contains_key(&member.learner_id);
                        if !replaced {
                            append_pool_record(
                                cfg.event_tape.as_deref(),
                                "pool_leave",
                                cfg.syncer_epoch,
                                &format!(
                                    "\"island_id\":{},\"generation\":{},\"reason\":\"{}\"",
                                    member.learner_id,
                                    member.generation,
                                    leave_reasons
                                        .get(&member)
                                        .map_or_else(|| "disconnected".to_string(), |r| leave_reason_name(*r))
                                ),
                            );
                        }
                    }
                    leave_reasons.remove(&member);
                    step_rates.remove(member);
                    let candidates = recovery_candidates(&registry, &inflight);
                    let out = recovery.on_gone(member, &candidates, Instant::now());
                    send_recovery(&registry, out).await;

                    let replacement = {
                        let registry = registry.lock().unwrap();
                        registry
                            .current
                            .get(&member.learner_id)
                            .copied()
                            .filter(|replacement| *replacement != member)
                    };
                    if !fixed_roster {
                        if let Some(replacement) = replacement {
                            let group = {
                                let registry = registry.lock().unwrap();
                                registry.groups.get(&replacement).cloned()
                            };
                            if let Some(group) = group {
                                // Catch-up must precede a replay even when the
                                // disconnect event wins the race with the queued
                                // replacement HELLO: the rebind runs only once
                                // the catch-up has finished.
                                let fragments =
                                    if st.global_step < last_step { st.owned_fragments() } else { 0..0 };
                                catch_ups.enqueue(replacement, group, fragments, CatchUpTail::REBIND);
                            }
                        }
                    }
                }
            },
        }
    }

    if cfg.learner_budget_steps.is_some() {
        cutoff_established |= budget_cutoff.deadline().is_some();
        if !cutoff_established {
            bail!("learner-budget scheduler exited without a cutoff report");
        }
        let deadline = budget_cutoff
            .deadline()
            .context("learner-budget scheduler exited without a cutoff deadline")?;
        let cancelled_inflight = inflight.len();
        let cancelled_ready = ready.len();
        let cancelled_computing = computing.len();
        let target = cfg.learner_budget_steps.unwrap();
        inflight.clear();
        ready.clear();
        busy_fragments.clear();
        let (drain_result, reports_result) = tokio::join!(
            quiesce_accepted_compute(&mut computing, &st, deadline),
            collect_budget_reports(
                target,
                cfg.learners,
                &mut budget_reports,
                &mut events,
                &registry,
                budget_cutoff.timeout(),
                cfg.event_tape.as_deref(),
                first_budget_report,
            )
        );
        // Do not short-circuit either bounded cleanup branch: malformed
        // reports and poisoned workers both fail closed only after the other
        // branch has had its bounded opportunity to quiesce.
        drain_result?;
        reports_result?;
        eval_checkpoints.finish().await?;
        save_budget_checkpoint(&cfg, &st)?;
        info!(
            cancelled_inflight,
            cancelled_computing,
            cancelled_ready,
            target_steps = target,
            step = st.global_step,
            "learner-budget cutoff established"
        );
        return Ok(());
    }

    if next_commit_t != schedule.step_after(last_step)
        || st.global_step != last_step
        || !busy_fragments.is_empty()
        || publishing.is_some()
    {
        bail!(
            "non-quiescent terminal cut: next_commit_t={next_commit_t} global_step={} busy={:?}",
            st.global_step,
            busy_fragments
        );
    }
    // A late push queued for a fragment whose last round already merged has
    // no round left to join (`--terminal-grace-ms` keeps last rounds open).
    if let Some(carry) = late_carry.as_ref() {
        let dropped: usize = carry.queued.values().map(Vec::len).sum();
        if dropped > 0 {
            let mut fragments: Vec<usize> = carry.queued.keys().copied().collect();
            fragments.sort_unstable();
            warn!(dropped, fragments = ?fragments, "late pushes left out of the final cut: no later round");
        }
    }
    // Explicitly drain and health-check every SVD worker before a checkpoint
    // or marker can publish the terminal cut.
    st.drain_iso_backend().await?;
    // The final checkpoint never shares the disk with an evaluation copy.
    eval_checkpoints.finish().await?;

    // The outer loop is now quiescent: every launched round has completed
    // and every computed result has committed in strict t order.
    // Persist this authoritative cut regardless of the periodic checkpoint
    // interval so a non-divisible total_steps can never leave a stale final
    // checkpoint behind.
    if let Some(path) = &cfg.checkpoint_path {
        if !checkpoint_each_round {
            if cfg.mark_final_checkpoint {
                remove_final_marker(path)?;
            }
            st.save_checkpoint(path)?;
            info!(
                step = st.global_step,
                path = %path.display(),
                "final checkpoint written"
            );
        }
    }
    if cfg.policy_sweep_fragments.is_some() {
        if let Some(tape) = cfg.event_tape.as_deref() {
            append_policy_sweep_ledger_snapshot(tape, &st, "complete")?;
        }
    }
    if cfg.mark_final_checkpoint {
        write_final_marker(cfg.checkpoint_path.as_ref().unwrap(), st.global_step)?;
    }
    if let Some(path) = &cfg.final_state {
        dump_state(&st, path)?;
        info!(path = %path.display(), "final global state written");
    }
    // An island still under peer recovery takes the final cut like any
    // member: it is promoted with the degraded source first.
    let recovering: Vec<Arc<Group>> = {
        let registry = registry.lock().unwrap();
        registry.recovering.values().filter_map(|member| registry.groups.get(member).cloned()).collect()
    };
    for group in recovering {
        admit_recovering_outside_training(&registry, &group, "finalization").await;
    }
    // Freeze terminal membership to the live groups at the final cut.
    // Learners already abandoned by fleet recovery are not valid artifact
    // producers and must not prevent surviving learners from finalizing.
    let final_members: HashSet<u32> = if fixed_roster {
        (0..cfg.learners).collect()
    } else {
        current_groups(&registry)
            .into_iter()
            .map(|group| group.member.learner_id)
            .collect()
    };
    if dynamic {
        write_pool_status(&cfg, &registry, st.global_step, "finishing");
    }
    finalize_learners(&cfg, &st, &mut events, &registry, &final_members).await?;
    if dynamic {
        write_pool_status(&cfg, &registry, st.global_step, "finished");
    }
    info!("training complete after {} outer steps", cfg.total_steps);
    // Give writer tasks a moment to flush the final control frames.
    tokio::time::sleep(Duration::from_secs(2)).await;
    return Ok(());
}

fn record_budget_report(
    reports: &mut HashSet<u32>,
    target_steps: u64,
    member: Member,
    local_steps: u64,
) -> Result<()> {
    if local_steps != target_steps {
        bail!("BUDGET_DONE reported {local_steps} steps, expected {target_steps}");
    }
    // Logical completion is idempotent across retries/reconnections. A
    // repeated report with the same required step cannot change the cut.
    reports.insert(member.learner_id);
    Ok(())
}

fn save_budget_checkpoint(cfg: &Config, st: &GlobalState) -> Result<()> {
    let path = cfg
        .checkpoint_path
        .as_ref()
        .context("learner budget cutoff requires --checkpoint-path")?;
    st.save_checkpoint(path)?;
    info!(
        step = st.global_step,
        path = %path.display(),
        "learner-budget cutoff checkpoint written"
    );
    Ok(())
}

async fn drain_iso_before(st: &GlobalState, deadline: tokio::time::Instant) -> Result<()> {
    tokio::time::timeout_at(deadline, st.drain_iso_backend())
        .await
        .context("timed out draining the ISO backend at learner-budget cutoff")??;
    Ok(())
}

/// Abort scheduler-owned compute futures and fence every job already accepted
/// by the backend.  Failure and timeout are both fail-closed: no cutoff
/// checkpoint is written unless this entire quiesce succeeds.
async fn quiesce_accepted_compute(
    computing: &mut tokio::task::JoinSet<Result<ComputedRound>>,
    st: &GlobalState,
    deadline: tokio::time::Instant,
) -> Result<()> {
    computing.abort_all();
    let mut task_error: Option<anyhow::Error> = None;
    loop {
        let joined = match tokio::time::timeout_at(deadline, computing.join_next()).await {
            Ok(joined) => joined,
            Err(_) => {
                task_error = Some(anyhow::anyhow!(
                    "timed out cancelling scheduler compute tasks at learner-budget cutoff"
                ));
                break;
            }
        };
        match joined {
            None => break,
            Some(Ok(Ok(_completed))) => {
                // The result linearized before cutoff but had not committed;
                // discard it as one whole round.
            }
            Some(Ok(Err(error))) => {
                if task_error.is_none() {
                    task_error =
                        Some(error.context(
                            "accepted compute failed while quiescing learner-budget cutoff",
                        ));
                }
            }
            Some(Err(error)) if error.is_cancelled() => {}
            Some(Err(error)) => {
                if task_error.is_none() {
                    task_error = Some(anyhow::anyhow!(
                        "scheduler compute task failed while quiescing cutoff: {error}"
                    ));
                }
            }
        }
    }

    // Run the backend fence even if a scheduler task failed.  This gives
    // accepted external jobs their bounded drain opportunity before failure.
    let backend_result = drain_iso_before(st, deadline).await;
    if let Some(error) = task_error {
        return match backend_result {
            Ok(()) => Err(error),
            Err(backend_error) => Err(error.context(format!(
                "ISO backend also failed while quiescing cutoff: {backend_error:#}"
            ))),
        };
    }
    backend_result
}

async fn collect_budget_reports(
    target: u64,
    expected_learners: u32,
    reports: &mut HashSet<u32>,
    events: &mut mpsc::Receiver<Event>,
    registry: &Registry,
    progress_timeout: Duration,
    event_tape: Option<&std::path::Path>,
    first_report: Option<(Member, u64)>,
) -> Result<()> {
    let mut first_report = first_report;
    let mut progress_deadlines: HashMap<u32, tokio::time::Instant> = (0..expected_learners)
        .filter(|learner_id| !reports.contains(learner_id))
        .map(|learner_id| (learner_id, tokio::time::Instant::now() + progress_timeout))
        .collect();
    while reports.len() < expected_learners as usize {
        let event = match first_report.take() {
            Some((member, local_steps)) => Event::BudgetDone {
                member,
                local_steps,
            },
            None => {
                let deadline = progress_deadlines
                    .values()
                    .copied()
                    .min()
                    .context("no progress lease for an unfinished learner")?;
                tokio::time::timeout_at(deadline, events.recv())
                    .await
                    .map_err(|_| {
                        let now = tokio::time::Instant::now();
                        let mut stalled: Vec<_> = progress_deadlines
                            .iter()
                            .filter_map(|(learner_id, deadline)| {
                                (*deadline <= now).then_some(*learner_id)
                            })
                            .collect();
                        stalled.sort_unstable();
                        anyhow::anyhow!(
                            "timed out collecting BUDGET_DONE after no progress for {:?}: received {}/{}",
                            stalled,
                            reports.len(),
                            expected_learners
                        )
                    })?
                    .context("event channel closed while collecting BUDGET_DONE reports")?
            }
        };
        match event {
            Event::Fatal { metric, message } => {
                append_strict_failure(event_tape, metric, &message);
                bail!("RL strict failure {metric}: {message}");
            }
            Event::BudgetDone {
                member,
                local_steps,
            } => {
                if !is_current_member(registry, member) {
                    bail!("BUDGET_DONE from a superseded learner");
                }
                record_budget_report(reports, target, member, local_steps)?;
                progress_deadlines.remove(&member.learner_id);
            }
            Event::Heartbeat { member, local_step } => {
                if is_current_member(registry, member)
                    && !reports.contains(&member.learner_id)
                    && local_step <= target
                {
                    progress_deadlines.insert(
                        member.learner_id,
                        tokio::time::Instant::now() + progress_timeout,
                    );
                }
            }
            Event::Push { member, push } => {
                if is_current_member(registry, member)
                    && !reports.contains(&member.learner_id)
                    && push.local_step <= target
                {
                    progress_deadlines.insert(
                        member.learner_id,
                        tokio::time::Instant::now() + progress_timeout,
                    );
                }
            }
            Event::Hello { group } => {
                admit_recovering_outside_training(registry, &group, "the budget cutoff").await;
                let learner_id = group.member.learner_id;
                if is_current_member(registry, group.member) && !reports.contains(&learner_id) {
                    progress_deadlines
                        .insert(learner_id, tokio::time::Instant::now() + progress_timeout);
                }
            }
            Event::Disconnected { member } => warn!(
                learner_id = member.learner_id,
                generation = member.generation,
                "disconnected while waiting for learner-budget cutoff"
            ),
            Event::Init { .. } | Event::InitDigest { .. } | Event::FinalAck { .. } | Event::Leave { .. } => {}
            Event::Recovery { member, .. } => ignore_recovery_message(member, "the budget cutoff"),
        }
    }
    Ok(())
}

/// One in-flight sync round: the pull for fragment `p` at global step `t`
/// and the pushes gathered so far.
struct Round {
    t: u64,
    p: usize,
    base_version: u64,
    attempt: u32,
    pull: bytes::Bytes,
    started: Instant,
    expected_members: Vec<Member>,
    quorum_deadline: Instant,
    grace_deadline: Option<Instant>,
    quorum_size: usize,
    /// Capacity quorum θ frozen at launch; None = count quorum `quorum_size`.
    capacity_quorum: Option<CapacityQuorum>,
    /// Soft deadline: merge what arrived (at least `soft_min`) even below
    /// quorum once it passes. None = off.
    soft_deadline: Option<Instant>,
    soft_min: usize,
    quorum_ms: Option<u64>,
    grace_ms: Option<u64>,
    /// Streaming reduction of accepted pushes (see `round_fold`); `None`
    /// keeps every push's gradient until the buffered merge.
    fold: Option<RoundFold>,
    pushes: HashMap<Member, Push>,
    /// Late deltas of earlier rounds of this fragment (async rounds, see
    /// `LateCarry`): merged here against their own base, in arrival order.
    /// They never count toward quorum and stay buffered until the merge.
    carried: Vec<(Member, Push)>,
}

impl Round {
    fn quorum_reached(&self) -> bool {
        match &self.capacity_quorum {
            Some(quorum) => quorum.reached(&self.expected_members, self.pushes.keys()),
            None => self.pushes.len() >= self.quorum_size,
        }
    }

    /// Drop an expired member from the islands this round still waits for.
    /// An accepted push is permanent and keeps its slot.
    fn forget_unanswered(&mut self, member: Member) -> bool {
        if self.pushes.contains_key(&member) {
            return false;
        }
        let before = self.expected_members.len();
        self.expected_members.retain(|expected| *expected != member);
        before != self.expected_members.len()
    }
}

#[derive(Debug, Eq, PartialEq)]
struct ReboundPull {
    t: u64,
    p: usize,
    attempt: u32,
    base_version: u64,
    old_generation: u64,
    pull: bytes::Bytes,
}

struct ComputedRound {
    round: Round,
    merge: ComputedMerge,
    compute_secs: f64,
}

enum SchedulerWake {
    Deadline,
    Event(Option<Event>),
    Computed(Option<std::result::Result<Result<ComputedRound>, tokio::task::JoinError>>),
    Durable(std::result::Result<Result<()>, tokio::task::JoinError>),
}

async fn sleep_until_opt(deadline: Option<tokio::time::Instant>) {
    match deadline {
        Some(deadline) => tokio::time::sleep_until(deadline).await,
        None => std::future::pending().await,
    }
}

type ComputeJoin = Option<std::result::Result<Result<ComputedRound>, tokio::task::JoinError>>;

/// Event delivery is intentionally biased over compute completion.  The
/// cutoff gate is the correctness boundary even when an older ordinary event
/// precedes BUDGET_DONE in the FIFO, while this preference makes the direct
/// simultaneous-ready case deterministic and minimizes discarded work.
async fn prefer_event_over_compute<EventFuture, ComputeFuture>(
    event: EventFuture,
    compute: ComputeFuture,
) -> SchedulerWake
where
    EventFuture: Future<Output = Option<Event>>,
    ComputeFuture: Future<Output = ComputeJoin>,
{
    tokio::select! {
        biased;
        event = event => SchedulerWake::Event(event),
        result = compute => SchedulerWake::Computed(result),
    }
}

fn take_next_commit<T>(ready: &mut BTreeMap<u64, T>, next_t: u64) -> Option<T> {
    ready.remove(&next_t)
}

#[derive(Debug, Eq, PartialEq)]
enum PushDisposition {
    Accepted,
    Duplicate,
    UnexpectedMember,
    StaleBase,
    FutureBase,
    OutOfRound,
}

#[derive(Debug, Eq, PartialEq)]
enum RoundAction {
    Wait,
    Complete,
    Restart,
}

fn round_action(round: &Round, now: Instant) -> RoundAction {
    // A round never merges nothing: if lease expiry emptied its expected
    // set, it waits for the quorum timeout and relaunches.
    if !round.pushes.is_empty() && round.pushes.len() >= round.expected_members.len() {
        return RoundAction::Complete;
    }
    if round.quorum_reached() {
        return match round.grace_deadline {
            Some(deadline) if now >= deadline => RoundAction::Complete,
            _ => RoundAction::Wait,
        };
    }
    if round.soft_deadline.is_some_and(|deadline| now >= deadline)
        && round.pushes.len() >= round.soft_min.max(1)
    {
        return RoundAction::Complete;
    }
    if now >= round.quorum_deadline {
        RoundAction::Restart
    } else {
        RoundAction::Wait
    }
}

/// Return a ready round only when it is the oldest in-flight global step.
/// Later rounds may finish gathering first, but cannot become durable until
/// every earlier step has committed.
fn next_committable_round(rounds: &[Round], now: Instant) -> Option<usize> {
    let (index, oldest) = rounds
        .iter()
        .enumerate()
        .min_by_key(|(_index, round)| round.t)?;
    (round_action(oldest, now) == RoundAction::Complete).then_some(index)
}

fn fragment_available(rounds: &[Round], fragment_id: usize) -> bool {
    !rounds.iter().any(|round| round.p == fragment_id)
}

fn should_replay_pull(round: &Round, member: Member) -> bool {
    round
        .expected_members
        .iter()
        .any(|expected| expected.learner_id == member.learner_id && *expected != member)
        && !round
            .pushes
            .keys()
            .any(|accepted| accepted.learner_id == member.learner_id)
}

/// Fixed-roster strict runs freeze logical learner IDs rather than connection
/// generations. Preserve that contract by replaying an unanswered pull to the
/// current generation without rewriting the captured roster entry.
fn fixed_roster_repulls(rounds: &[Round], replacement: Member) -> Vec<ReboundPull> {
    rounds
        .iter()
        .filter(|round| should_replay_pull(round, replacement))
        .map(|round| {
            let old_generation = round
                .expected_members
                .iter()
                .find(|expected| expected.learner_id == replacement.learner_id)
                .expect("replay predicate requires a captured logical learner")
                .generation;
            ReboundPull {
                t: round.t,
                p: round.p,
                attempt: round.attempt,
                base_version: round.base_version,
                old_generation,
                pull: round.pull.clone(),
            }
        })
        .collect()
}

/// Rebind unanswered slots in existing rounds to a reconnect generation.
///
/// Accepted pushes are permanent: if any generation of this logical learner
/// has already answered a round, that round remains frozen. A still-connected
/// captured generation also remains frozen; a replacement can take over only
/// after that exact connection is gone. The registry check prevents a delayed
/// HELLO from rebinding rounds backwards after a still newer connection
/// superseded it while catch-up was in progress.
fn rebind_current_unanswered_rounds(
    registry: &Registry,
    rounds: &mut [Round],
    replacement: Member,
) -> Vec<ReboundPull> {
    // Keep the registry lock through the membership rewrites.  Otherwise a
    // newer HELLO could become current after the check but before this loop,
    // letting a delayed HELLO rebind rounds backwards.
    let registry = registry.lock().unwrap();
    if registry.current.get(&replacement.learner_id) != Some(&replacement) {
        return Vec::new();
    }

    let mut repulls = Vec::new();
    for round in rounds {
        if round
            .pushes
            .keys()
            .any(|member| member.learner_id == replacement.learner_id)
        {
            continue;
        }

        let Some(expected_index) = round.expected_members.iter().position(|member| {
            member.learner_id == replacement.learner_id && *member != replacement
        }) else {
            continue;
        };

        let expected = round.expected_members[expected_index];
        if registry.groups.contains_key(&expected) {
            continue;
        }

        let old_generation = expected.generation;
        round.expected_members[expected_index] = replacement;
        round.expected_members.sort_unstable();
        repulls.push(ReboundPull {
            t: round.t,
            p: round.p,
            attempt: round.attempt,
            base_version: round.base_version,
            old_generation,
            pull: round.pull.clone(),
        });
    }
    repulls
}

fn route_push(
    rounds: &mut [Round],
    member: Member,
    push: Push,
    max_base_lag: Option<u64>,
    fixed_roster: bool,
) -> PushDisposition {
    let Some(round) = rounds.iter_mut().find(|round| {
        round.t == push.global_step
            && round.p == push.fragment_id as usize
            && round.attempt == push.round_attempt
    }) else {
        return PushDisposition::OutOfRound;
    };
    let expected = if fixed_roster {
        round
            .expected_members
            .iter()
            .any(|expected| expected.learner_id == member.learner_id)
    } else {
        round.expected_members.contains(&member)
    };
    if !expected {
        return PushDisposition::UnexpectedMember;
    }
    if push.base_version > round.base_version {
        return PushDisposition::FutureBase;
    }
    if max_base_lag.is_some_and(|limit| round.base_version - push.base_version > limit) {
        return PushDisposition::StaleBase;
    }
    let duplicate = if fixed_roster {
        round
            .pushes
            .keys()
            .any(|accepted| accepted.learner_id == member.learner_id)
    } else {
        round.pushes.contains_key(&member)
    };
    if duplicate {
        return PushDisposition::Duplicate;
    }
    round.pushes.insert(member, push);
    advance_fold(round, member);
    PushDisposition::Accepted
}

/// Late-push carry for asynchronous rounds.
///
/// A push that reaches the syncer after its round closed still holds the
/// island's work since its anchor (`local - raw anchor at base_version`).
/// Instead of dropping it, the syncer merges it into the next round of the
/// same fragment against its own base, weighted by its counters (and the
/// carry discount), as long as its base lag stays within `--max-base-lag`.
///
/// A carried delta counts once: after the syncer carries a delta from an
/// anchor (fragment version), a later push of that island connection from
/// the same or an older anchor of that fragment is refused
/// (`anchor_reused`), and a late push from an anchor whose delta was
/// already merged is not carried. Pushes that never involve a carry keep
/// the existing stale-base rules.
struct LateCarry {
    limit: u64,
    /// Late pushes waiting for their fragment's next round.
    queued: HashMap<usize, Vec<(Member, Push)>>,
    /// Newest base version of a carried delta per island connection and
    /// fragment.
    anchors: HashMap<(Member, usize), u64>,
    /// Newest base version of a delta gathered by its own round.
    gathered: HashMap<(Member, usize), u64>,
}

#[derive(Debug, Eq, PartialEq)]
enum CarryOutcome {
    /// Added to the open round of this fragment launched at `step`.
    Attached { step: u64, lag: u64 },
    /// Held until the fragment's next round launches.
    Queued,
    Rejected(PushDisposition),
}

impl LateCarry {
    /// Carry is on for asynchronous rounds only: a positive base-lag bound
    /// and no fixed roster. Strict (`--max-base-lag 0`) and unbounded runs
    /// keep refusing late pushes.
    fn new(max_base_lag: Option<u64>, fixed_roster: bool) -> Option<Self> {
        match max_base_lag {
            Some(limit) if limit > 0 && !fixed_roster => Some(Self {
                limit,
                queued: HashMap::new(),
                anchors: HashMap::new(),
                gathered: HashMap::new(),
            }),
            _ => None,
        }
    }

    /// True when `push` reuses an anchor this connection already spent.
    fn reuses_anchor(&self, member: Member, push: &Push) -> bool {
        self.anchors
            .get(&(member, push.fragment_id as usize))
            .is_some_and(|&spent| push.base_version <= spent)
    }

    fn spend_anchor(&mut self, member: Member, fragment: usize, base_version: u64) {
        let spent = self.anchors.entry((member, fragment)).or_insert(0);
        *spent = (*spent).max(base_version);
    }

    fn note_gathered(&mut self, member: Member, fragment: usize, base_version: u64) {
        let taken = self.gathered.entry((member, fragment)).or_insert(0);
        *taken = (*taken).max(base_version);
    }

    /// A restarted round drops its gathered pushes; their islands answer the
    /// new attempt from the same anchor, so the anchor is free again.
    fn release_anchor(&mut self, member: Member, push: &Push) {
        let key = (member, push.fragment_id as usize);
        if self.gathered.get(&key) == Some(&push.base_version) {
            if push.base_version == 0 {
                self.gathered.remove(&key);
            } else {
                self.gathered.insert(key, push.base_version - 1);
            }
        }
    }

    /// Route a push that `route_push` found no open round for.
    /// `next_t` is the next step to launch; `current_version` the fragment's
    /// committed version.
    fn carry(
        &mut self,
        rounds: &mut [Round],
        member: Member,
        push: Push,
        next_t: u64,
        current_version: u64,
    ) -> CarryOutcome {
        let p = push.fragment_id as usize;
        // A push for a round still open under another attempt, or for a
        // round never launched, is not late.
        if push.global_step >= next_t
            || rounds.iter().any(|round| round.t == push.global_step && round.p == p)
        {
            return CarryOutcome::Rejected(PushDisposition::OutOfRound);
        }
        if push.base_version > current_version {
            return CarryOutcome::Rejected(PushDisposition::FutureBase);
        }
        if self
            .gathered
            .get(&(member, p))
            .is_some_and(|&taken| push.base_version <= taken)
        {
            return CarryOutcome::Rejected(PushDisposition::Duplicate);
        }
        if current_version - push.base_version > self.limit {
            return CarryOutcome::Rejected(PushDisposition::StaleBase);
        }
        if let Some(round) = rounds
            .iter_mut()
            .find(|round| round.p == p && round.t > push.global_step)
        {
            return match attach_carry(round, member, push, self.limit) {
                Ok(lag) => CarryOutcome::Attached { step: round.t, lag },
                Err(disposition) => CarryOutcome::Rejected(disposition),
            };
        }
        self.queued.entry(p).or_default().push((member, push));
        CarryOutcome::Queued
    }

    /// Move the late pushes queued for `round`'s fragment into it.
    fn attach_queued(&mut self, round: &mut Round) -> Vec<(Member, u64, Result<u64, PushDisposition>)> {
        let Some(queued) = self.queued.remove(&round.p) else {
            return Vec::new();
        };
        queued
            .into_iter()
            .map(|(member, push)| {
                let missed = push.global_step;
                (member, missed, attach_carry(round, member, push, self.limit))
            })
            .collect()
    }
}

/// Log (and with dynamic membership, tape) the outcome of a late push.
fn log_carry(
    cfg: &Config,
    dynamic: bool,
    member: Member,
    missed_step: u64,
    into_step: u64,
    outcome: Result<u64, PushDisposition>,
) {
    let island = member.learner_id;
    match outcome {
        Ok(lag) => {
            info!(island, missed_step, into_step, lag, "late push carried into a later round");
            if dynamic {
                append_pool_record(
                    cfg.event_tape.as_deref(),
                    "delta_carried_over",
                    cfg.syncer_epoch,
                    &format!(
                        "\"island_id\":{island},\"missed_step\":{missed_step},\"step\":{into_step},\"lag\":{lag},\"discount\":{:?}",
                        cfg.membership.carry_discount(lag)
                    ),
                );
            }
        }
        Err(disposition) => {
            warn!(island, missed_step, reason = ?disposition, "late push not carried");
            if dynamic {
                let reason = match disposition {
                    PushDisposition::StaleBase => "carry_lag_exceeded",
                    PushDisposition::FutureBase => "future_base",
                    PushDisposition::Duplicate => "anchor_reused",
                    _ => "out_of_round",
                };
                append_pool_record(
                    cfg.event_tape.as_deref(),
                    "delta_rejected",
                    cfg.syncer_epoch,
                    &format!("\"island_id\":{island},\"reason\":\"{reason}\""),
                );
            }
        }
    }
}

/// Add a late delta to an open round; returns its base lag.
fn attach_carry(
    round: &mut Round,
    member: Member,
    push: Push,
    limit: u64,
) -> Result<u64, PushDisposition> {
    // The exact fold reproduces a fixed responder set bit for bit.
    if matches!(round.fold, Some(RoundFold::Exact(_))) {
        return Err(PushDisposition::OutOfRound);
    }
    if push.base_version > round.base_version {
        return Err(PushDisposition::FutureBase);
    }
    let lag = round.base_version - push.base_version;
    if lag > limit {
        return Err(PushDisposition::StaleBase);
    }
    round.carried.push((member, push));
    Ok(lag)
}

/// The streaming reduction for a new round attempt, or `None` to keep every
/// push's gradient until the buffered merge.
///
/// * Exact (bit-identical, see `fold.rs`): every expected learner must
///   answer (quorum covers the roster) and weights are equal.
/// * Streaming (`--weighted-fold streaming`): any other quorum or weight;
///   pushes fold in arrival order and normalise at close.
///
/// HeLoCo corrects each push against momentum and always stays buffered;
/// so do expert token weights, which weight each tensor separately.
///
/// Dynamic membership stays buffered: a lease expiry or soft deadline can
/// drop a push that was already folded, and a carry discount changes
/// per-push weights.
fn round_fold(
    cfg: &Config,
    st: &GlobalState,
    p: usize,
    expected_members: &[Member],
    quorum_size: usize,
) -> Option<RoundFold> {
    if st.delta_correction.is_some() || !cfg.membership.is_static() || cfg.expert_token_weights {
        return None;
    }
    let fragment = st.layout.fragments.get(p)?;
    if quorum_size >= expected_members.len() && cfg.learner_weight == LearnerWeight::Equal {
        let roster: Vec<u32> = expected_members.iter().map(|m| m.learner_id).collect();
        if let Some(fold) = FragmentFold::for_equal_full_roster(fragment, &roster) {
            return Some(RoundFold::Exact(fold));
        }
    }
    match cfg.weighted_fold {
        WeightedFoldMode::Buffered => None,
        WeightedFoldMode::Streaming => {
            StreamingFold::new(fragment, cfg.learner_weight.function()).map(RoundFold::Streaming)
        }
    }
}

/// Fold what the round's fold can take now and release those gradient
/// buffers.  The exact fold takes pushes in canonical learner order (later
/// ones wait in `round.pushes`); the streaming fold takes `member`'s push.
fn advance_fold(round: &mut Round, member: Member) {
    match round.fold.as_mut() {
        None => {}
        Some(RoundFold::Exact(fold)) => {
            while let Some(next) = fold.next_learner() {
                let Some(push) = round
                    .pushes
                    .iter_mut()
                    .find_map(|(member, push)| (member.learner_id == next).then_some(push))
                else {
                    break;
                };
                let gradient = std::mem::take(&mut push.outer_gradient);
                if let Err(error) = fold.fold_next(next, gradient) {
                    fold.poison(error);
                    break;
                }
            }
        }
        Some(RoundFold::Streaming(fold)) => {
            if fold.is_poisoned() {
                return;
            }
            let Some(push) = round.pushes.get_mut(&member) else {
                return;
            };
            let weight = fold.weight_of(push.c_tokens, u64::from(push.c_steps));
            let gradient = std::mem::take(&mut push.outer_gradient);
            if let Err(error) = fold.fold(member.learner_id, gradient, weight) {
                fold.poison(error);
            }
        }
    }
}

/// Fix learner order and prepare owned matrix jobs without mutating global
/// optimizer state. This exact order is the reduction order on every run.
fn prepare_round_compute(
    cfg: &Config,
    st: &GlobalState,
    round: &mut Round,
) -> Result<PreparedMerge> {
    if st.versions.get(round.p).copied() != Some(round.base_version) {
        bail!(
            "round t={} fragment {} base version {} != current {:?}",
            round.t,
            round.p,
            round.base_version,
            st.versions.get(round.p)
        );
    }
    let mut ordered_pushes: Vec<_> = round.pushes.iter().collect();
    ordered_pushes.sort_unstable_by_key(|(member, _)| **member);
    let responders = ordered_pushes.len();
    // Carried late deltas follow this round's own pushes, in arrival order.
    ordered_pushes.extend(round.carried.iter().map(|(member, push)| (member, push)));
    let mut outer_gradients = Vec::with_capacity(ordered_pushes.len());
    let mut weights = Vec::with_capacity(ordered_pushes.len());
    let mut tensor_scales = Vec::with_capacity(ordered_pushes.len());
    for (member, push) in &ordered_pushes {
        if push.base_version < round.base_version {
            warn!(
                learner_id = member.learner_id,
                generation = member.generation,
                step = round.t,
                base = push.base_version,
                expected = round.base_version,
                "stale base-relative delta admitted"
            );
        }
        outer_gradients.push(push.outer_gradient.as_slice());
        let weight = cfg.learner_weight.function()(push.c_tokens, u64::from(push.c_steps));
        // Carry discount γ^lag per fragment version (1 when off: exact).
        weights.push(weight * cfg.membership.carry_discount(round.base_version - push.base_version));
        if cfg.expert_token_weights {
            let scales = push.tensor_scales.as_deref().with_context(|| {
                format!("round t={} fragment {}: push without expert token tags", round.t, round.p)
            })?;
            tensor_scales.push(scales);
        }
    }
    if !cfg.membership.is_static() {
        append_outer_step_record(cfg, round, &ordered_pushes, &weights);
    }
    let Some(mut fold) = round.fold.take() else {
        let tensor_scales = cfg.expert_token_weights.then_some(tensor_scales.as_slice());
        return st.prepare_merge_scaled(
            round.p,
            round.base_version,
            &outer_gradients,
            &weights,
            tensor_scales,
        );
    };
    // Every responder must have been folded exactly once.  The exact fold
    // reduced in learner-id order, which equals the buffered merge's Member
    // order when ids are unique; the streaming fold in arrival order.
    let responders: Vec<u32> =
        ordered_pushes[..responders].iter().map(|(m, _)| m.learner_id).collect();
    let mut folded = fold.folded_learners().to_vec();
    folded.sort_unstable();
    if folded != responders {
        bail!(
            "round t={} fragment {}: streaming fold covered learners {:?}, responders {:?}",
            round.t,
            round.p,
            fold.folded_learners(),
            responders
        );
    }
    // Carried deltas stay buffered until now (only a streaming fold admits
    // them, see `attach_carry`).
    if !round.carried.is_empty() {
        let RoundFold::Streaming(streaming) = &mut fold else {
            bail!("round t={} fragment {}: carried deltas need a streaming fold", round.t, round.p);
        };
        for (index, (member, push)) in round.carried.iter_mut().enumerate() {
            let weight = weights[responders.len() + index];
            streaming.fold(member.learner_id, std::mem::take(&mut push.outer_gradient), weight)?;
        }
    }
    let delta = fold
        .finish()
        .with_context(|| format!("round t={} fragment {}", round.t, round.p))?;
    st.prepare_reduced_merge(round.p, round.base_version, delta)
}

/// A round whose merge is applied to `GlobalState` but not yet externally
/// visible: no BCAST, no tape record, no log line.
struct CommittedRound {
    t: u64,
    p: usize,
    base_version: u64,
    attempt: u32,
    started: Instant,
    expected_members: Vec<Member>,
    quorum_size: usize,
    quorum_ms: Option<u64>,
    grace_ms: Option<u64>,
    pushes: HashMap<Member, Push>,
    carried: Vec<(Member, Push)>,
    gnorm: f64,
    compute_secs: f64,
    sync_start: Instant,
}

/// Commit an already-computed round. Called only for the contiguous t prefix,
/// so Nesterov, versions, ledger, event tape, checkpoint, and broadcast all
/// share one deterministic order even when SVD completion is out of order.
async fn commit_round(
    cfg: &Config,
    st: &mut GlobalState,
    registry: &Registry,
    last_sync_secs: &mut f64,
    completed: ComputedRound,
    budget_cutoff: &BudgetCutoff,
) -> Result<()> {
    let committed = apply_round_commit(cfg, st, completed)?;
    let mut checkpoint_ms = None;
    if persist_before_broadcast(cfg, committed.t) {
        if let Some(path) = cfg.checkpoint_path.as_ref() {
            let checkpoint_start = Instant::now();
            st.save_checkpoint(path)?;
            checkpoint_ms = Some(checkpoint_start.elapsed().as_millis() as u64);
            info!(step = committed.t, path = %path.display(), "checkpoint committed");
        }
    }
    publish_round(cfg, st, registry, last_sync_secs, committed, checkpoint_ms, budget_cutoff).await
}

/// Apply the merge, version, step, and ledger updates of the next contiguous
/// round. Nothing leaves the process here.
fn apply_round_commit(
    cfg: &Config,
    st: &mut GlobalState,
    completed: ComputedRound,
) -> Result<CommittedRound> {
    let ComputedRound {
        round,
        merge,
        compute_secs,
    } = completed;
    let Round {
        t,
        p,
        base_version,
        attempt,
        started,
        expected_members,
        quorum_size,
        quorum_ms,
        grace_ms,
        pushes,
        carried,
        ..
    } = round;
    let expected_t = st.schedule().step_after(st.global_step);
    if t != expected_t {
        bail!("refusing non-contiguous round commit: got step {t}, expected {expected_t}");
    }
    if st.versions.get(p).copied() != Some(base_version) {
        bail!(
            "commit t={t} fragment {p} base version {base_version} != current {:?}",
            st.versions.get(p)
        );
    }
    let sync_start = Instant::now();
    let gnorm = st.commit_merge(merge)?;
    st.versions[p] = t;
    st.global_step = t;
    let mut ordered_pushes: Vec<_> = pushes.iter().collect();
    ordered_pushes.sort_unstable_by_key(|(member, _)| **member);
    // A carried delta was never counted by the round it missed.
    let merged = ordered_pushes.into_iter().chain(carried.iter().map(|(member, push)| (member, push)));
    for (_, push) in merged {
        if let Some(fragments) = cfg.policy_sweep_fragments {
            st.record_fragment_merge(
                push.learner_id,
                push.c_steps,
                push.c_tokens,
                t.is_multiple_of(u64::from(fragments)),
            );
        } else {
            // Keep the legacy accounting path exactly as before opt-in sweep
            // mode existed.
            st.record_merge(push.learner_id, push.c_steps, push.c_tokens);
        }
    }
    Ok(CommittedRound {
        t,
        p,
        base_version,
        attempt,
        started,
        expected_members,
        quorum_size,
        quorum_ms,
        grace_ms,
        pushes,
        carried,
        gnorm,
        compute_secs,
        sync_start,
    })
}

/// Expose a committed round: BCAST its fragment, log it, append its tape
/// record, and write the periodic (non-per-round) checkpoint. A per-round
/// checkpoint must already be durable when this is called.
async fn publish_round(
    cfg: &Config,
    st: &GlobalState,
    registry: &Registry,
    last_sync_secs: &mut f64,
    committed: CommittedRound,
    checkpoint_ms: Option<u64>,
    budget_cutoff: &BudgetCutoff,
) -> Result<()> {
    let CommittedRound {
        t,
        p,
        base_version,
        attempt,
        started,
        expected_members,
        quorum_size,
        quorum_ms,
        grace_ms,
        pushes,
        carried,
        gnorm,
        compute_secs,
        sync_start,
    } = committed;
    let mut responders: Vec<Member> = pushes.keys().copied().collect();
    responders.sort_unstable();
    // A disconnected learner can catch up after reconnecting. Deterministic
    // local encoding/allocation failures must terminate the coordinator
    // instead of silently committing a cut that was never broadcastable.
    // The terminal round is the sole exception: finalize_learners publishes
    // its exact f32 cut immediately after the scheduler reaches quiescence.
    if t < st.schedule().last_step(cfg.total_steps) {
        let dtype = bulk_dtype(st.wire_dtype);
        // bf16 downlink: encode this version once (advancing the shared
        // anchor) and send the same bytes to every learner.
        let published = match &st.downlink {
            Some(anchors) => {
                // The last version of each fragment is also its FINAL version.
                // Islands key stored fragments by (fragment, version) and
                // refuse a second copy with other bytes, so that broadcast
                // must carry the exact f32 masters (FULL), not a bf16 delta.
                let last_version = st.schedule().next_merge_of_fragment(t) > cfg.total_steps;
                let mut anchors = anchors.lock().unwrap();
                Some(if last_version {
                    anchors.publish_full(p, st.versions[p], &st.params[p])
                } else {
                    anchors.publish(p, st.versions[p], &st.params[p])?
                })
            }
            None => None,
        };
        for group in current_groups(registry) {
            let sent = match &published {
                Some(encoded) => {
                    send_tensor_until_cutoff(
                        &group,
                        MSG_BCAST_FRAGMENT,
                        &encoded.prefix,
                        encoded.dtype,
                        &encoded.values,
                        budget_cutoff,
                    )
                    .await
                }
                None => {
                    send_fragment_until_cutoff(st, &group, p, MSG_BCAST_FRAGMENT, dtype, budget_cutoff)
                        .await
                }
            };
            match sent
            {
                Ok(true) => {}
                Ok(false) => break,
                Err(error) if error.downcast_ref::<OutboundStreamClosed>().is_some() => warn!(
                    learner_id = group.member.learner_id,
                    generation = group.member.generation,
                    fragment = p,
                    %error,
                    "updated fragment broadcast skipped for closed learner stream"
                ),
                Err(error) => {
                    return Err(error.context(format!("cannot broadcast updated fragment {p}")));
                }
            }
        }
    }
    *last_sync_secs = compute_secs + sync_start.elapsed().as_secs_f64();
    let sync_ms = (*last_sync_secs * 1000.0).round() as u64;
    let ms = started.elapsed().as_millis() as u64;
    info!(
        step = t,
        fragment = p,
        responders = ?responders,
        gnorm = format!("{gnorm:.4}"),
        ms,
        "outer step"
    );
    if let Some(tape) = &cfg.event_tape {
        // Records land in strict t order, matching optimizer/version commits.
        append_tape(
            tape,
            t,
            p,
            base_version,
            attempt,
            &expected_members,
            quorum_size,
            quorum_ms,
            grace_ms,
            sync_ms,
            checkpoint_ms,
            &pushes,
            &carried,
            gnorm,
            ms,
            &st.layout_fingerprint,
            cfg.policy_sweep_fragments,
        )?;
    }
    // Consistent cut: this round is fully applied and every other in-flight
    // round is still gathering (it has not touched state). Non-terminal cuts
    // have also been broadcast; the terminal cut is delivered by FINAL.
    // A crash-resume loses those gathers; their fragments simply merge on
    // a later cycle, which the quorum design already tolerates.
    // Policy sweeps persist only before broadcast (see persist_before_broadcast).
    if cfg.policy_sweep_fragments.is_none() && !checkpoint_before_broadcast(cfg) {
        if let Some(path) = &cfg.checkpoint_path {
            if cfg.checkpoint_every > 0 && st.schedule().round_number(t) % cfg.checkpoint_every == 0 {
                st.save_checkpoint(path)?;
                info!(step = t, path = %path.display(), "checkpoint written");
            }
        }
    }
    Ok(())
}

/// A committed round whose per-round checkpoint is being written by a
/// blocking task (`--async-checkpoint`). Its fragment stays busy and its
/// BCAST is withheld until the write is durable, so no learner can observe
/// a version a crash could lose.
struct PendingPublish {
    committed: CommittedRound,
    write_started: Instant,
    write: tokio::task::JoinHandle<Result<()>>,
}

/// Snapshot the just-committed state and start persisting it off the
/// scheduler thread.
fn start_async_checkpoint(
    cfg: &Config,
    st: &GlobalState,
    committed: CommittedRound,
) -> Result<PendingPublish> {
    let path = cfg
        .checkpoint_path
        .clone()
        .context("--async-checkpoint requires --checkpoint-path")?;
    let write_started = Instant::now();
    // Shares fragment buffers with `st`; no model-sized copy is made here.
    let snapshot = st.checkpoint_snapshot()?;
    let write = tokio::task::spawn_blocking(move || snapshot.save(&path));
    Ok(PendingPublish {
        committed,
        write_started,
        write,
    })
}

/// Wait for a pending per-round checkpoint, then publish its round. A failed
/// write fails the coordinator before the round becomes visible; the
/// previous checkpoint stays the resume point.
async fn settle_pending_publish(
    cfg: &Config,
    st: &GlobalState,
    registry: &Registry,
    last_sync_secs: &mut f64,
    pending: PendingPublish,
    budget_cutoff: &BudgetCutoff,
    busy_fragments: &mut HashSet<usize>,
) -> Result<()> {
    let PendingPublish {
        committed,
        write_started,
        write,
    } = pending;
    finish_pending_publish(
        cfg,
        st,
        registry,
        last_sync_secs,
        committed,
        write_started,
        write.await,
        budget_cutoff,
        busy_fragments,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn finish_pending_publish(
    cfg: &Config,
    st: &GlobalState,
    registry: &Registry,
    last_sync_secs: &mut f64,
    committed: CommittedRound,
    write_started: Instant,
    written: std::result::Result<Result<()>, tokio::task::JoinError>,
    budget_cutoff: &BudgetCutoff,
    busy_fragments: &mut HashSet<usize>,
) -> Result<()> {
    let (t, p) = (committed.t, committed.p);
    written
        .context("async checkpoint task panicked")?
        .with_context(|| format!("async checkpoint for step {t} failed; step {t} not broadcast"))?;
    let checkpoint_ms = Some(write_started.elapsed().as_millis() as u64);
    if let Some(path) = cfg.checkpoint_path.as_ref() {
        info!(step = t, path = %path.display(), "checkpoint committed");
    }
    publish_round(cfg, st, registry, last_sync_secs, committed, checkpoint_ms, budget_cutoff).await?;
    if !busy_fragments.remove(&p) {
        bail!("fragment {p} lost its busy ownership at publish");
    }
    Ok(())
}

#[cfg(test)]
async fn complete_round(
    cfg: &Config,
    st: &mut GlobalState,
    registry: &Registry,
    last_sync_secs: &mut f64,
    mut round: Round,
) -> Result<()> {
    let prepared = prepare_round_compute(cfg, st, &mut round)?;
    let compute_start = Instant::now();
    let merge = prepared.compute().await?;
    let completed = ComputedRound {
        round,
        merge,
        compute_secs: compute_start.elapsed().as_secs_f64(),
    };
    let cutoff = BudgetCutoff::new(false, Duration::ZERO);
    commit_round(cfg, st, registry, last_sync_secs, completed, &cutoff).await
}

/// Test twin of `complete_round` through the `--async-checkpoint` path.
#[cfg(test)]
async fn complete_round_async(
    cfg: &Config,
    st: &mut GlobalState,
    registry: &Registry,
    last_sync_secs: &mut f64,
    mut round: Round,
) -> Result<()> {
    let prepared = prepare_round_compute(cfg, st, &mut round)?;
    let compute_start = Instant::now();
    let merge = prepared.compute().await?;
    let completed = ComputedRound {
        round,
        merge,
        compute_secs: compute_start.elapsed().as_secs_f64(),
    };
    let cutoff = BudgetCutoff::new(false, Duration::ZERO);
    let mut busy = HashSet::from([completed.round.p]);
    let committed = apply_round_commit(cfg, st, completed)?;
    if !persist_before_broadcast(cfg, committed.t) {
        return publish_round(cfg, st, registry, last_sync_secs, committed, None, &cutoff).await;
    }
    let pending = start_async_checkpoint(cfg, st, committed)?;
    settle_pending_publish(cfg, st, registry, last_sync_secs, pending, &cutoff, &mut busy).await
}

fn new_state_for(group: &Arc<Group>, cfg: &Config) -> Result<GlobalState> {
    if let Some(fragments) = cfg.policy_sweep_fragments {
        // Masters and every BCAST/FINAL stay f32; only push deltas may be bf16.
        if bulk_dtype(group.dtype) != DTYPE_F32 {
            bail!("--policy-sweep-fragments requires an FP32 HELLO session (optionally with bf16/int8 pushes)");
        }
        if group.layout.fragments.len() != fragments as usize {
            bail!(
                "--policy-sweep-fragments declares {fragments}, but decoded layout has {} fragments",
                group.layout.fragments.len()
            );
        }
    }
    let mut st = GlobalState::new_with_iso_backend(
        group.layout.clone(),
        cfg.outer_lr,
        cfg.outer_momentum,
        group.dtype,
        group.layout_fingerprint,
        &cfg.iso_backend,
    )?;
    st.policy_sweep_fragments = cfg.policy_sweep_fragments;
    st.session_contract_hash = Some(group.session_contract_hash);
    st.set_shard(cfg.shard)?;
    match (bcast_delta(group.dtype), cfg.bcast_delta_full_every) {
        (true, Some(full_every)) => {
            st.downlink = Some(Mutex::new(crate::downlink::DownlinkAnchors::new(
                group.layout.fragments.len(),
                full_every,
            )?));
        }
        (true, None) => bail!("session asks for bf16 downlink deltas, but --bcast-delta-full-every is unset"),
        (false, Some(_)) => bail!("--bcast-delta-full-every is set, but the session has no bf16 downlink deltas"),
        (false, None) => {}
    }
    if cfg.delta_correction {
        st.delta_correction = Some(crate::merge::Heloco::default());
    }
    st.set_delayed_nesterov_period(cfg.outer_delayed_nesterov_period)?;
    info!(
        iso_backend = %st.iso_backend_kind(),
        "initialized global state"
    );
    Ok(st)
}

/// Apply lease expiry: expired members leave the pool, stop counting toward
/// any round they have not answered, and are told to re-JOIN.
/// Returns the expired members.
async fn expire_member_leases(
    registry: &Registry,
    inflight: &mut [Round],
    step_rates: &mut StepRates,
) -> Vec<Member> {
    let expired = registry.lock().unwrap().expire_leases(Instant::now());
    let members = expired.iter().map(|(member, _)| *member).collect();
    for (member, group) in expired {
        let rounds = inflight
            .iter_mut()
            .map(|r| r.forget_unanswered(member))
            .filter(|&forgot| forgot)
            .count();
        step_rates.remove(member);
        warn!(
            learner_id = member.learner_id,
            generation = member.generation,
            rounds,
            "island lease expired; removed from membership"
        );
        if let Some(group) = group {
            let message = bytes::Bytes::from_static(LEASE_EXPIRED_MESSAGE.as_bytes());
            let _ = group.send_small(MSG_ERROR, message).await;
        }
    }
    members
}

/// MSG_ERROR text of a lease expiry; the Python client redials with a new
/// generation on it (`yeto/protocol.py` LEASE_EXPIRED_PREFIX).
const LEASE_EXPIRED_MESSAGE: &str = "island lease expired: re-JOIN required";

// --- peer recovery wiring (state machine in recovery.rs) -------------------

/// Current members as donor candidates with their unanswered pull count.
fn recovery_candidates(registry: &Registry, inflight: &[Round]) -> Vec<recovery::Candidate> {
    let members: Vec<Member> = registry.lock().unwrap().current.values().copied().collect();
    members
        .into_iter()
        .map(|member| recovery::Candidate {
            member,
            outstanding: inflight
                .iter()
                .filter(|r| r.expected_members.contains(&member) && !r.pushes.contains_key(&member))
                .count(),
        })
        .collect()
}

/// Send recovery control messages. A target that is gone is skipped: its
/// Disconnected event moves the recovery on.
async fn send_recovery(registry: &Registry, out: Vec<recovery::Outbound>) {
    for msg in out {
        let group = registry.lock().unwrap().groups.get(&msg.to).cloned();
        let sent = match group {
            Some(group) => group.send_small(msg.msg_type, msg.payload.into()).await,
            None => Err(anyhow::anyhow!("connection is gone")),
        };
        if let Err(error) = sent {
            warn!(learner_id = msg.to.learner_id, generation = msg.to.generation, msg_type = msg.msg_type, "recovery message not sent: {error:#}");
        }
    }
}

/// Outside the training loop (init, budget cutoff, finalization) there is
/// no round clock to recover against: the island is put on the roster at
/// once with the degraded source, and the phase's own catch-up follows.
async fn admit_recovering_outside_training(registry: &Registry, group: &Arc<Group>, phase: &str) {
    if !registry.lock().unwrap().promote_recovered(group.member) {
        return;
    }
    warn!(
        learner_id = group.member.learner_id,
        generation = group.member.generation,
        "peer recovery is not available during {phase}; degraded path"
    );
    let reason = format!("syncer is in {phase}, not training");
    let out = recovery::degraded_outside_training(group.member, group.layout.fragments.len(), &reason);
    send_recovery(registry, out).await;
}

/// Rebind the unanswered pulls of existing rounds to `group` (a reconnected
/// or recovered generation). False when the budget cutoff closed the gate.
async fn repull_rounds_for(
    group: &Arc<Group>,
    registry: &Registry,
    inflight: &mut [Round],
    fixed_roster: bool,
    budget_cutoff: &BudgetCutoff,
) -> bool {
    let repulls = if fixed_roster {
        fixed_roster_repulls(inflight, group.member)
    } else {
        rebind_current_unanswered_rounds(registry, inflight, group.member)
    };
    for repull in repulls {
        if !send_small_until_cutoff(group, MSG_PULL_REQ, repull.pull, budget_cutoff).await {
            return false;
        }
        info!(
            learner_id = group.member.learner_id,
            old_generation = repull.old_generation,
            generation = group.member.generation,
            step = repull.t,
            fragment = repull.p,
            attempt = repull.attempt,
            base_version = repull.base_version,
            "rebound outstanding pull to reconnected learner generation"
        );
    }
    true
}

fn ignore_recovery_message(member: Member, phase: &str) {
    warn!(
        learner_id = member.learner_id,
        generation = member.generation,
        "recovery message during {phase} dropped (no recovery in progress)"
    );
}

fn append_recovery_record(path: Option<&std::path::Path>, record: &recovery::RecoveryRecord) {
    use std::io::Write;

    let Some(path) = path else { return };
    let line = format!("{}\n", record.to_json());
    let result = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .and_then(|mut file| file.write_all(line.as_bytes()));
    if let Err(error) = result {
        warn!("event tape write failed: {error}");
    }
}

fn current_groups(registry: &Registry) -> Vec<Arc<Group>> {
    let registry = registry.lock().unwrap();
    let mut groups: Vec<_> = registry
        .current
        .values()
        .filter_map(|member| registry.groups.get(member).cloned())
        .collect();
    groups.sort_unstable_by_key(|group| group.member);
    groups
}

fn is_current_member(registry: &Registry, member: Member) -> bool {
    registry.lock().unwrap().current.get(&member.learner_id) == Some(&member)
}

async fn broadcast_all_fragments(
    st: &GlobalState,
    groups: &[Arc<Group>],
    budget_cutoff: &BudgetCutoff,
) {
    // Fan each fragment out to every learner before moving to the next one.
    // Each learner has independent socket-writer queues, so this keeps all
    // links busy concurrently.  Iterating learner-by-learner here serializes
    // one full model transfer per learner (tens of GiB for large models).
    // The current streaming encoder keeps peak host memory bounded while the
    // per-learner socket queues provide link concurrency.
    for p in st.owned_fragments() {
        for group in groups {
            match send_fragment_until_cutoff(
                st,
                group,
                p,
                MSG_BCAST_FRAGMENT,
                bulk_dtype(st.wire_dtype),
                budget_cutoff,
            )
            .await
            {
                Ok(true) => {}
                Ok(false) => return,
                Err(error) => warn!(
                    learner_id = group.member.learner_id,
                    generation = group.member.generation,
                    fragment = p,
                    %error,
                    "initial authoritative cut send failed"
                ),
            }
        }
    }
}

/// Send every authoritative f32 fragment, then publish the version manifest
/// on the control stream. Data/control streams may reorder; learners cache
/// terminal fragments and use the manifest to decide when the complete cut
/// is locally available. `manifest_only`: the manifest alone (islands that
/// acknowledge without holding the final parameters).
async fn send_final_cut(st: &GlobalState, group: &Arc<Group>, manifest_only: bool) -> Result<()> {
    if !manifest_only {
        for p in st.owned_fragments() {
            send_fragment(st, group, p, MSG_FINAL_FRAGMENT, DTYPE_F32).await?;
        }
    }
    group
        .send_small(
            MSG_FINAL_MANIFEST,
            bytes::Bytes::from(encode_final_manifest(st.global_step, &st.versions)),
        )
        .await
}

#[derive(Debug, Eq, PartialEq)]
enum FinalAckDisposition {
    Accepted,
    AlreadyAcknowledged,
    UnexpectedLearner,
    IneligibleGeneration,
    WrongStep,
}

fn record_final_ack(
    expected: &HashSet<u32>,
    eligible: &HashMap<u32, HashSet<Member>>,
    acknowledged: &mut HashMap<u32, Member>,
    member: Member,
    global_step: u64,
    expected_step: u64,
) -> FinalAckDisposition {
    let learner_id = member.learner_id;
    if !expected.contains(&learner_id) {
        return FinalAckDisposition::UnexpectedLearner;
    }
    if !eligible
        .get(&learner_id)
        .is_some_and(|members| members.contains(&member))
    {
        return FinalAckDisposition::IneligibleGeneration;
    }
    if global_step != expected_step {
        return FinalAckDisposition::WrongStep;
    }
    if acknowledged.contains_key(&learner_id) {
        return FinalAckDisposition::AlreadyAcknowledged;
    }
    acknowledged.insert(learner_id, member);
    FinalAckDisposition::Accepted
}

/// Hold the coordinator alive until every valid learner group that was live
/// at the final cut confirms it received and applied that exact cut. A target
/// that disconnects may reconnect within the bounded wait; learners already
/// abandoned before the cut are not members. SHUTDOWN is never sent as a
/// legacy escape hatch because that would let an old learner save a locally
/// blended artifact.
async fn finalize_learners(
    cfg: &Config,
    st: &GlobalState,
    events: &mut mpsc::Receiver<Event>,
    registry: &Registry,
    expected: &HashSet<u32>,
) -> Result<()> {
    if expected.is_empty() {
        bail!("cannot finalize: no live learner groups at the final cut");
    }
    // Dynamic membership: the final cut goes to every island that is (or
    // becomes) connected, the wait is bounded by the final grace window and
    // an island that left or never answered does not fail the run.
    let final_grace = cfg.membership.final_grace;
    let mut expected = expected.clone();
    let expected = &mut expected;
    let mut eligible: HashMap<u32, HashSet<Member>> = HashMap::new();
    for group in current_groups(registry) {
        if expected.contains(&group.member.learner_id) {
            match send_final_cut(st, &group, cfg.final_cut_manifest_only).await {
                Ok(()) => {
                    eligible
                        .entry(group.member.learner_id)
                        .or_default()
                        .insert(group.member);
                }
                Err(error) => {
                    warn!(
                        learner_id = group.member.learner_id,
                        generation = group.member.generation,
                        %error,
                        "final cut send failed; waiting for reconnect"
                    );
                }
            }
        }
    }

    let wait = final_grace.unwrap_or(Duration::from_secs(cfg.final_ack_timeout_s));
    let deadline = Instant::now() + wait;
    let mut acknowledged: HashMap<u32, Member> = HashMap::new();
    // With dynamic membership, an island whose lease runs out during
    // finalization is absent: the final cut no longer waits for its ACK.
    let mut absence = FinalAbsence::new(cfg, expected, registry);
    while !absence.pending(expected, &acknowledged).is_empty() {
        let now = Instant::now();
        let wake = absence.next_deadline(registry).map_or(deadline, |at| at.min(deadline));
        let event = match tokio::time::timeout(wake.saturating_duration_since(now), events.recv()).await {
            Ok(event) => event.context("event channel closed during finalization")?,
            Err(_) if Instant::now() < deadline => {
                for learner_id in absence.drop_expired(registry, expected, Instant::now()) {
                    warn!(learner_id, "lease expired during finalization; final ACK no longer required");
                }
                if absence.pending(expected, &acknowledged).is_empty() && acknowledged.is_empty() {
                    bail!("cannot finalize: every learner's lease expired before it acknowledged the final cut");
                }
                continue;
            }
            Err(_) => {
                let mut missing: Vec<u32> = absence.pending(expected, &acknowledged).into_iter().collect();
                missing.sort_unstable();
                if final_grace.is_some() {
                    warn!(?missing, "final grace window ended; finishing without these islands");
                    let acked: Vec<u32> = acknowledged.keys().copied().collect();
                    append_pool_record(
                        cfg.event_tape.as_deref(),
                        "final_grace_expired",
                        cfg.syncer_epoch,
                        &format!(
                            "\"final_step\":{},\"acknowledged\":{},\"missing\":{}",
                            st.global_step,
                            json_ids(&acked),
                            json_ids(&missing)
                        ),
                    );
                    break;
                }
                bail!(
                    "finalization timed out after {}s waiting for learner acknowledgements: {:?}",
                    cfg.final_ack_timeout_s,
                    missing
                );
            }
        };
        absence.observe(registry, &event);
        match event {
            Event::Fatal { metric, message } => {
                append_strict_failure(cfg.event_tape.as_deref(), metric, &message);
                bail!("RL strict failure {metric}: {message}");
            }
            Event::FinalAck {
                member,
                global_step,
            } => {
                let learner_id = member.learner_id;
                match record_final_ack(
                    &*expected,
                    &eligible,
                    &mut acknowledged,
                    member,
                    global_step,
                    st.global_step,
                ) {
                    FinalAckDisposition::Accepted => info!(
                        learner_id,
                        generation = member.generation,
                        step = global_step,
                        remaining = expected.len() - acknowledged.len(),
                        "learner finalized authoritative parameters"
                    ),
                    FinalAckDisposition::AlreadyAcknowledged => {}
                    FinalAckDisposition::UnexpectedLearner => warn!(
                        learner_id,
                        generation = member.generation,
                        "final acknowledgement from unknown learner dropped"
                    ),
                    FinalAckDisposition::IneligibleGeneration => warn!(
                        learner_id,
                        generation = member.generation,
                        "final acknowledgement from generation not sent the frozen cut dropped"
                    ),
                    FinalAckDisposition::WrongStep => warn!(
                        learner_id,
                        generation = member.generation,
                        received = global_step,
                        expected = st.global_step,
                        "final acknowledgement for wrong manifest dropped"
                    ),
                }
            }
            Event::Hello { group } => {
                admit_recovering_outside_training(registry, &group, "finalization").await;
                let member = group.member;
                if final_grace.is_some() && is_current_member(registry, member) {
                    // A late (re-)JOIN still gets the final cut and may
                    // acknowledge it within the grace window.
                    expected.insert(member.learner_id);
                }
                if expected.contains(&member.learner_id)
                    && !acknowledged.contains_key(&member.learner_id)
                    && is_current_member(registry, member)
                {
                    match send_final_cut(st, &group, cfg.final_cut_manifest_only).await {
                        Ok(()) => {
                            eligible
                                .entry(member.learner_id)
                                .or_default()
                                .insert(member);
                        }
                        Err(error) => {
                            warn!(
                                learner_id = member.learner_id,
                                generation = member.generation,
                                %error,
                                "final cut resend failed"
                            );
                        }
                    }
                } else {
                    warn!(
                        learner_id = member.learner_id,
                        generation = member.generation,
                        "unexpected or superseded learner during finalization"
                    );
                }
            }
            Event::Disconnected { member } => {
                if final_grace.is_some()
                    && !acknowledged.contains_key(&member.learner_id)
                    && !current_groups(registry)
                        .iter()
                        .any(|group| group.member.learner_id == member.learner_id)
                {
                    // The island left: stop waiting for it unless it re-JOINs.
                    expected.remove(&member.learner_id);
                    info!(
                        learner_id = member.learner_id,
                        "island left during the final grace window"
                    );
                    continue;
                }
                warn!(
                    learner_id = member.learner_id,
                    generation = member.generation,
                    "learner disconnected during finalization; waiting for reconnect"
                );
            }
            Event::Push { .. }
            | Event::Heartbeat { .. }
            | Event::Leave { .. }
            | Event::Init { .. }
            | Event::InitDigest { .. }
            | Event::Recovery { .. }
            | Event::BudgetDone { .. } => {
                // The authoritative cut is frozen; late training traffic is
                // intentionally ignored while learners finalize.
            }
        }
    }

    let acknowledged_groups: Vec<Arc<Group>> = {
        let registry = registry.lock().unwrap();
        acknowledged
            .values()
            .filter_map(|member| registry.groups.get(member).cloned())
            .collect()
    };
    for group in acknowledged_groups {
        let _ = group.send_small(MSG_SHUTDOWN, bytes::Bytes::new()).await;
    }
    info!(
        learners = acknowledged.len(),
        absent = absence.dropped.len(),
        "all learners acknowledged final cut"
    );
    Ok(())
}

/// Which expected learners the final cut still waits for. With static
/// membership (or no lease) it waits for every expected learner until the
/// final ACK timeout, as before. With a membership lease, a learner is dropped
/// from the wait when its lease expires, or when it stays disconnected for one
/// lease without a new generation saying HELLO.
struct FinalAbsence {
    lease: Option<Duration>,
    disconnected_since: HashMap<u32, Instant>,
    dropped: HashSet<u32>,
}

impl FinalAbsence {
    fn new(cfg: &Config, expected: &HashSet<u32>, registry: &Registry) -> Self {
        let lease = if cfg.membership.is_static() { None } else { cfg.membership.lease };
        let mut disconnected_since = HashMap::new();
        if lease.is_some() {
            let now = Instant::now();
            let registry = registry.lock().unwrap();
            for id in expected.iter().filter(|id| !registry.current.contains_key(*id)) {
                disconnected_since.insert(*id, now);
            }
        }
        Self { lease, disconnected_since, dropped: HashSet::new() }
    }

    /// `expected` is the finalization's live set (a late JOIN adds to it).
    fn pending(&self, expected: &HashSet<u32>, acknowledged: &HashMap<u32, Member>) -> HashSet<u32> {
        expected
            .iter()
            .copied()
            .filter(|id| !acknowledged.contains_key(id) && !self.dropped.contains(id))
            .collect()
    }

    /// Renew leases on traffic and track disconnects of the current generation.
    fn observe(&mut self, registry: &Registry, event: &Event) {
        if self.lease.is_none() {
            return;
        }
        let now = Instant::now();
        match event {
            Event::Heartbeat { member, .. } | Event::Push { member, .. } | Event::FinalAck { member, .. } => {
                registry.lock().unwrap().membership.renew(*member, now);
            }
            Event::Hello { group } => {
                self.disconnected_since.remove(&group.member.learner_id);
            }
            Event::Disconnected { member } => {
                let current = registry.lock().unwrap().current.get(&member.learner_id).copied();
                if current.is_none() || current == Some(*member) {
                    self.disconnected_since.entry(member.learner_id).or_insert(now);
                }
            }
            _ => {}
        }
    }

    fn next_deadline(&self, registry: &Registry) -> Option<Instant> {
        let lease = self.lease?;
        let expiry = registry.lock().unwrap().membership.next_expiry();
        self.disconnected_since.values().map(|since| *since + lease).chain(expiry).min()
    }

    /// Drop every pending learner whose lease expired; returns them in id order.
    fn drop_expired(&mut self, registry: &Registry, expected: &HashSet<u32>, now: Instant) -> Vec<u32> {
        let Some(lease) = self.lease else { return Vec::new() };
        let mut gone: Vec<u32> = registry
            .lock()
            .unwrap()
            .expire_leases(now)
            .into_iter()
            .map(|(member, _)| member.learner_id)
            .collect();
        gone.extend(
            self.disconnected_since
                .iter()
                .filter(|(_, since)| now.saturating_duration_since(**since) >= lease)
                .map(|(id, _)| *id),
        );
        gone.retain(|id| expected.contains(id) && self.dropped.insert(*id));
        for id in &gone {
            self.disconnected_since.remove(id);
        }
        gone.sort_unstable();
        gone
    }
}

/// What a finished catch-up still owes its learner.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct CatchUpTail {
    /// HELLO path: record the catch-up and repull the learner's open rounds.
    rejoin: bool,
    /// Disconnect path: move the captured generation's unanswered slots to
    /// the already-current replacement.
    rebind: bool,
}

impl CatchUpTail {
    const REJOIN: Self = Self { rejoin: true, rebind: false };
    const REBIND: Self = Self { rejoin: false, rebind: true };
}

/// One rejoined learner's catch-up: the current value of every owned fragment.
struct CatchUp<G> {
    member: Member,
    group: G,
    remaining: std::collections::VecDeque<usize>,
    tail: CatchUpTail,
}

enum CatchUpStep<G> {
    /// Send this fragment's current value to the learner.
    Send(Member, G, usize),
    /// Every fragment went out (or the stream failed); run the tail.
    Done(CatchUp<G>),
}

/// Rejoin catch-ups owned by the scheduler.
///
/// A catch-up is the whole model (about 110 GB of f32 for a 27B model). The
/// scheduler sends one fragment per pass and then serves events and round
/// deadlines again, so a catch-up delays a grace or quorum deadline by at
/// most one fragment transfer instead of by the whole model (s27-gh-a: a
/// 300 s grace window committed after 14 min). A learner does not join new
/// rounds before its catch-up is done: its pushes would be against stale
/// bases.
struct CatchUps<G> {
    queue: std::collections::VecDeque<CatchUp<G>>,
}

impl<G: Clone> CatchUps<G> {
    fn new() -> Self {
        Self { queue: std::collections::VecDeque::new() }
    }

    fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    fn holds(&self, member: Member) -> bool {
        self.queue.iter().any(|catch_up| catch_up.member == member)
    }

    /// Queue a catch-up; a second request for the same generation only adds
    /// its tail, because the fragments already sent are current.
    fn enqueue(
        &mut self,
        member: Member,
        group: G,
        fragments: impl IntoIterator<Item = usize>,
        tail: CatchUpTail,
    ) {
        if let Some(existing) = self.queue.iter_mut().find(|catch_up| catch_up.member == member) {
            existing.tail.rejoin |= tail.rejoin;
            existing.tail.rebind |= tail.rebind;
            return;
        }
        self.queue.push_back(CatchUp { member, group, remaining: fragments.into_iter().collect(), tail });
    }

    /// Remove catch-ups whose generation is no longer current.
    fn drop_superseded(&mut self, is_current: impl Fn(Member) -> bool) -> Vec<Member> {
        let mut dropped = Vec::new();
        self.queue.retain(|catch_up| {
            let keep = is_current(catch_up.member);
            if !keep {
                dropped.push(catch_up.member);
            }
            keep
        });
        dropped
    }

    /// Stop sending to `member` (its stream failed); the tail still runs.
    fn abandon(&mut self, member: Member) {
        if let Some(catch_up) = self.queue.iter_mut().find(|catch_up| catch_up.member == member) {
            catch_up.remaining.clear();
        }
    }

    /// The next step, round-robin over the queued learners.
    fn next_step(&mut self) -> Option<CatchUpStep<G>> {
        let mut catch_up = self.queue.pop_front()?;
        match catch_up.remaining.pop_front() {
            Some(p) => {
                let step = CatchUpStep::Send(catch_up.member, catch_up.group.clone(), p);
                self.queue.push_back(catch_up);
                Some(step)
            }
            None => Some(CatchUpStep::Done(catch_up)),
        }
    }
}

async fn broadcast_updated_fragment(st: &GlobalState, registry: &Registry, p: usize) -> Result<()> {
    let dtype = bulk_dtype(st.wire_dtype);
    for group in current_groups(registry) {
        match send_fragment(st, &group, p, MSG_BCAST_FRAGMENT, dtype).await {
            Ok(()) => {}
            Err(error) if error.downcast_ref::<OutboundStreamClosed>().is_some() => warn!(
                learner_id = group.member.learner_id,
                generation = group.member.generation,
                fragment = p,
                %error,
                "updated fragment broadcast skipped for closed learner stream"
            ),
            Err(error) => {
                return Err(error.context(format!("cannot broadcast updated fragment {p}")));
            }
        }
    }
    Ok(())
}

async fn send_fragment(
    st: &GlobalState,
    group: &Arc<Group>,
    p: usize,
    msg_type: u8,
    dtype: u8,
) -> Result<()> {
    if let Some(encoded) = recovery_bcast(st, p, msg_type) {
        return group
            .send_tensor_large(msg_type, &encoded.prefix, encoded.dtype, &encoded.values)
            .await;
    }
    let mut prefix = [0u8; 12];
    prefix[..4].copy_from_slice(&(p as u32).to_le_bytes());
    prefix[4..].copy_from_slice(&st.versions[p].to_le_bytes());
    group
        .send_tensor_large(msg_type, &prefix, dtype, &st.params[p])
        .await
}

/// One JSONL record per merge: the event tape.
/// One island-pool record on the event tape: `{"kind":..,"syncer_epoch":..,
/// <fields>,"time_unix":..}`. Written only with dynamic membership; the
/// kinds (`pool_join`, `pool_leave`, `delta_carried_over`, `delta_rejected`,
/// `outer_step`, `final_grace_expired`) are what the merge-metrics folder
/// and the launcher's pool reader consume.
fn append_pool_record(path: Option<&std::path::Path>, kind: &str, syncer_epoch: u64, fields: &str) {
    use std::io::Write;

    let Some(path) = path else {
        return;
    };
    let sep = if fields.is_empty() { "" } else { "," };
    let line = format!(
        "{{\"kind\":\"{kind}\",\"syncer_epoch\":{syncer_epoch}{sep}{fields},\"time_unix\":{:?}}}\n",
        unix_now()
    );
    let result = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .and_then(|mut file| file.write_all(line.as_bytes()));
    if let Err(error) = result {
        warn!("event tape write failed: {error}");
    }
}

/// The `outer_step` pool record of one merged round: who arrived, each
/// push's normalized merge weight (`[island, base_version, weight]`), and
/// which expected islands are absent. `timed_out` marks a soft-deadline
/// merge below quorum.
fn append_outer_step_record(
    cfg: &Config,
    round: &Round,
    ordered_pushes: &[(&Member, &Push)],
    weights: &[f64],
) {
    let capacity = |members: &mut dyn Iterator<Item = &Member>| -> f64 {
        members.map(|member| cfg.membership.capacity(member.learner_id)).sum()
    };
    let cap_arrived = capacity(&mut round.pushes.keys());
    let cap_total = capacity(&mut round.expected_members.iter());
    let total: f64 = weights.iter().sum();
    let entry = |(member, push): &(&Member, &Push), weight: f64| {
        format!("[{},{},{:?}]", member.learner_id, push.base_version, weight)
    };
    let raw: Vec<String> = ordered_pushes.iter().zip(weights).map(|(p, w)| entry(p, *w)).collect();
    let normalized: Vec<String> = ordered_pushes
        .iter()
        .zip(weights)
        .map(|(p, w)| entry(p, if total > 0.0 { w / total } else { 0.0 }))
        .collect();
    let absent: Vec<u32> = round
        .expected_members
        .iter()
        .filter(|member| !round.pushes.contains_key(member))
        .map(|member| member.learner_id)
        .collect();
    append_pool_record(
        cfg.event_tape.as_deref(),
        "outer_step",
        cfg.syncer_epoch,
        &format!(
            "\"step\":{},\"fragment\":{},\"base_version\":{},\"arrived\":{},\"cap_arrived\":{:?},\"cap_total\":{:?},\"timed_out\":{},\"raw_weights\":[{}],\"weights\":[{}],\"absent\":{},\"step_seconds\":{:?}",
            round.t,
            round.p,
            round.base_version,
            round.pushes.len(),
            cap_arrived,
            cap_total,
            !round.quorum_reached(),
            raw.join(","),
            normalized.join(","),
            json_ids(&absent),
            round.started.elapsed().as_secs_f64(),
        ),
    );
}

/// Schema of `status.json`, written next to the event tape with dynamic
/// membership (read by `yeto.rl.engine.island_status`).
const POOL_STATUS_SCHEMA: &str = "yeto.syncer.elastic-status/v1";

/// Atomically rewrite `status.json` beside the event tape: run state, syncer
/// epoch, committed step and the live islands. Best effort (a write failure
/// is logged, never fatal).
fn write_pool_status(cfg: &Config, registry: &Registry, outer_version: u64, state: &str) {
    let Some(tape) = cfg.event_tape.as_deref() else {
        return;
    };
    let (syncer_epoch, membership_epoch, live) = {
        let registry = registry.lock().unwrap();
        let membership = &registry.membership;
        (membership.syncer_epoch(), membership.epoch, membership.live_status(Instant::now()))
    };
    let cap_total: f64 = live.iter().map(|(_, capacity, _)| capacity).sum();
    let islands: Vec<String> = live
        .iter()
        .map(|(id, capacity, lease)| {
            let lease = lease.map_or_else(|| "null".to_string(), |d| format!("{:?}", d.as_secs_f64()));
            format!("\"{id}\":{{\"capacity\":{capacity:?},\"lease_remaining_s\":{lease}}}")
        })
        .collect();
    let body = format!(
        "{{\"schema\":\"{POOL_STATUS_SCHEMA}\",\"state\":\"{state}\",\"syncer_epoch\":{syncer_epoch},\"membership_epoch\":{membership_epoch},\"outer_version\":{outer_version},\"cap_total\":{cap_total:?},\"islands\":{{{}}}}}",
        islands.join(",")
    );
    let path = tape.with_file_name("status.json");
    let temporary = tape.with_file_name(".status.json.tmp");
    if let Err(error) = std::fs::write(&temporary, body).and_then(|()| std::fs::rename(&temporary, &path)) {
        warn!(path = %path.display(), "status write failed: {error}");
    }
}

fn append_strict_failure(path: Option<&std::path::Path>, metric: &str, message: &str) {
    use std::io::Write;

    let Some(path) = path else {
        return;
    };
    let escaped = message.replace('\\', "\\\\").replace('"', "\\\"");
    let line = format!(
        "{{\"event\":\"rl_strict_failure\",\"metric\":\"{metric}\",\"value\":1,\"error\":\"{escaped}\"}}\n"
    );
    let result = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .and_then(|mut file| file.write_all(line.as_bytes()));
    if let Err(error) = result {
        warn!("event tape write failed: {error}");
    }
}

/// Append an idempotent, fsynced accounting cut after a sweep checkpoint has
/// become authoritative. A crash between the checkpoint rename and the
/// ordinary per-merge tape append can lose that diagnostic merge record; the
/// resume and terminal snapshots make the durable ledger reconcilable without
/// charging local steps or tokens again.
fn append_policy_sweep_ledger_snapshot(
    path: &std::path::Path,
    st: &GlobalState,
    phase: &str,
) -> Result<()> {
    use std::io::Write;

    let fragments = st
        .policy_sweep_fragments
        .context("policy-sweep ledger snapshot requires a sweep profile")?;
    let layout_hash: String = st
        .layout_fingerprint
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    let event_id = format!(
        "policy-sweep-ledger:{layout_hash}:{phase}:{}",
        st.global_step
    );
    let existing = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(error) => return Err(error).with_context(|| format!("read {}", path.display())),
    };
    let needle = format!("\"event_id\":\"{event_id}\"");
    let already_committed = existing.split_inclusive(|byte| *byte == b'\n').any(|line| {
        line.ends_with(b"\n")
            && line
                .windows(needle.len())
                .any(|window| window == needle.as_bytes())
    });
    if already_committed {
        return Ok(());
    }

    let versions = st
        .versions
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let ledger = st
        .ledger
        .iter()
        .map(|(learner_id, entry)| {
            format!(
                "{{\"id\":{learner_id},\"merges\":{},\"steps\":{},\"tokens\":{}}}",
                entry.merges, entry.steps, entry.tokens
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let policy_round = if st.global_step == 0 {
        0
    } else {
        st.global_step.div_ceil(u64::from(fragments))
    };
    let sweep_complete = st.global_step.is_multiple_of(u64::from(fragments));
    let line = format!(
        "{{\"event\":\"policy_sweep_ledger\",\"event_id\":\"{event_id}\",\"phase\":\"{phase}\",\"protocol_version\":{PROTOCOL_VERSION},\"sync/layout_hash\":\"{layout_hash}\",\"global_step\":{},\"policy_round\":{policy_round},\"sweep_fragments\":{fragments},\"sweep_complete\":{sweep_complete},\"versions\":[{versions}],\"ledger\":[{ledger}]}}\n",
        st.global_step
    );
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("open {}", path.display()))?;
    if existing.last().is_some_and(|byte| *byte != b'\n') {
        // Preserve a torn diagnostic tail as an independently skippable line.
        file.write_all(b"\n")?;
    }
    file.write_all(line.as_bytes())?;
    file.flush()?;
    file.sync_all()?;
    Ok(())
}

fn policy_sweep_event_fields(step: u64, fragments: Option<u32>) -> String {
    let Some(fragments) = fragments else {
        return String::new();
    };
    let fragments_u64 = u64::from(fragments);
    let policy_round = step.div_ceil(fragments_u64);
    let sweep_fragment = (step - 1) % fragments_u64;
    let sweep_complete = step.is_multiple_of(fragments_u64);
    format!(
        ",\"policy_round\":{policy_round},\"sweep_fragment\":{sweep_fragment},\"sweep_fragments\":{fragments},\"sweep_complete\":{sweep_complete}"
    )
}

fn append_tape(
    path: &std::path::Path,
    step: u64,
    fragment: usize,
    launch_base_version: u64,
    attempt: u32,
    expected_members: &[Member],
    quorum: usize,
    quorum_ms: Option<u64>,
    grace_ms: Option<u64>,
    sync_ms: u64,
    checkpoint_ms: Option<u64>,
    pushes: &HashMap<Member, Push>,
    carried: &[(Member, Push)],
    gnorm: f64,
    ms: u64,
    layout_fingerprint: &[u8; 32],
    policy_sweep_fragments: Option<u32>,
) -> Result<()> {
    use std::io::Write;
    let mut responded_members: Vec<Member> = pushes.keys().copied().collect();
    responded_members.sort_unstable();
    let responded: Vec<u32> = responded_members
        .iter()
        .map(|member| member.learner_id)
        .collect();
    let missed_members: Vec<Member> = expected_members
        .iter()
        .copied()
        .filter(|member| !pushes.contains_key(member))
        .collect();
    let missed_grace: Vec<u32> = missed_members
        .iter()
        .map(|member| member.learner_id)
        .collect();
    let expected_learners: Vec<u32> = expected_members
        .iter()
        .map(|member| member.learner_id)
        .collect();
    let event_weight = |push: &Push| {
        if policy_sweep_fragments.is_some() {
            // The sweep profile is validated as equal-weighted; its event
            // tape must describe the merge that actually happened.
            1.0
        } else {
            crate::merge::learner_weight(push.c_tokens, u64::from(push.c_steps))
        }
    };
    let weight_sum: f64 = pushes
        .values()
        .chain(carried.iter().map(|(_, push)| push))
        .map(event_weight)
        .sum();
    // Late deltas of earlier rounds merged here (async rounds only; the field
    // is absent when there are none).
    let carried_field = if carried.is_empty() {
        String::new()
    } else {
        let entries: Vec<String> = carried
            .iter()
            .map(|(member, p)| {
                let weight = event_weight(p);
                let contribution = if weight_sum > 0.0 { weight / weight_sum } else { 0.0 };
                format!(
                    "{{\"id\":{},\"generation\":{},\"missed_step\":{},\"base_version\":{},\"lag\":{},\"c_steps\":{},\"c_tokens\":{},\"weight\":{},\"contribution\":{}}}",
                    p.learner_id,
                    member.generation,
                    p.global_step,
                    p.base_version,
                    launch_base_version.saturating_sub(p.base_version),
                    p.c_steps,
                    p.c_tokens,
                    weight,
                    contribution
                )
            })
            .collect();
        format!(",\"carried\":[{}]", entries.join(","))
    };
    let mut responders: Vec<String> = pushes
        .iter()
        .map(|(member, p)| {
            let weight = event_weight(p);
            let contribution = if weight_sum > 0.0 { weight / weight_sum } else { 0.0 };
            let staleness = launch_base_version.saturating_sub(p.base_version);
            let accounting_fields = policy_sweep_fragments.map_or_else(String::new, |fragments| {
                let (steps, tokens) = if step.is_multiple_of(u64::from(fragments)) {
                    (p.c_steps, p.c_tokens)
                } else {
                    (0, 0)
                };
                format!(
                    ",\"accounted_c_steps\":{steps},\"accounted_c_tokens\":{tokens}"
                )
            });
            format!(
                "{{\"id\":{},\"generation\":{},\"base_version\":{},\"staleness\":{},\"c_steps\":{},\"c_tokens\":{}{},\"weight\":{},\"contribution\":{},\"recv_seconds\":{},\"arrived_unix\":{:?}}}",
                p.learner_id,
                member.generation,
                p.base_version,
                staleness,
                p.c_steps,
                p.c_tokens,
                accounting_fields,
                weight,
                contribution,
                p.transfer.recv_seconds,
                p.transfer.arrived_unix
            )
        })
        .collect();
    responders.sort();
    let quorum_ms = json_opt_u64(quorum_ms);
    let grace_ms = json_opt_u64(grace_ms);
    let layout_hash: String = layout_fingerprint
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    let merge_seconds = sync_ms as f64 / 1000.0;
    let checkpoint_field = checkpoint_tape_field(checkpoint_ms);
    let sweep_fields = policy_sweep_event_fields(step, policy_sweep_fragments);
    let line = format!(
        "{{\"protocol_version\":{PROTOCOL_VERSION},\"delta_semantics\":\"local_minus_raw_anchor\",\"sync/layout_hash\":\"{layout_hash}\",\"sync/base_version\":{launch_base_version},\"sync/responders\":{},\"sync/quorum\":{quorum},\"sync/rejected_stale_updates\":0,\"sync/merge_seconds\":{merge_seconds},\"sync/global_delta_norm\":{gnorm},\"step\":{step},\"fragment\":{fragment}{sweep_fields},\"launch_base_version\":{launch_base_version},\"attempt\":{attempt},\"gnorm\":{gnorm},\"ms\":{ms},\"quorum\":{quorum},\"expected\":{},\"expected_members\":{},\"responded\":{},\"responded_members\":{},\"missed_grace\":{},\"missed_members\":{},\"quorum_ms\":{quorum_ms},\"grace_ms\":{grace_ms},\"sync_ms\":{sync_ms}{checkpoint_field}{carried_field},\"time_unix\":{:?},\"responders\":[{}]}}\n",
        responded.len(),
        json_ids(&expected_learners),
        json_members(expected_members),
        json_ids(&responded),
        json_members(&responded_members),
        json_ids(&missed_grace),
        json_members(&missed_members),
        unix_now(),
        responders.join(",")
    );
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .and_then(|mut f| f.write_all(line.as_bytes()))
        .with_context(|| format!("append event tape {}", path.display()))?;
    Ok(())
}

/// Optional top-level tape field: wall time of the per-round checkpoint
/// (synchronous save, or snapshot + async write until durable).
fn checkpoint_tape_field(checkpoint_ms: Option<u64>) -> String {
    checkpoint_ms.map_or_else(String::new, |ms| format!(",\"checkpoint_ms\":{ms}"))
}

fn json_ids(ids: &[u32]) -> String {
    let mut ids = ids.to_vec();
    ids.sort_unstable();
    let body = ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
    format!("[{body}]")
}

fn json_members(members: &[Member]) -> String {
    let mut members = members.to_vec();
    members.sort_unstable();
    let body = members
        .iter()
        .map(|member| {
            format!(
                "{{\"id\":{},\"generation\":{}}}",
                member.learner_id, member.generation
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!("[{body}]")
}

/// Wall clock in seconds since the epoch for the event tape (0.0 if the clock
/// is before 1970, so the tape stays valid JSON).
fn unix_now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
}

fn json_opt_u64(value: Option<u64>) -> String {
    value.map_or_else(|| "null".to_string(), |v| v.to_string())
}

fn dump_state(st: &GlobalState, path: &std::path::Path) -> Result<()> {
    use std::io::Write;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(&(st.layout.fragments.len() as u32).to_le_bytes())?;
    for p in &st.params {
        f.write_all(&(p.len() as u64).to_le_bytes())?;
        for v in p.iter() {
            f.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod hello_auth_tests {
    use super::*;

    fn sealed(key: &[u8], body: &[u8]) -> Vec<u8> {
        let mut input = HELLO_MAC_DOMAIN.to_vec();
        input.extend_from_slice(body);
        let mut out = body.to_vec();
        out.extend_from_slice(&crate::hmac::hmac_sha256(key, &input));
        out
    }

    #[test]
    fn bw_probe_body_is_two_bounded_sizes() {
        let mut body = Vec::new();
        body.extend_from_slice(&5u64.to_le_bytes());
        body.extend_from_slice(&7u64.to_le_bytes());
        assert_eq!(parse_bw_probe(&body).unwrap(), (5, 7));
        assert!(parse_bw_probe(&body[..15]).is_err());
        let mut too_big = Vec::new();
        too_big.extend_from_slice(&(BW_PROBE_MAX_BYTES + 1).to_le_bytes());
        too_big.extend_from_slice(&0u64.to_le_bytes());
        assert!(parse_bw_probe(&too_big).is_err());
    }

    #[test]
    fn bw_probe_slots_are_bounded() {
        let slots: Vec<_> = (0..BW_PROBE_MAX_CONNECTIONS).map(|_| BwProbeSlot::acquire()).collect();
        assert!(slots.iter().all(Option::is_some));
        assert!(BwProbeSlot::acquire().is_none());
        drop(slots);
        assert!(BwProbeSlot::acquire().is_some());
    }

    #[test]
    fn open_hello_accepts_the_right_key_only() {
        let body = b"hello body bytes";
        assert_eq!(open_hello(Some(b"k"), &sealed(b"k", body)).unwrap(), body);
        assert!(open_hello(Some(b"k"), &sealed(b"other", body)).is_err());
        assert!(open_hello(Some(b"k"), body).is_err());
        assert!(open_hello(Some(b"k"), b"short").is_err());
        // Unauthenticated mode passes the payload through unchanged.
        assert_eq!(open_hello(None, body).unwrap(), body);
    }

    #[test]
    fn mac_covers_the_compat_group_trailer_and_comes_last() {
        // body = ... num_streams:u16 | b"YCG1" | u32 n | n bytes ; then 32-byte MAC
        let mut rest = 0u16.to_le_bytes().to_vec();
        rest.extend_from_slice(HELLO_COMPAT_MAGIC);
        rest.extend_from_slice(&11u32.to_le_bytes());
        rest.extend_from_slice(b"nvidia-h100");
        let payload = sealed(b"k", &rest);
        assert_eq!(payload.len(), 2 + 8 + 11 + 32);
        let body = open_hello(Some(b"k"), &payload).unwrap();
        assert_eq!(hello_compat_trailer_len(body), 8 + 11);
        // A MAC-less parse of the sealed payload must not find the trailer.
        assert_eq!(hello_compat_trailer_len(&payload), 0);
        // Changing the clear-text group breaks the MAC.
        let mut forged = payload.clone();
        let at = 2 + 8;
        forged[at] = b'a';
        assert!(open_hello(Some(b"k"), &forged).is_err());
    }

    #[test]
    fn pinned_contract_matches_the_python_session_contract() {
        let layout_fp = [7u8; 32];
        let pinned = [9u8; 32];
        let good = pinned_session_contract(&layout_fp, &pinned);
        let spec = |contract| SessionSpec {
            dtype: DTYPE_F32,
            layout: Layout { fragments: Vec::new() },
            layout_fingerprint: layout_fp,
            session_contract_hash: contract,
            syncer_profile_hash: None,
            compat_group: String::new(),
        };
        assert!(check_pinned_contract(Some(pinned), &spec(good)).is_none());
        assert!(check_pinned_contract(Some(pinned), &spec([0u8; 32])).is_some());
        assert!(check_pinned_contract(None, &spec([0u8; 32])).is_none());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval_checkpoint::{eval_checkpoint_path, BeforeWrite, EvalCheckpoints};

    const CAP: Duration = Duration::from_millis(1000);

    fn member(learner_id: u32, generation: u64) -> Member {
        Member {
            learner_id,
            generation,
        }
    }

    fn spec(dtype: u8, layout: &Layout, fingerprint: u8, contract: u8) -> SessionSpec {
        SessionSpec {
            dtype,
            layout: layout.clone(),
            layout_fingerprint: [fingerprint; 32],
            session_contract_hash: [contract; 32],
            syncer_profile_hash: None,
            compat_group: "nvidia-h100".to_string(),
        }
    }

    /// yeto-framework-decoupling 7.7c: legacy HELLO with another compat_group
    /// (H200 or Ascend island in an H100 session) is refused with both groups
    /// named, the session is untouched; same group H200/H200 is admitted.
    #[test]
    fn hello_with_other_compat_group_is_refused_naming_both_card_types() {
        let layout = Layout { fragments: Vec::new() };
        let session: Session = Arc::new(Mutex::new(None));
        let h100 = spec(DTYPE_F32, &layout, 0xaf, 0x4f);
        assert_eq!(admit_session(&session, h100.clone()), None);
        for group in ["nvidia-h200", "ascend-910b"] {
            let mut other = spec(DTYPE_F32, &layout, 0xaf, 0x50);
            other.compat_group = group.to_string();
            let message = admit_session(&session, other).expect("must refuse");
            assert!(message.starts_with("session mismatch"), "{message}");
            assert!(message.contains(&format!("兼容组不同：nvidia-h100 对 {group}，容差未标定")), "{message}");
        }
        assert_eq!(session.lock().unwrap().as_ref(), Some(&h100));
        let session: Session = Arc::new(Mutex::new(None));
        let mut h200 = spec(DTYPE_F32, &layout, 0xaf, 0x4f);
        h200.compat_group = "nvidia-h200".to_string();
        assert_eq!(admit_session(&session, h200.clone()), None);
        assert_eq!(admit_session(&session, h200), None);
    }

    #[test]
    fn first_hello_establishes_session_and_mismatch_is_refused_without_reset() {
        let layout = Layout { fragments: Vec::new() };
        let session: Session = Arc::new(Mutex::new(None));
        let verl = spec(DTYPE_F32, &layout, 0xaf, 0x4f);
        // No session yet: the first HELLO defines the contract.
        assert_eq!(admit_session(&session, verl.clone()), None);
        // Same contract: admitted.
        assert_eq!(admit_session(&session, verl.clone()), None);
        // Same layout, other backend identity (Miles island in a verl session):
        // refused, with both hashes in the message.
        let miles = spec(DTYPE_F32, &layout, 0xaf, 0x69);
        let message = admit_session(&session, miles).expect("must refuse");
        assert!(message.starts_with("session mismatch"), "{message}");
        assert!(message.contains(&format!("expected session_contract_hash={}", "4f".repeat(32))));
        assert!(message.contains(&format!("got session_contract_hash={}", "69".repeat(32))));
        assert!(message.contains(&format!("layout_fingerprint={}", "af".repeat(32))));
        // The refusal does not touch the established session.
        assert_eq!(session.lock().unwrap().as_ref(), Some(&verl));
        assert_eq!(admit_session(&session, verl), None);
    }

    fn registry_with_current(current: Member) -> Registry {
        let registry = Arc::new(Mutex::new(RegistryState::default()));
        registry
            .lock()
            .unwrap()
            .current
            .insert(current.learner_id, current);
        registry
    }

    fn test_group(member: Member) -> Arc<Group> {
        test_group_with_layout(
            member,
            Layout {
                fragments: Vec::new(),
            },
        )
    }

    fn test_group_with_layout(member: Member, layout: Layout) -> Arc<Group> {
        test_group_with_session(member, layout, [0; 32], [0; 32])
    }

    fn test_group_with_session(
        member: Member,
        layout: Layout,
        layout_fingerprint: [u8; 32],
        session_contract_hash: [u8; 32],
    ) -> Arc<Group> {
        let (control, _receiver) = mpsc::channel(1);
        Arc::new(Group {
            member,
            dtype: DTYPE_F32,
            layout,
            layout_fingerprint,
            session_contract_hash,
            num_streams: 0,
            max_init_payload: 0,
            max_push_payload: 0,
            expert_tokens: false,
            max_chunked_inner: 13,
            control,
            data: Mutex::new(HashMap::new()),
            msg_id: AtomicU64::new(0),
            rr: AtomicUsize::new(0),
            reasm: Mutex::new(HashMap::new()),
        })
    }

    fn test_group_with_control(member: Member, control: mpsc::Sender<OutFrame>) -> Arc<Group> {
        Arc::new(Group {
            member,
            dtype: DTYPE_F32,
            layout: Layout {
                fragments: Vec::new(),
            },
            layout_fingerprint: [0; 32],
            session_contract_hash: [0; 32],
            num_streams: 0,
            max_init_payload: 0,
            max_push_payload: 0,
            expert_tokens: false,
            max_chunked_inner: 13,
            control,
            data: Mutex::new(HashMap::new()),
            msg_id: AtomicU64::new(0),
            rr: AtomicUsize::new(0),
            reasm: Mutex::new(HashMap::new()),
        })
    }

    fn streaming_test_group(
        dtype: u8,
        data_stream: bool,
    ) -> (Arc<Group>, mpsc::Receiver<OutFrame>) {
        let (control, control_rx) = mpsc::channel(64);
        let (data, data_rx) = mpsc::channel(64);
        let mut streams = HashMap::new();
        if data_stream {
            streams.insert(0, data);
        }
        let group = Arc::new(Group {
            member: member(0, 10),
            dtype,
            layout: Layout {
                fragments: Vec::new(),
            },
            layout_fingerprint: [0; 32],
            session_contract_hash: [0; 32],
            num_streams: u16::from(data_stream),
            max_init_payload: 0,
            max_push_payload: 0,
            expert_tokens: false,
            max_chunked_inner: u64::MAX,
            control,
            data: Mutex::new(streams),
            msg_id: AtomicU64::new(0),
            rr: AtomicUsize::new(0),
            reasm: Mutex::new(HashMap::new()),
        });
        (group, if data_stream { data_rx } else { control_rx })
    }

    fn buffered_tensor_inner(msg_type: u8, prefix: &[u8], dtype: u8, values: &[f32]) -> Vec<u8> {
        let mut tensor = Vec::new();
        encode_tensor(dtype, values, &mut tensor).unwrap();
        let payload_len = prefix.len() + tensor.len();
        let mut inner = Vec::with_capacity(13 + payload_len);
        inner.extend_from_slice(&MAGIC.to_le_bytes());
        inner.push(msg_type);
        inner.extend_from_slice(&(payload_len as u64).to_le_bytes());
        inner.extend_from_slice(prefix);
        inner.extend_from_slice(&tensor);
        inner
    }

    async fn streamed_tensor_inner(
        msg_type: u8,
        prefix: &[u8],
        dtype: u8,
        values: &[f32],
        chunk_size: usize,
        data_stream: bool,
    ) -> Vec<u8> {
        let (group, mut receiver) = streaming_test_group(dtype, data_stream);
        group
            .send_tensor_large_chunked(msg_type, prefix, dtype, values, chunk_size)
            .await
            .unwrap();

        let total = 13 + prefix.len() + tensor_nbytes(dtype, values.len()).unwrap();
        let frame_count = total.div_ceil(chunk_size);
        let mut inner = Vec::with_capacity(total);
        let mut msg_id = None;
        for _ in 0..frame_count {
            let frame = receiver.recv().await.unwrap();
            assert_eq!(frame.msg_type, MSG_CHUNK);
            assert_eq!(frame.parts.len(), 2);
            assert_eq!(frame.parts[0].len(), CHUNK_HEADER_SIZE as usize);
            assert!(frame.parts[1].len() <= chunk_size);
            assert!(
                frame.parts.iter().map(bytes::Bytes::len).sum::<usize>()
                    <= CHUNK_HEADER_SIZE as usize + chunk_size
            );

            let head = &frame.parts[0];
            let frame_msg_id = u64::from_le_bytes(head[0..8].try_into().unwrap());
            let frame_total = u64::from_le_bytes(head[8..16].try_into().unwrap());
            let frame_offset = u64::from_le_bytes(head[16..24].try_into().unwrap());
            assert_eq!(*msg_id.get_or_insert(frame_msg_id), frame_msg_id);
            assert_eq!(frame_total as usize, total);
            assert_eq!(frame_offset as usize, inner.len());
            inner.extend_from_slice(&frame.parts[1]);
        }
        assert_eq!(inner.len(), total);
        assert!(matches!(
            receiver.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));
        inner
    }

    #[tokio::test]
    async fn streamed_tensor_frames_are_byte_identical_and_chunk_bounded() {
        let values: Vec<f32> = (0..47).map(|value| value as f32 / 8.0 - 2.5).collect();
        let mut prefix = [0u8; 12];
        prefix[..4].copy_from_slice(&3u32.to_le_bytes());
        prefix[4..].copy_from_slice(&9u64.to_le_bytes());
        let chunk_size = 17;

        for (msg_type, dtype, data_stream) in [
            (MSG_BCAST_FRAGMENT, DTYPE_BF16, true),
            (MSG_FINAL_FRAGMENT, DTYPE_F32, false),
        ] {
            let expected = buffered_tensor_inner(msg_type, &prefix, dtype, &values);
            let actual =
                streamed_tensor_inner(msg_type, &prefix, dtype, &values, chunk_size, data_stream)
                    .await;
            assert_eq!(actual, expected);
        }
    }

    fn broadcast_test_state(dtype: u8) -> GlobalState {
        let layout = Layout {
            fragments: vec![crate::state::FragmentInfo {
                merge_mode: crate::state::MERGE_AVG,
                tensor_numels: vec![4],
                tensor_shapes: None,
            }],
        };
        let mut state = GlobalState::new(layout, 1.0, 0.0, dtype).unwrap();
        state.init_fragment(0, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        state
    }

    fn registry_with_group(group: Arc<Group>) -> Registry {
        let registry = Arc::new(Mutex::new(RegistryState::default()));
        registry.lock().unwrap().register_group(group, Instant::now()).unwrap();
        registry
    }

    #[test]
    fn resume_refuses_a_changed_or_missing_profile_record() {
        let dir = std::env::temp_dir().join(format!("yeto-profile-record-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let checkpoint = dir.join("actor.ckpt");
        std::fs::write(&checkpoint, b"x").unwrap();
        let record = profile_record_path(&checkpoint);
        let _ = std::fs::remove_file(&record);
        let mut config = round_test_config(4);
        config.checkpoint_path = Some(checkpoint.clone());
        config.resume = true;
        // No requirement: nothing is checked, the record is (re)written.
        check_and_record_profile(&config, &[1; 32]).unwrap();
        assert_eq!(std::fs::read_to_string(&record).unwrap().trim(), hex32(&[1; 32]));
        check_and_record_profile(&config, &[2; 32]).unwrap();
        check_and_record_profile(&config, &[1; 32]).unwrap();
        // Required: the same profile resumes; another one is refused with the remedy.
        config.require_profile_record = true;
        check_and_record_profile(&config, &[1; 32]).unwrap();
        let error = check_and_record_profile(&config, &[2; 32]).unwrap_err().to_string();
        assert!(error.contains("other sync settings"), "{error}");
        assert!(error.contains("--rl-sync-mode strict"), "{error}");
        // A missing record is refused when the launcher requires one.
        std::fs::remove_file(&record).unwrap();
        let error = check_and_record_profile(&config, &[1; 32]).unwrap_err().to_string();
        assert!(error.contains("no semantic profile record"), "{error}");
        // A fresh start always (re)writes the record.
        config.resume = false;
        check_and_record_profile(&config, &[3; 32]).unwrap();
        assert_eq!(std::fs::read_to_string(&record).unwrap().trim(), hex32(&[3; 32]));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    fn round_test_config(total_steps: u64) -> Config {
        Config {
            port: 0,
            learners: 1,
            quorum: 1,
            grace_ms: 0,
            grace_gamma: 0.5,
            grace_tau: 0.0,
            grace_min_ms: 0,
            grace_arrival_quantile: 0.0,
            terminal_grace_ms: 0,
            pipeline: 1,
            min_round_interval_ms: 0,
            sync_interval_steps: 0.0,
            delta_correction: false,
            quorum_timeout_s: 1,
            final_ack_timeout_s: 1,
            total_steps,
            policy_sweep_fragments: None,
            outer_lr: 1.0,
            outer_momentum: 0.0,
            outer_delayed_nesterov_period: None,
            iso_backend: crate::iso_worker::IsoBackendConfig::default(),
            final_state: None,
            checkpoint_path: None,
            checkpoint_every: 0,
            checkpoint_at_steps: Vec::new(),
            async_checkpoint: false,
            bcast_delta_full_every: None,
            resume: false,
            mark_final_checkpoint: false,
            require_profile_record: false,
            learner_budget_steps: None,
            event_tape: None,
            max_base_lag: Some(0),
            learner_weight: LearnerWeight::Equal,
            require_profile_binding: false,
            island_hmac_key: None,
            expected_island_contract: None,
            syncer_epoch: 0,
            membership: Default::default(),
            recovery: Default::default(),
            weighted_fold: Default::default(),
            expert_token_weights: false,
            final_cut_manifest_only: false,
            dispense: None,
            dispense_log: None,
            shard: Default::default(),
        }
    }

    const FINAL_CUT_MANIFEST_ONLY_VECTOR: &str = "e605792438d8fc745769a6b4334707d749856a1cb0eef84b0b03c9eb5072560f";
    const GRACE_FLOOR_VECTOR: &str = "6f9104e302f894575fe2203cfe91a93ff37d16e37b4e75c4367b060d13748d12";
    const TERMINAL_GRACE_VECTOR: &str = "6f40c5c1c02e6c63b99bf5aa77fab924fed928a42a9e0d88811583a19fc1aa39";

    #[test]
    fn semantic_profile_hash_matches_the_python_canonical_vector() {
        let mut config = round_test_config(8);
        config.learners = 2;
        config.quorum = 2;
        config.grace_gamma = 0.8;
        config.grace_tau = 2.0;
        config.pipeline = 2;
        config.sync_interval_steps = 24.0;
        config.quorum_timeout_s = 900;
        config.final_ack_timeout_s = 900;
        config.outer_lr = 0.7;
        config.outer_momentum = 0.9;
        config.checkpoint_path = Some(std::path::PathBuf::from("/ignored/actor.ckpt"));
        config.checkpoint_every = 1;
        config.resume = true;
        config.mark_final_checkpoint = true;
        config.require_profile_binding = true;
        let digest = config
            .semantic_profile_hash()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(
            digest,
            "85a50d0e4fd5103b81a2295536fca0e5ad96cd50531fb1396208af8459be571e"
        );

        let original = config.semantic_profile_hash();
        config.port = 30123;
        config.checkpoint_path = Some(std::path::PathBuf::from("/another/critic.ckpt"));
        config.final_state = Some(std::path::PathBuf::from("/ignored/final.bin"));
        config.event_tape = Some(std::path::PathBuf::from("/ignored/events.jsonl"));
        config.resume = false; // P0-1.7b.1-Q7: how the syncer starts, not semantics
        assert_eq!(config.semantic_profile_hash(), original);
        config.pipeline = 3;
        assert_ne!(config.semantic_profile_hash(), original);
        // Expert token weights append a domain; Python vector in
        // tests/test_rl_moe_expert_token_weights.py.
        config.pipeline = 2;
        config.expert_token_weights = true;
        let digest: String =
            config.semantic_profile_hash().iter().map(|byte| format!("{byte:02x}")).collect();
        assert_eq!(digest, "f51de8a095c0fa428c5483c7921da805bb8b3cd2736fe19fdd03f47e4f733913");
        // A manifest-only final cut appends its own domain; Python vector in
        // tests/test_s27_final_cut_ack.py.
        config.expert_token_weights = false;
        config.final_cut_manifest_only = true;
        let digest: String =
            config.semantic_profile_hash().iter().map(|byte| format!("{byte:02x}")).collect();
        assert_ne!(config.semantic_profile_hash(), original);
        assert_eq!(digest, FINAL_CUT_MANIFEST_ONLY_VECTOR);
        // A grace floor appends its own block; off keeps the hash. Python
        // vector in tests/test_s27_grace_floor.py.
        config.final_cut_manifest_only = false;
        config.grace_arrival_quantile = 0.0;
        config.grace_min_ms = 0;
        assert_eq!(config.semantic_profile_hash(), original);
        config.grace_min_ms = 15_000;
        config.grace_arrival_quantile = 0.9;
        let digest: String =
            config.semantic_profile_hash().iter().map(|byte| format!("{byte:02x}")).collect();
        assert_eq!(digest, GRACE_FLOOR_VECTOR);
        config.grace_min_ms = 0;
        assert_ne!(config.semantic_profile_hash(), original);
        // So does a terminal grace (same Python test file).
        config.grace_arrival_quantile = 0.0;
        assert_eq!(config.semantic_profile_hash(), original);
        config.terminal_grace_ms = 120_000;
        let digest: String =
            config.semantic_profile_hash().iter().map(|byte| format!("{byte:02x}")).collect();
        assert_eq!(digest, TERMINAL_GRACE_VECTOR);
    }

    #[test]
    fn delayed_nesterov_profile_hash_matches_the_python_canonical_vector() {
        // Python: tests/test_rl_inter_island_contract.py (DELAYED_NESTEROV_4).
        let mut config = round_test_config(8);
        config.learners = 2;
        config.quorum = 2;
        config.grace_gamma = 0.8;
        config.grace_tau = 2.0;
        config.pipeline = 2;
        config.sync_interval_steps = 24.0;
        config.quorum_timeout_s = 900;
        config.final_ack_timeout_s = 900;
        config.outer_lr = 0.7;
        config.outer_momentum = 0.9;
        config.checkpoint_path = Some(std::path::PathBuf::from("/ignored/actor.ckpt"));
        config.checkpoint_every = 1;
        config.mark_final_checkpoint = true;
        config.require_profile_binding = true;
        config.outer_delayed_nesterov_period = Some(4);
        let digest = config
            .semantic_profile_hash()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(
            digest,
            "2bf20f25b4d40d6d3db1d91e38d199e0347f9dace3fc481a2948fa7f1c99f126"
        );
    }

    #[test]
    fn fixed_roster_round_needs_every_learner_and_adds_no_extra_wait() {
        // Strict fixed roster: quorum == learners, grace 0.
        let first = member(0, 10);
        let second = member(1, 20);
        let mut round = test_round(vec![first, second]);
        round.quorum_size = 2;
        round.quorum_deadline = Instant::now() + Duration::from_secs(3600);
        round.pushes.insert(first, test_push(5));
        // One of two arrived: keep waiting (no capacity-fraction step).
        assert_eq!(round_action(&round, Instant::now()), RoundAction::Wait);
        round.pushes.insert(second, test_push(5));
        // Arrived == learners: complete at once, an hour before the deadline.
        assert_eq!(round_action(&round, Instant::now()), RoundAction::Complete);
    }

    #[test]
    fn fixed_roster_round_timeout_never_merges_a_partial_quorum() {
        let first = member(0, 10);
        let mut round = test_round(vec![first, member(1, 20)]);
        round.quorum_size = 2;
        round.pushes.insert(first, test_push(5));
        round.quorum_deadline = Instant::now() - Duration::from_millis(1);
        assert_eq!(round_action(&round, Instant::now()), RoundAction::Restart);
        assert!(next_committable_round(&[round], Instant::now()).is_none());
    }

    #[test]
    fn fixed_roster_round_rejects_a_late_delta_without_carrying_it() {
        let only = member(0, 10);
        let mut rounds = vec![test_round(vec![only])];
        assert_eq!(
            route_push(&mut rounds, only, test_exact_push(1, 4), Some(0), true),
            PushDisposition::StaleBase
        );
        assert!(rounds[0].pushes.is_empty());
    }

    fn sweep_test_config(fragments: u32, total_steps: u64) -> Config {
        let mut config = round_test_config(total_steps);
        config.policy_sweep_fragments = Some(fragments);
        config.checkpoint_path = Some(std::path::PathBuf::from("policy-sweep-test.ckpt"));
        config.checkpoint_every = 1;
        config.resume = true;
        config
    }

    #[test]
    fn sharded_head_takes_membership_and_delayed_nesterov_but_not_sweeps() {
        let mut sharded = round_test_config(8);
        sharded.shard = crate::shard::ShardPlan::new(1, 2).unwrap();
        assert!(validate_config(&sharded).is_ok());
        sharded.learners = 2;
        sharded.membership.lease = Some(Duration::from_secs(5));
        sharded.membership.soft_deadline = Some(Duration::from_secs(30));
        sharded.membership.quorum_rule =
            crate::membership::QuorumRule::Capacity { theta: 0.5, q_min: 1 };
        sharded.outer_delayed_nesterov_period = Some(4);
        assert!(validate_config(&sharded).is_ok());
        let mut sweep = sweep_test_config(2, 4);
        sweep.shard = sharded.shard;
        assert!(format!("{:#}", validate_config(&sweep).unwrap_err())
            .contains("does not support --policy-sweep-fragments"));
    }

    #[test]
    fn policy_sweep_static_profile_is_strict_and_legacy_profile_is_unchanged() {
        assert!(validate_config(&round_test_config(3)).is_ok());
        assert!(validate_config(&sweep_test_config(2, 4)).is_ok());

        let mut invalid = sweep_test_config(0, 4);
        assert!(
            format!("{:#}", validate_config(&invalid).unwrap_err()).contains("must be positive")
        );

        invalid = sweep_test_config(2, 0);
        assert!(format!("{:#}", validate_config(&invalid).unwrap_err())
            .contains("requires positive --total-steps"));

        let mut streamed = sweep_test_config(2, 4);
        streamed.pipeline = 2;
        streamed.async_checkpoint = true;
        assert!(validate_config(&streamed).is_ok());
        for pipeline in [0, 3] {
            invalid = sweep_test_config(2, 4);
            invalid.pipeline = pipeline;
            assert!(format!("{:#}", validate_config(&invalid).unwrap_err())
                .contains("requires --pipeline between 1 and the fragment count"));
        }
        let mut legacy_async = round_test_config(3);
        legacy_async.async_checkpoint = true;
        assert!(format!("{:#}", validate_config(&legacy_async).unwrap_err())
            .contains("--async-checkpoint requires --policy-sweep-fragments"));
        invalid = sweep_test_config(2, 4);
        invalid.async_checkpoint = true;
        invalid.learner_budget_steps = Some(1);
        assert!(format!("{:#}", validate_config(&invalid).unwrap_err())
            .contains("does not support --learner-budget-steps"));
        // Async persistence is not semantic: the profile hash ignores it.
        let mut hashed = sweep_test_config(2, 4);
        let before = hashed.semantic_profile_hash();
        hashed.async_checkpoint = true;
        assert_eq!(before, hashed.semantic_profile_hash());

        invalid = sweep_test_config(3, 4);
        assert!(
            format!("{:#}", validate_config(&invalid).unwrap_err()).contains("must be divisible")
        );

        for mutate in [
            |config: &mut Config| config.max_base_lag = None,
            |config: &mut Config| {
                config.learners = 2;
                config.quorum = 1;
            },
            |config: &mut Config| config.grace_ms = 1,
        ] {
            invalid = sweep_test_config(2, 4);
            mutate(&mut invalid);
            assert!(format!("{:#}", validate_config(&invalid).unwrap_err())
                .contains("strict fixed roster"));
        }

        invalid = sweep_test_config(2, 4);
        invalid.delta_correction = true;
        assert!(format!("{:#}", validate_config(&invalid).unwrap_err())
            .contains("requires --delta-correction none"));

        invalid = sweep_test_config(2, 4);
        invalid.learner_weight = LearnerWeight::Tokens2OverSteps;
        assert!(format!("{:#}", validate_config(&invalid).unwrap_err())
            .contains("requires --learner-weight equal"));

        for mutate in [
            |config: &mut Config| config.checkpoint_path = None,
            |config: &mut Config| config.checkpoint_every = 0,
            |config: &mut Config| config.checkpoint_every = 3,
        ] {
            invalid = sweep_test_config(2, 4);
            mutate(&mut invalid);
            assert!(format!("{:#}", validate_config(&invalid).unwrap_err())
                .contains("requires --checkpoint-path"));
        }
        // Sweep-aligned checkpoint intervals are accepted.
        let mut interval = sweep_test_config(2, 4);
        interval.checkpoint_every = 4;
        assert!(validate_config(&interval).is_ok());
        assert!(persist_before_broadcast(&interval, 4));
        assert!(!persist_before_broadcast(&interval, 2));
        assert!(persist_before_broadcast(&sweep_test_config(2, 4), 1));
    }

    #[test]
    fn policy_sweep_first_start_needs_no_resume_and_resume_needs_a_file() {
        // A fresh sweep starts without --resume (no checkpoint exists yet).
        let mut fresh = sweep_test_config(2, 4);
        fresh.resume = false;
        assert!(validate_config(&fresh).is_ok());
        assert!(check_resume_checkpoint(&fresh).is_ok());

        let directory = std::env::temp_dir().join(format!(
            "yeto-sweep-resume-file-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let checkpoint = directory.join("state.ckpt");

        // --resume with a missing file still fails (sweep and non-sweep).
        let mut sweep = sweep_test_config(2, 4);
        sweep.checkpoint_path = Some(checkpoint.clone());
        let mut legacy = round_test_config(3);
        legacy.checkpoint_path = Some(checkpoint.clone());
        legacy.resume = true;
        for config in [&sweep, &legacy] {
            assert!(format!("{:#}", check_resume_checkpoint(config).unwrap_err())
                .contains("does not exist or is not a file"));
        }

        // A restart with --resume and a checkpoint file is accepted.
        std::fs::write(&checkpoint, b"x").unwrap();
        assert!(check_resume_checkpoint(&sweep).is_ok());
        assert!(check_resume_checkpoint(&legacy).is_ok());
        std::fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    fn policy_sweep_must_equal_the_decoded_layout_fragment_count() {
        let layout = Layout {
            fragments: vec![
                crate::state::FragmentInfo {
                    merge_mode: crate::state::MERGE_AVG,
                    tensor_numels: vec![1],
                    tensor_shapes: None,
                },
                crate::state::FragmentInfo {
                    merge_mode: crate::state::MERGE_AVG,
                    tensor_numels: vec![1],
                    tensor_shapes: None,
                },
            ],
        };
        let group = test_group_with_layout(member(0, 1), layout);
        assert!(new_state_for(&group, &sweep_test_config(2, 4)).is_ok());
        let error = new_state_for(&group, &sweep_test_config(1, 4))
            .err()
            .unwrap();
        assert!(format!("{error:#}").contains("decoded layout has 2 fragments"));
    }

    #[test]
    fn generic_streaming_restart_requires_the_exact_session_contract() {
        let directory = std::env::temp_dir().join(format!(
            "yeto-generic-stream-resume-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let checkpoint = directory.join("state.ckpt");
        let layout = Layout {
            fragments: vec![crate::state::FragmentInfo {
                merge_mode: crate::state::MERGE_AVG,
                tensor_numels: vec![2],
                tensor_shapes: None,
            }],
        };
        let config = round_test_config(3);

        let first_group = test_group_with_session(member(0, 1), layout.clone(), [3; 32], [4; 32]);
        let mut before_restart = new_state_for(&first_group, &config).unwrap();
        assert_eq!(before_restart.policy_sweep_fragments, None);
        assert_eq!(before_restart.session_contract_hash, Some([4; 32]));
        before_restart.init_fragment(0, vec![1.0, -2.0]).unwrap();
        before_restart.global_step = 2;
        before_restart.versions[0] = 2;
        before_restart.record_merge(0, 5, 1234);
        before_restart.save_checkpoint(&checkpoint).unwrap();

        let matching_group =
            test_group_with_session(member(0, 2), layout.clone(), [3; 32], [4; 32]);
        let mut recovered = new_state_for(&matching_group, &config).unwrap();
        recovered.load_checkpoint(&checkpoint).unwrap();
        assert_eq!(recovered.global_step, 2);
        assert_eq!(recovered.versions, vec![2]);
        assert_eq!(recovered.params, before_restart.params);
        assert_eq!(recovered.ledger.get(&0).unwrap().tokens, 1234);

        let mismatched_group = test_group_with_session(member(0, 2), layout, [3; 32], [5; 32]);
        let mut mismatched = new_state_for(&mismatched_group, &config).unwrap();
        let error = mismatched.load_checkpoint(&checkpoint).unwrap_err();
        assert!(format!("{error:#}").contains("session contract hash does not match"));

        std::fs::remove_dir_all(&directory).ok();
    }

    #[test]
    fn policy_sweep_pushes_bind_fragment_steps_to_one_logical_local_step() {
        for (global_step, local_step) in [(1, 1), (2, 1), (3, 1), (4, 2), (6, 2)] {
            let push = Push {
                learner_id: 0,
                fragment_id: ((global_step - 1) % 3) as u32,
                global_step,
                round_attempt: 1,
                base_version: 0,
                local_step,
                c_steps: 1,
                c_tokens: 99,
                outer_gradient: vec![1.0],
                tensor_scales: None,
                transfer: PushTransfer::default(),
            };
            validate_policy_sweep_push(&push, 3).unwrap();
        }

        let mut bad_local_step = test_exact_push(0, 5);
        bad_local_step.global_step = 4;
        bad_local_step.local_step = 1;
        assert!(format!(
            "{:#}",
            validate_policy_sweep_push(&bad_local_step, 3).unwrap_err()
        )
        .contains("requires local_step 2"));

        let mut bad_c_steps = bad_local_step;
        bad_c_steps.local_step = 2;
        bad_c_steps.c_steps = 2;
        assert!(format!(
            "{:#}",
            validate_policy_sweep_push(&bad_c_steps, 3).unwrap_err()
        )
        .contains("requires c_steps=1"));
    }

    #[test]
    fn policy_sweep_resume_mapping_is_deterministic_mid_sweep() {
        let expected = [
            [0, 0, 0],
            [1, 0, 0],
            [1, 2, 0],
            [1, 2, 3],
            [4, 2, 3],
            [4, 5, 3],
            [4, 5, 6],
        ];
        for (global_step, versions) in expected.into_iter().enumerate() {
            let actual = [
                expected_sweep_fragment_version(global_step as u64, 0, 3),
                expected_sweep_fragment_version(global_step as u64, 1, 3),
                expected_sweep_fragment_version(global_step as u64, 2, 3),
            ];
            assert_eq!(actual, versions);
        }

        let layout = Layout {
            fragments: (0..3)
                .map(|_| crate::state::FragmentInfo {
                    merge_mode: crate::state::MERGE_AVG,
                    tensor_numels: vec![1],
                    tensor_shapes: None,
                })
                .collect(),
        };
        let mut state = GlobalState::new(layout, 1.0, 0.0, DTYPE_F32).unwrap();
        state.policy_sweep_fragments = Some(3);
        state.session_contract_hash = Some([2; 32]);
        for fragment in 0..3 {
            state.init_fragment(fragment, vec![0.0]).unwrap();
        }
        state.global_step = 2;
        state.versions = vec![1, 2, 0];
        for learner_id in 0..2 {
            state.record_fragment_merge(learner_id, 1, 50, false);
            state.record_fragment_merge(learner_id, 1, 50, false);
        }
        let mut config = sweep_test_config(3, 6);
        config.learners = 2;
        config.quorum = 2;
        validate_resumed_policy_sweep(&config, &state).unwrap();

        state.ledger.get_mut(&0).unwrap().steps = 1;
        assert!(format!(
            "{:#}",
            validate_resumed_policy_sweep(&config, &state).unwrap_err()
        )
        .contains("expected merges=2 steps=0"));
    }

    fn terminal_test_round(step: u64, member: Member) -> Round {
        Round {
            t: step,
            p: 0,
            base_version: 0,
            attempt: 1,
            pull: bytes::Bytes::new(),
            started: Instant::now(),
            expected_members: vec![member],
            quorum_deadline: Instant::now(),
            grace_deadline: None,
            quorum_size: 1,
            capacity_quorum: None,
            soft_deadline: None,
            soft_min: 1,
            quorum_ms: Some(0),
            grace_ms: Some(0),
            fold: None,
            carried: Vec::new(),
            pushes: HashMap::from([(
                member,
                Push {
                    learner_id: member.learner_id,
                    fragment_id: 0,
                    global_step: step,
                    round_attempt: 1,
                    base_version: 0,
                    local_step: step,
                    c_steps: 1,
                    c_tokens: 1,
                    outer_gradient: vec![0.25; 4],
                    tensor_scales: None,
                    transfer: PushTransfer::default(),
                },
            )]),
        }
    }

    fn two_fragment_sweep_state() -> GlobalState {
        let layout = Layout {
            fragments: (0..2)
                .map(|_| crate::state::FragmentInfo {
                    merge_mode: crate::state::MERGE_AVG,
                    tensor_numels: vec![1],
                    tensor_shapes: None,
                })
                .collect(),
        };
        let mut state =
            GlobalState::new_with_layout_fingerprint(layout, 1.0, 0.0, DTYPE_F32, [3; 32]).unwrap();
        state.policy_sweep_fragments = Some(2);
        state.session_contract_hash = Some([3; 32]);
        state.init_fragment(0, vec![0.0]).unwrap();
        state.init_fragment(1, vec![0.0]).unwrap();
        state
    }

    fn three_fragment_sweep_state() -> GlobalState {
        let layout = Layout {
            fragments: (0..3)
                .map(|_| crate::state::FragmentInfo {
                    merge_mode: crate::state::MERGE_AVG,
                    tensor_numels: vec![1],
                    tensor_shapes: None,
                })
                .collect(),
        };
        let mut state =
            GlobalState::new_with_layout_fingerprint(layout, 1.0, 0.0, DTYPE_F32, [4; 32]).unwrap();
        state.policy_sweep_fragments = Some(3);
        state.session_contract_hash = Some([4; 32]);
        for fragment in 0..3 {
            state.init_fragment(fragment, vec![0.0]).unwrap();
        }
        state
    }

    fn sweep_test_round(
        step: u64,
        fragment: usize,
        policy_round: u64,
        c_tokens: u64,
        member: Member,
    ) -> Round {
        Round {
            t: step,
            p: fragment,
            base_version: 0,
            attempt: 1,
            pull: bytes::Bytes::new(),
            started: Instant::now(),
            expected_members: vec![member],
            quorum_deadline: Instant::now(),
            grace_deadline: None,
            quorum_size: 1,
            capacity_quorum: None,
            soft_deadline: None,
            soft_min: 1,
            quorum_ms: Some(0),
            grace_ms: Some(0),
            fold: None,
            carried: Vec::new(),
            pushes: HashMap::from([(
                member,
                Push {
                    learner_id: member.learner_id,
                    fragment_id: fragment as u32,
                    global_step: step,
                    round_attempt: 1,
                    base_version: 0,
                    local_step: policy_round,
                    c_steps: 1,
                    c_tokens,
                    outer_gradient: vec![0.25],
                    tensor_scales: None,
                    transfer: PushTransfer::default(),
                },
            )]),
        }
    }

    fn two_learner_sweep_round(step: u64) -> Round {
        let policy_round = step.div_ceil(3);
        let fragment = ((step - 1) % 3) as usize;
        let base_version = step.saturating_sub(3);
        let members = [member(0, 30), member(1, 31)];
        let pushes = members
            .into_iter()
            .map(|member| {
                (
                    member,
                    Push {
                        learner_id: member.learner_id,
                        fragment_id: fragment as u32,
                        global_step: step,
                        round_attempt: 1,
                        base_version,
                        local_step: policy_round,
                        c_steps: 1,
                        c_tokens: u64::from(member.learner_id + 1) * policy_round * 100,
                        outer_gradient: vec![policy_round as f32 + 2.0 * member.learner_id as f32],
                        tensor_scales: None,
                        transfer: PushTransfer::default(),
                    },
                )
            })
            .collect();
        Round {
            t: step,
            p: fragment,
            base_version,
            attempt: 1,
            pull: bytes::Bytes::new(),
            started: Instant::now(),
            expected_members: members.to_vec(),
            quorum_deadline: Instant::now(),
            grace_deadline: None,
            quorum_size: 2,
            capacity_quorum: None,
            soft_deadline: None,
            soft_min: 1,
            quorum_ms: Some(0),
            grace_ms: Some(0),
            fold: None,
            carried: Vec::new(),
            pushes,
        }
    }

    fn two_fragment_streaming_state() -> GlobalState {
        let layout = Layout {
            fragments: (0..2)
                .map(|_| crate::state::FragmentInfo {
                    merge_mode: crate::state::MERGE_AVG,
                    tensor_numels: vec![1],
                    tensor_shapes: None,
                })
                .collect(),
        };
        let mut state = GlobalState::new(layout, 1.0, 0.0, DTYPE_F32).unwrap();
        state.init_fragment(0, vec![0.0]).unwrap();
        state.init_fragment(1, vec![0.0]).unwrap();
        state
    }

    #[test]
    fn pipelined_scheduler_commits_only_the_oldest_ready_round() {
        let learner = member(0, 30);
        let mut oldest = sweep_test_round(1, 0, 1, 1, learner);
        oldest.pushes.clear();
        oldest.quorum_deadline = Instant::now() + CAP;
        let later = sweep_test_round(2, 1, 2, 1, learner);
        let mut rounds = vec![later, oldest];

        assert_eq!(next_committable_round(&rounds, Instant::now()), None);

        rounds[1] = sweep_test_round(1, 0, 1, 1, learner);
        assert_eq!(next_committable_round(&rounds, Instant::now()), Some(1));
        rounds.remove(1);
        assert_eq!(next_committable_round(&rounds, Instant::now()), Some(0));
    }

    fn chunked_inner_type(frame: &OutFrame) -> u8 {
        assert_eq!(frame.msg_type, MSG_CHUNK);
        assert_eq!(frame.parts.len(), 2);
        frame.parts[1][4]
    }

    #[tokio::test]
    async fn terminal_round_skips_redundant_broadcast_and_uses_final_cut_once() {
        let learner = member(0, 30);

        let mut nonterminal_state = broadcast_test_state(DTYPE_F32);
        let (nonterminal_group, mut nonterminal_rx) = streaming_test_group(DTYPE_F32, false);
        let nonterminal_registry = registry_with_group(nonterminal_group);
        let mut nonterminal_sync_secs = 0.0;
        complete_round(
            &round_test_config(2),
            &mut nonterminal_state,
            &nonterminal_registry,
            &mut nonterminal_sync_secs,
            terminal_test_round(1, learner),
        )
        .await
        .unwrap();
        let bcast = nonterminal_rx.try_recv().unwrap();
        assert_eq!(chunked_inner_type(&bcast), MSG_BCAST_FRAGMENT);
        assert!(matches!(
            nonterminal_rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));

        let mut terminal_state = broadcast_test_state(DTYPE_F32);
        let (terminal_group, mut terminal_rx) = streaming_test_group(DTYPE_F32, false);
        let terminal_registry = registry_with_group(terminal_group.clone());
        let mut terminal_sync_secs = 0.0;
        complete_round(
            &round_test_config(1),
            &mut terminal_state,
            &terminal_registry,
            &mut terminal_sync_secs,
            terminal_test_round(1, learner),
        )
        .await
        .unwrap();
        assert!(matches!(
            terminal_rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));

        send_final_cut(&terminal_state, &terminal_group, false)
            .await
            .unwrap();
        let final_fragment = terminal_rx.try_recv().unwrap();
        assert_eq!(chunked_inner_type(&final_fragment), MSG_FINAL_FRAGMENT);
        assert_eq!(terminal_rx.try_recv().unwrap().msg_type, MSG_FINAL_MANIFEST);
        assert!(matches!(
            terminal_rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));

        // A manifest-only final cut sends the version manifest alone.
        send_final_cut(&terminal_state, &terminal_group, true)
            .await
            .unwrap();
        assert_eq!(terminal_rx.try_recv().unwrap().msg_type, MSG_FINAL_MANIFEST);
        assert!(matches!(
            terminal_rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));
    }

    #[tokio::test]
    async fn generic_checkpoint_failure_happens_before_nonterminal_broadcast() {
        let directory = std::env::temp_dir().join(format!(
            "yeto-generic-pre-broadcast-checkpoint-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let checkpoint = directory.join("state.ckpt");
        let checkpoint_tmp = checkpoint.with_extension("tmp");

        let initial = broadcast_test_state(DTYPE_F32);
        initial.save_checkpoint(&checkpoint).unwrap();
        // Atomic checkpoint writes create this exact temporary path. Turning
        // it into a directory makes the next save fail deterministically.
        std::fs::create_dir(&checkpoint_tmp).unwrap();

        let learner = member(0, 30);
        let (group, mut frames) = streaming_test_group(DTYPE_F32, false);
        let registry = registry_with_group(group);
        let mut config = round_test_config(2);
        config.checkpoint_path = Some(checkpoint.clone());
        config.checkpoint_every = 1;
        // This is the generic streaming profile that previously saved only
        // after exposing the merged version.
        config.max_base_lag = Some(2);
        let mut state = broadcast_test_state(DTYPE_F32);
        let mut sync_seconds = 0.0;

        complete_round(
            &config,
            &mut state,
            &registry,
            &mut sync_seconds,
            terminal_test_round(1, learner),
        )
        .await
        .unwrap_err();

        assert!(matches!(
            frames.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));
        let mut recovered = broadcast_test_state(DTYPE_F32);
        recovered.load_checkpoint(&checkpoint).unwrap();
        assert_eq!(recovered.global_step, 0);
        assert_eq!(recovered.versions, vec![0]);

        std::fs::remove_dir_all(&directory).ok();
    }

    #[tokio::test]
    async fn pipelined_checkpoint_refuses_holes_and_resumes_contiguously() {
        let directory = std::env::temp_dir().join(format!(
            "yeto-pipelined-contiguous-checkpoint-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let checkpoint = directory.join("state.ckpt");
        let learner = member(0, 30);
        let (group, mut frames) = streaming_test_group(DTYPE_F32, false);
        let registry = registry_with_group(group);
        let mut config = round_test_config(2);
        config.pipeline = 2;
        config.checkpoint_path = Some(checkpoint.clone());
        config.checkpoint_every = 1;
        config.resume = true;
        let mut sync_seconds = 0.0;

        let mut before_crash = two_fragment_streaming_state();
        before_crash.save_checkpoint(&checkpoint).unwrap();
        let error = complete_round(
            &config,
            &mut before_crash,
            &registry,
            &mut sync_seconds,
            sweep_test_round(2, 1, 2, 1, learner),
        )
        .await
        .unwrap_err();
        assert!(format!("{error:#}").contains("refusing non-contiguous round commit"));
        assert_eq!(before_crash.global_step, 0);
        assert_eq!(before_crash.versions, vec![0, 0]);
        assert!(matches!(
            frames.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));

        let mut resumed = two_fragment_streaming_state();
        resumed.load_checkpoint(&checkpoint).unwrap();
        assert_eq!(resumed.global_step, 0);
        assert_eq!(resumed.versions, vec![0, 0]);

        complete_round(
            &config,
            &mut resumed,
            &registry,
            &mut sync_seconds,
            sweep_test_round(1, 0, 1, 1, learner),
        )
        .await
        .unwrap();
        assert_eq!(
            chunked_inner_type(&frames.try_recv().unwrap()),
            MSG_BCAST_FRAGMENT
        );
        let mut after_first_commit = two_fragment_streaming_state();
        after_first_commit.load_checkpoint(&checkpoint).unwrap();
        assert_eq!(after_first_commit.global_step, 1);
        assert_eq!(after_first_commit.versions, vec![1, 0]);

        complete_round(
            &config,
            &mut resumed,
            &registry,
            &mut sync_seconds,
            sweep_test_round(2, 1, 2, 1, learner),
        )
        .await
        .unwrap();
        let mut final_reload = two_fragment_streaming_state();
        final_reload.load_checkpoint(&checkpoint).unwrap();
        assert_eq!(final_reload.global_step, 2);
        assert_eq!(final_reload.versions, vec![1, 2]);
        assert_eq!(final_reload.ledger.get(&0).unwrap().merges, 2);
        assert_eq!(final_reload.params, resumed.params);

        std::fs::remove_dir_all(&directory).ok();
    }

    fn unique_test_directory(tag: &str) -> std::path::PathBuf {
        let directory = std::env::temp_dir().join(format!(
            "yeto-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        directory
    }

    #[test]
    fn bf16_push_session_rounds_only_the_delta_within_bf16_error() {
        // dtype 4 decodes pushes as bf16 and keeps every bulk payload f32.
        assert_eq!(bulk_dtype(DTYPE_F32_PUSH_BF16), DTYPE_F32);
        let delta: Vec<f32> = (0..4096)
            .map(|i| ((i as f32) * 0.37).sin() * 10f32.powi((i % 9) - 4))
            .collect();
        let mut wire = Vec::new();
        encode_tensor(push_dtype(DTYPE_F32_PUSH_BF16), &delta, &mut wire).unwrap();
        assert_eq!(wire.len(), delta.len() * 2);
        let mut decoded = Vec::new();
        decode_tensor(push_dtype(DTYPE_F32_PUSH_BF16), &wire, &mut decoded).unwrap();
        for (exact, rounded) in delta.iter().zip(&decoded) {
            // bf16 keeps 8 significant bits: round-to-nearest error <= 2^-8 relative.
            assert!((exact - rounded).abs() <= exact.abs() * (1.0 / 256.0));
        }
        // A merge of bf16 deltas into f32 masters stays within the same bound
        // per learner, so a two-learner average is off by at most that much.
        let master = 1.0f32;
        for (exact, rounded) in delta.iter().zip(&decoded) {
            let merged_exact = master + exact;
            let merged = master + rounded;
            assert!((merged_exact - merged).abs() <= exact.abs() / 256.0 + f32::EPSILON * 2.0);
        }
    }

    #[tokio::test]
    async fn sweep_checkpoint_interval_persists_only_sweep_aligned_steps() {
        let directory = unique_test_directory("sweep-checkpoint-interval");
        let learner = member(0, 30);
        for asynchronous in [false, true] {
            let checkpoint = directory.join(format!("state-{asynchronous}.ckpt"));
            let (group, mut frames) = streaming_test_group(DTYPE_F32, false);
            let registry = registry_with_group(group);
            let mut config = sweep_test_config(2, 8);
            config.checkpoint_path = Some(checkpoint.clone());
            config.checkpoint_every = 4;
            config.async_checkpoint = asynchronous;
            let mut state = two_fragment_sweep_state();
            state.save_checkpoint(&checkpoint).unwrap(); // version zero
            let mut sync_seconds = 0.0;
            for step in 1..=6u64 {
                let fragment = ((step - 1) % 2) as usize;
                let mut round = sweep_test_round(step, fragment, step.div_ceil(2), 1, learner);
                let base = state.versions[fragment];
                round.base_version = base;
                for push in round.pushes.values_mut() {
                    push.base_version = base;
                }
                if asynchronous {
                    complete_round_async(&config, &mut state, &registry, &mut sync_seconds, round)
                        .await
                        .unwrap();
                } else {
                    complete_round(&config, &mut state, &registry, &mut sync_seconds, round)
                        .await
                        .unwrap();
                }
                assert_eq!(
                    chunked_inner_type(&frames.try_recv().unwrap()),
                    MSG_BCAST_FRAGMENT
                );
                let mut on_disk = two_fragment_sweep_state();
                on_disk.load_checkpoint(&checkpoint).unwrap();
                // Only step 4 (a sweep boundary and a multiple of 4) is durable.
                let expected = if step >= 4 { 4 } else { 0 };
                assert_eq!(on_disk.global_step, expected, "step {step}");
            }
            // The newest checkpoint resumes exactly at its sweep boundary.
            let mut resumed = two_fragment_sweep_state();
            resumed.load_checkpoint(&checkpoint).unwrap();
            validate_resumed_policy_sweep(&config, &resumed).unwrap();
            assert_eq!(resumed.versions, vec![3, 4]);
        }
        std::fs::remove_dir_all(&directory).ok();
    }

    #[tokio::test]
    async fn async_checkpoint_matches_the_synchronous_path_bit_for_bit() {
        let directory = unique_test_directory("async-checkpoint-equal");
        let learner = member(0, 30);
        let mut outputs = Vec::new();
        for asynchronous in [false, true] {
            let checkpoint = directory.join(format!("state-{asynchronous}.ckpt"));
            let tape = directory.join(format!("tape-{asynchronous}.jsonl"));
            let (group, mut frames) = streaming_test_group(DTYPE_F32, false);
            let registry = registry_with_group(group);
            let mut config = round_test_config(4);
            config.pipeline = 2;
            config.checkpoint_path = Some(checkpoint.clone());
            config.checkpoint_every = 1;
            config.async_checkpoint = asynchronous;
            config.event_tape = Some(tape.clone());
            let mut state = two_fragment_streaming_state();
            let mut sync_seconds = 0.0;
            for step in 1..=2u64 {
                let round = sweep_test_round(step, (step - 1) as usize, 1, step, learner);
                if asynchronous {
                    complete_round_async(&config, &mut state, &registry, &mut sync_seconds, round)
                        .await
                        .unwrap();
                } else {
                    complete_round(&config, &mut state, &registry, &mut sync_seconds, round)
                        .await
                        .unwrap();
                }
                // Every non-terminal round is broadcast exactly once.
                assert_eq!(
                    chunked_inner_type(&frames.try_recv().unwrap()),
                    MSG_BCAST_FRAGMENT
                );
            }
            let tape_text = std::fs::read_to_string(&tape).unwrap();
            assert_eq!(tape_text.lines().count(), 2);
            assert!(tape_text.lines().all(|line| line.contains("\"checkpoint_ms\":")));
            outputs.push((state.params.clone(), std::fs::read(&checkpoint).unwrap()));
            // The snapshot encoder writes exactly the streaming file format.
            let mut on_disk = two_fragment_streaming_state();
            on_disk.load_checkpoint(&checkpoint).unwrap();
            assert_eq!(on_disk.encode_checkpoint().unwrap(), std::fs::read(&checkpoint).unwrap());
        }
        assert_eq!(outputs[0].0, outputs[1].0);
        assert_eq!(outputs[0].1, outputs[1].1);
        std::fs::remove_dir_all(&directory).ok();
    }

    #[tokio::test]
    async fn async_checkpoint_failure_withholds_broadcast_and_keeps_previous_checkpoint() {
        let directory = unique_test_directory("async-checkpoint-failure");
        let checkpoint = directory.join("state.ckpt");
        let learner = member(0, 30);
        let (group, mut frames) = streaming_test_group(DTYPE_F32, false);
        let registry = registry_with_group(group);
        let mut config = round_test_config(4);
        config.pipeline = 2;
        config.checkpoint_path = Some(checkpoint.clone());
        config.checkpoint_every = 1;
        config.async_checkpoint = true;
        let mut state = two_fragment_streaming_state();
        let mut sync_seconds = 0.0;
        complete_round_async(
            &config,
            &mut state,
            &registry,
            &mut sync_seconds,
            sweep_test_round(1, 0, 1, 1, learner),
        )
        .await
        .unwrap();
        assert_eq!(
            chunked_inner_type(&frames.try_recv().unwrap()),
            MSG_BCAST_FRAGMENT
        );
        // The atomic writer creates this exact temporary path; a directory
        // there makes the next write fail deterministically.
        std::fs::create_dir(checkpoint.with_extension("tmp")).unwrap();
        let error = complete_round_async(
            &config,
            &mut state,
            &registry,
            &mut sync_seconds,
            sweep_test_round(2, 1, 1, 1, learner),
        )
        .await
        .unwrap_err();
        assert!(format!("{error:#}").contains("step 2 not broadcast"));
        assert!(matches!(
            frames.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));
        // The resume point is still the last durable, broadcast cut.
        let mut recovered = two_fragment_streaming_state();
        recovered.load_checkpoint(&checkpoint).unwrap();
        assert_eq!(recovered.global_step, 1);
        assert_eq!(recovered.versions, vec![1, 0]);
        std::fs::remove_dir_all(&directory).ok();
    }

    // --checkpoint-at-steps (crate::eval_checkpoint).

    /// A fragment round at `step` of a two-fragment state, on the fragment's
    /// current version.
    fn eval_test_round(state: &GlobalState, step: u64, learner: Member) -> Round {
        let fragment = ((step - 1) % 2) as usize;
        let mut round = sweep_test_round(step, fragment, 1, step, learner);
        round.base_version = state.versions[fragment];
        for push in round.pushes.values_mut() {
            push.base_version = round.base_version;
        }
        round
    }

    /// Holds every eval write before it starts: the writer reports its step
    /// on the returned receiver, and one message on the sender lets it go.
    fn eval_write_gate() -> (BeforeWrite, std::sync::mpsc::Receiver<u64>, std::sync::mpsc::Sender<()>) {
        let (entered, writes) = std::sync::mpsc::channel();
        let (release, released) = std::sync::mpsc::channel::<()>();
        let released = Mutex::new(released);
        let gate: BeforeWrite = Arc::new(move |step| {
            entered.send(step).unwrap();
            released.lock().unwrap().recv().unwrap();
        });
        (gate, writes, release)
    }

    fn eval_test_config(checkpoint: &std::path::Path, steps: &[u64]) -> Config {
        let mut config = round_test_config(6);
        config.checkpoint_path = Some(checkpoint.to_path_buf());
        config.checkpoint_every = 8;
        config.checkpoint_at_steps = steps.to_vec();
        validate_config(&config).unwrap();
        config
    }

    const WRITER_WAIT: Duration = Duration::from_secs(10);

    #[tokio::test]
    async fn eval_checkpoint_holds_its_step_while_later_rounds_merge_and_broadcast() {
        let directory = unique_test_directory("eval-checkpoint-cut");
        let checkpoint = directory.join("state.ckpt");
        let config = eval_test_config(&checkpoint, &[2]);
        let learner = member(0, 30);
        let (group, mut frames) = streaming_test_group(DTYPE_F32, false);
        let registry = registry_with_group(group);
        let mut state = two_fragment_streaming_state();
        let (gate, writes, release) = eval_write_gate();
        let mut evals = EvalCheckpoints::new(&config.checkpoint_at_steps, Some(&checkpoint), 0)
            .unwrap()
            .with_before_write(gate);
        let synchronous = directory.join("synchronous-at-2.ckpt");
        let mut sync_seconds = 0.0;
        for step in 1..=4u64 {
            let round = eval_test_round(&state, step, learner);
            complete_round(&config, &mut state, &registry, &mut sync_seconds, round).await.unwrap();
            if step == 2 {
                state.save_checkpoint(&synchronous).unwrap();
            }
            evals.after_commit(&state).await.unwrap();
            if step == 2 {
                assert_eq!(writes.recv_timeout(WRITER_WAIT).unwrap(), 2);
            }
            // Steps 3 and 4 merge and broadcast while the copy of step 2 is
            // still being written.
            assert_eq!(chunked_inner_type(&frames.try_recv().unwrap()), MSG_BCAST_FRAGMENT);
            assert_eq!(state.global_step, step);
            assert_eq!(evals.is_writing(), step >= 2);
        }
        let written = eval_checkpoint_path(&checkpoint, 2);
        assert_eq!(written, directory.join("state.ckpt.step-2"));
        assert!(!written.exists(), "the file appears only once it is complete");
        release.send(()).unwrap();
        evals.finish().await.unwrap();
        assert!(!evals.is_writing());
        assert_eq!(std::fs::read(&written).unwrap(), std::fs::read(&synchronous).unwrap());
        assert_ne!(state.encode_checkpoint().unwrap(), std::fs::read(&synchronous).unwrap());
        let mut restored = two_fragment_streaming_state();
        restored.load_checkpoint(&written).unwrap();
        assert_eq!((restored.global_step, restored.versions.clone()), (2, vec![1, 2]));
        assert!(!directory.join("state.ckpt.step-2.tmp").exists());
        assert!(!checkpoint.exists(), "an eval copy never writes --checkpoint-path");
        std::fs::remove_dir_all(&directory).ok();
    }

    #[tokio::test]
    async fn one_eval_write_at_a_time_and_the_terminal_waits_for_it() {
        let directory = unique_test_directory("eval-checkpoint-serial");
        let checkpoint = directory.join("state.ckpt");
        let config = eval_test_config(&checkpoint, &[1, 2]);
        let learner = member(0, 30);
        let (group, _frames) = streaming_test_group(DTYPE_F32, false);
        let registry = registry_with_group(group);
        let mut state = two_fragment_streaming_state();
        let (gate, writes, release) = eval_write_gate();
        let mut evals = EvalCheckpoints::new(&config.checkpoint_at_steps, Some(&checkpoint), 0)
            .unwrap()
            .with_before_write(gate);
        let mut sync_seconds = 0.0;
        let mut cuts = Vec::new();
        for step in 1..=2u64 {
            let round = eval_test_round(&state, step, learner);
            complete_round(&config, &mut state, &registry, &mut sync_seconds, round).await.unwrap();
            cuts.push(state.encode_checkpoint().unwrap());
            if step == 2 {
                // Step 2 is due while step 1 is still being written: it waits.
                assert!(evals.is_writing());
                let releaser = std::thread::spawn({
                    let release = release.clone();
                    move || {
                        std::thread::sleep(Duration::from_millis(200));
                        let at = Instant::now();
                        release.send(()).unwrap();
                        at
                    }
                });
                evals.after_commit(&state).await.unwrap();
                assert!(Instant::now() >= releaser.join().unwrap());
                assert!(eval_checkpoint_path(&checkpoint, 1).exists());
            } else {
                evals.after_commit(&state).await.unwrap();
            }
            assert_eq!(writes.recv_timeout(WRITER_WAIT).unwrap(), step);
        }
        // The terminal cut waits for the running write of step 2.
        let releaser = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(200));
            let at = Instant::now();
            release.send(()).unwrap();
            at
        });
        evals.finish().await.unwrap();
        assert!(Instant::now() >= releaser.join().unwrap());
        for (step, cut) in [1u64, 2].into_iter().zip(&cuts) {
            assert_eq!(&std::fs::read(eval_checkpoint_path(&checkpoint, step)).unwrap(), cut);
        }
        // Nothing is left to wait for.
        evals.finish().await.unwrap();
        std::fs::remove_dir_all(&directory).ok();
    }

    #[tokio::test]
    async fn resume_reads_only_the_checkpoint_path_and_rewrites_eval_steps_after_its_cut() {
        let directory = unique_test_directory("eval-checkpoint-resume");
        let checkpoint = directory.join("state.ckpt");
        let mut config = eval_test_config(&checkpoint, &[1, 3]);
        config.resume = true;
        let learner = member(0, 30);
        let (group, _frames) = streaming_test_group(DTYPE_F32, false);
        let registry = registry_with_group(group);
        let mut sync_seconds = 0.0;
        let mut before_crash = two_fragment_streaming_state();
        for step in 1..=2u64 {
            let round = eval_test_round(&before_crash, step, learner);
            complete_round(&config, &mut before_crash, &registry, &mut sync_seconds, round)
                .await
                .unwrap();
        }
        before_crash.save_checkpoint(&checkpoint).unwrap();
        // Stray eval files: an older step, the step the resumed run redoes,
        // and an unfinished temporary.
        let stale = b"not a checkpoint".as_slice();
        for path in [
            eval_checkpoint_path(&checkpoint, 1),
            eval_checkpoint_path(&checkpoint, 3),
            directory.join("state.ckpt.step-3.tmp"),
        ] {
            std::fs::write(path, stale).unwrap();
        }
        // None of the files the syncer reads next to the checkpoint is one.
        for sidecar in [
            checkpoint.with_extension("tmp"),
            profile_record_path(&checkpoint),
            membership_record_path(&checkpoint),
            crate::state::final_marker_path(&checkpoint),
        ] {
            assert!(!sidecar.to_string_lossy().contains(".step-"), "{}", sidecar.display());
        }
        check_resume_checkpoint(&config).unwrap();
        let mut resumed = two_fragment_streaming_state();
        resumed.load_checkpoint(config.checkpoint_path.as_ref().unwrap()).unwrap();
        assert_eq!(resumed.encode_checkpoint().unwrap(), before_crash.encode_checkpoint().unwrap());
        // Step 1 is behind the resumed cut (skipped, its file untouched);
        // step 3 is written again over the stale file.
        let mut evals =
            EvalCheckpoints::new(&config.checkpoint_at_steps, Some(&checkpoint), resumed.global_step)
                .unwrap();
        let round = eval_test_round(&resumed, 3, learner);
        complete_round(&config, &mut resumed, &registry, &mut sync_seconds, round).await.unwrap();
        evals.after_commit(&resumed).await.unwrap();
        evals.finish().await.unwrap();
        assert_eq!(std::fs::read(eval_checkpoint_path(&checkpoint, 1)).unwrap(), stale);
        assert_eq!(
            std::fs::read(eval_checkpoint_path(&checkpoint, 3)).unwrap(),
            resumed.encode_checkpoint().unwrap()
        );
        assert!(!directory.join("state.ckpt.step-3.tmp").exists());
        std::fs::remove_dir_all(&directory).ok();
    }

    #[test]
    fn eval_checkpoint_steps_are_checked_and_unsupported_profiles_refused() {
        let error = |config: &Config| format!("{:#}", validate_config(config).unwrap_err());
        let eval = |steps: &[u64]| {
            let mut config = round_test_config(10);
            config.checkpoint_path = Some(std::path::PathBuf::from("state.ckpt"));
            config.checkpoint_at_steps = steps.to_vec();
            config
        };
        // Any periodic interval but 1, including none, takes eval steps.
        for every in [0, 2, 8, 1000] {
            let mut config = eval(&[1, 4, 9]);
            config.checkpoint_every = every;
            validate_config(&config).unwrap();
        }
        let mut config = eval(&[3]);
        config.checkpoint_path = None;
        assert!(error(&config).contains("requires --checkpoint-path"));
        for steps in [vec![0, 3], vec![3, 3], vec![7, 3]] {
            assert!(error(&eval(&steps)).contains("positive and strictly increasing"), "{steps:?}");
        }
        assert!(error(&eval(&[3, 10])).contains("below --total-steps 10"));
        let mut config = eval(&[3]);
        config.checkpoint_every = 1;
        assert!(error(&config).contains("does not support --checkpoint-every 1"));
        let mut config = sweep_test_config(2, 10);
        config.checkpoint_at_steps = vec![4];
        assert!(error(&config).contains("does not support --policy-sweep-fragments"));
        let mut config = eval(&[3]);
        config.shard = crate::shard::ShardPlan::new(0, 2).unwrap();
        assert!(error(&config).contains("does not support --shard-count > 1"));
        // Without steps nothing changes.
        validate_config(&eval(&[])).unwrap();
    }

    const EVAL_CHECKPOINTS_BASE_VECTOR: &str = "bcbef730a6bc46e6c1901e90b6ce5a5382b035292863ed5cfea0285da408cafe";
    const EVAL_CHECKPOINTS_VECTOR: &str = "2e395d5104892c355f6cd2609bcf3f5011a26407848fb805b5e262bf912584f3";

    #[test]
    fn eval_checkpoint_steps_append_to_the_profile_hash_like_python() {
        // Python: tests/test_s27_eval_checkpoints.py.
        let mut config = round_test_config(8);
        config.learners = 2;
        config.quorum = 2;
        config.grace_gamma = 0.8;
        config.grace_tau = 2.0;
        config.pipeline = 2;
        config.sync_interval_steps = 24.0;
        config.quorum_timeout_s = 900;
        config.final_ack_timeout_s = 900;
        config.outer_lr = 0.7;
        config.outer_momentum = 0.9;
        config.checkpoint_path = Some(std::path::PathBuf::from("/ignored/actor.ckpt"));
        config.checkpoint_every = 8;
        config.mark_final_checkpoint = true;
        config.require_profile_binding = true;
        let hex = |config: &Config| -> String {
            config.semantic_profile_hash().iter().map(|byte| format!("{byte:02x}")).collect()
        };
        assert_eq!(hex(&config), EVAL_CHECKPOINTS_BASE_VECTOR);
        config.checkpoint_at_steps = vec![3, 5];
        assert_eq!(hex(&config), EVAL_CHECKPOINTS_VECTOR);
        config.checkpoint_at_steps = vec![3, 6];
        assert_ne!(hex(&config), EVAL_CHECKPOINTS_VECTOR);
    }

    #[tokio::test]
    async fn policy_sweep_checkpoint_resume_mid_sweep_does_not_double_account() {
        let directory = std::env::temp_dir().join(format!(
            "yeto-policy-sweep-resume-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let checkpoint = directory.join("state.ckpt");
        let learner = member(0, 30);
        let (group, mut frames) = streaming_test_group(DTYPE_F32, false);
        let registry = registry_with_group(group);
        let mut config = sweep_test_config(2, 2);
        config.checkpoint_path = Some(checkpoint.clone());
        config.checkpoint_every = 1;
        let mut sync_seconds = 0.0;

        let mut before_crash = two_fragment_sweep_state();
        complete_round(
            &config,
            &mut before_crash,
            &registry,
            &mut sync_seconds,
            sweep_test_round(1, 0, 1, 321, learner),
        )
        .await
        .unwrap();
        assert_eq!(before_crash.global_step, 1);
        assert_eq!(before_crash.versions, vec![1, 0]);
        let partial = before_crash.ledger.get(&0).unwrap();
        assert_eq!((partial.merges, partial.steps, partial.tokens), (1, 0, 0));
        assert!(checkpoint.is_file());
        assert_eq!(
            chunked_inner_type(&frames.try_recv().unwrap()),
            MSG_BCAST_FRAGMENT
        );

        // Simulate a process crash after the first fragment's durable commit.
        let mut resumed = two_fragment_sweep_state();
        resumed.load_checkpoint(&checkpoint).unwrap();
        validate_resumed_policy_sweep(&config, &resumed).unwrap();
        assert_eq!(resumed.global_step, 1);
        assert_eq!(resumed.versions, vec![1, 0]);
        let partial = resumed.ledger.get(&0).unwrap();
        assert_eq!((partial.merges, partial.steps, partial.tokens), (1, 0, 0));

        complete_round(
            &config,
            &mut resumed,
            &registry,
            &mut sync_seconds,
            sweep_test_round(2, 1, 1, 321, learner),
        )
        .await
        .unwrap();
        let complete = resumed.ledger.get(&0).unwrap();
        assert_eq!(
            (complete.merges, complete.steps, complete.tokens),
            (2, 1, 321)
        );
        assert_eq!(resumed.global_step, 2);
        assert_eq!(resumed.versions, vec![1, 2]);

        let mut final_reload = two_fragment_sweep_state();
        final_reload.load_checkpoint(&checkpoint).unwrap();
        validate_resumed_policy_sweep(&config, &final_reload).unwrap();
        let complete = final_reload.ledger.get(&0).unwrap();
        assert_eq!(
            (complete.merges, complete.steps, complete.tokens),
            (2, 1, 321)
        );
        assert_eq!(final_reload.params, resumed.params);
        std::fs::remove_dir_all(&directory).ok();
    }

    #[tokio::test]
    async fn two_learner_three_fragment_sweeps_resume_at_every_mid_sweep_cut_exactly() {
        let directory = std::env::temp_dir().join(format!(
            "yeto-policy-sweep-matrix-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let registry = registry_with_group(test_group(member(0, 30)));
        let mut config = sweep_test_config(3, 6);
        config.learners = 2;
        config.quorum = 2;
        let mut sync_seconds = 0.0;

        let uninterrupted_path = directory.join("uninterrupted.ckpt");
        config.checkpoint_path = Some(uninterrupted_path.clone());
        let mut uninterrupted = three_fragment_sweep_state();
        for step in 1..=6 {
            complete_round(
                &config,
                &mut uninterrupted,
                &registry,
                &mut sync_seconds,
                two_learner_sweep_round(step),
            )
            .await
            .unwrap();
        }
        assert_eq!(uninterrupted.versions, vec![4, 5, 6]);
        assert!(uninterrupted.params.iter().all(|p| p.as_slice() == [-5.0]));
        for learner_id in 0..2 {
            let ledger = uninterrupted.ledger.get(&learner_id).unwrap();
            assert_eq!(ledger.merges, 6);
            assert_eq!(ledger.steps, 2);
            assert_eq!(ledger.tokens, u64::from(learner_id + 1) * (100 + 200));
        }
        let uninterrupted_bytes = std::fs::read(&uninterrupted_path).unwrap();

        for crash_step in [1, 2, 4, 5] {
            let checkpoint = directory.join(format!("resume-{crash_step}.ckpt"));
            config.checkpoint_path = Some(checkpoint.clone());
            let mut before_crash = three_fragment_sweep_state();
            for step in 1..=crash_step {
                complete_round(
                    &config,
                    &mut before_crash,
                    &registry,
                    &mut sync_seconds,
                    two_learner_sweep_round(step),
                )
                .await
                .unwrap();
            }
            for learner_id in 0..2 {
                let ledger = before_crash.ledger.get(&learner_id).unwrap();
                let completed_sweeps = crash_step / 3;
                let expected_tokens =
                    u64::from(learner_id + 1) * 100 * (1..=completed_sweeps).sum::<u64>();
                assert_eq!(ledger.merges, crash_step);
                assert_eq!(ledger.steps, completed_sweeps);
                assert_eq!(ledger.tokens, expected_tokens);
            }

            let mut resumed = three_fragment_sweep_state();
            resumed.load_checkpoint(&checkpoint).unwrap();
            validate_resumed_policy_sweep(&config, &resumed).unwrap();
            for step in crash_step + 1..=6 {
                complete_round(
                    &config,
                    &mut resumed,
                    &registry,
                    &mut sync_seconds,
                    two_learner_sweep_round(step),
                )
                .await
                .unwrap();
            }
            validate_resumed_policy_sweep(&config, &resumed).unwrap();
            assert_eq!(std::fs::read(&checkpoint).unwrap(), uninterrupted_bytes);
        }
        std::fs::remove_dir_all(&directory).ok();
    }

    #[tokio::test]
    async fn updated_broadcast_propagates_local_errors_but_tolerates_closed_streams() {
        let invalid_state = broadcast_test_state(9);
        let (connected, _receiver) = streaming_test_group(DTYPE_F32, false);
        let error = broadcast_updated_fragment(&invalid_state, &registry_with_group(connected), 0)
            .await
            .unwrap_err();
        assert!(
            format!("{error:#}").contains("unknown tensor dtype 9"),
            "unexpected error: {error:#}"
        );

        let valid_state = broadcast_test_state(DTYPE_F32);
        let disconnected = test_group(member(0, 20));
        broadcast_updated_fragment(&valid_state, &registry_with_group(disconnected), 0)
            .await
            .unwrap();
    }

    #[test]
    fn duplicate_generation_registration_is_atomic_and_non_destructive() {
        let first = member(0, 10);
        let replacement = member(0, 11);
        let first_group = test_group(first);
        let mut registry = RegistryState::default();

        let now = Instant::now();
        assert_eq!(registry.register_group(first_group.clone(), now).map(|r| r.0), Ok(None));
        assert_eq!(
            registry.register_group(test_group(first), now).map(|r| r.0),
            Err(RegisterError::Duplicate)
        );
        assert!(Arc::ptr_eq(
            registry.groups.get(&first).unwrap(),
            &first_group
        ));
        assert_eq!(
            registry.register_group(test_group(replacement), now).map(|r| r.0),
            Ok(Some(first))
        );
        assert_eq!(registry.current.get(&0), Some(&replacement));
        assert!(registry.groups.contains_key(&first));
        assert!(registry.groups.contains_key(&replacement));
    }

    #[test]
    fn budget_reports_are_exact_idempotent_and_cover_logical_learners() {
        let mut reports = HashSet::new();
        record_budget_report(&mut reports, 8, member(0, 10), 8).unwrap();
        record_budget_report(&mut reports, 8, member(1, 20), 8).unwrap();
        assert_eq!(reports.len(), 2);
        record_budget_report(&mut reports, 8, member(1, 21), 8).unwrap();
        assert_eq!(reports.len(), 2);

        let mut wrong = HashSet::new();
        assert!(record_budget_report(&mut wrong, 8, member(0, 10), 7).is_err());
    }

    #[test]
    fn queued_budget_report_blocks_every_later_scheduler_action() {
        let cutoff = BudgetCutoff::new(true, Duration::from_secs(30));
        assert!(cutoff.try_linearize_work());

        // request() runs before dispatch queues Event::BudgetDone.  Repeating
        // it is idempotent and preserves the original absolute deadline.
        let deadline = cutoff.request().unwrap();
        assert_eq!(cutoff.request(), Some(deadline));

        // These three claims represent commit, pull/retry, and compute submit.
        // All use this same method in the scheduler and all must now fail.
        assert!(!cutoff.try_linearize_work());
        assert!(!cutoff.try_linearize_work());
        assert!(!cutoff.try_linearize_work());
    }

    #[tokio::test]
    async fn queued_budget_event_wins_simultaneous_compute_completion() {
        let cutoff = BudgetCutoff::new(true, Duration::from_secs(30));
        cutoff.request();
        let event = Some(Event::BudgetDone {
            member: member(0, 10),
            local_steps: 8,
        });
        // Both futures are Ready on their first poll.  An inner error is still
        // a completed scheduler compute task and avoids manufacturing private
        // ComputedMerge state solely for this arbitration test.
        let compute: ComputeJoin = Some(Ok(Err(anyhow::anyhow!(
            "synthetic ready compute completion"
        ))));

        let wake =
            prefer_event_over_compute(std::future::ready(event), std::future::ready(compute)).await;
        assert!(matches!(
            wake,
            SchedulerWake::Event(Some(Event::BudgetDone {
                member: Member {
                    learner_id: 0,
                    generation: 10
                },
                local_steps: 8
            }))
        ));
        assert!(!cutoff.try_linearize_work());
    }

    #[tokio::test]
    async fn cutoff_notification_interrupts_a_backpressured_pull_send() {
        let (control, _receiver) = mpsc::channel(1);
        assert!(control
            .try_send(OutFrame {
                msg_type: MSG_PULL_REQ,
                parts: vec![bytes::Bytes::new()],
            })
            .is_ok());
        let group = test_group_with_control(member(0, 10), control);
        let cutoff = Arc::new(BudgetCutoff::new(true, Duration::from_secs(30)));
        let task_group = group.clone();
        let task_cutoff = cutoff.clone();
        let send = tokio::spawn(async move {
            send_small_until_cutoff(&task_group, MSG_PULL_REQ, bytes::Bytes::new(), &task_cutoff)
                .await
        });
        tokio::task::yield_now().await;
        cutoff.request();
        let queued = tokio::time::timeout(Duration::from_secs(1), send)
            .await
            .expect("backpressured pull did not observe cutoff")
            .expect("pull send task panicked");
        assert!(!queued);
    }

    #[tokio::test]
    async fn budget_report_collection_times_out_on_missing_progress() {
        let (_sender, mut events) = mpsc::channel(1);
        let registry: Registry = Arc::new(Mutex::new(RegistryState::default()));
        let mut reports = HashSet::new();
        let result = collect_budget_reports(
            8,
            2,
            &mut reports,
            &mut events,
            &registry,
            Duration::from_millis(1),
            None,
            None,
        )
        .await;
        let error = match result {
            Ok(_) => panic!("pending report collection unexpectedly succeeded"),
            Err(error) => error,
        };
        let message = format!("{error:#}");
        assert!(message.contains("timed out"), "{message}");
        assert!(message.contains("no progress for [0, 1]"), "{message}");
        assert!(message.contains("received 0/2"), "{message}");
    }

    #[tokio::test]
    async fn heartbeat_renews_only_the_reporting_learners_progress_lease() {
        let (sender, mut events) = mpsc::channel(4);
        let completed = member(0, 10);
        let current = member(1, 20);
        let registry: Registry = Arc::new(Mutex::new(RegistryState::default()));
        {
            let mut registry = registry.lock().unwrap();
            registry.current.insert(0, completed);
            registry.current.insert(1, current);
        }
        let mut reports = HashSet::new();
        let collect = collect_budget_reports(
            8,
            2,
            &mut reports,
            &mut events,
            &registry,
            Duration::from_millis(40),
            None,
            Some((completed, 8)),
        );
        let produce = async move {
            tokio::time::sleep(Duration::from_millis(25)).await;
            sender
                .send(Event::Heartbeat {
                    member: current,
                    local_step: 7,
                })
                .await
                .unwrap();
            tokio::time::sleep(Duration::from_millis(25)).await;
            sender
                .send(Event::BudgetDone {
                    member: current,
                    local_steps: 8,
                })
                .await
                .unwrap();
        };
        let (result, ()) = tokio::join!(collect, produce);
        result.unwrap();
        assert_eq!(reports, HashSet::from([0, 1]));
    }

    #[test]
    fn final_ack_requires_an_eligible_generation_and_remains_permanent() {
        let first = member(0, 10);
        let ineligible = member(0, 11);
        let reconnected = member(1, 21);
        let expected = HashSet::from([0, 1]);
        let eligible = HashMap::from([
            (0, HashSet::from([first])),
            (1, HashSet::from([member(1, 20), reconnected])),
        ]);
        let mut acknowledged = HashMap::new();

        assert_eq!(
            record_final_ack(&expected, &eligible, &mut acknowledged, first, 7, 7),
            FinalAckDisposition::Accepted
        );
        assert_eq!(
            record_final_ack(&expected, &eligible, &mut acknowledged, ineligible, 7, 7,),
            FinalAckDisposition::IneligibleGeneration
        );
        // The first ACK stays accepted even if its socket later disconnects;
        // a reconnect generation explicitly sent the cut can satisfy ID 1.
        assert_eq!(acknowledged.get(&0), Some(&first));
        assert_eq!(
            record_final_ack(&expected, &eligible, &mut acknowledged, reconnected, 7, 7,),
            FinalAckDisposition::Accepted
        );
        assert_eq!(acknowledged, HashMap::from([(0, first), (1, reconnected)]));
    }

    #[tokio::test]
    async fn final_ack_after_quorum_timeout_but_before_final_timeout_succeeds() {
        let state = broadcast_test_state(DTYPE_F32);
        let (group, _frames) = streaming_test_group(DTYPE_F32, false);
        let registry = registry_with_group(group.clone());
        let expected = HashSet::from([group.member.learner_id]);
        let (event_sender, mut events) = mpsc::channel(1);
        let delayed_member = group.member;
        let delayed_step = state.global_step;
        let sender = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(1_100)).await;
            event_sender
                .send(Event::FinalAck {
                    member: delayed_member,
                    global_step: delayed_step,
                })
                .await
                .unwrap();
        });
        let mut cfg = round_test_config(0);
        cfg.quorum_timeout_s = 1;
        cfg.final_ack_timeout_s = 3;
        let started = Instant::now();

        finalize_learners(&cfg, &state, &mut events, &registry, &expected)
            .await
            .unwrap();
        sender.await.unwrap();

        assert!(started.elapsed() >= Duration::from_millis(1_000));
        assert!(started.elapsed() < Duration::from_secs(3));
    }

    #[tokio::test]
    async fn missing_final_ack_fails_at_final_timeout() {
        let state = broadcast_test_state(DTYPE_F32);
        let (group, _frames) = streaming_test_group(DTYPE_F32, false);
        let registry = registry_with_group(group.clone());
        let expected = HashSet::from([group.member.learner_id]);
        let (_event_sender, mut events) = mpsc::channel(1);
        let mut cfg = round_test_config(0);
        cfg.quorum_timeout_s = 3;
        cfg.final_ack_timeout_s = 1;
        let started = Instant::now();

        let error = finalize_learners(&cfg, &state, &mut events, &registry, &expected)
            .await
            .unwrap_err();

        assert!(
            format!("{error:#}").contains(
                "finalization timed out after 1s waiting for learner acknowledgements: [0]"
            ),
            "unexpected error: {error:#}"
        );
        assert!(started.elapsed() >= Duration::from_millis(900));
        assert!(started.elapsed() < Duration::from_secs(3));
    }

    #[tokio::test]
    async fn final_grace_finishes_without_a_silent_island_and_tapes_it() {
        let state = broadcast_test_state(DTYPE_F32);
        let (group, _frames) = streaming_test_group(DTYPE_F32, false);
        let registry = registry_with_group(group.clone());
        let expected = HashSet::from([group.member.learner_id]);
        let (_event_sender, mut events) = mpsc::channel(1);
        let dir = std::env::temp_dir().join(format!("yeto-final-grace-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let tape_path = dir.join("tape.jsonl");
        let _ = std::fs::remove_file(&tape_path);
        let mut cfg = round_test_config(0);
        cfg.final_ack_timeout_s = 30;
        cfg.membership.final_grace = Some(Duration::from_millis(300));
        cfg.event_tape = Some(tape_path.clone());
        let started = Instant::now();

        finalize_learners(&cfg, &state, &mut events, &registry, &expected)
            .await
            .expect("the grace window ends the run without the silent island");

        assert!(started.elapsed() >= Duration::from_millis(250));
        assert!(started.elapsed() < Duration::from_secs(5), "bounded by the grace, not the ack timeout");
        let tape = std::fs::read_to_string(&tape_path).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert!(tape.contains("\"kind\":\"final_grace_expired\""), "{tape}");
        assert!(tape.contains("\"missing\":[0]") && tape.contains("\"acknowledged\":[]"), "{tape}");
    }

    #[tokio::test]
    async fn final_grace_stops_waiting_for_an_island_that_left() {
        let state = broadcast_test_state(DTYPE_F32);
        let (group, _frames) = streaming_test_group(DTYPE_F32, false);
        let registry = registry_with_group(group.clone());
        let expected = HashSet::from([group.member.learner_id]);
        let (event_sender, mut events) = mpsc::channel(1);
        let member = group.member;
        registry.lock().unwrap().unregister(member);
        event_sender.send(Event::Disconnected { member }).await.unwrap();
        let mut cfg = round_test_config(0);
        cfg.membership.final_grace = Some(Duration::from_secs(30));
        let started = Instant::now();

        finalize_learners(&cfg, &state, &mut events, &registry, &expected).await.unwrap();

        assert!(started.elapsed() < Duration::from_secs(5), "a departed island is not waited for");
    }

    fn final_leased_registry(group: Arc<Group>, lease: Duration) -> (Registry, Config) {
        let mut cfg = round_test_config(0);
        cfg.learners = 2;
        cfg.membership.lease = Some(lease);
        let membership = Membership::new(cfg.membership.clone(), 2);
        let registry = Arc::new(Mutex::new(RegistryState::new(membership)));
        registry.lock().unwrap().register_group(group, Instant::now()).unwrap();
        (registry, cfg)
    }

    #[tokio::test]
    async fn final_cut_does_not_wait_for_an_absent_leased_learner() {
        let state = broadcast_test_state(DTYPE_F32);
        let (group, _frames) = streaming_test_group(DTYPE_F32, false);
        let (registry, mut cfg) = final_leased_registry(group.clone(), Duration::from_millis(300));
        cfg.final_ack_timeout_s = 5;
        // Learner 1 is expected but never connects; learner 0 acknowledges.
        let expected = HashSet::from([0, 1]);
        let (event_sender, mut events) = mpsc::channel(4);
        event_sender
            .send(Event::FinalAck { member: group.member, global_step: state.global_step })
            .await
            .unwrap();
        let started = Instant::now();

        finalize_learners(&cfg, &state, &mut events, &registry, &expected).await.unwrap();

        assert!(started.elapsed() >= Duration::from_millis(250));
        assert!(started.elapsed() < Duration::from_secs(3));
    }

    #[tokio::test]
    async fn final_cut_drops_a_learner_whose_lease_expires() {
        let state = broadcast_test_state(DTYPE_F32);
        let (group, _frames) = streaming_test_group(DTYPE_F32, false);
        let (registry, mut cfg) = final_leased_registry(group.clone(), Duration::from_millis(300));
        cfg.final_ack_timeout_s = 5;
        let expected = HashSet::from([0]);
        let (_event_sender, mut events) = mpsc::channel(1);

        let error = finalize_learners(&cfg, &state, &mut events, &registry, &expected)
            .await
            .unwrap_err();

        assert!(format!("{error:#}").contains("every learner's lease expired"), "{error:#}");
    }

    #[tokio::test]
    async fn final_cut_heartbeats_keep_a_slow_learner_expected() {
        let state = broadcast_test_state(DTYPE_F32);
        let (group, _frames) = streaming_test_group(DTYPE_F32, false);
        let (registry, mut cfg) = final_leased_registry(group.clone(), Duration::from_millis(300));
        cfg.final_ack_timeout_s = 5;
        let expected = HashSet::from([0]);
        let (event_sender, mut events) = mpsc::channel(4);
        let member = group.member;
        let step = state.global_step;
        let sender = tokio::spawn(async move {
            for _ in 0..4 {
                tokio::time::sleep(Duration::from_millis(150)).await;
                event_sender.send(Event::Heartbeat { member, local_step: 0 }).await.unwrap();
            }
            event_sender.send(Event::FinalAck { member, global_step: step }).await.unwrap();
        });

        finalize_learners(&cfg, &state, &mut events, &registry, &expected).await.unwrap();
        sender.await.unwrap();
    }

    #[test]
    fn startup_waits_for_all_learners_only_with_static_membership() {
        let mut cfg = round_test_config(0);
        cfg.learners = 4;
        cfg.quorum = 2;
        assert_eq!(startup_members(&cfg), 4);
        cfg.membership.lease = Some(Duration::from_secs(30));
        assert_eq!(startup_members(&cfg), 2);
        cfg.membership.quorum_rule = crate::membership::QuorumRule::Capacity { theta: 0.5, q_min: 3 };
        assert_eq!(startup_members(&cfg), 3);
    }

    #[test]
    fn grace_falls_back_to_cap_without_estimate() {
        assert_eq!(adaptive_grace(2.0, 0.8, None, 0.1, 0.1, CAP), CAP);
    }

    #[test]
    fn grace_uses_gamma_scaled_slack() {
        // slack = 2·1.0 − 0.5 − 0.5 = 1.0s; γ=0.8 → 800ms, under the 1s cap.
        let g = adaptive_grace(2.0, 0.8, Some(1.0), 0.5, 0.5, CAP);
        assert!((g.as_secs_f64() - 0.8).abs() < 1e-9, "got {g:?}");
    }

    #[test]
    fn grace_clamps_to_zero_and_cap() {
        // Negative slack → no grace.
        assert_eq!(
            adaptive_grace(2.0, 0.8, Some(0.1), 1.0, 1.0, CAP),
            Duration::ZERO
        );
        // Huge slack → capped.
        assert_eq!(adaptive_grace(2.0, 0.8, Some(60.0), 0.0, 0.0, CAP), CAP);
    }

    #[test]
    fn arrival_gaps_measure_joined_and_late_pushes() {
        let start = Instant::now();
        let mut gaps = ArrivalGaps::default();
        assert_eq!(gaps.quantile(0.9), None);
        gaps.note_accepted(9, 8, start); // first push starts the clock
        gaps.note_accepted(9, 8, start + Duration::from_secs(4)); // joined the round
        gaps.note_late(9, 8, start + Duration::from_secs(16)); // after the round closed
        gaps.note_late(10, 9, start + Duration::from_secs(30)); // round never seen: no gap
        assert_eq!(gaps.quantile(0.5), Some(4.0));
        assert_eq!(gaps.quantile(0.9), Some(16.0));
        assert_eq!(gaps.quantile(1.0), Some(16.0));
        // A relaunched attempt restarts its clock.
        gaps.forget(9, 8);
        gaps.note_accepted(9, 8, start + Duration::from_secs(40));
        gaps.note_accepted(9, 8, start + Duration::from_secs(41));
        assert_eq!(gaps.quantile(0.0), Some(1.0));
    }

    #[test]
    fn arrival_gaps_keep_a_bounded_window() {
        let start = Instant::now();
        let mut gaps = ArrivalGaps::default();
        for step in 0..(ARRIVAL_GAP_WINDOW as u64 + 10) {
            gaps.note_accepted(step, 0, start);
            gaps.note_accepted(step, 0, start + Duration::from_secs(step));
        }
        assert_eq!(gaps.gaps.len(), ARRIVAL_GAP_WINDOW);
        assert_eq!(gaps.quantile(0.0), Some(10.0)); // the 10 oldest gaps were dropped
        for step in 0..(ARRIVAL_FIRSTS_KEPT as u64 + 5) {
            gaps.note_accepted(1_000 + step, 1, start);
        }
        assert_eq!(gaps.firsts.len(), ARRIVAL_FIRSTS_KEPT);
        assert!(!gaps.firsts.contains_key(&(1_000, 1))); // oldest first arrival forgotten
    }

    #[test]
    fn terminal_rounds_are_the_last_round_of_each_fragment() {
        let unsharded = crate::shard::ShardPlan::default().schedule(4).unwrap();
        let terminal: Vec<u64> =
            (1..=12).filter(|&t| is_terminal_round(&unsharded, t, 12)).collect();
        assert_eq!(terminal, vec![9, 10, 11, 12]);
        // A total that is not a multiple of P: fragment 3 last merges at step 8.
        let terminal: Vec<u64> =
            (1..=10).filter(|&t| is_terminal_round(&unsharded, t, 10)).collect();
        assert_eq!(terminal, vec![7, 8, 9, 10]);
        // Shard 1 of 2 owns fragments 2..4 and runs steps 3, 4, 7, 8, 11, 12.
        let shard = crate::shard::ShardPlan::new(1, 2).unwrap().schedule(4).unwrap();
        let terminal: Vec<u64> = (1..=12)
            .filter(|&t| shard.owns_step(t) && is_terminal_round(&shard, t, 12))
            .collect();
        assert_eq!(terminal, vec![11, 12]);
    }

    #[test]
    fn terminal_grace_holds_a_last_round_past_the_cap_and_is_off_by_default() {
        let gaps = ArrivalGaps::default();
        let mut config = round_test_config(8);
        config.grace_ms = 30_000;
        config.grace_gamma = 0.8;
        config.grace_tau = 2.0;
        // No slack: the adaptive window is 0 s for a terminal round too when off.
        assert_eq!(round_grace(&config, true, Some(1.0), 5.0, 1.0, &gaps).0, Duration::ZERO);
        config.terminal_grace_ms = 120_000;
        assert_eq!(
            round_grace(&config, true, Some(1.0), 5.0, 1.0, &gaps),
            (Duration::from_secs(120), Duration::from_secs(120))
        );
        // Not a terminal round: unchanged.
        assert_eq!(round_grace(&config, false, Some(1.0), 5.0, 1.0, &gaps).0, Duration::ZERO);
    }

    #[test]
    fn grace_floor_takes_the_larger_bound_and_stays_off_by_default() {
        let start = Instant::now();
        let mut gaps = ArrivalGaps::default();
        assert_eq!(grace_floor(0, 0.9, &gaps), Duration::ZERO); // no gap measured yet
        assert_eq!(grace_floor(15_000, 0.9, &gaps), Duration::from_secs(15));
        for (step, secs) in [(1u64, 2u64), (2, 25), (3, 30)] {
            gaps.note_accepted(step, 0, start);
            gaps.note_late(step, 0, start + Duration::from_secs(secs));
        }
        assert_eq!(grace_floor(0, 0.0, &gaps), Duration::ZERO); // off
        assert_eq!(grace_floor(0, 0.5, &gaps), Duration::from_secs(25));
        assert_eq!(grace_floor(40_000, 0.5, &gaps), Duration::from_secs(40));
        // The arming site caps the larger of the adaptive window and the floor.
        // E1-like inputs: τ·ξ_step − ξ_quorum − ξ_sync = 54 − 33.7 − 8.2 s,
        // so the adaptive window alone is under 10 s.
        let mut config = round_test_config(8);
        config.grace_ms = 300_000;
        config.grace_gamma = 0.8;
        config.grace_tau = 2.0;
        let (grace, floor) = round_grace(&config, false, Some(27.0), 33.7, 8.2, &gaps);
        assert!(grace < Duration::from_secs(10));
        assert_eq!(floor, Duration::ZERO);
        config.grace_arrival_quantile = 0.9;
        assert_eq!(round_grace(&config, false, Some(27.0), 33.7, 8.2, &gaps).0, Duration::from_secs(30));
        config.grace_ms = 20_000; // the cap still bounds the floor
        assert_eq!(round_grace(&config, false, Some(27.0), 33.7, 8.2, &gaps).0, Duration::from_secs(20));
        config.grace_ms = 300_000;
        config.grace_arrival_quantile = 0.0;
        config.grace_min_ms = 45_000;
        assert_eq!(
            round_grace(&config, false, Some(27.0), 33.7, 8.2, &gaps),
            (Duration::from_secs(45), Duration::from_secs(45))
        );
    }

    #[test]
    fn launch_interval_adapts_to_step_time() {
        let floor = Duration::from_millis(100);
        // No estimate yet: manual floor only.
        assert_eq!(launch_interval(floor, 24.0, 4, None), floor);
        // Estimate present: H*step/P = 24*1.0/4 = 6s dominates the floor.
        assert_eq!(
            launch_interval(floor, 24.0, 4, Some(1.0)),
            Duration::from_secs_f64(6.0)
        );
        // Fast steps: adaptive floor (24*0.005/4 = 30ms) below manual -> manual wins.
        assert_eq!(launch_interval(floor, 24.0, 4, Some(0.005)), floor);
        // H target disabled: manual floor only.
        assert_eq!(launch_interval(floor, 0.0, 4, Some(1.0)), floor);
    }

    #[test]
    fn step_rates_estimate_from_consecutive_pushes() {
        let mut rates = StepRates::default();
        let t0 = Instant::now();
        let first = member(1, 10);
        let second = member(2, 20);
        rates.note(first, 100, t0);
        assert_eq!(rates.max_step_secs_for(&[first]), None); // one sample: no estimate yet
        rates.note(first, 110, t0 + Duration::from_secs(5));
        let est = rates.max_step_secs_for(&[first]).unwrap();
        assert!(
            (est - 0.5).abs() < 1e-9,
            "10 steps over 5s = 0.5 s/step, got {est}"
        );
        // A slower learner dominates the estimate.
        rates.note(second, 10, t0);
        rates.note(second, 12, t0 + Duration::from_secs(4));
        assert!((rates.max_step_secs_for(&[first, second]).unwrap() - 2.0).abs() < 1e-9);
        assert!((rates.max_step_secs_for(&[first]).unwrap() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn late_carry_is_off_for_strict_unbounded_and_fixed_roster_runs() {
        assert!(LateCarry::new(Some(0), false).is_none());
        assert!(LateCarry::new(None, false).is_none());
        assert!(LateCarry::new(Some(2), true).is_none());
        assert!(LateCarry::new(Some(2), false).is_some());
    }

    #[test]
    fn late_push_queues_then_joins_the_next_round_of_its_fragment() {
        let mut carry = LateCarry::new(Some(2), false).unwrap();
        let island = member(1, 10);
        // Round t=7 of fragment 1 (base 5) closed without island 1; the
        // fragment is now at version 7 and its next round is not open yet.
        let mut late = test_push_for(1, 5);
        late.global_step = 7;
        let mut rounds: Vec<Round> = Vec::new();
        assert_eq!(carry.carry(&mut rounds, island, late, 8, 7), CarryOutcome::Queued);
        let mut next = test_round(vec![member(0, 10), island]);
        next.t = 9;
        next.base_version = 7;
        let attached = carry.attach_queued(&mut next);
        assert_eq!(attached.len(), 1);
        assert_eq!(attached[0].2, Ok(2));
        assert_eq!(next.carried.len(), 1);
        assert!(next.pushes.is_empty(), "a carried delta takes no quorum slot");
        // The island still answers the new round from its new anchor.
        let mut fresh = test_push_for(1, 7);
        fresh.global_step = 9;
        assert_eq!(
            route_push(&mut [next], island, fresh, Some(2), false),
            PushDisposition::Accepted
        );
    }

    #[test]
    fn late_push_joins_an_already_open_later_round() {
        let mut carry = LateCarry::new(Some(2), false).unwrap();
        let mut open = test_round(vec![member(0, 10)]);
        open.t = 9;
        open.base_version = 7;
        let mut late = test_push_for(1, 5);
        late.global_step = 7;
        let mut rounds = vec![open];
        assert_eq!(
            carry.carry(&mut rounds, member(1, 10), late, 10, 7),
            CarryOutcome::Attached { step: 9, lag: 2 }
        );
        assert_eq!(rounds[0].carried.len(), 1);
    }

    #[test]
    fn late_push_beyond_the_lag_bound_or_for_an_open_attempt_is_refused() {
        let mut carry = LateCarry::new(Some(1), false).unwrap();
        let mut late = test_push_for(1, 5);
        late.global_step = 7;
        assert_eq!(
            carry.carry(&mut Vec::new(), member(1, 10), late, 8, 7),
            CarryOutcome::Rejected(PushDisposition::StaleBase)
        );
        // Same step and fragment still open under attempt 2: not late.
        let mut retried = test_round(vec![member(1, 10)]);
        retried.attempt = 2;
        let mut stale_attempt = test_push_for(1, 5);
        stale_attempt.global_step = 7;
        assert_eq!(
            carry.carry(&mut vec![retried], member(1, 10), stale_attempt, 8, 5),
            CarryOutcome::Rejected(PushDisposition::OutOfRound)
        );
        // A never-launched step is not late either.
        let mut future = test_push_for(1, 5);
        future.global_step = 9;
        assert_eq!(
            carry.carry(&mut Vec::new(), member(1, 10), future, 8, 7),
            CarryOutcome::Rejected(PushDisposition::OutOfRound)
        );
        assert!(carry.queued.is_empty());
    }

    #[test]
    fn a_carried_anchor_yields_one_delta() {
        let mut carry = LateCarry::new(Some(2), false).unwrap();
        let island = member(1, 10);
        let push = test_push_for(1, 5);
        assert!(!carry.reuses_anchor(island, &push));
        carry.spend_anchor(island, 1, 5);
        assert!(carry.reuses_anchor(island, &push));
        assert!(carry.reuses_anchor(island, &test_push_for(1, 3)));
        assert!(!carry.reuses_anchor(island, &test_push_for(1, 7)));
        // Another connection or fragment is unaffected.
        assert!(!carry.reuses_anchor(member(1, 11), &push));
        // A late push from an anchor already merged by its own round is not
        // carried; a restart clears the gathered push and frees the anchor.
        let other = member(2, 10);
        carry.note_gathered(other, 1, 5);
        let mut late = test_push_for(2, 5);
        late.global_step = 7;
        assert_eq!(
            carry.carry(&mut Vec::new(), other, late, 8, 7),
            CarryOutcome::Rejected(PushDisposition::Duplicate)
        );
        carry.release_anchor(other, &test_push_for(2, 5));
        let mut late = test_push_for(2, 5);
        late.global_step = 7;
        assert_eq!(carry.carry(&mut Vec::new(), other, late, 8, 7), CarryOutcome::Queued);
    }

    fn carried_merge(cfg: &Config) -> f32 {
        let mut st = two_fragment_streaming_state();
        st.versions[0] = 4;
        let island = member(0, 1);
        let late = member(1, 1);
        let mut round = test_round(vec![island, late]);
        round.p = 0;
        round.t = 6;
        round.base_version = 4;
        round.fold = round_fold(cfg, &st, 0, &round.expected_members, 1);
        let mut own = test_push_for(0, 4);
        own.fragment_id = 0;
        own.global_step = 6;
        own.outer_gradient = vec![1.0];
        assert_eq!(
            route_push(std::slice::from_mut(&mut round), island, own, Some(2), false),
            PushDisposition::Accepted
        );
        let mut carried = test_push_for(1, 2);
        carried.fragment_id = 0;
        carried.c_tokens = 40; // tokens^2/steps: 4x the own push's weight
        carried.outer_gradient = vec![3.0];
        assert_eq!(attach_carry(&mut round, late, carried, 2), Ok(2));
        let merge = prepare_round_compute(cfg, &st, &mut round).unwrap().finish_inline().unwrap();
        st.commit_merge(merge).unwrap();
        st.params[0][0]
    }

    #[test]
    fn carried_delta_merges_against_its_own_base_with_its_token_weight() {
        // Outer lr 1, no momentum: the fragment moves by minus the weighted
        // mean outer gradient, (1·1 + 4·3) / 5 = 2.6.
        let mut cfg = round_test_config(1);
        cfg.max_base_lag = Some(2);
        cfg.learner_weight = LearnerWeight::Tokens2OverSteps;
        assert!((carried_merge(&cfg) + 2.6).abs() < 1e-6);
        let mut streaming = cfg.clone();
        streaming.weighted_fold = WeightedFoldMode::Streaming;
        assert!((carried_merge(&streaming) + 2.6).abs() < 1e-6);
        // Carry discount 0.5 per version, lag 2: weight 4 · 0.25 = 1.
        let mut discounted = cfg.clone();
        discounted.membership.carry_gamma = 0.5;
        assert!((carried_merge(&discounted) + 2.0).abs() < 1e-6);
    }

    #[test]
    fn step_rates_time_a_step_from_its_first_fragment_push() {
        // One local step pushes every fragment; the repeats must not move the
        // step start, or the estimate shrinks to the gap between steps.
        let mut rates = StepRates::default();
        let t0 = Instant::now();
        let island = member(1, 10);
        for (step, start) in [(1u64, 0u64), (2, 10)] {
            for fragment in 0..4 {
                rates.note(island, step, t0 + Duration::from_secs(start + 2 * fragment));
            }
        }
        let est = rates.max_step_secs_for(&[island]).unwrap();
        assert!((est - 10.0).abs() < 1e-9, "step start to step start is 10 s, got {est}");
    }

    #[test]
    fn step_rates_smooth_with_ema() {
        let mut rates = StepRates::default();
        let t0 = Instant::now();
        rates.note(member(1, 10), 0, t0);
        rates.note(member(1, 10), 10, t0 + Duration::from_secs(5)); // seeds EMA at 0.5 s/step
                                                                    // A one-off 10x-slower interval (2 steps over 10s = 5.0 s/step sample)
                                                                    // must not replace the estimate wholesale: EMA -> 0.5·5.0 + 0.5·0.5.
        rates.note(member(1, 10), 12, t0 + Duration::from_secs(15));
        let est = rates.max_step_secs_for(&[member(1, 10)]).unwrap();
        assert!(
            (est - 2.75).abs() < 1e-9,
            "EMA of 0.5 then 5.0 should be 2.75, got {est}"
        );
    }

    #[test]
    fn step_rates_survive_learner_restart() {
        let mut rates = StepRates::default();
        let t0 = Instant::now();
        rates.note(member(1, 10), 100, t0);
        rates.note(member(1, 11), 5, t0 + Duration::from_secs(1));
        assert_eq!(rates.max_step_secs_for(&[member(1, 11)]), None);
        rates.note(member(1, 11), 15, t0 + Duration::from_secs(6));
        assert!((rates.max_step_secs_for(&[member(1, 11)]).unwrap() - 0.5).abs() < 1e-9);
        rates.remove(member(1, 10));
        assert!(!rates.0.contains_key(&member(1, 10)));
    }

    fn test_round(expected_members: Vec<Member>) -> Round {
        Round {
            t: 7,
            p: 1,
            base_version: 5,
            attempt: 1,
            pull: bytes::Bytes::new(),
            started: Instant::now(),
            expected_members,
            quorum_deadline: Instant::now() + Duration::from_secs(1),
            grace_deadline: None,
            quorum_size: 1,
            capacity_quorum: None,
            soft_deadline: None,
            soft_min: 1,
            quorum_ms: None,
            grace_ms: None,
            fold: None,
            carried: Vec::new(),
            pushes: HashMap::new(),
        }
    }

    fn test_push(base_version: u64) -> Push {
        Push {
            learner_id: 0,
            fragment_id: 1,
            global_step: 7,
            round_attempt: 1,
            base_version,
            local_step: 10,
            c_steps: 2,
            c_tokens: 20,
            outer_gradient: vec![1.0],
            tensor_scales: None,
            transfer: PushTransfer::default(),
        }
    }

    fn test_push_for(learner_id: u32, base_version: u64) -> Push {
        let mut push = test_push(base_version);
        push.learner_id = learner_id;
        push
    }

    #[test]
    fn unanswered_reconnect_rebinds_and_accepts_new_generation() {
        let old = member(0, 10);
        let replacement = member(0, 11);
        let other = member(1, 20);
        let registry = registry_with_current(replacement);
        let pull = bytes::Bytes::from_static(b"outstanding-pull");
        let mut round = test_round(vec![old, other]);
        round.pull = pull.clone();
        round.pushes.insert(other, test_push_for(1, 5));
        let mut rounds = vec![round];

        assert_eq!(
            rebind_current_unanswered_rounds(&registry, &mut rounds, replacement),
            vec![ReboundPull {
                t: 7,
                p: 1,
                attempt: 1,
                base_version: 5,
                old_generation: 10,
                pull,
            }]
        );
        assert_eq!(rounds[0].expected_members, vec![replacement, other]);
        assert_eq!(rounds[0].pushes.get(&other).unwrap().learner_id, 1);
        assert_eq!(
            route_push(&mut rounds, replacement, test_push(5), None, false),
            PushDisposition::Accepted
        );
        assert_eq!(
            route_push(&mut rounds, old, test_push(5), None, false),
            PushDisposition::UnexpectedMember
        );
        assert_eq!(rounds[0].pushes.len(), 2);
    }

    #[test]
    fn live_captured_generation_blocks_rebind_until_disconnect() {
        let old = member(0, 10);
        let replacement = member(0, 11);
        let registry = Arc::new(Mutex::new(RegistryState::default()));
        {
            let mut registry = registry.lock().unwrap();
            registry.register_group(test_group(old), Instant::now()).unwrap();
            registry.register_group(test_group(replacement), Instant::now()).unwrap();
        }
        let mut rounds = vec![test_round(vec![old])];

        assert!(rebind_current_unanswered_rounds(&registry, &mut rounds, replacement).is_empty());
        assert_eq!(rounds[0].expected_members, vec![old]);

        registry.lock().unwrap().groups.remove(&old);
        assert_eq!(
            rebind_current_unanswered_rounds(&registry, &mut rounds, replacement).len(),
            1
        );
        assert_eq!(rounds[0].expected_members, vec![replacement]);
    }

    #[test]
    fn answered_reconnect_does_not_rebind_or_repull() {
        let old = member(0, 10);
        let replacement = member(0, 11);
        let registry = registry_with_current(replacement);
        let mut round = test_round(vec![old]);
        round.pushes.insert(old, test_push(5));
        let mut rounds = vec![round];

        assert!(rebind_current_unanswered_rounds(&registry, &mut rounds, replacement).is_empty());
        assert_eq!(rounds[0].expected_members, vec![old]);
        assert!(rounds[0].pushes.contains_key(&old));
        assert_eq!(
            route_push(&mut rounds, replacement, test_push(5), None, false),
            PushDisposition::UnexpectedMember
        );
    }

    #[test]
    fn same_generation_rebind_is_idempotent_and_does_not_repull() {
        let old = member(0, 10);
        let replacement = member(0, 11);
        let registry = registry_with_current(replacement);
        let mut rounds = vec![test_round(vec![old])];

        assert_eq!(
            rebind_current_unanswered_rounds(&registry, &mut rounds, replacement).len(),
            1
        );
        assert!(rebind_current_unanswered_rounds(&registry, &mut rounds, replacement).is_empty());
        assert_eq!(rounds[0].expected_members, vec![replacement]);
    }

    #[test]
    fn superseded_hello_cannot_rebind_backwards() {
        let superseded = member(0, 11);
        let current = member(0, 12);
        let registry = registry_with_current(current);
        let mut rounds = vec![test_round(vec![current])];

        assert!(rebind_current_unanswered_rounds(&registry, &mut rounds, superseded).is_empty());
        assert_eq!(rounds[0].expected_members, vec![current]);
    }

    #[test]
    fn multiple_inflight_rounds_rebind_selectively() {
        let old = member(0, 10);
        let replacement = member(0, 11);
        let other = member(1, 20);
        let unrelated = member(2, 30);
        let registry = registry_with_current(replacement);

        let mut first = test_round(vec![old, other]);
        first.pull = bytes::Bytes::from_static(b"first");
        first.pushes.insert(other, test_push_for(1, 5));

        let mut second = test_round(vec![old]);
        second.t = 8;
        second.p = 2;
        second.pull = bytes::Bytes::from_static(b"second");

        let mut answered = test_round(vec![old]);
        answered.t = 9;
        answered.p = 3;
        let mut answered_push = test_push(5);
        answered_push.global_step = 9;
        answered_push.fragment_id = 3;
        answered.pushes.insert(old, answered_push);

        let mut unrelated_round = test_round(vec![unrelated]);
        unrelated_round.t = 10;
        unrelated_round.p = 4;

        let mut rounds = vec![first, second, answered, unrelated_round];
        let repulls = rebind_current_unanswered_rounds(&registry, &mut rounds, replacement);

        assert_eq!(
            repulls
                .iter()
                .map(|repull| (repull.t, repull.p, repull.pull.as_ref()))
                .collect::<Vec<_>>(),
            vec![(7, 1, b"first".as_slice()), (8, 2, b"second".as_slice())]
        );
        assert_eq!(rounds[0].expected_members, vec![replacement, other]);
        assert_eq!(rounds[1].expected_members, vec![replacement]);
        assert_eq!(rounds[2].expected_members, vec![old]);
        assert_eq!(rounds[3].expected_members, vec![unrelated]);
        assert!(rounds[0].pushes.contains_key(&other));
        assert!(rounds[2].pushes.contains_key(&old));
    }

    #[test]
    fn rebind_preserves_round_metadata() {
        let old = member(0, 10);
        let replacement = member(0, 11);
        let other = member(1, 20);
        let registry = registry_with_current(replacement);
        let started = Instant::now() - Duration::from_secs(2);
        let quorum_deadline = Instant::now() + Duration::from_secs(17);
        let grace_deadline = Some(Instant::now() + Duration::from_secs(9));
        let pull = bytes::Bytes::from_static(b"exact-wire-pull");
        let mut round = test_round(vec![other, old]);
        round.t = 41;
        round.p = 3;
        round.base_version = 17;
        round.attempt = 9;
        round.pull = pull.clone();
        round.started = started;
        round.quorum_deadline = quorum_deadline;
        round.grace_deadline = grace_deadline;
        round.quorum_size = 2;
        round.quorum_ms = Some(123);
        round.grace_ms = Some(456);
        let mut other_push = test_push_for(1, 16);
        other_push.global_step = 41;
        other_push.fragment_id = 3;
        other_push.round_attempt = 9;
        other_push.local_step = 77;
        round.pushes.insert(other, other_push);
        let mut rounds = vec![round];

        assert_eq!(
            rebind_current_unanswered_rounds(&registry, &mut rounds, replacement),
            vec![ReboundPull {
                t: 41,
                p: 3,
                attempt: 9,
                base_version: 17,
                old_generation: 10,
                pull: pull.clone(),
            }]
        );
        let rebound = &rounds[0];
        assert_eq!(rebound.t, 41);
        assert_eq!(rebound.p, 3);
        assert_eq!(rebound.base_version, 17);
        assert_eq!(rebound.attempt, 9);
        assert_eq!(rebound.pull, pull);
        assert_eq!(rebound.started, started);
        assert_eq!(rebound.quorum_deadline, quorum_deadline);
        assert_eq!(rebound.grace_deadline, grace_deadline);
        assert_eq!(rebound.quorum_size, 2);
        assert_eq!(rebound.quorum_ms, Some(123));
        assert_eq!(rebound.grace_ms, Some(456));
        assert_eq!(rebound.expected_members, vec![replacement, other]);
        assert_eq!(rebound.pushes.len(), 1);
        let preserved_push = rebound.pushes.get(&other).unwrap();
        assert_eq!(preserved_push.learner_id, 1);
        assert_eq!(preserved_push.base_version, 16);
        assert_eq!(preserved_push.local_step, 77);
    }

    #[test]
    fn round_membership_rejects_mid_round_join_and_new_generation() {
        let captured = member(0, 10);
        let mut rounds = vec![test_round(vec![captured])];
        assert_eq!(
            route_push(&mut rounds, member(1, 20), test_push(5), None, false),
            PushDisposition::UnexpectedMember
        );
        assert_eq!(
            route_push(&mut rounds, member(0, 11), test_push(5), None, false),
            PushDisposition::UnexpectedMember
        );
        assert!(rounds[0].pushes.is_empty());
        assert_eq!(
            route_push(&mut rounds, captured, test_push(5), None, false),
            PushDisposition::Accepted
        );
    }

    #[test]
    fn pull_replay_only_targets_a_reconnected_generation() {
        let captured = member(0, 10);
        let round = test_round(vec![captured]);

        assert!(!should_replay_pull(&round, captured));
        assert!(should_replay_pull(&round, member(0, 11)));
        assert!(!should_replay_pull(&round, member(1, 20)));
    }

    #[test]
    fn fragment_cannot_relaunch_while_an_older_round_still_owns_it() {
        let mut fragment_zero = test_round(vec![member(0, 10)]);
        fragment_zero.p = 0;
        fragment_zero.t = 1;
        let mut fragment_one = test_round(vec![member(0, 10)]);
        fragment_one.p = 1;
        fragment_one.t = 2;

        let mut inflight = vec![fragment_zero, fragment_one];
        inflight.remove(1); // Fragment 1 completes before fragment 0.

        assert!(!fragment_available(&inflight, 0));
        assert!(fragment_available(&inflight, 1));
    }

    #[test]
    fn out_of_order_compute_results_wait_for_contiguous_t_commit() {
        let mut ready = BTreeMap::new();
        ready.insert(12, "twelve");

        // cuda:N may finish t+1 first; the coordinator must not pop it.
        assert_eq!(take_next_commit(&mut ready, 11), None);
        assert_eq!(ready.get(&12), Some(&"twelve"));

        ready.insert(11, "eleven");
        assert_eq!(take_next_commit(&mut ready, 11), Some("eleven"));
        assert_eq!(take_next_commit(&mut ready, 12), Some("twelve"));
        assert!(ready.is_empty());
    }

    #[test]
    fn round_rejects_duplicate_future_base_and_out_of_round_pushes() {
        let captured = member(0, 10);
        let mut rounds = vec![test_round(vec![captured])];
        assert_eq!(
            route_push(&mut rounds, captured, test_push(6), None, false),
            PushDisposition::FutureBase
        );
        assert_eq!(
            route_push(&mut rounds, captured, test_push(4), None, false),
            PushDisposition::Accepted
        );
        assert_eq!(
            route_push(&mut rounds, captured, test_push(4), None, false),
            PushDisposition::Duplicate
        );
        let mut wrong_round = test_push(4);
        wrong_round.global_step = 99;
        assert_eq!(
            route_push(&mut rounds, captured, wrong_round, None, false),
            PushDisposition::OutOfRound
        );
        assert_eq!(rounds[0].pushes.len(), 1);
    }

    fn test_exact_push(learner_id: u32, base_version: u64) -> Push {
        Push {
            learner_id,
            fragment_id: 1,
            global_step: 7,
            round_attempt: 1,
            base_version,
            local_step: 7,
            c_steps: 1,
            c_tokens: 1,
            outer_gradient: vec![1.0],
            tensor_scales: None,
            transfer: PushTransfer::default(),
        }
    }

    #[test]
    fn max_base_lag_zero_is_exact_and_unique_per_logical_learner() {
        let original = member(0, 10);
        let replacement = member(0, 11);
        let second = member(1, 20);
        let mut rounds = vec![test_round(vec![original, second])];
        assert_eq!(
            route_push(
                &mut rounds,
                replacement,
                test_exact_push(0, 5),
                Some(0),
                true,
            ),
            PushDisposition::Accepted
        );
        assert_eq!(
            route_push(&mut rounds, original, test_exact_push(0, 5), Some(0), true,),
            PushDisposition::Duplicate
        );
        assert_eq!(
            route_push(&mut rounds, second, test_exact_push(1, 4), Some(0), true,),
            PushDisposition::StaleBase
        );
        assert_eq!(
            route_push(&mut rounds, second, test_exact_push(1, 6), Some(0), true,),
            PushDisposition::FutureBase
        );
        assert_eq!(rounds[0].pushes.len(), 1);
    }

    #[test]
    fn quorum_timeout_with_one_of_two_responses_restarts_without_merging() {
        let first = member(0, 10);
        let second = member(1, 20);
        let mut round = test_round(vec![first, second]);
        round.quorum_size = 2;
        round.quorum_deadline = Instant::now() - Duration::from_millis(1);
        round.pushes.insert(first, test_push(5));
        assert_eq!(round_action(&round, Instant::now()), RoundAction::Restart);

        // A new attempt uses a distinct wire token, so a delayed response
        // from the discarded attempt cannot be accepted into it.
        round.attempt = 2;
        round.pushes.clear();
        let mut rounds = vec![round];
        assert_eq!(
            route_push(&mut rounds, second, test_push(5), None, false),
            PushDisposition::OutOfRound
        );
        assert!(rounds[0].pushes.is_empty());
    }

    #[test]
    fn event_tape_records_rendezvous_metrics() {
        let path = std::env::temp_dir().join(format!(
            "yeto-event-tape-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut pushes = std::collections::HashMap::new();
        pushes.insert(
            member(0, 10),
            Push {
                learner_id: 0,
                fragment_id: 1,
                global_step: 7,
                round_attempt: 1,
                base_version: 3,
                local_step: 99,
                c_steps: 4,
                c_tokens: 40,
                outer_gradient: vec![1.0],
                tensor_scales: None,
                transfer: PushTransfer {
                    recv_seconds: 1.5,
                    arrived_unix: 1792000000.25,
                },
            },
        );
        append_tape(
            &path,
            7,
            1,
            3,
            1,
            &[member(0, 10), member(1, 20)],
            2,
            Some(11),
            Some(22),
            33,
            None,
            &pushes,
            &[],
            0.5,
            44,
            &[7; 32],
            None,
        )
        .unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("\"expected\":[0,1]"));
        assert!(text.contains("\"expected_members\":[{\"id\":0,\"generation\":10}"));
        assert!(text.contains("\"responded\":[0]"));
        assert!(text.contains("\"missed_grace\":[1]"));
        assert!(text.contains("\"quorum_ms\":11"));
        assert!(text.contains("\"grace_ms\":22"));
        assert!(text.contains("\"recv_seconds\":1.5,\"arrived_unix\":1792000000.25}"));
        let time_unix: f64 = text
            .split("\"time_unix\":")
            .nth(1)
            .and_then(|rest| rest.split(',').next())
            .unwrap()
            .parse()
            .unwrap();
        assert!(time_unix > 1.7e9);
        assert!(text.contains("\"sync_ms\":33"));
        assert!(text.contains("\"contribution\":1"));
        assert!(text.contains("\"protocol_version\":4"));
        assert!(text.contains("\"delta_semantics\":\"local_minus_raw_anchor\""));
        assert!(text.contains(&format!("\"sync/layout_hash\":\"{}\"", "07".repeat(32))));
        assert!(!text.contains("\"layout_hash\":"));
        assert!(!text.contains("\"policy_round\":"));
        assert!(!text.contains("\"accounted_c_steps\":"));

        std::fs::remove_file(&path).unwrap();
        let push = pushes.values_mut().next().unwrap();
        push.global_step = 5;
        push.local_step = 2;
        push.c_steps = 1;
        append_tape(
            &path,
            5,
            1,
            0,
            1,
            &[member(0, 10)],
            1,
            Some(1),
            Some(0),
            2,
            None,
            &pushes,
            &[],
            0.5,
            3,
            &[7; 32],
            Some(3),
        )
        .unwrap();
        let partial = std::fs::read_to_string(&path).unwrap();
        assert!(partial.contains("\"policy_round\":2"));
        assert!(partial.contains("\"sweep_fragment\":1"));
        assert!(partial.contains("\"sweep_fragments\":3"));
        assert!(partial.contains("\"sweep_complete\":false"));
        assert!(partial.contains("\"accounted_c_steps\":0"));
        assert!(partial.contains("\"accounted_c_tokens\":0"));

        std::fs::remove_file(&path).unwrap();
        pushes.values_mut().next().unwrap().global_step = 6;
        append_tape(
            &path,
            6,
            2,
            0,
            1,
            &[member(0, 10)],
            1,
            Some(1),
            Some(0),
            2,
            None,
            &pushes,
            &[],
            0.5,
            3,
            &[7; 32],
            Some(3),
        )
        .unwrap();
        let complete = std::fs::read_to_string(&path).unwrap();
        assert!(complete.contains("\"policy_round\":2"));
        assert!(complete.contains("\"sweep_fragment\":2"));
        assert!(complete.contains("\"sweep_complete\":true"));
        assert!(complete.contains("\"accounted_c_steps\":1"));
        assert!(complete.contains("\"accounted_c_tokens\":40"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn policy_sweep_ledger_snapshots_are_fsynced_idempotent_reconciliation_cuts() {
        let path = std::env::temp_dir().join(format!(
            "yeto-policy-sweep-ledger-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        // A killed diagnostic append may leave an unterminated line. The
        // authoritative snapshot must remain independently parseable.
        std::fs::write(&path, b"{\"torn\":").unwrap();
        let mut state = two_fragment_sweep_state();
        state.global_step = 2;
        state.versions = vec![1, 2];
        state.record_fragment_merge(0, 1, 321, false);
        state.record_fragment_merge(0, 1, 321, true);

        append_policy_sweep_ledger_snapshot(&path, &state, "resume").unwrap();
        append_policy_sweep_ledger_snapshot(&path, &state, "resume").unwrap();
        append_policy_sweep_ledger_snapshot(&path, &state, "complete").unwrap();
        append_policy_sweep_ledger_snapshot(&path, &state, "complete").unwrap();

        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text.matches("\"event\":\"policy_sweep_ledger\"").count(), 2);
        assert_eq!(text.matches(":resume:2\"").count(), 1);
        assert_eq!(text.matches(":complete:2\"").count(), 1);
        assert_eq!(
            text.matches("\"ledger\":[{\"id\":0,\"merges\":2,\"steps\":1,\"tokens\":321}]")
                .count(),
            2
        );
        assert!(text.contains("\"versions\":[1,2]"));
        assert!(text.contains("\"sweep_complete\":true"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn policy_sweep_event_tape_reports_the_enforced_equal_weights() {
        let path = std::env::temp_dir().join(format!(
            "yeto-policy-sweep-weights-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let pushes = HashMap::from([
            (
                member(0, 10),
                Push {
                    learner_id: 0,
                    fragment_id: 2,
                    global_step: 3,
                    round_attempt: 1,
                    base_version: 0,
                    local_step: 1,
                    c_steps: 1,
                    c_tokens: 10,
                    outer_gradient: vec![1.0],
                    tensor_scales: None,
                    transfer: PushTransfer::default(),
                },
            ),
            (
                member(1, 20),
                Push {
                    learner_id: 1,
                    fragment_id: 2,
                    global_step: 3,
                    round_attempt: 1,
                    base_version: 0,
                    local_step: 1,
                    c_steps: 1,
                    c_tokens: 1_000,
                    outer_gradient: vec![2.0],
                    tensor_scales: None,
                    transfer: PushTransfer::default(),
                },
            ),
        ]);
        append_tape(
            &path,
            3,
            2,
            0,
            1,
            &[member(0, 10), member(1, 20)],
            2,
            Some(1),
            Some(0),
            2,
            None,
            &pushes,
            &[],
            0.5,
            3,
            &[7; 32],
            Some(3),
        )
        .unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text.matches("\"weight\":1").count(), 2);
        assert_eq!(text.matches("\"contribution\":0.5").count(), 2);
        assert_eq!(text.matches("\"accounted_c_steps\":1").count(), 2);
        std::fs::remove_file(&path).ok();
    }

    // ----- dynamic membership (S27 E-1/E-2) -----

    fn capacity_round(expected: Vec<Member>, theta: f64, caps: &[(u32, f64)]) -> Round {
        let cfg = crate::membership::MembershipConfig {
            quorum_rule: crate::membership::QuorumRule::Capacity { theta, q_min: 1 },
            capacities: caps.iter().copied().collect(),
            ..Default::default()
        };
        let mut round = test_round(expected);
        round.capacity_quorum = cfg.round_quorum();
        round.quorum_deadline = Instant::now() + Duration::from_secs(3600);
        round
    }

    #[test]
    fn capacity_quorum_round_waits_for_theta_then_for_grace() {
        let (big, small, other) = (member(0, 1), member(1, 1), member(2, 1));
        let mut round = capacity_round(vec![big, small, other], 0.75, &[(0, 8.0)]);
        round.pushes.insert(small, test_push(5));
        round.pushes.insert(other, test_push(5));
        assert!(!round.quorum_reached(), "2 of 10 capacity");
        assert_eq!(round_action(&round, Instant::now()), RoundAction::Wait);
        round.pushes.remove(&other);
        round.pushes.insert(big, test_push(5));
        assert!(round.quorum_reached(), "9 of 10 capacity");
        round.grace_deadline = Some(Instant::now() + Duration::from_secs(3600));
        assert_eq!(round_action(&round, Instant::now()), RoundAction::Wait, "grace still open");
        round.grace_deadline = Some(Instant::now() - Duration::from_millis(1));
        assert_eq!(round_action(&round, Instant::now()), RoundAction::Complete);
    }

    #[test]
    fn soft_deadline_merges_below_quorum_but_never_merges_nothing() {
        let (first, second) = (member(0, 10), member(1, 20));
        let mut round = test_round(vec![first, second]);
        round.quorum_size = 2;
        round.quorum_deadline = Instant::now() + Duration::from_secs(3600);
        round.soft_deadline = Some(Instant::now() - Duration::from_millis(1));
        assert_eq!(round_action(&round, Instant::now()), RoundAction::Wait, "nothing arrived");
        round.pushes.insert(first, test_push(5));
        assert_eq!(round_action(&round, Instant::now()), RoundAction::Complete, "1 of K=2 at the soft deadline");
        round.soft_min = 2;
        assert_eq!(round_action(&round, Instant::now()), RoundAction::Wait, "q_min binds the soft merge");
        round.soft_deadline = None;
        round.soft_min = 1;
        round.quorum_deadline = Instant::now() - Duration::from_millis(1);
        assert_eq!(round_action(&round, Instant::now()), RoundAction::Restart, "off: unchanged");
    }

    #[test]
    fn expired_member_leaves_an_empty_round_waiting() {
        let only = member(0, 10);
        let mut round = test_round(vec![only]);
        assert!(round.forget_unanswered(only));
        assert!(round.expected_members.is_empty());
        assert_eq!(round_action(&round, Instant::now()), RoundAction::Wait);
        let answered = member(1, 20);
        let mut round = test_round(vec![answered]);
        round.pushes.insert(answered, test_push(5));
        assert!(!round.forget_unanswered(answered), "an accepted push keeps its slot");
    }

    fn leased_registry(lease_s: u64) -> Registry {
        let cfg = crate::membership::MembershipConfig {
            lease: Some(Duration::from_secs(lease_s)),
            ..Default::default()
        };
        Arc::new(Mutex::new(RegistryState::new(Membership::new(cfg, 2))))
    }

    fn catch_up_sends(catch_ups: &mut CatchUps<()>) -> Vec<(u32, usize)> {
        let mut sent = Vec::new();
        while let Some(step) = catch_ups.next_step() {
            match step {
                CatchUpStep::Send(member, (), p) => sent.push((member.learner_id, p)),
                CatchUpStep::Done(_) => {}
            }
        }
        sent
    }

    #[test]
    fn catch_up_yields_one_fragment_per_step_then_its_tail() {
        let rejoined = member(1, 7);
        let mut catch_ups = CatchUps::new();
        catch_ups.enqueue(rejoined, (), 0..3, CatchUpTail::REJOIN);
        assert!(catch_ups.holds(rejoined));
        for expected in 0..3 {
            match catch_ups.next_step() {
                Some(CatchUpStep::Send(who, (), p)) => assert_eq!((who, p), (rejoined, expected)),
                _ => panic!("fragment {expected} was not the next step"),
            }
        }
        match catch_ups.next_step() {
            Some(CatchUpStep::Done(done)) => {
                assert_eq!(done.member, rejoined);
                assert_eq!(done.tail, CatchUpTail::REJOIN);
            }
            _ => panic!("finished catch-up did not yield its tail"),
        }
        assert!(catch_ups.is_empty() && !catch_ups.holds(rejoined));
    }

    #[test]
    fn catch_up_interleaves_learners_and_merges_duplicate_requests() {
        let (a, b) = (member(1, 7), member(2, 9));
        let mut catch_ups = CatchUps::new();
        catch_ups.enqueue(a, (), 0..2, CatchUpTail::REJOIN);
        catch_ups.enqueue(b, (), 0..2, CatchUpTail::REJOIN);
        // The disconnect of a's old generation asks again: no second copy.
        catch_ups.enqueue(a, (), 0..2, CatchUpTail::REBIND);
        assert_eq!(catch_up_sends(&mut catch_ups), vec![(1, 0), (2, 0), (1, 1), (2, 1)]);

        catch_ups.enqueue(a, (), 0..1, CatchUpTail::REJOIN);
        catch_ups.enqueue(a, (), 0..1, CatchUpTail::REBIND);
        let _ = catch_ups.next_step();
        match catch_ups.next_step() {
            Some(CatchUpStep::Done(done)) => {
                assert_eq!(done.tail, CatchUpTail { rejoin: true, rebind: true })
            }
            _ => panic!("merged catch-up lost its tail"),
        }
    }

    #[test]
    fn catch_up_drops_superseded_generations_and_abandons_failed_streams() {
        let (old, current) = (member(1, 7), member(2, 9));
        let mut catch_ups = CatchUps::new();
        catch_ups.enqueue(old, (), 0..4, CatchUpTail::REJOIN);
        catch_ups.enqueue(current, (), 0..4, CatchUpTail::REJOIN);
        assert_eq!(catch_ups.drop_superseded(|m| m == current), vec![old]);
        assert!(!catch_ups.holds(old));

        // A failed stream stops the sends; the tail still runs.
        catch_ups.abandon(current);
        assert!(matches!(catch_ups.next_step(), Some(CatchUpStep::Done(done)) if done.member == current));
        assert!(catch_ups.next_step().is_none());
    }

    #[tokio::test]
    async fn lease_drop_rejoin_and_stale_generation_fencing() {
        let registry = leased_registry(1);
        let (silent, alive) = (member(0, 10), member(1, 20));
        let (tx, mut rx) = mpsc::channel(4);
        let t0 = Instant::now() - Duration::from_secs(5);
        {
            let mut r = registry.lock().unwrap();
            r.register_group(test_group_with_control(silent, tx), t0).unwrap();
            r.register_group(test_group(alive), t0).unwrap();
            assert!(r.membership.renew(alive, Instant::now()));
        }
        let mut rounds = vec![test_round(vec![silent, alive])];
        let mut rates = StepRates::default();

        // Drop: the silent island's lease ran out.
        expire_member_leases(&registry, &mut rounds, &mut rates).await;
        assert_eq!(rounds[0].expected_members, vec![alive]);
        assert!(!is_current_member(&registry, silent));
        assert!(is_current_member(&registry, alive));
        let frame = rx.recv().await.expect("re-JOIN request");
        assert_eq!(frame.msg_type, MSG_ERROR);
        assert!(std::str::from_utf8(&frame.parts[0]).unwrap().contains("re-JOIN"));
        // Its later push no longer fits the round.
        assert_eq!(
            route_push(&mut rounds, silent, test_push(5), None, false),
            PushDisposition::UnexpectedMember
        );

        let mut r = registry.lock().unwrap();
        r.unregister(silent);
        // Stale id: the expired generation cannot come back.
        assert_eq!(
            r.register_group(test_group(silent), Instant::now()).map(|x| x.1),
            Err(RegisterError::Fenced(AdmitError::RetiredGeneration))
        );
        assert_eq!(
            r.register_group(test_group(member(2, 1)), Instant::now()).map(|x| x.1),
            Err(RegisterError::Fenced(AdmitError::OutOfRange { limit: 2 }))
        );
        // Reconnect: a new generation is a re-JOIN with a per-fragment fence.
        let back = member(0, 11);
        assert_eq!(
            r.register_group(test_group(back), Instant::now()).map(|x| x.1),
            Ok(Admission::Rejoin)
        );
        r.membership.record_catch_up(back, &[3, 5]);
        assert!(r.membership.check_push_base(back, 1, 4).is_err());
        r.membership.check_push_base(back, 1, 5).unwrap();
        assert!(r.current.get(&0) == Some(&back));
    }

    #[tokio::test]
    async fn pool_member_under_peer_recovery_has_weight_zero_until_promoted() {
        let registry = leased_registry(1);
        let recover = HelloExt { syncer_epoch: Some(0), recover: true, ..Default::default() };
        let (donor, lost) = (member(1, 20), member(0, 10));
        let t0 = Instant::now() - Duration::from_secs(5);
        {
            let mut r = registry.lock().unwrap();
            r.register_group(test_group(donor), Instant::now()).unwrap();
            // A pool slot filled by an island that asks for peer recovery:
            // admitted to the pool, kept off the round roster and donors.
            r.register_group_with(test_group(lost), t0, recover).unwrap();
            assert!(r.is_recovering(lost));
            assert!(r.membership.is_live(lost));
        }
        assert!(!is_current_member(&registry, lost));
        let candidates = recovery_candidates(&registry, &[]);
        assert_eq!(candidates.iter().map(|c| c.member).collect::<Vec<_>>(), vec![donor]);

        // Its lease runs out during the restore: it leaves the pool and the
        // recovery; the next generation asks again (re-JOIN).
        let mut rounds = vec![test_round(vec![donor])];
        let mut rates = StepRates::default();
        assert_eq!(expire_member_leases(&registry, &mut rounds, &mut rates).await, vec![lost]);
        assert_eq!(rounds[0].expected_members, vec![donor]);
        let mut r = registry.lock().unwrap();
        assert!(!r.is_recovering(lost));
        r.unregister(lost);
        let back = member(0, 11);
        assert_eq!(r.register_group_with(test_group(back), Instant::now(), recover).map(|x| x.1),
                   Ok(Admission::Rejoin));
        assert!(r.is_recovering(back) && !r.current.contains_key(&0));

        // Promoted with the versions it holds after the replay: pushes based
        // on older versions are fenced.
        assert!(r.promote_recovered(back));
        r.membership.record_catch_up(back, &[7, 8]);
        assert_eq!(r.current.get(&0), Some(&back));
        assert!(r.membership.check_push_base(back, 0, 6).is_err());
        r.membership.check_push_base(back, 0, 7).unwrap();
        assert!(!r.promote_recovered(back), "promotion happens once per generation");
    }

    #[test]
    fn membership_options_are_refused_with_a_fixed_roster() {
        let mut cfg = round_test_config(4);
        cfg.learners = 2;
        cfg.quorum = 2;
        cfg.membership.soft_deadline = Some(Duration::from_secs(5));
        assert!(validate_config(&cfg).unwrap_err().to_string().contains("non-fixed rounds"));
        cfg.max_base_lag = None;
        validate_config(&cfg).unwrap();
        cfg.membership.quorum_rule =
            crate::membership::QuorumRule::Capacity { theta: 0.5, q_min: 1 };
        assert!(validate_config(&cfg).is_err(), "θ with K=2");
        cfg.quorum = 1;
        validate_config(&cfg).unwrap();
    }

    fn fold_test_state(mode: u8) -> GlobalState {
        let layout = Layout {
            fragments: vec![crate::state::FragmentInfo {
                merge_mode: mode,
                tensor_numels: vec![5, 3, 8],
                tensor_shapes: (mode == crate::state::MERGE_ISO)
                    .then(|| vec![(5, 1), (1, 3), (2, 4)]),
            }],
        };
        let mut state = GlobalState::new(layout, 0.7, 0.9, DTYPE_F32).unwrap();
        state.init_fragment(0, (0..16).map(|i| i as f32 * 0.125).collect()).unwrap();
        state
    }

    fn fold_test_push(learner_id: u32) -> Push {
        let gradient = (0..16u32)
            .map(|i| {
                let x = (learner_id * 31 + i * 17) % 23;
                if i % 7 == 0 {
                    0.0
                } else {
                    (x as f32 - 11.0) * 0.37f32.powi((learner_id + i) as i32 % 5)
                }
            })
            .collect();
        Push {
            learner_id,
            fragment_id: 0,
            global_step: 1,
            round_attempt: 1,
            base_version: 0,
            local_step: 1,
            c_steps: 1,
            c_tokens: 1 + u64::from(learner_id),
            outer_gradient: gradient,
            tensor_scales: None,
            transfer: PushTransfer::default(),
        }
    }

    fn fold_test_round(cfg: &Config, st: &GlobalState, members: &[Member], fold: bool) -> Round {
        let mut round = test_round(members.to_vec());
        round.t = 1;
        round.p = 0;
        round.base_version = 0;
        round.quorum_size = members.len();
        round.fold = if fold {
            round_fold(cfg, st, 0, members, members.len())
        } else {
            None
        };
        round
    }

    fn committed_bits(
        cfg: &Config,
        mode: u8,
        members: &[Member],
        order: &[usize],
        fold: bool,
    ) -> Vec<u32> {
        let mut st = fold_test_state(mode);
        let mut round = fold_test_round(cfg, &st, members, fold);
        assert_eq!(round.fold.is_some(), fold);
        for &index in order {
            let member = members[index];
            assert_eq!(
                route_one(&mut round, member, fold_test_push(member.learner_id)),
                PushDisposition::Accepted
            );
        }
        let merge = prepare_round_compute(cfg, &st, &mut round)
            .unwrap()
            .finish_inline()
            .unwrap();
        st.commit_merge(merge).unwrap();
        st.params[0].iter().map(|v| v.to_bits()).collect()
    }

    fn route_one(round: &mut Round, member: Member, push: Push) -> PushDisposition {
        super::route_push(std::slice::from_mut(round), member, push, None, false)
    }

    fn permutations(n: usize) -> Vec<Vec<usize>> {
        if n == 0 {
            return vec![Vec::new()];
        }
        let mut out = Vec::new();
        for rest in permutations(n - 1) {
            for at in 0..=rest.len() {
                let mut order = rest.clone();
                order.insert(at, n - 1);
                out.push(order);
            }
        }
        out
    }

    #[test]
    fn streaming_fold_commits_bit_identically_in_every_arrival_order() {
        let cfg = round_test_config(1);
        let members = [member(0, 3), member(2, 1), member(5, 9), member(7, 2)];
        for mode in [
            crate::state::MERGE_AVG,
            crate::state::MERGE_RDA,
            crate::state::MERGE_ISO,
        ] {
            let buffered = committed_bits(&cfg, mode, &members, &[3, 1, 0, 2], false);
            for order in permutations(members.len()) {
                assert_eq!(
                    committed_bits(&cfg, mode, &members, &order, true),
                    buffered,
                    "mode {mode} arrival order {order:?}"
                );
            }
        }
    }

    #[test]
    fn streaming_fold_releases_in_order_pushes_and_holds_early_ones() {
        let cfg = round_test_config(1);
        let st = fold_test_state(crate::state::MERGE_RDA);
        let members = [member(0, 1), member(1, 1), member(2, 1)];
        let mut round = fold_test_round(&cfg, &st, &members, true);
        let held = |round: &Round| -> Vec<u32> {
            let mut ids: Vec<u32> = round
                .pushes
                .iter()
                .filter(|(_, push)| !push.outer_gradient.is_empty())
                .map(|(member, _)| member.learner_id)
                .collect();
            ids.sort_unstable();
            ids
        };
        route_one(&mut round, members[2], fold_test_push(2));
        assert_eq!(held(&round), vec![2], "learner 2 waits for 0 and 1");
        route_one(&mut round, members[0], fold_test_push(0));
        assert_eq!(held(&round), vec![2], "learner 0 is folded at once");
        route_one(&mut round, members[1], fold_test_push(1));
        assert!(held(&round).is_empty(), "1 then the held 2 are folded");
        assert!(matches!(round.fold.as_ref(), Some(RoundFold::Exact(fold)) if fold.is_complete()));
    }

    #[test]
    fn streaming_fold_is_used_only_when_merge_inputs_are_fixed_at_launch() {
        let st = fold_test_state(crate::state::MERGE_AVG);
        let members = [member(0, 1), member(1, 1)];
        let cfg = round_test_config(1);
        assert!(round_fold(&cfg, &st, 0, &members, 2).is_some());
        assert!(round_fold(&cfg, &st, 0, &members, 1).is_none(), "partial quorum");
        let mut weighted = round_test_config(1);
        weighted.learner_weight = LearnerWeight::Tokens2OverSteps;
        assert!(round_fold(&weighted, &st, 0, &members, 2).is_none());
        let mut heloco = fold_test_state(crate::state::MERGE_AVG);
        heloco.delta_correction = Some(crate::merge::Heloco::default());
        assert!(round_fold(&cfg, &heloco, 0, &members, 2).is_none());
        let duplicate = [member(0, 1), member(0, 2)];
        assert!(round_fold(&cfg, &st, 0, &duplicate, 2).is_none());
        let mut leased = round_test_config(1);
        leased.membership.lease = Some(Duration::from_secs(5));
        assert!(round_fold(&leased, &st, 0, &members, 2).is_none(), "dynamic membership");
    }

    #[test]
    fn recovering_hello_stays_off_the_roster_until_promoted() {
        let mut r = RegistryState::new(Membership::default());
        let t0 = Instant::now();
        let old = Member { learner_id: 0, generation: 1 };
        let back = Member { learner_id: 0, generation: 2 };
        r.register_group(test_group(old), t0).unwrap();
        let recover = HelloExt { syncer_epoch: None, recover: true, expert_tokens: false };
        let (replaced, _) = r.register_group_with(test_group(back), t0, recover).unwrap();
        assert_eq!(replaced, Some(old), "the lost generation leaves the roster");
        assert!(r.current.is_empty() && r.is_recovering(back) && r.groups.contains_key(&back));
        assert!(!r.promote_recovered(old));
        assert!(r.promote_recovered(back));
        assert_eq!(r.current.get(&0), Some(&back));
        assert!(!r.is_recovering(back));
        // A recovering generation that disconnects is gone from both sets.
        let again = Member { learner_id: 0, generation: 3 };
        r.register_group_with(test_group(again), t0, recover).unwrap();
        r.unregister(again);
        assert!(r.current.is_empty() && !r.is_recovering(again));
    }

    #[test]
    fn hello_epoch_fence_and_membership_record_survive_restart() {
        let dir = std::env::temp_dir().join(format!("yeto-membership-record-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let checkpoint = dir.join("syncer.ckpt");
        let record = membership_record_path(&checkpoint);
        std::fs::remove_file(&record).ok();
        let mut cfg = Config { checkpoint_path: Some(checkpoint.clone()), syncer_epoch: 4, ..round_test_config(1) };
        let (membership, path) = initial_membership(&cfg, 2).unwrap();
        let mut r = RegistryState { membership_record: path, ..RegistryState::new(membership) };
        let t0 = Instant::now();
        let stale = HelloExt { syncer_epoch: Some(3), recover: false, expert_tokens: false };
        let gone = Member { learner_id: 1, generation: 9 };
        assert_eq!(
            r.register_group_with(test_group(gone), t0, stale).unwrap_err(),
            RegisterError::Fenced(AdmitError::StaleSyncerEpoch { offered: 3, current: 4 })
        );
        let current = HelloExt { syncer_epoch: Some(4), recover: false, expert_tokens: false };
        r.register_group_with(test_group(gone), t0, current).unwrap();
        r.unregister(gone);
        assert!(record.exists(), "a retirement is persisted at once");

        // Restart with a lower configured epoch: the persisted fence holds.
        cfg.syncer_epoch = 0;
        cfg.resume = true;
        let (membership, _) = initial_membership(&cfg, 2).unwrap();
        assert_eq!(membership.syncer_epoch(), 4);
        let mut r = RegistryState::new(membership);
        assert_eq!(
            r.register_group_with(test_group(gone), t0, HelloExt::default()).unwrap_err(),
            RegisterError::Fenced(AdmitError::RetiredGeneration)
        );
        // Without --resume the record is not read.
        cfg.resume = false;
        assert_eq!(initial_membership(&cfg, 2).unwrap().0.syncer_epoch(), 0);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Legacy HELLO body: one AVG fragment of one 4-element tensor.
    fn test_hello_payload(member: Member) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&PROTOCOL_VERSION.to_le_bytes());
        out.extend_from_slice(&member.learner_id.to_le_bytes());
        out.extend_from_slice(&member.generation.to_le_bytes());
        out.push(crate::protocol::DTYPE_F32);
        out.extend_from_slice(&1u32.to_le_bytes());
        out.push(crate::state::MERGE_AVG);
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&4u64.to_le_bytes());
        out.extend_from_slice(&[1; 32]); // layout fingerprint
        out.extend_from_slice(&[2; 32]); // session contract
        out.extend_from_slice(&2u16.to_le_bytes()); // num_streams
        out
    }

    #[test]
    fn hello_extension_trailer_is_parsed_after_the_legacy_body() {
        let member = Member { learner_id: 0, generation: 5 };
        let legacy = test_hello_payload(member);
        let parsed = parse_hello(&legacy, 1).unwrap();
        assert_eq!(parsed.ext, HelloExt::default());
        let ext = HelloExt { syncer_epoch: Some(12), recover: true, expert_tokens: false };
        let mut extended = legacy.clone();
        extended.extend_from_slice(&ext.encode());
        let parsed = parse_hello(&extended, 1).unwrap();
        assert_eq!(parsed.ext, ext);
        assert_eq!(parsed.generation, 5);
    }

    fn streaming_cfg() -> Config {
        let mut cfg = round_test_config(1);
        cfg.weighted_fold = WeightedFoldMode::Streaming;
        cfg.learner_weight = LearnerWeight::Tokens2OverSteps;
        cfg.max_base_lag = None;
        cfg
    }

    #[test]
    fn weighted_fold_mode_selects_the_reduction_per_round() {
        let st = fold_test_state(crate::state::MERGE_RDA);
        let members = [member(0, 1), member(1, 1), member(2, 1)];
        let kind = |cfg: &Config, st: &GlobalState, members: &[Member], quorum| {
            match round_fold(cfg, st, 0, members, quorum) {
                None => "buffered",
                Some(RoundFold::Exact(_)) => "exact",
                Some(RoundFold::Streaming(_)) => "streaming",
            }
        };
        let mut cfg = round_test_config(1);
        cfg.weighted_fold = WeightedFoldMode::Streaming;
        assert_eq!(kind(&cfg, &st, &members, 3), "exact");
        assert_eq!(kind(&cfg, &st, &members, 1), "streaming");
        assert_eq!(kind(&cfg, &st, &[member(0, 1), member(0, 2)], 2), "streaming");
        let weighted = streaming_cfg();
        assert_eq!(kind(&weighted, &st, &members, 3), "streaming");
        let mut heloco = fold_test_state(crate::state::MERGE_RDA);
        heloco.delta_correction = Some(crate::merge::Heloco::default());
        assert_eq!(kind(&weighted, &heloco, &members, 1), "buffered");
        let mut control = streaming_cfg();
        control.weighted_fold = WeightedFoldMode::Buffered;
        assert_eq!(kind(&control, &st, &members, 1), "buffered");
    }

    fn committed_params(cfg: &Config, mode: u8, members: &[Member], arrivals: &[usize]) -> Vec<f32> {
        let mut st = fold_test_state(mode);
        let mut round = test_round(members.to_vec());
        round.t = 1;
        round.p = 0;
        round.base_version = 0;
        round.quorum_size = 1;
        round.fold = round_fold(cfg, &st, 0, members, 1);
        for &index in arrivals {
            let member = members[index];
            let mut push = fold_test_push(member.learner_id);
            push.c_steps = 1 + member.learner_id % 3;
            assert_eq!(
                super::route_push(std::slice::from_mut(&mut round), member, push, None, false),
                PushDisposition::Accepted
            );
            if matches!(round.fold, Some(RoundFold::Streaming(_))) {
                // The streaming fold releases every push on arrival.
                assert!(round.pushes.values().all(|push| push.outer_gradient.is_empty()));
            }
        }
        let merge = prepare_round_compute(cfg, &st, &mut round)
            .unwrap()
            .finish_inline()
            .unwrap();
        st.commit_merge(merge).unwrap();
        st.params[0].to_vec()
    }

    #[test]
    fn streaming_partial_quorum_round_commits_within_rounding_of_buffered() {
        let streaming = streaming_cfg();
        let mut buffered_cfg = streaming_cfg();
        buffered_cfg.weighted_fold = WeightedFoldMode::Buffered;
        let members = [member(0, 3), member(2, 1), member(5, 9), member(7, 2)];
        for mode in [
            crate::state::MERGE_AVG,
            crate::state::MERGE_RDA,
            crate::state::MERGE_ISO,
        ] {
            // Quorum 1 of 4: any subset may close the round.
            for subset in [vec![2], vec![0, 3], vec![3, 1, 2], vec![0, 1, 2, 3]] {
                let buffered = committed_params(&buffered_cfg, mode, &members, &subset);
                let scale = buffered.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let first = committed_params(&streaming, mode, &members, &subset);
                let mut sorted = subset.clone();
                sorted.sort_unstable();
                for order in permutations(subset.len()) {
                    let arrivals: Vec<usize> = order.iter().map(|&i| subset[i]).collect();
                    let out = committed_params(&streaming, mode, &members, &arrivals);
                    let err = out
                        .iter()
                        .zip(&buffered)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    assert!(
                        err <= 32.0 * f32::EPSILON * scale,
                        "mode {mode} arrivals {arrivals:?}: error {err:e}, scale {scale:e}"
                    );
                    if arrivals == subset {
                        assert_eq!(out, first, "same arrival order must repeat exactly");
                    }
                }
            }
        }
    }
}
