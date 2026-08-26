//! Async TCP server implementing the syncer side of docs/PROTOCOL.md:
//! per-learner connection groups (control stream + striped data streams),
//! chunk reassembly, and the pull-driven quorum/grace merge scheduler
//! at the core of the training loop. Rounds are pipelined: up to
//! `Config::pipeline` fragments are in flight at once (arXiv 2604.21428's
//! "two fragments in flight"), so a slow quorum on one fragment never
//! delays pulling the next.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use tokio::io::AsyncWriteExt;
use tokio::net::tcp::OwnedWriteHalf;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::protocol::*;
use crate::state::{
    remove_final_marker, write_final_marker, ComputedMerge, GlobalState, Layout, PreparedMerge,
};

const CHUNK_SIZE: usize = 4 * 1024 * 1024;
const CHUNK_HEADER_SIZE: u64 = 24;
const MAX_PARTIAL_MESSAGES: usize = 64;
const WRITE_TIMEOUT: Duration = Duration::from_secs(180);
const WRITER_QUEUE: usize = 128;

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
    /// Fragment rounds in flight at once (the paper's "two fragments in
    /// flight" at τ=2). While one round sits in its quorum/grace window,
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
    /// H≈24 (the paper's design point) matches it. 0 disables; the manual
    /// min_round_interval_ms floor applies on top.
    pub sync_interval_steps: f64,
    /// HeLoCo per-tensor delta correction before merging.
    pub delta_correction: bool,
    pub quorum_timeout_s: u64,
    pub total_steps: u64,
    pub outer_lr: f32,
    pub outer_momentum: f32,
    /// Spectrum-flattening backend for MERGE_ISO tensors.
    pub iso_backend: crate::iso_worker::IsoBackendConfig,
    pub final_state: Option<std::path::PathBuf>,
    /// Consistent-snapshot file; written every `checkpoint_every` rounds at
    /// the quiescent cut between rounds, resumed from when `resume` is set.
    pub checkpoint_path: Option<std::path::PathBuf>,
    pub checkpoint_every: u64,
    pub resume: bool,
    /// Create the adjacent final marker only for a completed terminal cut.
    pub mark_final_checkpoint: bool,
    /// Benchmark-only exact local optimizer-step target for every learner.
    pub learner_budget_steps: Option<u64>,
    /// JSONL event tape: one record per merge.
    pub event_tape: Option<std::path::PathBuf>,
}

struct OutFrame {
    msg_type: u8,
    parts: Vec<bytes::Bytes>,
}

struct PartialMsg {
    buf: Vec<u8>,
    filled: usize,
    ranges: Vec<(usize, usize)>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct Member {
    learner_id: u32,
    generation: u64,
}

struct Group {
    member: Member,
    dtype: u8,
    layout: Layout,
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

    /// Send a large inner frame, striped as CHUNK envelopes across data
    /// streams (or unchunked on the control stream when none exist).
    async fn send_large(&self, msg_type: u8, payload: bytes::Bytes) -> Result<()> {
        let mut streams: Vec<(u16, mpsc::Sender<OutFrame>)> = self
            .data
            .lock()
            .unwrap()
            .iter()
            .map(|(index, sender)| (*index, sender.clone()))
            .collect();
        streams.sort_unstable_by_key(|(index, _)| *index);
        let streams: Vec<mpsc::Sender<OutFrame>> =
            streams.into_iter().map(|(_, sender)| sender).collect();
        if streams.is_empty() {
            return self.send_small(msg_type, payload).await;
        }
        // Inner frame = header + payload, chunked over its full byte length.
        let mut inner = Vec::with_capacity(13 + payload.len());
        inner.extend_from_slice(&MAGIC.to_le_bytes());
        inner.push(msg_type);
        inner.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        inner.extend_from_slice(&payload);
        let inner = bytes::Bytes::from(inner);

        let msg_id = self.msg_id.fetch_add(1, Ordering::Relaxed);
        let total = inner.len() as u64;
        let mut offset = 0usize;
        while offset < inner.len() {
            let end = (offset + CHUNK_SIZE).min(inner.len());
            let mut head = Vec::with_capacity(24);
            head.extend_from_slice(&msg_id.to_le_bytes());
            head.extend_from_slice(&total.to_le_bytes());
            head.extend_from_slice(&(offset as u64).to_le_bytes());
            let idx = self.rr.fetch_add(1, Ordering::Relaxed) % streams.len();
            streams[idx]
                .send(OutFrame {
                    msg_type: MSG_CHUNK,
                    parts: vec![bytes::Bytes::from(head), inner.slice(offset..end)],
                })
                .await
                .map_err(|_| {
                    anyhow::anyhow!(
                        "learner {} generation {} data stream closed",
                        self.member.learner_id,
                        self.member.generation
                    )
                })?;
            offset = end;
        }
        Ok(())
    }
}

enum Event {
    Hello {
        group: Arc<Group>,
    },
    Init {
        member: Member,
        fragment_id: u32,
        values: Vec<f32>,
    },
    Push {
        member: Member,
        push: Push,
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
}

#[derive(Clone, Debug, PartialEq)]
struct SessionSpec {
    dtype: u8,
    layout: Layout,
    layout_fingerprint: [u8; 32],
}

struct ParsedHello {
    learner_id: u32,
    generation: u64,
    dtype: u8,
    layout: Layout,
    layout_fingerprint: [u8; 32],
    num_streams: u16,
    max_init_payload: u64,
    max_push_payload: u64,
}

#[derive(Default)]
struct RegistryState {
    /// The generation eligible for future rounds for each logical learner.
    current: HashMap<u32, Member>,
    /// Every still-connected generation, including one retained solely for
    /// a round that captured it before a reconnect or duplicate HELLO.
    groups: HashMap<Member, Arc<Group>>,
}

impl RegistryState {
    /// Atomically register one control connection generation. Returning an
    /// error leaves both maps untouched, so an identical concurrent HELLO
    /// cannot replace a live group and later remove it on disconnect.
    fn register_group(&mut self, group: Arc<Group>) -> std::result::Result<Option<Member>, ()> {
        let member = group.member;
        if self.groups.contains_key(&member) {
            return Err(());
        }
        self.groups.insert(member, group);
        Ok(self.current.insert(member.learner_id, member))
    }
}

/// Decoupled DiLoCo's adaptive grace window (arXiv 2604.21428 Eq. 3): after
/// quorum, wait for stragglers only within the slack the learners' compute
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
/// (each push carries the learner's local_step), smoothed with an EMA as
/// the paper prescribes for the grace-window inputs (arXiv 2604.21428,
/// "ξ_step, ξ_quorum, ξ_sync can be tracked via exponential moving
/// averages") — a single push interval is too noisy to size the grace
/// window on its own.
#[derive(Default)]
struct StepRates(HashMap<Member, (Option<Instant>, u64, Option<f64>)>);

/// EMA smoothing: new estimate = α·sample + (1−α)·previous.
const STEP_EMA_ALPHA: f64 = 0.5;

impl StepRates {
    fn note(&mut self, member: Member, local_step: u64, now: Instant) {
        let entry = self.0.entry(member).or_insert((None, 0, None));
        if let Some(prev) = entry.0 {
            if local_step > entry.1 {
                let secs = now.duration_since(prev).as_secs_f64() / (local_step - entry.1) as f64;
                entry.2 = Some(match entry.2 {
                    Some(ema) => STEP_EMA_ALPHA * secs + (1.0 - STEP_EMA_ALPHA) * ema,
                    None => secs, // first sample seeds the EMA
                });
            }
        }
        *entry = (Some(now), local_step, entry.2);
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

pub async fn run(cfg: Config) -> Result<()> {
    if cfg.learners == 0 {
        bail!("--learners must be positive");
    }
    if cfg.quorum == 0 {
        bail!("--quorum must be positive");
    }
    if cfg.mark_final_checkpoint && cfg.checkpoint_path.is_none() {
        bail!("--mark-final-checkpoint requires --checkpoint-path");
    }
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
    info!(port = cfg.port, "syncer listening");
    let (event_tx, event_rx) = mpsc::channel::<Event>(1024);
    let registry: Registry = Arc::new(Mutex::new(RegistryState::default()));
    let session: Session = Arc::new(Mutex::new(None));

    let accept_registry = registry.clone();
    let accept_session = session.clone();
    let expected_learners = cfg.learners;
    tokio::spawn(async move {
        loop {
            match listener.accept().await {
                Ok((stream, peer)) => {
                    let reg = accept_registry.clone();
                    let session = accept_session.clone();
                    let tx = event_tx.clone();
                    tokio::spawn(async move {
                        if let Err(e) =
                            handle_connection(stream, reg, session, expected_learners, tx).await
                        {
                            warn!(%peer, "connection ended: {e:#}");
                        }
                    });
                }
                Err(e) => warn!("accept failed: {e}"),
            }
        }
    });

    scheduler(cfg, event_rx, registry).await
}

fn negotiated_payload_limits(layout: &Layout, dtype: u8) -> Result<(u64, u64)> {
    let mut max_init = 0u64;
    let mut max_push = 0u64;
    for fragment in &layout.fragments {
        let numel = fragment.numel()?;
        let init_bytes = tensor_nbytes(bulk_dtype(dtype), numel)? as u64;
        let delta_bytes = if dtype == DTYPE_Q4 {
            q4_nbytes(numel)?
        } else {
            tensor_nbytes(dtype, numel)?
        } as u64;
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

fn parse_hello(payload: &[u8], expected_learners: u32) -> Result<ParsedHello> {
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
    if !matches!(dtype, DTYPE_F32 | DTYPE_BF16 | DTYPE_Q4) {
        bail!("unsupported session dtype {dtype}");
    }
    let num_fragments = r.u32()?;
    let layout = Layout::decode(&mut r, num_fragments)?;
    let layout_fingerprint = r
        .take(32)?
        .try_into()
        .context("layout fingerprint must be 32 bytes")?;
    let num_streams = r.u16()?;
    if r.remaining() != 0 {
        bail!("trailing bytes in HELLO");
    }
    if num_streams > 256 {
        bail!("num_streams {num_streams} exceeds limit 256");
    }
    let (max_init_payload, max_push_payload) = negotiated_payload_limits(&layout, dtype)?;
    Ok(ParsedHello {
        learner_id,
        generation,
        dtype,
        layout,
        layout_fingerprint,
        num_streams,
        max_init_payload,
        max_push_payload,
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
    event_tx: mpsc::Sender<Event>,
) -> Result<()> {
    stream.set_nodelay(true)?;
    let (mut rd, mut wr) = stream.into_split();
    let first = match read_frame_limited(&mut rd, |msg_type| match msg_type {
        MSG_HELLO | MSG_DATA_HELLO => Ok(MAX_HELLO_FRAME),
        other => bail!("first frame must be HELLO/DATA_HELLO, got {other}"),
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
            let parsed = match parse_hello(&first.payload, expected_learners) {
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
                num_streams,
                max_init_payload,
                max_push_payload,
            } = parsed;
            let num_fragments = layout.fragments.len();
            let offered = SessionSpec {
                dtype,
                layout: layout.clone(),
                layout_fingerprint,
            };
            let mismatch = {
                let mut guard = session.lock().unwrap();
                match guard.as_ref() {
                    None => {
                        *guard = Some(offered.clone());
                        None
                    }
                    Some(expected) if expected == &offered => None,
                    Some(expected) => Some(format!(
                        "session mismatch: expected dtype {} and initialized layout, got dtype {} and {} fragments",
                        expected.dtype,
                        dtype,
                        layout.fragments.len()
                    )),
                }
            };
            if let Some(message) = mismatch {
                send_direct(&mut wr, MSG_ERROR, message.as_bytes()).await?;
                bail!(message);
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
                layout,
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
            let registration = { registry.lock().unwrap().register_group(group.clone()) };
            let replaced = match registration {
                Ok(replaced) => replaced,
                Err(()) => {
                    let message = format!(
                        "duplicate connection generation {generation} for learner {learner_id}"
                    );
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
            let res = read_loop(&mut rd, &group, &event_tx).await;
            {
                let mut registry = registry.lock().unwrap();
                registry.groups.remove(&member);
                if registry.current.get(&learner_id) == Some(&member) {
                    registry.current.remove(&learner_id);
                }
            }
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
            read_loop(&mut rd, &group, &event_tx).await
        }
        t => bail!("first frame must be HELLO/DATA_HELLO, got {t}"),
    }
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
) -> Result<()> {
    loop {
        let frame = match read_frame_limited(rd, |msg_type| match msg_type {
            MSG_INIT_PARAMS => Ok(group.max_init_payload),
            MSG_PUSH_FRAGMENT => Ok(group.max_push_payload),
            MSG_HEARTBEAT => Ok(12),
            MSG_FINAL_ACK => Ok(10),
            MSG_BUDGET_DONE => Ok(8),
            MSG_CHUNK => Ok(CHUNK_HEADER_SIZE + CHUNK_SIZE as u64),
            other => bail!(
                "unexpected direct message type {other} from learner {} generation {}",
                group.member.learner_id,
                group.member.generation
            ),
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
                Ok(Some(inner)) => {
                    dispatch_inner(group, inner.msg_type, &inner.payload, event_tx).await
                }
                Ok(None) => Ok(()),
                Err(error) => Err(error),
            },
            t => dispatch_inner(group, t, &frame.payload, event_tx).await,
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

fn reassemble(group: &Arc<Group>, payload: &[u8]) -> Result<Option<Frame>> {
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
    Ok(Some(Frame {
        msg_type,
        payload: payload.to_vec(),
    }))
}

async fn dispatch_inner(
    group: &Arc<Group>,
    msg_type: u8,
    payload: &[u8],
    event_tx: &mpsc::Sender<Event>,
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
            let mut values = Vec::new();
            decode_tensor(bulk_dtype(group.dtype), bytes, &mut values)?;
            event_tx
                .send(Event::Init {
                    member: group.member,
                    fragment_id,
                    values,
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
            let bytes = r.rest();
            let expected = if group.dtype == DTYPE_Q4 {
                q4_nbytes(numel)?
            } else {
                tensor_nbytes(group.dtype, numel)?
            };
            if bytes.len() != expected {
                bail!(
                    "PUSH_FRAGMENT fragment {fragment_id} has {} delta bytes, expected {expected}",
                    bytes.len()
                );
            }
            let mut outer_gradient = Vec::new();
            if group.dtype == DTYPE_Q4 {
                decode_q4(bytes, numel, &mut outer_gradient)?;
            } else {
                decode_tensor(group.dtype, bytes, &mut outer_gradient)?;
            }
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
                    },
                })
                .await
                .ok();
        }
        MSG_HEARTBEAT => {
            let mut r = Reader(payload);
            let learner_id = r.u32()?;
            let _local_step = r.u64()?;
            if learner_id != group.member.learner_id {
                bail!(
                    "HEARTBEAT learner id {learner_id} does not match connected group {}",
                    group.member.learner_id
                );
            }
            if r.remaining() != 0 {
                bail!("trailing bytes in HEARTBEAT");
            }
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
            event_tx
                .send(Event::BudgetDone {
                    member: group.member,
                    local_steps,
                })
                .await
                .ok();
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

async fn scheduler(
    cfg: Config,
    mut events: mpsc::Receiver<Event>,
    registry: Registry,
) -> Result<()> {
    let mut state: Option<GlobalState> = None;
    let mut budget_reports: HashSet<u32> = HashSet::new();

    // Phase 1: wait until every fragment is initialized (via INIT_PARAMS or
    // a resumed checkpoint) and all expected learners have connected (late
    // joiners are still served afterwards).
    info!(
        expected = cfg.learners,
        "waiting for learners and INIT_PARAMS"
    );
    loop {
        let connected = registry.lock().unwrap().current.len() as u32;
        if let Some(st) = &state {
            let budget_ready =
                budget_reports.is_empty() || budget_reports.len() == cfg.learners as usize;
            if st.all_initialized() && connected >= cfg.learners && budget_ready {
                break;
            }
        }
        match events.recv().await.context("event channel closed")? {
            Event::Hello { group } => {
                if state.is_none() {
                    // Layout comes from the HELLO of the first learner.
                    // (All learners must build identical layouts.)
                    let mut st = new_state_for(&group, &cfg)?;
                    if cfg.resume {
                        if let Some(path) = cfg.checkpoint_path.as_ref().filter(|p| p.exists()) {
                            st.load_checkpoint(path)?;
                            info!(step = st.global_step, "resumed from checkpoint");
                        }
                    }
                    state = Some(st);
                }
            }
            Event::Init {
                member,
                fragment_id,
                values,
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
                st.init_fragment(fragment_id as usize, values)?;
                if st.all_initialized() {
                    info!("global parameters initialized");
                }
            }
            Event::Push { .. } => warn!("push before initialization; dropped"),
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
        }
    }
    let mut st = state.unwrap();
    if !budget_reports.is_empty() {
        save_budget_checkpoint(&cfg, &st)?;
        return Ok(());
    }
    let num_fragments = st.layout.fragments.len() as u64;
    let mut step_rates = StepRates::default();
    let mut last_sync_secs = 0.0f64; // previous round's merge+broadcast time

    // Once a marked run is actually going to make more progress, the old
    // marker must stop being publishable before any new round can commit.
    if cfg.mark_final_checkpoint && st.global_step < cfg.total_steps {
        remove_final_marker(cfg.checkpoint_path.as_ref().unwrap())?;
    }

    // Send everyone the initial (or resumed) global parameters so all
    // learners start bit-identical (also serves recovery for late joiners).
    broadcast_all_fragments(&st, &registry).await;

    // Phase 2: gather, compute, and commit are separate stages.  Torch SVD
    // matrices execute concurrently and may finish out of order, but only
    // this scheduler mutates coordinator state, strictly in fragment-step t
    // order.  A fragment remains busy across all three stages, preventing a
    // second round from observing an uncommitted version/momentum state.
    let depth = (cfg.pipeline.max(1) as u64).min(num_fragments) as usize;
    let manual_floor = Duration::from_millis(cfg.min_round_interval_ms);
    let mut next_launch = Instant::now(); // earliest allowed next round launch
    let mut next_t = st.global_step + 1;
    let mut next_commit_t = st.global_step + 1;
    let mut inflight: Vec<Round> = Vec::new();
    let mut computing = tokio::task::JoinSet::<Result<ComputedRound>>::new();
    let mut ready: BTreeMap<u64, ComputedRound> = BTreeMap::new();
    let mut busy_fragments: HashSet<usize> = HashSet::new();
    let mut budget_cutoff = false;
    'outer: while next_t <= cfg.total_steps
        || !inflight.is_empty()
        || !computing.is_empty()
        || !ready.is_empty()
    {
        // Commit only the contiguous prefix.  A faster cuda:N worker can put
        // t+1 in `ready` first, but it cannot update Nesterov/version/tape or
        // broadcast until t has committed.
        while let Some(completed) = take_next_commit(&mut ready, next_commit_t) {
            let p = completed.round.p;
            commit_round(&cfg, &mut st, &registry, &mut last_sync_secs, completed).await?;
            if !busy_fragments.remove(&p) {
                bail!("fragment {p} lost its busy ownership at commit");
            }
            next_commit_t = next_commit_t
                .checked_add(1)
                .context("commit step overflow")?;
        }

        // Keep the pipeline full (throttled by min_round_interval_ms).
        while inflight.len() + computing.len() + ready.len() < depth
            && next_t <= cfg.total_steps
            && Instant::now() >= next_launch
        {
            let p = ((next_t - 1) % num_fragments) as usize;
            if busy_fragments.contains(&p) {
                break;
            }
            let groups = current_groups(&registry);
            if groups.is_empty() {
                next_launch = Instant::now() + Duration::from_millis(100);
                break;
            }
            let t = next_t;
            next_t += 1;
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
            let launch_quorum = (cfg.quorum as usize).min(expected_members.len());
            for g in &groups {
                let _ = g.send_small(MSG_PULL_REQ, pull.clone()).await;
            }
            if !busy_fragments.insert(p) {
                bail!("fragment {p} acquired twice");
            }
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
                quorum_ms: None,
                grace_ms: None,
                pushes: HashMap::new(),
            });
        }

        // Arm the grace window of any round that just reached quorum.
        for r in inflight.iter_mut() {
            if r.pushes.len() >= r.quorum_size && r.grace_deadline.is_none() {
                let grace = adaptive_grace(
                    cfg.grace_tau,
                    cfg.grace_gamma,
                    step_rates.max_step_secs_for(&r.expected_members),
                    r.started.elapsed().as_secs_f64(),
                    last_sync_secs,
                    Duration::from_millis(cfg.grace_ms),
                );
                r.grace_deadline = Some(Instant::now() + grace);
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
                    let round = inflight.remove(i);
                    let prepared = prepare_round_compute(&st, &round)?;
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
                    let r = &mut inflight[i];
                    let groups = current_groups(&registry);
                    warn!(
                        step = r.t,
                        attempt = r.attempt,
                        responses = r.pushes.len(),
                        quorum = r.quorum_size,
                        "quorum timeout below K; launching a new frozen-membership attempt"
                    );
                    if !groups.is_empty() {
                        r.expected_members = groups.iter().map(|group| group.member).collect();
                        r.expected_members.sort_unstable();
                        r.quorum_size = (cfg.quorum as usize).min(r.expected_members.len());
                        r.base_version = st.versions[r.p];
                    }
                    r.attempt += 1;
                    r.pull = encode_pull(r.p, r.t, r.attempt);
                    r.pushes.clear();
                    r.started = Instant::now();
                    r.grace_deadline = None;
                    r.quorum_ms = None;
                    r.grace_ms = None;
                    for g in groups {
                        let _ = g.send_small(MSG_PULL_REQ, r.pull.clone()).await;
                    }
                    r.quorum_deadline = Instant::now() + Duration::from_secs(cfg.quorum_timeout_s);
                }
                RoundAction::Wait => {}
            }
            i += 1;
        }
        if submitted_any {
            continue; // refill the pipeline before waiting again
        }
        // Wait for the next event, the earliest in-flight deadline, or the
        // launch throttle opening (whichever comes first). Without the
        // throttle term an empty pipeline would spin; without in-flight
        // deadlines a throttled launch would oversleep.
        let mut earliest = inflight
            .iter()
            .map(|r| r.grace_deadline.unwrap_or(r.quorum_deadline))
            .min();
        let next_fragment_available = inflight.len() + computing.len() + ready.len() < depth
            && next_t <= cfg.total_steps
            && !busy_fragments.contains(&(((next_t - 1) % num_fragments) as usize));
        if next_fragment_available {
            earliest = Some(earliest.map_or(next_launch, |d| d.min(next_launch)));
        }
        let wake = match earliest {
            Some(deadline) if !computing.is_empty() => tokio::select! {
                _ = tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)) => SchedulerWake::Deadline,
                result = computing.join_next() => SchedulerWake::Computed(result),
                event = events.recv() => SchedulerWake::Event(event),
            },
            Some(deadline) => tokio::select! {
                _ = tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)) => SchedulerWake::Deadline,
                event = events.recv() => SchedulerWake::Event(event),
            },
            None if !computing.is_empty() => tokio::select! {
                result = computing.join_next() => SchedulerWake::Computed(result),
                event = events.recv() => SchedulerWake::Event(event),
            },
            None => continue,
        };
        match wake {
            SchedulerWake::Deadline => continue,
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
                Event::Push { member, push } => {
                    let learner_id = member.learner_id;
                    let generation = member.generation;
                    let local_step = push.local_step;
                    let global_step = push.global_step;
                    let fragment_id = push.fragment_id;
                    match route_push(&mut inflight, member, push) {
                        PushDisposition::Accepted => {
                            step_rates.note(member, local_step, Instant::now());
                        }
                        disposition => warn!(
                            learner_id,
                            generation,
                            step = global_step,
                            fragment = fragment_id,
                            reason = ?disposition,
                            "push rejected"
                        ),
                    }
                }
                Event::Hello { group } => {
                    // Rejoining learner: catch it up to the current state.
                    send_all_fragments(&st, &group).await;
                }
                Event::Init { .. } => {} // already initialized; ignore
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
                    let target = cfg
                        .learner_budget_steps
                        .context("received BUDGET_DONE outside learner-budget mode")?;
                    if !is_current_member(&registry, member) {
                        bail!("BUDGET_DONE from a superseded learner");
                    }
                    record_budget_report(&mut budget_reports, target, member, local_steps)?;
                    let cancelled = inflight.len();
                    inflight.clear();
                    ready.clear();
                    computing.abort_all();
                    while computing.join_next().await.is_some() {}
                    st.drain_iso_backend().await?;
                    busy_fragments.clear();
                    budget_cutoff = true;
                    info!(
                        learner_id = member.learner_id,
                        cancelled_rounds = cancelled,
                        target_steps = target,
                        "learner-budget cutoff established"
                    );
                    break 'outer;
                }
                Event::Disconnected { member } => {
                    // Membership and accepted responses are immutable for an
                    // attempt. A disconnect never erases work, and a new
                    // generation can join only a later launch/retry.
                    warn!(
                        learner_id = member.learner_id,
                        generation = member.generation,
                        "learner connection generation disconnected"
                    );
                    step_rates.remove(member);
                }
            },
        }
    }

    if cfg.learner_budget_steps.is_some() {
        if !budget_cutoff {
            bail!("learner-budget scheduler exited without a cutoff report");
        }
        collect_budget_reports(&cfg, &mut budget_reports, &mut events, &registry).await?;
        st.drain_iso_backend().await?;
        save_budget_checkpoint(&cfg, &st)?;
        return Ok(());
    }

    if next_commit_t != cfg.total_steps.saturating_add(1)
        || st.global_step != cfg.total_steps
        || !busy_fragments.is_empty()
    {
        bail!(
            "non-quiescent terminal cut: next_commit_t={next_commit_t} global_step={} busy={:?}",
            st.global_step,
            busy_fragments
        );
    }
    // Explicitly drain and health-check every SVD worker before a checkpoint
    // or marker can publish the terminal cut.
    st.drain_iso_backend().await?;

    // The outer loop is now quiescent: every launched round has completed
    // and every computed result has committed in strict t order.
    // Persist this authoritative cut regardless of the periodic checkpoint
    // interval so a non-divisible total_steps can never leave a stale final
    // checkpoint behind.
    if let Some(path) = &cfg.checkpoint_path {
        if cfg.mark_final_checkpoint {
            remove_final_marker(path)?;
        }
        st.save_checkpoint(path)?;
        info!(
            step = st.global_step,
            path = %path.display(),
            "final checkpoint written"
        );
        if cfg.mark_final_checkpoint {
            write_final_marker(path, st.global_step)?;
        }
    }
    if let Some(path) = &cfg.final_state {
        dump_state(&st, path)?;
        info!(path = %path.display(), "final global state written");
    }
    // Freeze terminal membership to the live groups at the final cut.
    // Learners already abandoned by fleet recovery are not valid artifact
    // producers and must not prevent surviving learners from finalizing.
    let final_members: HashSet<u32> = current_groups(&registry)
        .into_iter()
        .map(|group| group.member.learner_id)
        .collect();
    finalize_learners(&cfg, &st, &mut events, &registry, &final_members).await?;
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
    if !reports.insert(member.learner_id) {
        bail!("duplicate BUDGET_DONE");
    }
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

async fn collect_budget_reports(
    cfg: &Config,
    reports: &mut HashSet<u32>,
    events: &mut mpsc::Receiver<Event>,
    registry: &Registry,
) -> Result<()> {
    let target = cfg
        .learner_budget_steps
        .context("learner budget reports require --learner-budget-steps")?;
    while reports.len() < cfg.learners as usize {
        match events.recv().await.context("event channel closed")? {
            Event::BudgetDone {
                member,
                local_steps,
            } => {
                if !is_current_member(registry, member) {
                    bail!("BUDGET_DONE from a superseded learner");
                }
                record_budget_report(reports, target, member, local_steps)?;
            }
            Event::Disconnected { member } => warn!(
                learner_id = member.learner_id,
                generation = member.generation,
                "disconnected while waiting for learner-budget cutoff"
            ),
            Event::Hello { .. }
            | Event::Push { .. }
            | Event::Init { .. }
            | Event::FinalAck { .. } => {}
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
    quorum_ms: Option<u64>,
    grace_ms: Option<u64>,
    pushes: HashMap<Member, Push>,
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
}

fn take_next_commit<T>(ready: &mut BTreeMap<u64, T>, next_t: u64) -> Option<T> {
    ready.remove(&next_t)
}

#[derive(Debug, Eq, PartialEq)]
enum PushDisposition {
    Accepted,
    Duplicate,
    UnexpectedMember,
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
    if round.pushes.len() >= round.expected_members.len() {
        return RoundAction::Complete;
    }
    if round.pushes.len() >= round.quorum_size {
        return match round.grace_deadline {
            Some(deadline) if now >= deadline => RoundAction::Complete,
            _ => RoundAction::Wait,
        };
    }
    if now >= round.quorum_deadline {
        RoundAction::Restart
    } else {
        RoundAction::Wait
    }
}

fn fragment_available(rounds: &[Round], fragment_id: usize) -> bool {
    !rounds.iter().any(|round| round.p == fragment_id)
}

fn route_push(rounds: &mut [Round], member: Member, push: Push) -> PushDisposition {
    let Some(round) = rounds.iter_mut().find(|round| {
        round.t == push.global_step
            && round.p == push.fragment_id as usize
            && round.attempt == push.round_attempt
    }) else {
        return PushDisposition::OutOfRound;
    };
    if !round.expected_members.contains(&member) {
        return PushDisposition::UnexpectedMember;
    }
    if push.base_version > round.base_version {
        return PushDisposition::FutureBase;
    }
    if round.pushes.contains_key(&member) {
        return PushDisposition::Duplicate;
    }
    round.pushes.insert(member, push);
    PushDisposition::Accepted
}

/// Fix learner order and prepare owned matrix jobs without mutating global
/// optimizer state. This exact order is the reduction order on every run.
fn prepare_round_compute(st: &GlobalState, round: &Round) -> Result<PreparedMerge> {
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
    let mut outer_gradients = Vec::with_capacity(ordered_pushes.len());
    let mut weights = Vec::with_capacity(ordered_pushes.len());
    for (member, push) in ordered_pushes {
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
        weights.push(crate::merge::learner_weight(push.c_tokens, push.c_steps));
    }
    st.prepare_merge(round.p, round.base_version, &outer_gradients, &weights)
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
) -> Result<()> {
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
        ..
    } = round;
    let expected_t = st
        .global_step
        .checked_add(1)
        .context("global step overflow")?;
    if t != expected_t {
        bail!("attempted out-of-order commit t={t}, expected t={expected_t}");
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
    let responders: Vec<Member> = ordered_pushes.iter().map(|(member, _)| **member).collect();
    for (_, push) in ordered_pushes {
        st.record_merge(push.learner_id, push.c_steps, push.c_tokens);
    }

    // Broadcast the updated fragment.
    let payload = encode_bcast(st, p)?;
    for g in current_groups(registry) {
        let _ = g.send_large(MSG_BCAST_FRAGMENT, payload.clone()).await;
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
            &pushes,
            gnorm,
            ms,
        )?;
    }
    // Consistent cut: this round is fully applied and broadcast, and every
    // other in-flight round is still gathering (it has not touched state).
    // A crash-resume loses those gathers; their fragments simply merge on
    // a later cycle, which the quorum design already tolerates.
    if let Some(path) = &cfg.checkpoint_path {
        if cfg.checkpoint_every > 0 && t % cfg.checkpoint_every == 0 {
            st.save_checkpoint(path)?;
            info!(step = t, path = %path.display(), "checkpoint written");
        }
    }
    Ok(())
}

fn new_state_for(group: &Arc<Group>, cfg: &Config) -> Result<GlobalState> {
    let mut st = GlobalState::new_with_iso_backend(
        group.layout.clone(),
        cfg.outer_lr,
        cfg.outer_momentum,
        group.dtype,
        &cfg.iso_backend,
    )?;
    if cfg.delta_correction {
        st.delta_correction = Some(crate::merge::Heloco::default());
    }
    info!(
        iso_backend = %st.iso_backend_kind(),
        "initialized global state"
    );
    Ok(st)
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

async fn broadcast_all_fragments(st: &GlobalState, registry: &Registry) {
    let groups = current_groups(registry);
    // Fan each fragment out to every learner before moving to the next one.
    // Each learner has independent socket-writer queues, so this keeps all
    // links busy concurrently.  Iterating learner-by-learner here serializes
    // one full model transfer per learner (tens of GiB for large models).
    // Encode once and clone Bytes cheaply; the backing allocation remains
    // shared until every learner's writer has drained it.
    for p in 0..st.layout.fragments.len() {
        match encode_bcast(st, p) {
            Ok(payload) => {
                for group in &groups {
                    let _ = group.send_large(MSG_BCAST_FRAGMENT, payload.clone()).await;
                }
            }
            Err(e) => warn!("encode fragment {p} failed: {e}"),
        }
    }
}

/// Send every authoritative f32 fragment, then publish the version manifest
/// on the control stream. Data/control streams may reorder; learners cache
/// terminal fragments and use the manifest to decide when the complete cut
/// is locally available.
async fn send_final_cut(st: &GlobalState, group: &Arc<Group>) -> Result<()> {
    for p in 0..st.layout.fragments.len() {
        group
            .send_large(MSG_FINAL_FRAGMENT, encode_final_fragment(st, p)?)
            .await?;
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
    let mut eligible: HashMap<u32, HashSet<Member>> = HashMap::new();
    for group in current_groups(registry) {
        if expected.contains(&group.member.learner_id) {
            match send_final_cut(st, &group).await {
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

    let deadline = Instant::now() + Duration::from_secs(cfg.quorum_timeout_s);
    let mut acknowledged: HashMap<u32, Member> = HashMap::new();
    while acknowledged.len() < expected.len() {
        let remaining = deadline.saturating_duration_since(Instant::now());
        let event = tokio::time::timeout(remaining, events.recv())
            .await
            .map_err(|_| {
                let mut missing: Vec<u32> = expected
                    .iter()
                    .copied()
                    .filter(|id| !acknowledged.contains_key(id))
                    .collect();
                missing.sort_unstable();
                anyhow::anyhow!(
                    "finalization timed out after {}s waiting for learner acknowledgements: {:?}",
                    cfg.quorum_timeout_s,
                    missing
                )
            })?
            .context("event channel closed during finalization")?;
        match event {
            Event::FinalAck {
                member,
                global_step,
            } => {
                let learner_id = member.learner_id;
                match record_final_ack(
                    expected,
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
                let member = group.member;
                if expected.contains(&member.learner_id)
                    && !acknowledged.contains_key(&member.learner_id)
                    && is_current_member(registry, member)
                {
                    match send_final_cut(st, &group).await {
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
                warn!(
                    learner_id = member.learner_id,
                    generation = member.generation,
                    "learner disconnected during finalization; waiting for reconnect"
                );
            }
            Event::Push { .. } | Event::Init { .. } | Event::BudgetDone { .. } => {
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
        "all learners acknowledged final cut"
    );
    Ok(())
}

async fn send_all_fragments(st: &GlobalState, group: &Arc<Group>) {
    for p in 0..st.layout.fragments.len() {
        match encode_bcast(st, p) {
            Ok(payload) => {
                let _ = group.send_large(MSG_BCAST_FRAGMENT, payload).await;
            }
            Err(e) => warn!("encode fragment {p} failed: {e}"),
        }
    }
}

fn encode_bcast(st: &GlobalState, p: usize) -> Result<bytes::Bytes> {
    // All learners share one dtype (validated at HELLO); use the state dtype.
    // Broadcasts are full parameters, so a q4 session still sends bf16.
    let mut body = Vec::new();
    encode_tensor(bulk_dtype(st.wire_dtype), &st.params[p], &mut body)?;
    let mut payload = Vec::with_capacity(12 + body.len());
    payload.extend_from_slice(&(p as u32).to_le_bytes());
    payload.extend_from_slice(&st.versions[p].to_le_bytes());
    payload.extend_from_slice(&body);
    Ok(bytes::Bytes::from(payload))
}

fn encode_final_fragment(st: &GlobalState, p: usize) -> Result<bytes::Bytes> {
    // The coordinator's authoritative params and checkpoint are f32. The
    // terminal path is deliberately independent of the ordinary wire dtype
    // so bf16/q4 sessions do not round the artifact a second time.
    let mut body = Vec::new();
    encode_tensor(DTYPE_F32, &st.params[p], &mut body)?;
    let mut payload = Vec::with_capacity(12 + body.len());
    payload.extend_from_slice(&(p as u32).to_le_bytes());
    payload.extend_from_slice(&st.versions[p].to_le_bytes());
    payload.extend_from_slice(&body);
    Ok(bytes::Bytes::from(payload))
}

/// One JSONL record per merge: the event tape.
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
    pushes: &HashMap<Member, Push>,
    gnorm: f64,
    ms: u64,
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
    let weight_sum: f64 = pushes
        .values()
        .map(|p| crate::merge::learner_weight(p.c_tokens, p.c_steps))
        .sum();
    let mut responders: Vec<String> = pushes
        .iter()
        .map(|(member, p)| {
            let weight = crate::merge::learner_weight(p.c_tokens, p.c_steps);
            let contribution = if weight_sum > 0.0 { weight / weight_sum } else { 0.0 };
            let staleness = launch_base_version.saturating_sub(p.base_version);
            format!(
                "{{\"id\":{},\"generation\":{},\"base_version\":{},\"staleness\":{},\"c_steps\":{},\"c_tokens\":{},\"weight\":{},\"contribution\":{}}}",
                p.learner_id,
                member.generation,
                p.base_version,
                staleness,
                p.c_steps,
                p.c_tokens,
                weight,
                contribution
            )
        })
        .collect();
    responders.sort();
    let quorum_ms = json_opt_u64(quorum_ms);
    let grace_ms = json_opt_u64(grace_ms);
    let line = format!(
        "{{\"protocol_version\":{PROTOCOL_VERSION},\"delta_semantics\":\"local_minus_raw_anchor\",\"step\":{step},\"fragment\":{fragment},\"launch_base_version\":{launch_base_version},\"attempt\":{attempt},\"gnorm\":{gnorm},\"ms\":{ms},\"quorum\":{quorum},\"expected\":{},\"expected_members\":{},\"responded\":{},\"responded_members\":{},\"missed_grace\":{},\"missed_members\":{},\"quorum_ms\":{quorum_ms},\"grace_ms\":{grace_ms},\"sync_ms\":{sync_ms},\"responders\":[{}]}}\n",
        json_ids(&expected_learners),
        json_members(expected_members),
        json_ids(&responded),
        json_members(&responded_members),
        json_ids(&missed_grace),
        json_members(&missed_members),
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

fn json_opt_u64(value: Option<u64>) -> String {
    value.map_or_else(|| "null".to_string(), |v| v.to_string())
}

fn dump_state(st: &GlobalState, path: &std::path::Path) -> Result<()> {
    use std::io::Write;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(&(st.layout.fragments.len() as u32).to_le_bytes())?;
    for p in &st.params {
        f.write_all(&(p.len() as u64).to_le_bytes())?;
        for v in p {
            f.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const CAP: Duration = Duration::from_millis(1000);

    fn member(learner_id: u32, generation: u64) -> Member {
        Member {
            learner_id,
            generation,
        }
    }

    fn test_group(member: Member) -> Arc<Group> {
        let (control, _receiver) = mpsc::channel(1);
        Arc::new(Group {
            member,
            dtype: DTYPE_F32,
            layout: Layout {
                fragments: Vec::new(),
            },
            num_streams: 0,
            max_init_payload: 0,
            max_push_payload: 0,
            max_chunked_inner: 13,
            control,
            data: Mutex::new(HashMap::new()),
            msg_id: AtomicU64::new(0),
            rr: AtomicUsize::new(0),
            reasm: Mutex::new(HashMap::new()),
        })
    }

    #[test]
    fn duplicate_generation_registration_is_atomic_and_non_destructive() {
        let first = member(0, 10);
        let replacement = member(0, 11);
        let first_group = test_group(first);
        let mut registry = RegistryState::default();

        assert_eq!(registry.register_group(first_group.clone()), Ok(None));
        assert_eq!(registry.register_group(test_group(first)), Err(()));
        assert!(Arc::ptr_eq(
            registry.groups.get(&first).unwrap(),
            &first_group
        ));
        assert_eq!(
            registry.register_group(test_group(replacement)),
            Ok(Some(first))
        );
        assert_eq!(registry.current.get(&0), Some(&replacement));
        assert!(registry.groups.contains_key(&first));
        assert!(registry.groups.contains_key(&replacement));
    }

    #[test]
    fn budget_reports_are_exact_unique_and_cover_logical_learners() {
        let mut reports = HashSet::new();
        record_budget_report(&mut reports, 8, member(0, 10), 8).unwrap();
        record_budget_report(&mut reports, 8, member(1, 20), 8).unwrap();
        assert_eq!(reports.len(), 2);
        assert!(record_budget_report(&mut reports, 8, member(1, 21), 8).is_err());

        let mut wrong = HashSet::new();
        assert!(record_budget_report(&mut wrong, 8, member(0, 10), 7).is_err());
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
            quorum_ms: None,
            grace_ms: None,
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
        }
    }

    #[test]
    fn round_membership_rejects_mid_round_join_and_new_generation() {
        let captured = member(0, 10);
        let mut rounds = vec![test_round(vec![captured])];
        assert_eq!(
            route_push(&mut rounds, member(1, 20), test_push(5)),
            PushDisposition::UnexpectedMember
        );
        assert_eq!(
            route_push(&mut rounds, member(0, 11), test_push(5)),
            PushDisposition::UnexpectedMember
        );
        assert!(rounds[0].pushes.is_empty());
        assert_eq!(
            route_push(&mut rounds, captured, test_push(5)),
            PushDisposition::Accepted
        );
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
            route_push(&mut rounds, captured, test_push(6)),
            PushDisposition::FutureBase
        );
        assert_eq!(
            route_push(&mut rounds, captured, test_push(4)),
            PushDisposition::Accepted
        );
        assert_eq!(
            route_push(&mut rounds, captured, test_push(4)),
            PushDisposition::Duplicate
        );
        let mut wrong_round = test_push(4);
        wrong_round.global_step = 99;
        assert_eq!(
            route_push(&mut rounds, captured, wrong_round),
            PushDisposition::OutOfRound
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
            route_push(&mut rounds, second, test_push(5)),
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
            &pushes,
            0.5,
            44,
        )
        .unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert!(text.contains("\"expected\":[0,1]"));
        assert!(text.contains("\"expected_members\":[{\"id\":0,\"generation\":10}"));
        assert!(text.contains("\"responded\":[0]"));
        assert!(text.contains("\"missed_grace\":[1]"));
        assert!(text.contains("\"quorum_ms\":11"));
        assert!(text.contains("\"grace_ms\":22"));
        assert!(text.contains("\"sync_ms\":33"));
        assert!(text.contains("\"contribution\":1"));
        assert!(text.contains("\"protocol_version\":4"));
        assert!(text.contains("\"delta_semantics\":\"local_minus_raw_anchor\""));
    }
}
