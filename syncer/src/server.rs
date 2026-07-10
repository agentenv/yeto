//! Async TCP server implementing the syncer side of docs/PROTOCOL.md:
//! per-learner connection groups (control stream + striped data streams),
//! chunk reassembly, and the pull-driven quorum/grace merge scheduler
//! at the core of the training loop. Rounds are pipelined: up to
//! `Config::pipeline` fragments are in flight at once (arXiv 2604.21428's
//! "two fragments in flight"), so a slow quorum on one fragment never
//! delays pulling the next.

use std::collections::HashMap;
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
use crate::state::{GlobalState, Layout, MergeStats};

const CHUNK_SIZE: usize = 4 * 1024 * 1024;
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
    /// Require the configured quorum even after learners disconnect, and do
    /// not commit under-quorum rounds on timeout.
    pub strict_quorum: bool,
    pub total_steps: u64,
    pub outer_lr: f32,
    pub outer_lr_by_fragment: Option<Vec<f32>>,
    pub outer_momentum: f32,
    pub outer_optimizer: crate::merge::OuterOptimizer,
    pub outer_restart_cos_threshold: f32,
    pub final_state: Option<std::path::PathBuf>,
    /// Consistent-snapshot file; written every `checkpoint_every` rounds at
    /// the quiescent cut between rounds, resumed from when `resume` is set.
    pub checkpoint_path: Option<std::path::PathBuf>,
    pub checkpoint_every: u64,
    pub resume: bool,
    /// JSONL event tape: one record per merge.
    pub event_tape: Option<std::path::PathBuf>,
    /// Optional offline probe capture directory. When enabled, complete_round
    /// writes a pre-merge syncer checkpoint and admitted candidate fragment
    /// tensors before applying the outer step.
    pub probe_capture_dir: Option<std::path::PathBuf>,
    /// Capture every Nth outer step. 0 disables capture.
    pub probe_capture_every: u64,
}

struct OutFrame {
    msg_type: u8,
    parts: Vec<bytes::Bytes>,
}

struct PartialMsg {
    buf: Vec<u8>,
    filled: usize,
}

struct Group {
    learner_id: u32,
    dtype: u8,
    layout: Layout,
    layout_meta: Option<String>,
    control: mpsc::Sender<OutFrame>,
    data: Mutex<Vec<mpsc::Sender<OutFrame>>>,
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
            .map_err(|_| anyhow::anyhow!("learner {} control stream closed", self.learner_id))
    }

    /// Send a large inner frame, striped as CHUNK envelopes across data
    /// streams (or unchunked on the control stream when none exist).
    async fn send_large(&self, msg_type: u8, payload: bytes::Bytes) -> Result<()> {
        let streams: Vec<mpsc::Sender<OutFrame>> = self.data.lock().unwrap().clone();
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
                .map_err(|_| anyhow::anyhow!("learner {} data stream closed", self.learner_id))?;
            offset = end;
        }
        Ok(())
    }
}

enum Event {
    Hello { group: Arc<Group> },
    Init { fragment_id: u32, values: Vec<f32> },
    Push(Push),
    Disconnected { learner_id: u32 },
}

struct Push {
    learner_id: u32,
    fragment_id: u32,
    global_step: u64,
    base_version: u64,
    local_step: u64,
    c_steps: u32,
    c_tokens: u64,
    values: Vec<f32>,
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
struct StepRates(HashMap<u32, (Option<Instant>, u64, Option<f64>)>);

/// EMA smoothing: new estimate = α·sample + (1−α)·previous.
const STEP_EMA_ALPHA: f64 = 0.5;

impl StepRates {
    fn note(&mut self, learner_id: u32, local_step: u64, now: Instant) {
        let entry = self.0.entry(learner_id).or_insert((None, 0, None));
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

    /// Slowest learner's estimated step time, if any estimate exists.
    fn max_step_secs(&self) -> Option<f64> {
        self.0
            .values()
            .filter_map(|(_, _, e)| *e)
            .fold(None, |acc, v| Some(acc.map_or(v, |a: f64| a.max(v))))
    }
}

type Registry = Arc<Mutex<HashMap<u32, Arc<Group>>>>;

pub async fn run(cfg: Config) -> Result<()> {
    let listener = TcpListener::bind(("0.0.0.0", cfg.port))
        .await
        .with_context(|| format!("bind port {}", cfg.port))?;
    info!(port = cfg.port, "syncer listening");
    let (event_tx, event_rx) = mpsc::channel::<Event>(1024);
    let registry: Registry = Arc::new(Mutex::new(HashMap::new()));

    let accept_registry = registry.clone();
    tokio::spawn(async move {
        loop {
            match listener.accept().await {
                Ok((stream, peer)) => {
                    let reg = accept_registry.clone();
                    let tx = event_tx.clone();
                    tokio::spawn(async move {
                        if let Err(e) = handle_connection(stream, reg, tx).await {
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

async fn handle_connection(
    stream: TcpStream,
    registry: Registry,
    event_tx: mpsc::Sender<Event>,
) -> Result<()> {
    stream.set_nodelay(true)?;
    let (mut rd, wr) = stream.into_split();
    let first = read_frame(&mut rd).await?;
    match first.msg_type {
        MSG_HELLO => {
            let mut r = Reader(&first.payload);
            let learner_id = r.u32()?;
            let dtype = r.u8()?;
            let num_fragments = r.u32()?;
            let layout = Layout::decode(&mut r, num_fragments)?;
            let num_streams = r.u16()?;
            let layout_meta = if r.0.is_empty() {
                None
            } else {
                let n = r.u32()? as usize;
                let bytes = r.take(n)?;
                if !r.0.is_empty() {
                    bail!("trailing bytes after HELLO layout metadata");
                }
                Some(String::from_utf8(bytes.to_vec())?)
            };
            let (tx, rx) = mpsc::channel::<OutFrame>(WRITER_QUEUE);
            tokio::spawn(writer_task(wr, rx));
            let group = Arc::new(Group {
                learner_id,
                dtype,
                layout,
                layout_meta,
                control: tx,
                data: Mutex::new(Vec::new()),
                msg_id: AtomicU64::new(0),
                rr: AtomicUsize::new(0),
                reasm: Mutex::new(HashMap::new()),
            });
            registry.lock().unwrap().insert(learner_id, group.clone());
            info!(
                learner_id,
                num_streams, "learner connected (layout: {} fragments)", num_fragments
            );
            event_tx
                .send(Event::Hello {
                    group: group.clone(),
                })
                .await
                .ok();
            let res = read_loop(&mut rd, &group, &event_tx).await;
            registry.lock().unwrap().remove(&learner_id);
            event_tx.send(Event::Disconnected { learner_id }).await.ok();
            res
        }
        MSG_DATA_HELLO => {
            let mut r = Reader(&first.payload);
            let learner_id = r.u32()?;
            let _stream_idx = r.u16()?;
            // The control socket's HELLO may still be in flight; wait for it.
            let mut group = None;
            for _ in 0..200 {
                group = registry.lock().unwrap().get(&learner_id).cloned();
                if group.is_some() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            let group =
                group.with_context(|| format!("DATA_HELLO for unknown learner {learner_id}"))?;
            let (tx, rx) = mpsc::channel::<OutFrame>(WRITER_QUEUE);
            tokio::spawn(writer_task(wr, rx));
            group.data.lock().unwrap().push(tx);
            read_loop(&mut rd, &group, &event_tx).await
        }
        t => bail!("first frame must be HELLO/DATA_HELLO, got {t}"),
    }
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
        let frame = read_frame(rd).await?;
        match frame.msg_type {
            MSG_CHUNK => {
                if let Some(inner) = reassemble(group, &frame.payload)? {
                    dispatch_inner(group, inner.msg_type, &inner.payload, event_tx).await?;
                }
            }
            t => dispatch_inner(group, t, &frame.payload, event_tx).await?,
        }
    }
}

fn reassemble(group: &Arc<Group>, payload: &[u8]) -> Result<Option<Frame>> {
    let mut r = Reader(payload);
    let msg_id = r.u64()?;
    let total = r.u64()? as usize;
    let offset = r.u64()? as usize;
    let data = r.rest();
    if offset + data.len() > total {
        bail!("chunk overflow");
    }
    let mut reasm = group.reasm.lock().unwrap();
    let entry = reasm.entry(msg_id).or_insert_with(|| PartialMsg {
        buf: vec![0; total],
        filled: 0,
    });
    entry.buf[offset..offset + data.len()].copy_from_slice(data);
    entry.filled += data.len();
    if entry.filled < total {
        return Ok(None);
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
            let mut r = Reader(payload);
            let fragment_id = r.u32()?;
            let mut values = Vec::new();
            decode_tensor(bulk_dtype(group.dtype), r.rest(), &mut values)?;
            event_tx
                .send(Event::Init {
                    fragment_id,
                    values,
                })
                .await
                .ok();
        }
        MSG_PUSH_FRAGMENT => {
            let mut r = Reader(payload);
            let learner_id = r.u32()?;
            let fragment_id = r.u32()?;
            let global_step = r.u64()?;
            let base_version = r.u64()?;
            let local_step = r.u64()?;
            let c_steps = r.u32()?;
            let c_tokens = r.u64()?;
            let mut values = Vec::new();
            if group.dtype == DTYPE_Q4 {
                // Q4 pushes carry the *delta* against base_version; the
                // scheduler reconstructs θ = Θ(base_version) + δ.
                let frag = group
                    .layout
                    .fragments
                    .get(fragment_id as usize)
                    .with_context(|| format!("push for unknown fragment {fragment_id}"))?;
                decode_q4(r.rest(), frag.numel(), &mut values)?;
            } else {
                decode_tensor(group.dtype, r.rest(), &mut values)?;
            }
            event_tx
                .send(Event::Push(Push {
                    learner_id,
                    fragment_id,
                    global_step,
                    base_version,
                    local_step,
                    c_steps,
                    c_tokens,
                    values,
                }))
                .await
                .ok();
        }
        MSG_HEARTBEAT => {}
        t => bail!(
            "unexpected message type {t} from learner {}",
            group.learner_id
        ),
    }
    Ok(())
}

// --- scheduler -------------------------------------------------------------

async fn scheduler(
    cfg: Config,
    mut events: mpsc::Receiver<Event>,
    registry: Registry,
) -> Result<()> {
    let mut state: Option<GlobalState> = None;

    // Phase 1: wait until every fragment is initialized (via INIT_PARAMS or
    // a resumed checkpoint) and all expected learners have connected (late
    // joiners are still served afterwards).
    info!(
        expected = cfg.learners,
        "waiting for learners and INIT_PARAMS"
    );
    loop {
        let connected = registry.lock().unwrap().len() as u32;
        if let Some(st) = &state {
            if st.all_initialized() && connected >= cfg.learners {
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
                            if st.layout_meta != group.layout_meta {
                                bail!("checkpoint layout metadata does not match HELLO metadata");
                            }
                            info!(step = st.global_step, "resumed from checkpoint");
                        }
                    }
                    state = Some(st);
                } else if let Some(st) = &state {
                    validate_group_compatible(st, &group)?;
                }
            }
            Event::Init {
                fragment_id,
                values,
            } => {
                let st = state.as_mut().context("INIT before HELLO")?;
                st.init_fragment(fragment_id as usize, values)?;
                if st.all_initialized() {
                    info!("global parameters initialized");
                }
            }
            Event::Push(_) => warn!("push before initialization; dropped"),
            Event::Disconnected { learner_id } => warn!(learner_id, "disconnected during init"),
        }
    }
    let mut st = state.unwrap();
    let num_fragments = st.layout.fragments.len() as u64;
    let mut step_rates = StepRates::default();
    let mut last_sync_secs = 0.0f64; // previous round's merge+broadcast time

    // Send everyone the initial (or resumed) global parameters so all
    // learners start bit-identical (also serves recovery for late joiners).
    broadcast_all_fragments(&st, &registry).await;

    // Phase 2: the outer loop. One fragment per global step, round-robin,
    // with up to `pipeline` rounds in flight at once: while round t sits in
    // its quorum/grace window, round t+1's pull is already out, so sync
    // latency overlaps learner compute (the paper's τ=2 "two fragments in
    // flight"). Depth is clamped to the fragment count, so concurrent
    // rounds always target DISTINCT fragments and every merge touches
    // disjoint params/momentum. Rounds may complete out of order; versions
    // are per fragment and global_step advances monotonically.
    let depth = (cfg.pipeline.max(1) as u64).min(num_fragments) as usize;
    let manual_floor = Duration::from_millis(cfg.min_round_interval_ms);
    let mut next_launch = Instant::now(); // earliest allowed next round launch
    let mut next_t = st.global_step + 1;
    let mut inflight: Vec<Round> = Vec::new();
    while next_t <= cfg.total_steps || !inflight.is_empty() {
        // Keep the pipeline full (throttled by min_round_interval_ms).
        while inflight.len() < depth && next_t <= cfg.total_steps && Instant::now() >= next_launch {
            next_launch = Instant::now()
                + launch_interval(
                    manual_floor,
                    cfg.sync_interval_steps,
                    num_fragments as usize,
                    step_rates.max_step_secs(),
                );
            let t = next_t;
            next_t += 1;
            let p = ((t - 1) % num_fragments) as usize;
            let pull = {
                let mut b = Vec::with_capacity(12);
                b.extend_from_slice(&(p as u32).to_le_bytes());
                b.extend_from_slice(&t.to_le_bytes());
                bytes::Bytes::from(b)
            };
            for g in current_groups(&registry) {
                let _ = g.send_small(MSG_PULL_REQ, pull.clone()).await;
            }
            inflight.push(Round {
                t,
                p,
                pull,
                started: Instant::now(),
                quorum_deadline: Instant::now() + Duration::from_secs(cfg.quorum_timeout_s),
                grace_deadline: None,
                pushes: HashMap::new(),
            });
        }

        let connected = registry.lock().unwrap().len();
        let k = if cfg.strict_quorum {
            cfg.quorum as usize
        } else {
            (cfg.quorum as usize).min(connected.max(1))
        };

        // Arm the grace window of any round that just reached quorum.
        for r in inflight.iter_mut() {
            if r.pushes.len() >= k && r.grace_deadline.is_none() {
                let grace = adaptive_grace(
                    cfg.grace_tau,
                    cfg.grace_gamma,
                    step_rates.max_step_secs(),
                    r.started.elapsed().as_secs_f64(),
                    last_sync_secs,
                    Duration::from_millis(cfg.grace_ms),
                );
                r.grace_deadline = Some(Instant::now() + grace);
            }
        }

        // Complete every round that is ready. Adaptive mode may commit a
        // non-empty partial round at its deadline; strict mode requires the
        // configured quorum and re-pulls until it arrives.
        let now = Instant::now();
        let mut completed_any = false;
        let mut i = 0;
        while i < inflight.len() {
            let deadline = inflight[i]
                .grace_deadline
                .unwrap_or(inflight[i].quorum_deadline);
            let expired = now >= deadline;
            if round_completion_ready(
                inflight[i].pushes.len(),
                connected,
                k,
                cfg.strict_quorum,
                expired,
            ) {
                let round = inflight.remove(i);
                complete_round(&cfg, &mut st, &registry, &mut last_sync_secs, round).await?;
                completed_any = true;
                continue;
            }
            if expired {
                let r = &mut inflight[i];
                warn!(step = r.t, "round not ready at deadline; re-sending pull");
                for g in current_groups(&registry) {
                    let _ = g.send_small(MSG_PULL_REQ, r.pull.clone()).await;
                }
                r.quorum_deadline = Instant::now() + Duration::from_secs(cfg.quorum_timeout_s);
            }
            i += 1;
        }
        if completed_any {
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
        if inflight.len() < depth && next_t <= cfg.total_steps {
            earliest = Some(earliest.map_or(next_launch, |d| d.min(next_launch)));
        }
        let Some(earliest) = earliest else {
            continue; // everything launched has completed; loop re-evaluates
        };
        let timeout = earliest.saturating_duration_since(Instant::now());
        match tokio::time::timeout(timeout, events.recv()).await {
            Err(_) => continue, // deadline hit; loop re-evaluates
            Ok(None) => bail!("event channel closed"),
            Ok(Some(ev)) => match ev {
                Event::Push(push) => {
                    step_rates.note(push.learner_id, push.local_step, Instant::now());
                    // Route to the in-flight round the pull came from.
                    if let Some(r) = inflight
                        .iter_mut()
                        .find(|r| r.t == push.global_step && r.p == push.fragment_id as usize)
                    {
                        r.pushes.insert(push.learner_id, push);
                    } // else: stale response from a completed round; drop
                }
                Event::Hello { group } => {
                    // Rejoining learner: catch it up to the current state.
                    validate_group_compatible(&st, &group)?;
                    send_all_fragments(&st, &group).await;
                }
                Event::Init { .. } => {} // already initialized; ignore
                Event::Disconnected { learner_id } => {
                    warn!(learner_id, "learner disconnected");
                    for r in inflight.iter_mut() {
                        r.pushes.remove(&learner_id);
                    }
                }
            },
        }
    }

    if let Some(path) = &cfg.final_state {
        dump_state(&st, path)?;
        info!(path = %path.display(), "final global state written");
    }
    for g in current_groups(&registry) {
        let _ = g.send_small(MSG_SHUTDOWN, bytes::Bytes::new()).await;
    }
    info!("training complete after {} outer steps", cfg.total_steps);
    // Give writer tasks a moment to flush the shutdown frames.
    tokio::time::sleep(Duration::from_secs(2)).await;
    return Ok(());
}

fn round_completion_ready(
    pushes: usize,
    connected: usize,
    quorum: usize,
    strict_quorum: bool,
    expired: bool,
) -> bool {
    if strict_quorum {
        return pushes >= quorum;
    }
    pushes >= connected.max(1) || (expired && pushes > 0)
}

/// One in-flight sync round: the pull for fragment `p` at global step `t`
/// and the pushes gathered so far.
struct Round {
    t: u64,
    p: usize,
    pull: bytes::Bytes,
    started: Instant,
    quorum_deadline: Instant,
    grace_deadline: Option<Instant>,
    pushes: HashMap<u32, Push>,
}

/// Merge a gathered round, apply the outer step, broadcast, and record it.
/// Called from the single scheduler task, so merges are serialized even
/// with several rounds in flight; concurrent rounds target distinct
/// fragments, so each merge touches disjoint params/momentum.
async fn complete_round(
    cfg: &Config,
    st: &mut GlobalState,
    registry: &Registry,
    last_sync_secs: &mut f64,
    round: Round,
) -> Result<()> {
    let Round {
        t,
        p,
        started,
        mut pushes,
        ..
    } = round;
    let prev_version = st.versions[p];
    if st.wire_dtype == DTYPE_Q4 {
        // Q4 pushes are deltas anchored at the learner's base_version;
        // reconstruction needs Θ at that exact version, and the syncer
        // only holds the current value. A matching base is the steady
        // state (learners anchor on the last broadcast); anything older
        // is unreconstructable and dropped.
        pushes.retain(|id, push| {
            if push.base_version != prev_version {
                warn!(
                    learner_id = id,
                    step = t,
                    base = push.base_version,
                    expected = prev_version,
                    "stale q4 delta dropped"
                );
                return false;
            }
            true
        });
        for push in pushes.values_mut() {
            for (v, a) in push.values.iter_mut().zip(&st.params[p]) {
                *v += *a;
            }
        }
    }
    capture_round_candidates(cfg, st, p, t, prev_version, &pushes)?;
    let (mut learners, mut weights, mut ids) = (Vec::new(), Vec::new(), Vec::new());
    for (id, push) in &pushes {
        if push.base_version < prev_version {
            // The learner had not yet applied this fragment's last merge;
            // its delta is anchored further back. The weight formula
            // compensates (larger c_steps); recorded for the event tape.
            warn!(
                learner_id = id,
                step = t,
                base = push.base_version,
                expected = prev_version,
                "stale push admitted"
            );
        }
        learners.push(push.values.as_slice());
        weights.push(crate::merge::learner_weight(push.c_tokens, push.c_steps));
        ids.push(*id);
    }
    let sync_start = Instant::now();
    let merge_stats = st.merge_and_step(p, &learners, &weights)?;
    st.versions[p] = t;
    // Pipelined rounds can complete out of order; the global step only
    // moves forward.
    st.global_step = st.global_step.max(t);
    for push in pushes.values() {
        st.record_merge(push.learner_id, push.c_steps, push.c_tokens);
    }

    // Broadcast the updated fragment.
    let payload = encode_bcast(st, p)?;
    for g in current_groups(registry) {
        let _ = g.send_large(MSG_BCAST_FRAGMENT, payload.clone()).await;
    }
    *last_sync_secs = sync_start.elapsed().as_secs_f64();
    let ms = started.elapsed().as_millis() as u64;
    info!(
        step = t,
        fragment = p,
        responders = ?ids,
        gnorm = format!("{:.4}", merge_stats.gnorm),
        ms,
        "outer step"
    );
    if let Some(tape) = &cfg.event_tape {
        // Records land in completion order, which under pipelining is not
        // necessarily step order.
        append_tape(tape, t, p, &pushes, &weights, &merge_stats, ms);
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

fn capture_round_candidates(
    cfg: &Config,
    st: &GlobalState,
    fragment: usize,
    step: u64,
    prev_version: u64,
    pushes: &HashMap<u32, Push>,
) -> Result<()> {
    let Some(root) = &cfg.probe_capture_dir else {
        return Ok(());
    };
    if cfg.probe_capture_every == 0 || step % cfg.probe_capture_every != 0 {
        return Ok(());
    }
    if pushes.is_empty() {
        return Ok(());
    }
    let state_dir = root.join("states");
    let candidate_dir = root.join("candidates");
    std::fs::create_dir_all(&state_dir)?;
    std::fs::create_dir_all(&candidate_dir)?;
    let state_name = format!("state_before_step_{step:08}.ckpt");
    let state_path = state_dir.join(&state_name);
    st.save_checkpoint(&state_path)?;

    for push in pushes.values() {
        if push.values.len() != st.params[fragment].len() {
            bail!(
                "capture candidate step {step} fragment {fragment} learner {}: got {} values, expected {}",
                push.learner_id,
                push.values.len(),
                st.params[fragment].len()
            );
        }
        let candidate_name = format!(
            "candidate_step_{step:08}_fragment_{fragment:04}_learner_{:04}.f32",
            push.learner_id
        );
        let candidate_path = candidate_dir.join(&candidate_name);
        write_f32_file(&candidate_path, &push.values)?;
        append_probe_index(
            root,
            step,
            fragment,
            prev_version,
            st.global_step,
            push,
            &state_name,
            &candidate_name,
        )?;
    }
    Ok(())
}

fn write_f32_file(path: &std::path::Path, values: &[f32]) -> Result<()> {
    use std::io::Write;
    let tmp = path.with_extension("tmp");
    {
        let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
        for v in values {
            f.write_all(&v.to_le_bytes())?;
        }
        f.flush()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn append_probe_index(
    root: &std::path::Path,
    step: u64,
    fragment: usize,
    current_fragment_version: u64,
    syncer_global_step: u64,
    push: &Push,
    state_name: &str,
    candidate_name: &str,
) -> Result<()> {
    use std::io::Write;
    let index = root.join("index.jsonl");
    let line = format!(
        concat!(
            "{{",
            "\"schema\":\"syncer_probe_capture_v1\",",
            "\"oracle_scope\":\"syncer_current_global_pending_offline\",",
            "\"step\":{step},",
            "\"syncer_global_step\":{syncer_global_step},",
            "\"fragment\":{fragment},",
            "\"current_fragment_version\":{current_fragment_version},",
            "\"learner_id\":{learner_id},",
            "\"base_version\":{base_version},",
            "\"local_step\":{local_step},",
            "\"c_steps\":{c_steps},",
            "\"c_tokens\":{c_tokens},",
            "\"weight\":{weight},",
            "\"state_checkpoint\":\"states/{state_name}\",",
            "\"candidate_f32\":\"candidates/{candidate_name}\"",
            "}}\n"
        ),
        step = step,
        syncer_global_step = syncer_global_step,
        fragment = fragment,
        current_fragment_version = current_fragment_version,
        learner_id = push.learner_id,
        base_version = push.base_version,
        local_step = push.local_step,
        c_steps = push.c_steps,
        c_tokens = push.c_tokens,
        weight = crate::merge::learner_weight(push.c_tokens, push.c_steps),
        state_name = state_name,
        candidate_name = candidate_name,
    );
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(index)?
        .write_all(line.as_bytes())?;
    Ok(())
}

fn new_state_for(group: &Arc<Group>, cfg: &Config) -> Result<GlobalState> {
    let mut st = GlobalState::new(
        group.layout.clone(),
        group.layout_meta.clone(),
        cfg.outer_lr,
        cfg.outer_momentum,
        group.dtype,
    );
    if let Some(rates) = &cfg.outer_lr_by_fragment {
        if rates.len() != group.layout.fragments.len() {
            bail!(
                "--outer-lr-by-fragment has {} values, layout has {} fragments",
                rates.len(),
                group.layout.fragments.len()
            );
        }
        st.outer_lr_by_fragment = Some(rates.clone());
    }
    st.outer_optimizer = cfg.outer_optimizer;
    st.outer_restart_cos_threshold = cfg.outer_restart_cos_threshold;
    if cfg.delta_correction {
        st.delta_correction = Some(crate::merge::Heloco::default());
    }
    Ok(st)
}

fn validate_group_compatible(st: &GlobalState, group: &Arc<Group>) -> Result<()> {
    if st.wire_dtype != group.dtype {
        bail!(
            "learner {} dtype differs from established syncer state",
            group.learner_id
        );
    }
    if st.layout != group.layout {
        bail!(
            "learner {} fragment layout differs from established syncer state",
            group.learner_id
        );
    }
    if st.layout_meta != group.layout_meta {
        bail!(
            "learner {} layout metadata differs from established syncer state",
            group.learner_id
        );
    }
    Ok(())
}

fn current_groups(registry: &Registry) -> Vec<Arc<Group>> {
    registry.lock().unwrap().values().cloned().collect()
}

async fn broadcast_all_fragments(st: &GlobalState, registry: &Registry) {
    for g in current_groups(registry) {
        send_all_fragments(st, &g).await;
    }
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

/// One JSONL record per merge: the event tape.
fn append_tape(
    path: &std::path::Path,
    step: u64,
    fragment: usize,
    pushes: &HashMap<u32, Push>,
    _weights: &[f64],
    stats: &MergeStats,
    ms: u64,
) {
    use std::io::Write;
    let line = format_tape_line(step, fragment, pushes, stats, ms);
    let res = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .and_then(|mut f| f.write_all(line.as_bytes()));
    if let Err(e) = res {
        warn!("event tape write failed: {e}");
    }
}

fn optional_json_number(value: Option<f64>) -> String {
    match value {
        Some(value) if value.is_finite() => value.to_string(),
        _ => "null".to_string(),
    }
}

fn json_number(value: f64) -> String {
    optional_json_number(value.is_finite().then_some(value))
}

fn format_tape_line(
    step: u64,
    fragment: usize,
    pushes: &HashMap<u32, Push>,
    stats: &MergeStats,
    ms: u64,
) -> String {
    let mut responders: Vec<String> = pushes
        .values()
        .map(|p| {
            format!(
                "{{\"id\":{},\"base_version\":{},\"c_steps\":{},\"c_tokens\":{},\"weight\":{}}}",
                p.learner_id,
                p.base_version,
                p.c_steps,
                p.c_tokens,
                crate::merge::learner_weight(p.c_tokens, p.c_steps)
            )
        })
        .collect();
    responders.sort();
    let outer = stats.outer;
    let gnorm = json_number(stats.gnorm);
    let outer_step_norm = json_number(outer.applied_step_norm);
    let outer_direction_cosine = optional_json_number(outer.direction_delta_cosine);
    let outer_history_current_ratio = optional_json_number(outer.history_current_norm_ratio);
    format!(
        "{{\"step\":{step},\"fragment\":{fragment},\"gnorm\":{gnorm},\"ms\":{ms},\"responders\":[{}],\"outer_step_norm\":{outer_step_norm},\"outer_direction_cosine\":{outer_direction_cosine},\"outer_history_current_ratio\":{outer_history_current_ratio},\"outer_restarted\":{}}}\n",
        responders.join(","),
        outer.restarted
    )
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

    fn merge_stats(
        gnorm: f64,
        applied_step_norm: f64,
        direction_delta_cosine: Option<f64>,
        history_current_norm_ratio: Option<f64>,
        restarted: bool,
    ) -> MergeStats {
        MergeStats {
            gnorm,
            outer: crate::merge::OuterStepStats {
                applied_step_norm,
                direction_delta_cosine,
                history_current_norm_ratio,
                restarted,
            },
        }
    }

    #[test]
    fn event_tape_line_preserves_old_fields_and_adds_outer_stats() {
        let mut pushes = HashMap::new();
        pushes.insert(
            4,
            Push {
                learner_id: 4,
                fragment_id: 2,
                global_step: 9,
                base_version: 7,
                local_step: 21,
                c_steps: 10,
                c_tokens: 100,
                values: Vec::new(),
            },
        );
        let stats = merge_stats(2.5, 0.75, Some(-0.25), Some(3.0), true);
        let line = format_tape_line(9, 2, &pushes, &stats, 17);

        assert!(
            line.starts_with("{\"step\":9,\"fragment\":2,\"gnorm\":2.5,\"ms\":17,\"responders\":[")
        );
        assert!(line.contains(
            "{\"id\":4,\"base_version\":7,\"c_steps\":10,\"c_tokens\":100,\"weight\":1000}"
        ));
        assert!(line.contains("\"outer_step_norm\":0.75"));
        assert!(line.contains("\"outer_direction_cosine\":-0.25"));
        assert!(line.contains("\"outer_history_current_ratio\":3"));
        assert!(line.contains("\"outer_restarted\":true"));
        assert!(line.ends_with("}\n"));
    }

    #[test]
    fn event_tape_uses_null_for_undefined_outer_ratios() {
        let stats = merge_stats(0.0, 0.0, None, None, false);
        let line = format_tape_line(1, 0, &HashMap::new(), &stats, 0);
        assert!(line.contains("\"gnorm\":0"));
        assert!(line.contains("\"outer_step_norm\":0"));
        assert!(line.contains("\"outer_direction_cosine\":null"));
        assert!(line.contains("\"outer_history_current_ratio\":null"));
        assert!(line.contains("\"outer_restarted\":false"));
        assert!(!line.contains("NaN"));
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
    fn strict_quorum_never_completes_an_under_quorum_tail() {
        assert!(!round_completion_ready(1, 1, 4, true, false));
        assert!(!round_completion_ready(1, 1, 4, true, true));
        assert!(round_completion_ready(4, 4, 4, true, false));
    }

    #[test]
    fn adaptive_quorum_preserves_disconnect_and_timeout_behavior() {
        assert!(round_completion_ready(1, 1, 1, false, false));
        assert!(round_completion_ready(1, 4, 4, false, true));
        assert!(!round_completion_ready(1, 4, 4, false, false));
    }

    #[test]
    fn step_rates_estimate_from_consecutive_pushes() {
        let mut rates = StepRates::default();
        let t0 = Instant::now();
        rates.note(1, 100, t0);
        assert_eq!(rates.max_step_secs(), None); // one sample: no estimate yet
        rates.note(1, 110, t0 + Duration::from_secs(5));
        let est = rates.max_step_secs().unwrap();
        assert!(
            (est - 0.5).abs() < 1e-9,
            "10 steps over 5s = 0.5 s/step, got {est}"
        );
        // A slower learner dominates the estimate.
        rates.note(2, 10, t0);
        rates.note(2, 12, t0 + Duration::from_secs(4));
        assert!((rates.max_step_secs().unwrap() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn step_rates_smooth_with_ema() {
        let mut rates = StepRates::default();
        let t0 = Instant::now();
        rates.note(1, 0, t0);
        rates.note(1, 10, t0 + Duration::from_secs(5)); // seeds EMA at 0.5 s/step
                                                        // A one-off 10x-slower interval (2 steps over 10s = 5.0 s/step sample)
                                                        // must not replace the estimate wholesale: EMA -> 0.5·5.0 + 0.5·0.5.
        rates.note(1, 12, t0 + Duration::from_secs(15));
        let est = rates.max_step_secs().unwrap();
        assert!(
            (est - 2.75).abs() < 1e-9,
            "EMA of 0.5 then 5.0 should be 2.75, got {est}"
        );
    }

    #[test]
    fn step_rates_survive_learner_restart() {
        let mut rates = StepRates::default();
        let t0 = Instant::now();
        rates.note(1, 100, t0);
        rates.note(1, 5, t0 + Duration::from_secs(1)); // local_step went backwards
        assert_eq!(rates.max_step_secs(), None);
        rates.note(1, 15, t0 + Duration::from_secs(6));
        assert!((rates.max_step_secs().unwrap() - 0.5).abs() < 1e-9);
    }
}
