//! Async TCP server implementing the syncer side of docs/PROTOCOL.md:
//! per-learner connection groups (control stream + striped data streams),
//! chunk reassembly, and the pull-driven quorum/grace merge scheduler
//! at the core of the training loop.

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
use crate::state::{GlobalState, Layout};

const CHUNK_SIZE: usize = 4 * 1024 * 1024;
const WRITE_TIMEOUT: Duration = Duration::from_secs(180);
const WRITER_QUEUE: usize = 128;

#[derive(Clone)]
pub struct Config {
    pub port: u16,
    pub learners: u32,
    pub quorum: u32,
    pub grace_ms: u64,
    pub quorum_timeout_s: u64,
    pub total_steps: u64,
    pub outer_lr: f32,
    pub outer_momentum: f32,
    pub final_state: Option<std::path::PathBuf>,
    /// Consistent-snapshot file; written every `checkpoint_every` rounds at
    /// the quiescent cut between rounds, resumed from when `resume` is set.
    pub checkpoint_path: Option<std::path::PathBuf>,
    pub checkpoint_every: u64,
    pub resume: bool,
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
}

struct Group {
    learner_id: u32,
    dtype: u8,
    layout: Layout,
    control: mpsc::Sender<OutFrame>,
    data: Mutex<Vec<mpsc::Sender<OutFrame>>>,
    msg_id: AtomicU64,
    rr: AtomicUsize,
    reasm: Mutex<HashMap<u64, PartialMsg>>,
}

impl Group {
    async fn send_small(&self, msg_type: u8, payload: bytes::Bytes) -> Result<()> {
        self.control
            .send(OutFrame { msg_type, parts: vec![payload] })
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
    c_steps: u32,
    c_tokens: u64,
    values: Vec<f32>,
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

async fn handle_connection(stream: TcpStream, registry: Registry, event_tx: mpsc::Sender<Event>) -> Result<()> {
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
            let (tx, rx) = mpsc::channel::<OutFrame>(WRITER_QUEUE);
            tokio::spawn(writer_task(wr, rx));
            let group = Arc::new(Group {
                learner_id,
                dtype,
                layout,
                control: tx,
                data: Mutex::new(Vec::new()),
                msg_id: AtomicU64::new(0),
                rr: AtomicUsize::new(0),
                reasm: Mutex::new(HashMap::new()),
            });
            registry.lock().unwrap().insert(learner_id, group.clone());
            info!(learner_id, num_streams, "learner connected (layout: {} fragments)", num_fragments);
            event_tx
                .send(Event::Hello { group: group.clone() })
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
            let group = group.with_context(|| format!("DATA_HELLO for unknown learner {learner_id}"))?;
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

async fn read_loop(rd: &mut (impl tokio::io::AsyncReadExt + Unpin), group: &Arc<Group>, event_tx: &mpsc::Sender<Event>) -> Result<()> {
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
    let entry = reasm
        .entry(msg_id)
        .or_insert_with(|| PartialMsg { buf: vec![0; total], filled: 0 });
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
    Ok(Some(Frame { msg_type, payload: payload.to_vec() }))
}

async fn dispatch_inner(group: &Arc<Group>, msg_type: u8, payload: &[u8], event_tx: &mpsc::Sender<Event>) -> Result<()> {
    match msg_type {
        MSG_INIT_PARAMS => {
            let mut r = Reader(payload);
            let fragment_id = r.u32()?;
            let mut values = Vec::new();
            decode_tensor(group.dtype, r.rest(), &mut values)?;
            event_tx.send(Event::Init { fragment_id, values }).await.ok();
        }
        MSG_PUSH_FRAGMENT => {
            let mut r = Reader(payload);
            let learner_id = r.u32()?;
            let fragment_id = r.u32()?;
            let global_step = r.u64()?;
            let base_version = r.u64()?;
            let _local_step = r.u64()?;
            let c_steps = r.u32()?;
            let c_tokens = r.u64()?;
            let mut values = Vec::new();
            decode_tensor(group.dtype, r.rest(), &mut values)?;
            event_tx
                .send(Event::Push(Push {
                    learner_id,
                    fragment_id,
                    global_step,
                    base_version,
                    c_steps,
                    c_tokens,
                    values,
                }))
                .await
                .ok();
        }
        MSG_HEARTBEAT => {}
        t => bail!("unexpected message type {t} from learner {}", group.learner_id),
    }
    Ok(())
}

// --- scheduler -------------------------------------------------------------

async fn scheduler(cfg: Config, mut events: mpsc::Receiver<Event>, registry: Registry) -> Result<()> {
    let mut state: Option<GlobalState> = None;

    // Phase 1: wait until every fragment is initialized (via INIT_PARAMS or
    // a resumed checkpoint) and all expected learners have connected (late
    // joiners are still served afterwards).
    info!(expected = cfg.learners, "waiting for learners and INIT_PARAMS");
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
                            info!(step = st.global_step, "resumed from checkpoint");
                        }
                    }
                    state = Some(st);
                }
            }
            Event::Init { fragment_id, values } => {
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

    // Send everyone the initial (or resumed) global parameters so all
    // learners start bit-identical (also serves recovery for late joiners).
    broadcast_all_fragments(&st, &registry).await;

    // Phase 2: the outer loop. One fragment per global step, round-robin.
    for t in (st.global_step + 1)..=cfg.total_steps {
        let p = ((t - 1) % num_fragments) as usize;
        let round_start = Instant::now();
        let pull = {
            let mut b = Vec::with_capacity(12);
            b.extend_from_slice(&(p as u32).to_le_bytes());
            b.extend_from_slice(&t.to_le_bytes());
            bytes::Bytes::from(b)
        };
        for g in current_groups(&registry) {
            let _ = g.send_small(MSG_PULL_REQ, pull.clone()).await;
        }

        // Gather pushes for round t: quorum K, then grace window.
        let mut pushes: HashMap<u32, Push> = HashMap::new();
        let quorum_deadline = Instant::now() + Duration::from_secs(cfg.quorum_timeout_s);
        let mut grace_deadline: Option<Instant> = None;
        loop {
            let connected = registry.lock().unwrap().len();
            let k = (cfg.quorum as usize).min(connected.max(1));
            if pushes.len() >= connected.max(1) {
                break; // everyone answered; no reason to wait
            }
            if pushes.len() >= k && grace_deadline.is_none() {
                grace_deadline = Some(Instant::now() + Duration::from_millis(cfg.grace_ms));
            }
            let deadline = grace_deadline.unwrap_or(quorum_deadline);
            let timeout = deadline.saturating_duration_since(Instant::now());
            if timeout.is_zero() {
                if pushes.is_empty() {
                    warn!(step = t, "quorum timeout with zero pushes; re-sending pull");
                    for g in current_groups(&registry) {
                        let _ = g.send_small(MSG_PULL_REQ, pull.clone()).await;
                    }
                    continue;
                }
                break;
            }
            match tokio::time::timeout(timeout, events.recv()).await {
                Err(_) => continue, // deadline hit; loop re-evaluates
                Ok(None) => bail!("event channel closed"),
                Ok(Some(ev)) => match ev {
                    Event::Push(push) => {
                        if push.fragment_id as usize == p && push.global_step == t {
                            pushes.insert(push.learner_id, push);
                        } // else: stale response from an earlier round; drop
                    }
                    Event::Hello { group } => {
                        // Rejoining learner: catch it up to the current state.
                        send_all_fragments(&st, &group).await;
                    }
                    Event::Init { .. } => {} // already initialized; ignore
                    Event::Disconnected { learner_id } => {
                        warn!(learner_id, step = t, "learner disconnected");
                        pushes.remove(&learner_id);
                    }
                },
            }
        }

        let (mut learners, mut weights, mut ids) = (Vec::new(), Vec::new(), Vec::new());
        let prev_version = st.versions[p];
        for (id, push) in &pushes {
            if push.base_version < prev_version {
                // The learner had not yet applied this fragment's last merge;
                // its delta is anchored further back. The weight formula
                // compensates (larger c_steps); recorded for the event tape.
                warn!(learner_id = id, step = t, base = push.base_version, expected = prev_version,
                      "stale push admitted");
            }
            learners.push(push.values.as_slice());
            weights.push(crate::merge::learner_weight(push.c_tokens, push.c_steps));
            ids.push(*id);
        }
        let gnorm = st.merge_and_step(p, &learners, &weights)?;
        st.versions[p] = t;
        st.global_step = t;
        for push in pushes.values() {
            st.record_merge(push.learner_id, push.c_steps, push.c_tokens);
        }

        // Broadcast the updated fragment.
        let payload = encode_bcast(&st, p)?;
        for g in current_groups(&registry) {
            let _ = g.send_large(MSG_BCAST_FRAGMENT, payload.clone()).await;
        }
        let ms = round_start.elapsed().as_millis() as u64;
        info!(
            step = t,
            fragment = p,
            responders = ?ids,
            gnorm = format!("{gnorm:.4}"),
            ms,
            "outer step"
        );
        if let Some(tape) = &cfg.event_tape {
            append_tape(tape, t, p, &pushes, &weights, gnorm, ms);
        }
        // Quiescent cut: round t is fully applied and broadcast, round t+1
        // has not begun — the snapshot is consistent by construction.
        if let Some(path) = &cfg.checkpoint_path {
            if cfg.checkpoint_every > 0 && t % cfg.checkpoint_every == 0 {
                st.save_checkpoint(path)?;
                info!(step = t, path = %path.display(), "checkpoint written");
            }
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
    Ok(())
}

fn new_state_for(group: &Arc<Group>, cfg: &Config) -> Result<GlobalState> {
    Ok(GlobalState::new(
        group.layout.clone(),
        cfg.outer_lr,
        cfg.outer_momentum,
        group.dtype,
    ))
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
    let mut body = Vec::new();
    encode_tensor(st.wire_dtype, &st.params[p], &mut body)?;
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
    gnorm: f64,
    ms: u64,
) {
    use std::io::Write;
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
    let line = format!(
        "{{\"step\":{step},\"fragment\":{fragment},\"gnorm\":{gnorm},\"ms\":{ms},\"responders\":[{}]}}\n",
        responders.join(",")
    );
    let res = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .and_then(|mut f| f.write_all(line.as_bytes()));
    if let Err(e) = res {
        warn!("event tape write failed: {e}");
    }
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
