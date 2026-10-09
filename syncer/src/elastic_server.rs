//! Elastic-mode server: runs `ElasticCoordinator` behind a TCP listener and
//! applies its `StepPlan` to a flat f32 parameter vector. Used only when
//! `--island-scheduling-mode elastic`; the legacy server path in
//! `server.rs` never reaches this file.
//!
//! Messages: JOIN/JOIN_ACK, LEAVE, LEASE_HEARTBEAT, ELASTIC_INIT (seed the
//! base), DELTA_TENSOR (update + merge-weight inputs), ELASTIC_BASE
//! (broadcast after each step), and the metadata-only DELTA_READY (control
//! plane only; refused for merging once a base exists).
//! Merge: plan weights (γ^lag for carried deltas, 0 for a joiner's first
//! round, departed islands' deltas never stored into the plan) feed
//! `merge::merge_avg`, then `merge::nesterov_step` with --outer-lr /
//! --outer-momentum. The legacy per-fragment RDA/ISO/HeLoCo pipeline and
//! layout/fragment framing are not used here.
//! Not yet: SAMPLE_INDEX handling (refused), D-S5 status export, D-S6
//! checkpoint fields.

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{bail, ensure, Context, Result};
use tokio::io::AsyncWriteExt;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::elastic::{ElasticCoordinator, ElasticMsg, ElasticParams};
use crate::protocol::*;

pub struct ElasticServerConfig {
    pub params: ElasticParams,
    pub key: Vec<u8>,
    pub syncer_epoch: u64,
    pub lease_s: f64,
    pub total_steps: u64,
    pub event_tape: Option<PathBuf>,
    pub tick: Duration,
    pub outer_lr: f32,
    pub outer_momentum: f32,
    /// After total_steps: keep serving FINISHED for this long (or until
    /// every member has left) so late islands exit cleanly.
    pub final_grace: Duration,
    /// D-S6 checkpoint (elastic-only format, tmp + fsync + rename).
    pub checkpoint_path: Option<PathBuf>,
    pub checkpoint_every: u64,
    pub resume: bool,
    /// secret-handling-hardening: backend identity pinned by the head's
    /// config. None keeps "first accepted JOIN pins it".
    pub expected_backend_identity: Option<[u8; 32]>,
}

/// v2 (s17-elastic-identity) adds the pinned backend identity; v1
/// checkpoints are refused (version boundary, see hash-migration.md).
const SERVER_STATE_MAGIC: &[u8; 8] = b"YELSRV2\0";
const SERVER_STATE_MAGIC_V1: &[u8; 8] = b"YELSRV1\0";

fn group_label(g: &str) -> &str {
    if g.is_empty() { "未声明" } else { g }
}

fn hex32(h: &[u8; 32]) -> String {
    if h.iter().all(|b| *b == 0) {
        return "<not declared>".into();
    }
    h.iter().map(|b| format!("{b:02x}")).collect()
}

fn put_f32s(b: &mut Vec<u8>, v: &[f32]) {
    b.extend_from_slice(&(v.len() as u64).to_le_bytes());
    v.iter().for_each(|x| b.extend_from_slice(&x.to_le_bytes()));
}

fn get_f32s(r: &mut Reader) -> Result<Vec<f32>> {
    let n = usize::try_from(r.u64()?)?;
    let bytes = r.take(n.checked_mul(4).context("tensor length overflow")?)?;
    let mut out = Vec::new();
    decode_tensor(DTYPE_F32, bytes, &mut out)?;
    Ok(out)
}

fn write_atomic(path: &std::path::Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::File::open(dir)?.sync_all()?;
    }
    Ok(())
}

/// D-S5 status snapshot refresh period.
const STATUS_EVERY: Duration = Duration::from_millis(250);

/// Largest elastic frame accepted (tensor messages included).
const MAX_ELASTIC_FRAME: u64 = 4 << 30;

type Outbox = mpsc::UnboundedSender<(u8, Vec<u8>)>;

struct Shared {
    coord: ElasticCoordinator,
    /// yeto-framework-decoupling 6.2a: backend identity hash pinned by the
    /// first accepted JOIN (like the strict HELLO session contract); every
    /// later JOIN must carry the same 32 bytes. Kept for the whole run and in
    /// the checkpoint, so a Miles session never admits a verl island.
    backend_identity: Option<[u8; 32]>,
    /// yeto-framework-decoupling 7.7c: compat_group ("<vendor>-<card>")
    /// pinned by the first accepted JOIN. A JOIN from another group is refused
    /// with both groups named (phase 1: tolerance table empty). Not in the
    /// checkpoint: the group is also inside the identity hash, so after
    /// --resume the identity pin still refuses it; the readable pin is set
    /// again by the first JOIN whose identity matches.
    compat_group: Option<String>,
    flushed: usize,
    round_started: Instant,
    tape: Option<std::fs::File>,
    key: Vec<u8>,
    outer_lr: f32,
    outer_momentum: f32,
    params: Option<Vec<f32>>,
    momentum_buf: Vec<f32>,
    /// Updates (θ − base) keyed by (island, base_version) awaiting a step.
    stored: HashMap<(u32, u64), Vec<f32>>,
    outboxes: HashMap<u32, Outbox>,
    checkpoint_path: Option<PathBuf>,
    checkpoint_every: u64,
    /// D-S5 snapshot and P3 index files, next to --event-tape.
    status_path: Option<PathBuf>,
    sample_index_path: Option<PathBuf>,
    last_status_write: Option<Instant>,
    total_steps: u64,
    /// When FINISHED was broadcast (start of the final grace window).
    finished_at: Option<Instant>,
}

fn policy_hash(params: &[f32]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    params.iter().for_each(|x| h.update(x.to_le_bytes()));
    h.finalize().into()
}

impl Shared {
    fn finished(&self) -> bool {
        self.coord.outer_version >= self.total_steps
    }

    fn finished_msg(&self) -> (u8, Vec<u8>) {
        ElasticMsg::Finished {
            syncer_epoch: self.coord.syncer_epoch,
            outer_version: self.coord.outer_version,
            policy_hash: self.coord.published.get(&self.coord.outer_version).copied().unwrap_or([0; 32]),
        }
        .encode(&self.key)
    }

    fn publish_hash(&mut self) {
        if let Some(p) = &self.params {
            let h = policy_hash(p);
            self.coord.published.insert(self.coord.outer_version, h);
        }
    }

    fn write_status(&mut self, now: f64, soft_remaining_s: f64) -> Result<()> {
        if let Some(path) = &self.status_path {
            write_atomic(
                path,
                self.coord
                    .status_json(now, soft_remaining_s, if self.finished() { "finished" } else { "running" })
                    .as_bytes(),
            )
                .with_context(|| format!("write status {}", path.display()))?;
        }
        self.last_status_write = Some(Instant::now());
        Ok(())
    }

    fn append_sample_index(&self, e: &crate::elastic::SampleIndexEntry, verdict: u8, reason: &str, lag: u64) -> Result<()> {
        if let Some(path) = &self.sample_index_path {
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
            writeln!(
                f,
                "{{\"syncer_epoch\":{},\"verdict\":\"{}\",\"reason\":\"{reason}\",\"outer_lag\":{lag},\"entry\":{}}}",
                self.coord.syncer_epoch,
                crate::elastic::verdict_name(verdict),
                e.to_json()
            )?;
            f.flush()?;
        }
        Ok(())
    }

    fn encode_checkpoint(&self) -> Vec<u8> {
        let mut b = SERVER_STATE_MAGIC.to_vec();
        match &self.backend_identity {
            Some(h) => {
                b.push(1);
                b.extend_from_slice(h);
            }
            None => b.push(0),
        }
        let coord = self.coord.encode_state();
        b.extend_from_slice(&(coord.len() as u64).to_le_bytes());
        b.extend_from_slice(&coord);
        match &self.params {
            Some(p) => {
                b.push(1);
                put_f32s(&mut b, p);
                put_f32s(&mut b, &self.momentum_buf);
            }
            None => b.push(0),
        }
        let mut keys: Vec<_> = self.stored.keys().copied().collect();
        keys.sort_unstable();
        b.extend_from_slice(&(keys.len() as u32).to_le_bytes());
        for k in keys {
            b.extend_from_slice(&k.0.to_le_bytes());
            b.extend_from_slice(&k.1.to_le_bytes());
            put_f32s(&mut b, &self.stored[&k]);
        }
        b
    }

    fn write_checkpoint(&self) -> Result<()> {
        if let Some(path) = &self.checkpoint_path {
            write_atomic(path, &self.encode_checkpoint())
                .with_context(|| format!("write elastic checkpoint {}", path.display()))?;
        }
        Ok(())
    }

    /// Restore coordinator, base, momentum and carried tensors.
    fn restore(&mut self, bytes: &[u8], lease_s: f64, now: f64) -> Result<()> {
        let mut r = Reader(bytes);
        let magic = r.take(8)?;
        ensure!(
            magic != SERVER_STATE_MAGIC_V1,
            "elastic syncer checkpoint predates the backend identity field (YELSRV1); \
             it cannot be resumed by this syncer (version boundary s17-elastic-identity)"
        );
        ensure!(magic == SERVER_STATE_MAGIC, "not an elastic syncer checkpoint");
        self.backend_identity = match r.u8()? {
            0 => None,
            1 => Some(r.take(32)?.try_into().context("backend identity")?),
            v => bail!("invalid backend identity flag {v} in elastic checkpoint"),
        };
        let n = usize::try_from(r.u64()?)?;
        self.coord = ElasticCoordinator::decode_state(self.coord.params, lease_s, r.take(n)?, now)?;
        if r.u8()? == 1 {
            self.params = Some(get_f32s(&mut r)?);
            self.momentum_buf = get_f32s(&mut r)?;
        }
        for _ in 0..r.u32()? {
            let k = (r.u32()?, r.u64()?);
            self.stored.insert(k, get_f32s(&mut r)?);
        }
        ensure!(r.remaining() == 0, "trailing bytes in elastic syncer checkpoint");
        Ok(())
    }

    fn flush_tape(&mut self) -> Result<()> {
        let new = &self.coord.tape[self.flushed..];
        if let Some(f) = self.tape.as_mut() {
            for ev in new {
                writeln!(f, "{}", ev.to_json(self.coord.syncer_epoch))?;
            }
            f.flush()?;
        }
        for ev in new {
            info!(kind = ev.kind(), outer_version = self.coord.outer_version, "elastic event");
        }
        self.flushed = self.coord.tape.len();
        Ok(())
    }

    fn base_msg(&self) -> Option<(u8, Vec<u8>)> {
        self.params.as_ref().map(|p| {
            ElasticMsg::ElasticBase {
                syncer_epoch: self.coord.syncer_epoch,
                outer_version: self.coord.outer_version,
                params: p.clone(),
            }
            .encode(&self.key)
        })
    }

    /// Bandwidth record for one tensor frame (ELASTIC_INIT / DELTA_TENSOR
    /// received, ELASTIC_BASE sent): frame bytes incl. the 13-byte header
    /// and wall seconds from header to last payload byte. Written straight
    /// to the event tape as `kind:"transfer"` (not part of the coordinator
    /// tape, so checkpoints and ledger golden cases are unchanged).
    fn record_transfer(&mut self, direction: &str, msg_type: u8, island: Option<u32>, bytes: u64, seconds: f64) -> Result<()> {
        let name = match msg_type {
            MSG_ELASTIC_INIT => "elastic_init",
            MSG_DELTA_TENSOR => "delta_tensor",
            MSG_ELASTIC_BASE => "elastic_base",
            _ => "other",
        };
        let island_json = island.map(|i| i.to_string()).unwrap_or_else(|| "null".into());
        let secs = if seconds.is_finite() { format!("{seconds:?}") } else { "null".into() };
        let line = format!(
            "{{\"kind\":\"transfer\",\"syncer_epoch\":{},\"direction\":\"{direction}\",\"msg\":\"{name}\",\"island_id\":{island_json},\"bytes\":{bytes},\"seconds\":{secs},\"outer_version\":{}}}",
            self.coord.syncer_epoch, self.coord.outer_version
        );
        if let Some(f) = self.tape.as_mut() {
            writeln!(f, "{line}")?;
            f.flush()?;
        }
        info!(direction, msg = name, island = ?island, bytes, seconds, "elastic transfer");
        Ok(())
    }

    /// Forget updates from islands that are no longer members.
    fn purge_departed(&mut self) {
        let coord = &self.coord;
        self.stored.retain(|(i, _), _| coord.is_member(*i));
        self.outboxes.retain(|i, _| coord.is_member(*i));
    }

    /// Step attempt; on a step, merge the planned updates into the base and
    /// broadcast it. Resets the soft-deadline clock on a step or timeout.
    fn advance(&mut self, timed_out: bool) -> Result<()> {
        let Some(plan) = self.coord.try_advance(timed_out) else {
            if timed_out {
                self.round_started = Instant::now();
            }
            return Ok(());
        };
        self.round_started = Instant::now();
        if let Some(params) = self.params.as_mut() {
            let mut grads = Vec::with_capacity(plan.weights.len());
            let mut weights = Vec::with_capacity(plan.weights.len());
            for &(island, base, w) in &plan.weights {
                let mut u = self.stored.remove(&(island, base)).with_context(|| {
                    format!("no DELTA_TENSOR for island {island} base {base} in a tensor run")
                })?;
                // Wire carries θ − base; merge.rs consumes outer gradients.
                u.iter_mut().for_each(|x| *x = -*x);
                grads.push(u);
                weights.push(w);
            }
            let mut merged = vec![0.0f32; params.len()];
            let refs: Vec<&[f32]> = grads.iter().map(Vec::as_slice).collect();
            crate::merge::merge_avg(&refs, &weights, &mut merged);
            crate::merge::nesterov_step(
                params,
                &mut self.momentum_buf,
                &merged,
                self.outer_lr,
                self.outer_momentum,
            );
        }
        // Every pending/carried entry was consumed or dropped by the step.
        self.stored.clear();
        self.publish_hash();
        if self.checkpoint_every > 0 && self.coord.outer_version % self.checkpoint_every == 0 {
            self.write_checkpoint()?;
        }
        if let Some(frame) = self.base_msg() {
            self.outboxes.retain(|_, tx| tx.send(frame.clone()).is_ok());
        }
        Ok(())
    }

    /// Handle one authenticated message; returns frames for the sender.
    fn handle(&mut self, msg: ElasticMsg, now: f64, outbox: &Outbox) -> Result<Vec<(u8, Vec<u8>)>> {
        self.coord.check_fence(msg.syncer_epoch())?;
        let mut replies = Vec::new();
        if self.finished() {
            // Final grace window: LEAVE is still applied; everything else
            // gets FINISHED so the island can exit 0.
            if let ElasticMsg::Leave { .. } = msg {
                self.coord.apply(&msg, now)?;
                self.purge_departed();
            } else {
                replies.push(self.finished_msg());
            }
            return Ok(replies);
        }
        match &msg {
            ElasticMsg::ElasticInit { island_id, params, .. } => {
                ensure!(self.coord.is_member(*island_id), "{island_id} is not a member");
                if self.params.is_none() {
                    ensure!(!params.is_empty(), "ELASTIC_INIT with empty parameters");
                    self.momentum_buf = vec![0.0; params.len()];
                    self.params = Some(params.clone());
                    self.publish_hash();
                }
                replies.extend(self.base_msg());
                return Ok(replies);
            }
            ElasticMsg::DeltaTensor { island_id, base_version, update, .. } => {
                let p = self.params.as_ref().context("DELTA_TENSOR before ELASTIC_INIT")?;
                ensure!(update.len() == p.len(), "DELTA_TENSOR length {} != {}", update.len(), p.len());
                ensure!(self.coord.is_member(*island_id), "{island_id} is not a member");
                self.stored.insert((*island_id, *base_version), update.clone());
            }
            ElasticMsg::DeltaReady { .. } if self.params.is_some() => {
                bail!("DELTA_READY carries no tensor; use DELTA_TENSOR once a base exists")
            }
            ElasticMsg::Join { island_id, backend_identity, compat_group, .. } => {
                if let Some(pinned) = &self.compat_group {
                    ensure!(
                        pinned == compat_group,
                        "compat_group mismatch, JOIN refused: 兼容组不同：{} 对 {}，容差未标定 \
                         (island {island_id} declares {}, this elastic session is pinned to {} by its \
                         first JOIN; tolerance not calibrated)",
                        group_label(pinned),
                        group_label(compat_group),
                        group_label(compat_group),
                        group_label(pinned)
                    );
                }
                if let Some(pinned) = &self.backend_identity {
                    ensure!(
                        pinned == backend_identity,
                        "backend identity mismatch, JOIN refused: island {island_id} declares {} \
                         but this elastic session is pinned to {} (head config or first JOIN); islands with a \
                         different training backend (e.g. Miles vs verl), engine commit, device \
                         family or parameter-name map cannot be merged",
                        hex32(backend_identity),
                        hex32(pinned)
                    );
                }
            }
            _ => {}
        }
        let result = self.coord.apply(&msg, now);
        if let ElasticMsg::DeltaTensor { island_id, base_version, .. } = &msg {
            if result.is_err() {
                self.stored.remove(&(*island_id, *base_version));
            }
        }
        let reply = result?;
        match &msg {
            ElasticMsg::Join { island_id, backend_identity, compat_group, .. } => {
                self.backend_identity.get_or_insert(*backend_identity);
                self.compat_group.get_or_insert_with(|| compat_group.clone());
                self.outboxes.insert(*island_id, outbox.clone());
                replies.extend(reply.map(|m| m.encode(&self.key)));
                replies.extend(self.base_msg());
            }
            ElasticMsg::Leave { .. } => self.purge_departed(),
            ElasticMsg::SampleIndex { entry, .. } => {
                if let Some(crate::elastic::TapeEvent::SampleIndexed { verdict, reason, outer_lag, duplicate: false, .. }) =
                    self.coord.tape.last()
                {
                    if *verdict != crate::elastic::VERDICT_REJECT {
                        self.append_sample_index(entry, *verdict, reason, *outer_lag)?;
                    }
                }
                replies.extend(reply.map(|m| m.encode(&self.key)));
            }
            ElasticMsg::DeltaReady { .. } | ElasticMsg::DeltaTensor { .. } => self.advance(false)?,
            _ => replies.extend(reply.map(|m| m.encode(&self.key))),
        }
        Ok(replies)
    }
}

fn island_of(cell: &AtomicU64) -> Option<u32> {
    u32::try_from(cell.load(Ordering::Relaxed)).ok()
}

fn msg_island(m: &ElasticMsg) -> Option<u32> {
    match m {
        ElasticMsg::Join { island_id, .. }
        | ElasticMsg::Leave { island_id, .. }
        | ElasticMsg::LeaseHeartbeat { island_id, .. }
        | ElasticMsg::DeltaReady { island_id, .. }
        | ElasticMsg::ElasticInit { island_id, .. }
        | ElasticMsg::DeltaTensor { island_id, .. } => Some(*island_id),
        _ => None,
    }
}

async fn send(wr: &mut tokio::net::tcp::OwnedWriteHalf, t: u8, p: &[u8]) -> Result<()> {
    let mut header = [0u8; 13];
    header[0..4].copy_from_slice(&MAGIC.to_le_bytes());
    header[4] = t;
    header[5..13].copy_from_slice(&(p.len() as u64).to_le_bytes());
    wr.write_all(&header).await?;
    wr.write_all(p).await?;
    Ok(())
}

async fn serve_conn(
    stream: TcpStream,
    shared: Arc<Mutex<Shared>>,
    key: Arc<Vec<u8>>,
    epoch0: Instant,
    fatal: mpsc::UnboundedSender<anyhow::Error>,
) -> Result<()> {
    let (mut rd, mut wr) = stream.into_split();
    let (tx, mut rx) = mpsc::unbounded_channel::<(u8, Vec<u8>)>();
    // Island on this connection, learned from the first decoded message.
    let island = Arc::new(AtomicU64::new(u64::MAX));
    let (w_shared, w_island) = (shared.clone(), island.clone());
    let writer = tokio::spawn(async move {
        while let Some((t, p)) = rx.recv().await {
            let started = Instant::now();
            if send(&mut wr, t, &p).await.is_err() {
                break;
            }
            if t == MSG_ELASTIC_BASE {
                let secs = started.elapsed().as_secs_f64();
                let isl = island_of(&w_island);
                let _ = w_shared.lock().unwrap().record_transfer("send", t, isl, p.len() as u64 + 13, secs);
            }
        }
    });
    let result = async {
        loop {
            let mut header_at: Option<Instant> = None;
            let frame = match read_frame_limited(&mut rd, |t| {
                header_at = Some(Instant::now());
                match t {
                    MSG_JOIN | MSG_LEAVE | MSG_LEASE_HEARTBEAT | MSG_SAMPLE_INDEX | MSG_DELTA_READY
                    | MSG_ELASTIC_INIT | MSG_DELTA_TENSOR => Ok(MAX_ELASTIC_FRAME),
                    other => bail!("message type {other} is not accepted by the elastic server"),
                }
            })
            .await
            {
                Ok(f) => f,
                Err(e) => {
                    if e.downcast_ref::<std::io::Error>()
                        .is_some_and(|io| io.kind() == std::io::ErrorKind::UnexpectedEof)
                    {
                        return Ok(());
                    }
                    let _ = tx.send((MSG_ERROR, format!("{e:#}").into_bytes()));
                    return Err(e);
                }
            };
            let now = epoch0.elapsed().as_secs_f64();
            let recv_secs = header_at.map(|t| t.elapsed().as_secs_f64());
            let outcome = ElasticMsg::decode(&key, frame.msg_type, &frame.payload).and_then(|msg| {
                if let Some(i) = msg_island(&msg) {
                    island.store(u64::from(i), Ordering::Relaxed);
                }
                let mut g = shared.lock().unwrap();
                if matches!(frame.msg_type, MSG_ELASTIC_INIT | MSG_DELTA_TENSOR) {
                    let bytes = frame.payload.len() as u64 + 13;
                    g.record_transfer("recv", frame.msg_type, island_of(&island), bytes, recv_secs.unwrap_or(f64::NAN))?;
                }
                let r = g.handle(msg, now, &tx);
                g.flush_tape()?;
                r
            });
            match outcome {
                Ok(frames) => {
                    for f in frames {
                        let _ = tx.send(f);
                    }
                }
                Err(e) if format!("{e:#}").contains("no DELTA_TENSOR") => {
                    // Merge invariant broken: stop the whole server.
                    let _ = fatal.send(anyhow::anyhow!("{e:#}"));
                    return Err(e);
                }
                Err(e) => {
                    // Fenced, unauthenticated or rejected: report and keep
                    // the connection (the island decides whether to rejoin).
                    warn!(error = %format!("{e:#}"), "elastic message refused");
                    let _ = tx.send((MSG_ERROR, format!("{e:#}").into_bytes()));
                }
            }
        }
    }
    .await;
    drop(tx);
    let _ = writer.await;
    result
}

/// Serve until `total_steps` outer steps have been published. Returns the
/// final outer version and parameters (None for a control-plane-only run).
pub async fn run(listener: TcpListener, cfg: ElasticServerConfig) -> Result<(u64, Option<Vec<f32>>)> {
    let tape = match &cfg.event_tape {
        Some(p) => Some(
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(p)
                .with_context(|| format!("open event tape {}", p.display()))?,
        ),
        None => None,
    };
    let shared = Arc::new(Mutex::new(Shared {
        coord: ElasticCoordinator::new(cfg.params, cfg.lease_s, cfg.syncer_epoch)?,
        backend_identity: cfg.expected_backend_identity,
        compat_group: None,
        flushed: 0,
        round_started: Instant::now(),
        tape,
        key: cfg.key.clone(),
        outer_lr: cfg.outer_lr,
        outer_momentum: cfg.outer_momentum,
        params: None,
        momentum_buf: Vec::new(),
        stored: HashMap::new(),
        outboxes: HashMap::new(),
        checkpoint_path: cfg.checkpoint_path.clone(),
        checkpoint_every: cfg.checkpoint_every,
        total_steps: cfg.total_steps,
        finished_at: None,
        status_path: cfg.event_tape.as_ref().map(|t| t.with_file_name("status.json")),
        sample_index_path: cfg.event_tape.as_ref().map(|t| t.with_file_name("sample_index.jsonl")),
        last_status_write: None,
    }));
    if cfg.resume {
        let path = cfg.checkpoint_path.as_ref().context("--resume requires --checkpoint-path")?;
        let bytes = std::fs::read(path)
            .with_context(|| format!("read elastic checkpoint {}", path.display()))?;
        let mut g = shared.lock().unwrap();
        g.restore(&bytes, cfg.lease_s, 0.0)?;
        if let Some(pinned) = cfg.expected_backend_identity {
            match g.backend_identity {
                Some(saved) if saved != pinned => bail!(
                    "elastic checkpoint pins backend identity {} but the head config pins {}",
                    hex32(&saved),
                    hex32(&pinned)
                ),
                _ => g.backend_identity = Some(pinned),
            }
        }
        g.publish_hash();
        // P9: every restart is a new coordinator incarnation.
        let epoch = (g.coord.syncer_epoch + 1).max(cfg.syncer_epoch);
        g.coord.syncer_epoch = epoch;
        g.write_checkpoint()?;
        info!(syncer_epoch = epoch, outer_version = g.coord.outer_version, "elastic syncer resumed");
    }
    let key = Arc::new(cfg.key);
    let epoch0 = Instant::now();
    let soft = Duration::from_secs(cfg.params.soft_deadline_s);
    let (fatal_tx, mut fatal_rx) = mpsc::unbounded_channel();
    info!(syncer_epoch = cfg.syncer_epoch, ?soft, lease_s = cfg.lease_s, "elastic syncer listening");
    let mut ticker = tokio::time::interval(cfg.tick);
    loop {
        tokio::select! {
            accepted = listener.accept() => {
                let (stream, peer) = accepted?;
                stream.set_nodelay(true).ok();
                let (s, k, f) = (shared.clone(), key.clone(), fatal_tx.clone());
                tokio::spawn(async move {
                    if let Err(e) = serve_conn(stream, s, k, epoch0, f).await {
                        warn!(%peer, error = %format!("{e:#}"), "elastic connection closed");
                    }
                });
            }
            Some(e) = fatal_rx.recv() => return Err(e),
            _ = ticker.tick() => {
                let mut g = shared.lock().unwrap();
                if !g.coord.expire_leases(epoch0.elapsed().as_secs_f64()).is_empty() {
                    g.purge_departed();
                }
                let timed_out = g.round_started.elapsed() >= soft;
                if !g.finished() {
                    g.advance(timed_out)?;
                }
                g.flush_tape()?;
                if g.last_status_write.is_none_or(|t| t.elapsed() >= STATUS_EVERY) {
                    let remaining = soft.saturating_sub(g.round_started.elapsed()).as_secs_f64();
                    g.write_status(epoch0.elapsed().as_secs_f64(), remaining)?;
                }
                if g.finished() {
                    if g.finished_at.is_none() {
                        let frame = g.finished_msg();
                        g.outboxes.retain(|_, tx| tx.send(frame.clone()).is_ok());
                        g.finished_at = Some(Instant::now());
                        g.write_status(epoch0.elapsed().as_secs_f64(), 0.0)?;
                        info!(outer_version = g.coord.outer_version, grace = ?cfg.final_grace, "elastic run finished; final grace window");
                    }
                    let grace_over = g.finished_at.is_some_and(|t| t.elapsed() >= cfg.final_grace);
                    if grace_over || g.coord.member_count() == 0 {
                        g.write_status(epoch0.elapsed().as_secs_f64(), 0.0)?;
                        return Ok((g.coord.outer_version, g.params.clone()));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};

    const KEY: &[u8] = b"test-key";
    const MILES_ID: [u8; 32] = [0x11; 32];

    struct Island {
        id: u32,
        epoch: u64,
        rd: OwnedReadHalf,
        wr: OwnedWriteHalf,
        identity: [u8; 32],
        compat: String,
    }

    impl Island {
        async fn connect(port: u16, id: u32, epoch: u64) -> Self {
            let (rd, wr) = TcpStream::connect(("127.0.0.1", port)).await.unwrap().into_split();
            Self { id, epoch, rd, wr, identity: MILES_ID, compat: "nvidia-h100".to_string() }
        }
        async fn send(&mut self, m: ElasticMsg) {
            let (t, p) = m.encode(KEY);
            send(&mut self.wr, t, &p).await.unwrap();
        }
        async fn recv(&mut self) -> (u8, Vec<u8>) {
            let f = read_frame_limited(&mut self.rd, |_| Ok(MAX_ELASTIC_FRAME)).await.unwrap();
            (f.msg_type, f.payload)
        }
        async fn join(&mut self) -> ElasticMsg {
            self.send(ElasticMsg::Join {
                syncer_epoch: self.epoch,
                island_id: self.id,
                incarnation: 0,
                capacity: 1.0,
                backend_identity: self.identity,
                compat_group: self.compat.clone(),
            })
            .await;
            let (t, p) = self.recv().await;
            ElasticMsg::decode(KEY, t, &p).unwrap()
        }
        async fn delta(&mut self, base_version: u64) {
            self.send(ElasticMsg::DeltaReady {
                syncer_epoch: self.epoch,
                island_id: self.id,
                base_version,
                c_tokens: 10,
                c_steps: 1,
            })
            .await;
        }
        async fn init(&mut self, params: Vec<f32>) {
            self.send(ElasticMsg::ElasticInit { syncer_epoch: self.epoch, island_id: self.id, params }).await;
        }
        async fn update(&mut self, base_version: u64, value: f32) {
            self.send(ElasticMsg::DeltaTensor {
                syncer_epoch: self.epoch,
                island_id: self.id,
                base_version,
                c_tokens: 10,
                c_steps: 1,
                update: vec![value; 4],
            })
            .await;
        }
        /// Next ELASTIC_BASE frame (skips other frames; panics on MSG_ERROR).
        async fn next_base(&mut self) -> (u64, Vec<f32>) {
            loop {
                let (t, p) = tokio::time::timeout(Duration::from_secs(5), self.recv()).await.unwrap();
                assert_ne!(t, MSG_ERROR, "{}", String::from_utf8_lossy(&p));
                if t == MSG_ELASTIC_BASE {
                    match ElasticMsg::decode(KEY, t, &p).unwrap() {
                        ElasticMsg::ElasticBase { outer_version, params, .. } => {
                            return (outer_version, params)
                        }
                        _ => unreachable!(),
                    }
                }
            }
        }
        async fn heartbeat(&mut self) {
            self.send(ElasticMsg::LeaseHeartbeat {
                syncer_epoch: self.epoch,
                island_id: self.id,
                membership_epoch: 0,
                inner_step: 0,
                round_wall_s: 0.0,
            })
            .await;
        }
    }

    fn tape_lines(path: &std::path::Path) -> Vec<String> {
        std::fs::read_to_string(path).unwrap_or_default().lines().map(str::to_string).collect()
    }

    async fn wait_for(path: &std::path::Path, needle: &str, count: usize) {
        for _ in 0..200 {
            if tape_lines(path).iter().filter(|l| l.contains(needle)).count() >= count {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("timed out waiting for {count}x {needle}: {:#?}", tape_lines(path));
    }

    /// 3 islands; one is late (carried over), one stops heartbeating and is
    /// expired with its uncommitted delta dropped, one joins mid-run with
    /// zero weight in its first round; a stale-epoch message is fenced.
    #[tokio::test]
    async fn three_islands_late_expiry_and_midrun_join() {
        let dir = std::env::temp_dir().join(format!("elastic-it-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let tape = dir.join("tape.jsonl");
        let _ = std::fs::remove_file(&tape);
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = tokio::spawn(run(
            listener,
            ElasticServerConfig {
                params: ElasticParams {
                    quorum_theta: 0.6,
                    carry_gamma: 0.5,
                    soft_deadline_s: 60,
                    q_min: 1,
                    max_carry_lag: 2,
                },
                key: KEY.to_vec(),
                syncer_epoch: 7,
                lease_s: 0.6,
                total_steps: 3,
                event_tape: Some(tape.clone()),
                tick: Duration::from_millis(20),
                outer_lr: 1.0,
                outer_momentum: 0.0,
                final_grace: Duration::ZERO,
                checkpoint_path: None,
                checkpoint_every: 0,
                resume: false,
                expected_backend_identity: None,
            },
        ));
        let (mut a, mut b, mut c) = (
            Island::connect(port, 1, 7).await,
            Island::connect(port, 2, 7).await,
            Island::connect(port, 3, 7).await,
        );
        for i in [&mut a, &mut b, &mut c] {
            assert!(matches!(i.join().await, ElasticMsg::JoinAck { catch_up: false, .. }));
        }

        // Fencing: an older syncer_epoch is refused with MSG_ERROR.
        let mut stale = Island::connect(port, 9, 6).await;
        stale.send(ElasticMsg::Join { syncer_epoch: 6, island_id: 9, incarnation: 0, capacity: 1.0, backend_identity: MILES_ID, compat_group: String::new() }).await;
        let (t, p) = stale.recv().await;
        assert_eq!(t, MSG_ERROR);
        assert!(String::from_utf8_lossy(&p).contains("fenced"));

        // Round 0: A and B (2/3 >= 0.6) step without C.
        a.delta(0).await;
        b.delta(0).await;
        wait_for(&tape, "\"kind\":\"outer_step\"", 1).await;
        // C is late on base 0 (lag 1) -> carried over; round 1 A+B+carried C.
        c.delta(0).await;
        wait_for(&tape, "delta_carried_over", 1).await;
        a.delta(1).await;
        b.delta(1).await;
        wait_for(&tape, "\"kind\":\"outer_step\"", 2).await;
        // C submits on base 2 then goes silent; A, B keep heartbeating.
        c.delta(2).await;
        wait_for(&tape, "\"kind\":\"delta_accepted\",\"syncer_epoch\":7,\"island_id\":3", 1).await;
        for _ in 0..10 {
            a.heartbeat().await;
            b.heartbeat().await;
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        wait_for(&tape, "lease_expired", 1).await;
        // D joins mid-run (catch-up), then A, B, D finish round 2.
        let mut d = Island::connect(port, 4, 7).await;
        assert!(matches!(d.join().await, ElasticMsg::JoinAck { catch_up: true, base_version: 2, .. }));
        a.delta(2).await;
        b.delta(2).await;
        d.delta(2).await;
        let (final_version, params) = tokio::time::timeout(Duration::from_secs(10), server).await.unwrap().unwrap().unwrap();
        assert!(params.is_none(), "control-plane-only run has no tensors");
        assert!(final_version >= 3);

        let status = std::fs::read_to_string(dir.join("status.json")).unwrap();
        assert!(status.contains("\"schema\":\"yeto.syncer.elastic-status/v1\""), "{status}");
        assert!(status.contains("\"syncer_epoch\":7"), "{status}");
        let lines = tape_lines(&tape);
        let steps: Vec<&String> = lines.iter().filter(|l| l.contains("\"kind\":\"outer_step\"")).collect();
        assert!(steps[0].contains("\"absent\":[3]"), "{}", steps[0]);
        // Round 1: A,B at 100 each, C carried at 100*0.5 on base 0.
        assert!(steps[1].contains("[3,0,50.0]"), "{}", steps[1]);
        assert!(steps[1].contains("\"weights\":[[1,1,0.4],[2,1,0.4],[3,0,0.2]]"), "{}", steps[1]);
        let expired = lines.iter().find(|l| l.contains("lease_expired")).unwrap();
        assert!(expired.contains("\"island_id\":3"), "{expired}");
        assert!(expired.contains("\"dropped_uncommitted\":{\"island_id\":3,\"outer_version\":2"), "{expired}");
        assert!(lines.iter().any(|l| l.contains("pool_join") && l.contains("\"island_id\":4") && l.contains("\"catch_up\":true")));
        // Round 2 may step on A+B (2/3) before D's delta arrives; if D is in
        // the round its weight must be 0.
        assert!(!steps[2].contains("[4,2,100.0]"), "{}", steps[2]);
        assert!(!steps[2].contains("[3,"), "expired island must not be merged: {}", steps[2]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Bandwidth records: every ELASTIC_INIT / DELTA_TENSOR received and
    /// ELASTIC_BASE sent lands on the event tape as kind "transfer" with the
    /// island, the exact frame bytes and a finite duration.
    #[tokio::test]
    async fn tensor_frames_record_transfer_bytes_and_seconds() {
        let dir = std::env::temp_dir().join(format!("elastic-xfer-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let tape = dir.join("tape.jsonl");
        let _ = std::fs::remove_file(&tape);
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = tokio::spawn(run(
            listener,
            ElasticServerConfig {
                params: ElasticParams { quorum_theta: 0.6, carry_gamma: 0.5, soft_deadline_s: 60, q_min: 1, max_carry_lag: 2 },
                key: KEY.to_vec(),
                syncer_epoch: 3,
                lease_s: 30.0,
                total_steps: 1,
                event_tape: Some(tape.clone()),
                tick: Duration::from_millis(20),
                outer_lr: 1.0,
                outer_momentum: 0.0,
                final_grace: Duration::ZERO,
                checkpoint_path: None,
                checkpoint_every: 0,
                resume: false,
                expected_backend_identity: None,
            },
        ));
        let mut a = Island::connect(port, 5, 3).await;
        a.join().await;
        a.init(vec![0.0; 4]).await;
        assert_eq!(a.next_base().await.0, 0);
        a.update(0, 1.0).await;
        assert_eq!(a.next_base().await.0, 1);
        wait_for(&tape, "\"msg\":\"elastic_base\"", 2).await;
        drop(a);
        let _ = tokio::time::timeout(Duration::from_secs(5), server).await;

        let frame_len = |m: ElasticMsg| m.encode(KEY).1.len() as u64 + 13;
        let init_bytes = frame_len(ElasticMsg::ElasticInit { syncer_epoch: 3, island_id: 5, params: vec![0.0; 4] });
        let delta_bytes = frame_len(ElasticMsg::DeltaTensor {
            syncer_epoch: 3, island_id: 5, base_version: 0, c_tokens: 10, c_steps: 1, update: vec![1.0; 4],
        });
        let lines: Vec<String> = tape_lines(&tape).into_iter().filter(|l| l.contains("\"kind\":\"transfer\"")).collect();
        let find = |dir: &str, msg: &str| -> Vec<String> {
            lines.iter().filter(|l| l.contains(&format!("\"direction\":\"{dir}\",\"msg\":\"{msg}\""))).cloned().collect()
        };
        let init = find("recv", "elastic_init");
        assert_eq!(init.len(), 1, "{lines:#?}");
        assert!(init[0].contains(&format!("\"bytes\":{init_bytes},")), "{}", init[0]);
        let delta = find("recv", "delta_tensor");
        assert_eq!(delta.len(), 1, "{lines:#?}");
        assert!(delta[0].contains(&format!("\"bytes\":{delta_bytes},")), "{}", delta[0]);
        let bases = find("send", "elastic_base");
        assert!(bases.len() >= 2, "{lines:#?}");
        for l in init.iter().chain(&delta).chain(&bases) {
            assert!(l.contains("\"island_id\":5,"), "{l}");
            assert!(l.contains("\"syncer_epoch\":3,"), "{l}");
            assert!(!l.contains("\"seconds\":null"), "{l}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn assert_close(got: &[f32], want: f32) {
        for g in got {
            assert!((g - want).abs() < 1e-5, "got {got:?}, want {want}");
        }
    }

    /// 3 islands, 3 tensor rounds, outer_lr 1 / momentum 0 so the new base
    /// is base + weighted mean of updates. Round 0: A=1, B=2 (C absent)
    /// -> 1.5. Round 1: C late on base 0 (4, weight 100*0.5), A=1, B=3 ->
    /// 1.5 + (100+300+200)/250 = 3.9. Round 2: C sends 100 on base 2 then
    /// expires (never merged); D joins and sends 1000 (weight 0, first
    /// round), then A=1 reaches capacity 2/3 >= 0.6 -> 3.9 + 1 = 4.9.
    #[tokio::test]
    async fn tensor_rounds_match_hand_computed_weights() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = tokio::spawn(run(
            listener,
            ElasticServerConfig {
                params: ElasticParams {
                    quorum_theta: 0.6,
                    carry_gamma: 0.5,
                    soft_deadline_s: 60,
                    q_min: 1,
                    max_carry_lag: 2,
                },
                key: KEY.to_vec(),
                syncer_epoch: 1,
                lease_s: 0.6,
                total_steps: 3,
                event_tape: None,
                tick: Duration::from_millis(20),
                outer_lr: 1.0,
                outer_momentum: 0.0,
                final_grace: Duration::ZERO,
                checkpoint_path: None,
                checkpoint_every: 0,
                resume: false,
                expected_backend_identity: None,
            },
        ));
        let (mut a, mut b, mut c) = (
            Island::connect(port, 1, 1).await,
            Island::connect(port, 2, 1).await,
            Island::connect(port, 3, 1).await,
        );
        a.join().await;
        a.init(vec![0.0; 4]).await;
        assert_eq!(a.next_base().await, (0, vec![0.0; 4]));
        b.join().await;
        c.join().await;
        assert_eq!(b.next_base().await.0, 0);
        assert_eq!(c.next_base().await.0, 0);

        a.update(0, 1.0).await;
        b.update(0, 2.0).await;
        let (v, p) = a.next_base().await;
        assert_eq!(v, 1);
        assert_close(&p, 1.5);
        assert_eq!(b.next_base().await.0, 1);
        assert_eq!(c.next_base().await.0, 1, "absent member still receives the base");

        c.update(0, 4.0).await; // late: lag 1, discounted by 0.5
        tokio::time::sleep(Duration::from_millis(50)).await;
        a.update(1, 1.0).await;
        b.update(1, 3.0).await;
        let (v, p) = a.next_base().await;
        assert_eq!(v, 2);
        assert_close(&p, 3.9);
        b.next_base().await;
        c.next_base().await;

        c.update(2, 100.0).await; // then C goes silent and expires
        for _ in 0..10 {
            a.heartbeat().await;
            b.heartbeat().await;
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        let mut d = Island::connect(port, 4, 1).await;
        assert!(matches!(d.join().await, ElasticMsg::JoinAck { catch_up: true, base_version: 2, .. }));
        assert_eq!(d.next_base().await.0, 2);
        d.update(2, 1000.0).await; // joiner's first round: weight 0
        tokio::time::sleep(Duration::from_millis(50)).await;
        a.update(2, 1.0).await;
        b.update(2, 1.0).await;
        let (v, p) = a.next_base().await;
        assert_eq!(v, 3);
        // D at weight 0, C's 100 dropped with its lease: mean of A (and B) = 1.
        assert_close(&p, 4.9);
        let (final_version, params) =
            tokio::time::timeout(Duration::from_secs(10), server).await.unwrap().unwrap().unwrap();
        assert_eq!(final_version, 3);
        assert_close(&params.unwrap(), 4.9);
    }

    /// D-S6: one tensor step with a checkpoint, then restart with --resume:
    /// syncer_epoch increments, members/base/outer_version are restored, the
    /// old epoch is fenced, and a further step continues from the base.
    #[tokio::test]
    async fn resume_restores_state_and_bumps_syncer_epoch() {
        let dir = std::env::temp_dir().join(format!("elastic-ckpt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ckpt = dir.join("elastic.ckpt");
        let _ = std::fs::remove_file(&ckpt);
        let mk = |total_steps, resume| ElasticServerConfig {
            params: ElasticParams { quorum_theta: 0.6, carry_gamma: 0.5, soft_deadline_s: 60, q_min: 1, max_carry_lag: 2 },
            key: KEY.to_vec(),
            syncer_epoch: 0,
            lease_s: 30.0,
            total_steps,
            event_tape: None,
            tick: Duration::from_millis(20),
            outer_lr: 1.0,
            outer_momentum: 0.0,
            final_grace: Duration::ZERO,
            checkpoint_path: Some(ckpt.clone()),
            checkpoint_every: 1,
            resume,
            expected_backend_identity: None,
        };
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = tokio::spawn(run(listener, mk(1, false)));
        let (mut a, mut b) = (Island::connect(port, 1, 0).await, Island::connect(port, 2, 0).await);
        a.join().await;
        a.init(vec![0.0; 4]).await;
        a.next_base().await;
        b.join().await;
        b.next_base().await;
        a.update(0, 2.0).await;
        b.update(0, 4.0).await;
        let (_, p) = tokio::time::timeout(Duration::from_secs(10), server).await.unwrap().unwrap().unwrap();
        assert_close(&p.unwrap(), 3.0);

        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = tokio::spawn(run(listener, mk(2, true)));
        let mut stale = Island::connect(port, 1, 0).await;
        stale.update(1, 1.0).await;
        let (t, msg) = stale.recv().await;
        assert_eq!(t, MSG_ERROR);
        assert!(String::from_utf8_lossy(&msg).contains("fenced"));
        // Restored members resend on the restored base with the new epoch.
        let (mut a, mut b) = (Island::connect(port, 1, 1).await, Island::connect(port, 2, 1).await);
        a.update(1, 1.0).await;
        b.update(1, 3.0).await;
        let (v, p) = tokio::time::timeout(Duration::from_secs(10), server).await.unwrap().unwrap().unwrap();
        assert_eq!(v, 2);
        assert_close(&p.unwrap(), 5.0);
        let saved = std::fs::read(&ckpt).unwrap();
        let mut sh = ElasticCoordinator::new(mk(0, false).params, 30.0, 0).unwrap();
        // magic 8 | identity flag 1 + 32 | coordinator length u64 | coordinator
        assert_eq!((saved[8], &saved[9..41]), (1, &MILES_ID[..]));
        let n = u64::from_le_bytes(saved[41..49].try_into().unwrap()) as usize;
        sh = ElasticCoordinator::decode_state(sh.params, 30.0, &saved[49..49 + n], 0.0).unwrap();
        assert_eq!((sh.syncer_epoch, sh.outer_version), (1, 2));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// P3 over the wire: a tensor run publishes sha256(base) per version;
    /// SAMPLE_INDEX gets ACCEPT / ACCEPT_IS / REJECT replies, accepted
    /// entries land in sample_index.jsonl, and nothing is forwarded.
    #[tokio::test]
    async fn sample_index_over_the_wire() {
        use crate::elastic::{SampleIndexEntry, SAMPLE_INDEX_SCHEMA, VERDICT_ACCEPT, VERDICT_ACCEPT_IS, VERDICT_REJECT};
        let dir = std::env::temp_dir().join(format!("elastic-idx-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let tape = dir.join("tape.jsonl");
        for f in ["tape.jsonl", "sample_index.jsonl", "status.json"] {
            let _ = std::fs::remove_file(dir.join(f));
        }
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = tokio::spawn(run(
            listener,
            ElasticServerConfig {
                params: ElasticParams { quorum_theta: 0.6, carry_gamma: 0.5, soft_deadline_s: 60, q_min: 1, max_carry_lag: 2 },
                key: KEY.to_vec(),
                syncer_epoch: 0,
                lease_s: 30.0,
                total_steps: 2,
                event_tape: Some(tape.clone()),
                tick: Duration::from_millis(20),
                outer_lr: 1.0,
                outer_momentum: 0.0,
                final_grace: Duration::ZERO,
                checkpoint_path: None,
                checkpoint_every: 0,
                resume: false,
                expected_backend_identity: None,
            },
        ));
        let mut a = Island::connect(port, 1, 0).await;
        a.join().await;
        a.init(vec![0.0; 4]).await;
        let (_, base0) = a.next_base().await;
        a.update(0, 1.0).await;
        let (v, base1) = a.next_base().await;
        assert_eq!(v, 1);
        let entry = |ver: u64, base: &[f32], group: &str| SampleIndexEntry {
            island_id: 1,
            outer_version: ver,
            inner_step: 0,
            policy_hash: policy_hash(base),
            group_id: group.into(),
            prompt_id: "p".into(),
            n: 8,
            uri: "s3://b/g".into(),
            size_bytes: 10,
            sha256: "0".repeat(64),
            has_behavior_logprob: true,
            advantage_included: true,
            created_at: 0.0,
            schema: SAMPLE_INDEX_SCHEMA.into(),
        };
        let mut verdicts = Vec::new();
        for e in [entry(1, &base1, "g1"), entry(0, &base0, "g0"), entry(0, &base1, "bad")] {
            a.send(ElasticMsg::SampleIndex { syncer_epoch: 0, entry: e }).await;
            let (t, p) = a.recv().await;
            match ElasticMsg::decode(KEY, t, &p).unwrap() {
                ElasticMsg::SampleVerdict { verdict, reason, .. } => verdicts.push((verdict, reason)),
                m => panic!("{m:?}"),
            }
        }
        assert_eq!(
            verdicts,
            vec![
                (VERDICT_ACCEPT, "same_version".to_string()),
                (VERDICT_ACCEPT_IS, "stale_within_bound".to_string()),
                (VERDICT_REJECT, "policy_hash_mismatch".to_string()),
            ]
        );
        a.update(1, 1.0).await;
        tokio::time::timeout(Duration::from_secs(10), server).await.unwrap().unwrap().unwrap();
        let idx = std::fs::read_to_string(dir.join("sample_index.jsonl")).unwrap();
        assert_eq!(idx.lines().count(), 2, "{idx}");
        assert!(idx.contains("\"group_id\":\"g1\"") && idx.contains("\"verdict\":\"ACCEPT_IS\""), "{idx}");
        assert_eq!(tape_lines(&tape).iter().filter(|l| l.contains("sample_indexed")).count(), 3);
        let status = std::fs::read_to_string(dir.join("status.json")).unwrap();
        assert!(status.contains("\"sample_index_count\":2"), "{status}");
        let _ = std::fs::remove_dir_all(&dir);
    }
    /// Final grace: after total_steps the server broadcasts FINISHED, answers
    /// late DELTA_TENSOR / heartbeat / JOIN with FINISHED, and exits once
    /// every member has left (before the grace window ends).
    #[tokio::test]
    async fn finished_grace_window_answers_late_islands_and_exits_on_leave() {
        let dir = std::env::temp_dir().join(format!("elastic-fin-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let tape = dir.join("tape.jsonl");
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let started = Instant::now();
        let server = tokio::spawn(run(
            listener,
            ElasticServerConfig {
                params: ElasticParams { quorum_theta: 0.5, carry_gamma: 0.5, soft_deadline_s: 60, q_min: 1, max_carry_lag: 2 },
                key: KEY.to_vec(),
                syncer_epoch: 0,
                lease_s: 30.0,
                total_steps: 1,
                event_tape: Some(tape),
                tick: Duration::from_millis(20),
                outer_lr: 1.0,
                outer_momentum: 0.0,
                final_grace: Duration::from_secs(30),
                checkpoint_path: None,
                checkpoint_every: 0,
                resume: false,
                expected_backend_identity: None,
            },
        ));
        let (mut a, mut b) = (Island::connect(port, 1, 0).await, Island::connect(port, 2, 0).await);
        a.join().await;
        a.init(vec![0.0; 4]).await;
        a.next_base().await;
        b.join().await;
        b.next_base().await;
        a.update(0, 1.0).await; // 1/2 >= 0.5 -> step to v1 = total_steps
        let expect_finished = |t: u8, p: &[u8]| match ElasticMsg::decode(KEY, t, p).unwrap() {
            ElasticMsg::Finished { outer_version, policy_hash, .. } => {
                assert_eq!(outer_version, 1);
                assert_ne!(policy_hash, [0; 32]);
            }
            m => panic!("{m:?}"),
        };
        // Broadcast reaches the slow island B unprompted (after BASE v1).
        loop {
            let (t, p) = b.recv().await;
            if t == MSG_FINISHED {
                expect_finished(t, &p);
                break;
            }
            assert_eq!(t, MSG_ELASTIC_BASE);
        }
        // B's late delta, a heartbeat and a fresh JOIN all get FINISHED.
        b.update(0, 5.0).await;
        let (t, p) = b.recv().await;
        expect_finished(t, &p);
        b.heartbeat().await;
        let (t, p) = b.recv().await;
        expect_finished(t, &p);
        let mut c = Island::connect(port, 3, 0).await;
        c.send(ElasticMsg::Join { syncer_epoch: 0, island_id: 3, incarnation: 0, capacity: 1.0, backend_identity: MILES_ID, compat_group: String::new() }).await;
        let (t, p) = c.recv().await;
        expect_finished(t, &p);
        let status = std::fs::read_to_string(dir.join("status.json")).unwrap();
        assert!(status.contains("\"state\":\"finished\""), "{status}");
        for i in [&mut a, &mut b] {
            let id = i.id;
            i.send(ElasticMsg::Leave { syncer_epoch: 0, island_id: id, reason: crate::elastic::LEAVE_REASON_REQUESTED }).await;
        }
        let (v, p) = tokio::time::timeout(Duration::from_secs(10), server).await.unwrap().unwrap().unwrap();
        assert_eq!(v, 1);
        assert_eq!(p.unwrap(), vec![1.0; 4], "late delta must not be merged");
        assert!(started.elapsed() < Duration::from_secs(20), "exited on all-left, not grace");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// yeto-framework-decoupling 6.2a: the first JOIN pins the backend
    /// identity; a JOIN with another identity (a "verl" island in a Miles
    /// session) gets MSG_ERROR naming both hashes and is not admitted, a
    /// second island with the same identity joins as before. The pin is in
    /// the checkpoint (survives --resume) and a pre-identity YELSRV1
    /// checkpoint is refused.
    #[tokio::test]
    async fn backend_identity_pinned_by_first_join_refuses_other_backend() {
        let dir = std::env::temp_dir().join(format!("elastic-ident-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ckpt = dir.join("elastic.ckpt");
        let _ = std::fs::remove_file(&ckpt);
        let verl: [u8; 32] = [0x22; 32];
        let mk = |total_steps, resume| ElasticServerConfig {
            params: ElasticParams { quorum_theta: 0.5, carry_gamma: 0.5, soft_deadline_s: 60, q_min: 1, max_carry_lag: 2 },
            key: KEY.to_vec(),
            syncer_epoch: 0,
            lease_s: 30.0,
            total_steps,
            event_tape: None,
            tick: Duration::from_millis(20),
            outer_lr: 1.0,
            outer_momentum: 0.0,
            final_grace: Duration::ZERO,
            checkpoint_path: Some(ckpt.clone()),
            checkpoint_every: 1,
            resume,
            expected_backend_identity: None,
        };
        let expect_refused = |t: u8, p: &[u8]| {
            assert_eq!(t, MSG_ERROR);
            let text = String::from_utf8_lossy(p).to_string();
            assert!(text.contains("backend identity mismatch, JOIN refused"), "{text}");
            assert!(text.contains(&"22".repeat(32)) && text.contains(&"11".repeat(32)), "{text}");
            assert!(text.contains("Miles vs verl"), "{text}");
        };
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = tokio::spawn(run(listener, mk(3, false)));
        let mut a = Island::connect(port, 1, 0).await;
        assert!(matches!(a.join().await, ElasticMsg::JoinAck { .. }));
        a.init(vec![0.0; 4]).await;
        a.next_base().await;
        let mut v = Island::connect(port, 2, 0).await;
        v.identity = verl;
        v.send(ElasticMsg::Join { syncer_epoch: 0, island_id: 2, incarnation: 0, capacity: 1.0, backend_identity: verl, compat_group: "nvidia-h100".to_string() }).await;
        let (t, p) = v.recv().await;
        expect_refused(t, &p);
        let mut b = Island::connect(port, 3, 0).await;
        assert!(matches!(b.join().await, ElasticMsg::JoinAck { .. }), "same backend must join as before");
        b.next_base().await;
        a.update(0, 1.0).await; // 1/2 >= 0.5 -> v1, checkpoint written before the broadcast
        assert_eq!(a.next_base().await.0, 1);
        server.abort();
        let bytes = std::fs::read(&ckpt).unwrap();
        assert_eq!(&bytes[..8], SERVER_STATE_MAGIC);
        assert_eq!(bytes[8], 1);
        assert_eq!(&bytes[9..41], &MILES_ID);

        // Resume keeps the pin: the verl island is still refused.
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = tokio::spawn(run(listener, mk(3, true)));
        let mut v = Island::connect(port, 2, 1).await;
        v.send(ElasticMsg::Join { syncer_epoch: 1, island_id: 2, incarnation: 0, capacity: 1.0, backend_identity: verl, compat_group: "nvidia-h100".to_string() }).await;
        let (t, p) = v.recv().await;
        expect_refused(t, &p);
        server.abort();

        // A v1 (pre-identity) checkpoint is refused with a clear message.
        let mut old = SERVER_STATE_MAGIC_V1.to_vec();
        old.extend_from_slice(&bytes[41..]);
        std::fs::write(&ckpt, &old).unwrap();
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let err = run(listener, mk(3, true)).await.unwrap_err();
        assert!(format!("{err:#}").contains("predates the backend identity field"), "{err:#}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// yeto-framework-decoupling 7.7c (user decision 2026-10-09): the first
    /// JOIN pins compat_group; an H200 island in an H100 session and an Ascend
    /// island are refused with both groups named, the syncer keeps running and
    /// a second H100 island still joins. An H200 session admits H200.
    #[tokio::test]
    async fn compat_group_pinned_by_first_join_refuses_other_card_type() {
        let mk = || ElasticServerConfig {
            params: ElasticParams { quorum_theta: 0.5, carry_gamma: 0.5, soft_deadline_s: 60, q_min: 1, max_carry_lag: 2 },
            key: KEY.to_vec(),
            syncer_epoch: 0,
            lease_s: 30.0,
            total_steps: 3,
            event_tape: None,
            tick: Duration::from_millis(20),
            outer_lr: 1.0,
            outer_momentum: 0.0,
            final_grace: Duration::ZERO,
            checkpoint_path: None,
            checkpoint_every: 1,
            resume: false,
            expected_backend_identity: None,
        };
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = tokio::spawn(run(listener, mk()));
        let mut a = Island::connect(port, 1, 0).await;
        assert!(matches!(a.join().await, ElasticMsg::JoinAck { .. }));
        a.init(vec![0.0; 4]).await;
        a.next_base().await;
        for (id, group) in [(2u32, "nvidia-h200"), (4, "ascend-910b")] {
            let mut h = Island::connect(port, id, 0).await;
            h.compat = group.to_string();
            h.identity = [0x33; 32]; // the group is inside the identity hash too
            h.send(ElasticMsg::Join {
                syncer_epoch: 0, island_id: id, incarnation: 0, capacity: 1.0,
                backend_identity: h.identity, compat_group: h.compat.clone(),
            }).await;
            let (t, p) = h.recv().await;
            assert_eq!(t, MSG_ERROR);
            let text = String::from_utf8_lossy(&p).to_string();
            assert!(text.contains("compat_group mismatch, JOIN refused"), "{text}");
            assert!(text.contains(&format!("兼容组不同：nvidia-h100 对 {group}，容差未标定")), "{text}");
        }
        let mut b = Island::connect(port, 3, 0).await;
        assert!(matches!(b.join().await, ElasticMsg::JoinAck { .. }), "same card type must still join");
        server.abort();

        // An H200 session admits a second H200 island.
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = tokio::spawn(run(listener, mk()));
        let mut a = Island::connect(port, 1, 0).await;
        a.compat = "nvidia-h200".to_string();
        assert!(matches!(a.join().await, ElasticMsg::JoinAck { .. }));
        a.init(vec![0.0; 4]).await;
        a.next_base().await;
        let mut b = Island::connect(port, 2, 0).await;
        b.compat = "nvidia-h200".to_string();
        assert!(matches!(b.join().await, ElasticMsg::JoinAck { .. }));
        server.abort();
    }
}
