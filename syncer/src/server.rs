//! Async TCP server implementing the syncer side of docs/PROTOCOL.md:
//! per-learner connection groups (control stream + striped data streams),
//! chunk reassembly, and the pull-driven quorum/grace merge scheduler
//! at the core of the training loop. Rounds are pipelined: up to
//! `Config::pipeline` fragments are in flight at once (arXiv 2604.21428's
//! "two fragments in flight"), so a slow quorum on one fragment never
//! delays pulling the next.

use std::collections::{hash_map::Entry, BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use tokio::io::AsyncWriteExt;
use tokio::net::tcp::OwnedWriteHalf;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::action_probe::{self, ActionProbeClient, CommitPolicy, RetainedPreviews};
use crate::protocol::*;
use crate::rho_telemetry::{PreparedRhoTelemetry, RhoTelemetry};
use crate::state::{GlobalState, Layout, MergeCandidate, MergeStats};

const CHUNK_SIZE: usize = 4 * 1024 * 1024;
const WRITE_TIMEOUT: Duration = Duration::from_secs(180);
const WRITER_QUEUE: usize = 128;
static NEXT_CONNECTION_ID: AtomicU64 = AtomicU64::new(1);

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
    /// v3 finite-horizon outer bias correction: divide the applied Nesterov
    /// outer step at a fragment's t-th outer commit by `1 - mu^(t+1)` (see
    /// `GlobalState::outer_bias_correction`). Default false = bit-identical
    /// production path.
    pub outer_bias_correction: bool,
    /// Explicit opt-in age-aware controller definition, bound to the fragment
    /// layout and planned total steps after HELLO.
    pub outer_lr_controller: Option<crate::outer_lr_controller::ControllerConfig>,
    /// CTTN's dimensionless transverse curvature budget.
    pub cttn_rho: f32,
    /// CTTN's internal damping momentum, independent of fallback momentum.
    pub cttn_mu: f32,
    /// Total number of HVP sample points for cttn_shadow_v1.
    pub cttn_shadow_samples: u32,
    /// Post-merge renormalization for mediation-control experiments: rescale
    /// every merged delta to this L2 norm before the outer step. 0 = off
    /// (byte-identical production path). See `GlobalState::delta_norm_ref`.
    pub delta_norm_ref: f32,
    /// EXP2.46 3-arm current-anchor causal control: difference each learner's
    /// delta against the retained global at the learner's pushed base_version
    /// (version-matched anchoring) instead of the current global. Default
    /// false = byte-identical current-anchor. See docs/ANCHOR_DRIFT_CONTROL.md.
    pub version_matched_anchor: bool,
    /// EXP2.46: retain prior globals and log per-push anchor-drift diagnostics
    /// even without version-matching (the current-anchor arm still reports the
    /// drift it injects). Implied by `version_matched_anchor`.
    pub anchor_drift_instrument: bool,
    pub final_state: Option<std::path::PathBuf>,
    /// Consistent-snapshot file; written every `checkpoint_every` rounds at
    /// the quiescent cut between rounds, resumed from when `resume` is set.
    pub checkpoint_path: Option<std::path::PathBuf>,
    pub checkpoint_every: u64,
    pub resume: bool,
    /// JSONL event tape: one record per merge.
    pub event_tape: Option<std::path::PathBuf>,
    /// Opt-in pseudo-gradient autocorrelation/norm/worker-cosine JSONL.
    pub rho_telemetry: Option<std::path::PathBuf>,
    /// Optional offline probe capture directory. When enabled, complete_round
    /// writes a pre-merge syncer checkpoint and admitted candidate fragment
    /// tensors before applying the outer step.
    pub probe_capture_dir: Option<std::path::PathBuf>,
    /// Capture every Nth outer step. 0 disables capture.
    pub probe_capture_every: u64,
    /// Production commit policy. token_weighted retains the legacy mutating
    /// merge path exactly; probe policies use sealed State previews.
    pub commit_policy: CommitPolicy,
    /// Sidecar contract and connection settings, present only for probe
    /// policies.
    pub action_probe: Option<action_probe::ClientConfig>,
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
    connection_id: u64,
    validated: AtomicBool,
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
    Push { group: Arc<Group>, push: Push },
    Disconnected { group: Arc<Group> },
}

struct Push {
    learner_id: u32,
    connection_id: u64,
    fragment_id: u32,
    global_step: u64,
    base_version: u64,
    local_step: u64,
    c_steps: u32,
    c_tokens: u64,
    values: Vec<f32>,
}

/// EXP2.46 anchor-drift diagnostics for one push. The syncer differences a
/// learner's upload against its CURRENT global; the learner trained from the
/// global at `base_version`. `anchor_drift = current_global - base_global` is
/// the version-mismatch contamination injected into the current-anchor delta
/// (`server_delta = true_local_delta - anchor_drift`). See
/// docs/ANCHOR_DRIFT_CONTROL.md.
#[derive(Clone, Copy)]
struct AnchorDrift {
    /// ||current_global - learner_base_global||.
    drift_norm: f64,
    /// ||learner_upload - learner_base_global|| (the true local displacement).
    local_delta_norm: f64,
    /// drift_norm / local_delta_norm; None when the local delta is zero.
    ratio: Option<f64>,
    /// cos(anchor_drift, outer momentum buffer); None when either is zero.
    momentum_cos: Option<f64>,
    /// Whether the base version was still resident in the retained history.
    /// When false the drift is unmeasured and version-matching falls back to
    /// the current anchor for that push.
    base_resolved: bool,
}

impl AnchorDrift {
    const UNRESOLVED: AnchorDrift = AnchorDrift {
        drift_norm: 0.0,
        local_delta_norm: 0.0,
        ratio: None,
        momentum_cos: None,
        base_resolved: false,
    };
}

/// Compute the anchor-drift diagnostics for `push` against the current global
/// of fragment `fid`, using `push.values` as the (already Q4-reconstructed)
/// full learner upload. Must run BEFORE any version-matched re-anchoring so the
/// local delta reflects the learner's true window update.
fn compute_anchor_drift(st: &GlobalState, fid: usize, push: &Push) -> AnchorDrift {
    let current = &st.params[fid];
    let Some(base) = st.anchor_at(fid, push.base_version) else {
        return AnchorDrift::UNRESOLVED;
    };
    let momentum = st.momentum_fragment(fid);
    let (mut drift_sq, mut local_sq, mut dot_dm, mut mom_sq) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..current.len() {
        let drift = current[i] as f64 - base[i] as f64;
        let local = push.values[i] as f64 - base[i] as f64;
        let mom = momentum[i] as f64;
        drift_sq += drift * drift;
        local_sq += local * local;
        dot_dm += drift * mom;
        mom_sq += mom * mom;
    }
    let drift_norm = drift_sq.sqrt();
    let local_delta_norm = local_sq.sqrt();
    let ratio = (local_delta_norm > 0.0).then(|| drift_norm / local_delta_norm);
    let mom_norm = mom_sq.sqrt();
    let momentum_cos =
        (drift_norm > 0.0 && mom_norm > 0.0).then(|| dot_dm / (drift_norm * mom_norm));
    AnchorDrift {
        drift_norm,
        local_delta_norm,
        ratio,
        momentum_cos,
        base_resolved: true,
    }
}

fn validate_push_identity(connection_learner_id: u32, payload_learner_id: u32) -> Result<()> {
    if connection_learner_id != payload_learner_id {
        bail!(
            "push learner id {payload_learner_id} does not match connection learner id {connection_learner_id}"
        );
    }
    Ok(())
}

fn validate_push_candidate(
    push: &Push,
    expected_step: u64,
    expected_fragment: usize,
    current_version: u64,
    current_params: &[f32],
    wire_dtype: u8,
) -> Result<f64> {
    if push.global_step != expected_step {
        bail!(
            "push from learner {} targets step {}, expected {expected_step}",
            push.learner_id,
            push.global_step
        );
    }
    if push.fragment_id as usize != expected_fragment {
        bail!(
            "push from learner {} targets fragment {}, expected {expected_fragment}",
            push.learner_id,
            push.fragment_id
        );
    }
    if push.values.len() != current_params.len() {
        bail!(
            "push from learner {} for fragment {expected_fragment} has {} values, expected {}",
            push.learner_id,
            push.values.len(),
            current_params.len()
        );
    }
    if push.values.iter().any(|value| !value.is_finite()) {
        bail!(
            "push from learner {} for fragment {expected_fragment} contains non-finite values",
            push.learner_id
        );
    }
    if push.base_version > current_version {
        bail!(
            "push from learner {} for fragment {expected_fragment} has future base version {}, current version is {current_version}",
            push.learner_id,
            push.base_version
        );
    }
    if wire_dtype == DTYPE_Q4 && push.base_version != current_version {
        bail!(
            "q4 push from learner {} for fragment {expected_fragment} has stale base version {}, expected {current_version}",
            push.learner_id,
            push.base_version
        );
    }
    if wire_dtype == DTYPE_Q4
        && push
            .values
            .iter()
            .zip(current_params)
            .any(|(delta, anchor)| !(*delta + *anchor).is_finite())
    {
        bail!(
            "q4 push from learner {} for fragment {expected_fragment} overflows during reconstruction",
            push.learner_id
        );
    }
    let weight = crate::merge::learner_weight(push.c_tokens, push.c_steps);
    if !weight.is_finite() || weight <= 0.0 {
        bail!(
            "push from learner {} for fragment {expected_fragment} has non-positive weight {weight}",
            push.learner_id
        );
    }
    Ok(weight)
}

fn admit_push(round: &mut Round, push: Push, st: &GlobalState) -> Result<()> {
    if round.pushes.contains_key(&push.learner_id) {
        bail!(
            "duplicate push from learner {} for step {} fragment {}",
            push.learner_id,
            round.t,
            round.p
        );
    }
    validate_push_candidate(
        &push,
        round.t,
        round.p,
        st.versions[round.p],
        &st.params[round.p],
        st.wire_dtype,
    )?;
    match round.pushes.entry(push.learner_id) {
        Entry::Vacant(entry) => {
            entry.insert(push);
            Ok(())
        }
        Entry::Occupied(_) => unreachable!("duplicate checked before candidate validation"),
    }
}

fn reserve_fragment(busy_fragments: &mut HashSet<usize>, fragment: usize) -> bool {
    busy_fragments.insert(fragment)
}

fn ensure_fragment_version_advances(fragment: usize, current: u64, next: u64) -> Result<()> {
    if next <= current {
        bail!(
            "fragment {fragment} version would not advance: current {current}, incoming round {next}"
        );
    }
    Ok(())
}

fn sorted_push_ids(pushes: &HashMap<u32, Push>) -> Vec<u32> {
    let mut ids: Vec<u32> = pushes.keys().copied().collect();
    ids.sort_unstable();
    ids
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

fn next_fragment_steps(versions: &[u64]) -> Vec<u64> {
    let num_fragments = versions.len() as u64;
    debug_assert!(num_fragments > 0);
    versions
        .iter()
        .enumerate()
        .map(|(fragment, version)| {
            if *version == 0 {
                fragment as u64 + 1
            } else {
                version.saturating_add(num_fragments)
            }
        })
        .collect()
}

fn next_launchable_round(
    next_steps: &[u64],
    busy_fragments: &HashSet<usize>,
    total_steps: u64,
) -> Option<(u64, usize)> {
    next_steps
        .iter()
        .copied()
        .enumerate()
        .filter(|(fragment, step)| *step <= total_steps && !busy_fragments.contains(fragment))
        .map(|(fragment, step)| (step, fragment))
        .min_by_key(|(step, _)| *step)
}

fn has_pending_rounds(next_steps: &[u64], total_steps: u64) -> bool {
    next_steps.iter().any(|step| *step <= total_steps)
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

fn install_validated_group(registry: &Registry, group: Arc<Group>) -> Option<Arc<Group>> {
    group.validated.store(true, Ordering::Release);
    registry.lock().unwrap().insert(group.learner_id, group)
}

fn group_is_current(registry: &Registry, group: &Arc<Group>) -> bool {
    registry
        .lock()
        .unwrap()
        .get(&group.learner_id)
        .is_some_and(|current| Arc::ptr_eq(current, group))
}

fn remove_group_if_current(registry: &Registry, group: &Arc<Group>) -> bool {
    let mut registry = registry.lock().unwrap();
    let is_current = registry
        .get(&group.learner_id)
        .is_some_and(|current| Arc::ptr_eq(current, group));
    if is_current {
        registry.remove(&group.learner_id);
    }
    is_current
}

fn validated_group_count(registry: &Registry) -> usize {
    registry
        .lock()
        .unwrap()
        .values()
        .filter(|group| group.validated.load(Ordering::Acquire))
        .count()
}

fn current_connection_ids(registry: &Registry) -> HashMap<u32, u64> {
    registry
        .lock()
        .unwrap()
        .iter()
        .filter_map(|(learner_id, group)| {
            group
                .validated
                .load(Ordering::Acquire)
                .then_some((*learner_id, group.connection_id))
        })
        .collect()
}

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
                connection_id: NEXT_CONNECTION_ID.fetch_add(1, Ordering::Relaxed),
                validated: AtomicBool::new(false),
                dtype,
                layout,
                layout_meta,
                control: tx,
                data: Mutex::new(Vec::new()),
                msg_id: AtomicU64::new(0),
                rr: AtomicUsize::new(0),
                reasm: Mutex::new(HashMap::new()),
            });
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
            event_tx
                .send(Event::Disconnected {
                    group: group.clone(),
                })
                .await
                .ok();
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
            validate_push_identity(group.learner_id, learner_id)?;
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
                .send(Event::Push {
                    group: group.clone(),
                    push: Push {
                        learner_id,
                        connection_id: group.connection_id,
                        fragment_id,
                        global_step,
                        base_version,
                        local_step,
                        c_steps,
                        c_tokens,
                        values,
                    },
                })
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
        let connected = validated_group_count(&registry) as u32;
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
                if install_validated_group(&registry, group.clone()).is_some() {
                    warn!(
                        learner_id = group.learner_id,
                        "learner session replaced during init"
                    );
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
            Event::Push { .. } => warn!("push before initialization; dropped"),
            Event::Disconnected { group } => {
                if remove_group_if_current(&registry, &group) {
                    warn!(learner_id = group.learner_id, "disconnected during init");
                }
            }
        }
    }
    let mut st = state.unwrap();
    let num_fragments = st.layout.fragments.len();
    if num_fragments == 0 {
        bail!("syncer layout must contain at least one fragment");
    }
    let (mut action_probe_client, action_probe_unavailable) = if cfg.commit_policy.requires_probe()
    {
        match cfg.action_probe.clone() {
            Some(client_config) => match ActionProbeClient::bind(client_config, &st) {
                Ok(client) => (Some(client), None),
                Err(error) => {
                    warn!("action-probe disabled; all probe decisions will fall back to A0: {error:#}");
                    (None, Some("probe_layout_or_config_error".to_owned()))
                }
            },
            None => {
                warn!("action-probe policy has no client configuration; all decisions will fall back to A0");
                (None, Some("probe_not_configured".to_owned()))
            }
        }
    } else {
        (None, None)
    };
    let mut step_rates = StepRates::default();
    let mut last_sync_secs = 0.0f64; // previous round's merge+broadcast time
    let mut cttn_shadow = CttnShadowTracker::new(
        cfg.total_steps,
        num_fragments,
        if cfg.commit_policy.is_cttn_shadow() {
            cfg.cttn_shadow_samples
        } else {
            0
        },
    )?;
    if cfg.rho_telemetry.is_some() && cfg.commit_policy != CommitPolicy::TokenWeighted {
        bail!(
            "rho telemetry requires token_weighted commits so telemetry matches the committed aggregate"
        );
    }
    let mut rho_telemetry = cfg
        .rho_telemetry
        .as_ref()
        .map(|path| RhoTelemetry::new(path, num_fragments))
        .transpose()?;

    // Send everyone the initial (or resumed) global parameters so all
    // learners start bit-identical (also serves recovery for late joiners).
    broadcast_all_fragments(&st, &registry).await;

    // Phase 2: the outer loop. One fragment per global step, round-robin,
    // with up to `pipeline` rounds in flight at once: while round t sits in
    // its quorum/grace window, round t+1's pull is already out, so sync
    // latency overlaps learner compute (the paper's τ=2 "two fragments in
    // flight"). Depth is clamped to the fragment count and busy-fragment
    // tracking prevents a delayed fragment from being launched again after
    // round-robin wraparound. Concurrent rounds therefore always target
    // distinct params/momentum. Rounds may complete out of order; versions
    // are per fragment and global_step advances monotonically.
    let depth = (cfg.pipeline.max(1) as usize).min(num_fragments);
    let manual_floor = Duration::from_millis(cfg.min_round_interval_ms);
    let mut next_launch = Instant::now(); // earliest allowed next round launch
    let mut next_steps = next_fragment_steps(&st.versions);
    let mut inflight: Vec<Round> = Vec::new();
    let mut busy_fragments = HashSet::with_capacity(depth);
    while has_pending_rounds(&next_steps, cfg.total_steps) || !inflight.is_empty() {
        // Keep the pipeline full (throttled by min_round_interval_ms).
        while inflight.len() < depth && Instant::now() >= next_launch {
            let Some((t, p)) = next_launchable_round(&next_steps, &busy_fragments, cfg.total_steps)
            else {
                break;
            };
            let reserved = reserve_fragment(&mut busy_fragments, p);
            debug_assert!(reserved, "launchable fragment was already marked busy");
            next_steps[p] = t.saturating_add(num_fragments as u64);
            next_launch = Instant::now()
                + launch_interval(
                    manual_floor,
                    cfg.sync_interval_steps,
                    num_fragments,
                    step_rates.max_step_secs(),
                );
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

        let current_connections = current_connection_ids(&registry);
        let connected = current_connections.len();
        let k = if cfg.strict_quorum {
            cfg.quorum as usize
        } else {
            (cfg.quorum as usize).min(connected.max(1))
        };
        prune_noncurrent_pushes(
            &mut inflight,
            &current_connections,
            k,
            Instant::now(),
            Duration::from_secs(cfg.quorum_timeout_s),
        );

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
                let released = busy_fragments.remove(&round.p);
                debug_assert!(released, "completed fragment was not marked busy");
                complete_round(
                    &cfg,
                    &mut st,
                    &registry,
                    &mut last_sync_secs,
                    &mut action_probe_client,
                    action_probe_unavailable.as_deref(),
                    &mut cttn_shadow,
                    &mut rho_telemetry,
                    round,
                )
                .await?;
                completed_any = true;
                continue;
            }
            if expired {
                let r = &mut inflight[i];
                warn!(step = r.t, "round not ready at deadline; re-sending pull");
                for g in current_groups(&registry) {
                    let _ = g.send_small(MSG_PULL_REQ, r.pull.clone()).await;
                }
                reset_round_wait(r, Instant::now(), Duration::from_secs(cfg.quorum_timeout_s));
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
        if inflight.len() < depth
            && next_launchable_round(&next_steps, &busy_fragments, cfg.total_steps).is_some()
        {
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
                Event::Push { group, push } => {
                    let learner_id = push.learner_id;
                    let local_step = push.local_step;
                    if push.connection_id != group.connection_id
                        || !group.validated.load(Ordering::Acquire)
                        || !group_is_current(&registry, &group)
                    {
                        warn!(
                            learner_id,
                            "push from stale or unvalidated session rejected"
                        );
                        continue;
                    }
                    // Route to the in-flight round the pull came from.
                    if let Some(r) = inflight
                        .iter_mut()
                        .find(|r| r.t == push.global_step && r.p == push.fragment_id as usize)
                    {
                        let step = r.t;
                        let fragment = r.p;
                        match admit_push(r, push, &st) {
                            Ok(()) => {
                                step_rates.note(learner_id, local_step, Instant::now());
                            }
                            Err(e) => warn!(learner_id, step, fragment, "push rejected: {e:#}"),
                        }
                    } // else: stale response from a completed round; drop
                }
                Event::Hello { group } => {
                    // Rejoining learner: catch it up to the current state.
                    validate_group_compatible(&st, &group)?;
                    if install_validated_group(&registry, group.clone()).is_some() {
                        warn!(learner_id = group.learner_id, "learner session replaced");
                    }
                    send_all_fragments(&st, &group).await;
                }
                Event::Init { .. } => {} // already initialized; ignore
                Event::Disconnected { group } => {
                    if !remove_group_if_current(&registry, &group) {
                        continue;
                    }
                    let learner_id = group.learner_id;
                    warn!(learner_id, "learner disconnected");
                    let connected_now = validated_group_count(&registry);
                    let quorum_now = if cfg.strict_quorum {
                        cfg.quorum as usize
                    } else {
                        (cfg.quorum as usize).min(connected_now.max(1))
                    };
                    remove_connection_pushes(
                        &mut inflight,
                        learner_id,
                        group.connection_id,
                        quorum_now,
                        Instant::now(),
                        Duration::from_secs(cfg.quorum_timeout_s),
                    );
                }
            },
        }
    }

    if cfg.commit_policy.is_cttn_shadow() && !cttn_shadow.pending.is_empty() {
        bail!(
            "CTTN shadow ended with {} unresolved t+4 samples",
            cttn_shadow.pending.len()
        );
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

fn reset_round_wait(round: &mut Round, now: Instant, timeout: Duration) {
    round.grace_deadline = None;
    round.quorum_deadline = now + timeout;
}

fn prune_noncurrent_pushes(
    rounds: &mut [Round],
    current_connections: &HashMap<u32, u64>,
    quorum: usize,
    now: Instant,
    timeout: Duration,
) {
    for round in rounds {
        let before = round.pushes.len();
        round.pushes.retain(|learner_id, push| {
            current_connections.get(learner_id) == Some(&push.connection_id)
        });
        if round.pushes.len() != before && round.pushes.len() < quorum {
            reset_round_wait(round, now, timeout);
        }
    }
}

fn remove_connection_pushes(
    rounds: &mut [Round],
    learner_id: u32,
    connection_id: u64,
    quorum: usize,
    now: Instant,
    timeout: Duration,
) {
    for round in rounds {
        let belongs_to_connection = round
            .pushes
            .get(&learner_id)
            .is_some_and(|push| push.connection_id == connection_id);
        if belongs_to_connection {
            round.pushes.remove(&learner_id);
        }
        if belongs_to_connection && round.pushes.len() < quorum {
            reset_round_wait(round, now, timeout);
        }
    }
}

#[derive(Clone, Debug)]
struct PendingCttnShadow {
    sample_step: u64,
    fragment: usize,
    r_norm: f64,
    z_matrix: Vec<f32>,
    z_scalar: Vec<f32>,
    matrix: action_probe::CttnDiagnostics,
    scalar: action_probe::CttnDiagnostics,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ResolvedCttnShadow {
    sample_step: u64,
    future_step: u64,
    fragment: usize,
    r_norm: f64,
    matrix_alignment: Option<f64>,
    scalar_alignment: Option<f64>,
    matrix: action_probe::CttnDiagnostics,
    scalar: action_probe::CttnDiagnostics,
}

#[derive(Debug, Default)]
struct CttnShadowTracker {
    sample_steps: HashSet<u64>,
    pending: BTreeMap<u64, PendingCttnShadow>,
}

impl CttnShadowTracker {
    fn new(total_steps: u64, fragments: usize, samples: u32) -> Result<Self> {
        if fragments == 0 {
            bail!("CTTN shadow requires at least one fragment");
        }
        if samples == 0 {
            return Ok(Self::default());
        }
        if fragments != 4 {
            bail!("cttn_shadow_v1 requires exactly four fragments for the t+4 lookup");
        }
        if samples as usize % fragments != 0 {
            bail!(
                "--cttn-shadow-samples ({samples}) must be divisible by the fragment count ({fragments})"
            );
        }
        let per_fragment = samples as usize / fragments;
        let mut sample_steps = HashSet::with_capacity(samples as usize);
        for fragment in 0..fragments {
            let all_steps: Vec<u64> = (fragment as u64 + 1..=total_steps)
                .step_by(fragments)
                .collect();
            let available = all_steps.len().saturating_sub(1);
            if available < per_fragment {
                bail!(
                    "CTTN shadow needs {per_fragment} resolvable samples for fragment {fragment}, but only {available} fit before the final merge"
                );
            }
            for stratum in 0..per_fragment {
                // Midpoint of each equal-width fragment-local training stratum.
                let index = ((2 * stratum + 1) * available) / (2 * per_fragment);
                sample_steps.insert(all_steps[index]);
            }
        }
        if sample_steps.len() != samples as usize {
            bail!("CTTN shadow sample schedule did not produce {samples} unique steps");
        }
        Ok(Self {
            sample_steps,
            pending: BTreeMap::new(),
        })
    }

    fn should_sample(&self, step: u64) -> bool {
        self.sample_steps.contains(&step)
    }

    fn insert(&mut self, target_step: u64, sample: PendingCttnShadow) -> Result<()> {
        if self.pending.insert(target_step, sample).is_some() {
            bail!("duplicate CTTN shadow target step {target_step}");
        }
        Ok(())
    }

    fn resolve(&mut self, step: u64, fragment: usize, future_g: &[f32]) -> Result<Option<ResolvedCttnShadow>> {
        let Some(pending) = self.pending.remove(&step) else {
            return Ok(None);
        };
        if pending.fragment != fragment {
            bail!(
                "CTTN shadow sample {} expected fragment {}, got fragment {fragment} at step {step}",
                pending.sample_step,
                pending.fragment
            );
        }
        Ok(Some(ResolvedCttnShadow {
            sample_step: pending.sample_step,
            future_step: step,
            fragment,
            r_norm: pending.r_norm,
            matrix_alignment: predictive_alignment(&pending.z_matrix, pending.r_norm, future_g),
            scalar_alignment: predictive_alignment(&pending.z_scalar, pending.r_norm, future_g),
            matrix: pending.matrix,
            scalar: pending.scalar,
        }))
    }
}

fn predictive_alignment(z: &[f32], r_norm: f64, future_g: &[f32]) -> Option<f64> {
    if z.len() != future_g.len() || !r_norm.is_finite() || r_norm <= 0.0 {
        return None;
    }
    let mut dot = 0.0f64;
    let mut g_norm_sq = 0.0f64;
    for (&z_value, &g_value) in z.iter().zip(future_g) {
        dot += z_value as f64 * g_value as f64;
        g_norm_sq += (g_value as f64).powi(2);
    }
    if !dot.is_finite() || !g_norm_sq.is_finite() || g_norm_sq <= 0.0 {
        return None;
    }
    let value = dot / (r_norm * g_norm_sq.sqrt());
    value.is_finite().then_some(value)
}

fn transverse_norm(g: &[f32], buffer: &[f32]) -> f64 {
    if g.len() != buffer.len() {
        return f64::NAN;
    }
    let g_norm_sq = g.iter().map(|value| (*value as f64).powi(2)).sum::<f64>();
    if g_norm_sq <= 0.0 {
        return 0.0;
    }
    let projection = g
        .iter()
        .zip(buffer)
        .map(|(g, b)| *g as f64 * *b as f64)
        .sum::<f64>()
        / g_norm_sq;
    g.iter()
        .zip(buffer)
        .map(|(g, b)| {
            let residual = *b as f64 - projection * *g as f64;
            residual * residual
        })
        .sum::<f64>()
        .sqrt()
}

#[derive(Clone, Debug)]
struct CommitDecision {
    policy: CommitPolicy,
    selected_action: String,
    committed_action: String,
    fallback: bool,
    fallback_reason: Option<String>,
    probe_latency_ms: Option<f64>,
    selected_mass: f64,
    norm_scale: f64,
    step_ratio: f64,
    selected_multiplier: f64,
    committed_multiplier: f64,
    request_digest: Option<String>,
    cttn_diagnostics: Option<action_probe::CttnDiagnostics>,
    cttn_shadow_resolved: Option<ResolvedCttnShadow>,
}

impl CommitDecision {
    fn token_weighted() -> Self {
        Self {
            policy: CommitPolicy::TokenWeighted,
            selected_action: "A0".to_owned(),
            committed_action: "A0".to_owned(),
            fallback: false,
            fallback_reason: None,
            probe_latency_ms: None,
            selected_mass: 1.0,
            norm_scale: 1.0,
            step_ratio: 1.0,
            selected_multiplier: 1.0,
            committed_multiplier: 1.0,
            request_digest: None,
            cttn_diagnostics: None,
            cttn_shadow_resolved: None,
        }
    }

    fn probe_fallback(policy: CommitPolicy, reason: impl Into<String>) -> Self {
        Self {
            policy,
            selected_action: "A0".to_owned(),
            committed_action: "A0".to_owned(),
            fallback: true,
            fallback_reason: Some(reason.into()),
            probe_latency_ms: None,
            selected_mass: 1.0,
            norm_scale: 1.0,
            step_ratio: 1.0,
            selected_multiplier: 1.0,
            committed_multiplier: 1.0,
            request_digest: None,
            cttn_diagnostics: None,
            cttn_shadow_resolved: None,
        }
    }
}

fn selected_preview_multiplier(previews: &RetainedPreviews, index: usize) -> f64 {
    previews.metadata(index).step_scale.unwrap_or_else(|| {
        if index == 0 {
            1.0
        } else {
            previews.metadata(index).norm_multiplier
        }
    })
}

fn commit_preview_index(policy: CommitPolicy, selected_index: usize) -> usize {
    if policy.is_shadow() {
        0
    } else {
        selected_index
    }
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
    action_probe_client: &mut Option<ActionProbeClient>,
    action_probe_unavailable: Option<&str>,
    cttn_shadow: &mut CttnShadowTracker,
    rho_telemetry: &mut Option<RhoTelemetry>,
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
    ensure_fragment_version_advances(p, prev_version, t)?;
    if pushes.is_empty() {
        bail!("round {t} fragment {p} has no admitted pushes");
    }
    // Every push here already passed validate_push_candidate at admission
    // (admit_push), and the fragment cannot change between admission and
    // completion: rounds on a fragment are serialized by the busy-fragment
    // reservation and merges run on the single scheduler task (the Q4
    // reconstruction below has always relied on exactly this invariant).
    // Re-running the O(N)-per-push validation here doubled the finite-scan
    // cost of every commit for no additional safety.
    if st.wire_dtype == DTYPE_Q4 {
        // Q4 pushes are deltas anchored at the learner's base_version;
        // reconstruction needs Θ at that exact version, and the syncer
        // only holds the current value. Admission already rejected every
        // non-matching base before it could count toward quorum.
        for push in pushes.values_mut() {
            for (v, a) in push.values.iter_mut().zip(&st.params[p]) {
                *v += *a;
            }
        }
    }
    capture_round_candidates(cfg, st, p, t, prev_version, &pushes)?;
    // EXP2.46: anchor-drift diagnostics + optional version-matched re-anchoring.
    // Computed from the original uploads (post-Q4-reconstruction full models)
    // BEFORE any re-anchoring, so the local delta reflects the true window
    // update. Empty (and the whole block skipped) unless retention is enabled,
    // keeping the default path byte-identical.
    let anchor_drift: HashMap<u32, AnchorDrift> = if st.anchor_retention_enabled() {
        let drift: HashMap<u32, AnchorDrift> = pushes
            .values()
            .map(|push| (push.learner_id, compute_anchor_drift(st, p, push)))
            .collect();
        if st.version_matched_anchor {
            // Re-anchor each upload so the current-anchor merge yields the delta
            // against the learner's OWN base: values' = values + (current -
            // base). Then current - values' = base - values (the version-matched
            // pseudo-gradient) for EVERY merge mode, without touching merge.rs.
            // A push whose base aged out of history keeps the current anchor
            // (its drift is reported unresolved).
            for push in pushes.values_mut() {
                let Some(base) = st.anchor_at(p, push.base_version) else {
                    continue;
                };
                for (v, (c, b)) in push.values.iter_mut().zip(st.params[p].iter().zip(base)) {
                    *v += *c - *b;
                }
            }
        }
        drift
    } else {
        HashMap::new()
    };
    let ids = sorted_push_ids(&pushes);
    let (mut learners, mut weights) = (Vec::new(), Vec::new());
    for id in &ids {
        let push = pushes.get(id).expect("sorted id must exist in push map");
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
    }
    let sync_start = Instant::now();
    let mut prepared_rho: Option<PreparedRhoTelemetry> = None;
    let (merge_stats, decision) = if cfg.commit_policy == CommitPolicy::TokenWeighted {
        // Keep the legacy production path intact. In particular, do not
        // replace this with preview+commit: token_weighted is the bit-for-bit
        // baseline comparator for every probe experiment.
        let merge_stats = if let Some(telemetry) = rho_telemetry.as_ref() {
            let candidates = ids
                .iter()
                .zip(&weights)
                .map(|(id, &weight)| {
                    let push = pushes.get(id).expect("sorted id must exist in push map");
                    MergeCandidate::new(*id, push.values.as_slice(), weight)
                })
                .collect::<Vec<_>>();
            // Trusted in-process aggregate: consumed by apply_aggregate_step
            // below in this same task; skips the O(N) base-state fingerprint.
            let aggregate = st.build_full_aggregate_trusted(p, &candidates)?;
            let worker_candidates = ids
                .iter()
                .map(|id| {
                    let push = pushes.get(id).expect("sorted id must exist in push map");
                    (*id, push.values.as_slice())
                })
                .collect::<Vec<_>>();
            prepared_rho = Some(telemetry.prepare(
                t,
                p,
                &st.layout.fragments[p].tensor_numels,
                aggregate.delta(),
                &st.params[p],
                &worker_candidates,
            )?);
            st.apply_aggregate_step(aggregate)?
        } else {
            st.merge_and_step(p, &learners, &weights)?
        };
        st.versions[p] = t;
        // Pipelined rounds can complete out of order; the global step only
        // moves forward.
        st.global_step = st.global_step.max(t);
        (merge_stats, CommitDecision::token_weighted())
    } else {
        let candidates = ids
            .iter()
            .zip(&weights)
            .map(|(id, &weight)| {
                let push = pushes.get(id).expect("sorted id must exist in push map");
                MergeCandidate::new(*id, push.values.as_slice(), weight)
            })
            .collect::<Vec<_>>();
        if cfg.commit_policy.is_cttn_shadow() {
            let aggregate = st.build_full_aggregate(p, &candidates)?;
            let inputs = st.cttn_shadow_inputs(&aggregate, cfg.cttn_mu)?;
            let resolved = cttn_shadow.resolve(t, p, &inputs.g)?;
            let mut decision = CommitDecision {
                policy: cfg.commit_policy,
                selected_action: "SGD".to_owned(),
                committed_action: "SGD".to_owned(),
                fallback: false,
                fallback_reason: None,
                probe_latency_ms: None,
                selected_mass: aggregate.selected_weight_mass(),
                norm_scale: 1.0,
                step_ratio: 1.0,
                selected_multiplier: 1.0,
                committed_multiplier: 1.0,
                request_digest: None,
                cttn_diagnostics: None,
                cttn_shadow_resolved: resolved,
            };
            if cttn_shadow.should_sample(t) {
                if let Some(reason) = action_probe_unavailable {
                    decision.fallback = true;
                    decision.fallback_reason = Some(reason.to_owned());
                } else {
                    let probe_started = Instant::now();
                    let client = action_probe_client
                        .as_mut()
                        .expect("available action probe must have a client");
                    match client
                        .cttn_shadow_step(
                            st,
                            &aggregate,
                            t,
                            &inputs.g,
                            &inputs.b,
                            inputs.mu,
                            cfg.cttn_rho,
                            4,
                        )
                        .await
                    {
                        Ok(verified) => {
                            decision.selected_action = "CTTN-SHADOW".to_owned();
                            decision.probe_latency_ms =
                                Some(probe_started.elapsed().as_secs_f64() * 1000.0);
                            decision.request_digest = Some(verified.request_digest);
                            let r_norm = transverse_norm(&inputs.g, &inputs.b);
                            cttn_shadow.insert(
                                t + 4,
                                PendingCttnShadow {
                                    sample_step: t,
                                    fragment: p,
                                    r_norm,
                                    z_matrix: verified.z_matrix,
                                    z_scalar: verified.z_scalar,
                                    matrix: verified.matrix_diagnostics,
                                    scalar: verified.scalar_diagnostics,
                                },
                            )?;
                        }
                        Err(error) => {
                            warn!(
                                step = t,
                                fragment = p,
                                "CTTN shadow sidecar failed; SGD trajectory is unchanged: {error}"
                            );
                            decision.fallback = true;
                            decision.fallback_reason = Some(error.code().to_owned());
                            decision.probe_latency_ms =
                                Some(probe_started.elapsed().as_secs_f64() * 1000.0);
                            decision.request_digest =
                                client.last_request_digest().map(str::to_owned);
                        }
                    }
                }
            }
            let stats = st.commit_cttn_shadow_sgd(&aggregate, t, cfg.cttn_mu)?;
            (stats, decision)
        } else if let Some(cttn_mode) = cfg.commit_policy.cttn_mode() {
            let aggregate = st.build_full_aggregate(p, &candidates)?;
            let inputs = st.cttn_inputs(&aggregate, cfg.cttn_mu)?;
            let baseline = st.preview_aggregate(&aggregate, t)?;
            if let Some(reason) = action_probe_unavailable {
                let stats = st.commit_preview(baseline)?;
                (
                    stats,
                    CommitDecision::probe_fallback(cfg.commit_policy, reason),
                )
            } else {
                let probe_started = Instant::now();
                let client = action_probe_client
                    .as_mut()
                    .expect("available action probe must have a client");
                match client
                    .cttn_step(
                        st,
                        &aggregate,
                        t,
                        &inputs.g,
                        &inputs.b,
                        inputs.mu,
                        cfg.cttn_rho,
                        4,
                        cttn_mode,
                    )
                    .await
                {
                    Ok(verified) => {
                        let latency = probe_started.elapsed().as_secs_f64() * 1000.0;
                        let diagnostics = verified.diagnostics;
                        let request_digest = verified.request_digest;
                        let action = if cttn_mode == "scalar" {
                            "CTTN-SCALAR"
                        } else {
                            "CTTN"
                        };
                        match st.commit_cttn_step(
                            &aggregate,
                            t,
                            &verified.d,
                            &verified.b_new,
                            inputs.outer_lr,
                        ) {
                            Ok(stats) => (
                                stats,
                                CommitDecision {
                                    policy: cfg.commit_policy,
                                    selected_action: action.to_owned(),
                                    committed_action: action.to_owned(),
                                    fallback: false,
                                    fallback_reason: None,
                                    probe_latency_ms: Some(latency),
                                    selected_mass: aggregate.selected_weight_mass(),
                                    norm_scale: 1.0,
                                    step_ratio: 1.0,
                                    selected_multiplier: 1.0,
                                    committed_multiplier: 1.0,
                                    request_digest: Some(request_digest),
                                    cttn_diagnostics: Some(diagnostics),
                                    cttn_shadow_resolved: None,
                                },
                            ),
                            Err(error) => {
                                warn!(
                                    step = t,
                                    fragment = p,
                                    "CTTN commit failed closed to baseline: {error:#}"
                                );
                                let stats = st.commit_preview(baseline)?;
                                let mut decision = CommitDecision::probe_fallback(
                                    cfg.commit_policy,
                                    "cttn_commit_error",
                                );
                                decision.probe_latency_ms = Some(latency);
                                decision.request_digest = Some(request_digest);
                                (stats, decision)
                            }
                        }
                    }
                    Err(error) => {
                        warn!(
                            step = t,
                            fragment = p,
                            "CTTN sidecar failed closed to baseline: {error}"
                        );
                        let stats = st.commit_preview(baseline)?;
                        let mut decision =
                            CommitDecision::probe_fallback(cfg.commit_policy, error.code());
                        decision.probe_latency_ms =
                            Some(probe_started.elapsed().as_secs_f64() * 1000.0);
                        decision.request_digest = client.last_request_digest().map(str::to_owned);
                        (stats, decision)
                    }
                }
            }
        } else {
            let baseline = action_probe::build_baseline_preview(st, p, t, &candidates)?;

            if cfg.commit_policy.is_leave_one_out() && candidates.len() != 4 {
                let reason = format!("incomplete_group_{}_of_4", candidates.len());
                let stats = st.commit_preview(baseline)?;
                (
                    stats,
                    CommitDecision::probe_fallback(cfg.commit_policy, reason),
                )
            } else {
                let retained = if cfg.commit_policy.is_leave_one_out() {
                    action_probe::build_leave_one_out_previews(st, p, t, &candidates, &baseline)
                        .and_then(|alternatives| {
                            RetainedPreviews::loo_v1(baseline.clone(), alternatives)
                        })
                } else {
                    let multipliers = cfg
                        .commit_policy
                        .step_scale_multipliers()
                        .context("scalar probe policy is missing its frozen multiplier grid")?;
                    action_probe::build_scaled_full_group_previews(st, &baseline, multipliers)
                };
                match retained {
                    Err(error) => {
                        warn!(
                            step = t,
                            fragment = p,
                            "action preview construction failed closed to A0: {error:#}"
                        );
                        let stats = st.commit_preview(baseline)?;
                        (
                            stats,
                            CommitDecision::probe_fallback(
                                cfg.commit_policy,
                                "preview_construction_error",
                            ),
                        )
                    }
                    Ok(mut preview_set) => {
                        let mut decision = if let Some(reason) = action_probe_unavailable {
                            CommitDecision::probe_fallback(cfg.commit_policy, reason)
                        } else {
                            let probe_started = Instant::now();
                            let client = action_probe_client
                                .as_mut()
                                .expect("available action probe must have a client");
                            match client.select(st, &preview_set, t, p).await {
                                Ok(selection) => {
                                    let metadata = preview_set.metadata(selection.action_index);
                                    CommitDecision {
                                        policy: cfg.commit_policy,
                                        selected_action: selection.action_name,
                                        committed_action: String::new(),
                                        fallback: selection.action_index == 0,
                                        fallback_reason: selection.fallback_reason,
                                        probe_latency_ms: Some(
                                            probe_started.elapsed().as_secs_f64() * 1000.0,
                                        ),
                                        selected_mass: metadata.selected_mass,
                                        norm_scale: metadata.norm_multiplier,
                                        step_ratio: metadata.step_norm_ratio,
                                        selected_multiplier: selected_preview_multiplier(
                                            &preview_set,
                                            selection.action_index,
                                        ),
                                        committed_multiplier: 1.0,
                                        request_digest: Some(selection.request_digest),
                                        cttn_diagnostics: None,
                                        cttn_shadow_resolved: None,
                                    }
                                }
                                Err(error) => {
                                    warn!(
                                        step = t,
                                        fragment = p,
                                        "action probe failed closed to A0: {error}"
                                    );
                                    let mut fallback = CommitDecision::probe_fallback(
                                        cfg.commit_policy,
                                        error.code(),
                                    );
                                    fallback.probe_latency_ms =
                                        Some(probe_started.elapsed().as_secs_f64() * 1000.0);
                                    fallback.request_digest =
                                        client.last_request_digest().map(str::to_owned);
                                    fallback
                                }
                            }
                        };
                        let selected_index =
                            preview_set.index_of(&decision.selected_action).unwrap_or(0);
                        let commit_index = commit_preview_index(cfg.commit_policy, selected_index);
                        decision.committed_action = preview_set.name(commit_index).to_owned();
                        decision.committed_multiplier =
                            selected_preview_multiplier(&preview_set, commit_index);
                        let stats = match st.commit_preview(preview_set.take(commit_index)) {
                            Ok(stats) => stats,
                            Err(error) if commit_index != 0 => {
                                warn!(
                                step = t,
                                fragment = p,
                                "selected preview was stale or invalid at commit; retrying exact A0: {error:#}"
                            );
                                decision.fallback = true;
                                decision.fallback_reason =
                                    Some("selected_preview_commit_error".to_owned());
                                decision.committed_action = "A0".to_owned();
                                decision.committed_multiplier =
                                    selected_preview_multiplier(&preview_set, 0);
                                st.commit_preview(preview_set.take(0)).with_context(|| {
                                format!(
                                    "step {t} fragment {p}: selected preview and A0 fallback both failed"
                                )
                            })?
                            }
                            Err(error) => return Err(error),
                        };
                        (stats, decision)
                    }
                }
            }
        }
    };
    for id in &ids {
        let push = pushes.get(id).expect("sorted id must exist in push map");
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
        policy = %decision.policy,
        selected_action = %decision.selected_action,
        committed_action = %decision.committed_action,
        selected_multiplier = decision.selected_multiplier,
        committed_multiplier = decision.committed_multiplier,
        fallback = decision.fallback,
        ms,
        "outer step"
    );
    if let Some(tape) = &cfg.event_tape {
        // Records land in completion order, which under pipelining is not
        // necessarily step order.
        append_tape(
            tape,
            t,
            p,
            &pushes,
            &weights,
            &merge_stats,
            &decision,
            &anchor_drift,
            ms,
        );
    }
    if let Some(prepared) = prepared_rho {
        rho_telemetry
            .as_mut()
            .expect("prepared rho telemetry requires an enabled writer")
            .append(prepared)?;
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
    st.outer_bias_correction = cfg.outer_bias_correction;
    if let Some(controller) = &cfg.outer_lr_controller {
        st.set_outer_lr_controller(
            controller.bind(cfg.total_steps, group.layout.fragments.len())?,
        )?;
    }
    st.delta_norm_ref = cfg.delta_norm_ref;
    st.version_matched_anchor = cfg.version_matched_anchor;
    st.anchor_drift_instrument = cfg.anchor_drift_instrument;
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
    registry
        .lock()
        .unwrap()
        .values()
        .filter(|group| group.validated.load(Ordering::Acquire))
        .cloned()
        .collect()
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
#[allow(clippy::too_many_arguments)]
fn append_tape(
    path: &std::path::Path,
    step: u64,
    fragment: usize,
    pushes: &HashMap<u32, Push>,
    _weights: &[f64],
    stats: &MergeStats,
    decision: &CommitDecision,
    anchor_drift: &HashMap<u32, AnchorDrift>,
    ms: u64,
) {
    use std::io::Write;
    let line = format_tape_line(step, fragment, pushes, stats, decision, anchor_drift, ms);
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

fn optional_json_string(value: Option<&str>) -> String {
    value
        .and_then(|value| serde_json::to_string(value).ok())
        .unwrap_or_else(|| "null".to_owned())
}

#[allow(clippy::too_many_arguments)]
fn format_tape_line(
    step: u64,
    fragment: usize,
    pushes: &HashMap<u32, Push>,
    stats: &MergeStats,
    decision: &CommitDecision,
    anchor_drift: &HashMap<u32, AnchorDrift>,
    ms: u64,
) -> String {
    let mut responders: Vec<String> = pushes
        .values()
        .map(|p| {
            let head = format!(
                "{{\"id\":{},\"base_version\":{},\"c_steps\":{},\"c_tokens\":{},\"weight\":{}",
                p.learner_id,
                p.base_version,
                p.c_steps,
                p.c_tokens,
                crate::merge::learner_weight(p.c_tokens, p.c_steps)
            );
            // EXP2.46: append anchor-drift diagnostics only when instrumentation
            // is active, so the default tape schema stays byte-identical.
            match anchor_drift.get(&p.learner_id) {
                None => format!("{head}}}"),
                Some(d) => format!(
                    "{head},\"anchor_drift_norm\":{},\"local_delta_norm\":{},\"anchor_drift_ratio\":{},\"anchor_drift_momentum_cos\":{},\"anchor_base_resolved\":{}}}",
                    json_number(d.drift_norm),
                    json_number(d.local_delta_norm),
                    optional_json_number(d.ratio),
                    optional_json_number(d.momentum_cos),
                    d.base_resolved,
                ),
            }
        })
        .collect();
    responders.sort();
    let outer = stats.outer;
    let gnorm = json_number(stats.gnorm);
    // Appended only when transient normalization is active, so the default
    // tape schema stays byte-identical (same idiom as anchor drift).
    let outer_bias_correction = stats
        .outer_bias_correction
        .map_or_else(String::new, |scale| {
            format!(",\"outer_bias_correction\":{}", json_number(scale))
        });
    let outer_lr_controller = stats.outer_lr_controller.map_or_else(String::new, |controller| {
        let mode = serde_json::to_string(controller.mode.as_str()).unwrap();
        format!(
            ",\"outer_lr_controller_mode\":{mode},\"outer_lr_scale\":{},\"outer_lr_transient_scale\":{},\"outer_lr_drift_scale\":{}",
            json_number(controller.scale),
            optional_json_number(controller.transient_scale),
            optional_json_number(controller.drift_scale),
        )
    });
    let outer_step_norm = json_number(outer.applied_step_norm);
    let outer_direction_cosine = optional_json_number(outer.direction_delta_cosine);
    let outer_history_current_ratio = optional_json_number(outer.history_current_norm_ratio);
    let policy = serde_json::to_string(&decision.policy.to_string()).unwrap();
    let selected_action = serde_json::to_string(&decision.selected_action).unwrap();
    let committed_action = serde_json::to_string(&decision.committed_action).unwrap();
    let fallback_reason = optional_json_string(decision.fallback_reason.as_deref());
    let probe_latency_ms = optional_json_number(decision.probe_latency_ms);
    let selected_mass = json_number(decision.selected_mass);
    let norm_scale = json_number(decision.norm_scale);
    let step_ratio = json_number(decision.step_ratio);
    let selected_multiplier = json_number(decision.selected_multiplier);
    let committed_multiplier = json_number(decision.committed_multiplier);
    let request_digest = optional_json_string(decision.request_digest.as_deref());
    let cttn_diagnostics = decision.cttn_diagnostics.map_or_else(String::new, |diag| {
        format!(
            ",\"cttn_bind\":{},\"cttn_tau\":{},\"cttn_retention\":{},\"cttn_e_before\":{},\"cttn_e_after\":{},\"cttn_budget\":{},\"cttn_n_modes_90\":{},\"cttn_ritz_max\":{},\"cttn_loss\":{}",
            diag.bind,
            optional_json_number(diag.tau),
            json_number(diag.retention),
            json_number(diag.e_before),
            json_number(diag.e_after),
            json_number(diag.budget),
            diag.n_modes_90,
            json_number(diag.ritz_max),
            json_number(diag.loss),
        )
    });
    let cttn_shadow = decision.cttn_shadow_resolved.map_or_else(String::new, |sample| {
        format!(
            ",\"cttn_shadow_sample_step\":{},\"cttn_shadow_future_step\":{},\"cttn_shadow_fragment\":{},\"cttn_shadow_r_norm\":{},\"cttn_shadow_bind\":{},\"cttn_shadow_retention\":{},\"cttn_shadow_ritz_max\":{},\"cttn_shadow_matrix_alignment\":{},\"cttn_shadow_scalar_alignment\":{},\"cttn_shadow_scalar_bind\":{},\"cttn_shadow_scalar_retention\":{}",
            sample.sample_step,
            sample.future_step,
            sample.fragment,
            json_number(sample.r_norm),
            sample.matrix.bind,
            json_number(sample.matrix.retention),
            json_number(sample.matrix.ritz_max),
            optional_json_number(sample.matrix_alignment),
            optional_json_number(sample.scalar_alignment),
            sample.scalar.bind,
            json_number(sample.scalar.retention),
        )
    });
    format!(
        "{{\"step\":{step},\"fragment\":{fragment},\"gnorm\":{gnorm},\"ms\":{ms},\"responders\":[{}],\"outer_step_norm\":{outer_step_norm},\"outer_direction_cosine\":{outer_direction_cosine},\"outer_history_current_ratio\":{outer_history_current_ratio},\"outer_restarted\":{},\"policy\":{policy},\"selected_action\":{selected_action},\"committed_action\":{committed_action},\"selected_multiplier\":{selected_multiplier},\"committed_multiplier\":{committed_multiplier},\"fallback\":{},\"fallback_reason\":{fallback_reason},\"probe_latency_ms\":{probe_latency_ms},\"selected_mass\":{selected_mass},\"norm_scale\":{norm_scale},\"step_ratio\":{step_ratio},\"request_digest\":{request_digest}{outer_bias_correction}{outer_lr_controller}{cttn_diagnostics}{cttn_shadow}}}\n",
        responders.join(","),
        outer.restarted,
        decision.fallback,
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
            outer_bias_correction: None,
            outer_lr_controller: None,
        }
    }

    fn test_state(dtype: u8) -> GlobalState {
        let layout = Layout {
            fragments: vec![crate::state::FragmentInfo {
                merge_mode: crate::state::MERGE_AVG,
                tensor_numels: vec![2],
                tensor_shapes: None,
            }],
        };
        let mut st = GlobalState::new(layout, None, 0.1, 0.0, dtype);
        st.versions[0] = 7;
        st
    }

    fn test_round() -> Round {
        let now = Instant::now();
        Round {
            t: 9,
            p: 0,
            pull: bytes::Bytes::new(),
            started: now,
            quorum_deadline: now + Duration::from_secs(1),
            grace_deadline: None,
            pushes: HashMap::new(),
        }
    }

    fn test_push(learner_id: u32) -> Push {
        Push {
            learner_id,
            connection_id: 1,
            fragment_id: 0,
            global_step: 9,
            base_version: 7,
            local_step: 21,
            c_steps: 10,
            c_tokens: 100,
            values: vec![1.0, 2.0],
        }
    }

    fn test_group(learner_id: u32, validated: bool) -> Arc<Group> {
        let (control, _control_rx) = mpsc::channel(1);
        Arc::new(Group {
            learner_id,
            connection_id: NEXT_CONNECTION_ID.fetch_add(1, Ordering::Relaxed),
            validated: AtomicBool::new(validated),
            dtype: DTYPE_F32,
            layout: test_state(DTYPE_F32).layout,
            layout_meta: None,
            control,
            data: Mutex::new(Vec::new()),
            msg_id: AtomicU64::new(0),
            rr: AtomicUsize::new(0),
            reasm: Mutex::new(HashMap::new()),
        })
    }

    #[test]
    fn push_identity_must_match_connection_identity() {
        validate_push_identity(3, 3).unwrap();
        let err = validate_push_identity(3, 4).unwrap_err().to_string();
        assert!(err.contains("does not match connection learner id"));
    }

    #[test]
    fn busy_fragment_cannot_be_reserved_twice() {
        let mut busy = HashSet::new();
        assert!(reserve_fragment(&mut busy, 2));
        assert!(!reserve_fragment(&mut busy, 2));
        assert!(busy.remove(&2));
        assert!(reserve_fragment(&mut busy, 2));
    }

    #[test]
    fn scheduler_skips_busy_fragment_without_launching_it_twice() {
        let next_steps = vec![5, 6, 7, 8];
        let mut busy = HashSet::from([0]);

        assert_eq!(next_launchable_round(&next_steps, &busy, 20), Some((6, 1)));
        assert!(reserve_fragment(&mut busy, 1));
        assert_eq!(next_launchable_round(&next_steps, &busy, 20), Some((7, 2)));

        assert!(busy.remove(&0));
        assert_eq!(next_launchable_round(&next_steps, &busy, 20), Some((5, 0)));
    }

    #[test]
    fn fragment_step_schedule_resumes_at_each_fragments_next_turn() {
        assert_eq!(next_fragment_steps(&[0, 0, 0, 0]), vec![1, 2, 3, 4]);
        assert_eq!(next_fragment_steps(&[5, 2, 3, 4]), vec![9, 6, 7, 8]);
        assert_eq!(next_fragment_steps(&[1, 6, 7, 8]), vec![5, 10, 11, 12]);
        assert!(has_pending_rounds(&[9, 6, 7, 8], 8));
        assert!(!has_pending_rounds(&[9, 10, 11, 12], 8));
    }

    #[test]
    fn fragment_version_must_advance_strictly() {
        ensure_fragment_version_advances(2, 8, 9).unwrap();
        assert!(ensure_fragment_version_advances(2, 8, 8).is_err());
        assert!(ensure_fragment_version_advances(2, 8, 7).is_err());
    }

    #[test]
    fn invalid_candidates_never_count_toward_quorum() {
        let st = test_state(DTYPE_F32);
        let mut invalid = Vec::new();

        let mut wrong_shape = test_push(1);
        wrong_shape.values.pop();
        invalid.push(wrong_shape);

        let mut non_finite = test_push(2);
        non_finite.values[1] = f32::NAN;
        invalid.push(non_finite);

        let mut future_base = test_push(3);
        future_base.base_version = 8;
        invalid.push(future_base);

        let mut zero_weight = test_push(4);
        zero_weight.c_tokens = 0;
        invalid.push(zero_weight);

        for push in invalid {
            let mut round = test_round();
            assert!(admit_push(&mut round, push, &st).is_err());
            assert!(round.pushes.is_empty());
            assert!(!round_completion_ready(
                round.pushes.len(),
                4,
                1,
                true,
                false
            ));
        }
    }

    #[test]
    fn stale_q4_candidate_is_rejected_before_quorum() {
        let st = test_state(DTYPE_Q4);
        let mut round = test_round();
        let mut push = test_push(1);
        push.base_version = 6;

        let err = admit_push(&mut round, push, &st).unwrap_err().to_string();
        assert!(err.contains("stale base version"));
        assert!(round.pushes.is_empty());
    }

    #[test]
    fn q4_reconstruction_overflow_is_rejected_before_quorum() {
        let mut st = test_state(DTYPE_Q4);
        st.params[0][0] = f32::MAX;
        let mut round = test_round();
        let mut push = test_push(1);
        push.values[0] = f32::MAX;

        let err = admit_push(&mut round, push, &st).unwrap_err().to_string();
        assert!(err.contains("overflows during reconstruction"));
        assert!(round.pushes.is_empty());
    }

    #[test]
    fn stale_full_tensor_candidate_remains_admissible() {
        let st = test_state(DTYPE_F32);
        let mut round = test_round();
        let mut push = test_push(1);
        push.base_version = 6;

        admit_push(&mut round, push, &st).unwrap();
        assert_eq!(round.pushes.len(), 1);
    }

    #[test]
    fn duplicate_push_is_rejected_without_replacement() {
        let st = test_state(DTYPE_F32);
        let mut round = test_round();
        let first = test_push(2);
        admit_push(&mut round, first, &st).unwrap();

        let mut duplicate = test_push(2);
        duplicate.values = vec![9.0, 9.0];
        let err = admit_push(&mut round, duplicate, &st)
            .unwrap_err()
            .to_string();

        assert!(err.contains("duplicate push"));
        assert_eq!(round.pushes.len(), 1);
        assert_eq!(round.pushes.get(&2).unwrap().values, vec![1.0, 2.0]);
    }

    #[test]
    fn admitted_candidates_are_sorted_by_learner_id() {
        let st = test_state(DTYPE_F32);
        let mut round = test_round();
        for learner_id in [10, 2, 7] {
            admit_push(&mut round, test_push(learner_id), &st).unwrap();
        }
        assert_eq!(sorted_push_ids(&round.pushes), vec![2, 7, 10]);
    }

    #[test]
    fn stale_connection_cannot_replace_or_disconnect_current_session() {
        let registry: Registry = Arc::new(Mutex::new(HashMap::new()));
        let stale = test_group(3, true);
        let current = test_group(3, true);
        registry.lock().unwrap().insert(3, stale.clone());
        registry.lock().unwrap().insert(3, current.clone());

        assert!(!group_is_current(&registry, &stale));
        assert!(group_is_current(&registry, &current));
        assert!(!remove_group_if_current(&registry, &stale));
        assert!(group_is_current(&registry, &current));
        assert!(remove_group_if_current(&registry, &current));
        assert!(!group_is_current(&registry, &current));
    }

    #[test]
    fn stale_session_push_is_pruned_before_quorum_counting() {
        let st = test_state(DTYPE_F32);
        let mut round = test_round();
        let now = Instant::now();
        round.grace_deadline = Some(now + Duration::from_millis(1));
        admit_push(&mut round, test_push(3), &st).unwrap();
        assert_eq!(round.pushes.len(), 1);

        let current_connections = HashMap::from([(3, 2)]);
        prune_noncurrent_pushes(
            std::slice::from_mut(&mut round),
            &current_connections,
            1,
            now,
            Duration::from_secs(3),
        );

        assert!(round.pushes.is_empty());
        assert_eq!(round.grace_deadline, None);
        assert_eq!(round.quorum_deadline, now + Duration::from_secs(3));
    }

    #[test]
    fn delayed_disconnect_does_not_remove_replacement_sessions_push() {
        let st = test_state(DTYPE_F32);
        let mut round = test_round();
        let mut current_push = test_push(3);
        current_push.connection_id = 2;
        admit_push(&mut round, current_push, &st).unwrap();
        let now = Instant::now();

        remove_connection_pushes(
            std::slice::from_mut(&mut round),
            3,
            1,
            1,
            now,
            Duration::from_secs(3),
        );
        assert_eq!(round.pushes.len(), 1);

        remove_connection_pushes(
            std::slice::from_mut(&mut round),
            3,
            2,
            1,
            now,
            Duration::from_secs(3),
        );
        assert!(round.pushes.is_empty());
    }

    #[test]
    fn unvalidated_connection_does_not_count_as_connected() {
        let registry: Registry = Arc::new(Mutex::new(HashMap::new()));
        let group = test_group(3, false);
        registry.lock().unwrap().insert(3, group.clone());
        assert_eq!(validated_group_count(&registry), 0);

        group.validated.store(true, Ordering::Release);
        assert_eq!(validated_group_count(&registry), 1);
    }

    #[test]
    fn retry_clears_expired_grace_deadline() {
        let mut round = test_round();
        let now = Instant::now();
        round.grace_deadline = Some(now - Duration::from_millis(1));

        reset_round_wait(&mut round, now, Duration::from_secs(3));

        assert_eq!(round.grace_deadline, None);
        assert_eq!(round.quorum_deadline, now + Duration::from_secs(3));
    }

    #[test]
    fn event_tape_line_preserves_old_fields_and_adds_outer_stats() {
        let mut pushes = HashMap::new();
        pushes.insert(
            4,
            Push {
                learner_id: 4,
                connection_id: 1,
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
        let decision = CommitDecision::token_weighted();
        let line = format_tape_line(9, 2, &pushes, &stats, &decision, &HashMap::new(), 17);

        assert!(
            line.starts_with("{\"step\":9,\"fragment\":2,\"gnorm\":2.5,\"ms\":17,\"responders\":[")
        );
        assert!(line.contains(
            "{\"id\":4,\"base_version\":7,\"c_steps\":10,\"c_tokens\":100,\"weight\":1000}"
        ));
        // Without instrumentation the responder object carries no drift fields.
        assert!(!line.contains("anchor_drift_norm"));
        assert!(line.contains("\"outer_step_norm\":0.75"));
        assert!(line.contains("\"outer_direction_cosine\":-0.25"));
        assert!(line.contains("\"outer_history_current_ratio\":3"));
        assert!(line.contains("\"outer_restarted\":true"));
        assert!(line.contains("\"policy\":\"token_weighted\""));
        assert!(line.contains("\"selected_action\":\"A0\""));
        assert!(line.contains("\"committed_action\":\"A0\""));
        assert!(line.contains("\"selected_multiplier\":1"));
        assert!(line.contains("\"committed_multiplier\":1"));
        assert!(line.contains("\"fallback\":false"));
        assert!(line.contains("\"probe_latency_ms\":null"));
        assert!(line.contains("\"selected_mass\":1"));
        assert!(line.contains("\"norm_scale\":1"));
        assert!(line.contains("\"step_ratio\":1"));
        assert!(line.contains("\"request_digest\":null"));
        assert!(line.ends_with("}\n"));
    }

    /// v3 arm B: the tape gains an `outer_bias_correction` field ONLY when
    /// the correction is active; the default schema stays byte-free of it.
    #[test]
    fn event_tape_appends_bias_correction_only_when_active() {
        let decision = CommitDecision::token_weighted();
        let off = merge_stats(1.0, 0.5, None, None, false);
        let line = format_tape_line(3, 0, &HashMap::new(), &off, &decision, &HashMap::new(), 1);
        assert!(!line.contains("outer_bias_correction"));

        let mut on = merge_stats(1.0, 0.5, None, None, false);
        on.outer_bias_correction = Some(1.25);
        let line = format_tape_line(3, 0, &HashMap::new(), &on, &decision, &HashMap::new(), 1);
        assert!(line.contains("\"outer_bias_correction\":1.25"));
        assert!(line.ends_with("}\n"));
    }

    #[test]
    fn event_tape_appends_controller_diagnostics_only_when_active() {
        let decision = CommitDecision::token_weighted();
        let off = merge_stats(1.0, 0.5, None, None, false);
        let line = format_tape_line(3, 0, &HashMap::new(), &off, &decision, &HashMap::new(), 1);
        assert!(!line.contains("outer_lr_controller_mode"));

        let mut on = off;
        on.outer_lr_controller = Some(crate::state::OuterLrControllerStats {
            mode: crate::outer_lr_controller::ControllerMode::MeasuredDrift,
            scale: 1.5,
            transient_scale: Some(2.0),
            drift_scale: Some(0.75),
        });
        let line = format_tape_line(3, 0, &HashMap::new(), &on, &decision, &HashMap::new(), 1);
        assert!(line.contains("\"outer_lr_controller_mode\":\"measured-drift\""));
        assert!(line.contains("\"outer_lr_scale\":1.5"));
        assert!(line.contains("\"outer_lr_transient_scale\":2"));
        assert!(line.contains("\"outer_lr_drift_scale\":0.75"));
    }

    #[test]
    fn event_tape_uses_null_for_undefined_outer_ratios() {
        let stats = merge_stats(0.0, 0.0, None, None, false);
        let decision = CommitDecision::probe_fallback(CommitPolicy::ProbeLooV1, "probe_timeout");
        let line = format_tape_line(1, 0, &HashMap::new(), &stats, &decision, &HashMap::new(), 0);
        assert!(line.contains("\"gnorm\":0"));
        assert!(line.contains("\"outer_step_norm\":0"));
        assert!(line.contains("\"outer_direction_cosine\":null"));
        assert!(line.contains("\"outer_history_current_ratio\":null"));
        assert!(line.contains("\"outer_restarted\":false"));
        assert!(line.contains("\"policy\":\"probe_loo_v1\""));
        assert!(line.contains("\"fallback\":true"));
        assert!(line.contains("\"fallback_reason\":\"probe_timeout\""));
        assert!(!line.contains("NaN"));
    }

    #[test]
    fn cttn_shadow_schedule_is_stratified_and_alignment_is_magnitude_aware() {
        let tracker = CttnShadowTracker::new(320, 4, 32).unwrap();
        assert_eq!(tracker.sample_steps.len(), 32);
        for fragment in 0..4 {
            let mut local: Vec<u64> = tracker
                .sample_steps
                .iter()
                .copied()
                .filter(|step| (*step - 1) % 4 == fragment as u64)
                .collect();
            local.sort_unstable();
            assert_eq!(local.len(), 8);
            assert!(local[0] < 40);
            assert!(local[7] > 280);
            assert!(local[7] + 4 <= 320);
        }
        assert_eq!(predictive_alignment(&[1.0, 0.0], 2.0, &[3.0, 4.0]), Some(0.3));
        assert_eq!(predictive_alignment(&[2.0, 0.0], 2.0, &[3.0, 4.0]), Some(0.6));
        assert_eq!(predictive_alignment(&[1.0], 2.0, &[0.0]), None);
    }

    // ---- EXP2.46 anchor-drift diagnostics / version-matched anchoring ------

    /// One f32 fragment of 4, lr 1 mu 0 (plain SGD), with retention on and a
    /// single prior global retained: base global v0 = [0;4], current global
    /// v1 = [2;4] (a learner pushing [-2;4] moved it via the SGD step).
    fn anchor_drift_state(version_matched: bool) -> GlobalState {
        let layout = Layout {
            fragments: vec![crate::state::FragmentInfo {
                merge_mode: crate::state::MERGE_AVG,
                tensor_numels: vec![4],
                tensor_shapes: None,
            }],
        };
        let mut st = GlobalState::new(layout, None, 1.0, 0.0, DTYPE_F32);
        st.anchor_drift_instrument = true;
        st.version_matched_anchor = version_matched;
        st.init_fragment(0, vec![0.0; 4]).unwrap();
        // Θ − (Θ − θ) = θ; a learner at [2;4] moves the global to [2;4] and
        // install retains the prior global (version 0, [0;4]).
        st.merge_and_step(0, &[&vec![2.0f32; 4]], &[1.0]).unwrap();
        st.versions[0] = 1;
        assert_eq!(st.params[0], vec![2.0f32; 4]);
        assert_eq!(st.anchor_at(0, 0).unwrap(), &[0.0f32; 4]);
        st
    }

    #[test]
    fn anchor_drift_measures_norms_and_ratio_against_learner_base() {
        let st = anchor_drift_state(false);
        // Learner trained from base v0 = [0;4] and uploaded [3;4].
        let mut push = test_push(1);
        push.base_version = 0;
        push.values = vec![3.0f32; 4];
        let d = compute_anchor_drift(&st, 0, &push);
        assert!(d.base_resolved);
        // drift = current − base = [2;4] → 2·sqrt(4) = 4; local = [3;4] → 6.
        assert!((d.drift_norm - 4.0).abs() < 1e-9);
        assert!((d.local_delta_norm - 6.0).abs() < 1e-9);
        assert!((d.ratio.unwrap() - 4.0 / 6.0).abs() < 1e-9);
    }

    #[test]
    fn anchor_drift_is_zero_for_current_base() {
        let st = anchor_drift_state(false);
        let mut push = test_push(1);
        push.base_version = 1; // the current version → no drift
        push.values = vec![5.0f32; 4];
        let d = compute_anchor_drift(&st, 0, &push);
        assert!(d.base_resolved);
        assert!(d.drift_norm.abs() < 1e-12);
        assert!(d.momentum_cos.is_none()); // zero drift → undefined cosine
    }

    #[test]
    fn anchor_drift_reports_unresolved_when_base_aged_out() {
        let st = anchor_drift_state(false);
        let mut push = test_push(1);
        push.base_version = 99; // neither current nor retained
        let d = compute_anchor_drift(&st, 0, &push);
        assert!(!d.base_resolved);
        assert!(d.ratio.is_none());
    }

    #[test]
    fn version_matched_reanchor_yields_base_anchored_delta() {
        // The re-anchoring identity used in complete_round: after
        // values' = values + (current − base), the current-anchor merge delta
        // (current − values') equals the version-matched delta (base − values).
        let st = anchor_drift_state(true);
        let base = st.anchor_at(0, 0).unwrap().to_vec();
        let current = st.params[0].clone();
        let original = vec![3.0f32; 4];
        let mut values = original.clone();
        for (v, (c, b)) in values.iter_mut().zip(current.iter().zip(&base)) {
            *v += *c - *b;
        }
        for i in 0..4 {
            let current_anchor_delta = current[i] - values[i];
            let version_matched_delta = base[i] - original[i];
            assert!((current_anchor_delta - version_matched_delta).abs() < 1e-9);
        }
    }

    #[test]
    fn event_tape_emits_anchor_drift_fields_when_instrumented() {
        let mut pushes = HashMap::new();
        pushes.insert(1, test_push(1));
        let mut drift = HashMap::new();
        drift.insert(
            1u32,
            AnchorDrift {
                drift_norm: 4.0,
                local_delta_norm: 6.0,
                ratio: Some(4.0 / 6.0),
                momentum_cos: Some(-0.5),
                base_resolved: true,
            },
        );
        let stats = merge_stats(1.0, 1.0, None, None, false);
        let decision = CommitDecision::token_weighted();
        let line = format_tape_line(9, 0, &pushes, &stats, &decision, &drift, 0);
        assert!(line.contains("\"anchor_drift_norm\":4"));
        assert!(line.contains("\"local_delta_norm\":6"));
        assert!(line.contains("\"anchor_drift_momentum_cos\":-0.5"));
        assert!(line.contains("\"anchor_base_resolved\":true"));
        assert!(!line.contains("NaN"));
    }

    #[test]
    fn scalar_shadow_commits_a0_while_active_commits_the_selected_preview() {
        assert_eq!(commit_preview_index(CommitPolicy::ProbeLrShadow, 4), 0);
        assert_eq!(commit_preview_index(CommitPolicy::ProbeLrV1, 4), 4);
        assert_eq!(commit_preview_index(CommitPolicy::ProbeShadow, 3), 0);
        assert_eq!(commit_preview_index(CommitPolicy::ProbeLooV1, 3), 3);
    }

    #[test]
    fn event_tape_records_distinct_scalar_selection_and_commit_multipliers() {
        let decision = CommitDecision {
            policy: CommitPolicy::ProbeLrShadow,
            selected_action: "A1".to_owned(),
            committed_action: "A0".to_owned(),
            fallback: false,
            fallback_reason: None,
            probe_latency_ms: Some(3.0),
            selected_mass: 1.0,
            norm_scale: 0.75,
            step_ratio: 0.75,
            selected_multiplier: 0.75,
            committed_multiplier: 1.0,
            request_digest: Some("a".repeat(64)),
            cttn_diagnostics: None,
            cttn_shadow_resolved: None,
        };
        let line = format_tape_line(
            1,
            0,
            &HashMap::new(),
            &merge_stats(1.0, 1.0, None, None, false),
            &decision,
            &HashMap::new(),
            4,
        );
        assert!(line.contains("\"selected_multiplier\":0.75"));
        assert!(line.contains("\"committed_multiplier\":1"));
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
