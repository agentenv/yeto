//! Task dispenser of the RL syncer (S27 P2): which island trains which rows.
//!
//! The dispenser hands out *positions* of one global row sequence. Position
//! `p` is row `order(p / rows)[p % rows]` of `--data`, where `rows` is the
//! row count and `order(pass)` the island-side pass order (pass 0: data
//! order; pass `k >= 1`: a reshuffle seeded by the run seed and `k`,
//! `yeto.rl.engine.data_split.pass_order`). The order does not depend on the
//! island count, so islands may join, leave and restart at any time.
//!
//! A *holder* is one island process (`learner_id`, random 64-bit
//! `incarnation`). It claims a *block* (one or more position spans) per
//! round, consumes it, and reports every position of it as trained,
//! dropped (consumed, not trained), requeued (a stale group: trained again,
//! at most `max_redo` times) or released (never consumed). A block whose
//! positions are all reported is retired, so the state grows with the
//! blocks in flight, never with the row count.
//!
//! Failure semantics (docs/PROTOCOL.md "Task dispenser"):
//! - every position is held by at most one live holder at a time;
//! - a holder that sends nothing for `lease` expires, and a new incarnation
//!   of the same island supersedes the old one at once: in both cases the
//!   holder is retired (fenced: its later requests are refused) and the
//!   positions it did not report go back to the FRONT of the queue, so the
//!   next claim of any island takes them first;
//! - a restart from the checkpoint (`restore_persisted`) retires every holder
//!   of the snapshot and puts their unreported positions back the same way.
//!
//! So a position is generated at least once and reported trained at most
//! once (an island reports a position when its group enters a training
//! batch). Islands without a dispenser never see any of this.
//!
//! Streams (S27 P3, `--data-mix` and `--rl-island-task-labels`): the rows
//! may be split into *streams* (one per source data set and task label). A
//! position then carries its stream in its high bits ([`STREAM_SHIFT`]):
//! position `(s << 40) | k` is the `k`-th position of stream `s`, i.e. row
//! `order_s(k / rows_s)[k % rows_s]` of that stream's rows. Fresh positions
//! are interleaved across the streams by their integer weights with
//! Tijdeman's earliest-deadline rule: when every claiming holder accepts
//! every stream, after any number `n` of fresh positions stream `s` has
//! given out within one position of `n * w_s / sum(w)` (a bound of
//! `1 - 1 / (2k - 2)` for `k` streams). A holder may accept only some
//! streams (task label affinity); it then gets only their positions, and the
//! mix of the positions given out follows the claims. One stream with
//! weight 1 and no name is exactly the P2 dispenser (positions are row
//! sequence numbers).
//!
//! Pure state machine like `membership.rs`: no sockets, no files; every
//! clock reading is passed in. Events for the `task-dispense.jsonl` log are
//! collected in `take_events`.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::time::{Duration, Instant};

use anyhow::{ensure, Result};

/// Magic and version of the persisted dispenser record (YDS1: the P2 record,
/// one stream; still restored).
const PERSIST_MAGIC_V1: &[u8; 4] = b"YDS1";
const PERSIST_MAGIC: &[u8; 4] = b"YDS2";
/// A position's stream is its bits from here up (module doc).
pub const STREAM_SHIFT: u32 = 40;
/// Positions per stream (the local part of a position).
pub const STREAM_POSITIONS: u64 = 1 << STREAM_SHIFT;
/// Streams one run may declare.
pub const MAX_STREAMS: usize = 256;
/// Bytes of one stream name (UTF-8, no control characters).
pub const MAX_STREAM_NAME: usize = 64;
/// Largest stream weight (keeps the interleave arithmetic inside i128).
pub const MAX_STREAM_WEIGHT: u64 = 1 << 62;

/// The stream of a position.
pub fn stream_of(position: u64) -> usize {
    (position >> STREAM_SHIFT) as usize
}

/// The position number inside its stream.
pub fn local_of(position: u64) -> u64 {
    position & (STREAM_POSITIONS - 1)
}

/// The position `local` of stream `stream`.
pub fn position_of(stream: usize, local: u64) -> u64 {
    ((stream as u64) << STREAM_SHIFT) | local
}
/// Retired incarnations remembered (fencing). Incarnations are random
/// 64-bit values; this only has to cover realistic restart storms.
const RETIRED_LIMIT: usize = 4096;
/// Live holders admitted at once (one per island process).
const LIVE_LIMIT: usize = 4096;
/// Hint for a WAIT answer: when to claim again.
pub const WAIT_RETRY: Duration = Duration::from_millis(1000);

pub const DEFAULT_LEASE: Duration = Duration::from_secs(300);
pub const DEFAULT_MAX_REDO: u32 = 1;
pub const DEFAULT_MAX_BLOCK: u32 = 1 << 16;

/// Start of a refusal that means "this incarnation is fenced": the client
/// drops its blocks and opens a new incarnation (`yeto.rl.engine.task_dispense`).
pub const FENCED_PREFIX: &str = "fenced: dispense";
/// Start of a refusal that means the island's data differs from the run's.
pub const MISMATCH_PREFIX: &str = "dispense mismatch:";
/// Start of a refusal of a malformed or impossible request.
pub const REFUSED_PREFIX: &str = "dispense refused:";

#[derive(Clone, Debug, PartialEq)]
pub struct DispenseConfig {
    /// A holder silent for this long expires; its unreported positions return.
    pub lease: Duration,
    /// Passes over the rows; None: unlimited (every pass after the first is
    /// reshuffled on the island side). Some(n): END after n passes.
    pub passes: Option<u64>,
    /// How often one position may be requeued (stale group trained again).
    pub max_redo: u32,
    /// Largest block one claim may ask for.
    pub max_block: u32,
}

impl Default for DispenseConfig {
    fn default() -> Self {
        Self { lease: DEFAULT_LEASE, passes: None, max_redo: DEFAULT_MAX_REDO, max_block: DEFAULT_MAX_BLOCK }
    }
}

impl DispenseConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(!self.lease.is_zero(), "--dispense-lease-s must be positive");
        ensure!(self.passes != Some(0), "--dispense-passes must be >= 1 (omit it for unlimited passes)");
        ensure!(self.max_block >= 1, "--dispense-max-block must be >= 1");
        Ok(())
    }
}

/// One island process.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Holder {
    pub learner_id: u32,
    pub incarnation: u64,
}

/// One stream of the run's rows (module doc): its row count, its integer
/// weight in the interleave of fresh positions, and its name (the source
/// data set and task label, for the log; "" for the single P2 stream).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StreamDecl {
    pub rows: u64,
    pub weight: u64,
    pub name: String,
}

/// What an island declares about its data: every island of a run must
/// declare the same streams and fingerprint.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DataDecl {
    pub streams: Vec<StreamDecl>,
    pub fingerprint: [u8; 32],
}

impl DataDecl {
    /// The P2 data: one stream of `rows` rows.
    pub fn single(rows: u64, fingerprint: [u8; 32]) -> Self {
        Self { streams: vec![StreamDecl { rows, weight: 1, name: String::new() }], fingerprint }
    }


    pub fn total_rows(&self) -> u64 {
        self.streams.iter().map(|s| s.rows).sum()
    }

    fn validate(&self) -> std::result::Result<(), String> {
        if self.streams.is_empty() || self.streams.len() > MAX_STREAMS {
            return Err(format!("{} streams (1..={MAX_STREAMS} allowed)", self.streams.len()));
        }
        for (i, s) in self.streams.iter().enumerate() {
            if s.rows == 0 || s.rows >= STREAM_POSITIONS {
                return Err(format!("stream {i} declares {} rows", s.rows));
            }
            if s.weight == 0 || s.weight > MAX_STREAM_WEIGHT {
                return Err(format!("stream {i} has weight {} (1..=2^62 allowed)", s.weight));
            }
            if s.name.len() > MAX_STREAM_NAME || s.name.chars().any(char::is_control) {
                return Err(format!("stream {i} has a bad name ({} bytes)", s.name.len()));
            }
        }
        Ok(())
    }
}

/// Positions `[start, end)`, each requeued `redo` times so far.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Span {
    pub start: u64,
    pub end: u64,
    pub redo: u32,
}

impl Span {
    pub fn len(&self) -> u64 {
        self.end - self.start
    }
}

/// What an island reports about positions of one of its blocks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReportKind {
    /// The position's group entered a training batch.
    Trained = 1,
    /// Consumed without training (e.g. a filtered group, a row the island cannot load).
    Dropped = 2,
    /// A stale group: give the position out again (at most `max_redo` times).
    Requeue = 3,
    /// Never consumed: give it out again (does not count as a redo).
    Release = 4,
}

impl ReportKind {
    pub fn from_u8(value: u8) -> Option<Self> {
        Some(match value {
            1 => Self::Trained,
            2 => Self::Dropped,
            3 => Self::Requeue,
            4 => Self::Release,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Trained => "trained",
            Self::Dropped => "dropped",
            Self::Requeue => "requeue",
            Self::Release => "release",
        }
    }
}

/// One entry of a report: `kind` for the `ranges` of `block`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReportEntry {
    pub block: u64,
    pub kind: ReportKind,
    pub ranges: Vec<(u64, u64)>,
}

/// The answer to a claim.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Claim {
    Grant { block: u64, ranges: Vec<(u64, u64)> },
    /// Nothing to give now (every remaining position is held by a live
    /// island and may come back); claim again after `WAIT_RETRY`.
    Wait,
    /// Every pass is given out and every position reported: the data ended.
    End,
}

/// Totals of one report.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ReportTotals {
    pub settled: u64,
    pub requeued: u64,
    /// Requeues refused because the position reached `max_redo` (settled).
    pub exhausted: u64,
}

#[derive(Debug, Eq, PartialEq)]
pub enum OpenOutcome {
    /// A new holder; `superseded` is the island's previous incarnation, if one was live.
    Joined { superseded: Option<u64>, reclaimed: u64 },
    /// The same incarnation reconnected (it keeps its blocks).
    Reconnected,
}

#[derive(Debug, Eq, PartialEq)]
pub enum DispenseError {
    Fenced(String),
    Mismatch(String),
    Refused(String),
}

impl std::fmt::Display for DispenseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fenced(m) => write!(f, "{FENCED_PREFIX} {m}"),
            Self::Mismatch(m) => write!(f, "{MISMATCH_PREFIX} {m}"),
            Self::Refused(m) => write!(f, "{REFUSED_PREFIX} {m}"),
        }
    }
}

impl std::error::Error for DispenseError {}

type DResult<T> = std::result::Result<T, DispenseError>;

/// Why a holder's unreported positions went back to the queue.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReclaimReason {
    Expired,
    Superseded,
}

impl ReclaimReason {
    fn name(self) -> &'static str {
        match self {
            Self::Expired => "lease_expired",
            Self::Superseded => "superseded",
        }
    }
}

/// One line of `task-dispense.jsonl` (without time and syncer epoch).
#[derive(Clone, Debug, PartialEq)]
pub enum DispenseEvent {
    Open { holder: Holder, superseded: Option<u64>, reconnect: bool, rows: u64, accept: Option<Vec<usize>> },
    /// The run's streams, once (the first island fixed them, or a restore).
    Streams { streams: Vec<StreamDecl>, fingerprint: [u8; 32] },
    Grant { holder: Holder, block: u64, ranges: Vec<(u64, u64)>, first_pass: u64, last_pass: u64 },
    PassStart { stream: usize, pass: u64 },
    Wait { holder: Holder },
    End { holder: Holder },
    Report { holder: Holder, block: u64, kind: ReportKind, ranges: Vec<(u64, u64)>, exhausted: Vec<(u64, u64)> },
    BlockDone { holder: Holder, block: u64 },
    Reclaim { holder: Holder, reason: ReclaimReason, ranges: Vec<(u64, u64)> },
    Fenced { holder: Holder, request: &'static str },
    Restore { next: u64, queued: u64, retired: usize },
    /// The record was captured for a checkpoint (the cut a restart returns to).
    Checkpoint { next: u64, queued: u64, outstanding: u64 },
}

fn ranges_json(ranges: &[(u64, u64)]) -> String {
    let items: Vec<String> = ranges.iter().map(|(a, b)| format!("[{a},{b}]")).collect();
    format!("[{}]", items.join(","))
}

fn count(ranges: &[(u64, u64)]) -> u64 {
    ranges.iter().map(|(a, b)| b - a).sum()
}

/// A JSON string literal (names are UTF-8 without control characters).
fn json_str(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

impl DispenseEvent {
    /// The JSON fields of this event (no braces), `"kind"` first.
    pub fn json_fields(&self) -> String {
        let h = |h: &Holder| format!("\"island_id\":{},\"incarnation\":\"{:016x}\"", h.learner_id, h.incarnation);
        match self {
            Self::Open { holder, superseded, reconnect, rows, accept } => format!(
                "\"kind\":\"dispense_open\",{},\"rows\":{rows},\"reconnect\":{reconnect},\"superseded\":{},\"accept\":{}",
                h(holder),
                superseded.map_or("null".to_string(), |s| format!("\"{s:016x}\"")),
                accept.as_ref().map_or("null".to_string(), |a| format!(
                    "[{}]",
                    a.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(",")
                ))
            ),
            Self::Streams { streams, fingerprint } => {
                let items: Vec<String> = streams
                    .iter()
                    .enumerate()
                    .map(|(i, s)| {
                        format!("{{\"stream\":{i},\"name\":{},\"rows\":{},\"weight\":{}}}", json_str(&s.name), s.rows, s.weight)
                    })
                    .collect();
                format!(
                    "\"kind\":\"dispense_streams\",\"stream_shift\":{STREAM_SHIFT},\"fingerprint\":\"{}\",\"streams\":[{}]",
                    hex8(fingerprint),
                    items.join(",")
                )
            }
            Self::Grant { holder, block, ranges, first_pass, last_pass } => format!(
                "\"kind\":\"dispense_grant\",{},\"block\":{block},\"positions\":{},\"ranges\":{},\"first_pass\":{first_pass},\"last_pass\":{last_pass}",
                h(holder),
                count(ranges),
                ranges_json(ranges)
            ),
            Self::PassStart { stream, pass } => {
                format!("\"kind\":\"dispense_pass_start\",\"stream\":{stream},\"pass\":{pass}")
            }
            Self::Wait { holder } => format!("\"kind\":\"dispense_wait\",{}", h(holder)),
            Self::End { holder } => format!("\"kind\":\"dispense_end\",{}", h(holder)),
            Self::Report { holder, block, kind, ranges, exhausted } => format!(
                "\"kind\":\"dispense_report\",{},\"block\":{block},\"report\":\"{}\",\"positions\":{},\"ranges\":{},\"redo_exhausted\":{}",
                h(holder),
                kind.name(),
                count(ranges),
                ranges_json(ranges),
                ranges_json(exhausted)
            ),
            Self::BlockDone { holder, block } => {
                format!("\"kind\":\"dispense_block_done\",{},\"block\":{block}", h(holder))
            }
            Self::Reclaim { holder, reason, ranges } => format!(
                "\"kind\":\"dispense_reclaim\",{},\"reason\":\"{}\",\"positions\":{},\"ranges\":{}",
                h(holder),
                reason.name(),
                count(ranges),
                ranges_json(ranges)
            ),
            Self::Fenced { holder, request } => {
                format!("\"kind\":\"dispense_fenced\",{},\"request\":\"{request}\"", h(holder))
            }
            Self::Restore { next, queued, retired } => format!(
                "\"kind\":\"dispense_restore\",\"next\":{next},\"queued\":{queued},\"retired\":{retired}"
            ),
            Self::Checkpoint { next, queued, outstanding } => format!(
                "\"kind\":\"dispense_checkpoint\",\"next\":{next},\"queued\":{queued},\"outstanding\":{outstanding}"
            ),
        }
    }
}

#[derive(Debug, Clone)]
struct Block {
    holder: Holder,
    /// Unreported positions: start -> (end, redo); disjoint, never empty spans.
    open: BTreeMap<u64, (u64, u32)>,
}

/// The answer to the holder's latest CLAIM or REPORT, kept so that a
/// request resent after a lost answer (same `seq`) gets the same answer
/// instead of being applied twice.
#[derive(Clone, Debug, Eq, PartialEq)]
enum Answer {
    Claim(Claim),
    Report(ReportTotals),
}

#[derive(Debug)]
struct Live {
    last_seen: Instant,
    blocks: BTreeSet<u64>,
    /// A WAIT was logged since the last grant (one log line per wait period).
    waiting: bool,
    /// Sequence number and answer of the latest CLAIM / REPORT.
    last: Option<(u64, Answer)>,
    /// The streams this holder takes positions of, ascending (every stream
    /// unless the island declared its task labels).
    accept: Vec<usize>,
}

/// Cumulative counters (positions).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Totals {
    pub granted: u64,
    pub trained: u64,
    pub dropped: u64,
    pub requeued: u64,
    pub exhausted: u64,
    pub released: u64,
    pub reclaimed: u64,
}

/// The give-out state of one stream.
#[derive(Debug, Default)]
struct StreamState {
    /// Next never-given local position.
    next: u64,
    /// Positions of this stream to give out before fresh ones, front first.
    queue: VecDeque<Span>,
    /// Passes of this stream whose first fresh position was given out.
    passes_started: u64,
}

#[derive(Debug)]
pub struct Dispenser {
    cfg: DispenseConfig,
    /// Streams and data fingerprint, fixed by the first island (or the
    /// restored record); every later island must declare the same.
    data: Option<DataDecl>,
    /// One per declared stream (empty until the data is fixed).
    streams: Vec<StreamState>,
    blocks: BTreeMap<u64, Block>,
    next_block: u64,
    live: BTreeMap<Holder, Live>,
    /// Live incarnation per island.
    current: HashMap<u32, u64>,
    retired: (VecDeque<Holder>, std::collections::HashSet<Holder>),
    pub totals: Totals,
    events: Vec<DispenseEvent>,
}

impl Dispenser {
    pub fn new(cfg: DispenseConfig) -> Self {
        Self {
            cfg,
            data: None,
            streams: Vec::new(),
            blocks: BTreeMap::new(),
            next_block: 1,
            live: BTreeMap::new(),
            current: HashMap::new(),
            retired: (VecDeque::new(), Default::default()),
            totals: Totals::default(),
            events: Vec::new(),
        }
    }

    pub fn config(&self) -> &DispenseConfig {
        &self.cfg
    }

    /// Rows of all streams (None until the data is fixed).
    #[cfg_attr(not(test), allow(dead_code))] // tests and offline tools
    pub fn rows(&self) -> Option<u64> {
        self.data.as_ref().map(DataDecl::total_rows)
    }

    /// Fresh positions given out so far (all streams).
    pub fn next_position(&self) -> u64 {
        self.streams.iter().map(|s| s.next).sum()
    }

    pub fn take_events(&mut self) -> Vec<DispenseEvent> {
        std::mem::take(&mut self.events)
    }

    #[cfg_attr(not(test), allow(dead_code))] // tests and offline tools
    pub fn is_live(&self, holder: Holder) -> bool {
        self.live.contains_key(&holder)
    }

    pub fn is_retired(&self, holder: Holder) -> bool {
        self.retired.1.contains(&holder)
    }

    /// Positions waiting in the queues (returned, requeued, released).
    pub fn queued(&self) -> u64 {
        self.streams.iter().flat_map(|s| s.queue.iter()).map(Span::len).sum()
    }

    /// Positions held by live holders and not reported yet.
    pub fn outstanding(&self) -> u64 {
        self.blocks.values().flat_map(|b| b.open.iter().map(|(s, (e, _))| e - s)).sum()
    }

    /// Blocks in flight (granted, not fully reported).
    #[cfg_attr(not(test), allow(dead_code))] // tests and offline tools
    pub fn blocks_in_flight(&self) -> usize {
        self.blocks.len()
    }

    fn stream_rows(&self, stream: usize) -> u64 {
        self.data.as_ref().expect("an open holder fixed the data").streams[stream].rows
    }

    /// Exclusive end of the fresh local positions of `stream`.
    fn limit(&self, stream: usize) -> u64 {
        let cap = STREAM_POSITIONS;
        match self.cfg.passes {
            Some(p) => self.stream_rows(stream).saturating_mul(p).min(cap),
            None => cap,
        }
    }

    fn exhausted(&self, stream: usize) -> bool {
        self.streams[stream].next >= self.limit(stream)
    }

    fn retire(&mut self, holder: Holder) {
        let (order, set) = &mut self.retired;
        if set.insert(holder) {
            order.push_back(holder);
            while order.len() > RETIRED_LIMIT {
                if let Some(old) = order.pop_front() {
                    set.remove(&old);
                }
            }
        }
    }

    /// Put spans back at the front of their streams' queues, ascending.
    fn push_front(&mut self, mut spans: Vec<Span>) {
        spans.sort_by_key(|s| s.start);
        for span in spans.into_iter().rev() {
            self.streams[stream_of(span.start)].queue.push_front(span);
        }
    }

    /// Retire `holder` and return its unreported positions to the queue front.
    fn reclaim(&mut self, holder: Holder, reason: ReclaimReason) -> u64 {
        let Some(live) = self.live.remove(&holder) else { return 0 };
        if self.current.get(&holder.learner_id) == Some(&holder.incarnation) {
            self.current.remove(&holder.learner_id);
        }
        self.retire(holder);
        let mut spans = Vec::new();
        for id in live.blocks {
            if let Some(block) = self.blocks.remove(&id) {
                spans.extend(block.open.into_iter().map(|(start, (end, redo))| Span { start, end, redo }));
            }
        }
        let positions: u64 = spans.iter().map(Span::len).sum();
        self.totals.reclaimed += positions;
        let mut ranges: Vec<(u64, u64)> = spans.iter().map(|s| (s.start, s.end)).collect();
        ranges.sort_unstable();
        self.events.push(DispenseEvent::Reclaim { holder, reason, ranges });
        self.push_front(spans);
        positions
    }

    /// Fix the run's data (the first island, or a restore).
    fn fix_data(&mut self, data: DataDecl) {
        self.streams = data.streams.iter().map(|_| StreamState::default()).collect();
        self.events.push(DispenseEvent::Streams { streams: data.streams.clone(), fingerprint: data.fingerprint });
        self.data = Some(data);
    }

    /// Admit an island process whose data is one stream of `rows` rows (P2).
    #[cfg_attr(not(test), allow(dead_code))] // tests and offline tools
    pub fn open(&mut self, holder: Holder, rows: u64, fingerprint: [u8; 32], now: Instant) -> DResult<OpenOutcome> {
        self.open_with(holder, &DataDecl::single(rows, fingerprint), None, now)
    }

    /// Admit an island process. `data` describes its data; `accept` the
    /// streams it takes positions of (None: every stream).
    pub fn open_with(
        &mut self,
        holder: Holder,
        data: &DataDecl,
        accept: Option<&[u16]>,
        now: Instant,
    ) -> DResult<OpenOutcome> {
        if self.is_retired(holder) {
            self.events.push(DispenseEvent::Fenced { holder, request: "open" });
            return Err(DispenseError::Fenced(format!(
                "incarnation {:016x} of island {} is retired (expired, superseded or from before a restart)",
                holder.incarnation, holder.learner_id
            )));
        }
        if data.total_rows() == 0 {
            return Err(DispenseError::Refused("the island declares 0 rows".into()));
        }
        data.validate().map_err(|m| DispenseError::Refused(format!("the island declares {m}")))?;
        match &self.data {
            Some(run) if run != data => {
                return Err(DispenseError::Mismatch(format!(
                    "island {} declares {} rows in {} stream(s) with fingerprint {}, the run has {} rows in {} stream(s) with {}{}",
                    holder.learner_id,
                    data.total_rows(),
                    data.streams.len(),
                    hex8(&data.fingerprint),
                    run.total_rows(),
                    run.streams.len(),
                    hex8(&run.fingerprint),
                    if run.fingerprint == data.fingerprint { " (same rows, other streams or weights)" } else { "" }
                )));
            }
            Some(_) => {}
            None => self.fix_data(data.clone()),
        }
        let streams = self.streams.len();
        let accept_list: Vec<usize> = match accept {
            None => (0..streams).collect(),
            Some(ids) => {
                let mut list: Vec<usize> = ids.iter().map(|&s| usize::from(s)).collect();
                list.sort_unstable();
                list.dedup();
                if list.is_empty() {
                    return Err(DispenseError::Refused("the island accepts no stream".into()));
                }
                if let Some(bad) = list.iter().find(|&&s| s >= streams) {
                    return Err(DispenseError::Refused(format!("the island accepts stream {bad}; the run has {streams}")));
                }
                list
            }
        };
        let logged_accept = (accept_list.len() < streams).then(|| accept_list.clone());
        if let Some(live) = self.live.get_mut(&holder) {
            live.last_seen = now;
            live.accept = accept_list;
            self.events.push(DispenseEvent::Open {
                holder,
                superseded: None,
                reconnect: true,
                rows: data.total_rows(),
                accept: logged_accept,
            });
            return Ok(OpenOutcome::Reconnected);
        }
        if self.live.len() >= LIVE_LIMIT {
            return Err(DispenseError::Refused(format!("more than {LIVE_LIMIT} live islands")));
        }
        let previous = self.current.get(&holder.learner_id).copied();
        let mut reclaimed = 0;
        if let Some(incarnation) = previous {
            reclaimed = self.reclaim(Holder { learner_id: holder.learner_id, incarnation }, ReclaimReason::Superseded);
        }
        self.current.insert(holder.learner_id, holder.incarnation);
        self.live.insert(
            holder,
            Live { last_seen: now, blocks: BTreeSet::new(), waiting: false, last: None, accept: accept_list },
        );
        self.events.push(DispenseEvent::Open {
            holder,
            superseded: previous,
            reconnect: false,
            rows: data.total_rows(),
            accept: logged_accept,
        });
        Ok(OpenOutcome::Joined { superseded: previous, reclaimed })
    }

    fn require_live(&mut self, holder: Holder, request: &'static str, now: Instant) -> DResult<()> {
        match self.live.get_mut(&holder) {
            Some(live) => {
                live.last_seen = now;
                Ok(())
            }
            None => {
                self.events.push(DispenseEvent::Fenced { holder, request });
                Err(DispenseError::Fenced(format!(
                    "incarnation {:016x} of island {} holds no lease",
                    holder.incarnation, holder.learner_id
                )))
            }
        }
    }

    /// Lease renewal without a request.
    pub fn heartbeat(&mut self, holder: Holder, now: Instant) -> DResult<()> {
        self.require_live(holder, "heartbeat", now)
    }

    /// A resent request (`seq` equal to the latest one) gets the stored
    /// answer; an older `seq` is refused.
    fn replay(&self, holder: Holder, seq: u64) -> DResult<Option<Answer>> {
        let live = self.live.get(&holder).expect("checked live");
        match &live.last {
            Some((last, answer)) if *last == seq => Ok(Some(answer.clone())),
            Some((last, _)) if *last > seq => {
                Err(DispenseError::Refused(format!("request {seq} is older than the latest request {last}")))
            }
            _ => Ok(None),
        }
    }

    fn remember(&mut self, holder: Holder, seq: u64, answer: Answer) {
        if let Some(live) = self.live.get_mut(&holder) {
            live.last = Some((seq, answer));
        }
    }

    /// Give `holder` up to `wanted` positions: requeued and returned ones
    /// first (queue front), then fresh ones. `seq` numbers the holder's
    /// CLAIM / REPORT requests (strictly increasing; a resent one repeats it).
    pub fn claim(&mut self, holder: Holder, seq: u64, wanted: u32, now: Instant) -> DResult<Claim> {
        self.require_live(holder, "claim", now)?;
        match self.replay(holder, seq)? {
            Some(Answer::Claim(claim)) => return Ok(claim),
            Some(Answer::Report(_)) => {
                return Err(DispenseError::Refused(format!("request {seq} was a report, not a claim")))
            }
            None => {}
        }
        let claim = self.claim_new(holder, wanted)?;
        self.remember(holder, seq, Answer::Claim(claim.clone()));
        Ok(claim)
    }

    /// How many fresh positions of each accepted stream the next `need`
    /// fresh positions take (module doc: Tijdeman's earliest-deadline rule
    /// over every stream's weight; among the accepted streams that are not
    /// exhausted; when none of them is due, the one furthest behind).
    fn pick_fresh(&self, accept: &[usize], need: u64) -> Vec<(usize, u64)> {
        let open: Vec<usize> = accept.iter().copied().filter(|&s| !self.exhausted(s)).collect();
        if open.is_empty() || need == 0 {
            return Vec::new();
        }
        let data = self.data.as_ref().expect("an open holder fixed the data");
        if self.streams.len() == 1 {
            let s = open[0];
            return vec![(s, need.min(self.limit(s) - self.streams[s].next))];
        }
        let weights: Vec<i128> = data.streams.iter().map(|s| i128::from(s.weight)).collect();
        let total: i128 = weights.iter().sum();
        let m = 2 * (self.streams.len() as i128 - 1);
        let limits: Vec<u64> = (0..self.streams.len()).map(|s| self.limit(s)).collect();
        let mut counts: Vec<u64> = self.streams.iter().map(|s| s.next).collect();
        let mut given: i128 = counts.iter().map(|&c| i128::from(c)).sum();
        let mut taken = vec![0u64; self.streams.len()];
        for _ in 0..need {
            let n = given + 1;
            let mut due: Option<(usize, i128)> = None; // (stream, deadline numerator)
            let mut behind: Option<(usize, i128)> = None; // (stream, deficit)
            for &s in &open {
                if counts[s] >= limits[s] {
                    continue;
                }
                let c = i128::from(counts[s]);
                let deficit = weights[s] * n - c * total; // (n * w_s / total - c_s) * total
                if m * deficit >= total {
                    // eligible: deadline (c + 1 - 1/m) / w_s, compared as a / w_a < b / w_b
                    let a = m * (c + 1) - 1;
                    if due.map_or(true, |(b, bn)| a * weights[b] < bn * weights[s]) {
                        due = Some((s, a));
                    }
                }
                if behind.map_or(true, |(_, d)| deficit > d) {
                    behind = Some((s, deficit));
                }
            }
            let Some((s, _)) = due.or(behind) else { break };
            counts[s] += 1;
            taken[s] += 1;
            given += 1;
        }
        taken.into_iter().enumerate().filter(|&(_, n)| n > 0).collect()
    }

    /// Whether a live holder holds an unreported position of one of `streams`.
    fn holds_any(&self, streams: &[usize]) -> bool {
        self.blocks
            .values()
            .any(|b| b.open.keys().any(|&start| streams.contains(&stream_of(start))))
    }

    fn claim_new(&mut self, holder: Holder, wanted: u32) -> DResult<Claim> {
        if wanted == 0 || wanted > self.cfg.max_block {
            return Err(DispenseError::Refused(format!(
                "claim of {wanted} positions (1..={} allowed)",
                self.cfg.max_block
            )));
        }
        let accept = self.live.get(&holder).expect("checked live").accept.clone();
        let mut need = u64::from(wanted);
        let mut spans: Vec<Span> = Vec::new();
        for &s in &accept {
            let queue = &mut self.streams[s].queue;
            while need > 0 {
                let Some(front) = queue.front_mut() else { break };
                if front.len() <= need {
                    need -= front.len();
                    spans.push(queue.pop_front().unwrap());
                } else {
                    spans.push(Span { start: front.start, end: front.start + need, redo: front.redo });
                    front.start += need;
                    need = 0;
                }
            }
            if need == 0 {
                break;
            }
        }
        for (s, n) in self.pick_fresh(&accept, need) {
            let rows = self.stream_rows(s);
            let state = &mut self.streams[s];
            let (start, end) = (state.next, state.next + n);
            for pass in start / rows..=(end - 1) / rows {
                if pass >= state.passes_started {
                    state.passes_started = pass + 1;
                    self.events.push(DispenseEvent::PassStart { stream: s, pass });
                }
            }
            state.next = end;
            spans.push(Span { start: position_of(s, start), end: position_of(s, end), redo: 0 });
        }
        if spans.is_empty() {
            let ended = accept.iter().all(|&s| self.exhausted(s)) && !self.holds_any(&accept);
            if ended {
                self.events.push(DispenseEvent::End { holder });
                return Ok(Claim::End);
            }
            let live = self.live.get_mut(&holder).expect("checked live");
            if !live.waiting {
                live.waiting = true;
                self.events.push(DispenseEvent::Wait { holder });
            }
            return Ok(Claim::Wait);
        }
        let id = self.next_block;
        self.next_block += 1;
        let mut open = BTreeMap::new();
        for span in &spans {
            open.insert(span.start, (span.end, span.redo));
        }
        let ranges: Vec<(u64, u64)> = spans.iter().map(|s| (s.start, s.end)).collect();
        let pass = |p: u64| local_of(p) / self.stream_rows(stream_of(p));
        let first_pass = ranges.iter().map(|&(a, _)| pass(a)).min().unwrap_or(0);
        let last_pass = ranges.iter().map(|&(_, b)| pass(b - 1)).max().unwrap_or(0);
        let live = self.live.get_mut(&holder).expect("checked live");
        live.waiting = false;
        live.blocks.insert(id);
        self.totals.granted += count(&ranges);
        self.blocks.insert(id, Block { holder, open });
        self.events.push(DispenseEvent::Grant { holder, block: id, ranges: ranges.clone(), first_pass, last_pass });
        Ok(Claim::Grant { block: id, ranges })
    }

    /// Apply a report. Every entry is checked before any state changes: a
    /// range must lie inside the block's unreported positions. `seq` as in
    /// `claim`.
    pub fn report(&mut self, holder: Holder, seq: u64, entries: &[ReportEntry], now: Instant) -> DResult<ReportTotals> {
        self.require_live(holder, "report", now)?;
        match self.replay(holder, seq)? {
            Some(Answer::Report(totals)) => return Ok(totals),
            Some(Answer::Claim(_)) => {
                return Err(DispenseError::Refused(format!("request {seq} was a claim, not a report")))
            }
            None => {}
        }
        let totals = self.report_new(holder, entries)?;
        self.remember(holder, seq, Answer::Report(totals));
        Ok(totals)
    }

    fn report_new(&mut self, holder: Holder, entries: &[ReportEntry]) -> DResult<ReportTotals> {
        // Stage every touched block on a copy, then commit.
        let mut staged: BTreeMap<u64, Block> = BTreeMap::new();
        let mut removed: Vec<(usize, Vec<Span>)> = Vec::new();
        for (index, entry) in entries.iter().enumerate() {
            if !staged.contains_key(&entry.block) {
                let original = self.blocks.get(&entry.block).filter(|b| b.holder == holder).ok_or_else(|| {
                    DispenseError::Refused(format!(
                        "block {} is not held by incarnation {:016x} of island {}",
                        entry.block, holder.incarnation, holder.learner_id
                    ))
                })?;
                staged.insert(entry.block, original.clone());
            }
            let block = staged.get_mut(&entry.block).expect("staged above");
            let mut spans = Vec::new();
            for &(start, end) in &entry.ranges {
                spans.extend(take_range(&mut block.open, start, end).map_err(|m| {
                    DispenseError::Refused(format!("block {}: {m}", entry.block))
                })?);
            }
            removed.push((index, spans));
        }
        let mut totals = ReportTotals::default();
        let mut front = Vec::new();
        for (index, spans) in removed {
            let entry = &entries[index];
            let positions: u64 = spans.iter().map(Span::len).sum();
            totals.settled += positions;
            let mut exhausted = Vec::new();
            match entry.kind {
                ReportKind::Trained => self.totals.trained += positions,
                ReportKind::Dropped => self.totals.dropped += positions,
                ReportKind::Release => {
                    self.totals.released += positions;
                    front.extend(spans);
                }
                ReportKind::Requeue => {
                    for span in spans {
                        if span.redo < self.cfg.max_redo {
                            totals.requeued += span.len();
                            self.totals.requeued += span.len();
                            front.push(Span { redo: span.redo + 1, ..span });
                        } else {
                            totals.exhausted += span.len();
                            self.totals.exhausted += span.len();
                            exhausted.push((span.start, span.end));
                        }
                    }
                }
            }
            self.events.push(DispenseEvent::Report {
                holder,
                block: entry.block,
                kind: entry.kind,
                ranges: entry.ranges.clone(),
                exhausted,
            });
        }
        for (id, block) in staged {
            if block.open.is_empty() {
                self.blocks.remove(&id);
                if let Some(live) = self.live.get_mut(&holder) {
                    live.blocks.remove(&id);
                }
                self.events.push(DispenseEvent::BlockDone { holder, block: id });
            } else {
                self.blocks.insert(id, block);
            }
        }
        self.push_front(front);
        Ok(totals)
    }

    /// Expire every holder silent for the lease; returns them in order.
    pub fn expire(&mut self, now: Instant) -> Vec<Holder> {
        let lease = self.cfg.lease;
        let gone: Vec<Holder> = self
            .live
            .iter()
            .filter(|(_, live)| now.saturating_duration_since(live.last_seen) >= lease)
            .map(|(&holder, _)| holder)
            .collect();
        for &holder in &gone {
            self.reclaim(holder, ReclaimReason::Expired);
        }
        gone
    }

    #[cfg_attr(not(test), allow(dead_code))] // tests and offline tools
    pub fn next_expiry(&self) -> Option<Instant> {
        self.live.values().map(|live| live.last_seen + self.cfg.lease).min()
    }

    /// Persisted record: the state as if every live holder were reclaimed
    /// (a restart retires them all). Layout, little-endian:
    /// magic `YDS2` | has_data u8 | fingerprint [32] | stream count u16 |
    /// per stream (rows u64, weight u64, name len u16, name, next u64,
    /// passes_started u64) | next_block u64 | 7 x u64 totals |
    /// queue count u32 | (start u64, end u64, redo u32) x count (each
    /// stream's held positions, then its queue, in stream order) |
    /// retired count u32 | (learner u32, incarnation u64) x count.
    pub fn encode_persisted(&self) -> Vec<u8> {
        let mut held: Vec<Span> = Vec::new();
        for block in self.blocks.values() {
            for (&start, &(end, redo)) in &block.open {
                held.push(Span { start, end, redo });
            }
        }
        held.sort_by_key(|s| s.start);
        let reclaimed: u64 = held.iter().map(Span::len).sum();
        let mut queue: Vec<Span> = Vec::new();
        for (s, state) in self.streams.iter().enumerate() {
            queue.extend(held.iter().copied().filter(|span| stream_of(span.start) == s));
            queue.extend(state.queue.iter().copied());
        }
        let mut retired: Vec<Holder> = self.retired.0.iter().copied().collect();
        retired.extend(self.live.keys().copied().filter(|h| !self.retired.1.contains(h)));
        if retired.len() > RETIRED_LIMIT {
            retired.drain(..retired.len() - RETIRED_LIMIT);
        }
        let mut out = Vec::new();
        out.extend_from_slice(PERSIST_MAGIC);
        out.push(u8::from(self.data.is_some()));
        let empty = DataDecl { streams: Vec::new(), fingerprint: [0; 32] };
        let data = self.data.as_ref().unwrap_or(&empty);
        out.extend_from_slice(&data.fingerprint);
        out.extend_from_slice(&(data.streams.len() as u16).to_le_bytes());
        for (decl, state) in data.streams.iter().zip(&self.streams) {
            out.extend_from_slice(&decl.rows.to_le_bytes());
            out.extend_from_slice(&decl.weight.to_le_bytes());
            out.extend_from_slice(&(decl.name.len() as u16).to_le_bytes());
            out.extend_from_slice(decl.name.as_bytes());
            out.extend_from_slice(&state.next.to_le_bytes());
            out.extend_from_slice(&state.passes_started.to_le_bytes());
        }
        out.extend_from_slice(&self.next_block.to_le_bytes());
        let t = self.totals;
        for value in [t.granted, t.trained, t.dropped, t.requeued, t.exhausted, t.released, t.reclaimed + reclaimed] {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out.extend_from_slice(&(queue.len() as u32).to_le_bytes());
        for span in &queue {
            out.extend_from_slice(&span.start.to_le_bytes());
            out.extend_from_slice(&span.end.to_le_bytes());
            out.extend_from_slice(&span.redo.to_le_bytes());
        }
        out.extend_from_slice(&(retired.len() as u32).to_le_bytes());
        for holder in &retired {
            out.extend_from_slice(&holder.learner_id.to_le_bytes());
            out.extend_from_slice(&holder.incarnation.to_le_bytes());
        }
        out
    }

    /// `encode_persisted` for a checkpoint cut, logged as such.
    pub fn checkpoint_record(&mut self) -> Vec<u8> {
        let bytes = self.encode_persisted();
        self.events.push(DispenseEvent::Checkpoint {
            next: self.next_position(),
            queued: self.queued(),
            outstanding: self.outstanding(),
        });
        bytes
    }

    /// Restore a record of `encode_persisted` (or a P2 `YDS1` record) into a
    /// fresh dispenser. Parsed fully before any state changes.
    pub fn restore_persisted(&mut self, bytes: &[u8]) -> Result<()> {
        ensure!(self.data.is_none() && self.live.is_empty(), "restore needs a fresh dispenser");
        let mut r = Reader(bytes);
        let magic = r.take(4)?;
        let (data, states, next_block, t, queue, holders) = if magic == PERSIST_MAGIC_V1 {
            let has_data = r.u8()? != 0;
            let rows = r.u64()?;
            let fp: [u8; 32] = r.take(32)?.try_into()?;
            let next = r.u64()?;
            let next_block = r.u64()?;
            let passes_started = r.u64()?;
            let t = r.totals()?;
            let queue = r.spans()?;
            let holders = r.holders()?;
            ensure!(has_data || (next == 0 && queue.is_empty()), "dispense record has positions but no rows");
            ensure!(next < STREAM_POSITIONS, "dispense record has {next} positions");
            let data = has_data.then(|| DataDecl::single(rows, fp));
            let states = vec![(next, passes_started)];
            (data, states, next_block, t, queue, holders)
        } else {
            ensure!(magic == PERSIST_MAGIC, "dispense record has a bad magic");
            let has_data = r.u8()? != 0;
            let fp: [u8; 32] = r.take(32)?.try_into()?;
            let n = usize::from(r.u16()?);
            ensure!(n <= MAX_STREAMS, "dispense record has {n} streams");
            let mut streams = Vec::with_capacity(n);
            let mut states = Vec::with_capacity(n);
            for _ in 0..n {
                let rows = r.u64()?;
                let weight = r.u64()?;
                let len = usize::from(r.u16()?);
                let name = String::from_utf8(r.take(len)?.to_vec())?;
                streams.push(StreamDecl { rows, weight, name });
                let next = r.u64()?;
                ensure!(next <= STREAM_POSITIONS, "dispense record stream has {next} positions");
                states.push((next, r.u64()?));
            }
            let next_block = r.u64()?;
            let t = r.totals()?;
            let queue = r.spans()?;
            let holders = r.holders()?;
            ensure!(has_data == !streams.is_empty(), "dispense record has streams without data or data without streams");
            let data = has_data.then_some(DataDecl { streams, fingerprint: fp });
            if let Some(data) = &data {
                data.validate().map_err(|m| anyhow::anyhow!("dispense record: {m}"))?;
            }
            (data, states, next_block, t, queue, holders)
        };
        ensure!(r.0.is_empty(), "dispense record has {} trailing bytes", r.0.len());
        let streams = data.as_ref().map_or(0, |d| d.streams.len());
        for span in &queue {
            let s = stream_of(span.start);
            ensure!(
                span.start < span.end
                    && s < streams
                    && stream_of(span.end - 1) == s
                    && local_of(span.end - 1) < states[s].0,
                "dispense record has a bad span {span:?}"
            );
        }
        if let Some(data) = data {
            self.fix_data(data);
        }
        for (state, (next, passes_started)) in self.streams.iter_mut().zip(states) {
            state.next = next;
            state.passes_started = passes_started;
        }
        self.next_block = next_block.max(1);
        self.totals = Totals {
            granted: t[0],
            trained: t[1],
            dropped: t[2],
            requeued: t[3],
            exhausted: t[4],
            released: t[5],
            reclaimed: t[6],
        };
        for span in queue {
            self.streams[stream_of(span.start)].queue.push_back(span);
        }
        let retired = holders.len();
        for holder in holders {
            self.retire(holder);
        }
        self.events.push(DispenseEvent::Restore { next: self.next_position(), queued: self.queued(), retired });
        Ok(())
    }

    /// Accounting invariant: every position given out is reported (trained,
    /// dropped, redo exhausted), queued, or outstanding exactly once, and
    /// every queued or held position lies below its stream's next position.
    #[cfg_attr(not(test), allow(dead_code))] // tests and offline tools
    pub fn check_invariants(&self) -> Result<()> {
        let mut covered: Vec<(u64, u64)> =
            self.streams.iter().flat_map(|s| s.queue.iter()).map(|s| (s.start, s.end)).collect();
        for block in self.blocks.values() {
            ensure!(self.live.get(&block.holder).is_some(), "a block of a dead holder is in flight");
            covered.extend(block.open.iter().map(|(&s, &(e, _))| (s, e)));
        }
        covered.sort_unstable();
        for pair in covered.windows(2) {
            ensure!(pair[0].1 <= pair[1].0, "positions {pair:?} are queued or held twice");
        }
        for &(a, b) in &covered {
            let s = stream_of(a);
            ensure!(
                s < self.streams.len() && stream_of(b - 1) == s && local_of(b - 1) < self.streams[s].next,
                "positions [{a}, {b}) are not given-out positions of one stream"
            );
        }
        let held: u64 = covered.iter().map(|(a, b)| b - a).sum();
        let t = self.totals;
        let settled = t.trained + t.dropped + t.exhausted;
        ensure!(
            settled + held == self.next_position(),
            "settled {settled} + queued/held {held} != given out {}",
            self.next_position()
        );
        Ok(())
    }
}

fn hex8(bytes: &[u8; 32]) -> String {
    bytes[..8].iter().map(|b| format!("{b:02x}")).collect()
}

/// Remove `[start, end)` from `open`; it must be covered without gaps.
fn take_range(open: &mut BTreeMap<u64, (u64, u32)>, start: u64, end: u64) -> std::result::Result<Vec<Span>, String> {
    if start >= end {
        return Err(format!("empty range [{start}, {end})"));
    }
    // The span that contains `start`, then spans that follow it without a gap.
    let mut pieces: Vec<(u64, u64, u32)> = Vec::new(); // whole open spans touched
    let first = open.range(..=start).next_back().map(|(&s, &(e, r))| (s, e, r));
    match first {
        Some((s, e, r)) if e > start => pieces.push((s, e, r)),
        _ => return Err(format!("position {start} is not an unreported position of the block")),
    }
    let mut cursor = pieces[0].1;
    while cursor < end {
        match open.get(&cursor) {
            Some(&(e, r)) => {
                pieces.push((cursor, e, r));
                cursor = e;
            }
            None => return Err(format!("position {cursor} is not an unreported position of the block")),
        }
    }
    let mut taken = Vec::new();
    for (s, e, r) in pieces {
        open.remove(&s);
        if s < start {
            open.insert(s, (start, r));
        }
        if e > end {
            open.insert(end, (e, r));
        }
        taken.push(Span { start: s.max(start), end: e.min(end), redo: r });
    }
    Ok(taken)
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        ensure!(self.0.len() >= n, "dispense record is truncated");
        let (head, tail) = self.0.split_at(n);
        self.0 = tail;
        Ok(head)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into()?))
    }
    fn totals(&mut self) -> Result<[u64; 7]> {
        let mut t = [0u64; 7];
        for value in &mut t {
            *value = self.u64()?;
        }
        Ok(t)
    }
    fn spans(&mut self) -> Result<Vec<Span>> {
        let n = self.u32()? as usize;
        ensure!(n <= self.0.len() / 20, "dispense record queue is truncated");
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(Span { start: self.u64()?, end: self.u64()?, redo: self.u32()? });
        }
        Ok(out)
    }
    fn holders(&mut self) -> Result<Vec<Holder>> {
        let n = self.u32()? as usize;
        ensure!(n <= RETIRED_LIMIT, "dispense record retires {n} incarnations");
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(Holder { learner_id: self.u32()?, incarnation: self.u64()? });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FP: [u8; 32] = [7; 32];

    /// Strictly increasing request numbers (per holder is enough; global is simpler).
    fn seq() -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(1);
        NEXT.fetch_add(1, Ordering::Relaxed)
    }

    fn h(learner_id: u32, incarnation: u64) -> Holder {
        Holder { learner_id, incarnation }
    }

    fn cfg(passes: Option<u64>) -> DispenseConfig {
        DispenseConfig { lease: Duration::from_secs(10), passes, max_redo: 1, max_block: 64 }
    }

    fn grant(d: &mut Dispenser, holder: Holder, n: u32, now: Instant) -> (u64, Vec<(u64, u64)>) {
        match d.claim(holder, seq(), n, now).unwrap() {
            Claim::Grant { block, ranges } => (block, ranges),
            other => panic!("expected a grant, got {other:?}"),
        }
    }

    fn report(d: &mut Dispenser, holder: Holder, block: u64, kind: ReportKind, ranges: &[(u64, u64)], now: Instant) -> ReportTotals {
        d.report(holder, seq(), &[ReportEntry { block, kind, ranges: ranges.to_vec() }], now).unwrap()
    }

    #[test]
    fn fresh_positions_follow_one_global_sequence_whatever_the_island() {
        let t = Instant::now();
        let mut d = Dispenser::new(cfg(None));
        d.open(h(0, 1), 10, FP, t).unwrap();
        d.open(h(1, 2), 10, FP, t).unwrap();
        assert_eq!(grant(&mut d, h(0, 1), 4, t).1, vec![(0, 4)]);
        assert_eq!(grant(&mut d, h(1, 2), 4, t).1, vec![(4, 8)]);
        // a fast island claims again: it simply gets the next positions (crossing into pass 1)
        assert_eq!(grant(&mut d, h(0, 1), 4, t).1, vec![(8, 12)]);
        let passes: Vec<u64> = d
            .take_events()
            .into_iter()
            .filter_map(|e| match e {
                DispenseEvent::PassStart { pass, .. } => Some(pass),
                _ => None,
            })
            .collect();
        assert_eq!(passes, vec![0, 1]);
        d.check_invariants().unwrap();
    }

    #[test]
    fn reported_blocks_retire_and_state_stays_small() {
        let t = Instant::now();
        let mut d = Dispenser::new(cfg(None));
        d.open(h(0, 1), 1000, FP, t).unwrap();
        for _ in 0..100 {
            let (block, ranges) = grant(&mut d, h(0, 1), 8, t);
            report(&mut d, h(0, 1), block, ReportKind::Trained, &ranges, t);
        }
        assert_eq!(d.blocks_in_flight(), 0);
        assert_eq!(d.totals.trained, 800);
        assert_eq!(d.next_position(), 800);
        d.check_invariants().unwrap();
    }

    #[test]
    fn a_new_incarnation_supersedes_and_its_unreported_positions_go_first_to_the_next_claim() {
        let t = Instant::now();
        let mut d = Dispenser::new(cfg(None));
        d.open(h(0, 1), 100, FP, t).unwrap();
        d.open(h(1, 9), 100, FP, t).unwrap();
        let (b, ranges) = grant(&mut d, h(0, 1), 10, t);
        assert_eq!(ranges, vec![(0, 10)]);
        report(&mut d, h(0, 1), b, ReportKind::Trained, &[(0, 4)], t);
        grant(&mut d, h(1, 9), 10, t); // 10..20
        // island 0 restarts (new process): 4..10 return to the queue front
        assert_eq!(
            d.open(h(0, 2), 100, FP, t).unwrap(),
            OpenOutcome::Joined { superseded: Some(1), reclaimed: 6 }
        );
        assert!(d.is_retired(h(0, 1)));
        // island 1 (fast) claims first and takes them, then fresh positions
        assert_eq!(grant(&mut d, h(1, 9), 8, t).1, vec![(4, 10), (20, 22)]);
        // the old incarnation is fenced
        let error = d.claim(h(0, 1), seq(), 4, t).unwrap_err();
        assert!(error.to_string().starts_with(FENCED_PREFIX), "{error}");
        assert!(d.report(h(0, 1), seq(), &[ReportEntry { block: b, kind: ReportKind::Trained, ranges: vec![(4, 5)] }], t).is_err());
        assert!(d.open(h(0, 1), 100, FP, t).is_err(), "a retired incarnation cannot come back");
        d.check_invariants().unwrap();
    }

    #[test]
    fn lease_expiry_returns_the_silent_islands_positions() {
        let t = Instant::now();
        let mut d = Dispenser::new(cfg(None));
        d.open(h(0, 1), 100, FP, t).unwrap();
        d.open(h(1, 2), 100, FP, t).unwrap();
        grant(&mut d, h(0, 1), 5, t);
        let s = Duration::from_secs(1);
        d.heartbeat(h(1, 2), t + 9 * s).unwrap();
        assert!(d.expire(t + 9 * s).is_empty());
        assert_eq!(d.next_expiry(), Some(t + 10 * s));
        assert_eq!(d.expire(t + 10 * s), vec![h(0, 1)]);
        assert_eq!(d.queued(), 5);
        assert_eq!(grant(&mut d, h(1, 2), 3, t + 11 * s).1, vec![(0, 3)]);
        d.check_invariants().unwrap();
    }

    #[test]
    fn a_reconnect_of_the_same_incarnation_keeps_its_blocks() {
        let t = Instant::now();
        let mut d = Dispenser::new(cfg(None));
        d.open(h(0, 1), 100, FP, t).unwrap();
        let (b, _) = grant(&mut d, h(0, 1), 5, t);
        assert_eq!(d.open(h(0, 1), 100, FP, t).unwrap(), OpenOutcome::Reconnected);
        report(&mut d, h(0, 1), b, ReportKind::Trained, &[(0, 5)], t);
        assert_eq!(d.blocks_in_flight(), 0);
    }

    #[test]
    fn requeue_redoes_a_stale_position_at_most_max_redo_times() {
        let t = Instant::now();
        let mut d = Dispenser::new(cfg(None));
        d.open(h(0, 1), 100, FP, t).unwrap();
        let (b, _) = grant(&mut d, h(0, 1), 4, t);
        let totals = report(&mut d, h(0, 1), b, ReportKind::Requeue, &[(1, 3)], t);
        assert_eq!(totals, ReportTotals { settled: 2, requeued: 2, exhausted: 0 });
        // the requeued positions come first, then fresh ones
        let (b2, ranges) = grant(&mut d, h(0, 1), 4, t);
        assert_eq!(ranges, vec![(1, 3), (4, 6)]);
        // second time stale: max_redo = 1 is reached, the position is settled
        let totals = report(&mut d, h(0, 1), b2, ReportKind::Requeue, &[(1, 3)], t);
        assert_eq!(totals, ReportTotals { settled: 2, requeued: 0, exhausted: 2 });
        assert_eq!(d.totals.exhausted, 2);
        report(&mut d, h(0, 1), b2, ReportKind::Trained, &[(4, 6)], t);
        report(&mut d, h(0, 1), b, ReportKind::Dropped, &[(0, 1), (3, 4)], t);
        assert_eq!(d.blocks_in_flight(), 0);
        d.check_invariants().unwrap();
    }

    #[test]
    fn release_returns_positions_without_counting_a_redo() {
        let t = Instant::now();
        let mut d = Dispenser::new(DispenseConfig { max_redo: 0, ..cfg(None) });
        d.open(h(0, 1), 100, FP, t).unwrap();
        let (b, _) = grant(&mut d, h(0, 1), 4, t);
        report(&mut d, h(0, 1), b, ReportKind::Release, &[(0, 4)], t);
        let (b, ranges) = grant(&mut d, h(0, 1), 4, t);
        assert_eq!(ranges, vec![(0, 4)]);
        // max_redo 0: a stale position is settled at once
        assert_eq!(report(&mut d, h(0, 1), b, ReportKind::Requeue, &[(0, 4)], t).exhausted, 4);
    }

    #[test]
    fn a_report_outside_the_block_changes_nothing() {
        let t = Instant::now();
        let mut d = Dispenser::new(cfg(None));
        d.open(h(0, 1), 100, FP, t).unwrap();
        d.open(h(1, 2), 100, FP, t).unwrap();
        let (b0, _) = grant(&mut d, h(0, 1), 4, t); // 0..4
        let (b1, _) = grant(&mut d, h(1, 2), 4, t); // 4..8
        let before = d.encode_persisted();
        // second entry is bad: nothing of the first applies
        let entries = [
            ReportEntry { block: b0, kind: ReportKind::Trained, ranges: vec![(0, 2)] },
            ReportEntry { block: b0, kind: ReportKind::Trained, ranges: vec![(1, 3)] },
        ];
        assert!(d.report(h(0, 1), seq(), &entries, t).is_err());
        // another island's block, a twice-reported range, an empty range
        assert!(d.report(h(0, 1), seq(), &[ReportEntry { block: b1, kind: ReportKind::Trained, ranges: vec![(4, 5)] }], t).is_err());
        assert!(d.report(h(0, 1), seq(), &[ReportEntry { block: b0, kind: ReportKind::Trained, ranges: vec![(3, 3)] }], t).is_err());
        assert!(d.report(h(0, 1), seq(), &[ReportEntry { block: b0, kind: ReportKind::Trained, ranges: vec![(3, 5)] }], t).is_err());
        assert_eq!(d.encode_persisted(), before);
        d.check_invariants().unwrap();
    }

    #[test]
    fn a_range_may_cover_adjacent_spans_of_one_block() {
        let t = Instant::now();
        let mut d = Dispenser::new(cfg(None));
        d.open(h(0, 1), 100, FP, t).unwrap();
        let (b, _) = grant(&mut d, h(0, 1), 6, t);
        report(&mut d, h(0, 1), b, ReportKind::Release, &[(2, 4)], t);
        let (b2, ranges) = grant(&mut d, h(0, 1), 4, t);
        assert_eq!(ranges, vec![(2, 4), (6, 8)]);
        report(&mut d, h(0, 1), b, ReportKind::Trained, &[(0, 2), (4, 6)], t);
        report(&mut d, h(0, 1), b2, ReportKind::Trained, &[(2, 4), (6, 8)], t);
        assert_eq!(d.blocks_in_flight(), 0);
    }

    #[test]
    fn limited_passes_end_only_after_every_position_is_reported() {
        let t = Instant::now();
        let mut d = Dispenser::new(cfg(Some(1)));
        d.open(h(0, 1), 6, FP, t).unwrap();
        d.open(h(1, 2), 6, FP, t).unwrap();
        let (b0, r0) = grant(&mut d, h(0, 1), 4, t);
        assert_eq!(r0, vec![(0, 4)]);
        assert_eq!(grant(&mut d, h(1, 2), 4, t).1, vec![(4, 6)], "the last block is short");
        assert_eq!(d.claim(h(1, 2), seq(), 4, t).unwrap(), Claim::Wait, "island 0 may still return positions");
        // island 0 dies: its positions go to island 1
        d.open(h(0, 3), 6, FP, t).unwrap();
        let (b, r) = grant(&mut d, h(1, 2), 4, t);
        assert_eq!(r, vec![(0, 4)]);
        assert!(d.report(h(0, 1), seq(), &[ReportEntry { block: b0, kind: ReportKind::Trained, ranges: vec![(0, 4)] }], t).is_err());
        report(&mut d, h(1, 2), b, ReportKind::Trained, &r, t);
        assert_eq!(d.claim(h(0, 3), seq(), 4, t).unwrap(), Claim::Wait);
        let held: Vec<u64> = d.blocks.keys().copied().collect();
        report(&mut d, h(1, 2), held[0], ReportKind::Trained, &[(4, 6)], t);
        assert_eq!(d.claim(h(0, 3), seq(), 4, t).unwrap(), Claim::End);
        d.check_invariants().unwrap();
    }

    #[test]
    fn data_mismatch_and_bad_claims_are_refused() {
        let t = Instant::now();
        let mut d = Dispenser::new(cfg(None));
        d.open(h(0, 1), 10, FP, t).unwrap();
        assert!(d.open(h(1, 2), 11, FP, t).unwrap_err().to_string().starts_with(MISMATCH_PREFIX));
        assert!(d.open(h(1, 2), 10, [8; 32], t).unwrap_err().to_string().starts_with(MISMATCH_PREFIX));
        assert!(d.open(h(1, 2), 0, FP, t).is_err());
        assert!(d.claim(h(0, 1), seq(), 0, t).unwrap_err().to_string().starts_with(REFUSED_PREFIX));
        assert!(d.claim(h(0, 1), seq(), 65, t).is_err());
        assert!(d.claim(h(5, 5), seq(), 1, t).unwrap_err().to_string().starts_with(FENCED_PREFIX));
        assert!(DispenseConfig { passes: Some(0), ..cfg(None) }.validate().is_err());
    }

    #[test]
    fn restore_retires_every_holder_and_returns_their_positions_first() {
        let t = Instant::now();
        let mut d = Dispenser::new(cfg(None));
        d.open(h(0, 1), 50, FP, t).unwrap();
        d.open(h(1, 2), 50, FP, t).unwrap();
        let (b0, _) = grant(&mut d, h(0, 1), 5, t); // 0..5
        grant(&mut d, h(1, 2), 5, t); // 5..10
        report(&mut d, h(0, 1), b0, ReportKind::Trained, &[(0, 3)], t);
        let (b, _) = grant(&mut d, h(1, 2), 2, t); // 10..12
        report(&mut d, h(1, 2), b, ReportKind::Requeue, &[(10, 12)], t);
        let bytes = d.encode_persisted();
        let mut r = Dispenser::new(cfg(None));
        r.restore_persisted(&bytes).unwrap();
        r.check_invariants().unwrap();
        assert_eq!(r.next_position(), 12);
        assert_eq!(r.queued(), 2 + 5 + 2);
        assert!(r.is_retired(h(0, 1)) && r.is_retired(h(1, 2)));
        assert!(r.open(h(0, 1), 50, FP, t).is_err());
        assert!(r.open(h(0, 4), 49, FP, t).is_err(), "the restored rows bind the run");
        r.open(h(0, 4), 50, FP, t).unwrap();
        assert_eq!(grant(&mut r, h(0, 4), 20, t).1, vec![(3, 5), (5, 10), (10, 12), (12, 23)]);
        // the requeued positions kept their redo count
        let (b, _) = (r.next_block - 1, ());
        let totals = report(&mut r, h(0, 4), b, ReportKind::Requeue, &[(10, 12)], t);
        assert_eq!(totals.exhausted, 2);
        // equal state gives equal bytes; a truncated record is refused
        let mut again = Dispenser::new(cfg(None));
        again.restore_persisted(&bytes).unwrap();
        assert_eq!(again.encode_persisted(), {
            let mut x = Dispenser::new(cfg(None));
            x.restore_persisted(&bytes).unwrap();
            x.encode_persisted()
        });
        assert!(Dispenser::new(cfg(None)).restore_persisted(&bytes[..bytes.len() - 1]).is_err());
    }

    #[test]
    fn a_resent_request_gets_the_same_answer_and_is_applied_once() {
        let t = Instant::now();
        let mut d = Dispenser::new(cfg(None));
        d.open(h(0, 1), 100, FP, t).unwrap();
        let first = d.claim(h(0, 1), 10, 4, t).unwrap();
        assert_eq!(d.claim(h(0, 1), 10, 4, t).unwrap(), first, "the answer was lost: same grant again");
        assert_eq!(d.next_position(), 4);
        let Claim::Grant { block, .. } = first else { panic!() };
        let entry = [ReportEntry { block, kind: ReportKind::Trained, ranges: vec![(0, 4)] }];
        let totals = d.report(h(0, 1), 11, &entry, t).unwrap();
        assert_eq!(d.report(h(0, 1), 11, &entry, t).unwrap(), totals);
        assert_eq!(d.totals.trained, 4);
        assert!(d.claim(h(0, 1), 11, 4, t).is_err(), "seq 11 was a report");
        assert!(d.claim(h(0, 1), 9, 4, t).is_err(), "an older request is refused");
        d.check_invariants().unwrap();
    }

    #[test]
    fn events_render_as_json_fields() {
        let holder = h(3, 0xab);
        let e = DispenseEvent::Grant { holder, block: 7, ranges: vec![(0, 4), (9, 10)], first_pass: 0, last_pass: 1 };
        assert_eq!(
            e.json_fields(),
            "\"kind\":\"dispense_grant\",\"island_id\":3,\"incarnation\":\"00000000000000ab\",\"block\":7,\"positions\":5,\"ranges\":[[0,4],[9,10]],\"first_pass\":0,\"last_pass\":1"
        );
    }

    /// Small deterministic generator (no rand crate).
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    /// Random claim / report / restart / expiry / restore sequences against a
    /// brute-force model: no position is held twice, the accounting adds up,
    /// a position is reported trained at most once per (redo) grant, and a
    /// limited run ends with every position settled.
    #[test]
    fn random_sequences_keep_the_invariants() {
        for seed in 0..200u64 {
            let mut rng = Lcg(seed * 7919 + 1);
            let passes = if seed % 2 == 0 { Some(1 + rng.below(3)) } else { None };
            let rows = 1 + rng.below(40);
            let config = DispenseConfig { lease: Duration::from_secs(5), passes, max_redo: (seed % 3) as u32, max_block: 16 };
            let mut d = Dispenser::new(config.clone());
            let t0 = Instant::now();
            let mut now = t0;
            let mut incarnation = [1u64, 1, 1, 1];
            // per island: blocks it believes it holds -> unreported positions
            let mut held: [Vec<(u64, Vec<u64>)>; 4] = Default::default();
            let mut trained_count: HashMap<u64, u32> = HashMap::new();
            for island in 0..4u32 {
                d.open(h(island, 1), rows, FP, now).unwrap();
            }
            for _step in 0..400 {
                now += Duration::from_millis(rng.below(1500));
                let island = rng.below(4) as usize;
                let me = h(island as u32, incarnation[island]);
                match rng.below(10) {
                    0..=3 => match d.claim(me, seq(), 1 + rng.below(16) as u32, now) {
                        Ok(Claim::Grant { block, ranges }) => {
                            let positions: Vec<u64> = ranges.iter().flat_map(|&(a, b)| a..b).collect();
                            held[island].push((block, positions));
                        }
                        Ok(_) => {}
                        Err(DispenseError::Fenced(_)) => {
                            held[island].clear();
                            incarnation[island] += 1;
                            d.open(h(island as u32, incarnation[island]), rows, FP, now).unwrap();
                        }
                        Err(e) => panic!("{e}"),
                    },
                    4..=7 if !held[island].is_empty() => {
                        let i = rng.below(held[island].len() as u64) as usize;
                        let (block, positions) = &mut held[island][i];
                        let block = *block;
                        let k = 1 + rng.below(positions.len() as u64) as usize;
                        let taken: Vec<u64> = positions.drain(..k).collect();
                        let kind = ReportKind::from_u8(1 + rng.below(4) as u8).unwrap();
                        let ranges: Vec<(u64, u64)> = taken.iter().map(|&p| (p, p + 1)).collect();
                        match d.report(me, seq(), &[ReportEntry { block, kind, ranges }], now) {
                            Ok(_) => {
                                if kind == ReportKind::Trained {
                                    for p in &taken {
                                        *trained_count.entry(*p).or_default() += 1;
                                    }
                                }
                            }
                            Err(DispenseError::Fenced(_)) => held[island].clear(),
                            Err(e) => panic!("seed {seed}: {e}"),
                        }
                        held[island].retain(|(_, p)| !p.is_empty());
                    }
                    8 => {
                        // the island process restarts
                        held[island].clear();
                        incarnation[island] += 1;
                        d.open(h(island as u32, incarnation[island]), rows, FP, now).unwrap();
                    }
                    9 if rng.below(8) == 0 => {
                        // the syncer restarts from its record
                        let bytes = d.encode_persisted();
                        d = Dispenser::new(config.clone());
                        d.restore_persisted(&bytes).unwrap();
                        for (i, inc) in incarnation.iter_mut().enumerate() {
                            held[i].clear();
                            *inc += 1;
                            d.open(h(i as u32, *inc), rows, FP, now).unwrap();
                        }
                    }
                    _ => {
                        let _ = d.heartbeat(me, now);
                    }
                }
                for gone in d.expire(now) {
                    held[gone.learner_id as usize].clear();
                }
                d.check_invariants().unwrap_or_else(|e| panic!("seed {seed}: {e}"));
            }
            // a position is reported trained at most once (a requeue is
            // not a training report; a restart keeps the trained ones)
            for (&p, &n) in &trained_count {
                assert!(n <= 1, "seed {seed}: position {p} trained {n} times");
            }
            // drain: every island reports everything it holds as trained
            if let Some(passes) = passes {
                for (island, inc) in incarnation.iter_mut().enumerate() {
                    if !d.is_live(h(island as u32, *inc)) {
                        held[island].clear();
                        *inc += 1;
                        d.open(h(island as u32, *inc), rows, FP, now).unwrap();
                    }
                }
                for _round in 0..10_000 {
                    let mut progressed = false;
                    for island in 0..4usize {
                        let me = h(island as u32, incarnation[island]);
                        for (block, positions) in std::mem::take(&mut held[island]) {
                            let ranges: Vec<(u64, u64)> = positions.iter().map(|&p| (p, p + 1)).collect();
                            if !ranges.is_empty() {
                                d.report(me, seq(), &[ReportEntry { block, kind: ReportKind::Trained, ranges }], now).unwrap();
                            }
                        }
                        if let Claim::Grant { block, ranges } = d.claim(me, seq(), 16, now).unwrap() {
                            held[island].push((block, ranges.iter().flat_map(|&(a, b)| a..b).collect()));
                            progressed = true;
                        }
                    }
                    if !progressed && held.iter().all(Vec::is_empty) {
                        break;
                    }
                }
                assert_eq!(d.claim(h(0, incarnation[0]), seq(), 1, now).unwrap(), Claim::End, "seed {seed}");
                assert_eq!(d.next_position(), rows * passes);
                d.check_invariants().unwrap();
            }
        }
    }

    // ------------------------------------------------------------- streams (S27 P3)

    fn decl(streams: &[(u64, u64, &str)]) -> DataDecl {
        DataDecl {
            streams: streams.iter().map(|&(rows, weight, name)| StreamDecl { rows, weight, name: name.into() }).collect(),
            fingerprint: FP,
        }
    }

    fn per_stream(ranges: &[(u64, u64)], streams: usize) -> Vec<u64> {
        let mut out = vec![0u64; streams];
        for &(a, b) in ranges {
            assert_eq!(stream_of(a), stream_of(b - 1), "a range never crosses streams");
            out[stream_of(a)] += b - a;
        }
        out
    }

    /// The mix of fresh positions: after every claim, every stream is within
    /// one position of its share (the P3 check "mix error within one block",
    /// here exact to one position), for any claim sizes and weights.
    #[test]
    fn streams_interleave_within_one_position_of_their_weights() {
        for seed in 0..300u64 {
            let mut rng = Lcg(seed * 104729 + 3);
            let k = 2 + rng.below(6) as usize;
            let streams: Vec<(u64, u64, &str)> = (0..k)
                .map(|_| {
                    let w = match rng.below(3) {
                        0 => 1 + rng.below(5),
                        1 => 1 + rng.below(1_000_000),
                        _ => 1 << (30 + rng.below(20)),
                    };
                    (1 + rng.below(50), w, "")
                })
                .collect();
            let data = decl(&streams);
            let total: u128 = streams.iter().map(|s| u128::from(s.1)).sum();
            let t = Instant::now();
            let mut d = Dispenser::new(cfg(None));
            for island in 0..3 {
                d.open_with(h(island, 1), &data, None, t).unwrap();
            }
            let mut given = vec![0u64; k];
            let mut n = 0u64;
            for _ in 0..60 {
                let island = rng.below(3) as u32;
                let (_, ranges) = grant(&mut d, h(island, 1), 1 + rng.below(40) as u32, t);
                for (s, c) in per_stream(&ranges, k).into_iter().enumerate() {
                    given[s] += c;
                    n += c;
                }
                for s in 0..k {
                    // |given_s - n w_s / total| < 1, in integers
                    let lhs = (u128::from(given[s]) * total).abs_diff(u128::from(n) * u128::from(streams[s].1));
                    assert!(lhs < total, "seed {seed}: stream {s} has {} of {n} (weights {streams:?})", given[s]);
                }
            }
            d.check_invariants().unwrap();
        }
    }

    #[test]
    fn a_holder_that_accepts_some_streams_gets_only_their_positions() {
        let t = Instant::now();
        let data = decl(&[(10, 10, "a/web"), (10, 10, "a/pwn"), (5, 5, "b")]);
        let mut d = Dispenser::new(cfg(Some(1)));
        d.open_with(h(0, 1), &data, Some(&[1]), t).unwrap();
        d.open_with(h(1, 2), &data, None, t).unwrap();
        let (b0, r0) = grant(&mut d, h(0, 1), 16, t);
        assert_eq!(per_stream(&r0, 3), vec![0, 10, 0], "island 0 takes only stream 1 (all 10 rows of one pass)");
        // island 0 holds the rest of stream 1: it waits, then ends once it reports them
        assert_eq!(d.claim(h(0, 1), seq(), 4, t).unwrap(), Claim::Wait);
        let (b1, r1) = grant(&mut d, h(1, 2), 64, t);
        assert_eq!(per_stream(&r1, 3), vec![10, 0, 5]);
        report(&mut d, h(0, 1), b0, ReportKind::Trained, &r0, t);
        assert_eq!(d.claim(h(0, 1), seq(), 4, t).unwrap(), Claim::End, "stream 1 is exhausted and settled");
        // island 1 still holds streams 0 and 2: an island taking every stream waits
        d.open_with(h(2, 3), &data, None, t).unwrap();
        assert_eq!(d.claim(h(2, 3), seq(), 4, t).unwrap(), Claim::Wait);
        report(&mut d, h(1, 2), b1, ReportKind::Trained, &r1, t);
        assert_eq!(d.claim(h(2, 3), seq(), 4, t).unwrap(), Claim::End);
        d.check_invariants().unwrap();
        // refusals: no stream, a stream the run does not have
        assert!(matches!(d.open_with(h(3, 4), &data, Some(&[]), t), Err(DispenseError::Refused(_))));
        assert!(matches!(d.open_with(h(3, 4), &data, Some(&[3]), t), Err(DispenseError::Refused(_))));
        // another stream table (same fingerprint) is a mismatch
        let other = decl(&[(10, 10, "a/web"), (10, 11, "a/pwn"), (5, 5, "b")]);
        let error = d.open_with(h(3, 4), &other, None, t).unwrap_err();
        assert!(matches!(&error, DispenseError::Mismatch(m) if m.contains("other streams or weights")), "{error}");
    }

    #[test]
    fn reclaimed_positions_go_back_to_their_own_stream() {
        let t = Instant::now();
        let data = decl(&[(100, 1, "web"), (100, 1, "pwn")]);
        let mut d = Dispenser::new(cfg(None));
        d.open_with(h(0, 1), &data, Some(&[0]), t).unwrap();
        d.open_with(h(1, 1), &data, Some(&[1]), t).unwrap();
        let (_, r0) = grant(&mut d, h(0, 1), 8, t);
        assert_eq!(r0, vec![(position_of(0, 0), position_of(0, 8))]);
        // island 0 restarts: its stream-0 positions wait for a stream-0 holder
        d.open_with(h(0, 2), &data, Some(&[0]), t).unwrap();
        let (_, r1) = grant(&mut d, h(1, 1), 4, t);
        assert_eq!(r1, vec![(position_of(1, 0), position_of(1, 4))], "a pwn island never gets web positions");
        let (_, r2) = grant(&mut d, h(0, 2), 8, t);
        assert_eq!(r2, vec![(position_of(0, 0), position_of(0, 8))], "the new incarnation takes them first");
        d.check_invariants().unwrap();
        let passes: Vec<(usize, u64)> = d
            .take_events()
            .into_iter()
            .filter_map(|e| match e {
                DispenseEvent::PassStart { stream, pass } => Some((stream, pass)),
                _ => None,
            })
            .collect();
        assert_eq!(passes, vec![(0, 0), (1, 0)]);
    }

    #[test]
    fn limited_passes_count_per_stream() {
        let t = Instant::now();
        let data = decl(&[(3, 7, "a"), (5, 3, "b")]);
        let mut d = Dispenser::new(cfg(Some(2)));
        d.open_with(h(0, 1), &data, None, t).unwrap();
        let mut got = vec![0u64; 2];
        loop {
            match d.claim(h(0, 1), seq(), 4, t).unwrap() {
                Claim::Grant { block, ranges } => {
                    for (s, c) in per_stream(&ranges, 2).into_iter().enumerate() {
                        got[s] += c;
                    }
                    report(&mut d, h(0, 1), block, ReportKind::Trained, &ranges, t);
                }
                Claim::End => break,
                Claim::Wait => panic!("one holder that reports everything never waits"),
            }
        }
        assert_eq!(got, vec![6, 10], "two passes of every stream, whatever the weights");
        d.check_invariants().unwrap();
    }

    #[test]
    fn the_stream_record_round_trips_and_a_p2_record_still_restores() {
        let t = Instant::now();
        let data = decl(&[(10, 3, "web"), (20, 1, "pwn \"x\"")]);
        let mut d = Dispenser::new(cfg(Some(3)));
        d.open_with(h(0, 1), &data, None, t).unwrap();
        let (b, r) = grant(&mut d, h(0, 1), 12, t);
        report(&mut d, h(0, 1), b, ReportKind::Trained, &r[..1], t);
        let _ = grant(&mut d, h(0, 1), 5, t);
        let bytes = d.encode_persisted();
        let mut e = Dispenser::new(cfg(Some(3)));
        e.restore_persisted(&bytes).unwrap();
        assert_eq!(e.next_position(), d.next_position());
        assert_eq!(e.queued(), d.outstanding() + d.queued());
        assert!(e.is_retired(h(0, 1)));
        e.check_invariants().unwrap();
        assert_eq!(e.encode_persisted(), bytes, "restore then encode is the identity");
        assert!(matches!(e.open_with(h(0, 2), &decl(&[(10, 3, "web")]), None, t), Err(DispenseError::Mismatch(_))));
        e.open_with(h(0, 2), &data, None, t).unwrap();
        let events: Vec<String> = e.take_events().iter().map(DispenseEvent::json_fields).collect();
        assert!(events[0].starts_with("\"kind\":\"dispense_streams\""), "{events:?}");
        assert!(events[0].contains("\"name\":\"pwn \\\"x\\\"\""), "{}", events[0]);
        // a P2 record (YDS1, one stream) restores as the single plain stream
        let mut v1 = Vec::new();
        v1.extend_from_slice(b"YDS1");
        v1.push(1);
        v1.extend_from_slice(&40u64.to_le_bytes());
        v1.extend_from_slice(&FP);
        for value in [12u64, 5, 1] {
            v1.extend_from_slice(&value.to_le_bytes()); // next, next_block, passes_started
        }
        for value in [12u64, 8, 0, 0, 0, 0, 0] {
            v1.extend_from_slice(&value.to_le_bytes());
        }
        v1.extend_from_slice(&1u32.to_le_bytes());
        for value in [8u64, 12] {
            v1.extend_from_slice(&value.to_le_bytes());
        }
        v1.extend_from_slice(&0u32.to_le_bytes());
        v1.extend_from_slice(&1u32.to_le_bytes());
        v1.extend_from_slice(&0u32.to_le_bytes());
        v1.extend_from_slice(&9u64.to_le_bytes());
        let mut f = Dispenser::new(cfg(None));
        f.restore_persisted(&v1).unwrap();
        f.check_invariants().unwrap();
        assert!(f.is_retired(h(0, 9)));
        f.open(h(1, 1), 40, FP, t).unwrap();
        assert_eq!(grant(&mut f, h(1, 1), 6, t).1, vec![(8, 12), (12, 14)]);
    }

    #[test]
    fn stream_events_render_as_json_fields() {
        let e = DispenseEvent::Open { holder: h(1, 0xab), superseded: None, reconnect: false, rows: 30, accept: Some(vec![0, 2]) };
        assert_eq!(
            e.json_fields(),
            "\"kind\":\"dispense_open\",\"island_id\":1,\"incarnation\":\"00000000000000ab\",\"rows\":30,\"reconnect\":false,\"superseded\":null,\"accept\":[0,2]"
        );
        let e = DispenseEvent::Streams { streams: decl(&[(10, 3, "a.jsonl/web"), (20, 1, "")]).streams, fingerprint: FP };
        assert_eq!(
            e.json_fields(),
            "\"kind\":\"dispense_streams\",\"stream_shift\":40,\"fingerprint\":\"0707070707070707\",\"streams\":[{\"stream\":0,\"name\":\"a.jsonl/web\",\"rows\":10,\"weight\":3},{\"stream\":1,\"name\":\"\",\"rows\":20,\"weight\":1}]"
        );
        assert_eq!(DispenseEvent::PassStart { stream: 1, pass: 2 }.json_fields(), "\"kind\":\"dispense_pass_start\",\"stream\":1,\"pass\":2");
    }

    /// The random sequences of `random_sequences_keep_the_invariants` with
    /// several streams and islands that accept only some of them.
    #[test]
    fn random_stream_sequences_keep_the_invariants() {
        for seed in 0..150u64 {
            let mut rng = Lcg(seed * 31337 + 11);
            let k = 1 + rng.below(4) as usize;
            let streams: Vec<(u64, u64, &str)> = (0..k).map(|_| (1 + rng.below(20), 1 + rng.below(9), "s")).collect();
            let data = decl(&streams);
            let passes = if seed % 2 == 0 { Some(1 + rng.below(2)) } else { None };
            let config = DispenseConfig { lease: Duration::from_secs(5), passes, max_redo: 1, max_block: 16 };
            // island 0 takes every stream (so every stream has a taker); the others a random subset
            let accepts: Vec<Option<Vec<u16>>> = (0..4)
                .map(|i| {
                    if i == 0 || rng.below(2) == 0 {
                        None
                    } else {
                        let picked: Vec<u16> = (0..k as u16).filter(|_| rng.below(2) == 0).collect();
                        if picked.is_empty() { Some(vec![0]) } else { Some(picked) }
                    }
                })
                .collect();
            let mut d = Dispenser::new(config.clone());
            let mut now = Instant::now();
            let mut incarnation = [1u64; 4];
            let mut held: [Vec<(u64, Vec<u64>)>; 4] = Default::default();
            let mut trained: HashMap<u64, u32> = HashMap::new();
            let open = |d: &mut Dispenser, i: usize, inc: u64, now| {
                d.open_with(h(i as u32, inc), &data, accepts[i].as_deref(), now).unwrap();
            };
            for i in 0..4 {
                open(&mut d, i, 1, now);
            }
            for _ in 0..300 {
                now += Duration::from_millis(rng.below(1500));
                let i = rng.below(4) as usize;
                let me = h(i as u32, incarnation[i]);
                match rng.below(9) {
                    0..=3 => match d.claim(me, seq(), 1 + rng.below(16) as u32, now) {
                        Ok(Claim::Grant { block, ranges }) => {
                            if let Some(a) = &accepts[i] {
                                for &(p, _) in &ranges {
                                    assert!(a.contains(&(stream_of(p) as u16)), "seed {seed}: island {i} got stream {}", stream_of(p));
                                }
                            }
                            held[i].push((block, ranges.iter().flat_map(|&(a, b)| a..b).collect()));
                        }
                        Ok(_) => {}
                        Err(DispenseError::Fenced(_)) => {
                            held[i].clear();
                            incarnation[i] += 1;
                            open(&mut d, i, incarnation[i], now);
                        }
                        Err(e) => panic!("{e}"),
                    },
                    4..=6 if !held[i].is_empty() => {
                        let j = rng.below(held[i].len() as u64) as usize;
                        let (block, positions) = &mut held[i][j];
                        let block = *block;
                        let n = 1 + rng.below(positions.len() as u64) as usize;
                        let taken: Vec<u64> = positions.drain(..n).collect();
                        let kind = ReportKind::from_u8(1 + rng.below(4) as u8).unwrap();
                        let ranges: Vec<(u64, u64)> = taken.iter().map(|&p| (p, p + 1)).collect();
                        match d.report(me, seq(), &[ReportEntry { block, kind, ranges }], now) {
                            Ok(_) if kind == ReportKind::Trained => {
                                for p in &taken {
                                    *trained.entry(*p).or_default() += 1;
                                }
                            }
                            Ok(_) => {}
                            Err(DispenseError::Fenced(_)) => held[i].clear(),
                            Err(e) => panic!("seed {seed}: {e}"),
                        }
                        held[i].retain(|(_, p)| !p.is_empty());
                    }
                    7 => {
                        held[i].clear();
                        incarnation[i] += 1;
                        open(&mut d, i, incarnation[i], now);
                    }
                    _ if rng.below(6) == 0 => {
                        let bytes = d.encode_persisted();
                        d = Dispenser::new(config.clone());
                        d.restore_persisted(&bytes).unwrap();
                        for (j, inc) in incarnation.iter_mut().enumerate() {
                            held[j].clear();
                            *inc += 1;
                            open(&mut d, j, *inc, now);
                        }
                    }
                    _ => {
                        let _ = d.heartbeat(me, now);
                    }
                }
                for gone in d.expire(now) {
                    held[gone.learner_id as usize].clear();
                }
                d.check_invariants().unwrap_or_else(|e| panic!("seed {seed}: {e}"));
            }
            for (&p, &n) in &trained {
                assert!(n <= 1, "seed {seed}: position {p} trained {n} times");
            }
        }
    }
}
