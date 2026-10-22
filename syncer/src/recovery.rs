//! Peer recovery of an island that lost its trainer state (S27 E-3).
//!
//! A short drop (TCP link lost, trainer alive) keeps the reconnect catch-up
//! in `server.rs`. This module covers the long failure: the island process
//! or node is gone, so its parameters *and* its inner optimizer state are
//! gone. The flow, seen from the syncer:
//!
//! 1. The island sends HELLO with the recover flag ([`HelloExt`]). The server
//!    registers the connection but keeps it out of the round roster and out
//!    of broadcasts, so its merge weight is 0.
//! 2. After any pending publish is durable, the scheduler calls
//!    [`PeerRecovery::begin`] with the syncer clock: `t_s` (global step) and
//!    `V_s` (per-fragment versions). The island gets `RECOVERY_CLOCK`; a
//!    healthy donor island gets `PEER_SNAPSHOT_REQ`.
//! 3. The donor keeps training. Once every fragment version it holds is at
//!    least `V_s[p]`, it snapshots its parameters and inner optimizer state
//!    to the shared snapshot store and answers `PEER_SNAPSHOT_READY` with its
//!    own versions `V_d`, the snapshot sha256 and the store locator. The
//!    syncer relays only this small record (`RECOVERY_SOURCE`), never the
//!    snapshot itself.
//! 4. The island restores the snapshot and sends `RECOVERY_RESTORED` with
//!    the versions it now holds. The syncer resends the current value of
//!    every fragment whose version is newer. Islands overwrite a fragment
//!    with a broadcast, so the newest value equals replaying every update
//!    after the donor's clock; the syncer keeps no update log. (A delta or
//!    blended downlink would need a bounded log of recent updates instead.)
//! 5. The syncer sends `RECOVERY_DONE` and puts the island back on the
//!    roster. The time from step 2 to step 5, in syncer steps, is checked
//!    against the configured bound H.
//!
//! Donor loss, a donor timeout, and a snapshot the island refuses
//! (`RECOVERY_FAILED`: sha256 mismatch, layout mismatch, read error) move to
//! the next donor. With no donor left, or after `max_donor_tries`, the
//! island gets the degraded source: it restores its own latest verified cut
//! (or fresh optimizer state) and then receives every fragment.
//!
//! The state machine is pure: it never touches sockets, and every clock
//! reading is passed in. The server executes the returned [`Outbound`]
//! messages.

use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

use anyhow::{bail, ensure, Context, Result};

use crate::membership::Member;

// Message types (numbered apart from protocol.rs so parallel protocol work
// does not collide). Syncer -> island unless noted.
/// `t_s u64 | P u32 | P x u64 V_s | donor u32 (NO_DONOR = none)`.
pub const MSG_RECOVERY_CLOCK: u8 = 40;
/// To the donor: `recovering_id u32 | recovering_generation u64 | t_s u64 |
/// P u32 | P x u64 V_s`.
pub const MSG_PEER_SNAPSHOT_REQ: u8 = 41;
/// Donor -> syncer: `recovering_id u32 | recovering_generation u64 | P u32 |
/// P x u64 V_d | sha256 [32] | n u32 | locator utf-8 [n]`.
pub const MSG_PEER_SNAPSHOT_READY: u8 = 42;
/// `kind u8 (1 peer, 2 degraded) | donor u32 | P u32 | P x u64 V_d |
/// sha256 [32] | n u32 | text utf-8 [n]` (text: locator, or the reason of a
/// degraded source; V_d and sha256 are zero for a degraded source).
pub const MSG_RECOVERY_SOURCE: u8 = 43;
/// Island -> syncer: `has_versions u8 | P u32 | P x u64 V_held`
/// (has_versions 0: the island holds nothing reliable; resend everything).
pub const MSG_RECOVERY_RESTORED: u8 = 44;
/// Island -> syncer: `n u32 | reason utf-8 [n]`.
pub const MSG_RECOVERY_FAILED: u8 = 45;
/// `t_s u64 | done_step u64 | replayed u32 | P u32 | P x u64 versions`
/// (the syncer versions the island holds once the replay has arrived).
pub const MSG_RECOVERY_DONE: u8 = 46;

pub const NO_DONOR: u32 = u32::MAX;
pub const SOURCE_PEER: u8 = 1;
pub const SOURCE_DEGRADED: u8 = 2;
/// Upper bound on any recovery control frame (locators and reasons are short).
pub const MAX_RECOVERY_FRAME: u64 = 1 << 20;
const MAX_TEXT: usize = 4096;

// --- HELLO extension -------------------------------------------------------

/// Magic of the optional HELLO extension trailer, always the last
/// `HELLO_EXT_LEN` bytes of the HELLO body:
/// `b"YEX1" | flags u8 | syncer_epoch u64`. Legacy clients send none.
pub const HELLO_EXT_MAGIC: &[u8; 4] = b"YEX1";
pub const HELLO_EXT_LEN: usize = 13;
const FLAG_HAS_EPOCH: u8 = 1;
const FLAG_RECOVER: u8 = 2;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HelloExt {
    /// The syncer incarnation the island was launched for (fence).
    pub syncer_epoch: Option<u64>,
    /// The island lost its trainer state and asks for peer recovery.
    pub recover: bool,
}

impl HelloExt {
    #[cfg(test)]
    pub fn encode(&self) -> [u8; HELLO_EXT_LEN] {
        let mut out = [0u8; HELLO_EXT_LEN];
        out[..4].copy_from_slice(HELLO_EXT_MAGIC);
        let mut flags = 0;
        if self.syncer_epoch.is_some() {
            flags |= FLAG_HAS_EPOCH;
        }
        if self.recover {
            flags |= FLAG_RECOVER;
        }
        out[4] = flags;
        out[5..].copy_from_slice(&self.syncer_epoch.unwrap_or(0).to_le_bytes());
        out
    }

    /// Split a HELLO body into (legacy body, extension). A body that does not
    /// end in a well-formed trailer is returned whole with no extension; the
    /// caller retries the whole body if the stripped one does not parse.
    pub fn split(body: &[u8]) -> (&[u8], Option<HelloExt>) {
        if body.len() < HELLO_EXT_LEN {
            return (body, None);
        }
        let (head, tail) = body.split_at(body.len() - HELLO_EXT_LEN);
        if &tail[..4] != HELLO_EXT_MAGIC || tail[4] & !(FLAG_HAS_EPOCH | FLAG_RECOVER) != 0 {
            return (body, None);
        }
        let flags = tail[4];
        let epoch = u64::from_le_bytes(tail[5..].try_into().unwrap());
        if flags & FLAG_HAS_EPOCH == 0 && epoch != 0 {
            return (body, None);
        }
        let ext = HelloExt {
            syncer_epoch: (flags & FLAG_HAS_EPOCH != 0).then_some(epoch),
            recover: flags & FLAG_RECOVER != 0,
        };
        (head, Some(ext))
    }
}

// --- wire codec ------------------------------------------------------------

struct Cursor<'a>(&'a [u8]);

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        ensure!(self.0.len() >= n, "recovery message is truncated");
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
    fn versions(&mut self, fragments: usize) -> Result<Vec<u64>> {
        let n = self.u32()? as usize;
        ensure!(n == fragments, "recovery message carries {n} fragment versions, layout has {fragments}");
        (0..n).map(|_| self.u64()).collect()
    }
    fn text(&mut self) -> Result<String> {
        let n = self.u32()? as usize;
        ensure!(n <= MAX_TEXT, "recovery text of {n} bytes exceeds {MAX_TEXT}");
        String::from_utf8(self.take(n)?.to_vec()).context("recovery text is not utf-8")
    }
    fn end(&self) -> Result<()> {
        ensure!(self.0.is_empty(), "recovery message has {} trailing bytes", self.0.len());
        Ok(())
    }
}

fn put_versions(out: &mut Vec<u8>, versions: &[u64]) {
    out.extend_from_slice(&(versions.len() as u32).to_le_bytes());
    for v in versions {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

fn put_text(out: &mut Vec<u8>, text: &str) {
    let text = &text.as_bytes()[..text.len().min(MAX_TEXT)];
    out.extend_from_slice(&(text.len() as u32).to_le_bytes());
    out.extend_from_slice(text);
}

/// The donor's answer to `PEER_SNAPSHOT_REQ`.
#[derive(Clone, Debug, PartialEq)]
pub struct SnapshotReady {
    pub recovering: Member,
    pub versions: Vec<u64>,
    pub sha256: [u8; 32],
    pub locator: String,
}

impl SnapshotReady {
    pub fn decode(payload: &[u8], fragments: usize) -> Result<Self> {
        let mut r = Cursor(payload);
        let recovering = Member { learner_id: r.u32()?, generation: r.u64()? };
        let versions = r.versions(fragments)?;
        let sha256 = r.take(32)?.try_into()?;
        let locator = r.text()?;
        r.end()?;
        ensure!(!locator.is_empty(), "PEER_SNAPSHOT_READY has an empty locator");
        Ok(Self { recovering, versions, sha256, locator })
    }

    #[cfg(test)]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&self.recovering.learner_id.to_le_bytes());
        out.extend_from_slice(&self.recovering.generation.to_le_bytes());
        put_versions(&mut out, &self.versions);
        out.extend_from_slice(&self.sha256);
        put_text(&mut out, &self.locator);
        out
    }
}

/// `RECOVERY_RESTORED`: the versions the island holds after its restore,
/// or None when it holds nothing reliable.
pub fn decode_restored(payload: &[u8], fragments: usize) -> Result<Option<Vec<u64>>> {
    let mut r = Cursor(payload);
    let held = match r.u8()? {
        0 => {
            ensure!(r.u32()? == 0, "RECOVERY_RESTORED without versions must list 0 fragments");
            None
        }
        1 => Some(r.versions(fragments)?),
        other => bail!("RECOVERY_RESTORED has_versions must be 0 or 1, got {other}"),
    };
    r.end()?;
    Ok(held)
}

#[cfg(test)]
pub fn encode_restored(held: Option<&[u64]>) -> Vec<u8> {
    let mut out = vec![held.is_some() as u8];
    put_versions(&mut out, held.unwrap_or(&[]));
    out
}

pub fn decode_failed(payload: &[u8]) -> Result<String> {
    let mut r = Cursor(payload);
    let reason = r.text()?;
    r.end()?;
    Ok(reason)
}

#[cfg(test)]
pub fn encode_failed(reason: &str) -> Vec<u8> {
    let mut out = Vec::new();
    put_text(&mut out, reason);
    out
}

fn encode_clock(t_s: u64, versions: &[u64], donor: Option<Member>) -> Vec<u8> {
    let mut out = t_s.to_le_bytes().to_vec();
    put_versions(&mut out, versions);
    out.extend_from_slice(&donor.map_or(NO_DONOR, |d| d.learner_id).to_le_bytes());
    out
}

fn encode_snapshot_request(recovering: Member, t_s: u64, versions: &[u64]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&recovering.learner_id.to_le_bytes());
    out.extend_from_slice(&recovering.generation.to_le_bytes());
    out.extend_from_slice(&t_s.to_le_bytes());
    put_versions(&mut out, versions);
    out
}

fn encode_source(kind: u8, donor: Option<Member>, versions: &[u64], sha256: &[u8; 32], text: &str) -> Vec<u8> {
    let mut out = vec![kind];
    out.extend_from_slice(&donor.map_or(NO_DONOR, |d| d.learner_id).to_le_bytes());
    put_versions(&mut out, versions);
    out.extend_from_slice(sha256);
    put_text(&mut out, text);
    out
}

fn encode_done(t_s: u64, done_step: u64, replayed: usize, versions: &[u64]) -> Vec<u8> {
    let mut out = t_s.to_le_bytes().to_vec();
    out.extend_from_slice(&done_step.to_le_bytes());
    out.extend_from_slice(&(replayed as u32).to_le_bytes());
    put_versions(&mut out, versions);
    out
}

// --- state machine ---------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct RecoveryConfig {
    /// Bound H on one recovery, in syncer steps; None: measured, not checked.
    pub bound_steps: Option<u64>,
    /// Donors tried before the degraded source.
    pub max_donor_tries: u32,
    /// Wait for one donor's snapshot before trying the next donor.
    pub donor_timeout: Duration,
}

impl Default for RecoveryConfig {
    fn default() -> Self {
        Self { bound_steps: None, max_donor_tries: 3, donor_timeout: Duration::from_secs(600) }
    }
}

impl RecoveryConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.max_donor_tries >= 1, "--recovery-max-donor-tries must be >= 1");
        ensure!(!self.donor_timeout.is_zero(), "--recovery-donor-timeout-s must be positive");
        Ok(())
    }
}

/// A current member that may serve as a donor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Candidate {
    pub member: Member,
    /// Pulls the member has not answered yet (fewer = less busy).
    pub outstanding: usize,
}

/// One message the server must send.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Outbound {
    pub to: Member,
    pub msg_type: u8,
    pub payload: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceKind {
    Peer { donor: Member },
    Degraded,
}

#[derive(Clone, Debug, PartialEq)]
enum Phase {
    AwaitSnapshot { donor: Member, deadline: Instant },
    AwaitRestore { source: SourceKind },
}

#[derive(Clone, Debug)]
struct Pending {
    member: Member,
    t_s: u64,
    versions: Vec<u64>,
    started: Instant,
    phase: Phase,
    tried: BTreeSet<u32>,
    /// Donors asked so far.
    attempts: u32,
    /// Why the last donor did not work out (for the record).
    last_failure: Option<String>,
}

/// What the server reports when a recovery completes.
#[derive(Clone, Debug, PartialEq)]
pub struct RecoveryRecord {
    pub member: Member,
    pub source: SourceKind,
    pub t_s: u64,
    pub done_step: u64,
    /// Syncer steps from the clock read to READY.
    pub steps: u64,
    pub bound_steps: Option<u64>,
    /// `steps` exceeded the bound (reported, never hidden).
    pub missed: bool,
    pub donor_attempts: u32,
    pub replayed: usize,
    pub seconds: f64,
    pub last_failure: Option<String>,
}

impl RecoveryRecord {
    pub fn to_json(&self) -> String {
        let (path, donor) = match self.source {
            SourceKind::Peer { donor } => ("peer", donor.learner_id.to_string()),
            SourceKind::Degraded => ("degraded", "null".to_string()),
        };
        format!(
            "{{\"kind\":\"recovery\",\"learner_id\":{},\"generation\":{},\"path\":\"{path}\",\"donor\":{donor},\"t_s\":{},\"done_step\":{},\"steps\":{},\"bound_steps\":{},\"missed\":{},\"donor_attempts\":{},\"replayed\":{},\"seconds\":{:.6},\"last_failure\":{}}}",
            self.member.learner_id,
            self.member.generation,
            self.t_s,
            self.done_step,
            self.steps,
            self.bound_steps.map_or("null".to_string(), |b| b.to_string()),
            self.missed,
            self.donor_attempts,
            self.replayed,
            self.seconds,
            self.last_failure
                .as_ref()
                .map_or("null".to_string(), |f| format!("{f:?}")),
        )
    }
}

#[derive(Debug, Default)]
pub struct PeerRecovery {
    cfg: RecoveryConfig,
    pending: BTreeMap<u32, Pending>,
}

impl PeerRecovery {
    pub fn new(cfg: RecoveryConfig) -> Self {
        Self { cfg, pending: BTreeMap::new() }
    }

    pub fn is_pending(&self, member: Member) -> bool {
        self.pending.get(&member.learner_id).is_some_and(|p| p.member == member)
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Start recovering `member` at the durable syncer clock `(t_s, versions)`.
    /// A pending recovery of an older generation of the same island is dropped.
    pub fn begin(
        &mut self,
        member: Member,
        t_s: u64,
        versions: &[u64],
        candidates: &[Candidate],
        now: Instant,
    ) -> Vec<Outbound> {
        let pending = Pending {
            member,
            t_s,
            versions: versions.to_vec(),
            started: now,
            phase: Phase::AwaitRestore { source: SourceKind::Degraded },
            tried: BTreeSet::new(),
            attempts: 0,
            last_failure: None,
        };
        self.pending.insert(member.learner_id, pending);
        let donor = self.pick_donor(member.learner_id, candidates);
        let mut out = vec![Outbound {
            to: member,
            msg_type: MSG_RECOVERY_CLOCK,
            payload: encode_clock(t_s, versions, donor),
        }];
        out.extend(self.ask(member.learner_id, donor, now));
        out
    }

    fn pick_donor(&self, learner_id: u32, candidates: &[Candidate]) -> Option<Member> {
        let pending = self.pending.get(&learner_id)?;
        if pending.attempts >= self.cfg.max_donor_tries {
            return None;
        }
        candidates
            .iter()
            .filter(|c| c.member.learner_id != learner_id)
            .filter(|c| !pending.tried.contains(&c.member.learner_id))
            .filter(|c| !self.pending.contains_key(&c.member.learner_id))
            .min_by_key(|c| (c.outstanding, c.member.learner_id))
            .map(|c| c.member)
    }

    /// Ask `donor` for a snapshot, or hand out the degraded source.
    fn ask(&mut self, learner_id: u32, donor: Option<Member>, now: Instant) -> Vec<Outbound> {
        let timeout = self.cfg.donor_timeout;
        let Some(pending) = self.pending.get_mut(&learner_id) else { return Vec::new() };
        match donor {
            Some(donor) => {
                pending.attempts += 1;
                pending.tried.insert(donor.learner_id);
                pending.phase = Phase::AwaitSnapshot { donor, deadline: now + timeout };
                vec![Outbound {
                    to: donor,
                    msg_type: MSG_PEER_SNAPSHOT_REQ,
                    payload: encode_snapshot_request(pending.member, pending.t_s, &pending.versions),
                }]
            }
            None => {
                pending.phase = Phase::AwaitRestore { source: SourceKind::Degraded };
                let reason = match &pending.last_failure {
                    Some(failure) => format!("no donor left after {} tries; last: {failure}", pending.attempts),
                    None => "no healthy donor island".to_string(),
                };
                vec![Outbound {
                    to: pending.member,
                    msg_type: MSG_RECOVERY_SOURCE,
                    payload: encode_source(SOURCE_DEGRADED, None, &vec![0; pending.versions.len()], &[0; 32], &reason),
                }]
            }
        }
    }

    fn next_donor(&mut self, learner_id: u32, failure: String, candidates: &[Candidate], now: Instant) -> Vec<Outbound> {
        if let Some(pending) = self.pending.get_mut(&learner_id) {
            pending.last_failure = Some(failure);
        }
        let donor = self.pick_donor(learner_id, candidates);
        self.ask(learner_id, donor, now)
    }

    /// A donor answered. Stale answers (wrong donor, superseded recovering
    /// generation, recovery already past this phase) are ignored. A donor
    /// whose versions are behind `V_s` broke the request; the next donor is
    /// asked.
    pub fn on_snapshot_ready(
        &mut self,
        from: Member,
        ready: SnapshotReady,
        candidates: &[Candidate],
        now: Instant,
    ) -> Vec<Outbound> {
        let learner_id = ready.recovering.learner_id;
        let Some(pending) = self.pending.get_mut(&learner_id) else { return Vec::new() };
        if pending.member != ready.recovering || pending.phase_donor() != Some(from) {
            return Vec::new();
        }
        if let Some(p) = (0..pending.versions.len()).find(|&p| ready.versions[p] < pending.versions[p]) {
            let failure = format!(
                "donor {} snapshot fragment {p} version {} is behind the clock version {}",
                from.learner_id, ready.versions[p], pending.versions[p]
            );
            return self.next_donor(learner_id, failure, candidates, now);
        }
        let source = SourceKind::Peer { donor: from };
        pending.phase = Phase::AwaitRestore { source };
        vec![Outbound {
            to: pending.member,
            msg_type: MSG_RECOVERY_SOURCE,
            payload: encode_source(SOURCE_PEER, Some(from), &ready.versions, &ready.sha256, &ready.locator),
        }]
    }

    /// The recovering island could not use the snapshot (sha256 mismatch,
    /// layout mismatch, read error). The next donor is asked.
    pub fn on_failed(&mut self, from: Member, reason: String, candidates: &[Candidate], now: Instant) -> Vec<Outbound> {
        if !self.is_pending(from) {
            return Vec::new();
        }
        let pending = &self.pending[&from.learner_id];
        if matches!(pending.phase, Phase::AwaitRestore { source: SourceKind::Degraded }) {
            // Nothing else to offer; the island must restore on its own.
            return Vec::new();
        }
        self.next_donor(from.learner_id, reason, candidates, now)
    }

    /// `member`'s connection or lease is gone. Its own recovery is dropped;
    /// a recovery waiting for it as a donor moves to the next donor.
    pub fn on_gone(&mut self, member: Member, candidates: &[Candidate], now: Instant) -> Vec<Outbound> {
        if self.is_pending(member) {
            self.pending.remove(&member.learner_id);
        }
        let waiting: Vec<u32> = self
            .pending
            .iter()
            .filter(|(_, p)| p.phase_donor() == Some(member))
            .map(|(&id, _)| id)
            .collect();
        let mut out = Vec::new();
        for learner_id in waiting {
            let failure = format!("donor {} left before its snapshot was ready", member.learner_id);
            out.extend(self.next_donor(learner_id, failure, candidates, now));
        }
        out
    }

    /// Donor timeouts.
    pub fn on_tick(&mut self, candidates: &[Candidate], now: Instant) -> Vec<Outbound> {
        let late: Vec<(u32, u32)> = self
            .pending
            .iter()
            .filter_map(|(&id, p)| match p.phase {
                Phase::AwaitSnapshot { donor, deadline } if now >= deadline => Some((id, donor.learner_id)),
                _ => None,
            })
            .collect();
        let mut out = Vec::new();
        for (learner_id, donor) in late {
            let failure = format!("donor {donor} timed out after {:?}", self.cfg.donor_timeout);
            out.extend(self.next_donor(learner_id, failure, candidates, now));
        }
        out
    }

    pub fn next_deadline(&self) -> Option<Instant> {
        self.pending
            .values()
            .filter_map(|p| match p.phase {
                Phase::AwaitSnapshot { deadline, .. } => Some(deadline),
                Phase::AwaitRestore { .. } => None,
            })
            .min()
    }

    /// The island restored a source and holds `held` (None: nothing
    /// reliable). Returns the fragments whose current version is newer, in
    /// order; the server resends them and then calls [`Self::finish`].
    pub fn on_restored(&self, from: Member, held: Option<&[u64]>, current: &[u64]) -> Result<Vec<usize>> {
        let pending = self
            .pending
            .get(&from.learner_id)
            .filter(|p| p.member == from)
            .context("RECOVERY_RESTORED from an island that is not recovering")?;
        ensure!(
            matches!(pending.phase, Phase::AwaitRestore { .. }),
            "RECOVERY_RESTORED before the syncer named a recovery source"
        );
        let Some(held) = held else { return Ok((0..current.len()).collect()) };
        ensure!(held.len() == current.len(), "RECOVERY_RESTORED lists {} fragments, layout has {}", held.len(), current.len());
        if let Some(p) = (0..held.len()).find(|&p| held[p] > current[p]) {
            bail!("restored fragment {p} version {} is newer than the syncer version {}", held[p], current[p]);
        }
        Ok((0..current.len()).filter(|&p| current[p] > held[p]).collect())
    }

    /// Complete the recovery of `member` at syncer step `done_step`; the
    /// island holds `versions` once the replay has arrived.
    pub fn finish(
        &mut self,
        member: Member,
        done_step: u64,
        versions: &[u64],
        replayed: usize,
        now: Instant,
    ) -> Option<(RecoveryRecord, Outbound)> {
        if !self.is_pending(member) {
            return None;
        }
        let pending = self.pending.remove(&member.learner_id)?;
        let Phase::AwaitRestore { source } = pending.phase else { return None };
        let steps = done_step.saturating_sub(pending.t_s);
        let record = RecoveryRecord {
            member,
            source,
            t_s: pending.t_s,
            done_step,
            steps,
            bound_steps: self.cfg.bound_steps,
            missed: self.cfg.bound_steps.is_some_and(|bound| steps > bound),
            donor_attempts: pending.attempts,
            replayed,
            seconds: now.saturating_duration_since(pending.started).as_secs_f64(),
            last_failure: pending.last_failure,
        };
        let done = Outbound { to: member, msg_type: MSG_RECOVERY_DONE, payload: encode_done(pending.t_s, done_step, replayed, versions) };
        Some((record, done))
    }
}

/// Degraded source and DONE for an island whose recovery cannot run
/// (the syncer is not in its training loop); the caller puts it on the
/// roster and the phase's own catch-up sends the parameters.
pub fn degraded_outside_training(member: Member, fragments: usize, reason: &str) -> Vec<Outbound> {
    vec![
        Outbound {
            to: member,
            msg_type: MSG_RECOVERY_SOURCE,
            payload: encode_source(SOURCE_DEGRADED, None, &vec![0; fragments], &[0; 32], reason),
        },
        Outbound { to: member, msg_type: MSG_RECOVERY_DONE, payload: encode_done(0, 0, 0, &vec![0; fragments]) },
    ]
}

impl Pending {
    fn phase_donor(&self) -> Option<Member> {
        match self.phase {
            Phase::AwaitSnapshot { donor, .. } => Some(donor),
            Phase::AwaitRestore { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(learner_id: u32, generation: u64) -> Member {
        Member { learner_id, generation }
    }

    fn c(learner_id: u32, outstanding: usize) -> Candidate {
        Candidate { member: m(learner_id, 100 + learner_id as u64), outstanding }
    }

    fn ready(recovering: Member, versions: Vec<u64>) -> SnapshotReady {
        SnapshotReady { recovering, versions, sha256: [7; 32], locator: "store://snap-1".into() }
    }

    fn cfg(tries: u32) -> RecoveryConfig {
        RecoveryConfig { bound_steps: Some(4), max_donor_tries: tries, donor_timeout: Duration::from_secs(10) }
    }

    #[test]
    fn hello_ext_round_trips_and_legacy_bodies_pass_through() {
        for ext in [
            HelloExt::default(),
            HelloExt { syncer_epoch: Some(9), recover: false },
            HelloExt { syncer_epoch: None, recover: true },
            HelloExt { syncer_epoch: Some(u64::MAX), recover: true },
        ] {
            let mut body = b"legacy-hello".to_vec();
            body.extend_from_slice(&ext.encode());
            let (head, parsed) = HelloExt::split(&body);
            assert_eq!(head, b"legacy-hello");
            assert_eq!(parsed, Some(ext));
        }
        assert_eq!(HelloExt::split(b"legacy"), (&b"legacy"[..], None));
        let mut unknown_flag = b"x".to_vec();
        unknown_flag.extend_from_slice(&HelloExt::default().encode());
        unknown_flag[5] = 0x80;
        assert_eq!(HelloExt::split(&unknown_flag).1, None);
    }

    #[test]
    fn codec_round_trips() {
        let r = ready(m(2, 5), vec![3, 4]);
        assert_eq!(SnapshotReady::decode(&r.encode(), 2).unwrap(), r);
        assert!(SnapshotReady::decode(&r.encode(), 3).is_err(), "fragment count is checked");
        let mut long = r.encode();
        long.push(0);
        assert!(SnapshotReady::decode(&long, 2).is_err());
        assert_eq!(decode_restored(&encode_restored(Some(&[1, 2])), 2).unwrap(), Some(vec![1, 2]));
        assert_eq!(decode_restored(&encode_restored(None), 2).unwrap(), None);
        assert_eq!(decode_failed(&encode_failed("sha256 mismatch")).unwrap(), "sha256 mismatch");
    }

    #[test]
    fn peer_path_picks_least_busy_donor_and_replays_only_newer_fragments() {
        let t0 = Instant::now();
        let mut rec = PeerRecovery::new(cfg(3));
        let me = m(1, 77);
        let out = rec.begin(me, 10, &[9, 10], &[c(0, 2), c(2, 0), c(3, 0)], t0);
        assert_eq!(out.len(), 2);
        assert_eq!((out[0].to, out[0].msg_type), (me, MSG_RECOVERY_CLOCK));
        assert_eq!((out[1].to, out[1].msg_type), (c(2, 0).member, MSG_PEER_SNAPSHOT_REQ), "fewest pulls, then lowest id");
        assert!(rec.on_restored(me, None, &[9, 10]).is_err(), "no source named yet");
        // A stale answer from another island is ignored.
        assert!(rec.on_snapshot_ready(c(3, 0).member, ready(me, vec![9, 10]), &[], t0).is_empty());
        let out = rec.on_snapshot_ready(c(2, 0).member, ready(me, vec![11, 10]), &[], t0);
        assert_eq!((out[0].to, out[0].msg_type, out[0].payload[0]), (me, MSG_RECOVERY_SOURCE, SOURCE_PEER));
        // Syncer moved on to [11, 12] meanwhile: only fragment 1 is newer than the donor.
        assert_eq!(rec.on_restored(me, Some(&[11, 10]), &[11, 12]).unwrap(), vec![1]);
        assert!(rec.on_restored(me, Some(&[12, 10]), &[11, 12]).is_err(), "island cannot be ahead");
        let (record, done) = rec.finish(me, 13, &[11, 12], 1, t0 + Duration::from_secs(2)).unwrap();
        assert_eq!(done.msg_type, MSG_RECOVERY_DONE);
        assert_eq!((record.steps, record.missed, record.source), (3, false, SourceKind::Peer { donor: c(2, 0).member }));
        assert!(rec.is_empty());
        assert!(record.to_json().contains("\"path\":\"peer\""));
    }

    #[test]
    fn donor_loss_timeout_and_refusal_move_to_next_donor_then_degrade() {
        let t0 = Instant::now();
        let mut rec = PeerRecovery::new(cfg(3));
        let me = m(0, 1);
        let all = [c(1, 0), c(2, 1), c(3, 2)];
        let out = rec.begin(me, 5, &[5], &all, t0);
        assert_eq!(out[1].to, all[0].member);
        // Donor 1 dies.
        let out = rec.on_gone(all[0].member, &all[1..], t0);
        assert_eq!((out[0].to, out[0].msg_type), (all[1].member, MSG_PEER_SNAPSHOT_REQ));
        // Donor 2 times out.
        assert_eq!(rec.next_deadline(), Some(t0 + Duration::from_secs(10)));
        assert!(rec.on_tick(&all, t0 + Duration::from_secs(9)).is_empty());
        let out = rec.on_tick(&all, t0 + Duration::from_secs(10));
        assert_eq!(out[0].to, all[2].member);
        // Donor 3 answers, but the island refuses the snapshot: tries are used up.
        rec.on_snapshot_ready(all[2].member, ready(me, vec![5]), &all, t0);
        let out = rec.on_failed(me, "sha256 mismatch".into(), &all, t0);
        assert_eq!((out[0].to, out[0].msg_type, out[0].payload[0]), (me, MSG_RECOVERY_SOURCE, SOURCE_DEGRADED));
        assert!(rec.on_failed(me, "again".into(), &all, t0).is_empty(), "degraded is final");
        assert_eq!(rec.on_restored(me, None, &[8]).unwrap(), vec![0]);
        let (record, _) = rec.finish(me, 10, &[8], 1, t0).unwrap();
        assert_eq!((record.source, record.donor_attempts, record.steps, record.missed), (SourceKind::Degraded, 3, 5, true));
        assert_eq!(record.last_failure.as_deref(), Some("sha256 mismatch"));
    }

    #[test]
    fn no_donor_is_degraded_at_once_and_a_lagging_donor_is_skipped() {
        let t0 = Instant::now();
        let mut rec = PeerRecovery::new(cfg(3));
        let out = rec.begin(m(0, 1), 3, &[3, 2], &[], t0);
        assert_eq!(out[1].payload[0], SOURCE_DEGRADED);
        let out = rec.begin(m(1, 1), 3, &[3, 2], &[c(2, 0), c(3, 0)], t0);
        assert_eq!(out[1].to, c(2, 0).member);
        // Donor 2 answers with a version behind the clock: next donor.
        let out = rec.on_snapshot_ready(c(2, 0).member, ready(m(1, 1), vec![3, 1]), &[c(2, 0), c(3, 0)], t0);
        assert_eq!((out[0].to, out[0].msg_type), (c(3, 0).member, MSG_PEER_SNAPSHOT_REQ));
        // An island that is itself recovering is never a donor.
        let out = rec.begin(m(4, 1), 3, &[3, 2], &[Candidate { member: m(1, 1), outstanding: 0 }], t0);
        assert_eq!(out[1].payload[0], SOURCE_DEGRADED);
    }

    #[test]
    fn a_new_generation_restarts_and_the_old_one_is_forgotten() {
        let t0 = Instant::now();
        let mut rec = PeerRecovery::new(cfg(3));
        rec.begin(m(0, 1), 3, &[3], &[c(1, 0)], t0);
        rec.begin(m(0, 2), 4, &[4], &[c(1, 0)], t0);
        assert!(!rec.is_pending(m(0, 1)) && rec.is_pending(m(0, 2)));
        assert!(rec.on_snapshot_ready(c(1, 0).member, ready(m(0, 1), vec![4]), &[], t0).is_empty(), "stale generation");
        rec.on_gone(m(0, 1), &[], t0);
        assert!(rec.is_pending(m(0, 2)), "an older generation leaving does not cancel");
        rec.on_gone(m(0, 2), &[], t0);
        assert!(rec.is_empty());
    }
}
