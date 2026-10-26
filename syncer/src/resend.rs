//! Bulk resends off the scheduler (S27 G-EL).
//!
//! A rejoin catch-up (reconnect, pool re-JOIN after a lease expiry) and the
//! replay after a peer or degraded recovery send a learner the current
//! value of many fragments: the whole model on the degraded path (126 GB of
//! f32 for Nemotron Nano). The scheduler used to send them itself, one
//! fragment after the other (the recovery replay) or one fragment per pass
//! (the catch-up), and each send waits for the learner's link. In run
//! s27-elastic-a the degraded replay held the scheduler for 8.8 min: no
//! heartbeat was read, both islands' 150 s leases expired, and both rejoined
//! with another full catch-up.
//!
//! Now the scheduler only offers fragment versions to a per-learner
//! [`Pending`] map (no await) and a forwarder task per learner sends them at
//! the link's pace. Merges keep offering every newly published version to
//! the learners still being resent, so when the map is empty and nothing is
//! in flight the learner holds the current value of every fragment it was
//! offered; then the scheduler runs the tail of the catch-up or recovery.
//!
//! [`Pending`] is also the structure of the G-EA receive-only subscribers
//! (`subscriber.rs` re-exports it): pending entries coalesce per fragment,
//! the newest version wins, and a version already handed to the forwarder
//! is not queued again.
//!
//! Tail condition (S27 G-EY). Run s27-elastic-c (Nemotron Nano) showed that
//! "nothing pending" is the wrong condition for the tail: with one island
//! merging, new versions were published at about 12 GB/min and the link to
//! the recovering island took about 13.5 GB/min, so the map almost never
//! emptied (16.6 min, about 210 GB sent for a 126 GB replay, no
//! RECOVERY_DONE). Now the fragment versions a resend owes when it starts
//! are *targets*: the tail runs once every target is handed over at that
//! version or a newer one. Versions published later still go out on the
//! same forwarder (newest wins, oldest first) until nothing is pending;
//! only then do publications go inline again, so a learner never has two
//! senders. Entries with a target are popped first, so the tail needs at
//! most one send per target however fast merges publish.

use std::collections::BTreeMap;
use std::sync::Arc;

/// Fragment versions waiting to be sent to one connection.
#[derive(Default)]
pub struct Pending {
    entries: BTreeMap<usize, (u64, Arc<Vec<f32>>)>,
    /// Newest version already handed to the forwarder, per fragment.
    sent: Vec<Option<u64>>,
    /// Version each fragment must be handed over at (or newer) before the
    /// owner's tail runs; None: no target (subscribers never set one).
    targets: Vec<Option<u64>>,
    /// Number of `Some` entries in `targets`.
    unmet: usize,
    pub offered: u64,
    pub coalesced: u64,
}

impl Pending {
    pub fn new(num_fragments: usize) -> Self {
        Pending { sent: vec![None; num_fragments], targets: vec![None; num_fragments], ..Pending::default() }
    }

    /// Make every pending entry a target at its version: the owner's tail
    /// waits until each of them (or a newer version of its fragment) is
    /// handed over, and they are sent before entries without a target.
    pub fn require_pending(&mut self) {
        for (&p, &(version, _)) in &self.entries {
            match &mut self.targets[p] {
                Some(target) => *target = (*target).max(version),
                slot @ None => {
                    *slot = Some(version);
                    self.unmet += 1;
                }
            }
        }
    }

    /// Number of fragments whose target is not handed over yet.
    pub fn unmet(&self) -> usize {
        self.unmet
    }

    /// A version of fragment `p` waits to be sent.
    pub fn is_pending(&self, p: usize) -> bool {
        self.entries.contains_key(&p)
    }

    /// Newest version of fragment `p` already handed to the forwarder.
    pub fn sent(&self, p: usize) -> Option<u64> {
        self.sent.get(p).copied().flatten()
    }

    /// Offer version `version` of fragment `p`. Ignored when this or a newer
    /// version was already sent or is already pending; replaces an older
    /// pending version (counted as coalesced). Returns true when queued.
    pub fn offer(&mut self, p: usize, version: u64, values: Arc<Vec<f32>>) -> bool {
        if p >= self.sent.len() {
            return false;
        }
        if matches!(self.sent[p], Some(sent) if sent >= version) {
            return false;
        }
        if let Some((pending, _)) = self.entries.get(&p) {
            if *pending >= version {
                return false;
            }
            self.coalesced += 1;
        }
        self.entries.insert(p, (version, values));
        self.offered += 1;
        true
    }

    /// The next entry to send: entries with a target first, then the rest;
    /// within each, the oldest version (ties: lowest fragment). Without
    /// targets this is the oldest pending version.
    pub fn pop(&mut self) -> Option<(usize, u64, Arc<Vec<f32>>)> {
        let targets = &self.targets;
        let p = self
            .entries
            .iter()
            .min_by_key(|(p, (version, _))| (targets[**p].is_none(), *version, **p))
            .map(|(p, _)| *p)?;
        let (version, values) = self.entries.remove(&p)?;
        self.sent[p] = Some(version);
        // An entry is never older than its fragment's target: targets are
        // taken from pending entries, and an entry only gets newer.
        if matches!(self.targets[p], Some(target) if version >= target) {
            self.targets[p] = None;
            self.unmet -= 1;
        }
        Some((p, version, values))
    }

    /// S27 G-FH: version `version` of fragment `p` was handed over outside
    /// this map (an initial-cut frame the forwarder sent first): an older or
    /// equal offer is then ignored, as for a popped entry.
    pub fn mark_sent(&mut self, p: usize, version: u64) {
        if let Some(sent) = self.sent.get_mut(p) {
            *sent = Some(sent.map_or(version, |sent| sent.max(version)));
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// BCAST_FRAGMENT prefix without a downlink anchor: `fragment u32 | version u64`.
pub fn bcast_prefix(p: usize, version: u64) -> [u8; 12] {
    let mut out = [0u8; 12];
    out[0..4].copy_from_slice(&(p as u32).to_le_bytes());
    out[4..12].copy_from_slice(&version.to_le_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f32) -> Arc<Vec<f32>> {
        Arc::new(vec![x; 2])
    }

    #[test]
    fn newest_version_wins_and_sent_versions_are_not_resent() {
        let mut pending = Pending::new(3);
        assert!(pending.offer(0, 1, v(1.0)));
        assert!(pending.offer(1, 2, v(2.0)));
        // A newer version of fragment 0 replaces the pending one.
        assert!(pending.offer(0, 4, v(4.0)));
        assert_eq!(pending.coalesced, 1);
        // An older or equal version is ignored.
        assert!(!pending.offer(0, 3, v(3.0)));
        assert!(!pending.offer(0, 4, v(4.0)));
        // Out-of-layout fragments are ignored.
        assert!(!pending.offer(7, 9, v(9.0)));
        // Oldest version first.
        let (p, version, values) = pending.pop().unwrap();
        assert_eq!((p, version, values[0]), (1, 2, 2.0));
        let (p, version, _) = pending.pop().unwrap();
        assert_eq!((p, version), (0, 4));
        assert!(pending.pop().is_none() && pending.is_empty());
        // Already sent: the same version is not queued again, a newer one is.
        assert!(!pending.offer(0, 4, v(4.0)));
        assert!(pending.offer(0, 5, v(5.0)));
        // Version 0 (initial values) is accepted before anything was sent.
        assert!(pending.offer(2, 0, v(0.0)));
        assert_eq!(pending.len(), 2);
    }

    /// S27 G-EY: the versions owed at the start are targets. They go out
    /// first even when merges republish their fragments meanwhile, and the
    /// owner sees them all handed over while newer versions still wait.
    #[test]
    fn targets_go_first_and_are_met_by_a_newer_version() {
        let mut pending = Pending::new(4);
        for p in 0..3 {
            assert!(pending.offer(p, 10 + p as u64, v(p as f32)));
        }
        pending.require_pending();
        assert_eq!(pending.unmet(), 3);
        // Fragment 0 is sent at its target version.
        assert_eq!(pending.pop().unwrap().0, 0);
        assert_eq!(pending.unmet(), 2);
        // Merges publish fragment 0 again (no target any more), fragment 1
        // (its pending target entry gets the newer version) and fragment 3
        // (not part of the resend).
        assert!(pending.offer(0, 13, v(0.5)));
        assert!(pending.offer(1, 14, v(1.5)));
        assert!(pending.offer(3, 15, v(3.0)));
        // Targets first: fragment 2 (version 12), then fragment 1 (now 14),
        // although fragment 0's version 13 is older than 14.
        assert_eq!(pending.pop().unwrap().0, 2);
        let (p, version, _) = pending.pop().unwrap();
        assert_eq!((p, version, pending.unmet()), (1, 14, 0));
        // Every target is handed over; newer versions are still pending.
        assert_eq!(pending.len(), 2);
        // Re-arming on the remaining entries (the recovery tail does this).
        pending.require_pending();
        assert_eq!(pending.unmet(), 2);
        assert_eq!(pending.pop().unwrap().0, 0);
        assert!(pending.offer(2, 16, v(2.5)));
        assert_eq!(pending.pop().unwrap().0, 3);
        assert_eq!(pending.unmet(), 0);
        assert_eq!(pending.pop().unwrap().0, 2);
        assert!(pending.is_empty());
    }

    #[test]
    fn prefix_is_fragment_then_version() {
        let prefix = bcast_prefix(5, 0x0102);
        assert_eq!(u32::from_le_bytes(prefix[0..4].try_into().unwrap()), 5);
        assert_eq!(u64::from_le_bytes(prefix[4..12].try_into().unwrap()), 0x0102);
    }
}
