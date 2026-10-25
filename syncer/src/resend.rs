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
//! [`Pending`] is the structure of the G-EA receive-only subscribers
//! (`subscriber.rs` on s27-spot-rollout, same API and rules): pending
//! entries coalesce per fragment, the newest version wins, and a version
//! already handed to the forwarder is not queued again. Once both branches
//! are merged, the subscriber module can use this one.

use std::collections::BTreeMap;
use std::sync::Arc;

/// Fragment versions waiting to be sent to one connection.
#[derive(Default)]
pub struct Pending {
    entries: BTreeMap<usize, (u64, Arc<Vec<f32>>)>,
    /// Newest version already handed to the forwarder, per fragment.
    sent: Vec<Option<u64>>,
    pub offered: u64,
    pub coalesced: u64,
}

impl Pending {
    pub fn new(num_fragments: usize) -> Self {
        Pending { sent: vec![None; num_fragments], ..Pending::default() }
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

    /// The pending entry with the oldest version (ties: lowest fragment).
    pub fn pop(&mut self) -> Option<(usize, u64, Arc<Vec<f32>>)> {
        let p = self
            .entries
            .iter()
            .min_by_key(|(p, (version, _))| (*version, **p))
            .map(|(p, _)| *p)?;
        let (version, values) = self.entries.remove(&p)?;
        self.sent[p] = Some(version);
        Some((p, version, values))
    }

    #[allow(dead_code)] // same API as the G-EA subscriber map
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

    #[test]
    fn prefix_is_fragment_then_version() {
        let prefix = bcast_prefix(5, 0x0102);
        assert_eq!(u32::from_le_bytes(prefix[0..4].try_into().unwrap()), 5);
        assert_eq!(u64::from_le_bytes(prefix[4..12].try_into().unwrap()), 0x0102);
    }
}
