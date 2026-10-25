//! Island process members with a lease (S27 P4): who is live, which
//! incarnation of an island is current, and which incarnations are retired
//! (fenced). Shared by every service of the syncer that keeps per-process
//! state for islands outside the learner session: the task dispenser
//! (`dispense.rs`) today, the spot rollout subscribers and sample relay
//! (S27 G-EA) next, so that one sampler process has one membership and one
//! lease whatever channel it talks on.
//!
//! Rules (docs/PROTOCOL.md "Sampler members"):
//! - a member is `(id, incarnation)`; `id` is the island's learner id, or
//!   for a sampler island `SAMPLER_MEMBER_BIT | 31 bits of SHA-256 of its
//!   sampler id` (`yeto.rl.engine.task_dispense.sampler_holder_id`); the
//!   incarnation is a random 64-bit value per island process;
//! - a new incarnation of an id supersedes the current one at once;
//! - a member that sends nothing on any channel for the lease expires;
//! - a superseded or expired member is retired: every later request of it is
//!   refused (fenced), and the caller returns what it held.
//!
//! Pure state machine: no sockets, no files, clocks passed in.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

/// High bit of a member id that marks a sampler island (the same bit as the
/// spot rollout subscribers' `SUBSCRIBER_ID_BIT`).
#[cfg_attr(not(test), allow(dead_code))] // the island side derives the ids
pub const SAMPLER_MEMBER_BIT: u32 = 0x8000_0000;
/// Retired incarnations remembered (fencing).
pub const RETIRED_LIMIT: usize = 4096;

/// One island process.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Holder {
    pub learner_id: u32,
    pub incarnation: u64,
}

/// The outcome of [`Members::join`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Join {
    /// A new member; `superseded` is the id's previous live incarnation (now retired).
    Joined { superseded: Option<Holder> },
    /// The same incarnation again (a reconnect): it keeps everything.
    Reconnected,
}

#[derive(Debug)]
pub struct Members {
    lease: Duration,
    last_seen: BTreeMap<Holder, Instant>,
    /// Live incarnation per id.
    current: HashMap<u32, u64>,
    retired: (VecDeque<Holder>, HashSet<Holder>),
}

impl Members {
    pub fn new(lease: Duration) -> Self {
        Self { lease, last_seen: BTreeMap::new(), current: HashMap::new(), retired: (VecDeque::new(), HashSet::new()) }
    }

    #[cfg_attr(not(test), allow(dead_code))] // the spot rollout services
    pub fn lease(&self) -> Duration {
        self.lease
    }

    #[cfg_attr(not(test), allow(dead_code))] // the spot rollout services
    pub fn is_live(&self, holder: Holder) -> bool {
        self.last_seen.contains_key(&holder)
    }

    pub fn is_retired(&self, holder: Holder) -> bool {
        self.retired.1.contains(&holder)
    }

    #[cfg_attr(not(test), allow(dead_code))] // the spot rollout services
    pub fn len(&self) -> usize {
        self.last_seen.len()
    }

    #[cfg_attr(not(test), allow(dead_code))] // the spot rollout services
    pub fn current(&self, learner_id: u32) -> Option<Holder> {
        self.current.get(&learner_id).map(|&incarnation| Holder { learner_id, incarnation })
    }

    /// Admit `holder` (the caller checked [`Members::is_retired`] first). A
    /// new incarnation retires the id's current one; the caller returns what
    /// that one held.
    pub fn join(&mut self, holder: Holder, now: Instant) -> Join {
        if let Some(seen) = self.last_seen.get_mut(&holder) {
            *seen = now;
            return Join::Reconnected;
        }
        let superseded = self.current(holder.learner_id);
        if let Some(old) = superseded {
            self.retire(old);
        }
        self.current.insert(holder.learner_id, holder.incarnation);
        self.last_seen.insert(holder, now);
        Join::Joined { superseded }
    }

    /// Renew the lease of a live member; false when it is not live.
    pub fn touch(&mut self, holder: Holder, now: Instant) -> bool {
        match self.last_seen.get_mut(&holder) {
            Some(seen) => {
                *seen = now;
                true
            }
            None => false,
        }
    }

    /// Retire `holder` (superseded, expired, left, or from before a restart).
    pub fn retire(&mut self, holder: Holder) {
        self.last_seen.remove(&holder);
        if self.current.get(&holder.learner_id) == Some(&holder.incarnation) {
            self.current.remove(&holder.learner_id);
        }
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

    /// The live members silent for the lease, in order (not retired yet: the
    /// caller returns what they hold, then retires them).
    pub fn expired(&self, now: Instant) -> Vec<Holder> {
        self.last_seen
            .iter()
            .filter(|(_, &seen)| now.saturating_duration_since(seen) >= self.lease)
            .map(|(&holder, _)| holder)
            .collect()
    }

    #[cfg_attr(not(test), allow(dead_code))] // tests and offline tools
    pub fn next_expiry(&self) -> Option<Instant> {
        self.last_seen.values().map(|&seen| seen + self.lease).min()
    }

    /// The retired incarnations (oldest first), then the live ones: what a
    /// restart must fence.
    pub fn fence_list(&self) -> Vec<Holder> {
        let mut out: Vec<Holder> = self.retired.0.iter().copied().collect();
        out.extend(self.last_seen.keys().copied().filter(|h| !self.retired.1.contains(h)));
        if out.len() > RETIRED_LIMIT {
            out.drain(..out.len() - RETIRED_LIMIT);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(id: u32, inc: u64) -> Holder {
        Holder { learner_id: id, incarnation: inc }
    }

    #[test]
    fn a_new_incarnation_supersedes_and_a_silent_member_expires() {
        let t = Instant::now();
        let mut m = Members::new(Duration::from_secs(10));
        let sampler = SAMPLER_MEMBER_BIT | 7;
        assert_eq!(m.join(h(sampler, 1), t), Join::Joined { superseded: None });
        assert_eq!(m.join(h(sampler, 1), t), Join::Reconnected);
        assert_eq!(m.join(h(sampler, 2), t), Join::Joined { superseded: Some(h(sampler, 1)) });
        assert!(m.is_retired(h(sampler, 1)) && !m.touch(h(sampler, 1), t));
        assert_eq!(m.current(sampler), Some(h(sampler, 2)));
        m.join(h(0, 5), t);
        assert!(m.touch(h(0, 5), t + Duration::from_secs(8)));
        assert_eq!(m.expired(t + Duration::from_secs(12)), vec![h(sampler, 2)]);
        assert_eq!(m.next_expiry(), Some(t + Duration::from_secs(10)));
        m.retire(h(sampler, 2));
        assert_eq!(m.current(sampler), None);
        assert_eq!(m.fence_list(), vec![h(sampler, 1), h(sampler, 2), h(0, 5)]);
        assert_eq!(m.len(), 1);
    }
}
