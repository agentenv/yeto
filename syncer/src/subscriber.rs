//! Receive-only subscribers (S27 G-EA): a sampler island on any cloud that
//! follows the global weights without being a learner.
//!
//! A subscriber connects with SUBSCRIBE (instead of HELLO): the same body as
//! HELLO, with `learner_id = SUBSCRIBER_ID_BIT | 31 bits` (the sampler id's
//! SHA-256, the task dispenser's sampler holder id) and dtype bf16, then the
//! trailer `YSI1 | incarnation u64` (the sampler process's member
//! incarnation, inside the MAC); at most `--max-subscribers` are connected
//! at once. Its data streams attach with the ordinary DATA_HELLO under that id.
//! The syncer then sends it `BCAST_FRAGMENT` frames in the bf16 session
//! format (`fragment_id u32 | version u64 | bf16 values`), one per merged
//! fragment version, and nothing else: no PULL, no final cut, no SHUTDOWN.
//!
//! A subscriber is never in the registry's roster or membership: it does not
//! count toward quorum, K, launch quorum or the final cut, and its departure
//! does not reach the scheduler. It is a member of the syncer's shared
//! island member table (`members.rs`, with the task dispenser and the sample
//! relay): its SUBSCRIBE joins it (a new incarnation of the sampler retires
//! the old one on every channel), its HEARTBEATs renew the member lease, and
//! a retired subscriber is sent ERROR `fenced: subscriber ...` and closed. The scheduler only offers
//! fragment versions to each subscriber's pending map (no await), and a
//! per-subscriber forwarder task sends them at the link's pace. Pending
//! entries coalesce per fragment (the newest version wins), so a slow link
//! skips versions instead of slowing the merges, and the head holds at most
//! one pending version per fragment and subscriber (shared `Arc`s of the
//! f32 masters; bf16 bytes are produced while sending).

use std::collections::BTreeMap;
use std::sync::Arc;

/// High bit of the wire learner id that marks a subscriber connection.
pub const SUBSCRIBER_ID_BIT: u32 = 0x8000_0000;
/// Domain of the SUBSCRIBE MAC: HMAC-SHA256(key, domain || body).
pub const SUBSCRIBE_MAC_DOMAIN: &[u8] = b"yeto-subscribe-mac-v1\0";
/// ERROR text sent to every subscriber when the run ends.
pub const RUN_FINISHED: &str = "subscriber: run finished";
/// Prefix of the ERROR that closes a retired subscriber.
pub const FENCED_PREFIX: &str = "fenced: subscriber";
/// Magic of the SUBSCRIBE member incarnation trailer.
pub const INCARNATION_MAGIC: &[u8; 4] = b"YSI1";
const INCARNATION_TRAILER_LEN: usize = 4 + 8;

/// Split the SUBSCRIBE body (MAC already removed) into the HELLO body and
/// the member incarnation of its `YSI1 | incarnation u64` trailer.
pub fn split_incarnation(body: &[u8]) -> anyhow::Result<(&[u8], u64)> {
    let at = body.len().checked_sub(INCARNATION_TRAILER_LEN);
    let Some(at) = at.filter(|&at| &body[at..at + 4] == INCARNATION_MAGIC) else {
        anyhow::bail!("SUBSCRIBE lacks its member incarnation trailer (YSI1)");
    };
    let incarnation = u64::from_le_bytes(body[at + 4..].try_into().unwrap());
    anyhow::ensure!(incarnation != 0, "SUBSCRIBE member incarnation must be nonzero");
    Ok((&body[..at], incarnation))
}

pub fn is_subscriber_id(learner_id: u32) -> bool {
    learner_id & SUBSCRIBER_ID_BIT != 0
}

/// Fragment versions waiting to be sent to one subscriber connection.
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

    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

/// BCAST_FRAGMENT prefix of the bf16 session format.
pub fn bcast_prefix(p: usize, version: u64) -> [u8; 12] {
    let mut out = [0u8; 12];
    out[0..4].copy_from_slice(&(p as u32).to_le_bytes());
    out[4..12].copy_from_slice(&version.to_le_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subscribe_trailer_carries_the_member_incarnation() {
        let mut body = b"hello-body".to_vec();
        body.extend_from_slice(INCARNATION_MAGIC);
        body.extend_from_slice(&0x1234u64.to_le_bytes());
        assert_eq!(split_incarnation(&body).unwrap(), (&b"hello-body"[..], 0x1234));
        assert!(split_incarnation(b"hello-body-without-a-trailer").is_err());
        let mut zero = b"x".to_vec();
        zero.extend_from_slice(INCARNATION_MAGIC);
        zero.extend_from_slice(&0u64.to_le_bytes());
        assert!(split_incarnation(&zero).is_err());
    }

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
        assert!(pending.pop().is_none());
        // Already sent: a snapshot of the same version is not queued again.
        assert!(!pending.offer(0, 4, v(4.0)));
        assert!(pending.offer(0, 5, v(5.0)));
        // Version 0 (initial values) is accepted before anything was sent.
        assert!(pending.offer(2, 0, v(0.0)));
        assert_eq!(pending.len(), 2);
    }

    #[test]
    fn subscriber_ids_and_prefix() {
        assert!(is_subscriber_id(SUBSCRIBER_ID_BIT | 3));
        assert!(!is_subscriber_id(3));
        let prefix = bcast_prefix(5, 0x0102);
        assert_eq!(u32::from_le_bytes(prefix[0..4].try_into().unwrap()), 5);
        assert_eq!(u64::from_le_bytes(prefix[4..12].try_into().unwrap()), 0x0102);
    }
}
