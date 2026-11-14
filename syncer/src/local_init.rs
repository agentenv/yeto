//! Local-init startup (`MSG_INIT_DIGEST` / `MSG_INIT_CONFIRM`).
//!
//! Every island loads the same base checkpoint, so all of them already hold
//! the version-0 cut. Instead of learner 0's INIT followed by a full
//! broadcast to every island, each island reports the SHA-256 of every
//! fragment's INIT_PARAMS tensor bytes. The syncer still initializes its f32
//! masters from learner 0's INIT (so merges, checkpoints and resume are
//! unchanged); the applied INIT is the reference.
//!
//! An island whose digests equal the applied INIT receives one 32-byte
//! INIT_CONFIRM instead of the initial broadcast; an island that did not
//! report digests gets the broadcast.
//!
//! By default (strict) any disagreement is fatal: every island must report
//! the same digests, and the applied INIT must hash to them.
//!
//! With `--local-init-fallback` a disagreeing island instead gets the
//! broadcast of the applied INIT: a different framework, card or dtype path
//! may canonicalize the same base to different bits, and the broadcast puts
//! it on the syncer's version 0 all the same. Each disagreeing island gets
//! one summary warning (`init_digest_fallback_upload` on the event tape). A different digest
//! layout or fragment range stays fatal: a broadcast cannot fix a layout.

use std::collections::BTreeMap;
use std::ops::Range;

use anyhow::{bail, Result};
use sha2::{Digest, Sha256};

use crate::protocol::Reader;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InitDigests {
    pub learner_id: u32,
    pub layout_hash: [u8; 32],
    pub first: u32,
    pub digests: Vec<[u8; 32]>,
}

fn hex(h: &[u8; 32]) -> String {
    h.iter().map(|b| format!("{b:02x}")).collect()
}

fn read32(r: &mut Reader<'_>) -> Result<[u8; 32]> {
    let mut out = [0u8; 32];
    for byte in &mut out {
        *byte = r.u8()?;
    }
    Ok(out)
}

impl InitDigests {
    pub fn decode(payload: &[u8], connected_learner: u32) -> Result<Self> {
        let mut r = Reader(payload);
        let learner_id = r.u32()?;
        if learner_id != connected_learner {
            bail!("INIT_DIGEST learner id {learner_id} does not match connected learner {connected_learner}");
        }
        let layout_hash = read32(&mut r)?;
        let first = r.u32()?;
        let count = r.u32()? as usize;
        if count == 0 || r.remaining() != 32 * count {
            bail!("INIT_DIGEST declares {count} digest(s) but carries {} byte(s)", r.remaining());
        }
        let digests = (0..count).map(|_| read32(&mut r)).collect::<Result<Vec<_>>>()?;
        Ok(Self { learner_id, layout_hash, first, digests })
    }

    pub fn range(&self) -> Range<usize> {
        self.first as usize..self.first as usize + self.digests.len()
    }

    /// INIT_CONFIRM payload: SHA-256 over the concatenated digests.
    #[cfg(test)]
    pub fn combined(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        for digest in &self.digests {
            hasher.update(digest);
        }
        hasher.finalize().into()
    }
}

/// Startup bookkeeping of the local-init check (one per syncer shard).
pub struct LocalInit {
    owned: Range<usize>,
    /// Default (no `--local-init-fallback`): any digest disagreement is an error.
    strict: bool,
    /// The first island's report: the layout every island must share and,
    /// strict, the digests every island must agree on.
    agreed: Option<InitDigests>,
    reported: BTreeMap<u32, InitDigests>,
    /// SHA-256 of each applied INIT (index = fragment id).
    applied: Vec<Option<[u8; 32]>>,
}

/// What a digest report or applied INIT changed: `Ok(None)` agrees,
/// `Ok(Some(warning))` is a lenient disagreement (the island falls back to
/// the broadcast), `Err` is fatal.
pub type Checked = Result<Option<String>>;

impl LocalInit {
    pub fn new(owned: Range<usize>, fragments: usize, strict: bool) -> Self {
        Self { owned, strict, agreed: None, reported: BTreeMap::new(), applied: vec![None; fragments] }
    }

    /// Record one island's digests.
    pub fn on_digests(&mut self, digests: InitDigests) -> Checked {
        if digests.range() != self.owned {
            bail!(
                "init digest mismatch: learner {} reported fragments {:?}, this syncer owns {:?}",
                digests.learner_id,
                digests.range(),
                self.owned
            );
        }
        if let Some(agreed) = &self.agreed {
            if agreed.layout_hash != digests.layout_hash {
                bail!(
                    "init digest mismatch: learner {} layout hash {} differs from learner {} \
                     layout hash {}; every island must build the same fragment layout",
                    digests.learner_id,
                    hex(&digests.layout_hash),
                    agreed.learner_id,
                    hex(&agreed.layout_hash)
                );
            }
        }
        let mut problems = Vec::new();
        for (offset, digest) in digests.digests.iter().enumerate() {
            let fid = self.owned.start + offset;
            if let Some(applied) = &self.applied[fid] {
                problems.extend(check(fid, digests.learner_id, digest, "the applied INIT_PARAMS", applied));
            }
        }
        if let Some(agreed) = &self.agreed {
            for (offset, (mine, theirs)) in digests.digests.iter().zip(&agreed.digests).enumerate() {
                let other = format!("learner {}", agreed.learner_id);
                problems.extend(check(self.owned.start + offset, digests.learner_id, mine, &other, theirs));
            }
        }
        if self.agreed.is_none() {
            self.agreed = Some(digests.clone());
        }
        self.reported.insert(digests.learner_id, digests);
        self.verdict(problems)
    }

    /// Record the hash of an applied INIT.
    pub fn on_init(&mut self, fid: usize, learner_id: u32, sha256: [u8; 32]) -> Checked {
        let mut problems = Vec::new();
        if let Some(slot) = self.owned.contains(&fid).then(|| fid - self.owned.start) {
            for reported in self.reported.values() {
                let other = format!("the digest of learner {}", reported.learner_id);
                problems.extend(check(fid, learner_id, &sha256, &other, &reported.digests[slot]));
                if self.strict {
                    break; // strict: every report equals the first one
                }
            }
        }
        if let Some(slot) = self.applied.get_mut(fid) {
            slot.get_or_insert(sha256);
        }
        self.verdict(problems)
    }

    fn verdict(&self, problems: Vec<String>) -> Checked {
        if problems.is_empty() {
            return Ok(None);
        }
        let message = problems.join("; ");
        if self.strict {
            bail!(
                "{message}; every island must load the same base checkpoint (or start with \
                 --rl-init-mode upload)"
            );
        }
        Ok(Some(format!(
            "{message}; the island falls back to the syncer's broadcast of learner 0's INIT \
             (without --local-init-fallback this is fatal)"
        )))
    }

    /// Whether `learner_id` reported digests equal to the applied INIT of
    /// every fragment of this shard (it gets INIT_CONFIRM).
    pub fn holds(&self, learner_id: u32) -> bool {
        self.reported.get(&learner_id).is_some_and(|reported| {
            self.owned
                .clone()
                .all(|fid| self.applied[fid].as_ref() == reported.digests.get(fid - self.owned.start))
        })
    }

    /// Islands that reported digests but do not hold the applied INIT: they
    /// fall back to the initial broadcast.
    pub fn fallbacks(&self) -> Vec<u32> {
        self.reported.keys().copied().filter(|learner| !self.holds(*learner)).collect()
    }

    /// How many fragments of this shard `learner_id` reported with a digest
    /// other than the applied INIT (the per-island fallback summary).
    pub fn mismatched_fragments(&self, learner_id: u32) -> usize {
        self.reported.get(&learner_id).map_or(0, |reported| {
            self.owned
                .clone()
                .filter(|fid| self.applied[*fid].as_ref() != reported.digests.get(fid - self.owned.start))
                .count()
        })
    }

    /// The INIT_CONFIRM payload (SHA-256 over the applied INIT digests), once
    /// an island reported digests and every INIT of this shard is applied;
    /// `None` means the initial broadcast is needed.
    pub fn confirm(&self) -> Option<[u8; 32]> {
        self.agreed.as_ref()?;
        let mut hasher = Sha256::new();
        for fid in self.owned.clone() {
            hasher.update(self.applied[fid].as_ref()?);
        }
        Some(hasher.finalize().into())
    }
}

fn check(fid: usize, learner: u32, mine: &[u8; 32], other: &str, theirs: &[u8; 32]) -> Option<String> {
    (mine != theirs).then(|| {
        format!(
            "init digest mismatch: fragment {fid} of learner {learner} has sha256 {} but {other} \
             has {}",
            hex(mine),
            hex(theirs)
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digests(learner_id: u32, values: &[u8]) -> InitDigests {
        InitDigests {
            learner_id,
            layout_hash: [7; 32],
            first: 0,
            digests: values.iter().map(|v| [*v; 32]).collect(),
        }
    }

    #[test]
    fn agreement_and_init_give_a_confirm() {
        for strict in [false, true] {
            let mut init = LocalInit::new(0..2, 2, strict);
            assert_eq!(init.on_digests(digests(0, &[1, 2])).unwrap(), None);
            assert_eq!(init.on_digests(digests(1, &[1, 2])).unwrap(), None);
            assert!(init.confirm().is_none(), "no INIT applied yet");
            assert_eq!(init.on_init(0, 0, [1; 32]).unwrap(), None);
            assert_eq!(init.on_init(1, 0, [2; 32]).unwrap(), None);
            // The INIT_CONFIRM payload is unchanged: SHA-256 over the digests.
            assert_eq!(init.confirm(), Some(digests(0, &[1, 2]).combined()));
            assert!(init.holds(0) && init.holds(1) && !init.holds(2));
            assert!(init.fallbacks().is_empty());
        }
    }

    #[test]
    fn strict_mismatches_are_refused() {
        let mut init = LocalInit::new(0..2, 2, true);
        init.on_digests(digests(0, &[1, 2])).unwrap();
        let error = init.on_digests(digests(1, &[1, 3])).unwrap_err().to_string();
        assert!(error.contains("fragment 1 of learner 1"), "{error}");
        assert!(error.contains("same base checkpoint"), "{error}");
        assert!(error.contains("--rl-init-mode upload"), "{error}");
        let mut other_layout = digests(1, &[1, 2]);
        other_layout.layout_hash = [8; 32];
        assert!(init.on_digests(other_layout).unwrap_err().to_string().contains("layout hash"));
        assert!(init.on_init(0, 0, [9; 32]).is_err());
        // An INIT applied before any digest is checked when digests arrive.
        let mut late = LocalInit::new(0..2, 2, true);
        late.on_init(0, 0, [5; 32]).unwrap();
        assert!(late.on_digests(digests(1, &[1, 2])).is_err());
    }

    #[test]
    fn lenient_disagreeing_island_falls_back_to_the_broadcast() {
        let mut init = LocalInit::new(0..2, 2, false);
        assert_eq!(init.on_digests(digests(0, &[1, 2])).unwrap(), None);
        let warning = init.on_digests(digests(1, &[1, 3])).unwrap().unwrap();
        assert!(warning.contains("fragment 1 of learner 1"), "{warning}");
        assert!(warning.contains("--local-init-fallback"), "{warning}");
        init.on_init(0, 0, [1; 32]).unwrap();
        let warning = init.on_init(1, 0, [2; 32]).unwrap().unwrap();
        assert!(warning.contains("the digest of learner 1"), "{warning}");
        // Learner 0's INIT is the reference: island 0 keeps its weights,
        // island 1 gets the broadcast.
        assert_eq!(init.confirm(), Some(digests(0, &[1, 2]).combined()));
        assert!(init.holds(0) && !init.holds(1));
        assert_eq!(init.fallbacks(), vec![1]);
    }

    #[test]
    fn lenient_init_that_differs_from_every_digest_broadcasts_to_all() {
        // The islands agree with each other but learner 0 uploads other bits
        // (e.g. a different conversion path on its card): the INIT wins.
        let mut init = LocalInit::new(0..2, 2, false);
        init.on_digests(digests(0, &[1, 2])).unwrap();
        init.on_digests(digests(1, &[1, 2])).unwrap();
        init.on_init(0, 0, [1; 32]).unwrap();
        assert!(init.on_init(1, 0, [9; 32]).unwrap().is_some());
        let confirm = init.confirm().unwrap();
        assert_eq!(confirm, digests(0, &[1, 9]).combined());
        assert!(!init.holds(0) && !init.holds(1));
        assert_eq!(init.fallbacks(), vec![0, 1]);
        // An INIT applied before the digests arrive is checked against them.
        let mut late = LocalInit::new(0..2, 2, false);
        late.on_init(0, 0, [5; 32]).unwrap();
        assert!(late.on_digests(digests(1, &[1, 2])).unwrap().is_some());
    }

    #[test]
    fn lenient_still_refuses_another_layout_or_range() {
        let mut init = LocalInit::new(0..2, 2, false);
        init.on_digests(digests(0, &[1, 2])).unwrap();
        let mut other_layout = digests(1, &[1, 2]);
        other_layout.layout_hash = [8; 32];
        assert!(init.on_digests(other_layout).unwrap_err().to_string().contains("layout hash"));
        let mut shifted = digests(1, &[1, 2]);
        shifted.first = 1;
        assert!(init.on_digests(shifted).is_err());
    }

    #[test]
    fn no_digests_means_no_confirm() {
        let mut init = LocalInit::new(0..1, 1, false);
        init.on_init(0, 0, [1; 32]).unwrap();
        assert!(init.confirm().is_none());
        assert!(init.fallbacks().is_empty());
    }
}
