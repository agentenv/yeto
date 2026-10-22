//! Local-init startup (S27, `MSG_INIT_DIGEST` / `MSG_INIT_CONFIRM`).
//!
//! Every island loads the same base checkpoint, so all of them already hold
//! the version-0 cut. Instead of learner 0's INIT followed by a full
//! broadcast to every island, each island reports the SHA-256 of every
//! fragment's INIT_PARAMS tensor bytes. The syncer still initializes its f32
//! masters from learner 0's INIT (so merges, checkpoints and resume are
//! unchanged), and checks that
//!
//! * every island reports the same layout hash and fragment digests, and
//! * the INIT bytes it applied hash to those digests.
//!
//! When both hold, an island that reported digests receives one 32-byte
//! INIT_CONFIRM instead of the initial broadcast; an island that did not
//! report digests still gets the broadcast. Any disagreement is fatal: the
//! islands did not start from the same weights.

use std::collections::BTreeSet;
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
    agreed: Option<InitDigests>,
    holders: BTreeSet<u32>,
    /// SHA-256 of each applied INIT (index = fragment id).
    applied: Vec<Option<[u8; 32]>>,
}

impl LocalInit {
    pub fn new(owned: Range<usize>, fragments: usize) -> Self {
        Self { owned, agreed: None, holders: BTreeSet::new(), applied: vec![None; fragments] }
    }

    /// Record one island's digests; an error means the islands disagree.
    pub fn on_digests(&mut self, digests: InitDigests) -> Result<()> {
        if digests.range() != self.owned {
            bail!(
                "init digest mismatch: learner {} reported fragments {:?}, this syncer owns {:?}",
                digests.learner_id,
                digests.range(),
                self.owned
            );
        }
        for (offset, digest) in digests.digests.iter().enumerate() {
            let fid = self.owned.start + offset;
            if let Some(applied) = &self.applied[fid] {
                check(fid, digests.learner_id, digest, "the applied INIT_PARAMS", applied)?;
            }
        }
        match &self.agreed {
            None => self.agreed = Some(digests.clone()),
            Some(agreed) => {
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
                for (offset, (mine, theirs)) in digests.digests.iter().zip(&agreed.digests).enumerate() {
                    let other = format!("learner {}", agreed.learner_id);
                    check(self.owned.start + offset, digests.learner_id, mine, &other, theirs)?;
                }
            }
        }
        self.holders.insert(digests.learner_id);
        Ok(())
    }

    /// Record the hash of an applied INIT; an error means it differs from
    /// the digests the islands agreed on.
    pub fn on_init(&mut self, fid: usize, learner_id: u32, sha256: [u8; 32]) -> Result<()> {
        if let Some(agreed) = &self.agreed {
            if let Some(expected) = agreed.digests.get(fid.wrapping_sub(self.owned.start)) {
                let other = format!("the islands' digest (learner {})", agreed.learner_id);
                check(fid, learner_id, &sha256, &other, expected)?;
            }
        }
        if let Some(slot) = self.applied.get_mut(fid) {
            slot.get_or_insert(sha256);
        }
        Ok(())
    }

    /// Whether `learner_id` reported digests that agree.
    pub fn holds(&self, learner_id: u32) -> bool {
        self.holders.contains(&learner_id)
    }

    /// The INIT_CONFIRM payload, once the agreed digests cover every applied
    /// INIT of this shard; `None` means the initial broadcast is needed.
    pub fn confirm(&self) -> Option<[u8; 32]> {
        let agreed = self.agreed.as_ref()?;
        let complete = self.owned.clone().all(|fid| {
            self.applied[fid].as_ref() == agreed.digests.get(fid - self.owned.start)
        });
        complete.then(|| agreed.combined())
    }
}

fn check(fid: usize, learner: u32, mine: &[u8; 32], other: &str, theirs: &[u8; 32]) -> Result<()> {
    if mine != theirs {
        bail!(
            "init digest mismatch: fragment {fid} of learner {learner} has sha256 {} but {other} \
             has {}; every island must load the same base checkpoint (or start with \
             --rl-init-mode upload)",
            hex(mine),
            hex(theirs)
        );
    }
    Ok(())
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
        let mut init = LocalInit::new(0..2, 2);
        init.on_digests(digests(0, &[1, 2])).unwrap();
        init.on_digests(digests(1, &[1, 2])).unwrap();
        assert!(init.confirm().is_none(), "no INIT applied yet");
        init.on_init(0, 0, [1; 32]).unwrap();
        init.on_init(1, 0, [2; 32]).unwrap();
        assert_eq!(init.confirm(), Some(digests(0, &[1, 2]).combined()));
        assert!(init.holds(1) && !init.holds(2));
    }

    #[test]
    fn mismatches_are_refused() {
        let mut init = LocalInit::new(0..2, 2);
        init.on_digests(digests(0, &[1, 2])).unwrap();
        let error = init.on_digests(digests(1, &[1, 3])).unwrap_err().to_string();
        assert!(error.contains("fragment 1 of learner 1"), "{error}");
        let mut other_layout = digests(1, &[1, 2]);
        other_layout.layout_hash = [8; 32];
        assert!(init.on_digests(other_layout).unwrap_err().to_string().contains("layout hash"));
        assert!(init.on_init(0, 0, [9; 32]).is_err());
        // An INIT applied before any digest is checked when digests arrive.
        let mut late = LocalInit::new(0..2, 2);
        late.on_init(0, 0, [5; 32]).unwrap();
        assert!(late.on_digests(digests(1, &[1, 2])).is_err());
    }
}
