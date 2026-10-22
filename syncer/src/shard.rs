//! Head sharding: one logical syncer split across N processes (S27 P4b).
//!
//! The syncer already works per fragment: every round merges
//! exactly one fragment, and the only cross-fragment state is the global step
//! counter that orders rounds.  The head can therefore be split by fragment
//! ranges.  Shard `i` of `N` owns the contiguous fragment range
//! `[⌊i·P/N⌋, ⌊(i+1)·P/N⌋)` and holds params, momentum, versions and its
//! checkpoint for that range only, so resident head memory drops to ~1/N.
//!
//! Global steps are interleaved instead of coordinated: shard `i` runs the
//! steps `t ≡ i+1 (mod N)`, i.e. `i+1, i+1+N, ...`, and its k-th round
//! (k = ⌊(t−1)/N⌋) merges fragment `start + k mod len`.  So
//!
//! * every global step has exactly one owner, computable from `t` alone;
//! * a fragment version (the step of its last merge) is unique across the
//!   whole head, and the union of the shards' per-fragment versions is the
//!   head's vector clock — no cross-shard message is needed;
//! * with `N = 1` every formula reduces to the unsharded scheduler
//!   (`t = 1, 2, ...`, fragment `(t−1) mod P`).
//!
//! Shards advance independently, so steps of different shards are not
//! ordered in wall time (step 7 of shard 0 may commit before step 6 of shard
//! 1).  Within one shard commits stay in strict step order.  The run ends when
//! every shard has committed its last step `≤ total_steps`; islands treat the
//! terminal cut as the union of the shards' final manifests.
//!
//! The island side mirrors `owned_range` in `yeto/syncer_shards.py`.

use std::ops::Range;

use anyhow::{bail, Result};

/// This process's slice of the head; `{index: 0, count: 1}` is unsharded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShardPlan {
    pub index: u32,
    pub count: u32,
}

impl Default for ShardPlan {
    fn default() -> Self {
        Self { index: 0, count: 1 }
    }
}

impl ShardPlan {
    pub fn new(index: u32, count: u32) -> Result<Self> {
        if count == 0 {
            bail!("--shard-count must be positive");
        }
        if index >= count {
            bail!("--shard-index {index} must be below --shard-count {count}");
        }
        Ok(Self { index, count })
    }

    pub fn is_sharded(&self) -> bool {
        self.count > 1
    }

    /// Fragments owned by this shard; an error if the layout is too small
    /// for every shard to own at least one fragment.
    pub fn owned_range(&self, num_fragments: usize) -> Result<Range<usize>> {
        let count = self.count as usize;
        if num_fragments < count {
            bail!("{count} syncer shards need at least {count} fragments, layout has {num_fragments}");
        }
        let index = self.index as usize;
        Ok(index * num_fragments / count..(index + 1) * num_fragments / count)
    }

    fn stride(&self) -> u64 {
        u64::from(self.count)
    }

    /// Whether this shard runs global step `t` (t ≥ 1).
    pub fn owns_step(&self, t: u64) -> bool {
        t >= 1 && (t - 1) % self.stride() == u64::from(self.index)
    }

    /// The first step this shard runs after `step` (its last committed step,
    /// or 0 before the first).
    pub fn step_after(&self, step: u64) -> u64 {
        let first = u64::from(self.index) + 1;
        if step < first {
            return first;
        }
        let k = (step - first) / self.stride() + 1;
        first + k * self.stride()
    }

    /// The last step this shard runs within `total_steps` (0 if none).
    pub fn last_step(&self, total_steps: u64) -> u64 {
        let first = u64::from(self.index) + 1;
        if total_steps < first {
            return 0;
        }
        first + (total_steps - first) / self.stride() * self.stride()
    }

    /// The 1-based round count of this shard at its step `t`; equals `t`
    /// unsharded.  Periodic intervals (checkpoint_every) count these.
    pub fn round_number(&self, t: u64) -> u64 {
        (t - 1) / self.stride() + 1
    }

    /// The fragment merged at this shard's step `t`.
    pub fn fragment_for_step(&self, t: u64, owned: &Range<usize>) -> usize {
        let k = (t - 1) / self.stride();
        owned.start + (k % owned.len() as u64) as usize
    }

    /// The version (step of last merge) of owned fragment `p` in a
    /// contiguous cut whose last committed step is `global_step`; 0 if `p`
    /// has not been merged yet.  Unsharded this is the strict round-robin
    /// version `p+1 + ⌊(g−p−1)/P⌋·P`.
    pub fn expected_version(&self, p: usize, global_step: u64, owned: &Range<usize>) -> u64 {
        let first = u64::from(self.index) + 1;
        if global_step < first {
            return 0;
        }
        let rounds = (global_step - first) / self.stride() + 1;
        let j = (p - owned.start) as u64;
        let len = owned.len() as u64;
        if rounds <= j {
            return 0;
        }
        let k = j + (rounds - 1 - j) / len * len;
        first + k * self.stride()
    }

    /// The step at which the fragment merged at `t` is merged next.
    pub fn next_merge_of_fragment(&self, t: u64, owned: &Range<usize>) -> u64 {
        t + self.stride() * owned.len() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsharded_plan_is_the_classic_schedule() {
        let plan = ShardPlan::default();
        let owned = plan.owned_range(5).unwrap();
        assert_eq!(owned, 0..5);
        for t in 1..=20 {
            assert!(plan.owns_step(t));
            assert_eq!(plan.step_after(t - 1), t);
            assert_eq!(plan.round_number(t), t);
            assert_eq!(plan.fragment_for_step(t, &owned), ((t - 1) % 5) as usize);
            assert_eq!(plan.next_merge_of_fragment(t, &owned), t + 5);
        }
        assert_eq!(plan.last_step(17), 17);
        for p in 0..5 {
            for g in 0..30u64 {
                let first = p as u64 + 1;
                let classic = if g < first { 0 } else { first + (g - first) / 5 * 5 };
                assert_eq!(plan.expected_version(p, g, &owned), classic);
            }
        }
        assert_eq!(plan.last_step(0), 0);
    }

    #[test]
    fn shards_partition_steps_and_fragments() {
        for count in 1..=4u32 {
            for fragments in count as usize..=9 {
                let plans: Vec<_> = (0..count).map(|i| ShardPlan::new(i, count).unwrap()).collect();
                let ranges: Vec<_> = plans.iter().map(|p| p.owned_range(fragments).unwrap()).collect();
                // Disjoint, contiguous, covering, non-empty.
                assert_eq!(ranges[0].start, 0);
                assert_eq!(ranges.last().unwrap().end, fragments);
                for w in ranges.windows(2) {
                    assert_eq!(w[0].end, w[1].start);
                }
                assert!(ranges.iter().all(|r| !r.is_empty()));
                let total = 37u64;
                let mut merges = vec![0u64; fragments];
                for t in 1..=total {
                    let owners: Vec<_> = plans.iter().filter(|p| p.owns_step(t)).collect();
                    assert_eq!(owners.len(), 1, "step {t}");
                    let plan = owners[0];
                    let owned = &ranges[plan.index as usize];
                    let p = plan.fragment_for_step(t, owned);
                    assert!(owned.contains(&p));
                    merges[p] += 1;
                    assert!(t <= plan.last_step(total));
                    let next = plan.step_after(t);
                    assert!(plan.owns_step(next) && next == t + u64::from(count));
                    let again = plan.next_merge_of_fragment(t, owned);
                    assert_eq!(plan.fragment_for_step(again, owned), p);
                    assert_eq!(plan.expected_version(p, t, owned), t);
                    for q in owned.clone() {
                        let v = plan.expected_version(q, t, owned);
                        assert!(v == 0 || (plan.owns_step(v) && plan.fragment_for_step(v, owned) == q));
                    }
                }
                for plan in &plans {
                    let last = plan.last_step(total);
                    assert!(last == 0 || (plan.owns_step(last) && last + u64::from(count) > total));
                }
                // Each shard sweeps its own range round-robin.
                assert!(merges.iter().all(|&m| m > 0));
            }
        }
    }

    #[test]
    fn invalid_plans_are_refused() {
        assert!(ShardPlan::new(0, 0).is_err());
        assert!(ShardPlan::new(2, 2).is_err());
        assert!(ShardPlan::new(1, 3).unwrap().owned_range(2).is_err());
        assert_eq!(ShardPlan::new(2, 3).unwrap().owned_range(7).unwrap(), 4..7);
    }
}
