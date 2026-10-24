//! Head sharding: one logical syncer split across N processes (S27 P4b).
//!
//! The syncer already works per fragment: every round merges
//! exactly one fragment, and the only cross-fragment state is the global step
//! counter that orders rounds.  The head can therefore be split by fragment
//! ranges.  Shard `i` of `N` owns the contiguous fragment range
//! `[⌊i·P/N⌋, ⌊(i+1)·P/N⌋)` and holds params, momentum, versions and its
//! checkpoint for that range only, so resident head memory drops to ~1/N.
//!
//! The step schedule is the unsharded one: global step `t` merges fragment
//! `(t−1) mod P`, and shard `i` runs exactly the steps whose fragment it
//! owns.  So
//!
//! * every global step has exactly one owner, computable from `t` alone;
//! * a fragment version (the step of its last merge), every PULL and every
//!   final manifest are the same as with one process: islands do not need
//!   to know N beyond routing, and N = 1 is the unsharded scheduler;
//! * the union of the shards' per-fragment versions is the head's vector
//!   clock — no cross-shard message is needed.
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

    /// This shard's step schedule over a layout of `num_fragments`.
    pub fn schedule(&self, num_fragments: usize) -> Result<ShardSchedule> {
        Ok(ShardSchedule { owned: self.owned_range(num_fragments)?, fragments: num_fragments as u64 })
    }
}

/// The global steps one shard runs: those whose round-robin fragment
/// `(t−1) mod P` lies in its owned range.  Unsharded this is every step.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShardSchedule {
    owned: Range<usize>,
    fragments: u64,
}

impl ShardSchedule {
    /// Every fragment of a `fragments`-fragment layout: the unsharded head.
    pub fn unsharded(fragments: usize) -> Self {
        Self { owned: 0..fragments, fragments: fragments as u64 }
    }

    fn first(&self) -> u64 {
        self.owned.start as u64
    }

    /// One past the last owned fragment.
    fn end(&self) -> u64 {
        self.owned.end as u64
    }

    /// Fragments this shard owns.
    pub fn owned(&self) -> Range<usize> {
        self.owned.clone()
    }

    /// The fragment merged at global step `t` (t ≥ 1), on any shard.
    pub fn fragment_for_step(&self, t: u64) -> usize {
        ((t - 1) % self.fragments) as usize
    }

    /// Whether this shard runs global step `t` (t ≥ 1).
    pub fn owns_step(&self, t: u64) -> bool {
        t >= 1 && self.owned.contains(&self.fragment_for_step(t))
    }

    /// The first step this shard runs after `step` (its last committed step,
    /// or 0 before the first).
    pub fn step_after(&self, step: u64) -> u64 {
        // Step `step + 1` merges fragment `step mod P`.
        let position = step % self.fragments;
        let next = step + 1;
        if position < self.first() {
            next + (self.first() - position)
        } else if position < self.end() {
            next
        } else {
            next + (self.fragments - position) + self.first()
        }
    }

    /// The last step this shard runs within `total_steps` (0 if none).
    pub fn last_step(&self, total_steps: u64) -> u64 {
        if total_steps == 0 {
            return 0;
        }
        let sweep = (total_steps - 1) / self.fragments;
        let position = (total_steps - 1) % self.fragments;
        if position >= self.first() {
            sweep * self.fragments + position.min(self.end() - 1) + 1
        } else if sweep == 0 {
            0
        } else {
            (sweep - 1) * self.fragments + self.end()
        }
    }

    /// The 1-based round count of this shard at its step `t`; equals `t`
    /// unsharded.  Periodic intervals (checkpoint_every) count these.
    pub fn round_number(&self, t: u64) -> u64 {
        let len = self.end() - self.first();
        let sweep = (t - 1) / self.fragments;
        let position = (t - 1) % self.fragments;
        sweep * len + (position - self.first()) + 1
    }

    /// The version (step of last merge) of owned fragment `p` in a
    /// contiguous cut whose last committed step is `global_step`; 0 if `p`
    /// has not been merged yet.  This is the strict round-robin version
    /// `p+1 + ⌊(g−p−1)/P⌋·P` on every shard.
    pub fn expected_version(&self, p: usize, global_step: u64) -> u64 {
        let first = p as u64 + 1;
        if global_step < first {
            return 0;
        }
        first + (global_step - first) / self.fragments * self.fragments
    }

    /// The step at which the fragment merged at `t` is merged next.
    pub fn next_merge_of_fragment(&self, t: u64) -> u64 {
        t + self.fragments
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsharded_plan_is_the_classic_schedule() {
        let schedule = ShardPlan::default().schedule(5).unwrap();
        assert_eq!(schedule.owned(), 0..5);
        for t in 1..=20 {
            assert!(schedule.owns_step(t));
            assert_eq!(schedule.step_after(t - 1), t);
            assert_eq!(schedule.round_number(t), t);
            assert_eq!(schedule.fragment_for_step(t), ((t - 1) % 5) as usize);
            assert_eq!(schedule.next_merge_of_fragment(t), t + 5);
        }
        assert_eq!(schedule.last_step(17), 17);
        for p in 0..5 {
            for g in 0..30u64 {
                let first = p as u64 + 1;
                let classic = if g < first { 0 } else { first + (g - first) / 5 * 5 };
                assert_eq!(schedule.expected_version(p, g), classic);
            }
        }
        assert_eq!(schedule.last_step(0), 0);
    }

    #[test]
    fn shards_partition_the_unsharded_steps() {
        for count in 1..=4u32 {
            for fragments in count as usize..=9 {
                let schedules: Vec<_> = (0..count)
                    .map(|i| ShardPlan::new(i, count).unwrap().schedule(fragments).unwrap())
                    .collect();
                let ranges: Vec<_> = schedules.iter().map(ShardSchedule::owned).collect();
                // Disjoint, contiguous, covering, non-empty.
                assert_eq!(ranges[0].start, 0);
                assert_eq!(ranges.last().unwrap().end, fragments);
                for w in ranges.windows(2) {
                    assert_eq!(w[0].end, w[1].start);
                }
                assert!(ranges.iter().all(|r| !r.is_empty()));
                for total in 0..=(3 * fragments as u64 + 2) {
                    let mut cursors = vec![0u64; count as usize];
                    let mut rounds = vec![0u64; count as usize];
                    for t in 1..=total {
                        let owners: Vec<_> =
                            (0..count as usize).filter(|&i| schedules[i].owns_step(t)).collect();
                        assert_eq!(owners.len(), 1, "step {t}");
                        let i = owners[0];
                        let schedule = &schedules[i];
                        let p = schedule.fragment_for_step(t);
                        assert_eq!(p, ((t - 1) % fragments as u64) as usize);
                        assert!(ranges[i].contains(&p));
                        // Each shard walks its own steps in order, with no gaps.
                        assert_eq!(schedule.step_after(cursors[i]), t);
                        cursors[i] = t;
                        rounds[i] += 1;
                        assert_eq!(schedule.round_number(t), rounds[i]);
                        assert_eq!(schedule.expected_version(p, t), t);
                        assert_eq!(schedule.fragment_for_step(schedule.next_merge_of_fragment(t)), p);
                    }
                    for (i, schedule) in schedules.iter().enumerate() {
                        assert_eq!(schedule.last_step(total), cursors[i], "shard {i} total {total}");
                        assert!(schedule.step_after(cursors[i]) > total);
                    }
                }
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
