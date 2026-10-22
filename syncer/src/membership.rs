//! Dynamic island membership for the sharded syncer (S27 E-1/E-2).
//!
//! This ports the membership half of `elastic::ElasticCoordinator` (leases,
//! heartbeats, re-JOIN with catch-up, fencing, capacity quorum θ, soft
//! deadline, carry discount) into the fragment-round server in `server.rs`,
//! so one syncer has both the fragment-round data plane (fragments, K +
//! adaptive grace, tokens²/steps weights, RDA) and dynamic membership.
//!
//! The module is a pure state machine: it never touches sockets or tensors,
//! and every clock reading is passed in. The server keeps one [`Membership`]
//! inside its connection registry, so HELLO admission and fencing happen
//! under the same lock that registers a connection.
//!
//! Every option defaults to the existing behavior:
//! - quorum is the learner count K (`QuorumRule::Count`);
//! - no lease (a member leaves only when its connection closes);
//! - no soft deadline (a round below quorum restarts at the quorum timeout);
//! - no carry discount (γ = 1).
//!
//! Concepts:
//! - A *member* is one logical island (`learner_id`) bound to one connection
//!   `generation`. A reconnect is a new generation of the same island.
//! - A *retired* generation can never be admitted again (fencing of stale
//!   connection ids, e.g. a delayed HELLO of a superseded process).
//! - A lease *expiry* removes the island from the pool. Its next HELLO is a
//!   re-JOIN: the island gets the current parameters (catch-up), and a push
//!   for fragment p whose base version predates that catch-up is fenced, per
//!   fragment, because it was computed against parameters from before the
//!   island was declared gone.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

use anyhow::{ensure, Result};

/// One connection generation of one logical island.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Member {
    pub learner_id: u32,
    pub generation: u64,
}

/// Retired generations remembered per island. Generations are random 64-bit
/// values, so this only has to cover realistic reconnect storms.
const RETIRED_PER_ISLAND: usize = 1024;

/// How a fragment round decides that enough islands have answered.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum QuorumRule {
    /// Count quorum: at least K answering islands (`--quorum`).
    #[default]
    Count,
    /// Elastic capacity quorum: the answering islands hold at least θ of the
    /// capacity of the islands the round expects, and at least `q_min`
    /// islands answered.
    Capacity { theta: f64, q_min: u32 },
}

#[derive(Clone, Debug, PartialEq)]
pub struct MembershipConfig {
    pub quorum_rule: QuorumRule,
    /// Remove an island that sent nothing (HEARTBEAT, PUSH, INIT) for this
    /// long. None: an island leaves only when its connection closes.
    pub lease: Option<Duration>,
    /// Upper bound on one round's wait: once it passes, the round merges
    /// whatever arrived (at least `soft_min_arrivals()`), even below quorum.
    /// The adaptive grace window is clamped to it. None: off.
    pub soft_deadline: Option<Duration>,
    /// Weight multiplier γ^lag for a push whose base version lags the
    /// round's fragment version by `lag`. 1 = no discount.
    pub carry_gamma: f64,
    /// Relative capacity per island id for `QuorumRule::Capacity`; islands
    /// not listed have capacity 1.
    pub capacities: BTreeMap<u32, f64>,
    /// Admit island ids in `0..max_learners` (scale-up past the initial
    /// roster). None: only `0..learners`.
    pub max_learners: Option<u32>,
}

impl Default for MembershipConfig {
    fn default() -> Self {
        Self {
            quorum_rule: QuorumRule::Count,
            lease: None,
            soft_deadline: None,
            carry_gamma: 1.0,
            capacities: BTreeMap::new(),
            max_learners: None,
        }
    }
}

impl MembershipConfig {
    /// True when every option keeps the pre-membership server behavior.
    pub fn is_static(&self) -> bool {
        *self == Self::default()
    }

    /// Exclusive upper bound on admitted island ids.
    pub fn id_limit(&self, learners: u32) -> u32 {
        self.max_learners.unwrap_or(learners)
    }

    /// Fewest arrivals a soft-deadline merge accepts.
    pub fn soft_min_arrivals(&self) -> usize {
        match self.quorum_rule {
            QuorumRule::Count => 1,
            QuorumRule::Capacity { q_min, .. } => q_min.max(1) as usize,
        }
    }

    /// Fewest connected islands needed to (re)launch a round.
    pub fn min_launch_members(&self, quorum: u32) -> usize {
        match self.quorum_rule {
            QuorumRule::Count => quorum as usize,
            QuorumRule::Capacity { q_min, .. } => q_min.max(1) as usize,
        }
    }

    pub fn capacity(&self, learner_id: u32) -> f64 {
        self.capacities.get(&learner_id).copied().unwrap_or(1.0)
    }

    /// Merge-weight multiplier for a push `lag` fragment versions behind.
    /// Exactly 1.0 when γ = 1, so the default merge is bit-identical.
    pub fn carry_discount(&self, lag: u64) -> f64 {
        if self.carry_gamma == 1.0 || lag == 0 {
            return 1.0;
        }
        self.carry_gamma.powi(lag.min(i32::MAX as u64) as i32)
    }

    pub fn validate(&self, learners: u32, quorum: u32) -> Result<()> {
        if let QuorumRule::Capacity { theta, q_min } = self.quorum_rule {
            ensure!(theta > 0.0 && theta <= 1.0, "--quorum-theta must be in (0, 1], got {theta}");
            ensure!(q_min >= 1, "--q-min must be >= 1");
            ensure!(quorum == 1, "--quorum-theta replaces --quorum; leave --quorum at 1");
        }
        if let Some(lease) = self.lease {
            ensure!(!lease.is_zero(), "--island-lease-s must be positive");
        }
        if let Some(soft) = self.soft_deadline {
            ensure!(!soft.is_zero(), "--soft-deadline-s must be positive");
        }
        ensure!(
            (0.0..=1.0).contains(&self.carry_gamma) && self.carry_gamma > 0.0,
            "--carry-gamma must be in (0, 1], got {}",
            self.carry_gamma
        );
        for (&id, &cap) in &self.capacities {
            ensure!(cap.is_finite() && cap > 0.0, "island {id} capacity must be > 0, got {cap}");
            ensure!(id < self.id_limit(learners), "island capacity id {id} is outside 0..{}", self.id_limit(learners));
        }
        if let Some(max) = self.max_learners {
            ensure!(max >= learners, "--max-learners must be >= --learners");
        }
        Ok(())
    }

    /// Snapshot of the quorum rule for one round; None keeps the K count.
    pub fn round_quorum(&self) -> Option<CapacityQuorum> {
        match self.quorum_rule {
            QuorumRule::Count => None,
            QuorumRule::Capacity { theta, q_min } => Some(CapacityQuorum {
                theta,
                q_min: q_min.max(1) as usize,
                capacities: self.capacities.clone(),
            }),
        }
    }
}

/// Capacity quorum frozen into one round at launch. The expected set may
/// still shrink (lease expiry), so θ applies to the islands still expected.
#[derive(Clone, Debug, PartialEq)]
pub struct CapacityQuorum {
    pub theta: f64,
    pub q_min: usize,
    capacities: BTreeMap<u32, f64>,
}

impl CapacityQuorum {
    fn capacity(&self, learner_id: u32) -> f64 {
        self.capacities.get(&learner_id).copied().unwrap_or(1.0)
    }

    /// (arrived capacity, expected capacity).
    pub fn capacity_arrived<'a>(
        &self,
        expected: &[Member],
        arrived: impl IntoIterator<Item = &'a Member>,
    ) -> (f64, f64) {
        let total = expected.iter().map(|m| self.capacity(m.learner_id)).sum();
        let arrived = arrived.into_iter().map(|m| self.capacity(m.learner_id)).sum();
        (arrived, total)
    }

    pub fn reached<'a>(
        &self,
        expected: &[Member],
        arrived: impl IntoIterator<Item = &'a Member>,
    ) -> bool {
        let arrived: Vec<&Member> = arrived.into_iter().collect();
        if arrived.len() < self.q_min {
            return false;
        }
        let (cap_arrived, cap_total) = self.capacity_arrived(expected, arrived);
        cap_total > 0.0 && cap_arrived >= self.theta * cap_total
    }
}

/// Why a HELLO was refused before registration.
#[derive(Debug, Eq, PartialEq)]
pub enum AdmitError {
    /// This generation was superseded, disconnected, or expired before.
    RetiredGeneration,
    OutOfRange { limit: u32 },
}

impl std::fmt::Display for AdmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AdmitError::RetiredGeneration => write!(f, "connection generation is retired (fenced)"),
            AdmitError::OutOfRange { limit } => write!(f, "learner id is outside 0..{limit}"),
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum Admission {
    /// First generation of this island, or a reconnect while it was a member.
    Join { replaced: Option<u64> },
    /// The island's previous generation expired: a re-JOIN with catch-up.
    Rejoin,
}

#[derive(Debug)]
struct LiveMember {
    generation: u64,
    last_seen: Instant,
    /// Re-JOIN after expiry and catch-up not yet recorded.
    awaiting_catch_up: bool,
    /// Per-fragment versions the island was caught up to on re-JOIN; a push
    /// whose base is older is fenced.
    catch_up_floor: Option<Vec<u64>>,
}

#[derive(Debug)]
pub struct Membership {
    cfg: MembershipConfig,
    id_limit: u32,
    live: BTreeMap<u32, LiveMember>,
    retired: HashMap<u32, (VecDeque<u64>, HashSet<u64>)>,
    /// Islands whose last generation expired (their next HELLO is a re-JOIN).
    expired: HashSet<u32>,
    /// Bumped on every join, leave, and expiry.
    pub epoch: u64,
}

impl Default for Membership {
    fn default() -> Self {
        Self::new(MembershipConfig::default(), u32::MAX)
    }
}

impl Membership {
    pub fn new(cfg: MembershipConfig, id_limit: u32) -> Self {
        Self {
            cfg,
            id_limit,
            live: BTreeMap::new(),
            retired: HashMap::new(),
            expired: HashSet::new(),
            epoch: 0,
        }
    }

    pub fn config(&self) -> &MembershipConfig {
        &self.cfg
    }

    pub fn is_live(&self, member: Member) -> bool {
        self.live.get(&member.learner_id).is_some_and(|m| m.generation == member.generation)
    }

    pub fn is_retired(&self, member: Member) -> bool {
        self.retired.get(&member.learner_id).is_some_and(|(_, set)| set.contains(&member.generation))
    }

    fn retire(&mut self, member: Member) {
        let (order, set) = self.retired.entry(member.learner_id).or_default();
        if set.insert(member.generation) {
            order.push_back(member.generation);
            while order.len() > RETIRED_PER_ISLAND {
                if let Some(old) = order.pop_front() {
                    set.remove(&old);
                }
            }
        }
    }

    /// Admit a HELLO. Fails without changing state when the id is out of
    /// range or the generation is retired. A live older generation of the
    /// same island is superseded (and retired).
    pub fn admit(&mut self, member: Member, now: Instant) -> std::result::Result<Admission, AdmitError> {
        if member.learner_id >= self.id_limit {
            return Err(AdmitError::OutOfRange { limit: self.id_limit });
        }
        if self.is_retired(member) || self.is_live(member) {
            return Err(AdmitError::RetiredGeneration);
        }
        let rejoin = self.expired.remove(&member.learner_id);
        let previous = self.live.insert(
            member.learner_id,
            LiveMember { generation: member.generation, last_seen: now, awaiting_catch_up: rejoin, catch_up_floor: None },
        );
        let replaced = previous.map(|p| p.generation);
        if let Some(generation) = replaced {
            self.retire(Member { learner_id: member.learner_id, generation });
        }
        self.epoch += 1;
        Ok(if rejoin { Admission::Rejoin } else { Admission::Join { replaced } })
    }

    /// Record the per-fragment versions a re-joined island was caught up to.
    /// No-op for ordinary joins and reconnects.
    pub fn record_catch_up(&mut self, member: Member, versions: &[u64]) {
        if let Some(m) = self.live.get_mut(&member.learner_id) {
            if m.generation == member.generation && m.awaiting_catch_up {
                m.awaiting_catch_up = false;
                m.catch_up_floor = Some(versions.to_vec());
            }
        }
    }

    /// Lease renewal on any traffic from the live generation.
    pub fn renew(&mut self, member: Member, now: Instant) -> bool {
        match self.live.get_mut(&member.learner_id) {
            Some(m) if m.generation == member.generation => {
                m.last_seen = now;
                true
            }
            _ => false,
        }
    }

    /// The connection of `member` closed. Its generation is retired; the
    /// island stays eligible for a plain reconnect (not a re-JOIN).
    pub fn depart(&mut self, member: Member) {
        if self.is_live(member) {
            self.live.remove(&member.learner_id);
            self.epoch += 1;
        }
        self.retire(member);
    }

    /// Remove every member whose lease ran out (at `next_expiry` or later,
    /// so a scheduler woken at that instant makes progress); returns them
    /// in id order.
    pub fn expire(&mut self, now: Instant) -> Vec<Member> {
        let Some(lease) = self.cfg.lease else { return Vec::new() };
        let gone: Vec<Member> = self
            .live
            .iter()
            .filter(|(_, m)| now.saturating_duration_since(m.last_seen) >= lease)
            .map(|(&learner_id, m)| Member { learner_id, generation: m.generation })
            .collect();
        for member in &gone {
            self.live.remove(&member.learner_id);
            self.expired.insert(member.learner_id);
            self.retire(*member);
            self.epoch += 1;
        }
        gone
    }

    /// Earliest instant at which some lease can expire.
    pub fn next_expiry(&self) -> Option<Instant> {
        let lease = self.cfg.lease?;
        self.live.values().map(|m| m.last_seen + lease).min()
    }

    /// Per-fragment fence for a push from a re-joined island: its base must
    /// not predate the versions it was caught up to. Pushes from any other
    /// generation pass here; rounds check who they expect.
    pub fn check_push_base(&self, member: Member, fragment: usize, base_version: u64) -> Result<()> {
        let Some(m) = self.live.get(&member.learner_id).filter(|m| m.generation == member.generation) else {
            return Ok(());
        };
        if let Some(floor) = m.catch_up_floor.as_ref().and_then(|f| f.get(fragment)) {
            ensure!(
                base_version >= *floor,
                "fragment {fragment} base version {base_version} predates the re-JOIN catch-up version {floor}"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(learner_id: u32, generation: u64) -> Member {
        Member { learner_id, generation }
    }

    fn leased(secs: u64) -> Membership {
        Membership::new(MembershipConfig { lease: Some(Duration::from_secs(secs)), ..Default::default() }, 4)
    }

    #[test]
    fn default_config_is_static_and_neutral() {
        let cfg = MembershipConfig::default();
        assert!(cfg.is_static());
        assert_eq!(cfg.carry_discount(5), 1.0);
        assert_eq!(cfg.min_launch_members(3), 3);
        assert_eq!(cfg.soft_min_arrivals(), 1);
        assert!(cfg.round_quorum().is_none());
        assert!(Membership::default().expire(Instant::now() + Duration::from_secs(1 << 30)).is_empty());
    }

    #[test]
    fn drop_by_lease_expiry_removes_only_the_silent_member() {
        let t0 = Instant::now();
        let mut ms = leased(10);
        ms.admit(m(0, 11), t0).unwrap();
        ms.admit(m(1, 21), t0).unwrap();
        assert!(ms.renew(m(1, 21), t0 + Duration::from_secs(8)));
        assert_eq!(ms.next_expiry(), Some(t0 + Duration::from_secs(10)));
        assert!(ms.expire(t0 + Duration::from_secs(9)).is_empty());
        assert_eq!(ms.expire(t0 + Duration::from_secs(10)), vec![m(0, 11)]);
        assert!(!ms.is_live(m(0, 11)) && ms.is_live(m(1, 21)));
        assert!(!ms.renew(m(0, 11), t0 + Duration::from_secs(12)), "expired generation cannot renew");
    }

    #[test]
    fn reconnect_after_expiry_is_a_rejoin_with_per_fragment_fence() {
        let t0 = Instant::now();
        let mut ms = leased(5);
        assert_eq!(ms.admit(m(0, 1), t0), Ok(Admission::Join { replaced: None }));
        ms.check_push_base(m(0, 1), 0, 0).unwrap();
        ms.expire(t0 + Duration::from_secs(6));
        assert_eq!(ms.admit(m(0, 2), t0 + Duration::from_secs(7)), Ok(Admission::Rejoin));
        // Catch-up to fragment versions [4, 7]: older bases are fenced per fragment.
        ms.record_catch_up(m(0, 2), &[4, 7]);
        assert!(ms.check_push_base(m(0, 2), 0, 3).is_err());
        ms.check_push_base(m(0, 2), 0, 4).unwrap();
        assert!(ms.check_push_base(m(0, 2), 1, 6).is_err());
        ms.check_push_base(m(0, 2), 1, 7).unwrap();
        // The catch-up floor is recorded once, for the re-JOIN only.
        ms.record_catch_up(m(0, 2), &[100, 100]);
        ms.check_push_base(m(0, 2), 0, 4).unwrap();
    }

    #[test]
    fn plain_reconnect_supersedes_without_fence() {
        let t0 = Instant::now();
        let mut ms = leased(5);
        ms.admit(m(0, 1), t0).unwrap();
        assert_eq!(ms.admit(m(0, 2), t0), Ok(Admission::Join { replaced: Some(1) }));
        ms.record_catch_up(m(0, 2), &[9]);
        ms.check_push_base(m(0, 2), 0, 0).unwrap();
        // A superseded generation may still answer a round that captured it.
        ms.check_push_base(m(0, 1), 0, 0).unwrap();
    }

    #[test]
    fn stale_generation_ids_are_rejected() {
        let t0 = Instant::now();
        let mut ms = leased(5);
        ms.admit(m(0, 1), t0).unwrap();
        assert_eq!(ms.admit(m(0, 1), t0), Err(AdmitError::RetiredGeneration), "live duplicate");
        ms.admit(m(0, 2), t0).unwrap();
        let epoch = ms.epoch;
        assert_eq!(ms.admit(m(0, 1), t0), Err(AdmitError::RetiredGeneration), "superseded");
        assert_eq!(ms.epoch, epoch, "a refused HELLO changes nothing");
        assert!(ms.is_live(m(0, 2)));
        ms.depart(m(0, 2));
        assert_eq!(ms.admit(m(0, 2), t0), Err(AdmitError::RetiredGeneration), "disconnected");
        ms.admit(m(0, 3), t0).unwrap();
        ms.expire(t0 + Duration::from_secs(6));
        assert_eq!(ms.admit(m(0, 3), t0), Err(AdmitError::RetiredGeneration), "expired");
        assert_eq!(ms.admit(m(4, 9), t0), Err(AdmitError::OutOfRange { limit: 4 }));
    }

    #[test]
    fn retired_set_is_bounded_per_island() {
        let mut ms = Membership::default();
        for g in 1..=(RETIRED_PER_ISLAND as u64 + 10) {
            ms.depart(m(0, g));
        }
        let (order, set) = &ms.retired[&0];
        assert_eq!(order.len(), RETIRED_PER_ISLAND);
        assert_eq!(set.len(), RETIRED_PER_ISLAND);
        assert!(!ms.is_retired(m(0, 1)) && ms.is_retired(m(0, RETIRED_PER_ISLAND as u64 + 10)));
    }

    #[test]
    fn capacity_quorum_theta_and_q_min() {
        let cfg = MembershipConfig {
            quorum_rule: QuorumRule::Capacity { theta: 0.75, q_min: 1 },
            capacities: [(0, 8.0), (1, 4.0), (2, 4.0)].into_iter().collect(),
            ..Default::default()
        };
        cfg.validate(3, 1).unwrap();
        let q = cfg.round_quorum().unwrap();
        let expected = [m(0, 1), m(1, 1), m(2, 1)];
        assert!(!q.reached(&expected, &[m(0, 1)]), "8/16 < 0.75");
        assert!(q.reached(&expected, &[m(0, 1), m(1, 1)]), "12/16 = 0.75");
        assert!(!q.reached(&expected, &[m(1, 1), m(2, 1)]), "8/16");
        // The round's expected set shrinks after an expiry: θ of what is left.
        assert!(q.reached(&expected[1..], &[m(1, 1), m(2, 1)]));
        let strict = CapacityQuorum { q_min: 3, ..q.clone() };
        assert!(!strict.reached(&expected, &[m(0, 1), m(1, 1)]), "q_min binds");
        assert!(!q.reached(&[], std::iter::empty()), "no capacity, no quorum");
        assert_eq!(cfg.min_launch_members(1), 1);
    }

    #[test]
    fn carry_discount_is_opt_in() {
        let cfg = MembershipConfig { carry_gamma: 0.5, ..Default::default() };
        assert_eq!(cfg.carry_discount(0), 1.0);
        assert_eq!(cfg.carry_discount(2), 0.25);
    }

    #[test]
    fn validate_rejects_bad_options() {
        let theta = |theta| MembershipConfig { quorum_rule: QuorumRule::Capacity { theta, q_min: 1 }, ..Default::default() };
        assert!(theta(0.0).validate(2, 1).is_err());
        assert!(theta(0.5).validate(2, 2).is_err(), "θ and K together");
        assert!(MembershipConfig { carry_gamma: 0.0, ..Default::default() }.validate(2, 1).is_err());
        assert!(MembershipConfig { lease: Some(Duration::ZERO), ..Default::default() }.validate(2, 1).is_err());
        assert!(MembershipConfig { max_learners: Some(1), ..Default::default() }.validate(2, 1).is_err());
        let caps = MembershipConfig { capacities: [(5, 1.0)].into_iter().collect(), ..Default::default() };
        assert!(caps.validate(2, 1).is_err());
        assert!(MembershipConfig { max_learners: Some(8), ..caps }.validate(2, 1).is_ok());
    }
}
