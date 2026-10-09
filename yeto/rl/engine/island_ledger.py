"""Cross-island policy-version ledger (rl-inter-island-scheduling, design Q2/Q3/Q4).

Pure Python reference implementation of the outer-layer bookkeeping the
coordinator (the syncer, per design Q2) is meant to own:

* ``SampleGroup`` -- one GRPO prompt group produced by one island under one
  policy version, judged as a whole: ACCEPT / ACCEPT_IS / REJECT.
* ``DeltaEntry`` -- one island's outer delta for a round, weighted like the
  syncer (``w = c_tokens**2 / c_steps``, ``syncer/src/merge.rs:26-31``);
  a member that joined (catch-up) in the current round contributes weight 0.
* Membership with ``membership_epoch``; a member that leaves (explicitly or by
  lease expiry) has its uncommitted delta dropped and recorded.
* ``try_advance`` -- P4 (user ruling 2026-10-07): step when the arrived
  capacity sum reaches ``theta * total capacity`` (and >= ``quorum_min``
  members), or on the soft deadline ``T_soft``; a late delta (computed on an
  older base, lag <= ``max_carry_lag``) is NOT dropped but carried over into
  the next round with a staleness discount ``gamma**lag``.

The CPU fake-island harness (``fake_islands.py``) drives this module as its
coordinator. Island-internal communication/merge optimisations (quantisation,
DyLU, sharded streaming, LoRA SVD) are out of scope; ``wire_dtype``/``shard``/
``extra`` are kept on entries as their interface points only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from yeto.rl.algos.mismatch_correction import CORRECTION_MECHANISMS

class IslandSchedulingMode(str, Enum):
    """Explicit choice between the old syncer behaviour and inter-island scheduling.

    LEGACY (default): fixed members, a round steps only when every member
    arrived (syncer strict: quorum == learners, grace 0), a quorum timeout is
    a failure, no discount, no cross-island samples, no pool_* journal writes.
    ELASTIC: the semantics of this change (capacity-weighted stepping, late
    deltas carried over, join/leave, cross-island samples). Must be opted in.
    The mode value goes into the session contract hash; peers in different
    modes are refused (:func:`check_same_mode`).
    """

    LEGACY = "legacy"
    ELASTIC = "elastic"


DEFAULT_MODE = IslandSchedulingMode.LEGACY


def parse_mode(value: "str | IslandSchedulingMode | None") -> IslandSchedulingMode:
    if value is None:
        return DEFAULT_MODE
    try:
        return IslandSchedulingMode(value)
    except ValueError as exc:
        raise ValueError(f"island scheduling mode must be legacy|elastic, got {value!r}") from exc


def contract_fields(mode: "str | IslandSchedulingMode | None", *, theta: float | None = None,
                    gamma: float | None = None, soft_deadline_s: float | None = None
                    ) -> dict[str, Any]:
    """Fields to fold into the session contract hash (syncer HELLO).

    Names agreed with the syncer (Rust) side: ``island_scheduling_mode``
    (``legacy``/``elastic``); elastic only: ``quorum_theta``, ``carry_gamma``,
    ``soft_deadline_s``. Messages carry ``syncer_epoch`` (u64).
    """
    m = parse_mode(mode)
    out: dict[str, Any] = {"island_scheduling_mode": m.value}
    if m is IslandSchedulingMode.ELASTIC:
        out.update(quorum_theta=DEFAULT_THETA if theta is None else theta,
                   carry_gamma=DEFAULT_GAMMA if gamma is None else gamma,
                   soft_deadline_s=soft_deadline_s)
    return out


def check_same_mode(local: "str | IslandSchedulingMode | None",
                    peer: "str | IslandSchedulingMode | None") -> None:
    if parse_mode(local) != parse_mode(peer):
        raise ValueError(f"island scheduling mode mismatch: local {parse_mode(local).value}, "
                         f"peer {parse_mode(peer).value}; mixing modes is refused")


ACCEPT = "ACCEPT"
ACCEPT_IS = "ACCEPT_IS"
REJECT = "REJECT"
VERDICTS = (ACCEPT, ACCEPT_IS, REJECT)

# Corrections that actually reweight off-policy tokens (observe-only excluded).
IS_CORRECTIONS = tuple(m for m in CORRECTION_MECHANISMS if m != "mismatch_observe")


class LedgerError(RuntimeError):
    pass


@dataclass(frozen=True)
class StalenessPolicy:
    # User ruling 2026-10-07: cross-island samples at most 2 outer steps stale;
    # no inner-step cap. The IS default stays tis until the offline M2PO vs
    # TIS/IcePop comparison (tasks 0.10) is done.
    max_outer_lag: int = 2
    max_inner_lag: int | None = None  # None: no inner-step cap
    correction: str = "tis"
    # Algorithm families whose value/critic needs on-policy data (design Q3).
    on_policy_only: bool = False

    def __post_init__(self) -> None:
        if self.correction not in IS_CORRECTIONS:
            raise LedgerError(f"correction {self.correction!r} is not one of {IS_CORRECTIONS}")
        if self.max_outer_lag < 0:
            raise LedgerError("max_outer_lag must be >= 0")


@dataclass(frozen=True)
class SampleGroup:
    island_id: str
    outer_version: int
    inner_step: int
    policy_hash: str
    n: int = 1
    behavior_logprob: tuple[float, ...] | None = None
    uri: str | None = None


@dataclass(frozen=True)
class DeltaEntry:
    island_id: str
    outer_version: int  # base version the delta was computed on
    inner_step: int
    policy_hash: str
    c_tokens: int
    c_steps: int
    wire_dtype: str = "f32"
    shard: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Verdict:
    verdict: str
    reason: str
    outer_lag: int
    inner_lag: int | None
    correction: str | None = None


def merge_weight(c_tokens: int, c_steps: int) -> float:
    """Syncer merge weight (merge.rs:26-31). Isolated so DyLU can change inputs later."""
    if c_steps <= 0 or c_tokens <= 0:
        return 0.0
    return float(c_tokens) ** 2 / float(c_steps)


def staleness(current_outer: int, current_inner: int | None,
              outer_version: int, inner_step: int) -> tuple[int, int | None]:
    outer_lag = current_outer - outer_version
    inner_lag = None
    if current_inner is not None and outer_lag == 0:
        inner_lag = current_inner - inner_step
    return outer_lag, inner_lag


@dataclass
class _Member:
    island_id: str
    joined_at: int  # outer version at join; zero weight while version == joined_at
    catch_up: bool
    last_heartbeat: float
    capacity: float = 1.0  # cap_i (P4); from the syncer step-time EMA in stage 1


# P4 defaults [待真机校准]: recommended, not yet measured on hardware.
DEFAULT_THETA = 0.75
DEFAULT_QUORUM_MIN = 1
DEFAULT_GAMMA = 0.5
DEFAULT_MAX_CARRY_LAG = 2


class CrossIslandLedger:
    def __init__(self, *, policy: StalenessPolicy = StalenessPolicy(),
                 theta: float = DEFAULT_THETA, quorum_min: int = DEFAULT_QUORUM_MIN,
                 gamma: float = DEFAULT_GAMMA, max_carry_lag: int = DEFAULT_MAX_CARRY_LAG,
                 lease_s: float = 30.0, initial_hash: str = "h0",
                 syncer_epoch: int = 0,
                 mode: "str | IslandSchedulingMode | None" = None) -> None:
        self.mode = parse_mode(mode)
        if not 0.0 < theta <= 1.0 or quorum_min < 1 or not 0.0 <= gamma <= 1.0:
            raise LedgerError("theta must be in (0,1], quorum_min >= 1, gamma in [0,1]")
        self.policy = policy
        self.theta = theta
        self.quorum_min = quorum_min
        self.gamma = gamma
        self.max_carry_lag = max_carry_lag
        self.syncer_epoch = syncer_epoch  # P9 fencing token
        self.carried: dict[str, tuple[DeltaEntry, int]] = {}
        self.lease_s = lease_s
        self.outer_version = 0
        self.published: dict[int, str] = {0: initial_hash}
        self.membership_epoch = 0
        self.members: dict[str, _Member] = {}
        self.pending: dict[str, DeltaEntry] = {}
        self.events: list[dict[str, Any]] = []

    # -- tape ------------------------------------------------------------------
    def _emit(self, kind: str, **fields: Any) -> dict[str, Any]:
        ev = {"kind": kind, "outer_version": self.outer_version,
              "membership_epoch": self.membership_epoch, **fields}
        self.events.append(ev)
        return ev

    # -- membership -----------------------------------------------------------
    def check_fence(self, syncer_epoch: int) -> None:
        """P9: refuse messages stamped by an older coordinator incarnation."""
        if syncer_epoch < self.syncer_epoch:
            raise LedgerError(f"fenced: syncer_epoch {syncer_epoch} < {self.syncer_epoch}")

    @property
    def legacy(self) -> bool:
        return self.mode is IslandSchedulingMode.LEGACY

    def join(self, island_id: str, *, now: float, catch_up: bool | None = None,
             capacity: float = 1.0, role: str = "train") -> dict[str, Any]:
        if role != "train":
            # D11.1 (rl-eval-difficulty-buckets 5.1): an eval island only serves
            # inference; it never joins the merge pool nor submits a delta.
            raise LedgerError(f"{island_id}: role {role!r} cannot join the merge pool (train islands only)")
        if island_id in self.members:
            raise LedgerError(f"{island_id} is already a member")
        if self.legacy:
            if self.outer_version > 0:
                raise LedgerError("legacy mode has fixed members: no join after the first step")
            catch_up = False
        # The founding round (version 0, nothing published yet) is not catch-up.
        cu = (self.outer_version > 0) if catch_up is None else catch_up
        self.members[island_id] = _Member(island_id, self.outer_version, cu, now, capacity)
        self.membership_epoch += 1
        return self._emit("pool_join", island_id=island_id, catch_up=cu,
                          base_version=self.outer_version,
                          policy_hash=self.published[self.outer_version])

    def leave(self, island_id: str, *, reason: str) -> dict[str, Any]:
        if island_id not in self.members:
            raise LedgerError(f"{island_id} is not a member")
        if self.legacy:
            raise LedgerError("legacy mode has fixed members: leave is not supported")
        del self.members[island_id]
        dropped = self.pending.pop(island_id, None)
        carried = self.carried.pop(island_id, None)
        if dropped is None and carried is not None:
            dropped = carried[0]
        self.membership_epoch += 1
        return self._emit("pool_leave", island_id=island_id, reason=reason,
                          dropped_uncommitted=None if dropped is None else {
                              "c_tokens": dropped.c_tokens, "c_steps": dropped.c_steps,
                              "outer_version": dropped.outer_version})

    def heartbeat(self, island_id: str, *, now: float) -> None:
        if island_id not in self.members:
            raise LedgerError(f"{island_id} is not a member (rejoin required)")
        self.members[island_id].last_heartbeat = now

    def expire_leases(self, *, now: float) -> list[str]:
        if self.legacy:  # fixed roster: the syncer's own progress lease decides
            return []
        gone = [m.island_id for m in self.members.values()
                if now - m.last_heartbeat > self.lease_s]
        for island_id in gone:
            self.leave(island_id, reason="lease_expired")
        return gone

    # -- samples ----------------------------------------------------------------
    def judge(self, group: SampleGroup, *, consumer_island: str,
              consumer_inner_step: int | None = None) -> Verdict:
        outer_lag, inner_lag = staleness(self.outer_version, consumer_inner_step,
                                         group.outer_version, group.inner_step)

        def out(v: str, reason: str, corr: str | None = None) -> Verdict:
            verdict = Verdict(v, reason, outer_lag, inner_lag, corr)
            self._emit("sample_verdict", island_id=group.island_id,
                       consumer=consumer_island, group_version=group.outer_version,
                       n=group.n, **{k: val for k, val in asdict(verdict).items()})
            return verdict

        if self.legacy and (group.island_id != consumer_island or outer_lag != 0):
            return out(REJECT, "legacy_mode")
        if group.island_id not in self.members:
            return out(REJECT, "producer_not_member")
        expected = self.published.get(group.outer_version)
        if expected is None or outer_lag < 0:
            return out(REJECT, "unknown_version")
        if group.policy_hash != expected:
            return out(REJECT, "policy_hash_mismatch")
        cap = self.policy.max_inner_lag
        if outer_lag == 0 and (inner_lag is None or cap is None or inner_lag <= cap):
            if group.island_id == consumer_island or inner_lag in (None, 0):
                return out(ACCEPT, "same_version")
        cross = group.island_id != consumer_island
        if self.policy.on_policy_only and (cross or outer_lag > 0):
            return out(REJECT, "on_policy_only")
        if outer_lag > self.policy.max_outer_lag:
            return out(REJECT, "outer_lag_exceeded")
        if cap is not None and inner_lag is not None and inner_lag > cap:
            return out(REJECT, "inner_lag_exceeded")
        if group.behavior_logprob is None:
            return out(REJECT, "missing_behavior_logprob")
        return out(ACCEPT_IS, "stale_within_bound", self.policy.correction)

    # -- deltas / quorum ----------------------------------------------------------
    def capacity_arrived(self) -> tuple[float, float]:
        total = sum(m.capacity for m in self.members.values())
        arrived = sum(self.members[i].capacity for i in self.pending)
        return arrived, total

    def submit(self, delta: DeltaEntry) -> dict[str, Any]:
        if delta.island_id not in self.members:
            return self._emit("delta_rejected", island_id=delta.island_id, reason="not_member")
        if self.published.get(delta.outer_version) != delta.policy_hash:
            return self._emit("delta_rejected", island_id=delta.island_id,
                              reason="policy_hash_mismatch")
        lag = self.outer_version - delta.outer_version
        if lag > 0 and self.legacy:
            return self._emit("delta_rejected", island_id=delta.island_id,
                              reason="stale_base", base=delta.outer_version, lag=lag)
        if lag > 0:
            if lag > self.max_carry_lag:
                return self._emit("delta_rejected", island_id=delta.island_id,
                                  reason="carry_lag_exceeded", base=delta.outer_version, lag=lag)
            self.carried[delta.island_id] = (delta, lag)
            return self._emit("delta_carried_over", island_id=delta.island_id,
                              base=delta.outer_version, lag=lag,
                              discount=self.gamma ** lag)
        self.pending[delta.island_id] = delta
        return self._emit("delta_accepted", island_id=delta.island_id,
                          c_tokens=delta.c_tokens, c_steps=delta.c_steps)

    def weight_of(self, delta: DeltaEntry) -> float:
        m = self.members[delta.island_id]
        if m.catch_up and m.joined_at == self.outer_version:
            return 0.0
        return merge_weight(delta.c_tokens, delta.c_steps)

    def try_advance(self, *, timed_out: bool, new_hash: str | None = None) -> dict[str, Any] | None:
        """P4 step; ``timed_out`` means the soft deadline T_soft passed."""
        cap_arrived, cap_total = self.capacity_arrived()
        arrived = len(self.pending)
        if self.legacy:
            # syncer strict: quorum == learners, grace 0; a timeout fails closed.
            if arrived < len(self.members):
                if timed_out:
                    self._emit("legacy_quorum_timeout", arrived=arrived, members=len(self.members))
                    raise LedgerError("legacy strict round timed out before all members arrived")
                return None
            reached = True
        else:
            reached = cap_total > 0 and cap_arrived >= self.theta * cap_total
        if not reached and not timed_out:
            return None
        if arrived < self.quorum_min:
            if timed_out:
                self._emit("round_idle", arrived=arrived, cap_arrived=cap_arrived,
                           cap_total=cap_total)
            return None
        weights = {i: self.weight_of(d) for i, d in sorted(self.pending.items())}
        carried_in = {}
        for i, (d, lag) in sorted(self.carried.items()):
            if i in self.members:
                key = f"{i}@{d.outer_version}"
                weights[key] = merge_weight(d.c_tokens, d.c_steps) * self.gamma ** lag
                carried_in[key] = {"lag": lag, "discount": self.gamma ** lag}
        self.carried.clear()
        total = sum(weights.values())
        normalized = {i: (w / total if total > 0 else 0.0) for i, w in weights.items()}
        absent = sorted(set(self.members) - set(self.pending))
        base_version = self.outer_version
        self.outer_version += 1
        self.published[self.outer_version] = new_hash or f"h{self.outer_version}"
        self.pending.clear()
        return self._emit("outer_step", base_version=base_version, arrived=arrived,
                          cap_arrived=cap_arrived, cap_total=cap_total, timed_out=timed_out,
                          carried_in=carried_in,
                          weights=normalized, raw_weights=weights, absent=absent,
                          policy_hash=self.published[self.outer_version])
