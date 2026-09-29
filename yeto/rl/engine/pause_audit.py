"""Per-profile pause audit for island reconfiguration (rl-infra-spec task 1.5, design D10).

Pure. It answers: at this moment, may the island pause its local loop for up to
``T`` seconds while its outer-sync connection stays alive, and what is the
upper bound? Every number here is sourced from the code audit recorded in
``openspec/changes/rl-infra-spec/pause-audit.md`` (file:line citations there).

Rules:

* only ``(execution_mode, outer_protocol)`` pairs that were audited get a
  budget; any other profile (including ``partitioned-overlap`` and any
  decoupled profile before X6-decoupled) has reconfiguration disabled;
* the only pausable phase is the round boundary AFTER ``boundary()`` returned
  a non-stop result and the policy was published (strict then already holds
  the next PULL permit);
* finalization, budget consolidation, a stop round, anything inside
  ``boundary()`` or a partial-rank collective veto the pause;
* the strict budget is the syncer quorum timeout minus a margin until the
  two-island X6 experiment certifies pauses across a quorum re-send;
  an idle-flow timeout on the network (no TCP keepalive) caps it further.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .execution_profile import ExecutionProfile

# Defaults from the audit (seconds).
SYNCER_QUORUM_TIMEOUT_S = 900.0  # syncer --quorum-timeout-s default
FINALIZATION_TIMEOUT_S = 900.0  # yeto/protocol.py FINALIZATION_TIMEOUT
SYNCER_WRITE_TIMEOUT_S = 180.0  # only if the learner stops reading its socket
MILES_DIST_TIMEOUT_S = 600.0  # --rl-distributed-timeout-minutes default 10
FLEET_RECOVER_TIMEOUT_S = 1200.0  # FleetController never checks progress
DEFAULT_MARGIN = 0.5

PAUSABLE_PHASE = "round-boundary-published"
OUTER_PHASES = (
    PAUSABLE_PHASE,
    "in-boundary",  # push / wait_global / drain / consolidate / _finish
    "stop-round",  # boundary returned stop=True or finish() pending
    "finalizing",
    "budget-consolidation",
    "mid-collective",
)


@dataclass(frozen=True)
class PauseAudit:
    """Audited facts for one (execution_mode, outer_protocol) pair."""

    outer_protocol: str
    certified: bool
    holds_permit_at_pause: bool
    stalls_peers: bool
    hard_limit_s: float | None  # a timeout that fails the run, if any
    budget_s: float | None  # certified pause upper bound (None = no outer limit)
    note: str


_OUTER_AUDIT: Mapping[str, PauseAudit] = {
    "none": PauseAudit(
        "none", True, False, False, None, None,
        "LocalOnlySync: no syncer connection; only in-island collectives must be quiescent",
    ),
    "strict-avg": PauseAudit(
        "strict-avg", True, True, True, None, SYNCER_QUORUM_TIMEOUT_S * DEFAULT_MARGIN,
        "fixed roster: the syncer re-sends the same PULL every quorum timeout and never "
        "drops the learner; peers stall for the pause. Budget stays under one quorum "
        "timeout until X6 certifies pauses across a re-send",
    ),
    "decoupled": PauseAudit(
        "decoupled", False, False, False, FINALIZATION_TIMEOUT_S, None,
        "learner-budget mode fails the run after a 900 s BUDGET_DONE lease; broadcast "
        "queues are unbounded. Disabled until X6-decoupled is run",
    ),
}
_AUDITED_MODES = frozenset({"colocated-serial", "partitioned-serial"})


@dataclass(frozen=True)
class PauseDecision:
    allowed: bool
    reason: str
    budget_s: float | None = None
    stalls_peers: bool = False


def audit_for(profile: ExecutionProfile) -> PauseAudit | None:
    """The audit entry for a profile, or None when the profile was not audited."""
    if profile.execution_mode not in _AUDITED_MODES:
        return None
    return _OUTER_AUDIT.get(profile.outer_protocol)


def pause_decision(
    profile: ExecutionProfile | None,
    *,
    outer_phase: str,
    expected_pause_s: float,
    budget_mode: bool = False,
    idle_flow_timeout_s: float | None = None,
) -> PauseDecision:
    """May the island pause now for ``expected_pause_s``? Fail closed."""
    if profile is None:
        return PauseDecision(False, "unknown profile: reconfiguration disabled")
    if outer_phase not in OUTER_PHASES:
        return PauseDecision(False, f"unknown outer phase {outer_phase!r}")
    audit = audit_for(profile)
    if audit is None:
        return PauseDecision(
            False, f"profile {profile.execution_mode}/{profile.outer_protocol} not audited"
        )
    if not audit.certified:
        return PauseDecision(False, f"{audit.outer_protocol}: {audit.note}")
    if outer_phase != PAUSABLE_PHASE:
        return PauseDecision(False, f"phase {outer_phase!r} vetoes a pause")
    if budget_mode and audit.outer_protocol != "none":
        return PauseDecision(False, "learner-budget mode: BUDGET_DONE lease may expire")
    if not isinstance(expected_pause_s, (int, float)) or expected_pause_s <= 0:
        return PauseDecision(False, "expected pause must be a positive bound")
    limits = [x for x in (audit.budget_s, idle_flow_timeout_s) if x is not None]
    budget = min(limits) if limits else None
    if idle_flow_timeout_s is not None and audit.outer_protocol == "none":
        budget = None  # no outer connection to keep alive
    if budget is not None and expected_pause_s > budget:
        return PauseDecision(
            False, f"expected pause {expected_pause_s:.0f}s exceeds budget {budget:.0f}s",
            budget, audit.stalls_peers,
        )
    return PauseDecision(True, audit.note, budget, audit.stalls_peers)
