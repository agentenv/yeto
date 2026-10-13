"""Island departure -> LEAVE reason code (rl-island-graceful-leave design D7).

The driver is the only place that knows why an island stops, so it picks the
code here and hands it to ``sync.close(reason)``; a bridge only sends it.
Only the class of the departure goes on the wire (one byte); the exception
type and text go to the island's event tape.

The table names exception classes by ``(module, class)`` and looks them up in
``sys.modules`` only: an exception whose module was never imported cannot be
raised, so nothing is imported here (no cycle with ``driver``, no torch).
Order matters: the first matching row wins, so subclasses come before their
bases (``IslandRecoveryRequired`` before the generic ``DriverError``).
Anything unmatched is ``INFRA_FAILED`` (D7 Q2: OOM, Ray, network).
"""

from __future__ import annotations

import sys

from yeto.rl.elastic_client import LeaveReason

_E = "yeto.rl.engine."

# (reason, ((module, class), ...)); first matching row wins.
_TABLE: tuple[tuple[LeaveReason, tuple[tuple[str, str], ...]], ...] = (
    (LeaveReason.STOPPED, (("builtins", "KeyboardInterrupt"), ("builtins", "SystemExit"))),
    (LeaveReason.INFRA_FAILED, (
        # node or trainer lost, reconfiguration / rebuild failed
        (_E + "driver", "IslandRecoveryRequired"),
        (_E + "driver", "RebuildNotStarted"),
        (_E + "controller", "RecoveryRequired"),
        (_E + "errors", "RecoveryRequired"),
        (_E + "controller", "TransactionFailed"),
        (_E + "controller", "RebuildRefused"),
        (_E + "trainer_transition", "TrainerTransitionFailed"),
        ("yeto.rl.elastic_client", "ElasticRejoinError"),
        # storage / resume state (D7 Q4)
        (_E + "cut", "CutError"),
        (_E + "journal", "JournalError"),
        (_E + "resume", "StoreIntegrityError"),
        (_E + "ledger", "LedgerError"),
        (_E + "island_ledger", "LedgerError"),
    )),
    (LeaveReason.CONFIG_MISMATCH, (
        (_E + "capabilities", "CapabilityMismatch"),
        (_E + "backend_identity", "BackendIdentityMismatch"),
        (_E + "runtime_manifest", "ManifestMismatch"),
        (_E + "execution_profile", "ProfileError"),
        (_E + "algorithm", "AlgorithmSpecError"),
        (_E + "selection", "UnsupportedPortsCombination"),
        (_E + "multinode", "TopologyError"),
        (_E + "backends", "UnknownBackend"),
        (_E + "trainable_state", "UnsupportedLayoutError"),
        ("yeto.rl.elastic_client", "ElasticJoinRefused"),  # tape only: never a member (D7 Q6)
    )),
    (LeaveReason.TRAINING_FAILED, (
        (_E + "driver", "DriverError"),  # Publication/PolicyIdentity/RoundFailed and checks
        ("yeto.rl.adapters.miles.publish", "PublicationError"),  # Miles publish refusal
        ("yeto.rl.core", "StrictRlInvariantError"),
        (_E + "bridges", "CrossChannelCommitError"),
        (_E + "policy_digest", "PolicyDigestError"),
        (_E + "overlap", "OverlapGuardError"),
        (_E + "version_segments", "ProvenanceError"),
        (_E + "sample_pool", "SampleIndexError"),
        (_E + "policy_age", "PolicyAgeError"),
        ("yeto.rl.grad_audit", "GradAuditError"),
        ("yeto.rl.applied_lr", "AppliedLrError"),
        ("yeto.rl.critic_state", "CriticStateError"),
        ("yeto.rl.tbench_outcome", "UntrustedTBenchOutcome"),
    )),
)


def _loaded(module: str, name: str) -> type | None:
    cls = getattr(sys.modules.get(module), name, None)
    return cls if isinstance(cls, type) else None


def leave_reason_for(error: BaseException | None) -> LeaveReason:
    """The LEAVE code for an island that stops with ``error`` (None: finished).
    A plain ``DriverError`` raised ``from`` another error (e.g. a wrapped
    ``ProfileError``) takes the code of that cause."""
    if error is None:
        return LeaveReason.FINISHED
    if type(error) is _loaded(_E + "driver", "DriverError") and error.__cause__ is not None:
        return leave_reason_for(error.__cause__)
    return _table_reason(error)


def _table_reason(error: BaseException) -> LeaveReason:
    for reason, names in _TABLE:
        classes = tuple(c for c in (_loaded(m, n) for m, n in names) if c is not None)
        if classes and isinstance(error, classes):
            return reason
    return LeaveReason.INFRA_FAILED


def sendable(reason: int) -> LeaveReason:
    """An island never sends LEASE_EXPIRED (syncer-only) or an unknown code:
    both are programming errors and go out as INFRA_FAILED."""
    try:
        code = LeaveReason(reason)
    except ValueError:
        return LeaveReason.INFRA_FAILED
    return LeaveReason.INFRA_FAILED if code is LeaveReason.LEASE_EXPIRED else code
