"""Per-round trajectory evidence of a full-parameter island.

A full-parameter role stream binds every local round to the exact accepted
batch (:class:`~yeto.rl.trajectory_evidence.TrajectoryBatchEvidence`).  The
rollout process writes that evidence into the island's private directory
before Miles discards the sample metadata; the sync reads it back at the
round boundary (``session.read_round_evidence``).

Runs inside the Miles rollout process.  Import-light: nothing here imports
miles or torch.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def evidence_enabled(args: Any) -> bool:
    """Whether this island binds rounds to trajectory evidence (full-parameter)."""

    return getattr(args, "yeto_rl_trajectory_evidence_dir", None) is not None


def persist_published_round_evidence(
    args: Any, groups: Sequence[Any], policy_token: str | None
) -> None:
    """Write one accepted batch, bound to the policy token the driver published.

    The caller (the ports rollout hook) calls this only when
    :func:`evidence_enabled`; it is the one writer of round evidence.  The
    driver publishes ``yeto:<rollout_id>:<policy_tensor_hash>`` before each
    rollout (``MilesRolloutPool``); that token is the batch's round and
    behavior policy.  The evidence kind and schema come from the launch
    context bound on ``args``; they are never inferred from sample metadata.
    """

    from yeto.rl.core import parse_policy_snapshot_token
    from yeto.rl.trajectory_evidence import (
        EVIDENCE_KINDS,
        EVIDENCE_SCHEMA_VERSIONS,
        build_trajectory_batch_evidence,
        write_trajectory_batch_evidence,
    )

    if not policy_token:
        raise RuntimeError("trajectory evidence needs the published policy token")
    evidence_kind = getattr(args, "yeto_rl_trajectory_evidence_kind", None)
    if evidence_kind not in EVIDENCE_KINDS:
        raise RuntimeError("trajectory evidence kind is not bound by the launch context")
    schema_version = getattr(args, "yeto_rl_trajectory_evidence_schema_version", None)
    if schema_version not in EVIDENCE_SCHEMA_VERSIONS:
        raise RuntimeError("trajectory evidence schema is not bound by the launch context")
    rollout_id, policy_hash = parse_policy_snapshot_token(policy_token)
    evidence = build_trajectory_batch_evidence(
        groups,
        rollout_id=rollout_id,
        behavior_policy_hash=policy_hash,
        reward_contract_hash=getattr(args, "yeto_rl_reward_sha256", None),
        evidence_kind=evidence_kind,
        schema_version=schema_version,
    )
    write_trajectory_batch_evidence(args.yeto_rl_trajectory_evidence_dir, evidence)
