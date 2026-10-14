"""Closed trajectory evidence for replayable RL updates.

Schema history:

* 1 -- flat SecRLEnv envelopes; outcome hashes included the reward HMAC.
* 2 -- benchmark-neutral envelopes with active-token and compaction-segment
  evidence; outcome hashes included the reward HMAC.
* 3 -- the schema 2 layout with outcome hashes over the canonical outcome
  only (rewards are no longer signed); every trajectory carries the round's
  policy (version = rollout).
* 4 -- the schema 3 layout where every trajectory carries the policy that
  generated it (P0-S: rollout and training may differ by a bounded number of
  versions). The batch keeps the round's policy hash; a trajectory of the
  round's own version must carry it, older trajectories carry their own.

New evidence is schema 4 when the caller names each trajectory's policy (the
ports rollout hook) and schema 3 otherwise.  Schema 1 and 2 files remain
readable: their structure and batch hash are checked, but the HMAC that was
mixed into their outcome hashes is not (and cannot be) re-verified.

Evidence kinds name where a trajectory's reward comes from:

* ``secrlenv`` and ``terminal-bench-2.1`` -- a trusted grader verdict carried
  in the sample metadata; the reward must equal the verdict.
* ``reward-function`` -- the output of the island's hash-bound reward function
  (``sample.reward``), e.g. a GSM8K answer check.  There is no separate
  verdict to verify: the evidence binds the reward, the prompt and the tokens,
  and the reward function itself is bound by ``reward_contract_hash``.
  Trajectories of this kind must not carry a grader verdict.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from .contracts import TrajectoryEnvelope

_LEGACY_SCHEMA = 1
_SIGNED_LAYERED_SCHEMA = 2
_SCHEMA = 3
_PER_TRAJECTORY_POLICY_SCHEMA = 4
_LAYERED_SCHEMAS = frozenset({_SIGNED_LAYERED_SCHEMA, _SCHEMA, _PER_TRAJECTORY_POLICY_SCHEMA})
_READABLE_SCHEMAS = frozenset({_LEGACY_SCHEMA, *_LAYERED_SCHEMAS})
_SHA256 = frozenset("0123456789abcdef")
REWARD_FUNCTION_EVIDENCE_KIND = "reward-function"
EVIDENCE_KINDS = frozenset({"secrlenv", "terminal-bench-2.1", REWARD_FUNCTION_EVIDENCE_KIND})
_GRADER_VERDICT_KEYS = ("secrlenv_trusted_outcome", "tbench_trusted_outcome")
# The launch context still names the signed-era layout (1 or 2) it was
# attested with; it must be bound, but new evidence is schema 3 or 4.
LAUNCH_CONTEXT_SCHEMA_VERSIONS = frozenset({_LEGACY_SCHEMA, _SIGNED_LAYERED_SCHEMA})
_LEGACY_ENVELOPE_FIELDS = (
    "trajectory_id",
    "task_id",
    "prompt_group_id",
    "sample_index",
    "behavior_policy_version",
    "behavior_policy_hash",
    "token_ids",
    "response_token_count",
    "behavior_logprobs_hash",
    "reward",
    "reward_contract_hash",
    "cleanup_evidence_hash",
)
_LAYERED_ENVELOPE_FIELDS = frozenset(field.name for field in fields(TrajectoryEnvelope))

BehaviorPolicyOf = Callable[[Sequence[Any]], tuple[int, str]]
"""The (version, weight hash) of the policy that generated one logical
trajectory, given its segments (one sample, or the samples of a compaction)."""


@dataclass(frozen=True)
class TrajectoryBatchEvidence:
    rollout_id: int
    behavior_policy_hash: str
    input_batch_hash: str
    trained_tokens: int
    envelopes: tuple[TrajectoryEnvelope, ...]
    evidence_kind: str = "secrlenv"
    schema_version: int = _SCHEMA
    # Schema 4 only: the policy-age limit every trajectory was checked against.
    max_policy_age: int | None = None

    @property
    def trajectory_ids(self) -> tuple[str, ...]:
        return tuple(envelope.trajectory_id for envelope in self.envelopes)


@dataclass(frozen=True)
class _VerifiedOutcome:
    task_id: str
    episode_id: str
    sample_id: str | None
    reward: float
    canonical: bytes


def build_trajectory_batch_evidence(
    groups: Sequence[Sequence[Any]],
    *,
    rollout_id: int,
    behavior_policy_hash: str,
    reward_contract_hash: str,
    evidence_kind: str = "secrlenv",
    behavior_policy_of: BehaviorPolicyOf | None = None,
    max_policy_age: int | None = None,
) -> TrajectoryBatchEvidence:
    """Validate accepted samples and bind their exact training batch.

    ``evidence_kind`` is an explicit, hash-bound runtime choice. It is never
    inferred from arbitrary sample metadata. The result is schema 3.

    ``behavior_policy_of`` selects schema 4: it names the policy of every
    logical trajectory (``behavior_policy_hash`` stays the round's policy),
    and ``max_policy_age`` bounds how many versions it may lag the round.
    Without it every trajectory carries ``(rollout_id, behavior_policy_hash)``.
    """

    _nonnegative_int("rollout_id", rollout_id)
    _sha256("behavior_policy_hash", behavior_policy_hash)
    _sha256("reward_contract_hash", reward_contract_hash)
    if evidence_kind not in EVIDENCE_KINDS:
        raise ValueError(f"unsupported trajectory evidence kind: {evidence_kind!r}")
    if not groups or any(
        not isinstance(group, Sequence)
        or isinstance(group, (str, bytes))
        or not group
        for group in groups
    ):
        raise ValueError("trajectory evidence requires non-empty RL groups")

    if behavior_policy_of is not None:
        schema_version = _PER_TRAJECTORY_POLICY_SCHEMA
        _nonnegative_int("max_policy_age", max_policy_age)
    elif max_policy_age is not None:
        raise ValueError("a policy-age limit is recorded only with per-trajectory policies")
    else:
        schema_version = _SCHEMA
    envelopes: list[TrajectoryEnvelope] = []
    seen_sample_indices: set[int] = set()
    seen_groups: set[int] = set()
    batch_compaction_context_budget: int | None = None

    for group in groups:
        current_group_index: int | None = None
        for logical_sample in group:
            compacted = isinstance(logical_sample, list)
            segments = logical_sample if compacted else [logical_sample]
            if not segments:
                raise ValueError("trajectory evidence has an empty compacted rollout")

            behavior_version, behavior_hash = _behavior_policy(
                behavior_policy_of, segments, rollout_id, behavior_policy_hash, max_policy_age
            )
            first = segments[0]
            group_index = getattr(first, "group_index", None)
            sample_index = getattr(first, "index", None)
            _nonnegative_int("sample.group_index", group_index)
            _nonnegative_int("sample.index", sample_index)
            if current_group_index is None:
                if group_index in seen_groups:
                    raise ValueError("trajectory batch repeats an RL group")
                current_group_index = group_index
                seen_groups.add(group_index)
            elif group_index != current_group_index:
                raise ValueError("trajectory RL group mixes group identities")
            if sample_index in seen_sample_indices:
                raise ValueError(
                    "trajectory batch contains duplicate logical sample indexes"
                )
            seen_sample_indices.add(sample_index)

            compaction_id: str | None = None
            compaction_context_budget: int | None = None
            expected_outcome: tuple[str, str, str | None, float, bytes] | None = None
            for segment_ordinal, sample in enumerate(segments):
                if (
                    getattr(sample, "group_index", None) != group_index
                    or getattr(sample, "index", None) != sample_index
                ):
                    raise ValueError(
                        "compaction segments changed their logical sample identity"
                    )
                status = getattr(getattr(sample, "status", None), "value", None)
                if status not in {"completed", "truncated"}:
                    raise ValueError("trajectory batch contains a non-trainable sample")
                metadata = getattr(sample, "metadata", None)
                if not isinstance(metadata, Mapping):
                    raise TypeError("trajectory sample metadata is not an object")

                segment_index: int | None = None
                segment_type: str | None = None
                if compacted:
                    (
                        compaction_id,
                        segment_index,
                        segment_type,
                        compaction_context_budget,
                    ) = _compaction_identity(
                        metadata,
                        expected_id=compaction_id,
                        expected_index=segment_ordinal,
                        expected_context_budget=compaction_context_budget,
                    )
                    if batch_compaction_context_budget is None:
                        batch_compaction_context_budget = compaction_context_budget
                    elif compaction_context_budget != batch_compaction_context_budget:
                        raise ValueError(
                            "compaction context budget changed within the batch"
                        )
                elif any(
                    metadata.get(name) is not None
                    for name in (
                        "compaction_schema_version",
                        "compaction_trajectory_id",
                        "compaction_segment_index",
                        "compaction_segment_type",
                        "compaction_context_budget",
                    )
                ):
                    raise ValueError(
                        "compaction-marked trajectory was not preserved as segments"
                    )

                verified = _verified_outcome(sample, metadata, evidence_kind)
                outcome_identity = (
                    verified.task_id,
                    verified.episode_id,
                    verified.sample_id,
                    verified.reward,
                    verified.canonical,
                )
                if expected_outcome is None:
                    expected_outcome = outcome_identity
                elif outcome_identity != expected_outcome:
                    raise ValueError(
                        "compaction segments do not share one outcome"
                    )
                _require_sample_reward(sample, verified.reward)

                tokens = _tokens(sample)
                response_length = _response_length(sample, tokens)
                logprobs = getattr(sample, "rollout_log_probs", None)
                behavior_logprobs_hash = _logprobs_hash(
                    logprobs, response_length
                )
                if evidence_kind == "terminal-bench-2.1" and logprobs is None:
                    raise ValueError(
                        "Terminal-Bench trajectory has no behavior log probabilities"
                    )
                token_hash = _hash_parts(
                    b"yeto-trajectory-tokens-v1\0",
                    struct.pack(f"<{len(tokens)}Q", *tokens),
                )
                cleanup_evidence_hash = _outcome_evidence_hash(
                    evidence_kind, verified
                )

                (
                    active_token_count,
                    loss_mask_hash,
                    active_token_ids_hash,
                    active_behavior_logprobs_hash,
                ) = _active_token_evidence(
                    sample,
                    tokens=tokens,
                    response_length=response_length,
                    logprobs=logprobs,
                )
                identity_parts = [
                    str(rollout_id).encode("ascii"),
                    str(group_index).encode("ascii"),
                    str(sample_index).encode("ascii"),
                    evidence_kind.encode("ascii"),
                    verified.task_id.encode("utf-8"),
                    verified.episode_id.encode("utf-8"),
                    (verified.sample_id or "").encode("utf-8"),
                    bytes.fromhex(token_hash),
                ]
                if compacted:
                    assert compaction_id is not None
                    assert segment_index is not None
                    assert segment_type is not None
                    assert compaction_context_budget is not None
                    identity_parts.extend(
                        [
                            compaction_id.encode("utf-8"),
                            str(segment_index).encode("ascii"),
                            segment_type.encode("ascii"),
                            str(compaction_context_budget).encode("ascii"),
                        ]
                    )
                if schema_version == _PER_TRAJECTORY_POLICY_SCHEMA:
                    identity_parts.extend(
                        [str(behavior_version).encode("ascii"), bytes.fromhex(behavior_hash)]
                    )
                trajectory_id = _hash_parts(
                    b"yeto-trajectory-id-v4\0"
                    if schema_version == _PER_TRAJECTORY_POLICY_SCHEMA
                    else b"yeto-trajectory-id-v3\0",
                    *identity_parts,
                )
                envelopes.append(
                    TrajectoryEnvelope(
                        trajectory_id=trajectory_id,
                        task_id=verified.task_id,
                        prompt_group_id=f"r{rollout_id}:g{group_index}",
                        sample_index=sample_index,
                        behavior_policy_version=behavior_version,
                        behavior_policy_hash=behavior_hash,
                        token_ids=tokens,
                        response_token_count=response_length,
                        behavior_logprobs_hash=behavior_logprobs_hash,
                        reward=verified.reward,
                        reward_contract_hash=reward_contract_hash,
                        cleanup_evidence_hash=cleanup_evidence_hash,
                        evidence_kind=evidence_kind,
                        active_token_count=active_token_count,
                        loss_mask_hash=loss_mask_hash,
                        active_token_ids_hash=active_token_ids_hash,
                        active_behavior_logprobs_hash=(
                            active_behavior_logprobs_hash
                        ),
                        compaction_trajectory_id=compaction_id,
                        compaction_segment_index=segment_index,
                        compaction_segment_type=segment_type,
                        compaction_context_budget=compaction_context_budget,
                    )
                )

            if (
                compacted
                and segments[-1].metadata["compaction_segment_type"] != "execution"
            ):
                raise ValueError("compaction trajectory ended in a summary segment")

    ordered = tuple(envelopes)
    _validate_envelope_order(ordered, schema_version=schema_version)
    trained_tokens = _trained_token_count(ordered)
    if trained_tokens < 1:
        raise ValueError("trajectory evidence contains no active training tokens")
    input_batch_hash = _envelope_batch_hash(
        ordered,
        trained_tokens,
        schema_version=schema_version,
        evidence_kind=evidence_kind,
        max_policy_age=max_policy_age,
    )
    return TrajectoryBatchEvidence(
        rollout_id,
        behavior_policy_hash,
        input_batch_hash,
        trained_tokens,
        ordered,
        evidence_kind,
        schema_version,
        max_policy_age,
    )


def write_trajectory_batch_evidence(
    directory: str | Path,
    evidence: TrajectoryBatchEvidence,
) -> Path:
    """Publish one immutable private replay artifact with exclusive creation."""

    _validate_batch_evidence(evidence)
    root = Path(directory).expanduser()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("trajectory evidence root is not a real directory")
    if root.stat().st_mode & 0o077:
        raise RuntimeError("trajectory evidence root is not private")
    path = trajectory_batch_evidence_path(root, evidence.rollout_id)
    payload: dict[str, Any] = {
        "schema": evidence.schema_version,
        "rollout_id": evidence.rollout_id,
        "behavior_policy_hash": evidence.behavior_policy_hash,
        "input_batch_hash": evidence.input_batch_hash,
        "trained_tokens": evidence.trained_tokens,
        "envelopes": [
            _envelope_mapping(envelope, evidence.schema_version)
            for envelope in evidence.envelopes
        ],
    }
    if evidence.schema_version in _LAYERED_SCHEMAS:
        payload["evidence_kind"] = evidence.evidence_kind
    if evidence.schema_version == _PER_TRAJECTORY_POLICY_SCHEMA:
        payload["max_policy_age"] = evidence.max_policy_age
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if read_trajectory_batch_evidence(path) != evidence:
            raise RuntimeError("trajectory rollout evidence already changed") from None
        return path
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            written = handle.write(raw)
            if written != len(raw):
                raise OSError("trajectory evidence write was incomplete")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def trajectory_batch_evidence_path(
    directory: str | Path,
    rollout_id: int,
) -> Path:
    """Return the deterministic cross-process path for one rollout batch."""

    if type(rollout_id) is not int or rollout_id < 0:
        raise ValueError("trajectory evidence rollout ID must be non-negative")
    return Path(directory).expanduser() / f"rollout-{rollout_id:08d}.json"


def read_trajectory_batch_evidence(path: str | Path) -> TrajectoryBatchEvidence:
    source = Path(path).expanduser()
    info = source.lstat()
    if source.is_symlink() or not source.is_file() or info.st_mode & 0o077:
        raise RuntimeError("trajectory evidence file is not private and regular")
    value = json.loads(source.read_bytes())
    if not isinstance(value, dict):
        raise TypeError("trajectory evidence envelope is not an object")
    schema_version = value.get("schema")
    expected_fields = {
        "schema",
        "rollout_id",
        "behavior_policy_hash",
        "input_batch_hash",
        "trained_tokens",
        "envelopes",
    }
    if schema_version in _LAYERED_SCHEMAS:
        expected_fields.add("evidence_kind")
        if schema_version == _PER_TRAJECTORY_POLICY_SCHEMA:
            expected_fields.add("max_policy_age")
    elif schema_version != _LEGACY_SCHEMA:
        raise ValueError("trajectory evidence schema is unsupported")
    if set(value) != expected_fields or not isinstance(value["envelopes"], list):
        raise ValueError("trajectory evidence envelope is malformed")

    envelopes = tuple(
        _envelope_from_mapping(item, schema_version) for item in value["envelopes"]
    )
    evidence = TrajectoryBatchEvidence(
        value["rollout_id"],
        value["behavior_policy_hash"],
        value["input_batch_hash"],
        value["trained_tokens"],
        envelopes,
        value.get("evidence_kind", "secrlenv"),
        schema_version,
        value.get("max_policy_age"),
    )
    _validate_batch_evidence(evidence)
    return evidence


def _verified_outcome(
    sample: Any, metadata: Mapping[str, Any], evidence_kind: str
) -> _VerifiedOutcome:
    if evidence_kind == REWARD_FUNCTION_EVIDENCE_KIND:
        return _reward_function_outcome(sample, metadata)
    if evidence_kind == "terminal-bench-2.1":
        if "secrlenv_trusted_outcome" in metadata:
            raise ValueError("trajectory metadata mixes benchmark evidence kinds")
        from .tbench_outcome import canonical_outcome, verified_outcome

        outcome, reward = verified_outcome(dict(metadata))
        return _VerifiedOutcome(
            task_id=outcome["task_id"],
            episode_id=outcome["episode_id"],
            sample_id=outcome["sample_id"],
            reward=reward,
            canonical=canonical_outcome(outcome),
        )

    if "tbench_trusted_outcome" in metadata:
        raise ValueError("trajectory metadata mixes benchmark evidence kinds")
    from yeto.rl.harness.codex.reward import (
        CLEANUP_ERROR_STATUS,
        INFRASTRUCTURE_STATUS,
        _canonical,
    )
    from yeto.rl.harness.codex.reward import (
        _verified_outcome as verified_secrlenv_outcome,
    )

    outcome, reward = verified_secrlenv_outcome(dict(metadata))
    if outcome["status"] in {INFRASTRUCTURE_STATUS, CLEANUP_ERROR_STATUS}:
        raise ValueError("trajectory has no terminal grader/cleanup evidence")
    task_id = outcome.get("task_id")
    episode_id = outcome.get("episode_id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("trajectory has no task identity")
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("trajectory has no episode identity")
    return _VerifiedOutcome(
        task_id=task_id,
        episode_id=episode_id,
        sample_id=None,
        reward=reward,
        canonical=_canonical(outcome),
    )


def _reward_function_outcome(
    sample: Any, metadata: Mapping[str, Any]
) -> _VerifiedOutcome:
    """The outcome of a trajectory scored by the island's reward function.

    The reward is ``sample.reward`` as the reward function wrote it.  The task
    is the prompt (by content hash) and the episode is the logical sample.
    """

    if any(key in metadata for key in _GRADER_VERDICT_KEYS):
        raise ValueError(
            "reward-function trajectory carries a grader verdict; "
            "bind the grader's evidence kind instead"
        )
    reward = getattr(sample, "reward", None)
    if (
        isinstance(reward, bool)
        or not isinstance(reward, (int, float))
        or not math.isfinite(float(reward))
    ):
        raise ValueError("reward-function trajectory has no finite scalar reward")
    prompt = json.dumps(
        getattr(sample, "prompt", None), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    task_id = "prompt-sha256:" + hashlib.sha256(prompt).hexdigest()
    episode_id = f"sample:{getattr(sample, 'index', None)}"
    canonical = json.dumps(
        {
            "kind": REWARD_FUNCTION_EVIDENCE_KIND,
            "task_id": task_id,
            "episode_id": episode_id,
            "reward": float(reward),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _VerifiedOutcome(
        task_id=task_id,
        episode_id=episode_id,
        sample_id=None,
        reward=float(reward),
        canonical=canonical,
    )


def _compaction_identity(
    metadata: Mapping[str, Any],
    *,
    expected_id: str | None,
    expected_index: int,
    expected_context_budget: int | None,
) -> tuple[str, int, str, int]:
    trajectory_id = metadata.get("compaction_trajectory_id")
    segment_index = metadata.get("compaction_segment_index")
    segment_type = metadata.get("compaction_segment_type")
    context_window = metadata.get("compaction_context_window")
    context_budget = metadata.get("compaction_context_budget")
    if metadata.get("compaction_schema_version") != 1:
        raise ValueError("compaction sample has an unsupported schema")
    if not isinstance(trajectory_id, str) or not trajectory_id:
        raise ValueError("compaction sample has no trajectory identity")
    if expected_id is not None and trajectory_id != expected_id:
        raise ValueError("compaction segments mix trajectory identities")
    if segment_index != expected_index:
        raise ValueError("compaction segment indexes are not contiguous")
    expected_type = "execution" if expected_index % 2 == 0 else "summary"
    if segment_type != expected_type:
        raise ValueError("compaction segment types do not alternate")
    if type(context_window) is not int or context_window != expected_index // 2:
        raise ValueError("compaction segment context window is inconsistent")
    if type(context_budget) is not int or context_budget < 1:
        raise ValueError("compaction context budget is invalid")
    if (
        expected_context_budget is not None
        and context_budget != expected_context_budget
    ):
        raise ValueError("compaction context budget changed within the trajectory")
    return trajectory_id, segment_index, segment_type, context_budget


def _require_sample_reward(sample: Any, verified_reward: float) -> None:
    sample_reward = getattr(sample, "reward", None)
    if (
        isinstance(sample_reward, bool)
        or not isinstance(sample_reward, (int, float))
        or not math.isfinite(float(sample_reward))
        or float(sample_reward) != verified_reward
    ):
        raise ValueError("trajectory reward differs from its outcome")


def _tokens(sample: Any) -> tuple[int, ...]:
    tokens = tuple(getattr(sample, "tokens", ()))
    if not tokens or any(
        type(token) is not int or token < 0 or token >= 2**64 for token in tokens
    ):
        raise ValueError("trajectory tokens are incomplete")
    return tokens


def _response_length(sample: Any, tokens: tuple[int, ...]) -> int:
    response_length = getattr(sample, "response_length", None)
    if (
        type(response_length) is not int
        or response_length < 1
        or response_length > len(tokens)
    ):
        raise ValueError("trajectory response length is invalid")
    return response_length


def _logprobs_hash(values: Any, response_length: int) -> str | None:
    if values is None:
        return None
    if (
        not isinstance(values, list)
        or len(values) != response_length
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        )
    ):
        raise ValueError("trajectory rollout log probabilities are malformed")
    return _hash_parts(
        b"yeto-trajectory-logprobs-v1\0",
        b"".join(struct.pack("<d", float(value)) for value in values),
    )


def _active_token_evidence(
    sample: Any,
    *,
    tokens: tuple[int, ...],
    response_length: int,
    logprobs: Any,
) -> tuple[int, str, str, str | None]:
    raw_mask = getattr(sample, "loss_mask", None)
    if raw_mask is None:
        mask = (1,) * response_length
    elif (
        not isinstance(raw_mask, list)
        or len(raw_mask) != response_length
        or any(type(value) is not int or value not in {0, 1} for value in raw_mask)
    ):
        raise ValueError("trajectory loss mask is malformed")
    else:
        mask = tuple(raw_mask)
    active_positions = tuple(index for index, value in enumerate(mask) if value)
    response_tokens = tokens[-response_length:]
    loss_mask_hash = _hash_parts(
        b"yeto-trajectory-loss-mask-v1\0", bytes(mask)
    )
    active_token_ids_hash = _hash_parts(
        b"yeto-trajectory-active-token-ids-v1\0",
        b"".join(
            struct.pack("<QQ", index, response_tokens[index])
            for index in active_positions
        ),
    )
    active_logprobs_hash = None
    if logprobs is not None:
        active_logprobs_hash = _hash_parts(
            b"yeto-trajectory-active-logprobs-v1\0",
            b"".join(
                struct.pack("<Qd", index, float(logprobs[index]))
                for index in active_positions
            ),
        )
    return (
        len(active_positions),
        loss_mask_hash,
        active_token_ids_hash,
        active_logprobs_hash,
    )


def _outcome_evidence_hash(
    evidence_kind: str, verified: _VerifiedOutcome
) -> str:
    domain = {
        "terminal-bench-2.1": b"yeto-tbench-verdict-evidence-v2\0",
        REWARD_FUNCTION_EVIDENCE_KIND: b"yeto-reward-function-evidence-v1\0",
    }.get(evidence_kind, b"yeto-secrlenv-cleanup-evidence-v2\0")
    return _hash_parts(domain, verified.canonical)


def _trained_token_count(envelopes: tuple[TrajectoryEnvelope, ...]) -> int:
    active = [envelope.active_token_count for envelope in envelopes]
    if any(value is not None for value in active):
        if any(value is None for value in active):
            raise ValueError("trajectory evidence mixes token-accounting schemas")
        return sum(value for value in active if value is not None)
    return sum(envelope.response_token_count for envelope in envelopes)


def _envelope_mapping(
    envelope: TrajectoryEnvelope, schema_version: int
) -> dict[str, Any]:
    value = asdict(envelope)
    if schema_version == _LEGACY_SCHEMA:
        return {name: value[name] for name in _LEGACY_ENVELOPE_FIELDS}
    return value


def _envelope_from_mapping(value: Any, schema_version: int) -> TrajectoryEnvelope:
    expected = (
        frozenset(_LEGACY_ENVELOPE_FIELDS)
        if schema_version == _LEGACY_SCHEMA
        else _LAYERED_ENVELOPE_FIELDS
    )
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("trajectory evidence item fields are malformed")
    return TrajectoryEnvelope(
        **{
            **value,
            "token_ids": tuple(value["token_ids"]),
        }
    )


def _validate_envelope_order(
    envelopes: tuple[TrajectoryEnvelope, ...], *, schema_version: int
) -> None:
    if not envelopes:
        raise ValueError("trajectory evidence has no envelopes")
    if len({item.trajectory_id for item in envelopes}) != len(envelopes):
        raise ValueError("trajectory evidence repeats a trajectory ID")
    if schema_version == _LEGACY_SCHEMA:
        indexes = tuple(item.sample_index for item in envelopes)
        if indexes != tuple(sorted(indexes)) or len(set(indexes)) != len(indexes):
            raise ValueError("trajectory samples are not in canonical index order")
        return

    order = tuple(
        (
            item.sample_index,
            -1
            if item.compaction_segment_index is None
            else item.compaction_segment_index,
        )
        for item in envelopes
    )
    if order != tuple(sorted(order)):
        raise ValueError("trajectory segments are not in canonical index order")

    logical: dict[int, str | None] = {}
    compacted: dict[str, list[TrajectoryEnvelope]] = {}
    counts: dict[int, int] = {}
    for item in envelopes:
        previous = logical.setdefault(item.sample_index, item.compaction_trajectory_id)
        if previous != item.compaction_trajectory_id:
            raise ValueError("trajectory sample index maps to multiple identities")
        counts[item.sample_index] = counts.get(item.sample_index, 0) + 1
        if item.compaction_trajectory_id is not None:
            compacted.setdefault(item.compaction_trajectory_id, []).append(item)
    if any(
        count != 1 and logical[index] is None for index, count in counts.items()
    ):
        raise ValueError("trajectory evidence repeats a flat sample index")
    compaction_context_budgets = {
        item.compaction_context_budget
        for item in envelopes
        if item.compaction_trajectory_id is not None
    }
    if len(compaction_context_budgets) > 1:
        raise ValueError("compaction context budget changed within the batch")

    for segments in compacted.values():
        indices = [item.compaction_segment_index for item in segments]
        if indices != list(range(len(segments))):
            raise ValueError(
                "trajectory evidence has non-contiguous compaction segments"
            )
        types = [item.compaction_segment_type for item in segments]
        if types != [
            "execution" if index % 2 == 0 else "summary"
            for index in range(len(segments))
        ] or types[-1] != "execution":
            raise ValueError("trajectory evidence has an invalid compaction sequence")
        shared = {
            (
                item.task_id,
                item.prompt_group_id,
                item.sample_index,
                item.behavior_policy_version,
                item.behavior_policy_hash,
                item.reward,
                item.reward_contract_hash,
                item.cleanup_evidence_hash,
                item.evidence_kind,
                item.compaction_context_budget,
            )
            for item in segments
        }
        if len(shared) != 1:
            raise ValueError("compaction segment evidence changed logical outcome")


def _validate_batch_evidence(evidence: TrajectoryBatchEvidence) -> None:
    _nonnegative_int("rollout_id", evidence.rollout_id)
    _sha256("behavior_policy_hash", evidence.behavior_policy_hash)
    _sha256("input_batch_hash", evidence.input_batch_hash)
    _nonnegative_int("trained_tokens", evidence.trained_tokens)
    if evidence.trained_tokens < 1:
        raise ValueError("trajectory evidence trained-token count is invalid")
    if evidence.schema_version not in _READABLE_SCHEMAS:
        raise ValueError("trajectory evidence schema is unsupported")
    if evidence.evidence_kind not in EVIDENCE_KINDS:
        raise ValueError("trajectory evidence kind is unsupported")
    if (
        evidence.schema_version == _LEGACY_SCHEMA
        and evidence.evidence_kind != "secrlenv"
    ):
        raise ValueError("legacy trajectory evidence can only represent SecRLEnv")
    if evidence.schema_version in _LAYERED_SCHEMAS and any(
        item.evidence_kind != evidence.evidence_kind
        or item.active_token_count is None
        or item.loss_mask_hash is None
        or item.active_token_ids_hash is None
        for item in evidence.envelopes
    ):
        raise ValueError("layered trajectory evidence item is incomplete")
    _validate_behavior_policies(evidence)
    _validate_envelope_order(
        evidence.envelopes, schema_version=evidence.schema_version
    )
    if _trained_token_count(evidence.envelopes) != evidence.trained_tokens:
        raise ValueError("trajectory evidence trained-token aggregate changed")
    if (
        _envelope_batch_hash(
            evidence.envelopes,
            evidence.trained_tokens,
            schema_version=evidence.schema_version,
            evidence_kind=evidence.evidence_kind,
            max_policy_age=evidence.max_policy_age,
        )
        != evidence.input_batch_hash
    ):
        raise ValueError("trajectory evidence batch hash changed")


def _behavior_policy(
    behavior_policy_of: BehaviorPolicyOf | None,
    segments: Sequence[Any],
    rollout_id: int,
    round_policy_hash: str,
    max_policy_age: int | None,
) -> tuple[int, str]:
    if behavior_policy_of is None:
        return rollout_id, round_policy_hash
    version, policy_hash = behavior_policy_of(segments)
    _nonnegative_int("behavior_policy_version", version)
    _sha256("behavior_policy_hash", policy_hash)
    problem = _behavior_policy_problem(
        version, policy_hash, rollout_id, round_policy_hash, max_policy_age
    )
    if problem is not None:
        raise ValueError(problem)
    return version, policy_hash


def _behavior_policy_problem(
    version: int,
    policy_hash: str,
    rollout_id: int,
    round_policy_hash: str,
    max_policy_age: int | None,
) -> str | None:
    """Why a schema-4 trajectory policy does not fit its round (None = fits)."""
    if version > rollout_id:
        return "trajectory was generated by a policy newer than its round"
    if version == rollout_id and policy_hash != round_policy_hash:
        return "trajectory names its round's version with other weights"
    if max_policy_age is None or rollout_id - version > max_policy_age:
        return "trajectory policy is older than the batch's policy-age limit"
    return None


def _validate_behavior_policies(evidence: TrajectoryBatchEvidence) -> None:
    """Schemas 1/2: every trajectory carries the round's policy. Schema 4: no
    trajectory is newer than its round, and one of the round's own version
    carries the round's hash."""
    per_trajectory = evidence.schema_version == _PER_TRAJECTORY_POLICY_SCHEMA
    if per_trajectory:
        _nonnegative_int("max_policy_age", evidence.max_policy_age)
    elif evidence.max_policy_age is not None:
        raise ValueError("trajectory evidence records a policy-age limit outside schema 4")
    for item in evidence.envelopes:
        _nonnegative_int("behavior_policy_version", item.behavior_policy_version)
        _sha256("behavior_policy_hash", item.behavior_policy_hash)
        if per_trajectory:
            problem = _behavior_policy_problem(
                item.behavior_policy_version,
                item.behavior_policy_hash,
                evidence.rollout_id,
                evidence.behavior_policy_hash,
                evidence.max_policy_age,
            )
        elif (item.behavior_policy_version, item.behavior_policy_hash) != (
            evidence.rollout_id,
            evidence.behavior_policy_hash,
        ):
            problem = "trajectory does not carry its round's policy"
        else:
            problem = None
        if problem is not None:
            raise ValueError(f"trajectory evidence behavior policy: {problem}")


def _envelope_batch_hash(
    envelopes: tuple[TrajectoryEnvelope, ...],
    trained_tokens: int,
    *,
    schema_version: int = _LEGACY_SCHEMA,
    evidence_kind: str = "secrlenv",
    max_policy_age: int | None = None,
) -> str:
    payload: dict[str, Any] = {
        "trained_tokens": trained_tokens,
        "envelopes": [
            _envelope_mapping(envelope, schema_version) for envelope in envelopes
        ],
    }
    domain = b"yeto-trajectory-batch-v1\0"
    if schema_version in _LAYERED_SCHEMAS:
        payload.update({"schema": schema_version, "evidence_kind": evidence_kind})
        if schema_version == _PER_TRAJECTORY_POLICY_SCHEMA:
            payload["max_policy_age"] = max_policy_age
        domain = f"yeto-trajectory-batch-v{schema_version}\0".encode("ascii")
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return _hash_parts(domain, raw)


def _hash_parts(domain: bytes, *parts: bytes) -> str:
    digest = hashlib.sha256(domain)
    for part in parts:
        digest.update(len(part).to_bytes(8, "little"))
        digest.update(part)
    return digest.hexdigest()


def _sha256(name: str, value: Any) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256 for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")


def _nonnegative_int(name: str, value: Any) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
