"""Closed, authenticated trajectory evidence for replayable RL updates."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from .contracts import TrajectoryEnvelope

_LEGACY_SCHEMA = 1
_SCHEMA = 2
_SHA256 = frozenset("0123456789abcdef")
_EVIDENCE_KINDS = frozenset({"secrlenv", "terminal-bench-2.1"})
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
_V2_ENVELOPE_FIELDS = frozenset(field.name for field in fields(TrajectoryEnvelope))


@dataclass(frozen=True)
class TrajectoryBatchEvidence:
    rollout_id: int
    behavior_policy_hash: str
    input_batch_hash: str
    trained_tokens: int
    envelopes: tuple[TrajectoryEnvelope, ...]
    evidence_kind: str = "secrlenv"
    schema_version: int = _SCHEMA

    @property
    def trajectory_ids(self) -> tuple[str, ...]:
        return tuple(envelope.trajectory_id for envelope in self.envelopes)


@dataclass(frozen=True)
class _VerifiedOutcome:
    task_id: str
    episode_id: str
    sample_id: str | None
    reward: float
    mac: str
    canonical: bytes


def build_trajectory_batch_evidence(
    groups: Sequence[Sequence[Any]],
    *,
    rollout_id: int,
    behavior_policy_hash: str,
    reward_contract_hash: str,
    evidence_kind: str = "secrlenv",
    schema_version: int | None = None,
) -> TrajectoryBatchEvidence:
    """Authenticate accepted samples and bind their exact training batch.

    ``evidence_kind`` is an explicit, hash-bound runtime choice. It is never
    inferred from arbitrary sample metadata. Schema 1 remains the byte-exact
    flat SecRLEnv representation; Terminal-Bench and nested compaction use
    schema 2 with active-token and composite-segment evidence.
    """

    _nonnegative_int("rollout_id", rollout_id)
    _sha256("behavior_policy_hash", behavior_policy_hash)
    _sha256("reward_contract_hash", reward_contract_hash)
    if evidence_kind not in _EVIDENCE_KINDS:
        raise ValueError(f"unsupported trajectory evidence kind: {evidence_kind!r}")
    if not groups or any(
        not isinstance(group, Sequence)
        or isinstance(group, (str, bytes))
        or not group
        for group in groups
    ):
        raise ValueError("trajectory evidence requires non-empty RL groups")

    nested = any(isinstance(sample, list) for group in groups for sample in group)
    inferred_schema = (
        _SCHEMA if evidence_kind != "secrlenv" or nested else _LEGACY_SCHEMA
    )
    if schema_version is None:
        schema_version = inferred_schema
    elif schema_version not in {_LEGACY_SCHEMA, _SCHEMA}:
        raise ValueError("trajectory evidence schema is unsupported")
    if schema_version == _LEGACY_SCHEMA and (
        evidence_kind != "secrlenv" or nested
    ):
        raise ValueError(
            "legacy trajectory evidence cannot represent this benchmark/layout"
        )
    envelopes: list[TrajectoryEnvelope] = []
    seen_sample_indices: set[int] = set()
    seen_groups: set[int] = set()
    batch_compaction_context_budget: int | None = None

    for group in groups:
        signed_group_index: int | None = None
        for logical_sample in group:
            compacted = isinstance(logical_sample, list)
            segments = logical_sample if compacted else [logical_sample]
            if not segments:
                raise ValueError("trajectory evidence has an empty compacted rollout")

            first = segments[0]
            group_index = getattr(first, "group_index", None)
            sample_index = getattr(first, "index", None)
            _nonnegative_int("sample.group_index", group_index)
            _nonnegative_int("sample.index", sample_index)
            if signed_group_index is None:
                if group_index in seen_groups:
                    raise ValueError("trajectory batch repeats an RL group")
                signed_group_index = group_index
                seen_groups.add(group_index)
            elif group_index != signed_group_index:
                raise ValueError("trajectory RL group mixes group identities")
            if sample_index in seen_sample_indices:
                raise ValueError(
                    "trajectory batch contains duplicate logical sample indexes"
                )
            seen_sample_indices.add(sample_index)

            compaction_id: str | None = None
            compaction_context_budget: int | None = None
            expected_outcome: tuple[str, str, str | None, float, str] | None = None
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

                verified = _verified_outcome(metadata, evidence_kind)
                outcome_identity = (
                    verified.task_id,
                    verified.episode_id,
                    verified.sample_id,
                    verified.reward,
                    verified.mac,
                )
                if expected_outcome is None:
                    expected_outcome = outcome_identity
                elif outcome_identity != expected_outcome:
                    raise ValueError(
                        "compaction segments do not share one signed outcome"
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

                if schema_version == _LEGACY_SCHEMA:
                    trajectory_id = _hash_parts(
                        b"yeto-trajectory-id-v1\0",
                        str(rollout_id).encode("ascii"),
                        str(group_index).encode("ascii"),
                        str(sample_index).encode("ascii"),
                        verified.task_id.encode("utf-8"),
                        verified.episode_id.encode("utf-8"),
                        bytes.fromhex(token_hash),
                        verified.mac.encode("ascii"),
                    )
                    envelopes.append(
                        TrajectoryEnvelope(
                            trajectory_id=trajectory_id,
                            task_id=verified.task_id,
                            prompt_group_id=f"r{rollout_id}:g{group_index}",
                            sample_index=sample_index,
                            behavior_policy_version=rollout_id,
                            behavior_policy_hash=behavior_policy_hash,
                            token_ids=tokens,
                            response_token_count=response_length,
                            behavior_logprobs_hash=behavior_logprobs_hash,
                            reward=verified.reward,
                            reward_contract_hash=reward_contract_hash,
                            cleanup_evidence_hash=cleanup_evidence_hash,
                        )
                    )
                    continue

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
                    verified.mac.encode("ascii"),
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
                trajectory_id = _hash_parts(
                    b"yeto-trajectory-id-v2\0", *identity_parts
                )
                envelopes.append(
                    TrajectoryEnvelope(
                        trajectory_id=trajectory_id,
                        task_id=verified.task_id,
                        prompt_group_id=f"r{rollout_id}:g{group_index}",
                        sample_index=sample_index,
                        behavior_policy_version=rollout_id,
                        behavior_policy_hash=behavior_policy_hash,
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
    )
    return TrajectoryBatchEvidence(
        rollout_id,
        behavior_policy_hash,
        input_batch_hash,
        trained_tokens,
        ordered,
        evidence_kind,
        schema_version,
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
    if evidence.schema_version == _SCHEMA:
        payload["evidence_kind"] = evidence.evidence_kind
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
    if schema_version == _SCHEMA:
        expected_fields.add("evidence_kind")
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
    )
    _validate_batch_evidence(evidence)
    return evidence


def _verified_outcome(
    metadata: Mapping[str, Any], evidence_kind: str
) -> _VerifiedOutcome:
    if evidence_kind == "terminal-bench-2.1":
        if any(
            key in metadata
            for key in (
                "secrlenv_trusted_outcome",
                "secrlenv_trusted_outcome_hmac",
            )
        ):
            raise ValueError("trajectory metadata mixes benchmark evidence kinds")
        from .tbench_outcome import MAC_KEY, canonical_outcome, verified_outcome

        outcome, reward = verified_outcome(dict(metadata))
        return _VerifiedOutcome(
            task_id=outcome["task_id"],
            episode_id=outcome["episode_id"],
            sample_id=outcome["sample_id"],
            reward=reward,
            mac=metadata[MAC_KEY],
            canonical=canonical_outcome(outcome),
        )

    if any(
        key in metadata
        for key in ("tbench_trusted_outcome", "tbench_trusted_outcome_hmac")
    ):
        raise ValueError("trajectory metadata mixes benchmark evidence kinds")
    from yeto_miles_secrlenv.reward import (
        CLEANUP_ERROR_STATUS,
        INFRASTRUCTURE_STATUS,
        MAC_KEY,
        _canonical,
    )
    from yeto_miles_secrlenv.reward import (
        _verified_outcome as verified_secrlenv_outcome,
    )

    outcome, reward = verified_secrlenv_outcome(dict(metadata))
    if outcome["status"] in {INFRASTRUCTURE_STATUS, CLEANUP_ERROR_STATUS}:
        raise ValueError("trajectory has no terminal grader/cleanup evidence")
    supplied_mac = metadata.get(MAC_KEY)
    if not isinstance(supplied_mac, str):
        raise TypeError("trajectory has no signed cleanup evidence")
    task_id = outcome.get("task_id")
    episode_id = outcome.get("episode_id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("trajectory has no signed task identity")
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("trajectory has no signed episode identity")
    return _VerifiedOutcome(
        task_id=task_id,
        episode_id=episode_id,
        sample_id=None,
        reward=reward,
        mac=supplied_mac,
        canonical=_canonical(outcome),
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
        raise ValueError("trajectory reward differs from signed evidence")


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
    domain = (
        b"yeto-tbench-verdict-evidence-v1\0"
        if evidence_kind == "terminal-bench-2.1"
        else b"yeto-secrlenv-cleanup-evidence-v1\0"
    )
    return _hash_parts(domain, verified.canonical, verified.mac.encode("ascii"))


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
        else _V2_ENVELOPE_FIELDS
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
    if evidence.schema_version not in {_LEGACY_SCHEMA, _SCHEMA}:
        raise ValueError("trajectory evidence schema is unsupported")
    if evidence.evidence_kind not in _EVIDENCE_KINDS:
        raise ValueError("trajectory evidence kind is unsupported")
    if (
        evidence.schema_version == _LEGACY_SCHEMA
        and evidence.evidence_kind != "secrlenv"
    ):
        raise ValueError("legacy trajectory evidence can only represent SecRLEnv")
    if evidence.schema_version == _SCHEMA and any(
        item.evidence_kind != evidence.evidence_kind
        or item.active_token_count is None
        or item.loss_mask_hash is None
        or item.active_token_ids_hash is None
        for item in evidence.envelopes
    ):
        raise ValueError("trajectory evidence v2 item is incomplete")
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
        )
        != evidence.input_batch_hash
    ):
        raise ValueError("trajectory evidence batch hash changed")


def _envelope_batch_hash(
    envelopes: tuple[TrajectoryEnvelope, ...],
    trained_tokens: int,
    *,
    schema_version: int = _LEGACY_SCHEMA,
    evidence_kind: str = "secrlenv",
) -> str:
    payload: dict[str, Any] = {
        "trained_tokens": trained_tokens,
        "envelopes": [
            _envelope_mapping(envelope, schema_version) for envelope in envelopes
        ],
    }
    domain = b"yeto-trajectory-batch-v1\0"
    if schema_version == _SCHEMA:
        payload.update({"schema": _SCHEMA, "evidence_kind": evidence_kind})
        domain = b"yeto-trajectory-batch-v2\0"
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
