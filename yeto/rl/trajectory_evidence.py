"""Closed, authenticated trajectory evidence for replayable RL updates."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .contracts import TrajectoryEnvelope

_SCHEMA = 1
_SHA256 = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class TrajectoryBatchEvidence:
    rollout_id: int
    behavior_policy_hash: str
    input_batch_hash: str
    trained_tokens: int
    envelopes: tuple[TrajectoryEnvelope, ...]

    @property
    def trajectory_ids(self) -> tuple[str, ...]:
        return tuple(envelope.trajectory_id for envelope in self.envelopes)


def build_trajectory_batch_evidence(
    groups: Sequence[Sequence[Any]],
    *,
    rollout_id: int,
    behavior_policy_hash: str,
    reward_contract_hash: str,
) -> TrajectoryBatchEvidence:
    """Authenticate accepted SecRLEnv samples and bind their GRPO batch.

    Only the signed SecRLEnv outcome and a closed allowlist of Miles sample
    fields enter the evidence.  Arbitrary agent/sample metadata is never
    serialized.
    """

    _nonnegative_int("rollout_id", rollout_id)
    _sha256("behavior_policy_hash", behavior_policy_hash)
    _sha256("reward_contract_hash", reward_contract_hash)
    if not groups or any(
        not isinstance(group, Sequence) or not group for group in groups
    ):
        raise ValueError("trajectory evidence requires non-empty GRPO groups")

    from yeto_miles_secrlenv.reward import (
        CLEANUP_ERROR_STATUS,
        INFRASTRUCTURE_STATUS,
        MAC_KEY,
        _canonical,
        _verified_outcome,
    )

    envelopes = []
    seen_indices = set()
    seen_groups = set()
    for group in groups:
        group_indices = set()
        signed_group_index = None
        for sample in group:
            group_index = getattr(sample, "group_index", None)
            sample_index = getattr(sample, "index", None)
            _nonnegative_int("sample.group_index", group_index)
            _nonnegative_int("sample.index", sample_index)
            if signed_group_index is None:
                if group_index in seen_groups:
                    raise ValueError("trajectory batch repeats a GRPO group")
                signed_group_index = group_index
                seen_groups.add(group_index)
            elif group_index != signed_group_index:
                raise ValueError("trajectory GRPO group mixes group identities")
            if sample_index in seen_indices or sample_index in group_indices:
                raise ValueError("trajectory batch contains duplicate sample indexes")
            group_indices.add(sample_index)
            seen_indices.add(sample_index)

            status = getattr(getattr(sample, "status", None), "value", None)
            if status not in {"completed", "truncated"}:
                raise ValueError("trajectory batch contains a non-trainable sample")
            metadata = getattr(sample, "metadata", None)
            outcome, reward = _verified_outcome(metadata)
            if outcome["status"] in {INFRASTRUCTURE_STATUS, CLEANUP_ERROR_STATUS}:
                raise ValueError("trajectory has no terminal grader/cleanup evidence")
            supplied_mac = (
                metadata.get(MAC_KEY) if isinstance(metadata, Mapping) else None
            )
            if not isinstance(supplied_mac, str):
                raise TypeError("trajectory has no signed cleanup evidence")
            sample_reward = getattr(sample, "reward", None)
            if (
                isinstance(sample_reward, bool)
                or not isinstance(sample_reward, (int, float))
                or not math.isfinite(float(sample_reward))
                or float(sample_reward) != reward
            ):
                raise ValueError("trajectory reward differs from signed evidence")

            tokens = tuple(getattr(sample, "tokens", ()))
            if not tokens or any(
                type(token) is not int or token < 0 for token in tokens
            ):
                raise ValueError("trajectory tokens are incomplete")
            response_length = getattr(sample, "response_length", None)
            if (
                type(response_length) is not int
                or response_length < 1
                or response_length > len(tokens)
            ):
                raise ValueError("trajectory response length is invalid")
            logprobs = getattr(sample, "rollout_log_probs", None)
            behavior_logprobs_hash = _logprobs_hash(logprobs, response_length)
            cleanup_evidence_hash = _hash_parts(
                b"yeto-secrlenv-cleanup-evidence-v1\0",
                _canonical(outcome),
                supplied_mac.encode("ascii"),
            )
            task_id = outcome.get("task_id")
            episode_id = outcome.get("episode_id")
            if not isinstance(task_id, str) or not task_id:
                raise ValueError("trajectory has no signed task identity")
            if not isinstance(episode_id, str) or not episode_id:
                raise ValueError("trajectory has no signed episode identity")
            token_hash = _hash_parts(
                b"yeto-trajectory-tokens-v1\0",
                struct.pack(f"<{len(tokens)}Q", *tokens),
            )
            trajectory_id = _hash_parts(
                b"yeto-trajectory-id-v1\0",
                str(rollout_id).encode("ascii"),
                str(group_index).encode("ascii"),
                str(sample_index).encode("ascii"),
                task_id.encode("utf-8"),
                episode_id.encode("utf-8"),
                bytes.fromhex(token_hash),
                supplied_mac.encode("ascii"),
            )
            envelopes.append(
                TrajectoryEnvelope(
                    trajectory_id=trajectory_id,
                    task_id=task_id,
                    prompt_group_id=f"r{rollout_id}:g{group_index}",
                    sample_index=sample_index,
                    behavior_policy_version=rollout_id,
                    behavior_policy_hash=behavior_policy_hash,
                    token_ids=tokens,
                    response_token_count=response_length,
                    behavior_logprobs_hash=behavior_logprobs_hash,
                    reward=reward,
                    reward_contract_hash=reward_contract_hash,
                    cleanup_evidence_hash=cleanup_evidence_hash,
                )
            )

    ordered = tuple(envelopes)
    if tuple(item.sample_index for item in ordered) != tuple(
        sorted(item.sample_index for item in ordered)
    ):
        raise ValueError("trajectory samples are not in canonical index order")
    trained_tokens = _trained_token_count(ordered)
    input_batch_hash = _envelope_batch_hash(ordered, trained_tokens)
    return TrajectoryBatchEvidence(
        rollout_id,
        behavior_policy_hash,
        input_batch_hash,
        trained_tokens,
        ordered,
    )


def write_trajectory_batch_evidence(
    directory: str | Path,
    evidence: TrajectoryBatchEvidence,
) -> Path:
    """Publish one immutable private replay artifact with exclusive creation."""

    root = Path(directory).expanduser()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("trajectory evidence root is not a real directory")
    if root.stat().st_mode & 0o077:
        raise RuntimeError("trajectory evidence root is not private")
    path = trajectory_batch_evidence_path(root, evidence.rollout_id)
    payload = {
        "schema": _SCHEMA,
        "rollout_id": evidence.rollout_id,
        "behavior_policy_hash": evidence.behavior_policy_hash,
        "input_batch_hash": evidence.input_batch_hash,
        "trained_tokens": evidence.trained_tokens,
        "envelopes": [asdict(envelope) for envelope in evidence.envelopes],
    }
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
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "rollout_id",
        "behavior_policy_hash",
        "input_batch_hash",
        "trained_tokens",
        "envelopes",
    }:
        raise ValueError("trajectory evidence envelope is malformed")
    if value["schema"] != _SCHEMA or not isinstance(value["envelopes"], list):
        raise ValueError("trajectory evidence schema is unsupported")
    envelopes = tuple(
        TrajectoryEnvelope(
            **{
                **item,
                "token_ids": tuple(item["token_ids"]),
            }
        )
        for item in value["envelopes"]
    )
    evidence = TrajectoryBatchEvidence(
        value["rollout_id"],
        value["behavior_policy_hash"],
        value["input_batch_hash"],
        value["trained_tokens"],
        envelopes,
    )
    _nonnegative_int("rollout_id", evidence.rollout_id)
    _sha256("behavior_policy_hash", evidence.behavior_policy_hash)
    _sha256("input_batch_hash", evidence.input_batch_hash)
    _nonnegative_int("trained_tokens", evidence.trained_tokens)
    if evidence.trained_tokens < 1:
        raise ValueError("trajectory evidence trained-token count is invalid")
    if _trained_token_count(envelopes) != evidence.trained_tokens:
        raise ValueError("trajectory evidence trained-token aggregate changed")
    if (
        _envelope_batch_hash(envelopes, evidence.trained_tokens)
        != evidence.input_batch_hash
    ):
        raise ValueError("trajectory evidence batch hash changed")
    return evidence


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


def _trained_token_count(envelopes: tuple[TrajectoryEnvelope, ...]) -> int:
    return sum(envelope.response_token_count for envelope in envelopes)


def _envelope_batch_hash(
    envelopes: tuple[TrajectoryEnvelope, ...],
    trained_tokens: int,
) -> str:
    payload = json.dumps(
        {
            "trained_tokens": trained_tokens,
            "envelopes": [asdict(envelope) for envelope in envelopes],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return _hash_parts(b"yeto-trajectory-batch-v1\0", payload)


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
