"""Strict RL checkpoint and final-marker readers shared by tests and export."""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from pathlib import Path

RL_CHECKPOINT_MAGIC = 0xD17052A1
RL_CHECKPOINT_SCHEMA = 1
RL_FINAL_MARKER_MAGIC = "YETO_RL_FINAL_V1"


@dataclass(frozen=True)
class RlCheckpoint:
    run_manifest_sha256: str
    layout_fingerprint: str
    roster_size: int
    policy_sha256: str
    global_step: int
    versions: tuple[int, ...]
    fragments: tuple[tuple[float, ...], ...]
    ledger: dict[int, tuple[int, int, int]]


@dataclass(frozen=True)
class RlFinalMarker:
    global_step: int
    roster_size: int
    run_manifest_sha256: str
    layout_fingerprint: str
    policy_sha256: str


def parse_rl_checkpoint(path: str | Path) -> RlCheckpoint:
    data = Path(path).read_bytes()
    offset = 0

    def take(size: int) -> bytes:
        nonlocal offset
        end = offset + size
        if end > len(data):
            raise ValueError("truncated RL checkpoint")
        value = data[offset:end]
        offset = end
        return value

    def scalar(fmt: str):
        return struct.unpack(fmt, take(struct.calcsize(fmt)))[0]

    if scalar("<I") != RL_CHECKPOINT_MAGIC:
        raise ValueError("not an rl-strict-avg checkpoint")
    if scalar("<H") != RL_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported RL checkpoint schema")
    manifest = take(32).hex()
    layout = take(32).hex()
    roster = scalar("<I")
    stored_policy = take(32).hex()
    global_step = scalar("<Q")
    fragment_count = scalar("<I")
    versions: list[int] = []
    fragments: list[tuple[float, ...]] = []
    for _ in range(fragment_count):
        versions.append(scalar("<Q"))
        numel = scalar("<Q")
        values = struct.unpack(f"<{numel}f", take(numel * 4))
        if any(not math.isfinite(value) for value in values):
            raise ValueError("RL checkpoint policy contains NaN or Inf")
        fragments.append(values)
        momentum = struct.unpack(f"<{numel}f", take(numel * 4))
        if any(not math.isfinite(value) for value in momentum):
            raise ValueError("RL checkpoint outer momentum contains NaN or Inf")
    learner_count = scalar("<I")
    ledger: dict[int, tuple[int, int, int]] = {}
    for _ in range(learner_count):
        learner_id = scalar("<I")
        if learner_id in ledger:
            raise ValueError("duplicate learner in RL checkpoint ledger")
        ledger[learner_id] = (scalar("<Q"), scalar("<Q"), scalar("<Q"))
    if offset != len(data):
        raise ValueError("trailing bytes in RL checkpoint")

    digest = hashlib.sha256()
    digest.update(b"yeto-rl-policy-v1\0")
    digest.update(bytes.fromhex(layout))
    for values in fragments:
        digest.update(struct.pack(f"<{len(values)}f", *values))
    if digest.hexdigest() != stored_policy:
        raise ValueError("RL checkpoint policy SHA256 mismatch")
    return RlCheckpoint(
        manifest,
        layout,
        roster,
        stored_policy,
        global_step,
        tuple(versions),
        tuple(fragments),
        ledger,
    )


def parse_rl_final_marker(path: str | Path) -> RlFinalMarker:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != RL_FINAL_MARKER_MAGIC:
        raise ValueError("not an RL final marker")
    fields: dict[str, str] = {}
    for line in lines[1:]:
        key, separator, value = line.partition("=")
        if not separator or key in fields:
            raise ValueError("malformed RL final marker")
        fields[key] = value
    expected = {
        "global_step",
        "roster_size",
        "run_manifest_sha256",
        "layout_fingerprint",
        "policy_sha256",
    }
    if set(fields) != expected:
        raise ValueError("incomplete RL final marker")
    for key in ("run_manifest_sha256", "layout_fingerprint", "policy_sha256"):
        value = fields[key]
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"invalid {key} in RL final marker")
    return RlFinalMarker(
        int(fields["global_step"]),
        int(fields["roster_size"]),
        fields["run_manifest_sha256"],
        fields["layout_fingerprint"],
        fields["policy_sha256"],
    )


def validate_rl_final_checkpoint(
    checkpoint_path: str | Path, manifest_sha256: str
) -> RlCheckpoint:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = parse_rl_checkpoint(checkpoint_path)
    marker = parse_rl_final_marker(Path(f"{checkpoint_path}.final"))
    if (
        checkpoint.run_manifest_sha256 != manifest_sha256
        or marker.run_manifest_sha256 != manifest_sha256
        or marker.global_step != checkpoint.global_step
        or marker.roster_size != checkpoint.roster_size
        or marker.layout_fingerprint != checkpoint.layout_fingerprint
        or marker.policy_sha256 != checkpoint.policy_sha256
    ):
        raise ValueError("RL checkpoint/final-marker identity mismatch")
    expected_ledger = {
        learner_id: (
            checkpoint.global_step,
            checkpoint.global_step,
            checkpoint.global_step,
        )
        for learner_id in range(checkpoint.roster_size)
    }
    if checkpoint.versions != (checkpoint.global_step,) or checkpoint.ledger != expected_ledger:
        raise ValueError("RL checkpoint does not contain a complete fixed-roster ledger")
    return checkpoint
