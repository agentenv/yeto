#!/usr/bin/env python3
"""Fail-closed matched capture-OFF/capture-ON production parity gate.

The gate compares artifacts produced by two otherwise matched
``scripts/compare_diloco.py`` runs.  It deliberately does not infer equality
from filenames or summary statistics: every fully sampled syncer-probe anchor,
candidate, and applied update is hashed byte-for-byte, as are the authoritative
final checkpoint and exported tensor payloads.

Only two forms of nondeterministic metadata are canonicalized:

* the three known path fields in ``syncer_probe/index.jsonl`` are replaced by
  role tokens after their referenced files have been validated and hashed; and
* JSON export metadata values under an explicit path-key allowlist may replace
  the corresponding OFF/ON arm-directory prefix with ``<ARM_DIR>``.

Everything else must match exactly.  The result is always written as canonical
JSON plus a SHA-256 sidecar, including on failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

SCHEMA = "optimizer_capture_parity_v1"
PROBE_SCHEMA = "syncer_probe_capture_v1"
CKPT_MAGIC = 0xD1705A7E
PATH_FIELDS = ("state_checkpoint", "candidate_f32", "applied_update_f32")
PATH_ROLE = {
    "state_checkpoint": "<STATE_CHECKPOINT>",
    "candidate_f32": "<CANDIDATE_F32>",
    "applied_update_f32": "<APPLIED_UPDATE_F32>",
}
EXPORT_PATH_KEYS = {
    "_name_or_path",
    "base_model_name_or_path",
    "model_name_or_path",
    "name_or_path",
    "output_dir",
}
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
ATTEMPT_DISPOSITIONS = {
    "admitted_pending",
    "rejected_duplicate",
    "rejected_invalid",
    "rejected_stale_session",
    "late_no_inflight",
    "pruned_connection",
}


class ParityError(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckpointSummary:
    global_step: int
    fragments: tuple[tuple[int, int], ...]
    ledger: tuple[tuple[int, int, int, int], ...]
    layout_meta_sha256: str | None


def _reject_constant(value: str) -> None:
    raise ParityError(f"non-finite JSON number {value!r}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ParityError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def strict_json_loads(raw: str, *, source: Path | str) -> Any:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeError, ParityError) as exc:
        raise ParityError(f"{source}: malformed JSON: {exc}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise ParityError(f"missing regular JSONL file: {path}")
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ParityError(f"cannot read {path}: {exc}") from exc
    if not lines:
        raise ParityError(f"{path}: empty JSONL")
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise ParityError(f"{path}:{line_number}: blank JSONL record")
        row = strict_json_loads(line, source=f"{path}:{line_number}")
        if not isinstance(row, dict):
            raise ParityError(f"{path}:{line_number}: record is not an object")
        rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_f32_payload(path: Path, *, label: str) -> tuple[str, int]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ParityError(f"cannot read {label} {path}: {exc}") from exc
    if not raw or len(raw) % 4:
        raise ParityError(f"{label} has invalid nonempty f32 length: {path}")
    for index, (value,) in enumerate(struct.iter_unpack("<f", raw)):
        if not math.isfinite(value):
            raise ParityError(
                f"{label} contains non-finite f32 at index {index}: {path}"
            )
    return hashlib.sha256(raw).hexdigest(), len(raw) // 4


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def parse_checkpoint_summary(path: Path) -> CheckpointSummary:
    """Validate the syncer checkpoint format without importing ML runtimes."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ParityError(f"cannot read checkpoint {path}: {exc}") from exc
    offset = 0

    def take(size: int, label: str) -> bytes:
        nonlocal offset
        if size < 0 or offset + size > len(data):
            raise ParityError(
                f"{path}: truncated checkpoint at {label}; "
                f"need {size} bytes at offset {offset}, have {len(data) - offset}"
            )
        result = data[offset : offset + size]
        offset += size
        return result

    (magic,) = struct.unpack("<I", take(4, "magic"))
    if magic != CKPT_MAGIC:
        raise ParityError(
            f"{path}: bad checkpoint magic 0x{magic:08x}; expected 0x{CKPT_MAGIC:08x}"
        )
    (global_step,) = struct.unpack("<Q", take(8, "global_step"))
    (fragment_count,) = struct.unpack("<I", take(4, "fragment_count"))
    fragments: list[tuple[int, int]] = []
    for fragment in range(fragment_count):
        version, numel = struct.unpack("<QQ", take(16, f"fragment {fragment} header"))
        # Both params and outer momentum are little-endian f32 arrays.  Reject
        # NaN/Inf even when OFF and ON happen to contain identical corruption.
        for tensor_name in ("params", "momentum"):
            tensor_raw = take(4 * numel, f"fragment {fragment} {tensor_name}")
            for index, (value,) in enumerate(struct.iter_unpack("<f", tensor_raw)):
                if not math.isfinite(value):
                    raise ParityError(
                        f"{path}: fragment {fragment} {tensor_name} contains "
                        f"non-finite f32 at index {index}"
                    )
        fragments.append((version, numel))
    (ledger_count,) = struct.unpack("<I", take(4, "ledger_count"))
    ledger: list[tuple[int, int, int, int]] = []
    learner_ids: set[int] = set()
    for index in range(ledger_count):
        entry = struct.unpack("<IQQQ", take(28, f"ledger entry {index}"))
        learner_id = entry[0]
        if learner_id in learner_ids:
            raise ParityError(f"{path}: duplicate ledger learner {learner_id}")
        learner_ids.add(learner_id)
        ledger.append(entry)

    layout_meta_sha256: str | None = None
    if offset != len(data):
        (metadata_size,) = struct.unpack("<I", take(4, "layout_meta_len"))
        metadata_raw = take(metadata_size, "layout_meta")
        if offset != len(data):
            raise ParityError(f"{path}: trailing bytes after layout metadata")
        if metadata_raw:
            try:
                metadata_text = metadata_raw.decode("utf-8")
            except UnicodeError as exc:
                raise ParityError(
                    f"{path}: layout metadata is not UTF-8: {exc}"
                ) from exc
            metadata = strict_json_loads(metadata_text, source=f"{path}:layout_meta")
            layout_meta_sha256 = canonical_sha256(metadata)

    return CheckpointSummary(
        global_step=global_step,
        fragments=tuple(fragments),
        ledger=tuple(ledger),
        layout_meta_sha256=layout_meta_sha256,
    )


def require_int(row: dict[str, Any], field: str, *, minimum: int = 0) -> int:
    value = row.get(field)
    if type(value) is not int or value < minimum:
        raise ParityError(f"field {field!r} must be an integer >= {minimum}")
    return value


def require_finite_number(row: dict[str, Any], field: str) -> float:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParityError(f"field {field!r} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ParityError(f"field {field!r} must be finite")
    return result


def safe_relative_file(root: Path, value: Any, field: str) -> tuple[Path, str]:
    if not isinstance(value, str) or not value:
        raise ParityError(f"field {field!r} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise ParityError(f"field {field!r} must not be absolute: {value!r}")
    root_real = root.resolve()
    path = root / relative
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ParityError(f"{field} references missing file {path}: {exc}") from exc
    try:
        resolved.relative_to(root_real)
    except ValueError as exc:
        raise ParityError(f"{field} escapes capture root: {value!r}") from exc
    if path.is_symlink() or not resolved.is_file():
        raise ParityError(f"{field} is not a regular non-symlink file: {path}")
    return resolved, relative.as_posix()


@dataclass
class ProbeCapture:
    root: Path
    canonical_rows: dict[tuple[int, int, int], dict[str, Any]]
    candidate_digests: dict[tuple[int, int, int], str]
    state_digests: dict[tuple[int, int], str]
    update_digests: dict[tuple[int, int], str]
    referenced_manifest: dict[str, str]

    @property
    def group_keys(self) -> set[tuple[int, int]]:
        return set(self.state_digests)


def load_probe_capture(root: Path) -> ProbeCapture:
    root = root.resolve()
    rows = read_jsonl(root / "index.jsonl")
    canonical_rows: dict[tuple[int, int, int], dict[str, Any]] = {}
    candidate_digests: dict[tuple[int, int, int], str] = {}
    state_digests: dict[tuple[int, int], str] = {}
    update_digests: dict[tuple[int, int], str] = {}
    referenced_manifest: dict[str, str] = {}
    referenced_by_dir: dict[str, set[str]] = {
        "states": set(),
        "candidates": set(),
        "applied_updates": set(),
    }
    group_paths: dict[tuple[int, int], tuple[str, str]] = {}
    group_metadata: dict[tuple[int, int], tuple[int, int, int]] = {}
    candidate_numel: dict[tuple[int, int, int], int] = {}
    update_numel: dict[tuple[int, int], int] = {}

    for row_number, row in enumerate(rows, 1):
        if row.get("schema") != PROBE_SCHEMA:
            raise ParityError(
                f"{root / 'index.jsonl'}:{row_number}: expected schema {PROBE_SCHEMA!r}"
            )
        step = require_int(row, "step", minimum=1)
        version = require_int(row, "version", minimum=1)
        if version != step:
            raise ParityError(f"probe version {version} does not equal step {step}")
        syncer_global_step = require_int(row, "syncer_global_step")
        fragment = require_int(row, "fragment")
        current_fragment_version = require_int(row, "current_fragment_version")
        learner = require_int(row, "learner_id")
        key = (step, fragment, learner)
        group = (step, fragment)
        if key in canonical_rows:
            raise ParityError(f"duplicate probe row for {key}")
        require_int(row, "base_version")
        require_int(row, "local_step")
        c_steps = require_int(row, "c_steps", minimum=1)
        c_tokens = require_int(row, "c_tokens", minimum=1)
        weight = require_finite_number(row, "weight")
        if weight <= 0:
            raise ParityError("field 'weight' must be positive")
        expected_weight = float(c_tokens) * float(c_tokens) / float(c_steps)
        if weight != expected_weight:
            raise ParityError(
                f"field 'weight' {weight} does not equal c_tokens^2/c_steps "
                f"({expected_weight})"
            )

        resolved: dict[str, tuple[Path, str]] = {}
        for field in PATH_FIELDS:
            resolved[field] = safe_relative_file(root, row.get(field), field)
        state_path, state_rel = resolved["state_checkpoint"]
        candidate_path, candidate_rel = resolved["candidate_f32"]
        update_path, update_rel = resolved["applied_update_f32"]
        if not state_rel.startswith("states/"):
            raise ParityError(f"state_checkpoint has unexpected location: {state_rel}")
        if not candidate_rel.startswith("candidates/"):
            raise ParityError(f"candidate_f32 has unexpected location: {candidate_rel}")
        if not update_rel.startswith("applied_updates/"):
            raise ParityError(
                f"applied_update_f32 has unexpected location: {update_rel}"
            )

        expected_group_paths = group_paths.setdefault(group, (state_rel, update_rel))
        if expected_group_paths != (state_rel, update_rel):
            raise ParityError(
                f"probe group {group} does not share one state/update payload"
            )
        metadata = (syncer_global_step, current_fragment_version, version)
        expected_metadata = group_metadata.setdefault(group, metadata)
        if expected_metadata != metadata:
            raise ParityError(f"probe group {group} has inconsistent round metadata")

        state_digest = sha256_file(state_path)
        candidate_digest, candidate_count = validate_f32_payload(
            candidate_path, label="candidate payload"
        )
        update_digest, update_count = validate_f32_payload(
            update_path, label="applied-update payload"
        )
        if group in state_digests and state_digests[group] != state_digest:
            raise ParityError(f"probe group {group} has inconsistent state bytes")
        if group in update_digests and update_digests[group] != update_digest:
            raise ParityError(f"probe group {group} has inconsistent update bytes")
        state_digests[group] = state_digest
        update_digests[group] = update_digest
        candidate_digests[key] = candidate_digest
        candidate_numel[key] = candidate_count
        update_numel[group] = update_count
        referenced_manifest[f"state:{step}:{fragment}"] = state_digest
        referenced_manifest[f"candidate:{step}:{fragment}:{learner}"] = candidate_digest
        referenced_manifest[f"update:{step}:{fragment}"] = update_digest
        referenced_by_dir["states"].add(state_rel)
        referenced_by_dir["candidates"].add(candidate_rel)
        referenced_by_dir["applied_updates"].add(update_rel)

        canonical = dict(row)
        for field in PATH_FIELDS:
            canonical[field] = PATH_ROLE[field]
        canonical_rows[key] = canonical

    if not canonical_rows:
        raise ParityError("probe index contains no candidate rows")

    # Validate every unique checkpoint structurally, even if OFF and ON happen
    # to contain the same malformed bytes.
    checkpoint_summaries: dict[str, CheckpointSummary] = {}
    for state_rel, _update_rel in set(group_paths.values()):
        try:
            checkpoint_summaries[state_rel] = parse_checkpoint_summary(root / state_rel)
        except Exception as exc:
            raise ParityError(
                f"invalid pre-merge checkpoint {state_rel}: {exc}"
            ) from exc
    for group, (state_rel, _update_rel) in group_paths.items():
        step, fragment = group
        checkpoint = checkpoint_summaries[state_rel]
        if fragment >= len(checkpoint.fragments):
            raise ParityError(
                f"probe group {group} fragment is absent from pre-merge checkpoint"
            )
        syncer_step, current_version, _version = group_metadata[group]
        if checkpoint.global_step != syncer_step:
            raise ParityError(
                f"probe group {group} syncer_global_step does not match checkpoint"
            )
        checkpoint_version, expected_numel = checkpoint.fragments[fragment]
        if checkpoint_version != current_version:
            raise ParityError(
                f"probe group {group} current fragment version does not match checkpoint"
            )
        if update_numel[group] != expected_numel:
            raise ParityError(
                f"probe group {group} update length {update_numel[group]} "
                f"does not match checkpoint numel {expected_numel}"
            )
        for key, count in candidate_numel.items():
            if key[:2] == (step, fragment) and count != expected_numel:
                raise ParityError(
                    f"probe candidate {key} length {count} does not match "
                    f"checkpoint numel {expected_numel}"
                )

    # A fully sampled parity capture has no unindexed payloads. Unexpected temp
    # or stale files are evidence of an incomplete/mixed run and fail closed.
    for directory, expected in referenced_by_dir.items():
        base = root / directory
        if not base.is_dir() or base.is_symlink():
            raise ParityError(f"missing regular probe payload directory: {base}")
        actual: set[str] = set()
        for path in base.rglob("*"):
            if path.is_symlink():
                raise ParityError(f"symlink in probe payload directory: {path}")
            if path.is_file():
                actual.add(path.relative_to(root).as_posix())
            elif not path.is_dir():
                raise ParityError(f"unsupported probe payload filesystem entry: {path}")
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ParityError(
                f"probe {directory} payload set mismatch; missing={missing}, extra={extra}"
            )

    return ProbeCapture(
        root=root,
        canonical_rows=canonical_rows,
        candidate_digests=candidate_digests,
        state_digests=state_digests,
        update_digests=update_digests,
        referenced_manifest=referenced_manifest,
    )


def compare_probe_captures(
    off_root: Path, on_root: Path
) -> tuple[dict[str, Any], ProbeCapture]:
    off = load_probe_capture(off_root)
    on = load_probe_capture(on_root)
    if set(off.canonical_rows) != set(on.canonical_rows):
        raise ParityError(
            "probe key set differs; "
            f"missing_on={sorted(set(off.canonical_rows) - set(on.canonical_rows))}, "
            f"extra_on={sorted(set(on.canonical_rows) - set(off.canonical_rows))}"
        )
    for key in sorted(off.canonical_rows):
        if off.canonical_rows[key] != on.canonical_rows[key]:
            raise ParityError(f"probe index metadata differs at {key}")
    if off.referenced_manifest != on.referenced_manifest:
        differences = [
            key
            for key in sorted(
                set(off.referenced_manifest) | set(on.referenced_manifest)
            )
            if off.referenced_manifest.get(key) != on.referenced_manifest.get(key)
        ]
        raise ParityError(f"probe payload bytes differ at {differences[:12]}")
    return (
        {
            "candidate_rows": len(off.candidate_digests),
            "commit_groups": len(off.group_keys),
            "payload_manifest_sha256": canonical_sha256(off.referenced_manifest),
            "canonical_index_sha256": canonical_sha256(
                [off.canonical_rows[key] for key in sorted(off.canonical_rows)]
            ),
            "canonicalized_fields": list(PATH_FIELDS),
        },
        on,
    )


def validate_on_transcript(path: Path, probe: ProbeCapture) -> dict[str, Any]:
    rows = read_jsonl(path)
    events: dict[int, dict[str, Any]] = {}
    session: str | None = None
    for row_number, row in enumerate(rows, 1):
        event_seq = require_int(row, "event_seq", minimum=1)
        if event_seq != row_number:
            raise ParityError(
                f"{path}:{row_number}: event_seq {event_seq} is not contiguous"
            )
        if event_seq in events:
            raise ParityError(f"duplicate transcript event_seq {event_seq}")
        events[event_seq] = row
        current_session = row.get("capture_session_uuid")
        if not isinstance(current_session, str) or not current_session:
            raise ParityError(f"transcript event {event_seq} lacks capture session")
        if session is None:
            session = current_session
        elif current_session != session:
            raise ParityError("transcript capture session changes within one file")

    if rows[0].get("schema") != "syncer_response_transcript_header_v1":
        raise ParityError("transcript must begin with its v1 header")
    attempts: dict[int, dict[str, Any]] = {}
    commits: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows[1:]:
        seq = int(row["event_seq"])
        schema = row.get("schema")
        if schema == "syncer_push_attempt_v1":
            disposition = row.get("disposition")
            if disposition not in ATTEMPT_DISPOSITIONS:
                raise ParityError(f"transcript event {seq} has bad disposition")
            digest = row.get("received_payload_sha256")
            if digest is not None and (
                not isinstance(digest, str) or not DIGEST_RE.fullmatch(digest)
            ):
                raise ParityError(
                    f"transcript event {seq} has malformed payload digest"
                )
            attempts[seq] = row
        elif schema == "syncer_round_commit_v1":
            step = require_int(row, "request_global_step", minimum=1)
            fragment = require_int(row, "fragment_id")
            key = (step, fragment)
            if key in commits:
                raise ParityError(f"duplicate transcript commit {key}")
            if require_int(row, "wire_dtype", minimum=1) != 1:
                raise ParityError(
                    f"commit {key}: exact payload/candidate parity requires f32 wire dtype 1"
                )
            responders = row.get("responders")
            if not isinstance(responders, list) or not responders:
                raise ParityError(f"commit {key} has no responder list")
            learner_ids: list[int] = []
            for expected_index, responder in enumerate(responders):
                if not isinstance(responder, dict):
                    raise ParityError(f"commit {key} responder is not an object")
                if require_int(responder, "responder_index") != expected_index:
                    raise ParityError(
                        f"commit {key} responder indices are not contiguous"
                    )
                learner = require_int(responder, "learner_id")
                learner_ids.append(learner)
                source = require_int(responder, "source_attempt_event_seq", minimum=1)
                attempt = attempts.get(source)
                if attempt is None or attempt.get("disposition") != "admitted_pending":
                    raise ParityError(
                        f"commit {key} learner {learner} does not source an admitted attempt"
                    )
                for field, expected in (
                    ("request_global_step", step),
                    ("fragment_id", fragment),
                    ("learner_id", learner),
                ):
                    if attempt.get(field) != expected:
                        raise ParityError(
                            f"commit {key} learner {learner} source mismatch in {field}"
                        )
                candidate_row = probe.canonical_rows.get((step, fragment, learner))
                if candidate_row is None:
                    raise ParityError(
                        f"commit {key} learner {learner} has no probe candidate"
                    )
                if require_int(attempt, "wire_dtype", minimum=1) != 1:
                    raise ParityError(
                        f"commit {key} learner {learner} attempt is not f32 wire"
                    )
                for field in ("base_version", "local_step", "c_steps", "c_tokens"):
                    expected = require_int(candidate_row, field)
                    if require_int(attempt, field) != expected:
                        raise ParityError(
                            f"commit {key} learner {learner} attempt/probe {field} mismatch"
                        )
                    if require_int(responder, field) != expected:
                        raise ParityError(
                            f"commit {key} learner {learner} responder/probe {field} mismatch"
                        )
                expected_weight = require_finite_number(candidate_row, "weight")
                if require_finite_number(attempt, "weight") != expected_weight:
                    raise ParityError(
                        f"commit {key} learner {learner} attempt/probe weight mismatch"
                    )
                if require_finite_number(responder, "weight") != expected_weight:
                    raise ParityError(
                        f"commit {key} learner {learner} responder/probe weight mismatch"
                    )
                window = responder.get("window_uuid")
                if not isinstance(window, str) or not UUID_RE.fullmatch(window):
                    raise ParityError(
                        f"commit {key} learner {learner} has malformed UUID"
                    )
                if attempt.get("window_uuid") != window:
                    raise ParityError(f"commit {key} learner {learner} UUID mismatch")
                digest = responder.get("received_payload_sha256")
                if not isinstance(digest, str) or not DIGEST_RE.fullmatch(digest):
                    raise ParityError(
                        f"commit {key} learner {learner} has malformed digest"
                    )
                if attempt.get("received_payload_sha256") != digest:
                    raise ParityError(
                        f"commit {key} learner {learner} attempt digest mismatch"
                    )
                if attempt.get("payload_digest_match") is not True:
                    raise ParityError(
                        f"commit {key} learner {learner} wire digest did not match"
                    )
                candidate_digest = probe.candidate_digests.get(
                    (step, fragment, learner)
                )
                if candidate_digest != digest:
                    raise ParityError(
                        f"commit {key} learner {learner} wire/candidate bytes differ"
                    )
            if learner_ids != sorted(learner_ids) or len(set(learner_ids)) != len(
                learner_ids
            ):
                raise ParityError(
                    f"commit {key} responders are not unique numeric order"
                )
            expected_learners = sorted(
                candidate_key[2]
                for candidate_key in probe.candidate_digests
                if candidate_key[:2] == key
            )
            if learner_ids != expected_learners:
                raise ParityError(
                    f"commit {key} responder/probe learner sets differ; "
                    f"expected={expected_learners}, actual={learner_ids}"
                )
            if row.get("responder_count") != len(responders):
                raise ParityError(f"commit {key} responder_count is inconsistent")
            commits[key] = row
        else:
            raise ParityError(f"unexpected transcript schema {schema!r} at event {seq}")

    if set(commits) != probe.group_keys:
        raise ParityError(
            "transcript/probe commit groups differ; "
            f"missing={sorted(probe.group_keys - set(commits))}, "
            f"extra={sorted(set(commits) - probe.group_keys)}"
        )
    return {
        "events": len(rows),
        "attempts": len(attempts),
        "commits": len(commits),
        "capture_session_uuid": session,
        "transcript_sha256": sha256_file(path),
    }


def _canonicalize_export_json(value: Any, arm_dir: Path, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {
            item_key: _canonicalize_export_json(item, arm_dir, item_key)
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [_canonicalize_export_json(item, arm_dir, key) for item in value]
    if key in EXPORT_PATH_KEYS and isinstance(value, str):
        prefix = str(arm_dir.resolve())
        if value == prefix:
            return "<ARM_DIR>"
        if value.startswith(prefix + os.sep):
            return "<ARM_DIR>/" + value[len(prefix + os.sep) :].replace(os.sep, "/")
    return value


def load_export_manifest(export_dir: Path, arm_dir: Path) -> dict[str, dict[str, str]]:
    if not export_dir.is_dir() or export_dir.is_symlink():
        raise ParityError(f"missing regular export directory: {export_dir}")
    manifest: dict[str, dict[str, str]] = {}
    for path in sorted(export_dir.rglob("*")):
        if path.is_symlink():
            raise ParityError(f"symlink in export directory: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ParityError(f"unsupported export filesystem entry: {path}")
        relative = path.relative_to(export_dir).as_posix()
        if path.suffix.lower() == ".json":
            try:
                raw = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise ParityError(f"cannot read export JSON {path}: {exc}") from exc
            value = strict_json_loads(raw, source=path)
            canonical = _canonicalize_export_json(value, arm_dir)
            manifest[relative] = {
                "mode": "canonical_json",
                "sha256": canonical_sha256(canonical),
            }
        else:
            manifest[relative] = {"mode": "exact_bytes", "sha256": sha256_file(path)}
    if not manifest:
        raise ParityError(f"export directory is empty: {export_dir}")
    return manifest


def compare_export_trees(off_arm: Path, on_arm: Path) -> dict[str, Any]:
    off = load_export_manifest(off_arm / "export", off_arm)
    on = load_export_manifest(on_arm / "export", on_arm)
    if off != on:
        differences = [
            path for path in sorted(set(off) | set(on)) if off.get(path) != on.get(path)
        ]
        raise ParityError(f"export payload trees differ at {differences[:12]}")
    return {
        "files": len(off),
        "manifest_sha256": canonical_sha256(off),
        "canonical_json_path_keys": sorted(EXPORT_PATH_KEYS),
    }


def compare_final_checkpoints(
    off_arm: Path, on_arm: Path, extra_relative_files: list[str]
) -> dict[str, Any]:
    relative_files = ["state.ckpt", *extra_relative_files]
    if len(relative_files) != len(set(relative_files)):
        raise ParityError("duplicate final-file request")
    manifest: dict[str, str] = {}
    checkpoint_summary: dict[str, Any] | None = None
    for relative_text in relative_files:
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise ParityError(f"unsafe final-file relative path: {relative_text!r}")
        off = off_arm / relative
        on = on_arm / relative
        if not off.is_file() or off.is_symlink() or not on.is_file() or on.is_symlink():
            raise ParityError(f"missing regular matched final file: {relative_text}")
        off_digest = sha256_file(off)
        on_digest = sha256_file(on)
        if off_digest != on_digest:
            raise ParityError(f"final file bytes differ: {relative_text}")
        manifest[relative.as_posix()] = off_digest
        if relative.as_posix() == "state.ckpt":
            try:
                off_ckpt = parse_checkpoint_summary(off)
                on_ckpt = parse_checkpoint_summary(on)
            except Exception as exc:
                raise ParityError(f"invalid final syncer checkpoint: {exc}") from exc
            checkpoint_summary = {
                "global_step": off_ckpt.global_step,
                "fragments": len(off_ckpt.fragments),
                "learners_in_ledger": len(off_ckpt.ledger),
            }
            if off_ckpt != on_ckpt:
                raise ParityError(
                    "final checkpoint semantics differ despite digest comparison"
                )
    return {
        "files": manifest,
        "manifest_sha256": canonical_sha256(manifest),
        "checkpoint": checkpoint_summary,
    }


def load_results(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        arm = row.get("arm")
        if not isinstance(arm, str) or not arm:
            raise ParityError(f"{path}: result row has invalid arm")
        if arm in result:
            raise ParityError(f"{path}: duplicate result arm {arm!r}")
        require_finite_number(row, "eval_loss")
        require_finite_number(row, "wall_s")
        result[arm] = row
    return result


def compare_results(
    off_path: Path,
    on_path: Path,
    off_arm: str,
    on_arm: str,
    overhead_limit: float,
    off_interval_ns: int,
    on_interval_ns: int,
) -> dict[str, Any]:
    off = load_results(off_path)
    on = load_results(on_path)
    if off_arm not in off:
        raise ParityError(f"OFF arm {off_arm!r} is absent from {off_path}")
    if on_arm not in on:
        raise ParityError(f"ON arm {on_arm!r} is absent from {on_path}")
    expected_wall_scope = "syncer_commit_1_to_commit_N"
    if off[off_arm].get("wall_scope") != expected_wall_scope:
        raise ParityError(
            f"OFF result must use steady-state wall scope {expected_wall_scope!r}"
        )
    if on[on_arm].get("wall_scope") != expected_wall_scope:
        raise ParityError(
            f"ON result must use steady-state wall scope {expected_wall_scope!r}"
        )
    off_row = dict(off[off_arm])
    on_row = dict(on[on_arm])
    off_row.pop("arm", None)
    on_row.pop("arm", None)
    off_row.pop("wall_s", None)
    on_row.pop("wall_s", None)
    if off_row != on_row:
        raise ParityError(
            f"result semantics/eval loss differ for OFF {off_arm!r} and ON {on_arm!r}"
        )

    off_wall = require_finite_number(off[off_arm], "wall_s")
    on_wall = require_finite_number(on[on_arm], "wall_s")
    if off_interval_ns <= 0 or on_interval_ns <= 0:
        raise ParityError("producer timing interval must be positive")
    off_exact = off_interval_ns / 1_000_000_000.0
    on_exact = on_interval_ns / 1_000_000_000.0
    if off_wall != round(off_exact, 1) or on_wall != round(on_exact, 1):
        raise ParityError(
            "results wall_s does not match the sealed producer timing interval"
        )
    point_overhead = on_interval_ns / off_interval_ns - 1.0
    if point_overhead > overhead_limit:
        raise ParityError(
            f"capture overhead fails: exact={point_overhead:.8f}, "
            f"limit={overhead_limit:.8f}"
        )
    return {
        "off_arm": off_arm,
        "on_arm": on_arm,
        "eval_loss": off[off_arm]["eval_loss"],
        "off_wall_s_rounded": off_wall,
        "on_wall_s_rounded": on_wall,
        "off_interval_ns": off_interval_ns,
        "on_interval_ns": on_interval_ns,
        "point_overhead_fraction": point_overhead,
        "limit_fraction": overhead_limit,
        "wall_time_rounding_s": 0.1,
        "wall_scope": expected_wall_scope,
        "off_result_rows": len(off),
        "on_result_rows": len(on),
    }


def load_commit_interval(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    if len(rows) < 2:
        raise ParityError(f"{path}: timing tape requires at least two commits")
    elapsed: list[int] = []
    identities: list[tuple[int, int, int]] = []
    semantic_identities: set[tuple[int, int]] = set()
    for expected_seq, row in enumerate(rows, 1):
        commit_seq = row.get("commit_seq")
        commit_elapsed_ns = row.get("commit_elapsed_ns")
        if type(commit_seq) is not int or commit_seq != expected_seq:
            raise ParityError(
                f"{path}: non-contiguous commit_seq at row {expected_seq}: "
                f"{commit_seq!r}"
            )
        if type(commit_elapsed_ns) is not int or commit_elapsed_ns < 0:
            raise ParityError(
                f"{path}: invalid commit_elapsed_ns at row {expected_seq}"
            )
        if elapsed and commit_elapsed_ns <= elapsed[-1]:
            raise ParityError(f"{path}: commit_elapsed_ns is not strictly increasing")
        step = require_int(row, "step", minimum=1)
        fragment = require_int(row, "fragment")
        semantic = (step, fragment)
        if semantic in semantic_identities:
            raise ParityError(f"{path}: duplicate timing commit identity {semantic}")
        semantic_identities.add(semantic)
        identities.append((commit_seq, step, fragment))
        elapsed.append(commit_elapsed_ns)
    interval_ns = elapsed[-1] - elapsed[0]
    if interval_ns <= 0:
        raise ParityError(f"{path}: producer timing interval is empty")
    return {
        "commits": len(rows),
        "first_excluded_commit_seq": 1,
        "first_included_commit_seq": 2,
        "final_included_commit_seq": len(rows),
        "first_commit_elapsed_ns": elapsed[0],
        "final_commit_elapsed_ns": elapsed[-1],
        "interval_ns": interval_ns,
        "ordered_commit_identities": [list(identity) for identity in identities],
        "tape_sha256": sha256_file(path),
    }


def write_evidence(path: Path, evidence: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        json.dumps(
            evidence,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    digest = hashlib.sha256(raw).hexdigest()
    sidecar = path.with_name(path.name + ".sha256")
    side_tmp = sidecar.with_name(sidecar.name + ".tmp")
    side_raw = f"{digest}  {path.name}\n".encode("ascii")
    with side_tmp.open("wb") as handle:
        handle.write(side_raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(side_tmp, sidecar)
    return digest


def _regular_tree_files(root: Path, *, label: str) -> list[Path]:
    if not root.is_dir() or root.is_symlink():
        raise ParityError(f"missing regular {label} directory: {root}")
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ParityError(f"symlink in {label} directory: {path}")
        if path.is_file():
            files.append(path.resolve())
        elif not path.is_dir():
            raise ParityError(f"unsupported {label} filesystem entry: {path}")
    if not files:
        raise ParityError(f"empty {label} directory: {root}")
    return files


def parity_input_files(
    *,
    off_arm_dir: Path,
    on_arm_dir: Path,
    off_results: Path,
    on_results: Path,
    extra_final_files: list[str],
) -> list[Path]:
    """Return every file consumed by a successful parity decision."""
    paths = [
        *_regular_tree_files(off_arm_dir / "syncer_probe", label="capture-OFF probe"),
        *_regular_tree_files(on_arm_dir / "syncer_probe", label="capture-ON probe"),
        *_regular_tree_files(off_arm_dir / "export", label="capture-OFF export"),
        *_regular_tree_files(on_arm_dir / "export", label="capture-ON export"),
        (off_arm_dir / "state.ckpt").resolve(strict=True),
        (on_arm_dir / "state.ckpt").resolve(strict=True),
        (off_arm_dir / "tape.jsonl").resolve(strict=True),
        (on_arm_dir / "tape.jsonl").resolve(strict=True),
        (on_arm_dir / "syncer_response_transcript.jsonl").resolve(strict=True),
        off_results.resolve(strict=True),
        on_results.resolve(strict=True),
    ]
    for relative_text in extra_final_files:
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise ParityError(f"unsafe final-file relative path: {relative_text!r}")
        paths.extend(
            [
                (off_arm_dir / relative).resolve(strict=True),
                (on_arm_dir / relative).resolve(strict=True),
            ]
        )
    unique: dict[str, Path] = {}
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ParityError(f"parity input is not a regular non-symlink file: {path}")
        unique[str(path)] = path
    return [unique[key] for key in sorted(unique)]


def write_input_manifest(path: Path, inputs: list[Path]) -> dict[str, Any]:
    """Atomically seal parity inputs in sha256sum-compatible form."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    for input_path in inputs:
        relative = os.path.relpath(input_path, start=path.parent)
        if "\n" in relative or "\r" in relative or "\\" in relative:
            raise ParityError(f"unsupported parity input path: {input_path}")
        rows.append(f"{sha256_file(input_path)}  {relative}\n")
    raw = "".join(rows).encode("utf-8")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {
        "path": str(path),
        "files": len(inputs),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "format": "sha256sum_relative_to_manifest_directory",
    }


def run_gate(
    *,
    off_arm_dir: Path,
    on_arm_dir: Path,
    off_results: Path,
    on_results: Path,
    off_arm: str,
    on_arm: str,
    output: Path,
    overhead_limit: float = 0.02,
    extra_final_files: list[str] | None = None,
    input_manifest: Path | None = None,
) -> dict[str, Any]:
    overhead_limit_error = None
    if not math.isfinite(overhead_limit) or overhead_limit < 0:
        overhead_limit_error = "overhead_limit must be finite and nonnegative"
    extra_final_files = list(extra_final_files or [])
    input_manifest = (
        input_manifest.resolve()
        if input_manifest is not None
        else output.with_suffix(".inputs.sha256").resolve()
    )
    off_arm_dir = off_arm_dir.resolve()
    on_arm_dir = on_arm_dir.resolve()
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    on_probe: ProbeCapture | None = None
    timing_intervals: dict[str, dict[str, Any]] = {}

    def execute(name: str, operation: Callable[[], dict[str, Any]]) -> None:
        try:
            detail = operation()
        except Exception as exc:  # fail evidence must survive every malformed input
            message = str(exc)
            checks.append({"name": name, "status": "FAIL", "error": message})
            errors.append(f"{name}: {message}")
        else:
            checks.append({"name": name, "status": "PASS", "detail": detail})

    def probe_operation() -> dict[str, Any]:
        nonlocal on_probe
        detail, on_probe = compare_probe_captures(
            off_arm_dir / "syncer_probe", on_arm_dir / "syncer_probe"
        )
        return detail

    execute("syncer_probe_exact_payload_parity", probe_operation)

    def transcript_operation() -> dict[str, Any]:
        off_transcript = off_arm_dir / "syncer_response_transcript.jsonl"
        if off_transcript.exists() or off_transcript.is_symlink():
            raise ParityError("capture-OFF arm unexpectedly has an audit transcript")
        if on_probe is None:
            raise ParityError(
                "cannot validate ON transcript after probe parity failure"
            )
        return validate_on_transcript(
            on_arm_dir / "syncer_response_transcript.jsonl", on_probe
        )

    execute("capture_on_transcript_join", transcript_operation)
    execute(
        "final_syncer_checkpoint_parity",
        lambda: compare_final_checkpoints(off_arm_dir, on_arm_dir, extra_final_files),
    )
    execute(
        "export_payload_parity",
        lambda: compare_export_trees(off_arm_dir, on_arm_dir),
    )

    def timing_operation() -> dict[str, Any]:
        if on_probe is None:
            raise ParityError("cannot validate timing after probe parity failure")
        off_timing = load_commit_interval(off_arm_dir / "tape.jsonl")
        on_timing = load_commit_interval(on_arm_dir / "tape.jsonl")
        if off_timing["commits"] != on_timing["commits"]:
            raise ParityError(
                "OFF/ON producer timing tapes have different commit counts"
            )
        if (
            off_timing["ordered_commit_identities"]
            != on_timing["ordered_commit_identities"]
        ):
            raise ParityError(
                "OFF/ON producer timing tapes have different ordered commit identities"
            )
        tape_groups = {
            (identity[1], identity[2])
            for identity in on_timing["ordered_commit_identities"]
        }
        if tape_groups != on_probe.group_keys:
            raise ParityError(
                "timing tape/probe commit groups differ; "
                f"missing={sorted(on_probe.group_keys - tape_groups)}, "
                f"extra={sorted(tape_groups - on_probe.group_keys)}"
            )
        timing_intervals["off"] = off_timing
        timing_intervals["on"] = on_timing
        return {"off": off_timing, "on": on_timing}

    execute("syncer_commit_interval_timing", timing_operation)

    def results_operation() -> dict[str, Any]:
        if overhead_limit_error is not None:
            raise ParityError(overhead_limit_error)
        if set(timing_intervals) != {"off", "on"}:
            raise ParityError("cannot evaluate overhead after producer timing failure")
        return compare_results(
            off_results,
            on_results,
            off_arm,
            on_arm,
            overhead_limit,
            timing_intervals["off"]["interval_ns"],
            timing_intervals["on"]["interval_ns"],
        )

    execute(
        "eval_and_wall_overhead",
        results_operation,
    )

    if not errors:
        execute(
            "sealed_input_tree",
            lambda: write_input_manifest(
                input_manifest,
                parity_input_files(
                    off_arm_dir=off_arm_dir,
                    on_arm_dir=on_arm_dir,
                    off_results=off_results,
                    on_results=on_results,
                    extra_final_files=extra_final_files,
                ),
            ),
        )
    elif input_manifest.exists() or input_manifest.is_symlink():
        input_manifest.unlink()

    evidence: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PASS" if not errors else "FAIL",
        "inputs": {
            "off_arm_dir": str(off_arm_dir),
            "on_arm_dir": str(on_arm_dir),
            "off_results": str(off_results.resolve()),
            "on_results": str(on_results.resolve()),
            "off_arm": off_arm,
            "on_arm": on_arm,
            "extra_final_files": extra_final_files,
            "input_manifest": str(input_manifest),
        },
        "thresholds": {
            "wall_overhead_fraction": (
                overhead_limit if overhead_limit_error is None else None
            ),
            "configuration_error": overhead_limit_error,
        },
        "checks": checks,
        "errors": errors,
        "canonicalization": {
            "probe_index_path_fields": list(PATH_FIELDS),
            "export_json_path_keys": sorted(EXPORT_PATH_KEYS),
            "all_other_values": "exact",
            "binary_payloads": "exact_sha256",
        },
        "limitations": [
            "The gate proves artifact equality, not that OFF and ON were scheduled on identical hardware or under identical external load.",
            "Overhead uses exact syncer monotonic timestamps from commit sequence 1 through N; rounded results.jsonl wall values are descriptive and must agree with the sealed tapes.",
            "The syncer probe must capture every commit and every responder; sampled or incomplete probe/transcript key sets fail.",
            "Only explicit path-valued metadata fields are canonicalized; arbitrary run names, labels, metrics, and unknown metadata remain exact.",
            "The sha256sum-compatible input manifest seals every file consumed by a successful parity decision and is independently required by the experiment harness.",
        ],
    }
    artifact_sha256 = write_evidence(output, evidence)
    # The digest cannot be embedded in the artifact it authenticates without
    # becoming self-referential.  Return it to callers (and the CLI) while the
    # durable value lives in ``<output>.sha256``.
    result = dict(evidence)
    result["artifact_sha256"] = artifact_sha256
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--off-arm-dir", type=Path, required=True)
    parser.add_argument("--on-arm-dir", type=Path, required=True)
    parser.add_argument("--off-results", type=Path, required=True)
    parser.add_argument("--on-results", type=Path, required=True)
    parser.add_argument(
        "--off-arm",
        required=True,
        help="capture-OFF arm name in --off-results",
    )
    parser.add_argument(
        "--on-arm",
        required=True,
        help="capture-ON arm name in --on-results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("report/optimizer_state_capture_parity.json"),
    )
    parser.add_argument("--overhead-limit", type=float, default=0.02)
    parser.add_argument(
        "--input-manifest",
        type=Path,
        help="sha256sum manifest for every consumed parity input; defaults next to --output",
    )
    parser.add_argument(
        "--extra-final-file",
        action="append",
        default=[],
        help="additional arm-relative final payload to compare exactly; repeatable",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    evidence = run_gate(
        off_arm_dir=args.off_arm_dir,
        on_arm_dir=args.on_arm_dir,
        off_results=args.off_results,
        on_results=args.on_results,
        off_arm=args.off_arm,
        on_arm=args.on_arm,
        output=args.output,
        overhead_limit=args.overhead_limit,
        extra_final_files=args.extra_final_file,
        input_manifest=args.input_manifest,
    )
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "output": str(args.output),
                "artifact_sha256": evidence["artifact_sha256"],
                "errors": evidence["errors"],
            },
            sort_keys=True,
        )
    )
    return 0 if evidence["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
