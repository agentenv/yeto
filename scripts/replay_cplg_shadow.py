#!/usr/bin/env python3
"""Fail-closed full-vector stock-tape shadow evaluator for angle-based CPLG.

Every tape row must bind the exact little-endian f32 stock vector through a
root-contained regular file and lowercase SHA-256.  Scalar norms, dot
products, PTI hashes, or checkpoint names are not substitutes: a stock tape
without the full vectors is rejected as unidentifiable before any CPLG score
is reported.

The result is historical, off-policy direction geometry only.  It is not a
finite-loss comparison and cannot establish that CPLG beats SGD-0.28.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any

from yeto.cplg_sgd import (
    CPLG_ANGLE_CAP_F32_BITS,
    CPLG_CROSS_RUNTIME_BIT_PARITY,
    CPLGF32Trig,
    CPLG_PLATFORM_CAP_ATAN2F_BITS,
    CPLG_PLATFORM_CAP_MATCHES_PINNED,
    CPLGRustLibmTrig,
    CPLG_RUST_LIBM_CROSS_RUNTIME_BIT_PARITY,
    CPLG_RUST_LIBM_TRIG_BACKEND,
    CPLG_RUST_LIBM_VERSION,
    CPLGReferenceMachine,
    CPLG_TRIG_BACKEND,
    CPLG_TRIG_PORTABILITY_LIMITATION,
    decode_f32le,
)


LOWER_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
STOCK_VECTOR_ROW_FIELDS = frozenset(
    {
        "schema",
        "capture_session_uuid",
        "commit_seq",
        "step",
        "fragment",
        "fragment_version_before",
        "fragment_version_after",
        "layout_sha256",
        "run_config_sha256",
        "merge_rule",
        "wire_dtype",
        "numel",
        "responders",
        "stock_f32le",
        "stock_f32le_sha256",
        "ledger_prev_sha256",
        "ledger_sha256",
    }
)
REPORT_SCHEMA = "cplg_full_vector_stock_shadow_v1"
SHADOW_GATE_FRAGMENTS = (0, 1, 2, 3)
SHADOW_GATE_FRAGMENT_ORDER = SHADOW_GATE_FRAGMENTS * 8
SHADOW_GATE_BOOTSTRAP_DRAWS = 20_000
SHADOW_GATE_BOOTSTRAP_BLOCK = 2
SHADOW_GATE_BOOTSTRAP_SEED = 0x0000000043504C47
SHADOW_GATE_BOOTSTRAP_LOWER_INDEX = 1000
U64_MASK = (1 << 64) - 1


@dataclass(frozen=True)
class _StockVector:
    commit_seq: int
    fragment: int
    numel: int
    raw: bytes
    sha256: str
    relative_path: str


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


class _SplitMix64:
    def __init__(self, seed: int) -> None:
        self.state = seed & U64_MASK

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & U64_MASK
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & U64_MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & U64_MASK
        return (value ^ (value >> 31)) & U64_MASK


def _fragment_stratified_circular_block_lower_endpoint(
    scores_by_fragment: dict[int, list[float]],
) -> float:
    if tuple(sorted(scores_by_fragment)) != SHADOW_GATE_FRAGMENTS:
        raise ValueError("bootstrap requires exact fragments 0,1,2,3")
    if any(
        len(scores_by_fragment[fragment]) != 5 for fragment in SHADOW_GATE_FRAGMENTS
    ):
        raise ValueError("bootstrap requires exactly five scores per fragment")
    generator = _SplitMix64(SHADOW_GATE_BOOTSTRAP_SEED)
    statistics: list[float] = []
    for _draw in range(SHADOW_GATE_BOOTSTRAP_DRAWS):
        resampled: list[float] = []
        for fragment in SHADOW_GATE_FRAGMENTS:
            source = scores_by_fragment[fragment]
            fragment_draw: list[float] = []
            while len(fragment_draw) < len(source):
                start = generator.next_u64() % len(source)
                for offset in range(SHADOW_GATE_BOOTSTRAP_BLOCK):
                    fragment_draw.append(source[(start + offset) % len(source)])
            resampled.extend(fragment_draw[: len(source)])
        statistics.append(math.fsum(resampled) / 20.0)
    statistics.sort()
    return statistics[SHADOW_GATE_BOOTSTRAP_LOWER_INDEX]


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _strict_json_object(text: str, context: str) -> dict[str, Any]:
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"{context}: invalid strict JSON: {error}") from error
    if type(value) is not dict:
        raise ValueError(f"{context}: top-level JSON value must be an object")
    return value


def _exact_int(row: dict[str, Any], field: str, context: str) -> int:
    if field not in row:
        raise ValueError(f"{context}: missing field {field!r}")
    value = row[field]
    if type(value) is not int:
        raise ValueError(f"{context}: {field} must be an exact JSON integer")
    return value


def _exact_string(row: dict[str, Any], field: str, context: str) -> str:
    if field not in row:
        raise ValueError(f"{context}: missing field {field!r}")
    value = row[field]
    if type(value) is not str or not value:
        raise ValueError(f"{context}: {field} must be a nonempty JSON string")
    return value


def _root_contained_regular_file(root: Path, relative: str, context: str) -> Path:
    candidate = Path(relative)
    if (
        candidate.is_absolute()
        or not candidate.parts
        or any(part in ("", ".", "..") for part in candidate.parts)
    ):
        raise ValueError(f"{context}: vector path must be relative and root-contained")
    current = root
    for part in candidate.parts:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as error:
            raise ValueError(f"{context}: vector file does not exist") from error
        if stat.S_ISLNK(mode):
            raise ValueError(f"{context}: symlink path component is forbidden")
    if not stat.S_ISREG(current.lstat().st_mode):
        raise ValueError(f"{context}: vector path must name a regular file")
    resolved = current.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{context}: vector path escapes tape root") from error
    return resolved


def read_full_vector_stock_tape(
    tape_path: Path,
) -> tuple[list[_StockVector], list[dict[str, Any]], str, dict[str, Any]]:
    """Read a chronological tape, rejecting absent or corrupt full vectors."""

    try:
        tape_mode = tape_path.lstat().st_mode
    except FileNotFoundError as error:
        raise ValueError(f"{tape_path}: stock tape does not exist") from error
    if stat.S_ISLNK(tape_mode) or not stat.S_ISREG(tape_mode):
        raise ValueError(f"{tape_path}: stock tape must be a regular non-symlink file")
    tape_path = tape_path.resolve(strict=True)
    root = tape_path.parent
    tape_raw = tape_path.read_bytes()
    try:
        tape_text = tape_raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{tape_path}: stock tape is not UTF-8") from error

    vectors: list[_StockVector] = []
    provenance: list[dict[str, Any]] = []
    expected_commit_seq = 1
    fragment_numel: dict[int, int] = {}
    fragment_version: dict[int, int] = {}
    ledger_head = "0" * 64
    capture_session: str | None = None
    layout_sha256: str | None = None
    run_config_sha256: str | None = None
    for line_number, line in enumerate(tape_text.splitlines(), 1):
        if not line.strip():
            continue
        context = f"{tape_path}:{line_number}"
        row = _strict_json_object(line, context)
        missing = sorted(STOCK_VECTOR_ROW_FIELDS - row.keys())
        if missing:
            raise ValueError(
                f"{context}: full stock vectors are required for CPLG; "
                f"missing fields {missing}"
            )
        unexpected = sorted(row.keys() - STOCK_VECTOR_ROW_FIELDS)
        if unexpected:
            raise ValueError(
                f"{context}: unexpected stock-vector row fields {unexpected}"
            )
        if row["schema"] != "cplg_stock_vector_row_v1":
            raise ValueError(f"{context}: unexpected stock-vector row schema")

        commit_seq = _exact_int(row, "commit_seq", context)
        step = _exact_int(row, "step", context)
        fragment = _exact_int(row, "fragment", context)
        version_before = _exact_int(row, "fragment_version_before", context)
        version_after = _exact_int(row, "fragment_version_after", context)
        numel = _exact_int(row, "numel", context)
        if commit_seq != expected_commit_seq:
            raise ValueError(
                f"{context}: commit_seq must be contiguous; expected "
                f"{expected_commit_seq}, observed {commit_seq}"
            )
        expected_commit_seq += 1
        if step != commit_seq:
            raise ValueError(
                f"{context}: step must equal deterministic commit sequence"
            )
        if fragment < 0 or numel <= 0:
            raise ValueError(
                f"{context}: fragment must be nonnegative and numel positive"
            )
        prior_numel = fragment_numel.setdefault(fragment, numel)
        if prior_numel != numel:
            raise ValueError(
                f"{context}: fragment {fragment} numel changed from "
                f"{prior_numel} to {numel}"
            )
        expected_version_before = fragment_version.get(fragment, 0)
        if version_before != expected_version_before or version_after != step:
            raise ValueError(
                f"{context}: fragment version continuity failed; expected "
                f"{expected_version_before}->{step}, observed "
                f"{version_before}->{version_after}"
            )
        fragment_version[fragment] = version_after

        session = _exact_string(row, "capture_session_uuid", context)
        row_layout = _exact_string(row, "layout_sha256", context)
        row_config = _exact_string(row, "run_config_sha256", context)
        for field, digest in (
            ("layout_sha256", row_layout),
            ("run_config_sha256", row_config),
        ):
            if LOWER_SHA256_RE.fullmatch(digest) is None:
                raise ValueError(f"{context}: {field} must be canonical SHA-256")
        if capture_session is None:
            capture_session = session
            layout_sha256 = row_layout
            run_config_sha256 = row_config
        elif (
            session != capture_session
            or row_layout != layout_sha256
            or row_config != run_config_sha256
        ):
            raise ValueError(f"{context}: stock-vector run identity changed")
        if (
            row["merge_rule"] != "production_weighted_rda"
            or row["wire_dtype"] != "f32_le"
        ):
            raise ValueError(
                f"{context}: merge or wire identity differs from frozen contract"
            )

        responders = row["responders"]
        if type(responders) is not list or len(responders) != 1:
            raise ValueError(
                f"{context}: shadow profile requires exactly one responder"
            )
        responder = responders[0]
        if type(responder) is not dict or set(responder) != {"id", "weight_f64_bits"}:
            raise ValueError(f"{context}: responder fields differ from closed schema")
        if type(responder["id"]) is not int or responder["id"] != 0:
            raise ValueError(f"{context}: shadow profile requires responder ID zero")
        weight_bits = responder["weight_f64_bits"]
        if (
            type(weight_bits) is not str
            or len(weight_bits) != 16
            or any(character not in "0123456789abcdef" for character in weight_bits)
        ):
            raise ValueError(f"{context}: responder weight bits are noncanonical")

        predecessor = _exact_string(row, "ledger_prev_sha256", context)
        row_digest = _exact_string(row, "ledger_sha256", context)
        if predecessor != ledger_head:
            raise ValueError(f"{context}: ledger predecessor mismatch")
        if LOWER_SHA256_RE.fullmatch(row_digest) is None:
            raise ValueError(f"{context}: ledger SHA-256 is noncanonical")
        unhashed = dict(row)
        del unhashed["ledger_sha256"]
        canonical = (
            json.dumps(
                unhashed,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        actual_row_digest = sha256_bytes(canonical)
        if actual_row_digest != row_digest:
            raise ValueError(f"{context}: ledger row SHA-256 mismatch")
        ledger_head = row_digest

        relative = _exact_string(row, "stock_f32le", context)
        expected_sha256 = _exact_string(row, "stock_f32le_sha256", context)
        if LOWER_SHA256_RE.fullmatch(expected_sha256) is None:
            raise ValueError(
                f"{context}: stock_f32le_sha256 must be lowercase hexadecimal SHA-256"
            )
        path = _root_contained_regular_file(
            root,
            relative,
            f"{context}:stock_f32le",
        )
        raw = path.read_bytes()
        actual_sha256 = sha256_bytes(raw)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"{context}: full stock vector SHA-256 mismatch")
        if len(raw) != numel * 4:
            raise ValueError(
                f"{context}: full stock vector requires {numel * 4} bytes, "
                f"received {len(raw)}"
            )
        values = decode_f32le(raw, (numel,))
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{context}: full stock vector contains nonfinite values")

        vector = _StockVector(
            commit_seq=commit_seq,
            fragment=fragment,
            numel=numel,
            raw=raw,
            sha256=actual_sha256,
            relative_path=relative,
        )
        vectors.append(vector)
        provenance.append(
            {
                "commit_seq": commit_seq,
                "fragment": fragment,
                "numel": numel,
                "stock_f32le": relative,
                "expected_sha256": expected_sha256,
                "actual_sha256": actual_sha256,
                "ledger_sha256": ledger_head,
            }
        )
    if not vectors:
        raise ValueError(f"{tape_path}: stock tape contains no records")
    tape_sha256 = sha256_bytes(tape_raw)
    manifest = _validate_stock_tape_manifest(
        tape_path,
        tape_sha256=tape_sha256,
        ledger_head=ledger_head,
        capture_session=capture_session,
        layout_sha256=layout_sha256,
        run_config_sha256=run_config_sha256,
        records=len(vectors),
        vector_bytes=sum(vector.numel * 4 for vector in vectors),
        fragment_counts=Counter(vector.fragment for vector in vectors),
    )
    return vectors, provenance, tape_sha256, manifest


def _validate_stock_tape_manifest(
    tape_path: Path,
    *,
    tape_sha256: str,
    ledger_head: str,
    capture_session: str | None,
    layout_sha256: str | None,
    run_config_sha256: str | None,
    records: int,
    vector_bytes: int,
    fragment_counts: Counter[int],
) -> dict[str, Any]:
    if tape_path.name != "stock_tape.jsonl":
        raise ValueError(f"{tape_path}: canonical tape basename is required")
    manifest_path = tape_path.with_name("stock_tape.manifest.json")
    checksum_path = Path(f"{manifest_path}.sha256")
    for path, label in ((manifest_path, "manifest"), (checksum_path, "checksum")):
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError as error:
            raise ValueError(f"{path}: stock-tape {label} is missing") from error
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise ValueError(f"{path}: stock-tape {label} must be a regular file")
    manifest_raw = manifest_path.read_bytes()
    manifest_sha256 = sha256_bytes(manifest_raw)
    expected_checksum = f"{manifest_sha256}  {manifest_path.name}\n".encode("ascii")
    if checksum_path.read_bytes() != expected_checksum:
        raise ValueError(f"{checksum_path}: manifest checksum sidecar mismatch")
    try:
        manifest_text = manifest_raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{manifest_path}: manifest is not UTF-8") from error
    manifest = _strict_json_object(manifest_text, str(manifest_path))
    manifest_fields = {
        "schema",
        "status",
        "capture_session_uuid",
        "layout_sha256",
        "initial_state_sha256",
        "run_config_sha256",
        "expected_records",
        "records",
        "vector_bytes",
        "fragment_counts",
        "stock_tape",
        "stock_tape_sha256",
        "ledger_head",
        "writer",
    }
    _require_closed_fields(manifest, manifest_fields, str(manifest_path))
    if manifest["schema"] != "cplg_stock_vector_tape_manifest_v1":
        raise ValueError(f"{manifest_path}: unexpected schema")
    if manifest["status"] != "COMPLETE":
        raise ValueError(f"{manifest_path}: capture status is not COMPLETE")
    initial_state_sha256 = manifest["initial_state_sha256"]
    if (
        type(initial_state_sha256) is not str
        or LOWER_SHA256_RE.fullmatch(initial_state_sha256) is None
    ):
        raise ValueError(f"{manifest_path}: initial_state_sha256 is noncanonical")
    expected_identities = {
        "capture_session_uuid": capture_session,
        "layout_sha256": layout_sha256,
        "run_config_sha256": run_config_sha256,
        "stock_tape": tape_path.name,
        "stock_tape_sha256": tape_sha256,
        "ledger_head": ledger_head,
    }
    for field, expected in expected_identities.items():
        if manifest[field] != expected:
            raise ValueError(f"{manifest_path}: {field} mismatch")
    for field, expected in (
        ("expected_records", records),
        ("records", records),
        ("vector_bytes", vector_bytes),
    ):
        if type(manifest[field]) is not int or manifest[field] != expected:
            raise ValueError(f"{manifest_path}: {field} differs from {expected}")
    expected_fragment_counts = {
        str(fragment): count for fragment, count in sorted(fragment_counts.items())
    }
    if manifest["fragment_counts"] != expected_fragment_counts:
        raise ValueError(f"{manifest_path}: fragment counts mismatch")

    writer = manifest["writer"]
    writer_fields = {
        "state",
        "accepted_items",
        "completed_items",
        "accepted_bytes",
        "completed_bytes",
        "dropped_items",
        "dropped_bytes",
        "abandoned_items",
        "abandoned_bytes",
        "pending_items",
        "pending_bytes",
        "error",
    }
    if type(writer) is not dict:
        raise ValueError(f"{manifest_path}: writer must be an object")
    _require_closed_fields(writer, writer_fields, f"{manifest_path}:writer")
    expected_writer = {
        "state": "closed",
        "accepted_items": records,
        "completed_items": records,
        "accepted_bytes": vector_bytes,
        "completed_bytes": vector_bytes,
        "dropped_items": 0,
        "dropped_bytes": 0,
        "abandoned_items": 0,
        "abandoned_bytes": 0,
        "pending_items": 0,
        "pending_bytes": 0,
        "error": None,
    }
    if writer != expected_writer:
        raise ValueError(f"{manifest_path}: writer did not close exactly")
    return {
        "path": manifest_path.name,
        "sha256": manifest_sha256,
        "ledger_head": ledger_head,
        "capture_session_uuid": capture_session,
        "layout_sha256": layout_sha256,
        "initial_state_sha256": initial_state_sha256,
        "run_config_sha256": run_config_sha256,
        "writer": writer,
    }


def _require_closed_fields(
    value: dict[str, Any],
    expected: set[str],
    context: str,
) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{context}: fields differ from closed schema; "
            f"missing={sorted(expected - value.keys())}, "
            f"unexpected={sorted(value.keys() - expected)}"
        )


def read_shadow_overhead_evidence(
    path: Path,
    *,
    stock_tape_sha256: str,
    total_vector_bytes: int,
    initial_state_sha256: str,
) -> dict[str, Any]:
    """Validate matched OFF/ON interval and writer evidence, then compute cost."""

    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise ValueError(f"{path}: overhead evidence does not exist") from error
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ValueError(f"{path}: overhead evidence must be a regular file")
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    checksum_path = Path(f"{path}.sha256")
    try:
        checksum_mode = checksum_path.lstat().st_mode
    except FileNotFoundError as error:
        raise ValueError(f"{path}: overhead checksum sidecar is missing") from error
    if stat.S_ISLNK(checksum_mode) or not stat.S_ISREG(checksum_mode):
        raise ValueError(f"{path}: overhead checksum must be a regular file")
    expected_checksum = f"{digest}  {path.name}\n".encode("ascii")
    if checksum_path.read_bytes() != expected_checksum:
        raise ValueError(f"{path}: overhead checksum sidecar mismatch")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path}: overhead evidence is not UTF-8") from error
    document = _strict_json_object(text, str(path))
    _require_closed_fields(document, {"schema", "off", "on"}, str(path))
    if document["schema"] != "cplg_shadow_overhead_v1":
        raise ValueError(f"{path}: unexpected overhead schema")

    arm_fields = {
        "capture_enabled",
        "interval_start_monotonic_ns",
        "interval_end_monotonic_ns",
        "commits",
        "local_steps",
        "fragment_order",
        "initial_state_sha256",
        "input_manifest_sha256",
        "schedule_sha256",
        "runner_exit_code",
        "evaluation_finite",
        "stock_tape_sha256",
        "writer",
    }
    writer_fields = {
        "state",
        "accepted_items",
        "completed_items",
        "accepted_bytes",
        "completed_bytes",
        "dropped_items",
        "dropped_bytes",
        "abandoned_items",
        "abandoned_bytes",
        "pending_items",
        "pending_bytes",
        "error",
    }
    arms: dict[str, dict[str, Any]] = {}
    for name in ("off", "on"):
        arm = document[name]
        if type(arm) is not dict:
            raise ValueError(f"{path}:{name}: arm must be an object")
        _require_closed_fields(arm, arm_fields, f"{path}:{name}")
        if type(arm["capture_enabled"]) is not bool:
            raise ValueError(f"{path}:{name}: capture_enabled must be boolean")
        for field in (
            "interval_start_monotonic_ns",
            "interval_end_monotonic_ns",
            "commits",
            "local_steps",
            "runner_exit_code",
        ):
            if type(arm[field]) is not int:
                raise ValueError(f"{path}:{name}:{field}: exact integer required")
        if (
            arm["interval_start_monotonic_ns"] < 0
            or arm["interval_end_monotonic_ns"] <= arm["interval_start_monotonic_ns"]
        ):
            raise ValueError(f"{path}:{name}: invalid monotonic interval")
        if arm["commits"] != 32 or arm["local_steps"] != 34:
            raise ValueError(f"{path}:{name}: matched work counts differ from 32/34")
        if (
            type(arm["fragment_order"]) is not list
            or any(type(fragment) is not int for fragment in arm["fragment_order"])
            or tuple(arm["fragment_order"]) != SHADOW_GATE_FRAGMENT_ORDER
        ):
            raise ValueError(f"{path}:{name}: fragment order differs from frozen order")
        for field in (
            "initial_state_sha256",
            "input_manifest_sha256",
            "schedule_sha256",
        ):
            if (
                type(arm[field]) is not str
                or LOWER_SHA256_RE.fullmatch(arm[field]) is None
            ):
                raise ValueError(f"{path}:{name}:{field}: canonical SHA-256 required")
        if arm["runner_exit_code"] != 0 or arm["evaluation_finite"] is not True:
            raise ValueError(f"{path}:{name}: arm did not finish safely")

        writer = arm["writer"]
        if type(writer) is not dict:
            raise ValueError(f"{path}:{name}: writer must be an object")
        _require_closed_fields(writer, writer_fields, f"{path}:{name}:writer")
        for field in writer_fields - {"state", "error"}:
            if type(writer[field]) is not int or writer[field] < 0:
                raise ValueError(
                    f"{path}:{name}:writer:{field}: nonnegative int required"
                )
        if writer["error"] is not None:
            raise ValueError(f"{path}:{name}: writer reports an error")
        arms[name] = arm

    if arms["off"]["capture_enabled"] is not False:
        raise ValueError(f"{path}: OFF arm unexpectedly enabled capture")
    if arms["on"]["capture_enabled"] is not True:
        raise ValueError(f"{path}: ON arm did not enable capture")
    for identity_field in (
        "initial_state_sha256",
        "input_manifest_sha256",
        "schedule_sha256",
        "fragment_order",
    ):
        if arms["off"][identity_field] != arms["on"][identity_field]:
            raise ValueError(
                f"{path}: matched arm identity differs at {identity_field}"
            )
    if arms["off"]["stock_tape_sha256"] is not None:
        raise ValueError(f"{path}: OFF arm must not publish a stock tape")
    if arms["on"]["stock_tape_sha256"] != stock_tape_sha256:
        raise ValueError(f"{path}: ON arm stock tape identity mismatch")
    if arms["on"]["initial_state_sha256"] != initial_state_sha256:
        raise ValueError(f"{path}: ON arm initial-state identity differs from tape")

    off_writer = arms["off"]["writer"]
    if off_writer["state"] != "disabled" or any(
        off_writer[field] != 0 for field in writer_fields - {"state", "error"}
    ):
        raise ValueError(f"{path}: OFF writer accounting is not disabled/zero")
    on_writer = arms["on"]["writer"]
    if on_writer["state"] != "closed":
        raise ValueError(f"{path}: ON writer is not closed")
    expected_on = {
        "accepted_items": 32,
        "completed_items": 32,
        "accepted_bytes": total_vector_bytes,
        "completed_bytes": total_vector_bytes,
    }
    for field, expected in expected_on.items():
        if on_writer[field] != expected:
            raise ValueError(f"{path}: ON writer {field} differs from {expected}")
    for field in (
        "dropped_items",
        "dropped_bytes",
        "abandoned_items",
        "abandoned_bytes",
        "pending_items",
        "pending_bytes",
    ):
        if on_writer[field] != 0:
            raise ValueError(f"{path}: ON writer {field} is nonzero")

    off_ns = (
        arms["off"]["interval_end_monotonic_ns"]
        - arms["off"]["interval_start_monotonic_ns"]
    )
    on_ns = (
        arms["on"]["interval_end_monotonic_ns"]
        - arms["on"]["interval_start_monotonic_ns"]
    )
    return {
        "schema": document["schema"],
        "off_interval_ns": off_ns,
        "on_interval_ns": on_ns,
        "overhead_fraction": (on_ns - off_ns) / off_ns,
        "evidence_sha256": digest,
    }


def evaluate_full_vector_stock_tape(
    tape_path: Path,
    *,
    trig: CPLGF32Trig | None = None,
    enforce_shadow_gate: bool = False,
    overhead_evidence_path: Path | None = None,
) -> dict[str, Any]:
    """Construct and score causal CPLG shadows from exact factual vectors."""

    vectors, provenance, tape_sha256, tape_manifest = read_full_vector_stock_tape(
        tape_path
    )
    if isinstance(trig, CPLGRustLibmTrig):
        trig_contract = {
            "trig_backend": CPLG_RUST_LIBM_TRIG_BACKEND,
            "trig_portability_limitation": None,
            "platform_cap_atan2f_bits": f"0x{CPLG_ANGLE_CAP_F32_BITS:08x}",
            "platform_cap_matches_pinned": True,
            "cross_runtime_bit_parity": CPLG_RUST_LIBM_CROSS_RUNTIME_BIT_PARITY,
            "authoritative_transcendental_runtime": "rust",
            "rust_libm_version": CPLG_RUST_LIBM_VERSION,
            "rust_libm_helper_sha256": trig.executable_sha256,
            "rust_libm_helper_basename": trig.executable.name,
        }
        trig_limitation = (
            "the hash-pinned Rust helper is authoritative only for f32 "
            "transcendentals; Python independently evaluates all vector geometry"
        )
    else:
        trig_contract = {
            "trig_backend": CPLG_TRIG_BACKEND,
            "trig_portability_limitation": CPLG_TRIG_PORTABILITY_LIMITATION,
            "platform_cap_atan2f_bits": (f"0x{CPLG_PLATFORM_CAP_ATAN2F_BITS:08x}"),
            "platform_cap_matches_pinned": CPLG_PLATFORM_CAP_MATCHES_PINNED,
            "cross_runtime_bit_parity": CPLG_CROSS_RUNTIME_BIT_PARITY,
            "authoritative_transcendental_runtime": None,
            "rust_libm_version": None,
            "rust_libm_helper_sha256": None,
            "rust_libm_helper_basename": None,
        }
        trig_limitation = (
            "host C libm f32 transcendental last bits are not cross-platform pinned"
        )
    machines: dict[int, CPLGReferenceMachine] = {}
    fragment_numels: dict[int, int] = {}
    records: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    scores: list[float] = []
    scores_by_fragment: dict[int, list[float]] = defaultdict(list)
    sealed_shadows = 0
    nonstock_candidates = 0
    simulated_actions = 0

    for current in vectors:
        prior_numel = fragment_numels.setdefault(current.fragment, current.numel)
        if prior_numel != current.numel:
            raise ValueError(f"fragment {current.fragment}: CPLG shape changed")
        machine = machines.get(current.fragment)
        if machine is None:
            machine = CPLGReferenceMachine((current.numel,), trig=trig)
            machines[current.fragment] = machine
        prior_pending = machine.state.pending_candidate_raw
        prior_source_commit_seq = None
        if prior_pending is not None:
            prior_source_commit_seq = next(
                record["commit_seq"]
                for record in reversed(records)
                if record["fragment"] == current.fragment and record["sealed_shadow"]
            )
        preview = machine.preview(current.raw)
        if preview.sealed_shadow:
            sealed_shadows += 1
        if preview.candidate_raw is not None and preview.candidate_raw != current.raw:
            nonstock_candidates += 1
        if preview.used_nonstock:
            simulated_actions += 1
        if preview.resolved_shadow_score is not None:
            scores.append(preview.resolved_shadow_score)
            scores_by_fragment[current.fragment].append(preview.resolved_shadow_score)
        candidate_sha256 = (
            sha256_bytes(preview.candidate_raw)
            if preview.candidate_raw is not None
            else None
        )
        reason_counts[preview.reason] += 1
        records.append(
            {
                "commit_seq": current.commit_seq,
                "fragment": current.fragment,
                "stock_sha256": current.sha256,
                "sealed_shadow": preview.sealed_shadow,
                "candidate_is_nonstock": (
                    preview.candidate_raw is not None
                    and preview.candidate_raw != current.raw
                ),
                "simulated_nonstock_action": preview.used_nonstock,
                "interlock_open": preview.interlock_open,
                "interlock_score_count": preview.interlock_score_count,
                "state_cleared": preview.state_cleared,
                "reason": preview.reason,
                "candidate_sha256": candidate_sha256,
                "action_sha256": sha256_bytes(preview.action_raw),
                "current_turn_angle_radians": (preview.current_turn_angle_radians),
                "previous_turn_angle_radians": (preview.previous_turn_angle_radians),
                "transported_forward_tangent_coherence": (
                    preview.transported_forward_tangent_coherence
                ),
                "commanded_angle_radians": preview.commanded_angle_radians,
                "resolved_source_commit_seq": prior_source_commit_seq,
                "resolved_shadow_cosine_gain": preview.resolved_shadow_score,
            }
        )
        machine.commit_preview(preview)

    per_fragment = {}
    for fragment in sorted(machines):
        fragment_scores = scores_by_fragment[fragment]
        per_fragment[str(fragment)] = {
            "records": sum(vector.fragment == fragment for vector in vectors),
            "resolved_shadows": len(fragment_scores),
            "positive_resolved_shadows": sum(score > 0.0 for score in fragment_scores),
            "mean_shadow_cosine_gain": (
                math.fsum(fragment_scores) / len(fragment_scores)
                if fragment_scores
                else None
            ),
            "unresolved_tail_shadow": (
                machines[fragment].state.pending_candidate_raw is not None
            ),
        }

    unresolved_tails = sum(
        machine.state.pending_candidate_raw is not None for machine in machines.values()
    )

    report = {
        "schema": REPORT_SCHEMA,
        "decision": "DIRECTION_SHADOW_ONLY",
        "identifiable": True,
        "full_vectors_verified": True,
        "causal_finite_loss_claim": False,
        "beats_sgd_0_28_claim": False,
        "input_stock_tape_sha256": tape_sha256,
        "input_stock_tape_manifest": tape_manifest,
        "input_provenance": provenance,
        "reference_contract": {
            "kernel": "angle_based_cplg_sequential_f32_no_fma",
            "angle_cap_f32_bits": f"0x{CPLG_ANGLE_CAP_F32_BITS:08x}",
            "forward_tangent_sign": "rho_times_current_minus_previous",
            "backward_tangent_sign_not_used": "previous_minus_rho_times_current",
            "command_is_angle_radians_not_tangent_ratio": True,
            **trig_contract,
        },
        "summary": {
            "records": len(vectors),
            "sealed_shadows": sealed_shadows,
            "constructed_nonstock_shadows": nonstock_candidates,
            "simulated_nonstock_actions": simulated_actions,
            "resolved_shadows": len(scores),
            "positive_resolved_shadows": sum(score > 0.0 for score in scores),
            "mean_shadow_cosine_gain": (
                math.fsum(scores) / len(scores) if scores else None
            ),
            "unresolved_tail_shadows": unresolved_tails,
            "reason_counts": dict(sorted(reason_counts.items())),
        },
        "per_fragment": per_fragment,
        "records": records,
        "limitations": [
            "historical stock directions are off-policy after the first hypothetical CPLG action",
            "directional cosine gain is not a finite-loss or convergence guarantee",
            trig_limitation,
        ],
    }
    if not enforce_shadow_gate:
        return report
    if overhead_evidence_path is None:
        raise ValueError("frozen shadow gate requires matched overhead evidence")

    gate_errors: list[str] = []
    fragment_order = tuple(vector.fragment for vector in vectors)
    fragment_counts = Counter(fragment_order)
    if len(vectors) != 32:
        gate_errors.append(f"records: expected 32, observed {len(vectors)}")
    if fragment_order != SHADOW_GATE_FRAGMENT_ORDER:
        gate_errors.append(
            "fragment_order: differs from [0,1,2,3] repeated eight times"
        )
    if dict(sorted(fragment_counts.items())) != {0: 8, 1: 8, 2: 8, 3: 8}:
        gate_errors.append(
            f"fragment_counts: observed {dict(sorted(fragment_counts.items()))}"
        )
    if sealed_shadows != 24:
        gate_errors.append(f"sealed_shadows: expected 24, observed {sealed_shadows}")
    if len(scores) != 20:
        gate_errors.append(f"resolved_shadows: expected 20, observed {len(scores)}")
    if unresolved_tails != 4:
        gate_errors.append(
            f"unresolved_tail_shadows: expected 4, observed {unresolved_tails}"
        )
    if any(record["state_cleared"] for record in records):
        gate_errors.append("state_cleared: at least one boundary cleared causal state")
    if simulated_actions < 8:
        gate_errors.append(
            f"simulated_nonstock_actions: required at least 8, observed {simulated_actions}"
        )

    mean_score = math.fsum(scores) / len(scores) if scores else None
    if mean_score is None or not mean_score > 0.001:
        gate_errors.append(
            f"mean_shadow_cosine_gain: required >0.001, observed {mean_score}"
        )
    positive_fragment_means = sum(
        value["mean_shadow_cosine_gain"] is not None
        and value["mean_shadow_cosine_gain"] > 0.0
        for value in per_fragment.values()
    )
    if positive_fragment_means < 3:
        gate_errors.append(
            "positive_fragment_means: required at least 3, "
            f"observed {positive_fragment_means}"
        )

    bootstrap_lower_endpoint: float | None = None
    try:
        bootstrap_lower_endpoint = _fragment_stratified_circular_block_lower_endpoint(
            scores_by_fragment
        )
    except ValueError as error:
        gate_errors.append(f"bootstrap: {error}")
    if bootstrap_lower_endpoint is None or not bootstrap_lower_endpoint > 0.0:
        gate_errors.append(
            "bootstrap_lower_endpoint: required >0, "
            f"observed {bootstrap_lower_endpoint}"
        )

    if not isinstance(trig, CPLGRustLibmTrig):
        gate_errors.append("numerical_authority: pinned Rust-libm helper was not used")
    total_vector_bytes = sum(vector.numel * 4 for vector in vectors)
    overhead = read_shadow_overhead_evidence(
        overhead_evidence_path,
        stock_tape_sha256=tape_sha256,
        total_vector_bytes=total_vector_bytes,
        initial_state_sha256=tape_manifest["initial_state_sha256"],
    )
    if overhead["overhead_fraction"] > 0.02:
        gate_errors.append(
            "matched_interval_overhead: required <=0.02, observed "
            f"{overhead['overhead_fraction']}"
        )

    report["decision"] = "PASS" if not gate_errors else "FAIL"
    report["shadow_gate"] = {
        "status": report["decision"],
        "errors": gate_errors,
        "activity": {
            "denominator_boundaries": 32,
            "simulated_nonstock_actions": simulated_actions,
            "minimum": 8,
        },
        "direction": {
            "resolved_scores": len(scores),
            "mean": mean_score,
            "required_mean_strictly_greater_than": 0.001,
            "positive_fragment_means": positive_fragment_means,
            "required_positive_fragment_means": 3,
            "bootstrap": {
                "draws": SHADOW_GATE_BOOTSTRAP_DRAWS,
                "seed_hex": f"0x{SHADOW_GATE_BOOTSTRAP_SEED:016x}",
                "block_length": SHADOW_GATE_BOOTSTRAP_BLOCK,
                "lower_order_index": SHADOW_GATE_BOOTSTRAP_LOWER_INDEX,
                "lower_endpoint": bootstrap_lower_endpoint,
                "required_strictly_greater_than": 0.0,
            },
        },
        "overhead": {
            **overhead,
            "maximum_fraction": 0.02,
        },
        "next_action": (
            "write_and_review_separate_cplg_e1_preregistration"
            if not gate_errors
            else "kill_cplg_v1"
        ),
    }
    return report


def _atomic_write_new(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.tmp.",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o644)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to replace stale evidence: {path}"
            ) from error
        temporary.unlink()
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_checksummed_report(path: Path, report: dict[str, Any]) -> str:
    checksum_path = Path(f"{path}.sha256")
    if path.exists() or checksum_path.exists():
        raise FileExistsError(f"analysis output path is not fresh: {path}")
    rendered = (
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    digest = sha256_bytes(rendered)
    _atomic_write_new(path, rendered)
    if sha256_file(path) != digest:
        raise OSError(f"{path}: atomic report verification failed")
    checksum = f"{digest}  {path.name}\n".encode("ascii")
    _atomic_write_new(checksum_path, checksum)
    if checksum_path.read_bytes() != checksum:
        raise OSError(f"{checksum_path}: atomic checksum verification failed")
    return digest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock-tape", type=Path, required=True)
    parser.add_argument(
        "--rust-libm-helper",
        type=Path,
        help=(
            "hash and use the pinned Rust-libm raw-bit helper as the sole "
            "authoritative f32 transcendental evaluator"
        ),
    )
    parser.add_argument(
        "--enforce-shadow-gate",
        action="store_true",
        help="apply the frozen CPLG 32-boundary activity/direction/cost verdict",
    )
    parser.add_argument(
        "--overhead-evidence",
        type=Path,
        help="strict matched capture-OFF/ON interval and writer evidence",
    )
    parser.add_argument("--out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.enforce_shadow_gate and args.rust_libm_helper is None:
        raise ValueError("frozen shadow gate requires --rust-libm-helper")
    if args.enforce_shadow_gate and args.overhead_evidence is None:
        raise ValueError("frozen shadow gate requires --overhead-evidence")
    if args.rust_libm_helper is None:
        report = evaluate_full_vector_stock_tape(
            args.stock_tape,
            enforce_shadow_gate=args.enforce_shadow_gate,
            overhead_evidence_path=args.overhead_evidence,
        )
    else:
        with CPLGRustLibmTrig(args.rust_libm_helper) as trig:
            report = evaluate_full_vector_stock_tape(
                args.stock_tape,
                trig=trig,
                enforce_shadow_gate=args.enforce_shadow_gate,
                overhead_evidence_path=args.overhead_evidence,
            )
    if args.out is None:
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    else:
        write_checksummed_report(args.out, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
