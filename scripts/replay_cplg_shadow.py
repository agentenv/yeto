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
    CPLG_PLATFORM_CAP_ATAN2F_BITS,
    CPLG_PLATFORM_CAP_MATCHES_PINNED,
    CPLG_TRIG_BACKEND,
    CPLG_TRIG_PORTABILITY_LIMITATION,
    cosine_gain_f32le,
    cplg_angle_based_direction_f32le,
    decode_f32le,
)


LOWER_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_FULL_VECTOR_FIELDS = frozenset(
    {
        "commit_seq",
        "fragment",
        "numel",
        "stock_f32le",
        "stock_f32le_sha256",
    }
)
REPORT_SCHEMA = "cplg_full_vector_stock_shadow_v1"


@dataclass(frozen=True)
class _StockVector:
    commit_seq: int
    fragment: int
    numel: int
    raw: bytes
    sha256: str
    relative_path: str


@dataclass(frozen=True)
class _PendingShadow:
    source: _StockVector
    candidate_raw: bytes
    candidate_sha256: str


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


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
) -> tuple[list[_StockVector], list[dict[str, Any]], str]:
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
    for line_number, line in enumerate(tape_text.splitlines(), 1):
        if not line.strip():
            continue
        context = f"{tape_path}:{line_number}"
        row = _strict_json_object(line, context)
        missing = sorted(REQUIRED_FULL_VECTOR_FIELDS - row.keys())
        if missing:
            raise ValueError(
                f"{context}: full stock vectors are required for CPLG; "
                f"missing fields {missing}"
            )

        commit_seq = _exact_int(row, "commit_seq", context)
        fragment = _exact_int(row, "fragment", context)
        numel = _exact_int(row, "numel", context)
        if commit_seq != expected_commit_seq:
            raise ValueError(
                f"{context}: commit_seq must be contiguous; expected "
                f"{expected_commit_seq}, observed {commit_seq}"
            )
        expected_commit_seq += 1
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
            }
        )
    if not vectors:
        raise ValueError(f"{tape_path}: stock tape contains no records")
    return vectors, provenance, sha256_bytes(tape_raw)


def evaluate_full_vector_stock_tape(tape_path: Path) -> dict[str, Any]:
    """Construct and score causal CPLG shadows from exact factual vectors."""

    vectors, provenance, tape_sha256 = read_full_vector_stock_tape(tape_path)
    history: dict[int, list[_StockVector]] = defaultdict(list)
    pending: dict[int, _PendingShadow] = {}
    records: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    scores: list[float] = []
    scores_by_fragment: dict[int, list[float]] = defaultdict(list)
    candidates = 0

    for current in vectors:
        resolved_source_commit_seq: int | None = None
        resolved_shadow_cosine_gain: float | None = None
        prior_pending = pending.pop(current.fragment, None)
        if prior_pending is not None:
            if prior_pending.source.numel != current.numel:
                raise ValueError(
                    f"fragment {current.fragment}: pending CPLG shape changed"
                )
            resolved_source_commit_seq = prior_pending.source.commit_seq
            resolved_shadow_cosine_gain = cosine_gain_f32le(
                prior_pending.candidate_raw,
                prior_pending.source.raw,
                current.raw,
                (current.numel,),
            )
            scores.append(resolved_shadow_cosine_gain)
            scores_by_fragment[current.fragment].append(resolved_shadow_cosine_gain)

        fragment_history = history[current.fragment]
        if len(fragment_history) < 2:
            decision = None
            reason = "insufficient_same_fragment_history"
            candidate_sha256 = None
            current_turn_angle_radians = 0.0
            previous_turn_angle_radians = 0.0
            coherence = 0.0
            commanded_angle_radians = 0.0
        else:
            previous = fragment_history[-1]
            previous_previous = fragment_history[-2]
            decision = cplg_angle_based_direction_f32le(
                current.raw,
                previous.raw,
                previous_previous.raw,
                (current.numel,),
            )
            reason = decision.reason
            candidate_sha256 = sha256_bytes(decision.raw)
            current_turn_angle_radians = decision.current_turn_angle_radians
            previous_turn_angle_radians = decision.previous_turn_angle_radians
            coherence = decision.transported_forward_tangent_coherence
            commanded_angle_radians = decision.commanded_angle_radians
            if decision.used_nonstock:
                candidates += 1
                pending[current.fragment] = _PendingShadow(
                    source=current,
                    candidate_raw=decision.raw,
                    candidate_sha256=candidate_sha256,
                )

        reason_counts[reason] += 1
        records.append(
            {
                "commit_seq": current.commit_seq,
                "fragment": current.fragment,
                "stock_sha256": current.sha256,
                "used_nonstock_shadow": bool(
                    decision is not None and decision.used_nonstock
                ),
                "reason": reason,
                "candidate_sha256": candidate_sha256,
                "current_turn_angle_radians": current_turn_angle_radians,
                "previous_turn_angle_radians": previous_turn_angle_radians,
                "transported_forward_tangent_coherence": coherence,
                "commanded_angle_radians": commanded_angle_radians,
                "resolved_source_commit_seq": resolved_source_commit_seq,
                "resolved_shadow_cosine_gain": resolved_shadow_cosine_gain,
            }
        )

        fragment_history.append(current)
        del fragment_history[:-2]

    per_fragment = {}
    for fragment in sorted(history):
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
            "unresolved_tail_shadow": fragment in pending,
        }

    return {
        "schema": REPORT_SCHEMA,
        "decision": "DIRECTION_SHADOW_ONLY",
        "identifiable": True,
        "full_vectors_verified": True,
        "causal_finite_loss_claim": False,
        "beats_sgd_0_28_claim": False,
        "input_stock_tape_sha256": tape_sha256,
        "input_provenance": provenance,
        "reference_contract": {
            "kernel": "angle_based_cplg_sequential_f32_no_fma",
            "angle_cap_f32_bits": f"0x{CPLG_ANGLE_CAP_F32_BITS:08x}",
            "forward_tangent_sign": "rho_times_current_minus_previous",
            "backward_tangent_sign_not_used": "previous_minus_rho_times_current",
            "command_is_angle_radians_not_tangent_ratio": True,
            "trig_backend": CPLG_TRIG_BACKEND,
            "trig_portability_limitation": CPLG_TRIG_PORTABILITY_LIMITATION,
            "platform_cap_atan2f_bits": (f"0x{CPLG_PLATFORM_CAP_ATAN2F_BITS:08x}"),
            "platform_cap_matches_pinned": CPLG_PLATFORM_CAP_MATCHES_PINNED,
            "cross_runtime_bit_parity": CPLG_CROSS_RUNTIME_BIT_PARITY,
        },
        "summary": {
            "records": len(vectors),
            "constructed_nonstock_shadows": candidates,
            "resolved_shadows": len(scores),
            "positive_resolved_shadows": sum(score > 0.0 for score in scores),
            "mean_shadow_cosine_gain": (
                math.fsum(scores) / len(scores) if scores else None
            ),
            "unresolved_tail_shadows": len(pending),
            "reason_counts": dict(sorted(reason_counts.items())),
        },
        "per_fragment": per_fragment,
        "records": records,
        "limitations": [
            "historical stock directions are off-policy after the first hypothetical CPLG action",
            "directional cosine gain is not a finite-loss or convergence guarantee",
            "host C libm f32 transcendental last bits are not cross-platform pinned",
        ],
    }


def _atomic_write(path: Path, raw: bytes) -> None:
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
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_checksummed_report(path: Path, report: dict[str, Any]) -> str:
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
    _atomic_write(path, rendered)
    if sha256_file(path) != digest:
        raise OSError(f"{path}: atomic report verification failed")
    checksum_path = Path(f"{path}.sha256")
    checksum = f"{digest}  {path.name}\n".encode("ascii")
    _atomic_write(checksum_path, checksum)
    if checksum_path.read_bytes() != checksum:
        raise OSError(f"{checksum_path}: atomic checksum verification failed")
    return digest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock-tape", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = evaluate_full_vector_stock_tape(args.stock_tape)
    if args.out is None:
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    else:
        write_checksummed_report(args.out, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
