#!/usr/bin/env python3
"""CPU-only, fail-closed validator for the frozen active CPLG E1 pair.

The program reads one immutable acquisition manifest, verifies its complete
local round-trip inventory, validates the closed scientific evidence, and
publishes a checksummed analysis report and terminal verdict in a distinct
analysis directory.  It deliberately contains no cloud, SSH, torch, or CUDA
integration.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import tempfile
from typing import Any


RUN_ID = "exp2-cplg-active-e1-m1-r1"
RUN_CONFIG_SHA256 = "5afe2d4900051fda1ac99cc682c489dfeae85f0eb34d1816646b5bff5f0c26df"
STOCK_ARM = "cplg_m1_stock"
CANDIDATE_ARM = "cplg_m1_candidate"
BASE_ARM = "base (untrained)"
EXPECTED_COMMITS = 32
EXPECTED_FRAGMENTS = 4
EXPECTED_FRAGMENT_ORDER = [0, 1, 2, 3] * 8
EXPECTED_TERMINAL_LOCAL_STEPS = 34
EXPECTED_RAW_TOKENS = 4_352
EXPECTED_H = 4
EXPECTED_SEQUENCE_LENGTH = 128
MINIMUM_ACTIONS = 8
MINIMUM_ACTIVE_FRAGMENTS = 3
MAXIMUM_LOSS_REGRESSION = 0.05
MAXIMUM_OVERHEAD = Fraction(1, 50)

MANIFEST_SCHEMA = "yeto.cplg-sgd.online-e1-acquisition-manifest.v1"
ACQUISITION_TERMINAL_SCHEMA = "yeto.cplg-sgd.online-e1-acquisition-terminal.v1"
ANALYSIS_SCHEMA = "yeto.cplg-sgd.online-e1-analysis.v1"
VERDICT_SCHEMA = "yeto.cplg-sgd.online-e1-terminal-verdict.v1"
LEARNER_COMPLETION_SCHEMA = "yeto.learner_completion.v1"
VERDICTS = {
    "PASS",
    "FAIL",
    "INCONCLUSIVE",
    "UNIDENTIFIABLE",
    "INFRA_FAILURE",
}

SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
SIDECAR_RE = re.compile(r"([0-9a-f]{64})  ([^/\n]+)\n\Z")
CKPT_MAGIC = 0xD170_5A7E
CPLG_CKPT_EXTENSION_MAGIC = 0x314C_5043

RESULT_FIELDS = {"arm", "m", "wall_s", "eval_loss"}
RESPONDER_FIELDS = {"id", "base_version", "c_steps", "c_tokens", "weight"}
TAPE_FIELDS = {
    "step",
    "fragment",
    "commit_seq",
    "commit_elapsed_ns",
    "gnorm",
    "ms",
    "responders",
    "outer_step_norm",
    "outer_direction_cosine",
    "outer_history_current_ratio",
    "outer_restarted",
    "pti_shadow_score",
    "pti_score_count",
    "pti_interlock_open",
    "pti_used_nonstock",
    "pti_state_cleared",
    "pti_reason",
    "pti_stock_sha256",
    "pti_previous_stock_sha256",
    "pti_candidate_sha256",
    "pti_action_sha256",
    "policy",
    "selected_action",
    "committed_action",
    "selected_multiplier",
    "committed_multiplier",
    "fallback",
    "fallback_reason",
    "probe_latency_ms",
    "selected_mass",
    "norm_scale",
    "step_ratio",
    "request_digest",
}
CPLG_FIELDS = {
    "cplg_rho",
    "cplg_theta",
    "cplg_previous_theta",
    "cplg_coherence",
    "cplg_phi",
    "cplg_shadow_score",
    "cplg_score_count",
    "cplg_interlock_open",
    "cplg_used_nonstock",
    "cplg_state_cleared",
    "cplg_reason",
    "cplg_stock_sha256",
    "cplg_previous_stock_sha256",
    "cplg_previous_tangent_sha256",
    "cplg_transported_tangent_sha256",
    "cplg_candidate_sha256",
    "cplg_action_sha256",
}
LEDGER_FIELDS = {
    "schema_version",
    "row_index",
    "run_id",
    "run_config_sha256",
    "source_commit",
    "commit_sequence",
    "fragment",
    "fragment_version",
    "responder_step",
    "responder_tokens",
    "weight_identity_sha256",
    "layout_sha256",
    "initial_state_sha256",
    *CPLG_FIELDS,
    "previous_row_sha256",
    "row_sha256",
}
REASONS = {
    "not_active",
    "stock_warmup",
    "phase_warmup",
    "interlock_closed",
    "candidate_selected",
    "degenerate_stock",
    "nonacute_turn",
    "invalid_geometry",
    "invalid_shadow_score",
    "zero_or_rounded_phase",
}
CLEARING_REASONS = {
    "degenerate_stock",
    "nonacute_turn",
    "invalid_geometry",
    "invalid_shadow_score",
}

INITIAL_FIELDS = {
    "schema_version",
    "run_id",
    "run_config_sha256",
    "source_commit",
    "arm",
    "outer_optimizer",
    "layout_sha256",
    "initial_state_sha256",
    "fragments",
    "expected_commits",
}
LEARNER_COMPLETION_FIELDS = {
    "schema",
    "learner_id",
    "local_step",
    "raw_tokens",
    "global_step",
    "reconnect_count",
    "terminal_status",
}
LEARNER_WORK_FIELDS = (
    "learner_id",
    "local_step",
    "raw_tokens",
    "global_step",
    "reconnect_count",
    "terminal_status",
)
COMPLETION_FIELDS = {
    "schema_version",
    "run_id",
    "arm",
    "terminal_local_steps",
    "raw_training_tokens",
    "final_global_step",
    "commits_observed",
    "commits_per_fragment",
    "interval_start_ns",
    "interval_end_ns",
    "interval_ns",
    "event_tape_sha256",
    "final_checkpoint_sha256",
    "ledger_head",
    "ledger_rows",
    "writer_dropped",
    "writer_abandoned",
    "writer_pending",
    "writer_errors",
}
LEDGER_MANIFEST_FIELDS = {
    "schema_version",
    "run_id",
    "run_config_sha256",
    "source_commit",
    "arm",
    "layout_sha256",
    "initial_state_sha256",
    "ledger_rows",
    "ledger_head",
    "final_checkpoint_sha256",
    "event_tape_sha256",
    "expected_commits",
    "fragments",
    "outer_optimizer",
    "unresolved_tail",
    "writer_dropped",
    "writer_abandoned",
    "writer_pending",
    "writer_errors",
}


class ValidationError(RuntimeError):
    """Acquisition evidence violates the frozen scientific contract."""


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValidationError(f"non-finite JSON constant {value!r}")


def _validate_finite_tree(value: Any, *, source: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValidationError(f"{source}: non-finite JSON number")
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite_tree(item, source=f"{source}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite_tree(item, source=f"{source}[{index}]")


def _regular_file(path: Path, *, label: str, nonempty: bool = True) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValidationError(f"missing {label}: {path}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValidationError(f"{label} is not a regular non-symlink file: {path}")
    if nonempty and metadata.st_size == 0:
        raise ValidationError(f"{label} is empty: {path}")
    return metadata


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ValidationError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    _regular_file(path, label=label)
    try:
        raw = path.read_text(encoding="utf-8")
        value = json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"malformed {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"{label} must be one JSON object")
    _validate_finite_tree(value, source=label)
    return value


def _read_jsonl(path: Path, *, label: str) -> tuple[list[dict[str, Any]], list[str]]:
    _regular_file(path, label=label)
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValidationError(f"cannot read {label} {path}: {exc}") from exc
    if not raw.endswith(b"\n"):
        raise ValidationError(f"{label} must end with one newline")
    lines = text.splitlines()
    if not lines or any(not line for line in lines):
        raise ValidationError(f"{label} contains an empty row")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        try:
            value = json.loads(
                line,
                object_pairs_hook=_strict_object,
                parse_constant=_reject_constant,
            )
        except json.JSONDecodeError as exc:
            raise ValidationError(
                f"malformed {label} JSON at line {line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ValidationError(f"{label} line {line_number} is not an object")
        _validate_finite_tree(value, source=f"{label}[{line_number}]")
        rows.append(value)
    return rows, lines


def _require_fields(value: dict[str, Any], expected: set[str], *, label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValidationError(
            f"{label} has a non-closed schema; missing={missing}, extra={extra}"
        )


def _integer(value: Any, *, label: str, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ValidationError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValidationError(f"{label} must be at least {minimum}")
    return value


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(f"{label} must be a finite number")
    return number


def _binary64(value: Any, *, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValidationError(f"{label} must be a finite binary64 JSON number")
    return value


def _f32_or_none(value: Any, *, label: str) -> float | None:
    if value is None:
        return None
    number = _finite_number(value, label=label)
    try:
        round_trip = struct.unpack("<f", struct.pack("<f", number))[0]
    except (OverflowError, struct.error) as exc:
        raise ValidationError(f"{label} is outside finite f32") from exc
    if not math.isfinite(round_trip) or float(round_trip) != number:
        raise ValidationError(f"{label} is not an exact f32-round-tripped value")
    return number


def _sha256(value: Any, *, label: str, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        qualifier = " or null" if optional else ""
        raise ValidationError(f"{label} must be 64 lowercase hex characters{qualifier}")
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _verify_sidecar(target: Path, *, label: str) -> str:
    digest = _sha256_file(target)
    sidecar = target.with_name(target.name + ".sha256")
    _regular_file(sidecar, label=f"{label} checksum sidecar")
    try:
        raw = sidecar.read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise ValidationError(f"cannot read {label} checksum sidecar: {exc}") from exc
    match = SIDECAR_RE.fullmatch(raw)
    if match is None:
        raise ValidationError(f"{label} checksum sidecar is not basename-bound")
    declared, basename = match.groups()
    if basename != target.name:
        raise ValidationError(
            f"{label} checksum sidecar names {basename!r}, expected {target.name!r}"
        )
    if declared != digest:
        raise ValidationError(f"{label} checksum sidecar digest mismatch")
    return digest


def _manifest_relative_path(value: Any, *, index: int) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"manifest file {index} path must be a nonempty string")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or value != pure.as_posix()
        or "\\" in value
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValidationError(f"manifest file {index} has unsafe path {value!r}")
    return value


def _walk_inventory(root: Path) -> set[str]:
    inventory: set[str] = set()
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in list(dirnames):
            path = directory_path / name
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ValidationError(f"acquisition contains symlink directory: {path}")
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValidationError(
                    f"acquisition contains non-directory entry: {path}"
                )
        for name in filenames:
            path = directory_path / name
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode):
                raise ValidationError(f"acquisition contains non-regular entry: {path}")
            inventory.add(path.relative_to(root).as_posix())
    return inventory


def _validate_manifest(
    manifest_path: Path,
) -> tuple[Path, dict[str, Any], dict[str, Path], str]:
    manifest_path = manifest_path.absolute()
    if (
        manifest_path.name != "acquisition_manifest.json"
        or manifest_path.parent.name != "report"
    ):
        raise ValidationError(
            "acquisition manifest must be named report/acquisition_manifest.json"
        )
    _regular_file(manifest_path, label="acquisition manifest")
    manifest_digest = _verify_sidecar(manifest_path, label="acquisition manifest")
    root = manifest_path.parent.parent
    if root.is_symlink():
        raise ValidationError("acquisition root may not be a symlink")
    manifest = _load_json(manifest_path, label="acquisition manifest")
    _require_fields(
        manifest,
        {
            "schema",
            "status",
            "run_id",
            "source_commit",
            "run_config_sha256",
            "arms",
            "files",
            "file_count",
            "total_bytes",
        },
        label="acquisition manifest",
    )
    if manifest["schema"] != MANIFEST_SCHEMA:
        raise ValidationError("unknown acquisition manifest schema")
    if manifest["status"] != "ACQUIRED":
        raise ValidationError(
            f"unknown acquisition manifest status {manifest['status']!r}"
        )
    if manifest["run_id"] != RUN_ID:
        raise ValidationError(f"acquisition run_id must be {RUN_ID!r}")
    source_commit = manifest["source_commit"]
    if not isinstance(source_commit, str) or COMMIT_RE.fullmatch(source_commit) is None:
        raise ValidationError(
            "acquisition source_commit must be 40 lowercase hex characters"
        )
    if manifest["run_config_sha256"] != RUN_CONFIG_SHA256:
        raise ValidationError(
            "acquisition run-config SHA-256 is not the frozen E1 digest"
        )
    if manifest["arms"] != {"stock": STOCK_ARM, "candidate": CANDIDATE_ARM}:
        raise ValidationError("acquisition arms do not match the frozen pair")
    files = manifest["files"]
    if not isinstance(files, list) or not files:
        raise ValidationError("acquisition manifest files must be a nonempty list")
    declared: dict[str, Path] = {}
    total_bytes = 0
    declared_order: list[str] = []
    for index, entry in enumerate(files):
        if not isinstance(entry, dict):
            raise ValidationError(f"manifest file {index} is not an object")
        _require_fields(
            entry, {"path", "bytes", "sha256"}, label=f"manifest file {index}"
        )
        relative = _manifest_relative_path(entry["path"], index=index)
        if relative in declared:
            raise ValidationError(f"manifest contains duplicate path {relative!r}")
        size = _integer(entry["bytes"], label=f"manifest file {index} bytes", minimum=0)
        digest = _sha256(entry["sha256"], label=f"manifest file {index} sha256")
        path = root.joinpath(*PurePosixPath(relative).parts)
        metadata = _regular_file(
            path, label=f"manifest object {relative}", nonempty=False
        )
        try:
            path.resolve(strict=True).relative_to(root.resolve(strict=True))
        except (OSError, ValueError) as exc:
            raise ValidationError(
                f"manifest object escapes acquisition root: {relative}"
            ) from exc
        if metadata.st_size != size:
            raise ValidationError(
                f"manifest object {relative} size {metadata.st_size} != declared {size}"
            )
        if _sha256_file(path) != digest:
            raise ValidationError(f"manifest object {relative} SHA-256 mismatch")
        declared[relative] = path
        declared_order.append(relative)
        total_bytes += size
    producer_order = sorted(declared_order, key=PurePosixPath)
    if declared_order != producer_order:
        raise ValidationError("manifest file paths are not in producer Path order")
    if manifest["file_count"] != len(files):
        raise ValidationError("acquisition manifest file_count mismatch")
    if manifest["total_bytes"] != total_bytes:
        raise ValidationError("acquisition manifest total_bytes mismatch")

    terminal = manifest_path.with_name("acquisition_terminal.json")
    reserved = {
        "report/acquisition_manifest.json",
        "report/acquisition_manifest.json.sha256",
        "report/acquisition_terminal.json",
        "report/acquisition_terminal.json.sha256",
    }
    if reserved & set(declared):
        raise ValidationError(
            "manifest files must exclude manifest and terminal receipts"
        )
    actual = _walk_inventory(root)
    expected = set(declared) | reserved
    if actual != expected:
        missing = sorted(expected - actual)
        stale = sorted(actual - expected)
        raise ValidationError(
            f"acquisition inventory is not closed; missing={missing}, stale={stale}"
        )

    terminal_digest = _verify_sidecar(terminal, label="acquisition terminal")
    terminal_value = _load_json(terminal, label="acquisition terminal")
    _require_fields(
        terminal_value,
        {
            "schema",
            "status",
            "run_id",
            "source_commit",
            "run_config_sha256",
            "acquisition_manifest",
            "acquisition_manifest_sha256",
            "file_count",
            "total_bytes",
            "gpu_analysis_performed",
            "scientific_verdict",
            "next_action",
        },
        label="acquisition terminal",
    )
    if terminal_value["schema"] != ACQUISITION_TERMINAL_SCHEMA:
        raise ValidationError("unknown acquisition terminal schema")
    if terminal_value["status"] != "GPU_ACQUISITION_COMPLETE":
        raise ValidationError(
            f"unknown acquisition status {terminal_value['status']!r}"
        )
    expected_terminal = {
        "run_id": RUN_ID,
        "source_commit": source_commit,
        "run_config_sha256": RUN_CONFIG_SHA256,
        "acquisition_manifest": manifest_path.name,
        "acquisition_manifest_sha256": manifest_digest,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "gpu_analysis_performed": False,
        "scientific_verdict": None,
        "next_action": "round_trip_verify_delete_gpu_then_cpu_validate",
    }
    for key, wanted in expected_terminal.items():
        if terminal_value[key] != wanted:
            raise ValidationError(
                f"acquisition terminal {key} must be {wanted!r}, got {terminal_value[key]!r}"
            )
    if terminal_digest in {manifest_digest, "0" * 64}:  # defensive independence
        raise ValidationError("acquisition terminal has an invalid receipt digest")
    return root, manifest, declared, manifest_digest


def _required_path(files: dict[str, Path], relative: str) -> Path:
    try:
        return files[relative]
    except KeyError as exc:
        raise ValidationError(
            f"acquisition manifest is missing required object {relative}"
        ) from exc


def _validate_required_inventory(files: dict[str, Path]) -> None:
    required = {
        "report/results.jsonl",
        "report/report.md",
        "report/frozen_run_config.json",
        "work/train.jsonl",
        "work/eval.jsonl",
    }
    for arm in (STOCK_ARM, CANDIDATE_ARM):
        base = f"work/{arm}"
        required.update(
            {
                f"{base}/tape.jsonl",
                f"{base}/state.ckpt",
                f"{base}/syncer.log",
                f"{base}/learner-0.log",
                f"{base}/learner-0/learner_completion.json",
                f"{base}/learner-0/learner_completion.json.sha256",
                f"{base}/cplg_online_initial_state.json",
                f"{base}/cplg_online_initial_state.json.sha256",
                f"{base}/cplg_online_completion.json",
                f"{base}/cplg_online_completion.json.sha256",
                f"{base}/export/adapter_config.json",
            }
        )
        model_paths = {
            f"{base}/export/adapter_model.safetensors",
            f"{base}/export/adapter_model.bin",
        } & set(files)
        if len(model_paths) != 1:
            raise ValidationError(
                f"{arm} must publish exactly one adapter model payload"
            )
        required.update(model_paths)
    candidate = f"work/{CANDIDATE_ARM}"
    required.update(
        {
            f"{candidate}/cplg_action_ledger.jsonl",
            f"{candidate}/cplg_action_ledger_manifest.json",
            f"{candidate}/cplg_action_ledger_manifest.json.sha256",
        }
    )
    missing = sorted(required - set(files))
    if missing:
        raise ValidationError(f"acquisition manifest misses required files: {missing}")
    frozen_config = _required_path(files, "report/frozen_run_config.json")
    if _sha256_file(frozen_config) != RUN_CONFIG_SHA256:
        raise ValidationError("frozen_run_config.json does not have the frozen digest")
    for relative in sorted(required):
        if relative.endswith(".sha256"):
            target_relative = relative.removesuffix(".sha256")
            _verify_sidecar(files[target_relative], label=target_relative)


def _validate_results(path: Path) -> tuple[list[dict[str, Any]], float, float, float]:
    rows, _ = _read_jsonl(path, label="results.jsonl")
    expected_arms = [BASE_ARM, STOCK_ARM, CANDIDATE_ARM]
    actual_arms = [row.get("arm") for row in rows]
    if actual_arms != expected_arms:
        raise ValidationError(
            f"results.jsonl arm order must be {expected_arms!r}, got {actual_arms!r}"
        )
    for index, (row, arm) in enumerate(zip(rows, expected_arms, strict=True)):
        _require_fields(row, RESULT_FIELDS, label=f"results row {index}")
        expected_m = 0 if index == 0 else 1
        if row["m"] != expected_m or type(row["m"]) is not int:
            raise ValidationError(f"results row {arm} m must be {expected_m}")
        wall = _binary64(row["wall_s"], label=f"results {arm}.wall_s")
        if wall < 0.0 or (index == 0 and wall != 0.0):
            raise ValidationError(f"results {arm}.wall_s is invalid")
        _binary64(row["eval_loss"], label=f"results {arm}.eval_loss")
    base_loss = rows[0]["eval_loss"]
    stock_loss = rows[1]["eval_loss"]
    candidate_loss = rows[2]["eval_loss"]
    return rows, base_loss, stock_loss, candidate_loss


def _validate_data_rows(files: dict[str, Path]) -> dict[str, int]:
    train_rows, _ = _read_jsonl(files["work/train.jsonl"], label="train rows")
    eval_rows, _ = _read_jsonl(files["work/eval.jsonl"], label="evaluation rows")
    if not train_rows or len(train_rows) > 5_000:
        raise ValidationError("train rows must contain between 1 and 5000 objects")
    if len(eval_rows) != 8:
        raise ValidationError(
            f"evaluation rows must contain exactly 8 objects, got {len(eval_rows)}"
        )
    return {"train_rows": len(train_rows), "evaluation_rows": len(eval_rows)}


def _validate_tape_scalar_defaults(row: dict[str, Any], *, label: str) -> None:
    for field in ("gnorm", "outer_step_norm"):
        if _finite_number(row[field], label=f"{label}.{field}") < 0.0:
            raise ValidationError(f"{label}.{field} must be nonnegative")
    for field in ("outer_direction_cosine", "outer_history_current_ratio"):
        if row[field] is not None:
            _finite_number(row[field], label=f"{label}.{field}")
    if type(row["outer_restarted"]) is not bool:
        raise ValidationError(f"{label}.outer_restarted must be boolean")
    expected_defaults: dict[str, Any] = {
        "pti_shadow_score": None,
        "pti_score_count": 0,
        "pti_interlock_open": False,
        "pti_used_nonstock": False,
        "pti_state_cleared": False,
        "pti_reason": None,
        "pti_stock_sha256": None,
        "pti_previous_stock_sha256": None,
        "pti_candidate_sha256": None,
        "pti_action_sha256": None,
        "policy": "token_weighted",
        "selected_action": "A0",
        "committed_action": "A0",
        "selected_multiplier": 1,
        "committed_multiplier": 1,
        "fallback": False,
        "fallback_reason": None,
        "probe_latency_ms": None,
        "selected_mass": 1,
        "norm_scale": 1,
        "step_ratio": 1,
        "request_digest": None,
    }
    for field, wanted in expected_defaults.items():
        if row[field] != wanted or (
            isinstance(wanted, bool) and type(row[field]) is not bool
        ):
            raise ValidationError(
                f"{label}.{field} must be {wanted!r}, got {row[field]!r}"
            )


def _validate_tape(
    path: Path, *, arm: str, candidate: bool
) -> tuple[list[dict[str, Any]], list[tuple[int, int, int, int, float]]]:
    rows, _ = _read_jsonl(path, label=f"{arm} event tape")
    if len(rows) != EXPECTED_COMMITS:
        raise ValidationError(
            f"{arm} event tape must contain exactly {EXPECTED_COMMITS} rows"
        )
    schedule: list[tuple[int, int, int, int, float]] = []
    previous_elapsed = -1
    for index, row in enumerate(rows):
        label = f"{arm} tape row {index}"
        _require_fields(
            row,
            TAPE_FIELDS | (CPLG_FIELDS if candidate else set()),
            label=label,
        )
        expected_sequence = index + 1
        if row["step"] != expected_sequence or type(row["step"]) is not int:
            raise ValidationError(f"{label}.step schedule drift")
        if row["commit_seq"] != expected_sequence or type(row["commit_seq"]) is not int:
            raise ValidationError(f"{label}.commit_seq schedule drift")
        expected_fragment = EXPECTED_FRAGMENT_ORDER[index]
        if row["fragment"] != expected_fragment or type(row["fragment"]) is not int:
            raise ValidationError(f"{label}.fragment schedule drift")
        elapsed = _integer(
            row["commit_elapsed_ns"], label=f"{label}.commit_elapsed_ns", minimum=0
        )
        if elapsed <= previous_elapsed:
            raise ValidationError(f"{arm} commit_elapsed_ns must increase strictly")
        previous_elapsed = elapsed
        _integer(row["ms"], label=f"{label}.ms", minimum=0)
        responders = row["responders"]
        if not isinstance(responders, list) or len(responders) != 1:
            raise ValidationError(f"{label} must have exactly learner 0 as responder")
        responder = responders[0]
        if not isinstance(responder, dict):
            raise ValidationError(f"{label} responder must be an object")
        _require_fields(responder, RESPONDER_FIELDS, label=f"{label} responder")
        expected_version = index // EXPECTED_FRAGMENTS
        if responder["id"] != 0 or type(responder["id"]) is not int:
            raise ValidationError(f"{label} responder ID drift")
        if (
            responder["base_version"] != expected_version
            or type(responder["base_version"]) is not int
        ):
            raise ValidationError(f"{label} responder base_version drift")
        c_steps = _integer(responder["c_steps"], label=f"{label}.c_steps", minimum=1)
        c_tokens = _integer(responder["c_tokens"], label=f"{label}.c_tokens", minimum=1)
        if c_steps != EXPECTED_H or c_tokens != EXPECTED_H * EXPECTED_SEQUENCE_LENGTH:
            raise ValidationError(
                f"{label} work drift: expected c_steps=4 and c_tokens=512"
            )
        weight = _finite_number(responder["weight"], label=f"{label}.weight")
        expected_weight = float(c_tokens) ** 2 / c_steps
        if weight != expected_weight:
            raise ValidationError(
                f"{label} responder weight does not equal c_tokens^2/c_steps"
            )
        _validate_tape_scalar_defaults(row, label=label)
        if candidate:
            _validate_cplg_field_types(row, label=label)
        schedule.append(
            (expected_sequence, expected_fragment, c_steps, c_tokens, weight)
        )
    return rows, schedule


def _validate_cplg_field_types(row: dict[str, Any], *, label: str) -> None:
    for field in (
        "cplg_rho",
        "cplg_theta",
        "cplg_previous_theta",
        "cplg_coherence",
        "cplg_phi",
        "cplg_shadow_score",
    ):
        _f32_or_none(row[field], label=f"{label}.{field}")
    count = _integer(
        row["cplg_score_count"], label=f"{label}.cplg_score_count", minimum=0
    )
    if count > 3:
        raise ValidationError(f"{label}.cplg_score_count exceeds three")
    for field in ("cplg_interlock_open", "cplg_used_nonstock", "cplg_state_cleared"):
        if type(row[field]) is not bool:
            raise ValidationError(f"{label}.{field} must be boolean")
    reason = row["cplg_reason"]
    if reason not in REASONS:
        raise ValidationError(f"{label} has unknown reason {reason!r}")
    for field in (
        "cplg_stock_sha256",
        "cplg_previous_stock_sha256",
        "cplg_previous_tangent_sha256",
        "cplg_transported_tangent_sha256",
        "cplg_candidate_sha256",
        "cplg_action_sha256",
    ):
        optional = field not in {"cplg_stock_sha256", "cplg_action_sha256"}
        _sha256(row[field], label=f"{label}.{field}", optional=optional)


def _validate_initial(
    path: Path,
    *,
    arm: str,
    optimizer: str,
    source_commit: str,
) -> dict[str, Any]:
    _verify_sidecar(path, label=f"{arm} initial-state receipt")
    value = _load_json(path, label=f"{arm} initial-state receipt")
    _require_fields(value, INITIAL_FIELDS, label=f"{arm} initial-state receipt")
    expected = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "run_config_sha256": RUN_CONFIG_SHA256,
        "source_commit": source_commit,
        "arm": arm,
        "outer_optimizer": optimizer,
        "fragments": EXPECTED_FRAGMENTS,
        "expected_commits": EXPECTED_COMMITS,
    }
    for field, wanted in expected.items():
        if value[field] != wanted or (
            isinstance(wanted, int) and type(value[field]) is not int
        ):
            raise ValidationError(f"{arm} initial-state {field} mismatch")
    _sha256(value["layout_sha256"], label=f"{arm} layout_sha256")
    _sha256(value["initial_state_sha256"], label=f"{arm} initial_state_sha256")
    return value


def _validate_learner_completion(path: Path, *, arm: str) -> dict[str, Any]:
    _verify_sidecar(path, label=f"{arm} learner completion receipt")
    value = _load_json(path, label=f"{arm} learner completion receipt")
    _require_fields(
        value,
        LEARNER_COMPLETION_FIELDS,
        label=f"{arm} learner completion receipt",
    )
    if value["schema"] != LEARNER_COMPLETION_SCHEMA:
        raise ValidationError(f"{arm} learner completion schema mismatch")
    for field in (
        "learner_id",
        "local_step",
        "raw_tokens",
        "global_step",
        "reconnect_count",
    ):
        _integer(
            value[field],
            label=f"{arm} learner completion {field}",
            minimum=0,
        )
    if not isinstance(value["terminal_status"], str):
        raise ValidationError(
            f"{arm} learner completion terminal_status must be a string"
        )
    return value


def _validate_matched_learner_work(
    stock: dict[str, Any], candidate: dict[str, Any]
) -> None:
    for field in LEARNER_WORK_FIELDS:
        if stock[field] != candidate[field]:
            raise ValidationError(f"arms have unequal learner observed {field}")


def _validate_observed_learner_work(value: dict[str, Any], *, arm: str) -> None:
    expected = {
        "learner_id": 0,
        "local_step": EXPECTED_TERMINAL_LOCAL_STEPS,
        "raw_tokens": EXPECTED_RAW_TOKENS,
        "global_step": EXPECTED_COMMITS,
        "reconnect_count": 0,
        "terminal_status": "syncer_shutdown",
    }
    for field, wanted in expected.items():
        if value[field] != wanted:
            raise ValidationError(
                f"{arm} learner observed {field} must be {wanted!r}, "
                f"got {value[field]!r}"
            )


def _validate_completion(
    path: Path,
    *,
    arm: str,
    candidate: bool,
    learner_completion: dict[str, Any],
) -> dict[str, Any]:
    _verify_sidecar(path, label=f"{arm} completion receipt")
    value = _load_json(path, label=f"{arm} completion receipt")
    _require_fields(value, COMPLETION_FIELDS, label=f"{arm} completion receipt")
    expected = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "arm": arm,
        "final_global_step": EXPECTED_COMMITS,
        "commits_observed": EXPECTED_COMMITS,
        "commits_per_fragment": [8, 8, 8, 8],
    }
    for field, wanted in expected.items():
        if value[field] != wanted:
            raise ValidationError(f"{arm} completion {field} mismatch")
    for field in (
        "schema_version",
        "terminal_local_steps",
        "raw_training_tokens",
        "final_global_step",
        "commits_observed",
    ):
        _integer(value[field], label=f"{arm} completion {field}", minimum=0)
    work_cross_checks = {
        "terminal_local_steps": "local_step",
        "raw_training_tokens": "raw_tokens",
    }
    for completion_field, learner_field in work_cross_checks.items():
        if value[completion_field] != learner_completion[learner_field]:
            raise ValidationError(
                f"{arm} completion {completion_field} does not equal "
                f"observed learner {learner_field}"
            )
    if any(type(item) is not int for item in value["commits_per_fragment"]):
        raise ValidationError(f"{arm} completion commits_per_fragment must be integers")
    start = _integer(
        value["interval_start_ns"], label=f"{arm} interval_start_ns", minimum=0
    )
    end = _integer(value["interval_end_ns"], label=f"{arm} interval_end_ns", minimum=0)
    interval = _integer(value["interval_ns"], label=f"{arm} interval_ns", minimum=1)
    if end <= start or end - start != interval:
        raise ValidationError(f"{arm} completion monotonic interval is inconsistent")
    _sha256(value["event_tape_sha256"], label=f"{arm} event_tape_sha256")
    _sha256(value["final_checkpoint_sha256"], label=f"{arm} final_checkpoint_sha256")
    if candidate:
        _sha256(value["ledger_head"], label=f"{arm} ledger_head")
        if (
            value["ledger_rows"] != EXPECTED_COMMITS
            or type(value["ledger_rows"]) is not int
        ):
            raise ValidationError(f"{arm} ledger_rows must be {EXPECTED_COMMITS}")
    elif value["ledger_head"] is not None or value["ledger_rows"] is not None:
        raise ValidationError(
            "stock completion ledger_head and ledger_rows must be null"
        )
    for field in (
        "writer_dropped",
        "writer_abandoned",
        "writer_pending",
        "writer_errors",
    ):
        if value[field] != 0 or type(value[field]) is not int:
            raise ValidationError(f"{arm} completion {field} must be zero")
    return value


class _CheckpointReader:
    def __init__(self, raw: bytes) -> None:
        self.raw = raw
        self.offset = 0

    def take(self, size: int, *, label: str) -> bytes:
        if size < 0 or self.offset + size > len(self.raw):
            raise ValidationError(f"candidate checkpoint is truncated at {label}")
        result = self.raw[self.offset : self.offset + size]
        self.offset += size
        return result

    def unpack(self, fmt: str, *, label: str) -> tuple[Any, ...]:
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.take(size, label=label))

    def u8(self, *, label: str) -> int:
        return self.unpack("<B", label=label)[0]

    def u32(self, *, label: str) -> int:
        return self.unpack("<I", label=label)[0]

    def u64(self, *, label: str) -> int:
        return self.unpack("<Q", label=label)[0]


def _validate_f32_bytes(raw: bytes, *, label: str) -> None:
    for (value,) in struct.iter_unpack("<f", raw):
        if not math.isfinite(value):
            raise ValidationError(f"candidate checkpoint has non-finite {label}")


def _checkpoint_header(path: Path, *, label: str) -> tuple[bytes, int, int]:
    _regular_file(path, label=label)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValidationError(f"cannot read {label}: {exc}") from exc
    if len(raw) < 16:
        raise ValidationError(f"{label} is truncated before its header")
    magic, global_step, fragments = struct.unpack("<IQI", raw[:16])
    if magic != CKPT_MAGIC:
        raise ValidationError(f"{label} has invalid syncer checkpoint magic")
    if global_step != EXPECTED_COMMITS:
        raise ValidationError(f"{label} global_step must be {EXPECTED_COMMITS}")
    if fragments != EXPECTED_FRAGMENTS:
        raise ValidationError(f"{label} fragment count must be {EXPECTED_FRAGMENTS}")
    return raw, global_step, fragments


def _candidate_checkpoint(
    path: Path, *, expected_head: str, expected_rows: int
) -> dict[str, Any]:
    raw, global_step, fragments = _checkpoint_header(
        path, label="candidate final checkpoint"
    )
    reader = _CheckpointReader(raw)
    reader.take(16, label="header")
    numels: list[int] = []
    versions: list[int] = []
    for fragment in range(fragments):
        version = reader.u64(label=f"fragment {fragment} version")
        numel = reader.u64(label=f"fragment {fragment} numel")
        if version != 8 or numel == 0:
            raise ValidationError(
                f"candidate checkpoint fragment {fragment} version/work drift"
            )
        vector_bytes = reader.take(
            numel * 8, label=f"fragment {fragment} params/momentum"
        )
        _validate_f32_bytes(vector_bytes, label=f"fragment {fragment} params/momentum")
        versions.append(version)
        numels.append(numel)
    learner_count = reader.u32(label="learner ledger count")
    if learner_count != 1:
        raise ValidationError(
            "candidate checkpoint must contain exactly learner 0 ledger"
        )
    learner_id = reader.u32(label="learner ID")
    merges = reader.u64(label="learner merges")
    learner_steps = reader.u64(label="learner steps")
    learner_tokens = reader.u64(label="learner tokens")
    if learner_id != 0 or merges != EXPECTED_COMMITS:
        raise ValidationError("candidate checkpoint learner ledger work drift")
    if learner_steps == 0 or learner_tokens == 0:
        raise ValidationError("candidate checkpoint learner ledger is empty")
    metadata_size = reader.u32(label="layout metadata size")
    metadata = reader.take(metadata_size, label="layout metadata")
    if metadata:
        try:
            metadata.decode("utf-8")
        except UnicodeError as exc:
            raise ValidationError(
                "candidate checkpoint layout metadata is not UTF-8"
            ) from exc
    if reader.u32(label="CPLG extension magic") != CPLG_CKPT_EXTENSION_MAGIC:
        raise ValidationError("candidate checkpoint has no CPLG causal-state extension")
    state_count = reader.u32(label="CPLG state count")
    if state_count != EXPECTED_FRAGMENTS:
        raise ValidationError("candidate checkpoint CPLG state count mismatch")
    unresolved_tail_count = 0
    for fragment, expected_numel in enumerate(numels):
        lengths: list[int] = []
        for vector_name in ("previous stock", "previous tangent", "pending candidate"):
            length = reader.u64(label=f"fragment {fragment} {vector_name} length")
            if length not in {0, expected_numel}:
                raise ValidationError(
                    f"candidate checkpoint fragment {fragment} {vector_name} length mismatch"
                )
            values = reader.take(length * 4, label=f"fragment {fragment} {vector_name}")
            _validate_f32_bytes(values, label=f"fragment {fragment} {vector_name}")
            lengths.append(length)
        tag = reader.u8(label=f"fragment {fragment} theta tag")
        if tag == 1:
            theta_raw = reader.take(4, label=f"fragment {fragment} theta")
            theta = struct.unpack("<f", theta_raw)[0]
            if not math.isfinite(theta) or theta <= 0.0:
                raise ValidationError(
                    f"candidate checkpoint fragment {fragment} theta invalid"
                )
        elif tag != 0:
            raise ValidationError(
                f"candidate checkpoint fragment {fragment} theta tag invalid"
            )
        if bool(lengths[1]) != (tag == 1):
            raise ValidationError(
                f"candidate checkpoint fragment {fragment} phase state invalid"
            )
        score_count = reader.u32(label=f"fragment {fragment} score count")
        if score_count > 3:
            raise ValidationError(
                f"candidate checkpoint fragment {fragment} has too many scores"
            )
        scores = reader.take(score_count * 4, label=f"fragment {fragment} scores")
        _validate_f32_bytes(scores, label=f"fragment {fragment} scores")
        if lengths[0] == 0 and (lengths[1] or lengths[2] or score_count):
            raise ValidationError(
                f"candidate checkpoint fragment {fragment} orphaned state"
            )
        if lengths[1] == 0 and (lengths[2] or score_count):
            raise ValidationError(
                f"candidate checkpoint fragment {fragment} shadow without phase"
            )
        if lengths[2] == 0 and score_count:
            raise ValidationError(
                f"candidate checkpoint fragment {fragment} scores without shadow"
            )
        unresolved_tail_count += int(lengths[2] > 0)
    ledger_rows = reader.u64(label="CPLG ledger row count")
    ledger_head = reader.take(32, label="CPLG ledger head").hex()
    if reader.offset != len(raw):
        raise ValidationError("candidate checkpoint has trailing bytes")
    if ledger_rows != expected_rows or ledger_head != expected_head:
        raise ValidationError("candidate checkpoint ledger head/count mismatch")
    return {
        "bytes": len(raw),
        "sha256": _sha256_file(path),
        "global_step": global_step,
        "fragments": fragments,
        "fragment_versions": versions,
        "learner_merges": merges,
        "learner_steps": learner_steps,
        "learner_tokens": learner_tokens,
        "ledger_rows": ledger_rows,
        "ledger_head": ledger_head,
        "unresolved_tail_count": unresolved_tail_count,
    }


def _stock_checkpoint(path: Path) -> dict[str, Any]:
    raw, global_step, fragments = _checkpoint_header(
        path, label="stock final checkpoint"
    )
    return {
        "bytes": len(raw),
        "sha256": _sha256_file(path),
        "global_step": global_step,
        "fragments": fragments,
    }


def _validate_ledger(
    path: Path,
    *,
    tape_rows: list[dict[str, Any]],
    source_commit: str,
    layout_sha256: str,
    initial_state_sha256: str,
) -> tuple[list[dict[str, Any]], str, int, set[int]]:
    rows, raw_lines = _read_jsonl(path, label="CPLG action ledger")
    if len(rows) != EXPECTED_COMMITS:
        raise ValidationError(
            f"CPLG action ledger must contain exactly {EXPECTED_COMMITS} rows"
        )
    predecessor = "0" * 64
    action_count = 0
    active_fragments: set[int] = set()
    score_history: dict[int, list[float]] = {fragment: [] for fragment in range(4)}
    for index, (row, raw_line, tape) in enumerate(
        zip(rows, raw_lines, tape_rows, strict=True)
    ):
        label = f"CPLG ledger row {index}"
        _require_fields(row, LEDGER_FIELDS, label=label)
        if raw_line.encode("utf-8") != _canonical_json(row):
            raise ValidationError(f"{label} is not canonical JSON")
        expected_values = {
            "schema_version": 1,
            "row_index": index,
            "run_id": RUN_ID,
            "run_config_sha256": RUN_CONFIG_SHA256,
            "source_commit": source_commit,
            "commit_sequence": index,
            "fragment": EXPECTED_FRAGMENT_ORDER[index],
            "fragment_version": index // EXPECTED_FRAGMENTS,
            "layout_sha256": layout_sha256,
            "initial_state_sha256": initial_state_sha256,
            "previous_row_sha256": predecessor,
        }
        for field, wanted in expected_values.items():
            if row[field] != wanted:
                raise ValidationError(f"{label}.{field} mismatch")
        for field in (
            "schema_version",
            "row_index",
            "commit_sequence",
            "fragment",
            "fragment_version",
            "responder_step",
            "responder_tokens",
        ):
            _integer(row[field], label=f"{label}.{field}", minimum=0)
        if row["responder_step"] == 0 or row["responder_tokens"] == 0:
            raise ValidationError(f"{label} responder work must be positive")
        _sha256(row["weight_identity_sha256"], label=f"{label}.weight_identity_sha256")
        _validate_cplg_field_types(row, label=label)
        if row["cplg_reason"] == "not_active":
            raise ValidationError(f"{label} cannot be not_active in the candidate arm")
        declared_digest = _sha256(row["row_sha256"], label=f"{label}.row_sha256")
        digest_input = dict(row)
        del digest_input["row_sha256"]
        recomputed = hashlib.sha256(_canonical_json(digest_input)).hexdigest()
        if declared_digest != recomputed:
            raise ValidationError(f"{label} row_sha256 mismatch")
        reason = row["cplg_reason"]
        stock = row["cplg_stock_sha256"]
        candidate_hash = row["cplg_candidate_sha256"]
        action = row["cplg_action_sha256"]
        used = row["cplg_used_nonstock"]
        if reason == "candidate_selected":
            if (
                used is not True
                or candidate_hash is None
                or action != candidate_hash
                or action == stock
            ):
                raise ValidationError(f"{label} false or malformed candidate action")
            if row["cplg_score_count"] != 3 or row["cplg_interlock_open"] is not True:
                raise ValidationError(f"{label} selected action has a closed interlock")
            action_count += 1
            active_fragments.add(row["fragment"])
        elif used is not False or action != stock:
            raise ValidationError(f"{label} bad exact-stock fallback")
        if row["cplg_state_cleared"] is not (reason in CLEARING_REASONS):
            raise ValidationError(f"{label} state-cleared flag disagrees with reason")
        shadow = row["cplg_shadow_score"]
        fragment = row["fragment"]
        history = score_history[fragment]
        if shadow is not None:
            history.append(shadow)
            del history[:-3]
        if row["cplg_score_count"] != len(history):
            raise ValidationError(f"{label} causal score count mismatch")
        expected_open = len(history) == 3 and all(score > 0.0 for score in history)
        if (
            reason not in CLEARING_REASONS
            and row["cplg_interlock_open"] is not expected_open
        ):
            raise ValidationError(f"{label} causal interlock mismatch")
        if reason == "candidate_selected" and not expected_open:
            raise ValidationError(
                f"{label} action lacks three positive resolved scores"
            )
        if reason in CLEARING_REASONS:
            history.clear()
        for field in CPLG_FIELDS:
            if row[field] != tape[field]:
                raise ValidationError(f"{label}.{field} differs from event tape")
        if tape["fragment"] != row["fragment"] or tape["commit_seq"] != index + 1:
            raise ValidationError(f"{label} differs from event-tape boundary schedule")
        predecessor = recomputed
    return rows, predecessor, action_count, active_fragments


def _validate_ledger_manifest(
    path: Path,
    *,
    source_commit: str,
    layout_sha256: str,
    initial_state_sha256: str,
    ledger_head: str,
    checkpoint_sha256: str,
    tape_sha256: str,
    checkpoint_unresolved_tail_count: int,
) -> dict[str, Any]:
    _verify_sidecar(path, label="CPLG ledger manifest")
    value = _load_json(path, label="CPLG ledger manifest")
    _require_fields(value, LEDGER_MANIFEST_FIELDS, label="CPLG ledger manifest")
    expected = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "run_config_sha256": RUN_CONFIG_SHA256,
        "source_commit": source_commit,
        "arm": CANDIDATE_ARM,
        "layout_sha256": layout_sha256,
        "initial_state_sha256": initial_state_sha256,
        "ledger_rows": EXPECTED_COMMITS,
        "ledger_head": ledger_head,
        "final_checkpoint_sha256": checkpoint_sha256,
        "event_tape_sha256": tape_sha256,
        "expected_commits": EXPECTED_COMMITS,
        "fragments": EXPECTED_FRAGMENTS,
        "outer_optimizer": "cplg-sgd",
        "unresolved_tail": checkpoint_unresolved_tail_count,
        "writer_dropped": 0,
        "writer_abandoned": 0,
        "writer_pending": 0,
        "writer_errors": 0,
    }
    for field, wanted in expected.items():
        if value[field] != wanted:
            raise ValidationError(f"CPLG ledger manifest {field} mismatch")
    for field in (
        "schema_version",
        "ledger_rows",
        "expected_commits",
        "fragments",
        "unresolved_tail",
        "writer_dropped",
        "writer_abandoned",
        "writer_pending",
        "writer_errors",
    ):
        _integer(value[field], label=f"CPLG ledger manifest {field}", minimum=0)
    return value


def _cross_validate_artifacts(
    *,
    files: dict[str, Path],
    stock_completion: dict[str, Any],
    candidate_completion: dict[str, Any],
    stock_checkpoint: dict[str, Any],
    candidate_checkpoint: dict[str, Any],
    ledger_head: str,
) -> None:
    stock_tape = files[f"work/{STOCK_ARM}/tape.jsonl"]
    candidate_tape = files[f"work/{CANDIDATE_ARM}/tape.jsonl"]
    expected = [
        (
            "stock event tape",
            _sha256_file(stock_tape),
            stock_completion["event_tape_sha256"],
        ),
        (
            "candidate event tape",
            _sha256_file(candidate_tape),
            candidate_completion["event_tape_sha256"],
        ),
        (
            "stock final checkpoint",
            stock_checkpoint["sha256"],
            stock_completion["final_checkpoint_sha256"],
        ),
        (
            "candidate final checkpoint",
            candidate_checkpoint["sha256"],
            candidate_completion["final_checkpoint_sha256"],
        ),
        ("candidate ledger head", ledger_head, candidate_completion["ledger_head"]),
    ]
    for label, actual, declared in expected:
        if actual != declared:
            raise ValidationError(f"{label} digest/head mismatch")
    if candidate_completion["ledger_rows"] != candidate_checkpoint["ledger_rows"]:
        raise ValidationError("candidate completion/checkpoint ledger count mismatch")


def validate_acquisition(manifest_path: Path) -> dict[str, Any]:
    """Validate one acquisition without writing to it and return gate evidence."""
    root, manifest, files, manifest_digest = _validate_manifest(manifest_path)
    _validate_required_inventory(files)
    source_commit = manifest["source_commit"]
    data_rows = _validate_data_rows(files)
    results_path = files["report/results.jsonl"]
    result_rows, base_loss, stock_loss, candidate_loss = _validate_results(results_path)

    stock_base = f"work/{STOCK_ARM}"
    candidate_base = f"work/{CANDIDATE_ARM}"
    stock_initial = _validate_initial(
        files[f"{stock_base}/cplg_online_initial_state.json"],
        arm=STOCK_ARM,
        optimizer="nesterov",
        source_commit=source_commit,
    )
    candidate_initial = _validate_initial(
        files[f"{candidate_base}/cplg_online_initial_state.json"],
        arm=CANDIDATE_ARM,
        optimizer="cplg-sgd",
        source_commit=source_commit,
    )
    for field in ("layout_sha256", "initial_state_sha256"):
        if stock_initial[field] != candidate_initial[field]:
            raise ValidationError(f"arms have unequal {field}")

    stock_learner_completion = _validate_learner_completion(
        files[f"{stock_base}/learner-0/learner_completion.json"],
        arm=STOCK_ARM,
    )
    candidate_learner_completion = _validate_learner_completion(
        files[f"{candidate_base}/learner-0/learner_completion.json"],
        arm=CANDIDATE_ARM,
    )
    stock_completion = _validate_completion(
        files[f"{stock_base}/cplg_online_completion.json"],
        arm=STOCK_ARM,
        candidate=False,
        learner_completion=stock_learner_completion,
    )
    candidate_completion = _validate_completion(
        files[f"{candidate_base}/cplg_online_completion.json"],
        arm=CANDIDATE_ARM,
        candidate=True,
        learner_completion=candidate_learner_completion,
    )
    _validate_matched_learner_work(
        stock_learner_completion, candidate_learner_completion
    )
    _validate_observed_learner_work(stock_learner_completion, arm=STOCK_ARM)
    _validate_observed_learner_work(candidate_learner_completion, arm=CANDIDATE_ARM)
    stock_tape_rows, stock_schedule = _validate_tape(
        files[f"{stock_base}/tape.jsonl"], arm=STOCK_ARM, candidate=False
    )
    candidate_tape_rows, candidate_schedule = _validate_tape(
        files[f"{candidate_base}/tape.jsonl"],
        arm=CANDIDATE_ARM,
        candidate=True,
    )
    if stock_schedule != candidate_schedule:
        raise ValidationError("stock and candidate responder schedules/work differ")
    ledger_path = files[f"{candidate_base}/cplg_action_ledger.jsonl"]
    _, ledger_head, action_count, active_fragments = _validate_ledger(
        ledger_path,
        tape_rows=candidate_tape_rows,
        source_commit=source_commit,
        layout_sha256=candidate_initial["layout_sha256"],
        initial_state_sha256=candidate_initial["initial_state_sha256"],
    )
    stock_checkpoint = _stock_checkpoint(files[f"{stock_base}/state.ckpt"])
    candidate_checkpoint = _candidate_checkpoint(
        files[f"{candidate_base}/state.ckpt"],
        expected_head=ledger_head,
        expected_rows=EXPECTED_COMMITS,
    )
    _cross_validate_artifacts(
        files=files,
        stock_completion=stock_completion,
        candidate_completion=candidate_completion,
        stock_checkpoint=stock_checkpoint,
        candidate_checkpoint=candidate_checkpoint,
        ledger_head=ledger_head,
    )
    ledger_manifest = _validate_ledger_manifest(
        files[f"{candidate_base}/cplg_action_ledger_manifest.json"],
        source_commit=source_commit,
        layout_sha256=candidate_initial["layout_sha256"],
        initial_state_sha256=candidate_initial["initial_state_sha256"],
        ledger_head=ledger_head,
        checkpoint_sha256=candidate_checkpoint["sha256"],
        tape_sha256=candidate_completion["event_tape_sha256"],
        checkpoint_unresolved_tail_count=candidate_checkpoint["unresolved_tail_count"],
    )

    loss_regression = candidate_loss - stock_loss
    stock_interval = stock_completion["interval_ns"]
    candidate_interval = candidate_completion["interval_ns"]
    overhead_numerator = candidate_interval - stock_interval
    overhead_fraction = Fraction(overhead_numerator, stock_interval)
    gates = {
        "matched_observed_work": {
            "arms": [STOCK_ARM, CANDIDATE_ARM],
            "observed": {
                field: stock_learner_completion[field] for field in LEARNER_WORK_FIELDS
            },
            "passed": True,
        },
        "valid_nonstock_actions": {
            "observed": action_count,
            "minimum": MINIMUM_ACTIONS,
            "passed": action_count >= MINIMUM_ACTIONS,
        },
        "active_fragments": {
            "observed": len(active_fragments),
            "fragment_ids": sorted(active_fragments),
            "minimum": MINIMUM_ACTIVE_FRAGMENTS,
            "passed": len(active_fragments) >= MINIMUM_ACTIVE_FRAGMENTS,
        },
        "loss_regression": {
            "stock_eval_loss": stock_loss,
            "candidate_eval_loss": candidate_loss,
            "candidate_minus_stock": loss_regression,
            "maximum": MAXIMUM_LOSS_REGRESSION,
            "passed": loss_regression <= MAXIMUM_LOSS_REGRESSION,
        },
        "matched_interval_overhead": {
            "stock_interval_ns": stock_interval,
            "candidate_interval_ns": candidate_interval,
            "numerator_ns": overhead_numerator,
            "denominator_ns": stock_interval,
            "fraction": float(overhead_fraction),
            "maximum_numerator": MAXIMUM_OVERHEAD.numerator,
            "maximum_denominator": MAXIMUM_OVERHEAD.denominator,
            "passed": overhead_fraction <= MAXIMUM_OVERHEAD,
        },
        "closed_evidence": {
            "results_rows": len(result_rows),
            "candidate_boundaries": len(candidate_tape_rows),
            "stock_boundaries": len(stock_tape_rows),
            "writer_nonclosure_count": 0,
            "passed": True,
        },
    }
    passed = all(gate["passed"] for gate in gates.values())
    return {
        "acquisition_root": str(root),
        "acquisition_manifest_sha256": manifest_digest,
        "run_id": RUN_ID,
        "source_commit": source_commit,
        "run_config_sha256": RUN_CONFIG_SHA256,
        "identities": {
            "layout_sha256": stock_initial["layout_sha256"],
            "initial_state_sha256": stock_initial["initial_state_sha256"],
        },
        "data": data_rows,
        "losses": {
            "base": base_loss,
            "stock": stock_loss,
            "candidate": candidate_loss,
        },
        "ledger": {
            "rows": EXPECTED_COMMITS,
            "head": ledger_head,
            "valid_nonstock_actions": action_count,
            "active_fragments": sorted(active_fragments),
            "unresolved_tail_count": ledger_manifest["unresolved_tail"],
        },
        "checkpoints": {
            "stock": stock_checkpoint,
            "candidate": candidate_checkpoint,
        },
        "gates": gates,
        "all_gates_passed": passed,
    }


def _atomic_publish(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _publish_json(path: Path, value: dict[str, Any]) -> str:
    raw = (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_publish(path, raw)
    digest = hashlib.sha256(raw).hexdigest()
    _atomic_publish(
        path.with_name(path.name + ".sha256"),
        f"{digest}  {path.name}\n".encode("ascii"),
    )
    return digest


def _prepare_analysis_dir(manifest_path: Path, analysis_dir: Path) -> Path:
    manifest_path = manifest_path.absolute()
    if manifest_path.parent.name != "report":
        raise ValidationError("cannot derive acquisition root from manifest path")
    acquisition_root = manifest_path.parent.parent.resolve(strict=False)
    destination = analysis_dir.absolute()
    resolved_destination = destination.resolve(strict=False)
    if (
        resolved_destination == acquisition_root
        or acquisition_root in resolved_destination.parents
        or resolved_destination in acquisition_root.parents
    ):
        raise ValidationError(
            "analysis directory must be a distinct prefix outside the acquisition prefix"
        )
    if destination.exists():
        raise ValidationError("analysis directory must be fresh and absent")
    destination.mkdir(parents=True)
    if destination.is_symlink():
        raise ValidationError("analysis directory may not be a symlink")
    return destination


def run_validation(*, acquisition_manifest: Path, analysis_dir: Path) -> dict[str, Any]:
    """Validate and publish one checksummed terminal verdict."""
    destination = _prepare_analysis_dir(acquisition_manifest, analysis_dir)
    detail: dict[str, Any] | None = None
    errors: list[str] = []
    verdict = "INCONCLUSIVE"
    try:
        detail = validate_acquisition(acquisition_manifest)
        verdict = "PASS" if detail["all_gates_passed"] else "FAIL"
    except ValidationError as exc:
        verdict = "INCONCLUSIVE"
        errors.append(str(exc))
    except OSError as exc:
        verdict = "INFRA_FAILURE"
        errors.append(str(exc))
    if verdict not in VERDICTS:
        raise AssertionError(
            "validator produced a verdict outside the closed vocabulary"
        )
    manifest_digest = (
        _sha256_file(acquisition_manifest)
        if acquisition_manifest.is_file() and not acquisition_manifest.is_symlink()
        else None
    )
    analysis = {
        "schema": ANALYSIS_SCHEMA,
        "verdict": verdict,
        "run_id": RUN_ID,
        "acquisition_manifest": str(acquisition_manifest.absolute()),
        "acquisition_manifest_sha256": manifest_digest,
        "validation": detail,
        "errors": errors,
        "bootstrap_performed": False,
        "superiority_claim": False,
        "claim_scope": "frozen_4352_token_matched_online_engineering_canary",
    }
    analysis_path = destination / "analysis_report.json"
    analysis_digest = _publish_json(analysis_path, analysis)
    claim = (
        "CPLG-SGD-v1 was active across the frozen 4,352-token matched online "
        "engineering workload, retained exact stock fallback, stayed within the "
        "absolute 0.05 terminal-loss regression and 2% matched-interval overhead "
        "limits, and completed the frozen artifact contract."
        if verdict == "PASS"
        else None
    )
    terminal = {
        "schema": VERDICT_SCHEMA,
        "verdict": verdict,
        "run_id": RUN_ID,
        "acquisition_manifest_sha256": manifest_digest,
        "analysis_report": analysis_path.name,
        "analysis_report_sha256": analysis_digest,
        "claim": claim,
        "superiority_claim": False,
        "bootstrap_performed": False,
    }
    terminal_path = destination / "terminal_verdict.json"
    terminal_digest = _publish_json(terminal_path, terminal)
    result = dict(terminal)
    result["terminal_verdict_sha256"] = terminal_digest
    result["analysis_dir"] = str(destination)
    result["errors"] = errors
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition-manifest", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_validation(
            acquisition_manifest=args.acquisition_manifest,
            analysis_dir=args.analysis_dir,
        )
    except (ValidationError, OSError) as exc:
        print(f"CPLG CPU validator could not publish: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
