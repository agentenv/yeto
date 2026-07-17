#!/usr/bin/env python3
"""Authorize and execute one all-at-once hidden checkpoint evaluation batch.

``authorize`` consumes the loss-blind checkpoint preseal created after every
training cell is terminal.  It validates every local checkpoint/export hash,
freezes the complete command registry and order, and (for A3) binds the already
sealed prediction freeze.  It does not materialize an audit surface or expose a
loss.

``evaluate`` runs only from that authorization.  It materializes the disjoint
confirmation-audit surface, or validates the already locked development
surface for A3, evaluates every checkpoint without streaming output, validates
per-sequence coverage, seals every private artifact, and records one shared
unblind timestamp only after the complete bundle and seal exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts import audit_135m_contract as audit
from scripts import compare_diloco as compare


REPO_ROOT = Path(__file__).resolve().parents[1]
AUTH_SCHEMA = "audit_135m_hidden_evaluation_authorization_v2"
EXPECTED_AUDIT_HASHES = {
    "audit_eval_rows_hash": "d71b90040a57731f25c78a2d191017ce90a12c1bb79f55a1cd2f3d085a706d7b",
    "audit_eval_packed_hash": "c08d196e15a0b1ee88e64da11521564de2c42d56857ef899b3ba91478ba47f7f",
    "audit_eval_example_ids_hash": "c7be2d71515850da85a3b9d9fa0bf27b56310a25f4b5d009d46ebe887edc1170",
    "audit_eval_token_ids_hash": "5b725289d3308a0b8f64ea0e2a49195e9aa95b8dfaf5b9359644250426c00b41",
    "audit_eval_source_indices_hash": "1b1051e80f559ec6a517fbcfd38e0d39c2ba0b4b880edfac48ddae8ef9963dba",
}


class HiddenEvaluationError(RuntimeError):
    """The hidden batch is premature, incomplete, or hash-drifted."""


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_time(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise HiddenEvaluationError(f"{label} is not a UTC timestamp")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HiddenEvaluationError(f"{label} is not ISO-8601") from exc


def load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise HiddenEvaluationError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise HiddenEvaluationError(f"{label} must be a JSON object")
    return value


def write_create_only(path: Path, value: object) -> None:
    if path.exists():
        raise HiddenEvaluationError(f"refusing to overwrite create-only artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise HiddenEvaluationError(f"{label} must be an array")
    return value


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise HiddenEvaluationError(f"{label} must be a JSON object")
    return dict(value)


def _verify_preseal(path: Path) -> dict[str, Any]:
    value = load_object(path, "checkpoint preseal")
    digest = value.get("preseal_canonical_sha256")
    preimage = dict(value)
    preimage.pop("preseal_canonical_sha256", None)
    attempts = value.get("attempts")
    if (
        value.get("schema") != "audit_135m_checkpoint_preseal_v1"
        or value.get("status") != "SEALED_TRAINING_AND_CHECKPOINT_REGISTRY"
        or value.get("loss_exposed") is not False
        or value.get("partial_outcomes_exposed") is not False
        or value.get("provider_lifecycle_final_pending") is not True
        or digest != canonical_sha256(preimage)
        or not isinstance(attempts, list)
        or value.get("attempts_canonical_sha256") != canonical_sha256(attempts)
    ):
        raise HiddenEvaluationError("checkpoint preseal identity/hash differs")
    maximum = parse_time(
        value.get("maximum_training_completion_utc"),
        "maximum training completion",
    )
    sealed = parse_time(value.get("sealed_at_utc"), "checkpoint preseal time")
    if sealed <= maximum:
        raise HiddenEvaluationError(
            "checkpoint preseal does not follow all terminal training work"
        )
    return value


def _checkpoint_registry(source: Mapping[str, Any]) -> dict[str, Any]:
    value = source.get("audit_checkpoint_registry")
    if not isinstance(value, Mapping):
        raise HiddenEvaluationError("training source lacks audit checkpoint registry")
    registry = dict(value)
    cells = registry.get("cells")
    if (
        registry.get("schema") != "audit_135m_checkpoint_registry_v1"
        or registry.get("loss_exposed") is not False
        or not isinstance(cells, list)
        or registry.get("checkpoint_registry_hash") != canonical_sha256(cells)
    ):
        raise HiddenEvaluationError("audit checkpoint registry identity/hash differs")
    return registry


def _attempt_by_id(source: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in _array(source.get("attempts"), "training attempts"):
        row = _mapping(raw, "training attempt")
        attempt_id = row.get("attempt_id")
        if not isinstance(attempt_id, str) or attempt_id in result:
            raise HiddenEvaluationError("training attempt IDs are missing/duplicated")
        result[attempt_id] = row
    return result


def _checkpoint_inventory_path(
    *, campaign_root: Path, attempt: Mapping[str, Any]
) -> Path:
    inventory = attempt.get("artifact_inventory")
    if not isinstance(inventory, Mapping):
        raise HiddenEvaluationError("attempt artifact inventory is absent")
    entry = inventory.get("checkpoint_inventory")
    if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
        raise HiddenEvaluationError("attempt lacks checkpoint inventory artifact")
    path = (campaign_root / str(entry["path"])).resolve()
    try:
        path.relative_to(campaign_root.resolve())
    except ValueError as exc:
        raise HiddenEvaluationError("checkpoint inventory escapes campaign root") from exc
    if (
        not path.is_file()
        or path.is_symlink()
        or sha256_file(path) != entry.get("sha256")
        or path.stat().st_size != entry.get("size_bytes")
    ):
        raise HiddenEvaluationError("checkpoint inventory artifact hash/size differs")
    return path


def _export_descriptor(
    inventory_path: Path, *, attempt_prefix: str
) -> dict[str, Any]:
    value = load_object(inventory_path, "checkpoint inventory")
    preimage = dict(value)
    digest = preimage.pop("inventory_canonical_sha256", None)
    if digest != canonical_sha256(preimage):
        raise HiddenEvaluationError("checkpoint inventory canonical hash differs")
    files = []
    export_paths = []
    for raw in _array(value.get("files"), "checkpoint inventory files"):
        row = _mapping(raw, "checkpoint inventory file")
        relative = row.get("path")
        if not isinstance(relative, str):
            raise HiddenEvaluationError("checkpoint inventory file path is malformed")
        if "/export/" in f"/{relative}":
            export_paths.append(relative)
    if not export_paths:
        raise HiddenEvaluationError("checkpoint inventory contains no exported model")
    prefix = export_paths[0].split("/export/", 1)[0] + "/export"
    if any(not path.startswith(prefix + "/") for path in export_paths):
        raise HiddenEvaluationError("checkpoint inventory mixes export directories")
    attempt_root = inventory_path.parents[2]
    for relative in export_paths:
        path = attempt_root / relative
        row = next(item for item in value["files"] if item["path"] == relative)
        if (
            not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != row.get("sha256")
            or path.stat().st_size != row.get("size_bytes")
        ):
            raise HiddenEvaluationError("checkpoint export file hash/size differs")
        files.append(
            {
                "path": relative.removeprefix(prefix + "/"),
                "sha256": row["sha256"],
                "size_bytes": row["size_bytes"],
            }
        )
    return {
        "checkpoint_inventory_canonical_sha256": digest,
        "checkpoint_inventory_raw_sha256": sha256_file(inventory_path),
        "export_relative_prefix": prefix,
        "checkpoint_export_uri": attempt_prefix.rstrip("/") + "/" + prefix + "/",
        "export_files": files,
        "export_files_hash": canonical_sha256(files),
        "local_export_dir": str((attempt_root / prefix).resolve()),
    }


def _command_preimage(
    *, cell_id: str, checkpoint_hash: str, evaluation_role: str
) -> list[str]:
    return [
        "<PYTHON_EXECUTABLE>",
        "<COMPARE_DILOCO>",
        "--eval-only",
        "--model",
        "<FROZEN_MODEL>",
        "--data",
        "<BOUND_EVALUATION_SURFACE>",
        "--seq-len",
        "128",
        "--device",
        "<EVALUATION_DEVICE>",
        "--tuning",
        "full",
        "--adapter-dir",
        f"<CHECKPOINT_EXPORT:{cell_id}:{checkpoint_hash}>",
        "--eval-output",
        f"<HIDDEN_PER_SEQUENCE:{cell_id}:{evaluation_role}>",
    ]


def authorize(args: argparse.Namespace) -> dict[str, Any]:
    audit.load_authority()
    preseal = _verify_preseal(args.checkpoint_preseal)
    bound = load_object(args.bound_manifest, "bound manifest")
    if preseal.get("bound_manifest_canonical_sha256") != canonical_sha256(bound):
        raise HiddenEvaluationError("checkpoint preseal bound-manifest hash differs")
    registry = _checkpoint_registry(preseal)
    attempts = _attempt_by_id(preseal)
    pending = _array(
        preseal.get("evaluation_required_cell_ids"), "evaluation-required cells"
    )
    if not pending or len(pending) != len(set(pending)):
        raise HiddenEvaluationError("evaluation-required checkpoint set is empty/duplicated")
    by_cell = {str(row["cell_id"]): row for row in registry["cells"]}
    if set(pending) != {
        cell_id
        for cell_id, row in by_cell.items()
        if row.get("evaluation_mode")
        in {"confirmation_audit_pending", "development_prediction_pending"}
    }:
        raise HiddenEvaluationError("preseal evaluation-required coverage differs")
    required_mode = (
        "confirmation_audit_pending"
        if args.evaluation_role == "confirmation_audit"
        else "development_prediction_pending"
    )
    if any(by_cell[cell_id].get("evaluation_mode") != required_mode for cell_id in pending):
        raise HiddenEvaluationError("evaluation role differs from checkpoint modes")
    max_completion = parse_time(
        preseal["maximum_training_completion_utc"], "maximum training completion"
    )
    preseal_time = parse_time(preseal["sealed_at_utc"], "checkpoint preseal time")
    if args.evaluation_role == "development_prediction_endpoint":
        if args.prediction_freeze is None:
            raise HiddenEvaluationError("A3 prediction-first authorization lacks a freeze")
        prediction = load_object(args.prediction_freeze, "A3 prediction freeze")
        if (
            prediction.get("schema") != "audit_135m_a3_prediction_freeze_v1"
            or prediction.get("status") != "SEALED"
            or prediction.get("loss_exposed_for_H8_H512") is not False
        ):
            raise HiddenEvaluationError("A3 prediction freeze identity/status differs")
        prediction_time = parse_time(
            prediction.get("sealed_at_utc"), "A3 prediction freeze time"
        )
        if prediction_time <= preseal_time:
            raise HiddenEvaluationError(
                "A3 prediction freeze does not follow the complete checkpoint registry"
            )
        prediction_hash = sha256_file(args.prediction_freeze)
    else:
        if args.prediction_freeze is not None:
            raise HiddenEvaluationError("confirmation audit may not bind a prediction freeze")
        prediction_hash = None

    commands = []
    for cell_id in sorted(pending, key=lambda value: value.encode("utf-8")):
        row = by_cell[cell_id]
        attempt = attempts.get(str(row.get("attempt_id")))
        if attempt is None or attempt.get("cell_id") != cell_id:
            raise HiddenEvaluationError("checkpoint registry attempt binding differs")
        if row["status"] == "COMPLETED":
            inventory_path = _checkpoint_inventory_path(
                campaign_root=args.campaign_root.resolve(), attempt=attempt
            )
            export = _export_descriptor(
                inventory_path, attempt_prefix=str(attempt["attempt_prefix"])
            )
            checkpoint_hash = str(export["checkpoint_inventory_canonical_sha256"])
            if checkpoint_hash != row.get("checkpoint_inventory_canonical_sha256"):
                raise HiddenEvaluationError("checkpoint registry/inventory hash differs")
        elif row["status"] == "DIVERGED":
            checkpoint_hash = "0" * 64
            export = None
        else:
            raise HiddenEvaluationError("checkpoint registry status is not terminal")
        command = _command_preimage(
            cell_id=cell_id,
            checkpoint_hash=checkpoint_hash,
            evaluation_role=args.evaluation_role,
        )
        commands.append(
            {
                "cell_id": cell_id,
                "training_status": row["status"],
                "evaluation_mode": row["evaluation_mode"],
                "checkpoint_inventory_canonical_sha256": (
                    None if export is None else checkpoint_hash
                ),
                "export": export,
                "command": command,
                "command_hash": canonical_sha256(command),
            }
        )
    domain = canonical_sha256(
        {
            "preseal": preseal["preseal_canonical_sha256"],
            "checkpoint_registry": registry["checkpoint_registry_hash"],
            "evaluation_role": args.evaluation_role,
            "prediction_freeze": prediction_hash,
        }
    )
    order = sorted(
        pending,
        key=lambda cell_id: hashlib.sha256(
            f"{domain}|{cell_id}".encode("utf-8")
        ).hexdigest(),
    )
    authorized_at = utc_now()
    if parse_time(authorized_at, "authorization") <= max_completion:
        raise HiddenEvaluationError("authorization does not follow all training completions")
    value = {
        "schema": AUTH_SCHEMA,
        "status": "SEALED",
        "stage_code": preseal["stage_code"],
        "evaluation_role": args.evaluation_role,
        "loss_blind": True,
        "partial_exposure_forbidden": True,
        "whole_batch_retry_only": True,
        "checkpoint_preseal_raw_sha256": sha256_file(args.checkpoint_preseal),
        "checkpoint_preseal_canonical_sha256": canonical_sha256(preseal),
        "bound_manifest_raw_sha256": sha256_file(args.bound_manifest),
        "bound_manifest_canonical_sha256": canonical_sha256(bound),
        "checkpoint_registry_hash": registry["checkpoint_registry_hash"],
        "audit_command_registry": commands,
        "audit_command_registry_hash": canonical_sha256(commands),
        "evaluation_order": order,
        "evaluation_order_hash": canonical_sha256(order),
        "expected_cell_ids_hash": canonical_sha256(
            sorted(pending, key=lambda value: value.encode("utf-8"))
        ),
        "prediction_freeze_sha256": prediction_hash,
        "maximum_training_completion_utc": preseal[
            "maximum_training_completion_utc"
        ],
        "authorized_at_utc": authorized_at,
        "authority_prereg_sha256": audit.PREREG_JSON_SHA256,
    }
    value["authorization_canonical_sha256"] = canonical_sha256(value)
    write_create_only(args.output, value)
    return {
        "status": "SEALED",
        "authorization": str(args.output),
        "authorization_sha256": sha256_file(args.output),
        "cell_count": len(order),
        "loss_exposed": False,
    }


def _materialize_confirmation_audit(
    *, source_data: Path, model: Path, output_dir: Path
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    split_dir = output_dir / "private-surface"
    _train, _development, _count = compare.split_data(
        str(source_data),
        split_dir,
        audit.DEVELOPMENT_EVAL_ROWS,
        audit.TRAIN_ROWS,
        shuffle_seed=audit.EVAL_SPLIT_SEED,
        eval_split_seed=audit.EVAL_SPLIT_SEED,
        confirmation_audit_rows=audit.AUDIT_EVAL_ROWS,
    )
    audit_path = split_dir / "confirmation-audit.jsonl"
    split = load_object(split_dir / "split_provenance.json", "audit split provenance")
    source_hash = canonical_sha256(split["audit_eval_source_indices"])
    provenance_dir = output_dir / "private-audit-provenance"
    provenance = compare.materialize_eval_provenance(
        str(model),
        audit_path,
        audit.SEQ_LEN,
        provenance_dir,
        split_provenance=split_dir / "split_provenance.json",
    )
    observed = {
        "audit_eval_rows_hash": sha256_file(audit_path),
        "audit_eval_packed_hash": provenance["eval_packed_hash"],
        "audit_eval_example_ids_hash": provenance["eval_example_ids_hash"],
        "audit_eval_token_ids_hash": provenance["eval_token_ids_hash"],
        "audit_eval_source_indices_hash": source_hash,
    }
    if observed != EXPECTED_AUDIT_HASHES:
        raise HiddenEvaluationError("materialized confirmation-audit surface hash differs")
    sequences = _read_jsonl(
        provenance_dir / "eval_sequences.jsonl", "audit evaluation sequences"
    )
    return audit_path, {**observed, "provenance": provenance}, sequences


def _development_surface(
    *,
    eval_path: Path,
    freeze_path: Path,
    preseal: Mapping[str, Any],
    bound: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    freeze = load_object(freeze_path, "development evaluation freeze")
    registry = _mapping(preseal.get("evaluation_registry"), "evaluation registry")
    matching = [
        row
        for row in registry.values()
        if isinstance(row, Mapping) and row.get("sha256") == sha256_file(freeze_path)
    ]
    frozen = _mapping(bound.get("frozen"), "bound frozen identities")
    if (
        len(matching) != 1
        or freeze.get("schema") != "yeto_parallel_eval_freeze_v1"
        or sha256_file(eval_path) != frozen.get("development_eval_rows_hash")
        or any(
            freeze.get(field) != frozen.get(field)
            for field in (
                "development_eval_rows_hash",
                "development_eval_packed_hash",
                "development_eval_example_ids_hash",
                "development_eval_token_ids_hash",
                "development_eval_source_indices_hash",
            )
        )
    ):
        raise HiddenEvaluationError("locked development prediction surface differs")
    sequences = [
        _mapping(row, "development evaluation sequence")
        for row in _array(freeze.get("sequences"), "development sequences")
    ]
    if not sequences:
        raise HiddenEvaluationError("development evaluation freeze has no sequences")
    surface = {
        "development_eval_rows_hash": frozen["development_eval_rows_hash"],
        "development_eval_packed_hash": frozen["development_eval_packed_hash"],
        "development_eval_example_ids_hash": frozen[
            "development_eval_example_ids_hash"
        ],
        "development_eval_token_ids_hash": frozen["development_eval_token_ids_hash"],
        "development_eval_source_indices_hash": frozen[
            "development_eval_source_indices_hash"
        ],
        "parallel_eval_freeze_sha256": sha256_file(freeze_path),
        "eval_sequence_count": len(sequences),
        "eval_supervised_token_count": freeze.get("supervised_token_count"),
    }
    return eval_path, surface, sequences


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HiddenEvaluationError(
                    f"{label} row {line_number} is invalid JSON"
                ) from exc
            if not isinstance(value, dict):
                raise HiddenEvaluationError(f"{label} row {line_number} is not an object")
            rows.append(value)
    return rows


def _parse_eval_loss(stdout: str, returncode: int) -> float:
    if returncode != 0:
        raise HiddenEvaluationError("hidden evaluation subprocess failed")
    matches = []
    for line in stdout.splitlines():
        if line.startswith("EVAL_LOSS "):
            try:
                matches.append(float(line.split()[1]))
            except (IndexError, ValueError):
                pass
    if len(matches) != 1 or not math.isfinite(matches[0]):
        raise HiddenEvaluationError("hidden evaluation emitted no unique finite aggregate")
    return matches[0]


def _validate_per_sequence(
    path: Path, expected: Sequence[Mapping[str, Any]], aggregate: float
) -> dict[str, Any]:
    rows = _read_jsonl(path, "hidden per-sequence loss")
    if len(rows) != len(expected) or not rows:
        raise HiddenEvaluationError("hidden per-sequence coverage differs")
    total_loss = 0.0
    total_tokens = 0
    identity_fields = (
        "sequence_index",
        "sequence_id",
        "input_ids_sha256",
        "labels_sha256",
        "attention_mask_sha256",
        "supervision_weights_sha256",
        "target_token_mask_sha256",
        "sequence_length",
    )
    for index, (row, frozen) in enumerate(zip(rows, expected)):
        if any(row.get(field) != frozen.get(field) for field in identity_fields):
            raise HiddenEvaluationError(
                f"hidden evaluation sequence identity differs at row {index}"
            )
        tokens = row.get("token_count")
        loss_sum = row.get("loss_sum")
        if (
            isinstance(tokens, bool)
            or not isinstance(tokens, int)
            or tokens <= 0
            or tokens != frozen.get("supervised_token_count")
            or isinstance(loss_sum, bool)
            or not isinstance(loss_sum, (int, float))
            or not math.isfinite(float(loss_sum))
        ):
            raise HiddenEvaluationError("hidden per-sequence arithmetic is malformed")
        total_tokens += tokens
        total_loss += float(loss_sum)
    reproduced = total_loss / total_tokens
    if not math.isclose(reproduced, aggregate, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise HiddenEvaluationError("hidden per-sequence losses do not reproduce aggregate")
    return {
        "sequence_count": len(rows),
        "supervised_token_count": total_tokens,
        "per_sequence_sha256": sha256_file(path),
        "per_sequence_size_bytes": path.stat().st_size,
    }


def _materialize_export(
    *,
    command_binding: Mapping[str, Any],
    local_campaign_root: Path | None,
    cache: Path,
    gcloud_executable: str,
) -> Path:
    descriptor = _mapping(command_binding.get("export"), "authorized export")
    files = _array(descriptor.get("export_files"), "authorized export files")
    if descriptor.get("export_files_hash") != canonical_sha256(files):
        raise HiddenEvaluationError("authorized export file registry hash differs")
    if local_campaign_root is not None:
        candidate = Path(str(descriptor["local_export_dir"]))
        if candidate.is_dir():
            export = candidate
        else:
            export = cache
    else:
        export = cache
    if export == cache:
        if cache.exists():
            raise HiddenEvaluationError("refusing to reuse checkpoint download cache")
        cache.mkdir(parents=True)
        completed = subprocess.run(
            [
                gcloud_executable,
                "storage",
                "rsync",
                "--recursive",
                str(descriptor["checkpoint_export_uri"]),
                str(cache),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode:
            raise HiddenEvaluationError("checkpoint export download failed")
    for raw in files:
        row = _mapping(raw, "authorized export file")
        path = export / str(row["path"])
        if (
            not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != row.get("sha256")
            or path.stat().st_size != row.get("size_bytes")
        ):
            raise HiddenEvaluationError("materialized checkpoint export hash/size differs")
    return export


def _artifact_entry(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    authorization = load_object(args.authorization, "hidden authorization")
    preseal = _verify_preseal(args.checkpoint_preseal)
    bound = load_object(args.bound_manifest, "bound manifest")
    auth_preimage = dict(authorization)
    digest = auth_preimage.pop("authorization_canonical_sha256", None)
    if (
        authorization.get("schema") != AUTH_SCHEMA
        or authorization.get("status") != "SEALED"
        or digest != canonical_sha256(auth_preimage)
        or authorization.get("checkpoint_preseal_raw_sha256")
        != sha256_file(args.checkpoint_preseal)
        or authorization.get("checkpoint_preseal_canonical_sha256")
        != canonical_sha256(preseal)
        or authorization.get("bound_manifest_raw_sha256")
        != sha256_file(args.bound_manifest)
        or authorization.get("bound_manifest_canonical_sha256")
        != canonical_sha256(bound)
        or authorization.get("authority_prereg_sha256") != audit.PREREG_JSON_SHA256
    ):
        raise HiddenEvaluationError("hidden evaluation authorization binding differs")
    if parse_time(utc_now(), "audit start") <= parse_time(
        authorization["authorized_at_utc"], "authorization time"
    ):
        raise HiddenEvaluationError("hidden evaluation does not follow authorization")
    output = args.output_dir.resolve()
    if output.exists():
        raise HiddenEvaluationError(f"refusing to reuse hidden output directory: {output}")
    output.mkdir(parents=True)
    role = authorization["evaluation_role"]
    if role == "confirmation_audit":
        if args.source_data is None or args.model is None:
            raise HiddenEvaluationError("confirmation audit lacks source data/model")
        surface_path, surface, sequences = _materialize_confirmation_audit(
            source_data=args.source_data,
            model=args.model,
            output_dir=output,
        )
    elif role == "development_prediction_endpoint":
        if args.development_eval is None or args.development_eval_freeze is None:
            raise HiddenEvaluationError("A3 evaluation lacks locked development surface")
        surface_path, surface, sequences = _development_surface(
            eval_path=args.development_eval,
            freeze_path=args.development_eval_freeze,
            preseal=preseal,
            bound=bound,
        )
    else:
        raise HiddenEvaluationError("hidden authorization has an unsupported role")
    if args.model is None:
        raise HiddenEvaluationError("hidden evaluation lacks the frozen model root")

    commands = {
        str(row["cell_id"]): row
        for row in _array(
            authorization.get("audit_command_registry"), "audit command registry"
        )
    }
    order = _array(authorization.get("evaluation_order"), "evaluation order")
    if (
        set(commands) != set(order)
        or authorization.get("audit_command_registry_hash")
        != canonical_sha256(authorization["audit_command_registry"])
        or authorization.get("evaluation_order_hash") != canonical_sha256(order)
    ):
        raise HiddenEvaluationError("authorized command/order coverage differs")
    started_at = utc_now()
    results = []
    hidden_logs = output / "private-logs"
    losses_dir = output / "private-per-sequence"
    checkpoint_cache = output / "private-checkpoint-cache"
    hidden_logs.mkdir()
    losses_dir.mkdir()
    checkpoint_cache.mkdir()
    private_artifacts = []
    local_root = args.campaign_root.resolve() if args.campaign_root else None
    env = dict(os.environ)
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    for order_index, cell_id in enumerate(order):
        binding = commands[str(cell_id)]
        if binding["training_status"] == "DIVERGED":
            now = utc_now()
            results.append(
                {
                    "cell_id": cell_id,
                    "order_index": order_index,
                    "training_status": "DIVERGED",
                    "audit_status": "SCIENTIFIC_DIVERGENCE",
                    "audit_loss": None,
                    "analysis_loss_kind": "positive_infinity_scientific_divergence",
                    "checkpoint_inventory_canonical_sha256": None,
                    "audit_command_hash": binding["command_hash"],
                    "per_sequence_sha256": None,
                    "started_at_utc": now,
                    "ended_at_utc": now,
                }
            )
            continue
        cache = checkpoint_cache / f"{order_index:04d}-{cell_id}"
        export = _materialize_export(
            command_binding=binding,
            local_campaign_root=local_root,
            cache=cache,
            gcloud_executable=args.gcloud_executable,
        )
        per_sequence = losses_dir / f"{order_index:04d}-{cell_id}.jsonl"
        command = [
            str(args.python_executable),
            str(args.compare_script),
            "--eval-only",
            "--model",
            str(args.model),
            "--data",
            str(surface_path),
            "--seq-len",
            str(audit.SEQ_LEN),
            "--device",
            args.device,
            "--tuning",
            "full",
            "--adapter-dir",
            str(export),
            "--eval-output",
            str(per_sequence),
        ]
        actual_preimage = _command_preimage(
            cell_id=str(cell_id),
            checkpoint_hash=str(binding["checkpoint_inventory_canonical_sha256"]),
            evaluation_role=role,
        )
        if canonical_sha256(actual_preimage) != binding["command_hash"]:
            raise HiddenEvaluationError("actual hidden command differs from authorization")
        cell_started = utc_now()
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        stdout_path = hidden_logs / f"{order_index:04d}-{cell_id}.stdout"
        stderr_path = hidden_logs / f"{order_index:04d}-{cell_id}.stderr"
        stdout_path.write_text(completed.stdout)
        stderr_path.write_text(completed.stderr)
        loss = _parse_eval_loss(completed.stdout, completed.returncode)
        evidence = _validate_per_sequence(per_sequence, sequences, loss)
        private_artifacts.extend(
            (
                _artifact_entry(stdout_path, output),
                _artifact_entry(stderr_path, output),
                _artifact_entry(per_sequence, output),
            )
        )
        results.append(
            {
                "cell_id": cell_id,
                "order_index": order_index,
                "training_status": "COMPLETED",
                "audit_status": "COMPLETED",
                "audit_loss": loss,
                "analysis_loss_kind": (
                    "finite_confirmation_audit_nll"
                    if role == "confirmation_audit"
                    else "finite_development_prediction_endpoint_nll"
                ),
                "checkpoint_inventory_canonical_sha256": binding[
                    "checkpoint_inventory_canonical_sha256"
                ],
                "audit_command_hash": binding["command_hash"],
                **evidence,
                "started_at_utc": cell_started,
                "ended_at_utc": utc_now(),
            }
        )
        if export == cache:
            shutil.rmtree(cache)
    if [row["cell_id"] for row in results] != order:
        raise HiddenEvaluationError("hidden audit result order/coverage differs")
    if any(path.is_dir() for path in checkpoint_cache.iterdir()):
        raise HiddenEvaluationError("hidden evaluator retained a downloaded checkpoint")
    ended_at = utc_now()
    artifact_by_path = {str(row["path"]): row for row in private_artifacts}
    for path in output.rglob("*"):
        if path.is_file() and not path.is_symlink():
            relative = path.relative_to(output).as_posix()
            if relative.startswith("private-checkpoint-cache/"):
                raise HiddenEvaluationError("hidden checkpoint cache retained a file")
            artifact_by_path[relative] = _artifact_entry(path, output)
    private_artifacts = sorted(
        artifact_by_path.values(), key=lambda row: row["path"].encode("utf-8")
    )
    bundle = {
        "schema": "audit_135m_hidden_evaluation_bundle_v2",
        "status": "SEALED",
        "stage_code": preseal["stage_code"],
        "evaluation_role": role,
        "authorization_raw_sha256": sha256_file(args.authorization),
        "authorization_canonical_sha256": canonical_sha256(authorization),
        "checkpoint_preseal_canonical_sha256": canonical_sha256(preseal),
        "bound_manifest_canonical_sha256": canonical_sha256(bound),
        "checkpoint_registry_hash": authorization["checkpoint_registry_hash"],
        "evaluation_order_hash": authorization["evaluation_order_hash"],
        "audit_command_registry_hash": authorization[
            "audit_command_registry_hash"
        ],
        "prediction_freeze_sha256": authorization["prediction_freeze_sha256"],
        "surface": surface,
        "batch_started_at_utc": started_at,
        "batch_ended_at_utc": ended_at,
        "partial_results_exposed": False,
        "private_artifacts": private_artifacts,
        "private_artifacts_hash": canonical_sha256(private_artifacts),
        "results": results,
    }
    bundle["bundle_canonical_sha256"] = canonical_sha256(bundle)
    bundle_path = output / "audit-bundle.json"
    write_create_only(bundle_path, bundle)
    seal = {
        "schema": "audit_135m_hidden_evaluation_seal_v2",
        "status": "sealed_results",
        "bundle_raw_sha256": sha256_file(bundle_path),
        "bundle_canonical_sha256": bundle["bundle_canonical_sha256"],
        "private_artifacts_hash": bundle["private_artifacts_hash"],
        "cell_count": len(results),
        "sealed_at_utc": utc_now(),
    }
    seal_path = output / "audit-seal.json"
    write_create_only(seal_path, seal)
    unblind = {
        "schema": "audit_135m_shared_unblind_v1",
        "bundle_raw_sha256": seal["bundle_raw_sha256"],
        "seal_raw_sha256": sha256_file(seal_path),
        "shared_unblind_at_utc": utc_now(),
    }
    write_create_only(output / "shared-unblind.json", unblind)
    return {
        "status": "SEALED_AND_UNBLINDED",
        "bundle": str(bundle_path),
        "bundle_sha256": seal["bundle_raw_sha256"],
        "cell_count": len(results),
        "shared_unblind_at_utc": unblind["shared_unblind_at_utc"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    auth = sub.add_parser("authorize")
    auth.add_argument("--checkpoint-preseal", type=Path, required=True)
    auth.add_argument("--bound-manifest", type=Path, required=True)
    auth.add_argument("--campaign-root", type=Path, required=True)
    auth.add_argument(
        "--evaluation-role",
        choices=("confirmation_audit", "development_prediction_endpoint"),
        required=True,
    )
    auth.add_argument("--prediction-freeze", type=Path)
    auth.add_argument("--output", type=Path, required=True)

    run = sub.add_parser("evaluate")
    run.add_argument("--authorization", type=Path, required=True)
    run.add_argument("--checkpoint-preseal", type=Path, required=True)
    run.add_argument("--bound-manifest", type=Path, required=True)
    run.add_argument("--campaign-root", type=Path)
    run.add_argument("--source-data", type=Path)
    run.add_argument("--development-eval", type=Path)
    run.add_argument("--development-eval-freeze", type=Path)
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--python-executable", type=Path, required=True)
    run.add_argument(
        "--compare-script",
        type=Path,
        default=REPO_ROOT / "scripts" / "compare_diloco.py",
    )
    run.add_argument("--device", default="cuda")
    run.add_argument("--gcloud-executable", default="gcloud")
    run.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = authorize(args) if args.action == "authorize" else evaluate(args)
    except (OSError, ValueError, KeyError, HiddenEvaluationError) as exc:
        if args.action == "authorize":
            print(f"hidden evaluation authorization error: {exc}", file=__import__("sys").stderr)
        else:
            print(
                "hidden evaluation batch failed; no loss was exposed and private logs "
                "remain sealed in the failed output namespace",
                file=__import__("sys").stderr,
            )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
