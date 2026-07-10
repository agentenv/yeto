#!/usr/bin/env python3
"""Production-faithful leave-one-out action-probe replay.

The policy compares the exact token-weighted production merge against four
leave-one-responder-out merges. Every alternative is norm-matched to the
baseline SGD step, selected only from paired anchor panels, and evaluated on a
separate oracle file. Oracle losses never affect the selected action.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bn = _load_script("replay_buffered_nesterov_syncer")
buffered = bn.buffered
syncer_eval = bn.syncer_eval

from yeto.data import load_rows  # noqa: E402
from yeto.export import parse_checkpoint  # noqa: E402
from yeto.fragments import build_layout  # noqa: E402
from yeto.learner import load_model_and_tokenizer, trainable_params  # noqa: E402
from yeto.losses import sft_loss  # noqa: E402
from yeto.tensor_io import apply_fragment  # noqa: E402


REPLAY_SCHEMA = "exact_loo_probe_replay_v2"
SUMMARY_SCHEMA = "exact_loo_probe_summary_v2"
COMPLETION_SCHEMA = "exact_loo_probe_completion_v2"
CONFIG_SCHEMA = "exact_loo_probe_config_v2"
CAPTURE_SCHEMA = "syncer_probe_capture_v1"
CAPTURE_ORACLE_SCOPE = "syncer_current_global_pending_offline"
CANONICALIZATION = "messages-tools-json-v1"
BASELINE_ACTION = "token_weighted"
DEFAULT_EXPECTED_GROUPS = 80
DEFAULT_MAX_NEXT_STATE_STEP_RELATIVE_ERROR = 1e-4


def mean(values: Sequence[float]) -> float:
    vals = [float(value) for value in values]
    if not vals:
        return float("nan")
    if any(not math.isfinite(value) for value in vals):
        raise ValueError("mean requires finite values")
    return math.fsum(vals) / len(vals)


def std(values: Sequence[float]) -> float:
    vals = [float(value) for value in values]
    if len(vals) < 2:
        return 0.0
    center = mean(vals)
    variance = math.fsum((value - center) ** 2 for value in vals) / (len(vals) - 1)
    if not math.isfinite(variance) or variance < 0.0:
        raise ValueError("sample variance is not finite")
    return math.sqrt(variance)


def _paired_gains(
    baseline_losses: Sequence[float],
    trial_losses: Sequence[float],
    *,
    context: str,
) -> list[float]:
    try:
        baseline = [float(value) for value in baseline_losses]
        trial = [float(value) for value in trial_losses]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{context}: paired losses must be numeric") from exc
    if not baseline or len(baseline) != len(trial):
        raise ValueError(
            f"{context}: paired losses must be non-empty and equal length "
            f"(got {len(baseline)} and {len(trial)})"
        )
    if any(not math.isfinite(value) or value < 0.0 for value in baseline + trial):
        raise ValueError(f"{context}: paired losses must be finite and nonnegative")
    gains = [left - right for left, right in zip(baseline, trial)]
    if any(not math.isfinite(value) for value in gains):
        raise ValueError(f"{context}: paired gain overflowed")
    return gains


def utility_estimate(base: Sequence[float], trial: Sequence[float]) -> dict[str, Any]:
    """Return a center and SE computed from the same paired-batch gains."""

    gains = _paired_gains(base, trial, context="utility")
    center = mean(gains)
    se = None if len(gains) < 2 else std(gains) / math.sqrt(len(gains))
    return {"center": center, "se": se, "batch_gains": gains}


def utility_se(base: list[float], trial: list[float]) -> float | None:
    return utility_estimate(base, trial)["se"]


def panel_means(values: list[float], panel_size: int) -> list[float]:
    if panel_size <= 0:
        raise ValueError("panel_size must be positive")
    vals = [float(value) for value in values]
    if any(not math.isfinite(value) for value in vals):
        raise ValueError("panel values must be finite")
    if len(vals) < panel_size or len(vals) % panel_size:
        raise ValueError(
            f"{len(vals)} paired values cannot form complete panels of {panel_size}"
        )
    return [
        mean(vals[idx : idx + panel_size]) for idx in range(0, len(vals), panel_size)
    ]


def _invalid_paired_decision(reason: str) -> dict[str, Any]:
    return {
        "gain": 0.0,
        "se": 0.0,
        "lcb": 0.0,
        "win_rate": 0.0,
        "panels": [],
        "eligible": False,
        "valid": False,
        "fallback_reason": reason,
    }


def paired_decision(
    baseline_losses: list[float],
    action_losses: list[float],
    *,
    panel_size: int,
    min_gain: float,
    lcb_z: float,
    min_win_rate: float,
) -> dict:
    """Evaluate paired losses, failing closed on malformed observations."""

    try:
        paired = _paired_gains(
            baseline_losses, action_losses, context="paired decision"
        )
        panels = panel_means(paired, panel_size)
        gain = mean(panels)
        se = 0.0 if len(panels) < 2 else std(panels) / math.sqrt(len(panels))
        lcb = gain - float(lcb_z) * se
        win_rate = sum(value > 0.0 for value in panels) / len(panels)
        thresholds = (float(min_gain), float(lcb_z), float(min_win_rate))
        if any(not math.isfinite(value) for value in thresholds):
            raise ValueError("selection thresholds must be finite")
        if not all(math.isfinite(value) for value in (gain, se, lcb, win_rate)):
            raise ValueError("paired statistic is not finite")
    except (TypeError, ValueError, OverflowError):
        return _invalid_paired_decision("invalid_paired_losses")
    return {
        "gain": gain,
        "se": se,
        "lcb": lcb,
        "win_rate": win_rate,
        "panels": panels,
        "eligible": (gain >= min_gain and lcb > 0.0 and win_rate >= min_win_rate),
        "valid": True,
        "fallback_reason": None,
    }


def jsonable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def infer_seed(path: Path) -> int | None:
    match = re.search(r"seed(\d+)", str(path))
    return None if match is None else int(match.group(1))


def _strict_json_loads(payload: str, *, context: str) -> Any:
    def reject_constant(value: str):
        raise ValueError(f"{context}: non-finite JSON constant {value}")

    try:
        value = json.loads(payload, parse_constant=reject_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{context}: malformed JSON: {exc}") from exc

    def require_finite(item: Any) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"{context}: non-finite JSON number")
        if isinstance(item, Mapping):
            for nested in item.values():
                require_finite(nested)
        elif isinstance(item, list):
            for nested in item:
                require_finite(nested)

    require_finite(value)
    return value


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                row = _strict_json_loads(line, context=f"{path}:{line_number}")
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: expected an object")
                rows.append(row)
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _capture_file_metadata(root: Path, value: str, *, context: str) -> dict[str, Any]:
    capture_root = root.expanduser().resolve()
    relative = Path(value)
    resolved = (
        relative.expanduser().resolve()
        if relative.is_absolute()
        else (capture_root / relative).resolve()
    )
    try:
        normalized = resolved.relative_to(capture_root)
    except ValueError as exc:
        raise ValueError(
            f"{context}: capture path escapes {capture_root}: {value}"
        ) from exc
    if not resolved.is_file():
        raise ValueError(f"{context}: capture file is missing: {resolved}")
    return {
        "path": normalized.as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def capture_payload_provenance(
    root: Path, groups: Sequence[Sequence[Mapping[str, Any]]]
) -> dict[str, Any]:
    states = []
    candidates = []
    seen_states = set()
    seen_candidates = set()
    for group in groups:
        state_value = str(group[0]["state_checkpoint"])
        if state_value not in seen_states:
            states.append(
                _capture_file_metadata(
                    root, state_value, context=f"state checkpoint {state_value}"
                )
            )
            seen_states.add(state_value)
        for row in group:
            candidate_value = str(row["candidate_f32"])
            if candidate_value in seen_candidates:
                continue
            candidates.append(
                _capture_file_metadata(
                    root,
                    candidate_value,
                    context=f"candidate tensor {candidate_value}",
                )
            )
            seen_candidates.add(candidate_value)
    manifest = {
        "state_checkpoints": states,
        "candidate_tensors": candidates,
    }
    manifest["manifest_sha256"] = _config_sha256(manifest)
    return manifest


def _config_sha256(config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_row(row: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        try:
            row = dict(row)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context}: expected an object row") from exc
    try:
        plain = json.loads(json.dumps(dict(row), ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: row is not finite JSON") from exc
    messages = plain.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{context}: expected a non-empty messages list")
    canonical = {"messages": messages}
    if plain.get("tools"):
        canonical["tools"] = plain["tools"]
    return canonical


def _canonical_payload(row: Mapping[str, Any]) -> bytes:
    return json.dumps(
        row,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_row_hash(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_payload(row)).hexdigest()


def data_provenance(path: Path, *, max_rows: int) -> dict[str, Any]:
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"probe data must be a regular file: {resolved}")
    try:
        source = load_rows(str(resolved))
        source_count = len(source)
        selected_count = min(source_count, max_rows)
        rows = [
            _canonical_row(source[index], context=f"{resolved}:row {index}")
            for index in range(selected_count)
        ]
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"cannot load probe data {resolved}: {exc}") from exc
    if not rows:
        raise ValueError(f"probe data is empty: {resolved}")
    row_hashes = [canonical_row_hash(row) for row in rows]
    if len(set(row_hashes)) != len(row_hashes):
        raise ValueError(
            f"{resolved}: selected probe rows contain canonical duplicates"
        )
    canonical_digest = hashlib.sha256()
    for row in rows:
        canonical_digest.update(_canonical_payload(row) + b"\n")
    row_hash_digest = hashlib.sha256()
    for row_hash in row_hashes:
        row_hash_digest.update(row_hash.encode("ascii") + b"\n")
    return {
        "source": str(resolved),
        "source_sha256": _sha256_file(resolved),
        "source_size_bytes": resolved.stat().st_size,
        "source_row_count": source_count,
        "selected_row_count": selected_count,
        "canonicalization": CANONICALIZATION,
        "canonical_row_hashes": row_hashes,
        "canonical_rows_sha256": canonical_digest.hexdigest(),
        "canonical_row_hashes_sha256": row_hash_digest.hexdigest(),
    }


def anchor_oracle_disjointness_metadata(
    anchor: Mapping[str, Any], oracle: Mapping[str, Any]
) -> dict[str, Any]:
    anchor_hashes = tuple(str(value) for value in anchor["canonical_row_hashes"])
    oracle_hashes = tuple(str(value) for value in oracle["canonical_row_hashes"])
    overlap = sorted(set(anchor_hashes) & set(oracle_hashes))
    return {
        "canonicalization": CANONICALIZATION,
        "anchor_row_count": len(anchor_hashes),
        "anchor_unique_row_count": len(set(anchor_hashes)),
        "oracle_row_count": len(oracle_hashes),
        "oracle_unique_row_count": len(set(oracle_hashes)),
        "overlap_count": len(overlap),
        "verified_disjoint": not overlap,
        "overlap_hashes": overlap,
    }


def build_data_provenance(args) -> dict[str, Any]:
    anchor = data_provenance(args.anchor_data, max_rows=args.max_rows)
    oracle = data_provenance(args.oracle_data, max_rows=args.max_rows)
    disjointness = anchor_oracle_disjointness_metadata(anchor, oracle)
    if not disjointness["verified_disjoint"]:
        raise ValueError(
            "anchor and oracle probe data overlap on "
            f"{disjointness['overlap_count']} canonical rows"
        )
    return {
        "anchor": anchor,
        "oracle": oracle,
        "disjointness": disjointness,
    }


def _require_int(row: Mapping[str, Any], field: str, *, context: str) -> int:
    value = row.get(field)
    if isinstance(value, bool):
        raise ValueError(f"{context}: {field} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{context}: {field} must be an integer") from exc
    if isinstance(value, float) and value != parsed:
        raise ValueError(f"{context}: {field} must be an integer")
    return parsed


def candidate_weight(row: Mapping[str, Any], *, context: str) -> float:
    raw = row.get("weight", row.get("c_tokens"))
    try:
        weight = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{context}: candidate weight must be numeric") from exc
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError(f"{context}: candidate weight must be finite and positive")
    return weight


def _group_key_from_row(row: Mapping[str, Any]) -> tuple[str, int, int]:
    return (
        str(row["state_checkpoint"]),
        int(row["step"]),
        int(row["fragment"]),
    )


def _group_key(group: Sequence[Mapping[str, Any]]) -> tuple[str, int, int]:
    return _group_key_from_row(group[0])


def group_id(group: Sequence[Mapping[str, Any]]) -> str:
    first = group[0]
    payload = {
        "state_checkpoint": str(first["state_checkpoint"]),
        "step": int(first["step"]),
        "fragment": int(first["fragment"]),
        "syncer_global_step": int(first["syncer_global_step"]),
        "current_fragment_version": int(first["current_fragment_version"]),
        "learner_ids": sorted(int(row["learner_id"]) for row in group),
    }
    return _config_sha256(payload)


def group_descriptor(group: Sequence[Mapping[str, Any]], index: int) -> dict[str, Any]:
    first = group[0]
    return {
        "group_id": group_id(group),
        "group_index": index,
        "group_ordinal": index + 1,
        "state_checkpoint": str(first["state_checkpoint"]),
        "step": int(first["step"]),
        "fragment": int(first["fragment"]),
        "syncer_global_step": int(first["syncer_global_step"]),
        "current_fragment_version": int(first["current_fragment_version"]),
        "learner_ids": sorted(int(row["learner_id"]) for row in group),
    }


def validate_candidate_groups(
    rows: list[dict], expected_candidates: int
) -> list[list[dict]]:
    """Validate all capture groups before any incomplete group can be dropped."""

    if expected_candidates < 1:
        raise ValueError("expected_candidates must be positive")
    if not rows:
        raise ValueError("capture contains no candidate rows")
    grouped: dict[tuple[str, int, int], list[dict]] = OrderedDict()
    problems = []
    for row_index, row in enumerate(rows):
        context = f"capture row {row_index}"
        if not isinstance(row, Mapping):
            problems.append(f"{context}: expected an object")
            continue
        try:
            schema = row.get("schema")
            if schema != CAPTURE_SCHEMA:
                raise ValueError(f"{context}: unsupported capture schema {schema!r}")
            if row.get("oracle_scope") != CAPTURE_ORACLE_SCOPE:
                raise ValueError(
                    f"{context}: unsupported oracle_scope {row.get('oracle_scope')!r}"
                )
            state_checkpoint = row.get("state_checkpoint")
            candidate_f32 = row.get("candidate_f32")
            if not isinstance(state_checkpoint, str) or not state_checkpoint:
                raise ValueError(f"{context}: state_checkpoint must be non-empty")
            if not isinstance(candidate_f32, str) or not candidate_f32:
                raise ValueError(f"{context}: candidate_f32 must be non-empty")
            step = _require_int(row, "step", context=context)
            fragment = _require_int(row, "fragment", context=context)
            learner_id = _require_int(row, "learner_id", context=context)
            syncer_step = _require_int(row, "syncer_global_step", context=context)
            current_version = _require_int(
                row, "current_fragment_version", context=context
            )
            base_version = _require_int(row, "base_version", context=context)
            local_step = _require_int(row, "local_step", context=context)
            c_steps = _require_int(row, "c_steps", context=context)
            c_tokens = _require_int(row, "c_tokens", context=context)
            if (
                min(
                    step,
                    fragment,
                    learner_id,
                    syncer_step,
                    current_version,
                    base_version,
                )
                < 0
            ):
                raise ValueError(f"{context}: capture metadata must be nonnegative")
            if local_step < 0 or c_steps <= 0 or c_tokens <= 0:
                raise ValueError(
                    f"{context}: local_step must be nonnegative and contribution "
                    "steps/tokens must be positive"
                )
            if current_version >= step:
                raise ValueError(
                    f"{context}: current_fragment_version must precede step"
                )
            if base_version > current_version:
                raise ValueError(
                    f"{context}: base_version exceeds current_fragment_version"
                )
            weight = candidate_weight(row, context=context)
            expected_weight = float(c_tokens) * float(c_tokens) / float(c_steps)
            if not math.isfinite(expected_weight) or not math.isclose(
                weight, expected_weight, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(
                    f"{context}: candidate weight does not match c_tokens^2/c_steps"
                )
            key = (state_checkpoint, step, fragment)
            grouped.setdefault(key, []).append(dict(row))
        except ValueError as exc:
            problems.append(str(exc))
    if problems:
        preview = "; ".join(problems[:5])
        suffix = "" if len(problems) <= 5 else f"; plus {len(problems) - 5} more"
        raise ValueError(f"candidate group validation failed: {preview}{suffix}")

    expected_ids = set(range(expected_candidates))
    validated = []
    seen_steps: dict[int, tuple[str, int, int]] = {}
    seen_state_paths: dict[str, tuple[str, int, int]] = {}
    seen_candidate_paths: set[str] = set()
    metadata_fields = (
        "schema",
        "oracle_scope",
        "state_checkpoint",
        "step",
        "syncer_global_step",
        "fragment",
        "current_fragment_version",
    )
    problems = []
    for key, group in grouped.items():
        group = sorted(group, key=lambda row: int(row["learner_id"]))
        first = group[0]
        label = f"step={first['step']} fragment={first['fragment']}"
        learner_ids = [int(row["learner_id"]) for row in group]
        if len(group) != expected_candidates:
            problems.append(
                f"{label} has {len(group)} candidates, expected {expected_candidates}"
            )
        if len(set(learner_ids)) != len(learner_ids):
            problems.append(f"{label} has duplicate learner IDs {learner_ids}")
        if set(learner_ids) != expected_ids:
            problems.append(
                f"{label} learner IDs are {learner_ids}, expected {sorted(expected_ids)}"
            )
        for field in metadata_fields:
            values = [row.get(field) for row in group]
            if any(value != values[0] for value in values[1:]):
                problems.append(f"{label} has inconsistent {field} metadata")
        candidate_paths = [str(row["candidate_f32"]) for row in group]
        if len(set(candidate_paths)) != len(candidate_paths):
            problems.append(f"{label} reuses a candidate tensor path")
        step = int(first["step"])
        if step in seen_steps and seen_steps[step] != key:
            problems.append(f"step={step} appears in multiple candidate groups")
        seen_steps[step] = key
        state_path = str(first["state_checkpoint"])
        if state_path in seen_state_paths and seen_state_paths[state_path] != key:
            problems.append(f"state checkpoint {state_path} is reused across groups")
        seen_state_paths[state_path] = key
        reused_candidates = sorted(set(candidate_paths) & seen_candidate_paths)
        if reused_candidates:
            problems.append(
                f"{label} reuses candidate paths from earlier groups: {reused_candidates}"
            )
        seen_candidate_paths.update(candidate_paths)
        validated.append(group)
    if problems:
        preview = "; ".join(problems[:5])
        suffix = "" if len(problems) <= 5 else f"; plus {len(problems) - 5} more"
        raise ValueError(f"candidate group validation failed: {preview}{suffix}")
    return sorted(
        validated,
        key=lambda group: (
            int(group[0]["step"]),
            int(group[0]["fragment"]),
            str(group[0]["state_checkpoint"]),
        ),
    )


def capture_order_next_checkpoints(rows: Sequence[Mapping[str, Any]]) -> dict:
    order: list[tuple[str, int, int]] = []
    closed: set[tuple[str, int, int]] = set()
    active = None
    for row in rows:
        key = _group_key_from_row(row)
        if key == active:
            continue
        if active is not None:
            closed.add(active)
        if key in closed:
            raise ValueError(f"capture group {key} is not contiguous in index.jsonl")
        order.append(key)
        active = key
    return {current: following[0] for current, following in zip(order, order[1:])}


def validate_expected_group_coverage(
    groups: Sequence[Sequence[Mapping[str, Any]]], expected_groups: int
) -> None:
    if len(groups) != expected_groups:
        raise ValueError(
            f"capture has {len(groups)} complete groups, expected {expected_groups}"
        )
    steps = sorted(int(group[0]["step"]) for group in groups)
    expected_steps = list(range(1, expected_groups + 1))
    if steps != expected_steps:
        missing = sorted(set(expected_steps) - set(steps))
        extra = sorted(set(steps) - set(expected_steps))
        raise ValueError(
            f"capture step coverage mismatch: missing={missing}, extra={extra}"
        )


def validate_candidate_tensor(tensor: torch.Tensor, *, context: str) -> None:
    if tensor.numel() == 0 or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{context}: candidate tensor must be non-empty and finite")


def probe_batches(args, tokenizer, device, data: Path, batches: int):
    probe_args = argparse.Namespace(
        data=data,
        probe_batches=batches,
        probe_batch_size=args.batch_size,
        probe_max_rows=args.max_rows,
        seq_len=args.seq_len,
        train_on=args.train_on,
    )
    result = syncer_eval._probe_batches(probe_args, tokenizer, device)
    if len(result) != batches:
        raise ValueError(
            f"{data}: produced {len(result)} probe batches, expected exactly {batches}"
        )
    for index, (_, weights) in enumerate(result):
        target_weight = float(weights[:, 1:].sum().item())
        if not math.isfinite(target_weight) or target_weight <= 0.0:
            raise ValueError(f"{data}: probe batch {index} has no finite target weight")
    return result


def _validate_loss_output(
    loss: float,
    by_batch: Sequence[float],
    *,
    expected_batches: int,
    context: str,
) -> tuple[float, list[float]]:
    total = float(loss)
    values = [float(value) for value in by_batch]
    if len(values) != expected_batches:
        raise ValueError(
            f"{context}: got {len(values)} batch losses, expected {expected_batches}"
        )
    if not math.isfinite(total) or total < 0.0:
        raise ValueError(f"{context}: aggregate loss must be finite and nonnegative")
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError(f"{context}: batch losses must be finite and nonnegative")
    return total, values


def losses_for_trial(
    model,
    batches,
    compute_loss,
    frag,
    params,
    current: torch.Tensor,
    trial: torch.Tensor,
    device,
) -> tuple[float, list[float]]:
    validate_candidate_tensor(trial, context="trial")
    apply_fragment(frag, trial.to(device), params)
    try:
        loss, by_batch = syncer_eval._losses(model, batches, compute_loss)
        return _validate_loss_output(
            loss,
            by_batch,
            expected_batches=len(batches),
            context="trial evaluation",
        )
    finally:
        apply_fragment(frag, current.to(device), params)


def norm_matched_trial(
    current: torch.Tensor,
    raw_trial: torch.Tensor,
    baseline_trial: torch.Tensor,
    *,
    min_scale: float,
    max_scale: float,
) -> tuple[torch.Tensor, float, bool]:
    if not all(
        bool(torch.isfinite(tensor).all())
        for tensor in (current, raw_trial, baseline_trial)
    ):
        return raw_trial, float("nan"), False
    raw_step = raw_trial - current
    baseline_step = baseline_trial - current
    raw_norm = float(raw_step.norm().item())
    baseline_norm = float(baseline_step.norm().item())
    if (
        not math.isfinite(raw_norm)
        or not math.isfinite(baseline_norm)
        or raw_norm <= 1e-12
        or baseline_norm <= 1e-12
    ):
        return raw_trial, float("nan"), False
    scale = baseline_norm / raw_norm
    trial = current + raw_step * scale
    valid = (
        math.isfinite(scale)
        and min_scale <= scale <= max_scale
        and bool(torch.isfinite(trial).all())
    )
    return trial, scale, valid


def deterministic_random_valid_action(
    actions: Sequence[Mapping[str, Any]],
    *,
    random_seed: int,
    seed: int,
    stable_group_id: str,
) -> Mapping[str, Any] | None:
    valid = sorted(
        (action for action in actions if bool(action.get("valid"))),
        key=lambda action: (
            int(action.get("dropped_learner", 1 << 30)),
            str(action.get("name", "")),
        ),
    )
    if not valid:
        return None
    payload = json.dumps(
        {
            "algorithm": "sha256_mod_valid_actions_v1",
            "random_seed": int(random_seed),
            "seed": int(seed),
            "group_id": stable_group_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    index = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % len(valid)
    return valid[index]


def next_state_parity(
    current: torch.Tensor,
    baseline_trial: torch.Tensor,
    captured_next: torch.Tensor,
) -> dict[str, float]:
    if current.shape != baseline_trial.shape or current.shape != captured_next.shape:
        raise ValueError("next-state parity tensors have different shapes")
    if not all(
        bool(torch.isfinite(tensor).all())
        for tensor in (current, baseline_trial, captured_next)
    ):
        raise ValueError("next-state parity tensors must be finite")
    absolute = float((baseline_trial - captured_next).norm().item())
    state_norm = float(captured_next.norm().item())
    captured_step_norm = float((captured_next - current).norm().item())
    relative = absolute / max(state_norm, 1e-12)
    if captured_step_norm <= 1e-12:
        step_relative = 0.0 if absolute <= 1e-12 else float("inf")
    else:
        step_relative = absolute / captured_step_norm
    return {
        "absolute_error": absolute,
        "relative_error": relative,
        "step_relative_error": step_relative,
        "captured_step_norm": captured_step_norm,
    }


def _record_group_id(record: Mapping[str, Any]) -> str:
    value = record.get("group_id")
    if not isinstance(value, str) or not value:
        raise ValueError("replay record is missing group_id")
    return value


def read_replay_artifact(path: Path) -> tuple[list[dict], dict | None]:
    if not path.exists():
        return [], None
    if path.stat().st_size == 0:
        return [], None
    rows = read_jsonl(path)
    completion_positions = [
        index
        for index, row in enumerate(rows)
        if row.get("schema") == COMPLETION_SCHEMA
    ]
    if len(completion_positions) > 1:
        raise ValueError(f"{path}: contains multiple completion records")
    if completion_positions and completion_positions[0] != len(rows) - 1:
        raise ValueError(f"{path}: completion record must be the final JSONL row")
    completion = rows[-1] if completion_positions else None
    records = rows[:-1] if completion is not None else rows
    return records, completion


def _ordered_id_digest(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("ascii") + b"\n")
    return digest.hexdigest()


def validate_completion_artifact(
    records: Sequence[Mapping[str, Any]], completion: Mapping[str, Any]
) -> None:
    if completion.get("schema") != COMPLETION_SCHEMA:
        raise ValueError("incompatible completion schema")
    if completion.get("replay_schema") != REPLAY_SCHEMA:
        raise ValueError("completion record names an incompatible replay schema")
    if completion.get("complete") is not True:
        raise ValueError("completion record does not assert complete=true")
    compatibility_config = completion.get("compatibility_config")
    capture_config = completion.get("capture_config")
    replay_config = completion.get("replay_config")
    if not all(
        isinstance(value, Mapping)
        for value in (
            compatibility_config,
            capture_config,
            replay_config,
        )
    ):
        raise ValueError("completion record is missing replay configurations")
    expected_digests = {
        "compatibility_config_sha256": _config_sha256(compatibility_config),
        "capture_config_sha256": _config_sha256(capture_config),
        "replay_config_sha256": _config_sha256(replay_config),
    }
    for field, expected in expected_digests.items():
        if completion.get(field) != expected:
            raise ValueError(f"completion {field} does not match its config")
    expected_ids = list(completion.get("expected_group_ids", []))
    completed_ids = list(completion.get("completed_group_ids", []))
    actual_ids = [_record_group_id(record) for record in records]
    if len(set(actual_ids)) != len(actual_ids):
        raise ValueError("completed replay contains duplicate group IDs")
    if actual_ids != completed_ids or completed_ids != expected_ids:
        raise ValueError("completion group coverage does not match replay records")
    if completion.get("record_count") != len(records):
        raise ValueError("completion record_count does not match replay records")
    if completion.get("completed_group_count") != len(completed_ids):
        raise ValueError("completion completed_group_count is inconsistent")
    if completion.get("expected_group_count") != len(expected_ids):
        raise ValueError("completion expected_group_count is inconsistent")
    if completion.get("expected_group_ids_sha256") != _ordered_id_digest(expected_ids):
        raise ValueError("completion expected group digest is inconsistent")
    seed = int(completion["seed"])
    for record in records:
        if record.get("schema") != REPLAY_SCHEMA:
            raise ValueError("completion artifact contains an incompatible replay row")
        if int(record["seed"]) != seed:
            raise ValueError("completion artifact mixes seeds")
        for field in expected_digests:
            if record.get(field) != completion.get(field):
                raise ValueError(f"replay row has incompatible {field}")


def _validate_resume_records(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    allowed_group_ids: set[str],
    compatibility_config_sha256: str,
    capture_config_sha256: str,
    replay_config_sha256: str,
) -> set[str]:
    seen = set()
    for record in records:
        if record.get("schema") != REPLAY_SCHEMA:
            raise ValueError("resume JSONL contains an incompatible replay schema")
        if int(record.get("seed", -1)) != seed:
            raise ValueError("resume JSONL seed does not match --seed")
        expected = {
            "compatibility_config_sha256": compatibility_config_sha256,
            "capture_config_sha256": capture_config_sha256,
            "replay_config_sha256": replay_config_sha256,
        }
        for field, digest in expected.items():
            if record.get(field) != digest:
                raise ValueError(f"resume JSONL was produced with a different {field}")
        identifier = _record_group_id(record)
        if identifier not in allowed_group_ids:
            raise ValueError(
                f"resume record {identifier} is outside the selected group shard"
            )
        if identifier in seen:
            raise ValueError(f"resume JSONL contains duplicate group {identifier}")
        seen.add(identifier)
    return seen


def _build_configs(
    args,
    *,
    index_path: Path,
    data_metadata: Mapping[str, Any],
    capture_payloads: Mapping[str, Any],
    capture_descriptors: Sequence[Mapping[str, Any]],
    full_shard_descriptors: Sequence[Mapping[str, Any]],
    expected_descriptors: Sequence[Mapping[str, Any]],
) -> tuple[dict, dict, dict]:
    data_identity = {
        label: {key: value for key, value in metadata.items() if key != "source"}
        for label, metadata in data_metadata.items()
        if label in {"anchor", "oracle"}
    }
    data_identity["disjointness"] = data_metadata["disjointness"]
    compatibility_config = {
        "schema": CONFIG_SCHEMA,
        "model": args.model,
        "model_layout": {
            "tuning": "lora",
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_targets": args.lora_targets,
            "fragments": args.fragments,
            "fragment_pattern": args.fragment_pattern,
        },
        "probe": {
            "seq_len": args.seq_len,
            "train_on": args.train_on,
            "loss_function": args.loss_function,
            "anchor_batches": args.anchor_batches,
            "oracle_batches": args.oracle_batches,
            "batch_size": args.batch_size,
            "max_rows": args.max_rows,
            "panel_size": args.panel_size,
        },
        "merge": {
            "outer_lr": args.outer_lr,
            "outer_momentum": args.outer_momentum,
            "delta_correction": args.delta_correction,
            "expected_candidates": args.expected_candidates,
            "min_selected_mass": args.min_selected_mass,
            "norm_scale_range": [args.min_norm_scale, args.max_norm_scale],
            "max_step_ratio_error": args.max_step_ratio_error,
        },
        "selection": {
            "min_gain": args.min_gain,
            "lcb_z": args.lcb_z,
            "min_win_rate": args.min_win_rate,
            "fallback_action": BASELINE_ACTION,
        },
        "random_control": {
            "algorithm": "sha256_mod_valid_actions_v1",
            "random_seed": args.random_seed,
        },
        "next_state_validation": {
            "max_step_relative_error": args.max_next_state_step_relative_error,
            "comparison": "strictly_less_than",
        },
        "expected_capture_group_count": args.expected_groups,
        "data": data_identity,
    }
    compatibility_digest = _config_sha256(compatibility_config)
    capture_config = {
        "schema": CONFIG_SCHEMA,
        "compatibility_config_sha256": compatibility_digest,
        "seed": args.seed,
        "capture_dir": str(args.capture_dir.expanduser().resolve()),
        "capture_index": str(index_path.resolve()),
        "capture_index_sha256": _sha256_file(index_path),
        "capture_payloads": capture_payloads,
        "capture_groups": list(capture_descriptors),
    }
    capture_digest = _config_sha256(capture_config)
    replay_config = {
        "schema": CONFIG_SCHEMA,
        "compatibility_config_sha256": compatibility_digest,
        "capture_config_sha256": capture_digest,
        "group_shard": {
            "start": args.group_start,
            "stride": args.group_stride,
            "max_groups": args.max_groups,
        },
        "full_shard_group_ids": [
            descriptor["group_id"] for descriptor in full_shard_descriptors
        ],
        "expected_group_ids": [
            descriptor["group_id"] for descriptor in expected_descriptors
        ],
    }
    return compatibility_config, capture_config, replay_config


def _write_jsonl_line(sink, value: Mapping[str, Any]) -> None:
    sink.write(
        json.dumps(
            jsonable(dict(value)),
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    sink.flush()
    try:
        os.fsync(sink.fileno())
    except (AttributeError, OSError):
        # Test and embedding callers may provide an in-memory text sink.
        pass


def make_completion_metadata(
    records: Sequence[Mapping[str, Any]], args
) -> dict[str, Any]:
    expected_ids = list(args._expected_group_ids)
    completed_ids = [_record_group_id(record) for record in records]
    if completed_ids != expected_ids:
        raise ValueError(
            "cannot mark replay complete: completed group order does not match expected coverage"
        )
    full_shard_ids = list(args._full_shard_group_ids)
    return {
        "schema": COMPLETION_SCHEMA,
        "replay_schema": REPLAY_SCHEMA,
        "complete": True,
        "full_shard_complete": (
            args.max_groups is None and expected_ids == full_shard_ids
        ),
        "seed": args.seed,
        "compatibility_config": args._compatibility_config,
        "compatibility_config_sha256": args._compatibility_config_sha256,
        "capture_config": args._capture_config,
        "capture_config_sha256": args._capture_config_sha256,
        "replay_config": args._replay_config,
        "replay_config_sha256": args._replay_config_sha256,
        "group_shard": {
            "start": args.group_start,
            "stride": args.group_stride,
            "max_groups": args.max_groups,
        },
        "capture_group_count": len(args._capture_group_ids),
        "capture_group_ids": list(args._capture_group_ids),
        "capture_group_coordinates": list(args._capture_group_coordinates),
        "full_shard_group_count": len(full_shard_ids),
        "full_shard_group_ids": full_shard_ids,
        "expected_group_count": len(expected_ids),
        "expected_group_ids": expected_ids,
        "expected_group_ids_sha256": _ordered_id_digest(expected_ids),
        "completed_group_count": len(completed_ids),
        "completed_group_ids": completed_ids,
        "record_count": len(records),
    }


def _next_state_validation(
    *,
    root: Path,
    next_checkpoint: str | None,
    checkpoint_cache: dict[Path, Any],
    checkpoint_metadata: Mapping[Path, Mapping[str, Any]],
    fragment: int,
    step: int,
    current: torch.Tensor,
    baseline_trial: torch.Tensor,
    max_step_relative_error: float,
) -> dict[str, Any]:
    prefix = "production_baseline_next_state"
    result = {
        f"{prefix}_available": False,
        f"{prefix}_checkpoint": next_checkpoint,
        f"{prefix}_fragment_version": None,
        f"{prefix}_absolute_error": None,
        f"{prefix}_relative_error": None,
        f"{prefix}_step_relative_error": None,
        f"{prefix}_captured_step_norm": None,
        f"{prefix}_unavailable_reason": None,
    }
    if next_checkpoint is None:
        result[f"{prefix}_unavailable_reason"] = "no_later_captured_checkpoint"
        return result
    path = buffered._resolve(root, next_checkpoint).resolve()
    metadata = checkpoint_metadata.get(path)
    if metadata is None:
        raise ValueError(f"{path}: next checkpoint is absent from capture manifest")
    if (
        path.stat().st_size != metadata["size_bytes"]
        or _sha256_file(path) != metadata["sha256"]
    ):
        raise ValueError(f"{path}: next checkpoint changed after capture validation")
    if path not in checkpoint_cache:
        checkpoint_cache[path] = parse_checkpoint(path)
    checkpoint = checkpoint_cache[path]
    if fragment >= len(checkpoint.fragments):
        raise ValueError(f"{path}: missing fragment {fragment}")
    next_version, captured_next, _ = checkpoint.fragments[fragment]
    result[f"{prefix}_fragment_version"] = int(next_version)
    if int(next_version) != step:
        result[f"{prefix}_unavailable_reason"] = (
            "captured_fragment_version_does_not_match_replayed_step"
        )
        return result
    parity = next_state_parity(current, baseline_trial, captured_next)
    result.update(
        {
            f"{prefix}_available": True,
            f"{prefix}_absolute_error": parity["absolute_error"],
            f"{prefix}_relative_error": parity["relative_error"],
            f"{prefix}_step_relative_error": parity["step_relative_error"],
            f"{prefix}_captured_step_norm": parity["captured_step_norm"],
        }
    )
    if not parity["step_relative_error"] < max_step_relative_error:
        raise RuntimeError(
            "production baseline replay diverged from the captured next state: "
            f"step={step} fragment={fragment} "
            f"step_relative_error={parity['step_relative_error']:.3e} "
            f"limit={max_step_relative_error:.3e}"
        )
    return result


def replay(args) -> list[dict]:
    if args.outer_momentum != 0.0:
        raise SystemExit("exact norm matching currently requires --outer-momentum 0")
    root = args.capture_dir
    index_path = root / "index.jsonl"
    try:
        index_rows = read_jsonl(index_path)
        groups = validate_candidate_groups(index_rows, args.expected_candidates)
        next_checkpoints = capture_order_next_checkpoints(index_rows)
        data_metadata = build_data_provenance(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.expected_groups is not None:
        try:
            validate_expected_group_coverage(groups, args.expected_groups)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    invalid_fragments = sorted(
        {
            int(group[0]["fragment"])
            for group in groups
            if int(group[0]["fragment"]) >= args.fragments
        }
    )
    if invalid_fragments:
        raise SystemExit(
            f"capture fragment IDs {invalid_fragments} exceed configured layout"
        )
    try:
        capture_payloads = capture_payload_provenance(root, groups)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    indexed_groups = list(enumerate(groups))
    capture_descriptors = [
        group_descriptor(group, index) for index, group in indexed_groups
    ]
    full_shard = indexed_groups[args.group_start :: args.group_stride]
    full_shard_descriptors = [capture_descriptors[index] for index, _ in full_shard]
    selected = full_shard
    if args.max_groups is not None:
        selected = selected[: args.max_groups]
    selected_descriptors = [capture_descriptors[index] for index, _ in selected]
    if not selected:
        raise SystemExit("no complete candidate groups selected by this shard")

    compatibility_config, capture_config, replay_config = _build_configs(
        args,
        index_path=index_path,
        data_metadata=data_metadata,
        capture_payloads=capture_payloads,
        capture_descriptors=capture_descriptors,
        full_shard_descriptors=full_shard_descriptors,
        expected_descriptors=selected_descriptors,
    )
    compatibility_digest = _config_sha256(compatibility_config)
    capture_digest = _config_sha256(capture_config)
    replay_digest = _config_sha256(replay_config)
    expected_ids = [descriptor["group_id"] for descriptor in selected_descriptors]
    allowed_ids = set(expected_ids)
    existing = list(getattr(args, "_existing_records", []))
    existing_completion = getattr(args, "_existing_completion", None)
    try:
        seen = _validate_resume_records(
            existing,
            seed=args.seed,
            allowed_group_ids=allowed_ids,
            compatibility_config_sha256=compatibility_digest,
            capture_config_sha256=capture_digest,
            replay_config_sha256=replay_digest,
        )
        existing_ids = [_record_group_id(record) for record in existing]
        if existing_ids != expected_ids[: len(existing_ids)]:
            raise ValueError(
                "resume JSONL records must be an ordered prefix of expected group coverage"
            )
        if existing_completion is not None:
            validate_completion_artifact(existing, existing_completion)
            expected_completion = {
                "seed": args.seed,
                "compatibility_config_sha256": compatibility_digest,
                "capture_config_sha256": capture_digest,
                "replay_config_sha256": replay_digest,
                "expected_group_ids": expected_ids,
            }
            for field, expected in expected_completion.items():
                if existing_completion.get(field) != expected:
                    raise ValueError(
                        f"resume completion metadata has incompatible {field}"
                    )
            if seen != allowed_ids:
                raise ValueError("resume completion record is missing expected groups")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    args._data_metadata = data_metadata
    args._capture_payloads = capture_payloads
    args._compatibility_config = compatibility_config
    args._compatibility_config_sha256 = compatibility_digest
    args._capture_config = capture_config
    args._capture_config_sha256 = capture_digest
    args._replay_config = replay_config
    args._replay_config_sha256 = replay_digest
    args._capture_group_ids = [
        descriptor["group_id"] for descriptor in capture_descriptors
    ]
    args._capture_group_coordinates = [
        {"step": descriptor["step"], "fragment": descriptor["fragment"]}
        for descriptor in capture_descriptors
    ]
    args._full_shard_group_ids = [
        descriptor["group_id"] for descriptor in full_shard_descriptors
    ]
    args._expected_group_ids = expected_ids
    if existing_completion is not None:
        return []

    selected = [
        (index, group)
        for index, group in selected
        if capture_descriptors[index]["group_id"] not in seen
    ]
    if not selected:
        return []

    device = torch.device(args.device)
    learner_args = argparse.Namespace(
        model=args.model,
        tuning="lora",
        shard="ddp",
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_targets=args.lora_targets,
    )
    model, tokenizer = load_model_and_tokenizer(learner_args, device)
    params = trainable_params(model)
    layout = build_layout(
        [(name, param.numel()) for name, param in params.items()],
        args.fragments,
        args.fragment_pattern,
    )
    try:
        anchor_batches = probe_batches(
            args, tokenizer, device, args.anchor_data, args.anchor_batches
        )
        oracle_batches = probe_batches(
            args, tokenizer, device, args.oracle_data, args.oracle_batches
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if len(anchor_batches) % args.panel_size:
        raise SystemExit("anchor batch count must be divisible by --panel-size")
    for label, metadata in (
        ("anchor", data_metadata["anchor"]),
        ("oracle", data_metadata["oracle"]),
    ):
        if _sha256_file(Path(metadata["source"])) != metadata["source_sha256"]:
            raise SystemExit(f"{label} probe data changed while replay was starting")

    compute_loss = lambda logits, ids, weights: sft_loss(  # noqa: E731
        logits, ids, args.loss_function, weights
    )
    capture_root = root.expanduser().resolve()
    state_payloads = {
        (capture_root / item["path"]).resolve(): item
        for item in capture_payloads["state_checkpoints"]
    }
    candidate_payloads = {
        (capture_root / item["path"]).resolve(): item
        for item in capture_payloads["candidate_tensors"]
    }
    checkpoint_cache: dict[Path, Any] = {}
    records = []
    was_training = model.training
    model.eval()
    try:
        for completed, (group_index, group) in enumerate(selected, start=1):
            first = group[0]
            descriptor = capture_descriptors[group_index]
            state_path = buffered._resolve(root, first["state_checkpoint"]).resolve()
            state_payload = state_payloads.get(state_path)
            if state_payload is None:
                raise RuntimeError(
                    f"{state_path}: absent from capture payload manifest"
                )
            if (
                state_path.stat().st_size != state_payload["size_bytes"]
                or _sha256_file(state_path) != state_payload["sha256"]
            ):
                raise RuntimeError(f"{state_path}: changed after capture validation")
            checkpoint = parse_checkpoint(state_path)
            if int(checkpoint.global_step) != int(first["syncer_global_step"]):
                raise RuntimeError(
                    f"{state_path}: global step {checkpoint.global_step} does not match "
                    f"capture metadata {first['syncer_global_step']}"
                )
            syncer_eval._apply_checkpoint(checkpoint, layout, params, device)
            fid = int(first["fragment"])
            if fid >= len(checkpoint.fragments):
                raise RuntimeError(f"{state_path}: missing fragment {fid}")
            frag = layout.fragments[fid]
            current_version, current, momentum = checkpoint.fragments[fid]
            if int(current_version) != int(first["current_fragment_version"]):
                raise RuntimeError(
                    f"{state_path}: fragment version {current_version} does not match "
                    f"capture metadata {first['current_fragment_version']}"
                )
            validate_candidate_tensor(
                current, context=f"{state_path}: current fragment"
            )
            validate_candidate_tensor(momentum, context=f"{state_path}: momentum")
            merge_momentum = (
                momentum
                if args.delta_correction == "heloco"
                else torch.zeros_like(momentum)
            )
            candidates = []
            candidate_file_metadata = []
            for row in group:
                tensor_path = buffered._resolve(root, row["candidate_f32"]).resolve()
                candidate_payload = candidate_payloads.get(tensor_path)
                if candidate_payload is None:
                    raise RuntimeError(
                        f"{tensor_path}: absent from capture payload manifest"
                    )
                if (
                    tensor_path.stat().st_size != candidate_payload["size_bytes"]
                    or _sha256_file(tensor_path) != candidate_payload["sha256"]
                ):
                    raise RuntimeError(
                        f"{tensor_path}: changed after capture validation"
                    )
                tensor = buffered._read_f32(tensor_path, frag.numel)
                validate_candidate_tensor(tensor, context=str(tensor_path))
                candidate = buffered._candidate(
                    row, tensor, current, merge_momentum, int(current_version)
                )
                weight = candidate_weight(
                    row,
                    context=(
                        f"step={first['step']} fragment={fid} "
                        f"learner={row['learner_id']}"
                    ),
                )
                if not math.isclose(
                    float(candidate.weight), weight, rel_tol=0.0, abs_tol=0.0
                ):
                    raise RuntimeError("candidate helper changed the validated weight")
                if not math.isfinite(candidate.norm):
                    raise RuntimeError("candidate update norm is not finite")
                candidates.append(candidate)
                candidate_file_metadata.append(
                    {
                        "learner_id": int(row["learner_id"]),
                        "path": candidate_payload["path"],
                        "sha256": candidate_payload["sha256"],
                        "size_bytes": candidate_payload["size_bytes"],
                    }
                )

            anchor_current, anchor_current_batches = _validate_loss_output(
                *syncer_eval._losses(model, anchor_batches, compute_loss),
                expected_batches=len(anchor_batches),
                context="anchor current-state evaluation",
            )
            baseline_update = bn._production_merge_update(
                candidates, merge_momentum, frag
            )
            validate_candidate_tensor(
                baseline_update, context="production merge update"
            )
            baseline_trial = bn._nesterov_trial(
                current, momentum, baseline_update, args.outer_lr, args.outer_momentum
            )
            validate_candidate_tensor(
                baseline_trial, context="production baseline trial"
            )
            parity = _next_state_validation(
                root=root,
                next_checkpoint=next_checkpoints.get(_group_key(group)),
                checkpoint_cache=checkpoint_cache,
                checkpoint_metadata=state_payloads,
                fragment=fid,
                step=int(first["step"]),
                current=current,
                baseline_trial=baseline_trial,
                max_step_relative_error=args.max_next_state_step_relative_error,
            )
            anchor_baseline, anchor_baseline_batches = losses_for_trial(
                model,
                anchor_batches,
                compute_loss,
                frag,
                params,
                current,
                baseline_trial,
                device,
            )
            total_weight = math.fsum(candidate.weight for candidate in candidates)
            if not math.isfinite(total_weight) or total_weight <= 0.0:
                raise RuntimeError("candidate group has no finite positive weight")
            baseline_step_norm = float((baseline_trial - current).norm().item())
            actions = []
            action_trials: dict[int, torch.Tensor] = {}
            for dropped in candidates:
                dropped_learner = int(dropped.row["learner_id"])
                selected_candidates = [
                    candidate
                    for candidate in candidates
                    if int(candidate.row["learner_id"]) != dropped_learner
                ]
                selected_weight = math.fsum(
                    candidate.weight for candidate in selected_candidates
                )
                selected_mass = selected_weight / total_weight
                update = bn._production_merge_update(
                    selected_candidates, merge_momentum, frag
                )
                validate_candidate_tensor(update, context="leave-one-out merge update")
                raw_trial = bn._nesterov_trial(
                    current, momentum, update, args.outer_lr, args.outer_momentum
                )
                trial, scale, scale_ok = norm_matched_trial(
                    current,
                    raw_trial,
                    baseline_trial,
                    min_scale=args.min_norm_scale,
                    max_scale=args.max_norm_scale,
                )
                anchor_loss, anchor_by_batch = losses_for_trial(
                    model,
                    anchor_batches,
                    compute_loss,
                    frag,
                    params,
                    current,
                    trial,
                    device,
                )
                decision = paired_decision(
                    anchor_baseline_batches,
                    anchor_by_batch,
                    panel_size=args.panel_size,
                    min_gain=args.min_gain,
                    lcb_z=args.lcb_z,
                    min_win_rate=args.min_win_rate,
                )
                step_norm = float((trial - current).norm().item())
                if baseline_step_norm <= 1e-12:
                    step_ratio = 1.0 if step_norm <= 1e-12 else float("inf")
                else:
                    step_ratio = step_norm / baseline_step_norm
                valid = (
                    math.isfinite(selected_mass)
                    and selected_mass >= args.min_selected_mass
                    and scale_ok
                    and math.isfinite(step_ratio)
                    and abs(step_ratio - 1.0) <= args.max_step_ratio_error
                )
                decision["eligible"] = bool(decision["eligible"] and valid)
                anchor_gain = utility_estimate(anchor_baseline_batches, anchor_by_batch)
                action_trials[dropped_learner] = trial
                actions.append(
                    {
                        "name": f"drop_learner_{dropped_learner}",
                        "dropped_learner": dropped_learner,
                        "selected_mass": selected_mass,
                        "norm_scale": scale,
                        "step_ratio": step_ratio,
                        "valid": valid,
                        "anchor_loss": anchor_loss,
                        "anchor_losses": anchor_by_batch,
                        "anchor_gain_vs_baseline": anchor_gain["center"],
                        "decision": decision,
                    }
                )

            eligible = [action for action in actions if action["decision"]["eligible"]]
            chosen = (
                max(
                    eligible,
                    key=lambda action: (
                        float(action["decision"]["lcb"]),
                        float(action["decision"]["gain"]),
                        float(action["decision"]["win_rate"]),
                        -int(action["dropped_learner"]),
                    ),
                )
                if eligible
                else None
            )
            random_action = deterministic_random_valid_action(
                actions,
                random_seed=args.random_seed,
                seed=args.seed,
                stable_group_id=descriptor["group_id"],
            )
            valid_actions = [action for action in actions if action["valid"]]

            # Freeze anchor-only selection and the random control before any
            # held-out oracle loss is evaluated.
            oracle_current, oracle_current_batches = _validate_loss_output(
                *syncer_eval._losses(model, oracle_batches, compute_loss),
                expected_batches=len(oracle_batches),
                context="oracle current-state evaluation",
            )
            oracle_baseline, oracle_baseline_batches = losses_for_trial(
                model,
                oracle_batches,
                compute_loss,
                frag,
                params,
                current,
                baseline_trial,
                device,
            )
            for action in actions:
                trial = action_trials[int(action["dropped_learner"])]
                oracle_loss, oracle_by_batch = losses_for_trial(
                    model,
                    oracle_batches,
                    compute_loss,
                    frag,
                    params,
                    current,
                    trial,
                    device,
                )
                oracle_gain = utility_estimate(oracle_baseline_batches, oracle_by_batch)
                oracle_utility = utility_estimate(
                    oracle_current_batches, oracle_by_batch
                )
                action.update(
                    {
                        "oracle_loss": oracle_loss,
                        "oracle_losses": oracle_by_batch,
                        "oracle_gain_vs_baseline": oracle_gain["center"],
                        "oracle_utility": oracle_utility["center"],
                        "oracle_utility_se": oracle_utility["se"],
                    }
                )
            best_oracle = (
                max(
                    valid_actions,
                    key=lambda action: (
                        float(action["oracle_gain_vs_baseline"]),
                        -int(action["dropped_learner"]),
                    ),
                )
                if valid_actions
                else None
            )
            chosen_oracle_loss = (
                oracle_baseline if chosen is None else chosen["oracle_loss"]
            )
            chosen_oracle_losses = (
                oracle_baseline_batches if chosen is None else chosen["oracle_losses"]
            )
            baseline_utility = utility_estimate(
                oracle_current_batches, oracle_baseline_batches
            )
            chosen_utility = utility_estimate(
                oracle_current_batches, chosen_oracle_losses
            )
            chosen_gain = utility_estimate(
                oracle_baseline_batches, chosen_oracle_losses
            )["center"]
            record = {
                "schema": REPLAY_SCHEMA,
                "compatibility_config_sha256": compatibility_digest,
                "capture_config_sha256": capture_digest,
                "replay_config_sha256": replay_digest,
                "seed": args.seed,
                "group_id": descriptor["group_id"],
                "group_index": group_index,
                "group_ordinal": group_index + 1,
                "step": int(first["step"]),
                "syncer_global_step": int(first["syncer_global_step"]),
                "fragment": fid,
                "current_fragment_version": int(current_version),
                "state_checkpoint": str(first["state_checkpoint"]),
                "state_checkpoint_sha256": state_payload["sha256"],
                "state_checkpoint_size_bytes": state_payload["size_bytes"],
                "capture_payload_manifest_sha256": capture_payloads["manifest_sha256"],
                "candidate_count": len(candidates),
                "candidate_learner_ids": [
                    int(candidate.row["learner_id"]) for candidate in candidates
                ],
                "candidate_base_versions": [
                    int(candidate.row["base_version"]) for candidate in candidates
                ],
                "candidate_weights": [candidate.weight for candidate in candidates],
                "candidate_files": candidate_file_metadata,
                "total_candidate_weight": total_weight,
                "anchor_file_sha256": data_metadata["anchor"]["source_sha256"],
                "oracle_file_sha256": data_metadata["oracle"]["source_sha256"],
                "anchor_rows_sha256": data_metadata["anchor"]["canonical_rows_sha256"],
                "oracle_rows_sha256": data_metadata["oracle"]["canonical_rows_sha256"],
                "anchor_oracle_disjoint": True,
                "anchor_oracle_overlap_count": 0,
                "utility_estimand": "mean_paired_batch_loss_difference",
                "anchor_current_loss": anchor_current,
                "anchor_current_losses": anchor_current_batches,
                "anchor_baseline_loss": anchor_baseline,
                "anchor_baseline_losses": anchor_baseline_batches,
                "oracle_current_loss": oracle_current,
                "oracle_current_losses": oracle_current_batches,
                "oracle_baseline_loss": oracle_baseline,
                "oracle_baseline_losses": oracle_baseline_batches,
                "baseline_oracle_utility": baseline_utility["center"],
                "baseline_oracle_utility_se": baseline_utility["se"],
                "baseline_oracle_negative": baseline_utility["center"] < 0.0,
                "baseline_oracle_strict_negative": (
                    None
                    if baseline_utility["se"] is None
                    else baseline_utility["center"] + baseline_utility["se"] < 0.0
                ),
                "chosen_action": BASELINE_ACTION if chosen is None else chosen["name"],
                "chosen_oracle_loss": chosen_oracle_loss,
                "chosen_oracle_utility": chosen_utility["center"],
                "chosen_oracle_utility_se": chosen_utility["se"],
                "chosen_oracle_negative": chosen_utility["center"] < 0.0,
                "chosen_oracle_strict_negative": (
                    None
                    if chosen_utility["se"] is None
                    else chosen_utility["center"] + chosen_utility["se"] < 0.0
                ),
                "chosen_gain_vs_baseline": chosen_gain,
                "best_loo_oracle_gain": (
                    0.0
                    if best_oracle is None
                    else best_oracle["oracle_gain_vs_baseline"]
                ),
                "best_loo_oracle_action": (
                    BASELINE_ACTION if best_oracle is None else best_oracle["name"]
                ),
                "random_control_algorithm": "sha256_mod_valid_actions_v1",
                "random_valid_action_count": len(valid_actions),
                "random_loo_action": (
                    BASELINE_ACTION if random_action is None else random_action["name"]
                ),
                "random_loo_oracle_gain": (
                    0.0
                    if random_action is None
                    else random_action["oracle_gain_vs_baseline"]
                ),
                "actions": actions,
                **parity,
            }
            records.append(record)
            _write_jsonl_line(args._sink, record)
            if args.progress_every and (
                completed == 1 or completed % args.progress_every == 0
            ):
                print(
                    f"[exact-loo] groups={completed}/{len(selected)} "
                    f"step={record['step']} action={record['chosen_action']}",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        model.train(was_training)
    return records


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return float("nan")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def summarize(records: list[dict]) -> dict:
    if not records:
        return {
            "schema": SUMMARY_SCHEMA,
            "records": 0,
            "seeds": [],
            "mean_gain_vs_baseline": None,
            "per_seed": {},
        }
    gains = [float(record["chosen_gain_vs_baseline"]) for record in records]
    random_gains = [float(record["random_loo_oracle_gain"]) for record in records]
    oracle_headroom = [
        max(0.0, float(record["best_loo_oracle_gain"])) for record in records
    ]
    if any(
        not math.isfinite(value) for value in gains + random_gains + oracle_headroom
    ):
        raise ValueError("replay summary inputs must be finite")
    captured = [
        gain / headroom
        for gain, headroom in zip(gains, oracle_headroom)
        if headroom > 0.0
    ]
    baseline_negative = mean(
        [1.0 if record["baseline_oracle_negative"] else 0.0 for record in records]
    )
    chosen_negative = mean(
        [1.0 if record["chosen_oracle_negative"] else 0.0 for record in records]
    )
    baseline_strict_rows = [
        record
        for record in records
        if record["baseline_oracle_strict_negative"] is not None
    ]
    chosen_strict_rows = [
        record
        for record in records
        if record["chosen_oracle_strict_negative"] is not None
    ]
    baseline_strict = (
        None
        if not baseline_strict_rows
        else mean(
            [
                1.0 if record["baseline_oracle_strict_negative"] else 0.0
                for record in baseline_strict_rows
            ]
        )
    )
    chosen_strict = (
        None
        if not chosen_strict_rows
        else mean(
            [
                1.0 if record["chosen_oracle_strict_negative"] else 0.0
                for record in chosen_strict_rows
            ]
        )
    )
    seeds = sorted({int(record["seed"]) for record in records})
    per_seed = {}
    for seed in seeds:
        seed_records = [record for record in records if int(record["seed"]) == seed]
        seed_gains = [
            float(record["chosen_gain_vs_baseline"]) for record in seed_records
        ]
        per_seed[str(seed)] = {
            "records": len(seed_records),
            "mean_gain_vs_baseline": mean(seed_gains),
            "action_rate": mean(
                [record["chosen_action"] != BASELINE_ACTION for record in seed_records]
            ),
            "mean_random_loo_gain": mean(
                [float(record["random_loo_oracle_gain"]) for record in seed_records]
            ),
        }
    step_errors = [
        float(record["production_baseline_next_state_step_relative_error"])
        for record in records
        if record.get("production_baseline_next_state_available")
    ]
    return {
        "schema": SUMMARY_SCHEMA,
        "records": len(records),
        "seeds": seeds,
        "per_seed": per_seed,
        "utility_estimand": "mean_paired_batch_loss_difference",
        "mean_gain_vs_baseline": mean(gains),
        "median_gain_vs_baseline": _median(gains),
        "all_seed_records_mean_positive": all(
            item["mean_gain_vs_baseline"] > 0.0 for item in per_seed.values()
        ),
        "action_rate": mean(
            [
                1.0 if record["chosen_action"] != BASELINE_ACTION else 0.0
                for record in records
            ]
        ),
        "baseline_negative_rate": baseline_negative,
        "chosen_negative_rate": chosen_negative,
        "negative_rate_relative_drop": (
            None
            if baseline_negative <= 0.0
            else (baseline_negative - chosen_negative) / baseline_negative
        ),
        "baseline_strict_negative_rate": baseline_strict,
        "chosen_strict_negative_rate": chosen_strict,
        "strict_negative_rate_relative_drop": (
            None
            if baseline_strict is None
            or chosen_strict is None
            or baseline_strict <= 0.0
            else (baseline_strict - chosen_strict) / baseline_strict
        ),
        "mean_oracle_loo_headroom": mean(oracle_headroom),
        "mean_headroom_captured": None if not captured else mean(captured),
        "headroom_excluded_fraction": 1.0 - len(captured) / len(records),
        "mean_random_loo_gain": mean(random_gains),
        "next_state_validation": {
            "available_records": len(step_errors),
            "unavailable_records": len(records) - len(step_errors),
            "max_step_relative_error": max(step_errors) if step_errors else None,
        },
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--anchor-data", required=True, type=Path)
    parser.add_argument("--oracle-data", required=True, type=Path)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lora-r", type=int, default=2)
    parser.add_argument("--lora-alpha", type=int, default=4)
    parser.add_argument(
        "--lora-targets",
        choices=["auto", "attention", "all-linear"],
        default="auto",
    )
    parser.add_argument("--fragments", type=int, default=4)
    parser.add_argument(
        "--fragment-pattern", choices=["binpack", "strided"], default="binpack"
    )
    parser.add_argument("--loss-function", default="cross_entropy")
    parser.add_argument("--train-on", choices=["assistant", "all"], default="assistant")
    parser.add_argument("--anchor-batches", type=int, default=16)
    parser.add_argument("--oracle-batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=256)
    parser.add_argument("--panel-size", type=int, default=2)
    parser.add_argument("--outer-lr", type=float, default=0.28)
    parser.add_argument("--outer-momentum", type=float, default=0.0)
    parser.add_argument(
        "--delta-correction", choices=["heloco", "none"], default="none"
    )
    parser.add_argument("--expected-candidates", type=int, default=4)
    parser.add_argument("--expected-groups", type=int, default=DEFAULT_EXPECTED_GROUPS)
    parser.add_argument("--min-selected-mass", type=float, default=0.70)
    parser.add_argument("--min-norm-scale", type=float, default=0.5)
    parser.add_argument("--max-norm-scale", type=float, default=2.0)
    parser.add_argument("--max-step-ratio-error", type=float, default=0.01)
    parser.add_argument(
        "--max-next-state-step-relative-error",
        type=float,
        default=DEFAULT_MAX_NEXT_STATE_STEP_RELATIVE_ERROR,
    )
    parser.add_argument("--min-gain", type=float, default=0.00025)
    parser.add_argument("--lcb-z", type=float, default=2.365)
    parser.add_argument("--min-win-rate", type=float, default=0.75)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--group-start", type=int, default=0)
    parser.add_argument("--group-stride", type=int, default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--out-jsonl", required=True, type=Path)
    parser.add_argument("--out-summary", required=True, type=Path)
    args = parser.parse_args(argv)
    args.seed = args.seed if args.seed is not None else infer_seed(args.capture_dir)
    if args.seed is None:
        parser.error("could not infer seed; pass --seed")
    if args.anchor_batches < 2 or args.oracle_batches < 2:
        parser.error("anchor and oracle batch counts must be at least 2")
    if args.panel_size < 1 or args.anchor_batches % args.panel_size:
        parser.error("--anchor-batches must be divisible by positive --panel-size")
    if args.batch_size < 1 or args.max_rows < 1 or args.seq_len < 2:
        parser.error("batch size/max rows must be positive and seq len must exceed 1")
    if args.fragments < 1 or args.lora_r < 1 or args.lora_alpha <= 0:
        parser.error("fragment count and LoRA rank/alpha must be positive")
    if args.expected_candidates < 2:
        parser.error("--expected-candidates must be at least 2")
    if args.expected_groups is not None and args.expected_groups < 1:
        parser.error("--expected-groups must be positive")
    if args.group_stride < 1:
        parser.error("--group-stride must be >= 1")
    if not 0 <= args.group_start < args.group_stride:
        parser.error("--group-start must be in [0, group-stride)")
    if args.max_groups is not None and args.max_groups < 1:
        parser.error("--max-groups must be positive")
    finite_values = {
        "--outer-lr": args.outer_lr,
        "--outer-momentum": args.outer_momentum,
        "--min-selected-mass": args.min_selected_mass,
        "--min-norm-scale": args.min_norm_scale,
        "--max-norm-scale": args.max_norm_scale,
        "--max-step-ratio-error": args.max_step_ratio_error,
        "--max-next-state-step-relative-error": args.max_next_state_step_relative_error,
        "--min-gain": args.min_gain,
        "--lcb-z": args.lcb_z,
        "--min-win-rate": args.min_win_rate,
    }
    for flag, value in finite_values.items():
        if not math.isfinite(value):
            parser.error(f"{flag} must be finite")
    if args.outer_lr <= 0.0:
        parser.error("--outer-lr must be positive")
    if not 0.0 < args.min_selected_mass <= 1.0:
        parser.error("--min-selected-mass must be in (0, 1]")
    if args.min_norm_scale <= 0.0 or args.max_norm_scale < args.min_norm_scale:
        parser.error("norm scale bounds must be positive and ordered")
    if args.max_step_ratio_error < 0.0:
        parser.error("--max-step-ratio-error must be nonnegative")
    if args.max_next_state_step_relative_error <= 0.0:
        parser.error("--max-next-state-step-relative-error must be positive")
    if args.min_gain < 0.0 or args.lcb_z < 0.0:
        parser.error("selection gain and LCB multiplier must be nonnegative")
    if not 0.0 <= args.min_win_rate <= 1.0:
        parser.error("--min-win-rate must be in [0, 1]")
    if args.out_jsonl.resolve() == args.out_summary.resolve():
        parser.error("--out-jsonl and --out-summary must be different paths")
    return args


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(jsonable(dict(value)), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.resume and args.out_jsonl.exists():
        try:
            existing, existing_completion = read_replay_artifact(args.out_jsonl)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    else:
        existing, existing_completion = [], None
    args._existing_records = existing
    args._existing_completion = existing_completion
    if existing_completion is None and args.out_summary.exists():
        args.out_summary.unlink()
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and args.out_jsonl.exists() else "w"
    with args.out_jsonl.open(mode, encoding="utf-8") as sink:
        args._sink = sink
        new_records = replay(args)
        order = {
            identifier: index
            for index, identifier in enumerate(args._expected_group_ids)
        }
        records = sorted(
            existing + new_records,
            key=lambda record: order[_record_group_id(record)],
        )
        completion = existing_completion
        if completion is None:
            completion = make_completion_metadata(records, args)
            _write_jsonl_line(sink, completion)
    try:
        validate_completion_artifact(records, completion)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    summary = summarize(records)
    summary["selection"] = args._compatibility_config["selection"]
    summary["group_shard"] = args._replay_config["group_shard"]
    summary["data"] = args._data_metadata
    summary["capture_payloads"] = args._capture_payloads
    summary["compatibility_config"] = args._compatibility_config
    summary["compatibility_config_sha256"] = args._compatibility_config_sha256
    summary["capture_config"] = args._capture_config
    summary["capture_config_sha256"] = args._capture_config_sha256
    summary["replay_config"] = args._replay_config
    summary["replay_config_sha256"] = args._replay_config_sha256
    summary["completion"] = completion
    _atomic_write_json(args.out_summary, summary)
    print(json.dumps(jsonable(summary), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
