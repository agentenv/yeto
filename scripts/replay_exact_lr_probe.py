#!/usr/bin/env python3
"""Replay a direct outer-SGD line search on captured complete groups.

Each action applies a scalar outer-LR multiplier to the exact production
token-weighted group merge with delta correction disabled. Selection uses
only paired, deterministic anchor panels and falls back to multiplier 1.0.
The selected action and every fixed action are then measured on a disjoint
oracle panel set for offline policy and headroom analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

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

from yeto.data import build_packed_dataset, load_rows  # noqa: E402
from yeto.export import parse_checkpoint  # noqa: E402
from yeto.fragments import MERGE_AVG, build_layout  # noqa: E402
from yeto.learner import load_model_and_tokenizer, trainable_params  # noqa: E402
from yeto.losses import sft_loss  # noqa: E402


REPLAY_SCHEMA = "exact_lr_probe_replay_v1"
SUMMARY_SCHEMA = "exact_lr_probe_summary_v1"
BASELINE_MULTIPLIER = 1.0
DEFAULT_MULTIPLIERS = (0.50, 0.75, 0.875, 1.0, 1.125, 1.25, 1.5)
DEFAULT_MIN_GAIN = 0.00025
DEFAULT_LCB_Z = 2.365
DEFAULT_MIN_WIN_RATE = 0.75
CANONICALIZATION = "messages-tools-json-v1"


def mean(values: Sequence[float]) -> float:
    checked = [float(value) for value in values]
    if not checked:
        raise ValueError("cannot compute a mean from an empty series")
    if not all(math.isfinite(value) for value in checked):
        raise ValueError("mean input contains a non-finite value")
    return sum(checked) / len(checked)


def std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(
        sum((float(value) - center) ** 2 for value in values) / (len(values) - 1)
    )


def utility_se(base: Sequence[float], trial: Sequence[float]) -> float | None:
    left = [float(value) for value in base]
    right = [float(value) for value in trial]
    if not left or len(left) != len(right):
        raise ValueError("paired loss series must be non-empty and equal length")
    if not all(math.isfinite(value) and value >= 0.0 for value in left + right):
        raise ValueError("paired loss series contains an invalid value")
    gains = [base_loss - trial_loss for base_loss, trial_loss in zip(left, right)]
    return None if len(gains) < 2 else std(gains) / math.sqrt(len(gains))


def paired_decision(
    baseline_losses: Sequence[float],
    action_losses: Sequence[float],
    *,
    min_gain: float,
    lcb_z: float,
    min_win_rate: float,
) -> dict:
    baseline = [float(value) for value in baseline_losses]
    action = [float(value) for value in action_losses]
    if not baseline or len(baseline) != len(action):
        raise ValueError("paired loss series must be non-empty and equal length")
    if not all(math.isfinite(value) and value >= 0.0 for value in baseline + action):
        raise ValueError("paired loss series contains an invalid value")
    panel_gains = [left - right for left, right in zip(baseline, action)]
    gain = mean(panel_gains)
    standard_error = (
        0.0 if len(panel_gains) < 2 else std(panel_gains) / math.sqrt(len(panel_gains))
    )
    lcb = gain - lcb_z * standard_error
    win_rate = sum(value > 0.0 for value in panel_gains) / len(panel_gains)
    return {
        "gain": gain,
        "se": standard_error,
        "lcb": lcb,
        "win_rate": win_rate,
        "panels": panel_gains,
        "eligible": gain >= min_gain and lcb > 0.0 and win_rate >= min_win_rate,
    }


def jsonable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def infer_seed(path: Path) -> int | None:
    match = re.search(r"seed(\d+)", str(path))
    return None if match is None else int(match.group(1))


def losses_for_trial(
    model,
    panels,
    compute_loss,
    frag,
    params,
    current: torch.Tensor,
    trial: torch.Tensor,
    device,
) -> tuple[float, list[float]]:
    from yeto.tensor_io import apply_fragment

    apply_fragment(frag, trial.to(device), params)
    try:
        return syncer_eval._losses(model, panels, compute_loss)
    finally:
        apply_fragment(frag, current.to(device), params)


def multiplier_key(multiplier: float) -> str:
    return format(float(multiplier), ".12g")


def action_name(multiplier: float) -> str:
    return f"sgd_outer_lr_x{multiplier_key(multiplier)}"


def normalize_multipliers(values: str | Sequence[float]) -> tuple[float, ...]:
    if isinstance(values, str):
        raw = [value.strip() for value in values.split(",") if value.strip()]
    else:
        raw = list(values)
    if not raw:
        raise ValueError("at least one outer-LR multiplier is required")
    parsed = []
    for value in raw:
        multiplier = float(value)
        if not math.isfinite(multiplier) or multiplier <= 0.0:
            raise ValueError("outer-LR multipliers must be finite and positive")
        if math.isclose(multiplier, BASELINE_MULTIPLIER, rel_tol=0.0, abs_tol=1e-12):
            multiplier = BASELINE_MULTIPLIER
        parsed.append(multiplier)
    if len(set(parsed)) != len(parsed):
        raise ValueError("outer-LR multipliers must be unique")
    if BASELINE_MULTIPLIER not in parsed:
        raise ValueError("outer-LR multipliers must include fallback multiplier 1.0")
    return tuple(sorted(parsed))


def _coerce_loss_map(
    losses_by_multiplier: Mapping[object, Sequence[float]],
) -> dict[float, tuple[float, ...]]:
    normalized: dict[float, tuple[float, ...]] = {}
    for raw_multiplier, raw_losses in losses_by_multiplier.items():
        if isinstance(raw_multiplier, str) and raw_multiplier.startswith(
            "sgd_outer_lr_x"
        ):
            raw_multiplier = raw_multiplier.removeprefix("sgd_outer_lr_x")
        multiplier = float(raw_multiplier)
        if multiplier in normalized:
            raise ValueError(f"duplicate loss series for multiplier {multiplier}")
        normalized[multiplier] = tuple(float(value) for value in raw_losses)
    return normalized


def _selection_order(decision: dict) -> tuple[float, float, float, float, float]:
    multiplier = float(decision["multiplier"])
    return (
        -float(decision["lcb"]),
        -float(decision["mean_gain"]),
        -float(decision["win_rate"]),
        abs(multiplier - BASELINE_MULTIPLIER),
        multiplier,
    )


def select_multiplier(
    losses_by_multiplier: Mapping[object, Sequence[float]],
    multipliers: Sequence[float] | None = None,
    *,
    min_gain: float = DEFAULT_MIN_GAIN,
    lcb_z: float = DEFAULT_LCB_Z,
    min_win_rate: float = DEFAULT_MIN_WIN_RATE,
    min_panels: int = 2,
) -> dict:
    """Select from paired anchor losses without accepting oracle inputs.

    Alternatives must pass the frozen mean, LCB, and panel win-rate gates.
    Eligible ties are resolved by LCB, mean gain, win rate, distance from the
    fallback, then the lower multiplier. Malformed losses fail closed to 1.0.
    """

    try:
        normalized = _coerce_loss_map(losses_by_multiplier)
        expected = normalize_multipliers(
            tuple(normalized) if multipliers is None else multipliers
        )
        if set(normalized) != set(expected):
            raise ValueError("loss map does not match the configured multipliers")
        panel_count = len(normalized[BASELINE_MULTIPLIER])
        if panel_count < min_panels:
            raise ValueError(f"need at least {min_panels} complete anchor panels")
        for multiplier in expected:
            losses = normalized[multiplier]
            if len(losses) != panel_count:
                raise ValueError("all actions must use the same anchor panels")
            if any(not math.isfinite(value) for value in losses):
                raise ValueError("anchor losses must be finite")
    except (KeyError, TypeError, ValueError):
        return {
            "chosen_multiplier": BASELINE_MULTIPLIER,
            "chosen_action": action_name(BASELINE_MULTIPLIER),
            "fallback_reason": "invalid_anchor_losses",
            "decisions": [],
        }

    baseline_losses = list(normalized[BASELINE_MULTIPLIER])
    decisions = []
    for multiplier in expected:
        if multiplier == BASELINE_MULTIPLIER:
            decision = {
                "multiplier": multiplier,
                "action": action_name(multiplier),
                "mean_gain": 0.0,
                "standard_error": 0.0,
                "lcb": 0.0,
                "win_rate": 0.0,
                "wins": 0,
                "panels": panel_count,
                "eligible": False,
                "is_fallback": True,
            }
        else:
            stats = paired_decision(
                baseline_losses,
                list(normalized[multiplier]),
                min_gain=min_gain,
                lcb_z=lcb_z,
                min_win_rate=min_win_rate,
            )
            decision = {
                "multiplier": multiplier,
                "action": action_name(multiplier),
                "mean_gain": float(stats["gain"]),
                "standard_error": float(stats["se"]),
                "lcb": float(stats["lcb"]),
                "win_rate": float(stats["win_rate"]),
                "wins": sum(value > 0.0 for value in stats["panels"]),
                "panels": len(stats["panels"]),
                "panel_gains": list(stats["panels"]),
                "eligible": bool(stats["eligible"]),
                "is_fallback": False,
            }
        decisions.append(decision)

    eligible = [decision for decision in decisions if decision["eligible"]]
    if not eligible:
        chosen = BASELINE_MULTIPLIER
        fallback_reason = "no_action_passed"
    else:
        chosen = float(sorted(eligible, key=_selection_order)[0]["multiplier"])
        fallback_reason = None
    return {
        "chosen_multiplier": chosen,
        "chosen_action": action_name(chosen),
        "fallback_reason": fallback_reason,
        "decisions": decisions,
    }


def _canonical_row(row, *, context: str) -> dict:
    if not isinstance(row, Mapping):
        try:
            row = dict(row)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context}: expected an object row") from exc
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{context}: expected a non-empty messages list")
    canonical = {"messages": messages}
    if row.get("tools"):
        canonical["tools"] = row["tools"]
    try:
        return json.loads(json.dumps(canonical, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: row is not finite JSON") from exc


def _canonical_payload(row: Mapping) -> bytes:
    return json.dumps(
        row,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_row_hash(row: Mapping) -> str:
    return hashlib.sha256(_canonical_payload(row)).hexdigest()


def anchor_oracle_disjointness_metadata(
    anchor_row_hashes: Sequence[str], oracle_row_hashes: Sequence[str]
) -> dict:
    anchor = tuple(str(value) for value in anchor_row_hashes)
    oracle = tuple(str(value) for value in oracle_row_hashes)
    overlap = sorted(set(anchor) & set(oracle))
    return {
        "canonicalization": CANONICALIZATION,
        "anchor_row_count": len(anchor),
        "anchor_unique_row_count": len(set(anchor)),
        "oracle_row_count": len(oracle),
        "oracle_unique_row_count": len(set(oracle)),
        "overlap_count": len(overlap),
        "verified_disjoint": not overlap,
        "overlap_sha256": hashlib.sha256("\n".join(overlap).encode()).hexdigest(),
    }


def target_block_indices(weights: torch.Tensor, count: int) -> list[int]:
    """Return the first packed blocks that contain supervised target tokens."""

    if weights.ndim != 2:
        raise ValueError("packed weights must be a rank-2 tensor")
    if count < 1:
        raise ValueError("target block count must be positive")
    indices = [
        index
        for index in range(weights.shape[0])
        if float(weights[index, 1:].sum().item()) > 0.0
    ]
    if len(indices) < count:
        raise ValueError(
            f"only {len(indices)} packed blocks contain target tokens; need {count}"
        )
    return indices[:count]


def build_row_panels(
    data: Path,
    tokenizer,
    *,
    seq_len: int,
    panel_count: int,
    blocks_per_panel: int,
    max_rows: int | None,
    train_on: str,
    device: torch.device | str,
) -> dict:
    """Pack deterministic round-robin row clusters into statistical panels."""

    if seq_len <= 1:
        raise ValueError("seq_len must be greater than 1")
    if panel_count < 2:
        raise ValueError("panel_count must be at least 2")
    if blocks_per_panel < 1:
        raise ValueError("blocks_per_panel must be positive")
    source = load_rows(str(data))
    source_count = len(source)
    selected_count = source_count if max_rows is None else min(source_count, max_rows)
    if selected_count < panel_count:
        raise ValueError(
            f"{data}: selected {selected_count} rows for {panel_count} panels"
        )
    rows = [
        _canonical_row(source[index], context=f"{data}:row {index}")
        for index in range(selected_count)
    ]
    row_hashes = tuple(canonical_row_hash(row) for row in rows)
    if len(set(row_hashes)) != len(row_hashes):
        raise ValueError(f"{data}: canonical duplicate rows would cross panel units")

    panels = []
    panel_row_counts = []
    panel_block_indices = []
    assignment_digest = hashlib.sha256()
    tensor_digest = hashlib.sha256()
    for panel_id in range(panel_count):
        row_indices = list(range(panel_id, selected_count, panel_count))
        panel_rows = [rows[index] for index in row_indices]
        panel_row_counts.append(len(panel_rows))
        assignment_digest.update(
            json.dumps(
                {
                    "panel": panel_id,
                    "indices": row_indices,
                    "row_hashes": [row_hashes[index] for index in row_indices],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        try:
            packed = build_packed_dataset(
                panel_rows,
                tokenizer,
                learner_id=0,
                num_learners=1,
                seq_len=seq_len,
                max_rows=len(panel_rows),
                train_on=train_on,
            )
        except ValueError as exc:
            raise ValueError(
                f"{data}: panel {panel_id} cannot form blocks: {exc}"
            ) from exc
        try:
            block_indices = target_block_indices(packed.weights, blocks_per_panel)
        except ValueError as exc:
            raise ValueError(f"{data}: panel {panel_id}: {exc}") from exc
        panel_block_indices.append(block_indices)
        ids = packed.blocks[block_indices].contiguous()
        weights = packed.weights[block_indices].contiguous()
        tensor_digest.update(ids.view(torch.uint8).numpy().tobytes())
        tensor_digest.update(weights.view(torch.uint8).numpy().tobytes())
        panels.append(
            (
                ids.to(device=device, non_blocking=False),
                weights.to(device=device, non_blocking=False),
            )
        )

    canonical_digest = hashlib.sha256()
    for row_hash in row_hashes:
        canonical_digest.update(row_hash.encode("ascii") + b"\n")
    metadata = {
        "source": str(data.expanduser().resolve()),
        "source_row_count": source_count,
        "selected_row_count": selected_count,
        "canonicalization": CANONICALIZATION,
        "canonical_rows_sha256": canonical_digest.hexdigest(),
        "panel_assignment": "round_robin_source_row",
        "panel_assignment_sha256": assignment_digest.hexdigest(),
        "panel_tensors_sha256": tensor_digest.hexdigest(),
        "panel_count": panel_count,
        "blocks_per_panel": blocks_per_panel,
        "panel_row_counts": panel_row_counts,
        "panel_block_indices": panel_block_indices,
        "seq_len": seq_len,
        "train_on": train_on,
    }
    return {
        "panels": tuple(panels),
        "row_hashes": row_hashes,
        "metadata": metadata,
    }


def build_panel_metadata(anchor: dict, oracle: dict) -> dict:
    disjointness = anchor_oracle_disjointness_metadata(
        anchor["row_hashes"], oracle["row_hashes"]
    )
    return {
        "anchor": anchor["metadata"],
        "oracle": oracle["metadata"],
        "disjointness": disjointness,
    }


def _record_panel_metadata(panel_metadata: dict) -> dict:
    return {
        "anchor_panels_sha256": panel_metadata["anchor"]["panel_tensors_sha256"],
        "oracle_panels_sha256": panel_metadata["oracle"]["panel_tensors_sha256"],
        "anchor_rows_sha256": panel_metadata["anchor"]["canonical_rows_sha256"],
        "oracle_rows_sha256": panel_metadata["oracle"]["canonical_rows_sha256"],
        "anchor_oracle_disjoint": panel_metadata["disjointness"]["verified_disjoint"],
        "anchor_oracle_overlap_count": panel_metadata["disjointness"]["overlap_count"],
    }


def validate_candidate_groups(
    rows: list[dict], expected_candidates: int
) -> list[list[dict]]:
    if expected_candidates < 1:
        raise ValueError("expected_candidates must be positive")
    groups = buffered._group_rows(rows, 1)
    if not groups:
        raise ValueError("capture contains no candidate groups")
    problems = []
    for group in groups:
        first = group[0]
        learner_ids = [int(row["learner_id"]) for row in group]
        if len(group) != expected_candidates:
            problems.append(
                f"step={first['step']} fragment={first['fragment']} has "
                f"{len(group)} candidates, expected {expected_candidates}"
            )
        elif len(set(learner_ids)) != expected_candidates:
            problems.append(
                f"step={first['step']} fragment={first['fragment']} has duplicate "
                f"learner IDs {learner_ids}"
            )
    if problems:
        preview = "; ".join(problems[:5])
        suffix = "" if len(problems) <= 5 else f"; plus {len(problems) - 5} more"
        raise ValueError(f"candidate group validation failed: {preview}{suffix}")
    return groups


def exact_group_delta(candidates, frag) -> torch.Tensor:
    """Return the production per-tensor token-weighted merge without HeLoCo."""

    if not candidates:
        raise ValueError("cannot merge an empty candidate group")
    zero_correction = torch.zeros_like(candidates[0].update)
    return bn._production_merge_update(candidates, zero_correction, frag)


def _step_ratio(step_norm: float, baseline_step_norm: float) -> float:
    if baseline_step_norm > 1e-12:
        return step_norm / baseline_step_norm
    return 1.0 if step_norm <= 1e-12 else float("inf")


def _record_key(record: Mapping) -> tuple[str, int, int]:
    return (
        str(record["state_checkpoint"]),
        int(record["step"]),
        int(record["fragment"]),
    )


def _group_key(group: Sequence[Mapping]) -> tuple[str, int, int]:
    first = group[0]
    return (
        str(first["state_checkpoint"]),
        int(first["step"]),
        int(first["fragment"]),
    )


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"{path}:{line_number}: malformed JSON: {exc}"
                ) from exc
    return records


def _config_sha256(config: Mapping) -> str:
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _validate_resume_records(
    records: Sequence[dict],
    *,
    seed: int,
    allowed_keys: set[tuple[str, int, int]],
    replay_config_sha256: str,
) -> set[tuple[str, int, int]]:
    seen = set()
    for record in records:
        if record.get("schema") != REPLAY_SCHEMA:
            raise SystemExit("resume JSONL contains an incompatible replay schema")
        if int(record["seed"]) != seed:
            raise SystemExit("resume JSONL seed does not match --seed")
        if record.get("replay_config_sha256") != replay_config_sha256:
            raise SystemExit("resume JSONL was produced with a different replay config")
        key = _record_key(record)
        if key not in allowed_keys:
            raise SystemExit(f"resume record {key} is outside the selected group shard")
        if key in seen:
            raise SystemExit(f"resume JSONL contains duplicate group {key}")
        seen.add(key)
    return seen


def _best_metric_multiplier(
    metric_by_multiplier: Mapping[float, float],
) -> float | None:
    finite = [
        multiplier
        for multiplier, value in metric_by_multiplier.items()
        if math.isfinite(float(value))
    ]
    if not finite:
        return None
    return min(
        finite,
        key=lambda multiplier: (
            -float(metric_by_multiplier[multiplier]),
            abs(float(multiplier) - BASELINE_MULTIPLIER),
            float(multiplier),
        ),
    )


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return float("nan")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _action_map(record: Mapping) -> dict[float, Mapping]:
    actions = record.get("actions", [])
    if isinstance(actions, Mapping):
        actions = actions.values()
    return {float(action["multiplier"]): action for action in actions}


def summarize(records: list[dict]) -> dict:
    if not records:
        return {
            "schema": SUMMARY_SCHEMA,
            "records": 0,
            "seeds": [],
            "action_multipliers": [],
            "best_fixed_multiplier": None,
            "best_fixed_mean_oracle_gain_vs_baseline": None,
            "mean_chosen_oracle_gain_vs_baseline": None,
            "mean_best_oracle_gain_vs_baseline": None,
            "oracle_headroom_captured": None,
            "per_multiplier": {},
        }

    action_maps = [_action_map(record) for record in records]
    multiplier_sets = [set(actions) for actions in action_maps]
    if not multiplier_sets[0] or any(
        values != multiplier_sets[0] for values in multiplier_sets
    ):
        raise ValueError("records do not share one complete action multiplier set")
    multipliers = tuple(sorted(multiplier_sets[0]))
    if BASELINE_MULTIPLIER not in multipliers:
        raise ValueError("records do not contain baseline multiplier 1.0")

    chosen_gains = [
        float(
            record.get(
                "chosen_oracle_gain_vs_baseline",
                record.get("oracle_gain_vs_baseline"),
            )
        )
        for record in records
    ]
    chosen_multipliers = [float(record["chosen_multiplier"]) for record in records]
    step_ratios = [float(record["step_norm_ratio"]) for record in records]
    best_oracle_gains = []
    best_oracle_multipliers = []
    for record, actions in zip(records, action_maps):
        gains = {
            multiplier: float(action["oracle_gain_vs_baseline"])
            for multiplier, action in actions.items()
        }
        best_multiplier = record.get("best_oracle_multiplier")
        if best_multiplier is None:
            best_multiplier = _best_metric_multiplier(gains)
        best_multiplier = float(best_multiplier)
        best_oracle_multipliers.append(best_multiplier)
        best_oracle_gains.append(
            float(record.get("best_oracle_gain_vs_baseline", gains[best_multiplier]))
        )

    per_multiplier = {}
    fixed_means = {}
    for multiplier in multipliers:
        gains = [
            float(actions[multiplier]["oracle_gain_vs_baseline"])
            for actions in action_maps
        ]
        ratios = [
            float(actions[multiplier]["step_norm_ratio"]) for actions in action_maps
        ]
        fixed_means[multiplier] = mean(gains)
        per_multiplier[multiplier_key(multiplier)] = {
            "multiplier": multiplier,
            "records": len(records),
            "mean_oracle_gain_vs_baseline": mean(gains),
            "median_oracle_gain_vs_baseline": _median(gains),
            "gain_positive_rate": mean([gain > 0.0 for gain in gains]),
            "mean_step_norm_ratio": mean(ratios),
            "chosen_count": sum(value == multiplier for value in chosen_multipliers),
            "oracle_best_count": sum(
                value == multiplier for value in best_oracle_multipliers
            ),
        }

    best_fixed_multiplier = _best_metric_multiplier(fixed_means)
    mean_chosen_gain = mean(chosen_gains)
    mean_best_oracle_gain = mean(best_oracle_gains)
    captured_rows = [
        chosen / best
        for chosen, best in zip(chosen_gains, best_oracle_gains)
        if best > 0.0
    ]
    fallback_reasons = [
        record.get("selection", {}).get("fallback_reason") for record in records
    ]
    summary = {
        "schema": SUMMARY_SCHEMA,
        "records": len(records),
        "seeds": sorted({int(record["seed"]) for record in records}),
        "action_multipliers": list(multipliers),
        "baseline_multiplier": BASELINE_MULTIPLIER,
        "mean_chosen_oracle_gain_vs_baseline": mean_chosen_gain,
        "median_chosen_oracle_gain_vs_baseline": _median(chosen_gains),
        "chosen_gain_positive_rate": mean([gain > 0.0 for gain in chosen_gains]),
        "selection_rate": mean(
            [multiplier != BASELINE_MULTIPLIER for multiplier in chosen_multipliers]
        ),
        "fallback_rate": mean([reason is not None for reason in fallback_reasons]),
        "chosen_multiplier_distribution": dict(
            sorted(
                Counter(multiplier_key(value) for value in chosen_multipliers).items()
            )
        ),
        "mean_step_norm_ratio": mean(step_ratios),
        "best_fixed_multiplier": best_fixed_multiplier,
        "best_fixed_mean_oracle_gain_vs_baseline": (
            None
            if best_fixed_multiplier is None
            else fixed_means[best_fixed_multiplier]
        ),
        "mean_best_oracle_gain_vs_baseline": mean_best_oracle_gain,
        "best_oracle_multiplier_distribution": dict(
            sorted(
                Counter(
                    multiplier_key(value) for value in best_oracle_multipliers
                ).items()
            )
        ),
        "oracle_headroom_captured": (
            None
            if mean_best_oracle_gain <= 0.0
            else mean_chosen_gain / mean_best_oracle_gain
        ),
        "mean_per_record_oracle_headroom_captured": (
            None if not captured_rows else mean(captured_rows)
        ),
        "per_multiplier": per_multiplier,
    }

    if all("baseline_oracle_utility" in record for record in records):
        baseline_negative = mean(
            [float(record["baseline_oracle_utility"]) < 0.0 for record in records]
        )
        chosen_negative = mean(
            [float(record["chosen_oracle_utility"]) < 0.0 for record in records]
        )
        summary.update(
            {
                "baseline_negative_rate": baseline_negative,
                "chosen_negative_rate": chosen_negative,
                "negative_rate_relative_drop": (
                    None
                    if baseline_negative <= 0.0
                    else (baseline_negative - chosen_negative) / baseline_negative
                ),
            }
        )

    per_seed = {}
    for seed in summary["seeds"]:
        indices = [
            index for index, record in enumerate(records) if int(record["seed"]) == seed
        ]
        per_seed[str(seed)] = {
            "records": len(indices),
            "mean_chosen_oracle_gain_vs_baseline": mean(
                [chosen_gains[index] for index in indices]
            ),
            "mean_best_oracle_gain_vs_baseline": mean(
                [best_oracle_gains[index] for index in indices]
            ),
            "selection_rate": mean(
                [chosen_multipliers[index] != BASELINE_MULTIPLIER for index in indices]
            ),
        }
    summary["per_seed"] = per_seed
    return summary


def replay(args) -> list[dict]:
    root = args.capture_dir
    groups = validate_candidate_groups(
        buffered._read_jsonl(root / "index.jsonl"), args.expected_candidates
    )
    indexed_groups = list(enumerate(groups))
    indexed_groups = indexed_groups[args.group_start :: args.group_stride]
    if args.max_groups is not None:
        indexed_groups = indexed_groups[: args.max_groups]
    if not indexed_groups:
        raise SystemExit("no candidate groups selected by the requested shard")
    allowed_keys = {_group_key(group) for _, group in indexed_groups}

    existing = list(getattr(args, "_existing_records", []))
    preliminary_seen = set()
    for record in existing:
        key = _record_key(record)
        if key not in allowed_keys:
            raise SystemExit(f"resume record {key} is outside the selected group shard")
        if key in preliminary_seen:
            raise SystemExit(f"resume JSONL contains duplicate group {key}")
        preliminary_seen.add(key)

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
        anchor = build_row_panels(
            args.anchor_data,
            tokenizer,
            seq_len=args.seq_len,
            panel_count=args.anchor_panels,
            blocks_per_panel=args.anchor_blocks_per_panel,
            max_rows=args.anchor_max_rows,
            train_on=args.train_on,
            device=device,
        )
        oracle = build_row_panels(
            args.oracle_data,
            tokenizer,
            seq_len=args.seq_len,
            panel_count=args.oracle_panels,
            blocks_per_panel=args.oracle_blocks_per_panel,
            max_rows=args.oracle_max_rows,
            train_on=args.train_on,
            device=device,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    panel_metadata = build_panel_metadata(anchor, oracle)
    if not panel_metadata["disjointness"]["verified_disjoint"]:
        raise SystemExit(
            "anchor and oracle sources overlap on "
            f"{panel_metadata['disjointness']['overlap_count']} canonical rows"
        )

    replay_config = {
        "schema": REPLAY_SCHEMA,
        "seed": args.seed,
        "capture_dir": str(args.capture_dir.expanduser().resolve()),
        "model": args.model,
        "outer_optimizer": "sgd",
        "outer_lr": args.outer_lr,
        "delta_correction": "none",
        "action_multipliers": list(args.action_multipliers),
        "expected_candidates": args.expected_candidates,
        "selection": {
            "min_gain": args.min_gain,
            "lcb_z": args.lcb_z,
            "min_win_rate": args.min_win_rate,
            "fallback_multiplier": BASELINE_MULTIPLIER,
        },
        "layout": {
            "fragments": args.fragments,
            "fragment_pattern": args.fragment_pattern,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_targets": args.lora_targets,
        },
        "panel_metadata": panel_metadata,
        "group_shard": {"start": args.group_start, "stride": args.group_stride},
    }
    replay_config_sha256 = _config_sha256(replay_config)
    seen = _validate_resume_records(
        existing,
        seed=args.seed,
        allowed_keys=allowed_keys,
        replay_config_sha256=replay_config_sha256,
    )
    indexed_groups = [
        (index, group)
        for index, group in indexed_groups
        if _group_key(group) not in seen
    ]
    args._panel_metadata = panel_metadata
    args._replay_config = replay_config
    args._replay_config_sha256 = replay_config_sha256
    if not indexed_groups:
        return []

    compute_loss = lambda logits, ids, weights: sft_loss(  # noqa: E731
        logits, ids, args.loss_function, weights
    )
    records = []
    record_panel_metadata = _record_panel_metadata(panel_metadata)
    was_training = model.training
    model.eval()
    try:
        for completed, (group_index, group) in enumerate(indexed_groups, start=1):
            first = group[0]
            state_checkpoint = str(first["state_checkpoint"])
            checkpoint = parse_checkpoint(buffered._resolve(root, state_checkpoint))
            syncer_eval._apply_checkpoint(checkpoint, layout, params, device)
            fragment_id = int(first["fragment"])
            frag = layout.fragments[fragment_id]
            current = checkpoint.fragments[fragment_id][1]
            current_version = int(checkpoint.fragments[fragment_id][0])
            zero_correction = torch.zeros_like(current)

            candidates = []
            for row in sorted(group, key=lambda item: int(item["learner_id"])):
                tensor = buffered._read_f32(
                    buffered._resolve(root, row["candidate_f32"]), frag.numel
                )
                if not bool(torch.isfinite(tensor).all()):
                    raise SystemExit(
                        f"step={first['step']} fragment={fragment_id}: "
                        "candidate contains NaN or Inf"
                    )
                candidate = buffered._candidate(
                    row, tensor, current, zero_correction, current_version
                )
                if not math.isfinite(candidate.weight) or candidate.weight < 0.0:
                    raise SystemExit(
                        f"step={first['step']} fragment={fragment_id}: "
                        f"invalid candidate weight {candidate.weight}"
                    )
                candidates.append(candidate)
            total_candidate_weight = sum(candidate.weight for candidate in candidates)
            if total_candidate_weight <= 0.0:
                raise SystemExit(
                    f"step={first['step']} fragment={fragment_id}: "
                    "candidate group has no positive token weight"
                )

            group_delta = exact_group_delta(candidates, frag)
            if not bool(torch.isfinite(group_delta).all()):
                raise SystemExit(
                    f"step={first['step']} fragment={fragment_id}: "
                    "merged group delta contains NaN or Inf"
                )
            trials = {
                multiplier: bn._nesterov_trial(
                    current,
                    zero_correction,
                    group_delta,
                    args.outer_lr * multiplier,
                    0.0,
                )
                for multiplier in args.action_multipliers
            }
            baseline_trial = trials[BASELINE_MULTIPLIER]
            baseline_step_norm = float((baseline_trial - current).norm().item())

            anchor_current_loss, anchor_current_panel_losses = syncer_eval._losses(
                model, anchor["panels"], compute_loss
            )
            anchor_results = {}
            for multiplier in args.action_multipliers:
                loss, panel_losses = losses_for_trial(
                    model,
                    anchor["panels"],
                    compute_loss,
                    frag,
                    params,
                    current,
                    trials[multiplier],
                    device,
                )
                anchor_results[multiplier] = {
                    "loss": loss,
                    "panel_losses": panel_losses,
                }

            # Selection is deliberately complete before any oracle loss is evaluated.
            selection = select_multiplier(
                {
                    multiplier: result["panel_losses"]
                    for multiplier, result in anchor_results.items()
                },
                args.action_multipliers,
                min_gain=args.min_gain,
                lcb_z=args.lcb_z,
                min_win_rate=args.min_win_rate,
                min_panels=args.anchor_panels,
            )
            chosen_multiplier = float(selection["chosen_multiplier"])
            decisions = {
                float(decision["multiplier"]): decision
                for decision in selection["decisions"]
            }

            oracle_current_loss, oracle_current_panel_losses = syncer_eval._losses(
                model, oracle["panels"], compute_loss
            )
            oracle_results = {}
            for multiplier in args.action_multipliers:
                loss, panel_losses = losses_for_trial(
                    model,
                    oracle["panels"],
                    compute_loss,
                    frag,
                    params,
                    current,
                    trials[multiplier],
                    device,
                )
                oracle_results[multiplier] = {
                    "loss": loss,
                    "panel_losses": panel_losses,
                }

            anchor_baseline_loss = anchor_results[BASELINE_MULTIPLIER]["loss"]
            oracle_baseline_loss = oracle_results[BASELINE_MULTIPLIER]["loss"]
            actions = []
            for multiplier in args.action_multipliers:
                trial = trials[multiplier]
                step_norm = float((trial - current).norm().item())
                step_norm_ratio = _step_ratio(step_norm, baseline_step_norm)
                oracle_loss = oracle_results[multiplier]["loss"]
                actions.append(
                    {
                        "name": action_name(multiplier),
                        "multiplier": multiplier,
                        "outer_lr": args.outer_lr * multiplier,
                        "step_norm": step_norm,
                        "step_norm_ratio": step_norm_ratio,
                        "expected_step_norm_ratio": multiplier,
                        "step_norm_ratio_error": abs(step_norm_ratio - multiplier),
                        "anchor_loss": anchor_results[multiplier]["loss"],
                        "anchor_panel_losses": anchor_results[multiplier][
                            "panel_losses"
                        ],
                        "anchor_gain_vs_baseline": (
                            anchor_baseline_loss - anchor_results[multiplier]["loss"]
                        ),
                        "selection_decision": decisions.get(multiplier),
                        "oracle_loss": oracle_loss,
                        "oracle_panel_losses": oracle_results[multiplier][
                            "panel_losses"
                        ],
                        "oracle_gain_vs_baseline": oracle_baseline_loss - oracle_loss,
                        "oracle_utility": oracle_current_loss - oracle_loss,
                        "oracle_utility_se": utility_se(
                            oracle_current_panel_losses,
                            oracle_results[multiplier]["panel_losses"],
                        ),
                    }
                )
            action_by_multiplier = {
                float(action["multiplier"]): action for action in actions
            }
            best_oracle_multiplier = _best_metric_multiplier(
                {
                    multiplier: action["oracle_gain_vs_baseline"]
                    for multiplier, action in action_by_multiplier.items()
                }
            )
            assert best_oracle_multiplier is not None
            chosen_action = action_by_multiplier[chosen_multiplier]
            baseline_action = action_by_multiplier[BASELINE_MULTIPLIER]
            best_oracle_action = action_by_multiplier[best_oracle_multiplier]
            record = {
                "schema": REPLAY_SCHEMA,
                "replay_config_sha256": replay_config_sha256,
                "seed": args.seed,
                "group_index": group_index,
                "group_ordinal": group_index + 1,
                "state_checkpoint": state_checkpoint,
                "step": int(first["step"]),
                "fragment": fragment_id,
                "merge_mode": "avg" if frag.merge_mode == MERGE_AVG else "rda",
                "outer_optimizer": "sgd",
                "outer_lr": args.outer_lr,
                "delta_correction": "none",
                "candidate_count": len(candidates),
                "candidate_learner_ids": [
                    int(candidate.row["learner_id"]) for candidate in candidates
                ],
                "candidate_weights": [candidate.weight for candidate in candidates],
                "total_candidate_weight": total_candidate_weight,
                "group_delta_norm": float(group_delta.norm().item()),
                "action_multipliers": list(args.action_multipliers),
                "baseline_multiplier": BASELINE_MULTIPLIER,
                "panel_metadata": record_panel_metadata,
                "anchor_current_loss": anchor_current_loss,
                "anchor_current_panel_losses": anchor_current_panel_losses,
                "anchor_baseline_loss": anchor_baseline_loss,
                "selection": selection,
                "chosen_multiplier": chosen_multiplier,
                "chosen_outer_lr": args.outer_lr * chosen_multiplier,
                "chosen_anchor_gain_vs_baseline": chosen_action[
                    "anchor_gain_vs_baseline"
                ],
                "oracle_current_loss": oracle_current_loss,
                "oracle_current_panel_losses": oracle_current_panel_losses,
                "oracle_baseline_loss": oracle_baseline_loss,
                "baseline_oracle_utility": baseline_action["oracle_utility"],
                "chosen_oracle_loss": chosen_action["oracle_loss"],
                "chosen_oracle_utility": chosen_action["oracle_utility"],
                "chosen_oracle_utility_se": chosen_action["oracle_utility_se"],
                "chosen_oracle_gain_vs_baseline": chosen_action[
                    "oracle_gain_vs_baseline"
                ],
                "oracle_gain_vs_baseline": chosen_action["oracle_gain_vs_baseline"],
                "baseline_step_norm": baseline_step_norm,
                "chosen_step_norm": chosen_action["step_norm"],
                "step_norm_ratio": chosen_action["step_norm_ratio"],
                "best_oracle_multiplier": best_oracle_multiplier,
                "best_oracle_gain_vs_baseline": best_oracle_action[
                    "oracle_gain_vs_baseline"
                ],
                "best_oracle_loss": best_oracle_action["oracle_loss"],
                "oracle_headroom_over_baseline": best_oracle_action[
                    "oracle_gain_vs_baseline"
                ],
                "actions": actions,
            }
            records.append(record)
            args._sink.write(
                json.dumps(
                    jsonable(record),
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
            args._sink.flush()
            if args.progress_every and (
                completed == 1 or completed % args.progress_every == 0
            ):
                print(
                    f"[exact-lr] groups={completed}/{len(indexed_groups)} "
                    f"step={record['step']} multiplier={chosen_multiplier:g}",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        model.train(was_training)
    return records


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
    parser.add_argument("--anchor-panels", type=int, default=8)
    parser.add_argument("--oracle-panels", type=int, default=8)
    parser.add_argument("--anchor-blocks-per-panel", type=int, default=2)
    parser.add_argument("--oracle-blocks-per-panel", type=int, default=2)
    parser.add_argument("--anchor-max-rows", type=int, default=256)
    parser.add_argument("--oracle-max-rows", type=int, default=256)
    parser.add_argument(
        "--multipliers",
        default=",".join(multiplier_key(value) for value in DEFAULT_MULTIPLIERS),
        help="comma-separated outer-SGD LR multipliers; must include 1.0",
    )
    parser.add_argument("--outer-lr", type=float, default=0.28)
    parser.add_argument("--min-gain", type=float, default=DEFAULT_MIN_GAIN)
    parser.add_argument("--lcb-z", type=float, default=DEFAULT_LCB_Z)
    parser.add_argument("--min-win-rate", type=float, default=DEFAULT_MIN_WIN_RATE)
    parser.add_argument("--expected-candidates", type=int, default=4)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--group-start", type=int, default=0)
    parser.add_argument("--group-stride", type=int, default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--out-jsonl", required=True, type=Path)
    parser.add_argument("--out-summary", required=True, type=Path)
    args = parser.parse_args(argv)

    args.seed = args.seed if args.seed is not None else infer_seed(args.capture_dir)
    if args.seed is None:
        parser.error("could not infer seed; pass --seed")
    try:
        args.action_multipliers = normalize_multipliers(args.multipliers)
    except ValueError as exc:
        parser.error(str(exc))
    if not math.isfinite(args.outer_lr) or args.outer_lr <= 0.0:
        parser.error("--outer-lr must be finite and positive")
    if args.seq_len <= 1:
        parser.error("--seq-len must be greater than 1")
    if args.anchor_panels < 2 or args.oracle_panels < 2:
        parser.error("anchor and oracle panel counts must be at least 2")
    if args.anchor_blocks_per_panel < 1 or args.oracle_blocks_per_panel < 1:
        parser.error("blocks per panel must be positive")
    if args.anchor_max_rows < args.anchor_panels:
        parser.error("--anchor-max-rows must be at least --anchor-panels")
    if args.oracle_max_rows < args.oracle_panels:
        parser.error("--oracle-max-rows must be at least --oracle-panels")
    if args.expected_candidates < 1:
        parser.error("--expected-candidates must be positive")
    if args.group_stride < 1:
        parser.error("--group-stride must be >= 1")
    if not 0 <= args.group_start < args.group_stride:
        parser.error("--group-start must be in [0, group-stride)")
    if args.max_groups is not None and args.max_groups < 1:
        parser.error("--max-groups must be positive")
    if not math.isfinite(args.min_gain) or args.min_gain < 0.0:
        parser.error("--min-gain must be finite and nonnegative")
    if not math.isfinite(args.lcb_z) or args.lcb_z < 0.0:
        parser.error("--lcb-z must be finite and nonnegative")
    if not 0.0 <= args.min_win_rate <= 1.0:
        parser.error("--min-win-rate must be in [0, 1]")
    if args.out_jsonl.resolve() == args.out_summary.resolve():
        parser.error("--out-jsonl and --out-summary must be different paths")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    existing = (
        read_jsonl(args.out_jsonl) if args.resume and args.out_jsonl.exists() else []
    )
    args._existing_records = existing
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and existing else "w"
    with args.out_jsonl.open(mode) as sink:
        args._sink = sink
        new_records = replay(args)
    records = existing + new_records
    summary = summarize(records)
    summary["selection"] = {
        "min_gain": args.min_gain,
        "lcb_z": args.lcb_z,
        "min_win_rate": args.min_win_rate,
        "fallback_multiplier": BASELINE_MULTIPLIER,
        "tie_break": [
            "lcb",
            "mean_gain",
            "win_rate",
            "closest_to_1.0",
            "lower_multiplier",
        ],
    }
    summary["group_shard"] = {
        "start": args.group_start,
        "stride": args.group_stride,
    }
    summary["panel_metadata"] = args._panel_metadata
    summary["replay_config"] = args._replay_config
    summary["replay_config_sha256"] = args._replay_config_sha256
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    text = (
        json.dumps(
            jsonable(summary),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    args.out_summary.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
