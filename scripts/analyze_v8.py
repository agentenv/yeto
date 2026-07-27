#!/usr/bin/env python3
"""Frozen tuned-loss phase-diagram analysis for outer-muP v8.

The primary estimand at each (T, mu, arm) coordinate is the independently
tuned quadratic minimum minus the independently tuned mu=0 minimum at the
same T.  Every curve is fit in x=log2(eta) from the three-seed mean loss at the
four prospectively registered eta values.  One paired training-seed bootstrap
draw is shared by all 15 curves so both the per-coordinate intervals and the
simultaneous phase-map interval retain the registered pairing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path


T_GRID = (2, 5, 20)
MU_GRID = (0.8, 0.95)
MOMENTUM_ARMS = ("raw", "corrected")
BASELINE_ARM = "mu0"
SEEDS = (801, 809, 811)
ETA_POINTS = 4
EXPECTED_CURVES = 15
EXPECTED_COMPARISONS = 12
EXPECTED_CELLS = 180
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260728
MIN_VALID_BOOTSTRAP_REPLICATES = 9_500
PRACTICAL_MARGIN_LOSS = 0.01
BOUNDARY_TOLERANCE_LOG2 = 1e-12


class AnalysisError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnalysisError(f"{path}: expected a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    try:
        with path.open(encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise AnalysisError(f"{path}:{number}: expected an object")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read {path}: {exc}") from exc
    return rows


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise AnalysisError("quantile of an empty sequence")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def solve3(matrix: list[list[float]], vector: list[float]) -> list[float]:
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise AnalysisError("quadratic normal equation has the wrong shape")
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-15:
            raise AnalysisError("singular quadratic normal equation")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(3):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                left - factor * right
                for left, right in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][-1] for row in range(3)]


def fit_quadratic(etas: list[float], losses: list[float]) -> dict:
    if len(etas) != ETA_POINTS or len(losses) != ETA_POINTS:
        raise AnalysisError("registered eta fit requires exactly four points")
    if any(not math.isfinite(value) or value <= 0.0 for value in etas):
        raise AnalysisError("eta fit received a nonpositive or nonfinite eta")
    if any(not math.isfinite(value) for value in losses):
        raise AnalysisError("eta fit received a nonfinite loss")
    xs = [math.log2(eta) for eta in etas]
    sums = [sum(x**power for x in xs) for power in range(5)]
    matrix = [
        [sums[4], sums[3], sums[2]],
        [sums[3], sums[2], sums[1]],
        [sums[2], sums[1], sums[0]],
    ]
    vector = [
        sum(loss * x * x for x, loss in zip(xs, losses)),
        sum(loss * x for x, loss in zip(xs, losses)),
        sum(losses),
    ]
    a, b, c = solve3(matrix, vector)
    vertex = -b / (2.0 * a) if a else math.nan
    interior = (
        a > 0.0
        and min(xs) + BOUNDARY_TOLERANCE_LOG2
        < vertex
        < max(xs) - BOUNDARY_TOLERANCE_LOG2
    )
    tuned_loss = c - b * b / (4.0 * a) if interior else None
    return {
        "a": a,
        "b": b,
        "c": c,
        "vertex_log2_eta": vertex if math.isfinite(vertex) else None,
        "eta_star": 2.0**vertex if interior else None,
        "tuned_loss": tuned_loss,
        "interior": interior,
        "status": "INTERIOR" if interior else "UNBRACKETED",
    }


def curve_key(t: int, arm: str, mu: float) -> tuple[int, str, float]:
    return (int(t), str(arm), float(mu))


def expected_curve_keys() -> tuple[tuple[int, str, float], ...]:
    keys = []
    for t in T_GRID:
        keys.append(curve_key(t, BASELINE_ARM, 0.0))
        for arm in MOMENTUM_ARMS:
            for mu in MU_GRID:
                keys.append(curve_key(t, arm, mu))
    return tuple(keys)


def curve_fit(
    losses: dict[tuple[int, str, float, int, float], float],
    t: int,
    arm: str,
    mu: float,
    sampled_indices: list[int] | None = None,
) -> dict:
    etas = sorted(
        {
            key[4]
            for key in losses
            if key[0] == t and key[1] == arm and key[2] == mu
        }
    )
    if len(etas) != ETA_POINTS:
        raise AnalysisError(
            f"T{t}/{arm}/mu{mu:g}: expected four etas, found {etas}"
        )
    selected_seeds = (
        [SEEDS[index] for index in sampled_indices]
        if sampled_indices is not None
        else list(SEEDS)
    )
    means = []
    for eta in etas:
        values = []
        for seed in selected_seeds:
            key = (t, arm, mu, seed, eta)
            if key not in losses:
                raise AnalysisError(f"missing scientific cell {key}")
            values.append(losses[key])
        means.append(sum(values) / len(values))
    fit = fit_quadratic(etas, means)
    fit.update(
        {
            "T": t,
            "S": 512 * t,
            "H": 512,
            "arm": arm,
            "mu": mu,
            "etas": etas,
            "seed_mean_losses": means,
        }
    )
    return fit


def calculate_all_curves(
    losses: dict[tuple[int, str, float, int, float], float],
    sampled_indices: list[int] | None = None,
) -> dict[tuple[int, str, float], dict]:
    return {
        key: curve_fit(losses, *key, sampled_indices)
        for key in expected_curve_keys()
    }


def tuned_differences(
    curves: dict[tuple[int, str, float], dict],
) -> dict[tuple[int, str, float], float]:
    differences = {}
    for t in T_GRID:
        baseline = curves[curve_key(t, BASELINE_ARM, 0.0)]
        if not baseline["interior"] or baseline["tuned_loss"] is None:
            raise AnalysisError(f"T{t} mu0 curve is unbracketed")
        for arm in MOMENTUM_ARMS:
            for mu in MU_GRID:
                curve = curves[curve_key(t, arm, mu)]
                if not curve["interior"] or curve["tuned_loss"] is None:
                    raise AnalysisError(f"T{t}/{arm}/mu{mu:g} is unbracketed")
                differences[curve_key(t, arm, mu)] = (
                    curve["tuned_loss"] - baseline["tuned_loss"]
                )
    if len(differences) != EXPECTED_COMPARISONS:
        raise AnalysisError("tuned-difference map is incomplete")
    return differences


def phase_label(interval: dict[str, float | None]) -> str:
    low = interval.get("low")
    high = interval.get("high")
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        return "NOT_EVALUABLE"
    if high < -PRACTICAL_MARGIN_LOSS:
        return "HELPS"
    if low > PRACTICAL_MARGIN_LOSS:
        return "HURTS"
    if low >= -PRACTICAL_MARGIN_LOSS and high <= PRACTICAL_MARGIN_LOSS:
        return "NEUTRAL"
    return "UNCERTAIN"


def joint_bootstrap(
    losses: dict[tuple[int, str, float, int, float], float],
    point_differences: dict[tuple[int, str, float], float],
) -> dict:
    rng = random.Random(BOOTSTRAP_SEED)
    samples = {key: [] for key in sorted(point_differences)}
    max_abs_deviations = []
    invalid = 0
    for _ in range(BOOTSTRAP_REPLICATES):
        draw = [rng.randrange(len(SEEDS)) for _ in SEEDS]
        try:
            curves = calculate_all_curves(losses, draw)
            if not all(curve["interior"] for curve in curves.values()):
                raise AnalysisError("bootstrap curve unbracketed")
            differences = tuned_differences(curves)
        except (AnalysisError, KeyError, ValueError, OverflowError):
            invalid += 1
            continue
        for key, value in differences.items():
            samples[key].append(value)
        max_abs_deviations.append(
            max(abs(differences[key] - point_differences[key]) for key in differences)
        )

    valid = BOOTSTRAP_REPLICATES - invalid
    status = (
        "VALID" if valid >= MIN_VALID_BOOTSTRAP_REPLICATES else "NOT_EVALUABLE"
    )
    simultaneous_radius = (
        quantile(max_abs_deviations, 0.95) if max_abs_deviations else None
    )
    comparisons = []
    for key in sorted(point_differences):
        t, arm, mu = key
        values = samples[key]
        point = point_differences[key]
        pointwise = {
            "low": quantile(values, 0.025) if values else None,
            "high": quantile(values, 0.975) if values else None,
        }
        simultaneous = {
            "low": point - simultaneous_radius
            if simultaneous_radius is not None
            else None,
            "high": point + simultaneous_radius
            if simultaneous_radius is not None
            else None,
        }
        comparisons.append(
            {
                "T": t,
                "S": 512 * t,
                "H": 512,
                "arm": arm,
                "mu": mu,
                "delta_tuned_loss_momentum_minus_mu0": point,
                "pointwise_ci_95": pointwise,
                "simultaneous_familywise_ci_95": simultaneous,
                "primary_phase_label": (
                    phase_label(simultaneous)
                    if status == "VALID"
                    else "NOT_EVALUABLE"
                ),
                "pointwise_phase_label_descriptive": (
                    phase_label(pointwise)
                    if status == "VALID"
                    else "NOT_EVALUABLE"
                ),
            }
        )
    return {
        "method": (
            "paired nonparametric training-seed bootstrap; one shared three-index "
            "draw refits all 15 curves and recomputes all 12 tuned-minimum "
            "differences"
        ),
        "replicates": BOOTSTRAP_REPLICATES,
        "rng_seed": BOOTSTRAP_SEED,
        "minimum_valid_replicates": MIN_VALID_BOOTSTRAP_REPLICATES,
        "valid_replicates": valid,
        "invalid_complete_refits": invalid,
        "status": status,
        "simultaneous_interval_method": (
            "95th percentile of the valid-draw maximum absolute deviation from "
            "the 12 point estimates; one common unstudentized radius"
        ),
        "simultaneous_radius_loss": simultaneous_radius,
        "comparisons": comparisons,
    }


def parse_node_roots(values: list[str]) -> dict[str, Path]:
    roots = {}
    for value in values:
        if "=" not in value:
            raise AnalysisError(f"node root must be NODE=PATH: {value!r}")
        node, raw_path = value.split("=", 1)
        if not node or node in roots:
            raise AnalysisError(f"invalid or duplicate node root: {value!r}")
        roots[node] = Path(raw_path).resolve()
    return roots


def completed_attempt(cell: dict, root: Path) -> tuple[Path, dict, int]:
    for attempt_number in (2, 1):
        attempt = root / cell["cell_id"] / f"attempt-{attempt_number}"
        evidence_path = attempt / "evidence.json"
        if not evidence_path.is_file():
            continue
        evidence = read_json(evidence_path)
        if evidence.get("status") != "COMPLETED":
            raise AnalysisError(
                f"attempt {attempt_number} is {evidence.get('status')}"
            )
        if attempt_number == 2 and not cell.get("registered_retry_commands"):
            raise AnalysisError("unregistered attempt-2 evidence exists")
        return attempt, evidence, attempt_number
    raise AnalysisError("no completed registered attempt evidence")


def load_losses(
    manifest: dict, node_roots: dict[str, Path]
) -> tuple[dict, list[dict], list[dict]]:
    losses = {}
    records = []
    errors = []
    for cell in manifest.get("cells", []):
        cell_id = cell.get("cell_id", "<missing-cell-id>")
        try:
            node = cell["assignment"]["node"]
            if node not in node_roots:
                raise AnalysisError(f"no results root supplied for node {node}")
            attempt, evidence, attempt_number = completed_attempt(
                cell, node_roots[node]
            )
            expected_hash = (
                cell["command_hash"]
                if attempt_number == 1
                else cell["registered_retry_commands"][0]["command_hash"]
            )
            if evidence.get("cell_id") != cell_id:
                raise AnalysisError("evidence cell_id mismatch")
            if evidence.get("command_hash") != expected_hash:
                raise AnalysisError("evidence command hash mismatch")
            if evidence.get("seed") != cell["seed"]:
                raise AnalysisError("evidence seed mismatch")
            results_path = attempt / "report" / "results.jsonl"
            observed = evidence.get("observed_artifacts", {}).get("results", {})
            if not results_path.is_file() or observed.get("sha256") != sha256_file(
                results_path
            ):
                raise AnalysisError("results hash does not match validated evidence")
            rows = read_jsonl(results_path)
            if len(rows) != 1:
                raise AnalysisError("expected exactly one endpoint result row")
            loss = rows[0].get("eval_loss")
            if not isinstance(loss, (int, float)) or not math.isfinite(loss):
                raise AnalysisError("endpoint loss is missing or nonfinite")
            key = (
                int(cell["t"]),
                str(cell["arm"]),
                float(cell["mu"]),
                int(cell["seed"]),
                float(cell["eta"]),
            )
            if key in losses:
                raise AnalysisError(f"duplicate scientific cell key {key}")
            losses[key] = float(loss)
            records.append(
                {
                    "cell_id": cell_id,
                    "node": node,
                    "gpu": cell["assignment"]["gpu"],
                    "attempt": attempt_number,
                    "T": int(cell["t"]),
                    "S": int(cell["s"]),
                    "H": int(cell["h"]),
                    "arm": str(cell["arm"]),
                    "mu": float(cell["mu"]),
                    "eta": float(cell["eta"]),
                    "seed": int(cell["seed"]),
                    "eval_loss": float(loss),
                    "evidence_path": str(attempt / "evidence.json"),
                    "evidence_sha256": sha256_file(attempt / "evidence.json"),
                    "results_sha256": observed["sha256"],
                }
            )
        except (AnalysisError, KeyError, OSError, ValueError) as exc:
            errors.append({"cell_id": cell_id, "error": str(exc)})
    return losses, records, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--node-root",
        action="append",
        required=True,
        help="NODE=PATH; repeat once for each results-bearing node",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    if manifest.get("schema") != "yeto_outer_mup_v8_launch_manifest_v1":
        raise SystemExit("not an outer-muP v8 launch manifest")
    if manifest.get("stage") != "V8_PHASE_DIAGRAM" or len(
        manifest.get("cells", [])
    ) != EXPECTED_CELLS:
        raise SystemExit("manifest is not the complete registered 180-cell v8 mini grid")
    if manifest.get("reuse", {}).get("reused_cell_count") != 0:
        raise SystemExit("v8 analyzer is frozen for the registered zero-reuse audit")

    node_roots = parse_node_roots(args.node_root)
    losses, cell_records, evidence_errors = load_losses(manifest, node_roots)
    curves = {}
    curve_errors = []
    for key in expected_curve_keys():
        try:
            curves[key] = curve_fit(losses, *key)
        except AnalysisError as exc:
            t, arm, mu = key
            curves[key] = {
                "T": t,
                "S": 512 * t,
                "H": 512,
                "arm": arm,
                "mu": mu,
                "status": "INVALID_INPUT",
                "interior": False,
                "eta_star": None,
                "tuned_loss": None,
                "error": str(exc),
            }
            curve_errors.append(f"T{t}/{arm}/mu{mu:g}: {exc}")

    evidence_complete = not evidence_errors and len(losses) == EXPECTED_CELLS
    all_curves_interior = all(curve["interior"] for curve in curves.values())
    if evidence_complete and all_curves_interior and not curve_errors:
        try:
            point_differences = tuned_differences(curves)
            bootstrap = joint_bootstrap(losses, point_differences)
        except AnalysisError as exc:
            point_differences = {}
            bootstrap = {
                "status": "NOT_EVALUABLE",
                "valid_replicates": 0,
                "error": str(exc),
            }
    else:
        point_differences = {}
        bootstrap = {
            "status": "NOT_EVALUABLE",
            "valid_replicates": 0,
            "error": "complete evidence and 15 interior point fits are required",
        }

    evaluable = (
        evidence_complete
        and all_curves_interior
        and not curve_errors
        and bootstrap.get("status") == "VALID"
    )
    comparisons = bootstrap.get("comparisons", []) if evaluable else []
    counts = {
        label: sum(
            item.get("primary_phase_label") == label for item in comparisons
        )
        for label in ("HELPS", "HURTS", "NEUTRAL", "UNCERTAIN")
    }
    note_line = (
        "V8 PHASE DIAGRAM: "
        + ("COMPLETE" if evaluable else "NOT_EVALUABLE")
        + " "
        + " ".join(f"{label.lower()}={counts[label]}" for label in counts)
    )
    readout = {
        "schema": "yeto_outer_mup_v8_phase_diagram_readout_v1",
        "created_at_utc": utc_now(),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "source_git_commit": manifest.get("source", {}).get("git_commit"),
        "expected_cells": EXPECTED_CELLS,
        "observed_completed_cells": len(cell_records),
        "reuse": manifest.get("reuse"),
        "evidence_errors": evidence_errors,
        "curve_errors": curve_errors,
        "cell_registry_sha256": canonical_sha256(
            sorted(cell_records, key=lambda item: item["cell_id"])
        ),
        "cell_records": sorted(cell_records, key=lambda item: item["cell_id"]),
        "curve_fits": [curves[key] for key in sorted(curves)],
        "analysis": {
            "estimand": (
                "quadratic tuned minimum(momentum arm) minus quadratic tuned "
                "minimum(mu0) at the same T"
            ),
            "practical_margin_loss": PRACTICAL_MARGIN_LOSS,
            "bootstrap": bootstrap,
            "primary_phase_counts": counts,
        },
        "gate": {
            "name": "G8_EVALUABILITY",
            "verdict": "COMPLETE" if evaluable else "NOT_EVALUABLE",
            "evaluable": evaluable,
            "conditions": {
                "complete_180_cell_evidence": evidence_complete,
                "all_15_eta_optima_interior": all_curves_interior,
                "joint_bootstrap_at_least_9500_valid": (
                    bootstrap.get("status") == "VALID"
                ),
            },
        },
        "note_line": note_line,
    }
    write_json_atomic(args.output.resolve(), readout)
    print(note_line)
    return 0 if evaluable else 2


if __name__ == "__main__":
    raise SystemExit(main())
