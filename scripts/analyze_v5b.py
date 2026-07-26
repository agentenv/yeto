#!/usr/bin/env python3
"""Apply the frozen G5B analysis to the combined v5 and v5b SNOO grids."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path


CONDITIONS = ("a", "b", "c")
SEEDS = (521, 523, 541, 547, 557)
V5_EXPECTED_CELLS = 90
V5B_EXPECTED_CELLS = 75
EXPECTED_COMBINED_CELLS = V5_EXPECTED_CELLS + V5B_EXPECTED_CELLS
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260726
MIN_VALID_BOOTSTRAP_REPLICATES = 9_500
V5_LAUNCH_MANIFEST_SHA256 = (
    "b9d54918da4e884f0b82db97df28996b0f88805038cd8d7472f4c244030405e8"
)
CODE_TRUE_MATCH_FACTOR = (1.0 - 0.9**5) / (1.0 - 0.9)
V5_BEST_CELLS = {
    "a": 0.5946035575013605,
    "b": 0.14508326803033197,
    "c": 0.5941304909110126,
}
V5_ETA_GRIDS = {
    condition: tuple(center * 2.0 ** ((index - 2.5) / 2.0) for index in range(6))
    for condition, center in V5_BEST_CELLS.items()
}
V5B_ETA_GRIDS = {
    "a": tuple(2.0**exponent for exponent in range(-9, -4)),
    "b": tuple(2.0**exponent for exponent in range(-11, -6)),
    "c": tuple(2.0**exponent for exponent in range(-9, -4)),
}
COMBINED_ETA_GRIDS = {
    condition: tuple(sorted((*V5B_ETA_GRIDS[condition], *V5_ETA_GRIDS[condition])))
    for condition in CONDITIONS
}


class AnalysisError(RuntimeError):
    pass


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
                    raise AnalysisError(f"{path}:{number}: expected a JSON object")
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
    return [augmented[row][3] for row in range(3)]


def fit_quadratic(etas: list[float], losses: list[float]) -> dict:
    if len(etas) != len(losses) or len(etas) < 3:
        raise AnalysisError("quadratic fit requires equal eta/loss vectors of length >=3")
    if any(not math.isfinite(value) or value <= 0 for value in etas):
        raise AnalysisError("quadratic fit received an invalid eta")
    if len(set(etas)) != len(etas):
        raise AnalysisError("quadratic fit received duplicate eta coordinates")
    if any(not math.isfinite(value) for value in losses):
        raise AnalysisError("quadratic fit received a nonfinite loss")
    xs = [math.log2(eta) for eta in etas]
    sums = [sum(x**power for x in xs) for power in range(5)]
    matrix = [
        [sums[4], sums[3], sums[2]],
        [sums[3], sums[2], sums[1]],
        [sums[2], sums[1], sums[0]],
    ]
    vector = [
        sum(y * x * x for x, y in zip(xs, losses)),
        sum(y * x for x, y in zip(xs, losses)),
        sum(losses),
    ]
    coefficient_a, coefficient_b, coefficient_c = solve3(matrix, vector)
    vertex = (
        -coefficient_b / (2.0 * coefficient_a)
        if coefficient_a
        else math.nan
    )
    interior = (
        coefficient_a > 0
        and min(xs) + 1e-12 < vertex < max(xs) - 1e-12
    )
    eta_star = 2.0**vertex if interior else None
    tuned_loss = (
        coefficient_a * vertex * vertex + coefficient_b * vertex + coefficient_c
        if interior
        else None
    )
    return {
        "a": coefficient_a,
        "b": coefficient_b,
        "c": coefficient_c,
        "vertex_log2_eta": vertex if math.isfinite(vertex) else None,
        "eta_star": eta_star,
        "tuned_loss": tuned_loss,
        "interior": interior,
        "status": "INTERIOR" if interior else "UNBRACKETED",
        "low_edge_log2_eta": min(xs),
        "high_edge_log2_eta": max(xs),
    }


def parse_node_roots(values: list[str], option_name: str) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise AnalysisError(f"{option_name} must be NODE=PATH, got {value!r}")
        node, raw_path = value.split("=", 1)
        if node in result or not node or not raw_path:
            raise AnalysisError(f"invalid or duplicate {option_name} {value!r}")
        result[node] = Path(raw_path).resolve()
    return result


def completed_attempt(cell: dict, root: Path) -> tuple[Path, dict, int]:
    attempt2 = root / cell["cell_id"] / "attempt-2"
    evidence2_path = attempt2 / "evidence.json"
    if evidence2_path.is_file():
        evidence2 = read_json(evidence2_path)
        if evidence2.get("status") != "COMPLETED":
            raise AnalysisError(
                f"{cell['cell_id']}: registered retry exists but is "
                f"{evidence2.get('status')}"
            )
        return attempt2, evidence2, 2
    attempt1 = root / cell["cell_id"] / "attempt-1"
    evidence1_path = attempt1 / "evidence.json"
    if not evidence1_path.is_file():
        raise AnalysisError(f"{cell['cell_id']}: attempt-1 evidence is missing")
    evidence1 = read_json(evidence1_path)
    if evidence1.get("status") != "COMPLETED":
        raise AnalysisError(
            f"{cell['cell_id']}: attempt 1 is {evidence1.get('status')}"
        )
    return attempt1, evidence1, 1


def validate_manifest(manifest: dict, campaign: str, manifest_path: Path) -> None:
    if campaign == "v5":
        if sha256_file(manifest_path) != V5_LAUNCH_MANIFEST_SHA256:
            raise AnalysisError("v5 launch manifest differs from the disclosed hash")
        expected_schema = "yeto_outer_mup_v5_snoo_launch_manifest_v1"
        expected_stage = "V5_SNOO_INTERIOR"
        expected_cells = V5_EXPECTED_CELLS
        points = 6
        grids = V5_ETA_GRIDS
    elif campaign == "v5b":
        expected_schema = "yeto_outer_mup_v5b_snoo_regrid_launch_manifest_v1"
        expected_stage = "V5B_SNOO_REGRID"
        expected_cells = V5B_EXPECTED_CELLS
        points = 5
        grids = V5B_ETA_GRIDS
    else:  # pragma: no cover - internal caller contract
        raise AnalysisError(f"unknown campaign {campaign}")
    if manifest.get("schema") != expected_schema:
        raise AnalysisError(f"{campaign} launch manifest schema mismatch")
    if manifest.get("stage") != expected_stage:
        raise AnalysisError(f"{campaign} launch manifest stage mismatch")
    cells = manifest.get("cells", [])
    if len(cells) != expected_cells:
        raise AnalysisError(
            f"{campaign}: expected {expected_cells} cells, found {len(cells)}"
        )
    coordinates = set()
    for cell in cells:
        condition = cell.get("condition")
        seed = cell.get("seed")
        eta = cell.get("eta")
        eta_index = cell.get("eta_index")
        if condition not in CONDITIONS or seed not in SEEDS:
            raise AnalysisError(f"invalid condition/seed in {cell.get('cell_id')}")
        if not isinstance(eta_index, int) or not 0 <= eta_index < points:
            raise AnalysisError(f"invalid eta index in {cell.get('cell_id')}")
        if not math.isclose(
            float(eta), grids[condition][eta_index], rel_tol=0, abs_tol=1e-15
        ):
            raise AnalysisError(f"registered eta mismatch in {cell.get('cell_id')}")
        coordinate = (condition, seed, eta_index)
        if coordinate in coordinates:
            raise AnalysisError(f"duplicate {campaign} scientific coordinate {coordinate}")
        coordinates.add(coordinate)
    if len(coordinates) != expected_cells:
        raise AnalysisError(f"{campaign} scientific coordinate count mismatch")


def load_losses(
    manifest: dict,
    node_roots: dict[str, Path],
    campaign: str,
) -> tuple[dict[tuple[str, int, float], float], list[dict], list[str]]:
    losses = {}
    records = []
    errors = []
    for cell in manifest["cells"]:
        cell_id = cell.get("cell_id", "<missing-cell-id>")
        try:
            node = cell["assignment"]["node"]
            if node not in node_roots:
                raise AnalysisError(f"no {campaign} results root supplied for node {node}")
            attempt, evidence, attempt_number = completed_attempt(cell, node_roots[node])
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
            if evidence.get("attempt_number") not in (None, attempt_number):
                raise AnalysisError("evidence attempt number mismatch")
            results_path = attempt / "report" / "results.jsonl"
            observed = evidence.get("observed_artifacts", {}).get("results", {})
            if observed.get("sha256") != sha256_file(results_path):
                raise AnalysisError("results hash does not match validated evidence")
            rows = read_jsonl(results_path)
            if len(rows) != 1:
                raise AnalysisError(f"expected one result row, found {len(rows)}")
            loss = rows[0].get("eval_loss")
            if not isinstance(loss, (int, float)) or not math.isfinite(loss):
                raise AnalysisError("endpoint evaluation loss is not finite")
            key = (cell["condition"], int(cell["seed"]), float(cell["eta"]))
            if key in losses:
                raise AnalysisError("duplicate scientific cell coordinate")
            losses[key] = float(loss)
            records.append(
                {
                    "campaign": campaign,
                    "cell_id": cell_id,
                    "node": node,
                    "gpu": cell["assignment"]["gpu"],
                    "attempt": attempt_number,
                    "condition": cell["condition"],
                    "eta": cell["eta"],
                    "seed": cell["seed"],
                    "eval_loss": float(loss),
                    "evidence_path": str(attempt / "evidence.json"),
                    "evidence_sha256": sha256_file(attempt / "evidence.json"),
                    "results_sha256": observed["sha256"],
                }
            )
        except (AnalysisError, KeyError, OSError, TypeError, ValueError) as exc:
            errors.append(f"{campaign}:{cell_id}: {exc}")
    return losses, records, errors


def fit_condition(
    campaign_losses: dict[str, dict[tuple[str, int, float], float]],
    condition: str,
    selected_seeds: list[int] | tuple[int, ...] = SEEDS,
) -> dict:
    points = []
    for campaign, grids in (("v5b", V5B_ETA_GRIDS), ("v5", V5_ETA_GRIDS)):
        for eta in grids[condition]:
            values = [
                campaign_losses[campaign][(condition, seed, eta)]
                for seed in selected_seeds
            ]
            points.append(
                {
                    "campaign": campaign,
                    "eta": eta,
                    "log2_eta": math.log2(eta),
                    "seed_mean_loss": sum(values) / len(values),
                }
            )
    points.sort(key=lambda item: item["eta"])
    etas = [item["eta"] for item in points]
    means = [item["seed_mean_loss"] for item in points]
    return {
        **fit_quadratic(etas, means),
        "etas": etas,
        "seed_mean_losses": means,
        "points": points,
        "point_count": len(points),
        "campaign_point_counts": {"v5": 6, "v5b": 5},
    }


def paired_bootstrap(
    campaign_losses: dict[str, dict[tuple[str, int, float], float]],
) -> dict:
    rng = random.Random(BOOTSTRAP_SEED)
    deltas = {"b_minus_a": [], "c_minus_a": []}
    eta_stars = {condition: [] for condition in CONDITIONS}
    invalid = 0
    for _ in range(BOOTSTRAP_REPLICATES):
        selected = [SEEDS[rng.randrange(len(SEEDS))] for _ in SEEDS]
        fits = {
            condition: fit_condition(campaign_losses, condition, selected)
            for condition in CONDITIONS
        }
        if not all(fit["interior"] for fit in fits.values()):
            invalid += 1
            continue
        for condition in CONDITIONS:
            eta_stars[condition].append(float(fits[condition]["eta_star"]))
        deltas["b_minus_a"].append(
            float(fits["b"]["tuned_loss"] - fits["a"]["tuned_loss"])
        )
        deltas["c_minus_a"].append(
            float(fits["c"]["tuned_loss"] - fits["a"]["tuned_loss"])
        )
    valid = BOOTSTRAP_REPLICATES - invalid
    status = "VALID" if valid >= MIN_VALID_BOOTSTRAP_REPLICATES else "NOT_EVALUABLE"
    result = {
        "method": "paired_nonparametric_training_seed_combined_curve_bootstrap",
        "pairing": (
            "one common five-index seed resample is used across both campaigns, "
            "all eta points, and all three conditions"
        ),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "valid_replicates": valid,
        "invalid_unbracketed_replicates": invalid,
        "minimum_valid_replicates": MIN_VALID_BOOTSTRAP_REPLICATES,
        "status": status,
        "deltas": {},
        "eta_star_ci_95": {},
    }
    if valid:
        for label, values in deltas.items():
            result["deltas"][label] = {
                "ci_95": {
                    "low": quantile(values, 0.025),
                    "high": quantile(values, 0.975),
                }
            }
        for condition, values in eta_stars.items():
            result["eta_star_ci_95"][condition] = {
                "low": quantile(values, 0.025),
                "high": quantile(values, 0.975),
            }
    return result


def signed(value: float) -> str:
    return f"{value:+.4f}"


def note_line(verdict: str, deltas: dict) -> str:
    def render(label: str) -> str:
        item = deltas[label]
        ci = item["ci_95"]
        return f"{signed(item['point'])} [{signed(ci['low'])},{signed(ci['high'])}]"

    return (
        f"G5B VERDICT: {verdict} "
        f"b-a={render('b_minus_a')} c-a={render('c_minus_a')}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v5-manifest", type=Path, required=True)
    parser.add_argument("--v5b-manifest", type=Path, required=True)
    parser.add_argument("--v5-node-root", action="append", default=[], metavar="NODE=PATH")
    parser.add_argument("--v5b-node-root", action="append", default=[], metavar="NODE=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    v5_manifest = read_json(args.v5_manifest)
    v5b_manifest = read_json(args.v5b_manifest)
    v5_roots = parse_node_roots(args.v5_node_root, "--v5-node-root")
    v5b_roots = parse_node_roots(args.v5b_node_root, "--v5b-node-root")
    validate_manifest(v5_manifest, "v5", args.v5_manifest)
    validate_manifest(v5b_manifest, "v5b", args.v5b_manifest)
    v5_losses, v5_records, v5_errors = load_losses(v5_manifest, v5_roots, "v5")
    v5b_losses, v5b_records, v5b_errors = load_losses(
        v5b_manifest, v5b_roots, "v5b"
    )
    campaign_losses = {"v5": v5_losses, "v5b": v5b_losses}
    records = v5_records + v5b_records
    errors = v5_errors + v5b_errors
    curves = {}
    bootstrap = None
    if not errors and len(records) == EXPECTED_COMBINED_CELLS:
        curves = {
            condition: fit_condition(campaign_losses, condition)
            for condition in CONDITIONS
        }
        bootstrap = paired_bootstrap(campaign_losses)

    all_work_valid = not errors and len(records) == EXPECTED_COMBINED_CELLS
    all_optima_interior = bool(curves) and all(
        curves[condition]["interior"] for condition in CONDITIONS
    )
    bootstrap_valid = bool(bootstrap) and bootstrap["status"] == "VALID"
    evaluable = all_work_valid and all_optima_interior and bootstrap_valid
    deltas = {}
    verdict = None
    frozen_note = None
    if all_optima_interior:
        deltas = {
            "b_minus_a": {
                "point": curves["b"]["tuned_loss"] - curves["a"]["tuned_loss"]
            },
            "c_minus_a": {
                "point": curves["c"]["tuned_loss"] - curves["a"]["tuned_loss"]
            },
        }
        if bootstrap:
            for label in deltas:
                if label in bootstrap["deltas"]:
                    deltas[label]["ci_95"] = bootstrap["deltas"][label]["ci_95"]
    if evaluable:
        b_ci = deltas["b_minus_a"]["ci_95"]
        if b_ci["high"] < 0.0:
            verdict = "SNOO_HELPS"
        elif b_ci["low"] > 0.0:
            verdict = "SNOO_HURTS"
        else:
            verdict = "SNOO_NULL"
        frozen_note = note_line(verdict, deltas)

    readout = {
        "schema": "yeto_outer_mup_v5b_snoo_combined_readout_v1",
        "status": "COMPLETE" if evaluable else "NOT_EVALUABLE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_git_commit": v5b_manifest.get("source", {}).get("git_commit"),
        "launch_manifests": {
            "v5": {
                "path": str(args.v5_manifest.resolve()),
                "sha256": sha256_file(args.v5_manifest),
                "source_git_commit": v5_manifest.get("source", {}).get("git_commit"),
            },
            "v5b": {
                "path": str(args.v5b_manifest.resolve()),
                "sha256": sha256_file(args.v5b_manifest),
                "source_git_commit": v5b_manifest.get("source", {}).get("git_commit"),
            },
        },
        "contract": v5b_manifest.get("contract"),
        "expected_cells": {
            "v5": V5_EXPECTED_CELLS,
            "v5b": V5B_EXPECTED_CELLS,
            "combined": EXPECTED_COMBINED_CELLS,
        },
        "observed_completed_cells": {
            "v5": len(v5_records),
            "v5b": len(v5b_records),
            "combined": len(records),
        },
        "invalid_cells": errors,
        "cell_evidence_registry_sha256": canonical_sha256(
            sorted(records, key=lambda item: (item["campaign"], item["cell_id"]))
        ),
        "cell_evidence": sorted(
            records, key=lambda item: (item["campaign"], item["cell_id"])
        ),
        "combined_eta_curves": curves,
        "paired_bootstrap": bootstrap,
        "deltas": deltas,
        "G5B": {
            "gate_id": "G5B_snoo_combined_regrid_tuned_optima",
            "evaluable": evaluable,
            "verdict": verdict,
            "closed_verdicts": ["SNOO_HELPS", "SNOO_NULL", "SNOO_HURTS"],
            "requirements": {
                "all_90_v5_and_75_v5b_cells_valid": all_work_valid,
                "all_three_combined_pooled_optima_interior": all_optima_interior,
                "paired_bootstrap_at_least_9500_valid": bootstrap_valid,
            },
        },
        "note_line": frozen_note,
    }
    output = args.output.resolve()
    write_json_atomic(output, readout)
    digest = sha256_file(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": digest,
                "status": readout["status"],
                "G5B": verdict,
                "note_line": frozen_note,
            },
            sort_keys=True,
        )
    )
    return 0 if evaluable else 2


if __name__ == "__main__":
    raise SystemExit(main())
