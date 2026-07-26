#!/usr/bin/env python3
"""Apply the frozen G5 analysis to the SNOO interior-optimum repair grid."""

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
EXPECTED_CELLS = 90
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260726
MIN_VALID_BOOTSTRAP_REPLICATES = 9_500
CODE_TRUE_MATCH_FACTOR = (1.0 - 0.9**5) / (1.0 - 0.9)
V3_BEST_CELLS = {
    "a": 0.5946035575013605,
    "b": 0.14508326803033197,
    "c": 0.5941304909110126,
}
ETA_GRIDS = {
    condition: tuple(center * 2.0 ** ((index - 2.5) / 2.0) for index in range(6))
    for condition, center in V3_BEST_CELLS.items()
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
    if len(etas) != 6 or len(losses) != 6:
        raise AnalysisError("registered eta fit requires exactly six points")
    if any(not math.isfinite(value) or value <= 0 for value in etas):
        raise AnalysisError("quadratic fit received an invalid eta")
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
    a, b, c = solve3(matrix, vector)
    vertex = -b / (2.0 * a) if a else math.nan
    interior = a > 0 and min(xs) + 1e-12 < vertex < max(xs) - 1e-12
    eta_star = 2.0**vertex if interior else None
    tuned_loss = a * vertex * vertex + b * vertex + c if interior else None
    return {
        "a": a,
        "b": b,
        "c": c,
        "vertex_log2_eta": vertex if math.isfinite(vertex) else None,
        "eta_star": eta_star,
        "tuned_loss": tuned_loss,
        "interior": interior,
        "status": "INTERIOR" if interior else "UNBRACKETED",
    }


def parse_node_roots(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise AnalysisError(f"--node-root must be NODE=PATH, got {value!r}")
        node, raw_path = value.split("=", 1)
        if node in result or not node or not raw_path:
            raise AnalysisError(f"invalid or duplicate --node-root {value!r}")
        result[node] = Path(raw_path).resolve()
    return result


def completed_attempt(cell: dict, root: Path) -> tuple[Path, dict, int]:
    attempt2 = root / cell["cell_id"] / "attempt-2"
    evidence2_path = attempt2 / "evidence.json"
    if evidence2_path.is_file():
        evidence2 = read_json(evidence2_path)
        if evidence2.get("status") != "COMPLETED":
            raise AnalysisError(
                f"{cell['cell_id']}: registered retry exists but is {evidence2.get('status')}"
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


def validate_manifest(manifest: dict) -> None:
    if manifest.get("schema") != "yeto_outer_mup_v5_snoo_launch_manifest_v1":
        raise AnalysisError("launch manifest schema mismatch")
    if manifest.get("stage") != "V5_SNOO_INTERIOR":
        raise AnalysisError("launch manifest stage mismatch")
    cells = manifest.get("cells", [])
    if len(cells) != EXPECTED_CELLS:
        raise AnalysisError(f"expected {EXPECTED_CELLS} cells, found {len(cells)}")
    coordinates = set()
    for cell in cells:
        condition = cell.get("condition")
        seed = cell.get("seed")
        eta = cell.get("eta")
        eta_index = cell.get("eta_index")
        if condition not in CONDITIONS or seed not in SEEDS:
            raise AnalysisError(f"invalid condition/seed in {cell.get('cell_id')}")
        if not isinstance(eta_index, int) or not 0 <= eta_index < 6:
            raise AnalysisError(f"invalid eta index in {cell.get('cell_id')}")
        if not math.isclose(float(eta), ETA_GRIDS[condition][eta_index], rel_tol=0, abs_tol=1e-15):
            raise AnalysisError(f"registered eta mismatch in {cell.get('cell_id')}")
        coordinate = (condition, seed, eta_index)
        if coordinate in coordinates:
            raise AnalysisError(f"duplicate scientific coordinate {coordinate}")
        coordinates.add(coordinate)
    if len(coordinates) != EXPECTED_CELLS:
        raise AnalysisError("scientific coordinate count mismatch")


def load_losses(
    manifest: dict, node_roots: dict[str, Path]
) -> tuple[dict[tuple[str, int, float], float], list[dict], list[str]]:
    losses = {}
    records = []
    errors = []
    for cell in manifest["cells"]:
        cell_id = cell.get("cell_id", "<missing-cell-id>")
        try:
            node = cell["assignment"]["node"]
            if node not in node_roots:
                raise AnalysisError(f"no results root supplied for node {node}")
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
            errors.append(f"{cell_id}: {exc}")
    return losses, records, errors


def fit_condition(
    losses: dict[tuple[str, int, float], float],
    condition: str,
    selected_seeds: list[int] | tuple[int, ...] = SEEDS,
) -> dict:
    means = []
    for eta in ETA_GRIDS[condition]:
        values = [losses[(condition, seed, eta)] for seed in selected_seeds]
        means.append(sum(values) / len(values))
    return {
        **fit_quadratic(list(ETA_GRIDS[condition]), means),
        "etas": list(ETA_GRIDS[condition]),
        "seed_mean_losses": means,
    }


def paired_bootstrap(
    losses: dict[tuple[str, int, float], float],
) -> dict:
    rng = random.Random(BOOTSTRAP_SEED)
    deltas = {"b_minus_a": [], "c_minus_a": []}
    eta_stars = {condition: [] for condition in CONDITIONS}
    invalid = 0
    for _ in range(BOOTSTRAP_REPLICATES):
        selected = [SEEDS[rng.randrange(len(SEEDS))] for _ in SEEDS]
        fits = {condition: fit_condition(losses, condition, selected) for condition in CONDITIONS}
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
        "method": "paired_nonparametric_training_seed_curve_bootstrap",
        "pairing": "one common five-index resample is used for all three condition curves",
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
        f"G5 VERDICT: {verdict} "
        f"b-a={render('b_minus_a')} c-a={render('c_minus_a')}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--node-root", action="append", default=[], metavar="NODE=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    node_roots = parse_node_roots(args.node_root)
    validate_manifest(manifest)
    losses, records, errors = load_losses(manifest, node_roots)
    curves = {}
    bootstrap = None
    if not errors and len(losses) == EXPECTED_CELLS:
        curves = {condition: fit_condition(losses, condition) for condition in CONDITIONS}
        bootstrap = paired_bootstrap(losses)

    all_work_valid = not errors and len(records) == EXPECTED_CELLS
    all_optima_interior = bool(curves) and all(
        curves[condition]["interior"] for condition in CONDITIONS
    )
    bootstrap_valid = bool(bootstrap) and bootstrap["status"] == "VALID"
    evaluable = all_work_valid and all_optima_interior and bootstrap_valid
    deltas = {}
    verdict = None
    frozen_note = None
    if curves:
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
        "schema": "yeto_outer_mup_v5_snoo_readout_v1",
        "status": "COMPLETE" if evaluable else "NOT_EVALUABLE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_git_commit": manifest.get("source", {}).get("git_commit"),
        "launch_manifest_sha256": sha256_file(args.manifest),
        "contract": manifest.get("contract"),
        "expected_cells": EXPECTED_CELLS,
        "observed_completed_cells": len(records),
        "invalid_cells": errors,
        "cell_evidence_registry_sha256": canonical_sha256(
            sorted(records, key=lambda item: item["cell_id"])
        ),
        "cell_evidence": sorted(records, key=lambda item: item["cell_id"]),
        "eta_curves": curves,
        "paired_bootstrap": bootstrap,
        "deltas": deltas,
        "G5": {
            "gate_id": "G5_snoo_interior_tuned_optima",
            "evaluable": evaluable,
            "verdict": verdict,
            "closed_verdicts": ["SNOO_HELPS", "SNOO_NULL", "SNOO_HURTS"],
            "requirements": {
                "all_90_cells_valid": all_work_valid,
                "all_three_pooled_optima_interior": all_optima_interior,
                "paired_bootstrap_valid": bootstrap_valid,
            },
        },
        "note_line": frozen_note,
    }
    write_json_atomic(args.output.resolve(), readout)
    digest = sha256_file(args.output.resolve())
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n"
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": digest,
                "status": readout["status"],
                "G5": verdict,
                "note_line": frozen_note,
            },
            sort_keys=True,
        )
    )
    return 0 if evaluable else 2


if __name__ == "__main__":
    raise SystemExit(main())
