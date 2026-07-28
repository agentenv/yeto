"""Frozen fitting/bootstrap helpers for tonight-8.5 registered studies."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Iterable


BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_MIN_VALID = 7_500
BOOTSTRAP_SEED = 20_260_729
NEAR_BRACKET_BITS = 0.5
SCAN_SEEDS = (981, 983, 991)
SCAN_T = (2, 5, 20)


class AnalysisError(RuntimeError):
    """Raised when frozen evidence or manifest structure is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnalysisError(f"{path}: expected a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict]:
    try:
        values = [json.loads(line) for line in path.read_text().splitlines() if line]
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read {path}: {exc}") from exc
    if not values or any(not isinstance(value, dict) for value in values):
        raise AnalysisError(f"{path}: expected nonempty JSON objects")
    return values


def solve3(matrix: list[list[float]], vector: list[float]) -> list[float]:
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-14:
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


def fit_quadratic(
    etas: Iterable[float], losses: Iterable[float], near_bits: float = NEAR_BRACKET_BITS
) -> dict:
    eta_values = [float(value) for value in etas]
    loss_values = [float(value) for value in losses]
    if len(eta_values) != len(loss_values) or len(eta_values) < 3:
        raise AnalysisError("quadratic fit requires at least three matched points")
    if any(not math.isfinite(value) or value <= 0 for value in eta_values):
        raise AnalysisError("eta values must be finite and positive")
    if any(not math.isfinite(value) for value in loss_values):
        raise AnalysisError("loss values must be finite")
    xs = [math.log2(value) for value in eta_values]
    features = [[x * x, x, 1.0] for x in xs]
    matrix = [
        [sum(row[i] * row[j] for row in features) for j in range(3)] for i in range(3)
    ]
    vector = [
        sum(row[i] * value for row, value in zip(features, loss_values))
        for i in range(3)
    ]
    a, b, c = solve3(matrix, vector)
    vertex = -b / (2.0 * a) if a else math.nan
    strict = a > 0 and math.isfinite(vertex) and min(xs) < vertex < max(xs)
    near = (
        a > 0
        and math.isfinite(vertex)
        and min(xs) - near_bits <= vertex <= max(xs) + near_bits
    )
    return {
        "a": a,
        "b": b,
        "c": c,
        "vertex_log2_eta": vertex if math.isfinite(vertex) else None,
        "eta_star": 2.0**vertex if near else None,
        "strict_interior": strict,
        "near_bracketed": near and not strict,
        "accepted": near,
        "status": "INTERIOR" if strict else "NEAR_BRACKETED" if near else "UNBRACKETED",
    }


def quantile(values: list[float], probability: float) -> float:
    if not values:
        raise AnalysisError("quantile of empty values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def cell_loss(cell: dict, result_root: Path) -> tuple[float, dict]:
    attempts = cell.get("attempts", [1, 2])
    completed = []
    for attempt in attempts:
        root = result_root / cell["cell_id"] / f"attempt-{attempt}"
        evidence_path = root / "evidence.json"
        if not evidence_path.is_file():
            continue
        evidence = read_json(evidence_path)
        if evidence.get("status") in {"COMPLETED", "SCIENTIFIC_DIVERGENCE"}:
            completed.append((root, evidence, attempt))
    if not completed:
        raise AnalysisError(f"{cell['cell_id']}: no completed attempt")
    if len(completed) > 1 and not cell.get("attempt2_supersedes_attempt1"):
        raise AnalysisError(f"{cell['cell_id']}: multiple completed attempts")
    root, evidence, attempt = max(completed, key=lambda record: record[2])
    if evidence.get("status") != "COMPLETED":
        raise AnalysisError(f"{cell['cell_id']}: scientific divergence")
    if evidence.get("command_hash") != cell.get("command_hash") and attempt == 1:
        raise AnalysisError(f"{cell['cell_id']}: command hash mismatch")
    result_path = root / "report" / "results.jsonl"
    rows = read_jsonl(result_path)
    if len(rows) != 1:
        raise AnalysisError(f"{cell['cell_id']}: expected exactly one result row")
    loss = rows[0].get("eval_loss")
    if not isinstance(loss, (int, float)) or not math.isfinite(loss):
        raise AnalysisError(f"{cell['cell_id']}: nonfinite endpoint loss")
    return float(loss), {
        "attempt": attempt,
        "evidence_path": str(root / "evidence.json"),
        "evidence_sha256": sha256_file(root / "evidence.json"),
        "results_path": str(result_path),
        "results_sha256": sha256_file(result_path),
    }


def load_program_losses(
    manifest: dict, program: str, result_root: Path
) -> tuple[dict, list]:
    losses = {}
    records = []
    cells = [cell for cell in manifest["cells"] if cell["program"] == program]
    expected = 72
    if len(cells) != expected:
        raise AnalysisError(f"{program}: expected {expected} cells, got {len(cells)}")
    for cell in cells:
        loss, evidence = cell_loss(cell, result_root)
        key = (
            int(cell["t"]),
            str(cell["arm"]),
            int(cell["eta_index"]),
            int(cell["seed"]),
        )
        if key in losses:
            raise AnalysisError(f"{program}: duplicate cell key {key}")
        losses[key] = (float(cell["eta"]), loss)
        records.append(
            {
                "cell_id": cell["cell_id"],
                "t": cell["t"],
                "arm": cell["arm"],
                "eta": cell["eta"],
                "eta_index": cell["eta_index"],
                "seed": cell["seed"],
                "eval_loss": loss,
                **evidence,
            }
        )
    return losses, records


def scan_fit(losses: dict, t: int, arm: str, seed_indices: tuple[int, ...]) -> dict:
    eta_indices = sorted({key[2] for key in losses if key[0] == t and key[1] == arm})
    if eta_indices != list(range(4)):
        raise AnalysisError(f"T{t}/{arm}: eta indices are not 0..3")
    etas = []
    means = []
    for eta_index in eta_indices:
        rows = [
            losses[(t, arm, eta_index, SCAN_SEEDS[index])] for index in seed_indices
        ]
        eta_values = {row[0] for row in rows}
        if len(eta_values) != 1:
            raise AnalysisError(f"T{t}/{arm}/eta{eta_index}: eta mismatch")
        etas.append(rows[0][0])
        means.append(sum(row[1] for row in rows) / len(rows))
    return {"etas": etas, "seed_mean_losses": means, **fit_quadratic(etas, means)}


def analyze_scan(manifest: dict, program: str, result_root: Path) -> dict:
    if program not in {"v12", "v13"}:
        raise AnalysisError(f"unsupported scan program {program}")
    losses, records = load_program_losses(manifest, program, result_root)
    fits = {}
    log2_d = {}
    for t in SCAN_T:
        for arm in ("mu0", "mu09"):
            fit = scan_fit(losses, t, arm, (0, 1, 2))
            fits[f"T{t}_{arm}"] = fit
            if not fit["accepted"]:
                continue
        if fits[f"T{t}_mu0"]["accepted"] and fits[f"T{t}_mu09"]["accepted"]:
            log2_d[t] = math.log2(
                fits[f"T{t}_mu09"]["eta_star"] / fits[f"T{t}_mu0"]["eta_star"] / 0.1
            )

    rng = random.Random(BOOTSTRAP_SEED + (12 if program == "v12" else 13))
    bootstrap_differences = {(2, 5): [], (5, 20): []}
    valid = 0
    for _ in range(BOOTSTRAP_DRAWS):
        draw = tuple(rng.randrange(3) for _ in range(3))
        draw_log_d = {}
        try:
            for t in SCAN_T:
                mu0 = scan_fit(losses, t, "mu0", draw)
                mu09 = scan_fit(losses, t, "mu09", draw)
                if not mu0["accepted"] or not mu09["accepted"]:
                    raise AnalysisError("unaccepted bootstrap curve")
                draw_log_d[t] = math.log2(mu09["eta_star"] / mu0["eta_star"] / 0.1)
        except (AnalysisError, ValueError, OverflowError, ZeroDivisionError):
            continue
        valid += 1
        for pair in bootstrap_differences:
            bootstrap_differences[pair].append(
                draw_log_d[pair[0]] - draw_log_d[pair[1]]
            )

    differences = {}
    for pair, values in bootstrap_differences.items():
        key = f"T{pair[0]}_minus_T{pair[1]}"
        differences[key] = {
            "point_bits": (
                log2_d.get(pair[0], math.nan) - log2_d.get(pair[1], math.nan)
                if pair[0] in log2_d and pair[1] in log2_d
                else None
            ),
            "bootstrap_95_percentile_ci_bits": (
                [quantile(values, 0.025), quantile(values, 0.975)] if values else None
            ),
        }
    complete = len(records) == 72
    all_curves = len(log2_d) == 3
    point_monotone = all_curves and log2_d[2] > log2_d[5] > log2_d[20]
    ci_monotone = all(
        record["bootstrap_95_percentile_ci_bits"] is not None
        and record["bootstrap_95_percentile_ci_bits"][0] > 0
        for record in differences.values()
    )
    evaluable = complete and all_curves and valid >= BOOTSTRAP_MIN_VALID
    verdict = (
        "PASS"
        if evaluable and point_monotone and ci_monotone
        else "FAIL"
        if evaluable
        else "NOT_EVALUABLE"
    )

    smollm2_nesterov = {
        "T2": 4.153638091620729,
        "T5": 2.554472462361996,
        "T20": 1.168877658562076,
        "source": "G6 S=2560 three-seed point fits",
    }
    comparison = {"smollm2_nesterov_D": smollm2_nesterov}
    if program == "v12":
        comparison.update(
            {
                "convention": "heavy-ball SGD momentum without lookahead",
                "lean_citation": (
                    "lean-mechanism/LeanMechanism/FiniteHorizonOuter.lean: "
                    "terminalMultiplier .heavyBall is geometricPrefix T; "
                    "heavyBallCoeff is the accumulated sum of those prefixes"
                ),
                "heavy_ball_terminal_multiplier": {
                    f"T{t}": (1 - 0.9**t) / (1 - 0.9) for t in SCAN_T
                },
                "heavy_ball_constant_gradient_D": {
                    f"T{t}": 1 / (0.1 * ((1 - 0.9**t) / (1 - 0.9))) for t in SCAN_T
                },
            }
        )
    else:
        comparison.update(
            {
                "family_and_data": "EleutherAI/pythia-160m on HuggingFaceH4/ultrachat_200k",
                "comparison_is_descriptive_not_a_gate": True,
            }
        )
    return {
        "schema": f"yeto_tonight85_{program}_readout_v1",
        "program": program,
        "gate": {
            "name": "G12" if program == "v12" else "G13",
            "verdict": verdict,
            "evaluable": evaluable,
            "success_rule": (
                "D(2)>D(5)>D(20) in point estimates and the lower endpoint of "
                "each paired 95% bootstrap CI for adjacent log2-D differences is >0"
            ),
            "conditions": {
                "complete_72_cells": complete,
                "all_six_curves_accepted": all_curves,
                "valid_bootstrap_at_least_7500": valid >= BOOTSTRAP_MIN_VALID,
                "point_estimates_monotone": point_monotone,
                "adjacent_difference_cis_above_zero": ci_monotone,
            },
        },
        "D": {f"T{t}": 2.0**value for t, value in sorted(log2_d.items())},
        "log2_D": {f"T{t}": value for t, value in sorted(log2_d.items())},
        "adjacent_differences": differences,
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "valid": valid,
            "seed": BOOTSTRAP_SEED + (12 if program == "v12" else 13),
            "paired_seed_resampling": True,
        },
        "fits": fits,
        "comparison": comparison,
        "cell_records": records,
    }


def analyze_v11(manifest: dict, predictions: dict, result_root: Path) -> dict:
    cells = [cell for cell in manifest["cells"] if cell["program"] == "v11_truth"]
    if len(cells) != 20:
        raise AnalysisError(f"v11 truth manifest has {len(cells)} cells, expected 20")
    coordinate_results = {}
    cell_records = []
    for coordinate_id in ("smollm2_135m_t80", "smollm2_1p7b_t40"):
        coordinate_cells = [
            cell for cell in cells if cell["coordinate_id"] == coordinate_id
        ]
        losses = {}
        etas = {}
        for cell in coordinate_cells:
            loss, evidence = cell_loss(cell, result_root)
            losses[(cell["eta_index"], cell["seed"])] = loss
            etas[cell["eta_index"]] = cell["eta"]
            cell_records.append(
                {"cell_id": cell["cell_id"], "eval_loss": loss, **evidence}
            )
        ordered_eta_indices = list(range(5))
        mean_losses = [
            sum(losses[(index, seed)] for seed in (971, 977)) / 2
            for index in ordered_eta_indices
        ]
        fit = fit_quadratic([etas[index] for index in ordered_eta_indices], mean_losses)
        predicted = float(
            predictions["coordinates"][coordinate_id]["predicted_eta_star_raw"]
        )
        error = abs(math.log2(fit["eta_star"] / predicted)) if fit["accepted"] else None
        coordinate_results[coordinate_id] = {
            "predicted_eta_star_raw": predicted,
            "ground_truth_fit": fit,
            "ground_truth_seed_mean_losses": mean_losses,
            "absolute_error_bits": error,
            "within_registered_0p35_bit_band": error is not None and error <= 0.35,
        }
    complete = len(cell_records) == 20
    evaluable = complete and all(
        record["ground_truth_fit"]["accepted"] for record in coordinate_results.values()
    )
    successes = sum(
        record["within_registered_0p35_bit_band"]
        for record in coordinate_results.values()
    )
    verdict = (
        "PASS"
        if evaluable and successes >= 1
        else "FAIL"
        if evaluable
        else "NOT_EVALUABLE"
    )
    return {
        "schema": "yeto_tonight85_v11_readout_v1",
        "gate": {
            "name": "G11",
            "verdict": verdict,
            "evaluable": evaluable,
            "success_rule": (
                "absolute log2 prediction error <=0.35 bits at at least one of "
                "the two prospectively registered coordinates"
            ),
            "successful_coordinates": successes,
        },
        "coordinates": coordinate_results,
        "prediction_preimage_canonical_sha256": predictions.get(
            "prediction_preimage_canonical_sha256"
        ),
        "cell_records": cell_records,
    }


def append_note(note: Path, line: str) -> None:
    note.parent.mkdir(parents=True, exist_ok=True)
    with note.open("a", encoding="utf-8") as destination:
        destination.write(line.rstrip() + "\n")
