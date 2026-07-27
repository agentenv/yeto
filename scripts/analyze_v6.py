#!/usr/bin/env python3
"""Apply the frozen G6 held-out surface analysis to the v6 factorial."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path


SEEDS = (601, 607, 613)
T_GRID = (2, 5, 10, 20)
S_GRID = (2560, 5120, 10240)
ARMS = ("mu0", "raw", "corrected")
MOMENTUM_ARMS = ("raw", "corrected")
MU_HIGH = 0.9
EXPECTED_CELLS = 540
ETA_POINTS = 5
HOLDOUTS = frozenset(((2, 10240), (5, 5120), (10, 2560), (20, 5120)))
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260727
MIN_VALID_BOOTSTRAP_REPLICATES = 9_500
SUCCESS_ABS_ERROR_BITS = 0.2
SUCCESS_MIN_HOLDOUTS = 3
TRAINING_CELLS = tuple(
    sorted((t, s) for t in T_GRID for s in S_GRID if (t, s) not in HOLDOUTS)
)
SURFACE_FAMILY_ORDER = ("F1", "F2", "F3")
SURFACE_FAMILIES = {
    "F1": {
        "formula": "log2_D=gamma+alpha*u+beta*v",
        "coefficient_order": ("gamma", "alpha_T", "beta_log2_S"),
    },
    "F2": {
        "formula": "log2_D=gamma+alpha*u+beta*v+delta*u*v",
        "coefficient_order": (
            "gamma",
            "alpha_T",
            "beta_log2_S",
            "delta_T_x_log2_S",
        ),
    },
    "F3": {
        "formula": "log2_D=gamma+alpha*u+beta*v+epsilon*u^2",
        "coefficient_order": (
            "gamma",
            "alpha_T",
            "beta_log2_S",
            "epsilon_T_squared",
        ),
    },
}
LOO_TIE_TOLERANCE_BITS = 1e-12


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
    temporary = path.with_name(f".{path.name}.tmp")
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


def solve_linear(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    if len(matrix) != size or any(len(row) != size for row in matrix):
        raise AnalysisError("linear system has inconsistent dimensions")
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(size):
        pivot = max(
            range(column, size), key=lambda row: abs(augmented[row][column])
        )
        if abs(augmented[pivot][column]) < 1e-14:
            raise AnalysisError("singular normal equation")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                left - factor * right
                for left, right in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][-1] for row in range(size)]


def least_squares(features: list[list[float]], outcomes: list[float]) -> list[float]:
    if not features or len(features) != len(outcomes):
        raise AnalysisError("least-squares inputs are empty or unmatched")
    width = len(features[0])
    if any(len(row) != width for row in features):
        raise AnalysisError("least-squares feature width changed")
    matrix = [
        [sum(row[i] * row[j] for row in features) for j in range(width)]
        for i in range(width)
    ]
    vector = [
        sum(row[i] * outcome for row, outcome in zip(features, outcomes))
        for i in range(width)
    ]
    return solve_linear(matrix, vector)


def fit_quadratic(etas: list[float], losses: list[float]) -> dict:
    if len(etas) != ETA_POINTS or len(losses) != ETA_POINTS:
        raise AnalysisError("registered eta fit requires exactly five points")
    if any(not math.isfinite(value) or value <= 0 for value in etas):
        raise AnalysisError("eta fit received a nonpositive or nonfinite eta")
    if any(not math.isfinite(value) for value in losses):
        raise AnalysisError("eta fit received a nonfinite loss")
    xs = [math.log2(eta) for eta in etas]
    features = [[x * x, x, 1.0] for x in xs]
    a, b, c = least_squares(features, losses)
    vertex = -b / (2.0 * a) if a else math.nan
    interior = a > 0 and min(xs) + 1e-12 < vertex < max(xs) - 1e-12
    return {
        "a": a,
        "b": b,
        "c": c,
        "vertex_log2_eta": vertex if math.isfinite(vertex) else None,
        "eta_star": 2.0**vertex if interior else None,
        "interior": interior,
        "status": "INTERIOR" if interior else "UNBRACKETED",
    }


def surface_features(t: int, s: int, family_id: str) -> list[float]:
    """Return the registered, centered coordinates for one candidate family.

    u=(T-5)/5 and v=log2(S/5120). Centering/scaling changes coefficient units,
    but not the F1/F2/F3 function spaces named in the preregistration.
    """

    if family_id not in SURFACE_FAMILIES:
        raise AnalysisError(f"unknown surface family {family_id!r}")
    u = (t - 5.0) / 5.0
    v = math.log2(s / 5120.0)
    if family_id == "F1":
        return [1.0, u, v]
    if family_id == "F2":
        return [1.0, u, v, u * v]
    return [1.0, u, v, u * u]


def training_outcomes(
    d_values: dict[tuple[int, int, str], float], arm: str
) -> dict[tuple[int, int], float]:
    if len(TRAINING_CELLS) != 8:
        raise AnalysisError("registered training split is not eight cells")
    outcomes = {}
    for t, s in TRAINING_CELLS:
        value = d_values.get((t, s, arm))
        if value is None or not math.isfinite(value) or value <= 0:
            raise AnalysisError(f"{arm}/T{t}/S{s}: D is unavailable")
        outcomes[(t, s)] = math.log2(value)
    return outcomes


def fit_candidate_coefficients(
    outcomes: dict[tuple[int, int], float],
    family_id: str,
    cells: tuple[tuple[int, int], ...] | list[tuple[int, int]],
) -> list[float]:
    return least_squares(
        [surface_features(t, s, family_id) for t, s in cells],
        [outcomes[(t, s)] for t, s in cells],
    )


def candidate_prediction(
    family_id: str, coefficients: list[float], t: int, s: int
) -> float:
    return sum(
        feature * coefficient
        for feature, coefficient in zip(
            surface_features(t, s, family_id), coefficients
        )
    )


def fit_surface(d_values: dict[tuple[int, int, str], float], arm: str) -> dict:
    """Select F1/F2/F3 by eight-cell LOO RMSE, then refit on all eight.

    Selection is independent for each momentum arm. Held-out outcomes are not
    read by this function and therefore cannot influence the selected family.
    """

    outcomes = training_outcomes(d_values, arm)
    scores = []
    for family_id in SURFACE_FAMILY_ORDER:
        loo_predictions = {}
        squared_errors = []
        for left_out in TRAINING_CELLS:
            fold_cells = tuple(
                cell for cell in TRAINING_CELLS if cell != left_out
            )
            coefficients = fit_candidate_coefficients(
                outcomes, family_id, fold_cells
            )
            t, s = left_out
            predicted = candidate_prediction(
                family_id, coefficients, t, s
            )
            observed = outcomes[left_out]
            error = predicted - observed
            squared_errors.append(error * error)
            loo_predictions[f"T{t}_S{s}"] = {
                "observed_log2_D": observed,
                "predicted_log2_D": predicted,
                "signed_error_bits_pred_minus_obs": error,
            }
        rmse = math.sqrt(sum(squared_errors) / len(squared_errors))
        scores.append(
            {
                "family_id": family_id,
                "family": SURFACE_FAMILIES[family_id]["formula"],
                "loo_rmse_bits": rmse,
                "loo_predictions": loo_predictions,
            }
        )

    best_rmse = min(record["loo_rmse_bits"] for record in scores)
    tied = {
        record["family_id"]
        for record in scores
        if abs(record["loo_rmse_bits"] - best_rmse)
        <= LOO_TIE_TOLERANCE_BITS
    }
    selected_family = next(
        family_id for family_id in SURFACE_FAMILY_ORDER if family_id in tied
    )
    coefficients = fit_candidate_coefficients(
        outcomes, selected_family, TRAINING_CELLS
    )
    fitted = {}
    for t, s in TRAINING_CELLS:
        observed = outcomes[(t, s)]
        predicted = candidate_prediction(
            selected_family, coefficients, t, s
        )
        fitted[f"T{t}_S{s}"] = {
            "observed_log2_D": observed,
            "fitted_log2_D": predicted,
            "residual_bits": observed - predicted,
        }
    return {
        "family_id": selected_family,
        "family": SURFACE_FAMILIES[selected_family]["formula"],
        "coordinates": {"u": "(T-5)/5", "v": "log2(S/5120)"},
        "coefficient_order": list(
            SURFACE_FAMILIES[selected_family]["coefficient_order"]
        ),
        "coefficients": coefficients,
        "training_cells": [list(cell) for cell in TRAINING_CELLS],
        "training_fit": fitted,
        "model_selection": {
            "criterion": "minimum eight-fold leave-one-out RMSE in log2(D) bits",
            "selection_uses_heldout_outcomes": False,
            "family_priority": list(SURFACE_FAMILY_ORDER),
            "tie_tolerance_bits": LOO_TIE_TOLERANCE_BITS,
            "tie_break_rule": (
                "among candidates within 1e-12 bits of the minimum LOO RMSE, "
                "choose the first of F1,F2,F3"
            ),
            "candidate_scores": scores,
            "selected_family": selected_family,
        },
    }


def predict_surface(surface: dict, t: int, s: int) -> float:
    return candidate_prediction(
        surface["family_id"], surface["coefficients"], t, s
    )


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
                f"registered retry exists but is {evidence2.get('status')}"
            )
        return attempt2, evidence2, 2
    attempt1 = root / cell["cell_id"] / "attempt-1"
    evidence1_path = attempt1 / "evidence.json"
    if not evidence1_path.is_file():
        raise AnalysisError("attempt-1 evidence is missing")
    evidence1 = read_json(evidence1_path)
    if evidence1.get("status") != "COMPLETED":
        raise AnalysisError(f"attempt 1 is {evidence1.get('status')}")
    return attempt1, evidence1, 1


def load_losses(
    manifest: dict, node_roots: dict[str, Path]
) -> tuple[dict, list[dict], list[str]]:
    losses: dict[tuple[int, int, str, int, float], float] = {}
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
            if observed.get("sha256") != sha256_file(results_path):
                raise AnalysisError("results hash does not match validated evidence")
            rows = read_jsonl(results_path)
            if len(rows) != 1:
                raise AnalysisError(f"expected one result row, found {len(rows)}")
            loss = rows[0].get("eval_loss")
            if not isinstance(loss, (int, float)) or not math.isfinite(loss):
                raise AnalysisError("endpoint evaluation loss is not finite")
            key = (
                int(cell["t"]),
                int(cell["s"]),
                str(cell["arm"]),
                int(cell["seed"]),
                float(cell["eta"]),
            )
            if key in losses:
                raise AnalysisError("duplicate scientific cell coordinate")
            losses[key] = float(loss)
            records.append(
                {
                    "cell_id": cell_id,
                    "node": node,
                    "gpu": cell["assignment"]["gpu"],
                    "attempt": attempt_number,
                    "t": cell["t"],
                    "s": cell["s"],
                    "h": cell["h"],
                    "arm": cell["arm"],
                    "eta": cell["eta"],
                    "seed": cell["seed"],
                    "eval_loss": float(loss),
                    "evidence_path": str(attempt / "evidence.json"),
                    "evidence_sha256": sha256_file(attempt / "evidence.json"),
                    "results_sha256": observed["sha256"],
                }
            )
        except (AnalysisError, KeyError, OSError, ValueError) as exc:
            errors.append(f"{cell_id}: {exc}")
    return losses, records, errors


def curve_fit(
    losses: dict[tuple[int, int, str, int, float], float],
    t: int,
    s: int,
    arm: str,
    sampled_indices: list[int] | None = None,
) -> dict:
    etas = sorted(
        {key[4] for key in losses if key[:3] == (t, s, arm)}
    )
    if len(etas) != ETA_POINTS:
        raise AnalysisError(
            f"T{t}/S{s}/{arm}: expected five etas, found {etas}"
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
            key = (t, s, arm, seed, eta)
            if key not in losses:
                raise AnalysisError(f"T{t}/S{s}/{arm}: missing {key}")
            values.append(losses[key])
        means.append(sum(values) / len(values))
    fit = fit_quadratic(etas, means)
    fit.update(
        {
            "t": t,
            "s": s,
            "h": s // t,
            "arm": arm,
            "etas": etas,
            "seed_mean_losses": means,
        }
    )
    return fit


def calculate_all_curves(
    losses: dict[tuple[int, int, str, int, float], float],
    sampled_indices: list[int] | None = None,
) -> dict[tuple[int, int, str], dict]:
    return {
        (t, s, arm): curve_fit(losses, t, s, arm, sampled_indices)
        for t in T_GRID
        for s in S_GRID
        for arm in ARMS
    }


def d_from_curves(curves: dict[tuple[int, int, str], dict]) -> dict:
    result = {}
    for t in T_GRID:
        for s in S_GRID:
            baseline = curves[(t, s, "mu0")]
            result[(t, s, "mu0")] = 1.0 if baseline["interior"] else None
            for arm in MOMENTUM_ARMS:
                momentum = curves[(t, s, arm)]
                if not baseline["interior"] or not momentum["interior"]:
                    result[(t, s, arm)] = None
                else:
                    result[(t, s, arm)] = (
                        momentum["eta_star"] / baseline["eta_star"]
                    ) / (1.0 - MU_HIGH)
    return result


def heldout_analysis(d_values: dict, arm: str) -> dict:
    surface = fit_surface(d_values, arm)
    predictions = {}
    successes = 0
    for t, s in sorted(HOLDOUTS):
        observed = d_values.get((t, s, arm))
        if observed is None or observed <= 0:
            raise AnalysisError(f"held-out {arm}/T{t}/S{s} D unavailable")
        observed_log = math.log2(observed)
        predicted_log = predict_surface(surface, t, s)
        error = predicted_log - observed_log
        success = abs(error) <= SUCCESS_ABS_ERROR_BITS
        successes += int(success)
        predictions[f"T{t}_S{s}"] = {
            "T": t,
            "S": s,
            "H": s // t,
            "D_observed": observed,
            "log2_D_observed": observed_log,
            "log2_D_predicted": predicted_log,
            "D_predicted": 2.0**predicted_log,
            "signed_error_bits_pred_minus_obs": error,
            "absolute_error_bits": abs(error),
            "within_registered_0.2_bit_band": success,
        }
    return {
        "surface": surface,
        "heldout_predictions": predictions,
        "success_count": successes,
        "required_success_count": SUCCESS_MIN_HOLDOUTS,
        "pass": successes >= SUCCESS_MIN_HOLDOUTS,
    }


def joint_bootstrap(losses: dict) -> dict:
    rng = random.Random(BOOTSTRAP_SEED)
    coefficient_samples = {
        arm: {
            family_id: {
                label: []
                for label in SURFACE_FAMILIES[family_id]["coefficient_order"]
            }
            for family_id in SURFACE_FAMILY_ORDER
        }
        for arm in MOMENTUM_ARMS
    }
    selection_counts = {
        arm: {family_id: 0 for family_id in SURFACE_FAMILY_ORDER}
        for arm in MOMENTUM_ARMS
    }
    error_samples = {
        arm: {cell: [] for cell in sorted(HOLDOUTS)} for arm in MOMENTUM_ARMS
    }
    success_counts = {arm: [] for arm in MOMENTUM_ARMS}
    invalid = 0
    for _ in range(BOOTSTRAP_REPLICATES):
        draw = [rng.randrange(len(SEEDS)) for _ in SEEDS]
        try:
            curves = calculate_all_curves(losses, draw)
            if not all(curve["interior"] for curve in curves.values()):
                raise AnalysisError("bootstrap curve unbracketed")
            d_values = d_from_curves(curves)
            replicate_results = {
                arm: heldout_analysis(d_values, arm) for arm in MOMENTUM_ARMS
            }
            for arm, result in replicate_results.items():
                surface = result["surface"]
                family_id = surface["family_id"]
                selection_counts[arm][family_id] += 1
                for label, value in zip(
                    surface["coefficient_order"], surface["coefficients"]
                ):
                    coefficient_samples[arm][family_id][label].append(value)
                success_counts[arm].append(result["success_count"])
                for cell in sorted(HOLDOUTS):
                    key = f"T{cell[0]}_S{cell[1]}"
                    error_samples[arm][cell].append(
                        result["heldout_predictions"][key][
                            "signed_error_bits_pred_minus_obs"
                        ]
                    )
        except (AnalysisError, KeyError, ValueError, OverflowError):
            invalid += 1

    valid = BOOTSTRAP_REPLICATES - invalid
    status = (
        "VALID" if valid >= MIN_VALID_BOOTSTRAP_REPLICATES else "NOT_EVALUABLE"
    )
    by_arm = {}
    for arm in MOMENTUM_ARMS:
        conditional_coefficients = {}
        for family_id in SURFACE_FAMILY_ORDER:
            intervals = []
            for label, values in coefficient_samples[arm][family_id].items():
                intervals.append(
                    {
                        "name": label,
                        "ci_95": {
                            "low": quantile(values, 0.025) if values else None,
                            "high": quantile(values, 0.975) if values else None,
                        },
                    }
                )
            conditional_coefficients[family_id] = {
                "selected_replicates": selection_counts[arm][family_id],
                "coefficient_intervals_conditional_on_selection": intervals,
            }
        heldout = {}
        for cell, values in error_samples[arm].items():
            heldout[f"T{cell[0]}_S{cell[1]}"] = {
                "signed_prediction_error_bits_ci_95": {
                    "low": quantile(values, 0.025) if values else None,
                    "high": quantile(values, 0.975) if values else None,
                }
            }
        counts = success_counts[arm]
        by_arm[arm] = {
            "model_selection_counts": selection_counts[arm],
            "model_selection_fractions": {
                family_id: (
                    selection_counts[arm][family_id] / valid if valid else None
                )
                for family_id in SURFACE_FAMILY_ORDER
            },
            "conditional_coefficient_intervals": conditional_coefficients,
            "heldout_error_intervals": heldout,
            "fraction_replicates_meeting_3_of_4": (
                sum(value >= SUCCESS_MIN_HOLDOUTS for value in counts) / len(counts)
                if counts
                else None
            ),
        }
    return {
        "method": (
            "paired nonparametric training-seed bootstrap; one shared three-index "
            "resample refits all 36 eta curves, independently repeats the "
            "registered eight-cell F1/F2/F3 LOO selection for both momentum "
            "arms, refits each selected surface, and recomputes all held-out "
            "errors"
        ),
        "replicates": BOOTSTRAP_REPLICATES,
        "rng_seed": BOOTSTRAP_SEED,
        "minimum_valid_replicates": MIN_VALID_BOOTSTRAP_REPLICATES,
        "valid_replicates": valid,
        "invalid_refit_replicates": invalid,
        "status": status,
        "by_arm": by_arm,
    }


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
    if manifest.get("schema") != "yeto_outer_mup_v6_launch_manifest_v1":
        raise SystemExit("not a v6 factorial launch manifest")
    if manifest.get("stage") != "V6_FACTORIAL" or len(
        manifest.get("cells", [])
    ) != EXPECTED_CELLS:
        raise SystemExit("manifest is not the complete 540-cell V6_FACTORIAL stage")
    node_roots = parse_node_roots(args.node_root)
    losses, cell_records, evidence_errors = load_losses(manifest, node_roots)

    curves = {}
    curve_errors = []
    for t in T_GRID:
        for s in S_GRID:
            for arm in ARMS:
                try:
                    curves[(t, s, arm)] = curve_fit(losses, t, s, arm)
                except AnalysisError as exc:
                    curves[(t, s, arm)] = {
                        "t": t,
                        "s": s,
                        "h": s // t,
                        "arm": arm,
                        "status": "INVALID_INPUT",
                        "interior": False,
                        "eta_star": None,
                        "error": str(exc),
                    }
                    curve_errors.append(f"T{t}/S{s}/{arm}: {exc}")

    all_curves_interior = all(curve["interior"] for curve in curves.values())
    d_values = d_from_curves(curves)
    arm_results = {}
    if all_curves_interior:
        for arm in MOMENTUM_ARMS:
            try:
                arm_results[arm] = heldout_analysis(d_values, arm)
            except AnalysisError as exc:
                arm_results[arm] = {"pass": False, "error": str(exc)}
                curve_errors.append(f"{arm} surface: {exc}")
    else:
        arm_results = {
            arm: {"pass": False, "error": "all 36 eta optima must be interior"}
            for arm in MOMENTUM_ARMS
        }

    evidence_complete = not evidence_errors and len(losses) == EXPECTED_CELLS
    if evidence_complete and all_curves_interior and not curve_errors:
        bootstrap = joint_bootstrap(losses)
    else:
        bootstrap = {
            "status": "NOT_EVALUABLE",
            "error": "complete evidence and 36 interior point fits are required",
            "valid_replicates": 0,
        }
    evaluable = (
        evidence_complete
        and all_curves_interior
        and not curve_errors
        and bootstrap.get("status") == "VALID"
    )
    if not evaluable:
        verdict = "NOT_EVALUABLE"
    elif all(arm_results[arm]["pass"] for arm in MOMENTUM_ARMS):
        verdict = "PASS"
    else:
        verdict = "FAIL"

    def count_display(arm: str) -> str:
        count = arm_results.get(arm, {}).get("success_count")
        return "NA" if count is None else f"{count}/4"

    note_line = (
        f"G6 SURFACE VERDICT: {verdict} "
        f"raw={count_display('raw')} corrected={count_display('corrected')}"
    )
    readout = {
        "schema": "yeto_outer_mup_v6_g6_readout_v1",
        "created_at_utc": utc_now(),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "source_git_commit": manifest.get("source", {}).get("git_commit"),
        "expected_cells": EXPECTED_CELLS,
        "observed_completed_cells": len(cell_records),
        "evidence_errors": evidence_errors,
        "curve_errors": curve_errors,
        "cell_records": cell_records,
        "curve_fits": [curves[key] for key in sorted(curves)],
        "D_obs": [
            {"T": t, "S": s, "H": s // t, "arm": arm, "D": d_values[key]}
            for key in sorted(d_values)
            for t, s, arm in [key]
        ],
        "registered_holdouts": [list(cell) for cell in sorted(HOLDOUTS)],
        "surface_results": arm_results,
        "bootstrap": bootstrap,
        "gate": {
            "name": "G6",
            "verdict": verdict,
            "evaluable": evaluable,
            "conditions": {
                "complete_540_cell_evidence": evidence_complete,
                "all_36_eta_optima_interior": all_curves_interior,
                "joint_bootstrap_at_least_9500_valid": (
                    bootstrap.get("status") == "VALID"
                ),
                "raw_at_least_3_of_4_within_0.2_bits": arm_results.get(
                    "raw", {}
                ).get("pass", False),
                "corrected_at_least_3_of_4_within_0.2_bits": arm_results.get(
                    "corrected", {}
                ).get("pass", False),
            },
            "success_rule": (
                "PASS iff each momentum arm independently has absolute held-out "
                "log2-D prediction error <=0.2 on at least 3 of 4 cells"
            ),
        },
        "note_line": note_line,
    }
    write_json_atomic(args.output.resolve(), readout)
    print(note_line)
    return 0 if verdict in ("PASS", "FAIL") else 2


if __name__ == "__main__":
    raise SystemExit(main())
