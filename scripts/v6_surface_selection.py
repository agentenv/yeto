#!/usr/bin/env python3
"""Registered v6 F1/F2/F3 response-surface selection.

The frozen G6 analyzer remains the authority for the final campaign verdict.
This module exposes the same training-only selection calculation for the
mechanism pre-fit lane and adds an explicitly non-final partial-data mode.

Held-out outcomes are deliberately absent from :func:`select_surface`.  They
can only be attached afterwards with :func:`predict_registered_holdouts`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


T_GRID = (2, 5, 10, 20)
S_GRID = (2560, 5120, 10240)
HOLDOUT_CELLS = ((2, 10240), (5, 5120), (10, 2560), (20, 5120))
HOLDOUTS = frozenset(HOLDOUT_CELLS)
TRAINING_CELLS = tuple(
    sorted((t, s) for t in T_GRID for s in S_GRID if (t, s) not in HOLDOUTS)
)
FAMILY_ORDER = ("F1", "F2", "F3")
TIE_TOLERANCE_BITS = 1e-12
SUCCESS_ABS_ERROR_BITS = 0.2

FAMILIES = {
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


class SelectionError(RuntimeError):
    """The registered response-surface calculation is not evaluable."""


def solve_linear(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve a small dense system using the frozen analyzer's elimination."""

    size = len(vector)
    if len(matrix) != size or any(len(row) != size for row in matrix):
        raise SelectionError("linear system has inconsistent dimensions")
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(size):
        pivot = max(
            range(column, size), key=lambda row: abs(augmented[row][column])
        )
        if abs(augmented[pivot][column]) < 1e-14:
            raise SelectionError("singular normal equation")
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


def least_squares(
    features: Sequence[Sequence[float]], outcomes: Sequence[float]
) -> list[float]:
    """Fit ordinary least squares through normal equations, as preregistered."""

    if not features or len(features) != len(outcomes):
        raise SelectionError("least-squares inputs are empty or unmatched")
    width = len(features[0])
    if width == 0 or any(len(row) != width for row in features):
        raise SelectionError("least-squares feature width changed")
    matrix = [
        [sum(row[i] * row[j] for row in features) for j in range(width)]
        for i in range(width)
    ]
    vector = [
        sum(row[i] * outcome for row, outcome in zip(features, outcomes))
        for i in range(width)
    ]
    return solve_linear(matrix, vector)


def surface_features(t: int, s: int, family_id: str) -> list[float]:
    """Return the amended-contract centered features for one factorial cell."""

    if family_id not in FAMILIES:
        raise SelectionError(f"unknown surface family {family_id!r}")
    u = (t - 5.0) / 5.0
    v = math.log2(s / 5120.0)
    if family_id == "F1":
        return [1.0, u, v]
    if family_id == "F2":
        return [1.0, u, v, u * v]
    return [1.0, u, v, u * u]


def candidate_prediction(
    family_id: str, coefficients: Sequence[float], t: int, s: int
) -> float:
    features = surface_features(t, s, family_id)
    if len(features) != len(coefficients):
        raise SelectionError(
            f"{family_id}: expected {len(features)} coefficients, "
            f"received {len(coefficients)}"
        )
    return sum(
        feature * coefficient
        for feature, coefficient in zip(features, coefficients)
    )


def _lookup_d(
    d_values: Mapping[Any, Any], t: int, s: int, arm: str
) -> float | None:
    value = d_values.get((t, s, arm))
    if value is None:
        value = d_values.get((t, s))
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result > 0.0 else None


def available_training_cells(
    d_values: Mapping[Any, Any], arm: str
) -> tuple[tuple[int, int], ...]:
    """Return only registered training cells having a finite positive D."""

    return tuple(
        cell
        for cell in TRAINING_CELLS
        if _lookup_d(d_values, cell[0], cell[1], arm) is not None
    )


def _fit_coefficients(
    outcomes: Mapping[tuple[int, int], float],
    family_id: str,
    cells: Sequence[tuple[int, int]],
) -> list[float]:
    return least_squares(
        [surface_features(t, s, family_id) for t, s in cells],
        [outcomes[(t, s)] for t, s in cells],
    )


def select_surface(
    d_values: Mapping[Any, Any],
    arm: str,
    *,
    cells: Sequence[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Run arm-specific training-only F1/F2/F3 LOO selection.

    With ``cells=None`` this requires all eight registered training cells and
    reproduces the amended v6 contract.  A caller may pass the available
    subset for an hourly shakedown.  Partial selection is returned only when
    *every* candidate and every leave-one-out fold is full rank; it is always
    labeled ``PROVISIONAL_PARTIAL_NON_FINAL``.
    """

    selected_cells = tuple(TRAINING_CELLS if cells is None else cells)
    if len(set(selected_cells)) != len(selected_cells):
        raise SelectionError("training cell list contains duplicates")
    if not set(selected_cells).issubset(set(TRAINING_CELLS)):
        raise SelectionError("surface selection received a held-out/unknown cell")
    if len(selected_cells) < 2:
        raise SelectionError("at least two training cells are required")

    outcomes: dict[tuple[int, int], float] = {}
    for t, s in selected_cells:
        value = _lookup_d(d_values, t, s, arm)
        if value is None:
            raise SelectionError(f"{arm}/T{t}/S{s}: D is unavailable")
        outcomes[(t, s)] = math.log2(value)

    scores: list[dict[str, Any]] = []
    for family_id in FAMILY_ORDER:
        loo_predictions: dict[str, dict[str, float]] = {}
        squared_errors: list[float] = []
        try:
            for left_out in selected_cells:
                fold_cells = tuple(cell for cell in selected_cells if cell != left_out)
                coefficients = _fit_coefficients(
                    outcomes, family_id, fold_cells
                )
                t, s = left_out
                predicted = candidate_prediction(family_id, coefficients, t, s)
                observed = outcomes[left_out]
                error = predicted - observed
                squared_errors.append(error * error)
                loo_predictions[f"T{t}_S{s}"] = {
                    "observed_log2_D": observed,
                    "predicted_log2_D": predicted,
                    "signed_error_bits_pred_minus_obs": error,
                }
            coefficients = _fit_coefficients(
                outcomes, family_id, selected_cells
            )
        except SelectionError as exc:
            scores.append(
                {
                    "family_id": family_id,
                    "family": FAMILIES[family_id]["formula"],
                    "evaluable": False,
                    "error": str(exc),
                    "loo_rmse_bits": None,
                    "loo_predictions": loo_predictions,
                }
            )
            continue
        scores.append(
            {
                "family_id": family_id,
                "family": FAMILIES[family_id]["formula"],
                "evaluable": True,
                "loo_rmse_bits": math.sqrt(
                    sum(squared_errors) / len(squared_errors)
                ),
                "loo_predictions": loo_predictions,
                "all_training_coefficients": coefficients,
            }
        )

    unavailable = [record for record in scores if not record["evaluable"]]
    if unavailable:
        detail = "; ".join(
            f"{record['family_id']}: {record['error']}" for record in unavailable
        )
        raise SelectionError(
            "all three registered candidates must be evaluable; " + detail
        )

    best_rmse = min(float(record["loo_rmse_bits"]) for record in scores)
    tied = {
        str(record["family_id"])
        for record in scores
        if abs(float(record["loo_rmse_bits"]) - best_rmse)
        <= TIE_TOLERANCE_BITS
    }
    selected_family = next(
        family_id for family_id in FAMILY_ORDER if family_id in tied
    )
    coefficients = _fit_coefficients(
        outcomes, selected_family, selected_cells
    )
    fitted: dict[str, dict[str, float]] = {}
    for t, s in selected_cells:
        observed = outcomes[(t, s)]
        predicted = candidate_prediction(selected_family, coefficients, t, s)
        fitted[f"T{t}_S{s}"] = {
            "observed_log2_D": observed,
            "fitted_log2_D": predicted,
            "residual_bits": observed - predicted,
        }

    complete = selected_cells == TRAINING_CELLS
    return {
        "status": (
            "REGISTERED_COMPLETE"
            if complete
            else "PROVISIONAL_PARTIAL_NON_FINAL"
        ),
        "non_final": not complete,
        "arm": arm,
        "family_id": selected_family,
        "family": FAMILIES[selected_family]["formula"],
        "coordinates": {"u": "(T-5)/5", "v": "log2(S/5120)"},
        "coefficient_order": list(
            FAMILIES[selected_family]["coefficient_order"]
        ),
        "coefficients": coefficients,
        "training_cells": [list(cell) for cell in selected_cells],
        "registered_training_cell_count": len(TRAINING_CELLS),
        "available_training_cell_count": len(selected_cells),
        "training_fit": fitted,
        "model_selection": {
            "criterion": (
                "minimum leave-one-training-cell-out RMSE in log2(D) bits"
            ),
            "selection_uses_heldout_outcomes": False,
            "family_priority": list(FAMILY_ORDER),
            "tie_tolerance_bits": TIE_TOLERANCE_BITS,
            "tie_break_rule": (
                "among candidates within 1e-12 bits of the minimum LOO RMSE, "
                "choose the first of F1,F2,F3"
            ),
            "candidate_scores": [
                {
                    key: value
                    for key, value in record.items()
                    if key != "all_training_coefficients"
                }
                for record in scores
            ],
            "selected_family": selected_family,
        },
    }


def predict_registered_holdouts(
    surface: Mapping[str, Any],
    d_values: Mapping[Any, Any] | None = None,
    *,
    arm: str | None = None,
) -> dict[str, Any]:
    """Predict all four registered holdouts after selection/refitting.

    Observed held-out values, if supplied, are used only to score predictions.
    They cannot affect the already-fitted ``surface`` object.
    """

    family_id = str(surface["family_id"])
    coefficients = [float(value) for value in surface["coefficients"]]
    resolved_arm = arm or str(surface.get("arm", ""))
    rows: dict[str, dict[str, Any]] = {}
    successes = 0
    observed_count = 0
    for t, s in HOLDOUT_CELLS:
        predicted_log = candidate_prediction(
            family_id, coefficients, t, s
        )
        observed = (
            _lookup_d(d_values, t, s, resolved_arm)
            if d_values is not None
            else None
        )
        row: dict[str, Any] = {
            "T": t,
            "S": s,
            "H": s // t,
            "D_predicted": 2.0**predicted_log,
            "log2_D_predicted": predicted_log,
            "D_observed": observed,
            "log2_D_observed": None,
            "signed_error_bits_pred_minus_obs": None,
            "absolute_error_bits": None,
            "within_registered_0.2_bit_band": None,
        }
        if observed is not None:
            observed_count += 1
            observed_log = math.log2(observed)
            error = predicted_log - observed_log
            success = abs(error) <= SUCCESS_ABS_ERROR_BITS
            successes += int(success)
            row.update(
                {
                    "log2_D_observed": observed_log,
                    "signed_error_bits_pred_minus_obs": error,
                    "absolute_error_bits": abs(error),
                    "within_registered_0.2_bit_band": success,
                }
            )
        rows[f"T{t}_S{s}"] = row
    return {
        "arm": resolved_arm,
        "heldout_predictions": rows,
        "observed_holdout_count": observed_count,
        "success_count_among_observed": successes,
        "registered_success_rule_evaluable": observed_count == len(HOLDOUT_CELLS),
        "registered_pass": (
            successes >= 3 if observed_count == len(HOLDOUT_CELLS) else None
        ),
    }


__all__ = [
    "FAMILIES",
    "FAMILY_ORDER",
    "HOLDOUTS",
    "HOLDOUT_CELLS",
    "S_GRID",
    "SelectionError",
    "T_GRID",
    "TRAINING_CELLS",
    "available_training_cells",
    "candidate_prediction",
    "least_squares",
    "predict_registered_holdouts",
    "select_surface",
    "surface_features",
]
