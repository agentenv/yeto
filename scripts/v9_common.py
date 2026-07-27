#!/usr/bin/env python3
"""Shared deterministic math and serialization for the sealed-scale v9 lane."""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


SURFACE_FAMILY_ORDER = ("F1", "F2", "F3")
SURFACE_COEFFICIENT_ORDERS = {
    "F1": ("gamma", "alpha_T", "beta_log2_S"),
    "F2": (
        "gamma",
        "alpha_T",
        "beta_log2_S",
        "delta_T_x_log2_S",
    ),
    "F3": (
        "gamma",
        "alpha_T",
        "beta_log2_S",
        "epsilon_T_squared",
    ),
}


class V9Error(RuntimeError):
    """Raised when a sealed v9 input or deterministic calculation is invalid."""


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
        raise V9Error(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise V9Error(f"{path}: expected one JSON object")
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
                    raise V9Error(f"{path}:{number}: expected one JSON object")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise V9Error(f"cannot read {path}: {exc}") from exc
    return rows


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise V9Error("quantile of an empty sequence")
    if not 0.0 <= probability <= 1.0:
        raise V9Error(f"invalid quantile probability {probability}")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def solve_linear(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    if not size or len(matrix) != size or any(len(row) != size for row in matrix):
        raise V9Error("linear system has inconsistent dimensions")
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-14:
            raise V9Error("singular normal equation")
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
        raise V9Error("least-squares inputs are empty or unmatched")
    width = len(features[0])
    if not width or any(len(row) != width for row in features):
        raise V9Error("least-squares feature width changed")
    matrix = [
        [sum(row[i] * row[j] for row in features) for j in range(width)]
        for i in range(width)
    ]
    vector = [
        sum(row[i] * outcome for row, outcome in zip(features, outcomes))
        for i in range(width)
    ]
    return solve_linear(matrix, vector)


def fit_quadratic(
    etas: Sequence[float],
    losses: Sequence[float],
    *,
    near_bracket_allowance_bits: float,
) -> dict:
    """Fit the registered pooled curve and classify strict/near bracketing.

    The unconstrained vertex is retained when it lies no farther than the
    prospectively registered allowance beyond either ladder endpoint.  This is
    the same kind of loss-blind near-bracket evaluability repair used by G4C;
    it never clips the reported optimum to an observed rung.
    """

    if len(etas) != len(losses) or len(etas) < 3:
        raise V9Error("quadratic fit requires at least three matched points")
    if near_bracket_allowance_bits < 0 or not math.isfinite(
        near_bracket_allowance_bits
    ):
        raise V9Error("near-bracket allowance must be finite and nonnegative")
    if any(not math.isfinite(value) or value <= 0 for value in etas):
        raise V9Error("eta values must be finite and positive")
    if any(not math.isfinite(value) for value in losses):
        raise V9Error("loss values must be finite")
    xs = [math.log2(float(eta)) for eta in etas]
    a, b, c = least_squares(
        [[x * x, x, 1.0] for x in xs], [float(value) for value in losses]
    )
    vertex = -b / (2.0 * a) if a else math.nan
    positive_curvature = a > 0.0 and math.isfinite(vertex)
    strict = positive_curvature and min(xs) + 1e-12 < vertex < max(xs) - 1e-12
    near = (
        positive_curvature
        and min(xs) - near_bracket_allowance_bits
        <= vertex
        <= max(xs) + near_bracket_allowance_bits
    )
    if strict:
        status = "INTERIOR"
    elif near:
        status = "NEAR_BRACKETED"
    else:
        status = "UNBRACKETED"
    return {
        "a": a,
        "b": b,
        "c": c,
        "vertex_log2_eta": vertex if math.isfinite(vertex) else None,
        "eta_star": 2.0**vertex if near else None,
        "strict_interior": strict,
        "near_bracketed": near and not strict,
        "accepted": near,
        "status": status,
    }


def surface_features(t: int, s: int, family_id: str) -> list[float]:
    if family_id not in SURFACE_FAMILY_ORDER:
        raise V9Error(f"unknown v6 surface family {family_id!r}")
    if t <= 0 or s <= 0:
        raise V9Error("surface coordinates T and S must be positive")
    u = (float(t) - 5.0) / 5.0
    v = math.log2(float(s) / 5120.0)
    if family_id == "F1":
        return [1.0, u, v]
    if family_id == "F2":
        return [1.0, u, v, u * v]
    return [1.0, u, v, u * u]


def validate_surface(surface: dict) -> None:
    family_id = surface.get("family_id")
    if family_id not in SURFACE_FAMILY_ORDER:
        raise V9Error(f"invalid selected family {family_id!r}")
    expected_order = list(SURFACE_COEFFICIENT_ORDERS[family_id])
    if surface.get("coefficient_order") != expected_order:
        raise V9Error(f"{family_id} coefficient order differs from v6")
    coefficients = surface.get("coefficients")
    if (
        not isinstance(coefficients, list)
        or len(coefficients) != len(expected_order)
        or any(
            not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in coefficients
        )
    ):
        raise V9Error(f"{family_id} coefficients are malformed")
    coordinates = surface.get("coordinates")
    if coordinates != {"u": "(T-5)/5", "v": "log2(S/5120)"}:
        raise V9Error("selected surface coordinates differ from frozen v6 coordinates")


def predict_log2_d(surface: dict, t: int, s: int) -> float:
    validate_surface(surface)
    return sum(
        feature * float(coefficient)
        for feature, coefficient in zip(
            surface_features(t, s, str(surface["family_id"])),
            surface["coefficients"],
        )
    )


def log_linear_interpolate(
    x0: float, y0: float, x1: float, y1: float, target_x: float
) -> tuple[float, float]:
    """Interpolate/extrapolate positive y as a power law in positive x.

    Returns ``(prediction, log-log slope)``.  No hidden clipping is applied.
    """

    values = (x0, y0, x1, y1, target_x)
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise V9Error("log-linear interpolation requires finite positive values")
    if math.isclose(x0, x1):
        raise V9Error("log-linear interpolation anchors have the same x")
    slope = math.log2(y1 / y0) / math.log2(x1 / x0)
    prediction = y0 * (target_x / x0) ** slope
    if not math.isfinite(prediction) or prediction <= 0:
        raise V9Error("log-linear interpolation produced an invalid prediction")
    return prediction, slope


def exact_offset_grid(center: float, offsets_bits: Iterable[float]) -> list[float]:
    if not math.isfinite(center) or center <= 0:
        raise V9Error("ladder center must be finite and positive")
    offsets = [float(value) for value in offsets_bits]
    if not offsets or offsets != sorted(offsets) or len(offsets) != len(set(offsets)):
        raise V9Error("ladder offsets must be nonempty, unique, and sorted")
    return [center * (2.0**offset) for offset in offsets]
