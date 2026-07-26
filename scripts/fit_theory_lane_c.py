#!/usr/bin/env python3
"""Reproduce the empirical response surfaces in docs/theory-lane-C.md.

This is a CPU-only audit.  It reads the immutable master D table produced on
h200-n1, adds the two new fixed-T disambiguation cells in each arm, factors out
the code-true raw-Nesterov transient, fits the frozen candidate surfaces, and
performs literal leave-one-observation-out validation.

The script deliberately has no training or GPU code.  Its only non-stdlib
dependency is NumPy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np


MASTER_D_SHA256 = "f1b132a5b4580a396da344a959f195c747d3b759d6db54243d303316eed77427"
DISAMBIG_NOTE_SHA256 = "4be85f66125472fca267f284912442d5bf4ce6e25dbaae9dbca8f381e4af4835"
HUBER_DELTA_BITS = 0.15
MU_REFERENCE = 0.9


@dataclass(frozen=True)
class Observation:
    point_id: str
    campaign: str
    arm: str
    H: int
    S: int
    T: int
    mu: float
    D: float
    ci_low: float
    ci_high: float
    status: str
    source: str

    @property
    def raw_transient(self) -> float:
        """Code-true final-step D under a constant pseudo-gradient."""
        return 1.0 / (1.0 - self.mu ** (self.T + 1))

    @property
    def response_bits(self) -> float:
        if self.arm == "raw":
            return math.log2(self.D / self.raw_transient)
        if self.arm == "corrected":
            return math.log2(self.D)
        raise ValueError(f"unsupported arm {self.arm!r}")


DISAMBIG_OBSERVATIONS = (
    Observation(
        "disambig-raw-H1024-T5",
        "disambig",
        "raw",
        1024,
        5120,
        5,
        0.9,
        2.5532,
        2.46784,
        2.7013,
        "INTERIOR",
        "/private/tmp/h200-disambig-note.md",
    ),
    Observation(
        "disambig-raw-H2048-T5",
        "disambig",
        "raw",
        2048,
        10240,
        5,
        0.9,
        2.37478,
        2.35935,
        2.39782,
        "INTERIOR",
        "/private/tmp/h200-disambig-note.md",
    ),
    Observation(
        "disambig-corrected-H1024-T5",
        "disambig",
        "corrected",
        1024,
        5120,
        5,
        0.9,
        0.809865,
        0.799752,
        0.832495,
        "INTERIOR",
        "/private/tmp/h200-disambig-note.md",
    ),
    Observation(
        "disambig-corrected-H2048-T5",
        "disambig",
        "corrected",
        2048,
        10240,
        5,
        0.9,
        0.747421,
        0.740221,
        0.752297,
        "INTERIOR",
        "/private/tmp/h200-disambig-note.md",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optional_float(value: str | None, fallback: float) -> float:
    if value is None or value == "":
        return fallback
    return float(value)


def load_master(path: Path) -> list[Observation]:
    actual_hash = sha256(path)
    if actual_hash != MASTER_D_SHA256:
        raise RuntimeError(
            f"master_D.csv SHA-256 mismatch: {actual_hash} != {MASTER_D_SHA256}"
        )
    observations: list[Observation] = []
    with path.open(newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), 2):
            if not row["D"]:
                # The two unbracketed v3 T=40 rows have no response to fit.
                continue
            arm = row["arm"]
            if arm not in {"raw", "corrected"}:
                continue
            d_value = float(row["D"])
            point_id = (
                f"{row['campaign']}-{arm}-H{row['H']}-T{row['T']}"
                f"-mu{float(row['mu']):g}-r{row_number}"
            )
            observations.append(
                Observation(
                    point_id=point_id,
                    campaign=row["campaign"],
                    arm=arm,
                    H=int(row["H"]),
                    S=int(row["S"]),
                    T=int(row["T"]),
                    mu=float(row["mu"]),
                    D=d_value,
                    ci_low=optional_float(row.get("D_ci_low"), d_value),
                    ci_high=optional_float(row.get("D_ci_high"), d_value),
                    status=row["status"],
                    source=row["source"],
                )
            )
    observations.extend(DISAMBIG_OBSERVATIONS)
    return observations


def coordinates(observation: Observation) -> tuple[float, float, float, float, float]:
    """Return u, q, h, s, and log-age coordinates.

    Legal experimental cells obey S = T*H.  The redundant s coordinate is
    retained only so an additive duration-curvature candidate can be audited.
    """
    if observation.S != observation.T * observation.H:
        raise ValueError(f"{observation.point_id}: expected S=T*H")
    u = observation.mu / MU_REFERENCE
    q = (observation.T - 5.0) / 10.0
    h = math.log2(observation.H / 512.0)
    s = math.log2(observation.S / 2560.0)
    log_age = math.log2(observation.T / 5.0)
    return u, q, h, s, log_age


def raw_features(name: str, observation: Observation) -> list[float]:
    u, q, h, s, log_age = coordinates(observation)
    bases: dict[str, list[float]] = {
        "constant": [1.0],
        "age_linear": [1.0, q],
        "age_log": [1.0, log_age],
        "age_scale": [1.0, q, h],
        "logage_scale": [1.0, log_age, h],
        "age_scale_interaction": [1.0, q, h, q * h],
        "age_scale_interaction_mu_bend": [1.0, q, h, q * h],
        "age_scale_duration_curvature": [1.0, q, h, s * s],
        "age_scale_separate_quadratics": [1.0, q, h, q * q, h * h],
    }
    features = [u * value for value in bases[name]]
    if name == "age_scale_interaction_mu_bend":
        features.append(u * (u - 1.0))
    return features


def corrected_features(name: str, observation: Observation) -> list[float]:
    u, q, h, s, log_age = coordinates(observation)
    bases: dict[str, list[float]] = {
        "constant": [1.0],
        "age_linear": [1.0, q],
        "age_log": [1.0, log_age],
        "scale_linear": [1.0, h],
        "age_scale": [1.0, q, h],
        "logage_scale": [1.0, log_age, h],
        "age_scale_duration_curvature": [1.0, q, h, s * s],
        "age_scale_separate_quadratics": [1.0, q, h, q * q, h * h],
    }
    return [u * value for value in bases[name]]


def ols_fit(design: np.ndarray, response: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(design, response, rcond=None)[0]


def huber_fit(
    design: np.ndarray,
    response: np.ndarray,
    delta: float = HUBER_DELTA_BITS,
    tolerance: float = 1e-13,
    max_iterations: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Minimize the fixed-delta Huber objective by IRLS.

    The threshold is in response bits.  At convergence the normal-equation
    weight is min(1, delta/abs(residual)); no observation is deleted.
    """
    coefficients = ols_fit(design, response)
    weights = np.ones(len(response), dtype=float)
    for _iteration in range(max_iterations):
        residual = response - design @ coefficients
        denominator = np.maximum(np.abs(residual), np.finfo(float).tiny)
        weights = np.minimum(1.0, delta / denominator)
        sqrt_weights = np.sqrt(weights)
        updated = ols_fit(
            design * sqrt_weights[:, None], response * sqrt_weights
        )
        if float(np.max(np.abs(updated - coefficients))) < tolerance:
            return updated, weights
        coefficients = updated
    raise RuntimeError("Huber IRLS failed to converge")


FitFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]


def raw_fit(design: np.ndarray, response: np.ndarray) -> np.ndarray:
    return huber_fit(design, response)[0]


def design_matrix(
    observations: Sequence[Observation],
    feature_function: Callable[[str, Observation], list[float]],
    model_name: str,
) -> np.ndarray:
    return np.asarray(
        [feature_function(model_name, observation) for observation in observations],
        dtype=float,
    )


def loo_predictions(
    design: np.ndarray, response: np.ndarray, fit_function: FitFunction
) -> np.ndarray:
    predictions = np.empty(len(response), dtype=float)
    for held_out in range(len(response)):
        keep = np.arange(len(response)) != held_out
        coefficients = fit_function(design[keep], response[keep])
        predictions[held_out] = float(design[held_out] @ coefficients)
    return predictions


def metrics(errors: np.ndarray) -> dict[str, float]:
    absolute = np.abs(errors)
    return {
        "rmse_bits": float(np.sqrt(np.mean(errors * errors))),
        "mae_bits": float(np.mean(absolute)),
        "median_ae_bits": float(np.median(absolute)),
        "max_ae_bits": float(np.max(absolute)),
        "mean_signed_error_bits": float(np.mean(errors)),
    }


def empirical_radius(absolute_errors: np.ndarray, coverage: float) -> dict[str, float | int]:
    """Finite-sample conservative LOO-residual radius.

    Uses k=ceil((n+1)*coverage), capped at n.  At n=19 the 95% radius is the
    largest held-out error; at n=6 the 80--95% radii all collapse to the max.
    """
    ordered = np.sort(np.asarray(absolute_errors, dtype=float))
    rank = min(len(ordered), math.ceil((len(ordered) + 1) * coverage))
    radius = float(ordered[rank - 1])
    return {"coverage": coverage, "rank": rank, "bits": radius, "factor": 2.0**radius}


def convex_hull(points: Iterable[tuple[float, float]]) -> list[list[float]]:
    """Return counter-clockwise vertices using Andrew's monotone chain."""
    unique = sorted(set(points))
    if len(unique) <= 1:
        return [[float(value) for value in point] for point in unique]

    def cross(
        origin: tuple[float, float],
        left: tuple[float, float],
        right: tuple[float, float],
    ) -> float:
        return (left[0] - origin[0]) * (right[1] - origin[1]) - (
            left[1] - origin[1]
        ) * (right[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return [
        [float(value) for value in point]
        for point in lower[:-1] + upper[:-1]
    ]


def observed_D_from_response(observation: Observation, response_bits: float) -> float:
    residual_factor = 2.0**response_bits
    if observation.arm == "raw":
        return observation.raw_transient * residual_factor
    return residual_factor


def audit_arm(
    observations: Sequence[Observation],
    candidates: Iterable[str],
    feature_function: Callable[[str, Observation], list[float]],
    fit_function: FitFunction,
    selected_model: str,
) -> dict[str, object]:
    response = np.asarray([item.response_bits for item in observations], dtype=float)
    candidate_results: list[dict[str, object]] = []
    selected_design: np.ndarray | None = None
    selected_predictions: np.ndarray | None = None
    for name in candidates:
        design = design_matrix(observations, feature_function, name)
        predictions = loo_predictions(design, response, fit_function)
        errors = predictions - response
        record: dict[str, object] = {
            "name": name,
            "parameters": int(design.shape[1]),
            "rank": int(np.linalg.matrix_rank(design)),
            **metrics(errors),
        }
        candidate_results.append(record)
        if name == selected_model:
            selected_design = design
            selected_predictions = predictions
    if selected_design is None or selected_predictions is None:
        raise ValueError(f"selected model {selected_model!r} was not audited")

    coefficients = fit_function(selected_design, response)
    selected_errors = selected_predictions - response
    full_predictions = selected_design @ coefficients
    pointwise = []
    for item, loo_bits, full_bits, error_bits in zip(
        observations,
        selected_predictions,
        full_predictions,
        selected_errors,
        strict=True,
    ):
        loo_d = observed_D_from_response(item, float(loo_bits))
        full_d = observed_D_from_response(item, float(full_bits))
        pointwise.append(
            {
                "point_id": item.point_id,
                "campaign": item.campaign,
                "arm": item.arm,
                "H": item.H,
                "S": item.S,
                "T": item.T,
                "mu": item.mu,
                "D_observed": item.D,
                "D_ci_low": item.ci_low,
                "D_ci_high": item.ci_high,
                "D_loo": loo_d,
                "D_full_fit": full_d,
                "loo_error_bits": float(error_bits),
                "loo_ratio": loo_d / item.D,
                "loo_percent_error": 100.0 * (loo_d / item.D - 1.0),
                "status": item.status,
            }
        )
    return {
        "n": len(observations),
        "selected_model": selected_model,
        "coefficients": coefficients.tolist(),
        "candidate_loo": candidate_results,
        "selected_loo_metrics": metrics(selected_errors),
        "prediction_radii": [
            empirical_radius(np.abs(selected_errors), coverage)
            for coverage in (0.5, 0.8, 0.9, 0.95)
        ],
        "pointwise_loo": pointwise,
        "observed_q_h_convex_hull": convex_hull(
            (coordinates(item)[1], coordinates(item)[2]) for item in observations
        ),
    }


def predict_raw(T: int, H: int, S: int, mu: float, coefficients: Sequence[float]) -> float:
    template = Observation("prediction", "prediction", "raw", H, S, T, mu, 1, 1, 1, "", "")
    bits = float(np.asarray(raw_features("age_scale_interaction", template)) @ coefficients)
    return template.raw_transient * 2.0**bits


def predict_corrected(
    T: int, H: int, S: int, mu: float, coefficients: Sequence[float]
) -> float:
    template = Observation(
        "prediction", "prediction", "corrected", H, S, T, mu, 1, 1, 1, "", ""
    )
    bits = float(np.asarray(corrected_features("age_scale", template)) @ coefficients)
    return 2.0**bits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--master-d",
        type=Path,
        default=Path("/root/two-param-analysis/data/master_D.csv"),
    )
    parser.add_argument(
        "--disambig-note",
        type=Path,
        help="optional provenance check; the four numbers are frozen above",
    )
    parser.add_argument("--output", type=Path, help="write JSON here instead of stdout")
    args = parser.parse_args()

    if args.disambig_note is not None:
        actual_hash = sha256(args.disambig_note)
        if actual_hash != DISAMBIG_NOTE_SHA256:
            raise RuntimeError(
                f"disambiguation note SHA-256 mismatch: {actual_hash} != {DISAMBIG_NOTE_SHA256}"
            )

    observations = load_master(args.master_d)
    raw = [item for item in observations if item.arm == "raw"]
    corrected = [item for item in observations if item.arm == "corrected"]
    raw_result = audit_arm(
        raw,
        (
            "constant",
            "age_linear",
            "age_log",
            "age_scale",
            "logage_scale",
            "age_scale_interaction",
            "age_scale_interaction_mu_bend",
            "age_scale_duration_curvature",
            "age_scale_separate_quadratics",
        ),
        raw_features,
        raw_fit,
        "age_scale_interaction",
    )
    corrected_result = audit_arm(
        corrected,
        (
            "constant",
            "age_linear",
            "age_log",
            "scale_linear",
            "age_scale",
            "logage_scale",
            "age_scale_duration_curvature",
            "age_scale_separate_quadratics",
        ),
        corrected_features,
        ols_fit,
        "age_scale",
    )

    raw_coefficients = raw_result["coefficients"]
    corrected_coefficients = corrected_result["coefficients"]
    raw_selected_design = design_matrix(raw, raw_features, "age_scale_interaction")
    raw_response = np.asarray([item.response_bits for item in raw], dtype=float)
    _raw_huber_coefficients, raw_weights = huber_fit(raw_selected_design, raw_response)
    raw_result["full_fit_huber_weights"] = [
        {"point_id": item.point_id, "weight": float(weight)}
        for item, weight in zip(raw, raw_weights, strict=True)
    ]
    raw_ols_predictions = loo_predictions(raw_selected_design, raw_response, ols_fit)
    raw_result["equal_weight_ols_sensitivity"] = {
        "coefficients": ols_fit(raw_selected_design, raw_response).tolist(),
        **metrics(raw_ols_predictions - raw_response),
    }
    raw_example_cells = []
    for T, H in ((2, 512), (5, 512), (10, 512), (20, 512), (40, 64), (160, 16), (5, 1024), (5, 2048)):
        S = T * H
        raw_example_cells.append(
            {
                "T": T,
                "H": H,
                "S": S,
                "mu": 0.9,
                "D_raw": predict_raw(T, H, S, 0.9, raw_coefficients),
            }
        )
    corrected_example_cells = []
    for T, H in ((2, 512), (5, 512), (10, 512), (20, 512), (5, 1024), (5, 2048)):
        S = T * H
        corrected_example_cells.append(
            {
                "T": T,
                "H": H,
                "S": S,
                "mu": 0.9,
                "D_corrected": predict_corrected(
                    T, H, S, 0.9, corrected_coefficients
                ),
            }
        )

    output = {
        "provenance": {
            "master_D": str(args.master_d),
            "master_D_sha256": MASTER_D_SHA256,
            "disambig_note_sha256": DISAMBIG_NOTE_SHA256,
            "finite_master_rows": len(observations) - len(DISAMBIG_OBSERVATIONS),
            "disambig_rows": len(DISAMBIG_OBSERVATIONS),
            "total_rows": len(observations),
            "cpu_only": os.environ.get("CUDA_VISIBLE_DEVICES") == "",
        },
        "coordinate_contract": {
            "legal_cells": "S = T*H",
            "u": "mu/0.9",
            "q": "(T-5)/10",
            "h": "log2(H/512) = log2(S/(512*T))",
            "raw_response": "log2(D_raw * (1-mu^(T+1)))",
            "corrected_response": "log2(D_corrected)",
        },
        "raw": {"estimator": f"Huber(delta={HUBER_DELTA_BITS} bits)", **raw_result},
        "corrected": {"estimator": "ordinary least squares", **corrected_result},
        "raw_example_predictions_mu_0.9": raw_example_cells,
        "corrected_example_predictions_mu_0.9": corrected_example_cells,
        "aa_mu_zero_check": {
            "raw": predict_raw(5, 512, 2560, 0.0, raw_coefficients),
            "corrected": predict_corrected(
                5, 512, 2560, 0.0, corrected_coefficients
            ),
        },
    }
    rendered = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
