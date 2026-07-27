#!/usr/bin/env python3
"""CPU-only, hourly-capable Lane A mechanism pre-fit for partial v6 data.

This is a development/shakedown pipeline, not the frozen G6 analyzer and not a
source of final campaign verdicts.  It performs four deliberately separated
steps:

1. reread validated ``COMPLETED`` endpoint losses from one or more read-only
   v6 result snapshots;
2. reproduce the registered quadratic-in-log2(eta) convention, provisionally
   using whatever registered rungs/seeds have landed;
3. run the amended arm-specific F1/F2/F3 training-only surface selection when
   the available training geometry supports every registered candidate; and
4. fit Lane A's four-quantity static power-law spectrum to available training
   D values, then predict (never fit) the four registered held-out cells.

Every output is labeled NON-FINAL.  The final 540-cell campaign analysis,
three-index bootstrap, G6 evaluability decision, and verdict remain the job of
the hash-frozen ``scripts/analyze_v6.py`` from the authorized source commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Enforce CPU-only execution before importing NumPy/SciPy.  Single-threaded
# BLAS also keeps an hourly diagnostic from competing with the live campaign.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from scipy.optimize import minimize, minimize_scalar

try:
    from v6_surface_selection import (
        HOLDOUT_CELLS,
        TRAINING_CELLS,
        SelectionError,
        available_training_cells,
        least_squares,
        predict_registered_holdouts,
        select_surface,
    )
except ImportError:  # pragma: no cover - supports ``python -m scripts...``
    from scripts.v6_surface_selection import (
        HOLDOUT_CELLS,
        TRAINING_CELLS,
        SelectionError,
        available_training_cells,
        least_squares,
        predict_registered_holdouts,
        select_surface,
    )


ARMS = ("mu0", "raw", "corrected")
MOMENTUM_ARMS = ("raw", "corrected")
MU_HIGH = 0.9
ETA_POINTS = 5
RHO_XI = 0.69
N_MODES = 24
SPECTRAL_BOUNDS = (
    (-14.0, -2.0),
    (-14.0, -2.0),
    (-3.0, 3.0),
    (-12.0, 5.0),
)
PUBLISHED_THETA = (
    -11.1105515085,
    -2.44090584885,
    -1.96595298003,
    -0.919819128707,
)
DEFAULT_SPECTRAL_STARTS = (
    PUBLISHED_THETA,
    (-8.0, -3.0, -2.5, -1.5),
    (-12.0, -4.0, -1.0, -0.5),
)
GREEN_LINE = "PIPELINE GREEN (partial-data shakedown ok)"


class MechanismFitError(RuntimeError):
    """Input validation or partial-fit plumbing failed."""


@dataclass(frozen=True)
class DObservation:
    """One provisional or complete v6 tuning-ratio observation."""

    point_id: str
    arm: str
    H: int
    T: int
    S: int
    D: float
    source_complete: bool


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compact_utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise MechanismFitError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MechanismFitError(f"{path}: expected a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise MechanismFitError(
                        f"{path}:{number}: expected a JSON object"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise MechanismFitError(f"cannot read {path}: {exc}") from exc
    return rows


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value)
    temporary.replace(path)


def parse_node_roots(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise MechanismFitError(
                f"--node-root must be NODE=PATH, got {value!r}"
            )
        node, raw_path = value.split("=", 1)
        if not node or not raw_path or node in result:
            raise MechanismFitError(
                f"invalid or duplicate --node-root {value!r}"
            )
        result[node] = Path(raw_path).resolve()
    return result


def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("schema") != "yeto_outer_mup_v6_launch_manifest_v1":
        raise MechanismFitError("not a v6 factorial launch manifest")
    if manifest.get("stage") != "V6_FACTORIAL":
        raise MechanismFitError("manifest stage is not V6_FACTORIAL")
    cells = manifest.get("cells")
    if not isinstance(cells, list) or not cells:
        raise MechanismFitError("manifest has no scientific cells")
    ids = [cell.get("cell_id") for cell in cells if isinstance(cell, dict)]
    if len(ids) != len(cells) or len(set(ids)) != len(ids):
        raise MechanismFitError("manifest cell IDs are missing or duplicated")
    seeds = sorted({int(cell["seed"]) for cell in cells})
    coordinates = {
        (int(cell["t"]), int(cell["s"])) for cell in cells
    }
    expected_coordinates = {
        (t, s)
        for t in (2, 5, 10, 20)
        for s in (2560, 5120, 10240)
    }
    if coordinates != expected_coordinates:
        raise MechanismFitError("manifest factorial coordinates changed")
    expected_cells = len(expected_coordinates) * len(ARMS) * ETA_POINTS * len(seeds)
    if len(cells) != expected_cells:
        raise MechanismFitError(
            f"manifest has {len(cells)} cells, expected {expected_cells} "
            f"from its {len(seeds)} registered seeds"
        )
    return {
        "expected_cells": expected_cells,
        "seeds": seeds,
        "seed_count": len(seeds),
        "coordinates": [list(cell) for cell in sorted(coordinates)],
    }


def _expected_command_hash(cell: dict[str, Any], attempt_number: int) -> str:
    if attempt_number == 1:
        return str(cell["command_hash"])
    retries = cell.get("registered_retry_commands", [])
    for retry in retries:
        if int(retry.get("attempt_number", -1)) == attempt_number:
            return str(retry["command_hash"])
    raise MechanismFitError(
        f"{cell.get('cell_id')}: attempt {attempt_number} is not registered"
    )


def _completed_attempt(
    cell: dict[str, Any], root: Path
) -> tuple[Path, dict[str, Any], int] | None:
    """Select a validated completed registered attempt, newest first.

    Immutable ``attempt-*-failed-*`` archives are intentionally ignored.  An
    in-progress or failed registered attempt is ordinary partial state, not a
    loader error.
    """

    for attempt_number in (2, 1):
        attempt = root / str(cell["cell_id"]) / f"attempt-{attempt_number}"
        evidence_path = attempt / "evidence.json"
        if not evidence_path.is_file():
            continue
        evidence = read_json(evidence_path)
        if evidence.get("status") == "COMPLETED":
            return attempt, evidence, attempt_number
    return None


def load_partial_losses(
    manifest: dict[str, Any], node_roots: dict[str, Path]
) -> tuple[
    dict[tuple[int, int, str, int, float], float],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Load only hash-validated COMPLETED results from read-only roots."""

    losses: dict[tuple[int, int, str, int, float], float] = {}
    records: list[dict[str, Any]] = []
    invalid: list[str] = []
    state_counts: Counter[str] = Counter()
    node_manifest_counts: Counter[str] = Counter()
    node_loaded_counts: Counter[str] = Counter()
    missing_root_nodes: Counter[str] = Counter()

    for cell in manifest["cells"]:
        cell_id = str(cell.get("cell_id", "<missing-cell-id>"))
        try:
            node = str(cell["assignment"]["node"])
            node_manifest_counts[node] += 1
            if node not in node_roots:
                missing_root_nodes[node] += 1
                state_counts["NODE_ROOT_UNAVAILABLE"] += 1
                continue
            root = node_roots[node]
            completed = _completed_attempt(cell, root)
            if completed is None:
                state_counts["NOT_COMPLETED"] += 1
                continue
            attempt, evidence, attempt_number = completed
            expected_hash = _expected_command_hash(cell, attempt_number)
            if evidence.get("cell_id") != cell_id:
                raise MechanismFitError("evidence cell_id mismatch")
            if evidence.get("command_hash") != expected_hash:
                raise MechanismFitError("evidence command hash mismatch")
            if int(evidence.get("seed", -1)) != int(cell["seed"]):
                raise MechanismFitError("evidence seed mismatch")
            results_path = attempt / "report" / "results.jsonl"
            observed = evidence.get("observed_artifacts", {}).get("results", {})
            if not results_path.is_file():
                raise MechanismFitError("completed results.jsonl is missing")
            results_sha = sha256_file(results_path)
            if observed.get("sha256") != results_sha:
                raise MechanismFitError(
                    "results hash does not match validated evidence"
                )
            rows = read_jsonl(results_path)
            if len(rows) != 1:
                raise MechanismFitError(
                    f"expected one result row, found {len(rows)}"
                )
            loss = rows[0].get("eval_loss")
            if (
                not isinstance(loss, (int, float))
                or isinstance(loss, bool)
                or not math.isfinite(float(loss))
            ):
                raise MechanismFitError("endpoint evaluation loss is not finite")
            key = (
                int(cell["t"]),
                int(cell["s"]),
                str(cell["arm"]),
                int(cell["seed"]),
                float(cell["eta"]),
            )
            if key in losses:
                raise MechanismFitError("duplicate scientific cell coordinate")
            losses[key] = float(loss)
            node_loaded_counts[node] += 1
            state_counts["COMPLETED_VALIDATED"] += 1
            evidence_path = attempt / "evidence.json"
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
                    "eta": float(cell["eta"]),
                    "seed": int(cell["seed"]),
                    "eval_loss": float(loss),
                    "evidence_path": str(evidence_path),
                    "evidence_sha256": sha256_file(evidence_path),
                    "results_path": str(results_path),
                    "results_sha256": results_sha,
                }
            )
        except (
            KeyError,
            OSError,
            TypeError,
            ValueError,
            MechanismFitError,
        ) as exc:
            state_counts["INVALID_COMPLETED_EVIDENCE"] += 1
            invalid.append(f"{cell_id}: {exc}")

    audit = {
        "loaded_validated_cells": len(records),
        "unique_loss_keys": len(losses),
        "state_counts": dict(sorted(state_counts.items())),
        "manifest_cells_by_node": dict(sorted(node_manifest_counts.items())),
        "loaded_cells_by_node": dict(sorted(node_loaded_counts.items())),
        "missing_node_root_cells": dict(sorted(missing_root_nodes.items())),
        "invalid_completed_evidence_count": len(invalid),
        "invalid_completed_evidence": invalid,
        "source_roots_read_only": {
            node: str(path) for node, path in sorted(node_roots.items())
        },
    }
    return losses, records, audit


def registered_curve_grid(
    manifest: dict[str, Any], t: int, s: int, arm: str
) -> tuple[list[float], list[int]]:
    etas = sorted(
        {
            float(cell["eta"])
            for cell in manifest["cells"]
            if (int(cell["t"]), int(cell["s"]), str(cell["arm"]))
            == (t, s, arm)
        }
    )
    seeds = sorted(
        {
            int(cell["seed"])
            for cell in manifest["cells"]
            if (int(cell["t"]), int(cell["s"]), str(cell["arm"]))
            == (t, s, arm)
        }
    )
    if len(etas) != ETA_POINTS:
        raise MechanismFitError(
            f"T{t}/S{s}/{arm}: manifest has {len(etas)} eta rungs"
        )
    return etas, seeds


def fit_partial_curve(
    manifest: dict[str, Any],
    losses: dict[tuple[int, int, str, int, float], float],
    t: int,
    s: int,
    arm: str,
) -> dict[str, Any]:
    """Fit a registered curve from all currently available rung means."""

    registered_etas, registered_seeds = registered_curve_grid(
        manifest, t, s, arm
    )
    available_etas: list[float] = []
    means: list[float] = []
    seed_counts: list[int] = []
    seeds_by_eta: dict[str, list[int]] = {}
    observation_count = 0
    for eta in registered_etas:
        observed = [
            (seed, losses[(t, s, arm, seed, eta)])
            for seed in registered_seeds
            if (t, s, arm, seed, eta) in losses
        ]
        seeds_by_eta[format(eta, ".17g")] = [seed for seed, _ in observed]
        if observed:
            available_etas.append(eta)
            means.append(sum(value for _, value in observed) / len(observed))
            seed_counts.append(len(observed))
            observation_count += len(observed)

    complete = (
        len(available_etas) == ETA_POINTS
        and all(count == len(registered_seeds) for count in seed_counts)
    )
    base: dict[str, Any] = {
        "T": t,
        "S": s,
        "H": s // t,
        "arm": arm,
        "non_final": not complete,
        "registered_etas": registered_etas,
        "registered_seeds": registered_seeds,
        "available_etas": available_etas,
        "available_seed_counts": seed_counts,
        "available_seeds_by_eta": seeds_by_eta,
        "available_observations": observation_count,
        "required_observations": ETA_POINTS * len(registered_seeds),
        "source_complete": complete,
        "seed_mean_losses": means,
        "a": None,
        "b": None,
        "c": None,
        "vertex_log2_eta": None,
        "eta_star": None,
        "interior": False,
    }
    if len(available_etas) < 3:
        base.update(
            {
                "status": "INSUFFICIENT_PARTIAL_RUNGS",
                "error": "at least three observed eta rungs are required",
            }
        )
        return base

    xs = [math.log2(eta) for eta in available_etas]
    features = [[x * x, x, 1.0] for x in xs]
    try:
        a, b, c = least_squares(features, means)
    except SelectionError as exc:
        base.update({"status": "SINGULAR_PARTIAL_CURVE", "error": str(exc)})
        return base
    vertex = -b / (2.0 * a) if a else math.nan
    full_xs = [math.log2(eta) for eta in registered_etas]
    interior = (
        a > 0
        and math.isfinite(vertex)
        and min(full_xs) + 1e-12 < vertex < max(full_xs) - 1e-12
    )
    if complete:
        status = "INTERIOR" if interior else "UNBRACKETED"
    else:
        status = (
            "PROVISIONAL_INTERIOR_NON_FINAL"
            if interior
            else "PROVISIONAL_UNBRACKETED_NON_FINAL"
        )
    base.update(
        {
            "status": status,
            "a": a,
            "b": b,
            "c": c,
            "vertex_log2_eta": vertex if math.isfinite(vertex) else None,
            "eta_star": 2.0**vertex if interior else None,
            "interior": interior,
            "fit_uses_all_five_registered_rungs": len(available_etas)
            == ETA_POINTS,
            "fit_uses_all_registered_seeds_per_rung": complete,
        }
    )
    return base


def calculate_partial_curves(
    manifest: dict[str, Any],
    losses: dict[tuple[int, int, str, int, float], float],
) -> dict[tuple[int, int, str], dict[str, Any]]:
    return {
        (t, s, arm): fit_partial_curve(manifest, losses, t, s, arm)
        for t in (2, 5, 10, 20)
        for s in (2560, 5120, 10240)
        for arm in ARMS
    }


def d_from_partial_curves(
    curves: dict[tuple[int, int, str], dict[str, Any]]
) -> tuple[dict[tuple[int, int, str], float], list[dict[str, Any]]]:
    values: dict[tuple[int, int, str], float] = {}
    rows: list[dict[str, Any]] = []
    for t in (2, 5, 10, 20):
        for s in (2560, 5120, 10240):
            baseline = curves[(t, s, "mu0")]
            for arm in MOMENTUM_ARMS:
                momentum = curves[(t, s, arm)]
                d_value = None
                if baseline["interior"] and momentum["interior"]:
                    d_value = (
                        float(momentum["eta_star"])
                        / float(baseline["eta_star"])
                        / (1.0 - MU_HIGH)
                    )
                    if math.isfinite(d_value) and d_value > 0.0:
                        values[(t, s, arm)] = d_value
                    else:
                        d_value = None
                rows.append(
                    {
                        "T": t,
                        "S": s,
                        "H": s // t,
                        "arm": arm,
                        "D": d_value,
                        "log2_D": (
                            math.log2(d_value) if d_value is not None else None
                        ),
                        "baseline_curve_status": baseline["status"],
                        "momentum_curve_status": momentum["status"],
                        "source_complete": bool(
                            baseline["source_complete"]
                            and momentum["source_complete"]
                        ),
                        "non_final": True,
                    }
                )
    return values, rows


def run_surface_selection(
    d_values: dict[tuple[int, int, str], float],
    d_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    source_complete = {
        (int(row["T"]), int(row["S"]), str(row["arm"])): bool(
            row["source_complete"]
        )
        for row in d_rows
    }
    results: dict[str, Any] = {}
    for arm in MOMENTUM_ARMS:
        cells = available_training_cells(d_values, arm)
        try:
            surface = select_surface(d_values, arm, cells=cells)
            inputs_complete = all(
                source_complete.get((t, s, arm), False) for t, s in cells
            )
            predictions = predict_registered_holdouts(
                surface, d_values, arm=arm
            )
            results[arm] = {
                "status": (
                    surface["status"]
                    if inputs_complete
                    else "PROVISIONAL_PARTIAL_D_NON_FINAL"
                ),
                "non_final": True,
                "all_selected_training_D_sources_complete": inputs_complete,
                "available_training_cells": [list(cell) for cell in cells],
                "surface": surface,
                **predictions,
            }
        except SelectionError as exc:
            results[arm] = {
                "status": "INSUFFICIENT_PARTIAL_GEOMETRY",
                "non_final": True,
                "available_training_cells": [list(cell) for cell in cells],
                "available_training_cell_count": len(cells),
                "error": str(exc),
                "heldout_predictions": {},
            }
    return results


def spectrum(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Lane A's power-law band: two edges, alpha, and sigma^2."""

    lo, hi = sorted((float(theta[0]), float(theta[1])))
    log_rates = np.linspace(lo, hi, N_MODES)
    rates = np.exp(log_rates)
    log_mass = (float(theta[2]) + 1.0) * log_rates
    log_mass -= np.max(log_mass)
    weights = np.exp(log_mass)
    weights /= np.sum(weights)
    return rates, weights, math.exp(float(theta[3]))


def propagate(
    etas: np.ndarray,
    H: int,
    T: int,
    mu: float,
    arm: str,
    theta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Propagate full Cov(e,b,xi) matrices for every eta/mode pair."""

    rates, weights, sigma2 = spectrum(theta)
    lam = -np.expm1(-H * rates)
    xi_var = sigma2 * (-np.expm1(-2.0 * H * rates))
    innovation = xi_var * (1.0 - RHO_XI**2)
    eta = np.asarray(etas, dtype=float)
    covariance = np.zeros((len(eta), len(rates), 3, 3), dtype=float)
    covariance[:, :, 0, 0] = 1.0
    covariance[:, :, 2, 2] = xi_var[None, :]
    delta_var = np.zeros((len(eta), len(rates)), dtype=float)

    for age in range(1, T + 1):
        correction = (
            1.0 / (1.0 - mu ** (age + 1))
            if arm == "corrected" and mu != 0.0
            else 1.0
        )
        step = eta[:, None] * correction
        transition = np.zeros_like(covariance)
        transition[:, :, 0, 0] = (
            1.0 - step * (1.0 + mu) * lam[None, :]
        )
        transition[:, :, 0, 1] = -step * mu**2
        transition[:, :, 0, 2] = -step * (1.0 + mu) * RHO_XI
        transition[:, :, 1, 0] = lam[None, :]
        transition[:, :, 1, 1] = mu
        transition[:, :, 1, 2] = RHO_XI
        transition[:, :, 2, 2] = RHO_XI

        impulse = np.zeros((len(eta), len(rates), 3), dtype=float)
        impulse[:, :, 0] = -step * (1.0 + mu)
        impulse[:, :, 1] = 1.0
        impulse[:, :, 2] = 1.0

        delta_vector = np.zeros((len(eta), len(rates), 3), dtype=float)
        delta_vector[:, :, 0] = lam[None, :]
        delta_vector[:, :, 2] = RHO_XI
        delta_var = np.einsum(
            "...i,...ij,...j->...", delta_vector, covariance, delta_vector
        ) + innovation[None, :]

        propagated = np.einsum(
            "...ij,...jk,...lk->...il", transition, covariance, transition
        )
        shock = (
            innovation[None, :, None, None]
            * impulse[:, :, :, None]
            * impulse[:, :, None, :]
        )
        covariance = propagated + shock

    return covariance, delta_var, rates, weights, lam


def terminal_losses(
    etas: np.ndarray,
    H: int,
    T: int,
    mu: float,
    arm: str,
    theta: np.ndarray,
) -> np.ndarray:
    covariance, _delta, rates, weights, _lam = propagate(
        etas, H, T, mu, arm, theta
    )
    return covariance[:, :, 0, 0] @ (weights * rates)


def spectral_eta_star(
    H: int, T: int, mu: float, arm: str, theta: np.ndarray
) -> float:
    rates, _weights, _sigma2 = spectrum(theta)
    max_lam = float(np.max(-np.expm1(-H * rates)))
    log_grid = np.linspace(
        math.log(1e-5 / max_lam), math.log(4.0 / max_lam), 42
    )
    grid = np.exp(log_grid)
    values = terminal_losses(grid, H, T, mu, arm, theta)
    if not np.all(np.isfinite(values)):
        raise MechanismFitError("spectral terminal loss became nonfinite")
    best = int(np.argmin(values))
    left = log_grid[max(0, best - 2)]
    right = log_grid[min(len(log_grid) - 1, best + 2)]
    result = minimize_scalar(
        lambda log_eta: float(
            terminal_losses(
                np.asarray([math.exp(log_eta)]), H, T, mu, arm, theta
            )[0]
        ),
        bounds=(left, right),
        method="bounded",
        options={"xatol": 2e-7},
    )
    if not result.success or not math.isfinite(float(result.x)):
        raise MechanismFitError("spectral eta minimizer failed")
    return math.exp(float(result.x))


def predict_spectral_d(
    observation: DObservation, theta: np.ndarray, cache: dict[Any, float]
) -> float:
    zero_key = (observation.H, observation.T, 0.0, "raw")
    arm_key = (observation.H, observation.T, MU_HIGH, observation.arm)
    if zero_key not in cache:
        cache[zero_key] = spectral_eta_star(*zero_key, theta)
    if arm_key not in cache:
        cache[arm_key] = spectral_eta_star(*arm_key, theta)
    return cache[arm_key] / cache[zero_key] / (1.0 - MU_HIGH)


def predict_spectral_dataset(
    theta: np.ndarray, observations: list[DObservation]
) -> list[float]:
    cache: dict[Any, float] = {}
    return [predict_spectral_d(item, theta, cache) for item in observations]


def spectral_objective(
    theta: np.ndarray, observations: list[DObservation]
) -> float:
    try:
        predictions = predict_spectral_dataset(theta, observations)
        residuals = [
            math.log(predicted / observed.D)
            for predicted, observed in zip(
                predictions, observations, strict=True
            )
        ]
        value = float(np.mean(np.square(residuals)))
    except (
        FloatingPointError,
        MechanismFitError,
        OverflowError,
        ValueError,
    ):
        return 1e12
    return value if math.isfinite(value) else 1e12


def d_observations_from_values(
    d_values: dict[tuple[int, int, str], float],
    d_rows: list[dict[str, Any]],
    *,
    include_holdouts: bool = False,
) -> list[DObservation]:
    complete_lookup = {
        (int(row["T"]), int(row["S"]), str(row["arm"])): bool(
            row["source_complete"]
        )
        for row in d_rows
    }
    allowed = set(TRAINING_CELLS)
    if include_holdouts:
        allowed.update(HOLDOUT_CELLS)
    observations: list[DObservation] = []
    for (t, s, arm), value in sorted(d_values.items()):
        if (t, s) not in allowed or arm not in MOMENTUM_ARMS:
            continue
        observations.append(
            DObservation(
                point_id=f"v6:{arm}:T{t}:S{s}:H{s // t}",
                arm=arm,
                H=s // t,
                T=t,
                S=s,
                D=value,
                source_complete=complete_lookup[(t, s, arm)],
            )
        )
    return observations


def _spectral_prediction_rows(
    theta: np.ndarray,
    observations: list[DObservation],
) -> list[dict[str, Any]]:
    predictions = predict_spectral_dataset(theta, observations)
    return [
        {
            "point_id": observed.point_id,
            "arm": observed.arm,
            "T": observed.T,
            "S": observed.S,
            "H": observed.H,
            "observed_D": observed.D,
            "predicted_D": predicted,
            "observed_log2_D": math.log2(observed.D),
            "predicted_log2_D": math.log2(predicted),
            "signed_error_bits_pred_minus_obs": math.log2(
                predicted / observed.D
            ),
            "source_complete": observed.source_complete,
        }
        for observed, predicted in zip(
            observations, predictions, strict=True
        )
    ]


def fit_spectral_density(
    training: list[DObservation],
    d_values: dict[tuple[int, int, str], float],
    d_rows: list[dict[str, Any]],
    *,
    maxiter: int,
    start_limit: int,
    warm_start: list[float] | None,
) -> dict[str, Any]:
    """Fit only registered training cells and predict the held-out block."""

    arm_counts = Counter(item.arm for item in training)
    # A tonight shakedown is allowed to be statistically underidentified: the
    # operator explicitly wants the numerical path exercised on whatever has
    # landed, with garbage point estimates expected.  Requiring at least one
    # training D from each arm still ensures that both raw and corrected state
    # transitions are exercised.  Parameter accounting and nominal degrees of
    # freedom below make an underidentified fit impossible to mistake for a
    # scientific result.
    if not all(arm_counts[arm] for arm in MOMENTUM_ARMS):
        return {
            "status": "INSUFFICIENT_PARTIAL_D_FOR_SPECTRAL_FIT",
            "non_final": True,
            "optimized_parameters": 4,
            "available_training_D": len(training),
            "available_training_D_by_arm": dict(sorted(arm_counts.items())),
            "minimum_requirements": (
                "at least one registered-training D value from each momentum arm"
            ),
            "fit_uses_heldout_outcomes": False,
            "heldout_predictions": [],
        }

    starts: list[tuple[float, ...]] = []
    if warm_start is not None and len(warm_start) == 4:
        starts.append(tuple(float(value) for value in warm_start))
    starts.extend(DEFAULT_SPECTRAL_STARTS)
    deduplicated: list[tuple[float, ...]] = []
    for start in starts:
        if start not in deduplicated:
            deduplicated.append(start)
    selected_starts = deduplicated[: max(1, start_limit)]

    runs: list[dict[str, Any]] = []
    best_theta: np.ndarray | None = None
    best_objective = math.inf
    for start_index, start in enumerate(selected_starts):
        result = minimize(
            lambda theta: spectral_objective(theta, training),
            np.asarray(start, dtype=float),
            method="Powell",
            bounds=SPECTRAL_BOUNDS,
            options={
                "maxiter": maxiter,
                "maxfev": maxiter * 25,
                "ftol": 1e-8,
                "xtol": 3e-4,
            },
        )
        objective = spectral_objective(result.x, training)
        run = {
            "start_index": start_index,
            "start_theta": list(start),
            "success": bool(result.success),
            "message": str(result.message),
            "iterations": int(result.nit),
            "function_evaluations": int(result.nfev),
            "objective_mean_squared_ln_error": objective,
            "theta": [float(value) for value in result.x],
        }
        runs.append(run)
        if objective < best_objective:
            best_objective = objective
            best_theta = np.asarray(result.x, dtype=float)

    if best_theta is None or not math.isfinite(best_objective):
        return {
            "status": "SPECTRAL_OPTIMIZATION_FAILED",
            "non_final": True,
            "optimized_parameters": 4,
            "fit_uses_heldout_outcomes": False,
            "optimizer_runs": runs,
            "heldout_predictions": [],
        }

    rates, weights, sigma2 = spectrum(best_theta)
    training_rows = _spectral_prediction_rows(best_theta, training)
    observed_holdouts = d_observations_from_values(
        d_values, d_rows, include_holdouts=True
    )
    holdout_observed_lookup = {
        (item.T, item.S, item.arm): item for item in observed_holdouts
        if (item.T, item.S) in set(HOLDOUT_CELLS)
    }
    heldout_targets = [
        DObservation(
            point_id=f"v6-heldout:{arm}:T{t}:S{s}:H{s // t}",
            arm=arm,
            H=s // t,
            T=t,
            S=s,
            D=(
                holdout_observed_lookup[(t, s, arm)].D
                if (t, s, arm) in holdout_observed_lookup
                else 1.0
            ),
            source_complete=(
                holdout_observed_lookup[(t, s, arm)].source_complete
                if (t, s, arm) in holdout_observed_lookup
                else False
            ),
        )
        for arm in MOMENTUM_ARMS
        for t, s in HOLDOUT_CELLS
    ]
    heldout_predictions = predict_spectral_dataset(best_theta, heldout_targets)
    holdout_rows: list[dict[str, Any]] = []
    for target, predicted in zip(
        heldout_targets, heldout_predictions, strict=True
    ):
        observed = holdout_observed_lookup.get((target.T, target.S, target.arm))
        holdout_rows.append(
            {
                "point_id": target.point_id,
                "arm": target.arm,
                "T": target.T,
                "S": target.S,
                "H": target.H,
                "predicted_D": predicted,
                "predicted_log2_D": math.log2(predicted),
                "observed_D": observed.D if observed is not None else None,
                "observed_log2_D": (
                    math.log2(observed.D) if observed is not None else None
                ),
                "signed_error_bits_pred_minus_obs": (
                    math.log2(predicted / observed.D)
                    if observed is not None
                    else None
                ),
                "observed_source_complete": (
                    observed.source_complete if observed is not None else None
                ),
                "used_for_fitting": False,
            }
        )

    rmse_ln = math.sqrt(best_objective)
    converged = any(
        run["success"]
        and math.isclose(
            float(run["objective_mean_squared_ln_error"]),
            best_objective,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        for run in runs
    )
    underidentified = len(training) <= 4
    if underidentified:
        status = (
            "FIT_NON_FINAL_UNDERIDENTIFIED"
            if converged
            else "FIT_NON_FINAL_UNDERIDENTIFIED_OPTIMIZER_WARNING"
        )
    else:
        status = (
            "FIT_NON_FINAL"
            if converged
            else "FIT_NON_FINAL_OPTIMIZER_WARNING"
        )
    return {
        "status": status,
        "non_final": True,
        "model": "Lane A static power-law spectral density",
        "fit_scope": "available registered training cells only",
        "fit_uses_heldout_outcomes": False,
        "optimized_parameters": 4,
        "nominal_degrees_of_freedom": len(training) - 4,
        "underidentified_from_available_D_count": underidentified,
        "identification_warning": (
            "NON-FINAL shakedown has no positive residual degrees of freedom; "
            "parameters are plumbing-only garbage numbers"
            if underidentified
            else None
        ),
        "parameter_accounting": {
            "optimized": [
                "log(kappa_edge_a)",
                "log(kappa_edge_b)",
                "alpha",
                "log(sigma^2)",
            ],
            "rho_xi": "fixed at 0.69 from telemetry",
            "postfit_absolute_eta_scale_c_A": "not fit; v6 D ratios do not identify it",
        },
        "theta": [float(value) for value in best_theta],
        "kappa_min": float(rates[0]),
        "kappa_max": float(rates[-1]),
        "alpha": float(best_theta[2]),
        "sigma2": sigma2,
        "rho_xi_fixed_from_telemetry": RHO_XI,
        "quadrature_modes": N_MODES,
        "spectrum_weight_minmax": [
            float(weights.min()),
            float(weights.max()),
        ],
        "training_D_count": len(training),
        "training_D_by_arm": dict(sorted(arm_counts.items())),
        "training_complete_D_count": sum(
            item.source_complete for item in training
        ),
        "training_log_rmse": rmse_ln,
        "training_bits_rmse": rmse_ln / math.log(2.0),
        "optimizer_runs": runs,
        "training_predictions": training_rows,
        "heldout_predictions": holdout_rows,
    }


def previous_theta(latest_path: Path) -> list[float] | None:
    if not latest_path.is_file():
        return None
    try:
        latest = read_json(latest_path)
        spectral = latest.get("spectral_density_fit", {})
        theta = spectral.get("theta")
        if isinstance(theta, list) and len(theta) == 4:
            return [float(value) for value in theta]
    except (MechanismFitError, TypeError, ValueError):
        pass
    return None


def self_hashes() -> dict[str, Any]:
    script = Path(__file__).resolve()
    selection = script.with_name("v6_surface_selection.py")
    return {
        "fit_script_path": str(script),
        "fit_script_sha256": sha256_file(script),
        "selection_module_path": str(selection),
        "selection_module_sha256": (
            sha256_file(selection) if selection.is_file() else None
        ),
    }


def build_summary(report: dict[str, Any], output_path: Path) -> str:
    inventory = report["partial_inventory"]
    curves = report["curve_summary"]
    surfaces = report["surface_selection"]
    spectral = report["spectral_density_fit"]

    def surface_label(arm: str) -> str:
        result = surfaces[arm]
        family = result.get("surface", {}).get("family_id")
        cells = len(result.get("available_training_cells", []))
        return f"{family or 'PENDING'}({cells}/8 train-D)"

    spectral_theta = spectral.get("theta")
    spectral_label = spectral["status"]
    if spectral_theta:
        spectral_label += (
            f" k=[{spectral['kappa_min']:.3g},{spectral['kappa_max']:.3g}]"
            f" alpha={spectral['alpha']:.3g} sigma2={spectral['sigma2']:.3g}"
        )
    by_node = ",".join(
        f"{node}:{count}"
        for node, count in sorted(
            inventory["loader_audit"]["loaded_cells_by_node"].items()
        )
    ) or "none"
    return (
        f"- {report['created_at_utc']} — NON-FINAL hourly partial fit: "
        f"validated={inventory['loaded_validated_cells']}/"
        f"{inventory['expected_manifest_cells']} [{by_node}]; "
        f"curves={curves['interior_or_provisional_interior']}/36 "
        f"(complete={curves['complete_curves']}); "
        f"D={report['D_summary']['available_D']}/24; "
        f"surface raw={surface_label('raw')} "
        f"corrected={surface_label('corrected')}; "
        f"spectral={spectral_label}; output={output_path}"
    )


def update_note(note_path: Path, summary: str, mark_green: bool) -> None:
    note_path.parent.mkdir(parents=True, exist_ok=True)
    existing = note_path.read_text() if note_path.is_file() else "# mechfit lane\n"
    if existing and not existing.endswith("\n"):
        existing += "\n"
    if mark_green and GREEN_LINE not in existing.splitlines():
        existing += GREEN_LINE + "\n"
    existing += summary + "\n"
    write_text_atomic(note_path, existing)


def run_once(args: argparse.Namespace) -> tuple[dict[str, Any], Path, str]:
    cycle_started = time.monotonic()
    manifest_path = args.manifest.resolve()
    manifest = read_json(manifest_path)
    manifest_audit = validate_manifest(manifest)
    node_roots = parse_node_roots(args.node_root)
    latest_path = args.output_root.resolve() / "latest.json"
    warm_start = previous_theta(latest_path)

    losses, cell_records, loader_audit = load_partial_losses(
        manifest, node_roots
    )
    curves = calculate_partial_curves(manifest, losses)
    d_values, d_rows = d_from_partial_curves(curves)
    surface_results = run_surface_selection(d_values, d_rows)
    training_observations = d_observations_from_values(d_values, d_rows)
    spectral_result = fit_spectral_density(
        training_observations,
        d_values,
        d_rows,
        maxiter=args.spectral_maxiter,
        start_limit=args.spectral_starts,
        warm_start=warm_start,
    )

    complete_curves = sum(
        bool(curve["source_complete"]) for curve in curves.values()
    )
    interior_curves = sum(
        bool(curve["interior"]) for curve in curves.values()
    )
    created = utc_now()
    report: dict[str, Any] = {
        "schema": "yeto_v6_mechanism_partial_fit_v1",
        "created_at_utc": created,
        "status": "NON-FINAL partial-data mechanism pre-fit",
        "non_final": True,
        "warning": (
            "Partial rung means, eta vertices, D values, selected surfaces, "
            "spectral parameters, and held-out scores may change as cells land. "
            "This artifact cannot issue G6 PASS/FAIL."
        ),
        "execution": {
            "hostname": socket.gethostname(),
            "working_directory": str(Path.cwd()),
            "python": sys.version,
            "cpu_only": True,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "elapsed_seconds": None,
        },
        "provenance": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "manifest_source": manifest.get("source"),
            "manifest_registration": manifest.get("registration"),
            "manifest_variant": manifest.get("manifest_variant"),
            **self_hashes(),
        },
        "partial_inventory": {
            "expected_manifest_cells": manifest_audit["expected_cells"],
            "registered_seeds": manifest_audit["seeds"],
            "loaded_validated_cells": len(cell_records),
            "completion_fraction": len(cell_records)
            / manifest_audit["expected_cells"],
            "loader_audit": loader_audit,
            "cell_records": cell_records,
        },
        "curve_convention": {
            "coordinate": "x=log2(eta)",
            "fit": "loss=a*x^2+b*x+c on available registered rung means",
            "complete_rule": (
                "all five exact rungs and every manifest-registered seed per rung"
            ),
            "partial_rule": (
                "at least three registered rungs; all partial vertices are "
                "explicitly provisional and non-final"
            ),
            "interior_rule": (
                "a>0 and full-grid min+1e-12 < -b/(2a) < full-grid max-1e-12"
            ),
            "D_definition": "(eta_star_arm/eta_star_mu0)/(1-0.9)",
        },
        "curve_summary": {
            "total_curves": len(curves),
            "complete_curves": complete_curves,
            "interior_or_provisional_interior": interior_curves,
            "status_counts": dict(
                sorted(Counter(curve["status"] for curve in curves.values()).items())
            ),
        },
        "curve_fits": [curves[key] for key in sorted(curves)],
        "D_summary": {
            "available_D": len(d_values),
            "complete_D": sum(row["source_complete"] for row in d_rows),
            "available_training_D": len(training_observations),
            "available_heldout_D": sum(
                1
                for (t, s, _arm) in d_values
                if (t, s) in set(HOLDOUT_CELLS)
            ),
        },
        "D_observations": d_rows,
        "surface_selection": surface_results,
        "spectral_density_fit": spectral_result,
        "plumbing_validation": {
            "two_parameter_conventions": {
                "endpoint_source": (
                    "fresh validated results.jsonl, not copied summary values"
                ),
                "quadratic_coordinate": "log2(eta)",
                "ratio_definition": "(eta_mu/eta0)/(1-mu)",
                "unbracketed_not_silently_used": True,
                "source_results_read_only": True,
                "sha256_registry_embedded_in_cell_records": True,
            },
            "amended_v6_surface_contract": {
                "arm_specific_selection": True,
                "training_cells": [list(cell) for cell in TRAINING_CELLS],
                "heldout_cells": [list(cell) for cell in HOLDOUT_CELLS],
                "candidate_order": ["F1", "F2", "F3"],
                "selection_uses_heldout_outcomes": False,
                "final_joint_bootstrap": (
                    "not run here; delegated to the hash-frozen authorized analyzer"
                ),
            },
            "lane_A_referee_corrections": {
                "optimized_spectral_quantities": 4,
                "sigma2_counted_as_optimized": True,
                "rho_xi_0.69_is_fixed_from_data_not_known_a_priori": True,
                "heldout_block_excluded_from_spectral_fit": True,
                "static_spectrum_remains_a_falsifiable_model_not_a_closed_claim": True,
            },
        },
    }
    report["execution"]["elapsed_seconds"] = time.monotonic() - cycle_started

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output_root = args.output_root.resolve()
    output_path = output_root / f"partial-fit-{stamp}.json"
    write_json_atomic(output_path, report)
    write_json_atomic(latest_path, report)
    summary = build_summary(report, output_path)
    if args.note is not None:
        update_note(args.note.resolve(), summary, args.mark_green)
    return report, output_path, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "/root/yeto-results-v6/_controller/launch-v6/launch-manifest-v6.json"
        ),
    )
    parser.add_argument(
        "--node-root",
        action="append",
        default=[],
        help=(
            "NODE=PATH to a live read-only root or minimal snapshot; repeat for "
            "each results-bearing node. Defaults to h200-n1=/root/yeto-results-v6."
        ),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("/root/mechfit/fits")
    )
    parser.add_argument(
        "--note",
        type=Path,
        default=Path("/root/mechfit/h200-mechfit-note.md"),
        help="append one clearly NON-FINAL summary per fit cycle",
    )
    parser.add_argument(
        "--mark-green",
        action="store_true",
        help=f"add the exact line {GREEN_LINE!r} after a successful cycle",
    )
    parser.add_argument(
        "--watch", action="store_true", help="refit forever at the fixed interval"
    )
    parser.add_argument(
        "--interval-seconds", type=float, default=3600.0
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="testing/operations bound for watch mode; omit for an ongoing watcher",
    )
    parser.add_argument(
        "--spectral-maxiter",
        type=int,
        default=55,
        help="Powell iteration ceiling for each spectral start",
    )
    parser.add_argument(
        "--spectral-starts",
        type=int,
        default=3,
        help="maximum bounded starts (an hourly watcher can use 1-2)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.node_root:
        args.node_root = ["h200-n1=/root/yeto-results-v6"]
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")
    if args.spectral_maxiter <= 0 or args.spectral_starts <= 0:
        parser.error("spectral iteration/start limits must be positive")
    if args.max_cycles is not None and args.max_cycles <= 0:
        parser.error("--max-cycles must be positive")

    cycle = 0
    while True:
        started = time.monotonic()
        try:
            _report, output_path, summary = run_once(args)
            print(summary, flush=True)
            print(f"NON-FINAL JSON: {output_path}", flush=True)
        except Exception as exc:
            # A watcher must survive a transient half-written/in-flight state,
            # but failures remain visible and do not write a green marker.
            print(
                f"{utc_now()} NON-FINAL FIT CYCLE ERROR: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if not args.watch:
                raise
        cycle += 1
        if not args.watch or (
            args.max_cycles is not None and cycle >= args.max_cycles
        ):
            return 0
        elapsed = time.monotonic() - started
        time.sleep(max(0.0, args.interval_seconds - elapsed))


if __name__ == "__main__":
    raise SystemExit(main())
