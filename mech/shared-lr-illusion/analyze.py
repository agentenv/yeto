#!/usr/bin/env python3
"""Post-hoc shared-learning-rate audit over sealed outer-momentum grids.

This script is descriptive only.  It does not launch training, alter a gate, or
read any mutable campaign state.  It consumes the already-sealed two-parameter,
G4C, G6, and G8 analysis artifacts and compares standard raw Nesterov
``mu=0.9`` with ``mu=0``.

For every exact-mu pair with complete eta-by-seed sweeps it reports

* shared-eta gain at the fitted no-momentum optimum;
* shared-eta gain at eta=0.7, but only when both observed grids contain 0.7;
* per-arm-tuned gain at the two fitted optima; and
* whether a statistically positive shared-eta result becomes null/harmful
  after per-arm retuning.

Gain is ``loss(mu=0) - loss(mu=.9)``; positive values favor momentum.  The
quadratic convention, seed pairing, replicate count, RNG seed, and percentile
interval convention follow the corresponding frozen analyzers.  No evaluation
example bootstrap is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


MU0 = 0.0
MU9 = 0.9
DEFAULT_ETA = 0.7
BOOTSTRAP_REPLICATES = 10_000
TWO_PARAM_BOOTSTRAP_SEED = 20260724
G4C_BOOTSTRAP_SEED = 20260726
G6_BOOTSTRAP_SEED = 20260727
DEFAULT_MIN_VALID = 9_500
G4C_MIN_VALID = 7_900
G4C_NEAR_BRACKET_BITS = 0.5
INTERIOR_TOLERANCE = 1e-12

GAIN_LABELS = ("HELP", "NULL", "HURT", "NOT_EVALUABLE")


class AnalysisError(RuntimeError):
    """A sealed input or analysis invariant was violated."""


@dataclass(frozen=True)
class Curve:
    """One eta-by-training-seed endpoint-loss grid."""

    mu: float
    etas: tuple[float, ...]
    seeds: tuple[int, ...]
    losses: np.ndarray  # shape: seeds x etas


@dataclass(frozen=True)
class Comparison:
    """One exact mu=0 versus raw-Nesterov mu=.9 comparison."""

    campaign: str
    campaign_order: int
    scale: str
    t: int
    s: int
    h: int
    fit_policy: str
    bootstrap_seed: int
    minimum_valid_replicates: int
    source_label: str
    source_sha256: str
    mu0: Curve
    mu9: Curve


@dataclass(frozen=True)
class PointFit:
    coefficients: np.ndarray
    vertex: float | None
    eta_star: float | None
    tuned_loss: float | None
    convex: bool
    strict_interior: bool
    accepted: bool
    status: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnalysisError(f"{path}: expected a JSON object")
    return value


def load_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            return list(csv.DictReader(source))
    except OSError as exc:
        raise AnalysisError(f"cannot read {path}: {exc}") from exc


def load_single_jsonl(path: Path) -> dict[str, Any]:
    try:
        lines = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except OSError as exc:
        raise AnalysisError(f"cannot read {path}: {exc}") from exc
    if len(lines) != 1:
        raise AnalysisError(f"{path}: expected exactly one nonblank JSONL row")
    try:
        value = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"cannot parse {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnalysisError(f"{path}: expected a JSON object")
    return value


def finite_float(value: Any, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"{context}: expected a number") from exc
    if not math.isfinite(result):
        raise AnalysisError(f"{context}: expected a finite number")
    return result


def validate_recorded_hash_list(
    path: Path,
    *,
    two_param_root: Path,
    expected_subdirectory: str,
    expected_count: int,
) -> dict[str, Any]:
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    except OSError as exc:
        raise AnalysisError(f"cannot read {path}: {exc}") from exc
    if len(lines) != expected_count:
        raise AnalysisError(
            f"{path}: expected {expected_count} recorded hashes, got {len(lines)}"
        )
    recorded_root = Path("/root/two-param-analysis")
    for line_number, line in enumerate(lines, 1):
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise AnalysisError(f"{path}:{line_number}: malformed SHA-256 row")
        expected_hash, recorded_path = parts
        try:
            relative = Path(recorded_path).relative_to(recorded_root)
        except ValueError as exc:
            raise AnalysisError(
                f"{path}:{line_number}: path is outside the recorded analysis root"
            ) from exc
        if not relative.parts or relative.parts[0] != expected_subdirectory:
            raise AnalysisError(
                f"{path}:{line_number}: unexpected snapshot path {relative}"
            )
        local_path = two_param_root / relative
        if sha256_file(local_path) != expected_hash:
            raise AnalysisError(f"{path}:{line_number}: SHA-256 mismatch for {local_path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "verified_file_hashes": len(lines),
    }


def validate_raw_two_parameter_snapshots(two_param_root: Path) -> dict[str, Any]:
    """Trace the sealed pilot/v3 seed-level CSV rows to their raw snapshots."""

    pilot_csv_path = two_param_root / "data" / "pilot_cells_eval_loss.csv"
    pilot_rows = load_csv(pilot_csv_path)
    pilot_paths = sorted(
        (two_param_root / "raw-pilot").glob("*/*/attempt-1/bank-result.json")
    )
    pilot_by_cell: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in pilot_paths:
        cell_id = path.parents[1].name
        if cell_id in pilot_by_cell:
            raise AnalysisError(f"pilot raw snapshots duplicate cell {cell_id}")
        pilot_by_cell[cell_id] = (path, load_json(path))
    pilot_csv_ids = {row["cell_id"] for row in pilot_rows}
    if len(pilot_rows) != 72 or pilot_csv_ids != set(pilot_by_cell):
        raise AnalysisError(
            "pilot seed-level CSV does not bijectively match 72 raw bank results"
        )
    for row in pilot_rows:
        path, raw = pilot_by_cell[row["cell_id"]]
        config = raw.get("config")
        result = raw.get("result")
        if not isinstance(config, dict) or not isinstance(result, dict):
            raise AnalysisError(f"{path}: missing config or result object")
        if row["node"] != path.parents[2].name or row["grid"] != config.get("grid"):
            raise AnalysisError(f"{path}: pilot CSV identity fields differ")
        for field, key in (("H", "h"), ("S", "s"), ("T", "t"), ("seed", "seed")):
            if int(row[field]) != int(config[key]):
                raise AnalysisError(f"{path}: pilot CSV {field} differs")
        for field, value in (
            ("mu", config["mu"]),
            ("eta", config["eta"]),
            ("loss", result["eval_loss"]),
        ):
            if finite_float(row[field], field) != finite_float(value, field):
                raise AnalysisError(f"{path}: pilot CSV {field} differs")

    manifest_path = two_param_root / "data" / "v3-launch-manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("schema") != "yeto_outer_mup_day1_launch_manifest_v1":
        raise AnalysisError(f"{manifest_path}: unexpected v3 manifest schema")
    cells = manifest.get("cells")
    if not isinstance(cells, list) or len(cells) != 460:
        raise AnalysisError(f"{manifest_path}: expected 460 manifest cells")
    manifest_by_cell = {str(cell["cell_id"]): cell for cell in cells}
    if len(manifest_by_cell) != len(cells):
        raise AnalysisError(f"{manifest_path}: duplicate cell IDs")
    v3_csv_path = two_param_root / "data" / "v3_cells_eval_loss.csv"
    v3_rows = load_csv(v3_csv_path)
    v3_csv_ids = {row["cell_id"] for row in v3_rows}
    if len(v3_rows) != 460 or v3_csv_ids != set(manifest_by_cell):
        raise AnalysisError(
            "v3 seed-level CSV does not bijectively match 460 manifest cells"
        )
    for row in v3_rows:
        cell = manifest_by_cell[row["cell_id"]]
        assignment = cell.get("initial_assignment")
        if not isinstance(assignment, dict):
            raise AnalysisError(f"{row['cell_id']}: missing initial assignment")
        node = str(assignment["node"])
        root = two_param_root / "raw-v3" / node / row["cell_id"] / "attempt-1"
        evidence_path = root / "evidence.json"
        result_path = root / "report" / "results.jsonl"
        evidence = load_json(evidence_path)
        result = load_single_jsonl(result_path)
        observed = evidence.get("observed_artifacts")
        if not isinstance(observed, dict) or not isinstance(observed.get("results"), dict):
            raise AnalysisError(f"{evidence_path}: missing results evidence")
        if (
            evidence.get("status") != "COMPLETED"
            or evidence.get("cell_id") != row["cell_id"]
            or int(evidence["seed"]) != int(row["seed"])
            or observed["results"].get("sha256") != sha256_file(result_path)
        ):
            raise AnalysisError(f"{evidence_path}: completion evidence differs")
        if row["node"] != node or row["manifest_arm"] != str(cell["arm"]):
            raise AnalysisError(f"{row['cell_id']}: v3 CSV identity fields differ")
        for field, key in (("H", "h"), ("S", "s"), ("T", "t"), ("seed", "seed")):
            if int(row[field]) != int(cell[key]):
                raise AnalysisError(f"{row['cell_id']}: v3 CSV {field} differs")
        expected_arm = (
            f"snoo-{cell['sub_arm']}"
            if cell["arm"] == "S"
            else "corrected"
            if cell["outer_bias_correction"]
            else "raw"
        )
        expected_sub_arm = str(cell.get("sub_arm") or "")
        if row["arm"] != expected_arm or row["sub_arm"] != expected_sub_arm:
            raise AnalysisError(f"{row['cell_id']}: v3 derived arm label differs")
        for field, value in (
            ("mu", cell["mu"]),
            ("eta", cell["eta"]),
            ("loss", result["eval_loss"]),
        ):
            if finite_float(row[field], field) != finite_float(value, field):
                raise AnalysisError(f"{row['cell_id']}: v3 CSV {field} differs")

    pilot_hashes = validate_recorded_hash_list(
        two_param_root / "data" / "raw-pilot-sha256.txt",
        two_param_root=two_param_root,
        expected_subdirectory="raw-pilot",
        expected_count=144,
    )
    v3_hashes = validate_recorded_hash_list(
        two_param_root / "data" / "raw-v3-sha256.txt",
        two_param_root=two_param_root,
        expected_subdirectory="raw-v3",
        expected_count=1_413,
    )
    return {
        "status": "PASS",
        "pilot_csv_rows_matched_to_raw": len(pilot_rows),
        "v3_csv_rows_matched_to_manifest_evidence_and_raw": len(v3_rows),
        "recorded_raw_hashes_verified": (
            pilot_hashes["verified_file_hashes"]
            + v3_hashes["verified_file_hashes"]
        ),
        "pilot_hash_list": pilot_hashes,
        "v3_hash_list": v3_hashes,
        "v3_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "verified_cells": len(cells),
        },
    }


def curve_from_rows(
    rows: Iterable[dict[str, Any]],
    *,
    mu: float,
    eta_key: str = "eta",
    seed_key: str = "seed",
    loss_key: str = "loss",
) -> Curve:
    selected = list(rows)
    if not selected:
        raise AnalysisError(f"mu={mu}: empty curve")
    etas = tuple(sorted({finite_float(row[eta_key], eta_key) for row in selected}))
    seeds = tuple(sorted({int(row[seed_key]) for row in selected}))
    if len(etas) < 4:
        raise AnalysisError(f"mu={mu}: expected at least four etas, found {etas}")
    if len(seeds) < 2:
        raise AnalysisError(f"mu={mu}: expected at least two seeds, found {seeds}")
    by_key: dict[tuple[int, float], float] = {}
    for row in selected:
        key = (int(row[seed_key]), finite_float(row[eta_key], eta_key))
        if key in by_key:
            raise AnalysisError(f"mu={mu}: duplicate seed/eta cell {key}")
        by_key[key] = finite_float(row[loss_key], loss_key)
    expected = {(seed, eta) for seed in seeds for eta in etas}
    if set(by_key) != expected:
        missing = sorted(expected - set(by_key))
        extra = sorted(set(by_key) - expected)
        raise AnalysisError(
            f"mu={mu}: eta-by-seed grid is not complete; "
            f"missing={missing[:4]}, extra={extra[:4]}"
        )
    losses = np.asarray(
        [[by_key[(seed, eta)] for eta in etas] for seed in seeds], dtype=float
    )
    losses.setflags(write=False)
    return Curve(mu=mu, etas=etas, seeds=seeds, losses=losses)


def validate_pair(mu0: Curve, mu9: Curve, context: str) -> None:
    if mu0.mu != MU0 or mu9.mu != MU9:
        raise AnalysisError(f"{context}: arm mu values are not exactly 0 and .9")
    if mu0.seeds != mu9.seeds:
        raise AnalysisError(
            f"{context}: paired bootstrap requires identical training seeds; "
            f"mu0={mu0.seeds}, mu9={mu9.seeds}"
        )


def design_projector(etas: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    xs = np.log2(np.asarray(etas, dtype=float))
    design = np.column_stack((xs * xs, xs, np.ones_like(xs)))
    gram = design.T @ design
    try:
        projector = np.linalg.solve(gram, design.T)
    except np.linalg.LinAlgError as exc:
        raise AnalysisError(f"singular eta-grid quadratic: {etas}") from exc
    return xs, projector


def acceptance_mask(
    a: np.ndarray,
    vertex: np.ndarray,
    xs: np.ndarray,
    policy: str,
) -> np.ndarray:
    finite = np.isfinite(a) & np.isfinite(vertex)
    convex = finite & (a > 0.0)
    low = float(xs.min())
    high = float(xs.max())
    if policy == "strict":
        return convex & (vertex > low + INTERIOR_TOLERANCE) & (
            vertex < high - INTERIOR_TOLERANCE
        )
    if policy == "pilot_convex_extrapolation":
        return convex
    if policy == "g4c_near_bracket":
        return convex & (vertex > low - G4C_NEAR_BRACKET_BITS) & (
            vertex < high + G4C_NEAR_BRACKET_BITS
        )
    raise AnalysisError(f"unknown fit policy {policy!r}")


def fit_point(curve: Curve, policy: str) -> PointFit:
    xs, projector = design_projector(curve.etas)
    coefficients = projector @ curve.losses.mean(axis=0)
    a, b, c = (float(value) for value in coefficients)
    vertex = -b / (2.0 * a) if a else math.nan
    convex = a > 0.0 and math.isfinite(vertex)
    strict = bool(
        convex
        and float(xs.min()) + INTERIOR_TOLERANCE < vertex
        < float(xs.max()) - INTERIOR_TOLERANCE
    )
    accepted = bool(
        acceptance_mask(
            np.asarray([a]), np.asarray([vertex]), xs, policy
        )[0]
    )
    if strict:
        status = "INTERIOR"
    elif accepted and policy == "pilot_convex_extrapolation":
        status = "EXTRAPOLATED"
    elif accepted and policy == "g4c_near_bracket":
        status = "NEAR_BRACKETED"
    elif not convex:
        status = "NONCONVEX"
    elif vertex <= float(xs.min()):
        status = "UNBRACKETED_LOW"
    else:
        status = "UNBRACKETED_HIGH"
    tuned = a * vertex * vertex + b * vertex + c if accepted else None
    return PointFit(
        coefficients=coefficients,
        vertex=float(vertex) if math.isfinite(vertex) else None,
        eta_star=2.0**vertex if accepted else None,
        tuned_loss=float(tuned) if tuned is not None else None,
        convex=convex,
        strict_interior=strict,
        accepted=accepted,
        status=status,
    )


def evaluate(coefficients: np.ndarray, x: float) -> float:
    a, b, c = coefficients
    return float(a * x * x + b * x + c)


def make_python_draws(seed: int, sample_size: int) -> np.ndarray:
    rng = random.Random(seed)
    return np.asarray(
        [
            [rng.randrange(sample_size) for _ in range(sample_size)]
            for _ in range(BOOTSTRAP_REPLICATES)
        ],
        dtype=np.int16,
    )


def make_v3_draws(sample_size: int) -> np.ndarray:
    """Return the first draw matrix from the frozen two-parameter PCG64 stream."""

    rng = np.random.default_rng(TWO_PARAM_BOOTSTRAP_SEED)
    return rng.integers(
        0, sample_size, size=(BOOTSTRAP_REPLICATES, sample_size), dtype=np.int16
    )


def make_pilot_draws(sample_size: int) -> np.ndarray:
    """Return the second draw matrix from the frozen two-parameter PCG64 stream.

    That report generated the five-seed v3 draw matrix first and the two-seed
    pilot matrix second from one PCG64 stream.
    """

    rng = np.random.default_rng(TWO_PARAM_BOOTSTRAP_SEED)
    rng.integers(0, 5, size=(BOOTSTRAP_REPLICATES, 5))
    return rng.integers(
        0, sample_size, size=(BOOTSTRAP_REPLICATES, sample_size), dtype=np.int16
    )


def refit_coefficients(curve: Curve, draws: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs, projector = design_projector(curve.etas)
    sampled_means = curve.losses[draws].mean(axis=1)
    coefficients = sampled_means @ projector.T
    return xs, coefficients


def percentile_interval(values: np.ndarray) -> dict[str, float] | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return None
    low, high = np.quantile(finite, (0.025, 0.975))
    return {"low": float(low), "high": float(high)}


def classify_gain(
    point: float | None,
    ci: dict[str, float] | None,
    valid: int,
    minimum_valid: int,
) -> str:
    if point is None or ci is None or valid < minimum_valid:
        return "NOT_EVALUABLE"
    if ci["low"] > 0.0:
        return "HELP"
    if ci["high"] < 0.0:
        return "HURT"
    return "NULL"


def grid_position(eta: float | None, grid: tuple[float, ...]) -> dict[str, Any]:
    if eta is None:
        return {
            "position": "not_evaluable",
            "in_grid": False,
            "distance_bits": None,
            "boundary_ratio": None,
        }
    low, high = min(grid), max(grid)
    if eta < low:
        return {
            "position": "below",
            "in_grid": False,
            "distance_bits": math.log2(low / eta),
            "boundary_ratio": eta / low,
        }
    if eta > high:
        return {
            "position": "above",
            "in_grid": False,
            "distance_bits": math.log2(eta / high),
            "boundary_ratio": eta / high,
        }
    return {
        "position": "inside",
        "in_grid": True,
        "distance_bits": 0.0,
        "boundary_ratio": 1.0,
    }


def metric_record(
    point: float | None,
    samples: np.ndarray,
    valid_mask: np.ndarray,
    minimum_valid: int,
) -> dict[str, Any]:
    selected = samples[valid_mask]
    ci = percentile_interval(selected)
    valid = int(valid_mask.sum())
    return {
        "gain": point,
        "ci_95": ci,
        "valid_bootstrap_replicates": valid,
        "invalid_bootstrap_replicates": BOOTSTRAP_REPLICATES - valid,
        "minimum_valid_replicates": minimum_valid,
        "label": classify_gain(point, ci, valid, minimum_valid),
    }


def analyze_comparison(
    comparison: Comparison,
    draws: np.ndarray,
) -> dict[str, Any]:
    validate_pair(comparison.mu0, comparison.mu9, comparison.campaign)
    if draws.shape != (
        BOOTSTRAP_REPLICATES,
        len(comparison.mu0.seeds),
    ):
        raise AnalysisError(
            f"{comparison.campaign}: draw shape {draws.shape} does not match "
            f"{len(comparison.mu0.seeds)} seeds"
        )

    fit0 = fit_point(comparison.mu0, comparison.fit_policy)
    fit9 = fit_point(comparison.mu9, comparison.fit_policy)
    point_joint = fit0.accepted and fit9.accepted

    shared_position = grid_position(fit0.eta_star, comparison.mu9.etas)
    shared_point = None
    tuned_point = None
    if fit0.accepted and fit0.vertex is not None:
        loss0_shared = evaluate(fit0.coefficients, fit0.vertex)
        loss9_shared = evaluate(fit9.coefficients, fit0.vertex)
        shared_point = loss0_shared - loss9_shared
    if point_joint:
        tuned_point = float(fit0.tuned_loss) - float(fit9.tuned_loss)

    default_in_grid = bool(
        min(comparison.mu0.etas) <= DEFAULT_ETA <= max(comparison.mu0.etas)
        and min(comparison.mu9.etas) <= DEFAULT_ETA <= max(comparison.mu9.etas)
    )
    default_point = None
    if default_in_grid:
        default_x = math.log2(DEFAULT_ETA)
        default_point = evaluate(fit0.coefficients, default_x) - evaluate(
            fit9.coefficients, default_x
        )

    xs0, coefficients0 = refit_coefficients(comparison.mu0, draws)
    xs9, coefficients9 = refit_coefficients(comparison.mu9, draws)
    a0, b0 = coefficients0[:, 0], coefficients0[:, 1]
    a9, b9 = coefficients9[:, 0], coefficients9[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        vertex0 = -b0 / (2.0 * a0)
        vertex9 = -b9 / (2.0 * a9)
    accepted0 = acceptance_mask(a0, vertex0, xs0, comparison.fit_policy)
    accepted9 = acceptance_mask(a9, vertex9, xs9, comparison.fit_policy)
    joint = accepted0 & accepted9

    loss0_at0 = (
        coefficients0[:, 0] * vertex0 * vertex0
        + coefficients0[:, 1] * vertex0
        + coefficients0[:, 2]
    )
    loss9_at0 = (
        coefficients9[:, 0] * vertex0 * vertex0
        + coefficients9[:, 1] * vertex0
        + coefficients9[:, 2]
    )
    loss9_at9 = (
        coefficients9[:, 0] * vertex9 * vertex9
        + coefficients9[:, 1] * vertex9
        + coefficients9[:, 2]
    )
    shared_samples = loss0_at0 - loss9_at0
    tuned_samples = loss0_at0 - loss9_at9
    shared_valid = accepted0 & np.isfinite(shared_samples)
    tuned_valid = joint & np.isfinite(tuned_samples)

    shared = metric_record(
        shared_point,
        shared_samples,
        shared_valid,
        comparison.minimum_valid_replicates,
    )
    shared.update(
        {
            "eta": fit0.eta_star,
            "eta_source": "mu0 fitted optimum",
            "momentum_grid_position": shared_position,
            "quadratic_extrapolation_on_momentum_arm": not shared_position[
                "in_grid"
            ],
        }
    )
    tuned = metric_record(
        tuned_point,
        tuned_samples,
        tuned_valid,
        comparison.minimum_valid_replicates,
    )
    tuned.update({"eta_mu0": fit0.eta_star, "eta_mu9": fit9.eta_star})

    if default_in_grid:
        default_x = math.log2(DEFAULT_ETA)
        default_samples = (
            coefficients0[:, 0] * default_x * default_x
            + coefficients0[:, 1] * default_x
            + coefficients0[:, 2]
            - coefficients9[:, 0] * default_x * default_x
            - coefficients9[:, 1] * default_x
            - coefficients9[:, 2]
        )
        default_metric = metric_record(
            default_point,
            default_samples,
            np.isfinite(default_samples),
            comparison.minimum_valid_replicates,
        )
    else:
        default_metric = {
            "gain": None,
            "ci_95": None,
            "valid_bootstrap_replicates": 0,
            "invalid_bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "minimum_valid_replicates": comparison.minimum_valid_replicates,
            "label": "NOT_EVALUABLE",
        }
    default_metric.update(
        {
            "eta": DEFAULT_ETA,
            "in_both_observed_grids": default_in_grid,
            "no_extrapolation_rule": True,
        }
    )

    a1_flip = (
        None
        if "NOT_EVALUABLE" in (shared["label"], tuned["label"])
        else shared["label"] == "HELP" and tuned["label"] in ("NULL", "HURT")
    )
    a2_flip = (
        None
        if "NOT_EVALUABLE" in (default_metric["label"], tuned["label"])
        else default_metric["label"] == "HELP"
        and tuned["label"] in ("NULL", "HURT")
    )

    # Algebraic check: for convex point fits, tuning the momentum curve cannot
    # reduce the gain relative to evaluating that same curve at eta0*.
    dominance_gap = (
        tuned_point - shared_point
        if tuned_point is not None and shared_point is not None
        else None
    )
    if dominance_gap is not None and dominance_gap < -1e-10:
        raise AnalysisError(
            f"{comparison.campaign}/T{comparison.t}/S{comparison.s}: "
            "tuned gain is unexpectedly below eta0-shared gain"
        )

    return {
        "campaign": comparison.campaign,
        "campaign_order": comparison.campaign_order,
        "scale": comparison.scale,
        "T": comparison.t,
        "S": comparison.s,
        "H": comparison.h,
        "mu0_seed_count": len(comparison.mu0.seeds),
        "mu9_seed_count": len(comparison.mu9.seeds),
        "mu0_eta_count": len(comparison.mu0.etas),
        "mu9_eta_count": len(comparison.mu9.etas),
        "mu0_eta_grid": list(comparison.mu0.etas),
        "mu9_eta_grid": list(comparison.mu9.etas),
        "fit_policy": comparison.fit_policy,
        "point_fit_status": {"mu0": fit0.status, "mu9": fit9.status},
        "point_fit_strict_interior": {
            "mu0": fit0.strict_interior,
            "mu9": fit9.strict_interior,
        },
        "source": {
            "label": comparison.source_label,
            "sha256": comparison.source_sha256,
        },
        "bootstrap": {
            "method": "paired nonparametric training-seed curve bootstrap",
            "replicates": BOOTSTRAP_REPLICATES,
            "rng_seed": comparison.bootstrap_seed,
            "rng_engine": (
                "NumPy Generator PCG64"
                if comparison.campaign
                in ("135M pilot S-scan", "135M v3 T-scan")
                else "Python random.Random MT19937"
            ),
            "draw_stream_position": (
                "second matrix after the five-seed v3 matrix"
                if comparison.campaign == "135M pilot S-scan"
                else "first matrix"
            ),
            "pairing": "one common seed-index draw for both arms and every eta",
            "interval": "pointwise percentile 95%",
        },
        "shared_at_mu0_optimum": shared,
        "shared_at_default_0p7": default_metric,
        "per_arm_tuned": tuned,
        "flip": {"a1_mu0_optimum": a1_flip, "a2_default_0p7": a2_flip},
        "tuned_minus_a1_gain_dominance_gap": dominance_gap,
    }


def comparison_from_grouped_rows(
    *,
    campaign: str,
    campaign_order: int,
    scale: str,
    t: int,
    s: int,
    h: int,
    fit_policy: str,
    bootstrap_seed: int,
    minimum_valid: int,
    source_label: str,
    source_sha256: str,
    rows0: list[dict[str, Any]],
    rows9: list[dict[str, Any]],
    loss_key: str = "loss",
) -> Comparison:
    # Some v1/v2 momentum curves have prospectively added top-up seeds that do
    # not exist on the control arm.  The frozen paired estimand uses only the
    # shared primary training seeds; unmatched top-ups cannot be treated as
    # paired observations.
    paired_seeds = {int(row["seed"]) for row in rows0} & {
        int(row["seed"]) for row in rows9
    }
    if len(paired_seeds) < 2:
        raise AnalysisError(
            f"{campaign}/T{t}/S{s}: fewer than two paired training seeds"
        )
    paired0 = [row for row in rows0 if int(row["seed"]) in paired_seeds]
    paired9 = [row for row in rows9 if int(row["seed"]) in paired_seeds]
    mu0 = curve_from_rows(paired0, mu=MU0, loss_key=loss_key)
    mu9 = curve_from_rows(paired9, mu=MU9, loss_key=loss_key)
    validate_pair(mu0, mu9, f"{campaign}/T{t}/S{s}")
    return Comparison(
        campaign=campaign,
        campaign_order=campaign_order,
        scale=scale,
        t=t,
        s=s,
        h=h,
        fit_policy=fit_policy,
        bootstrap_seed=bootstrap_seed,
        minimum_valid_replicates=minimum_valid,
        source_label=source_label,
        source_sha256=source_sha256,
        mu0=mu0,
        mu9=mu9,
    )


def load_pilot(two_param_root: Path) -> list[Comparison]:
    path = two_param_root / "data" / "pilot_cells_eval_loss.csv"
    rows = load_csv(path)
    selected = [
        row
        for row in rows
        if row.get("grid") == "grid1_s_variation"
        and float(row["mu"]) in (MU0, MU9)
    ]
    groups: defaultdict[tuple[int, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        groups[(int(row["T"]), int(row["S"]), int(row["H"]))].append(row)
    result = []
    for order, ((t, s, h), group) in enumerate(sorted(groups.items()), 10):
        result.append(
            comparison_from_grouped_rows(
                campaign="135M pilot S-scan",
                campaign_order=order,
                scale="135M",
                t=t,
                s=s,
                h=h,
                fit_policy="pilot_convex_extrapolation",
                bootstrap_seed=TWO_PARAM_BOOTSTRAP_SEED,
                minimum_valid=DEFAULT_MIN_VALID,
                source_label="sealed pilot_cells_eval_loss.csv",
                source_sha256=sha256_file(path),
                rows0=[row for row in group if float(row["mu"]) == MU0],
                rows9=[row for row in group if float(row["mu"]) == MU9],
            )
        )
    if len(result) != 3:
        raise AnalysisError(f"pilot: expected three exact-mu S-scan pairs, got {len(result)}")
    return result


READOUT_CELL_RE = re.compile(
    r"^e1(?P<v2>v2)?-135m-h(?P<h>\d+)-m04-mu(?P<mu>0(?:p9)?)"
    r"-e(?P<eta_index>\d+)-s(?P<seed>\d+)$"
)


def load_v1_or_v2(path: Path, *, version: str, order_base: int) -> list[Comparison]:
    readout = load_json(path)
    expected_schema = {
        "v1": "yeto_outer_mup_e1_selection_v1",
        "v2": "yeto_outer_mup_e1v2_selection_v1",
    }
    schema = readout.get("schema")
    if schema != expected_schema[version]:
        raise AnalysisError(f"{path}: unexpected {version} schema {schema!r}")
    curves = readout.get("eta_curves", {})
    grouped: defaultdict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
    for record in readout.get("cell_evidence", []):
        match = READOUT_CELL_RE.match(str(record.get("cell_id", "")))
        if not match:
            continue
        is_v2 = bool(match.group("v2"))
        if is_v2 != (version == "v2"):
            continue
        mu = MU9 if match.group("mu") == "0p9" else MU0
        h = int(match.group("h"))
        curve_key = f"H{h}_mu{'0.9' if mu == MU9 else '0'}"
        curve = curves.get(curve_key)
        if not isinstance(curve, dict):
            raise AnalysisError(f"{path}: missing {curve_key}")
        etas = curve.get("etas")
        index = int(match.group("eta_index"))
        if not isinstance(etas, list) or index >= len(etas):
            raise AnalysisError(f"{path}: {curve_key} has no eta index {index}")
        grouped[(h, mu)].append(
            {
                "eta": float(etas[index]),
                "seed": int(match.group("seed")),
                "loss": finite_float(record.get("eval_loss"), "eval_loss"),
            }
        )
    result = []
    for offset, h in enumerate(sorted({key[0] for key in grouped})):
        rows0 = grouped[(h, MU0)]
        rows9 = grouped[(h, MU9)]
        if not rows0 or not rows9:
            raise AnalysisError(f"{path}: H={h} does not have both exact arms")
        s = 2560
        if s % h:
            raise AnalysisError(f"{path}: nonintegral T for H={h}, S={s}")
        result.append(
            comparison_from_grouped_rows(
                campaign=f"135M {version}",
                campaign_order=order_base + offset,
                scale="135M",
                t=s // h,
                s=s,
                h=h,
                fit_policy="strict",
                bootstrap_seed=TWO_PARAM_BOOTSTRAP_SEED,
                minimum_valid=DEFAULT_MIN_VALID,
                source_label=f"sealed {path.name}",
                source_sha256=sha256_file(path),
                rows0=rows0,
                rows9=rows9,
            )
        )
    if len(result) != 4:
        raise AnalysisError(f"{version}: expected four H-scan pairs, got {len(result)}")
    return result


def load_v3(two_param_root: Path) -> list[Comparison]:
    path = two_param_root / "data" / "v3_cells_eval_loss.csv"
    rows = load_csv(path)
    selected = [
        row
        for row in rows
        if row.get("arm") == "raw" and float(row["mu"]) in (MU0, MU9)
    ]
    groups: defaultdict[tuple[int, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        groups[(int(row["T"]), int(row["S"]), int(row["H"]))].append(row)
    result = []
    for offset, ((t, s, h), group) in enumerate(sorted(groups.items())):
        result.append(
            comparison_from_grouped_rows(
                campaign="135M v3 T-scan",
                campaign_order=40 + offset,
                scale="135M",
                t=t,
                s=s,
                h=h,
                fit_policy="strict",
                bootstrap_seed=TWO_PARAM_BOOTSTRAP_SEED,
                minimum_valid=DEFAULT_MIN_VALID,
                source_label="sealed v3_cells_eval_loss.csv",
                source_sha256=sha256_file(path),
                rows0=[row for row in group if float(row["mu"]) == MU0],
                rows9=[row for row in group if float(row["mu"]) == MU9],
            )
        )
    if len(result) != 5:
        raise AnalysisError(f"v3: expected five T-scan pairs, got {len(result)}")
    return result


def load_g4c(path: Path) -> list[Comparison]:
    readout = load_json(path)
    if readout.get("schema") != "yeto_outer_mup_v4c_g4c_readout_v2":
        raise AnalysisError(f"{path}: not the canonical amended G4C readout")
    records = readout.get("cell_records", [])
    groups: defaultdict[tuple[int, int, int, float], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        mu = float(record["mu"])
        if mu not in (MU0, MU9):
            continue
        t, s = int(record["t"]), int(record["s"])
        groups[(t, s, s // t, mu)].append(
            {
                "eta": record["eta"],
                "seed": record["seed"],
                "loss": record["eval_loss"],
            }
        )
    coordinates = sorted({key[:3] for key in groups})
    result = []
    for offset, (t, s, h) in enumerate(coordinates):
        result.append(
            comparison_from_grouped_rows(
                campaign="1.7B v4-family combined",
                campaign_order=70 + offset,
                scale="1.7B",
                t=t,
                s=s,
                h=h,
                fit_policy="g4c_near_bracket",
                bootstrap_seed=G4C_BOOTSTRAP_SEED,
                minimum_valid=G4C_MIN_VALID,
                source_label="canonical amended G4C combined-grid readout",
                source_sha256=sha256_file(path),
                rows0=groups[(t, s, h, MU0)],
                rows9=groups[(t, s, h, MU9)],
            )
        )
    if len(result) != 2:
        raise AnalysisError(f"G4C: expected two combined-grid pairs, got {len(result)}")
    return result


def load_g6(path: Path) -> list[Comparison]:
    readout = load_json(path)
    if readout.get("schema") != "yeto_outer_mup_v6_g6_readout_v1":
        raise AnalysisError(f"{path}: not the canonical G6 readout")
    records = readout.get("cell_records", [])
    groups: defaultdict[tuple[int, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        arm = str(record["arm"])
        if arm not in ("mu0", "raw"):
            continue
        t, s, h = int(record["t"]), int(record["s"]), int(record["h"])
        groups[(t, s, h, arm)].append(
            {
                "eta": record["eta"],
                "seed": record["seed"],
                "loss": record["eval_loss"],
            }
        )
    coordinates = sorted({key[:3] for key in groups})
    result = []
    for offset, (t, s, h) in enumerate(coordinates):
        result.append(
            comparison_from_grouped_rows(
                campaign="135M v6 factorial",
                campaign_order=50 + offset,
                scale="135M",
                t=t,
                s=s,
                h=h,
                fit_policy="strict",
                bootstrap_seed=G6_BOOTSTRAP_SEED,
                minimum_valid=DEFAULT_MIN_VALID,
                source_label="canonical G6 factorial readout",
                source_sha256=sha256_file(path),
                rows0=groups[(t, s, h, "mu0")],
                rows9=groups[(t, s, h, "raw")],
            )
        )
    if len(result) != 12:
        raise AnalysisError(f"G6: expected twelve factorial pairs, got {len(result)}")
    return result


def audit_g8(path: Path) -> dict[str, Any]:
    readout = load_json(path)
    if readout.get("schema") != "yeto_outer_mup_v8_phase_diagram_readout_v1":
        raise AnalysisError(f"{path}: not the canonical G8 readout")
    fits = readout.get("curve_fits", [])
    coordinates = sorted(
        {
            (int(fit["T"]), int(fit["S"]), int(fit["H"]))
            for fit in fits
            if fit.get("arm") == "mu0"
        }
    )
    exclusions = []
    for t, s, h in coordinates:
        raw_mus = sorted(
            {
                float(fit["mu"])
                for fit in fits
                if (int(fit["T"]), int(fit["S"]), int(fit["H"])) == (t, s, h)
                and fit.get("arm") == "raw"
            }
        )
        exclusions.append(
            {
                "campaign": "135M v8 phase diagram",
                "scale": "135M",
                "T": t,
                "S": s,
                "H": h,
                "observed_raw_momentum_mus": raw_mus,
                "reason": (
                    "excluded from the exact-mu table: v8 banked raw-Nesterov "
                    "sweeps are mu=.8 and .95, not mu=.9; no interpolation in mu"
                ),
            }
        )
    return {
        "source_sha256": sha256_file(path),
        "excluded_cells": exclusions,
        "exact_mu0p9_cells": 0,
    }


def optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return finite_float(value, "expected frozen fit value")


def validate_frozen_point_fits(
    comparisons: list[Comparison],
    *,
    two_param_root: Path,
    g3_path: Path,
    g4c_path: Path,
    g6_path: Path,
) -> dict[str, Any]:
    """Check every recomputed point fit against its sealed source readout."""

    master_eta = load_csv(two_param_root / "data" / "master_eta.csv")
    v1 = load_json(two_param_root / "data" / "g1-readout.json")
    v2 = load_json(two_param_root / "data" / "g1v2-readout.json")
    g3 = load_json(g3_path)
    g4c = load_json(g4c_path)
    g6 = load_json(g6_path)

    max_eta_abs = 0.0
    max_coefficient_abs = 0.0
    max_seed_mean_loss_abs = 0.0
    checked = 0
    checks: list[dict[str, Any]] = []

    for comparison in comparisons:
        for curve, mu_label in ((comparison.mu0, "mu0"), (comparison.mu9, "mu9")):
            observed = fit_point(curve, comparison.fit_policy)
            expected: dict[str, Any]
            expected_source: str
            if comparison.campaign == "135M pilot S-scan":
                matches = [
                    row
                    for row in master_eta
                    if row["campaign"] == "pilot"
                    and row["arm"] == "raw"
                    and int(row["T"]) == comparison.t
                    and int(row["S"]) == comparison.s
                    and abs(float(row["mu"]) - curve.mu) < 1e-12
                    and row["coordinate"].startswith("grid1_s_variation")
                ]
                if len(matches) != 1:
                    raise AnalysisError(
                        f"pilot/T{comparison.t}/S{comparison.s}/{mu_label}: "
                        f"expected one frozen master-eta row, got {len(matches)}"
                    )
                expected = {"eta_star": optional_float(matches[0]["eta_star"])}
                expected_source = "master_eta.csv"
            elif comparison.campaign in ("135M v1", "135M v2"):
                readout = v1 if comparison.campaign.endswith("v1") else v2
                key = f"H{comparison.h}_mu{'0.9' if curve.mu == MU9 else '0'}"
                expected = readout["eta_curves"][key]
                expected_source = key
            elif comparison.campaign == "135M v3 T-scan":
                key = f"T_S{comparison.s}_mu{'0.9' if curve.mu == MU9 else '0'}"
                expected = g3["eta_curves"][key]
                expected_source = key
            elif comparison.campaign == "135M v6 factorial":
                arm = "raw" if curve.mu == MU9 else "mu0"
                matches = [
                    fit
                    for fit in g6["curve_fits"]
                    if int(fit["t"]) == comparison.t
                    and int(fit["s"]) == comparison.s
                    and fit["arm"] == arm
                ]
                if len(matches) != 1:
                    raise AnalysisError(
                        f"G6/T{comparison.t}/S{comparison.s}/{arm}: "
                        f"expected one frozen fit, got {len(matches)}"
                    )
                expected = matches[0]
                expected_source = f"G6/{arm}"
            elif comparison.campaign == "1.7B v4-family combined":
                matches = [
                    fit
                    for fit in g4c["curve_fits"]
                    if int(fit["t"]) == comparison.t
                    and int(fit["s"]) == comparison.s
                    and abs(float(fit["mu"]) - curve.mu) < 1e-12
                ]
                if len(matches) != 1:
                    raise AnalysisError(
                        f"G4C/T{comparison.t}/S{comparison.s}/{mu_label}: "
                        f"expected one frozen fit, got {len(matches)}"
                    )
                expected = matches[0]
                expected_source = "G4C"
            else:  # pragma: no cover - all registered loaders are enumerated
                raise AnalysisError(
                    f"no frozen-fit validator for {comparison.campaign}"
                )

            expected_eta = optional_float(expected.get("eta_star"))
            if (observed.eta_star is None) != (expected_eta is None):
                raise AnalysisError(
                    f"{comparison.campaign}/T{comparison.t}/S{comparison.s}/"
                    f"{mu_label}: eta-star evaluability differs from {expected_source}"
                )
            eta_error = (
                abs(float(observed.eta_star) - expected_eta)
                if observed.eta_star is not None and expected_eta is not None
                else 0.0
            )
            max_eta_abs = max(max_eta_abs, eta_error)
            coefficient_error = 0.0
            if all(key in expected for key in ("a", "b", "c")):
                expected_coefficients = np.asarray(
                    [expected["a"], expected["b"], expected["c"]], dtype=float
                )
                coefficient_error = float(
                    np.max(np.abs(observed.coefficients - expected_coefficients))
                )
                max_coefficient_abs = max(
                    max_coefficient_abs, coefficient_error
                )
            seed_mean_error = 0.0
            if isinstance(expected.get("seed_mean_losses"), list):
                expected_means = np.asarray(expected["seed_mean_losses"], dtype=float)
                observed_means = curve.losses.mean(axis=0)
                if expected_means.shape != observed_means.shape:
                    raise AnalysisError(
                        f"{comparison.campaign}/T{comparison.t}/S{comparison.s}/"
                        f"{mu_label}: frozen eta-grid shape differs"
                    )
                seed_mean_error = float(
                    np.max(np.abs(observed_means - expected_means))
                )
                max_seed_mean_loss_abs = max(
                    max_seed_mean_loss_abs, seed_mean_error
                )
            checked += 1
            checks.append(
                {
                    "campaign": comparison.campaign,
                    "T": comparison.t,
                    "S": comparison.s,
                    "mu": curve.mu,
                    "frozen_source": expected_source,
                    "eta_star_absolute_error": eta_error,
                    "max_coefficient_absolute_error": coefficient_error,
                    "max_seed_mean_loss_absolute_error": seed_mean_error,
                }
            )

    if checked != 60:
        raise AnalysisError(f"expected 60 point-fit parity checks, got {checked}")
    if max_eta_abs > 1e-10:
        raise AnalysisError(f"frozen eta-star parity failed: {max_eta_abs}")
    if max_coefficient_abs > 1e-9:
        raise AnalysisError(
            f"frozen quadratic-coefficient parity failed: {max_coefficient_abs}"
        )
    if max_seed_mean_loss_abs > 1e-12:
        raise AnalysisError(
            f"frozen seed-mean-loss parity failed: {max_seed_mean_loss_abs}"
        )
    return {
        "checked_curve_fits": checked,
        "maximum_eta_star_absolute_error": max_eta_abs,
        "maximum_quadratic_coefficient_absolute_error": max_coefficient_abs,
        "maximum_seed_mean_loss_absolute_error": max_seed_mean_loss_abs,
        "status": "PASS",
        "checks": checks,
    }


def draws_for(comparison: Comparison, cache: dict[tuple[str, int], np.ndarray]) -> np.ndarray:
    key = (comparison.campaign, len(comparison.mu0.seeds))
    if key in cache:
        return cache[key]
    if comparison.campaign == "135M pilot S-scan":
        draws = make_pilot_draws(len(comparison.mu0.seeds))
    elif comparison.campaign == "135M v3 T-scan":
        draws = make_v3_draws(len(comparison.mu0.seeds))
    else:
        draws = make_python_draws(
            comparison.bootstrap_seed, len(comparison.mu0.seeds)
        )
    cache[key] = draws
    return draws


def fmt_number(value: float | None, digits: int = 4, signed: bool = False) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    spec = f"{'+' if signed else ''}.{digits}f"
    return format(float(value), spec)


def fmt_metric(metric: dict[str, Any], *, markdown: bool = True) -> str:
    if metric["gain"] is None or metric["ci_95"] is None:
        return "NA"
    ci = metric["ci_95"]
    label = metric["label"]
    text = (
        f"{fmt_number(metric['gain'], signed=True)} "
        f"[{fmt_number(ci['low'], signed=True)}, {fmt_number(ci['high'], signed=True)}] "
        f"{label}"
    )
    if markdown:
        return text.replace("-", "−")
    return text


def fmt_position(record: dict[str, Any]) -> str:
    position = record["shared_at_mu0_optimum"]["momentum_grid_position"]
    if position["position"] == "inside":
        return "inside"
    if position["position"] == "above":
        return f"above {position['boundary_ratio']:.2f}x"
    if position["position"] == "below":
        return f"below {position['boundary_ratio']:.2f}x"
    return "NA"


def fmt_flip(value: bool | None) -> str:
    if value is None:
        return "NA"
    return "Y" if value else "N"


def summary_counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize(shared_key: str, flip_key: str) -> dict[str, int]:
        shared_evaluable = [
            record
            for record in records
            if record[shared_key]["label"] != "NOT_EVALUABLE"
        ]
        retuning_comparable = [
            record
            for record in shared_evaluable
            if record["per_arm_tuned"]["label"] != "NOT_EVALUABLE"
        ]
        shared_helps = [
            record for record in shared_evaluable if record[shared_key]["label"] == "HELP"
        ]
        flips = [
            record for record in retuning_comparable if record["flip"][flip_key]
        ]
        survives = [
            record
            for record in shared_helps
            if record["per_arm_tuned"]["label"] == "HELP"
        ]
        return {
            "evaluable_comparisons": len(shared_evaluable),
            "retuning_comparable_comparisons": len(retuning_comparable),
            "shared_lr_helps": len(shared_helps),
            "flips": len(flips),
            "shared_lr_helps_surviving_retuning": len(survives),
        }

    a1 = summarize("shared_at_mu0_optimum", "a1_mu0_optimum")
    a2 = summarize("shared_at_default_0p7", "a2_default_0p7")
    tuned_evaluable = [
        record for record in records if record["per_arm_tuned"]["label"] != "NOT_EVALUABLE"
    ]
    tuned_helps = [record for record in tuned_evaluable if record["per_arm_tuned"]["label"] == "HELP"]
    a1_in_grid = [
        record
        for record in records
        if record["shared_at_mu0_optimum"]["label"] != "NOT_EVALUABLE"
        and record["shared_at_mu0_optimum"]["momentum_grid_position"]["in_grid"]
    ]
    return {
        "a1_mu0_optimum": a1,
        "a2_default_0p7": a2,
        "per_arm_tuned": {
            "evaluable_comparisons": len(tuned_evaluable),
            "helps": len(tuned_helps),
            "null": sum(record["per_arm_tuned"]["label"] == "NULL" for record in tuned_evaluable),
            "hurts": sum(record["per_arm_tuned"]["label"] == "HURT" for record in tuned_evaluable),
        },
        "a1_in_momentum_grid": len(a1_in_grid),
        "a1_extrapolated": a1["evaluable_comparisons"] - len(a1_in_grid),
        "full_sweep_rows": len(records),
    }


def render_markdown(
    records: list[dict[str, Any]],
    counts: dict[str, Any],
    g8_audit: dict[str, Any],
    provenance: dict[str, Any],
) -> str:
    a1 = counts["a1_mu0_optimum"]
    a2 = counts["a2_default_0p7"]
    tuned = counts["per_arm_tuned"]
    lines = [
        "# Shared-LR illusion audit",
        "",
        "> **Post-hoc descriptive analysis of already-sealed data.** No result in this "
        "table was preregistered as a test of the shared-LR illusion; no gate was run "
        "or changed, and no new training was performed.",
        "",
        "Gain is endpoint evaluation loss for `mu=0` minus endpoint evaluation loss "
        "for standard raw Nesterov `mu=.9`; positive values favor momentum. Brackets "
        "are pointwise paired-training-seed bootstrap 95% intervals (10,000 refits). "
        "`HELP`, `NULL`, and `HURT` mean that the interval is respectively above, "
        "contains, or is below zero; `NOT_EVALUABLE` means the frozen minimum-valid-"
        "refit threshold was not met. `a1` evaluates both fitted quadratics at the "
        "no-momentum fitted optimum; `a2` evaluates both at eta=.7 only when both "
        "observed ladders contain .7; `b` evaluates each arm at its own fitted optimum. "
        "The position in parentheses after `a1` is the shared eta relative to the "
        "momentum ladder; `above 1.30x`, for example, means 30% above its highest rung.",
        "",
        "## Result",
        "",
        (
            f"Under the requested no-momentum-optimum shared-LR rule, momentum appears "
            f"to help in **{a1['shared_lr_helps']} of {a1['evaluable_comparisons']}** "
            f"bootstrap-evaluable shared-rate comparisons. Among the "
            f"**{a1['retuning_comparable_comparisons']}** cells that also have a "
            f"per-arm-tuned estimate, **{a1['flips']}** are HELP-to-NULL/HURT flips "
            f"after retuning, and "
            f"**{a1['shared_lr_helps_surviving_retuning']}** shared-LR HELP calls "
            f"survive retuning. However, only **{counts['a1_in_momentum_grid']}** of "
            f"those a1 evaluations lie inside the momentum ladder "
            f"({counts['a1_extrapolated']} require quadratic extrapolation). The "
            f"DiLoCo-style eta=.7 rule has **{a2['evaluable_comparisons']}** eligible "
            f"comparisons because none of the exact-mu paired ladders straddles .7. "
            f"With honest per-arm retuning, momentum is HELP/NULL/HURT in "
            f"**{tuned['helps']}/{tuned['null']}/{tuned['hurts']}** of "
            f"{tuned['evaluable_comparisons']} evaluable comparisons."
        ),
        "",
        "## Full table",
        "",
        "| campaign | scale | T | S | H | a1: shared at eta0* gain [95% CI] | a2: shared at .7 | b: per-arm-tuned gain [95% CI] | c: flip a1 / a2 |",
        "|---|---:|---:|---:|---:|---|---|---|:---:|",
    ]
    for record in records:
        a1_text = fmt_metric(record["shared_at_mu0_optimum"])
        if a1_text != "NA":
            a1_text += f" ({fmt_position(record)})"
        a2_text = fmt_metric(record["shared_at_default_0p7"])
        tuned_text = fmt_metric(record["per_arm_tuned"])
        flips = " / ".join(
            (
                fmt_flip(record["flip"]["a1_mu0_optimum"]),
                fmt_flip(record["flip"]["a2_default_0p7"]),
            )
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    record["campaign"],
                    record["scale"],
                    str(record["T"]),
                    str(record["S"]),
                    str(record["H"]),
                    a1_text,
                    a2_text,
                    tuned_text,
                    flips,
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Eligibility and exclusions",
            "",
            f"The main table contains all {counts['full_sweep_rows']} banked exact-mu "
            "pairs found in the sealed two-parameter, factorial, and 1.7B v4-family "
            "artifacts, including rows whose fitted optimum is unbracketed. The v4, "
            "v4b, and v4c stages are nested; only the final five-seed G4C combined "
            "grid is counted, avoiding duplicate outcomes. The standard raw Nesterov "
            "arm is used; the separately bias-corrected arm is not silently pooled "
            "with it. `NA` in a flip column means that the shared or tuned comparison "
            "is unavailable; it is not counted as a no-flip result.",
            "",
            "The requested v8 audit is present but cannot enter an exact `mu=.9` table:",
            "",
            "| campaign | scale | T | S | H | banked raw momentum mu values | disposition |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in g8_audit["excluded_cells"]:
        mus = ", ".join(f"{mu:g}" for mu in row["observed_raw_momentum_mus"])
        lines.append(
            f"| {row['campaign']} | {row['scale']} | {row['T']} | {row['S']} | "
            f"{row['H']} | {mus} | {row['reason']} |"
        )

    lines.extend(
        [
            "",
            "## Honest caveats",
            "",
            "- **The a1 comparison is structurally conservative for momentum wherever "
            "both optima are accepted.** For convex quadratics, the per-arm-tuned "
            "momentum loss cannot exceed its loss at eta0*, so the tuned gain is "
            "algebraically at least the a1 gain. A "
            "HELP-to-NULL/HURT point-estimate reversal is impossible under this exact "
            "estimand; only an interval-label reversal could occur because refitting "
            "changes uncertainty.",
            "- **a1 is outside the sampled momentum curve wherever marked `above` or "
            "`below`.** Those numerical values are quadratic extrapolations and are "
            "descriptive stress tests, not in-grid evidence. The table reports them "
            "because the requested eta0* rule otherwise has no estimate, but the count "
            "of supported in-grid comparisons is stated separately.",
            "- **The .7 default is not extrapolated.** None of the exact mu=0/.9 paired "
            "grids contains eta=.7. The data therefore cannot support a DiLoCo-default "
            "illusion claim; reporting a fitted value at .7 would be a long-range "
            "extrapolation.",
            "- **Grid resolution is coarse.** Curves have four to six eta rungs and two "
            "to five training seeds. Quadratic minima and percentile intervals inherit "
            "that model and resolution; pilot extrapolations retain the already-disclosed "
            "pilot convention and are labeled as such.",
            "- **Repeated coordinates are not independent replications.** Pilot, v1, "
            "v2, v3, and v6 differ in seeds, shuffled inputs, grids, or campaign source. "
            "The row count is descriptive and is not a meta-analytic sample size. "
            "Pointwise intervals are not multiplicity-adjusted.",
            "- **This question was posed after outcomes were known.** The frozen "
            "bootstrap mechanics are reused, but the shared-LR estimand, inclusion "
            "summary, and flip count are post-hoc. No causal claim about optimizer age "
            "follows from this table alone.",
            "",
            "## Reproduction and provenance",
            "",
            "Run `analyze.py` with explicit paths to the sealed two-parameter root and "
            "the canonical G4C, G6, and G8 readouts. `results.json` contains every eta "
            "grid, fit status, bootstrap-valid count, source hash, and unrounded value; "
            "`table.csv` is the flat machine-readable table. The reproducer also "
            "fails closed unless all 72 pilot and 460 v3 CSV rows match their raw "
            "snapshots and all 1,557 hashes in the recorded raw-source lists verify.",
            "",
            "| source | SHA-256 |",
            "|---|---|",
        ]
    )
    for label, item in provenance.items():
        lines.append(f"| {label} | `{item['sha256']}` |")
    lines.append("")
    return "\n".join(lines)


def render_summary(counts: dict[str, Any]) -> str:
    a1 = counts["a1_mu0_optimum"]
    a2 = counts["a2_default_0p7"]
    tuned = counts["per_arm_tuned"]
    return (
        "This post-hoc audit does **not** recover the proposed shared-LR illusion: "
        f"momentum appears to help in **{a1['shared_lr_helps']} of "
        f"{a1['evaluable_comparisons']}** bootstrap-evaluable shared-rate comparisons "
        "under the "
        "requested no-momentum-optimum shared-LR practice. Among the "
        f"**{a1['retuning_comparable_comparisons']}** cells with an evaluable per-arm "
        f"result, there are **{a1['flips']} flips** and "
        f"**{a1['shared_lr_helps_surviving_retuning']} shared-LR HELP calls that "
        "survive per-arm retuning**. All "
        f"{counts['a1_extrapolated']} computable shared-eta evaluations put the "
        "no-momentum optimum outside the momentum ladder, so none is supported "
        "in-grid; the DiLoCo-style eta=.7 comparison has "
        f"**{a2['evaluable_comparisons']} eligible cells** because no exact "
        "mu=0/.9 ladder pair contains .7. Per-arm retuning itself yields "
        f"HELP/NULL/HURT in **{tuned['helps']}/{tuned['null']}/{tuned['hurts']}** "
        f"of {tuned['evaluable_comparisons']} evaluable cells (the sole HELP is the "
        "1.7B v4-family T=5 cell). "
        "Thus these sealed grids cannot substantiate the requested K/N illusion "
        "claim without unsupported eta extrapolation or interpolation in momentum.\n"
    )


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def latex_metric(metric: dict[str, Any]) -> str:
    if metric["gain"] is None or metric["ci_95"] is None:
        return r"--"
    ci = metric["ci_95"]
    label = "NOT EVAL." if metric["label"] == "NOT_EVALUABLE" else metric["label"]
    return (
        f"${fmt_number(metric['gain'], signed=True)}$ "
        f"$[{fmt_number(ci['low'], signed=True)},"
        f"{fmt_number(ci['high'], signed=True)}]$ "
        f"{latex_escape(label)}"
    )


def render_latex(records: list[dict[str, Any]], counts: dict[str, Any]) -> str:
    a1 = counts["a1_mu0_optimum"]
    a2 = counts["a2_default_0p7"]
    lines = [
        r"% Requires \usepackage{booktabs,longtable,threeparttablex,pdflscape}.",
        r"\begin{landscape}",
        r"\begingroup\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{ThreePartTable}",
        r"\begin{TableNotes}[flushleft]\footnotesize",
        r"\item Gain is $L_{\mu=0}-L_{\mu=.9}$; positive values favor standard raw Nesterov momentum. Brackets are pointwise paired-seed bootstrap 95\% intervals from 10,000 quadratic refits. HELP/NULL/HURT indicate an interval above/containing/below zero; NOT EVALUABLE means the frozen valid-refit threshold was missed. The parenthetical a1 position is relative to the observed momentum eta ladder. NA means a required fitted optimum or the in-grid $\eta=.7$ comparison is unavailable; an NA flip is not a no-flip result. This is a post-hoc descriptive analysis of sealed data.",
        r"\end{TableNotes}",
        r"\begin{longtable}{@{}p{3.2cm}lrrr p{5.3cm} p{2.0cm} p{5.0cm} c@{}}",
        r"\caption{Shared-learning-rate comparisons versus per-arm retuning.}\label{tab:shared-lr-illusion}\\",
        r"\toprule",
        r"Campaign & Scale & $T$ & $S$ & $H$ & Shared at $\eta_0^\star$ (a1) & Shared at $.7$ (a2) & Per-arm tuned (b) & Flip a1/a2 \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{9}{c}{\tablename\ \thetable\ -- continued}\\",
        r"\toprule",
        r"Campaign & Scale & $T$ & $S$ & $H$ & Shared at $\eta_0^\star$ (a1) & Shared at $.7$ (a2) & Per-arm tuned (b) & Flip a1/a2 \\",
        r"\midrule",
        r"\endhead",
        r"\midrule \multicolumn{9}{r}{Continued on next page}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\insertTableNotes",
        r"\endlastfoot",
    ]
    for record in records:
        a1_text = latex_metric(record["shared_at_mu0_optimum"])
        if a1_text != r"--":
            a1_text += f" ({latex_escape(fmt_position(record))})"
        a2_text = latex_metric(record["shared_at_default_0p7"])
        tuned_text = latex_metric(record["per_arm_tuned"])
        flip_text = "/".join(
            (
                fmt_flip(record["flip"]["a1_mu0_optimum"]),
                fmt_flip(record["flip"]["a2_default_0p7"]),
            )
        )
        lines.append(
            " & ".join(
                (
                    latex_escape(record["campaign"]),
                    latex_escape(record["scale"]),
                    str(record["T"]),
                    str(record["S"]),
                    str(record["H"]),
                    a1_text,
                    a2_text,
                    tuned_text,
                    flip_text,
                )
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\end{longtable}",
            r"\end{ThreePartTable}",
            r"\endgroup",
            r"\end{landscape}",
            "",
            (
                "% Summary: shared eta0* HELP in "
                f"{a1['shared_lr_helps']}/{a1['evaluable_comparisons']}; "
                f"flips {a1['flips']}/{a1['retuning_comparable_comparisons']}; "
                f"survives {a1['shared_lr_helps_surviving_retuning']}. "
                f"In-grid eta=.7 comparisons {a2['evaluable_comparisons']}."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def render_subsection(counts: dict[str, Any]) -> str:
    a1 = counts["a1_mu0_optimum"]
    a2 = counts["a2_default_0p7"]
    tuned = counts["per_arm_tuned"]
    return "\n".join(
        [
            r"\subsection{Shared learning rates do not identify a momentum benefit}",
            r"\label{sec:shared-lr-illusion}",
            "",
            (
                "Because optimizer age changes the map from the nominal outer learning "
                "rate to the accumulated update, comparing momentum and no momentum at "
                "one nominal rate can answer a different question from comparing their "
                "independently tuned optima. We therefore performed a post-hoc descriptive "
                "audit of every sealed standard-Nesterov $\\mu=.9$ grid with a complete "
                "$\\mu=0$ partner. For each campaign cell we refit the registered quadratic "
                "$L=a(\\log_2\\eta)^2+b\\log_2\\eta+c$ under the campaign's frozen paired-"
                "seed bootstrap (10,000 refits) and defined gain as "
                "$L_{\\mu=0}-L_{\\mu=.9}$, so positive values favor momentum."
            ),
            "",
            (
                "For accepted convex fits, however, this particular a1 construction "
                "is one-sided:"
            ),
            r"\[",
            (
                r"G_b-G_{\mathrm{a1}}="
                r"L_{.9}(\eta_0^\star)-L_{.9}(\eta_{.9}^\star)="
                r"a_{.9}\!\left(\log_2\eta_0^\star-"
                r"\log_2\eta_{.9}^\star\right)^2\ge 0."
            ),
            r"\]",
            (
                "Consequently, a HELP-to-NULL/HURT point-estimate reversal is "
                "impossible under a1; only an interval-label reversal can occur because "
                "the bootstrap refits change uncertainty."
            ),
            "",
            (
                f"At the no-momentum fitted optimum, the shared-rate comparison labeled "
                f"momentum HELP in {a1['shared_lr_helps']} of "
                f"{a1['evaluable_comparisons']} bootstrap-evaluable shared-rate cells. "
                f"Among "
                f"the {a1['retuning_comparable_comparisons']} cells with accepted "
                f"optima for both arms, {a1['flips']} changed from HELP to NULL or "
                f"HURT after independent retuning, while "
                f"{a1['shared_lr_helps_surviving_retuning']} retained a HELP interval. "
                f"This apparent precision must not be overread: "
                f"only {counts['a1_in_momentum_grid']} a1 evaluations lie inside the "
                f"momentum ladder and {counts['a1_extrapolated']} require quadratic "
                f"extrapolation. Moreover, none of the exact $\\mu=0/.9$ ladder pairs "
                f"contains the community-default $\\eta\\simeq0.7$, leaving "
                f"{a2['evaluable_comparisons']} supported default-rate comparisons. "
                f"After honest per-arm retuning the cells are HELP/NULL/HURT in "
                f"{tuned['helps']}/{tuned['null']}/{tuned['hurts']} cases, respectively "
                f"(Table~\\ref{{tab:shared-lr-illusion}})."
            ),
            "",
            (
                "This audit was conceived after the outcomes were available and is not a "
                "preregistered test of the age mechanism. The grids contain only four to "
                "six rates and two to five seeds; intervals are pointwise; repeated "
                "campaign coordinates are not independent; and the v8 phase-diagram "
                "bank cannot be substituted because it sampled $\\mu\\in\\{.8,.95\\}$ "
                "rather than $.9$. We therefore treat the table as a transparent "
                "measurement of what these sealed curves can and cannot say, not as a "
                "causal estimate of momentum's effect or as evidence at the DiLoCo "
                "default."
            ),
            "",
        ]
    )


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "campaign",
        "scale",
        "T",
        "S",
        "H",
        "fit_policy",
        "mu0_fit_status",
        "mu9_fit_status",
        "eta0_star",
        "eta9_star",
        "a1_momentum_grid_position",
        "a1_momentum_boundary_ratio",
        "a1_gain",
        "a1_ci_low",
        "a1_ci_high",
        "a1_label",
        "a1_valid_bootstraps",
        "a2_gain",
        "a2_ci_low",
        "a2_ci_high",
        "a2_label",
        "tuned_gain",
        "tuned_ci_low",
        "tuned_ci_high",
        "tuned_label",
        "tuned_valid_bootstraps",
        "flip_a1",
        "flip_a2",
        "source_sha256",
    ]
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            a1 = record["shared_at_mu0_optimum"]
            a2 = record["shared_at_default_0p7"]
            tuned = record["per_arm_tuned"]
            position = a1["momentum_grid_position"]
            writer.writerow(
                {
                    "campaign": record["campaign"],
                    "scale": record["scale"],
                    "T": record["T"],
                    "S": record["S"],
                    "H": record["H"],
                    "fit_policy": record["fit_policy"],
                    "mu0_fit_status": record["point_fit_status"]["mu0"],
                    "mu9_fit_status": record["point_fit_status"]["mu9"],
                    "eta0_star": tuned["eta_mu0"],
                    "eta9_star": tuned["eta_mu9"],
                    "a1_momentum_grid_position": position["position"],
                    "a1_momentum_boundary_ratio": position["boundary_ratio"],
                    "a1_gain": a1["gain"],
                    "a1_ci_low": a1["ci_95"]["low"] if a1["ci_95"] else None,
                    "a1_ci_high": a1["ci_95"]["high"] if a1["ci_95"] else None,
                    "a1_label": a1["label"],
                    "a1_valid_bootstraps": a1["valid_bootstrap_replicates"],
                    "a2_gain": a2["gain"],
                    "a2_ci_low": a2["ci_95"]["low"] if a2["ci_95"] else None,
                    "a2_ci_high": a2["ci_95"]["high"] if a2["ci_95"] else None,
                    "a2_label": a2["label"],
                    "tuned_gain": tuned["gain"],
                    "tuned_ci_low": tuned["ci_95"]["low"] if tuned["ci_95"] else None,
                    "tuned_ci_high": tuned["ci_95"]["high"] if tuned["ci_95"] else None,
                    "tuned_label": tuned["label"],
                    "tuned_valid_bootstraps": tuned["valid_bootstrap_replicates"],
                    "flip_a1": record["flip"]["a1_mu0_optimum"],
                    "flip_a2": record["flip"]["a2_default_0p7"],
                    "source_sha256": record["source"]["sha256"],
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--two-param-root",
        required=True,
        type=Path,
        help="sealed two-param-analysis directory containing data/*.csv and readouts",
    )
    parser.add_argument("--g3-readout", required=True, type=Path)
    parser.add_argument("--g4c-readout", required=True, type=Path)
    parser.add_argument("--g6-readout", required=True, type=Path)
    parser.add_argument("--g8-readout", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    two_param_root = args.two_param_root.resolve()
    g3_path = args.g3_readout.resolve()
    g4c_path = args.g4c_readout.resolve()
    g6_path = args.g6_readout.resolve()
    g8_path = args.g8_readout.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    comparisons = []
    comparisons.extend(load_pilot(two_param_root))
    comparisons.extend(
        load_v1_or_v2(
            two_param_root / "data" / "g1-readout.json",
            version="v1",
            order_base=20,
        )
    )
    comparisons.extend(
        load_v1_or_v2(
            two_param_root / "data" / "g1v2-readout.json",
            version="v2",
            order_base=30,
        )
    )
    comparisons.extend(load_v3(two_param_root))
    comparisons.extend(load_g6(g6_path))
    comparisons.extend(load_g4c(g4c_path))
    comparisons.sort(key=lambda item: (item.campaign_order, item.t, item.s))
    if len(comparisons) != 30:
        raise AnalysisError(f"expected 30 exact-mu full-sweep rows, got {len(comparisons)}")

    raw_snapshot_validation = validate_raw_two_parameter_snapshots(two_param_root)
    fit_validation = validate_frozen_point_fits(
        comparisons,
        two_param_root=two_param_root,
        g3_path=g3_path,
        g4c_path=g4c_path,
        g6_path=g6_path,
    )

    draw_cache: dict[tuple[str, int], np.ndarray] = {}
    records = [
        analyze_comparison(comparison, draws_for(comparison, draw_cache))
        for comparison in comparisons
    ]
    counts = summary_counts(records)
    g8_audit = audit_g8(g8_path)

    provenance = {
        "two-parameter pilot cells": {
            "path": str(two_param_root / "data" / "pilot_cells_eval_loss.csv"),
            "sha256": sha256_file(two_param_root / "data" / "pilot_cells_eval_loss.csv"),
        },
        "two-parameter v1 readout": {
            "path": str(two_param_root / "data" / "g1-readout.json"),
            "sha256": sha256_file(two_param_root / "data" / "g1-readout.json"),
        },
        "two-parameter v2 readout": {
            "path": str(two_param_root / "data" / "g1v2-readout.json"),
            "sha256": sha256_file(two_param_root / "data" / "g1v2-readout.json"),
        },
        "two-parameter v3 cells": {
            "path": str(two_param_root / "data" / "v3_cells_eval_loss.csv"),
            "sha256": sha256_file(two_param_root / "data" / "v3_cells_eval_loss.csv"),
        },
        "two-parameter v3 manifest": raw_snapshot_validation["v3_manifest"],
        "two-parameter pilot raw hash list": raw_snapshot_validation[
            "pilot_hash_list"
        ],
        "two-parameter v3 raw hash list": raw_snapshot_validation["v3_hash_list"],
        "135M frozen G3 readout": {
            "path": str(g3_path),
            "sha256": sha256_file(g3_path),
        },
        "1.7B canonical G4C": {"path": str(g4c_path), "sha256": sha256_file(g4c_path)},
        "135M canonical G6": {"path": str(g6_path), "sha256": sha256_file(g6_path)},
        "135M canonical G8": {"path": str(g8_path), "sha256": sha256_file(g8_path)},
    }
    result = {
        "schema": "yeto_shared_lr_illusion_posthoc_v1",
        "analysis_status": "POST_HOC_DESCRIPTIVE_ONLY",
        "new_training_runs": 0,
        "gate_changes": 0,
        "estimand": {
            "gain": "eval_loss(mu=0) - eval_loss(raw Nesterov mu=.9)",
            "positive_direction": "momentum helps",
            "a1": "both quadratics at fitted mu0 optimum",
            "a2": "both quadratics at eta=.7, only if both observed grids contain .7",
            "b": "each quadratic at its own accepted fitted optimum",
            "flip": "shared label HELP and tuned label NULL or HURT",
            "bootstrap_evaluability": {
                "a1": "accepted mu0 optimum; momentum curve need not have an accepted optimum",
                "a2": "eta=.7 in both observed grid ranges; no fitted optimum required",
                "b": "accepted optima for both arms",
                "flip": "both the selected shared metric and b must be evaluable",
            },
        },
        "summary": counts,
        "raw_snapshot_validation": raw_snapshot_validation,
        "frozen_point_fit_validation": fit_validation,
        "v8_exact_mu_audit": g8_audit,
        "provenance": provenance,
        "records": records,
    }
    (output / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_csv(output / "table.csv", records)
    (output / "table.md").write_text(
        render_markdown(records, counts, g8_audit, provenance), encoding="utf-8"
    )
    (output / "summary.md").write_text(render_summary(counts), encoding="utf-8")
    (output / "table.tex").write_text(render_latex(records, counts), encoding="utf-8")
    (output / "subsection.tex").write_text(render_subsection(counts), encoding="utf-8")

    a1 = counts["a1_mu0_optimum"]
    print(
        "ILLUSION: "
        f"{a1['flips']}/{a1['retuning_comparable_comparisons']} jointly evaluable flips; "
        f"shared HELP={a1['shared_lr_helps']}/{a1['evaluable_comparisons']}; "
        f"{a1['shared_lr_helps_surviving_retuning']} shared HELP calls survive per-arm retuning; "
        f"in-grid a1={counts['a1_in_momentum_grid']}; "
        f"eta=.7 eligible={counts['a2_default_0p7']['evaluable_comparisons']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
