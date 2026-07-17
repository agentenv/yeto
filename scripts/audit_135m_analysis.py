#!/usr/bin/env python3
"""Frozen selection, seed-level inference, and gates for audit A1/A3/A4.

All operations are deterministic transformations of sealed cumulative phase
manifests.  Development selection uses the lowest arithmetic mean across the
registered training seeds, assigns scientific divergences positive-infinite
tuning loss, permits only the registered paired outward extension, and never
uses an interval to reselect an eta.

Confirmation inference is performed on one paired method-minus-control value
per training seed.  The report includes raw values, mean, median, sample SD,
ordinary Student-t intervals, exact sign-flip p-values, a robust seed-level
Student-t population interval, and stable Holm step-down intervals.  A4's
precision trigger uses the worst-rank Holm half-width, so it depends only on
sample size and dispersion—not the observed sign or mean.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import mpmath
import numpy as np

from scripts import audit_135m_contract as audit


ROBUST_DF = 4.0
ROBUST_LOCATION_PRIOR_SD = 0.25
ROBUST_SCALE_PRIOR_SD = 0.25
ROBUST_LOCATION_GRID = (-1.0, 1.0, 4001)
ROBUST_LOG_SCALE_GRID = (math.log(1.0e-6), math.log(1.0), 800)
FAMILY_ALPHA = 0.05
SMOLLM_EQUIVALENCE_EPSILON = 0.010


class AnalysisError(RuntimeError):
    """Sealed inputs do not support the exact registered analysis."""


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnalysisError(f"{label} must be a JSON object")
    return value


def write_create_only(path: Path, value: object) -> None:
    if path.exists():
        raise AnalysisError(f"refusing to overwrite create-only output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _sealed_at(value: str | None) -> str:
    timestamp = value or utc_now()
    if not timestamp.endswith("Z"):
        raise AnalysisError("seal time must be a UTC Z timestamp")
    try:
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AnalysisError("seal time is not ISO-8601") from exc
    return timestamp


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AnalysisError(f"{label} must be a JSON object")
    return dict(value)


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise AnalysisError(f"{label} must be an array")
    return value


def _expected_cells(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = _array(manifest.get("expected_cells"), "expected cells")
    result: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = _mapping(raw, "expected cell")
        cell_id = row.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id or cell_id in result:
            raise AnalysisError("expected cell IDs are missing/duplicated")
        result[cell_id] = row
    return result


def _final_rows(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    expected = _expected_cells(manifest)
    rows = _array(manifest.get("results"), "result rows")
    final: dict[str, tuple[int, int, dict[str, Any]]] = {}
    for order, raw in enumerate(rows):
        row = _mapping(raw, "result row")
        cell_id = row.get("cell_id")
        attempt = row.get("attempt")
        if cell_id not in expected:
            raise AnalysisError(f"result row cites unexpected cell {cell_id!r}")
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt <= 0:
            raise AnalysisError("result row lacks a positive integer attempt")
        prior = final.get(str(cell_id))
        if prior is None or (attempt, order) > (prior[0], prior[1]):
            final[str(cell_id)] = (attempt, order, row)
    return {cell_id: value[2] for cell_id, value in final.items()}


def _analysis_loss(row: Mapping[str, Any]) -> float:
    status = row.get("status")
    if status == "DIVERGED":
        return math.inf
    value = row.get("loss")
    if (
        status != "COMPLETED"
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise AnalysisError(
            f"cell {row.get('cell_id')} has no finite endpoint or retained divergence"
        )
    return float(value)


def _isclose(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-15)


def _find_cell(
    *,
    cells: Mapping[str, Mapping[str, Any]],
    h: int,
    m: int,
    mu: float,
    eta: float,
    seed: int,
    audit_stage: str | None = None,
    phases: set[str] | None = None,
    require_no_audit_stage: bool = False,
) -> str:
    matches = []
    for cell_id, cell in cells.items():
        cell_h = int(cell.get("h", cell.get("H", -1)))
        cell_m = int(cell.get("m", cell.get("M", 4)))
        if (
            cell_h != h
            or cell_m != m
            or int(cell.get("seed", -1)) != seed
            or not _isclose(float(cell.get("mu", math.nan)), mu)
            or not _isclose(float(cell.get("eta", math.nan)), eta)
        ):
            continue
        if audit_stage is not None and cell.get("audit_stage") != audit_stage:
            continue
        if require_no_audit_stage and cell.get("audit_stage") is not None:
            continue
        if phases is not None and cell.get("audit_phase") not in phases:
            continue
        matches.append(cell_id)
    if len(matches) != 1:
        raise AnalysisError(
            f"expected one cell for H={h},M={m},mu={mu},eta={eta},seed={seed}; "
            f"found {matches}"
        )
    return matches[0]


def _curve_evidence(
    *,
    label: str,
    h: int,
    m: int,
    mu: float,
    initial_grid: Sequence[float],
    allowed_grid: Sequence[float],
    seeds: Sequence[int],
    cells: Mapping[str, Mapping[str, Any]],
    final: Mapping[str, Mapping[str, Any]],
    audit_stage: str | None,
    phases: set[str] | None,
) -> dict[str, Any]:
    points = []
    for eta in sorted({float(value) for value in allowed_grid}):
        seed_rows = []
        missing = False
        for seed in seeds:
            try:
                cell_id = _find_cell(
                    cells=cells,
                    h=h,
                    m=m,
                    mu=mu,
                    eta=eta,
                    seed=seed,
                    audit_stage=audit_stage if seed != 347 else None,
                    phases=phases if seed != 347 else None,
                    require_no_audit_stage=(seed == 347),
                )
            except AnalysisError:
                missing = True
                break
            if cell_id not in final:
                raise AnalysisError(f"cell {cell_id} has no terminal result")
            loss = _analysis_loss(final[cell_id])
            seed_rows.append(
                {
                    "seed": seed,
                    "cell_id": cell_id,
                    "status": final[cell_id]["status"],
                    "loss": None if math.isinf(loss) else loss,
                    "tuning_loss_kind": (
                        "positive_infinity_scientific_divergence"
                        if math.isinf(loss)
                        else "finite_endpoint_nll"
                    ),
                }
            )
        if missing:
            continue
        losses = [
            math.inf if row["loss"] is None else float(row["loss"])
            for row in seed_rows
        ]
        pooled = sum(losses) / len(losses)
        points.append(
            {
                "eta": eta,
                "seed_rows": seed_rows,
                "pooled_mean": None if math.isinf(pooled) else pooled,
                "pooled_mean_kind": (
                    "positive_infinity_scientific_divergence"
                    if math.isinf(pooled)
                    else "finite_arithmetic_mean"
                ),
            }
        )
    if len(points) < len(initial_grid):
        raise AnalysisError(f"curve {label} lacks one or more initial eta points")

    def rank(point: Mapping[str, Any]) -> tuple[int, float, float]:
        mean = point["pooled_mean"]
        return (
            1 if mean is None else 0,
            math.inf if mean is None else float(mean),
            float(point["eta"]),
        )

    winner = min(points, key=rank)
    winner_index = points.index(winner)
    lower = points[winner_index - 1] if winner_index else None
    upper = points[winner_index + 1] if winner_index + 1 < len(points) else None
    winner_mean = winner["pooled_mean"]
    bracketed = (
        winner_mean is not None
        and lower is not None
        and upper is not None
        and lower["pooled_mean"] is not None
        and upper["pooled_mean"] is not None
        and float(lower["pooled_mean"]) > float(winner_mean)
        and float(upper["pooled_mean"]) > float(winner_mean)
    )
    initial_values = [float(value) for value in initial_grid]
    initial_boundary = _isclose(float(winner["eta"]), initial_values[0]) or _isclose(
        float(winner["eta"]), initial_values[-1]
    )
    return {
        "label": label,
        "H": h,
        "M": m,
        "mu": mu,
        "registered_initial_grid": initial_values,
        "sampled_points": points,
        "selected_eta": float(winner["eta"]),
        "selected_pooled_mean": winner_mean,
        "selected_pooled_mean_kind": winner["pooled_mean_kind"],
        "initial_boundary_winner": initial_boundary,
        "final_interior_with_worse_neighbors": bracketed,
        "lower_neighbor": (
            None
            if lower is None
            else {"eta": lower["eta"], "pooled_mean": lower["pooled_mean"]}
        ),
        "upper_neighbor": (
            None
            if upper is None
            else {"eta": upper["eta"], "pooled_mean": upper["pooled_mean"]}
        ),
    }


def _selection_outputs(
    *,
    schema: str,
    audit_stage: str,
    manifest_path: Path,
    curves: Sequence[Mapping[str, Any]],
    extension_already_run: bool,
    sealed_at_utc: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    boundary = [row for row in curves if row["initial_boundary_winner"]]
    if extension_already_run:
        boundary_status = (
            "COMPLETED"
            if all(row["final_interior_with_worse_neighbors"] for row in curves)
            else "UNBRACKETED"
        )
    else:
        boundary_status = "REQUIRED" if boundary else "NOT_REQUIRED"
    evidence = {
        "schema": f"{schema.removesuffix('_selection_v1')}_selection_evidence_v1",
        "status": "SEALED",
        "audit_stage": audit_stage,
        "authority_prereg_sha256": audit.PREREG_JSON_SHA256,
        "source_phase_manifest_raw_sha256": sha256_file(manifest_path),
        "source_phase_manifest_canonical_sha256": canonical_sha256(
            load_object(manifest_path, "cumulative phase manifest")
        ),
        "selection_rule": "lowest_pooled_development_mean",
        "scientific_divergence_tuning_loss": "positive_infinity",
        "interval_based_reselection": False,
        "one_paired_boundary_extension_maximum": True,
        "extension_already_run": extension_already_run,
        "curves": list(curves),
        "sealed_at_utc": sealed_at_utc,
    }
    evidence_hash = canonical_sha256(evidence)
    selection = {
        "schema": schema,
        "status": "SEALED",
        "audit_stage": audit_stage,
        "authority_prereg_sha256": audit.PREREG_JSON_SHA256,
        "development_evidence_canonical_sha256": evidence_hash,
        "selection_rule": "lowest_pooled_development_mean",
        "boundary_extension_status": boundary_status,
        "selected_etas": {
            str(row["label"]): float(row["selected_eta"]) for row in curves
        },
        "sealed_at_utc": sealed_at_utc,
    }
    return evidence, selection


def select_a1(args: argparse.Namespace) -> dict[str, Any]:
    audit.load_authority()
    manifest = load_object(args.phase_manifest, "A1 cumulative phase manifest")
    cells = _expected_cells(manifest)
    final = _final_rows(manifest)
    curves = []
    for h, method_mu in ((16, 0.9), (256, 0.5)):
        for mu in (0.0, method_mu):
            key = f"H{h}_mu{mu:g}"
            grid = audit.A1_GRIDS[(h, mu)]
            allowed = set(grid)
            allowed.update(
                (
                    audit._outward_extension(grid, float(grid[0])),
                    audit._outward_extension(grid, float(grid[-1])),
                )
            )
            curves.append(
                _curve_evidence(
                    label=key,
                    h=h,
                    m=4,
                    mu=mu,
                    initial_grid=grid,
                    allowed_grid=sorted(allowed),
                    seeds=(347, 359, 373),
                    cells=cells,
                    final=final,
                    audit_stage="A1",
                    phases={"development_initial", "development_boundary_extension"},
                )
            )
    extension_run = any(
        cell.get("audit_stage") == "A1"
        and cell.get("audit_phase") == "development_boundary_extension"
        for cell in cells.values()
    )
    sealed = _sealed_at(args.sealed_at_utc)
    evidence, selection = _selection_outputs(
        schema="audit_135m_a1_development_selection_v1",
        audit_stage="A1",
        manifest_path=args.phase_manifest,
        curves=curves,
        extension_already_run=extension_run,
        sealed_at_utc=sealed,
    )
    write_create_only(args.evidence_output, evidence)
    write_create_only(args.selection_output, selection)
    return {
        "status": "SEALED",
        "selection": str(args.selection_output),
        "selection_sha256": sha256_file(args.selection_output),
        "evidence": str(args.evidence_output),
        "boundary_extension_status": selection["boundary_extension_status"],
        "selected_etas": selection["selected_etas"],
    }


def select_a4(args: argparse.Namespace) -> dict[str, Any]:
    audit.load_authority()
    manifest = load_object(args.phase_manifest, "A4 cumulative phase manifest")
    cells = _expected_cells(manifest)
    final = _final_rows(manifest)
    curves = []
    for m in (1, 4):
        for h, method_mu in ((16, 0.9), (256, 0.5)):
            for mu in (0.0, method_mu):
                key = f"M{m}_H{h}_mu{mu:g}"
                grid = audit.A4_GRIDS[(h, mu)]
                allowed = set(grid)
                allowed.update(
                    (
                        audit._outward_extension(grid, float(grid[0])),
                        audit._outward_extension(grid, float(grid[-1])),
                    )
                )
                curves.append(
                    _curve_evidence(
                        label=key,
                        h=h,
                        m=m,
                        mu=mu,
                        initial_grid=grid,
                        allowed_grid=sorted(allowed),
                        seeds=(2069, 2081),
                        cells=cells,
                        final=final,
                        audit_stage="A4",
                        phases={
                            "development_initial",
                            "development_boundary_extension",
                        },
                    )
                )
    extension_run = any(
        cell.get("audit_stage") == "A4"
        and cell.get("audit_phase") == "development_boundary_extension"
        for cell in cells.values()
    )
    sealed = _sealed_at(args.sealed_at_utc)
    evidence, selection = _selection_outputs(
        schema="audit_135m_a4_development_selection_v1",
        audit_stage="A4",
        manifest_path=args.phase_manifest,
        curves=curves,
        extension_already_run=extension_run,
        sealed_at_utc=sealed,
    )
    write_create_only(args.evidence_output, evidence)
    write_create_only(args.selection_output, selection)
    return {
        "status": "SEALED",
        "selection": str(args.selection_output),
        "selection_sha256": sha256_file(args.selection_output),
        "evidence": str(args.evidence_output),
        "boundary_extension_status": selection["boundary_extension_status"],
        "selected_etas": selection["selected_etas"],
    }


def select_a3(args: argparse.Namespace) -> dict[str, Any]:
    audit.load_authority()
    manifest = load_object(args.phase_manifest, "A3 cumulative phase manifest")
    cells = _expected_cells(manifest)
    final = _final_rows(manifest)
    curves = []
    for h in (8, 512):
        allowed = set(audit.A3_GRIDS[h]) | set(audit.A3_ALLOWED_EXTENSIONS[h])
        curves.append(
            _curve_evidence(
                label=f"H{h}",
                h=h,
                m=4,
                mu=0.0,
                initial_grid=audit.A3_GRIDS[h],
                allowed_grid=sorted(allowed),
                seeds=(359, 373),
                cells=cells,
                final=final,
                audit_stage="A3",
                phases={"frontier_initial", "frontier_boundary_extension"},
            )
        )
    extension_run = any(
        cell.get("audit_stage") == "A3"
        and cell.get("audit_phase") == "frontier_boundary_extension"
        for cell in cells.values()
    )
    extension_etas: dict[str, float] = {}
    if not extension_run:
        for row in curves:
            if not row["initial_boundary_winner"]:
                continue
            h = int(row["H"])
            grid = audit.A3_GRIDS[h]
            winner = float(row["selected_eta"])
            if _isclose(winner, float(grid[0])):
                extension_etas[str(h)] = min(audit.A3_ALLOWED_EXTENSIONS[h])
            elif _isclose(winner, float(grid[-1])):
                extension_etas[str(h)] = max(audit.A3_ALLOWED_EXTENSIONS[h])
            else:
                raise AnalysisError("A3 boundary direction is not registered")
    sealed = _sealed_at(args.sealed_at_utc)
    evidence = {
        "schema": "audit_135m_a3_frontier_selection_evidence_v1",
        "status": "SEALED",
        "audit_stage": "A3",
        "authority_prereg_sha256": audit.PREREG_JSON_SHA256,
        "source_phase_manifest_raw_sha256": sha256_file(args.phase_manifest),
        "source_phase_manifest_canonical_sha256": canonical_sha256(manifest),
        "selection_rule": "lowest_pooled_development_mean",
        "scientific_divergence_tuning_loss": "positive_infinity",
        "extension_already_run": extension_run,
        "curves": curves,
        "sealed_at_utc": sealed,
    }
    decision = {
        "schema": "audit_135m_a3_frontier_selection_v1",
        "status": "SEALED",
        "audit_stage": "A3",
        "authority_prereg_sha256": audit.PREREG_JSON_SHA256,
        "frontier_evidence_canonical_sha256": canonical_sha256(evidence),
        "extension_rule": "one_registered_outward_boundary_extension_maximum",
        "extension_etas": extension_etas,
        "sealed_at_utc": sealed,
    }
    write_create_only(args.evidence_output, evidence)
    write_create_only(args.selection_output, decision)
    return {
        "status": "SEALED",
        "extension_required": bool(extension_etas),
        "extension_etas": extension_etas,
        "final_selected_etas": {
            str(row["label"]): row["selected_eta"] for row in curves
        },
        "all_final_curves_bracketed": all(
            row["final_interior_with_worse_neighbors"] for row in curves
        ),
    }


def _student_t_cdf(value: float, degrees: int) -> float:
    if degrees <= 0:
        raise AnalysisError("Student-t degrees of freedom must be positive")
    if value == 0.0:
        return 0.5
    x = degrees / (degrees + value * value)
    beta = float(
        mpmath.betainc(degrees / 2.0, 0.5, 0.0, x, regularized=True)
    )
    return 1.0 - 0.5 * beta if value > 0.0 else 0.5 * beta


def _student_t_ppf(probability: float, degrees: int) -> float:
    if not 0.0 < probability < 1.0:
        raise AnalysisError("Student-t quantile probability is outside (0,1)")
    if probability == 0.5:
        return 0.0
    if probability < 0.5:
        return -_student_t_ppf(1.0 - probability, degrees)
    low, high = 0.0, 1.0
    while _student_t_cdf(high, degrees) < probability:
        high *= 2.0
        if high > 1.0e6:
            raise AnalysisError("Student-t quantile failed to bracket")
    for _ in range(100):
        middle = (low + high) * 0.5
        if _student_t_cdf(middle, degrees) < probability:
            low = middle
        else:
            high = middle
    return (low + high) * 0.5


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(
        maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True)),
        axis=axis,
    )


def _weighted_quantile(grid: np.ndarray, weights: np.ndarray, probability: float) -> float:
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    return float(np.interp(probability, cumulative, grid))


def robust_student_t_interval(values: Sequence[float]) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64)
    if data.size < 2 or np.any(~np.isfinite(data)):
        raise AnalysisError("robust Student-t interval needs at least two finite seeds")
    theta = np.linspace(*ROBUST_LOCATION_GRID, dtype=np.float64)
    log_scale = np.linspace(*ROBUST_LOG_SCALE_GRID, dtype=np.float64)
    scale = np.exp(log_scale)
    log_marginal = np.empty(theta.size, dtype=np.float64)
    constant = math.lgamma((ROBUST_DF + 1.0) / 2.0) - math.lgamma(
        ROBUST_DF / 2.0
    ) - 0.5 * math.log(ROBUST_DF * math.pi)
    log_scale_prior = (
        math.log(math.sqrt(2.0 / math.pi) / ROBUST_SCALE_PRIOR_SD)
        - 0.5 * (scale / ROBUST_SCALE_PRIOR_SD) ** 2
        + log_scale
    )
    chunk = 128
    for start in range(0, theta.size, chunk):
        stop = min(theta.size, start + chunk)
        location = theta[start:stop, None, None]
        sigma = scale[None, :, None]
        residual = (data[None, None, :] - location) / sigma
        log_likelihood = np.sum(
            constant
            - np.log(sigma)
            - ((ROBUST_DF + 1.0) / 2.0)
            * np.log1p((residual * residual) / ROBUST_DF),
            axis=2,
        )
        log_marginal[start:stop] = _logsumexp(
            log_likelihood + log_scale_prior[None, :], axis=1
        )
    log_marginal += -0.5 * (theta / ROBUST_LOCATION_PRIOR_SD) ** 2
    log_marginal -= float(np.max(log_marginal))
    weights = np.exp(log_marginal)
    weights /= float(np.sum(weights))
    edge_mass = float(weights[:10].sum() + weights[-10:].sum())
    if edge_mass > 1.0e-6:
        raise AnalysisError("robust Student-t posterior hits its frozen location grid")
    return {
        "model": "seed_difference ~ StudentT(df=4, location=theta, scale=sigma)",
        "seed_is_top_level_cluster": True,
        "priors": {
            "theta": "Normal(0,0.25)",
            "sigma": "HalfNormal(0.25)",
        },
        "integration": {
            "theta_grid": list(ROBUST_LOCATION_GRID),
            "log_sigma_grid": list(ROBUST_LOG_SCALE_GRID),
        },
        "posterior_mean": float(np.sum(theta * weights)),
        "posterior_median": _weighted_quantile(theta, weights, 0.5),
        "interval_95": [
            _weighted_quantile(theta, weights, 0.025),
            _weighted_quantile(theta, weights, 0.975),
        ],
        "posterior_edge_mass": edge_mass,
    }


def _exact_sign_flip_p(values: Sequence[float]) -> float:
    observed = abs(sum(values) / len(values))
    exceed = 0
    total = 1 << len(values)
    for bits in range(total):
        candidate = sum(
            value if bits & (1 << index) else -value
            for index, value in enumerate(values)
        ) / len(values)
        if abs(candidate) >= observed - 1.0e-15:
            exceed += 1
    return exceed / total


def paired_summary(label: str, by_seed: Mapping[int, float]) -> dict[str, Any]:
    seeds = sorted(by_seed)
    values = [float(by_seed[seed]) for seed in seeds]
    if len(values) < 2:
        raise AnalysisError(f"paired contrast {label} needs at least two seeds")
    if any(not math.isfinite(value) for value in values):
        return {
            "label": label,
            "seeds": seeds,
            "raw_paired_values": [
                value if math.isfinite(value) else None for value in values
            ],
            "raw_paired_value_kinds": [
                (
                    "finite"
                    if math.isfinite(value)
                    else (
                        "positive_infinity"
                        if value > 0.0
                        else "negative_infinity" if value < 0.0 else "undefined"
                    )
                )
                for value in values
            ],
            "by_seed": {
                str(seed): (
                    by_seed[seed] if math.isfinite(by_seed[seed]) else None
                )
                for seed in seeds
            },
            "n": len(values),
            "finite_inference_available": False,
            "mean": None,
            "median": None,
            "sample_sd": None,
            "standard_error": None,
            "student_t_df": len(values) - 1,
            "student_t_statistic": None,
            "student_t_two_sided_p": None,
            "student_t_interval_95": [None, None],
            "exact_sign_flip_two_sided_p": None,
            "robust_student_t_hierarchical": None,
            "nonfinite_divergence_retained": True,
        }
    mean = sum(values) / len(values)
    median = float(np.median(np.asarray(values)))
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    sd = math.sqrt(variance)
    se = sd / math.sqrt(len(values))
    critical = _student_t_ppf(0.975, len(values) - 1)
    statistic = math.inf if se == 0.0 and mean != 0.0 else (0.0 if se == 0.0 else mean / se)
    p_value = 0.0 if math.isinf(statistic) else 2.0 * (
        1.0 - _student_t_cdf(abs(statistic), len(values) - 1)
    )
    return {
        "label": label,
        "seeds": seeds,
        "raw_paired_values": values,
        "by_seed": {str(seed): by_seed[seed] for seed in seeds},
        "n": len(values),
        "finite_inference_available": True,
        "mean": mean,
        "median": median,
        "sample_sd": sd,
        "standard_error": se,
        "student_t_df": len(values) - 1,
        "student_t_statistic": statistic,
        "student_t_two_sided_p": p_value,
        "student_t_interval_95": [mean - critical * se, mean + critical * se],
        "exact_sign_flip_two_sided_p": _exact_sign_flip_p(values),
        "robust_student_t_hierarchical": robust_student_t_interval(values),
    }


def holm_intervals(
    summaries: Mapping[str, Mapping[str, Any]], *, alpha: float = FAMILY_ALPHA
) -> dict[str, dict[str, Any]]:
    ordered = sorted(
        summaries,
        key=lambda label: (
            (
                math.inf
                if summaries[label]["student_t_two_sided_p"] is None
                else float(summaries[label]["student_t_two_sided_p"])
            ),
            label.encode("utf-8"),
        ),
    )
    family_size = len(ordered)
    result: dict[str, dict[str, Any]] = {}
    for rank, label in enumerate(ordered):
        summary = summaries[label]
        local_alpha = alpha / (family_size - rank)
        if summary.get("finite_inference_available") is not True:
            result[label] = {
                "holm_rank": rank + 1,
                "family_size": family_size,
                "two_sided_alpha": local_alpha,
                "critical_t": None,
                "half_width": None,
                "interval": [None, None],
                "finite_inference_available": False,
            }
            continue
        critical = _student_t_ppf(
            1.0 - local_alpha / 2.0, int(summary["student_t_df"])
        )
        half_width = critical * float(summary["standard_error"])
        mean = float(summary["mean"])
        result[label] = {
            "holm_rank": rank + 1,
            "family_size": family_size,
            "two_sided_alpha": local_alpha,
            "critical_t": critical,
            "half_width": half_width,
            "interval": [mean - half_width, mean + half_width],
            "finite_inference_available": True,
        }
    return result


def _confirmation_differences(
    *,
    manifest: Mapping[str, Any],
    audit_stage: str,
    phases: set[str],
    seeds: Iterable[int],
    dimensions: Iterable[tuple[int, int, float]],
) -> dict[tuple[int, int], dict[str, dict[int, float]]]:
    cells = _expected_cells(manifest)
    final = _final_rows(manifest)
    output: dict[tuple[int, int], dict[str, dict[int, float]]] = {}
    for m, h, method_mu in dimensions:
        roles: dict[int, dict[str, float]] = {}
        for seed in seeds:
            rows = {}
            for cell_id, cell in cells.items():
                if (
                    cell.get("audit_stage") == audit_stage
                    and cell.get("audit_phase") in phases
                    and int(cell.get("m", cell.get("M", -1))) == m
                    and int(cell.get("h", cell.get("H", -1))) == h
                    and int(cell.get("seed", -1)) == seed
                ):
                    role = cell.get("analysis_role")
                    if role in {
                        "fixed_control",
                        "fixed_method",
                        "tuned_control",
                        "tuned_method",
                    }:
                        if cell_id not in final:
                            raise AnalysisError(f"confirmation cell {cell_id} is unresolved")
                        loss = _analysis_loss(final[cell_id])
                        rows[str(role)] = loss
            if set(rows) != {
                "fixed_control",
                "fixed_method",
                "tuned_control",
                "tuned_method",
            }:
                raise AnalysisError(
                    f"confirmation block M={m},H={h},seed={seed} is incomplete"
                )
            roles[seed] = rows
        output[(m, h)] = {
            "fixed": {
                seed: _paired_difference(
                    roles[seed]["fixed_method"], roles[seed]["fixed_control"]
                )
                for seed in roles
            },
            "tuned": {
                seed: _paired_difference(
                    roles[seed]["tuned_method"], roles[seed]["tuned_control"]
                )
                for seed in roles
            },
        }
    return output


def _paired_difference(method: float, control: float) -> float:
    if math.isinf(method) and math.isinf(control):
        return math.nan
    return method - control


def _classification(
    *,
    fixed_interval: Sequence[float | None],
    tuned_interval: Sequence[float | None],
    tuned_mean: float | None,
) -> str:
    if (
        fixed_interval[0] is None
        or fixed_interval[1] is None
        or tuned_interval[0] is None
        or tuned_interval[1] is None
        or tuned_mean is None
    ):
        return "ATTENUATED_OR_INCONCLUSIVE"
    fixed_replicates = float(fixed_interval[0]) > 0.0
    if not fixed_replicates:
        return "FIXED_NOT_REPLICATED"
    if (
        float(tuned_interval[0]) >= -SMOLLM_EQUIVALENCE_EPSILON
        and float(tuned_interval[1]) <= SMOLLM_EQUIVALENCE_EPSILON
    ):
        return "COLLAPSES_WITH_TUNING"
    if float(tuned_interval[0]) > 0.0 and abs(tuned_mean) >= SMOLLM_EQUIVALENCE_EPSILON:
        return "SURVIVES_TUNING"
    if float(tuned_interval[1]) < 0.0:
        return "REVERSES_WITH_TUNING"
    return "ATTENUATED_OR_INCONCLUSIVE"


def analyze_a1(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_object(args.phase_manifest, "A1 final phase manifest")
    seeds = tuple(seed for seed, _training in audit.A1_CONFIRMATION_SEEDS)
    differences = _confirmation_differences(
        manifest=manifest,
        audit_stage="A1",
        phases={"confirmation"},
        seeds=seeds,
        dimensions=((4, 16, 0.9), (4, 256, 0.5)),
    )
    fixed = {
        f"H{h}_fixed": paired_summary(f"H{h}_fixed", differences[(4, h)]["fixed"])
        for h in (16, 256)
    }
    tuned = {
        f"H{h}_tuned": paired_summary(f"H{h}_tuned", differences[(4, h)]["tuned"])
        for h in (16, 256)
    }
    fixed_holm = holm_intervals(fixed)
    tuned_holm = holm_intervals(tuned)
    classifications = {}
    for h in (16, 256):
        classifications[f"H{h}"] = _classification(
            fixed_interval=fixed_holm[f"H{h}_fixed"]["interval"],
            tuned_interval=tuned_holm[f"H{h}_tuned"]["interval"],
            tuned_mean=tuned[f"H{h}_tuned"]["mean"],
        )
    g2 = all(
        fixed_holm[label]["interval"][0] is not None
        and fixed_holm[label]["interval"][0] > 0.0
        for label in fixed_holm
    )
    g3 = all(
        row["interval"][0] is not None
        and row["interval"][1] is not None
        and row["interval"][0] >= -SMOLLM_EQUIVALENCE_EPSILON
        and row["interval"][1] <= SMOLLM_EQUIVALENCE_EPSILON
        for row in tuned_holm.values()
    )
    report = {
        "schema": "audit_135m_a1_analysis_v1",
        "status": "SEALED",
        "audit_stage": "A1",
        "authority_prereg_sha256": audit.PREREG_JSON_SHA256,
        "source_phase_manifest_raw_sha256": sha256_file(args.phase_manifest),
        "source_phase_manifest_canonical_sha256": canonical_sha256(manifest),
        "selection_manifest_raw_sha256": sha256_file(args.selection_manifest),
        "selection_manifest_canonical_sha256": canonical_sha256(
            load_object(args.selection_manifest, "A1 selection manifest")
        ),
        "estimand": "method_minus_live_within_seed_control_nll",
        "equivalence_epsilon": SMOLLM_EQUIVALENCE_EPSILON,
        "fixed_contrasts": fixed,
        "tuned_contrasts": tuned,
        "holm_adjusted_fixed": fixed_holm,
        "holm_adjusted_tuned": tuned_holm,
        "classifications": classifications,
        "gates": {
            "G2_A1_fixed_replication": "PASS" if g2 else "FAIL",
            "G3_A1_tuned_collapse": "PASS" if g3 else "FAIL",
            "A1": "PASS" if g2 and g3 else "FAIL_CLASSIFIED",
        },
        "sealed_at_utc": _sealed_at(args.sealed_at_utc),
    }
    report["analysis_canonical_sha256"] = canonical_sha256(report)
    write_create_only(args.output, report)
    return {
        "status": "SEALED",
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "gates": report["gates"],
        "classifications": classifications,
    }


def precision_a4(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_object(args.phase_manifest, "A4 initial confirmation phase")
    seeds = tuple(seed for seed, _training in audit.A4_CONFIRMATION_SEEDS)
    differences = _confirmation_differences(
        manifest=manifest,
        audit_stage="A4",
        phases={"confirmation_initial"},
        seeds=seeds,
        dimensions=(
            (1, 16, 0.9),
            (1, 256, 0.5),
            (4, 16, 0.9),
            (4, 256, 0.5),
        ),
    )
    family_size = 4
    local_alpha = FAMILY_ALPHA / family_size
    widths = {}
    for (m, h), contrasts in differences.items():
        values = list(contrasts["tuned"].values())
        if any(not math.isfinite(value) for value in values):
            widths[f"M{m}_H{h}_tuned"] = {
                "n": len(values),
                "sample_sd": None,
                "worst_holm_rank_two_sided_alpha": local_alpha,
                "critical_t": None,
                "half_width": None,
                "exceeds_epsilon": True,
                "nonfinite_divergence_retained": True,
            }
            continue
        mean = sum(values) / len(values)
        sd = math.sqrt(
            sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        )
        critical = _student_t_ppf(1.0 - local_alpha / 2.0, len(values) - 1)
        half_width = critical * sd / math.sqrt(len(values))
        widths[f"M{m}_H{h}_tuned"] = {
            "n": len(values),
            "sample_sd": sd,
            "worst_holm_rank_two_sided_alpha": local_alpha,
            "critical_t": critical,
            "half_width": half_width,
            "exceeds_epsilon": half_width > SMOLLM_EQUIVALENCE_EPSILON,
        }
    expansion = any(row["exceeds_epsilon"] for row in widths.values())
    sealed = _sealed_at(args.sealed_at_utc)
    evidence = {
        "schema": "audit_135m_a4_precision_evidence_v1",
        "status": "SEALED",
        "authority_prereg_sha256": audit.PREREG_JSON_SHA256,
        "source_phase_manifest_raw_sha256": sha256_file(args.phase_manifest),
        "source_phase_manifest_canonical_sha256": canonical_sha256(manifest),
        "initial_confirmation_complete": True,
        "sign_or_mean_used": False,
        "family": "four_tuned_M_by_H_co_primary_contrasts",
        "adjustment": "worst_rank_Holm_envelope_alpha_0.05_over_4",
        "epsilon": SMOLLM_EQUIVALENCE_EPSILON,
        "widths": widths,
        "expansion_required": expansion,
        "sealed_at_utc": sealed,
    }
    trigger = {
        "schema": "audit_135m_a4_precision_trigger_v1",
        "status": "SEALED",
        "audit_stage": "A4",
        "authority_prereg_sha256": audit.PREREG_JSON_SHA256,
        "initial_confirmation_evidence_canonical_sha256": canonical_sha256(evidence),
        "initial_confirmation_complete": True,
        "precision_expansion_rule": (
            "adjusted_ci_half_width_exceeds_epsilon_after_complete_initial_seed_block"
        ),
        "epsilon": SMOLLM_EQUIVALENCE_EPSILON,
        "triggered_sign_blindly": True,
        "expansion_required": expansion,
        "run_all_registered_expansion_seeds": expansion,
        "sealed_at_utc": sealed,
    }
    write_create_only(args.evidence_output, evidence)
    write_create_only(args.trigger_output, trigger)
    return {
        "status": "SEALED",
        "expansion_required": expansion,
        "evidence": str(args.evidence_output),
        "trigger": str(args.trigger_output),
    }


def analyze_a4(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_object(args.phase_manifest, "A4 final phase manifest")
    cells = _expected_cells(manifest)
    expansion_present = any(
        cell.get("audit_stage") == "A4"
        and cell.get("audit_phase") == "confirmation_precision_expansion"
        for cell in cells.values()
    )
    seeds = [seed for seed, _training in audit.A4_CONFIRMATION_SEEDS]
    phases = {"confirmation_initial"}
    if expansion_present:
        seeds.extend(seed for seed, _training in audit.A4_EXPANSION_SEEDS)
        phases.add("confirmation_precision_expansion")
    differences = _confirmation_differences(
        manifest=manifest,
        audit_stage="A4",
        phases=phases,
        seeds=seeds,
        dimensions=(
            (1, 16, 0.9),
            (1, 256, 0.5),
            (4, 16, 0.9),
            (4, 256, 0.5),
        ),
    )
    fixed = {}
    tuned = {}
    for m, h in sorted(differences):
        fixed[f"M{m}_H{h}_fixed"] = paired_summary(
            f"M{m}_H{h}_fixed", differences[(m, h)]["fixed"]
        )
        tuned[f"M{m}_H{h}_tuned"] = paired_summary(
            f"M{m}_H{h}_tuned", differences[(m, h)]["tuned"]
        )
    fixed_holm = holm_intervals(fixed)
    tuned_holm = holm_intervals(tuned)
    classifications = {}
    for m, h in sorted(differences):
        classifications[f"M{m}_H{h}"] = _classification(
            fixed_interval=fixed_holm[f"M{m}_H{h}_fixed"]["interval"],
            tuned_interval=tuned_holm[f"M{m}_H{h}_tuned"]["interval"],
            tuned_mean=tuned[f"M{m}_H{h}_tuned"]["mean"],
        )
    interactions = {}
    for h in (16, 256):
        by_seed = {
            seed: (
                differences[(1, h)]["tuned"][seed]
                - differences[(1, h)]["fixed"][seed]
                - differences[(4, h)]["tuned"][seed]
                + differences[(4, h)]["fixed"][seed]
            )
            for seed in seeds
        }
        interactions[f"H{h}"] = paired_summary(
            f"H{h}_M_by_tuning_status_by_method", by_seed
        )
    interaction_holm = holm_intervals(interactions)
    fixed_pass = all(
        row["interval"][0] is not None and row["interval"][0] > 0.0
        for row in fixed_holm.values()
    )
    tuned_pass = all(
        row["interval"][0] is not None
        and row["interval"][1] is not None
        and row["interval"][0] >= -SMOLLM_EQUIVALENCE_EPSILON
        and row["interval"][1] <= SMOLLM_EQUIVALENCE_EPSILON
        for row in tuned_holm.values()
    )
    per_m = {
        f"M{m}": {
            "fixed_replication": all(
                fixed_holm[f"M{m}_H{h}_fixed"]["interval"][0] is not None
                and fixed_holm[f"M{m}_H{h}_fixed"]["interval"][0] > 0.0
                for h in (16, 256)
            ),
            "tuned_equivalence": all(
                tuned_holm[f"M{m}_H{h}_tuned"]["interval"][0] is not None
                and tuned_holm[f"M{m}_H{h}_tuned"]["interval"][1] is not None
                and tuned_holm[f"M{m}_H{h}_tuned"]["interval"][0]
                >= -SMOLLM_EQUIVALENCE_EPSILON
                and tuned_holm[f"M{m}_H{h}_tuned"]["interval"][1]
                <= SMOLLM_EQUIVALENCE_EPSILON
                for h in (16, 256)
            ),
        }
        for m in (1, 4)
    }
    report = {
        "schema": "audit_135m_a4_analysis_v1",
        "status": "SEALED",
        "audit_stage": "A4",
        "authority_prereg_sha256": audit.PREREG_JSON_SHA256,
        "source_phase_manifest_raw_sha256": sha256_file(args.phase_manifest),
        "source_phase_manifest_canonical_sha256": canonical_sha256(manifest),
        "selection_manifest_raw_sha256": sha256_file(args.selection_manifest),
        "selection_manifest_canonical_sha256": canonical_sha256(
            load_object(args.selection_manifest, "A4 selection manifest")
        ),
        "confirmation_seed_count": len(seeds),
        "precision_expansion_present": expansion_present,
        "estimand": "method_minus_live_within_seed_control_nll",
        "equivalence_epsilon": SMOLLM_EQUIVALENCE_EPSILON,
        "fixed_contrasts": fixed,
        "tuned_contrasts": tuned,
        "holm_adjusted_fixed": fixed_holm,
        "holm_adjusted_tuned": tuned_holm,
        "M_by_tuning_status_by_method_interactions": interactions,
        "holm_adjusted_interactions": interaction_holm,
        "classifications": classifications,
        "per_M_gate_components": per_m,
        "gates": {
            "G8_A4_M_axis": "PASS" if fixed_pass and tuned_pass else "FAIL_SCOPED",
            "fixed_all_M_H": fixed_pass,
            "tuned_equivalence_all_M_H": tuned_pass,
        },
        "sealed_at_utc": _sealed_at(args.sealed_at_utc),
    }
    report["analysis_canonical_sha256"] = canonical_sha256(report)
    write_create_only(args.output, report)
    return {
        "status": "SEALED",
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "gates": report["gates"],
        "per_M": per_m,
    }


def _selection_parser(sub, name: str) -> None:
    parser = sub.add_parser(name)
    parser.add_argument("--phase-manifest", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    parser.add_argument("--selection-output", type=Path, required=True)
    parser.add_argument("--sealed-at-utc")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    _selection_parser(sub, "select-a1")
    _selection_parser(sub, "select-a3")
    _selection_parser(sub, "select-a4")

    precision = sub.add_parser("precision-a4")
    precision.add_argument("--phase-manifest", type=Path, required=True)
    precision.add_argument("--evidence-output", type=Path, required=True)
    precision.add_argument("--trigger-output", type=Path, required=True)
    precision.add_argument("--sealed-at-utc")

    for name in ("analyze-a1", "analyze-a4"):
        analysis = sub.add_parser(name)
        analysis.add_argument("--phase-manifest", type=Path, required=True)
        analysis.add_argument("--selection-manifest", type=Path, required=True)
        analysis.add_argument("--output", type=Path, required=True)
        analysis.add_argument("--sealed-at-utc")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    actions = {
        "select-a1": select_a1,
        "select-a3": select_a3,
        "select-a4": select_a4,
        "precision-a4": precision_a4,
        "analyze-a1": analyze_a1,
        "analyze-a4": analyze_a4,
    }
    try:
        result = actions[args.action](args)
    except (AnalysisError, OSError, ValueError, KeyError, audit.AuditContractError) as exc:
        print(f"audit analysis error: {exc}", file=__import__("sys").stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
