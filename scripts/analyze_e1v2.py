#!/usr/bin/env python3
"""Apply the registered E1' (outer-muP v2 confirmatory) analysis and emit the G1' readout.

Registered under experiment-specs/outer-mup-v2-confirm-prereg.{json,md}.
Frozen before launch; its raw-file SHA-256 is recorded in the v2 contract JSON.

Registered v2 corrections relative to the v1 analyzer (outer-mup-v1-closure.md):
  1. rho is computed for every H from mu=0 telemetry alone, unconditionally on the
     validity of the paired mu=.9 curve fit (v1 control-flow defect skipped rho when
     either fit was unbracketed).
  2. The gate rho is the registered noise-corrected effective persistence
     rho_eff(H) = 1 - 4c/(1+3c), with c the fragment-balanced cross-worker mean
     cosine from mu=0 telemetry at the eta grid point nearest the pooled
     eta_star(mu=0). Raw lag-1..4 projected cosines are reported as diagnostics only.
  3. The quadratic fit in x = log2(eta) uses the actual registered eta values, so the
     registered 2x-spaced H64 mu.9 ladder is fitted exactly as registered.
  4. None-guards everywhere a fit can be UNBRACKETED or an input can be missing:
     the analyzer always emits a complete readout with verdict NOT_EVALUABLE instead
     of crashing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

PRIMARY_SEEDS = (211, 223, 227, 229, 233)
TOPUP_SEEDS = (239, 241, 251)
CONTESTED_MU09_HORIZONS = (64, 256, 512)
HORIZONS = (16, 64, 256, 512)
MU_HIGH = 0.9
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260724
EXPECTED_CELLS = 196
EXPECTED_PRIMARY = 160
TELEMETRY_SCHEMA = "yeto_rho_telemetry_v1"


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
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnalysisError(f"{path}: expected JSON object")
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
                    raise AnalysisError(f"{path}:{number}: expected object")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read {path}: {exc}") from exc
    return rows


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
    """Registered fit: loss = a*x^2 + b*x + c in x = log2(eta) on the exact
    registered eta values. Ladder spacing (sqrt2 or the registered 2x H64 mu.9
    ladder) enters only through the actual x coordinates."""
    if len(etas) != 4 or len(losses) != 4:
        raise AnalysisError("registered eta fit requires exactly four points")
    if any(not math.isfinite(value) for value in (*etas, *losses)):
        raise AnalysisError("eta fit received nonfinite input")
    xs = [math.log2(eta) for eta in etas]
    sums = [sum(x ** power for x in xs) for power in range(5)]
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
    vertex = -b / (2.0 * a) if a != 0 else math.nan
    interior = a > 0 and min(xs) < vertex < max(xs)
    eta_star = 2.0 ** vertex if interior else None
    return {
        "a": a,
        "b": b,
        "c": c,
        "vertex_log2_eta": vertex if math.isfinite(vertex) else None,
        "eta_star": eta_star,
        "interior": interior,
        "status": "INTERIOR" if interior else "UNBRACKETED",
    }


def quantile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise AnalysisError("quantile of empty sequence")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def curve_fit(
    losses: dict[tuple[int, float, int, float], float],
    h: int,
    mu: float,
    seeds: tuple[int, ...],
    sampled_indices: list[int] | None = None,
) -> dict:
    etas = sorted({key[3] for key in losses if key[0] == h and key[1] == mu})
    if len(etas) != 4:
        raise AnalysisError(f"H{h} mu{mu}: expected four etas, found {etas}")
    selected_seeds = (
        [seeds[index] for index in sampled_indices]
        if sampled_indices is not None else list(seeds)
    )
    means = []
    for eta in etas:
        values = []
        for seed in selected_seeds:
            key = (h, mu, seed, eta)
            if key not in losses:
                raise AnalysisError(f"H{h} mu{mu}: missing seed cell {key}")
            values.append(losses[key])
        means.append(sum(values) / len(values))
    fit = fit_quadratic(etas, means)
    fit.update({"etas": etas, "seed_mean_losses": means})
    return fit


def safe_curve_fit(
    losses: dict[tuple[int, float, int, float], float],
    h: int,
    mu: float,
    seeds: tuple[int, ...],
    sampled_indices: list[int] | None = None,
) -> dict:
    """None-guarded pooled fit: invalid/missing input yields an inert record
    instead of an exception so the readout is always emitted."""
    try:
        return curve_fit(losses, h, mu, seeds, sampled_indices)
    except AnalysisError as exc:
        return {
            "a": None, "b": None, "c": None,
            "vertex_log2_eta": None, "eta_star": None,
            "interior": False, "status": "INVALID_INPUT",
            "etas": sorted({key[3] for key in losses if key[0] == h and key[1] == mu}),
            "seed_mean_losses": None,
            "error": str(exc),
        }


def bootstrap_curve(
    losses: dict[tuple[int, float, int, float], float],
    h: int,
    mu: float,
    seeds: tuple[int, ...],
    draws: list[list[int]],
) -> dict:
    samples = []
    invalid = 0
    for draw in draws:
        fit = curve_fit(losses, h, mu, seeds, draw)
        if not fit["interior"]:
            invalid += 1
        else:
            samples.append(math.log2(fit["eta_star"]))
    valid_all = invalid == 0
    return {
        "method": "paired_nonparametric_training_seed_curve_bootstrap",
        "replicates": len(draws),
        "seed": BOOTSTRAP_SEED,
        "coordinate": "log2_eta",
        "invalid_unbracketed_replicates": invalid,
        "status": "VALID" if valid_all else "NOT_EVALUABLE",
        "ci_95_log2_eta": {
            "low": quantile(samples, 0.025) if samples else None,
            "high": quantile(samples, 0.975) if samples else None,
        },
        "ci_95_eta": {
            "low": 2.0 ** quantile(samples, 0.025) if samples else None,
            "high": 2.0 ** quantile(samples, 0.975) if samples else None,
        },
    }


def bootstrap_ratio(
    losses: dict[tuple[int, float, int, float], float],
    h: int,
    draws: list[list[int]],
) -> dict:
    log2_d = []
    invalid = 0
    for draw in draws:
        fit0 = curve_fit(losses, h, 0.0, PRIMARY_SEEDS, draw)
        fit9 = curve_fit(losses, h, MU_HIGH, PRIMARY_SEEDS, draw)
        if not fit0["interior"] or not fit9["interior"]:
            invalid += 1
            continue
        log2_d.append(math.log2((fit9["eta_star"] / fit0["eta_star"]) / (1.0 - MU_HIGH)))
    low_log = quantile(log2_d, 0.025) if log2_d else None
    high_log = quantile(log2_d, 0.975) if log2_d else None
    return {
        "method": "paired_nonparametric_training_seed_curve_bootstrap",
        "replicates": len(draws),
        "seed": BOOTSTRAP_SEED,
        "coordinate": "log2_D_obs",
        "invalid_unbracketed_replicates": invalid,
        "status": "VALID" if invalid == 0 else "NOT_EVALUABLE",
        "ci_95_log2_D_obs": {"low": low_log, "high": high_log},
        "ci_95_D_obs": {
            "low": 2.0 ** low_log if low_log is not None else None,
            "high": 2.0 ** high_log if high_log is not None else None,
        },
    }


def ranks(values: list[float]) -> list[float]:
    result = [0.0] * len(values)
    order = sorted(range(len(values)), key=lambda index: values[index])
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        midrank = ((start + 1) + end) / 2.0
        for offset in range(start, end):
            result[order[offset]] = midrank
        start = end
    return result


def spearman(left: list[float], right: list[float]) -> float:
    x = ranks(left)
    y = ranks(right)
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denominator = math.sqrt(
        sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)
    )
    if denominator == 0:
        raise AnalysisError("Spearman rank variance is zero")
    return numerator / denominator


def fragment_balanced_statistic(
    paths: list[tuple[int, Path]],
    extract: Callable[[dict], float | None],
    label: str,
) -> dict:
    """Registered aggregation: mean within fragment, then mean across the four
    fragments, then mean across the five primary seeds. Any per-seed failure is
    captured as an error string and yields estimate=None (fail-closed)."""
    seed_means = []
    per_seed: dict[str, object] = {}
    errors: list[str] = []
    for seed, path in paths:
        try:
            rows = read_jsonl(path)
            fragments: dict[int, list[float]] = defaultdict(list)
            for row in rows:
                if row.get("schema") != TELEMETRY_SCHEMA:
                    raise AnalysisError(f"{path}: telemetry schema mismatch")
                value = extract(row)
                if value is None:
                    continue
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise AnalysisError(f"{path}: nonfinite {label}")
                fragments[int(row["fragment"])].append(float(value))
            if set(fragments) != {0, 1, 2, 3} or any(
                not values for values in fragments.values()
            ):
                raise AnalysisError(f"{path}: {label} undefined for a fragment")
            fragment_means = {
                str(fragment): sum(values) / len(values)
                for fragment, values in sorted(fragments.items())
            }
            seed_mean = sum(fragment_means.values()) / 4.0
            per_seed[str(seed)] = {
                "fragment_means": fragment_means,
                "fragment_balanced_mean": seed_mean,
            }
            seed_means.append(seed_mean)
        except AnalysisError as exc:
            errors.append(str(exc))
            per_seed[str(seed)] = {"error": str(exc)}
    complete = len(seed_means) == len(paths) and len(paths) == len(PRIMARY_SEEDS)
    return {
        "estimate": sum(seed_means) / len(seed_means) if complete else None,
        "aggregation": "mean_within_fragment_then_four_fragment_mean_then_five_seed_mean",
        "statistic": label,
        "per_seed": per_seed,
        "errors": errors,
    }


def extract_lag(lag: int) -> Callable[[dict], float | None]:
    def extract(row: dict) -> float | None:
        return row.get("autocorrelation", {}).get(f"lag_{lag}")
    return extract


def extract_cross_worker_mean_cosine(row: dict) -> float | None:
    cross = row.get("cross_worker")
    if not isinstance(cross, dict):
        raise AnalysisError("telemetry row lacks cross_worker block")
    if cross.get("worker_count") != 4:
        raise AnalysisError("cross_worker worker_count != 4")
    if cross.get("pair_count") != 6 or cross.get("defined_pair_count") != 6:
        raise AnalysisError("cross_worker pairs incomplete")
    return cross.get("mean_cosine")


def rho_eff_from_cosine(c: float | None) -> dict:
    """Registered v2 estimator: rho_eff = 1 - 4c/(1+3c);
    A = 1 + 0.9/(1 - 0.9*rho_eff); D_pred = 10/A. All None-guarded."""
    record: dict[str, object] = {
        "formula": "rho_eff = 1 - 4c/(1+3c)",
        "c_fragment_balanced_cross_worker_cosine": c,
        "rho_eff": None,
        "A_mu0p9_rho_eff": None,
        "D_pred": None,
        "undefined_reason": None,
    }
    if c is None:
        record["undefined_reason"] = "cross-worker cosine undefined"
        return record
    if not math.isfinite(c) or (1.0 + 3.0 * c) <= 0.0:
        record["undefined_reason"] = "1+3c nonpositive or nonfinite"
        return record
    rho_eff = 1.0 - (4.0 * c) / (1.0 + 3.0 * c)
    record["rho_eff"] = rho_eff
    denominator = 1.0 - MU_HIGH * rho_eff
    if denominator <= 0.0:
        record["undefined_reason"] = "1-0.9*rho_eff nonpositive"
        return record
    amplification = 1.0 + MU_HIGH / denominator
    record["A_mu0p9_rho_eff"] = amplification
    record["D_pred"] = 10.0 / amplification
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    if manifest.get("stage") != "E1V2" or len(manifest.get("cells", [])) != EXPECTED_CELLS:
        raise SystemExit("not a complete E1' (v2) launch manifest")

    losses: dict[tuple[int, float, int, float], float] = {}
    cell_records = []
    primary_invalid = []
    topup_invalid = []
    cell_by_key = {}
    for cell in manifest["cells"]:
        node = cell["initial_assignment"]["node"]
        root = args.results / node / cell["cell_id"] / "attempt-1"
        evidence_path = root / "evidence.json"
        if not evidence_path.is_file():
            target = primary_invalid if cell["seed_role"] == "PRIMARY" else topup_invalid
            target.append({"cell_id": cell["cell_id"], "status": "MISSING_EVIDENCE"})
            continue
        evidence = read_json(evidence_path)
        if evidence.get("status") != "COMPLETED":
            target = primary_invalid if cell["seed_role"] == "PRIMARY" else topup_invalid
            target.append({"cell_id": cell["cell_id"], "status": evidence.get("status")})
            continue
        try:
            result_rows = read_jsonl(root / "report" / "results.jsonl")
        except AnalysisError:
            result_rows = []
        if len(result_rows) != 1:
            target = primary_invalid if cell["seed_role"] == "PRIMARY" else topup_invalid
            target.append({"cell_id": cell["cell_id"], "status": "RESULT_ROW_MISMATCH"})
            continue
        loss = result_rows[0].get("eval_loss")
        if not isinstance(loss, (int, float)) or not math.isfinite(loss):
            target = primary_invalid if cell["seed_role"] == "PRIMARY" else topup_invalid
            target.append({"cell_id": cell["cell_id"], "status": "NONFINITE"})
            continue
        key = (cell["h"], float(cell["mu"]), cell["seed"], float(cell["eta"]))
        if key in losses:
            raise SystemExit(f"duplicate scientific cell key: {key}")
        losses[key] = float(loss)
        cell_by_key[key] = (cell, root)
        artifacts = evidence.get("observed_artifacts", {})
        cell_records.append({
            "cell_id": cell["cell_id"],
            "evidence_sha256": sha256_file(evidence_path),
            "telemetry_sha256": artifacts.get("rho_telemetry", {}).get("sha256"),
            "checkpoint_sha256": artifacts.get("checkpoint", {}).get("sha256"),
            "eval_loss": float(loss),
        })

    primary_complete = (
        not primary_invalid
        and sum(1 for key in losses if key[2] in PRIMARY_SEEDS) == EXPECTED_PRIMARY
    )

    rng = random.Random(BOOTSTRAP_SEED)
    primary_draws = [
        [rng.randrange(len(PRIMARY_SEEDS)) for _ in PRIMARY_SEEDS]
        for _ in range(BOOTSTRAP_REPLICATES)
    ]
    topup_draws = [
        [rng.randrange(len(PRIMARY_SEEDS + TOPUP_SEEDS)) for _ in (PRIMARY_SEEDS + TOPUP_SEEDS)]
        for _ in range(BOOTSTRAP_REPLICATES)
    ]

    coordinates = [(h, mu) for h in HORIZONS for mu in (0.0, MU_HIGH)]
    curve_results = {}
    all_curves_interior = True
    for h, mu in coordinates:
        fit = safe_curve_fit(losses, h, mu, PRIMARY_SEEDS)
        curve_id = f"H{h}_mu{repr(mu) if mu else '0'}"
        if fit["status"] == "INVALID_INPUT":
            bootstrap = {"status": "NOT_EVALUABLE", "reason": "invalid or incomplete curve input"}
        else:
            bootstrap = bootstrap_curve(losses, h, mu, PRIMARY_SEEDS, primary_draws)
        curve_results[curve_id] = {**fit, "bootstrap": bootstrap}
        all_curves_interior = all_curves_interior and bool(fit["interior"])

    # rho_eff per H from mu=0 telemetry ONLY. Registered v2 correction: this loop
    # is unconditional on the mu=.9 curve state (v1 defect coupled them).
    ratios = {}
    rho = {}
    d_pred_by_h: dict[int, float | None] = {}
    d_obs_by_h: dict[int, float | None] = {}
    for h in HORIZONS:
        fit0 = curve_results[f"H{h}_mu0"]
        fit9 = curve_results[f"H{h}_mu0.9"]

        rho_record: dict[str, object] = {
            "source_mu": 0,
            "independent_of_mu0p9_curve_validity": True,
            "pooled_eta_star_mu0": fit0["eta_star"],
            "selected_grid_eta": None,
            "nearest_rule": "minimum absolute numeric eta distance; lower eta wins tie",
            "raw_lags_diagnostic_only": None,
            "rho_eff": None,
        }
        if fit0["interior"] and fit0["eta_star"] is not None and fit0.get("etas"):
            selected_eta = min(
                fit0["etas"], key=lambda eta: (abs(eta - fit0["eta_star"]), eta)
            )
            rho_record["selected_grid_eta"] = selected_eta
            paths = []
            missing = []
            for seed in PRIMARY_SEEDS:
                entry = cell_by_key.get((h, 0.0, seed, float(selected_eta)))
                if entry is None:
                    missing.append(seed)
                    continue
                _, root = entry
                paths.append((seed, root / "work" / "m4" / "rho-telemetry.jsonl"))
            if missing:
                rho_record["error"] = f"missing completed mu0 telemetry cells for seeds {missing}"
            else:
                lag_results = {
                    str(lag): fragment_balanced_statistic(
                        paths, extract_lag(lag), f"lag_{lag}_projected_cosine"
                    )
                    for lag in range(1, 5)
                }
                rho_record["raw_lags_diagnostic_only"] = lag_results
                cross = fragment_balanced_statistic(
                    paths, extract_cross_worker_mean_cosine, "cross_worker_mean_cosine"
                )
                rho_record["cross_worker"] = cross
                rho_record["rho_eff"] = rho_eff_from_cosine(cross["estimate"])
        else:
            rho_record["error"] = "mu0 eta_star not interior; nearest grid point undefined"
        rho[str(h)] = rho_record
        rho_eff_block = rho_record.get("rho_eff")
        d_pred_by_h[h] = (
            rho_eff_block.get("D_pred") if isinstance(rho_eff_block, dict) else None
        )

        if (
            fit0["interior"] and fit9["interior"]
            and fit0["eta_star"] is not None and fit9["eta_star"] is not None
        ):
            d_obs = (fit9["eta_star"] / fit0["eta_star"]) / (1.0 - MU_HIGH)
            ratio_bootstrap = bootstrap_ratio(losses, h, primary_draws)
            ratios[str(h)] = {"D_obs": d_obs, "bootstrap": ratio_bootstrap}
            d_obs_by_h[h] = d_obs
        else:
            ratios[str(h)] = {
                "D_obs": None,
                "bootstrap": {"status": "NOT_EVALUABLE", "reason": "required eta_star not interior"},
            }
            d_obs_by_h[h] = None

    high_h_ci_excludes_one = {}
    for h in (256, 512):
        ci = ratios.get(str(h), {}).get("bootstrap", {}).get("ci_95_D_obs", {})
        low = ci.get("low") if isinstance(ci, dict) else None
        high = ci.get("high") if isinstance(ci, dict) else None
        high_h_ci_excludes_one[str(h)] = (
            isinstance(low, (int, float)) and isinstance(high, (int, float))
            and (high < 1.0 or low > 1.0)
        )
    rho_eff_defined = all(d_pred_by_h[h] is not None for h in HORIZONS)
    d_obs_defined = all(d_obs_by_h[h] is not None for h in HORIZONS)
    high_h_ratio_bootstraps_valid = all(
        ratios.get(str(h), {}).get("bootstrap", {}).get("status") == "VALID"
        for h in (256, 512)
    )
    spearman_value = None
    if rho_eff_defined and d_obs_defined:
        spearman_value = spearman(
            [d_pred_by_h[h] for h in HORIZONS],
            [d_obs_by_h[h] for h in HORIZONS],
        )
    # Descriptive only, never gated (registered v2 change):
    descriptive_ratios = {
        str(h): (
            d_pred_by_h[h] / d_obs_by_h[h]
            if d_pred_by_h[h] is not None and d_obs_by_h[h] is not None
            and d_obs_by_h[h] != 0 else None
        )
        for h in HORIZONS
    }

    evaluable = (
        primary_complete and all_curves_interior
        and high_h_ratio_bootstraps_valid and rho_eff_defined
    )
    scientific_requirements = {
        "H256_D_obs_CI_excludes_1": high_h_ci_excludes_one.get("256", False),
        "H512_D_obs_CI_excludes_1": high_h_ci_excludes_one.get("512", False),
        "spearman_D_pred_D_obs_ge_0p8": (
            spearman_value is not None and spearman_value >= 0.8
        ),
        "all_E1v2_primary_work_valid": primary_complete,
        "all_required_eta_star_interior": all_curves_interior,
        "rho_eff_defined_at_all_four_H": rho_eff_defined,
    }
    verdict = "NOT_EVALUABLE" if not evaluable else (
        "PASS" if all(scientific_requirements.values()) else "FAIL"
    )

    topup_report = {}
    if not topup_invalid:
        all_eight = PRIMARY_SEEDS + TOPUP_SEEDS
        for h in CONTESTED_MU09_HORIZONS:
            fit9 = safe_curve_fit(losses, h, MU_HIGH, all_eight)
            mu0_star = curve_results[f"H{h}_mu0"]["eta_star"]
            entry: dict[str, object] = {
                "role": "secondary_robustness_only_cannot_change_G1",
                "mu0p9_all_eight_eta_star": fit9.get("eta_star"),
                "mu0_denominator": "five-primary-seed eta_star because mu0 top-up cells are not registered",
                "D_obs_secondary": None,
                "D_obs_secondary_status": "NOT_EVALUABLE_UNBRACKETED",
                "bootstrap": None,
            }
            if fit9.get("eta_star") is not None and mu0_star is not None:
                entry["D_obs_secondary"] = (fit9["eta_star"] / mu0_star) / (1.0 - MU_HIGH)
                entry["D_obs_secondary_status"] = "VALID"
            if fit9["status"] != "INVALID_INPUT" and curve_results[f"H{h}_mu0"]["status"] not in (
                "INVALID_INPUT",
            ):
                d_samples = []
                invalid = 0
                for draw8, draw5 in zip(topup_draws, primary_draws):
                    try:
                        boot9 = curve_fit(losses, h, MU_HIGH, all_eight, draw8)
                        boot0 = curve_fit(losses, h, 0.0, PRIMARY_SEEDS, draw5)
                    except AnalysisError:
                        invalid += 1
                        continue
                    if not boot9["interior"] or not boot0["interior"]:
                        invalid += 1
                        continue
                    d_samples.append(
                        (boot9["eta_star"] / boot0["eta_star"]) / (1.0 - MU_HIGH)
                    )
                entry["bootstrap"] = {
                    "method": "independent eight-seed mu0p9 and five-seed mu0 curve bootstrap",
                    "replicates": BOOTSTRAP_REPLICATES,
                    "invalid_unbracketed_replicates": invalid,
                    "status": "VALID" if invalid == 0 else "NOT_EVALUABLE",
                    "ci_95_D_obs_secondary": {
                        "low": quantile(d_samples, 0.025) if d_samples else None,
                        "high": quantile(d_samples, 0.975) if d_samples else None,
                    },
                }
            topup_report[str(h)] = entry

    selection_payload = {
        "schema": "yeto_outer_mup_e1v2_selection_v1",
        "status": "SEALED" if verdict in ("PASS", "FAIL") else "INCOMPLETE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "launch_manifest_sha256": sha256_file(args.manifest),
        "contract_json_sha256": manifest["contract"]["json_sha256"],
        "contract_md_sha256": manifest["contract"]["md_sha256"],
        "source_git_commit": manifest["source"]["git_commit"],
        "cell_evidence_registry_sha256": canonical_sha256(
            sorted(cell_records, key=lambda item: item["cell_id"])
        ),
        "cell_evidence": sorted(cell_records, key=lambda item: item["cell_id"]),
        "eta_curves": curve_results,
        "rho": rho,
        "G1": {
            "gate_id": "G1_v2_confirm",
            "verdict": verdict,
            "scientific_requirements": scientific_requirements,
            "D_obs": ratios,
            "D_pred": {str(h): d_pred_by_h[h] for h in HORIZONS},
            "D_pred_over_D_obs_descriptive_only": descriptive_ratios,
            "spearman_D_pred_D_obs": spearman_value,
            "primary_invalid": primary_invalid,
            "evaluable": evaluable,
        },
        "topup_report": topup_report,
        "topup_invalid": topup_invalid,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(selection_payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    digest = sha256_file(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n"
    )
    print(json.dumps({
        "output": str(args.output), "sha256": digest, "G1": verdict,
        "spearman": spearman_value, "requirements": scientific_requirements,
    }, sort_keys=True))
    return 0 if verdict != "NOT_EVALUABLE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
