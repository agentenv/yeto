#!/usr/bin/env python3
"""Apply the registered outer-muP v3 (finite-horizon) analysis and emit the readout.

Registered under experiment-specs/outer-mup-v3-finitehorizon-prereg.{json,md}.
Frozen before launch; its raw-file SHA-256 is recorded in the v3 contract JSON.

Structure copied from the frozen v2 analyzer (scripts/analyze_e1v2.py):
per-curve quadratic fits in x = log2(eta) on the exact registered eta values,
paired five-seed 10,000-replicate bootstrap with RNG seed 20260724, and
None-guards everywhere a fit can be UNBRACKETED or an input can be missing:
the analyzer always emits a complete readout with NOT_EVALUABLE verdicts
instead of crashing.

v3 arms and gates (this file computes them; the supervisor RUNS them):
  ARM T (raw Nesterov T-scan)  -> G3a: D_obs(T) monotone decreasing in T via
       successive-pair paired bootstrap CIs, plus within-CI replication of the
       pilot D_obs(T=5) = 2.441338 (v2 H512 primary point estimate).
  ARM B (--outer-bias-correction interventions) -> G3b: bias-corrected optima
       horizon-invariant: |log2 D_corrected(T)| <= 0.15 at >= 4 of 5 T values.
  ARM S (M=1 SNOO deflation)   -> G3c: descriptive only; equal-tuning-budget
       best-grid-point losses for (a) plain-AdamW-equivalent outer SGD,
       (b) SNOO-style outer Nesterov mu.9, (c) eta_eff-matched mu0 control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path

SEEDS = (301, 311, 313, 317, 331)
RESERVED_SEED = 307
S_GRID = (1024, 2560, 5120, 10240, 20480)
T_BY_S = {1024: 2, 2560: 5, 5120: 10, 10240: 20, 20480: 40}
MU_HIGH = 0.9
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260724
EXPECTED_CELLS = 460
EXPECTED_BY_ARM = {"T": 200, "B": 200, "S": 60}
PILOT_T5_D_OBS = 2.441338
G3B_BAND_LOG2 = 0.15
G3B_MIN_T_IN_BAND = 4
SNOO_MARGIN = 0.01
SNOO_SUBARMS = ("a", "b", "c")


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
    """Registered fit (v2-identical): loss = a*x^2 + b*x + c in x = log2(eta)
    on the exact registered eta values, so the 2x-spaced arm-T S10240 mu.9
    ladder is fitted exactly as registered."""
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


def quantile(values: list[float], probability: float) -> float:
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
    losses: dict[tuple[str, object, float, int, float], float],
    arm: str,
    coord: object,
    mu: float,
    sampled_indices: list[int] | None = None,
) -> dict:
    etas = sorted({
        key[4] for key in losses
        if key[0] == arm and key[1] == coord and key[2] == mu
    })
    if len(etas) != 4:
        raise AnalysisError(f"{arm}/{coord}/mu{mu}: expected four etas, found {etas}")
    selected_seeds = (
        [SEEDS[index] for index in sampled_indices]
        if sampled_indices is not None else list(SEEDS)
    )
    means = []
    for eta in etas:
        values = []
        for seed in selected_seeds:
            key = (arm, coord, mu, seed, eta)
            if key not in losses:
                raise AnalysisError(f"{arm}/{coord}/mu{mu}: missing seed cell {key}")
            values.append(losses[key])
        means.append(sum(values) / len(values))
    fit = fit_quadratic(etas, means)
    fit.update({"etas": etas, "seed_mean_losses": means})
    return fit


def safe_curve_fit(
    losses: dict[tuple[str, object, float, int, float], float],
    arm: str,
    coord: object,
    mu: float,
) -> dict:
    """None-guarded pooled fit (v2-identical guard): invalid/missing input
    yields an inert record instead of an exception."""
    try:
        return curve_fit(losses, arm, coord, mu)
    except AnalysisError as exc:
        return {
            "a": None, "b": None, "c": None,
            "vertex_log2_eta": None, "eta_star": None,
            "interior": False, "status": "INVALID_INPUT",
            "etas": sorted({
                key[4] for key in losses
                if key[0] == arm and key[1] == coord and key[2] == mu
            }),
            "seed_mean_losses": None,
            "error": str(exc),
        }


def bootstrap_curve(
    losses: dict[tuple[str, object, float, int, float], float],
    arm: str,
    coord: object,
    mu: float,
    draws: list[list[int]],
) -> dict:
    samples = []
    invalid = 0
    for draw in draws:
        try:
            fit = curve_fit(losses, arm, coord, mu, draw)
        except AnalysisError:
            invalid += 1
            continue
        if not fit["interior"]:
            invalid += 1
        else:
            samples.append(math.log2(fit["eta_star"]))
    return {
        "method": "paired_nonparametric_training_seed_curve_bootstrap",
        "replicates": len(draws),
        "seed": BOOTSTRAP_SEED,
        "coordinate": "log2_eta",
        "invalid_unbracketed_replicates": invalid,
        "status": "VALID" if invalid == 0 else "NOT_EVALUABLE",
        "ci_95_log2_eta": {
            "low": quantile(samples, 0.025) if samples else None,
            "high": quantile(samples, 0.975) if samples else None,
        },
        "ci_95_eta": {
            "low": 2.0 ** quantile(samples, 0.025) if samples else None,
            "high": 2.0 ** quantile(samples, 0.975) if samples else None,
        },
    }


def draw_log2_d(
    losses: dict[tuple[str, object, float, int, float], float],
    arm: str,
    coord: object,
    draw: list[int],
) -> float | None:
    """One paired-draw log2 D value for a (mu0.9, mu0) curve pair, or None if
    either replicate fit is unbracketed/invalid."""
    try:
        fit0 = curve_fit(losses, arm, coord, 0.0, draw)
        fit9 = curve_fit(losses, arm, coord, MU_HIGH, draw)
    except AnalysisError:
        return None
    if not fit0["interior"] or not fit9["interior"]:
        return None
    return math.log2((fit9["eta_star"] / fit0["eta_star"]) / (1.0 - MU_HIGH))


def bootstrap_ratio(
    losses: dict[tuple[str, object, float, int, float], float],
    arm: str,
    coord: object,
    draws: list[list[int]],
) -> dict:
    log2_d = []
    invalid = 0
    for draw in draws:
        value = draw_log2_d(losses, arm, coord, draw)
        if value is None:
            invalid += 1
        else:
            log2_d.append(value)
    low_log = quantile(log2_d, 0.025) if log2_d else None
    high_log = quantile(log2_d, 0.975) if log2_d else None
    return {
        "method": "paired_nonparametric_training_seed_curve_bootstrap",
        "replicates": len(draws),
        "seed": BOOTSTRAP_SEED,
        "coordinate": "log2_D",
        "invalid_unbracketed_replicates": invalid,
        "status": "VALID" if invalid == 0 else "NOT_EVALUABLE",
        "ci_95_log2_D": {"low": low_log, "high": high_log},
        "ci_95_D": {
            "low": 2.0 ** low_log if low_log is not None else None,
            "high": 2.0 ** high_log if high_log is not None else None,
        },
    }


def bootstrap_pair_gap(
    losses: dict[tuple[str, object, float, int, float], float],
    arm: str,
    coord_small_t: object,
    coord_large_t: object,
    draws: list[list[int]],
) -> dict:
    """Paired bootstrap of log2 D(T_small) - log2 D(T_large) with the SAME seed
    draw applied to all four curves (full pairing across the T pair)."""
    gaps = []
    invalid = 0
    for draw in draws:
        left = draw_log2_d(losses, arm, coord_small_t, draw)
        right = draw_log2_d(losses, arm, coord_large_t, draw)
        if left is None or right is None:
            invalid += 1
        else:
            gaps.append(left - right)
    return {
        "method": "paired_nonparametric_training_seed_curve_bootstrap",
        "replicates": len(draws),
        "seed": BOOTSTRAP_SEED,
        "coordinate": "log2_D_small_T_minus_log2_D_large_T",
        "invalid_unbracketed_replicates": invalid,
        "status": "VALID" if invalid == 0 else "NOT_EVALUABLE",
        "ci_95": {
            "low": quantile(gaps, 0.025) if gaps else None,
            "high": quantile(gaps, 0.975) if gaps else None,
        },
    }


def snoo_tuned_loss(
    losses: dict[tuple[str, object, float, int, float], float],
    sub_arm: str,
    mu: float,
    sampled_indices: list[int] | None = None,
) -> float:
    """Registered equal-tuning-budget estimand: the minimum over the four
    registered etas of the (sampled) seed-mean loss."""
    etas = sorted({
        key[4] for key in losses
        if key[0] == "S" and key[1] == sub_arm and key[2] == mu
    })
    if len(etas) != 4:
        raise AnalysisError(f"S/{sub_arm}: expected four etas, found {etas}")
    selected_seeds = (
        [SEEDS[index] for index in sampled_indices]
        if sampled_indices is not None else list(SEEDS)
    )
    means = []
    for eta in etas:
        values = []
        for seed in selected_seeds:
            key = ("S", sub_arm, mu, seed, eta)
            if key not in losses:
                raise AnalysisError(f"S/{sub_arm}: missing seed cell {key}")
            values.append(losses[key])
        means.append(sum(values) / len(values))
    return min(means)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    if manifest.get("stage") != "V3FH" or len(manifest.get("cells", [])) != EXPECTED_CELLS:
        raise SystemExit("not a complete v3 finite-horizon launch manifest")

    losses: dict[tuple[str, object, float, int, float], float] = {}
    cell_records = []
    invalid_cells = []
    for cell in manifest["cells"]:
        node = cell["initial_assignment"]["node"]
        root = args.results / node / cell["cell_id"] / "attempt-1"
        evidence_path = root / "evidence.json"
        if not evidence_path.is_file():
            invalid_cells.append({"cell_id": cell["cell_id"], "status": "MISSING_EVIDENCE"})
            continue
        evidence = read_json(evidence_path)
        if evidence.get("status") != "COMPLETED":
            invalid_cells.append({"cell_id": cell["cell_id"], "status": evidence.get("status")})
            continue
        try:
            result_rows = read_jsonl(root / "report" / "results.jsonl")
        except AnalysisError:
            result_rows = []
        if len(result_rows) != 1:
            invalid_cells.append({"cell_id": cell["cell_id"], "status": "RESULT_ROW_MISMATCH"})
            continue
        loss = result_rows[0].get("eval_loss")
        if not isinstance(loss, (int, float)) or not math.isfinite(loss):
            invalid_cells.append({"cell_id": cell["cell_id"], "status": "NONFINITE"})
            continue
        arm = cell["arm"]
        coord = cell["sub_arm"] if arm == "S" else int(cell["s"])
        key = (arm, coord, float(cell["mu"]), int(cell["seed"]), float(cell["eta"]))
        if key in losses:
            raise SystemExit(f"duplicate scientific cell key: {key}")
        losses[key] = float(loss)
        artifacts = evidence.get("observed_artifacts", {})
        cell_records.append({
            "cell_id": cell["cell_id"],
            "evidence_sha256": sha256_file(evidence_path),
            "telemetry_sha256": artifacts.get("rho_telemetry", {}).get("sha256"),
            "checkpoint_sha256": artifacts.get("checkpoint", {}).get("sha256"),
            "eval_loss": float(loss),
        })

    completed_by_arm = {
        arm: sum(1 for cell in manifest["cells"] if cell["arm"] == arm and any(
            record["cell_id"] == cell["cell_id"] for record in cell_records
        ))
        for arm in EXPECTED_BY_ARM
    }
    arm_complete = {
        arm: completed_by_arm[arm] == EXPECTED_BY_ARM[arm]
        for arm in EXPECTED_BY_ARM
    }

    rng = random.Random(BOOTSTRAP_SEED)
    draws = [
        [rng.randrange(len(SEEDS)) for _ in SEEDS]
        for _ in range(BOOTSTRAP_REPLICATES)
    ]

    # ---- ARM T and ARM B curve fits ------------------------------------
    curves: dict[str, dict] = {}
    d_records: dict[str, dict] = {}
    for arm in ("T", "B"):
        for s in S_GRID:
            for mu in (0.0, MU_HIGH):
                curve_id = f"{arm}_S{s}_mu{repr(mu) if mu else '0'}"
                fit = safe_curve_fit(losses, arm, s, mu)
                if fit["status"] == "INVALID_INPUT":
                    bootstrap = {
                        "status": "NOT_EVALUABLE",
                        "reason": "invalid or incomplete curve input",
                    }
                else:
                    bootstrap = bootstrap_curve(losses, arm, s, mu, draws)
                curves[curve_id] = {**fit, "bootstrap": bootstrap}
        for s in S_GRID:
            fit0 = curves[f"{arm}_S{s}_mu0"]
            fit9 = curves[f"{arm}_S{s}_mu0.9"]
            record: dict[str, object] = {
                "T": T_BY_S[s],
                "D": None,
                "log2_D": None,
                "bootstrap": {
                    "status": "NOT_EVALUABLE",
                    "reason": "required eta_star not interior",
                },
            }
            if (
                fit0["interior"] and fit9["interior"]
                and fit0["eta_star"] is not None and fit9["eta_star"] is not None
            ):
                d_value = (fit9["eta_star"] / fit0["eta_star"]) / (1.0 - MU_HIGH)
                record["D"] = d_value
                record["log2_D"] = math.log2(d_value)
                record["bootstrap"] = bootstrap_ratio(losses, arm, s, draws)
            d_records[f"{arm}_S{s}"] = record

    # ---- G3a: raw T-scan monotone + pilot replication -------------------
    arm_t_interior = all(
        curves[f"T_S{s}_mu{tag}"]["interior"]
        for s in S_GRID for tag in ("0", "0.9")
    )
    pair_gaps = {}
    monotone_flags = {}
    successive = list(zip(S_GRID, S_GRID[1:]))
    for s_small, s_large in successive:
        label = f"T{T_BY_S[s_small]}_vs_T{T_BY_S[s_large]}"
        gap = bootstrap_pair_gap(losses, "T", s_small, s_large, draws)
        pair_gaps[label] = gap
        low = gap["ci_95"].get("low")
        monotone_flags[label] = isinstance(low, (int, float)) and low > 0.0
    t5_ci = d_records["T_S2560"]["bootstrap"].get("ci_95_D", {})
    t5_low = t5_ci.get("low") if isinstance(t5_ci, dict) else None
    t5_high = t5_ci.get("high") if isinstance(t5_ci, dict) else None
    t5_replicates_pilot = (
        isinstance(t5_low, (int, float)) and isinstance(t5_high, (int, float))
        and t5_low <= PILOT_T5_D_OBS <= t5_high
    )
    pair_bootstraps_valid = all(
        gap["status"] == "VALID" for gap in pair_gaps.values()
    )
    t5_bootstrap_valid = d_records["T_S2560"]["bootstrap"].get("status") == "VALID"
    g3a_evaluable = (
        arm_complete["T"] and arm_t_interior
        and pair_bootstraps_valid and t5_bootstrap_valid
    )
    g3a_requirements = {
        "all_arm_T_work_valid": arm_complete["T"],
        "all_arm_T_eta_star_interior": arm_t_interior,
        "successive_pair_CIs_exclude_zero_from_below": monotone_flags,
        "all_pairs_monotone_decreasing": all(monotone_flags.values()),
        "T5_CI_contains_pilot_2p441338": t5_replicates_pilot,
    }
    g3a_verdict = "NOT_EVALUABLE" if not g3a_evaluable else (
        "PASS"
        if all(monotone_flags.values()) and t5_replicates_pilot
        else "FAIL"
    )

    # ---- G3b: bias-corrected horizon invariance --------------------------
    arm_b_interior = all(
        curves[f"B_S{s}_mu{tag}"]["interior"]
        for s in S_GRID for tag in ("0", "0.9")
    )
    in_band = {}
    for s in S_GRID:
        log2_d = d_records[f"B_S{s}"]["log2_D"]
        in_band[f"T{T_BY_S[s]}"] = (
            isinstance(log2_d, (int, float)) and abs(log2_d) <= G3B_BAND_LOG2
        )
    band_count = sum(1 for value in in_band.values() if value)
    raw_d_values = [
        d_records[f"T_S{s}"]["D"] for s in S_GRID
        if isinstance(d_records[f"T_S{s}"]["D"], (int, float))
    ]
    raw_span = (
        max(raw_d_values) / min(raw_d_values)
        if len(raw_d_values) == len(S_GRID) and min(raw_d_values) > 0 else None
    )
    g3b_evaluable = arm_complete["B"] and arm_b_interior
    g3b_requirements = {
        "all_arm_B_work_valid": arm_complete["B"],
        "all_arm_B_eta_star_interior": arm_b_interior,
        "abs_log2_D_corrected_le_0p15_by_T": in_band,
        "count_in_band": band_count,
        "min_required_in_band": G3B_MIN_T_IN_BAND,
    }
    g3b_verdict = "NOT_EVALUABLE" if not g3b_evaluable else (
        "PASS" if band_count >= G3B_MIN_T_IN_BAND else "FAIL"
    )

    # ---- G3c: SNOO deflation (descriptive only) --------------------------
    snoo: dict[str, object] = {
        "role": "descriptive_only_never_stops_program",
        "estimand": "min over the four registered etas of the five-seed mean loss",
        "margin_loss": SNOO_MARGIN,
        "tuned_loss": {},
        "eta_curves": {},
        "deltas": {},
        "verdict": "NOT_EVALUABLE",
    }
    sub_mu = {"a": 0.0, "b": MU_HIGH, "c": 0.0}
    tuned: dict[str, float] = {}
    try:
        for sub in SNOO_SUBARMS:
            tuned[sub] = snoo_tuned_loss(losses, sub, sub_mu[sub])
            snoo["tuned_loss"][sub] = tuned[sub]
            snoo["eta_curves"][sub] = safe_curve_fit(losses, "S", sub, sub_mu[sub])
        deltas = {
            "b_minus_a": tuned["b"] - tuned["a"],
            "c_minus_a": tuned["c"] - tuned["a"],
            "b_minus_c": tuned["b"] - tuned["c"],
        }
        delta_samples: dict[str, list[float]] = {name: [] for name in deltas}
        invalid_draws = 0
        for draw in draws:
            try:
                sample = {
                    sub: snoo_tuned_loss(losses, sub, sub_mu[sub], draw)
                    for sub in SNOO_SUBARMS
                }
            except AnalysisError:
                invalid_draws += 1
                continue
            delta_samples["b_minus_a"].append(sample["b"] - sample["a"])
            delta_samples["c_minus_a"].append(sample["c"] - sample["a"])
            delta_samples["b_minus_c"].append(sample["b"] - sample["c"])
        for name, point in deltas.items():
            samples = delta_samples[name]
            snoo["deltas"][name] = {
                "point": point,
                "ci_95": {
                    "low": quantile(samples, 0.025) if samples else None,
                    "high": quantile(samples, 0.975) if samples else None,
                },
                "invalid_replicates": invalid_draws,
            }
        ba = snoo["deltas"]["b_minus_a"]["ci_95"]
        bc = snoo["deltas"]["b_minus_c"]["ci_95"]
        if arm_complete["S"] and invalid_draws == 0:
            if ba["high"] is not None and ba["high"] < 0.0:
                if bc["high"] is not None and bc["high"] < 0.0:
                    snoo["verdict"] = "SNOO_GAIN_SURVIVES_TUNING"
                else:
                    snoo["verdict"] = "SNOO_GAIN_DEFLATED"
            elif ba["low"] is not None and ba["low"] > 0.0:
                snoo["verdict"] = "SNOO_REVERSED"
            else:
                snoo["verdict"] = "SNOO_NO_GAIN"
    except AnalysisError as exc:
        snoo["error"] = str(exc)

    readout = {
        "schema": "yeto_outer_mup_v3_readout_v1",
        "status": "SEALED" if (
            g3a_verdict in ("PASS", "FAIL") and g3b_verdict in ("PASS", "FAIL")
        ) else "INCOMPLETE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "launch_manifest_sha256": sha256_file(args.manifest),
        "contract_json_sha256": manifest["contract"]["json_sha256"],
        "contract_md_sha256": manifest["contract"]["md_sha256"],
        "source_git_commit": manifest["source"]["git_commit"],
        "cell_evidence_registry_sha256": canonical_sha256(
            sorted(cell_records, key=lambda item: item["cell_id"])
        ),
        "cell_evidence": sorted(cell_records, key=lambda item: item["cell_id"]),
        "invalid_cells": invalid_cells,
        "completed_by_arm": completed_by_arm,
        "eta_curves": curves,
        "D_by_arm_and_S": d_records,
        "G3a": {
            "gate_id": "G3a_v3_finite_horizon_t_scan",
            "verdict": g3a_verdict,
            "evaluable": g3a_evaluable,
            "requirements": g3a_requirements,
            "successive_pair_gaps": pair_gaps,
            "pilot_replication_constant": PILOT_T5_D_OBS,
        },
        "G3b": {
            "gate_id": "G3b_v3_bias_correction_invariance",
            "verdict": g3b_verdict,
            "evaluable": g3b_evaluable,
            "requirements": g3b_requirements,
            "raw_arm_D_span_descriptive": raw_span,
            "registered_prediction": (
                "abs(log2 D_corrected) <= 0.15 at every T while the raw arm's "
                "D spans >= 4x; gated at >= 4 of 5 T values"
            ),
        },
        "G3c": snoo,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(readout, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    digest = sha256_file(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n"
    )
    print(json.dumps({
        "output": str(args.output), "sha256": digest,
        "G3a": g3a_verdict, "G3b": g3b_verdict, "G3c": snoo["verdict"],
    }, sort_keys=True))
    return 0 if (g3a_verdict != "NOT_EVALUABLE" and g3b_verdict != "NOT_EVALUABLE") else 2


if __name__ == "__main__":
    raise SystemExit(main())
