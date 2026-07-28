#!/usr/bin/env python3
"""Validate the Lean scalar phase boundary against the v8 G8 readout.

The primary test implements the exactly-retuned iid criterion proved in
``LeanMechanism/PhaseBoundary.lean``.  That criterion depends only on ``T`` and
``mu`` because the positive noise/curvature scale cancels.  When v8 rho
telemetry is supplied, the script also evaluates the distinct finite steady-
prescription boundary as a sensitivity analysis.  It must not be substituted
for the primary test because G8 independently retuned every curve.

The optional telemetry proxy assumes a worker direction

    g_i = s + epsilon_i,

with independent, zero-mean worker noise.  For each outer event it estimates
``||s||^2`` by the mean pairwise worker dot product and the noise energy of the
four-worker mean by one quarter of the difference between the mean worker
squared norm and that signal estimate.  Pooling numerator and denominator
over events gives a dimensionless proxy for
``sigmaSq / (a^2 * theta0^2)`` without separately estimating curvature.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


COMPARISON_LABELS = ("HELPS", "HURTS", "NEUTRAL", "UNCERTAIN")
MOMENTUM_ARMS = ("raw", "corrected")
BOUNDARY_TOLERANCE = 1e-12


class ValidationError(RuntimeError):
    """Raised when a frozen input or telemetry invariant is not satisfied."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readout", required=True, type=Path)
    parser.add_argument(
        "--telemetry-root",
        type=Path,
        help="root containing recursively discovered rho-telemetry.jsonl files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write JSON here instead of printing it to stdout",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"{path}: expected a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def geometric_prefix(length: int, ratio: float) -> float:
    return math.fsum(ratio**power for power in range(length))


def effective_coefficient(horizon: int, momentum: float) -> float:
    return math.fsum(
        geometric_prefix(update + 1, momentum) for update in range(horizon)
    )


def iid_outer_noise_factor(horizon: int, momentum: float) -> float:
    return math.fsum(
        geometric_prefix(update + 1, momentum * momentum)
        for update in range(horizon)
    )


def exact_boundary(horizon: int, momentum: float) -> dict[str, float | str]:
    coefficient = effective_coefficient(horizon, momentum)
    noise_factor = iid_outer_noise_factor(horizon, momentum)
    advantage = coefficient * coefficient - horizon * noise_factor
    if advantage > BOUNDARY_TOLERANCE:
        prediction = "HELPS"
    elif advantage < -BOUNDARY_TOLERANCE:
        prediction = "HURTS"
    else:
        prediction = "NEUTRAL"
    return {
        "effective_coefficient_c": coefficient,
        "iid_outer_noise_factor_v": noise_factor,
        "iid_coherence_advantage_c2_minus_Tv": advantage,
        "prediction": prediction,
    }


def finite_critical_noise_ratio(horizon: int, momentum: float) -> float:
    coefficient = effective_coefficient(horizon, momentum)
    noise_factor = iid_outer_noise_factor(horizon, momentum)
    bias = 1.0 - coefficient / horizon * (1.0 - momentum)
    saving = horizon - (1.0 - momentum) ** 2 * noise_factor
    if saving <= 0.0:
        raise ValidationError(
            f"nonpositive finite-prescription variance saving at {(horizon, momentum)}"
        )
    return horizon * horizon * bias * bias / saving


def phase_from_noise_ratio(noise_ratio: float, critical_ratio: float) -> str:
    gap = noise_ratio - critical_ratio
    if gap < -BOUNDARY_TOLERANCE:
        return "HURTS"
    if gap > BOUNDARY_TOLERANCE:
        return "HELPS"
    return "NEUTRAL"


def comparison_key(record: dict[str, Any]) -> tuple[int, str, float]:
    return int(record["T"]), str(record["arm"]), float(record["mu"])


def binary_confusion(
    actual_labels: dict[tuple[int, str, float], str],
    predicted_labels: dict[tuple[int, str, float], str],
) -> dict[str, Any]:
    if actual_labels.keys() != predicted_labels.keys():
        raise ValidationError("actual and predicted comparison keys differ")
    true_positive = false_positive = true_negative = false_negative = 0
    for key, actual in actual_labels.items():
        predicted = predicted_labels[key]
        actual_positive = actual == "HURTS"
        predicted_positive = predicted == "HURTS"
        if actual_positive and predicted_positive:
            true_positive += 1
        elif not actual_positive and predicted_positive:
            false_positive += 1
        elif not actual_positive and not predicted_positive:
            true_negative += 1
        else:
            false_negative += 1
    total = len(actual_labels)
    predicted_positive_count = true_positive + false_positive
    actual_positive_count = true_positive + false_negative
    return {
        "positive_class": "HURTS",
        "negative_class": "all primary labels other than HURTS",
        "matrix": {
            "actual_HURTS": {
                "predicted_HURTS": true_positive,
                "predicted_NOT_HURTS": false_negative,
            },
            "actual_NOT_HURTS": {
                "predicted_HURTS": false_positive,
                "predicted_NOT_HURTS": true_negative,
            },
        },
        "TP": true_positive,
        "FP": false_positive,
        "TN": true_negative,
        "FN": false_negative,
        "accuracy": (true_positive + true_negative) / total,
        "precision": (
            true_positive / predicted_positive_count
            if predicted_positive_count
            else None
        ),
        "recall": (
            true_positive / actual_positive_count if actual_positive_count else None
        ),
    }


def multiclass_confusion(
    actual_labels: dict[tuple[int, str, float], str],
    predicted_labels: dict[tuple[int, str, float], str],
) -> dict[str, dict[str, int]]:
    labels = list(COMPARISON_LABELS)
    for value in (*actual_labels.values(), *predicted_labels.values()):
        if value not in labels:
            labels.append(value)
    matrix = {
        actual: {predicted: 0 for predicted in labels} for actual in labels
    }
    for key, actual in actual_labels.items():
        matrix[actual][predicted_labels[key]] += 1
    return matrix


def set_verdict(
    actual_labels: dict[tuple[int, str, float], str],
    predicted_labels: dict[tuple[int, str, float], str],
) -> str:
    observed = {key for key, value in actual_labels.items() if value == "HURTS"}
    predicted = {key for key, value in predicted_labels.items() if value == "HURTS"}
    if predicted == observed:
        return "PREDICTS"
    if predicted & observed:
        return "PARTIAL"
    return "FAILS"


def locate_cell_and_attempt(path: Path) -> tuple[str, int]:
    for parent in path.parents:
        if parent.name.startswith("attempt-"):
            try:
                attempt = int(parent.name.removeprefix("attempt-"))
            except ValueError as exc:
                raise ValidationError(f"malformed attempt directory in {path}") from exc
            return parent.parent.name, attempt
    raise ValidationError(f"cannot infer cell and attempt from {path}")


def finite_number(value: Any, context: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{context}: expected a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValidationError(f"{context}: expected a finite number")
    return result


def summarize_telemetry_file(path: Path) -> dict[str, Any]:
    signal_total = 0.0
    merged_noise_total = 0.0
    first_round_signal_total = 0.0
    first_round_merged_noise_total = 0.0
    event_count = 0
    first_round_event_count = 0
    nonpositive_signal_events = 0
    lag1_values: list[float] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValidationError(f"cannot read {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"{path}:{line_number}: invalid JSON") from exc
        if event.get("schema") != "yeto_rho_telemetry_v1":
            raise ValidationError(f"{path}:{line_number}: unsupported telemetry schema")
        workers = event.get("cross_worker", {}).get("workers", [])
        pairs = event.get("cross_worker", {}).get("pairs", [])
        norms = {
            int(worker["learner_id"]): finite_number(
                worker["l2_norm"], f"{path}:{line_number}: worker norm"
            )
            for worker in workers
        }
        worker_count = len(norms)
        if worker_count < 2:
            raise ValidationError(f"{path}:{line_number}: fewer than two workers")
        expected_pairs = worker_count * (worker_count - 1) // 2
        if len(pairs) != expected_pairs:
            raise ValidationError(
                f"{path}:{line_number}: expected {expected_pairs} worker pairs"
            )
        pair_dots = []
        for pair in pairs:
            learner_a = int(pair["learner_a"])
            learner_b = int(pair["learner_b"])
            cosine = finite_number(
                pair["cosine"], f"{path}:{line_number}: pair cosine"
            )
            pair_dots.append(cosine * norms[learner_a] * norms[learner_b])
        signal_squared = math.fsum(pair_dots) / len(pair_dots)
        individual_second_moment = math.fsum(
            norm * norm for norm in norms.values()
        ) / worker_count
        merged_noise = (individual_second_moment - signal_squared) / worker_count
        if merged_noise <= 0.0:
            raise ValidationError(f"{path}:{line_number}: nonpositive noise proxy")
        signal_total += signal_squared
        merged_noise_total += merged_noise
        event_count += 1
        if signal_squared <= 0.0:
            nonpositive_signal_events += 1
        if int(event.get("fragment_round", 0)) == 1:
            first_round_signal_total += signal_squared
            first_round_merged_noise_total += merged_noise
            first_round_event_count += 1
        lag1 = event.get("autocorrelation", {}).get("lag_1")
        if lag1 is not None:
            lag1_values.append(finite_number(lag1, f"{path}:{line_number}: lag_1"))
    if event_count == 0 or signal_total <= 0.0:
        raise ValidationError(f"{path}: empty telemetry or nonpositive pooled signal")
    if first_round_event_count == 0 or first_round_signal_total <= 0.0:
        raise ValidationError(f"{path}: missing positive-signal first fragment round")
    return {
        "path": str(path),
        "event_count": event_count,
        "signal_total": signal_total,
        "merged_noise_total": merged_noise_total,
        "noise_ratio": merged_noise_total / signal_total,
        "first_round_event_count": first_round_event_count,
        "first_round_signal_total": first_round_signal_total,
        "first_round_merged_noise_total": first_round_merged_noise_total,
        "first_round_noise_ratio": (
            first_round_merged_noise_total / first_round_signal_total
        ),
        "nonpositive_signal_events": nonpositive_signal_events,
        "lag1_values": lag1_values,
    }


def pool_run_summaries(
    summaries: Iterable[dict[str, Any]], *, first_round: bool = False
) -> dict[str, float | int]:
    selected = list(summaries)
    if not selected:
        raise ValidationError("cannot pool an empty run selection")
    signal_key = "first_round_signal_total" if first_round else "signal_total"
    noise_key = (
        "first_round_merged_noise_total" if first_round else "merged_noise_total"
    )
    event_key = "first_round_event_count" if first_round else "event_count"
    signal = math.fsum(float(run[signal_key]) for run in selected)
    noise = math.fsum(float(run[noise_key]) for run in selected)
    if signal <= 0.0 or noise <= 0.0:
        raise ValidationError("pooled telemetry proxy is not positive")
    return {
        "run_count": len(selected),
        "event_count": sum(int(run[event_key]) for run in selected),
        "signal_squared_sum": signal,
        "merged_noise_sum": noise,
        "noise_to_signal_ratio": noise / signal,
    }


def nearest_eta_runs(
    runs: list[dict[str, Any]],
    fits: dict[tuple[int, str, float], dict[str, Any]],
    key: tuple[int, str, float],
) -> tuple[float, list[dict[str, Any]]]:
    fit = fits[key]
    eta_star = finite_number(fit.get("eta_star"), f"{key}: eta_star")
    candidates = [
        run
        for run in runs
        if (int(run["T"]), str(run["arm"]), float(run["mu"])) == key
    ]
    if not candidates:
        raise ValidationError(f"no telemetry runs for curve {key}")
    eta_values = {float(run["eta"]) for run in candidates}
    nearest = min(eta_values, key=lambda eta: abs(math.log(eta / eta_star)))
    selected = [run for run in candidates if float(run["eta"]) == nearest]
    return nearest, selected


def finite_proxy_analysis(
    actual_labels: dict[tuple[int, str, float], str],
    fits: dict[tuple[int, str, float], dict[str, Any]],
    runs: list[dict[str, Any]],
    *,
    use_control_proxy: bool,
    first_round: bool,
) -> dict[str, Any]:
    records = []
    predictions: dict[tuple[int, str, float], str] = {}
    for key in sorted(actual_labels):
        horizon, arm, momentum = key
        proxy_key = (horizon, "mu0", 0.0) if use_control_proxy else key
        selected_eta, selected_runs = nearest_eta_runs(runs, fits, proxy_key)
        proxy = pool_run_summaries(selected_runs, first_round=first_round)
        noise_ratio = float(proxy["noise_to_signal_ratio"])
        critical = finite_critical_noise_ratio(horizon, momentum)
        prediction = phase_from_noise_ratio(noise_ratio, critical)
        predictions[key] = prediction
        records.append(
            {
                "T": horizon,
                "arm": arm,
                "mu": momentum,
                "proxy_curve": {
                    "T": proxy_key[0],
                    "arm": proxy_key[1],
                    "mu": proxy_key[2],
                },
                "selected_registered_eta": selected_eta,
                "noise_to_signal_proxy": noise_ratio,
                "critical_noise_ratio": critical,
                "prediction": prediction,
                "actual_primary_label": actual_labels[key],
            }
        )
    return {
        "proxy_source": "mu0 control at each T" if use_control_proxy else "same arm",
        "time_selection": "first fragment round" if first_round else "full trajectory",
        "records": records,
        "predicted_hurts_cells": [
            {"T": key[0], "arm": key[1], "mu": key[2]}
            for key, value in sorted(predictions.items())
            if value == "HURTS"
        ],
        "confusion_hurts_vs_rest": binary_confusion(actual_labels, predictions),
    }


def eta_ladder_sensitivity(
    actual_labels: dict[tuple[int, str, float], str],
    runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for key in sorted(actual_labels):
        horizon, arm, momentum = key
        curve_runs = [
            run
            for run in runs
            if (int(run["T"]), str(run["arm"]), float(run["mu"])) == key
        ]
        by_eta: defaultdict[float, list[dict[str, Any]]] = defaultdict(list)
        for run in curve_runs:
            by_eta[float(run["eta"])].append(run)
        critical = finite_critical_noise_ratio(horizon, momentum)
        ladder = []
        for eta, eta_runs in sorted(by_eta.items()):
            proxy = pool_run_summaries(eta_runs)
            ratio = float(proxy["noise_to_signal_ratio"])
            ladder.append(
                {
                    "eta": eta,
                    "noise_to_signal_proxy": ratio,
                    "prediction": phase_from_noise_ratio(ratio, critical),
                }
            )
        result.append(
            {
                "T": horizon,
                "arm": arm,
                "mu": momentum,
                "critical_noise_ratio": critical,
                "actual_primary_label": actual_labels[key],
                "ladder": ladder,
                "prediction_constant_across_ladder": (
                    len({entry["prediction"] for entry in ladder}) == 1
                ),
            }
        )
    return result


def telemetry_analysis(
    readout: dict[str, Any],
    telemetry_root: Path,
    actual_labels: dict[tuple[int, str, float], str],
) -> dict[str, Any]:
    cell_records = {
        (str(record["cell_id"]), int(record["attempt"])): record
        for record in readout.get("cell_records", [])
    }
    if not cell_records:
        raise ValidationError("readout has no cell records")
    discovered = sorted(telemetry_root.rglob("rho-telemetry.jsonl"))
    if not discovered:
        raise ValidationError(f"no rho telemetry under {telemetry_root}")
    run_summaries = []
    seen: set[tuple[str, int]] = set()
    all_lag1: list[float] = []
    for path in discovered:
        cell_attempt = locate_cell_and_attempt(path)
        if cell_attempt not in cell_records:
            raise ValidationError(f"telemetry not present in G8 cell registry: {path}")
        if cell_attempt in seen:
            raise ValidationError(f"duplicate telemetry for {cell_attempt}")
        seen.add(cell_attempt)
        summary = summarize_telemetry_file(path)
        metadata = cell_records[cell_attempt]
        summary.update(
            {
                "cell_id": cell_attempt[0],
                "attempt": cell_attempt[1],
                "T": int(metadata["T"]),
                "arm": str(metadata["arm"]),
                "mu": float(metadata["mu"]),
                "eta": float(metadata["eta"]),
                "seed": int(metadata["seed"]),
            }
        )
        all_lag1.extend(summary.pop("lag1_values"))
        run_summaries.append(summary)
    missing = sorted(cell_records.keys() - seen)
    if missing:
        raise ValidationError(f"missing telemetry for {len(missing)} readout cells")
    fits = {comparison_key(fit): fit for fit in readout.get("curve_fits", [])}
    expected_fit_keys = {
        (int(record["T"]), str(record["arm"]), float(record["mu"]))
        for record in readout.get("curve_fits", [])
    }
    if fits.keys() != expected_fit_keys:
        raise ValidationError("duplicate curve fits in readout")
    variants = {
        "same_arm_full_trajectory": finite_proxy_analysis(
            actual_labels,
            fits,
            run_summaries,
            use_control_proxy=False,
            first_round=False,
        ),
        "mu0_control_full_trajectory": finite_proxy_analysis(
            actual_labels,
            fits,
            run_summaries,
            use_control_proxy=True,
            first_round=False,
        ),
        "same_arm_frozen_first_round": finite_proxy_analysis(
            actual_labels,
            fits,
            run_summaries,
            use_control_proxy=False,
            first_round=True,
        ),
        "mu0_control_frozen_first_round": finite_proxy_analysis(
            actual_labels,
            fits,
            run_summaries,
            use_control_proxy=True,
            first_round=True,
        ),
    }
    return {
        "definition": (
            "event signal^2 = mean pairwise worker dot product; merged noise = "
            "(mean worker norm^2 - signal^2) / worker_count; pooled ratio = "
            "sum(merged noise) / sum(signal^2)"
        ),
        "file_count": len(discovered),
        "event_count": sum(int(run["event_count"]) for run in run_summaries),
        "nonpositive_event_signal_estimates": sum(
            int(run["nonpositive_signal_events"]) for run in run_summaries
        ),
        "runs_with_positive_pooled_signal": sum(
            float(run["signal_total"]) > 0.0 for run in run_summaries
        ),
        "runs_with_positive_pooled_noise": sum(
            float(run["merged_noise_total"]) > 0.0 for run in run_summaries
        ),
        "lag1_projected_cosine": {
            "count": len(all_lag1),
            "mean": statistics.fmean(all_lag1) if all_lag1 else None,
            "median": statistics.median(all_lag1) if all_lag1 else None,
            "min": min(all_lag1) if all_lag1 else None,
            "max": max(all_lag1) if all_lag1 else None,
        },
        "finite_prescription_variants": variants,
        "eta_ladder_sensitivity_same_arm_full_trajectory": eta_ladder_sensitivity(
            actual_labels, run_summaries
        ),
    }


def main() -> int:
    args = parse_args()
    readout = load_json(args.readout)
    comparisons = readout.get("analysis", {}).get("bootstrap", {}).get(
        "comparisons", []
    )
    if len(comparisons) != 12:
        raise ValidationError(f"expected 12 G8 comparisons, found {len(comparisons)}")
    actual_labels = {
        comparison_key(record): str(record["primary_phase_label"])
        for record in comparisons
    }
    if len(actual_labels) != len(comparisons):
        raise ValidationError("duplicate G8 comparison coordinates")
    exact_records = []
    exact_predictions = {}
    for key in sorted(actual_labels):
        horizon, arm, momentum = key
        if horizon <= 0 or not 0.0 < momentum < 1.0:
            raise ValidationError(f"the stable positive-momentum hypotheses fail at {key}")
        boundary = exact_boundary(horizon, momentum)
        prediction = str(boundary["prediction"])
        exact_predictions[key] = prediction
        exact_records.append(
            {
                "T": horizon,
                "arm": arm,
                "mu": momentum,
                **boundary,
                "actual_primary_label": actual_labels[key],
            }
        )
    verdict = set_verdict(actual_labels, exact_predictions)
    result: dict[str, Any] = {
        "schema": "yeto_phase_boundary_validation_v1",
        "source": {
            "readout": str(args.readout.resolve()),
            "readout_sha256": sha256_file(args.readout),
            "readout_schema": readout.get("schema"),
            "source_git_commit": readout.get("source_git_commit"),
        },
        "target": {
            "primary_phase_counts": readout.get("analysis", {}).get(
                "primary_phase_counts"
            ),
            "observed_hurts_cells": [
                {"T": key[0], "arm": key[1], "mu": key[2]}
                for key, value in sorted(actual_labels.items())
                if value == "HURTS"
            ],
        },
        "exactly_retuned_iid_test": {
            "criterion": "HURTS iff c(T,mu)^2 - T*v(T,mu) < 0",
            "measured_scale_role": (
                "sigmaSq/(a^2*theta0^2) cancels for positive noise, curvature, "
                "and nonzero signal; no measured threshold is available"
            ),
            "records": exact_records,
            "predicted_hurts_cells": [
                {"T": key[0], "arm": key[1], "mu": key[2]}
                for key, value in sorted(exact_predictions.items())
                if value == "HURTS"
            ],
            "confusion_hurts_vs_rest": binary_confusion(
                actual_labels, exact_predictions
            ),
            "confusion_primary_multiclass": multiclass_confusion(
                actual_labels, exact_predictions
            ),
            "verdict": verdict,
        },
        "hypothesis_audit": {
            "T_positive": True,
            "mu_nonnegative_and_stable": True,
            "experimental_retuning": (
                "all G8 fits are interior four-point quadratics in log2(eta), "
                "which approximates but is not the theorem's exact global retuning"
            ),
            "positive_scalar_curvature": (
                "not directly identified; positive readout fit coefficients are in "
                "log2(eta), not the theorem's scalar parameter-space curvature a"
            ),
            "frozen_scalar_gradient_and_zero_cross_update_covariance": (
                "strong structural assumptions, audited with telemetry when supplied"
            ),
        },
        "verdict": verdict,
        "note_line": f"PHASEVAL: {verdict}",
    }
    if args.telemetry_root is not None:
        telemetry = telemetry_analysis(readout, args.telemetry_root, actual_labels)
        result["telemetry_proxy_sensitivity"] = telemetry
        lag1 = telemetry["lag1_projected_cosine"]
        result["hypothesis_audit"][
            "telemetry_iid_zero_covariance_diagnostic"
        ] = (
            f"not supported: {lag1['count']} non-null projected lag-1 cosines, "
            f"mean={lag1['mean']:.6g}, median={lag1['median']:.6g}"
        )
        result["hypothesis_audit"]["positive_noise_and_signal_proxy"] = (
            telemetry["runs_with_positive_pooled_noise"]
            == telemetry["file_count"]
            and telemetry["runs_with_positive_pooled_signal"]
            == telemetry["file_count"]
        )
    serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
