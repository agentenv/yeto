#!/usr/bin/env python3
"""Frozen V18 FedAdam analyzer with evidence-first endpoint gating."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import day3_common as common
from analyze_v19 import (
    NotEvaluable,
    percentile,
    read_jsonl,
    selected_evidence,
    solve_3x3,
)


def fit_curve(etas: list[float], losses: list[float]) -> dict[str, object]:
    if len(etas) != 4 or len(losses) != 4:
        raise NotEvaluable("V18 curve does not contain four rung means")
    if any(not math.isfinite(value) for value in etas + losses):
        return {"accepted": False, "reason": "nonfinite_input"}
    x = [math.log2(eta) for eta in etas]
    sums = [sum(value**power for value in x) for power in range(5)]
    matrix = [
        [sums[4], sums[3], sums[2]],
        [sums[3], sums[2], sums[1]],
        [sums[2], sums[1], sums[0]],
    ]
    vector = [
        sum((value**2) * loss for value, loss in zip(x, losses)),
        sum(value * loss for value, loss in zip(x, losses)),
        sum(losses),
    ]
    a, b, c = solve_3x3(matrix, vector)
    vertex = -b / (2.0 * a) if a != 0.0 else math.nan
    accepted = (
        a > 0.0
        and math.isfinite(vertex)
        and min(x) - 0.5 <= vertex <= max(x) + 0.5
    )
    label = (
        "BRACKETED"
        if accepted and min(x) <= vertex <= max(x)
        else "NEAR_BRACKETED" if accepted else "UNACCEPTED"
    )
    return {
        "accepted": accepted,
        "acceptance_label": label,
        "a": a,
        "b": b,
        "c": c,
        "vertex_log2_eta": vertex,
        "eta_star": 2.0**vertex if math.isfinite(vertex) else None,
        "rung_means": losses,
    }


def validate_manifest(path: Path, manifest: dict) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text().split()[0] != common.sha256_file(path):
        raise NotEvaluable("V18 manifest sidecar mismatch")
    if (
        manifest.get("schema") != "yeto_day3_launch_manifest_v1"
        or manifest.get("program") != "v18"
        or manifest.get("status") != "AUTHORIZED"
    ):
        raise NotEvaluable("V18 launch manifest schema/status mismatch")
    if len(manifest.get("cells", [])) != 96:
        raise NotEvaluable("V18 manifest does not contain 96 cells")
    analyzer = manifest.get("execution_files", {}).get("scripts/analyze_v18.py")
    if analyzer is None or common.sha256_file(Path(__file__)) != analyzer.get("sha256"):
        raise NotEvaluable("frozen V18 analyzer hash mismatch")


def validate_fedadam_trace(cell: dict, tape_path: Path, manifest: dict) -> dict[str, object]:
    rows = read_jsonl(tape_path)
    expected_rows = 4 * int(cell["t"])
    if len(rows) != expected_rows:
        raise NotEvaluable(f"{cell['cell_id']}: FedAdam tape row count mismatch")
    fedadam_keys = {
        "fedadam_beta1",
        "fedadam_beta2",
        "fedadam_m_before_l2",
        "fedadam_v_before_l1",
        "fedadam_m_after_l2",
        "fedadam_v_after_l1",
        "fedadam_recurrence_max_abs_error",
        "fedadam_zero_safe_coordinates",
    }
    if cell["arm"] == "sgd":
        if any(fedadam_keys & set(row) for row in rows):
            raise NotEvaluable(f"{cell['cell_id']}: SGD tape contains FedAdam state")
        return {"rows": len(rows), "fedadam_fields": False}

    beta1 = float(manifest["contract"]["fedadam"]["beta1_f32"])
    beta2 = float(manifest["contract"]["fedadam"]["beta2_f32"])
    by_fragment: dict[int, list[dict]] = {fragment: [] for fragment in range(4)}
    for row in rows:
        if not fedadam_keys <= set(row):
            raise NotEvaluable(f"{cell['cell_id']}: incomplete FedAdam trace row")
        fragment = row.get("fragment")
        if fragment not in by_fragment:
            raise NotEvaluable(f"{cell['cell_id']}: invalid trace fragment")
        if row["fedadam_beta1"] != beta1 or row["fedadam_beta2"] != beta2:
            raise NotEvaluable(f"{cell['cell_id']}: FedAdam beta trace mismatch")
        if row["fedadam_recurrence_max_abs_error"] != 0.0:
            raise NotEvaluable(f"{cell['cell_id']}: nonzero FedAdam recurrence residual")
        for key in (
            "fedadam_m_before_l2",
            "fedadam_v_before_l1",
            "fedadam_m_after_l2",
            "fedadam_v_after_l1",
        ):
            value = row[key]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0.0:
                raise NotEvaluable(f"{cell['cell_id']}: invalid {key}")
        zero_safe = row["fedadam_zero_safe_coordinates"]
        if not isinstance(zero_safe, int) or zero_safe < 0:
            raise NotEvaluable(f"{cell['cell_id']}: invalid zero-safe trace count")
        by_fragment[fragment].append(row)
    for fragment, fragment_rows in by_fragment.items():
        fragment_rows.sort(key=lambda row: row["step"])
        if len(fragment_rows) != int(cell["t"]):
            raise NotEvaluable(f"{cell['cell_id']}: fragment {fragment} age mismatch")
        first = fragment_rows[0]
        if first["fedadam_m_before_l2"] != 0.0 or first["fedadam_v_before_l1"] != 0.0:
            raise NotEvaluable(f"{cell['cell_id']}: fragment {fragment} is not zero initialized")
        for previous, current in zip(fragment_rows, fragment_rows[1:]):
            if current["fedadam_m_before_l2"] != previous["fedadam_m_after_l2"]:
                raise NotEvaluable(f"{cell['cell_id']}: fragment {fragment} m state did not persist")
            if current["fedadam_v_before_l1"] != previous["fedadam_v_after_l1"]:
                raise NotEvaluable(f"{cell['cell_id']}: fragment {fragment} v state did not persist")
    return {
        "rows": len(rows),
        "fedadam_fields": True,
        "zero_initialized_fragments": 4,
        "calls_per_fragment": int(cell["t"]),
        "beta1_f32": beta1,
        "beta2_f32": beta2,
        "max_recurrence_error": 0.0,
    }


def analyze(manifest_path: Path, result_root: Path) -> dict[str, object]:
    manifest = common.read_json(manifest_path)
    validate_manifest(manifest_path, manifest)

    # Evidence-only pass: endpoint and event-tape files stay unopened until
    # all 96 cells have terminal evidence states.
    selections = {
        cell["cell_id"]: selected_evidence(cell, result_root)
        for cell in manifest["cells"]
    }
    manifest_sha = common.sha256_file(manifest_path)
    invalid_reasons = []
    endpoint_rows = []
    trace_rows = []
    losses: dict[tuple[int, str, int], dict[int, float]] = {}
    eta_values: dict[tuple[int, str, int], float] = {}
    for cell in manifest["cells"]:
        attempt, evidence_path, evidence = selections[cell["cell_id"]]
        expected_hash = (
            cell["command_hash"]
            if attempt == 1
            else cell["registered_retry_commands"][0]["command_hash"]
        )
        if evidence.get("status") != "COMPLETED":
            invalid_reasons.append(f"{cell['cell_id']}: {evidence.get('status')}")
            continue
        if evidence.get("failures"):
            invalid_reasons.append(f"{cell['cell_id']}: validation failures")
            continue
        if evidence.get("command_hash") != expected_hash:
            invalid_reasons.append(f"{cell['cell_id']}: command hash mismatch")
            continue
        if evidence.get("git_commit") != manifest["source"]["git_commit"]:
            invalid_reasons.append(f"{cell['cell_id']}: Git commit mismatch")
            continue
        if evidence.get("manifest_sha256") != manifest_sha:
            invalid_reasons.append(f"{cell['cell_id']}: manifest hash mismatch")
            continue
        tape_path = evidence_path.parent / "work" / "m4" / "tape.jsonl"
        if not tape_path.is_file():
            invalid_reasons.append(f"{cell['cell_id']}: missing optimizer trace")
            continue
        try:
            trace = validate_fedadam_trace(cell, tape_path, manifest)
        except NotEvaluable as exc:
            invalid_reasons.append(str(exc))
            continue
        trace_rows.append(
            {
                "cell_id": cell["cell_id"],
                "tape_sha256": common.sha256_file(tape_path),
                **trace,
            }
        )
        result_path = evidence_path.parent / "report" / "results.jsonl"
        if not result_path.is_file():
            invalid_reasons.append(f"{cell['cell_id']}: missing endpoint")
            continue
        rows = read_jsonl(result_path)
        if len(rows) != 1:
            invalid_reasons.append(f"{cell['cell_id']}: endpoint row count mismatch")
            continue
        loss = rows[0].get("eval_loss")
        if not isinstance(loss, (int, float)) or not math.isfinite(loss):
            invalid_reasons.append(f"{cell['cell_id']}: nonfinite endpoint")
            continue
        key = (int(cell["t"]), cell["arm"], int(cell["eta_index"]))
        losses.setdefault(key, {})[int(cell["seed"])] = float(loss)
        eta_values[key] = float(cell["eta"])
        endpoint_rows.append(
            {
                "cell_id": cell["cell_id"],
                "T": int(cell["t"]),
                "arm": cell["arm"],
                "eta_index": int(cell["eta_index"]),
                "eta": float(cell["eta"]),
                "seed": int(cell["seed"]),
                "attempt": attempt,
                "eval_loss": float(loss),
                "evidence_sha256": common.sha256_file(evidence_path),
                "results_sha256": common.sha256_file(result_path),
            }
        )
    if invalid_reasons:
        raise NotEvaluable("; ".join(invalid_reasons))

    seeds = tuple(manifest["contract"]["seeds"])
    curve_fits = {}
    ordered_losses = {}
    ordered_etas = {}
    for t in (2, 5, 20, 40):
        for arm in ("sgd", "fedadam"):
            curve_key = f"T{t}_{arm}"
            etas = []
            rung_seed_losses = []
            means = []
            for eta_index in range(4):
                key = (t, arm, eta_index)
                seed_map = losses.get(key, {})
                if tuple(sorted(seed_map)) != tuple(sorted(seeds)):
                    raise NotEvaluable(f"{curve_key}/rung{eta_index}: paired seed mismatch")
                values = [seed_map[seed] for seed in seeds]
                etas.append(eta_values[key])
                rung_seed_losses.append(values)
                means.append(sum(values) / 3.0)
            fitted = fit_curve(etas, means)
            if not fitted["accepted"]:
                raise NotEvaluable(f"{curve_key}: pooled fit is unaccepted")
            curve_fits[curve_key] = fitted
            ordered_losses[(t, arm)] = rung_seed_losses
            ordered_etas[(t, arm)] = etas

    d_obs = {
        t: float(curve_fits[f"T{t}_fedadam"]["eta_star"])
        / float(curve_fits[f"T{t}_sgd"]["eta_star"])
        for t in (2, 5, 20, 40)
    }
    rng = random.Random(int(manifest["bootstrap"]["seed"]))
    log_d_draws = {t: [] for t in (2, 5, 20, 40)}
    eta_draws = {(t, arm): [] for t in (2, 5, 20, 40) for arm in ("sgd", "fedadam")}
    valid = 0
    for _ in range(int(manifest["bootstrap"]["draws"])):
        indices = [rng.randrange(3) for _ in range(3)]
        draw_fits = {}
        accepted = True
        for t in (2, 5, 20, 40):
            for arm in ("sgd", "fedadam"):
                draw_means = [
                    sum(rung[index] for index in indices) / 3.0
                    for rung in ordered_losses[(t, arm)]
                ]
                fitted = fit_curve(ordered_etas[(t, arm)], draw_means)
                if not fitted["accepted"]:
                    accepted = False
                    break
                draw_fits[(t, arm)] = fitted
            if not accepted:
                break
        if not accepted:
            continue
        valid += 1
        for t in (2, 5, 20, 40):
            for arm in ("sgd", "fedadam"):
                eta_draws[(t, arm)].append(float(draw_fits[(t, arm)]["eta_star"]))
            ratio = float(draw_fits[(t, "fedadam")]["eta_star"]) / float(
                draw_fits[(t, "sgd")]["eta_star"]
            )
            log_d_draws[t].append(math.log2(ratio))
    minimum_valid = int(manifest["bootstrap"]["minimum_valid"])
    if valid < minimum_valid:
        raise NotEvaluable(f"only {valid} valid joint bootstrap draws; need {minimum_valid}")

    predictions = {int(t): float(value) for t, value in manifest["predictions"].items()}
    band = float(manifest["band_bits"])
    hits = {}
    d_ci = {}
    prediction_error_ci = {}
    eta_ci = {}
    for t in (2, 5, 20, 40):
        lower = percentile(log_d_draws[t], 0.025)
        upper = percentile(log_d_draws[t], 0.975)
        d_ci[t] = [2.0**lower, 2.0**upper]
        errors = [value - math.log2(predictions[t]) for value in log_d_draws[t]]
        prediction_error_ci[t] = [percentile(errors, 0.025), percentile(errors, 0.975)]
        hits[t] = abs(math.log2(d_obs[t] / predictions[t])) <= band
        for arm in ("sgd", "fedadam"):
            eta_ci[f"T{t}_{arm}"] = [
                percentile(eta_draws[(t, arm)], 0.025),
                percentile(eta_draws[(t, arm)], 0.975),
            ]
    verdict = "SHAPE_CONFIRMED" if sum(hits.values()) >= 3 else "SHAPE_WRONG"
    return {
        "schema": "yeto_v18_frozen_readout_v1",
        "status": "EVALUABLE",
        "manifest_sha256": manifest_sha,
        "source_git_commit": manifest["source"]["git_commit"],
        "scientific_cells": len(endpoint_rows),
        "curve_fits": curve_fits,
        "eta_star_ci95": eta_ci,
        "D_obs": {str(t): d_obs[t] for t in d_obs},
        "D_pred": {str(t): predictions[t] for t in predictions},
        "D_ci95": {str(t): d_ci[t] for t in d_ci},
        "prediction_error_log2_ci95": {
            str(t): prediction_error_ci[t] for t in prediction_error_ci
        },
        "hits": {str(t): hits[t] for t in hits},
        "hit_count": sum(hits.values()),
        "bootstrap": {
            "requested": int(manifest["bootstrap"]["draws"]),
            "valid": valid,
            "minimum_valid": minimum_valid,
            "seed": int(manifest["bootstrap"]["seed"]),
            "rng": "python_random.Random/randrange",
        },
        "verdict": verdict,
        "trace_evidence": sorted(trace_rows, key=lambda row: row["cell_id"]),
        "endpoints": sorted(endpoint_rows, key=lambda row: row["cell_id"]),
    }


def append_note(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as destination:
        destination.write(line.rstrip() + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--note", type=Path, default=Path("/private/tmp/h200-day3-note.md"))
    args = parser.parse_args()
    try:
        readout = analyze(args.manifest, args.result_root)
    except NotEvaluable as exc:
        readout = {
            "schema": "yeto_v18_frozen_readout_v1",
            "status": "NOT_EVALUABLE",
            "verdict": "NOT_EVALUABLE",
            "reason": str(exc),
            "manifest_sha256": common.sha256_file(args.manifest),
        }
    common.write_json_atomic(args.output, readout)
    line = f"V18 VERDICT: {readout['verdict']}"
    append_note(args.note, line)
    print(line)
    print(json.dumps({"output": str(args.output), "verdict": readout["verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
