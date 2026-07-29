#!/usr/bin/env python3
"""Frozen robust analyzer for the 612-cell V16 second-family program."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
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
    if len(etas) != 6 or len(losses) != 6:
        raise NotEvaluable("V16 curve does not contain six rung medians")
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
        and min(x) - 0.5 < vertex < max(x) + 0.5
    )
    label = (
        "INTERIOR"
        if accepted and min(x) < vertex < max(x)
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
        "rung_medians": losses,
    }


def validate_manifest(path: Path, manifest: dict) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text().split()[0] != common.sha256_file(path):
        raise NotEvaluable("V16 manifest sidecar mismatch")
    if (
        manifest.get("schema") != "yeto_day3_launch_manifest_v1"
        or manifest.get("program") != "v16"
        or manifest.get("status") != "AUTHORIZED"
    ):
        raise NotEvaluable("V16 launch manifest schema/status mismatch")
    if len(manifest.get("cells", [])) != 612:
        raise NotEvaluable("V16 manifest does not contain 612 cells")
    analyzer = manifest.get("execution_files", {}).get("scripts/analyze_v16.py")
    if analyzer is None or common.sha256_file(Path(__file__)) != analyzer.get("sha256"):
        raise NotEvaluable("frozen V16 analyzer hash mismatch")


def analyze(manifest_path: Path, result_root: Path) -> dict[str, object]:
    manifest = common.read_json(manifest_path)
    validate_manifest(manifest_path, manifest)

    # Evidence-only drainage proof comes first. No endpoint result is opened
    # unless all 612 scientific cells have terminal evidence states.
    selections = {
        cell["cell_id"]: selected_evidence(cell, result_root)
        for cell in manifest["cells"]
    }
    manifest_sha = common.sha256_file(manifest_path)
    invalid_reasons = []
    endpoint_rows = []
    losses: dict[tuple[int, str, int], dict[int, float]] = {}
    eta_values: dict[tuple[int, str, int], float] = {}
    for cell in manifest["cells"]:
        attempt, evidence_path, evidence = selections[cell["cell_id"]]
        if evidence.get("status") != "COMPLETED":
            invalid_reasons.append(f"{cell['cell_id']}: {evidence.get('status')}")
            continue
        if evidence.get("failures"):
            invalid_reasons.append(f"{cell['cell_id']}: validation failures")
            continue
        if not common.command_hash_allowed(cell, attempt, evidence.get("command_hash")):
            invalid_reasons.append(f"{cell['cell_id']}: command hash mismatch")
            continue
        if evidence.get("git_commit") != manifest["source"]["git_commit"]:
            invalid_reasons.append(f"{cell['cell_id']}: Git commit mismatch")
            continue
        if evidence.get("manifest_sha256") != manifest_sha:
            invalid_reasons.append(f"{cell['cell_id']}: manifest hash mismatch")
            continue
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
    for t in (2, 5, 20):
        for arm in ("mu0", "mu09"):
            curve_key = f"T{t}_{arm}"
            etas = []
            rung_seed_losses = []
            medians = []
            for eta_index in range(6):
                key = (t, arm, eta_index)
                seed_map = losses.get(key, {})
                if tuple(sorted(seed_map)) != tuple(sorted(seeds)):
                    raise NotEvaluable(f"{curve_key}/rung{eta_index}: 17-seed mismatch")
                values = [seed_map[seed] for seed in seeds]
                etas.append(eta_values[key])
                rung_seed_losses.append(values)
                medians.append(float(statistics.median(values)))
            fitted = fit_curve(etas, medians)
            if not fitted["accepted"]:
                raise NotEvaluable(f"{curve_key}: pooled median fit is unaccepted")
            curve_fits[curve_key] = fitted
            ordered_losses[(t, arm)] = rung_seed_losses
            ordered_etas[(t, arm)] = etas

    d_obs = {
        t: 10.0
        * float(curve_fits[f"T{t}_mu09"]["eta_star"])
        / float(curve_fits[f"T{t}_mu0"]["eta_star"])
        for t in (2, 5, 20)
    }
    rng = random.Random(int(manifest["bootstrap"]["seed"]))
    log_d_draws = {t: [] for t in (2, 5, 20)}
    adjacent_draws = {"D2_gt_D5": [], "D5_gt_D20": []}
    valid = 0
    for _ in range(int(manifest["bootstrap"]["draws"])):
        indices = [rng.randrange(17) for _ in range(17)]
        draw_fits = {}
        accepted = True
        for t in (2, 5, 20):
            for arm in ("mu0", "mu09"):
                draw_medians = [
                    float(statistics.median(rung[index] for index in indices))
                    for rung in ordered_losses[(t, arm)]
                ]
                fitted = fit_curve(ordered_etas[(t, arm)], draw_medians)
                if not fitted["accepted"]:
                    accepted = False
                    break
                draw_fits[(t, arm)] = fitted
            if not accepted:
                break
        if not accepted:
            continue
        log_d = {}
        for t in (2, 5, 20):
            ratio = (
                10.0
                * float(draw_fits[(t, "mu09")]["eta_star"])
                / float(draw_fits[(t, "mu0")]["eta_star"])
            )
            log_d[t] = math.log2(ratio)
            log_d_draws[t].append(log_d[t])
        adjacent_draws["D2_gt_D5"].append(log_d[2] - log_d[5])
        adjacent_draws["D5_gt_D20"].append(log_d[5] - log_d[20])
        valid += 1
    minimum_valid = int(manifest["bootstrap"]["minimum_valid"])
    if valid < minimum_valid:
        raise NotEvaluable(f"only {valid} valid joint bootstrap draws; need {minimum_valid}")

    d_ci = {
        str(t): [
            2.0 ** percentile(log_d_draws[t], 0.025),
            2.0 ** percentile(log_d_draws[t], 0.975),
        ]
        for t in (2, 5, 20)
    }
    adjacent_ci = {
        label: [percentile(values, 0.025), percentile(values, 0.975)]
        for label, values in adjacent_draws.items()
    }
    point_monotone = d_obs[2] > d_obs[5] > d_obs[20]
    interval_monotone = (
        adjacent_ci["D2_gt_D5"][0] > 0.0
        and adjacent_ci["D5_gt_D20"][0] > 0.0
    )
    verdict = (
        "SECOND_FAMILY_MONOTONE"
        if point_monotone and interval_monotone
        else "SECOND_FAMILY_NONMONOTONE"
    )
    return {
        "schema": "yeto_v16_frozen_readout_v1",
        "status": "EVALUABLE",
        "manifest_sha256": manifest_sha,
        "source_git_commit": manifest["source"]["git_commit"],
        "scientific_cells": len(endpoint_rows),
        "curve_fits": curve_fits,
        "D_obs": {str(t): d_obs[t] for t in d_obs},
        "D_ci95": d_ci,
        "adjacent_log2_difference_ci95": adjacent_ci,
        "point_monotone": point_monotone,
        "interval_monotone": interval_monotone,
        "bootstrap": {
            "requested": int(manifest["bootstrap"]["draws"]),
            "valid": valid,
            "minimum_valid": minimum_valid,
            "seed": int(manifest["bootstrap"]["seed"]),
            "rng": "python_random.Random/randrange",
        },
        "verdict": verdict,
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
            "schema": "yeto_v16_frozen_readout_v1",
            "status": "NOT_EVALUABLE",
            "verdict": "SECOND_FAMILY_NOT_EVALUABLE",
            "reason": str(exc),
            "manifest_sha256": common.sha256_file(args.manifest),
        }
    common.write_json_atomic(args.output, readout)
    line = f"V16 VERDICT: {readout['verdict']}"
    append_note(args.note, line)
    print(line)
    print(json.dumps({"output": str(args.output), "verdict": readout["verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
