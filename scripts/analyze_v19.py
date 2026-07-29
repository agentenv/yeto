#!/usr/bin/env python3
"""Frozen V19 analyzer; it refuses endpoint reads until all evidence is terminal."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import day3_common as common


TERMINAL_EVIDENCE = {
    "COMPLETED",
    "SCIENTIFIC_DIVERGENCE",
    "INFRA_FAILURE",
    "INVALID_WORK",
}
VALID_EVIDENCE = {"COMPLETED"}


class NotEvaluable(RuntimeError):
    """A contract validation failure, distinct from an analyzer code error."""


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise NotEvaluable(f"{path}:{number}: result row is not an object")
            rows.append(value)
    return rows


def solve_3x3(matrix: list[list[float]], vector: list[float]) -> list[float]:
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= 1e-18:
            raise NotEvaluable("singular V19 quadratic design")
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


def fit_curve(etas: list[float], losses: list[float]) -> dict[str, object]:
    if len(etas) != 6 or len(losses) != 6:
        raise NotEvaluable("V19 curve does not contain six rung means")
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
    accepted = a > 0.0 and math.isfinite(vertex) and min(x) < vertex < max(x)
    return {
        "accepted": accepted,
        "reason": "accepted" if accepted else "nonpositive_curvature_or_noninterior_vertex",
        "a": a,
        "b": b,
        "c": c,
        "vertex_log2_eta": vertex,
        "eta_star": 2.0**vertex if math.isfinite(vertex) else None,
        "rung_means": losses,
    }


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise NotEvaluable("percentile requested from no valid bootstrap draws")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def selected_evidence(cell: dict, result_root: Path) -> tuple[int, Path, dict]:
    attempts = []
    for attempt in (1, 2):
        path = result_root / cell["cell_id"] / f"attempt-{attempt}" / "evidence.json"
        if path.is_file():
            attempts.append((attempt, path, common.read_json(path)))
    if not attempts:
        raise RuntimeError(f"NOT_DRAINED: no evidence state for {cell['cell_id']}")
    attempt, path, evidence = attempts[-1]
    if attempt == 2 and not cell.get("attempt2_supersedes_attempt1"):
        raise NotEvaluable(f"unregistered attempt-2 selection: {cell['cell_id']}")
    if attempt == 2 and len(str(evidence.get("retry_authority_sha256", ""))) != 64:
        raise NotEvaluable(f"attempt-2 lacks retry authority binding: {cell['cell_id']}")
    if evidence.get("status") not in TERMINAL_EVIDENCE:
        raise RuntimeError(
            f"NOT_DRAINED: nonterminal evidence for {cell['cell_id']}: {evidence.get('status')}"
        )
    return attempt, path, evidence


def validate_manifest(path: Path, manifest: dict) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text().split()[0] != common.sha256_file(path):
        raise NotEvaluable("V19 manifest sidecar mismatch")
    if (
        manifest.get("schema") != "yeto_day3_launch_manifest_v1"
        or manifest.get("program") != "v19"
        or manifest.get("status") != "AUTHORIZED"
    ):
        raise NotEvaluable("V19 launch manifest schema/status mismatch")
    if len(manifest.get("cells", [])) != 54 or len(manifest.get("queues", [])) != 9:
        raise NotEvaluable("V19 manifest cardinality mismatch")
    analyzer = manifest.get("execution_files", {}).get("scripts/analyze_v19.py")
    if analyzer is None or common.sha256_file(Path(__file__)) != analyzer.get("sha256"):
        raise NotEvaluable("frozen V19 analyzer hash mismatch")


def analyze(manifest_path: Path, result_root: Path) -> dict[str, object]:
    manifest = common.read_json(manifest_path)
    validate_manifest(manifest_path, manifest)

    # Evidence-only pass.  No result, loss, or endpoint provenance file is
    # opened until every one of the 54 cells has a terminal evidence status.
    selections: dict[str, tuple[int, Path, dict]] = {}
    for cell in manifest["cells"]:
        selections[cell["cell_id"]] = selected_evidence(cell, result_root)

    manifest_sha = common.sha256_file(manifest_path)
    endpoint_rows = []
    invalid_reasons = []
    losses: dict[str, dict[int, dict[int, float]]] = {}
    eta_by_arm: dict[str, dict[int, float]] = {}
    for cell in manifest["cells"]:
        attempt, evidence_path, evidence = selections[cell["cell_id"]]
        if evidence.get("status") not in VALID_EVIDENCE:
            invalid_reasons.append(
                f"{cell['cell_id']}: terminal status {evidence.get('status')}"
            )
            continue
        if evidence.get("command_hash") != (
            cell["command_hash"]
            if attempt == 1
            else cell["registered_retry_commands"][0]["command_hash"]
        ):
            invalid_reasons.append(f"{cell['cell_id']}: command hash mismatch")
            continue
        if evidence.get("git_commit") != manifest["source"]["git_commit"]:
            invalid_reasons.append(f"{cell['cell_id']}: Git commit mismatch")
            continue
        if evidence.get("manifest_sha256") != manifest_sha:
            invalid_reasons.append(f"{cell['cell_id']}: manifest hash mismatch")
            continue
        if evidence.get("failures"):
            invalid_reasons.append(f"{cell['cell_id']}: cell validation failures")
            continue
        result_path = evidence_path.parent / "report" / "results.jsonl"
        if not result_path.is_file():
            invalid_reasons.append(f"{cell['cell_id']}: missing endpoint result")
            continue
        rows = read_jsonl(result_path)
        if len(rows) != 1:
            invalid_reasons.append(f"{cell['cell_id']}: endpoint row count is not one")
            continue
        loss = rows[0].get("eval_loss")
        if not isinstance(loss, (int, float)) or not math.isfinite(loss):
            invalid_reasons.append(f"{cell['cell_id']}: endpoint loss is nonfinite")
            continue
        arm = cell["arm"]
        eta_index = int(cell["eta_index"])
        seed = int(cell["seed"])
        losses.setdefault(arm, {}).setdefault(eta_index, {})[seed] = float(loss)
        eta_by_arm.setdefault(arm, {})[eta_index] = float(cell["eta"])
        endpoint_rows.append(
            {
                "cell_id": cell["cell_id"],
                "arm": arm,
                "eta_index": eta_index,
                "eta": float(cell["eta"]),
                "seed": seed,
                "attempt": attempt,
                "eval_loss": float(loss),
                "evidence_sha256": common.sha256_file(evidence_path),
                "results_sha256": common.sha256_file(result_path),
            }
        )
    if invalid_reasons:
        raise NotEvaluable("; ".join(invalid_reasons))

    required_arms = ("mu0", "nesterov_raw", "heavy_ball")
    required_seeds = tuple(manifest["contract"]["seeds"])
    fits: dict[str, dict[str, object]] = {}
    ordered_losses: dict[str, list[list[float]]] = {}
    ordered_etas: dict[str, list[float]] = {}
    for arm in required_arms:
        if set(losses.get(arm, {})) != set(range(6)):
            raise NotEvaluable(f"{arm}: incomplete six-rung curve")
        ordered_etas[arm] = [eta_by_arm[arm][index] for index in range(6)]
        ordered_losses[arm] = []
        means = []
        for eta_index in range(6):
            seed_values = losses[arm][eta_index]
            if tuple(sorted(seed_values)) != tuple(sorted(required_seeds)):
                raise NotEvaluable(f"{arm}/rung{eta_index}: incomplete paired seeds")
            values = [seed_values[seed] for seed in required_seeds]
            ordered_losses[arm].append(values)
            means.append(sum(values) / 3.0)
        fits[arm] = fit_curve(ordered_etas[arm], means)
        if not fits[arm]["accepted"]:
            raise NotEvaluable(f"{arm}: pooled curve is unaccepted")

    eta_mu0 = float(fits["mu0"]["eta_star"])
    eta_raw = float(fits["nesterov_raw"]["eta_star"])
    eta_hb = float(fits["heavy_ball"]["eta_star"])
    r_obs = 10.0 * eta_raw / eta_mu0
    h_obs = eta_hb / eta_raw

    rng = random.Random(int(manifest["bootstrap"]["seed"]))
    log2_r_draws = []
    log2_h_draws = []
    for _ in range(int(manifest["bootstrap"]["draws"])):
        indices = [rng.randrange(3) for _ in range(3)]
        draw_fits = {}
        valid = True
        for arm in required_arms:
            draw_means = [
                sum(rung[index] for index in indices) / 3.0
                for rung in ordered_losses[arm]
            ]
            fitted = fit_curve(ordered_etas[arm], draw_means)
            if not fitted["accepted"]:
                valid = False
                break
            draw_fits[arm] = fitted
        if not valid:
            continue
        draw_r = (
            10.0
            * float(draw_fits["nesterov_raw"]["eta_star"])
            / float(draw_fits["mu0"]["eta_star"])
        )
        draw_h = (
            float(draw_fits["heavy_ball"]["eta_star"])
            / float(draw_fits["nesterov_raw"]["eta_star"])
        )
        if draw_r > 0.0 and draw_h > 0.0 and math.isfinite(draw_r) and math.isfinite(draw_h):
            log2_r_draws.append(math.log2(draw_r))
            log2_h_draws.append(math.log2(draw_h))
    minimum_valid = int(manifest["bootstrap"]["minimum_valid"])
    if len(log2_r_draws) < minimum_valid:
        raise NotEvaluable(
            f"only {len(log2_r_draws)} valid joint bootstrap draws; need {minimum_valid}"
        )

    r_log_ci = [percentile(log2_r_draws, 0.025), percentile(log2_r_draws, 0.975)]
    h_log_ci = [percentile(log2_h_draws, 0.025), percentile(log2_h_draws, 0.975)]
    r_ci = [2.0 ** r_log_ci[0], 2.0 ** r_log_ci[1]]
    h_ci = [2.0 ** h_log_ci[0], 2.0 ** h_log_ci[1]]
    predictions = {key: float(value) for key, value in manifest["predictions"].items()}
    covered = [
        label for label, value in predictions.items() if r_log_ci[0] <= math.log2(value) <= r_log_ci[1]
    ]
    distances = {
        label: abs(math.log2(r_obs / value)) for label, value in predictions.items()
    }
    if len(covered) >= 2:
        verdict = "AMBIGUOUS"
    else:
        ordered = sorted(distances.items(), key=lambda item: (item[1], item[0]))
        verdict = (
            "AMBIGUOUS"
            if len(ordered) > 1 and abs(ordered[0][1] - ordered[1][1]) <= 1e-12
            else ordered[0][0]
        )
    hb_target = float(manifest["heavy_ball_target"])
    hb_signature_hit = h_log_ci[0] <= math.log2(hb_target) <= h_log_ci[1]
    return {
        "schema": "yeto_v19_frozen_readout_v1",
        "status": "EVALUABLE",
        "manifest_sha256": manifest_sha,
        "source_git_commit": manifest["source"]["git_commit"],
        "scientific_cells": len(endpoint_rows),
        "fits": fits,
        "r_obs": r_obs,
        "r_ci95": r_ci,
        "r_log2_ci95": r_log_ci,
        "h_obs": h_obs,
        "h_ci95": h_ci,
        "h_log2_ci95": h_log_ci,
        "heavy_ball_target": hb_target,
        "hb_signature_hit": hb_signature_hit,
        "bootstrap": {
            "requested": int(manifest["bootstrap"]["draws"]),
            "valid": len(log2_r_draws),
            "minimum_valid": minimum_valid,
            "seed": int(manifest["bootstrap"]["seed"]),
            "rng": "python_random.Random/randrange",
            "percentile": "linear interpolation at (n-1)*p",
        },
        "predictions": predictions,
        "covered_predictions": covered,
        "distances_log2_bits": distances,
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
            "schema": "yeto_v19_frozen_readout_v1",
            "status": "NOT_EVALUABLE",
            "verdict": "NOT_EVALUABLE",
            "r_obs": None,
            "reason": str(exc),
            "manifest_sha256": common.sha256_file(args.manifest),
        }
    common.write_json_atomic(args.output, readout)
    ratio = "NA" if readout.get("r_obs") is None else f"{readout['r_obs']:.12g}"
    line = f"V19 VERDICT: {readout['verdict']}, r={ratio}"
    append_note(args.note, line)
    print(line)
    print(json.dumps({"output": str(args.output), "verdict": readout["verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
