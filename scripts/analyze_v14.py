#!/usr/bin/env python3
"""Frozen V14 exact-rate transfer-matrix analyzer."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import day3_common as common
from analyze_v19 import NotEvaluable, read_jsonl, selected_evidence


def student_interval(values: list[float], t_star: float) -> dict[str, object]:
    if len(values) != 5 or any(not math.isfinite(value) for value in values):
        raise NotEvaluable("V14 Student interval requires five finite paired values")
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    half = t_star * sd / math.sqrt(5.0)
    return {"mean": mean, "sd": sd, "ci95": [mean - half, mean + half]}


def validate_manifest(path: Path, manifest: dict) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text().split()[0] != common.sha256_file(path):
        raise NotEvaluable("V14 manifest sidecar mismatch")
    if (
        manifest.get("schema") != "yeto_day3_launch_manifest_v1"
        or manifest.get("program") != "v14"
        or manifest.get("status") != "AUTHORIZED"
    ):
        raise NotEvaluable("V14 launch manifest schema/status mismatch")
    if len(manifest.get("cells", [])) != 160:
        raise NotEvaluable("V14 manifest does not contain 160 cells")
    analyzer = manifest.get("execution_files", {}).get("scripts/analyze_v14.py")
    if analyzer is None or common.sha256_file(Path(__file__)) != analyzer.get("sha256"):
        raise NotEvaluable("frozen V14 analyzer hash mismatch")


def analyze(manifest_path: Path, result_root: Path) -> dict[str, object]:
    manifest = common.read_json(manifest_path)
    validate_manifest(manifest_path, manifest)

    # Completion comes solely from terminal evidence metadata. Endpoint files
    # remain unopened until all 160 cells pass this evidence-only drainage pass.
    selections = {
        cell["cell_id"]: selected_evidence(cell, result_root)
        for cell in manifest["cells"]
    }
    manifest_sha = common.sha256_file(manifest_path)
    invalid_reasons = []
    endpoints = []
    losses: dict[tuple[str, int, int, str, int], float] = {}
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
        key = (
            cell["context"],
            int(cell["target_t"]),
            int(cell["seed"]),
            cell["role"],
            int(cell["source_t"]),
        )
        if key in losses:
            invalid_reasons.append(f"{cell['cell_id']}: duplicate V14 design key")
            continue
        losses[key] = float(loss)
        endpoints.append(
            {
                "cell_id": cell["cell_id"],
                "context": cell["context"],
                "target_t": int(cell["target_t"]),
                "source_t": int(cell["source_t"]),
                "role": cell["role"],
                "seed": int(cell["seed"]),
                "eta": float(cell["eta"]),
                "S": int(cell["s"]),
                "H": int(cell["h"]),
                "attempt": attempt,
                "eval_loss_nats": float(loss),
                "evidence_sha256": common.sha256_file(evidence_path),
                "results_sha256": common.sha256_file(result_path),
            }
        )
    if invalid_reasons:
        raise NotEvaluable("; ".join(invalid_reasons))

    seeds = tuple(manifest["contract"]["seeds"])
    clip = float(manifest["contract"]["clip_bits"])
    margin = float(manifest["contract"]["practical_margin_bits"])
    t_star = float(manifest["contract"]["student_t_df4"])
    pairs = {}
    bounded_by_seed: dict[int, dict[str, list[float]]] = {
        seed: {"upward": [], "downward": []} for seed in seeds
    }
    for context in ("FIXED_H512", "FIXED_S2560"):
        for source_t in (2, 5, 20, 40):
            for target_t in (2, 5, 20, 40):
                if source_t == target_t:
                    continue
                raw = []
                bounded = []
                for seed in seeds:
                    comparator_key = (context, target_t, seed, "comparator", target_t)
                    transfer_key = (context, target_t, seed, "transfer", source_t)
                    if comparator_key not in losses or transfer_key not in losses:
                        raise NotEvaluable(
                            f"missing V14 pairing for {context} T{source_t}->T{target_t} seed{seed}"
                        )
                    penalty = (
                        losses[transfer_key] - losses[comparator_key]
                    ) / math.log(2.0)
                    raw.append(penalty)
                    clipped = min(clip, max(-clip, penalty))
                    bounded.append(clipped)
                    direction = "upward" if source_t < target_t else "downward"
                    bounded_by_seed[seed][direction].append(clipped)
                stats = student_interval(bounded, t_star)
                raw_stats = {
                    "mean": statistics.fmean(raw),
                    "sd": statistics.stdev(raw),
                }
                if stats["ci95"][0] > margin:
                    label = "PAIR_PENALTY"
                elif stats["ci95"][1] < -margin:
                    label = "PAIR_BENEFIT"
                else:
                    label = "PAIR_NO_DECISIVE_PENALTY"
                pair_id = f"{context}:T{source_t}->T{target_t}"
                pairs[pair_id] = {
                    "context": context,
                    "source_t": source_t,
                    "target_t": target_t,
                    "raw_penalties_bits": raw,
                    "raw_mean_bits": raw_stats["mean"],
                    "raw_sd_bits": raw_stats["sd"],
                    "bounded_penalties_bits": bounded,
                    "bounded_mean_bits": stats["mean"],
                    "bounded_sd_bits": stats["sd"],
                    "bounded_ci95_bits": stats["ci95"],
                    "label": label,
                }
    if len(pairs) != 24:
        raise NotEvaluable("V14 did not produce all 24 directed pairs")

    upward_by_seed = []
    downward_by_seed = []
    difference_by_seed = []
    for seed in seeds:
        upward = bounded_by_seed[seed]["upward"]
        downward = bounded_by_seed[seed]["downward"]
        if len(upward) != 12 or len(downward) != 12:
            raise NotEvaluable(f"V14 asymmetry pairing count mismatch for seed {seed}")
        upward_mean = statistics.fmean(upward)
        downward_mean = statistics.fmean(downward)
        upward_by_seed.append(upward_mean)
        downward_by_seed.append(downward_mean)
        difference_by_seed.append(upward_mean - downward_mean)
    upward_stats = student_interval(upward_by_seed, t_star)
    downward_stats = student_interval(downward_by_seed, t_star)
    difference_stats = student_interval(difference_by_seed, t_star)
    if (
        difference_stats["ci95"][0] > margin
        and upward_stats["ci95"][0] > margin
        and abs(downward_stats["mean"]) <= margin
    ):
        verdict = "ASYMMETRY_CONFIRMED"
    elif difference_stats["ci95"][1] < -margin:
        verdict = "ASYMMETRY_REVERSED"
    else:
        verdict = "ASYMMETRY_NULL"
    return {
        "schema": "yeto_v14_frozen_readout_v1",
        "status": "EVALUABLE",
        "manifest_sha256": manifest_sha,
        "source_git_commit": manifest["source"]["git_commit"],
        "scientific_cells": len(endpoints),
        "pairs": pairs,
        "asymmetry": {
            "upward_by_seed_bits": upward_by_seed,
            "downward_by_seed_bits": downward_by_seed,
            "difference_by_seed_bits": difference_by_seed,
            "upward": upward_stats,
            "downward": downward_stats,
            "upward_minus_downward": difference_stats,
        },
        "verdict": verdict,
        "endpoints": sorted(endpoints, key=lambda row: row["cell_id"]),
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
            "schema": "yeto_v14_frozen_readout_v1",
            "status": "NOT_EVALUABLE",
            "verdict": "NOT_EVALUABLE",
            "reason": str(exc),
            "manifest_sha256": common.sha256_file(args.manifest),
        }
    common.write_json_atomic(args.output, readout)
    line = f"V14 VERDICT: {readout['verdict']}"
    append_note(args.note, line)
    print(line)
    print(json.dumps({"output": str(args.output), "verdict": readout["verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
