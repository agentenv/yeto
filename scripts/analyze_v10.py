#!/usr/bin/env python3
"""Frozen fresh-vs-fresh G10 analyzer for exact cross-horizon deployments."""

from __future__ import annotations

import argparse
import itertools
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v10_common import (  # noqa: E402
    V10Error,
    canonical_sha256,
    quantile,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
    write_json_atomic,
)


MANIFEST_SCHEMA = "yeto_outer_mup_v10_freshtransfer_launch_manifest_v1"
CONTRACT_SCHEMA = "yeto_outer_mup_v10_freshtransfer_prereg_v1"
EXPECTED_CELLS = 18
EXPECTED_SEEDS = (941, 947, 953)
PAIR_DEFINITIONS = (
    ("T5_TO_T20", "transfer_t5_to_t20", "comparator_t20"),
    ("T5_TO_T40", "transfer_t5_to_t40", "comparator_t40"),
    ("T20_TO_T5", "transfer_t20_to_t5", "comparator_t5"),
)
LN2 = math.log(2.0)


def parse_node_roots(values: list[str]) -> dict[str, Path]:
    roots = {}
    for value in values:
        if "=" not in value:
            raise V10Error(f"--node-root must be NODE=PATH, got {value!r}")
        node, raw_path = value.split("=", 1)
        if not node or not raw_path or node in roots:
            raise V10Error(f"invalid or duplicate --node-root {value!r}")
        roots[node] = Path(raw_path).resolve()
    return roots


def completed_attempt(cell: dict, root: Path) -> tuple[Path, dict, int, list[str]]:
    completed = []
    for attempt_number in (2, 1):
        attempt = root / cell["cell_id"] / f"attempt-{attempt_number}"
        evidence_path = attempt / "evidence.json"
        if not evidence_path.is_file():
            continue
        evidence = read_json(evidence_path)
        if evidence.get("status") == "COMPLETED":
            completed.append((attempt, evidence, attempt_number))
        elif attempt_number == 2:
            raise V10Error(f"attempt 2 is {evidence.get('status')}")
    if not completed:
        raise V10Error("no completed registered attempt")
    selected = completed[0]
    superseded = [str(item[0]) for item in completed[1:]]
    return selected[0], selected[1], selected[2], superseded


def load_losses(
    manifest: dict, node_roots: dict[str, Path]
) -> tuple[dict[tuple[str, int], float], list[dict], list[str]]:
    losses = {}
    records = []
    errors = []
    for cell in manifest.get("cells", []):
        cell_id = str(cell.get("cell_id", "<missing-cell-id>"))
        try:
            node = str(cell["assignment"]["node"])
            if node not in node_roots:
                raise V10Error(f"no result root supplied for {node}")
            attempt, evidence, attempt_number, superseded = completed_attempt(
                cell, node_roots[node]
            )
            command_record = (
                {"command": cell["command"], "command_hash": cell["command_hash"]}
                if attempt_number == 1
                else cell["registered_retry_commands"][0]
            )
            if evidence.get("cell_id") != cell_id:
                raise V10Error("evidence cell_id mismatch")
            if evidence.get("command_hash") != command_record["command_hash"]:
                raise V10Error("evidence command hash mismatch")
            if evidence.get("seed") != cell["seed"]:
                raise V10Error("evidence seed mismatch")
            if canonical_sha256(command_record["command"]) != command_record["command_hash"]:
                raise V10Error("manifest command no longer matches its hash")
            results_path = attempt / "report" / "results.jsonl"
            observed = evidence.get("observed_artifacts", {}).get("results", {})
            if observed.get("sha256") != sha256_file(results_path):
                raise V10Error("results hash differs from validated evidence")
            rows = read_jsonl(results_path)
            if len(rows) != 1:
                raise V10Error(f"expected one endpoint row, found {len(rows)}")
            loss = rows[0].get("eval_loss")
            if not isinstance(loss, (int, float)) or not math.isfinite(loss):
                raise V10Error("endpoint loss is not finite")
            key = (str(cell["role"]), int(cell["seed"]))
            if key in losses:
                raise V10Error(f"duplicate role/seed coordinate {key}")
            losses[key] = float(loss)
            records.append(
                {
                    "cell_id": cell_id,
                    "role": cell["role"],
                    "source_T": cell.get("source_t"),
                    "target_T": cell["t"],
                    "eta_exact": cell["eta"],
                    "seed": cell["seed"],
                    "training_seed": cell["training_seed"],
                    "eval_loss_nats_per_token": float(loss),
                    "eval_loss_bits_per_token": float(loss) / LN2,
                    "node": node,
                    "gpu": cell["assignment"]["gpu"],
                    "attempt": attempt_number,
                    "superseded_completed_attempts": superseded,
                    "evidence_path": str(attempt / "evidence.json"),
                    "evidence_sha256": sha256_file(attempt / "evidence.json"),
                    "results_sha256": observed["sha256"],
                }
            )
        except (KeyError, OSError, TypeError, ValueError, V10Error) as exc:
            errors.append(f"{cell_id}: {exc}")
    return losses, records, errors


def exact_bootstrap(values: list[float]) -> dict:
    if len(values) != len(EXPECTED_SEEDS):
        raise V10Error("paired bootstrap requires exactly three seeds")
    support = [
        statistics.mean(values[index] for index in draw)
        for draw in itertools.product(range(len(values)), repeat=len(values))
    ]
    return {
        "method": "all 3^3 ordered paired training-seed resamples",
        "support_size": len(support),
        "ci_95_bits_per_token": {
            "low": quantile(support, 0.025),
            "high": quantile(support, 0.975),
        },
        "minimum": min(support),
        "maximum": max(support),
    }


def analyze(losses: dict[tuple[str, int], float], threshold: float) -> dict:
    pairs = {}
    for transfer_id, transfer_role, comparator_role in PAIR_DEFINITIONS:
        seed_values = []
        for seed in EXPECTED_SEEDS:
            transfer = losses[transfer_role, seed]
            comparator = losses[comparator_role, seed]
            seed_values.append((transfer - comparator) / LN2)
        bootstrap = exact_bootstrap(seed_values)
        pairs[transfer_id] = {
            "transfer_role": transfer_role,
            "comparator_role": comparator_role,
            "seeds": list(EXPECTED_SEEDS),
            "paired_penalties_bits_per_token": [
                {"seed": seed, "value": value}
                for seed, value in zip(EXPECTED_SEEDS, seed_values)
            ],
            "mean_penalty_bits_per_token": statistics.mean(seed_values),
            "sample_sd_bits_per_token": statistics.stdev(seed_values),
            "bootstrap": bootstrap,
            "lower_clears_positive_threshold": (
                bootstrap["ci_95_bits_per_token"]["low"] >= threshold
            ),
            "upper_clears_negative_threshold": (
                bootstrap["ci_95_bits_per_token"]["high"] <= -threshold
            ),
        }
    if all(record["lower_clears_positive_threshold"] for record in pairs.values()):
        verdict = "PENALTY_CONFIRMED"
    elif all(record["upper_clears_negative_threshold"] for record in pairs.values()):
        verdict = "PENALTY_REVERSED"
    else:
        verdict = "PENALTY_NULL"
    seed_aggregate = []
    for seed in EXPECTED_SEEDS:
        seed_aggregate.append(
            statistics.mean(
                (losses[transfer_role, seed] - losses[comparator_role, seed]) / LN2
                for _, transfer_role, comparator_role in PAIR_DEFINITIONS
            )
        )
    return {
        "verdict": verdict,
        "penalty_threshold_bits_per_token": threshold,
        "directed_pairs": pairs,
        "equal_weight_three_direction_summary": {
            "paired_seed_values_bits_per_token": [
                {"seed": seed, "value": value}
                for seed, value in zip(EXPECTED_SEEDS, seed_aggregate)
            ],
            "mean_bits_per_token": statistics.mean(seed_aggregate),
            "bootstrap": exact_bootstrap(seed_aggregate),
        },
    }


def format_pair(analysis: dict, pair_id: str) -> str:
    record = analysis["directed_pairs"][pair_id]
    interval = record["bootstrap"]["ci_95_bits_per_token"]
    return (
        f"{record['mean_penalty_bits_per_token']:+.6f}"
        f"[{interval['low']:+.6f},{interval['high']:+.6f}]"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument(
        "--node-root",
        action="append",
        required=True,
        help="NODE=PATH; repeat for each results-bearing node or analysis mirror",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    contract = read_json(args.contract)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise SystemExit("manifest schema mismatch")
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise SystemExit("contract schema mismatch")
    if len(manifest.get("cells", [])) != EXPECTED_CELLS:
        raise SystemExit("manifest is not the complete 18-cell V10 design")
    if manifest.get("contract", {}).get("sha256") != sha256_file(args.contract):
        raise SystemExit("manifest binds another V10 contract")
    roots = parse_node_roots(args.node_root)
    losses, records, errors = load_losses(manifest, roots)
    if errors or len(records) != EXPECTED_CELLS or len(losses) != EXPECTED_CELLS:
        # G10 has no NOT_EVALUABLE category.  Incomplete or invalid work is an
        # execution error: emit no scientific verdict/readout and let the
        # controller recover the registered cells.
        detail = "; ".join(errors[:10]) or f"loaded {len(records)}/{EXPECTED_CELLS}"
        raise SystemExit(f"G10 execution incomplete; no verdict emitted: {detail}")
    threshold = float(contract["gate"]["penalty_threshold_bits_per_token"])
    try:
        analysis = analyze(losses, threshold)
    except (KeyError, TypeError, ValueError, V10Error) as exc:
        raise SystemExit(f"G10 analysis invariant failed; no verdict emitted: {exc}") from exc
    note_line = (
        f"G10 VERDICT: {analysis['verdict']} tau={threshold:.6f}bits "
        f"T5->T20={format_pair(analysis, 'T5_TO_T20')} "
        f"T5->T40={format_pair(analysis, 'T5_TO_T40')} "
        f"T20->T5={format_pair(analysis, 'T20_TO_T5')}"
    )
    readout = {
        "schema": "yeto_outer_mup_v10_g10_readout_v1",
        "created_at_utc": utc_now(),
        "fold_ready": True,
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "contract_path": str(args.contract.resolve()),
        "contract_sha256": sha256_file(args.contract),
        "source_git_commit": manifest.get("source", {}).get("git_commit"),
        "expected_cells": EXPECTED_CELLS,
        "observed_completed_cells": len(records),
        "heldout_evaluation": manifest.get("inputs", {}).get("heldout_audit_jsonl"),
        "cell_records": sorted(records, key=lambda item: item["cell_id"]),
        "analysis": analysis,
        "gate": {
            "name": "G10",
            "closed_vocabulary": [
                "PENALTY_CONFIRMED",
                "PENALTY_NULL",
                "PENALTY_REVERSED",
            ],
            "verdict": analysis["verdict"],
        },
        "note_line": note_line,
    }
    write_json_atomic(args.output.resolve(), readout)
    print(note_line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
