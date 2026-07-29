#!/usr/bin/env python3
"""Validate randomized Lane-E outputs and adjudicate frozen C3/C4 bands."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


SEEDS = (20260727, 20260728)
C3_BAND = (0.80, 1.25)
C4_BAND = (1.80, 3.40)
MAX_STABILITY_QUOTIENT = 1.25
DATA_SHA256 = "533838a0564b13519956a044d23ed8db6705ddc7ae5f0ddb96538f49460bcebc"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_result(
    path: Path, checkpoint: str, checkpoint_sha256: str, seed: int
) -> tuple[dict, float]:
    result = load_json(path)
    prefix = str(path)
    require(result.get("schema") == "yeto_checkpoint_spectrum_probe_v1", f"{prefix}: schema")
    require(result.get("status") == "COMPLETE", f"{prefix}: status")
    provenance = result["provenance"]
    require(provenance["checkpoint"] == checkpoint, f"{prefix}: checkpoint path")
    require(
        provenance["checkpoint_sha256"] == checkpoint_sha256,
        f"{prefix}: checkpoint hash",
    )
    require(provenance["data_sha256"] == DATA_SHA256, f"{prefix}: data hash")
    require(int(provenance["seed"]) == seed, f"{prefix}: seed")
    require(provenance["device"] == "cpu", f"{prefix}: device")
    probe = result["probe"]
    expected = {
        "seq_len": 128,
        "panels": 4,
        "batch_size": 1,
        "max_rows": 128,
        "train_on": "assistant",
        "block_steps": 4,
        "krylov_rank": 8,
        "start_vector_mode": "gradient_plus_seeded_random",
    }
    for key, value in expected.items():
        require(probe.get(key) == value, f"{prefix}: probe {key}")
    require(int(result["runtime"]["torch_threads"]) == 80, f"{prefix}: threads")
    ritz = [float(value) for value in probe["ritz_values"]]
    require(len(ritz) == 8, f"{prefix}: Ritz count")
    require(all(math.isfinite(value) for value in ritz), f"{prefix}: finite Ritz")
    lambda_max = max(ritz)
    require(lambda_max > 0.0, f"{prefix}: positive top Ritz")
    return result, lambda_max


def verdict(ratios: list[float]) -> tuple[str, bool]:
    low = min(ratios)
    high = max(ratios)
    stable = high / low <= MAX_STABILITY_QUOTIENT
    if stable and low >= C4_BAND[0] and high <= C4_BAND[1]:
        return "C4_SUPPORTED", stable
    if stable and low >= C3_BAND[0] and high <= C3_BAND[1]:
        return "C3_SUPPORTED", stable
    return "AMBIGUOUS", stable


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--spectrum-root", required=True, type=Path)
    parser.add_argument("--equivalence", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--csv-output", required=True, type=Path)
    parser.add_argument("--json-output", required=True, type=Path)
    args = parser.parse_args()

    equivalence = load_json(args.equivalence)
    require(equivalence.get("status") == "PASS", "adapter equivalence did not pass")
    with args.pairs.open(newline="", encoding="utf-8") as handle:
        pairs = list(csv.DictReader(handle, delimiter="\t"))
    require(len(pairs) == 9, f"pair count {len(pairs)} != 9")

    pair_results = []
    seed_rows = []
    for pair in pairs:
        ratios = []
        seed_records = []
        for seed in SEEDS:
            base = args.spectrum_root / pair["pair_id"] / f"seed_{seed}"
            momentum_path = base / "momentum.json"
            control_path = base / "control.json"
            momentum_result, momentum_lambda = validate_result(
                momentum_path,
                pair["momentum_checkpoint"],
                pair["momentum_sha256"],
                seed,
            )
            control_result, control_lambda = validate_result(
                control_path,
                pair["control_checkpoint"],
                pair["control_sha256"],
                seed,
            )
            ratio = momentum_lambda / control_lambda
            require(math.isfinite(ratio) and ratio > 0.0, f"{pair['pair_id']}: ratio")
            ratios.append(ratio)
            record = {
                "seed": seed,
                "lambda_momentum": momentum_lambda,
                "lambda_mu0": control_lambda,
                "ratio": ratio,
                "momentum_result": str(momentum_path.resolve()),
                "momentum_result_sha256": sha256_file(momentum_path),
                "control_result": str(control_path.resolve()),
                "control_result_sha256": sha256_file(control_path),
                "momentum_runtime_seconds": float(momentum_result["runtime"]["seconds"]),
                "control_runtime_seconds": float(control_result["runtime"]["seconds"]),
            }
            seed_records.append(record)
            seed_rows.append(
                {
                    "pair_id": pair["pair_id"],
                    "role": pair["role"],
                    "source": pair["source"],
                    "convention": pair["convention"],
                    "momentum_mu": pair["momentum_mu"],
                    "training_seed": pair["training_seed"],
                    "checkpoint_age": pair["checkpoint_age"],
                    **record,
                }
            )
        label, stable = verdict(ratios)
        low = min(ratios)
        high = max(ratios)
        pair_results.append(
            {
                "pair_id": pair["pair_id"],
                "role": pair["role"],
                "source": pair["source"],
                "convention": pair["convention"],
                "momentum_mu": float(pair["momentum_mu"]),
                "training_seed": int(pair["training_seed"]),
                "checkpoint_age": int(pair["checkpoint_age"]),
                "ratio_geometric_mean": math.exp(sum(math.log(value) for value in ratios) / 2.0),
                "ratio_uncertainty_low": low,
                "ratio_uncertainty_high": high,
                "log_half_spread": abs(math.log(ratios[0]) - math.log(ratios[1])) / 2.0,
                "stability_quotient": high / low,
                "stability_pass": stable,
                "verdict": label,
                "seeds": seed_records,
            }
        )

    primary = [row for row in pair_results if row["role"] == "primary_exact_target"]
    require(primary, "no primary exact-target pairs")
    primary_labels = {row["verdict"] for row in primary}
    overall = (
        next(iter(primary_labels))
        if len(primary_labels) == 1 and "AMBIGUOUS" not in primary_labels
        else "AMBIGUOUS"
    )

    fieldnames = (
        "pair_id",
        "role",
        "source",
        "convention",
        "momentum_mu",
        "training_seed",
        "checkpoint_age",
        "seed",
        "lambda_momentum",
        "lambda_mu0",
        "ratio",
        "momentum_result",
        "momentum_result_sha256",
        "control_result",
        "control_result_sha256",
        "momentum_runtime_seconds",
        "control_runtime_seconds",
    )
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(seed_rows)

    summary = {
        "schema": "yeto_c3_c4_curvature_discriminator_v1",
        "status": "COMPLETE",
        "bands": {
            "C3_SUPPORTED": list(C3_BAND),
            "C4_SUPPORTED": list(C4_BAND),
            "max_stability_quotient": MAX_STABILITY_QUOTIENT,
        },
        "probe": {
            "seeds": list(SEEDS),
            "lambda_definition": "largest algebraic Ritz value",
            "adapter": str(args.adapter.resolve()),
            "adapter_sha256": sha256_file(args.adapter),
            "protocol": str(args.protocol.resolve()),
            "protocol_sha256": sha256_file(args.protocol),
            "equivalence": str(args.equivalence.resolve()),
            "equivalence_sha256": sha256_file(args.equivalence),
        },
        "inventory": {
            "pair_count": len(pair_results),
            "seed_ratio_count": len(seed_rows),
            "primary_exact_target_pair_count": len(primary),
        },
        "pair_results": pair_results,
        "overall_exact_target_verdict": overall,
    }
    args.json_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "COMPLETE", "overall": overall, "pairs": len(pair_results)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
