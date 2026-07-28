#!/usr/bin/env python3
"""Derive the frozen v7 seed-noise prior from the final G4C readout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


EXPECTED_G4C_SHA256 = "16bab85dce3a83f42c8b1c91be8d91eb1ec1372cfba572f2ffd9b491044b04aa"
EXPECTED_SEEDS = (501, 503, 509, 541, 547)
EXPECTED_CURVES = ((2560, 0.0), (2560, 0.9), (10240, 0.0), (10240, 0.9))
EXPECTED_ETA_COUNTS = {
    (2560, 0.0): 4,
    (2560, 0.9): 6,
    (10240, 0.0): 6,
    (10240, 0.9): 6,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g4c-readout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_sha = sha256_file(args.g4c_readout)
    if source_sha != EXPECTED_G4C_SHA256:
        raise SystemExit(
            f"G4C readout hash differs from the registered final readout: {source_sha}"
        )
    readout = json.loads(args.g4c_readout.read_text())
    if readout.get("schema") != "yeto_outer_mup_v4c_g4c_readout_v2":
        raise SystemExit("input is not the final amended G4C readout")
    if len(readout.get("cell_records", [])) != 110:
        raise SystemExit("G4C readout does not contain all 110 cells")

    observed: dict[tuple[int, float, float], dict[int, float]] = defaultdict(dict)
    for row in readout["cell_records"]:
        key = (int(row["s"]), float(row["mu"]), float(row["eta"]))
        seed = int(row["seed"])
        loss = float(row["eval_loss"])
        if not math.isfinite(loss):
            raise SystemExit(f"nonfinite prior loss at {key}/seed{seed}")
        if seed in observed[key]:
            raise SystemExit(f"duplicate prior loss at {key}/seed{seed}")
        observed[key][seed] = loss

    fits = {
        (int(fit["s"]), float(fit["mu"])): fit for fit in readout.get("curve_fits", [])
    }
    if set(fits) != set(EXPECTED_CURVES):
        raise SystemExit("G4C readout curve set differs from the expected four curves")

    residuals_all: list[float] = []
    coordinate_sample_sds: list[float] = []
    curves = []
    for s, mu in EXPECTED_CURVES:
        coordinates = sorted(key for key in observed if key[:2] == (s, mu))
        expected_count = EXPECTED_ETA_COUNTS[(s, mu)]
        if len(coordinates) != expected_count:
            raise SystemExit(f"S{s}/mu{mu}: expected {expected_count} eta coordinates")
        residuals_by_seed = {str(seed): [] for seed in EXPECTED_SEEDS}
        coordinate_seed_sample_sd = []
        seed_mean_losses = []
        for coordinate in coordinates:
            seed_losses = observed[coordinate]
            if set(seed_losses) != set(EXPECTED_SEEDS):
                raise SystemExit(f"{coordinate}: prior seed set is incomplete")
            values = [seed_losses[seed] for seed in EXPECTED_SEEDS]
            mean = statistics.mean(values)
            sd = statistics.stdev(values)
            seed_mean_losses.append(mean)
            coordinate_seed_sample_sd.append(sd)
            coordinate_sample_sds.append(sd)
            for seed in EXPECTED_SEEDS:
                residual = seed_losses[seed] - mean
                residuals_by_seed[str(seed)].append(residual)
                residuals_all.append(residual)
        curve_residuals = [
            value for values in residuals_by_seed.values() for value in values
        ]
        fit = fits[(s, mu)]
        curves.append(
            {
                "s": s,
                "t": int(fit["t"]),
                "mu": mu,
                "etas": [coordinate[2] for coordinate in coordinates],
                "seed_mean_losses": seed_mean_losses,
                "residual_definition": "cell_loss_minus_same_eta_five_seed_mean",
                "residuals_by_seed": residuals_by_seed,
                "coordinate_seed_sample_sd": coordinate_seed_sample_sd,
                "residual_population_sd": statistics.pstdev(curve_residuals),
                "residual_mean_absolute": statistics.mean(
                    abs(value) for value in curve_residuals
                ),
                "source_curve_fit": {
                    "a": float(fit["a"]),
                    "eta_star": float(fit["eta_star"]),
                    "vertex_log2_eta": float(fit["vertex_log2_eta"]),
                },
            }
        )

    prior = {
        "schema": "yeto_outer_mup_v7_empirical_seed_noise_prior_v1",
        "source": {
            "path": str(args.g4c_readout),
            "sha256": source_sha,
            "schema": readout["schema"],
            "source_git_commit": readout.get("source_git_commit"),
            "v4_manifest_sha256": readout.get("v4_manifest_sha256"),
            "v4b_manifest_sha256": readout.get("v4b_manifest_sha256"),
            "v4c_manifest_sha256": readout.get("v4c_manifest_sha256"),
            "cell_count": len(readout["cell_records"]),
        },
        "seeds": list(EXPECTED_SEEDS),
        "coordinate_count": len(observed),
        "residual_count": len(residuals_all),
        "construction": (
            "For each of the 22 combined-grid (S,mu,eta) coordinates, subtract "
            "the same-coordinate five-seed mean. Preserve the resulting seed "
            "profiles jointly across every eta and all four curves."
        ),
        "summary": {
            "pooled_residual_population_sd": statistics.pstdev(residuals_all),
            "pooled_residual_sample_sd": statistics.stdev(residuals_all),
            "pooled_residual_mean_absolute": statistics.mean(
                abs(value) for value in residuals_all
            ),
            "pooled_residual_max_absolute": max(abs(value) for value in residuals_all),
            "coordinate_sample_sd_mean": statistics.mean(coordinate_sample_sds),
            "coordinate_sample_sd_median": statistics.median(coordinate_sample_sds),
            "coordinate_sample_sd_min": min(coordinate_sample_sds),
            "coordinate_sample_sd_max": max(coordinate_sample_sds),
        },
        "observed_constants": {
            "D5": float(readout["D_obs"]["T5"]),
            "D20": float(readout["D_obs"]["T20"]),
        },
        "curves": curves,
    }
    write_json_atomic(args.output, prior)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": sha256_file(args.output),
                "coordinate_count": prior["coordinate_count"],
                "residual_count": prior["residual_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
