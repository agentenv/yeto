#!/usr/bin/env python3
"""Power/feasibility simulation for the prospectively frozen G7 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path

try:
    import analyze_v7 as analysis
except ModuleNotFoundError:  # package import in tests
    from scripts import analyze_v7 as analysis


SIMULATION_SEED = 20260727
PRIMARY_SIMULATIONS = 5_000
SENSITIVITY_SIMULATIONS = 2_000
TARGET_OFFSETS = {
    "FULL_48": {
        (2560, 0.0): (-1.5, -0.5, 0.5, 1.5),
        (2560, 0.9): (-1.5, -0.5, 0.5, 1.5),
        (10240, 0.0): (-1.5, -0.5, 0.5, 1.5),
        (10240, 0.9): (-1.5, -0.5, 0.5, 1.5),
    },
    "REDUCED_T20_MU0_45": {
        (2560, 0.0): (-1.5, -0.5, 0.5, 1.5),
        (2560, 0.9): (-1.5, -0.5, 0.5, 1.5),
        (10240, 0.0): (-1.5, 0.0, 1.5),
        (10240, 0.9): (-1.5, -0.5, 0.5, 1.5),
    },
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


def interpolate_profile(values: list[float], target_count: int) -> list[float]:
    if len(values) < 2 or target_count < 2:
        raise ValueError("profile interpolation requires at least two points")
    output = []
    for index in range(target_count):
        quantile = index / (target_count - 1)
        position = quantile * (len(values) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        weight = position - lower
        output.append(values[lower] * (1.0 - weight) + values[upper] * weight)
    return output


def load_prior(path: Path) -> dict:
    prior = json.loads(path.read_text())
    if prior.get("schema") != "yeto_outer_mup_v7_empirical_seed_noise_prior_v1":
        raise SystemExit("input is not the registered v7 empirical-noise prior")
    if prior.get("coordinate_count") != 22 or prior.get("residual_count") != 110:
        raise SystemExit("empirical-noise prior is incomplete")
    if tuple(prior.get("seeds", [])) != (501, 503, 509, 541, 547):
        raise SystemExit("empirical-noise prior seed set changed")
    curves = {
        (int(curve["s"]), float(curve["mu"])): curve
        for curve in prior.get("curves", [])
    }
    expected = {(s, mu) for s in analysis.S_GRID for mu in analysis.MU_GRID}
    if set(curves) != expected:
        raise SystemExit("empirical-noise prior curve set changed")
    prior["_curves"] = curves
    return prior


def bootstrap_draw_multiplicities() -> Counter[tuple[int, int, int]]:
    rng = random.Random(analysis.BOOTSTRAP_SEED)
    return Counter(
        tuple(rng.randrange(len(analysis.SEEDS)) for _ in analysis.SEEDS)
        for _ in range(analysis.BOOTSTRAP_REPLICATES)
    )


def build_simulated_experiment(
    prior: dict,
    variant: str,
    source_seed_profile_ids: list[int],
    noise_scale: float,
) -> tuple[
    dict[tuple[int, float, int, float], float],
    dict[tuple[int, float], list[float]],
]:
    curves = prior["_curves"]
    true_d = {
        2560: float(prior["observed_constants"]["D5"]),
        10240: float(prior["observed_constants"]["D20"]),
    }
    centers = {
        (2560, 0.0): 1.0,
        (2560, 0.9): 0.1 * true_d[2560],
        (10240, 0.0): 1.0,
        (10240, 0.9): 0.1 * true_d[10240],
    }
    losses = {}
    grids = {}
    for coordinate, offsets in TARGET_OFFSETS[variant].items():
        center = centers[coordinate]
        etas = [center * 2.0**offset for offset in offsets]
        grids[coordinate] = etas
        source = curves[coordinate]
        curvature = float(source["source_curve_fit"]["a"])
        true_losses = [1.7 + curvature * offset * offset for offset in offsets]
        for new_seed, source_seed in zip(analysis.SEEDS, source_seed_profile_ids):
            residuals = interpolate_profile(
                [
                    float(value)
                    for value in source["residuals_by_seed"][str(source_seed)]
                ],
                len(offsets),
            )
            for eta, true_loss, residual in zip(etas, true_losses, residuals):
                losses[(coordinate[0], coordinate[1], new_seed, eta)] = (
                    true_loss + noise_scale * residual
                )
    return losses, grids


def assess_simulation(
    losses: dict[tuple[int, float, int, float], float],
    grids: dict[tuple[int, float], list[float]],
    draw_counts: Counter[tuple[int, int, int]],
) -> dict:
    point_fits = {
        (s, mu): analysis.curve_fit(losses, grids, s, mu)
        for s in analysis.S_GRID
        for mu in analysis.MU_GRID
    }
    point_accepted = all(fit["accepted"] for fit in point_fits.values())
    d5 = analysis.d_from_fits(point_fits[(2560, 0.0)], point_fits[(2560, 0.9)])
    d20 = analysis.d_from_fits(point_fits[(10240, 0.0)], point_fits[(10240, 0.9)])

    valid = 0
    gaps = []
    for draw, multiplicity in draw_counts.items():
        fits = {
            (s, mu): analysis.curve_fit(losses, grids, s, mu, list(draw))
            for s in analysis.S_GRID
            for mu in analysis.MU_GRID
        }
        draw_d5 = analysis.d_from_fits(fits[(2560, 0.0)], fits[(2560, 0.9)])
        draw_d20 = analysis.d_from_fits(fits[(10240, 0.0)], fits[(10240, 0.9)])
        if draw_d5 is None or draw_d20 is None:
            continue
        valid += multiplicity
        gaps.extend([math.log2(draw_d5 / draw_d20)] * multiplicity)

    evaluable = point_accepted and valid >= analysis.MIN_VALID_BOOTSTRAP_REPLICATES
    gap_low = analysis.quantile(gaps, 0.025) if gaps else None
    bands = bool(
        d5 is not None
        and d20 is not None
        and analysis.D_BANDS[5][0] <= d5 <= analysis.D_BANDS[5][1]
        and analysis.D_BANDS[20][0] <= d20 <= analysis.D_BANDS[20][1]
    )
    passed = bool(
        evaluable and bands and isinstance(gap_low, (int, float)) and gap_low > 0.0
    )
    return {
        "point_accepted": point_accepted,
        "valid_bootstrap_replicates": valid,
        "evaluable": evaluable,
        "inside_bands": bands,
        "monotone_gap_ci_low_positive": bool(
            isinstance(gap_low, (int, float)) and gap_low > 0.0
        ),
        "passed": passed,
    }


def simulate(
    prior: dict,
    *,
    variant: str,
    simulations: int,
    noise_scale: float,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    source_seeds = list(prior["seeds"])
    draw_counts = bootstrap_draw_multiplicities()
    assessments = []
    for _ in range(simulations):
        selected_profiles = [rng.choice(source_seeds) for _ in analysis.SEEDS]
        losses, grids = build_simulated_experiment(
            prior, variant, selected_profiles, noise_scale
        )
        assessments.append(assess_simulation(losses, grids, draw_counts))

    valid_counts = sorted(item["valid_bootstrap_replicates"] for item in assessments)

    def probability(field: str) -> float:
        return sum(bool(item[field]) for item in assessments) / simulations

    return {
        "variant": variant,
        "simulations": simulations,
        "rng_seed": seed,
        "noise_scale": noise_scale,
        "P_point_accepted": probability("point_accepted"),
        "P_eval": probability("evaluable"),
        "P_bands": probability("inside_bands"),
        "P_monotone_ci": probability("monotone_gap_ci_low_positive"),
        "P_pass": probability("passed"),
        "counts": {
            "evaluable": sum(item["evaluable"] for item in assessments),
            "passed": sum(item["passed"] for item in assessments),
        },
        "valid_bootstrap_replicates": {
            "min": min(valid_counts),
            "median": statistics.median(valid_counts),
            "max": max(valid_counts),
            "p05": valid_counts[math.floor(0.05 * (simulations - 1))],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simulations", type=int, default=PRIMARY_SIMULATIONS)
    parser.add_argument(
        "--sensitivity-simulations", type=int, default=SENSITIVITY_SIMULATIONS
    )
    args = parser.parse_args()
    if args.simulations <= 0 or args.sensitivity_simulations <= 0:
        raise SystemExit("simulation counts must be positive")

    prior = load_prior(args.prior)
    primary = [
        simulate(
            prior,
            variant=variant,
            simulations=args.simulations,
            noise_scale=1.0,
            seed=SIMULATION_SEED,
        )
        for variant in TARGET_OFFSETS
    ]
    sensitivity = [
        simulate(
            prior,
            variant=variant,
            simulations=args.sensitivity_simulations,
            noise_scale=scale,
            seed=SIMULATION_SEED + int(scale * 1000),
        )
        for scale in (0.5, 1.5)
        for variant in TARGET_OFFSETS
    ]
    report = {
        "schema": "yeto_outer_mup_v7_gate_feasibility_simulation_v1",
        "simulation_seed": SIMULATION_SEED,
        "prior": {
            "path": str(args.prior),
            "sha256": sha256_file(args.prior),
            "source_g4c_readout_sha256": prior["source"]["sha256"],
            "measured_pooled_residual_population_sd": prior["summary"][
                "pooled_residual_population_sd"
            ],
        },
        "true_science_model": {
            "D5": prior["observed_constants"]["D5"],
            "D20": prior["observed_constants"]["D20"],
            "curve_curvatures": {
                f"S{s}_mu{mu}": prior["_curves"][(s, mu)]["source_curve_fit"]["a"]
                for s in analysis.S_GRID
                for mu in analysis.MU_GRID
            },
            "mean_curve": "loss=1.7+a*(log2(eta)-log2(eta_star))^2",
        },
        "noise_model": {
            "description": (
                "Each simulated G7 training seed samples with replacement one "
                "of the five complete 1.7B seed-residual profiles. The same "
                "sampled profile is used jointly across all four curves; each "
                "source curve is linearly interpolated by ordinal eta quantile "
                "onto the registered three- or four-point target ladder."
            ),
            "primary_scale": 1.0,
            "sensitivity_scales": [0.5, 1.5],
        },
        "gate": {
            "bootstrap_replicates": analysis.BOOTSTRAP_REPLICATES,
            "bootstrap_seed": analysis.BOOTSTRAP_SEED,
            "minimum_valid_bootstrap_replicates": analysis.MIN_VALID_BOOTSTRAP_REPLICATES,
            "near_bracket_allowance_log2_eta": analysis.NEAR_BRACKET_ALLOWANCE_LOG2,
            "D_bands": {
                "T5": list(analysis.D_BANDS[5]),
                "T20": list(analysis.D_BANDS[20]),
            },
        },
        "target_offsets_log2_eta": {
            variant: {
                f"S{s}_mu{mu}": list(offsets) for (s, mu), offsets in curves.items()
            }
            for variant, curves in TARGET_OFFSETS.items()
        },
        "primary": primary,
        "sensitivity": sensitivity,
        "interpretation": (
            "P_eval is the probability that all four ratio-required point fits "
            "are accepted and at least 7,900 of the analyzer's exact 10,000 "
            "paired bootstrap draws are accepted. This is a transport prior "
            "from 1.7B full tuning, not a claim that 27B LoRA has identical noise."
        ),
    }
    report.pop("_curves", None)
    write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": sha256_file(args.output),
                "primary": [
                    {
                        "variant": item["variant"],
                        "P_eval": item["P_eval"],
                        "P_pass": item["P_pass"],
                    }
                    for item in primary
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
