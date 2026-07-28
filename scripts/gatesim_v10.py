#!/usr/bin/env python3
"""Measured-noise feasibility simulation for the V10 fresh-transfer gate."""

from __future__ import annotations

import argparse
import itertools
import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v10_common import (  # noqa: E402
    V10Error,
    quantile,
    read_json,
    sha256_file,
    utc_now,
    write_json_atomic,
)


SOURCE_READOUT_SHA256 = (
    "16bab85dce3a83f42c8b1c91be8d91eb1ec1372cfba572f2ffd9b491044b04aa"
)
SEEDS = (501, 503, 509, 541, 547)
FRESH_SEED_COUNT = 3
SIMULATIONS = 20_000
RNG_SEED = 20260730
MANDATORY_P_EVAL = 0.80
LN2 = math.log(2.0)

# Exact five-seed G4C raw eta-star vertices.  These are source prescriptions,
# not nearest target-grid rates.
ETA_T5 = 0.003191644884294105
ETA_T20 = 0.0008223020084526104

# V11's independently frozen far-horizon construction: extrapolated 1.7B
# mu0 anchor times the pre-outcome D(T40) transport ratio.
MU0_T40_PLACEMENT = 0.003800560784723474
D_T40 = 1.044905865516022
ETA_T40 = MU0_T40_PLACEMENT * 0.1 * D_T40

TRANSFERS = (
    ("T5_TO_T20", 5, 20, ETA_T5, ETA_T20),
    ("T5_TO_T40", 5, 40, ETA_T5, ETA_T40),
    ("T20_TO_T5", 20, 5, ETA_T20, ETA_T5),
)


def sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        raise V10Error("sample SD requires at least two values")
    return statistics.stdev(values)


def raw_matrix(readout: dict, t: int) -> tuple[list[float], dict[tuple[float, int], float]]:
    records = [
        record
        for record in readout.get("cell_records", [])
        if int(record.get("t", -1)) == t
        and float(record.get("mu", -1.0)) == 0.9
        and int(record.get("seed", -1)) in SEEDS
    ]
    etas = sorted({float(record["eta"]) for record in records})
    matrix = {
        (float(record["eta"]), int(record["seed"])): float(record["eval_loss"])
        / LN2
        for record in records
    }
    if len(etas) != 6 or len(matrix) != 6 * len(SEEDS):
        raise V10Error(f"T={t}: expected complete 6x5 raw G4C matrix")
    return etas, matrix


def curve_fit(readout: dict, t: int) -> dict:
    matches = [
        fit
        for fit in readout.get("curve_fits", [])
        if int(fit.get("t", -1)) == t and float(fit.get("mu", -1.0)) == 0.9
    ]
    if len(matches) != 1 or matches[0].get("status") != "INTERIOR":
        raise V10Error(f"T={t}: missing unique interior G4C raw fit")
    return matches[0]


def all_contrast_noise(readout: dict) -> dict:
    sds = []
    residuals = []
    by_t = {}
    for t in (5, 20):
        etas, matrix = raw_matrix(readout, t)
        local_sds = []
        for left, right in itertools.combinations(etas, 2):
            differences = [
                matrix[left, seed] - matrix[right, seed] for seed in SEEDS
            ]
            mean = statistics.mean(differences)
            sd = sample_sd(differences)
            local_sds.append(sd)
            sds.append(sd)
            residuals.extend(value - mean for value in differences)
        by_t[str(t)] = {
            "pair_count": len(local_sds),
            "minimum_paired_contrast_sd_bits": min(local_sds),
            "median_paired_contrast_sd_bits": statistics.median(local_sds),
            "maximum_paired_contrast_sd_bits": max(local_sds),
        }
    degrees_of_freedom = len(residuals) - len(sds)
    pooled = math.sqrt(sum(value * value for value in residuals) / degrees_of_freedom)
    return {
        "source": "all within-horizon raw-rate paired contrasts in the banked five-seed G4C 1.7B curves",
        "source_seeds": list(SEEDS),
        "contrast_count": len(sds),
        "by_target_horizon": by_t,
        "pooled_paired_contrast_sd_bits": pooled,
        "median_paired_contrast_sd_bits": statistics.median(sds),
        "maximum_paired_contrast_sd_bits": max(sds),
        "pooled_degrees_of_freedom": degrees_of_freedom,
    }


def predicted_penalties(readout: dict) -> dict[str, dict]:
    fits = {5: curve_fit(readout, 5), 20: curve_fit(readout, 20)}
    if not math.isclose(float(fits[5]["eta_star"]), ETA_T5, rel_tol=0, abs_tol=1e-18):
        raise V10Error("T5 eta-star constant differs from G4C")
    if not math.isclose(float(fits[20]["eta_star"]), ETA_T20, rel_tol=0, abs_tol=1e-18):
        raise V10Error("T20 eta-star constant differs from G4C")
    records = {}
    for transfer_id, source_t, target_t, source_eta, target_eta in TRANSFERS:
        # T40 has no prior 1.7B curve by design.  Its prospective penalty uses
        # the nearest registered long-horizon curvature donor, G4C/T20; this is
        # explicitly a placement/power calculation, never a V10 outcome.
        curvature_donor_t = target_t if target_t in fits else 20
        curvature_nats_per_log2_eta_squared = float(fits[curvature_donor_t]["a"])
        mismatch = math.log2(source_eta / target_eta)
        penalty_nats = curvature_nats_per_log2_eta_squared * mismatch * mismatch
        records[transfer_id] = {
            "source_T": source_t,
            "target_T": target_t,
            "source_eta_exact": source_eta,
            "target_eta_exact": target_eta,
            "signed_log2_eta_mismatch_bits": mismatch,
            "curvature_donor_T": curvature_donor_t,
            "curvature_nats_per_log2_eta_squared": curvature_nats_per_log2_eta_squared,
            "surface_predicted_penalty_nats_per_token": penalty_nats,
            "surface_predicted_penalty_bits_per_token": penalty_nats / LN2,
        }
    return records


def nearest_residual_profiles(
    readout: dict, target_t: int, requested_separation: float
) -> list[list[float]]:
    donor_t = target_t if target_t in (5, 20) else 20
    etas, matrix = raw_matrix(readout, donor_t)
    pairs = list(itertools.combinations(etas, 2))
    distances = [abs(math.log2(left / right)) for left, right in pairs]
    closest = min(abs(distance - requested_separation) for distance in distances)
    selected = [
        pair
        for pair, distance in zip(pairs, distances)
        if math.isclose(abs(distance - requested_separation), closest, abs_tol=1e-12)
    ]
    profiles = []
    for left, right in selected:
        values = [matrix[right, seed] - matrix[left, seed] for seed in SEEDS]
        mean = statistics.mean(values)
        profiles.append([value - mean for value in values])
    if not profiles:
        raise V10Error("no measured residual profile selected")
    return profiles


def exact_bootstrap_ci(values: list[float]) -> tuple[float, float]:
    if len(values) != FRESH_SEED_COUNT:
        raise V10Error("V10 bootstrap requires exactly three paired seeds")
    means = [
        statistics.mean(values[index] for index in draw)
        for draw in itertools.product(range(FRESH_SEED_COUNT), repeat=FRESH_SEED_COUNT)
    ]
    return quantile(means, 0.025), quantile(means, 0.975)


def simulate(
    readout: dict, predictions: dict[str, dict], threshold: float
) -> dict:
    rng = random.Random(RNG_SEED)
    profiles = {
        transfer_id: nearest_residual_profiles(
            readout,
            int(record["target_T"]),
            abs(float(record["signed_log2_eta_mismatch_bits"])),
        )
        for transfer_id, record in predictions.items()
    }
    evaluable = 0
    confirmed = 0
    reversed_count = 0
    null_count = 0
    for _ in range(SIMULATIONS):
        # A shared banked seed index preserves the observed cross-horizon seed
        # pairing.  The equally close rate-pair profile is sampled independently
        # for each directed contrast.
        banked_indices = [rng.randrange(len(SEEDS)) for _ in range(FRESH_SEED_COUNT)]
        intervals = {}
        for transfer_id, record in predictions.items():
            profile = rng.choice(profiles[transfer_id])
            expected = float(record["surface_predicted_penalty_bits_per_token"])
            values = [expected + profile[index] for index in banked_indices]
            intervals[transfer_id] = exact_bootstrap_ci(values)
        if all(math.isfinite(bound) for interval in intervals.values() for bound in interval):
            evaluable += 1
        if all(interval[0] >= threshold for interval in intervals.values()):
            confirmed += 1
        elif all(interval[1] <= -threshold for interval in intervals.values()):
            reversed_count += 1
        else:
            null_count += 1
    return {
        "simulations": SIMULATIONS,
        "rng_seed": RNG_SEED,
        "fresh_seed_count": FRESH_SEED_COUNT,
        "paired_bootstrap_support_size": FRESH_SEED_COUNT**FRESH_SEED_COUNT,
        "P_eval": evaluable / SIMULATIONS,
        "P_PENALTY_CONFIRMED_under_surface_prediction": confirmed / SIMULATIONS,
        "P_PENALTY_NULL_under_surface_prediction": null_count / SIMULATIONS,
        "P_PENALTY_REVERSED_under_surface_prediction": reversed_count / SIMULATIONS,
        "mandatory_P_eval_bar": MANDATORY_P_EVAL,
        "clears_mandatory_bar": evaluable / SIMULATIONS >= MANDATORY_P_EVAL,
        "noise_profile_rule": (
            "for each directed contrast use all banked target-horizon raw-rate pairs "
            "whose absolute log2 separation is closest to the exact V10 mismatch; "
            "T40 conservatively donates T20 profiles"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g4c-readout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha256_file(args.g4c_readout) != SOURCE_READOUT_SHA256:
        raise SystemExit("G4C source readout hash mismatch")
    readout = read_json(args.g4c_readout)
    noise = all_contrast_noise(readout)
    predictions = predicted_penalties(readout)
    minimum_prediction = min(
        float(record["surface_predicted_penalty_bits_per_token"])
        for record in predictions.values()
    )
    threshold = 0.5 * minimum_prediction
    feasibility = simulate(readout, predictions, threshold)
    if not feasibility["clears_mandatory_bar"]:
        raise SystemExit("V10 GATESIM did not clear mandatory P_eval >= 0.80")
    report = {
        "schema": "yeto_outer_mup_v10_freshtransfer_gatesim_v1",
        "created_at_utc": utc_now(),
        "pre_outcome": True,
        "source": {
            "g4c_readout_path": str(args.g4c_readout.resolve()),
            "g4c_readout_sha256": SOURCE_READOUT_SHA256,
            "v10_results_observed": False,
        },
        "exact_prescriptions": {
            "eta_T5": ETA_T5,
            "eta_T20": ETA_T20,
            "eta_T40": ETA_T40,
            "eta_T40_rule": "0.003800560784723474 * 0.1 * 1.044905865516022",
        },
        "surface_predicted_penalties": predictions,
        "gate": {
            "name": "G10",
            "closed_vocabulary": [
                "PENALTY_CONFIRMED",
                "PENALTY_NULL",
                "PENALTY_REVERSED",
            ],
            "penalty_threshold_bits_per_token": threshold,
            "threshold_rule": (
                "one half of the minimum of the three prospectively computed "
                "surface-predicted directed penalties"
            ),
            "minimum_surface_predicted_penalty_bits_per_token": minimum_prediction,
            "bootstrap": "all 3^3 ordered paired resamples; equal-tailed 95% interval",
            "classification": {
                "PENALTY_CONFIRMED": "every directed interval lower endpoint >= +threshold",
                "PENALTY_REVERSED": "every directed interval upper endpoint <= -threshold",
                "PENALTY_NULL": "all remaining complete finite outcomes",
            },
            "not_evaluable_forbidden": True,
        },
        "banked_v4_seed_noise": noise,
        "feasibility": feasibility,
    }
    write_json_atomic(args.output.resolve(), report)
    print(
        f"V10 GATESIM P_eval={feasibility['P_eval']:.6f} "
        f"P_confirm={feasibility['P_PENALTY_CONFIRMED_under_surface_prediction']:.6f} "
        f"tau={threshold:.9f} bits"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
