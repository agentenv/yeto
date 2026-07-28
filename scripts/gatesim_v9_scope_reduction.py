#!/usr/bin/env python3
"""Prospective gatesim for an outcome-blind v9 7B seed-panel reduction.

This reuses the sealed v9 G9B curvature/noise priors, quadratic fit,
near-bracket allowance, bootstrap PRNG sequence, 7,500-valid-refit threshold,
and 0.5-bit pass bands.  The only varied input is the symmetric retained-seed
panel at the three already sealed eta rungs.  A singleton rung is resampled as
the only member of its empirical distribution; a two-seed rung uses the same
shared two-index draw as the original frozen simulator.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from gatesim_v9 import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_GROUPS,
    BOOTSTRAP_SEED,
    MIN_GATE_FEASIBILITY,
    NEAR_BRACKET_ALLOWANCE_BITS,
    wilson,
)
from v9_common import fit_quadratic, quantile, sha256_file, utc_now, write_json_atomic


EXPECTED_PREDICTIONS_SHA256 = (
    "97e02dcad63782978ac51b320621e5a681236518cb0d5db19454b8981549ca9c"
)
FIXED_MINIMUM_VALID_REFITS = 7_500
SIMULATION_SEED = 9_700_000
SEEDS = (901, 907)
ETA_INDICES = (0, 1, 2)


LAYOUTS = {
    "symmetric_6_single_seed": {
        0: (907,),
        1: (907,),
        2: (907,),
    },
    "symmetric_8_center_replicated": {
        0: (907,),
        1: (901, 907),
        2: (907,),
    },
    "symmetric_10_center_high_replicated": {
        0: (907,),
        1: (901, 907),
        2: (901, 907),
    },
    "symmetric_12_full_panel": {
        0: (901, 907),
        1: (901, 907),
        2: (901, 907),
    },
}


def retained_mean(
    rows: dict[int, list[float]], eta_index: int, retained: tuple[int, ...]
) -> float:
    values = [rows[seed][eta_index] for seed in retained]
    return sum(values) / len(values)


def bootstrap_mean(
    rows: dict[int, list[float]],
    eta_index: int,
    retained: tuple[int, ...],
    draw: list[int],
) -> float:
    if len(retained) == 1:
        return rows[retained[0]][eta_index]
    selected = [rows[SEEDS[index]][eta_index] for index in draw]
    return sum(selected) / len(selected)


def simulate_record(
    *,
    rng: random.Random,
    priors: dict,
    layout: dict[int, tuple[int, ...]],
    truth: float,
) -> dict:
    point_fits = {}
    errors = {}
    target_rows = {}
    offsets = [float(value) for value in priors["ladder_offsets_log2"]]
    etas = [2.0**offset for offset in offsets]
    for name, target in priors["target_priors"].items():
        rows = {seed: [] for seed in SEEDS}
        for eta_index, offset in enumerate(offsets):
            mean = 2.0 + float(target["curvature_a"]) * (offset - truth) ** 2
            sd = float(target["rung_sd"][eta_index])
            for seed in SEEDS:
                rows[seed].append(rng.gauss(mean, sd))
        target_rows[name] = rows
        pooled = [
            retained_mean(rows, eta_index, layout[eta_index])
            for eta_index in ETA_INDICES
        ]
        fit = fit_quadratic(
            etas, pooled, near_bracket_allowance_bits=NEAR_BRACKET_ALLOWANCE_BITS
        )
        point_fits[name] = fit
        errors[name] = fit["vertex_log2_eta"] if fit["accepted"] else None

    valid = 0
    for draw, frequency in BOOTSTRAP_GROUPS:
        accepted = True
        for name, rows in target_rows.items():
            pooled = [
                bootstrap_mean(rows, eta_index, layout[eta_index], draw)
                for eta_index in ETA_INDICES
            ]
            fit = fit_quadratic(
                etas,
                pooled,
                near_bracket_allowance_bits=NEAR_BRACKET_ALLOWANCE_BITS,
            )
            if not fit["accepted"]:
                accepted = False
                break
        if accepted:
            valid += frequency
    return {
        "point_accepted": all(fit["accepted"] for fit in point_fits.values()),
        "strict_point_interior": all(
            fit["strict_interior"] for fit in point_fits.values()
        ),
        "valid_bootstrap_refits": valid,
        "signed_error_bits_estimate_minus_prediction": errors,
    }


def summarize(
    *, priors: dict, layout: dict[int, tuple[int, ...]], simulations: int
) -> dict:
    rng = random.Random(SIMULATION_SEED)
    centered = [
        simulate_record(rng=rng, priors=priors, layout=layout, truth=0.0)
        for _ in range(simulations)
    ]
    evaluable = [
        record
        for record in centered
        if record["point_accepted"]
        and record["valid_bootstrap_refits"] >= FIXED_MINIMUM_VALID_REFITS
    ]
    bands = {
        name: float(value)
        for name, value in priors["registered_absolute_error_band_bits"].items()
    }
    passed = [
        record
        for record in evaluable
        if all(
            abs(float(record["signed_error_bits_estimate_minus_prediction"][name]))
            <= bands[name]
            for name in bands
        )
    ]
    valid_counts = [record["valid_bootstrap_refits"] for record in centered]
    p_eval = len(evaluable) / simulations
    p_pass = len(passed) / len(evaluable) if evaluable else 0.0
    return {
        "cell_count": sum(len(layout[index]) for index in ETA_INDICES)
        * len(priors["target_priors"]),
        "retained_seeds_by_eta_index_per_arm": {
            str(index): list(layout[index]) for index in ETA_INDICES
        },
        "simulations_under_centered_prediction_null": simulations,
        "simulation_seed": SIMULATION_SEED,
        "point_acceptance_fraction": sum(
            record["point_accepted"] for record in centered
        )
        / simulations,
        "strict_point_interior_fraction": sum(
            record["strict_point_interior"] for record in centered
        )
        / simulations,
        "P_eval": p_eval,
        "P_eval_wilson95": wilson(len(evaluable), simulations),
        "P_pass_given_evaluable": p_pass,
        "P_pass_given_evaluable_wilson95": wilson(len(passed), len(evaluable)),
        "valid_bootstrap_refits": {
            "min": min(valid_counts),
            "q10": quantile(valid_counts, 0.1),
            "median": quantile(valid_counts, 0.5),
            "q90": quantile(valid_counts, 0.9),
            "max": max(valid_counts),
        },
        "feasibility_pass": bool(
            p_eval >= MIN_GATE_FEASIBILITY and p_pass >= MIN_GATE_FEASIBILITY
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-gatesim", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simulations", type=int, default=5_000)
    args = parser.parse_args()
    if args.simulations < 100:
        raise SystemExit("at least 100 simulations are required")
    if sha256_file(args.predictions) != EXPECTED_PREDICTIONS_SHA256:
        raise SystemExit("sealed predictions changed")
    original = json.loads(args.original_gatesim.read_text())
    gate = original["gates"]["G9B_7B"]
    if original.get("status") != "PASS" or not gate.get("feasibility_pass"):
        raise SystemExit("original frozen G9B gatesim is not PASS")
    if (
        gate.get("registered_minimum_valid_bootstrap_refits")
        != FIXED_MINIMUM_VALID_REFITS
    ):
        raise SystemExit("original G9B valid-refit threshold changed")
    started = time.monotonic()
    layouts = {
        name: summarize(priors=gate, layout=layout, simulations=args.simulations)
        for name, layout in LAYOUTS.items()
    }
    selected = next(
        (name for name, result in layouts.items() if result["feasibility_pass"]),
        None,
    )
    result = {
        "schema": "yeto_outer_mup_v9_7b_scope_reduction_gatesim_v1",
        "status": "PASS" if selected is not None else "FAIL",
        "created_at_utc": utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "pre_outcome": True,
        "verification_loss_seen": False,
        "minimum_required_P_eval": MIN_GATE_FEASIBILITY,
        "minimum_required_P_pass_given_evaluable": MIN_GATE_FEASIBILITY,
        "fixed_minimum_valid_bootstrap_refits": FIXED_MINIMUM_VALID_REFITS,
        "fixed_near_bracket_allowance_bits": NEAR_BRACKET_ALLOWANCE_BITS,
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "rng_seed": BOOTSTRAP_SEED,
            "sample_size_at_replicated_rungs": 2,
            "singleton_rung_rule": "reuse the singleton as its empirical distribution",
        },
        "source_artifacts": {
            "original_gatesim": {
                "path": str(args.original_gatesim),
                "sha256": sha256_file(args.original_gatesim),
            },
            "sealed_predictions": {
                "path": str(args.predictions),
                "sha256": sha256_file(args.predictions),
                "untouched": True,
            },
        },
        "candidate_order": list(LAYOUTS),
        "selection_rule": (
            "first symmetric candidate in ascending cell count with both P_eval and "
            "P_pass_given_evaluable at least 0.8 under the centered prediction null"
        ),
        "selected_layout": selected,
        "layouts": layouts,
    }
    write_json_atomic(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_layout": selected,
                "layouts": {
                    name: {
                        "cell_count": value["cell_count"],
                        "P_eval": value["P_eval"],
                        "P_pass_given_evaluable": value["P_pass_given_evaluable"],
                        "feasibility_pass": value["feasibility_pass"],
                    }
                    for name, value in layouts.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if selected is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
