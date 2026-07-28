#!/usr/bin/env python3
"""Prospective bracketing/noise feasibility simulation for the G13B regrid."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from gatesim_tonight85 import (
    ONE_MINUS_MU,
    SCAN_T,
    V13_OFFSETS,
    curve,
    fit_outcomes,
    interpolate,
    residual_profiles,
    scan_gate,
)
from tonight85_analysis import sha256_file, write_json_atomic


RNG_SEED = 20_260_743
REPLICATES = 2_000
G6_SHA256 = "7e7e9cd8901520cc8f0ca822151af92c4c6183ee99bf02199bc1452a2763af8c"

# Each center is exactly the v13 pooled discrete edge selected without any
# cross-arm loss comparison.  Five curves hit low eta; T20/mu0 hit high eta.
CENTERS = {
    (2, "mu0"): 0.025125139710338324,
    (2, "mu09"): 0.01047484475774391,
    (5, "mu0"): 0.015350998708982334,
    (5, "mu09"): 0.004211902420859095,
    (20, "mu0"): 0.06201671160649202,
    (20, "mu09"): 0.0008712255272127002,
}


def simulate_relative_curve(
    rng: random.Random,
    donor: dict,
    profiles: list[tuple[list[float], list[float]]],
    registered_center: float,
    truth_shift_bits: float,
) -> tuple[list[float], list[list[float]]]:
    """Translate donor residual shapes by relative log-eta, not absolute LR."""

    grid_center_x = math.log2(registered_center)
    truth_x = grid_center_x + truth_shift_bits
    donor_center_x = math.log2(float(donor["eta_star"]))
    xs = [grid_center_x + offset for offset in V13_OFFSETS]
    etas = [2.0**x for x in xs]
    curvature = float(donor["a"])
    outcomes = []
    for _ in range(3):
        profile = rng.choice(profiles)
        outcomes.append(
            [
                2.0
                + curvature * (x - truth_x) ** 2
                + interpolate(profile, donor_center_x + (x - truth_x))
                for x in xs
            ]
        )
    return etas, outcomes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g6", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha256_file(args.g6) != G6_SHA256:
        raise SystemExit("G6 donor readout hash mismatch")
    g6 = json.loads(args.g6.read_text())
    donors = {
        (t, arm): curve(g6, t=t, s=2560, arm="mu0" if arm == "mu0" else "raw")
        for t in SCAN_T
        for arm in ("mu0", "mu09")
    }
    profiles = {
        key: residual_profiles(
            g6, donor, arm="mu0" if key[1] == "mu0" else "raw"
        )
        for key, donor in donors.items()
    }
    rng = random.Random(RNG_SEED)
    evaluable = passed = 0
    for _ in range(REPLICATES):
        shared_shift = rng.gauss(0.0, 0.35)
        simulated = {}
        for t in SCAN_T:
            ratio_shift = rng.gauss(0.0, 0.10)
            simulated[t] = {
                "mu0": simulate_relative_curve(
                    rng, donors[(t, "mu0")], profiles[(t, "mu0")], CENTERS[(t, "mu0")], shared_shift
                ),
                "mu09": simulate_relative_curve(
                    rng,
                    donors[(t, "mu09")],
                    profiles[(t, "mu09")],
                    CENTERS[(t, "mu09")],
                    shared_shift + ratio_shift,
                ),
            }
        is_evaluable, is_pass = scan_gate(simulated)
        evaluable += int(is_evaluable)
        passed += int(is_evaluable and is_pass)

    implied_d = {
        f"T{t}": CENTERS[(t, "mu09")] / CENTERS[(t, "mu0")] / ONE_MINUS_MU
        for t in SCAN_T
    }
    output = {
        "schema": "yeto_v13b_gate_feasibility_v1",
        "status": "PASS" if evaluable == REPLICATES else "FAIL",
        "pre_outcome_v13b": True,
        "replicates": REPLICATES,
        "rng_seed": RNG_SEED,
        "scope": (
            "bracketing and sampling-noise feasibility only; v13 edge locations set "
            "the centers, while no v13 arm-comparison estimand or G13B outcome is used"
        ),
        "source": {"g6_path": str(args.g6), "g6_sha256": sha256_file(args.g6)},
        "registered_centers": {
            f"T{t}_{arm}": center for (t, arm), center in sorted(CENTERS.items())
        },
        "offsets_log2": list(V13_OFFSETS),
        "implied_center_D_descriptive": implied_d,
        "placement_stress": {
            "shared_log2_eta_sd_bits": 0.35,
            "per_horizon_mu09_relative_sd_bits": 0.10,
        },
        "evaluable": evaluable,
        "pass": passed,
        "P_evaluable": evaluable / REPLICATES,
        "P_pass_unconditional": passed / REPLICATES,
        "P_pass_given_evaluable": passed / max(evaluable, 1),
    }
    write_json_atomic(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if output["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
