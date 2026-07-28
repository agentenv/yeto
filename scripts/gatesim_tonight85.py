#!/usr/bin/env python3
"""Pre-outcome feasibility simulation for G11/G12/G13.

The simulator uses only frozen v6/G4C endpoint residual profiles.  It measures
bracketing and sampling-noise feasibility; it does not assume that a genuinely
new model family or far-horizon coordinate obeys the scientific transport
rule.  That distinction is recorded in the output contract.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from pathlib import Path

from tonight85_analysis import fit_quadratic, quantile, sha256_file, write_json_atomic


RNG_SEED = 20_260_727_85
REPLICATES = 2_000
MU = 0.9
ONE_MINUS_MU = 0.1
SCAN_T = (2, 5, 20)
V12_OFFSETS = (-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0)
V13_OFFSETS = (-1.5, -0.5, 0.5, 1.5)
V11_ANCHOR_OFFSETS = (-0.75, 0.0, 0.75)
V11_TRUTH_OFFSETS = (-1.0, -0.5, 0.0, 0.5, 1.0)
RAW_SURFACE_COEFFICIENTS = (
    1.3489008177233357,
    -0.9098513603667141,
    -0.10723867757601385,
    0.16020840966569636,
)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def surface_log2_d(t: int, s: int) -> float:
    gamma, alpha, beta, epsilon = RAW_SURFACE_COEFFICIENTS
    u = (t - 5.0) / 5.0
    v = math.log2(s / 5120.0)
    return gamma + alpha * u + beta * v + epsilon * u * u


def ratio_transport_d(t: int) -> float:
    """Shape-preserving far-horizon extrapolation of the fitted surface.

    F3 itself is not evaluated outside its T<=20 training domain: its u^2 term
    turns upward.  The registered rule extracts D-1 at the two constant-H=512
    donor points T=5 and T=10, fits the unique power law in T, and approaches
    the theory limit D=1 from above.
    """

    d5 = 2.0 ** surface_log2_d(5, 2560)
    d10 = 2.0 ** surface_log2_d(10, 5120)
    exponent = math.log2((d10 - 1.0) / (d5 - 1.0))
    return 1.0 + (d5 - 1.0) * (t / 5.0) ** exponent


def curve(
    readout: dict, *, t: int, s: int, arm: str | None = None, mu: float | None = None
) -> dict:
    candidates = [
        record
        for record in readout["curve_fits"]
        if int(record["t"]) == t and int(record["s"]) == s
    ]
    if arm is not None:
        candidates = [record for record in candidates if record.get("arm") == arm]
    if mu is not None:
        candidates = [record for record in candidates if float(record.get("mu")) == mu]
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one donor curve T{t}/S{s}/{arm}/{mu}, got {len(candidates)}"
        )
    return candidates[0]


def residual_profiles(
    readout: dict, donor: dict, *, arm: str | None = None, mu: float | None = None
) -> list[tuple[list[float], list[float]]]:
    etas = [float(value) for value in donor["etas"]]
    means = [float(value) for value in donor["seed_mean_losses"]]
    records = [
        record
        for record in readout["cell_records"]
        if int(record["t"]) == int(donor["t"]) and int(record["s"]) == int(donor["s"])
    ]
    if arm is not None:
        records = [record for record in records if record.get("arm") == arm]
    if mu is not None:
        records = [record for record in records if float(record.get("mu")) == mu]
    seeds = sorted({int(record["seed"]) for record in records})
    profiles = []
    for seed in seeds:
        by_eta = {
            float(record["eta"]): float(record["eval_loss"])
            for record in records
            if int(record["seed"]) == seed
        }
        if set(by_eta) != set(etas):
            raise RuntimeError("donor residual profile lacks registered eta values")
        profiles.append(
            (
                [math.log2(eta) for eta in etas],
                [by_eta[eta] - mean for eta, mean in zip(etas, means)],
            )
        )
    return profiles


def interpolate(profile: tuple[list[float], list[float]], x: float) -> float:
    xs, ys = profile
    if x <= xs[0]:
        left, right = 0, 1
    elif x >= xs[-1]:
        left, right = len(xs) - 2, len(xs) - 1
    else:
        right = next(index for index, value in enumerate(xs) if value >= x)
        left = right - 1
    weight = (x - xs[left]) / (xs[right] - xs[left])
    return ys[left] * (1.0 - weight) + ys[right] * weight


def simulate_curve(
    rng: random.Random,
    donor: dict,
    profiles: list[tuple[list[float], list[float]]],
    center: float,
    offsets: tuple[float, ...],
    seeds: int,
) -> tuple[list[float], list[list[float]]]:
    xs = [math.log2(center) + offset for offset in offsets]
    etas = [2.0**x for x in xs]
    curvature = float(donor["a"])
    outcomes = []
    for _ in range(seeds):
        profile = rng.choice(profiles)
        outcomes.append(
            [
                2.0 + curvature * (x - math.log2(center)) ** 2 + interpolate(profile, x)
                for x in xs
            ]
        )
    return etas, outcomes


def fit_outcomes(
    etas: list[float], outcomes: list[list[float]], draw: tuple[int, ...] | None = None
) -> dict:
    chosen = outcomes if draw is None else [outcomes[index] for index in draw]
    means = [
        sum(row[index] for row in chosen) / len(chosen) for index in range(len(etas))
    ]
    return fit_quadratic(etas, means)


def scan_gate(
    simulated: dict[int, dict[str, tuple[list[float], list[list[float]]]]],
) -> tuple[bool, bool]:
    point = {}
    for t in SCAN_T:
        mu0 = fit_outcomes(*simulated[t]["mu0"])
        mu09 = fit_outcomes(*simulated[t]["mu09"])
        if not mu0["accepted"] or not mu09["accepted"]:
            return False, False
        point[t] = math.log2(mu09["eta_star"] / mu0["eta_star"] / ONE_MINUS_MU)
    point_monotone = point[2] > point[5] > point[20]
    differences = {(2, 5): [], (5, 20): []}
    for draw in itertools.product(range(3), repeat=3):
        values = {}
        valid = True
        for t in SCAN_T:
            mu0 = fit_outcomes(*simulated[t]["mu0"], draw=draw)
            mu09 = fit_outcomes(*simulated[t]["mu09"], draw=draw)
            if not mu0["accepted"] or not mu09["accepted"]:
                valid = False
                break
            values[t] = math.log2(mu09["eta_star"] / mu0["eta_star"] / ONE_MINUS_MU)
        if valid:
            for pair in differences:
                differences[pair].append(values[pair[0]] - values[pair[1]])
    evaluable = all(len(values) >= 21 for values in differences.values())
    ci_monotone = evaluable and all(
        quantile(values, 0.025) > 0 for values in differences.values()
    )
    return evaluable, point_monotone and ci_monotone


def gatesim_scans(g6: dict, rng: random.Random) -> dict:
    donors = {
        (t, arm): curve(g6, t=t, s=2560, arm=arm)
        for t in SCAN_T
        for arm in ("mu0", "raw")
    }
    profiles = {
        key: residual_profiles(g6, donor, arm=key[1]) for key, donor in donors.items()
    }
    mu0_centers = {t: float(donors[(t, "mu0")]["eta_star"]) for t in SCAN_T}
    v12_raw_centers = {}
    v13_raw_centers = {}
    for t in SCAN_T:
        nesterov_d = 2.0 ** surface_log2_d(t, 2560)
        heavy = (1.0 - MU**t) / (1.0 - MU)
        nesterov = (1.0 - MU ** (t + 1)) / (1.0 - MU)
        v12_raw_centers[t] = (
            mu0_centers[t] * ONE_MINUS_MU * nesterov_d * nesterov / heavy
        )
        v13_raw_centers[t] = mu0_centers[t] * ONE_MINUS_MU * nesterov_d

    counters = {
        "v12": {"evaluable": 0, "pass": 0},
        "v13": {"evaluable": 0, "pass": 0},
    }
    for _ in range(REPLICATES):
        v12 = {}
        for t in SCAN_T:
            v12[t] = {
                "mu0": simulate_curve(
                    rng,
                    donors[(t, "mu0")],
                    profiles[(t, "mu0")],
                    mu0_centers[t],
                    V12_OFFSETS,
                    3,
                ),
                "mu09": simulate_curve(
                    rng,
                    donors[(t, "raw")],
                    profiles[(t, "raw")],
                    v12_raw_centers[t],
                    V12_OFFSETS,
                    3,
                ),
            }
        evaluable, passed = scan_gate(v12)
        counters["v12"]["evaluable"] += int(evaluable)
        counters["v12"]["pass"] += int(evaluable and passed)

        # Family/data feasibility stress: a shared LR-scale shift plus a
        # modest independent D shift at each horizon.  These are prospective
        # placement stresses, not assumptions used by the real G13 analyzer.
        shared_shift = rng.gauss(0.0, 0.35)
        v13 = {}
        for t in SCAN_T:
            ratio_shift = rng.gauss(0.0, 0.10)
            v13[t] = {
                "mu0": simulate_curve(
                    rng,
                    donors[(t, "mu0")],
                    profiles[(t, "mu0")],
                    mu0_centers[t] * 2.0**shared_shift,
                    V13_OFFSETS,
                    3,
                ),
                "mu09": simulate_curve(
                    rng,
                    donors[(t, "raw")],
                    profiles[(t, "raw")],
                    v13_raw_centers[t] * 2.0 ** (shared_shift + ratio_shift),
                    V13_OFFSETS,
                    3,
                ),
            }
        evaluable, passed = scan_gate(v13)
        counters["v13"]["evaluable"] += int(evaluable)
        counters["v13"]["pass"] += int(evaluable and passed)

    return {
        "centers": {
            "mu0": {f"T{t}": mu0_centers[t] for t in SCAN_T},
            "v12_heavy_ball_mu09": {f"T{t}": v12_raw_centers[t] for t in SCAN_T},
            "v13_nesterov_mu09": {f"T{t}": v13_raw_centers[t] for t in SCAN_T},
        },
        "offsets": {"v12": list(V12_OFFSETS), "v13": list(V13_OFFSETS)},
        "v12": {
            **counters["v12"],
            "P_evaluable": counters["v12"]["evaluable"] / REPLICATES,
            "P_pass_unconditional": counters["v12"]["pass"] / REPLICATES,
            "P_pass_given_evaluable": counters["v12"]["pass"]
            / max(counters["v12"]["evaluable"], 1),
        },
        "v13": {
            **counters["v13"],
            "P_evaluable": counters["v13"]["evaluable"] / REPLICATES,
            "P_pass_unconditional": counters["v13"]["pass"] / REPLICATES,
            "P_pass_given_evaluable": counters["v13"]["pass"]
            / max(counters["v13"]["evaluable"], 1),
            "placement_stress": {
                "shared_log2_eta_sd_bits": 0.35,
                "per_horizon_log2_D_sd_bits": 0.10,
            },
        },
    }


def extrapolate_mu0_135(g6: dict) -> float:
    values = [curve(g6, t=t, s=512 * t, arm="mu0") for t in (5, 10, 20)]
    xs = [math.log2(record["t"]) for record in values]
    ys = [math.log2(record["eta_star"]) for record in values]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / sum(
        (x - x_mean) ** 2 for x in xs
    )
    intercept = y_mean - slope * x_mean
    return 2.0 ** (intercept + slope * math.log2(80))


def extrapolate_mu0_1p7(g4c: dict) -> float:
    low = curve(g4c, t=5, s=2560, mu=0.0)
    high = curve(g4c, t=20, s=10240, mu=0.0)
    slope = math.log2(high["eta_star"] / low["eta_star"]) / math.log2(20 / 5)
    return high["eta_star"] * 2.0**slope


def simulate_v11_coordinate(
    rng: random.Random,
    anchor_donor: dict,
    anchor_profiles: list,
    raw_donor: dict,
    raw_profiles: list,
    placement: float,
    t: int,
    model_error: float,
) -> tuple[bool, float | None]:
    anchor = simulate_curve(
        rng, anchor_donor, anchor_profiles, placement, V11_ANCHOR_OFFSETS, 1
    )
    anchor_fit = fit_outcomes(*anchor)
    if not anchor_fit["accepted"]:
        return False, None
    d = ratio_transport_d(t)
    prediction = anchor_fit["eta_star"] * ONE_MINUS_MU * d
    truth = placement * ONE_MINUS_MU * d * 2.0**model_error
    ground = simulate_curve(
        rng,
        raw_donor,
        raw_profiles,
        truth,
        tuple(math.log2(prediction / truth) + offset for offset in V11_TRUTH_OFFSETS),
        2,
    )
    ground_fit = fit_outcomes(*ground)
    if not ground_fit["accepted"]:
        return False, None
    return True, abs(math.log2(prediction / ground_fit["eta_star"]))


def gatesim_v11(g6: dict, g4c: dict, coefficients: dict, rng: random.Random) -> dict:
    anchor_135 = curve(g6, t=20, s=10240, arm="mu0")
    raw_135 = curve(g6, t=20, s=10240, arm="raw")
    anchor_1p7 = curve(g4c, t=20, s=10240, mu=0.0)
    raw_1p7 = curve(g4c, t=20, s=10240, mu=0.9)
    profiles = {
        "anchor_135": residual_profiles(g6, anchor_135, arm="mu0"),
        "raw_135": residual_profiles(g6, raw_135, arm="raw"),
        "anchor_1p7": residual_profiles(g4c, anchor_1p7, mu=0.0),
        "raw_1p7": residual_profiles(g4c, raw_1p7, mu=0.9),
    }
    residual_population = [
        float(record["residual_bits"])
        for record in coefficients["selected_surfaces"]["raw"]["training_fit"].values()
    ]
    placements = {
        "smollm2_135m_t80": extrapolate_mu0_135(g6),
        "smollm2_1p7b_t40": extrapolate_mu0_1p7(g4c),
    }
    bands = {0.25: 0, 0.35: 0, 0.5: 0}
    evaluable = 0
    for _ in range(REPLICATES):
        eval_135, error_135 = simulate_v11_coordinate(
            rng,
            anchor_135,
            profiles["anchor_135"],
            raw_135,
            profiles["raw_135"],
            placements["smollm2_135m_t80"],
            80,
            rng.choice(residual_population),
        )
        eval_1p7, error_1p7 = simulate_v11_coordinate(
            rng,
            anchor_1p7,
            profiles["anchor_1p7"],
            raw_1p7,
            profiles["raw_1p7"],
            placements["smollm2_1p7b_t40"],
            40,
            rng.choice(residual_population),
        )
        if not (eval_135 and eval_1p7):
            continue
        evaluable += 1
        for band in bands:
            bands[band] += int(error_135 <= band or error_1p7 <= band)
    return {
        "placements": placements,
        "ratio_rule": {
            "D_T40": ratio_transport_d(40),
            "D_T80": ratio_transport_d(80),
            "donors": ["F3(T=5,S=2560)", "F3(T=10,S=5120)"],
            "extrapolation": "OLS-free two-point power law in D-1 versus T, asymptote D=1",
            "raw_F3_not_evaluated_beyond_T20": True,
        },
        "anchor_offsets": list(V11_ANCHOR_OFFSETS),
        "truth_offsets": list(V11_TRUTH_OFFSETS),
        "evaluable": evaluable,
        "P_evaluable": evaluable / REPLICATES,
        "band_candidates": {
            str(band): {
                "passes": passes,
                "P_pass_unconditional": passes / REPLICATES,
                "P_pass_given_evaluable": passes / max(evaluable, 1),
            }
            for band, passes in bands.items()
        },
        "selected_band_bits": 0.35,
        "selection_rule": (
            "0.35 bits is the user-mandated confirmatory band; 0.25 and 0.50 "
            "are retained only as pre-outcome sensitivity diagnostics"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g6", type=Path, required=True)
    parser.add_argument("--g4c", type=Path, required=True)
    parser.add_argument(
        "--coefficients",
        type=Path,
        default=Path("experiment-specs/outer-mup-v9-frozen-coefficients.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    g6 = read(args.g6)
    g4c = read(args.g4c)
    coefficients = read(args.coefficients)
    if (
        sha256_file(args.g6)
        != "7e7e9cd8901520cc8f0ca822151af92c4c6183ee99bf02199bc1452a2763af8c"
    ):
        raise SystemExit("G6 readout hash mismatch")
    if (
        sha256_file(args.g4c)
        != "16bab85dce3a83f42c8b1c91be8d91eb1ec1372cfba572f2ffd9b491044b04aa"
    ):
        raise SystemExit("G4C readout hash mismatch")
    rng = random.Random(RNG_SEED)
    output = {
        "schema": "yeto_tonight85_gate_feasibility_v1",
        "status": "PASS",
        "pre_outcome": True,
        "rng_seed": RNG_SEED,
        "replicates": REPLICATES,
        "source": {
            "g6_path": str(args.g6),
            "g6_sha256": sha256_file(args.g6),
            "g4c_path": str(args.g4c),
            "g4c_sha256": sha256_file(args.g4c),
            "coefficients_path": str(args.coefficients),
            "coefficients_sha256": sha256_file(args.coefficients),
        },
        "scope": (
            "measurement-noise/bracketing feasibility only; no simulation "
            "result is evidence that a new family or far-horizon transport law is true"
        ),
        "G11": gatesim_v11(g6, g4c, coefficients, rng),
        "G12_G13": gatesim_scans(g6, rng),
    }
    write_json_atomic(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
