#!/usr/bin/env python3
"""Prospective, outcome-blind feasibility simulation for both v9 gates.

The simulation is run after v6/G4C are immutable but before any v9 verification
cell.  It transports the flattest measured relevant curvature and the largest
measured per-rung seed noise into each target, then executes the exact two-seed
point fit and paired bootstrap evaluability logic intended for ``analyze_v9``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v9_common import (  # noqa: E402
    V9Error,
    fit_quadratic,
    quantile,
    read_json,
    sha256_file,
    utc_now,
    write_json_atomic,
)


SCHEMA = "yeto_outer_mup_v9_gate_simulation_v1"
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260728
NEAR_BRACKET_ALLOWANCE_BITS = 0.5
MIN_GATE_FEASIBILITY = 0.8
MIN_REGISTERED_BAND_BITS = 0.5
BAND_INCREMENT_BITS = 0.125
DEFAULT_SIMULATIONS = 5_000


def wilson(successes: int, trials: int) -> list[float | None]:
    if not trials:
        return [None, None]
    z = 1.959963984540054
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    return [center - radius, center + radius]


def bootstrap_groups() -> list[tuple[list[int], int]]:
    rng = random.Random(BOOTSTRAP_SEED)
    frequencies: dict[tuple[int, int], int] = {}
    representatives: dict[tuple[int, int], list[int]] = {}
    for _ in range(BOOTSTRAP_DRAWS):
        draw = [rng.randrange(2), rng.randrange(2)]
        counts = (draw.count(0), draw.count(1))
        frequencies[counts] = frequencies.get(counts, 0) + 1
        representatives.setdefault(counts, draw)
    if set(frequencies) != {(0, 2), (1, 1), (2, 0)}:
        raise V9Error("unexpected two-seed bootstrap support")
    return [(representatives[key], frequencies[key]) for key in sorted(frequencies)]


BOOTSTRAP_GROUPS = bootstrap_groups()


def find_curve(
    readout: dict,
    *,
    t: int,
    s: int,
    arm: str | None = None,
    mu: float | None = None,
) -> dict:
    matches = []
    for curve in readout.get("curve_fits", []):
        if int(curve.get("t", -1)) != t or int(curve.get("s", -1)) != s:
            continue
        if arm is not None and curve.get("arm") != arm:
            continue
        if mu is not None and not math.isclose(float(curve.get("mu", math.nan)), mu):
            continue
        matches.append(curve)
    if len(matches) != 1:
        raise V9Error(
            f"expected one curve T{t}/S{s}/arm={arm}/mu={mu}, found {len(matches)}"
        )
    curve = matches[0]
    curvature = curve.get("a")
    eta_star = curve.get("eta_star")
    if (
        not isinstance(curvature, (int, float))
        or not math.isfinite(curvature)
        or curvature <= 0
        or not isinstance(eta_star, (int, float))
        or not math.isfinite(eta_star)
        or eta_star <= 0
    ):
        raise V9Error("historical curve lacks a finite positive curvature/vertex")
    return curve


def cell_values(
    readout: dict, *, t: int, s: int, eta: float, arm: str | None, mu: float | None
) -> list[float]:
    values = []
    for record in readout.get("cell_records", []):
        if int(record.get("t", -1)) != t or int(record.get("s", -1)) != s:
            continue
        if arm is not None and record.get("arm") != arm:
            continue
        if mu is not None and not math.isclose(float(record.get("mu", math.nan)), mu):
            continue
        if math.isclose(float(record.get("eta", math.nan)), eta, rel_tol=1e-13):
            loss = record.get("eval_loss")
            if not isinstance(loss, (int, float)) or not math.isfinite(loss):
                raise V9Error("historical cell has a nonfinite endpoint loss")
            values.append(float(loss))
    if len(values) < 2:
        raise V9Error(
            f"historical T{t}/S{s}/eta={eta} has only {len(values)} seed values"
        )
    return values


def noise_profile(
    readout: dict,
    curve: dict,
    *,
    arm: str | None,
    mu: float | None,
) -> list[tuple[float, float]]:
    eta_star = float(curve["eta_star"])
    profile = []
    for raw_eta in curve.get("etas", []):
        eta = float(raw_eta)
        values = cell_values(
            readout,
            t=int(curve["t"]),
            s=int(curve["s"]),
            eta=eta,
            arm=arm,
            mu=mu,
        )
        profile.append((math.log2(eta / eta_star), statistics.stdev(values)))
    if len(profile) < 3:
        raise V9Error("historical noise profile has fewer than three eta levels")
    return sorted(profile)


def interpolate_profile(profile: list[tuple[float, float]], offset: float) -> float:
    if offset <= profile[0][0]:
        return profile[0][1]
    if offset >= profile[-1][0]:
        return profile[-1][1]
    for (x0, y0), (x1, y1) in zip(profile, profile[1:]):
        if x0 <= offset <= x1:
            weight = (offset - x0) / (x1 - x0)
            return y0 * (1.0 - weight) + y1 * weight
    raise AssertionError("profile interpolation interval not found")


def conservative_target(
    name: str,
    offsets: list[float],
    sources: list[tuple[str, dict, dict, str | None, float | None]],
) -> dict:
    """Use the flattest curve and largest interpolated SD across sources."""

    curvature = min(float(curve["a"]) for _, _, curve, _, _ in sources)
    profiles = [
        (
            label,
            noise_profile(readout, curve, arm=arm, mu=mu),
        )
        for label, readout, curve, arm, mu in sources
    ]
    rung_sd = [
        max(interpolate_profile(profile, offset) for _, profile in profiles)
        for offset in offsets
    ]
    return {
        "name": name,
        "curvature_a": curvature,
        "rung_sd": rung_sd,
        "sources": [
            {
                "label": label,
                "T": int(curve["t"]),
                "S": int(curve["s"]),
                "arm": arm,
                "mu": mu,
                "curvature_a": float(curve["a"]),
                "noise_profile": [
                    {"offset_log2_from_fitted_vertex": x, "seed_sd_loss": sd}
                    for x, sd in profile
                ],
            }
            for (label, _, curve, arm, mu), (_, profile) in zip(sources, profiles)
        ],
    }


def prepare_priors(v6: dict, g4c: dict) -> dict[str, dict]:
    v6_t10_raw = find_curve(v6, t=10, s=5120, arm="raw")
    v6_t10_corrected = find_curve(v6, t=10, s=5120, arm="corrected")
    v6_t5_mu0 = find_curve(v6, t=5, s=2560, arm="mu0")
    v6_t5_raw = find_curve(v6, t=5, s=2560, arm="raw")
    g4_t5_mu0 = find_curve(g4c, t=5, s=2560, mu=0.0)
    g4_t5_raw = find_curve(g4c, t=5, s=2560, mu=0.9)
    g4_t20_raw = find_curve(g4c, t=20, s=10240, mu=0.9)
    # The narrower dry-run draft failed the mandatory >=0.8 evaluability
    # check once the complete v6/G4C seed noise was available.  These are the
    # smallest prospectively gatesimmed symmetric widths that pass while
    # retaining exactly four 1.7B and three 7B eta levels per target.
    offsets_a = [-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0]
    offsets_b = [-0.75, 0.0, 0.75]
    return {
        "G9A_1P7B": {
            "offsets": offsets_a,
            "targets": {
                "raw": conservative_target(
                    "raw",
                    offsets_a,
                    [
                        ("v6_135m_T10_raw", v6, v6_t10_raw, "raw", None),
                        ("G4C_1p7b_T5_raw", g4c, g4_t5_raw, None, 0.9),
                        ("G4C_1p7b_T20_raw", g4c, g4_t20_raw, None, 0.9),
                    ],
                ),
                "corrected": conservative_target(
                    "corrected",
                    offsets_a,
                    [
                        (
                            "v6_135m_T10_corrected",
                            v6,
                            v6_t10_corrected,
                            "corrected",
                            None,
                        ),
                        ("G4C_1p7b_T5_raw_noise_proxy", g4c, g4_t5_raw, None, 0.9),
                        (
                            "G4C_1p7b_T20_raw_noise_proxy",
                            g4c,
                            g4_t20_raw,
                            None,
                            0.9,
                        ),
                    ],
                ),
            },
        },
        "G9B_7B": {
            "offsets": offsets_b,
            "targets": {
                "mu0": conservative_target(
                    "mu0",
                    offsets_b,
                    [
                        ("v6_135m_T5_mu0", v6, v6_t5_mu0, "mu0", None),
                        ("G4C_1p7b_T5_mu0", g4c, g4_t5_mu0, None, 0.0),
                    ],
                ),
                "raw": conservative_target(
                    "raw",
                    offsets_b,
                    [
                        ("v6_135m_T5_raw", v6, v6_t5_raw, "raw", None),
                        ("G4C_1p7b_T5_raw", g4c, g4_t5_raw, None, 0.9),
                    ],
                ),
            },
        },
    }


def simulate_record(
    *,
    rng: random.Random,
    gate: dict,
    truth_offset_bits: float,
) -> dict:
    offsets = gate["offsets"]
    per_target_losses = {}
    point_fits = {}
    errors = {}
    for name, target in gate["targets"].items():
        seed_rows = [[], []]
        for eta_index, offset in enumerate(offsets):
            mean = (
                2.0 + float(target["curvature_a"]) * (offset - truth_offset_bits) ** 2
            )
            sd = float(target["rung_sd"][eta_index])
            for seed_index in range(2):
                seed_rows[seed_index].append(rng.gauss(mean, sd))
        per_target_losses[name] = seed_rows
        pooled = [
            (seed_rows[0][index] + seed_rows[1][index]) / 2.0
            for index in range(len(offsets))
        ]
        etas = [2.0**offset for offset in offsets]
        fit = fit_quadratic(
            etas,
            pooled,
            near_bracket_allowance_bits=NEAR_BRACKET_ALLOWANCE_BITS,
        )
        point_fits[name] = fit
        errors[name] = fit["vertex_log2_eta"] if fit["accepted"] else None

    valid_count = 0
    for draw, frequency in BOOTSTRAP_GROUPS:
        all_accepted = True
        for name, rows in per_target_losses.items():
            pooled = [
                sum(rows[index][eta_index] for index in draw) / len(draw)
                for eta_index in range(len(offsets))
            ]
            fit = fit_quadratic(
                [2.0**offset for offset in offsets],
                pooled,
                near_bracket_allowance_bits=NEAR_BRACKET_ALLOWANCE_BITS,
            )
            if not fit["accepted"]:
                all_accepted = False
                break
        if all_accepted:
            valid_count += frequency
    return {
        "point_accepted": all(fit["accepted"] for fit in point_fits.values()),
        "strict_point_interior": all(
            fit["strict_interior"] for fit in point_fits.values()
        ),
        "valid_bootstrap_refits": valid_count,
        "signed_error_bits_estimate_minus_prediction": errors,
    }


def choose_threshold(records: list[dict]) -> int | None:
    need = math.ceil(MIN_GATE_FEASIBILITY * len(records) - 1e-12)
    eligible = sorted(
        (
            record["valid_bootstrap_refits"]
            for record in records
            if record["point_accepted"]
        ),
        reverse=True,
    )
    if len(eligible) < need:
        return None
    return (eligible[need - 1] // 100) * 100


def rounded_band(values: list[float]) -> float:
    measured = quantile(values, 0.95)
    rounded = math.ceil((measured - 1e-15) / BAND_INCREMENT_BITS) * BAND_INCREMENT_BITS
    return max(MIN_REGISTERED_BAND_BITS, rounded)


def summarize_gate(gate_id: str, gate: dict, *, simulations: int, seed: int) -> dict:
    rng = random.Random(seed)
    centered = [
        simulate_record(rng=rng, gate=gate, truth_offset_bits=0.0)
        for _ in range(simulations)
    ]
    threshold = choose_threshold(centered)
    if threshold is None:
        evaluable = []
        bands = {name: None for name in gate["targets"]}
    else:
        evaluable = [
            record
            for record in centered
            if record["point_accepted"]
            and record["valid_bootstrap_refits"] >= threshold
        ]
        bands = {
            name: rounded_band(
                [
                    abs(
                        float(
                            record["signed_error_bits_estimate_minus_prediction"][name]
                        )
                    )
                    for record in evaluable
                ]
            )
            for name in gate["targets"]
        }
    passes = [
        record
        for record in evaluable
        if all(
            abs(float(record["signed_error_bits_estimate_minus_prediction"][name]))
            <= float(bands[name])
            for name in gate["targets"]
        )
    ]
    p_eval = len(evaluable) / simulations
    p_pass = len(passes) / len(evaluable) if evaluable else 0.0

    sensitivity = {}
    stress_count = max(500, simulations // 2)
    for truth_offset in (-1.0, -0.5, 0.5, 1.0):
        records = [
            simulate_record(rng=rng, gate=gate, truth_offset_bits=truth_offset)
            for _ in range(stress_count)
        ]
        stress_evaluable = [
            record
            for record in records
            if threshold is not None
            and record["point_accepted"]
            and record["valid_bootstrap_refits"] >= threshold
        ]
        stress_pass = [
            record
            for record in stress_evaluable
            if all(
                abs(float(record["signed_error_bits_estimate_minus_prediction"][name]))
                <= float(bands[name])
                for name in gate["targets"]
            )
        ]
        sensitivity[f"truth_offset_{truth_offset:+g}_bits"] = {
            "simulations": stress_count,
            "P_eval": len(stress_evaluable) / stress_count,
            "P_pass_given_evaluable": (
                len(stress_pass) / len(stress_evaluable) if stress_evaluable else None
            ),
            "P_reject_given_evaluable": (
                1.0 - len(stress_pass) / len(stress_evaluable)
                if stress_evaluable
                else None
            ),
        }

    valid_counts = [record["valid_bootstrap_refits"] for record in centered]
    result = {
        "gate_id": gate_id,
        "simulation_seed": seed,
        "simulations_under_centered_prediction_null": simulations,
        "ladder_offsets_log2": gate["offsets"],
        "seeds_per_eta": 2,
        "near_bracket_allowance_bits": NEAR_BRACKET_ALLOWANCE_BITS,
        "registered_minimum_valid_bootstrap_refits": threshold,
        "registered_absolute_error_band_bits": bands,
        "P_eval": p_eval,
        "P_eval_wilson95": wilson(len(evaluable), simulations),
        "P_pass_given_evaluable_under_centered_null": p_pass,
        "P_pass_given_evaluable_wilson95": wilson(len(passes), len(evaluable)),
        "point_acceptance_fraction": sum(
            record["point_accepted"] for record in centered
        )
        / simulations,
        "strict_point_interior_fraction": sum(
            record["strict_point_interior"] for record in centered
        )
        / simulations,
        "valid_bootstrap_refits": {
            "min": min(valid_counts),
            "q10": quantile(valid_counts, 0.1),
            "median": quantile(valid_counts, 0.5),
            "q90": quantile(valid_counts, 0.9),
            "max": max(valid_counts),
        },
        "sensitivity": sensitivity,
        "target_priors": gate["targets"],
    }
    result["feasibility_pass"] = bool(
        threshold is not None
        and p_eval >= MIN_GATE_FEASIBILITY
        and p_pass >= MIN_GATE_FEASIBILITY
    )
    return result


def run_simulation(
    *,
    v6: dict,
    v6_path: Path,
    selection: dict,
    selection_path: Path,
    g4c: dict,
    g4c_path: Path,
    simulations: int,
) -> dict:
    if simulations < 100:
        raise V9Error("gate simulation requires at least 100 synthetic datasets")
    if v6.get("schema") != "yeto_outer_mup_v6_g6_readout_v1":
        raise V9Error("not a v6 G6 readout")
    if v6.get("gate", {}).get("verdict") != "PASS":
        raise V9Error("v9 gatesim requires immutable G6 PASS evidence")
    if selection.get("schema") != "yeto_outer_mup_v6_selected_surfaces_v1":
        raise V9Error("not a frozen v6 surface selection")
    if selection.get("v6_readout_sha256") != sha256_file(v6_path):
        raise V9Error("frozen selection binds another v6 readout")
    if g4c.get("gate", {}).get("verdict") != "PASS":
        raise V9Error("v9 gatesim requires immutable G4C PASS evidence")
    started = time.monotonic()
    priors = prepare_priors(v6, g4c)
    gates = {
        "G9A_1P7B": summarize_gate(
            "G9A_1P7B",
            priors["G9A_1P7B"],
            simulations=simulations,
            seed=9_101_700,
        ),
        "G9B_7B": summarize_gate(
            "G9B_7B",
            priors["G9B_7B"],
            simulations=simulations,
            seed=9_700_000,
        ),
    }
    status = (
        "PASS" if all(gate["feasibility_pass"] for gate in gates.values()) else "FAIL"
    )
    return {
        "schema": SCHEMA,
        "status": status,
        "created_at_utc": utc_now(),
        "pre_outcome": True,
        "verification_loss_seen": False,
        "source_artifacts": {
            "v6_readout": {
                "path": str(v6_path.resolve()),
                "sha256": sha256_file(v6_path),
            },
            "v6_selection": {
                "path": str(selection_path.resolve()),
                "sha256": sha256_file(selection_path),
            },
            "g4c_readout": {
                "path": str(g4c_path.resolve()),
                "sha256": sha256_file(g4c_path),
            },
        },
        "method": {
            "noise": (
                "independent Gaussian per seed and rung with SD equal to the "
                "largest linearly interpolated measured SD among relevant v6/G4C "
                "profiles"
            ),
            "curvature": "minimum positive measured curvature among the same sources",
            "point_estimator": "OLS quadratic in log2 eta on the two-seed rung mean",
            "bootstrap": (
                "one shared two-index resample across every gate target; all three "
                "multinomial count vectors evaluated with exact frequency in the "
                "registered 10,000-draw PRNG sequence"
            ),
            "threshold_selection": (
                "largest valid-refit threshold, rounded down to 100, retaining at "
                "least 0.8 evaluability under the centered prediction null"
            ),
            "band_selection": (
                "per-target 95th percentile absolute point-estimation error among "
                "evaluable centered-null simulations, rounded up to 0.125 bits, "
                "with a prospective minimum substantive band of 0.5 bits"
            ),
            "limitations": (
                "The simulation calibrates finite-seed curve estimation and gate "
                "evaluability under conservative measured noise/curvature transport; "
                "it cannot assume away model-family transport error, which is the "
                "scientific object tested by G9. Independent rung noise is used "
                "because cross-scale covariance is unidentified."
            ),
        },
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "sample_size": 2,
            "groups": [
                {"representative": draw, "frequency": frequency}
                for draw, frequency in BOOTSTRAP_GROUPS
            ],
        },
        "minimum_required_P_eval": MIN_GATE_FEASIBILITY,
        "minimum_required_P_pass_given_evaluable": MIN_GATE_FEASIBILITY,
        "gates": gates,
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v6-readout", type=Path, required=True)
    parser.add_argument("--v6-selection", type=Path, required=True)
    parser.add_argument("--g4c-readout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing gatesim output: {args.output}")
    try:
        result = run_simulation(
            v6=read_json(args.v6_readout),
            v6_path=args.v6_readout,
            selection=read_json(args.v6_selection),
            selection_path=args.v6_selection,
            g4c=read_json(args.g4c_readout),
            g4c_path=args.g4c_readout,
            simulations=args.simulations,
        )
    except (V9Error, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    output = args.output.resolve()
    write_json_atomic(output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(output),
                "sha256": sha256_file(output),
                "gates": {
                    gate_id: {
                        "P_eval": gate["P_eval"],
                        "P_pass": gate["P_pass_given_evaluable_under_centered_null"],
                        "threshold": gate["registered_minimum_valid_bootstrap_refits"],
                        "bands": gate["registered_absolute_error_band_bits"],
                    }
                    for gate_id, gate in result["gates"].items()
                },
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
