#!/usr/bin/env python3
"""Frozen G4C five-seed combined-grid analysis.

G4C preserves every v4/v4b endpoint and adds seeds 541 and 547 at each of
the 22 registered combined-grid eta values.  Quadratics are fit to the
five-seed mean at each eta.  The paired bootstrap resamples five training
seeds with one common index draw across every eta and all four curve refits.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

try:
    import analyze_v4b as base
except ModuleNotFoundError:  # package import in tests
    from scripts import analyze_v4b as base


ORIGINAL_SEEDS = (501, 503, 509)
ADDED_SEEDS = (541, 547)
ALL_SEEDS = ORIGINAL_SEEDS + ADDED_SEEDS
S_GRID = (2560, 10240)
T_BY_S = {2560: 5, 10240: 20}
MU_GRID = (0.0, 0.9)
MU_HIGH = 0.9
V4_EXPECTED_CELLS = 48
V4B_EXPECTED_CELLS = 18
V4C_EXPECTED_CELLS = 44
COMBINED_EXPECTED_CELLS = 110
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260726
MIN_VALID_BOOTSTRAP_REPLICATES = 9_500
D_BANDS = {5: (1.7, 3.2), 20: (0.8, 1.5)}

COMBINED_ETA_GRIDS = {
    (2560, 0.0): (
        0.017983712258967423,
        0.02543280977844702,
        0.035967424517934846,
        0.05086561955689404,
    ),
    (2560, 0.9): (
        0.0019188620980318239,
        0.0027136808033602967,
        0.0038377241960636478,
        0.005427361606720593,
        0.0076754483921272956,
        0.010854723213441187,
    ),
    (10240, 0.0): (
        0.00491202975922135,
        0.006946659104271081,
        0.0098240595184427,
        0.013893318208542163,
        0.0196481190368854,
        0.027786636417084325,
    ),
    (10240, 0.9): (
        0.0005516209419605576,
        0.0007801098174096424,
        0.0011032418839211152,
        0.0015602196348192849,
        0.0022064837678422303,
        0.0031204392696385697,
    ),
}


def curve_fit(
    losses: dict[tuple[int, float, int, float], float],
    s: int,
    mu: float,
    sampled_indices: list[int] | None = None,
) -> dict:
    registered_etas = list(COMBINED_ETA_GRIDS[(s, mu)])
    observed_etas = sorted(
        key[3] for key in losses if key[0] == s and key[1] == mu
    )
    observed_etas = sorted(set(observed_etas))
    if observed_etas != registered_etas:
        raise base.AnalysisError(
            f"S{s}/mu{mu}: combined eta grid differs: {observed_etas}"
        )
    selected_seeds = (
        [ALL_SEEDS[index] for index in sampled_indices]
        if sampled_indices is not None
        else list(ALL_SEEDS)
    )
    means = []
    for eta in registered_etas:
        values = []
        for seed in selected_seeds:
            key = (s, mu, seed, eta)
            if key not in losses:
                raise base.AnalysisError(f"S{s}/mu{mu}: missing {key}")
            values.append(losses[key])
        means.append(sum(values) / len(values))
    fit = base.fit_quadratic(registered_etas, means)
    fit.update(
        {
            "s": s,
            "t": T_BY_S[s],
            "mu": mu,
            "etas": registered_etas,
            "point_count": len(registered_etas),
            "seeds": list(ALL_SEEDS),
            "seed_mean_losses": means,
        }
    )
    return fit


def d_from_fits(fit0: dict, fit9: dict) -> float | None:
    if not fit0.get("interior") or not fit9.get("interior"):
        return None
    return (fit9["eta_star"] / fit0["eta_star"]) / (1.0 - MU_HIGH)


def bootstrap_all(
    losses: dict[tuple[int, float, int, float], float],
) -> dict:
    rng = random.Random(BOOTSTRAP_SEED)
    log2_d5_samples = []
    log2_d20_samples = []
    gap_samples = []
    invalid = 0
    for _ in range(BOOTSTRAP_REPLICATES):
        draw = [rng.randrange(len(ALL_SEEDS)) for _ in ALL_SEEDS]
        try:
            fits = {
                (s, mu): curve_fit(losses, s, mu, draw)
                for s in S_GRID
                for mu in MU_GRID
            }
            d5 = d_from_fits(fits[(2560, 0.0)], fits[(2560, 0.9)])
            d20 = d_from_fits(fits[(10240, 0.0)], fits[(10240, 0.9)])
            if d5 is None or d20 is None or d5 <= 0 or d20 <= 0:
                invalid += 1
                continue
            log2_d5 = math.log2(d5)
            log2_d20 = math.log2(d20)
            log2_d5_samples.append(log2_d5)
            log2_d20_samples.append(log2_d20)
            gap_samples.append(log2_d5 - log2_d20)
        except base.AnalysisError:
            invalid += 1

    def ratio_interval(samples: list[float]) -> dict:
        low = base.quantile(samples, 0.025) if samples else None
        high = base.quantile(samples, 0.975) if samples else None
        return {
            "coordinate": "log2_D",
            "ci_95_log2_D": {"low": low, "high": high},
            "ci_95_D": {
                "low": 2.0**low if low is not None else None,
                "high": 2.0**high if high is not None else None,
            },
        }

    status = (
        "VALID"
        if len(gap_samples) >= MIN_VALID_BOOTSTRAP_REPLICATES
        else "NOT_EVALUABLE"
    )
    return {
        "method": (
            "paired_nonparametric_seed_curve_bootstrap_"
            "refitting_all_four_five_seed_curves"
        ),
        "pairing": (
            "one shared five-index resample is used for every eta and "
            "all four curves"
        ),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "minimum_valid_replicates": MIN_VALID_BOOTSTRAP_REPLICATES,
        "minimum_valid_fraction": 0.95,
        "valid_replicates": len(gap_samples),
        "invalid_unbracketed_replicates": invalid,
        "status": status,
        "D5": ratio_interval(log2_d5_samples),
        "D20": ratio_interval(log2_d20_samples),
        "monotone_gap": {
            "coordinate": "log2_D_T5_minus_log2_D_T20",
            "point_mean_valid_replicates": (
                sum(gap_samples) / len(gap_samples) if gap_samples else None
            ),
            "ci_95": {
                "low": base.quantile(gap_samples, 0.025) if gap_samples else None,
                "high": base.quantile(gap_samples, 0.975) if gap_samples else None,
            },
        },
    }


def validate_manifests(v4: dict, v4b: dict, v4c: dict, args: argparse.Namespace) -> None:
    if (
        v4.get("schema") != "yeto_outer_mup_v4_scale_launch_manifest_v1"
        or v4.get("stage") != "V4_SCALE"
        or len(v4.get("cells", [])) != V4_EXPECTED_CELLS
    ):
        raise SystemExit("v4 manifest is not the complete 48-cell stage")
    if (
        v4b.get("schema")
        != "yeto_outer_mup_v4b_extension_launch_manifest_v1"
        or v4b.get("stage") != "V4B_EXTENSION"
        or len(v4b.get("cells", [])) != V4B_EXPECTED_CELLS
    ):
        raise SystemExit("v4b manifest is not the complete 18-cell stage")
    if (
        v4c.get("schema")
        != "yeto_outer_mup_v4c_seedpower_launch_manifest_v1"
        or v4c.get("stage") != "V4C_SEED_POWER"
        or len(v4c.get("cells", [])) != V4C_EXPECTED_CELLS
    ):
        raise SystemExit("v4c manifest is not the complete 44-cell stage")
    bindings = v4c.get("base_evidence", {})
    if bindings.get("v4_manifest_sha256") != base.sha256_file(args.v4_manifest):
        raise SystemExit("v4c is not bound to the supplied v4 manifest")
    if bindings.get("v4b_manifest_sha256") != base.sha256_file(args.v4b_manifest):
        raise SystemExit("v4c is not bound to the supplied v4b manifest")
    if v4b.get("base_v4", {}).get("manifest_sha256") != base.sha256_file(
        args.v4_manifest
    ):
        raise SystemExit("v4b is not bound to the supplied v4 manifest")


def display(value: float | None, interval: dict) -> str:
    low = interval.get("low")
    high = interval.get("high")
    if value is None or not isinstance(low, (int, float)) or not isinstance(
        high, (int, float)
    ):
        return "NA [NA,NA]"
    return f"{value:.6f} [{low:.6f},{high:.6f}]"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-manifest", type=Path, required=True)
    parser.add_argument("--v4b-manifest", type=Path, required=True)
    parser.add_argument("--v4c-manifest", type=Path, required=True)
    parser.add_argument("--v4-node-root", action="append", required=True)
    parser.add_argument("--v4b-node-root", action="append", required=True)
    parser.add_argument("--v4c-node-root", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    v4_manifest = base.read_json(args.v4_manifest)
    v4b_manifest = base.read_json(args.v4b_manifest)
    v4c_manifest = base.read_json(args.v4c_manifest)
    validate_manifests(v4_manifest, v4b_manifest, v4c_manifest, args)

    stages = (
        (
            "v4",
            v4_manifest,
            base.parse_node_roots(args.v4_node_root),
            V4_EXPECTED_CELLS,
        ),
        (
            "v4b",
            v4b_manifest,
            base.parse_node_roots(args.v4b_node_root),
            V4B_EXPECTED_CELLS,
        ),
        (
            "v4c",
            v4c_manifest,
            base.parse_node_roots(args.v4c_node_root),
            V4C_EXPECTED_CELLS,
        ),
    )
    stage_losses = {}
    stage_records = {}
    evidence_errors = []
    for name, manifest, roots, _expected in stages:
        losses, records, errors = base.load_losses(manifest, roots, name)
        stage_losses[name] = losses
        stage_records[name] = records
        evidence_errors.extend(errors)

    names = [stage[0] for stage in stages]
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = sorted(set(stage_losses[left]).intersection(stage_losses[right]))
            if overlap:
                evidence_errors.append(f"{left}/{right} coordinate overlap: {overlap}")

    losses = {}
    for name in names:
        losses.update(stage_losses[name])
    complete_evidence = (
        not evidence_errors
        and all(
            len(stage_losses[name]) == expected
            for name, _manifest, _roots, expected in stages
        )
        and len(losses) == COMBINED_EXPECTED_CELLS
    )

    curves = []
    curve_map = {}
    for s in S_GRID:
        for mu in MU_GRID:
            try:
                fit = curve_fit(losses, s, mu)
            except base.AnalysisError as exc:
                fit = {
                    "s": s,
                    "t": T_BY_S[s],
                    "mu": mu,
                    "status": "INVALID_INPUT",
                    "interior": False,
                    "eta_star": None,
                    "error": str(exc),
                }
            curves.append(fit)
            curve_map[(s, mu)] = fit

    d5 = d_from_fits(curve_map[(2560, 0.0)], curve_map[(2560, 0.9)])
    d20 = d_from_fits(curve_map[(10240, 0.0)], curve_map[(10240, 0.9)])
    bootstrap = (
        bootstrap_all(losses)
        if complete_evidence
        else {
            "status": "NOT_EVALUABLE",
            "error": "complete valid v4, v4b, and v4c evidence is required",
            "valid_replicates": 0,
            "D5": {"ci_95_D": {"low": None, "high": None}},
            "D20": {"ci_95_D": {"low": None, "high": None}},
            "monotone_gap": {"ci_95": {"low": None, "high": None}},
        }
    )

    all_interior = all(
        curve_map[(s, mu)].get("interior") for s in S_GRID for mu in MU_GRID
    )
    bands = {
        "T5": d5 is not None and D_BANDS[5][0] <= d5 <= D_BANDS[5][1],
        "T20": d20 is not None and D_BANDS[20][0] <= d20 <= D_BANDS[20][1],
    }
    gap_low = bootstrap.get("monotone_gap", {}).get("ci_95", {}).get("low")
    monotone = (
        bootstrap.get("status") == "VALID"
        and isinstance(gap_low, (int, float))
        and gap_low > 0.0
    )
    evaluable = complete_evidence and all_interior and bootstrap.get("status") == "VALID"
    if not evaluable:
        verdict = "NOT_EVALUABLE"
    elif bands["T5"] and bands["T20"] and monotone:
        verdict = "PASS"
    else:
        verdict = "FAIL"

    d5_ci = bootstrap.get("D5", {}).get("ci_95_D", {})
    d20_ci = bootstrap.get("D20", {}).get("ci_95_D", {})
    note_line = (
        f"G4C VERDICT: {verdict} "
        f"D5={display(d5, d5_ci)} D20={display(d20, d20_ci)}"
    )
    readout = {
        "schema": "yeto_outer_mup_v4c_g4c_readout_v1",
        "created_at_utc": base.utc_now(),
        "v4_manifest_path": str(args.v4_manifest.resolve()),
        "v4_manifest_sha256": base.sha256_file(args.v4_manifest),
        "v4b_manifest_path": str(args.v4b_manifest.resolve()),
        "v4b_manifest_sha256": base.sha256_file(args.v4b_manifest),
        "v4c_manifest_path": str(args.v4c_manifest.resolve()),
        "v4c_manifest_sha256": base.sha256_file(args.v4c_manifest),
        "source_git_commit": v4c_manifest.get("source", {}).get("git_commit"),
        "seeds": list(ALL_SEEDS),
        "expected_cells": {
            "v4": V4_EXPECTED_CELLS,
            "v4b": V4B_EXPECTED_CELLS,
            "v4c": V4C_EXPECTED_CELLS,
            "combined": COMBINED_EXPECTED_CELLS,
        },
        "observed_completed_cells": {
            **{name: len(stage_records[name]) for name in names},
            "combined": sum(len(stage_records[name]) for name in names),
        },
        "evidence_errors": evidence_errors,
        "cell_records": sum((stage_records[name] for name in names), []),
        "curve_fits": curves,
        "D_obs": {"T5": d5, "T20": d20},
        "bootstrap": bootstrap,
        "gate": {
            "name": "G4C",
            "verdict": verdict,
            "conditions": {
                "all_four_five_seed_combined_grid_optima_interior": all_interior,
                "D5_in_1.7_to_3.2": bands["T5"],
                "D20_in_0.8_to_1.5": bands["T20"],
                "D5_greater_than_D20_paired_bootstrap_ci_excludes_zero": monotone,
                "minimum_9500_valid_shared_bootstrap_refits": (
                    bootstrap.get("status") == "VALID"
                ),
            },
            "interpretation": (
                "G4B bands and paired monotonicity criterion applied to "
                "the prospectively completed five-seed combined grids"
            ),
        },
        "note_line": note_line,
    }
    base.write_json_atomic(args.output.resolve(), readout)
    print(note_line)
    return 0 if verdict in ("PASS", "FAIL") else 2


if __name__ == "__main__":
    raise SystemExit(main())
