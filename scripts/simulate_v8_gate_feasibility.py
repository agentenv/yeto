#!/usr/bin/env python3
"""Pre-outcome CPU gate-feasibility simulation for outer-muP v8.

The simulation transports the sealed v3 five-seed, eta-cell noise scale to the
registered v8 ladders.  It evaluates the exact v8 strict-interior point rule
and exact 10,000-draw shared paired-seed bootstrap, compressed to the 10
distinct three-index multinomial count vectors.  No GPU work or v8 outcome is
read or produced.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import analyze_v8 as a8  # noqa: E402


N_SIMULATIONS = 500
OUTER_SEED_BASE = 8_080_000
EXPECTED_G3_READOUT_SHA256 = (
    "d4a3cde6aa47580dff255c7a66030ab997a95f4072b1883bf71aa54d7da744c8"
)
EXPECTED_G3_MANIFEST_SHA256 = (
    "8fae6137d673d4c57861b37de09a42c5c462b0dff692cf10bde49e73caa554fc"
)
G3_SEEDS = (301, 311, 313, 317, 331)
G3_CURVATURE_MAGNITUDE_FLOOR = 0.0039
CFG: dict = {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"{path}: expected a JSON object")
    return value


def wilson(successes: int, trials: int) -> list[float]:
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return [center - radius, center + radius]


def draw_groups() -> list[tuple[list[int], int]]:
    rng = random.Random(a8.BOOTSTRAP_SEED)
    frequencies: collections.Counter[tuple[int, ...]] = collections.Counter()
    representatives: dict[tuple[int, ...], list[int]] = {}
    for _ in range(a8.BOOTSTRAP_REPLICATES):
        draw = [rng.randrange(len(a8.SEEDS)) for _ in a8.SEEDS]
        counts = tuple(draw.count(index) for index in range(len(a8.SEEDS)))
        frequencies[counts] += 1
        representatives.setdefault(counts, draw)
    expected_support = math.comb(2 * len(a8.SEEDS) - 1, len(a8.SEEDS) - 1)
    if len(frequencies) != expected_support or sum(frequencies.values()) != 10_000:
        raise RuntimeError("unexpected registered bootstrap support")
    return [
        (representatives[counts], frequencies[counts])
        for counts in sorted(frequencies)
    ]


DRAW_GROUPS = draw_groups()


def g3_curve_name(t: int, source_arm: str, mu: float) -> str:
    return f"{source_arm}_S{512 * t}_mu{mu:g}"


def build_config(contract_path: Path, readout_path: Path, manifest_path: Path) -> dict:
    if sha256_file(readout_path) != EXPECTED_G3_READOUT_SHA256:
        raise RuntimeError("sealed g3 readout hash mismatch")
    if sha256_file(manifest_path) != EXPECTED_G3_MANIFEST_SHA256:
        raise RuntimeError("sealed v3 launch-manifest hash mismatch")
    contract = read_json(contract_path)
    readout = read_json(readout_path)
    manifest = read_json(manifest_path)
    if contract.get("schema") != "yeto_outer_mup_v8_phasediagram_prereg_v1":
        raise RuntimeError("not the v8 phase-diagram contract")
    if len(readout.get("cell_evidence", [])) != 460 or readout.get("invalid_cells"):
        raise RuntimeError("sealed g3 readout lacks the complete measured cell panel")
    if len(manifest.get("cells", [])) != 460:
        raise RuntimeError("sealed v3 manifest is incomplete")

    curves = contract.get("design", {}).get("curves", [])
    if len(curves) != a8.EXPECTED_CURVES:
        raise RuntimeError("v8 contract does not enumerate 15 curves")
    design = {}
    for curve in curves:
        key = a8.curve_key(curve["T"], curve["arm"], curve["mu"])
        if key in design:
            raise RuntimeError(f"duplicate v8 curve {key}")
        if curve["S"] != 512 * curve["T"] or curve["H"] != 512:
            raise RuntimeError(f"v8 curve violates fixed-H closure: {key}")
        if len(curve["eta_grid"]) != a8.ETA_POINTS:
            raise RuntimeError(f"v8 curve is not four-point: {key}")
        design[key] = {
            "etas": [float(value) for value in curve["eta_grid"]],
            "center": float(curve["eta_center"]),
        }
    if set(design) != set(a8.expected_curve_keys()):
        raise RuntimeError("v8 curve coordinates differ from the frozen analyzer")

    losses_by_id = {
        record["cell_id"]: float(record["eval_loss"])
        for record in readout["cell_evidence"]
    }
    measured: dict[tuple[str, int, float, int], dict[int, float]] = {}
    for cell in manifest["cells"]:
        source_arm = str(cell["arm"])
        if source_arm not in ("T", "B"):
            continue
        t = int(cell["t"])
        mu = float(cell["mu"])
        key = (source_arm, t, mu, int(cell["eta_index"]))
        measured.setdefault(key, {})[int(cell["seed"])] = losses_by_id[
            cell["cell_id"]
        ]
    if any(len(values) != 5 for values in measured.values()):
        raise RuntimeError("v3 measured-noise cell lacks five seeds")

    cell_sd = {}
    for key, values in measured.items():
        cell_sd[key] = statistics.stdev(values[seed] for seed in G3_SEEDS)

    curvature = {}
    vertex = {}
    for t in a8.T_GRID:
        for label, source_arm, mu in (
            (a8.BASELINE_ARM, "T", 0.0),
            ("raw", "T", 0.9),
            ("corrected", "B", 0.9),
        ):
            record = readout["eta_curves"][g3_curve_name(t, source_arm, mu)]
            signed = float(record["a"])
            curvature[(t, label)] = max(
                abs(signed), G3_CURVATURE_MAGNITUDE_FLOOR
            )
            raw_vertex = record.get("vertex_log2_eta")
            vertex[(t, label)] = (
                2.0 ** float(raw_vertex)
                if isinstance(raw_vertex, (int, float))
                else None
            )

    target = {}
    target_sds = []
    target_curvatures = []
    for key in a8.expected_curve_keys():
        t, arm, mu = key
        source_arm = "T" if arm in (a8.BASELINE_ARM, "raw") else "B"
        source_mu = 0.0 if arm == a8.BASELINE_ARM else 0.9
        if arm == a8.BASELINE_ARM:
            curve_curvature = curvature[(t, a8.BASELINE_ARM)]
        else:
            weight = min(1.0, max(0.0, mu / 0.9))
            low = curvature[(t, a8.BASELINE_ARM)]
            high = curvature[(t, arm)]
            curve_curvature = math.exp(
                (1.0 - weight) * math.log(low) + weight * math.log(high)
            )
        sds = [
            cell_sd[(source_arm, t, source_mu, eta_index)]
            for eta_index in range(a8.ETA_POINTS)
        ]
        target[key] = {
            **design[key],
            "curvature": curve_curvature,
            "sds": sds,
        }
        target_sds.extend(sds)
        target_curvatures.append(curve_curvature)

    measured_shift_truth = {}
    transfer = contract["center_models"]["momentum_transfer"]
    for t in a8.T_GRID:
        baseline_center = design[a8.curve_key(t, a8.BASELINE_ARM, 0.0)]["center"]
        baseline_vertex = vertex[(t, a8.BASELINE_ARM)]
        baseline_ratio = (
            baseline_vertex / baseline_center if baseline_vertex is not None else 1.0
        )
        measured_shift_truth[a8.curve_key(t, a8.BASELINE_ARM, 0.0)] = (
            baseline_center * baseline_ratio
        )
        for arm in a8.MOMENTUM_ARMS:
            q = (t - 5.0) / 10.0
            if arm == "raw":
                coefficients = transfer["raw_coefficients"]
                log2_d_high = -math.log2(1.0 - 0.9 ** (t + 1)) + (
                    coefficients["b0"] + coefficients["bq"] * q
                )
            else:
                coefficients = transfer["corrected_coefficients"]
                log2_d_high = coefficients["b0"] + coefficients["bq"] * q
            high_center = baseline_center * 0.1 * (2.0**log2_d_high)
            high_vertex = vertex[(t, arm)]
            if high_vertex is None:
                high_ratio = baseline_ratio
            else:
                high_ratio = high_vertex / high_center
            for mu in a8.MU_GRID:
                ratio = math.exp(
                    (1.0 - min(mu / 0.9, 1.0)) * math.log(baseline_ratio)
                    + min(mu / 0.9, 1.0) * math.log(high_ratio)
                )
                center = design[a8.curve_key(t, arm, mu)]["center"]
                measured_shift_truth[a8.curve_key(t, arm, mu)] = center * ratio

    return {
        "target": target,
        "truth": {
            "centered": {key: item["center"] for key, item in target.items()},
            "v3_vertex_shift_sensitivity": measured_shift_truth,
        },
        "source_summary": {
            "g3_readout_sha256": EXPECTED_G3_READOUT_SHA256,
            "g3_manifest_sha256": EXPECTED_G3_MANIFEST_SHA256,
            "measured_seeds": list(G3_SEEDS),
            "transported_cell_sd": {
                "min": min(target_sds),
                "median": statistics.median(target_sds),
                "max": max(target_sds),
            },
            "transported_curvature": {
                "min": min(target_curvatures),
                "median": statistics.median(target_curvatures),
                "max": max(target_curvatures),
                "convexity_rule": (
                    "absolute magnitude of the sealed v3 quadratic coefficient, "
                    "floored at 0.0039; mu interpolation is geometric and clamps "
                    "above mu=0.9"
                ),
            },
        },
        "design_sha256": canonical_sha256(curves),
    }


def init_worker(config: dict) -> None:
    global CFG
    CFG = config


def make_dataset(replicate: int, scenario: str) -> dict:
    rng = random.Random(OUTER_SEED_BASE + replicate)
    data = {}
    truth = CFG["truth"][scenario]
    for key in a8.expected_curve_keys():
        item = CFG["target"][key]
        eta_star = truth[key]
        values = []
        for eta, sd in zip(item["etas"], item["sds"]):
            mean = 2.1 + item["curvature"] * math.log2(eta / eta_star) ** 2
            values.append([rng.gauss(mean, sd) for _ in a8.SEEDS])
        data[key] = values
    return data


def fit_dataset_curve(data: dict, key: tuple, draw: list[int] | None) -> dict:
    item = CFG["target"][key]
    if draw is None:
        means = [sum(values) / len(values) for values in data[key]]
    else:
        means = [sum(values[index] for index in draw) / len(draw) for values in data[key]]
    return a8.fit_quadratic(item["etas"], means)


def simulate_one(task: tuple[str, int]) -> tuple[str, dict]:
    scenario, replicate = task
    data = make_dataset(replicate, scenario)
    point = {
        key: fit_dataset_curve(data, key, None) for key in a8.expected_curve_keys()
    }
    point_interior_keys = {key for key, fit in point.items() if fit["interior"]}
    valid_count = 0
    for draw, frequency in DRAW_GROUPS:
        if all(
            fit_dataset_curve(data, key, draw)["interior"]
            for key in a8.expected_curve_keys()
        ):
            valid_count += frequency
    evaluable = (
        len(point_interior_keys) == a8.EXPECTED_CURVES
        and valid_count >= a8.MIN_VALID_BOOTSTRAP_REPLICATES
    )
    return scenario, {
        "evaluable": evaluable,
        "point_interior": len(point_interior_keys) == a8.EXPECTED_CURVES,
        "point_interior_keys": point_interior_keys,
        "valid_refits": valid_count,
    }


def summarize(records: list[dict]) -> dict:
    evaluable = sum(record["evaluable"] for record in records)
    point_interior = sum(record["point_interior"] for record in records)
    valid_counts = [record["valid_refits"] for record in records]
    per_curve = {}
    for key in a8.expected_curve_keys():
        t, arm, mu = key
        label = f"T{t}_{arm}_mu{mu:g}"
        count = sum(key in record["point_interior_keys"] for record in records)
        per_curve[label] = {
            "point_interior": count,
            "P_point_interior": count / len(records),
        }
    return {
        "simulations": len(records),
        "evaluable": evaluable,
        "P_eval": evaluable / len(records),
        "P_eval_wilson95": wilson(evaluable, len(records)),
        "all_15_point_interior": point_interior,
        "P_all_15_point_interior": point_interior / len(records),
        "valid_complete_refits": {
            "min": min(valid_counts),
            "q10": a8.quantile(valid_counts, 0.10),
            "median": a8.quantile(valid_counts, 0.50),
            "q90": a8.quantile(valid_counts, 0.90),
            "max": max(valid_counts),
        },
        "per_curve_point_interior": per_curve,
    }


def literal_spot_check(config: dict) -> dict:
    global CFG
    CFG = config
    data = make_dataset(0, "centered")
    compressed = 0
    for draw, frequency in DRAW_GROUPS:
        if all(
            fit_dataset_curve(data, key, draw)["interior"]
            for key in a8.expected_curve_keys()
        ):
            compressed += frequency
    rng = random.Random(a8.BOOTSTRAP_SEED)
    literal = 0
    for _ in range(a8.BOOTSTRAP_REPLICATES):
        draw = [rng.randrange(len(a8.SEEDS)) for _ in a8.SEEDS]
        if all(
            fit_dataset_curve(data, key, draw)["interior"]
            for key in a8.expected_curve_keys()
        ):
            literal += 1
    if literal != compressed:
        raise RuntimeError("compressed bootstrap differs from literal bootstrap")
    return {"replicate": 0, "literal_valid": literal, "compressed_valid": compressed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--g3-readout", type=Path, required=True)
    parser.add_argument("--g3-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 2) - 2)))
    args = parser.parse_args()
    config = build_config(args.contract, args.g3_readout, args.g3_manifest)
    spot_check = literal_spot_check(config)
    scenarios = ("centered", "v3_vertex_shift_sensitivity")
    tasks = [
        (scenario, replicate)
        for scenario in scenarios
        for replicate in range(N_SIMULATIONS)
    ]
    records = {scenario: [] for scenario in scenarios}
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers, initializer=init_worker, initargs=(config,)
    ) as executor:
        for scenario, record in executor.map(simulate_one, tasks, chunksize=4):
            records[scenario].append(record)
    result = {
        "schema": "yeto_outer_mup_v8_gate_feasibility_v1",
        "pre_outcome": True,
        "cpu_only": True,
        "simulation_replicates_per_scenario": N_SIMULATIONS,
        "outer_seed_rule": f"{OUTER_SEED_BASE} + replicate",
        "registered_bootstrap": {
            "draws": a8.BOOTSTRAP_REPLICATES,
            "rng_seed": a8.BOOTSTRAP_SEED,
            "minimum_valid_complete_refits": a8.MIN_VALID_BOOTSTRAP_REPLICATES,
            "unique_multinomial_count_vectors": len(DRAW_GROUPS),
            "literal_compression_spot_check": spot_check,
        },
        "model": (
            "exact centered quadratic mean curves; curvature transported by "
            "T/arm from sealed v3; independent Gaussian eta-cell noise with "
            "sealed v3 unbiased five-seed SD transported by T/arm/rung; shared "
            "three-seed paired bootstrap across the full 15-curve diagram"
        ),
        "independence_note": (
            "eta-cell noises are independent, deliberately discarding favorable "
            "cross-curve seed covariance; this is conservative for paired loss "
            "differences but remains a transport model rather than a guarantee"
        ),
        "primary_scenario": "centered",
        "scenario_definitions": {
            "centered": "every true eta optimum equals its registered ladder center",
            "v3_vertex_shift_sensitivity": (
                "move each T/arm truth by the pre-existing sealed v3 fitted-vertex "
                "offset, log-interpolated from mu=0 to mu=0.9 and clamped above "
                "0.9"
            ),
        },
        "source": config["source_summary"],
        "design_sha256": config["design_sha256"],
        "scenarios": {
            scenario: summarize(scenario_records)
            for scenario, scenario_records in records.items()
        },
    }
    a8.write_json_atomic(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
