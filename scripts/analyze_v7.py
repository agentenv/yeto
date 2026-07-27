#!/usr/bin/env python3
"""Frozen G7 analysis for the 27B FSDP+LoRA finite-horizon T-scan."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path


SEEDS = (701, 709, 719)
S_GRID = (2560, 10240)
T_BY_S = {2560: 5, 10240: 20}
MU_GRID = (0.0, 0.9)
MU_HIGH = 0.9
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260727
NEAR_BRACKET_ALLOWANCE_LOG2 = 0.5
MIN_VALID_BOOTSTRAP_REPLICATES = 7_900
GRID_VARIANT_CELLS = {
    "FULL_48": 48,
    "REDUCED_T20_MU0_45": 45,
}
G4C_OBSERVED_D = {
    5: 1.7416157949788522,
    20: 1.2806943474449415,
}
D_BANDS = {
    horizon: (0.5 * value, 1.5 * value) for horizon, value in G4C_OBSERVED_D.items()
}


class AnalysisError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnalysisError(f"{path}: expected a JSON object")
    return value


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise AnalysisError("quantile of an empty sequence")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def solve_linear(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    if len(matrix) != size or any(len(row) != size for row in matrix):
        raise AnalysisError("linear system has inconsistent dimensions")
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-14:
            raise AnalysisError("singular normal equation")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                left - factor * right
                for left, right in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][-1] for row in range(size)]


def fit_quadratic(etas: list[float], losses: list[float]) -> dict:
    if len(etas) not in (3, 4) or len(losses) != len(etas):
        raise AnalysisError("registered eta fit requires exactly three or four points")
    if len(set(etas)) != len(etas):
        raise AnalysisError("eta fit received a duplicate eta")
    if any(not math.isfinite(value) or value <= 0.0 for value in etas):
        raise AnalysisError("eta fit received a nonpositive or nonfinite eta")
    if any(not math.isfinite(value) for value in losses):
        raise AnalysisError("eta fit received a nonfinite loss")
    xs = [math.log2(value) for value in etas]
    features = [[x * x, x, 1.0] for x in xs]
    matrix = [
        [sum(row[i] * row[j] for row in features) for j in range(3)] for i in range(3)
    ]
    vector = [
        sum(row[i] * loss for row, loss in zip(features, losses)) for i in range(3)
    ]
    a, b, c = solve_linear(matrix, vector)
    vertex = -b / (2.0 * a) if a else math.nan
    strict_interior = bool(
        a > 0.0 and math.isfinite(vertex) and min(xs) + 1e-12 < vertex < max(xs) - 1e-12
    )
    accepted = bool(
        a > 0.0
        and math.isfinite(vertex)
        and min(xs) - NEAR_BRACKET_ALLOWANCE_LOG2
        < vertex
        < max(xs) + NEAR_BRACKET_ALLOWANCE_LOG2
    )
    near_bracketed = accepted and not strict_interior
    return {
        "a": a,
        "b": b,
        "c": c,
        "vertex_log2_eta": vertex if math.isfinite(vertex) else None,
        "eta_star": 2.0**vertex if accepted else None,
        "strict_interior": strict_interior,
        "accepted": accepted,
        "near_bracketed": near_bracketed,
        "status": (
            "INTERIOR"
            if strict_interior
            else "NEAR_BRACKETED"
            if near_bracketed
            else "UNBRACKETED"
        ),
    }


def normalized_eta_grids(raw: dict) -> dict[tuple[int, float], list[float]]:
    expected_keys = {
        "T5_mu0": (2560, 0.0),
        "T5_mu0.9": (2560, 0.9),
        "T20_mu0": (10240, 0.0),
        "T20_mu0.9": (10240, 0.9),
    }
    if set(raw) != set(expected_keys):
        raise AnalysisError("manifest eta-grid keys differ from the four G7 curves")
    grids = {}
    for key, coordinate in expected_keys.items():
        values = raw[key]
        if not isinstance(values, list) or len(values) not in (3, 4):
            raise AnalysisError(f"{key}: eta grid must contain three or four points")
        etas = [float(value) for value in values]
        if etas != sorted(etas) or len(set(etas)) != len(etas):
            raise AnalysisError(f"{key}: eta grid must be unique and increasing")
        if any(not math.isfinite(value) or value <= 0.0 for value in etas):
            raise AnalysisError(f"{key}: eta grid contains an invalid value")
        grids[coordinate] = etas
    return grids


def curve_fit(
    losses: dict[tuple[int, float, int, float], float],
    grids: dict[tuple[int, float], list[float]],
    s: int,
    mu: float,
    sampled_indices: list[int] | None = None,
) -> dict:
    etas = grids[(s, mu)]
    selected_seeds = (
        [SEEDS[index] for index in sampled_indices]
        if sampled_indices is not None
        else list(SEEDS)
    )
    means = []
    for eta in etas:
        values = []
        for seed in selected_seeds:
            key = (s, mu, seed, eta)
            if key not in losses:
                raise AnalysisError(f"missing loss at {key}")
            values.append(losses[key])
        means.append(sum(values) / len(values))
    fit = fit_quadratic(etas, means)
    fit.update(
        {
            "s": s,
            "t": T_BY_S[s],
            "mu": mu,
            "etas": list(etas),
            "point_count": len(etas),
            "seeds": list(SEEDS),
            "seed_mean_losses": means,
        }
    )
    return fit


def d_from_fits(fit0: dict, fit9: dict) -> float | None:
    if not fit0.get("accepted") or not fit9.get("accepted"):
        return None
    value = (fit9["eta_star"] / fit0["eta_star"]) / (1.0 - MU_HIGH)
    return value if math.isfinite(value) and value > 0.0 else None


def ratio_interval(log2_samples: list[float]) -> dict:
    low = quantile(log2_samples, 0.025) if log2_samples else None
    high = quantile(log2_samples, 0.975) if log2_samples else None
    return {
        "coordinate": "log2_D",
        "ci_95_log2_D": {"low": low, "high": high},
        "ci_95_D": {
            "low": 2.0**low if low is not None else None,
            "high": 2.0**high if high is not None else None,
        },
    }


def bootstrap_all(
    losses: dict[tuple[int, float, int, float], float],
    grids: dict[tuple[int, float], list[float]],
) -> dict:
    rng = random.Random(BOOTSTRAP_SEED)
    log2_d5_samples = []
    log2_d20_samples = []
    gap_samples = []
    invalid = 0
    for _ in range(BOOTSTRAP_REPLICATES):
        draw = [rng.randrange(len(SEEDS)) for _ in SEEDS]
        try:
            fits = {
                (s, mu): curve_fit(losses, grids, s, mu, draw)
                for s in S_GRID
                for mu in MU_GRID
            }
            d5 = d_from_fits(fits[(2560, 0.0)], fits[(2560, 0.9)])
            d20 = d_from_fits(fits[(10240, 0.0)], fits[(10240, 0.9)])
            if d5 is None or d20 is None:
                invalid += 1
                continue
            log2_d5 = math.log2(d5)
            log2_d20 = math.log2(d20)
            log2_d5_samples.append(log2_d5)
            log2_d20_samples.append(log2_d20)
            gap_samples.append(log2_d5 - log2_d20)
        except AnalysisError:
            invalid += 1
    valid = len(gap_samples)
    return {
        "method": "paired_nonparametric_training_seed_curve_bootstrap",
        "pairing": "one shared three-index draw is used at every eta and all four curves",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "near_bracket_allowance_log2_eta": NEAR_BRACKET_ALLOWANCE_LOG2,
        "minimum_valid_replicates": MIN_VALID_BOOTSTRAP_REPLICATES,
        "minimum_valid_fraction": MIN_VALID_BOOTSTRAP_REPLICATES / BOOTSTRAP_REPLICATES,
        "valid_replicates": valid,
        "invalid_unbracketed_replicates": invalid,
        "status": "VALID"
        if valid >= MIN_VALID_BOOTSTRAP_REPLICATES
        else "NOT_EVALUABLE",
        "D5": ratio_interval(log2_d5_samples),
        "D20": ratio_interval(log2_d20_samples),
        "monotone_gap": {
            "coordinate": "log2_D_T5_minus_log2_D_T20",
            "ci_95": {
                "low": quantile(gap_samples, 0.025) if gap_samples else None,
                "high": quantile(gap_samples, 0.975) if gap_samples else None,
            },
        },
    }


def analyze_losses(
    losses: dict[tuple[int, float, int, float], float],
    grids: dict[tuple[int, float], list[float]],
    *,
    complete_evidence: bool = True,
) -> dict:
    curves = []
    curve_map = {}
    errors = []
    for s in S_GRID:
        for mu in MU_GRID:
            try:
                fit = curve_fit(losses, grids, s, mu)
            except AnalysisError as exc:
                fit = {
                    "s": s,
                    "t": T_BY_S[s],
                    "mu": mu,
                    "accepted": False,
                    "strict_interior": False,
                    "near_bracketed": False,
                    "status": "INVALID_INPUT",
                    "eta_star": None,
                    "error": str(exc),
                }
                errors.append(str(exc))
            curves.append(fit)
            curve_map[(s, mu)] = fit

    d5 = d_from_fits(curve_map[(2560, 0.0)], curve_map[(2560, 0.9)])
    d20 = d_from_fits(curve_map[(10240, 0.0)], curve_map[(10240, 0.9)])
    expected_loss_count = sum(
        len(grids[(s, mu)]) for s in S_GRID for mu in MU_GRID
    ) * len(SEEDS)
    complete = complete_evidence and not errors and len(losses) == expected_loss_count
    bootstrap = (
        bootstrap_all(losses, grids)
        if complete
        else {
            "status": "NOT_EVALUABLE",
            "error": "complete valid G7 evidence is required",
            "valid_replicates": 0,
            "D5": {"ci_95_D": {"low": None, "high": None}},
            "D20": {"ci_95_D": {"low": None, "high": None}},
            "monotone_gap": {"ci_95": {"low": None, "high": None}},
        }
    )
    all_accepted = all(fit.get("accepted") for fit in curves)
    mu09_accepted = all(curve_map[(s, 0.9)].get("accepted") for s in S_GRID)
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
    evaluable = (
        complete
        and all_accepted
        and mu09_accepted
        and bootstrap.get("status") == "VALID"
    )
    if not evaluable:
        verdict = "NOT_EVALUABLE"
    elif bands["T5"] and bands["T20"] and monotone:
        verdict = "PASS"
    else:
        verdict = "FAIL"
    return {
        "curves": curves,
        "D_obs": {"T5": d5, "T20": d20},
        "bootstrap": bootstrap,
        "gate": {
            "name": "G7_27B_LoRA_finite_horizon_transient_law",
            "verdict": verdict,
            "evaluable": evaluable,
            "conditions": {
                "complete_evidence": complete,
                "all_ratio_required_optima_accepted": all_accepted,
                "both_mu0.9_optima_accepted": mu09_accepted,
                "D5_inside_registered_band": bands["T5"],
                "D20_inside_registered_band": bands["T20"],
                "paired_monotone_decrease_ci_excludes_zero": monotone,
            },
            "registered_bands": {
                "T5": list(D_BANDS[5]),
                "T20": list(D_BANDS[20]),
            },
        },
    }


def parse_node_roots(values: list[str]) -> dict[str, Path]:
    roots = {}
    for value in values:
        node, separator, path = value.partition("=")
        if not separator or not node or not path:
            raise AnalysisError(f"invalid --node-root {value!r}; expected NODE=PATH")
        if node in roots:
            raise AnalysisError(f"duplicate node root for {node}")
        roots[node] = Path(path)
    return roots


def load_retry_groups(path: Path | None, manifest_path: Path) -> set[str]:
    if path is None:
        return set()
    authority = read_json(path)
    errors = []
    if authority.get("schema") != "yeto_outer_mup_v7_retry_authority_v1":
        errors.append("retry authority schema mismatch")
    if authority.get("status") != "AUTHORIZED":
        errors.append("retry authority is not authorized")
    if authority.get("manifest_sha256") != sha256_file(manifest_path):
        errors.append("retry authority binds another manifest")
    groups = authority.get("retry_group_ids")
    if not isinstance(groups, list) or len(groups) != len(set(groups)):
        errors.append("retry groups are malformed or duplicated")
        groups = []
    if errors:
        raise AnalysisError("; ".join(errors))
    return set(groups)


def validate_manifest(
    manifest: dict,
) -> tuple[dict[tuple[int, float], list[float]], int]:
    if manifest.get("schema") != "yeto_outer_mup_v7_27b_lora_launch_manifest_v1":
        raise AnalysisError("manifest schema mismatch")
    if manifest.get("stage") != "V7_27B_LORA_GRID":
        raise AnalysisError("manifest stage mismatch")
    variant = manifest.get("grid", {}).get("variant")
    if variant not in GRID_VARIANT_CELLS:
        raise AnalysisError("unknown G7 grid variant")
    expected_cells = GRID_VARIANT_CELLS[variant]
    cells = manifest.get("cells")
    if not isinstance(cells, list) or len(cells) != expected_cells:
        raise AnalysisError(f"manifest does not contain {expected_cells} cells")
    grids = normalized_eta_grids(manifest.get("grid", {}).get("eta_grids", {}))
    if variant == "FULL_48" and any(len(values) != 4 for values in grids.values()):
        raise AnalysisError("FULL_48 requires four eta points on every curve")
    if variant == "REDUCED_T20_MU0_45":
        expected_counts = {
            (2560, 0.0): 4,
            (2560, 0.9): 4,
            (10240, 0.0): 3,
            (10240, 0.9): 4,
        }
        if {key: len(value) for key, value in grids.items()} != expected_counts:
            raise AnalysisError("REDUCED_T20_MU0_45 eta counts are incorrect")

    expected_coordinates = {
        (s, mu, seed, eta)
        for (s, mu), etas in grids.items()
        for eta in etas
        for seed in SEEDS
    }
    observed_coordinates = set()
    ids = set()
    for cell in cells:
        cell_id = cell.get("cell_id")
        if not isinstance(cell_id, str) or cell_id in ids:
            raise AnalysisError("manifest cell ids are missing or duplicated")
        ids.add(cell_id)
        coordinate = (
            int(cell.get("s", -1)),
            float(cell.get("mu", math.nan)),
            int(cell.get("seed", -1)),
            float(cell.get("eta", math.nan)),
        )
        if coordinate in observed_coordinates:
            raise AnalysisError(f"duplicate manifest coordinate {coordinate}")
        observed_coordinates.add(coordinate)
        if int(cell.get("t", -1)) != T_BY_S.get(coordinate[0]):
            raise AnalysisError(f"{cell_id}: T/S binding mismatch")
        if int(cell.get("h", -1)) != 512 or int(cell.get("m", -1)) != 2:
            raise AnalysisError(f"{cell_id}: H/M binding mismatch")
        if int(cell.get("training_seed", -1)) != int(f"{coordinate[2]}{coordinate[2]}"):
            raise AnalysisError(f"{cell_id}: training seed mismatch")
        if canonical_sha256(cell.get("command")) != cell.get("command_hash"):
            raise AnalysisError(f"{cell_id}: command hash mismatch")
    if observed_coordinates != expected_coordinates:
        raise AnalysisError("manifest coordinates differ from the registered grid")
    return grids, expected_cells


def load_losses(
    manifest: dict,
    manifest_path: Path,
    roots: dict[str, Path],
    retry_groups: set[str],
) -> tuple[dict[tuple[int, float, int, float], float], list[dict], list[str]]:
    losses = {}
    records = []
    errors = []
    known_groups = {cell.get("retry_group_id") for cell in manifest["cells"]}
    if retry_groups - known_groups:
        errors.append("retry authority contains an unknown group")
    for cell in manifest["cells"]:
        cell_id = cell["cell_id"]
        node = cell.get("assignment", {}).get("node")
        if node not in roots:
            errors.append(f"{cell_id}: no result root supplied for node {node!r}")
            continue
        attempt = 2 if cell.get("retry_group_id") in retry_groups else 1
        attempt_root = roots[node] / cell_id / f"attempt-{attempt}"
        evidence_path = attempt_root / "evidence.json"
        try:
            evidence = read_json(evidence_path)
            if evidence.get("schema") != "yeto_outer_mup_cell_evidence_v1":
                raise AnalysisError("evidence schema mismatch")
            if evidence.get("status") != "COMPLETED":
                raise AnalysisError(f"evidence status is {evidence.get('status')!r}")
            bindings = {
                "cell_id": cell_id,
                "attempt_number": attempt,
                "seed": cell["seed"],
                "training_seed": cell["training_seed"],
                "command_hash": cell["command_hash"]
                if attempt == 1
                else cell["registered_retry_commands"][0]["command_hash"],
            }
            for key, expected in bindings.items():
                if evidence.get(key) != expected:
                    raise AnalysisError(f"evidence {key} mismatch")
            results_path = attempt_root / "report" / "results.jsonl"
            result_record = evidence.get("observed_artifacts", {}).get("results", {})
            if result_record.get("sha256") != sha256_file(results_path):
                raise AnalysisError("results hash mismatch")
            result_rows = [
                json.loads(line)
                for line in results_path.read_text().splitlines()
                if line.strip()
            ]
            matching = [row for row in result_rows if row.get("arm") == "m2"]
            if len(matching) != 1:
                raise AnalysisError("results must contain exactly one m2 row")
            loss = float(matching[0]["eval_loss"])
            if not math.isfinite(loss):
                raise AnalysisError("endpoint loss is nonfinite")
            key = (
                int(cell["s"]),
                float(cell["mu"]),
                int(cell["seed"]),
                float(cell["eta"]),
            )
            if key in losses:
                raise AnalysisError(f"duplicate loss coordinate {key}")
            losses[key] = loss
            records.append(
                {
                    "cell_id": cell_id,
                    "node": node,
                    "attempt": attempt,
                    "s": key[0],
                    "t": int(cell["t"]),
                    "mu": key[1],
                    "seed": key[2],
                    "eta": key[3],
                    "eval_loss": loss,
                    "evidence_path": str(evidence_path.resolve()),
                    "evidence_sha256": sha256_file(evidence_path),
                    "results_sha256": result_record["sha256"],
                }
            )
        except (
            AnalysisError,
            OSError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
        ) as exc:
            errors.append(f"{cell_id}/attempt-{attempt}: {exc}")
    return losses, records, errors


def display(value: float | None, interval: dict) -> str:
    low = interval.get("low")
    high = interval.get("high")
    if (
        value is None
        or not isinstance(low, (int, float))
        or not isinstance(high, (int, float))
    ):
        return "NA [NA,NA]"
    return f"{value:.6f} [{low:.6f},{high:.6f}]"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--node-root", action="append", required=True)
    parser.add_argument("--retry-authority", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        manifest = read_json(args.manifest)
        grids, expected_cells = validate_manifest(manifest)
        roots = parse_node_roots(args.node_root)
        retry_groups = load_retry_groups(args.retry_authority, args.manifest)
        losses, records, evidence_errors = load_losses(
            manifest, args.manifest, roots, retry_groups
        )
        analysis = analyze_losses(
            losses,
            grids,
            complete_evidence=not evidence_errors and len(records) == expected_cells,
        )
    except AnalysisError as exc:
        raise SystemExit(str(exc)) from exc

    d5 = analysis["D_obs"]["T5"]
    d20 = analysis["D_obs"]["T20"]
    d5_ci = analysis["bootstrap"].get("D5", {}).get("ci_95_D", {})
    d20_ci = analysis["bootstrap"].get("D20", {}).get("ci_95_D", {})
    note_line = (
        f"G7 VERDICT: {analysis['gate']['verdict']} "
        f"D5={display(d5, d5_ci)} D20={display(d20, d20_ci)}"
    )
    readout = {
        "schema": "yeto_outer_mup_v7_g7_readout_v1",
        "created_at_utc": utc_now(),
        "source_git_commit": manifest.get("source", {}).get("git_commit"),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "grid_variant": manifest.get("grid", {}).get("variant"),
        "expected_cells": expected_cells,
        "observed_completed_cells": len(records),
        "evidence_errors": evidence_errors,
        "cell_records": records,
        "curve_fits": analysis["curves"],
        "D_obs": analysis["D_obs"],
        "bootstrap": analysis["bootstrap"],
        "gate": analysis["gate"],
        "note_line": note_line,
    }
    write_json_atomic(args.output, readout)
    print(note_line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
