#!/usr/bin/env python3
"""Apply the frozen G4b analysis to the combined v4 and v4b grids."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path


SEEDS = (501, 503, 509)
S_GRID = (2560, 10240)
T_BY_S = {2560: 5, 10240: 20}
MU_GRID = (0.0, 0.9)
MU_HIGH = 0.9
V4_EXPECTED_CELLS = 48
V4B_EXPECTED_CELLS = 18
COMBINED_EXPECTED_CELLS = V4_EXPECTED_CELLS + V4B_EXPECTED_CELLS
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260726
MIN_VALID_BOOTSTRAP_REPLICATES = 9_500
D_BANDS = {5: (1.7, 3.2), 20: (0.8, 1.5)}
EXTENDED_CURVES = frozenset(((2560, 0.9), (10240, 0.0), (10240, 0.9)))


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


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnalysisError(f"{path}: expected a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    try:
        with path.open(encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise AnalysisError(f"{path}:{number}: expected a JSON object")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read {path}: {exc}") from exc
    return rows


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


def solve3(matrix: list[list[float]], vector: list[float]) -> list[float]:
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-15:
            raise AnalysisError("singular quadratic normal equation")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(3):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                left - factor * right
                for left, right in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][3] for row in range(3)]


def fit_quadratic(etas: list[float], losses: list[float]) -> dict:
    if len(etas) not in (4, 6) or len(losses) != len(etas):
        raise AnalysisError("registered eta fit requires exactly four or six points")
    if any(not math.isfinite(value) for value in (*etas, *losses)):
        raise AnalysisError("quadratic fit received nonfinite input")
    if any(eta <= 0 for eta in etas):
        raise AnalysisError("eta values must be positive")
    xs = [math.log2(eta) for eta in etas]
    sums = [sum(x**power for x in xs) for power in range(5)]
    matrix = [
        [sums[4], sums[3], sums[2]],
        [sums[3], sums[2], sums[1]],
        [sums[2], sums[1], sums[0]],
    ]
    vector = [
        sum(y * x * x for x, y in zip(xs, losses)),
        sum(y * x for x, y in zip(xs, losses)),
        sum(losses),
    ]
    a, b, c = solve3(matrix, vector)
    vertex = -b / (2.0 * a) if a else math.nan
    interior = a > 0 and min(xs) + 1e-12 < vertex < max(xs) - 1e-12
    return {
        "a": a,
        "b": b,
        "c": c,
        "vertex_log2_eta": vertex if math.isfinite(vertex) else None,
        "eta_star": 2.0**vertex if interior else None,
        "interior": interior,
        "status": "INTERIOR" if interior else "UNBRACKETED",
    }


def parse_node_roots(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise AnalysisError(f"--node-root must be NODE=PATH, got {value!r}")
        node, raw_path = value.split("=", 1)
        if node in result or not node or not raw_path:
            raise AnalysisError(f"invalid or duplicate --node-root {value!r}")
        result[node] = Path(raw_path).resolve()
    return result


def completed_attempt(cell: dict, root: Path) -> tuple[Path, dict, int]:
    attempt2 = root / cell["cell_id"] / "attempt-2"
    evidence2_path = attempt2 / "evidence.json"
    if evidence2_path.is_file():
        evidence2 = read_json(evidence2_path)
        if evidence2.get("status") != "COMPLETED":
            raise AnalysisError(
                f"{cell['cell_id']}: registered retry exists but is "
                f"{evidence2.get('status')}"
            )
        return attempt2, evidence2, 2
    attempt1 = root / cell["cell_id"] / "attempt-1"
    evidence1_path = attempt1 / "evidence.json"
    if not evidence1_path.is_file():
        raise AnalysisError(f"{cell['cell_id']}: attempt-1 evidence is missing")
    evidence1 = read_json(evidence1_path)
    if evidence1.get("status") != "COMPLETED":
        raise AnalysisError(
            f"{cell['cell_id']}: attempt 1 is {evidence1.get('status')}"
        )
    return attempt1, evidence1, 1


def load_losses(
    manifest: dict,
    node_roots: dict[str, Path],
    source_stage: str,
) -> tuple[dict, list[dict], list[str]]:
    losses: dict[tuple[int, float, int, float], float] = {}
    records = []
    errors = []
    for cell in manifest.get("cells", []):
        cell_id = cell.get("cell_id", "<missing-cell-id>")
        try:
            node = cell["assignment"]["node"]
            if node not in node_roots:
                raise AnalysisError(f"no results root supplied for node {node}")
            attempt, evidence, attempt_number = completed_attempt(cell, node_roots[node])
            expected_hash = (
                cell["command_hash"]
                if attempt_number == 1
                else cell["registered_retry_commands"][0]["command_hash"]
            )
            if evidence.get("cell_id") != cell_id:
                raise AnalysisError("evidence cell_id mismatch")
            if evidence.get("command_hash") != expected_hash:
                raise AnalysisError("evidence command hash mismatch")
            if evidence.get("seed") != cell["seed"]:
                raise AnalysisError("evidence seed mismatch")
            results_path = attempt / "report" / "results.jsonl"
            observed = evidence.get("observed_artifacts", {}).get("results", {})
            if observed.get("sha256") != sha256_file(results_path):
                raise AnalysisError("results hash does not match validated evidence")
            rows = read_jsonl(results_path)
            if len(rows) != 1:
                raise AnalysisError(f"expected one result row, found {len(rows)}")
            loss = rows[0].get("eval_loss")
            if not isinstance(loss, (int, float)) or not math.isfinite(loss):
                raise AnalysisError("endpoint evaluation loss is not finite")
            key = (
                int(cell["s"]),
                float(cell["mu"]),
                int(cell["seed"]),
                float(cell["eta"]),
            )
            if key in losses:
                raise AnalysisError("duplicate scientific cell coordinate")
            losses[key] = float(loss)
            records.append(
                {
                    "source_stage": source_stage,
                    "cell_id": cell_id,
                    "node": node,
                    "gpu": cell["assignment"]["gpu"],
                    "attempt": attempt_number,
                    "s": cell["s"],
                    "t": cell["t"],
                    "mu": cell["mu"],
                    "eta": cell["eta"],
                    "seed": cell["seed"],
                    "eval_loss": float(loss),
                    "evidence_path": str(attempt / "evidence.json"),
                    "evidence_sha256": sha256_file(attempt / "evidence.json"),
                    "results_sha256": observed["sha256"],
                }
            )
        except (AnalysisError, KeyError, OSError, ValueError) as exc:
            errors.append(f"{source_stage}/{cell_id}: {exc}")
    return losses, records, errors


def curve_fit(
    losses: dict[tuple[int, float, int, float], float],
    s: int,
    mu: float,
    sampled_indices: list[int] | None = None,
) -> dict:
    etas = sorted({key[3] for key in losses if key[0] == s and key[1] == mu})
    expected_points = 6 if (s, mu) in EXTENDED_CURVES else 4
    if len(etas) != expected_points:
        raise AnalysisError(
            f"S{s}/mu{mu}: expected {expected_points} combined etas, found {etas}"
        )
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
                raise AnalysisError(f"S{s}/mu{mu}: missing {key}")
            values.append(losses[key])
        means.append(sum(values) / len(values))
    fit = fit_quadratic(etas, means)
    fit.update(
        {
            "s": s,
            "t": T_BY_S[s],
            "mu": mu,
            "etas": etas,
            "point_count": len(etas),
            "seed_mean_losses": means,
        }
    )
    return fit


def d_from_fits(fit0: dict, fit9: dict) -> float | None:
    if not fit0["interior"] or not fit9["interior"]:
        return None
    return (fit9["eta_star"] / fit0["eta_star"]) / (1.0 - MU_HIGH)


def bootstrap_all(losses: dict[tuple[int, float, int, float], float]) -> dict:
    rng = random.Random(BOOTSTRAP_SEED)
    log2_d5_samples = []
    log2_d20_samples = []
    gap_samples = []
    invalid = 0
    for _ in range(BOOTSTRAP_REPLICATES):
        draw = [rng.randrange(len(SEEDS)) for _ in SEEDS]
        try:
            fit5_0 = curve_fit(losses, 2560, 0.0, draw)
            fit5_9 = curve_fit(losses, 2560, 0.9, draw)
            fit20_0 = curve_fit(losses, 10240, 0.0, draw)
            fit20_9 = curve_fit(losses, 10240, 0.9, draw)
            d5 = d_from_fits(fit5_0, fit5_9)
            d20 = d_from_fits(fit20_0, fit20_9)
            if d5 is None or d20 is None or d5 <= 0 or d20 <= 0:
                invalid += 1
                continue
            log2_d5 = math.log2(d5)
            log2_d20 = math.log2(d20)
            log2_d5_samples.append(log2_d5)
            log2_d20_samples.append(log2_d20)
            gap_samples.append(log2_d5 - log2_d20)
        except AnalysisError:
            invalid += 1

    status = (
        "VALID"
        if len(gap_samples) >= MIN_VALID_BOOTSTRAP_REPLICATES
        else "NOT_EVALUABLE"
    )

    def ratio_interval(samples: list[float]) -> dict:
        low = quantile(samples, 0.025) if samples else None
        high = quantile(samples, 0.975) if samples else None
        return {
            "coordinate": "log2_D",
            "ci_95_log2_D": {"low": low, "high": high},
            "ci_95_D": {
                "low": 2.0**low if low is not None else None,
                "high": 2.0**high if high is not None else None,
            },
        }

    return {
        "method": "paired_nonparametric_seed_curve_bootstrap_refitting_all_four_curves",
        "pairing": "one shared three-index resample is used for every eta and all four curves",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "minimum_valid_replicates": MIN_VALID_BOOTSTRAP_REPLICATES,
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
                "low": quantile(gap_samples, 0.025) if gap_samples else None,
                "high": quantile(gap_samples, 0.975) if gap_samples else None,
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-manifest", type=Path, required=True)
    parser.add_argument("--v4b-manifest", type=Path, required=True)
    parser.add_argument(
        "--v4-node-root",
        action="append",
        required=True,
        help="NODE=PATH; repeat once for each v4 results-bearing node or mirror",
    )
    parser.add_argument(
        "--v4b-node-root",
        action="append",
        required=True,
        help="NODE=PATH; repeat once for each v4b results-bearing node or mirror",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    v4_manifest = read_json(args.v4_manifest)
    v4b_manifest = read_json(args.v4b_manifest)
    if v4_manifest.get("schema") != "yeto_outer_mup_v4_scale_launch_manifest_v1":
        raise SystemExit("not a v4 scale launch manifest")
    if (
        v4_manifest.get("stage") != "V4_SCALE"
        or len(v4_manifest.get("cells", [])) != V4_EXPECTED_CELLS
    ):
        raise SystemExit("v4 manifest is not the complete 48-cell stage")
    if v4b_manifest.get("schema") != "yeto_outer_mup_v4b_extension_launch_manifest_v1":
        raise SystemExit("not a v4b extension launch manifest")
    if (
        v4b_manifest.get("stage") != "V4B_EXTENSION"
        or len(v4b_manifest.get("cells", [])) != V4B_EXPECTED_CELLS
    ):
        raise SystemExit("v4b manifest is not the complete 18-cell extension")
    registered_v4_hash = v4b_manifest.get("base_v4", {}).get("manifest_sha256")
    if registered_v4_hash != sha256_file(args.v4_manifest):
        raise SystemExit("v4b manifest is not bound to the supplied v4 manifest")

    v4_roots = parse_node_roots(args.v4_node_root)
    v4b_roots = parse_node_roots(args.v4b_node_root)
    v4_losses, v4_records, v4_errors = load_losses(v4_manifest, v4_roots, "v4")
    v4b_losses, v4b_records, v4b_errors = load_losses(
        v4b_manifest, v4b_roots, "v4b"
    )
    overlap = sorted(set(v4_losses).intersection(v4b_losses))
    evidence_errors = v4_errors + v4b_errors
    if overlap:
        evidence_errors.append(f"v4/v4b coordinate overlap: {overlap}")
    losses = dict(v4_losses)
    losses.update(v4b_losses)

    curves = []
    curve_map = {}
    for s in S_GRID:
        for mu in MU_GRID:
            try:
                fit = curve_fit(losses, s, mu)
            except AnalysisError as exc:
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

    d_values = {}
    for s in S_GRID:
        d_values[T_BY_S[s]] = d_from_fits(
            curve_map[(s, 0.0)], curve_map[(s, 0.9)]
        )

    complete_evidence = (
        not evidence_errors
        and len(v4_losses) == V4_EXPECTED_CELLS
        and len(v4b_losses) == V4B_EXPECTED_CELLS
        and len(losses) == COMBINED_EXPECTED_CELLS
    )
    bootstrap = (
        bootstrap_all(losses)
        if complete_evidence
        else {
            "status": "NOT_EVALUABLE",
            "error": "complete valid v4 and v4b cell evidence is required before bootstrap",
            "valid_replicates": 0,
            "D5": {"ci_95_D": {"low": None, "high": None}},
            "D20": {"ci_95_D": {"low": None, "high": None}},
            "monotone_gap": {"ci_95": {"low": None, "high": None}},
        }
    )

    required_fits_interior = all(
        curve_map[(s, mu)]["interior"] for s in S_GRID for mu in MU_GRID
    )
    d5, d20 = d_values[5], d_values[20]
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
        complete_evidence
        and required_fits_interior
        and bootstrap.get("status") == "VALID"
    )
    if not evaluable:
        verdict = "NOT_EVALUABLE"
    elif bands["T5"] and bands["T20"] and monotone:
        verdict = "PASS"
    else:
        verdict = "FAIL"

    def display(value: float | None, interval: dict) -> str:
        low = interval.get("low")
        high = interval.get("high")
        if value is None or not isinstance(low, (int, float)) or not isinstance(
            high, (int, float)
        ):
            return "NA [NA,NA]"
        return f"{value:.6f} [{low:.6f},{high:.6f}]"

    d5_ci = bootstrap.get("D5", {}).get("ci_95_D", {})
    d20_ci = bootstrap.get("D20", {}).get("ci_95_D", {})
    note_line = (
        f"G4B VERDICT: {verdict} "
        f"D5={display(d5, d5_ci)} D20={display(d20, d20_ci)}"
    )
    readout = {
        "schema": "yeto_outer_mup_v4b_g4b_readout_v1",
        "created_at_utc": utc_now(),
        "v4_manifest_path": str(args.v4_manifest.resolve()),
        "v4_manifest_sha256": sha256_file(args.v4_manifest),
        "v4b_manifest_path": str(args.v4b_manifest.resolve()),
        "v4b_manifest_sha256": sha256_file(args.v4b_manifest),
        "source_git_commit": v4b_manifest.get("source", {}).get("git_commit"),
        "expected_cells": {
            "v4": V4_EXPECTED_CELLS,
            "v4b": V4B_EXPECTED_CELLS,
            "combined": COMBINED_EXPECTED_CELLS,
        },
        "observed_completed_cells": {
            "v4": len(v4_records),
            "v4b": len(v4b_records),
            "combined": len(v4_records) + len(v4b_records),
        },
        "evidence_errors": evidence_errors,
        "cell_records": v4_records + v4b_records,
        "curve_fits": curves,
        "D_obs": {"T5": d5, "T20": d20},
        "bootstrap": bootstrap,
        "gate": {
            "name": "G4b",
            "verdict": verdict,
            "conditions": {
                "all_four_combined_grid_optima_interior": required_fits_interior,
                "D5_in_1.7_to_3.2": bands["T5"],
                "D20_in_0.8_to_1.5": bands["T20"],
                "D5_greater_than_D20_paired_bootstrap_ci_excludes_zero": monotone,
                "minimum_9500_valid_shared_bootstrap_refits": (
                    bootstrap.get("status") == "VALID"
                ),
            },
            "interpretation": "original G4 bands and paired monotonicity criterion applied after the preregistered downward extension",
        },
        "note_line": note_line,
    }
    write_json_atomic(args.output.resolve(), readout)
    print(note_line)
    return 0 if verdict in ("PASS", "FAIL") else 2


if __name__ == "__main__":
    raise SystemExit(main())
