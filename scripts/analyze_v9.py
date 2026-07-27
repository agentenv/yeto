#!/usr/bin/env python3
"""Apply the frozen G9 analysis to the sealed 1.7B and 7B verification cells."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v9_common import (  # noqa: E402
    V9Error,
    fit_quadratic,
    quantile,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
    write_json_atomic,
)

try:
    from run_slot_v3 import canonical_sha256
except ModuleNotFoundError:  # imported through repository root in tests
    from scripts.run_slot_v3 import canonical_sha256


MANIFEST_SCHEMA = "yeto_outer_mup_v9_launch_manifest_v1"
PREDICTION_SCHEMA = "yeto_outer_mup_v9_sealed_predictions_v1"
EXPECTED_CELLS = 28
SEEDS = (901, 907)
GATE_TO_STAGE = {"G9A_1P7B": "stage_1p7b", "G9B_7B": "stage_7b"}
STAGE_TARGETS = {
    "stage_1p7b": ("raw", "corrected"),
    "stage_7b": ("mu0", "raw"),
}


def parse_node_roots(values: list[str]) -> dict[str, Path]:
    roots = {}
    for value in values:
        if "=" not in value:
            raise V9Error(f"--node-root must be NODE=PATH, got {value!r}")
        node, raw_path = value.split("=", 1)
        if not node or not raw_path or node in roots:
            raise V9Error(f"invalid or duplicate --node-root {value!r}")
        roots[node] = Path(raw_path).resolve()
    return roots


def completed_attempt(cell: dict, root: Path) -> tuple[Path, dict, int]:
    attempt2 = root / cell["cell_id"] / "attempt-2"
    evidence2 = attempt2 / "evidence.json"
    if evidence2.is_file():
        value = read_json(evidence2)
        if value.get("status") != "COMPLETED":
            raise V9Error(f"registered attempt 2 is {value.get('status')}")
        return attempt2, value, 2
    attempt1 = root / cell["cell_id"] / "attempt-1"
    evidence1 = attempt1 / "evidence.json"
    if not evidence1.is_file():
        raise V9Error("attempt-1 evidence is missing")
    value = read_json(evidence1)
    if value.get("status") != "COMPLETED":
        raise V9Error(f"attempt 1 is {value.get('status')}")
    return attempt1, value, 1


def load_losses(
    manifest: dict, node_roots: dict[str, Path]
) -> tuple[dict, list[dict], list[str]]:
    losses = {}
    records = []
    errors = []
    for cell in manifest.get("cells", []):
        cell_id = cell.get("cell_id", "<missing-cell-id>")
        try:
            node = cell["assignment"]["node"]
            if node not in node_roots:
                raise V9Error(f"no result root supplied for {node}")
            attempt, evidence, attempt_number = completed_attempt(
                cell, node_roots[node]
            )
            expected_hash = (
                cell["command_hash"]
                if attempt_number == 1
                else cell["registered_retry_commands"][0]["command_hash"]
            )
            if evidence.get("cell_id") != cell_id:
                raise V9Error("evidence cell_id mismatch")
            if evidence.get("command_hash") != expected_hash:
                raise V9Error("evidence command hash mismatch")
            if evidence.get("seed") != cell["seed"]:
                raise V9Error("evidence seed mismatch")
            command = (
                cell["command"]
                if attempt_number == 1
                else cell["registered_retry_commands"][0]["command"]
            )
            if canonical_sha256(command) != expected_hash:
                raise V9Error("manifest command no longer matches its hash")
            results_path = attempt / "report" / "results.jsonl"
            observed = evidence.get("observed_artifacts", {}).get("results", {})
            if observed.get("sha256") != sha256_file(results_path):
                raise V9Error("results hash differs from validated evidence")
            rows = read_jsonl(results_path)
            if len(rows) != 1:
                raise V9Error(f"expected one result row, found {len(rows)}")
            loss = rows[0].get("eval_loss")
            if not isinstance(loss, (int, float)) or not math.isfinite(loss):
                raise V9Error("endpoint loss is not finite")
            key = (
                str(cell["stage"]),
                str(cell["arm"]),
                int(cell["seed"]),
                float(cell["eta"]),
            )
            if key in losses:
                raise V9Error("duplicate scientific coordinate")
            losses[key] = float(loss)
            records.append(
                {
                    "cell_id": cell_id,
                    "stage": cell["stage"],
                    "arm": cell["arm"],
                    "seed": cell["seed"],
                    "eta": cell["eta"],
                    "eval_loss": float(loss),
                    "node": node,
                    "gpus": cell["assignment"]["gpus"],
                    "attempt": attempt_number,
                    "evidence_path": str(attempt / "evidence.json"),
                    "evidence_sha256": sha256_file(attempt / "evidence.json"),
                    "results_sha256": observed["sha256"],
                }
            )
        except (V9Error, KeyError, OSError, ValueError) as exc:
            errors.append(f"{cell_id}: {exc}")
    return losses, records, errors


def gate_contract(manifest: dict, gate_id: str) -> dict:
    record = manifest.get("analysis_contract", {}).get("gates", {}).get(gate_id)
    if not isinstance(record, dict):
        raise V9Error(f"manifest lacks frozen {gate_id} analysis contract")
    threshold = record.get("minimum_valid_bootstrap_refits")
    if not isinstance(threshold, int) or not 0 <= threshold <= 10_000:
        raise V9Error(f"{gate_id} bootstrap threshold is invalid")
    if record.get("near_bracket_allowance_bits") != 0.5:
        raise V9Error(f"{gate_id} near-bracket allowance differs from gatesim")
    return record


def fit_curve(
    losses: dict,
    predictions: dict,
    manifest: dict,
    stage: str,
    arm: str,
    sampled_indices: list[int] | None = None,
) -> dict:
    target = predictions[stage]["targets"][arm]
    etas = [float(value) for value in target["verification_etas"]]
    selected_seeds = (
        [SEEDS[index] for index in sampled_indices]
        if sampled_indices is not None
        else list(SEEDS)
    )
    means = []
    for eta in etas:
        values = []
        for seed in selected_seeds:
            key = (stage, arm, seed, eta)
            if key not in losses:
                raise V9Error(f"missing v9 loss {key}")
            values.append(losses[key])
        means.append(sum(values) / len(values))
    gate_id = next(key for key, value in GATE_TO_STAGE.items() if value == stage)
    contract = gate_contract(manifest, gate_id)
    fit = fit_quadratic(
        etas,
        means,
        near_bracket_allowance_bits=float(contract["near_bracket_allowance_bits"]),
    )
    predicted = float(target["predicted_eta_star"])
    fit.update(
        {
            "stage": stage,
            "arm": arm,
            "etas": etas,
            "seed_mean_losses": means,
            "seeds": selected_seeds,
            "predicted_eta_star": predicted,
            "signed_error_bits_estimate_minus_prediction": (
                math.log2(float(fit["eta_star"]) / predicted)
                if fit["accepted"]
                else None
            ),
            "registered_absolute_error_band_bits": float(
                contract["absolute_error_band_bits"][arm]
            ),
        }
    )
    error = fit["signed_error_bits_estimate_minus_prediction"]
    fit["within_registered_band"] = bool(
        fit["accepted"]
        and error is not None
        and abs(float(error)) <= fit["registered_absolute_error_band_bits"]
    )
    return fit


def point_fits(losses: dict, predictions: dict, manifest: dict) -> dict:
    return {
        (stage, arm): fit_curve(losses, predictions, manifest, stage, arm)
        for stage, arms in STAGE_TARGETS.items()
        for arm in arms
    }


def bootstrap(losses: dict, predictions: dict, manifest: dict) -> dict[str, dict]:
    # Frequencies are frozen by gatesim and copied byte-for-byte into the
    # manifest; using count-vector representatives is exactly equivalent to
    # the literal 10,000 ordered two-index resamples for pooled seed means.
    groups = manifest["analysis_contract"]["bootstrap"]["groups"]
    if (
        len(groups) != 3
        or sum(int(group["frequency"]) for group in groups) != 10_000
        or {
            tuple(sorted(int(value) for value in group["representative"]))
            for group in groups
        }
        != {(0, 0), (0, 1), (1, 1)}
    ):
        raise V9Error("manifest bootstrap groups differ from exact two-seed support")
    results = {}
    for gate_id, stage in GATE_TO_STAGE.items():
        valid = 0
        eta_samples = {arm: [] for arm in STAGE_TARGETS[stage]}
        error_samples = {arm: [] for arm in STAGE_TARGETS[stage]}
        for group in groups:
            draw = [int(value) for value in group["representative"]]
            frequency = int(group["frequency"])
            fits = {
                arm: fit_curve(losses, predictions, manifest, stage, arm, draw)
                for arm in STAGE_TARGETS[stage]
            }
            if not all(fit["accepted"] for fit in fits.values()):
                continue
            valid += frequency
            for arm, fit in fits.items():
                eta_samples[arm].extend([float(fit["eta_star"])] * frequency)
                error_samples[arm].extend(
                    [float(fit["signed_error_bits_estimate_minus_prediction"])]
                    * frequency
                )
        threshold = gate_contract(manifest, gate_id)["minimum_valid_bootstrap_refits"]
        results[gate_id] = {
            "valid_replicates": valid,
            "invalid_replicates": 10_000 - valid,
            "minimum_valid_replicates": threshold,
            "status": "VALID" if valid >= threshold else "NOT_EVALUABLE",
            "targets": {
                arm: {
                    "eta_star_ci_95": {
                        "low": quantile(eta_samples[arm], 0.025)
                        if eta_samples[arm]
                        else None,
                        "high": quantile(eta_samples[arm], 0.975)
                        if eta_samples[arm]
                        else None,
                    },
                    "signed_error_bits_ci_95": {
                        "low": quantile(error_samples[arm], 0.025)
                        if error_samples[arm]
                        else None,
                        "high": quantile(error_samples[arm], 0.975)
                        if error_samples[arm]
                        else None,
                    },
                }
                for arm in STAGE_TARGETS[stage]
            },
        }
    return results


def analyze_losses(
    *,
    losses: dict,
    predictions: dict,
    manifest: dict,
    stage_complete: dict[str, bool],
) -> dict:
    fits = point_fits(losses, predictions, manifest)
    boot = bootstrap(losses, predictions, manifest)
    gates = {}
    for gate_id, stage in GATE_TO_STAGE.items():
        stage_fits = {arm: fits[(stage, arm)] for arm in STAGE_TARGETS[stage]}
        evaluable = bool(
            stage_complete[stage]
            and all(fit["accepted"] for fit in stage_fits.values())
            and boot[gate_id]["status"] == "VALID"
        )
        if not evaluable:
            verdict = "NOT_EVALUABLE"
        elif all(fit["within_registered_band"] for fit in stage_fits.values()):
            verdict = "PASS"
        else:
            verdict = "FAIL"
        gates[gate_id] = {
            "verdict": verdict,
            "evaluable": evaluable,
            "complete_evidence": stage_complete[stage],
            "all_point_fits_accepted": all(
                fit["accepted"] for fit in stage_fits.values()
            ),
            "all_targets_within_registered_bands": all(
                fit["within_registered_band"] for fit in stage_fits.values()
            ),
            "curve_fits": stage_fits,
            "bootstrap": boot[gate_id],
        }
    if any(gate["verdict"] == "NOT_EVALUABLE" for gate in gates.values()):
        verdict = "NOT_EVALUABLE"
    elif all(gate["verdict"] == "PASS" for gate in gates.values()):
        verdict = "PASS"
    else:
        verdict = "FAIL"
    return {"verdict": verdict, "gates": gates}


def display_error(gate: dict, arm: str) -> str:
    value = (
        gate.get("curve_fits", {})
        .get(arm, {})
        .get("signed_error_bits_estimate_minus_prediction")
    )
    return "NA" if value is None else f"{float(value):+.6f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--node-root",
        action="append",
        required=True,
        help="NODE=PATH; repeat for each results-bearing node or analysis mirror",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    predictions = read_json(args.predictions)
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or len(manifest.get("cells", [])) != EXPECTED_CELLS
    ):
        raise SystemExit("manifest is not the complete 28-cell v9 launch")
    if (
        predictions.get("schema") != PREDICTION_SCHEMA
        or predictions.get("status") != "SEALED"
    ):
        raise SystemExit("predictions are not the sealed v9 predictions")
    if manifest.get("predictions", {}).get("sha256") != sha256_file(args.predictions):
        raise SystemExit("manifest binds another prediction file")
    roots = parse_node_roots(args.node_root)
    losses, cell_records, evidence_errors = load_losses(manifest, roots)
    expected_by_stage = {"stage_1p7b": 16, "stage_7b": 12}
    observed_by_stage = {
        stage: sum(record["stage"] == stage for record in cell_records)
        for stage in expected_by_stage
    }
    stage_complete = {
        stage: observed_by_stage[stage] == expected
        for stage, expected in expected_by_stage.items()
    }
    try:
        analysis = analyze_losses(
            losses=losses,
            predictions=predictions,
            manifest=manifest,
            stage_complete=stage_complete,
        )
    except (V9Error, KeyError, TypeError, ValueError) as exc:
        analysis = {
            "verdict": "NOT_EVALUABLE",
            "gates": {},
            "analysis_error": str(exc),
        }
    gate_a = analysis.get("gates", {}).get("G9A_1P7B", {})
    gate_b = analysis.get("gates", {}).get("G9B_7B", {})
    note_line = (
        f"G9 VERDICT: {analysis['verdict']} "
        f"G9A_1.7B={gate_a.get('verdict', 'NOT_EVALUABLE')} "
        f"raw_err={display_error(gate_a, 'raw')} "
        f"corrected_err={display_error(gate_a, 'corrected')} "
        f"G9B_7B={gate_b.get('verdict', 'NOT_EVALUABLE')} "
        f"mu0_err={display_error(gate_b, 'mu0')} "
        f"raw_err={display_error(gate_b, 'raw')}"
    )
    readout = {
        "schema": "yeto_outer_mup_v9_g9_readout_v1",
        "created_at_utc": utc_now(),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "predictions_path": str(args.predictions.resolve()),
        "predictions_sha256": sha256_file(args.predictions),
        "source_git_commit": manifest.get("source", {}).get("git_commit"),
        "expected_cells": EXPECTED_CELLS,
        "observed_completed_cells": len(cell_records),
        "observed_by_stage": observed_by_stage,
        "evidence_errors": evidence_errors,
        "cell_records": cell_records,
        "analysis": analysis,
        "gate": {"name": "G9", "verdict": analysis["verdict"]},
        "note_line": note_line,
    }
    write_json_atomic(args.output.resolve(), readout)
    print(note_line)
    return 0 if analysis["verdict"] in ("PASS", "FAIL") else 2


if __name__ == "__main__":
    raise SystemExit(main())
