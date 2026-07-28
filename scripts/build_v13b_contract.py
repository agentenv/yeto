#!/usr/bin/env python3
"""Materialize the prospective G13B regrid contract from the sealed G13 miss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tonight85_common as common
from gatesim_v13b import CENTERS, V13_OFFSETS


SEALED_V13_READOUT_SHA256 = "be69b796d80bf5a32828d803aee65459639643991bb8230eee30e1f6be0eef5a"
V13_CONTRACT_SHA256 = "0947cfbd809c9437843b70ce231b48b200fe7cc92eed1d1dbe92952dd1798223"
STATIC_MANIFEST_SHA256 = "de7f9e78a328b469d2aeeb3613cfadd2b619ea0137a8366af80a7384481610b2"
ORIGINAL_COMMIT = "c310dc69f471c60f208995602c144a34e57f86da"
EXPECTED_EDGES = {
    "T2_mu0": 0, "T2_mu09": 0,
    "T5_mu0": 0, "T5_mu09": 0,
    "T20_mu0": 3, "T20_mu09": 0,
}


def artifact(path: str) -> dict:
    target = common.REPO / path
    return {"path": path, "bytes": target.stat().st_size, "sha256": common.sha256_file(target)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v13-readout", type=Path, required=True)
    parser.add_argument("--v13-manifest", type=Path, required=True)
    parser.add_argument("--v13-result-root", type=Path, required=True)
    parser.add_argument("--gatesim", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if common.sha256_file(args.v13_readout) != SEALED_V13_READOUT_SHA256:
        raise SystemExit("sealed v13 readout hash mismatch")
    if common.sha256_file(args.v13_manifest) != STATIC_MANIFEST_SHA256:
        raise SystemExit("v13 static manifest hash mismatch")
    readout = common.read_json(args.v13_readout)
    manifest = common.read_json(args.v13_manifest)
    if readout["gate"]["verdict"] != "NOT_EVALUABLE" or len(readout["cell_records"]) != 72:
        raise SystemExit("G13 is not the complete sealed NOT_EVALUABLE readout")
    if readout["bootstrap"]["valid"] != 0:
        raise SystemExit("unexpected nonzero G13 bootstrap validity")

    disclosures = {}
    for key, expected_index in EXPECTED_EDGES.items():
        fit = readout["fits"][key]
        observed_index = min(range(4), key=lambda index: fit["seed_mean_losses"][index])
        if observed_index != expected_index:
            raise SystemExit(f"{key}: edge audit changed ({observed_index} != {expected_index})")
        direction = "LOWER_ETA" if observed_index == 0 else "HIGHER_ETA"
        t_text, arm = key.split("_")
        t = int(t_text[1:])
        edge_eta = float(fit["etas"][observed_index])
        if edge_eta != CENTERS[(t, arm)]:
            raise SystemExit(f"{key}: registered center is not the observed edge")
        disclosures[key] = {
            "frozen_fit_status": fit["status"],
            "frozen_fit_accepted": fit["accepted"],
            "quadratic_curvature": fit["a"],
            "vertex_log2_eta": fit["vertex_log2_eta"],
            "etas": fit["etas"],
            "pooled_three_seed_losses": fit["seed_mean_losses"],
            "pooled_discrete_minimum_eta_index": observed_index,
            "pooled_discrete_minimum_eta": edge_eta,
            "edge_direction": direction,
            "v13b_center_equals_this_edge": True,
        }

    token_hashes = set()
    example_hashes = set()
    eval_shapes = set()
    for cell in manifest["cells"]:
        if cell.get("program") != "v13":
            continue
        result_path = args.v13_result_root / cell["cell_id"] / "attempt-1" / "report" / "results.jsonl"
        rows = [json.loads(line) for line in result_path.read_text().splitlines() if line]
        if len(rows) != 1:
            raise SystemExit(f"{cell['cell_id']}: result row count changed")
        row = rows[0]
        token_hashes.add(row["eval_token_ids_hash"])
        example_hashes.add(row["eval_example_ids_hash"])
        eval_shapes.add((row["eval_rows"], row["eval_tokens"]))
    if len(token_hashes) != 1 or len(example_hashes) != 1 or eval_shapes != {(1024, 985008)}:
        raise SystemExit("v13 evaluation identity differs across cells")
    gatesim = common.read_json(args.gatesim)
    if gatesim.get("status") != "PASS" or gatesim.get("evaluable") != gatesim.get("replicates"):
        raise SystemExit("v13b gatesim did not pass every replicate")

    grids = []
    for t in (2, 5, 20):
        coordinate = {"T": t, "S": 2560, "H": 2560 // t}
        for arm, label in (("mu0", "mu0"), ("mu09", "mu0.9")):
            center = CENTERS[(t, arm)]
            coordinate[label] = {
                "center": center,
                "center_source": f"sealed G13 pooled discrete {disclosures[f'T{t}_{arm}']['edge_direction']} edge",
                "offsets_log2": list(V13_OFFSETS),
                "etas": [center * 2.0**offset for offset in V13_OFFSETS],
            }
        grids.append(coordinate)

    contract = {
        "schema": "yeto_outer_mup_v13b_pythia_ultrachat_regrid_prereg_v1",
        "program_id": "outer-mup-v13b-pythia-ultrachat-regrid",
        "status": "REGISTERED_PRE_OUTCOME",
        "registered_at_utc": common.utc_now(),
        "pre_outcome_v13b": True,
        "original_v13_disposition_is_immutable": "G13 VERDICT: NOT_EVALUABLE",
        "purpose": (
            "Repeat the same 72-cell second-family scan on prospectively frozen grids "
            "whose centers equal each curve's sealed v13 discrete edge location."
        ),
        "v13_trigger_disclosure": {
            "registration_commit": ORIGINAL_COMMIT,
            "contract_sha256": V13_CONTRACT_SHA256,
            "static_manifest_sha256": STATIC_MANIFEST_SHA256,
            "sealed_readout_path": str(args.v13_readout),
            "sealed_readout_sha256": SEALED_V13_READOUT_SHA256,
            "completed_cells": 72,
            "failed_cells": 0,
            "gate_conditions": readout["gate"]["conditions"],
            "valid_bootstrap_draws": readout["bootstrap"]["valid"],
            "curve_diagnostics": disclosures,
            "summary": (
                "All six pooled discrete minima were on a registered boundary: "
                "T2/mu0, T2/mu0.9, T5/mu0, T5/mu0.9, and T20/mu0.9 at the low-eta edge; "
                "T20/mu0 at the high-eta edge. Five of six frozen quadratic fits were unaccepted; "
                "the only accepted fit was T20/mu0.9."
            ),
            "arm_comparison_blinding": (
                "Each new arm grid is fixed solely by the eta index of that same arm's pooled "
                "discrete boundary minimum. No loss difference, D ratio, cross-arm ranking, "
                "G13 scientific ordering, or fitted arm-comparison estimand selects a center."
            ),
        },
        "defect_classification": {
            "cause": "GRID_MISCENTERING_FOR_NEW_MODEL_FAMILY",
            "direction": {
                "lower_eta": ["T2/mu0", "T2/mu0.9", "T5/mu0", "T5/mu0.9", "T20/mu0.9"],
                "higher_eta": ["T20/mu0"],
            },
            "eval_stream_or_tokenizer_mismatch_not_observed": {
                "all_72_example_registry_hash": next(iter(example_hashes)),
                "all_72_token_registry_hash": next(iter(token_hashes)),
                "eval_rows": 1024,
                "eval_supervised_tokens": 985008,
                "same_model_tokenizer_path_for_train_and_eval": True,
            },
            "analyzer_validity_vocabulary_changed": False,
        },
        "design": {
            "same_as_v13_except_registered_eta_grids": True,
            "model": "EleutherAI/pythia-160m",
            "model_revision": "50f5173d932e8e61f858120bcb800b97af589f46",
            "data": "HuggingFaceH4/ultrachat_200k frozen v13 train/eval files",
            "T": [2, 5, 20], "S": 2560, "mu": [0.0, 0.9],
            "eta_levels": 4, "seeds": [981, 983, 991],
            "training_seeds": [981981, 983983, 991991], "cells": 72,
            "grids": grids,
        },
        "analysis": {
            "unchanged_v13_core": artifact("scripts/tonight85_analysis.py"),
            "point": "unchanged OLS quadratic in pooled three-seed mean endpoint loss versus log2(eta)",
            "near_bracket_allowance_bits": 0.5,
            "bootstrap": "unchanged 10000 paired three-seed resamples with RNG seed 20260742",
            "minimum_valid_bootstrap_refits": 7500,
            "gate_name": "G13B",
            "success_rule": readout["gate"]["success_rule"],
            "missing_or_unbracketed_rule": "NOT_EVALUABLE; no v13b rung may be added or moved",
        },
        "gate_feasibility": {
            "report": {"path": str(args.gatesim.resolve().relative_to(common.REPO)), "sha256": common.sha256_file(args.gatesim), "bytes": args.gatesim.stat().st_size},
            "result": gatesim,
        },
        "frozen_artifacts": {
            "analyzer": artifact("scripts/analyze_v13b.py"),
            "original_analysis_core": artifact("scripts/tonight85_analysis.py"),
            "gatesim": artifact("scripts/gatesim_v13b.py"),
            "manifest_builder": artifact("scripts/build_v13b_manifest.py"),
            "slot_runner": artifact("scripts/run_slot_v13b.py"),
            "node_preflight": artifact("scripts/preflight_v13b.py"),
            "launch_authorizer": artifact("scripts/authorize_v13b_launch.py"),
        },
        "execution": {
            "isolated_checkout": "/root/yeto-v13b",
            "result_root": "/root/yeto-results-v13b -> /data/yeto-results-v13b",
            "slots": [{"node": "h200-n1", "gpus": [6, 7]}, {"node": "h200-n2", "gpus": list(range(8))}],
            "v11_reserved_queues": {"node": "h200-n1", "gpus": list(range(6)), "must_not_be_stopped_or_modified": True},
            "registration_order": (
                "commit and push contract plus every frozen artifact before manifest, authority, "
                "result-root creation, attempt directory, or GPU process"
            ),
            "analysis_cutoff": "2026-07-28T08:30:00-07:00",
            "cutoff_rule": "if not fully drained and analyzed by cutoff, DEFERRED_ARXIV_V2",
        },
        "retry_contract": {
            "attempt_limit": 2,
            "unit": "all four eta cells sharing T, arm, and training seed",
            "allowed_reasons": ["host_or_gpu_failure", "framework_or_driver_failure", "storage_or_network_failure", "registered_process_timeout_without_valid_endpoint"],
            "finite_loss_or_edge_retry_forbidden": True,
        },
        "immutability": {
            "never_modify": ["v11 trees or processes", "v12 tree or G12 verdict", "v7 trees or sealed verdicts", "original v13 results or G13 verdict"],
            "no_further_regrid_after_v13b_outcome": True,
        },
    }
    common.write_json_atomic(args.output, contract)
    print(json.dumps({"contract": str(args.output), "sha256": common.sha256_file(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
