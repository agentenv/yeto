#!/usr/bin/env python3
"""Deterministically materialize the four tonight-8.5 preregistrations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
REGISTERED_AT = "2026-07-28T00:34:04Z"
SEEDS = [981, 983, 991]
T_VALUES = [2, 5, 20]
MU0_CENTERS = {
    2: 0.07106462666975855,
    5: 0.04341918114042938,
    20: 0.021926218661920484,
}
V12_RAW_CENTERS = {
    2: 0.04225793576801869,
    5: 0.013630854038377603,
    20: 0.0024983032153624908,
}
V13_RAW_CENTERS = {
    2: 0.029627335040308306,
    5: 0.011913059053942006,
    20: 0.0024641979129397013,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def artifact(path: str) -> dict:
    target = REPO / path
    return {"path": path, "bytes": target.stat().st_size, "sha256": sha256_file(target)}


def grid(center: float, offsets: list[float]) -> dict:
    return {
        "center": center,
        "offsets_log2": offsets,
        "etas": [center * 2.0**offset for offset in offsets],
    }


def shared() -> dict:
    return {
        "registered_at_utc": REGISTERED_AT,
        "pre_outcome": True,
        "source_base_commit": "6ca24447a43e52369c5369610e7e873b14af5cc7",
        "registration_commit_rule": (
            "the Git commit containing this contract, its frozen analyzer, simulator, "
            "and launcher must be present on origin/experiment/tonight-8.5-lean before "
            "any corresponding GPU process starts"
        ),
        "outcome_aware_changes_forbidden": True,
        "result_root": "/root/yeto-results-tonight85 -> /data/yeto-results-tonight85",
        "note": "/private/tmp/h200-tonight85-note.md",
        "analysis_cutoff": "2026-07-28T08:30:00-07:00",
        "cutoff_rule": (
            "a gate not fully drained and analyzed by the cutoff is DEFERRED_ARXIV_V2; "
            "partial evidence cannot be promoted into tomorrow's submission"
        ),
        "retry_contract": {
            "attempt_limit": 2,
            "allowed_reasons": [
                "host_or_gpu_failure",
                "framework_or_driver_failure",
                "storage_or_network_failure",
                "registered_process_timeout_without_valid_endpoint",
            ],
            "finite_loss_retry_forbidden": True,
            "scientific_recenter_or_extension_forbidden": True,
            "retry_unit": (
                "all eta cells sharing program, coordinate/T, arm, and training seed; "
                "an anchor retry unit is all three cells at that coordinate"
            ),
        },
        "frozen_execution": {
            "common": artifact("scripts/tonight85_common.py"),
            "manifest_builder": artifact("scripts/build_tonight85_manifest.py"),
            "slot_runner": artifact("scripts/run_slot_tonight85.py"),
            "mac_conductor": artifact("scripts/tonight85_conductor.py"),
            "gate_simulator": artifact("scripts/gatesim_tonight85.py"),
        },
    }


def v11(gatesim: dict) -> dict:
    return {
        "schema": "yeto_outer_mup_v11_ratio_transport_prereg_v1",
        "program_id": "outer-mup-v11-ratio-transport",
        "status": "REGISTERED_PRE_ANCHOR",
        **shared(),
        "question": (
            "Can the raw-vs-mu0 ratio structure extracted from the fitted 135M "
            "surface predict raw eta-star after only a three-cell mu0 re-anchor "
            "at two never-measured coordinates?"
        ),
        "coordinates": {
            "smollm2_135m_t80": {
                "model": "HuggingFaceTB/SmolLM2-135M",
                "model_revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
                "T": 80,
                "S": 40960,
                "H": 512,
                "never_measured_assertion": True,
                "mu0_placement_center": gatesim["G11"]["placements"][
                    "smollm2_135m_t80"
                ],
            },
            "smollm2_1p7b_t40": {
                "model": "HuggingFaceTB/SmolLM2-1.7B",
                "model_revision": "effd688a12921b4cc83e3312b6feb579f70f9c71",
                "T": 40,
                "S": 20480,
                "H": 512,
                "never_measured_assertion": True,
                "mu0_placement_center": gatesim["G11"]["placements"][
                    "smollm2_1p7b_t40"
                ],
            },
        },
        "ratio_rule": {
            "fitted_surface": {
                "family": "F3",
                "formula": "log2(D)=gamma+alpha*u+beta*v+epsilon*u^2",
                "u": "(T-5)/5",
                "v": "log2(S/5120)",
                "coefficients": [
                    1.3489008177233357,
                    -0.9098513603667141,
                    -0.10723867757601385,
                    0.16020840966569636,
                ],
                "source": artifact(
                    "experiment-specs/outer-mup-v9-frozen-coefficients.json"
                ),
            },
            "far_horizon_extraction": (
                "Evaluate F3 only at constant-H donors (T=5,S=2560) and "
                "(T=10,S=5120); fit the unique D(T)=1+(D5-1)*(T/5)^p with "
                "p=log2((D10-1)/(D5-1)); do not evaluate F3's u^2 polynomial "
                "outside its registered T<=20 domain"
            ),
            "D_T40": gatesim["G11"]["ratio_rule"]["D_T40"],
            "D_T80": gatesim["G11"]["ratio_rule"]["D_T80"],
            "prediction": "eta_star_raw=eta_star_mu0_anchor*(1-0.9)*D(T)",
        },
        "anchor_probe": {
            "cells_per_coordinate": 3,
            "seed": 967,
            "training_seed": 967967,
            "mu": 0.0,
            "offsets_log2": [-0.75, 0.0, 0.75],
            "estimator": "OLS quadratic endpoint loss versus log2(eta)",
            "acceptance": (
                "a>0 and the unconstrained vertex lies within 0.5 bits of the "
                "three-rung range; an unaccepted fit blocks prediction and makes G11 NOT_EVALUABLE"
            ),
            "excluded_from_G11_ground_truth_fit": True,
        },
        "prediction_seal": {
            "artifact": "experiment-specs/outer-mup-v11-sealed-predictions.json",
            "must_be_hash_committed_after_anchor_before_truth": True,
            "must_be_committed_and_pushed_from_mac": True,
            "ground_truth_launch_is_forbidden_until_origin_contains_prediction_commit": True,
        },
        "ground_truth": {
            "arm": "raw",
            "mu": 0.9,
            "outer_optimizer": "nesterov",
            "eta_levels": 5,
            "offsets_from_sealed_prediction_log2": [-1.0, -0.5, 0.0, 0.5, 1.0],
            "seeds": [971, 977],
            "training_seeds": [971971, 977977],
            "cells_per_coordinate": 10,
            "total_cells_including_anchors": 26,
            "estimator": "OLS quadratic over the pooled two-seed mean endpoint loss",
            "near_bracket_allowance_bits": 0.5,
        },
        "gate": {
            "name": "G11",
            "absolute_prediction_error_band_bits": 0.35,
            "success_rule": (
                "PASS iff both coordinates are complete/evaluable and absolute "
                "log2(predicted eta-star / fitted ground-truth eta-star)<=0.35 "
                "at at least one of the two coordinates"
            ),
            "missing_or_unbracketed_rule": "NOT_EVALUABLE; never extend or recenter",
        },
        "gate_feasibility": {
            "report": artifact("experiment-specs/tonight85-gatesim.json"),
            "G11": gatesim["G11"],
        },
        "frozen_analyzer": artifact("scripts/analyze_v11.py"),
        "frozen_analysis_core": artifact("scripts/tonight85_analysis.py"),
        "frozen_prediction_builder": artifact("scripts/build_v11_predictions.py"),
    }


def scan_contract(program: str, gatesim: dict, v13_inputs: dict) -> dict:
    is_v12 = program == "v12"
    offsets = [-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0] if is_v12 else [-1.5, -0.5, 0.5, 1.5]
    raw_centers = V12_RAW_CENTERS if is_v12 else V13_RAW_CENTERS
    grids = []
    for t in T_VALUES:
        grids.append(
            {
                "T": t,
                "S": 2560,
                "H": 2560 // t,
                "mu0": grid(MU0_CENTERS[t], offsets),
                "mu0.9": grid(raw_centers[t], offsets),
            }
        )
    base = {
        "schema": f"yeto_outer_mup_{program}_prereg_v1",
        "program_id": f"outer-mup-{program}-"
        + ("heavy-ball" if is_v12 else "pythia-ultrachat"),
        "status": "REGISTERED",
        **shared(),
        "design": {
            "T": T_VALUES,
            "S": 2560,
            "mu": [0.0, 0.9],
            "eta_levels": 4,
            "seeds": SEEDS,
            "training_seeds": [981981, 983983, 991991],
            "cells": 72,
            "grids": grids,
        },
        "D_definition": "D(T)=[eta_star(T,mu=0.9)/eta_star(T,mu=0)]/(1-0.9)",
        "estimator": {
            "point": "OLS quadratic in pooled three-seed mean endpoint loss versus log2(eta)",
            "near_bracket_allowance_bits": 0.5,
            "bootstrap": (
                "10000 paired training-seed resamples shared across all six curves; "
                "rng seed 20260741 for G12 and 20260742 for G13"
            ),
        },
        "gate": {
            "name": "G12" if is_v12 else "G13",
            "success_rule": (
                "D(2)>D(5)>D(20) in point estimates and both paired 95% "
                "percentile CIs for log2(D(2))-log2(D(5)) and "
                "log2(D(5))-log2(D(20)) have lower endpoint >0"
            ),
            "minimum_valid_bootstrap_refits": 7500,
            "missing_or_unbracketed_rule": "NOT_EVALUABLE; no rung may be added or moved",
        },
        "gate_feasibility": {
            "report": artifact("experiment-specs/tonight85-gatesim.json"),
            "result": gatesim["G12_G13"][program],
        },
        "frozen_analyzer": artifact(f"scripts/analyze_{program}.py"),
        "frozen_analysis_core": artifact("scripts/tonight85_analysis.py"),
        "comparison_role": "descriptive only; not part of the gate",
    }
    if is_v12:
        base.update(
            {
                "question": (
                    "Under classical heavy-ball outer momentum without Nesterov "
                    "lookahead, does D(T) decrease over T={2,5,20}?"
                ),
                "model": {
                    "id": "HuggingFaceTB/SmolLM2-135M",
                    "revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
                    "path": "/root/yeto-data/model",
                },
                "data": {
                    "corpus": "trl-lib/Capybara",
                    "train_path": "/root/yeto-data/outer-mup-v3/scale-s2560/raw/train.jsonl",
                    "train_sha256": "e680a29ea8c8fc7c99efdceb4f62e485d3eed1ac2afd15bab43b506cb3f4ecaf",
                    "eval_path": "/root/yeto-data/outer-mup-v3/scale-s2560/raw/eval.jsonl",
                    "eval_sha256": "533838a0564b13519956a044d23ed8db6705ddc7ae5f0ddb96538f49460bcebc",
                },
                "outer_convention": {
                    "flag": "--outer-optimizer heavy-ball",
                    "recursion": "b_t=mu*b_(t-1)+delta_t; theta_t=theta_(t-1)-eta*b_t",
                    "no_nesterov_lookahead": True,
                    "implementation_commit": "44bad06991c8ea134ccde373c47c98b06b89c062",
                    "implementation": artifact("syncer/src/merge.rs"),
                    "cli": artifact("syncer/src/main.rs"),
                    "harness": artifact("scripts/compare_diloco.py"),
                    "off_path_test": "merge::tests::heavy_ball_off_path_is_bit_identical_to_production_nesterov",
                    "off_path_result": "PASS",
                },
                "finite_horizon_constants": {
                    "terminal_multiplier": "(1-mu^T)/(1-mu)",
                    "lean_citation": (
                        "FiniteHorizonOuter.lean terminalMultiplier .heavyBall; "
                        "heavyBallCoeff is the accumulated sum and heavyBallCoeff_eq_sum_div "
                        "expands it as sum_{t<T}(1-mu^(t+1))/(1-mu)"
                    ),
                    "descriptive_nesterov_D": {
                        "T2": 4.153638091620729,
                        "T5": 2.554472462361996,
                        "T20": 1.168877658562076,
                    },
                },
            }
        )
    else:
        verified_model = {
            **v13_inputs["model"],
            "architecture": "GPTNeoXForCausalLM",
            "exact_parameters": 162322944,
            "snapshot_load_dtype": "torch.float16",
            "production_h200_dtype": "torch.bfloat16 via accelerator_model_dtype",
            "cpu_load_verified_on": ["h200-n1", "h200-n2"],
        }
        base.update(
            {
                "question": (
                    "Does the Nesterov D(T) monotonicity reproduce in a second "
                    "model family on a different public corpus?"
                ),
                "model": verified_model,
                "data": {
                    **v13_inputs["source"],
                    "input_builder": artifact("scripts/prepare_v13_ultrachat.py"),
                    "selection": v13_inputs["selection"],
                    "files": v13_inputs["files"],
                    "input_manifest_sha256": "b77b353dcc8a548c8e4ff31a252ed74919f325482795a15732b4ae9261b08164",
                    "tokenizer_smoke": v13_inputs["tokenizer_smoke"],
                    "HF_DATASETS_CACHE": "/data/hf-datasets-cache",
                },
                "outer_convention": {
                    "flag": "--outer-optimizer nesterov",
                    "mu0.9_has_no_bias_correction": True,
                },
                "descriptive_reference": {
                    "family": "SmolLM2-135M on Capybara",
                    "D": {
                        "T2": 4.153638091620729,
                        "T5": 2.554472462361996,
                        "T20": 1.168877658562076,
                    },
                },
            }
        )
    return base


def v7_amendment() -> dict:
    return {
        "schema": "yeto_outer_mup_v7_lean_scope_amendment_v1",
        "program_id": "outer-mup-v7-27b-lora",
        "status": "ADOPTED_PRE_OUTCOME_SCOPE_REDUCTION",
        **shared(),
        "original_registration": {
            "branch": "origin/experiment/outer-mup-v7-27b-lora",
            "commit": "a8f113b0858d17749dfb08ae9702bc29effcc395",
            "contract": artifact("experiment-specs/outer-mup-v7-27b-lora-prereg.json"),
        },
        "pre_outcome_state": {
            "checked_at_utc": REGISTERED_AT,
            "v7_processes_on_h200_n1_n2": 0,
            "v7_evidence_files_on_h200_n1_n2": 0,
            "v7_scientific_losses_seen": False,
        },
        "prelaunch_validator_repair": {
            "status": "ADOPTED_BEFORE_ANY_V7_PROCESS_OR_EVIDENCE",
            "trigger": (
                "The v7 validator loaded the event tape into local variable rows but "
                "left tape_rows empty before barrier-registry replay, making every "
                "valid registry compare unequal to the validated tape."
            ),
            "repair": (
                "Assign tape_rows=rows immediately after the already-registered "
                "event-tape JSONL parse; training, commands, cells, estimators, and "
                "scientific outcomes are unchanged."
            ),
            "validator": artifact("scripts/run_node_v7.py"),
            "scientific_change": False,
        },
        "reason": (
            "Use the extended lease for a descriptive T=5 pilot/spot check while "
            "preserving the 8-GPU island requirement; withdraw the confirmatory G7 "
            "main grid before any v7 outcome rather than underpowering it after outcomes."
        ),
        "retained_wiring_smoke": {
            "scientific_cell": False,
            "seed": 683,
            "S": 64,
            "H": 16,
            "T": 4,
            "mu": 0.0,
            "eta": 0.28,
            "required_before_pilot": True,
        },
        "pilot": {
            "cells": 3,
            "seed": 691,
            "training_seed": 691691,
            "T": 5,
            "S": 2560,
            "H": 512,
            "mu": 0.0,
            "etas": [0.14, 0.28, 0.56],
            "selection_rule": (
                "retain the original v7 quadratic/fallback pilot rule exactly; "
                "pilot cells are excluded from any confirmatory claim"
            ),
        },
        "raw_spot_check": {
            "formula": "predicted_eta=selected_pilot_mu0_eta*0.1*1.7416157949788522",
            "D5_source": "final disclosed G4C constant already frozen in v7",
            "offsets_log2": [-0.5, 0.5],
            "interpretation": "one original-grid rung below and above the prediction",
            "seeds": [701, 709],
            "training_seeds": [701701, 709709],
            "cells": 4,
            "prediction_artifact_must_be_committed_and_pushed_from_mac_before_launch": True,
        },
        "scope": {
            "scientific_cells": 7,
            "plus_wiring_smoke": 1,
            "island_width_gpus": 8,
            "T": [5],
            "confirmatory_G7_withdrawn_pre_outcome": True,
            "analysis": "descriptive; partial result acceptable and no PASS/FAIL gate is defined",
        },
        "forbidden": (
            "No raw rung, seed, D constant, pilot eta, or fallback rule may be "
            "changed after any pilot or raw endpoint is observed"
        ),
        "frozen_prediction_builder": artifact("scripts/build_v7_lean_prediction.py"),
        "frozen_descriptive_analyzer": artifact("scripts/analyze_v7_lean.py"),
        "frozen_analysis_core": artifact("scripts/tonight85_analysis.py"),
    }


def main() -> int:
    gatesim_path = REPO / "experiment-specs/tonight85-gatesim.json"
    inputs_path = Path("/private/tmp/tonight85-v13-input-manifest.json")
    if (
        sha256_file(gatesim_path)
        != "1c4841b86120453364af85756bece1982f20f018a0985b5de6199feec0b8b47e"
    ):
        raise SystemExit("gate simulation hash changed")
    if (
        sha256_file(inputs_path)
        != "b77b353dcc8a548c8e4ff31a252ed74919f325482795a15732b4ae9261b08164"
    ):
        raise SystemExit("v13 staged input manifest hash changed")
    gatesim = json.loads(gatesim_path.read_text())
    v13_inputs = json.loads(inputs_path.read_text())
    specs = {
        "outer-mup-v11-ratio-transport-prereg.json": v11(gatesim),
        "outer-mup-v12-heavy-ball-prereg.json": scan_contract(
            "v12", gatesim, v13_inputs
        ),
        "outer-mup-v13-pythia-ultrachat-prereg.json": scan_contract(
            "v13", gatesim, v13_inputs
        ),
        "outer-mup-v7-lean-scope-amendment.json": v7_amendment(),
    }
    for name, value in specs.items():
        write(REPO / "experiment-specs" / name, value)
    print(
        json.dumps(
            {name: sha256_file(REPO / "experiment-specs" / name) for name in specs},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
