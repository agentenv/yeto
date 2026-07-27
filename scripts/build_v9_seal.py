#!/usr/bin/env python3
"""Build the prospective V9 preregistration and sealed scale predictions.

This is the single mechanical seal entry point.  It consumes the already
frozen G6-selected empirical coefficients, immutable G4C anchors, the
pre-outcome gate simulation, and a two-node no-exposure proof.  It emits the
JSON preregistration, sealed prediction JSON, SHA-256 sidecars, and the human
registration Markdown.  It never reads a V9 result directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_v9_predictions import build_predictions  # noqa: E402
from v9_common import (  # noqa: E402
    V9Error,
    read_json,
    sha256_file,
    utc_now,
    write_json_atomic,
)


CONTRACT_SCHEMA = "yeto_outer_mup_v9_sealed_scale_prereg_v1"
SELECTION_SCHEMA = "yeto_outer_mup_v6_selected_surfaces_v1"
GATESIM_SCHEMA = "yeto_outer_mup_v9_gate_simulation_v1"
PRESEAL_SCHEMA = "yeto_outer_mup_v9_preseal_proof_v1"
G4C_SCHEMA = "yeto_outer_mup_v4c_g4c_readout_v2"
G4C_MANIFEST_SCHEMA = "yeto_outer_mup_v4c_seedpower_launch_manifest_v1"

SMOLLM_135M_PARAMETERS = 134_515_008
SMOLLM_1P7B_PARAMETERS = 1_711_376_384
QWEN_7B_PARAMETERS = 7_615_616_512
SMOLLM_1P7B_PATH = (
    "/root/yeto-hf-cache/hub/models--HuggingFaceTB--SmolLM2-1.7B/"
    "snapshots/effd688a12921b4cc83e3312b6feb579f70f9c71"
)
QWEN_7B_PATH = (
    "/root/yeto-hf-cache/hub/models--Qwen--Qwen2.5-7B/"
    "snapshots/d149729398750b98c0af14eb82c78cfe92750796"
)

SMOLLM_REQUIRED_FILES = (
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)

# These raw hashes were computed independently on h200-n1 and h200-n2 and
# compared equal before sealing.  The slot preflight re-hashes every file.
QWEN_FILES = {
    "config.json": {
        "bytes": 686,
        "sha256": "267ce68584c5f24c3b267d934db2de68dd21d1ca677fb78ed809eb60067f7642",
    },
    "generation_config.json": {
        "bytes": 138,
        "sha256": "8c970692323e3ea0e9b8b0a4dca79388d31226e41f83c9fd6014804280ebf6e8",
    },
    "merges.txt": {
        "bytes": 1_671_839,
        "sha256": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    },
    "model-00001-of-00004.safetensors": {
        "bytes": 3_945_441_440,
        "sha256": "b6d4b6e881d17e9235934419b1c17a8bb0f1108579404e0a468a6afdeae6e868",
    },
    "model-00002-of-00004.safetensors": {
        "bytes": 3_864_726_352,
        "sha256": "687e8d08dc82b5f7d69322f1d8691d49b59db9039b07e7ea71010fb6434d5274",
    },
    "model-00003-of-00004.safetensors": {
        "bytes": 3_864_726_424,
        "sha256": "32652eb23537597cca8c7a56ec85b5d5f1c7245b16269a904d57e85aa1aff30e",
    },
    "model-00004-of-00004.safetensors": {
        "bytes": 3_556_377_672,
        "sha256": "b5a2298dddcf228129975a9a271912a9f8dc817957deecde523e3154481ec3fb",
    },
    "model.safetensors.index.json": {
        "bytes": 27_752,
        "sha256": "624bf7c47cd12468fdc16e38a47cf4f19e0415b859a223ba3c027eed2f0e1028",
    },
    "tokenizer.json": {
        "bytes": 7_031_645,
        "sha256": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    },
    "tokenizer_config.json": {
        "bytes": 7_228,
        "sha256": "c91efca15ceff6e9ee9424db58a6f59cd41294e550a86cbd07e3c1fb500b34f9",
    },
    "vocab.json": {
        "bytes": 2_776_833,
        "sha256": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    },
}


def artifact_record(path: Path, canonical_path: str | None = None) -> dict:
    return {
        "path": canonical_path or str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def frozen_code() -> dict:
    paths = {
        "seal_builder": "scripts/build_v9_seal.py",
        "surface_freezer": "scripts/freeze_v6_selection.py",
        "gate_simulator": "scripts/gatesim_v9.py",
        "prediction_builder": "scripts/build_v9_predictions.py",
        "common_math": "scripts/v9_common.py",
        "launch_manifest_builder": "scripts/build_v9_launch_manifest.py",
        "launch_authorizer": "scripts/authorize_v9_launch.py",
        "stage_launcher": "scripts/launch_v9_stage.py",
        "fleet_checker": "scripts/check_v9_fleet.py",
        "retry_authorizer": "scripts/authorize_v9_retry.py",
        "qwen_smoke": "scripts/smoke_v9_qwen.py",
        "preseal_checker": "scripts/check_v9_preseal.py",
        "analyzer": "scripts/analyze_v9.py",
    }
    return {
        label: {"path": path, "sha256": sha256_file(REPO / path)}
        for label, path in paths.items()
    }


def g4c_file(g4c_manifest: dict, exact_path: str) -> dict:
    matches = [
        record
        for record in g4c_manifest.get("inputs", {}).get("files", [])
        if record.get("path") == exact_path
    ]
    if len(matches) != 1:
        raise V9Error(f"G4C manifest has {len(matches)} records for {exact_path}")
    record = matches[0]
    return {
        "path": exact_path,
        "bytes": int(record["bytes"]),
        "sha256": str(record["sha256"]),
    }


def smollm_files(g4c_manifest: dict) -> dict:
    return {
        name: {
            key: value
            for key, value in g4c_file(
                g4c_manifest, f"{SMOLLM_1P7B_PATH}/{name}"
            ).items()
            if key != "path"
        }
        for name in SMOLLM_REQUIRED_FILES
    }


def validate_inputs(
    selection: dict,
    g4c: dict,
    g4c_manifest: dict,
    gatesim: dict,
    preseal: dict,
) -> None:
    if selection.get("schema") != SELECTION_SCHEMA or selection.get("status") != "FROZEN":
        raise V9Error("selected coefficients are not the frozen V6 selection")
    if selection.get("selection_uses_heldout_outcomes") is not False:
        raise V9Error("selected coefficients are not training-only")
    families = {
        arm: selection.get("selected_surfaces", {}).get(arm, {}).get("family_id")
        for arm in ("raw", "corrected")
    }
    if families != {"raw": "F3", "corrected": "F1"}:
        raise V9Error(f"unexpected G6-selected families: {families}")
    if g4c.get("schema") != G4C_SCHEMA or g4c.get("gate", {}).get("verdict") != "PASS":
        raise V9Error("G4C anchors are not the immutable PASS readout")
    if g4c_manifest.get("schema") != G4C_MANIFEST_SCHEMA:
        raise V9Error("not the G4C launch manifest")
    model = g4c_manifest.get("inputs", {}).get("model", {})
    if (
        model.get("exact_parameters") != SMOLLM_1P7B_PARAMETERS
        or model.get("revision") != "effd688a12921b4cc83e3312b6feb579f70f9c71"
    ):
        raise V9Error("G4C manifest binds another 1.7B model")
    if gatesim.get("schema") != GATESIM_SCHEMA or gatesim.get("status") != "PASS":
        raise V9Error("V9 gate simulation is not PASS")
    if gatesim.get("verification_loss_seen") is not False:
        raise V9Error("gate simulation does not assert pre-verification execution")
    if (
        preseal.get("schema") != PRESEAL_SCHEMA
        or preseal.get("status") != "PASS"
        or preseal.get("verification_loss_seen") is not False
        or preseal.get("result_root_absent_on_both_nodes") is not True
    ):
        raise V9Error("preseal proof does not establish zero V9 exposure")


def build_contract(
    *,
    selection: dict,
    selection_path: Path,
    g4c_path: Path,
    g4c_manifest: dict,
    g4c_manifest_path: Path,
    gatesim: dict,
    gatesim_path: Path,
    preseal_path: Path,
    created_at_utc: str,
) -> dict:
    gate_a = gatesim["gates"]["G9A_1P7B"]
    gate_b = gatesim["gates"]["G9B_7B"]
    offsets_a = [float(value) for value in gate_a["ladder_offsets_log2"]]
    offsets_b = [float(value) for value in gate_b["ladder_offsets_log2"]]
    if len(offsets_a) != 4 or len(offsets_b) != 3:
        raise V9Error("gatesim changed the registered 4-point/3-point design")
    train_path = "/root/yeto-data/outer-mup-v3/scale-s2560/raw/train.jsonl"
    eval_path = "/root/yeto-data/outer-mup-v3/scale-s2560/raw/eval.jsonl"
    scale_path = "/root/yeto-data/outer-mup-v3/scale-s2560/manifest.json"
    qwen_smoke_path = (
        "/root/yeto-data/outer-mup-v3/scale-s2560/qwen2.5-7b/m4/"
        "learner-00.safetensors"
    )
    source_artifacts = {
        "v6_selection": artifact_record(
            selection_path,
            "experiment-specs/outer-mup-v9-frozen-coefficients.json",
        ),
        "g4c_readout": artifact_record(g4c_path, "h200-n1:/root/g4c-readout.json"),
        "g4c_manifest": artifact_record(
            g4c_manifest_path,
            "h200-n1:/root/yeto-results-v4c/_controller/launch-v4c/"
            "launch-manifest-v4c.json",
        ),
        "gate_simulation": artifact_record(
            gatesim_path,
            "experiment-specs/outer-mup-v9-sealed-scale-gatesim.json",
        ),
        "preseal_proof": artifact_record(
            preseal_path,
            "experiment-specs/outer-mup-v9-preseal-proof.json",
        ),
    }
    code = frozen_code()
    contract = {
        "schema": CONTRACT_SCHEMA,
        "status": "SEALED_PRE_VERIFICATION",
        "registered_at_utc": created_at_utc,
        "program_id": "outer-mup-v9-sealed-scale",
        "objective": (
            "Prospectively test empirical optimal-outer-LR transport at never-run "
            "SmolLM2-1.7B T=10 and minimally at Qwen2.5-7B T=5."
        ),
        "verification_loss_seen": False,
        "source_artifacts": source_artifacts,
        "referee_mechanism_resolution": {
            "mechanism": "SURFACE-FALLBACK",
            "spectral_mechanism_confirmed_on_full_data": False,
            "confirmed_referee_lanes": [],
            "selected_empirical_surfaces": {"raw": "F3", "corrected": "F1"},
            "coefficient_source": "complete 540-cell mechfit refits equal to G6",
            "claim_boundary": (
                "The paper may claim prospective empirical surface transport only; "
                "it must not claim that the rejected static spectral closure is the "
                "identified mechanism."
            ),
            "referee_note": "/private/tmp/h200-referee-note.md",
            "theory_note": "docs/theory-lane-A.md",
        },
        "models": {
            "smollm2_135m": {
                "id": "HuggingFaceTB/SmolLM2-135M",
                "revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
                "exact_parameters": SMOLLM_135M_PARAMETERS,
                "role": "V6 parameter-scale anchor only",
            },
            "smollm2_1p7b": {
                "id": "HuggingFaceTB/SmolLM2-1.7B",
                "revision": "effd688a12921b4cc83e3312b6feb579f70f9c71",
                "exact_parameters": SMOLLM_1P7B_PARAMETERS,
                "path": SMOLLM_1P7B_PATH,
                "files": smollm_files(g4c_manifest),
            },
            "qwen2p5_7b": {
                "id": "Qwen/Qwen2.5-7B",
                "revision": "d149729398750b98c0af14eb82c78cfe92750796",
                "exact_parameters": QWEN_7B_PARAMETERS,
                "path": QWEN_7B_PATH,
                "files": QWEN_FILES,
            },
        },
        "machine_inputs": {
            "training_jsonl": g4c_file(g4c_manifest, train_path),
            "development_jsonl": g4c_file(g4c_manifest, eval_path),
            "scale_manifest": g4c_file(g4c_manifest, scale_path),
            "qwen_smoke_packed_input": {
                "path": qwen_smoke_path,
                "bytes": 3_932_496,
                "sha256": (
                    "0f851c37900ca9b6d70be96852a692a761a6b5ae33b8d61921f1e316c5d30bca"
                ),
            },
            "cache_environment": {
                "HF_HOME": "/root/yeto-hf-cache",
                "HF_HUB_CACHE": "/root/yeto-hf-cache/hub",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
        },
        "design": {
            "seeds": [901, 907],
            "stage_order": ["stage_1p7b", "stage_7b"],
            "stage_7b_requires_complete_1p7b_drain": True,
            "stage_1p7b": {
                "model": "smollm2_1p7b",
                "coordinate": {"T": 10, "S": 5120, "H": 512},
                "targets": ["raw", "corrected"],
                "ladder_offsets_log2": offsets_a,
                "eta_levels_per_target": 4,
                "cell_count": 16,
                "never_run_coordinate_assertion": True,
            },
            "stage_7b": {
                "model": "qwen2p5_7b",
                "coordinate": {"T": 5, "S": 2560, "H": 512},
                "targets": ["mu0", "raw"],
                "ladder_offsets_log2": offsets_b,
                "eta_levels_per_target": 3,
                "cell_count": 12,
                "learner_count": 4,
                "learner_gpu_count": 1,
            },
        },
        "gate_feasibility": {
            "status": "PASS",
            "narrow_draft_disclosure": (
                "The unregistered narrower draft failed on complete-data noise; "
                "the final symmetric widths are the smallest gatesimmed candidates "
                "meeting P_eval>=0.8 without adding cells."
            ),
            "G9A_1P7B": {
                "P_eval": gate_a["P_eval"],
                "P_pass_given_evaluable": gate_a[
                    "P_pass_given_evaluable_under_centered_null"
                ],
                "ladder_offsets_log2": offsets_a,
            },
            "G9B_7B": {
                "P_eval": gate_b["P_eval"],
                "P_pass_given_evaluable": gate_b[
                    "P_pass_given_evaluable_under_centered_null"
                ],
                "ladder_offsets_log2": offsets_b,
            },
        },
        "analysis_contract": {
            "frozen_analyzer": code["analyzer"],
            "point_estimator": "OLS quadratic in pooled two-seed mean loss vs log2(eta)",
            "bootstrap": gatesim["bootstrap"],
            "gates": {
                "G9A_1P7B": {
                    "stage": "stage_1p7b",
                    "targets": ["raw", "corrected"],
                    "near_bracket_allowance_bits": gate_a[
                        "near_bracket_allowance_bits"
                    ],
                    "minimum_valid_bootstrap_refits": gate_a[
                        "registered_minimum_valid_bootstrap_refits"
                    ],
                    "absolute_error_band_bits": gate_a[
                        "registered_absolute_error_band_bits"
                    ],
                    "pass_rule": (
                        "PASS iff complete/evaluable and both point eta-star errors "
                        "are within their registered absolute log2 bands"
                    ),
                },
                "G9B_7B": {
                    "stage": "stage_7b",
                    "targets": ["mu0", "raw"],
                    "near_bracket_allowance_bits": gate_b[
                        "near_bracket_allowance_bits"
                    ],
                    "minimum_valid_bootstrap_refits": gate_b[
                        "registered_minimum_valid_bootstrap_refits"
                    ],
                    "absolute_error_band_bits": gate_b[
                        "registered_absolute_error_band_bits"
                    ],
                    "pass_rule": (
                        "PASS iff complete/evaluable and both point eta-star errors "
                        "are within their registered absolute log2 bands"
                    ),
                },
            },
            "overall_rule": (
                "G9 PASS iff G9A and G9B PASS; FAIL iff both are evaluable and at "
                "least one fails; otherwise NOT_EVALUABLE"
            ),
            "outcome_aware_edits_after_seal_forbidden": True,
        },
        "retry_contract": {
            "attempts": [1, 2],
            "attempt_2_requires_separate_loss_blind_authority": True,
            "retry_unit": "whole seed-by-arm curve",
            "allowed_reasons": [
                "host_or_gpu_failure",
                "framework_or_driver_failure",
                "storage_or_network_failure",
                "registered_process_timeout_without_valid_endpoint",
            ],
            "scientific_recenter_or_extension_forbidden": True,
        },
        "wall_clock": {
            "ceiling_seconds": 108_000,
            "stage_1p7b_cell_timeout_minutes": 480,
            "stage_7b_cell_timeout_minutes": 720,
        },
        "storage": {
            "result_link": "/root/yeto-results-v9",
            "lvm_target": "/data/yeto-results-v9",
            "minimum_free_bytes": 1_000_000_000_000,
        },
        "execution": {
            "nodes": ["h200-n1", "h200-n2"],
            "gpus_per_node": 8,
            "stage_1p7b": "16 simultaneous one-GPU cells",
            "stage_7b": "four 4-GPU queues, three cells each; M=4 single-GPU learners",
            "cache_environment_required": True,
            "registration_commit_must_be_pushed_before_authority": True,
        },
        "frozen_code": code,
    }
    return contract


def markdown(contract: dict, predictions: dict, prediction_sha256: str) -> str:
    gate_a = contract["analysis_contract"]["gates"]["G9A_1P7B"]
    gate_b = contract["analysis_contract"]["gates"]["G9B_7B"]
    rows = []
    for stage, label in (("stage_1p7b", "SmolLM2-1.7B"), ("stage_7b", "Qwen2.5-7B")):
        for arm, target in predictions[stage]["targets"].items():
            rows.append(
                f"| {label} | {arm} | `{target['predicted_eta_star']:.12g}` | "
                f"`{target['verification_etas']}` | "
                f"{target['registered_absolute_error_band_bits']:.3f} |"
            )
    return "\n".join(
        [
            "# Outer-muP V9 sealed scale preregistration",
            "",
            "Status: **SEALED BEFORE ANY V9 VERIFICATION CELL**.",
            "",
            "`MECHANISM: SURFACE-FALLBACK`",
            "",
            "The referee confirmed no spectral lane on full data. The point predictions "
            "therefore use the G6-selected empirical surfaces: raw F3 and corrected F1, "
            "with coefficients copied from the complete mechfit refits after exact equality "
            "to G6. This registration supports an empirical transport claim only; it does "
            "not support a static-spectral-mechanism claim.",
            "",
            "## Frozen predictions",
            "",
            "| model | arm | predicted eta* | verification eta ladder | error band (bits) |",
            "|---|---:|---:|---|---:|",
            *rows,
            "",
            f"Sealed prediction SHA-256: `{prediction_sha256}`.",
            "",
            "The 1.7B T=10/S=5120/H=512 raw and corrected coordinates were never run "
            "before this seal. The 7B stage is T=5/S=2560/H=512 with seeds {901,907}, "
            "M=4, and one GPU per learner.",
            "",
            "## Prospective gates",
            "",
            f"- G9A uses bands `{gate_a['absolute_error_band_bits']}`, near-bracket "
            f"allowance `{gate_a['near_bracket_allowance_bits']}` bits, and at least "
            f"`{gate_a['minimum_valid_bootstrap_refits']}` valid refits.",
            f"- G9B uses bands `{gate_b['absolute_error_band_bits']}`, near-bracket "
            f"allowance `{gate_b['near_bracket_allowance_bits']}` bits, and at least "
            f"`{gate_b['minimum_valid_bootstrap_refits']}` valid refits.",
            "- The exact two-seed 10,000-draw bootstrap groups are copied from the "
            "pre-outcome gatesim. No post-seal recentering, ladder edit, band edit, "
            "target substitution, or analyzer edit is permitted.",
            "",
            "## Execution order",
            "",
            "1. Push the exact registration commit.",
            "2. Run and drain all 16 1.7B cells.",
            "3. Run the registered Qwen one-step admission smoke, then run and drain all "
            "12 7B cells in four 4-GPU queues.",
            "4. Run the frozen analyzer and publish G9 without outcome-aware edits.",
            "",
        ]
    )


def sidecar(path: Path) -> str:
    digest = sha256_file(path)
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n")
    return digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-coefficients", type=Path, required=True)
    parser.add_argument("--g4c-readout", type=Path, required=True)
    parser.add_argument("--g4c-manifest", type=Path, required=True)
    parser.add_argument("--gatesim", type=Path, required=True)
    parser.add_argument("--preseal-proof", type=Path, required=True)
    parser.add_argument("--contract-json", type=Path, required=True)
    parser.add_argument("--contract-md", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--created-at-utc")
    args = parser.parse_args()
    outputs = [
        args.contract_json,
        args.contract_md,
        args.predictions,
        args.contract_json.with_suffix(args.contract_json.suffix + ".sha256"),
        args.contract_md.with_suffix(args.contract_md.suffix + ".sha256"),
        args.predictions.with_suffix(args.predictions.suffix + ".sha256"),
    ]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise SystemExit(f"refusing existing seal outputs: {existing}")
    try:
        selection = read_json(args.selected_coefficients)
        g4c = read_json(args.g4c_readout)
        g4c_manifest = read_json(args.g4c_manifest)
        gatesim = read_json(args.gatesim)
        preseal = read_json(args.preseal_proof)
        validate_inputs(selection, g4c, g4c_manifest, gatesim, preseal)
        created = args.created_at_utc or utc_now()
        contract = build_contract(
            selection=selection,
            selection_path=args.selected_coefficients,
            g4c_path=args.g4c_readout,
            g4c_manifest=g4c_manifest,
            g4c_manifest_path=args.g4c_manifest,
            gatesim=gatesim,
            gatesim_path=args.gatesim,
            preseal_path=args.preseal_proof,
            created_at_utc=created,
        )
        write_json_atomic(args.contract_json.resolve(), contract)
        contract_sha = sidecar(args.contract_json.resolve())
        predictions = build_predictions(
            contract=contract,
            contract_sha256=contract_sha,
            selection=selection,
            selection_path=args.selected_coefficients.resolve(),
            g4c=g4c,
            g4c_path=args.g4c_readout.resolve(),
            gatesim=gatesim,
            gatesim_path=args.gatesim.resolve(),
            preseal=preseal,
            preseal_path=args.preseal_proof.resolve(),
            created_at_utc=created,
        )
        write_json_atomic(args.predictions.resolve(), predictions)
        prediction_sha = sidecar(args.predictions.resolve())
        args.contract_md.resolve().parent.mkdir(parents=True, exist_ok=True)
        temporary = args.contract_md.resolve().with_name(
            f".{args.contract_md.name}.tmp-{os.getpid()}"
        )
        temporary.write_text(markdown(contract, predictions, prediction_sha))
        temporary.replace(args.contract_md.resolve())
        md_sha = sidecar(args.contract_md.resolve())
    except (V9Error, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "contract": str(args.contract_json.resolve()),
                "contract_sha256": contract_sha,
                "registration_md": str(args.contract_md.resolve()),
                "registration_md_sha256": md_sha,
                "predictions": str(args.predictions.resolve()),
                "predictions_sha256": prediction_sha,
                "mechanism": "SURFACE-FALLBACK",
                "stage_1p7b_cells": predictions["stage_1p7b"]["cell_count"],
                "stage_7b_cells": predictions["stage_7b"]["cell_count"],
                "verification_loss_seen": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
