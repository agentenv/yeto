#!/usr/bin/env python3
"""Freeze the v6 training-only surface selection for downstream prediction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v9_common import (  # noqa: E402
    V9Error,
    canonical_sha256,
    read_json,
    sha256_file,
    utc_now,
    validate_surface,
    write_json_atomic,
)


OUTPUT_SCHEMA = "yeto_outer_mup_v6_selected_surfaces_v1"
EXPECTED_CELLS = 540


def freeze_selection(
    *,
    readout: dict,
    readout_path: Path,
    manifest: dict,
    manifest_path: Path,
    contract: dict,
    contract_path: Path,
    mechfit: dict | None = None,
    mechfit_path: Path | None = None,
    selected_at_utc: str,
) -> dict:
    if readout.get("schema") != "yeto_outer_mup_v6_g6_readout_v1":
        raise V9Error("not a v6 G6 readout")
    if manifest.get("schema") != "yeto_outer_mup_v6_launch_manifest_v1":
        raise V9Error("not a v6 launch manifest")
    if contract.get("schema") != "yeto_outer_mup_v6_factorial_prereg_v1":
        raise V9Error("not a v6 preregistration")
    if len(manifest.get("cells", [])) != EXPECTED_CELLS:
        raise V9Error("selection requires the complete 540-cell v6 manifest")
    if readout.get("manifest_sha256") != sha256_file(manifest_path):
        raise V9Error("v6 readout binds another launch manifest")
    if readout.get("observed_completed_cells") != EXPECTED_CELLS:
        raise V9Error("v6 readout does not contain 540 completed cells")
    if readout.get("evidence_errors") or readout.get("curve_errors"):
        raise V9Error("v6 readout contains evidence or curve errors")
    gate = readout.get("gate", {})
    if gate.get("verdict") != "PASS" or gate.get("evaluable") is not True:
        raise V9Error("the preregistered G6 gate must be evaluable PASS")
    if readout.get("bootstrap", {}).get("status") != "VALID":
        raise V9Error("v6 joint bootstrap is not valid")

    arm_results = readout.get("surface_results")
    if not isinstance(arm_results, dict):
        raise V9Error("v6 readout lacks surface results")
    mechfit_surfaces = None
    mechfit_provenance = None
    if mechfit is not None:
        if mechfit_path is None:
            raise V9Error("mechfit path is required with the mechfit artifact")
        inventory = mechfit.get("partial_inventory", {})
        if (
            inventory.get("expected_manifest_cells") != EXPECTED_CELLS
            or inventory.get("loaded_validated_cells") != EXPECTED_CELLS
            or inventory.get("completion_fraction") != 1.0
            or inventory.get("loader_audit", {}).get("unique_loss_keys")
            != EXPECTED_CELLS
        ):
            raise V9Error("mechfit artifact is not the complete 540-cell fit")
        if mechfit.get("D_summary", {}).get("complete_D") != 24:
            raise V9Error("mechfit artifact does not contain all 24 D estimates")
        mechfit_surfaces = mechfit.get("surface_selection")
        if not isinstance(mechfit_surfaces, dict):
            raise V9Error("mechfit artifact lacks surface_selection")
        mechfit_provenance = {
            "path": str(mechfit_path.resolve()),
            "sha256": sha256_file(mechfit_path),
            "created_at_utc": mechfit.get("created_at_utc"),
            "loaded_validated_cells": EXPECTED_CELLS,
            "complete_D": 24,
            "pipeline_scope_disclosure": (
                "the mechanism pre-fit wrapper labels every artifact NON-FINAL; "
                "only its nested all-eight empirical surface refits are consumed, "
                "and their bytes must exactly equal the frozen G6 refits"
            ),
        }
    selected_surfaces = {}
    selection_diagnostics = {}
    for arm in ("raw", "corrected"):
        result = arm_results.get(arm)
        if not isinstance(result, dict) or result.get("pass") is not True:
            raise V9Error(f"v6 {arm} held-out surface gate did not pass")
        surface = result.get("surface")
        if not isinstance(surface, dict):
            raise V9Error(f"v6 {arm} selection lacks a surface")
        validate_surface(surface)
        model_selection = surface.get("model_selection", {})
        if model_selection.get(
            "selection_uses_heldout_outcomes"
        ) is not False or model_selection.get("selected_family") != surface.get(
            "family_id"
        ):
            raise V9Error(f"v6 {arm} surface is not the training-only selection")
        # Copy only training-derived fields.  Held-out outcomes and errors remain
        # in the immutable G6 readout and are not inputs to the v9 point formula.
        selected_surface = {
            key: surface[key]
            for key in (
                "family_id",
                "family",
                "coordinates",
                "coefficient_order",
                "coefficients",
                "training_cells",
                "training_fit",
                "model_selection",
            )
        }
        if mechfit_surfaces is not None:
            fit_record = mechfit_surfaces.get(arm)
            if not isinstance(fit_record, dict):
                raise V9Error(f"mechfit artifact lacks the {arm} surface")
            fitted = fit_record.get("surface")
            if (
                fit_record.get("all_selected_training_D_sources_complete") is not True
                or fit_record.get("status") != "REGISTERED_COMPLETE"
                or not isinstance(fitted, dict)
                or fitted.get("status") != "REGISTERED_COMPLETE"
                or fitted.get("non_final") is not False
                or fitted.get("available_training_cell_count") != 8
            ):
                raise V9Error(f"mechfit {arm} surface is not a complete all-eight refit")
            validate_surface(fitted)
            if fitted.get("family_id") != surface.get("family_id"):
                raise V9Error(
                    f"mechfit {arm} family differs from the G6-selected family"
                )
            for key in (
                "family",
                "coordinates",
                "coefficient_order",
                "coefficients",
                "training_cells",
                "training_fit",
            ):
                if fitted.get(key) != surface.get(key):
                    raise V9Error(
                        f"mechfit {arm} full-data {key} differs from the G6 refit"
                    )
            # The family choice is authorized by G6; the literal coefficients
            # are copied from the independently materialized full-data mechfit.
            selected_surface["coefficients"] = list(fitted["coefficients"])
            selected_surface["coefficient_source"] = (
                "mechfit full-data all-eight refit constrained to G6-selected family"
            )
        selected_surfaces[arm] = selected_surface
        selection_diagnostics[arm] = {
            "selected_family": surface["family_id"],
            "candidate_loo_rmse_bits": {
                record["family_id"]: record["loo_rmse_bits"]
                for record in model_selection["candidate_scores"]
            },
        }

    mu0_curves = []
    for curve in readout.get("curve_fits", []):
        if curve.get("arm") != "mu0":
            continue
        if curve.get("interior") is not True or curve.get("eta_star") is None:
            raise V9Error("one v6 mu0 anchor curve is not strictly interior")
        mu0_curves.append(
            {
                key: curve[key]
                for key in (
                    "t",
                    "s",
                    "h",
                    "arm",
                    "eta_star",
                    "vertex_log2_eta",
                    "a",
                    "b",
                    "c",
                    "etas",
                    "seed_mean_losses",
                    "interior",
                    "status",
                )
            }
        )
        mu0_curves[-1]["accepted"] = True
    mu0_curves.sort(key=lambda item: (item["t"], item["s"]))
    if len(mu0_curves) != 12:
        raise V9Error(f"expected 12 v6 mu0 curves, found {len(mu0_curves)}")

    result = {
        "schema": OUTPUT_SCHEMA,
        "status": "FROZEN",
        "selected_at_utc": selected_at_utc,
        "selection_step": (
            "training-only F1/F2/F3 arm-specific LOO family selection is frozen "
            "by G6; coefficients are copied from mechfit's complete 540-cell "
            "all-eight refits after exact equality with the G6 refits"
        ),
        "selection_uses_heldout_outcomes": False,
        "g6_gate_required_and_observed": "PASS",
        "source_git_commit": readout.get("source_git_commit"),
        "v6_readout_path": str(readout_path.resolve()),
        "v6_readout_sha256": sha256_file(readout_path),
        "v6_manifest_path": str(manifest_path.resolve()),
        "v6_manifest_sha256": sha256_file(manifest_path),
        "v6_contract_path": str(contract_path.resolve()),
        "v6_contract_sha256": sha256_file(contract_path),
        "v6_analyzer_path": contract["frozen_analyzer"]["path"],
        "v6_analyzer_sha256": contract["frozen_analyzer"]["sha256"],
        "selected_surfaces": selected_surfaces,
        "selection_diagnostics": selection_diagnostics,
        "mechfit_full_data": mechfit_provenance,
        "mu0_curve_fits": mu0_curves,
    }
    result["selection_preimage_canonical_sha256"] = canonical_sha256(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--mechfit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-at-utc")
    args = parser.parse_args()
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.output.exists() or sidecar.exists():
        raise SystemExit(f"refusing existing frozen selection: {args.output}")
    try:
        result = freeze_selection(
            readout=read_json(args.readout),
            readout_path=args.readout,
            manifest=read_json(args.manifest),
            manifest_path=args.manifest,
            contract=read_json(args.contract),
            contract_path=args.contract,
            mechfit=read_json(args.mechfit),
            mechfit_path=args.mechfit,
            selected_at_utc=args.selected_at_utc or utc_now(),
        )
    except (V9Error, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    output = args.output.resolve()
    write_json_atomic(output, result)
    digest = sha256_file(output)
    sidecar.write_text(f"{digest}  {output.name}\n")
    print(
        json.dumps(
            {
                "selection": str(output),
                "sha256": digest,
                "families": {
                    arm: value["family_id"]
                    for arm, value in result["selected_surfaces"].items()
                },
                "selection_uses_heldout_outcomes": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
