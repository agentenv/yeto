#!/usr/bin/env python3
"""Build the prospective v9 prediction seal from frozen upstream evidence."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v9_common import (  # noqa: E402
    V9Error,
    canonical_sha256,
    exact_offset_grid,
    log_linear_interpolate,
    predict_log2_d,
    read_json,
    sha256_file,
    utc_now,
    validate_surface,
    write_json_atomic,
)


SELECTION_SCHEMA = "yeto_outer_mup_v6_selected_surfaces_v1"
CONTRACT_SCHEMA = "yeto_outer_mup_v9_sealed_scale_prereg_v1"
PREDICTION_SCHEMA = "yeto_outer_mup_v9_sealed_predictions_v1"
MU_HIGH = 0.9
ONE_MINUS_MU = 1.0 - MU_HIGH


def find_curve(
    curves: list[dict],
    *,
    t: int,
    s: int,
    arm: str | None = None,
    mu: float | None = None,
) -> dict:
    matches = []
    for curve in curves:
        if int(curve.get("t", -1)) != t or int(curve.get("s", -1)) != s:
            continue
        if arm is not None and curve.get("arm") != arm:
            continue
        if mu is not None and not math.isclose(float(curve.get("mu", math.nan)), mu):
            continue
        matches.append(curve)
    if len(matches) != 1:
        label = f"T{t}/S{s}/arm={arm}/mu={mu}"
        raise V9Error(f"expected one upstream curve for {label}, found {len(matches)}")
    curve = matches[0]
    eta_star = curve.get("eta_star")
    if (
        curve.get("accepted", curve.get("interior")) is not True
        or not isinstance(eta_star, (int, float))
        or not math.isfinite(eta_star)
        or eta_star <= 0
    ):
        raise V9Error(f"upstream curve is not an accepted finite optimum: {curve}")
    return curve


def verify_bound_file(contract: dict, field: str, path: Path) -> None:
    record = contract.get("source_artifacts", {}).get(field)
    if not isinstance(record, dict):
        raise V9Error(f"contract lacks source_artifacts.{field}")
    if record.get("sha256") != sha256_file(path):
        raise V9Error(f"{field} raw hash differs from the preregistration")


def build_predictions(
    *,
    contract: dict,
    contract_sha256: str,
    selection: dict,
    selection_path: Path,
    g4c: dict,
    g4c_path: Path,
    gatesim: dict,
    gatesim_path: Path,
    preseal: dict,
    preseal_path: Path,
    created_at_utc: str,
) -> dict:
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise V9Error("not the v9 sealed-scale contract")
    if selection.get("schema") != SELECTION_SCHEMA:
        raise V9Error("not the frozen v6 surface selection")
    if gatesim.get("schema") != "yeto_outer_mup_v9_gate_simulation_v1":
        raise V9Error("not the frozen v9 gate simulation")
    if gatesim.get("status") != "PASS":
        raise V9Error("both v9 gate simulations must pass before prediction sealing")
    if (
        preseal.get("schema") != "yeto_outer_mup_v9_preseal_proof_v1"
        or preseal.get("status") != "PASS"
        or preseal.get("verification_loss_seen") is not False
        or preseal.get("result_root_absent_on_both_nodes") is not True
    ):
        raise V9Error("v9 preseal proof does not establish zero verification exposure")
    verify_bound_file(contract, "v6_selection", selection_path)
    verify_bound_file(contract, "g4c_readout", g4c_path)
    verify_bound_file(contract, "gate_simulation", gatesim_path)
    verify_bound_file(contract, "preseal_proof", preseal_path)

    surfaces = selection.get("selected_surfaces")
    if not isinstance(surfaces, dict) or set(surfaces) != {"raw", "corrected"}:
        raise V9Error("selection must contain exactly raw and corrected surfaces")
    for surface in surfaces.values():
        validate_surface(surface)

    target_17 = contract["design"]["stage_1p7b"]["coordinate"]
    if target_17 != {"T": 10, "S": 5120, "H": 512}:
        raise V9Error("v9 1.7B coordinate differs from T10/S5120/H512")
    target_7 = contract["design"]["stage_7b"]["coordinate"]
    if target_7 != {"T": 5, "S": 2560, "H": 512}:
        raise V9Error("v9 7B coordinate differs from T5/S2560/H512")

    g4c_curves = g4c.get("curve_fits", [])
    anchor_17_t5 = find_curve(g4c_curves, t=5, s=2560, mu=0.0)
    anchor_17_t20 = find_curve(g4c_curves, t=20, s=10240, mu=0.0)
    eta0_17_t10, anchor_slope = log_linear_interpolate(
        2560.0,
        float(anchor_17_t5["eta_star"]),
        10240.0,
        float(anchor_17_t20["eta_star"]),
        5120.0,
    )

    eta_17 = {}
    d_17 = {}
    for arm in ("raw", "corrected"):
        log2_d = predict_log2_d(surfaces[arm], 10, 5120)
        d_value = 2.0**log2_d
        d_17[arm] = {"log2_D": log2_d, "D": d_value}
        eta_17[arm] = ONE_MINUS_MU * d_value * eta0_17_t10

    selection_mu0_curves = selection.get("mu0_curve_fits", [])
    eta0_135_t5 = float(
        find_curve(selection_mu0_curves, t=5, s=2560, arm="mu0")["eta_star"]
    )
    eta0_17_t5 = float(anchor_17_t5["eta_star"])
    model = contract["models"]
    p135 = int(model["smollm2_135m"]["exact_parameters"])
    p17 = int(model["smollm2_1p7b"]["exact_parameters"])
    p7 = int(model["qwen2p5_7b"]["exact_parameters"])
    eta0_7, scale_exponent = log_linear_interpolate(
        float(p135), eta0_135_t5, float(p17), eta0_17_t5, float(p7)
    )
    log2_d_7_raw = predict_log2_d(surfaces["raw"], 5, 2560)
    d_7_raw = 2.0**log2_d_7_raw
    eta_7_raw = ONE_MINUS_MU * d_7_raw * eta0_7

    offsets_17 = contract["design"]["stage_1p7b"]["ladder_offsets_log2"]
    offsets_7 = contract["design"]["stage_7b"]["ladder_offsets_log2"]
    predictions = {
        "schema": PREDICTION_SCHEMA,
        "status": "SEALED",
        "created_at_utc": created_at_utc,
        "verification_loss_seen": False,
        "source_artifacts": {
            "contract_sha256": contract_sha256,
            "v6_selection_sha256": sha256_file(selection_path),
            "v6_readout_sha256": selection["v6_readout_sha256"],
            "g4c_readout_sha256": sha256_file(g4c_path),
            "gate_simulation_sha256": sha256_file(gatesim_path),
            "preseal_proof_sha256": sha256_file(preseal_path),
        },
        "surface_fallback_disclosure": contract["referee_mechanism_resolution"],
        "shared": {
            "mu": MU_HIGH,
            "one_minus_mu": ONE_MINUS_MU,
            "seeds": [901, 907],
            "eta_ladders_are_fixed_before_verification": True,
        },
        "stage_1p7b": {
            "model": model["smollm2_1p7b"],
            "coordinate": target_17,
            "eta0_anchor_prediction": {
                "rule": (
                    "power-law interpolation in S along the two constant-H=512 "
                    "G4C mu0 anchors; T=S/H therefore maps to T=10 at S=5120"
                ),
                "anchor_T5_S2560": float(anchor_17_t5["eta_star"]),
                "anchor_T20_S10240": float(anchor_17_t20["eta_star"]),
                "log_eta_vs_log_S_slope": anchor_slope,
                "eta_star_mu0_T10_S5120": eta0_17_t10,
            },
            "targets": {
                arm: {
                    "arm": arm,
                    "mu": MU_HIGH,
                    "outer_bias_correction": arm == "corrected",
                    "selected_surface": surfaces[arm],
                    "surface_prediction": d_17[arm],
                    "formula": "eta_mu=eta_mu0*(1-mu)*D_surface(T,S)",
                    "predicted_eta_star": eta_17[arm],
                    "ladder_offsets_log2": offsets_17,
                    "verification_etas": exact_offset_grid(eta_17[arm], offsets_17),
                    "registered_absolute_error_band_bits": gatesim["gates"]["G9A_1P7B"][
                        "registered_absolute_error_band_bits"
                    ][arm],
                }
                for arm in ("raw", "corrected")
            },
            "cell_count": 16,
        },
        "stage_7b": {
            "model": model["qwen2p5_7b"],
            "coordinate": target_7,
            "eta0_scale_prediction": {
                "rule": (
                    "two-anchor power law in exact parameter count at fixed "
                    "T5/S2560/H512, using the v6 135M and G4C 1.7B mu0 optima"
                ),
                "parameters_135m": p135,
                "eta_star_mu0_135m": eta0_135_t5,
                "parameters_1p7b": p17,
                "eta_star_mu0_1p7b": eta0_17_t5,
                "parameters_7b": p7,
                "log_eta_vs_log_parameters_exponent": scale_exponent,
                "predicted_eta_star": eta0_7,
            },
            "targets": {
                "mu0": {
                    "arm": "mu0",
                    "mu": 0.0,
                    "outer_bias_correction": False,
                    "predicted_eta_star": eta0_7,
                    "ladder_offsets_log2": offsets_7,
                    "verification_etas": exact_offset_grid(eta0_7, offsets_7),
                    "registered_absolute_error_band_bits": gatesim["gates"]["G9B_7B"][
                        "registered_absolute_error_band_bits"
                    ]["mu0"],
                },
                "raw": {
                    "arm": "raw",
                    "mu": MU_HIGH,
                    "outer_bias_correction": False,
                    "selected_surface": surfaces["raw"],
                    "surface_prediction": {
                        "log2_D": log2_d_7_raw,
                        "D": d_7_raw,
                    },
                    "formula": "eta_mu=eta_mu0*(1-mu)*D_surface(T,S)",
                    "predicted_eta_star": eta_7_raw,
                    "ladder_offsets_log2": offsets_7,
                    "verification_etas": exact_offset_grid(eta_7_raw, offsets_7),
                    "registered_absolute_error_band_bits": gatesim["gates"]["G9B_7B"][
                        "registered_absolute_error_band_bits"
                    ]["raw"],
                },
            },
            "cell_count": 12,
        },
    }
    predictions["prediction_preimage_canonical_sha256"] = canonical_sha256(predictions)
    return predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--g4c-readout", type=Path, required=True)
    parser.add_argument("--gatesim", type=Path, required=True)
    parser.add_argument("--preseal-proof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--created-at-utc")
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.output.with_suffix(args.output.suffix + ".sha256").exists()
    ):
        raise SystemExit(f"refusing existing prediction seal: {args.output}")
    try:
        contract = read_json(args.contract)
        selection = read_json(args.selection)
        g4c = read_json(args.g4c_readout)
        gatesim = read_json(args.gatesim)
        preseal = read_json(args.preseal_proof)
        predictions = build_predictions(
            contract=contract,
            contract_sha256=sha256_file(args.contract.resolve()),
            selection=selection,
            selection_path=args.selection.resolve(),
            g4c=g4c,
            g4c_path=args.g4c_readout.resolve(),
            gatesim=gatesim,
            gatesim_path=args.gatesim.resolve(),
            preseal=preseal,
            preseal_path=args.preseal_proof.resolve(),
            created_at_utc=args.created_at_utc or utc_now(),
        )
    except (V9Error, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    output = args.output.resolve()
    write_json_atomic(output, predictions)
    digest = sha256_file(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n"
    )
    print(
        json.dumps(
            {
                "prediction_path": str(output),
                "prediction_sha256": digest,
                "prediction_preimage_canonical_sha256": predictions[
                    "prediction_preimage_canonical_sha256"
                ],
                "cells": 28,
                "verification_loss_seen": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
