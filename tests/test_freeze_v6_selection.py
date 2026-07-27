import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/freeze_v6_selection.py"


def load():
    spec = importlib.util.spec_from_file_location("freeze_v6_selection", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def selected_surface(family_id="F1"):
    orders = {
        "F1": ["gamma", "alpha_T", "beta_log2_S"],
        "F2": ["gamma", "alpha_T", "beta_log2_S", "delta_T_x_log2_S"],
        "F3": ["gamma", "alpha_T", "beta_log2_S", "epsilon_T_squared"],
    }
    return {
        "family_id": family_id,
        "family": family_id,
        "coordinates": {"u": "(T-5)/5", "v": "log2(S/5120)"},
        "coefficient_order": orders[family_id],
        "coefficients": [0.1] * len(orders[family_id]),
        "training_cells": [[2, 2560]],
        "training_fit": {"T2_S2560": {}},
        "model_selection": {
            "selection_uses_heldout_outcomes": False,
            "selected_family": family_id,
            "candidate_scores": [
                {"family_id": name, "loo_rmse_bits": index + 0.1}
                for index, name in enumerate(("F1", "F2", "F3"))
            ],
        },
    }


def materialize(tmp_path, *, gate="PASS", selection_uses_heldout=False):
    module = load()
    manifest = {
        "schema": "yeto_outer_mup_v6_launch_manifest_v1",
        "cells": [{}] * 540,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    raw = selected_surface("F1")
    corrected = selected_surface("F2")
    raw["model_selection"]["selection_uses_heldout_outcomes"] = selection_uses_heldout
    curves = []
    for t in (2, 5, 10, 20):
        for s in (2560, 5120, 10240):
            curves.append(
                {
                    "t": t,
                    "s": s,
                    "h": s // t,
                    "arm": "mu0",
                    "eta_star": 0.01,
                    "vertex_log2_eta": -6.0,
                    "a": 0.1,
                    "b": 1.0,
                    "c": 2.0,
                    "etas": [0.005, 0.01, 0.02],
                    "seed_mean_losses": [2.1, 2.0, 2.1],
                    "interior": True,
                    "status": "INTERIOR",
                }
            )
    readout = {
        "schema": "yeto_outer_mup_v6_g6_readout_v1",
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "observed_completed_cells": 540,
        "evidence_errors": [],
        "curve_errors": [],
        "gate": {"verdict": gate, "evaluable": True},
        "bootstrap": {"status": "VALID"},
        "surface_results": {
            "raw": {"pass": True, "surface": raw, "heldout_predictions": {"secret": 1}},
            "corrected": {
                "pass": True,
                "surface": corrected,
                "heldout_predictions": {"secret": 2},
            },
        },
        "curve_fits": curves,
        "source_git_commit": "c" * 40,
    }
    readout_path = tmp_path / "readout.json"
    readout_path.write_text(json.dumps(readout))
    contract = {
        "schema": "yeto_outer_mup_v6_factorial_prereg_v1",
        "frozen_analyzer": {"path": "scripts/analyze_v6.py", "sha256": "d" * 64},
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract))
    return (
        module,
        readout,
        readout_path,
        manifest,
        manifest_path,
        contract,
        contract_path,
    )


def test_freeze_selection_excludes_heldout_records(tmp_path):
    module, readout, readout_path, manifest, manifest_path, contract, contract_path = (
        materialize(tmp_path)
    )
    result = module.freeze_selection(
        readout=readout,
        readout_path=readout_path,
        manifest=manifest,
        manifest_path=manifest_path,
        contract=contract,
        contract_path=contract_path,
        selected_at_utc="2026-07-28T03:00:00Z",
    )
    assert result["selection_uses_heldout_outcomes"] is False
    assert result["selected_surfaces"]["raw"]["family_id"] == "F1"
    assert result["selected_surfaces"]["corrected"]["family_id"] == "F2"
    assert len(result["mu0_curve_fits"]) == 12
    assert "heldout_predictions" not in json.dumps(result)


def test_freeze_selection_requires_g6_pass(tmp_path):
    module, readout, readout_path, manifest, manifest_path, contract, contract_path = (
        materialize(tmp_path, gate="FAIL")
    )
    try:
        module.freeze_selection(
            readout=readout,
            readout_path=readout_path,
            manifest=manifest,
            manifest_path=manifest_path,
            contract=contract,
            contract_path=contract_path,
            selected_at_utc="2026-07-28T03:00:00Z",
        )
    except module.V9Error as exc:
        assert "G6 gate" in str(exc)
    else:
        raise AssertionError("G6 FAIL unexpectedly produced a frozen selection")


def test_freeze_selection_rejects_heldout_dependent_selection(tmp_path):
    module, readout, readout_path, manifest, manifest_path, contract, contract_path = (
        materialize(tmp_path, selection_uses_heldout=True)
    )
    try:
        module.freeze_selection(
            readout=readout,
            readout_path=readout_path,
            manifest=manifest,
            manifest_path=manifest_path,
            contract=contract,
            contract_path=contract_path,
            selected_at_utc="2026-07-28T03:00:00Z",
        )
    except module.V9Error as exc:
        assert "training-only" in str(exc)
    else:
        raise AssertionError("held-out-dependent selection was accepted")
