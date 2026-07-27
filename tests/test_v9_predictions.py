import hashlib
import importlib.util
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMMON_PATH = ROOT / "scripts/v9_common.py"
BUILDER_PATH = ROOT / "scripts/build_v9_predictions.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def surface(family_id, coefficients):
    common = load(COMMON_PATH, f"common_{family_id}")
    return {
        "family_id": family_id,
        "coordinates": {"u": "(T-5)/5", "v": "log2(S/5120)"},
        "coefficient_order": list(common.SURFACE_COEFFICIENT_ORDERS[family_id]),
        "coefficients": list(coefficients),
    }


def test_v6_surface_features_and_prediction_match_registered_coordinates():
    common = load(COMMON_PATH, "v9_common_surface")
    assert common.surface_features(5, 5120, "F1") == [1.0, 0.0, 0.0]
    assert common.surface_features(10, 10240, "F2") == [1.0, 1.0, 1.0, 1.0]
    assert common.surface_features(10, 5120, "F3") == [1.0, 1.0, 0.0, 1.0]
    selected = surface("F2", (0.25, 0.5, -0.125, 0.0625))
    assert math.isclose(common.predict_log2_d(selected, 10, 10240), 0.6875)


def test_log_linear_interpolation_is_geometric_at_log_midpoint():
    common = load(COMMON_PATH, "v9_common_interp")
    prediction, slope = common.log_linear_interpolate(1.0, 8.0, 4.0, 2.0, 2.0)
    assert math.isclose(prediction, 4.0)
    assert math.isclose(slope, -1.0)


def test_near_bracket_retains_unconstrained_vertex_without_clipping():
    common = load(COMMON_PATH, "v9_common_near")
    center = 0.01
    offsets = (-0.5, 0.0, 0.5)
    etas = common.exact_offset_grid(center, offsets)
    true_vertex = math.log2(center) + 0.75
    losses = [2.0 + 0.2 * (math.log2(eta) - true_vertex) ** 2 for eta in etas]
    fit = common.fit_quadratic(etas, losses, near_bracket_allowance_bits=0.5)
    assert fit["strict_interior"] is False
    assert fit["near_bracketed"] is True
    assert fit["status"] == "NEAR_BRACKETED"
    assert math.isclose(fit["eta_star"], 2.0**true_vertex, rel_tol=1e-9)
    assert fit["eta_star"] > max(etas)


def test_prediction_builder_freezes_16_plus_12_cells(tmp_path):
    builder = load(BUILDER_PATH, "build_v9_predictions_test")
    raw_surface = surface("F1", (math.log2(1.5), 0.0, 0.0))
    corrected_surface = surface("F3", (math.log2(0.8), 0.0, 0.0, 0.0))
    selection = {
        "schema": builder.SELECTION_SCHEMA,
        "v6_readout_sha256": "a" * 64,
        "selected_surfaces": {
            "raw": raw_surface,
            "corrected": corrected_surface,
        },
        "mu0_curve_fits": [
            {
                "t": 5,
                "s": 2560,
                "arm": "mu0",
                "accepted": True,
                "eta_star": 0.04,
            }
        ],
    }
    g4c = {
        "curve_fits": [
            {
                "t": 5,
                "s": 2560,
                "mu": 0.0,
                "accepted": True,
                "eta_star": 0.02,
            },
            {
                "t": 20,
                "s": 10240,
                "mu": 0.0,
                "accepted": True,
                "eta_star": 0.005,
            },
        ]
    }
    gatesim = {
        "schema": "yeto_outer_mup_v9_gate_simulation_v1",
        "status": "PASS",
        "gates": {
            "G9A_1P7B": {
                "registered_absolute_error_band_bits": {
                    "raw": 0.5,
                    "corrected": 0.625,
                }
            },
            "G9B_7B": {
                "registered_absolute_error_band_bits": {
                    "mu0": 0.5,
                    "raw": 0.75,
                }
            },
        },
    }
    selection_path = tmp_path / "selection.json"
    g4c_path = tmp_path / "g4c.json"
    gatesim_path = tmp_path / "gatesim.json"
    preseal_path = tmp_path / "preseal.json"
    selection_path.write_text(json.dumps(selection))
    g4c_path.write_text(json.dumps(g4c))
    gatesim_path.write_text(json.dumps(gatesim))
    preseal = {
        "schema": "yeto_outer_mup_v9_preseal_proof_v1",
        "status": "PASS",
        "verification_loss_seen": False,
        "result_root_absent_on_both_nodes": True,
    }
    preseal_path.write_text(json.dumps(preseal))
    contract = {
        "schema": builder.CONTRACT_SCHEMA,
        "source_artifacts": {
            "v6_selection": {
                "sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest()
            },
            "g4c_readout": {
                "sha256": hashlib.sha256(g4c_path.read_bytes()).hexdigest()
            },
            "gate_simulation": {
                "sha256": hashlib.sha256(gatesim_path.read_bytes()).hexdigest()
            },
            "preseal_proof": {
                "sha256": hashlib.sha256(preseal_path.read_bytes()).hexdigest()
            },
        },
        "referee_mechanism_resolution": {
            "deadline": "2026-07-27T20:00:00-07:00",
            "confirmed_mechanism": None,
            "registered_fallback": "selected empirical surface",
        },
        "models": {
            "smollm2_135m": {"exact_parameters": 100},
            "smollm2_1p7b": {"exact_parameters": 1600},
            "qwen2p5_7b": {"exact_parameters": 6400},
        },
        "design": {
            "stage_1p7b": {
                "coordinate": {"T": 10, "S": 5120, "H": 512},
                "ladder_offsets_log2": [-0.75, -0.25, 0.25, 0.75],
            },
            "stage_7b": {
                "coordinate": {"T": 5, "S": 2560, "H": 512},
                "ladder_offsets_log2": [-0.5, 0.0, 0.5],
            },
        },
    }
    result = builder.build_predictions(
        contract=contract,
        contract_sha256="b" * 64,
        selection=selection,
        selection_path=selection_path,
        g4c=g4c,
        g4c_path=g4c_path,
        gatesim=gatesim,
        gatesim_path=gatesim_path,
        preseal=preseal,
        preseal_path=preseal_path,
        created_at_utc="2026-07-28T03:00:00+00:00",
    )
    assert result["verification_loss_seen"] is False
    assert result["stage_1p7b"]["cell_count"] == 16
    assert result["stage_7b"]["cell_count"] == 12
    eta0_17 = result["stage_1p7b"]["eta0_anchor_prediction"]["eta_star_mu0_T10_S5120"]
    assert math.isclose(eta0_17, 0.01)
    assert math.isclose(
        result["stage_1p7b"]["targets"]["raw"]["predicted_eta_star"],
        0.0015,
    )
    assert math.isclose(
        result["stage_1p7b"]["targets"]["corrected"]["predicted_eta_star"],
        0.0008,
    )
    # eta scales as P^-1/4 for the synthetic 100->1600 anchors.
    assert math.isclose(
        result["stage_7b"]["targets"]["mu0"]["predicted_eta_star"],
        0.02 * 4.0**-0.25,
    )
    assert len(result["stage_1p7b"]["targets"]["raw"]["verification_etas"]) == 4
    assert len(result["stage_7b"]["targets"]["raw"]["verification_etas"]) == 3
