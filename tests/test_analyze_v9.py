import importlib.util
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/analyze_v9.py"


def load():
    spec = importlib.util.spec_from_file_location("analyze_v9", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def fixture(module, shift=0.0):
    predictions = {
        "stage_1p7b": {
            "targets": {
                "raw": {
                    "predicted_eta_star": 0.01,
                    "verification_etas": [
                        0.01 * 2**x for x in (-0.75, -0.25, 0.25, 0.75)
                    ],
                },
                "corrected": {
                    "predicted_eta_star": 0.005,
                    "verification_etas": [
                        0.005 * 2**x for x in (-0.75, -0.25, 0.25, 0.75)
                    ],
                },
            }
        },
        "stage_7b": {
            "targets": {
                "mu0": {
                    "predicted_eta_star": 0.004,
                    "verification_etas": [0.004 * 2**x for x in (-0.5, 0, 0.5)],
                },
                "raw": {
                    "predicted_eta_star": 0.0007,
                    "verification_etas": [0.0007 * 2**x for x in (-0.5, 0, 0.5)],
                },
            }
        },
    }
    # analyze_v9 intentionally consumes the gatesim groups from the manifest.
    groups = [
        {"representative": [0, 0], "frequency": 2500},
        {"representative": [0, 1], "frequency": 5000},
        {"representative": [1, 1], "frequency": 2500},
    ]
    manifest = {
        "analysis_contract": {
            "bootstrap": {"groups": groups},
            "gates": {
                "G9A_1P7B": {
                    "near_bracket_allowance_bits": 0.5,
                    "minimum_valid_bootstrap_refits": 7500,
                    "absolute_error_band_bits": {"raw": 0.5, "corrected": 0.5},
                },
                "G9B_7B": {
                    "near_bracket_allowance_bits": 0.5,
                    "minimum_valid_bootstrap_refits": 7500,
                    "absolute_error_band_bits": {"mu0": 0.5, "raw": 0.5},
                },
            },
        }
    }
    losses = {}
    for stage, arms in module.STAGE_TARGETS.items():
        for arm in arms:
            target = predictions[stage]["targets"][arm]
            center = target["predicted_eta_star"] * 2**shift
            for eta in target["verification_etas"]:
                mean = 2.0 + 0.2 * math.log2(eta / center) ** 2
                for seed, noise in ((901, -0.0001), (907, 0.0001)):
                    losses[(stage, arm, seed, eta)] = mean + noise
    return losses, predictions, manifest


def test_exact_centered_predictions_pass_both_gates():
    module = load()
    losses, predictions, manifest = fixture(module)
    result = module.analyze_losses(
        losses=losses,
        predictions=predictions,
        manifest=manifest,
        stage_complete={"stage_1p7b": True, "stage_7b": True},
    )
    assert result["verdict"] == "PASS"
    assert all(gate["verdict"] == "PASS" for gate in result["gates"].values())
    for gate in result["gates"].values():
        assert gate["bootstrap"]["valid_replicates"] == 10_000
    assert result["gates"]["G9A_1P7B"]["curve_fits"]["raw"]["seeds"] == [
        901,
        907,
    ]
    assert result["gates"]["G9B_7B"]["curve_fits"]["raw"]["seeds"] == [907]


def test_reduced_7b_gate_does_not_require_removed_seed_901():
    module = load()
    losses, predictions, manifest = fixture(module)
    losses = {
        key: value
        for key, value in losses.items()
        if key[0] != "stage_7b" or key[2] == 907
    }
    result = module.analyze_losses(
        losses=losses,
        predictions=predictions,
        manifest=manifest,
        stage_complete={"stage_1p7b": True, "stage_7b": True},
    )
    assert result["verdict"] == "PASS"
    assert result["gates"]["G9B_7B"]["bootstrap"]["valid_replicates"] == 10_000


def test_one_bit_transport_miss_fails_when_evaluable():
    module = load()
    losses, predictions, manifest = fixture(module, shift=1.0)
    result = module.analyze_losses(
        losses=losses,
        predictions=predictions,
        manifest=manifest,
        stage_complete={"stage_1p7b": True, "stage_7b": True},
    )
    assert result["verdict"] == "FAIL"
    assert all(gate["evaluable"] for gate in result["gates"].values())
    assert all(gate["verdict"] == "FAIL" for gate in result["gates"].values())


def test_missing_stage_is_not_evaluable():
    module = load()
    losses, predictions, manifest = fixture(module)
    result = module.analyze_losses(
        losses=losses,
        predictions=predictions,
        manifest=manifest,
        stage_complete={"stage_1p7b": True, "stage_7b": False},
    )
    assert result["verdict"] == "NOT_EVALUABLE"
    assert result["gates"]["G9A_1P7B"]["verdict"] == "PASS"
    assert result["gates"]["G9B_7B"]["verdict"] == "NOT_EVALUABLE"
