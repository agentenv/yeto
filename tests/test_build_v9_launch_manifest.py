import importlib.util
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_v9_launch_manifest.py"


def load():
    spec = importlib.util.spec_from_file_location("build_v9_launch_manifest", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def fixtures():
    contract = {
        "models": {
            "smollm2_1p7b": {"path": "/models/1p7b"},
            "qwen2p5_7b": {"path": "/models/7b"},
        },
        "machine_inputs": {
            "training_jsonl": {"path": "/data/train.jsonl"},
            "development_jsonl": {"path": "/data/eval.jsonl"},
        },
    }
    predictions = {
        "stage_1p7b": {
            "targets": {
                "raw": {
                    "mu": 0.9,
                    "outer_bias_correction": False,
                    "verification_etas": [
                        0.001 * 2**x for x in (-0.75, -0.25, 0.25, 0.75)
                    ],
                },
                "corrected": {
                    "mu": 0.9,
                    "outer_bias_correction": True,
                    "verification_etas": [
                        0.0008 * 2**x for x in (-0.75, -0.25, 0.25, 0.75)
                    ],
                },
            }
        },
        "stage_7b": {
            "targets": {
                "mu0": {
                    "mu": 0.0,
                    "outer_bias_correction": False,
                    "verification_etas": [0.004 * 2**x for x in (-0.5, 0, 0.5)],
                },
                "raw": {
                    "mu": 0.9,
                    "outer_bias_correction": False,
                    "verification_etas": [0.0007 * 2**x for x in (-0.5, 0, 0.5)],
                },
            }
        },
    }
    return contract, predictions


def test_v9_manifest_has_16_single_gpu_then_12_four_gpu_cells():
    module = load()
    contract, predictions = fixtures()
    cells = module.assign_cells(module.target_cells(predictions), "a" * 40, contract)
    scheduling = module.validate(cells)
    assert scheduling["stage_cell_counts"] == {
        "stage_1p7b": 16,
        "stage_7b": 12,
    }
    a = [cell for cell in cells if cell["stage"] == "stage_1p7b"]
    b = [cell for cell in cells if cell["stage"] == "stage_7b"]
    assert all(len(cell["assignment"]["gpus"]) == 1 for cell in a)
    assert all(
        module.command_value(cell["command"], "--gpu-slots") == "1" for cell in a
    )
    assert all(len(cell["assignment"]["gpus"]) == 4 for cell in b)
    assert all(
        module.command_value(cell["command"], "--gpu-slots") == "4" for cell in b
    )
    assert len(scheduling["stage_7b_queues"]) == 4
    assert all(len(queue) == 3 for queue in scheduling["stage_7b_queues"].values())


def test_only_corrected_1p7b_cells_enable_bias_correction():
    module = load()
    contract, predictions = fixtures()
    cells = module.assign_cells(module.target_cells(predictions), "a" * 40, contract)
    for cell in cells:
        enabled = "--outer-bias-correction" in cell["command"]
        assert enabled == (cell["stage"] == "stage_1p7b" and cell["arm"] == "corrected")
        assert math.isclose(
            float(module.command_value(cell["command"], "--outer-lr")), cell["eta"]
        )


def test_every_eta_has_exactly_both_registered_seeds():
    module = load()
    contract, predictions = fixtures()
    cells = module.assign_cells(module.target_cells(predictions), "a" * 40, contract)
    groups = {}
    for cell in cells:
        groups.setdefault((cell["stage"], cell["arm"], cell["eta"]), set()).add(
            cell["seed"]
        )
    assert len(groups) == 14
    assert all(seeds == {901, 907} for seeds in groups.values())


def test_analysis_bands_and_bootstrap_are_bound_to_predictions():
    module = load()
    _contract, predictions = fixtures()
    contract = {
        "analysis_contract": {
            "bootstrap": {
                "groups": [
                    {"representative": [0, 0], "frequency": 2500},
                    {"representative": [0, 1], "frequency": 5000},
                    {"representative": [1, 1], "frequency": 2500},
                ]
            },
            "gates": {
                "G9A_1P7B": {
                    "absolute_error_band_bits": {"raw": 0.5, "corrected": 0.625}
                },
                "G9B_7B": {"absolute_error_band_bits": {"mu0": 0.5, "raw": 0.75}},
            },
        }
    }
    predictions["stage_1p7b"]["targets"]["raw"][
        "registered_absolute_error_band_bits"
    ] = 0.5
    predictions["stage_1p7b"]["targets"]["corrected"][
        "registered_absolute_error_band_bits"
    ] = 0.625
    predictions["stage_7b"]["targets"]["mu0"]["registered_absolute_error_band_bits"] = (
        0.5
    )
    predictions["stage_7b"]["targets"]["raw"]["registered_absolute_error_band_bits"] = (
        0.75
    )
    module.validate_analysis_binding(contract, predictions)
