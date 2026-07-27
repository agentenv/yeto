import importlib.util
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/gatesim_v9.py"


def load():
    spec = importlib.util.spec_from_file_location("gatesim_v9", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_two_seed_bootstrap_groups_are_exact_and_exhaustive():
    module = load()
    assert len(module.BOOTSTRAP_GROUPS) == 3
    assert sum(frequency for _, frequency in module.BOOTSTRAP_GROUPS) == 10_000
    assert {tuple(sorted(draw)) for draw, _ in module.BOOTSTRAP_GROUPS} == {
        (0, 0),
        (0, 1),
        (1, 1),
    }


def test_noise_profile_interpolation_clamps_and_interpolates():
    module = load()
    profile = [(-1.0, 0.1), (0.0, 0.2), (1.0, 0.4)]
    assert module.interpolate_profile(profile, -2.0) == 0.1
    assert module.interpolate_profile(profile, 2.0) == 0.4
    assert math.isclose(module.interpolate_profile(profile, 0.5), 0.3)


def test_synthetic_low_noise_gate_is_feasible_and_bands_are_registered():
    module = load()
    gate = {
        "offsets": [-0.75, -0.25, 0.25, 0.75],
        "targets": {
            "raw": {
                "name": "raw",
                "curvature_a": 0.2,
                "rung_sd": [0.002] * 4,
                "sources": [],
            },
            "corrected": {
                "name": "corrected",
                "curvature_a": 0.15,
                "rung_sd": [0.002] * 4,
                "sources": [],
            },
        },
    }
    result = module.summarize_gate("G9A_1P7B", gate, simulations=100, seed=12345)
    assert result["feasibility_pass"] is True
    assert result["P_eval"] >= 0.8
    assert result["P_pass_given_evaluable_under_centered_null"] >= 0.8
    assert result["registered_minimum_valid_bootstrap_refits"] is not None
    assert result["registered_absolute_error_band_bits"] == {
        "raw": 0.5,
        "corrected": 0.5,
    }
