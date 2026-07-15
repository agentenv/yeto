from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "colored_quadratic_suite", ROOT / "scripts" / "colored_quadratic_suite.py"
)
suite = importlib.util.module_from_spec(SPEC)
sys.modules["colored_quadratic_suite"] = suite
assert SPEC.loader is not None
SPEC.loader.exec_module(suite)


def test_suite_is_deterministic_and_machine_json_safe() -> None:
    first = suite.run_suite(seed=1234, steps=256, dimension=3)
    second = suite.run_suite(seed=1234, steps=256, dimension=3)
    assert first == second
    rendered = json.dumps(first, sort_keys=True, allow_nan=False)
    assert json.loads(rendered) == first


def test_all_registered_colored_processes_are_present() -> None:
    report = suite.run_suite(seed=7, steps=192, dimension=2)
    assert [row["kind"] for row in report["processes"]] == [
        "ar1",
        "ar2",
        "ma",
        "oscillatory",
        "nonstationary",
    ]
    for row in report["processes"]:
        assert len(row["empirical_kernel"]) == 9
        assert row["filter"]["max_direct_convolution_error"] < 1e-12
        assert row["quadratic"]["memoryless"]["final_loss"] >= 0.0
        assert row["quadratic"]["nesterov"]["final_loss"] >= 0.0


def test_same_rho1_different_spectrum_changes_filter_moments() -> None:
    contrast = suite.same_rho1_contrast(mu=0.9, age=64)
    assert contrast["ar1"]["kernel_head"][1] == pytest.approx(0.4)
    assert contrast["ar2"]["kernel_head"][1] == pytest.approx(0.4)
    assert contrast["ar1"]["rho2"] == pytest.approx(0.16)
    assert contrast["ar2"]["rho2"] == pytest.approx(-0.176)
    assert abs(contrast["aligned_difference_ar1_minus_ar2"]) > 0.4
    assert abs(contrast["energy_difference_ar1_minus_ar2"]) > 5.0
    assert contrast["conclusion"] == "same_rho1_is_not_sufficient_for_filter_moments"


def test_finite_buffer_age_moves_from_first_step_to_stationary_regime() -> None:
    rows = suite.finite_buffer_age_sweep(mu=0.9, phi=0.6)
    assert rows[0] == {
        "age": 1,
        "aligned_ratio": pytest.approx(1.9),
        "energy_amplification": pytest.approx(3.61),
    }
    aligned = [row["aligned_ratio"] for row in rows]
    energy = [row["energy_amplification"] for row in rows]
    assert aligned == sorted(aligned)
    assert energy == sorted(energy)
    assert aligned[-1] == pytest.approx(2.9565217391304346)
    assert energy[-1] > 5.0 * energy[0]


def test_curvature_rotation_separates_equal_norm_transverse_steps() -> None:
    report = suite.curvature_rotation(eta=0.02)
    flat, sharp = report["arms"]
    assert flat["transverse_norm"] == pytest.approx(sharp["transverse_norm"])
    assert flat["gradient_dot_transverse"] == pytest.approx(0.0)
    assert sharp["gradient_dot_transverse"] == pytest.approx(0.0)
    assert sharp["curvature_energy"] == pytest.approx(10.0 * flat["curvature_energy"])
    assert sharp["observed_excess_loss"] == pytest.approx(
        10.0 * flat["observed_excess_loss"]
    )
    assert flat["prediction_error"] == pytest.approx(0.0, abs=1e-14)
    assert sharp["prediction_error"] == pytest.approx(0.0, abs=1e-14)


def test_nonstationary_process_has_incompatible_half_kernels() -> None:
    report = suite.run_suite(seed=99, steps=1024, dimension=4)
    row = next(row for row in report["processes"] if row["kind"] == "nonstationary")
    assert row["first_half_kernel"][1] > 0.65
    assert row["second_half_kernel"][1] < -0.25
    assert row["theoretical_kernel"] is None


def test_cli_writes_deterministic_machine_readable_output(tmp_path: Path) -> None:
    output = tmp_path / "suite.json"
    assert suite.main(
        [
            "--seed",
            "42",
            "--steps",
            "128",
            "--dimension",
            "2",
            "--output",
            str(output),
        ]
    ) == 0
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["suite"] == "colored_input_quadratic_falsification"
    assert loaded["deterministic_config"]["seed"] == 42
    assert loaded["deterministic_config"]["steps"] == 128
