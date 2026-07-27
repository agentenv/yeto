import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

from scripts import v6_surface_selection as selection


ROOT = Path(__file__).resolve().parents[1]
FROZEN_ANALYZER = ROOT / "scripts/analyze_v6.py"
MECHANISM_FITTER = ROOT / "scripts/fit_v6_mechanism.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def family_values(family_id, coefficients, arm="raw"):
    values = {}
    for t in selection.T_GRID:
        for s in selection.S_GRID:
            features = selection.surface_features(t, s, family_id)
            log2_d = sum(
                left * right
                for left, right in zip(features, coefficients)
            )
            values[(t, s, arm)] = 2.0**log2_d
    return values


def test_selection_module_matches_frozen_analyzer_exactly():
    analyzer = load_module(FROZEN_ANALYZER, "frozen_analyze_v6_for_mechanism")
    for family_id, coefficients in (
        ("F1", (0.2, 0.3, -0.1)),
        ("F2", (0.2, 0.3, -0.1, 0.05)),
        ("F3", (0.2, 0.3, -0.1, 0.05)),
    ):
        values = family_values(family_id, coefficients)
        expected = analyzer.fit_surface(values, "raw")
        observed = selection.select_surface(values, "raw")
        assert observed["status"] == "REGISTERED_COMPLETE"
        assert observed["family_id"] == expected["family_id"]
        assert np.allclose(
            observed["coefficients"], expected["coefficients"], atol=1e-12
        )
        expected_scores = expected["model_selection"]["candidate_scores"]
        observed_scores = observed["model_selection"]["candidate_scores"]
        assert [row["family_id"] for row in observed_scores] == [
            row["family_id"] for row in expected_scores
        ]
        assert np.allclose(
            [row["loo_rmse_bits"] for row in observed_scores],
            [row["loo_rmse_bits"] for row in expected_scores],
            atol=1e-12,
        )


def test_selection_never_reads_holdout_outcomes():
    values = family_values("F2", (0.2, 0.3, -0.1, 0.05))
    first = selection.select_surface(values, "raw")
    for t, s in selection.HOLDOUT_CELLS:
        values[(t, s, "raw")] *= 64.0
    second = selection.select_surface(values, "raw")
    assert first == second


def test_partial_selection_is_explicitly_non_final():
    values = family_values("F2", (0.2, 0.3, -0.1, 0.05))
    cells = (
        (2, 2560),
        (2, 5120),
        (5, 2560),
        (10, 5120),
        (20, 10240),
    )
    result = selection.select_surface(values, "raw", cells=cells)
    assert result["status"] == "PROVISIONAL_PARTIAL_NON_FINAL"
    assert result["non_final"] is True
    assert result["family_id"] == "F2"


def test_complete_partial_curve_matches_registered_quadratic_convention():
    fitter = load_module(MECHANISM_FITTER, "fit_v6_mechanism_curve_test")
    analyzer = load_module(FROZEN_ANALYZER, "frozen_analyze_v6_curve_test")
    t, s, arm = 5, 5120, "raw"
    etas = [0.01, 0.02, 0.04, 0.08, 0.16]
    seeds = [601, 607, 613]
    cells = [
        {"t": t, "s": s, "arm": arm, "eta": eta, "seed": seed}
        for eta in etas
        for seed in seeds
    ]
    manifest = {"cells": cells}
    losses = {
        (t, s, arm, seed, eta): (
            2.0 + 0.25 * math.log2(eta / 0.04) ** 2 + seed * 1e-8
        )
        for eta in etas
        for seed in seeds
    }
    result = fitter.fit_partial_curve(manifest, losses, t, s, arm)
    mean_losses = [
        sum(losses[(t, s, arm, seed, eta)] for seed in seeds) / len(seeds)
        for eta in etas
    ]
    expected = analyzer.fit_quadratic(etas, mean_losses)
    assert result["status"] == "INTERIOR"
    assert result["source_complete"] is True
    assert math.isclose(result["eta_star"], expected["eta_star"], rel_tol=1e-12)
    assert np.allclose(
        [result["a"], result["b"], result["c"]],
        [expected["a"], expected["b"], expected["c"]],
        atol=1e-12,
    )


def test_lane_a_full_matrix_spectral_predictions_reproduce_referee_values():
    fitter = load_module(MECHANISM_FITTER, "fit_v6_mechanism_spectral_test")
    theta = np.asarray(fitter.PUBLISHED_THETA, dtype=float)
    raw = fitter.DObservation(
        "raw-H512-T5", "raw", 512, 5, 2560, 1.0, True
    )
    corrected = fitter.DObservation(
        "corrected-H512-T20", "corrected", 512, 20, 10240, 1.0, True
    )
    predictions = fitter.predict_spectral_dataset(theta, [raw, corrected])
    assert math.isclose(predictions[0], 2.32742, rel_tol=4e-5)
    assert math.isclose(predictions[1], 0.711510, rel_tol=4e-5)


def test_spectral_training_observations_exclude_registered_holdouts():
    fitter = load_module(MECHANISM_FITTER, "fit_v6_mechanism_holdout_test")
    values = {
        (t, s, arm): 1.0 + 0.01 * t
        for arm in fitter.MOMENTUM_ARMS
        for t in (2, 5, 10, 20)
        for s in (2560, 5120, 10240)
    }
    rows = [
        {
            "T": t,
            "S": s,
            "arm": arm,
            "source_complete": True,
        }
        for t, s, arm in values
    ]
    training = fitter.d_observations_from_values(values, rows)
    assert len(training) == 16
    assert {(row.T, row.S) for row in training} == set(selection.TRAINING_CELLS)
    assert not ({(row.T, row.S) for row in training} & set(selection.HOLDOUT_CELLS))
