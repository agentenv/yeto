import hashlib
import importlib.util
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "experiment-specs/outer-mup-v6-factorial-prereg.json"
ANALYZER = ROOT / "scripts/analyze_v6.py"


def load_analyzer():
    spec = importlib.util.spec_from_file_location("analyze_v6", ANALYZER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_registered_factorial_and_hashes_close_exactly():
    contract = json.loads(SPEC.read_text())
    cells = contract["design"]["factorial_cells"]
    seeds = contract["design"]["seeds"]
    assert len(cells) == 12
    assert len(cells) * 3 * 5 * len(seeds) == 900
    assert {(cell["T"], cell["S"]) for cell in cells} == {
        (t, s)
        for t in (2, 5, 10, 20)
        for s in (2560, 5120, 10240)
    }
    for cell in cells:
        assert cell["S"] % cell["T"] == 0
        assert cell["H"] == cell["S"] // cell["T"]
        assert cell["S"] // cell["H"] == cell["T"]
        assert math.isclose(
            cell["code_true_D"], 1.0 / (1.0 - 0.9 ** (cell["T"] + 1))
        )
        for record in cell["eta_grids"].values():
            ratios = [eta / record["center"] for eta in record["etas"]]
            assert len(ratios) == 5
            assert all(
                math.isclose(observed, expected, rel_tol=1e-14)
                for observed, expected in zip(
                    ratios, (0.5, 2.0**-0.5, 1.0, 2.0**0.5, 2.0)
                )
            )
    expected_hash = contract["frozen_analyzer"]["sha256"]
    assert hashlib.sha256(ANALYZER.read_bytes()).hexdigest() == expected_hash


def test_quadratic_recovers_interior_vertex():
    analyzer = load_analyzer()
    eta_star = 0.04
    etas = [0.01, 0.02, 0.04, 0.08, 0.16]
    losses = [2.0 + 0.25 * math.log2(eta / eta_star) ** 2 for eta in etas]
    fit = analyzer.fit_quadratic(etas, losses)
    assert fit["status"] == "INTERIOR"
    assert math.isclose(fit["eta_star"], eta_star, rel_tol=1e-12)


def test_quadratic_rejects_boundary_vertex():
    analyzer = load_analyzer()
    etas = [0.01, 0.02, 0.04, 0.08, 0.16]
    losses = [math.log2(eta / 0.01) ** 2 for eta in etas]
    fit = analyzer.fit_quadratic(etas, losses)
    assert fit["status"] == "UNBRACKETED"
    assert fit["eta_star"] is None


def test_registered_surface_exactly_predicts_a_member_of_family():
    analyzer = load_analyzer()
    coefficients = (0.2, 0.3, -0.1, 0.05)
    d_values = {}
    for t in analyzer.T_GRID:
        for s in analyzer.S_GRID:
            features = analyzer.surface_features(t, s)
            log2_d = sum(left * right for left, right in zip(features, coefficients))
            d_values[(t, s, "raw")] = 2.0**log2_d
    result = analyzer.heldout_analysis(d_values, "raw")
    assert result["success_count"] == 4
    assert result["pass"] is True
    assert all(
        math.isclose(observed, expected, abs_tol=1e-12)
        for observed, expected in zip(
            result["surface"]["coefficients"], coefficients
        )
    )


def test_holdout_partition_is_fixed_and_exhaustive():
    analyzer = load_analyzer()
    contract = json.loads(SPEC.read_text())
    holdouts = set(map(tuple, contract["heldout_prediction"]["heldout_cells"]))
    training = set(map(tuple, contract["heldout_prediction"]["training_cells"]))
    factorial = {
        (t, s) for t in analyzer.T_GRID for s in analyzer.S_GRID
    }
    assert holdouts == analyzer.HOLDOUTS
    assert not holdouts & training
    assert holdouts | training == factorial
    assert len(holdouts) == 4
    assert len(training) == 8
