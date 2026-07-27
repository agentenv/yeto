import importlib.util
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_v7", ROOT / "scripts" / "analyze_v7.py"
)
v7 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(v7)


FULL_GRIDS = {
    (2560, 0.0): [2.0**offset for offset in (-1.5, -0.5, 0.5, 1.5)],
    (2560, 0.9): [
        0.1 * v7.G4C_OBSERVED_D[5] * 2.0**offset for offset in (-1.5, -0.5, 0.5, 1.5)
    ],
    (10240, 0.0): [0.4 * 2.0**offset for offset in (-1.5, -0.5, 0.5, 1.5)],
    (10240, 0.9): [
        0.04 * v7.G4C_OBSERVED_D[20] * 2.0**offset for offset in (-1.5, -0.5, 0.5, 1.5)
    ],
}


def synthetic_losses(grids=FULL_GRIDS, *, d20_shift_bits=0.0):
    losses = {}
    seed_offsets = {701: -0.002, 709: 0.0, 719: 0.002}
    centers = {
        (2560, 0.0): 1.0,
        (2560, 0.9): 0.1 * v7.G4C_OBSERVED_D[5],
        (10240, 0.0): 0.4,
        (10240, 0.9): 0.04 * v7.G4C_OBSERVED_D[20] * 2.0**d20_shift_bits,
    }
    for coordinate, etas in grids.items():
        center = centers[coordinate]
        for seed in v7.SEEDS:
            for eta in etas:
                x = math.log2(eta / center)
                losses[(coordinate[0], coordinate[1], seed, eta)] = (
                    1.7 + 0.1 * x * x + seed_offsets[seed]
                )
    return losses


def test_near_bracket_accepts_half_bit_extension_without_clipping():
    etas = [2.0**offset for offset in (-1.5, -0.5, 0.5, 1.5)]
    vertex = 1.75
    losses = [1.0 + 0.2 * (math.log2(eta) - vertex) ** 2 for eta in etas]
    fit = v7.fit_quadratic(etas, losses)
    assert fit["status"] == "NEAR_BRACKETED"
    assert fit["accepted"] is True
    assert math.isclose(fit["vertex_log2_eta"], vertex, abs_tol=1e-10)


def test_full_grid_recovers_registered_constants_and_passes():
    readout = v7.analyze_losses(synthetic_losses(), FULL_GRIDS)
    assert readout["gate"]["verdict"] == "PASS"
    assert math.isclose(readout["D_obs"]["T5"], v7.G4C_OBSERVED_D[5], rel_tol=1e-10)
    assert math.isclose(readout["D_obs"]["T20"], v7.G4C_OBSERVED_D[20], rel_tol=1e-10)
    assert readout["bootstrap"]["valid_replicates"] == 10_000
    assert readout["bootstrap"]["monotone_gap"]["ci_95"]["low"] > 0.0


def test_conditional_three_point_t20_mu0_grid_is_supported():
    grids = dict(FULL_GRIDS)
    grids[(10240, 0.0)] = [0.4 * 2.0**offset for offset in (-1.5, 0.0, 1.5)]
    readout = v7.analyze_losses(synthetic_losses(grids), grids)
    assert readout["gate"]["verdict"] == "PASS"
    t20_mu0 = next(
        fit for fit in readout["curves"] if fit["s"] == 10240 and fit["mu"] == 0.0
    )
    assert t20_mu0["point_count"] == 3


def test_complete_but_nonmonotone_science_fails_not_not_evaluable():
    readout = v7.analyze_losses(synthetic_losses(d20_shift_bits=1.0), FULL_GRIDS)
    assert readout["gate"]["evaluable"] is True
    assert readout["gate"]["verdict"] == "FAIL"
    assert (
        readout["gate"]["conditions"]["paired_monotone_decrease_ci_excludes_zero"]
        is False
    )


def test_missing_evidence_is_not_evaluable():
    losses = synthetic_losses()
    losses.pop(next(iter(losses)))
    readout = v7.analyze_losses(losses, FULL_GRIDS)
    assert readout["gate"]["verdict"] == "NOT_EVALUABLE"
