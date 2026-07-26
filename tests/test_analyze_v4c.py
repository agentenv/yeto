import hashlib
import json
import math
from pathlib import Path

import pytest

from scripts import analyze_v4c
from scripts import build_v4c_launch_manifest


def synthetic_losses():
    optima = {
        (2560, 0.0): 0.028,
        (2560, 0.9): 0.028 * 0.1 * 2.2,
        (10240, 0.0): 0.012,
        (10240, 0.9): 0.012 * 0.1 * 1.1,
    }
    losses = {}
    for coordinate, etas in analyze_v4c.COMBINED_ETA_GRIDS.items():
        optimum = optima[coordinate]
        for eta in etas:
            x = math.log2(eta / optimum)
            for seed_index, seed in enumerate(analyze_v4c.ALL_SEEDS):
                losses[(*coordinate, seed, eta)] = 3.0 + x * x + 0.001 * seed_index
    return losses


def test_five_seed_curves_and_shared_bootstrap():
    losses = synthetic_losses()
    fits = {
        (s, mu): analyze_v4c.curve_fit(losses, s, mu)
        for s in analyze_v4c.S_GRID
        for mu in analyze_v4c.MU_GRID
    }
    assert fits[(2560, 0.0)]["point_count"] == 4
    assert all(fit["interior"] for fit in fits.values())
    d5 = analyze_v4c.d_from_fits(fits[(2560, 0.0)], fits[(2560, 0.9)])
    d20 = analyze_v4c.d_from_fits(fits[(10240, 0.0)], fits[(10240, 0.9)])
    assert d5 == pytest.approx(2.2, rel=1e-10)
    assert d20 == pytest.approx(1.1, rel=1e-10)
    bootstrap = analyze_v4c.bootstrap_all(losses)
    assert bootstrap["status"] == "VALID"
    assert bootstrap["valid_replicates"] == 10_000
    assert bootstrap["minimum_valid_fraction"] == 0.95
    assert bootstrap["monotone_gap"]["ci_95"]["low"] > 0


def test_manifest_builder_registered_shape_and_queue_loads():
    cells = build_v4c_launch_manifest.build_cells("test-commit")
    build_v4c_launch_manifest.validate(cells)
    assert len(cells) == 44
    assert {cell["seed"] for cell in cells} == {541, 547}
    assert sum(cell["s"] == 10240 for cell in cells) == 24
    assert sum(cell["s"] == 2560 for cell in cells) == 20
    for node, gpu in build_v4c_launch_manifest.INITIAL_SLOTS:
        queue = [
            cell
            for cell in cells
            if cell["assignment"] == {"node": node, "gpu": gpu}
        ]
        assert sum(cell["estimated_cost_units"] for cell in queue) == 8
        assert min(cell["slot_queue_index"] for cell in queue if cell["s"] == 10240) == 0
    for node, gpu in build_v4c_launch_manifest.DEFERRED_SLOTS:
        queue = [
            cell
            for cell in cells
            if cell["assignment"] == {"node": node, "gpu": gpu}
        ]
        assert [cell["s"] for cell in sorted(queue, key=lambda cell: cell["slot_queue_index"])] == [10240, 2560]


def test_preregistration_binds_frozen_analyzer():
    repo = Path(__file__).resolve().parent.parent
    contract = json.loads(
        (repo / "experiment-specs/outer-mup-v4c-seedpower-prereg.json").read_text()
    )
    analyzer = repo / contract["frozen_analyzer"]["path"]
    dependency = repo / contract["frozen_analyzer"]["frozen_dependency"]["path"]
    assert hashlib.sha256(analyzer.read_bytes()).hexdigest() == contract["frozen_analyzer"]["sha256"]
    assert hashlib.sha256(dependency.read_bytes()).hexdigest() == contract["frozen_analyzer"]["frozen_dependency"]["sha256"]
