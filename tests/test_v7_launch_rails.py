import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_v7_launch_manifest as main_builder  # noqa: E402
import build_v7_prep_manifest as prep_builder  # noqa: E402
import check_v7_v6_drain as drain  # noqa: E402
import run_node_v7 as runner  # noqa: E402
import v7_common as common  # noqa: E402


SOURCE = "a" * 40


def test_prep_manifest_binds_exact_multirank_commands():
    cells = prep_builder.build_cells(SOURCE)
    prep_builder.validate(cells)
    assert len(cells) == 4
    smoke = next(cell for cell in cells if cell["stage"] == "SMOKE")
    pilot = [cell for cell in cells if cell["stage"] == "PILOT"]
    assert smoke["fixed_window_tokens"] == 16 * 4 * 128
    assert sorted(cell["eta"] for cell in pilot) == [0.14, 0.28, 0.56]
    for cell in cells:
        command = cell["command"]
        assert common.command_for(cell, 1) == command
        assert "--barrier-sync" not in command
        assert runner.command_value(command, "--learner-gpus") == "4"
        assert runner.command_value(command, "--settings") == "m2"
        assert runner.command_value(command, "--fixed-window-tokens") == str(
            cell["h"] * 4 * 128
        )
        assert runner.command_value(command, "--token-budget") == str(
            cell["s"] * 128 * 2 * 4
        )


def test_pilot_center_uses_vertex_then_registered_finite_fallback():
    etas = [0.14, 0.28, 0.56]
    center = 0.31
    losses = [1.5 + 0.2 * math.log2(eta / center) ** 2 for eta in etas]
    selected = common.select_pilot_center(etas, losses)
    assert selected["selection_method"] == "accepted_quadratic_vertex"
    assert math.isclose(selected["selected_eta_star"], center, rel_tol=1e-12)

    fallback = common.select_pilot_center(etas, [1.2, math.inf, 1.1])
    assert fallback["selection_method"] == "minimum_finite_pilot_eta_fallback"
    assert fallback["selected_eta_star"] == 0.56


def test_twenty_fleet_hour_boundary_is_deterministic():
    # 24 short + 24 four-times-long cells balance to 60 short-cell durations
    # per node, so a 1,200-second short cell projects to exactly 20 hours.
    full = common.select_grid_variant(1200.0)
    reduced = common.select_grid_variant(1200.0001)
    assert math.isclose(full["projected_fleet_hours"], 20.0)
    assert full["variant"] == "FULL_48"
    assert reduced["variant"] == "REDUCED_T20_MU0_45"


def pilot_readout(variant):
    center = 0.28
    return {
        "selected_grid": {
            "variant": variant,
            "eta_grids": common.derive_eta_grids(center, variant),
        }
    }


def test_full_main_manifest_has_48_balanced_longest_first_cells():
    pilot = pilot_readout("FULL_48")
    cells = main_builder.build_cells(pilot, SOURCE)
    main_builder.validate(cells, pilot)
    assert len(cells) == 48
    assert all(cell["expected"]["c_tokens"] == 262144 for cell in cells)
    loads = {
        node: sum(
            cell["estimated_cost_units"]
            for cell in cells
            if cell["assignment"]["node"] == node
        )
        for node in common.NODES
    }
    assert max(loads.values()) - min(loads.values()) <= 1


def test_reduced_main_manifest_drops_only_three_t20_mu0_seed_cells():
    pilot = pilot_readout("REDUCED_T20_MU0_45")
    cells = main_builder.build_cells(pilot, SOURCE)
    main_builder.validate(cells, pilot)
    assert len(cells) == 45
    by_curve = {}
    for cell in cells:
        by_curve.setdefault((cell["t"], cell["mu"]), set()).add(cell["eta"])
    assert len(by_curve[(20, 0.0)]) == 3
    assert all(
        len(etas) == 4
        for coordinate, etas in by_curve.items()
        if coordinate != (20, 0.0)
    )


def test_every_registered_process_poll_name_uses_bracket_idiom():
    assert drain.PROCESS_PATTERN == (
        "[r]un_slot_v6.py|[c]ompare_diloco.py|[y]eto.learner|[y]eto-syncer"
    )
    assert "run_slot_v6.py" not in drain.PROCESS_PATTERN
    assert "compare_diloco.py" not in drain.PROCESS_PATTERN
