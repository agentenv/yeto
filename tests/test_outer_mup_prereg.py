"""Machine/readable consistency checks for the outer-muP two-day contract."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "experiment-specs" / "outer-mup-2day-prereg.json"
MD_PATH = ROOT / "experiment-specs" / "outer-mup-2day-prereg.md"
THEORY_SENTENCE = (
    "Khaled et al. 2509.10439 Thm 2's 1/(1-mu) multiplier is the rho->1 "
    "limit of the registered filter law eta_eff=eta(1+mu/(1-mu*rho)); "
    "E1 discriminates them."
)


def _registry() -> dict:
    return json.loads(JSON_PATH.read_text(encoding="utf-8"))


def test_contract_files_cross_reference_and_freeze_theory_sentence():
    registry = _registry()
    markdown = MD_PATH.read_text(encoding="utf-8")
    assert registry["schema"] == "yeto_outer_mup_2day_prereg_v1"
    assert registry["status"] == "PREREGISTERED"
    assert registry["readable_contract"] == (
        "experiment-specs/outer-mup-2day-prereg.md"
    )
    assert registry["theory_positioning"] == THEORY_SENTENCE
    assert THEORY_SENTENCE in markdown
    assert "BIND_" not in JSON_PATH.read_text(encoding="utf-8")
    assert set(registry["stages"]) == {f"E{index}" for index in range(8)}


def test_e1_grid_seeds_ladders_work_and_run_counts_are_closed():
    registry = _registry()
    e1 = registry["stages"]["E1"]
    assert e1["h"] == [16, 64, 256, 512]
    assert e1["m"] == [4]
    assert e1["mu_full"] == [0.0, 0.9]
    assert e1["mu_partial"] == {"0.5": [16, 256]}
    assert e1["primary_seeds"] == [101, 103, 107, 109, 113]
    assert e1["contested_topup_seeds"] == [127, 131, 137]
    assert e1["contested_topup_coordinates"] == [
        {"h": 256, "mu": 0.9},
        {"h": 512, "mu": 0.9},
    ]

    offsets = registry["eta_ladder"]["log2_offsets"]
    assert offsets == [-0.75, -0.25, 0.25, 0.75]
    for center_text, grid in e1["eta_grid_by_center"].items():
        center = float(center_text)
        assert len(grid) == 4
        expected = [center * 2**offset for offset in offsets]
        assert grid == expected
        geometric_center = math.prod(grid) ** 0.25
        assert math.isclose(geometric_center, center, rel_tol=1e-14)

    assert e1["registered_coordinate_families"] == 10
    assert e1["primary_training_runs"] == 10 * 4 * 5 == 200
    assert e1["topup_training_runs"] == 2 * 4 * 3 == 24
    assert e1["total_training_runs"] == 224
    assert e1["work"]["learner_inner_steps"] == 2560
    assert e1["work"]["global_outer_commits_by_h"] == {
        "16": 640,
        "64": 160,
        "256": 40,
        "512": 20,
    }


def test_g1_is_five_seed_two_high_h_ci_plus_registered_spearman_and_stop():
    registry = _registry()
    g1 = registry["gates"]["G1"]
    assert g1["primary_seeds_only"] == [101, 103, 107, 109, 113]
    requirements = "\n".join(g1["requirements_for_PASS"])
    assert "H256" in requirements and "excludes 1.0" in requirements
    assert "H512" in requirements and "excludes 1.0" in requirements
    assert "H16,H64,H256,H512" in requirements
    assert ">=0.8" in requirements
    assert g1["on_FAIL"].startswith("stop the entire experimental program")
    assert registry["stages"]["E1"]["topup_role"].startswith(
        "prespecified robustness report only"
    )


def test_e2_m_axis_and_e3_prediction_lock_and_verification_denominator():
    registry = _registry()
    e2 = registry["stages"]["E2"]
    assert e2["new_m"] == [1, 2, 8]
    assert e2["anchor_m_reused_from_E1"] == 4
    assert e2["h"] == [16, 256]
    assert e2["mu"] == [0.0, 0.9]
    assert len(e2["seeds"]) == 5
    assert e2["new_training_runs"] == 3 * 2 * 2 * 4 * 5 == 240

    e3 = registry["stages"]["E3"]
    seal = e3["sealed_prediction"]
    target_count = (
        len(seal["target_scales"])
        * len(seal["target_h"])
        * len(seal["target_m"])
        * len(seal["target_mu"])
    )
    assert target_count == seal["sealed_cell_count"] == 48
    assert seal["must_be_committed_before_verification_launch"] is True
    assert (
        seal[
            "must_be_pushed_and_reachable_from_experiment_branch_before_verification_launch"
        ]
        is True
    )
    assert e3["verification"]["training_runs"] == 48 * 4 * 5 == 960
    assert e3["prediction_rule"]["eta_prediction"] == ("eta0_target/A(mu,rho_target)")

    g2 = registry["gates"]["G2"]
    assert g2["sealed_cell_count"] == 48
    assert g2["minimum_hit_fraction"] == 0.75
    assert g2["minimum_hits"] == 36
    assert "all 48 cells remain in the denominator" in g2["requirements_for_PASS"]


def test_boundary_interconnect_optional_stages_evidence_and_wall_clocks():
    registry = _registry()
    e4 = registry["stages"]["E4"]
    assert e4["mu"] == [0.9, 0.95]
    assert e4["h"] == [256, 512]
    assert e4["buffer_surgery"]["causal_pairs"] == [
        ["factual_buffer", "zero_buffer"],
        [
            "factual_buffer",
            "same_norm_current_gradient_aligned_buffer",
        ],
    ]

    e5 = registry["stages"]["E5"]
    assert e5["scale"] == "1.7b"
    assert e5["physical_nodes"] == 2
    assert e5["m"] == 8 and e5["learners_per_node"] == 4
    assert e5["loopback_only_or_single_node_emulation_forbidden"] is True

    e6 = registry["stages"]["E6"]
    assert e6["required"] is False and e6["slack_contingent"] is True
    assert "SKIPPED_NO_SLACK" in registry["closed_vocabularies"]["snoo_verdict"]
    e7 = registry["stages"]["E7"]
    assert e7["required"] is False and e7["proof_must_compile_in_lean_project"] is True

    evidence = registry["work_evidence_contract"]
    assert evidence["finite_eval_loss_required"] is True
    assert evidence["telemetry_required"] is True
    requirements = "\n".join(evidence["completed_cell_requires_all"])
    assert "full registered learner inner steps" in requirements
    assert "finite evaluation loss" in requirements
    assert "rho telemetry JSONL present" in requirements

    clocks = registry["wall_clock_contract"]
    assert clocks["program_ceiling_hours"] == 48
    assert clocks["stage_ceiling_hours"]["E1"] == 14
    assert clocks["stage_ceiling_hours"]["E3_verification"] == 14
