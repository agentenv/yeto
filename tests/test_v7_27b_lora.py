import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "experiment-specs" / "outer-mup-v7-27b-lora-prereg.json"
PRIOR = ROOT / "experiment-specs" / "outer-mup-v7-27b-lora-noise-prior.json"
GATESIM = ROOT / "experiment-specs" / "outer-mup-v7-27b-lora-gatesim.json"
ANALYZER = ROOT / "scripts" / "analyze_v7.py"
PRIOR_BUILDER = ROOT / "scripts" / "build_v7_noise_prior.py"
SIMULATOR = ROOT / "scripts" / "gatesim_v7.py"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_contract_binds_frozen_analysis_and_simulation_artifacts():
    contract = json.loads(CONTRACT.read_text())
    feasibility = contract["gate_feasibility_simulation"]
    assert contract["schema"] == "yeto_outer_mup_v7_27b_lora_prereg_v1"
    assert contract["frozen_analysis"]["sha256"] == sha256(ANALYZER)
    assert feasibility["prior_builder"]["sha256"] == sha256(PRIOR_BUILDER)
    assert feasibility["prior"]["sha256"] == sha256(PRIOR)
    assert feasibility["simulator"]["sha256"] == sha256(SIMULATOR)
    assert feasibility["report"]["sha256"] == sha256(GATESIM)


def test_design_closes_cell_counts_and_multirank_token_windows():
    contract = json.loads(CONTRACT.read_text())
    protocol = contract["common_protocol"]
    design = contract["main_design"]
    assert protocol["learner_islands_m"] == 2
    assert protocol["ranks_per_island"] == 4
    assert protocol["gpus_per_cell"] == 8
    assert protocol["fixed_window_tokens"] == 512 * 4 * 128
    assert protocol["barrier_sync"] is False
    assert protocol["strict_quorum"] is True
    assert design["full_variant"]["log2_offsets_from_each_center"] == [
        -1.5,
        -0.5,
        0.5,
        1.5,
    ]
    assert design["conditional_reduced_variant"]["T20_mu0_log2_offsets"] == [
        -1.5,
        0.0,
        1.5,
    ]
    assert 2 * 2 * 4 * 3 == 48
    assert 48 - 3 == 45


def test_gate_bands_are_exact_observed_constants_plus_or_minus_fifty_percent():
    contract = json.loads(CONTRACT.read_text())
    bands = contract["frozen_analysis"]["bands"]
    d5 = 1.7416157949788522
    d20 = 1.2806943474449415
    assert bands["T5"] == [0.5 * d5, 1.5 * d5]
    assert bands["T20"] == [0.5 * d20, 1.5 * d20]


def test_gate_simulation_records_primary_and_stress_evaluability():
    report = json.loads(GATESIM.read_text())
    primary = {item["variant"]: item for item in report["primary"]}
    assert set(primary) == {"FULL_48", "REDUCED_T20_MU0_45"}
    assert all(item["P_eval"] == 1.0 for item in primary.values())
    assert all(item["counts"]["evaluable"] == 5000 for item in primary.values())
    assert all(item["P_eval"] == 1.0 for item in report["sensitivity"])


def test_pilot_and_fleet_hour_variant_rule_are_closed_before_outcomes():
    contract = json.loads(CONTRACT.read_text())
    assert contract["pilot"]["etas"] == [0.14, 0.28, 0.56]
    assert contract["pilot"]["seed"] == 691
    assert contract["main_design"]["seeds"] == [701, 709, 719]
    rule = contract["fleet_hour_rule"]
    assert rule["threshold_fleet_hours"] == 20.0
    assert "FULL_48" in rule["selection"]
    assert "REDUCED_T20_MU0_45" in rule["selection"]
