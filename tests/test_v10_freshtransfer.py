import importlib.util
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiment-specs/outer-mup-v10-freshtransfer-prereg.json"
GATESIM_PATH = ROOT / "experiment-specs/outer-mup-v10-freshtransfer-gatesim.json"


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_contract_exact_rates_and_closed_gate():
    contract = json.loads(CONTRACT_PATH.read_text())
    predictions = contract["surface_prediction"]["directed_penalties_bits_per_token"]
    threshold = contract["gate"]["penalty_threshold_bits_per_token"]
    assert threshold == 0.5 * min(predictions.values())
    assert contract["prescriptions"] == {
        **contract["prescriptions"],
        "eta_T5": 0.003191644884294105,
        "eta_T20": 0.0008223020084526104,
        "eta_T40": 0.0003971228256207733,
    }
    assert contract["gate"]["closed_vocabulary"] == [
        "PENALTY_CONFIRMED",
        "PENALTY_NULL",
        "PENALTY_REVERSED",
    ]
    assert contract["gate"]["not_evaluable_forbidden"] is True
    assert contract["design"]["cell_count"] == 18


def test_gatesim_clears_mandatory_bar_and_binds_noise():
    report = json.loads(GATESIM_PATH.read_text())
    assert report["feasibility"]["P_eval"] >= 0.8
    assert report["feasibility"]["clears_mandatory_bar"] is True
    assert report["feasibility"]["P_PENALTY_CONFIRMED_under_surface_prediction"] >= 0.8
    noise = report["banked_v4_seed_noise"]
    assert noise["contrast_count"] == 30
    assert noise["maximum_paired_contrast_sd_bits"] < report["gate"][
        "penalty_threshold_bits_per_token"
    ]


def losses_at_pair_penalties(analyzer, penalties):
    losses = {}
    by_id = {item[0]: item[1:] for item in analyzer.PAIR_DEFINITIONS}
    for transfer_id, penalty in penalties.items():
        transfer_role, comparator_role = by_id[transfer_id]
        for seed in analyzer.EXPECTED_SEEDS:
            losses[comparator_role, seed] = 1.0
            losses[transfer_role, seed] = 1.0 + penalty * math.log(2.0)
    return losses


def test_analyzer_closed_vocabulary_trichotomy():
    analyzer = load_module("analyze_v10_test", "scripts/analyze_v10.py")
    threshold = json.loads(CONTRACT_PATH.read_text())["gate"][
        "penalty_threshold_bits_per_token"
    ]
    ids = [item[0] for item in analyzer.PAIR_DEFINITIONS]
    confirmed = analyzer.analyze(
        losses_at_pair_penalties(
            analyzer, {transfer_id: threshold + 0.1 for transfer_id in ids}
        ),
        threshold,
    )
    assert confirmed["verdict"] == "PENALTY_CONFIRMED"
    reversed_result = analyzer.analyze(
        losses_at_pair_penalties(
            analyzer, {transfer_id: -threshold - 0.1 for transfer_id in ids}
        ),
        threshold,
    )
    assert reversed_result["verdict"] == "PENALTY_REVERSED"
    null = analyzer.analyze(
        losses_at_pair_penalties(
            analyzer,
            {
                ids[0]: threshold + 0.1,
                ids[1]: 0.0,
                ids[2]: threshold + 0.1,
            },
        ),
        threshold,
    )
    assert null["verdict"] == "PENALTY_NULL"


def test_manifest_builder_materializes_six_by_three_exact_cells():
    builder = load_module(
        "build_v10_manifest_test", "scripts/build_v10_launch_manifest.py"
    )
    contract = json.loads(CONTRACT_PATH.read_text())
    cells = builder.materialize(
        builder.assign(builder.design_cells(contract)), "1" * 40
    )
    assert len(cells) == 18
    assert {cell["seed"] for cell in cells} == {941, 947, 953}
    assert len({cell["role"] for cell in cells}) == 6
    assert all(cell["mu"] == 0.9 for cell in cells)
    assert all("--outer-bias-correction" not in cell["command"] for cell in cells)
    assert all(
        builder.HELDOUT["path"]
        == cell["command"][cell["command"].index("--prebound-development-eval") + 1]
        for cell in cells
    )
    assert all(
        cell["eta"]
        == float(cell["command"][cell["command"].index("--outer-lr") + 1])
        for cell in cells
    )
