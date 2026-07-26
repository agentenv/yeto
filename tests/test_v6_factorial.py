import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
JSON_SPEC = ROOT / "experiment-specs/outer-mup-v6-factorial-prereg.json"
MD_SPEC = ROOT / "experiment-specs/outer-mup-v6-factorial-prereg.md"


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_builder_materializes_balanced_hash_bound_900_cell_grid():
    builder = load_script("build_v6_launch_manifest")
    contract = json.loads(JSON_SPEC.read_text())
    cells = builder.build_cells(contract, "f" * 40)
    balance = builder.validate(cells)
    assert len(cells) == 900
    assert len({cell["cell_id"] for cell in cells}) == 900
    eta_counts = Counter(
        (cell["t"], cell["s"], cell["arm"], cell["eta_index"])
        for cell in cells
    )
    assert len(eta_counts) == 4 * 3 * 3 * 5
    assert set(eta_counts.values()) == {5}
    assert max(balance["cost_loads"].values()) - min(
        balance["cost_loads"].values()
    ) <= 1
    assert sum(balance["queue_lengths"].values()) == 900


def test_commands_bind_H_work_arm_seed_gpu_and_roomy_volume():
    builder = load_script("build_v6_launch_manifest")
    contract = json.loads(JSON_SPEC.read_text())
    cells = builder.build_cells(contract, "e" * 40)
    for cell in cells:
        command = cell["command"]
        value = lambda flag: command[command.index(flag) + 1]
        assert int(value("--fixed-window-microsteps")) == cell["h"]
        assert int(value("--fixed-window-tokens")) == cell["h"] * 128
        assert int(value("--learner-max-steps")) == cell["s"]
        assert int(value("--syncer-total-steps")) == 4 * cell["t"]
        assert int(value("--gpu-offset")) == cell["assignment"]["gpu"]
        assert int(value("--training-seed")) == cell["training_seed"]
        assert f"seed-{cell['seed']}" in value("--data")
        assert value("--work-dir").startswith("/root/yeto-results-v6/")
        assert value("--report-dir").startswith("/root/yeto-results-v6/")
        assert ("--outer-bias-correction" in command) == (
            cell["arm"] == "corrected"
        )
        assert builder.canonical_sha256(command) == cell["command_hash"]


def test_retry_groups_are_whole_five_eta_paired_seed_curves():
    builder = load_script("build_v6_launch_manifest")
    contract = json.loads(JSON_SPEC.read_text())
    cells = builder.build_cells(contract, "d" * 40)
    groups = defaultdict(list)
    for cell in cells:
        groups[cell["retry_group_id"]].append(cell)
    assert len(groups) == 4 * 3 * 3 * 5
    for group in groups.values():
        assert len(group) == 5
        assert sorted(cell["eta_index"] for cell in group) == list(range(5))
        assert len({(cell["t"], cell["s"], cell["arm"], cell["seed"]) for cell in group}) == 1


def test_runner_constants_bind_frozen_contract_bytes():
    runner = load_script("run_slot_v6")
    contract = json.loads(JSON_SPEC.read_text())
    assert hashlib.sha256(JSON_SPEC.read_bytes()).hexdigest() == runner.CONTRACT_JSON_SHA256
    assert hashlib.sha256(MD_SPEC.read_bytes()).hexdigest() == runner.CONTRACT_MD_SHA256
    analyzer = ROOT / contract["frozen_analyzer"]["path"]
    assert hashlib.sha256(analyzer.read_bytes()).hexdigest() == runner.ANALYZER_SHA256


def test_gate_proof_requires_48_cells_no_v4_processes_and_g5(tmp_path):
    runner = load_script("run_slot_v6")
    valid = {
        "schema": "yeto_outer_mup_v6_gate_proof_v1",
        "status": "PASS",
        "v4": {
            "unique_completed_cells": 48,
            "run_slot_processes": {"h200-n1": [], "h200-n2": []},
        },
        "v5": {"verdict_line": "G5 VERDICT: SNOO_NULL"},
    }
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(valid))
    assert runner.load_gate_proof(path)["status"] == "PASS"
    valid["v4"]["unique_completed_cells"] = 47
    path.write_text(json.dumps(valid))
    with pytest.raises(SystemExit):
        runner.load_gate_proof(path)
