"""CPU-only contract tests for the TONIGHT-8.5 registered program."""

from __future__ import annotations

import json
import hashlib
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_tonight85_manifest as manifests  # noqa: E402
from tonight85_analysis import analyze_scan, fit_quadratic  # noqa: E402


def load_contract(name: str) -> dict:
    return json.loads((ROOT / "experiment-specs" / name).read_text())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_v12_and_v13_have_exact_72_cell_designs() -> None:
    for name, optimizer in (
        ("outer-mup-v12-heavy-ball-prereg.json", "heavy-ball"),
        ("outer-mup-v13-pythia-ultrachat-prereg.json", "nesterov"),
    ):
        contract = load_contract(name)
        assert contract["status"] == "REGISTERED"
        assert contract["design"]["cells"] == 72
        assert contract["design"]["T"] == [2, 5, 20]
        assert contract["design"]["seeds"] == [981, 983, 991]
        assert len(contract["design"]["grids"]) == 3
        for coordinate in contract["design"]["grids"]:
            assert coordinate["S"] == 2560
            assert coordinate["H"] * coordinate["T"] == 2560
            assert len(coordinate["mu0"]["etas"]) == 4
            assert len(coordinate["mu0.9"]["etas"]) == 4
        assert contract["outer_convention"]["flag"].endswith(optimizer)


def test_static_manifest_counts_and_interleaves_each_gpu_queue() -> None:
    manifest = manifests.build("static", "a" * 40, None)
    manifests.validate(manifest)
    counts = {
        program: sum(cell["program"] == program for cell in manifest["cells"])
        for program in {cell["program"] for cell in manifest["cells"]}
    }
    assert counts == {
        "v12": 72,
        "v13": 72,
        "v11_anchor": 6,
        "v7_smoke": 1,
        "v7_pilot": 3,
    }
    short = [cell for cell in manifest["cells"] if cell["stage"] == "short_scans"]
    for slot_id in {cell["slot_id"] for cell in short}:
        queue = sorted(
            [cell for cell in short if cell["slot_id"] == slot_id],
            key=lambda cell: cell["slot_queue_index"],
        )
        assert len(queue) == 9
        assert all(
            left["program"] != right["program"] for left, right in zip(queue, queue[1:])
        )
    assert all(
        cell["outer_optimizer"] == "heavy-ball"
        for cell in short
        if cell["program"] == "v12"
    )
    assert all(
        cell["outer_optimizer"] == "nesterov"
        for cell in short
        if cell["program"] == "v13"
    )


def test_v11_conditional_manifest_is_exactly_twenty_truth_cells() -> None:
    predictions = {
        "coordinates": {
            "smollm2_135m_t80": {
                "ground_truth_etas": [0.001, 0.002, 0.003, 0.004, 0.005]
            },
            "smollm2_1p7b_t40": {
                "ground_truth_etas": [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]
            },
        }
    }
    manifest = manifests.build("v11-truth", "b" * 40, predictions)
    manifests.validate(manifest)
    assert len(manifest["cells"]) == 20
    assert {cell["seed"] for cell in manifest["cells"]} == {971, 977}
    assert {cell["eta_index"] for cell in manifest["cells"]} == set(range(5))


def test_reduced_v7_keeps_original_island_arithmetic() -> None:
    manifest = manifests.build("static", "c" * 40, None)
    pilot = next(cell for cell in manifest["cells"] if cell["program"] == "v7_pilot")
    command = pilot["command"]
    assert pilot["assignment"]["gpus"] == list(range(8))
    assert command[command.index("--token-budget") + 1] == str(2560 * 128 * 2 * 4)
    assert command[command.index("--fixed-window-tokens") + 1] == str(512 * 4 * 128)
    assert command[command.index("--learner-gpus") + 1] == "4"
    assert pilot["expected"]["ranks_per_learner"] == 4


def test_v11_ratio_rule_is_shape_preserving_not_raw_f3_extrapolation() -> None:
    contract = load_contract("outer-mup-v11-ratio-transport-prereg.json")
    rule = contract["ratio_rule"]
    assert rule["D_T40"] > rule["D_T80"] > 1.0
    assert math.isclose(rule["D_T40"], 1.044905865516022, rel_tol=0, abs_tol=1e-15)
    assert math.isclose(rule["D_T80"], 1.01326103825243, rel_tol=0, abs_tol=1e-15)
    assert "do not evaluate F3" in rule["far_horizon_extraction"]


def test_pythia_and_ultrachat_are_revision_and_hash_bound() -> None:
    contract = load_contract("outer-mup-v13-pythia-ultrachat-prereg.json")
    assert contract["model"]["id"] == "EleutherAI/pythia-160m"
    assert contract["model"]["revision"] == "50f5173d932e8e61f858120bcb800b97af589f46"
    assert contract["data"]["dataset"] == "HuggingFaceH4/ultrachat_200k"
    assert contract["data"]["revision"] == "8049631c405ae6576f93f445c6b8166f76f5505a"
    assert contract["data"]["corpus_is_different"] is True
    assert contract["data"]["files"]["train"]["sha256"] == (
        "f9bcb68e84370667cfe2418450c9fcd112cd0cc9236936e48e3aa35f9dd27ace"
    )


def test_frozen_quadratic_recovers_exact_log_eta_vertex() -> None:
    center = 0.0125
    etas = [center * 2**offset for offset in (-1.0, -0.25, 0.5, 1.25)]
    losses = [1.75 + 0.2 * math.log2(eta / center) ** 2 for eta in etas]
    fit = fit_quadratic(etas, losses)
    assert fit["accepted"] is True
    assert math.isclose(fit["eta_star"], center, rel_tol=0, abs_tol=1e-14)


def test_g12_analyzer_passes_exact_monotone_synthetic_evidence(tmp_path: Path) -> None:
    manifest = manifests.build("static", "d" * 40, None)
    true_d = {2: 4.0, 5: 2.0, 20: 1.1}
    mu0_center = {
        2: 0.07106462666975855,
        5: 0.04341918114042938,
        20: 0.021926218661920484,
    }
    for cell in [cell for cell in manifest["cells"] if cell["program"] == "v12"]:
        center = (
            mu0_center[cell["t"]]
            if cell["arm"] == "mu0"
            else mu0_center[cell["t"]] * 0.1 * true_d[cell["t"]]
        )
        loss = (
            2.0
            + 0.08 * math.log2(cell["eta"] / center) ** 2
            + {981: -0.002, 983: 0.0, 991: 0.002}[cell["seed"]]
        )
        attempt = tmp_path / cell["cell_id"] / "attempt-1"
        report = attempt / "report"
        report.mkdir(parents=True)
        (attempt / "evidence.json").write_text(
            json.dumps(
                {
                    "status": "COMPLETED",
                    "command_hash": cell["command_hash"],
                }
            )
            + "\n"
        )
        (report / "results.jsonl").write_text(json.dumps({"eval_loss": loss}) + "\n")
    readout = analyze_scan(manifest, "v12", tmp_path)
    assert readout["gate"]["verdict"] == "PASS"
    assert readout["gate"]["conditions"]["adjacent_difference_cis_above_zero"] is True


def test_all_local_frozen_artifact_hashes_match_contract_bytes() -> None:
    names = (
        "outer-mup-v11-ratio-transport-prereg.json",
        "outer-mup-v12-heavy-ball-prereg.json",
        "outer-mup-v13-pythia-ultrachat-prereg.json",
        "outer-mup-v7-lean-scope-amendment.json",
    )

    def visit(value: object) -> None:
        if isinstance(value, dict):
            path = value.get("path")
            digest = value.get("sha256")
            if isinstance(path, str) and isinstance(digest, str):
                target = ROOT / path
                if target.is_file():
                    assert file_sha256(target) == digest, path
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for name in names:
        visit(load_contract(name))
