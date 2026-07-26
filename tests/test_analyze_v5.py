import hashlib
import json
import math
import sys

import pytest

from scripts import analyze_v5, build_v5_launch_manifest


def synthetic_losses():
    losses = {}
    optima = {"a": 0.56, "b": 0.11, "c": 0.55}
    bases = {"a": 3.0, "b": 2.98, "c": 3.005}
    for condition in analyze_v5.CONDITIONS:
        for seed_index, seed in enumerate(analyze_v5.SEEDS):
            for eta in analyze_v5.ETA_GRIDS[condition]:
                x = math.log2(eta / optima[condition])
                losses[(condition, seed, eta)] = (
                    bases[condition] + 0.15 * x * x + 0.001 * seed_index
                )
    return losses


def test_code_true_grid_pairing_and_interior_fit():
    for eta_b, eta_c in zip(
        analyze_v5.ETA_GRIDS["b"], analyze_v5.ETA_GRIDS["c"]
    ):
        assert eta_c == pytest.approx(
            eta_b * analyze_v5.CODE_TRUE_MATCH_FACTOR, abs=1e-15
        )
    losses = synthetic_losses()
    fits = {
        condition: analyze_v5.fit_condition(losses, condition)
        for condition in analyze_v5.CONDITIONS
    }
    assert all(fit["interior"] for fit in fits.values())
    assert fits["a"]["eta_star"] == pytest.approx(0.56)
    assert fits["b"]["eta_star"] == pytest.approx(0.11)
    assert fits["c"]["eta_star"] == pytest.approx(0.55)


def test_quadratic_rejects_boundary_vertex():
    etas = list(analyze_v5.ETA_GRIDS["a"])
    boundary = math.log2(etas[0])
    losses = [(math.log2(eta) - boundary) ** 2 for eta in etas]
    fit = analyze_v5.fit_quadratic(etas, losses)
    assert not fit["interior"]
    assert fit["eta_star"] is None


def test_paired_bootstrap_recovers_tuned_differences():
    bootstrap = analyze_v5.paired_bootstrap(synthetic_losses())
    assert bootstrap["status"] == "VALID"
    assert bootstrap["valid_replicates"] == 10_000
    b_ci = bootstrap["deltas"]["b_minus_a"]["ci_95"]
    c_ci = bootstrap["deltas"]["c_minus_a"]["ci_95"]
    assert b_ci["high"] < 0
    assert c_ci["low"] > 0


def test_full_cli_evidence_path_and_g5_verdict(tmp_path, monkeypatch):
    losses = synthetic_losses()
    roots = {node: tmp_path / node for node in ("h200-n1", "h200-n2")}
    cells = []
    index = 0
    for condition in analyze_v5.CONDITIONS:
        for eta_index, eta in enumerate(analyze_v5.ETA_GRIDS[condition]):
            for seed in analyze_v5.SEEDS:
                node = "h200-n1" if index % 2 == 0 else "h200-n2"
                cell_id = f"cell-{index:02d}"
                command_hash = f"hash-{index}"
                attempt = roots[node] / cell_id / "attempt-1"
                results_path = attempt / "report" / "results.jsonl"
                results_path.parent.mkdir(parents=True)
                results_path.write_text(
                    json.dumps({"eval_loss": losses[(condition, seed, eta)]}) + "\n"
                )
                results_sha = hashlib.sha256(results_path.read_bytes()).hexdigest()
                evidence = {
                    "status": "COMPLETED",
                    "cell_id": cell_id,
                    "command_hash": command_hash,
                    "seed": seed,
                    "observed_artifacts": {"results": {"sha256": results_sha}},
                }
                (attempt / "evidence.json").write_text(json.dumps(evidence) + "\n")
                cells.append(
                    {
                        "cell_id": cell_id,
                        "assignment": {"node": node, "gpu": index % 8},
                        "command_hash": command_hash,
                        "registered_retry_commands": [
                            {"command_hash": f"retry-{index}"}
                        ],
                        "condition": condition,
                        "eta": eta,
                        "eta_index": eta_index,
                        "seed": seed,
                    }
                )
                index += 1
    manifest = {
        "schema": "yeto_outer_mup_v5_snoo_launch_manifest_v1",
        "stage": "V5_SNOO_INTERIOR",
        "source": {"git_commit": "test-commit"},
        "contract": {"analyzer_sha256": "test"},
        "cells": cells,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest) + "\n")
    output = tmp_path / "readout.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_v5.py",
            "--manifest",
            str(manifest_path),
            "--node-root",
            f"h200-n1={roots['h200-n1']}",
            "--node-root",
            f"h200-n2={roots['h200-n2']}",
            "--output",
            str(output),
        ],
    )
    assert analyze_v5.main() == 0
    readout = json.loads(output.read_text())
    assert readout["G5"]["verdict"] == "SNOO_HELPS"
    assert readout["G5"]["requirements"]["all_three_pooled_optima_interior"]
    assert readout["observed_completed_cells"] == 90
    assert readout["note_line"].startswith("G5 VERDICT: SNOO_HELPS b-a=")
    assert " c-a=" in readout["note_line"]


def test_retry_attempt_cannot_fall_back_to_attempt_one(tmp_path):
    cell = {"cell_id": "retry-cell"}
    for attempt, status in ((1, "COMPLETED"), (2, "INFRA_FAILURE")):
        path = tmp_path / "retry-cell" / f"attempt-{attempt}"
        path.mkdir(parents=True)
        (path / "evidence.json").write_text(json.dumps({"status": status}) + "\n")
    with pytest.raises(analyze_v5.AnalysisError, match="registered retry exists"):
        analyze_v5.completed_attempt(cell, tmp_path)


def test_launch_manifest_is_balanced_and_hash_bound():
    cells = build_v5_launch_manifest.build_cells("test-commit")
    build_v5_launch_manifest.validate(cells)
    assert len(cells) == 90
    assert sum(cell["assignment"]["node"] == "h200-n1" for cell in cells) == 45
    assert sum(cell["assignment"]["node"] == "h200-n2" for cell in cells) == 45
    assert all(
        build_v5_launch_manifest.canonical_sha256(cell["command"])
        == cell["command_hash"]
        for cell in cells
    )
