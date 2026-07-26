import math
import hashlib
import json
import sys

import pytest

from scripts import analyze_v4


def synthetic_losses():
    losses = {}
    optima = {
        (2560, 0.0): 0.05,
        (2560, 0.9): 0.05 * 0.1 * 2.134,
        (10240, 0.0): 0.025,
        (10240, 0.9): 0.025 * 0.1 * 1.123,
    }
    offsets = (-0.75, -0.25, 0.25, 0.75)
    for (s, mu), optimum in optima.items():
        for eta in [optimum * 2**offset for offset in offsets]:
            for seed_index, seed in enumerate(analyze_v4.SEEDS):
                x = math.log2(eta / optimum)
                losses[(s, mu, seed, eta)] = 3.0 + x * x + 0.001 * seed_index
    return losses


def test_registered_d_definition_and_monotone_bootstrap():
    losses = synthetic_losses()
    fits = {
        (s, mu): analyze_v4.curve_fit(losses, s, mu)
        for s in analyze_v4.S_GRID
        for mu in analyze_v4.MU_GRID
    }
    d5 = analyze_v4.d_from_fits(fits[(2560, 0.0)], fits[(2560, 0.9)])
    d20 = analyze_v4.d_from_fits(fits[(10240, 0.0)], fits[(10240, 0.9)])
    assert d5 == pytest.approx(2.134, rel=1e-10)
    assert d20 == pytest.approx(1.123, rel=1e-10)
    bootstrap = analyze_v4.bootstrap_gap(losses)
    assert bootstrap["status"] == "VALID"
    assert bootstrap["invalid_unbracketed_replicates"] == 0
    assert bootstrap["ci_95"]["low"] > 0


def test_quadratic_rejects_boundary_vertex():
    etas = [1.0, 2.0, 4.0, 8.0]
    losses = [9.0, 4.0, 1.0, 0.0]
    fit = analyze_v4.fit_quadratic(etas, losses)
    assert not fit["interior"]
    assert fit["eta_star"] is None


def test_full_cli_evidence_path_and_gate(tmp_path, monkeypatch):
    losses = synthetic_losses()
    roots = {node: tmp_path / node for node in ("h200-n1", "h200-n2")}
    cells = []
    for index, ((s, mu, seed, eta), loss) in enumerate(sorted(losses.items())):
        node = "h200-n1" if index % 2 == 0 else "h200-n2"
        cell_id = f"cell-{index:02d}"
        command_hash = f"hash-{index}"
        attempt = roots[node] / cell_id / "attempt-1"
        results_path = attempt / "report" / "results.jsonl"
        results_path.parent.mkdir(parents=True)
        results_path.write_text(json.dumps({"eval_loss": loss}) + "\n")
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
                "registered_retry_commands": [{"command_hash": f"retry-{index}"}],
                "s": s,
                "t": analyze_v4.T_BY_S[s],
                "mu": mu,
                "seed": seed,
                "eta": eta,
            }
        )
    manifest = {
        "schema": "yeto_outer_mup_v4_scale_launch_manifest_v1",
        "stage": "V4_SCALE",
        "source": {"git_commit": "test-commit"},
        "cells": cells,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest) + "\n")
    output = tmp_path / "readout.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_v4.py",
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
    assert analyze_v4.main() == 0
    readout = json.loads(output.read_text())
    assert readout["gate"]["verdict"] == "PASS"
    assert readout["observed_completed_cells"] == 48


def test_retry_attempt_cannot_fall_back_to_attempt_one(tmp_path):
    cell = {"cell_id": "retry-cell"}
    for attempt, status in ((1, "COMPLETED"), (2, "INFRA_FAILURE")):
        path = tmp_path / "retry-cell" / f"attempt-{attempt}"
        path.mkdir(parents=True)
        (path / "evidence.json").write_text(json.dumps({"status": status}) + "\n")
    with pytest.raises(analyze_v4.AnalysisError, match="registered retry exists"):
        analyze_v4.completed_attempt(cell, tmp_path)
