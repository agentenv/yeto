import hashlib
import json
import math
import sys

import pytest

from scripts import analyze_v4b
from scripts import build_v4b_launch_manifest


def synthetic_split_losses():
    v4 = {}
    v4b = {}
    optima = {
        (2560, 0.0): 0.02,
        (2560, 0.9): 0.02 * 0.1 * 2.2,
        (10240, 0.0): 0.01,
        (10240, 0.9): 0.01 * 0.1 * 1.1,
    }
    for coordinate, optimum in optima.items():
        if coordinate in analyze_v4b.EXTENDED_CURVES:
            old_offsets = (0.0, 0.5, 1.0, 1.5)
            new_offsets = (-1.0, -0.5)
        else:
            old_offsets = (-0.75, -0.25, 0.25, 0.75)
            new_offsets = ()
        for target, offsets in ((v4, old_offsets), (v4b, new_offsets)):
            for offset in offsets:
                eta = optimum * 2**offset
                for seed_index, seed in enumerate(analyze_v4b.SEEDS):
                    x = math.log2(eta / optimum)
                    target[(*coordinate, seed, eta)] = 3.0 + x * x + 0.001 * seed_index
    return v4, v4b


def combined_losses():
    v4, v4b = synthetic_split_losses()
    return {**v4, **v4b}


def test_combined_grid_d_definition_and_shared_bootstrap():
    losses = combined_losses()
    fits = {
        (s, mu): analyze_v4b.curve_fit(losses, s, mu)
        for s in analyze_v4b.S_GRID
        for mu in analyze_v4b.MU_GRID
    }
    assert fits[(2560, 0.0)]["point_count"] == 4
    assert all(
        fits[coordinate]["point_count"] == 6
        for coordinate in analyze_v4b.EXTENDED_CURVES
    )
    d5 = analyze_v4b.d_from_fits(fits[(2560, 0.0)], fits[(2560, 0.9)])
    d20 = analyze_v4b.d_from_fits(fits[(10240, 0.0)], fits[(10240, 0.9)])
    assert d5 == pytest.approx(2.2, rel=1e-10)
    assert d20 == pytest.approx(1.1, rel=1e-10)
    bootstrap = analyze_v4b.bootstrap_all(losses)
    assert bootstrap["status"] == "VALID"
    assert bootstrap["valid_replicates"] == 10_000
    assert bootstrap["monotone_gap"]["ci_95"]["low"] > 0
    assert bootstrap["D5"]["ci_95_D"]["low"] == pytest.approx(2.2)


def write_stage(tmp_path, name, losses, schema, stage):
    roots = {node: tmp_path / name / node for node in ("h200-n1", "h200-n2")}
    cells = []
    for index, ((s, mu, seed, eta), loss) in enumerate(sorted(losses.items())):
        node = "h200-n1" if index % 2 == 0 else "h200-n2"
        cell_id = f"{name}-cell-{index:02d}"
        command_hash = f"{name}-hash-{index}"
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
                "t": analyze_v4b.T_BY_S[s],
                "mu": mu,
                "seed": seed,
                "eta": eta,
            }
        )
    manifest = {
        "schema": schema,
        "stage": stage,
        "source": {"git_commit": "test-commit"},
        "cells": cells,
    }
    return roots, manifest


def test_full_cli_combines_v4_and_v4b_evidence(tmp_path, monkeypatch):
    v4_losses, v4b_losses = synthetic_split_losses()
    v4_roots, v4_manifest = write_stage(
        tmp_path,
        "v4",
        v4_losses,
        "yeto_outer_mup_v4_scale_launch_manifest_v1",
        "V4_SCALE",
    )
    v4_path = tmp_path / "v4-manifest.json"
    v4_path.write_text(json.dumps(v4_manifest) + "\n")
    v4b_roots, v4b_manifest = write_stage(
        tmp_path,
        "v4b",
        v4b_losses,
        "yeto_outer_mup_v4b_extension_launch_manifest_v1",
        "V4B_EXTENSION",
    )
    v4b_manifest["base_v4"] = {
        "manifest_sha256": hashlib.sha256(v4_path.read_bytes()).hexdigest()
    }
    v4b_path = tmp_path / "v4b-manifest.json"
    v4b_path.write_text(json.dumps(v4b_manifest) + "\n")
    output = tmp_path / "readout.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_v4b.py",
            "--v4-manifest",
            str(v4_path),
            "--v4b-manifest",
            str(v4b_path),
            "--v4-node-root",
            f"h200-n1={v4_roots['h200-n1']}",
            "--v4-node-root",
            f"h200-n2={v4_roots['h200-n2']}",
            "--v4b-node-root",
            f"h200-n1={v4b_roots['h200-n1']}",
            "--v4b-node-root",
            f"h200-n2={v4b_roots['h200-n2']}",
            "--output",
            str(output),
        ],
    )
    assert analyze_v4b.main() == 0
    readout = json.loads(output.read_text())
    assert readout["gate"]["verdict"] == "PASS"
    assert readout["observed_completed_cells"] == {
        "v4": 48,
        "v4b": 18,
        "combined": 66,
    }
    assert readout["note_line"].startswith("G4B VERDICT: PASS D5=2.200000 [")


def test_builder_assigns_all_long_cells_first_and_leaves_v5_slots():
    contract = {
        "extension_design": {
            "curves": [
                {"s": 2560, "t": 5, "mu": 0.9, "new_etas": [0.01, 0.02]},
                {"s": 10240, "t": 20, "mu": 0.0, "new_etas": [0.03, 0.04]},
                {"s": 10240, "t": 20, "mu": 0.9, "new_etas": [0.001, 0.002]},
            ]
        }
    }
    cells = build_v4b_launch_manifest.build_cells(contract, "test-commit")
    build_v4b_launch_manifest.validate(cells)
    long_assignments = {
        (cell["assignment"]["node"], cell["assignment"]["gpu"])
        for cell in cells
        if cell["s"] == 10240
    }
    assert long_assignments == set(build_v4b_launch_manifest.LONG_SLOTS)
    assert all(cell["assignment"]["gpu"] < 6 for cell in cells)


def test_retry_attempt_cannot_fall_back_to_attempt_one(tmp_path):
    cell = {"cell_id": "retry-cell"}
    for attempt, status in ((1, "COMPLETED"), (2, "INFRA_FAILURE")):
        path = tmp_path / "retry-cell" / f"attempt-{attempt}"
        path.mkdir(parents=True)
        (path / "evidence.json").write_text(json.dumps({"status": status}) + "\n")
    with pytest.raises(analyze_v4b.AnalysisError, match="registered retry exists"):
        analyze_v4b.completed_attempt(cell, tmp_path)
