import hashlib
import json
import math
import sys

import pytest

from scripts import analyze_v5b, build_v5b_launch_manifest


def synthetic_campaign_losses():
    campaign_losses = {"v5": {}, "v5b": {}}
    optima = {"a": 2.0**-4.0, "b": 2.0**-8.5, "c": 2.0**-4.2}
    bases = {"a": 2.4, "b": 2.37, "c": 2.405}
    grids = {"v5": analyze_v5b.V5_ETA_GRIDS, "v5b": analyze_v5b.V5B_ETA_GRIDS}
    for campaign, campaign_grids in grids.items():
        for condition in analyze_v5b.CONDITIONS:
            for seed_index, seed in enumerate(analyze_v5b.SEEDS):
                for eta in campaign_grids[condition]:
                    x = math.log2(eta / optima[condition])
                    campaign_losses[campaign][(condition, seed, eta)] = (
                        bases[condition] + 0.02 * x * x + 0.001 * seed_index
                    )
    return campaign_losses


def test_registered_power_of_two_grids_and_disclosed_control_shift():
    assert [math.log2(value) for value in analyze_v5b.V5B_ETA_GRIDS["a"]] == [
        -9,
        -8,
        -7,
        -6,
        -5,
    ]
    assert [math.log2(value) for value in analyze_v5b.V5B_ETA_GRIDS["b"]] == [
        -11,
        -10,
        -9,
        -8,
        -7,
    ]
    for eta_b, eta_c in zip(
        analyze_v5b.V5B_ETA_GRIDS["b"], analyze_v5b.V5B_ETA_GRIDS["c"]
    ):
        assert eta_c == eta_b * 4.0
        assert eta_c != eta_b * analyze_v5b.CODE_TRUE_MATCH_FACTOR


def test_combined_eleven_point_fit_recovers_interior_optima():
    losses = synthetic_campaign_losses()
    expected = {"a": 2.0**-4.0, "b": 2.0**-8.5, "c": 2.0**-4.2}
    for condition in analyze_v5b.CONDITIONS:
        fit = analyze_v5b.fit_condition(losses, condition)
        assert fit["point_count"] == 11
        assert fit["status"] == "INTERIOR"
        assert fit["eta_star"] == pytest.approx(expected[condition])


def test_combined_paired_bootstrap_recovers_tuned_differences():
    bootstrap = analyze_v5b.paired_bootstrap(synthetic_campaign_losses())
    assert bootstrap["status"] == "VALID"
    assert bootstrap["valid_replicates"] == 10_000
    assert bootstrap["deltas"]["b_minus_a"]["ci_95"]["high"] < 0
    assert bootstrap["deltas"]["c_minus_a"]["ci_95"]["low"] > 0


def test_quadratic_rejects_boundary_vertex():
    etas = list(analyze_v5b.COMBINED_ETA_GRIDS["a"])
    boundary = math.log2(etas[0])
    losses = [(math.log2(eta) - boundary) ** 2 for eta in etas]
    fit = analyze_v5b.fit_quadratic(etas, losses)
    assert not fit["interior"]
    assert fit["tuned_loss"] is None


def _materialize_campaign(tmp_path, campaign, grids, losses):
    roots = {node: tmp_path / campaign / node for node in ("h200-n1", "h200-n2")}
    cells = []
    index = 0
    for condition in analyze_v5b.CONDITIONS:
        for eta_index, eta in enumerate(grids[condition]):
            for seed in analyze_v5b.SEEDS:
                node = "h200-n1" if index % 2 == 0 else "h200-n2"
                cell_id = f"{campaign}-cell-{index:03d}"
                command_hash = f"{campaign}-hash-{index}"
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
                    "attempt_number": 1,
                    "observed_artifacts": {"results": {"sha256": results_sha}},
                }
                (attempt / "evidence.json").write_text(json.dumps(evidence) + "\n")
                cells.append(
                    {
                        "cell_id": cell_id,
                        "assignment": {"node": node, "gpu": 6 + index % 2},
                        "command_hash": command_hash,
                        "registered_retry_commands": [
                            {"command_hash": f"{campaign}-retry-{index}"}
                        ],
                        "condition": condition,
                        "eta": eta,
                        "eta_index": eta_index,
                        "seed": seed,
                    }
                )
                index += 1
    if campaign == "v5":
        schema = "yeto_outer_mup_v5_snoo_launch_manifest_v1"
        stage = "V5_SNOO_INTERIOR"
    else:
        schema = "yeto_outer_mup_v5b_snoo_regrid_launch_manifest_v1"
        stage = "V5B_SNOO_REGRID"
    manifest = {
        "schema": schema,
        "stage": stage,
        "source": {"git_commit": "test-commit"},
        "contract": {"analyzer_sha256": "test"},
        "cells": cells,
    }
    path = tmp_path / f"{campaign}-manifest.json"
    path.write_text(json.dumps(manifest) + "\n")
    return path, roots


def test_full_cli_combines_both_campaigns_and_emits_g5b(tmp_path, monkeypatch):
    losses = synthetic_campaign_losses()
    v5_manifest, v5_roots = _materialize_campaign(
        tmp_path, "v5", analyze_v5b.V5_ETA_GRIDS, losses["v5"]
    )
    v5b_manifest, v5b_roots = _materialize_campaign(
        tmp_path, "v5b", analyze_v5b.V5B_ETA_GRIDS, losses["v5b"]
    )
    monkeypatch.setattr(
        analyze_v5b,
        "V5_LAUNCH_MANIFEST_SHA256",
        hashlib.sha256(v5_manifest.read_bytes()).hexdigest(),
    )
    output = tmp_path / "g5b.json"
    argv = [
        "analyze_v5b.py",
        "--v5-manifest",
        str(v5_manifest),
        "--v5b-manifest",
        str(v5b_manifest),
    ]
    for node, root in v5_roots.items():
        argv.extend(["--v5-node-root", f"{node}={root}"])
    for node, root in v5b_roots.items():
        argv.extend(["--v5b-node-root", f"{node}={root}"])
    argv.extend(["--output", str(output)])
    monkeypatch.setattr(sys, "argv", argv)
    assert analyze_v5b.main() == 0
    readout = json.loads(output.read_text())
    assert readout["status"] == "COMPLETE"
    assert readout["G5B"]["verdict"] == "SNOO_HELPS"
    assert readout["observed_completed_cells"] == {
        "v5": 90,
        "v5b": 75,
        "combined": 165,
    }
    assert readout["note_line"].startswith("G5B VERDICT: SNOO_HELPS b-a=")


def test_retry_attempt_cannot_fall_back_to_attempt_one(tmp_path):
    cell = {"cell_id": "retry-cell"}
    for attempt, status in ((1, "COMPLETED"), (2, "INFRA_FAILURE")):
        path = tmp_path / "retry-cell" / f"attempt-{attempt}"
        path.mkdir(parents=True)
        (path / "evidence.json").write_text(json.dumps({"status": status}) + "\n")
    with pytest.raises(analyze_v5b.AnalysisError, match="registered retry exists"):
        analyze_v5b.completed_attempt(cell, tmp_path)


def test_launch_manifest_uses_only_four_free_slots_and_is_hash_bound():
    cells = build_v5b_launch_manifest.build_cells("test-commit")
    build_v5b_launch_manifest.validate(cells)
    assert len(cells) == 75
    assignments = {
        (cell["assignment"]["node"], cell["assignment"]["gpu"]) for cell in cells
    }
    assert assignments == set(build_v5b_launch_manifest.SLOTS)
    loads = [
        sum(cell["assignment"] == {"node": node, "gpu": gpu} for cell in cells)
        for node, gpu in build_v5b_launch_manifest.SLOTS
    ]
    assert sorted(loads) == [18, 19, 19, 19]
    assert all(
        build_v5b_launch_manifest.canonical_sha256(cell["command"])
        == cell["command_hash"]
        for cell in cells
    )
