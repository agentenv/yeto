"""Tests for the RL elastic resource benchmark suite (no GPU, no runtime)."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from yeto.rl.elastic_benchmark import capabilities as caps
from yeto.rl.elastic_benchmark import evidence, legacy, results, work
from yeto.rl.elastic_benchmark.cli import main
from yeto.rl.elastic_benchmark.manifest import (
    ManifestError,
    dump_manifest,
    example_manifest,
    load_manifest,
    manifest_hash,
    validate_manifest,
)
from yeto.rl.elastic_benchmark.plan import build_plan

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "benchmark_rl_elastic.py"


def _attested(**overrides) -> caps.Attestation:
    payload = {
        "runtime_fingerprint": "sha256:runtime",
        "execution_modes": ["partitioned-serial"],
        "partitioned_driver": True,
        "certified_edges": [{"source": "P422", "target": "P44", "kind": "rollout-only"}],
    }
    payload.update(overrides)
    return caps.attestation_from_dict(payload)


# --- 1.1 manifest ---------------------------------------------------------


def test_calibration_manifest_round_trips_with_stable_hash(tmp_path):
    manifest = example_manifest()
    digest = dump_manifest(manifest, tmp_path / "study.json")
    loaded = load_manifest(tmp_path / "study.json")
    assert loaded["study_hash"] == digest == manifest_hash(loaded)
    reordered = json.loads(json.dumps(manifest, sort_keys=False))
    assert manifest_hash(reordered) == digest


def test_formal_manifest_rejects_mutable_revision_and_missing_quality_rules():
    formal = example_manifest("formal")
    validate_manifest(formal)
    mutable = copy.deepcopy(formal)
    mutable["identity"]["model"]["revision"] = "main"
    with pytest.raises(ManifestError, match="immutable identity.model.revision"):
        validate_manifest(mutable)
    no_rules = copy.deepcopy(formal)
    no_rules["evaluation"]["metrics"] = []
    with pytest.raises(ManifestError, match="predeclared bounds"):
        validate_manifest(no_rules)
    unresolved = copy.deepcopy(formal)
    unresolved["resources"]["pool_id"] = "unresolved"
    with pytest.raises(ManifestError, match="unresolved fields: resources.pool_id"):
        validate_manifest(unresolved)


def test_contradictory_budget_is_rejected():
    manifest = example_manifest()
    manifest["work"]["update_budget"] = 11
    with pytest.raises(ManifestError, match="contradicts the phase sum"):
        validate_manifest(manifest)
    manifest = example_manifest()
    manifest["timing"]["measured_updates"] = 99
    with pytest.raises(ManifestError, match="exceed work.update_budget"):
        validate_manifest(manifest)
    manifest = example_manifest()
    manifest["work"]["samples_per_group"] = 5
    with pytest.raises(ManifestError, match="must equal profile.global_batch"):
        validate_manifest(manifest)


def test_edited_manifest_is_rejected_by_recorded_hash(tmp_path):
    path = tmp_path / "study.json"
    dump_manifest(example_manifest(), path)
    payload = json.loads(path.read_text())
    payload["matrix"]["seeds"] = [1]
    path.write_text(json.dumps(payload))
    with pytest.raises(ManifestError, match="recorded study_hash"):
        load_manifest(path)


# --- 1.2 configs, edges, pool, attestation ----------------------------------


def test_config_rejections_keep_reasons_instead_of_substituting():
    manifest = example_manifest()
    manifest["resources"]["configs"]["P71"] = {"trainer": 7, "rollout": 1}
    manifest["resources"]["configs"]["P80"] = {"trainer": 8, "rollout": 0}
    configs = caps.parse_configs(manifest["resources"])
    table = {row["config"]: row for row in caps.config_table(configs, profile=manifest["profile"], pool_size=8)}
    assert table["P62"]["legal"] and table["P62"]["gradient_accumulation"] == 8
    assert "not divisible" in table["P71"]["reason"]
    assert "rollout" in table["P80"]["reason"]
    dense = dict(manifest["profile"], parameter_mode="full")
    assert "DP=1" in caps.config_rejection(configs["P62"], profile=dense, pool_size=8)
    assert caps.config_rejection(configs["P62"], profile=manifest["profile"], pool_size=4).startswith("uses 8")


def test_duplicate_gpu_uuid_and_wrong_edge_shape_are_rejected():
    resources = example_manifest("formal")["resources"]
    resources["gpus"].append(dict(resources["gpus"][0]))
    with pytest.raises(ManifestError, match="duplicate GPU uuid"):
        caps.validate_pool(resources)
    resources = example_manifest()["resources"]
    resources["edges"].append({"source": "P62", "target": "P26", "kind": "rollout-only"})
    with pytest.raises(ManifestError, match="changes trainer count"):
        caps.validate_edges(resources, caps.parse_configs(resources))


def test_uncertified_edges_block_without_downgrade():
    manifest = example_manifest()
    manifest["matrix"]["arms"].append(
        {"name": "b1", "kind": "scheduled-rebuild", "config": "P422", "switch_plan": [{"at_update": 3, "target": "P44"}]}
    )
    plan = build_plan(manifest, _attested(), study_hash="h")
    status = {(i.key.arm, i.key.config): (i.status, i.reason) for i in plan.items}
    assert status[("b1", None)] == ("supported", None)
    assert status[("rebuild", None)][0] == "blocked_dependency"
    assert "role-transfer" in status[("rebuild", None)][1]
    assert status[("auto", None)][0] == "blocked_dependency"
    assert status[("legacy", None)] == ("supported", None)
    assert status[("sweep", "P44")] == ("supported", None)
    none = build_plan(manifest, caps.Attestation.none(), study_hash="h")
    assert all(i.status == "blocked_dependency" for i in none.items if i.key.arm != "legacy")


# --- 1.3 evidence and resume --------------------------------------------------


def _completed_attempt(study_dir: Path, key: evidence.MatrixKey, study_hash: str, attempt: int = 1) -> Path:
    attempt_dir = study_dir / key.relative_dir(attempt)
    (attempt_dir / "rollout").mkdir(parents=True)
    (attempt_dir / "rollout" / "1.jsonl").write_text('{"reward":1}\n')
    (attempt_dir / "metrics.jsonl").write_text('{"step":1}\n')
    evidence.build_evidence_index(attempt_dir, study_hash=study_hash, key=key, attempt=attempt)
    evidence.write_result(attempt_dir, results.make_result(execution="completed", correctness="passed"))
    return attempt_dir


def test_tampered_or_missing_evidence_cannot_be_reused(tmp_path):
    key = evidence.MatrixKey("default", "stable", 17)
    attempt_dir = _completed_attempt(tmp_path, key, "h1")
    assert evidence.reusable_completion(tmp_path, study_hash="h1", key=key)["execution"] == "completed"
    (attempt_dir / "metrics.jsonl").write_text('{"step":2}\n')
    with pytest.raises(evidence.EvidenceError, match="modified: metrics.jsonl"):
        evidence.reusable_completion(tmp_path, study_hash="h1", key=key)
    (attempt_dir / "metrics.jsonl").unlink()
    with pytest.raises(evidence.EvidenceError, match="missing: metrics.jsonl"):
        evidence.reusable_completion(tmp_path, study_hash="h1", key=key)


def test_changed_study_identity_and_unindexed_files_are_rejected(tmp_path):
    key = evidence.MatrixKey("default", "stable", 17)
    attempt_dir = _completed_attempt(tmp_path, key, "h1")
    with pytest.raises(evidence.EvidenceError, match="different study"):
        evidence.reusable_completion(tmp_path, study_hash="h2", key=key)
    (attempt_dir / "extra.log").write_text("late")
    with pytest.raises(evidence.EvidenceError, match="unindexed"):
        evidence.reusable_completion(tmp_path, study_hash="h1", key=key)


def test_failed_attempts_are_kept_and_results_never_overwritten(tmp_path):
    key = evidence.MatrixKey("default", "stable", 17)
    failed_dir = tmp_path / key.relative_dir(1)
    failed_dir.mkdir(parents=True)
    evidence.write_result(failed_dir, results.make_result(execution="failed", reason="oom"))
    with pytest.raises(evidence.EvidenceError, match="refusing to overwrite"):
        evidence.write_result(failed_dir, results.make_result(execution="completed"))
    assert evidence.next_attempt(tmp_path, key) == 2
    assert evidence.reusable_completion(tmp_path, study_hash="h1", key=key) is None
    _completed_attempt(tmp_path, key, "h1", attempt=2)
    collected = evidence.collect_results(tmp_path, [key])[key]
    assert [r["execution"] for r in collected] == ["failed", "completed"]


# --- 1.4 CLI / dry-run ----------------------------------------------------------


def test_dry_run_plan_lists_budget_and_never_imports_runtimes(tmp_path):
    study = tmp_path / "study.json"
    assert main(["example", "--output", str(study)]) == 0
    caps_path = tmp_path / "caps.json"
    caps_path.write_text(json.dumps({"runtime_fingerprint": "x", "execution_modes": ["partitioned-serial"], "partitioned_driver": True}))
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "plan", "--study", str(study), "--capabilities", str(caps_path), "--json"],
        capture_output=True, text=True, check=False, cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["counts"] == {"supported": 30, "unsupported": 0, "blocked_dependency": 18}
    assert payload["budget"]["updates"] == 360
    assert any(i["status"] == "blocked_dependency" and i["arm"] == "auto" for i in payload["items"])
    forbidden = subprocess.run(
        [sys.executable, "-c", "import sys;from yeto.rl.elastic_benchmark import cli,plan,results;"
         "bad=[m for m in ('torch','ray','miles','sky','transformers') if m in sys.modules];print(bad)"],
        capture_output=True, text=True, check=True, cwd=REPO_ROOT,
    )
    assert forbidden.stdout.strip() == "[]"
    assert not list(tmp_path.glob("runs"))


def test_cli_reports_manifest_errors_as_exit_2(tmp_path, capsys):
    study = tmp_path / "study.json"
    manifest = example_manifest()
    manifest["matrix"]["scenarios"] = ["bogus"]
    study.write_text(json.dumps(manifest))
    assert main(["validate", "--study", str(study)]) == 2
    assert "matrix.scenarios" in capsys.readouterr().err


# --- 1.5 layered results ---------------------------------------------------------


def test_summary_is_incomplete_until_every_required_item_has_verified_evidence(tmp_path):
    manifest = example_manifest()
    manifest["matrix"]["arms"] = [{"name": "default", "kind": "target-fixed-default", "config": "P44"}]
    manifest["matrix"]["scenarios"] = ["stable"]
    manifest["matrix"]["seeds"] = [17, 29]
    plan = build_plan(manifest, _attested(), study_hash="h")
    _completed_attempt(tmp_path, evidence.MatrixKey("default", "stable", 17), "h")
    summary = results.summarize(plan, tmp_path)
    assert summary["status"] == "incomplete"
    assert summary["missing"] == [{"arm": "default", "scenario": "stable", "seed": 29, "config": None}]
    assert summary["benefit"] == "not_demonstrated"
    report = results.render_report(summary)
    assert "no survivor mean" in report
    _completed_attempt(tmp_path, evidence.MatrixKey("default", "stable", 29), "h")
    complete = results.summarize(plan, tmp_path)
    assert complete["status"] == "complete" and complete["benefit"] == "not_checked"


def test_benefit_needs_every_gate_and_negative_results_survive(tmp_path):
    manifest = example_manifest()
    manifest["matrix"]["arms"] = [
        {"name": "b1", "kind": "scheduled-rebuild", "config": "P422", "switch_plan": [{"at_update": 3, "target": "P44"}]}
    ]
    manifest["matrix"]["scenarios"], manifest["matrix"]["seeds"] = ["stable"], [17]
    plan = build_plan(manifest, _attested(), study_hash="h")
    key = evidence.MatrixKey("b1", "stable", 17)
    attempt_dir = tmp_path / key.relative_dir(1)
    attempt_dir.mkdir(parents=True)
    evidence.build_evidence_index(attempt_dir, study_hash="h", key=key, attempt=1)
    evidence.write_result(attempt_dir, results.make_result(execution="completed", correctness="passed", quality="insufficient", benefit="demonstrated"))
    assert results.summarize(plan, tmp_path)["benefit"] == "not_demonstrated"
    with pytest.raises(ValueError, match="only completed attempts"):
        results.make_result(execution="failed", benefit="demonstrated")


# --- 2.1 legacy reuse -------------------------------------------------------------


def test_legacy_pairing_is_reused_and_held_out_never_leaks(tmp_path):
    rows = [{"prompt": f"q{i}", "label": str(i)} for i in range(10)]
    streams = legacy.paired_streams(rows, islands=2, groups=2, rounds=1)
    assert streams.combined_ids == (0, 1, 2, 3)
    info = legacy.materialize_paired_inputs(
        rows, train_ids=[0, 1, 2, 3], held_out_ids=[8, 9], assignment=[[0, 1], [2, 3]], directory=tmp_path
    )
    combined = legacy.read_jsonl(Path(info["combined"]))
    assert [r["metadata"]["benchmark_prompt_id"] for r in combined] == [0, 1, 2, 3]
    assert [r["label"] for r in legacy.read_jsonl(Path(info["eval"]))] == ["8", "9"]
    with pytest.raises(ValueError, match="held-out rows leaked"):
        legacy.materialize_paired_inputs(rows, train_ids=[0], held_out_ids=[1], assignment=[[1]], directory=tmp_path / "x")


# --- 2.2 splits, seeds, schedules -----------------------------------------------------


def test_splits_are_disjoint_deterministic_and_seeds_reconcile():
    sizes = {"train": 6, "calibration": 2, "test": 1, "held_out": 1}
    a, b = work.split_rows(12, sizes, seed=17), work.split_rows(12, sizes, seed=17)
    assert a == b
    parts = [set(getattr(a, n)) for n in work.SPLIT_NAMES]
    assert sum(len(p) for p in parts) == 10 and not (parts[0] & parts[3])
    assert work.split_rows(12, sizes, seed=29) != a
    with pytest.raises(ManifestError, match="only 5 are available"):
        work.split_rows(5, sizes, seed=1)
    seeds = work.group_seeds(study_seed=17, groups=2, samples_per_group=2)
    assert seeds == work.group_seeds(study_seed=17, groups=2, samples_per_group=2)
    assert len({s for g in seeds for s in g}) == 4


def test_phase_schedule_binds_to_logical_updates_not_wall_time():
    phases = example_manifest()["work"]["phases"]
    slots = work.phase_schedule(phases)
    assert [s.update for s in slots] == list(range(1, 13))
    assert [s.phase for s in slots[3:6]] == ["A", "B", "B"]
    pools = {"short": [0, 1, 2], "long": [7, 8]}
    fast = work.assign_groups(slots, pools=pools, groups_per_update=5, seed=17)
    slow = work.assign_groups(slots, pools=pools, groups_per_update=5, seed=17)
    assert fast == slow and all(len(u) == 5 for u in fast)
    assert work.bucket_counts({"short": 0.8, "long": 0.2}, 5) == {"short": 4, "long": 1}
    with pytest.raises(ManifestError, match="shares must sum"):
        work.phase_schedule([{"name": "x", "updates": 1, "mix": {"short": 0.5}}])


# --- 2.3 scenarios ---------------------------------------------------------------------


def test_scenarios_only_shape_the_mix_and_need_calibration_facts():
    calibration = {
        "mixes": {
            "stable": {"short": 0.5, "long": 0.5},
            "short-heavy": {"short": 0.9, "long": 0.1},
            "long-heavy": {"short": 0.1, "long": 0.9},
            "tail": {"short": 0.9, "long": 0.1},
            "tool-wait": {"short": 1.0},
        },
        "payback_updates": 6,
        "environment": {"task_pack": "tp1", "tool_concurrency": 4, "response_contract": "v1"},
    }
    for scenario in work.SPLIT_NAMES and ("stable", "phased", "tail", "tool-wait", "oscillating"):
        phases = work.scenario_phases(scenario, updates=12, calibration=calibration)
        assert sum(p["updates"] for p in phases) == 12
        assert all(set(p) <= {"name", "updates", "mix", "environment"} for p in phases)
    oscillating = work.scenario_phases("oscillating", updates=12, calibration=calibration)
    assert all(p["updates"] < 6 for p in oscillating) and len(oscillating) == 4
    with pytest.raises(ManifestError, match="payback_updates"):
        work.scenario_phases("oscillating", updates=12, calibration={"mixes": calibration["mixes"]})
    with pytest.raises(ManifestError, match="calibration.environment"):
        work.scenario_phases("tool-wait", updates=4, calibration={"mixes": calibration["mixes"]})
    bad_tail = {"mixes": {"tail": {"short": 0.2, "long": 0.8}}}
    with pytest.raises(ManifestError, match="minority of long"):
        work.scenario_phases("tail", updates=4, calibration=bad_tail)


def test_length_diagnostics_report_caps_and_zero_advantage():
    samples = [
        {"group": 0, "reward": 1.0, "response_length": 10, "status": "completed"},
        {"group": 0, "reward": 1.0, "response_length": 64, "status": "truncated"},
        {"group": 1, "reward": 0.0, "response_length": 5, "status": "completed"},
        {"group": 1, "reward": 1.0, "response_length": 7, "status": "completed"},
    ]
    diag = work.length_diagnostics(samples, max_response_len=64)
    assert diag["cap_hit_ratio"] == 0.25 and diag["zero_advantage_ratio"] == 0.5
    assert diag["max_response_length"] == 64
