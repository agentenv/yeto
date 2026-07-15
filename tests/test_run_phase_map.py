from __future__ import annotations

import importlib.util
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HEAD = subprocess.check_output(
    ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
).strip()
SPEC = importlib.util.spec_from_file_location(
    "run_phase_map", ROOT / "scripts" / "run_phase_map.py"
)
rpm = importlib.util.module_from_spec(SPEC)
sys.modules["run_phase_map"] = rpm
assert SPEC.loader is not None
SPEC.loader.exec_module(rpm)

VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "run_phase_map_schema_validator", ROOT / "scripts" / "validate_phase_map.py"
)
validator = importlib.util.module_from_spec(VALIDATOR_SPEC)
sys.modules["run_phase_map_schema_validator"] = validator
assert VALIDATOR_SPEC.loader is not None
VALIDATOR_SPEC.loader.exec_module(validator)


def _args(tmp_path: Path, *extra: str):
    argv = [
        "--study-id",
        "bp-phase-map-p1-r0",
        "--study-phase",
        "p1_development",
        "--run-dir",
        str(tmp_path / "run"),
        "--artifact-uri",
        "gs://bucket/bp-p1-r0",
        "--git-commit",
        HEAD,
        "--image-digest",
        "b" * 64,
        "--image-numeric-id",
        "7290368630472593484",
        "--model-path",
        str(tmp_path / "model"),
        "--model-revision",
        "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
        "--data",
        str(tmp_path / "data.parquet"),
        "--provider-evidence",
        str(tmp_path / "run" / "provider-evidence.json"),
        "--h",
        "16,64,256",
        "--mu",
        "0,.5,.9",
        "--eta",
        ".021875,.04375,.0875,.175",
        "--seed",
        "347",
        "--training-seed",
        "347347",
        "--order-seed",
        "20260714",
        "--resource-class",
        "a2-highgpu-4g",
        *extra,
    ]
    return rpm.build_parser().parse_args(argv)


BOUND_HASHES = {
    "model_hash": "c" * 64,
    "data_hash": "d" * 64,
    "train_rows_hash": "e" * 64,
    "development_eval_rows_hash": "f" * 64,
    "development_eval_packed_hash": "0" * 64,
    "development_eval_example_ids_hash": "1" * 64,
    "development_eval_token_ids_hash": "2" * 64,
    "development_eval_source_indices_hash": "3" * 64,
    "audit_eval_rows_hash": "a" * 64,
    "audit_eval_packed_hash": "9" * 64,
    "audit_eval_example_ids_hash": "b" * 64,
    "audit_eval_token_ids_hash": "c" * 64,
    "audit_eval_source_indices_hash": "d" * 64,
    "audit_access_policy_hash": rpm.sha256_bytes(
        rpm.canonical_json(
            json.loads(
                (ROOT / "experiment-specs" / "best-paper-phase-map-p0-p1-prereg.json").read_text()
            )["confirmation_policy"]
        )
    ),
    "train_pool_source_indices_hash": "4" * 64,
    "train_source_indices_hash": "5" * 64,
}


def _fake_replay(tmp_path: Path, parent: dict, label: str):
    parent_path = tmp_path / f"{label}-parent.json"
    parent_path.write_text(json.dumps(parent, indent=2, sort_keys=True) + "\n")
    replay_report = {
        "schema": "yeto_p0_cpu_replay_v1",
        "status": "PASS",
        "gpu_deleted_before_replay": True,
        "all_steps_replayed": True,
        "phase_map_integrity_status": "VALIDATED",
        "phase_map_validator_report_sha256": "6" * 64,
        "replay_validator_git_commit": HEAD,
        "replay_validator_script_sha256": "7" * 64,
        "replay_validator_git_blob_sha256": "7" * 64,
        "acquisition_manifest_sha256": "8" * 64,
        "deletion_evidence_sha256": "9" * 64,
        "phase_map_manifest_sha256": rpm.sha256_file(parent_path),
        "phase_map_manifest_canonical_sha256": rpm.sha256_bytes(
            rpm.canonical_json(parent)
        ),
        "frozen_tolerance": {
            "param_atol": 2e-6,
            "param_rtol": 2e-6,
            "tape_norm_rtol": 2e-4,
            "replay_dtype": "numpy_little_endian_f32_with_f64_norm_accumulation",
        },
        "cell_count": len(parent["expected_cells"]),
        "cells": [
            {"cell_id": cell["cell_id"], "all_steps_replayed": True}
            for cell in parent["expected_cells"]
        ],
    }
    replay_path = tmp_path / f"{label}-replay.json"
    replay_path.write_text(json.dumps(replay_report, indent=2, sort_keys=True) + "\n")
    return parent_path, replay_path


def _canary_args(tmp_path: Path, stage: str):
    args = _args(tmp_path)
    args.study_id = f"bp-phase-map-{stage}"
    args.study_phase = f"{stage}_canary"
    args.h = [16]
    args.eta = [0.0875]
    args.seed = 337
    args.training_seed = 337337
    args.token_budget = 65_536
    args.capture_every_step = True
    args.parent_manifest = None
    args.expected_parent_manifest_hash = None
    args.parent_replay_report = None
    args.expected_parent_replay_report_hash = None
    args.require_distinct_learner_gpu_uuids = stage == "p0b"
    args.gpu_slots = 4 if stage == "p0b" else 1
    args.resource_class = "a2-highgpu-4g" if stage == "p0b" else "a2-highgpu-1g"
    return args


def _authority_bound_p1(tmp_path: Path):
    p0a = _canary_args(tmp_path, "p0a")
    p0a_plan = rpm.build_plan(p0a)
    p0a_parent = rpm.build_schema_fixture(
        rpm.build_bound_manifest(p0a, p0a_plan, **BOUND_HASHES), p0a_plan
    )
    p0a_path, p0a_replay_path = _fake_replay(tmp_path, p0a_parent, "p0a")

    p0b = _canary_args(tmp_path, "p0b")
    p0b.parent_manifest = p0a_path
    p0b.expected_parent_manifest_hash = rpm.sha256_bytes(
        rpm.canonical_json(p0a_parent)
    )
    p0b.parent_replay_report = p0a_replay_path
    p0b.expected_parent_replay_report_hash = rpm.sha256_file(p0a_replay_path)
    p0b_plan = rpm.build_plan(p0b)
    p0b_parent = rpm.build_schema_fixture(
        rpm.build_bound_manifest(p0b, p0b_plan, **BOUND_HASHES), p0b_plan
    )
    p0b_path, p0b_replay_path = _fake_replay(tmp_path, p0b_parent, "p0b")

    args = _args(tmp_path)
    args.parent_manifest = p0b_path
    args.expected_parent_manifest_hash = rpm.sha256_bytes(
        rpm.canonical_json(p0b_parent)
    )
    args.parent_replay_report = p0b_replay_path
    args.expected_parent_replay_report_hash = rpm.sha256_file(p0b_replay_path)
    plan = rpm.build_plan(args)
    return args, plan, rpm.build_bound_manifest(args, plan, **BOUND_HASHES)


def test_plan_is_exact_blocked_36_cell_grid(tmp_path):
    args = _args(tmp_path)
    plan = rpm.build_plan(args)

    assert len(plan["cells"]) == 36
    assert len(plan["randomization_plan_hash"]) == 64
    by_block = {}
    for cell in plan["cells"]:
        by_block.setdefault(cell["randomization"]["block_id"], []).append(cell)
        command = cell["command"]
        for flag in (
            "--skip-baseline",
            "--skip-untrained-eval",
            "--matrix-merge",
            "--strict-quorum",
            "--barrier-sync",
            "--version-matched-anchor",
        ):
            assert flag in command
        assert "--baseline-loss" not in command
        assert command[command.index("--matrix-merge") + 1] == "rda"
        assert command[command.index("--eval-rows") + 1] == "1024"
    assert len(by_block) == 12
    assert all({cell["mu"] for cell in block} == {0.0, 0.5, 0.9} for block in by_block.values())
    assert all(len({cell["paired_control_id"] for cell in block}) == 1 for block in by_block.values())


def test_plan_is_reproducible_and_order_seed_sensitive(tmp_path):
    first = rpm.build_plan(_args(tmp_path))
    second = rpm.build_plan(_args(tmp_path))
    changed = rpm.build_plan(_args(tmp_path, "--order-seed", "20260715"))

    assert first == second
    assert first["randomization_plan_hash"] != changed["randomization_plan_hash"]
    assert [c["cell_id"] for c in first["cells"]] != [c["cell_id"] for c in changed["cells"]]


def test_p0a_p0b_normalized_workload_ignores_only_hardware_and_stage_paths(
    tmp_path,
):
    p0a = _canary_args(tmp_path / "single", "p0a")
    p0b = _canary_args(tmp_path / "four", "p0b")
    command_a = rpm.compare_command(p0a, h=16, mu=0.5, eta=0.0875)
    command_b = rpm.compare_command(p0b, h=16, mu=0.5, eta=0.0875)
    assert command_a != command_b
    assert rpm.normalized_workload_command(command_a) == (
        rpm.normalized_workload_command(command_b)
    )


def test_plan_requires_live_control_and_exact_work(tmp_path):
    args = _args(tmp_path)
    args.mu = [0.5, 0.9]
    with pytest.raises(rpm.PhaseMapError, match="live mu=0"):
        rpm.build_plan(args)

    args = _args(tmp_path)
    args.token_budget += 1
    with pytest.raises(rpm.PhaseMapError, match="divisible"):
        rpm.build_plan(args)


def test_authority_split_has_disjoint_seed_invariant_dev_and_audit_sets(
    tmp_path, monkeypatch
):
    pytest.importorskip("torch")
    from scripts.compare_diloco import split_data

    rows = [
        {"messages": [{"role": "user", "content": f"row-{index}"}]}
        for index in range(7048)
    ]
    monkeypatch.setattr("yeto.data.load_rows", lambda _path: rows)
    outputs = []
    for seed in (337, 347):
        work = tmp_path / f"seed-{seed}"
        train, dev, count = split_data(
            "unused.parquet",
            work,
            1024,
            5000,
            seed,
            331,
            1024,
        )
        provenance = json.loads((work / "split_provenance.json").read_text())
        outputs.append((train, dev, work / "confirmation-audit.jsonl", provenance))
        assert count == 5000
        assert provenance["source_population_count"] == 7048
    expected = list(range(7048))
    import random

    random.Random(331).shuffle(expected)
    first, second = outputs
    assert first[3]["train_pool_source_indices"] == expected[:5000]
    assert first[3]["eval_source_indices"] == expected[5000:6024]
    assert first[3]["audit_eval_source_indices"] == expected[6024:7048]
    assert first[3]["eval_source_indices"] == second[3]["eval_source_indices"]
    assert first[3]["audit_eval_source_indices"] == second[3]["audit_eval_source_indices"]
    assert first[3]["train_source_indices"] != second[3]["train_source_indices"]
    sets = [
        set(first[3][key])
        for key in (
            "train_pool_source_indices",
            "eval_source_indices",
            "audit_eval_source_indices",
        )
    ]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert hashlib.sha256(first[1].read_bytes()).hexdigest() == hashlib.sha256(
        second[1].read_bytes()
    ).hexdigest()
    assert hashlib.sha256(first[2].read_bytes()).hexdigest() == hashlib.sha256(
        second[2].read_bytes()
    ).hexdigest()

def _cell_for_tape(args):
    args.h = [16]
    args.mu = [0.0]
    args.eta = [0.04375]
    args.token_budget = 32768
    return rpm.build_plan(args)["cells"][0]


def _write_tape(path: Path, cell: dict, *, partial_step: int | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for step in range(1, cell["target_work"]["outer_steps"] + 1):
        responders = []
        for learner in range(4 if step != partial_step else 3):
            responders.append(
                {
                    "id": learner,
                    "base_version": (step - 1) // 4,
                    "c_steps": cell["H"],
                    "c_tokens": cell["H"] * 128,
                    "anchor_base_resolved": True,
                }
            )
        rows.append(
            {
                "step": step,
                "fragment": (step - 1) % 4,
                "responders": responders,
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_tape_validator_proves_exact_full_quorum_work(tmp_path):
    args = _args(tmp_path)
    cell = _cell_for_tape(args)
    tape = tmp_path / "tape.jsonl"
    _write_tape(tape, cell)

    observed = rpm.validate_tape(tape, cell, args)

    assert observed["tokens"] == 32768
    assert observed["microsteps"] == 256
    assert observed["outer_steps"] == 16
    assert observed["per_fragment_outer_steps"] == {0: 4, 1: 4, 2: 4, 3: 4}


def test_tape_validator_rejects_partial_quorum(tmp_path):
    args = _args(tmp_path)
    cell = _cell_for_tape(args)
    tape = tmp_path / "tape.jsonl"
    _write_tape(tape, cell, partial_step=7)

    with pytest.raises(rpm.PhaseMapError, match="full quorum"):
        rpm.validate_tape(tape, cell, args)


def test_layout_validator_requires_identical_rda_layouts(tmp_path):
    fragments = [
        {"id": 0, "merge_mode": "avg", "tensors": []},
        {"id": 1, "merge_mode": "rda", "tensors": []},
        {"id": 2, "merge_mode": "rda", "tensors": []},
        {"id": 3, "merge_mode": "rda", "tensors": []},
    ]
    payload = {"matrix_merge": "rda", "fragment_count": 4, "fragments": fragments}
    for learner in range(4):
        path = tmp_path / "work" / "m4" / f"learner-{learner}" / "resolved-layout.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(payload, sort_keys=True) + "\n")

    digest, modes = rpm.validate_layout(tmp_path)

    assert len(digest) == 64
    assert modes == ["avg", "rda", "rda", "rda"]


def test_retry_metadata_is_fail_closed(tmp_path):
    args = _args(tmp_path)
    args.attempt = 2
    args.print_plan = False
    with pytest.raises(rpm.PhaseMapError, match="retry attempt"):
        # Exercise just the pre-input retry gate by giving a relative run dir.
        rpm.execute(args)


def test_retry_policy_hash_binds_exact_embedded_policy(tmp_path):
    _args_value, _plan, bound = _authority_bound_p1(tmp_path)
    policy = bound["retry_policy"]
    digest = rpm.sha256_bytes(rpm.canonical_json(policy))

    assert policy["loss_blind_only"] is True
    assert "poor_loss" in policy["forbidden_reasons"]
    assert digest == bound["frozen"]["retry_policy_hash"]


def test_runner_generated_schema_fixture_passes_authoritative_validator(tmp_path):
    args, plan, bound = _authority_bound_p1(tmp_path)
    fixture = rpm.build_schema_fixture(bound, plan)
    fixture["status"] = "sealed_results"

    report = validator.validate_and_summarize(
        fixture,
        claim_level="development",
        require_bracketed=True,
        parent_manifest=json.loads(args.parent_manifest.read_text()),
        parent_replay_report=json.loads(args.parent_replay_report.read_text()),
        parent_replay_report_sha256=args.expected_parent_replay_report_hash,
    )

    assert report["valid"] is True
    assert report["integrity_status"] == "VALIDATED"
    assert report["expected_cell_count"] == 36
    assert report["final_cell_count"] == 36
    assert {row["bracket_decision"] for row in report["phase_map"]} == {
        "BRACKETED"
    }


def test_retry_fixture_preserves_completed_peers_and_shared_authorization(tmp_path):
    args, plan, bound = _authority_bound_p1(tmp_path)

    fixture = rpm.build_retry_schema_fixture(bound, plan)
    retry_rows = fixture["results"][-3:]
    block_id = retry_rows[0]["block_id"]
    prior_rows = [
        row
        for row in fixture["results"][:-3]
        if row["block_id"] == block_id
    ]

    assert [row["status"] for row in prior_rows].count("COMPLETED") == 2
    assert [row["status"] for row in prior_rows].count("INFRA_FAILURE") == 1
    assert all(row["status"] == "COMPLETED" for row in retry_rows)
    assert len(
        {
            json.dumps(row["retry_authorization"], sort_keys=True)
            for row in retry_rows
        }
    ) == 1
    assert [row["retry_reason"] for row in retry_rows].count(
        rpm.PEER_BLOCK_RETRY_REASON
    ) == 2
    report = validator.validate_and_summarize(
        fixture,
        claim_level="development",
        require_bracketed=True,
        parent_manifest=json.loads(args.parent_manifest.read_text()),
        parent_replay_report=json.loads(args.parent_replay_report.read_text()),
        parent_replay_report_sha256=args.expected_parent_replay_report_hash,
    )
    assert report["valid"] is True
    assert report["retry_count"] == 3


def _eval_identity(index: int) -> dict:
    digest = f"{index + 1:064x}"
    return {
        "sequence_index": index,
        "sequence_id": digest,
        "input_ids_sha256": digest,
        "labels_sha256": digest,
        "attention_mask_sha256": digest,
        "supervision_weights_sha256": digest,
        "sequence_length": 8,
        "supervised_token_count": 4,
    }


def _eval_validation_fixture(tmp_path: Path, *, nonfinite: bool = False):
    report = tmp_path / "report"
    provenance = report / "eval-provenance"
    losses_dir = report / "per-example-loss"
    provenance.mkdir(parents=True)
    losses_dir.mkdir(parents=True)
    frozen_path = tmp_path / "frozen_sequences.jsonl"
    frozen = [_eval_identity(0), _eval_identity(1)]
    frozen_path.write_text("".join(json.dumps(row) + "\n" for row in frozen))
    loss_rows = []
    for row in frozen:
        item = {key: value for key, value in row.items() if key != "supervised_token_count"}
        item["token_count"] = row["supervised_token_count"]
        item["loss_sum"] = float("nan") if nonfinite and not loss_rows else 8.0
        item["loss_per_token"] = (
            float("nan") if nonfinite and not loss_rows else 2.0
        )
        loss_rows.append(item)
    (losses_dir / "m4.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in loss_rows)
    )
    expected = {
        "eval_file_sha256": "a" * 64,
        "eval_example_ids_hash": "b" * 64,
        "eval_token_ids_hash": "c" * 64,
        "eval_row_count": 2,
        "eval_supervised_token_count": 8,
        "_eval_sequences_path": str(frozen_path),
    }
    (provenance / "eval_provenance.json").write_text(
        json.dumps({key: value for key, value in expected.items() if not key.startswith("_")})
    )
    return report, expected, loss_rows


def test_eval_validation_rejects_permuted_sequence_identity(tmp_path):
    report, expected, rows = _eval_validation_fixture(tmp_path)
    rows.reverse()
    (report / "per-example-loss" / "m4.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    with pytest.raises(rpm.PhaseMapError, match="identity mismatch"):
        rpm.validate_eval(report, 2.0, expected)


def test_eval_validation_retains_nonfinite_scientific_divergence(tmp_path):
    report, expected, _rows = _eval_validation_fixture(tmp_path, nonfinite=True)
    _summary, path = rpm.validate_eval(report, float("nan"), expected)
    assert path.name == "m4.jsonl"


def test_process_exit_is_never_inferred_to_be_retryable_infra(tmp_path):
    attempt = tmp_path / "attempt"
    assert rpm.classify_unmarked_process_exit(attempt).startswith("process_exit_without")
    marker = attempt / "report" / "acquisition-state.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({"phase": "endpoint_started"}))
    assert rpm.classify_unmarked_process_exit(attempt) == (
        "process_exit_after_scientific_endpoint_started"
    )
    assert (
        rpm.classify_unmarked_process_exit(attempt)
        not in rpm.DIRECT_INFRASTRUCTURE_FAILURE_REASONS
    )


def test_semantic_result_validation_failure_is_never_retryable_infra():
    assert rpm.result_validation_failure_is_retryable(FileNotFoundError()) is True
    assert (
        rpm.result_validation_failure_is_retryable(
            rpm.PhaseMapError("well-formed tape has wrong work")
        )
        is False
    )
    assert rpm.result_validation_failure_is_retryable(ValueError("bad arithmetic")) is False
