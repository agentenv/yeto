from __future__ import annotations

import importlib.util
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_p0b_allows_only_the_adopted_fixed_production_source_rebind(tmp_path):
    p0a = _canary_args(tmp_path, "p0a")
    p0a.git_commit = rpm.P0A_SOURCE_REBIND_FROM_COMMIT
    p0a_plan = rpm.build_plan(p0a)
    parent = rpm.build_schema_fixture(
        rpm.build_bound_manifest(p0a, p0a_plan, **BOUND_HASHES), p0a_plan
    )
    parent_path, replay_path = _fake_replay(tmp_path, parent, "p0a-rebind")
    replay = json.loads(replay_path.read_text())
    replay["replay_validator_git_commit"] = HEAD
    replay["replay_source_rebind_from_git_commit"] = rpm.P0A_SOURCE_REBIND_FROM_COMMIT
    replay["replay_source_rebind_amendment_path"] = (
        rpm.ADOPTED_PARALLEL_AMENDMENT_PATH.as_posix()
    )
    replay["replay_source_rebind_amendment_sha256"] = (
        rpm.ADOPTED_PARALLEL_AMENDMENT_SHA256
    )
    replay_path.write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n")

    p0b = _canary_args(tmp_path, "p0b")
    p0b.git_commit = HEAD
    p0b.parent_manifest = parent_path
    p0b.expected_parent_manifest_hash = rpm.sha256_bytes(rpm.canonical_json(parent))
    p0b.parent_replay_report = replay_path
    p0b.expected_parent_replay_report_hash = rpm.sha256_file(replay_path)
    plan = rpm.build_plan(p0b)
    bound = rpm.build_bound_manifest(p0b, plan, **BOUND_HASHES)

    assert bound["frozen"]["git_commit"] == HEAD
    assert parent["frozen"]["git_commit"] == rpm.P0A_SOURCE_REBIND_FROM_COMMIT
    assert [
        cell["normalized_workload_command_hash"] for cell in bound["expected_cells"]
    ] == [
        cell["normalized_workload_command_hash"] for cell in parent["expected_cells"]
    ]
    report = validator.validate_and_summarize(
        bound,
        claim_level="integrity",
        parent_manifest=parent,
        parent_replay_report=replay,
        parent_replay_report_sha256=p0b.expected_parent_replay_report_hash,
    )
    assert report["integrity_status"] == "BOUND_LAUNCH_AUTHORITY_VALIDATED"


def test_initial_p1_accepts_only_erratum_bound_p0b_replay_rebind(
    tmp_path, monkeypatch
):
    args, _plan, _bound = _authority_bound_p1(tmp_path)
    parent = json.loads(args.parent_manifest.read_text())
    parent["frozen"]["git_commit"] = rpm.P0B_REPLAY_SOURCE_REBIND_FROM_COMMIT
    for row in parent["results"]:
        row["git_commit"] = rpm.P0B_REPLAY_SOURCE_REBIND_FROM_COMMIT
    args.parent_manifest.write_text(json.dumps(parent, indent=2, sort_keys=True) + "\n")

    replay = json.loads(args.parent_replay_report.read_text())
    replay["phase_map_manifest_sha256"] = rpm.sha256_file(args.parent_manifest)
    replay["phase_map_manifest_canonical_sha256"] = rpm.sha256_bytes(
        rpm.canonical_json(parent)
    )
    replay["replay_validator_git_commit"] = HEAD
    replay["replay_source_rebind_from_git_commit"] = (
        rpm.P0B_REPLAY_SOURCE_REBIND_FROM_COMMIT
    )
    replay["replay_source_rebind_erratum_path"] = (
        rpm.P0B_REPLAY_ERRATUM_PATH.as_posix()
    )
    replay["replay_source_rebind_erratum_sha256"] = rpm.P0B_REPLAY_ERRATUM_SHA256
    args.parent_replay_report.write_text(
        json.dumps(replay, indent=2, sort_keys=True) + "\n"
    )
    args.expected_parent_manifest_hash = rpm.sha256_bytes(rpm.canonical_json(parent))
    args.expected_parent_replay_report_hash = rpm.sha256_file(
        args.parent_replay_report
    )
    monkeypatch.setattr(
        rpm,
        "authorize_p0b_replay_source_rebind",
        lambda observed_parent, candidate: (
            observed_parent == parent and candidate == HEAD
        ),
    )
    template = rpm.verify_authoritative_prereg(args)

    observed_parent, observed_replay = rpm.validate_parent_and_replay(
        args, template, "initial_bound_p1_r0"
    )

    assert observed_parent == parent
    assert observed_replay == replay

    replay.pop("replay_source_rebind_erratum_sha256")
    args.parent_replay_report.write_text(
        json.dumps(replay, indent=2, sort_keys=True) + "\n"
    )
    args.expected_parent_replay_report_hash = rpm.sha256_file(
        args.parent_replay_report
    )
    with pytest.raises(rpm.PhaseMapError, match="source-rebind attestation"):
        rpm.validate_parent_and_replay(args, template, "initial_bound_p1_r0")


def test_adopted_legacy_p0a_work_is_accepted_only_with_full_replay_attestation(
    tmp_path,
):
    args = _canary_args(tmp_path, "p0a")
    args.git_commit = rpm.P0A_SOURCE_REBIND_FROM_COMMIT
    plan = rpm.build_plan(args)
    parent = rpm.build_schema_fixture(
        rpm.build_bound_manifest(args, plan, **BOUND_HASHES), plan
    )
    parent = json.loads(json.dumps(parent))
    for cell in parent["expected_cells"]:
        cell.pop("expected_learner_count")
        cell.pop("expected_learner_steps")
    for row in parent["results"]:
        row.pop("exit_statuses")
        row["observed_work"].pop("learner_step_counts")
        row["hardware"].update(
            {
                "barrier_trace_validated": True,
                "barrier_trace_commit_count": 32,
                "barrier_trace_inner_steps_per_learner": 128,
                "barrier_trace_learner_count": 4,
            }
        )

    replay = {
        "cells": [
            {
                "cell_id": row["cell_id"],
                "final_attempt": 1,
                "replayed_attempt_count": 1,
                "all_steps_replayed": True,
                "replayed_attempts": [
                    {
                        "attempt": 1,
                        "all_steps_replayed": True,
                        "commit_count": 32,
                        "barrier_trace_commit_count": 32,
                        "barrier_trace_inner_steps_per_learner": 128,
                        "barrier_trace_learner_count": 4,
                        "barrier_trace_validated": True,
                        "no_inner_step_while_blocked": True,
                        "base_versions_match": True,
                        "state_chain_contiguous": True,
                        "first_momentum_buffer_exact_zero": True,
                        "capture_tape_responder_join_exact": True,
                    }
                ],
            }
            for row in parent["results"]
        ]
    }

    rpm.validate_replay_attested_parent_work(parent, replay)

    replay["cells"][0]["replayed_attempts"][0]["commit_count"] = 31
    with pytest.raises(rpm.PhaseMapError, match="replay lacks frozen work evidence"):
        rpm.validate_replay_attested_parent_work(parent, replay)


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
            "--pipeline-depth",
            "--wan-streams",
            "--barrier-sync",
            "--version-matched-anchor",
        ):
            assert flag in command
        assert command[command.index("--pipeline-depth") + 1] == "4"
        assert command[command.index("--wan-streams") + 1] == "0"
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
                    "base_version": step - 4 if step > 4 else 0,
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


def _write_barrier_evidence(attempt: Path, cell: dict, args):
    tape = attempt / "work" / "m4" / "tape.jsonl"
    _write_tape(tape, cell)
    tape_rows = rpm.read_jsonl(tape)
    learner_entries = []
    for learner in range(4):
        trace = (
            attempt
            / "work"
            / "m4"
            / f"learner-{learner}"
            / "barrier-version-trace.jsonl"
        )
        trace.parent.mkdir(parents=True, exist_ok=True)
        events = []
        local_step = 0

        def emit(event, **fields):
            events.append(
                {
                    "schema": "yeto_barrier_trace_v1",
                    "event_seq": len(events) + 1,
                    "learner_id": learner,
                    "local_step": local_step,
                    "event": event,
                    **fields,
                }
            )

        for fragment in range(4):
            emit(
                "initial_broadcast_applied",
                fragment=fragment,
                broadcast_version=0,
                awaiting_fragments=[],
            )
        for group_start in range(0, len(tape_rows), 4):
            for _ in range(cell["H"]):
                local_step += 1
                emit("inner_step_started", awaiting_fragments=[])
            awaiting = []
            group = tape_rows[group_start : group_start + 4]
            for tape_row in group:
                fragment = tape_row["fragment"]
                responder = tape_row["responders"][learner]
                awaiting.append(fragment)
                emit(
                    "push_sent",
                    fragment=fragment,
                    pull_step=tape_row["step"],
                    base_version=responder["base_version"],
                    c_steps=responder["c_steps"],
                    c_tokens=responder["c_tokens"],
                    awaiting_fragments=list(awaiting),
                )
            for tape_row in group:
                fragment = tape_row["fragment"]
                responder = tape_row["responders"][learner]
                awaiting.remove(fragment)
                emit(
                    "broadcast_applied",
                    fragment=fragment,
                    pushed_base_version=responder["base_version"],
                    broadcast_version=tape_row["step"],
                    awaiting_fragments=list(awaiting),
                )
        trace.write_text("".join(json.dumps(event) + "\n" for event in events))
        learner_entries.append(
            {
                "learner_id": learner,
                "path": trace.relative_to(attempt).as_posix(),
                "sha256": rpm.sha256_file(trace),
                "size_bytes": trace.stat().st_size,
            }
        )
    registry = {
        "schema": "yeto_barrier_version_trace_v1",
        "learner_count": 4,
        "syncer_tape": {
            "path": tape.relative_to(attempt).as_posix(),
            "sha256": rpm.sha256_file(tape),
            "size_bytes": tape.stat().st_size,
        },
        "learner_traces": learner_entries,
    }
    registry_path = attempt / "report" / "barrier-version-trace.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry) + "\n")
    return tape_rows, registry_path


def _refresh_barrier_registry_entry(registry_path: Path, learner: int):
    registry = json.loads(registry_path.read_text())
    trace = registry_path.parent.parent / registry["learner_traces"][learner]["path"]
    registry["learner_traces"][learner]["sha256"] = rpm.sha256_file(trace)
    registry["learner_traces"][learner]["size_bytes"] = trace.stat().st_size
    registry_path.write_text(json.dumps(registry) + "\n")


@pytest.mark.parametrize("stage", ["p0a", "p0b"])
def test_barrier_trace_validator_proves_four_local_state_machines(tmp_path, stage):
    args = _canary_args(tmp_path, stage)
    cell = rpm.build_plan(args)["cells"][0]
    attempt = tmp_path / "attempt"
    tape_rows, _registry = _write_barrier_evidence(attempt, cell, args)

    proof = rpm.validate_barrier_version_trace(attempt, tape_rows, cell, args)

    assert proof["barrier_trace_validated"] is True
    assert proof["learner_count"] == 4
    assert proof["commit_count"] == cell["target_work"]["outer_steps"]
    assert proof["inner_steps_per_learner"] == 128
    learner_zero = rpm.read_jsonl(
        attempt / "work/m4/learner-0/barrier-version-trace.jsonl"
    )
    assert [
        (event["fragment"], event["broadcast_version"], event["local_step"])
        for event in learner_zero[:4]
    ] == [(fragment, 0, 0) for fragment in range(4)]
    assert [
        event["local_step"]
        for event in learner_zero
        if event["event"] == "push_sent"
    ] == [boundary for boundary in range(16, 129, 16) for _ in range(4)]
    assert not any(
        event["event"] == "inner_step_started" and event["local_step"] == 129
        for event in learner_zero
    )


def test_barrier_trace_validator_rejects_inner_step_before_broadcast(tmp_path):
    args = _args(tmp_path)
    cell = _cell_for_tape(args)
    attempt = tmp_path / "attempt"
    tape_rows, registry = _write_barrier_evidence(attempt, cell, args)
    trace = attempt / "work/m4/learner-0/barrier-version-trace.jsonl"
    events = rpm.read_jsonl(trace)
    push_index = next(
        index for index, event in enumerate(events) if event["event"] == "push_sent"
    )
    events.insert(
        push_index + 1,
        {
            "schema": "yeto_barrier_trace_v1",
            "event_seq": 0,
            "learner_id": 0,
            "local_step": events[push_index]["local_step"] + 1,
            "event": "inner_step_started",
            "awaiting_fragments": [],
        },
    )
    for sequence, event in enumerate(events, 1):
        event["event_seq"] = sequence
    trace.write_text("".join(json.dumps(event) + "\n" for event in events))
    _refresh_barrier_registry_entry(registry, 0)

    with pytest.raises(rpm.PhaseMapError, match="while blocked"):
        rpm.validate_barrier_version_trace(attempt, tape_rows, cell, args)


def test_barrier_trace_validator_rejects_rehashed_late_initial_broadcast(tmp_path):
    args = _args(tmp_path)
    cell = _cell_for_tape(args)
    attempt = tmp_path / "attempt"
    tape_rows, registry = _write_barrier_evidence(attempt, cell, args)
    trace = attempt / "work/m4/learner-0/barrier-version-trace.jsonl"
    events = rpm.read_jsonl(trace)
    events[2]["local_step"] = 1
    trace.write_text("".join(json.dumps(event) + "\n" for event in events))
    _refresh_barrier_registry_entry(registry, 0)

    with pytest.raises(rpm.PhaseMapError, match="initial broadcast prefix"):
        rpm.validate_barrier_version_trace(attempt, tape_rows, cell, args)


@pytest.mark.parametrize(
    ("field", "value"),
    [("fragment", False), ("broadcast_version", False)],
)
def test_barrier_trace_validator_rejects_boolean_initial_coordinates(
    tmp_path, field, value
):
    args = _args(tmp_path)
    cell = _cell_for_tape(args)
    attempt = tmp_path / "attempt"
    tape_rows, registry = _write_barrier_evidence(attempt, cell, args)
    trace = attempt / "work/m4/learner-0/barrier-version-trace.jsonl"
    events = rpm.read_jsonl(trace)
    events[0][field] = value
    trace.write_text("".join(json.dumps(event) + "\n" for event in events))
    _refresh_barrier_registry_entry(registry, 0)

    with pytest.raises(rpm.PhaseMapError, match="initial broadcast prefix"):
        rpm.validate_barrier_version_trace(attempt, tape_rows, cell, args)


def test_barrier_trace_validator_rejects_wrong_broadcast_version(tmp_path):
    args = _args(tmp_path)
    cell = _cell_for_tape(args)
    attempt = tmp_path / "attempt"
    tape_rows, registry = _write_barrier_evidence(attempt, cell, args)
    trace = attempt / "work/m4/learner-2/barrier-version-trace.jsonl"
    events = rpm.read_jsonl(trace)
    broadcast = next(
        event for event in events if event["event"] == "broadcast_applied"
    )
    broadcast["broadcast_version"] += 1
    trace.write_text("".join(json.dumps(event) + "\n" for event in events))
    _refresh_barrier_registry_entry(registry, 2)

    with pytest.raises(rpm.PhaseMapError, match="exact pushed round"):
        rpm.validate_barrier_version_trace(attempt, tape_rows, cell, args)


def test_barrier_trace_validator_rejects_rehashed_late_fragment_window(tmp_path):
    args = _args(tmp_path)
    cell = _cell_for_tape(args)
    attempt = tmp_path / "attempt"
    tape_rows, registry = _write_barrier_evidence(attempt, cell, args)
    trace = attempt / "work/m4/learner-1/barrier-version-trace.jsonl"
    events = rpm.read_jsonl(trace)
    for event in events:
        if (
            event["event"] == "push_sent" and event["pull_step"] in range(1, 5)
        ) or (
            event["event"] == "broadcast_applied"
            and event["broadcast_version"] in range(1, 5)
        ):
            event["local_step"] = cell["H"] + 1
    trace.write_text("".join(json.dumps(event) + "\n" for event in events))
    _refresh_barrier_registry_entry(registry, 1)

    with pytest.raises(rpm.PhaseMapError, match="push does not biject"):
        rpm.validate_barrier_version_trace(attempt, tape_rows, cell, args)


def test_barrier_trace_validator_rejects_registry_hash_mismatch(tmp_path):
    args = _args(tmp_path)
    cell = _cell_for_tape(args)
    attempt = tmp_path / "attempt"
    tape_rows, _registry = _write_barrier_evidence(attempt, cell, args)
    trace = attempt / "work/m4/learner-3/barrier-version-trace.jsonl"
    trace.write_text(trace.read_text() + "\n")

    with pytest.raises(rpm.PhaseMapError, match="registry hash mismatch"):
        rpm.validate_barrier_version_trace(attempt, tape_rows, cell, args)


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


def test_eval_validation_accepts_frozen_zero_supervision_as_valid_excluded(tmp_path):
    report, expected, rows = _eval_validation_fixture(tmp_path)
    frozen_path = Path(expected["_eval_sequences_path"])
    frozen = [json.loads(line) for line in frozen_path.read_text().splitlines()]
    frozen[0]["supervised_token_count"] = 0
    frozen_path.write_text("".join(json.dumps(row) + "\n" for row in frozen))
    rows[0]["token_count"] = 0
    rows[0]["loss_sum"] = -0.0
    rows[0]["loss_per_token"] = -0.0
    (report / "per-example-loss" / "m4.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    expected["eval_supervised_token_count"] = 4
    provenance_path = report / "eval-provenance" / "eval_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["eval_supervised_token_count"] = 4
    provenance_path.write_text(json.dumps(provenance))

    _summary, path = rpm.validate_eval(report, 2.0, expected)

    assert path.name == "m4.jsonl"


def test_eval_validation_rejects_nonzero_loss_for_zero_supervision(tmp_path):
    report, expected, rows = _eval_validation_fixture(tmp_path)
    frozen_path = Path(expected["_eval_sequences_path"])
    frozen = [json.loads(line) for line in frozen_path.read_text().splitlines()]
    frozen[0]["supervised_token_count"] = 0
    frozen_path.write_text("".join(json.dumps(row) + "\n" for row in frozen))
    rows[0].update(token_count=0, loss_sum=1.0, loss_per_token=0.0)
    (report / "per-example-loss" / "m4.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    expected["eval_supervised_token_count"] = 4
    provenance_path = report / "eval-provenance" / "eval_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["eval_supervised_token_count"] = 4
    provenance_path.write_text(json.dumps(provenance))

    with pytest.raises(rpm.PhaseMapError, match="zero-target sequence"):
        rpm.validate_eval(report, 2.0, expected)


def test_provider_evidence_is_precopied_into_phase_map_root(tmp_path):
    source = tmp_path / "provider-input.json"
    source.write_text(json.dumps({"instance_id": "123"}, sort_keys=True) + "\n")
    run_dir = tmp_path / "phase-map"
    run_dir.mkdir()
    args = SimpleNamespace(
        provider_evidence=source,
        run_dir=run_dir,
        artifact_uri="gs://bucket/run/phase-map",
    )
    digest = rpm.sha256_file(source)

    snapshot, uri = rpm.snapshot_provider_evidence(
        args, {"instance_id": "123"}, digest
    )

    root_copy = run_dir / "provider-evidence.json"
    assert root_copy.read_bytes() == source.read_bytes()
    assert not root_copy.is_symlink()
    assert snapshot.read_bytes() == source.read_bytes()
    assert uri.endswith(f"provider-evidence/instance-123-{digest}.json")


def test_preconfirmation_scan_ignores_only_saved_tokenizer_vocabularies(tmp_path):
    attempt = tmp_path / "attempt-1"
    tokenizer = attempt / "work" / "m4" / "learner-0" / "tokenizer.json"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_text(json.dumps({"vocab": {"Ġaudit": 1}}))

    rpm.validate_preconfirmation_surface(attempt, ["compare_diloco.py"])

    result = attempt / "report" / "result.json"
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps({"audit_loss": 1.0}))
    with pytest.raises(rpm.PhaseMapError, match="audit field"):
        rpm.validate_preconfirmation_surface(attempt, ["compare_diloco.py"])


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


def _work_evidence_manifest(
    *,
    status: str = "COMPLETED",
    loss: float | None = 2.0,
    learner_steps: int = 128,
    runner_exit: int = 0,
) -> dict:
    cell_id = "bp-phase-map-p0a-h16-mu0-eta0p0875-s337"
    return {
        "status": "bound_launch_authority",
        "lineage": {"descendant_kind": "p0a_single_gpu_bound"},
        "expected_cells": [
            {
                "cell_id": cell_id,
                "expected_learner_count": 4,
                "expected_learner_steps": 128,
            }
        ],
        "results": [
            {
                "cell_id": cell_id,
                "attempt": 1,
                "status": status,
                "loss": loss,
                "observed_work": {
                    "learner_step_counts": {
                        str(learner): learner_steps for learner in range(4)
                    }
                },
                "exit_statuses": {
                    "runner": runner_exit,
                    "syncer": 0 if runner_exit == 0 else None,
                    "learners": [0, 0, 0, 0] if runner_exit == 0 else [1],
                },
            }
        ],
    }


def test_pre_step_learner_exit_records_failed_nonzero_and_no_clean_seal(
    tmp_path, monkeypatch
):
    args = SimpleNamespace(run_dir=tmp_path)
    manifest = _work_evidence_manifest(
        status="FAILED", loss=None, learner_steps=0, runner_exit=1
    )

    monkeypatch.setattr(
        rpm,
        "build_parser",
        lambda: SimpleNamespace(parse_args=lambda _argv: args),
    )
    monkeypatch.setattr(
        rpm, "execute", lambda _args: rpm.finalize_campaign(args, manifest)
    )

    assert rpm.main([]) == 2
    partial = json.loads((tmp_path / "phase-map-manifest.partial.json").read_text())
    assert partial["status"] == "FAILED"
    assert partial["results"][0]["status"] == "FAILED"
    assert not (tmp_path / "acquisition-seal.json").exists()
    assert not (tmp_path / "phase-map-manifest.json").exists()


def test_fewer_than_frozen_learner_steps_fails_campaign(tmp_path):
    args = SimpleNamespace(run_dir=tmp_path)
    manifest = _work_evidence_manifest(learner_steps=127)

    with pytest.raises(rpm.WorkEvidenceError, match="frozen step count"):
        rpm.finalize_campaign(args, manifest)

    partial = json.loads((tmp_path / "phase-map-manifest.partial.json").read_text())
    assert partial["status"] == "FAILED"
    assert partial["results"][0]["status"] == "FAILED"
    assert not (tmp_path / "acquisition-seal.json").exists()


def test_nonfinite_terminal_loss_fails_campaign(tmp_path):
    args = SimpleNamespace(run_dir=tmp_path)
    manifest = _work_evidence_manifest(loss=float("nan"))

    with pytest.raises(rpm.WorkEvidenceError, match="finite terminal loss"):
        rpm.finalize_campaign(args, manifest)

    partial = json.loads((tmp_path / "phase-map-manifest.partial.json").read_text())
    assert partial["status"] == "FAILED"
    assert partial["results"][0]["status"] == "FAILED"
    assert not (tmp_path / "acquisition-seal.json").exists()


def test_complete_mocked_learners_seal_with_per_cell_loss(tmp_path, monkeypatch):
    args = SimpleNamespace(run_dir=tmp_path)
    manifest = _work_evidence_manifest(loss=1.75)
    monkeypatch.setattr(
        rpm,
        "acquisition_paths",
        lambda run_dir, _manifest: [
            run_dir / "phase-map-manifest.json",
            run_dir / "phase-map-acquisition-manifest.json",
            run_dir / "acquisition-seal.json",
        ],
    )

    rpm.finalize_campaign(args, manifest)

    sealed = json.loads((tmp_path / "phase-map-acquisition-manifest.json").read_text())
    assert sealed["status"] == "sealed_acquisition_pending_teardown"
    assert sealed["results"][0]["loss"] == 1.75
    assert sealed["results"][0]["observed_work"]["learner_step_counts"] == {
        str(learner): 128 for learner in range(4)
    }
    assert sealed["results"][0]["exit_statuses"] == {
        "runner": 0,
        "syncer": 0,
        "learners": [0, 0, 0, 0],
    }
    assert (tmp_path / "acquisition-seal.json").is_file()


def test_learner_data_path_survives_different_launch_cwd(tmp_path, monkeypatch):
    from scripts import compare_diloco as compare

    invocation = tmp_path / "invocation"
    data = invocation / "inputs" / "train.jsonl"
    data.parent.mkdir(parents=True)
    data.write_text('{"messages": []}\n')
    launch_dir = tmp_path / "different-cwd"
    launch_dir.mkdir()
    monkeypatch.chdir(invocation)
    args = SimpleNamespace(
        model="model-id",
        data="inputs/train.jsonl",
        probe_data=None,
        prebound_development_eval=None,
        action_probe_anchor_manifest=None,
        adapter_dir=None,
        work_dir=Path("work"),
        report_dir=Path("report"),
        lora_r=16,
        lora_alpha=32,
        seq_len=128,
        micro_batch_size=1,
        inner_lr=3e-4,
        device="cpu",
        shard="ddp",
        learner_gpus=0,
        training_seed=337337,
    )
    compare.resolve_subprocess_paths(args)
    args.learner_data = Path(args.data)
    command = compare.learner_command(
        args,
        Path("work/m4"),
        learner_id=0,
        num_learners=4,
        syncer="127.0.0.1:1",
        max_steps=128,
        arm=compare.PRESETS["m4"],
    )
    learner_data = command[command.index("--data") + 1]

    assert args.data == str(data.resolve())
    assert Path(learner_data).is_absolute()
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; print(Path(sys.argv[1]).read_text())",
            learner_data,
        ],
        cwd=launch_dir,
        text=True,
        capture_output=True,
    )
    assert check.returncode == 0
    assert check.stdout.strip() == '{"messages": []}'


def _passing_sealed_p1(tmp_path: Path) -> dict:
    _args_value, plan, bound = _authority_bound_p1(tmp_path)
    parent = rpm.build_schema_fixture(bound, plan)
    tuned = {
        (16, 0.0): 2.00,
        (16, 0.5): 2.01,
        (16, 0.9): 2.03,
        (64, 0.0): 2.00,
        (64, 0.5): 2.00,
        (64, 0.9): 2.00,
        (256, 0.0): 2.00,
        (256, 0.5): 1.98,
        (256, 0.9): 2.01,
    }
    eta_penalty = {
        0.021875: 0.02,
        0.04375: 0.0,
        0.0875: 0.02,
        0.175: 0.04,
    }
    for row in parent["results"]:
        row["loss"] = tuned[(int(row["h"]), float(row["mu"]))] + eta_penalty[
            float(row["eta"])
        ]
    parent["status"] = "sealed_results"
    return parent


def test_adaptive_p1_uses_registered_midpoints_as_complete_cumulative_blocks(
    tmp_path,
):
    args, initial_plan, initial_bound = _authority_bound_p1(tmp_path)
    parent = rpm.build_schema_fixture(initial_bound, initial_plan)
    parent_path = tmp_path / "sealed-initial-p1.json"
    parent_path.write_text(json.dumps(parent, indent=2, sort_keys=True) + "\n")
    parent = json.loads(parent_path.read_text())
    lower_midpoint = (0.021875 * 0.04375) ** 0.5
    upper_midpoint = (0.04375 * 0.0875) ** 0.5
    args.study_id = "bp-phase-map-p1-adaptive-1"
    args.study_phase = "p1_adaptive_bracket"
    args.h = [16, 64, 256]
    args.eta = [lower_midpoint, upper_midpoint]
    args.parent_manifest = parent_path
    args.expected_parent_manifest_hash = rpm.sha256_bytes(rpm.canonical_json(parent))
    args.parent_replay_report = None
    args.expected_parent_replay_report_hash = None

    plan = rpm.build_plan(args)
    bound = rpm.build_bound_manifest(args, plan, **BOUND_HASHES)

    assert len(plan["cells"]) == 18
    assert bound["expected_cells"][:36] == parent["expected_cells"]
    assert bound["results"] == parent["results"]
    assert bound["lineage"]["descendant_kind"] == "adaptive_bracket_round"
    report = validator.validate_and_summarize(
        bound,
        claim_level="integrity",
        parent_manifest=parent,
    )
    assert report["expected_cell_count"] == 54


def _p2_builder_fixture(tmp_path: Path):
    parent = _passing_sealed_p1(tmp_path)
    parent_path = tmp_path / "sealed-p1.json"
    parent_path.write_text(json.dumps(parent, indent=2, sort_keys=True) + "\n")
    parent = json.loads(parent_path.read_text())
    args = _args(tmp_path)
    args.study_id = "bp-phase-map-p2"
    args.study_phase = "p2_additional_development"
    args.eta = [0.021875, 0.04375, 0.0875]
    args.seed = 359
    args.training_seed = 359359
    args.additional_seed = [373]
    args.additional_training_seed = [373373]
    args.parent_manifest = parent_path
    args.expected_parent_manifest_hash = rpm.sha256_bytes(rpm.canonical_json(parent))
    args.parent_replay_report = None
    args.expected_parent_replay_report_hash = None
    plan = rpm.build_plan(args)
    bound = rpm.build_bound_manifest(
        args,
        plan,
        **BOUND_HASHES,
        additional_train_rows_hashes={373: "6" * 64},
        additional_train_source_indices_hashes={373: "7" * 64},
    )
    return args, parent, plan, bound


def test_registered_cumulative_builder_constructs_p2_from_sealed_p1(tmp_path):
    _args_value, parent, plan, bound = _p2_builder_fixture(tmp_path)

    assert len(plan["cells"]) == 54
    blocks = {}
    for cell in plan["cells"]:
        key = (cell["H"], cell["eta"], cell["seed"])
        blocks.setdefault(key, set()).add(cell["mu"])
    assert len(blocks) == 18
    assert all(mu == {0.0, 0.5, 0.9} for mu in blocks.values())
    assert bound["expected_cells"][: len(parent["expected_cells"])] == parent[
        "expected_cells"
    ]
    assert bound["results"] == parent["results"]
    assert len(bound["expected_cells"]) == 90
    assert bound["expected_grid"]["seeds"] == [347, 359, 373]
    assert bound["lineage"]["descendant_kind"] == "additional_development_stage"
    assert bound["lineage"]["parent_manifest_sha256"] == rpm.sha256_bytes(
        rpm.canonical_json(parent)
    )

    report = validator.validate_and_summarize(
        bound,
        claim_level="integrity",
        parent_manifest=parent,
    )
    assert report["integrity_status"] == "BOUND_LAUNCH_AUTHORITY_VALIDATED"
    assert report["expected_cell_count"] == 90
    assert report["attempt_count"] == 36


def test_cumulative_descendant_rejects_any_old_result_mutation(tmp_path):
    _args_value, parent, _plan, bound = _p2_builder_fixture(tmp_path)
    corrupted = copy.deepcopy(bound)
    corrupted["results"][0]["loss"] += 0.001

    with pytest.raises(rpm.PhaseMapError, match="mutates or reorders parent result"):
        rpm.verify_parent_hash_chain([parent, corrupted])
    with pytest.raises(validator.ManifestError, match="exact immutable prefix"):
        validator.validate_and_summarize(
            corrupted,
            claim_level="integrity",
            parent_manifest=parent,
        )


def test_cumulative_descendant_enforces_complete_three_mu_new_blocks(tmp_path):
    _args_value, parent, _plan, bound = _p2_builder_fixture(tmp_path)
    incomplete = copy.deepcopy(bound)
    removed = incomplete["expected_cells"].pop()
    del incomplete["frozen"]["cell_command_hashes"][removed["cell_id"]]

    with pytest.raises(validator.ManifestError) as caught:
        validator.validate_and_summarize(
            incomplete,
            claim_level="integrity",
            parent_manifest=parent,
        )
    assert "complete live-control block" in str(caught.value)


def test_parent_hash_chain_verifies_each_exact_canonical_parent(tmp_path):
    _args_value, parent, _plan, bound = _p2_builder_fixture(tmp_path)
    rpm.verify_parent_hash_chain([parent, bound])

    corrupted = copy.deepcopy(bound)
    corrupted["lineage"]["parent_manifest_sha256"] = "0" * 64
    with pytest.raises(rpm.PhaseMapError, match="parent canonical hash mismatch"):
        rpm.verify_parent_hash_chain([parent, corrupted])


def test_cumulative_lineage_rejects_reordered_parent_prefix(tmp_path):
    _args_value, parent, _plan, bound = _p2_builder_fixture(tmp_path)
    reordered = copy.deepcopy(bound)
    reordered["results"][0], reordered["results"][1] = (
        reordered["results"][1],
        reordered["results"][0],
    )

    with pytest.raises(rpm.PhaseMapError, match="mutates or reorders parent result"):
        rpm.verify_parent_hash_chain([parent, reordered])


def test_synthetic_p2_seals_with_parent_rows_as_exact_prefix(tmp_path):
    args, parent, plan, bound = _p2_builder_fixture(tmp_path)
    sealed = rpm.build_schema_fixture(bound, plan)

    assert sealed["results"][: len(parent["results"])] == parent["results"]
    assert len(sealed["results"]) == len(sealed["expected_cells"]) == 90
    rpm.validate_campaign_work_evidence(sealed)
    report = validator.validate_and_summarize(
        sealed,
        claim_level="development",
        parent_manifest=parent,
    )
    assert report["integrity_status"] == "VALIDATED"
    assert report["independent_seeds"] == [347, 359, 373]
