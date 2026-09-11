"""CPU-only tests for independent v4 dynamic-exclusion promotion."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from monitoring import promote_masked_native_gap_v4_exclusions_20260911 as promote
from training.qwen38_no_cot.prepare_data import SEED as BASELINE_SEED


HASH = "a" * 64


def test_replay_split_uses_the_frozen_baseline_seed():
    # This group is copied from the immutable baseline replay selection.  It is
    # a validation group with the qualified converter seed, but would be train
    # under the short date-only seed that previously slipped into this helper.
    group = "session:01a01c5b-1727-7103-8942-30f3d970b52b"
    assert promote.SEED == BASELINE_SEED == "qwen38-no-cot-sft-20260908/v1"
    assert promote._split_for_group(group) == "validation"


def _record(identity="trace-1", *, reason="UnsupportedTrace", **changed):
    duplicate = reason.startswith("duplicate_")
    value = {
        "identity": identity,
        "reason": reason,
        "generated_gaps": 2,
        "source_digest": "b" * 64,
        "row_sha256": "c" * 64,
        "duplicate_basis": (
            "normalized_messages_sha256"
            if reason == "duplicate_normalized_messages"
            else "retained_arrays_sha256" if duplicate else None
        ),
        "duplicate_sha256": "d" * 64 if duplicate else None,
        "duplicate_of_source": "replay" if duplicate else None,
        "duplicate_of_identity": "replay:" + "e" * 64 if duplicate else None,
        "native_failure_sha256": "f" * 64 if reason in {
            "UnsupportedTrace", "UnsupportedMessage"} else None,
    }
    value.update(changed)
    return value


def _write_policy(path, records):
    value = {
        "schema": promote.POLICY_SCHEMA,
        "snapshot_sha256": "1" * 64,
        "source_sha256": "2" * 64,
        "record_count": len(records),
        "generated_gap_count": sum(record["generated_gaps"] for record in records),
        "records": records,
    }
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return value


def test_candidate_policy_strictly_validates_evidence_shape(tmp_path):
    path = tmp_path / "candidate.json"
    expected = _write_policy(path, [_record()])
    raw, policy, records = promote._candidate_policy(path, {
        "snapshot_sha256": "1" * 64, "source_sha256": "2" * 64})
    assert raw == path.read_bytes() and policy == expected
    assert records == {"trace-1": expected["records"][0]}

    malformed = _record(native_failure_sha256=None)
    _write_policy(path, [malformed])
    with pytest.raises(ValueError, match="record is invalid"):
        promote._candidate_policy(path, {
            "snapshot_sha256": "1" * 64, "source_sha256": "2" * 64})

    malformed = _record(reason="unrepresentable_text_after_tool",
                        duplicate_sha256="d" * 64)
    _write_policy(path, [malformed])
    with pytest.raises(ValueError, match="record is invalid"):
        promote._candidate_policy(path, {
            "snapshot_sha256": "1" * 64, "source_sha256": "2" * 64})


def test_strict_json_refuses_duplicate_keys_and_jsonl_noncanonical(tmp_path):
    with pytest.raises(ValueError, match="duplicate JSON key"):
        promote._strict_json(b'{"a":1,"a":2}', "fixture")
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b'{"z":1, "a":2}\n')
    with pytest.raises(ValueError, match="not canonical JSONL"):
        promote._strict_jsonl(path, "fixture")


def test_replay_loader_preserves_exact_converter_job_shape(tmp_path):
    replay = tmp_path / "rollout.jsonl"
    replay.write_text('{"fixture":true}\n')
    replay_sha = hashlib.sha256(replay.read_bytes()).hexdigest()
    manifest = tmp_path / "inventory.jsonl"
    job = {"source": "replay", "format": "rollout", "sha256": replay_sha,
           "identity": "replay:" + replay_sha, "path": str(replay)}
    manifest.write_text(promote.canonical(job) + "\n")
    selection = tmp_path / "selection.jsonl"
    group = "fixture-group"
    selected = {**job, "original_baseline_group_id": group,
                "original_baseline_split": promote._split_for_group(group)}
    selection.write_text(promote.canonical(selected) + "\n")
    receipt = {"replay_manifests": [{"path": str(manifest),
        "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}]}
    all_jobs, kept = promote._load_replay_jobs(receipt, selection)
    assert len(all_jobs) == len(kept) == 1
    assert "kind" not in all_jobs[0]
    assert kept[0]["kind"] == "replay"
    assert kept[0]["source_jobs_line"] == 1


def test_derive_records_requires_exact_job_partition_and_source_identity():
    source_row = {"source_digest": "b" * 64, "row_sha256": "c" * 64}
    jobs = [{"kind": "trace", "source": "trace", "identity": "trace-1",
             "source_row": source_row},
            {"kind": "replay", "source": "replay", "identity": "replay-1"}]
    record = _record()
    exclusion = {"ok": False, "source": "trace", "identity": "trace-1",
                 "reason": "UnsupportedTrace", "generated_gaps": 2,
                 "error_message_sha256": "f" * 64}
    accepted = [{"ok": True, "source": "replay", "identity": "replay-1"}]
    derived, accepted_map, _ = promote._derive_records(
        {"trace-1": record}, [exclusion], jobs, accepted)
    assert derived == {"trace-1": record}
    assert set(accepted_map) == {("replay", "replay-1")}
    with pytest.raises(ValueError, match="partition"):
        promote._derive_records({"trace-1": record}, [exclusion], jobs, [])


def test_rerender_verifier_proves_exact_earlier_duplicate_owner():
    owner_identity = "replay:" + "e" * 64
    jobs = [
        {"source": "replay", "identity": owner_identity},
        {"source": "trace", "identity": "trace-1"},
    ]
    record = _record(reason="duplicate_retained_arrays",
                     duplicate_of_identity=owner_identity)
    candidate = {"trace-1": record}
    accepted = {("replay", owner_identity): {
        "ok": True, "source": "replay", "identity": owner_identity,
        "retained_arrays_sha256": "d" * 64}}
    results = [
        {"ok": True, "source": "replay", "identity": owner_identity,
         "retained_arrays_sha256": "d" * 64},
        {"ok": True, "source": "trace", "identity": "trace-1",
         "retained_arrays_sha256": "d" * 64, "generated_gaps": 2},
    ]
    receipt = promote._verify_rerender_results(jobs, candidate, accepted, results)
    assert receipt["duplicate_exclusions_verified"] == 1
    assert receipt["unique_owner_rows_rerendered"] == 1
    assert len(receipt["result_set_sha256"]) == 64
    results[0]["retained_arrays_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="duplicate owner proof failed"):
        promote._verify_rerender_results(jobs, candidate, accepted, results)


def test_rerender_verifier_reproduces_native_failure_digest():
    jobs = [{"source": "trace", "identity": "trace-1"}]
    candidate = {"trace-1": _record()}
    result = {"ok": False, "source": "trace", "identity": "trace-1",
              "reason": "UnsupportedTrace", "generated_gaps": 2,
              "native_failure_sha256": "f" * 64}
    assert promote._verify_rerender_results(jobs, candidate, {}, [result])[
        "excluded_traces_rerendered"] == 1
    result["native_failure_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="rerender differs"):
        promote._verify_rerender_results(jobs, candidate, {}, [result])


def test_manifest_gate_refuses_any_non_dynamic_failure(tmp_path):
    candidate = tmp_path / "dynamic-trace-exclusions.candidate.json"
    candidate.write_text("{}")
    manifest = {
        "full_export": True, "all_train": True, "internal_validation": False,
        "splits": {"train": []}, "unexpected_trace_failures": 0,
        "all_frozen_generations_accounted_for": True,
        "cross_arm_no_cot_parity": {"verified": True, "adapter_divergence_rows": 0},
        "semantic_action_boundary_audit": {"verified": True},
        "raw_tool_cardinality_audit": {"verified": True},
        "trace_source_membership": {"verified": True,
            "full_frozen_coverage_required": True,
            "preaudited_excluded_trace_sources": 0, "processed_trace_sources": 1,
            "accepted_trace_sources": 0, "failed_trace_sources": 1},
        "replay_source_membership": {"verified": True,
            "expected_selected_sources": 1, "accepted_sources": 1,
            "excluded_source_count": 0},
        "journal_cardinality_audit": {"verified": True},
        "end_of_build_input_stability": {"verified": True},
        "build_runtime": {"verified": True}, "source_jobs": 2,
        "input_source_jobs": 3, "source_exclusion_policy": None,
        "freeze_receipt_sha256": HASH, "generation_snapshot_sha256": "1" * 64,
        "source_sha256": "2" * 64, "frozen_valid_candidate_count": 2,
        "included_valid_candidate_count": 0, "excluded_valid_candidate_count": 2,
        "omitted_valid_candidate_count": 0, "dynamic_trace_exclusion_policy": None,
        "dynamic_trace_exclusion_audit": {"verified": False,
            "expected_record_count": 0, "observed_record_count": 1,
            "candidate_path": candidate.name,
            "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest()},
    }
    receipt = {"snapshot_sha256": "1" * 64, "source_sha256": "2" * 64,
               "counts": {"valid_candidates": 2}}
    gates = promote._manifest_gates(manifest, candidate, [{}, {}], 3, receipt)
    assert all(gates.values())
    manifest["replay_source_membership"]["excluded_source_count"] = 1
    with pytest.raises(ValueError, match="replay_membership"):
        promote._manifest_gates(manifest, candidate, [{}, {}], 3, receipt)


def test_policy_publication_is_atomic_exact_and_immutable(tmp_path):
    parent = tmp_path / "policies"
    parent.mkdir(mode=0o700)
    output = parent / "cot-masked-native-gap-v4-fixture"
    policy_raw = b'{"fixture":true}\n'
    receipt = {"schema": promote.PROMOTION_SCHEMA, "status": "complete"}
    published = promote._publish(output, policy_raw, receipt)
    assert (output / "dynamic-trace-exclusions.json").read_bytes() == policy_raw
    assert json.loads((output / "promotion-receipt.json").read_bytes()) == receipt
    assert output.stat().st_mode & 0o777 == 0o500
    assert all(path.stat().st_mode & 0o777 == 0o400 for path in output.iterdir())
    assert published["policy_sha256"] == hashlib.sha256(policy_raw).hexdigest()
    with pytest.raises(ValueError, match="fresh policy directory"):
        promote._publish(output, policy_raw, receipt)


def test_promotion_receipt_contract_matches_final_builder_requirements():
    assert len(promote.PROMOTION_RECEIPT_KEYS) == 53
    assert {"candidate_build_receipt_path", "candidate_tree_sha256",
            "candidate_build_core_sha256", "candidate_non_dynamic_gates",
            "independent_journal_cardinality_audit", "independent_rerender",
            "policy_path", "policy_sha256", "policy_bytes"} <= \
        promote.PROMOTION_RECEIPT_KEYS
