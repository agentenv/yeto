"""CPU-only immutable full-source export for the unreviewed CoT experiment.

The caller first freezes the paused journal with SQLite backup and supplies its
hash receipt. The original journal is never opened for writing or resumed.
Every original source is read once per worker job, including zero-filled traces.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
import traceback

from cot_filler.core import canonical, digest, validate_trace
from training.qwen38_no_cot.prepare_data import SEED, sha_file
from . import masked_replay as replay
from .masked import CrossArmParityError,MaskedBoundaryExclusion
from training.qwen38_no_cot.prepare_data import UnsupportedTrace
from training.qwen38_no_cot.upstream_manager.render import UnsupportedMessage

VERSION = "qwen38-full-native-gap-cot-cpu-export/v4"
_renderer = _db = _identity = _source = _spool = None
_all_train = False
ALL_TRAIN_POLICY = "all-source-train-only-capture-identity-for-missing-session/v1"
DYNAMIC_EXCLUSION_SCHEMA = "qwen38-exact-dynamic-trace-exclusions/v2"
DYNAMIC_EXCLUSION_REASONS = {
    "UnsupportedMessage", "UnsupportedTrace", "unrepresentable_text_after_tool",
    "duplicate_normalized_messages", "duplicate_retained_arrays",
}
RUNTIME_IMAGE = "sha256:b0bd0e50b29cb0a3ec69c3cbbb49652d3c1aabedbfe22ee0fa6d053b6bb33aee"
CONVERSION_DEPENDENCIES = (
    "training/__init__.py",
    "cot_filler/__init__.py", "cot_filler/core.py", "cot_filler/corpus_worker.py",
    "cot_filler/corpus_source.py", "cot_filler/provider.py", "cot_filler/regeneration.py",
    "cot_filler/grounding_review.py", "cot_filler/grounding_review_ids.py",
    "cot_filler/grounding_review_v3.py", "cot_filler/grounding_review_v4.py",
    "cot_filler/grounding_review_fast.py",
    "training/qwen38_no_cot/__init__.py",
    "training/qwen38_no_cot/prepare_data.py", "training/qwen38_no_cot/render.py",
    "training/qwen38_no_cot/upstream_manager/__init__.py",
    "training/qwen38_no_cot/upstream_manager/render.py",
    "training/qwen38_cot_masked/__init__.py",
    "training/qwen38_cot_masked/render.py",
    "training/qwen38_cot_experimental/__init__.py",
    "training/qwen38_cot_experimental/render.py",
    "training/qwen38_cot_experimental/generation_contract.py",
    "training/qwen38_turn_boundary_v2/__init__.py",
    "training/qwen38_turn_boundary_v2/masked.py",
    "training/qwen38_turn_boundary_v2/normalize.py",
    "training/qwen38_turn_boundary_v2/source_adapters.py",
    "training/qwen38_native_gap_v3/__init__.py",
    "training/qwen38_native_gap_v3/masked.py",
    "training/qwen38_native_gap_v3/masked_data.py",
    "training/qwen38_native_gap_v3/normalize.py",
    "training/qwen38_native_gap_v3/source_adapters.py",
    "training/qwen38_native_gap_v3/masked_replay.py",
    "training/qwen38_native_gap_v3/prepare_masked_full.py",
)


def _hash(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def conversion_dependency_sha256():
    root = Path(__file__).resolve().parents[2]
    return {name: sha_file(root / name) for name in CONVERSION_DEPENDENCIES}


def split_for_group(group):
    return "validation" if int(hashlib.sha256((SEED + ":split:" + group).encode()).hexdigest()[:8], 16) % 100 == 0 else "train"


def _readonly(path):
    db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    db.row_factory = sqlite3.Row
    return db


def _path_on_read_only_mount(path):
    mountinfo = Path("/proc/self/mountinfo")
    if not mountinfo.is_file():
        return False
    target = str(Path(path).resolve(strict=True))
    matches = []
    for line in mountinfo.read_text().splitlines():
        before, separator, _ = line.partition(" - ")
        if not separator:
            continue
        fields = before.split()
        if len(fields) < 6:
            continue
        mount = fields[4].replace("\\040", " ").replace("\\134", "\\")
        if target == mount or target.startswith(mount.rstrip("/") + "/"):
            matches.append((len(mount), set(fields[5].split(","))))
    return bool(matches) and "ro" in max(matches, key=lambda item: item[0])[1]


def verify_read_only_build_inputs(paths):
    """Require the frozen source, journal, policies, tokenizer and code on RO mounts."""
    if os.environ.get("YETA_TRAINING_IMAGE") != RUNTIME_IMAGE:
        raise ValueError("CPU export must run in the exact pinned NeMo image")
    checked = {}
    for label, supplied in paths:
        lexical = Path(supplied)
        if lexical.is_symlink():
            raise ValueError("Build input cannot be a symlink: " + label)
        path = lexical.resolve(strict=True)
        if not (path.is_file() or path.is_dir()) or not _path_on_read_only_mount(path):
            raise ValueError("Build input requires a dedicated read-only mount: " + label)
        checked[label] = str(path)
    return {"schema": "qwen38-native-gap-v4-read-only-build-inputs/v1",
            "verified": True, "runtime_image": RUNTIME_IMAGE,
            "paths": dict(sorted(checked.items()))}


def load_freeze(receipt_path):
    receipt_path = Path(receipt_path).resolve(strict=True)
    receipt = json.loads(receipt_path.read_text())
    snapshot = Path(receipt["snapshot_path"]).resolve(strict=True)
    if (receipt.get("schema") != "qwen38-frozen-generation-journal/v1"
            or sha_file(snapshot) != receipt["snapshot_sha256"]
            or snapshot == Path(receipt["original_journal"]).resolve()):
        raise ValueError("A distinct immutable generation snapshot with exact hash is required")
    db = _readonly(snapshot)
    try:
        identity = json.loads(db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()[0])
        if digest(identity) != receipt["original_identity_sha256"]:
            raise ValueError("Frozen journal identity changed")
        sources = [dict(row) for row in db.execute("SELECT * FROM sources ORDER BY id")]
        candidates = db.execute("SELECT count(*) FROM candidates").fetchone()[0]
        gap_states = dict(db.execute(
            "SELECT state,count(*) FROM gaps GROUP BY state ORDER BY state"))
        pending = db.execute("SELECT count(*) FROM gaps WHERE state='review_pending'").fetchone()[0]
        valid = db.execute("SELECT count(*) FROM candidates c JOIN gaps g ON g.id=c.gap_id WHERE g.state='review_pending'").fetchone()[0]
        generation_invalid_candidates = db.execute(
            "SELECT count(*) FROM candidates c JOIN gaps g ON g.id=c.gap_id "
            "WHERE g.state='generation_invalid'").fetchone()[0]
        unexpected_state_candidates = db.execute(
            "SELECT count(*) FROM candidates c JOIN gaps g ON g.id=c.gap_id "
            "WHERE g.state NOT IN ('review_pending','generation_invalid')").fetchone()[0]
        orphan_candidates = db.execute(
            "SELECT count(*) FROM candidates c LEFT JOIN gaps g ON g.id=c.gap_id WHERE g.id IS NULL").fetchone()[0]
        orphan_candidate_sources = db.execute(
            "SELECT count(*) FROM candidates c JOIN gaps g ON g.id=c.gap_id "
            "LEFT JOIN sources s ON s.id=g.source_id WHERE s.id IS NULL").fetchone()[0]
        ambiguous_targets = db.execute(
            "SELECT count(*) FROM ("
            "SELECT g.source_id,g.event_id FROM gaps g JOIN candidates c ON c.gap_id=g.id "
            "WHERE g.state='review_pending' GROUP BY g.source_id,g.event_id HAVING count(*)!=1 "
            "UNION ALL SELECT g.source_id,CAST(g.event_index AS TEXT) FROM gaps g "
            "JOIN candidates c ON c.gap_id=g.id WHERE g.state='review_pending' "
            "GROUP BY g.source_id,g.event_index HAVING count(*)!=1)").fetchone()[0]
        missing_exact_response = db.execute(
            "SELECT count(*) FROM candidates c JOIN gaps g ON g.id=c.gap_id "
            "WHERE g.state='review_pending' AND NOT EXISTS (SELECT 1 FROM responses r "
            "WHERE r.gap_id=c.gap_id AND r.stage='generate' AND r.text=c.text)").fetchone()[0]
        all_candidates_missing_exact_response = db.execute(
            "SELECT count(*) FROM candidates c WHERE NOT EXISTS (SELECT 1 FROM responses r "
            "WHERE r.gap_id=c.gap_id AND r.stage='generate' AND r.text=c.text)").fetchone()[0]
        candidate_gap_cardinality_failures = db.execute(
            "SELECT count(*) FROM (SELECT gap_id,count(*) AS n FROM candidates "
            "GROUP BY gap_id HAVING n!=1)").fetchone()[0]
        pending_without_candidate = db.execute(
            "SELECT count(*) FROM gaps g WHERE g.state='review_pending' "
            "AND NOT EXISTS (SELECT 1 FROM candidates c WHERE c.gap_id=g.id)").fetchone()[0]
        if (len(sources) != receipt["counts"]["sources"]
                or len({row["trace_id"] for row in sources}) != len(sources)
                or candidates != valid + generation_invalid_candidates
                or pending != valid or unexpected_state_candidates
                or receipt.get("gap_states") != gap_states
                or receipt.get("invalid_candidate_count_preserved") !=
                    generation_invalid_candidates
                or orphan_candidates or orphan_candidate_sources or ambiguous_targets
                or missing_exact_response or all_candidates_missing_exact_response
                or candidate_gap_cardinality_failures or pending_without_candidate
                or valid != receipt["counts"]["valid_candidates"]):
            raise ValueError("Frozen source/candidate membership differs from receipt")
        if db.execute("SELECT count(*) FROM gaps WHERE state IN ('generating','reviewing')").fetchone()[0]:
            raise ValueError("Freeze must come from the drained generation run")
    finally:
        db.close()
    source = Path(identity["source_path"]).resolve(strict=True)
    if (receipt.get("source_path") != identity.get("source_path")
            or receipt.get("source_sha256") != identity.get("source_sha256")
            or type(receipt.get("source_bytes")) is not int
            or receipt["source_bytes"] != source.stat().st_size):
        raise ValueError("Freeze receipt source identity differs from the snapshot metadata")
    signature = lambda: tuple(getattr(source.stat(), key) for key in
        ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"))
    before = signature()
    if sha_file(source) != identity["source_sha256"] or before != signature():
        raise ValueError("Original full source bytes changed")
    receipt = dict(receipt)
    receipt["journal_cardinality_audit"] = {
        "schema": "qwen38-frozen-generation-cardinality/v1", "verified": True,
        "review_pending_gaps": pending, "selected_candidates": valid,
        "all_candidates": candidates,
        "generation_invalid_candidates_excluded": generation_invalid_candidates,
        "unexpected_state_candidates": unexpected_state_candidates,
        "orphan_candidates": orphan_candidates,
        "orphan_candidate_sources": orphan_candidate_sources,
        "ambiguous_source_event_targets": ambiguous_targets,
        "selected_candidates_missing_exact_generate_response": missing_exact_response,
        "all_candidates_missing_exact_generate_response": all_candidates_missing_exact_response,
        "candidate_gap_cardinality_failures": candidate_gap_cardinality_failures,
        "review_pending_gaps_missing_candidate": pending_without_candidate,
        "frozen_gap_states": gap_states,
    }
    jobs = []
    for inventory in receipt["replay_manifests"]:
        if sha_file(inventory["path"]) != inventory["sha256"]:
            raise ValueError("Frozen original replay inventory changed")
    replay_jobs, inventories = replay.load_jobs([p["path"] for p in receipt["replay_manifests"]])
    for source_row in sources:
        jobs.append({"kind": "trace", "identity": source_row["trace_id"], "source": "trace", "source_row": source_row})
    jobs.extend({"kind": "replay", **job} for job in replay_jobs)
    jobs.sort(key=lambda job: hashlib.sha256((SEED + ":shuffle:" + job["identity"]).encode()).hexdigest())
    return receipt, identity, jobs


def verify_frozen_inputs(*, freeze_receipt, receipt, identity, jobs, tokenizer_dir,
                         renderer_identity, preparation_sha256, replay_adapter_sha256,
                         auxiliary_files, build_input_paths, dependency_sha256):
    """Close the multiprocess-build TOCTOU window before publishing metadata."""
    if (sha_file(freeze_receipt) != receipt["freeze_receipt_sha256_at_start"]
            or sha_file(receipt["snapshot_path"]) != receipt["snapshot_sha256"]
            or sha_file(identity["source_path"]) != identity["source_sha256"]
            or sha_file(__file__) != preparation_sha256
            or sha_file(replay.__file__) != replay_adapter_sha256
            or conversion_dependency_sha256() != dependency_sha256):
        raise ValueError("Frozen trace/journal/code input changed during export")
    for inventory in receipt["replay_manifests"]:
        if sha_file(inventory["path"]) != inventory["sha256"]:
            raise ValueError("Replay inventory changed during export")
    for job in jobs:
        if job["kind"] == "replay" and sha_file(job["path"]) != job["sha256"]:
            raise ValueError("Selected replay source changed during export")
    for path, expected in auxiliary_files:
        if sha_file(path) != expected:
            raise ValueError("Hash-bound export policy changed during export")
    from .masked import Qwen38TurnBoundaryMaskedRenderer
    if Qwen38TurnBoundaryMaskedRenderer(tokenizer_dir).identity != renderer_identity:
        raise ValueError("Tokenizer/template/renderer identity changed during export")
    verify_read_only_build_inputs(build_input_paths)
    return {
        "verified": True,
        "schema": "qwen38-end-of-build-input-stability/v1",
        "selected_replay_files_rehashed": sum(job["kind"] == "replay" for job in jobs),
        "replay_inventories_rehashed": len(receipt["replay_manifests"]),
        "snapshot_rehashed": True,
        "source_rehashed": True,
        "renderer_and_policy_inputs_rehashed": True,
    }


def select_baseline_replay(jobs, path, expected_sha):
    """Retain the exact baseline's replay representatives and real group IDs."""
    if path is None:
        if expected_sha is not None: raise ValueError('Selection SHA requires selection file')
        return jobs, None
    if not expected_sha or sha_file(path) != expected_sha:
        raise ValueError('Original baseline source selection changed')
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    selected = {row['identity']: row for row in rows if row['source'] == 'replay'}
    if len(selected) != sum(row['source'] == 'replay' for row in rows):
        raise ValueError('Duplicate baseline replay identity')
    frozen = {job['identity']: job for job in jobs if job['kind'] == 'replay'}
    if not selected or not set(selected) <= set(frozen):
        raise ValueError('Baseline replay selection is not a subset of the frozen inventory')
    kept = []
    for job in jobs:
        if job['kind'] != 'replay': kept.append(job); continue
        if job['identity'] not in selected: continue
        original = selected[job['identity']]
        if any(original.get(k) != job.get(k) for k in ('sha256','format')):
            raise ValueError('Baseline replay source bytes or representation changed')
        group = original.get('original_baseline_group_id')
        if not isinstance(group, str) or not group or original.get('original_baseline_split') != split_for_group(group):
            raise ValueError('Baseline replay session/split provenance is invalid')
        kept.append({**job, 'original_baseline_group_id': group,
                     'original_baseline_split': original['original_baseline_split']})
    return kept, {'policy':'exact-original-baseline-replay-selection/v2',
        'sha256':expected_sha,'input_replay_sources':len(frozen),'selected_replay_sources':len(selected),
        'excluded_replay_sources':len(frozen)-len(selected)}


def _dynamic_exclusion_record(result, source_row):
    """Return the complete text-free identity used by the exclusion allowlist."""
    return {
        "identity": result["identity"],
        "reason": result["reason"],
        "generated_gaps": result.get("generated_gaps", 0),
        "source_digest": source_row["source_digest"],
        "row_sha256": source_row["row_sha256"],
        # Duplicate exclusions are accepted only when the corrected converter
        # reproduces the exact earlier retained row and basis in deterministic
        # shuffled job order.  Non-duplicate exclusions carry explicit nulls so
        # the policy cannot silently acquire a different evidence shape.
        "duplicate_basis": result.get("duplicate_basis"),
        "duplicate_sha256": result.get("duplicate_sha256"),
        "duplicate_of_source": result.get("duplicate_of_source"),
        "duplicate_of_identity": result.get("duplicate_of_identity"),
        "native_failure_sha256": result.get("error_message_sha256"),
    }


def load_dynamic_exclusion_policy(path, expected_sha, receipt, jobs):
    """Load an exact, source-bound allowlist for intentional dynamic exclusions.

    The policy is external so a corrected full conversion cannot silently
    bless a newly appearing failure or duplicate.  It contains only hashes,
    identities, reasons and generated-gap counts; never trace bodies.
    """
    if path is None:
        if expected_sha is not None:
            raise ValueError("A dynamic exclusion digest requires its policy file")
        return {}, None
    path = Path(path).resolve(strict=True)
    if not expected_sha or sha_file(path) != expected_sha:
        raise ValueError("Dynamic trace exclusion policy hash changed")
    policy = json.loads(path.read_text())
    if (policy.get("schema") != DYNAMIC_EXCLUSION_SCHEMA
            or policy.get("snapshot_sha256") != receipt["snapshot_sha256"]
            or policy.get("source_sha256") != receipt["source_sha256"]
            or not isinstance(policy.get("records"), list)):
        raise ValueError("Dynamic exclusion policy is not bound to the frozen source")
    available = {job["identity"]: job["source_row"] for job in jobs if job["kind"] == "trace"}
    records = {}
    for record in policy["records"]:
        if not isinstance(record, dict) or set(record) != {
                "identity", "reason", "generated_gaps", "source_digest", "row_sha256",
                "duplicate_basis", "duplicate_sha256", "duplicate_of_source",
                "duplicate_of_identity", "native_failure_sha256"}:
            raise ValueError("Dynamic exclusion policy record has the wrong shape")
        identity = record["identity"]
        source_row = available.get(identity)
        if (not isinstance(identity, str) or not identity or identity in records
                or record["reason"] not in DYNAMIC_EXCLUSION_REASONS
                or type(record["generated_gaps"]) is not int or record["generated_gaps"] < 0
                or source_row is None
                or record["source_digest"] != source_row["source_digest"]
                or record["row_sha256"] != source_row["row_sha256"]):
            raise ValueError("Dynamic exclusion is not an exact frozen trace identity")
        duplicate_reason = record["reason"] in {
            "duplicate_normalized_messages", "duplicate_retained_arrays"}
        expected_basis = ({"duplicate_normalized_messages": "normalized_messages_sha256",
                           "duplicate_retained_arrays": "retained_arrays_sha256"}
                          .get(record["reason"]))
        if duplicate_reason:
            if (record["duplicate_basis"] != expected_basis
                    or not _hash(record["duplicate_sha256"])
                    or record["duplicate_of_source"] not in {"trace", "replay"}
                    or not isinstance(record["duplicate_of_identity"], str)
                    or not record["duplicate_of_identity"]):
                raise ValueError("Duplicate exclusion lacks its exact earlier retained identity")
        elif any(record[key] is not None for key in (
                "duplicate_basis", "duplicate_sha256", "duplicate_of_source",
                "duplicate_of_identity")):
            raise ValueError("Non-duplicate exclusion cannot carry duplicate evidence")
        native_reason = record["reason"] in {"UnsupportedMessage", "UnsupportedTrace"}
        if native_reason != _hash(record["native_failure_sha256"]):
            raise ValueError("Native exclusion requires its exact stable failure digest only")
        records[identity] = record
    if (policy.get("record_count") != len(records)
            or policy.get("generated_gap_count") != sum(
                record["generated_gaps"] for record in records.values())):
        raise ValueError("Dynamic exclusion policy aggregate changed")
    return records, {
        "path": str(path), "sha256": expected_sha,
        "schema": DYNAMIC_EXCLUSION_SCHEMA,
        "record_count": len(records),
        "generated_gap_count": sum(record["generated_gaps"] for record in records.values()),
    }


def init_worker(tokenizer_dir, snapshot, source_path, spool, all_train=False):
    global _renderer, _db, _identity, _source, _spool, _all_train
    if type(all_train) is not bool:
        raise ValueError("Explicit boolean train-only policy required")
    _all_train = all_train
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    from .masked import Qwen38TurnBoundaryMaskedRenderer
    _renderer = Qwen38TurnBoundaryMaskedRenderer(tokenizer_dir)
    _db = _readonly(snapshot)
    _identity = json.loads(_db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()[0])
    _source, _spool = Path(source_path), Path(spool)


def entries_for_source(db, source_row):
    entries = []
    for row in db.execute("SELECT g.* FROM gaps g JOIN candidates c ON c.gap_id=g.id WHERE g.source_id=? AND g.state='review_pending' ORDER BY g.event_index", (source_row["id"],)):
        gap = {**dict(row), **{k: source_row[k] for k in ("source_digest", "row_sha256", "trace_id")}}
        candidate = dict(db.execute("SELECT * FROM candidates WHERE gap_id=?", (gap["id"],)).fetchone())
        receipts = [dict(r) for r in db.execute(
            "SELECT * FROM responses WHERE gap_id=? AND stage='generate' AND text=? ORDER BY id DESC",
            (gap["id"], candidate["text"]))]
        if not receipts:
            raise ValueError("Frozen candidate has no genuine original generation response")
        # Latest exact-text response is subsequently fully checked against the
        # stored candidate metadata by generation_contract.bind_generations.
        entries.append({"gap": gap, "candidate": candidate, "generation_receipt": receipts[0]})
    return entries


def _trace_row(job):
    from .masked import training_row
    saved = job["source_row"]
    with _source.open("rb") as stream:
        stream.seek(saved["offset"])
        raw = stream.read(saved["length"])
    if hashlib.sha256(raw).hexdigest() != saved["row_sha256"]:
        raise ValueError("Source row bytes changed")
    source = validate_trace(json.loads(raw))
    if source["trace_id"] != saved["trace_id"] or digest(source) != saved["source_digest"]:
        raise ValueError("Original source identity changed")
    selection = source.get("metadata", {}).get("selection", {})
    group = selection.get("group_id")
    quality, confidence = selection.get("quality_score"), selection.get("overall_confidence_score")
    if (selection.get("capture_id") != source["trace_id"]
            or type(quality) not in (int, float) or not 4 <= quality <= 5
            or type(confidence) not in (int, float) or not 3 <= confidence <= 5):
        raise ValueError("Original source label does not meet the requested thresholds")
    original_group = group
    session_verified = isinstance(group, str) and bool(group)
    if not session_verified:
        if group not in (None, ""):
            raise ValueError("Original source session identity is missing")
        # A capture is not a recovered session. These rows stay training-only
        # even when verified session groups use a validation split.
        group = "capture-only:" + source["trace_id"]
    entries = entries_for_source(_db, saved)
    result = _renderer.render_generated_trace(source, entries, _identity["generator_config"])
    return training_row(result, group_id=group, source="trace", provenance={
        "source_row": saved, "quality_score": quality, "overall_confidence_score": confidence,
        "selection_sha256": digest(selection), "original_reasoning_used": False,
        "group_basis": "original_label_session" if session_verified else "capture_id",
        "session_identity_verified": session_verified, "original_session_group_id": original_group,
        "preparation_version": VERSION, "preparation_sha256": sha_file(__file__)})


def convert_job(job):
    try:
        if job["kind"] == "trace":
            row = _trace_row(job)
        else:
            result = replay.convert_job(job, renderer=_renderer)
            if not result["ok"]:
                return result
            row = result["row"]
        metadata = row["metadata"]
        audit, cot = metadata["sequence_audit"], metadata["cot_mask_audit"]
        parity = metadata['provenance']['generated_cot']['cross_arm_no_cot_parity']
        boundary = metadata['turn_boundary_audit']['semantic_action_boundary']
        tools = metadata['turn_boundary_audit']['raw_tool_cardinality']
        raw = canonical(row).encode() + b"\n"
        target = _spool / (hashlib.sha256((job["source"] + ":" + job["identity"]).encode()).hexdigest() + ".jsonl")
        with target.open("xb") as stream:
            stream.write(raw)
        converted = {"ok": True, "identity": job["identity"], "source": job["source"],
            "path": str(target), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
            "group_id": row["group_id"], "split": "train" if _all_train or metadata["provenance"].get("session_identity_verified") is False else split_for_group(row["group_id"]),
            "session_identity_unresolved": metadata["provenance"].get("session_identity_verified") is False,
            "source_digest": metadata["provenance"]["generated_cot"]["source_digest"],
            "normalized_messages_sha256": metadata['turn_boundary_audit']['output_messages_sha256'],
            "retained_arrays_sha256": hashlib.sha256(json.dumps([row['input_ids'],row['labels']],separators=(',',':')).encode()).hexdigest(),
            "input_tokens": len(row["input_ids"]), "target_tokens": sum(x != -100 for x in row["labels"]),
            "cross_arm_baseline_input_tokens": parity['baseline_input_tokens'],
            "cross_arm_zero_cot_input_tokens": parity['masked_zero_cot_input_tokens'],
            "cross_arm_baseline_target_tokens": parity['baseline_target_tokens'],
            "cross_arm_zero_cot_target_tokens": parity['masked_zero_cot_target_tokens'],
            "cross_arm_target_delta_vs_baseline": parity['actual_target_delta_vs_baseline'],
            "filled_gaps": cot["filled_gap_count"], "retained_filled_gaps": cot["retained_filled_gap_count"],
            "omitted_gaps":cot["omitted_generation_count"], "bound_input_gaps":cot["bound_input_generation_count"],
            "dropped_input_tokens": audit["dropped_input_tokens"],
            "dropped_target_tokens": audit["dropped_supervised_tokens"],
            "boundary_rows_verified": int(boundary["verified"]),
            "boundary_combined_visible_actions": boundary["combined_visible_action_messages"],
            "boundary_explicit_announcement_call_splits": boundary["explicitly_split_announcement_call_pairs"],
            "boundary_unbarriered_announcement_call_splits": boundary["unbarriered_announcement_call_splits"],
            "boundary_explicit_boundary_merges": boundary["explicit_boundary_merges"],
            "tool_rows_verified": int(tools["verified"]),
            "source_tool_calls": tools["source_tool_calls"],
            "native_structured_tool_calls": tools["native_structured_tool_calls"],
            "raw_tool_call_events": tools["raw_tool_call_events"],
            "intentionally_omitted_incomplete_terminal_tool_calls":
                tools["intentionally_omitted_incomplete_terminal_tool_calls"],
            "source_tool_results": tools["source_tool_results"],
            "native_tool_results": tools["native_tool_results"],
            "raw_tool_call_event_content_bytes": tools["raw_tool_call_event_content_bytes"],
            "raw_tool_call_event_content_projected_bytes": tools["raw_tool_call_event_content_projected_bytes"]}
        return converted
    except CrossArmParityError as exc:
        return {"ok": False, "identity": job["identity"], "source": job["source"],
            "reason": "cross_arm_no_cot_parity_failed", "cross_arm_parity_failure": True,
            "cross_arm_parity_audit": exc.audit,
            "generated_gaps": len(entries_for_source(_db, job["source_row"])) if job["kind"] == "trace" else 0}
    except MaskedBoundaryExclusion as exc:
        saved = job.get("source_row", {})
        count = _db.execute("SELECT count(*) FROM gaps g JOIN candidates c ON c.gap_id=g.id WHERE g.source_id=? AND g.state='review_pending'", (saved.get("id"),)).fetchone()[0] if job["kind"] == "trace" else 0
        if exc.audit["excluded_generated_gap_count"] != count:
            return {"ok": False, "identity": job["identity"], "source": job["source"],
                    "reason": "boundary_exclusion_generation_count_mismatch"}
        return {"ok": False, "identity": job["identity"], "source": job["source"],
            "reason": exc.code, "turn_boundary_exclusion": True,
            "generated_gaps": count, "boundary_exclusion_audit": exc.audit}
    except (UnsupportedTrace, UnsupportedMessage) as exc:
        saved = job.get("source_row", {})
        count = _db.execute("SELECT count(*) FROM gaps g JOIN candidates c ON c.gap_id=g.id WHERE g.source_id=? AND g.state='review_pending'", (saved.get("id"),)).fetchone()[0] if job["kind"] == "trace" else 0
        return {"ok": False, "identity": job["identity"], "source": job["source"],
            "reason": type(exc).__name__, "native_source_exclusion": True,
            "generated_gaps": count, "native_exclusion_policy": "unchanged-native-shape-guards-whole-source/v2",
            "source_digest": saved.get("source_digest"), "error_message_sha256": hashlib.sha256(str(exc).encode()).hexdigest()}
    except Exception as exc:
        # A compact category, never trace snippets or arbitrary exception text.
        frame = traceback.extract_tb(exc.__traceback__)[-1]
        return {"ok": False, "identity": job["identity"], "source": job["source"],
                "reason": type(exc).__name__,
                "error_location": Path(frame.filename).name + ":" + str(frame.lineno)}


def source_exclusion_policy(path, expected_sha, receipt, jobs, tokenizer_dir):
    """Allow only exact audited whole-source failures already excluded by baseline."""
    if not path:
        if expected_sha:
            raise ValueError("An exclusion digest requires its policy file")
        return {}, None
    if not expected_sha or sha_file(path) != expected_sha:
        raise ValueError("Source exclusion policy hash changed")
    policy = json.loads(Path(path).read_text())
    if (policy.get("schema") != "qwen38-baseline-native-source-exclusions/v1"
            or policy.get("snapshot_sha256") != receipt["snapshot_sha256"]
            or policy.get("source_sha256") != receipt["source_sha256"]
            or sha_file(policy["native_audit_path"]) != policy["native_audit_sha256"]
            or sha_file(policy["baseline_exclusions_path"]) != policy["baseline_exclusions_sha256"]):
        raise ValueError("Exclusions must bind the exact source, native audit and original baseline receipts")
    native = json.loads(Path(policy["native_audit_path"]).read_text())
    if (native.get("snapshot_sha256") != receipt["snapshot_sha256"]
            or native.get("source_sha256") != receipt["source_sha256"]
            or native.get("all_invalid_prior_baseline_excluded") is not True):
        raise ValueError("Native exclusion audit differs from the original source")
    baseline = {row["identity"]: row["reason"] for row in map(json.loads,
        Path(policy["baseline_exclusions_path"]).read_text().splitlines())}
    available = {j["identity"]: j["source_row"] for j in jobs if j["kind"] == "trace"}
    records = {}
    with _readonly(receipt["snapshot_path"]) as db:
        counts = dict(db.execute("SELECT g.source_id,count(*) FROM gaps g JOIN candidates c ON c.gap_id=g.id WHERE g.state='review_pending' GROUP BY g.source_id"))
    from training.qwen38_cot_masked.render import _canonical_messages
    from training.qwen38_no_cot.prepare_data import UnsupportedTrace
    from training.qwen38_no_cot.upstream_manager.render import UnsupportedMessage
    from .masked import Qwen38TurnBoundaryMaskedRenderer
    renderer = Qwen38TurnBoundaryMaskedRenderer(tokenizer_dir)
    for record in native["bad_sources"]:
        key = record["identity"]
        if (key in records or key not in available or record["source_id"] != available[key]["id"]
                or baseline.get(key) != record.get("prior_baseline_reason") or not baseline.get(key)
                or record.get("reason") not in {"UnsupportedTrace", "UnsupportedMessage"}
                or record["generated_gaps"] != counts.get(record["source_id"], 0)):
            raise ValueError("An exclusion is not an exact known-baseline native source failure")
        row = available[key]
        with Path(receipt["source_path"]).open("rb") as stream:
            stream.seek(row["offset"]);raw = stream.read(row["length"])
        if hashlib.sha256(raw).hexdigest() != row["row_sha256"]:
            raise ValueError("Excluded original source bytes changed")
        try:
            messages, _, _ = _canonical_messages(json.loads(raw)["events"], {})
            renderer._baseline._normalize(messages, None)
        except (UnsupportedTrace, UnsupportedMessage) as exc:
            if type(exc).__name__ != record["reason"]:
                raise ValueError("Rechecked source failure differs from its native audit") from exc
        else:
            raise ValueError("A natively renderable source cannot be excluded by this policy")
        records[key] = {**record, "row_sha256": available[key]["row_sha256"]}
    if (len(records) != native["invalid_sources"]
            or sum(r["generated_gaps"] for r in records.values()) != native["invalid_source_generated_gaps"]):
        raise ValueError("Excluded source/generation counts changed")
    return records, {"path": str(Path(path).resolve()), "sha256": expected_sha,
                     "native_audit_sha256": policy["native_audit_sha256"],
                     "baseline_exclusions_sha256": policy["baseline_exclusions_sha256"]}


def _prepare(*, freeze_receipt, tokenizer_dir, output, workers=32, shard_rows=256, limit=None, all_train=False,
            source_exclusions=None, source_exclusions_sha256=None,
            baseline_selection=None, baseline_selection_sha256=None,
            dynamic_trace_exclusions=None, dynamic_trace_exclusions_sha256=None):
    if type(workers) is not int or not 1 <= workers <= 96 or not 1 <= shard_rows <= 1024:
        raise ValueError("Bounded CPU worker/shard settings required")
    if type(all_train) is not bool:
        raise ValueError("Explicit boolean train-only policy required")
    if limit is None and all_train is not True:
        raise ValueError("The full manager-authorized mix requires --all-train")
    # V4 independently replays every frozen trace through the corrected
    # converter.  The earlier native-source policy was produced before the
    # tool-call repair and records only broad exception classes, so accepting
    # it would let 124 sources bypass the new exact failure digest audit.
    if source_exclusions is not None or source_exclusions_sha256 is not None:
        raise ValueError("V4 forbids inherited source exclusions; process every frozen trace")
    from . import masked_data as data
    from .masked import Qwen38TurnBoundaryMaskedRenderer
    receipt, identity, jobs = load_freeze(freeze_receipt)
    receipt = dict(receipt)
    receipt["freeze_receipt_sha256_at_start"] = sha_file(freeze_receipt)
    input_jobs = len(jobs)
    jobs, replay_selection = select_baseline_replay(jobs, baseline_selection, baseline_selection_sha256)
    dynamic_policy, dynamic_policy_identity = load_dynamic_exclusion_policy(
        dynamic_trace_exclusions, dynamic_trace_exclusions_sha256, receipt, jobs)
    audited_exclusions, exclusion_identity = {}, None
    if limit is not None:
        if type(limit) is not int or not 1 <= limit <= len(jobs):
            raise ValueError("Invalid finite preflight limit")
        # Genuine mixed shapes plus a filled source are tested by caller's
        # separate preflight; this limit is only a bounded export smoke.
        jobs = jobs[:limit]
    selected_trace_ids = {job["identity"] for job in jobs if job["kind"] == "trace"}
    expected_dynamic_exclusions = {
        identity: record for identity, record in dynamic_policy.items()
        if identity in selected_trace_ids
    }
    expected_replay_sources = sum(job["kind"] == "replay" for job in jobs)
    expected_job_keys = {(job["source"], job["identity"]) for job in jobs}
    if len(expected_job_keys) != len(jobs):
        raise ValueError("Frozen selected source jobs contain duplicate identities")
    expected_selected_trace_sources = sum(job["kind"] == "trace" for job in jobs)
    output = Path(output).resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    spool = output / "spool"
    spool.mkdir(mode=0o700)
    renderer_identity = Qwen38TurnBoundaryMaskedRenderer(tokenizer_dir).identity
    preparation_sha256 = sha_file(__file__)
    replay_adapter_sha256 = sha_file(replay.__file__)
    dependency_sha256 = conversion_dependency_sha256()
    auxiliary_files = [(path, expected) for path, expected in (
        (source_exclusions, source_exclusions_sha256),
        (baseline_selection, baseline_selection_sha256),
        (dynamic_trace_exclusions, dynamic_trace_exclusions_sha256),
    ) if path is not None]
    build_input_paths = [
        ("freeze receipt", freeze_receipt),
        ("generation snapshot", receipt["snapshot_path"]),
        ("original trace source", identity["source_path"]),
        ("tokenizer/template directory", tokenizer_dir),
        ("project code", Path(__file__).resolve().parents[2]),
    ]
    build_input_paths.extend(
        ("replay inventory " + str(index), item["path"])
        for index, item in enumerate(receipt["replay_manifests"]))
    build_input_paths.extend(
        ("selected replay " + job["identity"], job["path"])
        for job in jobs if job["kind"] == "replay")
    build_input_paths.extend(
        ("hash-bound policy " + Path(path).name, path) for path, _ in auxiliary_files)
    build_runtime = verify_read_only_build_inputs(build_input_paths)
    (output / "jobs.private.jsonl").write_text("".join(canonical(j) + "\n" for j in jobs))
    counts, tokens, exclusions, grouping, boundary_counts, tool_counts = (
        Counter(), Counter(), Counter(), Counter(), Counter(), Counter())
    boundary_excluded_sources, boundary_excluded_generations = Counter(), Counter()
    native_excluded_sources, native_excluded_generations = Counter(), Counter()
    duplicate_excluded_sources, duplicate_excluded_generations = Counter(), Counter()
    parity_excluded_sources, parity_excluded_generations = Counter(), Counter()
    replay_excluded_sources = Counter()
    dynamic_trace_records = {}
    trace_source_rows = {job["identity"]: job["source_row"] for job in jobs if job["kind"] == "trace"}
    # Hash -> (source kind, identity) for the first accepted job in the exact
    # deterministic order.  This proves why each later duplicate is excluded.
    seen_normalized, seen_retained = {}, {}
    unexpected_trace_failures = 0
    processed_job_keys, accepted_trace_keys, failed_trace_keys = set(), set(), set()
    shards = {"train": [], "validation": []}
    streams, row_counts, shard_hashes, shard_paths = {}, Counter(), {}, {}
    group_splits, digest_splits = {}, {}
    started = time.monotonic()
    def close_shard(split):
        if split in streams:
            stream = streams.pop(split)
            stream.flush();os.fsync(stream.fileno());stream.close()
            path = shard_paths.pop(split)
            entry = {"path": path.name, "sha256": shard_hashes.pop(split).hexdigest(),
                     "rows": row_counts.pop(split), "bytes": path.stat().st_size}
            os.chmod(path, 0o400)
            shards[split].append(entry)
            # Only closed/fsynced immutable shards enter this transfer feed.
            # Manifest/index/COMPLETE remain the final training admission gate.
            with (output / "finalized-shards.jsonl").open("a") as feed:
                feed.write(canonical({"split": split, **entry}) + "\n")
                feed.flush();os.fsync(feed.fileno())
    def progress(processed):
        record = {"jobs_processed": processed, "jobs_total": len(jobs), "elapsed_seconds": time.monotonic() - started,
            "counts": dict(counts), "token_counts": dict(tokens), "exclusions": dict(exclusions)}
        pending = output / "progress.json.tmp"
        pending.write_text(json.dumps(record, sort_keys=True) + "\n")
        pending.replace(output / "progress.json")
        print(json.dumps(record, sort_keys=True), flush=True)
    with (output / "row-receipts.jsonl").open("x") as rows_out, (output / "exclusions.jsonl").open("x") as errors:
        for record in audited_exclusions.values():
            errors.write(canonical({**record, "ok": False, "source": "trace",
                "preaudited_known_baseline_exclusion": True,
                "policy_sha256": exclusion_identity["sha256"]}) + "\n")
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker,
                initargs=(tokenizer_dir, receipt["snapshot_path"], identity["source_path"], str(spool), all_train)) as pool:
            pending, source = deque(), iter(jobs)
            for _ in range(workers * 2):
                job = next(source, None)
                if job is not None:
                    pending.append(pool.submit(convert_job, job))
            processed = 0
            while pending:
                result = pending.popleft().result()
                processed += 1
                result_key = (result.get("source"), result.get("identity"))
                if result_key not in expected_job_keys or result_key in processed_job_keys:
                    raise ValueError("Worker returned a missing, duplicate or substituted source identity")
                processed_job_keys.add(result_key)
                job = next(source, None)
                if job is not None:
                    pending.append(pool.submit(convert_job, job))
                if result['ok']:
                    normalized_sha = result['normalized_messages_sha256']
                    retained_sha = result['retained_arrays_sha256']
                    reason = ('duplicate_normalized_messages' if normalized_sha in seen_normalized
                              else 'duplicate_retained_arrays' if retained_sha in seen_retained else None)
                    if reason:
                        basis = ('normalized_messages_sha256' if reason == 'duplicate_normalized_messages'
                                 else 'retained_arrays_sha256')
                        duplicate_sha = result[basis]
                        duplicate_of_source, duplicate_of_identity = (
                            seen_normalized[duplicate_sha] if reason == 'duplicate_normalized_messages'
                            else seen_retained[duplicate_sha])
                        Path(result['path']).unlink()
                        result = {**result, 'ok':False, 'reason':reason, 'duplicate_source_exclusion':True,
                                  'generated_gaps':result['bound_input_gaps'],
                                  'duplicate_basis':basis, 'duplicate_sha256':duplicate_sha,
                                  'duplicate_of_source':duplicate_of_source,
                                  'duplicate_of_identity':duplicate_of_identity}
                    else:
                        owner = (result['source'], result['identity'])
                        seen_normalized[normalized_sha] = owner
                        seen_retained[retained_sha] = owner
                if not result["ok"]:
                    if result["source"] == "trace":
                        failed_trace_keys.add(result_key)
                    exclusions[result["source"] + "/" + result["reason"]] += 1
                    errors.write(canonical(result) + "\n")
                    if result["source"] == "replay":
                        replay_excluded_sources[result["reason"]] += 1
                    if result.get('cross_arm_parity_failure') is True:
                        parity_excluded_sources[result['source']] += 1
                        parity_excluded_generations[result['source']] += result.get('generated_gaps',0)
                    if result["source"] == "trace":
                        if result.get('cross_arm_parity_failure') is True:
                            unexpected_trace_failures += 1
                        elif result.get("turn_boundary_exclusion") is True:
                            boundary_excluded_sources[result["reason"]] += 1
                            boundary_excluded_generations[result["reason"]] += result["generated_gaps"]
                            dynamic_trace_records[result["identity"]] = _dynamic_exclusion_record(
                                result, trace_source_rows[result["identity"]])
                        elif result.get('duplicate_source_exclusion') is True:
                            duplicate_excluded_sources[result['reason']] += 1
                            duplicate_excluded_generations[result['reason']] += result['generated_gaps']
                            dynamic_trace_records[result["identity"]] = _dynamic_exclusion_record(
                                result, trace_source_rows[result["identity"]])
                        elif result.get("native_source_exclusion") is True:
                            native_excluded_sources[result["reason"]] += 1
                            native_excluded_generations[result["reason"]] += result["generated_gaps"]
                            dynamic_trace_records[result["identity"]] = _dynamic_exclusion_record(
                                result, trace_source_rows[result["identity"]])
                        else:
                            unexpected_trace_failures += 1
                else:
                    if result["source"] == "trace":
                        accepted_trace_keys.add(result_key)
                    split = result["split"]
                    for value, mapping in ((result["group_id"], group_splits), (result["source_digest"], digest_splits)):
                        if value in mapping and mapping[value] != split:
                            raise ValueError("Original group/source crosses training and validation")
                        mapping[value] = split
                    if split not in streams:
                        path = output / (split + "-" + str(len(shards[split])).zfill(5) + ".jsonl")
                        streams[split] = path.open("xb")
                        shard_paths[split], shard_hashes[split] = path, hashlib.sha256()
                    row_bytes = Path(result["path"]).read_bytes()
                    if hashlib.sha256(row_bytes).hexdigest() != result["sha256"]:
                        raise ValueError("Rendered row changed before shard assembly")
                    streams[split].write(row_bytes)
                    shard_hashes[split].update(row_bytes)
                    row_counts[split] += 1
                    if row_counts[split] >= shard_rows:
                        close_shard(split)
                    Path(result["path"]).unlink()
                    rows_out.write(canonical({k: v for k, v in result.items() if k != "path"}) + "\n")
                    counts[split + "/" + result["source"]] += 1
                    if result["session_identity_unresolved"]:
                        grouping["unresolved_session_sources"] += 1
                        grouping["unresolved_session_generated_gaps"] += result["bound_input_gaps"]
                    for key in ("input_tokens", "target_tokens", "cross_arm_baseline_input_tokens",
                            "cross_arm_zero_cot_input_tokens", "cross_arm_baseline_target_tokens",
                            "cross_arm_zero_cot_target_tokens", "cross_arm_target_delta_vs_baseline",
                            "filled_gaps", "retained_filled_gaps", "dropped_input_tokens",
                            "dropped_target_tokens", "omitted_gaps", "bound_input_gaps"):
                        tokens[key] += result[key]
                    for key in ("boundary_rows_verified", "boundary_combined_visible_actions",
                            "boundary_explicit_announcement_call_splits",
                            "boundary_unbarriered_announcement_call_splits",
                            "boundary_explicit_boundary_merges"):
                        boundary_counts[key] += result[key]
                    for key in ("tool_rows_verified", "source_tool_calls",
                            "native_structured_tool_calls", "raw_tool_call_events",
                            "intentionally_omitted_incomplete_terminal_tool_calls",
                            "source_tool_results",
                            "native_tool_results", "raw_tool_call_event_content_bytes",
                            "raw_tool_call_event_content_projected_bytes"):
                        tool_counts[key] += result[key]
                if processed % 32 == 0 or processed == len(jobs):
                    rows_out.flush(); errors.flush()
                    for stream in streams.values(): stream.flush()
                    progress(processed)
    for split in list(streams): close_shard(split)
    # No transient directory may survive into an admitted portable export.
    # Each worker row has already been assembled and unlinked at this point.
    spool.rmdir()
    if processed_job_keys != expected_job_keys:
        raise ValueError("Not every selected frozen source job produced exactly one outcome")
    # Every excluded original trace is material and explicit. A full export may
    # not silently claim all frozen generations were included if some failed.
    excluded_generations = (sum(r["generated_gaps"] for r in audited_exclusions.values())
        + sum(boundary_excluded_generations.values()) + sum(native_excluded_generations.values())
        + sum(duplicate_excluded_generations.values()) + sum(parity_excluded_generations.values()))
    complete_generations = tokens["filled_gaps"] == receipt["counts"]["valid_candidates"]
    all_filled_retain_tokens = tokens["retained_filled_gaps"] == tokens["filled_gaps"]
    accounted_generations = tokens["filled_gaps"] + tokens["omitted_gaps"] + excluded_generations == receipt["counts"]["valid_candidates"]
    accepted_rows=sum(counts.values())
    trace_source_membership = {
        "schema": "qwen38-exact-trace-source-membership/v1",
        "verified": (
            len(accepted_trace_keys) + len(failed_trace_keys) == expected_selected_trace_sources
            and not accepted_trace_keys & failed_trace_keys
            and (limit is not None or
                 receipt["counts"]["sources"] == len(audited_exclusions) + expected_selected_trace_sources)),
        "full_frozen_coverage_required": limit is None,
        "frozen_trace_sources": receipt["counts"]["sources"],
        "preaudited_excluded_trace_sources": len(audited_exclusions),
        "selected_trace_sources": expected_selected_trace_sources,
        "processed_trace_sources": len(accepted_trace_keys) + len(failed_trace_keys),
        "accepted_trace_sources": len(accepted_trace_keys),
        "failed_trace_sources": len(failed_trace_keys),
    }
    dynamic_candidate = {
        "schema": DYNAMIC_EXCLUSION_SCHEMA,
        "snapshot_sha256": receipt["snapshot_sha256"],
        "source_sha256": receipt["source_sha256"],
        "record_count": len(dynamic_trace_records),
        "generated_gap_count": sum(record["generated_gaps"] for record in dynamic_trace_records.values()),
        "records": [dynamic_trace_records[key] for key in sorted(dynamic_trace_records)],
    }
    dynamic_candidate_path = output / "dynamic-trace-exclusions.candidate.json"
    dynamic_candidate_path.write_text(json.dumps(dynamic_candidate, indent=2, sort_keys=True) + "\n")
    dynamic_verified = dynamic_trace_records == expected_dynamic_exclusions
    dynamic_audit = {
        "schema": DYNAMIC_EXCLUSION_SCHEMA,
        "verified": dynamic_verified,
        "expected_record_count": len(expected_dynamic_exclusions),
        "expected_generated_gap_count": sum(
            record["generated_gaps"] for record in expected_dynamic_exclusions.values()),
        "observed_record_count": len(dynamic_trace_records),
        "observed_generated_gap_count": dynamic_candidate["generated_gap_count"],
        "candidate_path": dynamic_candidate_path.name,
        "candidate_sha256": sha_file(dynamic_candidate_path),
        "allowlist_sha256": (dynamic_policy_identity or {}).get("sha256"),
    }
    accepted_replay_sources=sum(counts[split+"/replay"] for split in ("train","validation"))
    replay_membership={"schema":"qwen38-exact-replay-membership/v1",
        "verified":(accepted_replay_sources==expected_replay_sources
            and not sum(replay_excluded_sources.values())),
        "expected_selected_sources":expected_replay_sources,
        "accepted_sources":accepted_replay_sources,
        "excluded_sources":dict(replay_excluded_sources),
        "excluded_source_count":sum(replay_excluded_sources.values())}
    try:
        input_stability = verify_frozen_inputs(
            freeze_receipt=freeze_receipt, receipt=receipt, identity=identity, jobs=jobs,
            tokenizer_dir=tokenizer_dir, renderer_identity=renderer_identity,
            preparation_sha256=preparation_sha256,
            replay_adapter_sha256=replay_adapter_sha256,
            auxiliary_files=auxiliary_files, build_input_paths=build_input_paths,
            dependency_sha256=dependency_sha256)
    except Exception:
        (output / "BLOCKED.json").write_text(json.dumps({
            "reason": "input_changed_during_export",
            "freeze_receipt_sha256_at_start": receipt["freeze_receipt_sha256_at_start"],
        }, indent=2, sort_keys=True) + "\n")
        raise
    cross_arm_parity={"schema":"qwen38-cross-arm-corpus-parity/v1","verified":(
            accepted_rows>0 and not sum(parity_excluded_sources.values())
            and tokens['target_tokens']<=tokens['cross_arm_zero_cot_target_tokens']
            <=tokens['cross_arm_baseline_target_tokens']
            and tokens['cross_arm_target_delta_vs_baseline']==
                tokens['target_tokens']-tokens['cross_arm_baseline_target_tokens']),
        "rows_verified":accepted_rows,"adapter_divergence_rows":sum(parity_excluded_sources.values()),
        "target_inflation_limit_tokens":0,"actual_target_tokens":tokens['target_tokens'],
        "zero_cot_target_tokens":tokens['cross_arm_zero_cot_target_tokens'],
        "baseline_target_tokens":tokens['cross_arm_baseline_target_tokens'],
        "actual_target_delta_vs_baseline":tokens['target_tokens']-tokens['cross_arm_baseline_target_tokens'],
        "zero_cot_target_delta_vs_baseline":tokens['cross_arm_zero_cot_target_tokens']-tokens['cross_arm_baseline_target_tokens']}
    semantic_boundary_audit = {
        "schema": "qwen38-assistant-action-boundary-corpus-audit/v1",
        "verified": (boundary_counts["boundary_rows_verified"] == accepted_rows
                     and boundary_counts["boundary_unbarriered_announcement_call_splits"] == 0
                     and boundary_counts["boundary_explicit_boundary_merges"] == 0),
        "rows_verified": boundary_counts["boundary_rows_verified"],
        "combined_visible_action_messages": boundary_counts["boundary_combined_visible_actions"],
        "explicitly_split_announcement_call_pairs": boundary_counts["boundary_explicit_announcement_call_splits"],
        "unbarriered_announcement_call_splits": boundary_counts["boundary_unbarriered_announcement_call_splits"],
        "explicit_boundary_merges": boundary_counts["boundary_explicit_boundary_merges"],
    }
    raw_tool_cardinality_audit = {
        "schema": "qwen38-raw-tool-cardinality-corpus-audit/v1",
        "verified": (tool_counts["tool_rows_verified"] == accepted_rows
                     and tool_counts["source_tool_calls"] ==
                         tool_counts["native_structured_tool_calls"]
                     and tool_counts["raw_tool_call_events"] ==
                         tool_counts["source_tool_calls"]
                         + tool_counts["intentionally_omitted_incomplete_terminal_tool_calls"]
                     and tool_counts["source_tool_results"] == tool_counts["native_tool_results"]
                     and tool_counts["raw_tool_call_event_content_projected_bytes"] == 0),
        "rows_verified": tool_counts["tool_rows_verified"],
        "source_tool_calls": tool_counts["source_tool_calls"],
        "native_structured_tool_calls": tool_counts["native_structured_tool_calls"],
        "raw_tool_call_events": tool_counts["raw_tool_call_events"],
        "intentionally_omitted_incomplete_terminal_tool_calls":
            tool_counts["intentionally_omitted_incomplete_terminal_tool_calls"],
        "source_tool_results": tool_counts["source_tool_results"],
        "native_tool_results": tool_counts["native_tool_results"],
        "raw_tool_call_event_content_bytes": tool_counts["raw_tool_call_event_content_bytes"],
        "raw_tool_call_event_content_projected_bytes":
            tool_counts["raw_tool_call_event_content_projected_bytes"],
    }
    manifest = {"schema": data.MANIFEST_SCHEMA, "training_contract": data.TRAINING_CONTRACT,
        "version": VERSION, "renderer_identity": renderer_identity, "splits": {k: v for k, v in shards.items() if v},
        "experimental": True, "acceptance_policy": data.ACCEPTANCE_POLICY, "reviewed": False,
        "semantic_review_required": False, "semantic_quality_qualified": False, "future_information_checked": False,
        "unreviewed_gap_count": tokens["filled_gaps"], "frozen_valid_candidate_count": receipt["counts"]["valid_candidates"],
        "included_valid_candidate_count": tokens["filled_gaps"], "excluded_valid_candidate_count": excluded_generations,
        "omitted_valid_candidate_count":tokens["omitted_gaps"],
        "gap_omission_policy":"keep-native-leading-gap-omit-only-conflicting-cot/v4",
        "all_frozen_generations_accounted_for": accounted_generations,
        "excluded_source_count": len(audited_exclusions) + sum(boundary_excluded_sources.values()) + sum(native_excluded_sources.values()) + sum(duplicate_excluded_sources.values()) + sum(parity_excluded_sources.values()) + sum(replay_excluded_sources.values()),
        "duplicate_source_exclusion_policy":"normalized-messages-or-retained-arrays-whole-source/v2",
        "duplicate_excluded_sources":dict(duplicate_excluded_sources),
        "duplicate_excluded_generations":dict(duplicate_excluded_generations),
        "native_render_excluded_sources": dict(native_excluded_sources),
        "native_render_excluded_generations": dict(native_excluded_generations),
        "native_render_exclusion_policy": "unchanged-native-shape-guards-whole-source/v2",
        "cross_arm_parity_excluded_sources":dict(parity_excluded_sources),
        "cross_arm_parity_excluded_generations":dict(parity_excluded_generations),
        "turn_boundary_excluded_sources": dict(boundary_excluded_sources),
        "turn_boundary_excluded_generations": dict(boundary_excluded_generations),
        "turn_boundary_exclusion_policy": "whole-source-exclusion-no-cot-relocation/v2",
        "unexpected_trace_failures": unexpected_trace_failures,
        "eligible_source_count": receipt["counts"]["sources"] - len(audited_exclusions) - sum(boundary_excluded_sources.values()) - sum(native_excluded_sources.values()) - sum(duplicate_excluded_sources.values()) - parity_excluded_sources['trace'],
        "source_exclusion_policy": exclusion_identity,
        "dynamic_trace_exclusion_policy": dynamic_policy_identity,
        "dynamic_trace_exclusion_audit": dynamic_audit,
        "baseline_replay_selection": replay_selection,
        "trace_source_membership": trace_source_membership,
        "replay_source_membership":replay_membership,
        "end_of_build_input_stability": input_stability,
        "build_runtime": build_runtime,
        "all_frozen_generations_bound_before_truncation": complete_generations,
        "all_frozen_generations_included_before_cutoff": complete_generations and all_filled_retain_tokens,
        "bound_generated_gaps_wholly_after_cutoff": tokens["filled_gaps"] - tokens["retained_filled_gaps"],
        "all_bound_generated_gaps_retain_at_least_one_token": all_filled_retain_tokens,
        "full_export": limit is None, "input_source_jobs": input_jobs, "source_jobs": len(jobs),
        "counts": dict(counts), "token_counts": dict(tokens),
        "cross_arm_no_cot_parity":cross_arm_parity,
        "semantic_action_boundary_audit": semantic_boundary_audit,
        "raw_tool_cardinality_audit": raw_tool_cardinality_audit,
        "exclusions": dict(exclusions), "seed": SEED,
        "shuffle": "source-identity-seeded-sha256-order",
        "group_split": ALL_TRAIN_POLICY if all_train else "identical-baseline-sha256-seed-group-mod100",
        "all_train": all_train, "internal_validation": not all_train,
        "unknown_session_policy": "capture-identity-train-only-no-holdout-claim/v2",
        "session_grouping_complete": grouping["unresolved_session_sources"] == 0,
        "unresolved_session_sources": grouping["unresolved_session_sources"],
        "unresolved_session_generated_gaps": grouping["unresolved_session_generated_gaps"],
        "freeze_receipt_sha256": sha_file(freeze_receipt), "generation_snapshot_sha256": receipt["snapshot_sha256"],
        "source_sha256": identity["source_sha256"], "preparation_sha256": sha_file(__file__),
        "conversion_dependency_sha256": dependency_sha256,
        "journal_cardinality_audit": receipt["journal_cardinality_audit"],
        "replay_adapter_sha256": sha_file(replay.__file__), "elapsed_seconds": time.monotonic() - started}
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    parity_failed = (not cross_arm_parity['verified'] or not semantic_boundary_audit['verified']
                     or not raw_tool_cardinality_audit['verified']
                     or cross_arm_parity['adapter_divergence_rows'] != 0)
    replay_failed = not replay_membership['verified']
    dynamic_failed = not dynamic_verified
    coverage_failed = (not trace_source_membership["verified"] or unexpected_trace_failures
                       or (limit is None and not accounted_generations))
    if parity_failed or replay_failed or dynamic_failed or coverage_failed:
        reason = ("cross_arm_no_cot_parity_failed" if parity_failed
                  else "exact_replay_source_membership_failed" if replay_failed
                  else "dynamic_trace_exclusion_allowlist_mismatch" if dynamic_failed
                  else "original_trace_or_generation_coverage_incomplete")
        (output / "BLOCKED.json").write_text(json.dumps({"reason": reason,
            "manifest_sha256": sha_file(manifest_path)}, indent=2) + "\n")
        if parity_failed:
            raise ValueError("Cross-arm no-CoT parity failed; dataset is blocked from training")
        if replay_failed:
            raise ValueError("Exact selected replay membership failed; dataset is blocked from training")
        if dynamic_failed:
            raise ValueError("Dynamic trace exclusions differ from the exact allowlist; dataset is blocked from training")
        raise ValueError("Full export has explicit original-trace/generation exclusions; fix before training")
    index = data.build_index(manifest_path, output / "index.json", workers=min(workers,32))
    final = {"manifest": str(manifest_path), "manifest_sha256": sha_file(manifest_path), "index": index,
             "counts": dict(counts), "token_counts": dict(tokens)}
    final["complete"] = data.publish_complete(manifest_path, index, output / "COMPLETE.json")
    print(json.dumps(final, sort_keys=True), flush=True)
    return final


def prepare(*, freeze_receipt, tokenizer_dir, output, workers=32, shard_rows=256, limit=None, all_train=False,
            source_exclusions=None, source_exclusions_sha256=None,
            baseline_selection=None, baseline_selection_sha256=None,
            dynamic_trace_exclusions=None, dynamic_trace_exclusions_sha256=None):
    """Run one fresh export and leave a sanitized atomic BLOCKED receipt on failure."""
    arguments = dict(freeze_receipt=freeze_receipt, tokenizer_dir=tokenizer_dir,
        output=output, workers=workers, shard_rows=shard_rows, limit=limit,
        all_train=all_train, source_exclusions=source_exclusions,
        source_exclusions_sha256=source_exclusions_sha256,
        baseline_selection=baseline_selection,
        baseline_selection_sha256=baseline_selection_sha256,
        dynamic_trace_exclusions=dynamic_trace_exclusions,
        dynamic_trace_exclusions_sha256=dynamic_trace_exclusions_sha256)
    try:
        return _prepare(**arguments)
    except BaseException as error:
        destination = Path(output).resolve()
        blocked = destination / "BLOCKED.json"
        complete = destination / "COMPLETE.json"
        if destination.is_dir() and not complete.exists() and not blocked.exists():
            payload = {"schema": "qwen38-native-gap-v4-blocked/v1", "status": "blocked",
                "reason": "build_failed_before_atomic_completion",
                "error_type": type(error).__name__}
            pending = destination / (".BLOCKED.json.tmp-" + str(os.getpid()))
            with pending.open("x") as stream:
                stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
                stream.flush(); os.fsync(stream.fileno())
            pending.replace(blocked)
            descriptor = os.open(destination, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-receipt", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--shard-rows", type=int, default=256)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--all-train", action="store_true",
                        help="Train-only experiment; retain missing-session captures without claiming session identities or a holdout")
    parser.add_argument("--source-exclusions", help="Exact known-baseline native-source exclusion policy")
    parser.add_argument("--source-exclusions-sha256")
    parser.add_argument("--baseline-selection", help="Hash-bound exact original baseline source selection")
    parser.add_argument("--baseline-selection-sha256")
    parser.add_argument("--dynamic-trace-exclusions",
                        help="Exact source-bound allowlist for intentional boundary/native/dedup exclusions")
    parser.add_argument("--dynamic-trace-exclusions-sha256")
    args = parser.parse_args()
    prepare(**vars(args))


if __name__ == "__main__":
    main()
