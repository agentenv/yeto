"""Independently audit and promote a blocked v4 dynamic-exclusion candidate.

The candidate builder is discovery only.  This helper re-derives the frozen job
set, verifies every candidate exclusion from immutable source evidence, and
re-renders every excluded trace.  Only the exact candidate bytes are promoted;
the helper never edits a candidate dataset or a frozen input.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sqlite3
import uuid

from training.qwen38_no_cot.prepare_data import SEED


IMAGE = "sha256:b0bd0e50b29cb0a3ec69c3cbbb49652d3c1aabedbfe22ee0fa6d053b6bb33aee"
ROOT = Path("/data/sft_baseline_20260908")
POLICY_SCHEMA = "qwen38-exact-dynamic-trace-exclusions/v2"
PROMOTION_SCHEMA = "qwen38-native-gap-v4-dynamic-exclusion-promotion/v1"
PROMOTION_RECEIPT_KEYS = {
    "schema", "status", "at", "output_dir", "policy_path", "runtime_image",
    "candidate_dir", "candidate_build_receipt_path",
    "candidate_build_receipt_sha256", "candidate_tree_sha256",
    "candidate_build_core_sha256", "candidate_manifest_sha256",
    "candidate_blocked_sha256", "candidate_jobs_sha256", "candidate_jobs_count",
    "candidate_exclusions_sha256", "candidate_exclusions_count",
    "candidate_row_receipts_sha256", "candidate_row_receipts_count",
    "candidate_policy_path", "candidate_policy_sha256",
    "candidate_record_set_sha256", "freeze_receipt_path", "freeze_receipt_sha256",
    "snapshot_path", "snapshot_sha256", "source_path", "source_sha256",
    "source_bytes", "baseline_selection_path", "baseline_selection_sha256",
    "code_manifest_path", "code_manifest_sha256", "promotion_helper_sha256",
    "renderer_identity", "generator_identity_sha256", "policy_schema",
    "policy_sha256", "policy_bytes", "record_count", "generated_gap_count",
    "reason_counts", "selected_source_jobs", "input_source_jobs",
    "candidate_non_dynamic_gates", "all_non_dynamic_gates_passed",
    "candidate_bytes_promoted_unchanged", "independent_record_derivation",
    "independent_rerender", "independent_journal_cardinality_audit",
    "replay_manifests", "runtime_code_file_count", "elapsed_seconds",
}
ALLOWED_REASONS = {
    "UnsupportedMessage", "UnsupportedTrace", "unrepresentable_text_after_tool",
    "duplicate_normalized_messages", "duplicate_retained_arrays",
}
RECORD_KEYS = {
    "identity", "reason", "generated_gaps", "source_digest", "row_sha256",
    "duplicate_basis", "duplicate_sha256", "duplicate_of_source",
    "duplicate_of_identity", "native_failure_sha256",
}

_renderer = _database = _source = _generator_config = None


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _hash(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _strict_json(raw, label):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(label + " contains a duplicate JSON key")
            result[key] = value
        return result
    def constant(value):
        raise ValueError(label + " contains a non-finite number: " + value)
    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(label + " is not strict JSON") from error


def _strict_json_file(path, label):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(label + " must be a regular file")
    return _strict_json(path.read_bytes(), label)


def _strict_jsonl(path, label, *, require_final_newline=False,
                  require_canonical=True):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(label + " must be a regular file")
    raw = path.read_bytes()
    if not raw or (require_final_newline and not raw.endswith(b"\n")):
        raise ValueError(label + " must be a nonempty JSONL file with the required terminator")
    rows = []
    for index, line in enumerate(raw.splitlines(), 1):
        if not line:
            raise ValueError(label + " contains a blank line")
        row = _strict_json(line, f"{label} line {index}")
        if (not isinstance(row, dict)
                or (require_canonical and canonical(row).encode() != line)):
            raise ValueError(label + " is not canonical JSONL")
        rows.append(row)
    return rows


def _signature(path):
    value = Path(path).stat()
    return tuple(getattr(value, key) for key in
                 ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"))


def _tree_inventory(root):
    root = Path(root)
    members = sorted(root.rglob("*"))
    if any(path.is_symlink() or not (path.is_file() or path.is_dir()) for path in members):
        raise ValueError("Candidate tree contains a non-regular member")
    if any(path.stat().st_mode & 0o222 for path in [root, *members]):
        raise ValueError("Candidate tree is not immutable by mode")
    records = []
    for path in members:
        if path.is_file():
            before = _signature(path)
            record = {"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                      "sha256": sha256(path)}
            if before != _signature(path):
                raise ValueError("Candidate member changed while hashing")
            records.append(record)
    if len({record["path"].casefold() for record in records}) != len(records):
        raise ValueError("Candidate tree has case-colliding names")
    return records


def _tree_sha(records):
    return hashlib.sha256(json.dumps(records, sort_keys=True,
                        separators=(",", ":")).encode()).hexdigest()


def _verify_code_tree(root, expected_sha):
    root = Path(root)
    manifest_path = root / "code-manifest.json"
    if sha256(manifest_path) != expected_sha:
        raise ValueError("Runtime code manifest changed")
    manifest = _strict_json_file(manifest_path, "runtime code manifest")
    if manifest.get("schema") != "qwen38-native-gap-v4-runtime-code/v1":
        raise ValueError("Wrong runtime code schema")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("Runtime code manifest is empty")
    expected, folded = {}, set()
    for record in records:
        name = record.get("path") if isinstance(record, dict) else None
        relative = PurePosixPath(name) if isinstance(name, str) else None
        if (set(record) != {"path", "bytes", "sha256"} or relative is None
                or relative.is_absolute() or "\\" in name
                or any(part in {"", ".", ".."} for part in relative.parts)
                or name.casefold() in folded or type(record.get("bytes")) is not int
                or record["bytes"] < 1 or not _hash(record.get("sha256"))):
            raise ValueError("Unsafe runtime code manifest member")
        expected[name] = record
        folded.add(name.casefold())
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    if actual != set(expected) | {"code-manifest.json"}:
        raise ValueError("Runtime code tree membership changed")
    for name, record in expected.items():
        path = root / name
        if (path.is_symlink() or path.stat().st_size != record["bytes"]
                or sha256(path) != record["sha256"]):
            raise ValueError("Runtime code member changed: " + name)
    helper = "monitoring/promote_masked_native_gap_v4_exclusions_20260911.py"
    if helper not in expected or expected[helper]["sha256"] != sha256(__file__):
        raise ValueError("Promotion helper is not bound into the reviewed code manifest")
    return manifest


def _on_read_only_mount(path):
    target = str(Path(path).resolve(strict=True))
    mountinfo = Path("/proc/self/mountinfo")
    if not mountinfo.is_file():
        return False
    matches = []
    for line in mountinfo.read_text().splitlines():
        before, separator, _ = line.partition(" - ")
        fields = before.split()
        if not separator or len(fields) < 6:
            continue
        mount = fields[4].replace("\\040", " ").replace("\\134", "\\")
        if target == mount or target.startswith(mount.rstrip("/") + "/"):
            matches.append((len(mount), set(fields[5].split(","))))
    return bool(matches) and "ro" in max(matches, key=lambda item: item[0])[1]


def _split_for_group(group):
    value = hashlib.sha256((SEED + ":split:" + group).encode()).hexdigest()
    return "validation" if int(value[:8], 16) % 100 == 0 else "train"


def _load_replay_jobs(receipt, baseline_selection):
    jobs, seen = [], set()
    inventories = receipt.get("replay_manifests")
    if not isinstance(inventories, list):
        raise ValueError("Freeze receipt replay inventory is invalid")
    for inventory in inventories:
        if (not isinstance(inventory, dict) or not isinstance(inventory.get("path"), str)
                or not _hash(inventory.get("sha256"))
                or sha256(inventory["path"]) != inventory["sha256"]):
            raise ValueError("Replay inventory changed")
        for lineno, job in enumerate(_strict_jsonl(
                inventory["path"], "replay inventory", require_canonical=False), 1):
            signature = job.get("sha256")
            if (job.get("source") != "replay" or job.get("format") not in {"rollout", "atif"}
                    or not _hash(signature) or job.get("identity") != "replay:" + signature):
                raise ValueError("Replay inventory has an invalid job")
            path = Path(job.get("path", ""))
            if path.is_symlink() or not path.is_file() or sha256(path) != signature:
                raise ValueError("Replay source bytes changed")
            if signature in seen:
                continue
            seen.add(signature)
            jobs.append({**job, "path": str(path.resolve(strict=True)),
                         "source_jobs_sha256": inventory["sha256"],
                         "source_jobs_line": lineno})
    selection_rows = _strict_jsonl(baseline_selection, "baseline selection")
    selected = {}
    for row in selection_rows:
        if row.get("source") != "replay":
            continue
        identity = row.get("identity")
        if not isinstance(identity, str) or identity in selected:
            raise ValueError("Baseline selection has a duplicate replay identity")
        selected[identity] = row
    frozen = {job["identity"]: job for job in jobs}
    if not selected or not set(selected) <= set(frozen):
        raise ValueError("Baseline selection is not an exact nonempty replay subset")
    kept = []
    for job in jobs:
        row = selected.get(job["identity"])
        if row is None:
            continue
        if any(row.get(key) != job.get(key) for key in ("sha256", "format")):
            raise ValueError("Selected replay source identity changed")
        group = row.get("original_baseline_group_id")
        if (not isinstance(group, str) or not group
                or row.get("original_baseline_split") != _split_for_group(group)):
            raise ValueError("Selected replay grouping provenance is invalid")
        kept.append({"kind": "replay", **job, "original_baseline_group_id": group,
                     "original_baseline_split": row["original_baseline_split"]})
    return jobs, kept


def _load_frozen_jobs(freeze_receipt, baseline_selection, baseline_selection_sha256):
    if sha256(baseline_selection) != baseline_selection_sha256:
        raise ValueError("Baseline replay selection changed")
    receipt = _strict_json_file(freeze_receipt, "freeze receipt")
    if receipt.get("schema") != "qwen38-frozen-generation-journal/v1":
        raise ValueError("Wrong freeze receipt")
    snapshot = Path(receipt.get("snapshot_path", ""))
    source = Path(receipt.get("source_path", ""))
    original = Path(receipt.get("original_journal", ""))
    snapshot_before = _signature(snapshot) if snapshot.is_file() else None
    source_before = _signature(source) if source.is_file() else None
    if (snapshot.is_symlink() or source.is_symlink() or not snapshot.is_file()
            or not source.is_file() or sha256(snapshot) != receipt.get("snapshot_sha256")
            or source.stat().st_size != receipt.get("source_bytes")
            or sha256(source) != receipt.get("source_sha256")
            or not isinstance(receipt.get("original_journal"), str)
            or not original.is_absolute()
            or snapshot_before != _signature(snapshot) or source_before != _signature(source)
            or snapshot.resolve(strict=True) == original.resolve()):
        raise ValueError("Frozen generation source changed")
    database = sqlite3.connect(snapshot.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    database.row_factory = sqlite3.Row
    try:
        identity_row = database.execute(
            "SELECT value FROM meta WHERE key='identity'").fetchone()
        if identity_row is None:
            raise ValueError("Frozen generation identity is missing")
        identity = _strict_json(identity_row[0].encode(), "frozen generation identity")
        sources = [dict(row) for row in database.execute("SELECT * FROM sources ORDER BY id")]
        states = dict(database.execute("SELECT state,count(*) FROM gaps GROUP BY state ORDER BY state"))
        candidates = database.execute("SELECT count(*) FROM candidates").fetchone()[0]
        pending = database.execute(
            "SELECT count(*) FROM gaps WHERE state='review_pending'").fetchone()[0]
        valid = database.execute(
            "SELECT count(*) FROM candidates c JOIN gaps g ON g.id=c.gap_id "
            "WHERE g.state='review_pending'").fetchone()[0]
        invalid = database.execute(
            "SELECT count(*) FROM candidates c JOIN gaps g ON g.id=c.gap_id "
            "WHERE g.state='generation_invalid'").fetchone()[0]
        unexpected = database.execute(
            "SELECT count(*) FROM candidates c JOIN gaps g ON g.id=c.gap_id "
            "WHERE g.state NOT IN ('review_pending','generation_invalid')").fetchone()[0]
        orphan_candidates = database.execute(
            "SELECT count(*) FROM candidates c LEFT JOIN gaps g ON g.id=c.gap_id "
            "WHERE g.id IS NULL").fetchone()[0]
        orphan_candidate_sources = database.execute(
            "SELECT count(*) FROM candidates c JOIN gaps g ON g.id=c.gap_id "
            "LEFT JOIN sources s ON s.id=g.source_id WHERE s.id IS NULL").fetchone()[0]
        ambiguous_targets = database.execute(
            "SELECT count(*) FROM ("
            "SELECT g.source_id,g.event_id FROM gaps g JOIN candidates c ON c.gap_id=g.id "
            "WHERE g.state='review_pending' GROUP BY g.source_id,g.event_id HAVING count(*)!=1 "
            "UNION ALL SELECT g.source_id,CAST(g.event_index AS TEXT) FROM gaps g "
            "JOIN candidates c ON c.gap_id=g.id WHERE g.state='review_pending' "
            "GROUP BY g.source_id,g.event_index HAVING count(*)!=1)").fetchone()[0]
        missing_exact_response = database.execute(
            "SELECT count(*) FROM candidates c JOIN gaps g ON g.id=c.gap_id "
            "WHERE g.state='review_pending' AND NOT EXISTS (SELECT 1 FROM responses r "
            "WHERE r.gap_id=c.gap_id AND r.stage='generate' AND r.text=c.text)").fetchone()[0]
        all_candidates_missing_exact_response = database.execute(
            "SELECT count(*) FROM candidates c WHERE NOT EXISTS (SELECT 1 FROM responses r "
            "WHERE r.gap_id=c.gap_id AND r.stage='generate' AND r.text=c.text)").fetchone()[0]
        candidate_gap_cardinality_failures = database.execute(
            "SELECT count(*) FROM (SELECT gap_id,count(*) AS n FROM candidates "
            "GROUP BY gap_id HAVING n!=1)").fetchone()[0]
        pending_without_candidate = database.execute(
            "SELECT count(*) FROM gaps g WHERE g.state='review_pending' "
            "AND NOT EXISTS (SELECT 1 FROM candidates c WHERE c.gap_id=g.id)").fetchone()[0]
        inflight = database.execute(
            "SELECT count(*) FROM gaps WHERE state IN ('generating','reviewing')").fetchone()[0]
        gap_counts = dict(database.execute(
            "SELECT g.source_id,count(*) FROM gaps g JOIN candidates c ON c.gap_id=g.id "
            "WHERE g.state='review_pending' GROUP BY g.source_id"))
    finally:
        database.close()
    counts = receipt.get("counts", {})
    valid_sources = all(
        type(row.get("id")) is int and row["id"] > 0
        and isinstance(row.get("trace_id"), str) and bool(row["trace_id"])
        and type(row.get("offset")) is int and row["offset"] >= 0
        and type(row.get("length")) is int and row["length"] > 0
        and row["offset"] + row["length"] <= receipt.get("source_bytes", -1)
        and _hash(row.get("row_sha256")) and _hash(row.get("source_digest"))
        for row in sources)
    if (digest(identity) != receipt.get("original_identity_sha256")
            or identity.get("source_path") != str(source)
            or identity.get("source_sha256") != receipt.get("source_sha256")
            or len(sources) != counts.get("sources")
            or not valid_sources or len({row.get("id") for row in sources}) != len(sources)
            or len({row.get("trace_id") for row in sources}) != len(sources)
            or states != receipt.get("gap_states") or pending != valid
            or valid != counts.get("valid_candidates")
            or invalid != receipt.get("invalid_candidate_count_preserved")
            or candidates != valid + invalid or unexpected or orphan_candidates
            or orphan_candidate_sources or ambiguous_targets or missing_exact_response
            or all_candidates_missing_exact_response or candidate_gap_cardinality_failures
            or pending_without_candidate or inflight):
        raise ValueError("Frozen journal cardinality or identity changed")
    replay_all, replay_selected = _load_replay_jobs(receipt, baseline_selection)
    jobs = [{"kind": "trace", "identity": row["trace_id"], "source": "trace",
             "source_row": row} for row in sources]
    jobs.extend(replay_selected)
    jobs.sort(key=lambda job: hashlib.sha256(
        (SEED + ":shuffle:" + job["identity"]).encode()).hexdigest())
    receipt = dict(receipt)
    receipt["independent_journal_cardinality_audit"] = {
        "schema": "qwen38-frozen-generation-cardinality/v1",
        "verified": True,
        "review_pending_gaps": pending, "selected_candidates": valid,
        "all_candidates": candidates,
        "generation_invalid_candidates_excluded": invalid,
        "unexpected_state_candidates": unexpected,
        "orphan_candidates": orphan_candidates,
        "orphan_candidate_sources": orphan_candidate_sources,
        "ambiguous_source_event_targets": ambiguous_targets,
        "selected_candidates_missing_exact_generate_response": missing_exact_response,
        "all_candidates_missing_exact_generate_response": all_candidates_missing_exact_response,
        "candidate_gap_cardinality_failures": candidate_gap_cardinality_failures,
        "review_pending_gaps_missing_candidate": pending_without_candidate,
        "frozen_gap_states": states,
    }
    stability_inputs = [
        (str(snapshot), receipt["snapshot_sha256"]),
        (str(source), receipt["source_sha256"]),
        *((inventory["path"], inventory["sha256"])
          for inventory in receipt["replay_manifests"]),
        *((job["path"], job["sha256"]) for job in replay_all),
    ]
    return (receipt, identity, jobs, len(sources) + len(replay_all), gap_counts,
            stability_inputs)


def _manifest_gates(manifest, candidate_path, jobs, input_jobs, receipt):
    trace = manifest.get("trace_source_membership", {})
    replay = manifest.get("replay_source_membership", {})
    journal = manifest.get("journal_cardinality_audit", {})
    dynamic = manifest.get("dynamic_trace_exclusion_audit", {})
    gates = {
        "full_export": manifest.get("full_export") is True,
        "all_train": manifest.get("all_train") is True,
        "internal_validation": manifest.get("internal_validation") is False,
        "train_only_split": set(manifest.get("splits", {})) == {"train"},
        "no_unexpected_trace_failures": manifest.get("unexpected_trace_failures") == 0,
        "all_generations_accounted": manifest.get("all_frozen_generations_accounted_for") is True,
        "cross_arm_parity": (manifest.get("cross_arm_no_cot_parity", {}).get("verified") is True
                             and manifest.get("cross_arm_no_cot_parity", {}).get(
                                 "adapter_divergence_rows") == 0),
        "action_boundary": manifest.get("semantic_action_boundary_audit", {}).get("verified") is True,
        "raw_tool_cardinality": manifest.get("raw_tool_cardinality_audit", {}).get("verified") is True,
        "trace_membership": (trace.get("verified") is True
                             and trace.get("full_frozen_coverage_required") is True
                             and trace.get("preaudited_excluded_trace_sources") == 0
                             and trace.get("processed_trace_sources") ==
                             trace.get("accepted_trace_sources", 0) + trace.get("failed_trace_sources", 0)),
        "replay_membership": (replay.get("verified") is True
                              and replay.get("expected_selected_sources") == replay.get("accepted_sources")
                              and replay.get("excluded_source_count") == 0),
        "journal": journal.get("verified") is True,
        "input_stability": manifest.get("end_of_build_input_stability", {}).get("verified") is True,
        "runtime": manifest.get("build_runtime", {}).get("verified") is True,
        "source_jobs": (manifest.get("source_jobs") == len(jobs)
                        and manifest.get("input_source_jobs") == input_jobs),
        "no_inherited_source_policy": manifest.get("source_exclusion_policy") is None,
        "freeze": (manifest.get("freeze_receipt_sha256") is not None
                   and manifest.get("generation_snapshot_sha256") == receipt.get("snapshot_sha256")
                   and manifest.get("source_sha256") == receipt.get("source_sha256")
                   and manifest.get("frozen_valid_candidate_count") ==
                       receipt.get("counts", {}).get("valid_candidates")),
        "candidate_only_dynamic_gate": (manifest.get("dynamic_trace_exclusion_policy") is None
            and dynamic.get("verified") is False and dynamic.get("expected_record_count") == 0
            and type(dynamic.get("observed_record_count")) is int
            and dynamic.get("observed_record_count") > 0
            and dynamic.get("candidate_path") == candidate_path.name
            and dynamic.get("candidate_sha256") == sha256(candidate_path)),
        "generation_partition": (manifest.get("frozen_valid_candidate_count") ==
            manifest.get("included_valid_candidate_count", -1)
            + manifest.get("excluded_valid_candidate_count", -1)
            + manifest.get("omitted_valid_candidate_count", -1)),
    }
    failed = sorted(name for name, value in gates.items() if not value)
    if failed:
        raise ValueError("Candidate has a non-promotable gate failure: " + ",".join(failed))
    return gates


def _candidate_policy(path, receipt):
    raw = Path(path).read_bytes()
    policy = _strict_json(raw, "dynamic exclusion candidate")
    if raw != (json.dumps(policy, indent=2, sort_keys=True, allow_nan=False) + "\n").encode():
        raise ValueError("Dynamic exclusion candidate bytes are not canonical")
    records = policy.get("records")
    if (policy.get("schema") != POLICY_SCHEMA
            or policy.get("snapshot_sha256") != receipt.get("snapshot_sha256")
            or policy.get("source_sha256") != receipt.get("source_sha256")
            or not isinstance(records, list)):
        raise ValueError("Dynamic exclusion candidate is not source-bound")
    found = {}
    for record in records:
        reason = record.get("reason") if isinstance(record, dict) else None
        duplicate = reason in {"duplicate_normalized_messages", "duplicate_retained_arrays"}
        expected_basis = {
            "duplicate_normalized_messages": "normalized_messages_sha256",
            "duplicate_retained_arrays": "retained_arrays_sha256",
        }.get(reason)
        native = reason in {"UnsupportedMessage", "UnsupportedTrace"}
        if (not isinstance(record, dict) or set(record) != RECORD_KEYS
                or not isinstance(record.get("identity"), str) or not record["identity"]
                or record["identity"] in found or reason not in ALLOWED_REASONS
                or type(record.get("generated_gaps")) is not int or record["generated_gaps"] < 0
                or not _hash(record.get("source_digest")) or not _hash(record.get("row_sha256"))
                or (duplicate and (record.get("duplicate_basis") != expected_basis
                    or not _hash(record.get("duplicate_sha256"))
                    or record.get("duplicate_of_source") not in {"trace", "replay"}
                    or not isinstance(record.get("duplicate_of_identity"), str)
                    or not record["duplicate_of_identity"]))
                or (not duplicate and any(record.get(key) is not None for key in (
                    "duplicate_basis", "duplicate_sha256", "duplicate_of_source",
                    "duplicate_of_identity")))
                or (native != _hash(record.get("native_failure_sha256")))):
            raise ValueError("Dynamic exclusion candidate record is invalid")
        found[record["identity"]] = record
    if (records != [found[key] for key in sorted(found)]
            or policy.get("record_count") != len(found)
            or policy.get("generated_gap_count") != sum(
                record["generated_gaps"] for record in found.values())):
        raise ValueError("Dynamic exclusion candidate ordering or aggregate changed")
    return raw, policy, found


def _derive_records(candidate_records, exclusions, jobs, accepted):
    trace_jobs = {job["identity"]: job for job in jobs if job.get("source") == "trace"}
    job_keys = {(job.get("source"), job.get("identity")) for job in jobs}
    if len(job_keys) != len(jobs):
        raise ValueError("Frozen selected jobs have duplicate identities")
    accepted_map = {}
    for row in accepted:
        key = (row.get("source"), row.get("identity"))
        if row.get("ok") is not True or key not in job_keys or key in accepted_map:
            raise ValueError("Accepted row receipt has an invalid identity")
        accepted_map[key] = row
    exclusion_map, derived = {}, {}
    for row in exclusions:
        key = (row.get("source"), row.get("identity"))
        if (row.get("ok") is not False or row.get("source") != "trace"
                or row.get("reason") not in ALLOWED_REASONS or key not in job_keys
                or key in exclusion_map):
            raise ValueError("Candidate contains a non-dynamic or duplicate exclusion")
        exclusion_map[key] = row
        source_row = trace_jobs[row["identity"]]["source_row"]
        record = {
            "identity": row["identity"], "reason": row["reason"],
            "generated_gaps": row.get("generated_gaps", 0),
            "source_digest": source_row["source_digest"],
            "row_sha256": source_row["row_sha256"],
            "duplicate_basis": row.get("duplicate_basis"),
            "duplicate_sha256": row.get("duplicate_sha256"),
            "duplicate_of_source": row.get("duplicate_of_source"),
            "duplicate_of_identity": row.get("duplicate_of_identity"),
            "native_failure_sha256": row.get("error_message_sha256"),
        }
        if record != candidate_records.get(row["identity"]):
            raise ValueError("Candidate record is not exactly derived from exclusion evidence")
        derived[row["identity"]] = record
    if derived != candidate_records:
        raise ValueError("Candidate/exclusion record sets differ")
    if set(accepted_map) & set(exclusion_map) or set(accepted_map) | set(exclusion_map) != job_keys:
        raise ValueError("Accepted and excluded receipts do not partition every frozen job")
    return derived, accepted_map, exclusion_map


def _init_rerender(tokenizer_dir, snapshot, source, generator_config):
    global _renderer, _database, _source, _generator_config
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    from training.qwen38_native_gap_v3.masked import Qwen38TurnBoundaryMaskedRenderer
    _renderer = Qwen38TurnBoundaryMaskedRenderer(tokenizer_dir)
    _database = sqlite3.connect(Path(snapshot).resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    _database.row_factory = sqlite3.Row
    _source = Path(source)
    _generator_config = generator_config


def _entries(source_row):
    rows = _database.execute(
        "SELECT g.* FROM gaps g JOIN candidates c ON c.gap_id=g.id "
        "WHERE g.source_id=? AND g.state='review_pending' ORDER BY g.event_index",
        (source_row["id"],))
    result = []
    for row in rows:
        gap = {**dict(row), **{key: source_row[key] for key in
                              ("source_digest", "row_sha256", "trace_id")}}
        candidate = _database.execute(
            "SELECT * FROM candidates WHERE gap_id=?", (gap["id"],)).fetchone()
        if candidate is None:
            raise ValueError("Frozen gap lacks one candidate")
        candidate = dict(candidate)
        receipt = _database.execute(
            "SELECT * FROM responses WHERE gap_id=? AND stage='generate' AND text=? "
            "ORDER BY id DESC LIMIT 1", (gap["id"], candidate["text"])).fetchone()
        if receipt is None:
            raise ValueError("Frozen candidate lacks its exact generate response")
        result.append({"gap": gap, "candidate": candidate,
                       "generation_receipt": dict(receipt)})
    return result


def _rerender_job(job):
    from cot_filler.core import validate_trace
    from training.qwen38_native_gap_v3.masked import CrossArmParityError, MaskedBoundaryExclusion
    from training.qwen38_native_gap_v3 import masked_replay
    from training.qwen38_no_cot.prepare_data import UnsupportedTrace
    from training.qwen38_no_cot.upstream_manager.render import UnsupportedMessage
    try:
        if job["source"] == "trace":
            saved = job["source_row"]
            with _source.open("rb") as stream:
                stream.seek(saved["offset"]); raw = stream.read(saved["length"])
            if hashlib.sha256(raw).hexdigest() != saved["row_sha256"]:
                raise ValueError("Source row changed")
            source = validate_trace(_strict_json(raw, "source trace"))
            if source.get("trace_id") != saved["trace_id"] or digest(source) != saved["source_digest"]:
                raise ValueError("Source trace identity changed")
            selection = source.get("metadata", {}).get("selection", {})
            quality = selection.get("quality_score")
            confidence = selection.get("overall_confidence_score")
            if (selection.get("capture_id") != source["trace_id"]
                    or type(quality) not in (int, float) or not 4 <= quality <= 5
                    or type(confidence) not in (int, float) or not 3 <= confidence <= 5):
                raise ValueError("Trace fails the frozen quality filter")
            entries = _entries(saved)
            rendered = _renderer.render_generated_trace(source, entries, _generator_config)
            source_digest = saved["source_digest"]
            generated_gaps = rendered["cot_mask_audit"]["bound_input_generation_count"]
        else:
            messages, counts, _ = masked_replay.normalize_job(job)
            rendered = _renderer.render_replay_messages(
                messages, source_digest=job["sha256"], trace_id=job["identity"],
                boundaries=[{"barrier_before": True} for _ in messages],
                source_tool_audit=counts["turn_boundary_audit"]["raw_tool_cardinality"])
            source_digest = job["sha256"]
            generated_gaps = 0
        return {"identity": job["identity"], "source": job["source"], "ok": True,
            "source_digest": source_digest, "generated_gaps": generated_gaps,
            "normalized_messages_sha256": rendered["turn_boundary_audit"]["output_messages_sha256"],
            "retained_arrays_sha256": hashlib.sha256(json.dumps(
                [rendered["input_ids"], rendered["labels"]], separators=(",", ":")).encode()).hexdigest()}
    except MaskedBoundaryExclusion as error:
        if job["source"] != "trace":
            return {"identity": job["identity"], "source": job["source"], "ok": False,
                    "reason": "unexpected:MaskedBoundaryExclusion"}
        count = _database.execute(
            "SELECT count(*) FROM gaps g JOIN candidates c ON c.gap_id=g.id "
            "WHERE g.source_id=? AND g.state='review_pending'",
            (job["source_row"]["id"],)).fetchone()[0]
        return {"identity": job["identity"], "source": job["source"], "ok": False,
                "reason": error.code, "generated_gaps": count}
    except (UnsupportedTrace, UnsupportedMessage) as error:
        if job["source"] != "trace":
            return {"identity": job["identity"], "source": job["source"], "ok": False,
                    "reason": "unexpected:" + type(error).__name__}
        count = _database.execute(
            "SELECT count(*) FROM gaps g JOIN candidates c ON c.gap_id=g.id "
            "WHERE g.source_id=? AND g.state='review_pending'",
            (job["source_row"]["id"],)).fetchone()[0]
        return {"identity": job["identity"], "source": job["source"], "ok": False,
                "reason": type(error).__name__, "generated_gaps": count,
                "native_failure_sha256": hashlib.sha256(str(error).encode()).hexdigest()}
    except CrossArmParityError:
        return {"identity": job["identity"], "source": job["source"], "ok": False,
                "reason": "cross_arm_no_cot_parity_failed"}
    except Exception as error:
        return {"identity": job["identity"], "source": job["source"], "ok": False,
                "reason": "unexpected:" + type(error).__name__}


def _verify_rerender_results(jobs, candidate_records, accepted, results):
    by_key = {(job["source"], job["identity"]): job for job in jobs}
    order = {key: index for index, key in enumerate(by_key)}
    required = {("trace", identity) for identity in candidate_records}
    for record in candidate_records.values():
        if record["reason"].startswith("duplicate_"):
            required.add((record["duplicate_of_source"], record["duplicate_of_identity"]))
    if not required <= set(by_key):
        raise ValueError("Duplicate owner is outside the selected frozen jobs")
    actual = {(row["source"], row["identity"]): row for row in results}
    if len(actual) != len(required) or set(actual) != required:
        raise ValueError("Independent rerender returned duplicate or missing identities")
    duplicate_count = 0
    for identity, record in candidate_records.items():
        key = ("trace", identity); row = actual[key]
        reason = record["reason"]
        if reason.startswith("duplicate_"):
            duplicate_count += 1
            basis = ("normalized_messages_sha256" if reason == "duplicate_normalized_messages"
                     else "retained_arrays_sha256")
            owner_key = (record["duplicate_of_source"], record["duplicate_of_identity"])
            owner = actual[owner_key]
            if (record["duplicate_basis"] != basis or not row.get("ok") or not owner.get("ok")
                    or row.get(basis) != record["duplicate_sha256"]
                    or owner.get(basis) != record["duplicate_sha256"]
                    or owner_key not in accepted or order[owner_key] >= order[key]
                    or accepted[owner_key].get(basis) != record["duplicate_sha256"]):
                raise ValueError("Independent duplicate owner proof failed")
            earlier = [accepted[candidate_key] for candidate_key in accepted
                       if order[candidate_key] < order[key]
                       and accepted[candidate_key].get(basis) == record["duplicate_sha256"]]
            if len(earlier) != 1 or (earlier[0]["source"], earlier[0]["identity"]) != owner_key:
                raise ValueError("Duplicate owner is not the exact earlier retained row")
            generated_gaps = row.get("generated_gaps")
        else:
            if (row.get("ok") is not False or row.get("reason") != reason
                    or (reason in {"UnsupportedTrace", "UnsupportedMessage"}
                        and row.get("native_failure_sha256") != record["native_failure_sha256"])):
                raise ValueError("Independent exclusion rerender differs")
            generated_gaps = row.get("generated_gaps")
        if generated_gaps != record["generated_gaps"]:
            raise ValueError("Independent exclusion generated-gap count differs")
    return {"excluded_traces_rerendered": len(candidate_records),
            "duplicate_exclusions_verified": duplicate_count,
            "unique_owner_rows_rerendered": len(required) - len(candidate_records),
            "exact_earlier_owner_proof": True,
            "result_set_sha256": digest(sorted(results,
                key=lambda row: (row["source"], row["identity"]))) }


def _independent_rerender(jobs, candidate_records, accepted, *, tokenizer_dir,
                          snapshot, source, generator_config, workers):
    by_key = {(job["source"], job["identity"]): job for job in jobs}
    order = {key: index for index, key in enumerate(by_key)}
    required = {("trace", identity) for identity in candidate_records}
    for record in candidate_records.values():
        if record["reason"].startswith("duplicate_"):
            required.add((record["duplicate_of_source"], record["duplicate_of_identity"]))
    if not required or not required <= set(by_key):
        raise ValueError("Independent rerender task set is empty or has an unknown owner")
    tasks = [by_key[key] for key in sorted(required, key=lambda key: order[key])]
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks)), initializer=_init_rerender,
            initargs=(str(tokenizer_dir), str(snapshot), str(source), generator_config)) as pool:
        results = list(pool.map(_rerender_job, tasks))
    return _verify_rerender_results(jobs, candidate_records, accepted, results)


def _publish(output_dir, policy_raw, receipt):
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("Promotion requires a fresh policy directory")
    if (output_dir.parent.is_symlink() or not output_dir.parent.is_dir()
            or output_dir.parent.resolve(strict=True) != output_dir.parent):
        raise ValueError("Promotion policy parent must be one canonical directory")
    incoming = output_dir.with_name("." + output_dir.name + ".incoming-" + uuid.uuid4().hex)
    failed = output_dir.with_name("." + output_dir.name + ".failed-" + uuid.uuid4().hex)
    incoming.mkdir(mode=0o700)
    published = False
    try:
        policy = incoming / "dynamic-trace-exclusions.json"
        with policy.open("xb") as stream:
            stream.write(policy_raw); stream.flush(); os.fsync(stream.fileno())
        policy.chmod(0o400)
        receipt_path = incoming / "promotion-receipt.json"
        raw = (json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        with receipt_path.open("xb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        receipt_path.chmod(0o400)
        descriptor = os.open(incoming, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        incoming.chmod(0o500)
        parent = os.open(output_dir.parent, os.O_RDONLY)
        try:
            incoming.rename(output_dir); published = True; os.fsync(parent)
        finally:
            os.close(parent)
        members = sorted(path.name for path in output_dir.iterdir())
        if (members != ["dynamic-trace-exclusions.json", "promotion-receipt.json"]
                or any(path.is_symlink() or not path.is_file() for path in output_dir.iterdir())
                or output_dir.stat().st_mode & 0o777 != 0o500
                or any(path.stat().st_mode & 0o777 != 0o400 for path in output_dir.iterdir())
                or (output_dir / "dynamic-trace-exclusions.json").read_bytes() != policy_raw
                or (output_dir / "promotion-receipt.json").read_bytes() != raw):
            raise RuntimeError("Promoted policy changed across atomic publication")
    except Exception:
        evidence = output_dir if published and output_dir.exists() else incoming
        if evidence.exists() and not failed.exists():
            evidence.rename(failed)
            parent = os.open(output_dir.parent, os.O_RDONLY)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
        raise
    return {"policy": str(output_dir / "dynamic-trace-exclusions.json"),
            "policy_sha256": hashlib.sha256(policy_raw).hexdigest(),
            "receipt": str(output_dir / "promotion-receipt.json"),
            "receipt_sha256": hashlib.sha256(raw).hexdigest()}


def promote(*, candidate_dir, build_receipt, build_receipt_sha256, freeze_receipt,
            baseline_selection, tokenizer_dir, code_root, code_manifest_sha256,
            output_dir, workers=32, _enforce_runtime=True, _rerender=None):
    started = datetime.datetime.now(datetime.timezone.utc)
    candidate_dir, build_receipt = Path(candidate_dir), Path(build_receipt)
    freeze_receipt, baseline_selection = Path(freeze_receipt), Path(baseline_selection)
    tokenizer_dir, code_root, output_dir = map(Path, (tokenizer_dir, code_root, output_dir))
    if type(workers) is not int or not 1 <= workers <= 96:
        raise ValueError("Promotion worker count must be 1..96")
    if (candidate_dir.parent != ROOT / "datasets"
            or not candidate_dir.name.startswith("cot-masked-native-gap-v4-")
            or candidate_dir.is_symlink() or candidate_dir.resolve(strict=True) != candidate_dir
            or build_receipt != candidate_dir.with_name(candidate_dir.name + ".build-receipt.json")
            or build_receipt.is_symlink() or not build_receipt.is_file()
            or build_receipt.stat().st_mode & 0o222
            or output_dir.parent != ROOT / "policies"
            or not output_dir.name.startswith("cot-masked-native-gap-v4-")
            or tokenizer_dir != code_root / "training/qwen38_no_cot/assets"
            or output_dir.exists() or output_dir.is_symlink()):
        raise ValueError("Promotion paths must use fresh canonical v4 namespaces")
    if sha256(build_receipt) != build_receipt_sha256:
        raise ValueError("Candidate build receipt changed")
    code_manifest = _verify_code_tree(code_root, code_manifest_sha256)
    receipt_record = _strict_json_file(build_receipt, "candidate build receipt")
    records = _tree_inventory(candidate_dir)
    if (receipt_record.get("schema") != "qwen38-native-gap-v4-build-publication/v1"
            or receipt_record.get("status") != "candidate_blocked_for_review"
            or receipt_record.get("mode") != "candidate"
            or receipt_record.get("output") != str(candidate_dir)
            or receipt_record.get("runtime_image") != IMAGE
            or receipt_record.get("code_manifest_sha256") != code_manifest_sha256
            or receipt_record.get("freeze_receipt_sha256") != sha256(freeze_receipt)
            or receipt_record.get("baseline_selection_sha256") != sha256(baseline_selection)
            or receipt_record.get("dynamic_trace_exclusions_sha256") is not None
            or receipt_record.get("atomic_direct_dataset_publication") is not True
            or receipt_record.get("cpu_only") is not True
            or receipt_record.get("network") != "none"
            or receipt_record.get("tree_inventory") != records
            or receipt_record.get("tree_sha256") != _tree_sha(records)
            or receipt_record.get("files") != len(records)
            or receipt_record.get("bytes") != sum(record["bytes"] for record in records)):
        raise ValueError("Candidate build publication receipt is not exact")
    core = receipt_record.get("core_sha256", {})
    expected_core = {"manifest.json", "row-receipts.jsonl", "exclusions.jsonl",
                     "dynamic-trace-exclusions.candidate.json", "BLOCKED.json"}
    if (not isinstance(core, dict) or set(core) != expected_core
            or any(not _hash(value) or sha256(candidate_dir / name) != value
                   for name, value in core.items())):
        raise ValueError("Candidate build core hashes changed")
    manifest_path = candidate_dir / "manifest.json"
    manifest = _strict_json_file(manifest_path, "candidate manifest")
    blocked = _strict_json_file(candidate_dir / "BLOCKED.json", "candidate BLOCKED receipt")
    if (blocked != {"reason": "dynamic_trace_exclusion_allowlist_mismatch",
                    "manifest_sha256": sha256(manifest_path)}):
        raise ValueError("Candidate did not block only at dynamic exclusion review")
    frozen, generator_identity, jobs, input_jobs, gap_counts, stability_inputs = _load_frozen_jobs(
        freeze_receipt, baseline_selection, receipt_record["baseline_selection_sha256"])
    if manifest.get("freeze_receipt_sha256") != sha256(freeze_receipt):
        raise ValueError("Candidate manifest freeze receipt changed")
    candidate_path = candidate_dir / "dynamic-trace-exclusions.candidate.json"
    candidate_gates = _manifest_gates(
        manifest, candidate_path, jobs, input_jobs, frozen)
    policy_raw, policy, candidate_records = _candidate_policy(candidate_path, frozen)
    if (manifest["dynamic_trace_exclusion_audit"].get("observed_record_count") != len(candidate_records)
            or manifest["dynamic_trace_exclusion_audit"].get("observed_generated_gap_count") !=
               policy["generated_gap_count"]):
        raise ValueError("Manifest dynamic exclusion aggregates changed")
    jobs_raw = _strict_jsonl(candidate_dir / "jobs.private.jsonl", "candidate jobs",
                             require_final_newline=True)
    if jobs_raw != jobs:
        raise ValueError("Candidate jobs differ from independently derived frozen jobs")
    accepted = _strict_jsonl(candidate_dir / "row-receipts.jsonl", "accepted row receipts",
                             require_final_newline=True)
    exclusions = _strict_jsonl(candidate_dir / "exclusions.jsonl", "candidate exclusions",
                               require_final_newline=True)
    derived, accepted_map, _ = _derive_records(candidate_records, exclusions, jobs, accepted)
    trace_jobs = {job["identity"]: job for job in jobs if job["source"] == "trace"}
    if (not isinstance(manifest.get("counts"), dict)
            or any(type(value) is not int or value < 0
                   for value in manifest["counts"].values())
            or sum(manifest["counts"].values()) != len(accepted)
            or manifest.get("excluded_source_count") != len(exclusions)
            or manifest.get("trace_source_membership", {}).get("failed_trace_sources") != len(exclusions)
            or any(record["generated_gaps"] != gap_counts.get(
                trace_jobs[identity]["source_row"]["id"], 0)
                for identity, record in derived.items())):
        raise ValueError("Candidate source/exclusion aggregates changed")
    renderer_identity = None
    if _enforce_runtime:
        if os.environ.get("YETA_TRAINING_IMAGE") != IMAGE:
            raise ValueError("Promotion must run in the exact pinned image")
        required_ro = [candidate_dir, build_receipt, freeze_receipt, baseline_selection,
                       tokenizer_dir, code_root, frozen["snapshot_path"], frozen["source_path"]]
        required_ro.extend(item["path"] for item in frozen["replay_manifests"])
        required_ro.extend(path for path, _ in stability_inputs)
        if any(not _on_read_only_mount(path) for path in required_ro):
            raise ValueError("Every promotion input must be on a dedicated read-only mount")
    from training.qwen38_native_gap_v3.masked import Qwen38TurnBoundaryMaskedRenderer
    renderer_identity = Qwen38TurnBoundaryMaskedRenderer(tokenizer_dir).identity
    if renderer_identity != manifest.get("renderer_identity"):
        raise ValueError("Promotion renderer/tokenizer identity differs from candidate")
    rerender = _rerender or _independent_rerender
    rerender_receipt = rerender(jobs, candidate_records, accepted_map,
        tokenizer_dir=tokenizer_dir, snapshot=frozen["snapshot_path"],
        source=frozen["source_path"], generator_config=generator_identity["generator_config"],
        workers=workers)
    # Recheck all small gates and immutable file identities after expensive rendering.
    if (sha256(build_receipt) != build_receipt_sha256
            or _tree_inventory(candidate_dir) != records
            or sha256(freeze_receipt) != receipt_record["freeze_receipt_sha256"]
            or sha256(baseline_selection) != receipt_record["baseline_selection_sha256"]
            or any(sha256(path) != expected for path, expected in stability_inputs)):
        raise ValueError("A promotion input changed during independent review")
    _verify_code_tree(code_root, code_manifest_sha256)
    finished = datetime.datetime.now(datetime.timezone.utc)
    promotion = {
        "schema": PROMOTION_SCHEMA, "status": "complete", "at": finished.isoformat(),
        "output_dir": str(output_dir),
        "policy_path": str(output_dir / "dynamic-trace-exclusions.json"),
        "runtime_image": IMAGE, "candidate_dir": str(candidate_dir),
        "candidate_build_receipt_path": str(build_receipt),
        "candidate_build_receipt_sha256": build_receipt_sha256,
        "candidate_tree_sha256": receipt_record["tree_sha256"],
        "candidate_build_core_sha256": core,
        "candidate_manifest_sha256": sha256(manifest_path),
        "candidate_blocked_sha256": sha256(candidate_dir / "BLOCKED.json"),
        "candidate_jobs_sha256": sha256(candidate_dir / "jobs.private.jsonl"),
        "candidate_jobs_count": len(jobs_raw),
        "candidate_exclusions_sha256": sha256(candidate_dir / "exclusions.jsonl"),
        "candidate_exclusions_count": len(exclusions),
        "candidate_row_receipts_sha256": sha256(candidate_dir / "row-receipts.jsonl"),
        "candidate_row_receipts_count": len(accepted),
        "candidate_policy_path": str(candidate_path),
        "candidate_policy_sha256": sha256(candidate_path),
        "candidate_record_set_sha256": digest(policy["records"]),
        "freeze_receipt_path": str(freeze_receipt),
        "freeze_receipt_sha256": sha256(freeze_receipt),
        "snapshot_path": frozen["snapshot_path"],
        "snapshot_sha256": frozen["snapshot_sha256"],
        "source_path": frozen["source_path"],
        "source_sha256": frozen["source_sha256"],
        "source_bytes": frozen["source_bytes"],
        "baseline_selection_path": str(baseline_selection),
        "baseline_selection_sha256": sha256(baseline_selection),
        "code_manifest_path": str(code_root / "code-manifest.json"),
        "code_manifest_sha256": code_manifest_sha256,
        "promotion_helper_sha256": sha256(__file__),
        "renderer_identity": renderer_identity,
        "generator_identity_sha256": digest(generator_identity),
        "policy_schema": POLICY_SCHEMA, "policy_sha256": hashlib.sha256(policy_raw).hexdigest(),
        "policy_bytes": len(policy_raw),
        "record_count": len(candidate_records),
        "generated_gap_count": policy["generated_gap_count"],
        "reason_counts": dict(sorted(Counter(
            record["reason"] for record in candidate_records.values()).items())),
        "selected_source_jobs": len(jobs), "input_source_jobs": input_jobs,
        "candidate_non_dynamic_gates": candidate_gates,
        "all_non_dynamic_gates_passed": all(candidate_gates.values()),
        "candidate_bytes_promoted_unchanged": True,
        "independent_record_derivation": True, "independent_rerender": rerender_receipt,
        "independent_journal_cardinality_audit":
            frozen["independent_journal_cardinality_audit"],
        "replay_manifests": frozen["replay_manifests"],
        "runtime_code_file_count": len(code_manifest["files"]),
        "elapsed_seconds": (finished - started).total_seconds(),
    }
    if set(promotion) != PROMOTION_RECEIPT_KEYS:
        raise AssertionError("Promotion receipt schema drifted")
    output_dir.parent.mkdir(mode=0o700, exist_ok=True)
    published = _publish(output_dir, policy_raw, promotion)
    return {**promotion, **published}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--build-receipt", required=True)
    parser.add_argument("--build-receipt-sha256", required=True)
    parser.add_argument("--freeze-receipt", required=True)
    parser.add_argument("--baseline-selection", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--code-manifest-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=32)
    print(json.dumps(promote(**vars(parser.parse_args())), sort_keys=True))
