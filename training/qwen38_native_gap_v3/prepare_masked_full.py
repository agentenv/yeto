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
from .masked import MaskedBoundaryExclusion
from training.qwen38_no_cot.prepare_data import UnsupportedTrace
from training.qwen38_no_cot.upstream_manager.render import UnsupportedMessage

VERSION = "qwen38-full-native-gap-cot-cpu-export/v3"
_renderer = _db = _identity = _source = _spool = None
_all_train = False
ALL_TRAIN_POLICY = "all-source-train-only-capture-identity-for-missing-session/v1"


def split_for_group(group):
    return "validation" if int(hashlib.sha256((SEED + ":split:" + group).encode()).hexdigest()[:8], 16) % 100 == 0 else "train"


def _readonly(path):
    db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    db.row_factory = sqlite3.Row
    return db


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
        valid = db.execute("SELECT count(*) FROM candidates c JOIN gaps g ON g.id=c.gap_id WHERE g.state='review_pending'").fetchone()[0]
        if (len(sources) != receipt["counts"]["sources"] or candidates < valid
                or valid != receipt["counts"]["valid_candidates"]):
            raise ValueError("Frozen source/candidate membership differs from receipt")
        if db.execute("SELECT count(*) FROM gaps WHERE state IN ('generating','reviewing')").fetchone()[0]:
            raise ValueError("Freeze must come from the drained generation run")
    finally:
        db.close()
    source = Path(identity["source_path"]).resolve(strict=True)
    signature = lambda: tuple(getattr(source.stat(), key) for key in
        ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"))
    before = signature()
    if sha_file(source) != identity["source_sha256"] or before != signature():
        raise ValueError("Original full source bytes changed")
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
        raw = canonical(row).encode() + b"\n"
        target = _spool / (hashlib.sha256((job["source"] + ":" + job["identity"]).encode()).hexdigest() + ".jsonl")
        with target.open("xb") as stream:
            stream.write(raw)
        return {"ok": True, "identity": job["identity"], "source": job["source"],
            "path": str(target), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
            "group_id": row["group_id"], "split": "train" if _all_train or metadata["provenance"].get("session_identity_verified") is False else split_for_group(row["group_id"]),
            "session_identity_unresolved": metadata["provenance"].get("session_identity_verified") is False,
            "source_digest": metadata["provenance"]["generated_cot"]["source_digest"],
            "normalized_messages_sha256": metadata['turn_boundary_audit']['output_messages_sha256'],
            "retained_arrays_sha256": hashlib.sha256(json.dumps([row['input_ids'],row['labels']],separators=(',',':')).encode()).hexdigest(),
            "input_tokens": len(row["input_ids"]), "target_tokens": sum(x != -100 for x in row["labels"]),
            "filled_gaps": cot["filled_gap_count"], "retained_filled_gaps": cot["retained_filled_gap_count"],
            "omitted_gaps":cot["omitted_generation_count"], "bound_input_gaps":cot["bound_input_generation_count"],
            "dropped_input_tokens": audit["dropped_input_tokens"],
            "dropped_target_tokens": audit["dropped_supervised_tokens"]}
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


def prepare(*, freeze_receipt, tokenizer_dir, output, workers=32, shard_rows=256, limit=None, all_train=False,
            source_exclusions=None, source_exclusions_sha256=None,
            baseline_selection=None, baseline_selection_sha256=None):
    if type(workers) is not int or not 1 <= workers <= 96 or not 1 <= shard_rows <= 1024:
        raise ValueError("Bounded CPU worker/shard settings required")
    if type(all_train) is not bool:
        raise ValueError("Explicit boolean train-only policy required")
    from . import masked_data as data
    from .masked import Qwen38TurnBoundaryMaskedRenderer
    receipt, identity, jobs = load_freeze(freeze_receipt)
    input_jobs = len(jobs)
    jobs, replay_selection = select_baseline_replay(jobs, baseline_selection, baseline_selection_sha256)
    audited_exclusions, exclusion_identity = source_exclusion_policy(
        source_exclusions, source_exclusions_sha256, receipt, jobs, tokenizer_dir)
    jobs = [job for job in jobs if job["kind"] != "trace" or job["identity"] not in audited_exclusions]
    if limit is not None:
        if type(limit) is not int or not 1 <= limit <= len(jobs):
            raise ValueError("Invalid finite preflight limit")
        # Genuine mixed shapes plus a filled source are tested by caller's
        # separate preflight; this limit is only a bounded export smoke.
        jobs = jobs[:limit]
    output = Path(output).resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    spool = output / "spool"
    spool.mkdir(mode=0o700)
    renderer_identity = Qwen38TurnBoundaryMaskedRenderer(tokenizer_dir).identity
    (output / "jobs.private.jsonl").write_text("".join(canonical(j) + "\n" for j in jobs))
    counts, tokens, exclusions, grouping = Counter(), Counter(), Counter(), Counter()
    boundary_excluded_sources, boundary_excluded_generations = Counter(), Counter()
    native_excluded_sources, native_excluded_generations = Counter(), Counter()
    duplicate_excluded_sources, duplicate_excluded_generations = Counter(), Counter()
    seen_normalized, seen_retained = set(), set()
    unexpected_trace_failures = 0
    shards = {"train": [], "validation": []}
    streams, row_counts, shard_hashes, shard_paths = {}, Counter(), {}, {}
    group_splits, digest_splits = {}, {}
    started = time.monotonic()
    def close_shard(split):
        if split in streams:
            stream = streams.pop(split)
            stream.flush();os.fsync(stream.fileno());stream.close()
            path = shard_paths.pop(split)
            entry = {"path": str(path), "sha256": shard_hashes.pop(split).hexdigest(),
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
                job = next(source, None)
                if job is not None:
                    pending.append(pool.submit(convert_job, job))
                if result['ok']:
                    reason = ('duplicate_normalized_messages' if result['normalized_messages_sha256'] in seen_normalized
                              else 'duplicate_retained_arrays' if result['retained_arrays_sha256'] in seen_retained else None)
                    if reason:
                        Path(result['path']).unlink()
                        result = {**result, 'ok':False, 'reason':reason, 'duplicate_source_exclusion':True,
                                  'generated_gaps':result['bound_input_gaps']}
                    else:
                        seen_normalized.add(result['normalized_messages_sha256'])
                        seen_retained.add(result['retained_arrays_sha256'])
                if not result["ok"]:
                    exclusions[result["source"] + "/" + result["reason"]] += 1
                    errors.write(canonical(result) + "\n")
                    if result["source"] == "trace":
                        if result.get("turn_boundary_exclusion") is True:
                            boundary_excluded_sources[result["reason"]] += 1
                            boundary_excluded_generations[result["reason"]] += result["generated_gaps"]
                        elif result.get('duplicate_source_exclusion') is True:
                            duplicate_excluded_sources[result['reason']] += 1
                            duplicate_excluded_generations[result['reason']] += result['generated_gaps']
                        elif result.get("native_source_exclusion") is True:
                            native_excluded_sources[result["reason"]] += 1
                            native_excluded_generations[result["reason"]] += result["generated_gaps"]
                        else:
                            unexpected_trace_failures += 1
                else:
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
                    for key in ("input_tokens", "target_tokens", "filled_gaps", "retained_filled_gaps", "dropped_input_tokens", "dropped_target_tokens", "omitted_gaps", "bound_input_gaps"):
                        tokens[key] += result[key]
                if processed % 32 == 0 or processed == len(jobs):
                    rows_out.flush(); errors.flush()
                    for stream in streams.values(): stream.flush()
                    progress(processed)
    for split in list(streams): close_shard(split)
    # Every excluded original trace is material and explicit. A full export may
    # not silently claim all frozen generations were included if some failed.
    excluded_generations = sum(r["generated_gaps"] for r in audited_exclusions.values()) + sum(boundary_excluded_generations.values()) + sum(native_excluded_generations.values()) + sum(duplicate_excluded_generations.values())
    complete_generations = tokens["filled_gaps"] == receipt["counts"]["valid_candidates"]
    accounted_generations = tokens["filled_gaps"] + tokens["omitted_gaps"] + excluded_generations == receipt["counts"]["valid_candidates"]
    manifest = {"schema": data.MANIFEST_SCHEMA, "training_contract": data.TRAINING_CONTRACT,
        "version": VERSION, "renderer_identity": renderer_identity, "splits": {k: v for k, v in shards.items() if v},
        "experimental": True, "acceptance_policy": data.ACCEPTANCE_POLICY, "reviewed": False,
        "semantic_review_required": False, "semantic_quality_qualified": False, "future_information_checked": False,
        "unreviewed_gap_count": tokens["filled_gaps"], "frozen_valid_candidate_count": receipt["counts"]["valid_candidates"],
        "included_valid_candidate_count": tokens["filled_gaps"], "excluded_valid_candidate_count": excluded_generations,
        "omitted_valid_candidate_count":tokens["omitted_gaps"],
        "gap_omission_policy":"keep-native-leading-gap-omit-only-conflicting-cot/v3",
        "all_frozen_generations_accounted_for": accounted_generations,
        "excluded_source_count": len(audited_exclusions) + sum(boundary_excluded_sources.values()) + sum(native_excluded_sources.values()) + sum(duplicate_excluded_sources.values()),
        "duplicate_source_exclusion_policy":"normalized-messages-or-retained-arrays-whole-source/v2",
        "duplicate_excluded_sources":dict(duplicate_excluded_sources),
        "duplicate_excluded_generations":dict(duplicate_excluded_generations),
        "native_render_excluded_sources": dict(native_excluded_sources),
        "native_render_excluded_generations": dict(native_excluded_generations),
        "native_render_exclusion_policy": "unchanged-native-shape-guards-whole-source/v2",
        "turn_boundary_excluded_sources": dict(boundary_excluded_sources),
        "turn_boundary_excluded_generations": dict(boundary_excluded_generations),
        "turn_boundary_exclusion_policy": "whole-source-exclusion-no-cot-relocation/v2",
        "unexpected_trace_failures": unexpected_trace_failures,
        "eligible_source_count": receipt["counts"]["sources"] - len(audited_exclusions) - sum(boundary_excluded_sources.values()) - sum(native_excluded_sources.values()) - sum(duplicate_excluded_sources.values()),
        "source_exclusion_policy": exclusion_identity,
        "baseline_replay_selection": replay_selection,
        "all_frozen_generations_included_before_cutoff": complete_generations,
        "full_export": limit is None, "input_source_jobs": input_jobs, "source_jobs": len(jobs),
        "counts": dict(counts), "token_counts": dict(tokens),
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
        "replay_adapter_sha256": sha_file(replay.__file__), "elapsed_seconds": time.monotonic() - started}
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if limit is None and (not accounted_generations or unexpected_trace_failures):
        (output / "BLOCKED.json").write_text(json.dumps({"reason": "original_trace_or_generation_coverage_incomplete",
            "manifest_sha256": sha_file(manifest_path)}, indent=2) + "\n")
        raise ValueError("Full export has explicit original-trace/generation exclusions; fix before training")
    index = data.build_index(manifest_path, output / "index.json", workers=min(workers,32))
    final = {"manifest": str(manifest_path), "manifest_sha256": sha_file(manifest_path), "index": index,
             "counts": dict(counts), "token_counts": dict(tokens)}
    (output / "COMPLETE.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    print(json.dumps(final, sort_keys=True), flush=True)
    return final


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
    args = parser.parse_args()
    prepare(**vars(args))


if __name__ == "__main__":
    main()
