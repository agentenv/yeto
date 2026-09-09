"""Rebuild no-CoT token shards from a frozen source-job list, on CPU only.

Uses a fresh directory, accounts for every source and never rewrites a running
dataset. Source selection and the existing session-group split are preserved.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path

from training.qwen38_no_cot import prepare_data as original
from . import source_adapters
from .normalize import ContinuationMappingError

VERSION = "qwen38-corrected-turn-dataset/v2"
_renderer = None


def init_worker(tokenizer_dir):
    global _renderer
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from .render import Qwen38Renderer
    _renderer = Qwen38Renderer(tokenizer_dir)


def convert_job(job):
    from .render import training_row
    try:
        raw = Path(job["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != job["sha256"]:
            raise original.UnsupportedTrace("source_file_hash_mismatch")
        if job["format"] == "canonical":
            messages, audit = source_adapters.canonical_messages([json.loads(line) for line in raw.splitlines() if line.strip()])
            group = job["group_id"]
        elif job["format"] == "rollout":
            messages, audit, group = source_adapters.rollout_messages([json.loads(line) for line in raw.splitlines() if line.strip()])
        elif job["format"] == "atif":
            messages, audit, group = source_adapters.atif_messages(json.loads(raw))
        else:
            raise original.UnsupportedTrace("unknown_job_format")
        result = _renderer.render(messages, turn_boundary_audit=audit["turn_boundary_audit"])
        provenance = {**job, "normalization_version": source_adapters.VERSION,
                      "source_normalization_counts": audit["source_normalization_counts"],
                      "messages_sha256": result["turn_boundary_audit"]["output_messages_sha256"],
                      "original_reasoning_used": False, "capability_schemas_included": False,
                      "custom_adapter": "raw-input-as-function-input-string/v1"}
        row = training_row(result, group_id=group, source=job["source"], provenance=provenance)
        return {"ok": True, "row": row, "job": job}
    except Exception as exc:
        reason = exc.code if isinstance(exc, ContinuationMappingError) else str(exc) if isinstance(exc, original.UnsupportedTrace) else type(exc).__name__
        return {"ok": False, "identity": job["identity"], "source": job["source"], "reason": reason,
                "source_sha256": job["sha256"]}


def rebuild(*, source_jobs, source_jobs_sha256, tokenizer_dir, output, workers=8, shards=128):
    if type(workers) is not int or not 1 <= workers <= 128 or type(shards) is not int or not 1 <= shards <= 256:
        raise ValueError("workers and shard counts must be bounded positive integers")
    source_raw = Path(source_jobs).read_bytes()
    if hashlib.sha256(source_raw).hexdigest() != source_jobs_sha256:
        raise ValueError("Frozen source-job identity changed")
    jobs = [json.loads(line) for line in source_raw.splitlines() if line.strip()]
    if not jobs or any(job.get("source") not in ("trace", "replay") for job in jobs):
        raise ValueError("Expected a nonempty original trace/replay job inventory")
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "source_jobs.jsonl").write_bytes(source_raw)
    handles, entries, split_counts = {}, {}, Counter()
    counts, token_counts, exclusions, normalization = Counter(), Counter(), Counter(), Counter()
    seen, seen_retained = set(), set()
    with (out / "exclusions.jsonl").open("x") as excluded:
        try:
            with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(tokenizer_dir,)) as pool:
                def results():
                    source = iter(jobs)
                    pending = deque()
                    for _ in range(workers * 2):
                        job = next(source, None)
                        if job is not None: pending.append(pool.submit(convert_job, job))
                    while pending:
                        yield pending.popleft().result()
                        job = next(source, None)
                        if job is not None: pending.append(pool.submit(convert_job, job))

                for result in results():
                    if result["ok"]:
                        row = result["row"]
                        identity = row["metadata"]["provenance"]["messages_sha256"]
                        retained_identity = hashlib.sha256(json.dumps([row["input_ids"], row["labels"]], separators=(",", ":")).encode()).hexdigest()
                        if identity in seen or retained_identity in seen_retained:
                            result = {"ok": False, "identity": result["job"]["identity"], "source": row["source"],
                                      "source_sha256": result["job"]["sha256"], "reason": "duplicate_normalized_messages" if identity in seen else "duplicate_retained_tokens_and_labels"}
                        else:
                            seen.add(identity)
                            seen_retained.add(retained_identity)
                            row["metadata"]["retained_training_arrays_sha256"] = retained_identity
                    if not result["ok"]:
                        exclusions[result["reason"]] += 1
                        excluded.write(json.dumps(result, sort_keys=True) + "\n")
                        continue
                    row = result["row"]
                    # Same group-to-split function as the original baseline.
                    split_hash = hashlib.sha256((original.SEED + ":split:" + row["group_id"]).encode()).hexdigest()
                    split = "validation" if int(split_hash[:8], 16) % 100 == 0 else "train"
                    key = (split, split_counts[split] % shards)
                    split_counts[split] += 1
                    if key not in handles:
                        name = f"{split}-{key[1]:05d}.jsonl"
                        handles[key] = (out / name).open("xb")
                        entries[key] = {"path": name, "rows": [], "digest": hashlib.sha256()}
                    stream, entry = handles[key], entries[key]
                    raw = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
                    offset = stream.tell(); stream.write(raw); entry["digest"].update(raw)
                    entry["rows"].append([offset, len(raw), hashlib.sha256(raw).hexdigest(), len(row["input_ids"])])
                    counts[split + "/" + row["source"]] += 1
                    token_counts[split + "/input"] += len(row["input_ids"])
                    token_counts[split + "/targets"] += sum(t != -100 for t in row["labels"])
                    token_counts["dropped_input_tokens"] += row["metadata"]["sequence_audit"]["dropped_input_tokens"]
                    for name, value in row["metadata"]["turn_boundary_audit"].items():
                        if type(value) is int: normalization[name] += value
        finally:
            for stream in handles.values():
                stream.flush(); os.fsync(stream.fileno()); stream.close()
            excluded.flush(); os.fsync(excluded.fileno())
    if sum(counts.values()) + sum(exclusions.values()) != len(jobs):
        raise RuntimeError("Source accounting is incomplete")
    if not split_counts["train"]:
        raise ValueError("The corrected dataset has no accepted training rows")
    init_worker(tokenizer_dir)
    manifest = {"schema": VERSION, "source_jobs_sha256": source_jobs_sha256, "renderer_identity": _renderer.identity,
                "splits": {"train": [], "validation": []}, "counts": dict(counts), "exclusions": dict(exclusions),
                "normalization": dict(normalization), "token_counts": dict(token_counts), "source_jobs": len(jobs),
                "source_accounting_complete": True, "behavioral_recovery_validated": False,
                "split_policy": "unchanged-original-session-group-hash", "training_ready": False}
    index = {"schema": "turn-normalized-token-index/v2", "seq_len": 262144,
             "renderer_identity": _renderer.identity, "splits": {"train": [], "validation": []}}
    for (split, number), entry in sorted(entries.items()):
        checksum = entry["digest"].hexdigest()
        manifest["splits"][split].append({"path": entry["path"], "sha256": checksum, "rows": len(entry["rows"])})
        index["splits"][split].append({"path": entry["path"], "sha256": checksum,
                                       "size_bytes": (out / entry["path"]).stat().st_size, "rows": entry["rows"]})
    manifest_raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    (out / "manifest.json").write_bytes(manifest_raw)
    index["manifest_sha256"] = hashlib.sha256(manifest_raw).hexdigest()
    index_raw = (json.dumps(index, separators=(",", ":"), sort_keys=True) + "\n").encode()
    (out / "index.json").write_bytes(index_raw)
    receipt = {"schema": "turn-normalized-cpu-rebuild/v2", "manifest_sha256": index["manifest_sha256"],
               "index_sha256": hashlib.sha256(index_raw).hexdigest(), "source_jobs_sha256": source_jobs_sha256,
               "source_jobs": len(jobs), "accepted": sum(counts.values()), "excluded": sum(exclusions.values()),
               "source_accounting_complete": True, "training_ready": False,
               "remaining": ["exact-training-loader-validation", "corrected-data-training", "behavioral-evaluation"]}
    (out / "COMPLETE.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jobs", required=True)
    parser.add_argument("--source-jobs-sha256", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shards", type=int, default=128)
    print(json.dumps(rebuild(**vars(parser.parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
