"""CPU-only, streaming preflight of fresh CoT source shards and old sidecars.

No SQLite writes, provider calls, candidate approvals, or GPU use. Current
canonical inputs are checked one trace at a time. Use a fresh sidecar for a new
prompt version instead of rewriting immutable legacy prompt provenance.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3

from .core import PROMPT_VERSION, canonical, gaps_for, prompt_messages, validate_trace
from .provider import ContextOverflow, DeepSeekV4Tokenizer, LocalTokenizer, check_budget


def audit_sidecar(path):
    """Report incompatible stored prompt versions without opening a writable DB."""
    path = Path(path).resolve(strict=True)
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        versions = Counter()
        gaps = 0
        for (data,) in db.execute("SELECT data FROM gaps"):
            gap = json.loads(data)
            versions[gap.get("prompt_version", "missing")] += 1
            gaps += 1
        candidates = db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
        mismatches = sum(n for version, n in versions.items() if version != PROMPT_VERSION)
    return {
        "sidecar": str(path), "stored_gaps": gaps, "stored_candidates": candidates,
        "prompt_versions": dict(versions), "current_prompt_version": PROMPT_VERSION,
        "incompatible_prompt_gaps": mismatches,
        "current_store_resume_compatible": mismatches == 0,
        "action": "use_fresh_versioned_sidecar" if mismatches else "source_and_context_checks_still_required",
        "read_only": True,
    }


def _signature(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def audit_source(path, *, tokenizer=None, config=None, max_traces=20000, max_gaps=None, row_limit_bytes=64 * 1024 * 1024):
    """Stream JSONL; retain one trace/window and aggregate numeric evidence only.

    SHA-256 covers the complete source file even when max_traces/max_gaps limits
    validation. A successful result explicitly reports the validated scope.
    No semantic-generation quality claim is made by this structural preflight.
    """
    if type(max_traces) is not int or max_traces < 1:
        raise ValueError("max_traces must be a positive integer")
    if max_gaps is not None and (type(max_gaps) is not int or max_gaps < 1):
        raise ValueError("max_gaps must be a positive integer when supplied")
    if (tokenizer is None) != (config is None):
        raise ValueError("Tokenizer and context configuration must be supplied together")
    path = Path(path).resolve(strict=True)
    before = _signature(path.stat())
    source_hash = hashlib.sha256()
    traces = lines = gaps = oversized = checked_tokens = total_gap_bytes = 0
    min_tokens = max_tokens = None
    kinds = Counter()
    ids = set()
    scope_limited = False
    with path.open("rb") as stream:
        while raw := stream.readline(row_limit_bytes + 1):
            if len(raw) > row_limit_bytes:
                raise ValueError("Source row exceeds the explicit bounded JSONL record size")
            source_hash.update(raw)
            if not raw.strip():
                continue
            lines += 1
            if traces >= max_traces or (max_gaps is not None and gaps >= max_gaps):
                scope_limited = True
                continue
            trace = json.loads(raw)
            validate_trace(trace)
            if trace["trace_id"] in ids:
                raise ValueError("Source contains duplicate trace IDs")
            ids.add(trace["trace_id"])
            traces += 1
            kinds.update(event.get("kind", "message") for event in trace["events"])
            for gap in gaps_for(trace):
                if max_gaps is not None and gaps >= max_gaps:
                    scope_limited = True
                    break
                gaps += 1
                total_gap_bytes += len(canonical(gap).encode())
                if tokenizer is not None:
                    count = tokenizer.count(prompt_messages(gap), config.get("chat_template_kwargs", {}))
                    checked_tokens += 1
                    min_tokens = count if min_tokens is None else min(count, min_tokens)
                    max_tokens = count if max_tokens is None else max(count, max_tokens)
                    try:
                        check_budget(count, int(config["max_output_tokens"]), int(config["context_limit"]), int(config.get("safety_margin", 256)))
                    except ContextOverflow:
                        oversized += 1
    if before != _signature(path.stat()):
        raise ValueError("Source changed during preflight; discard this result")
    return {
        "schema": "cot.source-preflight/v1", "source_path": str(path),
        "source_file_sha256": source_hash.hexdigest(), "source_bytes": before[2],
        "source_jsonl_records": lines, "validated_traces": traces,
        "validated_gaps": gaps, "validation_scope_limited": scope_limited,
        "visible_event_kinds": dict(kinds), "prompt_version": PROMPT_VERSION,
        "projected_gap_json_bytes": total_gap_bytes,
        "gap_storage_note": "Existing Store duplicates the full prefix per gap; this is JSON bytes before SQLite overhead.",
        "context_checked_gaps": checked_tokens, "context_rejected_gaps": oversized,
        "prompt_tokens_min": min_tokens, "prompt_tokens_max": max_tokens,
        "tokenizer_identity": getattr(tokenizer, "identity", None),
        "inference_started": False, "semantic_quality_validated": False,
        "ready_for_bulk_inference": False,
        "remaining_checks": ["freeze selected label/source identities", "shared endpoint reservation", "scalable execution journal", "prefix-only semantic review for every candidate"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", help="Fresh canonical cot.trace.v1 JSONL")
    group.add_argument("--sidecar", help="Read-only legacy SQLite compatibility audit")
    parser.add_argument("--config", help="Existing teacher config for CPU tokenizer counting only")
    parser.add_argument("--max-traces", type=int, default=20000)
    parser.add_argument("--max-gaps", type=int)
    parser.add_argument("--output", help="New numeric report file (exclusive creation)")
    args = parser.parse_args(argv)
    if args.sidecar:
        if args.config:
            parser.error("--config applies only to --source")
        result = audit_sidecar(args.sidecar)
    else:
        config = json.loads(Path(args.config).read_text()) if args.config else None
        tokenizer = None
        if config:
            kind = DeepSeekV4Tokenizer if config.get("tokenizer_type") == "deepseek_v4" else LocalTokenizer
            tokenizer = kind(config["tokenizer_path"])
        result = audit_source(args.source, tokenizer=tokenizer, config=config, max_traces=args.max_traces, max_gaps=args.max_gaps)
    if args.output:
        with Path(args.output).open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(canonical(result))


if __name__ == "__main__":
    main()
