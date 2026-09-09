"""Linear-size CPU audit of a completed selected-corpus source preparation.

Unlike per-gap prompt preflight, this never materializes every historical prefix.
Exact tokenizer budgets remain enforced just in time by the inference worker.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .core import digest, validate_trace
from .corpus_source import SOURCE_VERSION, file_sha256


def audit_prepared_source(report_path):
    report = json.loads(Path(report_path).read_text())
    if report["schema"] != SOURCE_VERSION:
        raise ValueError("Unknown source-preparation report version")
    manifest = Path(report["manifest_path"])
    if file_sha256(manifest) != report["manifest_sha256"]:
        raise ValueError("Frozen selection manifest changed")
    selections = {}
    with manifest.open() as stream:
        for line in stream:
            record = json.loads(line)
            if record["capture_id"] in selections:
                raise ValueError("Duplicate selected capture ID")
            selections[record["capture_id"]] = digest(record)
    source = Path(report["source_path"])
    before = source.stat()
    source_hash = hashlib.sha256()
    seen, kinds = set(), Counter()
    traces = gaps = total_events = 0
    with source.open("rb") as stream:
        while line := stream.readline(64 * 1024 * 1024 + 1):
            if len(line) > 64 * 1024 * 1024 or not line.endswith(b"\n"):
                raise ValueError("Incomplete or oversized canonical source row")
            source_hash.update(line)
            trace = validate_trace(json.loads(line))
            trace_id = trace["trace_id"]
            if trace_id in seen or trace_id not in selections:
                raise ValueError("Duplicate or unselected trace in prepared source")
            seen.add(trace_id)
            if trace.get("source_format") != SOURCE_VERSION or trace["metadata"].get("projection_version") != SOURCE_VERSION:
                raise ValueError("Source projection version changed")
            selection = trace["metadata"]["selection"]
            if digest(selection) != selections[trace_id]:
                raise ValueError("Source selection/label/archive provenance differs from the frozen manifest")
            if not (4 <= selection["quality_score"] <= 5 and 3 <= selection["overall_confidence_score"] <= 5):
                raise ValueError("Source trace violates the requested score predicate")
            positions = {event["event_id"]: i for i, event in enumerate(trace["events"])}
            for target in trace["gap_targets"]:
                if target["visible_position"] != positions[target["event_id"]] or target["reason"] != "original_reasoning_removed":
                    raise ValueError("Original reasoning replacement boundary changed")
                markers = target.get("source_reasoning_events", [])
                if not markers or any(marker["event_id"] in positions for marker in markers):
                    raise ValueError("Replacement marker provenance is missing or still visible")
                if any(a["archive_position"] >= b["archive_position"] for a, b in zip(markers, markers[1:])):
                    raise ValueError("Reasoning-marker chronology is not ordered")
            kinds.update(event["kind"] for event in trace["events"])
            total_events += len(trace["events"])
            gaps += len(trace["gap_targets"])
            traces += 1
    after = source.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("Source changed during full structural audit")
    if source_hash.hexdigest() != report["source_file_sha256"] or traces != report["retained_traces"] or gaps != report["eligible_gaps"]:
        raise ValueError("Prepared source bytes/counts do not match completion report")
    return {"schema": "cot.full-source-structural-audit/v1", "source_path": str(source),
            "source_file_sha256": source_hash.hexdigest(), "manifest_sha256": report["manifest_sha256"],
            "validated_traces": traces, "validated_gaps": gaps, "visible_events": total_events,
            "visible_event_kinds": dict(kinds), "excluded_traces": len(report["excluded_traces"]),
            "exclusion_reasons": dict(Counter(item["reason"] for item in report["excluded_traces"])),
            "structural_audit_passed": True, "context_counted_gaps": 0,
            "context_policy": "exact local token budget checked per request by corpus_worker before network admission",
            "semantic_quality_validated": False, "inference_started": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit_prepared_source(args.report)
    with Path(args.output).open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
