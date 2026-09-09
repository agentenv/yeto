"""CPU-only, bounded cross-trace pilot selection from a frozen canonical source."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

from .core import canonical, digest, prompt_messages, validate_trace
from .corpus_source import file_sha256
from .corpus_worker import _budget, gap_at
from .grounding_review_v3 import review_messages
from .provider import ContextOverflow, DeepSeekV4Tokenizer, LocalTokenizer


POLICY = "cot.diverse-cross-trace-pilot/v2-disjoint-groups"
NON_LATIN = re.compile(r"[\u0400-\u04ff\u0600-\u06ff\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7a3]")


def selected_trace(trace, targets, provenance):
    """Only select existing boundaries; retain the entire original event array."""
    original = {target["event_id"]: target for target in trace["gap_targets"]}
    if len({t["event_id"] for t in targets}) != len(targets) or any(original.get(t["event_id"]) != t for t in targets):
        raise ValueError("Pilot targets must be distinct unchanged original boundaries")
    result = copy.deepcopy(trace)
    result["gap_targets"] = copy.deepcopy(targets)
    result["metadata"]["pilot_selection"] = {
        "policy": POLICY, "original_trace_digest": digest(trace),
        "original_gap_count": len(trace["gap_targets"]),
        "selected_event_ids": [target["event_id"] for target in targets], **provenance,
    }
    if result["events"] != trace["events"]:
        raise ValueError("Pilot selection changed original events")
    return validate_trace(result)


def prepare(source, expected_sha256, output, report_path, generator, reviewer, *, traces=24, pool_size=48, max_scan=1000,
            exclude_source=None, exclude_source_sha256=None):
    if not 1 <= traces <= 64 or not traces <= pool_size <= 128 or not pool_size <= max_scan <= 20000:
        raise ValueError("Pilot selection bounds are invalid")
    source, output, report_path = map(Path, (source, output, report_path))
    if output.exists() or report_path.exists() or source in {output, report_path}:
        raise ValueError("Pilot source/report paths must be new")
    before = source.stat()
    if file_sha256(source) != expected_sha256:
        raise ValueError("Frozen canonical source hash mismatch")
    after_hash = source.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after_hash.st_ino, after_hash.st_size, after_hash.st_mtime_ns, after_hash.st_ctime_ns):
        raise ValueError("Frozen source changed during initial hash verification")
    excluded_groups = set()
    if bool(exclude_source) != bool(exclude_source_sha256):
        raise ValueError("Excluded pilot requires both its source path and frozen hash")
    if exclude_source:
        excluded_bytes = Path(exclude_source).read_bytes()
        if hashlib.sha256(excluded_bytes).hexdigest() != exclude_source_sha256:
            raise ValueError("Excluded pilot source hash mismatch")
        for raw in excluded_bytes.splitlines():
            trace = validate_trace(json.loads(raw))
            excluded_groups.add(trace["metadata"]["selection"]["group_id"])
    pool, groups, exclusions = [], set(), Counter()
    scanned = 0
    with source.open("rb") as stream:
        while scanned < max_scan and len(pool) < pool_size:
            offset = stream.tell()
            raw = stream.readline(64 * 1024 * 1024 + 1)
            if not raw:
                break
            if len(raw) > 64 * 1024 * 1024 or not raw.endswith(b"\n"):
                raise ValueError("Incomplete or oversized source trace")
            scanned += 1
            trace = validate_trace(json.loads(raw))
            group = trace["metadata"]["selection"]["group_id"]
            if group in excluded_groups:
                exclusions["previous_pilot_group"] += 1
                continue
            if group in groups:
                exclusions["duplicate_group"] += 1
                continue
            targets = trace["gap_targets"]
            if len(targets) < 2:
                exclusions["fewer_than_two_original_gaps"] += 1
                continue
            positions = {event["event_id"]: i for i, event in enumerate(trace["events"])}
            # First boundary plus middle/quarter/eighth positions; fall back to
            # the second boundary when long original context exceeds capacity.
            trial = [0, len(targets)//2, len(targets)//4, len(targets)//8, 1]
            valid, budgets = [], []
            for index in dict.fromkeys(trial):
                target = targets[index]
                gap = gap_at(trace, positions[target["event_id"]])
                try:
                    generation_budget = _budget(generator, prompt_messages(gap))
                    review_budget = _budget(reviewer, review_messages(gap, "I will inspect the available evidence to decide the next appropriate step."))
                except ContextOverflow:
                    exclusions["candidate_gap_context_overflow"] += 1
                    continue
                valid.append(target)
                budgets.append({"event_id": target["event_id"], "visible_position": positions[target["event_id"]],
                                "generation_reserved_tokens": generation_budget,
                                "review_probe_reserved_tokens": review_budget})
                if len(valid) == 2:
                    break
            if len(valid) < 2:
                exclusions["fewer_than_two_fitting_gaps"] += 1
                continue
            groups.add(group)
            non_latin = sum(len(NON_LATIN.findall(e.get("content", ""))) for e in trace["events"] if e["kind"] == "message" and isinstance(e.get("content"), str))
            language = "non_latin_text_present" if non_latin >= 50 else "ascii_or_latin_text"
            shapes = sorted({str((e.get("data") or {}).get("block", e.get("data") or {}).get("type", "unspecified"))
                             for e in trace["events"] if e["kind"] == "tool_call" and isinstance(e.get("data"), dict) and isinstance((e.get("data") or {}).get("block", e.get("data") or {}), dict)})
            longest = max(x["generation_reserved_tokens"] for x in budgets)
            bucket = "under_32k" if longest < 32768 else "32k_to_128k" if longest < 131072 else "128k_to_context_limit"
            pool.append({"trace": trace, "targets": valid, "group": group, "language": language, "tools": "+".join(shapes), "length_bucket": bucket,
                         "offset": offset, "row_sha256": hashlib.sha256(raw).hexdigest(), "budgets": budgets})
    if len(pool) < traces:
        raise ValueError("Insufficient distinct fitting traces within the bounded scan")
    counts = {key: Counter() for key in ("language", "tools", "length_bucket")}
    chosen = []
    while len(chosen) < traces:
        item = min(pool, key=lambda x: (counts["language"][x["language"]], counts["length_bucket"][x["length_bucket"]], counts["tools"][x["tools"]], x["offset"]))
        pool.remove(item); chosen.append(item)
        for key in counts:
            counts[key][item[key]] += 1
    after = source.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("Frozen source changed during pilot selection")
    source_hash = hashlib.sha256()
    selections = []
    with output.open("xb") as dst:
        for item in sorted(chosen, key=lambda x: x["offset"]):
            provenance = {"source_path": str(source.resolve()), "source_sha256": expected_sha256,
                          "source_row_offset": item["offset"], "source_row_sha256": item["row_sha256"]}
            if exclude_source:
                provenance.update(excluded_pilot_source_sha256=exclude_source_sha256)
            trace = selected_trace(item["trace"], item["targets"], provenance)
            raw = (canonical(trace) + "\n").encode();dst.write(raw);source_hash.update(raw)
            selections.append({"trace_id": trace["trace_id"], "group_id": item["group"], "language": item["language"], "tool_shapes": item["tools"],
                               "length_bucket": item["length_bucket"], "gaps": item["budgets"], **provenance})
    report = {"schema": POLICY, "source_path": str(output.resolve()), "source_file_sha256": source_hash.hexdigest(),
              "traces": len(chosen), "distinct_groups": len(chosen), "selected_gaps": sum(len(x["targets"]) for x in chosen),
              "source_rows_scanned": scanned, "eligible_pool_size": len(chosen) + len(pool),
              "distributions": {k: dict(v) for k,v in counts.items()}, "selection_exclusions": dict(exclusions), "selections": selections,
              "review_budget_probe_only": True, "actual_candidate_budget_checked_before_each_request": True,
              "excluded_pilot_source_sha256": exclude_source_sha256, "excluded_group_count": len(excluded_groups),
              "inference_started": False, "semantic_quality_validated": False}
    with report_path.open("x") as f:
        f.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "source-sha256", "output", "report", "config", "review-config"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--traces", type=int, default=24)
    p.add_argument("--pool-size", type=int, default=48)
    p.add_argument("--max-scan", type=int, default=1000)
    p.add_argument("--exclude-source")
    p.add_argument("--exclude-source-sha256")
    args = p.parse_args()
    def local_counter(path):
        config = json.loads(Path(path).read_text())
        tokenizer = DeepSeekV4Tokenizer(config["tokenizer_path"]) if config.get("tokenizer_type") == "deepseek_v4" else LocalTokenizer(config["tokenizer_path"])
        return SimpleNamespace(config=config, tokenizer=tokenizer)
    result = prepare(args.source, args.source_sha256, args.output, args.report,
                     local_counter(args.config), local_counter(args.review_config), traces=args.traces,
                     pool_size=args.pool_size, max_scan=args.max_scan, exclude_source=args.exclude_source,
                     exclude_source_sha256=args.exclude_source_sha256)
    print(canonical({key: value for key, value in result.items() if key != "selections"}))


if __name__ == "__main__":
    main()
