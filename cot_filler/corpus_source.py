"""Freeze selected normalized archives as clean, original canonical CoT inputs."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re

from .core import REASONING_FIELDS, canonical, digest, validate_trace


SOURCE_VERSION = "cot.selected-normalized-archives/v2-typed-text"
PRIVATE_BLOCK = re.compile(r"<(think|thinking|analysis|reasoning|cot)\b[^>]*>.*?(?:</\1\s*>|\Z)", re.I | re.S)
PRIVATE_MARKER = re.compile(r"\[thinking\]|</?(?:think|thinking|analysis|reasoning|cot)\b", re.I)


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def _visible(value):
    if not isinstance(value, str):
        raise ValueError("Assistant visible text must be a string")
    value = PRIVATE_BLOCK.sub("", value)
    if "</think>" in value:
        value = value.rsplit("</think>", 1)[-1]
    if PRIVATE_MARKER.search(value):
        raise ValueError("Unresolved inline reasoning marker")
    return value


def _guard(value):
    if not isinstance(value, dict):
        return
    if any(key in REASONING_FIELDS or "encrypt" in key.lower() for key in value):
        raise ValueError("Unclean reasoning schema boundary")
    if value.get("channel") in {"analysis", "reasoning"} or value.get("type") in {"reasoning", "thinking", "redacted_thinking", "encrypted_reasoning"}:
        raise ValueError("Reasoning item at visible schema boundary")
    content = value.get("content")
    if isinstance(content, list) and any(isinstance(b, dict) and b.get("type") in {"reasoning", "thinking", "redacted_thinking", "encrypted_reasoning"} for b in content):
        raise ValueError("Reasoning block inside visible content")


def _visible_message_value(value):
    """Clean explicit typed text blocks without flattening or reinterpreting them.

    Canonical archives already supply a string visible projection, but retain
    the original agent_message content list in data.block. Keep that original
    structure and metadata while applying the same reasoning removal to each
    known text/refusal field. Media, reasoning, tools, and unknown blocks remain
    exclusions instead of being silently discarded or converted to text.
    """
    if isinstance(value, str):
        return _visible(value)
    if not isinstance(value, list):
        raise ValueError("Assistant message content must be text or typed text blocks")
    cleaned = []
    for block in value:
        if not isinstance(block, dict) or block.get("type") not in {"text", "input_text", "output_text", "refusal"}:
            raise ValueError("Unsupported assistant message content block")
        _guard(block)
        required = "refusal" if block["type"] == "refusal" else "text"
        if required not in block:
            raise ValueError("Assistant typed text block is missing its text field")
        block = copy.deepcopy(block)
        for key in ("text", "refusal"):
            if key in block:
                block[key] = _visible(block[key])
        cleaned.append(block)
    return cleaned


def _clean_assistant(event):
    if event["role"] != "assistant":
        return event
    _guard(event)
    _guard(event.get("data"))
    if isinstance(event.get("data"), dict):
        _guard(event["data"].get("message_fields"))
        _guard(event["data"].get("block"))
    if event["kind"] != "message":
        return event
    event["content"] = _visible(event["content"])
    payload = event["data"]
    if not isinstance(payload, dict):
        raise ValueError("Unknown assistant message payload")
    if "block" not in payload:
        raise ValueError("Assistant archive message needs explicit block wrapper")
    block = payload["block"]
    if isinstance(block, str):
        payload["block"] = _visible(block)
    elif isinstance(block, dict) and block.get("type") in {"text", "input_text", "output_text", "refusal"}:
        for key in ("text", "refusal"):
            if key in block:
                block[key] = _visible(block[key])
    elif isinstance(block, dict) and block.get("type", "message") in {"message", "agent_message"}:
        for key in ("text", "content", "message"):
            if key in block:
                block[key] = _visible_message_value(block[key])
    else:
        raise ValueError("Unknown assistant archive message block")
    return event


def normalize_archive(archive, record):
    """Preserve original event order and exact marker-to-assistant boundaries.

    Original reasoning payloads are removed without inspecting or interpreting
    them. Marker event IDs alone locate replacement gaps. Ambiguous boundaries
    and unknown event shapes reject a trace instead of moving the insertion.
    """
    if not isinstance(archive, list) or not archive:
        raise ValueError("Archive must contain events")
    events, pending, targets = [], [], []
    removed = skipped_empty = cleaned = 0
    ids = set()
    for position, original in enumerate(archive):
        if not isinstance(original, dict) or not {"event_id", "role", "kind", "content", "data"} <= original.keys():
            raise ValueError("Unknown normalized archive event shape")
        event_id = original["event_id"]
        if not isinstance(event_id, str) or event_id in ids:
            raise ValueError("Invalid or duplicate original event IDs")
        ids.add(event_id)
        kind = original["kind"]
        if kind == "reasoning" or original.get("channel") in {"analysis", "reasoning"}:
            pending.append({"event_id": event_id, "archive_position": position})
            removed += 1
            continue
        if kind in {"empty_message", "capture_metadata"}:
            continue
        if kind not in {"message", "tool_call", "tool_result", "tool_definition"}:
            raise ValueError("Unsupported normalized archive event kind: " + str(kind))
        event = copy.deepcopy(original)
        event["role"] = {"agent": "assistant"}.get(event["role"], event["role"])
        before_clean = copy.deepcopy(event)
        event = _clean_assistant(event)
        cleaned += event != before_clean
        if kind == "message" and not event["content"]:
            skipped_empty += 1
            continue
        if pending:
            if event["role"] != "assistant" or kind == "tool_definition":
                raise ValueError("Reasoning marker does not immediately precede an assistant action")
            targets.append({"event_id": event_id, "reason": "original_reasoning_removed", "source_reasoning_events": pending,
                            "visible_position": len(events), "eligibility_policy": SOURCE_VERSION})
            event["original_reasoning_present"] = True
            pending = []
        if kind in {"tool_call", "tool_result", "tool_definition"}:
            expected = copy.deepcopy(original)
            expected["role"] = {"agent": "assistant"}.get(expected["role"], expected["role"])
            if {k: v for k, v in event.items() if k != "original_reasoning_present"} != expected:
                raise ValueError("Tool payload or ordering was changed by normalization")
        events.append(event)
    if pending:
        raise ValueError("Trailing reasoning marker has no original assistant action")
    if not targets:
        raise ValueError("No original reasoning replacement boundaries")
    trace = {"schema": "cot.trace.v1", "trace_id": record["capture_id"], "source_format": SOURCE_VERSION,
             "events": events, "gap_targets": targets,
             "metadata": {"selection": copy.deepcopy(record), "projection_version": SOURCE_VERSION,
                          "removed_original_reasoning_events": removed, "cleaned_assistant_messages": cleaned,
                          "skipped_empty_messages": skipped_empty,
                          "gap_policy": "all original reasoning boundaries, including final short lookahead",
                          "original_reasoning_encryption_status": "not_inferred"}}
    return validate_trace(trace)


def prepare_source(manifest, output, report_path, *, expected_manifest_sha256, max_traces=20000, max_archive_bytes=64 * 1024 * 1024):
    if type(max_traces) is not int or not 1 <= max_traces <= 20000:
        raise ValueError("Selected source is bounded to at most 20000 traces")
    manifest, output, report_path = Path(manifest).resolve(), Path(output).resolve(), Path(report_path).resolve()
    if len({manifest, output, report_path}) != 3:
        raise ValueError("Source, manifest, and report paths must differ")
    if file_sha256(manifest) != expected_manifest_sha256:
        raise ValueError("Frozen selection manifest hash mismatch")
    if output.exists() or report_path.exists():
        raise ValueError("New source/report paths are required; existing runs are immutable")
    selected = retained = gaps = total_bytes = 0
    seen = set()
    errors = []
    source_hash = hashlib.sha256()
    with manifest.open("r", encoding="utf-8") as src, output.open("xb") as dst:
        for line in src:
            if not line.strip():
                continue
            if selected >= max_traces:
                break
            record = json.loads(line)
            capture_id = record["capture_id"]
            if not isinstance(capture_id, str) or capture_id in seen:
                raise ValueError("Duplicate or invalid selected capture ID")
            seen.add(capture_id)
            selected += 1
            quality, confidence = record.get("quality_score"), record.get("overall_confidence_score")
            if type(quality) not in {int, float} or type(confidence) not in {int, float} or not (4 <= quality <= 5 and 3 <= confidence <= 5):
                raise ValueError("Frozen selection violates the requested 5-point quality/confidence filter")
            try:
                archive_path = Path(record["archive_path"])
                if archive_path.is_symlink() or archive_path.stat().st_size > max_archive_bytes:
                    raise ValueError("Archive is a symlink or exceeds source record bound")
                raw = archive_path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != record["archive_file_sha256"]:
                    raise ValueError("Selected archive SHA-256 mismatch")
                archive = [json.loads(event) for event in raw.splitlines() if event.strip()]
                trace = normalize_archive(archive, record)
                encoded = (canonical(trace) + "\n").encode()
                dst.write(encoded)
                source_hash.update(encoded)
                retained += 1
                gaps += len(trace["gap_targets"])
                total_bytes += len(encoded)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                # Shape errors are numeric/provenance findings, not fabricated fixes.
                errors.append({"capture_id": capture_id, "category": type(exc).__name__, "reason": str(exc)})
        dst.flush()
        __import__("os").fsync(dst.fileno())
    if file_sha256(manifest) != expected_manifest_sha256:
        raise ValueError("Selection manifest changed during preparation; discard unsealed source")
    report = {"schema": SOURCE_VERSION, "manifest_path": str(manifest), "manifest_sha256": expected_manifest_sha256,
              "source_path": str(output), "source_file_sha256": source_hash.hexdigest(), "source_bytes": total_bytes,
              "selected_traces": selected, "retained_traces": retained, "eligible_gaps": gaps,
              "excluded_traces": errors, "inference_started": False, "automatic_approval": False}
    with report_path.open("x", encoding="utf-8") as dst:
        dst.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-traces", type=int, default=20000)
    args = parser.parse_args(argv)
    report = prepare_source(args.manifest, args.output, args.report, expected_manifest_sha256=args.manifest_sha256, max_traces=args.max_traces)
    print(canonical({k: v for k, v in report.items() if k != "excluded_traces"} | {"excluded_count": len(report["excluded_traces"])}))


if __name__ == "__main__":
    main()
