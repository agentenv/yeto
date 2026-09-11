"""Render accepted synthetic reasoning as masked Qwen3.8 xhigh input.

Only explicit source-bound prefix-review receipts admit filled reasoning. Loss
masking does not remove information from model input and is not a leakage test.
This module does not review candidates, read a live journal, or start training.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
from pathlib import Path
import re

from transformers.utils.chat_template_utils import _compile_jinja_template, _render_with_assistant_indices

from cot_filler.core import BLOCKING_FLAGS, canonical, digest, validate_candidate, validate_trace
from cot_filler.corpus_worker import gap_at
from cot_filler.regeneration import revalidate_review
from training.qwen38_no_cot.prepare_data import (
    UnsupportedTrace, observation, tool_call, validate_messages, visible_text,
)
from training.qwen38_no_cot.render import (
    MAX_SEQUENCE_LENGTH, SEQUENCE_POLICY, prefix_token_cutoff, training_row as baseline_training_row,
)
from training.qwen38_no_cot.upstream_manager.render import (
    EOT, Qwen38Renderer as BaselineExactRenderer, UnsupportedMessage,
    UnsafeTokenBoundary, _replace_once, align_loss_mask,
)

VERSION = "qwen3.8-xhigh-reviewed-cot-input-only/v1"
MASK_POLICY = "assistant_content_and_eos_only_reviewed_cot_masked_xhigh_v1"
ADAPTER_VERSION = "reviewed-events-to-native-qwen-reasoning-content/v1"
TEMPLATE_OPTIONS = {"enable_thinking": True, "preserve_thinking": True,
                    "reasoning_effort": "xhigh", "add_vision_id": False}
_SOURCE_LEADING_REASONING = re.compile(
    r"\A\s*<(think|thinking|analysis|reasoning)\b[^>]*>.*?(?:</\1\s*>|\Z)", re.I | re.S)


def _source_text(value, role, counts):
    """Match the no-CoT adapter's per-event visible-text projection."""
    text = visible_text(value)
    if role == "assistant":
        while (cleaned := _SOURCE_LEADING_REASONING.sub("", text, count=1)) != text:
            text = cleaned
            counts["leading_reasoning_blocks_removed_before_turn_mapping"] += 1
    return text


def _approved_reasoning(row, original_trace, approved_reviews, reviewer_config):
    """Recheck source, coverage, insertion positions and existing model receipts."""
    source = validate_trace(deepcopy(original_trace))
    if not isinstance(row, dict) or row.get("schema") != "cot.reviewed-events.v1":
        raise ValueError("Expected an explicit reviewed-events export")
    meta = row.get("metadata", {})
    if (row.get("trace_id") != source["trace_id"] or meta.get("source_digest") != digest(source)
            or meta.get("arm") != "masked" or meta.get("synthetic_reasoning") is not True):
        raise ValueError("Reviewed export must match the original source and masked arm")
    events = row.get("events")
    if not isinstance(events, list) or not events or not isinstance(approved_reviews, dict):
        raise ValueError("Export events and explicit approved review records are required")
    if not isinstance(reviewer_config, dict) or not reviewer_config.get("model"):
        raise ValueError("The immutable reviewer configuration is required")
    originals = [e for e in events if isinstance(e, dict) and e.get("kind") != "synthetic_rationale"]
    if canonical(originals) != canonical(source["events"]):
        raise ValueError("Export changed, omitted or reordered original events")
    refs = meta.get("approved_gaps")
    if not isinstance(refs, list) or not refs or any(not isinstance(r, dict) for r in refs):
        raise ValueError("Explicit approved gap references are required")
    references = {r.get("gap_id"): r for r in refs}
    if len(references) != len(refs) or None in references:
        raise ValueError("Duplicate or invalid approved gap references")
    positions = {event["event_id"]: i for i, event in enumerate(source["events"])}
    source_digest = digest(source)
    reasoning, provenance, seen_ids, expected_segments = {}, [], set(), []
    for index, event in enumerate(events):
        if not isinstance(event, dict) or not isinstance(event.get("event_id"), str):
            raise ValueError("Every exported event requires an explicit event ID")
        if event["event_id"] in seen_ids:
            raise ValueError("Duplicate exported event IDs")
        seen_ids.add(event["event_id"])
        synthetic = event.get("kind") == "synthetic_rationale"
        expected_segments.append({"event_id": event["event_id"],
                                  "loss_intent": "mask" if synthetic or event.get("role") != "assistant" else "train"})
        if not synthetic:
            continue
        target_id = event.get("before_event_id")
        if (index + 1 >= len(events) or events[index + 1].get("event_id") != target_id
                or target_id not in positions or event.get("role") != "assistant"
                or event.get("synthetic") is not True or target_id in reasoning):
            raise ValueError("Rationale must immediately precede its unique original assistant action")
        gap = gap_at(source, positions[target_id], source_digest)
        ref = references.get(gap["id"])
        text = event.get("content")
        if (not isinstance(text, str) or not text.strip()
                or BLOCKING_FLAGS.intersection(validate_candidate(text))):
            raise ValueError("Accepted rationale failed structural validation")
        if (not ref or ref.get("text_sha256") != digest(text)
                or ref.get("prompt_hash") != gap["prompt_hash"]):
            raise ValueError("Accepted candidate reference differs from the original gap or text")
        record = approved_reviews.get(gap["id"])
        validated = revalidate_review(gap, text, record, reviewer_config)
        if validated["decision"] != "pass":
            raise ValueError("Only passing prefix-only model reviews admit synthetic reasoning")
        reasoning[target_id] = text
        provenance.append({"event_id": target_id, "gap_id": gap["id"], "candidate_sha256": digest(text),
                           "source_digest": source_digest, "prefix_hash": gap["prefix_hash"],
                           "review_record_sha256": digest(record), "review_prompt_hash": validated["review_prompt_hash"]})
    if not reasoning or {p["gap_id"] for p in provenance} != set(references):
        raise ValueError("Approved references and inserted reasoning must have exact coverage")
    if canonical(meta.get("loss_segments")) != canonical(expected_segments):
        raise ValueError("Export loss segments disagree with masked reasoning and original events")
    return source, reasoning, provenance


def _canonical_messages(events, reasoning, *, event_boundaries=None):
    """Keep baseline source adaptations, retaining exact filled action boundaries."""
    messages, mappings, counts = [], [], Counter()
    previous_key = None
    message_boundaries = []
    if event_boundaries is not None and (
            not isinstance(event_boundaries, (list, tuple)) or len(event_boundaries) != len(events)
            or any(not isinstance(item, dict) for item in event_boundaries)):
        raise ValueError("Event boundaries must align with source events")
    for event_index, event in enumerate(events):
        kind, role = event.get("kind"), event.get("role")
        data = event.get("data", {})
        block = data.get("block", data) if isinstance(data, dict) else data
        source = event.get("source", {})
        pointer = source.get("pointer", event.get("event_id")) if isinstance(source, dict) else event.get("event_id")
        boundary = event_boundaries[event_index] if event_boundaries is not None else None
        event_id = event["event_id"]
        filled = reasoning.get(event_id)
        if kind == "tool_definition":
            if filled is not None:
                raise UnsupportedTrace("filled_gap_targets_capability_definition")
            counts["capability_schemas_omitted"] += 1
            continue
        if kind in {"empty_message", "capture_metadata"}:
            if filled is not None:
                raise UnsupportedTrace("filled_gap_targets_nonvisible_event")
            counts["nonvisible_events_removed"] += 1
            continue
        if kind == "unknown" and isinstance(block, dict) and block.get("type") == "tool_search_output":
            kind, role = "tool_result", "tool"
            counts["tool_search_output_as_observation"] += 1
        elif (kind == "unknown" and block is None and not event.get("content")
              and event_index == len(events) - 1 and isinstance(pointer, str) and pointer.startswith("/response/")):
            if filled is not None:
                raise UnsupportedTrace("filled_gap_targets_empty_terminal_event")
            counts["empty_terminal_capture_event_omitted"] += 1
            continue
        if kind in {"media", "unknown", "compaction"}:
            raise UnsupportedTrace("unsupported_canonical_" + kind)
        if role == "developer":
            role = "system"
            counts["developer_as_chronological_system"] += 1
        if role == "agent":
            role = "assistant"
        key = re.sub(r"/content/\d+$", "", pointer) if isinstance(pointer, str) else None
        if filled is not None and role != "assistant":
            raise UnsupportedTrace("filled_gap_did_not_map_to_assistant")
        if kind == "message":
            if role not in {"system", "user", "assistant"}:
                raise UnsupportedTrace("unsupported_message_role")
            if isinstance(block, dict) and block.get("type") not in {None, "message", "agent_message", "text", "input_text", "output_text", "refusal"}:
                raise UnsupportedTrace("unsupported_message_block")
            text = _source_text(event.get("content"), role, counts)
            if (filled is None and key is not None and previous_key == (key, role) and messages
                    and messages[-1]["role"] == role and "tool_calls" not in messages[-1]
                    and (event_boundaries is None or message_boundaries[-1] == boundary)):
                messages[-1]["content"] += text
                mappings[-1].append(event_id)
                counts["same_source_message_blocks_joined"] += 1
                continue
            message = {"role": role, "content": text}
            previous_key = (key, role)
        elif kind == "tool_call":
            if role != "assistant" or not isinstance(block, dict):
                raise UnsupportedTrace("invalid_canonical_tool_call")
            if (event_index == len(events) - 1 and isinstance(pointer, str) and pointer.startswith("/response/output/")
                    and block.get("type") == "function_call" and "arguments" not in block
                    and "input" not in block and "function" not in block):
                if filled is not None:
                    raise UnsupportedTrace("filled_gap_targets_incomplete_terminal_call")
                counts["incomplete_terminal_call_omitted"] += 1
                continue
            calls = [tool_call(block, name=event.get("name"), call_id=event.get("call_id"), counts=counts)]
            # Standalone tool-call events often repeat the raw call payload in
            # ``event.content``.  The no-CoT source adapter deliberately ignores
            # that projection and trains only the canonical structured call.
            message = {"role": "assistant", "content": "", "tool_calls": calls}
            previous_key = None
        elif kind == "tool_result":
            if not isinstance(block, dict):
                raise UnsupportedTrace("invalid_canonical_tool_result")
            identity = (event.get("call_id") or block.get("call_id")
                        or block.get("tool_call_id") or block.get("tool_use_id"))
            if not isinstance(identity, str) or not identity:
                raise UnsupportedTrace("tool_result_missing_identity")
            message = {"role": "tool", "tool_call_id": identity, "content": observation(event.get("content"))}
            previous_key = None
        else:
            raise UnsupportedTrace("unknown_canonical_kind")
        if filled is not None:
            message["reasoning_content"] = filled
        messages.append(message)
        mappings.append([event_id])
        message_boundaries.append(boundary)
    validate_messages(messages)
    return messages, mappings, dict(counts)


class Qwen38MaskedCotRenderer:
    """Approved-export-only entry point; all arrays contain unshifted labels."""

    def __init__(self, tokenizer_dir, *, max_tokens=MAX_SEQUENCE_LENGTH):
        if type(max_tokens) is not int or not 1 <= max_tokens <= MAX_SEQUENCE_LENGTH:
            raise ValueError("max_tokens must be between 1 and 262144")
        self.max_tokens = max_tokens
        self._baseline = BaselineExactRenderer(tokenizer_dir, max_tokens=None)
        self.tokenizer = self._baseline.tokenizer
        official = (Path(tokenizer_dir) / "chat_template.jinja").read_text()
        plain = _replace_once(official,
            "{{- raise_exception('System message must be at the beginning.') }}",
            "{{- '<|im_start|>system\\n' + content + '<|im_end|>\\n' }}")
        before = """        {%- if preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index %}
            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\\n' + content }}
        {%- endif %}"""
        after = """        {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' }}
        {%- generation %}
        {{- content }}"""
        instrumented = _replace_once(plain, before, after)
        instrumented = _replace_once(instrumented,
            "        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}",
            "        {{- '<|im_end|>' }}\n        {%- endgeneration %}\n        {{- '\\n' }}\n    {%- elif message.role == \"tool\" %}")
        self._plain = _compile_jinja_template(plain)
        self._instrumented = _compile_jinja_template(instrumented)
        reasoning_instrumented = _replace_once(plain, before, """        {{- '<|im_start|>' + message.role + '\\n' }}
        {%- generation %}
        {{- '<think>\\n' + reasoning_content + '\\n</think>\\n\\n' }}
        {%- endgeneration %}
        {{- content }}""")
        self._reasoning_instrumented = _compile_jinja_template(reasoning_instrumented)
        self.identity = deepcopy(self._baseline.identity)
        self.identity.update({"version": VERSION, "mask_policy": MASK_POLICY,
            "adapter_version": ADAPTER_VERSION, **TEMPLATE_OPTIONS,
            "sequence_policy": SEQUENCE_POLICY, "max_sequence_length": max_tokens,
            "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "baseline_normalizer_sha256": self._baseline.identity["renderer_sha256"],
            "source_adapter_sha256": hashlib.sha256(Path(__file__).resolve().parents[1].joinpath("qwen38_no_cot/prepare_data.py").read_bytes()).hexdigest(),
            "adapted_template_sha256": hashlib.sha256(plain.encode()).hexdigest(),
            "loss_template_sha256": hashlib.sha256(instrumented.encode()).hexdigest(),
            "reasoning_audit_template_sha256": hashlib.sha256(reasoning_instrumented.encode()).hexdigest(),
            "prefix_adapter_sha256": hashlib.sha256(Path(__file__).resolve().parents[1].joinpath("qwen38_no_cot/render.py").read_bytes()).hexdigest(),
            "secret_redaction": False,
            "loss_0": ["system", "user", "tool_observations", "assistant_header", "reviewed_synthetic_reasoning",
                       "think_wrapper", "post_eot_newline"],
            "adaptations": ["later_system_messages_preserve_chronology", "reviewed_reasoning_content_input_only",
                            "assistant_body_and_eot_only_generation_spans"]})

    def render_reviewed_trace(self, row, *, original_trace, approved_reviews, reviewer_config):
        source, reasoning, refs = _approved_reasoning(row, original_trace, approved_reviews, reviewer_config)
        messages, event_mapping, counts = _canonical_messages(source["events"], reasoning)
        clean, safe_tools, normalization, edits = self._baseline._normalize(messages, None)
        # Baseline source sanitization must not admit arbitrary original reasoning.
        # Reinsert ONLY the accepted text checked above, escaped as ordinary data.
        for index, message in enumerate(messages):
            if "reasoning_content" in message:
                clean[index]["reasoning_content"] = self._baseline._escape(
                    message["reasoning_content"], f"/messages/{index}/reasoning_content", edits)
        text, spans = _render_with_assistant_indices(self._instrumented, clean, None, None, False, **TEMPLATE_OPTIONS)
        plain = self._plain.render(messages=clean, tools=None, documents=None,
                                   add_generation_prompt=False, **TEMPLATE_OPTIONS)
        if text != plain:
            raise UnsupportedMessage("Loss instrumentation changed native xhigh rendered bytes")
        reasoning_text, think_spans = _render_with_assistant_indices(
            self._reasoning_instrumented, clean, None, None, False, **TEMPLATE_OPTIONS)
        if reasoning_text != plain:
            raise UnsupportedMessage("Reasoning audit instrumentation changed native xhigh rendered bytes")
        encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                                 padding=False, truncation=False)
        ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
        mask = align_loss_mask(offsets, spans)
        if len(spans) != sum(m["role"] == "assistant" for m in clean):
            raise UnsafeTokenBoundary("Assistant generation-span coverage mismatch")
        assistants = [m for m in clean if m["role"] == "assistant"]
        if len(think_spans) != len(assistants):
            raise UnsafeTokenBoundary("Thinking-wrapper coverage mismatch")
        filled_spans = [span for span, message in zip(think_spans, assistants) if "reasoning_content" in message]
        if len(filled_spans) != len(refs):
            raise UnsafeTokenBoundary("Filled reasoning was omitted or duplicated during message mapping")
        reasoning_mask = align_loss_mask(offsets, filled_spans)
        if any(a and b for a, b in zip(mask, reasoning_mask)):
            raise UnsafeTokenBoundary("Reviewed reasoning overlaps a supervised token")
        retained_ranges = []
        cursor = 0
        for start, end in filled_spans:
            while cursor < len(offsets) and offsets[cursor][1] <= start:
                cursor += 1
            lo = cursor
            while cursor < len(offsets) and offsets[cursor][0] < end:
                cursor += 1
            hi = min(cursor, self.max_tokens)
            if lo < hi:
                retained_ranges.append([lo, hi])
        if any(text[end - len(EOT):end] != EOT for start, end in spans):
            raise UnsafeTokenBoundary("Assistant supervised span does not end at EOT")
        result = {"input_ids": ids, "labels": [i if m else -100 for i, m in zip(ids, mask)],
                  "loss_mask": mask, "attention_mask": [1] * len(ids), "rendered_text": text,
                  "assistant_spans": [{"start": start, "end": end} for start, end in spans],
                  "token_offsets": offsets, "normalization": {**normalization, **counts},
                  "control_escape_edits": edits, "identity": self.identity}
        result = prefix_token_cutoff(result, max_tokens=self.max_tokens)
        result["cot_provenance"] = {"source_digest": digest(source), "trace_id": source["trace_id"],
            "approved_gaps": refs, "reviewer_config_sha256": digest(reviewer_config),
            "event_message_mapping": event_mapping, "filled_gaps": len(refs),
            "model_review_revalidated": True, "semantic_certainty_claimed": False,
            "reasoning_loss": 0, "reasoning_effort": "xhigh"}
        result["cot_mask_audit"] = {"policy": "filled_native_think_wrapper_and_body_loss0/v1",
            "retained_token_ranges": retained_ranges, "filled_gap_count": len(refs),
            "approved_review_count": len(refs), "retained_filled_gap_count": len(retained_ranges)}
        return result


def training_row(result, *, group_id, source="trace", provenance=None):
    """Use existing unshifted arrays, with explicit masked-CoT provenance."""
    if result.get("identity", {}).get("mask_policy") != MASK_POLICY or not result.get("cot_provenance"):
        raise ValueError("Expected a validated masked-CoT renderer result")
    row = baseline_training_row(result, group_id=group_id, source=source,
                                provenance={**(provenance or {}), "masked_cot": result["cot_provenance"]})
    row["metadata"]["cot_mask_audit"] = deepcopy(result["cot_mask_audit"])
    return row
