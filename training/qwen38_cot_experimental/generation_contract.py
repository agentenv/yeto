"""Bind generated text to original placement and durable generation receipts.

These checks establish provenance and structural completeness, never semantic
quality or absence of future information. No inference/review is performed.
"""
from copy import deepcopy
import json

from cot_filler.core import BLOCKING_FLAGS, PROMPT_VERSION, canonical, digest, validate_candidate, validate_trace
from cot_filler.corpus_worker import gap_at

ACCEPTANCE_POLICY = "all-structurally-valid-frozen-generations-experimental/v1"
VERSION = "cot.frozen-generation-receipt-binding/v1"


def _object(value):
    value = json.loads(value) if isinstance(value, str) else deepcopy(value)
    if not isinstance(value, dict):
        raise ValueError("A genuine journal metadata object is required")
    return value


def bind_generations(original_trace, generated_candidates, generation_config):
    """Return exact original source, event-to-text mapping and receipt references.

Entries are copied from the immutable original generation snapshot: gap,
candidate and the matching responses row. Model review state is deliberately
irrelevant to this explicitly unreviewed experiment.
"""
    source = validate_trace(deepcopy(original_trace))
    if not isinstance(generated_candidates, list) or not isinstance(generation_config, dict):
        raise ValueError("Explicit frozen generations and original configuration are required")
    if not isinstance(generation_config.get("model"), str) or not generation_config["model"]:
        raise ValueError("The original configured generation model is required")
    signature = digest(source)
    positions = {event["event_id"]: i for i, event in enumerate(source["events"])}
    reasoning, references = {}, []
    for entry in generated_candidates:
        if not isinstance(entry, dict) or any(k not in entry for k in ("gap", "candidate", "generation_receipt")):
            raise ValueError("Each filled gap requires original gap/candidate/response rows")
        saved, candidate, receipt = (entry[k] for k in ("gap", "candidate", "generation_receipt"))
        if not all(isinstance(row, dict) for row in (saved, candidate, receipt)):
            raise ValueError("Frozen journal rows must be explicit objects")
        event_id = saved.get("event_id")
        if event_id not in positions or event_id in reasoning:
            raise ValueError("Generated reasoning has a missing or duplicate original target")
        gap = gap_at(source, positions[event_id], signature)
        if (saved.get("id") != gap["id"] or saved.get("event_index") != positions[event_id]
                or ("source_digest" in saved and saved["source_digest"] != signature)):
            raise ValueError("Generated rationale moved from its original reasoning boundary")
        text = candidate.get("text")
        if not isinstance(text, str) or BLOCKING_FLAGS.intersection(validate_candidate(text)):
            raise ValueError("Only structurally complete original generations may be filled")
        if (candidate.get("gap_id") != gap["id"] or receipt.get("gap_id") != gap["id"]
                or receipt.get("stage") != "generate" or receipt.get("text") != text
                or any(candidate.get(k) != gap[k] for k in ("prefix_hash", "lookahead_hash", "prompt_hash"))):
            raise ValueError("Candidate and original generation receipt do not match this exact window")
        raw, stored = _object(receipt.get("data")), _object(candidate.get("data"))
        generator = raw.get("generator", {})
        expected_parameters = {"model": generation_config["model"],
            "max_tokens": int(generation_config["max_output_tokens"]),
            "temperature": generation_config.get("temperature", 0.3)}
        expected_parameters.update({k: generation_config[k] for k in ("reasoning_effort", "chat_template_kwargs")
                                    if k in generation_config})
        if (raw.get("finish_reason") != "stop" or generator.get("provider") != "openai-compatible"
                or generator.get("model") != generation_config["model"]
                or generator.get("prompt_version") != PROMPT_VERSION
                or generator.get("parameters") != expected_parameters
                or raw.get("prompt_hash") != gap["prompt_hash"]
                or raw.get("counted_chat_template_kwargs") != generation_config.get("chat_template_kwargs", {})
                or not isinstance(generator.get("tokenizer_sha256"), str) or not generator["tokenizer_sha256"]
                or type(raw.get("prompt_tokens_local")) is not int or raw["prompt_tokens_local"] < 1
                or not isinstance(raw.get("usage"), dict)
                or raw.get("synthetic") is not True or raw.get("lookahead_conditioned") is not True):
            raise ValueError("Original generation provider/configuration/completion provenance changed")
        flags = sorted(set(validate_candidate(text) + raw.get("flags", [])))
        if BLOCKING_FLAGS.intersection(flags) or "demo_fixture_not_quality_evidence" in flags:
            raise ValueError("Incomplete or fixture generation cannot enter the experiment")
        expected_stored = {**raw, "flags": flags, "candidate_hash": digest(text),
                           "synthetic": True, "lookahead_conditioned": True}
        if canonical(stored) != canonical(expected_stored):
            raise ValueError("Stored candidate no longer matches its durable original response")
        reasoning[event_id] = text
        references.append({"event_id": event_id, "gap_id": gap["id"], "candidate_sha256": digest(text),
            "source_digest": signature, "prefix_hash": gap["prefix_hash"], "lookahead_hash": gap["lookahead_hash"],
            "prompt_hash": gap["prompt_hash"], "generation_receipt_sha256": digest(receipt),
            "candidate_record_sha256": digest(candidate), "generation_config_sha256": digest(generation_config)})
    references.sort(key=lambda ref: positions[ref["event_id"]])
    return source, reasoning, references
