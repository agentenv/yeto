"""Qwen3.8 no-CoT rendering with the manager-authorized prefix token cutoff.

Uses the pinned, byte-parity-checked manager renderer unchanged. There is no
secret redaction or retokenization. Token labels remain unshifted; the training
loader must apply the causal shift exactly once.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .upstream_manager.render import (
    Qwen38Renderer as ExactRenderer,
    UnsupportedMessage,
    UnsafeTokenBoundary,
    adapt_custom_tool,
)

MAX_SEQUENCE_LENGTH = 262144
SEQUENCE_POLICY = "first_262144_tokens_drop_remainder/v1"
MASK_POLICY = "assistant_content_and_eos_only_no_cot_no_headers_v1"


def prefix_token_cutoff(result: dict, *, max_tokens: int = MAX_SEQUENCE_LENGTH) -> dict:
    """Keep the first N complete tokenizer IDs and their original labels.

    This never appends an end-of-turn token or trains the first token. When a
    target is cut mid-answer its missing suffix is counted, not synthesized.
    Rendered text/spans remain the *pre-cutoff* audit evidence; consumers must
    use only input_ids/labels/attention_mask, never tokenize that evidence again.
    """
    if type(max_tokens) is not int or not 1 <= max_tokens <= MAX_SEQUENCE_LENGTH:
        raise ValueError("max_tokens must be between 1 and 262144")
    ids, labels, mask, attention = (result[name] for name in
                                  ("input_ids", "labels", "loss_mask", "attention_mask"))
    length = len(ids)
    if not length or any(len(values) != length for values in (labels, mask, attention)):
        raise UnsafeTokenBoundary("Token IDs, labels, loss mask and attention mask differ in length")
    for token, label, active, attended in zip(ids, labels, mask, attention):
        if type(token) is not int or token < 0 or active not in (0, 1) or attended != 1:
            raise UnsafeTokenBoundary("Invalid unpadded token or binary-mask contract")
        if label != (token if active else -100):
            raise UnsafeTokenBoundary("Labels disagree with exact token loss mask")
    if labels[0] != -100:
        raise UnsafeTokenBoundary("The first token has no preceding causal prediction and must be masked")
    kept = min(length, max_tokens)
    if not any(mask[:kept]):
        raise UnsupportedMessage("The retained prefix contains no assistant training targets")
    output = dict(result)
    for name in ("input_ids", "labels", "loss_mask", "attention_mask", "token_offsets"):
        if name in result:
            output[name] = result[name][:kept]
    output["full_rendered_text"] = output.pop("rendered_text", None)
    output["full_assistant_spans"] = output.pop("assistant_spans", [])
    output["input_tokens"] = kept
    output["supervised_tokens"] = sum(mask[:kept])
    output["sequence_audit"] = {
        "policy": SEQUENCE_POLICY,
        "max_sequence_length": max_tokens,
        "original_input_tokens": length,
        "retained_input_tokens": kept,
        "dropped_input_tokens": length - kept,
        "original_supervised_tokens": sum(mask),
        "retained_supervised_tokens": sum(mask[:kept]),
        "dropped_supervised_tokens": sum(mask[kept:]),
        "truncated": kept != length,
        "eot_appended": False,
        "causal_shift_applied": False,
    }
    return output


class Qwen38Renderer:
    """Exact model-native renderer plus explicit prefix truncation policy."""

    def __init__(self, tokenizer_dir, *, max_tokens=MAX_SEQUENCE_LENGTH):
        if type(max_tokens) is not int or not 1 <= max_tokens <= MAX_SEQUENCE_LENGTH:
            raise ValueError("max_tokens must be between 1 and 262144")
        self.max_tokens = max_tokens
        self.exact = ExactRenderer(tokenizer_dir, max_tokens=None)
        self.tokenizer = self.exact.tokenizer
        self.identity = deepcopy(self.exact.identity)
        self.identity.update({
            "sequence_policy": SEQUENCE_POLICY,
            "max_sequence_length": max_tokens,
            "mask_policy": MASK_POLICY,
            "secret_redaction": False,
            "prefix_adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        })

    def render(self, messages, tools=None):
        result = prefix_token_cutoff(self.exact.render(messages, tools), max_tokens=self.max_tokens)
        result["identity"] = self.identity
        return result


def training_row(result, *, group_id: str, source: str, provenance=None):
    """Emit trainer-ready arrays without copying source text into metadata."""
    if not isinstance(group_id, str) or not group_id:
        raise ValueError("A stable source group_id is required")
    if source not in {"trace", "replay"}:
        raise ValueError("source must be trace or replay")
    return {
        "input_ids": result["input_ids"],
        "labels": result["labels"],
        "attention_mask": result["attention_mask"],
        "group_id": group_id,
        "source": source,
        "metadata": {
            "sequence_audit": result["sequence_audit"],
            "renderer_identity": result["identity"],
            "full_rendered_sha256": hashlib.sha256(result["full_rendered_text"].encode()).hexdigest(),
            "retained_token_ids_sha256": hashlib.sha256(json.dumps(result["input_ids"], separators=(",", ":")).encode()).hexdigest(),
            "normalization": result.get("normalization", {}),
            "control_escape_count": len(result.get("control_escape_edits", [])),
            "provenance": provenance or {},
        },
    }
