"""Native Qwen xhigh rendering for the explicitly unreviewed CoT experiment.

Generated reasoning conditions the model with loss zero. It may contain future
information; no semantic-review or leakage qualification is claimed.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

from transformers.utils.chat_template_utils import _render_with_assistant_indices

from cot_filler.core import digest
from training.qwen38_cot_masked import render as reviewed_render
from training.qwen38_no_cot.render import (
    MAX_SEQUENCE_LENGTH, SEQUENCE_POLICY, prefix_token_cutoff,
    training_row as baseline_training_row,
)
from training.qwen38_no_cot.upstream_manager.render import EOT, UnsupportedMessage, UnsafeTokenBoundary, align_loss_mask
from . import generation_contract
from .generation_contract import ACCEPTANCE_POLICY, bind_generations

VERSION = "qwen3.8-xhigh-generated-cot-input-only-experimental/v1"
MASK_POLICY = "assistant_content_and_eos_only_generated_cot_masked_xhigh_experimental_v1"
ADAPTER_VERSION = "original-events-to-native-generated-reasoning-content-experimental/v1"
TEMPLATE_OPTIONS = deepcopy(reviewed_render.TEMPLATE_OPTIONS)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Qwen38ExperimentalCotRenderer(reviewed_render.Qwen38MaskedCotRenderer):
    def __init__(self, tokenizer_dir, *, max_tokens=MAX_SEQUENCE_LENGTH):
        super().__init__(tokenizer_dir, max_tokens=max_tokens)
        self.identity.update({"version": VERSION, "mask_policy": MASK_POLICY, "adapter_version": ADAPTER_VERSION,
            "renderer_sha256": _sha(__file__), "reviewed_renderer_sha256": _sha(reviewed_render.__file__),
            "generation_contract_sha256": _sha(generation_contract.__file__),
            "acceptance_policy": ACCEPTANCE_POLICY, "semantic_quality_qualified": False,
            "future_information_checked": False,
            "loss_0": ["system", "user", "tool_observations", "assistant_header", "generated_synthetic_reasoning",
                       "think_wrapper", "post_eot_newline"],
            "adaptations": ["later_system_messages_preserve_chronology", "generated_reasoning_content_input_only",
                            "assistant_body_and_eot_only_generation_spans", "experimental_unreviewed_generations"]})

    def render_reviewed_trace(self, *args, **kwargs):
        raise ValueError("Use the separate reviewed renderer for reviewed exports")

    def render_generated_trace(self, original_trace, generated_candidates, generation_config):
        source, reasoning, refs = bind_generations(original_trace, generated_candidates, generation_config)
        messages, mapping, counts = reviewed_render._canonical_messages(source["events"], reasoning)
        provenance = {"source_digest": digest(source), "trace_id": source["trace_id"], "source_kind": "trace",
            "generated_gaps": refs, "generation_config_sha256": digest(generation_config),
            "event_message_mapping": mapping, "original_gap_count": len(source["gap_targets"]),
            "unfilled_gap_count": len(source["gap_targets"]) - len(refs),
            "original_trace_complete": True, "generation_lookahead_conditioned": bool(refs)}
        return self._render(messages, refs, provenance, counts, reinsert_generated=True)

    def render_replay_messages(self, messages, *, source_digest, trace_id):
        if (not isinstance(source_digest, str) or len(source_digest) != 64
                or not isinstance(trace_id, str) or not trace_id):
            raise ValueError("Replay requires the original source digest and stable trace identity")
        provenance = {"source_digest": source_digest, "trace_id": trace_id, "source_kind": "replay",
            "generated_gaps": [], "generation_config_sha256": None,
            "generation_lookahead_conditioned": False, "replay_original_reasoning_policy": "removed-like-baseline"}
        return self._render(deepcopy(messages), [], provenance, {}, reinsert_generated=False)

    def _render(self, messages, refs, provenance, counts, *, reinsert_generated):
        clean, _, normalization, edits = self._baseline._normalize(deepcopy(messages), None)
        # The baseline normalizer strips original reasoning. Reinsert exclusively
        # the exact receipt-bound generations; replay reasoning stays absent.
        if reinsert_generated:
            for index, message in enumerate(messages):
                if "reasoning_content" in message:
                    clean[index]["reasoning_content"] = self._baseline._escape(
                        message["reasoning_content"], f"/messages/{index}/reasoning_content", edits)
        text, spans = _render_with_assistant_indices(self._instrumented, clean, None, None, False, **TEMPLATE_OPTIONS)
        plain = self._plain.render(messages=clean, tools=None, documents=None,
                                   add_generation_prompt=False, **TEMPLATE_OPTIONS)
        reasoning_text, think_spans = _render_with_assistant_indices(
            self._reasoning_instrumented, clean, None, None, False, **TEMPLATE_OPTIONS)
        if text != plain or reasoning_text != plain:
            raise UnsupportedMessage("Loss instrumentation changed native xhigh rendered bytes")
        encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                                 padding=False, truncation=False)
        ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
        mask = align_loss_mask(offsets, spans)
        assistants = [m for m in clean if m["role"] == "assistant"]
        if len(spans) != len(assistants) or len(think_spans) != len(assistants):
            raise UnsafeTokenBoundary("Native assistant/thinking span coverage mismatch")
        filled_spans = [span for span, message in zip(think_spans, assistants) if "reasoning_content" in message]
        if len(filled_spans) != len(refs):
            raise UnsafeTokenBoundary("Generated reasoning was omitted or duplicated during mapping")
        reasoning_mask = align_loss_mask(offsets, filled_spans)
        if any(a and b for a, b in zip(mask, reasoning_mask)):
            raise UnsafeTokenBoundary("Generated reasoning overlaps a supervised token")
        retained_ranges, cursor = [], 0
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
        result = {"input_ids": ids, "labels": [token if selected else -100 for token, selected in zip(ids, mask)],
            "loss_mask": mask, "attention_mask": [1] * len(ids), "rendered_text": text,
            "assistant_spans": [{"start": start, "end": end} for start, end in spans], "token_offsets": offsets,
            "normalization": {**normalization, **counts}, "control_escape_edits": edits, "identity": self.identity}
        result = prefix_token_cutoff(result, max_tokens=self.max_tokens)
        result["cot_provenance"] = {**provenance, "filled_gaps": len(refs),
            "generation_receipts_verified": True, "semantic_review_required": False,
            "reviewed": False, "unreviewed_gap_count": len(refs),
            "semantic_quality_qualified": False, "future_information_checked": False,
            "semantic_certainty_claimed": False, "reasoning_loss": 0, "reasoning_effort": "xhigh",
            "acceptance_policy": ACCEPTANCE_POLICY}
        result["cot_mask_audit"] = {"policy": "filled_native_think_wrapper_and_body_loss0/v1",
            "retained_token_ranges": retained_ranges, "filled_gap_count": len(refs),
            "generated_receipt_count": len(refs), "retained_filled_gap_count": len(retained_ranges)}
        return result


def training_row(result, *, group_id, source="trace", provenance=None):
    if (result.get("identity", {}).get("mask_policy") != MASK_POLICY
            or result.get("cot_provenance", {}).get("acceptance_policy") != ACCEPTANCE_POLICY
            or result["cot_provenance"].get("source_kind") != source):
        raise ValueError("Expected the explicit generated-CoT experimental renderer result")
    row = baseline_training_row(result, group_id=group_id, source=source,
        provenance={**(provenance or {}), "generated_cot": result["cot_provenance"]})
    row["metadata"]["cot_mask_audit"] = deepcopy(result["cot_mask_audit"])
    return row
