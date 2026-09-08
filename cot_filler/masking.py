"""Token aligned loss masks for an explicitly rendered training example.

Character spans from the review export are intentionally not training masks.  This
module aligns those spans against the *exact* string rendered by the target model's
chat-template and refuses ambiguous or unavailable tokenizer metadata.
"""
from __future__ import annotations


def align_char_segments(rendered_text, char_segments, tokenizer):
    """Return token ids and 0/1 loss mask from character spans.

    ``tokenizer`` must expose ``encode(text)`` returning a tokenizers-style
    Encoding with ``ids`` and ``offsets``.  Offsets are half-open Python string
    offsets. Special tokens with ``(0, 0)`` are masked.
    """
    if not isinstance(rendered_text, str) or not rendered_text:
        raise ValueError("Rendered text must be a non-empty string")
    if not isinstance(char_segments, list) or not char_segments:
        raise ValueError("At least one character loss segment is required")
    previous = 0
    spans = []
    for segment in char_segments:
        start, end, intent = segment.get("start_char"), segment.get("end_char"), segment.get("loss_intent")
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(rendered_text):
            raise ValueError("Character segments must be non-empty and within rendered text")
        if start < previous or intent not in {"train", "mask"}:
            raise ValueError("Character segments must be ordered, non-overlapping, and use train/mask")
        spans.append((start, end, intent))
        previous = end
    try:
        encoding = tokenizer.encode(rendered_text)
    except (AttributeError, TypeError) as exc:
        raise ValueError("Tokenizer must provide ids and character offsets") from exc
    ids, offsets = getattr(encoding, "ids", None), getattr(encoding, "offsets", None)
    if ids is None or offsets is None or len(ids) != len(offsets):
        raise ValueError("Tokenizer must provide ids and character offsets")
    mask = []
    for start, end in offsets:
        if start == end:
            mask.append(0)
            continue
        overlaps = [intent for lo, hi, intent in spans if start < hi and end > lo]
        if len(set(overlaps)) > 1:
            raise ValueError("A token overlaps character segments with conflicting loss intent")
        mask.append(1 if overlaps and overlaps[0] == "train" else 0)
    if not any(mask):
        raise ValueError("Character segments produced no trainable tokens")
    return {"input_ids": list(ids), "loss_mask": mask,
            "token_count": len(ids), "mask_token_count": len(ids) - sum(mask),
            "train_token_count": sum(mask)}


def attach_token_masks(example, rendered_text, tokenizer, *, template_id, tokenizer_id):
    """Attach an attested token mask to an export example."""
    if not template_id or not tokenizer_id:
        raise ValueError("template_id and tokenizer_id are required attestations")
    # Per-message offsets from `as_messages` are not offsets in the final
    # template string. An adapter must provide converted offsets explicitly.
    segments = example.get("metadata", {}).get("rendered_char_segments")
    if not segments:
        raise ValueError("Target adapter must provide rendered_char_segments; message offsets are not template offsets")
    aligned = align_char_segments(rendered_text, segments, tokenizer)
    result = dict(example)
    metadata = dict(example.get("metadata", {}))
    metadata.update({"training_ready": True, "template_id": template_id,
                     "tokenizer_id": tokenizer_id, "mask_alignment": "char-overlap-v1"})
    result["metadata"] = metadata
    result["token_mask"] = aligned
    return result
