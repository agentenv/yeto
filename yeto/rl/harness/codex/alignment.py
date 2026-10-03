"""Mask / token / logprob alignment contract for agentic samples (design D5 assertions).

A sample produced by the session server (``compute_samples_from_openai_records``
+ ``merge_samples``) must satisfy, fail closed:

- ``len(tokens) == len(loss_mask)`` (loss_mask covers the whole sequence), or
  ``len(loss_mask) == response_length`` (per-turn layout before merge);
- ``rollout_log_probs`` has one entry per response token;
- every mask=1 position lies in the response region and has a finite logprob;
- mask=1 positions are exactly the generated spans when ``weight_versions``
  spans are present (they delimit SGLang output); tool/template tokens are mask 0.

Violations raise ``AlignmentError``; callers abort the trajectory and never
convert the failure into a 0 reward.
"""

from __future__ import annotations

import math
from typing import Any


class AlignmentError(ValueError):
    pass


def _response_offset(sample: Any) -> tuple[int, int]:
    tokens = list(getattr(sample, "tokens", None) or [])
    response_length = getattr(sample, "response_length", None)
    if not isinstance(response_length, int) or response_length < 0 or response_length > len(tokens):
        raise AlignmentError("response_length is missing or out of range")
    return len(tokens) - response_length, len(tokens)


def generated_spans(sample: Any) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for call in getattr(sample, "weight_versions", None) or []:
        for span in getattr(call, "spans", None) or (call.get("spans", []) if isinstance(call, dict) else []):
            start = getattr(span, "abs_start", None) if not isinstance(span, dict) else span.get("abs_start")
            end = getattr(span, "abs_end", None) if not isinstance(span, dict) else span.get("abs_end")
            if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
                raise AlignmentError("weight version span is malformed")
            spans.append((start, end))
    return spans


def assert_sample_alignment(sample: Any) -> None:
    tokens = list(getattr(sample, "tokens", None) or [])
    loss_mask = getattr(sample, "loss_mask", None)
    logprobs = getattr(sample, "rollout_log_probs", None)
    if loss_mask is None:
        raise AlignmentError("loss_mask is missing")
    if logprobs is None:
        raise AlignmentError("rollout_log_probs is missing")
    start, end = _response_offset(sample)
    if len(loss_mask) == len(tokens):
        mask_offset = 0
    elif len(loss_mask) == end - start:
        mask_offset = start
    else:
        raise AlignmentError(f"len(loss_mask)={len(loss_mask)} matches neither len(tokens)={len(tokens)} nor response_length={end - start}")
    if len(logprobs) != end - start:
        raise AlignmentError(f"len(rollout_log_probs)={len(logprobs)} != response_length={end - start}")
    ones = {mask_offset + i for i, m in enumerate(loss_mask) if int(m) == 1}
    for position in sorted(ones):
        if position < start:
            raise AlignmentError(f"loss_mask=1 at prompt position {position}")
        value = logprobs[position - start]
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise AlignmentError(f"loss_mask=1 at {position} without a finite rollout logprob")
    spans = generated_spans(sample)
    if spans:
        generated = {p for s, e in spans for p in range(s, e)}
        if ones - generated:
            raise AlignmentError(f"loss_mask=1 outside generated spans at {sorted(ones - generated)[:5]}")
