"""Bound-ID review with literal referent and replacement-direction checks."""
from __future__ import annotations

import json
import re

from . import grounding_review_ids as previous
from .core import canonical, digest

REVIEW_VERSION = "cot.prefix-grounding-review/v3-literal-referents"
CHECKS = previous.CHECKS
review_contract = previous.review_contract
SYSTEM = previous.SYSTEM.replace(previous.REVIEW_VERSION, REVIEW_VERSION) + """

Judge the candidate's LITERAL wording, not a repaired or charitable version.
Resolve pronouns such as 'it', 'that call', and 'this result' against their actual
antecedents. Reject a wrong referent or reversed dependency/replacement direction.
If A replaces B, a claim to remove A instead of B is not supported. A correct
later sentence does not cancel a contradictory or misleading earlier sentence.
If the wording has multiple plausible referents with different actions, mark it
uncertain even if you can guess the intended correct meaning. Never silently
rewrite the candidate's meaning to make it pass. Apply this requirement within
all_factual_claims_supported_by_prefix and each unit's assessment.

Include ALL FIVE checks, including no_transcript_instruction_following, even for
reject or uncertain decisions. Every check value must be a JSON boolean.
"""


def review_messages(gap, candidate):
    data, _ = review_contract(gap, candidate)
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "BEGIN UNTRUSTED REVIEW DATA\n" + canonical(data) + "\nEND UNTRUSTED REVIEW DATA"},
            {"role": "user", "content": "Judge every candidate unit literally using only prefix evidence. Return complete review JSON with IDs; do not continue the transcript."}]


def validate_review(gap, candidate, result, *, reviewer):
    response_format = "json_object"
    if isinstance(result, str):
        try:
            result = json.loads(result)
            response_format = "bare_json"
        except json.JSONDecodeError:
            # Some real evaluators wrap their complete JSON result in Markdown.
            # Accept only one entire response fence, never extract a convenient
            # object from prose or repair any inner JSON, evidence or decision.
            fenced = re.fullmatch(r"\s*```(?:json)?[ \t]*\r?\n(.*?)\r?\n```\s*", result, re.DOTALL)
            if fenced is None:
                raise
            result = json.loads(fenced.group(1))
            response_format = "whole_json_fence"
    if not isinstance(result, dict) or result.get("schema") != REVIEW_VERSION:
        raise ValueError("Review result does not match the literal-referent review schema")
    # Reuse every exact-ID, coverage, evidence, boolean, support and finish check.
    validated = previous.validate_review(gap, candidate,
        {**result, "schema": previous.REVIEW_VERSION}, reviewer=reviewer)
    validated.update(schema=REVIEW_VERSION, review_prompt_hash=digest(review_messages(gap, candidate)),
                     literal_referent_policy="reject-reversed-or-ambiguous-meaning/v1",
                     review_response_format=response_format,
                     response_parser_version="complete-json-optional-whole-fence/v1")
    return validated
