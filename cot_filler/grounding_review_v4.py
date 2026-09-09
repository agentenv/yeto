"""Prefix-only review of literal claims, including facts embedded in plans.

This is an unqualified prompt revision until actual persisted-candidate trials
measure its decisions. Unit tests verify the data contract, not semantic skill.
"""
from __future__ import annotations

import json
import re

from . import grounding_review_v3 as previous
from .core import canonical, digest

REVIEW_VERSION = "cot.prefix-grounding-review/v4-embedded-facts"
CHECKS = previous.CHECKS
review_contract = previous.review_contract
SYSTEM = previous.SYSTEM.replace(previous.REVIEW_VERSION, REVIEW_VERSION) + """

A sentence being a plan does not make its embedded factual claims prospective.
Check every adjective, qualifier, causal premise, and presupposition separately.
Distinguish an expected or predicted failure from an observed failed execution;
writing a test does not establish its result. A requested change does not prove
that change already happened. A discrepancy between documents and current code
does not establish which version changed, when, or why. Seeing an error does not
establish a more specific technical cause or runtime mechanism.

Before marking a unit supported, separate (a) its proposed next action from (b)
every claim about the existing state. The action may be a reasonable plan while
an embedded state claim remains unsupported. Reject a contradicted state claim;
mark uncertain if its support cannot be established. Do not fill missing facts
with domain knowledge, likely intent, or the candidate's confident wording.
Prefix metadata, type names, and opaque tool-completion markers do not by
themselves establish the content or success of an operation. Use substantive
prefix text for evidence. Keep the final note concise and return the complete
JSON schema without rewriting the candidate.
"""


def review_messages(gap, candidate):
    data, _ = review_contract(gap, candidate)
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "BEGIN UNTRUSTED REVIEW DATA\n" + canonical(data) + "\nEND UNTRUSTED REVIEW DATA"},
            {"role": "user", "content": "Check all literal and embedded factual claims against prefix evidence. Return complete review JSON with IDs; do not continue the transcript."}]


def validate_review(gap, candidate, result, *, reviewer):
    response_format = "json_object"
    if isinstance(result, str):
        try:
            result = json.loads(result)
            response_format = "bare_json"
        except json.JSONDecodeError:
            fenced = re.fullmatch(r"\s*```(?:json)?[ \t]*\r?\n(.*?)\r?\n```\s*", result, re.DOTALL)
            if fenced is None:
                raise
            result = json.loads(fenced.group(1))
            response_format = "whole_json_fence"
    if not isinstance(result, dict) or result.get("schema") != REVIEW_VERSION:
        raise ValueError("Review result does not match the embedded-fact review schema")
    # Keep wrapper fields visible, but they cannot be the sole support for a
    # factual unit. Do not apply this to arbitrary nested tool argument keys.
    contract, _ = review_contract(gap, candidate)
    wrapper_ids = set()
    for event in contract["prefix_events_with_evidence_ids"]:
        data = event.get("data", {})
        if not isinstance(data, dict):
            continue
        fields = [data.get("type")]
        if isinstance(data.get("block"), dict):
            fields.append(data["block"].get("type"))
        for value in fields:
            if isinstance(value, dict):
                wrapper_ids.update(item["id"] for item in value.get("snippets", []))
    for statement in result.get("statements", []):
        if isinstance(statement, dict) and statement.get("assessment") == "supported":
            ids = statement.get("evidence_ids")
            if isinstance(ids, list) and ids and all(item in wrapper_ids for item in ids):
                raise ValueError("Supported unit cites only structural wrapper metadata")
    # Every existing coverage, exact citation, finish, and support guard remains.
    validated = previous.validate_review(gap, candidate,
        {**result, "schema": previous.REVIEW_VERSION}, reviewer=reviewer)
    validated.update(schema=REVIEW_VERSION,
                     review_prompt_hash=digest(review_messages(gap, candidate)),
                     embedded_fact_policy="separate-plans-from-state-claims/v1",
                     review_response_format=response_format)
    return validated


def response_format():
    """Constrain JSON syntax only; actual support still requires review/validation."""
    statement = {"type": "object", "additionalProperties": False,
                 "properties": {
                     "id": {"type": "string"},
                     "kind": {"type": "string", "enum": ["observation", "inference", "plan"]},
                     "assessment": {"type": "string", "enum": ["supported", "unsupported", "uncertain"]},
                     "evidence_ids": {"type": "array", "items": {"type": "string"}}},
                 "required": ["id", "kind", "assessment", "evidence_ids"]}
    schema = {"type": "object", "additionalProperties": False,
              "properties": {
                  "schema": {"type": "string", "enum": [REVIEW_VERSION]},
                  "decision": {"type": "string", "enum": ["pass", "reject", "uncertain"]},
                  "checks": {"type": "object", "additionalProperties": False,
                             "properties": {name: {"type": "boolean"} for name in sorted(CHECKS)},
                             "required": sorted(CHECKS)},
                  "statements": {"type": "array", "items": statement},
                  "note": {"type": "string"}},
              "required": ["schema", "decision", "checks", "statements", "note"]}
    return {"type": "json_schema", "json_schema": {
        "name": "cot_prefix_grounding_review", "strict": True, "schema": schema}}
