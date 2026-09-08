"""Validated import of finished, externally authored rationale artifacts."""
from __future__ import annotations

import copy
import re

from .core import BLOCKING_FLAGS, digest, validate_candidate

EXTERNAL_SCHEMA = "cot.external-candidate.v1"
FIELDS = {"schema", "gap_id", "source_digest", "prefix_hash", "lookahead_hash", "prompt_hash", "expected_revision", "text", "generator"}
GENERATOR_FIELDS = {"provider", "model", "authoring_task", "prompt_version", "token_usage"}


def validate_external(record, gap):
    if not isinstance(record, dict) or set(record) != FIELDS or record.get("schema") != EXTERNAL_SCHEMA:
        raise ValueError("External candidate must use the exact cot.external-candidate.v1 fields; review/approval fields are prohibited")
    if type(record["expected_revision"]) is not int or record["expected_revision"] < 0:
        raise ValueError("expected_revision must be a nonnegative integer (0 for a new gap)")
    for key in ("source_digest", "prefix_hash", "lookahead_hash", "prompt_hash"):
        if record[key] != gap[key]:
            raise ValueError("External candidate " + key + " does not match the immutable gap")
    if record["gap_id"] != gap["id"]:
        raise ValueError("External candidate gap_id does not match")
    generator = record["generator"]
    if not isinstance(generator, dict) or set(generator) != GENERATOR_FIELDS:
        raise ValueError("External generator provenance must contain provider, model, authoring_task, prompt_version, and token_usage")
    if generator["provider"] != "codex-session" or generator["model"] != "not_attested" or generator["token_usage"] is not None:
        raise ValueError("This pilot requires honest codex-session/not_attested provenance and unavailable token usage (null)")
    task = generator["authoring_task"]
    if not isinstance(task, str) or len(task) > 512 or re.fullmatch(r"/root(?:/[a-z0-9_]+)*", task) is None:
        raise ValueError("authoring_task must identify the actual canonical collaboration task")
    if generator["prompt_version"] != gap["prompt_version"]:
        raise ValueError("External generator prompt_version does not match the gap")
    flags = validate_candidate(record["text"])
    if BLOCKING_FLAGS.intersection(flags):
        raise ValueError("External candidate is empty or malformed; supply a finished plain-text rationale")
    return {
        "generator": copy.deepcopy(generator),
        "prompt_hash": gap["prompt_hash"], "source_digest": gap["source_digest"],
        "prefix_hash": gap["prefix_hash"], "lookahead_hash": gap["lookahead_hash"],
        "external_record_sha256": digest(record),
        "completion_status": "assistant_authored_finished_artifact",
        "synthetic": True, "lookahead_conditioned": True,
        "flags": flags + ["externally_authored", "codex_session_pilot", "semantic_review_required", "exact_backend_model_unavailable", "token_usage_unavailable"],
    }
