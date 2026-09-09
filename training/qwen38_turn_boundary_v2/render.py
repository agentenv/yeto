"""Native no-CoT renderer requiring a hash-bound v2 source-turn audit."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from training.qwen38_no_cot.render import Qwen38Renderer as OriginalRenderer
from training.qwen38_no_cot.render import training_row as original_training_row
from . import normalize, source_adapters

CONTRACT = "qwen38-no-cot-source-turns/v2"


def message_digest(messages):
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def normalizer_identity():
    return {"source_adapter_version": source_adapters.VERSION,
            "source_adapter_sha256": hashlib.sha256(Path(source_adapters.__file__).read_bytes()).hexdigest(),
            "turn_normalizer_sha256": hashlib.sha256(Path(normalize.__file__).read_bytes()).hexdigest(),
            "turn_renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


class Qwen38Renderer(OriginalRenderer):
    def __init__(self, tokenizer_dir, *, max_tokens=262144):
        super().__init__(tokenizer_dir, max_tokens=max_tokens)
        self.identity.update({"turn_boundary_contract": CONTRACT, **normalizer_identity()})

    def render(self, messages, tools=None, *, turn_boundary_audit):
        if (not isinstance(turn_boundary_audit, dict)
                or turn_boundary_audit.get("output_messages_sha256") != message_digest(messages)):
            raise ValueError("The source-turn audit does not bind these normalized messages")
        if turn_boundary_audit.get("reasoning_policy") != "drop":
            raise ValueError("No-CoT rendering requires the no-reasoning normalization policy")
        result = super().render(messages, tools)
        result["turn_boundary_audit"] = deepcopy(turn_boundary_audit)
        return result


def training_row(result, *, group_id, source, provenance=None):
    if result.get("identity", {}).get("turn_boundary_contract") != CONTRACT or not result.get("turn_boundary_audit"):
        raise ValueError("A corrected training row requires its versioned source-turn rendering")
    row = original_training_row(result, group_id=group_id, source=source, provenance=provenance)
    row["metadata"]["turn_boundary_audit"] = deepcopy(result["turn_boundary_audit"])
    row["metadata"]["behavioral_recovery_validated"] = False
    return row
