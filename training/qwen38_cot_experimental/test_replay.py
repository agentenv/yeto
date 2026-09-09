import hashlib
import json
from pathlib import Path

import pytest

from . import replay
from .render import Qwen38ExperimentalCotRenderer
from .data import validate_row

ASSETS = Path(__file__).resolve().parents[1] / "qwen38_no_cot/assets"


def rollout_job(tmp_path):
    events = [
        {"type": "session_meta", "payload": {"id": "same-original-session", "base_instructions": "System setup."}},
        {"type": "response_item", "payload": {"type": "message", "role": "user", "content": "Inspect the file. SECRET_VISIBLE_KEEP"}},
        {"type": "response_item", "payload": {"type": "reasoning", "text": "PRIVATE_OMIT"}},
        {"type": "response_item", "payload": {"type": "function_call", "name": "read", "call_id": "c1", "arguments": '{"path":"a.py"}'}},
        {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "c1", "output": "OBSERVATION_MASKED"}},
        {"type": "response_item", "payload": {"type": "message", "role": "assistant", "content": "The file is present."}},
    ]
    path = tmp_path / "rollout-test.jsonl"
    raw = b"".join(json.dumps(row).encode() + b"\n" for row in events)
    path.write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    return {"identity": "replay:" + sha, "path": str(path), "sha256": sha, "source": "replay", "format": "rollout"}


def test_original_replay_native_xhigh_masks_and_session_preserved(tmp_path):
    renderer = Qwen38ExperimentalCotRenderer(ASSETS)
    job = rollout_job(tmp_path)
    result = replay.convert_job(job, renderer=renderer)
    assert result["ok"]
    row = result["row"]
    validate_row(row, renderer.identity)
    text = renderer.tokenizer.decode(row["input_ids"])
    targets = renderer.tokenizer.decode([i for i, label in zip(row["input_ids"], row["labels"]) if label != -100])
    assert row["group_id"] == "session:same-original-session"
    assert "PRIVATE_OMIT" not in text and "SECRET_VISIBLE_KEEP" in text
    assert "OBSERVATION_MASKED" in text and "OBSERVATION_MASKED" not in targets
    assert "<think>" in text and "<think>" not in targets
    assert "The file is present.<|im_end|>" in targets
    assert "<tool_call>" in targets and "<parameter=path>\na.py" in targets
    provenance = row["metadata"]["provenance"]
    assert provenance["generated_cot"]["filled_gaps"] == 0
    assert provenance["baseline_token_arrays_reused"] is False


def test_source_hash_checked_before_parsing(tmp_path):
    job = rollout_job(tmp_path)
    Path(job["path"]).write_text("malformed and changed")
    with pytest.raises(ValueError, match="source_file_hash_mismatch"):
        replay.normalize_job(job)


def test_frozen_job_inventory_deduplicates_same_original_sha(tmp_path):
    job = rollout_job(tmp_path)
    manifest = tmp_path / "jobs.jsonl"
    manifest.write_text(json.dumps(job) + "\n" + json.dumps(job) + "\n")
    jobs, identities = replay.load_jobs([manifest])
    assert len(jobs) == 1 and len(identities) == 1
    assert jobs[0]["source_jobs_sha256"] == identities[0]["sha256"]


def test_atif_private_reasoning_removed_without_losing_observation(tmp_path):
    doc = {"session_id": "atif-session", "steps": [
        {"source": "user", "message": "Inspect."},
        {"source": "agent", "message": "Reading.", "extra": {"thinking": "PRIVATE_OMIT"},
         "tool_calls": [{"id": "x", "function_name": "read", "arguments": {"path": "a"}}],
         "observation": {"results": [{"source_call_id": "x", "content": "original result"}]}},
        {"source": "agent", "message": "Done."}]}
    path = tmp_path / "trajectory.json";path.write_text(json.dumps(doc))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    messages, counts, group = replay.normalize_job({"source": "replay", "format": "atif", "path": str(path), "sha256": sha})
    assert group == "session:atif-session" and counts["atif_thinking_omitted"] == 1
    assert messages[2]["role"] == "tool" and messages[2]["content"] == "original result"
    assert "PRIVATE_OMIT" not in json.dumps(messages)
