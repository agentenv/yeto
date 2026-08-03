"""Tests for the standalone Miles/CyberGym comparison adapter."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from yeto_miles_cybergym import reward
from yeto_miles_cybergym.launcher import build_train_command
from yeto_miles_cybergym.prompts import prompt_rows


class _Response:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {"exit_code": 1}
        self.text = text

    def json(self):
        return self._payload


def test_score_uses_cybergym_multipart_checksum_and_maps_reward(monkeypatch):
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return _Response(payload={"exit_code": 0})

    monkeypatch.setattr(reward.requests, "post", post)
    sample = SimpleNamespace(
        response="poc bytes",
        metadata={"task_id": "arvo:47101"},
    )

    assert asyncio.run(reward.score(None, sample)) == -1.0
    url, kwargs = calls[0]
    assert url == "http://127.0.0.1:8666/submit-vul"
    submitted = json.loads(kwargs["files"]["metadata"][1])
    expected = hashlib.sha256(
        b"arvo:47101yeto_agentCyberGym"
    ).hexdigest()
    assert submitted == {
        "agent_id": "yeto_agent",
        "task_id": "arvo:47101",
        "checksum": expected,
        "require_flag": False,
    }
    assert kwargs["files"]["file"][1] == b"poc bytes"


def test_score_supports_miles_batch_and_does_not_hide_http_errors(monkeypatch):
    exit_codes = iter([1, 300])

    monkeypatch.setattr(
        reward.requests,
        "post",
        lambda *args, **kwargs: _Response(payload={"exit_code": next(exit_codes)}),
    )
    samples = [
        SimpleNamespace(response=b"one", metadata={"task_id": "arvo:1"}),
        SimpleNamespace(response=b"two", metadata={"task_id": "arvo:2"}),
    ]
    assert asyncio.run(reward.score(None, samples)) == [1.0, -1.0]

    monkeypatch.setattr(
        reward.requests,
        "post",
        lambda *args, **kwargs: _Response(
            status_code=500,
            text='{"detail":"No such image"}',
        ),
    )
    with pytest.raises(RuntimeError, match="HTTP 500.*No such image"):
        asyncio.run(reward.score(None, samples[0]))


def test_prompt_rows_are_chat_template_ready_and_keep_task_metadata():
    rows = prompt_rows(["arvo:47101"], repeats=2)
    assert len(rows) == 2
    assert rows[0]["metadata"] == {"task_id": "arvo:47101"}
    assert rows[0]["messages"][0]["role"] == "user"
    assert "arvo:47101" in rows[0]["messages"][0]["content"]


def test_direct_launcher_builds_valid_grpo_shape():
    args = SimpleNamespace(
        miles_root=Path("/workspace/miles"),
        samples_per_iteration=4,
        samples_per_prompt=2,
        model="Qwen/Qwen2.5-0.5B-Instruct",
        trainer_gpus=1,
        rollout_gpus=1,
        prompt_data="/workspace/yeto/cybergym_prompts.jsonl",
        iterations=3,
        max_context_len=1024,
        max_response_len=128,
        temperature=0.7,
        lr=1e-5,
    )
    command = build_train_command(args)
    assert command[1:3] == ["/workspace/miles/train.py", "--train-backend"]
    assert "--custom-rm-path" in command
    assert command[command.index("--rollout-batch-size") + 1] == "2"
    assert command[command.index("--n-samples-per-prompt") + 1] == "2"
    assert "--sglang-rl-on-policy-target" not in command
