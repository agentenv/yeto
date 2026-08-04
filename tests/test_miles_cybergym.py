"""Tests for the Miles-compatible CyberGym reward and prompt adapter."""

import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest

from yeto.rl.learner import prepare_prompt_data
from yeto_miles_cybergym import reward
from yeto_miles_cybergym.prompts import prompt_rows


class Response:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self.payload = payload if payload is not None else {"exit_code": 1}
        self.text = text

    def json(self):
        return self.payload


def test_score_uses_cybergym_multipart_checksum_and_maps_reward(monkeypatch):
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return Response(payload={"exit_code": 0})

    monkeypatch.setattr(reward.requests, "post", post)
    sample = SimpleNamespace(
        response="poc bytes", metadata={"task_id": "arvo:47101"}
    )

    assert asyncio.run(reward.score(None, sample)) == -1.0
    url, kwargs = calls[0]
    assert url == "http://127.0.0.1:8666/submit-vul"
    submitted = json.loads(kwargs["files"]["metadata"][1])
    expected = hashlib.sha256(b"arvo:47101yeto_agentCyberGym").hexdigest()
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
        lambda *args, **kwargs: Response(
            payload={"exit_code": next(exit_codes)}
        ),
    )
    samples = [
        SimpleNamespace(response=b"one", metadata={"task_id": "arvo:1"}),
        SimpleNamespace(response=b"two", metadata={"task_id": "arvo:2"}),
    ]
    assert asyncio.run(reward.score(None, samples)) == [1.0, -1.0]

    monkeypatch.setattr(
        reward.requests,
        "post",
        lambda *args, **kwargs: Response(
            status_code=500, text='{"detail":"No such image"}'
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


def test_prompt_metadata_reaches_miles_without_an_extra_nesting_level(
    tmp_path, monkeypatch
):
    from yeto import data

    monkeypatch.setattr(
        data,
        "load_rows",
        lambda source, revision=None: prompt_rows(["arvo:47101"]),
    )

    output = prepare_prompt_data("unused", None, tmp_path / "prompts.jsonl")
    row = json.loads(output.read_text())

    assert row["metadata"] == {"task_id": "arvo:47101"}
