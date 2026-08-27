from __future__ import annotations

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

pytest.importorskip(
    "yeto_miles_secrlenv",
    reason="the Miles SecRLEnv adapter is an optional external integration",
)

from yeto_miles_secrlenv import agent


class FakeMessage:
    def __init__(self, value):
        self.value = value

    def model_dump(self, *, exclude_none=True):
        return copy.deepcopy(self.value)


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        response = self.responses.pop(0)
        usage = None
        if isinstance(response, tuple):
            response, total_tokens = response
            usage = SimpleNamespace(total_tokens=total_tokens)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=FakeMessage(response))],
            usage=usage,
        )


class FakePolicy:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


class FakeEpisodeClient:
    def __init__(self):
        self.exec_calls = []
        self.submissions = []

    async def execute(
        self, episode_id, command, *, timeout_seconds, output_bytes
    ):
        self.exec_calls.append(
            (episode_id, command, timeout_seconds, output_bytes)
        )
        return {
            "exit_code": 0,
            "output": "command output",
            "timed_out": False,
            "truncated": False,
        }

    async def submit(self, episode_id, submission):
        self.submissions.append((episode_id, submission))
        return {"accepted": True}


def _tool_call(identifier, name, arguments):
    return {
        "id": identifier,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _assistant_tool(identifier, name, arguments):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [_tool_call(identifier, name, arguments)],
    }


def test_forwarded_max_seq_len_metadata_is_strict():
    assert agent._metadata_max_seq_len({}) is None
    assert agent._metadata_max_seq_len({"max_seq_len": 8192}) == 8192
    for invalid in (True, 0, -1, 8192.0, "8192"):
        with pytest.raises(ValueError, match="positive integer"):
            agent._metadata_max_seq_len({"max_seq_len": invalid})


def test_agent_caps_later_turns_from_model_usage(monkeypatch):
    monkeypatch.setenv("SECRLENV_MAX_TURNS", "10")
    policy = FakePolicy(
        [
            (_assistant_tool("call-1", "terminal.exec", {"command": "id"}), 7000),
            (
                _assistant_tool(
                    "call-2", "terminal.exec", {"command": "uname -a"}
                ),
                8500,
            ),
            (
                _assistant_tool(
                    "call-3", "submit", {"evidence": "must not execute"}
                ),
                9000,
            ),
        ]
    )
    client = FakeEpisodeClient()
    metrics = agent.AgentMetrics()

    status = asyncio.run(
        agent._drive_policy(
            policy,
            client,
            {"episode_id": "e" * 24, "prompt": "solve"},
            {"max_tokens": 4096},
            metrics,
            max_seq_len=8192,
        )
    )

    assert status == "max_seq_len"
    assert len(policy.chat.completions.requests) == 2
    assert policy.chat.completions.requests[0]["extra_body"]["max_tokens"] == 4096
    assert policy.chat.completions.requests[1]["extra_body"]["max_tokens"] == 1192
    assert [call[1] for call in client.exec_calls] == ["id", "uname -a"]
    assert client.submissions == []
    assert metrics.max_model_total_tokens == 8500
    assert metrics.max_seq_len_hit == 1
    assert metrics.usage_missing == 0


def test_agent_fails_closed_when_usage_is_missing_under_cap(monkeypatch):
    monkeypatch.setenv("SECRLENV_MAX_TURNS", "10")
    policy = FakePolicy(
        [_assistant_tool("call-1", "terminal.exec", {"command": "id"})]
    )
    metrics = agent.AgentMetrics()

    status = asyncio.run(
        agent._drive_policy(
            policy,
            FakeEpisodeClient(),
            {"episode_id": "f" * 24, "prompt": "solve"},
            {"max_tokens": 4096},
            metrics,
            max_seq_len=8192,
        )
    )

    assert status == "max_seq_len"
    assert metrics.max_seq_len_hit == 1
    assert metrics.usage_missing == 1


def test_submit_on_limit_crossing_still_wins(monkeypatch):
    monkeypatch.setenv("SECRLENV_MAX_TURNS", "10")
    policy = FakePolicy(
        [
            (
                _assistant_tool(
                    "call-1", "submit", {"evidence": "verified result"}
                ),
                9000,
            )
        ]
    )
    client = FakeEpisodeClient()
    metrics = agent.AgentMetrics()

    status = asyncio.run(
        agent._drive_policy(
            policy,
            client,
            {"episode_id": "g" * 24, "prompt": "solve"},
            {"max_tokens": 4096},
            metrics,
            max_seq_len=8192,
        )
    )

    assert status == "completed"
    assert client.submissions == [
        ("g" * 24, {"evidence": "verified result"})
    ]
    assert metrics.max_model_total_tokens == 9000
