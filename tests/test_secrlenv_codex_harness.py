from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import aiohttp
import pytest
from aiohttp import web

from yeto_miles_secrlenv import codex_harness_agent as harness


def _completion(
    name: str,
    raw_arguments: str,
    call_id: str,
    reasoning: str,
    total_tokens: int,
) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{call_id}",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": reasoning,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": raw_arguments,
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {
            "prompt_tokens": total_tokens - 5,
            "completion_tokens": 5,
            "total_tokens": total_tokens,
        },
    }


async def _fake_miles(
    completions: Iterable[dict[str, Any]],
) -> tuple[web.AppRunner, str, list[dict[str, Any]]]:
    queue = list(completions)
    requests: list[dict[str, Any]] = []

    async def chat(request: web.Request) -> web.Response:
        requests.append(await request.json())
        if not queue:
            return web.json_response({"error": "unexpected sample"}, status=409)
        return web.json_response(queue.pop(0))

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    sockets = site._server.sockets if site._server is not None else []
    assert len(sockets) == 1
    return runner, f"http://127.0.0.1:{sockets[0].getsockname()[1]}", requests


def _codex_body(input_items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model": harness.BACKEND_MODEL,
        "instructions": harness.BASE_INSTRUCTIONS,
        "input": input_items,
        "parallel_tool_calls": False,
        "reasoning": {"effort": "xhigh", "summary": "none"},
        "store": False,
        "stream": True,
        "tool_choice": "auto",
        "tools": [
            {
                "type": "function",
                "name": "update_plan",
                "description": "Update the task plan.",
                "strict": False,
                "parameters": {"type": "object"},
            },
            *copy.deepcopy(harness._EXPECTED_RESPONSE_TOOLS),
        ],
    }


def _sse_events(raw: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for block in raw.split("\n\n"):
        data = [line[6:] for line in block.splitlines() if line.startswith("data: ")]
        if data and data[0] != "[DONE]":
            events.append(json.loads("\n".join(data)))
    return events


def _set_bridge_env(monkeypatch: pytest.MonkeyPatch, max_tokens: int = 256) -> None:
    monkeypatch.setenv("YETO_CODEX_BACKEND_MAX_TOKENS", str(max_tokens))


def test_identity_is_live_exact_and_fail_closed(monkeypatch):
    identity = harness.codex_harness_identity()
    assert set(identity) == {
        "base_instructions_sha256",
        "terminal_exec_tool_schema_sha256",
        "submit_tool_schema_sha256",
        "dynamic_tools_schema_sha256",
    }
    assert all(len(value) == 64 and int(value, 16) >= 0 for value in identity.values())
    assert harness._MILES_TOOLS == harness.legacy.TOOLS
    assert "model-facing tools, `terminal.exec` and `submit`" in (
        harness.BASE_INSTRUCTIONS
    )
    assert "Codex's internal\n`terminal_exec` dynamic-tool alias" in (
        harness.BASE_INSTRUCTIONS
    )
    monkeypatch.setattr(harness, "BASE_INSTRUCTIONS", harness.BASE_INSTRUCTIONS + " drift")
    with pytest.raises(harness.CodexHarnessError, match="identity constants drifted"):
        harness.codex_harness_identity()


def test_runtime_attests_live_binary_version_and_all_env(tmp_path, monkeypatch):
    harness._RUNTIME_ATTESTATION_CACHE.clear()
    binary = tmp_path / "codex-fake"
    binary.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = \"--version\" ]; then\n"
        "  printf 'codex-cli 0.145.0\\n'\n"
        "  exit 0\n"
        "fi\n"
        "exit 1\n"
    )
    binary.chmod(0o700)
    for name, value in harness._IDENTITY_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("YETO_CODEX_BINARY_PATH", str(binary))
    monkeypatch.setenv(
        "YETO_CODEX_BINARY_SHA256", hashlib.sha256(binary.read_bytes()).hexdigest()
    )
    monkeypatch.setenv("YETO_CODEX_BINARY_SIZE_BYTES", str(binary.stat().st_size))
    monkeypatch.setenv("YETO_CODEX_VERSION", harness.CODEX_CLI_VERSION)
    monkeypatch.setenv("YETO_CODEX_BACKEND_MAX_TOKENS", "256")
    real_run = harness.subprocess.run
    version_calls = 0

    def counted_run(*args, **kwargs):
        nonlocal version_calls
        version_calls += 1
        return real_run(*args, **kwargs)

    monkeypatch.setattr(harness.subprocess, "run", counted_run)
    assert harness._attest_runtime() == binary
    assert harness._attest_runtime() == binary
    assert version_calls == 1

    binary.write_text(binary.read_text().replace("0.145.0", "0.145.1"))
    monkeypatch.setenv(
        "YETO_CODEX_BINARY_SHA256", hashlib.sha256(binary.read_bytes()).hexdigest()
    )
    monkeypatch.setenv("YETO_CODEX_BINARY_SIZE_BYTES", str(binary.stat().st_size))
    with pytest.raises(harness.CodexHarnessError, match="version drifted"):
        harness._attest_runtime()
    assert version_calls == 2
    harness._RUNTIME_ATTESTATION_CACHE.clear()


def test_history_and_responses_contract_reject_compaction_and_extra_tools():
    with pytest.raises(harness.CodexHarnessError, match="compaction"):
        harness._canonical_history_item({"type": "compaction"})
    body = _codex_body(
        [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "task"}],
            }
        ]
    )
    body["tools"].append(
        {
            "type": "function",
            "name": "shell",
            "strict": False,
            "parameters": {"type": "object"},
        }
    )
    with pytest.raises(harness.CodexHarnessError, match="tool surface drifted"):
        harness._validate_codex_request(body, harness.BACKEND_MODEL)
    body = _codex_body(body["input"])
    body["reasoning"]["effort"] = "max"
    with pytest.raises(harness.CodexHarnessError, match="xhigh"):
        harness._validate_codex_request(body, harness.BACKEND_MODEL)


def test_sampling_cannot_override_signed_miles_fields(monkeypatch):
    _set_bridge_env(monkeypatch)
    metrics = harness.legacy.AgentMetrics()
    with pytest.raises(harness.CodexHarnessError, match="override"):
        harness._ResponsesBridge(
            "http://127.0.0.1:1",
            "task",
            {"max_tokens": 64, "parallel_tool_calls": True},
            metrics,
            max_seq_len=None,
        )
    with pytest.raises(harness.CodexHarnessError, match="unknown sampling"):
        harness._ResponsesBridge(
            "http://127.0.0.1:1",
            "task",
            {"max_tokens": 64, "untrusted_body_field": "value"},
            metrics,
            max_seq_len=None,
        )


def test_fake_codex_bridge_preserves_history_and_filters_only_update_plan(monkeypatch):
    _set_bridge_env(monkeypatch)

    async def scenario() -> None:
        raw_terminal = '{ "command": "id" }'
        runner, miles_url, miles_requests = await _fake_miles(
            [
                _completion(
                    "terminal.exec", raw_terminal, "call-terminal", "inspect", 25
                ),
                _completion(
                    "submit",
                    '{"evidence":"uid=1000"}',
                    "call-submit",
                    "finish",
                    40,
                ),
            ]
        )
        metrics = harness.legacy.AgentMetrics()
        try:
            async with harness._ResponsesBridge(
                miles_url,
                "solve target",
                {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 128,
                    "stream": True,
                    "n": 1,
                },
                metrics,
                max_seq_len=512,
            ) as bridge:
                headers = {"Authorization": f"Bearer {bridge.token}"}
                initial = [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "solve target"}
                        ],
                    }
                ]
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{bridge.url}/v1/responses",
                        json=_codex_body(initial),
                        headers=headers,
                    ) as response:
                        assert response.status == 200
                        first_events = _sse_events(await response.text())
                    completed = first_events[-1]["response"]
                    function_call = completed["output"][1]
                    assert function_call["name"] == "terminal_exec"
                    assert function_call["arguments"] == raw_terminal
                    assert function_call["call_id"] == "call-terminal"

                    rendered = '{"exit_code":0,"output":"uid=1000","timed_out":false,"truncated":false}'
                    bridge.expect_tool_output("call-terminal", rendered)
                    second_input = copy.deepcopy(bridge._expected_input)
                    assert second_input is not None
                    async with session.post(
                        f"{bridge.url}/v1/responses",
                        json=_codex_body(second_input),
                        headers=headers,
                    ) as response:
                        assert response.status == 200
                        second_events = _sse_events(await response.text())
                    submit = second_events[-1]["response"]["output"][1]
                    assert submit["name"] == "submit"
                    assert submit["call_id"] == "call-submit"
                    bridge.mark_terminal()
        finally:
            await runner.cleanup()

        assert len(miles_requests) == 2
        assert [tool["function"]["name"] for tool in miles_requests[0]["tools"]] == [
            "terminal.exec",
            "submit",
        ]
        assert miles_requests[0]["max_tokens"] == 128
        assert miles_requests[0]["temperature"] == 0.7
        assert miles_requests[0]["top_p"] == 0.9
        assert miles_requests[0]["stream"] is False
        assert miles_requests[0]["parallel_tool_calls"] is False
        assert miles_requests[0]["reasoning_effort"] == "max"
        assert miles_requests[0]["thinking"] == {"type": "enabled"}
        assert miles_requests[0]["chat_template_kwargs"] == (
            harness.BACKEND_CHAT_TEMPLATE_KWARGS
        )
        assert miles_requests[1]["messages"][2] == {
            "role": "assistant",
            "content": "",
            "reasoning_content": "inspect",
            "tool_calls": [
                {
                    "id": "call-terminal",
                    "type": "function",
                    "function": {
                        "name": "terminal.exec",
                        "arguments": raw_terminal,
                    },
                }
            ],
        }
        assert miles_requests[1]["messages"][3] == {
            "role": "tool",
            "tool_call_id": "call-terminal",
            "content": rendered,
        }
        assert metrics.turns == 2
        assert metrics.tool_calls == 2

    asyncio.run(scenario())


def test_bridge_rejects_history_mutation_without_an_extra_sample(monkeypatch):
    _set_bridge_env(monkeypatch)

    async def scenario() -> None:
        runner, miles_url, miles_requests = await _fake_miles(
            [_completion("terminal.exec", '{"command":"id"}', "call-1", "think", 10)]
        )
        try:
            async with harness._ResponsesBridge(
                miles_url,
                "immutable task",
                {"max_tokens": 64},
                harness.legacy.AgentMetrics(),
                max_seq_len=None,
            ) as bridge:
                headers = {"Authorization": f"Bearer {bridge.token}"}
                initial = [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "immutable task"}
                        ],
                    }
                ]
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{bridge.url}/v1/responses",
                        json=_codex_body(initial),
                        headers=headers,
                    ) as response:
                        assert response.status == 200
                    bridge.expect_tool_output("call-1", "tool result")
                    mutated = copy.deepcopy(bridge._expected_input)
                    assert mutated is not None
                    mutated[0]["content"][0]["text"] = "changed task"
                    async with session.post(
                        f"{bridge.url}/v1/responses",
                        json=_codex_body(mutated),
                        headers=headers,
                    ) as response:
                        assert response.status == 400
                assert isinstance(bridge.fatal.result(), harness.CodexHarnessError)
        finally:
            await runner.cleanup()
        assert len(miles_requests) == 1

    asyncio.run(scenario())


class _FakeEpisodeClient:
    def __init__(self, *, output: str = "uid=1000", truncated: bool = False) -> None:
        self.output = output
        self.truncated = truncated
        self.exec_calls: list[tuple[str, str]] = []
        self.submissions: list[tuple[str, dict[str, Any]]] = []

    async def execute(self, episode_id: str, command: str, **_kwargs: Any):
        self.exec_calls.append((episode_id, command))
        return {
            "exit_code": 0,
            "output": self.output,
            "timed_out": False,
            "truncated": self.truncated,
        }

    async def submit(self, episode_id: str, value: dict[str, Any]):
        self.submissions.append((episode_id, value))
        return {"accepted": True}


class _RecoveringEpisodeClient(_FakeEpisodeClient):
    def __init__(self) -> None:
        super().__init__()
        self.execute_attempts = 0
        self.submit_attempts = 0

    async def execute(self, episode_id: str, command: str, **kwargs: Any):
        self.execute_attempts += 1
        if self.execute_attempts == 1:
            self.exec_calls.append((episode_id, command))
            raise harness.EpisodeAPIError(400, "invalid_command", "command rejected")
        return await super().execute(episode_id, command, **kwargs)

    async def submit(self, episode_id: str, value: dict[str, Any]):
        self.submit_attempts += 1
        if self.submit_attempts == 1:
            self.submissions.append((episode_id, value))
            raise harness.EpisodeAPIError(400, "invalid_submission", "evidence rejected")
        return await super().submit(episode_id, value)


def test_driver_rejects_mutated_arguments_but_preserves_bounded_truncation(monkeypatch):
    _set_bridge_env(monkeypatch)

    async def scenario() -> None:
        metrics = harness.legacy.AgentMetrics()
        bridge = harness._ResponsesBridge(
            "http://127.0.0.1:1", "task", {"max_tokens": 64}, metrics, max_seq_len=None
        )
        bridge._pending = harness._PendingTool(
            "call-1", "terminal_exec", "terminal.exec", '{"command":"id"}'
        )
        bridge._history_after_model = []
        client = _FakeEpisodeClient(truncated=True)
        driver = harness._AppServerDriver(
            Path("/unused"), bridge, client, "episode", "task", metrics
        )
        driver._thread_id = "thread"
        driver._turn_id = "turn"
        with pytest.raises(harness.CodexHarnessError, match="mutated raw"):
            await driver._handle_tool_request(
                {
                    "id": 1,
                    "method": "item/tool/call",
                    "params": {
                        "threadId": "thread",
                        "turnId": "turn",
                        "callId": "call-1",
                        "namespace": None,
                        "tool": "terminal_exec",
                        "arguments": {"command": "whoami"},
                    },
                }
            )
        observation = await driver._execute_terminal({"command": "id"})
        assert observation == {
            "exit_code": 0,
            "output": "uid=1000",
            "timed_out": False,
            "truncated": True,
        }
        client.output = "x" * (harness.MAX_TOOL_OUTPUT_BYTES + 1)
        with pytest.raises(harness.CodexHarnessError, match="signed boundary"):
            await driver._execute_terminal({"command": "id"})

    asyncio.run(scenario())


def test_driver_prefers_policy_boundary_set_while_stdout_read_is_pending(monkeypatch):
    _set_bridge_env(monkeypatch)

    async def scenario() -> None:
        bridge = harness._ResponsesBridge(
            "http://127.0.0.1:1",
            "task",
            {"max_tokens": 64},
            harness.legacy.AgentMetrics(),
            max_seq_len=None,
        )
        bridge._fatal = asyncio.get_running_loop().create_future()
        boundary = harness.CodexModelFailure("sampled policy failure")

        class RaceStdout:
            async def readline(self) -> bytes:
                bridge._fatal.set_result(boundary)
                return b""

        class RaceProcess:
            stdout = RaceStdout()
            returncode = None

        driver = harness._AppServerDriver(
            Path("/unused"),
            bridge,
            _FakeEpisodeClient(),
            "episode",
            "task",
            harness.legacy.AgentMetrics(),
        )
        driver._process = RaceProcess()  # type: ignore[assignment]
        with pytest.raises(harness.CodexModelFailure) as caught:
            await driver._read_message()
        assert caught.value is boundary

    asyncio.run(scenario())


def _stock_codex_binary() -> Path:
    binary_name = shutil.which("codex")
    if binary_name is None:
        pytest.skip("stock Codex is not installed on this controller")
    version = subprocess.run(
        [binary_name, "--version"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if version != harness.CODEX_CLI_VERSION:
        pytest.skip(f"requires {harness.CODEX_CLI_VERSION}, found {version}")
    return Path(binary_name)


def test_stock_codex_0145_app_server_fake_miles_integration(monkeypatch):
    binary = _stock_codex_binary()
    _set_bridge_env(monkeypatch, max_tokens=32_768)

    async def scenario() -> None:
        long_reasoning = "r" * 100_000
        maximum_output = "x" * harness.MAX_TOOL_OUTPUT_BYTES
        runner, miles_url, miles_requests = await _fake_miles(
            [
                _completion(
                    "terminal.exec",
                    '{"command":"id"}',
                    "call-1",
                    long_reasoning,
                    25_000,
                ),
                _completion(
                    "submit",
                    '{"evidence":"uid=1000"}',
                    "call-2",
                    "submit",
                    60_000,
                ),
            ]
        )
        metrics = harness.legacy.AgentMetrics()
        client = _FakeEpisodeClient(output=maximum_output, truncated=True)
        try:
            async with harness._ResponsesBridge(
                miles_url,
                "solve target",
                {"temperature": 0.7, "max_tokens": 32_768},
                metrics,
                max_seq_len=200_000,
            ) as bridge, harness._AppServerDriver(
                binary,
                bridge,
                client,
                "episode-stock",
                "solve target",
                metrics,
            ) as driver:
                status = await asyncio.wait_for(driver.drive(), timeout=20.0)
                assert status == "completed"
        finally:
            await runner.cleanup()

        assert client.exec_calls == [("episode-stock", "id")]
        assert client.submissions == [
            ("episode-stock", {"evidence": "uid=1000"})
        ]
        assert len(miles_requests) == 2
        assert [tool["function"]["name"] for tool in miles_requests[0]["tools"]] == [
            "terminal.exec",
            "submit",
        ]
        tool_message = miles_requests[1]["messages"][-1]
        assert tool_message["role"] == "tool"
        assert json.loads(tool_message["content"])["output"] == maximum_output
        assert json.loads(tool_message["content"])["truncated"] is True
        assert len(json.loads(tool_message["content"])["output"]) == (
            harness.MAX_TOOL_OUTPUT_BYTES
        )

    asyncio.run(scenario())


def test_stock_codex_turn_limit_is_a_normal_policy_boundary(monkeypatch):
    binary = _stock_codex_binary()
    _set_bridge_env(monkeypatch)
    monkeypatch.setenv("SECRLENV_MAX_TURNS", "1")

    async def scenario() -> None:
        runner, miles_url, miles_requests = await _fake_miles(
            [
                _completion(
                    "terminal.exec", '{"command":"id"}', "call-1", "inspect", 20
                )
            ]
        )
        metrics = harness.legacy.AgentMetrics()
        client = _FakeEpisodeClient()
        try:
            async with harness._ResponsesBridge(
                miles_url,
                "solve target",
                {"max_tokens": 128},
                metrics,
                max_seq_len=512,
            ) as bridge, harness._AppServerDriver(
                binary,
                bridge,
                client,
                "episode-limit",
                "solve target",
                metrics,
            ) as driver:
                assert await asyncio.wait_for(driver.drive(), timeout=20.0) == "max_turns"
        finally:
            await runner.cleanup()

        assert client.exec_calls == [("episode-limit", "id")]
        assert client.submissions == []
        assert len(miles_requests) == 1
        assert metrics.turns == 1

    asyncio.run(scenario())


def test_stock_codex_model_format_failure_retains_sample_as_normal_failure(monkeypatch):
    binary = _stock_codex_binary()
    _set_bridge_env(monkeypatch)

    async def scenario() -> None:
        completion = {
            "id": "chatcmpl-policy-failure",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "I will only describe what to do.",
                        "reasoning_content": "decline tool use",
                        "tool_calls": [],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        runner, miles_url, miles_requests = await _fake_miles([completion])
        metrics = harness.legacy.AgentMetrics()
        client = _FakeEpisodeClient()
        try:
            async with harness._ResponsesBridge(
                miles_url,
                "solve target",
                {"max_tokens": 128},
                metrics,
                max_seq_len=512,
            ) as bridge, harness._AppServerDriver(
                binary,
                bridge,
                client,
                "episode-policy-failure",
                "solve target",
                metrics,
            ) as driver:
                assert await asyncio.wait_for(driver.drive(), timeout=20.0) == "max_turns"
        finally:
            await runner.cleanup()

        assert client.exec_calls == []
        assert client.submissions == []
        assert len(miles_requests) == 1
        assert metrics.turns == 1
        assert metrics.parse_failures == 1

    asyncio.run(scenario())


def test_stock_codex_daemon_400_tool_errors_recover_without_dropping_sample(monkeypatch):
    binary = _stock_codex_binary()
    _set_bridge_env(monkeypatch)

    async def scenario() -> None:
        runner, miles_url, miles_requests = await _fake_miles(
            [
                _completion(
                    "terminal.exec", '{"command":"bad"}', "call-1", "try", 15
                ),
                _completion(
                    "terminal.exec", '{"command":"id"}', "call-2", "recover", 30
                ),
                _completion(
                    "submit",
                    '{"evidence":"too vague"}',
                    "call-3",
                    "submit",
                    45,
                ),
                _completion(
                    "submit",
                    '{"evidence":"uid=1000"}',
                    "call-4",
                    "repair",
                    60,
                ),
            ]
        )
        metrics = harness.legacy.AgentMetrics()
        client = _RecoveringEpisodeClient()
        try:
            async with harness._ResponsesBridge(
                miles_url,
                "solve target",
                {"max_tokens": 128},
                metrics,
                max_seq_len=1024,
            ) as bridge, harness._AppServerDriver(
                binary,
                bridge,
                client,
                "episode-recovery",
                "solve target",
                metrics,
            ) as driver:
                assert await asyncio.wait_for(driver.drive(), timeout=20.0) == "completed"
        finally:
            await runner.cleanup()

        assert client.exec_calls == [
            ("episode-recovery", "bad"),
            ("episode-recovery", "id"),
        ]
        assert client.submissions == [
            ("episode-recovery", {"evidence": "too vague"}),
            ("episode-recovery", {"evidence": "uid=1000"}),
        ]
        assert len(miles_requests) == 4
        assert json.loads(miles_requests[1]["messages"][-1]["content"]) == {
            "error": "command rejected"
        }
        assert json.loads(miles_requests[3]["messages"][-1]["content"]) == {
            "error": "evidence rejected"
        }
        assert metrics.tool_calls == 4
        assert metrics.terminal_calls == 1
        assert metrics.submit_calls == 1

    asyncio.run(scenario())
