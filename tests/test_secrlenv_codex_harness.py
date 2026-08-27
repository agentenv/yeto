from __future__ import annotations

import asyncio
import copy
import errno
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import aiohttp
import pytest
from aiohttp import web

from yeto_miles_secrlenv import codex_harness_agent as harness
from yeto_miles_secrlenv import reward as secrlenv_reward


def test_stock_codex_qwen38_adapter_process_binds_exact_xhigh_profile():
    environment = dict(os.environ)
    environment["YETO_CODEX_CHAT_TEMPLATE"] = "qwen38"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
from yeto_miles_secrlenv import codex_harness_agent as adapter
assert adapter.BACKEND_MODEL == "qwen38"
assert adapter.BACKEND_REASONING_EFFORT == "xhigh"
assert adapter.BACKEND_CHAT_TEMPLATE_KWARGS == {
    "enable_thinking": True,
    "preserve_thinking": True,
    "reasoning_effort": "xhigh",
}
""",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
        env=environment,
    )
    assert result.returncode == 0, result.stderr


def test_stock_codex_qwen35_adapter_process_binds_exact_fixed_profile():
    environment = dict(os.environ)
    environment["YETO_CODEX_CHAT_TEMPLATE"] = "qwen35"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
from yeto_miles_secrlenv import codex_harness_agent as adapter
assert adapter.BACKEND_MODEL == "qwen35"
assert adapter.BACKEND_REASONING_EFFORT == "xhigh"
assert adapter.BACKEND_CHAT_TEMPLATE_KWARGS == {"clear_thinking": False}
""",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
        env=environment,
    )
    assert result.returncode == 0, result.stderr


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
        "created": 1,
        "model": harness.BACKEND_MODEL,
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


def _fake_miles_sse(completion: dict[str, Any]) -> bytes:
    choice = completion["choices"][0]
    message = choice["message"]
    delta = {
        "role": message.get("role", "assistant"),
        "content": message.get("content"),
    }
    if message.get("reasoning_content") is not None:
        delta["reasoning_content"] = message["reasoning_content"]
    if message.get("tool_calls"):
        delta["tool_calls"] = [
            {**tool_call, "index": index}
            for index, tool_call in enumerate(message["tool_calls"])
        ]
    chunk = {
        "id": completion.get("id"),
        "object": "chat.completion.chunk",
        "created": completion.get("created", 1),
        "model": completion.get("model", harness.BACKEND_MODEL),
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": choice.get("finish_reason"),
            }
        ],
        "usage": completion.get("usage"),
    }
    return (
        b"data: "
        + json.dumps(
            chunk,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
        + b"\n\ndata: [DONE]\n\n"
    )


async def _fake_miles(
    completions: Iterable[dict[str, Any]],
) -> tuple[web.AppRunner, str, list[dict[str, Any]]]:
    queue = list(completions)
    requests: list[dict[str, Any]] = []

    async def chat(request: web.Request) -> web.Response:
        requests.append(await request.json())
        if not queue:
            return web.json_response({"error": "unexpected sample"}, status=409)
        return web.Response(
            body=_fake_miles_sse(queue.pop(0)),
            content_type="text/event-stream",
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    sockets = site._server.sockets if site._server is not None else []
    assert len(sockets) == 1
    return runner, f"http://127.0.0.1:{sockets[0].getsockname()[1]}", requests


async def _fake_miles_wire(
    body: bytes,
    *,
    content_type: str = "text/event-stream",
) -> tuple[web.AppRunner, str]:
    async def chat(_request: web.Request) -> web.Response:
        return web.Response(body=body, headers={"Content-Type": content_type})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    sockets = site._server.sockets if site._server is not None else []
    assert len(sockets) == 1
    return runner, f"http://127.0.0.1:{sockets[0].getsockname()[1]}"


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


def test_miles_fake_stream_round_trips_usage_reasoning_and_tool_call():
    raw_arguments = '{ "command": "printf café" }'
    completion = _completion(
        "terminal.exec",
        raw_arguments,
        "call-stream",
        "inspect café",
        321,
    )

    normalized = harness._parse_miles_sse(_fake_miles_sse(completion))

    assert normalized["object"] == "chat.completion"
    assert normalized["usage"] == completion["usage"]
    choice = normalized["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"] == {
        "role": "assistant",
        "content": "",
        "reasoning_content": "inspect café",
        "tool_calls": [
            {
                "id": "call-stream",
                "type": "function",
                "function": {
                    "name": "terminal.exec",
                    "arguments": raw_arguments,
                },
            }
        ],
    }


def test_completion_accepts_parser_whitespace_but_rejects_real_prose(monkeypatch):
    _set_bridge_env(monkeypatch)
    whitespace_completion = _completion(
        "terminal.exec",
        '{"command":"id"}',
        "call-whitespace",
        "inspect",
        32,
    )
    whitespace_completion["choices"][0]["message"]["content"] = "\n\n"
    bridge = harness._ResponsesBridge(
        "http://127.0.0.1:1",
        "task",
        {"max_tokens": 64},
        harness.legacy.AgentMetrics(),
        max_seq_len=None,
    )

    output = bridge._translate_completion(whitespace_completion)

    assert output[-1]["type"] == "function_call"
    assert output[-1]["name"] == "terminal_exec"
    assert bridge._messages[-1]["content"] == "\n\n"

    prose_completion = copy.deepcopy(whitespace_completion)
    prose_completion["choices"][0]["message"]["content"] = "I will describe it."
    prose_bridge = harness._ResponsesBridge(
        "http://127.0.0.1:1",
        "task",
        {"max_tokens": 64},
        harness.legacy.AgentMetrics(),
        max_seq_len=None,
    )
    with pytest.raises(harness.CodexModelFailure, match="mixed prose"):
        prose_bridge._translate_completion(prose_completion)


def test_miles_fake_stream_preserves_length_as_sequence_policy_boundary(monkeypatch):
    _set_bridge_env(monkeypatch)
    completion = _completion(
        "terminal.exec",
        '{"command":"id"}',
        "call-length",
        "inspect",
        321,
    )
    completion["choices"][0]["finish_reason"] = "length"

    normalized = harness._parse_miles_sse(_fake_miles_sse(completion))

    assert normalized["choices"][0]["finish_reason"] == "length"
    bridge = harness._ResponsesBridge(
        "http://127.0.0.1:1",
        "task",
        {"max_tokens": 64},
        harness.legacy.AgentMetrics(),
        max_seq_len=512,
    )
    with pytest.raises(harness.CodexSequenceLimit, match="truncated"):
        bridge._translate_completion(normalized)


def test_miles_fake_stream_rejects_malformed_framing_and_shapes():
    valid = _fake_miles_sse(
        _completion(
            "terminal.exec",
            '{"command":"id"}',
            "call-stream",
            "inspect",
            32,
        )
    )
    json_event, done_event, _empty = valid.split(b"\n\n")
    chunk = json.loads(json_event[len(b"data: ") :])
    invalid_chunks = []
    for mutate in (
        lambda value: value.update({"extra": True}),
        lambda value: value.update({"choices": []}),
        lambda value: value["choices"][0].update({"index": False}),
        lambda value: value["choices"][0]["delta"]["tool_calls"][0].update(
            {"index": False}
        ),
        lambda value: value["choices"][0].update({"delta": {"role": "assistant"}}),
    ):
        value = copy.deepcopy(chunk)
        mutate(value)
        invalid_chunks.append(
            b"data: "
            + json.dumps(value, separators=(",", ":")).encode()
            + b"\n\ndata: [DONE]\n\n"
        )
    malformed = [
        b"",
        b"data: [DONE]\n\n",
        json_event + b"\n\n",
        done_event + b"\n\n" + json_event + b"\n\n",
        json_event + b"\n\n" + done_event + b"\n\n" + done_event + b"\n\n",
        json_event + b"\n\n" + json_event + b"\n\n" + done_event + b"\n\n",
        b"data: \xff\n\ndata: [DONE]\n\n",
        b"data: {not-json}\n\ndata: [DONE]\n\n",
        b'data: {"id":"first","id":"second"}\n\ndata: [DONE]\n\n',
        b'data: {"created":NaN}\n\ndata: [DONE]\n\n',
        *invalid_chunks,
    ]

    for raw in malformed:
        with pytest.raises(harness.CodexHarnessError, match="Miles session returned"):
            harness._parse_miles_sse(raw)


def test_miles_fake_stream_omits_large_tito_metadata_from_client(monkeypatch):
    _set_bridge_env(monkeypatch, max_tokens=32_768)

    async def scenario() -> None:
        completion = _completion(
            "terminal.exec",
            '{"command":"id"}',
            "call-large-record",
            "inspect",
            32_768,
        )
        completion["choices"][0]["meta_info"] = {
            "output_token_logprobs": "x" * (harness.MAX_MILES_RESPONSE_BYTES + 1)
        }
        assert len(json.dumps(completion).encode()) > (harness.MAX_MILES_RESPONSE_BYTES)
        runner, miles_url, miles_requests = await _fake_miles([completion])
        try:
            async with harness._ResponsesBridge(
                miles_url,
                "task",
                {"max_tokens": 32_768},
                harness.legacy.AgentMetrics(),
                max_seq_len=65_536,
            ) as bridge:
                normalized = await bridge._sample_miles()
        finally:
            await runner.cleanup()

        assert normalized["choices"][0]["message"]["reasoning_content"] == ("inspect")
        assert "meta_info" not in normalized["choices"][0]
        assert miles_requests[0]["stream"] is True

    asyncio.run(scenario())


def test_miles_response_reader_rejects_exact_cap_plus_one_without_unbounded_read():
    class NeverReadContent:
        calls = 0

        def iter_chunked(self, _size):
            self.calls += 1
            raise AssertionError("oversized Content-Length must not be read")

        async def read(self):
            raise AssertionError("unbounded read is forbidden")

    declared_content = NeverReadContent()
    declared = SimpleNamespace(
        content_length=harness.MAX_MILES_RESPONSE_BYTES + 1,
        content=declared_content,
    )
    with pytest.raises(harness.CodexHarnessError, match="oversized"):
        asyncio.run(harness._read_bounded_miles_response(declared))
    assert declared_content.calls == 0

    class ChunkedContent:
        def __init__(self):
            self.yields = 0
            self.read_past_limit = False

        def iter_chunked(self, size):
            assert size == harness._MILES_RESPONSE_READ_CHUNK_BYTES

            async def chunks():
                block = b"x" * size
                for _ in range(harness.MAX_MILES_RESPONSE_BYTES // size):
                    self.yields += 1
                    yield block
                self.yields += 1
                yield b"x"
                self.read_past_limit = True
                yield b"must-not-be-read"

            return chunks()

        async def read(self):
            raise AssertionError("unbounded read is forbidden")

    chunked_content = ChunkedContent()
    chunked = SimpleNamespace(content_length=None, content=chunked_content)
    with pytest.raises(harness.CodexHarnessError, match="oversized"):
        asyncio.run(harness._read_bounded_miles_response(chunked))
    assert chunked_content.yields == 65
    assert chunked_content.read_past_limit is False


def test_miles_non_200_response_is_closed_without_reading_body(monkeypatch):
    _set_bridge_env(monkeypatch)

    class NeverReadContent:
        calls = 0

        def iter_chunked(self, _size):
            self.calls += 1
            raise AssertionError("non-200 response body must not be read")

        async def read(self):
            raise AssertionError("unbounded read is forbidden")

    class Response:
        status = 503
        content_type = "application/json"
        content_length = harness.MAX_MILES_RESPONSE_BYTES + 1
        content = NeverReadContent()
        closed = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            self.closed = True

    class Session:
        def __init__(self, response):
            self.response = response
            self.payload = None
            self.headers = None

        def post(self, _url, *, json, headers):
            self.payload = json
            self.headers = headers
            return self.response

    async def scenario() -> None:
        response = Response()
        session = Session(response)
        bridge = harness._ResponsesBridge(
            "http://127.0.0.1:1",
            "task",
            {"max_tokens": 64},
            harness.legacy.AgentMetrics(),
            max_seq_len=512,
        )
        bridge._session = session
        with pytest.raises(harness.CodexHarnessError, match="HTTP 503"):
            await bridge._sample_miles()
        assert session.payload["stream"] is True
        assert session.headers == {}
        assert response.content.calls == 0
        assert response.closed is True

    asyncio.run(scenario())


def test_miles_sample_accepts_content_type_parameters_and_enforces_cap(monkeypatch):
    _set_bridge_env(monkeypatch)

    async def accepted() -> None:
        completion = _completion(
            "terminal.exec",
            '{"command":"id"}',
            "call-media-type",
            "inspect",
            32,
        )
        runner, miles_url = await _fake_miles_wire(
            _fake_miles_sse(completion),
            content_type="text/event-stream; charset=utf-8",
        )
        try:
            async with harness._ResponsesBridge(
                miles_url,
                "task",
                {"max_tokens": 64},
                harness.legacy.AgentMetrics(),
                max_seq_len=512,
            ) as bridge:
                assert await bridge._sample_miles() == harness._parse_miles_sse(
                    _fake_miles_sse(completion)
                )
        finally:
            await runner.cleanup()

    async def rejected(body: bytes, content_type: str, match: str) -> None:
        runner, miles_url = await _fake_miles_wire(
            body,
            content_type=content_type,
        )
        try:
            async with harness._ResponsesBridge(
                miles_url,
                "task",
                {"max_tokens": 64},
                harness.legacy.AgentMetrics(),
                max_seq_len=512,
            ) as bridge:
                with pytest.raises(harness.CodexHarnessError, match=match):
                    await bridge._sample_miles()
        finally:
            await runner.cleanup()

    asyncio.run(accepted())
    asyncio.run(rejected(b"{}", "application/json", "media type"))
    asyncio.run(
        rejected(
            b"x" * (harness.MAX_MILES_RESPONSE_BYTES + 1),
            "text/event-stream",
            "oversized",
        )
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
        assert miles_requests[0]["stream"] is True
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


def test_bridge_compaction_is_same_session_trainable_and_keeps_atomic_tail(monkeypatch):
    _set_bridge_env(monkeypatch, max_tokens=128)
    monkeypatch.setenv("YETO_CODEX_COMPACTION_ENABLED", "1")
    monkeypatch.setenv("YETO_CODEX_COMPACTION_TRIGGER_TOKENS", "300")
    monkeypatch.setenv("YETO_CODEX_COMPACTION_SUMMARY_MAX_TOKENS", "128")
    monkeypatch.setenv("YETO_CODEX_MAX_COMPACTIONS", "3")

    summary = {
        "id": "chatcmpl-summary",
        "object": "chat.completion",
        "created": 1,
        "model": harness.BACKEND_MODEL,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "Goal: solve target. Evidence: uid=1000. Next: submit.",
                    "reasoning_content": "compress state",
                },
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
    }

    async def scenario() -> None:
        requests: list[tuple[dict[str, Any], dict[str, str]]] = []
        queue = [
            _completion("terminal.exec", '{"command":"id"}', "call-1", "inspect", 25),
            summary,
            _completion("submit", '{"evidence":"uid=1000"}', "call-2", "finish", 40),
        ]

        async def chat(request: web.Request) -> web.Response:
            requests.append((await request.json(), dict(request.headers)))
            return web.Response(
                body=_fake_miles_sse(queue.pop(0)),
                content_type="text/event-stream",
            )

        app = web.Application()
        app.router.add_post("/v1/chat/completions", chat)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        sockets = site._server.sockets if site._server is not None else []
        miles_url = f"http://127.0.0.1:{sockets[0].getsockname()[1]}"
        try:
            async with harness._ResponsesBridge(
                miles_url,
                "solve target",
                {"max_tokens": 128},
                harness.legacy.AgentMetrics(),
                max_seq_len=2048,
            ) as bridge:
                headers = {"Authorization": f"Bearer {bridge.token}"}
                initial = [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "solve target"}],
                    }
                ]
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{bridge.url}/v1/responses",
                        json=_codex_body(initial),
                        headers=headers,
                    ) as response:
                        assert response.status == 200
                    bridge.expect_tool_output("call-1", '{"output":"uid=1000"}')
                    second_input = copy.deepcopy(bridge._expected_input)
                    async with session.post(
                        f"{bridge.url}/v1/responses",
                        json=_codex_body(second_input),
                        headers=headers,
                    ) as response:
                        assert response.status == 200
        finally:
            await runner.cleanup()

        assert len(requests) == 3
        _first, first_headers = requests[0]
        summary_request, summary_headers = requests[1]
        resumed, resumed_headers = requests[2]
        assert first_headers["X-Miles-Compaction-Segment-Type"] == "execution"
        assert summary_headers["X-Miles-Compaction-Segment-Type"] == "summary"
        assert summary_headers["X-Miles-Compaction-Segment-Index"] == "1"
        assert summary_headers["X-Miles-Compaction-Context-Budget"] == "2048"
        assert summary_request["tools"] == []
        assert summary_request["max_tokens"] == 128
        assert summary_request["messages"][-1]["content"] == harness.COMPACTION_SUMMARY_INSTRUCTION
        assert resumed_headers["X-Miles-Compaction-Context-Window"] == "1"
        assert resumed_headers["X-Miles-Compaction-Segment-Index"] == "2"
        assert resumed_headers["X-Miles-Compaction-Segment-Type"] == "execution"
        assert resumed["messages"][0] == {"role": "system", "content": harness.BASE_INSTRUCTIONS}
        assert "uid=1000" in resumed["messages"][1]["content"]
        assert resumed["messages"][2:] == summary_request["messages"][2:-1]

    asyncio.run(scenario())


def test_bridge_compaction_trigger_is_consumed_context_boundary(monkeypatch):
    _set_bridge_env(monkeypatch, max_tokens=1024)
    monkeypatch.setenv("YETO_CODEX_COMPACTION_ENABLED", "1")
    monkeypatch.setenv("YETO_CODEX_COMPACTION_TRIGGER_TOKENS", "6144")
    monkeypatch.setenv("YETO_CODEX_COMPACTION_SUMMARY_MAX_TOKENS", "1024")
    monkeypatch.setenv("YETO_CODEX_MAX_COMPACTIONS", "3")

    bridge = harness._ResponsesBridge(
        "http://127.0.0.1:1",
        "boundary",
        {"max_tokens": 1024},
        harness.legacy.AgentMetrics(),
        max_seq_len=8192,
    )

    assert not bridge._should_compact(6143)
    assert bridge._should_compact(6144)
    assert (
        6144
        + harness.COMPACTION_SUMMARY_INSTRUCTION_TOKEN_RESERVE
        + bridge._compaction_summary_max_tokens
        <= 8192
    )


def test_bridge_compaction_rejects_trigger_without_summary_reserve(monkeypatch):
    _set_bridge_env(monkeypatch, max_tokens=512)
    monkeypatch.setenv("YETO_CODEX_COMPACTION_ENABLED", "1")
    monkeypatch.setenv("YETO_CODEX_COMPACTION_TRIGGER_TOKENS", "500")
    monkeypatch.setenv("YETO_CODEX_COMPACTION_SUMMARY_MAX_TOKENS", "512")

    with pytest.raises(harness.CodexHarnessError, match="insufficient room"):
        harness._ResponsesBridge(
            "http://127.0.0.1:1",
            "reserve",
            {"max_tokens": 512},
            harness.legacy.AgentMetrics(),
            max_seq_len=1024,
        )


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


def test_codex_exhausted_precreate_503_returns_signed_admission(monkeypatch):
    attempts = []

    class AdmissionClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def create(self, task_id, tier):
            attempts.append((task_id, tier))
            raise harness.EpisodeAPIError(
                503, "infrastructure_error", "unavailable"
            )

    monkeypatch.setenv("SECRLENV_REWARD_HMAC_KEY", "a" * 48)
    monkeypatch.setenv("SECRLENV_CAPACITY_MAX_WAIT_SECONDS", "30")
    monkeypatch.setattr(harness, "EpisodeClient", AdmissionClient)
    monkeypatch.setattr(harness, "_attest_runtime", lambda: Path("/codex"))
    monkeypatch.setattr(harness.legacy.random, "uniform", lambda *_args: 0.0)

    result = asyncio.run(
        harness.run(
            "http://127.0.0.1:1",
            None,
            metadata={"task_id": "CVE-2024-1234", "prompt_tier": "l2"},
        )
    )

    assert result is not None
    outcome, value = secrlenv_reward._verified_outcome(result)
    assert len(attempts) == 2
    assert outcome["schema"] == 2
    assert outcome["episode_id"] is None
    assert outcome[harness.INFRASTRUCTURE_503_RETRIES_KEY] == 1
    assert value == 0.0
    harness.legacy.require_no_episode_residue()


def test_codex_precreate_infrastructure_retry_timeout_remains_unsigned(
    monkeypatch,
):
    attempts = 0

    class AdmissionTimeoutClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def create(self, _task_id, _tier):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise harness.EpisodeAPIError(
                    503, "infrastructure_error", "unavailable"
                )
            await asyncio.Event().wait()

    monkeypatch.setenv("SECRLENV_REWARD_HMAC_KEY", "a" * 48)
    monkeypatch.setenv("SECRLENV_CAPACITY_MAX_WAIT_SECONDS", "0.01")
    monkeypatch.setattr(harness, "EpisodeClient", AdmissionTimeoutClient)
    monkeypatch.setattr(harness, "_attest_runtime", lambda: Path("/codex"))
    monkeypatch.setattr(harness.legacy.random, "uniform", lambda *_args: 0.0)

    result = asyncio.run(
        harness.run(
            "http://127.0.0.1:1",
            None,
            metadata={"task_id": "CVE-2024-1234", "prompt_tier": "l2"},
        )
    )

    assert result is None
    assert attempts == 2
    harness.legacy.require_no_episode_residue()


def test_run_does_not_start_codex_before_capacity_retry_creates_episode(
    monkeypatch,
):
    episode_id = "e" * 24
    events: list[str] = []

    class CapacityThenReadyClient:
        def __init__(self) -> None:
            self.create_calls = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def create(self, task_id: str, tier: str):
            assert (task_id, tier) == ("CVE-2024-1234", "l2")
            self.create_calls += 1
            events.append("create")
            if self.create_calls == 1:
                raise harness.EpisodeAPIError(
                    503, "capacity_reached", "busy"
                )
            return {
                "episode_id": episode_id,
                "prompt": "immutable task",
            }

        async def evaluate(self, value: str):
            assert value == episode_id
            events.append("evaluate")
            return {
                "task_id": "CVE-2024-1234",
                "episode_id": episode_id,
                "reward": 1.0,
                "passed": True,
            }

        async def close(self, value: str):
            assert value == episode_id
            events.append("close")
            return {"closed": True}

    async def drive_after_create(
        _binary,
        _base_url,
        _client,
        episode,
        _request_kwargs,
        _metrics,
        *,
        max_seq_len,
    ):
        assert episode["episode_id"] == episode_id
        assert max_seq_len is None
        assert events == ["create", "create"]
        events.append("drive")
        return "completed"

    monkeypatch.setenv("SECRLENV_CAPACITY_MAX_WAIT_SECONDS", "30")
    monkeypatch.setattr(harness, "EpisodeClient", CapacityThenReadyClient)
    monkeypatch.setattr(harness, "_attest_runtime", lambda: Path("/codex"))
    monkeypatch.setattr(harness, "_drive_codex", drive_after_create)
    monkeypatch.setattr(harness, "sign_outcome", lambda _outcome: "signed")
    monkeypatch.setattr(harness.legacy.random, "uniform", lambda *_args: 0.0)
    try:
        result = asyncio.run(
            harness.run(
                "http://127.0.0.1:1",
                None,
                metadata={"task_id": "CVE-2024-1234", "prompt_tier": "l2"},
            )
        )
    finally:
        harness.legacy._release_episode(episode_id)

    assert result is not None
    assert result["exit_status"] == "completed"
    assert events == ["create", "create", "drive", "evaluate", "close"]


def test_run_close_failure_returns_terminal_cleanup_evidence(monkeypatch):
    episode_id = "d" * 24
    close_calls = []

    class CloseFailingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def create(self, _task_id: str, _tier: str):
            return {"episode_id": episode_id, "prompt": "immutable task"}

        async def evaluate(self, value: str):
            assert value == episode_id
            return {
                "task_id": "CVE-2024-1234",
                "episode_id": episode_id,
                "reward": 1.0,
                "passed": True,
            }

        async def close(self, value: str):
            assert value == episode_id
            close_calls.append(value)
            raise harness.EpisodeClientError("close failed")

    async def complete_policy(*_args, **_kwargs):
        return "completed"

    monkeypatch.setattr(harness, "EpisodeClient", CloseFailingClient)
    monkeypatch.setattr(harness, "_attest_runtime", lambda: Path("/codex"))
    monkeypatch.setattr(harness, "_drive_codex", complete_policy)
    monkeypatch.setattr(harness, "sign_outcome", lambda _outcome: "signed")
    try:
        result = asyncio.run(
            harness.run(
                "http://127.0.0.1:1",
                None,
                metadata={"task_id": "CVE-2024-1234", "prompt_tier": "l2"},
            )
        )
        assert result is not None
        assert result["exit_status"] == harness.legacy.CLEANUP_ERROR_STATUS
        assert (
            result[harness.OUTCOME_KEY]["status"]
            == harness.legacy.CLEANUP_ERROR_STATUS
        )
        assert len(close_calls) == 2
        assert harness.legacy._EPISODE_PHASES[episode_id] == "cleanup_pending"
    finally:
        harness.legacy._release_episode(episode_id)


def test_driver_reaps_isolated_process_group_before_home_cleanup(monkeypatch):
    events: list[object] = []

    class FakeProcess:
        pid = 4321
        returncode = None

        async def wait(self):
            events.append("wait-parent")
            self.returncode = 0
            return 0

        def terminate(self):
            raise AssertionError("process-group TERM should be used")

        def kill(self):
            raise AssertionError("process-group KILL should be used")

    async def scenario() -> None:
        driver = harness._AppServerDriver(
            Path("/unused"),
            SimpleNamespace(),  # type: ignore[arg-type]
            _FakeEpisodeClient(),
            "episode",
            "task",
            harness.legacy.AgentMetrics(),
        )
        driver._process = FakeProcess()  # type: ignore[assignment]
        driver._process_group_id = 4321
        monkeypatch.setattr(harness, "_CODEX_DESCENDANT_TERM_GRACE_SECONDS", 0.0)
        monkeypatch.setattr(
            harness.os,
            "killpg",
            lambda process_group_id, sig: events.append((process_group_id, sig)),
        )

        await driver._stop()

        assert events == [
            (4321, harness.signal.SIGTERM),
            "wait-parent",
            (4321, harness.signal.SIGKILL),
        ]
        assert driver._process_group_id is None

    asyncio.run(scenario())


def test_codex_argv_disables_background_plugin_sync():
    bridge = SimpleNamespace(url="http://127.0.0.1:1234")

    argv = harness._codex_argv(Path("/codex"), bridge)  # type: ignore[arg-type]

    assert argv.count("features.plugins=false") == 1


def test_driver_retries_transient_nonempty_isolated_home_cleanup(monkeypatch, tmp_path):
    home_path = tmp_path / "codex-home"
    home_path.mkdir()

    class RacingHome:
        name = str(home_path)
        attempts = 0

        def cleanup(self):
            self.attempts += 1
            if self.attempts < 3:
                raise OSError(errno.ENOTEMPTY, "Directory not empty", "plugins")
            home_path.rmdir()

    async def scenario() -> None:
        home = RacingHome()
        driver = harness._AppServerDriver(
            Path("/unused"),
            SimpleNamespace(),  # type: ignore[arg-type]
            _FakeEpisodeClient(),
            "episode",
            "task",
            harness.legacy.AgentMetrics(),
        )
        driver._isolated_home = home  # type: ignore[assignment]
        monkeypatch.setattr(
            harness,
            "_CODEX_HOME_CLEANUP_RETRY_DELAYS_SECONDS",
            (0.0, 0.0, 0.0),
        )

        await driver.__aexit__(None, None, None)

        assert home.attempts == 3
        assert not home_path.exists()

    asyncio.run(scenario())


def test_driver_home_cleanup_exhaustion_is_best_effort_and_nonfatal(
    monkeypatch, tmp_path
):
    home_path = tmp_path / "codex-home"
    (home_path / "plugins").mkdir(parents=True)

    class PersistentlyRacingHome:
        name = str(home_path)
        attempts = 0

        def cleanup(self):
            self.attempts += 1
            raise OSError(errno.ENOTEMPTY, "Directory not empty", "plugins")

    async def scenario() -> None:
        home = PersistentlyRacingHome()
        driver = harness._AppServerDriver(
            Path("/unused"),
            SimpleNamespace(),  # type: ignore[arg-type]
            _FakeEpisodeClient(),
            "episode",
            "task",
            harness.legacy.AgentMetrics(),
        )
        driver._isolated_home = home  # type: ignore[assignment]
        monkeypatch.setattr(
            harness,
            "_CODEX_HOME_CLEANUP_RETRY_DELAYS_SECONDS",
            (0.0, 0.0),
        )

        # A resource-hygiene race must not change a successful episode result.
        await driver.__aexit__(None, None, None)

        assert home.attempts == 2
        assert not home_path.exists()

    asyncio.run(scenario())


def test_driver_teardown_failures_do_not_mask_active_episode_error(
    monkeypatch, tmp_path
):
    class NoLaunchDriver(harness._AppServerDriver):
        async def __aenter__(self):
            return self

    home_path = tmp_path / "codex-home"
    home_path.mkdir()

    class FailedHome:
        name = str(home_path)

        def cleanup(self):
            raise RuntimeError("secondary home cleanup failure")

    async def scenario() -> None:
        driver = NoLaunchDriver(
            Path("/unused"),
            SimpleNamespace(),  # type: ignore[arg-type]
            _FakeEpisodeClient(),
            "episode",
            "task",
            harness.legacy.AgentMetrics(),
        )
        driver._isolated_home = FailedHome()  # type: ignore[assignment]

        async def failed_stop():
            raise RuntimeError("secondary shutdown failure")

        monkeypatch.setattr(driver, "_stop", failed_stop)
        with pytest.raises(ValueError, match="primary episode failure"):
            async with driver:
                raise ValueError("primary episode failure")
        assert not home_path.exists()

    asyncio.run(scenario())


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
