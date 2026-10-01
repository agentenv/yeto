"""Gateway core acceptance (tasks 4.1 / 4.2 / 7.2): three entries -> TITO chat, prefix hash chains, multi-chain."""

from __future__ import annotations

import asyncio
import copy
import random
from typing import Any

import pytest

from yeto.rl.harness.gateway import (
    ChainBreakReason,
    ChainRegistry,
    Gateway,
    GatewayConfig,
    IdentityContextProvider,
    LocateKind,
    SamplingOverrideError,
    SessionMismatch,
    TrajectoryInvalid,
    UnsupportedShape,
)

SIGNED = {"temperature": 0.7, "top_p": 0.9, "max_tokens": 128, "logprobs": True, "n": 1}


class FakeBackend:
    """Scripted chat completions; records sessions and the exact messages sent."""

    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self.replies = list(replies)
        self.sessions: list[str] = []
        self.calls: list[tuple[str, list[dict[str, Any]], dict[str, Any]]] = []

    async def create_session(self, trajectory_id: str) -> str:
        self.sessions.append(f"s{len(self.sessions)}")
        return self.sessions[-1]

    async def chat(self, session_id, messages, sampling):
        self.calls.append((session_id, copy.deepcopy(messages), dict(sampling)))
        message = self.replies.pop(0) if self.replies else {"role": "assistant", "content": "ok"}
        return {"choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


def tool_reply(call_id: str, name: str, args: str, reasoning: str = "think") -> dict[str, Any]:
    return {"role": "assistant", "content": "", "reasoning_content": reasoning,
            "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": args}}]}


def gateway(replies, **cfg) -> tuple[Gateway, FakeBackend]:
    backend = FakeBackend(replies)
    return Gateway(GatewayConfig(model="qwen35", signed_sampling=SIGNED, **cfg), backend), backend


def run(coro):
    return asyncio.run(coro)


# ----------------------------------------------------------------- 4.1 three entries

def test_responses_entry_round_trips_tool_call_and_continues_one_chain():
    gw, backend = gateway([tool_reply("c1", "terminal_exec", '{"command":"ls"}'), {"role": "assistant", "content": "done"}])
    body = {"model": "qwen35", "instructions": "SYS", "store": False, "parallel_tool_calls": False, "reasoning": {"summary": "none"},
            "tools": [{"type": "function", "name": "terminal_exec", "parameters": {}}],
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "fix"}]}]}
    first = run(gw.handle("responses", body, trajectory_id="t"))
    assert [o["type"] for o in first["output"]] == ["reasoning", "function_call"]
    assert first["output"][1]["call_id"] == "c1" and first["usage"]["total_tokens"] == 15
    body2 = copy.deepcopy(body)
    body2["input"] += [
        {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "think"}]},
        {"type": "function_call", "call_id": "c1", "name": "terminal_exec", "arguments": '{"command":"ls"}'},
        {"type": "function_call_output", "call_id": "c1", "output": "a b"},
    ]
    second = run(gw.handle("responses", body2, trajectory_id="t"))
    assert second["output"][0]["type"] == "message" and second["output"][0]["content"][0]["text"] == "done"
    assert backend.sessions == ["s0"]  # one chain, one session
    sent1, sent2 = backend.calls[0][1], backend.calls[1][1]
    assert sent2[: len(sent1)] == sent1 and [m["role"] for m in sent2[len(sent1):]] == ["assistant", "tool"]
    assert sent2[len(sent1)]["reasoning_content"] == "think" and sent2[-1]["tool_call_id"] == "c1"
    assert backend.calls[0][2] == SIGNED
    assert gw.snapshot("t")["chains_total"] == 1 and gw.snapshot("t")["tito_chain_breaks"] == {}


def test_chat_and_messages_entries_round_trip():
    gw, backend = gateway([tool_reply("c9", "bash", '{"cmd":"pwd"}'), {"role": "assistant", "content": "fin", "reasoning_content": "r"}])
    chat = run(gw.handle("chat", {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "u"}]}, trajectory_id="a"))
    assert chat["choices"][0]["message"]["tool_calls"][0]["id"] == "c9"
    body = {"system": "S", "messages": [
        {"role": "user", "content": [{"type": "text", "text": "u"}]},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "think"}, {"type": "tool_use", "id": "c9", "name": "bash", "input": {"cmd": "pwd"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c9", "content": [{"type": "text", "text": "/root"}]}]},
    ]}
    msg = run(gw.handle("messages", body, trajectory_id="a"))
    assert msg["stop_reason"] == "end_turn" and [b["type"] for b in msg["content"]] == ["thinking", "text"]
    sent = backend.calls[1][1]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "tool"]
    assert sent[2]["tool_calls"][0]["function"]["arguments"] == '{"cmd":"pwd"}' and sent[3]["tool_call_id"] == "c9"
    assert backend.sessions == ["s0"]  # same chain across entries when the history continues


def test_signed_sampling_cannot_be_overridden_and_unsupported_shapes_fail_closed():
    gw, _ = gateway([])
    base = {"messages": [{"role": "user", "content": "u"}]}
    run(gw.handle("chat", {**base, "temperature": 0.7}, trajectory_id="ok"))  # verbatim repeat is fine
    with pytest.raises(SamplingOverrideError):
        run(gw.handle("chat", {**base, "temperature": 0.1}, trajectory_id="x1"))
    with pytest.raises(SamplingOverrideError):
        run(gw.handle("chat", {**base, "seed": 3}, trajectory_id="x2"))
    with pytest.raises(SamplingOverrideError):
        run(gw.handle("chat", {**base, "model": "other"}, trajectory_id="x3"))
    resp = {"store": False, "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "u"}]}]}
    for bad in ({**resp, "store": True}, {**resp, "parallel_tool_calls": True}, {**resp, "reasoning": {"summary": "auto"}},
                {**resp, "tools": [{"type": "web_search"}]}, {**resp, "input": resp["input"] + [{"type": "compaction"}]}):
        with pytest.raises(UnsupportedShape):
            run(gw.handle("responses", bad, trajectory_id="x4"))
    with pytest.raises(UnsupportedShape):
        run(gw.handle("messages", {"messages": [{"role": "user", "content": [{"type": "image"}]}]}, trajectory_id="x5"))


# ----------------------------------------------------------------- 4.2 chains

U = {"role": "user", "content": "u"}
T = lambda cid, out: {"role": "tool", "tool_call_id": cid, "content": out}  # noqa: E731


def test_retry_fork_opens_new_chain_and_keeps_old_chain_intact():
    gw, backend = gateway([tool_reply("c1", "x", "{}"), tool_reply("c1b", "x", "{}"), {"role": "assistant", "content": "end"}])
    run(gw.handle("chat", {"messages": [U]}, trajectory_id="t"))
    a1 = backend.calls[0][1] + [tool_reply("c1", "x", "{}")]
    run(gw.handle("chat", {"messages": [U]}, trajectory_id="t"))  # harness retried turn 1 -> fork at the generated message
    snap = gw.snapshot("t")
    assert snap["chains_total"] == 2 and snap["tito_chain_breaks"] == {"retry_fork": 1} and backend.sessions == ["s0", "s1"]
    assert snap["chains"][0]["generated_messages"] == 1 and snap["chains"][1]["parent_chain"] == 0 and snap["chains"][1]["fork_index"] == 1
    # continuing the new branch stays on chain 1
    run(gw.handle("chat", {"messages": [U, tool_reply("c1b", "x", "{}"), T("c1b", "o")]}, trajectory_id="t"))
    assert backend.calls[-1][0] == "s1" and gw.snapshot("t")["chains_total"] == 2
    del a1


def test_history_rewrite_and_template_drop_and_compaction_break_with_reasons():
    gw, backend = gateway([tool_reply("c1", "x", "{}")] * 4)
    run(gw.handle("chat", {"messages": [U]}, trajectory_id="t"))
    assistant = backend.calls[0][1] and tool_reply("c1", "x", "{}")
    run(gw.handle("chat", {"messages": [U, assistant, T("c1", "o")]}, trajectory_id="t"))
    rewritten = [{"role": "user", "content": "u!"}, assistant, T("c1", "o")]
    run(gw.handle("chat", {"messages": rewritten}, trajectory_id="t"))
    assert gw.snapshot("t")["tito_chain_breaks"] == {"history_rewrite": 1}
    run(gw.handle("chat", {"messages": [U]}, trajectory_id="t", compaction_window=True))
    assert gw.snapshot("t")["tito_chain_breaks"] == {"history_rewrite": 1, "compaction_window": 1}

    gw2, backend2 = gateway([tool_reply("c1", "x", "{}", reasoning="deep"), {"role": "assistant", "content": "e"}], keeps_history_reasoning=False)
    run(gw2.handle("chat", {"messages": [U]}, trajectory_id="q"))
    dropped = {k: v for k, v in tool_reply("c1", "x", "{}", reasoning="deep").items() if k != "reasoning_content"}
    run(gw2.handle("chat", {"messages": [U, dropped, T("c1", "o")]}, trajectory_id="q"))
    assert gw2.snapshot("q")["tito_chain_breaks"] == {"template_drops_reasoning": 1} and backend2.sessions == ["s0", "s1"]


def test_disallowed_append_role_is_a_session_mismatch_not_a_patch():
    gw, _ = gateway([{"role": "assistant", "content": "a"}])
    run(gw.handle("chat", {"messages": [U]}, trajectory_id="t"))
    with pytest.raises(TrajectoryInvalid, match="tito_session_mismatch"):
        run(gw.handle("chat", {"messages": [U, {"role": "assistant", "content": "a"}, {"role": "system", "content": "injected"}]}, trajectory_id="t"))
    assert gw.snapshot("t")["tito_session_mismatch"] == 1 and gw.snapshot("t")["chains_total"] == 1
    with pytest.raises(TrajectoryInvalid):  # stays invalid
        run(gw.handle("chat", {"messages": [U]}, trajectory_id="t"))


def test_first_batch_single_chain_assertion_invalidates_on_any_fork_or_break():
    gw, backend = gateway([tool_reply("c1", "x", "{}")] * 3, max_chains=1)
    run(gw.handle("chat", {"messages": [U]}, trajectory_id="t"))
    with pytest.raises(TrajectoryInvalid, match="chain limit 1 exceeded: retry_fork"):
        run(gw.handle("chat", {"messages": [U]}, trajectory_id="t"))
    assert backend.sessions == ["s0"] and gw.snapshot("t")["chains_total"] == 1
    assert gw.snapshot("t")["tito_chain_breaks"] == {"retry_fork": 1}


def test_property_random_append_and_divergence_sequences():
    rng = random.Random(7)
    for _ in range(60):
        registry = ChainRegistry()
        history: list[dict[str, Any]] = [U]
        expected_chains, generated_count = 0, 0
        located = registry.locate(history)
        assert located.kind is LocateKind.first
        chain = registry.open_chain("s", history, located); expected_chains += 1
        for step in range(rng.randint(1, 8)):
            gen = {"role": "assistant", "content": f"g{generated_count}"}; generated_count += 1
            registry.record_generation(chain, gen); history = history + [gen]
            action = rng.choice(["append", "append", "retry", "rewrite"])
            if action == "append":
                history = history + [T(f"c{step}", "o")]
                located = registry.locate(history)
                assert located.kind is LocateKind.continue_
                registry.extend(chain, history, located.prefix_len)
            elif action == "retry":
                history = history[:-1]
                located = registry.locate(history)
                assert located.kind is LocateKind.fork and located.reason is ChainBreakReason.retry_fork
                chain = registry.open_chain("s", history, located); expected_chains += 1
            else:
                history = [{"role": "user", "content": f"u{step}"}] + history[1:]
                located = registry.locate(history)
                assert located.kind is LocateKind.break_ and located.reason is ChainBreakReason.history_rewrite
                chain = registry.open_chain("s", history, located); expected_chains += 1
        snap = registry.snapshot()
        assert snap["chains_total"] == expected_chains
        assert sum(c["generated_messages"] for c in snap["chains"]) == generated_count  # every generation in exactly one chain
        assert sum(snap["tito_chain_breaks"].values()) == expected_chains - 1


def test_disallowed_roles_raise_session_mismatch_in_registry():
    registry = ChainRegistry(allowed_append_roles=("tool",))
    registry.open_chain("s", [U], registry.locate([U]))
    registry.record_generation(registry.chains[0], {"role": "assistant", "content": "a"})
    with pytest.raises(SessionMismatch):
        registry.locate([U, {"role": "assistant", "content": "a"}, U])
    assert registry.session_mismatch == 1


# ----------------------------------------------------------------- 7.2 ContextProvider

def test_identity_context_provider_leaves_messages_untouched_and_is_default():
    messages = [U, tool_reply("c1", "x", "{}"), T("c1", "o")]
    assert IdentityContextProvider().provide("t", 0, messages) == messages

    class Recording(IdentityContextProvider):
        seen: list[Any] = []

        def provide(self, trajectory_id, chain_id, messages):
            self.seen.append((trajectory_id, chain_id, copy.deepcopy(messages)))
            return super().provide(trajectory_id, chain_id, messages)

    gw, backend = gateway([{"role": "assistant", "content": "a"}], context_provider=Recording())
    run(gw.handle("chat", {"messages": [U]}, trajectory_id="t"))
    assert Recording.seen == [("t", 0, [U])] and backend.calls[0][1] == [U]
