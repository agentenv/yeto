"""Gateway chains behind a compacting records proxy (CPU).

pi keeps its own whole history (its compaction is off) and sends all of it on
every call; the records proxy compacts behind the gateway and feeds the engine
``segment.head + messages[segment.base:]``.  These tests drive the real
gateway + :class:`RecordingBackend` with a scripted :class:`FakeEngine` and a
pi-like chat loop: the history keeps growing past the engine's window while it
stays one chain and one session; a resent request (lost reply) is replayed
instead of forked; parallel tool calls stay whole steps; the HTTP routes take
bodies over aiohttp's 1 MiB default even under a sub-app prefix.
"""

from __future__ import annotations

import asyncio
import copy
import json
import random
import threading
from typing import Any

import pytest
from hf_assets import require_hf_repo

from yeto.rl.adapters.sampler_engines.fake import FakeEngine
from yeto.rl.compaction import CompactionConfig
from yeto.rl.engine.sampler_engine import SamplerEngineError
from yeto.rl.harness.chat_templates import TEMPLATES, ChatTemplate
from yeto.rl.harness.gateway import TrajectoryInvalid
from yeto.rl.harness.gateway import chains as chains_mod
from yeto.rl.harness.gateway.chains import Chain, ChainRegistry, LocateKind, message_hash
from yeto.rl.harness.gateway.proxy import MAX_REQUEST_BYTES, RecordsProxy, TrajectoryForbidden
from yeto.rl.harness.gateway.records import ContextFull

SIGNED = {"temperature": 1.0, "top_p": 1.0, "max_tokens": 256}
TOOLS = [{"type": "function", "function": {
    "name": "shell", "description": "run a command",
    "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}}]
SUMMARY_MARK = "Do not call any tool in this turn."
STEPS = 9
CFG = CompactionConfig(context_budget=1200, t_comp=300, keep_recent_steps=1, max_compactions=3)
MAX_MODEL_LEN = 1700


@pytest.fixture(scope="module")
def glm() -> ChatTemplate:
    """GLM keeps every turn's reasoning (clear_thinking=False): the summary request extends the stream."""
    return ChatTemplate.load("glm5", tokenizer_path=require_hf_repo(TEMPLATES["glm5"].tokenizer))


def assistant(reasoning: str, content: str = "", calls: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content, "reasoning_content": reasoning}
    if calls:
        message["tool_calls"] = calls
    return message


def call(call_id: str, cmd: str) -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": "shell", "arguments": json.dumps({"cmd": cmd})}}


def model_text(template: ChatTemplate, message: dict[str, Any]) -> str:
    """``message`` as the template renders it after a generation prompt, without the turn end."""
    rendered = template.render([{"role": "user", "content": "x"}, message], add_generation_prompt=False)
    opener = template.render([{"role": "user", "content": "x"}], add_generation_prompt=True)
    body = rendered[len(opener):]
    for stop in template.spec.stop_tokens:
        body = body.split(stop, 1)[0]
    return body.rstrip("\n")


class Policy:
    """Scripted token policy: a summary when the prompt ends in the summary
    request, else the next step (two parallel tool calls), then a final answer."""

    def __init__(self, template: ChatTemplate, steps: int = STEPS) -> None:
        self.template, self.steps = template, steps
        self.actions = 0
        self.summaries = 0
        self.prompts: list[tuple[int, ...]] = []
        self.gate: threading.Event | None = None  # set: block until released
        self.entered = threading.Event()

    def __call__(self, prompt: tuple[int, ...], index: int) -> list[int]:
        self.prompts.append(prompt)
        self.entered.set()
        if self.gate is not None:
            self.gate.wait(10)
        if SUMMARY_MARK in self.template.decode(list(prompt[-40:])):
            self.summaries += 1
            message = assistant("condense", content=f"<summary>1. ran {self.actions} steps</summary>")
            end = "<|user|>"
        elif self.actions < self.steps:
            self.actions += 1
            k = self.actions
            message = assistant(f"step {k}", calls=[call(f"a{k}", f"cat a{k}.txt"), call(f"b{k}", f"cat b{k}.txt")])
            end = "<|observation|>"
        else:
            message = assistant("finish", content="All files read.")
            end = "<|user|>"
        return self.template.encode(model_text(self.template, message)) + self.template.encode(end)


def make_proxy(template: ChatTemplate, *, compaction: CompactionConfig | None = CFG,
               max_model_len: int = MAX_MODEL_LEN, steps: int = STEPS) -> tuple[RecordsProxy, FakeEngine, Policy]:
    policy = Policy(template, steps)
    engine = FakeEngine(token_reply=policy, max_model_len=max_model_len)
    engine.start(None, 3)
    return RecordsProxy(engine, template, model="policy", signed_sampling=SIGNED, compaction=compaction), engine, policy


def tool_text(name: str) -> str:
    return f"contents of {name}: " + " ".join(f"line{i} of {name}" for i in range(12))


def pi_echo(message: dict[str, Any]) -> dict[str, Any]:
    """What pi sends back for a reply: tool-call arguments re-serialized (``JSON.stringify``)."""
    message = copy.deepcopy(message)
    for c in message.get("tool_calls") or ():
        c["function"]["arguments"] = json.dumps(json.loads(c["function"]["arguments"]), separators=(",", ":"))
    return message


def body(history: list[dict[str, Any]]) -> dict[str, Any]:
    return {"model": "policy", "tools": TOOLS, "max_tokens": SIGNED["max_tokens"], "messages": copy.deepcopy(history)}


def answer_tools(reply: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"role": "tool", "tool_call_id": c["id"], "content": tool_text(json.loads(c["function"]["arguments"])["cmd"])}
            for c in reply.get("tool_calls") or ()]


async def pi_loop(proxy: RecordsProxy, tid: str, *, on_turn: Any = None) -> list[dict[str, Any]]:
    """pi: the whole history on every call until a reply without tool calls."""
    history = [{"role": "system", "content": "You are a coding agent."}, {"role": "user", "content": "read every file"}]
    while True:
        completion = await proxy.handle("chat", body(history), trajectory_id=tid)
        reply = completion["choices"][0]["message"]
        history.append(pi_echo(reply))
        if on_turn is not None:
            await on_turn(history)
        if not reply.get("tool_calls"):
            return history
        history += answer_tools(reply)


def past_window(template: ChatTemplate, history: list[dict[str, Any]]) -> bool:
    """The whole history no longer fits the engine (a session from message 0 cannot be served)."""
    return len(template.encode(template.render(history, tools=TOOLS, add_generation_prompt=True))) >= MAX_MODEL_LEN


def session_of(proxy: RecordsProxy, tid: str) -> Any:
    (session,) = [s for s in proxy.backend.sessions.values() if s.trajectory_id == tid]
    return session


# ---------------------------------------------------------------------- 1. one chain across compactions
def test_growing_history_stays_one_chain_one_session_across_compactions(glm):
    proxy, engine, policy = make_proxy(glm)
    history = asyncio.run(pi_loop(proxy, "t"))
    assert history[-1]["content"] == "All files read."
    records = proxy.records("t")
    gateway = records["gateway"]
    assert gateway["chains_total"] == 1 and gateway["tito_chain_breaks"] == {}
    assert gateway["tito_session_mismatch"] == 0 and gateway["invalidated"] is None
    assert gateway["retry_replays"] == 0 and gateway["compaction_fork_lost"] == 0
    (session,) = records["sessions"]
    assert records["counters"]["compactions"] == policy.summaries >= 2
    assert len(session["samples"]) == policy.summaries + 1
    assert records["counters"].get("render_drift_turns", 0) == 0 and "context_full" not in records["counters"]
    # the agent's history outgrew the engine window; every engine prompt stayed inside it
    full = len(glm.encode(glm.render(history, tools=TOOLS, add_generation_prompt=False)))
    assert full > MAX_MODEL_LEN
    assert all(len(p) < MAX_MODEL_LEN for p in policy.prompts)
    # the agent never saw a summary: its history has every generated action and nothing else
    assistants = [m for m in history if m["role"] == "assistant"]
    assert len(assistants) == STEPS + 1 and not any("<summary>" in (m["content"] or "") for m in assistants)
    (chain,) = proxy.gateway.registries["t"].chains
    assert [m["role"] for m in chain.messages] == [m["role"] for m in history]
    assert sum(chain.generated) == STEPS + 1


def test_segment_base_points_at_kept_whole_steps_of_the_chain(glm):
    """``base`` indexes the chain's (= the agent's) messages: each later segment
    starts at an assistant message and keeps its parallel tool outputs whole."""
    proxy, _, _ = make_proxy(glm)
    asyncio.run(pi_loop(proxy, "t"))
    (chain,) = proxy.gateway.registries["t"].chains
    segments = session_of(proxy, "t").segments
    bases = [s.base for s in segments]
    assert bases[0] == 0 and bases == sorted(bases) and len(set(bases)) == len(bases)
    for seg in segments[1:]:
        assert chain.messages[seg.base]["role"] == "assistant" and chain.generated[seg.base]
        assert seg.head[-1]["content"].startswith("This session continues")
        text = glm.decode(seg.token_ids)
        # the kept step's two parallel outputs, nothing from before it
        for c in chain.messages[seg.base]["tool_calls"]:
            assert f"contents of {json.loads(c['function']['arguments'])['cmd']}" in text
        for m in chain.messages[2:seg.base]:
            if m["role"] == "tool":
                assert m["content"] not in text


# ---------------------------------------------------------------------- 3. resent requests
def test_lost_reply_resend_after_compaction_is_replayed_not_forked(glm):
    """pi retries a call whose reply it lost: same history again.  Replayed from
    the recorded generation: no engine call, no new chain, the stream untouched."""
    proxy, engine, policy = make_proxy(glm)
    state = {"resent": 0}

    async def resend_after_compaction(history: list[dict[str, Any]]) -> None:
        if policy.summaries and not state["resent"] and history[-1].get("tool_calls"):
            state["resent"] = 1
            calls = len(engine.token_requests)
            again = await proxy.handle("chat", body(history[:-1]), trajectory_id="t")
            assert pi_echo(again["choices"][0]["message"]) == history[-1]
            assert len(engine.token_requests) == calls

    asyncio.run(pi_loop(proxy, "t", on_turn=resend_after_compaction))
    assert state["resent"]
    gateway = proxy.records("t")["gateway"]
    assert gateway["chains_total"] == 1 and gateway["tito_chain_breaks"] == {} and gateway["retry_replays"] == 1
    assert len(proxy.backend.sessions) == 1


class Stop(Exception):
    pass


def test_main_behavior_forks_from_the_whole_history_and_cannot_fit(glm):
    """Why compaction needs both gateway guards (replaying a resent request and
    refusing a fork of a compacted session): without them a resend is a
    fork, a new session from message 0, and after compactions the whole history
    exceeds the engine's window (ContextFull, the trial lost)."""
    proxy, _, _ = make_proxy(glm)
    proxy.gateway.config.replay_retries = False
    proxy.gateway.config.session_compacted = None
    state: dict[str, Any] = {}

    async def resend(history: list[dict[str, Any]]) -> None:
        if "error" not in state and history[-1].get("tool_calls") and past_window(glm, history[:-1]):
            with pytest.raises(ContextFull) as error:
                await proxy.handle("chat", body(history[:-1]), trajectory_id="t")
            state["error"] = error.value

    asyncio.run(pi_loop(proxy, "t", on_turn=resend))
    assert state["error"]
    assert proxy.records("t")["gateway"]["tito_chain_breaks"] == {"retry_fork": 1}


def test_a_fork_from_a_compacted_session_fails_closed_without_an_engine_call(glm):
    """A fork that is not a replayable resend (here: replay off) from a session
    that has compacted invalidates the trajectory up front, counted, instead of
    opening a session from message 0."""
    proxy, engine, policy = make_proxy(glm)
    proxy.gateway.config.replay_retries = False

    async def resend(history: list[dict[str, Any]]) -> None:
        if policy.summaries and history[-1].get("tool_calls"):
            calls, sessions = len(engine.token_requests), len(proxy.backend.sessions)
            with pytest.raises(TrajectoryInvalid, match="compaction_fork_lost: retry_fork"):
                await proxy.handle("chat", body(history[:-1]), trajectory_id="t")
            assert len(engine.token_requests) == calls and len(proxy.backend.sessions) == sessions
            raise Stop

    with pytest.raises(Stop):
        asyncio.run(pi_loop(proxy, "t", on_turn=resend))
    snap = proxy.records("t")["gateway"]
    assert snap["compaction_fork_lost"] == 1 and snap["tito_chain_breaks"] == {"retry_fork": 1}
    assert snap["chains_total"] == 1 and snap["invalidated"].startswith("compaction_fork_lost")
    with pytest.raises(TrajectoryInvalid):
        asyncio.run(proxy.handle("chat", body([{"role": "user", "content": "x"}]), trajectory_id="t"))


def test_a_fork_before_any_compaction_opens_a_session_as_on_main(glm):
    proxy, _, _ = make_proxy(glm)
    proxy.gateway.config.replay_retries = False

    async def go() -> None:
        history = [{"role": "system", "content": "sys"}, {"role": "user", "content": "read"}]
        await proxy.handle("chat", body(history), trajectory_id="t")
        await proxy.handle("chat", body(history), trajectory_id="t")

    asyncio.run(go())
    snap = proxy.records("t")["gateway"]
    assert snap["chains_total"] == 2 and snap["tito_chain_breaks"] == {"retry_fork": 1}
    assert snap["compaction_fork_lost"] == 0 and snap["invalidated"] is None


def test_compaction_off_keeps_the_retry_fork(glm):
    """Default behavior unchanged: without compaction a resend is a ``retry_fork``."""
    proxy, _, _ = make_proxy(glm, compaction=None, max_model_len=8192, steps=2)
    assert proxy.gateway.config.replay_retries is False

    async def go() -> None:
        history = [{"role": "user", "content": "read"}]
        first = await proxy.handle("chat", body(history), trajectory_id="t")
        await proxy.handle("chat", body(history), trajectory_id="t")
        assert first["choices"][0]["message"]["tool_calls"]

    asyncio.run(go())
    snap = proxy.records("t")["gateway"]
    assert snap["chains_total"] == 2 and snap["tito_chain_breaks"] == {"retry_fork": 1}
    assert "retry_replays" not in snap and "compaction_fork_lost" not in snap


def test_resend_while_the_first_call_still_runs_waits_for_its_reply(glm):
    """The resent request arrives during the generation: it gets the same reply,
    one engine call in all; the next turn continues the one chain."""
    proxy, engine, policy = make_proxy(glm)
    history = [{"role": "system", "content": "sys"}, {"role": "user", "content": "read"}]

    async def go() -> None:
        policy.gate = threading.Event()
        first = asyncio.ensure_future(proxy.handle("chat", body(history), trajectory_id="t"))
        await asyncio.to_thread(policy.entered.wait, 10)
        second = asyncio.ensure_future(proxy.handle("chat", body(history), trajectory_id="t"))
        await asyncio.sleep(0.05)
        assert not second.done()
        policy.gate.set()
        a, b = await asyncio.gather(first, second)
        assert a["choices"][0]["message"] == b["choices"][0]["message"]
        policy.gate = None
        reply = a["choices"][0]["message"]
        await proxy.handle("chat", body(history + [pi_echo(reply)] + answer_tools(reply)), trajectory_id="t")

    asyncio.run(go())
    assert len(engine.token_requests) == 2
    snap = proxy.records("t")["gateway"]
    assert snap["chains_total"] == 1 and snap["retry_replays"] == 1 and snap["tito_chain_breaks"] == {}


def test_resend_waiting_on_a_failed_call_generates_itself(glm):
    """The first call fails upstream (502, nothing recorded): the waiting resend
    is served by its own engine call on the same chain."""
    proxy, engine, policy = make_proxy(glm)
    real = engine.generate_tokens
    gate = threading.Event()
    calls = {"n": 0}

    def flaky(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            policy.entered.set()
            gate.wait(10)
            raise SamplerEngineError("HTTP 0: ReadTimeout")
        return real(*args, **kwargs)

    engine.generate_tokens = flaky
    history = [{"role": "system", "content": "sys"}, {"role": "user", "content": "read"}]

    async def go() -> dict[str, Any]:
        first = asyncio.ensure_future(proxy.handle("chat", body(history), trajectory_id="t"))
        await asyncio.to_thread(policy.entered.wait, 10)
        second = asyncio.ensure_future(proxy.handle("chat", body(history), trajectory_id="t"))
        await asyncio.sleep(0.05)
        gate.set()
        with pytest.raises(SamplerEngineError):
            await first
        return await second

    reply = asyncio.run(go())
    assert reply["choices"][0]["message"]["tool_calls"]
    snap = proxy.records("t")["gateway"]
    assert calls["n"] == 2 and snap["chains_total"] == 1 and snap["retry_replays"] == 0 and not snap["invalidated"]


def test_a_history_rewrite_after_compaction_is_a_counted_fork_loss(glm):
    """Rewriting an earlier tool output after compactions is still found
    (``history_rewrite``, not silently continued) and fails closed."""
    proxy, _, policy = make_proxy(glm)

    async def rewrite_an_old_output(history: list[dict[str, Any]]) -> None:
        if policy.summaries and history[-1].get("tool_calls"):
            rewritten = copy.deepcopy(history) + answer_tools(history[-1])
            rewritten[3]["content"] = "rewritten output"
            with pytest.raises(TrajectoryInvalid, match="compaction_fork_lost: history_rewrite at message 3"):
                await proxy.handle("chat", body(rewritten), trajectory_id="t")
            raise Stop

    with pytest.raises(Stop):
        asyncio.run(pi_loop(proxy, "t", on_turn=rewrite_an_old_output))
    snap = proxy.records("t")["gateway"]
    assert snap["tito_chain_breaks"] == {"history_rewrite": 1} and snap["compaction_fork_lost"] == 1


def test_error_tool_output_then_retried_call_continues_the_chain(glm):
    proxy, _, _ = make_proxy(glm, steps=3)

    async def go() -> None:
        history = [{"role": "system", "content": "sys"}, {"role": "user", "content": "read"}]
        reply = (await proxy.handle("chat", body(history), trajectory_id="t"))["choices"][0]["message"]
        history.append(pi_echo(reply))
        history += [{"role": "tool", "tool_call_id": c["id"], "content": "Error: timed out"} for c in reply["tool_calls"]]
        for _ in range(3):
            reply = (await proxy.handle("chat", body(history), trajectory_id="t"))["choices"][0]["message"]
            history.append(pi_echo(reply))
            history += answer_tools(reply)

    asyncio.run(go())
    snap = proxy.records("t")["gateway"]
    assert snap["chains_total"] == 1 and snap["tito_chain_breaks"] == {}


# ---------------------------------------------------------------------- 2. locate cost
def _hash_prefix(chain: Chain, request: list[dict[str, Any]]) -> int:
    """The cumulative-hash comparison the chains made before (reference)."""
    previous, n = "", 0
    for stored_hash, message in zip(chain.hashes, request):
        previous = message_hash(previous, message)
        if previous != stored_hash:
            break
        n += 1
    return n


def test_common_prefix_equals_the_cumulative_hash_comparison():
    rng = random.Random(7)
    variants = [
        lambda m: m,
        lambda m: {**m, "content": f"{m['content']}x"},
        lambda m: {**m, "content": 1} if m["role"] == "user" else m,
        lambda m: {**m, "content": True} if m["role"] == "user" else m,
        lambda m: {**m, "content": 1.0} if m["role"] == "user" else m,
        lambda m: {**m, "content": -0.0} if m["role"] == "user" else m,
        lambda m: {k: v for k, v in m.items() if k != "reasoning_content"} if m.get("reasoning_content") == "" else m,
        lambda m: pi_echo(m) if m.get("tool_calls") else m,
        lambda m: {**m, "tool_calls": [{**m["tool_calls"][0], "function": {
            **m["tool_calls"][0]["function"], "arguments": '{"cmd": "other"}'}}]} if m.get("tool_calls") else m,
        lambda m: dict(reversed(list(m.items()))),
    ]
    for _ in range(300):
        stored = [{"role": "system", "content": "s"}]
        for i in range(rng.randint(1, 8)):
            stored.append(rng.choice([{"role": "user", "content": rng.choice(["u", 1, 0.0])},
                                      {"role": "assistant", "content": "", "reasoning_content": rng.choice(["", "r"]),
                                       "tool_calls": [call(f"c{i}", rng.choice(["ls", "cat"]))]},
                                      {"role": "tool", "tool_call_id": f"c{i}", "content": "out"}]))
        chain = Chain(0, "s")
        for m in stored:
            chain.append(m, generated=m["role"] == "assistant")
        request = [rng.choice(variants)(copy.deepcopy(m)) if rng.random() < 0.3 else copy.deepcopy(m) for m in stored]
        request = request[: rng.randint(1, len(request))] + [{"role": "tool", "content": "new"}] * rng.randint(0, 2)
        assert chain.common_prefix(request) == _hash_prefix(chain, request), (stored, request)


def test_locate_hashes_only_messages_that_differ_in_form_once_for_all_chains(monkeypatch):
    """A long history: messages sent back as they were stored are compared as
    values (no JSON, no hash); re-serialized arguments are hashed once per
    request however many chains there are."""
    turns = 400
    history = [{"role": "system", "content": "s" * 5000}, {"role": "user", "content": "task"}]
    for i in range(turns):
        history.append(assistant(f"think {i}", calls=[call(f"c{i}", f"cmd {i}")]))
        history.append({"role": "tool", "tool_call_id": f"c{i}", "content": "x" * 20000})
    registry = ChainRegistry()
    located = registry.locate(history[:2])
    chain = registry.open_chain("s0", history[:2], located)
    for m in history[2:]:
        chain.append(m, generated=m["role"] == "assistant")
    # two older forks of the same history
    for cut in (100, 300):
        fork = registry.open_chain("s", history[:cut], registry.locate(history[:cut]))
        fork.append({"role": "assistant", "content": f"other {cut}"}, generated=True)
    hashed = []
    real = chains_mod.message_digest
    monkeypatch.setattr(chains_mod, "message_digest", lambda m: hashed.append(m) or real(m))

    request = copy.deepcopy(history) + [{"role": "tool", "tool_call_id": "z", "content": "new"}]
    out = registry.locate(request)
    assert (out.kind, out.chain, out.prefix_len) == (LocateKind.continue_, chain, len(history))
    assert len(hashed) == 2  # same values: only the two fork points are serialized
    hashed.clear()
    echoed = [pi_echo(m) for m in request]
    out = registry.locate(echoed)
    assert (out.kind, out.chain, out.prefix_len) == (LocateKind.continue_, chain, len(history))
    assert len(hashed) == turns  # each re-serialized assistant once, for three chains


# ---------------------------------------------------------------------- 4. HTTP body size
def _big_history(n_bytes: int) -> list[dict[str, Any]]:
    words = " ".join(f"w{i}" for i in range(n_bytes // 6))
    return [{"role": "system", "content": "sys"}, {"role": "user", "content": words}]


def test_model_route_takes_a_body_over_1_mib_under_a_subapp_prefix(glm):
    """aiohttp applies the root app's ``client_max_size`` (1 MiB default) to a
    sub-app's requests: the recorder mounts the proxy under ``/k/<secret>/``.
    The routes read the body with their own limit, so a long history passes."""
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    proxy, engine, _ = make_proxy(glm, compaction=None, max_model_len=1 << 23, steps=1)
    tid = proxy.open("big")
    payload = json.dumps(body(_big_history(3 * 1024**2))).encode()
    assert len(payload) > 2 * 1024**2

    async def go() -> None:
        outer = web.Application()  # as recorder.ProxyTokenRecorder and the Modal endpoint mount it
        outer.add_subapp("/k/secret-prefix-0123456789abcdef/", proxy.app())
        async with TestClient(TestServer(outer)) as client:
            url = f"/k/secret-prefix-0123456789abcdef{proxy.path(tid)}/v1/chat/completions"
            reply = await client.post(url, data=payload, headers={"Content-Type": "application/json"})
            assert reply.status == 200, await reply.text()
            assert (await reply.json())["choices"][0]["message"]["tool_calls"]

    asyncio.run(go())
    assert len(engine.token_requests) == 1


def test_model_route_refuses_a_body_over_its_limit_413_and_checks_the_token_first(glm):
    from aiohttp.test_utils import TestClient, TestServer

    proxy, engine, _ = make_proxy(glm, compaction=None, steps=1)
    assert proxy.max_request_bytes == MAX_REQUEST_BYTES >= 64 * 1024**2
    proxy.max_request_bytes = 100_000
    tid = proxy.open("big")
    payload = json.dumps(body(_big_history(200_000))).encode()

    async def chunks() -> Any:  # chunked: no Content-Length, the stream limit applies
        for i in range(0, len(payload), 30_000):
            yield payload[i:i + 30_000]

    async def go() -> None:
        async with TestClient(TestServer(proxy.app())) as client:
            url = f"{proxy.path(tid)}/v1/chat/completions"
            for data in (payload, chunks()):
                reply = await client.post(url, data=data, headers={"Content-Type": "application/json"})
                assert reply.status == 413, await reply.text()
                assert (await reply.json())["error"]["type"] == "RequestTooLarge"
            wrong = await client.post(f"/t/{tid}/not-the-token/v1/chat/completions", data=payload,
                                      headers={"Content-Type": "application/json"})
            assert wrong.status == 403
            small = await client.post(url, json=body([{"role": "user", "content": "read"}]))
            assert small.status == 200, await small.text()

    asyncio.run(go())
    assert len(engine.token_requests) == 1 and not proxy.gateway.snapshot(tid)["invalidated"]


@pytest.mark.parametrize("handler_cancellation", [False, True])
def test_http_client_timeout_then_resend_during_the_call(glm, handler_cancellation):
    """pi's idle timeout fires during a long call (summary + action in one
    request) and pi sends the call again.  As the recorder serves it (aiohttp's
    default: a disconnect does not cancel the handler) the first call keeps
    generating and the resend gets its reply: one engine call.  A server that
    cancels handlers drops the first call (nothing recorded) and the resend
    samples on the same chain: one recorded turn either way."""
    import aiohttp
    from aiohttp import web

    proxy, engine, policy = make_proxy(glm)
    tid = proxy.open("slow")
    history = [{"role": "system", "content": "sys"}, {"role": "user", "content": "read"}]

    async def go() -> dict[str, Any]:
        policy.gate = threading.Event()
        runner = web.AppRunner(proxy.app(), handler_cancellation=handler_cancellation)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        url = f"http://127.0.0.1:{runner.addresses[0][1]}{proxy.path(tid)}/v1/chat/completions"
        try:
            async with aiohttp.ClientSession() as client:
                with pytest.raises(asyncio.TimeoutError):
                    await client.post(url, json=body(history), timeout=aiohttp.ClientTimeout(total=0.3))
                resent = asyncio.ensure_future(client.post(url, json=body(history)))
                await asyncio.sleep(0.2)
                policy.gate.set()
                reply = await resent
                assert reply.status == 200, await reply.text()
                return await reply.json()
        finally:
            await runner.cleanup()

    reply = asyncio.run(go())
    assert reply["choices"][0]["message"]["tool_calls"]
    snap = proxy.gateway.snapshot(tid)
    assert snap["chains_total"] == 1 and snap["tito_chain_breaks"] == {}
    (session,) = proxy.records(tid)["sessions"]
    (sample,) = session["samples"]
    assert [t["kind"] for t in sample["turns"]] == ["action"]
    assert len(engine.token_requests) == (2 if handler_cancellation else 1)
    assert snap["retry_replays"] == (0 if handler_cancellation else 1)


def test_a_body_that_arrives_after_its_trajectory_was_forgotten_makes_no_state(glm):
    """A trial detached or cancelled while its request body is still arriving:
    the route refuses it (403) instead of making the trajectory's state again
    and sampling."""
    from aiohttp.test_utils import TestClient, TestServer

    proxy, engine, _ = make_proxy(glm)
    tid = proxy.open("gone")
    payload = json.dumps(body([{"role": "user", "content": "read"}])).encode()

    async def go() -> int:
        async def slow() -> Any:
            yield payload[:10]
            await asyncio.sleep(0.1)
            proxy.forget(tid)
            yield payload[10:]

        async with TestClient(TestServer(proxy.app())) as client:
            url = f"{proxy.path(tid)}/v1/chat/completions"
            reply = await client.post(url, data=slow(), headers={"Content-Type": "application/json"})
            return reply.status

    assert asyncio.run(go()) == 403
    assert engine.token_requests == [] and proxy.backend.sessions == {}
    assert proxy.gateway.registries == {} and proxy.backend.tools == {}


def test_handle_refuses_a_forgotten_trajectory_until_it_is_opened_again(glm):
    """The late request of the test above without HTTP (``handle``): no state, no engine call."""
    proxy, engine, _ = make_proxy(glm)
    proxy.open("t")
    proxy.forget("t")
    with pytest.raises(TrajectoryForbidden, match="forgotten"):
        asyncio.run(proxy.handle("chat", body([{"role": "user", "content": "read"}]), trajectory_id="t"))
    assert engine.token_requests == [] and proxy.backend.sessions == {} and proxy.gateway.registries == {}
    proxy.open("t")  # an id opened again is served
    asyncio.run(proxy.handle("chat", body([{"role": "user", "content": "read"}]), trajectory_id="t"))
    assert len(engine.token_requests) == 1


# ---------------------------------------------------------------------- SSE keep-alive
async def _serve(proxy: RecordsProxy) -> tuple[Any, str]:
    from aiohttp import web

    runner = web.AppRunner(proxy.app())
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    return runner, f"http://127.0.0.1:{runner.addresses[0][1]}"


def _sse_events(text: str) -> tuple[int, list[Any]]:
    comments = sum(1 for line in text.split("\n") if line == ": keep-alive")
    data = [line[len("data: "):] for line in text.split("\n") if line.startswith("data: ")]
    return comments, [d if d == "[DONE]" else json.loads(d) for d in data]


def test_a_long_streamed_call_sends_keep_alive_comments_then_its_reply(glm):
    import aiohttp

    proxy, engine, policy = make_proxy(glm)
    proxy.keepalive_s = 0.05
    tid = proxy.open("slow")

    async def go() -> tuple[int, str, dict[str, str]]:
        policy.gate = threading.Event()
        runner, base = await _serve(proxy)
        try:
            async with aiohttp.ClientSession() as client:
                url = f"{base}{proxy.path(tid)}/v1/chat/completions"
                post = asyncio.ensure_future(client.post(url, json={**body([{"role": "user", "content": "read"}]),
                                                                    "stream": True}))
                await asyncio.sleep(0.3)
                policy.gate.set()
                reply = await post
                return reply.status, await reply.text(), dict(reply.headers)
        finally:
            await runner.cleanup()

    status, text, headers = asyncio.run(go())
    assert status == 200 and headers["Content-Type"].startswith("text/event-stream")
    comments, events = _sse_events(text)
    assert comments >= 2 and events[-1] == "[DONE]"
    assert events[0]["choices"][0]["delta"]["tool_calls"]
    assert len(engine.token_requests) == 1


def test_an_error_after_the_keep_alive_headers_is_an_sse_error_event(glm):
    import aiohttp

    proxy, engine, _ = make_proxy(glm)
    proxy.keepalive_s = 0.05
    tid = proxy.open("slow")
    gate = threading.Event()

    def failing(*args: Any, **kwargs: Any) -> Any:
        gate.wait(10)
        raise SamplerEngineError("HTTP 0: ReadTimeout")

    engine.generate_tokens = failing

    async def go() -> tuple[int, str]:
        runner, base = await _serve(proxy)
        try:
            async with aiohttp.ClientSession() as client:
                url = f"{base}{proxy.path(tid)}/v1/chat/completions"
                post = asyncio.ensure_future(client.post(url, json={**body([{"role": "user", "content": "read"}]),
                                                                    "stream": True}))
                await asyncio.sleep(0.2)
                gate.set()
                reply = await post
                return reply.status, await reply.text()
        finally:
            await runner.cleanup()

    status, text = asyncio.run(go())
    comments, events = _sse_events(text)
    assert status == 200 and comments >= 1
    assert events == [{"error": {"type": "SamplerEngineError", "message": "HTTP 0: ReadTimeout", "code": 502}}]


def test_quick_and_non_streamed_calls_answer_as_before(glm):
    """Within the interval a streamed call answers its status and one SSE
    chunk; a non-streamed call never gets comments; without compaction there
    is no keep-alive at all."""
    import aiohttp

    assert make_proxy(glm, compaction=None)[0].keepalive_s is None
    proxy, _, policy = make_proxy(glm)
    assert proxy.keepalive_s == 30.0
    tid = proxy.open("t")

    async def go() -> list[tuple[int, str]]:
        runner, base = await _serve(proxy)
        out = []
        try:
            async with aiohttp.ClientSession() as client:
                url = f"{base}{proxy.path(tid)}/v1/chat/completions"
                history = [{"role": "user", "content": "read"}]
                reply = await client.post(url, json={**body(history), "stream": True})
                out.append((reply.status, await reply.text()))
                bad = await client.post(url, json={**body(history), "stream": True, "temperature": 0.2})
                out.append((bad.status, await bad.text()))
                proxy.keepalive_s = 0.05
                policy.gate = threading.Event()
                post = asyncio.ensure_future(client.post(url, json=body(history + [{"role": "user", "content": "x"}])))
                await asyncio.sleep(0.2)
                policy.gate.set()
                reply = await post
                out.append((reply.status, await reply.text()))
        finally:
            await runner.cleanup()
        return out

    (s1, t1), (s2, t2), (s3, t3) = asyncio.run(go())
    assert s1 == 200 and _sse_events(t1)[0] == 0 and _sse_events(t1)[1][-1] == "[DONE]"
    assert s2 == 400 and json.loads(t2)["error"]["type"] == "SamplingOverrideError"
    assert s3 == 200 and "keep-alive" not in t3 and json.loads(t3)["choices"]


def test_a_client_that_leaves_during_keep_alive_gets_the_reply_on_its_resend(glm):
    import aiohttp

    proxy, engine, policy = make_proxy(glm)
    proxy.keepalive_s = 0.05
    tid = proxy.open("slow")
    request = {**body([{"role": "user", "content": "read"}]), "stream": True}

    async def go() -> str:
        policy.gate = threading.Event()
        runner, base = await _serve(proxy)
        try:
            url = f"{base}{proxy.path(tid)}/v1/chat/completions"
            async with aiohttp.ClientSession() as client:
                reply = await client.post(url, json=request)
                assert reply.status == 200
                assert (await reply.content.readline()) == b": keep-alive\n"
            # the connection is closed: the call goes on
            await asyncio.sleep(0.15)
            async with aiohttp.ClientSession() as client:
                post = asyncio.ensure_future(client.post(url, json=request))
                await asyncio.sleep(0.1)
                policy.gate.set()
                return await (await post).text()
        finally:
            await runner.cleanup()

    _, events = _sse_events(asyncio.run(go()))
    assert events[-1] == "[DONE]" and events[0]["choices"][0]["delta"]["tool_calls"]
    assert len(engine.token_requests) == 1
    snap = proxy.gateway.snapshot(tid)
    assert snap["retry_replays"] == 1 and snap["chains_total"] == 1


def test_a_resend_waiting_on_a_trajectory_forgotten_meanwhile_is_refused_and_counts_nothing(glm):
    """The first call returns, then the trajectory is forgotten before the waiting
    resend resumes: the resend is refused (TrajectoryInvalid) and no replay
    counter is made again for the forgotten trajectory."""
    proxy, engine, policy = make_proxy(glm)
    history = [{"role": "system", "content": "sys"}, {"role": "user", "content": "read"}]

    async def go() -> Any:
        policy.gate = threading.Event()
        first = asyncio.ensure_future(proxy.handle("chat", body(history), trajectory_id="t"))
        await asyncio.to_thread(policy.entered.wait, 10)
        second = asyncio.ensure_future(proxy.handle("chat", body(history), trajectory_id="t"))
        await asyncio.sleep(0.05)
        (_, generation), = proxy.gateway._generations.values()
        generation.add_done_callback(lambda _: proxy.forget("t"))  # runs before the resend resumes
        policy.gate.set()
        return await asyncio.gather(first, second, return_exceptions=True)

    a, b = asyncio.run(go())
    assert a["choices"][0]["message"]["tool_calls"]
    assert isinstance(b, TrajectoryInvalid) and "forgotten" in str(b)
    assert "t" not in proxy.gateway.replays and "t" not in proxy.gateway.registries
    assert len(engine.token_requests) == 1
