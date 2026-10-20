"""S25 N10.5 records proxy and N10.6 template token alignment (CPU).

The model is a scripted :class:`FakeEngine`: each reply is the text the chat
template itself renders for a chosen assistant message, so a correct proxy
shows no render drift. Tokenizers come from the local HF cache (skip if absent):
Qwen3-0.6B (synced to the remote test box) and GLM-4.7-Flash (stand-in for the
GLM5.3-Flash tokenizer, which is not available: GLM5.3 itself is not verified).
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
from typing import Any

import pytest

from yeto.rl.adapters.sampler_engines.fake import FakeEngine
from yeto.rl.compaction import CompactionConfig
from yeto.rl.engine.sampler_engine import SamplingParams, TokenSamplerEngine
from yeto.rl.harness.chat_templates import ChatTemplate, TemplateError, check_alignment, parse_glm, parse_qwen3, parse_qwen38
from yeto.rl.harness.gateway import Gateway, GatewayConfig
from yeto.rl.harness.gateway.proxy import RecordsProxy, chat_tools
from yeto.rl.harness.gateway.records import RecordingBackend, RecordsMismatch
from yeto.rl.harness.gateway.translate import UnsupportedShape, responses_to_messages

SIGNED = {"temperature": 1.0, "top_p": 1.0, "max_tokens": 256}
TOOLS = [{"type": "function", "name": "shell", "description": "run a command",
          "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}]


def _template(name: str) -> ChatTemplate:
    try:
        return ChatTemplate.load(name)
    except (OSError, ValueError, ImportError) as exc:  # tokenizer not in the local cache
        pytest.skip(f"{name} tokenizer not available: {exc}")


# The Qwen3.8-27B tokenizer files (not on the HF hub cache by default).
QWEN38_TOKENIZER = os.environ.get("YETO_QWEN38_TOKENIZER", "")


def _template_at(name: str, path: str) -> ChatTemplate:
    if not path:
        pytest.skip(f"set YETO_QWEN38_TOKENIZER to run the {name} template tests")
    return ChatTemplate.load(name, tokenizer_path=path)


@pytest.fixture(scope="module")
def qwen() -> ChatTemplate:
    return _template("qwen3")


@pytest.fixture(scope="module")
def glm() -> ChatTemplate:
    return _template("glm5")


@pytest.fixture(scope="module")
def qwen3_2507() -> ChatTemplate:
    return _template("qwen3_2507")


FIXTURES = {"qwen3": "qwen", "glm5": "glm", "qwen3_2507": "qwen3_2507"}


def call(call_id: str, cmd: str) -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": "shell", "arguments": json.dumps({"cmd": cmd})}}


def assistant(reasoning: str, content: str = "", calls: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    message = {"role": "assistant", "content": content, "reasoning_content": reasoning}
    if calls:
        message["tool_calls"] = calls
    return message


class ScriptedModel:
    """``token_reply`` for FakeEngine: replies with the template's own rendering
    of the next scripted assistant message, plus the stop token."""

    def __init__(self, template: ChatTemplate, replies: list[dict[str, Any]], *, return_stop: bool = True) -> None:
        self.template = template
        self.replies = list(replies)
        self.return_stop = return_stop

    def __call__(self, prompt: tuple[int, ...], index: int) -> list[int]:
        message = self.replies.pop(0)
        reply = model_text(self.template, message)
        ids = self.template.encode(reply)
        return ids + ([self.template.encode(turn_end(self.template, message))[0]] if self.return_stop else [])


def turn_end(template: ChatTemplate, message: dict[str, Any]) -> str:
    """The stop token the model emits after ``message``: GLM opens the next turn
    (tool output or user), Qwen3 closes its own turn."""
    if template.spec.name == "glm5":
        return "<|observation|>" if message.get("tool_calls") else "<|user|>"
    return "<|im_end|>"


def model_text(template: ChatTemplate, message: dict[str, Any]) -> str:
    """The text of ``message`` as the template renders it after a generation
    prompt, without the turn end."""
    rendered = template.render([{"role": "user", "content": "x"}, message], add_generation_prompt=False)
    opener = template.render([{"role": "user", "content": "x"}], add_generation_prompt=True)
    body = rendered[len(opener):]
    for stop in template.spec.stop_tokens:
        body = body.split(stop, 1)[0]
    return body.rstrip("\n") if template.spec.name == "glm5" else body


def engine_for(model: ScriptedModel, **kw: Any) -> FakeEngine:
    engine = FakeEngine(token_reply=model, **kw)
    engine.start(None, 3)
    return engine


def responses_body(items: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
    return {"model": "policy", "store": False, "instructions": "You are a coding agent.", "tools": TOOLS,
            "input": items, **kw}


def user_item(text: str) -> dict[str, Any]:
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def echo(output: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """What Codex sends back: the output items as input items."""
    items = []
    for item in output:
        item = {k: v for k, v in item.items() if k not in ("id", "status")}
        if item["type"] == "message":
            item["content"] = [{"type": "output_text", "text": c["text"]} for c in item["content"]]
        items.append(item)
    return items


def tool_output(call_id: str, text: str) -> dict[str, Any]:
    return {"type": "function_call_output", "call_id": call_id, "output": text}


SCRIPT = [assistant("look first", calls=[call("x", "ls")]),
          assistant("now read", calls=[call("x", "cat a.txt")]),
          assistant("done", content="The file says hello.")]


def run_codex(proxy: Any, trajectory_id: str, *, turns: int = 3) -> list[dict[str, Any]]:
    """Drive a Codex-like Responses loop: echo output, answer each tool call."""
    items = [user_item("read a.txt")]
    outputs = []
    for turn in range(turns):
        response = asyncio.run(proxy.handle("responses", responses_body(copy.deepcopy(items)), trajectory_id=trajectory_id))
        outputs.append(response)
        items += echo(response["output"])
        for out in response["output"]:
            if out["type"] == "function_call":
                items.append(tool_output(out["call_id"], f"result {turn}"))
    return outputs


# ---------------------------------------------------------------------- N10.5 records
@pytest.mark.parametrize("name", ["qwen3", "glm5", "qwen3_2507"])
def test_codex_responses_loop_records_one_aligned_stream(name, request):
    template = request.getfixturevalue(FIXTURES[name])
    engine = engine_for(ScriptedModel(template, SCRIPT))
    proxy = RecordsProxy(engine, template, model="policy", signed_sampling=SIGNED, max_chains=1)
    outputs = run_codex(proxy, "t1")
    # Qwen3-4B-Instruct-2507 has no reasoning block: its template renders no reasoning.
    first = ["function_call"] if name == "qwen3_2507" else ["reasoning", "function_call"]
    assert [o["type"] for o in outputs[0]["output"]] == first
    assert outputs[-1]["output"][-1]["content"][0]["text"] == "The file says hello."

    records = proxy.records("t1")
    assert records["gateway"]["chains_total"] == 1 and records["gateway"]["invalidated"] is None
    (session,) = records["sessions"]
    (sample,) = session["samples"]
    ids, mask, logprobs = sample["token_ids"], sample["loss_mask"], sample["logprobs"]
    assert len(ids) == len(mask) == len(logprobs)
    turns = sample["turns"]
    assert [t["kind"] for t in turns] == ["action"] * 3
    # Each completion sits at mask 1 right after its prompt ids, with its log-probs.
    pos = 0
    for turn in turns:
        pos += len(turn["stop_added"]) + len(turn["prompt_added"])
        n = len(turn["completion"])
        assert ids[pos:pos + n] == list(turn["completion"]) and mask[pos:pos + n] == [1] * n
        assert logprobs[pos:pos + n] == list(turn["logprobs"])
        pos += n
    assert pos == len(ids) and sum(mask) == sum(len(t["completion"]) for t in turns)
    # The prompts the engine got are prefixes of the final stream (append-only).
    prompts = [p for p, *_ in engine.token_requests]
    assert all(list(p[0]) == ids[:len(p[0])] for p in prompts)
    # N10.6: the stream equals a fresh tokenization of the rendering on every turn.
    assert records["counters"].get("render_drift_turns", 0) == 0
    assert all(t["render_diff"] is None for t in turns)
    # Source tag and policy version on every turn.
    assert {t["policy_version"] for t in turns} == {3}
    assert turns[0]["provenance"]["engine"] == "fake" and turns[0]["provenance"]["sampler_id"] == "records-proxy"
    assert turns[0]["provenance"]["sampling_digest"] == SamplingParams(max_tokens=256).digest()


def test_proxy_matches_the_existing_gateway_on_the_same_conversation(qwen):
    """Pair test: the gateway with the scripted session backend (TB2 path) and
    the records proxy give the same Responses output and the same chains."""

    class SessionBackend:
        def __init__(self, replies):
            self.replies = list(replies)

        async def create_session(self, trajectory_id):
            return "s0"

        async def chat(self, session_id, messages, sampling):
            message = self.replies.pop(0)
            return {"choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}

    proxy = RecordsProxy(engine_for(ScriptedModel(qwen, SCRIPT)), qwen, model="policy", signed_sampling=SIGNED)
    ours = run_codex(proxy, "t")
    # The reference backend replies with the messages the proxy parsed (same call ids).
    parsed = [qwen.parse(qwen.decode(qwen.encode(model_text(qwen, m)))) for m in SCRIPT]
    reference = Gateway(GatewayConfig(model="policy", signed_sampling=SIGNED,
                                      keeps_history_reasoning=qwen.spec.keeps_history_reasoning),
                        SessionBackend(parsed))
    theirs = run_codex(reference, "t")

    def strip(response):
        return [{k: v for k, v in item.items() if k != "id"} for item in response["output"]]

    assert [strip(r) for r in ours] == [strip(r) for r in theirs]
    a, b = proxy.gateway.snapshot("t"), reference.snapshot("t")
    for snap in (a, b):
        for chain in snap["chains"]:
            chain.pop("session_id")
    assert a == b


def test_meta_info_is_the_sglang_shape_the_miles_session_server_reads(qwen):
    engine = engine_for(ScriptedModel(qwen, SCRIPT[:1]))
    backend = RecordingBackend(engine, qwen)

    async def go():
        sid = await backend.create_session("t")
        return await backend.chat(sid, [{"role": "user", "content": "hi"}], SIGNED)

    completion = asyncio.run(go())
    meta = completion["choices"][0]["meta_info"]
    pairs = meta["output_token_logprobs"]
    assert len(pairs) == meta["completion_tokens"] == completion["usage"]["completion_tokens"]
    assert all(isinstance(lp, float) and isinstance(tid, int) for lp, tid, _ in pairs)
    assert pairs[-1][1] == qwen.encode("<|im_end|>")[0]


def test_stop_token_not_returned_by_the_engine_is_added_with_mask_zero(glm):
    engine = engine_for(ScriptedModel(glm, SCRIPT, return_stop=False))
    proxy = RecordsProxy(engine, glm, model="policy", signed_sampling=SIGNED)
    run_codex(proxy, "t")
    records = proxy.records("t")
    sample = records["sessions"][0]["samples"][0]
    added = [t["stop_added"] for t in sample["turns"]]
    assert [tuple(a) for a in added] == [(glm.encode("<|observation|>")[0],)] * 2 + [()]
    assert records["counters"]["stop_added"] == 2
    assert records["counters"].get("render_drift_turns", 0) == 0
    for turn in sample["turns"]:
        assert turn["completion"][-1] not in glm.stop_token_ids


def test_length_truncated_turn_and_changed_tools_fail_closed(qwen):
    engine = engine_for(ScriptedModel(qwen, SCRIPT * 2))
    proxy = RecordsProxy(engine, qwen, model="policy", signed_sampling={**SIGNED, "max_tokens": 3})
    first = asyncio.run(proxy.handle("responses", responses_body([user_item("go")]), trajectory_id="t"))
    assert first["status"] == "completed"
    items = [user_item("go")] + echo(first["output"]) + [user_item("more")]
    with pytest.raises(RecordsMismatch, match="length-truncated"):
        asyncio.run(proxy.handle("responses", responses_body(items), trajectory_id="t"))
    with pytest.raises(RecordsMismatch, match="tool schemas changed"):
        asyncio.run(proxy.handle("responses", responses_body([user_item("go")], tools=[]), trajectory_id="t"))


def test_signed_repetition_penalty_is_refused_not_dropped():
    from yeto.rl.engine.sampler_engine import SamplingConfigError
    from yeto.rl.harness.gateway.records import sampling_params

    with pytest.raises(SamplingConfigError, match="repetition_penalty"):
        sampling_params({**SIGNED, "repetition_penalty": 1.1})


def test_a_reload_during_a_turn_fails_closed(qwen):
    model = ScriptedModel(qwen, SCRIPT)
    engine = engine_for(model)

    def reply_then_reload(prompt, index):
        engine.start(None, 4)  # the engine serves a new version while the turn runs
        return model(prompt, index)

    engine.token_reply = reply_then_reload
    proxy = RecordsProxy(engine, qwen, model="policy", signed_sampling=SIGNED)
    with pytest.raises(RecordsMismatch, match="served version changed"):
        asyncio.run(proxy.handle("responses", responses_body([user_item("go")]), trajectory_id="t"))


def test_backend_refuses_an_engine_without_token_input(qwen):
    class TextOnly:
        pass

    assert isinstance(FakeEngine(), TokenSamplerEngine)
    with pytest.raises(TypeError, match="generate_tokens"):
        RecordingBackend(TextOnly(), qwen)  # type: ignore[arg-type]


def _compaction_proxy(template: ChatTemplate) -> tuple[RecordsProxy, Any]:
    summary = assistant("check", content="<summary>1. listed files</summary>")
    dry = engine_for(ScriptedModel(template, SCRIPT))
    run_codex(RecordsProxy(dry, template, model="policy", signed_sampling=SIGNED), "dry")
    third_prompt = len(dry.token_requests[2][0][0])
    # Budget so the third request (after two tool steps) triggers, not the second.
    cfg = CompactionConfig(context_budget=third_prompt, t_comp=1, keep_recent_steps=1, max_compactions=1)
    engine = engine_for(ScriptedModel(template, [SCRIPT[0], SCRIPT[1], summary, SCRIPT[2]]))
    return RecordsProxy(engine, template, model="policy", signed_sampling=SIGNED, compaction=cfg), engine


def test_compaction_summarizes_and_starts_a_new_segment(glm):
    """GLM with clear_thinking=False keeps every turn's reasoning, so the
    summary request appends to the same stream."""
    proxy, _ = _compaction_proxy(glm)
    outputs = run_codex(proxy, "t")
    assert outputs[-1]["output"][-1]["content"][0]["text"] == "The file says hello."
    records = proxy.records("t")
    assert records["counters"]["compactions"] == 1
    first, second = records["sessions"][0]["samples"]
    assert [t["kind"] for t in first["turns"]] == ["action", "action", "summary"]
    assert first["summary_ok"] is True and second["kept_steps"] in (0, 1)
    assert [t["kind"] for t in second["turns"]] == ["action"]
    second_prompt = glm.decode(second["token_ids"])
    assert "1. listed files" in second_prompt and "continues from an earlier one" in second_prompt
    assert "result 0" not in second_prompt  # the first step was summarized away


def test_compaction_on_a_template_that_drops_history_reasoning_fails_closed(qwen):
    """Qwen3 drops earlier reasoning once a user message follows, so the
    summary request cannot extend the recorded stream token-exactly."""
    proxy, _ = _compaction_proxy(qwen)
    with pytest.raises(RecordsMismatch, match="rendering of the earlier turns changed"):
        run_codex(proxy, "t")


# ---------------------------------------------------------------------- N10.6 alignment
CONVERSATION = [
    {"role": "system", "content": "You are a coding agent."},
    {"role": "user", "content": "列出文件 and read them"},
    assistant("plan", calls=[call("c1", "ls -la")]),
    {"role": "tool", "tool_call_id": "c1", "content": "a.txt\nb.txt"},
    assistant("next", calls=[call("c2", "cat a.txt"), call("c3", "cat b.txt")]),
    {"role": "tool", "tool_call_id": "c2", "content": "hello"},
    {"role": "tool", "tool_call_id": "c3", "content": "{\"k\": 1}"},
    assistant("done", content="Both files read."),
    {"role": "user", "content": "thanks"},
]


def test_codex_namespace_and_hosted_tools_are_left_out_other_types_refused():
    # Codex 0.145.0 sends its sub-agent tools as a namespace and a hosted web_search tool.
    namespace = {"type": "namespace", "name": "multi_agent_v1", "tools": [TOOLS[0]]}
    tools = chat_tools("responses", {"tools": [*TOOLS, namespace, {"type": "web_search"}]})
    assert [t["function"]["name"] for t in tools] == ["shell"]
    with pytest.raises(UnsupportedShape):
        chat_tools("responses", {"tools": [{"type": "custom", "name": "apply_patch"}]})
    body = {"store": False, "parallel_tool_calls": False, "tools": [*TOOLS, namespace, {"type": "web_search"}],
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}]}
    assert responses_to_messages(body) == [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize("name", ["qwen3", "glm5", "qwen3_2507"])
def test_template_rendering_matches_tokenizer_ids_token_by_token(name, request):
    template = request.getfixturevalue(FIXTURES[name])
    tools = chat_tools("responses", {"tools": TOOLS})
    for end in range(2, len(CONVERSATION) + 1):
        report = check_alignment(template, CONVERSATION[:end], tools=tools)
        assert report.aligned, (end, report)
        # A text prompt tokenized with special tokens would not double the template's BOS.
        assert report.engine_text_extra == 0


def test_glm_parser_round_trips_its_own_rendering(glm):
    for message in (SCRIPT[0], assistant("r", content="text", calls=[call("c", "echo 1")]),
                    assistant("r", calls=[{"id": "c", "type": "function",
                                           "function": {"name": "f", "arguments": json.dumps({"n": 3, "s": "x"})}}])):
        parsed = parse_glm(model_text(glm, message))
        assert parsed["reasoning_content"] == message["reasoning_content"]
        assert parsed["content"] == message["content"]
        assert [json.loads(c["function"]["arguments"]) for c in parsed.get("tool_calls", [])] == \
               [json.loads(c["function"]["arguments"]) for c in message.get("tool_calls", [])]


def test_parsers_fail_closed_on_malformed_tool_calls():
    with pytest.raises(TemplateError):
        parse_qwen3("<tool_call>\nnot json\n</tool_call>")
    with pytest.raises(TemplateError):
        parse_glm("r</think><tool_call><arg_key>a</arg_key><arg_value>1</arg_value></tool_call>")
    with pytest.raises(TemplateError):
        parse_qwen38('r\n</think>\n\n<tool_call>\n{"name": "f", "arguments": {}}\n</tool_call>')


def test_qwen38_merges_codex_system_messages_and_template_refusals_fail_closed():
    template = _template_at("qwen38", QWEN38_TOKENIZER)
    tools = chat_tools("responses", {"tools": TOOLS})
    # Codex: instructions + a developer message, both "system" in the gateway form.
    messages = [{"role": "system", "content": "a"}, {"role": "system", "content": "b"}, {"role": "user", "content": "u"}]
    text = template.render(messages, tools=tools, add_generation_prompt=True)
    assert "a\n\nb" in text and text.count("<|im_start|>system") == 1
    assert check_alignment(template, messages, tools=tools).aligned
    with pytest.raises(TemplateError, match="refused"):
        template.render([*messages, {"role": "system", "content": "late"}], add_generation_prompt=True)


def test_qwen38_parser_reads_the_xml_tool_call_format():
    # The Qwen3.8 template's own rendering of one assistant turn (after "<think>\n").
    text = ("look first\n</think>\n\nChecking.\n\n<tool_call>\n<function=exec_command>\n"
            "<parameter=cmd>\nls -la\nwc -l x\n</parameter>\n<parameter=n>\n3\n</parameter>\n"
            "</function>\n</tool_call>\n<tool_call>\n<function=noop>\n</function>\n</tool_call>")
    parsed = parse_qwen38(text)
    assert parsed["reasoning_content"] == "look first"
    assert parsed["content"] == "Checking."
    assert [c["function"]["name"] for c in parsed["tool_calls"]] == ["exec_command", "noop"]
    assert json.loads(parsed["tool_calls"][0]["function"]["arguments"]) == {"cmd": "ls -la\nwc -l x", "n": 3}
    assert parse_qwen38("still thinking") == {"role": "assistant", "content": "", "reasoning_content": "still thinking"}


# ---------------------------------------------------------------------- adapters
def test_sglang_and_vllm_adapters_send_token_id_prompts(tmp_path):
    from test_rl_hetero_sampler_engines import _config, _Http, _Process, _vllm

    from yeto.rl.adapters.sampler_engines.sglang_server import SGLangServerEngine
    from yeto.rl.adapters.sampler_engines.vllm_server import VllmServerEngine

    def sglang(body):
        return [{"text": "x", "meta_info": {"output_token_logprobs": [[-0.1, 7, None], [-0.2, 2, None]],
                                            "finish_reason": {"type": "stop"}}} for _ in body["input_ids"]]

    http = _Http({("GET", "/health"): {}, ("POST", "/generate"): sglang})
    engine = SGLangServerEngine(_config(tmp_path), spawn=lambda a, e: _Process(), http=http, sleep=lambda s: None)
    engine.start(None, 0)
    (group,) = engine.generate_tokens([[1, 2, 3]], SamplingParams(max_tokens=4, n=2), stop_token_ids=[2])
    sent = [b for m, p, b in http.calls if p == "/generate"][-1]
    assert sent["input_ids"] == [[1, 2, 3], [1, 2, 3]] and sent["sampling_params"]["stop_token_ids"] == [2]
    assert group[0].token_ids == (7, 2) and group[0].prompt_token_ids == (1, 2, 3)

    def completions(body):
        return {"choices": [{"index": i, "text": "t", "finish_reason": "stop",
                             "logprobs": {"tokens": ["token_id:4", "token_id:9"], "token_logprobs": [-0.3, -0.4]}}
                            for i in range(len(body["prompt"]) * body["n"])]}

    http = _Http({("GET", "/health"): {}, ("POST", "/v1/completions"): completions})
    engine = VllmServerEngine(_vllm(tmp_path), spawn=lambda a, e: _Process(), http=http, sleep=lambda s: None)
    engine.start(None, 0)
    (group,) = engine.generate_tokens([[5, 6]], SamplingParams(max_tokens=4), stop_token_ids=[9])
    sent = [b for m, p, b in http.calls if p == "/v1/completions"][-1]
    assert sent["prompt"] == [[5, 6]] and sent["stop_token_ids"] == [9]
    assert group[0].token_ids == (4, 9) and group[0].prompt_token_ids == (5, 6)


# ---------------------------------------------------------------------- HTTP shell
def test_http_shell_serves_codex_sse_and_records(qwen):
    from aiohttp.test_utils import TestClient, TestServer

    proxy = RecordsProxy(engine_for(ScriptedModel(qwen, SCRIPT)), qwen, model="policy", signed_sampling=SIGNED)

    async def go():
        async with TestClient(TestServer(proxy.app())) as client:
            created = await (await client.post("/trajectories")).json()
            tid = created["trajectory_id"]
            assert created["base_url"].endswith(f"/t/{tid}/v1")
            reply = await client.post(f"/t/{tid}/v1/responses", json=responses_body([user_item("go")], stream=True))
            assert reply.status == 200 and reply.headers["Content-Type"].startswith("text/event-stream")
            text = await reply.text()
            assert "event: response.completed" in text and text.endswith("data: [DONE]\n\n")
            bad = await client.post(f"/t/{tid}/v1/responses", json=responses_body([user_item("go")], store=True))
            assert bad.status == 400
            override = await client.post(f"/t/{tid}/v1/chat/completions",
                                         json={"model": "policy", "temperature": 0.1,
                                               "tools": chat_tools("responses", {"tools": TOOLS}),
                                               "messages": [{"role": "user", "content": "x"}]})
            assert override.status == 400
            records = await (await client.get(f"/t/{tid}/records")).json()
            assert records["sessions"][0]["samples"][0]["loss_mask"].count(1) > 0
            assert (await client.post(f"/t/{tid}/close")).status == 200

    asyncio.run(go())
