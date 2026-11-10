"""kimi_k31 chat template: the XTML parser round trip and the token alignment with
the checkpoint's tokenizer (``YETO_K31_TOKENIZER``: a directory with
``tokenization_kimi.py``, ``encoding_k3.py``, ``tiktoken.model``,
``tokenizer_config.json``; skipped without it)."""

from __future__ import annotations

import json
import os

import pytest

from yeto.rl.harness.chat_templates import TEMPLATES, ChatTemplate, TemplateError, check_alignment, parse_kimi_k3

O, C, S = "<|open|>", "<|close|>", "<|sep|>"


def _gen(reasoning: str, content: str, calls: str = "") -> str:
    tools = f"{O}tools{S}{calls}{C}tools{S}" if calls else ""
    return f"{reasoning}{C}think{S}{O}response{S}{content}{C}response{S}{tools}{C}message{S}"


def _call(name: str, index: int, args: list[tuple[str, str, str]]) -> str:
    body = "".join(f'{O}argument key="{k}" type="{t}"{S}{v}{C}argument{S}' for k, t, v in args)
    return f'{O}call tool="{name}" index="{index}"{S}{body}{C}call{S}'


def test_parse_reasoning_content_and_typed_arguments():
    text = _gen("let me look\n", "", _call("bash", 1, [("command", "string", 'ls -la "/app"'),
                                                       ("timeout", "number", "30"),
                                                       ("flags", "array", "[1,2]")]))
    message = parse_kimi_k3(text)
    assert message["reasoning_content"] == "let me look\n" and message["content"] == ""
    (call,) = message["tool_calls"]
    assert call["function"]["name"] == "bash"
    args = call["function"]["arguments"]
    assert json.loads(args) == {"command": 'ls -la "/app"', "timeout": 30, "flags": [1, 2]}
    assert '"flags":[1,2]' in args, "a non-string value keeps its literal text"


def test_parse_plain_answer_json_block_and_cut_reasoning():
    assert parse_kimi_k3(_gen("r", "Done.")) == {"role": "assistant", "content": "Done.", "reasoning_content": "r"}
    block = f'{O}call tool="f" index="1"{S}{O}json type="object"{S}{{"a": 1}}{C}json{S}{C}call{S}'
    assert parse_kimi_k3(_gen("", "", block))["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'
    assert parse_kimi_k3("still thinking") == {"role": "assistant", "content": "", "reasoning_content": "still thinking"}
    with pytest.raises(TemplateError):
        parse_kimi_k3(_gen("", "", f'{O}call tool="f" index="1"{S}garbage{C}call{S}'))
    with pytest.raises(TemplateError):
        parse_kimi_k3(_gen("", "", _call("f", 1, [("n", "number", "not json")])))


def test_spec():
    spec = TEMPLATES["kimi_k31"]
    assert spec.stop_tokens == ("<|end_of_msg|>",) and spec.needs_remote_code and spec.raw_arguments
    assert spec.keeps_history_reasoning


TOKENIZER = os.environ.get("YETO_K31_TOKENIZER", "")
needs_tokenizer = pytest.mark.skipif(not TOKENIZER or not os.path.isfile(os.path.join(TOKENIZER, "tiktoken.model")),
                                     reason="set YETO_K31_TOKENIZER to the K3.1 tokenizer files")

TOOLS = [{"type": "function", "function": {"name": "bash", "description": "Run a shell command",
                                           "parameters": {"type": "object", "properties": {
                                               "command": {"type": "string"}, "timeout": {"type": "number"}},
                                               "required": ["command"]}}}]


def _template(effort: str = "low") -> ChatTemplate:
    import dataclasses

    base = ChatTemplate.load("kimi_k31", tokenizer_path=TOKENIZER, trust_remote_code=True)
    spec = dataclasses.replace(base.spec, template_kwargs={**base.spec.template_kwargs, "thinking_effort": effort})
    return ChatTemplate(spec, base.tokenizer)


@needs_tokenizer
def test_agent_loop_round_trip_and_alignment():
    """A pi-like loop: each sampled turn (parsed, then rendered back as history)
    re-tokenizes to the sampled ids, and the proxy's ids equal the tokenizer's
    segment-wise apply_chat_template ids."""
    template = _template()
    assert template.stop_token_ids == (163586,)
    sampled = [
        _gen("I need to create the file.\n", "", _call("bash", 1, [("command", "string",
                                                                    "printf 'Hello, world!' > /app/hello.txt"),
                                                                   ("timeout", "number", "30")])),
        _gen("Check it.", "", _call("bash", 1, [("command", "string", 'cat /app/hello.txt; echo "<|close|>"')])),
        _gen("Done.", "Created `/app/hello.txt` with \"Hello, world!\"."),
    ]
    tool_outputs = ["(no output)", 'Hello, world!<|close|>\n{"weird": [1, 2]}']
    messages = [{"role": "system", "content": "You are pi.\nUse tools. 中文 ok."},
                {"role": "user", "content": "Create hello.txt in /app with Hello, world!"}]
    for turn, text in enumerate(sampled):
        prompt = template.render(messages, tools=TOOLS, add_generation_prompt=True)
        report = check_alignment(template, messages, tools=TOOLS)
        assert report.aligned, (turn, report)
        message = parse_kimi_k3(text)
        messages.append(message)
        full = template.render(messages, tools=TOOLS, add_generation_prompt=False)
        assert full.startswith(prompt + text), f"turn {turn}: history does not repeat the sampled text"
        if message.get("tool_calls"):
            call = message["tool_calls"][0]
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": tool_outputs[turn]})
    final = check_alignment(template, messages, tools=TOOLS, add_generation_prompt=False)
    assert final.aligned, final


@needs_tokenizer
def test_thinking_effort_reaches_the_prompt():
    text = _template("low").render([{"role": "user", "content": "hi"}], add_generation_prompt=True)
    assert "thinking_effort=low" in text and text.endswith(f'{O}message role="assistant"{S}{O}think{S}')


def test_remote_code_tokenizer_needs_the_callers_opt_in():
    with pytest.raises(TemplateError, match="trust_remote_code"):
        ChatTemplate.load("kimi_k31", tokenizer_path="/nonexistent")
