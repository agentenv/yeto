"""The ``deepseek_v41`` chat template: the checkpoint's ``encoding/encoding.py`` renders
(tools on the first system message), DSML tool calls parse back to what renders again."""
from __future__ import annotations

import json

import pytest

from yeto.rl.harness.chat_templates import TEMPLATES, ChatTemplate, TemplateError, parse_deepseek_v41

EOS = "<｜end▁of▁sentence｜>"
CALL = ('Running it.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="bash">\n'
        '<｜DSML｜ parameter name="cmd" string="true">ls -la</｜DSML｜ parameter>\n'
        '<｜DSML｜ parameter name="timeout" string="false">30</｜DSML｜ parameter>\n'
        '</｜DSML｜ invoke>\n</｜DSML｜ calls>')

# a stand-in of the checkpoint encoder: records what it gets, renders a simple form
ENCODER = '''
import json
CALLS = []
def encode_messages(messages, thinking_mode, drop_thinking=True, reasoning_effort=None):
    CALLS.append({"messages": messages, "thinking_mode": thinking_mode, "drop_thinking": drop_thinking,
                  "reasoning_effort": reasoning_effort})
    out = "<bos>"
    for m in messages:
        out += "<" + m["role"] + ">" + str(m.get("content") or "") + ("[tools]" if m.get("tools") else "")
    if messages[-1]["role"] != "assistant":
        out += "<｜Assistant｜>" + ("<think>" if thinking_mode == "thinking" else "</think>")
    return out
'''


class _Tokenizer:
    def __init__(self, path):
        self.name_or_path = str(path)

    def encode(self, text, add_special_tokens=False):
        return [1] if text == EOS else [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


def _template(tmp_path, **kwargs):
    (tmp_path / "encoding").mkdir(exist_ok=True)
    (tmp_path / "encoding" / "encoding.py").write_text(ENCODER)
    spec = TEMPLATES["deepseek_v41"]
    if kwargs:
        import dataclasses
        spec = dataclasses.replace(spec, template_kwargs={**spec.template_kwargs, **kwargs})
    return ChatTemplate(spec, _Tokenizer(tmp_path))


def test_parse_reasoning_content_and_dsml_calls():
    message = parse_deepseek_v41("I should run ls.</think>" + CALL + EOS)
    assert message["reasoning_content"] == "I should run ls." and message["content"] == "Running it."
    (call,) = message["tool_calls"]
    assert call["function"]["name"] == "bash"
    assert json.loads(call["function"]["arguments"]) == {"cmd": "ls -la", "timeout": 30}
    assert parse_deepseek_v41("still thinking") == {"role": "assistant", "content": "", "reasoning_content": "still thinking"}
    assert parse_deepseek_v41("2", thinking=False) == {"role": "assistant", "content": "2"}
    assert parse_deepseek_v41("r</think>done") == {"role": "assistant", "content": "done", "reasoning_content": "r"}
    with pytest.raises(TemplateError, match="not DSML"):
        parse_deepseek_v41("r</think>x\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=bash>\n</｜DSML｜ calls>")


def test_render_through_the_checkpoint_encoder(tmp_path):
    template = _template(tmp_path)
    tools = [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}}]
    text = template.render([{"role": "user", "content": "hi"}], tools=tools, add_generation_prompt=True)
    assert text == "<bos><system>[tools]<user>hi<｜Assistant｜><think>"  # a system message carries the tools
    module = template._encoder()
    call = module.CALLS[-1]
    assert call["thinking_mode"] == "thinking" and call["reasoning_effort"] == "high" and call["drop_thinking"]
    assert call["messages"][0] == {"role": "system", "content": "", "tools": tools}
    template.render([{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}], tools=tools,
                    add_generation_prompt=True)
    assert len(module.CALLS[-1]["messages"]) == 2 and module.CALLS[-1]["messages"][0]["tools"] == tools
    head = template.render([{"role": "user", "content": "hi"}], add_generation_prompt=False)
    assert head == "<bos><user>hi"
    assert template.stop_token_ids == (1,) and template.encode("ab") == [97, 98]
    # tool-call arguments reach the encoder as objects
    template.render([{"role": "user", "content": "x"}, {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c", "type": "function", "function": {"name": "bash", "arguments": '{"cmd": "ls"}'}}]}],
        add_generation_prompt=True)
    assert module.CALLS[-1]["messages"][1]["tool_calls"][0]["function"]["arguments"] == {"cmd": "ls"}
    chat = _template(tmp_path, thinking=False, reasoning_effort="max")
    assert chat.render([{"role": "user", "content": "hi"}], add_generation_prompt=True).endswith("</think>")
    assert chat._encoder().CALLS[-1]["reasoning_effort"] == "max"
    assert chat.parse("2") == {"role": "assistant", "content": "2"}


def test_missing_encoder_is_refused(tmp_path):
    template = ChatTemplate(TEMPLATES["deepseek_v41"], _Tokenizer(tmp_path / "nowhere"))
    with pytest.raises(TemplateError, match="no prompt encoder"):
        template.render([{"role": "user", "content": "hi"}], add_generation_prompt=True)
