"""Qwen3 / Qwen3.5 official templates gain exact native assistant masks.

Uses tokenizers from the local Hugging Face cache only; each case skips when
its tokenizer is not cached, so the test never needs the network.
"""

from __future__ import annotations

import pytest

from yeto.data import (
    ExactAssistantMaskError,
    _row_tokens,
    normalize_sft_row,
    qwen_generation_template,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "f",
            "description": "d",
            "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}},
        },
    }
]

ROW = {
    "messages": [
        {"role": "system", "content": "SYSMARK"},
        {"role": "user", "content": "USERMARK one"},
        {"role": "assistant", "content": "hist", "reasoning_content": "old thought"},
        {"role": "user", "content": "USERMARK two"},
        {
            "role": "assistant",
            "content": "<think>\nnew thought\n</think>\n\ncalling",
            "tool_calls": [
                {"type": "function", "function": {"name": "f", "arguments": {"x": 1}}}
            ],
        },
        {"role": "tool", "content": "TOOLMARK"},
        {"role": "assistant", "content": "", "reasoning_content": "r3"},
        {"role": "assistant", "content": "final"},
    ]
}


def _cached_tokenizer(name):
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception as exc:  # not in the local cache
        pytest.skip(f"{name} tokenizer not cached: {exc}")


@pytest.mark.parametrize("name", ["Qwen/Qwen3-0.6B", "Qwen/Qwen3.5-0.8B"])
@pytest.mark.parametrize("tools", [None, TOOLS])
def test_qwen_native_mask_matches_official_ids_and_covers_only_assistant(name, tools):
    tokenizer = _cached_tokenizer(name)
    row = dict(ROW, tools=tools) if tools else ROW
    example = normalize_sft_row(row, "auto", row_index=0)
    official = tokenizer.apply_chat_template(
        example.messages, tools=example.tools, tokenize=True, return_dict=True
    )["input_ids"]

    ids, weights = _row_tokens(tokenizer, row, assistant_mask_mode="native")

    assert ids == list(official)
    trained = tokenizer.decode([t for t, w in zip(ids, weights) if w])
    untrained = tokenizer.decode([t for t, w in zip(ids, weights) if not w])
    for marker in ("SYSMARK", "USERMARK", "TOOLMARK", "<tool_response>", "system", "old thought"):
        assert marker not in trained
    assert "<|im_start|>" not in trained
    for marker in ("hist", "new thought", "calling", "<tool_call>", "r3", "final"):
        assert marker in trained
    assert trained.count("<|im_end|>") == 4
    assert "TOOLMARK" in untrained and "USERMARK two" in untrained


def test_rewrite_leaves_non_qwen_templates_alone():
    plain = "{% for message in messages %}{{ message.content }}{% endfor %}"
    assert qwen_generation_template(plain) is None


def test_non_qwen_template_without_generation_still_fails():
    class Tok:
        name_or_path = "other"
        chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}"

        def get_chat_template(self, chat_template=None, tools=None):
            return self.chat_template

    with pytest.raises(ExactAssistantMaskError, match="does not contain"):
        _row_tokens(Tok(), {"messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]})
