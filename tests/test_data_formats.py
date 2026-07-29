import json

import pytest

from yeto.data_formats import (
    DataNormalizationError,
    detect_data_format,
    normalize_sft_row,
)


OPENAI_ROW = {
    "messages": [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Say hello"},
        {"role": "assistant", "content": "hello"},
    ],
    "tools": [{"type": "function", "function": {"name": "noop"}}],
}

SHAREGPT_ROW = {
    "conversations": [
        {"from": "system", "value": "Be concise."},
        {"from": "human", "value": "Say hello"},
        {"from": "gpt", "value": "hello"},
    ],
    "tools": json.dumps([{"type": "function", "function": {"name": "noop"}}]),
}

ALPACA_ROW = {
    "system": "Be concise.",
    "instruction": "Say hello",
    "input": "",
    "output": "hello",
    "tools": [{"type": "function", "function": {"name": "noop"}}],
}


@pytest.mark.parametrize(
    ("row", "expected_format"),
    [
        (OPENAI_ROW, "openai"),
        (SHAREGPT_ROW, "sharegpt"),
        (ALPACA_ROW, "alpaca"),
    ],
)
def test_normalizes_common_sft_formats_to_identical_messages(row, expected_format):
    normalized = normalize_sft_row(row)

    assert normalized.source_format == expected_format
    assert normalized.messages == OPENAI_ROW["messages"]
    assert normalized.tools == OPENAI_ROW["tools"]


def test_alpaca_history_becomes_multiturn_messages():
    row = {
        "instruction": "final prompt",
        "input": "extra context",
        "output": "final answer",
        "history": [["old prompt", "old answer"]],
    }

    normalized = normalize_sft_row(row, "alpaca")

    assert normalized.messages == [
        {"role": "user", "content": "old prompt"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "final prompt\nextra context"},
        {"role": "assistant", "content": "final answer"},
    ]


def test_openai_preserves_tool_call_fields_and_normalizes_legacy_roles():
    row = {
        "messages": [
            {"role": "developer", "content": "policy"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call-1", "function": {"name": "lookup"}}],
            },
            {"role": "function", "name": "lookup", "content": "result"},
        ]
    }

    messages = normalize_sft_row(row, "openai").messages

    assert messages[0] == {"role": "system", "content": "policy"}
    assert messages[1]["tool_calls"][0]["id"] == "call-1"
    assert messages[2] == {"name": "lookup", "role": "tool", "content": "result"}


def test_auto_detection_rejects_unknown_and_ambiguous_rows():
    with pytest.raises(DataNormalizationError, match="could not detect"):
        detect_data_format({"text": "not an SFT row"}, row_index=4)
    with pytest.raises(DataNormalizationError, match="ambiguous"):
        detect_data_format(
            {"messages": [], "instruction": "prompt", "output": "answer"},
            row_index=7,
        )


@pytest.mark.parametrize(
    ("row", "data_format", "error"),
    [
        ({"messages": []}, "openai", "non-empty messages"),
        ({"messages": [{"role": "critic", "content": "x"}]}, "openai", "unsupported role"),
        ({"conversations": [{"from": "human", "value": 3}]}, "sharegpt", "content must"),
        ({"instruction": "prompt", "output": ""}, "alpaca", "non-empty 'output'"),
        ({"instruction": "prompt", "output": "answer", "tools": "{"}, "alpaca", "valid JSON"),
    ],
)
def test_invalid_rows_fail_with_row_context(row, data_format, error):
    with pytest.raises(DataNormalizationError, match=error) as exc:
        normalize_sft_row(row, data_format, row_index=12)
    assert str(exc.value).startswith("row 12:")
