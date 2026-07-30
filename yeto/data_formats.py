"""Normalize common SFT dataset schemas into Yeto's canonical chat rows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping


DATA_FORMATS = ("auto", "openai", "sharegpt", "alpaca")
CANONICAL_ROLES = ("system", "user", "assistant", "tool")


class DataNormalizationError(ValueError):
    """A dataset row cannot be converted into a canonical SFT example."""


@dataclass
class NormalizedSFTExample:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    source_format: str


def _fail(message: str, row_index: int | None = None) -> DataNormalizationError:
    prefix = f"row {row_index}: " if row_index is not None else ""
    return DataNormalizationError(prefix + message)


def validate_data_format(data_format: str) -> str:
    if data_format not in DATA_FORMATS:
        raise DataNormalizationError(
            f"data_format must be one of {DATA_FORMATS}, got {data_format!r}"
        )
    return data_format


def _normalize_tools(value: Any, row_index: int | None) -> list[dict[str, Any]] | None:
    if value in (None, "", []):
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise _fail(f"tools is not valid JSON: {exc}", row_index) from exc
    if not isinstance(value, list) or not all(isinstance(tool, Mapping) for tool in value):
        raise _fail("tools must be a list of objects or a JSON-encoded list", row_index)
    return [dict(tool) for tool in value]


def _normalize_message(
    message: Any,
    *,
    role_key: str,
    content_key: str,
    role_map: Mapping[str, str],
    row_index: int | None,
    message_index: int,
) -> dict[str, Any]:
    if not isinstance(message, Mapping):
        raise _fail(f"message {message_index} must be an object", row_index)

    raw_role = message.get(role_key)
    if not isinstance(raw_role, str) or not raw_role.strip():
        raise _fail(f"message {message_index} has no string {role_key!r}", row_index)
    role = role_map.get(raw_role.strip().lower())
    if role is None:
        accepted = ", ".join(sorted(role_map))
        raise _fail(
            f"message {message_index} has unsupported role {raw_role!r}; "
            f"expected one of: {accepted}",
            row_index,
        )

    content = message.get(content_key)
    if content is not None and not isinstance(content, (str, list)):
        raise _fail(
            f"message {message_index} content must be a string, list, or null",
            row_index,
        )
    if isinstance(content, list) and not all(isinstance(part, Mapping) for part in content):
        raise _fail(
            f"message {message_index} content blocks must all be objects",
            row_index,
        )

    # Preserve tool-call ids, names, and multimodal content fields for the
    # tokenizer's native chat template. Only the schema keys themselves are
    # replaced by the canonical role/content names.
    normalized = {
        key: value
        for key, value in message.items()
        if key not in {role_key, content_key}
    }
    normalized["role"] = role
    normalized["content"] = content
    return normalized


def _normalize_openai(row: Mapping[str, Any], row_index: int | None) -> NormalizedSFTExample:
    raw_messages = row.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise _fail("OpenAI format requires a non-empty messages list", row_index)
    role_map = {
        "system": "system",
        "developer": "system",
        "user": "user",
        "assistant": "assistant",
        "tool": "tool",
        "function": "tool",
    }
    messages = [
        _normalize_message(
            message,
            role_key="role",
            content_key="content",
            role_map=role_map,
            row_index=row_index,
            message_index=index,
        )
        for index, message in enumerate(raw_messages)
    ]
    return NormalizedSFTExample(
        messages=messages,
        tools=_normalize_tools(row.get("tools"), row_index),
        source_format="openai",
    )


def _normalize_sharegpt(row: Mapping[str, Any], row_index: int | None) -> NormalizedSFTExample:
    raw_messages = row.get("conversations")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise _fail("ShareGPT format requires a non-empty conversations list", row_index)
    role_map = {
        "system": "system",
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "function_call": "assistant",
        "observation": "tool",
        "tool": "tool",
    }
    messages = [
        _normalize_message(
            message,
            role_key="from",
            content_key="value",
            role_map=role_map,
            row_index=row_index,
            message_index=index,
        )
        for index, message in enumerate(raw_messages)
    ]
    return NormalizedSFTExample(
        messages=messages,
        tools=_normalize_tools(row.get("tools"), row_index),
        source_format="sharegpt",
    )


def _string_field(
    row: Mapping[str, Any], name: str, row_index: int | None, *, required: bool = False
) -> str:
    value = row.get(name, "")
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise _fail(f"Alpaca field {name!r} must be a string", row_index)
    if required and not value.strip():
        raise _fail(f"Alpaca format requires a non-empty {name!r} field", row_index)
    return value


def _normalize_alpaca(row: Mapping[str, Any], row_index: int | None) -> NormalizedSFTExample:
    instruction = _string_field(row, "instruction", row_index)
    query = _string_field(row, "input", row_index)
    response = _string_field(row, "output", row_index, required=True)
    if not instruction.strip() and not query.strip():
        raise _fail("Alpaca format requires instruction or input text", row_index)

    messages: list[dict[str, Any]] = []
    system = _string_field(row, "system", row_index)
    if system.strip():
        messages.append({"role": "system", "content": system})

    history = row.get("history")
    if history not in (None, []):
        if not isinstance(history, list):
            raise _fail("Alpaca history must be a list of [prompt, response] pairs", row_index)
        for history_index, pair in enumerate(history):
            if (
                not isinstance(pair, (list, tuple))
                or len(pair) != 2
                or not all(isinstance(value, str) for value in pair)
            ):
                raise _fail(
                    f"Alpaca history item {history_index} must be a [prompt, response] string pair",
                    row_index,
                )
            messages.append({"role": "user", "content": pair[0]})
            messages.append({"role": "assistant", "content": pair[1]})

    prompt = (
        instruction
        if not query.strip()
        else f"{instruction}\n{query}"
        if instruction.strip()
        else query
    )
    messages.append({"role": "user", "content": prompt})
    messages.append({"role": "assistant", "content": response})
    return NormalizedSFTExample(
        messages=messages,
        tools=_normalize_tools(row.get("tools"), row_index),
        source_format="alpaca",
    )


def detect_data_format(row: Mapping[str, Any], row_index: int | None = None) -> str:
    matches = []
    if "messages" in row:
        matches.append("openai")
    if "conversations" in row:
        matches.append("sharegpt")
    if "instruction" in row or "output" in row:
        matches.append("alpaca")
    if not matches:
        raise _fail(
            "could not detect data format; expected messages, conversations, "
            "or instruction/output columns. Pass --data-format explicitly",
            row_index,
        )
    if len(matches) != 1:
        raise _fail(
            f"ambiguous data format matched {matches}; pass --data-format explicitly",
            row_index,
        )
    return matches[0]


def normalize_sft_row(
    row: Any,
    data_format: str = "auto",
    *,
    row_index: int | None = None,
) -> NormalizedSFTExample:
    """Convert one OpenAI, ShareGPT, or Alpaca row into canonical messages."""
    validate_data_format(data_format)
    if not isinstance(row, Mapping):
        raise _fail(f"dataset row must be an object, got {type(row).__name__}", row_index)
    if data_format == "auto":
        data_format = detect_data_format(row, row_index)
    if data_format == "openai":
        return _normalize_openai(row, row_index)
    if data_format == "sharegpt":
        return _normalize_sharegpt(row, row_index)
    return _normalize_alpaca(row, row_index)
