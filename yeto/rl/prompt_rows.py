"""The prompt and label of one RL ``--data`` row, the same on every backend.

``--rl-prompt-column`` names the prompt column (default: ``messages``, else
``prompt``, else ``input``); a string prompt becomes one user message.
``--rl-label-column`` names the label column (default: ``label``). A named
column a row does not have is an error naming the flag, never a silent
fallback to another column. The Miles island writes its prompt file with these
rules (``adapters/miles/island_entry.prepare_prompt_data``) and the verl island
its parquet files (``adapters/verl/island_entry.prepare_data``).
"""

from __future__ import annotations

from typing import Any


def column(row: dict[str, Any], name: str, flag: str) -> Any:
    """A user-named column; a row without it is an error naming the flag,
    never a silent fallback to another column."""
    if name not in row:
        raise ValueError(f"RL rows have no column {name!r} ({flag})")
    return row[name]


def messages(row: dict[str, Any], prompt_column: str | None = None) -> list[dict[str, Any]]:
    """The row's prompt as a chat message list."""
    if prompt_column is not None:
        value = column(row, prompt_column, "--rl-prompt-column")
    else:
        value = row.get("messages", row.get("prompt", row.get("input")))
    if isinstance(value, str):
        value = [{"role": "user", "content": value}]
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, dict) for item in value)
    ):
        raise ValueError("RL rows must contain messages or a string prompt/input")
    return value


def label(row: dict[str, Any], label_column: str | None = None) -> Any:
    """The row's label (``None`` when the default ``label`` column is absent)."""
    return column(row, label_column, "--rl-label-column") if label_column is not None else row.get("label")
