"""Dataset preparation: chat traces -> packed causal-LM token blocks.

Designed for datasets like armand0e/claude-fable-5-claude-code: rows carry a
`messages` list ({role, content}, assistant messages may hold `tool_calls`)
plus a `tools` schema list. Conversations are rendered to text (the model's
chat template when it accepts the row, otherwise a plain fallback), tokenized
as one long stream per row, and packed into fixed-length blocks.

Sharding: row i belongs to learner (i mod num_learners), giving disjoint data
shards D_m.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


def _render_fallback(messages: list[dict], tools: list | None) -> str:
    parts = []
    if tools:
        parts.append(f"<|system|>\nAvailable tools:\n{json.dumps(tools, indent=None)}\n")
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content") or ""
        if isinstance(content, list):  # multi-part content blocks
            content = "\n".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in content
            )
        text = content
        if msg.get("tool_calls"):
            calls = json.dumps(msg["tool_calls"])
            text = f"{content}\n<|tool_calls|>{calls}"
        parts.append(f"<|{role}|>\n{text}\n")
    return "".join(parts)


def render_conversation(tokenizer, messages: list[dict], tools: list | None) -> str:
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages, tools=tools, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            pass  # tool_calls formats vary; fall back to the plain rendering
    return _render_fallback(messages, tools)


@dataclass
class PackedDataset(Dataset):
    blocks: torch.Tensor  # (N, seq_len) int64

    def __len__(self) -> int:
        return self.blocks.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.blocks[idx]


def build_packed_dataset(
    dataset_name: str,
    tokenizer,
    learner_id: int,
    num_learners: int,
    seq_len: int,
    max_rows: int | None = None,
    split: str = "train",
) -> PackedDataset:
    from datasets import load_dataset

    try:
        ds = load_dataset(dataset_name, split=split)
    except Exception:
        # Raw JSONL with heterogeneous schemas (e.g. message content that is
        # sometimes a string, sometimes content blocks) breaks arrow schema
        # inference; the Hub's parquet conversion is normalized.
        ds = load_dataset(dataset_name, revision="refs/convert/parquet", split=split)
    token_stream: list[int] = []
    rows_used = 0
    for i in range(len(ds)):
        if i % num_learners != learner_id:
            continue
        if max_rows is not None and rows_used >= max_rows:
            break
        row = ds[i]
        messages = row.get("messages")
        if not messages:
            continue
        text = render_conversation(tokenizer, messages, row.get("tools"))
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if tokenizer.bos_token_id is not None:
            token_stream.append(tokenizer.bos_token_id)
        token_stream.extend(ids)
        if tokenizer.eos_token_id is not None:
            token_stream.append(tokenizer.eos_token_id)
        rows_used += 1

    n_blocks = len(token_stream) // seq_len
    if n_blocks == 0:
        raise ValueError(
            f"learner {learner_id}: not enough tokens ({len(token_stream)}) for one "
            f"block of {seq_len}; use more rows or a smaller --seq-len"
        )
    blocks = torch.tensor(token_stream[: n_blocks * seq_len], dtype=torch.long).view(
        n_blocks, seq_len
    )
    return PackedDataset(blocks)
