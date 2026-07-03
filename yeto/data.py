"""Dataset preparation: chat traces -> packed causal-LM token blocks.

Designed for datasets like armand0e/claude-fable-5-claude-code: rows carry a
`messages` list ({role, content}, assistant messages may hold `tool_calls`)
plus a `tools` schema list. Conversations are rendered to text (the model's
chat template when it accepts the row, otherwise a plain fallback), tokenized
as one long stream per row, and packed into fixed-length blocks.

Sharding: row i belongs to learner (i mod num_learners), giving disjoint data
shards D_m. Within a learner, streaming mode further splits rows across DDP
ranks and DataLoader workers.

Two modes:
  - StreamingPackedBlocks (default): an infinite IterableDataset that
    renders/tokenizes/packs in DataLoader worker processes, ahead of the GPU.
    Training starts immediately; tokenization overlaps compute.
  - build_packed_dataset (--tokenize preload): materialize every block up
    front; simple, deterministic epochs, pays the full tokenization bill at
    startup.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, IterableDataset


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


def load_rows(dataset_name, split: str = "train"):
    """Load the raw dataset; a pre-materialized sequence passes through
    unchanged (used by tests)."""
    if not isinstance(dataset_name, str):
        return dataset_name
    from datasets import load_dataset

    try:
        return load_dataset(dataset_name, split=split)
    except Exception:
        # Raw JSONL with heterogeneous schemas (e.g. message content that is
        # sometimes a string, sometimes content blocks) breaks arrow schema
        # inference; the Hub's parquet conversion is normalized.
        return load_dataset(dataset_name, revision="refs/convert/parquet", split=split)


def _row_tokens(tokenizer, row: dict) -> list[int]:
    messages = row.get("messages")
    if not messages:
        return []
    text = render_conversation(tokenizer, messages, row.get("tools"))
    ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
    if tokenizer.bos_token_id is not None:
        ids.insert(0, tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
    return ids


def _learner_rows(num_rows: int, learner_id: int, num_learners: int, max_rows: int | None):
    rows = list(range(learner_id, num_rows, num_learners))
    return rows[:max_rows] if max_rows is not None else rows


class StreamingPackedBlocks(IterableDataset):
    """Infinite stream of (seq_len,) token blocks, tokenized on the fly.

    Rows are split learner -> rank -> DataLoader worker, so every consumer
    tokenizes a disjoint row subset. The stream cycles forever, reshuffling
    row order each pass; termination is the training loop's job (syncer
    SHUTDOWN or --max-local-steps), which also sidesteps DDP's uneven-batch
    deadlock at epoch ends.
    """

    def __init__(
        self,
        dataset_name,
        tokenizer,
        learner_id: int,
        num_learners: int,
        seq_len: int,
        max_rows: int | None = None,
        rank: int = 0,
        world: int = 1,
        seed: int = 0,
        split: str = "train",
    ):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.learner_id = learner_id
        self.num_learners = num_learners
        self.seq_len = seq_len
        self.max_rows = max_rows
        self.rank = rank
        self.world = world
        self.seed = seed
        self.split = split

    def __iter__(self):
        ds = load_rows(self.dataset_name, self.split)
        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info else 0
        num_workers = info.num_workers if info else 1
        shard = _learner_rows(len(ds), self.learner_id, self.num_learners, self.max_rows)
        consumer = self.rank * num_workers + worker_id
        consumers = self.world * num_workers
        my_rows = shard[consumer::consumers]
        if not my_rows:
            raise ValueError(
                f"learner {self.learner_id} rank {self.rank} worker {worker_id}: "
                f"no rows left after splitting {len(shard)} across {consumers} "
                f"consumers; lower --stream-workers or use more rows"
            )
        rng = random.Random(self.seed + consumer)
        buf: list[int] = []
        while True:
            order = my_rows[:]
            rng.shuffle(order)
            for i in order:
                buf.extend(_row_tokens(self.tokenizer, ds[i]))
                while len(buf) >= self.seq_len:
                    yield torch.tensor(buf[: self.seq_len], dtype=torch.long)
                    del buf[: self.seq_len]


@dataclass
class PackedDataset(Dataset):
    blocks: torch.Tensor  # (N, seq_len) int64

    def __len__(self) -> int:
        return self.blocks.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.blocks[idx]


def build_packed_dataset(
    dataset_name,
    tokenizer,
    learner_id: int,
    num_learners: int,
    seq_len: int,
    max_rows: int | None = None,
    split: str = "train",
) -> PackedDataset:
    ds = load_rows(dataset_name, split)
    token_stream: list[int] = []
    for i in _learner_rows(len(ds), learner_id, num_learners, max_rows):
        token_stream.extend(_row_tokens(tokenizer, ds[i]))

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
