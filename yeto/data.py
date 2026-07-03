"""Dataset preparation: chat traces -> packed causal-LM token blocks.

Designed for datasets like armand0e/claude-fable-5-claude-code: rows carry a
`messages` list ({role, content}, assistant messages may hold `tool_calls`)
plus a `tools` schema list. Conversations are rendered to text (the model's
chat template when it accepts the row, otherwise a plain fallback), tokenized
as one long stream per row, and packed into fixed-length blocks. Every sample
is a pair (input_ids, weights): per-token float loss weights that stay aligned
with the ids across block boundaries. With train_on="assistant" (default) only
assistant-message tokens (and the per-row EOS) carry weight 1.0; with
train_on="all" every token trains.

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


def _fallback_segments(messages: list[dict], tools: list | None) -> list[tuple[str, float]]:
    """The fallback rendering as (text, weight) segments, one per message
    (plus the tools preamble). Weight 1.0 marks assistant-authored segments —
    the training targets — and 0.0 everything else (context only)."""
    segments = []
    if tools:
        segments.append((f"<|system|>\nAvailable tools:\n{json.dumps(tools, indent=None)}\n", 0.0))
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
        segments.append((f"<|{role}|>\n{text}\n", 1.0 if role == "assistant" else 0.0))
    return segments


def _render_fallback(messages: list[dict], tools: list | None) -> str:
    return "".join(text for text, _ in _fallback_segments(messages, tools))


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


TRAIN_ON_CHOICES = ("assistant", "all")


def _row_tokens(tokenizer, row: dict, train_on: str = "assistant") -> tuple[list[int], list[float]]:
    """Tokenize one row into (ids, per-token loss weights).

    train_on="assistant": always the fallback rendering (chat templates don't
    expose assistant spans), tokenized one message segment at a time so the
    assistant spans are exact: assistant-segment tokens weigh 1.0, everything
    else 0.0. The per-row EOS weighs 1.0 (teaches stopping), BOS 0.0.
    train_on="all": whole-conversation rendering (chat template when
    available), all weights 1.0.
    """
    if train_on not in TRAIN_ON_CHOICES:
        raise ValueError(f"train_on must be one of {TRAIN_ON_CHOICES}, got {train_on!r}")
    messages = row.get("messages")
    if not messages:
        return [], []
    ids: list[int] = []
    weights: list[float] = []
    if train_on == "assistant":
        for text, w in _fallback_segments(messages, row.get("tools")):
            seg_ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
            ids.extend(seg_ids)
            weights.extend([w] * len(seg_ids))
    else:
        text = render_conversation(tokenizer, messages, row.get("tools"))
        ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
        weights = [1.0] * len(ids)
    if tokenizer.bos_token_id is not None:
        ids.insert(0, tokenizer.bos_token_id)
        weights.insert(0, 0.0 if train_on == "assistant" else 1.0)
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
        weights.append(1.0)
    return ids, weights


def _learner_rows(num_rows: int, learner_id: int, num_learners: int, max_rows: int | None):
    rows = list(range(learner_id, num_rows, num_learners))
    return rows[:max_rows] if max_rows is not None else rows


class StreamingPackedBlocks(IterableDataset):
    """Infinite stream of (input_ids, weights) block pairs — each a
    (seq_len,) LongTensor and its aligned (seq_len,) FloatTensor of per-token
    loss weights — tokenized on the fly.

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
        train_on: str = "assistant",
    ):
        if train_on not in TRAIN_ON_CHOICES:
            raise ValueError(f"train_on must be one of {TRAIN_ON_CHOICES}, got {train_on!r}")
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
        self.train_on = train_on

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
        buf_ids: list[int] = []
        buf_weights: list[float] = []
        while True:
            order = my_rows[:]
            rng.shuffle(order)
            for i in order:
                ids, weights = _row_tokens(self.tokenizer, ds[i], self.train_on)
                buf_ids.extend(ids)
                buf_weights.extend(weights)
                while len(buf_ids) >= self.seq_len:
                    yield (
                        torch.tensor(buf_ids[: self.seq_len], dtype=torch.long),
                        torch.tensor(buf_weights[: self.seq_len], dtype=torch.float),
                    )
                    del buf_ids[: self.seq_len]
                    del buf_weights[: self.seq_len]


@dataclass
class PackedDataset(Dataset):
    blocks: torch.Tensor  # (N, seq_len) int64
    weights: torch.Tensor  # (N, seq_len) float32 per-token loss weights

    def __len__(self) -> int:
        return self.blocks.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.blocks[idx], self.weights[idx]


def build_packed_dataset(
    dataset_name,
    tokenizer,
    learner_id: int,
    num_learners: int,
    seq_len: int,
    max_rows: int | None = None,
    split: str = "train",
    train_on: str = "assistant",
) -> PackedDataset:
    ds = load_rows(dataset_name, split)
    token_stream: list[int] = []
    weight_stream: list[float] = []
    for i in _learner_rows(len(ds), learner_id, num_learners, max_rows):
        ids, weights = _row_tokens(tokenizer, ds[i], train_on)
        token_stream.extend(ids)
        weight_stream.extend(weights)

    n_blocks = len(token_stream) // seq_len
    if n_blocks == 0:
        raise ValueError(
            f"learner {learner_id}: not enough tokens ({len(token_stream)}) for one "
            f"block of {seq_len}; use more rows or a smaller --seq-len"
        )
    blocks = torch.tensor(token_stream[: n_blocks * seq_len], dtype=torch.long).view(
        n_blocks, seq_len
    )
    weights = torch.tensor(weight_stream[: n_blocks * seq_len], dtype=torch.float).view(
        n_blocks, seq_len
    )
    return PackedDataset(blocks, weights)
