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


def _message_text(msg: dict) -> str:
    content = msg.get("content") or ""
    if isinstance(content, list):  # multi-part content blocks
        content = "\n".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return str(content)


def _message_training_parts(msg: dict) -> list[str]:
    parts = []
    reasoning = msg.get("reasoning_content")
    if msg.get("role") == "assistant" and isinstance(reasoning, str) and reasoning.strip():
        parts.append(reasoning.strip())
    content = _message_text(msg)
    if content:
        parts.append(content)
    if msg.get("tool_calls"):
        parts.append(json.dumps(msg["tool_calls"]))
    return parts


def _fallback_segments(messages: list[dict], tools: list | None) -> list[tuple[str, float]]:
    """The fallback rendering as (text, weight) segments, one per message
    (plus the tools preamble). Weight 1.0 marks assistant-authored segments —
    the training targets — and 0.0 everything else (context only)."""
    segments = []
    if tools:
        segments.append((f"<|system|>\nAvailable tools:\n{json.dumps(tools, indent=None)}\n", 0.0))
    for msg in messages:
        role = msg.get("role", "user")
        content = _message_text(msg)
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


def _as_list(value) -> list:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _ensure_weighted_terminal_eos(tokenizer, ids: list[int], weights: list[float]) -> None:
    """Teach stopping without duplicating an EOS already emitted by a template."""
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        return
    if eos_token_id in ids:
        eos_index = len(ids) - 1 - ids[::-1].index(eos_token_id)
        try:
            trailing_text = tokenizer.decode(ids[eos_index + 1 :], skip_special_tokens=False)
        except Exception:
            trailing_text = None
        if eos_index == len(ids) - 1 or (trailing_text is not None and not trailing_text.strip()):
            weights[eos_index] = 1.0
            return
    ids.append(eos_token_id)
    weights.append(1.0)


def _chat_template_assistant_tokens(
    tokenizer, messages: list[dict], tools: list | None
) -> tuple[list[int], list[float]] | None:
    """Render with the tokenizer's native chat template and mask assistant text.

    Prefer tokenizer-provided assistant masks when available. If the tokenizer
    can render a native template but does not expose masks, derive assistant
    spans from the rendered string using token offset mappings. This keeps
    model-native role/control tokens in the training sequence instead of
    silently falling back to synthetic ``<|assistant|>`` delimiters.
    """
    if not getattr(tokenizer, "chat_template", None):
        return None

    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
    except Exception:
        rendered = None
    if isinstance(rendered, dict):
        input_ids = rendered.get("input_ids")
        assistant_mask = rendered.get("assistant_masks")
        if assistant_mask is None:
            assistant_mask = rendered.get("assistant_tokens_mask")
        if input_ids is not None and assistant_mask is not None:
            ids = _as_list(input_ids)
            mask = _as_list(assistant_mask)
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            if mask and isinstance(mask[0], list):
                mask = mask[0]
            if len(ids) == len(mask):
                has_assistant_content = any(
                    (msg.get("role") == "assistant")
                    and bool(_message_text(msg) or msg.get("tool_calls"))
                    for msg in messages
                )
                if not has_assistant_content or any(mask):
                    weights = [float(w) for w in mask]
                    _ensure_weighted_terminal_eos(tokenizer, ids, weights)
                    return ids, weights

    try:
        text = tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        return None

    spans: list[tuple[int, int]] = []
    cursor = 0
    for msg in messages:
        parts = _message_training_parts(msg)
        for part in parts:
            start = text.find(part, cursor)
            if start < 0:
                return None
            end = start + len(part)
            if msg.get("role") == "assistant":
                spans.append((start, end))
            cursor = end
    if any(msg.get("role") == "assistant" and _message_training_parts(msg) for msg in messages) and not spans:
        return None

    try:
        tokenized = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except Exception:
        return None
    ids = _as_list(tokenized.get("input_ids", []))
    offsets = _as_list(tokenized.get("offset_mapping", []))
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(offsets[0][0], tuple):
        offsets = offsets[0]
    if len(ids) != len(offsets):
        return None

    weights = []
    for offset in offsets:
        start, end = int(offset[0]), int(offset[1])
        weights.append(float(any(start < span_end and end > span_start for span_start, span_end in spans)))
    _ensure_weighted_terminal_eos(tokenizer, ids, weights)
    return ids, weights


def load_rows(dataset_name, split: str = "train"):
    """Load the raw dataset; a pre-materialized sequence passes through
    unchanged (used by tests). Accepts an HF dataset id or a local path —
    cloud sources arrive as local paths via the launcher's sky file mounts
    (see yeto/datasource.py)."""
    if not isinstance(dataset_name, str):
        return dataset_name
    import os

    path = os.path.expanduser(dataset_name)
    if os.path.exists(path):
        return _load_local(path, split)
    from datasets import load_dataset

    try:
        return load_dataset(dataset_name, split=split)
    except Exception:
        # Raw JSONL with heterogeneous schemas (e.g. message content that is
        # sometimes a string, sometimes content blocks) breaks arrow schema
        # inference; the Hub's parquet conversion is normalized.
        return load_dataset(dataset_name, revision="refs/convert/parquet", split=split)


_LOCAL_FORMATS = {".jsonl": "json", ".json": "json", ".parquet": "parquet"}


def _load_local(path: str, split: str):
    """A save_to_disk directory, a directory of jsonl/json/parquet files, or
    a single such file."""
    import glob
    import os

    from datasets import load_dataset, load_from_disk

    if os.path.isdir(path):
        if os.path.exists(os.path.join(path, "dataset_info.json")) or os.path.exists(
            os.path.join(path, "dataset_dict.json")
        ):
            ds = load_from_disk(path)
            if hasattr(ds, "keys"):  # DatasetDict: pick the split, else the only one
                return ds[split] if split in ds else next(iter(ds.values()))
            return ds
        for ext, fmt in _LOCAL_FORMATS.items():
            files = sorted(glob.glob(os.path.join(path, f"*{ext}")))
            if files:
                return load_dataset(fmt, data_files=files, split="train")
        raise ValueError(f"{path}: no dataset_info.json and no *.jsonl/*.json/*.parquet files")
    ext = os.path.splitext(path)[1]
    if ext not in _LOCAL_FORMATS:
        raise ValueError(f"{path}: unsupported data file type {ext!r} (use jsonl/json/parquet)")
    return load_dataset(_LOCAL_FORMATS[ext], data_files=[path], split="train")


TRAIN_ON_CHOICES = ("assistant", "all")


def _row_tokens(tokenizer, row: dict, train_on: str = "assistant") -> tuple[list[int], list[float]]:
    """Tokenize one row into (ids, per-token loss weights).

    train_on="assistant": native chat template with assistant-token masking
    when available. If a tokenizer cannot expose or derive assistant spans,
    fall back to Yeto's synthetic segment rendering. The per-row EOS weighs
    1.0 (teaches stopping), BOS 0.0.
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
        native = _chat_template_assistant_tokens(tokenizer, messages, row.get("tools"))
        if native is not None:
            return native
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


def _append_target_blocks(
    out_ids: list[list[int]],
    out_weights: list[list[float]],
    ids: list[int],
    weights: list[float],
    seq_len: int,
    carry_ids: list[int],
    carry_weights: list[float],
) -> None:
    """Append assistant-training blocks while skipping pure-context windows.

    Native chat rendering can produce very long user/tool prefixes before the
    first assistant target. Emitting every fixed window from that stream makes
    assistant-only SFT spend most optimizer steps on zero-loss blocks. Keep
    full windows that contain target tokens, and pack target-bearing remainders
    together across rows so short useful examples are not dropped.
    """
    for start in range(0, len(ids), seq_len):
        chunk_ids = ids[start : start + seq_len]
        chunk_weights = weights[start : start + seq_len]
        if not chunk_ids or not any(chunk_weights):
            continue
        if len(chunk_ids) == seq_len:
            out_ids.append(chunk_ids)
            out_weights.append(chunk_weights)
            continue
        carry_ids.extend(chunk_ids)
        carry_weights.extend(chunk_weights)
        while len(carry_ids) >= seq_len:
            out_ids.append(carry_ids[:seq_len])
            out_weights.append(carry_weights[:seq_len])
            del carry_ids[:seq_len]
            del carry_weights[:seq_len]


def _target_packed_blocks(rows, tokenizer, row_ids: list[int], seq_len: int, train_on: str):
    block_ids: list[list[int]] = []
    block_weights: list[list[float]] = []
    carry_ids: list[int] = []
    carry_weights: list[float] = []
    for i in row_ids:
        ids, weights = _row_tokens(tokenizer, rows[i], train_on)
        if train_on == "assistant":
            _append_target_blocks(
                block_ids, block_weights, ids, weights, seq_len, carry_ids, carry_weights
            )
        else:
            carry_ids.extend(ids)
            carry_weights.extend(weights)
            while len(carry_ids) >= seq_len:
                block_ids.append(carry_ids[:seq_len])
                block_weights.append(carry_weights[:seq_len])
                del carry_ids[:seq_len]
                del carry_weights[:seq_len]
    return block_ids, block_weights


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
        while True:
            order = my_rows[:]
            rng.shuffle(order)
            block_ids, block_weights = _target_packed_blocks(
                ds, self.tokenizer, order, self.seq_len, self.train_on
            )
            if not block_ids:
                raise ValueError(
                    f"learner {self.learner_id} rank {self.rank} worker {worker_id}: "
                    f"no trainable blocks after target-aware packing; use more rows, "
                    f"a smaller --seq-len, or --train-on all"
                )
            paired = list(zip(block_ids, block_weights))
            rng.shuffle(paired)
            for ids, weights in paired:
                yield (
                    torch.tensor(ids, dtype=torch.long),
                    torch.tensor(weights, dtype=torch.float),
                )


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
    block_ids, block_weights = _target_packed_blocks(
        ds,
        tokenizer,
        _learner_rows(len(ds), learner_id, num_learners, max_rows),
        seq_len,
        train_on,
    )

    if not block_ids:
        raise ValueError(
            f"learner {learner_id}: no trainable blocks of {seq_len} tokens; "
            f"use more rows, a smaller --seq-len, or --train-on all"
        )
    blocks = torch.tensor(block_ids, dtype=torch.long)
    weights = torch.tensor(block_weights, dtype=torch.float)
    return PackedDataset(blocks, weights)
