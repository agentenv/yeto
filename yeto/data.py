"""Dataset preparation: chat traces -> packed causal-LM token blocks.

Designed for datasets like armand0e/claude-fable-5-claude-code: rows carry a
`messages` list ({role, content}, assistant messages may hold `tool_calls`)
plus a `tools` schema list. Conversations are tokenized as one long stream per
row and packed into fixed-length blocks. Every sample is a pair (input_ids,
weights): per-token float loss weights that stay aligned with the ids across
block boundaries. With train_on="assistant" (default), the model tokenizer's
native chat template and assistant mask define the training targets. Templates
without exact native assistant-mask support fail clearly; the old synthetic
role format is available only through assistant_mask_mode="legacy". With
train_on="all" every token trains and retains the existing rendering behavior.

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
import re
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset, IterableDataset


def _fallback_segments(messages: list[dict], tools: list | None) -> list[tuple[str, float]]:
    """The legacy rendering as (text, weight) segments, one per message
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
    """Render the unchanged train_on="all" representation.

    That mode historically prefers the tokenizer's model-native template but
    accepts the synthetic formatter for tokenizers/templates that reject a
    row. Assistant-only training does not call this function: it uses the
    strict token-level native-mask path below, unless legacy mode is explicit.
    """
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages, tools=tools, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            pass  # tool_calls formats vary; fall back to the plain rendering
    return _render_fallback(messages, tools)


def load_rows(dataset_name, split: str = "train", revision: str | None = None):
    """Load the raw dataset; a pre-materialized sequence passes through
    unchanged (used by tests). Accepts an HF dataset id or a local path —
    cloud sources arrive as local paths via the launcher's sky file mounts
    (see yeto/datasource.py)."""
    if not isinstance(dataset_name, str):
        return dataset_name
    import os

    path = os.path.expanduser(dataset_name)
    if os.path.exists(path):
        if revision is not None:
            raise ValueError(
                f"dataset revision {revision!r} cannot be used with local path {path!r}"
            )
        return _load_local(path, split)
    from datasets import load_dataset

    import inspect

    kwargs = {"split": split}
    # datasets 3.x supports executable dataset scripts behind this flag;
    # datasets 4+ removed scripts and the argument itself. Do not pass it as
    # an arbitrary BuilderConfig key on versions where it is no longer a
    # declared security control.
    if "trust_remote_code" in inspect.signature(load_dataset).parameters:
        kwargs["trust_remote_code"] = False
    if revision is not None:
        from .provenance import is_immutable_commit

        if not is_immutable_commit(revision):
            raise ValueError(
                f"remote dataset revision {revision!r} is not an immutable commit"
            )
        kwargs["revision"] = revision
    try:
        return load_dataset(dataset_name, **kwargs)
    except Exception as exc:
        if revision is not None:
            # refs/convert/parquet is a separately moving Hub ref.  Falling
            # back to it after pinning the source dataset would silently load
            # different data under the recorded commit identity.
            raise RuntimeError(
                f"failed to load dataset {dataset_name!r} at pinned revision "
                f"{revision}; refusing the moving refs/convert/parquet fallback. "
                "Materialize a normalized local dataset if schema inference fails."
            ) from exc
        # Raw JSONL with heterogeneous schemas (e.g. message content that is
        # sometimes a string, sometimes content blocks) breaks arrow schema
        # inference; the Hub's parquet conversion is normalized.
        fallback_kwargs = dict(kwargs)
        fallback_kwargs["revision"] = "refs/convert/parquet"
        return load_dataset(dataset_name, **fallback_kwargs)


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
ASSISTANT_MASK_MODES = ("native", "legacy")
_GENERATION_TAG = re.compile(r"\{%-?\s*generation\s*-?%\}")


class ExactAssistantMaskError(ValueError):
    """The tokenizer cannot provide an exact model-native assistant mask."""


def _tokenizer_label(tokenizer) -> str:
    return str(getattr(tokenizer, "name_or_path", None) or type(tokenizer).__name__)


def _legacy_mask_hint() -> str:
    return (
        "Use a tokenizer whose selected chat template contains `{% generation %}`, "
        "or pass `--assistant-mask-mode legacy` to explicitly use Yeto's "
        "synthetic <|role|> compatibility format."
    )


def _selected_chat_template(tokenizer, tools: list | None) -> str:
    """Return the exact template that apply_chat_template will select.

    Transformers tokenizers may store multiple templates (for example default
    and tool-use variants). get_chat_template owns that selection, so inspect
    its result rather than guessing from tokenizer.chat_template.
    """
    label = _tokenizer_label(tokenizer)
    getter = getattr(tokenizer, "get_chat_template", None)
    try:
        if callable(getter):
            template = getter(chat_template=None, tools=tools)
        else:
            template = getattr(tokenizer, "chat_template", None)
    except Exception as exc:
        raise ExactAssistantMaskError(
            f"tokenizer {label!r} could not select its native chat template: {exc}. "
            + _legacy_mask_hint()
        ) from exc

    if not isinstance(template, str) or not template:
        raise ExactAssistantMaskError(
            f"tokenizer {label!r} has no native chat template, so an exact "
            f"assistant loss mask cannot be produced. {_legacy_mask_hint()}"
        )
    if _GENERATION_TAG.search(template) is None:
        raise ExactAssistantMaskError(
            f"tokenizer {label!r}'s selected chat template does not contain "
            f"`{{% generation %}}`, so Transformers cannot produce an exact "
            f"assistant loss mask. {_legacy_mask_hint()}"
        )
    return template


def _flat_list(value: Any, field: str, tokenizer) -> list:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ExactAssistantMaskError(
            f"tokenizer {_tokenizer_label(tokenizer)!r} returned {field} as "
            f"{type(value).__name__}, expected a flat list. {_legacy_mask_hint()}"
        )
    value = list(value)
    if value and isinstance(value[0], (list, tuple)):
        raise ExactAssistantMaskError(
            f"tokenizer {_tokenizer_label(tokenizer)!r} returned batched {field} "
            "for one conversation, so the assistant mask cannot be aligned exactly. "
            + _legacy_mask_hint()
        )
    return value


def _assistant_has_payload(messages: list[dict]) -> bool:
    return any(
        msg.get("role") == "assistant"
        and (msg.get("content") not in (None, "", []) or msg.get("tool_calls"))
        for msg in messages
    )


def _native_assistant_tokens(
    tokenizer, messages: list[dict], tools: list | None
) -> tuple[list[int], list[float]]:
    """Tokenize once through the model template and return its exact mask.

    No BOS/EOS tokens are injected here: a model-native chat template owns all
    control tokens, and its `{% generation %}` blocks decide whether those
    tokens are assistant targets. Altering either sequence after templating
    would make the returned mask inexact.
    """
    _selected_chat_template(tokenizer, tools)
    label = _tokenizer_label(tokenizer)
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
    except Exception as exc:
        raise ExactAssistantMaskError(
            f"tokenizer {label!r} failed to tokenize this conversation with an "
            f"exact native assistant mask: {exc}. {_legacy_mask_hint()}"
        ) from exc

    if not hasattr(encoded, "get"):
        raise ExactAssistantMaskError(
            f"tokenizer {label!r} returned {type(encoded).__name__} instead of a "
            "mapping for return_dict=True; exact assistant masking is unavailable. "
            + _legacy_mask_hint()
        )
    if encoded.get("input_ids") is None:
        raise ExactAssistantMaskError(
            f"tokenizer {label!r} did not return input_ids from apply_chat_template. "
            + _legacy_mask_hint()
        )
    if encoded.get("assistant_masks") is None:
        raise ExactAssistantMaskError(
            f"tokenizer {label!r} did not return assistant_masks even though "
            "return_assistant_tokens_mask=True. " + _legacy_mask_hint()
        )

    ids = _flat_list(encoded["input_ids"], "input_ids", tokenizer)
    native_mask = _flat_list(encoded["assistant_masks"], "assistant_masks", tokenizer)
    if len(ids) != len(native_mask):
        raise ExactAssistantMaskError(
            f"tokenizer {label!r} returned {len(ids)} input ids but "
            f"{len(native_mask)} assistant-mask values. {_legacy_mask_hint()}"
        )
    if any(value not in (0, 1, False, True) for value in native_mask):
        raise ExactAssistantMaskError(
            f"tokenizer {label!r} returned a non-binary assistant mask. "
            + _legacy_mask_hint()
        )
    if _assistant_has_payload(messages) and not any(native_mask):
        raise ExactAssistantMaskError(
            f"tokenizer {label!r} returned an all-zero assistant mask for a "
            "conversation containing assistant output. " + _legacy_mask_hint()
        )
    return [int(token_id) for token_id in ids], [float(value) for value in native_mask]


def _row_tokens(
    tokenizer,
    row: dict,
    train_on: str = "assistant",
    assistant_mask_mode: str = "native",
) -> tuple[list[int], list[float]]:
    """Tokenize one row into (ids, per-token loss weights).

    train_on="assistant", assistant_mask_mode="native": tokenize the whole row
    once with apply_chat_template and use Transformers' assistant_masks. This
    requires a selected template with `{% generation %}` and never falls back.
    assistant_mask_mode="legacy" explicitly restores the old synthetic
    per-message rendering, including BOS weight 0 and closing EOS weight 1.
    train_on="all": whole-conversation rendering (chat template when
    available), all weights 1.0, with its existing BOS/EOS behavior unchanged.
    """
    if train_on not in TRAIN_ON_CHOICES:
        raise ValueError(f"train_on must be one of {TRAIN_ON_CHOICES}, got {train_on!r}")
    if assistant_mask_mode not in ASSISTANT_MASK_MODES:
        raise ValueError(
            f"assistant_mask_mode must be one of {ASSISTANT_MASK_MODES}, "
            f"got {assistant_mask_mode!r}"
        )
    messages = row.get("messages")
    if not messages:
        return [], []
    ids: list[int] = []
    weights: list[float] = []
    if train_on == "assistant":
        if assistant_mask_mode == "native":
            return _native_assistant_tokens(tokenizer, messages, row.get("tools"))
        for text, weight in _fallback_segments(messages, row.get("tools")):
            segment_ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
            ids.extend(segment_ids)
            weights.extend([weight] * len(segment_ids))
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
    """Append assistant-training blocks while skipping pure-context windows."""
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


def _target_packed_blocks(
    rows,
    tokenizer,
    row_ids: list[int],
    seq_len: int,
    train_on: str,
    assistant_mask_mode: str,
):
    block_ids: list[list[int]] = []
    block_weights: list[list[float]] = []
    carry_ids: list[int] = []
    carry_weights: list[float] = []
    for index in row_ids:
        ids, weights = _row_tokens(
            tokenizer,
            rows[index],
            train_on,
            assistant_mask_mode,
        )
        if train_on == "assistant":
            _append_target_blocks(
                block_ids,
                block_weights,
                ids,
                weights,
                seq_len,
                carry_ids,
                carry_weights,
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
        assistant_mask_mode: str = "native",
        revision: str | None = None,
    ):
        if train_on not in TRAIN_ON_CHOICES:
            raise ValueError(f"train_on must be one of {TRAIN_ON_CHOICES}, got {train_on!r}")
        if assistant_mask_mode not in ASSISTANT_MASK_MODES:
            raise ValueError(
                f"assistant_mask_mode must be one of {ASSISTANT_MASK_MODES}, "
                f"got {assistant_mask_mode!r}"
            )
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
        self.assistant_mask_mode = assistant_mask_mode
        self.revision = revision

    def __iter__(self):
        ds = load_rows(self.dataset_name, self.split, self.revision)
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
                ds,
                self.tokenizer,
                order,
                self.seq_len,
                self.train_on,
                self.assistant_mask_mode,
            )
            if not block_ids:
                raise ValueError(
                    f"learner {self.learner_id} rank {self.rank} worker {worker_id}: "
                    "no trainable blocks after target-aware packing; use more rows, "
                    "a smaller --seq-len, or --train-on all"
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
    assistant_mask_mode: str = "native",
    revision: str | None = None,
) -> PackedDataset:
    ds = load_rows(dataset_name, split, revision)
    block_ids, block_weights = _target_packed_blocks(
        ds,
        tokenizer,
        _learner_rows(len(ds), learner_id, num_learners, max_rows),
        seq_len,
        train_on,
        assistant_mask_mode,
    )
    if not block_ids:
        raise ValueError(
            f"learner {learner_id}: no trainable blocks of {seq_len} tokens; "
            "use more rows, a smaller --seq-len, or --train-on all"
        )
    blocks = torch.tensor(block_ids, dtype=torch.long)
    weights = torch.tensor(block_weights, dtype=torch.float)
    return PackedDataset(blocks, weights)
