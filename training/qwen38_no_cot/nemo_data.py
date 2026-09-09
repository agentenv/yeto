"""Pretokenized dataset adapter for NeMo Automodel's unshifted-loss API.

Stored rows use HF-style unshifted labels. NeMo's loss has ``shift=False``;
this adapter alone shifts labels once, before any context-parallel sharding.
It never renders, tokenizes, packs, samples with replacement, or truncates.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def shift_next_token_row(row: dict, *, seq_len: int = 262144) -> dict:
    ids, labels = row["input_ids"], row["labels"]
    if not isinstance(ids, list) or not 2 <= len(ids) <= seq_len or len(labels) != len(ids):
        raise ValueError("Expected a nonempty pretruncated row with aligned HF labels")
    if row.get("causal_shift_applied") or row.get("labels_are_shifted"):
        raise ValueError("Input row labels have already been shifted")
    if labels[0] != -100:
        raise ValueError("First canonical label must be masked")
    if any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("Invalid token IDs")
    if any(type(label) is not int or label not in (-100, token) for token, label in zip(ids, labels)):
        raise ValueError("Canonical labels must be -100 or the token at the same position")
    if not any(label != -100 for label in labels[1:]):
        raise ValueError("The row has no causal training targets")
    if row.get("attention_mask", [1] * len(ids)) != [1] * len(ids):
        raise ValueError("Stored dataset rows must be unpadded")
    # Retain all N input tokens so a 262144-token example exercises the full
    # requested context. The final token has no available future label.
    return {"input_ids": list(ids), "labels": labels[1:] + [-100],
            "attention_mask": [1] * len(ids)}


def _file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_qualified_index(manifest_path, shards, *, split, seq_len, require_provenance,
                          index_path, index_sha256):
    """Use audited offsets only after checking complete immutable byte coverage.

    The index is created by the original qualified validator, with a separately
    supplied digest. Whole shards are rehashed at each open, and every actual
    row is rehashed and structurally validated by the unchanged __getitem__.
    """
    if not index_path or not isinstance(index_sha256, str) or len(index_sha256) != 64:
        raise ValueError("Offset index requires its explicit SHA256 identity")
    # This index is qualified for one exact renderer, just like its original
    # GPU receipts. A renderer change requires rebuilding/requalifying the index.
    folder = Path(__file__).parent
    expected_renderers = {
        folder / "render.py": "3e5ef75c849acb4158f6c873e333c16af3c6fe1ea452215990be16ab91ace18a",
        folder / "upstream_manager/render.py": "e2eacc680cbea4343458ad76d57ad1d36e84b2a324a1b8cc9d90692c5ddab1e8",
    }
    if any(_file_digest(path) != expected for path, expected in expected_renderers.items()):
        raise ValueError("Offset index renderer provenance differs from the qualified source")
    raw = Path(index_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != index_sha256:
        raise ValueError("Offset index identity changed")
    index = json.loads(raw)
    if (index.get("schema") != "qualified_exact_token_index/v1"
            or index.get("manifest_sha256") != _file_digest(manifest_path)
            or index.get("validator_source_sha256") != "9b0e75afcf0fc6a2efe81494cd56fc70a27f813348a125953e5ee1a154766367"
            or index.get("seq_len") != seq_len or index.get("require_provenance") is not True
            or require_provenance is not True):
        raise ValueError("Offset index does not match the qualified dataset audit")
    cached = index.get("splits", {}).get(split)
    if not isinstance(cached, list) or len(cached) != len(shards):
        raise ValueError("Offset index shard coverage differs from manifest")
    refs, seen = [], set()
    for entry, indexed in zip(shards, cached):
        path = Path(entry["path"]).resolve()
        if path in seen or str(path) != indexed.get("path"):
            raise ValueError("Offset index has duplicate, reordered or mismatched shards")
        seen.add(path)
        if (not entry.get("sha256") or indexed.get("sha256") != entry["sha256"]
                or _file_digest(path) != entry["sha256"]):
            raise ValueError("Training shard identity changed")
        size = path.stat().st_size
        if indexed.get("size_bytes") != size:
            raise ValueError("Offset index size differs from shard")
        position = 0
        rows = indexed.get("rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError("Offset index has no row coverage")
        for row in rows:
            if not isinstance(row, list) or len(row) != 4:
                raise ValueError("Invalid offset index row")
            offset, length, row_hash, tokens = row
            if (type(offset) is not int or offset != position or type(length) is not int or length < 1
                    or type(tokens) is not int or not 2 <= tokens <= seq_len
                    or not isinstance(row_hash, str) or len(row_hash) != 64):
                raise ValueError("Offset index rows must cover the shard exactly in source order")
            try:
                row_digest = bytes.fromhex(row_hash)
            except ValueError as exc:
                raise ValueError("Invalid row digest") from exc
            if len(row_digest) != 32:
                raise ValueError("Invalid row digest")
            position += length
            if position > size:
                raise ValueError("Offset index extends beyond the shard")
            refs.append((path, offset, length, row_digest, tokens))
        if position != size:
            raise ValueError("Offset index omitted source bytes")
    return refs


class ExactTokenDataset:
    """JSONL map dataset; token arrays are loaded lazily from byte offsets.

    ``path_or_dataset`` may be one JSONL shard, a list of JSONL shards, or a
    manifest with ``splits[split]`` shard entries containing path and sha256.
    Whole-shard identities and row hashes are validated without logging text.
    """

    def __init__(self, path_or_dataset, *, split="train", seq_len=262144, tokenizer=None,
                 max_input_tokens=None, order="source", max_samples=None, require_provenance=False,
                 index_path=None, index_sha256=None):
        del tokenizer
        self.seq_len = seq_len
        self.refs = []
        if isinstance(path_or_dataset, (str, Path)):
            path = Path(path_or_dataset)
            if path.suffix == ".json":
                manifest = json.loads(path.read_text())
                shards = [{**entry, "path": str((path.parent / entry["path"]).resolve())}
                          for entry in manifest["splits"][split]]
            else:
                shards = [{"path": str(path)}]
        else:
            shards = [{"path": str(path)} for path in path_or_dataset]
        if index_path is not None or index_sha256 is not None:
            if not isinstance(path_or_dataset, (str, Path)) or Path(path_or_dataset).suffix != ".json":
                raise ValueError("An offset index requires the original JSON manifest")
            self.refs = _load_qualified_index(path_or_dataset, shards, split=split, seq_len=seq_len,
                require_provenance=require_provenance, index_path=index_path, index_sha256=index_sha256)
            if max_input_tokens is not None:
                self.refs = [ref for ref in self.refs if ref[4] <= max_input_tokens]
            shards = []
        seen = set()
        for entry in shards:
            path = Path(entry["path"]).resolve()
            if path in seen:
                raise ValueError("Duplicate shard path would duplicate training rows")
            seen.add(path)
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while True:
                    offset = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    digest.update(line)
                    if not line.strip():
                        raise ValueError("Blank lines are not valid training rows")
                    row = json.loads(line)
                    shift_next_token_row(row, seq_len=self.seq_len)
                    if require_provenance:
                        identity = row.get("metadata", {}).get("renderer_identity", {})
                        folder = Path(__file__).parent
                        expected = {
                            "mask_policy": "assistant_content_and_eos_only_no_cot_no_headers_v1",
                            "sequence_policy": "first_262144_tokens_drop_remainder/v1",
                            "max_sequence_length": 262144,
                            "prefix_adapter_sha256": hashlib.sha256((folder / "render.py").read_bytes()).hexdigest(),
                            "renderer_sha256": hashlib.sha256((folder / "upstream_manager/render.py").read_bytes()).hexdigest(),
                        }
                        if any(identity.get(key) != value for key, value in expected.items()):
                            raise ValueError("Training row renderer/mask provenance is missing or differs")
                        if not row.get("group_id") or row.get("source") not in ("trace", "replay"):
                            raise ValueError("Missing source or split-group provenance")
                    if max_input_tokens is None or len(row["input_ids"]) <= max_input_tokens:
                        self.refs.append((path, offset, len(line), hashlib.sha256(line).digest(), len(row["input_ids"])))
            if entry.get("sha256") and digest.hexdigest() != entry["sha256"]:
                raise ValueError("Training shard identity changed")
        if order == "longest_first":
            self.refs.sort(key=lambda ref: -ref[4])
        elif order != "source":
            raise ValueError("Unsupported dataset ordering")
        if max_samples is not None:
            if type(max_samples) is not int or max_samples < 1:
                raise ValueError("max_samples must be positive")
            self.refs = self.refs[:max_samples]
        if not self.refs:
            raise ValueError("Empty training dataset")

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, index):
        path, offset, length, expected, _ = self.refs[index]
        with path.open("rb") as stream:
            stream.seek(offset)
            line = stream.read(length)
        if hashlib.sha256(line).digest() != expected:
            raise ValueError("Training row changed after dataset audit")
        return shift_next_token_row(json.loads(line), seq_len=self.seq_len)


def collate_exact(batch, *, processor=None, pad_token_id=248044, pad_to_multiple_of=16):
    """Pad only to the longest row's CP-friendly length; pad labels never train."""
    import torch

    # NeMo's VLM loader always binds its processor. These rows are already
    # rendered and tokenized; invoking it would destroy the audited masks.
    del processor

    if not batch or type(pad_to_multiple_of) is not int or pad_to_multiple_of < 1:
        raise ValueError("Invalid batch or padding multiple")
    length = max(len(row["input_ids"]) for row in batch)
    length = ((length + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
    if length > 262144:
        raise ValueError("Padding would exceed the configured context")
    result = {}
    for key, fill in (("input_ids", pad_token_id), ("labels", -100), ("attention_mask", 0)):
        result[key] = torch.tensor([row[key] + [fill] * (length - len(row[key])) for row in batch],
                                   dtype=torch.long)
    # Qwen's CP-aware model owns positional expansion and sharding.
    return result
