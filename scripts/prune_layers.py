#!/usr/bin/env python3
"""Cut the first N decoder layers out of a Hugging Face safetensors checkpoint.

    scripts/prune_layers.py SRC DEST --layers 4 [--keep-layer-tables] [--drop-vision]
                                     [--revision REV] [--dry-run]

``SRC`` is a local checkpoint directory or ``hf://ORG/NAME``; with ``hf://`` only
the shards that hold a kept tensor are downloaded (plus the small files), never
the whole checkpoint. ``DEST`` receives:

* the shards that hold kept tensors. A shard whose tensors are all kept is
  copied (or hard-linked) as it is; one that mixes kept and dropped tensors is
  rewritten with the kept tensors only. Tensors are moved as bytes (the
  safetensors header and byte ranges), so any dtype -- bf16, the MXFP4 packed
  uint8 experts and their scales -- goes through unchanged, on CPU, without torch;
* ``model.safetensors.index.json`` over the kept tensors, with ``total_size``;
* ``config.json`` with ``num_hidden_layers`` = N and the layer-type tables cut
  to the kept layers (Kimi K3.1: ``linear_attn_config.full_attn_layers`` /
  ``kda_layers`` and ``dsa_config.dsa_layers``, 1-based entries; the general
  ``layer_types`` list by position), ``num_nextn_predict_layers`` = 0 and no
  MTP / draft tensors (``--keep-layer-tables`` leaves the tables as they are,
  as some published cut-down checkpoints do);
* every other file of the checkpoint (tokenizer, processors, modeling code,
  generation config) copied as it is.

Kept: layers ``0..N-1`` with all their experts, the embeddings, the final norm
and head, the vision tower and projector (``--drop-vision`` removes those two).
Dropped: layers ``>= N`` and any next-token-prediction (``draft`` / ``mtp`` /
``nextn``) tensors.

Kimi-K3.1-Pro, ``--layers 4``: layer 0 is the dense-MLP layer, layers 1-2 KDA
and layer 3 the first MLA/DSA layer, all three MoE layers with their 896
experts; 7 of the 96 shards are needed (about 58 GB), as in the upstream
4-layer reference.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import sys
from pathlib import Path
from typing import Any, Iterable

INDEX_FILE = "model.safetensors.index.json"
SINGLE_FILE = "model.safetensors"
LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
# next-token-prediction / draft heads: never kept (num_nextn_predict_layers -> 0)
DRAFT_RE = re.compile(r"(?:^|\.)(draft|mtp|nextn|eh_proj|enorm|hnorm|shared_head)(?:$|\.)")
VISION_PREFIXES = ("vision_tower.", "mm_projector.", "vision_model.", "visual.", "multi_modal_projector.")
# config keys that list layer indices, 1-based (Kimi K3 / K3.1)
ONE_BASED_LAYER_LISTS = {"linear_attn_config": ("full_attn_layers", "kda_layers"),
                         "dsa_config": ("dsa_layers",)}
# config keys that list layer indices, 0-based (other families)
ZERO_BASED_LAYER_LISTS = ("moe_layers", "dense_layers", "sliding_window_layers", "full_attention_layers")
HEADER_LEN = 8


class PruneError(RuntimeError):
    pass


# --- tensor selection ----------------------------------------------------------

def layer_of(name: str) -> int | None:
    """The decoder layer index of a tensor name, None for a non-layer tensor.
    Vision encoder blocks say ``blocks.N``, not ``layers.N``, so they stay None."""
    match = LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def keep_tensor(name: str, layers: int, *, drop_vision: bool = False) -> bool:
    if DRAFT_RE.search(name):
        return False
    if drop_vision and name.startswith(VISION_PREFIXES):
        return False
    index = layer_of(name)
    return index is None or index < layers


# --- safetensors by bytes -----------------------------------------------------

def read_header(path: Path) -> tuple[dict[str, Any], int]:
    """The safetensors header (tensor name -> dtype/shape/data_offsets, plus
    ``__metadata__``) and the offset of the data section."""
    with open(path, "rb") as fh:
        raw = fh.read(HEADER_LEN)
        if len(raw) != HEADER_LEN:
            raise PruneError(f"{path}: not a safetensors file")
        (length,) = struct.unpack("<Q", raw)
        header = json.loads(fh.read(length))
    return header, HEADER_LEN + length


def write_subset(src: Path, dest: Path, names: Iterable[str], *, chunk: int = 64 << 20) -> int:
    """Write the tensors ``names`` of ``src`` into a new safetensors file ``dest``,
    bytes copied range by range; returns the data bytes written."""
    header, data_start = read_header(src)
    meta = header.get("__metadata__")
    kept = [n for n in names if n in header]
    missing = sorted(set(names) - set(kept))
    if missing:
        raise PruneError(f"{src}: tensors not in the file: {missing[:5]}")
    kept.sort(key=lambda n: header[n]["data_offsets"][0])
    new_header: dict[str, Any] = {}
    if meta is not None:
        new_header["__metadata__"] = meta
    offset = 0
    for name in kept:
        begin, end = header[name]["data_offsets"]
        new_header[name] = {**header[name], "data_offsets": [offset, offset + (end - begin)]}
        offset += end - begin
    encoded = json.dumps(new_header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)  # safetensors pads the header to 8 bytes
    tmp = dest.with_name(dest.name + ".tmp")
    with open(src, "rb") as fin, open(tmp, "wb") as fout:
        fout.write(struct.pack("<Q", len(encoded)))
        fout.write(encoded)
        for name in kept:
            begin, end = header[name]["data_offsets"]
            fin.seek(data_start + begin)
            left = end - begin
            while left:
                buf = fin.read(min(chunk, left))
                if not buf:
                    raise PruneError(f"{src}: short read in {name}")
                fout.write(buf)
                left -= len(buf)
    os.replace(tmp, dest)
    return offset


def tensor_bytes(header: dict[str, Any], name: str) -> int:
    begin, end = header[name]["data_offsets"]
    return end - begin


# --- config ----------------------------------------------------------------------

def prune_config(config: dict[str, Any], layers: int, *, keep_tables: bool = False,
                 drop_vision: bool = False) -> dict[str, Any]:
    """The config of the N-layer model: ``num_hidden_layers`` on the text config
    (nested ``text_config`` or top level), the layer tables cut to the kept layers
    unless ``keep_tables``, no next-token-prediction layers."""
    out = json.loads(json.dumps(config))  # deep copy
    text = out.get("text_config") if isinstance(out.get("text_config"), dict) else out
    if "num_hidden_layers" not in text:
        raise PruneError("config has no num_hidden_layers (looked in text_config and at the top level)")
    original = int(text["num_hidden_layers"])
    if not 1 <= layers <= original:
        raise PruneError(f"--layers {layers} must be in 1..{original} (the checkpoint's num_hidden_layers)")
    text["num_hidden_layers"] = layers
    if "num_nextn_predict_layers" in text:
        text["num_nextn_predict_layers"] = 0
    if not keep_tables:
        for section, keys in ONE_BASED_LAYER_LISTS.items():
            block = text.get(section)
            if isinstance(block, dict):
                for key in keys:
                    if isinstance(block.get(key), list):
                        block[key] = [i for i in block[key] if 1 <= int(i) <= layers]
        for key in ZERO_BASED_LAYER_LISTS:
            if isinstance(text.get(key), list):
                text[key] = [i for i in text[key] if 0 <= int(i) < layers]
        if isinstance(text.get("layer_types"), list):
            text["layer_types"] = text["layer_types"][:layers]
        if isinstance(text.get("first_k_dense_replace"), int):
            text["first_k_dense_replace"] = min(int(text["first_k_dense_replace"]), layers)
    if drop_vision:
        out.pop("vision_config", None)
    return out


def kept_layer_kinds(config: dict[str, Any], layers: int) -> dict[int, str]:
    """For the report: ``dense`` / ``kda`` / ``mla`` per kept layer of a Kimi K3
    config (empty for other families)."""
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    linear = text.get("linear_attn_config") if isinstance(text.get("linear_attn_config"), dict) else {}
    full = {int(i) for i in linear.get("full_attn_layers", [])}
    dense_k = int(text.get("first_k_dense_replace", 0) or 0)
    kinds = {}
    if full:
        for i in range(layers):
            kinds[i] = "dense-mlp " + ("mla" if i + 1 in full else "kda") if i < dense_k else ("mla" if i + 1 in full else "kda")
    return kinds


# --- sources -------------------------------------------------------------------

class LocalSource:
    def __init__(self, root: Path) -> None:
        self.root = root

    def file(self, name: str) -> Path:
        path = self.root / name
        if not path.exists():
            raise PruneError(f"{path} does not exist")
        return path

    def exists(self, name: str) -> bool:
        return (self.root / name).exists()

    def list(self) -> list[str]:
        return sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*") if p.is_file())


class HubSource:
    """``hf://ORG/NAME``: files fetched one by one (``huggingface_hub.hf_hub_download``)."""

    def __init__(self, repo: str, revision: str | None = None, cache_dir: str | None = None) -> None:
        self.repo, self.revision, self.cache_dir = repo, revision, cache_dir
        self._names: list[str] | None = None

    def list(self) -> list[str]:
        if self._names is None:
            from huggingface_hub import list_repo_files  # noqa: PLC0415

            self._names = sorted(list_repo_files(self.repo, revision=self.revision))
        return self._names

    def exists(self, name: str) -> bool:
        return name in self.list()

    def file(self, name: str) -> Path:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        return Path(hf_hub_download(self.repo, name, revision=self.revision, cache_dir=self.cache_dir))


def open_source(spec: str, *, revision: str | None = None, cache_dir: str | None = None):
    if spec.startswith("hf://"):
        return HubSource(spec[len("hf://"):].strip("/"), revision=revision, cache_dir=cache_dir)
    root = Path(spec).expanduser()
    if not root.is_dir():
        raise PruneError(f"{spec}: not a directory (use hf://ORG/NAME for a Hub checkpoint)")
    return LocalSource(root)


# --- the cut -------------------------------------------------------------------

def weight_map_of(source) -> tuple[dict[str, str], dict[str, Any]]:
    """``{tensor: shard file}`` of the checkpoint (an index, or the single file's header)."""
    if source.exists(INDEX_FILE):
        index = json.loads(source.file(INDEX_FILE).read_text(encoding="utf-8"))
        return dict(index["weight_map"]), dict(index.get("metadata") or {})
    if source.exists(SINGLE_FILE):
        header, _ = read_header(source.file(SINGLE_FILE))
        return {n: SINGLE_FILE for n in header if n != "__metadata__"}, {}
    raise PruneError(f"no {INDEX_FILE} or {SINGLE_FILE} in the checkpoint")


def plan(weight_map: dict[str, str], layers: int, *, drop_vision: bool = False) -> dict[str, Any]:
    """Which tensors stay, grouped by shard, and which shards are copied whole."""
    kept = {n for n in weight_map if keep_tensor(n, layers, drop_vision=drop_vision)}
    by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(name)
    shards_kept = {s: sorted(n for n in names if n in kept) for s, names in by_shard.items()}
    shards_kept = {s: names for s, names in shards_kept.items() if names}
    whole = sorted(s for s, names in shards_kept.items() if len(names) == len(by_shard[s]))
    partial = sorted(s for s in shards_kept if s not in whole)
    return {"kept": kept, "shards": shards_kept, "whole": whole, "partial": partial,
            "dropped": len(weight_map) - len(kept),
            "kept_layers": sorted({layer_of(n) for n in kept if layer_of(n) is not None})}


def copy_or_link(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    try:
        os.link(src, dest)
    except OSError:
        shutil.copyfile(src, dest)


def prune(source, dest: Path, layers: int, *, keep_tables: bool = False, drop_vision: bool = False,
          dry_run: bool = False, log=print) -> dict[str, Any]:
    weight_map, _old_meta = weight_map_of(source)
    config = json.loads(source.file("config.json").read_text(encoding="utf-8"))
    new_config = prune_config(config, layers, keep_tables=keep_tables, drop_vision=drop_vision)
    p = plan(weight_map, layers, drop_vision=drop_vision)
    if p["kept_layers"] != list(range(layers)):
        raise PruneError(f"the checkpoint has no tensors for layers {sorted(set(range(layers)) - set(p['kept_layers']))}")
    kinds = kept_layer_kinds(config, layers)
    log(f"[prune] {len(p['kept'])} tensors kept, {p['dropped']} dropped; shards: {len(p['whole'])} whole, "
        f"{len(p['partial'])} rewritten, of {len(set(weight_map.values()))}")
    for i in range(layers):
        log(f"[prune] layer {i}: {kinds.get(i, '?')}")
    if dry_run:
        return {"kept": len(p["kept"]), "dropped": p["dropped"], "shards": sorted(p["shards"]),
                "config": new_config, "layer_kinds": kinds}
    dest.mkdir(parents=True, exist_ok=True)
    new_map: dict[str, str] = {}
    total = 0
    for shard in sorted(p["shards"]):
        src = source.file(shard)
        header, _ = read_header(src)
        names = p["shards"][shard]
        if shard in p["whole"]:
            copy_or_link(src, dest / shard)
            log(f"[prune] {shard}: kept whole ({len(names)} tensors)")
        else:
            write_subset(src, dest / shard, names)
            log(f"[prune] {shard}: rewritten with {len(names)} of {len([n for n in header if n != '__metadata__'])} tensors")
        for name in names:
            new_map[name] = shard
            total += tensor_bytes(header, name)
    (dest / INDEX_FILE).write_text(json.dumps({"metadata": {"total_size": total},
                                               "weight_map": dict(sorted(new_map.items()))}, indent=2) + "\n",
                                   encoding="utf-8")
    (dest / "config.json").write_text(json.dumps(new_config, indent=2) + "\n", encoding="utf-8")
    for name in source.list():
        if name in (INDEX_FILE, "config.json") or name.endswith(".safetensors") or name.startswith("."):
            continue
        if "/" in name and not name.startswith(("assets/",)):
            continue  # nested directories other than assets are not part of the checkpoint proper
        copy_or_link(source.file(name), dest / name)
    (dest / "PRUNE.json").write_text(json.dumps({
        "layers": layers, "kept_tensors": len(p["kept"]), "dropped_tensors": p["dropped"],
        "shards_whole": p["whole"], "shards_rewritten": p["partial"], "total_size": total,
        "layer_kinds": {str(k): v for k, v in kinds.items()}, "keep_layer_tables": keep_tables,
        "drop_vision": drop_vision,
        "source": getattr(source, "repo", None) or str(getattr(source, "root", "")),
    }, indent=2) + "\n", encoding="utf-8")
    log(f"[prune] wrote {dest} ({total / 1e9:.1f} GB of tensors)")
    return {"kept": len(p["kept"]), "dropped": p["dropped"], "shards": sorted(p["shards"]),
            "total_size": total, "config": new_config, "layer_kinds": kinds}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("src", help="checkpoint directory, or hf://ORG/NAME")
    ap.add_argument("dest", type=Path)
    ap.add_argument("--layers", type=int, required=True, help="keep decoder layers 0..N-1")
    ap.add_argument("--keep-layer-tables", action="store_true",
                    help="leave the layer-type tables as they are")
    ap.add_argument("--drop-vision", action="store_true", help="drop the vision tower and projector")
    ap.add_argument("--revision", default=None, help="hf://: the revision to read")
    ap.add_argument("--cache-dir", default=None, help="hf://: huggingface_hub cache directory")
    ap.add_argument("--dry-run", action="store_true", help="report the plan and the new config, write nothing")
    args = ap.parse_args(argv)
    try:
        source = open_source(args.src, revision=args.revision, cache_dir=args.cache_dir)
        result = prune(source, args.dest, args.layers, keep_tables=args.keep_layer_tables,
                       drop_vision=args.drop_vision, dry_run=args.dry_run)
    except PruneError as error:
        print(f"prune_layers: {error}", file=sys.stderr)
        return 1
    if args.dry_run:
        print(json.dumps({"kept": result["kept"], "dropped": result["dropped"], "shards": result["shards"],
                          "layer_kinds": {str(k): v for k, v in result["layer_kinds"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
