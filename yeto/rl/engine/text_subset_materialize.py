"""Write the text subset of a VL checkpoint as a plain text-only checkpoint (S27 G-EZ).

The selection itself is :mod:`yeto.rl.engine.text_subset`; this module only
moves bytes. Tensor data is copied byte for byte from the source shard to the
output shard, so a converted tensor is bitwise equal to its source tensor and
no dtype or shape is reinterpreted. Excluded tensors (vision tower,
projectors, MTP block) are not written at all.

safetensors is read and written with the standard library: a file is an
8-byte little-endian header length, that many bytes of JSON mapping each
tensor name to ``{dtype, shape, data_offsets}`` (offsets relative to the end
of the header), then the data blob. No torch, no safetensors package: this
runs under a bare ``python3 -I`` on a training node.

Output shard files keep their source file names, so a shard that loses its
vision tensors keeps its name and gets smaller. ``model.safetensors.index.json``
is the authority on which file holds which tensor, so the ``-of-000NN`` part
of a name stays as it was.

* :func:`materialize`: write the whole converted checkpoint plus a manifest;
* :func:`verify_tensor_bytes`: re-read sampled tensors from source and output
  and compare the bytes;
* ``python3 -I -m yeto.rl.engine.text_subset_materialize SRC DST`` (or run the
  file directly) for use on a training node.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import sys
from collections.abc import Mapping
from typing import Any

from yeto.rl.engine.text_subset import (
    TextSubset,
    TextSubsetError,
    plan_subset,
    rule_for_config,
    text_config,
)

_HEADER_LEN = 8
_ALIGN = 8
_COPY_CHUNK = 16 * 1024 * 1024
# Files to carry over unchanged. The vision side's preprocessor and processor
# configs are deliberately not carried: the converted checkpoint has no vision
# tower, so a processor config would describe modules that do not exist.
CARRY_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
)
SKIP_FILES = (
    "preprocessor_config.json",
    "processor_config.json",
    "configuration.json",
    "model.safetensors.index.json",
    "config.json",
)


def read_header(path: str) -> tuple[dict[str, Any], int]:
    """``(header, data_start)`` of a safetensors file. Reads the header only."""
    with open(path, "rb") as fh:
        raw = fh.read(_HEADER_LEN)
        if len(raw) != _HEADER_LEN:
            raise TextSubsetError(f"{path}: shorter than a safetensors header")
        (header_len,) = struct.unpack("<Q", raw)
        header = json.loads(fh.read(header_len))
    return header, _HEADER_LEN + header_len


def _build_header(entries: list[tuple[str, dict[str, Any]]], metadata: Mapping[str, Any] | None):
    """Header JSON bytes and the output entries with recomputed contiguous offsets."""
    out: dict[str, Any] = {}
    if metadata:
        out["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    cursor = 0
    plan: list[tuple[str, int, int]] = []
    for name, meta in entries:
        begin, end = meta["data_offsets"]
        size = int(end) - int(begin)
        out[name] = {
            "dtype": meta["dtype"],
            "shape": list(meta["shape"]),
            "data_offsets": [cursor, cursor + size],
        }
        plan.append((name, int(begin), size))
        cursor += size
    blob = json.dumps(out, separators=(",", ":")).encode("utf-8")
    pad = (-(_HEADER_LEN + len(blob))) % _ALIGN
    return blob + b" " * pad, plan, cursor


def _copy_range(src, dst, offset: int, size: int) -> None:
    src.seek(offset)
    left = size
    while left:
        chunk = src.read(min(_COPY_CHUNK, left))
        if not chunk:
            raise TextSubsetError("source shard ended early while copying tensor data")
        dst.write(chunk)
        left -= len(chunk)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_COPY_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_shard(src_path: str, dst_path: str, keep: Mapping[str, str]) -> dict[str, Any]:
    """Write the kept tensors of one source shard under their text names.

    ``keep`` maps source tensor name -> output tensor name. Returns the output
    file's tensor names, byte size and sha256. The tensor data is copied
    without being decoded.
    """
    header, data_start = read_header(src_path)
    entries = [
        (keep[name], meta)
        for name, meta in header.items()
        if name != "__metadata__" and name in keep
    ]
    if not entries:
        return {"written": False}
    entries.sort(key=lambda item: item[1]["data_offsets"][0])
    header_blob, plan, total = _build_header(entries, header.get("__metadata__"))
    tmp_path = dst_path + ".partial"
    with open(src_path, "rb") as src, open(tmp_path, "wb") as dst:
        dst.write(struct.pack("<Q", len(header_blob)))
        dst.write(header_blob)
        for _name, begin, size in plan:
            _copy_range(src, dst, data_start + begin, size)
    os.replace(tmp_path, dst_path)
    return {
        "written": True,
        "tensors": [name for name, _b, _s in plan],
        "data_bytes": total,
        "file_bytes": os.path.getsize(dst_path),
        "sha256": _sha256(dst_path),
    }


def materialize(src_dir: str, dst_dir: str, *, progress=None) -> dict[str, Any]:
    """Write the text-only checkpoint of ``src_dir`` into ``dst_dir``.

    Returns the manifest (also written to ``dst_dir/TEXT_SUBSET.json``):
    the rule as data, the per-file sha256 and sizes, the tensor and parameter
    counts, and the excluded tensors by prefix.
    """
    with open(os.path.join(src_dir, "config.json")) as fh:
        config = json.load(fh)
    rule = rule_for_config(config)
    if rule is None:
        raise TextSubsetError(
            f"{src_dir}: model_type {config.get('model_type')!r} has no text-subset rule")
    index_path = os.path.join(src_dir, "model.safetensors.index.json")
    with open(index_path) as fh:
        index = json.load(fh)
    weight_map: dict[str, str] = index["weight_map"]
    subset: TextSubset = plan_subset(weight_map.keys(), rule)

    by_file: dict[str, dict[str, str]] = {}
    for src_name, out_name in subset.renames.items():
        by_file.setdefault(weight_map[src_name], {})[src_name] = out_name

    os.makedirs(dst_dir, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    out_weight_map: dict[str, str] = {}
    total_bytes = 0
    for position, file_name in enumerate(sorted(by_file), start=1):
        if progress is not None:
            progress(f"shard {position}/{len(by_file)} {file_name}")
        result = write_shard(
            os.path.join(src_dir, file_name),
            os.path.join(dst_dir, file_name),
            by_file[file_name],
        )
        if not result.get("written"):
            continue
        for tensor in result["tensors"]:
            out_weight_map[tensor] = file_name
        total_bytes += int(result["data_bytes"])
        files[file_name] = {
            "file_bytes": result["file_bytes"],
            "sha256": result["sha256"],
            "tensors": len(result["tensors"]),
        }

    missing = sorted(set(subset.renames.values()) - set(out_weight_map))
    if missing:
        raise TextSubsetError(
            f"{len(missing)} planned tensor(s) were not written, first {missing[:3]}")

    with open(os.path.join(dst_dir, "model.safetensors.index.json"), "w") as fh:
        json.dump(
            {"metadata": {"total_size": total_bytes}, "weight_map": out_weight_map},
            fh,
            indent=2,
            sort_keys=True,
        )
    with open(os.path.join(dst_dir, "config.json"), "w") as fh:
        json.dump(text_config(config, rule), fh, indent=2, sort_keys=True)

    carried = []
    for name in CARRY_FILES:
        src_path = os.path.join(src_dir, name)
        if os.path.isfile(src_path):
            shutil.copy2(src_path, os.path.join(dst_dir, name))
            carried.append(name)

    manifest = {
        "schema": "yeto-text-subset-manifest-v1",
        "source_dir": os.path.abspath(src_dir),
        "output_dir": os.path.abspath(dst_dir),
        "source_index_sha256": _sha256(index_path),
        "text_tensors": subset.text_tensors,
        "excluded_tensors": subset.excluded_tensors,
        "excluded_by_prefix": {k: len(v) for k, v in sorted(subset.excluded.items())},
        "rule": rule.as_data(),
        "total_size": total_bytes,
        "shards": files,
        "carried_files": carried,
        "skipped_files": [n for n in SKIP_FILES if os.path.isfile(os.path.join(src_dir, n))],
    }
    with open(os.path.join(dst_dir, "TEXT_SUBSET.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    return manifest


def verify_tensor_bytes(src_dir: str, dst_dir: str, *, sample: int = 24, seed: int = 0):
    """Re-read sampled tensors from both checkpoints and compare their bytes.

    Picks the largest tensor, the smallest tensor and an evenly spaced sample
    of the rest, so the sample is deterministic and covers both ends.
    Returns ``(checked, details)``; raises on the first mismatch.
    """
    with open(os.path.join(src_dir, "config.json")) as fh:
        config = json.load(fh)
    rule = rule_for_config(config)
    if rule is None:
        raise TextSubsetError(f"{src_dir}: no text-subset rule")
    with open(os.path.join(src_dir, "model.safetensors.index.json")) as fh:
        src_map = json.load(fh)["weight_map"]
    with open(os.path.join(dst_dir, "model.safetensors.index.json")) as fh:
        dst_map = json.load(fh)["weight_map"]
    subset = plan_subset(src_map.keys(), rule)

    sizes: list[tuple[int, str]] = []
    headers: dict[str, tuple[dict[str, Any], int]] = {}
    for src_name in subset.renames:
        file_name = src_map[src_name]
        if file_name not in headers:
            headers[file_name] = read_header(os.path.join(src_dir, file_name))
        meta = headers[file_name][0][src_name]
        begin, end = meta["data_offsets"]
        sizes.append((int(end) - int(begin), src_name))
    sizes.sort()
    picks = {sizes[0][1], sizes[-1][1]}
    if sample > 2 and len(sizes) > 2:
        step = max(1, len(sizes) // (sample - 2))
        picks.update(name for _s, name in sizes[::step][: sample - 2])

    details = []
    for src_name in sorted(picks):
        out_name = subset.renames[src_name]
        src_file = os.path.join(src_dir, src_map[src_name])
        dst_file = os.path.join(dst_dir, dst_map[out_name])
        src_header, src_start = headers[src_map[src_name]]
        dst_header, dst_start = read_header(dst_file)
        src_meta, dst_meta = src_header[src_name], dst_header[out_name]
        if src_meta["dtype"] != dst_meta["dtype"] or list(src_meta["shape"]) != list(
            dst_meta["shape"]
        ):
            raise TextSubsetError(
                f"{src_name} -> {out_name}: dtype/shape changed "
                f"({src_meta['dtype']}{src_meta['shape']} vs {dst_meta['dtype']}{dst_meta['shape']})")
        sb, se = (int(x) for x in src_meta["data_offsets"])
        db, de = (int(x) for x in dst_meta["data_offsets"])
        if se - sb != de - db:
            raise TextSubsetError(f"{src_name} -> {out_name}: byte length changed")
        left = se - sb
        src_pos, dst_pos = src_start + sb, dst_start + db
        with open(src_file, "rb") as sfh, open(dst_file, "rb") as dfh:
            sfh.seek(src_pos)
            dfh.seek(dst_pos)
            digest_s, digest_d = hashlib.sha256(), hashlib.sha256()
            while left:
                take = min(_COPY_CHUNK, left)
                a, b = sfh.read(take), dfh.read(take)
                if a != b:
                    raise TextSubsetError(f"{src_name} -> {out_name}: tensor bytes differ")
                digest_s.update(a)
                digest_d.update(b)
                left -= take
        details.append(
            {
                "source": src_name,
                "output": out_name,
                "bytes": se - sb,
                "sha256": digest_s.hexdigest(),
                "equal": digest_s.hexdigest() == digest_d.hexdigest(),
            }
        )
    return len(details), details


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        print("usage: text_subset_materialize.py SRC_DIR DST_DIR [--verify N]")
        return 2
    src_dir, dst_dir = argv[1], argv[2]
    sample = 24
    if "--verify" in argv:
        sample = int(argv[argv.index("--verify") + 1])
    manifest = materialize(src_dir, dst_dir, progress=lambda m: print(m, flush=True))
    print(
        json.dumps(
            {k: v for k, v in manifest.items() if k != "shards"}, indent=2, sort_keys=True
        )
    )
    print(f"shards written: {len(manifest['shards'])}", flush=True)
    checked, details = verify_tensor_bytes(src_dir, dst_dir, sample=sample)
    bad = [d for d in details if not d["equal"]]
    print(f"byte-compared {checked} tensors, mismatches {len(bad)}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
