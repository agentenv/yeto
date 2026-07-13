"""Fragment layout metadata shared by learners, syncer checkpoints, and export."""

from __future__ import annotations

import hashlib
import json
from typing import Mapping

import torch

from .fragments import FragmentLayout, MERGE_AVG, MERGE_ISO, MERGE_RDA, MERGE_WORKER_SNR

LAYOUT_META_VERSION = 1
MERGE_MODE_NAMES = {
    MERGE_AVG: "avg",
    MERGE_RDA: "rda",
    MERGE_ISO: "iso",
    MERGE_WORKER_SNR: "worker-snr",
}


def build_layout_metadata(
    *,
    task: str,
    layout: FragmentLayout,
    params: Mapping[str, torch.Tensor],
    backend_version: str = "1",
    **extra,
) -> dict:
    """Return deterministic JSON-serializable metadata for a fragment layout."""
    fragments = []
    for fid, frag in enumerate(layout.fragments):
        tensors = []
        for name, numel in frag.tensors:
            tensor = params[name]
            tensors.append(
                {
                    "name": name,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype).replace("torch.", ""),
                    "numel": int(numel),
                }
            )
        fragments.append(
            {
                "id": fid,
                "merge_mode": MERGE_MODE_NAMES.get(frag.merge_mode, str(frag.merge_mode)),
                "tensors": tensors,
            }
        )
    meta = {
        "layout_meta_version": LAYOUT_META_VERSION,
        "task": task,
        "backend_version": backend_version,
        "fragments": fragments,
    }
    meta.update({k: v for k, v in extra.items() if v is not None})
    meta["layout_hash"] = layout_metadata_hash(meta, include_hash=False)
    return meta


def encode_layout_metadata(meta: dict | bytes | str | None) -> bytes | None:
    if meta is None:
        return None
    if isinstance(meta, bytes):
        return meta
    if isinstance(meta, str):
        return meta.encode("utf-8")
    return json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8")


def decode_layout_metadata(raw: bytes | str | None) -> dict | None:
    if raw in (None, b"", ""):
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def layout_metadata_hash(meta: dict, *, include_hash: bool = False) -> str:
    payload = dict(meta) if include_hash else {k: v for k, v in meta.items() if k != "layout_hash"}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_layout_metadata(
    meta: dict | None,
    layout: FragmentLayout,
    params: Mapping[str, torch.Tensor],
    *,
    expected_task: str | None = None,
    base_checkpoint_sha256: str | None = None,
    allow_base_mismatch: bool = False,
) -> None:
    """Validate checkpoint metadata against a freshly built model layout."""
    if not meta:
        return
    if expected_task is not None and meta.get("task") != expected_task:
        raise ValueError(f"checkpoint task {meta.get('task')!r} != expected {expected_task!r}")
    expected_hash = meta.get("layout_hash")
    if expected_hash and expected_hash != layout_metadata_hash(meta, include_hash=False):
        raise ValueError("checkpoint layout metadata hash is invalid")
    problems: list[str] = []
    meta_frags = meta.get("fragments") or []
    if len(meta_frags) != layout.num_fragments:
        problems.append(f"metadata has {len(meta_frags)} fragments, rebuilt layout has {layout.num_fragments}")
    for fid, (meta_frag, frag) in enumerate(zip(meta_frags, layout.fragments)):
        if meta_frag.get("id") != fid:
            problems.append(f"fragment {fid}: metadata id is {meta_frag.get('id')}")
        expected_mode = MERGE_MODE_NAMES.get(frag.merge_mode, str(frag.merge_mode))
        if meta_frag.get("merge_mode") != expected_mode:
            problems.append(f"fragment {fid}: merge mode {meta_frag.get('merge_mode')} != {expected_mode}")
        meta_tensors = meta_frag.get("tensors") or []
        if len(meta_tensors) != len(frag.tensors):
            problems.append(f"fragment {fid}: metadata has {len(meta_tensors)} tensors, layout has {len(frag.tensors)}")
            continue
        for (name, numel), meta_tensor in zip(frag.tensors, meta_tensors):
            if meta_tensor.get("name") != name:
                problems.append(f"fragment {fid}: tensor name {meta_tensor.get('name')!r} != {name!r}")
            if int(meta_tensor.get("numel", -1)) != int(numel):
                problems.append(f"fragment {fid}/{name}: numel {meta_tensor.get('numel')} != {numel}")
            shape = meta_tensor.get("shape")
            if shape is not None and list(params[name].shape) != list(shape):
                problems.append(f"fragment {fid}/{name}: shape {shape} != {list(params[name].shape)}")
    if problems:
        raise ValueError("checkpoint layout metadata does not match rebuilt layout: " + "; ".join(problems[:12]))

    meta_base = meta.get("base_checkpoint_sha256")
    if meta_base and base_checkpoint_sha256 and meta_base != base_checkpoint_sha256 and not allow_base_mismatch:
        raise ValueError(
            "base checkpoint sha256 mismatch: "
            f"checkpoint metadata has {meta_base}, current base is {base_checkpoint_sha256}; "
            "pass --allow-base-mismatch to override"
        )
