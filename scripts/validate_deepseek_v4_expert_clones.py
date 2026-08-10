#!/usr/bin/env python3
"""Independently audit a DeepSeek-V4 logical-expert clone checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from collections import Counter, defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

from yeto.rl.deepseek_v4_expert_clone import (
    CLONES_PER_LAYER,
    NUM_LAYERS,
    ORIGINAL_EXPERTS,
    TOTAL_EXPERTS,
    contract_from_config,
    contract_from_selection,
    sha256_file,
)


_BF16 = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?P<suffix>(?:gate_proj|up_proj|down_proj)\.weight)$"
)
_FP8 = re.compile(
    r"^layers\.(?P<layer>\d+)\.ffn\.experts\."
    r"(?P<expert>\d+)\.(?P<suffix>w[123]\.(?:weight|scale))$"
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expanded", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--selection-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--index-sha256", required=True)
    parser.add_argument("--kind", choices=("bf16", "fp8"), required=True)
    parser.add_argument("--require-hardlinks", action="store_true")
    return parser.parse_args()


def _index(path: Path):
    value = json.loads((path / "model.safetensors.index.json").read_text())
    if not isinstance(value.get("weight_map"), dict):
        raise ValueError("checkpoint has no weight map")
    return value


def _regular_tree(root: Path) -> int:
    files = 0
    for current, directories, names in os.walk(root, followlinks=False):
        for name in (*directories, *names):
            path = Path(current) / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ValueError(f"expanded checkpoint contains a symlink: {path}")
            if name in directories and not stat.S_ISDIR(mode):
                raise ValueError(f"expanded checkpoint contains a special directory: {path}")
            if name in names:
                if not stat.S_ISREG(mode):
                    raise ValueError(f"expanded checkpoint contains a special file: {path}")
                files += 1
    return files


def _headers_match_index(root: Path, weight_map: dict[str, str]):
    expected = defaultdict(set)
    for key, shard in weight_map.items():
        expected[shard].add(key)
    discovered = set()
    dtype_histogram = Counter()
    for shard, keys in sorted(expected.items()):
        path = root / shard
        if not path.is_file():
            raise FileNotFoundError(f"index shard is absent: {path}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            actual = set(handle.keys())
            if actual != keys:
                raise ValueError(
                    f"index/header mismatch for {shard}: "
                    f"missing={sorted(keys-actual)[:3]} extra={sorted(actual-keys)[:3]}"
                )
            for key in actual:
                dtype_histogram[str(handle.get_slice(key).get_dtype())] += 1
            discovered.update(actual)
    if discovered != set(weight_map):
        raise ValueError("expanded index does not cover every tensor exactly once")
    return dtype_histogram


def _tensor(root: Path, weight_map: dict[str, str], key: str) -> torch.Tensor:
    shard = weight_map.get(key)
    if shard is None:
        raise KeyError(f"tensor is absent from index: {key}")
    with safe_open(root / shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def _key(kind: str, layer: int, expert: int, suffix: str) -> str:
    if kind == "bf16":
        return f"model.layers.{layer}.mlp.experts.{expert}.{suffix}"
    return f"layers.{layer}.ffn.experts.{expert}.{suffix}"


def _expert_contract(weight_map, kind: str):
    pattern = _BF16 if kind == "bf16" else _FP8
    values = defaultdict(lambda: defaultdict(set))
    for key in weight_map:
        match = pattern.fullmatch(key)
        if match is not None:
            values[int(match.group("layer"))][int(match.group("expert"))].add(
                match.group("suffix")
            )
    if set(values) != set(range(NUM_LAYERS)):
        raise ValueError("expanded expert tensors do not cover 43 layers")
    suffixes = None
    for layer in range(NUM_LAYERS):
        if set(values[layer]) != set(range(TOTAL_EXPERTS)):
            raise ValueError(f"expanded layer {layer} does not contain experts 0..287")
        for expert in range(TOTAL_EXPERTS):
            current = values[layer][expert]
            if suffixes is None:
                suffixes = current
            if current != suffixes:
                raise ValueError(f"expanded layer {layer} expert {expert} is incomplete")
    expected = 3 if kind == "bf16" else 6
    if suffixes is None or len(suffixes) != expected:
        raise ValueError("expanded expert tensor suffix contract is invalid")
    return tuple(sorted(suffixes))


def _router_checks(root: Path, weight_map: dict[str, str], kind: str) -> None:
    if kind == "bf16":
        gate_key = "model.layers.3.mlp.gate.weight"
        bias_key = "model.layers.3.mlp.gate.e_score_correction_bias"
        hash_key = "model.layers.0.mlp.topk.tid2eid"
    else:
        gate_key = "layers.3.ffn.gate.weight"
        bias_key = "layers.3.ffn.gate.bias"
        hash_key = "layers.0.ffn.gate.tid2eid"
    gate = _tensor(root, weight_map, gate_key)
    bias = _tensor(root, weight_map, bias_key)
    table = _tensor(root, weight_map, hash_key)
    if gate.shape[0] != ORIGINAL_EXPERTS or bias.shape != (ORIGINAL_EXPERTS,):
        raise ValueError("expanded checkpoint incorrectly expanded router tensors")
    if table.ndim != 2 or table.shape[1] != 6:
        raise ValueError("expanded checkpoint has an invalid hash route table")
    if int(table.min()) < 0 or int(table.max()) >= ORIGINAL_EXPERTS:
        raise ValueError("expanded hash route table does not remain in base category space")


def main() -> None:
    args = _args()
    for path in (args.source, args.expanded):
        if not path.is_dir():
            raise FileNotFoundError(path)
    expected = {
        "manifest": args.manifest_sha256.lower(),
        "config": args.config_sha256.lower(),
        "index": args.index_sha256.lower(),
    }
    actual = {
        "manifest": sha256_file(args.expanded / "yeto_expert_clone_manifest.json"),
        "config": sha256_file(args.expanded / "config.json"),
        "index": sha256_file(args.expanded / "model.safetensors.index.json"),
    }
    if actual != expected:
        raise ValueError(f"expanded checkpoint provenance mismatch: {actual}")
    contract = contract_from_selection(
        args.selection,
        expected_selection_sha256=args.selection_sha256,
    )
    config = json.loads((args.expanded / "config.json").read_text())
    if contract_from_config(config) != contract:
        raise ValueError("expanded config clone contract differs from selection")
    manifest = json.loads(
        (args.expanded / "yeto_expert_clone_manifest.json").read_text()
    )
    if manifest["output_config_sha256"] != actual["config"] or manifest[
        "output_index_sha256"
    ] != actual["index"]:
        raise ValueError("clone manifest output hashes are inconsistent")

    source_index = _index(args.source)
    expanded_index = _index(args.expanded)
    source_map = source_index["weight_map"]
    expanded_map = expanded_index["weight_map"]
    if any(expanded_map.get(key) != shard for key, shard in source_map.items()):
        raise ValueError("expanded index changed a source tensor mapping")
    if len(expanded_map) != len(source_map) + manifest["clone_tensor_count"]:
        raise ValueError("expanded weight-map size does not match its manifest")
    if expanded_index["metadata"]["total_size"] != (
        source_index["metadata"]["total_size"]
        + manifest["clone_logical_tensor_bytes"]
    ):
        raise ValueError("expanded index logical byte total is inconsistent")

    regular_files = _regular_tree(args.expanded)
    dtype_histogram = _headers_match_index(args.expanded, expanded_map)
    suffixes = _expert_contract(expanded_map, args.kind)
    _router_checks(args.expanded, expanded_map, args.kind)

    if args.require_hardlinks:
        for shard in sorted(set(source_map.values())):
            source = (args.source / shard).resolve(strict=True)
            expanded = args.expanded / shard
            left, right = source.stat(), expanded.stat()
            if (left.st_dev, left.st_ino) != (right.st_dev, right.st_ino):
                raise ValueError(f"source shard was not hard-linked locally: {shard}")

    clone_records = {int(row["layer"]): row for row in manifest["clone_shards"]}
    if set(clone_records) != set(range(NUM_LAYERS)):
        raise ValueError("clone manifest does not cover every layer")
    verified_tensors = 0
    verified_logical_bytes = 0
    for layer, sources in enumerate(contract.source_experts_by_layer):
        record = clone_records[layer]
        clone_shard = args.expanded / record["file"]
        if sha256_file(clone_shard) != record["sha256"]:
            raise ValueError(f"clone shard hash mismatch at layer {layer}")
        with safe_open(clone_shard, framework="pt", device="cpu") as clone_handle:
            for rank, source_expert in enumerate(sources):
                clone_expert = ORIGINAL_EXPERTS + rank
                for suffix in suffixes:
                    source_key = _key(args.kind, layer, source_expert, suffix)
                    clone_key = _key(args.kind, layer, clone_expert, suffix)
                    source_value = _tensor(args.source, source_map, source_key)
                    clone_value = clone_handle.get_tensor(clone_key)
                    if not torch.equal(source_value, clone_value):
                        raise ValueError(
                            f"clone differs from source: {clone_key} <- {source_key}"
                        )
                    verified_tensors += 1
                    verified_logical_bytes += (
                        clone_value.numel() * clone_value.element_size()
                    )
        print(
            json.dumps(
                {
                    "event": "verified_layer",
                    "kind": args.kind,
                    "layer": layer,
                    "tensor_count": verified_tensors,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if verified_tensors != manifest["clone_tensor_count"] or verified_logical_bytes != manifest[
        "clone_logical_tensor_bytes"
    ]:
        raise ValueError("verified clone totals differ from manifest")

    print(
        json.dumps(
            {
                "status": "validated",
                "kind": args.kind,
                "manifest_sha256": actual["manifest"],
                "config_sha256": actual["config"],
                "index_sha256": actual["index"],
                "selection_sha256": contract.selection_sha256,
                "original_tensor_count": len(source_map),
                "expanded_tensor_count": len(expanded_map),
                "clone_tensor_count": verified_tensors,
                "clone_logical_tensor_bytes": verified_logical_bytes,
                "expanded_logical_tensor_bytes": expanded_index["metadata"][
                    "total_size"
                ],
                "regular_files": regular_files,
                "dtype_histogram": dict(sorted(dtype_histogram.items())),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
