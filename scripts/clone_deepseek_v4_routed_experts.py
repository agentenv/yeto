#!/usr/bin/env python3
"""Create an immutable DeepSeek-V4 256→288 logical-expert checkpoint.

Unchanged model files are hard-linked on the same node.  Each decoder layer
gets one new safetensors shard containing exact copies of the 32 experts chosen
by the audited ``always`` routing census.  Router tensors remain 256-way; the
config embeds Yeto's deterministic token-ID clone-split contract.

The MTP layer was not profiled and is not used by this RL recipe.  Its weights
remain in the source shards while ``num_nextn_predict_layers`` is set to zero.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from yeto.rl.deepseek_v4_expert_clone import (
    CLONES_PER_LAYER,
    NUM_LAYERS,
    ORIGINAL_EXPERTS,
    TOTAL_EXPERTS,
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
_EXPECTED_SUFFIXES = {
    "bf16": {
        "down_proj.weight",
        "gate_proj.weight",
        "up_proj.weight",
    },
    "fp8": {
        "w1.scale",
        "w1.weight",
        "w2.scale",
        "w2.weight",
        "w3.scale",
        "w3.weight",
    },
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--selection-sha256", required=True)
    parser.add_argument("--source-config-sha256", required=True)
    parser.add_argument("--source-index-sha256", required=True)
    parser.add_argument("--kind", choices=("bf16", "fp8"), required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--profiler-counts", type=Path, required=True)
    parser.add_argument("--profiler-counts-sha256", required=True)
    parser.add_argument("--profiler-scores", type=Path, required=True)
    parser.add_argument("--profiler-scores-sha256", required=True)
    parser.add_argument("--script-sha256", required=True)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _validated_sha(value: str, name: str) -> str:
    normalized = value.lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError(f"{name} is not a SHA256")
    return normalized


def _validate_paths(source: Path, output: Path, selection: Path) -> None:
    if not source.is_dir() or not selection.is_file():
        raise FileNotFoundError("source checkpoint or selection does not exist")
    source_resolved = source.resolve()
    output_parent = output.parent.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"output parent does not exist: {output.parent}")
    if source_resolved == output.resolve(strict=False):
        raise ValueError("source and output checkpoint paths are identical")
    if source_resolved in output_parent.parents or output_parent == source_resolved:
        raise ValueError("output checkpoint must not be nested inside its source")


def _link_regular(source: Path, destination: Path) -> None:
    resolved = source.resolve(strict=True) if source.is_symlink() else source
    if not resolved.is_file():
        raise ValueError(f"checkpoint tree contains a non-file entry: {source}")
    os.link(resolved, destination)


def _link_source_tree(source: Path, output: Path) -> None:
    excluded = {
        "config.json",
        "model.safetensors.index.json",
        "yeto_expert_clone_manifest.json",
    }
    for current, directories, files in os.walk(source, followlinks=False):
        current_path = Path(current)
        relative = current_path.relative_to(source)
        destination_root = output / relative
        destination_root.mkdir(exist_ok=True)
        for directory in list(directories):
            child = current_path / directory
            if child.is_symlink():
                raise ValueError(f"checkpoint contains a symlinked directory: {child}")
        for name in files:
            if relative == Path(".") and (
                name in excluded or name.startswith("yeto-clones-layer-")
            ):
                continue
            _link_regular(current_path / name, destination_root / name)


def _index_contract(index: dict[str, Any], kind: str):
    weight_map = index.get("weight_map")
    metadata = index.get("metadata")
    if not isinstance(weight_map, dict) or not isinstance(metadata, dict):
        raise ValueError("invalid safetensors index")
    if not isinstance(metadata.get("total_size"), int) or metadata["total_size"] <= 0:
        raise ValueError("safetensors index has no valid total_size")
    pattern = _BF16 if kind == "bf16" else _FP8
    by_layer: dict[int, dict[int, dict[str, str]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for key, shard in weight_map.items():
        match = pattern.fullmatch(key)
        if match is None:
            continue
        if not isinstance(shard, str):
            raise ValueError(f"invalid shard name for {key}")
        layer = int(match.group("layer"))
        expert = int(match.group("expert"))
        suffix = match.group("suffix")
        by_layer[layer][expert][suffix] = shard
    if set(by_layer) != set(range(NUM_LAYERS)):
        raise ValueError("checkpoint does not contain exactly 43 main expert layers")
    expected_suffixes = _EXPECTED_SUFFIXES[kind]
    for layer in range(NUM_LAYERS):
        if set(by_layer[layer]) != set(range(ORIGINAL_EXPERTS)):
            raise ValueError(f"layer {layer} does not contain experts 0..255")
        for expert in range(ORIGINAL_EXPERTS):
            if set(by_layer[layer][expert]) != expected_suffixes:
                raise ValueError(
                    f"layer {layer} expert {expert} tensor contract is incomplete"
                )
    return weight_map, int(metadata["total_size"]), by_layer


def _key(kind: str, layer: int, expert: int, suffix: str) -> str:
    if kind == "bf16":
        return f"model.layers.{layer}.mlp.experts.{expert}.{suffix}"
    return f"layers.{layer}.ffn.experts.{expert}.{suffix}"


def _load_layer_clones(
    source: Path,
    kind: str,
    layer: int,
    sources: tuple[int, ...],
    by_layer,
) -> tuple[dict[str, torch.Tensor], int, dict[str, str]]:
    requests: dict[str, list[tuple[str, str]]] = defaultdict(list)
    output_map = {}
    suffixes = sorted(_EXPECTED_SUFFIXES[kind])
    for rank, source_expert in enumerate(sources):
        clone_expert = ORIGINAL_EXPERTS + rank
        for suffix in suffixes:
            source_key = _key(kind, layer, source_expert, suffix)
            clone_key = _key(kind, layer, clone_expert, suffix)
            shard = by_layer[layer][source_expert][suffix]
            requests[shard].append((source_key, clone_key))
            output_map[clone_key] = source_key

    tensors = {}
    logical_bytes = 0
    for shard, pairs in sorted(requests.items()):
        shard_path = source / shard
        if not shard_path.is_file():
            raise FileNotFoundError(f"missing source shard {shard_path}")
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for source_key, clone_key in pairs:
                if source_key not in available:
                    raise KeyError(f"source tensor is absent from its index shard: {source_key}")
                value = handle.get_tensor(source_key).contiguous()
                if value.device.type != "cpu" or value.requires_grad:
                    raise ValueError(f"invalid source tensor state for {source_key}")
                tensors[clone_key] = value
                logical_bytes += value.numel() * value.element_size()
    if len(tensors) != CLONES_PER_LAYER * len(suffixes):
        raise RuntimeError(f"layer {layer} clone tensor count is incomplete")
    return tensors, logical_bytes, output_map


def _plan(args, config, contract, base_bytes: int) -> dict[str, Any]:
    return {
        "schema": 1,
        "kind": args.kind,
        "source": str(args.source.resolve()),
        "output": str(args.output.resolve(strict=False)),
        "source_revision": args.source_revision,
        "source_config_sha256": sha256_file(args.source / "config.json"),
        "source_index_sha256": sha256_file(
            args.source / "model.safetensors.index.json"
        ),
        "selection_sha256": contract.selection_sha256,
        "selection_contract_sha256": contract.selection_contract_sha256,
        "profiler_counts_sha256": args.profiler_counts_sha256,
        "profiler_scores_sha256": args.profiler_scores_sha256,
        "script_sha256": args.script_sha256,
        "base_logical_tensor_bytes": base_bytes,
        "original_experts": ORIGINAL_EXPERTS,
        "total_experts": TOTAL_EXPERTS,
        "clones_per_layer": CLONES_PER_LAYER,
        "layers": NUM_LAYERS,
        "source_experts_by_layer": [
            list(row) for row in contract.source_experts_by_layer
        ],
        "mtp": "disabled-unmodified",
        "source_num_nextn_predict_layers": int(
            config.get("num_nextn_predict_layers", 0)
        ),
    }


def main() -> None:
    args = _args()
    _validate_paths(args.source, args.output, args.selection)
    expected_config_sha = _validated_sha(
        args.source_config_sha256,
        "source config SHA256",
    )
    expected_index_sha = _validated_sha(
        args.source_index_sha256,
        "source index SHA256",
    )
    for name in (
        "profiler_counts_sha256",
        "profiler_scores_sha256",
        "script_sha256",
    ):
        setattr(args, name, _validated_sha(getattr(args, name), name))
    actual_config_sha = sha256_file(args.source / "config.json")
    actual_index_sha = sha256_file(args.source / "model.safetensors.index.json")
    if actual_config_sha != expected_config_sha or actual_index_sha != expected_index_sha:
        raise ValueError("source checkpoint config/index provenance mismatch")
    if sha256_file(Path(__file__)) != args.script_sha256:
        raise ValueError("checkpoint clone script SHA256 mismatch")
    if (
        not args.profiler_counts.is_file()
        or sha256_file(args.profiler_counts) != args.profiler_counts_sha256
        or not args.profiler_scores.is_file()
        or sha256_file(args.profiler_scores) != args.profiler_scores_sha256
    ):
        raise ValueError("profiler count/score provenance mismatch")

    contract = contract_from_selection(
        args.selection,
        expected_selection_sha256=args.selection_sha256,
    )
    config = json.loads((args.source / "config.json").read_text(encoding="utf-8"))
    if (
        config.get("num_hidden_layers") != NUM_LAYERS
        or config.get("n_routed_experts") != ORIGINAL_EXPERTS
        or config.get("num_experts_per_tok") != 6
        or config.get("num_nextn_predict_layers") != 1
    ):
        raise ValueError("unexpected source DeepSeek V4 architecture")
    index = json.loads(
        (args.source / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    weight_map, base_bytes, by_layer = _index_contract(index, args.kind)
    plan = _plan(args, config, contract, base_bytes)
    if args.plan_only:
        print(json.dumps(plan, sort_keys=True))
        return

    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    temporary.mkdir(mode=0o755)
    _link_source_tree(args.source, temporary)

    clone_bytes = 0
    clone_tensors = 0
    clone_shards = []
    new_weight_map = dict(weight_map)
    for layer, sources in enumerate(contract.source_experts_by_layer):
        tensors, layer_bytes, clone_sources = _load_layer_clones(
            args.source,
            args.kind,
            layer,
            sources,
            by_layer,
        )
        shard_name = f"yeto-clones-layer-{layer:02d}-of-{NUM_LAYERS:02d}.safetensors"
        shard_path = temporary / shard_name
        save_file(
            tensors,
            shard_path,
            metadata={
                "format": "pt",
                "yeto_selection_sha256": contract.selection_sha256,
                "yeto_layer": str(layer),
            },
        )
        with shard_path.open("rb") as handle:
            os.fsync(handle.fileno())
        for clone_key in tensors:
            if clone_key in new_weight_map:
                raise ValueError(f"clone tensor already exists in source index: {clone_key}")
            new_weight_map[clone_key] = shard_name
        clone_shards.append(
            {
                "layer": layer,
                "file": shard_name,
                "sha256": sha256_file(shard_path),
                "physical_bytes": shard_path.stat().st_size,
                "logical_tensor_bytes": layer_bytes,
                "tensor_count": len(tensors),
                "source_keys": clone_sources,
            }
        )
        clone_bytes += layer_bytes
        clone_tensors += len(tensors)
        print(
            json.dumps(
                {
                    "event": "clone_layer",
                    "kind": args.kind,
                    "layer": layer,
                    "logical_tensor_bytes": layer_bytes,
                    "tensor_count": len(tensors),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del tensors
        gc.collect()

    expected_tensor_count = NUM_LAYERS * CLONES_PER_LAYER * len(
        _EXPECTED_SUFFIXES[args.kind]
    )
    if clone_tensors != expected_tensor_count:
        raise RuntimeError("expanded checkpoint clone tensor count mismatch")

    expanded_config = dict(config)
    expanded_config["n_routed_experts"] = TOTAL_EXPERTS
    expanded_config["num_nextn_predict_layers"] = 0
    expanded_config["yeto_routed_expert_clone"] = contract.config_value()
    _write(temporary / "config.json", _canonical_json(expanded_config))

    expanded_index = dict(index)
    expanded_index["metadata"] = dict(index["metadata"])
    expanded_index["metadata"]["total_size"] = base_bytes + clone_bytes
    expanded_index["weight_map"] = dict(sorted(new_weight_map.items()))
    _write(
        temporary / "model.safetensors.index.json",
        _canonical_json(expanded_index),
    )

    manifest = {
        **plan,
        "clone_logical_tensor_bytes": clone_bytes,
        "expanded_logical_tensor_bytes": base_bytes + clone_bytes,
        "clone_tensor_count": clone_tensors,
        "clone_shards": clone_shards,
        "output_config_sha256": sha256_file(temporary / "config.json"),
        "output_index_sha256": sha256_file(
            temporary / "model.safetensors.index.json"
        ),
        "output_weight_map_entries": len(new_weight_map),
    }
    _write(
        temporary / "yeto_expert_clone_manifest.json",
        _canonical_json(manifest),
    )
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "status": "complete",
                "kind": args.kind,
                "output": str(args.output),
                "manifest_sha256": sha256_file(
                    args.output / "yeto_expert_clone_manifest.json"
                ),
                "config_sha256": manifest["output_config_sha256"],
                "index_sha256": manifest["output_index_sha256"],
                "clone_logical_tensor_bytes": clone_bytes,
                "clone_tensor_count": clone_tensors,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
