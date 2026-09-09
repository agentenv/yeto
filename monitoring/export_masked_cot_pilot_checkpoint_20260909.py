"""CPU-only conversion of a pinned masked-CoT pilot checkpoint to HF safetensors.

The checkpoint already has HF tensor names. Preserve its tensor values and
dtypes exactly; the unused MTP auxiliary module was not trained and is copied
from the original model solely for a complete HF artifact. Eval must disable
speculative decoding. This never loads or writes optimizer state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


PINNED_ASSETS = {
    "config.json": "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab",
    "tokenizer.json": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
    "tokenizer_config.json": "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27",
    "chat_template.jinja": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--expected-metadata-sha256", required=True)
    parser.add_argument("--completed-updates", required=True, type=int)
    parser.add_argument("--epoch", type=int, default=0)
    args = parser.parse_args(argv)
    if not args.source.is_absolute() or not args.destination.is_absolute():
        parser.error("Source and destination must be absolute")
    if (len(args.expected_metadata_sha256) != 64
            or any(c not in "0123456789abcdef" for c in args.expected_metadata_sha256)):
        parser.error("Expected metadata SHA256 must be lowercase hexadecimal")
    if args.completed_updates < 1 or args.epoch < 0:
        parser.error("Expected completed updates must be positive and epoch nonnegative")
    if args.source.resolve() == args.destination.resolve() or args.source.resolve() in args.destination.resolve().parents:
        parser.error("Destination must be separate from the source checkpoint")
    return args


def main(argv=None):
    args = parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    import torch.distributed.checkpoint as dcp
    from safetensors import safe_open
    from safetensors.torch import save_file

    torch.set_num_threads(8)
    source = args.source.resolve()
    original = Path("/data/sft_baseline_20260908/models/Qwen3.8-27B")
    dest = args.destination.resolve()
    partial = dest.with_name(dest.name + ".incomplete")
    expected_metadata = args.expected_metadata_sha256
    if not source.is_dir() or (source / ".incomplete").exists():
        raise ValueError("Source checkpoint is absent or still incomplete")
    if dest == original or original in dest.parents:
        raise ValueError("Destination must not modify the original model")
    if sha(source / "model/.metadata") != expected_metadata:
        raise ValueError("Source model metadata differs from its pinned digest")
    if dest.exists() or partial.exists():
        raise ValueError("Destination or incomplete export already exists; preserve prior artifacts")
    expected_scheduler = {"step": args.completed_updates, "epoch": args.epoch}
    if torch.load(source / "step_scheduler.pt", weights_only=True, map_location="cpu") != expected_scheduler:
        raise ValueError("Source scheduler does not match expected completed updates and epoch")
    for filename, expected in PINNED_ASSETS.items():
        if sha(original / filename) != expected:
            raise ValueError("Original Qwen asset identity changed: " + filename)
    original_index_sha256 = sha(original / "model.safetensors.index.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir()
    started = time.time()
    reader = dcp.FileSystemReader(source / "model")
    metadata = reader.read_metadata()
    index = json.loads((original / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    checkpoint_keys = set(metadata.state_dict_metadata)
    missing = set(weight_map) - checkpoint_keys
    assert len(checkpoint_keys) == 1184 and len(missing) == 15
    assert all(key.startswith("mtp.") for key in missing)
    assert not checkpoint_keys - set(weight_map)

    # Compare the complete checkpoint inventory with the pinned base model.
    for filename in sorted(set(weight_map.values())):
        with safe_open(original / filename, framework="pt", device="cpu") as f:
            for key in f.keys():
                assert weight_map[key] == filename
                if key in checkpoint_keys:
                    assert list(f.get_slice(key).get_shape()) == list(metadata.state_dict_metadata[key].size), key

    state = {key: torch.empty(tuple(value.size), dtype=value.properties.dtype, device="cpu")
             for key, value in metadata.state_dict_metadata.items()}
    print(json.dumps({"phase": "loading_cpu", "checkpoint_tensors": len(state)}), flush=True)
    dcp.load(state, storage_reader=reader, no_dist=True)
    assert not torch.cuda.is_initialized()
    for filename in sorted(set(weight_map[key] for key in missing)):
        with safe_open(original / filename, framework="pt", device="cpu") as f:
            for key in sorted(missing):
                if weight_map[key] == filename:
                    state[key] = f.get_tensor(key)

    output_files = []
    total_size = 0
    for filename in sorted(set(weight_map.values())):
        tensors = {key: value for key, value in state.items() if weight_map[key] == filename}
        assert all(t.device.type == "cpu" and t.is_contiguous() for t in tensors.values())
        assert all(torch.isfinite(t).all().item() for t in tensors.values())
        save_file(tensors, partial / filename, metadata={"format": "pt"})
        # Exact readback checks tensor values/dtypes, not just file presence.
        with safe_open(partial / filename, framework="pt", device="cpu") as f:
            assert set(f.keys()) == set(tensors)
            for key, value in tensors.items():
                actual = f.get_tensor(key)
                assert actual.dtype == value.dtype and torch.equal(actual, value), key
                total_size += value.numel() * value.element_size()
        output_files.append({"path": filename, "size": (partial / filename).stat().st_size,
                             "sha256": sha(partial / filename)})
        print(json.dumps({"phase": "saved_verified_shard", "file": filename}), flush=True)

    assets = []
    for filename in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                     "chat_template.jinja", "preprocessor_config.json", "processor_config.json",
                     "video_preprocessor_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"):
        path = original / filename
        if path.is_file():
            shutil.copyfile(path, partial / filename)
            assert sha(path) == sha(partial / filename)
            assets.append({"path": filename, "sha256": sha(path)})
    assert {"config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"} <= {a["path"] for a in assets}
    index["metadata"]["total_size"] = total_size
    (partial / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    assert sha(source / "model/.metadata") == expected_metadata
    if (source / ".incomplete").exists() or sha(original / "model.safetensors.index.json") != original_index_sha256:
        raise ValueError("Source checkpoint completeness or original index changed during conversion")
    if torch.load(source / "step_scheduler.pt", weights_only=True, map_location="cpu") != expected_scheduler:
        raise ValueError("Source scheduler changed during conversion")
    for filename, expected in PINNED_ASSETS.items():
        if sha(original / filename) != expected:
            raise ValueError("Original Qwen asset changed during conversion: " + filename)
    receipt = {"schema": "qwen-masked-cot-pilot-cpu-hf-export/v1", "status": "complete",
               "checkpoint_step_zero_based": args.completed_updates - 1,
               "completed_optimizer_updates": args.completed_updates, "epoch": args.epoch,
               "original_model": str(original), "original_index_sha256": original_index_sha256,
               "purpose": "finite-pilot-inference-qualification", "source": str(source), "source_metadata_sha256": expected_metadata,
               "checkpoint_tensor_count": len(checkpoint_keys), "untrained_base_auxiliary_keys": sorted(missing),
               "speculative_decoding_allowed": False, "output_tensor_count": len(state), "files": output_files,
               "assets": assets, "index_sha256": sha(partial / "model.safetensors.index.json"),
               "exact_tensor_readback": True, "gpu_used": False, "seconds": time.time() - started,
               "code_sha256": sha(Path(__file__))}
    (partial / "export-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    partial.rename(dest)
    print(json.dumps({"status": "complete", "path": str(dest), "seconds": receipt["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
