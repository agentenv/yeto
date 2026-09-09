"""Run corrected no-CoT data through the unchanged proven training lifecycle."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import threading

from training.qwen38_no_cot import nemo_train as baseline
from . import data, recipe

_ORIGINAL_REQUIRE, _ORIGINAL_IDENTITY = baseline.require_recipe, baseline.contract_identity
_LOCK = threading.Lock()
MODEL_CONFIG_SHA = "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"
MODEL_INDEX_SHA = "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"


def verify_base(config):
    path = Path(config["model"]["pretrained_model_name_or_path"]).resolve()
    if path != Path("/data/sft_baseline_20260908/models/Qwen3.8-27B"):
        raise ValueError("Corrected restart requires the pinned original base location")
    if data.digest_file(path / "config.json") != MODEL_CONFIG_SHA or data.digest_file(path / "model.safetensors.index.json") != MODEL_INDEX_SHA:
        raise ValueError("Pinned original model config/index changed")
    weights = json.loads((path / "model.safetensors.index.json").read_bytes())["weight_map"]
    if not weights or any(not (path / filename).is_file() for filename in set(weights.values())):
        raise ValueError("Original base weight shards are missing")
    return {"config_sha256": MODEL_CONFIG_SHA, "weight_index_sha256": MODEL_INDEX_SHA,
            "weight_validation": "original-pinned-location-config-index-shard-presence", "initialization": "fresh-base"}


def contract_identity(config):
    result = json.loads(json.dumps(_ORIGINAL_IDENTITY(config), sort_keys=True, allow_nan=False))
    folder = Path(__file__).parent
    files = ("normalize.py", "source_adapters.py", "render.py", "data.py", "recipe.py", "train.py", "prepare_data.py")
    result.update(training_contract=data.CONTRACT, thinking_mode="none", initialization="pinned-base-or-explicit-same-contract-resume",
        source_turn_code_sha256={name: data.digest_file(folder / name) for name in files},
        dataset_manifest_sha256=data.digest_file(config["dataset"]["path_or_dataset"]),
        qualified_index_sha256=config["dataset"]["index_sha256"], base_assets=verify_base(config),
        rng_config=config["rng"], loss_config=config["loss_fn"], lr_scheduler_config=config["lr_scheduler"],
        clip_config=config["clip_grad_norm"], checkpoint_config=config["checkpoint"],
        step_scheduler_config=config["step_scheduler"])
    return json.loads(json.dumps(result, sort_keys=True, allow_nan=False))


def prepare_run(config, *, config_path, resume=None):
    recipe.require_recipe(config)
    checkpoint_root = Path(config["checkpoint"]["checkpoint_dir"]).resolve()
    if Path(config_path).resolve().parent != checkpoint_root.parent:
        raise ValueError("Config must be inside its dedicated corrected run directory")
    if checkpoint_root.exists() and any(checkpoint_root.iterdir()) and resume is None:
        raise ValueError("Select a fresh corrected run directory or explicitly resume this same corrected contract")
    assets = verify_base(config)
    counts = {}
    for key in ("dataset", "validation_dataset"):
        if key not in config: continue
        item = config[key]
        checked = data.TurnTokenDataset(item["path_or_dataset"], index_path=item["index_path"], index_sha256=item["index_sha256"],
                                       split=item["split"], seq_len=item["seq_len"], require_provenance=True)
        counts[item["split"]] = len(checked)
    return {"schema": "corrected-no-cot-preflight/v2", "training_contract": data.CONTRACT, "dataset_rows": counts,
            "base_assets": assets, "training_started": False, "gpu_runtime_qualified": False,
            "resume_requested": resume is not None}


@contextmanager
def bound_baseline(config):
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError("Training lifecycle delegation is already active")
    old_require, old_identity = baseline.require_recipe, baseline.contract_identity
    expected = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    def require_bound(actual):
        if json.dumps(actual, sort_keys=True, separators=(",", ":"), allow_nan=False) != expected:
            raise ValueError("Training configuration changed after corrected-run validation")
        recipe.require_recipe(actual)
    try:
        if old_require is not _ORIGINAL_REQUIRE or old_identity is not _ORIGINAL_IDENTITY:
            raise RuntimeError("Unexpected training lifecycle replacement")
        baseline.require_recipe = require_bound
        baseline.contract_identity = contract_identity
        yield
    finally:
        baseline.require_recipe, baseline.contract_identity = old_require, old_identity
        _LOCK.release()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("preflight", "train"), default="preflight")
    parser.add_argument("--validate-memory-repair-in-production", action="store_true")
    parser.add_argument("--resume")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_bytes())
    prepared = prepare_run(config, config_path=args.config, resume=args.resume)
    if args.mode == "train" and (os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True"
                                 or not args.validate_memory_repair_in_production):
        raise ValueError("Train requires proven allocator settings and explicit production memory validation")
    forwarded = [str(Path(__file__).resolve()), "--config", args.config, "--mode", args.mode]
    if args.validate_memory_repair_in_production: forwarded.append("--validate-memory-repair-in-production")
    if args.resume is not None: forwarded.extend(("--resume", args.resume))
    print(json.dumps(prepared, sort_keys=True), flush=True)
    old_argv = sys.argv
    try:
        with bound_baseline(config):
            sys.argv = forwarded
            return baseline.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
