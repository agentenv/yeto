"""Full-run entry point for user-authorized generated-CoT masked Qwen SFT.

This delegates the qualified NeMo lifecycle without changing any baseline file.
Only this Python process temporarily binds the new dataset validators/identity.
Run with torchrun in the pinned image after separately reserving an idle host.
Explicit same-contract checkpoint resume is supported; no service management is performed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import threading

from training.qwen38_no_cot import nemo_train as baseline
from . import masked_data as data, masked_recipe as recipe

VERSION = "qwen38-generated-cot-masked-turn-boundary-runner/v2"
BASE_MODEL_DIR = Path("/data/sft_baseline_20260908/models/Qwen3.8-27B")
MODEL_CONFIG_SHA256 = "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"
MODEL_INDEX_SHA256 = "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"
RUNTIME_IMAGE = "sha256:b0bd0e50b29cb0a3ec69c3cbbb49652d3c1aabedbfe22ee0fa6d053b6bb33aee"
BASELINE_ENTRY_SHA256 = "2be1d6d89e0922be5520faa82bcad14a1cb5ad2f277e100d0095b41068a2e5df"
PHASES = {"train"}
_ORIGINAL_REQUIRE = baseline.require_recipe
_ORIGINAL_IDENTITY = baseline.contract_identity
_DELEGATION_LOCK = threading.Lock()


def verify_base_assets(config):
    """Pin original downloaded model location/metadata, never a saved SFT run.

    Weight shards must exist under the pinned original directory. This verifies
    the pinned config/index and file locations, not a new full-byte weight audit.
    """
    model = config["model"]
    path = Path(model["pretrained_model_name_or_path"])
    if (not path.is_absolute() or path != BASE_MODEL_DIR or path.resolve(strict=True) != BASE_MODEL_DIR
            or model.get("_target_") != "nemo_automodel.NeMoAutoModelForImageTextToText.from_pretrained"
            or model.get("local_files_only") is not True or model.get("trust_remote_code") is not False
            or any(key in model for key in ("state_dict", "from_tf", "from_flax", "adapter_name_or_path"))):
        raise ValueError("This fresh-base runner only loads the pinned original Qwen model directory")
    if (data.digest_file(path / "config.json") != MODEL_CONFIG_SHA256
            or data.digest_file(path / "model.safetensors.index.json") != MODEL_INDEX_SHA256):
        raise ValueError("Pinned base model config or weight-index identity changed")
    index = json.loads((path / "model.safetensors.index.json").read_bytes())
    mapping = index.get("weight_map")
    if not isinstance(mapping, dict) or not mapping or any(not isinstance(v, str) for v in mapping.values()):
        raise ValueError("Pinned base model index has no valid weight map")
    shards = sorted(set(mapping.values()))
    for name in shards:
        shard = path / name
        if (Path(name).name != name or not name.endswith(".safetensors")
                or not shard.is_file() or shard.resolve(strict=True) != shard
                or shard.stat().st_size < 1):
            raise ValueError("Every original base weight shard must exist inside the pinned model directory")
    if os.environ.get("YETA_TRAINING_IMAGE") != RUNTIME_IMAGE:
        raise ValueError("The separate runner requires the exact qualified NeMo image identity")
    if data.digest_file(baseline.__file__) != BASELINE_ENTRY_SHA256:
        raise ValueError("The delegated baseline lifecycle source changed; requalify it first")
    return {"model_config_sha256": MODEL_CONFIG_SHA256,
            "model_weight_index_sha256": MODEL_INDEX_SHA256,
            "runtime_image": RUNTIME_IMAGE, "base_weight_shards": len(shards),
            "weight_validation": "pinned-config-index-original-location-and-shard-presence/v1",
            "full_weight_bytes_rehashed": False}


def assert_output(config, config_path, resume=None):
    checkpoint = Path(config["checkpoint"]["checkpoint_dir"])
    if not checkpoint.is_absolute() or checkpoint.name != "checkpoints":
        raise ValueError("Use an absolute new run/checkpoints directory")
    root = checkpoint.parent
    if root == BASE_MODEL_DIR or root in BASE_MODEL_DIR.parents or BASE_MODEL_DIR in root.parents:
        raise ValueError("Training output must be separate from the original model")
    if root.exists() and (not root.is_dir() or root.resolve() != root):
        raise ValueError("Training output cannot alias an existing location")
    if Path(config["wandb"]["dir"]) != root:
        raise ValueError("Metrics and checkpoints must belong to the same run")
    if resume is not None:
        if checkpoint.is_symlink() or not checkpoint.is_dir() or checkpoint.resolve() != checkpoint:
            raise ValueError("Explicit resume requires this run's canonical checkpoint directory")
        return  # Exact dataset/model/runtime identity and cursor are checked before GPU work.
    if checkpoint.is_symlink() or (checkpoint.exists() and (not checkpoint.is_dir() or any(checkpoint.iterdir()))):
        raise ValueError("Fresh-base training refuses an occupied checkpoint directory")
    if Path(config["wandb"]["dir"]) != root:
        raise ValueError("Metrics and checkpoints must belong to the same fresh run")
    # A launcher may already have copied this config, its launch receipt and a
    # compilation cache. Existing training artifacts do not qualify as fresh.
    allowed = {"checkpoints", "launch.json", "triton-cache"}
    config_path = Path(config_path).resolve()
    if config_path.parent == root:
        allowed.add(config_path.name)
    if root.exists() and any(path.name not in allowed or path.is_symlink() for path in root.iterdir()):
        raise ValueError("Existing run artifacts require a different fresh output directory")


def _check_phase(config, mode):
    if mode not in {"preflight", "train"} or config.get("experiment_phase") != "train":
        raise ValueError("This experiment uses one full shuffled training pass")
    return mode


def prepare_run(config, *, mode, config_path, resume=None):
    recipe.require_recipe(config)
    forwarded_mode = _check_phase(config, mode)
    assert_output(config, config_path, resume)
    assets = verify_base_assets(config)
    bound = recipe.contract_identity(config)
    counts = {}
    runtime_counts = {}
    for name in ("dataset", "validation_dataset"):
        if name not in config:
            continue
        item = config[name]
        # Check complete supplied manifest membership, without smoke filters.
        checked = data.GeneratedMaskedTokenDataset(item["path_or_dataset"], index_path=item["index_path"],
            index_sha256=item["index_sha256"], split=item["split"], seq_len=item["seq_len"],
            require_provenance=item["require_provenance"])
        counts[item["split"]] = len(checked)
        # Reuse the validated index lengths instead of rehashing every shard
        # a second time merely to report the validation subset size.
        retained = sum(1 for ref in checked.refs if item.get("max_input_tokens") is None
                       or ref[4] <= item["max_input_tokens"])
        runtime_counts[item["split"]] = min(retained, item.get("max_samples") or retained)
    batch = config["step_scheduler"]["global_batch_size"]
    rows = counts.get("train", 0)
    if rows < 1:
        raise ValueError("The full run needs a nonempty training split")
    resume_receipt = None
    if resume is not None:
        from training.qwen38_no_cot.nemo_checkpoint import resume_identity, resolve_explicit_resume
        resume_receipt = resolve_explicit_resume(config["checkpoint"]["checkpoint_dir"], resume,
                                                resume_identity(config, contract_identity(config)))
    return {"schema": VERSION, "mode": mode, "forwarded_mode": forwarded_mode,
            "dataset_rows": counts,
            "runtime_dataset_rows": runtime_counts,
            "expected_optimizer_updates": (rows + batch - 1) // batch,
            "final_partial_window_microbatches": rows % batch,
            "manifest_sha256": bound["manifest_sha256"], "base_assets": assets,
            "training_contract": data.TRAINING_CONTRACT,
            "semantic_quality_qualified": False, "unreviewed_cot_included": True,
            "resume_requested": resume is not None, "resume_receipt": resume_receipt,
            "gpu_runtime_qualified": False, "training_started": False}


def contract_identity(config):
    """Keep the qualified lifecycle identity and replace all no-CoT semantics."""
    bound = recipe.contract_identity(config)
    assets = verify_base_assets(config)
    result = _ORIGINAL_IDENTITY(config)
    project = Path(__file__).resolve().parents[2]
    files = ["training/qwen38_turn_boundary_v2/" + name for name in (
        "masked_train.py", "masked_recipe.py", "masked_data.py", "masked.py", "normalize.py",
        "source_adapters.py", "prepare_masked_full.py", "masked_replay.py")]
    files += ["training/qwen38_cot_experimental/" + name for name in (
        "train.py", "recipe.py", "data.py", "render.py", "prepare_full.py", "generation_contract.py", "replay.py")]
    files += ["cot_filler/" + name for name in (
        "core.py", "corpus_worker.py", "regeneration.py", "grounding_review.py",
        "grounding_review_ids.py", "grounding_review_v3.py", "grounding_review_v4.py", "grounding_review_fast.py")]
    result.update(training_contract=data.TRAINING_CONTRACT, runner_version=VERSION,
                  mask_policy=data.MASK_POLICY, sequence_policy=data.SEQUENCE_POLICY,
                  thinking_mode="xhigh", reasoning_loss=0, initialization="pinned_base_or_explicit_same_contract_resume",
                  filled_cot_contract=bound, base_assets=assets,
                  filled_cot_code_sha256={name: data.digest_file(project / name) for name in files})
    return result


@contextmanager
def bound_baseline(config, config_path, mode, resume=None):
    """Temporarily bind only this process; reject nested/concurrent delegation."""
    if not _DELEGATION_LOCK.acquire(blocking=False):
        raise RuntimeError("A baseline delegation is already active in this process")
    old_require, old_identity = baseline.require_recipe, baseline.contract_identity
    expected = None

    def require_bound(actual):
        if json.dumps(actual, sort_keys=True, separators=(",", ":"), allow_nan=False) != expected:
            raise ValueError("Training configuration changed after wrapper validation")
        recipe.require_recipe(actual)
        _check_phase(actual, mode)
        assert_output(actual, config_path, resume)

    try:
        expected = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if (old_require is not _ORIGINAL_REQUIRE or old_identity is not _ORIGINAL_IDENTITY
                or recipe.require_baseline_runtime is not _ORIGINAL_REQUIRE):
            raise RuntimeError("Delegation requires the original captured baseline guards")
        baseline.require_recipe = require_bound
        baseline.contract_identity = contract_identity
        yield
    finally:
        baseline.require_recipe, baseline.contract_identity = old_require, old_identity
        _DELEGATION_LOCK.release()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("preflight", "train"), default="preflight")
    parser.add_argument("--short-receipt")
    parser.add_argument("--long-receipt")
    parser.add_argument("--resident-long-receipt")
    parser.add_argument("--validate-memory-repair-in-production", action="store_true")
    parser.add_argument("--resume", help="Explicit LATEST or same-run checkpoint; exact identity and cursor must match")
    args = parser.parse_args(argv)
    config_path = Path(args.config).resolve(strict=True)
    config = json.loads(config_path.read_bytes())
    prepared = prepare_run(config, mode=args.mode, config_path=config_path,
                           resume=args.resume)
    forwarded_mode = prepared["forwarded_mode"]
    if forwarded_mode == "train" and os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True":
        parser.error("Full training requires the qualified expandable-segments allocator setting")
    if forwarded_mode == "train" and not (args.resident_long_receipt or args.validate_memory_repair_in_production):
        parser.error("Train requires its matching resident receipt or explicit production memory validation")
    forwarded = [str(Path(__file__).resolve()), "--config", str(config_path), "--mode", forwarded_mode]
    for name in ("short_receipt", "long_receipt", "resident_long_receipt"):
        value = getattr(args, name)
        if value:
            forwarded.extend(("--" + name.replace("_", "-"), value))
    if args.validate_memory_repair_in_production:
        forwarded.append("--validate-memory-repair-in-production")
    if args.resume is not None:
        forwarded.extend(("--resume", args.resume))
    print(json.dumps(prepared, sort_keys=True), flush=True)
    old_argv = sys.argv
    try:
        with bound_baseline(config, config_path, args.mode, args.resume):
            sys.argv = forwarded
            return baseline.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
