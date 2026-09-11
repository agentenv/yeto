"""Generate a distinct experimental generated-CoT masked recipe, without launching a trainer.

The baseline recipe supplies the unchanged model/optimizer/topology settings.
This recipe requires its own manifest, index, output and training contract; the
no-CoT entry point and no-CoT checkpoints do not qualify this experiment.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from training.qwen38_no_cot.nemo_recipe import make_recipe as baseline_recipe
from training.qwen38_no_cot.nemo_train import require_recipe as require_baseline_runtime
from .masked_data import (
    BASE_ASSET_RECEIPT, TRAINING_CONTRACT, GeneratedMaskedTokenDataset, digest_file,
    _strict_json, read_complete, read_manifest, validator_identity,
)

DATASET_TARGET = "training.qwen38_native_gap_v3.masked_data.GeneratedMaskedTokenDataset"
COLLATOR_TARGET = "training.qwen38_native_gap_v3.masked_data.collate_exact"
WANDB_NAME_PREFIX = "qwen38-27b-generated-cot-masked-native-gap-v4-"
WANDB_PROJECT = "yeto-h200"
WANDB_ENTITY = "yeta"
RUNTIME_COMPATIBILITY_SCHEMA = "qwen38-native-gap-v4-dataset-runtime-compatibility/v1"


def _require_runtime_compatibility(contract):
    value = contract.get("runtime_compatibility")
    if value is None:
        return None
    keys = {
        "schema", "receipt_path", "receipt_sha256", "dataset_tree_sha256",
        "dataset_build_receipt_sha256", "build_code_manifest_sha256",
        "runtime_code_manifest_sha256", "conversion_dependency_map_sha256",
        "cp_guard_probe_path", "cp_guard_probe_sha256",
        "cp_guard_nested_receipt_sha256",
    }
    def sha256(value):
        return (isinstance(value, str) and len(value) == 64
                and all(character in "0123456789abcdef" for character in value))
    if (not isinstance(value, dict) or set(value) != keys
            or value.get("schema") != RUNTIME_COMPATIBILITY_SCHEMA
            or not isinstance(value.get("receipt_path"), str)
            or not isinstance(value.get("cp_guard_probe_path"), str)
            or any(not sha256(value.get(key))
                   for key in keys - {"schema", "receipt_path", "cp_guard_probe_path"})
            or value.get("build_code_manifest_sha256") ==
               value.get("runtime_code_manifest_sha256")):
        raise ValueError("Dataset/runtime compatibility contract is malformed")
    path = Path(value.get("receipt_path", ""))
    if (not path.is_absolute() or path.is_symlink() or not path.is_file()
            or path.stat().st_mode & 0o222 or digest_file(path) != value["receipt_sha256"]):
        raise ValueError("Dataset/runtime compatibility receipt changed")
    receipt = _strict_json(path.read_bytes())
    if (receipt.get("schema") != RUNTIME_COMPATIBILITY_SCHEMA
            or receipt.get("status") != "passed"
            or receipt.get("dataset_tree_sha256") != value["dataset_tree_sha256"]
            or receipt.get("dataset_build_receipt_sha256") !=
               value["dataset_build_receipt_sha256"]
            or receipt.get("old_code_manifest_sha256") !=
               value["build_code_manifest_sha256"]
            or receipt.get("new_code_manifest_sha256") !=
               value["runtime_code_manifest_sha256"]
            or receipt.get("conversion_dependency_map_sha256") !=
               value["conversion_dependency_map_sha256"]
            or receipt.get("cp_guard_probe_path") != value["cp_guard_probe_path"]
            or receipt.get("cp_guard_probe_sha256") != value["cp_guard_probe_sha256"]
            or receipt.get("cp_guard_nested_receipt_sha256") !=
               value["cp_guard_nested_receipt_sha256"]
            or receipt.get("exact_runtime_only_delta") is not True
            or receipt.get("full_dataset_tree_rehashed") is not True):
        raise ValueError("Dataset/runtime compatibility receipt has the wrong claims")
    return value


def require_recipe(config):
    if config.get("training_contract") != TRAINING_CONTRACT:
        raise ValueError("This is not an experimental generated-CoT masked experiment")
    if config.get("runtime_patches", {}).get("dataset_contract") != TRAINING_CONTRACT:
        raise ValueError("The explicit generated-CoT runtime guard is required")
    # Keep all baseline model/memory/optimizer guardrails, but do not weaken its
    # exact patch identity. The unchanged baseline entry point rejects this new
    # marker, preventing it from mislabelling generated-CoT data as no-CoT.
    common = deepcopy(config)
    common["runtime_patches"].pop("dataset_contract")
    require_baseline_runtime(common)
    reference = baseline_recipe(model_dir=config["model"]["pretrained_model_name_or_path"],
        manifest=config["dataset"]["path_or_dataset"],
        output_dir=str(Path(config["checkpoint"]["checkpoint_dir"]).parent), phase="train",
        wandb_project=config["wandb"]["project"], seed=config.get("rng", {}).get("seed", 20260908),
        global_batch_size=8)
    reference["runtime_patches"]["dataset_contract"] = TRAINING_CONTRACT
    reference["dataloader"]["collate_fn"] = COLLATOR_TARGET
    for key in ("model", "processor", "optimizer", "loss_fn", "lr_scheduler", "clip_grad_norm",
                "freeze_config", "distributed", "dist_env", "dataloader", "rng", "runtime_patches", "step_scheduler"):
        if config.get(key) != reference[key]:
            raise ValueError("Qualified training setting changed: " + key)
    if ({k:v for k,v in config["checkpoint"].items() if k != "checkpoint_dir"}
            != {k:v for k,v in reference["checkpoint"].items() if k != "checkpoint_dir"}):
        raise ValueError("Qualified checkpoint policy changed or implicit restore was supplied")
    if config.get("checkpoint", {}).get("restore_from"):
        raise ValueError("Do not put restore_from in the recipe; use explicit --resume with identity validation")
    if config.get("peft") or config.get("quantization"):
        raise ValueError("The requested experiment is full-parameter SFT")
    for key, expected in {"strategy": "fsdp2", "cp_size": 8, "tp_size": 1, "pp_size": 1}.items():
        if config.get("distributed", {}).get(key) != expected:
            raise ValueError("The recipe requires the qualified one-host CP8/FSDP2 topology")
    memory_settings = {"activation_checkpointing": True, "reshard_after_forward": True,
                       "enable_fsdp2_prefetch": False, "defer_fsdp_grad_sync": False,
                       "sequence_parallel": False, "ep_size": 1}
    for key, value in memory_settings.items():
        actual = config.get("distributed", {}).get(key)
        if (actual is not value if type(value) is bool else actual != value):
            raise ValueError("The full run must retain the qualified memory setting: " + key)
    if config.get("step_scheduler", {}).get("local_batch_size") != 1:
        raise ValueError("Long-context memory qualification uses one local sequence")
    if config.get("step_scheduler", {}).get("global_batch_size") != 8:
        raise ValueError("Preserve the qualified eight-sequence accumulation schedule")
    text = config.get("model", {}).get("text_config", {})
    if (text.get("use_cache") is not False or text.get("num_nextn_predict_layers") != 0
            or text.get("mtp_num_hidden_layers") != 0):
        raise ValueError("KV cache and MTP must remain disabled during SFT")
    if config.get("optimizer", {}).get("lr") != 1e-5:
        raise ValueError("The requested learning rate is 1e-5")
    if config.get("freeze_config", {}).get("freeze_language_model") is not False:
        raise ValueError("All text parameters must remain trainable")
    if config.get("step_scheduler", {}).get("num_epochs") != 1:
        raise ValueError("The recipe consumes one shuffled pass")
    if config.get("experiment_phase") != "train":
        raise ValueError("The experimental full run cannot be shortened into a smoke phase")
    manifest_sha256 = digest_file(config.get("dataset", {}).get("path_or_dataset", ""))
    expected_wandb_name = WANDB_NAME_PREFIX + manifest_sha256[:12] + "-train"
    if (config.get("wandb", {}).get("name") != expected_wandb_name
            or config.get("wandb", {}).get("project") != WANDB_PROJECT
            or config.get("wandb", {}).get("entity") != WANDB_ENTITY):
        raise ValueError("W&B run name must bind the corrected native-gap manifest identity")
    if config.get("dataset", {}).get("_target_") != DATASET_TARGET:
        raise ValueError("A no-CoT dataset must not enter the generated-CoT recipe")
    if config.get("dataloader", {}).get("collate_fn") != COLLATOR_TARGET:
        raise ValueError("The exact once-shifted generated-CoT collator is required")
    for key in ("dataset", "validation_dataset"):
        dataset = config.get(key)
        if dataset is None:
            continue
        if (dataset.get("_target_") != DATASET_TARGET or dataset.get("seq_len") != 262144
                or dataset.get("require_provenance") is not True
                or not dataset.get("index_path") or not dataset.get("index_sha256")
                or not dataset.get("complete_path") or not dataset.get("complete_sha256")):
            raise ValueError("Every dataset requires the audited index and atomic COMPLETE identity")
    if config.get("validation_dataset") is not None:
        validation = config["validation_dataset"]
        if any(validation.get(key) != config['dataset'].get(key) for key in (
                'path_or_dataset','index_path','index_sha256','complete_path','complete_sha256',
                'seq_len','require_provenance')):
            raise ValueError('Training and heldout validation require the same immutable manifest/index')
        if (validation.get("split") != "validation" or validation.get("max_samples") != 32
                or validation.get("max_input_tokens") != 32768
                or config.get("validation_dataloader") != {**reference["validation_dataloader"], "collate_fn": COLLATOR_TARGET}):
            raise ValueError("Preserve the fixed bounded heldout validation contract")
    if config.get("experiment_phase") == "train":
        dataset, scheduler = config["dataset"], config["step_scheduler"]
        if (any(key in dataset for key in ("max_input_tokens", "max_samples"))
                or dataset.get("order", "source") != "source"
                or config["dataloader"].get("shuffle") is not True
                or scheduler.get("max_steps") is not None):
            raise ValueError("Production consumes the complete supplied manifest in one shuffled pass")
    if config.get("runtime_patches") != {
            "cp_memory": "contiguous_gather_local_reorder/v1",
            "activation_checkpointing": "native_qwen_whole_block/v1",
            "dataset_contract": TRAINING_CONTRACT}:
        raise ValueError("Both qualified memory repairs are required")
    checkpoint = config.get("checkpoint", {})
    if any(checkpoint.get(key) != value for key, value in {
            "enabled": True, "is_async": False, "cpu_offload": True, "max_recent_checkpoints": 3,
            "model_save_format": "torch_save", "save_consolidated": False}.items()):
        raise ValueError("Preserve the existing full checkpoint policy")
    contract = config.get("dataset_contract", {})
    if contract.get("base_asset_receipt") != BASE_ASSET_RECEIPT:
        raise ValueError("The recipe must bind the durable full-byte base-model receipt")
    _require_runtime_compatibility(contract)
    manifest = read_manifest(config["dataset"]["path_or_dataset"])
    if (manifest.get("full_export") is not True
            or manifest.get("unexpected_trace_failures") != 0
            or manifest.get("trace_source_membership", {}).get("full_frozen_coverage_required") is not True
            or manifest.get("all_train") is not True
            or manifest.get("internal_validation") is not False
            or set(manifest.get("splits", {})) != {"train"}):
        raise ValueError("Production training requires the complete frozen source export")
    read_complete(config["dataset"]["complete_path"], config["dataset"]["complete_sha256"],
                  manifest_path=config["dataset"]["path_or_dataset"],
                  index_path=config["dataset"]["index_path"],
                  index_sha256=config["dataset"]["index_sha256"])
    return config


def make_recipe(*, model_dir, manifest, index_path, index_sha256, output_dir,
                phase="train", wandb_project=WANDB_PROJECT,
                wandb_entity=WANDB_ENTITY, seed=20260908, global_batch_size=8):
    """Audit CPU data identities and return configuration for a fresh run only."""
    if phase != "train":
        raise ValueError("This user-authorized experiment consumes the full manifest, without another smoke gate")
    output = Path(output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Use a fresh generated-CoT output; existing baseline/checkpoints are forbidden")
    supplied_manifest, supplied_index = Path(manifest), Path(index_path)
    if supplied_manifest.is_symlink() or supplied_index.is_symlink():
        raise ValueError("Manifest and index cannot be symlinks")
    manifest = supplied_manifest.resolve(strict=True)
    index_path = supplied_index.resolve(strict=True)
    complete_path = manifest.parent / "COMPLETE.json"
    complete_sha256 = digest_file(complete_path)
    contents = read_manifest(manifest)
    if (contents.get("full_export") is not True
            or contents.get("unexpected_trace_failures") != 0
            or contents.get("trace_source_membership", {}).get("full_frozen_coverage_required") is not True
            or contents.get("all_train") is not True
            or contents.get("internal_validation") is not False
            or set(contents.get("splits", {})) != {"train"}):
        raise ValueError("Production training requires the complete frozen source export")
    # Opening verifies the full immutable shard hashes and index provenance;
    # no model/tokenizer object is constructed and no row is retokenized.
    for split in contents["splits"]:
        GeneratedMaskedTokenDataset(manifest, index_path=index_path, index_sha256=index_sha256,
                                    complete_path=complete_path, complete_sha256=complete_sha256,
                                    split=split)
    config = baseline_recipe(model_dir=model_dir, manifest=manifest, output_dir=output,
                             phase=phase, wandb_project=wandb_project,
                             wandb_entity=wandb_entity, seed=seed,
                             global_batch_size=global_batch_size)
    config["training_contract"] = TRAINING_CONTRACT
    config["runtime_patches"]["dataset_contract"] = TRAINING_CONTRACT
    config["experiment_phase"] = phase
    for key in ("dataset", "validation_dataset"):
        config[key].update(_target_=DATASET_TARGET, index_path=str(index_path),
                           index_sha256=index_sha256, complete_path=str(complete_path),
                           complete_sha256=complete_sha256)
    # Keep the same fixed bounded validation budget as the baseline: first
    # 32 eligible known-session heldout rows, each at most 32768 tokens.
    for key in ("dataloader", "validation_dataloader"):
        config[key]["collate_fn"] = COLLATOR_TARGET
    if "validation" not in contents["splits"]:
        config.pop("validation_dataset")
        config.pop("validation_dataloader")
    config["wandb"]["name"] = WANDB_NAME_PREFIX + digest_file(manifest)[:12] + "-" + phase
    # A marker is useful to orchestration, but never a GPU qualification claim.
    config["dataset_contract"] = {
        "manifest_sha256": digest_file(manifest),
        "renderer_identity": deepcopy(contents["renderer_identity"]),
        "index_sha256": index_sha256,
        "complete_sha256": complete_sha256,
        "base_asset_receipt": deepcopy(BASE_ASSET_RECEIPT),
        "gpu_runtime_qualified": False,
    }
    return require_recipe(config)


def contract_identity(config):
    """Bind a future new runner/resume guard to this dataset and renderer."""
    require_recipe(config)
    manifest = read_manifest(config["dataset"]["path_or_dataset"])
    expected = config.get("dataset_contract", {})
    if (expected.get("manifest_sha256") != digest_file(config["dataset"]["path_or_dataset"])
            or expected.get("renderer_identity") != manifest["renderer_identity"]
            or expected.get("index_sha256") != config["dataset"]["index_sha256"]
            or expected.get("complete_sha256") != config["dataset"]["complete_sha256"]
            or expected.get("base_asset_receipt") != BASE_ASSET_RECEIPT):
        raise ValueError("The recipe dataset contract changed")
    return {
        "training_contract": TRAINING_CONTRACT,
        "manifest_sha256": expected["manifest_sha256"],
        "index_sha256": expected["index_sha256"],
        "complete_sha256": expected["complete_sha256"],
        "base_asset_receipt": deepcopy(expected["base_asset_receipt"]),
        "runtime_compatibility": deepcopy(expected.get("runtime_compatibility")),
        "renderer_identity": deepcopy(expected["renderer_identity"]),
        "model_settings": deepcopy(config["model"]),
        "runtime_patches": deepcopy(config["runtime_patches"]),
        "distributed": deepcopy(config["distributed"]),
        "optimizer": deepcopy(config["optimizer"]),
        "freeze_config": deepcopy(config["freeze_config"]),
        "rng": deepcopy(config["rng"]),
        "runtime_configuration": {key: deepcopy(config.get(key)) for key in (
            "loss_fn", "clip_grad_norm", "lr_scheduler", "step_scheduler", "dataloader",
            "validation_dataset", "validation_dataloader", "dist_env", "processor")},
        "checkpoint_policy": {k: v for k, v in config["checkpoint"].items() if k not in {"checkpoint_dir", "restore_from"}},
        "recipe_sha256": digest_file(__file__),
        **validator_identity(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--index-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phase", choices=("train",), default="train")
    parser.add_argument("--wandb-project", default=WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default=WANDB_ENTITY)
    parser.add_argument("--write", required=True)
    args = vars(parser.parse_args())
    path = Path(args.pop("write"))
    config = make_recipe(**args)
    with path.open("x") as stream:
        stream.write(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
