"""Corrected no-CoT recipe retaining the proven CP8 memory/checkpoint settings."""
from copy import deepcopy
import json
from pathlib import Path

from training.qwen38_no_cot import nemo_recipe, nemo_train
from . import data

_BASE_REQUIRE = nemo_train.require_recipe
DATASET_TARGET = "training.qwen38_turn_boundary_v2.data.TurnTokenDataset"


def make_recipe(*, model_dir, manifest, index, index_sha256, output_dir, wandb_project, wandb_entity=None,
                seed=20260908, global_batch_size=8):
    config = nemo_recipe.make_recipe(model_dir=model_dir, manifest=manifest, output_dir=output_dir,
        phase="train", wandb_project=wandb_project, wandb_entity=wandb_entity,
        seed=seed, global_batch_size=global_batch_size)
    document = json.loads(Path(manifest).read_bytes())
    data.validate_identity(document["renderer_identity"])
    for key in ("dataset", "validation_dataset"):
        config[key].update(_target_=DATASET_TARGET, index_path=str(index), index_sha256=index_sha256)
    if not document["splits"].get("validation"):
        config.pop("validation_dataset"); config.pop("validation_dataloader")
    config["wandb"]["name"] = "qwen38-27b-no-cot-turns-v2"
    config["turn_boundary_contract"] = data.CONTRACT
    config["initialization"] = "pinned-original-base-fresh-optimizer"
    require_recipe(config)
    return config


def require_recipe(config):
    _BASE_REQUIRE(config)
    reference = nemo_recipe.make_recipe(model_dir=config["model"]["pretrained_model_name_or_path"],
        manifest=config["dataset"]["path_or_dataset"], output_dir=str(Path(config["checkpoint"]["checkpoint_dir"]).parent),
        phase="train", wandb_project=config["wandb"]["project"],
        seed=config.get("rng", {}).get("seed", 20260908), global_batch_size=8)
    # Comparing the complete small contracts also rejects hidden overrides such
    # as CE shift=True, a state_dict restore, a different optimizer implementation
    # or a frozen language backbone. A descriptive initialization flag is not a guard.
    for key in ("model", "processor", "optimizer", "loss_fn", "lr_scheduler", "clip_grad_norm",
                "freeze_config", "distributed", "dist_env", "dataloader", "rng", "runtime_patches", "step_scheduler"):
        if config.get(key) != reference[key]:
            raise ValueError("Qualified training setting changed: " + key)
    actual_checkpoint = {k: v for k, v in config["checkpoint"].items() if k != "checkpoint_dir"}
    reference_checkpoint = {k: v for k, v in reference["checkpoint"].items() if k != "checkpoint_dir"}
    if actual_checkpoint != reference_checkpoint:
        raise ValueError("Qualified checkpoint settings changed or an implicit restore was supplied")
    if config.get("turn_boundary_contract") != data.CONTRACT:
        raise ValueError("Missing corrected assistant-turn training contract")
    if config.get("initialization") != "pinned-original-base-fresh-optimizer":
        raise ValueError("This corrected baseline starts from the original base and fresh optimizer")
    if config["checkpoint"].get("restore_from"):
        raise ValueError("Do not automatically restore the old faulty-data run")
    for name in ("dataset", "validation_dataset"):
        if name not in config: continue
        item = config[name]
        if item.get("_target_") != DATASET_TARGET or not item.get("index_path") or not item.get("index_sha256"):
            raise ValueError("Corrected run requires its qualified source-turn data loader")
    if (config["dataloader"].get("shuffle") is not True or config["dataloader"].get("drop_last") is not False
            or config["dataloader"].get("collate_fn") != "training.qwen38_no_cot.nemo_data.collate_exact"
            or any(key in config["dataset"] for key in ("max_samples", "max_input_tokens"))
            or config["step_scheduler"].get("max_steps") is not None):
        raise ValueError("The corrected run consumes its full shuffled corpus with the existing exact collator")
    expected = {"cp_size": 8, "tp_size": 1, "pp_size": 1, "strategy": "fsdp2", "activation_checkpointing": True,
                "defer_fsdp_grad_sync": False, "enable_fsdp2_prefetch": False, "reshard_after_forward": True}
    if any(config["distributed"].get(key) != value for key, value in expected.items()):
        raise ValueError("The proven full-context memory topology changed")
    if (config["model"]["text_config"].get("use_cache") is not False
            or config["checkpoint"].get("cpu_offload") is not True
            or config["optimizer"].get("master_weight_dtype") != "torch.float32"
            or config["optimizer"].get("exp_avg_dtype") != "torch.float32"
            or config["optimizer"].get("exp_avg_sq_dtype") != "torch.float32"):
        raise ValueError("The qualified cache/checkpoint/optimizer precision policy changed")
