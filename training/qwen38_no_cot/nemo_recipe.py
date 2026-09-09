"""Create a one-host NeMo FSDP2/CP8 baseline or bounded smoke recipe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def make_recipe(*, model_dir, manifest, output_dir, phase, wandb_project, wandb_entity=None,
                seed=20260908, global_batch_size=8):
    if phase not in {"short-smoke", "long-smoke", "resident-long-smoke", "train"}:
        raise ValueError("Unknown phase")
    smoke = phase != "train"
    train_dataset = {"_target_": "training.qwen38_no_cot.nemo_data.ExactTokenDataset",
                     "path_or_dataset": str(manifest), "split": "train", "seq_len": 262144,
                     "require_provenance": True}
    if phase == "short-smoke":
        train_dataset.update(max_input_tokens=32768, max_samples=2)
    elif phase in {"long-smoke", "resident-long-smoke"}:
        train_dataset.update(order="longest_first", max_samples=2 if phase == "resident-long-smoke" else 1)
    validation = {**train_dataset, "split": "validation", "max_input_tokens": 32768,
                  "order": "source", "max_samples": 4 if smoke else 32}
    recipe = {
        "recipe": "FinetuneRecipeForVLM",
        "runtime_patches": {"cp_memory": "contiguous_gather_local_reorder/v1",
                            "activation_checkpointing": "native_qwen_whole_block/v1"},
        "step_scheduler": {"global_batch_size": 1 if smoke else global_batch_size,
                           "local_batch_size": 1, "num_epochs": 1,
                           "ckpt_every_steps": 1 if smoke else 5,
                           "val_every_steps": 1 if smoke else 100,
                           "save_checkpoint_every_epoch": True,
                           "log_remote_every_steps": 1,
                           "gc_every_steps": 10},
        "dist_env": {"backend": "nccl", "timeout_minutes": 60},
        "rng": {"_target_": "nemo_automodel.components.training.rng.StatefulRNG",
                "seed": seed, "ranked": True},
        "model": {
            "_target_": "nemo_automodel.NeMoAutoModelForImageTextToText.from_pretrained",
            "pretrained_model_name_or_path": str(model_dir),
            "trust_remote_code": False, "local_files_only": True,
            "torch_dtype": "torch.bfloat16", "attn_implementation": "sdpa",
            "num_nextn_predict_layers": 0,
            "text_config": {"output_hidden_states": True, "num_nextn_predict_layers": 0,
                            "mtp_num_hidden_layers": 0, "use_cache": False},
            "backend": {"_target_": "nemo_automodel.components.models.common.BackendConfig",
                        "attn": "sdpa", "linear": "torch", "rms_norm": "torch_fp32",
                        "rope_fusion": False, "enable_hf_state_dict_adapter": True,
                        "enable_fsdp_optimizations": True},
        },
        "processor": {"_target_": "transformers.AutoProcessor.from_pretrained",
                      "pretrained_model_name_or_path": str(model_dir),
                      "local_files_only": True, "trust_remote_code": False},
        "distributed": {"strategy": "fsdp2", "dp_size": None, "tp_size": 1,
                        "cp_size": 8, "pp_size": 1, "ep_size": 1,
                        "sequence_parallel": False, "activation_checkpointing": True,
                        "defer_fsdp_grad_sync": False, "reshard_after_forward": True,
                        "enable_fsdp2_prefetch": False},
        "freeze_config": {"freeze_embeddings": False, "freeze_vision_tower": True,
                          "freeze_audio_tower": True, "freeze_language_model": False},
        "loss_fn": {"_target_": "nemo_automodel.components.loss.linear_ce.FusedLinearCrossEntropy"},
        "dataset": train_dataset,
        "dataloader": {"_target_": "torchdata.stateful_dataloader.StatefulDataLoader",
                       "num_workers": 0, "pin_memory": False, "drop_last": False,
                       "shuffle": not smoke,
                       "collate_fn": "training.qwen38_no_cot.nemo_data.collate_exact"},
        "validation_dataset": validation,
        "validation_dataloader": {"_target_": "torchdata.stateful_dataloader.StatefulDataLoader",
                                  "num_workers": 0, "pin_memory": False, "drop_last": False,
                                  "shuffle": False,
                                  "collate_fn": "training.qwen38_no_cot.nemo_data.collate_exact"},
        "optimizer": {"_target_": "transformer_engine.pytorch.optimizers.FusedAdam",
                      "lr": 1e-5, "betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 0.1,
                      "adam_w_mode": True, "bias_correction": True, "master_weights": True,
                      "master_weight_dtype": "torch.float32", "store_param_remainders": True,
                      "exp_avg_dtype": "torch.float32", "exp_avg_sq_dtype": "torch.float32"},
        "lr_scheduler": {"lr_decay_style": "constant", "lr_warmup_steps": 0},
        "clip_grad_norm": {"max_norm": 1.0},
        "checkpoint": {"enabled": True, "checkpoint_dir": str(Path(output_dir) / "checkpoints"),
                       "model_save_format": "torch_save", "save_consolidated": False,
                       "is_async": False, "cpu_offload": True,
                       "max_recent_checkpoints": 3},
        "wandb": {"project": wandb_project, "name": "qwen38-27b-no-cot-" + phase,
                  "dir": str(output_dir)},
    }
    if phase in {"short-smoke", "resident-long-smoke"}:
        recipe["step_scheduler"]["max_steps"] = 2
    elif phase == "long-smoke":
        recipe["step_scheduler"]["max_steps"] = 1
    if wandb_entity:
        recipe["wandb"]["entity"] = wandb_entity
    return recipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phase", choices=("short-smoke", "long-smoke", "resident-long-smoke", "train"), required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-entity")
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--write", required=True)
    args = parser.parse_args()
    values = vars(args).copy()
    destination = Path(values.pop("write"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(make_recipe(**values), indent=2) + "\n")


if __name__ == "__main__":
    main()
