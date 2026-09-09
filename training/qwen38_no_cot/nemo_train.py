"""Guarded NeMo entry point with finite-step and checkpoint round-trip receipts.

Run under torchrun on the selected eight-GPU host. CPU preflight only parses
the installed NeMo configuration; GPU qualification is earned by smoke runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract_identity(config):
    source = Path(__file__).parent
    model = Path(config["model"]["pretrained_model_name_or_path"])
    return {
        "model_id": "Qwen/Qwen3.8-27B",
        "model_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "model_config_sha256": digest(model / "config.json"),
        "model_weight_index_sha256": digest(model / "model.safetensors.index.json"),
        "runtime_image": os.environ.get("YETA_TRAINING_IMAGE", "unrecorded"),
        "code_sha256": {str(path.relative_to(source)): digest(path) for path in (
            source / "nemo_train.py", source / "nemo_data.py", source / "nemo_recipe.py", source / "nemo_checkpoint_probe.py", source / "nemo_model_probe.py",
            source / "nemo_cp_memory.py", source / "nemo_block_checkpoint.py", source / "nemo_checkpoint.py",
            source / "render.py", source / "upstream_manager/render.py")},
        "runtime_patches": config["runtime_patches"],
        "distributed": config["distributed"],
        "model_settings": config["model"],
        "optimizer": config["optimizer"],
        "freeze_config": config["freeze_config"],
        "mask_policy": "assistant_content_and_eos_only_no_cot_no_headers_v1",
        "sequence_policy": "first_262144_tokens_drop_remainder/v1",
    }


def require_recipe(config):
    if config.get("runtime_patches") != {"cp_memory": "contiguous_gather_local_reorder/v1",
                                         "activation_checkpointing": "native_qwen_whole_block/v1"}:
        raise ValueError("Both qualified memory policies are required")
    checkpoint = config["checkpoint"]
    if any(checkpoint.get(key) != value for key, value in {
            "enabled": True, "is_async": False, "max_recent_checkpoints": 3,
            "model_save_format": "torch_save", "save_consolidated": False}.items()):
        raise ValueError("The baseline requires synchronous full checkpoints with three retained")
    distributed = config["distributed"]
    expected = {"strategy": "fsdp2", "cp_size": 8, "tp_size": 1, "pp_size": 1}
    if any(distributed.get(key) != value for key, value in expected.items()):
        raise ValueError("This launch must use one host with FSDP2 and CP8")
    if config["optimizer"]["lr"] != 1e-5 or config.get("peft") or config.get("quantization"):
        raise ValueError("Full SFT requires LR1e-5 without LoRA or quantization")
    if config["step_scheduler"]["num_epochs"] != 1:
        raise ValueError("The baseline must mix once and train one pass")
    if config["dataset"]["seq_len"] != 262144:
        raise ValueError("The authorized prefix context is 262144")
    if config["dataset"].get("require_provenance") is not True:
        raise ValueError("Every dataset must validate its actual renderer and mask provenance")
    if config["loss_fn"]["_target_"] != "nemo_automodel.components.loss.linear_ce.FusedLinearCrossEntropy":
        raise ValueError("Full-vocabulary logits are not the qualified long-context loss path")
    if not config.get("wandb", {}).get("project"):
        raise ValueError("An explicit W&B project is required")
    if config["model"].get("num_nextn_predict_layers") != 0:
        raise ValueError("The baseline uses ordinary next-token SFT with MTP disabled")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("preflight", "short-smoke", "long-smoke", "resident-long-smoke", "train"), default="preflight")
    parser.add_argument("--short-receipt")
    parser.add_argument("--long-receipt")
    parser.add_argument("--resident-long-receipt")
    parser.add_argument("--validate-memory-repair-in-production", action="store_true",
                        help="Explicit user-authorized launch without a separate resident-long GPU test")
    parser.add_argument("--resume", help="Explicit checkpoint path; required to reuse an occupied output")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    require_recipe(config)
    if config["checkpoint"].get("restore_from") and not args.resume:
        raise ValueError("Checkpoint reuse requires explicit --resume and cursor validation")
    from nemo_automodel.components.config.loader import ConfigNode
    from nemo_automodel.recipes._typed_config import RecipeConfig
    typed = RecipeConfig(ConfigNode(config))
    for name in ("checkpoint", "step_scheduler", "lr_scheduler", "optimizer", "loss_fn", "wandb", "dataloader"):
        getattr(typed, name)
    if args.mode == "preflight":
        print(json.dumps({"status": "configuration_parsed_gpu_unqualified", "cp_size": 8,
                          "max_tokens": 262144, "lr": 1e-5, "label_shift": "dataset_once_before_cp"}))
        return

    identity = contract_identity(config)
    output = Path(config["checkpoint"]["checkpoint_dir"])
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("Refusing automatic checkpoint reuse: select a fresh output or explicit --resume")
    from .nemo_checkpoint import (CheckpointResumeContract, resume_identity, prepare_checkpoint_boundary,
                                  resolve_explicit_resume, verify_restored_cursor)
    checkpoint_identity = resume_identity(config, identity)
    resume_receipt = None
    if args.resume:
        resume_receipt = resolve_explicit_resume(str(output), args.resume, checkpoint_identity)
        config["checkpoint"]["restore_from"] = resume_receipt["checkpoint"]
    if args.mode == "train":
        if (any(key in config["dataset"] for key in ("max_input_tokens", "max_samples"))
                or config["dataset"].get("order", "source") != "source"
                or config["dataloader"].get("shuffle") is not True
                or config["step_scheduler"].get("max_steps") is not None):
            raise ValueError("Production must consume the whole mixed dataset in one shuffled pass, without smoke filters")
        if not args.resident_long_receipt and not args.validate_memory_repair_in_production:
            raise ValueError("The repaired runtime requires a new two-update resident-Adam full262k receipt")
        if args.resident_long_receipt:
            receipt = json.loads(Path(args.resident_long_receipt).read_text())
            if (receipt.get("status") != "passed" or receipt.get("phase") != "resident-long-smoke"
                    or receipt.get("contract_identity") != identity
                    or receipt.get("optimizer_steps") != 2
                    or receipt.get("resume_cursor_probe", {}).get("dataloader_and_rng_restored") is not True
                    or receipt.get("checkpoint_roundtrip") is not True
                    or receipt.get("observed_input_tokens") != [262144, 262144]
                    or receipt.get("cuda_collective_parity", {}).get("ranks") != 8
                    or receipt.get("block_checkpoint_coverage", {}).get("whole_block_wrappers") != 64):
                raise ValueError("Resident-Adam qualification does not match the exact repaired runtime")

    import torch
    from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy
    from nemo_automodel.recipes.vlm.finetune import FinetuneRecipeForVLM
    from .nemo_checkpoint_probe import perturb_optimizer_state, verify_optimizer_state_restored
    from .nemo_model_probe import perturb_model_state, verify_model_state_restored

    from . import nemo_cp_memory, nemo_block_checkpoint
    nemo_cp_memory.install()
    nemo_block_checkpoint.install()

    class VerifiedRecipe(FinetuneRecipeForVLM):
        def _finalize_and_close_checkpointer(self):
            if args.mode == "train":
                return super()._finalize_and_close_checkpointer()
            # Keep checkpoint process groups alive until the smoke reload.
            checkpointer = getattr(self, "checkpointer", None)
            if checkpointer is not None and checkpointer.config.enabled:
                checkpointer.async_wait()
                checkpointer.lifecycle.complete_pending()

        def save_checkpoint(self, epoch, step, train_loss, val_loss=None, best_metric_key="default"):
            # Keep the latest three full production checkpoints; a separate
            # LOWEST_VAL pointer must not retain additional permanent copies.
            return super().save_checkpoint(epoch, step, train_loss,
                                           None if args.mode == "train" else val_loss,
                                           best_metric_key=best_metric_key)

        def setup(self):
            self.resume_guard = CheckpointResumeContract(checkpoint_identity)
            super().setup()
            if resume_receipt is not None:
                self.resume_cursor = verify_restored_cursor(self, resume_receipt)
            if self.dist_env.world_size != 8 or self._get_cp_group_size() != 8:
                raise RuntimeError("Actual training topology is not eight ranks with CP8")
            if not isinstance(self.loss_fn, FusedLinearCrossEntropy):
                raise RuntimeError("NeMo silently downgraded the required fused loss")
            model = self.model_parts[0]
            text = getattr(model.config, "text_config", model.config)
            if text.num_hidden_layers != 64 or text.hidden_size != 5120:
                raise RuntimeError("Loaded model does not match the pinned Qwen27B architecture")
            if getattr(model, "mtp", None) is not None:
                raise RuntimeError("Optional MTP unexpectedly enabled")
            trainable, frozen = 0, 0
            for name, parameter in model.named_parameters():
                vision = ".visual." in "." + name or name.startswith("visual.")
                if not vision and not parameter.requires_grad:
                    raise RuntimeError("A text parameter was frozen: " + name)
                if vision and parameter.requires_grad:
                    raise RuntimeError("Unused vision parameters must remain preserved and frozen")
                if parameter.requires_grad:
                    trainable += parameter.numel()
                else:
                    frozen += parameter.numel()
            if trainable < 20_000_000_000:
                raise RuntimeError("Trainable parameter count is inconsistent with full27B text SFT")
            self.block_checkpoint_coverage = nemo_block_checkpoint.verify_model(model)
            self.cuda_collective_parity = (nemo_cp_memory.verify_cuda_parity()
                                          if args.mode == "resident-long-smoke" or args.validate_memory_repair_in_production else {})
            self.observed_input_tokens = []
            self.baseline_steps = []
            self.max_observed_input_tokens = 0
            self.parameter_coverage = {"trainable_parameters": trainable, "preserved_parameters": frozen}

        def _run_train_optim_step(self, batches, max_grad_norm=None):
            if args.mode == "train":
                prepare_checkpoint_boundary(self.step_scheduler)
            for batch in batches:
                self.observed_input_tokens.append(batch["input_ids"].shape[-1])
                self.max_observed_input_tokens = max(self.max_observed_input_tokens, batch["input_ids"].shape[-1])
            result = super()._run_train_optim_step(batches, max_grad_norm)
            metrics = result.metrics
            if any(not math.isfinite(float(metrics[name])) for name in ("loss", "grad_norm", "lr")):
                raise RuntimeError("Nonfinite loss, gradient norm or learning rate")
            if float(metrics["grad_norm"]) <= 0 or int(metrics["num_label_tokens"]) <= 0:
                raise RuntimeError("Smoke step has no nonzero gradient or supervised targets")
            measurement = {key: float(metrics[key]) for key in
                           ("loss", "grad_norm", "lr", "mem", "tps", "num_label_tokens")}
            peak = torch.tensor([torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()],
                                dtype=torch.float64, device=self.dist_env.device)
            torch.distributed.all_reduce(peak, op=torch.distributed.ReduceOp.MAX)
            measurement.update(max_rank_peak_allocated_gib=peak[0].item()/2**30,
                               max_rank_peak_reserved_gib=peak[1].item()/2**30,
                               observed_input_tokens=[batch["input_ids"].shape[-1] for batch in batches])
            self.baseline_steps.append(measurement)
            if self.dist_env.is_main:
                print(json.dumps({"event": "verified_optimizer_update", "step": self.step_scheduler.step,
                                  "measurements": measurement}), flush=True)
            return result

    trainer = VerifiedRecipe(ConfigNode(config))
    trainer.setup()
    if trainer.dist_env.is_main and args.validate_memory_repair_in_production:
        (output.parent / "production-memory-validation.json").write_text(json.dumps({
            "status": "gpu_memory_and_new_checkpoint_resume_unqualified",
            "authorization": "user_requested_production_instead_of_separate_two_update_test",
            "contract_identity": identity,
            "cuda_collective_parity": trainer.cuda_collective_parity,
            "block_checkpoint_coverage": trainer.block_checkpoint_coverage,
            "checkpoint_policy": "after_update_1_then_every_5_keep_latest_3",
        }, indent=2) + "\n")
    trainer.run_train_validation_loop()
    if args.mode == "train":
        return
    expected_steps = 2 if args.mode in {"short-smoke", "resident-long-smoke"} else 1
    if len(trainer.baseline_steps) != expected_steps:
        raise RuntimeError("Smoke did not perform the required number of optimizer updates")
    if args.mode == "resident-long-smoke" and trainer.observed_input_tokens != [262144, 262144]:
        raise RuntimeError("Resident-Adam smoke requires two actual full262144-token updates")
    if args.mode == "long-smoke" and trainer.max_observed_input_tokens != 262144:
        raise RuntimeError("Smoke never exercised a full262144-token training sequence")
    before = float(trainer._run_validation_epoch(trainer.val_dataloader).metrics["val_loss"])
    # Prove that load_checkpoint restores real model state, rather than merely
    # comparing an unchanged in-memory model with itself.
    model_probe = perturb_model_state(trainer.model_parts)
    optimizer_probes = perturb_optimizer_state(trainer.optimizer)
    smoke_resume = resolve_explicit_resume(str(output), "LATEST", checkpoint_identity)
    trainer.load_checkpoint(smoke_resume["checkpoint"])
    cursor_coverage = verify_restored_cursor(trainer, smoke_resume)
    restored, model_coverage = verify_model_state_restored(trainer.model_parts, model_probe)
    restored_count = torch.tensor(int(restored), device=trainer.dist_env.device)
    torch.distributed.all_reduce(restored_count)
    if restored_count.item() != 8:
        raise RuntimeError("Checkpoint failed to restore perturbed rank-local model state")
    optimizer_restored, optimizer_coverage = verify_optimizer_state_restored(trainer.optimizer, optimizer_probes)
    optimizer_restored_count = torch.tensor(int(optimizer_restored), device=trainer.dist_env.device)
    torch.distributed.all_reduce(optimizer_restored_count)
    if optimizer_restored_count.item() != 8:
        raise RuntimeError("Checkpoint failed to restore perturbed optimizer moments and step counters")
    after = float(trainer._run_validation_epoch(trainer.val_dataloader).metrics["val_loss"])
    if not math.isfinite(before) or not math.isfinite(after) or abs(before - after) > 1e-5 * max(1, abs(before)):
        raise RuntimeError("Checkpoint round trip changed held-out validation loss")
    if trainer.dist_env.is_main:
        receipt = {"status": "passed", "phase": args.mode, "contract_identity": identity,
                   "smoke_manifest_sha256": digest(config["dataset"]["path_or_dataset"]),
                   "optimizer_steps": len(trainer.baseline_steps), "checkpoint_roundtrip": True,
                   "model_checkpoint_probe": model_coverage,
                   "optimizer_checkpoint_probe": optimizer_coverage,
                   "resume_cursor_probe": cursor_coverage,
                   "max_observed_input_tokens": trainer.max_observed_input_tokens,
                   "observed_input_tokens": trainer.observed_input_tokens,
                   "cuda_collective_parity": trainer.cuda_collective_parity,
                   "block_checkpoint_coverage": trainer.block_checkpoint_coverage,
                   "parameter_coverage": trainer.parameter_coverage,
                   "validation_loss_before_reload": before, "validation_loss_after_reload": after,
                   "steps": trainer.baseline_steps}
        destination = output.parent / "smoke-receipt.json"
        destination.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({"status": "smoke_passed", "receipt": str(destination)}), flush=True)
    torch.distributed.barrier()
    FinetuneRecipeForVLM._finalize_and_close_checkpointer(trainer)


if __name__ == "__main__":
    main()
