"""Ports Miles argv of the expanded DeepSeek V4 Flash recipe (P0 1.7b.2).

The legacy translation (``island_entry._legacy_miles_argv``) carries the same
recipe for the legacy Miles fork.  Two of its flags exist only there:
``--moe-router-freeze-gate`` (LoRA already freezes every base parameter except
the clone experts expert-full unfreezes) and
``--freeze-e-score-correction-bias`` (here: Megatron's
``--moe-router-bias-update-rate 0``).
"""

from __future__ import annotations

from typing import Any

MODEL_NAME = "deepseekv4"

# Trainer memory margin the legacy recipe validated on 8x H200.
_TRAIN_MEMORY_MARGIN_BYTES = 3 * 1024**3

RECIPE_ARGV: tuple[str, ...] = (
    "--transformer-impl", "transformer_engine",
    "--qkv-format", "bshd",
    "--recompute-granularity", "full",
    "--recompute-method", "uniform",
    "--recompute-num-layers", "1",
    "--micro-batch-size", "1",
    "--train-memory-margin-bytes", str(_TRAIN_MEMORY_MARGIN_BYTES),
    "--moe-token-dispatcher-type", "alltoall",
    "--moe-router-bias-update-rate", "0",
    "--attention-dropout", "0.0",
    "--hidden-dropout", "0.0",
    "--sglang-moe-runner-backend", "triton",
    "--sglang-disable-shared-experts-fusion",
)

# Expert-full: Adam over replicated FP32 masters, streamed by the full-parameter
# transport as one LoRA-hybrid shard (Miles --full-parameter-lora-hybrid).
EXPERT_FULL_ARGV: tuple[str, ...] = (
    "--optimizer", "adam",
    "--adam-beta1", "0.9",
    "--adam-beta2", "0.98",
    "--adam-eps", "1e-08",
    "--weight-decay", "0",
    "--clip-grad", "1.0",
    "--full-parameter-lora-hybrid",
)


# Clone-only expert LoRA: SGLang serves the packed expert adapters per expert,
# as the legacy recipe did.
CLONE_ONLY_ARGV: tuple[str, ...] = ("--no-sglang-lora-use-virtual-experts",)


def recipe_argv(trainable: Any) -> list[str]:
    """Recipe flags, plus the flags of the selected expert tuning."""

    return [
        *RECIPE_ARGV,
        *(EXPERT_FULL_ARGV if trainable.expert_full else ()),
        *(CLONE_ONLY_ARGV if trainable.routed_expert_lora else ()),
    ]
