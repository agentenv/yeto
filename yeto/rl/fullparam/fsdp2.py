"""FSDP2 full-parameter training of an HF causal LM (S25 N7.2, design D1).

One configuration, two consumers:

* :func:`shard_model` / :func:`train_step` run the plan directly with
  ``torch.distributed.fsdp.fully_shard`` (used by the CPU test and the
  single-card probe; any device whose torch build has the collective);
* :func:`verl_overrides` writes the same plan as verl 0.9 Hydra overrides
  (``actor_rollout_ref.actor.fsdp_config``, ``engine/fsdp.yaml``), for the
  GLM stack (FSDP2 + HF ``glm5_next``), with LoRA off (``lora_rank=0``).

Offload modes follow :mod:`.memory`: ``none`` and ``full`` (FSDP2
``CPUOffloadPolicy``) are supported; ``optimizer`` (plan B, host optimizer
step with BF16 weights on the card) is refused: neither FSDP2 nor verl runs it.
verl's ``param_offload`` / ``optimizer_offload`` move state to the host between
training phases (a time-shared island); they do not lower the step's peak.

Ulysses: verl's Ulysses patch replaces ``_flash_attention_forward`` only
(verl/models/transformers/monkey_patch.py:330). A model without the flash
attention path (``glm5_next``: ``_supports_flash_attn = False``,
modeling_glm5_next.py:1338) and with linear-attention layers (KDA) gets no
sequence split there, so :func:`check_ulysses` refuses ``ulysses > 1`` for it.
A Ulysses patch for KDA and DSA is open work (not done).

Status: CPU, 2 gloo ranks, one step on a tiny HF model and on a tiny
``glm5_next`` (tests/test_rl_fullparam_fsdp2.py); not run on PPU or NPU.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .memory import OFFLOAD_MODES

# The collective backend of each device family (D4). PPU keeps the CUDA device
# name and NCCL API (over ACCL-P; not verified on a PPU machine).
BACKENDS = {"cpu": "gloo", "cuda": "nccl", "ppu": "nccl", "npu": "hccl"}


class FullParamConfigError(ValueError):
    """The full-parameter plan cannot run as configured."""


@dataclass(frozen=True)
class FullParamConfig:
    fsdp_size: int = 0  # 0: every rank in one shard group
    ulysses: int = 1
    offload: str = "none"  # see memory.OFFLOAD_MODES
    offload_between_phases: bool = False  # verl param/optimizer offload (time-shared islands)
    param_dtype: torch.dtype = torch.bfloat16
    reduce_dtype: torch.dtype = torch.float32
    reshard_after_forward: bool = True
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        if self.offload not in OFFLOAD_MODES:
            raise FullParamConfigError(f"offload must be one of {OFFLOAD_MODES}, got {self.offload!r}")
        if self.offload == "optimizer":
            raise FullParamConfigError("offload='optimizer' (host optimizer step, BF16 weights on the card) "
                                       "is not implemented in FSDP2 or verl; use 'none' or 'full'")
        if self.fsdp_size < 0 or self.ulysses < 1:
            raise FullParamConfigError("fsdp_size must be >= 0 and ulysses >= 1")


def check_ulysses(model: nn.Module, cfg: FullParamConfig) -> None:
    """Refuse a Ulysses degree the model cannot run (module doc)."""
    if cfg.ulysses == 1:
        return
    if not getattr(model, "_supports_flash_attn", False):
        raise FullParamConfigError(
            f"ulysses={cfg.ulysses}: {type(model).__name__} has no flash-attention path, and verl's Ulysses "
            "patch only replaces _flash_attention_forward; KDA/DSA layers need their own sequence split")
    heads = [getattr(_text_config(model.config), k, None) for k in ("num_attention_heads", "num_key_value_heads")]
    for h in (h for h in heads if h):
        if h % cfg.ulysses:
            raise FullParamConfigError(f"ulysses={cfg.ulysses} does not divide a head count {h}")


def _text_config(config: Any) -> Any:
    return getattr(config, "text_config", None) or config


def decoder_layers(model: nn.Module) -> list[nn.Module]:
    """The modules FSDP2 shards one by one: every module whose class is in the
    model's ``_no_split_modules`` (the HF wrap unit, also verl's)."""
    names = set(getattr(model, "_no_split_modules", None) or ())
    layers = [m for m in model.modules() if type(m).__name__ in names]
    if not layers:
        raise FullParamConfigError(f"{type(model).__name__}: no module of _no_split_modules {sorted(names)}")
    return layers


def shard_model(model: nn.Module, cfg: FullParamConfig, mesh: Any = None) -> nn.Module:
    """Apply FSDP2 to every decoder layer, then to the root (in place).

    ``mesh``: a 1-D device mesh (the shard group) or a 2-D (replicate, shard)
    mesh for HSDP; default: all ranks of the default group."""
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, OffloadPolicy, fully_shard

    check_ulysses(model, cfg)
    if cfg.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    kwargs: dict[str, Any] = {
        "mp_policy": MixedPrecisionPolicy(param_dtype=cfg.param_dtype, reduce_dtype=cfg.reduce_dtype),
        "offload_policy": CPUOffloadPolicy() if cfg.offload == "full" else OffloadPolicy(),
    }
    if mesh is not None:
        kwargs["mesh"] = mesh
    for layer in decoder_layers(model):
        fully_shard(layer, reshard_after_forward=cfg.reshard_after_forward, **kwargs)
    fully_shard(model, reshard_after_forward=False, **kwargs)  # root: kept gathered for the backward
    return model


@dataclass
class StepResult:
    loss: float
    grad_norm: float

    @property
    def finite(self) -> bool:
        return math.isfinite(self.loss) and math.isfinite(self.grad_norm)


def train_step(model: nn.Module, optimizer: torch.optim.Optimizer, input_ids: torch.Tensor,
               labels: torch.Tensor | None = None, *, max_grad_norm: float = 1.0) -> StepResult:
    """One forward, backward, clip and optimizer step; returns the loss and the
    total gradient norm before clipping (both reduced over the shard group)."""
    model.train()
    out = model(input_ids=input_ids, labels=input_ids if labels is None else labels, use_cache=False)
    out.loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    if hasattr(norm, "full_tensor"):  # DTensor under FSDP2
        norm = norm.full_tensor()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return StepResult(loss=float(out.loss.detach()), grad_norm=float(norm))


def verl_overrides(cfg: FullParamConfig, *, role: str = "actor") -> list[str]:
    """The Hydra overrides of this plan for a verl 0.9 FSDP2 actor (keys from
    verl/trainer/config/_generated_ppo_trainer.yaml and engine/fsdp.yaml)."""
    base = f"actor_rollout_ref.{role}"
    fsdp = f"{base}.fsdp_config"
    between = str(cfg.offload_between_phases).lower()
    return [
        f"{base}.strategy=fsdp2",
        "actor_rollout_ref.model.lora_rank=0",
        f"actor_rollout_ref.model.enable_gradient_checkpointing={str(cfg.gradient_checkpointing).lower()}",
        f"{base}.ulysses_sequence_parallel_size={cfg.ulysses}",
        f"{base}.grad_clip={cfg.max_grad_norm}",
        f"{fsdp}.fsdp_size={cfg.fsdp_size or -1}",
        f"{fsdp}.offload_policy={str(cfg.offload == 'full').lower()}",
        f"{fsdp}.param_offload={between}",
        f"{fsdp}.optimizer_offload={between}",
        f"{fsdp}.reshard_after_forward={str(cfg.reshard_after_forward).lower()}",
    ]


def parameter_count(params: Iterable[torch.nn.Parameter]) -> int:
    return sum(p.numel() for p in params)
