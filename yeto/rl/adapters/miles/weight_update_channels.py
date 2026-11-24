"""Give the trainer's weight-update NCCL group the SGLang engines' channel count.

SGLang's deterministic inference (on by default in yeto) with TP > 1 on CUDA
sets ``NCCL_MIN_NCHANNELS = NCCL_MAX_NCHANNELS =
SGLANG_DETERMINISTIC_NCCL_NCHANNELS`` (8) in every engine process. NCCL
applies those per rank, after the ranks agreed on a channel count, so in the
weight-update group (trainer rank 0 plus every engine GPU, built by Miles'
``connect_rollout_engines_from_distributed``) the engines connect 8 channels
and the trainer its own topology count (24 on an H100 node). The first
broadcast's connection setup then fails in every engine with "Message
truncated : received 6144 bytes instead of 2048" (24 versus 8 connection
records) and ``update_weights_from_distributed`` returns 400 with the trainer
holding the weight-update lock.

Setting the same variables for the trainer would pin every Megatron collective
too, so only the update group's communicator gets the engines' count, through
NCCL's per-communicator ``minCTAs``/``maxCTAs`` (``ProcessGroupNCCL.Options``).
The island exports :data:`ENV` through the job ``runtime_env``
(:func:`yeto.rl.adapters.miles.entry.connect_island_ray`) and each Ray worker's
``sitecustomize`` patches the Miles module that creates the group.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from types import ModuleType
from typing import Any

from yeto.rl.adapters.miles.models.partial_experts.hooks import install_once, install_or_defer

ENV = "YETO_WEIGHT_UPDATE_NCCL_CHANNELS"
SGLANG_CHANNELS_ENV = "SGLANG_DETERMINISTIC_NCCL_NCHANNELS"
SGLANG_DEFAULT_CHANNELS = 8  # SGLang envs.SGLANG_DETERMINISTIC_NCCL_NCHANNELS default
_BROADCAST = "miles.backends.training_utils.weight_update.protocols.broadcast"


def island_env(miles_args: Any, environ: Mapping[str, str]) -> dict[str, str]:
    """``{ENV: channels}`` when this island's engines pin their NCCL channels, else empty.

    That is: the island starts SGLang rollout engines (not a trainer-only island),
    each engine spans more than one GPU, and deterministic inference is on.
    """

    if miles_args is None or getattr(miles_args, "debug_train_only", False):
        return {}
    if not int(getattr(miles_args, "rollout_num_gpus", 0) or 0):
        return {}
    if int(getattr(miles_args, "rollout_num_gpus_per_engine", 1) or 1) <= 1:
        return {}
    if not getattr(miles_args, "sglang_enable_deterministic_inference", False):
        return {}
    channels = int(environ.get(SGLANG_CHANNELS_ENV) or SGLANG_DEFAULT_CHANNELS)
    if channels < 1:
        raise ValueError(f"{SGLANG_CHANNELS_ENV} must be a positive channel count, got {channels}")
    return {ENV: str(channels)}


def channels_from_env() -> int | None:
    raw = os.environ.get(ENV)
    return int(raw) if raw else None


def nccl_options(channels: int, dist_module: Any = None) -> Any:
    """``ProcessGroupNCCL.Options`` with ``min_ctas = max_ctas = channels``."""

    if dist_module is None:
        import torch.distributed as dist_module
    try:
        options = dist_module.ProcessGroupNCCL.Options()
        config = options.config
        config.min_ctas = channels
        config.max_ctas = channels
    except AttributeError as exc:
        raise RuntimeError(
            "this PyTorch build cannot set a per-communicator NCCL channel count "
            f"(ProcessGroupNCCL.Options().config.min_ctas/max_ctas); the weight-update group would "
            f"not match the SGLang engines' {channels} channels") from exc
    return options


def _is_rocm() -> bool:
    import torch

    return bool(getattr(torch.version, "hip", None))


def _patch(module: ModuleType) -> None:
    original = module.init_process_group

    def init_process_group(*args, **kwargs):
        channels = channels_from_env()
        if (channels and kwargs.get("pg_options") is None and str(kwargs.get("backend", "")).lower() == "nccl"
                and not _is_rocm()):  # SGLang pins nothing on ROCm
            kwargs["pg_options"] = nccl_options(channels)
            print(f"[yeto] weight-update group {kwargs.get('group_name')}: {channels} NCCL channels "
                  "(the SGLang engines' deterministic-inference pin)", flush=True)
        return original(*args, **kwargs)

    module.init_process_group = init_process_group


def install_on_broadcast_module(module: ModuleType) -> None:
    install_once(module, _patch, namespace="yeto_weight_update_channels")


def install() -> None:
    """Patch Miles' broadcast protocol module now or on import (imports nothing heavy: every
    Ray worker, the SGLang engine processes included, runs this from ``sitecustomize``)."""

    if channels_from_env() is None:
        return
    install_or_defer(_BROADCAST, install_on_broadcast_module)
