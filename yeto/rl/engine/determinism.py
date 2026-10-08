"""Neutral determinism switch: backend x device family -> environment
(yeto-framework-decoupling task 4.9, audit P4). Import-light, core.

``--deterministic`` (alias of the historical ``--rl-deterministic-trainer``)
asks for a deterministic trainer. What that means in environment variables
depends on the training backend and the accelerator family, so it is a table
lookup here; the backend's own switches (Miles: Megatron
``--deterministic-mode``) stay in its adapter.

Only measured rows exist. A missing row is refused, never guessed: the NPU
(HCCL) equivalent of ``NCCL_ALGO=Ring`` is not verified (design Open
Questions), so Ascend has no row yet.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

# E2 plan-v2 §0 (moved unchanged from miles_adapter/entry.py). NVTE_ALLOW_
# NONDETERMINISTIC_ALGO=0: Megatron's --deterministic-mode only setdefaults it
# in the process that validates the args, while Transformer Engine reads it in
# each trainer rank (Ray worker), so it is set for every rank.
_NVIDIA_MEGATRON = MappingProxyType({
    "NCCL_ALGO": "Ring", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "NVIDIA_TF32_OVERRIDE": "0", "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
})

DETERMINISM_ENV_TABLE: Mapping[tuple[str, str], Mapping[str, str]] = MappingProxyType({
    ("miles", "nvidia"): _NVIDIA_MEGATRON,
})


class DeterminismNotCalibrated(ValueError):
    """No measured determinism environment for this backend x device family."""


def determinism_env(backend: str = "miles", device_family: str = "nvidia") -> dict[str, str]:
    """A fresh copy of the environment for ``backend`` on ``device_family``."""

    try:
        return dict(DETERMINISM_ENV_TABLE[(backend, device_family)])
    except KeyError:
        known = sorted(DETERMINISM_ENV_TABLE)
        raise DeterminismNotCalibrated(
            f"no determinism environment for backend={backend!r} device_family="
            f"{device_family!r} (measured rows: {known})"
        ) from None
