"""Accelerator-family abstraction for torch learners.

torch has no device-generic API for seeding, memory queries, or the caching
allocator: each vendor gets its own namespace (``torch.cuda``, ``torch.npu``)
that happens to spell those calls identically. The learner needs six of them
plus the matching collective backend, so the namespace is resolved once here
rather than letting a second accelerator family spread ``device.type ==
"cuda"`` through model loading and the training loop.

Ascend NPUs live in the out-of-tree ``torch_npu`` extension: importing it is
what registers both ``torch.npu`` and the ``npu`` device type, so nothing
below can see a card before ``register_backends`` has run. See docs/ASCEND.md.
"""

from __future__ import annotations

import os
from types import ModuleType

import torch

# Accelerator device types, in auto-detection priority order.
_ACCELERATORS = ("cuda", "npu")

# torch.distributed backend per device family.
_DIST_BACKENDS = {"cuda": "nccl", "npu": "hccl", "cpu": "gloo"}


def register_backends() -> None:
    """Register out-of-tree device types. Idempotent; safe without the card."""
    if hasattr(torch, "npu"):
        return
    try:
        import torch_npu  # noqa: F401  (registers torch.npu and the npu device)
    except ImportError:
        pass


def backend(device: torch.device) -> ModuleType | None:
    """``torch.cuda``/``torch.npu`` for an accelerator, ``None`` for the CPU."""
    if device.type not in _ACCELERATORS:
        return None
    return getattr(torch, device.type)


def is_accelerator(device: torch.device) -> bool:
    """Whether ``device`` carries its own memory and RNG, unlike the CPU."""
    return device.type in _ACCELERATORS


def available_type() -> str:
    """The accelerator family visible on this node, or ``"cpu"``."""
    register_backends()
    for name in _ACCELERATORS:
        module = getattr(torch, name, None)
        if module is not None and module.is_available():
            return name
    return "cpu"


def dist_backend(device: torch.device | None = None) -> str:
    """The collective backend matching the selected device or this node."""
    device_type = available_type() if device is None else device.type
    return _DIST_BACKENDS.get(device_type, "gloo")


def detect(explicit: str | None) -> torch.device:
    """Resolve the training device, binding this rank to its local card."""
    register_backends()
    if explicit:
        if explicit.startswith("npu") and not hasattr(torch, "npu"):
            raise RuntimeError(
                "--device npu needs the torch_npu extension; install the "
                "torch_npu build matching this torch and CANN release"
            )
        device = torch.device(explicit)
        if not is_accelerator(device):
            return device
        index = device.index
        if index is None:
            index = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(device.type, index)
        getattr(torch, device.type).set_device(device)
        return device
    device_type = available_type()
    if device_type == "cpu":
        return torch.device("cpu")
    device = torch.device(device_type, int(os.environ.get("LOCAL_RANK", 0)))
    getattr(torch, device_type).set_device(device)
    return device


def manual_seed_all(device: torch.device, seed: int) -> None:
    """Seed every card of ``device``'s family; a no-op on the CPU."""
    module = backend(device)
    if module is not None:
        module.manual_seed_all(seed)


def mem_get_info(device: torch.device) -> tuple[int, int] | None:
    """Free and total device memory, or ``None`` when the family reports none."""
    getter = getattr(backend(device), "mem_get_info", None)
    return None if getter is None else getter(device)


def empty_cache(device: torch.device) -> None:
    """Return cached blocks to the allocator; a no-op on the CPU."""
    module = backend(device)
    if module is not None:
        module.empty_cache()


def oom_error(device: torch.device) -> type[BaseException]:
    """The out-of-memory exception ``device``'s allocator raises."""
    return getattr(backend(device), "OutOfMemoryError", torch.cuda.OutOfMemoryError)


def fork_rng(device: torch.device):
    """Fork the CPU RNG plus ``device``'s generator; both restore on exit.

    ``fork_rng`` forks the CUDA generators unless told otherwise, so only a
    non-CUDA accelerator has to name its family. Passing ``device_type`` in
    just that case keeps the call compatible with the oldest supported torch.
    """
    module = backend(device)
    if module is None:
        return torch.random.fork_rng(devices=[])
    index = device.index if device.index is not None else module.current_device()
    named = {} if device.type == "cuda" else {"device_type": device.type}
    return torch.random.fork_rng(devices=[index], **named)
