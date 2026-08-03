"""Accelerator-family abstraction for torch learners.

torch has no device-generic API for seeding, memory queries, or the caching
allocator: each vendor gets its own namespace (``torch.cuda``, ``torch.npu``)
that happens to spell those calls identically. The learners also have a few
family-specific dtype and launcher policies, so those decisions live here
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

# Visible-card environment variables, in accelerator auto-detection priority.
_VISIBLE_DEVICE_ENVS = {
    "cuda": "CUDA_VISIBLE_DEVICES",
    "npu": "ASCEND_RT_VISIBLE_DEVICES",
}
_ACCELERATORS = tuple(_VISIBLE_DEVICE_ENVS)

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


def visible_devices_env(device_type: str) -> str | None:
    """The environment variable that limits visible cards of this family."""
    return _VISIBLE_DEVICE_ENVS.get(device_type)


def available_type() -> str:
    """The accelerator family visible on this node, or ``"cpu"``."""
    register_backends()
    for name in _ACCELERATORS:
        module = getattr(torch, name, None)
        if module is not None and module.is_available():
            return name
    return "cpu"


def device_count(device: torch.device | None = None) -> int:
    """Number of cards in the selected family, or zero for the CPU.

    With no explicit device, count the family auto-detection would select.
    This is also the set whose RNG state must be preserved before a device is
    selected, such as while constructing a diffusion pipeline.
    """
    if device is None:
        device_type = available_type()
        module = (
            getattr(torch, device_type, None)
            if device_type in _ACCELERATORS
            else None
        )
    else:
        module = backend(device)
    return 0 if module is None else module.device_count()


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


def loss_metric_dtype(device: torch.device) -> torch.dtype:
    """Highest-precision loss telemetry dtype supported by its collective.

    HCCL reductions use float32 for this scalar. The established CUDA and CPU
    paths retain float64 telemetry; this policy does not affect model math.
    """
    return torch.float32 if device.type == "npu" else torch.float64


def diffusion_dtype(device: torch.device) -> torch.dtype:
    """Floating dtype used to load a diffusion pipeline on ``device``.

    Ascend uses bfloat16 and non-accelerator devices stay in float32. CUDA
    uses bfloat16 when both the architecture and torch report support,
    otherwise float16. A failed capability query retains the historical
    fallback to torch's bfloat16 support probe.
    """
    if device.type == "npu":
        return torch.bfloat16
    if device.type != "cuda":
        return torch.float32

    module = backend(device)
    get_capability = getattr(module, "get_device_capability", None)
    if get_capability is not None:
        try:
            major, _ = get_capability(device)
        except (AssertionError, RuntimeError, TypeError):
            try:
                major, _ = get_capability()
            except (AssertionError, RuntimeError, TypeError):
                major = None
        if major is not None and major < 8:
            return torch.float16
    if getattr(module, "is_bf16_supported", lambda: False)():
        return torch.bfloat16
    return torch.float16


def fork_rng(device: torch.device | None):
    """Fork the CPU RNG plus selected accelerator generators; restore all.

    ``fork_rng`` forks the CUDA generators unless told otherwise, so only a
    non-CUDA accelerator has to name its family. Passing ``device_type`` in
    just that case keeps the call compatible with the oldest supported torch.
    With no selected device, every card in the auto-detected family is forked.
    """
    if device is None:
        device_type = available_type()
        devices = list(range(device_count()))
        named = {} if device_type in ("cpu", "cuda") else {"device_type": device_type}
        return torch.random.fork_rng(devices=devices, **named)

    module = backend(device)
    if module is None:
        return torch.random.fork_rng(devices=[])
    index = device.index if device.index is not None else module.current_device()
    named = {} if device.type == "cuda" else {"device_type": device.type}
    return torch.random.fork_rng(devices=[index], **named)
