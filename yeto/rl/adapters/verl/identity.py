"""Backend identity of the verl adapter (decoupling 6.1; rl-verl-backend D6).

Engine ``verl``, the pinned fork commit, the device family of the card this
island runs on (``nvidia`` on CUDA, ``ascend`` on an NPU) and the hash of the
parameter-name map (``param_names.PARAM_MAP``).  A verl island and a Miles island therefore never
share a syncer session contract (first version forbids mixing, D6).
"""

from __future__ import annotations

from yeto.rl.engine.backend_identity import BackendIdentity, param_map_sha256

from .param_names import PARAM_MAP
from .pins import VERL_COMMIT

ENGINE = "verl"
# Default family when the caller names none and no accelerator is visible
# (unit tests, dry runs). A real island resolves it from its own device:
# ``yeto.hw.catalog.device_family()``.
DEVICE_FAMILY = "nvidia"
PARAM_MAP_SHA256 = param_map_sha256(PARAM_MAP)


def backend_identity(rl_engine: str = "ports", *, device_family: str | None = None,
                     train_backend: str = "fsdp2", compat_group: str | None = None) -> BackendIdentity:
    """Identity of a verl island (``rl_engine`` is accepted for the Miles-shaped call sites).

    ``device_family`` None means "read it from this node's accelerator", so an
    Ascend island declares ``ascend`` without any call site change. The family
    is part of the identity hash: a GPU island and an NPU island therefore get
    different session contracts and cannot merge weights (task 3.10).
    """
    if train_backend != "fsdp2":
        raise ValueError(f"verl train backend {train_backend!r} has no parameter-name map yet")
    from yeto.hw.catalog import device_family as detect_device_family
    from yeto.hw.catalog import island_compat_group

    family = device_family if device_family is not None else detect_device_family()
    return BackendIdentity(ENGINE, VERL_COMMIT, family, PARAM_MAP_SHA256,
                           island_compat_group(compat_group))
