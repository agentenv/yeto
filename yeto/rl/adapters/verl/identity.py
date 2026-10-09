"""Backend identity of the verl adapter (decoupling 6.1; rl-verl-backend D6).

Engine ``verl``, the pinned fork commit, device family ``nvidia`` (first
step; Ascend later) and the hash of the parameter-name map
(``param_names.PARAM_MAP``).  A verl island and a Miles island therefore never
share a syncer session contract (first version forbids mixing, D6).
"""

from __future__ import annotations

from yeto.rl.engine.backend_identity import BackendIdentity, param_map_sha256

from .param_names import PARAM_MAP
from .pins import VERL_COMMIT

ENGINE = "verl"
DEVICE_FAMILY = "nvidia"
PARAM_MAP_SHA256 = param_map_sha256(PARAM_MAP)


def backend_identity(rl_engine: str = "ports", *, device_family: str = DEVICE_FAMILY,
                     train_backend: str = "fsdp2", compat_group: str | None = None) -> BackendIdentity:
    """Identity of a verl island (``rl_engine`` is accepted for the Miles-shaped call sites)."""
    if train_backend != "fsdp2":
        raise ValueError(f"verl train backend {train_backend!r} has no parameter-name map yet")
    from yeto.hw.catalog import island_compat_group

    return BackendIdentity(ENGINE, VERL_COMMIT, device_family, PARAM_MAP_SHA256,
                           island_compat_group(compat_group))
