"""Backend identity of the Miles adapter (yeto-framework-decoupling 6.1, design D7).

Fixed values: engine ``miles``, the pinned Miles commit (``MILES_NEXT_COMMIT``),
device family ``nvidia`` (the only family Miles runs on here), and the hash of
the parameter-name map.  Miles' LoRA tensors already use yeto's canonical
names (``yeto.rl.core`` canonical specs), so the map is the identity map; it is
still hashed so a future non-identity map changes the identity.
"""

from __future__ import annotations

from yeto.rl.engine.backend_identity import BackendIdentity, param_map_sha256

from .pins import MILES_NEXT_COMMIT

ENGINE = "miles"
DEVICE_FAMILY = "nvidia"
PARAM_MAP = {"schema": "yeto-param-map-v1", "backend": ENGINE, "kind": "identity",
             "note": "Miles/Megatron LoRA tensor names are yeto canonical names"}
PARAM_MAP_SHA256 = param_map_sha256(PARAM_MAP)


def backend_identity(*, device_family: str = DEVICE_FAMILY,
                     compat_group: str | None = None) -> BackendIdentity:
    """Identity of a Miles island.
    ``compat_group`` None reads ``$YETO_RL_COMPAT_GROUP``; unset -> error (7.7a)."""
    from yeto.hw.catalog import island_compat_group

    return BackendIdentity(ENGINE, MILES_NEXT_COMMIT, device_family, PARAM_MAP_SHA256,
                           island_compat_group(compat_group))
