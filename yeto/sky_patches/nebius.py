"""Runtime fix for SkyPilot's Nebius provisioner (sky/provision/nebius/utils.py).

SkyPilot 0.13.0 asks for a preemptible VM with ``InstanceSpec(preemptible=...)``
only. The Nebius API now refuses that (seen 2026-10-10, S19 real-spot drill):
``INVALID_ARGUMENT: spec.pricing_model - pricing model must be specified for a
preemptible instance``. The SDK's ``InstanceSpec`` has a ``pricing_model`` oneof:
``on_demand`` | ``follows_spot_price`` | ``spot_pricing_policy``.

The patch makes every preemptible ``InstanceSpec`` built by that module carry
``follows_spot_price`` ("the preemptible VM accepts the current spot price") when
no pricing model was given. On-demand specs are untouched. Applied only on the
verified SkyPilot version.
"""

from __future__ import annotations

import sys

TARGET_MODULE = "sky.provision.nebius.utils"
VERIFIED_SKY_VERSIONS = frozenset({"0.13.0"})
PATCH_MARK = "_yeto_nebius_pricing_patched"
PRICING_FIELDS = ("on_demand", "follows_spot_price", "spot_pricing_policy")


def _sky_version() -> str | None:
    try:
        import sky

        return getattr(sky, "__version__", None)
    except Exception:  # noqa: BLE001
        return None


def with_spot_pricing(kwargs: dict, compute) -> dict:
    """InstanceSpec kwargs with ``follows_spot_price`` added to a preemptible spec
    that has no pricing model."""
    if kwargs.get("preemptible") is None or any(kwargs.get(f) is not None for f in PRICING_FIELDS):
        return kwargs
    return {**kwargs, "follows_spot_price": compute.FollowsSpotPriceSpec()}


class _ComputeProxy:
    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def InstanceSpec(self, *args, **kwargs):  # noqa: N802 - mirrors the SDK class name
        return self._inner.InstanceSpec(*args, **with_spot_pricing(kwargs, self._inner))


class _AdaptorProxy:
    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def compute(self, *args, **kwargs):
        return _ComputeProxy(self._inner.compute(*args, **kwargs))


def apply(module, version: str | None = None) -> tuple[bool, str]:
    if getattr(module, PATCH_MARK, False):
        return True, "already applied"
    version = version if version is not None else _sky_version()
    if version not in VERIFIED_SKY_VERSIONS:
        why = f"sky {version} not in verified {sorted(VERIFIED_SKY_VERSIONS)}"
        print(f"[yeto] NOT patching sky Nebius pricing: {why}", file=sys.stderr)
        return False, why
    if not hasattr(module, "nebius") or not hasattr(module, "launch"):
        return False, "sky nebius utils lacks nebius/launch"
    module.nebius = _AdaptorProxy(module.nebius)
    setattr(module, PATCH_MARK, True)
    return True, ""
