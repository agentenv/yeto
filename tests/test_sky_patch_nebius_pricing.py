"""yeto.sky_patches.nebius: a preemptible Nebius InstanceSpec carries a pricing model
(the Nebius API refused sky 0.13.0's request without one, S19 real-spot drill 10-10)."""

from __future__ import annotations

from types import SimpleNamespace

from yeto import sky_patches
from yeto.sky_patches import nebius as n


class _Compute:
    class FollowsSpotPriceSpec:
        pass

    def InstanceSpec(self, **kw):  # noqa: N802
        return kw


def _module():
    adaptor = SimpleNamespace(compute=lambda: _Compute(), sdk=lambda: "sdk")
    return SimpleNamespace(nebius=adaptor, launch=lambda: None)


def test_preemptible_spec_gets_follows_spot_price():
    m = _module()
    assert n.apply(m, version="0.13.0") == (True, "")
    spec = m.nebius.compute().InstanceSpec(preemptible="P", resources="R")
    assert isinstance(spec["follows_spot_price"], _Compute.FollowsSpotPriceSpec)
    assert spec["preemptible"] == "P" and spec["resources"] == "R"
    assert m.nebius.sdk() == "sdk"  # everything else passes through


def test_on_demand_and_explicit_pricing_untouched():
    m = _module()
    n.apply(m, version="0.13.0")
    c = m.nebius.compute()
    assert c.InstanceSpec(preemptible=None, resources="R") == {"preemptible": None, "resources": "R"}
    assert "follows_spot_price" not in c.InstanceSpec(preemptible="P", spot_pricing_policy="X")


def test_version_guard_and_idempotent():
    m = _module()
    ok, why = n.apply(m, version="0.14.0")
    assert not ok and "not in verified" in why
    assert n.apply(m, version="0.13.0")[0] and n.apply(m, version="0.13.0") == (True, "already applied")


def test_registered_and_in_pth_hook():
    assert n.TARGET_MODULE in sky_patches.status()
    assert "sky.provision.nebius.utils" in sky_patches.pth_line("/r")
