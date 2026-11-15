"""The single hardware-family table (gap T8-T11, S7).

Every assertion about an NVIDIA or ROCm value is a compatibility assertion:
those are the two families of the async GRPO delivery and none of their
externally visible values may change.
"""

from __future__ import annotations

import pytest

from yeto import accel, gpu_spec
from yeto.hw import card_telemetry, catalog


def test_launcher_families_unchanged():
    assert gpu_spec.DEVICE_FAMILIES == ("cuda", "npu", "ppu", "rocm")


@pytest.mark.parametrize(
    "name,family,vendor,device_type,backend,ray,telemetry",
    [
        ("cuda", "cuda", "nvidia", "cuda", "nccl", "GPU", "cuda"),
        ("nvidia", "cuda", "nvidia", "cuda", "nccl", "GPU", "cuda"),
        ("npu", "npu", "ascend", "npu", "hccl", "NPU", "npu"),
        ("ascend", "npu", "ascend", "npu", "hccl", "NPU", "npu"),
        ("rocm", "rocm", "amd", "cuda", "nccl", "GPU", "rocm"),
        ("amd", "rocm", "amd", "cuda", "nccl", "GPU", "rocm"),
        ("ppu", "ppu", "thead", "cuda", "nccl", "GPU", "ppu"),
        ("thead", "ppu", "thead", "cuda", "nccl", "GPU", "ppu"),
    ],
)
def test_hw_family_both_spellings(name, family, vendor, device_type, backend, ray, telemetry):
    """T10: the launcher family and the identity vendor resolve to one record."""
    hw = gpu_spec.hw_family(name.upper())
    assert (hw.family, hw.vendor, hw.device_type) == (family, vendor, device_type)
    assert (hw.dist_backend, hw.ray_resource, hw.telemetry) == (backend, ray, telemetry)
    assert gpu_spec.normalize_family(name) == family
    # every telemetry key is a collector the card-telemetry module knows
    assert card_telemetry.collector_for(hw.telemetry, nvml_loader=lambda: None) is not None


def test_unknown_family_refused_but_fallback_available():
    with pytest.raises(gpu_spec.UnknownFamily):
        gpu_spec.hw_family("gaudi")
    assert gpu_spec.hw_family_or("gaudi").family == "cuda"
    assert gpu_spec.hw_family_or(None).family == "cuda"
    assert gpu_spec.hw_family_or("npu").family == "npu"


def test_accel_knows_rocm():
    """T8: YETO_DEVICE_FAMILY=rocm used to raise out of device_family()."""
    assert accel.device_family("rocm") == "rocm"
    assert accel.family_device_type("rocm") == "cuda"
    assert accel.family_dist_backend("rocm") == "nccl"
    # unchanged families
    assert accel.FAMILY_DEVICE_TYPES["cuda"] == "cuda"
    assert accel.family_dist_backend("npu") == "hccl"
    assert accel.family_dist_backend("cpu") == "gloo"


def test_identity_vendor_default_is_unchanged(monkeypatch):
    """T9/T11: the identity family only changes with the switch on."""
    monkeypatch.setenv(gpu_spec.DEVICE_FAMILY_ENV, "rocm")
    monkeypatch.delenv(catalog.IDENTITY_FROM_ENV_FAMILY, raising=False)
    assert catalog.device_family() == "nvidia"  # historical value, identity hash
    assert catalog.ray_accelerator_resource() == "GPU"
    monkeypatch.setenv(catalog.IDENTITY_FROM_ENV_FAMILY, "1")
    assert catalog.device_family() == "amd"
    monkeypatch.setenv(gpu_spec.DEVICE_FAMILY_ENV, "npu")
    assert catalog.device_family() == "ascend"
    assert catalog.ray_accelerator_resource() == "NPU"
    monkeypatch.setenv(gpu_spec.DEVICE_FAMILY_ENV, "ppu")
    assert catalog.device_family() == "thead"
    monkeypatch.setenv(gpu_spec.DEVICE_FAMILY_ENV, "gaudi")  # unknown: no change
    assert catalog.device_family() == "nvidia"


def test_vendor_and_ray_tables_match_catalog_cards():
    assert catalog.VENDOR_BY_DEVICE_FAMILY == {"cuda": "nvidia", "npu": "ascend",
                                               "ppu": "thead", "rocm": "amd"}
    assert catalog.RAY_RESOURCE_BY_DEVICE_FAMILY["nvidia"] == "GPU"
    assert catalog.RAY_RESOURCE_BY_DEVICE_FAMILY["ascend"] == "NPU"
    # every card's family/vendor pair agrees between the two modules
    for card, vendor in catalog.CARD_VENDOR.items():
        assert catalog.VENDOR_BY_DEVICE_FAMILY[gpu_spec.device_family_of(card)] == vendor
        assert gpu_spec.hw_family(vendor).device_type == gpu_spec.device_type_of(card)



def test_telemetry_unknown_family_warns(monkeypatch, caplog):
    """An unknown $YETO_DEVICE_FAMILY falls back to cuda but says so."""
    from yeto.rl.engine import telemetry

    monkeypatch.setenv(gpu_spec.DEVICE_FAMILY_ENV, "gaudi")
    with caplog.at_level("WARNING", logger=telemetry.__name__):
        telemetry.ResourceSampler(emit=lambda *a, **k: None, nvml_loader=lambda: None)
    msg = " ".join(r.getMessage() for r in caplog.records if r.levelname == "WARNING")
    assert "'gaudi'" in msg and "'cuda'" in msg
    caplog.clear()
    monkeypatch.setenv(gpu_spec.DEVICE_FAMILY_ENV, "rocm")
    with caplog.at_level("WARNING", logger=telemetry.__name__):
        telemetry.ResourceSampler(emit=lambda *a, **k: None, nvml_loader=lambda: None)
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_accel_spellings_consistent():
    """The three accel helpers accept the same names (no vendor spellings)."""
    for fn in (accel.device_family, accel.family_device_type, accel.family_dist_backend):
        with pytest.raises(ValueError):
            fn("ascend")
