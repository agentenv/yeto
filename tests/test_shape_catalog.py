from types import SimpleNamespace

import pytest

from yeto.shape import catalog
from yeto.shape.catalog import (
    PEAK_TFLOPS_BF16,
    Offering,
    effective_tflops,
    efa_capable,
    goodput,
    list_offerings,
    mfu,
)


def _offering(**kw) -> Offering:
    base = dict(
        gpu="A100-80GB",
        instance_type="p4de.24xlarge",
        gpus_per_node=8,
        vcpus=96,
        region="us-east-1",
        spot_price=12.0,
        on_demand_price=40.96,
        gpu_mem_gb=80,
    )
    base.update(kw)
    return Offering(**base)


def test_effective_tflops_single_node_a100_80():
    off = _offering()
    # 1 node, no multi-node penalty (mfu 0.35), score 8 -> goodput 0.5+0.05*8
    assert effective_tflops(off, nodes=1, score=8) == 1 * 8 * 312.0 * 0.35 * (0.5 + 0.05 * 8)


def test_effective_tflops_two_node_efa_h100_score_none():
    off = _offering(gpu="H100", instance_type="p5.48xlarge", gpu_mem_gb=80)
    # p5 is EFA-capable -> mfu 0.30; score None -> goodput fallback 0.85
    assert effective_tflops(off, nodes=2, score=None) == 2 * 8 * 989.0 * 0.30 * 0.85


def test_goodput_edges():
    assert goodput(None) == 0.85
    assert goodput(10) == 0.98  # capped: 0.5 + 0.5 would be 1.0
    assert goodput(9) == 0.95  # below the cap, linear region
    assert goodput(0) == 0.5


def test_mfu_and_efa():
    assert mfu(1, efa=False) == 0.35
    assert mfu(1, efa=True) == 0.35  # single node never pays a fabric penalty
    assert mfu(2, efa=True) == 0.30
    assert mfu(2, efa=False) == 0.20
    assert efa_capable("p4d.24xlarge")
    assert efa_capable("p4de.24xlarge")
    assert efa_capable("p5.48xlarge")
    assert not efa_capable("g5.48xlarge")
    # so a multi-node g5 island lands on the TCP mfu
    assert mfu(2, efa_capable("g5.48xlarge")) == 0.20


def _fake_raw():
    def info(instance_type, count, cpus, price, spot, region):
        return SimpleNamespace(
            instance_type=instance_type,
            accelerator_count=count,
            cpu_count=cpus,
            price=price,
            spot_price=spot,
            region=region,
        )

    return {
        "A100-80GB": [
            info("p4de.24xlarge", 8.0, 96.0, 40.96, 12.3, "us-east-1"),
            info("fake.4x", 4.0, 48.0, 20.0, 6.0, "us-east-1"),
            info("p4de.24xlarge", 8.0, 96.0, 40.96, 14.0, "eu-west-1"),  # region filtered
            info(None, 8.0, 96.0, None, None, "us-east-1"),  # no instance type
        ],
        "H100": [info("p5.48xlarge", 8.0, 192.0, 98.32, 30.0, "us-east-1")],
        "K80": [info("p2.xlarge", 1.0, 4.0, 0.9, 0.3, "us-east-1")],  # unknown GPU
    }


def test_list_offerings_filters_converts_and_sorts(monkeypatch):
    monkeypatch.setattr(catalog, "_fetch_raw", lambda regions, gpus, clouds: _fake_raw())

    offs = list_offerings(["us-east-1"])
    # K80 (not in PEAK_TFLOPS_BF16), eu-west-1, and the None instance_type
    # rows are all gone; A100-80GB sorts before H100, and within a
    # (gpu, region) group bigger nodes come first.
    assert [(o.gpu, o.instance_type) for o in offs] == [
        ("A100-80GB", "p4de.24xlarge"),
        ("A100-80GB", "fake.4x"),
        ("H100", "p5.48xlarge"),
    ]
    first = offs[0]
    assert first.gpus_per_node == 8 and isinstance(first.gpus_per_node, int)
    assert first.vcpus == 96 and isinstance(first.vcpus, int)
    assert first.gpu_mem_gb == 80  # pulled from launcher.GPU_MEM_GB
    assert first.spot_price == 12.3
    assert first.on_demand_price == 40.96


def test_list_offerings_gpu_filter(monkeypatch):
    monkeypatch.setattr(catalog, "_fetch_raw", lambda regions, gpus, clouds: _fake_raw())
    offs = list_offerings(["us-east-1"], gpus=["H100"])
    assert [o.gpu for o in offs] == ["H100"]


class FakeCache:
    def __init__(self):
        self.store = {}

    def get_or(self, key, fetch):
        if key not in self.store:
            self.store[key] = fetch()
        return self.store[key]


def test_list_offerings_uses_cache(monkeypatch):
    calls = {"n": 0}

    def counting_fetch(regions, gpus, clouds):
        calls["n"] += 1
        return _fake_raw()

    monkeypatch.setattr(catalog, "_fetch_raw", counting_fetch)
    fake = FakeCache()
    a = list_offerings(["us-east-1"], gpus=["A100-80GB"], cache=fake)
    b = list_offerings(["us-east-1"], gpus=["A100-80GB"], cache=fake)
    assert calls["n"] == 1  # second call served from the fake cache
    assert a == b
    assert list(fake.store) == ["catalog:v2:aws:us-east-1:A100-80GB"]


def test_peak_tflops_keys_subset_of_launcher_mem_table():
    from yeto import launcher

    assert set(PEAK_TFLOPS_BF16) <= set(launcher.GPU_MEM_GB)


def test_bf16_capability_gate():
    from yeto.shape.catalog import supports_bf16

    assert supports_bf16("B200")
    assert supports_bf16("A100")
    assert not supports_bf16("V100")  # pre-Ampere, no bf16
    assert not supports_bf16("T4")
