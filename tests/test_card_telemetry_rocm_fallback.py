"""ROCm card telemetry without amd-smi / pynvml: rocm-smi (short flag set), then the
amdgpu sysfs files; a cuda-family island on an AMD node falls back to that chain."""

from __future__ import annotations

import json

import pytest

from yeto.hw import card_telemetry as ct
from yeto.rl.engine.telemetry import ResourceSampler

ROCM_MIN = json.dumps({"card0": {"GPU use (%)": "37", "VRAM Total Memory (B)": str(288 * 2**30),
                                 "VRAM Total Used Memory (B)": str(100 * 2**30)},
                       "card1": {"GPU use (%)": "0", "VRAM Total Memory (B)": str(288 * 2**30),
                                 "VRAM Total Used Memory (B)": str(2**30)}})


def _runner(outputs):
    calls = []

    def run(argv):
        calls.append(tuple(argv))
        out = outputs.get(tuple(argv))
        if out is None:
            raise ct.CardTelemetryError(f"{argv[0]}: FileNotFoundError: not installed")
        return out

    run.calls = calls
    return run


def _sysfs(root, cards):
    for i, (used, total, busy) in enumerate(cards):
        dev = root / f"card{i}" / "device"
        dev.mkdir(parents=True)
        (dev / "mem_info_vram_used").write_text(f"{used}\n")
        (dev / "mem_info_vram_total").write_text(f"{total}\n")
        (dev / "gpu_busy_percent").write_text(f"{busy}\n")
    (root / "card0-DP-1").mkdir()  # a connector entry is skipped
    return root


def test_rocm_smi_short_flag_set(tmp_path):
    run = _runner({ct.RocmSmiCollector.ROCM_SMI_MIN_ARGV: ROCM_MIN})
    c = ct.rocm_collector(run, sysfs_root=str(tmp_path / "none"))
    c.open()
    assert c.source == "rocm-smi"
    cards = c.sample()
    assert [x["index"] for x in cards] == [0, 1]
    assert cards[0]["mem_used_mb"] == 100 * 1024 and cards[0]["util_pct"] == 37.0
    assert ("rocm-smi", "--showmeminfo", "vram", "--showuse", "--json") in run.calls


def test_sysfs_when_no_smi_tool(tmp_path):
    root = _sysfs(tmp_path / "drm", [(50 * 2**30, 288 * 2**30, 90), (2**30, 288 * 2**30, 0)])
    c = ct.rocm_collector(_runner({}), sysfs_root=str(root))
    c.open()
    assert c.source == "sysfs-amdgpu"
    cards = c.sample()
    assert [x["mem_used_mb"] for x in cards] == [50 * 1024, 1024]
    assert cards[0]["mem_total_mb"] == 288 * 1024 and cards[0]["util_pct"] == 90.0


def test_nothing_available_names_every_attempt(tmp_path):
    c = ct.rocm_collector(_runner({}), sysfs_root=str(tmp_path / "none"))
    with pytest.raises(ct.CardTelemetryError) as e:
        c.open()
    assert "amd-smi" in str(e.value) and "rocm-smi" in str(e.value) and "sysfs-amdgpu" in str(e.value)


def test_cuda_family_without_pynvml_falls_back_to_rocm(tmp_path, monkeypatch):
    def no_nvml():
        raise ModuleNotFoundError("No module named 'pynvml'")

    root = _sysfs(tmp_path / "drm", [(10 * 2**30, 288 * 2**30, 5)])
    monkeypatch.setattr(ct, "SYSFS_DRM", str(root))
    monkeypatch.setattr(ct, "rocm_collector", lambda run=ct.run_command, sysfs_root=str(root):
                        ct.FallbackCollector(ct.RocmSmiCollector(_runner({})), ct.SysfsAmdCollector(sysfs_root)))
    monkeypatch.delenv("YETO_DEVICE_FAMILY", raising=False)
    events = []
    s = ResourceSampler(lambda ev, **f: events.append(f), nvml_loader=no_nvml, interval_s=3600)
    assert s.tick()
    (rec,) = events
    assert rec["available"] is True and rec["source"] == "sysfs-amdgpu"
    assert rec["gpus"][0]["mem_used_mb"] == 10 * 1024


def test_cuda_with_nvml_is_unchanged():
    class Nvml:
        def nvmlInit(self):
            pass

        def nvmlDeviceGetCount(self):
            return 0

    c = ct.collector_for("cuda", nvml_loader=lambda: Nvml(), run=_runner({}))
    c.open()
    assert c.source == "nvml"
