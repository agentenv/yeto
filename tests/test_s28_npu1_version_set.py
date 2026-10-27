"""S28 G-M2 (W1): npu1's measured version set is a second known-good Ascend set, beside the
build-plan pins; the island accepts a full match of either set and nothing in between."""

from __future__ import annotations

import importlib.metadata as md
import sys
import types
from types import SimpleNamespace

import pytest

from yeto.rl.adapters.verl import pins


def test_primary_npu_pins_are_unchanged():
    assert pins.expected_versions("ascend") == pins.EXPECTED_VERSIONS_NPU
    assert pins.EXPECTED_VERSIONS_NPU["vllm"] == "0.23.0"
    sets = pins.known_version_sets("fsdp", "ascend")
    assert list(sets) == ["vllm-ascend-0.23-build", "npu1-q38-verl-loop-8.5.2"]
    assert sets["npu1-q38-verl-loop-8.5.2"]["vllm_ascend"] == "0.17.0rc2.dev109+g54879467c"


def test_megatron_sets_carry_the_megatron_stack():
    for vs in pins.known_version_sets("megatron", "ascend").values():
        assert vs["mindspeed"] == pins.EXPECTED_VERSIONS_MEGATRON_ASCEND["mindspeed"]


def test_unknown_family_is_refused():
    with pytest.raises(ValueError):
        pins.known_version_sets("fsdp", "amd")


def _manifest(monkeypatch, versions: dict):
    from yeto.rl.adapters.verl import island_entry

    monkeypatch.setattr(md, "version", lambda name: versions[name.replace("-", "_")]
                        if name.replace("-", "_") in versions else (_ for _ in ()).throw(md.PackageNotFoundError(name)))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(__version__=versions["torch"]))
    monkeypatch.setitem(sys.modules, "torch_npu", types.SimpleNamespace(__version__=versions["torch_npu"]))
    monkeypatch.setattr(island_entry.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="", returncode=0))
    import yeto.hw.catalog as catalog

    monkeypatch.setattr(catalog, "runtime_versions", lambda family: {})
    return island_entry.runtime_manifest(0, device_family="ascend")


def test_npu1_set_matches_in_full(monkeypatch):
    m = _manifest(monkeypatch, dict(pins.EXPECTED_VERSIONS_NPU_NPU1))
    assert m["version_set"] == "npu1-q38-verl-loop-8.5.2"
    assert m["expected_versions"] == pins.EXPECTED_VERSIONS_NPU_NPU1
    assert [p for p in m["problems"] if "verl commit" not in p] == []


def test_a_mix_of_two_sets_is_refused(monkeypatch):
    mixed = dict(pins.EXPECTED_VERSIONS_NPU_NPU1, vllm="0.23.0")
    m = _manifest(monkeypatch, mixed)
    assert m["version_set"] is None
    assert any(p.startswith("no known version set matches") for p in m["problems"])
    assert any(p.startswith("vllm_ascend ") for p in m["problems"])
