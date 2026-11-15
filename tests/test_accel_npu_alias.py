"""Ascend ranks under ``torch_npu.contrib.transfer_to_npu`` (MindSpeed imports it):
``torch.cuda`` answers for the NPU, so the device family must not read as CUDA,
and a verl cut must save and restore the NPU RNG. CPU only (a fake ``torch.npu``)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from yeto import accel
from yeto.hw import catalog
from yeto.rl.adapters.verl import cut as vcut
from yeto.rl.engine.cut import CutError


class _FakeNpu:
    def __init__(self, available: bool = True) -> None:
        self.available = available
        self.state = torch.tensor([7, 7, 7], dtype=torch.uint8)
        self.restored = None

    def is_available(self) -> bool:
        return self.available

    def get_rng_state(self):
        return self.state.clone()

    def set_rng_state(self, state) -> None:
        self.restored = state


@pytest.fixture()
def ascend_alias(monkeypatch):
    """A torch_npu build (no CUDA, no HIP) whose torch.cuda is the NPU alias."""
    npu = _FakeNpu()
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    return npu


def test_cuda_alias_on_an_ascend_build_reads_as_npu(ascend_alias):
    assert accel.cuda_is_npu_alias()
    assert accel.available_type() == "npu"
    assert catalog.device_family() == "ascend"
    assert accel.dist_backend() == "hccl"
    assert catalog.ray_accelerator_resource() == "NPU"


def test_a_cuda_build_with_cuda_cards_still_reads_as_cuda(monkeypatch):
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert not accel.cuda_is_npu_alias()
    assert accel.available_type() == "cuda"
    assert catalog.device_family() == "nvidia"


def test_cut_rng_saves_and_restores_the_npu_state(ascend_alias):
    rng = vcut._rng_state()
    assert set(rng) == {"cpu", "npu"}
    assert torch.equal(rng["npu"], ascend_alias.state)
    vcut._set_rng_state(rng)
    assert torch.equal(ascend_alias.restored, ascend_alias.state)
    assert vcut.rng_digest(rng) == vcut.rng_digest(vcut._rng_state())


def test_an_npu_rng_state_needs_an_npu(monkeypatch):
    monkeypatch.setattr(torch, "npu", SimpleNamespace(is_available=lambda: False), raising=False)
    with pytest.raises(CutError, match="NPU RNG state but this rank has no NPU device"):
        vcut._set_rng_state({"cpu": torch.get_rng_state(), "npu": torch.zeros(3, dtype=torch.uint8)})


def test_a_cut_without_device_rng_restores_on_cpu():
    vcut._set_rng_state({"cpu": torch.get_rng_state()})


def test_an_old_alias_cut_with_a_cuda_rng_key_restores_with_the_same_digest(ascend_alias, monkeypatch):
    """main, on a MindSpeed (transfer_to_npu) rank, saved the NPU state as ``cuda``."""
    monkeypatch.setattr(torch.cuda, "set_rng_state", ascend_alias.set_rng_state)  # the alias
    old = {"cpu": torch.get_rng_state(), "cuda": ascend_alias.state.clone()}
    vcut._set_rng_state(old)
    assert torch.equal(ascend_alias.restored, ascend_alias.state)
    assert vcut.rng_digest(vcut._rng_state_as_saved(old)) == vcut.rng_digest(old)
    assert set(vcut._rng_state_as_saved({"cpu": old["cpu"], "npu": old["cuda"]})) == {"cpu", "npu"}


def test_a_rocm_build_still_reads_as_cuda(monkeypatch):
    monkeypatch.setattr(torch, "npu", _FakeNpu(), raising=False)
    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(torch.version, "hip", "7.0", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert not accel.cuda_is_npu_alias()
    assert accel.available_type() == "cuda"


def test_a_cuda_cut_rng_keeps_the_cuda_key_and_digest(monkeypatch):
    state = torch.tensor([1, 2, 3], dtype=torch.uint8)
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: state.clone())
    rng = vcut._rng_state()
    assert set(rng) == {"cpu", "cuda"}
    assert vcut._rng_state_as_saved(rng).keys() == rng.keys()
    assert vcut.rng_digest(vcut._rng_state_as_saved(rng)) == vcut.rng_digest(rng)
