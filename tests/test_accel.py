"""Accelerator-family dispatch, exercised without an accelerator.

The Ascend path cannot be run here, so ``torch.npu`` is stubbed with the
namespace ``torch_npu`` would install. That covers everything except the
literal ``torch.device("npu", ...)`` construction, which torch rejects until
the real extension registers the device type; the surrounding logic is shared
with CUDA and is covered through a stubbed CUDA namespace instead.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

from yeto import accel


class FakeOutOfMemory(RuntimeError):
    pass


class FakeAccelerator:
    """The subset of ``torch.cuda``/``torch.npu`` that accel.py calls."""

    OutOfMemoryError = FakeOutOfMemory

    def __init__(
        self,
        available: bool = True,
        free_total: tuple[int, int] = (2, 8),
        count: int = 4,
    ):
        self.available = available
        self.free_total = free_total
        self.count = count
        self.seeded: list[int] = []
        self.bound: list[torch.device] = []
        self.empties = 0

    def is_available(self) -> bool:
        return self.available

    def set_device(self, device) -> None:
        self.bound.append(device)

    def manual_seed_all(self, seed: int) -> None:
        self.seeded.append(seed)

    def mem_get_info(self, device):
        return self.free_total

    def empty_cache(self) -> None:
        self.empties += 1

    def current_device(self) -> int:
        return 3

    def device_count(self) -> int:
        return self.count


def device(device_type: str, index: int | None = None):
    """A device stand-in: accel.py reads only ``.type`` and ``.index``."""
    return SimpleNamespace(type=device_type, index=index)


@pytest.fixture
def npu(monkeypatch):
    """Install a stub ``torch.npu``, as importing torch_npu would."""
    stub = FakeAccelerator()
    monkeypatch.setattr(torch, "npu", stub, raising=False)
    return stub


@pytest.fixture
def no_cuda(monkeypatch):
    """Hide any real CUDA so auto-detection reaches the next family."""
    monkeypatch.setattr(torch, "cuda", FakeAccelerator(available=False))


# --- family resolution ---------------------------------------------------


def test_backend_resolves_the_vendor_namespace(npu):
    assert accel.backend(device("cuda", 0)) is torch.cuda
    assert accel.backend(device("npu", 0)) is npu
    assert accel.backend(device("cpu")) is None


def test_cpu_is_not_an_accelerator():
    assert accel.is_accelerator(device("cuda", 0))
    assert accel.is_accelerator(device("npu", 0))
    assert not accel.is_accelerator(device("cpu"))


def test_visible_devices_env_is_owned_by_the_accelerator_abstraction():
    assert accel.visible_devices_env("cuda") == "CUDA_VISIBLE_DEVICES"
    assert accel.visible_devices_env("npu") == "ASCEND_RT_VISIBLE_DEVICES"
    assert accel.visible_devices_env("cpu") is None


def test_available_type_falls_back_to_cpu(no_cuda, monkeypatch):
    monkeypatch.delattr(torch, "npu", raising=False)
    assert accel.available_type() == "cpu"


def test_available_type_finds_the_npu_when_cuda_is_absent(no_cuda, npu):
    assert accel.available_type() == "npu"


def test_device_count_uses_the_selected_family(no_cuda, npu):
    npu.count = 2
    assert accel.device_count() == 2
    assert accel.device_count(device("npu")) == 2
    assert accel.device_count(device("cpu")) == 0


def test_device_count_is_zero_when_auto_detection_selects_cpu(no_cuda, monkeypatch):
    monkeypatch.delattr(torch, "npu", raising=False)
    assert accel.device_count() == 0


def test_cuda_wins_when_both_families_are_present(npu, monkeypatch):
    monkeypatch.setattr(torch, "cuda", FakeAccelerator(available=True))
    assert accel.available_type() == "cuda"


# --- collective backend --------------------------------------------------


def test_dist_backend_is_hccl_on_ascend(no_cuda, npu):
    assert accel.dist_backend() == "hccl"


def test_dist_backend_is_gloo_without_an_accelerator(no_cuda, monkeypatch):
    monkeypatch.delattr(torch, "npu", raising=False)
    assert accel.dist_backend() == "gloo"


def test_dist_backend_is_nccl_on_cuda(monkeypatch):
    monkeypatch.setattr(torch, "cuda", FakeAccelerator(available=True))
    assert accel.dist_backend() == "nccl"


def test_dist_backend_honors_the_selected_family(npu, monkeypatch):
    monkeypatch.setattr(torch, "cuda", FakeAccelerator(available=True))
    assert accel.dist_backend(device("npu", 0)) == "hccl"


# --- device detection ----------------------------------------------------


def test_explicit_device_is_honored(no_cuda):
    assert accel.detect("cpu").type == "cpu"


def test_explicit_accelerator_without_index_binds_this_rank(monkeypatch):
    stub = FakeAccelerator(available=True)
    monkeypatch.setattr(torch, "cuda", stub)
    monkeypatch.setenv("LOCAL_RANK", "2")

    resolved = accel.detect("cuda")

    assert (resolved.type, resolved.index) == ("cuda", 2)
    assert stub.bound == [resolved]


def test_explicit_npu_without_the_extension_names_torch_npu(monkeypatch):
    monkeypatch.delattr(torch, "npu", raising=False)
    with pytest.raises(RuntimeError, match="torch_npu"):
        accel.detect("npu")


def test_detect_binds_this_rank_to_its_local_card(monkeypatch):
    """The auto path the NPU shares with CUDA: LOCAL_RANK selects the card."""
    stub = FakeAccelerator(available=True)
    monkeypatch.setattr(torch, "cuda", stub)
    monkeypatch.setenv("LOCAL_RANK", "2")

    resolved = accel.detect(None)

    assert (resolved.type, resolved.index) == ("cuda", 2)
    assert stub.bound == [resolved]


def test_detect_returns_cpu_when_nothing_is_visible(no_cuda, monkeypatch):
    monkeypatch.delattr(torch, "npu", raising=False)
    assert accel.detect(None) == torch.device("cpu")


def test_register_backends_survives_a_missing_extension(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch_npu", raising=False)
    accel.register_backends()  # must not raise


# --- per-device primitives -----------------------------------------------


def test_seeding_reaches_the_npu_and_skips_the_cpu(npu):
    accel.manual_seed_all(device("npu", 0), 17)
    accel.manual_seed_all(device("cpu"), 99)
    assert npu.seeded == [17]


def test_mem_get_info_is_none_on_cpu(npu):
    assert accel.mem_get_info(device("npu", 0)) == (2, 8)
    assert accel.mem_get_info(device("cpu")) is None


def test_empty_cache_reaches_the_npu_and_skips_the_cpu(npu):
    accel.empty_cache(device("npu", 0))
    accel.empty_cache(device("cpu"))
    assert npu.empties == 1


def test_oom_error_is_the_vendor_exception(npu):
    assert accel.oom_error(device("npu", 0)) is FakeOutOfMemory
    assert accel.oom_error(device("cuda", 0)) is torch.cuda.OutOfMemoryError
    assert accel.oom_error(device("cpu")) is torch.cuda.OutOfMemoryError


def test_dtype_policies_preserve_each_device_family(monkeypatch):
    assert accel.loss_metric_dtype(device("npu")) is torch.float32
    assert accel.loss_metric_dtype(device("cuda")) is torch.float64
    assert accel.loss_metric_dtype(device("cpu")) is torch.float64
    assert accel.diffusion_dtype(device("npu")) is torch.bfloat16
    assert accel.diffusion_dtype(device("cpu")) is torch.float32

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (7, 5))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert accel.diffusion_dtype(device("cuda")) is torch.float16

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 0))
    assert accel.diffusion_dtype(device("cuda")) is torch.bfloat16


# --- probe RNG isolation -------------------------------------------------


@pytest.fixture
def recorded_fork(monkeypatch):
    calls = []
    real_fork_rng = torch.random.fork_rng

    def record(**kwargs):
        calls.append(kwargs)
        return real_fork_rng(devices=[])

    monkeypatch.setattr(torch.random, "fork_rng", record)
    return calls


def test_fork_rng_names_the_family_only_off_cuda(npu, recorded_fork):
    """torch<2.7 has no device_type kwarg, so CUDA must not pass one."""
    with accel.fork_rng(device("npu", 1)):
        pass
    with accel.fork_rng(device("cuda", 1)):
        pass
    assert recorded_fork == [
        {"devices": [1], "device_type": "npu"},
        {"devices": [1]},
    ]


def test_fork_rng_defaults_to_the_current_card(npu, recorded_fork):
    with accel.fork_rng(device("npu")):
        pass
    assert recorded_fork == [{"devices": [3], "device_type": "npu"}]


def test_fork_rng_forks_no_accelerator_on_cpu(recorded_fork):
    with accel.fork_rng(device("cpu")):
        pass
    assert recorded_fork == [{"devices": []}]


def test_fork_rng_without_a_device_forks_every_card_in_detected_family(
    no_cuda, npu, recorded_fork
):
    npu.count = 2
    with accel.fork_rng(None):
        pass
    assert recorded_fork == [{"devices": [0, 1], "device_type": "npu"}]


def test_fork_rng_without_a_device_forks_no_cards_on_cpu(
    no_cuda, monkeypatch, recorded_fork
):
    monkeypatch.delattr(torch, "npu", raising=False)
    with accel.fork_rng(None):
        pass
    assert recorded_fork == [{"devices": []}]
