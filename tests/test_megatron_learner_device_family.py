"""S25 N7.7: the Megatron learner takes its card and collective from the device family."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from yeto import accel
from yeto.gpu_spec import DEVICE_FAMILY_ENV
from yeto.megatron import learner


@pytest.fixture
def megatron_stub(monkeypatch):
    """The two Megatron-Core entry points _init_distributed calls, recorded."""
    calls: list[tuple] = []
    parallel_state = SimpleNamespace(initialize_model_parallel=lambda **kw: calls.append(("mp", kw)))
    random = SimpleNamespace(model_parallel_cuda_manual_seed=lambda seed: calls.append(("seed", seed)))
    modules = {
        "megatron": ModuleType("megatron"),
        "megatron.core": SimpleNamespace(parallel_state=parallel_state),
        "megatron.core.tensor_parallel": ModuleType("megatron.core.tensor_parallel"),
        "megatron.core.tensor_parallel.random": random,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(dist, "init_process_group", lambda **kw: calls.append(("pg", kw)))
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    return calls


def _args(family):
    return SimpleNamespace(device_family=family, tensor_parallel=1, pipeline_parallel=1,
                           expert_parallel=1, seed=7)


@pytest.mark.parametrize(("family", "backend"), [("cuda", "nccl"), ("ppu", "nccl"), ("npu", "hccl")])
def test_init_distributed_follows_the_family(megatron_stub, monkeypatch, family, backend):
    bound = []
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setattr(accel, "bind_family_device",
                        lambda fam, rank: bound.append((fam, rank)) or SimpleNamespace(type=fam, index=rank))
    rank, world, device = learner._init_distributed(_args(family))
    assert (rank, world, device.index) == (0, 1, 3)
    assert bound == [(family, 3)]
    assert megatron_stub[0] == ("pg", {"backend": backend})
    assert megatron_stub[-1] == ("seed", 7)


def test_family_comes_from_the_launcher_env(megatron_stub, monkeypatch):
    monkeypatch.setenv(DEVICE_FAMILY_ENV, "npu")
    monkeypatch.setattr(accel, "bind_family_device", lambda fam, rank: torch.device("cpu"))
    learner._init_distributed(_args(None))
    assert megatron_stub[0] == ("pg", {"backend": "hccl"})


def test_cli_accepts_only_known_families():
    base = ["--model", "m", "--data", "d"]
    assert learner.parse_args(base).device_family is None
    assert learner.parse_args([*base, "--device-family", "ppu"]).device_family == "ppu"
    with pytest.raises(SystemExit):
        learner.parse_args([*base, "--device-family", "tpu"])
