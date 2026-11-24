"""The inter-node fabric of a multi-node cloud island, from (device family, fabric, source).

A Nebius 8x H100 / H200 island is ordered with sky's network_tier="best" (InfiniBand),
so NCCL's IB transport stays on there; an AWS island ordered with the tier runs over
EFA; Modal keeps its RDMA; every other source runs NCCL over TCP, as before.
"""

from __future__ import annotations

import sys
import types

import pytest

from yeto import launcher
from yeto.gpu_spec import parse_gpu_spec
from yeto.launch import topology

OLD_TCP = ["export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}", "export NCCL_DEBUG=${NCCL_DEBUG:-WARN}"]
OLD_RDMA = ["export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}", "export NCCL_DEBUG=${NCCL_DEBUG:-WARN}"]


@pytest.mark.parametrize("cloud, gpu, per_node, tier, fabric", [
    ("nebius", "H100", 8, "auto", "ib"),
    ("nebius", "h200", 8, "auto", "ib"),
    ("nebius", "H100", 8, "none", "tcp"),  # no fabric tier ordered
    ("nebius", "L40S", 1, "auto", "tcp"),  # Nebius offers IB for 8x H100 / H200 only
    ("nebius", "H100", 4, "auto", "tcp"),
    ("aws", "H100", 8, "auto", "efa"),
    ("aws", "A10G", 1, "auto", "efa"),
    ("aws", "H100", 8, "none", "tcp"),
    ("gcp", "H100", 8, "auto", "tcp"),
    ("ssh", "B300", 8, "auto", "tcp"),
    ("modal", "B300", 8, "auto", "rdma"),
    ("modal", "H100", 8, "none", "rdma"),  # Modal clustered islands always ask for RDMA
])
def test_the_fabric_follows_the_order(cloud, gpu, per_node, tier, fabric):
    assert topology.multinode_fabric(cloud, gpu, per_node, num_nodes=2, network_tier=tier) == fabric


def test_a_cloud_alone_keeps_the_rule_of_before():
    # callers that pass no shape (no order known): Modal RDMA, TCP elsewhere
    assert topology.multinode_fabric("modal") == "rdma"
    for cloud in ("nebius", "aws", "gcp", "ssh", "verda"):
        assert topology.multinode_fabric(cloud) == "tcp"
        assert topology.multinode_env_prelude(cloud, 2).splitlines()[:2] == OLD_TCP
    assert topology.multinode_env_prelude("modal", 2).splitlines() == OLD_RDMA
    assert topology.multinode_env_prelude("nebius", 1, fabric="ib") == ""


def test_nebius_8xh100_two_nodes_keeps_infiniband_on():
    [spec] = parse_gpu_spec("nebius:2x8xh100")
    fabric = topology.island_fabric(types.SimpleNamespace(rl_island_network_tier="auto"), spec)
    lines = topology.multinode_env_prelude(spec.cloud, spec.num_nodes, fabric=fabric).splitlines()
    assert fabric == "ib"
    assert lines[0] == "export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}"
    assert not any("NCCL_IB_DISABLE:-1" in line for line in lines)
    assert any("GLOO_SOCKET_IFNAME" in line for line in lines)  # the Nebius NIC is still detected
    off = topology.island_fabric(types.SimpleNamespace(rl_island_network_tier="none"), spec)
    assert topology.multinode_env_prelude(spec.cloud, 2, fabric=off).splitlines()[0] == OLD_TCP[0]


def test_aws_over_efa_points_libfabric_at_it():
    lines = topology.multinode_env_prelude("aws", 2, fabric="efa").splitlines()
    assert lines == OLD_TCP + ["export FI_PROVIDER=${FI_PROVIDER:-efa}"]


def test_an_unknown_fabric_is_refused():
    with pytest.raises(ValueError, match="unknown multi-node fabric"):
        topology.multinode_env_prelude("nebius", 2, fabric="roce")


def _task(monkeypatch, gpu: str, *extra: str):
    from test_rl_launcher import _Resources, _Storage, _StorageMode, _Task, _args, _prepare_rl_args

    monkeypatch.setitem(sys.modules, "sky", types.SimpleNamespace(
        Task=_Task, Resources=_Resources, Storage=_Storage, StorageMode=_StorageMode))
    args = _args(("--gpu", gpu, "--rl-engine", "ports", "--rl-placement", "fixed-partition",
                  "--rl-rollout-gpus", "8", "--rollout-num-gpus-per-engine", "8", "--tensor-parallel", "2",
                  *extra))
    args.model_revision = "a" * 40
    args.data_revision = "b" * 40
    args.source_sha256 = "c" * 64
    args.reward_sha256 = "d" * 64
    _prepare_rl_args(args)
    return launcher.make_miles_island_task(args, parse_gpu_spec(args.gpu)[0], 0, 1, "127.0.0.1:29400")


def test_the_nebius_island_task_orders_ib_and_runs_nccl_over_it(monkeypatch):
    task = _task(monkeypatch, "nebius:2x8xh100")
    assert task.num_nodes == 2 and task.resources.network_tier == "best"
    assert "export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}" in task.run
    assert "NCCL_IB_DISABLE:-1" not in task.run


def test_an_island_that_orders_no_tier_runs_over_tcp(monkeypatch):
    task = _task(monkeypatch, "nebius:2x8xh100", "--rl-island-network-tier", "none")
    assert getattr(task.resources, "network_tier", None) is None
    assert "export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}" in task.run
