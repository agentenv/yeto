"""The inter-node fabric of a multi-node cloud island, from (device family, fabric, source).

A Nebius 8x H100 / H200 island is ordered with sky's network_tier="best" (InfiniBand),
so NCCL's IB transport stays on there unless the container cannot reach it; an AWS EFA
instance ordered with the tier runs over EFA when the device is there; Modal keeps its
RDMA; every other source and shape runs NCCL over TCP, as before.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types

import pytest

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
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
    ("aws", "A10G", 1, "auto", "tcp"),  # no EFA on the instance
    ("aws", "A10G", 8, "auto", "tcp"),
    ("aws", "H200", 8, "auto", "efa"),
    ("aws", "H100", 4, "auto", "tcp"),
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
    assert lines[2] == topology._IB_GUARD  # off, and said why, when the container cannot reach IB
    assert not any("NCCL_IB_DISABLE:-1" in line for line in lines)
    assert any("GLOO_SOCKET_IFNAME" in line for line in lines)  # the Nebius NIC is still detected
    off = topology.island_fabric(types.SimpleNamespace(rl_island_network_tier="none"), spec)
    assert topology.multinode_env_prelude(spec.cloud, 2, fabric=off).splitlines()[0] == OLD_TCP[0]


def test_aws_over_efa_points_libfabric_at_it_when_the_device_is_there():
    lines = topology.multinode_env_prelude("aws", 2, fabric="efa").splitlines()
    assert lines == OLD_TCP + [topology._EFA_GUARD]
    assert "FI_PROVIDER=${FI_PROVIDER:-efa}" in topology._EFA_GUARD and "uverbs" in topology._EFA_GUARD


def _run_guard(guard: str, memlock: str) -> tuple[str, str]:
    # the guard as a shell runs it, with ulimit stubbed (the test host's real limit varies)
    script = f'ulimit() {{ echo {memlock}; }}\nexport NCCL_IB_DISABLE=0\n{guard}\necho "IB=$NCCL_IB_DISABLE"'
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout
    return out.strip().splitlines()[-1], out


def test_the_ib_guard_turns_ib_off_with_a_limited_memlock():
    last, out = _run_guard(topology._IB_GUARD, "65536")
    assert last == "IB=1" and "NCCL IB off" in out and "memlock 65536" in out


def test_the_ib_guard_keeps_ib_only_with_the_device_and_unlimited_memlock():
    last, _ = _run_guard(topology._IB_GUARD, "unlimited")
    assert last == ("IB=0" if os.path.isdir("/dev/infiniband") else "IB=1")


def test_the_inventory_fabrics_are_the_same_vocabulary():
    from yeto.launch.node_inventory import FABRICS

    assert topology.RDMA_FABRICS == {"ib", "roce"} and topology.RDMA_FABRICS <= set(FABRICS)
    assert topology.RDMA_FABRICS < topology.IB_FABRICS == {"rdma", "ib", "roce"}
    roce = topology.multinode_env_prelude("ssh", 2, fabric="roce").splitlines()
    assert roce[0] == "export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}" and roce[2] == topology._IB_GUARD


def test_modal_rdma_has_no_guard():
    # Modal's RDMA is what the platform provides: the prelude stays byte for byte
    assert topology.multinode_env_prelude("modal", 2, fabric="rdma").splitlines() == OLD_RDMA


def test_an_unknown_fabric_is_refused():
    with pytest.raises(ValueError, match="unknown multi-node fabric"):
        topology.multinode_env_prelude("nebius", 2, fabric="hccs")


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


# --- the dry-run note for an island on TCP ------------------------------------------------

def _notes(gpu: str, tier: str = "auto", inventory=None):
    args = types.SimpleNamespace(rl_island_network_tier=tier, rl_node_inventory_nodes=inventory)
    return topology.fabric_bandwidth_notes(args, parse_gpu_spec(gpu))


def test_an_island_on_tcp_gets_the_bandwidth_note():
    [note] = _notes("nebius:2x1xl40s")
    assert note.startswith("[preflight] note: island 0 (nebius:2x1xL40S)")
    assert "over TCP sockets" in note and "cross-node bandwidth is limited" in note
    assert "nebius offers IB for H100:8, H200:8 only" in note
    [note] = _notes("nebius:2x8xh100", tier="none")
    assert "--rl-island-network-tier none" in note
    [note] = _notes("gcp:2x8xh100")
    assert "yeto orders no IB or EFA tier on gcp" in note
    [note] = _notes("aws:1x8xh100,aws:2x8xa10g")  # one note, for the multi-node TCP island
    assert note.startswith("[preflight] note: island 1 (aws:2x8xA10G)") and "aws offers EFA for" in note


@pytest.mark.parametrize("gpu", ["nebius:2x8xh100", "aws:2x8xh100", "modal:2x8xb300", "nebius:1x8xh100",
                                 "nebius:1x1xl40s", "gcp:1x8xh100"])
def test_islands_on_ib_efa_rdma_or_one_node_get_no_note(gpu):
    assert _notes(gpu) == []


def _inventory(fabric: str):
    from yeto.launch.node_inventory import parse_inventory

    path = topology.__file__.rsplit("/yeto/", 1)[0] + "/examples/k31/inventory.b300-6x8-baremetal.yaml"
    import yaml

    doc = yaml.safe_load(open(path))
    for node in doc["nodes"]:
        node["fabric"] = fabric
    return parse_inventory(doc)


def test_an_ssh_island_follows_its_inventory_fabric():
    assert _notes("ssh:6x8xb300", inventory=_inventory("ib")) == []  # the customer bare-metal kit
    assert _notes("ssh:6x8xb300", inventory=_inventory("roce")) == []
    [note] = _notes("ssh:6x8xb300", inventory=_inventory("tcp"))
    assert "inventory trainer node(s)" in note and "declare fabric: tcp" in note
    assert _notes("ssh:6x8xb300", inventory=None) == []  # no inventory: the launch refuses it itself


def test_the_dry_run_prints_the_note(monkeypatch, capsys, _fake_sky, _no_modal_listing):  # noqa: F811
    from test_verl_full_parameter_launch import _prepared

    from yeto.launcher import dry_run_plan

    args = _prepared(_launcher_args("ports", ("--reward-function", "gsm8k_reward:score",
                                                "--rollout-batch-size", "8"), gpu="nebius:2x8xh100"))
    args.rl_island_network_tier = "none"
    plan = dry_run_plan(args)
    out = capsys.readouterr().out
    assert "[preflight] note: island 0 (nebius:2x8xH100) runs NCCL between its 2 nodes over TCP sockets" in out
    assert "cross-node bandwidth" not in json.dumps(plan)  # a note only: the plan is unchanged
