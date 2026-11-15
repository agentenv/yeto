"""``YETO_ISLAND_COMM_INTERFACE``: comm variables for a verl island that spans Ray nodes.

Measured 2026-11-15 on two nodes on one subnet plus a third on another: without the switch
the 3-node island dies in gloo ``connect: Network is unreachable``; with
``eth0.100,eth1.200`` it trains. Unset, the runtime env must not change.
"""

from __future__ import annotations

import json
import os
import sys
import types

import pytest

from yeto.rl.adapters.verl import comm_env

COMM_KEYS = ("HCCL_CONNECT_TIMEOUT", "HCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "NCCL_SOCKET_IFNAME",
             "NCCL_IB_DISABLE", "NCCL_DEBUG", comm_env.INTERFACE_ENV, comm_env.FABRIC_ENV)


@pytest.fixture
def verl_main(monkeypatch):
    pkg = types.ModuleType("verl.trainer.constants_ppo")
    pkg.get_ppo_ray_runtime_env = lambda _x: {"env_vars": {"TOKENIZERS_PARALLELISM": "true"}}
    for name in ("verl", "verl.trainer"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "verl.trainer.constants_ppo", pkg)
    ray_stub = types.ModuleType("ray")
    ray_stub.remote = lambda *a, **k: (lambda cls: cls)
    monkeypatch.setitem(sys.modules, "ray", sys.modules.get("ray") or ray_stub)
    monkeypatch.setitem(sys.modules, "yeto.rl.adapters.verl.verl_main", None)
    monkeypatch.delitem(sys.modules, "yeto.rl.adapters.verl.verl_main")
    for key in COMM_KEYS:
        monkeypatch.delenv(key, raising=False)
    from yeto.rl.adapters.verl import verl_main as module

    return module


def test_switch_unset_plans_nothing_and_leaves_the_runtime_env_alone(verl_main, monkeypatch):
    assert comm_env.island_plan(3, device_family="ascend") is None
    monkeypatch.setenv("HCCL_SOCKET_IFNAME", "eth9")  # a user's own value travels as before
    base = verl_main._runtime_env(fully_async=True)
    assert verl_main._runtime_env(fully_async=True, comm_env=None) == base
    assert verl_main._runtime_env(fully_async=True, comm_env=comm_env.island_plan(3)) == base
    assert "worker_process_setup_hook" not in base and base["env_vars"]["HCCL_SOCKET_IFNAME"] == "eth9"


def test_one_card_node_plans_nothing_even_with_the_switch(monkeypatch):
    monkeypatch.setenv(comm_env.INTERFACE_ENV, "eth0.100")
    assert comm_env.island_plan(1, device_family="ascend") is None
    assert comm_env.island_plan(0, device_family="ascend") is None


def test_auto_is_refused_by_name(monkeypatch):
    monkeypatch.setenv(comm_env.INTERFACE_ENV, "auto")
    with pytest.raises(SystemExit, match="auto is not supported"):
        comm_env.island_plan(2, device_family="ascend")


def test_one_interface_travels_in_env_vars_and_the_users_value_wins(verl_main, monkeypatch):
    monkeypatch.setenv(comm_env.INTERFACE_ENV, "eth0.100")
    plan = comm_env.island_plan(3, device_family="ascend")
    assert plan["family"] == "npu" and plan["env"] == {
        "HCCL_CONNECT_TIMEOUT": "1800", "HCCL_SOCKET_IFNAME": "eth0.100", "GLOO_SOCKET_IFNAME": "eth0.100"}
    plan = json.loads(json.dumps(plan))  # the plan file round trip
    env = verl_main._runtime_env(comm_env=plan)
    assert env["env_vars"]["GLOO_SOCKET_IFNAME"] == "eth0.100"
    assert env["env_vars"]["HCCL_SOCKET_IFNAME"] == "eth0.100"
    assert env["env_vars"]["HCCL_CONNECT_TIMEOUT"] == "1800"
    assert "worker_process_setup_hook" not in env

    monkeypatch.setenv("HCCL_SOCKET_IFNAME", "eth9")  # forwarded by prefix: kept
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "eth8")  # not a forwarded prefix: still the user's
    monkeypatch.setenv("HCCL_CONNECT_TIMEOUT", "600")
    env = verl_main._runtime_env(comm_env=plan)["env_vars"]
    assert (env["HCCL_SOCKET_IFNAME"], env["GLOO_SOCKET_IFNAME"], env["HCCL_CONNECT_TIMEOUT"]) == (
        "eth9", "eth8", "600")


def test_cuda_family_gets_nccl_names(monkeypatch):
    monkeypatch.setenv(comm_env.INTERFACE_ENV, "ens5")
    monkeypatch.setenv(comm_env.FABRIC_ENV, "roce")
    plan = comm_env.island_plan(2, device_family="nvidia")
    assert plan["env"] == {"NCCL_IB_DISABLE": "0", "NCCL_DEBUG": "WARN",
                           "NCCL_SOCKET_IFNAME": "ens5", "GLOO_SOCKET_IFNAME": "ens5"}


def test_a_list_is_resolved_on_each_node_and_wraps_the_ascend_hook(verl_main, monkeypatch, tmp_path):
    monkeypatch.setenv(comm_env.INTERFACE_ENV, "eth0.100, eth1.200")
    plan = comm_env.island_plan(3, device_family="ascend")
    assert plan["env"] == {"HCCL_CONNECT_TIMEOUT": "1800"}  # no global interface
    from yeto.rl import ascend_plugin_stack

    env = verl_main._runtime_env(ascend_stack="/stack", comm_env=plan)
    monkeypatch.delenv(ascend_plugin_stack.ENV, raising=False)
    assert env["worker_process_setup_hook"] == comm_env.SETUP_HOOK
    assert env["env_vars"][comm_env.INNER_HOOK_ENV] == ascend_plugin_stack.SETUP_HOOK
    assert "GLOO_SOCKET_IFNAME" not in env["env_vars"]

    # a worker on node-c: only eth1.200 exists; the wrapped hook still runs
    net = tmp_path / "net"
    (net / "eth1.200").mkdir(parents=True)
    called = []
    inner = types.ModuleType("fake_inner_hook")
    inner.setup = lambda: called.append(True)
    monkeypatch.setitem(sys.modules, "fake_inner_hook", inner)
    worker = dict(env["env_vars"], **{comm_env.INNER_HOOK_ENV: "fake_inner_hook.setup"})
    monkeypatch.setattr(os, "environ", dict(worker, GLOO_SOCKET_IFNAME="from-ray-start"))
    comm_env.setup_worker(str(net))
    assert os.environ["HCCL_SOCKET_IFNAME"] == "eth1.200"
    assert os.environ["GLOO_SOCKET_IFNAME"] == "from-ray-start"  # the node's own value wins
    assert called == [True]

    monkeypatch.setattr(os, "environ", dict(worker))
    (net / "eth1.200").rmdir()
    with pytest.raises(RuntimeError, match="none of eth0.100, eth1.200"):
        comm_env.setup_worker(str(net))


def test_a_list_without_another_hook_installs_only_its_own(verl_main, monkeypatch):
    monkeypatch.setenv(comm_env.INTERFACE_ENV, "a,b")
    env = verl_main._runtime_env(comm_env=comm_env.island_plan(2, device_family="ascend"))
    assert env["worker_process_setup_hook"] == comm_env.SETUP_HOOK
    assert comm_env.INNER_HOOK_ENV not in env["env_vars"]
