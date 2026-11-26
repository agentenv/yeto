"""Fabric groups (own-cluster-fleet-stability 2.1-2.4): the inventory field, ``yeto cluster
probe`` over a fake SSH cluster (the preflight's ib_write_bw and NCCL steps; groups are
cliques), the idle-node gate, stable group names, ``--write`` as the only (atomic) writer,
``pick_placement`` / ``check_placement`` for multi-node islands (an inventory without groups
only gets a hint, which a dry run and the preflight keep out of their output), and
``yeto cluster check``."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from test_cluster_preflight import Behavior, FakeCluster, b300, ibstat_text
from yeto import cluster_cli
from yeto.launch import fabric_groups as fg
from yeto.launch.node_inventory import FABRIC_UNAVAILABLE, InventoryError, load_inventory, parse_inventory
from yeto.launch.preflight import cluster as cp
from yeto.launch.ssh_trainer_island import SshIslandError, inventory_trainer_nodes

REPO = Path(__file__).resolve().parents[1]
CUSTOMER_INVENTORY = REPO / "examples" / "k31" / "inventory.b300-6x8-baremetal.yaml"
DOC = REPO / "docs" / "OWN_CLUSTERS.md"
IMAGE = "ghcr.io/owner/yeto-miles-k31@sha256:" + "ab" * 32
NULL = object()  # _grouped: the key with a null value


@pytest.fixture(autouse=True)
def _fresh_hints():
    fg._hinted.clear()
    yield
    fg._hinted.clear()


def _entry(i: int, **kw) -> dict:
    base = {"name": f"b300-{i}", "host": f"ops@b300-{i}", "device_family": "cuda", "cpu_arch": "x86_64",
            "card": "B300", "cards_per_node": 8, "hbm_gib": 288, "host_memory_gib": 2048, "fabric": "ib",
            "interface": "bond0", "address": f"192.0.2.{10 + i}"}
    return base | kw


def _grouped(*groups) -> list:
    def one(i, g):
        if g is NULL:
            return _entry(i, fabric_group=None)
        return _entry(i, **({"fabric_group": g} if g else {}))

    return parse_inventory({"nodes": [one(i, g) for i, g in enumerate(groups)]})


class IdleCluster(FakeCluster):
    """The preflight's fake B300 cluster, answering the probe's busy check too."""

    def __init__(self, nodes, *, gpu_apps=None, containers=None, smi_rc=None, ps_rc=None, silent=(), **per_node):
        super().__init__(nodes, **per_node)
        self.gpu_apps, self.containers, self.smi_rc = gpu_apps or {}, containers or {}, smi_rc or {}
        self.ps_rc, self.silent = ps_rc or {}, set(silent)  # silent: the idle check gets no answer

    def run(self, node, script, *, timeout=300.0):
        if f"{cp.SECTION} compute_apps" not in script:
            return super().run(node, script, timeout=timeout)
        self.scripts.append((node.name, script))
        if self.b[node.name].ssh_rc or node.name in self.silent:
            return self.done(255, "ssh: connect to host port 22: Connection timed out")
        out = [f"{cp.SECTION} compute_apps", self.gpu_apps.get(node.name, ""), f"{cp.RC} {self.smi_rc.get(node.name, 0)}"]
        if f"{cp.SECTION} yeto_containers" in script:
            out += [f"{cp.SECTION} yeto_containers", self.containers.get(node.name, ""),
                    f"{cp.RC} {self.ps_rc.get(node.name, 0)}"]
        return self.done(0, "\n".join(out) + "\n")


def _check(scope, name, status, detail, **data):
    return cp.Check(scope, name, status, detail, data=data)


def _pair_checks(names, *, socket=(), ib_status=None, link=400.0):
    """Synthetic preflight checks: every pair passes ib_write_bw and NCCL over NET/IB, except
    the ``socket`` pairs (NET/Socket) and ``ib_status`` overrides (pair -> status, detail)."""
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            status, detail = (ib_status or {}).get((a, b), ("PASS", "8 rails, 392.5 GB/s total"))
            out.append(_check(f"{a}<->{b}", "ib_write_bw", status, detail, link_gbps=[link] * 8))
            if (a, b) in socket:
                out.append(_check(f"{a}+{b}", "nccl_transport", "FAIL", "NCCL runs over NET/Socket (TCP), not NET/IB"))
            else:
                out.append(_check(f"{a}+{b}", "nccl_transport", "PASS", "NET/IB with GPUDirect RDMA"))
    return out


# ------------------------------------------------------------------ 2.1 schema
def test_old_inventories_load_unchanged_and_carry_no_group():
    nodes = load_inventory(CUSTOMER_INVENTORY)
    assert len(nodes) == 6 and all(n.fabric_group is None for n in nodes)
    assert all("fabric_group" not in n.record() for n in nodes)  # the manifest row keeps its keys
    assert not fg.grouped(nodes)


def test_fabric_group_parses_and_a_wrong_type_or_name_is_refused():
    (node,) = parse_inventory({"nodes": [_entry(0, fabric_group="fg-1a2b3c4d")]})
    assert node.fabric_group == "fg-1a2b3c4d"
    for bad in (3, ["a"], {"a": 1}, True, "", "bad name", "-lead", "a" * 65, "x;y"):
        with pytest.raises(InventoryError, match="fabric_group"):
            parse_inventory({"nodes": [_entry(0, fabric_group=bad)]})


def test_a_null_key_makes_the_inventory_grouped():
    nodes = _grouped(NULL, None, None)
    assert nodes[0].fabric_group == "" and fg.grouped(nodes)
    ok, message, hint = fg.check_placement(nodes, 3)
    assert not ok and not hint and "3 node(s) without fabric_group" in message


def test_unavailable_never_joins_a_multi_node_island_but_a_single_node_one():
    nodes = _grouped(FABRIC_UNAVAILABLE, FABRIC_UNAVAILABLE, "A")
    with pytest.raises(fg.PlacementError, match="2 node\\(s\\) fabric unavailable"):
        fg.pick_placement(nodes, 2)
    assert fg.pick_placement(nodes, 1).nodes == (nodes[0],)
    assert fg.check_placement(nodes[:1], 1) == (True, "", False)


def test_a_node_profile_doc_never_carries_fabric_group():
    from yeto.launch.node_inventory import NodeProfile
    from yeto.node.profile import profile_doc

    for node in _grouped(None, "fg-1a2b3c4d"):
        assert "fabric_group" not in profile_doc(NodeProfile.from_node(node))


# ------------------------------------------------------------- 2.3 placement
def test_pick_placement_takes_the_group_with_the_most_free_nodes_by_name():
    nodes = _grouped("B", "A", "A", "B", "B", "C", "C", "C")
    p = fg.pick_placement(list(reversed(nodes)), 2)
    assert p.group == "B" and [n.name for n in p.nodes] == ["b300-0", "b300-3"]  # 3 free in B and C: name
    p = fg.pick_placement(nodes, 3, excluded={"b300-5": "leased by run x"})
    assert p.group == "B"
    p = fg.pick_placement(nodes, 2, excluded=["b300-0", "b300-3", "b300-4"])
    assert p.group == "C" and [n.name for n in p.nodes] == ["b300-5", "b300-6"]


def test_pick_placement_without_groups_keeps_the_candidates_order():
    nodes = list(reversed(_grouped(None, None, None)))
    p = fg.pick_placement(nodes, 2, excluded={"b300-2"})
    assert p.group is None and [n.name for n in p.nodes] == ["b300-1", "b300-0"]
    with pytest.raises(fg.PlacementError, match="needs 3 node\\(s\\); 2 of 3"):
        fg.pick_placement(nodes, 3, excluded={"b300-2": "quarantined"})


def test_a_lost_node_is_replaced_by_the_next_of_its_group():
    nodes = _grouped("A", "A", "A", "B")
    first = fg.pick_placement(nodes, 2)
    assert [n.name for n in first.nodes] == ["b300-0", "b300-1"]
    again = fg.pick_placement(nodes, 2, excluded={"b300-1": "lease held by another run"})
    assert again.group == "A" and [n.name for n in again.nodes] == ["b300-0", "b300-2"]


def test_placement_one_group_with_enough_healthy_nodes_passes():
    nodes = _grouped("A", "A", "A", "B")
    ok, message, hint = fg.check_placement(nodes[:3], 3)
    assert ok and not hint and "fabric group A" in message


def test_placement_refuses_too_few_nodes_of_one_group_and_counts_every_group():
    nodes = _grouped("A", "A", "A", "A", "B", "B")
    ok, message, hint = fg.check_placement(nodes, 6)
    assert not ok and not hint
    assert "needs 6 healthy nodes of one fabric group" in message
    assert "group A has only 4" in message and "group B has only 2" in message


def test_placement_unhealthy_nodes_do_not_count():
    nodes = _grouped("A", "A", "A")
    assert fg.check_placement(nodes, 3)[0]
    ok, message, _ = fg.check_placement(nodes, 3, unhealthy={"b300-1": "GPU 5 XID 79"})
    assert not ok and "group A has only 2" in message and "b300-1: GPU 5 XID 79" in message


def test_placement_partly_grouped_inventory_refuses_and_names_the_ungrouped_nodes():
    nodes = _grouped("A", "A", None, None)
    ok, message, hint = fg.check_placement(nodes, 4)
    assert not ok and not hint
    assert "group A has only 2" in message and "2 node(s) without fabric_group (b300-2, b300-3)" in message
    ok, _, hint = fg.check_placement(nodes[2:], 2, inventory=nodes)  # trainer candidates without groups
    assert not ok and not hint


def test_placement_without_any_group_is_a_hint_only_and_single_node_is_free():
    nodes = _grouped(None, None, None)
    assert fg.check_placement(nodes, 3) == (True, fg.HINT, True)
    assert fg.check_placement(_grouped("A", "B"), 1) == (True, "", False)


def _island(n: int):
    return SimpleNamespace(num_nodes=n, gpus_per_node=8, gpu="B300")


def test_trainer_island_refuses_split_groups_before_any_ssh():
    args = argparse.Namespace(rl_node_inventory_nodes=_grouped("A", "A", "B"), dry_run=True)
    with pytest.raises(SshIslandError, match="not one fabric group: .*group A has only 2, group B has only 1"):
        inventory_trainer_nodes(args, _island(3))
    args = argparse.Namespace(rl_node_inventory_nodes=_grouped("A", "A", "A"), dry_run=True)
    assert [n.name for n in inventory_trainer_nodes(args, _island(3))] == ["b300-0", "b300-1", "b300-2"]


def test_ungrouped_inventory_hint_goes_to_the_log_once_and_never_into_a_dry_run(capsys, caplog):
    nodes = _grouped(None, None)
    caplog.set_level(logging.INFO, logger=fg.__name__)
    for _ in range(2):
        inventory_trainer_nodes(argparse.Namespace(rl_node_inventory_nodes=nodes, dry_run=True), _island(2))
    inventory_trainer_nodes(argparse.Namespace(rl_node_inventory_nodes=nodes), _island(2), hint_to_log=True)
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""
    assert [r.getMessage() for r in caplog.records] == [fg.HINT]
    for _ in range(2):  # a real launch: one line in the launcher log, though the gate logged it already
        inventory_trainer_nodes(argparse.Namespace(rl_node_inventory_nodes=nodes, dry_run=False), _island(2))
    err = capsys.readouterr().err
    assert err.count("yeto cluster probe") == 1 and err.startswith("[launcher] note: ")


def test_the_customer_dry_run_reaches_the_check_and_prints_nothing_new(tmp_path, monkeypatch, capsys, caplog):
    import test_k31_customer_plan_golden as golden

    seen = []

    class Tap:
        def readouterr(self):
            got = capsys.readouterr()
            seen.append(got)
            return got

    caplog.set_level(logging.INFO, logger=fg.__name__)
    plan = golden._plan(golden.CASES["examples/run.baremetal-b300-262k.json"], tmp_path, monkeypatch, Tap())
    want = json.loads(golden.GOLDEN.read_text())["examples/run.baremetal-b300-262k.json"]
    assert plan == want
    assert [r.getMessage() for r in caplog.records] == [fg.HINT]  # reached: the customer island ran the check
    assert not any("fabric" in got.err or "cluster probe" in got.err for got in seen)


def _runbook_preflight(tmp_path, extra=()):
    """`yeto preflight` of the runbook (docs/K31_BARE_METAL_B300.md section 9) on a fake cluster."""
    from yeto.cli import build_parser

    text = (REPO / "docs" / "K31_BARE_METAL_B300.md").read_text()
    section = text[text.index("## 9. Bring-up order"):text.index("## 10. Launch")]
    commands = re.findall(r"^\s*(yeto preflight [^\n]+)$", section, re.M)
    assert "yeto preflight --gpu ssh:6x8xb300 --rl-node-inventory inventory.yaml --cluster-prefix <run> --no-bench" \
        in commands
    shutil.copy(CUSTOMER_INVENTORY, tmp_path / "inventory.yaml")
    out = []
    for command in commands:
        argv = command.replace("inventory.yaml", str(tmp_path / "inventory.yaml")).replace("<run>", "k31").split()
        args = build_parser().parse_args([*argv[1:], "--run-dir", str(tmp_path / "pf"), *extra])
        nodes = load_inventory(tmp_path / "inventory.yaml")
        out.append((command, args, FakeCluster(nodes)))
    return out


def test_the_runbook_preflight_commands_print_what_main_prints(tmp_path, monkeypatch, capsys, caplog):
    caplog.set_level(logging.INFO, logger=fg.__name__)
    for command, args, ops in _runbook_preflight(tmp_path):
        cp.main(args, ops=ops)
        now = capsys.readouterr().err
        with monkeypatch.context() as m:  # main: no placement check at all
            m.setattr(fg, "check_placement", lambda *a, **k: (True, "", False))
            cp.main(args, ops=FakeCluster(list(ops.nodes.values())))
        before = capsys.readouterr().err
        assert now == before, command
        assert "fabric" not in now and "cluster probe" not in now
    assert [r.getMessage() for r in caplog.records] == [fg.HINT]


def _preflight_args(gpu, tmp_path, *extra):
    from yeto.cli import build_parser

    return build_parser().parse_args(["preflight", "--gpu", gpu, "--run-dir", str(tmp_path / "pf"), "--json", *extra])


def test_split_groups_are_a_preflight_fail_and_every_node_is_still_checked(tmp_path, capsys):
    path = tmp_path / "nodes.yaml"
    entries = [_entry(0, fabric_group="A"), _entry(1, fabric_group="B")]
    path.write_text(yaml.safe_dump({"nodes": [{**e, "docker_args": list(b300(0).docker_args), "image": IMAGE,
                                               "env": dict(b300(0).env)} for e in entries]}))
    for gpu, extra in ((f"ssh:{path}", ()), ("ssh:2x8xb300", ("--rl-node-inventory", str(path)))):
        nodes = load_inventory(path)
        ops = FakeCluster(nodes)
        assert cp.main(_preflight_args(gpu, tmp_path, *extra), ops=ops) == 1
        report = json.loads(capsys.readouterr().out)
        (row,) = [c for c in report["checks"] if c["name"] == "fabric_group"]
        assert row["status"] == "FAIL" and row["scope"] == "inventory" and "group A has only 1" in row["detail"]
        assert {c["scope"] for c in report["checks"] if c["name"] == "ssh"} == {"b300-0", "b300-1"}  # diagnosed
    path.write_text(yaml.safe_dump({"nodes": [_entry(0, fabric_group="A"), _entry(1, fabric_group="A")]}))
    assert cp.fabric_check(load_inventory(path))[0].status == "PASS"
    path.write_text(yaml.safe_dump({"nodes": [_entry(0), _entry(1)]}))
    assert len(cp.resolve_nodes(f"ssh:{path}", None)) == 2 and cp.fabric_check(load_inventory(path)) == []
    assert capsys.readouterr().err == ""


# ------------------------------------------------------------------- 2.2 probe
def _gids(node, prefix="fe80:0000:0000:0000"):
    return "\n".join(f"mlx5_{j} 1 0 {prefix}:0a00:00ff:fe00:00{j:02x} ib{j} IB/RoCE v1" for j in range(8))


def _cluster(n: int, *, socket=(), prefix=None, **per):
    nodes = [b300(i) for i in range(n)]
    behaviors = {}
    for node in nodes:
        b = per.get(node.name) or Behavior()
        if node.name in socket:
            b.nccl = "socket"
        if prefix:
            b.host = {**b.host, "gids": (0, _gids(node, prefix))}
        behaviors[node.name] = b
    return nodes, IdleCluster(nodes, **behaviors)


def _probe(nodes, ops, nccl_pairs="all", **kw):
    pf = cp.ClusterPreflight(nodes, ops=ops, log=lambda m: None, clock=lambda: FakeCluster.NOW,
                             steps=cp.Steps(ports=False, nccl_all=False, nccl_pairs=nccl_pairs))
    report = pf.run()
    return fg.group_nodes([x.name for x in nodes], report.checks, states=pf.states, nccl_pairs=nccl_pairs, **kw)


def test_three_connected_nodes_form_a_group_and_a_socket_node_is_fabric_unavailable():
    nodes, ops = _cluster(4, socket={"b300-3"})
    result = _probe(nodes, ops)
    ((name, members),) = result.groups.items()
    assert members == ["b300-0", "b300-1", "b300-2"] and re.fullmatch(r"fg-[0-9a-f]{8}", name)
    assert list(result.unavailable) == ["b300-3"]
    why = " ".join(result.unavailable["b300-3"])
    assert "NET/Socket" in why and "to b300-0" in why
    text = "\n".join(result.lines())
    assert f"fabric group {name}: b300-0, b300-1, b300-2" in text and "fabric unavailable: b300-3" in text
    assert result.assignment() == {"b300-0": name, "b300-1": name, "b300-2": name, "b300-3": FABRIC_UNAVAILABLE}
    assert sum("torch.distributed.run" in s for _, s in ops.scripts) == 2 * 6  # every pair ran its own NCCL


def _tcp_inventory(tmp_path, fabrics):
    path = tmp_path / "nodes.yaml"
    path.write_text(yaml.safe_dump({"nodes": [_entry(i, fabric=f) for i, f in enumerate(fabrics)]}))
    return path


def test_a_tcp_node_is_no_group_and_no_probe_failure_with_a_next_step(tmp_path, capsys):
    """Rehearsal: tcp nodes were first said to "get no group", then reported "fabric
    unavailable" (with the ib_write_bw noise of their pairs) and the probe exited 1."""
    path = _tcp_inventory(tmp_path, ["ib", "ib", "tcp"])
    assert cluster_cli.main(_args(path, "--write"), ops=_ops_for(path), log=lambda m: None) == 0
    cap = capsys.readouterr()
    assert "nodes ['b300-2'] declare no RDMA fabric (ib / roce): they get no group" in cap.err
    (tcp,) = [x for x in cap.out.splitlines() if "b300-2" in x]
    assert tcp.startswith(f"{fg.NO_RDMA}: b300-2: fabric tcp in the inventory: no group, single-node islands only")
    assert f"next: {fg.NO_RDMA_NEXT}" in tcp and fg.UNAVAILABLE not in cap.out
    groups = {n.name: n.fabric_group for n in load_inventory(path)}
    assert groups["b300-0"] == groups["b300-1"] != FABRIC_UNAVAILABLE and groups["b300-2"] == FABRIC_UNAVAILABLE


def test_a_probe_of_tcp_nodes_only_says_so_for_every_node_and_exits_0(tmp_path, capsys):
    path = _tcp_inventory(tmp_path, ["tcp", "tcp"])
    assert cluster_cli.main(_args(path, "--json"), ops=_ops_for(path), log=lambda m: None) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["groups"] == {} and record["unavailable"] == {} and record["bad_edges"] == []
    assert record["no_rdma"] == {"b300-0": "tcp", "b300-1": "tcp"}


def test_a_tcp_node_that_fails_its_node_checks_fails_the_probe_with_the_reason(tmp_path, capsys):
    """REV-904b M2: a tcp node whose image pull is refused is a probe failure (main: rc 1 with
    the image_pull reason), not "nothing to do"."""
    path = _tcp_inventory(tmp_path, ["ib", "ib", "tcp"])
    ops = _ops_for(path, behaviors={"b300-2": Behavior(pull=(1, "pull access denied for x"))})
    assert cluster_cli.main(_args(path), ops=ops, log=lambda m: None) == 1
    (tcp,) = [x for x in capsys.readouterr().out.splitlines() if "b300-2" in x]
    assert tcp.startswith(f"{fg.NO_RDMA}: b300-2: fabric tcp") and "node checks FAILED: image_pull:" in tcp
    assert "pull access denied" in tcp and f"next: {fg.NO_RDMA_FAILED_NEXT}" in tcp and "nothing to do" not in tcp
    path = _tcp_inventory(tmp_path, ["tcp", "tcp"])
    ops = _ops_for(path, behaviors={"b300-1": Behavior(pull=(1, "pull access denied for x"))})
    assert cluster_cli.main(_args(path, "--json"), ops=ops, log=lambda m: None) == 1
    record = json.loads(capsys.readouterr().out)
    assert list(record["no_rdma_failed"]) == ["b300-1"] and record["no_rdma"] == {"b300-0": "tcp", "b300-1": "tcp"}
    path = _tcp_inventory(tmp_path, ["ib", "ib", "tcp"])
    ops = _ops_for(path, behaviors={"b300-2": Behavior(ssh_rc=255)})
    assert cluster_cli.main(_args(path, "--force"), ops=ops, log=lambda m: None) == 1  # past the idle check
    (tcp,) = [x for x in capsys.readouterr().out.splitlines() if "b300-2" in x]
    assert "node checks FAILED: ssh:" in tcp  # unreachable: a failure (a clock skew alone is not)


def test_declared_roce_nodes_are_grouped_and_never_no_rdma():
    names = ["r-0", "r-1"]
    states = [SimpleNamespace(node=SimpleNamespace(name=n, fabric="roce", env={}, cards_per_node=8), ports=None)
              for n in names]
    result = fg.group_nodes(names, _pair_checks(names), states=states)
    assert result.no_rdma == {} and list(result.groups.values()) == [names] and result.clean
    tcp = [SimpleNamespace(node=SimpleNamespace(name=n, fabric="tcp", env={}, cards_per_node=8), ports=None)
           for n in names]
    assert fg.group_nodes(names, _pair_checks(names), states=tcp).no_rdma == {"r-0": "tcp", "r-1": "tcp"}


def test_a_declared_ib_node_that_runs_over_sockets_stays_fabric_unavailable():
    nodes, ops = _cluster(2, socket={"b300-1"})
    result = _probe(nodes, ops)
    assert result.no_rdma == {} and list(result.unavailable) == ["b300-0", "b300-1"] and not result.clean


def test_a_triangle_with_a_missing_edge_is_no_group_of_three():
    names = ["b300-0", "b300-1", "b300-2"]
    result = fg.group_nodes(names, _pair_checks(names, socket={("b300-0", "b300-2")}))
    assert result.groups and list(result.groups.values()) == [["b300-0", "b300-1"]]
    assert list(result.unavailable) == ["b300-2"]
    assert "to b300-0: NCCL: NCCL runs over NET/Socket" in result.unavailable["b300-2"][0]
    assert [e[:2] for e in result.bad_edges] == [("b300-0", "b300-2")] and not result.clean
    assert any(line.startswith("missing edge b300-0<->b300-2: NCCL: NCCL runs over NET/Socket")
               for line in result.lines())


def test_cliques_cut_a_component_by_its_most_missing_edges():
    square = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")]
    assert fg.cliques(list("abcd"), square) == [["a", "b"], ["c", "d"]]
    full = [(x, y) for i, x in enumerate("abcd") for y in "abcd"[i + 1:]]
    assert fg.cliques(list("abcd"), full) == [list("abcd")]
    star = [("a", "b"), ("a", "c"), ("a", "d")]
    assert fg.cliques(list("abcd"), star) == [["a", "b"], ["c"], ["d"]]


def test_a_failed_hca_check_or_a_missing_rail_gives_a_node_no_edge():
    names = ["b300-0", "b300-1", "b300-2"]
    checks = _pair_checks(names) + [_check("b300-1", "ib_ports", "FAIL", "mlx5_3 port Down")]
    result = fg.group_nodes(names, checks)
    assert list(result.groups.values()) == [["b300-0", "b300-2"]]
    assert result.unavailable["b300-1"][0] == "ib_ports: mlx5_3 port Down"
    ports = [cp.IbPort(f"mlx5_{j}", 1, "Active" if j != 3 else "Down", rate_gbps=400) for j in range(8)]
    state = SimpleNamespace(node=b300(2), ports=ports, compute=ports, gids=[], host={})
    assert fg.node_blocks(names, [], [state]) == {"b300-2": "7 of 8 declared HCA rails active"}
    state = SimpleNamespace(node=b300(2), ports=ports[:7], compute=ports[:7], gids=[], host={})
    assert fg.node_blocks(names, [], [state]) == {"b300-2": "6 of 8 declared HCA rails active"}


def test_a_down_rail_on_the_fake_cluster_takes_the_node_out():
    down = Behavior(host={"ibstat": (0, ibstat_text(down=(3,)))})
    nodes, ops = _cluster(3, **{"b300-1": down})
    result = _probe(nodes, ops)
    assert list(result.groups.values()) == [["b300-0", "b300-2"]] and "b300-1" in result.unavailable


def test_an_unknown_link_rate_is_a_warning_and_no_edge():
    names = ["b300-0", "b300-1"]
    checks = _pair_checks(names, ib_status={("b300-0", "b300-1"): ("PASS", "8 rails (link rate unknown)")}, link=0.0)
    result = fg.group_nodes(names, checks)
    assert result.groups == {} and "link rate is unknown" in " ".join(result.unavailable["b300-0"])
    assert result.warnings and result.warnings[0].startswith("[WARN] b300-0<->b300-1 ib_write_bw: link rate unknown")


def test_a_slow_rail_breaks_the_edge_and_names_the_pair():
    slow = Behavior(rail_gbps={"mlx5_2": 150.0})  # b300-1 as the client of its pairs: 37 % of 400
    nodes, ops = _cluster(2, **{"b300-1": slow})
    result = _probe(nodes, ops)
    assert result.groups == {} and set(result.unavailable) == {"b300-0", "b300-1"}
    assert "ib_write_bw FAIL" in " ".join(result.unavailable["b300-0"])


def test_an_unreachable_node_gets_its_node_reason():
    nodes, ops = _cluster(3, **{"b300-2": Behavior(ssh_rc=255)})
    result = _probe(nodes, ops)
    assert list(result.groups.values()) == [["b300-0", "b300-1"]]
    assert result.unavailable["b300-2"][0].startswith("ssh: ")


def test_disjoint_nccl_pairs_count_pairs_whose_nodes_ran_net_ib_elsewhere():
    nodes, ops = _cluster(4)
    result = _probe(nodes, ops, nccl_pairs="disjoint")
    assert list(result.groups.values()) == [["b300-0", "b300-1", "b300-2", "b300-3"]]
    assert sum("torch.distributed.run" in s for _, s in ops.scripts) == 2 * 2
    nodes, ops = _cluster(4, socket={"b300-3"})
    result = _probe(nodes, ops, nccl_pairs="disjoint")
    assert list(result.groups.values()) == [["b300-0", "b300-1"]]  # b300-2 ran NCCL only with b300-3


def test_a_pair_without_gpudirect_still_counts_and_a_socket_warn_does_not():
    nodes, ops = _cluster(2, **{"b300-1": Behavior(nccl="ib-nogdr")})
    assert list(_probe(nodes, ops).groups.values()) == [["b300-0", "b300-1"]]
    socket_warn = _check("b300-0+b300-1", "nccl_transport", "WARN", "NCCL runs over NET/Socket (TCP), not NET/IB")
    ib = _check("b300-0<->b300-1", "ib_write_bw", "PASS", "8 rails", link_gbps=[400.0])
    edges, problems, _ = fg.evaluate_pairs(["b300-0", "b300-1"], [ib, socket_warn])
    assert edges == [] and "NET/Socket" in problems["b300-0<->b300-1"]


def test_group_names_are_stable_when_nodes_are_added_or_removed():
    nodes, ops = _cluster(4, socket={"b300-3"})
    first = _probe(nodes, ops)
    (name,) = first.groups
    previous = first.assignment()
    nodes, ops = _cluster(5, socket={"b300-3"})  # a fifth node joins the fabric
    grown = _probe(nodes, ops, previous=previous)
    assert grown.groups == {name: ["b300-0", "b300-1", "b300-2", "b300-4"]}
    nodes, ops = _cluster(5, socket={"b300-3"})
    shrunk = _probe([n for n in nodes if n.name != "b300-0"], ops, previous=grown.assignment())
    assert shrunk.groups == {name: ["b300-1", "b300-2", "b300-4"]}


def test_a_fresh_probe_names_a_group_by_its_ib_subnet_whatever_its_nodes():
    names = set()
    for n in (3, 4):
        nodes, ops = _cluster(n, prefix="fe80:0000:0000:00a1")
        (name,) = _probe(nodes, ops).groups
        names.add(name)
    assert len(names) == 1
    nodes, ops = _cluster(3)  # no GIDs read: the hash of the node names
    assert _probe(nodes, ops).groups != {next(iter(names)): ["b300-0", "b300-1", "b300-2"]}


def test_old_names_go_by_the_most_shared_nodes_over_all_groups():
    # the larger group shares 2 nodes with X, the smaller 3: X goes to the smaller one
    comps = [["a", "b", "c", "d"], ["e", "f", "g"]]
    previous = {"a": "X", "b": "X", "c": "Y", "e": "X", "f": "X", "g": "X"}
    assert fg.assign_ids(comps, previous=previous) == ["Y", "X"]


def test_assign_ids_keeps_names_unique_and_respects_reserved_ones():
    comps = [["a", "b", "c"], ["d", "e"]]
    ids = fg.assign_ids(comps, previous={"a": "X", "b": "X", "d": "X", "e": "X"})
    assert ids[0] == "X" and ids[1] != "X"  # a tie: the first group keeps the shared name
    same = fg.assign_ids(comps, keys={n: {"ib:fe80"} for n in "abcde"})
    assert len(set(same)) == 2  # one subnet for two groups: named by their nodes
    held = fg.assign_ids([["a", "b"]])[0]
    assert fg.assign_ids([["a", "b"]], reserved={held})[0] != held
    assert fg.assign_ids([["a", "b"]], previous={"a": FABRIC_UNAVAILABLE, "b": FABRIC_UNAVAILABLE})[0] != \
        FABRIC_UNAVAILABLE


def test_components_merge_transitively():
    assert fg.components(list("abcde"), [("a", "b"), ("b", "c"), ("d", "e")]) == [["a", "b", "c"], ["d", "e"]]


def test_roce_ports_name_the_group_by_their_ip_network():
    state = SimpleNamespace(node=SimpleNamespace(fabric="roce"),
                            compute=[cp.IbPort("mlx5_0", 1, "Active")],
                            gids=cp.parse_gids("mlx5_0 1 3 0000:0000:0000:0000:0000:ffff:c633:6414 eth0 RoCE v2"),
                            host={"addr": cp.Section(0, "2: eth0    inet 198.51.100.20/24 brd x scope global")})
    assert fg.fabric_keys(state) == {"roce:198.51.100.0/24"}


# ------------------------------------------------------------ the CLI, --write
def _inventory_file(tmp_path: Path) -> Path:
    path = tmp_path / "nodes.yaml"
    path.write_text(CUSTOMER_INVENTORY.read_text())  # anchors, flow mappings, comments: the customer form
    return path


def _ops_for(path: Path, socket=(), **kw):
    nodes = load_inventory(path)
    behaviors = {n.name: Behavior(nccl="socket" if n.name in socket else "ib") for n in nodes}
    for name, b in kw.pop("behaviors", {}).items():
        behaviors[name] = b
    return IdleCluster(nodes, **kw, **behaviors)


def _args(path: Path, *extra: str):
    from yeto.cli import build_parser

    return build_parser().parse_args(["cluster", "probe", "--inventory", str(path), "--image", IMAGE, *extra])


def _files(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_probe_without_write_writes_no_file(tmp_path, monkeypatch, capsys):
    path = _inventory_file(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    before = _files(tmp_path)
    rc = cluster_cli.main(_args(path), ops=_ops_for(path, socket={"b300-5"}), log=lambda m: None)
    assert rc == 1  # a node is fabric unavailable
    out = capsys.readouterr()
    assert "fabric group fg-" in out.out and "fabric unavailable: b300-5" in out.out
    assert "nothing written" in out.err and fg.HINT in out.err  # the probe's own output carries the hint
    assert _files(tmp_path) == before


def test_probe_write_sets_only_the_fabric_group_lines(tmp_path, capsys):
    path = _inventory_file(tmp_path)
    original = path.read_text()
    path.chmod(0o640)
    rc = cluster_cli.main(_args(path, "--write"), ops=_ops_for(path, socket={"b300-5"}), log=lambda m: None)
    assert rc == 1
    names = {n.name: n.fabric_group for n in load_inventory(path)}
    group = names["b300-0"]
    assert re.fullmatch(r"fg-[0-9a-f]{8}", group)
    assert names == {**{f"b300-{i}": group for i in range(5)}, "b300-5": FABRIC_UNAVAILABLE}
    new = path.read_text()
    removed = [ln for ln in original.splitlines() if ln not in new.splitlines()]
    added = [ln for ln in new.splitlines() if ln not in original.splitlines()]
    # one line under the anchor (every merged node inherits it), the one unavailable node set apart
    assert len(removed) == 1 and "name: b300-5" in removed[0]
    assert added == [f"    fabric_group: {group}", removed[0].replace("}", ", fabric_group: unavailable}")]
    assert oct(path.stat().st_mode & 0o777) == oct(0o640)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["nodes.yaml"]  # no temp file left
    rc = cluster_cli.main(_args(path, "--write"), ops=_ops_for(path, socket={"b300-5"}), log=lambda m: None)
    assert path.read_text() == new and "unchanged" in capsys.readouterr().err
    rc = cluster_cli.main(_args(path, "--write"), ops=_ops_for(path), log=lambda m: None)
    assert rc == 0 and {n.fabric_group for n in load_inventory(path)} == {group}


def test_probe_write_of_two_nodes_of_the_customer_form_keeps_the_others(tmp_path):
    path = _inventory_file(tmp_path)
    rc = cluster_cli.main(_args(path, "--write", "--nodes", "b300-0,b300-1"), ops=_ops_for(path), log=lambda m: None)
    assert rc == 0
    names = {n.name: n.fabric_group for n in load_inventory(path)}
    group = names["b300-0"]
    assert re.fullmatch(r"fg-[0-9a-f]{8}", group) and names["b300-1"] == group
    assert all(names[f"b300-{i}"] == "" for i in range(2, 6))  # kept out of the group (null on themselves)
    path = _inventory_file(tmp_path)  # the anchor's node not probed: the anchor is left alone
    rc = cluster_cli.main(_args(path, "--write", "--nodes", "b300-1,b300-2"), ops=_ops_for(path), log=lambda m: None)
    assert rc == 0
    names = {n.name: n.fabric_group for n in load_inventory(path)}
    assert names["b300-0"] is None and names["b300-1"] == names["b300-2"] and names["b300-3"] is None
    anchor = path.read_text().split("  - {<<")[0]
    assert "fabric_group" not in anchor


def test_write_is_verified_and_a_json_inventory_is_rewritten(tmp_path):
    path = tmp_path / "nodes.json"
    path.write_text(json.dumps({"nodes": [_entry(0), _entry(1)]}))
    assert fg.write_groups(path, {"b300-0": "A", "b300-1": "A"})
    assert [n.fabric_group for n in load_inventory(path)] == ["A", "A"]
    assert not fg.write_groups(path, {"b300-0": "A", "b300-1": "A"})
    with pytest.raises(InventoryError, match="not in the inventory"):
        fg.updated_text(path.read_text(), {"nope": "A"}, json_file=True)


def test_write_into_a_plain_block_yaml_keeps_comments():
    text = ("# site nodes\nnodes:\n" + "".join(
        f"  - name: n{i}   # rack {i}\n    host: ops@n{i}\n    device_family: cuda\n    cpu_arch: x86_64\n"
        f"    card: B300\n    cards_per_node: 8\n    hbm_gib: 288\n    host_memory_gib: 2048\n"
        + ("    fabric_group: old\n" if i == 1 else "") for i in range(3)))
    new = fg.updated_text(text, {"n0": "A", "n1": "A", "n2": FABRIC_UNAVAILABLE})
    assert "# site nodes" in new and "# rack 0" in new and "fabric_group: old" not in new
    assert [n.fabric_group for n in parse_inventory(yaml.safe_load(new))] == ["A", "A", FABRIC_UNAVAILABLE]


def test_an_edit_that_reads_back_differently_is_refused(monkeypatch):
    text = yaml.safe_dump({"nodes": [_entry(0), _entry(1)]}, sort_keys=False)
    real = fg._yaml_edits
    monkeypatch.setattr(fg, "_yaml_edits", lambda t, todo: real(t, todo).replace("hbm_gib: 288", "hbm_gib: 144", 1))
    with pytest.raises(InventoryError, match="could not write fabric_group in place"):
        fg.updated_text(text, {"b300-0": "A", "b300-1": "A"})


def test_write_is_atomic_synced_and_follows_a_symlink(tmp_path, monkeypatch):
    real = tmp_path / "site" / "nodes.yaml"
    real.parent.mkdir()
    real.write_text(yaml.safe_dump({"nodes": [_entry(0), _entry(1)]}, sort_keys=False))
    real.chmod(0o600)
    link = tmp_path / "nodes.yaml"
    link.symlink_to(real)
    original = real.read_text()
    calls = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(os, "fsync", lambda fd: (calls.append("fsync"), real_fsync(fd))[1])

    def failing_replace(src, dst):
        calls.append("replace")
        raise OSError("disk gone")

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(OSError, match="disk gone"):
        fg.write_groups(link, {"b300-0": "A", "b300-1": "A"})
    assert real.read_text() == original and calls == ["fsync", "replace"]  # synced before the rename
    assert sorted(p.name for p in real.parent.iterdir()) == ["nodes.yaml"]  # the temp file is gone
    monkeypatch.setattr(os, "replace", real_replace)
    assert fg.write_groups(link, {"b300-0": "A", "b300-1": "A"})
    assert link.is_symlink() and [n.fabric_group for n in load_inventory(link)] == ["A", "A"]
    assert oct(real.stat().st_mode & 0o777) == oct(0o600)


def test_probe_dry_run_runs_no_ssh_and_says_how_long(tmp_path, capsys):
    path = _inventory_file(tmp_path)
    ops = _ops_for(path)
    assert cluster_cli.main(_args(path, "--dry-run"), ops=ops) == 0
    assert ops.scripts == []
    err = capsys.readouterr().err
    assert "6 node(s)" in err and "15 pair(s) in 5 round(s)" in err and "nothing run" in err
    assert re.search(r"about \d+ min \(not counting the image pull\), deadline 3600 s", err)


GPU_APP = "4242, python3, 81920 MiB"


def _probed(ops):
    return any("ib_write_bw" in s or "docker pull" in s for _, s in ops.scripts)


def test_probe_refuses_nodes_not_idle_or_not_confirmed_idle(tmp_path, capsys):
    path = _inventory_file(tmp_path)
    ops = _ops_for(path, gpu_apps={"b300-2": GPU_APP}, containers={"b300-4": "yeto-trainer-0 ghcr.io/x Up 2 hours"},
                   smi_rc={"b300-5": 9}, ps_rc={"b300-3": 1}, silent={"b300-1"})
    assert cluster_cli.main(_args(path, "--write"), ops=ops, log=lambda m: None) == cluster_cli.EXIT_BUSY
    err = capsys.readouterr().err
    assert "5 node(s) not idle or not confirmed idle: refused" in err
    assert f"b300-2: GPU process {GPU_APP} (--force-busy-gpu)" in err
    assert "b300-4: container yeto-trainer-0 ghcr.io/x Up 2 hours (--force)" in err
    assert "b300-5: cannot confirm it is idle: GPU processes not listed" in err
    assert "b300-3: cannot confirm it is idle: containers not listed" in err
    assert "b300-1: cannot confirm it is idle: no answer" in err  # an ssh hiccup never waves a node through
    assert not _probed(ops) and "fabric_group" not in path.read_text()


def test_force_lets_unconfirmed_and_container_nodes_through_but_not_a_gpu_process(tmp_path, capsys):
    path = _inventory_file(tmp_path)
    ops = _ops_for(path, containers={"b300-4": "yeto-trainer-0 x Up"}, silent={"b300-1"})
    assert cluster_cli.main(_args(path, "--force", "--nodes", "b300-0,b300-4"), ops=ops, log=lambda m: None) == 0
    assert "[WARN] b300-4: container yeto-trainer-0 x Up: probing anyway (--force)" in capsys.readouterr().err
    ops = _ops_for(path, gpu_apps={"b300-1": GPU_APP})
    assert cluster_cli.main(_args(path, "--force", "--nodes", "b300-0,b300-1"), ops=ops,
                            log=lambda m: None) == cluster_cli.EXIT_BUSY
    assert "b300-1: GPU process" in capsys.readouterr().err and not _probed(ops)
    ops = _ops_for(path, gpu_apps={"b300-1": GPU_APP}, containers={"b300-1": "yeto-x i Up"})
    assert cluster_cli.main(_args(path, "--force-busy-gpu", "--nodes", "b300-0,b300-1"), ops=ops,
                            log=lambda m: None) == cluster_cli.EXIT_BUSY  # the container still needs --force
    capsys.readouterr()
    ops = _ops_for(path, gpu_apps={"b300-1": GPU_APP})
    assert cluster_cli.main(_args(path, "--force-busy-gpu", "--nodes", "b300-0,b300-1"), ops=ops,
                            log=lambda m: None) == 0
    assert f"[WARN] b300-1: GPU process {GPU_APP}: probing anyway (--force-busy-gpu)" in capsys.readouterr().err
    assert _probed(ops)


def test_a_node_another_run_leases_refuses_the_probe_whatever_the_flags(tmp_path, capsys):
    from yeto.launch.node_lease import NodeLeases

    store = f"file://{tmp_path / 'leases'}"
    path = tmp_path / "nodes.yaml"
    path.write_text(f"lease_store: {store}\n" + CUSTOMER_INVENTORY.read_text())
    other = NodeLeases(store, "other-run")
    assert other.acquire(load_inventory(path)[1]).ok
    ops = _ops_for(path)
    rc = cluster_cli.main(_args(path, "--force", "--force-busy-gpu", "--nodes", "b300-0,b300-1"), ops=ops,
                          log=lambda m: None)
    assert rc == cluster_cli.EXIT_BUSY and not _probed(ops)
    err = capsys.readouterr().err
    assert "b300-1: leased by other-run" in err and "nothing overrides a lease" in err
    other.release_all()
    assert cluster_cli.main(_args(path, "--nodes", "b300-0,b300-1"), ops=_ops_for(path), log=lambda m: None) == 0


class _Leases:
    """A lease store: ``holders`` per node, ``after`` reads before they show; ``broken``
    nodes' reads raise."""

    def __init__(self, holders=None, *, after=0, broken=()):
        self.holders, self.after, self.broken, self.reads = holders or {}, after, set(broken), {}

    def holder(self, node):
        self.reads[node.name] = self.reads.get(node.name, 0) + 1
        if node.name in self.broken:
            raise OSError("the store did not answer")
        return self.holders.get(node.name) if self.reads[node.name] > self.after else None


def test_an_unreadable_lease_needs_force(tmp_path, capsys):
    path = _inventory_file(tmp_path)
    ops = _ops_for(path)
    rc = cluster_cli.main(_args(path, "--nodes", "b300-0,b300-1"), ops=ops, log=lambda m: None,
                          leases=_Leases(broken={"b300-1"}))
    assert rc == cluster_cli.EXIT_BUSY and not _probed(ops)
    assert "b300-1: cannot read its lease (OSError: the store did not answer) (--force)" in capsys.readouterr().err
    ops = _ops_for(path)
    rc = cluster_cli.main(_args(path, "--force", "--nodes", "b300-0,b300-1"), ops=ops, log=lambda m: None,
                          leases=_Leases(broken={"b300-1"}))
    assert rc == 0 and _probed(ops)
    err = capsys.readouterr().err
    assert "[WARN] b300-1: cannot read its lease (OSError: the store did not answer): probing anyway (--force)" in err


def test_a_node_leased_during_the_probe_refuses_the_write(tmp_path, capsys):
    path = _inventory_file(tmp_path)
    before = path.read_text()
    leases = _Leases({"b300-1": "other-run"}, after=1)  # free at the start, taken before the write
    rc = cluster_cli.main(_args(path, "--write", "--force", "--force-busy-gpu", "--nodes", "b300-0,b300-1"),
                          ops=_ops_for(path), log=lambda m: None, leases=leases)
    assert rc == cluster_cli.EXIT_BUSY and path.read_text() == before and leases.reads == {"b300-0": 2, "b300-1": 2}
    err = capsys.readouterr().err
    assert "1 node(s) leased by another run during the probe" in err and "nothing written" in err
    assert "b300-1: leased by other-run (nothing overrides a lease)" in err
    leases = _Leases(after=1, broken=())  # nobody took them: written
    assert cluster_cli.main(_args(path, "--write", "--nodes", "b300-0,b300-1"), ops=_ops_for(path),
                            log=lambda m: None, leases=leases) == 0
    assert "fabric_group" in path.read_text()


def test_a_lease_store_that_stops_answering_during_the_probe_needs_force_to_write(tmp_path, capsys):
    class Breaks(_Leases):
        def holder(self, node):
            self.broken = {"b300-1"} if self.reads.get(node.name) else set()
            return super().holder(node)

    path = _inventory_file(tmp_path)
    before = path.read_text()
    assert cluster_cli.main(_args(path, "--write", "--nodes", "b300-0,b300-1"), ops=_ops_for(path),
                            log=lambda m: None, leases=Breaks()) == cluster_cli.EXIT_BUSY
    assert path.read_text() == before and "b300-1: cannot read its lease" in capsys.readouterr().err
    assert cluster_cli.main(_args(path, "--write", "--force", "--nodes", "b300-0,b300-1"), ops=_ops_for(path),
                            log=lambda m: None, leases=Breaks()) == 0
    assert "cannot read its lease (OSError: the store did not answer): keeping the result (--force)" in \
        capsys.readouterr().err
    assert "fabric_group" in path.read_text()


@pytest.mark.parametrize("bad", [{"container": "lxc"}, {"paths": {"bogus": "/x"}}])
def test_a_bad_profile_key_is_an_inventory_error(tmp_path, capsys, bad):
    path = tmp_path / "nodes.yaml"
    path.write_text(yaml.safe_dump({"nodes": [_entry(0, **bad), _entry(1)]}))
    ops = IdleCluster(parse_inventory({"nodes": [_entry(0), _entry(1)]}))
    assert cluster_cli.main(_args(path), ops=ops, log=lambda m: None) == 2
    err = capsys.readouterr().err
    assert f"[yeto] cluster probe: {path}: node b300-0: {next(iter(bad))}" in err and "Traceback" not in err
    assert ops.scripts == []


def test_the_probe_reads_the_inventory_once(tmp_path, monkeypatch):
    path = _inventory_file(tmp_path)
    ops, reads, real = _ops_for(path), [], Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: (reads.append(self), real(self, *a, **k))[1])
    assert cluster_cli.main(_args(path, "--nodes", "b300-0,b300-1"), ops=ops, log=lambda m: None) == 0
    assert [r for r in reads if r.resolve() == path.resolve()] == [path]


def test_an_idle_check_that_raises_is_not_confirmed_idle(tmp_path, capsys):
    class Raising(IdleCluster):
        def run(self, node, script, *, timeout=300.0):
            if f"{cp.SECTION} compute_apps" in script and node.name == "b300-1":
                raise subprocess.TimeoutExpired(["ssh", node.host], timeout)
            return super().run(node, script, timeout=timeout)

    path = _inventory_file(tmp_path)
    ops = Raising(load_inventory(path))
    assert cluster_cli.main(_args(path, "--nodes", "b300-0,b300-1"), ops=ops, log=lambda m: None) == cluster_cli.EXIT_BUSY
    assert "b300-1: cannot confirm it is idle: no answer (TimeoutExpired" in capsys.readouterr().err
    assert not _probed(ops)
    ops = Raising(load_inventory(path))
    assert cluster_cli.main(_args(path, "--force", "--nodes", "b300-0,b300-1"), ops=ops, log=lambda m: None) == 0
    assert "[WARN] b300-1: cannot confirm it is idle: no answer (TimeoutExpired" in capsys.readouterr().err


def test_the_probe_lists_containers_with_the_engine_the_inventory_declares(tmp_path):
    path = tmp_path / "nodes.yaml"
    path.write_text(yaml.safe_dump({"nodes": [_entry(i, container="podman") for i in range(2)]}))
    with pytest.warns(UserWarning, match="node profile keys"):
        ops = IdleCluster(load_inventory(path))
    cluster_cli.main(_args(path), ops=ops, log=lambda m: None)  # the probe reads the profile keys: no warning
    idle = [s for _, s in ops.scripts if f"{cp.SECTION} compute_apps" in s]
    assert len(idle) == 2 and all("podman ps --filter name=yeto-" in s and "docker ps" not in s for s in idle)


def test_the_idle_check_lists_the_containers_of_the_nodes_engine():
    from yeto.launch.node_inventory import NodeProfile

    docker, = _grouped(None)
    sections = dict(cluster_cli.idle_sections(docker))
    assert "docker ps --filter name=yeto-" in sections["yeto_containers"]
    podman = NodeProfile.from_node(docker, container="podman")
    assert "podman ps --filter name=yeto-" in dict(cluster_cli.idle_sections(podman))["yeto_containers"]
    process = parse_inventory({"nodes": [_entry(0, runtime="rootfs")]})[0]
    assert [n for n, _ in cluster_cli.idle_sections(process)] == ["compute_apps"]


@pytest.mark.parametrize("engine, plain, sudo, rc, listed, procs", [
    ("docker", "ok:yeto-a", "none", 0, ["yeto-a"], "sshd"),
    ("docker", "denied", "fail", 1, [], "sshd"),  # neither view lists: not confirmed
    ("docker", "denied", "ok:yeto-b", 0, ["yeto-b"], "sshd"),  # the sudo view is enough for docker's one daemon
    ("docker", "down", "fail", 0, [], "sshd"),  # no daemon running, no shim: no containers
    ("docker", "down", "fail", 1, [], "containerd-shim"),  # live-restore: containers without the daemon
    ("docker", "down", "fail", 1, [], "/usr/bin/containerd-shim-runc-v2"),
    ("docker", "absent", "none", 0, [], "sshd"),
    ("podman", "denied", "ok:yeto-c", 1, ["yeto-c"], "sshd"),  # podman: the user's own view must list
    ("podman", "ok:", "ok:yeto-d", 0, ["yeto-d"], "sshd"),
    ("podman", "down", "fail", 1, [], "sshd"),  # the "daemon down" rule is docker's only
])
def test_the_container_listing_runs_under_bash(tmp_path, engine, plain, sudo, rc, listed, procs):
    shims = tmp_path / "bin"
    shims.mkdir()

    def tool(name, body):
        (shims / name).write_text("#!/bin/sh\n" + body + "\n")
        (shims / name).chmod(0o755)

    def answer(what):
        if what.startswith("ok:"):
            return f"echo '{what[3:]}'" if what[3:] else "exit 0"
        if what == "denied":
            return "echo 'permission denied while trying to connect' >&2; exit 1"
        if what == "down":
            return "echo 'Cannot connect to the Docker daemon at unix:///var/run/docker.sock' >&2; exit 1"
        return "exit 1"

    if plain != "absent":
        tool(engine, answer(plain))
    if sudo == "none":
        tool("sudo", "exit 1")
    else:
        tool("sudo", f'[ "$1" = -n ] && shift; [ "$1" = true ] && exit 0; {answer(sudo)}')
    tool("ps", f'[ "$*" = "-eo comm=" ] || exit 2; printf "%s\\n" init {procs}')
    env = {"PATH": f"{shims}:/usr/bin:/bin", "HOME": str(tmp_path)}
    script = cluster_cli.containers_command(engine)
    if plain == "absent":
        env["PATH"] = f"{shims}:/bin"  # no engine on the PATH at all
    out = subprocess.run(["/bin/bash", "-c", script], env=env, capture_output=True, text=True, timeout=30)
    assert out.returncode == rc, out.stderr
    assert [ln.split()[0] for ln in out.stdout.splitlines() if ln.strip()] == listed


def test_probe_reports_busy_ports_as_the_preflight_does(tmp_path, capsys):
    path = _inventory_file(tmp_path)
    ops = _ops_for(path, behaviors={"b300-1": Behavior(busy=(6379,))})
    assert cluster_cli.main(_args(path, "--nodes", "b300-0,b300-1"), ops=ops, log=lambda m: None) == 0
    assert "[WARN] b300-1 tcp_ports: ports ['6379'] already in use" in capsys.readouterr().err


def test_probe_exits_1_on_a_missing_edge(tmp_path, monkeypatch, capsys):
    path = _inventory_file(tmp_path)
    split = fg.ProbeResult(groups={"fg-a": ["b300-0", "b300-1"]}, bad_edges=[("b300-0", "b300-2", "NCCL: x")])
    monkeypatch.setattr(fg, "group_nodes", lambda *a, **k: split)
    assert cluster_cli.main(_args(path, "--nodes", "b300-0,b300-1,b300-2"), ops=_ops_for(path),
                            log=lambda m: None) == 1
    assert "missing edge b300-0<->b300-2: NCCL: x" in capsys.readouterr().out


def test_probe_deadline_cuts_it_short_and_writes_nothing(tmp_path, capsys):
    path = _inventory_file(tmp_path)
    ticks = iter(range(0, 10 ** 6, 100))  # the idle check takes 4 ticks, the preflight starts past 450
    rc = cluster_cli.main(_args(path, "--write", "--deadline", "450", "--nodes", "b300-0,b300-1"),
                          ops=_ops_for(path), log=lambda m: None, clock=lambda: next(ticks))
    assert rc == 1
    assert "cut the probe short" in capsys.readouterr().err and "fabric_group" not in path.read_text()


def test_the_deadline_shortens_each_call_and_counts_a_late_return():
    seen = []
    now = [0.0]

    class Slow:
        def run(self, node, script, *, timeout=300.0):
            seen.append(timeout)
            now[0] += 50.0  # the call returns, but after the deadline
            return SimpleNamespace(stdout="", stderr="", returncode=0)

    timed = cluster_cli._DeadlineOps(Slow(), 40.0, lambda: now[0])
    timed.run(None, "x", timeout=300.0)
    assert seen == [40.0] and timed.hit
    with pytest.raises(TimeoutError):
        timed.run(None, "x")
    fresh = cluster_cli._DeadlineOps(Slow(), 1000.0, lambda: now[0])
    fresh.run(None, "x", timeout=30.0)
    assert seen[-1] == 30.0 and not fresh.hit


def test_probe_refuses_bad_limits_unknown_nodes_and_too_few(tmp_path, capsys):
    path = _inventory_file(tmp_path)
    assert cluster_cli.main(_args(path, "--nodes", "b300-0,zz")) == 2
    assert cluster_cli.main(_args(path, "--nodes", "b300-0")) == 2
    assert cluster_cli.main(_args(path, "--max-parallel", "1")) == 2
    assert cluster_cli.main(_args(path, "--deadline", "0")) == 2
    err = capsys.readouterr().err
    assert "['zz'] are not in the inventory" in err and "two nodes or more" in err and "--max-parallel" in err


def test_probe_json_output(tmp_path, capsys):
    path = _inventory_file(tmp_path)
    assert cluster_cli.main(_args(path, "--json", "--nodes", "b300-0,b300-1"), ops=_ops_for(path),
                            log=lambda m: None) == 0
    rec = json.loads(capsys.readouterr().out)
    assert list(rec["groups"].values()) == [["b300-0", "b300-1"]] and rec["edges"] == [["b300-0", "b300-1"]]
    assert rec["bad_edges"] == []


def test_probe_max_parallel_bounds_the_sessions(tmp_path):
    import threading
    import time

    path = _inventory_file(tmp_path)
    ops = _ops_for(path)
    live, peak, lock = [0], [0], threading.Lock()
    run = ops.run

    def counted(node, script, *, timeout=300.0):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        try:
            time.sleep(0.02)  # calls overlap as SSH sessions do
            return run(node, script, timeout=timeout)
        finally:
            with lock:
                live[0] -= 1

    ops.run = counted
    assert cluster_cli.main(_args(path, "--max-parallel", "2"), ops=ops, log=lambda m: None) == 0
    assert 3 <= peak[0] <= 4  # 2 pairs at once, both ends of each


def test_preflight_default_nccl_pairs_stay_disjoint():
    nodes, ops = _cluster(4)
    cp.ClusterPreflight(nodes, ops=ops, log=lambda m: None, clock=lambda: FakeCluster.NOW,
                        steps=cp.Steps(ports=False, nccl_all=False)).run()
    assert sum("torch.distributed.run" in s for _, s in ops.scripts) == 2 * 2


# ------------------------------------------------------------------ cluster check
def _check_args(path: Path, *extra: str):
    from yeto.cli import build_parser

    return build_parser().parse_args(["cluster", "check", "--inventory", str(path), *extra])


def test_cluster_check_lists_the_groups_and_places_an_island(tmp_path, capsys):
    path = tmp_path / "nodes.yaml"
    path.write_text(yaml.safe_dump({"nodes": [_entry(0, fabric_group="A"), _entry(1, fabric_group="A"),
                                              _entry(2, fabric_group=FABRIC_UNAVAILABLE), _entry(3)]}))
    assert cluster_cli.main(_check_args(path, "--need", "2")) == 0
    out = capsys.readouterr().out
    assert "fabric group A: b300-0, b300-1 (2 node(s))" in out and "fabric unavailable: b300-2" in out
    assert "no fabric_group: b300-3" in out and "placement of 2 node(s): group A: b300-0, b300-1" in out
    assert cluster_cli.main(_check_args(path, "--need", "3")) == 1
    assert "refused: the island needs 3 healthy nodes" in capsys.readouterr().out
    assert cluster_cli.main(_check_args(CUSTOMER_INVENTORY)) == 0
    assert fg.HINT in capsys.readouterr().err


# --------------------------------------------------------------------- 2.4 doc
def test_the_doc_probe_dry_run_runs_as_written(tmp_path):
    found = re.search(r"<!-- doc-test: probe -->\n```bash\n(.*?)```", DOC.read_text(), re.S)
    assert found
    lines = [ln.split("#")[0].strip() for ln in found.group(1).splitlines() if "--dry-run" in ln or " check " in ln]
    assert any("--dry-run" in ln for ln in lines) and any(" check " in ln for ln in lines)
    shutil.copy(CUSTOMER_INVENTORY, tmp_path / "nodes.yaml")
    env = {**os.environ, "PYTHONPATH": str(REPO), "HOME": str(tmp_path / "home")}
    before = _files(tmp_path)
    for line in lines:
        argv = [sys.executable, "-m", "yeto.cli", *line.split()[1:]]
        out = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
        assert out.returncode == 0, out.stderr
    assert _files(tmp_path) == before
    assert "## Fabric groups" in DOC.read_text()


# ------------------------------------------------------------------ review 892b
def test_a_partial_probe_never_lends_a_name_to_an_unprobed_node(tmp_path):
    """b300-0/1/2 are group A, b300-3 is new: probing b300-0 with b300-3 must not make
    b300-3 an A (b300-3 was never measured against b300-1 / b300-2)."""
    path = tmp_path / "nodes.yaml"
    entries = [_entry(i, fabric_group="A") for i in range(3)] + [_entry(3)]
    path.write_text(yaml.safe_dump({"nodes": [{**e, "image": IMAGE, "docker_args": list(b300(0).docker_args),
                                               "env": dict(b300(0).env)} for e in entries]}, sort_keys=False))
    assert cluster_cli.main(_args(path, "--write", "--nodes", "b300-0,b300-3"), ops=_ops_for(path),
                            log=lambda m: None) == 0
    groups = {n.name: n.fabric_group for n in load_inventory(path)}
    assert groups["b300-1"] == groups["b300-2"] == "A"
    assert groups["b300-3"] != "A" and groups["b300-0"] == groups["b300-3"]
    with pytest.raises(fg.PlacementError):
        fg.pick_placement(load_inventory(path), 4)
    # every member of the new group carried A, and others still hold it: the name stays
    assert fg.assign_ids([["b300-0", "b300-1"]], previous={"b300-0": "A", "b300-1": "A"}, reserved={"A"}) == ["A"]
    assert fg.assign_ids([["b300-0", "b300-3"]], previous={"b300-0": "A"}, reserved={"A"}) != ["A"]
    assert fg.assign_ids([["b300-0", "b300-3"]], previous={"b300-0": "A"}) == ["A"]  # nothing outside holds A


def _no_hca_env(i, **kw):
    return b300(i, env={}, **kw)


def test_a_spare_port_down_without_nccl_ib_hca_keeps_the_edges():
    ports = [cp.IbPort(f"mlx5_{j}", 1, "Active", rate_gbps=400, link_layer="InfiniBand") for j in range(8)]
    ports.append(cp.IbPort("mlx5_8", 1, "Down", rate_gbps=200, link_layer="InfiniBand"))  # an unused (storage) port
    state = SimpleNamespace(node=_no_hca_env(0), ports=ports, compute=ports, gids=[], host={})
    assert fg.node_blocks(["b300-0"], [], [state]) == {}
    ports[0] = cp.IbPort("mlx5_0", 1, "Down", rate_gbps=400, link_layer="InfiniBand")
    assert fg.node_blocks(["b300-0"], [], [state]) == {
        "b300-0": "7 of 9 HCA ports active, fewer than 8 rails"}
    names = ["b300-0", "b300-1"]
    assert fg.group_nodes(names, _pair_checks(names), states=[
        SimpleNamespace(node=_no_hca_env(0), ports=ports[1:] + [ports[0]], compute=[], gids=[], host={})]).groups == {}


def test_eight_active_and_one_down_on_the_fake_cluster_has_edges():
    down = Behavior(host={"ibstat": (0, ibstat_text(n=9, down=(8,), storage=False))})
    nodes = [_no_hca_env(0), _no_hca_env(1)]
    ops = IdleCluster(nodes, **{"b300-1": down})
    result = _probe(nodes, ops)
    assert list(result.groups.values()) == [["b300-0", "b300-1"]]


def _brute_cliques(names, edges):
    import itertools

    edge = {frozenset(e) for e in edges}
    rest, out = sorted(names), []
    while rest:
        for k in range(len(rest), 0, -1):
            found = [list(c) for c in itertools.combinations(rest, k)
                     if all(frozenset(p) in edge for p in itertools.combinations(c, 2))]
            if found:
                best = min(found)
                break
        out.append(best)
        rest = [n for n in rest if n not in best]
    return sorted(out, key=lambda c: (-len(c), c))


def test_cliques_are_optimal_on_every_graph_of_five_nodes():
    import itertools

    names = [f"b{i}" for i in range(5)]
    pairs = list(itertools.combinations(names, 2))
    for mask in range(1 << len(pairs)):
        edges = [p for i, p in enumerate(pairs) if mask >> i & 1]
        assert fg.cliques(names, edges) == _brute_cliques(names, edges), edges


def test_the_greedy_cut_misses_a_three_clique_the_exact_one_finds():
    names = [f"b{i}" for i in range(5)]
    missing = {("b0", "b1"), ("b0", "b4"), ("b2", "b3"), ("b3", "b4")}
    edges = [(a, b) for i, a in enumerate(names) for b in names[i + 1:] if (a, b) not in missing]
    assert fg.cliques(names, edges) == [["b1", "b2", "b4"], ["b0", "b3"]]
    assert any(len(c) == 1 for c in fg._greedy_cliques(names, {frozenset(e) for e in edges}))


def test_a_large_component_is_cut_greedily_and_a_lone_node_joins_a_clique_it_fits():
    names = [f"n{i:02d}" for i in range(18)]
    full = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    outsider = [e for e in full if "n17" not in e or e == ("n00", "n17")]  # n17 reaches only n00
    assert fg.cliques(names, outsider) == [names[:17], ["n17"]]  # the most missing edges leave first
    # n16 misses one edge (to n17): it leaves alone, and joins the 16 others it reaches all of
    joined = [e for e in full if e != ("n16", "n17")]
    out = fg.cliques(names, joined)
    assert len(out) == 2 and sorted(map(len, out)) == [1, 17] and "n16" in out[0]


def test_an_unparseable_inventory_is_an_inventory_error():
    with pytest.raises(InventoryError, match="cannot be edited in place"):
        fg.updated_text("nodes: [\n  - {name: a\n", {"a": "A"})
    with pytest.raises(InventoryError, match="cannot be edited in place"):
        fg.updated_text("{not json", {"a": "A"}, json_file=True)


def test_crlf_inventories_keep_their_newlines(tmp_path):
    path = _inventory_file(tmp_path)
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    assert cluster_cli.main(_args(path, "--write", "--nodes", "b300-0,b300-1"), ops=_ops_for(path),
                            log=lambda m: None) == 0
    data = path.read_bytes()
    assert b"fabric_group" in data and data.count(b"\n") == data.count(b"\r\n")
    js = tmp_path / "nodes.json"
    js.write_bytes(json.dumps({"nodes": [_entry(0), _entry(1)]}, indent=2).replace("\n", "\r\n").encode())
    assert fg.write_groups(js, {"b300-0": "A", "b300-1": "A"})
    data = js.read_bytes()
    assert data.count(b"\n") == data.count(b"\r\n") > 0


def test_pick_placement_keeps_the_required_nodes_and_fills_from_their_group():
    nodes = _grouped("A", "A", "A", "A", "B", "B", "B", "B", "B")
    p = fg.pick_placement(nodes, 3, required=["b300-2"], excluded={"b300-0": "lost"})
    assert p.group == "A" and [n.name for n in p.nodes] == ["b300-2", "b300-1", "b300-3"]  # B is bigger: not taken
    p = fg.pick_placement(nodes, 2, required=[nodes[3]])
    assert [n.name for n in p.nodes] == ["b300-3", "b300-0"]
    with pytest.raises(fg.PlacementError, match="not one fabric group"):
        fg.pick_placement(nodes, 3, required=["b300-0", "b300-4"])
    with pytest.raises(fg.PlacementError, match="required node\\(s\\) excluded: b300-1: XID 79"):
        fg.pick_placement(nodes, 3, required=["b300-1"], excluded={"b300-1": "XID 79"})
    with pytest.raises(fg.PlacementError, match="group A has only 3"):
        fg.pick_placement(nodes, 4, required=["b300-1"], excluded=["b300-0"])
    with pytest.raises(fg.PlacementError, match="not candidates"):
        fg.pick_placement(nodes[:2], 2, required=["b300-7"])
    plain = _grouped(None, None, None)
    assert [n.name for n in fg.pick_placement(plain, 2, required=["b300-2"]).nodes] == ["b300-2", "b300-0"]


def test_excluded_or_required_as_one_string_is_a_type_error():
    nodes = _grouped("A", "A")
    for one in ("b300-0", b"b300-0", bytearray(b"b300-0")):
        with pytest.raises(TypeError, match="excluded"):
            fg.pick_placement(nodes, 1, excluded=one)
        with pytest.raises(TypeError, match="required"):
            fg.pick_placement(nodes, 1, required=one)


def test_required_nodes_come_first_in_the_order_given():
    nodes = _grouped("B", "B", "B", "A", "A", "A")
    p = fg.pick_placement(nodes, 3, required=["b300-5", "b300-3"])
    assert p.group == "A" and [n.name for n in p.nodes] == ["b300-5", "b300-3", "b300-4"]
    plain = _grouped(None, None, None, None)
    assert [n.name for n in fg.pick_placement(plain, 3, required=["b300-3", "b300-1"]).nodes] == [
        "b300-3", "b300-1", "b300-0"]


def test_the_reserved_value_is_read_in_any_case():
    (node,) = parse_inventory({"nodes": [_entry(0, fabric_group="UnAvailable")]})
    assert node.fabric_group == FABRIC_UNAVAILABLE
    loose = b300(0, fabric_group="UNAVAILABLE")  # a node built by hand
    assert not fg.placeable(loose)
    with pytest.raises(fg.PlacementError, match="fabric unavailable"):
        fg.pick_placement([loose, b300(1, fabric_group="UNAVAILABLE")], 2)


def test_a_warn_ib_write_bw_or_a_pair_without_its_own_nccl_run_gives_no_edge():
    names = ["b300-0", "b300-1", "b300-2"]
    checks = _pair_checks(names, ib_status={("b300-0", "b300-1"): ("WARN", "8 rails; worst rail 62% of link")})
    result = fg.group_nodes(names, checks)
    assert ("b300-0", "b300-1") not in result.edges and "ib_write_bw WARN" in result.pair_problems["b300-0<->b300-1"]
    checks = [c for c in _pair_checks(names) if c.scope != "b300-0+b300-2"]  # all mode: that pair has no run
    edges, problems, _ = fg.evaluate_pairs(names, checks, nccl_pairs="all")
    assert ("b300-0", "b300-2") not in edges and problems["b300-0<->b300-2"] == "no NCCL result"
    edges, _, _ = fg.evaluate_pairs(names, checks, nccl_pairs="disjoint")
    assert ("b300-0", "b300-2") in edges  # disjoint mode infers it


def test_net_ib_without_gpudirect_is_a_warn_transport_and_an_edge():
    nodes, ops = _cluster(2, **{"b300-0": Behavior(nccl="ib-nogdr"), "b300-1": Behavior(nccl="ib-nogdr")})
    pf = cp.ClusterPreflight(nodes, ops=ops, log=lambda m: None, clock=lambda: FakeCluster.NOW,
                             steps=cp.Steps(ports=False, nccl_all=False, nccl_pairs="all"))
    report = pf.run()
    (transport,) = [c for c in report.checks if c.name == "nccl_transport"]
    assert transport.status == "WARN" and transport.detail.startswith("NET/IB without GPUDirect")
    assert fg.group_nodes([n.name for n in nodes], report.checks, states=pf.states).edges == [("b300-0", "b300-1")]


def test_the_preflight_runs_both_ends_of_a_pair_together():
    nodes, ops = _cluster(2)
    assert cp.ClusterPreflight(nodes, ops=ops, max_parallel=1).max_parallel == 2
    assert cp.ClusterPreflight(nodes, ops=ops).max_parallel == 32


def test_a_leased_inventory_picks_the_island_in_one_fabric_group(tmp_path, monkeypatch):
    from yeto.launch.node_lease import NodeLeases

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    store = f"file://{tmp_path / 'leases'}"
    doc = {"lease_store": store, "nodes": [_entry(i, fabric_group=g) for i, g in enumerate("AABB")]}
    nodes = parse_inventory(doc)
    other = NodeLeases(store, "other-run")
    assert other.acquire(nodes[0]).ok  # group A keeps one free node, B two
    leases = NodeLeases(store, "this-run")
    args = argparse.Namespace(rl_node_inventory_nodes=nodes, cluster_prefix="fgl", rl_node_lease_wait_s=0.0,
                              gpu="ssh:2x8xb300", dry_run=False)
    picked = inventory_trainer_nodes(args, _island(2), leases=leases, name="fgl-0")
    assert [n.name for n in picked] == ["b300-2", "b300-3"]
    leases.release_all()
    other.release_all()


def test_a_lone_node_of_the_greedy_cut_joins_the_clique_it_reaches_whole():
    names = [f"n{i:02d}" for i in range(17)]
    missing = {("n00", "n02"), ("n02", "n12"), ("n03", "n04"), ("n04", "n10"), ("n08", "n09"), ("n09", "n10")}
    edges = [(a, b) for i, a in enumerate(names) for b in names[i + 1:] if (a, b) not in missing]
    assert ["n10"] in fg._greedy_cliques(names, {frozenset(e) for e in edges})
    big = [n for n in names if n not in ("n02", "n04", "n09")]
    assert fg.cliques(names, edges) == [big, ["n02", "n04", "n09"]]
