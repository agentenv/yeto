"""The join checks of an ssh: trainer island added (or moved) at run time
(own-cluster-fleet-stability 9.3): its inventory's trainer nodes must be free in this fleet,
healthy (node health, absolute rules: a node's first check), in one fabric group and not
leased to another run; any one of them failing refuses ``yeto fleet island add`` / ``move``
with its reason, writing nothing.

The checks run on the real modules: the real :class:`~yeto.hw.node_health.CudaChecker` over
the recorded tool outputs of the node health tests (``node_text``: what its script prints on
an 8 x B300 IB node), the real :class:`~yeto.hw.node_health.NullChecker`, the real
``fabric_groups`` placement and real :class:`~yeto.launch.node_lease.NodeLeases` on a directory
lease store. Only SSH is a stand-in (it answers each node with its recorded output)."""

from __future__ import annotations

import dataclasses
import json
import subprocess

import pytest
from test_cluster_preflight import ibstat_text  # tests/ is on sys.path
from test_fleet_island_changes import fleet_store  # noqa: F401 - fixture
from test_node_health import FABRIC_OK, gpus_text, node_text, xid_line, xid_text

from yeto.hw import node_health
from yeto.launch import island_join as ij
from yeto.launch.learner_ids import claim_key
from yeto.launch.node_inventory import load_inventory
from yeto.launch.node_lease import NodeLeases
from yeto.rl import fleet_islands

GPU = "ssh:2x8xb300@bm"
HCA = "=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7"
XID_79 = node_text(xid=(0, xid_text(xid_line(812.5, 5, 79))))  # in the kernel log of this boot already
ECC_5 = node_text(gpus=(0, gpus_text(ecc={2: 5})))
FM_DOWN = node_text(fabricmanager=(3, "inactive"))  # NVSwitch GPUs (fabric Completed), its fabric manager down
PCIE = node_text(fabricmanager=(3, "inactive"), fabric=(0, "    Fabric\n        State : N/A\n        Status : N/A\n" * 8))
SLOW = node_text(ibstat=(0, ibstat_text(rate=100)))
PORT_DOWN = node_text(ibstat=(0, ibstat_text(down=(3,))))
NVSMI_TIMEOUT = node_text(gpus=(124, ""))
BUSY = f"{node_health.BUSY_MARK} 4242\n"
CUT = node_text()[:node_text().index("@@yeto-section ibstat")]
IB_UNREAD = node_text(ibstat=(1, "ibstat: not found"), ibv_devinfo=(1, "ibv_devinfo: not found"))


class Ssh:
    """Answers each node's check with its recorded output (default: a healthy node)."""

    def __init__(self, out=None, unreachable=()):
        self.out, self.unreachable = dict(out or {}), set(unreachable)
        self.calls: list[str] = []
        self.timeouts: list[float | None] = []

    def run(self, node, script, *, timeout=None):
        self.calls.append(node.name)
        self.timeouts.append(timeout)
        if node.name in self.unreachable:
            return subprocess.CompletedProcess([], 255, stdout="", stderr="ssh: connect to host: timed out\n")
        return subprocess.CompletedProcess([], 0, stdout=self.out.get(node.name, node_text()), stderr="")


def _entry(name, i, *, env=None, group=None, host=None, address=None, min_ib=None, family="cuda"):
    number = int(name.rsplit("-", 1)[1]) if name.rsplit("-", 1)[-1].isdigit() else 100 + i
    return (f"  - {{name: {name}, host: {host or f'ops@{name}.example'}, "  # one machine: one address
            f"address: {address or f'198.51.100.{10 + number}'}, interface: bond0, "
            f"device_family: {family}, cpu_arch: x86_64, card: B300, cards_per_node: 8, hbm_gib: 288, "
            "host_memory_gib: 2048, fabric: ib, runtime: docker, docker_args: [--gpus, all]"
            + (f", env: {{NCCL_IB_HCA: '{env}'}}" if env else "")
            + (f", fabric_group: {group}" if group else "")
            + (f", min_ib_gbps: {min_ib}" if min_ib else "") + "}\n")


def _inventory(path, names, *, env=HCA, groups=None, lease_store=None, hosts=None, addresses=None, min_ib=None):
    path.write_text((f"lease_store: {lease_store}\n" if lease_store else "") + "nodes:\n" + "".join(
        _entry(name, i, env=env, group=(groups or {}).get(name), host=(hosts or {}).get(name),
               address=(addresses or {}).get(name), min_ib=min_ib)
        for i, name in enumerate(names)))
    return path


def _nodes(tmp_path, names=("b300-4", "b300-5"), **kw):
    return load_inventory(_inventory(tmp_path / "inv.yaml", names, **kw))


def _check(nodes, ssh, need=2, **kw):
    return ij.check(nodes, need=need, run_id="r-i2", ssh=ssh, inventory=nodes, **kw)


# -- health: the real checker, absolute rules ------------------------------------------------------

def test_healthy_nodes_join_with_one_bounded_check_each(tmp_path):
    ssh = Ssh()
    out = _check(_nodes(tmp_path), ssh)
    assert out.ok and out.nodes == ["b300-4", "b300-5"] and out.warnings == []
    assert ssh.calls == ["b300-4", "b300-5"]
    assert all(t is not None and 0 < t <= ij.HEALTH_TIMEOUT_S for t in ssh.timeouts)  # never unbounded


@pytest.mark.parametrize(("text", "why"), [
    (XID_79, "GPU 5 XID 79" + ij.XID_HINT),
    (ECC_5, "GPU 2 uncorrectable ECC 5 (volatile)"),
    (FM_DOWN, "nvidia-fabricmanager inactive on an NVSwitch node"),
    (node_text(gpus=(0, gpus_text(n=7))), "7 of 8 GPUs visible"),
])
def test_a_fault_a_first_check_cannot_see_as_a_change_refuses(tmp_path, text, why):
    """The checker's rules compare with the previous check; a joining node has none: a
    hardware XID of this boot, a volatile uncorrectable ECC count, a stopped fabric manager
    on NVSwitch GPUs refuse it."""
    out = _check(_nodes(tmp_path), Ssh({"b300-5": text}))
    assert out.refusals == [f"node b300-5 is not healthy: {why}"] and out.nodes == []


def test_a_pcie_node_without_nvswitch_joins_with_its_fabric_manager_inactive(tmp_path):
    assert _check(_nodes(tmp_path), Ssh({"b300-4": PCIE, "b300-5": PCIE})).ok


def test_a_port_down_refuses_a_multi_node_island_only(tmp_path):
    assert _check(_nodes(tmp_path), Ssh({"b300-4": PORT_DOWN})).refusals == [
        "node b300-4 is not healthy: mlx5_3 port DOWN"]
    (one,) = _nodes(tmp_path, names=("b300-4",))
    assert _check([one], Ssh({"b300-4": PORT_DOWN}), need=1).ok


def test_a_slow_port_is_refused_against_the_island_s_fastest_or_min_ib_gbps(tmp_path):
    out = _check(_nodes(tmp_path), Ssh({"b300-5": SLOW}))
    assert len(out.refusals) == 1 and out.refusals[0].startswith(
        "node b300-5 is not healthy: mlx5_0 rate 100 < 200 Gb/s (half the fastest job port of the island's nodes, "
        "400 Gb/s)")
    assert _check(_nodes(tmp_path), Ssh({"b300-4": SLOW, "b300-5": SLOW})).ok  # all alike: no slower one
    declared = _nodes(tmp_path, min_ib=400)  # the inventory's min_ib_gbps: every node held to it
    out = _check(declared, Ssh({"b300-4": SLOW, "b300-5": SLOW}))
    assert [r.split(":")[0] for r in out.refusals] == ["node b300-4 is not healthy", "node b300-5 is not healthy"]
    assert "mlx5_0 rate 100 < 400 Gb/s" in out.refusals[0]


@pytest.mark.parametrize(("text", "why"), [
    (NVSMI_TIMEOUT, f"nvidia-smi: timed out after {node_health.SECTION_TIMEOUT_S}s"),
    (BUSY, "node health check skipped: the previous one still runs (a tool stuck on the node)"),
    (CUT, "node health output incomplete (no ibstat, ibv_devinfo)"),
])
def test_a_check_that_does_not_read_refuses_unless_forced(tmp_path, text, why):
    nodes = _nodes(tmp_path)
    out = _check(nodes, Ssh({"b300-4": text}))
    assert out.refusals == [f"node b300-4: its health check did not read ({why})"]
    forced = _check(nodes, Ssh({"b300-4": text}), force=True)
    assert forced.ok and forced.warnings == [f"node b300-4: its health check did not read ({why}); --force: it "
                                             "joins unchecked"]


def test_ib_tools_unread_refuse_a_multi_node_island_unless_forced(tmp_path):
    """ok(multi) None: the GPUs pass, the job's ports did not read."""
    nodes = _nodes(tmp_path)
    out = _check(nodes, Ssh({"b300-5": IB_UNREAD}))
    assert out.refusals == [f"node b300-5: its health check did not read ({node_health.IB_UNREAD})"]
    assert _check(nodes, Ssh({"b300-5": IB_UNREAD}), force=True).warnings
    (one,) = _nodes(tmp_path, names=("b300-5",))
    assert _check([one], Ssh({"b300-5": IB_UNREAD}), need=1).ok  # alone: its GPUs only


def test_force_never_lets_an_unhealthy_node_join(tmp_path):
    assert _check(_nodes(tmp_path), Ssh({"b300-5": XID_79}), force=True).refusals == [
        "node b300-5 is not healthy: GPU 5 XID 79" + ij.XID_HINT]


def test_a_node_that_does_not_answer_is_refused(tmp_path):
    out = _check(_nodes(tmp_path), Ssh(unreachable={"b300-4"}))
    assert out.refusals == ["node b300-4: its health check did not answer (ssh exit 255: ssh: connect to host: "
                            "timed out)"]


def test_the_real_null_checker_is_a_note_with_no_ssh_call(tmp_path):
    nodes = [dataclasses.replace(n, device_family="no-such-family") for n in _nodes(tmp_path)]
    assert isinstance(ij._checker_for(nodes[0]), node_health.NullChecker)
    ssh = Ssh()
    out = _check(nodes, ssh)
    assert out.ok and ssh.calls == []
    assert out.notes[:2] == [f"node {n}: no health check for device family no-such-family" for n in ("b300-4", "b300-5")]


def test_a_node_with_its_health_check_off_is_noted_and_not_called(tmp_path):
    first, second = _nodes(tmp_path)
    ssh = Ssh({"b300-4": XID_79})
    out = _check([dataclasses.replace(first, node_health="off"), second], ssh)
    assert out.ok and ssh.calls == ["b300-5"]
    assert "node b300-4: node_health off in the inventory: its health is not checked" in out.notes


def test_the_island_s_own_nccl_ib_hca_wins_over_the_node_s(tmp_path):
    """The job's HCAs are {**node.env, **island env}, as the island's launch exports them."""
    nodes = _nodes(tmp_path, env="=mlx5_0,mlx5_1")
    assert _check(nodes, Ssh({"b300-4": PORT_DOWN})).ok  # mlx5_3 is not the node's job port
    out = _check(nodes, Ssh({"b300-4": PORT_DOWN}), env={"NCCL_IB_HCA": "=mlx5_2,mlx5_3"})
    assert out.refusals == ["node b300-4 is not healthy: mlx5_3 port DOWN"]
    assert ij.island_env_of(["--island-env", "A=1", "--island-env=NCCL_IB_HCA==mlx5_3"]) == {
        "A": "1", "NCCL_IB_HCA": "=mlx5_3"}
    # the member's launch argv: the fleet's --island-env, then the island's own
    inventory = _inventory(tmp_path / "island.yaml", ["b300-4", "b300-5"], env="=mlx5_0,mlx5_1")
    member = {"name": "north", "gpu": GPU, "argv": ["--rl-node-inventory", str(inventory),
                                                   "--island-env", "NCCL_IB_HCA==mlx5_2,mlx5_3"]}
    out = ij.check_member(member, {"members": []}, fleet_argv=["--island-env", "NCCL_IB_HCA==mlx5_0"],
                          run_id="r-i2", ssh=Ssh({"b300-4": PORT_DOWN}), log=lambda m: None)
    assert out.refusals == ["node b300-4 is not healthy: mlx5_3 port DOWN"]


# -- this fleet, without lease_store: exact count, overlap by machine -------------------------------

def test_the_wrong_count_and_a_node_of_another_island_are_refused(tmp_path):
    nodes = _nodes(tmp_path)
    assert _check(nodes, Ssh(), need=3).refusals == [
        "the island needs 3 trainer node(s); its inventory lists 2 ['b300-4', 'b300-5']"]
    other = load_inventory(_inventory(tmp_path / "other.yaml", ["b300-5"], addresses={"b300-5": "203.0.113.5"}))
    taken = {k: "sglang" for k in ij.node_keys(other[0])}  # its name
    assert _check(nodes, Ssh(), taken=taken).refusals == ["node b300-5 is island sglang's"]


@pytest.mark.parametrize(("host", "address"), [("root@B300-5.Example", "203.0.113.9"),  # the same SSH target
                                               ("b300-5-alias", "198.51.100.15")])  # the same address
def test_the_same_machine_under_another_name_or_user_is_another_island_s(tmp_path, host, address):
    (theirs,) = load_inventory(_inventory(tmp_path / "theirs.yaml", ["their-node"], hosts={"their-node": host},
                                          addresses={"their-node": address}))
    taken = {k: "south" for k in ij.node_keys(theirs)}
    out = _check(_nodes(tmp_path), Ssh(), taken=taken)  # b300-5: ops@b300-5.example, 198.51.100.15
    assert out.refusals == ["node b300-5 is island south's"]


# -- fabric groups ----------------------------------------------------------------------------------

def test_the_real_placement_refuses_nodes_of_two_fabric_groups(tmp_path):
    def run(groups, names=("b300-4", "b300-5")):
        inventory = load_inventory(_inventory(tmp_path / "g.yaml", ["b300-3", *names], groups=groups))
        nodes = [n for n in inventory if n.name in names]
        return ij.check(nodes, need=2, run_id="r", ssh=Ssh(), inventory=inventory)

    split = run({"b300-3": "a", "b300-4": "a", "b300-5": "b"})
    assert len(split.refusals) == 1 and split.refusals[0].startswith("fabric: ")
    assert run({"b300-3": "a", "b300-4": "b", "b300-5": "b"}).ok
    plain = run(None)  # no group anywhere: placed as before, with the hint as a note
    assert plain.ok and any(n.startswith("fabric: ") and "fabric_group" in n for n in plain.notes)
    other = run({"b300-3": "a"})  # groups in the inventory, none on the island's nodes
    assert not other.ok and other.refusals[0].startswith("fabric: ")


# -- leases: the real NodeLeases on a directory store ----------------------------------------------

def _hold(store, node, run, identity="x"):
    assert NodeLeases(store, run, identity=identity).acquire(node).ok


def test_a_node_another_run_holds_is_refused_naming_that_run(tmp_path):
    store = str(tmp_path / "leases")
    nodes = load_inventory(_inventory(tmp_path / "inv.yaml", ["b300-4", "b300-5"], lease_store=store))
    _hold(store, nodes[0], "other-run/trainer")
    out = _check(nodes, Ssh(), lease_store=store)
    assert out.refusals == ["node b300-4 is leased to run other-run",
                            "1 of 2 trainer node(s) are free and healthy, 2 needed"]


@pytest.mark.parametrize(("holder", "own"), [
    ("r-i2/trainer", True), ("r-i2-m1/trainer", True), ("r-i2-m12/trainer", True),
    ("r-i2-mtest/trainer", False), ("r-i2-m/pool", False), ("r-i2-m/trainer", False), ("r-i2/pool", False),
    ("r-i20/trainer", False), ("xr-i2/trainer", False), ("r-i2-m1-m2/trainer", False)])
def test_only_the_island_s_own_launch_owns_its_lease(tmp_path, holder, own):
    store = str(tmp_path / "leases")
    nodes = load_inventory(_inventory(tmp_path / "inv.yaml", ["b300-4", "b300-5"], lease_store=store))
    _hold(store, nodes[0], holder)
    out = _check(nodes, Ssh(), lease_store=store)
    assert out.ok is own, out.refusals
    assert ij.own_lease(f"{holder}@x", "r-i2") is own


def test_a_lease_that_does_not_read_refuses(tmp_path, monkeypatch):
    class Broken:
        def holder(self, node):
            raise OSError("store unreachable")

    monkeypatch.setattr(ij, "_node_leases", lambda uri, run: Broken())
    nodes = load_inventory(_inventory(tmp_path / "inv.yaml", ["b300-4", "b300-5"], lease_store="s3://b/l"))
    out = _check(nodes, Ssh(), lease_store="s3://b/l")
    assert out.refusals == [f"node {n}: its lease does not read (OSError: store unreachable)" for n in ("b300-4", "b300-5")]
    monkeypatch.setattr(ij, "_node_leases", lambda uri, run: (_ for _ in ()).throw(RuntimeError("no such bucket")))
    assert _check(nodes, Ssh(), lease_store="s3://b/l").refusals == ["lease store s3://b/l: RuntimeError: no such bucket"]


# -- a shared inventory with lease_store: the island takes need of more nodes -----------------------

FOUR = ("b300-0", "b300-1", "b300-2", "b300-3")


def _shared(tmp_path, **kw):
    store = str(tmp_path / "leases")
    return store, load_inventory(_inventory(tmp_path / "shared.yaml", FOUR, lease_store=store, **kw))


def test_two_free_nodes_of_a_shared_inventory_make_the_island(tmp_path):
    """The spec's scenario: the fleet's island north holds two nodes of the inventory; the
    new island takes the other two free ones."""
    store, nodes = _shared(tmp_path)
    for node in nodes[:2]:
        _hold(store, node, "r-i0/trainer")
    ssh = Ssh()
    out = _check(nodes, ssh, lease_store=store, islands={"r-i0": "north"})
    assert out.ok and out.nodes == ["b300-2", "b300-3"] and ssh.calls == ["b300-2", "b300-3"]
    assert "left out: node b300-0 is island north's (its launch holds the lease)" in out.notes


def test_a_shared_inventory_with_one_node_leased_and_one_unhealthy_is_refused(tmp_path):
    store, nodes = _shared(tmp_path)
    for node in nodes[:2]:
        _hold(store, node, "r-i0/trainer")
    _hold(store, nodes[2], "other-run/trainer")
    out = _check(nodes, Ssh({"b300-3": XID_79}), lease_store=store, islands={"r-i0": "north"})
    assert out.refusals == ["node b300-0 is island north's (its launch holds the lease)",
                            "node b300-1 is island north's (its launch holds the lease)",
                            "node b300-2 is leased to run other-run",
                            "node b300-3 is not healthy: GPU 5 XID 79" + ij.XID_HINT,
                            "0 of 4 trainer node(s) are free and healthy, 2 needed"]


def test_the_group_placement_picks_from_the_free_healthy_nodes(tmp_path):
    store, nodes = _shared(tmp_path, groups={"b300-0": "a", "b300-1": "a", "b300-2": "b", "b300-3": "b"})
    _hold(store, nodes[0], "other-run/trainer")
    out = _check(nodes, Ssh(), lease_store=store)
    assert out.ok and out.nodes == ["b300-2", "b300-3"]  # group a has one free node
    out = _check(nodes, Ssh({"b300-3": ECC_5}), lease_store=store)
    assert not out.ok and out.refusals[-1].startswith("fabric: the island needs 2 healthy nodes of one fabric group")
    assert "node b300-3 is not healthy: GPU 2 uncorrectable ECC 5 (volatile)" in out.refusals


MAY_TAKE = "the island's launch picks its nodes by their leases and may take {}; take such a node out of the " \
    "inventory's trainer nodes (roles) or clear its fault"


def test_an_unhealthy_node_the_launch_would_pick_by_its_leases_is_refused(tmp_path):
    """The island's launch picks by leases only (in order): any unhealthy candidate refuses,
    rather than pass nodes the launch does not take."""
    store, nodes = _shared(tmp_path)
    _hold(store, nodes[0], "other-run/trainer")
    out = _check(nodes, Ssh({"b300-1": XID_79}), lease_store=store)
    assert out.refusals == ["node b300-1 is not healthy: GPU 5 XID 79" + ij.XID_HINT, MAY_TAKE.format("b300-1")]


def test_the_launch_s_pick_after_a_move_takes_no_node_as_its_own(tmp_path):
    """Scenario A. After a move the launch runs as ``r-i2-m1/trainer@<new identity>``: the
    nodes the earlier incarnation (``r-i2``) holds are not its own, so it takes the free ones
    in the inventory's order, as the real ``pick_nodes`` does."""
    from yeto.launch.node_lease import pick_nodes

    store = str(tmp_path / "leases")
    nodes = load_inventory(_inventory(tmp_path / "inv.yaml", ["c-1", "d-2", "a-3", "b-4"], lease_store=store))
    for node in nodes[2:]:
        _hold(store, node, "r-i2/trainer")
    launch = NodeLeases(store, "r-i2-m1/trainer", identity="new")
    picked = pick_nodes(list(nodes), 2, launch, wait_s=0, sleep=lambda s: None)
    assert [n.name for n in picked] == ["c-1", "d-2"]
    for node in picked:
        launch.release(node)
    out = _check(nodes, Ssh({"c-1": XID_79}), lease_store=store)
    assert out.refusals == ["node c-1 is not healthy: GPU 5 XID 79" + ij.XID_HINT, MAY_TAKE.format("c-1")]
    fine = _check(nodes, Ssh(), lease_store=store)
    assert fine.ok and fine.nodes == ["c-1", "d-2"]
    _hold(store, nodes[1], "other-run/trainer")  # one free node: the free and the earlier incarnation's
    assert _check(nodes, Ssh(), lease_store=store).nodes == ["c-1", "a-3"]


def test_scenario_b_the_free_nodes_after_the_earlier_incarnation_s(tmp_path):
    """[a-1, b-2 (the earlier incarnation's), c-3, d-4 (free)]: the launch takes c-3, d-4; an
    unhealthy c-3 refuses although a-1 and b-2 are fine."""
    store = str(tmp_path / "leases")
    nodes = load_inventory(_inventory(tmp_path / "inv.yaml", ["a-1", "b-2", "c-3", "d-4"], lease_store=store))
    for node in nodes[:2]:
        _hold(store, node, "r-i2/trainer")
    assert _check(nodes, Ssh({"c-3": XID_79}), lease_store=store).refusals[-1] == MAY_TAKE.format("c-3")
    ok = _check(nodes, Ssh(), lease_store=store)
    assert ok.nodes == ["c-3", "d-4"] and not any("meanwhile" in n for n in ok.notes)


def test_scenario_d_an_unhealthy_candidate_no_predicted_pick_takes_still_refuses(tmp_path):
    """[c-1, d-2 free; a-3, b-4 the earlier incarnation's; e-5 free and unhealthy]: the picks
    take c-1, d-2, then a-3, b-4; e-5 still refuses (a pick may come to any candidate)."""
    store = str(tmp_path / "leases")
    nodes = load_inventory(_inventory(tmp_path / "inv.yaml", ["c-1", "d-2", "a-3", "b-4", "e-5"], lease_store=store))
    for node in nodes[2:4]:
        _hold(store, node, "r-i2/trainer")
    assert _check(nodes, Ssh({"e-5": XID_79}), lease_store=store).refusals[-1] == MAY_TAKE.format("e-5")
    assert _check(nodes, Ssh(), lease_store=store).nodes == ["c-1", "d-2"]


def test_a_launch_that_would_take_another_island_s_nodes_is_refused(tmp_path):
    """North (no lease_store) runs on b300-0 and b300-1 of south's shared inventory: south's
    launch sees them free in its store and takes them first: refused."""
    store, nodes = _shared(tmp_path)
    taken = {k: "north" for n in nodes[:2] for k in ij.node_keys(n)}
    out = _check(nodes, Ssh(), lease_store=store, taken=taken)
    assert out.refusals == ["launch would take b300-0, b300-1, island north's: give both islands the same "
                            "lease_store, or use inventories without shared nodes"]
    late = {k: "north" for n in nodes[2:] for k in ij.node_keys(n)}  # after the free nodes: the second pick
    assert _check(nodes, Ssh(), lease_store=store, taken=late).refusals == [
        "launch would take b300-2, b300-3, island north's: give both islands the same lease_store, or use "
        "inventories without shared nodes"]
    three = load_inventory(_inventory(tmp_path / "three.yaml", FOUR[:3], lease_store=store))
    one = {k: "north" for k in ij.node_keys(three[2])}  # no pick comes to b300-2: it may join
    out = _check(three, Ssh(), lease_store=store, taken=one)
    assert out.ok and out.nodes == ["b300-0", "b300-1"] and "left out: node b300-2 is island north's" in out.notes


def test_the_same_machine_under_another_name_on_one_lease_store_is_another_island_s(tmp_path):
    store, nodes = _shared(tmp_path)
    (theirs,) = load_inventory(_inventory(tmp_path / "theirs.yaml", ["north-a"], lease_store=store,
                                          hosts={"north-a": "root@B300-0.example"},
                                          addresses={"north-a": "203.0.113.50"}))
    aliases = {k: ("north", "north-a") for k in ij.node_keys(theirs) - {"name:north-a"}}
    out = _check(nodes, Ssh(), lease_store=store, aliases=aliases)
    assert out.refusals == ["launch would take b300-0, island north's (its node north-a, the same machine): give "
                            "the machine one name in both inventories, or use inventories without shared nodes"]
    same = {k: ("north", "b300-0") for k in ij.node_keys(nodes[0]) - {"name:b300-0"}}  # one name: the lease decides
    assert _check(nodes, Ssh(), lease_store=store, aliases=same).ok


def test_lease_stores_compare_in_one_spelling():
    assert ij.store_key("file:///srv/leases/") == ij.store_key("/srv/leases") == "/srv/leases"
    assert ij.store_key("s3://b/l/") == "s3://b/l" and ij.store_key(None) is None


def test_min_ib_gbps_without_nccl_ib_hca_holds_the_fabric_s_ports(tmp_path):
    """No NCCL_IB_HCA on an ib fabric: NCCL takes the InfiniBand ports; min_ib_gbps holds each
    active one (the checker's rule, the same on a running island)."""
    nodes = _nodes(tmp_path, env=None, min_ib=400)
    out = _check(nodes, Ssh({"b300-4": SLOW, "b300-5": SLOW}))
    assert [r.split(": ", 1)[0] for r in out.refusals] == ["node b300-4 is not healthy", "node b300-5 is not healthy"]
    assert "mlx5_0 rate 100 < 400 Gb/s" in out.refusals[0]
    assert _check(nodes, Ssh()).ok
    checker = node_health.checker_for("cuda", expect_gpus=8, min_ib_gbps=400, fabric="ib")
    assert "mlx5_7 rate 100 < 400 Gb/s" in checker.parse(SLOW, [], None).reasons
    assert checker.parse(SLOW, [], None).ok_single is True  # a single-node job: its GPUs only


def test_a_rate_of_exactly_half_the_fastest_passes(tmp_path):
    half = node_text(ibstat=(0, ibstat_text(rate=200)))
    assert _check(_nodes(tmp_path), Ssh({"b300-5": half})).ok
    below = node_text(ibstat=(0, ibstat_text(rate=199)))
    assert not _check(_nodes(tmp_path), Ssh({"b300-5": below})).ok


def test_the_relative_rate_compares_nodes_of_one_fabric_group_only(tmp_path):
    store, nodes = _shared(tmp_path, groups={"b300-0": "a", "b300-1": "a", "b300-2": "b", "b300-3": "b"})
    slow_b = {"b300-2": SLOW, "b300-3": SLOW}  # group b runs at 100 Gb/s, group a at 400: not compared
    out = _check(nodes, Ssh(slow_b), lease_store=store)
    assert out.ok and out.nodes == ["b300-0", "b300-1"]
    out = _check(nodes, Ssh({"b300-3": SLOW}), lease_store=store)
    assert out.refusals[0] == ("node b300-3 is not healthy: mlx5_0 rate 100 < 200 Gb/s (half the fastest job port of "
                               "the island's candidate nodes in fabric group b, 400 Gb/s)"
                               + "".join(f"; mlx5_{i} rate 100 < 200 Gb/s (half the fastest job port of the island's "
                                         "candidate nodes in fabric group b, 400 Gb/s)" for i in range(1, 8)))


def test_one_uncorrectable_ecc_error_refuses(tmp_path):
    out = _check(_nodes(tmp_path), Ssh({"b300-4": node_text(gpus=(0, gpus_text(ecc={0: 1})))}))
    assert out.refusals == ["node b300-4 is not healthy: GPU 0 uncorrectable ECC 1 (volatile)"]


@pytest.mark.parametrize(("over", "why"), [
    ({"xid": (0, xid_text(source="none"))}, ij.XID_UNREAD),
    ({"fabric": (124, "")}, "nvidia-smi -q (NVLink fabric) timed out"),
    ({"fabric": (137, "")}, "nvidia-smi -q (NVLink fabric) timed out"),  # killed after its timeout
    ({"xid": (124, "boot=" + "0" * 8)}, ij.XID_TIMED_OUT),  # timed out: not "restricted"
    ({"fabricmanager": (124, "")}, "systemctl is-active nvidia-fabricmanager timed out"),
])
def test_a_kernel_log_or_fabric_that_does_not_read_refuses_unless_forced(tmp_path, over, why):
    nodes = _nodes(tmp_path)
    out = _check(nodes, Ssh({"b300-4": node_text(**over)}))
    assert out.refusals == [f"node b300-4: its health check did not read ({why})"]
    assert _check(nodes, Ssh({"b300-4": node_text(**over)}), force=True).ok


def test_the_island_s_inventory_wins_over_the_fleet_s(tmp_path):
    fleet_inv = _inventory(tmp_path / "fleet.yaml", ["b300-0", "b300-1"])
    island_inv = _inventory(tmp_path / "island.yaml", ["b300-4", "b300-5"])
    member = {"name": "north", "gpu": GPU, "argv": ["--rl-node-inventory", str(island_inv)]}
    ssh = Ssh()
    out = ij.check_member(member, {"members": []}, fleet_argv=["--rl-node-inventory", str(fleet_inv)],
                          run_id="r-i2", ssh=ssh, log=lambda m: None)
    assert out.ok and out.nodes == ["b300-4", "b300-5"] and ssh.calls == ["b300-4", "b300-5"]


def test_a_recheck_prints_only_what_the_first_check_did_not(tmp_path):
    inventory = _inventory(tmp_path / "island.yaml", ["b300-4", "b300-5"])
    member = {"name": "north", "gpu": GPU, "argv": ["--rl-node-inventory", str(inventory)]}
    lines: list[str] = []
    first = ij.check_member(member, {"members": []}, fleet_argv=[], run_id="r-i2", ssh=Ssh({"b300-4": BUSY}),
                            force=True, log=lines.append)
    assert any("WARNING" in line for line in lines) and any("note:" in line for line in lines)
    lines.clear()
    ij.check_member(member, {"members": []}, fleet_argv=[], run_id="r-i2", ssh=Ssh(), force=True,
                    log=lines.append, health=first.health, seen=[*first.notes, *first.warnings])
    assert lines == []
    ij.check_member(member, {"members": []}, fleet_argv=[], run_id="r-i2", ssh=Ssh({"b300-5": BUSY}), force=True,
                    log=lines.append, seen=[*first.notes, *first.warnings])
    assert lines == ["fleet island: WARNING: island north: node b300-5: its health check did not read (node health "
                     "check skipped: the previous one still runs (a tool stuck on the node)); --force: it joins "
                     "unchecked"]


# -- yeto fleet island add / move -----------------------------------------------------------------

def _add(run_cli, samplers, inventory, *extra, name="north"):
    return run_cli("add", "--name", name, "--gpu", GPU, "--usd-per-h", "0", "--samplers", str(samplers),
                   "--inventory", str(inventory), *extra)


@pytest.fixture
def wired(monkeypatch):
    """SSH answers healthy (``ssh.out``: per node); the launch dry run and session check pass."""
    ssh = Ssh()
    monkeypatch.setattr("yeto.launch.ssh_trainer_island.SshOps", lambda: ssh)
    monkeypatch.setattr(fleet_islands, "launch_dry_run", lambda member, **kw: None)
    monkeypatch.setattr(fleet_islands, "session_check", lambda *a, **kw: None)
    return ssh


@pytest.mark.parametrize("problem", ["health", "fabric", "lease", "unread"])
def test_add_refuses_an_ssh_island_failing_one_check_and_writes_nothing(fleet_store, tmp_path,  # noqa: F811
                                                                         monkeypatch, capsys, problem):
    _cfg, objects, store, run_cli, samplers = fleet_store
    ssh = Ssh()
    monkeypatch.setattr("yeto.launch.ssh_trainer_island.SshOps", lambda: ssh)
    inventory = _inventory(tmp_path / "north.yaml", ["b300-4", "b300-5"])
    if problem == "health":
        ssh.out["b300-4"] = node_text(gpus=(0, gpus_text(n=7)))
        why = "node b300-4 is not healthy: 7 of 8 GPUs visible"
    elif problem == "fabric":
        inventory = _inventory(tmp_path / "north.yaml", ["b300-4", "b300-5"], groups={"b300-4": "a", "b300-5": "b"})
        why = "fabric: the island needs 2 healthy nodes of one fabric group"
    elif problem == "lease":
        leases = str(tmp_path / "leases")
        inventory = _inventory(tmp_path / "north.yaml", ["b300-4", "b300-5"], lease_store=leases)
        _hold(leases, load_inventory(inventory)[1], "another-run/trainer")
        why = "node b300-5 is leased to run another-run"
    else:
        ssh.out["b300-5"] = NVSMI_TIMEOUT
        why = "node b300-5: its health check did not read (nvidia-smi: timed out after"
    monkeypatch.setattr(fleet_islands, "launch_dry_run", lambda *a, **kw: pytest.fail("checked before the dry run"))
    for dry in ((), ("--dry-run",)):
        assert _add(run_cli, samplers, inventory, *dry) == 2
        assert "refused: island north's nodes cannot join: " + why in capsys.readouterr().err
    assert store.read()["generation"] == 1 and objects.read(claim_key(2)) is None


def test_add_with_force_takes_an_island_whose_check_does_not_read_with_a_warning(fleet_store, wired, tmp_path,  # noqa: F811
                                                                                   capsys):
    _cfg, _objects, store, run_cli, samplers = fleet_store
    wired.out["b300-5"] = BUSY
    inventory = _inventory(tmp_path / "north.yaml", ["b300-4", "b300-5"])
    assert _add(run_cli, samplers, inventory, "--force") == 0
    assert "WARNING: island north: node b300-5: its health check did not read" in capsys.readouterr().err
    assert store.read()["generation"] == 2
    wired.out["b300-6"] = XID_79
    south = _inventory(tmp_path / "south.yaml", ["b300-6", "b300-7"])
    assert _add(run_cli, samplers, south, "--force", name="south") == 2  # unhealthy: never forced


def test_add_takes_an_ssh_island_whose_nodes_pass(fleet_store, wired, tmp_path, monkeypatch, capsys):  # noqa: F811
    _cfg, objects, store, run_cli, samplers = fleet_store
    inventory = _inventory(tmp_path / "north.yaml", ["b300-4", "b300-5"])
    dry_runs = []
    monkeypatch.setattr(fleet_islands, "launch_dry_run", lambda member, **kw: dry_runs.append(member))
    assert _add(run_cli, samplers, inventory) == 0
    out = capsys.readouterr()
    assert "island north takes nodes b300-4, b300-5; its nodes pass the join checks" in out.out
    assert "note: island north: no lease_store" in out.err
    (member,) = dry_runs
    assert member["argv"][-2:] == ["--rl-node-inventory", str(inventory.resolve())] and member["gpu"] == GPU
    assert store.read()["generation"] == 2 and objects.read(claim_key(2)) is not None


def test_the_write_s_recheck_does_not_print_the_first_check_s_notes_again(fleet_store, wired, tmp_path,  # noqa: F811
                                                                          capsys):
    _cfg, _objects, _store, run_cli, samplers = fleet_store
    wired.out["b300-5"] = BUSY
    assert _add(run_cli, samplers, _inventory(tmp_path / "north.yaml", ["b300-4", "b300-5"]), "--force") == 0
    err = capsys.readouterr().err
    assert err.count("note: island north: no lease_store") == 1
    assert err.count("WARNING: island north: node b300-5") == 1


def test_the_dry_run_of_a_shared_inventory_prints_the_nodes_the_island_takes(fleet_store, wired, tmp_path,  # noqa: F811
                                                                              capsys):
    _cfg, _objects, store, run_cli, samplers = fleet_store
    leases = str(tmp_path / "leases")
    inventory = _inventory(tmp_path / "shared.yaml", FOUR, lease_store=leases)
    _hold(leases, load_inventory(inventory)[0], "r-i0/trainer")  # island mixed's launch (prefix r-i0)
    _hold(leases, load_inventory(inventory)[1], "other-run/trainer")
    assert _add(run_cli, samplers, inventory, "--dry-run") == 0
    captured = capsys.readouterr()
    assert "island north takes nodes b300-2, b300-3" in captured.out
    assert "left out: node b300-0 is island mixed's (its launch holds the lease)" in captured.err
    assert store.read()["generation"] == 1


def test_an_inventory_path_is_kept_absolute_whatever_the_cwd(fleet_store, wired, tmp_path, monkeypatch,  # noqa: F811
                                                             capsys):
    """A relative --inventory is resolved when the island is written: a later add from
    another directory still reads the first island's nodes."""
    cfg, _objects, store, run_cli, samplers = fleet_store
    (here := tmp_path / "a").mkdir()
    (there := tmp_path / "b").mkdir()
    _inventory(here / "north.yaml", ["b300-4", "b300-5"])
    _inventory(there / "south.yaml", ["b300-5", "b300-6"])
    monkeypatch.chdir(here)
    assert _add(run_cli, samplers, "north.yaml") == 0
    north = next(m for m in store.read()["members"] if m.get("name") == "north")
    assert ij.inventory_of(north["argv"]) == str((here / "north.yaml").resolve())
    monkeypatch.chdir(there)
    other = samplers.parent / "south.samplers.json"
    other.write_text(json.dumps([dict(cfg["islands"][1]["samplers"][0], name="sglang-south")]))
    capsys.readouterr()
    assert _add(run_cli, other, "south.yaml", name="south") == 2
    assert "node b300-5 is island north's" in capsys.readouterr().err


def test_two_adds_at_once_on_overlapping_inventories_one_is_refused_at_its_write(fleet_store, wired, tmp_path,  # noqa: F811
                                                                                 monkeypatch, capsys):
    """South checks against the desired state without north; north is written meanwhile and
    takes south's learner id: south's write checks again on the new state (north's nodes) and
    under its new prefix (north's lease, no longer south's own), and is refused."""
    cfg, _objects, store, run_cli, samplers = fleet_store
    leases = str(tmp_path / "leases")
    north_inv = _inventory(tmp_path / "north.yaml", ["b300-4", "b300-5"])
    south_inv = _inventory(tmp_path / "south.yaml", ["b300-5", "b300-6"])
    south_samplers = samplers.parent / "south.samplers.json"
    south_samplers.write_text(json.dumps([dict(cfg["islands"][1]["samplers"][0], name="sglang-south")]))
    from yeto.launch import learner_ids

    real = learner_ids.allocate_learner_id
    meanwhile = []

    def allocate(objects, **kw):
        if kw.get("island") == "south" and not meanwhile:
            meanwhile.append(_add(run_cli, samplers, north_inv))  # north's add runs between south's check and write
        return real(objects, **kw)

    monkeypatch.setattr(learner_ids, "allocate_learner_id", allocate)
    assert _add(run_cli, south_samplers, south_inv, name="south") == 3 and meanwhile == [0]
    err = capsys.readouterr().err
    assert "island south's nodes cannot join (checked again on the desired state now): node b300-5 is island north's" in err
    trainers = {m["name"]: m for m in store.read()["members"] if m.get("kind") == "trainer"}
    assert "south" not in trainers and trainers["north"]["learner_id"] == 2  # north took the id south checked under


def test_an_id_claimed_meanwhile_checks_the_leases_again_under_the_new_prefix(fleet_store, wired, tmp_path,  # noqa: F811
                                                                             monkeypatch, capsys):
    """South's check runs under prefix r-i2 (a lease of r-i2 is its own); north takes id 2
    meanwhile, so south is written as r-i3: that lease is north's now, and south is refused."""
    cfg, _objects, store, run_cli, samplers = fleet_store
    leases = str(tmp_path / "leases")
    shared = _inventory(tmp_path / "shared.yaml", ["b300-7", "b300-8"], lease_store=leases)
    _hold(leases, load_inventory(shared)[0], "r-i2/trainer")
    south_samplers = samplers.parent / "south.samplers.json"
    south_samplers.write_text(json.dumps([dict(cfg["islands"][1]["samplers"][0], name="sglang-south")]))
    north = _inventory(tmp_path / "north.yaml", ["b300-9", "b300-10"])
    from yeto.launch import learner_ids

    real, meanwhile = learner_ids.allocate_learner_id, []

    def allocate(objects, **kw):
        if kw.get("island") == "south" and not meanwhile:
            meanwhile.append(_add(run_cli, samplers, north))
        return real(objects, **kw)

    monkeypatch.setattr(learner_ids, "allocate_learner_id", allocate)
    assert _add(run_cli, south_samplers, shared, name="south") == 3 and meanwhile == [0]
    err = capsys.readouterr().err
    assert ("island south's nodes cannot join (checked again on the desired state now): node b300-7 is island "
            "north's (its launch holds the lease)") in err
    assert "south" not in {m["name"] for m in store.read()["members"]}


@pytest.mark.parametrize("north_store", [None, "l1"])
def test_an_island_on_another_lease_store_or_none_keeps_its_nodes(fleet_store, wired, tmp_path, capsys,  # noqa: F811
                                                                   north_store):
    """North's inventory has no lease_store (or another one): its nodes are not in south's
    lease store, so south's shared inventory leaves them out by the inventory, not by leases."""
    cfg, _objects, _store, run_cli, samplers = fleet_store
    south_samplers = samplers.parent / "south.samplers.json"
    south_samplers.write_text(json.dumps([dict(cfg["islands"][1]["samplers"][0], name="sglang-south")]))
    north = _inventory(tmp_path / "north.yaml", ["b300-4", "b300-5"],
                       lease_store=north_store and str(tmp_path / north_store))
    assert _add(run_cli, samplers, north) == 0
    south = _inventory(tmp_path / "south.yaml", ["b300-4", "b300-5", "b300-6"], lease_store=str(tmp_path / "l2"))
    capsys.readouterr()
    assert _add(run_cli, south_samplers, south, name="south") == 2
    assert "node b300-4 is island north's; node b300-5 is island north's" in capsys.readouterr().err


def test_islands_on_one_lease_store_are_judged_by_their_leases(fleet_store, wired, tmp_path):  # noqa: F811
    """North (not launched yet) holds no lease: south may list its nodes; the launches then
    wait for each other's leases (docs/OWN_CLUSTERS.md: the window of two adds)."""
    cfg, _objects, _store, run_cli, samplers = fleet_store
    south_samplers = samplers.parent / "south.samplers.json"
    south_samplers.write_text(json.dumps([dict(cfg["islands"][1]["samplers"][0], name="sglang-south")]))
    store = str(tmp_path / "l2")
    assert _add(run_cli, samplers, _inventory(tmp_path / "north.yaml", ["b300-4", "b300-5"], lease_store=store)) == 0
    south = _inventory(tmp_path / "south.yaml", ["b300-4", "b300-5", "b300-6"], lease_store=store)
    assert _add(run_cli, south_samplers, south, name="south") == 0


def test_an_island_on_one_lease_store_naming_the_machines_otherwise_is_refused(fleet_store, wired, tmp_path,  # noqa: F811
                                                                                capsys):
    """North and south share one lease store (written ``/x`` and ``file:///x``); north's
    inventory names b300-4 and b300-5 ``n-a`` and ``n-b``: their leases do not keep south off."""
    cfg, _objects, _store, run_cli, samplers = fleet_store
    south_samplers = samplers.parent / "south.samplers.json"
    south_samplers.write_text(json.dumps([dict(cfg["islands"][1]["samplers"][0], name="sglang-south")]))
    store = str(tmp_path / "l")
    north = _inventory(tmp_path / "north.yaml", ["n-a", "n-b"], lease_store=store,
                       hosts={"n-a": "root@b300-4.example", "n-b": "root@b300-5.example"},
                       addresses={"n-a": "198.51.100.14", "n-b": "198.51.100.15"})
    assert _add(run_cli, samplers, north) == 0
    south = _inventory(tmp_path / "south.yaml", ["b300-6", "b300-7", "b300-4", "b300-5"], lease_store="file://" + store)
    capsys.readouterr()
    assert _add(run_cli, south_samplers, south, name="south") == 2
    err = capsys.readouterr().err
    assert "launch would take b300-4, island north's (its node n-a, the same machine): give the machine one name" in err


def test_move_to_ssh_runs_the_same_join_checks(fleet_store, wired, tmp_path, capsys):  # noqa: F811
    _cfg, _objects, store, run_cli, samplers = fleet_store

    def move(inventory, *extra):
        return run_cli("move", "--name", "sglang", "--gpu", GPU, "--usd-per-h", "0", "--inventory", str(inventory),
                       *extra)

    three = _inventory(tmp_path / "three.yaml", ["b300-1", "b300-2", "b300-3"])
    assert move(three) == 2 and "the island needs 2 trainer node(s); its inventory lists 3" in capsys.readouterr().err
    wired.out["b300-2"] = XID_79
    two = _inventory(tmp_path / "two.yaml", ["b300-1", "b300-2"])
    assert move(two) == 2 and "node b300-2 is not healthy: GPU 5 XID 79" in capsys.readouterr().err
    split = _inventory(tmp_path / "split.yaml", ["b300-3", "b300-4"], groups={"b300-3": "a", "b300-4": "b"})
    assert move(split) == 2 and "fabric: the island needs 2 healthy nodes" in capsys.readouterr().err
    leases = str(tmp_path / "leases")
    held = _inventory(tmp_path / "held.yaml", ["b300-5", "b300-6"], lease_store=leases)
    _hold(leases, load_inventory(held)[0], "other-run/trainer")
    assert move(held) == 2 and "node b300-5 is leased to run other-run" in capsys.readouterr().err
    assert store.read()["generation"] == 1
    assert _add(run_cli, samplers, _inventory(tmp_path / "north.yaml", ["b300-7", "b300-8"])) == 0
    onto = _inventory(tmp_path / "onto.yaml", ["b300-8", "b300-9"])
    assert move(onto) == 2 and "node b300-8 is island north's" in capsys.readouterr().err
    ok = _inventory(tmp_path / "ok.yaml", ["b300-10", "b300-11"])
    assert move(ok, "--dry-run") == 0 and "island sglang takes nodes b300-10, b300-11" in capsys.readouterr().out
    assert move(ok) == 0
    sglang = next(m for m in store.read()["members"] if m.get("name") == "sglang")
    assert sglang["gpu"] == GPU and ij.inventory_of(sglang["argv"]) == str(ok.resolve())
    assert ij.without_inventory(sglang["argv"]) == [a for a in sglang["argv"][:-2]]
    # its own nodes (its inventory now) are not another island's: a move onto them again passes
    assert move(ok) == 0


def test_an_inventory_is_for_an_ssh_island_only(fleet_store, tmp_path, capsys):  # noqa: F811
    _cfg, _objects, _store, run_cli, samplers = fleet_store
    inventory = _inventory(tmp_path / "north.yaml", ["b300-4", "b300-5"])
    assert run_cli("add", "--name", "north", "--gpu", "modal:1x2xh100", "--usd-per-h", "8", "--samplers",
                   str(samplers), "--inventory", str(inventory)) == 2
    assert "--inventory is the node inventory of an ssh: island" in capsys.readouterr().err
    assert run_cli("move", "--name", "sglang", "--gpu", "modal:1x2xh100", "--inventory", str(inventory)) == 2
    assert "--inventory is the node inventory of an ssh: island" in capsys.readouterr().err


def test_min_ib_gbps_is_a_positive_number_of_the_inventory(tmp_path):
    from yeto.launch.node_inventory import InventoryError

    assert {n.min_ib_gbps for n in _nodes(tmp_path, min_ib=400)} == {400.0}
    assert {n.min_ib_gbps for n in _nodes(tmp_path)} == {None}
    with pytest.raises(InventoryError, match="min_ib_gbps must be a positive number"):
        _nodes(tmp_path, min_ib="fast")


def test_join_prev_counts_every_xid_of_this_boot(tmp_path):
    """The previous check a joining node lacks: a cursor of no boot (every XID of this boot is new)."""
    checker = node_health.checker_for("cuda", expect_gpus=8)
    assert checker.parse(XID_79, [], None).ok_single is True  # the checker alone: history
    assert checker.parse(XID_79, [], dict(ij.JOIN_PREV)).ok_single is False
    assert FABRIC_OK in node_text()
