"""The join checks of an ssh: trainer island added at run time (own-cluster-fleet-stability
9.3): its inventory's trainer nodes must be free in this fleet, healthy (node health), in one
fabric group (fabric_groups.check_placement) and not leased to another run (NodeLeases);
any one of them failing refuses ``yeto fleet island add`` with its reason, writing nothing.

Node health and leases are the real modules (``yeto.hw.node_health``, ``yeto.launch.node_lease``):
most tests answer through a stand-in checker (a real :class:`~yeto.hw.node_health.Checker` whose
script prints the node's name) and stand-in holders in the real holder format, and some use the
real checker of a family without one and real :class:`~yeto.launch.node_lease.NodeLeases` on a
directory store. The fabric placement check is a stand-in knob in most tests and the real
``fabric_groups.check_placement`` in one."""

from __future__ import annotations

import json
import subprocess

import pytest
from test_fleet_island_changes import fleet_store  # noqa: F401 - fixture

from yeto.hw import node_health
from yeto.hw.node_health import HealthResult
from yeto.launch import island_join as ij
from yeto.launch.learner_ids import claim_key
from yeto.launch.node_inventory import load_inventory
from yeto.launch.node_lease import NodeLeases
from yeto.rl import fleet_islands

GPU = "ssh:2x8xb300@bm"


class Checker(node_health.Checker):
    """Answers per node (the script prints the node's name here): ``bad`` nodes fail with
    their (GPU) reason; ``ib_down`` nodes are healthy alone but not for a multi-node island."""

    family = "cuda"

    def __init__(self, bad=None, ib_down=(), checked=True):
        self.bad, self.ib_down, self.checked = dict(bad or {}), set(ib_down), checked
        self.scripts: list[list[str]] = []

    def script(self, hcas):
        self.scripts.append(list(hcas))
        return "health-probe"

    def parse(self, text, hcas, prev):
        name = text.strip()
        if not self.checked:
            return HealthResult(False, None, None, ["not checked"])
        if name in self.bad:
            return HealthResult(True, False, False, [self.bad[name]], {"gpu_reasons": [self.bad[name]]})
        if name in self.ib_down:
            return HealthResult(True, True, False, ["mlx5_3 DOWN"], {"gpu_reasons": [], "ib_reasons": ["mlx5_3 DOWN"]})
        return HealthResult(True, True, True, [], {"gpu_reasons": []})


class Ssh:
    def __init__(self, unreachable=()):
        self.unreachable = set(unreachable)
        self.calls: list[str] = []

    def run(self, node, script, *, timeout):
        self.calls.append(node.name)
        if node.name in self.unreachable:
            return subprocess.CompletedProcess([], 255, stdout="", stderr="ssh: connect to host: timed out\n")
        return subprocess.CompletedProcess([], 0, stdout=node.name + "\n", stderr="")


class Leases:
    """Who holds each node, as NodeLeases.holder answers: a holder id ``<run>/trainer@<identity>``."""

    made: list[tuple[str, str]] = []

    def __init__(self, store_uri, run_id, holders=None):
        Leases.made.append((store_uri, run_id))
        self.holders = dict(holders or {})

    def holder(self, node):
        run = self.holders.get(getattr(node, "name", node))
        return None if run is None else f"{run}/trainer@host-{run}"


@pytest.fixture
def stand_ins(monkeypatch):
    """The three checks' modules, as the contract has them; returns their knobs."""
    knobs = {"checker": Checker(), "placement": (True, "", False), "holders": {}, "placed": [], "inventories": []}
    monkeypatch.setattr(ij, "_checker_for", lambda node: knobs["checker"])

    def placement(nodes, need, inventory):
        knobs["placed"].append(([n.name for n in nodes], need))
        knobs["inventories"].append(None if inventory is None else [n.name for n in inventory])
        return knobs["placement"]

    monkeypatch.setattr(ij, "_check_placement", placement)
    monkeypatch.setattr(ij, "_node_leases", lambda uri, run: Leases(uri, run, knobs["holders"]))
    return knobs


def _inventory(path, names, *, env=None, groups=None):
    path.write_text("nodes:\n" + "".join(
        f"  - {{name: {name}, host: ops@{name}.example, address: 198.51.100.{10 + i}, interface: bond0, "
        "device_family: cuda, cpu_arch: x86_64, card: B300, cards_per_node: 8, hbm_gib: 288, "
        "host_memory_gib: 2048, fabric: ib, runtime: docker, docker_args: [--gpus, all]"
        + (f", env: {{NCCL_IB_HCA: '{env}'}}" if env else "")
        + (f", fabric_group: {groups[name]}" if groups and name in groups else "") + "}\n"
        for i, name in enumerate(names)))
    return path


def _nodes(tmp_path, names=("b300-4", "b300-5"), **kw):
    return load_inventory(_inventory(tmp_path / "inv.yaml", names, **kw))


def test_healthy_free_nodes_of_one_group_join(tmp_path, stand_ins):
    nodes = _nodes(tmp_path, env="mlx5_0,mlx5_1")
    out = ij.check(nodes, need=2, run_id="r", lease_store="s3://b/leases", ssh=Ssh())
    assert out.ok and out.refusals == [] and out.notes == []
    assert stand_ins["checker"].scripts == [["mlx5_0", "mlx5_1"]] * 2  # a multi-node island checks its HCAs
    assert stand_ins["placed"] == [(["b300-4", "b300-5"], 2)] and Leases.made[-1] == ("s3://b/leases", "r")


def test_an_unhealthy_node_refuses_the_island_naming_why(tmp_path, stand_ins):
    stand_ins["checker"] = Checker(bad={"b300-5": "Xid 79 since the last check"})
    out = ij.check(_nodes(tmp_path), need=2, run_id="r", ssh=Ssh())
    assert out.refusals == ["node b300-5 is not healthy: Xid 79 since the last check"]


def test_a_port_down_refuses_a_multi_node_island_only(tmp_path, stand_ins):
    stand_ins["checker"] = Checker(ib_down={"b300-4"})
    assert ij.check(_nodes(tmp_path), need=2, run_id="r", ssh=Ssh()).refusals == [
        "node b300-4 is not healthy: mlx5_3 DOWN"]
    one = _nodes(tmp_path, names=("b300-4",))
    assert ij.check(one, need=1, run_id="r", ssh=Ssh()).ok
    assert stand_ins["checker"].scripts[-1] == []  # alone: the GPU checks only


def test_a_single_node_island_is_refused_for_its_gpus_only(tmp_path, stand_ins):
    class Both(Checker):
        def parse(self, text, hcas, prev):
            return HealthResult(True, False, False, ["GPU count 7, expected 8", "mlx5_3 DOWN"],
                                {"gpu_reasons": ["GPU count 7, expected 8"], "ib_reasons": ["mlx5_3 DOWN"]})

    stand_ins["checker"] = Both()
    one = _nodes(tmp_path, names=("b300-4",))
    assert ij.check(one, need=1, run_id="r", ssh=Ssh()).refusals == [
        "node b300-4 is not healthy: GPU count 7, expected 8"]
    assert ij.check(_nodes(tmp_path), need=2, run_id="r", ssh=Ssh()).refusals[0] == (
        "node b300-4 is not healthy: GPU count 7, expected 8; mlx5_3 DOWN")


def test_a_node_with_its_health_check_off_is_noted_and_not_called(tmp_path, stand_ins):
    import dataclasses

    stand_ins["checker"] = Checker(bad={"b300-4": "Xid 79"})
    first, second = _nodes(tmp_path)
    ssh = Ssh()
    out = ij.check([dataclasses.replace(first, node_health="off"), second], need=2, run_id="r", ssh=ssh)
    assert out.ok and ssh.calls == ["b300-5"]
    assert "node b300-4: node_health off in the inventory: its health is not checked" in out.notes

def test_a_node_that_does_not_answer_is_refused(tmp_path, stand_ins):
    out = ij.check(_nodes(tmp_path), need=2, run_id="r", ssh=Ssh(unreachable={"b300-4"}))
    assert out.refusals == [("node b300-4: its health check did not answer (ssh exit 255: ssh: connect to host: "
                            "timed out)")]


def test_a_family_without_a_checker_is_noted_not_refused(tmp_path, stand_ins):
    stand_ins["checker"] = Checker(checked=False)
    out = ij.check(_nodes(tmp_path), need=2, run_id="r", ssh=Ssh())
    assert out.ok and any("no health check for device family cuda" in n for n in out.notes)


def test_nodes_across_fabric_groups_refuse_the_island(tmp_path, stand_ins):
    stand_ins["placement"] = (False, ("the island needs 2 healthy nodes of one fabric group; group a has only 1, "
                                     "group b has only 1"), False)
    out = ij.check(_nodes(tmp_path), need=2, run_id="r", ssh=Ssh())
    assert out.refusals == [("fabric: the island needs 2 healthy nodes of one fabric group; group a has only 1, "
                            "group b has only 1")]


def test_the_real_placement_check_refuses_nodes_of_two_fabric_groups(tmp_path, monkeypatch):
    """fabric_groups.check_placement itself, on the island's nodes and its whole inventory."""
    monkeypatch.setattr(ij, "_checker_for", lambda node: Checker())

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

def test_an_inventory_without_groups_is_only_a_note(tmp_path, stand_ins):
    stand_ins["placement"] = (True, "no fabric_group in the inventory: run yeto cluster probe --write", True)
    out = ij.check(_nodes(tmp_path), need=2, run_id="r", ssh=Ssh())
    assert out.ok and out.notes[0].startswith("fabric: no fabric_group")


def test_a_node_leased_to_another_run_is_refused_and_this_island_s_own_is_free(tmp_path, stand_ins):
    stand_ins["holders"] = {"b300-4": "other-run", "b300-5": "r"}
    out = ij.check(_nodes(tmp_path), need=2, run_id="r", lease_store="s3://b/leases", ssh=Ssh())
    assert out.refusals == ["node b300-4 is leased to run other-run"]
    stand_ins["holders"] = {"b300-4": "r-m1", "b300-5": "r-i0"}  # its own moved launch; another island
    out = ij.check(_nodes(tmp_path), need=2, run_id="r", lease_store="s3://b/leases", ssh=Ssh())
    assert out.refusals == ["node b300-5 is leased to run r-i0"]
    out = ij.check(_nodes(tmp_path), need=2, run_id="r", ssh=Ssh())  # no lease_store: noted
    assert out.ok and any("no lease_store" in n for n in out.notes)


def test_the_real_leases_refuse_a_node_another_run_holds(tmp_path):
    """NodeLeases itself on a directory lease store: another run's live lease refuses the node,
    the island's own launch's lease does not."""
    store = str(tmp_path / "leases")
    nodes = _nodes(tmp_path)
    assert NodeLeases(store, "other-run/trainer", identity="a").acquire(nodes[0]).ok
    assert NodeLeases(store, "r-i2/trainer", identity="b").acquire(nodes[1]).ok
    out = ij.JoinCheck()
    ij.check_leases(nodes, lease_store=store, run_id="r-i2", out=out)
    assert out.refusals == ["node b300-4 is leased to run other-run"]


def test_the_real_checker_of_a_family_without_one_runs_nothing(tmp_path):
    import dataclasses

    (node,) = _nodes(tmp_path, names=("b300-4",))
    checker = ij._checker_for(node)  # cuda: the real checker, built with the node's card count
    assert checker.enabled and checker.expect_gpus == 8
    other = dataclasses.replace(node, device_family="no-such-family")
    ssh, out = Ssh(), ij.JoinCheck()
    ij.check_health([other], multi=False, ssh=ssh, out=out)
    assert ssh.calls == [] and out.ok and out.notes == ["node b300-4: no health check for device family no-such-family"]

def test_a_node_of_another_island_or_the_wrong_count_is_refused(tmp_path, stand_ins):
    nodes = _nodes(tmp_path)
    out = ij.check(nodes, need=2, run_id="r", taken={"ops@b300-5.example": "sglang"}, ssh=Ssh())
    assert out.refusals == ["node b300-5 is island sglang's"]
    out = ij.check(nodes, need=3, run_id="r", ssh=Ssh())
    assert out.refusals == ["the island needs 3 trainer node(s); its inventory lists 2 ['b300-4', 'b300-5']"]


def test_checks_whose_modules_are_missing_refuse(tmp_path, monkeypatch):
    def missing(*a, **kw):
        raise ImportError("No module named 'yeto.hw.node_health'")

    for name in ("_checker_for", "_check_placement", "_node_leases"):
        monkeypatch.setattr(ij, name, missing)
    out = ij.check(_nodes(tmp_path), need=2, run_id="r", lease_store="s3://b/l", ssh=Ssh())
    assert len(out.refusals) == 4 and all("ImportError" in r for r in out.refusals)


# -- yeto fleet island add --------------------------------------------------------------------------

def _add(run_cli, samplers, inventory, *extra):
    return run_cli("add", "--name", "north", "--gpu", GPU, "--usd-per-h", "0", "--samplers", str(samplers),
                   "--inventory", str(inventory), *extra)


@pytest.mark.parametrize("problem", ["health", "fabric", "lease"])
def test_add_refuses_an_ssh_island_failing_one_check_and_writes_nothing(fleet_store, stand_ins, tmp_path,  # noqa: F811
                                                                         monkeypatch, capsys, problem):
    _cfg, objects, store, run_cli, samplers = fleet_store
    inventory = _inventory(tmp_path / "north.yaml", ["b300-4", "b300-5"])
    monkeypatch.setattr("yeto.launch.ssh_trainer_island.SshOps", Ssh)
    if problem == "health":
        stand_ins["checker"] = Checker(bad={"b300-4": "GPU count 7, expected 8"})
        why = "node b300-4 is not healthy: GPU count 7, expected 8"
    elif problem == "fabric":
        stand_ins["placement"] = (False, "group a has only 1, group b has only 1", False)
        why = "fabric: group a has only 1, group b has only 1"
    else:
        inventory.write_text("lease_store: s3://b/leases\n" + inventory.read_text())
        stand_ins["holders"] = {"b300-5": "another-run"}
        why = "node b300-5 is leased to run another-run"
    monkeypatch.setattr(fleet_islands, "launch_dry_run", lambda *a, **kw: pytest.fail("checked before the dry run"))
    for dry in ((), ("--dry-run",)):
        assert _add(run_cli, samplers, inventory, *dry) == 2
        err = capsys.readouterr().err
        assert "refused: island north's nodes cannot join: " + why in err
    assert store.read()["generation"] == 1 and objects.read(claim_key(2)) is None


def test_add_takes_an_ssh_island_whose_nodes_pass(fleet_store, stand_ins, tmp_path, monkeypatch, capsys):  # noqa: F811
    _cfg, objects, store, run_cli, samplers = fleet_store
    inventory = _inventory(tmp_path / "north.yaml", ["b300-4", "b300-5"])
    monkeypatch.setattr("yeto.launch.ssh_trainer_island.SshOps", Ssh)
    dry_runs = []
    monkeypatch.setattr(fleet_islands, "launch_dry_run", lambda member, **kw: dry_runs.append(member))
    monkeypatch.setattr(fleet_islands, "session_check", lambda *a, **kw: None)
    assert _add(run_cli, samplers, inventory) == 0
    out = capsys.readouterr()
    assert "island north's nodes pass the join checks" in out.out
    assert stand_ins["inventories"] == [["b300-4", "b300-5"]]  # placed against its whole inventory
    assert "note: island north: no lease_store" in out.err
    (member,) = dry_runs
    assert member["argv"][-2:] == ["--rl-node-inventory", str(inventory)] and member["gpu"] == GPU
    assert store.read()["generation"] == 2 and objects.read(claim_key(2)) is not None


def test_a_second_ssh_island_on_the_nodes_of_the_first_is_refused(fleet_store, stand_ins, tmp_path,  # noqa: F811
                                                                    monkeypatch, capsys):
    cfg, _objects, _store, run_cli, samplers = fleet_store
    monkeypatch.setattr("yeto.launch.ssh_trainer_island.SshOps", Ssh)
    monkeypatch.setattr(fleet_islands, "launch_dry_run", lambda member, **kw: None)
    monkeypatch.setattr(fleet_islands, "session_check", lambda *a, **kw: None)
    assert _add(run_cli, samplers, _inventory(tmp_path / "north.yaml", ["b300-4", "b300-5"])) == 0
    other = samplers.parent / "south.samplers.json"
    other.write_text(json.dumps([dict(cfg["islands"][1]["samplers"][0], name="sglang-south")]))
    south = _inventory(tmp_path / "south.yaml", ["b300-5", "b300-6"])
    assert run_cli("add", "--name", "south", "--gpu", GPU, "--usd-per-h", "0", "--samplers", str(other),
                   "--inventory", str(south)) == 2
    assert "node b300-5 is island north's" in capsys.readouterr().err


def test_an_inventory_is_for_an_ssh_island_only(fleet_store, tmp_path, capsys):  # noqa: F811
    _cfg, _objects, _store, run_cli, samplers = fleet_store
    inventory = _inventory(tmp_path / "north.yaml", ["b300-4", "b300-5"])
    assert run_cli("add", "--name", "north", "--gpu", "modal:1x2xh100", "--usd-per-h", "8", "--samplers",
                   str(samplers), "--inventory", str(inventory)) == 2
    assert "--inventory is the node inventory of an ssh: island" in capsys.readouterr().err
