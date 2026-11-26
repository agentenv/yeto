"""The checks of an ssh pool on own nodes: one machine per inventory address, the sampler
role of a named node, the tensor parallel size against the pool's cards, the release
warning of a pool that bills nothing, where the sampler reads its prompts (and the
``prompts_file`` upload at every start), and the cross-pool / trainer node refusals of
``yeto run up`` (:mod:`yeto.launch.own_node_checks`)."""

from __future__ import annotations

import hashlib
import json
import re
from types import SimpleNamespace

import pytest

from tests.test_ssh_sampler_pool import FakeRun, conf, npu_node, pool_cfg, provider_section
from yeto.launch import own_node_checks as checks
from yeto.launch import ssh_sampler_pool as sp
from yeto.launch.nc import cli, pool_run
from yeto.launch.node_inventory import InventoryError, parse_inventory, parse_node


def _raw(name, host, **kw):
    return {"name": name, "host": host, "device_family": "cuda", "cpu_arch": "x86_64", "card": "H100",
            "cards_per_node": 8, "hbm_gib": 80, "host_memory_gib": 1024, **kw}


# -- 6: one address per node --------------------------------------------------------


def test_the_inventory_refuses_one_address_on_two_nodes():
    with pytest.raises(InventoryError, match=r"address '10\.0\.0\.5' twice \(nodes a and b\)"):
        parse_inventory({"nodes": [_raw("a", "a", address="10.0.0.5"), _raw("b", "b", address="10.0.0.5")]})


def test_nodes_without_an_address_or_on_loopback_are_no_duplicates():
    assert len(parse_inventory({"nodes": [_raw("a", "a"), _raw("b", "b")]})) == 2
    # each container's own loopback (the stage-1 Modal mini inventory)
    assert len(parse_inventory({"nodes": [_raw("a", "a", address="127.0.0.1"),
                                          _raw("b", "b", address="127.0.0.1")]})) == 2
    assert len(parse_inventory({"nodes": [_raw("a", "a", address="10.0.0.5"),
                                          _raw("b", "b", address="10.0.0.6")]})) == 2


# -- 3: a named node must have the sampler role ---------------------------------------


@pytest.mark.parametrize("roles", [["trainer"], ["trainer-spare"]])
def test_a_named_node_without_the_sampler_role_is_refused(roles):
    with pytest.raises(ValueError, match=re.escape(f"nodes node-a (roles: {roles[0]}) have no sampler role")):
        sp.parse_config({"provider": provider_section()}, inventory=[npu_node(roles=roles)])


@pytest.mark.parametrize("roles", [[], ["sampler"], ["trainer", "sampler"]])
def test_a_named_node_with_the_sampler_role_or_no_roles_is_taken(roles):
    c = sp.parse_config({"provider": provider_section()}, inventory=[npu_node(roles=roles)])
    assert [n.name for n in c.nodes] == ["node-a"]


# -- 4: tensor parallel size against the pool's cards ---------------------------------


def _spec(**options):
    return {"engine": {"adapter": "vllm-server", "options": options}}


def test_a_tensor_parallel_size_above_the_pool_devices_is_refused():
    (problem,) = sp.spec_node_problems(conf(devices=[0, 1]), _spec(tp=4))
    assert problem == ("ssh provider: node node-a: the spec's engine tensor parallel size is 4 cards, provider.devices "
                       "[0, 1] give it 2; lower engine.options.tp or give the pool more cards of the node")
    assert sp.spec_node_problems(conf(devices=[0, 1, 2, 3]), _spec(tp=4)) == []


def test_without_devices_the_tensor_parallel_size_is_held_to_the_node_cards():
    (problem,) = sp.spec_node_problems(conf(devices=None), _spec(tp=16))
    assert "tensor parallel size is 16 cards, it has 8;" in problem
    assert sp.spec_node_problems(conf(devices=None), _spec(tp=8)) == []


def test_the_adapter_default_tensor_parallel_size_is_checked():
    assert sp.spec_tp({"engine": {"adapter": "vllm-server"}}) == 4
    assert sp.spec_tp({"engine": {"adapter": "sglang-server"}}) == 8
    assert sp.spec_tp({"engine": {"adapter": "fake"}}) is None
    assert sp.spec_tp(_spec(tp="eight")) is None  # the sampler's own checks report it
    (problem,) = sp.spec_node_problems(conf(devices=[0]), {"engine": {"adapter": "vllm-server"}})
    assert "tensor parallel size is 4 cards" in problem


# -- 2: a pool that bills nothing -----------------------------------------------------


def test_an_own_ssh_pool_at_no_price_is_not_warned_as_billing(tmp_path):
    text = pool_run.lease_warning(pool_cfg(tmp_path), "pool.json")
    assert "bill until" not in text and "not billed" in text
    assert text == ("Own SSH nodes (ssh, price_usd_h 0) are not billed by this pool, but its samplers hold their "
                    "cards until the pool stops: keep this pool running, or stop it with "
                    "`python -m yeto.launch.nc pool-stop --config pool.json`.")


def test_an_own_ssh_pool_with_a_price_keeps_the_billing_warning(tmp_path):
    text = pool_run.lease_warning(pool_cfg(tmp_path, price_usd_h=2.5), "pool.json")
    assert text == pool_run.CLOUD_WARNING.format(kind="ssh").replace("<file>", "pool.json")
    sky = {"provider": {"kind": "sky"}, "shard_base": "x"}
    assert "bill until they are released" in pool_run.lease_warning(sky, "pool.json")


# -- 7: where an ssh pool's sampler reads its prompts --------------------------------


def _inventory_cfg(tmp_path, *, runtime="container", prompts_path, prompts_file=None, **section):
    inv = tmp_path / "inv.yaml"
    inv.write_text(json.dumps({"nodes": [_raw("own-0", "ops@own-0", mounts={"/mnt/nvme/models": "/data/models"},
                                              roles=["sampler"])]}))
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"prompts": {"path": prompts_path}}))
    provider = {"kind": "ssh", "inventory": str(inv), "workdir": "/srv/yeto/run0", "runtime": runtime, **section}
    if runtime == "container":
        provider["container"] = {"engine": "docker", "image": "img@sha256:" + "0" * 64}
    else:
        provider["venv"] = {"python": "/opt/venv/bin/python"}
    cfg = {"provider": provider, "spec": str(spec), "shard_base": "r", "run_prefix": "r", "target": 1, "slots": [0]}
    if prompts_file is not None:
        cfg["prompts_file"] = str(prompts_file)
    return cfg, json.loads(spec.read_text())


def test_prompts_under_a_mount_are_read_from_the_host_path_behind_it(tmp_path):
    cfg, spec = _inventory_cfg(tmp_path, prompts_path="/data/models/prompts/p.jsonl")
    lines, problems, (note,) = cli.prompts_file_report(cfg, spec)
    assert lines == problems == []
    assert note == ("the sampler spec reads prompts.path /data/models/prompts/p.jsonl from the node (an inventory or "
                    "pool mount; on the host /mnt/nvme/models/prompts/p.jsonl): put the file there before the run, "
                    "or set prompts_file (uploaded and checked by size and sha256 at every start)")
    assert "fails" not in note


def test_prompts_under_the_workdir_are_read_from_the_workdir(tmp_path):
    cfg, spec = _inventory_cfg(tmp_path, prompts_path="/srv/yeto/run0/prompts.jsonl")
    lines, problems, (note,) = cli.prompts_file_report(cfg, spec)
    assert lines == problems == []
    assert ("reads prompts.path /srv/yeto/run0/prompts.jsonl from the node (the pool's workdir /srv/yeto/run0; "
            "on the host /srv/yeto/run0/prompts.jsonl)") in note


def test_prompts_neither_under_a_mount_nor_the_workdir_are_a_warning(tmp_path):
    cfg, spec = _inventory_cfg(tmp_path, prompts_path="/opt/prompts.jsonl")
    (line,), problems, notes = cli.prompts_file_report(cfg, spec)
    assert problems == notes == []
    assert line.startswith("WARNING: the sampler spec reads prompts.path /opt/prompts.jsonl, which the sampler "
                           "container on nodes ['own-0'] cannot read (no mount or workdir holds it")
    assert "set prompts_file to upload it" in line


def test_a_venv_sampler_reads_the_host_itself(tmp_path):
    cfg, spec = _inventory_cfg(tmp_path, runtime="venv", prompts_path="/opt/prompts.jsonl")
    lines, problems, (note,) = cli.prompts_file_report(cfg, spec)
    assert lines == problems == [] and "(the node's own filesystem; on the host /opt/prompts.jsonl)" in note


def test_a_prompts_file_goes_to_the_host_path_and_an_unseen_path_is_a_problem(tmp_path):
    local = tmp_path / "p.jsonl"
    local.write_text('{"q": 1}\n')
    cfg, spec = _inventory_cfg(tmp_path, prompts_path="/data/models/p.jsonl", prompts_file=local)
    lines, problems, notes = cli.prompts_file_report(cfg, spec)
    assert problems == notes == [] and lines[-1] == "prompts_file: uploaded to /mnt/nvme/models/p.jsonl on every " \
                                                    "node of the pool"
    cfg, spec = _inventory_cfg(tmp_path, prompts_path="/opt/p.jsonl", prompts_file=local)
    _lines, (problem,), _notes = cli.prompts_file_report(cfg, spec)
    assert "is under no mount or workdir of nodes ['own-0']" in problem


def test_a_rented_pool_keeps_its_prompts_note(tmp_path):
    cfg = {"provider": {"kind": "sky"}, "spec": "x"}
    _lines, _problems, (note,) = cli.prompts_file_report(cfg, {"prompts": {"path": "/work/p.jsonl"}})
    assert "each sampler start fails unless the file is already on the node" in note


# -- 5: the prompts_file upload at every start ----------------------------------------


class Node(FakeRun):
    """FakeRun that keeps the bytes put to a path and answers the size / sha256 check."""

    def __init__(self, *, corrupt=False):
        super().__init__()
        self.files: dict[str, bytes] = {}
        self.corrupt = corrupt

    def __call__(self, argv, **kw):
        cmd = argv[-1] if argv and argv[0] == "ssh" else ""
        if cmd.startswith("umask 077; cat > ") and isinstance(kw.get("input"), bytes):
            path = cmd.split("cat > ", 1)[1].split(".tmp", 1)[0].strip("'")
            self.files[path] = kw["input"] + (b"x" if self.corrupt else b"")
        if cmd.startswith("echo YETO_PROMPTS"):
            path = cmd.split("wc -c < ", 1)[1].split(")", 1)[0].strip("'")
            data = self.files.get(path, b"")
            self.calls.append((argv, kw))
            digest = hashlib.sha256(data).hexdigest()
            return SimpleNamespace(returncode=0, stdout=f"YETO_PROMPTS {len(data)} {digest}\n", stderr="")
        return super().__call__(argv, **kw)


def _ops(tmp_path, monkeypatch, run, *, prompts_path="/srv/yeto/shared/prompts/p.jsonl"):
    class Pop:
        stdout = None

        def __init__(self, *a, **k):
            pass

        def wait(self):
            return 0

    monkeypatch.setattr(sp.subprocess, "Popen", Pop)
    monkeypatch.setattr(cli.tasks, "sampler_spec", lambda spec, sampler_id, shards: dict(spec, sampler_id=sampler_id))
    monkeypatch.setattr(cli.tasks, "apply_sampler_overrides", lambda spec, **kw: spec)
    local = tmp_path / "prompts.jsonl"
    local.write_bytes(b'{"question": "1+1?", "answer": "2"}\n' * 3)
    cfg = pool_cfg(tmp_path, node_runtime=False)
    spec = json.loads((tmp_path / "spec.json").read_text())
    spec["prompts"]["path"] = prompts_path
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    cfg["prompts_file"] = str(local)
    ops = sp.SshSamplerPoolOps(cfg, conf=conf(node_runtime=False), run=run, state_dir=tmp_path)
    return ops, local


def _member(ops):
    return SimpleNamespace(account=0, ticket=ops.order(0), shard="f09-0", data_slot=None, replaces_shard=None,
                           predecessor_lost=False)


def test_every_start_uploads_the_prompts_file_and_checks_it_on_the_node(tmp_path, monkeypatch):
    run = Node()
    ops, local = _ops(tmp_path, monkeypatch, run)
    m = _member(ops)
    ops.start(m, ["f09-0"])
    ops.start(m, ["f09-0"])  # a restart: copied and checked again
    host = "/srv/yeto/shared/prompts/p.jsonl"
    assert run.files[host] == local.read_bytes()
    cmds = [c[0][-1] for c in run.calls if c[0][0] == "ssh"]
    assert sum(c.startswith("echo YETO_PROMPTS") for c in cmds) == 2
    assert f"mkdir -p {'/srv/yeto/shared/prompts'}" in cmds
    first_check = next(i for i, c in enumerate(cmds) if c.startswith("echo YETO_PROMPTS"))
    assert first_check < next(i for i, c in enumerate(cmds) if "run -d" in c)  # checked before the sampler starts


def test_a_copy_that_does_not_match_stops_the_start(tmp_path, monkeypatch):
    run = Node(corrupt=True)
    ops, _local = _ops(tmp_path, monkeypatch, run)
    with pytest.raises(sp.SshSamplerError, match="does not match .* after the upload"):
        ops.start(_member(ops), ["f09-0"])
    assert not any("run -d" in c[0][-1] for c in run.calls if c[0][0] == "ssh")


def test_a_prompts_path_the_sampler_cannot_see_stops_the_start(tmp_path, monkeypatch):
    run = Node()
    ops, _local = _ops(tmp_path, monkeypatch, run, prompts_path="/opt/p.jsonl")
    with pytest.raises(sp.SshSamplerError, match="under no mount or workdir of node-a"):
        ops.start(_member(ops), ["f09-0"])
    assert run.files == {}


def test_without_prompts_file_nothing_is_uploaded(tmp_path, monkeypatch):
    run = Node()
    ops, _local = _ops(tmp_path, monkeypatch, run)
    ops.cfg.pop("prompts_file")
    ops.start(_member(ops), ["f09-0"])
    assert run.files == {} and not any("YETO_PROMPTS" in c[0][-1] for c in run.calls if c[0][0] == "ssh")


# -- 1, 3: pools of one run against each other and the trainer -----------------------


def _pool(tmp_path, name, nodes, *, lease_store=None, inventory_nodes=None, devices=None):
    inv = tmp_path / f"inv-{name}.yaml"
    doc = {"nodes": inventory_nodes or [_raw("own-2", "ops@own-2", address="10.0.0.2", roles=["sampler"])]}
    if lease_store:
        doc["lease_store"] = lease_store
    inv.write_text(json.dumps(doc))
    provider = {"kind": "ssh", "inventory": str(inv), "nodes": nodes, "workdir": f"/srv/yeto/{name}",
                "runtime": "venv", "venv": {"python": "/opt/venv/bin/python"}}
    if devices is not None:
        provider["devices"] = devices
    return name, {"provider": provider}


def test_two_pools_on_one_leased_node_are_refused_naming_both_and_the_ways_out(tmp_path):
    a = _pool(tmp_path, "a", ["own-2"], lease_store="file:///srv/leases", devices=[0, 1, 2, 3])
    b = _pool(tmp_path, "b", ["own-2"], lease_store="file:///srv/leases/", devices=[4, 5, 6, 7])
    (problem,) = checks.pool_problems([a, b])
    assert problem.startswith("pools a and b both start samplers on node own-2, whose inventory keeps node leases "
                              "(lease_store file:///srv/leases)")
    assert "inventory without lease_store and split the node's cards with provider.devices" in problem
    assert "or put the pools on different nodes" in problem


def test_two_pools_share_a_node_without_lease_store_or_use_their_own_nodes(tmp_path):
    a = _pool(tmp_path, "a", ["own-2"], devices=[0, 1, 2, 3])
    b = _pool(tmp_path, "b", ["own-2"], devices=[4, 5, 6, 7])
    assert checks.pool_problems([a, b]) == []
    two = [_raw("own-2", "ops@own-2", roles=["sampler"]), _raw("own-3", "ops@own-3", roles=["sampler"])]
    a = _pool(tmp_path, "a", ["own-2"], lease_store="file:///srv/leases", inventory_nodes=two)
    b = _pool(tmp_path, "b", ["own-3"], lease_store="file:///srv/leases", inventory_nodes=two)
    assert checks.pool_problems([a, b]) == []
    assert checks.pool_problems([a, ("nc", {"provider": {"kind": "nc"}})]) == []


@pytest.mark.parametrize("trainer_raw", [
    _raw("t-0", "ops@own-2", roles=["trainer"]),  # one SSH target
    _raw("t-0", "ops@other", address="10.0.0.2", roles=["trainer"]),  # one cluster address
])
def test_a_pool_on_a_trainer_island_node_is_refused(tmp_path, trainer_raw):
    pool = _pool(tmp_path, "p", ["own-2"])
    trainer = checks.TrainerNode(parse_node(trainer_raw, index=0), "ssh:1x8xh100", str(tmp_path / "t.yaml"))
    (problem,) = checks.pool_problems([pool], [trainer])
    assert problem.startswith("pool p starts samplers on node own-2, which is a node of this run's trainer island "
                              "ssh:1x8xh100 (t-0): the trainer takes every card of its nodes")
    spare = checks.TrainerNode(trainer.node, "ssh:1x8xh100", None, spare=True)
    (problem,) = checks.pool_problems([pool], [spare])
    assert "which is a trainer-spare node of this run's trainer island" in problem
    other = checks.TrainerNode(parse_node(_raw("t-1", "ops@t-1", address="10.0.0.9"), index=0), "ssh:x", None)
    assert checks.pool_problems([pool], [other]) == []


def test_one_name_is_one_machine_only_in_one_inventory_file():
    a, b = parse_node(_raw("n", "h1"), index=0), parse_node(_raw("n", "h2"), index=0)
    assert checks.same_machine(a, "/i.yaml", b, "/i.yaml")
    assert not checks.same_machine(a, "/i.yaml", b, "/j.yaml")
    lo = parse_node(_raw("x", "h3", address="127.0.0.1"), index=0)
    assert not checks.same_machine(lo, None, parse_node(_raw("y", "h4", address="127.0.0.1"), index=0), None)


def test_the_trainer_side_holds_the_island_nodes_and_its_spares(tmp_path):
    inv = tmp_path / "trainers.yaml"
    inv.write_text(json.dumps({"nodes": [_raw("t-0", "ops@t-0", roles=["trainer"]),
                                         _raw("t-1", "ops@t-1", roles=["trainer"]),
                                         _raw("s-0", "ops@s-0", roles=["trainer-spare"]),
                                         _raw("p-0", "ops@p-0", roles=["sampler"])]}))
    side = checks.trainer_nodes([f"ssh:{inv}"], None)
    assert [(t.node.name, t.spare) for t in side] == [("t-0", False), ("t-1", False), ("s-0", True)]
    assert {t.inventory for t in side} == {str(inv.resolve())} and {t.entry for t in side} == {f"ssh:{inv}"}
    assert checks.trainer_nodes(["ssh:/nonexistent/inventory.yaml"], None) == []  # its launch dry run says why
    pool = _pool(tmp_path, "p", ["s-0"], inventory_nodes=[_raw("s-0", "ops@s-0", roles=["sampler"])])
    (problem,) = checks.pool_problems([pool], side)
    assert "which is a trainer-spare node of this run's trainer island" in problem


def test_a_fleet_trainer_side_is_every_node_its_islands_may_take(tmp_path):
    inv = tmp_path / "trainers.yaml"
    inv.write_text(json.dumps({"nodes": [_raw("t-0", "ops@t-0", roles=["trainer"]),
                                         _raw("t-1", "ops@t-1", roles=["trainer"])]}))
    assert [t.node.name for t in checks.trainer_nodes(["ssh:2x8xh100"], str(inv))] == ["t-0", "t-1"]
    # fleet islands sharing one inventory pick their nodes at launch: any trainer node counts
    side = checks.trainer_nodes(["ssh:1x8xh100", "ssh:1x8xh100"], str(inv))  # two islands, one entry each
    assert [t.node.name for t in side] == ["t-0", "t-1"] and side[0].inventory == str(inv.resolve())


def test_a_leased_trainer_inventory_counts_every_node_the_launch_may_take(tmp_path):
    inv = tmp_path / "trainers.yaml"
    inv.write_text(json.dumps({"lease_store": str(tmp_path / "leases"),
                               "nodes": [_raw(f"t-{i}", f"ops@t-{i}", roles=["trainer"]) for i in range(3)]}))
    assert [t.node.name for t in checks.trainer_nodes(["ssh:1x8xh100"], str(inv))] == ["t-0", "t-1", "t-2"]
    inv.write_text(json.dumps({"nodes": [_raw(f"t-{i}", f"ops@t-{i}", roles=["trainer"]) for i in range(2)]}))
    assert [t.node.name for t in checks.trainer_nodes(["ssh:2x8xh100"], str(inv))] == ["t-0", "t-1"]
