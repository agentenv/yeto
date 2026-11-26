"""The checks of an ssh pool on own nodes: one machine per inventory address, the sampler
role of a named node, the tensor parallel size against the pool's cards, the release
warning of a pool that bills nothing, where the sampler reads its prompts (and the
``prompts_file`` upload at every start), and the cross-pool / trainer node refusals of
``yeto run up`` (:mod:`yeto.launch.own_node_checks`)."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
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


def test_devices_beyond_the_node_cards_are_refused_and_not_counted():
    node = npu_node(cards_per_node=4)
    c = sp.parse_config({"provider": provider_section(devices=[0, 1, 2, 3, 4, 5, 6, 7])}, inventory=[node])
    problems = sp.spec_node_problems(c, _spec(tp=8))
    assert problems == ["ssh provider: node node-a has 4 cards (indexes 0-3); provider.devices [4, 5, 6, 7] are "
                        "not on it",
                        "ssh provider: node node-a: the spec's engine tensor parallel size is 8 cards, "
                        "provider.devices [0, 1, 2, 3, 4, 5, 6, 7] give it 4; lower engine.options.tp or give the "
                        "pool more cards of the node"]


def test_the_adapter_default_tensor_parallel_size_is_checked():
    from yeto.rl.adapters.sampler_engines.sglang_server import SGLangOptions
    from yeto.rl.adapters.sampler_engines.vllm_server import VllmOptions

    assert sp.ADAPTER_DEFAULT_TP == {"vllm-server": VllmOptions().tp, "sglang-server": SGLangOptions().tp}
    assert sp.spec_tp({"engine": {"adapter": "vllm-server"}}) == 4
    assert sp.spec_tp({"engine": {"adapter": "sglang-server"}}) == 8
    assert sp.spec_tp({"engine": {"adapter": "fake"}}) is None
    assert sp.spec_tp(_spec(tp="8")) == 8  # the adapters take the decimal string
    with pytest.raises(ValueError, match="engine.options.tp must be a whole number of cards >= 1, not 'eight'"):
        sp.spec_tp(_spec(tp="eight"))
    (problem,) = sp.spec_node_problems(conf(devices=None), _spec(tp="eight"))
    assert problem == "ssh provider: engine.options.tp must be a whole number of cards >= 1, not 'eight'"
    (problem,) = sp.spec_node_problems(conf(devices=[0, 1]), _spec(tp="4"))
    assert "tensor parallel size is 4 cards" in problem
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
    """FakeRun that keeps the bytes streamed to a path and answers the size / sha256 check;
    ``write_rc``: the exit of the write; ``corrupt``: the write keeps other bytes."""

    def __init__(self, *, corrupt=False, write_rc=0):
        super().__init__()
        self.files: dict[str, bytes] = {}
        self.corrupt, self.write_rc = corrupt, write_rc
        self.writes: list[str] = []

    def __call__(self, argv, **kw):
        cmd = argv[-1] if argv and argv[0] == "ssh" else ""
        if cmd.startswith("umask 077; t=") and kw.get("stdin") is not None:
            self.calls.append((argv, kw))
            path = shlex.split(cmd.split("t=", 1)[1].split(".tmp.$$", 1)[0])[0]
            self.writes.append(cmd)
            if self.write_rc == 0:
                self.files[path] = kw["stdin"].read() + (b"x" if self.corrupt else b"")
            return SimpleNamespace(returncode=self.write_rc, stdout=b"", stderr=b"disk full" if self.write_rc else b"")
        if cmd.startswith("echo YETO_PROMPTS"):
            self.calls.append((argv, kw))
            path = shlex.split(cmd.split("wc -c < ", 1)[1].split(")", 1)[0])[0]
            if path not in self.files:
                return SimpleNamespace(returncode=0, stdout="YETO_PROMPTS\n", stderr="")
            data = self.files[path]
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


def _cmds(run):
    return [c[0][-1] for c in run.calls if c[0][0] == "ssh"]


def test_a_start_uploads_the_prompts_file_and_checks_it_before_the_sampler(tmp_path, monkeypatch):
    run = Node()
    ops, local = _ops(tmp_path, monkeypatch, run)
    ops.start(_member(ops), ["f09-0"])
    host = "/srv/yeto/shared/prompts/p.jsonl"
    assert run.files[host] == local.read_bytes() and len(run.writes) == 1
    # a temp name of this shell (two starts on one path do not collide), then renamed
    assert "t='/srv/yeto/shared/prompts/p.jsonl'.tmp.$$" in run.writes[0] or \
        "t=/srv/yeto/shared/prompts/p.jsonl.tmp.$$" in run.writes[0]
    assert 'mv -f "$t" /srv/yeto/shared/prompts/p.jsonl' in run.writes[0]
    cmds = _cmds(run)
    assert "mkdir -p /srv/yeto/shared/prompts" in cmds
    checks_at = [i for i, c in enumerate(cmds) if c.startswith("echo YETO_PROMPTS")]
    assert len(checks_at) == 2  # before (no copy there yet) and after the upload
    assert checks_at[-1] < next(i for i, c in enumerate(cmds) if "run -d" in c)  # checked before the start


def test_a_restart_checks_the_copy_and_keeps_it_when_it_matches(tmp_path, monkeypatch):
    run = Node()
    ops, local = _ops(tmp_path, monkeypatch, run)
    m = _member(ops)
    ops.start(m, ["f09-0"])
    ops.start(m, ["f09-0"])  # a restart: checked again at the start, the matching copy is kept
    assert len(run.writes) == 1
    assert sum(c.startswith("echo YETO_PROMPTS") for c in _cmds(run)) == 3
    local.write_bytes(b'{"question": "2+2?", "answer": "4"}\n')  # a changed file is copied again
    ops.start(m, ["f09-0"])
    assert len(run.writes) == 2 and run.files["/srv/yeto/shared/prompts/p.jsonl"] == local.read_bytes()


def test_the_upload_streams_the_file_with_a_timeout_by_its_size(tmp_path, monkeypatch):
    run = Node()
    ops, _local = _ops(tmp_path, monkeypatch, run)
    ops.start(_member(ops), ["f09-0"])
    (write,) = [c for c in run.calls if c[0][0] == "ssh" and c[0][-1].startswith("umask 077; t=")]
    assert write[1]["stdin"] is not None and write[1].get("input") is None
    assert write[1]["timeout"] == sp.upload_timeout_s(108)
    assert sp.upload_timeout_s(0) == 60 and sp.upload_timeout_s(100 * 1024 * 1024) == 70
    assert sp.upload_timeout_s(10 ** 12) == 1800


def test_a_failed_write_stops_the_start_naming_it(tmp_path, monkeypatch):
    run = Node(write_rc=1)
    ops, _local = _ops(tmp_path, monkeypatch, run)
    with pytest.raises(sp.SshSamplerError, match=r"prompts upload /srv/yeto/shared/prompts/p.jsonl to node-a "
                                                 r"\(exit 1\): disk full"):
        ops.start(_member(ops), ["f09-0"])
    assert not any("run -d" in c for c in _cmds(run))


def test_a_copy_that_does_not_match_stops_the_start(tmp_path, monkeypatch):
    run = Node(corrupt=True)
    ops, _local = _ops(tmp_path, monkeypatch, run)
    with pytest.raises(sp.SshSamplerError, match="does not match .* after the upload") as error:
        ops.start(_member(ops), ["f09-0"])
    assert not getattr(error.value, "config_error", False)  # the node's, retried
    assert not any("run -d" in c for c in _cmds(run))


def test_a_prompts_path_the_sampler_cannot_see_is_a_config_error(tmp_path, monkeypatch):
    from yeto.launch.pool.island_pool import is_config_error

    run = Node()
    ops, _local = _ops(tmp_path, monkeypatch, run, prompts_path="/opt/p.jsonl")
    with pytest.raises(sp.SshSamplerError, match="under no mount or workdir of node-a") as error:
        ops.start(_member(ops), ["f09-0"])
    assert is_config_error(error.value) and run.files == {}


def test_a_missing_local_prompts_file_is_a_config_error(tmp_path, monkeypatch):
    from yeto.launch.pool.island_pool import is_config_error

    run = Node()
    ops, local = _ops(tmp_path, monkeypatch, run)
    local.unlink()
    with pytest.raises(sp.SshSamplerError, match="prompts_file .*prompts.jsonl: No such file") as error:
        ops.start(_member(ops), ["f09-0"])
    assert is_config_error(error.value) and run.files == {}


def test_without_prompts_file_nothing_is_uploaded(tmp_path, monkeypatch):
    run = Node()
    ops, _local = _ops(tmp_path, monkeypatch, run)
    ops.cfg.pop("prompts_file")
    ops.start(_member(ops), ["f09-0"])
    assert run.files == {} and not any("YETO_PROMPTS" in c for c in _cmds(run))


# -- 1: two pools of one run on the nodes of one lease store ----------------------------


def _sampler(name, **kw):
    return _raw(name, f"ops@{name}", roles=["sampler"], **kw)


def _pool(tmp_path, name, nodes=None, *, lease_store=None, inventory_nodes=None, devices=None, target=1,
          slots=None, inventory=None):
    if inventory is None:
        inventory = tmp_path / f"inv-{name}.yaml"
        doc = {"nodes": inventory_nodes or [_sampler("own-2", address="10.0.0.2")]}
        if lease_store:
            doc["lease_store"] = lease_store
        inventory.write_text(json.dumps(doc))
    provider = {"kind": "ssh", "inventory": str(inventory), "workdir": f"/srv/yeto/{name}",
                "runtime": "venv", "venv": {"python": "/opt/venv/bin/python"}}
    if nodes is not None:
        provider["nodes"] = nodes
    if devices is not None:
        provider["devices"] = devices
    cfg = {"provider": provider, "target": target}
    if slots is not None:
        cfg["slots"] = slots
    return name, cfg


def _shared_inventory(tmp_path, n, lease_store="file:///srv/leases"):
    inv = tmp_path / "shared.yaml"
    inv.write_text(json.dumps({"lease_store": lease_store, "nodes": [_sampler(f"own-{i}") for i in range(n)]}))
    return inv


def _levels(found):
    return [f.level for f in found]


def test_two_pools_with_enough_free_nodes_of_one_lease_store_are_a_note(tmp_path):
    inv = _shared_inventory(tmp_path, 3)
    a = _pool(tmp_path, "a", inventory=inv, slots=[0, 1, 2])
    b = _pool(tmp_path, "b", inventory=inv, slots=[0, 1, 2])
    (found,) = checks.findings([a, b])
    assert found.level == checks.NOTE
    assert found.text == ("pools a and b may both take node(s) own-0, own-1, own-2 (lease_store /srv/leases): a pool "
                          "skips a node the other holds; their targets (1 + 1) fit the 3 node(s) they can take")


def test_two_pools_short_of_nodes_of_one_lease_store_are_refused(tmp_path):
    inv = _shared_inventory(tmp_path, 2)
    a = _pool(tmp_path, "a", inventory=inv, slots=[0, 1], target=2)
    b = _pool(tmp_path, "b", inventory=inv, slots=[0, 1], target=2)
    (found,) = checks.findings([a, b])
    assert found.level == checks.ERROR
    assert found.text.startswith("pools a and b share node(s) own-0, own-1 of an inventory with node leases "
                                 "(lease_store /srv/leases): their targets (2 + 2) are more than the 2 node(s) the "
                                 "two can take, so a pool waits for a node the other holds.")
    assert "without lease_store and split the nodes' cards with provider.devices" in found.text
    assert "or put the pools on different nodes" in found.text


def test_two_pools_naming_one_leased_node_are_refused(tmp_path):
    inv = _shared_inventory(tmp_path, 3)
    a = _pool(tmp_path, "a", ["own-0"], inventory=inv, devices=[0, 1, 2, 3])
    b = _pool(tmp_path, "b", ["own-0", "own-1"], inventory=inv, slots=[0, 1], devices=[4, 5, 6, 7])
    (found,) = checks.findings([a, b])
    assert found.level == checks.ERROR and found.text.startswith(
        "pools a and b both name node(s) own-0 in provider.nodes on an inventory with node leases "
        "(lease_store /srv/leases)")


def test_one_lease_store_written_two_ways_is_one_store(tmp_path):
    a = _pool(tmp_path, "a", ["own-2"], lease_store="file:///srv/leases")
    b = _pool(tmp_path, "b", ["own-2"], lease_store="/srv/leases/")
    assert _levels(checks.findings([a, b])) == [checks.ERROR]
    assert checks.store_key("file:///srv/leases/") == checks.store_key("/srv/leases") == "/srv/leases"


def test_pools_of_two_lease_stores_or_none_do_not_wait_for_each_other(tmp_path):
    a = _pool(tmp_path, "a", ["own-2"], lease_store="file:///srv/leases-a")
    b = _pool(tmp_path, "b", ["own-2"], lease_store="file:///srv/leases-b")
    assert checks.findings([a, b]) == []  # each lease store leases the node to its own pool
    a = _pool(tmp_path, "a", ["own-2"], devices=[0, 1, 2, 3])
    b = _pool(tmp_path, "b", ["own-2"], devices=[4, 5, 6, 7])
    assert checks.findings([a, b]) == []  # no leases: the pools split the node's cards
    two = [_sampler("own-2"), _sampler("own-3")]
    a = _pool(tmp_path, "a", ["own-2"], lease_store="file:///srv/leases", inventory_nodes=two)
    b = _pool(tmp_path, "b", ["own-3"], lease_store="file:///srv/leases", inventory_nodes=two)
    assert checks.findings([a, b]) == []
    assert checks.findings([a, ("nc", {"provider": {"kind": "nc"}})]) == []


# -- 3: an ssh pool on the trainer's nodes ---------------------------------------------


def _side(nodes, *, store=None, need=1, spares=(), inventory="/i/trainers.yaml", entry="ssh:1x8xh100"):
    return checks.TrainerSide(inventory, store, entries=[entry], nodes=[parse_node(n, index=0) for n in nodes],
                              spares=[parse_node(n, index=0) for n in spares], need=need)


@pytest.mark.parametrize("trainer_raw", [
    _raw("t-0", "ops@own-2", roles=["trainer"]),  # one SSH target
    _raw("t-0", "root@own-2", roles=["trainer"]),  # one host, another user
    _raw("t-0", "ops@other", address="10.0.0.2", roles=["trainer"]),  # one cluster address
])
def test_a_pool_on_a_trainer_node_without_common_leases_is_refused(tmp_path, trainer_raw):
    pool = _pool(tmp_path, "p", ["own-2"])
    (found,) = checks.findings([pool], [_side([trainer_raw])])
    assert found.level == checks.ERROR
    assert found.text.startswith("pool p starts samplers on node own-2, which is a node of this run's trainer island "
                                 "ssh:1x8xh100 (t-0): the trainer takes every card of its nodes")
    (found,) = checks.findings([pool], [_side([], spares=[trainer_raw])])
    assert found.level == checks.ERROR and "which is a trainer-spare node of this run's trainer island" in found.text
    other = _raw("t-1", "ops@t-1", address="10.0.0.9")
    assert checks.findings([pool], [_side([other])]) == []


def _leased(tmp_path, nodes, store="/srv/leases"):
    inv = tmp_path / "leased.yaml"
    inv.write_text(json.dumps({"lease_store": store, "nodes": nodes}))
    return inv


def test_b_a_trainer_and_sampler_node_picked_by_the_pool_is_a_warning_when_enough_remain(tmp_path):
    """Review case B: a [trainer, sampler] node on a leased inventory, taken by the pool only."""
    nodes = [_raw(f"n-{i}", f"ops@n-{i}", roles=["trainer", "sampler"]) for i in range(3)]
    inv = _leased(tmp_path, nodes)
    pool = _pool(tmp_path, "p", ["n-2"], inventory=inv)
    (side,) = checks.trainer_sides(["ssh:2x8xh100"], str(inv))
    assert [n.name for n in side.nodes] == ["n-0", "n-1", "n-2"] and side.need == 2
    (found,) = checks.findings([pool], [side], quarantined=lambda s: set())
    assert found.level == checks.WARN
    assert found.text == ("pool p on n-2: trainer nodes of this run's trainer island(s) ssh:2x8xh100; the trainer "
                          "skips a node a pool holds (lease_store /srv/leases): 3 trainer node(s), the pools may hold "
                          "1 of them, 2 left for the 2 node(s) the islands need")


def test_c_a_leased_inventory_without_roles_refuses_only_when_too_few_trainer_nodes_remain(tmp_path):
    """Review case C: nodes without roles on a leased inventory: any node is a trainer and a sampler node."""
    inv = _leased(tmp_path, [_raw(f"n-{i}", f"ops@n-{i}") for i in range(3)])
    (side,) = checks.trainer_sides(["ssh:2x8xh100"], str(inv))
    one = _pool(tmp_path, "p", inventory=inv, slots=[0, 1, 2], target=1)
    assert _levels(checks.findings([one], [side], quarantined=lambda s: set())) == [checks.WARN]
    two = _pool(tmp_path, "p", inventory=inv, slots=[0, 1, 2], target=2)
    (found,) = checks.findings([two], [side], quarantined=lambda s: set())
    assert found.level == checks.ERROR
    assert "3 trainer node(s), the pools may hold 2 of them, 1 left for the 2 node(s) the islands need" in found.text
    assert "the trainer would wait for a node a pool holds" in found.text
    # a quarantined trainer node is no candidate: one pool of target 1 is then too many
    (found,) = checks.findings([one], [side], quarantined=lambda s: {"n-0"})
    assert found.level == checks.ERROR and "(1 quarantined not counted)" in found.text


def test_c_the_quarantine_is_read_from_the_lease_store(tmp_path):
    from yeto.launch import node_quarantine
    from yeto.launch.node_lease import open_objects

    store = tmp_path / "leases"
    store.mkdir()
    inv = _leased(tmp_path, [_raw(f"n-{i}", f"ops@n-{i}") for i in range(3)], store=str(store))
    (side,) = checks.trainer_sides(["ssh:2x8xh100"], str(inv))
    assert checks.lease_quarantines(side) == set()
    open_objects(str(store)).write(node_quarantine.shared_key("n-1"),
                                   json.dumps({"run": "r", "reason": "lost"}).encode())
    assert checks.lease_quarantines(side) == {"n-1"}


def test_d_without_leases_a_pool_on_any_trainer_node_is_refused(tmp_path):
    """Review case D: no lease_store: the launch takes every trainer node (and refuses an
    inventory that lists more trainer nodes than the island needs)."""
    nodes = [_raw(f"n-{i}", f"ops@n-{i}", roles=["trainer", "sampler"]) for i in range(3)]
    inv = tmp_path / "plain.yaml"
    inv.write_text(json.dumps({"nodes": nodes}))
    (side,) = checks.trainer_sides(["ssh:2x8xh100"], str(inv))  # refused at launch: every trainer node counts
    assert [n.name for n in side.nodes] == ["n-0", "n-1", "n-2"]
    pool = _pool(tmp_path, "p", ["n-2"], inventory=inv)
    (found,) = checks.findings([pool], [side])
    assert found.level == checks.ERROR and "the trainer takes every card of its nodes" in found.text


def test_a_pool_of_another_lease_store_on_a_leased_trainer_node_is_refused(tmp_path):
    nodes = [_raw(f"n-{i}", f"ops@n-{i}", roles=["trainer", "sampler"]) for i in range(3)]
    inv = _leased(tmp_path, nodes)
    (side,) = checks.trainer_sides(["ssh:2x8xh100"], str(inv))
    pool = _pool(tmp_path, "p", ["own-2"], lease_store="/srv/other-leases",
                 inventory_nodes=[_raw("own-2", "ops@n-2", roles=["sampler"])])
    (found,) = checks.findings([pool], [side], quarantined=lambda s: set())
    assert found.level == checks.ERROR  # the trainer's leases do not see the pool's


def test_one_name_is_one_machine_only_in_one_inventory_file():
    a, b = parse_node(_raw("n", "h1"), index=0), parse_node(_raw("n", "h2"), index=0)
    assert checks.same_machine(a, "/i.yaml", b, "/i.yaml")
    assert not checks.same_machine(a, "/i.yaml", b, "/j.yaml")
    lo = parse_node(_raw("x", "h3", address="127.0.0.1"), index=0)
    assert not checks.same_machine(lo, None, parse_node(_raw("y", "h4", address="127.0.0.1"), index=0), None)
    assert checks.same_machine(parse_node(_raw("x", "ops@Box-1"), index=0), None,
                               parse_node(_raw("y", "root@box-1"), index=0), None)


def test_the_trainer_side_holds_the_island_nodes_and_its_spares(tmp_path):
    inv = tmp_path / "trainers.yaml"
    inv.write_text(json.dumps({"nodes": [_raw("t-0", "ops@t-0", roles=["trainer"]),
                                         _raw("t-1", "ops@t-1", roles=["trainer"]),
                                         _raw("s-0", "ops@s-0", roles=["trainer-spare"]),
                                         _raw("p-0", "ops@p-0", roles=["sampler"])]}))
    (side,) = checks.trainer_sides([f"ssh:{inv}"], None)
    assert [n.name for n in side.nodes] == ["t-0", "t-1"] and [n.name for n in side.spares] == ["s-0"]
    assert side.inventory == str(inv.resolve()) and side.entries == [f"ssh:{inv}"] and side.need == 2
    assert side.store is None
    assert checks.trainer_sides(["ssh:/nonexistent/inventory.yaml"], None) == []  # its launch dry run says why
    pool = _pool(tmp_path, "p", ["s-0"], inventory_nodes=[_sampler("s-0")])
    (found,) = checks.findings([pool], [side])
    assert "which is a trainer-spare node of this run's trainer island" in found.text


def test_islands_of_one_inventory_need_their_nodes_together(tmp_path):
    inv = tmp_path / "trainers.yaml"
    inv.write_text(json.dumps({"nodes": [_raw(f"t-{i}", f"ops@t-{i}", roles=["trainer"]) for i in range(2)]}))
    (side,) = checks.trainer_sides(["ssh:2x8xh100"], str(inv))
    assert [n.name for n in side.nodes] == ["t-0", "t-1"] and side.need == 2
    (side,) = checks.trainer_sides(["ssh:1x8xh100", "ssh:1x8xh100"], str(inv))  # two islands
    assert [n.name for n in side.nodes] == ["t-0", "t-1"] and side.need == 2 and len(side.entries) == 2


def test_a_leased_trainer_inventory_counts_every_node_the_launch_may_take(tmp_path):
    inv = _leased(tmp_path, [_raw(f"t-{i}", f"ops@t-{i}", roles=["trainer"]) for i in range(3)],
                  store=str(tmp_path / "leases"))
    (side,) = checks.trainer_sides(["ssh:1x8xh100"], str(inv))
    assert [n.name for n in side.nodes] == ["t-0", "t-1", "t-2"] and side.need == 1


def test_a_fleet_counts_each_island_once_from_the_trainer_gpu_and_the_islands():
    assert checks.island_entries("ssh:1x8xb300,modal:1x8xb300", ["ssh:1x8xb300", "modal:1x8xb300"]) == \
        ["ssh:1x8xb300"]
    assert checks.island_entries("ssh:1x8xb300,ssh:1x8xb300", []) == ["ssh:1x8xb300"] * 2
    assert checks.island_entries(None, ["ssh:2x8xb300", "ssh:1x8xb300"]) == ["ssh:2x8xb300", "ssh:1x8xb300"]


def test_one_node_more_than_the_two_pools_can_take_is_refused(tmp_path):
    inv = _shared_inventory(tmp_path, 3)
    a = _pool(tmp_path, "a", inventory=inv, slots=[0, 1, 2], target=2)
    b = _pool(tmp_path, "b", inventory=inv, slots=[0, 1, 2], target=2)
    assert _levels(checks.findings([a, b])) == [checks.ERROR]
    b = _pool(tmp_path, "b", inventory=inv, slots=[0, 1, 2], target=1)
    assert _levels(checks.findings([a, b])) == [checks.NOTE]  # 2 + 1 fit the 3 nodes


def test_a_node_one_pool_names_and_the_other_may_take_is_counted(tmp_path):
    inv = _shared_inventory(tmp_path, 3)
    a = _pool(tmp_path, "a", ["own-0"], inventory=inv)
    b = _pool(tmp_path, "b", inventory=inv, slots=[0, 1, 2])
    (found,) = checks.findings([a, b])
    assert found.level == checks.NOTE and found.text.startswith("pools a and b may both take node(s) own-0")
