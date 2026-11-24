"""The SSH + docker pool on ``provider.kind: sky`` starts its samplers through the node
runtime (yeto.node.ssh_docker): the node's family flags (a CUDA VM gets ``--gpus all``,
not the MI355X ones), the image from the engine image catalog, the assignment in the
run store, the token only on stdin. ``node_runtime: false`` keeps the docker route of
yeto.launch.nc.ssh_sampler; National Compute pools never take the node runtime."""

from __future__ import annotations

import json
import shlex
from types import SimpleNamespace

import pytest

from yeto.launch.nc import cli, ssh_sampler
from yeto.launch.nc.client import NodeLease
from yeto.launch.rollout.offers import Offer
from yeto.node import records, ssh_docker
from yeto.node.route import NodeRoute
from yeto.rl.engine.object_store import DirectoryObjectStore

SHA = "abcdef123456"
SPEC = {"sampler_id": "sky-0", "engine": {"adapter": "vllm-server", "server": {"port": 8000}},
        "prompts": {"path": "/work/prompts/p.jsonl", "groups_per_round": 2}}


@pytest.fixture(autouse=True)
def _state(tmp_path, monkeypatch):
    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path / "state"))


class Archive:
    stdout = None

    def __init__(self, *a, **k):
        pass

    def wait(self):
        return 0


class Run:
    def __init__(self, prompts: bytes = b""):
        self.calls, self.prompts = [], prompts

    def __call__(self, argv, **kw):
        self.calls.append((list(argv), kw))
        if argv[:2] == ["git", "-C"]:
            return SimpleNamespace(returncode=0, stdout=SHA + "\n", stderr="")
        text = argv[-1]
        if "YETO_PROMPTS" in text:
            import hashlib  # noqa: PLC0415

            digest = hashlib.sha256(self.prompts).hexdigest()
            return SimpleNamespace(returncode=0, stdout=f"YETO_PROMPTS {len(self.prompts)} {digest}\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def ssh(self):
        return [c[0][-1] for c in self.calls if c[0][0] == "ssh"]


def provider(kind="sky", accelerators="H100:8"):
    return SimpleNamespace(kind=kind, offers=[Offer("h100", {"accelerators": accelerators})],
                           offer_of=lambda ticket: "h100", attach=lambda lease, ssh_run: None)


def _ops(tmp_path, monkeypatch, *, kind="sky", accelerators="H100:8", **extra):
    monkeypatch.setattr(ssh_sampler.subprocess, "Popen", Archive)
    (tmp_path / "spec.json").write_text(json.dumps(SPEC))
    (tmp_path / "r2.env").write_text("R2_BUCKET=b\n")
    (tmp_path / "p.jsonl").write_text('{"q": "x"}\n')
    cfg = {"target": 1, "slots": [0], "shard_base": "sky", "run": "run-s", "spec": str(tmp_path / "spec.json"),
           "run_prefix": "runs1", "r2_env_file": str(tmp_path / "r2.env"), "prompts_file": str(tmp_path / "p.jsonl"),
           "model_pull": {"src_key": "models/k31"},
           "provider": {"kind": kind, "resources": {"accelerators": accelerators}}, **extra}
    run = Run((tmp_path / "p.jsonl").read_bytes())
    ops = cli.NcSshOps(cfg, run=run, provider=provider(kind, accelerators))
    monkeypatch.setattr(ops, "_spec", lambda m, shards: dict(SPEC))
    lease = NodeLease(0, "yeto-smp-run-s-0", "192.0.2.9", "ubuntu", "", None, None, 8, None)
    cli.save_lease(lease)
    m = SimpleNamespace(account=0, ticket=lease.ticket, shard="sky-0", data_slot=None, replaces_shard=None,
                        predecessor_lost=False)
    return ops, run, m, cfg


def _store_route(ops, tmp_path, run):
    route = NodeRoute(ops.cfg, kind="sky", objects=lambda: DirectoryObjectStore(tmp_path / "store"),
                      store=lambda: "s3://b/runs/runs1", run=run, publish_code=False)
    ops._node_route = route  # noqa: SLF001 - the R2 store of a real pool, here a directory
    return route


def test_a_cuda_vm_runs_the_node_runtime_with_cuda_flags(tmp_path, monkeypatch):
    ops, run, m, _cfg = _ops(tmp_path, monkeypatch)
    route = _store_route(ops, tmp_path, run)
    ops.start(m, ["sky-0"])
    cmds = run.ssh()
    assert not any("YETO_NC_PULL" in c or "pull.sh" in c for c in cmds)  # the node pulls by its plan
    docker = next(c for c in cmds if "docker run -d" in c)
    argv = shlex.split(docker.split("; ", 1)[1])
    assert argv[argv.index("--gpus") + 1] == "all"
    assert "/dev/kfd" not in docker and "HIP_VISIBLE_DEVICES" not in docker
    assert "-v" in argv and "/home/ubuntu/yeto/work:/work" in argv and "/home/ubuntu/yeto/models:/models" in argv
    assert "/home/ubuntu/yeto/cache:/root/.cache" in argv and "YETO_DEVICE_FAMILY=cuda" in argv
    assert argv[argv.index("--env-file") + 1] == "/home/ubuntu/yeto/work/sampler/.store.env"
    image = argv[argv.index("--entrypoint") + 2]
    assert "rocm" not in image and image != ssh_sampler.IMAGE
    inner = argv[-1]
    assert f"-m yeto.node run --assignment {m.ticket} --store s3://b/runs/runs1" in inner
    assert "--token-file /work/sampler/sky-0.token" in inner and "--log -" in inner
    assert "yeto.rl.engine.engine_sampler" not in docker
    token = next(c[1]["input"] for c in run.calls if c[0][0] == "ssh" and "sky-0.token" in c[0][-1])
    assert all(token not in str(c[0]) for c in run.calls)  # the token only ever on stdin
    profile = json.loads(next(c[1]["input"] for c in run.calls if c[0][0] == "ssh" and "profile.json" in c[0][-1]))
    assert profile["container"] == "process" and profile["paths"]["work"] == "/work" and profile["card"] == "H100"
    doc = json.loads(route.objects.read(f"{records.ASSIGNMENTS_DIR}/{m.ticket}.json"))
    assert doc["source"] == "sky" and doc["code"]["sha"] == SHA and doc["image"] == image
    assert doc["model"]["source"] == "store" and doc["model"]["key"] == "models/k31"
    assert doc["model"]["dest"] == "/models/kimi-k3.1-pro"  # the spec's path on the old route
    assert doc["token_sha256"] == records.token_digest(token) and doc["spec"]["start_on_latest"] is True
    assert any("YETO_PROMPTS" in c and "/home/ubuntu/yeto/work/prompts/p.jsonl" in c for c in cmds)


def test_the_probe_reads_the_store_without_ssh(tmp_path, monkeypatch):
    ops, run, m, _cfg = _ops(tmp_path, monkeypatch)
    route = _store_route(ops, tmp_path, run)
    ops.start(m, ["sky-0"])
    before = len(run.calls)
    assert ops.probe(m).sampler != "running"  # assigned, not ready
    assignment = records.read_assignment(route.objects, m.ticket, next(
        c[1]["input"] for c in run.calls if c[0][0] == "ssh" and "sky-0.token" in c[0][-1]))
    records.NodeReporter(route.objects, assignment.ticket, assignment.generation).ready(shard="sky-0")
    assert ops.probe(m).joined
    assert len(run.calls) == before


def test_an_mi355x_vm_gets_the_rocm_flags(tmp_path, monkeypatch):
    ops, run, m, _cfg = _ops(tmp_path, monkeypatch, accelerators="MI355X:8")
    _store_route(ops, tmp_path, run)
    ops.start(m, ["sky-0"])
    docker = next(c for c in run.ssh() if "docker run -d" in c)
    assert "--device /dev/kfd" in docker and "--gpus" not in docker and "YETO_DEVICE_FAMILY=rocm" in docker


def test_node_runtime_false_keeps_the_docker_route(tmp_path, monkeypatch):
    ops, run, m, cfg = _ops(tmp_path, monkeypatch, node_runtime=False)
    cli.pool_config(cfg)
    assert ops.node_route is None
    monkeypatch.setattr(ops, "_pull", lambda lease, pull: run.calls.append((["pull"], {})))
    ops.start(m, ["sky-0"])
    assert ["pull"] in [c[0] for c in run.calls]
    cmds = run.ssh()
    assert any("yeto.rl.engine.engine_sampler" in c for c in cmds)
    assert not any("yeto.node" in c for c in cmds)


def test_only_a_sky_pool_on_the_r2_path_takes_the_node_runtime(tmp_path, monkeypatch):
    assert isinstance(_ops(tmp_path, monkeypatch)[0].node_route, NodeRoute)
    for kind, extra in (("nc", {}), ("daytona", {}), ("sky", {"data_path": "syncer"}),
                        ("sky", {"sampler_route": "process"})):
        ops = _ops(tmp_path, monkeypatch, kind=kind, **extra)[0]
        assert ops.node_route is None, (kind, extra)
    with pytest.raises(ValueError, match="node_runtime must be true or false"):
        cli.pool_config(dict(_ops(tmp_path, monkeypatch)[3], node_runtime="no"))


def test_no_image_for_the_card_is_refused_before_docker_run(tmp_path, monkeypatch):
    ops, run, m, _cfg = _ops(tmp_path, monkeypatch, accelerators="910B4:8")
    _store_route(ops, tmp_path, run)
    monkeypatch.setattr(ops, "_spec", lambda m, shards: dict(SPEC, engine={"adapter": "sglang-server"}))
    with pytest.raises(ssh_docker.SshDockerError, match="910B4|ascend"):
        ops.start(m, ["sky-0"])
    assert not any("docker run" in c for c in run.ssh())


def test_container_dest_follows_the_routes_mounts():
    assert ssh_docker.container_dest(None) == "/models/kimi-k3.1-pro"
    assert ssh_docker.container_dest("/mnt/nvme/models/Other") == "/models/Other"
    assert ssh_docker.container_dest("/models/x") == "/models/x"
    with pytest.raises(ssh_docker.SshDockerError):
        ssh_docker.container_dest("/data/x")


def test_the_pool_configs_node_profile_replaces_offer_fields(tmp_path, monkeypatch):
    ops, run, m, _cfg = _ops(tmp_path, monkeypatch, node_profile={"card": "H200", "paths": {"work": "/data/w"}})
    _store_route(ops, tmp_path, run)
    ops.start(m, ["sky-0"])
    docker = next(c for c in run.ssh() if "docker run -d" in c)
    assert "/data/w:/work" in docker and "/home/ubuntu/yeto/models:/models" in docker
    profile = json.loads(next(c[1]["input"] for c in run.calls if c[0][0] == "ssh" and "profile.json" in c[0][-1]))
    assert profile["card"] == "H200" and profile["hbm_gib"] == 141.0
