"""Weights onto SSH trainer nodes (yeto.launch.ssh_weights): stage of each node
in Miles' placement order, host paths through the inventory mounts, the node
script, the launch-argument plan, and the preflight copy run end to end on
local "nodes" against a fake S3 endpoint."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_weights_sync import LAYER_FILES, MODEL, REV, SHARED_FILE, FakeS3, tiny_model
from yeto.hw import card_tools
from yeto.launch import ssh_weights as sw
from yeto.launch.node_inventory import Node
from yeto.launch.ssh_trainer_island import (
    CUDA_RDMA_DOCKER_ARGS,
    SshIslandError,
    SshOps,
    SshTrainerIslands,
    SshTrainerSpec,
)
from yeto.launch.storage import model_store_env, model_store_filesystem, model_store_local_path

STORE = "/mnt/yeto-models"


def b300_nodes(n, *, mounts=None, address=None) -> list[Node]:
    """``n`` docker B300 nodes of the 6 x 8 inventory shape; ``mounts`` / ``address``: per index."""
    return [Node(name=f"b300-{i}", host=f"ops@10.0.0.{10 + i}", device_family="cuda", cpu_arch="x86_64",
                 card="B300", cards_per_node=8, hbm_gib=288.0, host_memory_gib=2048.0, fabric="ib",
                 interface="bond0", address=address(i) if address else f"10.1.0.{10 + i}",
                 mounts=mounts(i) if mounts else {"/nvme/models": STORE}, runtime="docker",
                 image="ghcr.io/o/yeto-miles-k31@sha256:" + "a" * 64, docker_args=CUDA_RDMA_DOCKER_ARGS)
            for i in range(n)]


# ---------------------------------------------------------------- stages of each node
def test_k31_six_nodes_pp3_map_two_nodes_per_stage_in_node_ip_order():
    addresses = [f"10.1.0.{i}" for i in (12, 9, 10, 15, 11, 13)]  # inventory order is not IP order
    stages = sw.node_stages(addresses, 8, 3)
    by_ip = dict(zip(addresses, stages))
    # Miles sorts the trainer's GPUs by node IP as numbers (9 < 10), then GPU id: 16 ranks per stage
    assert [by_ip[f"10.1.0.{i}"] for i in (9, 10, 11, 12, 13, 15)] == [(0,), (0,), (1,), (1,), (2,), (2,)]
    # a node whose GPUs straddle two stages gets both
    assert sw.node_stages(["10.0.0.1", "10.0.0.2"], 8, 4) == [(0, 1), (2, 3)]
    assert sw.node_stages(["10.0.0.1", "10.0.0.2", "10.0.0.3"], 8, 2) == [(0,), (0, 1), (1,)]
    with pytest.raises(sw.WeightsSyncError, match="do not split"):
        sw.node_stages(["10.0.0.1"], 8, 3)


def test_pipeline_split_reads_the_train_extra_argv():
    raw = ("--log-probs-chunk-size 16384 --decoder-first-pipeline-num-layers 32 "
           "--decoder-last-pipeline-num-layers=33")
    assert sw.pipeline_split(raw) == (32, 33)
    assert sw.pipeline_split("--decoder-last-pipeline-num-layers,27") == (None, 27)
    assert sw.pipeline_split(None) == (None, None)
    with pytest.raises(sw.WeightsSyncError, match="integer"):
        sw.pipeline_split("--decoder-first-pipeline-num-layers x")


def test_host_path_goes_through_the_deepest_inventory_mount():
    node = SimpleNamespace(name="b0", runtime="docker",
                           mounts={"/nvme/yeto": "/root/.cache", "/nvme/models": STORE, "/": "/host"})
    assert sw.host_path(node, STORE) == "/nvme/models"
    assert sw.host_path(node, STORE + "/k31") == "/nvme/models/k31"
    assert sw.host_path(node, "/host/data/m") == "/data/m"
    with pytest.raises(sw.WeightsSyncError, match="not under any inventory mount"):
        sw.host_path(SimpleNamespace(name="b1", runtime="docker", mounts={"/nvme": "/data"}), STORE)
    assert sw.host_path(SimpleNamespace(name="r", runtime="rootfs", mounts={}), STORE) == STORE


# ---------------------------------------------------------------- credentials and the plan
def test_source_credentials_never_take_the_launchers_cloud_keys():
    store = {"YETO_RL_STORE_S3_ACCESS_KEY_ID": "sk", "YETO_RL_STORE_S3_SECRET_ACCESS_KEY": "ss",
             "YETO_RL_STORE_S3_ENDPOINT_URL": "https://r2.example"}
    launcher = {"AWS_ACCESS_KEY_ID": "cloud", "AWS_SECRET_ACCESS_KEY": "cloud"}
    assert sw.source_env("s3://b/m", store, launcher) == {
        "YETO_WEIGHTS_S3_ACCESS_KEY_ID": "sk", "YETO_WEIGHTS_S3_SECRET_ACCESS_KEY": "ss",
        "YETO_WEIGHTS_S3_ENDPOINT_URL": "https://r2.example"}
    own = {"YETO_WEIGHTS_S3_ACCESS_KEY_ID": "wk", "YETO_WEIGHTS_S3_SECRET_ACCESS_KEY": "ws"}
    assert sw.source_env("r2://b/m", store, own)["YETO_WEIGHTS_S3_ACCESS_KEY_ID"] == "wk"
    with pytest.raises(sw.WeightsSyncError, match="no credentials"):
        sw.source_env("s3://b/m", {}, launcher)
    assert sw.source_env("/nfs/models/k31", {}, launcher) == {}


def _args(**kw):
    base = dict(model=MODEL, model_revision=REV, model_store=f"local://{STORE}", model_store_source="/nfs/tiny",
                model_store_shards="auto", pipeline_parallel=3, rl_sample_role="trainer", modal_env=[])
    return SimpleNamespace(**(base | kw))


def test_plan_from_the_launch_arguments(monkeypatch):
    nodes = b300_nodes(6)
    monkeypatch.delenv("YETO_MILES_TRAIN_EXTRA_ARGV", raising=False)
    env = {"YETO_MILES_TRAIN_EXTRA_ARGV": "--decoder-first-pipeline-num-layers 32 "
                                          "--decoder-last-pipeline-num-layers 32"}
    plan = sw.weights_plan_from_args(_args(), env, nodes, environ={})
    assert plan.stage_shards and (plan.pipeline_parallel, plan.pp_first, plan.pp_last) == (3, 32, 32)
    assert plan.store == STORE and plan.env == {}
    # the split may also come from --modal-env
    plan = sw.weights_plan_from_args(
        _args(modal_env=["YETO_MILES_TRAIN_EXTRA_ARGV=--decoder-last-pipeline-num-layers 27"]), {}, nodes,
        environ={})
    assert (plan.pp_first, plan.pp_last) == (None, 27)
    # auto: a trainer that also samples (or PP 1) needs every file on every node
    assert not sw.weights_plan_from_args(_args(rl_sample_role=None), {}, nodes, environ={}).stage_shards
    assert not sw.weights_plan_from_args(_args(pipeline_parallel=1), {}, nodes, environ={}).stage_shards
    assert sw.weights_plan_from_args(_args(model_store_shards="all"), {}, nodes, environ={}).stage_shards is False
    assert sw.weights_plan_from_args(_args(model_store_source=None), {}, nodes) is None
    for bad, match in ((dict(model_revision="main"), "40-hex"), (dict(model="/data/models/k31"), "Hub id"),
                       (dict(model_store=None), "local:///PATH"),
                       (dict(model_store_shards="stage", pipeline_parallel=1), "pipeline-parallel > 1"),
                       (dict(actor_num_gpus_per_node=4), "every GPU")):
        with pytest.raises(sw.WeightsSyncError, match=match):
            sw.weights_plan_from_args(_args(**bad), {}, nodes, environ={})
    unmounted = [n.__class__(**{**n.__dict__, "mounts": {}}) for n in nodes]
    with pytest.raises(sw.WeightsSyncError, match="not under any inventory mount"):
        sw.weights_plan_from_args(_args(), {}, unmounted, environ={})


def test_local_model_store_uri():
    assert model_store_local_path(f"local://{STORE}/") == STORE
    assert model_store_local_path("nebius-fs://computefilesystem-x") is None
    assert model_store_local_path(None) is None
    for bad in ("local://relative/path", "local:///a/../b", "local://"):
        with pytest.raises(ValueError, match="local:///ABSOLUTE"):
            model_store_local_path(bad)
    assert model_store_filesystem(f"local://{STORE}", "ssh", None) is None


# ---------------------------------------------------------------- node script
def test_node_script_is_valid_bash_and_keeps_the_secrets_out_of_any_argv():
    node = b300_nodes(1)[0]
    plan = sw.WeightsPlan(model=MODEL, revision=REV, store=STORE, source="s3://bkt/m", stage_shards=True,
                          pipeline_parallel=3, pp_first=32, pp_last=32,
                          env={"YETO_WEIGHTS_S3_SECRET_ACCESS_KEY": "s e$cret"})
    script = sw.node_script(plan, node, (1,), image="ghcr.io/o/img@sha256:" + "a" * 64)
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    assert "export YETO_WEIGHTS_S3_SECRET_ACCESS_KEY='s e$cret'" in script
    argv_lines = [line for line in script.splitlines() if "python3 -I" in line]
    assert argv_lines and not any("e$cret" in line for line in argv_lines)
    assert "--stages 1 --pp 3 --pp-first 32 --pp-last 32" in script and "--root /nvme/models" in script
    assert "-e YETO_WEIGHTS_S3_SECRET_ACCESS_KEY --entrypoint python3" in script  # the no-python3 fallback
    with pytest.raises(sw.WeightsSyncError, match="delimiter"):
        sw.node_script(plan, node, None, tool_source=f"x = 1\n{sw.HEREDOC}\n")


# ---------------------------------------------------------------- end to end on local "nodes"
class LocalOps(SshOps):
    """Runs each node's script with the local bash (the "node" is this host); the
    card and arch checks answer as a B300 node would."""

    def __init__(self):
        self.started = []

    def run(self, node, script, *, timeout=300.0):
        if script == "uname -m":
            return subprocess.CompletedProcess([], 0, stdout="x86_64\n", stderr="")
        return subprocess.CompletedProcess([], 0, stdout="ok", stderr="")  # cards, docker pull

    def start(self, node, script):
        self.started.append(node.name)
        return subprocess.Popen(["bash", "-c", script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, text=True, bufsize=1)


@pytest.fixture
def fake_s3():
    fake = FakeS3({f"models/tiny/{k}": v for k, v in tiny_model().items()})
    yield fake
    fake.close()


def _local_plan(fake, **kw):
    return sw.WeightsPlan(model=MODEL, revision=REV, store=STORE, source="s3://bkt/models/tiny",
                          env=fake.env(), workers=2, procs=False, **kw)


def test_preflight_copies_each_nodes_stage_files_in_parallel_then_the_store_hits(fake_s3, tmp_path, monkeypatch):
    monkeypatch.setattr(card_tools, "check_cards", lambda *a, **k: [])
    nodes = b300_nodes(3, mounts=lambda i: {str(tmp_path / f"n{i}"): STORE},
                       address=lambda i: f"10.1.0.{30 - i}")  # IP order is the reverse of the inventory
    plan = _local_plan(fake_s3, stage_shards=True, pipeline_parallel=3)
    spec = SshTrainerSpec(nodes=tuple(nodes), run_id="k31", run_script="true", workdir="/opt/yeto", weights=plan)
    lines = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: lines.append(" ".join(map(str, a))))
    ops = LocalOps()
    SshTrainerIslands(spec, name="t", jobs_path=tmp_path / "j.json", ops=ops).preflight()
    assert sorted(ops.started) == ["b300-0", "b300-1", "b300-2"]
    model = tiny_model()
    layer_files = list(LAYER_FILES)
    for i, expect_stage in enumerate((2, 1, 0)):  # node 0 has the highest IP: the last stage
        snap = tmp_path / f"n{i}" / "hub" / "models--org--tiny" / "snapshots" / REV
        shards = sorted(p.name for p in snap.glob("*.safetensors"))
        assert shards == sorted([layer_files[expect_stage], SHARED_FILE])
        assert all((snap / name).read_bytes() == model[name] for name in shards)
        marker = json.loads((tmp_path / f"n{i}" / "yeto-complete" / f"org--tiny@{REV}.json").read_text())
        assert marker["stages"] == [expect_stage] and marker["partial"]
        # the island's setup / run scripts see the completed store at the container path
        hit = subprocess.run(["bash", "-c", model_store_env(MODEL, REV, str(tmp_path / f"n{i}"))
                              + '\necho "HUB=$HF_HUB_CACHE HIT=$YETO_MODEL_STORE_HIT"'],
                             capture_output=True, text=True, check=True,
                             env={k: v for k, v in os.environ.items() if k != "HF_HUB_CACHE"})
        assert f"HUB={tmp_path / f'n{i}'}/hub HIT=1" in hit.stdout
    text = "\n".join(lines)
    assert "[weights b300-0] [yeto-weights] stages 2 of PP 3" in text
    assert "b300-2 (10.1.0.28) runs stage(s) 0" in text
    assert "[launcher] weights on b300-1:" in text and "GB/s" in text


def test_a_failing_node_fails_the_preflight_with_its_reason(fake_s3, tmp_path, monkeypatch):
    monkeypatch.setattr(card_tools, "check_cards", lambda *a, **k: [])
    nodes = b300_nodes(2, mounts=lambda i: {str(tmp_path / f"n{i}"): STORE})
    plan = _local_plan(fake_s3, headroom=1e9)  # no file system has that much room
    spec = SshTrainerSpec(nodes=tuple(nodes), run_id="k31", run_script="true", workdir="/opt/yeto", weights=plan)
    monkeypatch.setattr("builtins.print", lambda *a, **k: None)
    islands = SshTrainerIslands(spec, name="t", jobs_path=tmp_path / "j.json", ops=LocalOps())
    with pytest.raises(SshIslandError, match=r"weights sync failed on b300-0 \(rc 1\).*not enough disk space"):
        islands.preflight()
    assert not islands.checked


def test_a_node_that_does_not_finish_in_time_is_stopped(tmp_path, monkeypatch):
    class Hung:
        def __init__(self):
            self.stdout, self.rc = iter(()), None

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("ssh", timeout)

        def poll(self):
            return self.rc

        def terminate(self):
            self.rc = -15

    procs = []
    ops = SimpleNamespace(start=lambda node, script: procs.append(Hung()) or procs[-1])
    nodes = b300_nodes(2, mounts=lambda i: {str(tmp_path): STORE})
    plan = sw.WeightsPlan(model=MODEL, revision=REV, store=STORE, source="/nfs/m", timeout_s=5)
    with pytest.raises(sw.WeightsSyncError, match="did not finish within 5 s"):
        sw.sync_weights(plan, nodes, [n.address for n in nodes], ops, out=lambda line: None)
    assert all(p.rc == -15 for p in procs)


# ---------------------------------------------------------------- the island scripts
def test_island_task_uses_a_local_store_and_the_worker_skips_its_fetch_on_a_hit(monkeypatch):
    from tests.test_rl_engine_selection import _cli, _island_task
    from yeto.launcher import _prepare_rl_args

    args = _cli(["--gpu", "nebius:2x1xl40s@eu-north1", "--model-store", f"local://{STORE}"])
    _prepare_rl_args(args)
    task = _island_task(args, monkeypatch)
    env = model_store_env("org/model", "a" * 40, STORE)
    assert env in task.setup and '[ -n "$YETO_MODEL_STORE_HIT" ] || (nohup sh -c ' in task.setup
    assert task.run.index(env) < task.run.index("set -e")
    fetch = [line for line in task.run.splitlines() if '[ -n "${YETO_MODEL_STORE_HIT:-}" ] || python3 -c ' in line]
    assert len(fetch) == 1 and "snapshot_download" in fetch[0]
    for hit, rc in (("1", 0), ("", 1)):  # PATH without python3: a fetch that runs fails the node
        done = subprocess.run(["/bin/bash", "-c", fetch[0]], check=False, capture_output=True, text=True,
                              env={"PATH": "/nonexistent", "YETO_MODEL_STORE_HIT": hit})
        assert done.returncode == rc


def test_model_fetches_stay_off_the_hub_on_a_store_hit(monkeypatch):
    import huggingface_hub

    from yeto.rl.engine import init_from_release

    calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kw: calls.append(kw) or "/hub/snap")
    monkeypatch.setenv("YETO_MODEL_STORE_HIT", "1")
    assert init_from_release._base_path("org/model", REV) == "/hub/snap"
    monkeypatch.delenv("YETO_MODEL_STORE_HIT")
    init_from_release._base_path("org/model", REV)
    assert [c["local_files_only"] for c in calls] == [True, False]
    source = Path("yeto/rl/adapters/miles/island_entry.py").read_text()
    assert 'local_files_only=bool(os.environ.get("YETO_MODEL_STORE_HIT"))' in source


def test_cli_flags_reach_the_launcher_args():
    from tests.test_rl_engine_selection import _cli

    args = _cli(["--model-store", f"local://{STORE}", "--model-store-source", "s3://bkt/models/K",
                 "--model-store-shards", "stage"])
    assert (args.model_store, args.model_store_source, args.model_store_shards) == \
        (f"local://{STORE}", "s3://bkt/models/K", "stage")
    assert _cli([]).model_store_shards == "auto"


def test_the_launch_preflight_checks_the_model_store_source_not_the_hub(fake_s3, monkeypatch):
    from yeto.launch import model_check

    for k, v in fake_s3.env().items():
        monkeypatch.setenv(k, v)
    args = SimpleNamespace(model=MODEL, model_revision=REV, model_store_source="s3://bkt/models/tiny")

    def no_hub(*a):
        raise AssertionError("the Hub was asked")

    record = model_check.trainer_model_check(args, hub=no_hub)
    assert record["source"] == "model-store-source" and record["problems"] == [] and record["warnings"] == []
    del fake_s3.objects[f"models/tiny/{SHARED_FILE}"]
    [problem] = model_check.trainer_model_check(args, hub=no_hub)["problems"]
    assert "1 of 4 weight shards" in problem and SHARED_FILE in problem
    unreachable = SimpleNamespace(model=MODEL, model_revision=REV, model_store_source="/nonexistent/nfs/m")
    record = model_check.trainer_model_check(unreachable, hub=no_hub)
    assert record["problems"] == [] and "cannot list --model-store-source" in record["warnings"][0]


def test_a_node_without_a_usable_python3_runs_the_tool_in_the_trainer_image(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python3").write_text("#!/bin/sh\nexit 1\n")  # e.g. python 3.6: the version probe fails
    (bin_dir / "docker").write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n" "$@" > {tmp_path}/docker-argv\n'
        'for a in "$@"; do case "$a" in *:/yeto-weights-sync.py:ro) cp "${a%%:*}" '
        f'{tmp_path}/tool.py;; esac; done\n')
    for f in ("python3", "docker"):
        (bin_dir / f).chmod(0o755)
    node = b300_nodes(1, mounts=lambda i: {str(tmp_path / "nvme"): STORE})[0]
    plan = sw.WeightsPlan(model=MODEL, revision=REV, store=STORE, source="s3://bkt/m",
                          env={"YETO_WEIGHTS_S3_ACCESS_KEY_ID": "k"})
    script = sw.node_script(plan, node, None, image="ghcr.io/o/img@sha256:" + "a" * 64)
    subprocess.run(["/bin/bash", "-c", script], check=True, env={"PATH": f"{bin_dir}:/usr/bin:/bin"})
    argv = (tmp_path / "docker-argv").read_text().split("\n")
    assert argv[:3] == ["run", "--rm", "--network"] and "--user" in argv
    assert argv[argv.index("-e") + 1] == "YETO_WEIGHTS_S3_ACCESS_KEY_ID" and "k" not in argv
    at = argv.index("--entrypoint")
    assert argv[at + 1:at + 5] == ["python3", "ghcr.io/o/img@sha256:" + "a" * 64, "-I", "/yeto-weights-sync.py"]
    assert f"{tmp_path / 'nvme'}:{tmp_path / 'nvme'}" in argv and argv[at + 5] == "sync"
    assert (tmp_path / "tool.py").read_text().rstrip() == sw.TOOL.read_text().rstrip()
    assert (tmp_path / "nvme").is_dir()
