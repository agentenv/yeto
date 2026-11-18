"""Trainer island on SSH nodes, W4A8 re-quantized releases, and the
head's view of unmerged shard records."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_rl_oss_store import _entry, _objects
from tests.test_rl_quant_requant import world  # noqa: F401 - fixture
from yeto.hw import card_tools
from yeto.launch.node_inventory import InventoryError, Node, parse_node
from yeto.launch.ssh_trainer_island import (
    CUDA_RDMA_DOCKER_ARGS,
    READ_SCRIPT,
    SshIslandError,
    SshOps,
    SshTrainerIslands,
    SshTrainerSpec,
    kill_script,
    remote_paths,
    requant_argv,
    ssh_binding,
    cuda_docker_problems,
    ssh_trainer_spec,
    start_script,
)
from yeto.launch.trainer_island import TrainerIsland
from yeto.rl import event_echo
from yeto.rl.engine.policy_release import (
    FULL,
    W4A8_RELEASE_DIR,
    PolicyFollower,
    PolicyReleaseError,
    PolicyReleaseStore,
)
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import CommittedBacklog, FileSampleBuffer, UnmergedShards, shard_index
from yeto.rl.fullparam.memory import MemoryPlanError, ModelShape
from yeto.rl.quant.release import RequantReleaser


def node(name="ppu0", host="u@10.0.0.1", **kw) -> Node:
    base = dict(device_family="ppu", cpu_arch="x86_64", card="PPU", cards_per_node=8, hbm_gib=144.0, host_memory_gib=2048.0,
                runtime="rootfs", fabric="roce", interface="eth0")
    return Node(name=name, host=host, **(base | kw))


def spec(nodes=None, **kw) -> SshTrainerSpec:
    return SshTrainerSpec(nodes=tuple(nodes or [node(), node("ppu1", "u@10.0.0.2")]), run_id="run1",
                          command=("-m", "trainer.entry", "--x", "a b"), workdir="/work/yeto", **kw)


# ---------------------------------------------------------------- spec and scripts
def test_spec_refuses_mixed_families_docker_and_empty():
    with pytest.raises(SshIslandError, match="device family"):
        spec([node(), node("n1", "h1", device_family="npu")])
    with pytest.raises(SshIslandError, match="docker_args"):
        spec([node(runtime="docker", image="img")])
    with pytest.raises(SshIslandError, match="at least one node"):
        SshTrainerSpec(nodes=(), run_id="r", command=("x",), workdir="/")


def test_spec_is_refused_when_the_memory_plan_does_not_fit():
    shape = ModelShape(params=313_000_000_000, largest_layer_params=7_400_000_000, num_layers=45,
                       hidden_size=4096, vocab_size=154_880, residual_streams=4)
    small = [node(hbm_gib=64.0, host_memory_gib=512.0), node("n1", "h1", hbm_gib=64.0, host_memory_gib=512.0)]
    with pytest.raises(MemoryPlanError, match="does not fit"):
        ssh_trainer_spec(small, shape=shape, plan_overrides={"seq_len": 8192}, run_id="r", command=("x",),
                         workdir="/")
    ok = ssh_trainer_spec([node(name=f"n{i}", host=f"h{i}", address=f"10.0.0.{i + 1}") for i in range(8)], shape=shape,
                          plan_overrides={"seq_len": 8192, "offload": "full"}, run_id="r", command=("x",),
                          workdir="/")
    assert ok.memory["fits"] and ok.memory["label"].startswith("derived")


def test_start_script_runs_torchrun_with_rank_rendezvous_and_comm_env():
    s = spec(requant=requant_argv("oss://b/run", "/tpl"))
    paths = remote_paths("run1", "trainer", 3)
    script0 = start_script(s, 0, {"YETO_X": "1"}, paths)
    script1 = start_script(s, 1, {"YETO_X": "1"}, paths)
    for script in (script0, script1):
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)
        assert "--nnodes=2" in script and "--master-addr=10.0.0.1" in script and "--nproc-per-node=8" in script
        assert "export NCCL_SOCKET_IFNAME=eth0" in script and "export NCCL_IB_DISABLE=0" in script
        assert "export YETO_X=1" in script and "setsid bash -c" in script
        assert f"export YETO_DEVICE_FAMILY={s.nodes[0].device_family}" in script
    assert "--node-rank=0" in script0 and "--node-rank=1" in script1
    assert "yeto.rl.quant.release" in script0 and "--loop" in script0 and "--no-awq" in script0
    assert "yeto.rl.quant.release" not in script1
    assert "'a b'" in script0  # arguments are quoted


def test_env_cannot_override_the_inventory_device_family():
    family = spec().device_family
    other = "npu" if family != "npu" else "ppu"
    with pytest.raises(SshIslandError, match="inventory decides"):
        spec(env={"YETO_DEVICE_FAMILY": other})
    assert spec(env={"YETO_DEVICE_FAMILY": family}).device_family == family


def test_requant_argv_needs_both_awq_inputs():
    assert "--recipe" in requant_argv("s", "t", recipe="r.yaml", awq_scales="d")
    with pytest.raises(SshIslandError, match="both"):
        requant_argv("s", "t", recipe="r.yaml")


# ---------------------------------------------------------------- islands over fake SSH
class FakeProc:
    def __init__(self, lines=()):
        self.stdout = iter(lines)
        self.rc = None

    def poll(self):
        return self.rc

    def terminate(self):
        self.rc = -15


class FakeOps(SshOps):
    def __init__(self, card_rc=0, uname="x86_64"):
        self.runs, self.starts, self.card_rc, self.uname = [], [], card_rc, uname

    def run(self, n, script, *, timeout=300.0):
        self.runs.append((n.name, script))
        if script == "uname -m":
            return subprocess.CompletedProcess([], 0, stdout=self.uname + "\n", stderr="")
        rc = self.card_rc if "smi" in script else 0
        return subprocess.CompletedProcess([], rc, stdout="cards", stderr="")

    def start(self, n, script):
        self.starts.append((n.name, script))
        return FakeProc(["log line\n"])

    def fetch(self, n, script, target, *, timeout):
        self.fetches = [*getattr(self, "fetches", []), (n.name, script, target)]
        return 0


@pytest.fixture
def checked_cards(monkeypatch):
    seen = []
    monkeypatch.setattr(card_tools, "check_cards", lambda family, text, **kw: seen.append((family, kw)) or [])
    return seen


def test_launch_checks_cards_once_starts_every_node_and_releases_by_process_group(tmp_path, checked_cards):
    ops = FakeOps()
    islands = SshTrainerIslands(spec(), name="trainer", jobs_path=tmp_path / "jobs.json", ops=ops,
                                events_dir=tmp_path / "ev")
    job = islands.launch(1, None)
    assert [f for f, _ in checked_cards] == ["ppu", "ppu"] and [n for n, _ in ops.starts] == ["ppu0", "ppu1"]
    assert islands.alive(job)
    islands.launch(2, None)
    assert len(checked_cards) == 2  # preflight once
    assert json.loads((tmp_path / "jobs.json").read_text()) == {"1": "ssh:trainer-i1", "2": "ssh:trainer-i2"}
    assert islands.release(job) and not islands.alive(job)
    kills = [s for _, s in ops.runs if "kill -TERM" in s]
    assert len(kills) == 2 and all("trainer-i1/pgid" in s for s in kills)
    assert islands.release_incarnation(2) and not islands.release_incarnation(7)


def test_preflight_fails_when_no_card_tool_answers(tmp_path, checked_cards):
    islands = SshTrainerIslands(spec(), name="t", jobs_path=tmp_path / "j.json", ops=FakeOps(card_rc=127))
    with pytest.raises(SshIslandError, match="no card tool"):
        islands.launch(1, None)


@pytest.mark.parametrize("uname", ["aarch64", "ppc64le"])
def test_preflight_refuses_a_node_of_another_cpu_arch(tmp_path, checked_cards, uname):
    ops = FakeOps(uname=uname)
    islands = SshTrainerIslands(spec(), name="t", jobs_path=tmp_path / "j.json", ops=ops)
    with pytest.raises(SshIslandError, match=f"ppu0: uname -m reports '?{uname}'?.*cpu_arch x86_64"):
        islands.launch(1, None)
    assert not ops.starts and not checked_cards


def test_preflight_accepts_the_docker_spelling_of_an_arch(tmp_path, checked_cards):
    islands = SshTrainerIslands(spec(nodes=[node(cpu_arch="aarch64")]), name="t",
                                jobs_path=tmp_path / "j.json", ops=FakeOps(uname="aarch64"))
    islands.preflight()
    assert islands.checked


def test_ssh_binding_reads_a_sharded_store_by_key(tmp_path, checked_cards):
    args = SimpleNamespace(rollout_batch_size=2, rl_checkpoint_store="oss://bkt/runs/a",
                           rl_trainer_command_poll_seconds=None, tuning="full")
    island = TrainerIsland(name="trainer", cloud="ssh", ssh=spec())
    binding = ssh_binding(args, island, journal_dir=tmp_path, files=_objects())
    assert isinstance(binding.buffer, CommittedBacklog) and binding.buffer.unmerged is not None
    with pytest.raises(SshIslandError, match="oss://"):
        ssh_binding(SimpleNamespace(**{**vars(args), "rl_checkpoint_store": "/shared/x"}), island,
                    journal_dir=tmp_path)


# ---------------------------------------------------------------- a real process group, locally
# The node scripts target Linux nodes and start the island under setsid (util-linux; absent on macOS).
needs_setsid = pytest.mark.skipif(shutil.which("setsid") is None, reason="setsid not installed (the node scripts need it)")


class LocalOps(SshOps):
    """Runs the node scripts with local bash instead of SSH (one node only)."""

    def __init__(self, home: Path):
        self.home = home

    def command(self, n):
        return ["env", f"HOME={self.home}", "bash", "-c", READ_SCRIPT]


@needs_setsid
def test_start_and_kill_scripts_on_a_local_process_group(tmp_path):
    fake = tmp_path / "torchrun"
    event = {"island_id": "t", "time_unix": 1.0, "event": "x"}
    fake.write_text(f"#!/bin/bash\necho {shlex.quote(event_echo.PREFIX + json.dumps(event))}\nsleep 300 & wait\n")
    fake.chmod(0o755)
    s = SshTrainerSpec(nodes=(node(),), run_id="r", command=("-m", "x"), workdir=str(tmp_path), torchrun=str(fake))
    islands = SshTrainerIslands(s, name="t", jobs_path=tmp_path / "j.json", ops=LocalOps(tmp_path), kill_grace_s=5)
    islands.checked = True
    job = islands.launch(1, None)
    pgid_file = tmp_path / ".yeto/trainer-islands/r/t-i1/pgid"
    for _ in range(100):
        if pgid_file.exists() and islands.start_events():
            break
        time.sleep(0.1)
    assert islands.alive(job) and islands.start_events() == [event]
    pgid = int(pgid_file.read_text())
    assert islands.release(job)
    for _ in range(50):
        if subprocess.run(["kill", "-0", "--", f"-{pgid}"], capture_output=True).returncode:
            break
        time.sleep(0.1)
    else:
        pytest.fail("process group still alive")
    subprocess.run(["bash", "-n"], input=kill_script(remote_paths("r", "t", 1)), text=True, check=True)


# ---------------------------------------------------------------- unmerged shard records
def test_head_counts_shard_records_the_trainer_has_not_merged():
    objects = _objects()
    npu = FileSampleBuffer(objects, batch_size=1, index=shard_index("npu"))
    ppu = FileSampleBuffer(objects, batch_size=1, index=shard_index("ppu"))
    head = FileSampleBuffer(objects, batch_size=2)
    plain = CommittedBacklog(head, lambda: 0)
    view = CommittedBacklog(head, lambda: 0, unmerged=UnmergedShards(objects))
    npu.append(_entry("n0", version=1, appended_at=1.0))
    ppu.append(_entry("p0", version=2, appended_at=2.0))
    assert plain.backlog(3).samples == 0  # the gap this closes: a stopped trainer merges nothing
    stats = view.backlog(3)
    assert (stats.samples, stats.batches, stats.oldest_version) == (2, 1.0, 1)
    trainer = FileSampleBuffer(objects, batch_size=1, merge_shards=True)  # merges n0, p0
    npu.append(_entry("n1", version=3, appended_at=3.0))
    assert view.backlog(3).samples == 3  # 2 merged + 1 not merged; nothing counted twice
    trainer.refresh()
    assert view.backlog(3).samples == 3 and not view.backlog(3).data_ended
    npu.mark_data_end()
    ppu.mark_data_end()
    assert view.backlog(3).data_ended


# ---------------------------------------------------------------- W4A8 releases
def _bf16_release(releases: PolicyReleaseStore, version: int, tag: str, **kw):
    def write(d: Path) -> None:
        (d / "model.safetensors").write_text(f"bf16 {tag}")

    return releases.release(version, f"h{tag}", write, buffer_selected=version * 10, **kw)


def _fake_requant(calls):
    def run(source: Path, out: Path):
        calls.append(source)
        (out / "model.safetensors").write_text("w4a8 " + (source / "model.safetensors").read_text())
        return {"shards": 1}

    return run


def test_each_bf16_release_becomes_one_w4a8_release_the_sampler_follows(tmp_path):
    store = CheckpointStore(tmp_path / "store")
    bf16 = PolicyReleaseStore(store)
    w4a8 = PolicyReleaseStore(store, release_dir=W4A8_RELEASE_DIR)
    calls = []
    releaser = RequantReleaser(bf16, w4a8, _fake_requant(calls), log=lambda m: None)
    assert releaser.sync() is None  # nothing released yet
    _bf16_release(bf16, 1, "a", buffer_shards={"npu": 4})
    rec = releaser.sync()
    assert rec.version == 1 and rec.policy_hash == "ha+w4a8-int4-channel" and rec.buffer_selected == 10
    assert rec.buffer_shards == {"npu": 4}
    assert rec.derived_from == {"release_dir": "policy-releases", "seq": 1, "version": 1, "policy_hash": "ha",
                                "scheme": "w4a8-int4-channel", "awq": "none", "awq_source": None,
                                "expert_matrices": None}
    assert releaser.sync() is None and len(calls) == 1  # converted once
    loaded = []
    follower = PolicyFollower(PolicyReleaseStore(store, release_dir=W4A8_RELEASE_DIR),
                              lambda r, d: loaded.append((d / "model.safetensors").read_text()))
    assert follower.refresh().version == 1 and loaded == ["w4a8 bf16 a"]
    _bf16_release(bf16, 2, "b")
    assert releaser.sync().version == 2 and follower.refresh().version == 2 and loaded[-1] == "w4a8 bf16 b"
    assert PolicyReleaseStore(store).latest().version == 2  # the BF16 directory is untouched


def test_a_restarted_releaser_writes_nothing_for_a_converted_release(tmp_path):
    store = CheckpointStore(tmp_path / "s")
    bf16, w4a8 = PolicyReleaseStore(store), PolicyReleaseStore(store, release_dir=W4A8_RELEASE_DIR)
    _bf16_release(bf16, 1, "a")
    RequantReleaser(bf16, w4a8, _fake_requant([]), log=lambda m: None).sync()
    calls = []
    assert RequantReleaser(bf16, w4a8, _fake_requant(calls), log=lambda m: None).sync() is None and not calls


def test_lora_releases_and_same_directories_are_refused(tmp_path):
    store = CheckpointStore(tmp_path / "s")
    bf16 = PolicyReleaseStore(store)
    with pytest.raises(ValueError, match="same release directory"):
        RequantReleaser(bf16, PolicyReleaseStore(store), _fake_requant([]))
    bf16.release(1, "h", lambda d: (d / "adapter.bin").write_text("x"), weights="lora")
    releaser = RequantReleaser(bf16, PolicyReleaseStore(store, release_dir=W4A8_RELEASE_DIR), _fake_requant([]))
    with pytest.raises(PolicyReleaseError, match="full-parameter"):
        releaser.sync()
    assert FULL == "full"


def test_release_cli_one_pass_with_the_real_converter(world):
    """The CLI end to end on the tiny artefact of tests/test_rl_quant_requant.py (option R2):
    the W4A8 release of the untrained source equals the original artefact byte for byte."""
    store = CheckpointStore(world["tmp"] / "store")
    PolicyReleaseStore(store).release(
        1, "h1", lambda d: [os.link(p, d / p.name) for p in world["source"].iterdir()])
    original = world["original"]
    done = subprocess.run([sys.executable, "-m", "yeto.rl.quant.release", "--store", str(store.root),
                           "--template", str(world["template"]), "--recipe", str(original / "recipe.yaml"),
                           "--awq-scales", str(world["tmp"] / "awq-scales")], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1])})
    assert done.returncode == 0, done.stderr
    rec = PolicyReleaseStore(store, release_dir=W4A8_RELEASE_DIR).latest()
    assert rec.version == 1 and rec.derived_from["awq"] == "fixed-artefact"
    assert "recipe.yaml" in rec.derived_from["awq_source"] and rec.derived_from["expert_matrices"] == 6
    out = store.root / W4A8_RELEASE_DIR / rec.path
    shards = sorted(p.name for p in original.glob("*.safetensors"))
    assert shards and sorted(p.name for p in out.glob("*.safetensors")) == shards
    for name in shards:
        assert (out / name).read_bytes() == (original / name).read_bytes()


@needs_setsid
def test_island_end_closes_the_stream_and_runs_a_last_requant_pass(tmp_path):
    fake = tmp_path / "torchrun"
    fake.write_text("#!/bin/bash\nsleep 0.5\nexit 3\n")
    req = tmp_path / "requant"
    req.write_text(f"#!/bin/bash\necho \"$*\" >> {tmp_path}/requant.calls\n"
                   "[[ \" $* \" == *' --loop '* ]] && exec sleep 300\nexit 0\n")
    fake.chmod(0o755)
    req.chmod(0o755)
    s = SshTrainerSpec(nodes=(node(),), run_id="r", command=("x",), workdir=str(tmp_path), torchrun=str(fake),
                       requant=(str(req), "--store", "s"))
    islands = SshTrainerIslands(s, name="t", jobs_path=tmp_path / "j.json", ops=LocalOps(tmp_path), kill_grace_s=5)
    islands.checked = True
    job = islands.launch(1, None)
    for _ in range(100):
        if not islands.alive(job):
            break
        time.sleep(0.1)
    assert not islands.alive(job)
    state = tmp_path / ".yeto/trainer-islands/r/t-i1"
    assert (state / "rc").read_text().strip() == "3"
    calls = (tmp_path / "requant.calls").read_text().splitlines()
    assert calls == ["--store s --loop --poll-seconds 10", "--store s"]


def test_ssh_argv_never_carries_the_start_script_secrets(monkeypatch, tmp_path):
    """The start script exports the task's secrets (store key, HF token): it goes to
    the node on ssh's stdin, never in an argv that ``ps`` (local or on the node) or a
    ``TimeoutExpired`` message shows."""
    secret = "s3cr3t-value-of-the-store-key"
    seen: list[tuple[list[str], str | None]] = []

    class Proc:
        def __init__(self, argv, **kw):
            seen.append((list(argv), None))
            self.stdin = self
            self.stdout = iter(())
            self.buf = []

        def write(self, text):
            self.buf.append(text)

        def close(self):
            seen[-1] = (seen[-1][0], "".join(self.buf))

        def poll(self):
            return 0

    def fake_run(argv, **kw):
        seen.append((list(argv), kw.get("input")))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "Popen", Proc)
    monkeypatch.setattr(subprocess, "run", fake_run)
    s = SshTrainerSpec(nodes=(node(),), run_id="r", command=("-m", "x"), workdir=str(tmp_path),
                       env={"AWS_SECRET_ACCESS_KEY": secret})
    script = start_script(s, 0, {}, remote_paths("r", "t", 1))
    assert secret in script
    ops = SshOps()
    ops.start(s.nodes[0], script)
    ops.run(s.nodes[0], script)
    assert len(seen) == 2
    for argv, sent in seen:
        assert secret not in " ".join(argv)
        assert sent == script


def test_the_remote_command_runs_the_whole_stdin_script_past_a_profile_that_reads_stdin(tmp_path):
    """The script is read before the login shell starts: a profile that reads stdin
    (a vendor /etc/profile.d script, a ~/.bash_profile) gets /dev/null, not the
    script's first line; quoting, heredocs and a long script come through intact."""
    (tmp_path / ".bash_profile").write_text("read -r eaten || true\nexport FROM_PROFILE=yes\n")
    value = "a'b\"c $HOME `x` \\ \nline2 $(echo no)"
    script = (f"echo first\nset -eu\nexport V={shlex.quote(value)}\n"
              "cat <<'EOT'\nliteral $V\nEOT\n"
              'printf "%s|%s\\n" "$FROM_PROFILE" "$V"\n'
              "read -r more && echo leak || echo eof\n"
              + "x=0\n" + "x=$((x+1))\n" * 20000 + "echo $x\n")
    done = subprocess.run(["env", f"HOME={tmp_path}", "bash", "-c", READ_SCRIPT], input=script,
                          text=True, capture_output=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert done.stdout == f"first\nliteral $V\nyes|{value}\neof\n20000\n"


# ---------------------------------------------------------------- NVIDIA bare-metal nodes (6 x 8 B300)
def b300(i=0, **kw) -> Node:
    base = {"device_family": "cuda", "cpu_arch": "x86_64", "card": "B300", "cards_per_node": 8,
            "hbm_gib": 288.0, "host_memory_gib": 2048.0, "runtime": "docker", "fabric": "ib",
            "interface": "bond0", "address": f"10.1.0.{10 + i}",
            "image": "ghcr.io/o/yeto-miles-k31@sha256:" + "a" * 64, "docker_args": CUDA_RDMA_DOCKER_ARGS}
    return Node(name=f"b300-{i}", host=f"ops@10.0.0.{10 + i}", **(base | kw))


def b300_spec(nodes=None, **kw) -> SshTrainerSpec:
    return SshTrainerSpec(nodes=tuple(nodes or [b300(i) for i in range(6)]), run_id="k31",
                          run_script="cd ~/sky_workdir && ray start --head", workdir="/opt/yeto", **kw)


def test_inventory_env_is_validated_and_exported_after_the_fabric_defaults():
    raw = {"name": "b300-0", "host": "ops@10.0.0.10", "device_family": "cuda", "cpu_arch": "x86_64",
           "card": "B300", "cards_per_node": 8, "hbm_gib": 288, "host_memory_gib": 2048, "fabric": "roce",
           "interface": "bond0", "runtime": "rootfs",
           "env": {"NCCL_IB_GID_INDEX": 3, "NCCL_IB_HCA": "mlx5_0,mlx5_1"}}
    parsed = parse_node(raw, index=0)
    assert parsed.env == {"NCCL_IB_GID_INDEX": "3", "NCCL_IB_HCA": "mlx5_0,mlx5_1"}
    for bad in ({"NCCL IB": "x"}, {"X": True}, {"YETO_DEVICE_FAMILY": "npu"}, ["NCCL_IB_HCA"]):
        with pytest.raises(InventoryError, match="env"):
            parse_node({**raw, "env": bad}, index=0)
    nodes = [b300(i, env={"NCCL_IB_HCA": "mlx5_0", "NCCL_DEBUG": "INFO"}) for i in range(2)]
    script = start_script(b300_spec(nodes, env={"NCCL_DEBUG": "TRACE"}), 1, {}, remote_paths("k31", "t", 1),
                          container="c1")
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    assert "export NCCL_IB_HCA=mlx5_0" in script and "export NCCL_IB_DISABLE=0" in script
    assert "export NCCL_DEBUG=TRACE" in script and "NCCL_DEBUG=INFO" not in script  # the run's env wins
    assert "-e NCCL_IB_HCA" in script  # reaches the container by name


def test_cuda_docker_nodes_need_gpus_and_on_rdma_the_verbs_devices_and_memlock():
    b300_spec()  # the recommended flags pass
    assert cuda_docker_problems(["--gpus=all", "-v", "/dev/infiniband:/dev/infiniband",
                                 "--ulimit=memlock=-1"], rdma=True) == []
    assert cuda_docker_problems(["--runtime", "nvidia", "--privileged", "--ulimit", "memlock=-1:-1"],
                                rdma=True) == []
    with pytest.raises(SshIslandError, match="the GPUs"):
        b300_spec([b300(i, docker_args=("--device", "/dev/infiniband")) for i in range(2)])
    with pytest.raises(SshIslandError, match="RDMA devices.*locked-memory"):
        b300_spec([b300(i, docker_args=("--gpus", "all")) for i in range(2)])
    # one node, or a tcp fabric: the GPUs are enough
    b300_spec([b300(0, docker_args=("--gpus", "all"))])
    b300_spec([b300(i, docker_args=("--gpus", "all"), fabric="tcp") for i in range(2)])


def test_run_script_enters_the_workdir_through_sky_workdir():
    script = start_script(b300_spec(), 0, {}, remote_paths("k31", "t", 1), container="c0")
    assert '[ -e "$HOME/sky_workdir" ] || ln -s /opt/yeto "$HOME/sky_workdir"' in script
    paths = remote_paths("k31", "t", 1)
    for rank in range(6):  # the run script reads the inventory addresses in rank order
        assert f"export SKYPILOT_NODE_RANK={rank}" in start_script(b300_spec(), rank, {}, paths, container="c")


class PullOps(FakeOps):
    def __init__(self, pull_rc=0, pull_err=""):
        super().__init__()
        self.pull_rc, self.pull_err = pull_rc, pull_err

    def run(self, n, script, *, timeout=300.0):
        if script.startswith("docker pull"):
            self.runs.append((n.name, script))
            return subprocess.CompletedProcess([], self.pull_rc, stdout="", stderr=self.pull_err)
        return super().run(n, script, timeout=timeout)


def test_preflight_pulls_the_image_on_every_docker_node(tmp_path, checked_cards):
    ops = PullOps()
    SshTrainerIslands(b300_spec(), name="t", jobs_path=tmp_path / "j.json", ops=ops).preflight()
    pulls = [n for n, s in ops.runs if s.startswith("docker pull -q ghcr.io/o/yeto-miles-k31@sha256:")]
    assert sorted(pulls) == [f"b300-{i}" for i in range(6)]  # the nodes are checked in parallel
    err = ("failed to register layer: openat opt/sglang/lib/python3.12/site-packages/"
           "sglang-0.5.24.dev80+g35708f7.dist-info/INSTALLER: not a directory")
    islands = SshTrainerIslands(b300_spec(), name="t", jobs_path=tmp_path / "j.json", ops=PullOps(1, err))
    with pytest.raises(SshIslandError, match="b300-0: docker pull .* failed.*make_layer.py"):
        islands.launch(1, None)
    # rootfs nodes pull nothing
    ops = PullOps()
    SshTrainerIslands(spec(), name="t", jobs_path=tmp_path / "j.json", ops=ops).preflight()
    assert not [s for _, s in ops.runs if s.startswith("docker pull")]


def test_an_overlay_rejected_pinned_image_names_its_repacked_digest(tmp_path, checked_cards):
    """The K3.1 FULL_E image fails on overlay stores; the pull error names the repack of
    the same image (pins.OVERLAY_SAFE_DIGESTS), not only the generic rebuild hint."""
    from yeto.rl.adapters.miles import pins

    image = pins.MILES_K31_FULL_IMAGE.removeprefix("docker:")
    err = ("mkparent /var/lib/containerd/io.containerd.snapshotter.v1.overlayfs/snapshots/2/fs/opt/sglang/"
           "lib/python3.12/site-packages/sglang-0.5.24.dev80+g35708f7.dist-info: not a directory")
    islands = SshTrainerIslands(b300_spec([b300(i, image=image) for i in range(6)]), name="t",
                                jobs_path=tmp_path / "j.json", ops=PullOps(1, err))
    with pytest.raises(SshIslandError) as raised:
        islands.preflight()
    fixed = pins.MILES_K31_FULL_OVERLAYFIX_IMAGE.removeprefix("docker:")
    assert f"layer repacked: {fixed})" in str(raised.value) and "make_layer.py" not in str(raised.value)


class FlakyOps(FakeOps):
    """``down``: node name -> how many more SSH calls fail (rc 255, as a node
    without power or network); ``kills_fail``: node names whose kill script fails."""

    def __init__(self, down=None, kills_fail=()):
        super().__init__()
        self.down = dict(down or {})
        self.kills_fail = set(kills_fail)

    def run(self, n, script, *, timeout=300.0):
        if self.down.get(n.name, 0) > 0:
            self.down[n.name] -= 1
            self.runs.append((n.name, script))
            return subprocess.CompletedProcess([], 255, stdout="", stderr="ssh: connect to host: No route to host")
        if "kill -TERM" in script and n.name in self.kills_fail:
            self.runs.append((n.name, script))
            return subprocess.CompletedProcess([], 255, stdout="", stderr="No route to host")
        return super().run(n, script, timeout=timeout)


def _flaky_islands(tmp_path, ops, **kw):
    now = [0.0]
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        now[0] += s

    islands = SshTrainerIslands(spec(), name="t", jobs_path=tmp_path / "j.json", ops=ops, clock=lambda: now[0],
                                sleep=sleep, node_poll_s=15.0, **kw)
    return islands, sleeps


def test_a_later_start_waits_for_a_node_that_is_unreachable(tmp_path, checked_cards, capsys):
    ops = FlakyOps()
    islands, sleeps = _flaky_islands(tmp_path, ops)
    islands.launch(0, None)
    ops.down = {"ppu1": 2}  # rebooting: two probes fail
    ops.starts.clear()
    islands.launch(1, None)
    assert sleeps == [15.0, 15.0] and [n for n, _ in ops.starts] == ["ppu0", "ppu1"]
    assert "waiting for node(s) ['ppu1'], unreachable over SSH" in capsys.readouterr().out


def test_a_later_start_refuses_a_node_still_unreachable_after_the_wait(tmp_path, checked_cards):
    ops = FlakyOps()
    islands, _ = _flaky_islands(tmp_path, ops, node_wait_s=60.0)
    islands.launch(0, None)
    ops.down = {"ppu1": 100}
    ops.starts.clear()
    with pytest.raises(SshIslandError, match=r"node\(s\) \['ppu1'\] unreachable over SSH for 60 s.*No route"):
        islands.launch(1, None)
    assert ops.starts == []  # no node got the incarnation


def test_an_unconfirmed_release_is_stopped_again_before_the_next_start(tmp_path, checked_cards):
    ops = FlakyOps(kills_fail={"ppu1"})
    islands, _ = _flaky_islands(tmp_path, ops)
    job = islands.launch(0, None)
    assert islands.release(job) is False and islands.unreleased == {0}
    ops.starts.clear()
    with pytest.raises(SshIslandError, match="incarnation 0 is not confirmed stopped on every node"):
        islands.launch(1, None)  # the node answers again but still cannot stop it
    assert ops.starts == []
    ops.kills_fail.clear()
    islands.launch(2, None)
    kills = [(n, s) for n, s in ops.runs if "kill -TERM" in s and "t-i0/pgid" in s]
    assert [n for n, _ in kills][-2:] == ["ppu0", "ppu1"] and islands.unreleased == set()
    assert [n for n, _ in ops.starts] == ["ppu0", "ppu1"]
