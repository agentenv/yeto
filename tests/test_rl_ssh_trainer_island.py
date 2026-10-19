"""S25 N7.3: trainer island on SSH nodes, W4A8 re-quantized releases, and the
head's view of unmerged shard records (N3.6)."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_rl_oss_store import _entry, _objects
from tests.test_rl_quant_requant import world  # noqa: F401 - fixture
from yeto.hw import card_tools
from yeto.launch.node_inventory import Node
from yeto.launch.ssh_trainer_island import (
    SshIslandError,
    SshOps,
    SshTrainerIslands,
    SshTrainerSpec,
    kill_script,
    remote_paths,
    requant_argv,
    ssh_binding,
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
    base = dict(device_family="ppu", card="PPU", cards_per_node=8, hbm_gib=144.0, host_memory_gib=2048.0,
                runtime="rootfs", fabric="roce", interface="eth0")
    return Node(name=name, host=host, **(base | kw))


def spec(nodes=None, **kw) -> SshTrainerSpec:
    return SshTrainerSpec(nodes=tuple(nodes or [node(), node("ppu1", "u@10.0.0.2")]), run_id="run1",
                          command=("-m", "trainer.entry", "--x", "a b"), workdir="/work/yeto", **kw)


# ---------------------------------------------------------------- spec and scripts
def test_spec_refuses_mixed_families_docker_and_empty():
    with pytest.raises(SshIslandError, match="device family"):
        spec([node(), node("n1", "h1", device_family="npu")])
    with pytest.raises(SshIslandError, match="docker"):
        spec([node(runtime="docker")])
    with pytest.raises(SshIslandError, match="at least one node"):
        SshTrainerSpec(nodes=(), run_id="r", command=("x",), workdir="/")


def test_spec_is_refused_when_the_memory_plan_does_not_fit():
    shape = ModelShape(params=313_000_000_000, largest_layer_params=7_400_000_000, num_layers=45,
                       hidden_size=4096, vocab_size=154_880, residual_streams=4)
    small = [node(hbm_gib=64.0, host_memory_gib=512.0), node("n1", "h1", hbm_gib=64.0, host_memory_gib=512.0)]
    with pytest.raises(MemoryPlanError, match="does not fit"):
        ssh_trainer_spec(small, shape=shape, plan_overrides={"seq_len": 8192}, run_id="r", command=("x",),
                         workdir="/")
    ok = ssh_trainer_spec([node(name=f"n{i}", host=f"h{i}") for i in range(8)], shape=shape,
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
    def __init__(self, card_rc=0):
        self.runs, self.starts, self.card_rc = [], [], card_rc

    def run(self, n, script, *, timeout=300.0):
        self.runs.append((n.name, script))
        rc = self.card_rc if "smi" in script else 0
        return subprocess.CompletedProcess([], rc, stdout="cards", stderr="")

    def start(self, n, script):
        self.starts.append((n.name, script))
        return FakeProc(["log line\n"])


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
class LocalOps(SshOps):
    """Runs the node scripts with local bash instead of SSH (one node only)."""

    def __init__(self, home: Path):
        self.home = home

    def command(self, n, script):
        return ["env", f"HOME={self.home}", "bash", "-c", script]


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


# ---------------------------------------------------------------- N3.6 unmerged shard records
def test_head_counts_shard_records_the_trainer_has_not_merged():
    objects = _objects()
    npu = FileSampleBuffer(objects, batch_size=1, index=shard_index("npu"))
    ppu = FileSampleBuffer(objects, batch_size=1, index=shard_index("ppu"))
    head = FileSampleBuffer(objects, batch_size=2)
    plain = CommittedBacklog(head, lambda: 0)
    view = CommittedBacklog(head, lambda: 0, unmerged=UnmergedShards(objects))
    npu.append(_entry("n0", version=1, appended_at=1.0))
    ppu.append(_entry("p0", version=2, appended_at=2.0))
    assert plain.backlog(3).samples == 0  # the gap N3.6 closes: a stopped trainer merges nothing
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
                                "scheme": "w4a8-int4-channel", "awq": "none"}
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
    out = store.root / W4A8_RELEASE_DIR / rec.path
    shards = sorted(p.name for p in original.glob("*.safetensors"))
    assert shards and sorted(p.name for p in out.glob("*.safetensors")) == shards
    for name in shards:
        assert (out / name).read_bytes() == (original / name).read_bytes()


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
