"""RL full-parameter asynchronous rounds on a fragment-sharded syncer head.

The launcher's actor syncer command with ``--syncer-shards 2`` (two real
Rust syncer processes under one supervisor) is driven by two fake islands,
each holding a ``ShardedSyncerClient``; the same islands then drive the
unsharded command. Every shard runs the unsharded step schedule, so PULLs,
fragment versions, the final manifest and the final weights must be equal.
No GPU, no torch, no Ray.
"""

from __future__ import annotations

import os
import shlex
import shutil
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from test_rl_algorithm_provenance import _fake_sky, _no_modal_listing  # noqa: F401
from test_rl_full_parameter_async import FULL, _plan
from yeto.fragments import MERGE_AVG, Fragment, FragmentLayout
from yeto.protocol import DTYPE_F32, SyncerClient
from yeto.syncer_shards import ShardedSyncerClient

ROOT = Path(__file__).resolve().parent.parent
CARGO = shutil.which("cargo") or str(Path.home() / ".cargo/bin/cargo")
NUMEL = 3
P = 4
SYNCER_PORT = 29400
needs_cargo = pytest.mark.skipif(not Path(CARGO).exists(), reason="needs cargo to build the syncer")


@pytest.fixture(scope="module")
def syncer_binary() -> Path:
    subprocess.run([CARGO, "build", "-q"], cwd=ROOT / "syncer", check=True)
    return ROOT / "syncer/target/debug/yeto-syncer"


@pytest.fixture
def _hmac_key(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "sharded-rounds-test-key")


def _async_args(shards: int, total_steps: int, *extra: str):
    # These tests compare the shards' final f32 cut with one syncer's.
    return _plan(FULL[:-1] + (str(P), "--rl-sync-full-parameter-rounds", "async",
                              "--total-steps", str(total_steps),
                              "--syncer-shards", str(shards), "--rl-sync-final-cut", "download", *extra))


def _free_ports(count: int) -> list[int]:
    socks = [socket.socket() for _ in range(count)]
    try:
        for sock in socks:
            sock.bind(("127.0.0.1", 0))
        return [sock.getsockname()[1] for sock in socks]
    finally:
        for sock in socks:
            sock.close()


def _layout() -> FragmentLayout:
    return FragmentLayout([Fragment(MERGE_AVG, [(f"actor.w{i}", NUMEL)]) for i in range(P)])


def _initial(fid: int) -> np.ndarray:
    return np.asarray([float(fid) + 0.25 * i for i in range(NUMEL)], dtype=np.float32)


def _delta(learner: int, step: int) -> np.ndarray:
    return np.asarray([0.5 + 0.125 * learner + 0.03125 * step + 0.001 * i for i in range(NUMEL)],
                      dtype=np.float32)


def _start(binary: Path, home: Path, shards: int, total_steps: int, extra=(), syncer_flags=""):
    """The launcher's actor syncer under ``home`` (its ``~``) on free ports."""
    from yeto import launcher

    args, _ = _async_args(shards, total_steps, *extra)
    command = launcher.role_syncer_commands(args, 2, binary=str(binary))["actor"]
    ports = _free_ports(shards)
    for index, port in enumerate(ports):
        command = command.replace(f"--port {SYNCER_PORT + index} ", f"--port {port} ")
    if syncer_flags:
        command = command.replace(" --event-tape", f" {syncer_flags} --event-tape")
    (home / "yeto-output").mkdir(parents=True)
    env = {**os.environ, "HOME": str(home)}
    process = subprocess.Popen(["bash", "-c", command], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, env=env)
    return process, ports, launcher.rl_syncer_profile(args, 2).sha256, command


def _client(learner: int, ports: list[int], profile_hash: str):
    kwargs = dict(dtype=DTYPE_F32, num_streams=1, connect_timeout=10, max_reconnects=0,
                  finalization_timeout=30, syncer_profile_hash=bytes.fromhex(profile_hash))
    addrs = [("127.0.0.1", port) for port in ports]
    if len(addrs) == 1:
        return SyncerClient(addrs[0], learner, _layout(), **kwargs)
    return ShardedSyncerClient(addrs, learner, _layout(), **kwargs)


def _drive(binary: Path, home: Path, shards: int, *, total_steps: int, push, extra=(),
           syncer_flags=""):
    """Run one head with two fake islands; return (finals, manifest, pulls, progress).

    ``push(learner, permit)`` returns ``(c_steps, c_tokens, delta)`` or None
    (stay silent). Every island answers against the strict round-robin base
    ``max(0, step - P)``, as the role stream does."""
    process, ports, profile_hash, _ = _start(binary, home, shards, total_steps, extra,
                                             syncer_flags)
    clients = [_client(learner, ports, profile_hash) for learner in (0, 1)]
    pulls: list[list[tuple[int, int]]] = [[], []]

    def poll():
        for learner, client in enumerate(clients):
            client.check_health()
            client.drain_updates()
            for permit in client.drain_pulls():
                if (permit.global_step, permit.fragment_id) in pulls[learner]:
                    continue
                pulls[learner].append((permit.global_step, permit.fragment_id))
                answer = push(learner, permit)
                if answer is None:
                    continue
                c_steps, c_tokens, delta = answer
                client.push_fragment(permit.fragment_id, permit.global_step, permit.round_attempt,
                                     max(0, permit.global_step - P), permit.global_step,
                                     c_steps, c_tokens, delta.tobytes())
        return all(client.finalizing.is_set() for client in clients)

    finals, manifests = [], []
    try:
        for client in clients:
            client.start()
        assert clients[0].send_initial_fragments(lambda fid: [_initial(fid).tobytes()])
        deadline = time.monotonic() + 90
        while not poll():
            assert time.monotonic() < deadline, "the head never finalized"
            time.sleep(0.005)
        progress = clients[0].send_progress()
        for client in clients:
            manifest, fragments = client.wait_for_final_fragments(timeout=30)
            finals.append({f.fragment_id: np.frombuffer(bytes(f.data), np.float32)[-NUMEL:].copy()
                           for f in fragments})
            manifests.append(manifest)
            client.acknowledge_finalization(manifest)
        assert process.wait(timeout=20) == 0
    finally:
        for client in clients:
            client.close()
        if process.poll() is None:
            process.kill()
        log, _ = process.communicate(timeout=10)
    assert manifests[0] == manifests[1], log
    for fid in range(P):
        assert np.array_equal(finals[0][fid], finals[1][fid])  # one global cut
    return finals[0], manifests[0], [sorted(p) for p in pulls], progress, log


def _assert_same_run(sharded, single):
    finals, manifest, pulls, *_ = sharded
    ref_finals, ref_manifest, ref_pulls, *_ = single
    assert manifest == ref_manifest
    assert pulls == ref_pulls
    for fid in range(P):
        assert np.array_equal(finals[fid], ref_finals[fid]), fid


def _all_islands(learner, permit):
    return 1 + learner, 10 * (learner + 1), _delta(learner, permit.global_step)


def _only_island_zero(learner, permit):
    return None if learner == 1 else (1, 7, _delta(0, permit.global_step))


# -- the real Rust syncer ---------------------------------------------------------


@needs_cargo
@pytest.mark.usefixtures("_hmac_key")
@pytest.mark.parametrize(("extra", "push"), [
    ((), _all_islands),
    (("--rl-sync-learner-weight", "tokens2-over-steps"), _all_islands),
    (("--rl-sync-quorum", "1"), _only_island_zero),
], ids=["equal", "tokens2", "quorum1-absent-island"])
def test_two_shards_equal_one_syncer(syncer_binary, tmp_path, extra, push):
    total_steps = 3  # three outer rounds per fragment: momentum carries over
    sharded = _drive(syncer_binary, tmp_path / "sharded", 2, total_steps=total_steps,
                     push=push, extra=extra)
    single = _drive(syncer_binary, tmp_path / "single", 1, total_steps=total_steps,
                    push=push, extra=extra)
    _assert_same_run(sharded, single)
    finals, manifest, pulls, progress, log = sharded
    assert manifest.global_step == total_steps * P
    assert manifest.versions == tuple(range(total_steps * P - P + 1, total_steps * P + 1))
    assert [step for step, _ in pulls[0]] == list(range(1, total_steps * P + 1))
    assert all(fid == (step - 1) % P for step, fid in pulls[0])
    assert progress.bytes_sent > 0 and progress.connected
    assert "push rejected" not in log
    # Each shard checkpointed and taped only its own steps.
    home = tmp_path / "sharded" / "yeto-output"
    assert (home / "yeto-state.ckpt").is_file() and (home / "yeto-state.shard1.ckpt").is_file()
    for tape, owned in (("yeto-tape.jsonl", {0, 1}), ("yeto-tape.shard1.jsonl", {2, 3})):
        merged = [line for line in (home / tape).read_text().splitlines()
                  if '"responders"' in line]
        assert merged
        assert all(any(f'"fragment":{fid}' in line for fid in owned) for line in merged)


@needs_cargo
@pytest.mark.usefixtures("_hmac_key")
def test_two_shards_take_delayed_nesterov_and_membership_options(syncer_binary, tmp_path):
    """Delayed Nesterov keeps per-fragment state, so a shard holds exactly its
    fragments' part; leases are tracked per shard. Both runs stay equal."""
    extra = ("--outer-delayed-nesterov-period", "2", "--rl-sync-quorum", "1")
    flags = "--island-lease-s 60"
    sharded = _drive(syncer_binary, tmp_path / "sharded", 2, total_steps=4,
                     push=_only_island_zero, extra=extra, syncer_flags=flags)
    single = _drive(syncer_binary, tmp_path / "single", 1, total_steps=4,
                    push=_only_island_zero, extra=extra, syncer_flags=flags)
    _assert_same_run(sharded, single)


# -- launcher ---------------------------------------------------------------------


def _flags(command: str, flag: str) -> list[str]:
    tokens = shlex.split(command.replace(";", " ; "))
    return [tokens[i + 1] for i, token in enumerate(tokens) if token == flag]


def test_launcher_runs_one_actor_syncer_per_shard():
    from yeto import launcher

    args, plan = _async_args(2, 3)
    [actor] = plan["role_syncers"]
    command = actor["command"]
    assert _flags(command, "--port") == ["29400", "29401"]
    assert _flags(command, "--shard-index") == ["0", "1"]
    assert _flags(command, "--shard-count") == ["2", "2"]
    assert _flags(command, "--checkpoint-path") == [
        "~/yeto-output/yeto-state.ckpt", "~/yeto-output/yeto-state.shard1.ckpt"]
    assert _flags(command, "--event-tape") == [
        "~/yeto-output/yeto-tape.jsonl", "~/yeto-output/yeto-tape.shard1.jsonl"]
    assert [shard["port"] for shard in actor["shards"]] == [29400, 29401]
    # One profile for every shard: the shard count is not a merge semantic.
    single_args, single_plan = _async_args(1, 3)
    assert launcher.rl_syncer_profile(args, 2) == launcher.rl_syncer_profile(single_args, 2)
    assert "shards" not in single_plan["role_syncers"][0]
    assert launcher.syncer_ports(args) == [29400, 29401]
    assert launcher.syncer_command(args, 2).count("--shard-count 2") == 2


@pytest.mark.parametrize(("extra", "message"), [
    (FULL + ("--syncer-shards", "2"), "full-parameter-rounds async"),
    (FULL[:-1] + ("2", "--rl-sync-full-parameter-rounds", "async", "--syncer-shards", "3"),
     "--fragments >= 3"),
])
def test_launcher_refuses_shards_outside_asynchronous_rounds(extra, message, monkeypatch):
    from yeto.launcher import _prepare_rl_args
    from test_rl_full_parameter_async import _launcher_args  # strict control
    from test_rl_full_parameter_async import TWO_ISLANDS

    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "k3y")
    args = _launcher_args("ports", extra, gpu=TWO_ISLANDS)
    with pytest.raises(ValueError, match=message):
        _prepare_rl_args(args)


# -- island side ------------------------------------------------------------------


def test_island_turns_a_comma_separated_syncer_into_shards():
    from types import SimpleNamespace

    from yeto.rl.adapters.miles.island_entry import _syncer_address, _syncer_addresses, probe_syncers

    assert _syncer_addresses("h:1,h:2") == [("h", 1), ("h", 2)]
    with pytest.raises(ValueError, match="full-parameter-rounds async"):
        _syncer_address("h:1,h:2")
    seen = []

    class _Socket:
        def close(self):
            pass

    def connect(addr, timeout):
        seen.append(addr)
        return _Socket()

    probe_syncers(SimpleNamespace(syncer="h:1,h:2", critic_syncer=None), connect=connect)
    assert seen == [("h", 1), ("h", 2)]


def test_role_stream_config_checks_the_shard_list():
    from dataclasses import replace

    from test_rl_full_parameter_sync import _role_stream_config

    config = replace(_role_stream_config("actor", 30100), partial_quorum=True)
    first = config.syncer_addr
    sharded = replace(config, syncer_shards=(first, (first[0], first[1] + 1)))
    assert sharded.syncer_shards[0] == first
    with pytest.raises(ValueError, match="first being the syncer address"):
        replace(config, syncer_shards=((first[0], first[1] + 1), first))
    with pytest.raises(ValueError, match="first being the syncer address"):
        replace(config, syncer_shards=(first,))
    with pytest.raises(ValueError, match="asynchronous rounds"):
        replace(sharded, partial_quorum=False)


def test_sharded_client_routes_init_replay_and_merges_progress():
    from yeto.protocol import SendProgress
    from yeto.wan_streams import StreamSnapshot

    client = ShardedSyncerClient([("127.0.0.1", 1), ("127.0.0.1", 2)], 0, _layout(), DTYPE_F32,
                                 num_streams=1)
    sent: list[tuple[int, list[int]]] = []
    for index, shard in enumerate(client.shards):
        shard.send_initial_fragments = (
            lambda parts_for, index=index, shard=shard:
            sent.append((index, list(shard.owned_fragments))) or True)
    due = {0: False, 1: True}
    for index, shard in enumerate(client.shards):
        shard.init_replay_due = lambda index=index: due[index]
    assert client.send_initial_fragments(lambda fid: [b""])
    assert sent == [(0, [0, 1]), (1, [2, 3])]
    assert client.init_replay_due()
    sent.clear()
    assert client.send_initial_fragments(lambda fid: [b""])
    assert sent == [(1, [2, 3])]  # a replay goes only to the shard that lost INIT

    for index, shard in enumerate(client.shards):
        shard.send_progress = lambda index=index: SendProgress(10 * (index + 1), index, 1, True,
                                                               index)
        shard.stream_snapshots = lambda index=index: (StreamSnapshot(index, 0.0, 0, None),)
        shard._upload_drained_at = 5.0 + index
        shard.connection_generation = 100 + index
    assert client.send_progress() == SendProgress(30, 1, 2, True, 1)
    assert [s.bytes_sent for s in client.stream_snapshots()] == [0, 1]
    assert client.upload_drained_at == 6.0
    assert [client.connection_generation_of(fid) for fid in range(P)] == [100, 100, 101, 101]

    installed = []
    for shard in client.shards:
        shard.install_inbound_chunk_sink = installed.append
    sink = object()
    client.install_inbound_chunk_sink(sink)
    assert installed == [sink, sink]
    client.mark_initial_cut_received()
    assert all(shard._initial_cut_received.is_set() for shard in client.shards)


class _ShardedPullClient:
    """Two links: fragment 0 on shard 0, fragment 1 on shard 1."""

    def __init__(self):
        self.generations = [7, 9]
        self.pulls = []

    def connection_generation_of(self, fragment_id):
        return self.generations[fragment_id]

    def drain_pulls(self):
        pulls, self.pulls = self.pulls, []
        return pulls


def test_role_stream_drops_only_the_permits_of_the_shard_that_redialed():
    from dataclasses import replace

    from test_rl_full_parameter_sync import _role_stream_config
    from yeto.protocol import PullRequest
    from yeto.rl.adapters.miles.full_parameter.role_stream import MilesFullParameterRoleStream

    stream = MilesFullParameterRoleStream(
        replace(_role_stream_config("actor", 30100), partial_quorum=True))
    stream.client = _ShardedPullClient()
    stream.client.pulls = [PullRequest(0, 1, 1), PullRequest(1, 2, 1)]
    stream._queue_pulls()
    assert set(stream._pending) == {0, 1}
    stream._answered = {(1, 1): 7, (2, 1): 9}
    stream.client.generations[1] = 10  # shard 1 redialed
    stream._queue_pulls()
    assert stream._pending == {0: PullRequest(0, 1, 1)}
    assert stream._answered == {(1, 1): 7}
    # The resent PULL on the new link is accepted again.
    stream.client.pulls = [PullRequest(1, 2, 1)]
    stream._queue_pulls()
    assert stream._pending[1] == PullRequest(1, 2, 1)
