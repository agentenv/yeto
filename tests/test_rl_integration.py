"""Real protocol-v4 clients against the fixed-roster Rust RL scheduler."""

from __future__ import annotations

import socket
import subprocess
import threading
import time
from pathlib import Path

import pytest
import torch

from yeto.export import parse_checkpoint
from yeto.protocol import DTYPE_F32, SyncerClient
from yeto.rl.core import CanonicalTensorSpec, build_avg_layout
from yeto.rl.bridge import BridgeConfig, StrictRlBridge
from yeto.rl.core import LocalRoundStats, canonical_state
from yeto.tensor_io import pack_tensor, unpack_fragment

ROOT = Path(__file__).resolve().parent.parent
MODEL_REVISION = "a" * 40
LORA_CONFIG_HASH = "b" * 64


def _state(version, tensors):
    return canonical_state(
        version,
        tensors,
        base_model_revision=MODEL_REVISION,
        lora_config_hash=LORA_CONFIG_HASH,
    )


@pytest.fixture(scope="module")
def syncer_binary():
    subprocess.run(["cargo", "build", "-q"], cwd=ROOT / "syncer", check=True)
    return ROOT / "syncer/target/debug/yeto-syncer"


def _port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _layout():
    return build_avg_layout(
        (
            CanonicalTensorSpec(
                "base_model.model.layer.lora_A.weight",
                (1, 2),
                "float32",
                2,
            ),
        )
    )


def _start(binary, port, checkpoint, rounds, *, learners=2, event_tape=None):
    return subprocess.Popen(
        [
            str(binary),
            "--port",
            str(port),
            "--learners",
            str(learners),
            "--quorum",
            str(learners),
            "--grace-ms",
            "0",
            "--pipeline",
            "1",
            "--sync-interval-steps",
            "0",
            "--delta-correction",
            "none",
            "--total-steps",
            str(rounds),
            "--outer-lr",
            "1",
            "--outer-momentum",
            "0",
            "--quorum-timeout-s",
            "10",
            "--checkpoint-path",
            str(checkpoint),
            "--checkpoint-every",
            "1",
            "--resume",
            "--max-base-lag",
            "0",
            "--learner-weight",
            "equal",
            *(
                ["--event-tape", str(event_tape)]
                if event_tape is not None
                else []
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _client(port, learner_id, layout):
    client = SyncerClient(
        ("127.0.0.1", port),
        learner_id,
        layout,
        dtype=DTYPE_F32,
        num_streams=0,
        connect_timeout=10,
    )
    client.start()
    return client


def _wait_item(drain, predicate=lambda item: True, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for item in drain():
            if predicate(item):
                return item
        time.sleep(0.02)
    raise TimeoutError("protocol item did not arrive")


def _wait_checkpoint(path, version, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            try:
                checkpoint = parse_checkpoint(path)
            except (OSError, ValueError):
                pass
            else:
                if checkpoint.global_step == version:
                    return checkpoint
        time.sleep(0.02)
    raise TimeoutError(f"checkpoint v{version} did not appear")


def _push(client, step, base, values, *, c_tokens=1):
    client.push_fragment(
        0,
        step,
        1,
        base,
        step,
        1,
        c_tokens,
        pack_tensor(torch.tensor(values, dtype=torch.float32), DTYPE_F32),
    )


def _close_all(*clients):
    for client in clients:
        client.close()


class _FakeMiles:
    def __init__(self, delta, learner_id=0):
        self.delta = torch.tensor(delta, dtype=torch.float32).reshape(1, 2)
        self.learner_id = learner_id
        self.current = _state(
            0,
            {"base_model.model.layer.lora_A.weight": torch.zeros(1, 2)},
        )
        self.local = self.current
        self.applied = []

    def initialize(self):
        return self.current

    def apply_global_policy(self, state):
        self.current = state
        self.local = state
        self.applied.append(state.policy_version)

    def run_local_round(
        self,
        *,
        expected_policy_version,
        groups,
        samples_per_group,
        optimizer_steps,
    ):
        assert self.current.policy_version == expected_policy_version
        self.local = _state(
            expected_policy_version,
            {name: value + self.delta for name, value in self.current.tensors.items()},
        )
        return LocalRoundStats(
            island_id=self.learner_id,
            local_round_id=expected_policy_version + 1,
            base_policy_version=expected_policy_version,
            active_groups=groups,
            completed_groups=groups,
            cancelled_groups=0,
            completed_trajectories=groups * samples_per_group,
            action_tokens=groups * samples_per_group,
            tool_wait_seconds=0.0,
            group_p50_seconds=0.1,
            group_p95_seconds=0.1,
            group_p99_seconds=0.1,
            reward_mean=1.0,
            reward_std=0.0,
            zero_variance_group_ratio=1.0,
            mean_kl=None,
            ess_ratio=None,
            clip_fraction=0.0,
            delta_l2_norm=0.0,
            rollout_seconds=0.1,
            train_seconds=0.1,
        )

    def export_local_policy(self):
        return self.local

    def record_local_round(self, stats):
        self.stats = stats

    def shutdown(self):
        pass


class _BridgeThread(threading.Thread):
    def __init__(self, bridge):
        super().__init__(daemon=True)
        self.bridge = bridge
        self.result = None
        self.error = None

    def run(self):
        try:
            self.result = self.bridge.run()
        except BaseException as error:
            self.error = error


def test_two_fake_miles_islands_run_the_bridge_against_real_syncer(
    syncer_binary, tmp_path
):
    checkpoint_path = tmp_path / "state.ckpt"
    port = _port()
    process = _start(syncer_binary, port, checkpoint_path, rounds=1)
    runtimes = (_FakeMiles([1, 3], 0), _FakeMiles([3, 5], 1))
    threads = []
    try:
        for learner_id, runtime in enumerate(runtimes):
            bridge = StrictRlBridge(
                runtime,
                BridgeConfig(
                    syncer_addr=("127.0.0.1", port),
                    learner_id=learner_id,
                    global_rounds=1,
                    groups_per_round=1,
                    samples_per_group=2,
                    local_optimizer_steps=1,
                    wan_streams=0,
                    expected_specs=runtime.current.specs,
                    base_model_revision=runtime.current.base_model_revision,
                    lora_config_hash=runtime.current.lora_config_hash,
                    layout_hash=runtime.current.layout_hash,
                    event_tape=str(tmp_path / f"island-{learner_id}.jsonl"),
                ),
            )
            threads.append(_BridgeThread(bridge))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)
            assert not thread.is_alive()
            assert thread.error is None
            value = next(iter(thread.result.tensors.values()))
            assert torch.equal(value, torch.tensor([[2.0, 4.0]]))
        assert process.wait(timeout=10) == 0
        assert all(runtime.applied[-1] == 1 for runtime in runtimes)
    finally:
        for thread in threads:
            thread.bridge.client.close()
        if process.poll() is None:
            process.kill()
            process.wait()


def test_single_island_runs_the_real_syncer_parity_path(syncer_binary, tmp_path):
    checkpoint_path = tmp_path / "state.ckpt"
    port = _port()
    process = _start(
        syncer_binary,
        port,
        checkpoint_path,
        rounds=1,
        learners=1,
    )
    runtime = _FakeMiles([1, 3], 0)
    bridge = StrictRlBridge(
        runtime,
        BridgeConfig(
            syncer_addr=("127.0.0.1", port),
            learner_id=0,
            global_rounds=1,
            groups_per_round=1,
            samples_per_group=2,
            local_optimizer_steps=1,
            wan_streams=0,
            expected_specs=runtime.current.specs,
            base_model_revision=runtime.current.base_model_revision,
            lora_config_hash=runtime.current.lora_config_hash,
            layout_hash=runtime.current.layout_hash,
            event_tape=str(tmp_path / "island.jsonl"),
        ),
    )
    thread = _BridgeThread(bridge)
    try:
        thread.start()
        thread.join(timeout=15)
        assert not thread.is_alive()
        assert thread.error is None
        assert torch.equal(
            next(iter(thread.result.tensors.values())),
            torch.tensor([[1.0, 3.0]]),
        )
        assert runtime.stats.delta_l2_norm == pytest.approx(10**0.5)
        assert process.wait(timeout=10) == 0
        checkpoint = parse_checkpoint(checkpoint_path)
        assert checkpoint.ledger == {0: (1, 1, 1)}
        assert checkpoint.layout_hash == runtime.current.layout_hash
    finally:
        bridge.client.close()
        if process.poll() is None:
            process.kill()
            process.wait()


def test_fixed_roster_exact_base_duplicate_disconnect_and_manual_average(
    syncer_binary, tmp_path
):
    layout = _layout()
    checkpoint_path = tmp_path / "state.ckpt"
    port = _port()
    process = _start(syncer_binary, port, checkpoint_path, rounds=2)
    client0 = client1 = replacement1 = None
    try:
        client0 = _client(port, 0, layout)
        client0.send_init(0, pack_tensor(torch.zeros(2), DTYPE_F32))
        client1 = _client(port, 1, layout)
        for client in (client0, client1):
            update = _wait_item(client.drain_updates)
            assert update.version == 0
            pull = _wait_item(client.drain_pulls)
            assert (pull.global_step, pull.round_attempt) == (1, 1)

        _push(client0, 1, 0, [1, 3], c_tokens=100)
        _push(client0, 1, 0, [9, 9])  # duplicate logical ID is ignored
        client1.close()
        client1 = None
        time.sleep(0.2)
        assert parse_checkpoint(checkpoint_path).global_step == 0
        assert not [item for item in client0.drain_updates() if item.version == 1]

        replacement1 = _client(port, 1, layout)
        assert _wait_item(replacement1.drain_updates).version == 0
        assert _wait_item(replacement1.drain_pulls).global_step == 1
        _push(replacement1, 1, 0, [3, 5], c_tokens=1)

        updates = []
        for client in (client0, replacement1):
            update = _wait_item(client.drain_updates, lambda item: item.version == 1)
            updates.append(unpack_fragment(layout.fragments[0], update.data, DTYPE_F32))
            assert _wait_item(client.drain_pulls).global_step == 2
        assert all(torch.equal(value, torch.tensor([2.0, 4.0])) for value in updates)

        _push(client0, 2, 1, [2, 0])
        _push(replacement1, 2, 1, [0, 2])
        for client in (client0, replacement1):
            update = _wait_item(client.drain_updates, lambda item: item.version == 2)
            value = unpack_fragment(layout.fragments[0], update.data, DTYPE_F32)
            assert torch.equal(value, torch.tensor([3.0, 5.0]))

        finals = []
        for client in (client0, replacement1):
            manifest, fragments = client.wait_for_final_fragments(timeout=10)
            assert manifest.global_step == 2
            finals.append(
                unpack_fragment(layout.fragments[0], fragments[0].data, DTYPE_F32)
            )
            client.acknowledge_finalization(manifest, timeout=10)
        assert all(torch.equal(value, torch.tensor([3.0, 5.0])) for value in finals)
        assert process.wait(timeout=10) == 0

        checkpoint = parse_checkpoint(checkpoint_path)
        assert checkpoint.global_step == 2
        assert checkpoint.ledger == {0: (2, 2, 101), 1: (2, 2, 2)}
    finally:
        _close_all(*(c for c in (client0, client1, replacement1) if c is not None))
        if process.poll() is None:
            process.kill()
            process.wait()


def test_checkpoint_failure_happens_before_any_broadcast(syncer_binary, tmp_path):
    layout = _layout()
    checkpoint_path = tmp_path / "state.ckpt"
    port = _port()
    process = _start(syncer_binary, port, checkpoint_path, rounds=1)
    client0 = client1 = None
    try:
        client0 = _client(port, 0, layout)
        client0.send_init(0, pack_tensor(torch.zeros(2), DTYPE_F32))
        client1 = _client(port, 1, layout)
        for client in (client0, client1):
            assert _wait_item(client.drain_updates).version == 0
            assert _wait_item(client.drain_pulls).global_step == 1

        checkpoint_path.with_suffix(".tmp").mkdir()
        _push(client0, 1, 0, [1, 1])
        _push(client1, 1, 0, [1, 1])
        assert process.wait(timeout=10) != 0
        time.sleep(0.1)
        assert not [item for item in client0.drain_updates() if item.version == 1]
        assert not [item for item in client1.drain_updates() if item.version == 1]
        assert parse_checkpoint(checkpoint_path).global_step == 0
    finally:
        _close_all(*(c for c in (client0, client1) if c is not None))
        if process.poll() is None:
            process.kill()
            process.wait()


def test_stale_update_fails_strict_run_and_records_metric(syncer_binary, tmp_path):
    layout = _layout()
    checkpoint_path = tmp_path / "state.ckpt"
    event_tape = tmp_path / "events.jsonl"
    port = _port()
    process = _start(
        syncer_binary,
        port,
        checkpoint_path,
        rounds=2,
        event_tape=event_tape,
    )
    client0 = client1 = None
    try:
        client0 = _client(port, 0, layout)
        client0.send_init(0, pack_tensor(torch.zeros(2), DTYPE_F32))
        client1 = _client(port, 1, layout)
        for client in (client0, client1):
            assert _wait_item(client.drain_updates).version == 0
            assert _wait_item(client.drain_pulls).global_step == 1
        _push(client0, 1, 0, [1, 1])
        _push(client1, 1, 0, [1, 1])
        for client in (client0, client1):
            assert _wait_item(client.drain_updates).version == 1
            assert _wait_item(client.drain_pulls).global_step == 2

        _push(client0, 2, 0, [9, 9])
        assert process.wait(timeout=10) != 0
        assert parse_checkpoint(checkpoint_path).global_step == 1
        assert '"metric":"rejected_stale_updates"' in event_tape.read_text()
    finally:
        _close_all(*(c for c in (client0, client1) if c is not None))
        if process.poll() is None:
            process.kill()
            process.wait()


def test_rl_resume_layout_mismatch_is_a_strict_failure(syncer_binary, tmp_path):
    checkpoint_path = tmp_path / "state.ckpt"
    port = _port()
    process = _start(
        syncer_binary, port, checkpoint_path, rounds=1, learners=1
    )
    client = _client(port, 0, _layout())
    client.send_init(0, pack_tensor(torch.zeros(2), DTYPE_F32))
    _wait_checkpoint(checkpoint_path, 0)
    process.kill()
    process.wait()
    client.close()

    mismatched = build_avg_layout(
        (
            CanonicalTensorSpec(
                "base_model.model.other.lora_A.weight",
                (1, 2),
                "float32",
                2,
            ),
        )
    )
    event_tape = tmp_path / "events.jsonl"
    port = _port()
    process = _start(
        syncer_binary,
        port,
        checkpoint_path,
        rounds=1,
        learners=1,
        event_tape=event_tape,
    )
    client = _client(port, 0, mismatched)
    try:
        assert process.wait(timeout=10) != 0
        assert '"metric":"layout_hash_mismatch"' in event_tape.read_text()
    finally:
        client.close()
        if process.poll() is None:
            process.kill()
            process.wait()


def test_restart_before_and_after_commit_recovers_old_and_new_cut(
    syncer_binary, tmp_path
):
    layout = _layout()
    checkpoint_path = tmp_path / "state.ckpt"

    # Crash with one accepted result: only version zero is authoritative.
    port = _port()
    process = _start(syncer_binary, port, checkpoint_path, rounds=1)
    client0 = _client(port, 0, layout)
    client0.send_init(0, pack_tensor(torch.zeros(2), DTYPE_F32))
    client1 = _client(port, 1, layout)
    for client in (client0, client1):
        _wait_item(client.drain_updates)
        _wait_item(client.drain_pulls)
    _push(client0, 1, 0, [2, 2])
    process.kill()
    process.wait()
    _close_all(client0, client1)
    assert parse_checkpoint(checkpoint_path).global_step == 0

    # Resume and commit; kill as soon as the durable version-one file exists.
    port = _port()
    process = _start(syncer_binary, port, checkpoint_path, rounds=1)
    client0 = _client(port, 0, layout)
    client0.send_init(0, pack_tensor(torch.zeros(2), DTYPE_F32))
    client1 = _client(port, 1, layout)
    for client in (client0, client1):
        assert _wait_item(client.drain_updates).version == 0
        _wait_item(client.drain_pulls)
    _push(client0, 1, 0, [2, 2])
    _push(client1, 1, 0, [4, 4])
    _wait_checkpoint(checkpoint_path, 1)
    process.kill()
    process.wait()
    _close_all(client0, client1)
    checkpoint_path.with_suffix(".tmp").mkdir()

    # A second restart broadcasts/finalizes the committed new cut, without
    # asking either learner to merge version zero again.
    port = _port()
    process = _start(syncer_binary, port, checkpoint_path, rounds=1)
    client0 = _client(port, 0, layout)
    client0.send_init(0, pack_tensor(torch.zeros(2), DTYPE_F32))
    client1 = _client(port, 1, layout)
    try:
        for client in (client0, client1):
            update = _wait_item(client.drain_updates, lambda item: item.version == 1)
            value = unpack_fragment(layout.fragments[0], update.data, DTYPE_F32)
            assert torch.equal(value, torch.tensor([3.0, 3.0]))
            assert client.drain_pulls() == []
            manifest, _ = client.wait_for_final_fragments(timeout=10)
            client.acknowledge_finalization(manifest, timeout=10)
        assert process.wait(timeout=10) == 0
        checkpoint = parse_checkpoint(checkpoint_path)
        assert checkpoint.global_step == 1
        assert checkpoint.ledger == {0: (1, 1, 1), 1: (1, 1, 1)}
    finally:
        _close_all(client0, client1)
        if process.poll() is None:
            process.kill()
            process.wait()
