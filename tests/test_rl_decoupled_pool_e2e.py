"""Decoupled islands in an island pool against the real Rust syncer. CPU, no Ray.

Two decoupled bridges (``yeto.rl.decoupled``) on a syncer with membership
options: a soft deadline merges rounds without the island that did not push,
the heartbeat keeps an idle island's lease, a silent island's lease expires
and it re-JOINs (catch-up skips the versions it missed), and both islands
then finish the run on the final cut.
"""

from __future__ import annotations

import socket
import subprocess
import time
from pathlib import Path

import pytest
import torch

from yeto.rl.core import canonical_state
from yeto.rl.decoupled import DecoupledBridgeConfig, DecoupledRlBridge

ROOT = Path(__file__).resolve().parent.parent
MODEL_REVISION = "a" * 40
LORA_CONFIG_HASH = "b" * 64
LEASE_S = 1.5
SOFT_DEADLINE_S = 1
TOTAL_STEPS = 6  # three sweeps of two fragments


@pytest.fixture(scope="module")
def syncer_binary():
    subprocess.run(["cargo", "build", "-q"], cwd=ROOT / "syncer", check=True)
    return ROOT / "syncer/target/debug/yeto-syncer"


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _state(version, tensors):
    return canonical_state(version, tensors, base_model_revision=MODEL_REVISION,
                           lora_config_hash=LORA_CONFIG_HASH)


def _start_pool_syncer(binary, port, tmp_path, log):
    """The decoupled role syncer as the launcher builds it for a pool with
    --rl-sync-quorum-theta 1 --rl-sync-soft-deadline-s 1 --rl-sync-island-lease-s 1.5."""
    argv = [str(binary), "--port", str(port), "--learners", "2", "--quorum", "1",
            "--grace-ms", "0", "--pipeline", "2", "--sync-interval-steps", "2",
            "--delta-correction", "none", "--total-steps", str(TOTAL_STEPS),
            "--outer-lr", "0.7", "--outer-momentum", "0.9", "--quorum-timeout-s", "30",
            "--quorum-theta", "1.0", "--soft-deadline-s", str(SOFT_DEADLINE_S),
            "--island-lease-s", str(LEASE_S), "--syncer-epoch", "0",
            "--final-grace-s", "10", "--max-base-lag", "2", "--learner-weight", "equal",
            "--checkpoint-path", str(tmp_path / "pool.ckpt"), "--checkpoint-every", "1",
            "--event-tape", str(tmp_path / "tape.jsonl")]
    return subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)


def _bridge(initial, port, learner_id):
    return DecoupledRlBridge(initial, DecoupledBridgeConfig(
        syncer_addr=("127.0.0.1", port), learner_id=learner_id, total_fragment_steps=TOTAL_STEPS,
        num_fragments=2, pipeline=2, local_horizon=2, expected_specs=initial.specs,
        base_model_revision=MODEL_REVISION, lora_config_hash=LORA_CONFIG_HASH,
        canonical_layout_hash=initial.layout_hash, wan_streams=0,
        island_lease_s=LEASE_S, syncer_epoch=0))


def _wait(predicate, timeout=20.0, what="condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


class _Island:
    """One decoupled island: its local state and progress counters."""

    def __init__(self, bridge):
        self.bridge = bridge
        self.state = None
        self.steps = 0
        self.tokens = 0

    def take_initial_cut(self):
        cut = self.bridge.wait_for_initial_cut(optimizer_steps=0, action_tokens=0)
        self.bridge.commit_initial_cut(cut, optimizer_steps=0, action_tokens=0)
        self.state = cut.state

    def drain(self):
        batch = self.bridge.drain_broadcasts(self.state, optimizer_steps=self.steps,
                                             action_tokens=self.tokens)
        self.bridge.commit_broadcasts(batch, optimizer_steps=self.steps, action_tokens=self.tokens)
        self.state = batch.state
        return self.bridge.fragment_versions

    def train(self, offset):
        """One local round of H=2 optimizer steps: every value moves by ``offset``."""
        self.steps += 2
        self.tokens += 20
        self.state = _state(self.steps, {name: value + offset
                                         for name, value in self.state.tensors.items()})

    def submit(self):
        return self.bridge.submit_ready(self.state, optimizer_steps=self.steps,
                                        action_tokens=self.tokens)

    def push_both_fragments(self):
        fragments = set()
        _wait(lambda: fragments.update(s.fragment_id for s in self.submit()) or fragments == {0, 1},
              what=f"island {self.bridge.config.learner_id} pushes both fragments")


def _tape_kinds(path: Path) -> list[str]:
    import json

    if not path.exists():
        return []
    return [json.loads(line).get("kind") for line in path.read_text().splitlines() if line.strip()]


def test_decoupled_pool_soft_deadline_lease_and_rejoin(syncer_binary, tmp_path):
    initial = _state(0, {f"base_model.model.layer{i}.lora_A.weight": torch.zeros(1, i + 1)
                         for i in range(4)})
    port = _port()
    log = open(tmp_path / "syncer.log", "w")
    process = _start_pool_syncer(syncer_binary, port, tmp_path, log)
    islands = [_Island(_bridge(initial, port, learner_id)) for learner_id in range(2)]
    try:
        for island in islands:
            island.bridge.start()
        for island in islands:
            island.take_initial_cut()
        fast, slow = islands

        # Sweep 1: only island 0 pushes; the soft deadline merges both
        # fragment rounds without island 1, which keeps its lease while idle.
        fast.train(1.0)
        fast.push_both_fragments()
        time.sleep(3 * LEASE_S)
        _wait(lambda: slow.drain() == (1, 2), what="island 1 sees sweep 1")
        assert fast.drain() == (1, 2)
        assert slow.bridge.client.lease_expiries == 0, "an idle island lost its lease"
        assert fast.bridge.client.lease_expiries == 0

        # Island 1 goes silent: its lease expires, the client redials (re-JOIN,
        # outside the reconnect budget of the fixed roster) and is caught up.
        generation = slow.bridge.client.connection_generation
        slow.bridge._heartbeat._paused = lambda: True
        _wait(lambda: slow.bridge.client.lease_expiries >= 1, what="lease expiry")
        slow.bridge._heartbeat._paused = lambda: False
        _wait(lambda: slow.bridge.client.connection_generation != generation, what="re-JOIN")

        # Sweep 2 merges without island 1 again; it then sees versions it
        # skipped and answers the next sweep's PULLs on its new connection.
        fast.drain()
        fast.train(1.0)
        fast.push_both_fragments()
        _wait(lambda: slow.drain() == (3, 4), what="island 1 sees sweep 2")
        assert fast.drain() == (3, 4)

        # Sweep 3: both islands push and finish on the final cut.
        slow.train(3.0)
        fast.train(1.0)
        slow.push_both_fragments()
        fast.push_both_fragments()
        finals = []
        for island in islands:
            manifest, final = island.bridge.wait_for_final_cut(policy_version=island.steps)
            island.bridge.acknowledge_finalization(manifest)
            finals.append(final)
        assert process.wait(timeout=30) == 0
        assert finals[0].tensors.keys() == finals[1].tensors.keys()
        for name in finals[0].tensors:
            assert torch.equal(finals[0].tensors[name], finals[1].tensors[name])
        kinds = _tape_kinds(tmp_path / "tape.jsonl")
        assert "pool_leave" in kinds and kinds.count("pool_join") >= 3  # two joins + one re-JOIN
    finally:
        for island in islands:
            island.bridge.close()
        if process.poll() is None:
            process.kill()
            process.wait()
        log.close()
