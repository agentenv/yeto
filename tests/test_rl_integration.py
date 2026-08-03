import hashlib
import json
import socket
import subprocess
import threading
import time
from pathlib import Path

import pytest
import torch

from yeto.rl.bridge import BridgeConfig, LocalRoundResult, StrictRlBridge
from yeto.rl.checkpoint import parse_rl_final_marker, validate_rl_final_checkpoint
from yeto.rl.core import CanonicalLoraState, PolicyIdentity, build_avg_layout, canonical_state
from yeto.protocol import DTYPE_F32, SyncerClient
from yeto.tensor_io import pack_tensor

ROOT = Path(__file__).resolve().parent.parent


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def build_syncer():
    subprocess.run(["cargo", "build", "-q"], cwd=ROOT / "syncer", check=True)
    return ROOT / "syncer/target/debug/yeto-syncer"


def start_syncer(
    binary, port, checkpoint, manifest_sha, *, resume=False, event_tape=None
):
    command = [
        str(binary),
        "--port",
        str(port),
        "--learners",
        "2",
        "--quorum",
        "2",
        "--pipeline",
        "1",
        "--sync-interval-steps",
        "0",
        "--delta-correction",
        "none",
        "--total-steps",
        "2",
        "--outer-lr",
        "1",
        "--outer-momentum",
        "0",
        "--checkpoint-path",
        str(checkpoint),
        "--checkpoint-every",
        "1",
        "--mark-final-checkpoint",
        "--quorum-timeout-s",
        "1",
        "--rl-round-timeout-s",
        "30",
        "--run-manifest-sha256",
        manifest_sha,
        "--rl-strict-avg",
    ]
    if resume:
        command.append("--resume")
    if event_tape is not None:
        command.extend(("--event-tape", str(event_tape)))
    return subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


class FakeMilesRuntime:
    def __init__(self, learner_id):
        self.learner_id = learner_id
        self.tensors = {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(1, 2),
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.zeros(2, 1),
        }
        self.trainer_identity = self.rollout_identity = None
        self.rounds = 0
        self.applied = []
        self.stopped = False

    def initialize(self):
        return self.tensors

    def cancel_or_drain_rollouts(self):
        pass

    def apply_global_policy(self, state: CanonicalLoraState):
        self.tensors = {name: value.clone() for name, value in state.tensors.items()}
        self.trainer_identity = self.rollout_identity = state.identity
        self.applied.append(state.identity)

    def run_local_round(self, policy_identity, *, groups, samples_per_group, optimizer_steps):
        self.rounds += 1
        increment = 1.0 + 2.0 * self.learner_id
        self.tensors = {name: value + increment for name, value in self.tensors.items()}
        return LocalRoundResult(
            groups,
            samples_per_group,
            optimizer_steps,
            frozenset({policy_identity}),
            rollout_seconds=0.25,
            train_seconds=0.5,
        )

    def export_local_policy(self):
        return self.tensors

    def read_trainer_policy_identity(self):
        return self.trainer_identity

    def read_rollout_policy_identity(self):
        return self.rollout_identity

    def shutdown(self):
        self.stopped = True


class BridgeThread(threading.Thread):
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


@pytest.mark.timeout(120)
def test_two_fake_miles_islands_match_manual_f32_average_with_real_syncer(tmp_path):
    binary = build_syncer()
    port = free_port()
    checkpoint = tmp_path / "state.ckpt"
    syncer_events = tmp_path / "syncer-events.jsonl"
    manifest_sha = hashlib.sha256(b"fixed test manifest").hexdigest()
    process = start_syncer(
        binary,
        port,
        checkpoint,
        manifest_sha,
        event_tape=syncer_events,
    )
    resumed_process = None
    runtimes = [FakeMilesRuntime(0), FakeMilesRuntime(1)]
    threads = [
        BridgeThread(
            StrictRlBridge(
                runtime,
                BridgeConfig(
                    ("127.0.0.1", port),
                    learner_id,
                    manifest_sha,
                    groups_per_round=2,
                    samples_per_group=3,
                    local_optimizer_steps=1,
                    cache_dir=tmp_path / f"cache-{learner_id}",
                    run_id="fixed-test-run",
                    event_tape=tmp_path / f"events-{learner_id}.jsonl",
                    wan_streams=0,
                    round_timeout_s=30,
                ),
            )
        )
        for learner_id, runtime in enumerate(runtimes)
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=90)
            assert not thread.is_alive(), "RL bridge did not finalize"
            assert thread.error is None, repr(thread.error)
        output, _ = process.communicate(timeout=20)
        assert process.returncode == 0, output

        island_events = {}
        for runtime, thread in zip(runtimes, threads, strict=True):
            assert runtime.rounds == 2
            assert runtime.stopped
            assert [identity.version for identity in runtime.applied] == [0, 1, 2]
            assert thread.result.policy_version == 2
            assert all(
                torch.equal(value, torch.full_like(value, 4.0))
                for value in thread.result.tensors.values()
            )
            events = [
                json.loads(line)
                for line in (tmp_path / f"events-{runtime.learner_id}.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            island_events[runtime.learner_id] = events
            assert [event["committed_version"] for event in events] == [1, 2]
            assert all(
                event["run_id"] == "fixed-test-run"
                and event["groups"] == 2
                and event["trajectories"] == 6
                and event["rollout_seconds"] == 0.25
                and event["train_seconds"] == 0.5
                and event["cache_resend_count"] >= 0
                and event["trainer_applied_identity"]
                == event["rollout_applied_identity"]
                for event in events
            )
        commits = [
            json.loads(line)
            for line in syncer_events.read_text(encoding="utf-8").splitlines()
        ]
        assert [event["committed_version"] for event in commits] == [1, 2]
        for commit in commits:
            version = commit["committed_version"]
            assert commit["fixed_roster"] == 2
            assert commit["responded"] == [0, 1]
            assert [item["id"] for item in commit["delta_digests"]] == [0, 1]
            assert all(
                item["delta_sha256"]
                == island_events[item["id"]][version - 1]["delta_sha256"]
                for item in commit["delta_digests"]
            )
        checkpoint_data = validate_rl_final_checkpoint(checkpoint, manifest_sha)
        assert checkpoint_data.global_step == 2
        assert checkpoint_data.versions == (2,)
        assert checkpoint_data.fragments == ((4.0, 4.0, 4.0, 4.0),)
        assert checkpoint_data.ledger == {0: (2, 2, 2), 1: (2, 2, 2)}
        marker = parse_rl_final_marker(Path(f"{checkpoint}.final"))
        assert marker.policy_sha256 == checkpoint_data.policy_sha256
        finalized_files = {
            path: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in (checkpoint, Path(f"{checkpoint}.final"))
        }

        resumed_port = free_port()
        resumed_process = start_syncer(
            binary,
            resumed_port,
            checkpoint,
            manifest_sha,
            resume=True,
        )
        # A final marker is the run-completion fact: replay must not wait for
        # the full historical roster when only one residual learner reconnects.
        resumed_runtimes = [FakeMilesRuntime(0)]
        resumed_threads = [
            BridgeThread(
                StrictRlBridge(
                    runtime,
                    BridgeConfig(
                        ("127.0.0.1", resumed_port),
                        learner_id,
                        manifest_sha,
                        groups_per_round=2,
                        samples_per_group=3,
                        local_optimizer_steps=1,
                        cache_dir=tmp_path / f"resumed-cache-{learner_id}",
                        run_id="fixed-test-run",
                        event_tape=tmp_path / f"resumed-events-{learner_id}.jsonl",
                        wan_streams=0,
                        round_timeout_s=30,
                    ),
                )
            )
            for learner_id, runtime in enumerate(resumed_runtimes)
        ]
        for thread in resumed_threads:
            thread.start()
        for runtime, thread in zip(resumed_runtimes, resumed_threads, strict=True):
            thread.join(timeout=30)
            assert not thread.is_alive(), "resumed RL bridge did not finalize"
            assert thread.error is None, repr(thread.error)
            assert runtime.rounds == 0
            assert [identity.version for identity in runtime.applied] == [2]
            assert thread.result.policy_hash == checkpoint_data.policy_sha256
        resumed_output, _ = resumed_process.communicate(timeout=20)
        assert resumed_process.returncode == 0, resumed_output
        assert all(
            (path.read_bytes(), path.stat().st_mtime_ns) == contents
            for path, contents in finalized_files.items()
        )
    finally:
        for running in (process, resumed_process):
            if running is not None and running.poll() is None:
                running.kill()
                running.wait(timeout=10)


@pytest.mark.timeout(60)
def test_strict_scheduler_reports_a_fatal_contract_error_to_learners(tmp_path):
    binary = build_syncer()
    port = free_port()
    manifest_sha = hashlib.sha256(b"fatal contract test").hexdigest()
    process = start_syncer(binary, port, tmp_path / "state.ckpt", manifest_sha)
    state = canonical_state(
        0,
        {
            "base_model.model.layer.q_proj.lora_A.weight": torch.zeros(1, 2),
            "base_model.model.layer.q_proj.lora_B.weight": torch.zeros(2, 1),
        },
    )
    layout = build_avg_layout(state.specs)
    clients = [
        SyncerClient(("127.0.0.1", port), learner_id, layout, DTYPE_F32, num_streams=0)
        for learner_id in range(2)
    ]
    try:
        for client in clients:
            client.start()
        clients[0].send_init(0, pack_tensor(torch.zeros(4), DTYPE_F32))

        deadline = time.monotonic() + 15
        permit = None
        while permit is None and time.monotonic() < deadline:
            clients[0].check_health()
            pulls = clients[0].drain_pulls()
            permit = pulls[-1] if pulls else None
            time.sleep(0.02)
        assert permit is not None, "strict syncer did not issue a PULL permit"

        clients[0].push_fragment(
            0,
            permit.global_step,
            permit.round_attempt,
            base_version=0,
            local_step=permit.global_step,
            c_steps=1,
            c_tokens=2,
            tensor_bytes=pack_tensor(torch.ones(4), DTYPE_F32),
        )
        deadline = time.monotonic() + 10
        with pytest.raises(RuntimeError, match="counters=\\(1,1\\)"):
            while time.monotonic() < deadline:
                clients[0].check_health()
                time.sleep(0.02)
            raise AssertionError("learner did not receive the strict scheduler error")

        output, _ = process.communicate(timeout=10)
        assert process.returncode != 0
        assert "does not match permit" in output
        fatal_marker = tmp_path / "state.ckpt.fatal"
        assert f"run_manifest_sha256={manifest_sha}\n" in fatal_marker.read_text()

        resumed = start_syncer(
            binary,
            free_port(),
            tmp_path / "state.ckpt",
            manifest_sha,
            resume=True,
        )
        resumed_output, _ = resumed.communicate(timeout=10)
        assert resumed.returncode != 0
        assert "permanently failed" in resumed_output
    finally:
        for client in clients:
            client.close()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
