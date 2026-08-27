import socket
import subprocess
import threading
import time
from pathlib import Path

import pytest
import torch

from yeto.export import parse_checkpoint
from yeto.protocol import (
    DTYPE_F32,
    BcastFragment,
    FinalFragment,
    FinalManifest,
    PullRequest,
    SyncerClient,
)
from yeto.rl.contracts import LocalStepReceipt
from yeto.rl.dense_full_parameter_sweep import (
    DenseFullParameterSweep,
    DenseSweepConfig,
)
from yeto.rl.dense_sweep_wire import DenseSweepWire
from yeto.rl.local_learner import (
    ComponentIdentity,
    ParameterLayout,
    ParameterSpec,
    dense_sweep_session_contract_hash,
    make_dense_trainer_update,
    make_parameter_cut,
    parameter_cut_from_fragment_flats,
    parameter_values,
)


_CENTRAL_DENSE_ATOL = 1e-7


class _PauseBeforeFragmentClient:
    """Hold one real client immediately before it queues fragment two."""

    def __init__(self, client, paused, release):
        self._client = client
        self._paused = paused
        self._release = release
        self.finalizing = client.finalizing

    def __getattr__(self, name):
        return getattr(self._client, name)

    def push_fragment_parts(self, fragment_id, *args, **kwargs):
        if fragment_id == 1:
            self._paused.set()
            if not self._release.wait(timeout=20):
                raise TimeoutError("restart test did not release fragment two")
        return self._client.push_fragment_parts(fragment_id, *args, **kwargs)


class _SweepClient:
    """One-learner exact-AVG syncer model with canonical sweep ordering."""

    def __init__(self, layout, policy_rounds):
        self.layout = layout
        self.policy_rounds = policy_rounds
        self.finalizing = threading.Event()
        self.initial = {}
        self.current = {}
        self.updates = []
        self.pulls = []
        self.pushes = []
        self.final = None
        self.acked = None
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def close(self):
        self.closed = True

    def check_health(self):
        assert self.started and not self.closed

    @staticmethod
    def _join(parts):
        return b"".join(bytes(part) for part in parts)

    def send_init_parts(self, fragment_id, tensor_parts):
        payload = self._join(tensor_parts)
        self.initial[fragment_id] = payload
        self.current[fragment_id] = payload
        if len(self.initial) == self.layout.fragments.num_fragments:
            self.updates = [
                BcastFragment(fragment_id, 0, self.current[fragment_id])
                for fragment_id in range(self.layout.fragments.num_fragments)
            ]
            self.pulls.append(PullRequest(0, 1, 1))
        return True

    def push_fragment_parts(
        self,
        fragment_id,
        global_step,
        round_attempt,
        base_version,
        local_step,
        c_steps,
        c_tokens,
        tensor_parts,
        *,
        before_last_enqueue=None,
    ):
        assert before_last_enqueue is None
        delta = torch.frombuffer(
            bytearray(self._join(tensor_parts)), dtype=torch.float32
        ).clone()
        base = torch.frombuffer(
            bytearray(self.current[fragment_id]), dtype=torch.float32
        ).clone()
        target = (base + delta).contiguous()
        payload = bytes(memoryview(target.numpy()).cast("B"))
        self.current[fragment_id] = payload
        self.pushes.append(
            (
                fragment_id,
                global_step,
                round_attempt,
                base_version,
                local_step,
                c_steps,
                c_tokens,
            )
        )
        count = self.layout.fragments.num_fragments
        if global_step == self.policy_rounds * count:
            versions = tuple(
                global_step - count + fragment + 1 for fragment in range(count)
            )
            manifest = FinalManifest(global_step, versions)
            fragments = [
                FinalFragment(fragment, versions[fragment], self.current[fragment])
                for fragment in range(count)
            ]
            self.final = (manifest, fragments)
            self.finalizing.set()
        else:
            self.updates.append(
                BcastFragment(fragment_id, global_step, self.current[fragment_id])
            )
            next_step = global_step + 1
            self.pulls.append(PullRequest((next_step - 1) % count, next_step, 1))
        return True

    def drain_pulls(self):
        pulls, self.pulls = self.pulls, []
        return pulls

    def drain_updates(self):
        updates, self.updates = self.updates, []
        return updates

    def wait_for_final_fragments(self, timeout=None):
        assert timeout is not None and self.final is not None
        return self.final

    def acknowledge_finalization(self, manifest):
        assert self.final is not None and manifest == self.final[0]
        self.acked = manifest


class _RestartBeforeCommitClient(_SweepClient):
    def __init__(self, layout, policy_rounds, replay_steps):
        super().__init__(layout, policy_rounds)
        self.replay_steps = set(replay_steps)
        self.attempts = {}

    def push_fragment_parts(self, fragment_id, global_step, *args, **kwargs):
        self.attempts[global_step] = self.attempts.get(global_step, 0) + 1
        if global_step in self.replay_steps and self.attempts[global_step] == 1:
            # Model a process restart before checkpoint commit: the durable
            # base is rebroadcast and the exact same step is re-PULLed.
            count = self.layout.fragments.num_fragments
            for fragment in range(count):
                latest = global_step - 1
                first = fragment + 1
                version = 0 if latest < first else latest - ((latest - first) % count)
                self.updates.append(
                    BcastFragment(fragment, version, self.current[fragment])
                )
            self.pulls.append(PullRequest(fragment_id, global_step, 1))
            return True
        return super().push_fragment_parts(fragment_id, global_step, *args, **kwargs)


def _layout():
    component = ComponentIdentity("actor", "a" * 40, "b" * 64)
    specs = (
        ParameterSpec("actor", "left", (2,), "float32", 2, "rank0"),
        ParameterSpec("actor", "right", (2,), "float32", 2, "rank1"),
    )
    return ParameterLayout.create(
        algorithm="grpo",
        components=(component,),
        specs=specs,
        num_fragments=2,
        fragment_strategy="owner_affine",
    )


def _initial(layout):
    return make_parameter_cut(
        layout,
        policy_version=0,
        values={
            "actor::rank0::left": torch.tensor([1.0, 2.0]),
            "actor::rank1::right": torch.tensor([3.0, 4.0]),
        },
    )


def _update(layout, anchor, deltas, learner_generation=7, learner_id=0):
    local = parameter_cut_from_fragment_flats(
        layout,
        policy_version=anchor.policy_version,
        fragments={
            fragment.fragment_id: fragment.flat + deltas[fragment.fragment_id]
            for fragment in anchor.fragments
        },
    )
    receipt = LocalStepReceipt(
        algorithm="grpo",
        learner_id=learner_id,
        learner_generation=learner_generation,
        base_policy_version=anchor.policy_version,
        base_policy_hash=anchor.policy_hash,
        input_batch_hash="c" * 64,
        trajectory_ids=(f"trajectory-{anchor.policy_version}",),
        trained_tokens=17,
        optimizer_steps=1,
        optimizer_step_succeeded=True,
        parameter_layout_hash=layout.layout_hash,
    )
    return make_dense_trainer_update(
        layout,
        anchor,
        local,
        receipt,
        learner_id=learner_id,
        learner_generation=learner_generation,
        target_policy_version=anchor.policy_version + 1,
    )


def _grpo_adam_step(initial, features, advantages):
    """One deterministic centralized/local GRPO optimizer step."""
    parameter = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.Adam(
        (parameter,),
        lr=1e-2,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=0,
    )
    old_logprobs = features @ initial
    new_logprobs = features @ parameter
    loss = -(torch.exp(new_logprobs - old_logprobs) * advantages).mean()
    loss.backward()
    optimizer.step()
    return parameter.detach()


def _equal_weight_delta_oracle(layout, initial, learner_deltas):
    """Apply the explicit mean of learner-local deltas to one policy cut."""
    return parameter_cut_from_fragment_flats(
        layout,
        policy_version=initial.policy_version + 1,
        fragments={
            fragment.fragment_id: fragment.flat
            + torch.stack(
                [deltas[fragment.fragment_id] for deltas in learner_deltas]
            ).mean(dim=0)
            for fragment in initial.fragments
        },
    )


def test_two_policy_sweeps_apply_only_after_complete_cut_and_ack_terminal():
    layout = _layout()
    initial = _initial(layout)
    client = _SweepClient(layout, policy_rounds=2)
    sweep = DenseFullParameterSweep(
        layout,
        initial,
        DenseSweepConfig(("unused", 0), 0, 7, 2, poll_seconds=0.001),
        client=client,
    )

    assert sweep.start().policy_hash == initial.policy_hash
    first = sweep.exchange(
        _update(
            layout,
            initial,
            {0: torch.tensor([0.5, 1.0]), 1: torch.tensor([1.5, 2.0])},
        )
    )
    assert first.cut.policy_version == 1
    assert first.fragment_versions == (1, 2)
    assert not first.terminal
    # The network can finish a sweep, but the model policy remains at v0 until
    # the Miles safe-boundary apply succeeds.
    assert sweep.current.policy_version == 0
    with pytest.raises(RuntimeError, match="previous dense policy"):
        sweep.exchange(_update(layout, initial, {0: torch.zeros(2), 1: torch.zeros(2)}))
    first_cut = sweep.commit_applied(first)
    assert first_cut.policy_version == 1
    assert client.acked is None

    second = sweep.exchange(
        _update(
            layout,
            first_cut,
            {0: torch.tensor([1.0, 1.0]), 1: torch.tensor([-1.0, -1.0])},
        )
    )
    assert second.cut.policy_version == 2
    assert second.fragment_versions == (3, 4)
    assert second.terminal
    assert sweep.current.policy_version == 1
    sweep.commit_applied(second)
    assert client.acked == FinalManifest(4, (3, 4))
    assert client.pushes == [
        (0, 1, 1, 0, 1, 1, 17),
        (1, 2, 1, 0, 1, 1, 17),
        (0, 3, 1, 1, 2, 1, 17),
        (1, 4, 1, 2, 2, 1, 17),
    ]


def test_h1_receipt_and_exact_learner_generation_are_fail_closed():
    layout = _layout()
    initial = _initial(layout)
    client = _SweepClient(layout, policy_rounds=1)
    sweep = DenseFullParameterSweep(
        layout,
        initial,
        DenseSweepConfig(("unused", 0), 0, 7, 1, poll_seconds=0.001),
        client=client,
    )
    sweep.start()

    bad = _update(
        layout,
        initial,
        {0: torch.ones(2), 1: torch.ones(2)},
        learner_generation=8,
    )
    with pytest.raises(ValueError, match="bind this H=1 round"):
        sweep.exchange(bad)
    assert client.pushes == []


def test_conflicting_duplicate_broadcast_is_rejected_before_initial_commit():
    layout = _layout()
    initial = _initial(layout)
    client = _SweepClient(layout, policy_rounds=1)
    client.started = True
    client.updates = [
        BcastFragment(0, 0, bytes(8)),
        BcastFragment(0, 0, bytes([1]) * 8),
    ]
    sweep = DenseFullParameterSweep(
        layout,
        initial,
        DenseSweepConfig(
            ("unused", 0),
            1,
            0,
            1,
            send_initial_params=False,
            poll_seconds=0.001,
        ),
        client=client,
    )
    client.started = False
    with pytest.raises(RuntimeError, match="conflicting broadcasts"):
        sweep.start()


def test_wire_replays_only_exact_uncommitted_pull_before_advancing_sweep():
    layout = _layout()
    initial = _initial(layout)
    client = _RestartBeforeCommitClient(layout, policy_rounds=1, replay_steps={1, 2})
    wire = DenseSweepWire(
        layout.fragments,
        DenseSweepConfig(("unused", 0), 0, 7, 1, poll_seconds=0.001),
        client=client,
    )
    initial_payloads = {
        fragment.fragment_id: (
            lambda fragment=fragment: (memoryview(fragment.flat.numpy()).cast("B"),)
        )
        for fragment in initial.fragments
    }
    received = wire.start(initial_payloads)
    assert len(received) == 2
    deltas = {
        0: lambda: (memoryview(torch.ones(2).numpy()).cast("B"),),
        1: lambda: (memoryview(torch.full((2,), 2.0).numpy()).cast("B"),),
    }

    pending = wire.exchange(
        base_policy_version=0,
        trained_tokens=17,
        sweep_update_id="a" * 64,
        delta_parts=deltas,
    )

    assert [submission.push_attempts for submission in pending.submissions] == [2, 2]
    assert client.attempts == {1: 2, 2: 2}
    assert pending.fragment_versions == (1, 2)
    assert pending.terminal
    assert wire.policy_version == 0
    wire.commit_applied(pending)
    assert wire.policy_version == 1
    assert client.acked == FinalManifest(2, (1, 2))


def test_two_real_clients_match_central_grpo_for_identical_frozen_batch(tmp_path):
    root = Path(__file__).resolve().parent.parent
    subprocess.run(["cargo", "build", "-q"], cwd=root / "syncer", check=True)
    binary = root / "syncer/target/debug/yeto-syncer"
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    checkpoint = tmp_path / "dense-sweep.ckpt"
    process = subprocess.Popen(
        [
            str(binary),
            "--port",
            str(port),
            "--learners",
            "2",
            "--quorum",
            "2",
            "--grace-ms",
            "0",
            "--pipeline",
            "1",
            "--sync-interval-steps",
            "0",
            "--delta-correction",
            "none",
            "--total-steps",
            "2",
            "--policy-sweep-fragments",
            "2",
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
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    layout = _layout()
    initial = _initial(layout)
    generations = (7, 8)
    initial_flat = torch.cat([fragment.flat for fragment in initial.fragments])
    frozen_features = torch.tensor(
        [[1.0, 0.5, -0.5, 0.25], [-0.25, 1.0, 0.75, -1.0]],
        dtype=torch.float32,
    )
    frozen_advantages = torch.tensor([1.0, -1.0], dtype=torch.float32)
    local = _grpo_adam_step(initial_flat, frozen_features, frozen_advantages)
    centralized = _grpo_adam_step(
        initial_flat,
        frozen_features.repeat((2, 1)),
        frozen_advantages.repeat(2),
    )
    local_delta = local - initial_flat
    identical_deltas = {
        0: local_delta[:2].clone(),
        1: local_delta[2:].clone(),
    }
    deltas = (identical_deltas, identical_deltas)
    results = [None, None]
    failures = []

    def run(learner_id):
        try:
            sweep = DenseFullParameterSweep(
                layout,
                initial,
                DenseSweepConfig(
                    ("127.0.0.1", port),
                    learner_id,
                    generations[learner_id],
                    1,
                    wan_streams=1,
                    wait_timeout=15,
                    poll_seconds=0.001,
                ),
                learner_generations=dict(enumerate(generations)),
            )
            anchor = sweep.start()
            pending = sweep.exchange(
                _update(
                    layout,
                    anchor,
                    deltas[learner_id],
                    learner_generation=generations[learner_id],
                    learner_id=learner_id,
                )
            )
            results[learner_id] = sweep.commit_applied(pending)
            sweep.close()
        except Exception as error:  # noqa: BLE001 - preserve thread failure for parent assertion
            failures.append(error)

    threads = [
        threading.Thread(target=run, args=(learner_id,)) for learner_id in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)
    try:
        assert not any(thread.is_alive() for thread in threads)
        assert not failures
        assert process.wait(timeout=10) == 0
        assert results[0].policy_hash == results[1].policy_hash
        values = parameter_values(layout, results[0])
        dense = torch.cat(
            [values["actor::rank0::left"], values["actor::rank1::right"]]
        )
        assert torch.allclose(
            dense,
            centralized,
            rtol=0,
            atol=_CENTRAL_DENSE_ATOL,
        )
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


def test_two_real_clients_resume_mid_sweep_without_double_accounting(tmp_path):
    root = Path(__file__).resolve().parent.parent
    subprocess.run(["cargo", "build", "-q"], cwd=root / "syncer", check=True)
    binary = root / "syncer/target/debug/yeto-syncer"
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    checkpoint = tmp_path / "dense-sweep-restart.ckpt"
    command = [
        str(binary),
        "--port",
        str(port),
        "--learners",
        "2",
        "--quorum",
        "2",
        "--grace-ms",
        "0",
        "--pipeline",
        "1",
        "--sync-interval-steps",
        "0",
        "--delta-correction",
        "none",
        "--total-steps",
        "2",
        "--policy-sweep-fragments",
        "2",
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
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    layout = _layout()
    initial = _initial(layout)
    generations = (7, 8)
    roster = dict(enumerate(generations))
    contract_hash = dense_sweep_session_contract_hash(
        layout,
        policy_rounds=1,
        learner_generations=roster,
    )
    deltas = (
        {0: torch.tensor([1.0, 1.0]), 1: torch.tensor([2.0, 2.0])},
        {0: torch.tensor([3.0, 3.0]), 1: torch.tensor([-2.0, -2.0])},
    )
    paused = (threading.Event(), threading.Event())
    release = threading.Event()
    clients = [None, None]
    results = [None, None]
    failures = []

    def run(learner_id):
        sweep = None
        try:
            client = SyncerClient(
                ("127.0.0.1", port),
                learner_id,
                layout.fragments,
                dtype=DTYPE_F32,
                num_streams=1,
                connect_timeout=15,
                max_reconnects=None,
                session_contract_hash=contract_hash,
            )
            clients[learner_id] = client
            sweep = DenseFullParameterSweep(
                layout,
                initial,
                DenseSweepConfig(
                    ("127.0.0.1", port),
                    learner_id,
                    generations[learner_id],
                    1,
                    wan_streams=1,
                    wait_timeout=25,
                    poll_seconds=0.001,
                ),
                client=_PauseBeforeFragmentClient(
                    client,
                    paused[learner_id],
                    release,
                ),
            )
            anchor = sweep.start()
            pending = sweep.exchange(
                _update(
                    layout,
                    anchor,
                    deltas[learner_id],
                    learner_generation=generations[learner_id],
                    learner_id=learner_id,
                )
            )
            results[learner_id] = sweep.commit_applied(pending)
        except Exception as error:  # noqa: BLE001 - report worker failure in parent
            failures.append(error)
        finally:
            if sweep is not None:
                sweep.close()

    threads = [
        threading.Thread(target=run, args=(learner_id,)) for learner_id in range(2)
    ]
    for thread in threads:
        thread.start()
    try:
        assert all(event.wait(timeout=20) for event in paused)
        deadline = time.monotonic() + 10
        partial = None
        while time.monotonic() < deadline:
            if checkpoint.is_file():
                try:
                    candidate = parse_checkpoint(checkpoint)
                except (OSError, ValueError):
                    pass
                else:
                    if candidate.global_step == 1:
                        partial = candidate
                        break
            time.sleep(0.02)
        assert partial is not None
        assert [fragment[0] for fragment in partial.fragments] == [1, 0]
        assert partial.ledger == {0: (1, 0, 0), 1: (1, 0, 0)}

        process.kill()
        assert process.wait(timeout=5) != 0
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        release.set()
        for thread in threads:
            thread.join(timeout=30)

        assert not any(thread.is_alive() for thread in threads)
        assert not failures
        assert process.wait(timeout=10) == 0
        expected = _equal_weight_delta_oracle(layout, initial, deltas)
        assert {result.policy_hash for result in results} == {expected.policy_hash}
        assert all(client.finalized.is_set() for client in clients)

        final = parse_checkpoint(checkpoint)
        assert final.global_step == 2
        assert [fragment[0] for fragment in final.fragments] == [1, 2]
        assert final.ledger == {0: (2, 1, 17), 1: (2, 1, 17)}
    finally:
        release.set()
        for client in clients:
            if client is not None:
                client.close()
        for thread in threads:
            thread.join(timeout=5)
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
