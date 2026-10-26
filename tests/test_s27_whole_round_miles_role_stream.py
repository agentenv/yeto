"""S27 G-AA: one boundary pushes a whole model round on the real Miles role stream.

``--total-steps 1 --rl-sync-fragment-pipeline P`` (s27-gr-f): the syncer opens
every fragment round at once, the island pushes all P fragments at its first
boundary and then trains no more.  Every broadcast is then a fragment's last
round, followed only by the FINAL cut, so the pairing retires those rounds
instead of applying each one to the trainer.  The final trainer values must be
bit-identical to the run that applies every broadcast.

The coordinator drives the real ``MilesFullParameterRoleStream`` and
``MilesChunkedFullParameterAdapter``; only the trainer group, Ray and the
syncer are fakes (``test_e3_overlap_miles_role_stream``).

Needs Miles (``importorskip``).
"""

from __future__ import annotations

# ruff: noqa: I001 -- the Miles test helpers install a bridge stub before import.

import asyncio
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("miles")

from test_e3_overlap_miles_role_stream import (
    OTHER_WEIGHT,
    P,
    SIZES,
    _adapter,
    _Client,
    _Driver,
    _Syncer,
    _Trainer,
)
from test_rl_miles_chunked_full_parameter import FakeRay, _evidence, component

from yeto.protocol import PullRequest
from yeto.rl.adapters.miles.full_parameter import role_stream as role_stream_module
from yeto.rl.adapters.miles.full_parameter.role_stream import (
    TERMINAL_RETIRE_ENV,
    MilesFullParameterRoleStream,
    RoleStreamConfig,
)
from yeto.rl.adapters.miles.models.chunked_full_parameter import (
    MilesChunkedFullParameterAdapter,
)
from yeto.rl.engine.full_parameter_sync import (
    FullParameterSync,
    FullParameterSyncProfile,
    RolePairing,
)


class _AsyncSyncer(_Syncer):
    """Asynchronous fragment rounds, fixed roster of this island and one
    virtual peer: a round merges as soon as this island pushed it and is
    broadcast at once; all rounds open at INIT (pipeline P); the last step
    arrives as FINAL only."""

    def commit_ready(self) -> None:
        for step in sorted(self.received):
            delta = self.received.pop(step)
            f, k = (step - 1) % P, (step - 1) // P
            self.global_[f] = self.global_[f] + OTHER_WEIGHT * (delta + self.peer(k)[f])
            self.versions[f] = step
            self.committed.append(step)
            if step < self.sweeps * P:
                self.client.broadcast(f, step, self.global_[f])
        if len(self.committed) == self.sweeps * P and not self.client.finalizing.is_set():
            self.client.publish_final()


class _WholeRoundClient(_Client):
    def pull_sweep(self, sweep, attempt) -> None:
        # pipeline P: every round of the (only) model round opens at INIT.
        if sweep == 0:
            self.pulls.extend(PullRequest(f, f + 1, attempt) for f in range(P))


class _CountingTrainer(_Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.applies: list[tuple[int, ...]] = []

    async def apply_full_parameter_chunked_cut(self, cut, *, commit_token=None, max_commit_attempts=3):
        self.applies.append(tuple(sorted(d.fragment_id for shard in cut.shards for d in shard.fragments)))
        return await super().apply_full_parameter_chunked_cut(
            cut, commit_token=commit_token, max_commit_attempts=max_commit_attempts)


def _config() -> RoleStreamConfig:
    adapter = _adapter()
    return RoleStreamConfig(
        algorithm="grpo", role="actor", component=component(), syncer_addr=("127.0.0.1", 30100),
        learner_id=0, learner_generation=0, learner_generations=(0, 0),
        total_fragment_steps=P, expected_fragments=P,
        expected_layout_hash=adapter.layout.layout_hash, local_horizon=1, optimizer_steps_per_round=1,
        training_contract_hash="2" * 64, syncer_profile_hash="3" * 64, pipeline_depth=P,
        algorithm_spec_sha256="a1" * 32, wait_timeout=5.0, poll_seconds=0.001,
        max_fragment_bytes=64, max_chunk_bytes=8, partial_quorum=True)


def _run(monkeypatch, *, retire: bool):
    monkeypatch.setenv(TERMINAL_RETIRE_ENV, "1" if retire else "0")
    generator = torch.Generator().manual_seed(23)

    def rand(scale=1.0):
        return {f: torch.randn(n, generator=generator) * scale for f, n in enumerate(SIZES)}

    initial, step, peer = rand(), rand(0.1), rand(0.1)
    ray = FakeRay()
    adapter = _adapter()
    trainer = _CountingTrainer(adapter, ray, initial, lambda _k: step)
    syncer = _AsyncSyncer(initial, 1, lambda _k: peer)
    client = _WholeRoundClient(syncer)

    async def initialize(cls, group, *, policy_version, **_kwargs):
        cut = await group.export_full_parameter_chunked_cut(policy_version, 0,
                                                            max_chunk_bytes=adapter.max_chunk_bytes)
        return adapter, adapter.validate_cut(cut, policy_version, 0)

    monkeypatch.setattr(MilesChunkedFullParameterAdapter, "initialize", classmethod(initialize))
    monkeypatch.setattr("yeto.rl.engine.full_parameter_stream.SyncerClient", lambda *_args, **_kwargs: client)
    stream = MilesFullParameterRoleStream(_config(), ray_module=ray)
    profile = FullParameterSyncProfile.from_configs({"actor": stream.config})
    assert profile.partial_quorum and profile.effective_pipeline_depth == P
    session = FullParameterSync(
        RolePairing({"actor": stream}, profile), groups={"actor": trainer}, run=asyncio.run,
        trajectories=lambda rollout_id: _evidence(rollout_id=rollout_id,
                                                  behavior_policy_hash=stream.current.policy_hash))
    driver = _Driver()
    session.start(driver)
    boundary = session.boundary(driver, rollout_id=0, stats=None)
    assert boundary.stop  # one boundary: every last round was ready
    session.published(driver, rollout_id=0, policy_hash=boundary.state.policy_tensor_hash())
    session.finish(driver)
    return SimpleNamespace(trainer=trainer, syncer=syncer, stream=stream, driver=driver,
                           pairing=session.pairing, initial=initial, step=step, peer=peer)


def test_the_terminal_retire_keeps_the_final_parameters_bit_identical(monkeypatch):
    applied = _run(monkeypatch, retire=False)
    retired = _run(monkeypatch, retire=True)

    for run in (applied, retired):
        assert run.syncer.committed == list(range(1, P + 1))
        assert run.stream.final_applied and run.stream.final_acknowledged
        for f in range(P):
            assert torch.equal(run.trainer.values[f], run.syncer.global_[f])  # FINAL applied
    for f in range(P):
        assert torch.equal(retired.trainer.values[f].view(torch.int32), applied.trainer.values[f].view(torch.int32))
        assert torch.equal(retired.syncer.global_[f], applied.syncer.global_[f])
    # The pushed deltas are the same bytes in both runs.
    assert [(s, a, d.view(torch.int32).tolist()) for s, a, d in retired.syncer.pushed] == [
        (s, a, d.view(torch.int32).tolist()) for s, a, d in applied.syncer.pushed]
    # Applied: each non-terminal broadcast, then FINAL.  Retired: FINAL only.
    assert applied.trainer.applies == [(0, 1)] * P
    assert retired.trainer.applies == [(0, 1)]
    assert retired.pairing.retired_rounds == P - 1 and applied.pairing.retired_rounds == 0
    assert retired.stream.retired_fragments == set(range(P - 1))
    rounds = [fields for name, fields in retired.driver.events if name == "rl_full_parameter_round"]
    assert rounds[-1]["sync/retired_rounds"] == P - 1
