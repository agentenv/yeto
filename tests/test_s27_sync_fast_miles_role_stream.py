"""S27 G-AA: deferred capture and batched applies on the real Miles role stream.

Asynchronous rounds (partial quorum), several local rounds per island:

* deferred capture: each local round's capture hashes every fragment but
  stores none in Ray; the pairing materializes only the fragments it pushes;
* batched apply: the broadcasts one commit drains are applied in one trainer
  apply (one pair of optimizer proofs, one whole-model commit);
* optimizer proofs without the whole-model model/master comparison.

The trainer fake models the Miles contract (``full_parameter_state``): a
deferred row has no bytes, a skip-resident apply skips a fragment whose
payload hash the trainer recorded at its last export or commit, a local step
drops that record, and materialization refuses a fragment whose bytes
changed.  Miles' own tests check that side against the real state code.

The run with every feature on must give bit-identical trainer values, syncer
values and pushed deltas to the run with every feature off.

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
    _build_cut,
    _Client,
    _Driver,
    _Syncer,
    _Trainer,
)
from test_rl_miles_chunked_full_parameter import FakeRay, _evidence, component

_state = pytest.importorskip("miles.backends.megatron_utils.full_parameter_state")
if not hasattr(_state, "DEFERRED_FULL_PARAMETER_CHUNK"):
    pytest.skip("needs a Miles with the deferred export (yeto/s27-sync-fast)", allow_module_level=True)
DEFERRED_FULL_PARAMETER_CHUNK = _state.DEFERRED_FULL_PARAMETER_CHUNK
is_deferred_full_parameter_chunk = _state.is_deferred_full_parameter_chunk
from yeto.protocol import PullRequest
from yeto.rl.adapters.miles.full_parameter import role_stream as role_stream_module
from yeto.rl.adapters.miles.full_parameter.role_stream import (
    TERMINAL_RETIRE_ENV,
    MilesFullParameterRoleStream,
    RoleStreamConfig,
)
from yeto.rl.adapters.miles.models import chunked_full_parameter as adapter_module
from yeto.rl.adapters.miles.models.chunked_full_parameter import (
    DEFER_CAPTURE_ENV,
    SKIP_RESIDENT_ENV,
    MilesChunkedFullParameterAdapter,
)
from yeto.rl.engine.full_parameter_sync import (
    BATCH_APPLY_ENV,
    FullParameterSync,
    FullParameterSyncProfile,
    RolePairing,
)

ROUNDS = 3  # local rounds per island; every fragment merges ROUNDS times
DEPTH = 2


class _AsyncSyncer(_Syncer):
    """Asynchronous fragment rounds with this island and one virtual peer:
    a round merges when this island pushed it and is broadcast at once; the
    round ``DEPTH`` steps later opens then; the last step arrives as FINAL."""

    def commit_ready(self) -> None:
        total = self.sweeps * P
        for step in sorted(self.received):
            delta = self.received.pop(step)
            f, k = (step - 1) % P, (step - 1) // P
            self.global_[f] = self.global_[f] + OTHER_WEIGHT * (delta + self.peer(k)[f])
            self.versions[f] = step
            self.committed.append(step)
            if step < total:
                self.client.broadcast(f, step, self.global_[f])
            if step + DEPTH <= total:
                self.client.pulls.append(PullRequest((step + DEPTH - 1) % P, step + DEPTH, self.attempt))
        if len(self.committed) == total and not self.client.finalizing.is_set():
            self.client.publish_final()


class _AsyncClient(_Client):
    def pull_sweep(self, sweep, attempt) -> None:
        if sweep == 0:
            self.pulls.extend(PullRequest(step - 1, step, attempt) for step in range(1, DEPTH + 1))


def _hashes(cut):
    return {d.fragment_id: d.payload_hash for shard in cut.shards for d in shard.fragments}


class _MilesContractTrainer(_Trainer):
    """``_Trainer`` plus the Miles deferred-export / skip-resident contract."""

    def __init__(self, *args, fast: bool, **kwargs):
        super().__init__(*args, **kwargs)
        self.fast = fast
        self.resident: dict[int, str] | None = None
        self.stored_bytes = 0
        self.applies = 0
        self.read_fragments = 0
        self.model_checks = 0
        self.deferred_exports = 0

    def _stored(self, cut):
        self.stored_bytes += sum(4 * d.numel for shard in cut.shards for d in shard.fragments)
        return cut

    async def record_full_parameter_local_step(self, **kwargs):
        self.resident = None  # the masters changed
        self.model_checks += 1
        return await super().record_full_parameter_local_step(**kwargs)

    async def export_full_parameter_chunked_cut(self, policy_version, local_step_generation=0, *,
                                                max_chunk_bytes, defer_payload=False):
        assert defer_payload == self.fast or local_step_generation == 0
        if not defer_payload:
            cut = self._stored(await super().export_full_parameter_chunked_cut(
                policy_version, local_step_generation, max_chunk_bytes=max_chunk_bytes))
        else:
            self.deferred_exports += 1
            cut = _build_cut(self.adapter, self.values, policy_version=policy_version,
                             generation=local_step_generation, ray=FakeRay())  # hashes only
            for shard in cut.shards:
                for refs in shard.chunk_refs:
                    refs[:] = [DEFERRED_FULL_PARAMETER_CHUNK] * len(refs)
        self.resident = _hashes(cut)
        return cut

    async def materialize_full_parameter_fragments(self, cut, fragment_ids):
        assert self.fast
        now = _hashes(_build_cut(self.adapter, self.values, policy_version=0, generation=0, ray=FakeRay()))
        wanted = _hashes(cut)
        stored = {}
        fresh = _build_cut(self.adapter, self.values, policy_version=cut.policy_version,
                           generation=cut.local_step_generation, ray=self.ray)
        for fragment_id in fragment_ids:
            assert self.resident is not None and self.resident[fragment_id] == wanted[fragment_id], \
                "materialized a fragment the masters no longer hold"
            assert now[fragment_id] == wanted[fragment_id]
            ((refs, numel),) = [(r, d.numel) for shard in fresh.shards
                                for d, r in zip(shard.fragments, shard.chunk_refs, strict=True)
                                if d.fragment_id == fragment_id]
            stored[fragment_id] = list(refs)
            self.stored_bytes += 4 * numel
        return stored

    async def apply_full_parameter_chunked_cut(self, cut, *, commit_token=None, max_commit_attempts=3,
                                               skip_resident_fragments=False):
        assert commit_token
        assert skip_resident_fragments == self.fast
        self.applies += 1
        for shard in cut.shards:
            for descriptor, refs in zip(shard.fragments, shard.chunk_refs, strict=True):
                if skip_resident_fragments and self.resident is not None and \
                        self.resident.get(descriptor.fragment_id) == descriptor.payload_hash:
                    continue  # the masters hold these bytes
                assert not any(is_deferred_full_parameter_chunk(ref) for ref in refs), "read a deferred row"
                assert all(ref is not None for ref in refs), "applied a released chunk"
                self.read_fragments += 1
                self.values[descriptor.fragment_id] = torch.cat([self.ray.get(ref) for ref in refs]).clone()
        self.installed = cut.policy_version
        self.generation = 0
        self.resident = _hashes(cut)
        self.model_checks += 1
        return self.adapter.expected_parameter_tensor_count

    async def full_parameter_optimizer_states(self, *, check_model=True):
        assert check_model is False and self.fast
        return await super().full_parameter_optimizer_states()


class _OriginalContractTrainer(_MilesContractTrainer):
    """The trainer before S27 G-AA: every optimizer proof compares the model."""

    async def full_parameter_optimizer_states(self):
        self.model_checks += 1
        return await _Trainer.full_parameter_optimizer_states(self)


def _config() -> RoleStreamConfig:
    adapter = _adapter()
    return RoleStreamConfig(
        algorithm="grpo", role="actor", component=component(), syncer_addr=("127.0.0.1", 30100),
        learner_id=0, learner_generation=0, learner_generations=(0, 0),
        total_fragment_steps=ROUNDS * P, expected_fragments=P,
        expected_layout_hash=adapter.layout.layout_hash, local_horizon=1, optimizer_steps_per_round=1,
        training_contract_hash="2" * 64, syncer_profile_hash="3" * 64, pipeline_depth=DEPTH,
        algorithm_spec_sha256="a1" * 32, wait_timeout=5.0, poll_seconds=0.001,
        max_fragment_bytes=64, max_chunk_bytes=8, partial_quorum=True)


def _run(monkeypatch, *, fast: bool):
    switch = "1" if fast else "0"
    for env in (DEFER_CAPTURE_ENV, BATCH_APPLY_ENV, SKIP_RESIDENT_ENV, TERMINAL_RETIRE_ENV):
        monkeypatch.setenv(env, switch)
    monkeypatch.setattr(adapter_module, "_local_transport_features",
                        lambda: frozenset({"skip_resident", "deferred_export", "optimizer_state_check_model"}))
    generator = torch.Generator().manual_seed(29)

    def rand(scale=1.0):
        return {f: torch.randn(n, generator=generator) * scale for f, n in enumerate(SIZES)}

    initial = rand()
    steps = {k: rand(0.1) for k in range(ROUNDS)}
    peer = {k: rand(0.1) for k in range(ROUNDS)}
    ray = FakeRay()
    adapter = _adapter()
    trainer = (_MilesContractTrainer if fast else _OriginalContractTrainer)(
        adapter, ray, initial, lambda k: steps[k], fast=fast)
    syncer = _AsyncSyncer(initial, ROUNDS, lambda k: peer[k])
    client = _AsyncClient(syncer)

    async def initialize(cls, group, *, policy_version, **_kwargs):
        cut = await group.export_full_parameter_chunked_cut(policy_version, 0,
                                                            max_chunk_bytes=adapter.max_chunk_bytes)
        return adapter, adapter.validate_cut(cut, policy_version, 0)

    monkeypatch.setattr(MilesChunkedFullParameterAdapter, "initialize", classmethod(initialize))
    monkeypatch.setattr(role_stream_module, "SyncerClient", lambda *_args, **_kwargs: client)
    stream = MilesFullParameterRoleStream(_config(), ray_module=ray)
    profile = FullParameterSyncProfile.from_configs({"actor": stream.config})
    session = FullParameterSync(
        RolePairing({"actor": stream}, profile), groups={"actor": trainer}, run=asyncio.run,
        trajectories=lambda rollout_id: _evidence(rollout_id=rollout_id,
                                                  behavior_policy_hash=stream.current.policy_hash))
    driver = _Driver()
    session.start(driver)
    stops = []
    for rollout_id in range(ROUNDS + 2):
        boundary = session.boundary(driver, rollout_id=rollout_id, stats=None)
        stops.append(boundary.stop)
        if boundary.stop:
            session.published(driver, rollout_id=rollout_id, policy_hash=boundary.state.policy_tensor_hash())
            break
    session.finish(driver)
    return SimpleNamespace(trainer=trainer, syncer=syncer, stream=stream, stops=stops,
                           pairing=session.pairing, driver=driver)


def _bits(values):
    return {f: value.view(torch.int32).tolist() for f, value in values.items()}


def test_deferred_capture_and_batched_applies_keep_every_value_bit_identical(monkeypatch):
    slow = _run(monkeypatch, fast=False)
    fast = _run(monkeypatch, fast=True)

    for run in (slow, fast):
        assert run.stops == [False] * (ROUNDS - 1) + [True]
        assert run.syncer.committed == list(range(1, ROUNDS * P + 1))
        assert run.stream.final_applied and run.stream.final_acknowledged
    assert _bits(fast.trainer.values) == _bits(slow.trainer.values)
    assert _bits(fast.syncer.global_) == _bits(slow.syncer.global_)
    assert [(s, a, d.view(torch.int32).tolist()) for s, a, d in fast.syncer.pushed] == [
        (s, a, d.view(torch.int32).tolist()) for s, a, d in slow.syncer.pushed]
    # Every trainer value went through the same local rounds.
    assert len(fast.trainer.trained_from) == len(slow.trainer.trained_from) == ROUNDS
    for got, want in zip(fast.trainer.trained_from, slow.trainer.trained_from, strict=True):
        assert _bits(got) == _bits(want)

    # Deferred capture: the round cuts store only what is pushed (once each).
    assert fast.trainer.deferred_exports == ROUNDS
    model_bytes = 4 * sum(SIZES)
    assert slow.trainer.stored_bytes == (1 + ROUNDS) * model_bytes
    assert fast.trainer.stored_bytes == (1 + ROUNDS) * model_bytes  # INIT + every pushed fragment once
    print(f"applies {slow.trainer.applies}->{fast.trainer.applies} "
          f"fragment reads {slow.trainer.read_fragments}->{fast.trainer.read_fragments} "
          f"whole-model checks {slow.trainer.model_checks}->{fast.trainer.model_checks}")
    # Batched applies: one apply per drained commit instead of one per fragment.
    assert fast.trainer.applies < slow.trainer.applies
    assert fast.trainer.read_fragments < slow.trainer.read_fragments
    assert fast.trainer.model_checks < slow.trainer.model_checks
