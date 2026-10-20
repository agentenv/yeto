"""S26 E3: sync/train overlap on the real Miles role stream (CPU tensors, no GPU).

``tests/test_e3_sync_overlap.py`` drives ``FullParameterSync`` with a numeric
fake role. Here the coordinator drives the real ``MilesFullParameterRoleStream``
and ``MilesChunkedFullParameterAdapter`` (``rebase_fragment``, held
submissions, ``finish_rebase``, ``mark_round_base``, retained replays). Only
the Megatron trainer group, Ray and the syncer connection are fakes:

* the trainer keeps one f32 tensor per fragment and adds a fixed progress per
  round, so every pushed delta must equal that round's progress exactly;
* the syncer merges the island's delta with a second, virtual island's delta
  (plain average) so the global weights differ from the local ones and a
  wrong rebase shows up in the numbers.

The syncer-restart tests cut the run while the island trains the overlapped
round: the syncer resumes from its last checkpoint (uncommitted pushes lost,
PULLs re-sent with a new attempt, committed fragments re-broadcast) and the
run must still count every round exactly once.

Needs Miles (``importorskip``): run it in the Miles image.
"""

from __future__ import annotations

# ruff: noqa: I001 -- the Miles test helpers install a bridge stub before import.

import asyncio
import threading
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("miles")

from test_rl_miles_chunked_full_parameter import FakeRay, _evidence, component, manifest  # noqa: E402

from miles.backends.megatron_utils.full_parameter_state import (  # noqa: E402
    FullParameterChunkDescriptor,
    FullParameterChunkedShardState,
    FullParameterFragmentDescriptor,
    _chunk_hash,
    _fragment_hasher,
    _tensor_bytes,
)
from miles.ray.full_parameter_transport import FullParameterChunkedCut  # noqa: E402
from yeto.protocol import BcastFragment, FinalFragment, FinalManifest, PullRequest  # noqa: E402
from yeto.rl.adapters.miles.full_parameter import role_stream as role_stream_module  # noqa: E402
from yeto.rl.adapters.miles.full_parameter.role_stream import (  # noqa: E402
    MilesFullParameterRoleStream,
    RoleStreamConfig,
)
from yeto.rl.adapters.miles.models.chunked_full_parameter import (  # noqa: E402
    MilesChunkedFullParameterAdapter,
)
from yeto.rl.engine.full_parameter_sync import (  # noqa: E402
    FullParameterSync,
    FullParameterSyncProfile,
    RolePairing,
)

P = 2
SIZES = (3, 2)  # one parameter per owner shard; one fragment per shard
OTHER_WEIGHT = 0.5  # plain average of this island and one virtual peer


def _adapter() -> MilesChunkedFullParameterAdapter:
    return MilesChunkedFullParameterAdapter.create(
        (manifest(0, (SIZES[0],)), manifest(1, (SIZES[1],))),
        algorithm="grpo",
        components=(component(),),
        minimum_fragments=P,
        max_fragment_bytes=64,
        max_chunk_bytes=8,  # two floats per chunk: fragment 0 spans two chunks
    )


def _build_cut(adapter, values, *, policy_version, generation, ray) -> FullParameterChunkedCut:
    """A reference-backed Miles cut holding ``values[fragment_id]``."""
    shards = []
    for plan in adapter.plans:
        descriptors, refs = [], []
        for fragment in plan.fragments:
            value = values[fragment.fragment_id].to(torch.float32).contiguous()
            assert value.numel() == fragment.numel
            digest = _fragment_hasher(fragment.fragment_id, adapter.layout.layout_hash, plan.plan_hash)
            chunks, chunk_refs = [], []
            step = adapter.max_chunk_bytes // 4
            for index, start in enumerate(range(0, value.numel(), step)):
                chunk = value[start:start + step].clone().contiguous()
                digest.update(_tensor_bytes(chunk))
                chunks.append(FullParameterChunkDescriptor(
                    index, start, chunk.numel(),
                    _chunk_hash(fragment.fragment_id, index, start, chunk,
                                parameter_layout_hash=adapter.layout.layout_hash,
                                plan_hash=plan.plan_hash)))
                chunk_refs.append(ray.put(chunk))
            descriptors.append(FullParameterFragmentDescriptor(
                fragment.fragment_id, value.numel(), digest.hexdigest(), tuple(chunks)))
            refs.append(chunk_refs)
        shards.append(FullParameterChunkedShardState(
            policy_version, generation, plan.topology, adapter.layout.layout_hash,
            plan.plan_hash, tuple(descriptors), refs))
    return FullParameterChunkedCut(policy_version, generation, adapter.layout.layout_hash,
                                   tuple(sorted(shards, key=lambda state: state.topology)))


class _Trainer:
    """The Megatron trainer group: one f32 tensor per fragment, fixed progress per round."""

    def __init__(self, adapter, ray, initial, progress) -> None:
        self.adapter = adapter
        self.ray = ray
        self.values = {f: value.clone() for f, value in initial.items()}
        self.progress = progress
        self.installed = 0
        self.generation = 0
        self.scheduler_steps = 0
        self.trained_from: list[dict[int, torch.Tensor]] = []
        self.on_train = None  # hook: runs while a round trains (cut point)

    async def record_full_parameter_local_step(self, *, base_policy_version, rollout_id):
        assert base_policy_version == self.installed
        self.trained_from.append({f: v.clone() for f, v in self.values.items()})
        for f, delta in self.progress(rollout_id).items():
            self.values[f] = self.values[f] + delta
        self.generation += 1
        start, self.scheduler_steps = self.scheduler_steps, self.scheduler_steps + 1
        if self.on_train is not None:
            self.on_train(rollout_id)
        return tuple(
            SimpleNamespace(topology=shard.topology, role="actor", base_policy_version=base_policy_version,
                            local_step_generation=self.generation, rollout_id=rollout_id,
                            optimizer_steps=1, scheduler_start_steps=start,
                            scheduler_end_steps=self.scheduler_steps)
            for shard in self.adapter.manifests)

    async def export_full_parameter_chunked_cut(self, policy_version, local_step_generation=0, *,
                                                max_chunk_bytes):
        assert max_chunk_bytes == self.adapter.max_chunk_bytes
        return _build_cut(self.adapter, self.values, policy_version=policy_version,
                          generation=local_step_generation, ray=self.ray)

    async def apply_full_parameter_chunked_cut(self, cut, *, commit_token=None, max_commit_attempts=3):
        assert commit_token
        for shard in cut.shards:
            for descriptor, refs in zip(shard.fragments, shard.chunk_refs, strict=True):
                assert all(ref is not None for ref in refs), "applied a released chunk"
                self.values[descriptor.fragment_id] = torch.cat([self.ray.get(ref) for ref in refs]).clone()
        self.installed = cut.policy_version
        self.generation = 0
        return self.adapter.expected_parameter_tensor_count

    async def full_parameter_optimizer_states(self):
        return tuple(SimpleNamespace(role="actor", installed_policy_version=self.installed,
                                     local_step_generation=self.generation,
                                     scheduler_num_steps=self.scheduler_steps)
                     for _ in self.adapter.manifests)


class _Syncer:
    """In-process strict sweep (``--pipeline P``, checkpoint every round).

    ``G_f <- G_f + OTHER_WEIGHT * (island delta + peer delta)``; committed state
    is checkpointed at once, so a restart loses only uncommitted pushes."""

    def __init__(self, initial, sweeps, peer) -> None:
        self.global_ = {f: v.clone() for f, v in initial.items()}
        self.versions = [0] * P
        self.sweeps = sweeps
        self.peer = peer  # rollout_id -> {fragment: peer delta}
        self.received: dict[int, torch.Tensor] = {}  # uncommitted pushes by step
        self.pushed: list[tuple[int, int, torch.Tensor]] = []  # (step, attempt, delta), every PUSH seen
        self.committed: list[int] = []
        self.attempt = 1
        self.client = None

    def commit_ready(self) -> None:
        """Commit fragments in strict step order once their PUSH arrived."""
        while True:
            step = len(self.committed) + 1
            if step > self.sweeps * P or step not in self.received:
                return
            f, k = (step - 1) % P, (step - 1) // P
            delta = self.received.pop(step)
            self.global_[f] = self.global_[f] + OTHER_WEIGHT * (delta + self.peer(k)[f])
            self.versions[f] = step
            self.committed.append(step)
            if step == self.sweeps * P:
                self.client.publish_final()
            elif step % P == 0:
                # The sweep is committed: broadcast it, then PULL the next one.
                for fragment in range(P):
                    self.client.broadcast(fragment, self.versions[fragment], self.global_[fragment])
                self.client.pull_sweep(step // P, self.attempt)

    def restart_from_checkpoint(self) -> None:
        """Crash + ``--resume``: uncommitted pushes are lost; the reconnect
        re-broadcasts every committed fragment and re-sends the open PULLs."""
        self.received.clear()
        self.attempt += 1
        self.client.reconnect()
        for fragment in range(P):
            if self.versions[fragment]:
                self.client.broadcast(fragment, self.versions[fragment], self.global_[fragment])
        if len(self.committed) % P == 0 and len(self.committed) < self.sweeps * P:
            self.client.pull_sweep(len(self.committed) // P, self.attempt)
        else:
            # A partly committed sweep: only its uncommitted fragments are open.
            sweep = len(self.committed) // P
            for step in range(len(self.committed) + 1, (sweep + 1) * P + 1):
                self.client.pulls.append(PullRequest((step - 1) % P, step, self.attempt))


class _Client:
    """The ``SyncerClient`` surface the role stream uses, backed by ``_Syncer``."""

    def __init__(self, syncer: _Syncer) -> None:
        self.syncer = syncer
        syncer.client = self
        self.connection_generation = 1
        self.finalizing = threading.Event()
        self.pulls: list[PullRequest] = []
        self.updates: list[BcastFragment] = []
        self.inbound = None
        self.final: tuple[FinalManifest, list[FinalFragment]] | None = None
        self.acknowledged: FinalManifest | None = None
        self.closed = False

    # -- syncer side
    def _stored(self, fragment_id, version, value):
        return self.inbound.sink(fragment_id, version, _tensor_bytes(value.to(torch.float32).contiguous()))

    def broadcast(self, fragment_id, version, value) -> None:
        self.updates.append(BcastFragment(fragment_id, version, self._stored(fragment_id, version, value),
                                          stored=True))

    def pull_sweep(self, sweep, attempt) -> None:
        self.pulls.extend(PullRequest(f, sweep * P + f + 1, attempt) for f in range(P))

    def reconnect(self) -> None:
        self.connection_generation += 1
        self.pulls.clear()

    def publish_final(self) -> None:
        last = self.syncer.sweeps * P
        versions = tuple(range(last - P + 1, last + 1))
        fragments = [FinalFragment(f, versions[f], self._stored(f, versions[f], self.syncer.global_[f]),
                                   stored=True) for f in range(P)]
        self.final = (FinalManifest(last, versions), fragments)
        self.finalizing.set()

    # -- island side (SyncerClient API)
    def install_inbound_chunk_sink(self, sink) -> None:
        self.inbound = sink

    def start(self) -> None:
        pass

    def send_init_parts(self, fragment_id, parts) -> bool:
        payload = b"".join(bytes(part) for part in parts)
        value = torch.frombuffer(bytearray(payload), dtype=torch.float32)
        assert torch.equal(value, self.syncer.global_[fragment_id])
        self.broadcast(fragment_id, 0, value)
        if fragment_id == P - 1:
            self.pull_sweep(0, 1)
        return True

    def check_health(self) -> None:
        assert not self.closed

    def drain_updates(self):
        out, self.updates = self.updates, []
        return out

    def drain_pulls(self):
        out, self.pulls = self.pulls, []
        return out

    def push_fragment_parts(self, fragment_id, global_step, round_attempt, base_version, local_step,
                            c_steps, c_tokens, parts) -> bool:
        assert (global_step - 1) % P == fragment_id and c_steps == 1 and c_tokens >= 1
        assert base_version == self.syncer.versions[fragment_id], "PUSH on a stale base"
        delta = torch.frombuffer(bytearray(b"".join(bytes(part) for part in parts)), dtype=torch.float32)
        self.syncer.pushed.append((global_step, round_attempt, delta.clone()))
        self.syncer.received[global_step] = delta.clone()
        self.syncer.commit_ready()
        return True

    def wait_for_final_fragments(self, timeout):
        assert self.final is not None, "no FINAL before the terminal barrier"
        return self.final

    def acknowledge_finalization(self, manifest) -> None:
        self.acknowledged = manifest

    def close(self) -> None:
        self.closed = True


class _Driver:
    def __init__(self) -> None:
        self.events = []

    def emit(self, event, **fields) -> None:
        self.events.append((event, fields))


def _config(sweeps: int) -> RoleStreamConfig:
    adapter = _adapter()
    return RoleStreamConfig(
        algorithm="grpo", role="actor", component=component(), syncer_addr=("127.0.0.1", 30100),
        learner_id=0, learner_generation=0, learner_generations=(0, 0),
        total_fragment_steps=sweeps * P, expected_fragments=P,
        expected_layout_hash=adapter.layout.layout_hash, local_horizon=1, optimizer_steps_per_round=1,
        training_contract_hash="2" * 64, syncer_profile_hash="3" * 64, pipeline_depth=1,
        algorithm_spec_sha256="a1" * 32, wait_timeout=5.0, poll_seconds=0.001,
        max_fragment_bytes=64, max_chunk_bytes=8, overlap=True)


def _run(monkeypatch, sweeps: int, *, cut=None):
    """Run ``sweeps`` overlapped rounds; ``cut(rollout_id, syncer)`` runs while
    that round trains (the mid-overlap cut point)."""
    generator = torch.Generator().manual_seed(11)

    def rand(scale=1.0):
        return {f: torch.randn(n, generator=generator) * scale for f, n in enumerate(SIZES)}

    initial = rand()
    steps = {k: rand(0.1) for k in range(sweeps)}
    peer = {k: rand(0.1) for k in range(sweeps)}
    ray = FakeRay()
    adapter = _adapter()
    trainer = _Trainer(adapter, ray, initial, lambda k: steps[k])
    syncer = _Syncer(initial, sweeps, lambda k: peer[k])
    client = _Client(syncer)

    async def initialize(cls, group, *, policy_version, **_kwargs):
        cut = await group.export_full_parameter_chunked_cut(policy_version, 0,
                                                            max_chunk_bytes=adapter.max_chunk_bytes)
        return adapter, adapter.validate_cut(cut, policy_version, 0)

    monkeypatch.setattr(MilesChunkedFullParameterAdapter, "initialize", classmethod(initialize))
    monkeypatch.setattr(role_stream_module, "SyncerClient", lambda *_args, **_kwargs: client)

    stream = MilesFullParameterRoleStream(_config(sweeps), ray_module=ray)
    profile = FullParameterSyncProfile.from_configs({"actor": stream.config}, sweep=True)
    assert profile.overlap
    session = FullParameterSync(
        RolePairing({"actor": stream}, profile), groups={"actor": trainer}, run=asyncio.run,
        trajectories=lambda rollout_id: _evidence(rollout_id=rollout_id,
                                                  behavior_policy_hash=stream.current.policy_hash))
    if cut is not None:
        trainer.on_train = lambda rollout_id: cut(rollout_id, syncer)
    driver = _Driver()
    session.start(driver)
    stops = []
    for rollout_id in range(sweeps):
        boundary = session.boundary(driver, rollout_id=rollout_id, stats=None)
        stops.append(boundary.stop)
        if boundary.stop:
            session.published(driver, rollout_id=rollout_id, policy_hash=boundary.state.policy_tensor_hash())
            break
    session.finish(driver)
    return SimpleNamespace(initial=initial, steps=steps, peer=peer, trainer=trainer, syncer=syncer,
                           client=client, stream=stream, stops=stops, ray=ray)


def _expected_global(run, sweeps):
    return {f: run.initial[f] + sum(OTHER_WEIGHT * (run.steps[k][f] + run.peer[k][f])
                                    for k in range(sweeps)) for f in range(P)}


def _committed_deltas(run):
    """The delta of each committed step (the last PUSH seen for that step)."""
    last = {}
    for step, attempt, delta in run.syncer.pushed:
        if attempt >= last.get(step, (0, None))[0]:
            last[step] = (attempt, delta)
    return {step: delta for step, (_attempt, delta) in last.items()}


def _check_exactly_once(run, sweeps):
    assert run.stops == [False] * (sweeps - 1) + [True]
    assert run.syncer.committed == list(range(1, sweeps * P + 1))
    deltas = _committed_deltas(run)
    for k in range(sweeps):
        for f in range(P):
            # Each push is exactly that round's progress: no round is pushed
            # twice, none is lost, the rebase kept the local progress.
            torch.testing.assert_close(deltas[k * P + f + 1], run.steps[k][f], rtol=0, atol=1e-6)
    want = _expected_global(run, sweeps)
    for f in range(P):
        torch.testing.assert_close(run.syncer.global_[f], want[f], rtol=0, atol=1e-5)
        torch.testing.assert_close(run.trainer.values[f], want[f], rtol=0, atol=1e-5)
    assert run.stream.final_applied and run.stream.final_acknowledged
    assert run.client.acknowledged == run.stream.terminal_manifest


def test_overlap_on_the_miles_role_stream_rebases_and_counts_every_round_once(monkeypatch):
    sweeps = 4
    run = _run(monkeypatch, sweeps)
    _check_exactly_once(run, sweeps)
    trained = run.trainer.trained_from
    for f in range(P):
        # Round 0 starts at the initial weights, round 1 at its own round-0
        # result (sweep 0 in flight), round k >= 2 at G_{k-2} + progress of
        # round k-1: the rebase ran in the trainer through rebase_fragment.
        torch.testing.assert_close(trained[0][f], run.initial[f])
        torch.testing.assert_close(trained[1][f], run.initial[f] + run.steps[0][f])
        for k in range(2, sweeps):
            g = run.initial[f] + sum(OTHER_WEIGHT * (run.steps[j][f] + run.peer[j][f]) for j in range(k - 1))
            torch.testing.assert_close(trained[k][f], g + run.steps[k - 1][f], rtol=0, atol=1e-5)
    # Exactly one PUSH per step without a restart.
    assert sorted(step for step, _attempt, _delta in run.syncer.pushed) == list(range(1, sweeps * P + 1))


def test_overlap_role_stream_releases_every_superseded_cut(monkeypatch):
    run = _run(monkeypatch, 3)
    stream = run.stream
    assert not stream._pinned_cuts and not stream._pinned_anchors
    assert not stream._staged


@pytest.mark.parametrize("cut_round", [1, 2])
def test_syncer_restart_before_the_commit_mid_overlap_replays_the_retained_push(monkeypatch, cut_round):
    """Cut while round ``cut_round`` trains: sweep ``cut_round - 1`` was pushed
    but not committed. The resumed syncer re-PULLs it with attempt 2; the
    island replays the retained bytes at its next boundary."""
    sweeps = 4

    def cut(rollout_id, syncer):
        if rollout_id == cut_round and syncer.attempt == 1:
            # Undo the commit of the in-flight sweep: the crash came before it.
            sweep = cut_round - 1
            for step in range(sweep * P + 1, (sweep + 1) * P + 1):
                syncer.committed.remove(step)
            syncer.global_ = {f: g.clone() for f, g in checkpoints[sweep].items()}
            syncer.versions = [sweep * P - P + f + 1 if sweep else 0 for f in range(P)]
            syncer.client.updates.clear()  # its BCAST never left the syncer
            syncer.restart_from_checkpoint()

    checkpoints = {}
    original = _Syncer.commit_ready

    def commit_ready(self):
        if len(self.committed) % P == 0:
            checkpoints.setdefault(len(self.committed) // P, {f: g.clone() for f, g in self.global_.items()})
        original(self)

    monkeypatch.setattr(_Syncer, "commit_ready", commit_ready)
    run = _run(monkeypatch, sweeps, cut=cut)
    _check_exactly_once(run, sweeps)
    replayed = [(step, attempt) for step, attempt, _ in run.syncer.pushed if attempt == 2]
    first = (cut_round - 1) * P + 1
    assert [step for step, _ in replayed][:P] == list(range(first, first + P))


@pytest.mark.parametrize("cut_round", [1, 2])
def test_syncer_restart_after_the_commit_mid_overlap_rebases_on_the_resent_broadcast(monkeypatch, cut_round):
    """Cut while round ``cut_round`` trains: sweep ``cut_round - 1`` committed
    and checkpointed, but its BCAST was lost. The resumed syncer re-sends it
    (FULL) and PULLs the next sweep; the island rebases on the re-sent bytes."""
    sweeps = 4

    def cut(rollout_id, syncer):
        if rollout_id == cut_round and syncer.attempt == 1:
            syncer.client.updates.clear()  # the BCAST was lost with the connection
            syncer.restart_from_checkpoint()

    run = _run(monkeypatch, sweeps, cut=cut)
    _check_exactly_once(run, sweeps)
    assert all(attempt == 1 for step, attempt, _ in run.syncer.pushed if step <= cut_round * P)


def test_syncer_restart_with_a_partly_committed_sweep_mid_overlap(monkeypatch):
    """Cut while round 2 trains: fragment 0 of sweep 1 committed (and is
    re-broadcast), fragment 1 was lost. Only fragment 1 is re-PULLed and
    replayed; the island applies the sweep only once both fragments arrived."""
    sweeps = 3
    lost = 2 * P  # sweep 1, fragment 1

    def cut(rollout_id, syncer):
        if rollout_id == 2 and syncer.attempt == 1:
            f = (lost - 1) % P
            pushed = [delta for step, _attempt, delta in syncer.pushed if step == lost][-1]
            syncer.committed.remove(lost)
            syncer.global_[f] = syncer.global_[f] - OTHER_WEIGHT * (pushed + syncer.peer(1)[f])
            syncer.versions[f] = lost - P
            syncer.client.updates.clear()
            syncer.restart_from_checkpoint()

    run = _run(monkeypatch, sweeps, cut=cut)
    _check_exactly_once(run, sweeps)
    assert [step for step, attempt, _ in run.syncer.pushed if attempt == 2][0] == lost
