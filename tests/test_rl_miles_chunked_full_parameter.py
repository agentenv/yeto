from __future__ import annotations

# ruff: noqa: I001 -- install the optional bridge stub before Miles import.

import asyncio
import hashlib
import socket
import struct
import subprocess
import sys
import types
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

bridge_stub = types.ModuleType("yeto.rl.deepseek_v4_bridge")
bridge_stub.ensure_deepseek_v4_bridge = lambda: None
sys.modules.setdefault("yeto.rl.deepseek_v4_bridge", bridge_stub)

from miles.backends.megatron_utils.full_parameter_state import (
    FullParameterChunkDescriptor,
    FullParameterChunkedShardState,
    FullParameterFragmentDescriptor,
    FullParameterShardManifest,
    FullParameterShardSpec,
    FullParameterTopology,
    _chunk_hash,
    _fragment_hasher,
    _layout_hash,
    _tensor_bytes,
)
from miles.ray.full_parameter_transport import FullParameterChunkedCut
from yeto.rl.local_learner import ComponentIdentity
from yeto.rl.contracts import TrajectoryEnvelope
from yeto.protocol import (
    _CHUNK_HEAD,
    _HEADER,
    DTYPE_F32,
    MAGIC,
    MSG_BCAST_FRAGMENT,
    MSG_FINAL_FRAGMENT,
    SyncerClient,
)
from yeto.rl.dense_sweep_wire import _MilesRayInboundChunkSink
from yeto.rl.dense_sweep_wire import DenseSweepConfig, DenseSweepWire
from yeto.rl.miles_chunked_full_parameter import (
    MilesChunkedFullParameterAdapter,
)
from yeto.rl.trajectory_evidence import TrajectoryBatchEvidence


class FakeRay:
    def __init__(self) -> None:
        self.objects = {}

    def put(self, value):
        reference = object()
        self.objects[reference] = value.clone()
        return reference

    def get(self, reference):
        return self.objects[reference]

    def drop(self, reference):
        self.objects.pop(reference, None)


def topology(rank: int) -> FullParameterTopology:
    return FullParameterTopology(
        tp_rank=rank,
        tp_size=2,
        pp_rank=0,
        pp_size=1,
        ep_rank=0,
        ep_size=1,
        cp_rank=0,
        cp_size=1,
        dp_rank=0,
        dp_size=1,
    )


def manifest(
    rank: int,
    sizes: tuple[int, ...],
    *,
    role: str = "actor",
) -> FullParameterShardManifest:
    owner = topology(rank)
    specs = tuple(
        sorted(
            FullParameterShardSpec(
                role=role,
                shard_id=owner.shard_id,
                name=f"p{index}",
                shape=(size,),
                dtype="float32",
                numel=size,
            )
            for index, size in enumerate(sizes)
        )
    )
    return FullParameterShardManifest(
        owner,
        role,
        _layout_hash(owner, specs),
        specs,
    )


def component(role: str = "actor") -> ComponentIdentity:
    return ComponentIdentity(role, "a" * 40, "b" * 64)


def test_production_layout_is_owner_affine_and_enforces_fragment_bound():
    adapter = MilesChunkedFullParameterAdapter.create(
        (manifest(0, (100, 1)), manifest(1, (99, 1))),
        algorithm="grpo",
        components=(component(),),
        minimum_fragments=2,
        max_fragment_bytes=400,
        max_chunk_bytes=64,
    )

    assert adapter.layout.fragment_strategy == "owner_affine"
    assert adapter.layout.fragments.num_fragments == 3
    assert all(
        len(
            {
                (spec.role, spec.shard_id)
                for spec in adapter.layout.fragment_specs(fragment_id)
            }
        )
        == 1
        for fragment_id in range(adapter.layout.fragments.num_fragments)
    )
    assert all(
        fragment.numel * 4 <= 400 for fragment in adapter.layout.fragments.fragments
    )


def test_sao_adapter_creates_independent_actor_and_critic_role_streams():
    actor = MilesChunkedFullParameterAdapter.create(
        (manifest(0, (8,)), manifest(1, (8,))),
        algorithm="sao",
        components=(component("actor"),),
        minimum_fragments=2,
        max_fragment_bytes=64,
        max_chunk_bytes=32,
        stream_role="actor",
    )
    critic = MilesChunkedFullParameterAdapter.create(
        (
            manifest(0, (8,), role="critic"),
            manifest(1, (8,), role="critic"),
        ),
        algorithm="sao",
        components=(component("critic"),),
        minimum_fragments=2,
        max_fragment_bytes=64,
        max_chunk_bytes=32,
        stream_role="critic",
    )

    assert actor.layout.stream_role == "actor"
    assert critic.layout.stream_role == "critic"
    assert {spec.role for spec in actor.layout.specs} == {"actor"}
    assert {spec.role for spec in critic.layout.specs} == {"critic"}
    assert actor.layout.layout_hash != critic.layout.layout_hash


def test_initialize_rejects_probe_layout_drift_before_install_or_export():
    class Group:
        installed = False

        async def full_parameter_shard_manifests(self):
            return (manifest(1, (99, 1)), manifest(0, (100, 1)))

        async def install_full_parameter_fragment_plans(self, _plans):
            self.installed = True
            raise AssertionError("layout drift must fail before plan installation")

    group = Group()
    with pytest.raises(
        RuntimeError,
        match=r"expected_fragments=2, actual_fragments=3",
    ):
        asyncio.run(
            MilesChunkedFullParameterAdapter.initialize(
                group,
                policy_version=0,
                algorithm="grpo",
                components=(component(),),
                minimum_fragments=1,
                expected_fragments=2,
                expected_layout_hash="c" * 64,
                max_fragment_bytes=400,
                max_chunk_bytes=64,
            )
        )
    assert group.installed is False


def _cut(adapter, values, *, policy_version, generation, ray):
    shards = []
    for plan in adapter.plans:
        descriptors = []
        refs = []
        for fragment in plan.fragments:
            value = torch.full(
                (fragment.numel,),
                float(values[fragment.fragment_id]),
                dtype=torch.float32,
            )
            digest = _fragment_hasher(
                fragment.fragment_id,
                adapter.layout.layout_hash,
                plan.plan_hash,
            )
            chunks = []
            chunk_refs = []
            chunk_numel = adapter.max_chunk_bytes // 4
            for chunk_index, start in enumerate(range(0, value.numel(), chunk_numel)):
                chunk_value = value[start : start + chunk_numel].clone().contiguous()
                digest.update(_tensor_bytes(chunk_value))
                chunks.append(
                    FullParameterChunkDescriptor(
                        chunk_index,
                        start,
                        chunk_value.numel(),
                        _chunk_hash(
                            fragment.fragment_id,
                            chunk_index,
                            start,
                            chunk_value,
                            parameter_layout_hash=adapter.layout.layout_hash,
                            plan_hash=plan.plan_hash,
                        ),
                    )
                )
                chunk_refs.append(ray.put(chunk_value))
            descriptors.append(
                FullParameterFragmentDescriptor(
                    fragment.fragment_id,
                    value.numel(),
                    digest.hexdigest(),
                    tuple(chunks),
                )
            )
            refs.append(chunk_refs)
        shards.append(
            FullParameterChunkedShardState(
                policy_version,
                generation,
                plan.topology,
                adapter.layout.layout_hash,
                plan.plan_hash,
                tuple(descriptors),
                refs,
            )
        )
    return FullParameterChunkedCut(
        policy_version,
        generation,
        adapter.layout.layout_hash,
        tuple(sorted(shards, key=lambda state: state.topology)),
    )


def _evidence(*, rollout_id: int, behavior_policy_hash: str):
    trajectory = TrajectoryEnvelope(
        trajectory_id=f"trajectory-{rollout_id}",
        task_id="CVE-2026-0001",
        prompt_group_id=f"r{rollout_id}:g0",
        sample_index=0,
        behavior_policy_version=rollout_id,
        behavior_policy_hash=behavior_policy_hash,
        token_ids=(1, 2, 3),
        response_token_count=2,
        behavior_logprobs_hash="c" * 64,
        reward=1.0,
        reward_contract_hash="d" * 64,
        cleanup_evidence_hash="e" * 64,
    )
    return TrajectoryBatchEvidence(
        rollout_id,
        behavior_policy_hash,
        "f" * 64,
        2,
        (trajectory,),
    )


class _ReceiptGroup:
    def __init__(
        self,
        adapter,
        *,
        role: str,
        local_step_generation: int,
        optimizer_steps: int,
    ) -> None:
        self.adapter = adapter
        self.role = role
        self.local_step_generation = local_step_generation
        self.optimizer_steps = optimizer_steps
        self.calls = []

    async def record_full_parameter_local_step(
        self,
        *,
        base_policy_version,
        rollout_id,
    ):
        self.calls.append((base_policy_version, rollout_id))
        return tuple(
            SimpleNamespace(
                topology=manifest.topology,
                role=self.role,
                base_policy_version=base_policy_version,
                local_step_generation=self.local_step_generation,
                rollout_id=rollout_id,
                optimizer_steps=self.optimizer_steps,
                scheduler_start_steps=10,
                scheduler_end_steps=20,
            )
            for manifest in self.adapter.manifests
        )


@pytest.mark.asyncio
async def test_sao_local_round_receipts_are_role_generic_and_content_bound():
    ray = FakeRay()
    actor_adapter = MilesChunkedFullParameterAdapter.create(
        (manifest(0, (3,)), manifest(1, (2,))),
        algorithm="sao",
        components=(component("actor"),),
        minimum_fragments=2,
        max_fragment_bytes=64,
        max_chunk_bytes=16,
        stream_role="actor",
    )
    actor_anchor = actor_adapter.validate_cut(
        _cut(
            actor_adapter,
            {0: 0, 1: 0},
            policy_version=4,
            generation=0,
            ray=ray,
        ),
        4,
        0,
    )
    relabeled = actor_adapter.validate_cut(
        _cut(
            actor_adapter,
            {0: 0, 1: 0},
            policy_version=9,
            generation=3,
            ray=ray,
        ),
        9,
        3,
    )
    assert actor_anchor.policy_hash != relabeled.policy_hash
    assert actor_anchor.content_hash == relabeled.content_hash
    actor_group = _ReceiptGroup(
        actor_adapter,
        role="actor",
        local_step_generation=4,
        optimizer_steps=3,
    )
    actor_receipt = await actor_adapter.record_local_round(
        actor_group,
        anchor=relabeled,
        rollout_id=17,
        learner_id=2,
        learner_generation=5,
        trajectories=_evidence(
            rollout_id=17,
            behavior_policy_hash=relabeled.content_hash,
        ),
        role="actor",
        expected_local_step_generation=4,
        expected_optimizer_steps=3,
    )
    assert actor_group.calls == [(9, 17)]
    assert actor_receipt.algorithm == "sao"
    assert actor_receipt.base_policy_hash == actor_anchor.content_hash
    assert actor_receipt.optimizer_steps == 3

    critic_adapter = MilesChunkedFullParameterAdapter.create(
        (
            manifest(0, (3,), role="critic"),
            manifest(1, (2,), role="critic"),
        ),
        algorithm="sao",
        components=(component("critic"),),
        minimum_fragments=2,
        max_fragment_bytes=64,
        max_chunk_bytes=16,
        stream_role="critic",
    )
    critic_anchor = critic_adapter.validate_cut(
        _cut(
            critic_adapter,
            {0: 2, 1: 2},
            policy_version=4,
            generation=6,
            ray=ray,
        ),
        4,
        6,
    )
    critic_group = _ReceiptGroup(
        critic_adapter,
        role="critic",
        local_step_generation=7,
        optimizer_steps=2,
    )
    critic_receipt = await critic_adapter.record_local_round(
        critic_group,
        anchor=critic_anchor,
        rollout_id=17,
        learner_id=2,
        learner_generation=5,
        # A critic consumes the actor's batch; its behavior identity is not the
        # critic parameter identity.
        trajectories=_evidence(
            rollout_id=17,
            behavior_policy_hash=actor_anchor.content_hash,
        ),
        role="critic",
        expected_local_step_generation=7,
        expected_optimizer_steps=2,
    )
    assert critic_receipt.base_policy_hash == critic_anchor.content_hash
    assert critic_receipt.input_batch_hash == actor_receipt.input_batch_hash
    assert critic_receipt.trajectory_ids == actor_receipt.trajectory_ids


@pytest.mark.asyncio
async def test_sao_actor_local_round_rejects_wrong_behavior_or_rank_progress():
    ray = FakeRay()
    adapter = MilesChunkedFullParameterAdapter.create(
        (manifest(0, (3,)), manifest(1, (2,))),
        algorithm="sao",
        components=(component("actor"),),
        minimum_fragments=2,
        max_fragment_bytes=64,
        max_chunk_bytes=16,
        stream_role="actor",
    )
    anchor = adapter.validate_cut(
        _cut(adapter, {0: 0, 1: 0}, policy_version=2, generation=1, ray=ray),
        2,
        1,
    )
    group = _ReceiptGroup(
        adapter,
        role="actor",
        local_step_generation=2,
        optimizer_steps=1,
    )
    with pytest.raises(ValueError, match="does not bind"):
        await adapter.record_local_round(
            group,
            anchor=anchor,
            rollout_id=8,
            learner_id=0,
            learner_generation=0,
            trajectories=_evidence(
                rollout_id=8,
                behavior_policy_hash="0" * 64,
            ),
            role="actor",
            expected_local_step_generation=2,
            expected_optimizer_steps=1,
        )
    assert group.calls == []

    group.local_step_generation = 3
    with pytest.raises(RuntimeError, match="receipt changed"):
        await adapter.record_local_round(
            group,
            anchor=anchor,
            rollout_id=8,
            learner_id=0,
            learner_generation=0,
            trajectories=_evidence(
                rollout_id=8,
                behavior_policy_hash=anchor.content_hash,
            ),
            role="actor",
            expected_local_step_generation=2,
            expected_optimizer_steps=1,
        )


@pytest.mark.asyncio
async def test_grpo_h1_wrapper_preserves_versioned_policy_binding():
    ray = FakeRay()
    adapter = MilesChunkedFullParameterAdapter.create(
        (manifest(0, (3,)), manifest(1, (2,))),
        algorithm="grpo",
        components=(component(),),
        minimum_fragments=2,
        max_fragment_bytes=64,
        max_chunk_bytes=16,
    )
    anchor = adapter.validate_cut(
        _cut(adapter, {0: 0, 1: 0}, policy_version=4, generation=0, ray=ray),
        4,
        0,
    )
    group = _ReceiptGroup(
        adapter,
        role="actor",
        local_step_generation=1,
        optimizer_steps=1,
    )

    receipt = await adapter.record_grpo_local_step(
        group,
        anchor=anchor,
        rollout_id=4,
        learner_id=0,
        learner_generation=0,
        trajectories=_evidence(
            rollout_id=4,
            behavior_policy_hash=anchor.policy_hash,
        ),
    )

    assert receipt.algorithm == "grpo"
    assert receipt.base_policy_hash == anchor.policy_hash
    assert receipt.optimizer_steps == 1


def test_inbound_chunks_commit_directly_to_exact_typed_miles_refs():
    ray = FakeRay()
    adapter = MilesChunkedFullParameterAdapter.create(
        (manifest(0, (32,)), manifest(1, (32,))),
        algorithm="grpo",
        components=(component(),),
        minimum_fragments=2,
        max_fragment_bytes=256,
        max_chunk_bytes=16,
    )
    anchor = adapter.validate_cut(
        _cut(adapter, {0: 0, 1: 0}, policy_version=0, generation=0, ray=ray),
        0,
        0,
    )
    anchor_refs = {
        reference
        for shard in anchor.transport_cut.shards
        for references in shard.chunk_refs
        for reference in references
    }
    sink = adapter.fragment_sink(anchor, ray_module=ray)
    sink_refs = {
        reference
        for shard in sink.reference.transport_cut.shards
        for references in shard.chunk_refs
        for reference in references
    }
    assert sink_refs.isdisjoint(anchor_refs)
    assert adapter.release(anchor) == len(anchor_refs)
    client = SyncerClient(
        ("unused", 0),
        0,
        adapter.layout.fragments,
        dtype=DTYPE_F32,
        num_streams=2,
    )
    client._gen = 7
    client.install_inbound_chunk_sink(_MilesRayInboundChunkSink(sink))
    expected_descriptor = next(
        descriptor
        for shard in anchor.transport_cut.shards
        for descriptor in shard.fragments
        if descriptor.fragment_id == 0
    )
    tensor = torch.zeros(expected_descriptor.numel, dtype=torch.float32)

    def receive(msg_id, msg_type, value):
        payload = struct.pack("<IQ", 0, 0) + _tensor_bytes(value)
        inner = _HEADER.pack(MAGIC, msg_type, len(payload)) + payload
        cuts = ((0, 31), (31, 80), (80, len(inner)))
        completed = None
        for index in (2, 0, 1):
            start, stop = cuts[index]
            result = client._reassemble(
                7,
                _CHUNK_HEAD.pack(msg_id, len(inner), start) + inner[start:stop],
            )
            if result is not None:
                completed = result
        assert completed is not None
        client._dispatch_streamed(7, completed)

    receive(10, MSG_BCAST_FRAGMENT, tensor)
    updates = client.drain_updates()
    assert len(updates) == 1 and updates[0].stored
    stored = updates[0].data
    assert stored is sink._stored[(0, 0)]
    assert stored.descriptor == expected_descriptor
    assert stored.wire_payload_hash == hashlib.sha256(_tensor_bytes(tensor)).hexdigest()
    assert all(value.dtype == torch.float32 for value in ray.objects.values())

    object_count = len(ray.objects)
    receive(11, MSG_BCAST_FRAGMENT, tensor)
    assert client.drain_updates() == []
    assert len(ray.objects) == object_count

    receive(12, MSG_FINAL_FRAGMENT, tensor)
    assert client._final_fragments[0].stored
    assert client._final_fragments[0].data is stored
    assert len(ray.objects) == object_count

    with pytest.raises(RuntimeError, match="replay changed bytes"):
        receive(13, MSG_BCAST_FRAGMENT, torch.ones_like(tensor))
    assert len(ray.objects) == object_count

    payload = struct.pack("<IQ", 0, 1) + _tensor_bytes(torch.ones_like(tensor))
    inner = _HEADER.pack(MAGIC, MSG_BCAST_FRAGMENT, len(payload)) + payload
    assert (
        client._reassemble(
            7,
            _CHUNK_HEAD.pack(14, len(inner), 80) + inner[80:],
        )
        is None
    )
    assert not isinstance(client._reasm[14], tuple)
    assert any(value.dtype == torch.uint8 for value in ray.objects.values())
    with pytest.raises(ValueError, match="chunk total changed"):
        client._reassemble(
            7,
            _CHUNK_HEAD.pack(14, len(inner) - 1, 0) + inner[:31],
        )
    assert 14 not in client._reasm
    assert len(ray.objects) == object_count

    transactional = _MilesRayInboundChunkSink(sink)
    aborted = transactional.begin_message(15, 1)
    transactional.bind_fragment(
        aborted,
        MSG_BCAST_FRAGMENT,
        1,
        1,
        tensor.numel() * tensor.element_size(),
    )
    before_abort = len(ray.objects)
    transactional.consume_chunk(aborted, 0, _tensor_bytes(tensor))
    assert len(ray.objects) == before_abort + len(expected_descriptor.chunks)
    transactional.abort_message(aborted, ())
    assert len(ray.objects) == before_abort
    transactional.release_all()
    assert sink._stored == {}
    assert len(ray.objects) == before_abort - len(expected_descriptor.chunks)


def test_real_syncer_initial_and_terminal_cuts_stream_to_miles_refs(tmp_path):
    root = Path(__file__).resolve().parent.parent
    subprocess.run(["cargo", "build", "-q"], cwd=root / "syncer", check=True)
    binary = root / "syncer/target/debug/yeto-syncer"
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    checkpoint = tmp_path / "streamed-sweep.ckpt"
    process = subprocess.Popen(
        [
            str(binary),
            "--port",
            str(port),
            "--learners",
            "1",
            "--quorum",
            "1",
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
            "--max-base-lag",
            "0",
            "--learner-weight",
            "equal",
            "--checkpoint-path",
            str(checkpoint),
            "--checkpoint-every",
            "1",
            "--resume",
            "--quorum-timeout-s",
            "10",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    ray = FakeRay()
    adapter = MilesChunkedFullParameterAdapter.create(
        (manifest(0, (32,)), manifest(1, (32,))),
        algorithm="grpo",
        components=(component(),),
        minimum_fragments=2,
        max_fragment_bytes=256,
        max_chunk_bytes=256,
    )
    anchor = adapter.validate_cut(
        _cut(adapter, {0: 0, 1: 0}, policy_version=0, generation=0, ray=ray),
        0,
        0,
    )
    local = adapter.validate_cut(
        _cut(adapter, {0: 1, 1: 1}, policy_version=0, generation=1, ray=ray),
        0,
        1,
    )
    wire = DenseSweepWire(
        adapter.layout.fragments,
        DenseSweepConfig(
            ("127.0.0.1", port),
            0,
            7,
            1,
            wan_streams=2,
            wait_timeout=15,
            poll_seconds=0.001,
            max_fragment_bytes=256,
        ),
    )
    initial_sink = adapter.fragment_sink(anchor, ray_module=ray)
    update_sink = adapter.fragment_sink(anchor, ray_module=ray)
    try:
        initial = wire.start(
            {
                fragment_id: adapter.fragment_parts(
                    anchor,
                    fragment_id,
                    ray_module=ray,
                )
                for fragment_id in range(2)
            },
            payload_sink=initial_sink,
        )
        adapter.verify_initial_fragments(anchor, initial)
        pending = wire.exchange(
            base_policy_version=0,
            trained_tokens=17,
            sweep_update_id="a" * 64,
            delta_parts={
                fragment_id: adapter.delta_parts(
                    anchor,
                    local,
                    fragment_id,
                    ray_module=ray,
                )
                for fragment_id in range(2)
            },
            payload_sink=update_sink,
        )
        assert pending.terminal
        assert all(
            item is update_sink._stored[(index, index + 1)]
            for index, item in enumerate(pending.payloads)
        )
        assert all(value.dtype == torch.float32 for value in ray.objects.values())
        wire.commit_applied(pending)
        assert process.wait(timeout=10) == 0
    finally:
        wire.close()
        initial_sink.release_all()
        update_sink.release_all()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


@pytest.mark.asyncio
async def test_reference_delta_sink_assemble_apply_and_release():
    ray = FakeRay()
    adapter = MilesChunkedFullParameterAdapter.create(
        (manifest(0, (3,)), manifest(1, (2,))),
        algorithm="grpo",
        components=(component(),),
        minimum_fragments=2,
        max_fragment_bytes=64,
        max_chunk_bytes=16,
    )
    anchor = adapter.validate_cut(
        _cut(adapter, {0: 0, 1: 0}, policy_version=0, generation=0, ray=ray),
        0,
        0,
    )
    local = adapter.validate_cut(
        _cut(adapter, {0: 1, 1: 1}, policy_version=0, generation=1, ray=ray),
        0,
        1,
    )
    for fragment_id in range(2):
        payload = b"".join(
            bytes(part)
            for part in adapter.delta_parts(
                anchor,
                local,
                fragment_id,
                ray_module=ray,
            )()
        )
        assert torch.equal(
            torch.frombuffer(bytearray(payload), dtype=torch.float32),
            torch.ones(adapter.layout.fragments.fragments[fragment_id].numel),
        )

    sink = adapter.fragment_sink(anchor, ray_module=ray)
    stored = []
    for fragment_id, fragment in enumerate(adapter.layout.fragments.fragments):
        value = torch.full((fragment.numel,), 2.0, dtype=torch.float32)
        item = sink(
            fragment_id,
            fragment_id + 1,
            _tensor_bytes(value),
        )
        assert (
            sink(
                fragment_id,
                fragment_id + 1,
                _tensor_bytes(value),
            )
            is item
        )
        stored.append(item)
    target = adapter.assemble_target(
        anchor,
        target_policy_version=1,
        fragments=stored,
    )
    group = SimpleNamespace(
        apply_full_parameter_chunked_cut=lambda *_args, **_kwargs: None
    )

    async def apply(cut, **_kwargs):
        assert cut is target.transport_cut
        return adapter.expected_parameter_tensor_count

    group.apply_full_parameter_chunked_cut = apply
    assert await adapter.apply(group, target) == 2
    assert sink.release_all() == 2
    assert sink.release_all() == 0


def test_arbitrary_horizon_delta_and_mixed_target_stay_reference_backed():
    ray = FakeRay()
    adapter = MilesChunkedFullParameterAdapter.create(
        (manifest(0, (3,)), manifest(1, (2,))),
        algorithm="sao",
        components=(component("actor"),),
        minimum_fragments=2,
        max_fragment_bytes=64,
        max_chunk_bytes=16,
        stream_role="actor",
    )
    anchor = adapter.validate_cut(
        _cut(adapter, {0: 0, 1: 0}, policy_version=4, generation=0, ray=ray),
        4,
        0,
    )
    local = adapter.validate_cut(
        _cut(adapter, {0: 5, 1: 5}, policy_version=4, generation=3, ray=ray),
        4,
        3,
    )
    sink = adapter.fragment_sink(anchor, ray_module=ray)
    authoritative = sink(
        0,
        11,
        _tensor_bytes(
            torch.full(
                (adapter.layout.fragments.fragments[0].numel,),
                2.0,
                dtype=torch.float32,
            )
        ),
    )

    payload = b"".join(
        bytes(part)
        for part in adapter.delta_parts_from_authoritative(
            authoritative,
            local,
            0,
            ray_module=ray,
        )()
    )
    assert torch.equal(
        torch.frombuffer(bytearray(payload), dtype=torch.float32),
        torch.full(
            (adapter.layout.fragments.fragments[0].numel,),
            3.0,
            dtype=torch.float32,
        ),
    )

    target = adapter.assemble_mixed_target(
        local,
        target_policy_version=5,
        authoritative_fragments=(authoritative,),
    )
    observed = []
    for fragment_id in range(adapter.layout.fragments.num_fragments):
        fragment = b"".join(
            bytes(part)
            for part in adapter.fragment_parts(
                target,
                fragment_id,
                ray_module=ray,
            )()
        )
        observed.append(torch.frombuffer(bytearray(fragment), dtype=torch.float32))
    assert torch.equal(observed[0], torch.full_like(observed[0], 2.0))
    assert torch.equal(observed[1], torch.full_like(observed[1], 5.0))
    assert target.policy_version == 5
    assert target.local_step_generation == 0

    with pytest.raises(ValueError, match="provenance changed"):
        adapter.assemble_mixed_target(
            local,
            target_policy_version=5,
            authoritative_fragments=(
                replace(authoritative, parameter_layout_hash="0" * 64),
            ),
        )
    with pytest.raises(ValueError, match="provenance changed"):
        adapter.delta_parts_from_authoritative(
            replace(authoritative, fragment_id=1),
            local,
            0,
            ray_module=ray,
        )
