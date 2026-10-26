"""verl binding of the full-parameter role stream (S27 B10 phase 2, V4/V5).

The engine-neutral stream (:class:`yeto.rl.engine.full_parameter_stream.
FullParameterRoleStream`) answers the unified syncer's PULL permits, stages its
broadcasts and keeps the anchors.  This module gives it a verl FSDP2 trainer:

* :class:`VerlFullParameterGroup` -- the trainer group the stream talks to: the
  verl actor workers' manifests, fragment export and apply
  (:mod:`.full_parameter` worker functions over ``VerlIsland.call_workers``)
  and the receipt of the round the island just trained;
* :class:`VerlChunkedFullParameterAdapter` -- the chunked cut protocol the
  stream expects (the one ``adapters/miles/models/chunked_full_parameter.py``
  defines): an owner-affine layout over the island's FP32 parameters, cuts of
  flat FP32 fragments held in the island driver's memory, ``local - anchor``
  push deltas and fragment applies;
* :class:`VerlInboundChunkSink` -- the ``SyncerClient`` transactional sink that
  streams BCAST/FINAL bytes into one flat FP32 tensor per fragment;
* :class:`VerlFullParameterRoleStream` -- the stream with the verl backend
  identity in its session contract.

Differences from the Miles binding, all deliberate:

* the cut lives in the island driver's process, not in a Ray object store:
  every capture gathers each parameter whole (``DTensor.full_tensor``) on the
  workers and copies it to the driver.  Simple and exact; it costs one model
  copy per capture over Ray (fine for the small models of S27 P4, a known cost
  for large ones);
* the layout carries no shard identity (``full_parameter`` module doc), so
  islands with different card counts share one session;
* an apply writes only the fragments it replaces, then exports them again and
  compares their hashes with the target (the proof that the trainer holds the
  target); verl offers no optimizer-state receipts, so the stream's optimizer
  proof is off (``_optimizer_states`` returns None);
* not offered: sync/train overlap, deferred or lazy capture, MoE expert token
  weights and peer recovery (refused at configuration);
* bf16 broadcast deltas (``--bcast-delta``, S27 G-DS) need nothing here: the
  ``SyncerClient`` decodes them into the f32 fragment before the inbound sink
  sees a byte (``yeto.downlink`` module doc), so the sink stays f32-only.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from yeto.protocol import MSG_BCAST_FRAGMENT, MSG_FINAL_FRAGMENT, StreamedInboundPayload
from yeto.rl.contracts import LocalStepReceipt
from yeto.rl.engine.full_parameter_stream import (
    FullParameterRoleStream,
    RoleStreamConfig,
    session_contract_hash,
)
from yeto.rl.engine.full_parameter_identity import (  # noqa: F401 - full_parameter_identity is re-exported
    full_parameter_identity,
    identity_sha256,
    is_sha256,
    local_init as local_init_rule,
)
from yeto.rl.engine.loop_runner import LoopRunner
from yeto.rl.local_learner import ParameterLayout

from . import full_parameter as fp
from .full_parameter import DEFAULT_MAX_FRAGMENT_BYTES  # noqa: F401 - the island's default bound

LAYOUT_PROBE_LOG_PREFIX = "[yeto-verl-layout-probe] "
_CUT_DOMAIN = b"yeto-verl-full-parameter-cut-v1\0"
_CONTENT_DOMAIN = b"yeto-verl-full-parameter-content-v1\0"


def refuse_unsupported(config: RoleStreamConfig) -> None:
    """The stream options a verl island does not offer, refused before any work."""

    refused = [name for name, on in (
        ("sync/train overlap (--sweep-overlap)", config.overlap),
        ("MoE expert token weights", config.expert_token_weights),
        ("peer recovery", config.peer_recovery is not None),
    ) if on]
    if config.algorithm != "grpo" or config.role != "actor":
        refused.append(f"{config.algorithm}/{config.role} (verl trains the GRPO actor only)")
    if refused:
        raise ValueError("verl 全参数岛不支持: " + "; ".join(refused))


# ----------------------------------------------------------------------------- cuts and fragments
def fragment_hash(flat: torch.Tensor) -> str:
    """SHA256 of one fragment's FP32 little-endian bytes (= its INIT/BCAST wire hash)."""

    return hashlib.sha256(_bytes(flat)).hexdigest()


def _bytes(flat: torch.Tensor) -> memoryview:
    return memoryview(flat.numpy()).cast("B")


def _cut_hashes(layout_hash: str, policy_version: int, local_step_generation: int,
                fragment_hashes: Sequence[str]) -> tuple[str, str]:
    """(policy hash, content hash): the policy hash names the version labels too."""

    joined = b"".join(bytes.fromhex(value) for value in fragment_hashes)
    content = hashlib.sha256(_CONTENT_DOMAIN + layout_hash.encode("ascii") + joined).hexdigest()
    policy = hashlib.sha256(
        _CUT_DOMAIN + layout_hash.encode("ascii") + int(policy_version).to_bytes(8, "little")
        + int(local_step_generation).to_bytes(8, "little") + joined).hexdigest()
    return policy, content


@dataclass(frozen=True)
class VerlFullParameterCut:
    """One complete cut of the island's parameters: one flat FP32 tensor per
    fragment, held in the island driver.  The tensors are never written after
    the cut is made (cuts and stored fragments share them)."""

    policy_version: int
    local_step_generation: int
    layout_hash: str
    policy_hash: str
    content_hash: str
    fragments: tuple[torch.Tensor, ...] = field(repr=False, compare=False)
    fragment_hashes: tuple[str, ...]

    @classmethod
    def create(cls, layout_hash: str, *, policy_version: int, local_step_generation: int,
               fragments: Sequence[torch.Tensor],
               fragment_hashes: Sequence[str] | None = None) -> VerlFullParameterCut:
        hashes = tuple(fragment_hashes) if fragment_hashes is not None else tuple(
            fragment_hash(flat) for flat in fragments)
        policy, content = _cut_hashes(layout_hash, policy_version, local_step_generation, hashes)
        return cls(int(policy_version), int(local_step_generation), layout_hash, policy, content,
                   tuple(fragments), hashes)


@dataclass
class VerlStoredFragment:
    """One authoritative fragment (a syncer broadcast, a FINAL fragment, or this
    island's own version-0 fragment under local init)."""

    fragment_id: int
    version: int
    wire_payload_hash: str
    flat: torch.Tensor | None = field(repr=False, compare=False)

    def release(self) -> int:
        if self.flat is None:
            return 0
        self.flat = None
        return 1

    def values(self) -> torch.Tensor:
        if self.flat is None:
            raise RuntimeError(f"authoritative fragment {self.fragment_id} v{self.version} was released")
        return self.flat


class VerlFragmentSink:
    """Idempotent store of authoritative fragments (``adapter.fragment_sink``)."""

    def __init__(self, adapter: VerlChunkedFullParameterAdapter) -> None:
        self.adapter = adapter
        self.lock = threading.Lock()
        self.stored: dict[tuple[int, int], VerlStoredFragment] = {}

    def __call__(self, fragment_id: int, version: int,
                 payload: bytes | bytearray | memoryview) -> VerlStoredFragment:
        raw = memoryview(payload).cast("B")
        numel = self.adapter.fragment_numel(fragment_id)
        if raw.nbytes != numel * 4:
            raise ValueError(f"fragment {fragment_id} payload has {raw.nbytes} bytes, expected {numel * 4}")
        flat = torch.frombuffer(bytearray(raw), dtype=torch.float32)
        return self.keep(VerlStoredFragment(fragment_id, version, hashlib.sha256(raw).hexdigest(), flat))

    def keep(self, stored: VerlStoredFragment) -> VerlStoredFragment:
        """``stored``, or the fragment already kept for its (id, version)."""

        if stored.flat is not None and not torch.isfinite(stored.flat).all().item():
            raise ValueError(f"authoritative fragment {stored.fragment_id} v{stored.version} has NaN/Inf")
        key = (stored.fragment_id, stored.version)
        with self.lock:
            previous = self.stored.get(key)
            if previous is not None and previous.flat is not None:
                if previous.wire_payload_hash != stored.wire_payload_hash:
                    raise RuntimeError("authoritative fragment replay changed bytes")
                return previous
            self.stored[key] = stored
            return stored

    def adopt(self, cut: VerlFullParameterCut, fragment_id: int, version: int,
              wire_payload_hash: str) -> VerlStoredFragment:
        """Local init: ``cut``'s own fragment is the version-0 anchor (shared, not copied)."""

        if cut.layout_hash != self.adapter.layout.layout_hash:
            raise ValueError("adopted cut layout changed")
        if cut.fragment_hashes[fragment_id] != wire_payload_hash:
            raise ValueError("adopted fragment hash differs from the cut's")
        return self.keep(VerlStoredFragment(fragment_id, version, wire_payload_hash,
                                            cut.fragments[fragment_id]))

    def forget(self, stored: VerlStoredFragment) -> None:
        with self.lock:
            if self.stored.get((stored.fragment_id, stored.version)) is stored:
                self.stored.pop((stored.fragment_id, stored.version))

    def release_all(self) -> int:
        with self.lock:
            stored, self.stored = list(self.stored.values()), {}
        return sum(value.release() for value in stored)


@dataclass
class _Transaction:
    message_id: int
    total_bytes: int
    msg_type: int | None = None
    fragment_id: int | None = None
    version: int | None = None
    expected: int = 0
    consumed: int = 0
    buffer: bytearray | None = None
    hasher: Any = None
    finished: bool = False


class VerlInboundChunkSink:
    """Stream one inbound BCAST/FINAL fragment into a flat FP32 tensor.

    Only a complete fragment of exactly ``numel * 4`` bytes, all finite,
    becomes visible (``finish_message``); an aborted message leaves nothing."""

    def __init__(self, sink: VerlFragmentSink) -> None:
        if not isinstance(sink, VerlFragmentSink):
            raise TypeError("verl inbound sink needs a VerlFragmentSink")
        self.sink = sink

    def begin_message(self, message_id: int, total_bytes: int) -> _Transaction:
        return _Transaction(message_id, total_bytes)

    def bind_fragment(self, transaction: _Transaction, msg_type: int, fragment_id: int,
                      version: int, payload_bytes: int) -> None:
        if transaction.msg_type is not None:
            raise RuntimeError("verl inbound transaction was bound twice")
        if msg_type not in (MSG_BCAST_FRAGMENT, MSG_FINAL_FRAGMENT):
            raise ValueError("verl inbound transaction has an invalid message type")
        if payload_bytes != self.sink.adapter.fragment_numel(fragment_id) * 4:
            raise ValueError("verl inbound fragment byte count changed (f32 broadcasts only)")
        transaction.msg_type, transaction.fragment_id, transaction.version = msg_type, fragment_id, version
        transaction.expected = payload_bytes
        transaction.buffer = bytearray(payload_bytes)
        transaction.hasher = hashlib.sha256()

    def stage_chunk(self, transaction: _Transaction, payload) -> bytes:
        del transaction
        return bytes(memoryview(payload).cast("B"))

    def consume_staged_chunk(self, transaction: _Transaction, offset: int, token: bytes,
                             nbytes: int) -> None:
        if len(token) != nbytes:
            raise ValueError("staged inbound chunk changed size")
        self.consume_chunk(transaction, offset, token)

    def consume_chunk(self, transaction: _Transaction, offset: int, payload) -> None:
        if transaction.buffer is None:
            raise RuntimeError("verl inbound transaction is not bound")
        if offset != transaction.consumed:
            raise ValueError("verl inbound chunks are not contiguous")
        raw = memoryview(payload).cast("B")
        end = offset + raw.nbytes
        if end > transaction.expected:
            raise ValueError("verl inbound fragment exceeds its payload bound")
        transaction.buffer[offset:end] = raw
        transaction.hasher.update(raw)
        transaction.consumed = end

    def finish_message(self, transaction: _Transaction) -> StreamedInboundPayload:
        if (transaction.finished or transaction.buffer is None
                or transaction.consumed != transaction.expected):
            raise RuntimeError("verl inbound transaction is incomplete")
        digest = transaction.hasher.digest()
        flat = torch.frombuffer(transaction.buffer, dtype=torch.float32)
        stored = self.sink.keep(VerlStoredFragment(transaction.fragment_id, transaction.version,
                                                   digest.hex(), flat))
        transaction.finished = True
        transaction.buffer = None
        if stored.flat is not flat:  # a replay of a fragment already kept
            return StreamedInboundPayload(stored, digest)

        def discard() -> None:
            self.sink.forget(stored)
            stored.release()

        return StreamedInboundPayload(stored, digest, discard)

    def abort_message(self, transaction: _Transaction, staged_tokens) -> None:
        del staged_tokens  # plain bytes: nothing to drop
        transaction.buffer = None

    def release_all(self) -> None:
        self.sink.release_all()


# ----------------------------------------------------------------------------- the trainer group
@dataclass(frozen=True)
class VerlRoundEvidence:
    """What the stream binds a local round to (the verl counterpart of Miles'
    persisted trajectory evidence): the round, the policy that sampled it and
    the exact trained batch."""

    rollout_id: int
    behavior_policy_hash: str
    input_batch_hash: str
    trained_tokens: int
    trajectory_ids: tuple[str, ...]


class VerlFullParameterGroup:
    """The trainer group of one verl island, as the role stream sees it.

    ``island`` is the ports' ``VerlIsland`` (``call_workers`` reaches every actor
    worker); ``receipts`` maps a rollout id to the :class:`LocalStepReceipt` the
    trainer port returned for it and ``evidence`` to its :class:`VerlRoundEvidence`."""

    def __init__(self, island: Any) -> None:
        self.island = island
        self.receipts: dict[int, LocalStepReceipt] = {}
        self.evidence: dict[int, VerlRoundEvidence] = {}

    def record_round(self, receipt: LocalStepReceipt, evidence: VerlRoundEvidence) -> None:
        self.receipts[evidence.rollout_id] = receipt
        self.evidence[evidence.rollout_id] = evidence

    def round_evidence(self, rollout_id: int) -> VerlRoundEvidence:
        evidence = self.evidence.get(rollout_id)
        if evidence is None:
            raise RuntimeError(f"verl island has no trained round {rollout_id}")
        return evidence

    async def manifests(self) -> list[dict]:
        return await asyncio.to_thread(self.island.call_workers, fp.remote_manifest)

    async def export_parameters(self, names: Sequence[str]) -> dict[str, torch.Tensor]:
        """Whole FP32 CPU tensors of ``names`` (collective on the workers; rank 0 reads)."""

        results = await asyncio.to_thread(self.island.call_workers, fp.remote_export_for_reader, list(names))
        readers = [result for result in results if "tensors" in result]
        if len(readers) != 1:
            raise RuntimeError(f"verl export returned {len(readers)} reader results, expected 1")
        return readers[0]["tensors"]

    async def apply_parameters(self, tensors: Mapping[str, torch.Tensor]) -> list[dict]:
        """Write whole tensors into every worker's shards (collective)."""

        return await asyncio.to_thread(self.island.call_workers_shared, fp.remote_apply_fragment,
                                       dict(tensors), False)

    async def receipt(self, rollout_id: int) -> LocalStepReceipt:
        receipt = self.receipts.get(rollout_id)
        if receipt is None:
            raise RuntimeError(f"verl island has no receipt for round {rollout_id}")
        return receipt


# ----------------------------------------------------------------------------- the adapter
class VerlChunkedFullParameterAdapter:
    """The chunked full-parameter cut protocol over one verl island."""

    def __init__(self, layout: ParameterLayout, *, max_chunk_bytes: int) -> None:
        self.layout = layout
        self.max_chunk_bytes = int(max_chunk_bytes)
        self.expected_parameter_tensor_count = len(layout.specs)
        self.expected_parameter_scalar_count = sum(spec.numel for spec in layout.specs)
        self._names = tuple(fp.fragment_tensor_names(layout, fragment_id)
                            for fragment_id in range(layout.fragments.num_fragments))
        # The cut the trainer holds: the last capture or apply.  The stream
        # captures right after every local round and builds every apply target
        # from its current cut, so between two of these calls the trainer's
        # weights do not move; an apply then writes only what differs.
        self.resident: VerlFullParameterCut | None = None

    # -- construction ----------------------------------------------------
    @classmethod
    async def initialize(cls, group: VerlFullParameterGroup, *, config: RoleStreamConfig
                         ) -> tuple[VerlChunkedFullParameterAdapter, VerlFullParameterCut]:
        shapes = fp.agree_on_shapes(await group.manifests())
        layout = fp.plan_layout(fp.island_specs(shapes), component=config.component,
                                minimum_fragments=config.expected_fragments,
                                max_fragment_bytes=config.max_fragment_bytes,
                                matrix_merge=config.matrix_merge)
        adapter = cls(layout, max_chunk_bytes=config.max_chunk_bytes)
        if (layout.fragments.num_fragments != config.expected_fragments
                or layout.layout_hash != config.expected_layout_hash):
            # The refusal doubles as a layout probe: the observed values to freeze.
            print(LAYOUT_PROBE_LOG_PREFIX + json.dumps(adapter.probe_record(), sort_keys=True), flush=True)
            raise RuntimeError(
                "verl full-parameter layout differs from the frozen probe: "
                f"expected_fragments={config.expected_fragments}, "
                f"actual_fragments={layout.fragments.num_fragments}, "
                f"expected_layout_hash={config.expected_layout_hash}, "
                f"actual_layout_hash={layout.layout_hash}")
        cut = await adapter.capture(group, policy_version=0, local_step_generation=0)
        return adapter, cut

    def probe_record(self) -> dict[str, Any]:
        return {"parameter_layout_sha256": self.layout.layout_hash,
                "expected_fragments": self.layout.fragments.num_fragments,
                "parameter_tensor_count": self.expected_parameter_tensor_count,
                "parameter_scalar_count": self.expected_parameter_scalar_count,
                "fragment_bytes": [fragment.numel * 4 for fragment in self.layout.fragments.fragments]}

    def fragment_numel(self, fragment_id: int) -> int:
        if not 0 <= fragment_id < self.layout.fragments.num_fragments:
            raise ValueError(f"fragment {fragment_id} is outside the layout")
        return self.layout.fragments.fragments[fragment_id].numel

    # -- cuts --------------------------------------------------------------
    def flatten(self, fragment_id: int, values: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """One fragment's flat FP32 payload from whole parameter tensors."""

        flat = torch.empty(self.fragment_numel(fragment_id), dtype=torch.float32)
        offset = 0
        for unit in self.layout.fragment_slices(fragment_id):
            whole = values[unit.spec.name]
            if tuple(whole.shape) != unit.spec.shape:
                raise ValueError(f"{unit.spec.name!r} has shape {tuple(whole.shape)}, layout {unit.spec.shape}")
            start, stop = unit.flat_range
            flat[offset:offset + unit.numel].copy_(whole.reshape(-1)[start:stop])
            offset += unit.numel
        if not torch.isfinite(flat).all().item():
            raise ValueError(f"fragment {fragment_id} has NaN/Inf")
        return flat

    def unflatten(self, cut: VerlFullParameterCut, names: Sequence[str]) -> dict[str, torch.Tensor]:
        """Whole tensors of ``names`` from a complete cut (split parameters are joined)."""

        wanted = set(names)
        specs = {spec.name: spec for spec in self.layout.specs}
        out = {name: torch.empty(specs[name].shape, dtype=torch.float32) for name in wanted}
        for fragment_id in range(self.layout.fragments.num_fragments):
            offset = 0
            for unit in self.layout.fragment_slices(fragment_id):
                if unit.spec.name in wanted:
                    start, stop = unit.flat_range
                    out[unit.spec.name].reshape(-1)[start:stop].copy_(
                        cut.fragments[fragment_id][offset:offset + unit.numel])
                offset += unit.numel
        return out

    async def capture(self, group: VerlFullParameterGroup, *, policy_version: int,
                      local_step_generation: int, defer_payload: bool = False,
                      lazy: bool = False) -> VerlFullParameterCut:
        if defer_payload or lazy:
            raise ValueError("verl cuts are always captured whole (no deferred or lazy capture)")
        fragments = []
        for fragment_id, names in enumerate(self._names):
            values = await group.export_parameters(names)
            fragments.append(self.flatten(fragment_id, values))
            del values
        cut = VerlFullParameterCut.create(self.layout.layout_hash, policy_version=policy_version,
                                          local_step_generation=local_step_generation, fragments=fragments)
        self.resident = cut
        return cut

    def _check_cut(self, cut: VerlFullParameterCut) -> None:
        if cut.layout_hash != self.layout.layout_hash or len(cut.fragments) != self.layout.fragments.num_fragments:
            raise ValueError("verl cut layout changed")

    def _chunks(self, flat: torch.Tensor) -> Iterator[memoryview]:
        raw = _bytes(flat)
        for start in range(0, raw.nbytes, self.max_chunk_bytes):
            yield raw[start:start + self.max_chunk_bytes]

    def fragment_parts(self, cut: VerlFullParameterCut, fragment_id: int, *,
                       ray_module=None) -> Callable[[], Iterator[memoryview]]:
        del ray_module
        self._check_cut(cut)
        flat = cut.fragments[fragment_id]
        return lambda: self._chunks(flat)

    def delta_parts_from_authoritative(self, stored: VerlStoredFragment, local: VerlFullParameterCut,
                                       local_fragment_id: int, *, ray_module=None,
                                       local_descriptor: Any = None) -> Callable[[], Iterator[memoryview]]:
        """``local - authoritative`` of one fragment (the push of a local horizon)."""

        del ray_module
        if local_descriptor is not None:
            raise ValueError("verl cuts carry no lazy descriptors")
        self._check_cut(local)
        if local.local_step_generation < 1:
            raise ValueError("verl local cut has no synchronized local horizon")
        if stored.fragment_id != local_fragment_id or stored.version < 0:
            raise ValueError("authoritative fragment provenance changed")
        anchor, values = stored.values(), local.fragments[local_fragment_id]
        if anchor.numel() != values.numel():
            raise ValueError("authoritative fragment geometry changed")
        step = max(1, self.max_chunk_bytes // 4)

        def parts() -> Iterator[memoryview]:
            for start in range(0, values.numel(), step):
                yield _bytes(values[start:start + step] - anchor[start:start + step])

        return parts

    def deferred_fragment_ids(self, cut: VerlFullParameterCut) -> frozenset[int]:
        del cut
        return frozenset()

    async def materialize(self, group, cut, fragment_ids, *, describe: bool = False) -> dict:
        del group, cut, fragment_ids, describe
        return {}

    # -- local rounds ------------------------------------------------------
    async def record_local_round(self, group: VerlFullParameterGroup, *, anchor: VerlFullParameterCut,
                                 rollout_id: int, learner_id: int, learner_generation: int,
                                 trajectories: VerlRoundEvidence, role: str,
                                 expected_local_step_generation: int, expected_optimizer_steps: int,
                                 check_model: bool = True) -> LocalStepReceipt:
        del check_model
        if role != "actor":
            raise ValueError("verl trains the actor role only")
        if (anchor.layout_hash != self.layout.layout_hash
                or anchor.local_step_generation != expected_local_step_generation - 1
                or trajectories.rollout_id != rollout_id
                or trajectories.behavior_policy_hash != anchor.policy_hash
                or not trajectories.trajectory_ids):
            raise ValueError("verl local-round evidence does not bind the anchor")
        receipt = await group.receipt(rollout_id)
        if (not receipt.optimizer_step_succeeded
                or receipt.optimizer_steps != expected_optimizer_steps
                or receipt.input_batch_hash != trajectories.input_batch_hash
                or receipt.base_policy_hash != anchor.policy_hash):
            raise RuntimeError(f"verl local round {rollout_id} receipt does not match its evidence")
        return LocalStepReceipt(
            algorithm=self.layout.algorithm, learner_id=learner_id, learner_generation=learner_generation,
            base_policy_version=anchor.policy_version, base_policy_hash=anchor.policy_hash,
            input_batch_hash=trajectories.input_batch_hash, trajectory_ids=trajectories.trajectory_ids,
            trained_tokens=trajectories.trained_tokens, optimizer_steps=expected_optimizer_steps,
            optimizer_step_succeeded=True, parameter_layout_hash=self.layout.layout_hash)

    # -- authoritative fragments -------------------------------------------
    def fragment_sink(self, reference: VerlFullParameterCut, *, ray_module=None) -> VerlFragmentSink:
        del ray_module
        self._check_cut(reference)
        return VerlFragmentSink(self)

    def verify_initial_fragments(self, reference: VerlFullParameterCut,
                                 fragments: Sequence[VerlStoredFragment], *,
                                 allow_newer: bool = False) -> None:
        """The syncer's version-0 bytes must equal this island's own cut
        (``allow_newer``: a rejoin's newer fragments only need the right size)."""

        observed = {}
        for stored in fragments:
            fragment_id = stored.fragment_id
            if (not 0 <= fragment_id < self.layout.fragments.num_fragments or fragment_id in observed
                    or (stored.version != 0 and not allow_newer)
                    or stored.values().numel() != self.fragment_numel(fragment_id)
                    or (stored.version == 0
                        and stored.wire_payload_hash != reference.fragment_hashes[fragment_id])):
                raise RuntimeError(f"verl version-zero fragment {fragment_id} differs from the checkpoint")
            observed[fragment_id] = stored
        if set(observed) != set(range(self.layout.fragments.num_fragments)):
            raise RuntimeError("verl version-zero cut is incomplete")

    def assemble_mixed_target(self, current: VerlFullParameterCut, *, target_policy_version: int,
                              authoritative_fragments: Sequence[VerlStoredFragment]) -> VerlFullParameterCut:
        """``current`` with some fragments replaced by authoritative values (no copies)."""

        self._check_cut(current)
        fragments, hashes = list(current.fragments), list(current.fragment_hashes)
        replaced: set[int] = set()
        for stored in authoritative_fragments:
            fragment_id = stored.fragment_id
            if fragment_id in replaced or not 0 <= fragment_id < len(fragments):
                raise ValueError("authoritative fragment subset is malformed")
            values = stored.values()
            if values.numel() != fragments[fragment_id].numel():
                raise ValueError("authoritative fragment geometry changed")
            fragments[fragment_id], hashes[fragment_id] = values, stored.wire_payload_hash
            replaced.add(fragment_id)
        return VerlFullParameterCut.create(self.layout.layout_hash, policy_version=target_policy_version,
                                           local_step_generation=0, fragments=fragments,
                                           fragment_hashes=hashes)

    async def apply(self, group: VerlFullParameterGroup, target: VerlFullParameterCut, *,
                    commit_token: str | None = None) -> int:
        """Make the trainer hold ``target``: write the fragments that differ from
        :attr:`resident` (every fragment when none is known), export them again
        and compare their hashes with the target's.  Returns the parameters written."""

        del commit_token
        self._check_cut(target)
        resident = self.resident
        changed = [fragment_id for fragment_id in range(self.layout.fragments.num_fragments)
                   if resident is None
                   or resident.fragment_hashes[fragment_id] != target.fragment_hashes[fragment_id]]
        self.resident = None  # unknown until the apply is proven
        names = sorted({name for fragment_id in changed for name in self._names[fragment_id]})
        if names:
            await group.apply_parameters(self.unflatten(target, names))
        for fragment_id in changed:
            values = await group.export_parameters(self._names[fragment_id])
            if fragment_hash(self.flatten(fragment_id, values)) != target.fragment_hashes[fragment_id]:
                raise RuntimeError(f"verl apply of fragment {fragment_id} did not land (hash differs)")
        self.resident = target
        return len(names)

    def release(self, cut: VerlFullParameterCut) -> int:
        """Cuts are plain tensors in this process; dropping the last reference frees them."""

        del cut
        return 0

    # -- not offered on verl (refused by :func:`refuse_unsupported`) --------
    def rebase_fragment(self, *_args, **_kwargs):
        raise ValueError("verl islands do not offer sync/train overlap")

    def relabel_local_progress(self, *_args, **_kwargs):
        raise ValueError("verl islands do not offer sync/train overlap")


# ----------------------------------------------------------------------------- the stream
class VerlFullParameterRoleStream(FullParameterRoleStream):
    """The engine-neutral role stream over one verl FSDP2 island."""

    stored_fragment_type = VerlStoredFragment

    def __init__(self, config: RoleStreamConfig, *, backend_identity_sha256: str) -> None:
        refuse_unsupported(config)
        super().__init__(config)
        self.backend_identity_sha256 = backend_identity_sha256

    async def _initialize_adapter(self, group) -> tuple[VerlChunkedFullParameterAdapter, VerlFullParameterCut]:
        return await VerlChunkedFullParameterAdapter.initialize(group, config=self.config)

    def _inbound_chunk_sink(self, sink) -> VerlInboundChunkSink:
        return VerlInboundChunkSink(sink)

    def _session_contract_hash(self, layout_hash: str) -> bytes:
        return session_contract_hash(layout_hash, self.config,
                                     backend_identity_sha256=self.backend_identity_sha256)


# ----------------------------------------------------------------------------- island wiring
PLAN_KEY = "full_parameter"
SCHEMA_TRAINING_CONTRACT = "yeto-verl-full-training-contract-v1"
# VerlRunConfig fields that change what a round trains: part of the training
# contract every island of one session must share.  Left out on purpose: the
# island-local paths and names, the seed (each island samples its own data),
# the card count (the layout is shape-free) and the vLLM memory share.
CONTRACT_RUN_FIELDS = (
    "parameter_mode", "lora_rank", "lora_alpha", "target_modules", "lr", "groups_per_round",
    "samples_per_group", "max_prompt_len", "max_response_len", "temperature", "top_p", "top_k",
    "allow_sampling_shift", "mode", "full_determinism", "enforce_eager", "bypass_mode",
    "criteria_enabled", "correction", "tis_lower", "tis_upper", "engine_commit", "max_model_len",
    "model_dtype", "reward_name", "chat_template_kwargs",
)


# ``full_parameter_identity`` (re-exported here) and the identity hash are the
# Miles island's rules too: yeto/rl/engine/full_parameter_identity.py.
_canonical_sha256 = identity_sha256


def training_contract_sha256(run: Mapping[str, Any], *, model: str, model_revision: str,
                             model_config_sha256: str, reward_sha256: str, global_rounds: int) -> str:
    """The training semantics every island of one full-parameter session shares."""

    return _canonical_sha256({
        "schema": SCHEMA_TRAINING_CONTRACT,
        "model": model, "model_revision": model_revision.lower(),
        "model_config_sha256": model_config_sha256, "reward_sha256": reward_sha256,
        "global_rounds": int(global_rounds),
        "run": {name: run[name] for name in CONTRACT_RUN_FIELDS},
    })


def syncer_addresses(value: str) -> list[tuple[str, int]]:
    """``HOST:PORT[,HOST:PORT...]``: one syncer, or every shard in shard order."""

    out = []
    for item in value.split(","):
        host, _, port = item.strip().rpartition(":")
        if not host or not port.isdigit():
            raise ValueError(f"syncer address must be HOST:PORT, got {item!r}")
        out.append((host, int(port)))
    return out


def role_stream_config(section: Mapping[str, Any], *, syncer: str, learner_id: int,
                       model_revision: str) -> RoleStreamConfig:
    """The actor's :class:`RoleStreamConfig` from the island plan's full-parameter section."""

    from yeto.rl.local_learner import ComponentIdentity

    async_rounds = bool(section["async_rounds"])
    addresses = syncer_addresses(syncer)
    generations = [0] * int(section["num_learners"])
    generations[learner_id] = int(section.get("learner_generation", 0))
    pool = section.get("island_pool") or {}
    return RoleStreamConfig(
        algorithm="grpo", role="actor",
        component=ComponentIdentity(role="actor", model_revision=model_revision.lower(),
                                    config_hash=section["model_config_sha256"]),
        syncer_addr=addresses[0], syncer_shards=tuple(addresses) if len(addresses) > 1 else (),
        learner_id=learner_id, learner_generation=generations[learner_id],
        learner_generations=tuple(generations),
        total_fragment_steps=int(section["total_fragment_steps"]),
        expected_fragments=int(section["fragments"]),
        expected_layout_hash=section["parameter_layout_sha256"],
        local_horizon=int(section["local_horizon"]) if async_rounds else 1,
        optimizer_steps_per_round=1,
        training_contract_hash=section["training_contract_sha256"],
        syncer_profile_hash=section["syncer_profile_sha256"],
        pipeline_depth=int(section["pipeline"]) if async_rounds else 1,
        algorithm_spec_sha256=section["algorithm_spec_sha256"],
        wait_timeout=float(section["wait_timeout_s"]),
        wan_streams=int(section["wan_streams"]),
        wan_stream_schedule=section.get("wan_stream_schedule", "rr"),
        wan_stall_timeout=float(section.get("wan_stall_timeout", 0.0)),
        push_dtype=section["push_dtype"],
        push_int8_block=int(section.get("push_int8_block", 128)),
        push_int4_block=int(section.get("push_int4_block", 128)),
        push_int4_stochastic=bool(section.get("push_int4_stochastic", False)),
        push_int4_error_feedback=section.get("push_int4_error_feedback", "off"),
        push_tensor_blocks=bool(section.get("push_tensor_blocks", True)),
        max_fragment_bytes=int(section["max_fragment_bytes"]),
        final_cut=section["final_cut"],
        partial_quorum=async_rounds,
        local_init=bool(section["local_init"]),
        matrix_merge=section["matrix_merge"],
        bcast_delta=bool(section.get("bcast_delta", False)),
        **({"island_lease_s": float(pool["lease_s"]), "syncer_epoch": int(pool["syncer_epoch"])} if pool else {}),
    )


# One private event loop for the island's stream calls (the engine-neutral
# runner the Miles ports use). It is never made the thread's current loop, so
# verl code that reads that loop is unaffected.
_LoopRunner = LoopRunner


def build_full_parameter_sync(island: Any, plan: Mapping[str, Any], *, backend_identity_sha256: str):
    """The island's outer sync: the engine-neutral ``FullParameterSync`` over one
    verl actor stream, its policy state and the stream (for the plan record)."""

    from yeto.rl.engine.full_parameter_sync import (FullParameterPolicyState, FullParameterSync,
                                                    FullParameterSyncProfile, RolePairing)

    section = plan[PLAN_KEY]
    config = role_stream_config(section, syncer=plan["syncer"], learner_id=int(plan["learner_id"]),
                                model_revision=plan["model_revision"])
    stream = VerlFullParameterRoleStream(config, backend_identity_sha256=backend_identity_sha256)
    group = island.full_group
    profile = FullParameterSyncProfile.from_configs({"actor": config}, sweep=not section["async_rounds"],
                                                    max_local_rounds_ahead=section.get("max_local_rounds_ahead"))
    sync = FullParameterSync(RolePairing({"actor": stream}, profile), groups={"actor": group},
                             run=_LoopRunner().run, trajectories=group.round_evidence)
    return sync, FullParameterPolicyState(sync), stream


class VerlFullParameterLocalState:
    """``PolicyState`` of a full-parameter island without an outer sync
    (``--rl-single-island-no-sync``): every export is a whole cut of the
    trainer, returned as a trainer-resident handle (no tensors leave)."""

    def __init__(self, group: VerlFullParameterGroup, *, component, max_fragment_bytes: int,
                 max_chunk_bytes: int = 256 << 20, matrix_merge: str = "avg") -> None:
        self.group = group
        self.component = component
        self.max_fragment_bytes = int(max_fragment_bytes)
        self.max_chunk_bytes = int(max_chunk_bytes)
        self.matrix_merge = matrix_merge
        self.adapter: VerlChunkedFullParameterAdapter | None = None
        self.runner = _LoopRunner()
        self.policy_version = 0

    @property
    def layout_hash(self) -> str:
        return self._adapter().layout.layout_hash

    def _adapter(self) -> VerlChunkedFullParameterAdapter:
        if self.adapter is None:
            shapes = fp.agree_on_shapes(self.runner.run(self.group.manifests()))
            layout = fp.plan_layout(fp.island_specs(shapes), component=self.component, minimum_fragments=1,
                                    max_fragment_bytes=self.max_fragment_bytes, matrix_merge=self.matrix_merge)
            self.adapter = VerlChunkedFullParameterAdapter(layout, max_chunk_bytes=self.max_chunk_bytes)
        return self.adapter

    def export(self, *, policy_version: int | None = None):
        from yeto.rl.engine.policy_digest import PolicyDigest, TrainerResidentState
        from yeto.rl.engine.trainable_state import LAYOUT_FRAGMENTS, UnsupportedLayoutError

        version = self.policy_version if policy_version is None else int(policy_version)
        adapter = self._adapter()
        cut = self.runner.run(adapter.capture(self.group, policy_version=version, local_step_generation=0))

        def never(_version: int):
            raise UnsupportedLayoutError(LAYOUT_FRAGMENTS)

        return TrainerResidentState(
            layout=LAYOUT_FRAGMENTS, base_model_revision=self.component.model_revision,
            config_hash=self.component.config_hash, layout_hash=cut.layout_hash, policy_version=version,
            digest=PolicyDigest(cut.policy_hash, cut.content_hash, adapter.expected_parameter_scalar_count * 4, ()),
            materialize_fn=never)

    def apply(self, state, *, optimizer: str, local_step: int) -> None:
        raise RuntimeError("a full-parameter verl island without an outer sync never applies a cut "
                           "(round-cut resume of full-parameter islands is not wired, V9)")


def add_island_flags(parser) -> None:
    """The full-parameter flags of the verl island entry (Miles island spellings)."""

    parser.add_argument("--parameter-mode", choices=("lora", "full"), default="lora")
    parser.add_argument("--num-learners", type=int, default=1)
    parser.add_argument("--learner-generation", type=int, default=0)
    parser.add_argument("--fragments", type=int, default=1)
    parser.add_argument("--parameter-layout-sha256", default=None)
    parser.add_argument("--syncer-profile-sha256", default=None)
    parser.add_argument("--full-parameter-rounds", choices=("sweep", "async"), default="sweep")
    parser.add_argument("--init-mode", choices=("auto", "local", "upload"), default="auto")
    parser.add_argument("--sweep-push-dtype", choices=("f32", "bf16", "int8", "int4", "int4-log"),
                        default="f32")
    parser.add_argument("--sweep-fragment-max-bytes", type=int, default=DEFAULT_MAX_FRAGMENT_BYTES)
    parser.add_argument("--sweep-wait-timeout-s", type=float, default=900.0)
    parser.add_argument("--sweep-final-cut", choices=("download", "ack"), default="download")
    parser.add_argument("--pipeline", type=int, default=1)
    parser.add_argument("--local-horizon", type=int, default=1)
    parser.add_argument("--total-fragment-steps", type=int, default=None)
    parser.add_argument("--rl-sync-matrix-merge", choices=("avg", "rda"), default="avg")
    # S27 G-DN: the rest of the launcher's full-parameter island flags
    # (launch.rl_island_task._rl_trainable_island_flags), so none is ignored:
    # engine-neutral push and transport options go to the stream config; the
    # options the verl stream does not offer are parsed only to be refused.
    parser.add_argument("--sweep-push-int8-block", type=int, default=128)
    parser.add_argument("--sweep-push-int4-block", type=int, default=128)
    parser.add_argument("--sweep-push-int4-stochastic", action="store_true")
    parser.add_argument("--sweep-push-int4-error-feedback", choices=("off", "bf16", "f32"), default="off")
    parser.add_argument("--sweep-push-no-tensor-blocks", action="store_true")
    parser.add_argument("--sweep-max-local-rounds-ahead", type=int, default=None)
    parser.add_argument("--wan-stream-schedule", default="rr")
    parser.add_argument("--wan-stall-timeout", type=float, default=0.0)
    parser.add_argument("--sweep-overlap", action="store_true")
    parser.add_argument("--sweep-expert-token-weights", action="store_true")
    parser.add_argument("--rl-peer-recovery-store", default=None)
    # S27 G-DS: the island's HF exports and data position (Miles island spellings;
    # see full_parameter_extras). Added only when the island parser has none yet:
    # the verl Megatron island (S27 G-DC) parses the export flags itself.
    for flag in ("--rl-export-hf-rounds", "--rl-export-hf-store", "--rl-data-position-store"):
        if flag not in parser._option_string_actions:
            parser.add_argument(flag, default=None)


def full_parameter_extras(args: Any, *, sync: str) -> dict[str, Any]:
    """S27 G-DS: the plan keys of a full-parameter island's HF exports and data
    position, both absent when off (the plans of other runs are unchanged).

    * ``hf_exports`` (``--rl-export-hf-rounds`` with ``--rl-export-hf-store``,
      ``yeto.rl.engine.hf_export``): ``{"rounds", "store"}`` with the island's
      own export store, as on a Miles island (``miles.island_entry.hf_export_config``);
      :mod:`.driver_extras` gives the driver the :mod:`.hf_export` writer;
    * ``data_position_store`` (``--rl-data-position-store``,
      ``yeto.rl.engine.data_position``): the synced island's data position log,
      so a relaunched island continues at its next row.

    Refused before any work: an export on an island that does not generate and
    train, a data position store on an island without an outer sync."""
    from yeto.rl.engine import sample_roles
    from yeto.rl.engine.hf_export import EXPORT_HF_FLAG, ExportRounds, export_store_path

    out: dict[str, Any] = {}
    rounds = ExportRounds.parse(getattr(args, "rl_export_hf_rounds", None))
    if bool(rounds) != bool(getattr(args, "rl_export_hf_store", None)):
        raise ValueError(f"{EXPORT_HF_FLAG} and --rl-export-hf-store go together")
    if rounds:
        if sample_roles.sample_role(args) != sample_roles.BOTH:
            raise ValueError(f"{EXPORT_HF_FLAG} needs an island that generates and trains (sample role both)")
        out["hf_exports"] = {"rounds": str(rounds),
                             "store": export_store_path(str(args.rl_export_hf_store),
                                                        learner_id=int(args.learner_id))}
    store = getattr(args, "rl_data_position_store", None)
    if store:
        if sync == "none":
            raise ValueError("--rl-data-position-store belongs to a synced full-parameter island")
        out["data_position_store"] = str(store)
    return out


_sha256_hex = is_sha256  # yeto/rl/engine/full_parameter_identity.py


def plan_section(args: Any, run: Mapping[str, Any], *, model_path: str, sync: str,
                 algorithm_spec_sha256: str) -> dict[str, Any]:
    """``plan["model_config_sha256"]`` and, for a synced island, ``plan["full_parameter"]``.

    Refuses, before any GPU work, the full-parameter options this island cannot
    run: a pool without asynchronous rounds, a fragment budget that is not whole
    sweeps, a missing layout or syncer profile hash."""

    from pathlib import Path

    from yeto.provenance import file_sha256

    from .reward_fn import function_for

    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        raise ValueError("full-parameter model has no config.json")
    model_config_sha256 = file_sha256(config_path)
    out: dict[str, Any] = {"model_config_sha256": model_config_sha256, **full_parameter_extras(args, sync=sync)}
    if sync == "none":
        return out
    async_rounds = args.full_parameter_rounds == "async"
    if sync == "pool" and not async_rounds:
        raise ValueError("an island pool (--rl-island-pool) needs --full-parameter-rounds async")
    if not async_rounds and (args.local_horizon != 1 or args.pipeline != 1):
        raise ValueError("the full-parameter strict sweep needs --local-horizon 1 and --pipeline 1")
    if args.global_rounds < 1 or args.fragments < 1 or args.total_fragment_steps != args.global_rounds * args.fragments:
        raise ValueError("full-parameter fragment budget: --total-fragment-steps must equal "
                         "--global-rounds x --fragments")
    if not _sha256_hex(args.parameter_layout_sha256) or not _sha256_hex(args.syncer_profile_sha256):
        raise ValueError("a synced full-parameter island needs --parameter-layout-sha256 and "
                         "--syncer-profile-sha256 (64 lowercase hex)")
    if not 0 <= args.learner_id < args.num_learners:
        raise ValueError("--learner-id must be below --num-learners")
    refused = [flag for flag, on in (
        ("--sweep-overlap", getattr(args, "sweep_overlap", False)),
        ("--sweep-expert-token-weights", getattr(args, "sweep_expert_token_weights", False)),
        ("--rl-peer-recovery-store", getattr(args, "rl_peer_recovery_store", None) is not None),
    ) if on]
    if refused:
        raise ValueError("verl 全参数岛不支持: " + ", ".join(refused))
    lead = getattr(args, "sweep_max_local_rounds_ahead", None)
    if lead is not None and not async_rounds:
        raise ValueError("--sweep-max-local-rounds-ahead needs --full-parameter-rounds async")
    reward_sha256 = _canonical_sha256({
        "reward_function": args.reward_function, "verl_reward_name": function_for(args.reward_function),
        "reward_fn_py": file_sha256(Path(__file__).with_name("reward_fn.py"))})
    pool = ({"lease_s": 150.0 if args.rl_sync_island_lease_s is None else float(args.rl_sync_island_lease_s),
             "syncer_epoch": int(args.rl_syncer_epoch or 0)} if sync == "pool" else None)
    out["full_parameter"] = {
        "async_rounds": async_rounds,
        "num_learners": int(args.num_learners),
        "learner_generation": int(args.learner_generation),
        "fragments": int(args.fragments),
        "total_fragment_steps": int(args.total_fragment_steps),
        "parameter_layout_sha256": args.parameter_layout_sha256,
        "syncer_profile_sha256": args.syncer_profile_sha256,
        "model_config_sha256": model_config_sha256,
        "training_contract_sha256": training_contract_sha256(
            run, model=args.model, model_revision=args.model_revision,
            model_config_sha256=model_config_sha256, reward_sha256=reward_sha256,
            global_rounds=args.global_rounds),
        "algorithm_spec_sha256": algorithm_spec_sha256,
        "local_horizon": int(args.local_horizon),
        "pipeline": int(args.pipeline),
        "wait_timeout_s": float(args.sweep_wait_timeout_s),
        "wan_streams": int(args.wan_streams),
        "push_dtype": args.sweep_push_dtype,
        "push_int8_block": int(getattr(args, "sweep_push_int8_block", 128)),
        "push_int4_block": int(getattr(args, "sweep_push_int4_block", 128)),
        "push_int4_stochastic": bool(getattr(args, "sweep_push_int4_stochastic", False)),
        "push_int4_error_feedback": getattr(args, "sweep_push_int4_error_feedback", "off"),
        "push_tensor_blocks": not getattr(args, "sweep_push_no_tensor_blocks", False),
        "max_local_rounds_ahead": lead,
        "wan_stream_schedule": getattr(args, "wan_stream_schedule", "rr"),
        "wan_stall_timeout": float(getattr(args, "wan_stall_timeout", 0.0) or 0.0),
        "max_fragment_bytes": int(args.sweep_fragment_max_bytes),
        "final_cut": args.sweep_final_cut,
        "local_init": local_init_rule(args.init_mode, async_rounds),
        "matrix_merge": args.rl_sync_matrix_merge,
        "island_pool": pool,
    }
    if getattr(args, "bcast_delta", False):  # S27 G-DS: absent when off (plans unchanged)
        out["full_parameter"]["bcast_delta"] = True
    return out
