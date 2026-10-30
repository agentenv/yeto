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

* the cut lives in the island driver's process, not in a Ray object store: a
  read gathers each parameter whole (``DTensor.full_tensor``) on the workers,
  rank 0 builds the fragment's flat payload, checks it finite and hashes it,
  and the payload reaches the driver as a numpy array (Ray stores numpy out of
  band; torch tensors would go through ``torch.save`` pickling).  A local
  round's capture is lazy (S27 G-EC, the engine's lazy capture): it reads
  nothing, and only the fragments a push needs are read, at materialize time,
  while the trainer still holds them; every
  ``YETO_FULL_PARAMETER_FULL_CHECK_EVERY``-th round and the version-0 cut read
  and hash every fragment;
* the layout carries no shard identity (``full_parameter`` module doc), so
  islands with different card counts share one session;
* an apply writes only the fragments it replaces, then hashes them again on
  the workers and compares with the target (the proof that the trainer holds
  the target); verl offers no optimizer-state receipts, so the stream's
  optimizer proof is off (``_optimizer_states`` returns None);
* not offered: sync/train overlap and MoE expert token weights (refused at
  configuration);
* peer recovery (S27 G-ED): the engine-neutral stream island
  (:mod:`yeto.rl.engine.stream_recovery`) over the FSDP2 rank files of
  :mod:`.cut` (:mod:`.peer_recovery`);
* bf16 broadcast deltas (``--bcast-delta``, S27 G-DS) need nothing here: the
  ``SyncerClient`` decodes them into the f32 fragment before the inbound sink
  sees a byte (``yeto.downlink`` module doc), so the sink stays f32-only.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import torch

from yeto.protocol import MSG_BCAST_FRAGMENT, MSG_FINAL_FRAGMENT, StreamedInboundPayload
from yeto.rl.contracts import LocalStepReceipt
from yeto.rl.engine.full_parameter_stream import (
    DEFER_CAPTURE_ENV,
    LAZY_CAPTURE_ENV,
    FullParameterRoleStream,
    RoleStreamConfig,
    capture_switch_on,
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
# The Miles island's switch of the per-operation time split lines
# (adapters/miles/full_parameter/rank_stats.py), shared so one flag works on both.
RANK_STATS_ENV = "YETO_FULL_PARAMETER_RANK_STATS"
_FEATURES_REPORTED: set[str] = set()


def _report_feature(name: str, enabled: bool, switch: str) -> bool:
    """Print once per process whether a capture option is on (the Miles line)."""

    key = f"{name}:{enabled}"
    if key not in _FEATURES_REPORTED:
        _FEATURES_REPORTED.add(key)
        print(f"[yeto-full-parameter] {name}: {'enabled' if enabled else 'disabled'} "
              f"(verl; {switch}={'1' if enabled else '0'})", file=sys.stderr, flush=True)
    return enabled
_CUT_DOMAIN = b"yeto-verl-full-parameter-cut-v1\0"
_CONTENT_DOMAIN = b"yeto-verl-full-parameter-content-v1\0"
_LAZY_DOMAIN = b"yeto-verl-full-parameter-lazy-fragment-v1\0"


def refuse_unsupported(config: RoleStreamConfig) -> None:
    """The stream options a verl island does not offer, refused before any work."""

    refused = [name for name, on in (
        ("sync/train overlap (--sweep-overlap)", config.overlap),
        ("MoE expert token weights", config.expert_token_weights),
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
    the cut is made (cuts and stored fragments share them).

    A lazy cut (S27 G-EC, :meth:`VerlChunkedFullParameterAdapter.capture`)
    reads nothing: every fragment's entry in ``fragments`` is ``None`` and its
    hash in ``fragment_hashes`` is a label, unique to this capture
    (``lazy_ids``).  :meth:`VerlChunkedFullParameterAdapter.materialize` reads
    a fragment from the trainer while the trainer still holds it, stores it
    in ``fragments`` and its content hash in ``contents``.  The cut identity
    (``policy_hash``, ``content_hash``) never changes."""

    policy_version: int
    local_step_generation: int
    layout_hash: str
    policy_hash: str
    content_hash: str
    fragments: list = field(repr=False, compare=False)
    fragment_hashes: tuple[str, ...]
    lazy_ids: frozenset[int] = frozenset()
    contents: dict = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def create(cls, layout_hash: str, *, policy_version: int, local_step_generation: int,
               fragments: Sequence[torch.Tensor | None],
               fragment_hashes: Sequence[str] | None = None,
               lazy_ids: frozenset[int] = frozenset(),
               contents: Mapping[int, str] | None = None) -> VerlFullParameterCut:
        hashes = tuple(fragment_hashes) if fragment_hashes is not None else tuple(
            fragment_hash(flat) for flat in fragments)
        policy, content = _cut_hashes(layout_hash, policy_version, local_step_generation, hashes)
        return cls(int(policy_version), int(local_step_generation), layout_hash, policy, content,
                   list(fragments), hashes, frozenset(lazy_ids), dict(contents or {}))

    def content_of(self, fragment_id: int) -> str | None:
        """The content hash of one fragment's bytes, or None while a lazy
        fragment is not materialized."""

        if fragment_id in self.lazy_ids:
            return self.contents.get(fragment_id)
        return self.fragment_hashes[fragment_id]


@dataclass(frozen=True)
class VerlFragmentContent:
    """The content descriptor of one materialized fragment of a lazy cut (the
    stream's ``local_descriptor``): a push of that fragment checks it against
    the bytes it reads."""

    fragment_id: int
    numel: int
    payload_hash: str


def _lazy_labels(layout_hash: str, fragments: int, *, policy_version: int,
                 local_step_generation: int, nonce: bytes) -> tuple[str, ...]:
    """Per-fragment labels of one lazy capture: SHA256 over their own domain,
    the cut's identity fields and a fresh ``nonce``, so a label names one
    capture and never equals a content hash (the rule of Miles' lazy export)."""

    labels = []
    for fragment_id in range(fragments):
        digest = hashlib.sha256(_LAZY_DOMAIN)
        digest.update(layout_hash.encode("ascii"))
        digest.update(int(policy_version).to_bytes(8, "little"))
        digest.update(int(local_step_generation).to_bytes(8, "little"))
        digest.update(int(fragment_id).to_bytes(8, "little"))
        digest.update(nonce)
        labels.append(digest.hexdigest())
    return tuple(labels)


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
def _owned(array) -> torch.Tensor:
    """A CPU FP32 tensor over ``array`` that this process owns: a read-only
    array (a zero-copy view of Ray's object store) is copied once, a
    writable one (an in-process call) is wrapped."""

    import numpy as np

    if isinstance(array, torch.Tensor):
        return array
    if not array.flags.writeable:
        array = np.array(array, copy=True)
    return torch.from_numpy(array)


@dataclass(frozen=True)
class VerlRoundEvidence:
    """What the stream binds a local round to (the verl counterpart of Miles'
    persisted trajectory evidence): the round, the policy that sampled it and
    the exact trained batch.

    ``generated_by`` (S27 G-EV, rollout ahead): set by :meth:`bound_to_round_policy`
    to the policy that generated a batch one version before the round's own."""

    rollout_id: int
    behavior_policy_hash: str
    input_batch_hash: str
    trained_tokens: int
    trajectory_ids: tuple[str, ...]
    generated_by: str | None = None

    def bound_to_round_policy(self, policy_hash: str) -> VerlRoundEvidence:
        """Rollout ahead (``full_parameter_sync._round_trajectories``, the verl form of
        ``trajectory_evidence.bind_round_policy``): the batch was generated by the
        previous publication and trains from ``policy_hash``, the round's own; the
        generating policy stays in ``generated_by``. The driver checked the batch's
        groups against the policy-age limit when it accepted them."""
        if self.generated_by is not None:
            raise ValueError(f"round {self.rollout_id} evidence is already bound to its round policy")
        return replace(self, behavior_policy_hash=policy_hash, generated_by=self.behavior_policy_hash)


class VerlFullParameterGroup:
    """The trainer group of one verl island, as the role stream sees it.

    ``island`` is the ports' ``VerlIsland`` (``call_workers`` reaches every actor
    worker); ``receipts`` maps a rollout id to the :class:`LocalStepReceipt` the
    trainer port returned for it and ``evidence`` to its :class:`VerlRoundEvidence`."""

    def __init__(self, island: Any) -> None:
        self.island = island
        self.receipts: dict[int, LocalStepReceipt] = {}
        self.evidence: dict[int, VerlRoundEvidence] = {}
        # Local rounds trained so far (one per record_round, which the trainer
        # port calls right after each optimizer step): a lazy cut is read from
        # the trainer only while this count has not moved since its capture.
        self.trained_rounds = 0

    def record_round(self, receipt: LocalStepReceipt, evidence: VerlRoundEvidence) -> None:
        self.receipts[evidence.rollout_id] = receipt
        self.evidence[evidence.rollout_id] = evidence
        self.trained_rounds += 1

    def round_evidence(self, rollout_id: int) -> VerlRoundEvidence:
        evidence = self.evidence.get(rollout_id)
        if evidence is None:
            raise RuntimeError(f"verl island has no trained round {rollout_id}")
        return evidence

    async def manifests(self) -> list[dict]:
        return await asyncio.to_thread(self.island.call_workers, fp.remote_manifest)

    async def export_parameters(self, names: Sequence[str]) -> dict[str, torch.Tensor]:
        """Whole FP32 CPU tensors of ``names`` (collective on the workers; rank 0 reads)."""

        results = await asyncio.to_thread(self.island.call_workers, fp.remote_export_arrays_for_reader,
                                          list(names))
        readers = [result for result in results if "arrays" in result]
        if len(readers) != 1:
            raise RuntimeError(f"verl export returned {len(readers)} reader results, expected 1")
        return {name: _owned(array) for name, array in readers[0]["arrays"].items()}

    async def release_memory(self, host: bool = True) -> list[dict]:
        """Free the workers' cached device blocks (:func:`.full_parameter.remote_release_memory`);
        the island calls it after an export, whose gathers leave large blocks cached.
        ``host``: also trim the host heap when ``YETO_HOST_MEMORY=trim`` (S27 G-EZ3)."""

        return await asyncio.to_thread(self.island.call_workers, fp.remote_release_memory, host)

    async def export_fragment(self, units: Sequence[tuple[str, int, int]], numel: int, *,
                              payload: bool, sha256: bool) -> dict[str, Any]:
        """One fragment read on the workers (:func:`.full_parameter.remote_export_flat`):
        ``{"finite", "flat" (with payload: a CPU FP32 tensor this process owns),
        "sha256" (with sha256), "timings"}``; ``timings`` adds this call's
        wall seconds (``call_s``) and the copy out of Ray's store (``own_s``)."""

        started = time.perf_counter()
        results = await asyncio.to_thread(self._export_on_workers, [tuple(unit) for unit in units], int(numel),
                                          bool(payload), bool(sha256))
        call_s = time.perf_counter() - started
        readers = [result for result in results if "finite" in result]
        if len(readers) != 1:
            raise RuntimeError(f"verl fragment export returned {len(readers)} reader results, expected 1")
        reader = dict(readers[0])
        timings = {f"worker_{key}": value for key, value in dict(reader.get("timings") or {}).items()}
        timings["call_s"] = call_s
        if payload:
            mark = time.perf_counter()
            reader["flat"] = _owned(reader["flat"])
            timings["own_s"] = time.perf_counter() - mark
            if reader["flat"].numel() != int(numel):
                raise RuntimeError("verl fragment export returned the wrong element count")
        reader["timings"] = timings
        return reader

    async def apply_parameters(self, tensors: Mapping[str, torch.Tensor]) -> list[dict]:
        """Write whole tensors into every worker's shards (collective; sent as
        numpy arrays, which Ray stores without torch pickling)."""

        arrays = {name: tensor.contiguous().numpy() for name, tensor in tensors.items()}
        return await asyncio.to_thread(self._apply_on_workers, arrays)

    # The worker calls of a fragment read and an apply (S27 G-FG: the Megatron
    # engine island's group, .megatron_full_parameter, replaces these two).
    def _export_on_workers(self, units: list, numel: int, payload: bool, sha256: bool) -> list[dict]:
        return self.island.call_workers(fp.remote_export_flat, units, numel, payload, sha256)

    def _apply_on_workers(self, arrays: Mapping[str, Any]) -> list[dict]:
        return self.island.call_workers_shared(fp.remote_apply_fragment, arrays, False)

    async def receipt(self, rollout_id: int) -> LocalStepReceipt:
        receipt = self.receipts.get(rollout_id)
        if receipt is None:
            raise RuntimeError(f"verl island has no receipt for round {rollout_id}")
        return receipt


# ----------------------------------------------------------------------------- the adapter
class _Stats:
    """The time split of one adapter operation: summed seconds per step
    (driver and worker side), fragments read and their bytes."""

    def __init__(self, operation: str, **flags: Any) -> None:
        self.operation = operation
        self.started = time.perf_counter()
        self.values: dict[str, Any] = {"fragments": 0, "bytes": 0, **flags}

    def add(self, timings: Mapping[str, float], *, fragments: int = 0, nbytes: int = 0,
            written: int = 0) -> None:
        for key, value in timings.items():
            self.values[key] = self.values.get(key, 0.0) + float(value)
        self.values["fragments"] += fragments
        self.values["bytes"] += nbytes
        if written:
            self.values["parameters_written"] = self.values.get("parameters_written", 0) + written

    def done(self) -> dict[str, Any]:
        self.values["seconds"] = time.perf_counter() - self.started
        return {key: round(value, 4) if isinstance(value, float) else value for key, value in self.values.items()}


class VerlChunkedFullParameterAdapter:
    """The chunked full-parameter cut protocol over one verl island."""

    def __init__(self, layout: ParameterLayout, *, max_chunk_bytes: int) -> None:
        self.layout = layout
        self.max_chunk_bytes = int(max_chunk_bytes)
        self.expected_parameter_tensor_count = len(layout.specs)
        self.expected_parameter_scalar_count = sum(spec.numel for spec in layout.specs)
        self._names = tuple(fp.fragment_tensor_names(layout, fragment_id)
                            for fragment_id in range(layout.fragments.num_fragments))
        self._units = tuple(fp.fragment_units(layout, fragment_id)
                            for fragment_id in range(layout.fragments.num_fragments))
        # The cut the trainer holds: the last capture or apply.  The stream
        # captures right after every local round and builds every apply target
        # from its current cut, so between two of these calls the trainer's
        # weights do not move; an apply then writes only what differs.
        self.resident: VerlFullParameterCut | None = None
        # The group's trained-round count when ``resident`` was established: a
        # lazy fragment is read from the trainer only while it is unchanged.
        self.resident_rounds: int | None = None
        # S27 G-EC: the time split of the last export, materialize and apply
        # (the stream prints it as a rank-stats line).
        self.last_stats: dict[str, dict[str, Any]] = {}

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
                    if cut.fragments[fragment_id] is None:
                        raise ValueError(f"{unit.spec.name!r} spans fragment {fragment_id}, which the cut never read")
                    start, stop = unit.flat_range
                    out[unit.spec.name].reshape(-1)[start:stop].copy_(
                        cut.fragments[fragment_id][offset:offset + unit.numel])
                offset += unit.numel
        return out

    def _sharing_fragments(self, fragment_ids: Sequence[int]) -> set[int]:
        """Other fragments holding a slice of a parameter that ``fragment_ids``
        hold (a split parameter is written whole)."""

        names = {name for fragment_id in fragment_ids for name in self._names[fragment_id]}
        return {other for other in range(self.layout.fragments.num_fragments)
                if other not in fragment_ids and names.intersection(self._names[other])}

    async def fragment_digests(self, group: VerlFullParameterGroup) -> tuple[str, ...]:
        """The content hash of every fragment, hashed on the reading rank: no
        payload crosses Ray and the driver keeps nothing (S27 G-EZ2). The hashes
        are the ones :meth:`capture` gets from the same worker call."""

        stats = _Stats("digest", lazy=False)
        hashes = []
        for fragment_id in range(self.layout.fragments.num_fragments):
            numel = self.fragment_numel(fragment_id)
            result = await group.export_fragment(self._units[fragment_id], numel, payload=False, sha256=True)
            stats.add(result["timings"], fragments=1, nbytes=numel * 4)
            if not result["finite"]:
                raise ValueError(f"fragment {fragment_id} has NaN/Inf")
            hashes.append(result["sha256"])
        self.last_stats["digest"] = stats.done()
        return tuple(hashes)

    async def capture(self, group: VerlFullParameterGroup, *, policy_version: int,
                      local_step_generation: int, defer_payload: bool = False,
                      lazy: bool = False) -> VerlFullParameterCut:
        """Cut the trainer's current parameters.

        ``lazy`` (with ``defer_payload``; S27 G-EC, the engine's lazy capture):
        read nothing; every fragment carries a fresh label instead of a content
        hash, and :meth:`materialize` reads the fragments a push needs.
        Otherwise read and hash every fragment on the workers.  A deferred
        capture that is not lazy is a whole capture here: hashing a fragment
        needs its bytes, and reading them is the cost a deferral would save."""

        if lazy and not defer_payload:
            raise ValueError("a lazy capture must defer its payload")
        fragments = self.layout.fragments.num_fragments
        stats = _Stats("export", lazy=bool(lazy))
        if lazy:
            labels = _lazy_labels(self.layout.layout_hash, fragments, policy_version=policy_version,
                                  local_step_generation=local_step_generation, nonce=os.urandom(32))
            cut = VerlFullParameterCut.create(
                self.layout.layout_hash, policy_version=policy_version,
                local_step_generation=local_step_generation, fragments=[None] * fragments,
                fragment_hashes=labels, lazy_ids=frozenset(range(fragments)))
        else:
            flats, hashes = [], []
            for fragment_id in range(fragments):
                flat, digest = await self._read(group, fragment_id, stats)
                flats.append(flat)
                hashes.append(digest)
            cut = VerlFullParameterCut.create(self.layout.layout_hash, policy_version=policy_version,
                                              local_step_generation=local_step_generation, fragments=flats,
                                              fragment_hashes=hashes)
        self.resident = cut
        self.resident_rounds = getattr(group, "trained_rounds", None)
        self.last_stats["export"] = stats.done()
        return cut

    async def _read(self, group: VerlFullParameterGroup, fragment_id: int,
                    stats: _Stats) -> tuple[torch.Tensor, str]:
        """``(flat FP32 payload, content hash)`` of one fragment, read on the workers."""

        numel = self.fragment_numel(fragment_id)
        result = await group.export_fragment(self._units[fragment_id], numel, payload=True, sha256=True)
        stats.add(result["timings"], fragments=1, nbytes=numel * 4)
        if not result["finite"]:
            raise ValueError(f"fragment {fragment_id} has NaN/Inf")
        return result["flat"], result["sha256"]

    def _require_trainer_holds(self, group: VerlFullParameterGroup, cut: VerlFullParameterCut,
                               fragment_id: int, *, resident: VerlFullParameterCut | None,
                               resident_rounds: int | None) -> None:
        """Refuse to read ``cut``'s fragment from the trainer unless the trainer
        still holds it: no local round trained since the cut and no apply wrote
        that fragment (the resident cut still names it)."""

        if (resident is None
                or resident_rounds != getattr(group, "trained_rounds", None)
                or resident.fragment_hashes[fragment_id] != cut.fragment_hashes[fragment_id]):
            raise RuntimeError(f"verl trainer no longer holds fragment {fragment_id} of the cut "
                               "(a local round or an apply moved it since the capture)")

    def deferred_fragment_ids(self, cut: VerlFullParameterCut) -> frozenset[int]:
        """Fragments of ``cut`` not read yet (a lazy cut's unmaterialized rows)."""

        return frozenset(fragment_id for fragment_id, flat in enumerate(cut.fragments) if flat is None)

    async def materialize(self, group: VerlFullParameterGroup, cut: VerlFullParameterCut,
                          fragment_ids: Sequence[int], *, describe: bool = False) -> dict[int, Any]:
        """Read the unread fragments ``fragment_ids`` of ``cut`` from the trainer
        (refused unless the trainer still holds them), hash them, check them
        finite and keep them in ``cut``.  Returns
        ``{fragment_id: VerlFragmentContent}`` for every fragment asked for
        (also those already read), which the stream passes back as a push's
        ``local_descriptor``; ``describe`` does not change it."""

        del describe
        self._check_cut(cut)
        stats = _Stats("materialize")
        out: dict[int, VerlFragmentContent] = {}
        for fragment_id in sorted(set(int(value) for value in fragment_ids)):
            if cut.fragments[fragment_id] is None:
                self._require_trainer_holds(group, cut, fragment_id, resident=self.resident,
                                            resident_rounds=self.resident_rounds)
                flat, digest = await self._read(group, fragment_id, stats)
                cut.fragments[fragment_id] = flat
                cut.contents[fragment_id] = digest
            out[fragment_id] = VerlFragmentContent(fragment_id, self.fragment_numel(fragment_id),
                                                   cut.content_of(fragment_id))
        self.last_stats["materialize"] = stats.done()
        return out

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
        if flat is None:
            raise ValueError(f"verl cut fragment {fragment_id} was never read (lazy capture)")
        return lambda: self._chunks(flat)

    def delta_parts_from_authoritative(self, stored: VerlStoredFragment, local: VerlFullParameterCut,
                                       local_fragment_id: int, *, ray_module=None,
                                       local_descriptor: Any = None) -> Callable[[], Iterator[memoryview]]:
        """``local - authoritative`` of one fragment (the push of a local horizon)."""

        del ray_module
        self._check_cut(local)
        if local.local_step_generation < 1:
            raise ValueError("verl local cut has no synchronized local horizon")
        if stored.fragment_id != local_fragment_id or stored.version < 0:
            raise ValueError("authoritative fragment provenance changed")
        anchor, values = stored.values(), local.fragments[local_fragment_id]
        if values is None:
            raise ValueError(f"verl local fragment {local_fragment_id} was not materialized (lazy capture)")
        if local_descriptor is not None:
            if (not isinstance(local_descriptor, VerlFragmentContent)
                    or local_descriptor.fragment_id != local_fragment_id
                    or local_descriptor.payload_hash != local.content_of(local_fragment_id)):
                raise ValueError(f"verl local fragment {local_fragment_id} differs from its content descriptor")
        elif local_fragment_id in local.lazy_ids:
            raise ValueError(f"a push of lazy fragment {local_fragment_id} needs its content descriptor")
        if anchor.numel() != values.numel():
            raise ValueError("authoritative fragment geometry changed")
        step = max(1, self.max_chunk_bytes // 4)

        def parts() -> Iterator[memoryview]:
            for start in range(0, values.numel(), step):
                yield _bytes(values[start:start + step] - anchor[start:start + step])

        return parts

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
        """``current`` with some fragments replaced by authoritative values (no
        copies); the fragments it keeps keep their lazy labels."""

        self._check_cut(current)
        fragments, hashes = list(current.fragments), list(current.fragment_hashes)
        replaced: set[int] = set()
        for stored in authoritative_fragments:
            fragment_id = stored.fragment_id
            if fragment_id in replaced or not 0 <= fragment_id < len(fragments):
                raise ValueError("authoritative fragment subset is malformed")
            values = stored.values()
            if values.numel() != self.fragment_numel(fragment_id):
                raise ValueError("authoritative fragment geometry changed")
            fragments[fragment_id], hashes[fragment_id] = values, stored.wire_payload_hash
            replaced.add(fragment_id)
        return VerlFullParameterCut.create(
            self.layout.layout_hash, policy_version=target_policy_version, local_step_generation=0,
            fragments=fragments, fragment_hashes=hashes, lazy_ids=current.lazy_ids - replaced,
            contents={key: value for key, value in current.contents.items() if key not in replaced})

    async def apply(self, group: VerlFullParameterGroup, target: VerlFullParameterCut, *,
                    commit_token: str | None = None) -> int:
        """Make the trainer hold ``target``: write the fragments that differ from
        :attr:`resident` (every fragment when none is known), hash them again on
        the workers and compare with the target's hashes.  Returns the
        parameters written.

        A split parameter is written whole, so its slices in fragments the
        target keeps come from the target; a lazy target's unread slice is read
        from the trainer first, which still holds it (the resident cut names
        it and no round trained since)."""

        del commit_token
        self._check_cut(target)
        resident, resident_rounds = self.resident, self.resident_rounds
        changed = [fragment_id for fragment_id in range(self.layout.fragments.num_fragments)
                   if resident is None
                   or resident.fragment_hashes[fragment_id] != target.fragment_hashes[fragment_id]]
        stats = _Stats("apply")
        for fragment_id in sorted(self._sharing_fragments(changed)):
            if target.fragments[fragment_id] is None:
                self._require_trainer_holds(group, target, fragment_id, resident=resident,
                                            resident_rounds=resident_rounds)
                target.fragments[fragment_id], _digest = await self._read(group, fragment_id, stats)
        self.resident = None  # unknown until the apply is proven
        names = sorted({name for fragment_id in changed for name in self._names[fragment_id]})
        if names:
            mark = time.perf_counter()
            tensors = self.unflatten(target, names)
            stats.add({"unflatten_s": time.perf_counter() - mark})
            mark = time.perf_counter()
            await group.apply_parameters(tensors)
            stats.add({"write_s": time.perf_counter() - mark}, written=len(names))
            del tensors
        for fragment_id in changed:
            numel = self.fragment_numel(fragment_id)
            result = await group.export_fragment(self._units[fragment_id], numel, payload=False, sha256=True)
            stats.add({f"verify_{key}": value for key, value in result["timings"].items()},
                      fragments=1, nbytes=numel * 4)
            if not result["finite"] or result["sha256"] != target.fragment_hashes[fragment_id]:
                raise RuntimeError(f"verl apply of fragment {fragment_id} did not land (hash differs)")
        self.resident = target
        self.resident_rounds = getattr(group, "trained_rounds", None)
        self.last_stats["apply"] = stats.done()
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

    # S27 G-EC: the engine's deferred and lazy capture.  The verl adapter reads
    # every fragment from the trainer on demand (no store to defer into), so
    # both follow only their switches.
    def _supports_deferred_capture(self) -> bool:
        return _report_feature("deferred capture", capture_switch_on(DEFER_CAPTURE_ENV), DEFER_CAPTURE_ENV)

    def _supports_lazy_capture(self) -> bool:
        return _report_feature("lazy capture", capture_switch_on(LAZY_CAPTURE_ENV), LAZY_CAPTURE_ENV)

    async def _report_rank_stats(self, phase: str) -> None:
        """One ``rank stats`` line (the Miles format) with the time split of the
        adapter's last ``phase``; ``YETO_FULL_PARAMETER_RANK_STATS=0`` (the Miles
        switch) turns it off."""

        if os.environ.get(RANK_STATS_ENV, "1") == "0" or self.adapter is None:
            return
        stats = self.adapter.last_stats.pop(phase, None)
        if stats and (stats.get("fragments") or stats.get("lazy")):
            print(f"[yeto-full-parameter] rank stats op={phase} role={self.config.role} rank=0 "
                  f"{json.dumps({phase: stats}, sort_keys=True)}", file=sys.stderr, flush=True)

    def _recovery_island(self, group, adapter, *, recovering: bool):
        """S27 G-ED: peer recovery over the FSDP2 rank files (:mod:`.peer_recovery`)."""
        from yeto.rl.engine.stream_recovery import recovery_island

        from .peer_recovery import VerlRankShards

        # VerlRankShards does not capture: the donor writes its rank files at the safe point
        return recovery_island(self, VerlRankShards(group, adapter), asyncio.get_running_loop(),
                               recovering=recovering)


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

    from yeto.rl.engine.stream_recovery import PeerRecoveryOptions
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
        peer_recovery=PeerRecoveryOptions.from_plan(section.get("peer_recovery")),
        **({"island_lease_s": float(pool["lease_s"]), "syncer_epoch": int(pool["syncer_epoch"])} if pool
           # S27 G-ED: a fixed roster with peer recovery names the syncer epoch in its HELLO
           else {"syncer_epoch": int(section["syncer_epoch"])} if "syncer_epoch" in section else {}),
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


def full_parameter_policy(island: Any, plan: Mapping[str, Any], *, backend_identity_sha256: str,
                          group: VerlFullParameterGroup | None = None):
    """``(sync, policy state)`` of a full-parameter verl island (S27 B10), for every
    trainer that holds the FSDP2 actor workers behind ``island.call_workers``: the
    synchronous trainer (``trainer._full_parameter_ports``) and the fully_async and
    buffer trainers (:mod:`.fully_async_full`, S27 G-EW).

    Sets ``island.full_group`` (the trainer port records each round on it). With an
    outer sync (``plan["full_parameter"]``): the engine-neutral ``FullParameterSync``
    over the verl role stream. Without one (``plan["sync"] == "none"``): the
    local-only sync over :class:`VerlFullParameterLocalState`, whose layout hash
    becomes ``island.layout_hash``.

    ``group`` (S27 G-FG): the trainer group to read and write through; the
    Megatron engine island passes its own (``.megatron_full_parameter``), the
    FSDP2 islands keep the default :class:`VerlFullParameterGroup`."""
    from yeto.rl.engine.bridges import LocalOnlySync
    from yeto.rl.local_learner import ComponentIdentity

    island.full_group = group if group is not None else VerlFullParameterGroup(island)
    if plan["sync"] == "none":
        component = ComponentIdentity(role="actor", model_revision=plan["model_revision"].lower(),
                                      config_hash=plan["model_config_sha256"])
        policy_state = VerlFullParameterLocalState(island.full_group, component=component,
                                                   max_fragment_bytes=DEFAULT_MAX_FRAGMENT_BYTES)
        island.layout_hash = policy_state.layout_hash
        return LocalOnlySync(int(plan["global_rounds"])), policy_state
    if not plan.get(PLAN_KEY):
        raise ValueError("a synced full-parameter verl island needs the full-parameter sync plan")
    sync, policy_state, _stream = build_full_parameter_sync(
        island, plan, backend_identity_sha256=backend_identity_sha256)
    return sync, policy_state


def _never_materialize(_version: int):
    from yeto.rl.engine.trainable_state import LAYOUT_FRAGMENTS, UnsupportedLayoutError

    raise UnsupportedLayoutError(LAYOUT_FRAGMENTS)


LIGHT_POLICY_PREFIX = b"yeto-verl-light-policy-v1\0"


class VerlFullParameterLocalState:
    """``PolicyState`` of a full-parameter island without an outer sync
    (``--rl-single-island-no-sync``): every export hashes the whole trainer on
    its reading rank and returns a trainer-resident handle (no tensors leave).

    S27 G-EZ2: the export used to be a whole :meth:`~VerlChunkedFullParameterAdapter.capture`,
    which brings every fragment's FP32 payload into the island driver only to
    hash it, and keeps it there as the adapter's resident cut. For the 120B
    text model that is 449 GiB in the driver process on the head node, on top of
    that node's share of the training state (npu1 n3: the driver's RSS reached
    374 GiB before the first step). The digest is the same content hash, taken
    from the same worker call without the payload."""

    def __init__(self, group: VerlFullParameterGroup, *, component, max_fragment_bytes: int,
                 max_chunk_bytes: int = 256 << 20, matrix_merge: str = "avg",
                 light_digest: bool | None = None) -> None:
        from . import local_digest

        self.group = group
        self.component = component
        self.max_fragment_bytes = int(max_fragment_bytes)
        self.max_chunk_bytes = int(max_chunk_bytes)
        self.matrix_merge = matrix_merge
        self.adapter: VerlChunkedFullParameterAdapter | None = None
        self.runner = _LoopRunner()
        self.policy_version = 0
        # S28 G-LHASH: a group that can hash its shards in place (the Megatron group)
        # gets the light digest on the rounds the driver does not fully verify;
        # YETO_RL_LIGHT_DIGEST=0 (or light_digest=False) keeps the full check every round.
        wanted = local_digest.enabled() if light_digest is None else bool(light_digest)
        self.light = wanted and callable(getattr(group, "light_digest", None))
        if self.light:
            self.supports_unverified_rounds = True
            self.full_digest_at_boundaries = True

    @property
    def layout_hash(self) -> str:
        return self._adapter().layout.layout_hash

    def _adapter(self) -> VerlChunkedFullParameterAdapter:
        if self.adapter is None:
            manifests = self.runner.run(self.group.manifests())
            # S27 G-EQ3: a persistent buffer is not sharded, so a rank whose copy did
            # not come from the checkpoint computes wrong log-probs with perfectly
            # fine parameters; this raises before any training happens.
            digests = fp.agree_on_buffers(manifests)
            emit = getattr(getattr(self.group, "island", None), "emit", None)
            if callable(emit) and digests:
                emit("rl_verl_buffer_digests", buffers=len(digests),
                     values={name: digests[name] for name in sorted(digests)[:64]})
            shapes = fp.agree_on_shapes(manifests)
            layout = fp.plan_layout(fp.island_specs(shapes), component=self.component, minimum_fragments=1,
                                    max_fragment_bytes=self.max_fragment_bytes, matrix_merge=self.matrix_merge)
            self.adapter = VerlChunkedFullParameterAdapter(layout, max_chunk_bytes=self.max_chunk_bytes)
        return self.adapter

    # S27 G-EU (VERL-PARITY #26): --rl-verify-every N turns this on (allow_unverified_rounds);
    # the rounds the driver does not verify publish export_identity (no whole cut, no hash)
    supports_unverified_rounds = False

    def export(self, *, policy_version: int | None = None):
        from yeto.rl.engine.policy_digest import PolicyDigest, TrainerResidentState
        from yeto.rl.engine.trainable_state import LAYOUT_FRAGMENTS

        version = self.policy_version if policy_version is None else int(policy_version)
        adapter = self._adapter()
        # S27 G-EZ2 (s27-verl-pool-profile): hash every fragment on its reading rank
        # instead of capturing the whole cut into the island driver, which for the 120B
        # text model is 449 GiB of FP32 payload in the driver process. Same content hash.
        hashes = self.runner.run(adapter.fragment_digests(self.group))
        layout_hash = adapter.layout.layout_hash
        policy_hash, content_hash = _cut_hashes(layout_hash, version, 0, hashes)
        state = TrainerResidentState(
            layout=LAYOUT_FRAGMENTS, base_model_revision=self.component.model_revision,
            config_hash=self.component.config_hash, layout_hash=layout_hash, policy_version=version,
            digest=PolicyDigest(policy_hash, content_hash, adapter.expected_parameter_scalar_count * 4, ()),
            materialize_fn=_never_materialize, weights_mark=self._mark(), holds_fn=self._holds)
        self._last_verified = state
        return state

    def _mark(self) -> dict:
        """The trainer weights version (``TrainerResidentState.weights_mark``): this
        state object and the group's trained rounds (``VerlFullParameterGroup.record_round``,
        called by the trainer port after every optimizer step)."""
        if not hasattr(self, "_process_id"):
            import uuid

            self._process_id = uuid.uuid4().hex
        return {"process_id": self._process_id, "version": int(getattr(self.group, "trained_rounds", 0) or 0)}

    def _holds(self, state) -> None:
        from yeto.rl.engine.policy_digest import WeightsChanged

        if dict(state.weights_mark or {}) != self._mark():
            raise WeightsChanged(f"the trainer trained since policy v{state.policy_version} was exported "
                                 f"({state.weights_mark} -> {self._mark()})")

    def export_digest(self, *, policy_version: int, deferred: bool = False):
        """The driver's resident export (rl-publish-fastpath): the same whole cut as
        ``export().with_version`` (the driver's fallback), so the policy hash is
        unchanged; ``deferred`` is never asked (no ``supports_deferred_digest``)."""
        del deferred
        state = self.export().with_version(int(policy_version))
        self._last_verified = state
        return state

    def allow_unverified_rounds(self) -> None:
        """S27 G-EU (``--rl-verify-every N``, as Miles' S27 C5): publications the driver
        does not verify use :meth:`export_identity` (no whole cut, no SHA-256)."""
        self.supports_unverified_rounds = True

    def export_identity(self, *, policy_version: int):
        """The resident state of a round whose publication is not verified: the
        trainer weights version stands in for the content hash
        (``policy_digest.unverified_identity``). None before the first verified cut
        (the driver then verifies that round)."""
        from yeto.rl.engine.policy_digest import PolicyDigest, TrainerResidentState, unverified_identity
        from yeto.rl.engine.trainable_state import LAYOUT_FRAGMENTS

        last = getattr(self, "_last_verified", None)
        if last is None:
            return None
        mark = self._mark()
        # S29 --rl-measure-no-verify: no release reads the light digest either, and it
        # reads every shard (~45 s a round on the 120B island)
        if self.light and not getattr(self, "measure_no_verify", False):
            return self._export_light(int(policy_version), last, mark)
        identity = unverified_identity(int(policy_version), mark)
        print(f"[yeto-verl] full-parameter publication v{policy_version}: unverified (no whole cut; last "
              f"verified hash {last.policy_tensor_hash()[:12]})", flush=True)
        return TrainerResidentState(
            layout=LAYOUT_FRAGMENTS, base_model_revision=self.component.model_revision,
            config_hash=self.component.config_hash, layout_hash=last.layout_hash,
            policy_version=int(policy_version),
            digest=PolicyDigest(identity, identity, last.digest.payload_bytes, ()),
            materialize_fn=_never_materialize, weights_mark=mark, holds_fn=self._holds, verified=False)

    def _export_light(self, policy_version: int, last, mark: dict):
        """S28 G-LHASH: the resident state of a round that is not fully verified, with
        the light digest (:mod:`.local_digest`) as its content hash.  ``verified`` stays
        False: the light digest depends on the TP/PP/EP split and is not the
        HF-canonical hash, so nothing compares it with the full check's."""
        from yeto.rl.engine.policy_digest import PolicyDigest, TrainerResidentState
        from yeto.rl.engine.trainable_state import LAYOUT_FRAGMENTS

        result = self.runner.run(self.group.light_digest())
        content = str(result["digest"])
        raw = json.dumps({"policy_version": policy_version, "light": content, "layout": last.layout_hash},
                         sort_keys=True).encode()
        policy_hash = hashlib.sha256(LIGHT_POLICY_PREFIX + raw).hexdigest()
        self.last_light = result
        print(f"[yeto-verl] full-parameter publication v{policy_version}: light digest "
              f"{result['algorithm']} {content[:12]} ({result['rows']} shard(s), "
              f"{result['bytes'] / 2**30:.1f} GiB, {result['seconds']:.1f} s slowest rank)", flush=True)
        return TrainerResidentState(
            layout=LAYOUT_FRAGMENTS, base_model_revision=self.component.model_revision,
            config_hash=self.component.config_hash, layout_hash=last.layout_hash,
            policy_version=policy_version,
            digest=PolicyDigest(policy_hash, content, last.digest.payload_bytes, ()),
            materialize_fn=_never_materialize, weights_mark=mark, holds_fn=self._holds, verified=False)

    def apply(self, state, *, optimizer: str, local_step: int) -> None:
        raise RuntimeError("a full-parameter verl island without an outer sync never applies a cut "
                           "(a round-cut resume restores the trainer's rank files instead)")


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
    # S27 G-ED: peer recovery (launch.peer_recovery.island_flags; Miles island spellings)
    parser.add_argument("--rl-peer-recovery-ttl-s", type=float, default=None)
    parser.add_argument("--rl-peer-recover", action="store_true")
    parser.add_argument("--rl-peer-recovery-staging", default=None)
    # S27 G-EL (neutral since the s27-next2 merge, engine.stream_recovery): the
    # MANIFEST re-read wait; the donor write mode (no capture on verl: inline either way)
    parser.add_argument("--rl-peer-recovery-manifest-wait-s", type=float, default=None)
    parser.add_argument("--rl-peer-recovery-donor-write", choices=("background", "inline"), default=None)
    # S27 G-DS: the island's HF exports and data position (Miles island spellings;
    # see full_parameter_extras). Added only when the island parser has none yet:
    # the verl Megatron island (S27 G-DC) parses the export flags itself.
    for flag in ("--rl-export-hf-rounds", "--rl-export-hf-store", "--rl-data-position-store"):
        if flag not in parser._option_string_actions:
            parser.add_argument(flag, default=None)
    # S27 G-EC: the Miles training and observation options verl can express.
    from .train_options import add_island_flags as add_train_option_flags

    add_train_option_flags(parser)


def full_parameter_extras(args: Any, *, sync: str) -> dict[str, Any]:
    """S27 G-DS: the plan keys of a full-parameter island's HF exports and data
    position, both absent when off (the plans of other runs are unchanged);
    S27 G-EC: also the observation keys of :func:`.train_options.observation_plan`.

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
    from .train_options import observation_plan  # S27 G-EC: absent when off

    out.update(observation_plan(args))
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
    recovery = _peer_recovery(args, sync=sync)
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
    if recovery is not None:  # S27 G-ED: absent when off (plans unchanged)
        if not async_rounds or getattr(args, "bcast_delta", False):
            raise ValueError("--rl-peer-recovery-store needs --full-parameter-rounds async without "
                             "--bcast-delta (the syncer resends whole fragments)")
        out["full_parameter"]["peer_recovery"] = recovery.to_plan()
        if pool is None:
            out["full_parameter"]["syncer_epoch"] = int(getattr(args, "rl_syncer_epoch", 0) or 0)
    return out


def _peer_recovery(args: Any, *, sync: str):
    """S27 G-ED: the island's peer recovery options (``--rl-peer-recovery-*``,
    and the launcher's relaunch marker), or None; refused on an island
    without an outer sync."""
    from yeto.rl.engine.stream_recovery import peer_recovery_options

    recovery = peer_recovery_options(args)
    if recovery is not None and sync == "none":
        raise ValueError("--rl-peer-recovery-store belongs to a synced full-parameter island")
    return recovery
