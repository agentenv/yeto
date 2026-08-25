"""Generic atomic policy-sweep state machine for the dense syncer profile."""

from __future__ import annotations

import hashlib
import re
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Protocol, TypeAlias

from ..fragments import FragmentLayout
from ..protocol import (
    DTYPE_F32,
    MSG_BCAST_FRAGMENT,
    MSG_FINAL_FRAGMENT,
    FinalManifest,
    PullRequest,
    StreamedInboundPayload,
    SyncerClient,
)

BytePart = bytes | bytearray | memoryview
PartsFactory = Callable[[], Iterable[BytePart]]
StoredPayload: TypeAlias = object
PayloadSink = Callable[[int, int, BytePart], StoredPayload]
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class _ReceivedPayload:
    value: StoredPayload
    payload_hash: bytes
    stored: bool
    discard: Callable[[], None] | None = None


@dataclass
class _MilesInboundTransaction:
    message_id: int
    total_bytes: int
    msg_type: int | None = None
    fragment_id: int | None = None
    version: int | None = None
    expected_payload_bytes: int | None = None
    shard: object | None = None
    reference_descriptor: object | None = None
    previous: object | None = None
    wire_hasher: object | None = None
    fragment_hasher: object | None = None
    consumed_bytes: int = 0
    chunk_index: int = 0
    current_tensor: object | None = None
    current_view: memoryview | None = None
    current_written: int = 0
    descriptors: list[object] = field(default_factory=list)
    refs: list[object | None] = field(default_factory=list)
    finished: bool = False


class _MilesRayInboundChunkSink:
    """Stream exact wire bytes into the production Miles typed Ray layout."""

    def __init__(self, sink: object) -> None:
        from miles.backends.megatron_utils.full_parameter_state import (
            _resolve_ray_module,
        )

        from .miles_chunked_full_parameter import AuthoritativeFragmentSink

        if not isinstance(sink, AuthoritativeFragmentSink):
            raise TypeError("Miles transactional sink has the wrong type")
        self.sink = sink
        self.ray = _resolve_ray_module(sink.ray_module)
        self._lock = threading.Lock()

    @staticmethod
    def _fragment_row(reference_cut: object, fragment_id: int) -> tuple[object, object]:
        matches = [
            (shard, descriptor)
            for shard in reference_cut.shards
            for descriptor in shard.fragments
            if descriptor.fragment_id == fragment_id
        ]
        if len(matches) != 1:
            raise ValueError("streamed authoritative fragment has no unique owner")
        return matches[0]

    def begin_message(
        self, message_id: int, total_bytes: int
    ) -> _MilesInboundTransaction:
        return _MilesInboundTransaction(message_id, total_bytes)

    def bind_fragment(
        self,
        transaction: _MilesInboundTransaction,
        msg_type: int,
        fragment_id: int,
        version: int,
        payload_bytes: int,
    ) -> None:
        from miles.backends.megatron_utils.full_parameter_state import (
            _fragment_hasher,
        )

        if transaction.msg_type is not None:
            raise RuntimeError("Miles inbound transaction was bound twice")
        if msg_type not in (MSG_BCAST_FRAGMENT, MSG_FINAL_FRAGMENT):
            raise ValueError("Miles inbound transaction has an invalid message type")
        shard, descriptor = self._fragment_row(
            self.sink.reference.transport_cut,
            fragment_id,
        )
        if payload_bytes != descriptor.numel * 4:
            raise ValueError("Miles inbound fragment byte count changed")
        transaction.msg_type = msg_type
        transaction.fragment_id = fragment_id
        transaction.version = version
        transaction.expected_payload_bytes = payload_bytes
        transaction.shard = shard
        transaction.reference_descriptor = descriptor
        transaction.wire_hasher = hashlib.sha256()
        transaction.fragment_hasher = _fragment_hasher(
            fragment_id,
            shard.parameter_layout_hash,
            shard.plan_hash,
        )
        with self._lock:
            key = (fragment_id, version)
            previous = self.sink._stored.get(key)
            if previous is not None and any(
                reference is None for reference in previous.refs
            ):
                self.sink._stored.pop(key)
                previous = None
            transaction.previous = previous

    def stage_chunk(
        self,
        transaction: _MilesInboundTransaction,
        payload: BytePart,
    ) -> object:
        del transaction
        import torch

        raw = memoryview(payload).cast("B")
        try:
            value = torch.empty(raw.nbytes, dtype=torch.uint8, device="cpu")
            target = memoryview(value.numpy()).cast("B")
            try:
                target[:] = raw
            finally:
                target.release()
        finally:
            raw.release()
        reference = self.ray.put(value)
        if reference is None:
            raise RuntimeError("Ray returned no staged inbound chunk reference")
        return reference

    def consume_staged_chunk(
        self,
        transaction: _MilesInboundTransaction,
        offset: int,
        token: object,
        nbytes: int,
    ) -> None:
        import torch

        try:
            value = self.ray.get(token)
            if (
                not isinstance(value, torch.Tensor)
                or value.device.type != "cpu"
                or value.dtype != torch.uint8
                or value.ndim != 1
                or not value.is_contiguous()
                or value.numel() != nbytes
            ):
                raise ValueError("staged inbound Ray chunk is malformed")
            raw = memoryview(value.numpy()).cast("B")
            try:
                self.consume_chunk(transaction, offset, raw)
            finally:
                raw.release()
        finally:
            self._drop_staged_reference(token)

    def consume_chunk(
        self,
        transaction: _MilesInboundTransaction,
        offset: int,
        payload: BytePart,
    ) -> None:
        if transaction.msg_type is None or transaction.expected_payload_bytes is None:
            raise RuntimeError("Miles inbound transaction is not bound")
        if offset != transaction.consumed_bytes:
            raise ValueError("Miles inbound chunks are not contiguous")
        raw = memoryview(payload).cast("B")
        try:
            if (
                transaction.consumed_bytes + raw.nbytes
                > transaction.expected_payload_bytes
            ):
                raise ValueError("Miles inbound fragment exceeds its payload bound")
            transaction.wire_hasher.update(raw)
            transaction.fragment_hasher.update(raw)
            if transaction.previous is None:
                self._copy_into_typed_chunks(transaction, raw)
            transaction.consumed_bytes += raw.nbytes
        finally:
            raw.release()

    def _copy_into_typed_chunks(
        self,
        transaction: _MilesInboundTransaction,
        raw: memoryview,
    ) -> None:
        import torch

        descriptor = transaction.reference_descriptor
        source_offset = 0
        while source_offset < raw.nbytes:
            if transaction.current_tensor is None:
                if transaction.chunk_index >= len(descriptor.chunks):
                    raise ValueError("Miles inbound fragment has extra typed chunks")
                expected = descriptor.chunks[transaction.chunk_index]
                transaction.current_tensor = torch.empty(
                    expected.numel,
                    dtype=torch.float32,
                    device="cpu",
                )
                transaction.current_view = memoryview(
                    transaction.current_tensor.numpy()
                ).cast("B")
                transaction.current_written = 0
            target = transaction.current_view
            take = min(
                raw.nbytes - source_offset,
                target.nbytes - transaction.current_written,
            )
            target[transaction.current_written : transaction.current_written + take] = (
                raw[source_offset : source_offset + take]
            )
            transaction.current_written += take
            source_offset += take
            if transaction.current_written == target.nbytes:
                self._finish_typed_chunk(transaction)

    def _finish_typed_chunk(self, transaction: _MilesInboundTransaction) -> None:
        import torch
        from miles.backends.megatron_utils.full_parameter_state import (
            FullParameterChunkDescriptor,
            _chunk_hash,
        )

        value = transaction.current_tensor
        view = transaction.current_view
        expected = transaction.reference_descriptor.chunks[transaction.chunk_index]
        if view is not None:
            view.release()
        transaction.current_view = None
        transaction.current_tensor = None
        transaction.current_written = 0
        if (
            not isinstance(value, torch.Tensor)
            or value.numel() != expected.numel
            or not torch.isfinite(value).all().item()
        ):
            raise ValueError("Miles inbound typed chunk is invalid")
        descriptor = FullParameterChunkDescriptor(
            expected.chunk_index,
            expected.flat_offset,
            expected.numel,
            _chunk_hash(
                transaction.fragment_id,
                expected.chunk_index,
                expected.flat_offset,
                value,
                parameter_layout_hash=transaction.shard.parameter_layout_hash,
                plan_hash=transaction.shard.plan_hash,
            ),
        )
        reference = self.ray.put(value)
        if reference is None:
            raise RuntimeError("Ray returned no typed inbound chunk reference")
        transaction.descriptors.append(descriptor)
        transaction.refs.append(reference)
        transaction.chunk_index += 1

    def finish_message(
        self,
        transaction: _MilesInboundTransaction,
    ) -> StreamedInboundPayload:
        from miles.backends.megatron_utils.full_parameter_state import (
            FullParameterFragmentDescriptor,
        )

        from .miles_chunked_full_parameter import StoredAuthoritativeFragment

        if (
            transaction.finished
            or transaction.expected_payload_bytes is None
            or transaction.consumed_bytes != transaction.expected_payload_bytes
            or transaction.current_tensor is not None
        ):
            raise RuntimeError("Miles inbound transaction is incomplete")
        digest = transaction.wire_hasher.digest()
        digest_hex = digest.hex()
        previous = transaction.previous
        if previous is not None:
            if previous.wire_payload_hash != digest_hex:
                raise RuntimeError("authoritative fragment replay changed bytes")
            transaction.finished = True
            return StreamedInboundPayload(previous, digest)
        if len(transaction.descriptors) != len(transaction.reference_descriptor.chunks):
            raise RuntimeError("Miles inbound typed chunk coverage is incomplete")
        fragment_descriptor = FullParameterFragmentDescriptor(
            transaction.fragment_id,
            transaction.reference_descriptor.numel,
            transaction.fragment_hasher.hexdigest(),
            tuple(transaction.descriptors),
        )
        stored = StoredAuthoritativeFragment(
            fragment_id=transaction.fragment_id,
            version=transaction.version,
            parameter_layout_hash=transaction.shard.parameter_layout_hash,
            topology=transaction.shard.topology,
            plan_hash=transaction.shard.plan_hash,
            descriptor=fragment_descriptor,
            refs=transaction.refs,
            wire_payload_hash=digest_hex,
        )
        key = (transaction.fragment_id, transaction.version)
        with self._lock:
            raced = self.sink._stored.get(key)
            if raced is not None:
                if raced.wire_payload_hash != digest_hex:
                    self._release_stored(stored)
                    raise RuntimeError("authoritative fragment replay changed bytes")
                self._release_stored(stored)
                transaction.finished = True
                return StreamedInboundPayload(raced, digest)
            self.sink._stored[key] = stored

        discarded = False

        def discard() -> None:
            nonlocal discarded
            with self._lock:
                if discarded:
                    return
                discarded = True
                if self.sink._stored.get(key) is stored:
                    self.sink._stored.pop(key)
            self._release_stored(stored)

        transaction.finished = True
        return StreamedInboundPayload(stored, digest, discard)

    def abort_message(
        self,
        transaction: _MilesInboundTransaction,
        staged_tokens: Iterable[object],
    ) -> None:
        for token in staged_tokens:
            self._drop_staged_reference(token)
        if transaction.finished:
            return
        if transaction.current_view is not None:
            transaction.current_view.release()
            transaction.current_view = None
        transaction.current_tensor = None
        transaction.current_written = 0
        for index, reference in enumerate(transaction.refs):
            if reference is not None:
                self._drop_staged_reference(reference)
                transaction.refs[index] = None
        transaction.descriptors.clear()

    def _drop_staged_reference(self, reference: object) -> None:
        drop = getattr(self.ray, "drop", None)
        if callable(drop):
            drop(reference)

    def _release_stored(self, stored: object) -> None:
        for reference in tuple(stored.refs):
            if reference is not None:
                self._drop_staged_reference(reference)
        stored.release()

    def release_all(self) -> None:
        """Release every object still owned by this inbound transaction set."""

        with self._lock:
            stored = tuple(self.sink._stored.values())
            self.sink._stored.clear()
        for fragment in stored:
            self._release_stored(fragment)


class DenseSweepClient(Protocol):
    finalizing: object

    def start(self) -> None: ...

    def close(self) -> None: ...

    def check_health(self) -> None: ...

    def send_init_parts(
        self,
        fragment_id: int,
        tensor_parts: Iterable[BytePart],
    ) -> bool: ...

    def push_fragment_parts(
        self,
        fragment_id: int,
        global_step: int,
        round_attempt: int,
        base_version: int,
        local_step: int,
        c_steps: int,
        c_tokens: int,
        tensor_parts: Iterable[BytePart],
        *,
        before_last_enqueue: Callable[[], None] | None = None,
    ) -> bool: ...

    def drain_pulls(self) -> list[PullRequest]: ...

    def drain_updates(self) -> list[object]: ...

    def wait_for_final_fragments(
        self,
        timeout: float | None = None,
    ) -> tuple[FinalManifest, list[object]]: ...

    def acknowledge_finalization(self, manifest: FinalManifest) -> None: ...


@dataclass(frozen=True)
class DenseSweepConfig:
    syncer_addr: tuple[str, int]
    learner_id: int
    learner_generation: int
    policy_rounds: int
    wan_streams: int = 4
    send_initial_params: bool = True
    wait_timeout: float = 900.0
    poll_seconds: float = 0.01
    max_fragment_bytes: int = 2 << 30

    def __post_init__(self) -> None:
        for name in ("learner_id", "learner_generation"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if (
            isinstance(self.policy_rounds, bool)
            or not isinstance(self.policy_rounds, int)
            or self.policy_rounds < 1
        ):
            raise ValueError("dense policy rounds must be positive")
        if (
            isinstance(self.wan_streams, bool)
            or not isinstance(self.wan_streams, int)
            or self.wan_streams < 1
        ):
            raise ValueError("dense WAN stream count must be positive")
        if self.wait_timeout <= 0 or not 0 < self.poll_seconds <= 1:
            raise ValueError("dense sweep wait timing is invalid")
        if (
            type(self.max_fragment_bytes) is not int
            or not 4 <= self.max_fragment_bytes <= 2 << 30
            or self.max_fragment_bytes % 4
        ):
            raise ValueError("dense semantic fragment byte bound is invalid")


@dataclass(frozen=True)
class DenseFragmentSubmission:
    fragment_id: int
    global_step: int
    base_fragment_version: int
    target_fragment_version: int
    payload_bytes: int
    push_attempts: int
    pull_to_push_seconds: float


@dataclass(frozen=True)
class PendingDenseWirePolicy:
    sweep_update_id: str
    policy_version: int
    fragment_versions: tuple[int, ...]
    payloads: tuple[StoredPayload, ...]
    submissions: tuple[DenseFragmentSubmission, ...]
    terminal_manifest: FinalManifest | None

    @property
    def terminal(self) -> bool:
        return self.terminal_manifest is not None


@dataclass
class _InFlightDenseSweep:
    sweep_update_id: str
    base_policy_version: int
    target_policy_version: int
    trained_tokens: int
    expected_versions: tuple[int, ...]
    delta_parts: dict[int, PartsFactory]
    next_fragment: int
    submissions: list[DenseFragmentSubmission]
    committed_payloads: dict[int, StoredPayload]


class DenseSweepWire:
    """Stream sequential fragments while committing progress only per sweep."""

    def __init__(
        self,
        layout: FragmentLayout,
        config: DenseSweepConfig,
        *,
        client: DenseSweepClient | None = None,
    ) -> None:
        self.layout = layout
        self.config = config
        oversized = [
            fragment_id
            for fragment_id, fragment in enumerate(layout.fragments)
            if fragment.numel * 4 > config.max_fragment_bytes
        ]
        if oversized:
            raise ValueError(
                "dense semantic fragment exceeds the configured byte bound"
            )
        self.client = client or SyncerClient(
            config.syncer_addr,
            config.learner_id,
            layout,
            dtype=DTYPE_F32,
            num_streams=config.wan_streams,
            # A durable strict-sweep syncer may restart between queueing and
            # checkpoint commit.  The controller retains immutable fragment
            # sources and replays only an exact re-PULL of that same step.
            max_reconnects=None,
        )
        self.policy_version: int | None = None
        self.fragment_versions: tuple[int, ...] | None = None
        self.pending: PendingDenseWirePolicy | None = None
        self.in_flight: _InFlightDenseSweep | None = None
        self._permits: dict[int, PullRequest] = {}
        self._updates: dict[tuple[int, int], _ReceivedPayload] = {}
        self._transactional_sink: object | None = None
        self._accept_current_versions = False
        self._started = False
        self._closed = False

    @property
    def num_fragments(self) -> int:
        return self.layout.num_fragments

    @property
    def total_fragment_steps(self) -> int:
        return self.config.policy_rounds * self.num_fragments

    def start(
        self,
        initial_parts: dict[int, PartsFactory],
        *,
        policy_version: int = 0,
        fragment_versions: tuple[int, ...] | None = None,
        payload_sink: PayloadSink | None = None,
    ) -> tuple[StoredPayload, ...]:
        if self._started or self._closed:
            raise RuntimeError("dense sweep can only be started once")
        if set(initial_parts) != set(range(self.num_fragments)):
            raise ValueError("dense initial parts do not cover the layout")
        if (
            type(policy_version) is not int
            or policy_version < 0
            or policy_version > self.config.policy_rounds
        ):
            raise ValueError("dense start policy version is invalid")
        expected_versions = self.versions_for_policy(policy_version)
        if fragment_versions is None:
            fragment_versions = expected_versions
        if fragment_versions != expected_versions:
            raise ValueError("dense start fragment versions are not a complete policy")
        self._started = True
        self.policy_version = policy_version
        self.fragment_versions = fragment_versions
        self._install_payload_sink(payload_sink)
        self.client.start()
        if (
            policy_version == 0
            and self.config.send_initial_params
            and self.config.learner_id == 0
        ):
            for fragment_id in range(self.num_fragments):
                if (
                    self.client.send_init_parts(
                        fragment_id,
                        initial_parts[fragment_id](),
                    )
                    is not True
                ):
                    raise RuntimeError("dense initial fragment was not fully queued")
        self._accept_current_versions = True
        try:
            return self._wait_for_broadcast_versions(
                fragment_versions,
                payload_sink=payload_sink,
            )
        finally:
            self._accept_current_versions = False

    def exchange(
        self,
        *,
        base_policy_version: int,
        trained_tokens: int,
        sweep_update_id: str,
        delta_parts: dict[int, PartsFactory],
        payload_sink: PayloadSink | None = None,
    ) -> PendingDenseWirePolicy:
        self._require_active()
        if self.pending is not None:
            raise RuntimeError("previous dense policy is not committed")
        if self.policy_version != base_policy_version or self.fragment_versions is None:
            raise RuntimeError("dense sweep base policy changed")
        if type(trained_tokens) is not int or trained_tokens < 1:
            raise ValueError("dense sweep trained-token count must be positive")
        if set(delta_parts) != set(range(self.num_fragments)):
            raise ValueError("dense update parts do not cover the layout")
        if not isinstance(sweep_update_id, str) or not _SHA256.fullmatch(
            sweep_update_id
        ):
            raise ValueError("dense sweep update identity must be a SHA256")
        target_policy_version = base_policy_version + 1
        if target_policy_version > self.config.policy_rounds:
            raise RuntimeError("dense update exceeds the configured policy budget")
        self._install_payload_sink(payload_sink)

        expected_versions = self.versions_for_policy(target_policy_version)
        if self.in_flight is None:
            self.in_flight = _InFlightDenseSweep(
                sweep_update_id=sweep_update_id,
                base_policy_version=base_policy_version,
                target_policy_version=target_policy_version,
                trained_tokens=trained_tokens,
                expected_versions=expected_versions,
                delta_parts=dict(delta_parts),
                next_fragment=0,
                submissions=[],
                committed_payloads={},
            )
        active = self.in_flight
        if (
            active.sweep_update_id != sweep_update_id
            or active.base_policy_version != base_policy_version
            or active.target_policy_version != target_policy_version
            or active.trained_tokens != trained_tokens
            or active.expected_versions != expected_versions
            or set(active.delta_parts) != set(delta_parts)
        ):
            raise RuntimeError(
                "a different dense update cannot replace an in-flight sweep"
            )
        for fragment_id in range(active.next_fragment, self.num_fragments):
            global_step = self.global_step(target_policy_version, fragment_id)
            permit = self._wait_for_permit(global_step, fragment_id)
            base_fragment_version = self.fragment_versions[fragment_id]
            expected_base = self.fragment_base_version(
                target_policy_version,
                fragment_id,
            )
            if base_fragment_version != expected_base:
                raise RuntimeError("dense fragment anchor version changed")
            first_pushed_at = time.monotonic()
            push_attempts = 0
            parts_factory = active.delta_parts[fragment_id]

            def push(
                active_permit: PullRequest,
                *,
                active_fragment_id: int = fragment_id,
                active_global_step: int = global_step,
                active_base_version: int = base_fragment_version,
                active_parts_factory: PartsFactory = parts_factory,
            ) -> None:
                nonlocal push_attempts
                queued = self.client.push_fragment_parts(
                    active_fragment_id,
                    active_global_step,
                    active_permit.round_attempt,
                    active_base_version,
                    target_policy_version,
                    1,
                    trained_tokens,
                    active_parts_factory(),
                )
                if queued is True:
                    push_attempts += 1
                elif queued is not False:
                    raise RuntimeError(
                        "dense fragment enqueue returned an invalid result"
                    )

            push(permit)
            if global_step == self.total_fragment_steps:
                self._wait_for_terminal_commit(
                    global_step,
                    fragment_id,
                    push,
                )
            else:
                active.committed_payloads[fragment_id] = self._wait_for_fragment_commit(
                    global_step,
                    fragment_id,
                    expected_versions[fragment_id],
                    push,
                    payload_sink=payload_sink,
                )
            active.submissions.append(
                DenseFragmentSubmission(
                    fragment_id,
                    global_step,
                    base_fragment_version,
                    expected_versions[fragment_id],
                    self.layout.fragments[fragment_id].numel * 4,
                    push_attempts,
                    max(0.0, first_pushed_at - permit.received_at),
                )
            )
            active.next_fragment = fragment_id + 1

        terminal_manifest = None
        if target_policy_version == self.config.policy_rounds:
            terminal_manifest, final = self.client.wait_for_final_fragments(
                timeout=self.config.wait_timeout,
            )
            if (
                terminal_manifest.global_step != self.total_fragment_steps
                or terminal_manifest.versions != expected_versions
            ):
                raise RuntimeError("dense terminal manifest is not a complete sweep")
            payloads = self._final_payloads(
                final,
                terminal_manifest,
                payload_sink=payload_sink,
            )
        else:
            if set(active.committed_payloads) != set(range(self.num_fragments)):
                raise RuntimeError("dense nonterminal sweep is not fully committed")
            payloads = tuple(
                active.committed_payloads[fragment_id]
                for fragment_id in range(self.num_fragments)
            )
        self.pending = PendingDenseWirePolicy(
            sweep_update_id,
            target_policy_version,
            expected_versions,
            payloads,
            tuple(active.submissions),
            terminal_manifest,
        )
        self.in_flight = None
        return self.pending

    def commit_applied(self, pending: PendingDenseWirePolicy) -> None:
        if self.pending is None or pending is not self.pending:
            raise RuntimeError("dense wire commit does not match the pending sweep")
        if (
            self.policy_version is None
            or pending.policy_version != self.policy_version + 1
        ):
            raise RuntimeError("dense wire commit is not monotonic")
        if pending.terminal_manifest is not None:
            self.client.acknowledge_finalization(pending.terminal_manifest)
        self.policy_version = pending.policy_version
        self.fragment_versions = pending.fragment_versions
        self.pending = None

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            for received in self._updates.values():
                if received.discard is not None:
                    received.discard()
            self._updates.clear()
            try:
                self.client.close()
            finally:
                release = getattr(self._transactional_sink, "release_all", None)
                if callable(release):
                    release()
                self._transactional_sink = None

    def versions_for_policy(self, policy_version: int) -> tuple[int, ...]:
        if policy_version == 0:
            return (0,) * self.num_fragments
        return tuple(
            (policy_version - 1) * self.num_fragments + fragment_id + 1
            for fragment_id in range(self.num_fragments)
        )

    def fragment_base_version(self, policy_version: int, fragment_id: int) -> int:
        if policy_version == 1:
            return 0
        return (policy_version - 2) * self.num_fragments + fragment_id + 1

    def global_step(self, policy_version: int, fragment_id: int) -> int:
        return (policy_version - 1) * self.num_fragments + fragment_id + 1

    def _wait_for_permit(self, global_step: int, fragment_id: int) -> PullRequest:
        deadline = time.monotonic() + self.config.wait_timeout
        while True:
            self._drain_wire()
            permit = self._permits.pop(global_step, None)
            if permit is not None:
                if permit.fragment_id != fragment_id or permit.round_attempt != 1:
                    raise RuntimeError("dense sweep received an invalid PULL permit")
                return permit
            if time.monotonic() >= deadline:
                raise TimeoutError("dense sweep timed out waiting for a PULL permit")
            time.sleep(self.config.poll_seconds)

    def _wait_for_fragment_commit(
        self,
        global_step: int,
        fragment_id: int,
        target_version: int,
        replay: Callable[[PullRequest], None],
        *,
        payload_sink: PayloadSink | None,
    ) -> StoredPayload:
        deadline = time.monotonic() + self.config.wait_timeout
        while True:
            self._drain_wire()
            key = (fragment_id, target_version)
            received = self._updates.get(key)
            if received is not None:
                stored = self._store_received_payload(
                    fragment_id,
                    target_version,
                    received,
                    payload_sink,
                )
                self._updates.pop(key, None)
                return stored
            repeated = self._permits.pop(global_step, None)
            if repeated is not None:
                if repeated.fragment_id != fragment_id:
                    raise RuntimeError("dense replay PULL changed fragment identity")
                replay(repeated)
            if time.monotonic() >= deadline:
                raise TimeoutError("dense sweep timed out waiting for fragment commit")
            time.sleep(self.config.poll_seconds)

    def _wait_for_terminal_commit(
        self,
        global_step: int,
        fragment_id: int,
        replay: Callable[[PullRequest], None],
    ) -> None:
        deadline = time.monotonic() + self.config.wait_timeout
        while True:
            self._drain_wire()
            if self.client.finalizing.is_set():
                return
            repeated = self._permits.pop(global_step, None)
            if repeated is not None:
                if repeated.fragment_id != fragment_id:
                    raise RuntimeError(
                        "dense terminal replay changed fragment identity"
                    )
                replay(repeated)
            if time.monotonic() >= deadline:
                raise TimeoutError("dense sweep timed out waiting for terminal commit")
            time.sleep(self.config.poll_seconds)

    def _wait_for_broadcast_versions(
        self,
        expected_versions: tuple[int, ...],
        *,
        payload_sink: PayloadSink | None,
    ) -> tuple[StoredPayload, ...]:
        deadline = time.monotonic() + self.config.wait_timeout
        while True:
            self._drain_wire()
            if all(
                (fragment_id, version) in self._updates
                for fragment_id, version in enumerate(expected_versions)
            ):
                stored = tuple(
                    self._store_received_payload(
                        fragment_id,
                        version,
                        self._updates[(fragment_id, version)],
                        payload_sink,
                    )
                    for fragment_id, version in enumerate(expected_versions)
                )
                for fragment_id, version in enumerate(expected_versions):
                    self._updates.pop((fragment_id, version), None)
                return stored
            if time.monotonic() >= deadline:
                raise TimeoutError("dense sweep timed out waiting for a complete cut")
            time.sleep(self.config.poll_seconds)

    def _drain_wire(self) -> None:
        self.client.check_health()
        for update in self.client.drain_updates():
            fragment_id = update.fragment_id
            version = update.version
            if not 0 <= fragment_id < self.num_fragments or version < 0:
                raise RuntimeError("dense sweep received an invalid broadcast")
            if self.fragment_versions is not None:
                current = self.fragment_versions[fragment_id]
                if version < current or version > current + self.num_fragments:
                    raise RuntimeError("dense sweep received a nonmonotonic broadcast")
            key = (fragment_id, version)
            previous = self._updates.get(key)
            if getattr(update, "stored", False):
                incoming_hash = getattr(update, "payload_hash", None)
                if not isinstance(incoming_hash, bytes) or len(incoming_hash) != 32:
                    raise RuntimeError("stored dense broadcast has no payload hash")
                received = _ReceivedPayload(
                    update.data,
                    incoming_hash,
                    True,
                    getattr(update, "discard", None),
                )
            else:
                incoming_hash = hashlib.sha256(
                    memoryview(update.data).cast("B")
                ).digest()
                received = _ReceivedPayload(update.data, incoming_hash, False)
            if (
                not self._accept_current_versions
                and self.fragment_versions is not None
                and version <= self.fragment_versions[fragment_id]
            ):
                if received.discard is not None:
                    received.discard()
                continue
            if previous is not None and previous.payload_hash != incoming_hash:
                if received.discard is not None:
                    received.discard()
                raise RuntimeError("dense sweep received conflicting broadcasts")
            if previous is None:
                self._updates[key] = received
            elif received.discard is not None:
                received.discard()
        for permit in self.client.drain_pulls():
            if (
                not 0 <= permit.fragment_id < self.num_fragments
                or permit.global_step < 1
                or permit.global_step > self.total_fragment_steps
                or permit.round_attempt != 1
            ):
                raise RuntimeError("dense sweep received an invalid PULL permit")
            expected_fragment = (permit.global_step - 1) % self.num_fragments
            if permit.fragment_id != expected_fragment:
                raise RuntimeError("dense sweep PULL is outside canonical sweep order")
            previous = self._permits.get(permit.global_step)
            if previous is None or permit.round_attempt > previous.round_attempt:
                self._permits[permit.global_step] = permit
            elif permit.round_attempt == previous.round_attempt and previous != permit:
                raise RuntimeError("dense sweep received conflicting PULL permits")

    def _final_payloads(
        self,
        fragments: list[object],
        manifest: FinalManifest,
        *,
        payload_sink: PayloadSink | None,
    ) -> tuple[StoredPayload, ...]:
        if len(fragments) != self.num_fragments:
            raise RuntimeError("dense terminal cut is incomplete")
        payloads = {}
        for fragment in fragments:
            fragment_id = fragment.fragment_id
            if (
                not 0 <= fragment_id < self.num_fragments
                or fragment_id in payloads
                or fragment.version != manifest.versions[fragment_id]
            ):
                raise RuntimeError("dense terminal fragment identity changed")
            if getattr(fragment, "stored", False):
                payload_hash = getattr(fragment, "payload_hash", None)
                if not isinstance(payload_hash, bytes) or len(payload_hash) != 32:
                    raise RuntimeError("stored terminal fragment has no payload hash")
                payloads[fragment_id] = _ReceivedPayload(
                    fragment.data,
                    payload_hash,
                    True,
                    getattr(fragment, "discard", None),
                )
            else:
                payloads[fragment_id] = _ReceivedPayload(
                    fragment.data,
                    hashlib.sha256(memoryview(fragment.data).cast("B")).digest(),
                    False,
                )
        if set(payloads) != set(range(self.num_fragments)):
            raise RuntimeError("dense terminal cut does not cover the layout")
        return tuple(
            self._store_received_payload(
                index,
                manifest.versions[index],
                payloads[index],
                payload_sink,
            )
            for index in range(self.num_fragments)
        )

    @staticmethod
    def _store_received_payload(
        fragment_id: int,
        version: int,
        received: _ReceivedPayload,
        sink: PayloadSink | None,
    ) -> StoredPayload:
        if received.stored:
            return received.value
        if sink is None:
            return received.value
        stored = sink(fragment_id, version, received.value)
        if stored is None:
            raise RuntimeError("dense payload sink returned no durable reference")
        return stored

    def _install_payload_sink(self, sink: PayloadSink | None) -> None:
        install = getattr(self.client, "install_inbound_chunk_sink", None)
        if not callable(install):
            return
        if sink is None:
            self._transactional_sink = None
            self._install_transactional_sink(install, None)
            return
        factory = getattr(sink, "inbound_chunk_sink", None)
        if callable(factory):
            transactional = factory()
        else:
            try:
                from .miles_chunked_full_parameter import (
                    AuthoritativeFragmentSink,
                )
            except ImportError:
                AuthoritativeFragmentSink = ()
            transactional = (
                _MilesRayInboundChunkSink(sink)
                if isinstance(sink, AuthoritativeFragmentSink)
                else None
            )
        self._transactional_sink = transactional
        self._install_transactional_sink(install, transactional)

    def _install_transactional_sink(
        self,
        install: Callable[[object | None], None],
        transactional: object | None,
    ) -> None:
        deadline = time.monotonic() + self.config.wait_timeout
        while True:
            try:
                install(transactional)
                return
            except RuntimeError as exc:
                if "mid-message" not in str(exc):
                    raise
                self.client.check_health()
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "dense sweep timed out replacing its inbound object sink"
                    ) from exc
                time.sleep(self.config.poll_seconds)

    def _require_active(self) -> None:
        if not self._started or self._closed:
            raise RuntimeError("dense sweep is not active")
