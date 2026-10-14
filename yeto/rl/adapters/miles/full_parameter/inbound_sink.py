"""Transactional Ray sink that streams syncer fragment bytes into Miles.

Every full-parameter role stream installs one on its ``SyncerClient``: each
inbound BCAST/FINAL fragment is written chunk by chunk into the Miles typed
Ray layout of the authoritative cut, and only a complete, hash-checked
fragment becomes visible to the trainer.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field

from yeto.protocol import (
    MSG_BCAST_FRAGMENT,
    MSG_FINAL_FRAGMENT,
    StreamedInboundPayload,
)

BytePart = bytes | bytearray | memoryview


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


class MilesRayInboundChunkSink:
    """Stream exact wire bytes into the production Miles typed Ray layout."""

    def __init__(self, sink: object) -> None:
        from miles.backends.megatron_utils.full_parameter_state import (
            _resolve_ray_module,
        )

        from yeto.rl.adapters.miles.models.chunked_full_parameter import AuthoritativeFragmentSink

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

        from yeto.rl.adapters.miles.models.chunked_full_parameter import StoredAuthoritativeFragment

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
