"""Learner side of bf16 downlink deltas (S26 E3; syncer ``src/downlink.rs``).

A session whose dtype carries ``DTYPE_FLAG_BCAST_DELTA`` receives every
BCAST_FRAGMENT with a 41-byte downlink header after ``fragment_id, version``::

    kind u8 (0 FULL, 1 DELTA) | base_version u64 | anchor_sha256 [32]

FULL carries f32 values; DELTA carries ``bf16(theta - anchor)``. The client
keeps one f32 *anchor* per fragment, bit-identical to the syncer's: FULL sets
it, DELTA adds the decoded delta in f32 (numpy float32 addition rounds like
Rust's). The digest is checked after every update. Callers above the client
(``drain_updates``, inbound chunk sinks) only ever see the reconstructed f32
anchor, so adapters, N7.6 cuts and manifests are unchanged.

A missing or mismatched anchor raises :class:`DownlinkResync` (a
``ConnectionError``): the client drops its connection group, and the
syncer's reconnect recovery re-sends every fragment as FULL.
"""

from __future__ import annotations

import hashlib
import struct
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # numpy is imported lazily: yeto.protocol stays import-light
    import numpy as np

KIND_FULL = 0
KIND_DELTA = 1
HEADER = struct.Struct("<BQ32s")
HEADER_LEN = HEADER.size  # 41


class DownlinkResync(ConnectionError):
    """The learner's anchor cannot take this BCAST; reconnect for a FULL."""


def tensor_nbytes(kind: int, numel: int) -> int:
    if kind == KIND_FULL:
        return 4 * numel
    if kind == KIND_DELTA:
        return 2 * numel
    raise ValueError(f"unknown downlink kind {kind}")


@dataclass
class _Held:
    version: int
    values: np.ndarray  # float32, never mutated after commit


class DownlinkDecode:
    """Streams one BCAST's tensor bytes into the next f32 anchor."""

    def __init__(self, fragment_id: int, version: int, kind: int, base: np.ndarray | None,
                 digest: bytes, numel: int) -> None:
        import numpy as np

        self.fragment_id = fragment_id
        self.version = version
        self.kind = kind
        self._base = base
        self._digest = digest
        self.values = np.empty(numel, dtype=np.float32)
        self._width = 4 if kind == KIND_FULL else 2
        self._carry = b""
        self._filled = 0
        self._hasher = hashlib.sha256()

    def feed(self, data: bytes | bytearray | memoryview) -> bytes:
        """Decode the next wire bytes; return the f32 bytes they complete."""
        import numpy as np

        raw = self._carry + bytes(data)
        whole = len(raw) - len(raw) % self._width
        self._carry = raw[whole:]
        if not whole:
            return b""
        count = whole // self._width
        start, end = self._filled, self._filled + count
        if end > self.values.size:
            raise ValueError("downlink BCAST has more values than its fragment")
        out = self.values[start:end]
        if self.kind == KIND_FULL:
            out[:] = np.frombuffer(raw, dtype="<f4", count=count)
        else:
            bits = np.frombuffer(raw, dtype="<u2", count=count).astype(np.uint32) << 16
            np.add(self._base[start:end], bits.view(np.float32), out=out)
        if not np.isfinite(out).all():
            raise ValueError("downlink BCAST decodes to a non-finite value")
        self._filled = end
        encoded = out.astype("<f4", copy=False).tobytes()
        self._hasher.update(encoded)
        return encoded

    def finish(self) -> None:
        if self._carry or self._filled != self.values.size:
            raise ValueError("downlink BCAST ended before its fragment was complete")
        if self._hasher.digest() != self._digest:
            raise DownlinkResync(
                f"downlink anchor digest mismatch for fragment {self.fragment_id} "
                f"version {self.version}")


class DownlinkAnchors:
    """Per-fragment f32 anchors shared (bit for bit) with the syncer."""

    def __init__(self, num_fragments: int) -> None:
        self._held: list[_Held | None] = [None] * num_fragments
        self._lock = threading.Lock()

    def version(self, fragment_id: int) -> int | None:
        with self._lock:
            held = self._held[fragment_id]
            return None if held is None else held.version

    def begin(self, fragment_id: int, version: int, header: bytes, numel: int) -> DownlinkDecode:
        kind, base_version, digest = HEADER.unpack(header)
        if kind == KIND_FULL:
            return DownlinkDecode(fragment_id, version, kind, None, digest, numel)
        if kind != KIND_DELTA:
            raise ValueError(f"unknown downlink kind {kind}")
        with self._lock:
            held = self._held[fragment_id]
        if held is None or held.version != base_version or held.values.size != numel:
            raise DownlinkResync(
                f"downlink delta for fragment {fragment_id} version {version} needs anchor "
                f"version {base_version}; learner holds "
                f"{None if held is None else held.version}")
        return DownlinkDecode(fragment_id, version, kind, held.values, digest, numel)

    def commit(self, decode: DownlinkDecode) -> None:
        """Adopt a finished decode as the anchor unless a newer one is held."""
        with self._lock:
            held = self._held[decode.fragment_id]
            if held is None or decode.version >= held.version:
                self._held[decode.fragment_id] = _Held(decode.version, decode.values)

    def decode_whole(self, fragment_id: int, version: int, payload: bytes | memoryview,
                     numel: int) -> bytes:
        """Non-streamed BCAST: header + tensor -> committed f32 anchor bytes."""
        if len(payload) < HEADER_LEN:
            raise ValueError("downlink BCAST header truncated")
        decode = self.begin(fragment_id, version, bytes(payload[:HEADER_LEN]), numel)
        if len(payload) - HEADER_LEN != tensor_nbytes(decode.kind, numel):
            raise ValueError(f"downlink BCAST {fragment_id} has {len(payload) - HEADER_LEN} "
                             f"tensor bytes, expected {tensor_nbytes(decode.kind, numel)}")
        data = decode.feed(payload[HEADER_LEN:])
        decode.finish()
        self.commit(decode)
        return data


@dataclass
class _Transaction:
    inner: object
    passthrough: bool = True
    fragment_id: int | None = None
    version: int | None = None
    numel: int = 0
    header: bytearray = field(default_factory=bytearray)
    decode: DownlinkDecode | None = None
    out_offset: int = 0


class DownlinkChunkSink:
    """Wraps a transactional inbound chunk sink for a downlink session.

    BCAST payload bytes (downlink header + f32/bf16 tensor) are decoded in
    order and handed to the wrapped sink as the plain f32 fragment; FINAL
    passes through. Out-of-order chunks are staged here as bytes (the
    wrapped sink sees only in-order f32 bytes)."""

    def __init__(self, inner: object, anchors: DownlinkAnchors, numels: list[int],
                 bcast_type: int) -> None:
        self.inner = inner
        self.anchors = anchors
        self.numels = numels
        self.bcast_type = bcast_type

    def begin_message(self, message_id: int, total_bytes: int) -> _Transaction:
        return _Transaction(self.inner.begin_message(message_id, total_bytes))

    def bind_fragment(self, tx: _Transaction, msg_type: int, fragment_id: int, version: int,
                      payload_bytes: int) -> None:
        if msg_type != self.bcast_type:
            self.inner.bind_fragment(tx.inner, msg_type, fragment_id, version, payload_bytes)
            return
        tx.passthrough = False
        tx.fragment_id, tx.version = fragment_id, version
        tx.numel = self.numels[fragment_id]
        self.inner.bind_fragment(tx.inner, msg_type, fragment_id, version, 4 * tx.numel)

    def stage_chunk(self, tx: _Transaction, payload) -> bytes:
        del tx
        return bytes(payload)

    def consume_staged_chunk(self, tx: _Transaction, offset: int, token: bytes, nbytes: int) -> None:
        if len(token) != nbytes:
            raise ValueError("staged downlink chunk is malformed")
        self.consume_chunk(tx, offset, token)

    def consume_chunk(self, tx: _Transaction, offset: int, payload) -> None:
        if tx.passthrough:
            self.inner.consume_chunk(tx.inner, offset, payload)
            return
        data = memoryview(payload).cast("B")
        if tx.decode is None:
            take = min(HEADER_LEN - len(tx.header), data.nbytes)
            tx.header += data[:take]
            data = data[take:]
            if len(tx.header) < HEADER_LEN:
                return
            tx.decode = self.anchors.begin(tx.fragment_id, tx.version, bytes(tx.header), tx.numel)
        out = tx.decode.feed(data)
        if out:
            self.inner.consume_chunk(tx.inner, tx.out_offset, out)
            tx.out_offset += len(out)

    def finish_message(self, tx: _Transaction):
        if tx.passthrough:
            return self.inner.finish_message(tx.inner)
        if tx.decode is None:
            raise ValueError("downlink BCAST ended inside its header")
        tx.decode.finish()
        committed = self.inner.finish_message(tx.inner)
        self.anchors.commit(tx.decode)
        return committed

    def abort_message(self, tx: _Transaction, staged_tokens) -> None:
        del staged_tokens  # plain bytes held here
        self.inner.abort_message(tx.inner, ())
