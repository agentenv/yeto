"""Python side of the learner <-> syncer wire protocol (docs/PROTOCOL.md v4).

A learner owns a connection *group*: stream 0 carries control messages
(HELLO, PULL_REQ, HEARTBEAT, FINAL_MANIFEST/ACK, SHUTDOWN); large payloads
(INIT_PARAMS, PUSH_FRAGMENT, BCAST_FRAGMENT, FINAL_FRAGMENT) are split into
4 MiB CHUNK envelopes striped round-robin across extra data sockets, which
multiplies WAN throughput over what a single congestion-window-limited TCP
stream can carry.

The training loop must never block on the WAN, so each socket gets a sender
thread (draining a queue) and a receiver thread (reassembling chunks and
sorting inbound messages into pull-request / broadcast inboxes consumed at
inner-step boundaries).

The client survives socket failures: a supervisor thread tears down the whole
connection group when any socket errors, then redials with exponential
backoff and re-sends HELLO/DATA_HELLO. On HELLO the syncer rebroadcasts every
initialized fragment at its current version, so a reconnected learner is
caught up within one message exchange; anything queued toward the old
connection (pushes, heartbeats, pending pull requests, partial chunk
reassemblies) is stale by then and is dropped. Broadcasts already received
are kept. The training loop keeps calling push_fragment/heartbeat/drain_*
during an outage; those calls silently drop instead of raising, and
check_health only raises once reconnection has been given up for good.
"""

from __future__ import annotations

import hashlib
import itertools
import queue
import secrets
import socket
import struct
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .downlink import HEADER_LEN as DOWNLINK_HEADER_LEN
from .downlink import DownlinkAnchors, DownlinkChunkSink
from .downlink import tensor_nbytes as downlink_tensor_nbytes
from .fragments import MERGE_ISO, FragmentLayout
from . import recovery_protocol as rp
from .wan_streams import STREAM_SCHEDULES, StreamMeter, StreamSnapshot, StreamStalledError, pick_stream

if TYPE_CHECKING:
    import numpy as np

MAGIC = 0xD170C0DE
PROTOCOL_VERSION = 4

MSG_HELLO = 1
MSG_INIT_PARAMS = 2
MSG_PULL_REQ = 3
MSG_PUSH_FRAGMENT = 4
MSG_BCAST_FRAGMENT = 5
MSG_HEARTBEAT = 6
MSG_SHUTDOWN = 7
MSG_DATA_HELLO = 8
MSG_CHUNK = 9
MSG_ERROR = 10
MSG_FINAL_MANIFEST = 11
MSG_FINAL_ACK = 12
MSG_FINAL_FRAGMENT = 13
MSG_BUDGET_DONE = 14
# S26 E3b: link-bandwidth probe connection (yeto/wan_probe.py; syncer serve_bw_probe).
MSG_BW_PROBE = 26
# An island leaves the pool: u32 learner_id | u8 reason (yeto.rl.pool_client.LeaveReason).
MSG_LEAVE = 27
# Local-init startup (S27): the island's per-fragment INIT digests, and the
# syncer's confirmation that every island holds the same version 0 (see
# syncer/src/local_init.rs).
MSG_INIT_DIGEST = 28
MSG_INIT_CONFIRM = 29
# S27 G-EA: receive-only subscriber (syncer subscriber.rs); 60-66 are the
# sample relay (yeto.rl.spot_rollout.relay, syncer sample_relay.rs).
MSG_SUBSCRIBE = 67
SUBSCRIBER_ID_BIT = 0x8000_0000
SUBSCRIBE_MAC_DOMAIN = b"yeto-subscribe-mac-v1\0"
# SUBSCRIBE body trailer (inside the MAC): the sampler process's member
# incarnation (docs/PROTOCOL.md "Sampler members"; syncer subscriber.rs).
SUBSCRIBE_INCARNATION_MAGIC = b"YSI1"


def encode_subscribe_incarnation(incarnation: int) -> bytes:
    """The ``YSI1 | incarnation u64`` trailer of a SUBSCRIBE body."""
    if not 0 < int(incarnation) < 1 << 64:
        raise ValueError("a member incarnation is a nonzero u64")
    return SUBSCRIBE_INCARNATION_MAGIC + struct.pack("<Q", int(incarnation))

FINALIZATION_REVISION = 1
FINALIZATION_TIMEOUT = 900.0

DTYPE_F32 = 1
DTYPE_BF16 = 2
# Session dtype 3: PUSH_FRAGMENT payloads are block-quantized 4-bit E3M0
# base-relative learner deltas; INIT_PARAMS and BCAST_FRAGMENT stay bf16
# (see bulk_dtype and docs/PROTOCOL.md v4).
DTYPE_Q4 = 3
# Session dtype 4 (S26 E3): PUSH_FRAGMENT deltas travel as bf16; INIT_PARAMS,
# BCAST_FRAGMENT and FINAL stay exact f32. Callers still hand the client f32
# delta bytes; the client rounds them (round-to-nearest-even, as the syncer's
# ``half::bf16::from_f32``) on the way out.
DTYPE_F32_PUSH_BF16 = 4
# Session dtype 5 (S26 E3): PUSH_FRAGMENT deltas travel as block-scaled int8
# (``f32_bytes_to_i8``); INIT/BCAST/FINAL stay f32. Callers hand f32 bytes.
DTYPE_F32_PUSH_I8 = 5
# Session dtype 6 (S27 P3): PUSH_FRAGMENT deltas travel in the dtype-3 q4
# block codec (``f32_bytes_to_q4``); INIT/BCAST/FINAL stay f32. Callers hand
# f32 bytes; only the outer gradient is quantized.
DTYPE_F32_PUSH_Q4 = 6
# Session dtype 7 (S27 P3): PUSH_FRAGMENT deltas travel as linear symmetric
# int4 with a per-block absmax scale (``f32_bytes_to_i4``); INIT/BCAST/FINAL
# stay f32. Callers hand f32 bytes.
DTYPE_F32_PUSH_I4 = 7
# Full-parameter push options (CLI names) -> session dtype; f32 is exact.
FULL_PARAMETER_PUSH_DTYPES = {
    "f32": DTYPE_F32,
    "bf16": DTYPE_F32_PUSH_BF16,
    "int8": DTYPE_F32_PUSH_I8,
    "int4": DTYPE_F32_PUSH_I4,
    "int4-log": DTYPE_F32_PUSH_Q4,
}
# Session-dtype flag (S26 E3): BCAST_FRAGMENT carries bf16 downlink deltas
# between f32 refreshes (yeto/downlink.py). Only on f32-bulk sessions.
DTYPE_FLAG_BCAST_DELTA = 0x80
# Session-dtype flag (S27): quantization blocks of PUSH_FRAGMENT deltas
# restart at every tensor boundary of the fragment, so a tensor replicated on
# several ranks quantizes the same whatever its neighbours are. Only on the
# block-quantized push dtypes (int8, int4, int4-log). Sessions without it
# keep blocks running across the whole fragment, byte for byte as before.
DTYPE_FLAG_TENSOR_BLOCKS = 0x40
I8_BLOCK_MIN = 32
I8_BLOCK_MAX = 1 << 20
DEFAULT_I8_BLOCK = 128
I4_BLOCK_MIN = 64
I4_BLOCK_MAX = 256
DEFAULT_I4_BLOCK = 128

CHUNK_SIZE = 4 * 1024 * 1024
MAX_HANDSHAKE_FRAME = 16 * 1024 * 1024
MAX_ERROR_FRAME = 64 * 1024
Q4_BLOCK = 256
MAX_PARTIAL_MESSAGES = 64

RECONNECT_BACKOFF_START = 1.0
RECONNECT_BACKOFF_CAP = 30.0
RECONNECT_DIAL_TIMEOUT = 20.0

_HEADER = struct.Struct("<IBQ")  # magic, type, payload length
_CHUNK_HEAD = struct.Struct("<QQQ")  # msg_id, total_len, offset
_HELLO_HEAD = struct.Struct("<HIQBI")  # version, learner, generation, dtype, fragments
HELLO_COMPAT_MAGIC = b"YCG1"  # server.rs HELLO_COMPAT_MAGIC
_DATA_HELLO = struct.Struct("<HIQH")  # version, learner, generation, stream index
_FINAL_MANIFEST_HEAD = struct.Struct("<HQI")  # revision, global_step, fragments
_FINAL_ACK = struct.Struct("<HQ")  # revision, global_step
_BUDGET_DONE = struct.Struct("<Q")  # local steps


# Session dtypes whose INIT/BCAST/FINAL tensors are f32 (downlink deltas allowed).
_F32_BULK_PUSH_DTYPES = (
    DTYPE_F32, DTYPE_F32_PUSH_BF16, DTYPE_F32_PUSH_I8, DTYPE_F32_PUSH_Q4, DTYPE_F32_PUSH_I4,
)


_BLOCK_QUANT_PUSH_DTYPES = (DTYPE_F32_PUSH_I8, DTYPE_F32_PUSH_Q4, DTYPE_F32_PUSH_I4)


def session_base(dtype: int) -> int:
    """The session dtype without its flags."""
    return dtype & ~(DTYPE_FLAG_BCAST_DELTA | DTYPE_FLAG_TENSOR_BLOCKS)


def tensor_blocks(dtype: int) -> bool:
    """Whether push quantization blocks restart at each tensor boundary."""
    return bool(dtype & DTYPE_FLAG_TENSOR_BLOCKS)


def bcast_delta(dtype: int) -> bool:
    """Whether the session receives bf16 downlink deltas."""
    return bool(dtype & DTYPE_FLAG_BCAST_DELTA)


def valid_session_dtype(dtype: int) -> bool:
    base = session_base(dtype)
    if tensor_blocks(dtype) and base not in _BLOCK_QUANT_PUSH_DTYPES:
        return False
    if base in _F32_BULK_PUSH_DTYPES:
        return 0 <= dtype <= 0xFF
    return base in (DTYPE_BF16, DTYPE_Q4) and not bcast_delta(dtype)


def bulk_dtype(dtype: int) -> int:
    """Dtype of INIT_PARAMS/BCAST_FRAGMENT tensors for a session dtype.

    Q4 applies only to push deltas (small dynamic range); full parameter
    payloads would not survive 4 bits, so they travel as bf16. A downlink
    session's BCAST is reconstructed to f32 before callers see it.
    """
    base = session_base(dtype)
    if base == DTYPE_Q4:
        return DTYPE_BF16
    if base in _F32_BULK_PUSH_DTYPES:
        return DTYPE_F32
    return base


def push_wire_dtype(dtype: int) -> int:
    """Dtype of unquantized PUSH_FRAGMENT deltas on the wire for a session dtype."""
    base = session_base(dtype)
    return DTYPE_BF16 if base == DTYPE_F32_PUSH_BF16 else base


def i8_nbytes(numel: int, block: int, segments: Sequence[int] | None = None) -> int:
    """Encoded size of an int8 push: u32 block, then per block f32 scale + i8s.

    ``segments`` are the tensor sizes of a tensor-block session; blocks
    restart at each one (``numel`` is then ignored).
    """
    if not I8_BLOCK_MIN <= block <= I8_BLOCK_MAX:
        raise ValueError(f"int8 block size {block} outside {I8_BLOCK_MIN}..{I8_BLOCK_MAX}")
    return 4 + sum(4 * -(-n // block) + n for n in (segments if segments is not None else (numel,)))


def f32_bytes_to_i8(
    data: bytes | bytearray | memoryview,
    block: int,
    *,
    header: bool = True,
    segments: Sequence[int] | None = None,
) -> bytes:
    """Quantize little-endian f32 values to block-scaled int8 (syncer ``decode_i8``).

    Per block of ``block`` values (the last may be shorter): f32 scale =
    absmax / 127, then ``round_half_even(x / scale)`` clipped to [-127, 127].
    Decoding ``q * scale`` errs by at most absmax / 254 per element (plus f32
    rounding). Blocks must start at multiples of ``block`` values; with
    ``header`` the u32 block size is prepended. With ``segments`` (tensor
    sizes) the blocks restart at every segment instead.
    """
    import numpy as np

    i8_nbytes(0, block)  # validates the block size
    if segments is not None:
        head = struct.pack("<I", block) if header else b""
        return head + _encode_segments(data, segments, lambda c: f32_bytes_to_i8(c, block, header=False))
    values = np.frombuffer(data, dtype="<f4")
    if not np.isfinite(values).all():
        raise ValueError("int8 push delta contains a non-finite value")
    count = values.size
    full = count - count % block
    parts = [struct.pack("<I", block)] if header else []

    def encode(chunk: np.ndarray) -> None:  # chunk: (blocks, width)
        absmax = np.abs(chunk).max(axis=1)
        scale = (absmax / np.float32(127)).astype(np.float32)
        safe = np.where(scale > 0, scale, np.float32(1))
        q = np.clip(np.rint(chunk / safe[:, None]), -127, 127).astype(np.int8)
        q[scale == 0] = 0
        rows = np.empty((chunk.shape[0], 4 + chunk.shape[1]), dtype=np.uint8)
        rows[:, :4] = scale.astype("<f4").view(np.uint8).reshape(-1, 4)
        rows[:, 4:] = q.view(np.uint8)
        parts.append(rows.tobytes())

    if full:
        encode(values[:full].reshape(-1, block))
    if count > full:
        encode(values[full:].reshape(1, -1))
    return b"".join(parts)


def _encode_segments(data, segments: Sequence[int], encode: Callable[[memoryview], bytes]) -> bytes:
    """Encode each tensor-sized segment of f32 ``data`` alone, so blocks restart at its start."""
    view = memoryview(data).cast("B")
    if view.nbytes != 4 * sum(segments):
        raise ValueError(f"push delta has {view.nbytes} bytes for segments of {sum(segments)} values")
    parts, at = [], 0
    for n in segments:
        parts.append(encode(view[at : at + 4 * n]))
        at += 4 * n
    return b"".join(parts)


def _block_aligned_chunks(
    parts: Iterable[bytes | bytearray | memoryview], block: int, segments: Sequence[int] | None
) -> Iterator[bytes]:
    """Regroup f32 byte parts into chunks that start on a block boundary of a segment.

    A chunk ends on a block boundary or on the end of its segment, so each
    chunk encodes alone and only a segment's last block is short. Without
    ``segments`` the whole stream is one segment.
    """
    ahead = iter(segments) if segments is not None else None
    left = next(ahead, 0) if ahead is not None else float("inf")
    carry = b""
    for part in parts:
        # Zero-copy: chunks are read-only views of the part.  Slicing bytes
        # here copied the rest of the part at every segment cut, quadratic in
        # the segments per part (S27 G-AA: ~14 s/GiB with 10 MiB tensors).
        data = memoryview(part).cast("B")
        start = 0
        if carry:
            # A carried partial block (under one block) is completed from the
            # head of this part and sent alone; only those bytes are copied
            # (S27 G-AP: joining the carry to the whole part copied ~2 parts
            # per part boundary).
            if left == 0:
                raise ValueError("f32 push delta is longer than its tensor segments")
            start = min(4 * min(left, block) - len(carry), len(data))
            carry += bytes(data[:start])
            if len(carry) < 4 * min(left, block):
                continue
            yield carry
            left -= len(carry) // 4
            carry = b""
            if left == 0 and ahead is not None:
                left = next(ahead, 0)
        while start < len(data):
            if left == 0:
                raise ValueError("f32 push delta is longer than its tensor segments")
            take = min((len(data) - start) // 4, left)
            cut = take if take == left else take - take % block
            if not cut:
                break
            yield data[start : start + 4 * cut]
            start += 4 * cut
            left -= cut
            if left == 0 and ahead is not None:
                left = next(ahead, 0)
        carry = bytes(data[start:])
    if len(carry) % 4:
        raise ValueError("f32 push delta ends mid-value")
    if carry:
        if left == 0:
            raise ValueError("f32 push delta is longer than its tensor segments")
        yield carry


def _f32_parts_to_i8(
    parts: Iterable[bytes | bytearray | memoryview], block: int, segments: Sequence[int] | None = None
) -> Iterator[bytes]:
    """Stream ``f32_bytes_to_i8`` over parts; blocks may span part boundaries."""
    yield struct.pack("<I", block)
    for chunk in _block_aligned_chunks(parts, block, segments):
        yield f32_bytes_to_i8(chunk, block, header=False)


def i4_nbytes(numel: int, block: int, segments: Sequence[int] | None = None) -> int:
    """Encoded size of a linear int4 push: u32 block, per block f32 scale, 2 values/byte.

    ``segments`` are the tensor sizes of a tensor-block session; blocks
    restart at each one (``numel`` is then ignored), so each pads its own tail.
    """
    if not (I4_BLOCK_MIN <= block <= I4_BLOCK_MAX and block % 2 == 0):
        raise ValueError(f"int4 block size {block} must be even and within {I4_BLOCK_MIN}..{I4_BLOCK_MAX}")
    return 4 + sum(4 * -(-n // block) + -(-n // 2) for n in (segments if segments is not None else (numel,)))


def f32_bytes_to_i4(
    data: bytes | bytearray | memoryview,
    block: int,
    *,
    header: bool = True,
    rng: np.random.Generator | None = None,
    segments: Sequence[int] | None = None,
) -> bytes:
    """Quantize little-endian f32 values to linear symmetric int4 (syncer ``decode_i4``).

    Per block of ``block`` values (the last may be shorter): f32 scale =
    absmax / 7, then ``q = round(x / scale)`` in [-7, 7] as two's-complement
    nibbles, low nibble first; an odd last block pads a zero nibble. Nearest
    rounding errs by at most absmax / 14 per value. With a numpy ``rng`` the
    rounding is stochastic (unbiased, error below one scale step). Blocks must
    start at multiples of ``block`` values; ``header`` prepends the u32 block.
    With ``segments`` (tensor sizes) the blocks restart at every segment.
    """
    import numpy as np  # noqa: PLC0415 - keep the protocol import light

    i4_nbytes(0, block)  # validates the block size
    if segments is not None:
        head = struct.pack("<I", block) if header else b""
        return head + _encode_segments(data, segments, lambda c: f32_bytes_to_i4(c, block, header=False, rng=rng))
    if rng is None:
        # Nearest rounding: the cache-blocked encoder computes these exact bytes.
        from .push_codec import encode_i4_nearest  # noqa: PLC0415 - keep the protocol import light

        head = struct.pack("<I", block) if header else b""
        return head + encode_i4_nearest(np.frombuffer(data, dtype="<f4"), block)
    values = np.frombuffer(data, dtype="<f4")
    if not np.isfinite(values).all():
        raise ValueError("int4 push delta contains a non-finite value")
    count = values.size
    full = count - count % block
    parts = [struct.pack("<I", block)] if header else []

    def encode(chunk: np.ndarray) -> None:  # chunk: (blocks, width)
        absmax = np.abs(chunk).max(axis=1)
        scale = (absmax / np.float32(7)).astype(np.float32)
        safe = np.where(scale > 0, scale, np.float32(1))
        scaled = chunk / safe[:, None]
        if rng is None:
            rounded = np.rint(scaled)
        else:
            rounded = np.floor(scaled + rng.random(scaled.shape, dtype=np.float32))
        q = np.clip(rounded, -7, 7).astype(np.int8)
        q[scale == 0] = 0
        nibbles = q.view(np.uint8) & 0x0F
        if nibbles.shape[1] % 2:
            nibbles = np.concatenate([nibbles, np.zeros((nibbles.shape[0], 1), np.uint8)], axis=1)
        rows = np.empty((chunk.shape[0], 4 + nibbles.shape[1] // 2), dtype=np.uint8)
        rows[:, :4] = scale.astype("<f4").view(np.uint8).reshape(-1, 4)
        rows[:, 4:] = nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)
        parts.append(rows.tobytes())

    if full:
        encode(values[:full].reshape(-1, block))
    if count > full:
        encode(values[full:].reshape(1, -1))
    return b"".join(parts)


def i4_bytes_to_f32(
    data: bytes | bytearray | memoryview, numel: int, segments: Sequence[int] | None = None
) -> np.ndarray:
    """Decode a linear int4 push (with its u32 block header) into f32 values.

    ``segments`` are the tensor sizes of a tensor-block push.
    """
    import numpy as np  # noqa: PLC0415 - keep the protocol import light

    if segments is not None:
        if len(data) < 4:
            raise ValueError("int4 payload is shorter than its block header")
        block = struct.unpack_from("<I", data)[0]
        view, out, at = memoryview(data), [], 4
        for n in segments:
            size = i4_nbytes(n, block) - 4
            out.append(i4_bytes_to_f32(struct.pack("<I", block) + bytes(view[at : at + size]), n))
            at += size
        if at != len(data):
            raise ValueError(f"int4 payload has {len(data)} bytes, expected {at}")
        return np.concatenate(out) if out else np.empty(0, np.float32)

    if len(data) < 4:
        raise ValueError("int4 payload is shorter than its block header")
    block = struct.unpack_from("<I", data)[0]
    if len(data) != i4_nbytes(numel, block):
        raise ValueError(f"int4 payload has {len(data)} bytes, expected {i4_nbytes(numel, block)}")
    raw = np.frombuffer(data, np.uint8)[4:]

    def decode(rows: np.ndarray, width: int) -> np.ndarray:  # rows: (blocks, 4 + bytes)
        scale = rows[:, :4].copy().view("<f4")
        nibbles = np.empty((rows.shape[0], 2 * (rows.shape[1] - 4)), np.int8)
        nibbles[:, 0::2] = rows[:, 4:] & 0x0F
        nibbles[:, 1::2] = rows[:, 4:] >> 4
        q = np.where(nibbles > 7, nibbles - 16, nibbles)[:, :width]
        return (q.astype(np.float32) * scale).reshape(-1)

    full = numel // block
    row = 4 + block // 2
    parts = [decode(raw[:full * row].reshape(full, row), block)] if full else []
    if numel > full * block:
        parts.append(decode(raw[full * row:].reshape(1, -1), numel - full * block))
    return np.concatenate(parts) if parts else np.empty(0, np.float32)


def _f32_parts_to_i4(
    parts: Iterable[bytes | bytearray | memoryview],
    block: int,
    rng: np.random.Generator | None = None,
    segments: Sequence[int] | None = None,
) -> Iterator[bytes]:
    """Stream ``f32_bytes_to_i4`` over parts; blocks may span part boundaries."""
    yield struct.pack("<I", block)
    for chunk in _block_aligned_chunks(parts, block, segments):
        yield f32_bytes_to_i4(chunk, block, header=False, rng=rng)


def f32_bytes_to_q4(data: bytes | bytearray | memoryview, *, segments: Sequence[int] | None = None) -> bytes:
    """Quantize little-endian f32 values to the q4 wire codec (syncer ``decode_q4``).

    Per block of ``Q4_BLOCK`` values (the last is zero-padded): f32 absmax
    scale, then 128 bytes of nibbles, low nibble first. A nibble is a sign
    bit (bit 3) and a level L: 0 is exact zero, 1..7 decodes to
    ``2**(L-7) * scale``. L is the nearest level in log2 space, so values
    below ``scale * 2**-6.5`` become zero and others err by at most a factor
    of ``sqrt(2)``. Shared by SFT dtype 3 and full-parameter dtype 6. With
    ``segments`` (tensor sizes) the blocks restart, and pad, at every segment.
    """
    import numpy as np  # noqa: PLC0415 - keep the protocol import light

    if segments is not None:
        return _encode_segments(data, segments, f32_bytes_to_q4)

    values = np.frombuffer(data, dtype="<f4")
    if not np.isfinite(values).all():
        raise ValueError("q4 push delta contains a non-finite value")
    blocks = -(-values.size // Q4_BLOCK)
    padded = np.zeros(blocks * Q4_BLOCK, dtype=np.float32)
    padded[: values.size] = values
    padded = padded.reshape(blocks, Q4_BLOCK)
    magnitude = np.abs(padded)
    scale = magnitude.max(axis=1, initial=np.float32(0))
    safe = np.where(scale > 0, scale, np.float32(1))
    normalized = magnitude / safe[:, None]
    with np.errstate(divide="ignore"):
        level = np.where(
            normalized > 0, np.clip(np.rint(np.log2(normalized)) + 7, 0, 7), 0
        ).astype(np.uint8)
    nibbles = (((padded < 0) & (level > 0)).astype(np.uint8) << 3) | level
    rows = np.empty((blocks, 4 + Q4_BLOCK // 2), dtype=np.uint8)
    rows[:, :4] = scale.astype("<f4").view(np.uint8).reshape(-1, 4)
    rows[:, 4:] = nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)
    return rows.tobytes()


def _f32_parts_to_q4(
    parts: Iterable[bytes | bytearray | memoryview], segments: Sequence[int] | None = None
) -> Iterator[bytes]:
    """Stream ``f32_bytes_to_q4`` over parts; only a segment's final block is padded."""
    for chunk in _block_aligned_chunks(parts, Q4_BLOCK, segments):
        yield f32_bytes_to_q4(chunk)


def f32_bytes_to_bf16(data: bytes | bytearray | memoryview) -> bytes:
    """Round little-endian f32 values to bf16 (nearest, ties to even)."""
    import numpy as np  # noqa: PLC0415 - keep the protocol import light

    bits = np.frombuffer(data, dtype="<u4").astype(np.uint64)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    nan = (bits & 0x7FFF_FFFF) > 0x7F80_0000
    rounded[nan] = (bits[nan] >> 16) | 0x40  # keep NaN quiet, never round to inf
    return rounded.astype("<u2").tobytes()


def _f32_parts_to_bf16(parts: Iterable[bytes | bytearray | memoryview]) -> Iterator[bytes]:
    """Stream ``f32_bytes_to_bf16`` over parts whose boundaries may split a value."""
    carry = b""
    for part in parts:
        data = carry + bytes(part)
        whole = len(data) - len(data) % 4
        carry = data[whole:]
        if whole:
            yield f32_bytes_to_bf16(data[:whole])
    if carry:
        raise ValueError("f32 push delta ends mid-value")


def layout_fingerprint(layout: FragmentLayout) -> bytes:
    """Canonical semantic identity for tensor names, order, shapes, and modes."""
    digest = hashlib.sha256()
    digest.update(b"yeto-layout-v1\0")
    digest.update(struct.pack("<I", layout.num_fragments))
    for frag in layout.fragments:
        digest.update(struct.pack("<BI", frag.merge_mode, len(frag.tensors)))
        for name, numel in frag.tensors:
            encoded_name = name.encode("utf-8")
            digest.update(struct.pack("<I", len(encoded_name)))
            digest.update(encoded_name)
            digest.update(struct.pack("<Q", numel))
            if frag.identity_shapes is not None:
                shape = frag.identity_shapes[name]
            elif frag.shapes is not None and name in frag.shapes:
                shape = frag.shapes[name]
            else:
                shape = (numel,)
            digest.update(struct.pack("<I", len(shape)))
            digest.update(struct.pack(f"<{len(shape)}Q", *shape))
    return digest.digest()


def write_frame(sock: socket.socket, msg_type: int, payload: bytes) -> None:
    sock.sendall(_HEADER.pack(MAGIC, msg_type, len(payload)) + payload)


def _read_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        r = sock.recv_into(view[got:], n - got)
        if r == 0:
            raise ConnectionError("syncer closed connection")
        got += r
    return bytes(buf)


def read_frame(
    sock: socket.socket,
    max_payload: int | Callable[[int], int] = MAX_HANDSHAKE_FRAME,
) -> tuple[int, bytes]:
    """Read one frame after checking its type-aware limit before allocation."""
    magic, msg_type, length = _HEADER.unpack(_read_exact(sock, _HEADER.size))
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic:#x}")
    limit = max_payload(msg_type) if callable(max_payload) else max_payload
    if length > limit:
        raise ValueError(
            f"frame type {msg_type} has length {length}, exceeds limit {limit}"
        )
    return msg_type, _read_exact(sock, length)


HELLO_MAC_DOMAIN = b"yeto-hello-mac-v1\0"
ISLAND_HMAC_KEY_ENV = "YETO_ISLAND_HMAC_KEY"

# Test-only (S27 G-EY): a client started with ``recover=True`` reads its data
# streams at most this many bytes per second (all streams together) until its
# initial cut has arrived (``mark_initial_cut_received``: the recovery replay
# and the versions RECOVERY_DONE names are applied). It puts a link slower
# than the merges under a recovery replay on a GPU run (``--modal-env``).
# Unset or not positive: no limit.
RECOVERY_READ_BPS_ENV = "YETO_SYNC_TEST_RECOVERY_READ_BPS"


def recovery_read_bps_from_env(environ=None) -> float | None:
    import os

    value = (os.environ if environ is None else environ).get(RECOVERY_READ_BPS_ENV, "").strip()
    if not value:
        return None
    bps = float(value)
    return bps if bps > 0 else None


class _ReadRate:
    """Bytes-per-second budget shared by a client's data streams."""

    CHUNK = 64 * 1024

    def __init__(self, bps: float, active: Callable[[], bool]) -> None:
        self.bps = bps
        self.active = active
        self._lock = threading.Lock()
        self._next = time.monotonic()

    def spend(self, nbytes: int) -> None:
        """Wait until reading ``nbytes`` more stays within the budget."""
        with self._lock:
            now = time.monotonic()
            self._next = max(self._next, now) + nbytes / self.bps
            delay = self._next - now
        time.sleep(delay)


# Test-only (S27 G-FE): a slow downlink while a learner runs normally (it is
# not recovering). Value "LEARNER:BPS:FROM_STEP:SECONDS": the client of
# learner LEARNER reads its data streams at most BPS bytes per second (all
# streams together) for SECONDS seconds, from the first PULL it gets for a
# syncer step >= FROM_STEP. Other learners read at full speed. The control
# stream (PULLs) is never limited. Unset or empty: no limit.
DOWNLINK_READ_LIMIT_ENV = "YETO_SYNC_TEST_DOWNLINK_READ_LIMIT"


class _DownlinkWindow:
    """The time window of the test-only downlink limit (DOWNLINK_READ_LIMIT_ENV)."""

    def __init__(self, learner_id: int, bps: float, from_step: int, seconds: float) -> None:
        if not (bps > 0 and seconds > 0 and from_step >= 0):
            raise ValueError(f"{DOWNLINK_READ_LIMIT_ENV}: BPS and SECONDS must be positive, FROM_STEP >= 0")
        self.learner_id = learner_id
        self.bps = bps
        self.from_step = from_step
        self.seconds = seconds
        self._lock = threading.Lock()
        self._started: float | None = None
        self._ended = False

    def see_pull(self, step: int) -> None:
        """Start the window at the first PULL for a step >= FROM_STEP."""
        if self._started is not None or step < self.from_step:
            return
        with self._lock:
            if self._started is not None:
                return
            self._started = time.monotonic()
        _connect_log(f"test downlink limit on: {self.bps:.0f} B/s for {self.seconds:.0f}s, "
                     f"PULL step {step} (learner {self.learner_id})")

    def active(self) -> bool:
        started = self._started
        if started is None or self._ended:
            return False
        if time.monotonic() - started < self.seconds:
            return True
        with self._lock:
            if self._ended:
                return False
            self._ended = True
        _connect_log(f"test downlink limit off after {self.seconds:.0f}s (learner {self.learner_id})")
        return False


def downlink_read_limit_from_env(learner_id: int, environ=None) -> _DownlinkWindow | None:
    """The test-only downlink window of ``learner_id``, or None (no limit)."""
    import os

    value = (os.environ if environ is None else environ).get(DOWNLINK_READ_LIMIT_ENV, "").strip()
    if not value:
        return None
    parts = value.split(":")
    if len(parts) != 4:
        raise ValueError(f"{DOWNLINK_READ_LIMIT_ENV} must be LEARNER:BPS:FROM_STEP:SECONDS, got {value!r}")
    if int(parts[0]) != learner_id:
        return None
    return _DownlinkWindow(learner_id, float(parts[1]), int(parts[2]), float(parts[3]))


class _RateLimitedSocket:
    """A data socket read through a :class:`_ReadRate` while it is active."""

    def __init__(self, sock: socket.socket, rate: _ReadRate) -> None:
        self._sock = sock
        self._rate = rate

    def recv_into(self, buffer, nbytes: int = 0, flags: int = 0) -> int:
        if not self._rate.active():
            return self._sock.recv_into(buffer, nbytes, flags)
        got = self._sock.recv_into(buffer, min(nbytes or len(buffer), _ReadRate.CHUNK), flags)
        self._rate.spend(got)
        return got

    def __getattr__(self, name: str) -> Any:
        return getattr(self._sock, name)


def island_hmac_key_from_env(environ=None) -> bytes | None:
    """The island HMAC key (secret-handling-hardening), or None when unset."""
    import os

    value = (os.environ if environ is None else environ).get(ISLAND_HMAC_KEY_ENV, "")
    return value.encode() if value else None


def seal_hello(body: bytes, key: bytes | None, domain: bytes = HELLO_MAC_DOMAIN) -> bytes:
    """Append HMAC-SHA256(key, domain || body); no-op without a key.
    The legacy syncer with a key refuses a HELLO without this trailer.
    ``domain``: HELLO_MAC_DOMAIN, or another first-frame type's domain
    (SUBSCRIBE, RELAY_HELLO)."""
    if not key:
        return body
    import hashlib
    import hmac

    return body + hmac.new(key, domain + body, hashlib.sha256).digest()


def encode_hello(
    learner_id: int,
    dtype: int,
    layout: FragmentLayout,
    num_streams: int,
    connection_generation: int = 0,
    session_contract_hash: bytes | None = None,
    syncer_profile_hash: bytes | None = None,
    compat_group: str | None = None,
    syncer_epoch: int | None = None,
    recover: bool = False,
    expert_tokens: bool = False,
) -> bytes:
    """HELLO body. ``syncer_epoch``, ``recover`` and ``expert_tokens`` add the
    extension trailer (``yeto.recovery_protocol``); with none of them the
    bytes are the legacy ones."""
    if session_contract_hash is None:
        # Generic/legacy callers still bind the session to the semantic tensor
        # layout. Restricted profiles may supply a stronger opaque contract.
        session_contract_hash = layout_fingerprint(layout)
    if not isinstance(session_contract_hash, bytes) or len(session_contract_hash) != 32:
        raise ValueError("session contract hash must be exactly 32 bytes")
    if syncer_profile_hash is not None and (
        not isinstance(syncer_profile_hash, bytes)
        or len(syncer_profile_hash) != 32
    ):
        raise ValueError("syncer profile hash must be exactly 32 bytes")
    parts = [
        _HELLO_HEAD.pack(
            PROTOCOL_VERSION,
            learner_id,
            connection_generation,
            dtype,
            layout.num_fragments,
        )
    ]
    for frag in layout.fragments:
        parts.append(struct.pack("<BI", frag.merge_mode, len(frag.tensors)))
        parts.append(struct.pack(f"<{len(frag.tensors)}Q", *(n for _, n in frag.tensors)))
        if frag.merge_mode == MERGE_ISO:
            # Iso fragments append (rows, cols) per tensor so the syncer can
            # take the 2D view; avg/RDA keep the original wire format.
            dims = [d for name, _ in frag.tensors for d in frag.shapes[name]]
            parts.append(struct.pack(f"<{len(dims)}Q", *dims))
    parts.append(layout_fingerprint(layout))
    parts.append(session_contract_hash)
    if syncer_profile_hash is not None:
        parts.append(syncer_profile_hash)
    parts.append(struct.pack("<H", num_streams))
    if compat_group:
        # yeto-framework-decoupling 7.7c: optional trailer b"YCG1" | u32 len |
        # utf-8 ("nvidia-h100"); the syncer refuses another group by name.
        raw = compat_group.encode("utf-8")
        if not 0 < len(raw) <= 64:
            raise ValueError("compat_group must be 1..64 bytes")
        parts.append(HELLO_COMPAT_MAGIC + struct.pack("<I", len(raw)) + raw)
    parts.append(rp.encode_hello_ext(syncer_epoch, recover, expert_tokens))
    return b"".join(parts)


# One routed-token tag per fragment tensor (syncer protocol.rs
# decode_expert_tags): routed u64 | layer_routed u64 | layer_experts u32.
_EXPERT_TAG = struct.Struct("<QQI")
DENSE_TENSOR_TAG = (0, 0, 0)


def encode_expert_tags(tags: Sequence[tuple[int, int, int]], tensors: int) -> bytes:
    """The routed-token tag block of one PUSH_FRAGMENT.

    ``tags[i]`` belongs to fragment tensor ``i``: ``DENSE_TENSOR_TAG`` for a
    dense tensor, else ``(tokens routed to this expert, tokens routed in its
    layer, experts in its layer)`` over the outer interval of the push. The
    syncer scales the island's weight for that tensor by
    ``routed * layer_experts / layer_routed`` (0 when the layer saw no token).
    """
    if len(tags) != tensors:
        raise ValueError(f"expert tags cover {len(tags)} tensors, fragment has {tensors}")
    parts = [struct.pack("<I", tensors)]
    for index, (routed, layer_routed, experts) in enumerate(tags):
        if experts == 0 and (routed or layer_routed):
            raise ValueError(f"dense tensor {index} carries routed tokens")
        if not 0 <= routed <= layer_routed:
            raise ValueError(f"tensor {index}: routed {routed} of {layer_routed} layer tokens")
        parts.append(_EXPERT_TAG.pack(routed, layer_routed, experts))
    return b"".join(parts)


def encode_data_hello(
    learner_id: int, connection_generation: int, stream_index: int
) -> bytes:
    return _DATA_HELLO.pack(
        PROTOCOL_VERSION, learner_id, connection_generation, stream_index
    )


def _tensor_nbytes(dtype: int, numel: int) -> int:
    width = {DTYPE_F32: 4, DTYPE_BF16: 2}.get(dtype)
    if width is None:
        raise ValueError(f"unknown tensor dtype {dtype}")
    return numel * width


def _q4_nbytes(numel: int, segments: Sequence[int] | None = None) -> int:
    return sum(-(-n // Q4_BLOCK) * (4 + Q4_BLOCK // 2) for n in (segments if segments is not None else (numel,)))


def _contiguous_bytes_view(
    part: bytes | bytearray | memoryview,
    index: int,
    label: str,
) -> memoryview:
    try:
        view = memoryview(part)
    except TypeError as exc:
        raise TypeError(f"{label} part {index} is not bytes-like") from exc
    if not view.c_contiguous:
        view.release()
        raise ValueError(f"{label} part {index} is not C-contiguous")
    try:
        return view.cast("B")
    except (TypeError, ValueError) as exc:
        view.release()
        raise ValueError(
            f"{label} part {index} cannot be viewed as contiguous bytes"
        ) from exc


class ProtocolError(RuntimeError):
    """The peer rejected this client as wire- or session-incompatible."""


# MSG_ERROR prefixes the syncer uses when it refuses one HELLO whose contract
# differs from the running session (server.rs admit_session / profile check).
# The syncer and the admitted islands keep running; only this island stops.
SESSION_REJECT_PREFIXES = (
    ("session mismatch", "layout_hash_mismatch"),
    ("HELLO syncer semantic profile", "syncer_profile_mismatch"),
    ("HELLO is missing the required syncer semantic profile", "syncer_profile_mismatch"),
    # secret-handling-hardening: wrong or missing island HMAC key.
    ("HELLO authentication failed", "island_auth_failed"),
    # The island was launched for an older syncer incarnation.
    (rp.SYNCER_EPOCH_FENCE_PREFIX, "syncer_epoch_fenced"),
    # Local init: the islands did not load the same base weights.
    ("init digest mismatch", "init_digest_mismatch"),
)


class LeaseExpiredError(ConnectionError):
    """The syncer expired this island's lease. The client redials with a new
    connection generation (a re-JOIN); the syncer catches it up."""


class SessionRejectedError(ProtocolError):
    """The syncer refused this island's HELLO: its session contract (layout,
    backend identity, profile) differs from the running session.  Deterministic
    configuration error -- never retried; surfaced as a strict RL failure."""

    def __init__(self, metric: str, message: str) -> None:
        super().__init__(message)
        self.metric = metric


class PartialMessageGenerationLost(RuntimeError):
    """A chunked outbound message was split by connection-generation loss.

    This is a transient operation failure, not a payload-validation failure.
    The partially queued logical message cannot be completed or reused on its
    original connection generation.  A request/response caller must abandon
    that attempt and answer only a request replayed after reconnect.
    """

    def __init__(
        self,
        *,
        connection_generation: int,
        connection_epoch: int,
        queued_chunks: int,
        operation: str,
    ) -> None:
        super().__init__(
            f"connection generation {connection_generation} "
            f"(client epoch {connection_epoch}) dropped after "
            f"{queued_chunks} {operation} chunks were queued"
        )
        self.connection_generation = connection_generation
        self.connection_epoch = connection_epoch
        self.queued_chunks = queued_chunks
        self.operation = operation


def encode_final_manifest(global_step: int, versions: tuple[int, ...] | list[int]) -> bytes:
    """Encode a versioned description of the authoritative terminal cut."""
    return _FINAL_MANIFEST_HEAD.pack(
        FINALIZATION_REVISION, global_step, len(versions)
    ) + struct.pack(f"<{len(versions)}Q", *versions)


def decode_final_manifest(payload: bytes, expected_fragments: int) -> "FinalManifest":
    if len(payload) < _FINAL_MANIFEST_HEAD.size:
        raise ValueError("truncated FINAL_MANIFEST header")
    revision, global_step, count = _FINAL_MANIFEST_HEAD.unpack_from(payload)
    if revision != FINALIZATION_REVISION:
        raise ValueError(
            f"unsupported finalization revision {revision}; "
            f"expected {FINALIZATION_REVISION}"
        )
    if count != expected_fragments:
        raise ValueError(
            f"FINAL_MANIFEST has {count} fragments, expected {expected_fragments}"
        )
    expected_size = _FINAL_MANIFEST_HEAD.size + count * 8
    if len(payload) != expected_size:
        raise ValueError(
            f"FINAL_MANIFEST has {len(payload)} bytes, expected {expected_size}"
        )
    versions = struct.unpack_from(f"<{count}Q", payload, _FINAL_MANIFEST_HEAD.size)
    return FinalManifest(global_step, tuple(versions))


@dataclass
class PullRequest:
    fragment_id: int
    global_step: int
    round_attempt: int
    received_at: float = field(default_factory=time.monotonic, compare=False)


@dataclass
class BcastFragment:
    fragment_id: int
    version: int
    data: object  # raw tensor buffer, or a typed object-store reference bundle
    received_at: float = field(default_factory=time.monotonic, compare=False)
    payload_hash: bytes | None = field(default=None, compare=False)
    stored: bool = field(default=False, compare=False)
    discard: Callable[[], None] | None = field(default=None, repr=False, compare=False)


@dataclass
class FinalFragment:
    fragment_id: int
    version: int
    data: object  # raw f32 tensor buffer, or a typed object-store reference bundle
    payload_hash: bytes | None = field(default=None, compare=False)
    stored: bool = field(default=False, compare=False)
    discard: Callable[[], None] | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class FinalManifest:
    global_step: int
    versions: tuple[int, ...]


@dataclass
class StreamedInboundPayload:
    """One validated tensor payload committed directly into a typed sink."""

    data: object
    payload_hash: bytes
    discard: Callable[[], None] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.payload_hash, bytes) or len(self.payload_hash) != 32:
            raise ValueError("streamed inbound payload hash must be a SHA256")


@dataclass
class _StagedInboundChunk:
    offset: int
    nbytes: int
    token: object | None


@dataclass
class _StreamedReassembly:
    total: int
    sink: object
    transaction: object
    filled: int = 0
    ranges: list[tuple[int, int]] = field(default_factory=list)
    pending: dict[int, _StagedInboundChunk] = field(default_factory=dict)
    next_payload_offset: int = 0
    msg_type: int | None = None
    fragment_id: int | None = None
    version: int | None = None
    payload_bytes: int | None = None


@dataclass(frozen=True)
class _StreamedInboundFragment:
    msg_type: int
    fragment_id: int
    version: int
    payload: StreamedInboundPayload


@dataclass(frozen=True)
class SendProgress:
    """Outbound progress of one SyncerClient, for upload diagnostics.

    ``bytes_sent`` counts bytes handed to the kernel by ``sendall`` across
    every connection group; ``queued_chunks`` is what the live group's send
    queues still hold; ``reconnects`` counts redial attempts."""

    bytes_sent: int
    queued_chunks: int
    reconnects: int
    connected: bool
    stalls: int = 0


@dataclass
class _Outbound:
    data: bytes | bytearray
    sent: threading.Event | None = None


def _connect_log(message: str) -> None:
    """Island-side connection progress on stderr (lands in the island log)."""
    import sys

    print(f"[yeto-syncer-client] {message}", file=sys.stderr, flush=True)


@dataclass(frozen=True)
class EncodedPush:
    """Wire parts of one PUSH_FRAGMENT payload (``SyncerClient.encode_push_parts``)."""

    fragment_id: int
    push_base: int
    nbytes: int
    parts: tuple[bytes, ...]


def timed_parts(parts: Iterable[Any], timings: dict[str, float], key: str) -> Iterator[Any]:
    """Yield ``parts`` unchanged; add the seconds spent producing each one
    (inside the upstream iterator) to ``timings[key]``."""
    iterator = iter(parts)
    timings.setdefault(key, 0.0)
    while True:
        started = time.monotonic()
        try:
            part = next(iterator)
        except StopIteration:
            timings[key] += time.monotonic() - started
            return
        timings[key] += time.monotonic() - started
        yield part


class SyncerClient:
    """Non-blocking striped syncer connection owned by one learner process.

    Self-healing: any socket error tears down and redials the whole group.
    ``max_reconnects`` bounds the number of redial attempts (None = retry
    forever); once exhausted, check_health raises.
    """

    # push_fragment_parts takes ``timings`` (S27 G-AG push time split).
    push_timings_supported = True
    # First frame and its MAC domain; a receive-only subscriber
    # (yeto.rl.spot_rollout.subscriber) replaces both.
    _hello_msg_type: int = MSG_HELLO
    _hello_mac_domain: bytes = HELLO_MAC_DOMAIN

    def _hello_body_suffix(self) -> bytes:
        """Bytes after the HELLO body, inside the MAC (a subscriber's member
        incarnation trailer); empty for a learner, whose HELLO is unchanged."""
        return b""

    # Transport defaults (S26 E3b); __init__ overrides them per client.
    stream_schedule: str = "rr"
    stream_stall_timeout: float | None = None

    def __init__(
        self,
        addr: tuple[str, int],
        learner_id: int,
        layout: FragmentLayout,
        dtype: int = DTYPE_BF16,
        num_streams: int = 4,
        connect_timeout: float = 900.0,
        max_reconnects: int | None = None,
        finalization_timeout: float = FINALIZATION_TIMEOUT,
        session_contract_hash: bytes | None = None,
        syncer_profile_hash: bytes | None = None,
        hmac_key: bytes | None = None,
        compat_group: str | None = None,
        push_i8_block: int = DEFAULT_I8_BLOCK,
        push_i4_block: int = DEFAULT_I4_BLOCK,
        push_i4_stochastic_seed: int | None = None,
        stream_schedule: str = "rr",
        stream_stall_timeout: float | None = None,
        syncer_epoch: int | None = None,
        recover: bool = False,
        owned_fragments: range | None = None,
        expert_tokens: bool = False,
    ):
        """``syncer_epoch``: the syncer incarnation this island was launched
        for; a syncer with a newer epoch refuses it (not retried). ``recover``:
        this island lost its trainer state; every HELLO asks for peer recovery
        until the syncer sends RECOVERY_DONE (see ``yeto.rl.peer_recovery``).
        ``expert_tokens``: every push carries per-expert routed-token tags
        (``encode_expert_tags``) for a syncer run with expert token weights."""
        if stream_schedule not in STREAM_SCHEDULES:
            raise ValueError(f"stream_schedule must be one of {STREAM_SCHEDULES}, got {stream_schedule!r}")
        if stream_stall_timeout is not None and not stream_stall_timeout > 0:
            raise ValueError("stream_stall_timeout must be positive or None")
        if not 0 <= learner_id <= 0xFFFF_FFFF:
            raise ValueError(f"learner_id must fit u32, got {learner_id}")
        if not valid_session_dtype(dtype):
            raise ValueError(f"unsupported session dtype {dtype}")
        i8_nbytes(0, push_i8_block)  # validates the int8 block size
        i4_nbytes(0, push_i4_block)  # validates the int4 block size
        if layout.num_fragments < 1:
            raise ValueError("layout must contain at least one fragment")
        if not 0 <= num_streams <= 256:
            raise ValueError(f"num_streams must be in [0, 256], got {num_streams}")
        if session_contract_hash is None:
            session_contract_hash = layout_fingerprint(layout)
        if not isinstance(session_contract_hash, bytes) or len(session_contract_hash) != 32:
            raise ValueError("session contract hash must be exactly 32 bytes")
        if syncer_profile_hash is not None and (
            not isinstance(syncer_profile_hash, bytes)
            or len(syncer_profile_hash) != 32
        ):
            raise ValueError("syncer profile hash must be exactly 32 bytes")
        if owned_fragments is None:
            owned_fragments = range(layout.num_fragments)
        elif (
            owned_fragments.step != 1
            or not 0 <= owned_fragments.start < owned_fragments.stop <= layout.num_fragments
        ):
            raise ValueError(
                f"owned_fragments {owned_fragments} is not a non-empty slice of "
                f"{layout.num_fragments} fragments"
            )
        self.addr = addr
        self.learner_id = learner_id
        self.layout = layout
        # S27 P4b: the fragments this connection's syncer shard owns (all of
        # them unsharded). INIT and PUSH for other fragments are refused, and
        # finalization waits only for these (yeto/syncer_shards.py).
        self.owned_fragments = owned_fragments
        self.dtype = dtype
        self._push_base = session_base(dtype)
        self.push_i8_block = push_i8_block
        self.push_i4_block = push_i4_block
        # Stochastic int4 rounding (off by default): one seeded stream per client.
        self._push_i4_rng = None
        if push_i4_stochastic_seed is not None:
            import numpy as np  # noqa: PLC0415 - keep the protocol import light

            self._push_i4_rng = np.random.default_rng(push_i4_stochastic_seed)
        # S26 E3: f32 anchors of a bf16-downlink session (yeto/downlink.py).
        self._downlink = DownlinkAnchors(layout.num_fragments) if bcast_delta(dtype) else None
        self._bcast_prefix = 12 + (DOWNLINK_HEADER_LEN if self._downlink is not None else 0)
        self.num_streams = num_streams
        # S26 E3b: chunk placement over the data streams and the per-stream
        # no-progress timeout (None = off, the historical behaviour).
        self.stream_schedule = stream_schedule
        self.stream_stall_timeout = stream_stall_timeout
        # Index 0 is the control socket; the meters outlive connection groups
        # so their counters are cumulative for the whole session.
        self._meters = [StreamMeter() for _ in range(num_streams + 1)]
        self._upload_drained_at: float | None = None
        self._stalls = 0
        # S26 E3b: set once the initial cut arrived. Before that a redial must
        # replay INIT_PARAMS (learner 0), and TCP_USER_TIMEOUT stays unset:
        # the multi-GB INIT upload and initial broadcast must not be reset by
        # the kernel on a slow link (s26-e3b-p06-C32a).
        self._initial_cut_received = threading.Event()
        # Test-only (S27 G-EY, RECOVERY_READ_BPS_ENV): a slow link under the
        # recovery replay of this client.
        bps = recovery_read_bps_from_env() if recover else None
        self._recovery_read_rate = (
            None if bps is None else _ReadRate(bps, lambda: not self._initial_cut_received.is_set())
        )
        # Test-only (S27 G-FE, DOWNLINK_READ_LIMIT_ENV): a slow downlink in
        # normal running. A recovery limit of this client takes precedence.
        self._downlink_window = downlink_read_limit_from_env(learner_id)
        self._downlink_read_rate = (
            None if self._downlink_window is None
            else _ReadRate(self._downlink_window.bps, self._downlink_window.active)
        )
        self._init_generation: int | None = None
        # Local init: the INIT_DIGEST frame sent after every HELLO until the
        # initial cut arrives, and the INIT_CONFIRM digest it must earn.
        self._init_digest_frame: bytes | None = None
        self._init_combined: bytes | None = None
        self._init_confirmed = threading.Event()
        self.session_contract_hash = session_contract_hash
        self.syncer_profile_hash = syncer_profile_hash
        # secret-handling-hardening: HELLO carries an HMAC when the island has
        # the key (the launcher ships it as a secret env).
        self.hmac_key = hmac_key if hmac_key is not None else island_hmac_key_from_env()
        self.compat_group = compat_group
        rp.encode_hello_ext(syncer_epoch, False)  # validates the epoch
        self.syncer_epoch = syncer_epoch
        self.expert_tokens = expert_tokens
        # Set until RECOVERY_DONE: a redial during recovery asks again.
        self._recover = threading.Event()
        if recover:
            self._recover.set()
        self._recovery_events: queue.Queue = queue.Queue()
        self._snapshot_requests: queue.Queue = queue.Queue()
        self.lease_expiries = 0
        self.connect_timeout = connect_timeout
        self.max_reconnects = max_reconnects
        self.finalization_timeout = finalization_timeout
        self._max_bcast_payload = max(
            self._bcast_prefix + _tensor_nbytes(bulk_dtype(dtype), fragment.numel)
            for fragment in layout.fragments
        )
        self._max_final_payload = max(
            12 + _tensor_nbytes(DTYPE_F32, fragment.numel)
            for fragment in layout.fragments
        )
        self._max_chunked_inner = _HEADER.size + max(
            self._max_bcast_payload, self._max_final_payload
        )
        self._queues: list[queue.Queue[_Outbound | None]] = []
        self._socks: list[socket.socket] = []
        self._threads: list[threading.Thread] = []
        self._pulls: queue.Queue[PullRequest] = queue.Queue()
        self._bcasts: queue.Queue[BcastFragment] = queue.Queue()
        self._bcast_seen: list[tuple[int, bytes] | None] = [
            None
        ] * layout.num_fragments
        self._bcast_lock = threading.Lock()
        # Raw globals are retained independently of the drain queue so a
        # final fragment that outruns its control-stream manifest can be
        # re-applied exactly after the manifest arrives.
        self._final_fragments: dict[int, FinalFragment] = {}
        self._final_manifest: FinalManifest | None = None
        self._final_ack_step: int | None = None
        self.finalizing = threading.Event()
        self.finalized = threading.Event()
        self._final_cond = threading.Condition()
        self._reasm: dict[
            int,
            tuple[bytearray, list[int], list[tuple[int, int]]]
            | _StreamedReassembly,
        ] = {}
        self._reasm_lock = threading.Lock()
        self._reasm_touched: dict[int, float] = {}  # msg_id -> last chunk (monotonic)
        self._inbound_chunk_sink: object | None = None
        self._msg_id = itertools.count()
        self._rr = itertools.count()
        self._err: BaseException | None = None
        self._last_err: BaseException | None = None
        self.shutdown = threading.Event()
        # Connection-group generation. Bumped on every teardown/redial so
        # threads and messages belonging to a dead group can be recognized
        # and ignored. Guarded by _lock together with _socks/_queues.
        self._lock = threading.RLock()
        self._gen = 0
        self.connection_generation = 0
        self._connected = threading.Event()
        self._failure = threading.Event()  # set by any thread of the live group
        self._closed = threading.Event()
        self._reconnects_used = 0
        self._bytes_sent = 0
        self._bytes_sent_lock = threading.Lock()
        self._reset_bcasts_on_reconnect = False
        self._supervisor: threading.Thread | None = None
        self._watchdog: threading.Thread | None = None

    # -- lifecycle -------------------------------------------------------------

    def _dial(self, timeout: float) -> socket.socket:
        sock = socket.create_connection(self.addr, timeout=timeout)
        sock.settimeout(None)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        enable_tcp_keepalive(sock)
        if self.stream_stall_timeout is not None and self._initial_cut_received.is_set():
            set_tcp_user_timeout(sock, self.stream_stall_timeout)
        return sock

    def _connect_one(self) -> socket.socket:
        last: OSError | None = None
        t0 = time.monotonic()
        attempts = 0
        next_report = 0.0
        _connect_log(f"dialing syncer {self.addr[0]}:{self.addr[1]} "
                     f"(learner {self.learner_id}, timeout {self.connect_timeout:.0f}s)")
        while time.monotonic() - t0 < self.connect_timeout:
            attempts += 1
            try:
                sock = self._dial(30)
                _connect_log(f"connected to syncer {self.addr[0]}:{self.addr[1]} after "
                             f"{attempts} attempt(s), {time.monotonic() - t0:.1f}s")
                return sock
            except OSError as e:  # syncer may not be up yet
                last = e
                waited = time.monotonic() - t0
                if waited >= next_report:  # s13-g3b: a silent 900 s redial looked like a hang
                    _connect_log(f"syncer {self.addr[0]}:{self.addr[1]} not reachable yet "
                                 f"(attempt {attempts}, {waited:.0f}s): {type(e).__name__}: {e}")
                    next_report = waited + 30.0
                time.sleep(2.0)
        raise ConnectionError(f"cannot reach syncer at {self.addr} after {attempts} attempt(s): {last}")

    def start(self) -> None:
        self._connect_group(patient=True)
        self._supervisor = threading.Thread(target=self._supervise, name="yeto-supervisor", daemon=True)
        self._supervisor.start()
        if self.stream_stall_timeout is not None:
            self._watchdog = threading.Thread(target=self._watch_stalls, name="yeto-stall-watchdog", daemon=True)
            self._watchdog.start()

    def _connect_group(self, patient: bool) -> None:
        """Dial control + data sockets, send HELLO/DATA_HELLO, start workers.

        ``patient`` retries each dial for connect_timeout (initial startup,
        where the syncer may not be up yet); otherwise each socket gets a
        single attempt and the supervisor's backoff handles retries.
        """
        socks: list[socket.socket] = []
        connection_generation = secrets.randbits(64)
        while connection_generation == 0 or connection_generation == self.connection_generation:
            connection_generation = secrets.randbits(64)
        try:
            control = self._connect_one() if patient else self._dial(RECONNECT_DIAL_TIMEOUT)
            write_frame(
                control,
                self._hello_msg_type,
                seal_hello(
                    encode_hello(
                        self.learner_id,
                        self.dtype,
                        self.layout,
                        self.num_streams,
                        connection_generation,
                        self.session_contract_hash,
                        self.syncer_profile_hash,
                        self.compat_group,
                        syncer_epoch=self.syncer_epoch,
                        recover=self._recover.is_set(),
                        expert_tokens=self.expert_tokens,
                    ) + self._hello_body_suffix(),
                    self.hmac_key,
                    self._hello_mac_domain,
                ),
            )
            if self._init_digest_frame is not None and not self._initial_cut_received.is_set():
                control.sendall(self._init_digest_frame)
            socks.append(control)
            for idx in range(self.num_streams):
                s = self._connect_one() if patient else self._dial(RECONNECT_DIAL_TIMEOUT)
                write_frame(
                    s,
                    MSG_DATA_HELLO,
                    encode_data_hello(self.learner_id, connection_generation, idx),
                )
                socks.append(s)
        except BaseException:
            for s in socks:
                _close_socket(s)
            raise
        with self._lock:
            self._gen += 1
            gen = self._gen
            self.connection_generation = connection_generation
            self._socks = socks
            self._queues = []
            self._threads = []
            self._failure.clear()
            self._connected.set()
            for i, s in enumerate(socks):
                q: queue.Queue[_Outbound | None] = queue.Queue(maxsize=256)
                self._queues.append(q)
                ts = threading.Thread(
                    target=self._send_loop, args=(gen, s, q, self._meters[i]), name=f"yeto-send-{i}", daemon=True
                )
                tr = threading.Thread(target=self._recv_loop, args=(gen, s), name=f"yeto-recv-{i}", daemon=True)
                self._threads += [ts, tr]
                ts.start()
                tr.start()
        with self._final_cond:
            self._final_cond.notify_all()

    def _teardown_group(self) -> None:
        """Kill the current group's sockets/threads and drop state that a
        fresh HELLO makes obsolete (the syncer rebroadcasts every fragment at
        its current version on reconnect). Received broadcasts are kept."""
        with self._lock:
            self._gen += 1  # invalidates the old group's threads and messages
            self._connected.clear()
            socks, queues, threads = self._socks, self._queues, self._threads
            self._socks, self._queues, self._threads = [], [], []
        for q in queues:
            try:
                q.put_nowait(None)  # wake the sender; socket close covers a full queue
            except queue.Full:
                pass
        for s in socks:
            _close_socket(s)
        for meter in self._meters:
            meter.drop_pending()
        self._clear_partial_reassemblies()
        self._drain(self._pulls)  # stale pull requests; answering them is pointless
        with self._lock:
            reset_bcasts = self._reset_bcasts_on_reconnect
            self._reset_bcasts_on_reconnect = False
        if reset_bcasts:
            self._drain(self._bcasts)
            with self._bcast_lock:
                self._bcast_seen = [None] * self.layout.num_fragments
        for t in threads:
            t.join(timeout=5.0)
        with self._final_cond:
            self._final_cond.notify_all()

    def _supervise(self) -> None:
        """Reconnect loop: on any socket failure of the live group, tear the
        whole group down, back off exponentially, redial, resume."""
        while True:
            self._failure.wait()
            if self._closed.is_set() or self.shutdown.is_set():
                return
            _connect_log(f"connection to syncer {self.addr[0]}:{self.addr[1]} lost "
                         f"(learner {self.learner_id}): {self._last_err!r}; redialing")
            self._teardown_group()
            backoff = RECONNECT_BACKOFF_START
            # A lease expiry is the syncer asking for a re-JOIN, not a link
            # failure: its first redial does not use the reconnect budget.
            rejoin = isinstance(self._last_err, LeaseExpiredError)
            while True:
                if self._closed.is_set() or self.shutdown.is_set():
                    return
                if (
                    not rejoin
                    and self.max_reconnects is not None
                    and self._reconnects_used >= self.max_reconnects
                ):
                    with self._lock:
                        self._err = self._last_err or ConnectionError("syncer connection lost")
                    with self._final_cond:
                        self._final_cond.notify_all()
                    return
                if rejoin:
                    rejoin = False
                else:
                    self._reconnects_used += 1
                if self._closed.wait(backoff):
                    return
                backoff = min(backoff * 2.0, RECONNECT_BACKOFF_CAP)
                try:
                    self._connect_group(patient=False)
                    _connect_log(f"reconnected to syncer {self.addr[0]}:{self.addr[1]} "
                                 f"(redial {self._reconnects_used})")
                    break
                except (OSError, ConnectionError) as e:
                    self._last_err = e
                    _connect_log(f"redial {self._reconnects_used} to syncer failed: "
                                 f"{type(e).__name__}: {e}")

    def close(self) -> None:
        self._closed.set()
        self._failure.set()  # unblock the supervisor so it exits instead of redialing
        with self._lock:
            queues, socks = list(self._queues), list(self._socks)
        for q in queues:
            try:
                q.put_nowait(None)
            except queue.Full:
                pass
        for s in socks:
            _close_socket(s)
        if self._supervisor is not None:
            self._supervisor.join(timeout=5.0)
        self._clear_partial_reassemblies()
        with self._final_cond:
            self._final_cond.notify_all()

    def check_health(self) -> None:
        """Raise only for unrecoverable failures. While a reconnect is being
        attempted this is a no-op; the training loop keeps stepping locally."""
        if isinstance(self._err, SessionRejectedError):
            # Wrong island for this session: a strict failure of this island
            # only (bridge prints [yeto-rl-strict-failure], exit 1), not a
            # network failure to retry and not a syncer crash.
            from .rl.core import StrictRlInvariantError

            raise StrictRlInvariantError(
                self._err.metric, f"syncer refused this island: {self._err}"
            ) from self._err
        if self._err is not None:
            raise RuntimeError(f"syncer connection failed: {self._err}") from self._err

    def send_progress(self) -> SendProgress:
        with self._lock:
            queued = sum(q.qsize() for q in self._queues)
            connected = self._connected.is_set()
        with self._bytes_sent_lock:
            sent = self._bytes_sent
        return SendProgress(
            bytes_sent=sent,
            queued_chunks=queued,
            reconnects=self._reconnects_used,
            connected=connected,
            stalls=self._stalls,
        )

    def stream_snapshots(self) -> tuple[StreamSnapshot, ...]:
        """Cumulative counters of each data stream (control stream excluded;
        with ``num_streams == 0`` the control stream carries the data)."""
        meters = self._meters[1:] if self.num_streams else self._meters[:1]
        return tuple(meter.snapshot() for meter in meters)

    @property
    def upload_drained_at(self) -> float | None:
        """Unix time when the data streams last became idle with nothing
        queued, i.e. when this island's latest upload left the host."""
        return self._upload_drained_at

    # -- learner-facing API ------------------------------------------------------

    def send_initial_fragments(
        self,
        parts_for: Callable[[int], Iterable[bytes | bytearray | memoryview]],
    ) -> bool:
        """Learner 0: stream INIT_PARAMS of every fragment on the live group.

        ``parts_for(fragment_id)`` returns a fresh part iterable each call, so
        the same INIT can be replayed after a redial. Returns True when every
        fragment was queued on one connection generation; False on an outage
        (nothing to undo: the syncer ignores INIT it already applied, and a
        partial INIT of a dead generation is dropped by the syncer)."""
        generation = self.connection_generation
        try:
            for fragment_id in self.owned_fragments:
                if self.send_init_parts(fragment_id, parts_for(fragment_id)) is not True:
                    return False
        except PartialMessageGenerationLost:
            return False
        if generation != self.connection_generation:
            return False
        self._init_generation = generation
        return True

    def _check_owned(self, kind: str, fragment_id: int) -> None:
        if not 0 <= fragment_id < self.layout.num_fragments:
            raise ValueError(f"{kind} for unknown fragment {fragment_id}")
        if fragment_id not in self.owned_fragments:
            raise ValueError(
                f"{kind} for fragment {fragment_id} sent to the syncer shard "
                f"owning fragments {self.owned_fragments.start}..{self.owned_fragments.stop - 1}"
            )

    def init_replay_due(self) -> bool:
        """True when the live group never carried a complete INIT and the
        initial cut has not arrived yet (learner 0 must send INIT again)."""
        with self._lock:
            live = self._connected.is_set()
            generation = self.connection_generation
        return (
            live
            and not self._initial_cut_received.is_set()
            and generation != self._init_generation
        )

    def set_init_digests(self, layout_hash: bytes, digests: Sequence[bytes]) -> None:
        """Local init: report the SHA-256 of every fragment's INIT_PARAMS
        tensor bytes (one per layout fragment). Call before :meth:`start`;
        the digests then follow every HELLO until the initial cut arrives,
        and :meth:`initial_cut_confirmed` turns true when the syncer agrees."""
        if len(layout_hash) != 32 or len(digests) != self.layout.num_fragments:
            raise ValueError("local init needs a 32-byte layout hash and one digest per fragment")
        if bulk_dtype(self.dtype) != DTYPE_F32:
            raise ValueError("local init needs an f32-bulk session")
        owned = [bytes(digests[index]) for index in self.owned_fragments]
        if any(len(digest) != 32 for digest in owned):
            raise ValueError("fragment digests must be 32-byte SHA-256 values")
        if self._connected.is_set():
            raise RuntimeError("set_init_digests must be called before start()")
        payload = (struct.pack("<I", self.learner_id) + bytes(layout_hash)
                   + struct.pack("<II", self.owned_fragments.start, len(owned)) + b"".join(owned))
        self._init_combined = hashlib.sha256(b"".join(owned)).digest()
        self._init_digest_frame = self._frame(MSG_INIT_DIGEST, payload)

    def initial_cut_confirmed(self) -> bool:
        """True once the syncer confirmed this island's local version 0."""
        return self._init_confirmed.is_set()

    def mark_initial_cut_received(self) -> None:
        """The initial cut arrived: INIT is no longer replayed, and the stall
        timeout now also bounds unacknowledged data in the kernel."""
        self._initial_cut_received.set()
        if self.stream_stall_timeout is None:
            return
        with self._lock:
            socks = list(self._socks)
        for sock in socks:
            try:
                set_tcp_user_timeout(sock, self.stream_stall_timeout)
            except OSError:
                pass  # a dying socket; the group is being replaced anyway

    def send_init(self, fragment_id: int, tensor_bytes: bytes) -> None:
        self._check_owned("INIT_PARAMS", fragment_id)
        expected = _tensor_nbytes(
            bulk_dtype(self.dtype), self.layout.fragments[fragment_id].numel
        )
        if len(tensor_bytes) != expected:
            raise ValueError(
                f"INIT_PARAMS fragment {fragment_id} has {len(tensor_bytes)} "
                f"tensor bytes, expected {expected}"
            )
        self._send_large(MSG_INIT_PARAMS, struct.pack("<I", fragment_id) + tensor_bytes)

    def send_init_parts(
        self,
        fragment_id: int,
        tensor_parts: Iterable[bytes | bytearray | memoryview],
    ) -> bool:
        """Stream INIT_PARAMS without materializing its complete tensor."""
        self._check_owned("INIT_PARAMS", fragment_id)
        expected = _tensor_nbytes(
            bulk_dtype(self.dtype), self.layout.fragments[fragment_id].numel
        )
        return self._send_large_parts(
            MSG_INIT_PARAMS,
            struct.pack("<I", fragment_id),
            tensor_parts,
            expected,
            label=f"INIT_PARAMS fragment {fragment_id}",
        )

    def push_segments(self, fragment_id: int) -> list[int] | None:
        """Tensor sizes of a fragment on a tensor-block session, else ``None``."""
        if not tensor_blocks(self.dtype):
            return None
        return [numel for _, numel in self.layout.fragments[fragment_id].tensors]

    def push_fragment(
        self,
        fragment_id: int,
        global_step: int,
        round_attempt: int,
        base_version: int,
        local_step: int,
        c_steps: int,
        c_tokens: int,
        tensor_bytes: bytes,
        *,
        expert_tags: Sequence[tuple[int, int, int]] | None = None,
    ) -> None:
        """``expert_tags``: required exactly when the session sends expert
        token counts (see ``encode_expert_tags``)."""
        self._check_owned("PUSH_FRAGMENT", fragment_id)
        if round_attempt < 1:
            raise ValueError("round_attempt must be positive")
        if c_steps < 1:
            raise ValueError("c_steps must be positive")
        numel = self.layout.fragments[fragment_id].numel
        segments = self.push_segments(fragment_id)
        expected = (
            _q4_nbytes(numel)
            if self._push_base == DTYPE_Q4
            # dtypes 4-7 callers pass f32 bytes; the client encodes them below
            else _tensor_nbytes(bulk_dtype(self.dtype), numel)
        )
        if len(tensor_bytes) != expected:
            raise ValueError(
                f"PUSH_FRAGMENT fragment {fragment_id} has {len(tensor_bytes)} "
                f"delta bytes, expected {expected}"
            )
        head = struct.pack(
            "<IIQIQQIQ",
            self.learner_id,
            fragment_id,
            global_step,
            round_attempt,
            base_version,
            local_step,
            c_steps,
            c_tokens,
        )
        if self._push_base == DTYPE_F32_PUSH_BF16:
            tensor_bytes = f32_bytes_to_bf16(tensor_bytes)
        elif self._push_base == DTYPE_F32_PUSH_I8:
            tensor_bytes = f32_bytes_to_i8(tensor_bytes, self.push_i8_block, segments=segments)
        elif self._push_base == DTYPE_F32_PUSH_Q4:
            tensor_bytes = f32_bytes_to_q4(tensor_bytes, segments=segments)
        elif self._push_base == DTYPE_F32_PUSH_I4:
            tensor_bytes = f32_bytes_to_i4(
                tensor_bytes, self.push_i4_block, rng=self._push_i4_rng, segments=segments
            )
        head += self._expert_tag_block(fragment_id, expert_tags)
        self._send_large(MSG_PUSH_FRAGMENT, head + tensor_bytes)

    def push_fragment_parts(
        self,
        fragment_id: int,
        global_step: int,
        round_attempt: int,
        base_version: int,
        local_step: int,
        c_steps: int,
        c_tokens: int,
        tensor_parts: Iterable[bytes | bytearray | memoryview],
        *,
        before_last_enqueue: Callable[[], None] | None = None,
        expert_tags: Sequence[tuple[int, int, int]] | None = None,
        timings: dict[str, float] | None = None,
        encoded: EncodedPush | None = None,
    ) -> bool:
        """Stream one PUSH_FRAGMENT from bounded contiguous byte parts.

        ``timings`` (S27 G-AG diagnostics): when given, ``produce_s`` gets the
        seconds spent producing the wire parts (the caller's ``tensor_parts``
        plus the wire encoding) and ``push_s`` the whole call; the rest of
        ``push_s`` is framing and enqueueing.  The bytes do not change.

        ``encoded`` (S27 G-AG): wire parts :meth:`encode_push_parts` already
        produced for this fragment (``tensor_parts`` must then be empty); the
        frame is byte-identical to streaming the same deltas.

        The syncer observes the same logical inner frame as ``push_fragment``.
        Only one CHUNK-sized envelope is assembled at a time; neither the
        complete tensor nor the complete inner frame is materialized here.
        Returns ``True`` only after every chunk is queued and ``False`` when
        none could be queued. A failure after the first chunk poisons that
        connection generation and raises so callers cannot mark the push done.
        """
        self._check_owned("PUSH_FRAGMENT", fragment_id)
        if round_attempt < 1:
            raise ValueError("round_attempt must be positive")
        if c_steps < 1:
            raise ValueError("c_steps must be positive")
        if encoded is None:
            expected, tensor_parts = self._push_wire_parts(fragment_id, tensor_parts)
        else:
            if encoded.fragment_id != fragment_id or encoded.push_base != self._push_base:
                raise ValueError("encoded PUSH_FRAGMENT parts belong to another fragment or codec")
            expected, tensor_parts = encoded.nbytes, encoded.parts
        head = struct.pack(
            "<IIQIQQIQ",
            self.learner_id,
            fragment_id,
            global_step,
            round_attempt,
            base_version,
            local_step,
            c_steps,
            c_tokens,
        )
        head += self._expert_tag_block(fragment_id, expert_tags)
        if timings is None:
            return self._send_large_parts(
                MSG_PUSH_FRAGMENT,
                head,
                tensor_parts,
                expected,
                label=f"PUSH_FRAGMENT fragment {fragment_id}",
                before_last_enqueue=before_last_enqueue,
            )
        started = time.monotonic()
        try:
            return self._send_large_parts(
                MSG_PUSH_FRAGMENT,
                head,
                timed_parts(tensor_parts, timings, "produce_s"),
                expected,
                label=f"PUSH_FRAGMENT fragment {fragment_id}",
                before_last_enqueue=before_last_enqueue,
            )
        finally:
            timings["push_s"] = timings.get("push_s", 0.0) + time.monotonic() - started

    def _push_wire_parts(
        self, fragment_id: int, tensor_parts: Iterable[bytes | bytearray | memoryview]
    ) -> tuple[int, Iterable[bytes | bytearray | memoryview]]:
        """The wire size and wire parts of one PUSH_FRAGMENT's f32 deltas."""
        numel = self.layout.fragments[fragment_id].numel
        segments = self.push_segments(fragment_id)
        if self._push_base == DTYPE_Q4:
            expected = _q4_nbytes(numel)
        elif self._push_base == DTYPE_F32_PUSH_I8:
            expected = i8_nbytes(numel, self.push_i8_block, segments)
            tensor_parts = _f32_parts_to_i8(tensor_parts, self.push_i8_block, segments)
        elif self._push_base == DTYPE_F32_PUSH_Q4:
            expected = _q4_nbytes(numel, segments)
            tensor_parts = _f32_parts_to_q4(tensor_parts, segments)
        elif self._push_base == DTYPE_F32_PUSH_I4:
            expected = i4_nbytes(numel, self.push_i4_block, segments)
            tensor_parts = _f32_parts_to_i4(tensor_parts, self.push_i4_block, self._push_i4_rng, segments)
        else:
            expected = _tensor_nbytes(push_wire_dtype(self.dtype), numel)
        if self._push_base == DTYPE_F32_PUSH_BF16:
            tensor_parts = _f32_parts_to_bf16(tensor_parts)
        return expected, tensor_parts

    @property
    def can_encode_ahead(self) -> bool:
        """Whether :meth:`encode_push_parts` may run off the sending thread
        and ahead of the push: a deterministic block codec (int8, q4, int4
        without stochastic rounding) whose output parts are fresh bytes."""
        return self._push_base in (DTYPE_F32_PUSH_I8, DTYPE_F32_PUSH_Q4) or (
            self._push_base == DTYPE_F32_PUSH_I4 and self._push_i4_rng is None)

    def encode_push_parts(
        self,
        fragment_id: int,
        tensor_parts: Iterable[bytes | bytearray | memoryview],
        *,
        timings: dict[str, float] | None = None,
    ) -> EncodedPush:
        """Encode one fragment's f32 deltas to wire parts now (S27 G-AG).

        Pure: reads only the layout and codec settings, so another thread may
        encode the next fragments while this one sends.  ``timings`` gets
        ``produce_s``, as in :meth:`push_fragment_parts`."""
        if not self.can_encode_ahead:
            raise ValueError("this push codec cannot be encoded ahead of its push")
        self._check_owned("PUSH_FRAGMENT", fragment_id)
        expected, parts = self._push_wire_parts(fragment_id, tensor_parts)
        if timings is not None:
            parts = timed_parts(parts, timings, "produce_s")
        materialized = tuple(bytes(part) if not isinstance(part, bytes) else part for part in parts)
        if sum(len(part) for part in materialized) != expected:
            raise ValueError(f"PUSH_FRAGMENT fragment {fragment_id} encoded to the wrong size")
        return EncodedPush(fragment_id, self._push_base, expected, materialized)

    def _expert_tag_block(
        self, fragment_id: int, expert_tags: Sequence[tuple[int, int, int]] | None
    ) -> bytes:
        if (expert_tags is not None) != self.expert_tokens:
            raise ValueError(
                "PUSH_FRAGMENT expert tags are required exactly when the session "
                "sends expert token counts"
            )
        if expert_tags is None:
            return b""
        return encode_expert_tags(expert_tags, len(self.layout.fragments[fragment_id].tensors))

    def heartbeat(self, local_step: int) -> None:
        self._enqueue(0, self._frame(MSG_HEARTBEAT, struct.pack("<IQ", self.learner_id, local_step)))

    def send_leave(self, reason: int, timeout: float = 5.0) -> bool:
        """Tell the syncer why this island leaves; True once the frame was
        written. The caller closes the client afterwards."""
        sent = threading.Event()
        frame = self._frame(MSG_LEAVE, struct.pack("<IB", self.learner_id, int(reason)))
        return self._enqueue(0, frame, sent=sent) and sent.wait(timeout)

    def send_budget_done(
        self, local_steps: int, timeout: float | None = None
    ) -> int:
        """Send the exact frozen learner budget and wait until bytes leave."""
        if not 0 < local_steps <= 0xFFFF_FFFF_FFFF_FFFF:
            raise ValueError("local_steps must be a positive u64")
        timeout = self.finalization_timeout if timeout is None else timeout
        deadline = time.monotonic() + timeout
        payload = _BUDGET_DONE.pack(local_steps)
        with self._lock:
            if not self._connected.is_set():
                raise RuntimeError("cannot send BUDGET_DONE while disconnected")
            gen = self._gen
            self._reset_bcasts_on_reconnect = True
        sent = threading.Event()
        if not self._enqueue(
            0,
            self._frame(MSG_BUDGET_DONE, payload),
            gen=gen,
            sent=sent,
        ):
            raise RuntimeError("connection closed before BUDGET_DONE was queued")
        while not sent.wait(min(0.1, max(0.0, deadline - time.monotonic()))):
            self.check_health()
            with self._lock:
                if gen != self._gen:
                    raise RuntimeError("connection lost while sending BUDGET_DONE")
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"BUDGET_DONE was not sent within {timeout:.1f}s"
                )
        return gen

    def wait_for_budget_restart(
        self, previous_generation: int, timeout: float | None = None
    ) -> None:
        """Wait for the fresh connection group used by consolidation."""
        timeout = self.finalization_timeout if timeout is None else timeout
        deadline = time.monotonic() + timeout
        while True:
            self.check_health()
            with self._lock:
                ready = self._connected.is_set() and self._gen > previous_generation
            if ready:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"budget consolidation syncer did not reconnect within {timeout:.1f}s"
                )
            with self._final_cond:
                self._final_cond.wait(min(remaining, 0.5))

    def drain_pulls(self) -> list[PullRequest]:
        return self._drain(self._pulls)

    def drain_updates(self) -> list[BcastFragment]:
        return self._drain(self._bcasts)

    # -- peer recovery (yeto.rl.peer_recovery drives these) --------------------

    @property
    def recovering(self) -> bool:
        return self._recover.is_set()

    def next_recovery_event(self, timeout: float):
        """Next RecoveryClock / RecoverySource / RecoveryDone, or None."""
        try:
            return self._recovery_events.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain_snapshot_requests(self) -> list[rp.SnapshotRequest]:
        """Donor side: snapshot requests the syncer sent this island."""
        return self._drain(self._snapshot_requests)

    def send_snapshot_ready(self, request: rp.SnapshotRequest, versions, sha256: bytes, locator: str) -> bool:
        payload = rp.encode_snapshot_ready(request, versions, sha256, locator)
        return self._enqueue(0, self._frame(rp.MSG_PEER_SNAPSHOT_READY, payload))

    def send_recovery_restored(self, versions) -> bool:
        """``versions`` held after the restore, or None: resend every fragment."""
        if versions is not None and len(versions) != self.layout.num_fragments:
            raise ValueError("restored versions must list every fragment")
        return self._enqueue(0, self._frame(rp.MSG_RECOVERY_RESTORED, rp.encode_restored(versions)))

    def send_recovery_failed(self, reason: str) -> bool:
        return self._enqueue(0, self._frame(rp.MSG_RECOVERY_FAILED, rp.encode_failed(reason)))

    def install_inbound_chunk_sink(self, sink: object | None) -> None:
        """Select a transactional sink for future chunked fragment receives.

        Switching is allowed only between complete logical messages, so a
        reconnect or policy-boundary sink replacement cannot mix anchors.
        """

        if sink is not None:
            required = (
                "begin_message",
                "bind_fragment",
                "stage_chunk",
                "consume_chunk",
                "consume_staged_chunk",
                "finish_message",
                "abort_message",
            )
            if any(not callable(getattr(sink, name, None)) for name in required):
                raise TypeError(
                    "inbound chunk sink does not implement the transaction API"
                )
        if sink is not None and self._downlink is not None:
            sink = DownlinkChunkSink(
                sink, self._downlink, [f.numel for f in self.layout.fragments], MSG_BCAST_FRAGMENT)
        with self._reasm_lock:
            if self._reasm:
                raise RuntimeError("cannot replace inbound chunk sink mid-message")
            self._inbound_chunk_sink = sink

    def wait_for_final_fragments(
        self, timeout: float | None = None
    ) -> tuple[FinalManifest, list[FinalFragment]]:
        """Wait for a manifest and its exact fragment versions.

        FINAL_MANIFEST travels on control while fragments may be striped over
        data sockets, so either can arrive first. The retained payload cache
        makes both orders equivalent and survives a connection-group redial.
        """
        timeout = self.finalization_timeout if timeout is None else timeout
        deadline = time.monotonic() + timeout
        with self._final_cond:
            while True:
                self.check_health()
                manifest = self._final_manifest
                if manifest is not None:
                    ready = []
                    missing = []
                    for fid, version in enumerate(manifest.versions):
                        if fid not in self.owned_fragments:
                            continue  # another syncer shard's fragment
                        cached = self._final_fragments.get(fid)
                        if cached is None or cached.version != version:
                            latest = None if cached is None else cached.version
                            missing.append((fid, version, latest))
                        else:
                            ready.append(cached)
                    if not missing:
                        return manifest, ready
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    detail = (
                        "manifest not received"
                        if manifest is None
                        else f"missing fragment versions {missing}"
                    )
                    raise TimeoutError(
                        f"finalization timed out after {timeout:.1f}s: {detail}"
                    )
                self._final_cond.wait(min(remaining, 0.5))

    def wait_for_final_manifest(self, timeout: float | None = None) -> FinalManifest:
        """Wait for the final version manifest only (a syncer that runs
        ``--final-cut-payload manifest`` sends no final fragments).

        The manifest names the final version of every fragment; the final
        parameters stay with the syncer. :meth:`acknowledge_finalization`
        then confirms this exact manifest."""
        timeout = self.finalization_timeout if timeout is None else timeout
        deadline = time.monotonic() + timeout
        with self._final_cond:
            while True:
                self.check_health()
                if self._final_manifest is not None:
                    return self._final_manifest
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"finalization timed out after {timeout:.1f}s: manifest not received"
                    )
                self._final_cond.wait(min(remaining, 0.5))

    def acknowledge_finalization(
        self, manifest: FinalManifest, timeout: float | None = None
    ) -> None:
        """Confirm the exact cut and wait boundedly until ACK bytes are sent.

        The training loop may exit as soon as this returns: the sender thread
        has completed ``sendall`` for FINAL_ACK, so saving cannot race client
        teardown and silently lose the acknowledgement.
        """
        if manifest != self._final_manifest:
            raise RuntimeError("cannot acknowledge a stale final manifest")
        timeout = self.finalization_timeout if timeout is None else timeout
        deadline = time.monotonic() + timeout
        payload = _FINAL_ACK.pack(FINALIZATION_REVISION, manifest.global_step)
        self._final_ack_step = manifest.global_step
        while True:
            self.check_health()
            if self.shutdown.is_set():
                # The coordinator may receive an ACK from the previous live
                # generation just before a reconnect. Receipt of SHUTDOWN for
                # this exact manifest is equivalent confirmation.
                self.finalized.set()
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"finalization ACK was not sent within {timeout:.1f}s"
                )
            with self._lock:
                connected = self._connected.is_set()
                gen = self._gen
            if not connected:
                with self._final_cond:
                    self._final_cond.wait(min(remaining, 0.5))
                continue
            sent = threading.Event()
            if not self._enqueue(
                0,
                self._frame(MSG_FINAL_ACK, payload),
                gen=gen,
                sent=sent,
            ):
                continue
            while not sent.wait(min(0.1, max(0.0, deadline - time.monotonic()))):
                self.check_health()
                with self._lock:
                    if gen != self._gen:
                        break
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"finalization ACK was not sent within {timeout:.1f}s"
                    )
            if sent.is_set():
                self.finalized.set()
                with self._final_cond:
                    self._final_cond.notify_all()
                return

    @staticmethod
    def _drain(q: queue.Queue) -> list:
        out = []
        while True:
            try:
                out.append(q.get_nowait())
            except queue.Empty:
                return out

    # -- internals -----------------------------------------------------------------

    @staticmethod
    def _frame(msg_type: int, payload: bytes) -> bytes:
        return _HEADER.pack(MAGIC, msg_type, len(payload)) + payload

    def _enqueue(
        self,
        stream: int,
        data: bytes,
        gen: int | None = None,
        sent: threading.Event | None = None,
    ) -> bool:
        """Enqueue-or-drop. Never raises and never blocks unboundedly: during
        an outage the message is dropped (the syncer's quorum tolerates
        missing responders, and learner counters only reset on broadcast
        receipt, so a lost push is retried implicitly by the next pull).
        ``gen`` pins the message to one connection group so a multi-chunk
        message never straddles two connections. Returns True if enqueued."""
        while True:
            with self._lock:
                if self._closed.is_set() or self.shutdown.is_set():
                    return False
                if not self._connected.is_set():
                    return False
                if gen is not None and gen != self._gen:
                    return False
                q = self._queues[stream]
            meter = self._meters[stream]
            meter.enqueued(len(data))  # before put: the sender may drain it at once
            try:
                q.put(
                    _Outbound(data, sent),
                    timeout=0.5,
                )  # bounded wait, then re-check the group
                return True
            except queue.Full:
                meter.enqueued(-len(data))
                continue

    def _next_data_stream(self) -> int:
        """Socket index (1-based; 0 only without data streams) for the next
        CHUNK envelope, by ``stream_schedule``."""
        if self.num_streams == 0:
            return 0
        return 1 + pick_stream(self.stream_schedule, self._meters[1:], next(self._rr))

    def _send_large(self, msg_type: int, payload: bytes) -> None:
        inner = self._frame(msg_type, payload)
        if self.num_streams == 0:
            self._enqueue(0, inner)
            return
        with self._lock:
            if not self._connected.is_set():
                return  # outage: drop the whole message rather than send a torso
            gen = self._gen
        msg_id = next(self._msg_id)
        total = len(inner)
        for offset in range(0, total, CHUNK_SIZE):
            chunk = inner[offset : offset + CHUNK_SIZE]
            envelope = self._frame(MSG_CHUNK, _CHUNK_HEAD.pack(msg_id, total, offset) + chunk)
            stream = self._next_data_stream()
            if not self._enqueue(stream, envelope, gen=gen):
                return  # group died mid-message; drop the remainder

    def _send_large_parts(
        self,
        msg_type: int,
        payload_prefix: bytes,
        parts: Iterable[bytes | bytearray | memoryview],
        parts_size: int,
        *,
        label: str,
        before_last_enqueue: Callable[[], None] | None = None,
    ) -> bool:
        """Emit one logical frame without joining its bytes-like parts."""
        payload_size = len(payload_prefix) + parts_size
        fixed = _HEADER.pack(MAGIC, msg_type, payload_size) + payload_prefix
        total = len(fixed) + parts_size
        with self._lock:
            if (
                self._closed.is_set()
                or self.shutdown.is_set()
                or not self._connected.is_set()
            ):
                return False  # outage: explicitly report a whole-message drop
            gen = self._gen
            connection_generation = self.connection_generation

        part_iter = iter(parts)
        part_index = 0
        pending: memoryview | None = None
        pending_offset = 0
        parts_seen = 0
        msg_id = next(self._msg_id)
        offset = 0
        envelope_prefix = _HEADER.size + _CHUNK_HEAD.size
        enqueued_chunks = 0

        try:
            while offset < total:
                inner_size = min(CHUNK_SIZE, total - offset)
                envelope = bytearray(envelope_prefix + inner_size)
                _HEADER.pack_into(
                    envelope,
                    0,
                    MAGIC,
                    MSG_CHUNK,
                    _CHUNK_HEAD.size + inner_size,
                )
                _CHUNK_HEAD.pack_into(
                    envelope,
                    _HEADER.size,
                    msg_id,
                    total,
                    offset,
                )

                inner_end = offset + inner_size
                write_at = envelope_prefix
                if offset < len(fixed):
                    fixed_end = min(inner_end, len(fixed))
                    fixed_bytes = fixed[offset:fixed_end]
                    envelope[write_at : write_at + len(fixed_bytes)] = fixed_bytes
                    write_at += len(fixed_bytes)

                remaining = envelope_prefix + inner_size - write_at
                while remaining:
                    if pending is None:
                        try:
                            part = next(part_iter)
                        except StopIteration:
                            raise ValueError(
                                f"{label} has {parts_seen} delta bytes, "
                                f"expected {parts_size}"
                            ) from None
                        pending = _contiguous_bytes_view(part, part_index, label)
                        part_index += 1
                        pending_offset = 0
                        if pending.nbytes == 0:
                            pending.release()
                            pending = None
                            continue
                        if parts_seen + pending.nbytes > parts_size:
                            raise ValueError(
                                f"{label} exceeds expected {parts_size} delta bytes"
                            )

                    available = pending.nbytes - pending_offset
                    take = min(remaining, available)
                    envelope[write_at : write_at + take] = pending[
                        pending_offset : pending_offset + take
                    ]
                    write_at += take
                    remaining -= take
                    pending_offset += take
                    parts_seen += take
                    if pending_offset == pending.nbytes:
                        pending.release()
                        pending = None

                if inner_end == total:
                    if parts_seen != parts_size:
                        raise ValueError(
                            f"{label} has {parts_seen} delta bytes, "
                            f"expected {parts_size}"
                        )
                    for part in part_iter:
                        extra = _contiguous_bytes_view(part, part_index, label)
                        part_index += 1
                        try:
                            if extra.nbytes:
                                raise ValueError(
                                    f"{label} exceeds expected {parts_size} delta bytes"
                                )
                        finally:
                            extra.release()
                    if before_last_enqueue is not None:
                        callback = before_last_enqueue
                        before_last_enqueue = None
                        callback()

                stream = self._next_data_stream()
                if not self._enqueue(stream, envelope, gen=gen):
                    if enqueued_chunks:
                        raise PartialMessageGenerationLost(
                            connection_generation=connection_generation,
                            connection_epoch=gen,
                            queued_chunks=enqueued_chunks,
                            operation=label,
                        )
                    return False
                enqueued_chunks += 1
                offset = inner_end
            return True
        except BaseException as exc:
            if enqueued_chunks:
                self._poison_outbound_generation(gen, exc)
            raise
        finally:
            if pending is not None:
                pending.release()

    def _poison_outbound_generation(self, gen: int, exc: BaseException) -> None:
        """Invalidate and promptly close a generation holding a partial frame."""
        with self._lock:
            if gen != self._gen:
                return
            self._last_err = exc
            self._connected.clear()
            self._failure.set()
            socks = list(self._socks)
        for sock in socks:
            _close_socket(sock)
        with self._final_cond:
            self._final_cond.notify_all()

    def _socket_failed(self, gen: int, exc: BaseException) -> None:
        if self.shutdown.is_set() or self._closed.is_set():
            return
        with self._lock:
            if gen != self._gen:
                return  # a thread of an already-torn-down group; ignore
            self._last_err = exc
            self._connected.clear()
            self._failure.set()
        with self._final_cond:
            self._final_cond.notify_all()

    def _protocol_failed(self, gen: int, exc: BaseException) -> None:
        with self._lock:
            if gen != self._gen:
                return
            error = exc if isinstance(exc, ProtocolError) else ProtocolError(str(exc))
            self._err = error
            self._last_err = error
            self._connected.clear()
            self._closed.set()
            self._failure.set()
        with self._final_cond:
            self._final_cond.notify_all()

    def _send_loop(
        self,
        gen: int,
        sock: socket.socket,
        q: queue.Queue,
        meter: StreamMeter | None = None,
    ) -> None:
        if meter is None:  # a sender outside a connection group (tests)
            meter = StreamMeter()
        try:
            while True:
                item = q.get()
                if item is None:
                    return
                # send() instead of sendall() so the meter sees byte progress
                # inside one 4 MiB envelope (stall watchdog, per-stream rate).
                view = memoryview(item.data)
                meter.begin()
                try:
                    while view:
                        sent = sock.send(view)
                        meter.progress(sent)
                        view = view[sent:]
                finally:
                    meter.end()
                    view.release()
                with self._bytes_sent_lock:
                    self._bytes_sent += len(item.data)
                if all(m.idle for m in self._meters):
                    self._upload_drained_at = time.time()
                if item.sent is not None:
                    item.sent.set()
        except BaseException as e:
            self._socket_failed(gen, e)

    def _stalled_stream(self) -> str | None:
        """Describe the first stream without progress for the stall timeout."""
        limit = self.stream_stall_timeout
        now = time.monotonic()
        for index, meter in enumerate(self._meters):
            idle_for = meter.stalled_for(now)
            if idle_for > limit:
                return f"outbound stream {index} made no progress for {idle_for:.0f}s"
        with self._reasm_lock:
            touched = dict(self._reasm_touched)
        for msg_id, last in touched.items():
            if now - last > limit:
                return f"inbound message {msg_id} got no chunk for {now - last:.0f}s"
        return None

    def _watch_stalls(self) -> None:
        """Fail the live group when one stream stops moving bytes.

        A stalled TCP stream can sit for many minutes before the kernel gives
        up (C32: 900 s with every socket still open). The failure goes
        through the normal reconnect path, so the syncer replays its PULLs
        and the island pushes the interrupted fragment again."""
        period = min(1.0, self.stream_stall_timeout / 4)
        while not self._closed.wait(period):
            if self.shutdown.is_set():
                return
            with self._lock:
                gen = self._gen
                live = self._connected.is_set()
            if not live:
                continue
            reason = self._stalled_stream()
            if reason is None:
                continue
            with self._lock:
                if gen != self._gen:
                    continue
                self._stalls += 1
                socks = list(self._socks)
            _connect_log(f"{reason} (learner {self.learner_id}); dropping the connection group")
            self._socket_failed(gen, StreamStalledError(reason))
            for sock in socks:  # unblock senders stuck in send()
                _close_socket(sock)

    def _recv_loop(self, gen: int, sock: socket.socket) -> None:
        test_rate = self._recovery_read_rate or self._downlink_read_rate
        if test_rate is not None and self._socks and sock is not self._socks[0]:
            sock = _RateLimitedSocket(sock, test_rate)
        try:
            while True:
                msg_type, payload = read_frame(sock, self._incoming_frame_limit)
                if msg_type == MSG_CHUNK:
                    inner = self._reassemble(gen, payload)
                    if inner is not None:
                        if isinstance(inner, _StreamedInboundFragment):
                            self._dispatch_streamed(gen, inner)
                        else:
                            self._dispatch(gen, *inner)
                else:
                    self._dispatch(gen, msg_type, payload)
        except (ValueError, ProtocolError) as e:
            self._protocol_failed(gen, e)
        except BaseException as e:
            self._socket_failed(gen, e)

    def _incoming_frame_limit(self, msg_type: int) -> int:
        limits = {
            MSG_PULL_REQ: 16,
            MSG_BCAST_FRAGMENT: self._max_bcast_payload,
            MSG_FINAL_FRAGMENT: self._max_final_payload,
            MSG_FINAL_MANIFEST: _FINAL_MANIFEST_HEAD.size
            + 8 * self.layout.num_fragments,
            MSG_SHUTDOWN: 0,
            MSG_INIT_CONFIRM: 32,
            MSG_ERROR: MAX_ERROR_FRAME,
            MSG_CHUNK: _CHUNK_HEAD.size + CHUNK_SIZE,
            rp.MSG_RECOVERY_CLOCK: rp.MAX_RECOVERY_FRAME,
            rp.MSG_PEER_SNAPSHOT_REQ: rp.MAX_RECOVERY_FRAME,
            rp.MSG_RECOVERY_SOURCE: rp.MAX_RECOVERY_FRAME,
            rp.MSG_RECOVERY_DONE: rp.MAX_RECOVERY_FRAME,
        }
        try:
            return limits[msg_type]
        except KeyError:
            raise ValueError(f"unexpected syncer message type {msg_type}") from None

    def _reassemble(
        self, gen: int, payload: bytes
    ) -> tuple[int, memoryview] | _StreamedInboundFragment | None:
        if len(payload) < _CHUNK_HEAD.size:
            raise ValueError("chunk header truncated")
        msg_id, total, offset = _CHUNK_HEAD.unpack_from(payload)
        data = payload[_CHUNK_HEAD.size :]
        if not data:
            raise ValueError("empty chunk")
        if total < _HEADER.size:
            raise ValueError("chunked inner frame is shorter than its header")
        if total > self._max_chunked_inner:
            raise ValueError(
                f"chunked frame has length {total}, exceeds negotiated limit "
                f"{self._max_chunked_inner}"
            )
        end = offset + len(data)
        if end > total:
            raise ValueError("chunk overflow")
        with self._reasm_lock:
            if gen != self._gen:
                return None  # chunk from a dead group; msg_ids restarted server-side
            if msg_id not in self._reasm:
                if len(self._reasm) >= MAX_PARTIAL_MESSAGES:
                    raise ValueError("too many partial chunked messages")
                if self._inbound_chunk_sink is None:
                    self._reasm[msg_id] = (bytearray(total), [0], [])
                else:
                    sink = self._inbound_chunk_sink
                    self._reasm[msg_id] = _StreamedReassembly(
                        total=total,
                        sink=sink,
                        transaction=sink.begin_message(msg_id, total),
                    )
            self._reasm_touched[msg_id] = time.monotonic()
            partial = self._reasm[msg_id]
            if isinstance(partial, _StreamedReassembly):
                if partial.total != total:
                    self._reasm.pop(msg_id, None)
                    self._abort_streamed_partial(partial)
                    raise ValueError("chunk total changed within one message")
                try:
                    completed = self._stream_chunk_locked(
                        msg_id,
                        partial,
                        offset,
                        data,
                    )
                except (OSError, RuntimeError, TypeError, ValueError):
                    self._reasm.pop(msg_id, None)
                    self._abort_streamed_partial(partial)
                    raise
                if completed is not None:
                    self._reasm.pop(msg_id, None)
                    self._reasm_touched.pop(msg_id, None)
                return completed
            buf, filled, ranges = partial
            if len(buf) != total:
                raise ValueError("chunk total changed within one message")
            if any(offset < old_end and old_start < end for old_start, old_end in ranges):
                raise ValueError("overlapping chunk")
            buf[offset:end] = data
            filled[0] += len(data)
            ranges.append((offset, end))
            if filled[0] < total:
                return None
            if filled[0] != total:
                raise ValueError("chunk byte count exceeds frame length")
            del self._reasm[msg_id]
            self._reasm_touched.pop(msg_id, None)
        magic, msg_type, length = _HEADER.unpack_from(buf)
        if magic != MAGIC or length != total - _HEADER.size:
            raise ValueError("corrupt reassembled frame")
        return msg_type, memoryview(buf)[_HEADER.size :]

    def _stream_chunk_locked(
        self,
        msg_id: int,
        partial: _StreamedReassembly,
        offset: int,
        data: bytes,
    ) -> _StreamedInboundFragment | None:
        """Validate/reorder one bounded CHUNK and feed tensor bytes to its sink."""

        del msg_id
        total = partial.total
        end = offset + len(data)
        if any(
            offset < old_end and old_start < end
            for old_start, old_end in partial.ranges
        ):
            raise ValueError("overlapping chunk")
        if offset == 0:
            tensor_start = _HEADER.size + 12
            if len(data) < tensor_start:
                raise ValueError("first streamed fragment chunk is header-truncated")
            magic, msg_type, length = _HEADER.unpack_from(data)
            if magic != MAGIC or length != total - _HEADER.size:
                raise ValueError("corrupt streamed inner frame")
            if msg_type not in (MSG_BCAST_FRAGMENT, MSG_FINAL_FRAGMENT):
                raise ValueError(
                    f"transactional CHUNK has unsupported inner type {msg_type}"
                )
            fragment_id, version = struct.unpack_from("<IQ", data, _HEADER.size)
            if fragment_id >= self.layout.num_fragments:
                raise ValueError(
                    f"streamed fragment has unknown fragment {fragment_id}"
                )
            dtype = (
                bulk_dtype(self.dtype)
                if msg_type == MSG_BCAST_FRAGMENT
                else DTYPE_F32
            )
            expected_payload = _tensor_nbytes(
                dtype,
                self.layout.fragments[fragment_id].numel,
            )
            if msg_type == MSG_BCAST_FRAGMENT and self._downlink is not None:
                # The downlink header (kind, base version, digest) is part of
                # the payload the wrapping sink decodes; its kind fixes the size.
                if len(data) < tensor_start + DOWNLINK_HEADER_LEN:
                    raise ValueError("first downlink fragment chunk is header-truncated")
                expected_payload = DOWNLINK_HEADER_LEN + downlink_tensor_nbytes(
                    data[tensor_start], self.layout.fragments[fragment_id].numel)
            if length != 12 + expected_payload:
                raise ValueError(
                    f"streamed fragment {fragment_id} has {length - 12} tensor "
                    f"bytes, expected {expected_payload}"
                )
            if partial.msg_type is not None and (
                partial.msg_type,
                partial.fragment_id,
                partial.version,
                partial.payload_bytes,
            ) != (msg_type, fragment_id, version, expected_payload):
                raise ValueError("streamed fragment header changed")
            partial.msg_type = msg_type
            partial.fragment_id = fragment_id
            partial.version = version
            partial.payload_bytes = expected_payload
            partial.sink.bind_fragment(
                partial.transaction,
                msg_type,
                fragment_id,
                version,
                expected_payload,
            )

        partial.ranges.append((offset, end))
        partial.filled += len(data)
        tensor_start = _HEADER.size + 12
        data_start = max(offset, tensor_start)
        if end > data_start:
            relative_offset = data_start - tensor_start
            view = memoryview(data)[data_start - offset :]
            try:
                if (
                    partial.msg_type is not None
                    and relative_offset == partial.next_payload_offset
                ):
                    partial.sink.consume_chunk(
                        partial.transaction,
                        relative_offset,
                        view,
                    )
                    partial.next_payload_offset += view.nbytes
                    self._drain_staged_chunks_locked(partial)
                else:
                    if relative_offset in partial.pending:
                        raise ValueError("duplicate streamed tensor chunk offset")
                    token = partial.sink.stage_chunk(partial.transaction, view)
                    if token is None:
                        raise RuntimeError(
                            "inbound chunk sink returned no staged object"
                        )
                    partial.pending[relative_offset] = _StagedInboundChunk(
                        relative_offset,
                        view.nbytes,
                        token,
                    )
            finally:
                view.release()

        if partial.filled < total:
            return None
        if partial.filled != total:
            raise ValueError("chunk byte count exceeds frame length")
        if (
            partial.msg_type is None
            or partial.fragment_id is None
            or partial.version is None
            or partial.payload_bytes is None
        ):
            raise ValueError("streamed fragment completed without its header")
        self._drain_staged_chunks_locked(partial)
        if partial.pending or partial.next_payload_offset != partial.payload_bytes:
            raise ValueError("streamed fragment tensor coverage is incomplete")
        committed = partial.sink.finish_message(partial.transaction)
        if not isinstance(committed, StreamedInboundPayload):
            raise TypeError("inbound chunk sink returned an invalid committed payload")
        return _StreamedInboundFragment(
            partial.msg_type,
            partial.fragment_id,
            partial.version,
            committed,
        )

    @staticmethod
    def _drain_staged_chunks_locked(partial: _StreamedReassembly) -> None:
        while partial.next_payload_offset in partial.pending:
            staged = partial.pending.pop(partial.next_payload_offset)
            token = staged.token
            if token is None:
                raise RuntimeError("staged inbound chunk lost its object reference")
            partial.sink.consume_staged_chunk(
                partial.transaction,
                staged.offset,
                token,
                staged.nbytes,
            )
            staged.token = None
            partial.next_payload_offset += staged.nbytes

    @staticmethod
    def _abort_streamed_partial(partial: _StreamedReassembly) -> None:
        tokens = [
            staged.token
            for staged in partial.pending.values()
            if staged.token is not None
        ]
        partial.pending.clear()
        partial.sink.abort_message(partial.transaction, tokens)

    def _clear_partial_reassemblies(self) -> None:
        with self._reasm_lock:
            partials = list(self._reasm.values())
            self._reasm.clear()
            self._reasm_touched.clear()
        for partial in partials:
            if isinstance(partial, _StreamedReassembly):
                try:
                    self._abort_streamed_partial(partial)
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    # Teardown must still close the entire connection group,
                    # but retain a diagnostic if the typed sink misbehaved.
                    self._last_err = exc

    def _dispatch(
        self, gen: int, msg_type: int, payload: bytes | memoryview
    ) -> None:
        with self._lock:
            if gen != self._gen:
                return  # late message from a dead group
            # Keep the generation check and every resulting state mutation
            # atomic with respect to teardown/redial. Otherwise an obsolete
            # receiver could pass the check, lose the race to a reconnect,
            # then publish a stale terminal manifest or shutdown afterward.
            self._dispatch_live(gen, msg_type, payload)

    def _dispatch_streamed(
        self,
        gen: int,
        inbound: _StreamedInboundFragment,
    ) -> None:
        with self._lock:
            if gen != self._gen:
                if inbound.payload.discard is not None:
                    inbound.payload.discard()
                return
            self._dispatch_streamed_live(inbound)

    def _dispatch_streamed_live(self, inbound: _StreamedInboundFragment) -> None:
        fragment_id = inbound.fragment_id
        version = inbound.version
        committed = inbound.payload
        digest = committed.payload_hash
        if inbound.msg_type == MSG_BCAST_FRAGMENT:
            with self._bcast_lock:
                seen = self._bcast_seen[fragment_id]
                if seen is not None:
                    seen_version, seen_digest = seen
                    if version < seen_version:
                        if committed.discard is not None:
                            committed.discard()
                        return
                    if version == seen_version:
                        if digest != seen_digest:
                            if committed.discard is not None:
                                committed.discard()
                            raise ProtocolError(
                                "conflicting BCAST_FRAGMENT payloads for fragment "
                                f"{fragment_id} version {version}"
                            )
                        if committed.discard is not None:
                            committed.discard()
                        return
                self._bcast_seen[fragment_id] = (version, digest)
                self._bcasts.put(
                    BcastFragment(
                        fragment_id,
                        version,
                        committed.data,
                        payload_hash=digest,
                        stored=True,
                        discard=committed.discard,
                    )
                )
            return
        if inbound.msg_type == MSG_FINAL_FRAGMENT:
            update = FinalFragment(
                fragment_id,
                version,
                committed.data,
                payload_hash=digest,
                stored=True,
                discard=committed.discard,
            )
            with self._final_cond:
                current = self._final_fragments.get(fragment_id)
                if current is not None:
                    current_digest = current.payload_hash
                    if current_digest is None:
                        current_digest = hashlib.sha256(
                            memoryview(current.data).cast("B")
                        ).digest()
                    if version < current.version:
                        if committed.discard is not None:
                            committed.discard()
                        return
                    if version == current.version:
                        if digest != current_digest:
                            if committed.discard is not None:
                                committed.discard()
                            self._set_protocol_error(
                                "conflicting FINAL_FRAGMENT payloads for fragment "
                                f"{fragment_id} version {version}"
                            )
                            return
                        if committed.discard is not None:
                            committed.discard()
                        return
                self._final_fragments[fragment_id] = update
                self._final_cond.notify_all()
            return
        if committed.discard is not None:
            committed.discard()
        self._set_protocol_error(
            f"unsupported streamed syncer message type {inbound.msg_type}"
        )

    def _dispatch_live(
        self, gen: int, msg_type: int, payload: bytes | memoryview
    ) -> None:
        if msg_type == MSG_ERROR:
            message = bytes(payload).decode("utf-8", errors="replace")
            if message.startswith(rp.LEASE_EXPIRED_PREFIX):
                # Not fatal: redial with a new generation; the syncer treats
                # it as a re-JOIN and catches the island up.
                self.lease_expiries += 1
                self._socket_failed(gen, LeaseExpiredError(message))
                return
            text = f"syncer rejected protocol session: {message}"
            metric = next(
                (m for prefix, m in SESSION_REJECT_PREFIXES if message.startswith(prefix)),
                None,
            )
            self._protocol_failed(
                gen,
                SessionRejectedError(metric, text) if metric else ProtocolError(text),
            )
            return
        if msg_type == MSG_SHUTDOWN:
            if payload:
                self._set_protocol_error("SHUTDOWN payload must be empty")
                return
            manifest = self._final_manifest
            if manifest is None or self._final_ack_step != manifest.global_step:
                self._set_protocol_error(
                    "syncer sent legacy SHUTDOWN before the versioned final "
                    "manifest was applied; upgrade both syncer and learner"
                )
                return
            self.shutdown.set()
            with self._final_cond:
                self._final_cond.notify_all()
            return
        if msg_type in _RECOVERY_DECODERS:
            event = _RECOVERY_DECODERS[msg_type](payload, self.layout.num_fragments)
            if isinstance(event, rp.SnapshotRequest):
                self._snapshot_requests.put(event)
                return
            if isinstance(event, rp.RecoveryDone):
                self._recover.clear()
            self._recovery_events.put(event)
            return
        if msg_type == MSG_INIT_CONFIRM:
            if self._init_combined is None:
                raise ProtocolError("INIT_CONFIRM for an island that sent no INIT_DIGEST")
            if bytes(payload) != self._init_combined:
                raise ProtocolError("INIT_CONFIRM digest differs from this island's fragments")
            self._init_confirmed.set()
            return
        if msg_type == MSG_PULL_REQ:
            if len(payload) != 16:
                raise ValueError("PULL_REQ payload must be 16 bytes")
            fid, step, round_attempt = struct.unpack("<IQI", payload)
            if fid >= self.layout.num_fragments:
                raise ValueError(f"PULL_REQ for unknown fragment {fid}")
            if self._downlink_window is not None:
                self._downlink_window.see_pull(step)
            self._pulls.put(PullRequest(fid, step, round_attempt))
        elif msg_type == MSG_BCAST_FRAGMENT:
            if len(payload) < 12:
                raise ValueError("BCAST_FRAGMENT header truncated")
            fid, version = struct.unpack_from("<IQ", payload)
            if fid >= self.layout.num_fragments:
                raise ValueError(f"BCAST_FRAGMENT for unknown fragment {fid}")
            if self._downlink is not None:
                data = self._downlink.decode_whole(
                    fid, version, payload[12:], self.layout.fragments[fid].numel)
            else:
                expected = _tensor_nbytes(
                    bulk_dtype(self.dtype), self.layout.fragments[fid].numel
                )
                if len(payload) - 12 != expected:
                    raise ValueError(
                        f"BCAST_FRAGMENT {fid} has {len(payload) - 12} tensor bytes, "
                        f"expected {expected}"
                    )
                data = payload[12:]
            digest = hashlib.sha256(data).digest()
            with self._bcast_lock:
                seen = self._bcast_seen[fid]
                if seen is not None:
                    seen_version, seen_digest = seen
                    if version < seen_version:
                        return
                    if version == seen_version:
                        if digest != seen_digest:
                            raise ProtocolError(
                                f"conflicting BCAST_FRAGMENT payloads for fragment "
                                f"{fid} version {version}"
                            )
                        return
                self._bcast_seen[fid] = (version, digest)
                # Receiver threads run once per striped socket. Queue the
                # accepted update under the same lock as the monotonicity
                # check so two versions cannot validate in order but enqueue
                # in reverse order after a thread reschedule.
                self._bcasts.put(BcastFragment(fid, version, data))
        elif msg_type == MSG_FINAL_FRAGMENT:
            if len(payload) < 12:
                self._set_protocol_error("truncated FINAL_FRAGMENT")
                return
            fid, version = struct.unpack_from("<IQ", payload)
            if fid >= self.layout.num_fragments:
                self._set_protocol_error(f"FINAL_FRAGMENT has unknown fragment {fid}")
                return
            expected_size = 12 + _tensor_nbytes(
                DTYPE_F32, self.layout.fragments[fid].numel
            )
            if len(payload) != expected_size:
                self._set_protocol_error(
                    f"FINAL_FRAGMENT {fid} has {len(payload) - 12} tensor bytes, "
                    f"expected {expected_size - 12} f32 bytes"
                )
                return
            update = FinalFragment(fid, version, payload[12:])
            update_digest = hashlib.sha256(memoryview(update.data).cast("B")).digest()
            with self._final_cond:
                current = self._final_fragments.get(fid)
                if current is not None:
                    if version < current.version:
                        return
                    if version == current.version:
                        current_digest = current.payload_hash
                        if current_digest is None:
                            current_digest = hashlib.sha256(
                                memoryview(current.data).cast("B")
                            ).digest()
                        if update_digest != current_digest:
                            self._set_protocol_error(
                                "conflicting FINAL_FRAGMENT payloads for fragment "
                                f"{fid} version {version}"
                            )
                        return
                self._final_fragments[fid] = update
                self._final_cond.notify_all()
        elif msg_type == MSG_FINAL_MANIFEST:
            try:
                manifest = decode_final_manifest(payload, self.layout.num_fragments)
            except ValueError as exc:
                self._set_protocol_error(f"invalid FINAL_MANIFEST: {exc}")
                return
            with self._final_cond:
                if self._final_manifest is not None and self._final_manifest != manifest:
                    self._set_protocol_error(
                        "syncer sent conflicting FINAL_MANIFEST payloads"
                    )
                    return
                self._final_manifest = manifest
                self.finalizing.set()
                self._final_cond.notify_all()
        else:
            self._set_protocol_error(
                f"unsupported syncer message type {msg_type}; upgrade both peers"
            )

    def _set_protocol_error(self, message: str) -> None:
        self._protocol_failed(self._gen, ProtocolError(message))


_RECOVERY_DECODERS = {
    rp.MSG_RECOVERY_CLOCK: rp.RecoveryClock.decode,
    rp.MSG_PEER_SNAPSHOT_REQ: rp.SnapshotRequest.decode,
    rp.MSG_RECOVERY_SOURCE: rp.RecoverySource.decode,
    rp.MSG_RECOVERY_DONE: rp.RecoveryDone.decode,
}


# TCP keepalive for syncer connections (rl-infra-spec pause-audit: without it
# a NAT/LB that drops an idle flow leaves the learner blocked on recv forever;
# seen in the 2.4 T2R2 s29 run). Probes start after KEEPIDLE s of silence and a
# dead peer is detected after KEEPIDLE + KEEPINTVL*KEEPCNT = 120 s, well below
# common NAT idle timeouts (>= 300 s); the socket then errors, the supervisor
# records the failure and check_health() raises (max_reconnects=0) instead of
# waiting silently.
TCP_KEEPIDLE_S = 60
TCP_KEEPINTVL_S = 15
TCP_KEEPCNT = 4


def enable_tcp_keepalive(sock: socket.socket) -> None:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    for name, value in (
        ("TCP_KEEPIDLE", TCP_KEEPIDLE_S),
        ("TCP_KEEPINTVL", TCP_KEEPINTVL_S),
        ("TCP_KEEPCNT", TCP_KEEPCNT),
    ):
        option = getattr(socket, name, None)
        if option is not None:  # Linux; other platforms keep their defaults
            sock.setsockopt(socket.IPPROTO_TCP, option, value)


def set_tcp_user_timeout(sock: socket.socket, seconds: float) -> None:
    """Bound how long unacknowledged data may sit before the kernel resets
    the connection (Linux TCP_USER_TIMEOUT); other platforms keep defaults."""
    option = getattr(socket, "TCP_USER_TIMEOUT", None)
    if option is not None:
        sock.setsockopt(socket.IPPROTO_TCP, option, max(1, int(seconds * 1000)))


def _close_socket(sock: socket.socket) -> None:
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    sock.close()
