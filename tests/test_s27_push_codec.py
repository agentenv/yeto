"""S27 G-AP: the cache-blocked int4 encoder and the block-aligned regrouping
produce exactly the bytes of the reference codec.

The reference below is the int4 nearest-rounding encoder as it stood before
the fast path (``yeto/protocol.py`` at 869078a5), kept here as the oracle.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from yeto.protocol import (
    _block_aligned_chunks,
    _f32_parts_to_i4,
    f32_bytes_to_i4,
    i4_bytes_to_f32,
    i4_nbytes,
)
from yeto.push_codec import BATCH_ROWS, encode_i4_nearest


def _reference_i4(values: np.ndarray, block: int) -> bytes:
    count = values.size
    full = count - count % block
    parts = []

    def encode(chunk: np.ndarray) -> None:
        absmax = np.abs(chunk).max(axis=1)
        scale = (absmax / np.float32(7)).astype(np.float32)
        safe = np.where(scale > 0, scale, np.float32(1))
        q = np.clip(np.rint(chunk / safe[:, None]), -7, 7).astype(np.int8)
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


def _reference_segments(values: np.ndarray, block: int, segments: list[int]) -> bytes:
    out, at = [struct.pack("<I", block)], 0
    for n in segments:
        out.append(_reference_i4(values[at : at + n], block))
        at += n
    return b"".join(out)


def _tricky(n: int, seed: int) -> np.ndarray:
    """Values that stress the scale and rounding: zero rows, -0.0, subnormals,
    exact half steps and a wide dynamic range."""
    rng = np.random.default_rng(seed)
    values = (rng.standard_normal(n) * 10.0 ** rng.integers(-8, 2, n)).astype(np.float32)
    if n:
        values[: n // 7] = 0.0
        values[n // 7 : n // 5] = -0.0
        values[n // 5 : n // 4] = np.float32(1e-45)
        values[n // 4 : n // 3] = np.float32(0.5)  # x / (absmax / 7) lands on .5 often
    return values


@pytest.mark.parametrize("block", [64, 66, 128, 130, 256])
@pytest.mark.parametrize("n", [0, 1, 3, 63, 64, 65, 129, 1000, 3 * BATCH_ROWS * 128 + 17])
@pytest.mark.parametrize("threads", [1, 3])
def test_fast_encoder_matches_the_reference_bytes(block, n, threads):
    values = _tricky(n, seed=n + block)
    assert encode_i4_nearest(values, block, threads=threads) == _reference_i4(values, block)


def test_protocol_int4_uses_the_same_bytes_with_and_without_segments():
    values = _tricky(50_000, seed=7)
    assert f32_bytes_to_i4(values.tobytes(), 128) == struct.pack("<I", 128) + _reference_i4(values, 128)
    segments = [1, 127, 128, 129, 9_999, 39_616]
    assert f32_bytes_to_i4(values.tobytes(), 128, segments=segments) == _reference_segments(values, 128, segments)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_fast_encoder_refuses_non_finite_values_anywhere(bad):
    for position in (0, 5_000, 2 * BATCH_ROWS * 128 + 3, 4 * BATCH_ROWS * 128 + 100):
        values = np.zeros(4 * BATCH_ROWS * 128 + 101, dtype=np.float32)
        values[position] = bad
        with pytest.raises(ValueError, match="non-finite"):
            encode_i4_nearest(values, 128, threads=2)


def test_fast_encoder_round_trips_within_half_a_scale_step():
    values = _tricky(100_003, seed=3)
    decoded = i4_bytes_to_f32(struct.pack("<I", 128) + encode_i4_nearest(values, 128), values.size)
    rows = np.pad(values, (0, -values.size % 128)).reshape(-1, 128)
    step = np.repeat(np.abs(rows).max(axis=1) / 7, 128)[: values.size]
    assert np.all(np.abs(decoded - values) <= step / 2 + 1e-12)


@pytest.mark.parametrize("part_bytes", [4, 12, 512, 4 * 1000 + 2, 4 * 4096, 4 * 70_001])
def test_streaming_regroup_copies_only_the_carry_and_keeps_the_bytes(part_bytes):
    """Parts that cut segments and blocks anywhere (even mid-value) give the
    reference bytes; every regrouped chunk starts on a block of its segment."""
    values = _tricky(150_000, seed=11)
    segments = [3, 128, 1_000, 70_000, 77, 78_792]
    raw = values.tobytes()
    parts = [raw[i : i + part_bytes] for i in range(0, len(raw), part_bytes)]
    streamed = b"".join(_f32_parts_to_i4(parts, 128, None, segments))
    assert streamed == _reference_segments(values, 128, segments)
    assert len(streamed) == i4_nbytes(values.size, 128, segments)

    starts, at, bounds = set(), 0, []
    for n in segments:
        bounds.append((at, at + n))
        at += n
    for begin, end in bounds:
        starts.update(range(begin, end, 128))
    offset = 0
    for chunk in _block_aligned_chunks(parts, 128, segments):
        assert offset in starts
        offset += len(chunk) // 4
    assert offset == values.size


def test_streaming_regroup_still_refuses_bad_lengths():
    with pytest.raises(ValueError, match="mid-value"):
        list(_block_aligned_chunks([b"\0" * 6], 64, [2]))
    with pytest.raises(ValueError, match="longer than its tensor segments"):
        list(_block_aligned_chunks([b"\0" * 8, b"\0" * 8], 64, [3]))
