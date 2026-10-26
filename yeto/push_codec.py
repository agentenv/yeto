"""Fast linear int4 push encoder (byte-identical to the reference codec).

``yeto.protocol.f32_bytes_to_i4`` defines the wire format: per block of
``block`` values one little-endian f32 scale (absmax / 7) and the values
rounded to nearest in [-7, 7] as two's-complement nibbles, low nibble first.
Its reference implementation makes about ten whole-array passes, each with a
fresh allocation, so a 0.5B-value fragment costs several seconds of one core
and much more when several pushes share the memory bus (S27 G-AP: 20-45 s
per fragment on the Nemotron 30B islands).

:func:`encode_i4_nearest` computes the same bytes in batches of rows that fit
in the CPU cache, reuses its scratch buffers, writes straight into one output
buffer and can split the batches over threads (numpy releases the GIL in its
array loops).  Every step is the reference arithmetic in the same order
(f32 absmax, f32 division by 7, f32 division, round half to even, clip), so
the bytes do not depend on the batch size or the thread count.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# Rows per batch: 8192 rows of 128 values = 4 MiB of f32 per scratch buffer.
BATCH_ROWS = 1 << 13
# Encoder threads per call (YETO_PUSH_ENCODE_THREADS); 1 = the calling thread.
ENCODE_THREADS_ENV = "YETO_PUSH_ENCODE_THREADS"
DEFAULT_ENCODE_THREADS = 4
# Below this many values one thread is faster than a thread handoff.
_PARALLEL_MIN_VALUES = 1 << 22

_POOL_LOCK = threading.Lock()
_POOL: list[tuple[int, ThreadPoolExecutor]] = []


def encode_threads() -> int:
    """Threads one encode call may use (``YETO_PUSH_ENCODE_THREADS``, >= 1)."""
    try:
        return max(1, int(os.environ.get(ENCODE_THREADS_ENV, DEFAULT_ENCODE_THREADS)))
    except ValueError:
        return 1


def _pool(threads: int) -> ThreadPoolExecutor:
    with _POOL_LOCK:
        if not _POOL or _POOL[0][0] != threads:
            if _POOL:
                _POOL[0][1].shutdown(wait=False)
            _POOL[:] = [(threads, ThreadPoolExecutor(max_workers=threads, thread_name_prefix="yeto-i4-encode"))]
        return _POOL[0][1]


def i4_row_bytes(width: int) -> int:
    """Encoded bytes of one block of ``width`` values: scale + nibbles."""
    return 4 + (width + 1) // 2


def encode_i4_nearest(values: np.ndarray, block: int, *, threads: int | None = None) -> bytes:
    """Encode 1-D little-endian f32 ``values`` as headerless int4 blocks.

    Same bytes as ``f32_bytes_to_i4(values, block, header=False)`` with
    nearest rounding and no segments: full blocks of ``block`` values, then
    one shorter block for the remainder.  Raises ``ValueError`` when a value
    is not finite (no bytes are returned then)."""

    values = np.asarray(values)
    if values.dtype != np.dtype("<f4") or values.ndim != 1:
        raise ValueError("int4 encoder needs a 1-D little-endian f32 array")
    count = values.size
    full_rows, tail = divmod(count, block)
    row_bytes = i4_row_bytes(block)
    out = np.empty(full_rows * row_bytes + (i4_row_bytes(tail) if tail else 0), dtype=np.uint8)
    rows = values[: full_rows * block].reshape(full_rows, block)
    out_rows = out[: full_rows * row_bytes].reshape(full_rows, row_bytes)
    batches = [(start, min(start + BATCH_ROWS, full_rows)) for start in range(0, full_rows, BATCH_ROWS)]
    workers = encode_threads() if threads is None else max(1, threads)
    if workers > 1 and len(batches) > 1 and count >= _PARALLEL_MIN_VALUES:
        # Contiguous runs of batches per task keep each worker's scratch reused.
        per_task = -(-len(batches) // workers)
        tasks = [batches[i : i + per_task] for i in range(0, len(batches), per_task)]
        futures = [_pool(workers).submit(_encode_batches, rows, out_rows, task) for task in tasks]
        finite = [future.result() for future in futures]
    else:
        finite = [_encode_batches(rows, out_rows, batches)]
    if tail:
        finite.append(_encode_rows(values[full_rows * block :].reshape(1, tail),
                                   out[full_rows * row_bytes :].reshape(1, -1)))
    if not all(finite):
        raise ValueError("int4 push delta contains a non-finite value")
    return out.tobytes()


def _encode_batches(rows: np.ndarray, out_rows: np.ndarray, batches: list[tuple[int, int]]) -> bool:
    """Encode ``rows[start:stop]`` into ``out_rows[start:stop]`` for every batch;
    False when a value is not finite."""

    if not batches:
        return True
    width = rows.shape[1]
    largest = max(stop - start for start, stop in batches)
    scratch = _Scratch(largest, width)
    finite = True
    for start, stop in batches:
        finite &= _encode_rows(rows[start:stop], out_rows[start:stop], scratch)
    return finite


class _Scratch:
    """Reusable per-thread buffers for one batch of rows."""

    def __init__(self, rows: int, width: int) -> None:
        self.work = np.empty((rows, width), dtype=np.float32)
        self.absmax = np.empty(rows, dtype=np.float32)
        self.quantized = np.empty((rows, width + width % 2), dtype=np.int8)

    def view(self, rows: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.work[:rows, :width], self.absmax[:rows], self.quantized[:rows]


def _encode_rows(chunk: np.ndarray, out: np.ndarray, scratch: _Scratch | None = None) -> bool:
    """The reference arithmetic for ``chunk`` (blocks, width) into ``out``
    (blocks, 4 + ceil(width / 2)); False when a value is not finite."""

    blocks, width = chunk.shape
    if scratch is None:
        scratch = _Scratch(blocks, width)
    work, absmax, quantized = scratch.view(blocks, width)
    np.abs(chunk, out=work)
    np.max(work, axis=1, out=absmax)
    # max propagates NaN and keeps inf, so one check covers every value.
    if not np.isfinite(absmax).all():
        return False
    scale = (absmax / np.float32(7)).astype(np.float32, copy=False)
    safe = np.where(scale > 0, scale, np.float32(1))
    np.divide(chunk, safe[:, None], out=work)
    np.rint(work, out=work)
    np.clip(work, -7, 7, out=work)
    quantized[:, :width] = work  # exact: integral values in [-7, 7]
    if width % 2:
        quantized[:, width] = 0
    zero = scale == 0
    if zero.any():
        quantized[zero] = 0
    nibbles = quantized.view(np.uint8)
    np.bitwise_and(nibbles, 0x0F, out=nibbles)
    out[:, :4] = scale.astype("<f4", copy=False).view(np.uint8).reshape(-1, 4)
    np.left_shift(nibbles[:, 1::2], 4, out=out[:, 4:])
    np.bitwise_or(out[:, 4:], nibbles[:, 0::2], out=out[:, 4:])
    return True
