"""Island-side error feedback for linear int4 pushes (session dtype 7).

An int4 push rounds each outer delta. Error feedback keeps that rounding
error (the residual) and adds it to the same fragment's next delta, so the
error does not accumulate over a run: the syncer receives
``delta_k + residual_{k-1}`` encoded, and ``residual_k`` is what that
encoding lost.

Scope and lifetime of the residual:

* One residual per fragment. A fragment belongs to exactly one owner rank's
  shard, so the store holds, per rank, only that rank's own shard. It is
  allocated when the fragment is first pushed, in ``bf16`` (2 bytes/value,
  default) or ``f32`` (4 bytes/value).
* While a push streams, a pending residual of the same size is built; it
  replaces the stored one only after the client queued every chunk
  (``commit``). A push that fails or is replayed after a reconnect reuses
  the previous residual, so a replay sends the same bytes.
* The residual is used only when the next push of that fragment is exactly
  one sweep later (``global_step + fragments_per_sweep``). If the island
  skipped rounds (re-JOIN, catch-up to a newer authoritative fragment), the
  residual is dropped: it belongs to a base the syncer has moved past.
* A new stream (fresh start or resume from a cut) starts with no residual;
  residuals are not saved in checkpoints.

The wrapper emits f32 bytes; ``SyncerClient`` encodes them with the same
block size, so the residual matches the bytes on the wire exactly. Error
feedback therefore requires nearest rounding (no stochastic rounding).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

from .protocol import f32_bytes_to_bf16, f32_bytes_to_i4, i4_bytes_to_f32, i4_nbytes

RESIDUAL_DTYPES = ("bf16", "f32")


def _bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


class PushErrorFeedback:
    """Per-fragment int4 residuals for one island role stream."""

    def __init__(self, block: int, residual_dtype: str = "bf16") -> None:
        i4_nbytes(0, block)  # validates the block size
        if residual_dtype not in RESIDUAL_DTYPES:
            raise ValueError(f"residual dtype must be one of {', '.join(RESIDUAL_DTYPES)}")
        self.block = block
        self.residual_dtype = residual_dtype
        # fragment -> (global_step of the push that produced it, residual)
        self._residuals: dict[int, tuple[int, np.ndarray]] = {}
        self._pending: dict[int, tuple[int, np.ndarray]] = {}

    @property
    def bytes_per_value(self) -> int:
        return 2 if self.residual_dtype == "bf16" else 4

    def _store(self, values: np.ndarray) -> np.ndarray:
        if self.residual_dtype == "f32":
            return values.astype(np.float32)
        return np.frombuffer(f32_bytes_to_bf16(values.astype("<f4").tobytes()), "<u2").copy()

    def _load(self, stored: np.ndarray) -> np.ndarray:
        return stored if self.residual_dtype == "f32" else _bf16_to_f32(stored)

    def parts(
        self,
        fragment_id: int,
        global_step: int,
        fragments_per_sweep: int,
        numel: int,
        f32_parts: Iterable[bytes | bytearray | memoryview],
    ) -> Iterator[bytes]:
        """Yield ``delta + residual`` as f32 bytes and stage the new residual."""

        carried = None
        previous = self._residuals.get(fragment_id)
        if previous is not None:
            if previous[0] + fragments_per_sweep == global_step:
                carried = previous[1]
            else:  # skipped rounds (catch-up): the residual's base is gone
                del self._residuals[fragment_id]
        self._pending.pop(fragment_id, None)
        staged = np.empty(numel, np.float32 if self.residual_dtype == "f32" else np.uint16)
        span = 4 * self.block
        carry = b""
        offset = 0

        def emit(data: bytes) -> bytes:
            nonlocal offset
            delta = np.frombuffer(data, dtype="<f4")
            end = offset + delta.size
            if end > numel:
                raise ValueError(f"push delta for fragment {fragment_id} exceeds {numel} values")
            to_send = delta if carried is None else delta + self._load(carried[offset:end])
            to_send = to_send.astype("<f4")
            raw = to_send.tobytes()
            sent = i4_bytes_to_f32(f32_bytes_to_i4(raw, self.block), to_send.size)
            staged[offset:end] = self._store(to_send - sent)
            offset = end
            return raw

        for part in f32_parts:
            data = carry + bytes(part)
            whole = len(data) - len(data) % span
            carry = data[whole:]
            if whole:
                yield emit(data[:whole])
        if len(carry) % 4:
            raise ValueError("f32 push delta ends mid-value")
        if carry:
            yield emit(carry)
        if offset != numel:
            raise ValueError(f"push delta for fragment {fragment_id} has {offset} values, expected {numel}")
        self._pending[fragment_id] = (global_step, staged)

    def commit(self, fragment_id: int, global_step: int) -> None:
        """Adopt the staged residual once its push was fully queued."""

        staged = self._pending.pop(fragment_id, None)
        if staged is None or staged[0] != global_step:
            raise RuntimeError(f"no staged residual for fragment {fragment_id} step {global_step}")
        self._residuals[fragment_id] = staged

    def discard(self, fragment_id: int) -> None:
        """Drop a staged residual whose push was not queued."""

        self._pending.pop(fragment_id, None)

    def reset(self) -> None:
        self._residuals.clear()
        self._pending.clear()

    def resident_bytes(self) -> dict[int, int]:
        """Bytes held per fragment (stored plus staged residuals)."""

        held: dict[int, int] = {}
        for store in (self._residuals, self._pending):
            for fragment_id, (_, values) in store.items():
                held[fragment_id] = held.get(fragment_id, 0) + values.nbytes
        return held
