"""Island-side timing of one sync boundary (rl-trainer-start-stop D10, L5.6).

The boundary is split into three parts so a run can tell transfer from wait:

* ``submit_ts`` - Unix time when the island finished handing its update to
  the transport.
* ``global_ready_ts`` - Unix time when the next global policy was fully
  received on the island.
* ``recv_seconds`` - wall seconds to receive that global payload, from its
  frame header to its last byte. ``None`` when the transport does not expose
  it; the wait then includes the receive time.

``sync_wait_seconds`` is the rest: the time the island waited for the syncer
(quorum, merge, other islands) before the global payload started to arrive.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class SyncTiming:
    submit_ts: float | None
    global_ready_ts: float | None
    recv_seconds: float | None = None

    @property
    def sync_wait_seconds(self) -> float | None:
        if self.submit_ts is None or self.global_ready_ts is None:
            return None
        # A base that was already there before the submit means no wait.
        return max(0.0, self.global_ready_ts - self.submit_ts - (self.recv_seconds or 0.0))

    def fields(self) -> dict[str, float | None]:
        return {
            "sync/submit_ts": self.submit_ts,
            "sync/global_ready_ts": self.global_ready_ts,
            "sync/recv_seconds": self.recv_seconds,
            "sync/sync_wait_seconds": self.sync_wait_seconds,
        }


def unix_now() -> float:
    return time.time()


def upload_fields(
    window,
    *,
    drained_ts: float | None,
    timing: SyncTiming,
    stalls: int,
) -> dict[str, object]:
    """Per-boundary upload diagnostics of one island (S26 E3b).

    ``window`` is a :class:`yeto.wan_streams.UploadWindow` covering this
    boundary. ``sync/upload_stream_mbps`` lists each data stream's rate over
    the time it was busy; ``sync/upload_mbps`` is the bytes over the longest
    busy stream (the streams run in parallel). ``sync/wait_after_upload_seconds``
    is the time from this island's last byte leaving to the global policy
    being ready: waiting for slower islands, the merge and the broadcast."""
    rates = window.stream_mbps()
    busy = max(window.busy_seconds_per_stream, default=0.0)
    measured = [rate for rate in rates if rate is not None]
    wait_after = None
    if (
        drained_ts is not None
        and timing.submit_ts is not None
        and timing.global_ready_ts is not None
        and drained_ts <= timing.global_ready_ts
    ):
        wait_after = timing.global_ready_ts - max(drained_ts, timing.submit_ts)
    return {
        "sync/upload_bytes": window.total_bytes,
        "sync/upload_busy_seconds": busy,
        "sync/upload_mbps": (window.total_bytes / busy / 1e6) if busy > 0 else None,
        "sync/upload_stream_mbps": [None if rate is None else round(rate, 3) for rate in rates],
        "sync/upload_min_stream_mbps": min(measured) if measured else None,
        "sync/upload_active_streams": len(measured),
        "sync/upload_drained_ts": drained_ts,
        "sync/wait_after_upload_seconds": wait_after,
        "sync/stream_stalls": stalls,
    }
