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
