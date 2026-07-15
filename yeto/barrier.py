"""Small fail-closed helpers for barrier-synchronized learner execution."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any


def drain_required_broadcasts(
    awaiting_broadcast: dict[int, int],
    client: Any,
    drain_actions: Callable[[], list],
    apply_actions: Callable[[list], None],
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Apply all required broadcasts, even if SHUTDOWN is already queued.

    Broadcasts and SHUTDOWN share a FIFO stream in the phase-map transport,
    but the receiver thread may enqueue several broadcasts and then set its
    shutdown event before the learner thread drains that queue.  Shutdown is
    therefore terminal only when the broadcast queue is empty.  Returning
    with an outstanding fragment is forbidden.
    """
    while awaiting_broadcast:
        client.check_health()
        actions = drain_actions()
        if actions:
            apply_actions(actions)
            continue
        if client.shutdown.is_set():
            # The receiver can enqueue the final FIFO-ordered broadcasts and
            # set shutdown after our first drain but before this observation.
            # Re-drain *after* observing shutdown before treating an empty
            # queue as terminal.
            actions = drain_actions()
            if actions:
                apply_actions(actions)
                continue
            raise RuntimeError(
                "barrier-sync received shutdown before all required broadcasts"
            )
        sleep(0.002)
