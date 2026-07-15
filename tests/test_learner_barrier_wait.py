from __future__ import annotations

import importlib.util
from collections import deque
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BARRIER_SPEC = importlib.util.spec_from_file_location(
    "learner_barrier_wait", ROOT / "yeto" / "barrier.py"
)
barrier = importlib.util.module_from_spec(BARRIER_SPEC)
assert BARRIER_SPEC.loader is not None
BARRIER_SPEC.loader.exec_module(barrier)
drain_required_broadcasts = barrier.drain_required_broadcasts


class _Shutdown:
    def __init__(self, value: bool):
        self.value = value

    def is_set(self) -> bool:
        return self.value


class _Client:
    def __init__(self, *, shutdown: bool):
        self.shutdown = _Shutdown(shutdown)
        self.health_checks = 0

    def check_health(self) -> None:
        self.health_checks += 1


def test_shutdown_observation_redrains_broadcasts_queued_after_empty_drain():
    awaiting = {0: 0, 1: 0, 2: 0, 3: 0}
    queued = deque()
    client = _Client(shutdown=False)
    applied = []
    drain_calls = 0

    def drain():
        nonlocal drain_calls
        drain_calls += 1
        if drain_calls == 1:
            # Deterministically force the receiver-thread interleaving: the
            # learner's first drain sees no work, then all FIFO-ordered
            # broadcasts are queued and SHUTDOWN is observed. The helper must
            # perform a post-observation drain before deciding to fail.
            queued.append([(0, 1), (1, 2), (2, 3), (3, 4)])
            client.shutdown.value = True
            return []
        return queued.popleft() if queued else []

    def apply(actions):
        for fragment, version in actions:
            assert version > awaiting[fragment]
            del awaiting[fragment]
            applied.append((fragment, version))

    drain_required_broadcasts(
        awaiting, client, drain, apply, sleep=lambda _seconds: None
    )

    assert awaiting == {}
    assert applied == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert drain_calls == 2
    assert client.health_checks == 1


def test_shutdown_with_empty_queue_and_outstanding_fragment_fails_closed():
    awaiting = {3: 0}
    client = _Client(shutdown=True)

    with pytest.raises(RuntimeError, match="before all required broadcasts"):
        drain_required_broadcasts(
            awaiting,
            client,
            lambda: [],
            lambda _actions: None,
            sleep=lambda _seconds: None,
        )
