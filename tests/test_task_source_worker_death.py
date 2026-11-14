"""The task source's worker thread must not die silently: an unexpected error
in it (a bug, a decode slip, a failing emit) ends the source with that error,
so ``take`` / ``flush`` raise it instead of waiting for a worker that is gone."""

from types import SimpleNamespace

import pytest

from yeto.rl.engine import task_dispense as td

class _Conn:
    def __init__(self, claim_error: BaseException | None = None) -> None:
        self.welcome = td.Welcome(lease_s=60.0, passes=None, max_block=64, reconnect=False,
                                  superseded=None, reclaimed=0, next=0)
        self.claim_error = claim_error
        self.closed = False

    def claim(self, seq, count):
        if self.claim_error is not None:
            raise self.claim_error
        return SimpleNamespace(status=td.GRANT, block=1, ranges=[(0, count)], retry_s=0.0)

    def report(self, seq, entries):
        return None

    def heartbeat(self):
        return None

    def holds(self, positions):
        return [True] * len(positions)

    def close(self):
        self.closed = True

def _source(conn, **kw):
    return td.TaskSource(connect=lambda incarnation: conn, block=4, prefetch_blocks=1,
                         retry_s=0.01, incarnation=lambda: 0x42, **kw)

def test_an_unexpected_worker_error_reaches_take():
    source = _source(_Conn(claim_error=KeyError("bug in the claim path"))).start()
    try:
        with pytest.raises(KeyError, match="bug in the claim path"):
            source.take(4, timeout_s=3.0)
        source._thread.join(timeout=3.0)
        assert not source._thread.is_alive()
    finally:
        source.close(release=False, timeout_s=1.0)

def test_a_failing_emit_in_the_refusal_path_still_ends_the_source():
    def emit(kind, **fields):
        if kind == "rl_dispense_refused":
            raise ValueError("I/O operation on closed file")

    source = _source(_Conn(claim_error=td.DispenseError("refused")), emit=emit).start()
    try:
        with pytest.raises(td.DispenseError, match="refused"):
            source.take(4, timeout_s=3.0)
    finally:
        source.close(release=False, timeout_s=1.0)

def test_a_healthy_source_still_grants():
    source = _source(_Conn()).start()
    try:
        assert source.take(4, timeout_s=3.0) == [0, 1, 2, 3]
    finally:
        source.close(release=False, timeout_s=1.0)
