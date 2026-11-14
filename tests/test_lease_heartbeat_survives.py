"""LeaseHeartbeat keeps the lease when an on_tick hook or progress raises."""
import threading

from yeto.rl.pool_client import LeaseHeartbeat


class _Client:
    def __init__(self, beats: int) -> None:
        self.sent: list[int] = []
        self.done = threading.Event()
        self._beats = beats

    def heartbeat(self, progress: int) -> None:
        self.sent.append(progress)
        if len(self.sent) >= self._beats:
            self.done.set()


def test_on_tick_error_does_not_kill_the_heartbeat_thread():
    client = _Client(beats=3)
    ticks = [0]

    def on_tick() -> None:  # e.g. a tape write failing while logging a re-JOIN
        ticks[0] += 1
        if ticks[0] == 1:
            raise OSError("tape: no space left on device")

    beat = LeaseHeartbeat(client, lease_s=0.03, on_tick=on_tick)
    beat.start()
    try:
        assert client.done.wait(5.0), "heartbeats stopped after an on_tick error"
    finally:
        beat.stop()
    assert any("on_tick" in e or "no space" in e for e in beat.errors)


def test_paused_error_does_not_kill_the_heartbeat_thread():
    client = _Client(beats=2)
    calls = [0]

    def paused() -> bool:
        calls[0] += 1
        if calls[0] == 1:
            raise RuntimeError("boom")
        return False

    beat = LeaseHeartbeat(client, lease_s=0.03, paused=paused)
    beat.start()
    try:
        assert client.done.wait(5.0)
    finally:
        beat.stop()
    assert beat.errors
