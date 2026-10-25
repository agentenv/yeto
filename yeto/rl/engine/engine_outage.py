"""Engine outage: the rollout engines go away, the trainer stays (S27 G-DI, design v2).

A two-cluster island (``yeto.launch.engine_cluster``) runs its rollout engines on a
second, preemptible sky cluster. When that cluster is reclaimed the trainer must not
stop: the island waits with the trainer, the optimizer state and the sync session
intact; when the launcher has brought the engine cluster back, the engines restart,
receive the current policy and generation continues.

States (one outage at a time)::

    SERVING --engines gone--> WAITING --nodes back--> RESTORING --restored--> SERVING
                                  \\-- deadline --> failed (OutageTimeout: the old
                                                   island failure path)

The driver calls this object at three points (``IslandDriver`` with ``engine_outage``):

* ``guard_publish``: right before a publication. Engines gone -> the publication is
  deferred (no broadcast into dead NCCL peers) and the outage starts.
* ``watch`` (context manager around a generation): a monitor thread asks the port
  every ``poll_s`` whether the engines are alive and, on a loss, aborts the
  generation (the backend's abort), so a generation never hangs on dead engines.
  A generation that raised, or that returned while the monitor saw a loss, is lost.
* ``recover``: waits for the port's engines (bounded by ``wait_s``), emitting a
  heartbeat event every ``heartbeat_s`` (the fleet's stall detector sees progress),
  then asks the port to restore them. The driver then republishes the current policy
  (the deferred one, else the last published) and regenerates the lost batch.

The backend port (:class:`EngineRecoveryPort`) owns the engine verbs. Miles:
``yeto.rl.adapters.miles.engine_recovery``. verl: no port; the launcher refuses a
two-cluster island on verl.

Unset (every other island): the driver has no ``engine_outage`` and nothing changes.
"""

from __future__ import annotations

import contextlib
import threading
import time
from typing import Any, Callable, Iterator, Protocol, runtime_checkable

ENGINE_RECOVERY_ENV = "YETO_ISLAND_ENGINE_RECOVERY_S"
OUTAGE_EVENT = "rl_engine_outage"


class EnginesLost(RuntimeError):
    """The rollout engines are gone (their node died or the backend says so)."""


class OutageTimeout(RuntimeError):
    """The engines did not come back within the outage budget."""


@runtime_checkable
class EngineRecoveryPort(Protocol):
    """Backend verbs of an engine outage."""

    def engines_alive(self) -> bool:
        """Every rollout engine's node is alive (cheap; polled during generation)."""

    def is_engine_loss(self, error: BaseException) -> bool:
        """``error`` (from a generation or a publication) means lost engines."""

    def abort_generation(self) -> None:
        """Make an in-flight generation return or raise (best effort)."""

    def nodes_ready(self) -> bool:
        """The engines' nodes are back and the engines can be started there."""

    def restore_engines(self) -> dict[str, Any]:
        """Start the engines on the returned nodes (no weights published yet)."""


class EngineOutage:
    def __init__(self, port: EngineRecoveryPort, *, wait_s: float,
                 emit: Callable[..., Any], poll_s: float = 10.0, heartbeat_s: float = 60.0,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Any] = time.sleep) -> None:
        self.port, self.wait_s, self.emit = port, float(wait_s), emit
        self.poll_s, self.heartbeat_s, self.clock, self.sleep = poll_s, heartbeat_s, clock, sleep
        self.lost_since: float | None = None
        self.deferred: tuple[Any, int] | None = None
        self.outages = 0
        self._seen_loss = threading.Event()

    @property
    def active(self) -> bool:
        return self.lost_since is not None

    def _start(self, rollout_id: int | None, where: str, reason: str) -> None:
        if self.lost_since is None:
            self.lost_since = self.clock()
            self.outages += 1
            self.emit(OUTAGE_EVENT, phase="lost", where=where, rollout_id=rollout_id,
                      reason=reason[:500], outage=self.outages)

    def guard_publish(self, state: Any, rollout_id: int) -> bool:
        """True: publish now. False: the engines are gone; the publication of
        ``rollout_id`` is deferred to :meth:`recover`."""
        if not self.active and self.port.engines_alive():
            return True
        self.deferred = (state, rollout_id)
        self._start(rollout_id, "publish", "engines not alive before the publication")
        self.emit(OUTAGE_EVENT, phase="publish_deferred", policy_version=rollout_id,
                  outage=self.outages)
        return False

    def lost(self, error: BaseException | None) -> bool:
        """Whether a generation that raised ``error`` (None: it returned) lost its
        engines. Non-engine errors are not an outage."""
        if self._seen_loss.is_set():
            return True
        if error is None:
            return False
        return isinstance(error, EnginesLost) or bool(self.port.is_engine_loss(error))

    @contextlib.contextmanager
    def watch(self) -> Iterator[None]:
        """Around one generation: abort it when the engines die meanwhile."""
        self._seen_loss.clear()
        stop = threading.Event()

        def monitor() -> None:
            while not stop.wait(self.poll_s):
                try:
                    alive = self.port.engines_alive()
                except Exception:  # noqa: BLE001 - a failed probe is not a loss
                    continue
                if not alive:
                    self._seen_loss.set()
                    try:
                        self.port.abort_generation()
                    except Exception:  # noqa: BLE001
                        pass
                    return

        thread = threading.Thread(target=monitor, name="yeto-engine-outage", daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=self.poll_s + 5)

    def recover(self, rollout_id: int | None, *, where: str, error: BaseException | None) -> dict[str, Any]:
        """Wait for the engines' nodes, restore the engines; raise OutageTimeout at the
        deadline. Returns the port's restore record (plus waited seconds)."""
        self._start(rollout_id, where, f"{type(error).__name__}: {error}" if error else "engines lost")
        self._seen_loss.clear()
        start = self.lost_since if self.lost_since is not None else self.clock()
        last_beat = self.clock()
        while not self.port.nodes_ready():
            now = self.clock()
            if now - start >= self.wait_s:
                self.emit(OUTAGE_EVENT, phase="timeout", rollout_id=rollout_id,
                          waited_s=round(now - start, 1), outage=self.outages)
                raise OutageTimeout(f"rollout engines did not come back within {self.wait_s:.0f}s "
                                    f"({ENGINE_RECOVERY_ENV})")
            if now - last_beat >= self.heartbeat_s:
                self.emit(OUTAGE_EVENT, phase="waiting", rollout_id=rollout_id,
                          waited_s=round(now - start, 1), outage=self.outages)
                last_beat = now
            self.sleep(self.poll_s)
        waited = self.clock() - start
        self.emit(OUTAGE_EVENT, phase="restoring", rollout_id=rollout_id,
                  waited_s=round(waited, 1), outage=self.outages)
        record = dict(self.port.restore_engines() or {})
        self.lost_since = None
        record["waited_s"] = round(waited, 1)
        return record

    def take_deferred(self) -> tuple[Any, int] | None:
        deferred, self.deferred = self.deferred, None
        return deferred


def outage_budget_s(environ) -> float | None:
    """``YETO_ISLAND_ENGINE_RECOVERY_S`` as seconds; None when unset or 0."""
    raw = environ.get(ENGINE_RECOVERY_ENV)
    if not raw:
        return None
    value = float(raw)
    return value if value > 0 else None
