"""Provider-agnostic cap on the sandboxes one trial process holds at a time.

Every :class:`~.trial.HarborHarness` trial (each setup retry included) holds
one slot of its provider for as long as its sandbox can exist: from
``Trial.create`` to the end of ``Trial.run`` (Harbor tears the sandbox down
inside ``run``).  A trial that finds no free slot waits; it never starts a
sandbox above the cap.

``HARBOR_MAX_SANDBOXES``:
  ``N``                       every provider: at most N sandboxes
  ``{"remote-docker": 8, "*": 32}``  per provider, ``*`` for the others
  unset / ``{}``              no cap, except providers in :data:`CAP_REQUIRED`

Providers in :data:`CAP_REQUIRED` (a sandbox host of our own with a finite
size) refuse to run without a cap: an uncapped sampler would start as many
containers as it has trials in flight.

The cap is per process: several rollout processes that share one sandbox host
each need their share, and the host keeps its own limit (ops: sandbox host
checklist).  Slots are shared by every event loop of the process (verl and
Miles run one loop per worker thread), so the limiter is thread-safe and
cancellation-safe: a waiter that is cancelled holds nothing.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager

from yeto.rl.agentic import HarnessConfigError

MAX_SANDBOXES_ENV = "HARBOR_MAX_SANDBOXES"
CAP_REQUIRED = frozenset({"remote-docker"})
WILDCARD = "*"
POLL_S = 0.05


class SandboxSlots:
    """A counting limit usable from any thread's event loop (module doc)."""

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise HarnessConfigError(f"a sandbox cap must be >= 1, got {limit}")
        self.limit = limit
        self._lock = threading.Lock()
        self._held = 0
        self.peak = 0

    @property
    def held(self) -> int:
        with self._lock:
            return self._held

    def try_acquire(self) -> bool:
        with self._lock:
            if self._held >= self.limit:
                return False
            self._held += 1
            self.peak = max(self.peak, self._held)
            return True

    def release(self) -> None:
        with self._lock:
            if self._held <= 0:
                raise RuntimeError("sandbox slot released more often than acquired")
            self._held -= 1

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        # Polling instead of a loop-bound primitive: waiters sit on different
        # event loops, and a cancelled waiter must not take a slot later.
        while not self.try_acquire():
            await asyncio.sleep(POLL_S)
        try:
            yield
        finally:
            self.release()


def parse_caps(raw: str) -> dict[str, int]:
    """``HARBOR_MAX_SANDBOXES`` -> {provider or ``*``: cap} (module doc)."""
    raw = raw.strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise HarnessConfigError(f"{MAX_SANDBOXES_ENV} must be N or a JSON object, got {raw!r}") from exc
    caps = {WILDCARD: value} if isinstance(value, int) and not isinstance(value, bool) else value
    if not isinstance(caps, dict) or not all(
            isinstance(k, str) and isinstance(v, int) and not isinstance(v, bool) and v >= 1 for k, v in caps.items()):
        raise HarnessConfigError(f"{MAX_SANDBOXES_ENV} caps must be integers >= 1, got {raw!r}")
    return dict(caps)


class SandboxLimits:
    """One :class:`SandboxSlots` per provider, from a caps mapping."""

    def __init__(self, caps: Mapping[str, int] | None = None) -> None:
        self.caps = dict(caps or {})
        self._slots: dict[str, SandboxSlots] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> SandboxLimits:
        environ = os.environ if environ is None else environ
        return cls(parse_caps(str(environ.get(MAX_SANDBOXES_ENV, "") or "")))

    def cap(self, provider: str) -> int | None:
        return self.caps.get(provider, self.caps.get(WILDCARD))

    def slots(self, provider: str) -> SandboxSlots | None:
        """The provider's slots; None when uncapped.  Refuses an uncapped :data:`CAP_REQUIRED` provider."""
        limit = self.cap(provider)
        if limit is None:
            if provider in CAP_REQUIRED:
                raise HarnessConfigError(f"provider {provider} needs a sandbox cap: set {MAX_SANDBOXES_ENV} "
                                         f"(e.g. {{\"{provider}\": 8}}) to what the sandbox host holds")
            return None
        with self._lock:
            slots = self._slots.get(provider)
            if slots is None:
                slots = self._slots[provider] = SandboxSlots(limit)
            return slots


_process_limits: SandboxLimits | None = None
_process_lock = threading.Lock()


def process_limits() -> SandboxLimits:
    """The process-wide limits (``HARBOR_MAX_SANDBOXES`` read once), shared by every harness."""
    global _process_limits
    with _process_lock:
        if _process_limits is None:
            _process_limits = SandboxLimits.from_env()
        return _process_limits


def reset_process_limits() -> None:
    """Tests: read ``HARBOR_MAX_SANDBOXES`` again on next use."""
    global _process_limits
    with _process_lock:
        _process_limits = None
