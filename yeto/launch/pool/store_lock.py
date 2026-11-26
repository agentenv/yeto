"""A single-writer lock on an object store, with a fencing token.

The fleet control plane keeps each pool's state in the run's store
(``fleet/pools/<pool>/state.json``, :mod:`.store_state`) so a controller on any
host can take a pool over. Two controllers must never drive one pool at once:
both would order machines, fence shards and write the state. The lock uses the
one primitive the store linearizes, an object that is created once
(``ObjectStore.create``, S3 / R2 ``If-None-Match: *``), the same way the
trainer generation lease does (``yeto.rl.engine.trainer_lease``).

Layout under ``<prefix>`` (e.g. ``fleet/pools/<pool>/lock``)::

    <prefix>/<token:08d>.json    created once, never rewritten:
        {"schema", "token", "holder", "nonce", "claimed_at"}
    <prefix>/<token:08d>.lease   rewritten by the token's holder only:
        {"token", "holder", "until", "released"}

* :meth:`StoreLock.acquire` lists the tokens and takes the highest one ``t``.
  When ``t``'s lease is still running (``until`` in the future, not
  ``released``) the lock is held: nothing is written, None is returned. Else
  it creates ``t + 1``; a lost create means another claimer won: None.
* The token is the fencing token. :meth:`StoreLock.check` (before every write
  of the guarded state) lists the tokens again: a higher one means this holder
  lost the lock for good (:class:`LockLost`, sticky); it must stop changing the
  pool and its machines.
* :meth:`StoreLock.renew` (every tick) moves the lease end forward;
  :meth:`StoreLock.release` marks it released, so the next controller takes the
  lock at once (a detach, :mod:`.controller`). A holder that died lets its
  lease run out (``ttl_s``) before anyone else takes over.

Clocks: a lease end is the holder's clock plus ``ttl_s``; a claimer compares it
with its own clock. Keep ``ttl_s`` well above the clock skew between control
hosts (minutes, not seconds). A store without an atomic create is refused.
"""

from __future__ import annotations

import json
import os
import re
import socket
import time
import uuid
from collections.abc import Callable
from typing import Any

SCHEMA = "yeto.fleet.store-lock/v1"
DEFAULT_TTL_S = 600.0
DEFAULT_CHECK_CACHE_S = 5.0
_TOKEN = re.compile(r"(\d{8,})\.json")


class StoreLockError(RuntimeError):
    """The lock cannot be used on this store (no atomic create, an unreadable record)."""


class LockLost(BaseException):
    """A higher token exists: this holder no longer owns what the lock guards.

    A ``BaseException``: the pool wraps every call that changes something (order, start,
    fence, drain, release) in a lock check (``IslandPool``'s guarded ops), and no
    ``except Exception`` around a provider call may count it as that call's failure; the
    pool's tick catches it and detaches."""

    def __init__(self, what: str, token: int, by: int) -> None:
        super().__init__(f"{what}: lock token {token} is fenced by token {by} (another controller took over)")
        self.token, self.by = int(token), int(by)


class LockReleased(LockLost):
    """This holder gave the lock up (:meth:`StoreLock.release`) or its pool stopped being
    driven (a detach, a stop that gave up on a blocked tick): nothing it guards may change
    from here, and the lease is never renewed again. A :class:`LockLost`, so every handler
    of a lost lock (the tick's detach, the stop's) treats it alike."""

    def __init__(self, what: str, token: int, why: str = "the lock was released") -> None:
        BaseException.__init__(self, f"{what}: lock token {token}: {why}")
        self.token = self.by = int(token)


class LockUnconfirmed(BaseException):
    """The store did not answer the lock check: nothing that changes the pool may run now
    (the tick ends; the next one checks again). A ``BaseException`` like :class:`LockLost`."""


def holder_id() -> str:
    """``<host>:<pid>:<random>``: who holds a lock (information only)."""
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


def require_atomic_create(objects: Any) -> None:
    from yeto.rl.engine.object_store import atomic_create  # noqa: PLC0415

    if not atomic_create(objects):
        raise StoreLockError("the object store has no atomic exclusive create (If-None-Match refused, or a "
                             "check-then-write store): the fleet control plane cannot lock a pool there")


class StoreLock:
    """One claim on ``prefix`` (module doc). Build it with :meth:`acquire`."""

    def __init__(self, objects: Any, prefix: str, token: int, holder: str, *, ttl_s: float = DEFAULT_TTL_S,
                 clock: Callable[[], float] = time.time, check_cache_s: float = DEFAULT_CHECK_CACHE_S) -> None:
        self.objects, self.prefix, self.token, self.holder = objects, prefix.rstrip("/"), int(token), holder
        self.ttl_s, self.clock, self.check_cache_s = float(ttl_s), clock, float(check_cache_s)
        self.lost: LockLost | None = None
        self.released = False
        self._checked_at: float | None = None
        self.renewed_at: float | None = None

    # -- keys
    @staticmethod
    def record_key(prefix: str, token: int) -> str:
        return f"{prefix.rstrip('/')}/{int(token):08d}.json"

    @staticmethod
    def lease_key(prefix: str, token: int) -> str:
        return f"{prefix.rstrip('/')}/{int(token):08d}.lease"

    @staticmethod
    def tokens(objects: Any, prefix: str) -> list[int]:
        """Every token claimed under ``prefix``, ascending (one LIST)."""
        base = prefix.rstrip("/") + "/"
        out = []
        for key in objects.list(base):
            match = _TOKEN.fullmatch(key[len(base):])
            if match is not None:
                out.append(int(match.group(1)))
        return sorted(out)

    @staticmethod
    def _doc(objects: Any, key: str) -> dict[str, Any] | None:
        data = objects.read(key)
        if data is None:
            return None
        try:
            doc = json.loads(data)
        except ValueError:
            return None
        return doc if isinstance(doc, dict) else None

    @classmethod
    def holder_of(cls, objects: Any, prefix: str, *, now: float | None = None) -> dict[str, Any] | None:
        """The current holder (``token``, ``holder``, ``until``) while its lease runs, else None."""
        tokens = cls.tokens(objects, prefix)
        return cls.holder_at(objects, prefix, tokens[-1], now=now) if tokens else None

    @classmethod
    def holder_at(cls, objects: Any, prefix: str, top: int, *, now: float | None = None) -> dict[str, Any] | None:
        """:meth:`holder_of` for token ``top``, the highest of a listing already made (two
        reads, no LIST)."""
        lease = cls._doc(objects, cls.lease_key(prefix, top))
        record = cls._doc(objects, cls.record_key(prefix, top)) or {}
        if lease is None:  # claimed, the first lease write not (yet) done: the record's time counts
            until = float(record.get("claimed_at") or 0.0) + float(record.get("ttl_s") or DEFAULT_TTL_S)
            lease = {"token": top, "holder": record.get("holder"), "until": until, "released": False}
        now = time.time() if now is None else now
        if lease.get("released"):
            return None
        kept = bool(lease.get("kept"))  # kept (``until`` null; older: Infinity): held until released
        until = float("inf") if kept else float(lease.get("until") or 0.0)
        if until <= now:
            return None
        return {"token": top, "holder": lease.get("holder") or record.get("holder"), "until": until}

    # -- claim
    @classmethod
    def acquire(cls, objects: Any, prefix: str, *, holder: str | None = None, ttl_s: float = DEFAULT_TTL_S,
                clock: Callable[[], float] = time.time, reclaim: bool = False) -> StoreLock | None:
        """Claim the lock: a new :class:`StoreLock` with the next token, or None when a
        running lease holds it or another claimer won the create. ``reclaim``: a running
        lease of this very ``holder`` name does not block the claim (the same owner
        restarted elsewhere: its new token fences the old one).

        One LIST: the running lease is judged on the highest token of that listing and only
        the token after it is created. A claimer that listed before another one claimed
        therefore races it for the same token (the create decides), never past it."""
        require_atomic_create(objects)
        holder = holder or holder_id()
        now = clock()
        tokens = cls.tokens(objects, prefix)
        top = tokens[-1] if tokens else 0
        current = cls.holder_at(objects, prefix, top, now=now) if top else None
        if current is not None and not (reclaim and current.get("holder") == holder):
            return None
        return cls.claim(objects, prefix, top + 1, holder=holder, ttl_s=ttl_s, clock=clock)

    @classmethod
    def claim(cls, objects: Any, prefix: str, token: int, *, holder: str, ttl_s: float = DEFAULT_TTL_S,
              clock: Callable[[], float] = time.time) -> StoreLock | None:
        """Create token ``token`` (one past the highest the caller listed) and hold it: the
        :class:`StoreLock`, or None when another claimer created it first. A store found
        to have no atomic create by this very create (an endpoint refusing If-None-Match:
        the create fell back to check-then-write) is refused (:class:`StoreLockError`)."""
        from yeto.rl.engine.object_store import atomic_create  # noqa: PLC0415

        require_atomic_create(objects)
        nonce = uuid.uuid4().hex
        record = {"schema": SCHEMA, "token": int(token), "holder": holder, "nonce": nonce, "claimed_at": clock(),
                  "ttl_s": float(ttl_s)}
        created = objects.create(cls.record_key(prefix, token), json.dumps(record).encode())
        if not atomic_create(objects):  # this create learnt the endpoint has no If-None-Match
            raise StoreLockError("the object store has no atomic exclusive create (it refused If-None-Match on "
                                 "the lock's create): the claim is not exclusive there")
        if not created:
            mine = cls._doc(objects, cls.record_key(prefix, token)) or {}
            if mine.get("nonce") != nonce:  # a lost race (a lost answer of our own create reads back our nonce)
                return None
        lock = cls(objects, prefix, int(token), holder, ttl_s=ttl_s, clock=clock)
        lock.renew()
        return lock

    # -- holding
    def check(self, *, force: bool = False) -> None:
        """Raise :class:`LockLost` once a higher token exists (sticky), :class:`LockReleased`
        once this holder released the lock. Listed at most every ``check_cache_s`` unless ``force``."""
        if self.lost is not None:
            raise self.lost
        if self.released:
            raise LockReleased(self.prefix, self.token)
        self._check_tokens(force=force)

    def _check_tokens(self, *, force: bool) -> None:
        now = self.clock()
        if not force and self._checked_at is not None and now - self._checked_at < self.check_cache_s:
            return
        tokens = self.tokens(self.objects, self.prefix)
        top = tokens[-1] if tokens else 0
        if top > self.token:
            self.lost = LockLost(self.prefix, self.token, top)
            raise self.lost
        self._checked_at = now

    def renew(self) -> None:
        """Move the lease end ``ttl_s`` past now (the holder's heartbeat). Raises :class:`LockLost`."""
        self.check(force=True)
        now = self.clock()
        self.objects.write(self.lease_key(self.prefix, self.token), json.dumps(
            {"token": self.token, "holder": self.holder, "until": now + self.ttl_s,
             "released": False}).encode())
        self.renewed_at = now

    def lease_left(self) -> float:
        """Seconds of this holder's own lease left (by its last renew; 0 before the first)."""
        if self.renewed_at is None:
            return 0.0
        return self.renewed_at + self.ttl_s - self.clock()

    def guard(self) -> None:
        """Before an action that changes what the lock guards: the token is still the highest
        (one LIST; :class:`LockLost` otherwise), and the lease is renewed once half of it is
        gone (a long tick never acts on a lease that ran out).

        A store that does not answer does not stop the action while this holder's own lease
        still runs (more than :attr:`margin_s` left): by the lock's definition nobody else
        can hold it then. Past that, :class:`LockUnconfirmed`. Only a higher token seen is
        :class:`LockLost`."""
        try:
            self.check(force=True)
            if self.renewed_at is None or self.clock() - self.renewed_at > self.ttl_s / 2:
                self.renew()
        except LockLost:
            raise
        except Exception as error:  # noqa: BLE001 - not confirmed: allowed only inside our own lease
            if self.lease_left() > self.margin_s:
                return
            raise LockUnconfirmed(f"{self.prefix}: lock not confirmed and the lease is (nearly) over "
                                  f"({error!r})"[:300]) from error

    @property
    def margin_s(self) -> float:
        """The part of the lease end a store outage may not act into (clock skew, latency)."""
        return min(60.0, self.ttl_s * 0.1)

    def release(self) -> None:
        """Give the lock up now (the next claimer does not wait for the lease). Never raises."""
        if self.released or self.lost is not None:
            return
        self.released = True  # from here on check / guard / renew raise LockReleased
        try:
            self._check_tokens(force=True)
            self.objects.write(self.lease_key(self.prefix, self.token), json.dumps(
                {"token": self.token, "holder": self.holder, "until": 0.0, "released": True}).encode())
        except LockLost:  # another holder already: its lease is not ours to end
            pass
        except Exception:  # noqa: BLE001 - the lease then runs out on its own
            pass

    close = release

    def detail(self) -> dict[str, Any]:
        return {"prefix": self.prefix, "token": self.token, "holder": self.holder, "ttl_s": self.ttl_s}
