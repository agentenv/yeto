"""Run / island / incarnation labels on boat sandboxes, and the orphan sweep (S29 D15).

A spot VM that is stopped hard (preemption) runs no atexit or signal handler,
so :mod:`.boat_pool`'s exit cleanup never deletes its sandboxes; they ran
until their boat TTL.  Every sandbox a labelled process creates therefore
names its owner, and the owner's successors delete what an older incarnation
left:

* labels: ``run`` (:data:`RUN_ID_ENV`, the same value on every island of one
  run), ``island`` (:data:`ISLAND_ENV`, the spot sampler id) and ``inc``
  (:data:`INCARNATION_ENV`, the sampler process incarnation).  Without a run
  id nothing is labelled and nothing is swept.
* carrier: boat-sdk 1.10 has no ``labels`` field on a sandbox; its name
  (PATCH ``name``, up to 120 characters, listed by ``GET /sandboxes``) carries
  them as ``ytl1.<run>.<island>.<inc>.<base name>``.  The name is set right
  after the create; a process killed between the two leaves one unlabelled
  sandbox, which its TTL stops.
* sweep (:func:`sweep`): delete the sandboxes whose run equals ours and whose
  island / incarnation match a selector.  A sandbox without the ``ytl1``
  prefix, or of another run, is never touched.

Callers: the spot sampler at start (this island, every incarnation but its
own; ``yeto.rl.spot_rollout.sampler``) and the trainer's relay pump
(:class:`OrphanWatchdog`: an island superseded by a newer incarnation, or
silent for longer than its lease).
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

RUN_ID_ENV = "YETO_BOAT_RUN_ID"
ISLAND_ENV = "YETO_BOAT_ISLAND"
INCARNATION_ENV = "YETO_BOAT_INCARNATION"
ORPHAN_LEASE_ENV = "YETO_BOAT_ORPHAN_LEASE_S"  # trainer watchdog: silent island -> sweep (0 = off)
DEFAULT_ORPHAN_LEASE_S = 2400.0

PREFIX = "ytl1"
MAX_NAME = 120
MAX_VALUE = 32
_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")
# states in which a sandbox no longer runs (nothing to delete)
GONE_STATES = frozenset({"archiving", "archived", "stopped", "stopping", "cancelled", "deleted", "error"})


@dataclass(frozen=True)
class Labels:
    run: str
    island: str = ""
    inc: str = ""


@dataclass(frozen=True)
class SandboxInfo:
    sandbox_id: str
    name: str
    state: str


def clean(value: object) -> str:
    """A label value as it appears in a name: ``[A-Za-z0-9_-]``, at most 32 characters."""
    return _UNSAFE.sub("-", str(value))[:MAX_VALUE]


def labels_from_env(environ: Mapping[str, str] | None = None) -> Labels | None:
    env = os.environ if environ is None else environ
    run = clean(env.get(RUN_ID_ENV, "").strip())
    if not run:
        return None
    return Labels(run=run, island=clean(env.get(ISLAND_ENV, "").strip()),
                  inc=clean(env.get(INCARNATION_ENV, "").strip()))


def labelled_name(labels: Labels, base: str) -> str:
    head = ".".join((PREFIX, clean(labels.run), clean(labels.island), clean(labels.inc)))
    return f"{head}.{base}"[:MAX_NAME]


def parse_labels(name: str | None) -> Labels | None:
    """The labels of a sandbox name; None for a name this module did not write."""
    if not name:
        return None
    parts = str(name).split(".", 4)
    if len(parts) < 4 or parts[0] != PREFIX or not parts[1]:
        return None
    if any(_UNSAFE.search(p) for p in parts[1:4]):
        return None
    return Labels(run=parts[1], island=parts[2], inc=parts[3])


def select(sandboxes: Iterable[SandboxInfo], *, run: str, island: str | None = None,
           keep_inc: str | None = None, only_inc: str | None = None) -> list[SandboxInfo]:
    """The running sandboxes of ``run`` (and ``island``) whose incarnation is not
    ``keep_inc`` (and is ``only_inc`` when given)."""
    run = clean(run)
    if not run:
        raise ValueError("a sweep needs a run id")
    out = []
    for box in sandboxes:
        labels = parse_labels(box.name)
        if labels is None or labels.run != run:
            continue
        if island is not None and labels.island != clean(island):
            continue
        if keep_inc is not None and labels.inc == clean(keep_inc):
            continue
        if only_inc is not None and labels.inc != clean(only_inc):
            continue
        if str(box.state).lower() in GONE_STATES:
            continue
        out.append(box)
    return out


def sweep(client: Any, *, run: str, island: str | None = None, keep_inc: str | None = None,
          only_inc: str | None = None, reason: str = "") -> list[str]:
    """Delete the selected sandboxes (:func:`select`); returns the deleted ids.
    A failed delete is logged and skipped (the TTL still stops it)."""
    victims = select(client.list_sandboxes(), run=run, island=island, keep_inc=keep_inc, only_inc=only_inc)
    deleted = []
    for box in victims:
        try:
            client.delete(box.sandbox_id)
        except Exception:  # noqa: BLE001 - best effort; the boat TTL is the backstop
            logger.warning("boat sweep: delete of %s (%s) failed", box.sandbox_id, box.name, exc_info=True)
            continue
        deleted.append(box.sandbox_id)
    if deleted:
        logger.warning("boat sweep (%s): deleted %d orphan sandbox(es) of run %s island %s: %s",
                       reason or "orphans", len(deleted), run, island, deleted)
    return deleted


def sweep_stale_incarnations(environ: Mapping[str, str] | None = None, *,
                             client_factory: Callable[[], Any] | None = None) -> list[str]:
    """Sampler start: delete this island's sandboxes of every other incarnation
    of this run.  No-op without a run id, an island or an incarnation; errors
    are logged (a sweep never stops a sampler)."""
    labels = labels_from_env(environ)
    if labels is None or not labels.island or not labels.inc:
        return []
    try:
        client = client_factory() if client_factory is not None else _default_client()
        return sweep(client, run=labels.run, island=labels.island, keep_inc=labels.inc,
                     reason="sampler start")
    except Exception:  # noqa: BLE001
        logger.warning("boat sweep at sampler start failed", exc_info=True)
        return []


def _default_client() -> Any:
    from .boat_client import process_client

    return process_client()


class OrphanWatchdog:
    """Trainer side (relay pump): sweep the sandboxes of a spot island that left.

    ``observe(sampler, incarnation)`` is called for every relay item (its
    ``sampler_round``).  An island whose incarnation changed gets the older
    incarnations' sandboxes deleted (the syncer fenced that process); an
    island with no item for ``lease_s`` gets the sandboxes of its last
    incarnation deleted (its VM is presumed gone; a live but silent island
    loses its running trials, which then fail and are retried).  ``poll()``
    runs the checks at most every ``check_s``."""

    def __init__(self, run: str, *, lease_s: float = DEFAULT_ORPHAN_LEASE_S, check_s: float = 60.0,
                 client_factory: Callable[[], Any] | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 emit: Callable[..., None] | None = None) -> None:
        if not clean(run):
            raise ValueError("the orphan watchdog needs a run id")
        self.run = clean(run)
        self.lease_s = float(lease_s)
        self.check_s = float(check_s)
        self._client_factory = client_factory or _default_client
        self._client: Any = None
        self._clock = clock
        self._emit = emit
        self._lock = threading.Lock()
        self._islands: dict[str, tuple[str, float]] = {}  # island -> (incarnation, last seen)
        self._superseded: list[tuple[str, str]] = []  # (island, current incarnation) to sweep
        self._expired: set[tuple[str, str]] = set()  # (island, incarnation) already swept for lease
        self._last_check = float("-inf")
        self.deleted: list[str] = []

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None, **kwargs: Any) -> OrphanWatchdog | None:
        env = os.environ if environ is None else environ
        run = clean(env.get(RUN_ID_ENV, "").strip())
        if not run:
            return None
        lease = env.get(ORPHAN_LEASE_ENV, "").strip()
        lease_s = float(lease) if lease else DEFAULT_ORPHAN_LEASE_S
        return cls(run, lease_s=lease_s, **kwargs)

    def observe(self, sampler: object, incarnation: object) -> None:
        island, inc = clean(sampler), clean(incarnation)
        if not island or not inc:
            return
        with self._lock:
            previous = self._islands.get(island)
            self._islands[island] = (inc, self._clock())
            if previous is not None and previous[0] != inc:
                self._superseded.append((island, inc))

    def _client_or_none(self) -> Any:
        if self._client is None:
            self._client = self._client_factory()
        return self._client

    def poll(self, *, force: bool = False) -> list[str]:
        now = self._clock()
        if not force and now - self._last_check < self.check_s:
            return []
        self._last_check = now
        with self._lock:
            superseded, self._superseded = self._superseded, []
            expired = [(island, inc) for island, (inc, seen) in self._islands.items()
                       if self.lease_s > 0 and now - seen >= self.lease_s and (island, inc) not in self._expired]
        if not superseded and not expired:
            return []
        deleted: list[str] = []
        try:
            client = self._client_or_none()
            for island, inc in superseded:
                deleted += self._sweep(client, island=island, keep_inc=inc, reason="superseded incarnation")
            for island, inc in expired:
                deleted += self._sweep(client, island=island, only_inc=inc, reason="island lease expired")
                self._expired.add((island, inc))
        except Exception:  # noqa: BLE001 - the pump must keep running
            logger.warning("boat orphan watchdog: sweep failed", exc_info=True)
            with self._lock:  # retry the supersessions at the next check
                self._superseded = superseded + self._superseded
        self.deleted += deleted
        return deleted

    def _sweep(self, client: Any, *, island: str, reason: str, keep_inc: str | None = None,
               only_inc: str | None = None) -> list[str]:
        ids = sweep(client, run=self.run, island=island, keep_inc=keep_inc, only_inc=only_inc, reason=reason)
        if ids and self._emit is not None:
            self._emit("rl_boat_orphans_deleted", run=self.run, island=island, reason=reason,
                       sandboxes=ids)
        return ids
