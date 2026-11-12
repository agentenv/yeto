"""Trainer generation lease: one trainer owns a run's store at a time.

A trainer can outlive its own replacement: a container that was taken for dead
(a platform retry, a resume, a second launch by mistake) may still run while a
new trainer already writes. Both would merge the sample buffer, publish policy
releases and commit round cuts into the same keys. Nothing here compares clocks
or checks before it writes; it uses the one primitive the store linearizes, a
key that is created once (``ObjectStore.create``, S3 ``If-None-Match``).

A *generation* is one trainer's claim on the store: the record
``trainer-lease/<generation:08d>.json``, created once and never rewritten or
deleted::

    {"schema": "yeto.rl.trainer-lease/v1", "generation": G,
     "incarnation": "<the trainer's incarnation id>", "claim": "<nonce of this claim>",
     "claimed_at": <seconds, information only>, "host": "<hostname, information only>",
     "fork": {...} | null}

* :meth:`TrainerLease.claim` lists ``trainer-lease/``, takes the highest
  generation ``g`` (none: 0) and creates ``g + 1``; a claimer that lost the
  create lists again and creates the new highest + 1 (it never keeps the
  generation it lost). N claimers that race get N different, consecutive
  generations; the highest one is the trainer, the others are fenced. A create whose answer was lost (the write was applied, the
  caller saw a refusal or an error) is told from a lost race by reading the
  record back: it names this claim's nonce.
* A store without any lease record is generation 0 (:meth:`TrainerLease.current`):
  a legacy store, written by code older than this module.
* :meth:`TrainerLease.check`, before every ownership-sensitive write and
  periodically: a higher generation exists -> :class:`TrainerFenced` (sticky).
  The fenced trainer must stop for good; what it still writes lands under keys
  that carry its own generation, which no reader of a higher one follows.
* ``fork`` is an opaque mapping kept in the record (which point of which earlier
  generation this one starts from; its writer and readers are the policy version
  identity code). :meth:`TrainerLease.chain` returns every record, oldest first.

A record with another schema (a newer yeto) refuses the claim instead of being
misread. An endpoint without conditional writes and a bucket mount have no atomic
create and are refused too. Nothing here talks to a cloud.
"""

from __future__ import annotations

import json
import os
import re
import socket
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from typing import Any

LEASE_DIR = "trainer-lease"
SCHEMA = "yeto.rl.trainer-lease/v1"
GENERATION_EVENT = "rl_trainer_generation"  # a claim: generation, incarnation, fork
FENCED_EVENT = "rl_trainer_fenced"
LEGACY_GENERATION = 0  # a store without lease records
DEFAULT_CHECK_CACHE_S = 5.0

_LEASE_NAME = re.compile(r"(\d{8,})\.json")


class TrainerLeaseError(RuntimeError):
    """The lease cannot be claimed or read on this store (no atomic create, a
    record of another schema, a record that went missing)."""


class TrainerFenced(RuntimeError):
    """A higher trainer generation exists: this trainer no longer owns the store."""

    def __init__(self, generation: int, fenced_by: int) -> None:
        super().__init__(f"trainer generation {generation} is fenced: generation {fenced_by} "
                         "claimed the store; this trainer must stop and must not be restarted")
        self.generation = int(generation)
        self.fenced_by = int(fenced_by)


def lease_key(generation: int) -> str:
    return f"{LEASE_DIR}/{int(generation):08d}.json"


def lease_objects(store: Any) -> Any:
    """The object API the lease lives on: a store's ``objects``, a bare object
    API, or a local directory (tests, one host). A bucket mount has no atomic
    exclusive create and is refused."""
    from .object_store import DirectoryObjectStore  # noqa: PLC0415
    from .resume import BucketMountStore  # noqa: PLC0415

    objects = getattr(store, "objects", None)
    if objects is not None:
        return objects
    if all(callable(getattr(store, name, None)) for name in ("create", "list", "read")):
        return store
    if isinstance(store, BucketMountStore):
        raise TrainerLeaseError("the trainer lease needs the object store itself (S3 If-None-Match), "
                                "not a bucket mount: its exclusive create is not atomic")
    if isinstance(store, (str, os.PathLike)):  # a path is the directory itself (Path.root is "/")
        return DirectoryObjectStore(store)
    return DirectoryObjectStore(store.root)


def _generations(objects: Any) -> list[int]:
    """The claimed generations, ascending (one LIST)."""
    out = []
    for key in objects.list(LEASE_DIR + "/"):
        match = _LEASE_NAME.fullmatch(key[len(LEASE_DIR) + 1:])
        if match is None:  # never guess at a key another version wrote
            raise TrainerLeaseError(f"unexpected object {key!r} under {LEASE_DIR}/")
        out.append(int(match.group(1)))
    return sorted(out)


def _read(objects: Any, generation: int) -> dict[str, Any]:
    data = objects.read(lease_key(generation))
    if data is None:
        raise TrainerLeaseError(f"{lease_key(generation)} is listed but missing: lease records are never deleted")
    try:
        record = json.loads(data)
    except ValueError as error:
        raise TrainerLeaseError(f"{lease_key(generation)} is not JSON: {error}") from None
    if not isinstance(record, dict):
        raise TrainerLeaseError(f"{lease_key(generation)} is not a JSON object")
    return record


def _checked(record: dict[str, Any], generation: int) -> dict[str, Any]:
    if record.get("schema") != SCHEMA:
        raise TrainerLeaseError(
            f"{lease_key(generation)} has schema {record.get('schema')!r}, this yeto reads {SCHEMA!r}: "
            "the store was claimed by another yeto version; refusing to take it over")
    if record.get("generation") != generation:
        raise TrainerLeaseError(f"{lease_key(generation)} names generation {record.get('generation')!r}")
    return record


class TrainerLease:
    """One trainer's claim on a store (module doc). Built by :meth:`claim`."""

    def __init__(self, objects: Any, record: Mapping[str, Any], *,
                 check_cache_s: float = DEFAULT_CHECK_CACHE_S,
                 monotonic: Callable[[], float] = time.monotonic) -> None:
        self.objects = objects
        self.record = dict(record)
        self.generation = int(record["generation"])
        self.incarnation = str(record["incarnation"])
        self.check_cache_s = float(check_cache_s)
        self.monotonic = monotonic
        self.fenced_by: int | None = None
        self.checks = 0  # listings made by check()
        self._checked_at = -float("inf")
        self._lock = threading.Lock()

    @property
    def fork(self) -> dict[str, Any] | None:
        return self.record.get("fork")

    @property
    def fenced(self) -> bool:
        return self.fenced_by is not None

    @classmethod
    def claim(cls, objects: Any, *, incarnation: str, fork: Mapping[str, Any] | None = None,
              host: str | None = None, clock: Callable[[], float] = time.time,
              check_cache_s: float = DEFAULT_CHECK_CACHE_S,
              monotonic: Callable[[], float] = time.monotonic) -> "TrainerLease":
        """Claim the next generation (module doc). ``objects``: an object API or a
        store (:func:`lease_objects`)."""
        objects = lease_objects(objects)
        nonce = uuid.uuid4().hex
        # A claimer that lost the create of generation N + 1 never keeps N + 1: it lists
        # again and creates the new highest + 1, until it wins. The winner of the highest
        # generation is the trainer; every other one is fenced at its next check().
        while True:
            taken = _generations(objects)
            if taken:
                _checked(_read(objects, taken[-1]), taken[-1])
            generation = (taken[-1] if taken else LEGACY_GENERATION) + 1
            record = {"schema": SCHEMA, "generation": generation, "incarnation": str(incarnation),
                      "claim": nonce, "claimed_at": float(clock()),
                      "host": socket.gethostname() if host is None else str(host),
                      "fork": None if fork is None else dict(fork)}
            key = lease_key(generation)
            try:
                won = bool(objects.create(key, json.dumps(record, sort_keys=True).encode()))
            except Exception:
                # the create may have been applied before its answer was lost
                if not _is_claim(objects, generation, nonce):
                    raise
                won = True
            if getattr(objects, "conditional_writes", True) is False:
                raise TrainerLeaseError("the object store answered NotImplemented to If-None-Match: "
                                        "the trainer lease needs an atomic exclusive create")
            # a refused create is this claim's own when its answer was lost and a retry
            # of the same request met the record it had just written
            if won or _is_claim(objects, generation, nonce):
                return cls(objects, record, check_cache_s=check_cache_s, monotonic=monotonic)

    @staticmethod
    def current(objects: Any) -> int:
        """The highest claimed generation; 0 for a legacy store (one LIST)."""
        taken = _generations(lease_objects(objects))
        return taken[-1] if taken else LEGACY_GENERATION

    @staticmethod
    def chain(objects: Any) -> list[dict[str, Any]]:
        """Every lease record, by ascending generation."""
        objects = lease_objects(objects)
        return [_checked(_read(objects, generation), generation) for generation in _generations(objects)]

    def check(self, *, force: bool = False) -> None:
        """Raise :class:`TrainerFenced` once a higher generation exists (sticky).
        The listing is reused for ``check_cache_s`` seconds unless ``force``. A
        listing that fails raises its own error: the caller must not go on to
        write on a verdict it could not get."""
        with self._lock:
            if self.fenced_by is None and (force or self.monotonic() - self._checked_at >= self.check_cache_s):
                self.checks += 1
                taken = _generations(self.objects)
                if not taken or taken[-1] < self.generation:
                    raise TrainerLeaseError(f"{lease_key(self.generation)} is gone: lease records are never "
                                            "deleted; without it another trainer may reuse this generation")
                self._checked_at = self.monotonic()
                if taken[-1] > self.generation:
                    self.fenced_by = taken[-1]
            if self.fenced_by is not None:
                raise TrainerFenced(self.generation, self.fenced_by)

    def detail(self) -> dict[str, Any]:
        """Fields of the :data:`FENCED_EVENT` event."""
        return {"generation": self.generation, "incarnation": self.incarnation,
                "fenced_by": self.fenced_by, "lease_checks": self.checks}


def _is_claim(objects: Any, generation: int, nonce: str) -> bool:
    data = objects.read(lease_key(generation))
    if data is None:
        return False
    try:
        record = json.loads(data)
    except ValueError:
        return False
    return isinstance(record, dict) and record.get("claim") == nonce
