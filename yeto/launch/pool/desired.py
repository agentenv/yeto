"""The fleet's desired state in the run store, with a generation and conditional writes.

What the fleet should run is one document, ``fleet/desired.json`` (a copy of the newest
generation), whose generations are ``fleet/desired/<generation:08d>.json``, each created
once with an exclusive create (S3 / R2 ``If-None-Match: *``). A change is a new
generation: the writer read generation ``g`` and creates ``g + 1``; two writers that both
read ``g`` race for one key and the second is refused (:class:`DesiredConflict`: read
again, apply the change again). Readers take the highest generation listed.

Format (``schema`` ``yeto.fleet.desired/v1``)::

    {"schema": "yeto.fleet.desired/v1", "generation": 3, "run": "<run>",
     "budget": {"usd": 300},                       # optional, informational here
     "members": [
       {"kind": "sampler_pool", "name": "nc", "island": 0, "required": true,
        "pool": {<a samplers entry of the run config without "name": provider, config,
                 target, spec, interval_s, usd_per_h, data_path, inline pool keys ...>},
        "retire": ["<ticket>"], "mode": "drain"},   # optional one-shot fields, see below
       {"kind": "trainer", "name": "island-1", "learner_id": 1, ...}
     ],
     "updated_at": 1795000000.0, "updated_by": "<host:pid>", "note": "..."}

Member kinds:

* ``sampler_pool``: one sampler pool (the control plane's pool workers reconcile it).
  ``pool`` is a ``samplers`` entry of the run config (``yeto.launch.rollout.pools``),
  so ``pool.target`` is the pool's sampler count. ``retire`` / ``mode`` apply once, with
  the generation that carries them (``yeto run scale --remove``).
* ``trainer``: a trainer island (reserved for the trainer worker of the
  ``unified-trainer-islands`` change). Its fields are kept as written; the sampler
  control plane only checks ``name`` and an integer ``learner_id``.

The control plane acts on a *change*: a new generation is compared with the one it
applied before, pool by pool (a target change, a pool added, removed or reconfigured).
A pool's own target changes in between (a manual release it detected) are not undone.
What the controller applied is kept in the records ``fleet/applied/<seq>.json``, each created once
(``fleet/applied.json`` is a copy of the newest; :meth:`DesiredStore.write_applied`):
a controller that starts later applies every generation written since.
"""

from __future__ import annotations

import copy
import json
import os
import re
import socket
import sys
import time
from collections.abc import Callable
from typing import Any

SCHEMA = "yeto.fleet.desired/v1"
DESIRED_KEY = "fleet/desired.json"
APPLIED_KEY = "fleet/applied.json"  # what the controller applied (FleetReconciler): generation, pools, removals
APPLIED_DIR = "fleet/applied"  # its records, fleet/applied/<seq>.json, each created once (the writer fence)
APPLIED_KEEP = 20  # the newest records kept (older ones are deleted after a write)
APPLIED_SCHEMA = "yeto.fleet.applied/v1"
GENERATIONS_DIR = "fleet/desired"
KINDS = ("sampler_pool", "trainer")
_GEN = re.compile(r"(\d{8,})\.json")
ONE_SHOT = ("retire", "mode")  # sampler_pool fields that apply with their generation only


class DesiredError(ValueError):
    """A desired state document this code refuses."""


class DesiredConflict(RuntimeError):
    """The desired state changed since it was read: read it again and apply the change again."""

    def __init__(self, expected: int, current: int) -> None:
        super().__init__(f"the fleet's desired state is at generation {current}, not {expected}: someone changed "
                         "it meanwhile. Read it again (`yeto fleet show`) and apply your change to that")
        self.expected, self.current = int(expected), int(current)


def validate(doc: Any) -> dict[str, Any]:
    """``doc`` checked (a copy); raises :class:`DesiredError`."""
    if not isinstance(doc, dict):
        raise DesiredError("the desired state must be a JSON object")
    doc = copy.deepcopy(doc)
    schema = doc.setdefault("schema", SCHEMA)
    if schema != SCHEMA:
        raise DesiredError(f"desired state schema {schema!r}: this yeto reads {SCHEMA!r}")
    members = doc.get("members")
    if not isinstance(members, list):
        raise DesiredError("desired state: 'members' must be a list")
    names: set[str] = set()
    for i, m in enumerate(members):
        where = f"desired state members[{i}]"
        if not isinstance(m, dict):
            raise DesiredError(f"{where} must be an object")
        kind = m.get("kind")
        if kind not in KINDS:
            raise DesiredError(f"{where}: kind {kind!r} is not one of {', '.join(KINDS)}")
        name = m.get("name")
        if not isinstance(name, str) or not name or "/" in name or name in (".", ".."):
            raise DesiredError(f"{where}: 'name' must be a non-empty word without '/'")
        key = f"{kind}:{name}"
        if key in names:
            raise DesiredError(f"{where}: two {kind} members named {name!r}")
        names.add(key)
        if kind == "sampler_pool":
            pool = m.get("pool")
            if not isinstance(pool, dict):
                raise DesiredError(f"{where}: a sampler_pool needs 'pool' (a samplers entry of the run config)")
            if "name" in pool and pool["name"] != name:
                raise DesiredError(f"{where}: pool.name {pool['name']!r} differs from the member name {name!r}")
            target = pool.get("target")
            if target is not None and (isinstance(target, bool) or not isinstance(target, int) or target < 0):
                raise DesiredError(f"{where}: pool.target must be a whole number >= 0")
            if not isinstance(m.get("required", True), bool):
                raise DesiredError(f"{where}: required must be true or false")
            if m.get("retire") is not None and not isinstance(m["retire"], list):
                raise DesiredError(f"{where}: retire must be a list of tickets")
        else:
            if isinstance(m.get("learner_id"), bool) or not isinstance(m.get("learner_id"), int):
                raise DesiredError(f"{where}: a trainer member needs an integer learner_id")
    return doc


def sampler_pools(doc: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """``{name: member}`` of the sampler pools of a desired state (None: {})."""
    return {m["name"]: m for m in (doc or {}).get("members") or () if m.get("kind") == "sampler_pool"}


def trainers(doc: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    return {m["name"]: m for m in (doc or {}).get("members") or () if m.get("kind") == "trainer"}


def pool_target(member: dict[str, Any]) -> int | None:
    target = (member.get("pool") or {}).get("target")
    return None if target is None else int(target)


def pool_shape(member: dict[str, Any]) -> dict[str, Any]:
    """What a change of reconfigures (rolls) a pool: the pool entry without its target."""
    return {k: v for k, v in (member.get("pool") or {}).items() if k != "target"}


def from_run_config(cfg: dict[str, Any], *, generation: int = 0) -> dict[str, Any]:
    """The desired state a run config declares (its ``samplers`` list, or its one ``pool``)."""
    members = []
    if "pool" in cfg:
        members.append({"kind": "sampler_pool", "name": "pool", "required": True,
                        "pool": dict(cfg.get("pool") or {})})
    for entry in cfg.get("samplers") or ():
        entry = dict(entry)
        name = str(entry.pop("name", None) or _kind(entry))
        required = entry.pop("required", True)
        members.append({"kind": "sampler_pool", "name": name, "required": required, "pool": entry})
    return validate({"schema": SCHEMA, "generation": generation, "run": cfg.get("run"), "members": members})


def _kind(entry: dict[str, Any]) -> str:
    provider = entry.get("provider")
    if isinstance(provider, dict):
        return str(provider.get("kind") or "nc")
    return str(provider or "nc")


class AppliedUnsafe(RuntimeError):
    """The store cannot fence the applied state: its create is not atomic (a check-then-write
    store such as the OSS relay, or an endpoint that refuses ``If-None-Match``)."""


class AppliedFenced(RuntimeError):
    """The applied state (``fleet/applied/<seq>.json``) belongs to another controller now (it took the fleet over):
    this one writes it no more."""


class DesiredStore:
    """Reads and writes the desired state on an object API (module doc)."""

    def __init__(self, objects: Any, *, clock: Callable[[], float] = time.time,
                 log: Callable[[str], None] | None = None) -> None:
        self.objects, self.clock = objects, clock
        self.log = log or (lambda msg: print(msg, file=sys.stderr, flush=True))

    @staticmethod
    def key(generation: int) -> str:
        return f"{GENERATIONS_DIR}/{int(generation):08d}.json"

    def generations(self) -> list[int]:
        base = GENERATIONS_DIR + "/"
        out = []
        for key in self.objects.list(base):
            match = _GEN.fullmatch(key[len(base):])
            if match is not None:
                out.append(int(match.group(1)))
        return sorted(out)

    def generation(self) -> int:
        gens = self.generations()
        return gens[-1] if gens else 0

    def read(self, generation: int | None = None) -> dict[str, Any] | None:
        """The newest generation (or ``generation``); None when there is none."""
        gen = self.generation() if generation is None else int(generation)
        if gen <= 0:
            return None
        data = self.objects.read(self.key(gen))
        if data is None:
            return None
        return validate(json.loads(data))

    def apply(self, doc: dict[str, Any], *, expect: int, note: str | None = None,
              by: str | None = None) -> dict[str, Any]:
        """Write ``doc`` as generation ``expect + 1``; :class:`DesiredConflict` when the store
        is no longer at ``expect`` (another change won). Returns the written document."""
        doc = validate(doc)
        current = self.generation()
        if current != int(expect):
            raise DesiredConflict(expect, current)
        new = dict(doc, generation=current + 1, updated_at=self.clock(),
                   updated_by=by or f"{socket.gethostname()}:{os.getpid()}")
        if note is not None:
            new["note"] = note
        body = json.dumps(new, indent=1, sort_keys=True, default=str).encode()
        if not self.objects.create(self.key(current + 1), body):
            raise DesiredConflict(expect, self.generation())
        try:
            self.objects.write(DESIRED_KEY, body)  # the convenience copy of the newest generation
        except Exception:  # noqa: BLE001 - the generation record is the truth
            pass
        return new

    # -- what the controller applied (one writer: the controller of the run)
    @staticmethod
    def applied_key(seq: int) -> str:
        return f"{APPLIED_DIR}/{int(seq):08d}.json"

    def _applied_seqs(self) -> list[int]:
        base = APPLIED_DIR + "/"
        return sorted(int(m.group(1)) for key in self.objects.list(base)
                      if (m := _GEN.fullmatch(key[len(base):])) is not None)

    def _doc(self, key: str) -> dict[str, Any] | None:
        data = self.objects.read(key)
        if data is None:
            return None
        doc = json.loads(data)
        return doc if isinstance(doc, dict) and doc.get("schema") == APPLIED_SCHEMA else None

    def read_applied(self) -> dict[str, Any] | None:
        """The generation the controller last applied, with its sampler pools (``members``),
        the pools it was still removing (``removing``), its ``writer`` and the record's
        ``seq``; None before the first apply. The newest record is the higher of the newest
        listed one and the copy's (``fleet/applied.json``, written after each record: a listing
        that lags behind a create does not hide it). No record (a run that wrote only the copy,
        before the records): the copy, so pools being removed are still driven. Both formats
        side by side (an upgraded run): the higher ``seq`` wins, a copy without one counts as 0,
        so the first record written after the upgrade replaces it."""
        copy_doc = self._doc(APPLIED_KEY)
        seqs = self._applied_seqs()
        top = max([*seqs[-1:], int((copy_doc or {}).get("seq") or 0)], default=0)
        if top <= 0:
            return copy_doc  # the copy only (no seq): the older format
        doc = self._doc(self.applied_key(top))
        return doc if doc is not None else copy_doc

    def write_applied(self, generation: int, pools: dict[str, dict[str, Any]],
                      removing: dict[str, dict[str, Any]], *, writer: str | None = None,
                      claim: bool = False) -> int:
        """Write the applied state as the next record ``fleet/applied/<seq>.json``, created
        once (exclusive create, as a desired generation): of two writers that read the same
        record, one wins and the other gets :class:`AppliedFenced`. A controller claims it
        once when it starts (``claim``); from then on a write needs the newest record to be
        its own (``writer``: unique per controller, so a create whose answer was lost and is
        sent again still finds its own record), and a generation older than the stored one
        is never written back. A store whose create is not atomic raises :class:`AppliedUnsafe`
        (it cannot tell two writers apart). Returns the new record's ``seq``;
        ``fleet/applied.json`` is a copy of the newest record."""
        from yeto.rl.engine.object_store import atomic_create  # noqa: PLC0415

        if not atomic_create(self.objects):
            raise AppliedUnsafe("the store has no atomic exclusive create: the applied state cannot be fenced")
        current = self.read_applied()
        seq = int(current.get("seq") or 0) if current is not None else 0
        if writer is not None and current is not None and not claim:
            if current.get("writer") not in (None, writer):
                raise AppliedFenced(f"the applied state is written by {current.get('writer')} now "
                                    f"(record {seq}; not {writer})")
        if current is not None and int(current.get("generation") or 0) > int(generation):
            raise AppliedFenced(f"the applied state is at generation {current.get('generation')}, past {generation}")
        for _ in range(3):
            seq += 1
            doc = {"schema": APPLIED_SCHEMA, "seq": seq, "generation": int(generation),
                   "members": list(pools.values()), "removing": list(removing.values()), "at": self.clock(),
                   "writer": writer}
            body = json.dumps(doc, indent=1, sort_keys=True, default=str).encode()
            if self.objects.create(self.applied_key(seq), body):
                break
            taken = self._doc(self.applied_key(seq)) or {}
            if writer is None or taken.get("writer") != writer:
                raise AppliedFenced(f"{taken.get('writer') or 'another controller'} wrote applied "
                                    f"record {seq} first")
            # our own record (a create whose answer was lost, or a listing that lagged): the next one
        else:
            raise AppliedFenced(f"applied records past {seq - 3} kept being taken")
        if not atomic_create(self.objects):  # an endpoint found to refuse If-None-Match on that create
            raise AppliedUnsafe("the store's create turned out not atomic: the applied state cannot be fenced")
        newer = self._newer_than(seq, writer)
        if newer is not None:
            # a number pruned after another controller wrote past it was created again (ABA):
            # this writer read a stale state; its record is withdrawn and it is fenced
            try:
                self.objects.delete(self.applied_key(seq))
            except Exception as error:  # noqa: BLE001 - below the newest record: never read, pruned later
                self.log(f"fleet: withdrawn applied record {seq} not deleted ({error!r}); it is below record "
                         f"{newer.get('seq')} and never read")
            raise AppliedFenced(f"{newer.get('writer')} wrote applied record {newer.get('seq')} past {seq}")
        copy_seq = int((self._doc(APPLIED_KEY) or {}).get("seq") or 0)
        if copy_seq < seq:  # best effort forward (read, then write: a racing writer may still put it back
            # one record); readers take the newest record, the copy is only their fallback
            try:
                self.objects.write(APPLIED_KEY, body)
            except Exception as error:  # noqa: BLE001 - the record is the truth; readers fall back to it
                self.log(f"fleet: WARNING: fleet/applied.json (the copy of record {seq}) not written: {error!r}")
        self._prune_applied(seq)
        return seq

    def _newer_than(self, seq: int, writer: str | None) -> dict[str, Any] | None:
        """After a create: the newest record past ``seq`` that another writer holds (listed, or
        named by the copy), None when there is none."""
        copy_doc = self._doc(APPLIED_KEY) or {}
        top = max([*self._applied_seqs()[-1:], int(copy_doc.get("seq") or 0)], default=0)
        if top <= seq:
            return None
        doc = self._doc(self.applied_key(top)) or copy_doc
        return doc if doc.get("writer") != writer else None

    def _prune_applied(self, newest: int) -> None:
        """Delete the records older than the newest :data:`APPLIED_KEEP` (best effort)."""
        if newest <= APPLIED_KEEP:
            return
        try:
            for old in self._applied_seqs():
                if old <= newest - APPLIED_KEEP:
                    self.objects.delete(self.applied_key(old))
        except Exception as error:  # noqa: BLE001 - tried again after the next write
            self.log(f"fleet: old fleet/applied records not deleted: {error!r}")

    def update(self, change: Callable[[dict[str, Any]], None], *, seed: dict[str, Any] | None = None,
               note: str | None = None, attempts: int = 1) -> dict[str, Any]:
        """Read, ``change(doc)`` in place, write the next generation. With ``attempts`` > 1 a
        conflict reads again and applies ``change`` again (for changes that are relative,
        e.g. ``--add 1``); 1 (the default) raises it. ``seed``: generation 0's document."""
        for i in range(max(1, attempts)):
            current = self.read()
            expect = 0 if current is None else int(current["generation"])
            doc = copy.deepcopy(current if current is not None else (seed or {"members": []}))
            for m in doc.get("members") or ():  # one-shot fields belong to the generation that set them
                for key in ONE_SHOT:
                    m.pop(key, None)
            change(doc)
            try:
                return self.apply(doc, expect=expect, note=note)
            except DesiredConflict:
                if i + 1 >= max(1, attempts):
                    raise
        raise AssertionError("unreachable")  # pragma: no cover
