"""Synced trainer islands feed sampler islands (S25 C3).

N trainer islands (``--rl-sample-role trainer`` without
``--rl-single-island-no-sync``) train against one syncer (strict or elastic
OuterStep merge) and read their rounds from one durable sample buffer. One or
more sampler islands (separate no-sync launches over the same store) write the
buffer and follow the policy releases. This module holds the engine-neutral
rules; the Miles wiring is in ``adapters.miles.sampler_island`` and
``adapters.miles.buffer_rollout``.

Who releases: exactly one island, the *publisher*, the trainer with learner id
:data:`PUBLISHER_LEARNER_ID` (0). Every trainer applies the same merged policy
at a merge boundary (the syncer sends one base to all), so one writer is enough,
and one writer keeps ``policy-releases/`` free of write races between hosts (a
Modal Volume or a bucket has no cross-host lock). The other trainers release
nothing.

When: after every successful publish of the publisher, the same point a no-sync
trainer releases (``ReleasingPublisher``): the merged policy is applied,
verified by hash and loaded on the trainer's own engines before the sampler sees
it.

Version rule: a release carries the *merged* version, the syncer's outer
version of the applied base (``IslandDriver.outer_version``; the strict syncer
has none, its committed version is the policy version). The publisher releases
only versions above the newest release (:class:`MergedReleaseGuard`), so the
release versions are strictly increasing. They are consecutive (v, v+1, ...)
when the publisher applies every merge; a publisher that misses merges (it was
late or re-joined) releases the next merged version it applies, and the gap is
reported (``rl_policy_release_gap``). The sampler stamps the release version on
its engines, so every buffered group carries the merged version as its behavior
version, and each trainer gates with ``max_policy_age`` against its own merged
version (:func:`gate_version`).

Buffer: the trainers read disjoint *lanes* of the one buffer: trainer m of N
trains the positions ``p`` with ``p % N == m`` (:class:`BufferLane`), so no
group is trained by two islands. A trainer keeps its lane cursor, one record per
round, written when the round is selected (:func:`lane_cursor_log`), and a
restarted or re-joined trainer continues after it: a group selected by a round
that did not finish is lost, never trained twice (at most once).

Stores: a shared directory (local disk, Modal Volume) or an object store read
through its object API (``CheckpointStore.objects``: S3, OSS). Each lane has one
writer, so every layout below has one writer per file or key:

* lane cursor: ``sample-buffer/lanes/<m>-of-<N>.jsonl`` in a directory
  (:class:`LaneCursorLog`, appended); ``sample-buffer/lanes/<m>-of-<N>/<seq>.json``
  in an object store (:class:`ObjectLaneCursorLog`, one new object per round:
  an object store cannot append);
* buffer index: on a store whose samplers write their own shards
  (``objects.buffer_shards``, OSS), the index has one writer, the merger
  (``sample_buffer`` module doc). With synced trainers the publisher merges
  (:func:`merges_shards`) and the other trainers only read the merged index;
  so a lane whose publisher is stopped waits for the publisher's return;
* releases: written by the publisher only, through the store mount, in the
  order of ``PolicyReleaseStore.release`` (weights, record, commit, then
  ``LATEST.json``), so a sampler that reads ``LATEST.json`` by key sees a
  complete release. Nothing here needs a cross-host lock or create-if-absent.

Own lanes (``--rl-sampler-follow trainer``, ``sample_roles`` module doc): each
trainer has its own samplers and its own store sub-directory, so it reads lane 0
of 1 there (:data:`OWN_LANE`: the whole buffer, with the same lane cursor log),
is the publisher of that sub-store, and releases and gates with its *local*
policy version (:func:`release_version`): its samplers serve the policy the
trainer itself trains, so the policy age counts the trainer's own versions.
The merged broadcasts reach the samplers inside those versions (a local version
contains the fragments the syncer overwrote). The release guard still keeps
the versions increasing (a restarted trainer continues the syncer's round).

Failures:

* the publisher island stops: no new release appears. The samplers keep the
  last release, up to their backlog cap; the other trainers keep training on
  groups within ``max_policy_age`` and pass over older ones (consumed, not
  trained). When the publisher re-joins (a replacement island with learner id
  0), it applies the syncer's newest base and releases that version: the gap
  is reported, the order stays increasing;
* another trainer stops: its lane is not read until it re-joins; the samplers'
  backlog cap follows the publisher's read position, so the buffer grows by at
  most one lane share of that cap;
* a sampler stops: the trainers wait up to ``--rl-sample-wait-s`` for groups.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PUBLISHER_LEARNER_ID = 0
LANES_DIR = "lanes"
LANE_ENV = "YETO_RL_SAMPLE_BUFFER_LANE"  # "M/N" on a synced trainer island


@dataclass(frozen=True)
class BufferLane:
    """Lane ``index`` of ``count``: the buffer positions ``p % count == index``."""

    index: int
    count: int

    def __post_init__(self) -> None:
        if not 0 <= int(self.index) < int(self.count):
            raise ValueError(f"buffer lane {self.index}/{self.count}: need 0 <= M < N")

    def owns(self, position: int) -> bool:
        return int(position) % self.count == self.index

    def env(self) -> str:
        return f"{self.index}/{self.count}"

    @classmethod
    def parse(cls, text: str | None) -> "BufferLane | None":
        if not text:
            return None
        index, sep, count = str(text).partition("/")
        if not sep or not index.strip().isdigit() or not count.strip().isdigit():
            raise ValueError(f"{LANE_ENV}={text!r}: expected M/N")
        return cls(int(index), int(count))

    def name(self) -> str:
        return f"{self.index}-of-{self.count}"

    @property
    def is_own(self) -> bool:
        """The whole buffer of one trainer whose samplers follow it (module doc)."""
        return self.count == 1

    def file_name(self) -> str:
        return f"{self.name()}.jsonl"


# A trainer whose samplers follow it (--rl-sampler-follow trainer) reads the whole
# buffer of its own sub-store and releases its local versions (module doc).
OWN_LANE = BufferLane(0, 1)


def release_version(driver: Any, state: Any, lane: BufferLane) -> int:
    """The version a synced trainer releases and gates with on ``lane``: its local
    policy version on its own lane, else the merged version."""
    return int(state.policy_version) if lane.is_own else merged_version(driver, state)


def merges_shards(lane: BufferLane | None) -> bool:
    """Whether this trainer merges the sampler shards into the buffer index (on a
    store that shards, ``sample_buffer.needs_shards``): a single trainer does, and
    of synced trainers only the publisher (module doc: one index writer)."""
    return lane is None or lane.index == PUBLISHER_LEARNER_ID


def is_publisher(learner_id: int, num_learners: int) -> bool:
    """Whether learner ``learner_id`` of ``num_learners`` synced trainers releases."""
    if not 0 <= int(learner_id) < int(num_learners):
        raise ValueError(f"learner id {learner_id} outside 0..{int(num_learners) - 1}")
    return int(learner_id) == PUBLISHER_LEARNER_ID


def merged_version(driver: Any, state: Any) -> int:
    """The version a synced trainer releases and gates with: the syncer's outer
    version of the applied base when the sync session sets one (elastic), else
    the policy version (strict and no-sync: the committed version)."""
    outer = getattr(driver, "outer_version", None)
    return int(state.policy_version) if outer is None else int(outer)


def gate_version(weight_version: int | None, merged: int | None) -> int | None:
    """The trainer version the ``max_policy_age`` gate compares against: the
    merged version once the island published one, else the engines' version."""
    return merged if merged is not None else weight_version


class MergedReleaseGuard:
    """Keeps the publisher's release versions strictly increasing.

    ``check(version, latest)`` returns ``"release"``, or ``"stale"`` for a
    version at or below the newest release (nothing is released: the samplers
    already serve that version or a newer one). A version more than one above
    the newest release is released and reported by ``on_gap(previous, version)``."""

    def __init__(self, on_gap: Callable[[int, int], None]) -> None:
        self.on_gap = on_gap

    def check(self, version: int, latest_version: int | None) -> str:
        if latest_version is None:
            return "release"
        if int(version) <= int(latest_version):
            return "stale"
        if int(version) > int(latest_version) + 1:
            self.on_gap(int(latest_version), int(version))
        return "release"


class LaneCursorLog:
    """The lane cursor of one synced trainer, durable in a shared-directory store.

    One JSON line per selected round: ``{"rollout_id", "positions", "cursor"}``
    (``cursor``: the buffer read position after the round). :meth:`last_cursor`
    is where a restarted trainer continues. ``commit`` makes the line durable
    (the store's ``commit``; a Modal Volume commit)."""

    def __init__(self, root: str | Path, lane: "BufferLane | str",
                 commit: Callable[[], None] = lambda: None) -> None:
        """``lane``: the reader's lane, or a plain name for a reader without one
        (S27 G-EQ3: a single buffer trainer keeps a cursor too)."""
        name = lane if isinstance(lane, str) else lane.name()
        self.path = Path(root) / LANES_DIR / f"{name}.jsonl"
        self.commit = commit

    def last_cursor(self) -> int:
        if not self.path.is_file():
            return 0
        cursor = 0
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                cursor = int(json.loads(line)["cursor"])
            except (ValueError, KeyError, TypeError):
                continue  # a torn last line (crash during the write)
        return cursor

    def append(self, rollout_id: int, positions: list[int], cursor: int) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = _record(rollout_id, positions, cursor)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        self.commit()


def _record(rollout_id: int, positions: list[int], cursor: int) -> str:
    return json.dumps({"rollout_id": int(rollout_id), "positions": [int(p) for p in positions],
                       "cursor": int(cursor)}, sort_keys=True)


class ObjectLaneCursorLog:
    """The lane cursor of one synced trainer in an object store (module doc).

    Round ``k`` of this lane (counted over every start of the lane) is the object
    ``<prefix><k>.json`` with the record of :class:`LaneCursorLog`. One writer per
    lane, so the next sequence number is the count of the listed objects; the
    write is a plain put (no create-if-absent between hosts needed)."""

    def __init__(self, objects: Any, prefix: str, lane: "BufferLane | str") -> None:
        self.objects = objects
        name = lane if isinstance(lane, str) else lane.name()
        self.prefix = f"{prefix.rstrip('/')}/{LANES_DIR}/{name}/"
        self._keys: list[str] | None = None

    def _listed(self) -> list[str]:
        if self._keys is None:
            self._keys = sorted(self.objects.list(self.prefix))
        return self._keys

    def last_cursor(self) -> int:
        for key in reversed(self._listed()):
            raw = self.objects.read(key)
            try:
                return int(json.loads(raw)["cursor"])
            except (TypeError, ValueError, KeyError):
                continue  # an unreadable object (should not happen: a put is atomic)
        return 0

    def append(self, rollout_id: int, positions: list[int], cursor: int) -> None:
        keys = self._listed()
        key = f"{self.prefix}{len(keys):010d}.json"
        self.objects.write(key, _record(rollout_id, positions, cursor).encode())
        keys.append(key)


SOLO_CURSOR = "solo{learner}"  # the cursor name of a buffer trainer without a lane


def lane_cursor_log(buffer: Any, learner_id: int | None = None
                    ) -> LaneCursorLog | ObjectLaneCursorLog | None:
    """The cursor log of a buffer reader: in the store's object API when it has
    one, else in the store directory (committed after each line).

    A lane reader gets its lane's log. A reader **without** a lane gets
    ``solo<learner_id>`` when ``learner_id`` is given (S27 G-EQ3: a single buffer
    trainer that restarts then continues after the groups it already trained,
    instead of reading the buffer from the first one again); None when it is not,
    which is the behaviour every caller had before.
    """
    lane = getattr(buffer, "lane", None)
    if lane is None:
        if learner_id is None:
            return None
        lane = SOLO_CURSOR.format(learner=int(learner_id))
    from .resume import CheckpointStore  # noqa: PLC0415
    from .sample_buffer import BUFFER_DIR  # noqa: PLC0415 - sample_buffer imports this module

    store = buffer.store  # a CheckpointStore, or an ObjectStore at the store root
    objects = store.objects if isinstance(store, CheckpointStore) else store
    if objects is not None:
        return ObjectLaneCursorLog(objects, BUFFER_DIR, lane)
    return LaneCursorLog(f"{store.root}/{BUFFER_DIR}", lane, commit=store.commit)

