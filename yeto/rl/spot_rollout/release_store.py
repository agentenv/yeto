"""Policy releases from the syncer instead of a store (S27 G-EA).

A sampler island follows its trainers through
:class:`~yeto.rl.engine.policy_release.PolicyFollower` over a
:class:`~yeto.rl.engine.policy_release.PolicyReleaseStore` (Modal Volume,
object store). :class:`SubscriberReleaseStore` gives the same follower the
same two calls without any store:

* ``latest()``: the newest cut the syncer subscriber received
  (:class:`~.subscriber.WeightFollower`) as a :class:`PolicyRelease` record:
  ``version`` = the merged version of the cut
  (:func:`~yeto.rl.engine.full_parameter_sync.merged_version`, the version
  synced trainers gate buffered groups on), a new ``seq`` once the merged
  version rose by ``reload_every``; ``buffer_selected`` = how many groups the
  trainer took from the relay (the sampler loop's backlog reference);
* ``resolve(release)``: the HF directory of that cut, written on local disk
  (:mod:`.materialize`).

Everything after that is unchanged: the follower's load hook
(``EngineReleaseLoader`` of the Miles sampler island,
``PortReleaseLoader`` of the engine sampler island) loads the directory,
:class:`~yeto.rl.engine.sampler_loop.SamplerLoop` admits rounds by backlog.
Only the weight channel differs (the volume or object store becomes the
syncer's broadcast).
"""

from __future__ import annotations

import dataclasses
import hashlib
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from yeto.rl.engine.policy_release import PolicyRelease, ReleaseIncomplete

from .subscriber import WeightCut, WeightFollower


class SubscriberReleaseStore:
    """``PolicyReleaseStore`` surface over a syncer subscriber (module doc)."""

    def __init__(self, follower: WeightFollower, materialize: Callable[[WeightCut], Path], *,
                 read_position: Callable[[], int | None] = lambda: None,
                 base_version: int = 0, reload_every: int = 1,
                 finished: Callable[[], bool] = lambda: False,
                 wall: Callable[[], float] = time.time) -> None:
        if int(reload_every) < 1:
            raise ValueError("reload_every must be >= 1")
        self.follower = follower
        self.materialize = materialize
        self.read_position = read_position
        self.reload_every = int(reload_every)
        self.wall = wall
        self._finished = finished
        self._offered = int(base_version)
        self._seq = 0
        self._current: PolicyRelease | None = None
        self._cuts: dict[int, WeightCut] = {}

    def finished(self) -> bool:
        """The syncer said the run ended (the sampler loop's stop condition)."""
        return bool(self._finished())

    def latest(self) -> PolicyRelease | None:
        self.follower.drain()
        cut = self.follower.cut()
        if cut is not None and cut.merged_version >= self._offered + self.reload_every:
            self._seq += 1
            self._offered = int(cut.merged_version)
            digest = hashlib.sha256(repr(cut.versions).encode()).hexdigest()
            self._current = PolicyRelease(
                seq=self._seq, version=int(cut.merged_version), policy_hash=digest,
                path=f"subscriber-r{self._seq}-v{cut.merged_version}", released_at=round(self.wall(), 3),
                origin={"kind": "subscriber", "fragment_versions_min": min(cut.versions),
                        "fragment_versions_max": max(cut.versions)})
            self._cuts = {self._seq: cut}  # only the newest cut is kept
        if self._current is None:
            return None
        return dataclasses.replace(self._current, buffer_selected=self.read_position())

    def resolve(self, release: PolicyRelease, *, from_store: bool = False) -> Path:
        cut = self._cuts.get(int(release.seq))
        if cut is None:
            raise ReleaseIncomplete(f"r{release.seq}: a newer cut replaced it")
        return Path(self.materialize(cut))

    def records(self) -> list[PolicyRelease]:
        return [] if self._current is None else [self._current]

    def fragment_versions(self) -> tuple[int, ...]:
        cut = self._cuts.get(self._seq)
        return () if cut is None else tuple(cut.versions)


def subscriber_id_for(sampler_id: str) -> int:
    """The sampler island's id on the head: ``2^31`` plus 31 bits of the SHA-256
    of its sampler id. The same formula as the task dispenser's sampler holder id
    (``yeto.rl.engine.task_dispense.sampler_holder_id``, S27 P4), so one sampler
    island has one id for its weights, its relay items and its rows."""
    digest = hashlib.sha256(f"yeto-sampler:{sampler_id}".encode()).digest()
    return (1 << 31) | (int.from_bytes(digest[:4], "big") & 0x7FFFFFFF)


def spot_release_store(environ: Mapping[str, str] | None, *, buffer: Any, base_model: str,
                       work_dir: Path, sampler_id: str, emit: Callable[..., None],
                       model_name: str | None = None, num_streams: int = 4,
                       layout_timeout_s: float = 1800.0) -> "SubscriberReleaseStore | None":
    """The Miles sampler island's release store when :data:`~.relay_buffer.SPOT_SYNCER_ENV`
    is set (None otherwise): the layout from the relay board, a subscriber
    (closed with the process), HF directories under ``work_dir``."""
    import os

    from yeto.protocol import island_hmac_key_from_env

    from . import layout_manifest
    from .materialize import materialize
    from .relay import RelayBoard
    from .relay_buffer import SPOT_SYNCER_ENV, parse_addr
    from .subscriber import SubscriberClient, WeightFollower

    env = os.environ if environ is None else environ
    text = env.get(SPOT_SYNCER_ENV)
    if not text:
        return None
    addr, key = parse_addr(text), island_hmac_key_from_env(env)
    subscriber_id = subscriber_id_for(sampler_id)
    board = RelayBoard(addr, layout_manifest.LAYOUT_BOARD, subscriber_id, key=key)
    try:
        layout, meta = layout_manifest.fetch(board, timeout_s=layout_timeout_s)
    finally:
        board.close()
    client = SubscriberClient(addr, subscriber_id, layout, num_streams=num_streams, hmac_key=key)
    client.start()
    emit("rl_spot_sampler_channels", syncer=text, subscriber_id=subscriber_id,
         fragments=layout.num_fragments, layout_meta=meta)
    out = Path(work_dir) / "spot-weights"
    return SubscriberReleaseStore(
        WeightFollower(client, layout.num_fragments),
        lambda cut: materialize(layout, cut, base_model=base_model, out_root=out, model_name=model_name),
        read_position=getattr(buffer, "read_position", lambda: None), finished=client.run_finished)
