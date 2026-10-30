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

The release's ``policy_hash`` is the cut digest
(:func:`yeto.rl.engine.external_policy.cut_digest`: layout fingerprint, every
fragment's version and the SHA-256 of its bytes). :meth:`cut_identity` names
the cut of every merged version this sampler loaded, for the relay items
(S27 G-EM: the trainer's evidence names the cut that generated each sample);
the base version (the weights served before the first cut) is named by
:func:`~yeto.rl.engine.external_policy.base_cut_digest` of ``base_dir``.

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

from yeto.rl.engine.external_policy import ExternalCutError, base_cut_digest, cut_digest
from yeto.rl.engine.policy_release import PolicyRelease, ReleaseIncomplete

from .subscriber import WeightCut, WeightFollower


class SubscriberReleaseStore:
    """``PolicyReleaseStore`` surface over a syncer subscriber (module doc)."""

    def __init__(self, follower: WeightFollower, materialize: Callable[[WeightCut], Path], *,
                 read_position: Callable[[], int | None] = lambda: None,
                 base_version: int = 0, reload_every: int = 1,
                 finished: Callable[[], bool] = lambda: False,
                 wall: Callable[[], float] = time.time,
                 layout_fingerprint: str = "", keep_served: int = 16,
                 base_dir: str | Path | None = None) -> None:
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
        self.layout_fingerprint = str(layout_fingerprint)
        self.keep_served = int(keep_served)
        # merged version -> identity of the cut loaded for it (resolve)
        self._served: dict[int, dict[str, Any]] = {}
        self.base_version = int(base_version)
        self.base_dir = base_dir
        self._base_identity: dict[str, Any] | None = None

    def past_base(self) -> bool:
        """The run merged past the base weights: some fragment this subscriber received
        is newer than ``base_version`` (S29 spot run, defect D9). A sampler that
        (re)joins such a run must not generate on the base weights while its first
        cut is still arriving: those groups are past any policy-age limit."""
        self.follower.drain()
        return any(int(v) > self.base_version for v in self.follower.versions())

    def finished(self) -> bool:
        """The syncer said the run ended (the sampler loop's stop condition)."""
        return bool(self._finished())

    def latest(self) -> PolicyRelease | None:
        self.follower.drain()
        cut = self.follower.cut()
        if cut is not None and cut.merged_version >= self._offered + self.reload_every:
            self._seq += 1
            self._offered = int(cut.merged_version)
            digest = self._digest(cut)
            self._current = PolicyRelease(
                seq=self._seq, version=int(cut.merged_version), policy_hash=digest,
                path=f"subscriber-r{self._seq}-v{cut.merged_version}", released_at=round(self.wall(), 3),
                origin={"kind": "subscriber", "fragment_versions_min": min(cut.versions),
                        "fragment_versions_max": max(cut.versions)})
            self._cuts = {self._seq: cut}  # only the newest cut is kept
        if self._current is None:
            return None
        return dataclasses.replace(self._current, buffer_selected=self.read_position())

    def _digest(self, cut: WeightCut) -> str:
        if not cut.digests:  # a cut built without fragment digests (tests)
            return hashlib.sha256(repr(cut.versions).encode()).hexdigest()
        return cut_digest(self.layout_fingerprint, cut.versions, cut.digests)

    def resolve(self, release: PolicyRelease, *, from_store: bool = False) -> Path:
        cut = self._cuts.get(int(release.seq))
        if cut is None:
            raise ReleaseIncomplete(f"r{release.seq}: a newer cut replaced it")
        path = Path(self.materialize(cut))
        self._served[int(release.version)] = {"merged_version": int(release.version),
                                              "digest": str(release.policy_hash),
                                              "fragment_versions": [int(v) for v in cut.versions]}
        for old in sorted(self._served)[:-self.keep_served]:
            del self._served[old]
        return path

    def cut_identity(self, merged_version: int) -> dict[str, Any] | None:
        """The cut this sampler loaded for ``merged_version`` (None: not loaded, or
        older than the last ``keep_served`` loads)."""
        found = self._served.get(int(merged_version))
        if found is None and int(merged_version) == self.base_version and self.base_dir is not None:
            if self._base_identity is None:  # computed once, on the first base-version group
                try:
                    digest = base_cut_digest(self.base_dir)
                except (ExternalCutError, OSError) as error:  # the base groups are not sent
                    print(f"[spot-sampler] no identity for the base weights: {error}", flush=True)
                    self.base_dir = None
                    return None
                self._base_identity = {"merged_version": self.base_version, "digest": digest,
                                       "fragment_versions": []}
            found = self._base_identity
        return None if found is None else dict(found)

    def records(self) -> list[PolicyRelease]:
        return [] if self._current is None else [self._current]

    def fragment_versions(self) -> tuple[int, ...]:
        cut = self._cuts.get(self._seq)
        return () if cut is None else tuple(cut.versions)


def subscriber_id_for(sampler_id: str) -> int:
    """The sampler island's id on the head: ``2^31`` plus 31 bits of the SHA-256
    of its sampler id. It is the task dispenser's sampler holder id
    (``yeto.rl.engine.task_dispense.sampler_holder_id``, S27 P4), so one sampler
    island has one id for its weights, its relay items and its rows."""
    from yeto.rl.engine.task_dispense import sampler_holder_id

    return sampler_holder_id(sampler_id)


def spot_release_store(environ: Mapping[str, str] | None, *, buffer: Any, base_model: str,
                       work_dir: Path, sampler_id: str, emit: Callable[..., None],
                       model_name: str | None = None, num_streams: int = 4,
                       layout_timeout_s: float = 1800.0) -> "SubscriberReleaseStore | None":
    """The Miles sampler island's release store when :data:`~.relay_buffer.SPOT_SYNCER_ENV`
    is set (None otherwise): the layout from the relay board, a subscriber
    (closed with the process), HF directories under ``work_dir``."""
    import os

    from yeto.protocol import island_hmac_key_from_env, layout_fingerprint

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
    from yeto.rl.engine.task_dispense import member_incarnation

    client = SubscriberClient(addr, subscriber_id, layout, num_streams=num_streams, hmac_key=key,
                              incarnation=member_incarnation(env))
    client.start()
    emit("rl_spot_sampler_channels", syncer=text, subscriber_id=subscriber_id,
         member_incarnation=f"{client.incarnation:016x}",
         fragments=layout.num_fragments, layout_meta=meta)
    out = Path(work_dir) / "spot-weights"
    return SubscriberReleaseStore(
        WeightFollower(client, layout.num_fragments),
        lambda cut: materialize(layout, cut, base_model=base_model, out_root=out, model_name=model_name),
        read_position=getattr(buffer, "read_position", lambda: None),
        # fenced: a newer process of this sampler took its id on the head
        finished=lambda: client.run_finished() or client.fenced(),
        layout_fingerprint=layout_fingerprint(layout).hex(), base_dir=base_model)
