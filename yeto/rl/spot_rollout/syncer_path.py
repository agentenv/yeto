"""The syncer data path of the rollout runtime (``data_path: syncer``).

:class:`SyncerBroadcast` is the :class:`~yeto.rl.engine.rollout_data_path.WeightSource`
over a syncer subscriber (bf16 fragments of the actor's layout, materialized to
an HF directory per merged cut, :mod:`.release_store`); :class:`SyncerRelay` the
:class:`~yeto.rl.engine.rollout_data_path.SampleSink` over the syncer's sample
relay (:mod:`.relay_buffer`). :class:`SyncerDataPath` opens both on one member
incarnation and reports the syncer's stop conditions: the run finished, this
member fenced (superseded or retired), the link down for ``syncer_lost_s``, the
relay silent for ``relay_gone_s``, or the run's merged-version board regressed
(a fresh syncer replaced the run). Spec keys: :mod:`.sampler` module doc.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from yeto.rl.engine.rollout_data_path import SYNCER, DataPath

from .relay_buffer import RelaySampleBuffer
from .release_store import SubscriberReleaseStore


class SyncerBroadcast(SubscriberReleaseStore):
    """Weights from the syncer's bf16 broadcast (``SubscriberReleaseStore``)."""

    kind = SYNCER


class SyncerRelay(RelaySampleBuffer):
    """Groups through the syncer's sample relay (``RelaySampleBuffer``)."""

    kind = SYNCER

    def shard_entries(self, shard: str) -> list[Any]:
        return []  # the relay keeps no index another incarnation could read


class SyncerDataPath(DataPath):
    """The syncer data path of one sampler process: its subscriber (weights) and
    relay producer (groups) connections, one member incarnation for both."""

    kind = SYNCER
    shares_member_store = False  # the host's own directory; the syncer member table fences

    def __init__(self, spec: Mapping[str, Any], environ: Mapping[str, str] | None = None, *,
                 incarnation: str, emit: Callable[..., None]) -> None:
        from yeto.protocol import island_hmac_key_from_env, layout_fingerprint
        from yeto.rl.engine.task_dispense import member_incarnation

        from . import layout_manifest
        from .materialize import converter_from_spec, materialize
        from .relay import RelayBoard, RelayProducer
        from .relay_buffer import parse_addr
        from .release_store import subscriber_id_for
        from .subscriber import SubscriberClient, WeightFollower

        self.addr = parse_addr(str(spec["syncer"]))
        key = island_hmac_key_from_env(environ)
        self.subscriber_id = subscriber_id_for(str(spec["sampler_id"]))
        board = RelayBoard(self.addr, layout_manifest.LAYOUT_BOARD, self.subscriber_id, key=key)
        try:
            self.layout, self.meta = layout_manifest.fetch(
                board, timeout_s=float(spec.get("layout_timeout_s", 1800.0)))
        finally:
            board.close()
        # One member incarnation for the process's subscriber, relay producer and
        # task dispenser connections (engine_sampler_entry.run derives it from the
        # process incarnation); a newer process of this sampler supersedes all three.
        self.incarnation = member_incarnation(environ)
        self.client = SubscriberClient(self.addr, self.subscriber_id, self.layout,
                                       num_streams=int(spec.get("num_streams", 4)), hmac_key=key,
                                       incarnation=self.incarnation)
        self.client.start()
        self.follower = WeightFollower(self.client, self.layout.num_fragments)
        # The producer's stream is random per connection object: a restarted writer
        # never reuses the sequence numbers of an earlier one.
        self.producer = RelayProducer(self.addr, int(spec["channel"]), self.subscriber_id, key=key,
                                      incarnation=self.incarnation,
                                      gone_s=float(spec.get("relay_gone_s", 900.0)))
        self.syncer_lost_s = float(spec.get("syncer_lost_s", 1800.0))
        self._lost_since: float | None = None
        convert = converter_from_spec(spec)
        work = Path(spec["work_dir"]).expanduser()
        work.mkdir(parents=True, exist_ok=True)
        base_model = spec.get("base_model") or spec.get("base_dir")
        self.releases = SyncerBroadcast(
            self.follower,
            lambda cut: materialize(self.layout, cut, base_model=base_model, out_root=work / "weights",
                                    convert=convert, model_name=spec.get("model_name")),
            read_position=lambda: self.buffer.read_position(),
            reload_every=int(spec.get("reload_every", 1)), finished=self.client.run_finished,
            layout_fingerprint=layout_fingerprint(self.layout).hex(),
            base_version=int(spec.get("base_version", 0)), base_dir=spec.get("base_dir") or base_model)
        from .version_board import VERSION_BOARD, VersionReader

        # The run's merged version, read at every round admission
        self.version = VersionReader(lambda: RelayBoard(self.addr, VERSION_BOARD, self.subscriber_id, key=key))
        if spec.get("version_board", True):  # False: drills that need stale groups
            self.releases.version_reader = self.version
        self.buffer = SyncerRelay(self.producer, fragment_versions=self.releases.fragment_versions,
                                        should_stop=self.stopping, cut_of=self.releases.cut_identity)
        from yeto.rl.engine.resume import store_for

        self.member_store = store_for(str(spec.get("store") or work / "store"), environ=environ)
        emit("rl_spot_sampler_channels", syncer=f"{self.addr[0]}:{self.addr[1]}",
             subscriber_id=self.subscriber_id, channel=int(spec["channel"]),
             member_incarnation=f"{self.incarnation:016x}",
             fragments=self.layout.num_fragments, layout_meta=self.meta)

    @property
    def weights(self) -> Any:
        return self.releases

    @property
    def samples(self) -> Any:
        return self.buffer

    def syncer_lost(self) -> bool:
        if self.client._connected.is_set():
            self._lost_since = None
            return False
        now = time.monotonic()
        self._lost_since = now if self._lost_since is None else self._lost_since
        return now - self._lost_since >= self.syncer_lost_s

    def stopping(self) -> bool:
        # fenced: a newer process of this sampler took its id on the head
        # version.regressed: the board went back below a version this sampler read,
        # i.e. a fresh syncer replaced the run it joined
        return (self.client.run_finished() or self.client.fenced() or self.syncer_lost()
                or self.buffer.gone or self.version.regressed)

    def stop_reason(self) -> tuple[Any, dict[str, Any]] | None:
        """(LeaveReason, detail) of a fence stop, None otherwise (the island
        then leaves FINISHED as before). A sampler that left because a fresh
        syncer replaced its run leaves FENCED, fenced="other run"; one the
        syncer retired names the retire reason. A run that ended leaves
        FINISHED even if a fence came with it."""
        from yeto.rl.pool_client import LeaveReason

        from .release_store import fence_reason

        if self.client.run_finished():
            return None
        if self.version.regressed:
            return LeaveReason.FENCED, {"fenced": "other run", "run_merged_version": self.version.merged_version,
                                        "highest_merged_version": self.version.highest}
        return fence_reason(self.client)

    def close(self) -> None:
        try:
            self.client.close()
        finally:
            self.producer.close()
            self.version.close()


SpotChannels = SyncerDataPath  # the earlier name


def open_syncer(spec: Mapping[str, Any], environ: Mapping[str, str] | None = None, *, incarnation: str,
                emit: Callable[..., None]) -> SyncerDataPath:
    return SyncerDataPath(spec, environ, incarnation=incarnation, emit=emit)
