"""Island side of a fragment-sharded syncer head (S27 P4b).

The head can run as N syncer processes. Shard ``i`` owns the contiguous
fragment range ``[i*P//N, (i+1)*P//N)`` and runs its own rounds, state and
checkpoint (syncer/src/shard.rs holds the same rule and the global-step
interleaving). An island keeps one ``SyncerClient`` per shard and routes
every fragment-addressed message to the fragment's owner:

* INIT_PARAMS and PUSH_FRAGMENT go to the owner of their fragment;
* PULL and BCAST from all shards are merged (each shard only sends its own);
* finalization completes when every shard has delivered its final cut; the
  island's terminal cut is the union of the shards' manifests.

Every shard runs the unsharded step schedule restricted to its fragments
(step ``t`` merges fragment ``(t-1) mod P``), so PULL steps, fragment
versions and the final manifest do not depend on the shard count.

With one address this module is not used: the learner keeps a plain
``SyncerClient``.
"""

from __future__ import annotations

import time
from typing import Callable, Iterable

from .protocol import (
    BcastFragment,
    FinalFragment,
    FinalManifest,
    PullRequest,
    SendProgress,
    SyncerClient,
)
from .wan_streams import StreamSnapshot


def owned_range(index: int, count: int, num_fragments: int) -> range:
    """Fragments owned by shard ``index`` of ``count`` (mirrors shard.rs)."""
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"invalid syncer shard {index} of {count}")
    if num_fragments < count:
        raise ValueError(
            f"{count} syncer shards need at least {count} fragments, layout has {num_fragments}"
        )
    return range(index * num_fragments // count, (index + 1) * num_fragments // count)


def owner_of(fragment_id: int, count: int, num_fragments: int) -> int:
    """The shard that owns ``fragment_id``."""
    if not 0 <= fragment_id < num_fragments:
        raise ValueError(f"unknown fragment {fragment_id}")
    for index in range(count):
        if fragment_id in owned_range(index, count, num_fragments):
            return index
    raise AssertionError("shard ranges cover every fragment")


def parse_syncer_addrs(spec: str) -> list[tuple[str, int]]:
    """``host:port[,host:port...]`` in shard order."""
    addrs = []
    for item in spec.split(","):
        host, sep, port = item.strip().rpartition(":")
        if not sep or not host or not port.isdigit():
            raise ValueError(f"syncer address {item!r} is not host:port")
        addrs.append((host, int(port)))
    return addrs


class _AllSet:
    """Read-only view of several ``threading.Event``: set when all are set."""

    def __init__(self, events) -> None:
        self._events = tuple(events)

    def is_set(self) -> bool:
        return all(event.is_set() for event in self._events)

    def wait(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        for event in self._events:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if not event.wait(remaining):
                return False
        return True


class ShardedSyncerClient:
    """One island's connections to every shard of a sharded syncer head.

    Exposes the subset of the ``SyncerClient`` interface the torch learner
    (yeto/learner.py, yeto/diloco_sync.py, yeto/finalization.py) and the RL
    full-parameter role stream
    (yeto/rl/adapters/miles/full_parameter/role_stream.py) use.
    """

    def __init__(self, addrs: list[tuple[str, int]], learner_id: int, layout, dtype: int,
                 num_streams: int = 4, **client_kwargs) -> None:
        if len(addrs) < 2:
            raise ValueError("a sharded syncer head needs at least two addresses")
        count = len(addrs)
        self.layout = layout
        self.dtype = dtype
        self.learner_id = learner_id
        self.shards = [
            SyncerClient(addr, learner_id, layout, dtype, num_streams,
                         owned_fragments=owned_range(i, count, layout.num_fragments),
                         **client_kwargs)
            for i, addr in enumerate(addrs)
        ]
        # The island stops training only once every shard is done; a shard
        # that finishes early simply stops pulling.
        self.finalizing = _AllSet(c.finalizing for c in self.shards)
        self.finalized = _AllSet(c.finalized for c in self.shards)
        self.shutdown = _AllSet(c.shutdown for c in self.shards)
        self._manifests: list[FinalManifest] | None = None
        self._merged: FinalManifest | None = None
        self._init_sent = False

    @property
    def num_fragments(self) -> int:
        return self.layout.num_fragments

    def _owner(self, fragment_id: int) -> SyncerClient:
        return self.shards[owner_of(fragment_id, len(self.shards), self.layout.num_fragments)]

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        for client in self.shards:
            client.start()

    def close(self) -> None:
        for client in self.shards:
            client.close()

    def check_health(self) -> None:
        for client in self.shards:
            client.check_health()

    # -- routed sends -------------------------------------------------------

    def send_init(self, fragment_id: int, tensor_bytes: bytes) -> None:
        self._owner(fragment_id).send_init(fragment_id, tensor_bytes)

    def send_initial_fragments(
        self, parts_for: Callable[[int], Iterable[bytes | bytearray | memoryview]]
    ) -> bool:
        """INIT of every fragment, each to its owner.

        The first call sends to every shard; a later call (a replay after
        :meth:`init_replay_due`) resends only to the shards whose live group
        never carried a complete INIT, so one shard's outage does not resend
        the whole model."""
        targets = (self.shards if not self._init_sent
                   else [client for client in self.shards if client.init_replay_due()])
        self._init_sent = True
        return all([client.send_initial_fragments(parts_for) for client in targets])

    def init_replay_due(self) -> bool:
        """True when some shard needs INIT again (see ``SyncerClient``)."""
        return any(client.init_replay_due() for client in self.shards)

    def set_init_digests(self, layout_hash: bytes, digests) -> None:
        """Local init: each shard gets the digests of the fragments it owns."""
        for client in self.shards:
            client.set_init_digests(layout_hash, digests)

    def initial_cut_confirmed(self) -> bool:
        return all(client.initial_cut_confirmed() for client in self.shards)

    def mark_initial_cut_received(self) -> None:
        for client in self.shards:
            client.mark_initial_cut_received()

    def install_inbound_chunk_sink(self, sink: object | None) -> None:
        """One transactional sink for every shard. Each shard's receive
        loop opens its own transactions; a fragment arrives from exactly
        one shard, so transactions never overlap on a fragment."""
        for client in self.shards:
            client.install_inbound_chunk_sink(sink)

    def push_fragment(self, fragment_id: int, *args, **kwargs) -> None:
        self._owner(fragment_id).push_fragment(fragment_id, *args, **kwargs)

    push_timings_supported = True  # forwarded to the owner shard's SyncerClient

    @property
    def can_encode_ahead(self) -> bool:
        return all(client.can_encode_ahead for client in self.shards)

    def encode_push_parts(self, fragment_id: int, *args, **kwargs):
        return self._owner(fragment_id).encode_push_parts(fragment_id, *args, **kwargs)

    def push_fragment_parts(self, fragment_id: int, *args, **kwargs) -> bool:
        return self._owner(fragment_id).push_fragment_parts(fragment_id, *args, **kwargs)

    def heartbeat(self, local_step: int) -> None:
        for client in self.shards:
            client.heartbeat(local_step)

    # -- links ----------------------------------------------------------------

    def connection_generation_of(self, fragment_id: int) -> int:
        """Connection generation of the shard that owns ``fragment_id``: a
        PULL is answered on its owner's link, and only that shard's redial
        invalidates it."""
        return self._owner(fragment_id).connection_generation

    def send_progress(self) -> SendProgress:
        """Outbound progress summed over the shards; connected iff all are."""
        progress = [client.send_progress() for client in self.shards]
        return SendProgress(
            bytes_sent=sum(p.bytes_sent for p in progress),
            queued_chunks=sum(p.queued_chunks for p in progress),
            reconnects=sum(p.reconnects for p in progress),
            connected=all(p.connected for p in progress),
            stalls=sum(p.stalls for p in progress),
        )

    def stream_snapshots(self) -> tuple[StreamSnapshot, ...]:
        """Every shard's data streams, in shard order."""
        return tuple(snapshot for client in self.shards for snapshot in client.stream_snapshots())

    @property
    def upload_drained_at(self) -> float | None:
        """When the last shard's data streams became idle (None until one did)."""
        drained = [client.upload_drained_at for client in self.shards]
        return max((t for t in drained if t is not None), default=None)

    # -- merged receives ----------------------------------------------------

    def drain_pulls(self) -> list[PullRequest]:
        return [pull for client in self.shards for pull in client.drain_pulls()]

    def drain_updates(self) -> list[BcastFragment]:
        return [update for client in self.shards for update in client.drain_updates()]

    # -- finalization -------------------------------------------------------

    def wait_for_final_fragments(
        self, timeout: float | None = None
    ) -> tuple[FinalManifest, list[FinalFragment]]:
        """The union of every shard's final cut.

        Each shard's manifest lists all fragments but is authoritative only
        for the ones it owns; the merged manifest takes each fragment's
        version from its owner and the largest final step of any shard.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        manifests, fragments = [], []
        versions = [0] * self.layout.num_fragments
        for client in self.shards:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            manifest, ready = client.wait_for_final_fragments(remaining)
            manifests.append(manifest)
            fragments.extend(ready)
            for fid in client.owned_fragments:
                versions[fid] = manifest.versions[fid]
        merged = FinalManifest(max(m.global_step for m in manifests), tuple(versions))
        self._manifests, self._merged = manifests, merged
        return merged, sorted(fragments, key=lambda fragment: fragment.fragment_id)

    def wait_for_final_manifest(self, timeout: float | None = None) -> FinalManifest:
        """The union of every shard's final manifest (no final fragments):
        each fragment's version from its owner, the largest final step."""
        deadline = None if timeout is None else time.monotonic() + timeout
        manifests = []
        versions = [0] * self.layout.num_fragments
        for client in self.shards:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            manifest = client.wait_for_final_manifest(remaining)
            manifests.append(manifest)
            for fid in client.owned_fragments:
                versions[fid] = manifest.versions[fid]
        merged = FinalManifest(max(m.global_step for m in manifests), tuple(versions))
        self._manifests, self._merged = manifests, merged
        return merged

    def acknowledge_finalization(self, manifest: FinalManifest, timeout: float | None = None) -> None:
        if self._manifests is None or manifest != self._merged:
            raise RuntimeError("cannot acknowledge a final manifest that was not received")
        for client, own in zip(self.shards, self._manifests):
            client.acknowledge_finalization(own, timeout)
