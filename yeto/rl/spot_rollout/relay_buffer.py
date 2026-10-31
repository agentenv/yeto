"""Rollout groups through the sample relay (S27 G-EA).

Sampler side: :class:`RelaySampleBuffer` has the buffer surface of
:class:`~yeto.rl.engine.sample_buffer.FileSampleBuffer` that the unchanged
writers use (``GroupWriter``, the Miles buffer writer) and that
``SamplerLoop`` reads its backlog from. One relay item per group:

    b"YRG1" | u32 header length | header JSON | payload

``header``: ``sample_id``, ``behavior_version`` (oldest engine version of any
token: the policy version the trainer gates on), ``tokens``, ``provenance``,
``sampler_round``, ``sampler_id``, ``fragment_versions`` (the version vector
of the cut that generated the oldest tokens), ``cuts`` (S27 G-EM: for every
engine version in the group, the merged cut the sampler loaded for it:
``merged_version``, ``digest``, ``fragment_versions``), ``written_at``.
``payload``: the buffer payload of the group (``encode_group``:
``Sample.to_dict()`` mappings with per-token ``rollout_log_probs`` and
``metadata.weight_version``).

Trainer side: :class:`RelayPump` moves items into the trainer's own
:class:`~yeto.rl.engine.sample_buffer.FileSampleBuffer` on a local directory
(payload first, then the index entry), and acknowledges them only after the
append. The trainer's :class:`~yeto.rl.adapters.miles.buffer_rollout.BatchReader`
reads that buffer unchanged (policy age filter, prefetch, watermark). The
pump takes items only while the reader has fewer than ``prefetch_groups``
unread groups, so the relay depth a sampler sees is the trainer's real
backlog. A sample id already in the local buffer is acknowledged and
skipped (an item delivered again after a pump restart).

S27 G-EM: the pump writes each sample's external merged cut into its metadata
(``yeto.rl.engine.external_policy.CUT_METADATA_KEY``: the cut of the sample's
oldest engine version, from the item's ``cuts``), so the trainer's evidence
names the weights that generated every trajectory. An item without the cut of
one of its versions, or whose versions disagree with its header, is rejected
(acknowledged, not trained). The pump stops when the relay is gone
(:class:`~.relay.RelayGone`).
"""

from __future__ import annotations

import json
import os
import struct
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from yeto.rl.engine.buffer_payload import behavior_versions, decode_group, encode_group
from yeto.rl.engine.external_policy import CUT_METADATA_KEY, cut_record
from yeto.rl.engine.sample_buffer import PAYLOADS, SampleBufferError, SampleIndexEntry, payload_name

from .relay import PUT_FULL, RelayConsumer, RelayGone, RelayProducer, RelayStopped

MAGIC = b"YRG1"
HEADER_SCHEMA = "yeto.rl.relay-group/v1"

# Trainer-side environment (forwarded to the Miles rollout workers, buffer_rollout.ISLAND_ENVS).
RELAY_ADDR_ENV = "YETO_SAMPLE_RELAY_ADDR"  # host:port of the syncer that serves the relay
RELAY_CHANNEL_ENV = "YETO_SAMPLE_RELAY_CHANNEL"  # the training island's channel (its id)
RELAY_PREFETCH_ENV = "YETO_SAMPLE_RELAY_PREFETCH"  # groups the pump keeps unread in the local buffer
RELAY_ENVS = (RELAY_ADDR_ENV, RELAY_CHANNEL_ENV, RELAY_PREFETCH_ENV)


class RelayItemError(ValueError):
    pass


def encode_item(header: Mapping[str, Any], payload: bytes) -> bytes:
    raw = json.dumps(dict(header), sort_keys=True).encode()
    return MAGIC + struct.pack("<I", len(raw)) + raw + bytes(payload)


def decode_item(data: bytes) -> tuple[dict[str, Any], bytes]:
    if len(data) < 8 or data[:4] != MAGIC:
        raise RelayItemError("relay item does not start with YRG1")
    (length,) = struct.unpack_from("<I", data, 4)
    if 8 + length > len(data):
        raise RelayItemError("relay item header is truncated")
    header = json.loads(data[8:8 + length])
    if header.get("schema") != HEADER_SCHEMA:
        raise RelayItemError(f"relay item schema {header.get('schema')!r} != {HEADER_SCHEMA}")
    return header, data[8 + length:]


def parse_addr(text: str) -> tuple[str, int]:
    host, _, port = str(text).rpartition(":")
    if not host or not port.isdigit():
        raise ValueError(f"relay address must be host:port, got {text!r}")
    return host.strip("[]"), int(port)


class RelaySampleBuffer:
    """The sampler's sample buffer over the relay (module doc): the buffer surface
    that :class:`~yeto.rl.engine.buffer_payload.GroupWriter`,
    :class:`~yeto.rl.engine.sampler_loop.SamplerLoop`, ``sampler_resume`` and
    the Miles buffer writer use, with every group sent as one relay item.

    ``len()`` is the number of items the channel ever stored, ``read_position()``
    the number the trainer took (acknowledged), so the loop's backlog is the
    channel depth. ``entries()`` is empty (a new incarnation starts at round 0;
    its sample ids carry its incarnation). There is no data end: the run ends at
    the syncer."""

    shard = None

    def __init__(self, producer: RelayProducer, *, fragment_versions: Callable[[], Sequence[int]] = lambda: (),
                 should_stop: Callable[[], bool] = lambda: False, full_wait_s: float = 2.0,
                 clock: Callable[[], float] = time.time,
                 cut_of: Callable[[int], Mapping[str, Any] | None] | None = None) -> None:
        """``cut_of(version)``: the identity of the cut loaded for an engine
        version (``SubscriberReleaseStore.cut_identity``); None: items carry no
        ``cuts`` (a relay trainer rejects them)."""
        self.producer = producer
        self.fragment_versions = fragment_versions
        self.should_stop = should_stop
        self.full_wait_s = float(full_wait_s)
        self.clock = clock
        self.cut_of = cut_of
        self._payloads: dict[str, bytes] = {}
        self._depth = None
        self.written = 0
        self.dropped = 0  # groups not stored because the sampler is stopping
        self.uncut = 0  # groups not sent: a version without a known cut
        self.gone = False  # the relay did not answer for gone_s (RelayGone)
        # a stopping sampler ends the producer's retries of a broken link
        self.producer.should_stop = lambda: self.should_stop()

    # writing (GroupWriter: payload first, then the index entry)
    def write_payload(self, sample_id: str, data: bytes) -> str:
        ref = f"relay/{sample_id}"
        self._payloads[ref] = bytes(data)
        return ref

    def _cuts(self, payload: bytes) -> dict[str, dict[str, Any]] | None:
        """{engine version: cut identity} of every version in the group; None when
        one version has no known cut."""
        assert self.cut_of is not None
        versions = {v for sample in decode_group(payload) for v in behavior_versions(sample)}
        cuts = {}
        for version in sorted(versions):
            cut = self.cut_of(version)
            if cut is None:
                return None
            cuts[str(version)] = dict(cut)
        return cuts

    def append(self, entry: SampleIndexEntry) -> None:
        payload = self._payloads.pop(entry.payload_ref)
        header = {
            "schema": HEADER_SCHEMA, "sample_id": entry.sample_id,
            "behavior_version": int(entry.behavior_version), "tokens": int(entry.tokens),
            "provenance": None if entry.provenance is None else dict(entry.provenance),
            "sampler_round": None if entry.sampler_round is None else dict(entry.sampler_round),
            "fragment_versions": [int(v) for v in self.fragment_versions()],
            "written_at": float(self.clock()),
        }
        if self.cut_of is not None:
            cuts = self._cuts(payload)
            if cuts is None:
                self.uncut += 1
                if self.uncut == 1:
                    print(f"[yeto-relay] group {entry.sample_id} has an engine version without a loaded "
                          "cut; not sent (a relay trainer names the cut of every sample)", flush=True)
                return
            header["cuts"] = cuts
            header["fragment_versions"] = list(cuts[str(int(entry.behavior_version))]["fragment_versions"])
        if self.gone:
            self.dropped += 1
            return
        try:
            ack = self.producer.put_blocking(encode_item(header, payload), should_stop=self.should_stop,
                                             wait_s=self.full_wait_s)
        except RelayStopped:
            ack = None
        except RelayGone as error:
            self.gone = True
            print(f"[yeto-relay] {error}; the sampler stops writing", flush=True)
            ack = None
        if ack is None or ack.status == PUT_FULL:
            self.dropped += 1
            return
        self.written += 1
        self._depth = ack.depth

    @contextmanager
    def one_commit(self) -> Iterator[None]:
        yield

    def mark_data_end(self) -> None:
        """No data end on the relay: the trainers end with the syncer."""

    # reading (the loop's backlog and the resume point)
    def refresh(self) -> int:
        if self.gone:
            return len(self)
        try:
            self._depth = self.producer.depth()
        except RelayStopped:
            pass  # stopping: the last known depth stands
        except RelayGone as error:
            self.gone = True
            print(f"[yeto-relay] {error}; the sampler stops", flush=True)
        return len(self)

    def __len__(self) -> int:
        return 0 if self._depth is None else int(self._depth.next_id) - 1

    def read_position(self) -> int:
        depth = self._depth
        return 0 if depth is None else int(depth.next_id) - 1 - int(depth.items)

    def pending_groups(self) -> int:
        """The sampler loop's backlog on the relay: groups in the channel that no
        trainer took yet (S29 spot run, defect D5). The release-based backlog
        (``len - last_seen.buffer_selected``) counts from 0 until the first merged
        cut is loaded; a full-parameter run merges one fragment per outer step, so
        the first cut needs ``fragments`` trainer rounds, which need groups, while
        a sampler at ``max_backlog_groups`` waits for that cut: both sides wait."""
        depth = self._depth
        return 0 if depth is None else int(depth.items)

    def entries(self) -> list[SampleIndexEntry]:
        return []

    @property
    def data_ended(self) -> bool:
        return False


class RelayPump:
    """Relay channel -> the trainer's local sample buffer (module doc)."""

    def __init__(self, consumer: RelayConsumer, buffer: Any, *, reader: Any | None = None,
                 prefetch_groups: int = 64, wait_s: float = 5.0,
                 emit: Callable[..., None] | None = None, watchdog: Any | None = None) -> None:
        """``buffer``: the writer instance of the local buffer; ``reader``: the
        reader's instance (its ``read_cursor`` bounds the prefetch; None: the
        writer's own cursor); ``watchdog``: a ``boat_labels.OrphanWatchdog``
        (S29 D15) that sees every item's sampler incarnation and deletes the
        boat sandboxes of a spot island that left."""
        if int(prefetch_groups) < 1:
            raise ValueError("prefetch_groups must be >= 1")
        self.consumer = consumer
        self.buffer = buffer
        self.reader = reader
        self.prefetch_groups = int(prefetch_groups)
        self.wait_s = float(wait_s)
        self.emit = emit
        self.watchdog = watchdog
        self.appended = 0
        self.duplicates = 0
        self.rejected = 0
        self.error: BaseException | None = None
        self._known = {e.sample_id for e in buffer.entries()}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def unread(self) -> int:
        reader = self.reader if self.reader is not None else self.buffer
        try:
            reader.refresh()
        except Exception:  # noqa: BLE001 - a refresh failure only delays the next take
            pass
        return max(0, len(self.buffer) - int(reader.read_cursor))

    def step(self) -> int:
        """Take at most the free prefetch room once; returns groups appended."""
        self.buffer.refresh()
        room = self.prefetch_groups - self.unread()
        if room <= 0:
            return 0
        items = self.consumer.take(max_items=room, wait_s=self.wait_s)
        appended = 0
        for item in items:
            try:
                header, payload = decode_item(item.data)
            except (RelayItemError, ValueError) as error:
                self.rejected += 1
                if self.emit is not None:
                    self.emit("rl_relay_item_rejected", item_id=item.item_id, error=str(error)[:300])
                self.consumer.ack(item.item_id)
                continue
            if self.watchdog is not None:
                round_ = header.get("sampler_round")
                if isinstance(round_, Mapping):
                    self.watchdog.observe(round_.get("sampler"), round_.get("incarnation"))
            sample_id = str(header["sample_id"])
            if sample_id in self._known:
                self.duplicates += 1
                self.consumer.ack(item.item_id)
                continue
            try:
                payload = with_cut_metadata(header, payload)
            except (RelayItemError, ValueError, KeyError, TypeError) as error:
                self.rejected += 1
                print(f"[yeto-relay] item {item.item_id} ({sample_id}) rejected: {str(error)[:300]}", flush=True)
                if self.emit is not None:
                    self.emit("rl_relay_item_rejected", item_id=item.item_id, error=str(error)[:300])
                self.consumer.ack(item.item_id)
                continue
            try:
                ref = self.buffer.write_payload(sample_id, payload)
            except SampleBufferError:
                # A pump that stopped between the payload and its index entry:
                # the payload of this sample id is already there (same bytes).
                ref = f"{PAYLOADS}/{payload_name(sample_id)}"
                if not self.buffer.index.has_payload(ref):
                    raise
            self.buffer.append(SampleIndexEntry(
                sample_id=sample_id, behavior_version=int(header["behavior_version"]),
                payload_ref=ref, tokens=int(header["tokens"]),
                provenance=header.get("provenance"), sampler_round=header.get("sampler_round")))
            self._known.add(sample_id)
            self.consumer.ack(item.item_id)
            appended += 1
        self.appended += appended
        return appended

    def run(self) -> None:
        while not self._stop.is_set():
            if self.watchdog is not None:
                self.watchdog.poll()  # never raises (S29 D15)
            try:
                if self.step() == 0:
                    self._stop.wait(min(1.0, self.wait_s))
            except RelayGone as error:  # the head is gone: nothing more will arrive
                self.error = error
                print(f"[yeto-relay] pump stopped: {error}", flush=True)
                if self.emit is not None:
                    self.emit("rl_relay_pump_error", error=f"RelayGone: {str(error)[:300]}")
                return
            except Exception as error:  # noqa: BLE001 - recorded; the reader sees no new data
                self.error = error
                if self.emit is not None:
                    self.emit("rl_relay_pump_error", error=f"{type(error).__name__}: {str(error)[:300]}")
                self._stop.wait(5.0)

    def start(self) -> "RelayPump":
        self._thread = threading.Thread(target=self.run, name="yeto-relay-pump", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self.consumer.close()


def with_cut_metadata(header: Mapping[str, Any], payload: bytes) -> bytes:
    """The payload with every sample's external merged cut in its metadata
    (module doc): the cut of the sample's oldest engine version, from the
    item's ``cuts``. Raises when a cut is missing or the versions disagree."""
    cuts = header.get("cuts")
    if not isinstance(cuts, Mapping) or not cuts:
        raise RelayItemError("relay item names no merged cut (a sampler without cut identities)")
    samples = decode_group(payload)
    oldest_of_group = None
    for sample in samples:
        oldest = min(behavior_versions(sample))
        cut = cuts.get(str(oldest))
        if not isinstance(cut, Mapping) or int(cut.get("merged_version", -1)) != oldest:
            raise RelayItemError(f"relay item names no merged cut for engine version {oldest}")
        meta = sample.setdefault("metadata", {})
        if not isinstance(meta, dict):
            raise RelayItemError("relay sample metadata is not an object")
        meta[CUT_METADATA_KEY] = cut_record(oldest, str(cut.get("digest")))
        oldest_of_group = oldest if oldest_of_group is None else min(oldest_of_group, oldest)
    if oldest_of_group != int(header["behavior_version"]):
        raise RelayItemError(f"relay item behavior version {header['behavior_version']} != its samples' "
                             f"oldest version {oldest_of_group}")
    return encode_group(samples)


SPOT_SYNCER_ENV = "YETO_SPOT_SYNCER"  # sampler island: subscribe to this syncer (host:port)


def trainer_relay_env(syncer: tuple[str, int], learner_id: int, *, groups_per_round: int,
                      prefetch: int | None = None) -> dict[str, str]:
    """The relay variables of a training island (``--rl-sample-relay``; both backends,
    S27 G-EA / G-EV): the syncer that serves the relay, the island's channel (its
    learner id) and the groups the pump keeps unread (default two rounds)."""
    host, port = syncer
    if prefetch is None:
        prefetch = 2 * int(groups_per_round)
    return {RELAY_ADDR_ENV: f"{host}:{port}", RELAY_CHANNEL_ENV: str(int(learner_id)),
            RELAY_PREFETCH_ENV: str(int(prefetch))}


def relay_sampler_buffer(environ: Mapping[str, str] | None = None) -> "RelaySampleBuffer | None":
    """A sampler's buffer over the relay when :data:`RELAY_ADDR_ENV` is set (the
    Miles sampler island's ``open_buffer(sampler=True)``); None otherwise. The
    producer id is the sampler's subscriber id (``release_store.subscriber_id_for``)."""
    env = os.environ if environ is None else environ
    addr = env.get(RELAY_ADDR_ENV)
    if not addr:
        return None
    from yeto.rl.engine.task_dispense import member_incarnation

    from .release_store import subscriber_id_for

    sampler_id = env.get("YETO_SAMPLE_BUFFER_SAMPLER_ID") or "sampler"
    # the sampler process's member incarnation (YETO_SAMPLER_INCARNATION, which
    # the rollout workers inherit): the one its subscriber and dispenser use
    producer = RelayProducer(parse_addr(addr), int(env.get(RELAY_CHANNEL_ENV, "0")),
                             subscriber_id_for(sampler_id), incarnation=member_incarnation(env))
    return RelaySampleBuffer(producer)


def maybe_start_relay_pump(reader_buffer: Any, environ: Mapping[str, str] | None = None) -> RelayPump | None:
    """Start the pump for a trainer's buffer reader when :data:`RELAY_ADDR_ENV`
    is set (None otherwise: the buffer is read as before). The pump writes
    through its own buffer instance on the reader's store."""
    env = os.environ if environ is None else environ
    addr = env.get(RELAY_ADDR_ENV)
    if not addr:
        return None
    from yeto.rl.engine.sample_buffer import FileSampleBuffer

    channel = int(env.get(RELAY_CHANNEL_ENV, "0"))
    prefetch = int(env.get(RELAY_PREFETCH_ENV, "64"))
    writer = FileSampleBuffer(reader_buffer.store, batch_size=1)
    from yeto.rl.harness.harbor.boat_labels import OrphanWatchdog

    watchdog = OrphanWatchdog.from_env(env)  # S29 D15: None without YETO_BOAT_RUN_ID
    pump = RelayPump(RelayConsumer(parse_addr(addr), channel), writer, reader=reader_buffer,
                     prefetch_groups=prefetch, watchdog=watchdog)
    if watchdog is not None:
        print(f"[yeto-relay] boat orphan watchdog: run {watchdog.run} lease {watchdog.lease_s:.0f} s",
              flush=True)
    print(f"[yeto-relay] pump started: relay {addr} channel {channel} prefetch {prefetch} groups",
          flush=True)
    return pump.start()
