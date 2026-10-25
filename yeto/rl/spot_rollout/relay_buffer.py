"""Rollout groups through the sample relay (S27 G-EA).

Sampler side: :class:`RelayGroupWriter` has the writer surface of
:class:`~yeto.rl.engine.buffer_payload.GroupWriter` (``sampler_id``,
``write(samples, group_key=, provenance=, sampler_round=)``), so
:class:`~yeto.rl.engine.engine_sampler.EngineRound` writes to the relay as it
writes to a buffer. One relay item per group:

    b"YRG1" | u32 header length | header JSON | payload

``header``: ``sample_id``, ``behavior_version`` (oldest engine version of any
token: the policy version the trainer gates on), ``tokens``, ``provenance``,
``sampler_round``, ``sampler_id``, ``fragment_versions`` (the subscriber's
version vector when it generated), ``written_at``. ``payload``: the buffer
payload of the group (``encode_group``: ``Sample.to_dict()`` mappings with
per-token ``rollout_log_probs`` and ``metadata.weight_version``).

Trainer side: :class:`RelayPump` moves items into the trainer's own
:class:`~yeto.rl.engine.sample_buffer.FileSampleBuffer` on a local directory
(payload first, then the index entry), and acknowledges them only after the
append. The trainer's :class:`~yeto.rl.adapters.miles.buffer_rollout.BatchReader`
reads that buffer unchanged (policy age filter, prefetch, watermark). The
pump takes items only while the reader has fewer than ``prefetch_groups``
unread groups, so the relay depth a sampler sees is the trainer's real
backlog. A sample id already in the local buffer is acknowledged and
skipped (an item delivered again after a pump restart).
"""

from __future__ import annotations

import json
import os
import struct
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from yeto.rl.engine.buffer_payload import encode_group, group_entry_fields
from yeto.rl.engine.sample_buffer import PAYLOADS, SampleBufferError, SampleIndexEntry, payload_name

from .relay import PUT_FULL, RelayConsumer, RelayProducer

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


class RelayGroupWriter:
    """``GroupWriter`` surface over a :class:`RelayProducer` (module doc)."""

    def __init__(self, producer: RelayProducer, sampler_id: str, *,
                 fragment_versions: Callable[[], Sequence[int]] = lambda: (),
                 should_stop: Callable[[], bool] = lambda: False,
                 full_wait_s: float = 2.0, clock: Callable[[], float] = time.time) -> None:
        if not sampler_id:
            raise ValueError("sampler_id must be non-empty")
        self.producer = producer
        self.sampler_id = str(sampler_id)
        self.fragment_versions = fragment_versions
        self.should_stop = should_stop
        self.full_wait_s = float(full_wait_s)
        self.clock = clock
        self.written = 0
        self.dropped = 0  # groups not stored because the sampler is stopping

    def write(self, samples: Sequence[Mapping[str, Any]], *, group_key: str,
              provenance: Mapping[str, str] | None = None,
              sampler_round: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
        behavior_version, tokens = group_entry_fields(samples)
        header = {
            "schema": HEADER_SCHEMA, "sample_id": f"{self.sampler_id}-{group_key}",
            "behavior_version": int(behavior_version), "tokens": int(tokens),
            "provenance": None if provenance is None else dict(provenance),
            "sampler_round": None if sampler_round is None else dict(sampler_round),
            "sampler_id": self.sampler_id,
            "fragment_versions": [int(v) for v in self.fragment_versions()],
            "written_at": float(self.clock()),
        }
        ack = self.producer.put_blocking(encode_item(header, encode_group(samples)),
                                         should_stop=self.should_stop, wait_s=self.full_wait_s)
        if ack is None or ack.status == PUT_FULL:
            self.dropped += 1
            return None
        self.written += 1
        return header

    def backlog(self) -> int:
        """Groups in the channel the trainer has not taken (last answer)."""
        depth = self.producer.last_depth
        return 0 if depth is None else int(depth.items)


class RelayPump:
    """Relay channel -> the trainer's local sample buffer (module doc)."""

    def __init__(self, consumer: RelayConsumer, buffer: Any, *, reader: Any | None = None,
                 prefetch_groups: int = 64, wait_s: float = 5.0,
                 emit: Callable[..., None] | None = None) -> None:
        """``buffer``: the writer instance of the local buffer; ``reader``: the
        reader's instance (its ``read_cursor`` bounds the prefetch; None: the
        writer's own cursor)."""
        if int(prefetch_groups) < 1:
            raise ValueError("prefetch_groups must be >= 1")
        self.consumer = consumer
        self.buffer = buffer
        self.reader = reader
        self.prefetch_groups = int(prefetch_groups)
        self.wait_s = float(wait_s)
        self.emit = emit
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
            sample_id = str(header["sample_id"])
            if sample_id in self._known:
                self.duplicates += 1
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
            try:
                if self.step() == 0:
                    self._stop.wait(min(1.0, self.wait_s))
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
    pump = RelayPump(RelayConsumer(parse_addr(addr), channel), writer, reader=reader_buffer,
                     prefetch_groups=prefetch)
    print(f"[yeto-relay] pump started: relay {addr} channel {channel} prefetch {prefetch} groups",
          flush=True)
    return pump.start()
