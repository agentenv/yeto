"""Receive-only subscriber of the global weights (S27 G-EA; syncer ``subscriber.rs``).

A sampler island on any cloud follows the merged weights without being a
learner. It opens the usual connection group (control socket + striped data
streams) with SUBSCRIBE instead of HELLO:

* body = the HELLO body (``encode_hello``) with learner id
  ``SUBSCRIBER_ID_BIT | 31 bits`` (``release_store.subscriber_id_for``: the task
  dispenser's sampler holder id) and session dtype bf16 (2), then the trailer
  ``YSI1 | incarnation u64`` (the process's member incarnation,
  ``task_dispense.member_incarnation``), sealed with the subscribe MAC domain;
  the syncer admits at most ``--max-subscribers`` at once;
* the syncer checks the layout and fingerprint against the running session
  (not the session contract or the compat group: a subscriber never merges,
  so it may run on another card type), and never puts it in the roster:
  no quorum, no lease, no final cut, no PULL;
* it then receives ``BCAST_FRAGMENT`` in the bf16 session format
  (``fragment u32 | version u64 | bf16 values``): every fragment at its
  current version once (at the first merge after it connected), then each
  merged version; a slow link skips versions (newest wins per fragment);
* at the end of the run the syncer sends ERROR ``subscriber: run finished``
  (S27 G-EM: the client keeps that notice even when it arrives on a link the
  client is already tearing down, and then stops redialing).

A reconnect (new generation) gets every fragment again. The session is
never blocked by a subscriber: the syncer only queues versions for it.

Members (docs/PROTOCOL.md "Sampler members"): the subscriber is the member
``(id, incarnation)`` of the syncer's island member table, shared with the
task dispenser and the sample relay. A HEARTBEAT every ``heartbeat_s``
renews its lease. A newer incarnation of the same sampler (on any of the
three channels) retires it: the syncer sends ERROR ``fenced: subscriber ...``
and the client stops (:meth:`SubscriberClient.fenced`); a fenced process
does not come back under the same incarnation.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from yeto.fragments import FragmentLayout
from yeto.protocol import (
    DTYPE_BF16,
    MSG_ERROR,
    MSG_SUBSCRIBE,
    SUBSCRIBE_MAC_DOMAIN,
    SUBSCRIBER_ID_BIT,
    BcastFragment,
    SyncerClient,
    encode_subscribe_incarnation,
)
from yeto.rl.engine.external_policy import fragment_digest
from yeto.rl.engine.full_parameter_sync import merged_version

RUN_FINISHED = "subscriber: run finished"
FENCED_PREFIX = "fenced: subscriber"  # syncer subscriber.rs FENCED_PREFIX
DEFAULT_HEARTBEAT_S = 10.0  # well under the member lease (--dispense-lease-s, 300 s)


class SubscriberClient(SyncerClient):
    """A :class:`SyncerClient` that subscribes instead of joining."""

    _hello_msg_type = MSG_SUBSCRIBE
    _hello_mac_domain = SUBSCRIBE_MAC_DOMAIN

    def __init__(self, addr: tuple[str, int], subscriber_id: int, layout: FragmentLayout, *,
                 num_streams: int = 4, connect_timeout: float = 900.0,
                 hmac_key: bytes | None = None, syncer_profile_hash: bytes | None = None,
                 incarnation: int | None = None, heartbeat_s: float | None = DEFAULT_HEARTBEAT_S,
                 **kwargs) -> None:
        """``subscriber_id``: the wire id, ``SUBSCRIBER_ID_BIT`` set
        (``release_store.subscriber_id_for(sampler_id)``). ``incarnation``: the
        process's member incarnation (default
        :func:`~yeto.rl.engine.task_dispense.member_incarnation`, the one its task
        dispenser and relay connections use). ``heartbeat_s``: lease renewal
        period (None: no heartbeats, for tests)."""
        if not SUBSCRIBER_ID_BIT <= int(subscriber_id) <= 0xFFFF_FFFF:
            raise ValueError(f"subscriber_id must have the subscriber bit 2^31 set, got {subscriber_id}")
        if incarnation is None:
            from yeto.rl.engine.task_dispense import member_incarnation

            incarnation = member_incarnation()
        self.incarnation = int(incarnation)
        self._trailer = encode_subscribe_incarnation(self.incarnation)
        super().__init__(addr, int(subscriber_id), layout, dtype=DTYPE_BF16,
                         num_streams=num_streams, connect_timeout=connect_timeout,
                         hmac_key=hmac_key, syncer_profile_hash=syncer_profile_hash, **kwargs)
        self.subscriber_id = int(subscriber_id)
        self.heartbeat_s = None if heartbeat_s is None else float(heartbeat_s)
        self._beats = threading.Event()
        self._run_finished = threading.Event()

    def _dispatch(self, gen: int, msg_type: int, payload) -> None:
        # S27 G-EM: the end-of-run notice counts on any connection generation. The
        # syncer closes the data streams right after it; when their end reaches
        # the client first, the client starts a new generation and the base class
        # would drop the notice as a late message (A run s27-ea-a1: the sampler
        # kept redialing a finished head).
        if msg_type == MSG_ERROR and RUN_FINISHED in bytes(payload).decode("utf-8", errors="replace"):
            self._run_finished.set()
            self._closed.set()  # no redial: the run is over
            self._failure.set()
        super()._dispatch(gen, msg_type, payload)

    def _hello_body_suffix(self) -> bytes:
        return self._trailer

    def start(self) -> None:
        super().start()
        if self.heartbeat_s is not None:
            threading.Thread(target=self._heartbeat_loop, name="subscriber-heartbeat", daemon=True).start()

    def _heartbeat_loop(self) -> None:
        # The member lease is shared with the process's other channels; a
        # subscriber has nothing else to send, so it renews it here.
        while not self._beats.wait(self.heartbeat_s):
            if self.run_finished() or self.fenced():
                return
            try:
                self.heartbeat(0)
            except Exception:  # noqa: BLE001 - the client's supervisor redials or reports
                pass

    def close(self) -> None:
        self._beats.set()
        super().close()

    def _error_text(self) -> str:
        error = getattr(self, "_err", None) or getattr(self, "_last_err", None)
        return "" if error is None else str(error)

    def run_finished(self) -> bool:
        """True once the syncer said the run ended (the client is closed)."""
        return self._run_finished.is_set() or RUN_FINISHED in self._error_text()

    def fenced(self) -> bool:
        """True once the syncer retired this member (a newer incarnation of the
        sampler joined, or its lease expired; the client is closed)."""
        return FENCED_PREFIX in self._error_text()


@dataclass(frozen=True)
class WeightCut:
    """The newest bf16 bytes of every fragment and their versions."""

    versions: tuple[int, ...]
    fragments: tuple[bytes, ...]
    # S27 G-EM: SHA-256 of each fragment's bytes (empty: not computed)
    digests: tuple[bytes, ...] = ()

    @property
    def merged_version(self) -> int:
        return merged_version(self.versions)


class WeightFollower:
    """Collect the fragments a :class:`SubscriberClient` receives.

    ``cut()`` returns a :class:`WeightCut` once every fragment arrived at least
    once; ``merged_version`` (:func:`~yeto.rl.engine.full_parameter_sync.merged_version`:
    merges every fragment has received) is the policy version of that cut, the
    unit synced trainers gate buffered samples on.
    """

    def __init__(self, client: SyncerClient, num_fragments: int, *,
                 on_update: Callable[[BcastFragment], None] | None = None) -> None:
        self.client = client
        self.num_fragments = int(num_fragments)
        self._data: list[bytes | None] = [None] * self.num_fragments
        self._versions: list[int] = [-1] * self.num_fragments
        self._digests: list[bytes | None] = [None] * self.num_fragments
        self._lock = threading.Lock()
        self.on_update = on_update
        self.received = 0
        self.received_bytes = 0
        self.first_at: float | None = None
        self.complete_at: float | None = None

    def drain(self) -> int:
        """Take every queued update; returns how many were newer than held."""
        taken = 0
        for update in self.client.drain_updates():
            with self._lock:
                if update.version <= self._versions[update.fragment_id]:
                    continue
            data = bytes(update.data)
            digest = fragment_digest(data)  # outside the lock: about 1 s per GB
            with self._lock:
                if update.version <= self._versions[update.fragment_id]:
                    continue
                self._versions[update.fragment_id] = int(update.version)
                self._data[update.fragment_id] = data
                self._digests[update.fragment_id] = digest
                taken += 1
                self.received += 1
                self.received_bytes += len(update.data)
                if self.first_at is None:
                    self.first_at = time.monotonic()
                if self.complete_at is None and all(d is not None for d in self._data):
                    self.complete_at = time.monotonic()
            if self.on_update is not None:
                self.on_update(update)
        return taken

    @property
    def complete(self) -> bool:
        with self._lock:
            return all(d is not None for d in self._data)

    @property
    def versions(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(self._versions)

    def merged_version(self) -> int | None:
        with self._lock:
            if any(v < 0 for v in self._versions):
                return None
            return merged_version(self._versions)

    def cut(self) -> WeightCut | None:
        with self._lock:
            if any(d is None for d in self._data):
                return None
            return WeightCut(tuple(self._versions), tuple(d for d in self._data if d is not None),
                             tuple(d for d in self._digests if d is not None))

    def wait_complete(self, timeout_s: float, *, poll_s: float = 0.2,
                      should_stop: Callable[[], bool] = lambda: False) -> WeightCut | None:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline and not should_stop():
            self.drain()
            cut = self.cut()
            if cut is not None:
                return cut
            self.client.check_health()
            time.sleep(poll_s)
        return None


def bf16_fragment_tensors(layout: FragmentLayout, fragment_id: int, data: bytes) -> list[tuple[str, object]]:
    """Split one bf16 fragment payload into ``(wire name, flat torch bf16 tensor)``
    in layout order (the payload concatenates the tensors, ``yeto/fragments.py``)."""
    import torch

    fragment = layout.fragments[fragment_id]
    flat = torch.frombuffer(bytearray(data), dtype=torch.bfloat16)
    if flat.numel() != fragment.numel:
        raise ValueError(f"fragment {fragment_id}: {flat.numel()} bf16 values, layout has {fragment.numel}")
    out, offset = [], 0
    for name, numel in fragment.tensors:
        out.append((name, flat[offset:offset + numel]))
        offset += numel
    return out


def fragment_versions_text(versions: Sequence[int]) -> str:
    """Short text of a version vector for logs (min..max and count)."""
    if not versions:
        return "-"
    return f"{min(versions)}..{max(versions)} over {len(versions)} fragments"
