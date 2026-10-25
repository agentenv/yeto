"""Receive-only subscriber of the global weights (S27 G-EA; syncer ``subscriber.rs``).

A sampler island on any cloud follows the merged weights without being a
learner. It opens the usual connection group (control socket + striped data
streams) with SUBSCRIBE instead of HELLO:

* body = the HELLO body (``encode_hello``) with learner id
  ``SUBSCRIBER_ID_BIT | subscriber_id`` and session dtype bf16 (2), sealed with
  the subscribe MAC domain;
* the syncer checks the layout and fingerprint against the running session
  (not the session contract or the compat group: a subscriber never merges,
  so it may run on another card type), and never puts it in the roster:
  no quorum, no lease, no final cut, no PULL;
* it then receives ``BCAST_FRAGMENT`` in the bf16 session format
  (``fragment u32 | version u64 | bf16 values``): every fragment at its
  current version once (at the first merge after it connected), then each
  merged version; a slow link skips versions (newest wins per fragment);
* at the end of the run the syncer sends ERROR ``subscriber: run finished``.

A reconnect (new generation) gets every fragment again. The session is
never blocked by a subscriber: the syncer only queues versions for it.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from yeto.fragments import FragmentLayout
from yeto.protocol import (
    DTYPE_BF16,
    MSG_SUBSCRIBE,
    SUBSCRIBE_MAC_DOMAIN,
    SUBSCRIBER_ID_BIT,
    BcastFragment,
    SyncerClient,
)
from yeto.rl.engine.full_parameter_sync import merged_version

RUN_FINISHED = "subscriber: run finished"


class SubscriberClient(SyncerClient):
    """A :class:`SyncerClient` that subscribes instead of joining."""

    _hello_msg_type = MSG_SUBSCRIBE
    _hello_mac_domain = SUBSCRIBE_MAC_DOMAIN

    def __init__(self, addr: tuple[str, int], subscriber_id: int, layout: FragmentLayout, *,
                 num_streams: int = 4, connect_timeout: float = 900.0,
                 hmac_key: bytes | None = None, syncer_profile_hash: bytes | None = None,
                 **kwargs) -> None:
        if not 0 <= int(subscriber_id) < SUBSCRIBER_ID_BIT:
            raise ValueError(f"subscriber_id must be in [0, 2^31), got {subscriber_id}")
        super().__init__(addr, SUBSCRIBER_ID_BIT | int(subscriber_id), layout, dtype=DTYPE_BF16,
                         num_streams=num_streams, connect_timeout=connect_timeout,
                         hmac_key=hmac_key, syncer_profile_hash=syncer_profile_hash, **kwargs)
        self.subscriber_id = int(subscriber_id)

    def run_finished(self) -> bool:
        """True once the syncer said the run ended (the client is closed)."""
        error = getattr(self, "_err", None) or getattr(self, "_last_err", None)
        return error is not None and RUN_FINISHED in str(error)


@dataclass(frozen=True)
class WeightCut:
    """The newest bf16 bytes of every fragment and their versions."""

    versions: tuple[int, ...]
    fragments: tuple[bytes, ...]

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
                self._versions[update.fragment_id] = int(update.version)
                self._data[update.fragment_id] = bytes(update.data)
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
            return WeightCut(tuple(self._versions), tuple(d for d in self._data if d is not None))

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
