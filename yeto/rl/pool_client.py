"""Island side of the island pool: one RL island on the fragment-round syncer
with dynamic membership (leases, re-JOIN, capacity quorum, soft deadline,
carry discount, final grace window).

The wire protocol is the ordinary fragment protocol (``yeto.protocol``): the
canonical LoRA vector is one f32 fragment (``core.build_avg_layout``). This
module only adds what an island in a dynamic pool needs on top of
:class:`yeto.protocol.SyncerClient`:

* heartbeats every ``lease_s / 3`` so the syncer keeps the island's lease;
* a pushed delta answers the newest PULL the island has not answered yet, so
  a late island's delta is carried into the next round (the syncer applies
  ``--max-base-lag`` and the carry discount);
* bases are read from broadcasts, and the final cut (FINAL_MANIFEST) is
  acknowledged and handed over as the last base;
* a lease expiry makes the client redial with a new generation (a re-JOIN);
  each one goes on the island tape as ``elastic_rejoin``.

Tape event names (``elastic_*``) are kept: dashboards and the launcher read them.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import torch

from ..protocol import DTYPE_F32, FragmentLayout, PullRequest, SyncerClient
from ..tensor_io import pack_tensor, unpack_fragment


class LeaveReason(IntEnum):
    """Why an island left the pool (one byte of MSG_LEAVE; the syncer's
    ``pool_leave`` record names it, see ``protocol.rs`` LEAVE_REASON_NAMES).
    ``LEASE_EXPIRED`` is assigned by the syncer only."""

    FINISHED = 0
    LEASE_EXPIRED = 1
    STOPPED = 2
    PREEMPTED = 3
    INFRA_FAILED = 4
    TRAINING_FAILED = 5
    CONFIG_MISMATCH = 6
    # S29 D4: the member is not part of this run any more: a newer incarnation
    # took its id ("superseded") or a fresh syncer replaced the run it joined
    # ("other run"). An older syncer names the byte unknown_7.
    FENCED = 7


# Island lease on the syncer; the island heartbeats every lease/3. Same
# default as the launcher's --rl-sync-island-lease-s.
DEFAULT_ISLAND_LEASE_S = 150.0
# Consecutive failed redials (lease re-JOIN or lost link) before the island gives up.
DEFAULT_MAX_REJOIN_FAILURES = 5

DEBUG_DELAY_ENV = "YETO_RL_ELASTIC_DEBUG_DELAY"
DEBUG_PAUSE_ENV = "YETO_RL_ELASTIC_DEBUG_PAUSE"


def parse_elastic_debug_delay(spec: str | None) -> dict[int, float]:
    """``"ISLAND:SECONDS[,ISLAND:SECONDS]"`` (bare ``SECONDS`` = every island)."""
    out: dict[int, float] = {}
    for item in (spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        island, sep, seconds = item.rpartition(":")
        key = int(island) if sep else -1
        value = float(seconds)
        if value < 0 or (sep and key < 0):
            raise ValueError(f"--rl-elastic-debug-delay-s: bad item {item!r}")
        out[key] = value
    return out


def parse_elastic_debug_pause(spec: str | None) -> dict[int, tuple[int, float]]:
    """``"ISLAND:AFTER_V:SECONDS[,...]"``: once the island applied a base with
    outer version >= AFTER_V, it stops heartbeating and holds its pushes for
    SECONDS while the process keeps training (exercises lease expiry and the
    automatic re-JOIN)."""
    out: dict[int, tuple[int, float]] = {}
    for item in (spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        try:
            if len(parts) != 3:
                raise ValueError
            island, after_v, seconds = int(parts[0]), int(parts[1]), float(parts[2])
        except (ValueError, IndexError):
            raise ValueError(f"--rl-elastic-debug-pause: bad item {item!r} (want ISLAND:AFTER_V:SECONDS)") from None
        if island < 0 or after_v < 0 or seconds <= 0:
            raise ValueError(f"--rl-elastic-debug-pause: bad item {item!r}")
        out[island] = (after_v, seconds)
    return out


class LeaseHeartbeat:
    """Keep one island's lease on the syncer: a daemon thread sends a
    heartbeat every ``lease_s / 3`` on ``client`` (a :class:`SyncerClient`).

    A lease expiry is handled by the client itself (it redials and the
    syncer treats the new generation as a re-JOIN); ``on_tick`` runs before
    each heartbeat (e.g. to log re-JOINs) and ``paused`` lets a test hold
    the heartbeats. ``progress`` is the local step reported in each one."""

    def __init__(self, client: Any, lease_s: float, *,
                 progress: Callable[[], int] = lambda: 0,
                 paused: Callable[[], bool] = lambda: False,
                 on_tick: Callable[[], None] = lambda: None,
                 name: str = "pool-heartbeat") -> None:
        if not lease_s > 0:
            raise ValueError("island lease must be > 0 seconds")
        self.client = client
        self.period_s = float(lease_s) / 3.0
        self.errors: list[str] = []
        self._progress = progress
        self._paused = paused
        self._on_tick = on_tick
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name=name, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def _loop(self) -> None:
        while not self._stop.wait(self.period_s):
            self._on_tick()
            if self._paused():
                continue
            try:
                self.client.heartbeat(int(self._progress()))
            except Exception as error:  # noqa: BLE001 - the client's supervisor redials
                self.errors.append(f"heartbeat: {error}")


class PoolFinished(Exception):
    """The syncer finished the run (final cut sent) while this island was
    still training or pushing."""

    def __init__(self, final_outer_version: int, reason: str = "finished") -> None:
        super().__init__(f"syncer finished at outer version {final_outer_version} ({reason})")
        self.final_outer_version = int(final_outer_version)
        self.reason = reason


@dataclass(frozen=True)
class JoinAck:
    """What the island learned when it joined: the syncer's current version
    and whether it joined a run already in progress."""

    base_version: int
    catch_up: bool


@dataclass(frozen=True)
class PoolBase:
    """A global LoRA vector (flat f32 in layout order) at ``outer_version``."""

    params: torch.Tensor
    outer_version: int
    final: bool = False


@dataclass
class PoolClientConfig:
    syncer_addr: tuple[str, int]
    island_id: int
    lease_s: float = DEFAULT_ISLAND_LEASE_S
    syncer_epoch: int = 0
    session_contract_hash: bytes | None = None
    compat_group: str | None = None
    wan_streams: int = 4
    join_timeout_s: float = 900.0
    max_rejoin_failures: int = DEFAULT_MAX_REJOIN_FAILURES


class PoolIslandClient:
    """Heartbeat, PULL bookkeeping, base tracking and finalization for one
    island over :class:`SyncerClient`."""

    FRAGMENT = 0

    def __init__(self, config: PoolClientConfig, layout: FragmentLayout, *,
                 on_event: Callable[[dict], None] | None = None,
                 client_factory: Callable[..., Any] = SyncerClient) -> None:
        if layout.num_fragments != 1:
            raise ValueError("the island pool syncs the LoRA vector as one fragment")
        self.config = config
        self.layout = layout
        self._on_event = on_event or (lambda ev: None)
        self.client = client_factory(
            config.syncer_addr, int(config.island_id), layout, dtype=DTYPE_F32,
            num_streams=config.wan_streams, session_contract_hash=config.session_contract_hash,
            compat_group=config.compat_group, max_reconnects=config.max_rejoin_failures,
            syncer_epoch=int(config.syncer_epoch))
        self.latest_base: PoolBase | None = None
        # L5.6: (Unix time the base became visible, seconds from receipt to visible)
        self.latest_base_timing: tuple[float, float] | None = None
        self.inner_step = 0
        self.final_outer_version: int | None = None
        self.finished: PoolFinished | None = None
        self._permits: dict[int, PullRequest] = {}
        self._answered: set[tuple[int, int]] = set()  # (global_step, round_attempt)
        self._stop = threading.Event()
        self._pause_until = 0.0
        self._rejoins_seen = 0
        self._heartbeat = LeaseHeartbeat(self.client, config.lease_s,
                                         progress=lambda: self.inner_step,
                                         paused=self.link_paused, on_tick=self._note_rejoins)
        self.errors = self._heartbeat.errors

    # -- lifecycle -----------------------------------------------------------
    def join(self) -> JoinAck:
        """Connect (HELLO) and start heartbeating. The base version is the
        first broadcast already received (0 before the run starts)."""
        self.client.start()
        self._heartbeat.start()
        self._drain()
        version = 0 if self.latest_base is None else self.latest_base.outer_version
        return JoinAck(base_version=version, catch_up=version > 0)

    def leave(self, reason: int = LeaveReason.FINISHED) -> None:
        """Leave the pool: LEAVE with the reason, then close the connection."""
        self._stop.set()
        if not self.client.shutdown.is_set():
            self.client.send_leave(int(reason))
        self.close()

    def close(self) -> None:
        self._stop.set()
        self.client.close()
        self._heartbeat.stop()

    # -- link ----------------------------------------------------------------
    @property
    def heartbeat_period_s(self) -> float:
        return self.config.lease_s / 3.0

    def pause_link(self, seconds: float) -> None:
        """Test switch: no heartbeats and no pushes for ``seconds``."""
        start = time.time()
        self._pause_until = time.monotonic() + float(seconds)
        self._on_event({"event": "elastic_debug_pause", "pause_s": float(seconds), "start_unix": start})

        def end() -> None:
            if not self._stop.wait(float(seconds)):
                self._on_event({"event": "elastic_debug_pause_end", "pause_s": float(seconds),
                                "start_unix": start, "end_unix": time.time()})
        threading.Thread(target=end, name="pool-debug-pause", daemon=True).start()

    def link_paused(self) -> bool:
        return time.monotonic() < self._pause_until

    def _note_rejoins(self) -> None:
        count = int(getattr(self.client, "lease_expiries", 0))
        while self._rejoins_seen < count:
            self._rejoins_seen += 1
            self._on_event({"event": "elastic_rejoin", "rejoins": self._rejoins_seen,
                            "syncer_epoch": int(self.config.syncer_epoch)})

    # -- bases ---------------------------------------------------------------
    def _drain(self) -> None:
        self.client.check_health()
        self._note_rejoins()
        fragment = self.layout.fragments[self.FRAGMENT]
        for update in self.client.drain_updates():
            if update.fragment_id != self.FRAGMENT:
                raise RuntimeError(f"island pool received fragment {update.fragment_id}")
            if self.latest_base is not None and update.version <= self.latest_base.outer_version:
                continue
            params = unpack_fragment(fragment, update.data, DTYPE_F32)
            self._set_base(PoolBase(params, int(update.version)),
                           max(0.0, time.monotonic() - update.received_at))
        for permit in self.client.drain_pulls():
            if permit.fragment_id != self.FRAGMENT:
                raise RuntimeError(f"island pool received a PULL for fragment {permit.fragment_id}")
            # A restarted round (higher attempt) needs a fresh answer.
            if (permit.global_step, permit.round_attempt) not in self._answered:
                previous = self._permits.get(permit.global_step)
                if previous is None or permit.round_attempt > previous.round_attempt:
                    self._permits[permit.global_step] = permit
        if self.client.finalizing.is_set() and self.finished is None:
            self._take_final_cut()

    def _set_base(self, base: PoolBase, recv_seconds: float) -> None:
        self.latest_base = base
        self.latest_base_timing = (time.time(), recv_seconds)

    def _take_final_cut(self) -> None:
        manifest, fragments = self.client.wait_for_final_fragments()
        final = fragments[0]
        params = unpack_fragment(self.layout.fragments[self.FRAGMENT], final.data, DTYPE_F32)
        self._set_base(PoolBase(params, int(manifest.global_step), final=True), 0.0)
        self.client.acknowledge_finalization(manifest)
        self.finished = PoolFinished(manifest.global_step)

    def wait_base(self, *, newer_than: int | None, timeout_s: float) -> PoolBase | None:
        """The newest base with ``outer_version > newer_than`` (any base when
        None), or None after ``timeout_s``. The final cut always qualifies."""
        deadline = time.monotonic() + timeout_s
        while True:
            self._drain()
            base = self.latest_base
            if base is not None and (newer_than is None or base.outer_version > newer_than or base.final):
                return base
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.05)

    def known_final(self) -> bool:
        return self.finished is not None

    def raise_if_finished(self) -> None:
        self._drain()
        if self.finished is not None:
            raise self.finished

    # -- pushes --------------------------------------------------------------
    def elastic_init(self, params: torch.Tensor) -> None:
        """Seed the syncer's initial vector (the first one received wins)."""
        self.client.send_init(self.FRAGMENT, pack_tensor(params.reshape(-1), DTYPE_F32))

    def delta_tensor(self, *, base_version: int, c_tokens: int, c_steps: int, update: torch.Tensor,
                     timeout_s: float = 3600.0) -> int:
        """Answer the newest unanswered PULL with ``update`` (local - base at
        ``base_version``); returns the round's global step. Raises
        :class:`PoolFinished` when the run ends first."""
        deadline = time.monotonic() + timeout_s
        while True:
            self.raise_if_finished()
            if self._permits and not self.link_paused():
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"no PULL from the syncer within {timeout_s:.0f}s")
            time.sleep(0.05)
        step = max(self._permits)
        permit = self._permits.pop(step)
        self._permits.clear()  # older rounds are closed or will close without this island
        self._answered.add((step, permit.round_attempt))
        self.client.push_fragment(
            self.FRAGMENT, step, permit.round_attempt, int(base_version), int(self.inner_step),
            max(int(c_steps), 1), int(c_tokens), pack_tensor(update.reshape(-1), DTYPE_F32))
        return step

