"""In-process stand-ins for an island-pool syncer at the session level.

:class:`FakePoolSyncer` keeps one global LoRA vector and its outer version;
:meth:`FakePoolSyncer.client` returns a :class:`FakePoolClient` with the
interface :class:`yeto.rl.engine.bridges.PoolAvgSync` uses from
:class:`yeto.rl.pool_client.PoolIslandClient`. Each delta is merged at once
(base += update, version += 1). ``final`` ends the run at that version: the
next base is the final cut and a later push raises ``PoolFinished``.
"""

from __future__ import annotations

import threading
import time

import torch

from yeto.rl.pool_client import JoinAck, PoolBase, PoolFinished


class FakePoolSyncer:
    def __init__(self, base=None, *, version: int = 0, final: int | None = None,
                 catch_up: bool | None = None, reply_delay_s: float = 0.0,
                 base_delay_s: float = 0.0) -> None:
        self.base = None if base is None else torch.tensor(base, dtype=torch.float32)
        self.version = version
        self.final = final
        self.catch_up = catch_up
        self.reply_delay_s = reply_delay_s
        self.base_delay_s = base_delay_s  # a merged base becomes visible this long after the push
        self.deltas: list[dict] = []
        self.leaves: list[int] = []
        self.inits = 0
        self.lock = threading.Lock()

    def client(self, island_id: int = 0) -> "FakePoolClient":
        return FakePoolClient(self, island_id)


class FakePoolClient:
    def __init__(self, syncer: FakePoolSyncer, island_id: int) -> None:
        self.syncer = syncer
        self.island_id = island_id
        self.latest_base: PoolBase | None = None
        self.latest_base_timing: tuple[float, float] | None = None
        self.errors: list[str] = []
        self.inner_step = 0
        self.final_outer_version: int | None = None
        self.paused: list[float] = []
        self.closed = False
        self._on_event = lambda ev: None

    def _publish(self) -> None:
        s = self.syncer
        if s.base is not None:
            final = s.final is not None and s.version >= s.final
            self.latest_base = PoolBase(s.base.clone(), s.version, final=final)
            self.latest_base_timing = (time.time(), 0.0)

    def join(self) -> JoinAck:
        s = self.syncer
        catch_up = s.version > 0 if s.catch_up is None else s.catch_up
        return JoinAck(base_version=s.version, catch_up=catch_up)

    def wait_base(self, *, newer_than, timeout_s):
        if newer_than is not None and self.syncer.base_delay_s:
            time.sleep(self.syncer.base_delay_s)
        self._publish()
        base = self.latest_base
        if base is not None and (newer_than is None or base.outer_version > newer_than or base.final):
            return base
        return None

    def elastic_init(self, params) -> None:
        with self.syncer.lock:
            self.syncer.inits += 1
            if self.syncer.base is None:
                self.syncer.base = params.clone()

    def raise_if_finished(self) -> None:
        s = self.syncer
        if s.final is not None and s.version >= s.final:
            self._publish()
            raise PoolFinished(s.final)

    def delta_tensor(self, *, base_version, c_tokens, c_steps, update) -> int:
        s = self.syncer
        if s.reply_delay_s:
            time.sleep(s.reply_delay_s)
        with s.lock:
            s.deltas.append({"island": self.island_id, "base_version": base_version,
                             "c_tokens": c_tokens, "c_steps": c_steps, "update": update.clone()})
            s.base = s.base + update
            s.version = s.final if s.final is not None else s.version + 1
            return s.version

    def pause_link(self, seconds: float) -> None:
        self.paused.append(float(seconds))
        self._on_event({"event": "elastic_debug_pause", "pause_s": float(seconds)})

    def leave(self, reason: int = 0) -> None:
        self.syncer.leaves.append(int(reason))

    def close(self) -> None:
        self.closed = True
