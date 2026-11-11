"""The pool's data paths: how a scale-down or an eviction reaches the work contract
(``yeto.rl.engine.work_contract``) of the run.

An :class:`~.island_pool.IslandPool` never touches tasks itself; its ops call a
:class:`~.provider.Fencer`. The two implementations here follow the contract's
retire procedure:

* :class:`R2DataPath` (``data_path: r2``, async GRPO): drain = ``store_dispense.
  request_drain`` until ``drain_answered`` (the pool bounds it with
  ``elastic.scale_drain_timeout_s``); retire = fence the shard (handoff record +
  incarnation lease, ``yeto.launch.nc.elastic.fence_shard``) **and**
  ``store_dispense.retire`` (every open claim of the sampler released at once);
* :class:`SyncerDataPath` (``data_path: syncer``, decoupled DiLoCo): drain =
  ``task_dispense.retire_member(mode="drain")`` until its status reads
  ``drained``; retire = ``mode="retire"``; ``cancel_drain`` = ``mode="undrain"``.
  The member id of a sampler is ``task_dispense.sampler_holder_id(shard)``.

Every call is idempotent and may be retried.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

DATA_PATHS = ("r2", "syncer")


class R2DataPath:
    """``records``: the run's store as fence records (``elastic.s3_fence_store``:
    ``.objects``); ``dispense``: ``(objects, namespace)`` of the store dispenser, or
    None (no dispenser: fence only, drain unavailable). Both may be callables
    (built on first use: the store credentials are read late)."""

    kind = "r2"

    def __init__(self, records: Any, dispense: Any = None, *, clock: Callable[[], float] | None = None):
        self._records, self._dispense, self.clock = records, dispense, clock

    @property
    def records(self) -> Any:
        if callable(self._records):
            self._records = self._records()
        return self._records

    @property
    def dispense(self) -> tuple[Any, str] | None:
        if callable(self._dispense):
            self._dispense = self._dispense()
        return self._dispense

    def _kw(self) -> dict[str, Any]:
        return {} if self.clock is None else {"clock": self.clock}

    def fence_shard(self, shard: str, by: str) -> Any:
        from yeto.launch.nc import elastic  # noqa: PLC0415

        lease = elastic.fence_shard(self.records, shard, by=by, **self._kw())
        if self.dispense is not None:
            from yeto.rl.engine import store_dispense  # noqa: PLC0415

            objects, namespace = self.dispense
            store_dispense.retire(objects, namespace, shard, reason=f"pool fence by {by}")
        return lease

    def fence_incarnation(self, shard: str, by: str) -> Any:
        from yeto.launch.nc import elastic  # noqa: PLC0415

        return elastic.fence_incarnation(self.records, shard, by=by, **self._kw())

    def clear_stale_handoff(self, shard: str, keep: Any = ()) -> dict[str, Any] | None:
        """Set aside a stop request for ``shard`` whose successor is not in ``keep``
        (``sampler_resume.set_aside_handoff``: this run's store only); the record or None."""
        from yeto.rl.engine.sampler_resume import set_aside_handoff  # noqa: PLC0415

        return set_aside_handoff(self.records, shard, keep=keep, reason="new pool member on this shard name")

    def request_drain(self, shard: str, seq: str) -> str:
        from yeto.rl.engine import store_dispense  # noqa: PLC0415

        objects, namespace = self._need_dispense()
        return store_dispense.request_drain(objects, namespace, shard, seq=seq)

    def drain_answered(self, shard: str, seq: str) -> bool:
        from yeto.rl.engine import store_dispense  # noqa: PLC0415

        objects, namespace = self._need_dispense()
        return store_dispense.drain_answered(objects, namespace, shard, seq)

    def cancel_drain(self, shard: str) -> None:
        from yeto.rl.engine import store_dispense  # noqa: PLC0415

        objects, namespace = self._need_dispense()
        store_dispense.request_drain(objects, namespace, shard, on=False)

    def _need_dispense(self) -> tuple[Any, str]:
        if self.dispense is None:
            raise RuntimeError("no store dispenser in this run: nothing to drain")
        return self.dispense


class SyncerDataPath:
    """``address``: the syncer's ``(host, port)``; ``key``: the island HMAC key (None:
    the syncer accepts only its own host); ``retire_member``: the call (tests pass a
    fake; default ``task_dispense.retire_member``)."""

    kind = "syncer"

    def __init__(self, address: tuple[str, int], *, key: bytes | None = None, timeout_s: float = 10.0,
                 retire_member: Callable[..., dict[str, Any]] | None = None,
                 holder_id: Callable[[str], int] | None = None):
        self.address, self.key, self.timeout_s = tuple(address), key, float(timeout_s)
        self._retire, self._holder = retire_member, holder_id

    def _call(self, shard: str, mode: str, reason: str) -> dict[str, Any]:
        from yeto.rl.engine import task_dispense  # noqa: PLC0415

        retire = self._retire or task_dispense.retire_member
        holder = (self._holder or task_dispense.sampler_holder_id)(shard)
        return retire(self.address, holder, None, reason=reason, key=self.key, timeout_s=self.timeout_s, mode=mode)

    def fence_shard(self, shard: str, by: str) -> Any:
        return self._call(shard, "retire", f"pool fence by {by}")

    def fence_incarnation(self, shard: str, by: str) -> Any:
        # the syncer member table has no lease-only fence: retiring the current
        # incarnation frees its positions; the sampler restarted in place joins anew
        return self._call(shard, "retire", f"pool incarnation fence by {by}")

    def request_drain(self, shard: str, seq: str) -> str:
        self._call(shard, "drain", "scale_down")
        return seq

    def drain_answered(self, shard: str, seq: str) -> bool:  # noqa: ARG002 - the syncer keys on the member
        return str(self._call(shard, "drain", "scale_down").get("status")) == "drained"

    def cancel_drain(self, shard: str) -> None:
        self._call(shard, "undrain", "scale_up")

    def reassert_drain(self, shard: str) -> None:
        """Send the drain again (idempotent on the syncer). The syncer keeps its drain
        set in memory only: a restarted syncer forgets it, so the control plane re-sends
        every drain it still wants on each pool tick instead of the syncer persisting it."""
        self._call(shard, "drain", "scale_down")
