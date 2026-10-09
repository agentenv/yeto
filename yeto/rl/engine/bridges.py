"""Outer-sync sessions for :class:`yeto.rl.engine.driver.IslandDriver`.

The strict-avg and decoupled protocol state machines are reused unchanged:
``yeto.rl.bridge.StrictRlBridge`` and ``yeto.rl.decoupled.DecoupledRlBridge``.
What this module replaces is ``yeto.rl.adapters.miles.legacy.engine.MilesPolicySync`` /
``DecoupledMilesPolicySync``: every call into Miles internals (actor
``apply_trainable_state``/``export_trainable_state``, SGLang
``update_weight_version``) goes through the ``PolicyState`` port (via
``driver.apply_policy``) and the ``Publisher`` port (via ``driver.publish``).

Island progress checkpoints keep the legacy on-disk formats (strict schema 3,
decoupled schema 4) through the core progress module
(:mod:`yeto.rl.engine.progress`, shared with the legacy engine); they never
contain LoRA tensors or optimizer state.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from yeto.rl.bridge import BridgeConfig, StrictRlBridge
from yeto.rl.core import (
    CanonicalLoraState,
    LocalRoundStats,
    PolicySnapshot,
    canonical_state,
    policy_hash,
    policy_tensor_hash,
)

from . import progress as legacy_progress
from .pause_audit import PAUSABLE_PHASE
from .driver import IslandDriver, SyncBoundary, SyncStart
from .trainable_state import TrainableState


def _lora(state: TrainableState) -> CanonicalLoraState:
    return state.to_lora()


def _at_version(state: CanonicalLoraState, version: int) -> CanonicalLoraState:
    return canonical_state(
        version,
        state.tensors,
        base_model_revision=state.base_model_revision,
        lora_config_hash=state.lora_config_hash,
        layout_hash=state.layout_hash,
        expected_specs=state.specs,
    )


def _local_state(driver: Any, version: int) -> Any:
    """rl-publish-fastpath: with no outer sync nothing on the driver reads the tensors,
    so keep them in the trainer and bring back only the digests; drivers without the
    fast path get the full export as before."""

    resident = getattr(driver, "export_local_resident", None)
    state = resident(policy_version=version) if callable(resident) else None
    if state is not None:
        return state
    return TrainableState.from_lora(_at_version(_lora(driver.export_local()), version))


# ---------------------------------------------------------------------------
# No outer sync (single island smoke; task 4.1)
# ---------------------------------------------------------------------------
class LocalOnlySync:
    """Train-only island: the local result is the next policy."""

    def __init__(self, num_rollout: int) -> None:
        if num_rollout < 1:
            raise ValueError("num_rollout must be positive")
        self.num_rollout = num_rollout
        # rl-multinode-island M4: miles_adapter.round_cut.RoundCutCheckpoint, wired when
        # a checkpoint store is configured (resume from the newest round cut).
        self.round_cuts: Any = None
        # rl-resume-from-checkpoint: our own stop after this many rounds (exclusive
        # rollout id; None = run to num_rollout). A final cut is kept at the stop.
        self.stop_after: int | None = None

    def start(self, driver: IslandDriver) -> SyncStart:
        resumed = self.round_cuts.resume(driver) if self.round_cuts is not None else None
        rollout_id = 0 if resumed is None else int(resumed["next_rollout_id"])
        return SyncStart(_local_state(driver, rollout_id), rollout_id,
                         finished=rollout_id >= self._end())

    def at_safe_point(self, driver: IslandDriver, *, rollout_id: int, final: bool = False) -> None:
        """M4: keep a round cut at the safe point (every ``--rl-cut-every`` rounds, and
        always at the end). A failed cut is reported, not fatal: the previous pointer
        stays valid."""
        if self.round_cuts is None:
            return
        try:
            info = (self.round_cuts.save(driver, rollout_id=rollout_id, final=True) if final
                    else self.round_cuts.save(driver, rollout_id=rollout_id))
        except Exception as error:  # noqa: BLE001
            driver.emit("rl_round_cut", rollout_id=rollout_id, ok=False, error=repr(error)[:2000])
            return
        if info is not None:
            driver.emit("rl_round_cut", rollout_id=rollout_id, ok=True, **info)
            driver.emit("rl_cut_saved", rollout_id=rollout_id, **{
                k: info.get(k) for k in ("cut_id", "cut_bytes", "save_s", "store_copy_s",
                                         "store_commit_s", "total_s", "write_bytes_per_s",
                                         "policy_hash", "store_synced", "latest_seq", "pruned",
                                         "incarnation_index", "final", "trainer_onloaded_for_cut")})

    def _end(self) -> int:
        return self.num_rollout if self.stop_after is None else min(self.num_rollout, int(self.stop_after))

    def is_final_round(self, driver, *, rollout_id: int) -> bool:
        return rollout_id + 1 >= self.num_rollout

    def boundary(self, driver, *, rollout_id, stats) -> SyncBoundary:
        check = getattr(self.round_cuts, "check_first_round", None)
        if callable(check):  # resume: the first resumed round's lr continues (raises)
            check(driver, rollout_id=rollout_id, applied_lrs=getattr(stats, "applied_lrs", None))
        local = _local_state(driver, rollout_id + 1)
        driver.emit("rl_local_round", **asdict(stats))
        stop = rollout_id + 1 >= self._end()
        if stop and rollout_id + 1 < self.num_rollout:
            driver.emit("rl_stop_requested", rollout_id=rollout_id, stop_after=self.stop_after,
                        num_rollout=self.num_rollout, reason="--rl-stop-after-rounds")
        return SyncBoundary(local, stop=stop)

    def published(self, driver, *, rollout_id, policy_hash) -> None:
        pass

    def finish(self, driver) -> None:
        """rl-resume-from-checkpoint: the last round (or our own stop) always leaves a
        cut, so a later launch continues from exactly here."""
        version = getattr(driver, "published_version", None)
        if self.round_cuts is not None and version is not None:
            # after the last round's boundary + publish the island is at a round-boundary
            # safe point (nothing in flight); the cut requires it settled (G2 A: refused
            # "the outer commit of this cut is not settled" without this)
            driver.at_safe_point = True
            self.at_safe_point(driver, rollout_id=int(version), final=True)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Strict-avg
# ---------------------------------------------------------------------------
class StrictIslandProgress:
    """Strict island progress in the legacy schema-3 file (no LoRA/optimizer).

    The rollout process may already have written this round's record together
    with its completed-group queue (legacy ``_save_completed_groups``); that
    record is kept. Otherwise the driver writes this generation's metrics with
    an empty queue, replacing any record a killed learner left for the same
    round (as legacy's recovery rewrites it).
    """

    def __init__(self, args: Any) -> None:
        self.args = args
        self.path = Path(args.yeto_rl_completed_groups_path).expanduser()

    def _load(self) -> dict | None:
        if not self.path.is_file():
            return None
        try:
            payload = torch.load(self.path, map_location="cpu", weights_only=True)
        except Exception:  # the local queue is disposable, global state is not
            return None
        return payload if isinstance(payload, dict) else None

    def after_generate(self, *, rollout_id, policy_token, metrics) -> None:
        payload = self._load()
        if (
            payload is not None
            and payload.get("schema_version") == legacy_progress._ISLAND_CHECKPOINT_SCHEMA
            and payload.get("policy_version") == rollout_id
            and payload.get("config") == legacy_progress._island_checkpoint_config(self.args)
            and isinstance(payload.get("rollout_metrics"), Mapping)
            and payload["rollout_metrics"]
            and payload.get("completed_groups")
        ):
            return
        legacy_progress._atomic_save_island_checkpoint(
            self.path,
            {
                "schema_version": legacy_progress._ISLAND_CHECKPOINT_SCHEMA,
                "config": legacy_progress._island_checkpoint_config(self.args),
                "local_round_id": rollout_id + 1,
                "policy_version": rollout_id,
                "rollout_metrics": {k: float(v) for k, v in metrics.items()},
                "local_round_stats": None,
                "completed_groups": [],
            },
        )

    def commit_round(self, stats: LocalRoundStats) -> None:
        # Same contract as legacy ``_BridgeRuntime.record_local_round``.
        legacy_progress.record_strict_local_round(self, stats)


class _PortsStrictRuntime:
    """``IslandRuntime`` for ``StrictRlBridge``; applies happen via the driver."""

    def __init__(self, initial: CanonicalLoraState, progress: StrictIslandProgress | None):
        self.initial = initial
        self.progress = progress

    def initialize(self) -> CanonicalLoraState:
        if self.initial is None:
            raise RuntimeError("initial policy was already consumed")
        initial, self.initial = self.initial, None
        return initial

    def apply_global_policy(self, _state: CanonicalLoraState) -> None:
        pass

    def record_local_round(self, stats: LocalRoundStats) -> None:
        if self.progress is not None:
            self.progress.commit_round(stats)

    def shutdown(self) -> None:
        pass


class StrictAvgSync:
    """Strict-avg at the safe boundary: export+PUSH, wait v+1, reset-apply."""

    def __init__(
        self,
        config: BridgeConfig,
        *,
        progress: StrictIslandProgress | None = None,
        client_factory: Callable[[StrictRlBridge], Any] | None = None,
    ) -> None:
        self.config = config
        self.progress = progress
        self.client_factory = client_factory
        self.bridge: StrictRlBridge | None = None
        self.current: CanonicalLoraState | None = None
        self.permit = None

    def _apply(self, driver: IslandDriver, state: CanonicalLoraState) -> TrainableState:
        return driver.apply_policy(
            TrainableState.from_lora(state),
            optimizer="reset",
            # Scheduler progress is counted in optimizer steps, not rounds:
            # every strict round runs ``local_optimizer_steps`` steps.
            local_step=state.policy_version * self.config.local_optimizer_steps,
        )

    def start(self, driver: IslandDriver) -> SyncStart:
        initial = _lora(driver.export_local())
        self.bridge = StrictRlBridge(_PortsStrictRuntime(initial, self.progress), self.config)
        if self.client_factory is not None:
            self.bridge.client.close()
            self.bridge.client = self.client_factory(self.bridge)
        del initial
        self.bridge.start()
        self.current = self.bridge.wait_for_initial_policy()
        state = self._apply(driver, self.current)
        version = self.current.policy_version
        finished = version >= self.config.global_rounds
        if not finished:
            self.permit = self.bridge.wait_for_round()
        return SyncStart(state, version, finished)

    def is_final_round(self, driver, *, rollout_id: int) -> bool:
        # Strict: local round ``rollout_id + 1`` is the last iff it reaches
        # global_rounds (fix-decoupled-lr-schedule D4).
        return rollout_id + 1 >= self.config.global_rounds

    def boundary(self, driver, *, rollout_id, stats) -> SyncBoundary:
        self._submit(driver, rollout_id=rollout_id, stats=stats)
        return self._commit(driver, self._await(driver, rollout_id=rollout_id))

    # boundary = _submit -> _await -> _commit; DualStrictAvgSync (rl-algo-critic-family
    # 4.2.3) interleaves the actor and critic channels between the three steps.
    def _submit(self, driver, *, rollout_id, stats) -> None:
        if self.current is None or self.permit is None:
            raise RuntimeError("strict sync called outside an active round")
        if rollout_id != self.current.policy_version:
            raise RuntimeError("rollout ID differs from the global policy version")
        driver.phase("export_push", rollout_id=rollout_id)
        # The strict delta is taken against the exact base version.
        local = _at_version(_lora(driver.export_local()), rollout_id)
        self.bridge.submit_local_state(self.permit, self.current, local, stats)
        self.bridge.release_current(rollout_id)
        self.current = None

    def _await(self, driver, *, rollout_id) -> CanonicalLoraState:
        driver.phase("wait_global", policy_version=rollout_id + 1)
        return self.bridge.wait_for_global_policy(rollout_id + 1)

    def _commit(self, driver, current: CanonicalLoraState) -> SyncBoundary:
        state = self._apply(driver, current)
        self.current = current
        stop = current.policy_version >= self.config.global_rounds
        self.permit = None if stop else self.bridge.wait_for_round()
        return SyncBoundary(state, stop)

    def published(self, driver, *, rollout_id, policy_hash) -> None:
        pass

    def outer_phase(self, driver, *, rollout_id: int) -> str:
        """3.8: pausable only while the next PULL permit is held and the syncer
        has not started finalizing (pause_audit.md: strict row)."""
        client = getattr(self.bridge, "client", None)
        finalizing = getattr(client, "finalizing", None)
        if finalizing is not None and finalizing.is_set():
            return "finalizing"
        if self.current is None:
            return "in-boundary"
        if self.permit is None:
            return "stop-round"
        return PAUSABLE_PHASE

    def finish(self, driver) -> None:
        if self.bridge is None or self.current is None:
            raise RuntimeError("finalized an uninitialized strict sync")
        final = self.bridge.finalize()
        if policy_hash(final) != policy_hash(self.current):
            raise RuntimeError("final policy differs from the committed global policy")

    def close(self) -> None:
        if self.bridge is not None:
            self.bridge.client.close()


def _elastic_local_round_fields(stats: LocalRoundStats, base_version: int, payload_bytes: int) -> dict:
    """0.24: the strict bridge's ``rl_local_round`` fields (yeto/rl/bridge.py
    ``_record_submission``) for an elastic round; base = the syncer's outer version."""
    return {
        **asdict(stats),
        "rl/active_groups": stats.active_groups,
        "rl/completed_groups": stats.completed_groups,
        "rl/cancelled_groups": stats.cancelled_groups,
        "rl/completed_trajectories": stats.completed_trajectories,
        "rl/action_tokens": stats.action_tokens,
        "rl/tool_wait_seconds": stats.tool_wait_seconds,
        "rl/reward_mean": stats.reward_mean,
        "rl/reward_std": stats.reward_std,
        "rl/rollout_seconds": stats.rollout_seconds,
        "rl/group_p50_seconds": stats.group_p50_seconds,
        "rl/group_p95_seconds": stats.group_p95_seconds,
        "rl/group_p99_seconds": stats.group_p99_seconds,
        "rl/zero_variance_group_ratio": stats.zero_variance_group_ratio,
        "rl/global_policy_version": base_version,
        "rl/rollout_policy_version": base_version,
        "rl/mixed_version_group_count": 0,
        "rl/local_delta_norm": stats.delta_l2_norm,
        "rl/current_vs_rollout_kl": stats.mean_kl,
        "rl/ess_ratio": stats.ess_ratio,
        "rl/clip_fraction": stats.clip_fraction,
        "sync/bytes_sent": payload_bytes,
    }


class ElasticAvgSync:
    """``--rl-island-scheduling elastic`` sync session (rl-inter-island-scheduling 0.15).

    Same session interface as :class:`StrictAvgSync`, backed by the elastic
    syncer: JOIN at start (ELASTIC_INIT when no base exists yet), one
    DELTA_TENSOR per boundary (theta - base as a flat f32 vector; c_tokens =
    ``stats.action_tokens``, c_steps = ``local_optimizer_steps``), then the next
    ELASTIC_BASE is reset-applied. The driver's rollout ids stay local round
    ids; the syncer's outer version is tracked separately (``base_version``)
    because a late island may see it jump. LEAVE at finish/close.
    """

    OUTER_SYNC_KIND = "elastic"

    def __init__(self, config: BridgeConfig, *, progress: StrictIslandProgress | None = None,
                 client: Any = None, syncer_epoch: int = 0, base_wait_s: float = 3600.0,
                 groups_per_round: int | None = None) -> None:
        self.config = config
        # 0.21: groups one round draws from the data source; used to advance the
        # cursor on a restart whose ledger holds no cursor (fresh machine).
        self._sleep = __import__("time").sleep
        self.groups_per_round = groups_per_round if groups_per_round is not None else getattr(
            config, "groups_per_round", None)
        self.progress = progress
        self.syncer_epoch = int(syncer_epoch)
        self.base_wait_s = base_wait_s
        self.client = client
        self.template: CanonicalLoraState | None = None
        self.specs = None
        self.base: list[float] | None = None
        self.base_version: int | None = None
        self.in_boundary = False
        self.left = False

    def _client(self):
        if self.client is None:
            from yeto.rl.elastic_client import (ElasticClientConfig, ElasticIslandClient,
                                                hmac_key_from_env)

            self.client = ElasticIslandClient(
                ElasticClientConfig(self.config.syncer_addr, self.config.learner_id,
                                    syncer_epoch=self.syncer_epoch,
                                    backend_identity_sha256=getattr(
                                        self.config, "backend_identity_sha256", None)),
                hmac_key_from_env(), on_event=self._client_event)
        return self.client

    _TAPE_CLIENT_EVENTS = ("elastic_rejoin", "elastic_rejoin_failed", "elastic_debug_pause",
                           "elastic_debug_pause_end")

    def _client_event(self, ev: dict) -> None:
        """0.22: re-JOIN outcomes go onto the island tape (sent/received frames do not)."""
        driver = getattr(self, "_driver", None)
        if driver is not None and ev.get("event") in self._TAPE_CLIENT_EVENTS:
            fields = {k: v for k, v in ev.items() if k != "event"}
            driver.emit(ev["event"], **fields)
            limit = getattr(getattr(self.client, "config", None), "max_rejoin_failures", None)
            if ev["event"] == "elastic_rejoin_failed" and limit and ev.get("failures", 0) >= limit:
                # 0.27: the launcher reads this as "left the elastic pool" (exit 7)
                driver.emit("elastic_left_pool", reason="rejoin_exhausted", failures=ev["failures"])

    def _wait(self, newer_than):
        base = self.client.wait_base(newer_than=newer_than, timeout_s=self.base_wait_s)
        if base is None:
            raise RuntimeError(f"no ELASTIC_BASE newer than {newer_than} within "
                               f"{self.base_wait_s}s; errors: {self.client.errors}")
        return base

    def _apply(self, driver: IslandDriver, base, rollout_id: int) -> TrainableState:
        from yeto.rl.bridge import unflatten_state

        self.base, self.base_version = list(base.params), base.outer_version
        state = unflatten_state(self.base, rollout_id, self.template, self.specs)
        return driver.apply_policy(TrainableState.from_lora(state), optimizer="reset",
                                   local_step=rollout_id * self.config.local_optimizer_steps)

    def start(self, driver: IslandDriver) -> SyncStart:
        from yeto.rl.bridge import flatten_state

        self.template = _lora(driver.export_local())
        self.specs = tuple(self.config.expected_specs or ()) or tuple(self.template.specs)
        self._driver = driver
        injected = self.client is not None  # a client built by _client() already reports to the tape
        client = self._client()
        if injected and getattr(client, "_on_event", None) is not None \
                and not getattr(client, "_tape_hooked", False):
            inner = client._on_event  # an injected client: chain the tape hook

            def chained(ev, _inner=inner):
                _inner(ev)
                self._client_event(ev)
            client._on_event, client._tape_hooked = chained, True
        client.final_outer_version = int(self.config.global_rounds)  # syncer total_steps (0.26)
        ack = client.join()
        driver.phase("elastic_join", catch_up=ack.catch_up, base_version=ack.base_version)
        base = client.wait_base(newer_than=None, timeout_s=1.0)
        if base is None:
            client.elastic_init(flatten_state(self.template, self.specs))
            base = self._wait(None)
        # 0.21: a restarted island resumes after its own trained rounds and never
        # behind the syncer's version (the driver then seeks the data cursor).
        ledger = getattr(driver, "ledger", None)
        ledger_next = ledger.next_rollout_id() if ledger is not None else 0
        start = max(ledger_next, int(ack.base_version), 0)
        if start > 0:
            driver.phase("elastic_resume", rollout_id=start, ledger_next=ledger_next,
                         base_version=ack.base_version, outer_version=base.outer_version)
        self.client.inner_step = start * self.config.local_optimizer_steps
        state = self._apply(driver, base, start)
        return SyncStart(state, start, start >= self.config.global_rounds)

    def debug_delay_s(self) -> float:
        """0.25: ``YETO_RL_ELASTIC_DEBUG_DELAY`` = "ISLAND:S[,...]" (bare S = all)."""
        import os

        from yeto.rl.elastic_client import DEBUG_DELAY_ENV, parse_elastic_debug_delay

        spec = os.environ.get(DEBUG_DELAY_ENV, "")
        if not spec:
            return 0.0

        table = parse_elastic_debug_delay(spec)
        return float(table.get(int(self.config.learner_id), table.get(-1, 0.0)))

    def debug_pause(self) -> tuple[int, float] | None:
        """``YETO_RL_ELASTIC_DEBUG_PAUSE`` = "ISLAND:AFTER_V:S[,...]" -> (after_v, s) for this island."""
        import os

        from yeto.rl.elastic_client import DEBUG_PAUSE_ENV, parse_elastic_debug_pause

        spec = os.environ.get(DEBUG_PAUSE_ENV, "")
        if not spec:
            return None
        return parse_elastic_debug_pause(spec).get(int(self.config.learner_id))

    def _maybe_pause_link(self, driver) -> None:
        """Test switch: once per process, after a base >= AFTER_V was applied."""
        if getattr(self, "_pause_done", False):
            return
        pause = self.debug_pause()
        if pause is None or self.base_version is None or self.base_version < pause[0]:
            return
        self._pause_done = True
        driver.phase("elastic_debug_pause_armed", base_version=self.base_version, pause_s=pause[1])
        self.client.pause_link(pause[1])

    def restart_cursor_fallback(self, current: Mapping[str, int] | None,
                                start_rollout_id: int) -> dict[str, int] | None:
        """0.21: the ledger has no cursor for ``start_rollout_id`` (e.g. a fresh
        machine joining at base_version > 0): skip ``start * groups_per_round``
        groups from the fresh data source position (same shift as
        cut_injection.write_shifted_dataset_state). None when unknown."""
        if not self.groups_per_round or start_rollout_id <= 0:
            return None
        cur = {"sample_offset": 0, "epoch_id": 0, "sample_group_index": 0, "sample_index": 0}
        cur.update({k: int(v) for k, v in dict(current or {}).items() if k in cur})
        groups = int(start_rollout_id) * int(self.groups_per_round)
        cur["sample_offset"] += groups
        cur["sample_group_index"] += groups
        return cur

    def is_final_round(self, driver, *, rollout_id: int) -> bool:
        return rollout_id + 1 >= self.config.global_rounds

    def boundary(self, driver, *, rollout_id, stats) -> SyncBoundary:
        from yeto.rl.bridge import flatten_state

        if self.base is None:
            raise RuntimeError("elastic sync called outside an active round")
        self.in_boundary = True
        driver.phase("export_push", rollout_id=rollout_id, base_version=self.base_version)
        local = flatten_state(_lora(driver.export_local()), self.specs)
        update = [a - b for a, b in zip(local, self.base)]
        sent_on = self.base_version
        stats = replace(stats, delta_l2_norm=math.sqrt(sum(u * u for u in update)))
        delay = self.debug_delay_s()
        if delay > 0:  # 0.25 test switch (heartbeats continue on their own thread)
            driver.phase("elastic_debug_delay", rollout_id=rollout_id, delay_s=delay)
            self._sleep(delay)
        driver.emit("rl_local_round", **_elastic_local_round_fields(stats, sent_on, 4 * len(update)))
        from yeto.rl.elastic_client import ElasticFinished

        self.client.inner_step += self.config.local_optimizer_steps
        try:
            self.client.raise_if_finished()
            self.client.delta_tensor(base_version=sent_on, c_tokens=int(stats.action_tokens),
                                     c_steps=self.config.local_optimizer_steps, update=update)
            if self.progress is not None:
                self.progress.commit_round(stats)
            driver.phase("wait_global", base_version=sent_on)
            state = self._apply(driver, self._wait(sent_on), rollout_id + 1)
            self._maybe_pause_link(driver)
        except ElasticFinished as done:
            return self._finished(driver, done, rollout_id=rollout_id, sent_on=sent_on)
        self.in_boundary = False
        return SyncBoundary(state, rollout_id + 1 >= self.config.global_rounds)

    def _finished(self, driver, done, *, rollout_id: int, sent_on: int) -> SyncBoundary:
        """0.26: the syncer ended the run while this (late) island was still
        pushing: end normally on the newest base (exit 0), not BrokenPipe/exit 1."""
        base = self.client.latest_base
        driver.emit("elastic_finished", rollout_id=rollout_id, reason=done.reason,
                    final_outer_version=done.final_outer_version, sent_on=sent_on,
                    base_version=None if base is None else base.outer_version)
        # FINISHED: LEAVE first so the syncer exits as soon as every island left
        # instead of waiting out its final grace window; best-effort otherwise.
        if not self.left:
            try:
                self.client.leave()
            except Exception:  # noqa: BLE001 - the syncer may already be gone
                pass
        self.left = True
        self.in_boundary = False
        if base is not None and base.outer_version > sent_on:
            state = self._apply(driver, base, rollout_id + 1)
        else:
            state = TrainableState.from_lora(_at_version(_lora(driver.export_local()), rollout_id + 1))
        return SyncBoundary(state, True)

    def published(self, driver, *, rollout_id, policy_hash) -> None:
        pass

    def outer_phase(self, driver, *, rollout_id: int) -> str:
        return "in-boundary" if self.in_boundary else PAUSABLE_PHASE

    def finish(self, driver) -> None:
        if self.client is not None and not self.left:
            from yeto.rl.elastic_client import ElasticFinished

            try:
                self.client.leave()
            except (ElasticFinished, BrokenPipeError, ConnectionResetError):
                pass  # 0.26: the syncer already finished
            self.left = True

    def close(self) -> None:
        if self.client is not None:
            if not self.left:
                try:
                    self.client.leave()
                except Exception:
                    pass
                self.left = True
            self.client.close()


# ---------------------------------------------------------------------------
# Strict-avg with a critic: two syncer channels, one atomic commit
# (rl-algo-critic-family 4.2.3, design D4 plan a)
# ---------------------------------------------------------------------------
class CrossChannelCommitError(RuntimeError):
    """The actor or critic channel did not reach v+1: neither role was applied."""


class _CriticDriverView:
    """What StrictAvgSync needs from a driver, for the critic role: export and
    apply go to the trainer's critic (full-parameter tensors, critic layout)."""

    def __init__(self, driver: IslandDriver, identity: tuple[str, str]) -> None:
        # identity: (base model revision, critic_layout_hash). The critic layout
        # hash (value head, param_mode, real dtypes) rides as the channel's config
        # hash; the channel's tensor layout hash is the canonical fp32 one.
        self.driver = driver
        self.base_model_revision, self.critic_layout_hash = identity

    PREFIX = "critic."

    def critic_state(self, version: int, tensors=None) -> CanonicalLoraState:
        tensors = self.driver.trainer.export_critic_state() if tensors is None else tensors
        return canonical_state(
            version,
            {self.PREFIX + k: v for k, v in tensors.items()},
            base_model_revision=self.base_model_revision,
            lora_config_hash=self.critic_layout_hash,
        )

    def export_local(self) -> TrainableState:
        return TrainableState.from_lora(self.critic_state(0))

    def apply_policy(self, state: TrainableState, *, optimizer: str, local_step: int) -> TrainableState:
        # The critic keeps its own optimizer state across rounds (saved by the round
        # cut, 4.3); only the weights are replaced by the committed average.
        del optimizer, local_step
        written = self.driver.trainer.import_critic_state(
            {k[len(self.PREFIX):]: v for k, v in state.to_lora().tensors.items()})
        # Evidence of the applied average (both hashes over FP32 values): the
        # channel's canonical critic state, and the critic masters as written
        # back (import_critic_state re-hashes them; bf16 params are checked
        # against their master casts there).
        self.driver.emit(
            "rl_critic_apply",
            policy_version=state.policy_version,
            **{"sync/global_critic_hash": state.policy_tensor_hash(),
               "rl/critic/applied_weights_sha256": written},
        )
        return state

    def phase(self, name: str, **fields: Any) -> None:
        self.driver.phase(f"critic_{name}", **fields)


class DualStrictAvgSync:
    """Strict-avg of actor (LoRA, actor syncer) + critic (full parameters, critic
    syncer). Round v -> v+1 is applied to the trainer only when BOTH channels
    returned v+1; otherwise neither is applied, ``CrossChannelCommitError`` is
    raised and the island stays at the last committed round v (in memory when
    ``keep_committed``; durably through the round cut, whose critic round must
    equal the actor round, 4.3)."""

    OUTER_SYNC_KIND = "strict"

    def __init__(self, config: BridgeConfig, *, critic_syncer_addr: tuple[str, int],
                 progress: StrictIslandProgress | None = None,
                 client_factory: Callable[[StrictRlBridge], Any] | None = None,
                 critic_client_factory: Callable[[StrictRlBridge], Any] | None = None,
                 keep_committed: bool = False) -> None:
        self.config = config
        self.critic_syncer_addr = critic_syncer_addr
        self.actor = StrictAvgSync(config, progress=progress, client_factory=client_factory)
        self.critic: StrictAvgSync | None = None
        self.critic_client_factory = critic_client_factory
        self.view: _CriticDriverView | None = None
        self.keep_committed = keep_committed
        self.committed: tuple[int, TrainableState, dict] | None = None
        self.committed_version: int | None = None

    def _critic_config(self, driver: IslandDriver) -> BridgeConfig:
        layout = driver.trainer.critic_layout()
        self.view = _CriticDriverView(driver, (self.config.base_model_revision, layout))
        initial = self.view.critic_state(0)
        tape = Path(self.config.event_tape)
        return replace(
            self.config,
            syncer_addr=self.critic_syncer_addr,
            expected_specs=initial.specs,
            lora_config_hash=layout,
            layout_hash=initial.layout_hash,
            event_tape=str(tape.with_name(tape.stem + "-critic" + tape.suffix)),
            audit_dir=None,
        )

    def _remember(self, driver: IslandDriver, version: int) -> None:
        self.committed_version = version
        if self.keep_committed:
            self.committed = (version, driver.export_local(), driver.trainer.export_critic_state())

    def start(self, driver: IslandDriver) -> SyncStart:
        self.critic = StrictAvgSync(self._critic_config(driver), client_factory=self.critic_client_factory)
        critic = self.critic.start(self.view)
        start = self.actor.start(driver)
        if critic.rollout_id != start.rollout_id or critic.finished != start.finished:
            raise CrossChannelCommitError(
                f"actor channel starts at round {start.rollout_id}, critic channel at {critic.rollout_id}")
        self._remember(driver, start.rollout_id)
        return start

    def is_final_round(self, driver, *, rollout_id: int) -> bool:
        return self.actor.is_final_round(driver, rollout_id=rollout_id)

    def boundary(self, driver, *, rollout_id, stats) -> SyncBoundary:
        target = rollout_id + 1
        try:
            # push both channels before waiting on either (no cross-channel wait cycle)
            self.actor._submit(driver, rollout_id=rollout_id, stats=stats)
            self.critic._submit(self.view, rollout_id=rollout_id, stats=stats)
            actor = self.actor._await(driver, rollout_id=rollout_id)
            critic = self.critic._await(self.view, rollout_id=rollout_id)
            if actor.policy_version != target or critic.policy_version != target:
                raise CrossChannelCommitError(
                    f"channels returned actor v{actor.policy_version} / critic v{critic.policy_version}, "
                    f"expected v{target}")
        except Exception as error:
            self._rollback(driver)
            raise CrossChannelCommitError(
                f"round {target} not committed: {error}; neither actor nor critic applied, "
                f"the island stays at committed round {self.committed_version}") from error
        self.critic._commit(self.view, critic)
        boundary = self.actor._commit(driver, actor)
        self._remember(driver, target)
        driver.phase("dual_commit", policy_version=target)
        return boundary

    def _rollback(self, driver: IslandDriver) -> None:
        if self.committed is None:
            return  # durable rollback: restore the last round cut (critic round == actor round)
        version, actor, critic = self.committed
        driver.apply_policy(actor, optimizer="reset", local_step=version * self.config.local_optimizer_steps)
        driver.trainer.import_critic_state(critic)

    def published(self, driver, *, rollout_id, policy_hash) -> None:
        pass

    def outer_phase(self, driver, *, rollout_id: int) -> str:
        return self.actor.outer_phase(driver, rollout_id=rollout_id)

    def finish(self, driver) -> None:
        self.actor.finish(driver)
        self.critic.finish(self.view)

    def close(self) -> None:
        self.actor.close()
        if self.critic is not None:
            self.critic.close()


# ---------------------------------------------------------------------------
# Decoupled
# ---------------------------------------------------------------------------
class DecoupledIslandProgress:
    """Decoupled island progress in the legacy schema-4 file."""

    def __init__(self, args: Any) -> None:
        self.args = args

    def after_generate(self, *, rollout_id, policy_token, metrics) -> None:
        payload = legacy_progress._load_decoupled_checkpoint(self.args)
        if (
            payload is None
            or payload.get("next_rollout_id") != rollout_id
            or payload.get("policy_token") != policy_token
        ):
            raise RuntimeError("decoupled RL progress changed during rollout")
        if payload.get("rollout_metrics"):
            return  # written by the rollout process with its group queue
        legacy_progress._save_decoupled_checkpoint(
            self.args,
            snapshot=PolicySnapshot(
                rollout_id, tuple(payload["fragment_versions"]), payload["policy_hash"]
            ),
            optimizer_steps=payload["optimizer_steps"],
            action_tokens=payload["action_tokens"],
            rollout_metrics=metrics,
            local_round_stats=payload.get("local_round_stats"),
            completed_groups=payload["completed_groups"],
        )


class DecoupledSync:
    """Run-until-stop decoupled fragments through PolicyState/Publisher.

    Mirrors ``DecoupledMilesPolicySync`` step for step; ``_record_local_round``
    is the legacy implementation (called unbound, it only needs
    ``_append_event`` and ``snapshot``).
    """

    # read by the driver handshake (rl-algo-critic-family 2.3: no critic here)
    OUTER_SYNC_KIND = "decoupled"

    def __init__(
        self,
        args: Any,
        *,
        bridge_factory: Callable[..., Any] | None = None,
        client_factory: Callable[[Any], Any] | None = None,
    ) -> None:
        self.args = args
        self.bridge_factory = bridge_factory
        self.client_factory = client_factory
        self.bridge = None
        self.current: CanonicalLoraState | None = None
        self.snapshot: PolicySnapshot | None = None
        self.optimizer_steps = 0
        self.action_tokens = 0
        self.finished = False
        self._driver: IslandDriver | None = None

    # legacy-compatible helpers ------------------------------------------
    def _append_event(self, event: dict[str, Any]) -> None:
        self._driver.events.append(event)

    _record_local_round = legacy_progress.DecoupledProgress._record_local_round
    _record_final_payload = legacy_progress.DecoupledProgress._record_final_payload
    _save_progress = legacy_progress.DecoupledProgress._save_progress

    def _apply(self, state: CanonicalLoraState, *, reset: bool) -> CanonicalLoraState:
        self._driver.apply_policy(
            TrainableState.from_lora(state),
            optimizer="reset" if reset else "preserve",
            local_step=self.optimizer_steps,
        )
        return state

    def _snapshot_state(self, snapshot: PolicySnapshot) -> TrainableState:
        self._driver.emit(
            "rl_policy_snapshot",
            **{
                "rl/rollout_id": snapshot.rollout_id,
                "rl/policy_token": snapshot.token,
                "rl/policy_hash": snapshot.policy_hash,
                "rl/fragment_versions": list(snapshot.fragment_versions),
                "rl/canonical_layout_hash": self.current.layout_hash,
                "rl/sync_layout_fingerprint": self.args.yeto_rl_sync_layout_fingerprint,
                "rl/mixed_version_group_count": 0,
            },
        )
        return TrainableState.from_lora(_at_version(self.current, snapshot.rollout_id))

    # session -------------------------------------------------------------
    def start(self, driver: IslandDriver) -> SyncStart:
        from yeto.rl import decoupled

        self._driver = driver
        args = self.args
        initial = _lora(driver.export_local())
        adapter = getattr(args, "yeto_rl_initial_adapter", None)
        adapter_sha = getattr(args, "yeto_rl_initial_adapter_sha256", None)
        if (adapter is None) != (adapter_sha is None):
            raise ValueError(
                "decoupled RL initial adapter path and SHA256 must be set together"
            )
        parent_hash = None
        if adapter is not None:
            from yeto.rl.initial_adapter import load_initial_adapter

            initial = load_initial_adapter(
                adapter, adapter_sha, model=args.yeto_rl_model, expected=initial
            )
            parent_hash = policy_tensor_hash(initial)
        checkpoint_error = None
        try:
            checkpoint = legacy_progress._load_decoupled_checkpoint(args)
        except RuntimeError as error:
            checkpoint, checkpoint_error = None, error
        if checkpoint is not None:
            self.optimizer_steps = checkpoint["optimizer_steps"]
            self.action_tokens = checkpoint["action_tokens"]

        factory = self.bridge_factory or decoupled.DecoupledRlBridge
        self.bridge = factory(initial, args.yeto_rl_bridge_config)
        if self.client_factory is not None:
            self.bridge.client.close()
            self.bridge.client = self.client_factory(self.bridge)
        self.bridge.start()
        cut = self.bridge.wait_for_initial_cut(
            optimizer_steps=self.optimizer_steps, action_tokens=self.action_tokens
        )
        versions = cut.fragment_versions
        if any(versions) and checkpoint is None:
            detail = "missing" if checkpoint_error is None else str(checkpoint_error)
            raise RuntimeError(
                f"nonzero decoupled RL cut has no valid island checkpoint: {detail}"
            )
        if (
            parent_hash is not None
            and not any(versions)
            and policy_tensor_hash(cut.state) != parent_hash
        ):
            raise RuntimeError(
                "decoupled RL initial adapter policy differs from version-zero cut"
            )
        if parent_hash is not None and checkpoint is None:
            self._append_event(
                {
                    "event": "rl_initial_adapter",
                    "parent_adapter_sha256": adapter_sha,
                    "parent_policy_hash": parent_hash,
                }
            )
        if checkpoint is not None and any(
            current < saved for current, saved in zip(versions, checkpoint["fragment_versions"])
        ):
            raise RuntimeError("decoupled RL syncer cut predates island checkpoint")
        # Restart: the authoritative cut is applied with an optimizer reset.
        self.current = self._apply(_at_version(cut.state, self.optimizer_steps), reset=True)
        self.bridge.commit_initial_cut(
            cut, optimizer_steps=self.optimizer_steps, action_tokens=self.action_tokens
        )
        self.snapshot = PolicySnapshot.create(self.optimizer_steps, self.current, versions)
        self._save_progress(self.snapshot, stats=None)
        startup_manifest = getattr(self.bridge, "startup_final_manifest", None)
        if startup_manifest is not None:
            self.bridge.acknowledge_finalization(startup_manifest)
            self._record_final_payload(self.bridge.final_payload_bytes_received)
            self.finished = True
        return SyncStart(
            self._snapshot_state(self.snapshot), self.snapshot.rollout_id, self.finished
        )

    def published(self, driver, *, rollout_id, policy_hash) -> None:
        if (
            self.snapshot is None
            or rollout_id != self.snapshot.rollout_id
            or policy_hash != self.snapshot.policy_hash
        ):
            raise RuntimeError("published policy differs from the decoupled snapshot")
        self.args.yeto_rl_policy_token = self.snapshot.token

    def _commit_final_cut(self, manifest, final, *, stats, final_payload_bytes_received=0):
        self.current = self._apply(final, reset=False)
        self.snapshot = PolicySnapshot.create(
            final.policy_version, self.current, manifest.versions
        )
        self._save_progress(self.snapshot, stats=stats)
        self.bridge.acknowledge_finalization(manifest)
        if final_payload_bytes_received:
            self._record_final_payload(final_payload_bytes_received)
        self.finished = True
        return SyncBoundary(self._snapshot_state(self.snapshot), stop=True)

    def _finish(self, *, policy_version, stats) -> SyncBoundary:
        manifest, final = self.bridge.wait_for_final_cut(policy_version=policy_version)
        return self._commit_final_cut(
            manifest,
            final,
            stats=stats,
            final_payload_bytes_received=self.bridge.final_payload_bytes_received,
        )

    def outer_phase(self, driver, *, rollout_id: int) -> str:
        # Decoupled is not pause-certified (pause_audit); report finalization
        # anyway so the veto reason is the precise one.
        if self.bridge is not None and self.bridge.finalizing:
            return "finalizing"
        return PAUSABLE_PHASE

    def is_final_round(self, driver, *, rollout_id: int) -> bool:
        # Decoupled: final once the syncer's final cut is known (finalizing),
        # or this round exhausts the island's optional step budget (D4).
        if self.bridge is not None and self.bridge.finalizing:
            return True
        budget = getattr(self.args, "yeto_rl_learner_budget_steps", None)
        return budget is not None and budget == self.optimizer_steps + 1

    def boundary(self, driver, *, rollout_id, stats) -> SyncBoundary:
        import time

        started = time.monotonic()
        result = self._boundary(driver, rollout_id=rollout_id, stats=stats)
        self._append_event(
            {
                "event": "rl_sync_hook",
                "rl/rollout_id": rollout_id,
                "sync/hook_seconds": time.monotonic() - started,
                "sync/remote_quorum_wait_seconds": 0.0,
                "sync/finalization": result.stop,
            }
        )
        return result

    def _boundary(self, driver, *, rollout_id, stats) -> SyncBoundary:
        if (
            self.current is None
            or self.snapshot is None
            or rollout_id != self.snapshot.rollout_id
            or self.args.yeto_rl_policy_token != self.snapshot.token
        ):
            raise RuntimeError("decoupled synchronization outside its policy snapshot")
        next_rollout_id = rollout_id + 1
        local = _at_version(_lora(driver.export_local()), next_rollout_id)
        self.optimizer_steps += 1
        self.action_tokens += stats.action_tokens
        if self.optimizer_steps != next_rollout_id:
            raise RuntimeError("decoupled RL optimizer progress diverged from rollout ID")
        if self.bridge.finalizing:
            self._record_local_round(stats, rollout_id=rollout_id)
            return self._finish(policy_version=next_rollout_id, stats=stats)
        if getattr(self.args, "yeto_rl_learner_budget_steps", None) == self.optimizer_steps:
            consolidation = self.bridge.consolidate_budget(
                local, optimizer_steps=self.optimizer_steps, action_tokens=self.action_tokens
            )
            stats = replace(
                stats,
                delta_l2_norm=math.sqrt(
                    sum(s.delta_l2_norm**2 for s in consolidation.submissions)
                ),
            )
            self._record_local_round(
                stats,
                rollout_id=rollout_id,
                submissions=consolidation.submissions,
                additional_payload_bytes_received=consolidation.bytes_received,
            )
            return self._commit_final_cut(consolidation.manifest, consolidation.state, stats=stats)

        driver.phase("drain_bcast", rollout_id=rollout_id)
        batch = self.bridge.drain_broadcasts(
            local, optimizer_steps=self.optimizer_steps, action_tokens=self.action_tokens
        )
        current = batch.state
        if batch.fragment_ids:
            current = self._apply(current, reset=False)
            self.bridge.commit_broadcasts(
                batch, optimizer_steps=self.optimizer_steps, action_tokens=self.action_tokens
            )
        driver.phase("drain_pull", rollout_id=rollout_id)
        submissions = self.bridge.submit_ready(
            current, optimizer_steps=self.optimizer_steps, action_tokens=self.action_tokens
        )
        stats = replace(
            stats,
            delta_l2_norm=math.sqrt(sum(s.delta_l2_norm**2 for s in submissions)),
        )
        self._record_local_round(
            stats, rollout_id=rollout_id, batch=batch, submissions=submissions
        )
        if self.bridge.finalizing or any(
            s.global_step == self.args.yeto_rl_total_fragment_steps for s in submissions
        ):
            return self._finish(policy_version=next_rollout_id, stats=stats)
        self.current = current
        self.snapshot = PolicySnapshot.create(
            next_rollout_id, current, self.bridge.fragment_versions
        )
        self._save_progress(self.snapshot, stats=stats)
        return SyncBoundary(self._snapshot_state(self.snapshot), stop=False)

    def finish(self, driver) -> None:
        if not self.finished:
            raise RuntimeError("stopped before decoupled RL finalization")

    def close(self) -> None:
        if self.bridge is not None:
            self.bridge.close()
