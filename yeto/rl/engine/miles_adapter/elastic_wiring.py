"""E1 wiring for ``compose_island`` (rl-infra-spec 3.2-3.7).

:func:`build_elastic` turns a resources manifest (1.6 / PR #66 ``resources``
block: ``configs`` and ``edges``) plus a capability attestation into the
objects ``compose_island(elastic=...)`` passes to the driver: the
:class:`~..controller.IslandController`, the :class:`~..ledger.BatchLedger`,
the startup-declared rollout cells and the pool GPUs. Nothing here starts,
stops or leases anything; without ``elastic`` the island is unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ElasticWiring:
    controller: Any  # IslandController
    ledger: Any  # BatchLedger
    declared_cells: tuple[str, ...]
    pool_gpus: tuple[str, ...] | None = None
    track_timeout_s: float = 600.0
    tool_wait_board: Any = None


def build_elastic(
    *,
    state_dir: str | Path,
    resources: dict[str, Any] | str | Path,
    attestation: str | Path | None,
    profile: Any,
    initial_config: str,
    runtime_fingerprint: str,
    declared_cells: Sequence[str],
    pool_gpus: Sequence[str] | None = None,
    timeouts: Any = None,
    on_watchdog: Any = "kill-target-generation",
    tool_wait_board: Any = None,
    quorum_timeout_s: float | None = None,
    idle_flow_timeout_s: float | None = None,
    pause_margin: float | None = None,
    trainer_edges: Any = None,
    max_recovery_attempts: int | None = None,
) -> ElasticWiring:
    """``on_watchdog(tx_id, phase)`` runs on the watchdog thread when the absolute
    transaction deadline passes while a step is still blocked. Default
    (``"kill-target-generation"``): :func:`kill_target_generation` kills the
    Ray worker actors of the cells this transaction is starting/verifying (the
    target generation), so a blocked start/publish on them fails and the
    transaction goes to REBUILD_OLD. Old members are never killed; a blocked
    call on the OLD set (drain, REBUILD_OLD restart) is still not bounded.
    ``None`` disables the action (journal + flag only).

    ``trainer_edges`` (E3, 4.7; default None = trainer edges refused) is the
    ``IslandController(trainer_edges=...)`` callable returning
    ``{"spec", "args", "global_batch_size", "micro_batch_size", "ops"}``,
    ``ops`` being a :class:`.trainer_resize.MilesTrainerOps`.
    """
    from yeto.rl.elastic_benchmark.capabilities import load_attestation, parse_configs

    from ..controller import CommandInbox, IslandController, Timeouts
    from ..ledger import BatchLedger

    if isinstance(on_watchdog, str) and on_watchdog != "kill-target-generation":
        raise ValueError(f"unknown watchdog action {on_watchdog!r}")
    if not isinstance(resources, dict):
        resources = json.loads(Path(resources).expanduser().read_text(encoding="utf-8"))
    configs = parse_configs(resources)
    state = Path(state_dir).expanduser()
    controller = IslandController(
        state_dir=state,
        configs=configs,
        attestation=load_attestation(Path(attestation).expanduser() if attestation else None),
        profile=profile,
        initial_config=initial_config,
        runtime_fingerprint=runtime_fingerprint,
        timeouts=timeouts or Timeouts(),
        inbox=CommandInbox(state / "inbox"),
        on_watchdog=None if isinstance(on_watchdog, str) else on_watchdog,
        trainer_edges=trainer_edges,
        # 3.7 restart recovery: consecutive unverified recoveries before RECOVERY_REQUIRED
        **({} if max_recovery_attempts is None else {"max_recovery_attempts": int(max_recovery_attempts)}),
        # 3.8: the strict pause budget is min(margin x the syncer's
        # --quorum-timeout-s, measured idle-flow timeout); None keeps the
        # audited defaults (syncer default 900 s, margin 0.5).
        **{k: float(v) for k, v in (("quorum_timeout_s", quorum_timeout_s),
                                    ("idle_flow_timeout_s", idle_flow_timeout_s),
                                    ("pause_margin", pause_margin)) if v is not None},
    )
    if on_watchdog == "kill-target-generation":
        controller.set_on_watchdog(kill_target_generation(controller))
    return ElasticWiring(
        controller=controller,
        ledger=BatchLedger(state),
        declared_cells=tuple(str(c) for c in declared_cells),
        pool_gpus=None if pool_gpus is None else tuple(pool_gpus),
        tool_wait_board=tool_wait_board,
    )


def kill_target_generation(controller: Any, *, manager: Any = None, ray_module: Any = None,
                           timeout_s: float = 30.0):
    """Default watchdog action (3.7 / review H2): kill the target generation.

    For each cell :meth:`IslandController.watchdog_target_cells` names, read its
    workers from the fork's ``RayWorkerManager`` (named actor) and ``ray.kill``
    each worker actor of exactly the generation just read
    (``get_actor_handle(name, expected_generation=...)`` asserts it). The
    manager is not behind the InferenceController lock that a blocked
    ``update_weights``/``check_weights`` holds. Everything done (or failed) is
    journaled as ``watchdog_action``.
    """

    def on_watchdog(tx_id: str, phase: str) -> None:
        cells = controller.watchdog_target_cells()
        if not cells:
            controller.record_watchdog_action(tx_id, phase=phase, killed=[],
                                              note="no target generation in this phase")
            return
        if ray_module is None:
            import ray as ray_mod
        else:
            ray_mod = ray_module
        mgr = manager
        if mgr is None:
            from miles.utils.workers.ray_worker_manager import RayWorkerManager

            mgr = RayWorkerManager.get_handle()
        from .rollout import cell_of

        killed, errors, skipped = [], [], []
        for member in cells:
            # the controller speaks member ids ("engine:<cell>"); the fork's
            # RayWorkerManager is keyed by the bare cell id (GPU a4s3/a4s4: the
            # prefixed id matched nothing and nothing was killed)
            try:
                cell = cell_of(member)
                infos = ray_mod.get(mgr.get_worker_infos.remote(cell), timeout=timeout_s)
            except Exception as exc:  # noqa: BLE001
                # the fork does not know the id (declared/mapped wrongly): distinguishable
                errors.append({"cell": member, "fork_cell": cell, "kind": "unknown_target",
                               "error": repr(exc)})
                continue
            if not infos:
                # the cell exists but has no live worker actors (its engine already died /
                # was never registered, e.g. killed during start_cells - GPU d123 chain 3):
                # nothing to kill, and nothing keeps the step alive on our side either, so
                # this is journaled distinctly but is NOT an unresolved target (the fork call
                # fails or returns on its own and the transaction goes to REBUILD_OLD).
                # A27 (GPU 6r1/6r2 d2): the fork's liveness scan tears a cell down when its
                # workers die and reports it ``workers_lost`` (not ``stopped``); journal that
                # distinctly so "its engine died under us" and "never started" are told apart.
                lost = _lost_workers_of(mgr, ray_mod, cell, timeout_s)
                if lost is not None:
                    skipped.append({"cell": member, "fork_cell": cell, "kind": "workers_lost",
                                    "lost_workers": lost,
                                    "note": "the fork's liveness scan found the cell's workers dead "
                                            "and tore the cell down; nothing left to kill"})
                else:
                    skipped.append({"cell": member, "fork_cell": cell, "kind": "no_workers",
                                    "note": "cell has no live worker actors (not started or already stopped)"})
                continue
            for info in infos:
                try:
                    handle = ray_mod.get(
                        mgr.get_actor_handle.remote(info.name, expected_generation=info.generation),
                        timeout=timeout_s,
                    )
                    ray_mod.kill(handle, no_restart=True)
                    killed.append({"cell": member, "fork_cell": cell, "worker": info.name,
                                   "generation": info.generation})
                except Exception as exc:  # noqa: BLE001
                    errors.append({"cell": member, "fork_cell": cell, "worker": info.name,
                                   "kind": "kill_failed", "error": repr(exc)})
        controller.record_watchdog_action(tx_id, phase=phase, killed=killed, errors=errors,
                                          skipped=skipped)
        if errors:
            # a target the watchdog could not kill keeps the blocked step alive and its
            # engines unmanaged: do not wait silently, the island needs recovery
            controller.watchdog_unresolved(tx_id, phase, errors)

    return on_watchdog


def _lost_workers_of(mgr: Any, ray_mod: Any, cell: str, timeout_s: float) -> list[str] | None:
    """The worker names the fork's ``RayWorkerManager.describe_cells`` reports as lost for
    ``cell`` (state ``workers_lost``), or None when the cell is in another state or the fork
    cannot say (older fork without the state, or the lookup fails: never block the watchdog)."""
    describe = getattr(mgr, "describe_cells", None)
    if describe is None:
        return None
    try:
        described = ray_mod.get(describe.remote(pool_ids=None), timeout=timeout_s) or {}
    except Exception:  # noqa: BLE001 - classification only; the kill result stands on its own
        return None
    entry = described.get(cell) if isinstance(described, dict) else None
    if not isinstance(entry, dict) or entry.get("state") != "workers_lost":
        return None
    return [str(w) for w in (entry.get("lost_workers") or [])]


class LazyBoardActor:
    """The island's named ``ToolWaitBoard`` actor, looked up/created on first use
    (``tool_wait.board_actor``): the elastic wiring is built before Ray is
    connected. Attribute access forwards to the actor handle, so
    ``tool_wait.read_tool_wait`` works on it unchanged."""

    def __init__(self, learner_id: int, *, factory: Any = None) -> None:
        self.learner_id = int(learner_id)
        self._factory = factory
        self._handle = None

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if self._handle is None:
            factory = self._factory
            if factory is None:
                from ..tool_wait import board_actor as factory
            self._handle = factory(self.learner_id)
        return getattr(self._handle, name)
