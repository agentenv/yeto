"""4.4 wiring: controller -> save_cut -> same-shape rebuild -> restore_cut -> re-publish.

:func:`make_trainer_rebuilder` returns the ``trainer_rebuilder`` the
:class:`~..controller.IslandController` calls at a round-boundary safe point
(``request_trainer_rebuild``). It composes INFRA-E2's pieces without changing
them:

1. ``MilesTrainerGroup.save_cut`` with a :class:`~.trainer.CutContext` built
   from the driver (local step, published policy), the rollout data cursor,
   the batch ledger (3.6 ``cut_summary``) and the outer state (settled: the
   safe point is after the outer boundary returned). A refusal here raises
   :class:`~..controller.RebuildRefused` -- the trainer was not touched.
2. ``IslandDriver.rebuild_trainer(rebuild, cut_policy_hash=...)``: at the safe
   point, ``rebuild`` = ``trainer_rebuild.rebuild_same_shape`` (dispose +
   fork ``rebuild_training_models`` + ``SwappableActor.swap``) whose
   ``restore`` is ``MilesTrainerGroup.restore_cut`` of that cut. The driver
   keeps its port objects (no rebind, no ``initialize``, no
   ``after_local_train``, the sync session/bridge is not called), checks the
   restored policy hash and re-publishes the same policy to every member.
3. The driver's outer progress (rounds, local step, ledger, published
   version/token) is not touched: nothing is replayed.

Any failure after step 1 is RECOVERY_REQUIRED (the controller stops the
driver); ``rebuild_same_shape`` already retries once (REBUILD_OLD) inside.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


def _algorithm_identity(algorithm: Any, *, ref_model: Mapping[str, Any] | None) -> Any:
    from yeto.rl.engine.cut import AlgorithmIdentity

    runtime_attrs = None
    to_attrs = getattr(algorithm, "to_legacy_runtime_attrs", None)
    if callable(to_attrs):
        runtime_attrs = dict(to_attrs()) or None
    return AlgorithmIdentity.from_spec(algorithm, runtime_attrs=runtime_attrs, ref_model=ref_model)


class CutSource:
    """What the driver contributes to a cut (E2 ``CutContext``) and what a
    restore must match (``RestoreExpectation``); shared by the 4.4 same-shape
    rebuilder and E3's ``MilesTrainerOps`` (4.7)."""

    def __init__(self, *, driver: Any, trainer: Any, rollout: Any, ledger: Any, algorithm: Any,
                 backend_fingerprint: str, cut_root: str, global_batch_size: int,
                 ref_model: Mapping[str, Any] | None = None,
                 shared_filesystem: bool = True) -> None:
        self.driver_ref = driver  # the driver, or a zero-arg callable returning it
        self.trainer = trainer
        self.rollout = rollout
        self.ledger = ledger
        self.identity = _algorithm_identity(algorithm, ref_model=ref_model)
        self.backend_fingerprint = backend_fingerprint
        self.cut_root = cut_root
        self.global_batch_size = int(global_batch_size)
        self.shared_filesystem = shared_filesystem

    @property
    def driver(self) -> Any:
        ref = self.driver_ref
        return ref() if callable(ref) and not hasattr(ref, "published_state") else ref

    def context(self, cut_id: str) -> Any:
        from yeto.rl.engine.controller import RebuildRefused
        from yeto.rl.engine.cut import CutProgress
        from .trainer import CutContext

        driver = self.driver
        state = driver.published_state
        if state is None or driver.published_version is None:
            raise RebuildRefused("no published policy to cut")
        cursor = (self.rollout.data_cursor()
                  if callable(getattr(self.rollout, "data_cursor", None)) else None)
        if cursor is None:
            raise RebuildRefused("rollout data cursor unknown (needs the elastic metadata)")
        data = dict(cursor)
        summary = dict(self.ledger.cut_summary()) if self.ledger is not None else {}
        if summary.get("engine_buffer_length") is not None:
            data.setdefault("buffer_length", summary["engine_buffer_length"])
        in_flight: tuple = ()
        limit = int(getattr(getattr(driver, "profile", None), "max_policy_age", 0) or 0)
        if limit > 0:
            # agentic-rollout-utilization 3.3: the unfinished work the rollout
            # process holds NOW (buffered groups, suspended agentic trajectories).
            exporter = getattr(self.rollout, "export_in_flight", None)
            if not callable(exporter):
                raise RebuildRefused("the rollout pool cannot export its in-flight trajectories")
            from .in_flight import check_export

            exported = exporter()
            problems = check_export(exported)
            if problems:
                raise RebuildRefused("; ".join(problems))
            in_flight = tuple(exported["entries"])
            data["buffer_length"] = int(exported["buffer_groups"])
            summary["carried_over"] = len(in_flight)
            summary["max_policy_age"] = limit
        progress = CutProgress(
            local_step=int(driver.local_step),
            scheduler_samples=int(driver.local_step) * self.global_batch_size,
            global_batch_size=self.global_batch_size,
            next_rollout_id=int(driver.published_version),
            policy_version=int(state.policy_version),
            policy_hash=state.policy_tensor_hash(),
        )
        return CutContext(
            root=self.cut_root, cut_id=cut_id, backend_fingerprint=self.backend_fingerprint,
            progress=progress, algorithm=self.identity, data=data, ledger=summary,
            # The safe point is after boundary() returned non-stop and every
            # member acknowledged the policy: the outer commit is settled.
            outer={"settled": bool(driver.at_safe_point), "policy_version": state.policy_version,
                   "policy_token": driver.expected_token},
            shared_filesystem=self.shared_filesystem,
            in_flight=in_flight,
        )

    def expectation(self, layout: Mapping[str, int], *, epoch: int) -> Any:
        from yeto.rl.engine.cut import RestoreExpectation

        driver = self.driver
        return RestoreExpectation(
            algorithm=self.identity, layout=dict(layout),
            backend_fingerprint=self.backend_fingerprint, local_step=int(driver.local_step),
            policy_version=int(driver.published_state.policy_version), epoch=epoch,
        )


def make_trainer_rebuilder(
    *,
    trainer: Any,  # MilesTrainerGroup
    rollout: Any,  # MilesRolloutPool (data_cursor)
    ledger: Any,  # BatchLedger (cut_summary)
    algorithm: Any,  # AlgorithmSpec
    backend_fingerprint: str,
    cut_root: str,
    global_batch_size: int,
    rebuild_same_shape: Callable[..., Any],
    ref_model: Mapping[str, Any] | None = None,
    shared_filesystem: bool = True,
    preconditions: Callable[[], list[str]] | None = None,
) -> Callable[..., Mapping[str, Any]]:
    """``preconditions()`` (e.g. ``trainer_rebuild.rebuild_preconditions(args)``)
    is checked before anything is written: problems -> RebuildRefused.

    ``rebuild_same_shape(*, restore) -> RebuildResult`` is a closure over
    ``trainer_rebuild.rebuild_same_shape`` with the island's args, rollout
    executor, :class:`SwappableActor`, runner and data-cursor source."""

    from yeto.rl.engine.controller import RebuildRefused
    from yeto.rl.engine.cut import CutError

    def rebuilder(driver: Any, *, epoch: int, cut_id: str) -> Mapping[str, Any]:
        problems = list(preconditions()) if preconditions is not None else []
        if problems:
            raise RebuildRefused("same-shape trainer rebuild refused: " + "; ".join(problems))
        source = CutSource(driver=driver, trainer=trainer, rollout=rollout, ledger=ledger,
                           algorithm=algorithm, backend_fingerprint=backend_fingerprint,
                           cut_root=cut_root, global_batch_size=global_batch_size,
                           ref_model=ref_model, shared_filesystem=shared_filesystem)
        context = source.context(cut_id)
        progress = context.progress
        try:
            trainer.save_cut(epoch=epoch, context=context)
        except CutError as exc:
            raise RebuildRefused(f"cut refused: {exc}") from exc
        expect = source.expectation(trainer.actual_layout(), epoch=epoch)
        before = (driver.rounds_completed, driver.local_step, driver.published_version,
                  driver.expected_token)

        def restore() -> Any:
            return trainer.restore_cut(cut_id, epoch=epoch, root=cut_root, expect=expect,
                                       shared_filesystem=shared_filesystem)

        result = driver.rebuild_trainer(lambda: rebuild_same_shape(restore=restore),
                                        cut_policy_hash=progress.policy_hash)
        after = (driver.rounds_completed, driver.local_step, driver.published_version,
                 driver.expected_token)
        if after != before:
            raise RuntimeError(f"outer progress moved across the trainer rebuild: {before} -> {after}")
        return {
            "cut_id": cut_id,
            "outcome": getattr(result, "outcome", None),
            "generation": getattr(result, "generation", None),
            "attempts": list(getattr(result, "attempts", []) or []),
            "policy_hash": progress.policy_hash,
            "local_step": progress.local_step,
        }

    return rebuilder
