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

from yeto.rl.engine.round_cut import CutSource as _CoreCutSource


class CutSource(_CoreCutSource):
    """:class:`yeto.rl.engine.round_cut.CutSource` with the Miles in-flight export check."""

    @staticmethod
    def check_in_flight(exported: Mapping[str, Any]) -> list[str]:
        from .in_flight import check_export

        return check_export(exported)


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
