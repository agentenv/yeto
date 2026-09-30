"""Miles implementation of :class:`~yeto.rl.engine.trainer_transition.TrainerOps` (rl-infra-spec 4.7; E3).

Glue only: ``save_cut`` -> :meth:`MilesTrainerGroup.save_cut` with the
driver-supplied :class:`CutContext`; ``resize`` -> :func:`rebuild_resharded`
(fork-M6 rebuild on the target bundle view with resized args, resharded
restore, one fallback to the old shape); ``restore_source`` -> the same
rebuild in the source shape with an exact restore. The driver/entry
(INFRA-E1) supplies the callables below; see
the interface request in ``evidence/infra-e3/plan.md`` §5 (no entry patch yet).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .trainer_rebuild import rebuild_resharded, resized_args


@dataclass
class MilesTrainerOps:
    trainer: Any  # MilesTrainerGroup (over a SwappableActor)
    actor: Any  # SwappableActor
    rollout_executor: Any
    run: Callable[[Any], Any]
    rollout: Any  # has data_cursor()
    root: str
    context_for: Callable[[str], Any]  # cut_id -> CutContext (driver: progress, cursor, ledger, outer)
    expect_for: Callable[[Mapping[str, int]], Any]  # layout -> RestoreExpectation (source layout)
    view_for: Callable[[tuple[str, ...]], Any]  # trainer GPU ids -> placement-group view (M1 map)
    policy_hash_fn: Callable[[], str]
    certified_for: Callable[[Any], Any]  # plan -> certified spec hashes (A4)
    worker_manager: Any = None
    rebuild: Any = None

    def save_cut(self, *, epoch: int, cut_id: str) -> str:
        self._epoch = int(epoch)
        return self.trainer.save_cut(epoch=epoch, context=self.context_for(cut_id))

    def data_cursor(self) -> Mapping[str, int]:
        return dict(self.rollout.data_cursor())

    def actual_layout(self) -> Mapping[str, int]:
        return self.trainer.actual_layout()

    def policy_hash(self) -> str:
        return self.policy_hash_fn()

    def _restore_exact(self, cut_id: str, layout: Mapping[str, int]):
        return lambda: self.trainer.restore_cut(cut_id, epoch=self._epoch, root=self.root,
                                                expect=self.expect_for(layout))

    def resize(self, plan: Any, cut_id: str) -> Any:
        old_args = self.trainer._args
        new_args = resized_args(old_args, int(plan.reshard.target["world"]))
        return rebuild_resharded(
            self.trainer, old_args=old_args, new_args=new_args, rollout_executor=self.rollout_executor,
            actor=self.actor, run=self.run, rollout=self.rollout,
            restore_new=lambda: self.trainer.restore_cut_resharded(
                cut_id, epoch=self._epoch, root=self.root, expect=self.expect_for(plan.reshard.source),
                plan=plan.reshard, certified=self.certified_for(plan)),
            restore_old=self._restore_exact(cut_id, plan.reshard.source),
            new_layout=plan.reshard.target, old_layout=plan.reshard.source,
            new_view=self.view_for(plan.target_trainer_gpus), old_view=self.view_for(plan.source_trainer_gpus),
            worker_manager=self.worker_manager, rebuild=self.rebuild,
        )

    def restore_source(self, plan: Any, cut_id: str) -> Any:
        current = self.trainer._args
        old_args = resized_args(current, int(plan.reshard.source["world"]))
        restore = self._restore_exact(cut_id, plan.reshard.source)
        view = self.view_for(plan.source_trainer_gpus)
        # No further fallback: new == old shape; a failure raises RecoveryRequired.
        return rebuild_resharded(
            self.trainer, old_args=old_args, new_args=old_args, rollout_executor=self.rollout_executor,
            actor=self.actor, run=self.run, rollout=self.rollout, restore_new=restore, restore_old=restore,
            new_layout=plan.reshard.source, old_layout=plan.reshard.source, new_view=view, old_view=view,
            worker_manager=self.worker_manager, rebuild=self.rebuild,
        )
