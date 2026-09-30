"""Same-shape trainer rebuild + full cut restore (rl-infra-spec 4.3; failure paths for 4.5).

Only the trainer subprocesses are rebuilt (fork-M6 ``rebuild_training_models``
with the startup sizes, i.e. same TP/PP/CP/EP/DP); learner, driver, bridge,
rollout manager and engines stay alive. The actor handle is shared by the
trainer, policy-state and publisher ports (and the eval dispatcher), so it is
held in a :class:`SwappableActor`: the driver keeps its port objects and only
the handle behind them changes (design D1: no ``rebind``, no second
``initialize``/``after_local_train``).

Preconditions checked before anything is disposed (4.1 audit, review H1):

* ``args.requested_load is None``. NOT ``args.load``: Miles parse sets
  ``requested_load = load`` and then, in bridge mode, overwrites ``args.load``
  with ``--ref-load`` and ``start_rollout_id`` with 0
  (``arguments.py:3426``, ``megatron_config.py:366``), so ``args.load`` is
  never None on the ports path;
* the run is not fault-tolerant / indep-DP / externally deployed (fork asserts
  the same).

``create_training_models`` calls ``rollout_executor.load(start_rollout_id-1)``
in the rollout process (``rollout_executor.py:304-310``: data source load plus
``generate_rollout.load``/``eval_generate_rollout.load``; stock rollout
functions are no-ops, the data source reads the rollout process' own
``args.load``, i.e. ``--ref-load`` in bridge mode, and returns when no dataset
state file exists there). Whatever a (custom) load does, the data cursor is
read through ``data_cursor()`` (``RolloutPool.data_cursor``; E1 interface)
before disposal and after the rebuild: any difference -> RECOVERY_REQUIRED.
yeto's ports engine never reads ``args.start_rollout_id`` (review L3: only the
legacy non-ports ``yeto/rl/miles.py`` and other runtimes do).

Failure semantics (fork ``TrainerRebuildError``: no automatic rollback):

* ``cleanup_error`` set -> workers may still hold bundles -> ``RECOVERY_REQUIRED``;
* otherwise rebuild again once more with a new attempt (same shape; the
  previous view when it was not restored) and restore the SAME cut
  (``REBUILD_OLD``); a second failure -> ``RECOVERY_REQUIRED``;
* once a new handle is swapped in, ANY exception (restore, layout read-back,
  cursor read) -> ``RECOVERY_REQUIRED``; training never resumes on it.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Any, Literal, Protocol

Outcome = Literal["RESTORED", "REBUILD_OLD", "RECOVERY_REQUIRED"]


class RecoveryRequired(RuntimeError):
    """The trainer cannot be brought back from the cut; stop data consumption."""

    def __init__(self, message: str, *, attempts: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.attempts = attempts


class SwappableActor:
    """Forwarding proxy over the upstream actor ``TrainGroup`` handle."""

    def __init__(self, target: Any) -> None:
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "generation", 0)

    @property
    def target(self) -> Any:
        return self._target

    def swap(self, new_target: Any) -> int:
        if not callable(getattr(new_target, "run_plugin", None)):
            raise RuntimeError("rebuilt actor group exposes no run_plugin")
        object.__setattr__(self, "_target", new_target)
        object.__setattr__(self, "generation", self.generation + 1)
        return self.generation

    async def dispose(self) -> None:
        """Resolve the CURRENT target at call time (review H3).

        Miles ``Disposer.add`` binds ``item.dispose`` when the item is added;
        through ``__getattr__`` that would be the startup target's method.
        """
        dispose = getattr(self._target, "dispose", None)
        if dispose is not None:
            result = dispose()
            if hasattr(result, "__await__"):
                await result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._target, name, value)


def rebuild_preconditions(args: Any) -> list[str]:
    out = []
    if getattr(args, "requested_load", None) is not None:
        out.append("--load was requested: create_training_models would resume trainer/data state from it")
    for flag in ("use_fault_tolerance", "indep_dp"):
        if getattr(args, flag, False):
            out.append(f"--{flag.replace('_', '-')} is not a same-shape rebuild path")
    if getattr(args, "trainer_controller_addrs", None) is not None:
        out.append("an independently deployed trainer is not rebuilt from here")
    return out


class DataCursorSource(Protocol):
    """Placeholder for E1's ``RolloutPool.data_cursor()`` (cut-audit.md §5)."""

    def data_cursor(self) -> Mapping[str, int]: ...


@dataclass
class RebuildResult:
    outcome: Outcome
    generation: int
    manifest: Any
    attempts: list[dict[str, Any]] = field(default_factory=list)


def _default_rebuild() -> Callable[..., Awaitable[tuple[Any, Any]]]:
    from miles.ray.placement_group import rebuild_training_models

    return rebuild_training_models


def _default_worker_manager() -> Any:
    from miles.utils.workers.ray_worker_manager import RayWorkerManager

    return RayWorkerManager.get_handle()


def _is_rebuild_error(error: BaseException) -> bool:
    return type(error).__name__ == "TrainerRebuildError" and hasattr(error, "stage")


def rebuild_same_shape(
    trainer: Any,
    *,
    args: Any,
    rollout_executor: Any,
    actor: SwappableActor,
    run: Callable[[Awaitable[Any]], Any],
    restore: Callable[[], Any],
    rollout: DataCursorSource,
    trainer_id: str = "actor",
    worker_manager: Any = None,
    rebuild: Callable[..., Awaitable[tuple[Any, Any]]] | None = None,
    max_attempts: int = 2,
) -> RebuildResult:
    """Dispose + rebuild the trainer (same shape) and restore the cut via ``restore()``.

    ``trainer`` is the :class:`~.trainer.MilesTrainerGroup`; its layout is
    read back from the running ranks before and after. ``restore`` calls
    ``trainer.restore_cut(...)`` for the cut taken before this call.
    """
    problems = rebuild_preconditions(args)
    if problems:
        raise RuntimeError("same-shape trainer rebuild refused: " + "; ".join(problems))
    layout = trainer.actual_layout()
    cursor = dict(rollout.data_cursor())
    rebuild = rebuild or _default_rebuild()
    manager = worker_manager if worker_manager is not None else _default_worker_manager()
    attempts: list[dict[str, Any]] = []
    view = None
    for attempt in range(max_attempts):
        try:
            new_actor, critic = run(
                rebuild(
                    args,
                    rollout_executor,
                    old_handles={trainer_id: actor.target},
                    worker_manager=manager,
                    trainer_pg_view=view,
                )
            )
        except Exception as error:  # noqa: BLE001
            if not _is_rebuild_error(error):
                attempts.append({"attempt": attempt, "stage": "unknown", "error": repr(error)})
                raise RecoveryRequired(f"trainer rebuild failed outside the fork contract: {error!r}",
                                       attempts=attempts) from error
            attempts.append({"attempt": attempt, "stage": error.stage, "error": repr(error.__cause__ or error),
                             "cleanup_error": repr(error.cleanup_error) if error.cleanup_error else None})
            if error.cleanup_error is not None:
                raise RecoveryRequired("trainer pools could not be stopped after a failed rebuild",
                                       attempts=attempts) from error
            if error.previous_view is not None and not error.view_restored:
                view = error.previous_view
            continue
        # From here on a new trainer exists: every failure is RECOVERY_REQUIRED (review M1).
        try:
            if critic is not None:
                raise RuntimeError("rebuilt trainer has a critic (ports engine drives none)")
            generation = actor.swap(new_actor)
            after = dict(rollout.data_cursor())
            if after != cursor:
                raise RuntimeError(f"data cursor changed across the trainer rebuild: {cursor} -> {after}")
            now = trainer.actual_layout()
            if now != layout:
                raise RuntimeError(f"layout changed across a same-shape rebuild: {layout} -> {now}")
            manifest = restore()
        except Exception as error:  # noqa: BLE001
            attempts.append({"attempt": attempt, "stage": "after_rebuild", "error": repr(error)})
            raise RecoveryRequired(f"trainer unusable after rebuild: {error}", attempts=attempts) from error
        outcome: Outcome = "RESTORED" if attempt == 0 else "REBUILD_OLD"
        attempts.append({"attempt": attempt, "stage": "done", "generation": generation})
        return RebuildResult(outcome, generation, manifest, attempts)
    raise RecoveryRequired(f"trainer rebuild failed {max_attempts} times", attempts=attempts)
