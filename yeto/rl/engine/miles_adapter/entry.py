"""``--rl-engine ports`` composition root (tasks 5.1, design D2).

``run_ports_island`` is what ``yeto.rl.learner.run_miles`` calls on the ports
path, after the legacy-shared validation has produced the parsed upstream Miles
namespace (with the ``yeto_rl_*`` attributes the bridges read):

1. refuse a Miles checkout without ``run_plugin`` (before any component);
2. upstream init exactly as ``train.py`` does it -- ``init_orchestration_script``,
   ``create_rollout_components``, ``create_training_models`` -- inside one
   upstream ``Disposer`` on the island's persistent loop;
3. :func:`compose_island`: the adapter ports + sync session + ``IslandDriver``;
4. ``driver.run()``, then teardown through the disposer.

:func:`compose_island` takes the upstream objects as arguments so the CPU
integration test drives the real adapter classes over stubs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

from ..algorithm import BOUNDED_NONZERO_STD_FILTER, STOCK_NONZERO_STD_FILTER, AlgorithmSpec
from ..capabilities import EngineCapabilities
from . import LoopRunner

ENGINE_NAME = "miles-upstream"


def runtime_fingerprint(launch: Any, miles_commit: str) -> str:
    """Identity of the engine runtime this island drives."""

    payload = {"miles_commit": miles_commit, "argv": list(launch.argv)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def miles_capabilities(fingerprint: str) -> EngineCapabilities:
    """R0 ports capabilities (spec rl-engine-selection support matrix)."""

    return EngineCapabilities(
        engine=ENGINE_NAME,
        runtime_fingerprint=fingerprint,
        parameter_layouts={"lora"},
        placements={"colocated"},
        advantage_estimators={"grpo"},
        dynamic_sampling_filters={BOUNDED_NONZERO_STD_FILTER, STOCK_NONZERO_STD_FILTER},
        execution_modes={"colocated-serial"},
    )


def build_sync(miles_args: Any, *, yeto_policy_sync: bool) -> tuple[Any, Any]:
    """(sync session, progress store) for the preset the learner selected."""

    from ..bridges import (
        DecoupledIslandProgress,
        DecoupledSync,
        LocalOnlySync,
        StrictAvgSync,
        StrictIslandProgress,
    )

    if not yeto_policy_sync:
        return LocalOnlySync(int(miles_args.num_rollout)), None
    if getattr(miles_args, "yeto_rl_sync_preset", "strict-avg") == "decoupled":
        return DecoupledSync(miles_args), DecoupledIslandProgress(miles_args)
    progress = StrictIslandProgress(miles_args)
    return StrictAvgSync(miles_args.yeto_rl_bridge_config, progress=progress), progress


def compose_island(
    *,
    miles_args: Any,
    launch: Any,
    algorithm: AlgorithmSpec,
    inference_controller: Any,
    rollout_executor: Any,
    actor_model: Any,
    learner_id: int,
    base_model_revision: str,
    lora_config_hash: str,
    layout_hash: str,
    sync: Any,
    progress: Any,
    metadata: Any,
    capabilities: EngineCapabilities,
    runner: LoopRunner,
    events: Any = None,
    evaluate: Callable[[int], Any] | None = None,
    eval_interval: int | None = None,
    update_weights: Callable[..., Any] | None = None,
    release_refs: Callable[[Any, Any], None] | None = None,
    flatten_checksums: Callable[[Any], list[dict[str, Any]]] | None = None,
    verify_engine_checksums: bool = True,
    placement: Any = None,
):
    """Wire the adapter ports into an ``IslandDriver`` (no upstream imports)."""

    from yeto.rl.core import parse_policy_snapshot_token

    from ..driver import DriverError, EventTape, IslandDriver
    from .placement import MilesPlacement
    from .publish import MilesPublisher
    from .rollout import MilesRolloutPool
    from .state import MilesPolicyState
    from .trainer import MilesTrainerGroup

    holder: dict[str, IslandDriver] = {}

    def expected_policy() -> tuple[int, str]:
        driver = holder["driver"]
        if driver.expected_token is None:
            raise DriverError("rollout requested before any complete publication")
        return parse_policy_snapshot_token(driver.expected_token)

    policy_state = MilesPolicyState(
        actor_model=actor_model,
        base_model_revision=base_model_revision,
        config_hash=lora_config_hash,
        expected_layout_hash=layout_hash,
        runner=runner,
    )
    driver = IslandDriver(
        learner_id=learner_id,
        rollout=MilesRolloutPool(
            inference_controller=inference_controller,
            rollout_executor=rollout_executor,
            metadata=metadata,
            expected_policy=expected_policy,
            runner=runner,
            args=miles_args,
        ),
        trainer=MilesTrainerGroup(
            args=miles_args,
            actor_model=actor_model,
            learner_id=learner_id,
            learner_generation=0,
            parameter_layout_hash=lambda: layout_hash,
            algorithm=algorithm.advantage_estimator,
            release_refs=release_refs,
            runner=runner,
        ),
        policy_state=policy_state,
        publisher=MilesPublisher(
            args=miles_args,
            actor_model=actor_model,
            rollout_executor=rollout_executor,
            inference_controller=inference_controller,
            export_trainer_state=policy_state.export,
            verify_engine_checksums=verify_engine_checksums,
            update_weights=update_weights,
            flatten_checksums=flatten_checksums,
            runner=runner,
        ),
        placement=placement
        or MilesPlacement.from_parsed_args(launch.placement, miles_args),
        capabilities=capabilities,
        algorithm=algorithm,
        sync=sync,
        events=events
        or EventTape(miles_args.yeto_rl_event_tape, learner_id, args=miles_args),
        progress=progress,
        evaluate=evaluate,
        eval_interval=eval_interval,
    )
    holder["driver"] = driver
    return driver


def selection_event(*, launch: Any, algorithm: AlgorithmSpec, miles_commit: str) -> dict[str, Any]:
    return {
        "event": "rl_engine_selected",
        "rl_engine": "ports",
        "miles_commit": miles_commit,
        "rl/algorithm_spec_sha256": algorithm.sha256(),
        "placement": launch.placement.kind,
    }


def run_ports_island(
    miles_args: Any,
    launch: Any,
    algorithm: AlgorithmSpec,
    *,
    learner_id: int,
    base_model_revision: str,
    lora_config_hash: str,
    layout_hash: str,
    yeto_policy_sync: bool,
):
    """Run one ports-path island to completion on real upstream Miles."""

    from yeto.rl import MILES_NEXT_COMMIT
    from yeto.rl.miles import _append_rl_event

    from .state import require_run_plugin

    require_run_plugin()  # before any upstream component or model exists

    from miles.ray.placement_group import create_rollout_components, create_training_models
    from miles.ray.rollout.eval_dispatch import EvalDispatcher
    from miles.utils.async_utils import Disposer
    from miles.utils.orchestration_utils import init_orchestration_script

    from .rollout import RayMetadataSink

    _append_rl_event(
        miles_args,
        selection_event(launch=launch, algorithm=algorithm, miles_commit=MILES_NEXT_COMMIT),
    )
    runner = LoopRunner()
    disposer = Disposer()

    async def init():
        await disposer.__aenter__()
        init_orchestration_script(miles_args, disposer=disposer)
        controller, executor, _ = await create_rollout_components(miles_args)
        disposer.add(controller, executor)
        actor, critic = await create_training_models(miles_args, executor)
        if critic is not None:
            raise RuntimeError("the ports engine does not drive a critic")
        disposer.add(actor)
        dispatcher = EvalDispatcher(miles_args, actor, executor)
        disposer.add(dispatcher.drain)
        return controller, executor, actor, dispatcher

    error: BaseException | None = None
    try:
        controller, executor, actor, dispatcher = runner.run(init())
        if not callable(getattr(actor, "run_plugin", None)):
            raise RuntimeError("upstream actor group exposes no run_plugin")

        async def evaluate(rollout_id: int) -> dict[str, float]:
            await controller.prepare_eval()
            await dispatcher.dispatch(rollout_id)
            return {}

        sync, progress = build_sync(miles_args, yeto_policy_sync=yeto_policy_sync)
        driver = compose_island(
            miles_args=miles_args,
            launch=launch,
            algorithm=algorithm,
            inference_controller=controller,
            rollout_executor=executor,
            actor_model=actor,
            learner_id=learner_id,
            base_model_revision=base_model_revision,
            lora_config_hash=lora_config_hash,
            layout_hash=layout_hash,
            sync=sync,
            progress=progress,
            metadata=RayMetadataSink(),
            capabilities=miles_capabilities(runtime_fingerprint(launch, MILES_NEXT_COMMIT)),
            runner=runner,
            evaluate=lambda rollout_id: runner.run(evaluate(rollout_id)),
            eval_interval=getattr(miles_args, "eval_interval", None),
        )
        return driver.run()
    except BaseException as exc:
        error = exc
        raise
    finally:
        try:
            runner.run(
                disposer.__aexit__(
                    None if error is None else type(error),
                    error,
                    None if error is None else error.__traceback__,
                )
            )
        finally:
            runner.close()
