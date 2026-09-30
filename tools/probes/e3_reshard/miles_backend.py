"""Miles backend of the E3 harness (GPU container only; fork 5c1b49eb, image 5c1b49e-9f29303).

Every arm is a fresh trainer in a fresh driver process: ``create_rollout_components``
+ ``create_training_models`` with ``actor_num_gpus_per_node = dp`` and the
frozen rollouts replayed through Miles' own ``--load-debug-rollout-data``
(``debug_train_only``: no SGLang engine). ``RolloutExecutor.get`` then runs
``convert_samples_to_train_data`` and the PRODUCTION ``split_train_data_by_dp``
with the train_parallel_config the new trainer advertised (so the scheduled
split is exercised inside the fork's object store; the rank probe reads the
shard back). ``generate_frozen`` writes the frozen rollouts once with
``--save-debug-rollout-data`` (``debug_rollout_only``, base policy).

Nothing in this module runs on CPU; its CPU coverage is the dry-run of
``harness.py`` with a fake backend. DEV-GATHER (2xA10G) is its first run.
"""

from __future__ import annotations

import copy
from typing import Any

SCHEDULE = {"cp_size": 1, "vpp_size": 1, "microbatch_group_size_per_vp_stage": 1}


class MilesBackend:
    def __init__(self, miles_args: Any, algorithm: Any, *, frozen_template: str, fingerprint: str) -> None:
        from yeto.rl.engine.miles_adapter import LoopRunner
        from yeto.rl.engine.miles_adapter.entry import connect_island_ray
        from yeto.rl.engine.miles_adapter.state import require_run_plugin

        require_run_plugin()
        connect_island_ray()
        self.base_args = miles_args
        self.algorithm = algorithm
        self.frozen_template = frozen_template
        self.fingerprint = fingerprint
        self.global_batch_size = int(miles_args.global_batch_size)
        self.micro_batch_size = int(getattr(miles_args, "micro_batch_size", 1) or 1)
        self.runner = LoopRunner()
        self.trainer = None
        self._actor = self._executor = self._disposer = None

    # ------------------------------------------------------------------ helpers
    def _args(self, **overrides: Any) -> Any:
        args = copy.copy(self.base_args)
        args.actor_num_nodes = 1
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def _open(self, args: Any, *, trainer: bool):
        from miles.ray.placement_group import create_rollout_components, create_training_models
        from miles.utils.async_utils import Disposer
        from miles.utils.orchestration_utils import init_orchestration_script

        disposer = Disposer()

        async def init():
            await disposer.__aenter__()
            init_orchestration_script(args, disposer=disposer)
            controller, executor, _ = await create_rollout_components(args)
            disposer.add(controller, executor)
            actor = None
            if trainer:
                actor, critic = await create_training_models(args, executor)
                if critic is not None:
                    raise RuntimeError("critic not supported")
                disposer.add(actor)
            return executor, actor

        executor, actor = self.runner.run(init())
        self._disposer, self._executor, self._actor = disposer, executor, actor
        return executor, actor

    def _close(self) -> None:
        if self._disposer is not None:
            self.runner.run(self._disposer.__aexit__(None, None, None))
        self._disposer = self._executor = self._actor = self.trainer = None

    # ------------------------------------------------------------------ phases
    def generate_frozen(self, count: int) -> None:
        args = self._args(debug_rollout_only=True, save_debug_rollout_data=self.frozen_template,
                          load_debug_rollout_data=None)
        executor, _ = self._open(args, trainer=False)
        try:
            # split needs a config; dp=1 here, the arms re-split the saved samples themselves.
            self.runner.run(executor.set_train_parallel_config({"dp_size": 1, **SCHEDULE}))
            for rollout_id in range(count):
                self.runner.run(executor.get(rollout_id))
        finally:
            self._close()

    def start_arm(self, dp: int) -> None:
        from yeto.rl.engine.miles_adapter.trainer import MilesTrainerGroup

        args = self._args(actor_num_gpus_per_node=int(dp), load_debug_rollout_data=self.frozen_template,
                          debug_train_only=True, save_debug_rollout_data=None)
        _, actor = self._open(args, trainer=True)
        self.trainer = MilesTrainerGroup(args=args, actor_model=actor, learner_id=0, learner_generation=0,
                                         parameter_layout_hash=lambda: "e3-harness", runner=self.runner,
                                         spec=self.algorithm)

    def plugin(self, fn_path: str, kwargs: dict | None = None) -> list:
        return list(self.runner.run(self._actor.run_plugin(fn_path, kwargs or {})))

    def train(self, rollout_id: int) -> dict:
        from yeto.rl.engine.miles_adapter.state_plugin import APPLIED_LRS, GRAD_NORM, STEP_LOSSES

        pack = self.runner.run(self._executor.get(rollout_id))
        try:
            outputs = list(self.runner.run(self._actor.train(rollout_id, pack)) or [])
        finally:
            self.trainer._release(self.trainer._args, pack)
        norms = [float(n) for n in self.plugin(GRAD_NORM)]
        lrs = [list(v) for v in self.plugin(APPLIED_LRS)]
        self.plugin(STEP_LOSSES)  # drained so a later save_cut sees a step boundary
        return {"grad_norm": max(norms), "grad_norms": norms, "applied_lrs": lrs, "outputs": len(outputs)}

    def stop_arm(self) -> None:
        self._close()
