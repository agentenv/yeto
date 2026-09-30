"""Miles backend of the E3 harness (GPU container only; Miles pin and image from yeto/rl/__init__.py).

Every arm is a fresh trainer in a fresh driver process: ``create_rollout_components``
+ ``create_training_models`` with ``actor_num_gpus_per_node = dp`` and the
frozen rollouts replayed through Miles' own ``--load-debug-rollout-data``
(``debug_train_only``: no SGLang engine). ``RolloutExecutor.get`` then runs
``convert_samples_to_train_data`` and the PRODUCTION ``split_train_data_by_dp``
with the train_parallel_config the new trainer advertised (so the scheduled
split is exercised inside the fork's object store; the rank probe reads the
shard back). ``generate_frozen`` writes the frozen rollouts once with
``--save-debug-rollout-data``, following upstream ``train.py``'s start-up
order exactly (create rollout components -> create the trainer (launcher size, colocated) ->
``update_weights`` -> ``onload_kv`` when rollout is offloaded -> per rollout
``prepare_rollout`` + ``get``), no training step, so the samples come from
the base policy. DEV-GATHER run 3 (B3) hung here: the earlier version set
``debug_rollout_only`` after parse and never pushed weights / onloaded the KV
cache of the colocated engines, which then answered /generate with 400/503.

Nothing in this module runs on CPU; its CPU coverage is the dry-run of
``harness.py`` with a fake backend. DEV-GATHER (2xA10G) is its first run.
"""

from __future__ import annotations

import copy
from typing import Any

# What Miles parse_args derives from --load-debug-rollout-data (fork 2f23a0fc arguments.py:3563-3568,
# 3692-3694); set after parse they must be set together, otherwise the colocated rollout GPUs still
# declare engine cells beyond the trainer-only placement group (DEV-GATHER run 6, IndexError in
# RayWorkerManager.init -> _CellManager.bundles).
ARM_PARSE_DERIVED = {"debug_train_only": True, "rollout_num_gpus": 0, "starts_inference_engines": False}

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
        self._controller = None
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
            self._controller = controller
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
        self._disposer = self._executor = self._actor = self.trainer = self._controller = None

    # ------------------------------------------------------------------ phases
    def generate_frozen(self, count: int, progress=None, *, identity: dict | None = None,
                        metadata_sink=None) -> None:
        """Frozen rollouts through the PRODUCTION round components, in the IslandDriver order.

        See ``evidence/infra-e3/dev-gather-parity.md``. Per rollout r (no train step; base policy):
        publish(state0, token r) [MilesPublisher: onload_weights (not first) -> update_weights ->
        onload_kv -> start_update_weights / update_weight_version(token) / end_update_weights ->
        check_weights checksum] -> trainer.offload() -> MilesRolloutPool.generate(r) [sink token ->
        prepare_rollout -> executor.get (saves the samples) -> offload rollout -> metadata take] ->
        release refs -> trainer.onload().
        """
        from yeto.rl.engine.miles_adapter.publish import MilesPublisher
        from yeto.rl.engine.miles_adapter.rollout import MilesRolloutPool, RayMetadataSink
        from yeto.rl.engine.miles_adapter.state import MilesPolicyState
        from yeto.rl.engine.miles_adapter.trainer import MilesTrainerGroup

        identity = dict(identity or {})
        say = progress or (lambda _msg: None)
        # Keep the launcher's trainer size: in the colocated profile every engine shares a GPU with a
        # trainer rank; a DP=1 trainer next to 2 engines is a "hybrid colocated+distributed" deployment
        # whose LoRA weight sync Miles refuses (DEV-GATHER run 4, cuda_ipc.py:98).
        args = self._args(save_debug_rollout_data=self.frozen_template, load_debug_rollout_data=None)
        executor, actor = self._open(args, trainer=True)
        try:
            say("gen components up")
            # entry.run_ports_island: metadata=RayMetadataSink() (DEV-GATHER run 5)
            sink = metadata_sink if metadata_sink is not None else RayMetadataSink()
            state_port = MilesPolicyState(
                actor_model=actor, base_model_revision=identity.get("base_model_revision", ""),
                config_hash=identity.get("lora_config_hash", ""),
                expected_layout_hash=identity.get("layout_hash"), runner=self.runner)
            trainer = MilesTrainerGroup(args=args, actor_model=actor, learner_id=0, learner_generation=0,
                                        parameter_layout_hash=lambda: identity.get("layout_hash", ""),
                                        runner=self.runner, spec=self.algorithm)
            state = state_port.export(policy_version=0)  # LocalOnlySync.start: export at version 0
            publisher = MilesPublisher(args=args, actor_model=actor, rollout_executor=executor,
                                       inference_controller=self._controller,
                                       export_trainer_state=state_port.export, runner=self.runner)
            current = {"version": 0}
            pool = MilesRolloutPool(inference_controller=self._controller, rollout_executor=executor,
                                    metadata=sink, runner=self.runner, args=args,
                                    expected_policy=lambda: (current["version"], state.policy_tensor_hash()))
            colocated = bool(getattr(args, "colocate", False))
            for rollout_id in range(count):
                current["version"] = rollout_id
                publisher.publish(state, token_rollout_id=rollout_id)
                say(f"gen published token {rollout_id}")
                if colocated:
                    trainer.offload()
                batch = pool.generate(rollout_id)
                trainer._release(args, batch.payload)
                say(f"gen rollout {rollout_id} saved ({len(batch.groups)} groups)")
                if colocated:
                    trainer.onload()
        finally:
            self._close()

    def start_arm(self, dp: int) -> None:
        from yeto.rl.engine.miles_adapter.trainer import MilesTrainerGroup

        args = self._args(actor_num_gpus_per_node=int(dp), world_size=int(dp), **ARM_PARSE_DERIVED,
                          load_debug_rollout_data=self.frozen_template, save_debug_rollout_data=None)
        _, actor = self._open(args, trainer=True)
        self.trainer = MilesTrainerGroup(args=args, actor_model=actor, learner_id=0, learner_generation=0,
                                         parameter_layout_hash=lambda: "e3-harness", runner=self.runner,
                                         spec=self.algorithm)

    def plugin(self, fn_path: str, kwargs: dict | None = None) -> list:
        return list(self.runner.run(self._actor.run_plugin(fn_path, kwargs or {})))

    def train(self, rollout_id: int) -> dict:
        from yeto.rl.engine.miles_adapter.state_plugin import APPLIED_LRS, GRAD_NORM, STEP_LOSSES

        # IslandDriver.run_round: colocated -> trainer.onload() before the train step.
        if bool(getattr(self.trainer._args, "colocate", False)):
            self.trainer.onload()
        pack = self.runner.run(self._executor.get(rollout_id))
        try:
            outputs = list(self.runner.run(self._actor.train(rollout_id, pack)) or [])
        finally:
            self.trainer._release(self.trainer._args, pack)
        # upstream train.py: offload (or clear memory) after every train step
        self.trainer.offload()
        norms = [float(n) for n in self.plugin(GRAD_NORM)]
        lrs = [list(v) for v in self.plugin(APPLIED_LRS)]
        self.plugin(STEP_LOSSES)  # drained so a later save_cut sees a step boundary
        return {"grad_norm": max(norms), "grad_norms": norms, "applied_lrs": lrs, "outputs": len(outputs)}

    def stop_arm(self) -> None:
        self._close()
