"""verl hydra entry with yeto's task runner (runs inside the verl image).

``python -m yeto.rl.adapters.verl.verl_main <hydra overrides>``: verl's own
``main_ppo.main`` (config composition + ``validate_config``), with the V1 task
runner replaced by :class:`YetoTaskRunner`, which builds verl's trainer and
then hands the round loop to yeto's neutral ``IslandDriver`` over the verl
ports (``ports_impl``).  The island plan (sync mode, syncer, learner id,
canonical layout ...) comes from the JSON file named by ``$YETO_VERL_PLAN``.

Ray is initialised here (not by verl) so that the runtime environment of every
Ray worker -- and of the vLLM worker processes they spawn -- carries
``PYTHONPATH`` (yeto importable for the read-back hook) and the yeto env vars,
on top of verl's own ``get_ppo_ray_runtime_env``.
"""

from __future__ import annotations

import json
import os
import sys

import ray

PLAN_ENV = "YETO_VERL_PLAN"
FORWARD_ENV_PREFIXES = ("YETO_", "HF_", "PYTHONPATH", "SYNCER_ADDR")


@ray.remote(num_cpus=1)
class YetoTaskRunner:
    def run(self, config, plan: dict):
        import transfer_queue as tq
        from omegaconf import OmegaConf
        from verl.trainer.ppo.v1 import AgentLoopManagerTQ, get_trainer_cls
        from verl.utils.logging_utils import configure_verl_logging

        import yeto.rl.adapters.verl.trainer  # noqa: F401  (registers trainer_mode yeto_sync)

        configure_verl_logging()
        trainer_cls = get_trainer_cls(config.trainer.v1.trainer_mode)
        config.transfer_queue.enable = True
        OmegaConf.resolve(config)
        tq.init(config.transfer_queue)
        try:
            trainer = trainer_cls(config=config)
            trainer.init()
            manager = AgentLoopManagerTQ.create(
                config=config, llm_client=trainer.get_llm_client(),
                teacher_client=trainer.get_teacher_client(),
                reward_loop_worker_handles=trainer.get_reward_handles())
            from yeto.rl.adapters.verl.trainer import run_island

            return run_island(trainer, manager, plan, OmegaConf.to_container(config, resolve=True))
        finally:
            tq.close()


def _runtime_env() -> dict:
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

    env = get_ppo_ray_runtime_env(None)
    env.setdefault("env_vars", {})
    env["env_vars"]["TRANSFER_QUEUE_ENABLE"] = "1"
    for key, value in os.environ.items():
        if key.startswith(FORWARD_ENV_PREFIXES):
            env["env_vars"][key] = value
    return env


def main(argv: list[str] | None = None) -> None:
    import verl.trainer.main_ppo as main_ppo

    plan = json.loads(open(os.environ[PLAN_ENV]).read())
    if not ray.is_initialized():
        address = os.environ.get("RAY_ADDRESS") or None
        ray.init(address=address, runtime_env=_runtime_env(), namespace="yeto-verl")

    def run_ppo(config, task_runner_class=None):  # replaces verl's: our runner, our plan
        result = ray.get(YetoTaskRunner.remote().run.remote(config, plan))
        print("[yeto-verl] island result " + json.dumps(result, sort_keys=True, default=str), flush=True)

    main_ppo.run_ppo = run_ppo
    if argv is not None:
        sys.argv = [sys.argv[0], *argv]
    main_ppo.main()


if __name__ == "__main__":
    main()
