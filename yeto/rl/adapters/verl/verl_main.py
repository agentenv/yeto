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
FORWARD_ENV_PREFIXES = ("YETO_", "HF_", "VERL_FILE_LOGGER", "PYTHONPATH", "SYNCER_ADDR")


@ray.remote(num_cpus=1)
class YetoTaskRunner:
    def run(self, config, plan: dict):
        from yeto.island_credential_guard import check_island_credentials

        check_island_credentials()  # D4: the Ray actor that holds the syncer client
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
        # YETO_ covers YETO_ISLAND_HMAC_KEY (HELLO HMAC in the Ray actor that
        # opens the syncer client) and YETO_SANDBOX_MODAL_TOKEN_*; no cloud
        # credential matches these prefixes (secret-handling-hardening).
        if key.startswith(FORWARD_ENV_PREFIXES):
            env["env_vars"][key] = value
    return env


def _plan_and_ray() -> dict:
    from yeto.island_credential_guard import check_island_credentials

    check_island_credentials()  # secret-handling-hardening D4 (driver, both paths)
    plan = json.loads(open(os.environ[PLAN_ENV]).read())
    if not ray.is_initialized():
        address = os.environ.get("RAY_ADDRESS") or None
        ray.init(address=address, runtime_env=_runtime_env(), namespace="yeto-verl")
    return plan


def main(argv: list[str] | None = None) -> None:
    import verl.trainer.main_ppo as main_ppo

    plan = _plan_and_ray()

    def run_ppo(config, task_runner_class=None):  # replaces verl's: our runner, our plan
        result = ray.get(YetoTaskRunner.remote().run.remote(config, plan))
        print("[yeto-verl] island result " + json.dumps(result, sort_keys=True, default=str), flush=True)

    main_ppo.run_ppo = run_ppo
    if argv is not None:
        sys.argv = [sys.argv[0], *argv]
    main_ppo.main()


def main_fully_async(argv: list[str] | None = None) -> None:
    """agentic-rollout-utilization 6.4b: verl's ``fully_async_main.main`` (its hydra
    config ``fully_async_ppo_trainer`` + config fix-ups), with ``run_ppo``
    replaced so that :class:`~.fully_async_runner.YetoFullyAsyncTaskRunner` runs
    the island under the yeto driver."""
    import verl.trainer.main_ppo as main_ppo
    from verl.experimental.fully_async_policy import fully_async_main as fa_main

    plan = _plan_and_ray()

    def run_ppo(config, task_runner_class=None):
        from yeto.rl.adapters.verl.fully_async_runner import YetoFullyAsyncTaskRunner

        runner = ray.remote(num_cpus=1)(YetoFullyAsyncTaskRunner)
        result = ray.get(runner.remote().run.remote(config, plan))
        print("[yeto-verl] island result " + json.dumps(result, sort_keys=True, default=str), flush=True)

    main_ppo.run_ppo = run_ppo  # fully_async_main.main imports it at call time
    if argv is not None:
        sys.argv = [sys.argv[0], *argv]
    fully_async_hydra_entry(fa_main)()


def fully_async_hydra_entry(fa_main):
    """``fa_main.main`` re-decorated with an absolute config directory.

    verl's ``@hydra.main(config_path="config")`` resolves ``config`` as the
    package ``verl.experimental.fully_async_policy.config`` when the module is
    imported (not run with ``-m``); that directory has no ``__init__.py``, so
    hydra fails with "Primary config module ... not found" (S19 6.4b GPU run
    s19-verl64b-async1-20261009a).  A file-system path works either way."""
    import hydra

    config_dir = os.path.join(os.path.dirname(os.path.abspath(fa_main.__file__)), "config")
    inner = getattr(fa_main.main, "__wrapped__", None)
    if inner is None:
        raise RuntimeError("verl fully_async_main.main is not a hydra.main wrapper")
    return hydra.main(config_path=config_dir, config_name="fully_async_ppo_trainer",
                      version_base=None)(inner)


if __name__ == "__main__":
    main()
