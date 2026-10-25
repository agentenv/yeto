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
# S27 G-EQ2 (npu1): HCCL_ and ASCEND_ join the list because an Ascend island's
# collective and runtime settings are set on the island process (the launcher's
# activate script: HCCL_SOCKET_IFNAME, HCCL_CONNECT_TIMEOUT, HCCL_BUFFSIZE,
# ASCEND_RT_VISIBLE_DEVICES, ASCEND_GLOBAL_LOG_LEVEL ...). Without forwarding,
# a Ray worker of an already-running cluster starts with none of them and picks
# the wrong interface or the default buffer size. Neither prefix can carry a
# cloud credential (secret-handling-hardening: the credential names are
# AWS_, OSS_, MODAL_, NEBIUS_, HF_TOKEN; HF_ was already here and HF_TOKEN is
# refused by the island credential guard).
FORWARD_ENV_PREFIXES = ("YETO_", "HF_", "VERL_FILE_LOGGER", "PYTHONPATH", "SYNCER_ADDR",
                        "HCCL_", "ASCEND_")


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

        megatron = plan.get("engine") == "megatron"
        if megatron or plan.get("separate"):  # S27 G-DC / G-EK: registers trainer_mode yeto_separate
            import yeto.rl.adapters.verl.separate_trainer  # noqa: F401
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
            resolved = OmegaConf.to_container(config, resolve=True)
            if megatron:
                from yeto.rl.adapters.verl.megatron_ports import run_megatron_island

                return run_megatron_island(trainer, manager, plan, resolved)
            from yeto.rl.adapters.verl.trainer import run_island

            return run_island(trainer, manager, plan, resolved)
        finally:
            tq.close()


# NCCL host buffers on /dev/shm in every process of the disaggregated weight sync.
# On Modal one process could use cuMem host buffers while its peer fell back to
# /dev/shm, and the peer then failed to attach (S26 E0.1 eff01-yeto-verl-async-e,
# eff01-bare-verl-async-d with NCCL_DEBUG=INFO). The caller's own value wins.
CUMEM_HOST_ENV = "NCCL_CUMEM_HOST_ENABLE"


# Ray assigns each worker its own cards and writes them into this variable, so
# the driver's value must not be forwarded: a worker would see every card of the
# node and bind the wrong one (S27 G-EQ2). ``NPU_NOSET_ENV`` below is how an
# Ascend island keeps the cards visible where Ray would blank them instead.
NEVER_FORWARD_ENV = ("ASCEND_RT_VISIBLE_DEVICES",)


def _runtime_env(fully_async: bool = False, ascend_stack: str | None = None) -> dict:
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

    env = get_ppo_ray_runtime_env(None)
    env.setdefault("env_vars", {})
    env["env_vars"]["TRANSFER_QUEUE_ENABLE"] = "1"
    if fully_async:
        env["env_vars"][CUMEM_HOST_ENV] = os.environ.get(CUMEM_HOST_ENV, "0")
    if ascend_stack:  # S27 G-FL: install the Ascend plugin stack in every Ray worker
        from yeto.rl import ascend_plugin_stack

        os.environ[ascend_plugin_stack.ENV] = str(ascend_stack)  # forwarded by the YETO_ prefix below
        env["worker_process_setup_hook"] = ascend_plugin_stack.SETUP_HOOK
    for key, value in os.environ.items():
        # YETO_ covers YETO_ISLAND_HMAC_KEY (HELLO HMAC in the Ray actor that
        # opens the syncer client) and YETO_SANDBOX_MODAL_TOKEN_*; no cloud
        # credential matches these prefixes (secret-handling-hardening).
        if key.startswith(FORWARD_ENV_PREFIXES) and key not in NEVER_FORWARD_ENV:
            env["env_vars"][key] = value
    # S27 G-EQ2: the Ascend NOSET switch is set on the driver by
    # _keep_npu_visible_in_cpu_actors, but a Ray worker of an already-running
    # cluster inherits the raylet's environment, not the driver's, so it has to
    # travel in the runtime env as well (the caller's own value still wins).
    if os.environ.get(NPU_NOSET_ENV):
        env["env_vars"].setdefault(NPU_NOSET_ENV, os.environ[NPU_NOSET_ENV])
    return env


NPU_NOSET_ENV = "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"


def _keep_npu_visible_in_cpu_actors() -> None:
    """On an Ascend node, stop Ray from hiding the NPUs from CPU-only actors.

    Ray sets ``ASCEND_RT_VISIBLE_DEVICES=""`` in actors that request no NPU,
    such as ``YetoTaskRunner``. verl then reads ``is_npu_available=False`` there,
    and ``verl.third_party.vllm`` refuses the vLLM-Ascend version
    (S20 ModelArts 910B4 run, runY_try2). Only set on an NPU node; the
    caller's own value wins.
    """
    from yeto.hw.catalog import device_family

    if device_family() == "ascend":
        os.environ.setdefault(NPU_NOSET_ENV, "1")


def _plan_and_ray(fully_async: bool = False) -> dict:
    from yeto.island_credential_guard import check_island_credentials

    check_island_credentials()  # secret-handling-hardening D4 (driver, both paths)
    plan = json.loads(open(os.environ[PLAN_ENV]).read())
    # S27 G-DC: the Megatron island syncs weights over the same disaggregated NCCL
    # checkpoint engine as the fully_async path, so it gets the same NCCL host setting
    disaggregated = fully_async or plan.get("engine") == "megatron" or bool(plan.get("separate"))  # S27 G-EK
    stack = (plan.get("ascend_plugin_stack") or {}).get("root")
    if not ray.is_initialized():
        _keep_npu_visible_in_cpu_actors()
        address = os.environ.get("RAY_ADDRESS") or None
        ray.init(address=address, runtime_env=_runtime_env(disaggregated, ascend_stack=stack),
                 namespace="yeto-verl")
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

    plan = _plan_and_ray(fully_async=True)

    def run_ppo(config, task_runner_class=None):
        from yeto.rl.adapters.verl.fully_async_runner import YetoFullyAsyncTaskRunner

        runner = ray.remote(num_cpus=1)(YetoFullyAsyncTaskRunner)
        result = ray.get(runner.remote().run.remote(config, plan))
        print("[yeto-verl] island result " + json.dumps(result, sort_keys=True, default=str), flush=True)

    main_ppo.run_ppo = run_ppo  # fully_async_main.main imports it at call time
    if argv is not None:
        sys.argv = [sys.argv[0], *argv]
    fully_async_hydra_entry(fa_main)()


def main_buffer_trainer(argv: list[str] | None = None) -> None:
    """S25 C2: the fully_async hydra config and fix-ups as :func:`main_fully_async`,
    with :class:`~.buffer_trainer_runner.YetoBufferTaskRunner`: a trainer island
    over the sample buffer (the plan's ``sample_buffer`` section), no rollouter."""
    import verl.trainer.main_ppo as main_ppo
    from verl.experimental.fully_async_policy import fully_async_main as fa_main

    plan = _plan_and_ray(fully_async=True)

    def run_ppo(config, task_runner_class=None):
        from yeto.rl.adapters.verl.buffer_trainer_runner import YetoBufferTaskRunner

        runner = ray.remote(num_cpus=1)(YetoBufferTaskRunner)
        result = ray.get(runner.remote().run.remote(config, plan))
        print("[yeto-verl] island result " + json.dumps(result, sort_keys=True, default=str), flush=True)

    main_ppo.run_ppo = run_ppo
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
