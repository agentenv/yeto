"""verl's own synchronous trainer on the Harbor agentic path (S25 N10.9; runs inside the verl image).

``python -m yeto.rl.adapters.verl.agentic_main <hydra overrides>`` is verl's
``main_ppo.main`` unchanged, with Ray started here so that every Ray worker:

* imports the yeto pieces verl selects by name -- the ``yeto_grpo`` advantage
  estimator (trainer process) and the ``yeto_harbor`` agent loop (agent-loop
  workers) -- through Ray's ``worker_process_setup_hook``;
* carries the agentic environment: ``YETO_*`` (recorder, records proxy, boat
  caps), ``HARBOR_*`` (task directory, provider) and ``BOAT_API_KEY`` (the
  sandbox account, read by ``boat_client``), on top of verl's own runtime env.

The round loop, weight updates and rollout servers are verl's.  The records
proxy runs in each agent-loop worker over verl's rollout client
(:mod:`.rollout_engine`); select it with
``YETO_RECORDS_PROXY=yeto.rl.adapters.verl.rollout_engine:records_proxy_from_env``.
"""

from __future__ import annotations

import os
import sys

FORWARD_ENV_PREFIXES = ("YETO_", "HARBOR_", "HF_", "PYTHONPATH")
FORWARD_ENV_NAMES = ("BOAT_API_KEY",)
SETUP_HOOK = "yeto.rl.adapters.verl.agentic_main.setup_worker"


def setup_worker() -> None:
    """Ray ``worker_process_setup_hook``: register the yeto estimator and agent loop."""
    import yeto.rl.adapters.verl.advantage  # noqa: F401  (registers yeto_grpo)
    import yeto.rl.adapters.verl.harbor_agent_loop  # noqa: F401  (registers yeto_harbor)


def runtime_env(environ: dict[str, str] | None = None) -> dict:
    """verl's PPO runtime env plus the forwarded agentic variables and the setup hook."""
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

    environ = dict(os.environ if environ is None else environ)
    env = get_ppo_ray_runtime_env(None)
    env.setdefault("env_vars", {})
    env["env_vars"]["TRANSFER_QUEUE_ENABLE"] = "1"  # verl's V1 trainer runs on TransferQueue
    for key, value in environ.items():
        if key.startswith(FORWARD_ENV_PREFIXES) or key in FORWARD_ENV_NAMES:
            env["env_vars"][key] = value
    env["worker_process_setup_hook"] = SETUP_HOOK
    return env


def main(argv: list[str] | None = None) -> None:
    import ray
    import verl.trainer.main_ppo as main_ppo

    setup_worker()  # the driver validates algorithm.adv_estimator against the registry too
    if not ray.is_initialized():
        ray.init(runtime_env=runtime_env(), namespace="yeto-verl-agentic")
    if argv is not None:
        sys.argv = [sys.argv[0], *argv]
    main_ppo.main()


if __name__ == "__main__":
    main()
