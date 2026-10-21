"""verl's own synchronous trainer on the Harbor agentic path (S25 N10.9; runs inside the verl image).

``python -m yeto.rl.adapters.verl.agentic_main <hydra overrides>`` is verl's
``main_ppo.main`` unchanged, with Ray started here so that every Ray worker:

* registers the ``yeto_grpo`` advantage estimator right after verl's
  ``core_algos`` is imported (Ray ``worker_process_setup_hook`` installs an
  import hook).  The hook itself must not import verl or torch: it runs
  before Ray sets the actor's ``CUDA_VISIBLE_DEVICES``, and an early CUDA
  init puts every rank on GPU 0 (NCCL "Duplicate GPU detected"; S24 T-GLM,
  S25 T-N109 run a).  The ``yeto_harbor`` agent loop is imported by verl
  through ``agent_loop_config_path``;
* carries the agentic environment: ``YETO_*`` (recorder, records proxy, boat
  caps), ``HARBOR_*`` (task directory, provider) and ``BOAT_API_KEY`` (the
  sandbox account, read by ``boat_client``), on top of verl's own runtime env.

The round loop, weight updates and rollout servers are verl's.  The records
proxy runs in each agent-loop worker over verl's rollout client
(:mod:`.rollout_engine`); select it with
``YETO_RECORDS_PROXY=yeto.rl.adapters.verl.rollout_engine:records_proxy_from_env``.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import os
import sys
from typing import Any

FORWARD_ENV_PREFIXES = ("YETO_", "HARBOR_", "HF_", "PYTHONPATH")
FORWARD_ENV_NAMES = ("BOAT_API_KEY",)
SETUP_HOOK = "yeto.rl.adapters.verl.agentic_main.setup_worker"


ESTIMATOR_HOST = "verl.trainer.ppo.core_algos"
ESTIMATOR_MODULE = "yeto.rl.adapters.verl.advantage"


class _AfterImport(importlib.abc.MetaPathFinder):
    """Import ``then`` right after ``target`` has been executed (no import of either before that)."""

    def __init__(self, target: str, then: str) -> None:
        self.target, self.then = target, then

    def find_spec(self, name: str, path: Any, target: Any = None) -> Any:
        if name != self.target:
            return None
        sys.meta_path.remove(self)  # find the real spec without this finder
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        loader, then = spec.loader, self.then

        class _Loader(importlib.abc.Loader):
            def create_module(self, spec: Any) -> Any:
                return loader.create_module(spec)

            def exec_module(self, module: Any) -> None:
                loader.exec_module(module)
                importlib.import_module(then)

        spec.loader = _Loader()
        return spec


def setup_worker() -> None:
    """Ray ``worker_process_setup_hook``: register yeto_grpo once verl's core_algos loads (module doc)."""
    if ESTIMATOR_HOST in sys.modules:
        importlib.import_module(ESTIMATOR_MODULE)
        return
    if not any(isinstance(f, _AfterImport) for f in sys.meta_path):
        sys.meta_path.insert(0, _AfterImport(ESTIMATOR_HOST, ESTIMATOR_MODULE))


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
    from yeto.launch.image_check import check_image_at_start

    check_image_at_start()  # before Ray: a mismatched image stops here with the failed checks
    import ray
    import verl.trainer.main_ppo as main_ppo

    importlib.import_module(ESTIMATOR_MODULE)  # the driver validates algorithm.adv_estimator too
    if not ray.is_initialized():
        ray.init(runtime_env=runtime_env(), namespace="yeto-verl-agentic")
    if argv is not None:
        sys.argv = [sys.argv[0], *argv]
    main_ppo.main()


if __name__ == "__main__":
    main()
