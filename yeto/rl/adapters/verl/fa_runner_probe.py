"""Ray probe of the fully_async task runner inside the verl image (Modal CPU
dry run, no GPU).  Prints one JSON line.

s19-verl64b-async7-20261010a: the trainer verl built was its own
``FullyAsyncTrainer``, not ``YetoFullyAsyncTrainer``.  This starts a local Ray
(CPU), creates the task runner as an actor the way ``verl_main`` does, and
reports which ``_create_trainer`` the actor resolves and whether verl's
methods see the real ``fully_async_main`` module globals.

Usage: python -m yeto.rl.adapters.verl.fa_runner_probe
"""

from __future__ import annotations

import json
import traceback


def _inspect(runner):
    import inspect
    import sys

    fa = sys.modules["verl.experimental.fully_async_policy.fully_async_main"]
    create = type(runner)._create_trainer
    init = type(runner)._initialize_components
    return {
        "mro": [c.__name__ for c in type(runner).__mro__][:5],
        "create_trainer_qualname": create.__qualname__,
        "create_builds_yeto": "YetoFullyAsyncTrainer.remote" in inspect.getsource(create),
        "verl_init_globals_is_module": init.__globals__ is fa.__dict__,
    }


def run() -> dict:
    import ray

    from yeto.rl.adapters.verl.fully_async_runner import YetoFullyAsyncTaskRunner, YetoFullyAsyncTrainer

    ray.init(num_cpus=2, include_dashboard=False)
    try:
        actor = ray.remote(num_cpus=1)(YetoFullyAsyncTaskRunner).remote()
        info = ray.get(actor.__ray_call__.remote(_inspect))
    finally:
        ray.shutdown()
    info["trainer_has_yeto_methods"] = hasattr(YetoFullyAsyncTrainer, "yeto_configure")
    info["pass"] = (info["create_trainer_qualname"].startswith("YetoFullyAsyncTaskRunner.")
                    and info["create_builds_yeto"] and info["trainer_has_yeto_methods"])
    return info


def main() -> None:
    try:
        out = run()
    except Exception as exc:  # noqa: BLE001
        out = {"pass": False, "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-3000:]}
    print("YETO_FA_RUNNER_PROBE " + json.dumps(out, default=str))


if __name__ == "__main__":
    main()
