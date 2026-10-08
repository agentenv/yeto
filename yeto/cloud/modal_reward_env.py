"""Cloud layer: Modal sandbox backend for the reward environment's TB2 adapter.

Moved out of ``yeto/rl/harness/reward_env/tb2.py`` (neutral core must not
import ``modal``; spec rl-framework-neutral-core).  The neutral adapter only
knows ``tb2_provider.SandboxBackend`` / ``Tb2EnvironmentProvider``; this module
is the Modal implementation, injected at launch with
``YETO_HARNESS_ENVIRONMENT_PROVIDER=yeto.cloud.modal_reward_env:modal_provider``.
Behaviour is unchanged from the original location.
"""

from __future__ import annotations

import os
import shlex
from typing import Any

from yeto.rl.harness.codex import tb2_provider as tb2
from yeto.rl.harness.reward_env import tb2 as rt
from yeto.rl.harness.reward_env.benchmark import PrebakePlan



def modal_image(plan: PrebakePlan, *, prebake: bool = True) -> Any:
    from modal import Image

    image = Image.from_registry(plan.base_image)
    if prebake and not plan.empty:
        image = image.run_commands("bash -lc " + shlex.quote(rt.prebake_script(plan)))
    return image


def _prebake_enabled() -> bool:
    return os.environ.get(rt.PREBAKE_ENV, "1") != "0"


class PrebakedModalSandboxBackend(tb2.ModalSandboxBackend):
    """``tb2_provider.ModalSandboxBackend`` with the task's prebaked image."""

    def __init__(self, *, prebake: bool = True, **kwargs: Any) -> None:
        kwargs.setdefault("app_name", rt.DEFAULT_APP)
        super().__init__(**kwargs)
        self.prebake = prebake

    def create(self, task: tb2.Tb2Task, trajectory_id: str) -> tb2.ModalSandbox:
        from modal import Sandbox

        plan = rt.prebake_from_test_sh((task.tests_dir / "test.sh").read_text(), task.docker_image)
        tags = {"yeto-tb2-task": task.task_id, "yeto-trajectory": trajectory_id[:63],
                "yeto-prebake": plan.digest()[:16] if self.prebake and not plan.empty else "none"}
        if self.run_id:
            tags["yeto-run-id"] = self.run_id[:63]
        sandbox = Sandbox.create(
            "sleep", "infinity", app=self._get_app(), image=modal_image(plan, prebake=self.prebake),
            timeout=self.ttl_s, idle_timeout=self.idle_timeout_s, cpu=float(task.cpus),
            memory=int(task.memory_mb), workdir=task.workdir, tags=tags,
        )
        return tb2.ModalSandbox(sandbox, task.workdir)


def modal_provider(miles_args: Any = None) -> tb2.Tb2EnvironmentProvider:
    """``YETO_HARNESS_ENVIRONMENT_PROVIDER=yeto.cloud.modal_reward_env:modal_provider``."""
    del miles_args
    tb2.require_modal_client()
    backend = PrebakedModalSandboxBackend(
        prebake=_prebake_enabled(),
        app_name=os.environ.get(rt.APP_ENV, rt.DEFAULT_APP),
        ttl_s=int(tb2._env_float(tb2.SANDBOX_TTL_ENV, tb2.DEFAULT_SANDBOX_TTL_S) or tb2.DEFAULT_SANDBOX_TTL_S),
        idle_timeout_s=int(tb2._env_float(tb2.IDLE_TIMEOUT_ENV, tb2.DEFAULT_IDLE_TIMEOUT_S)
                           or tb2.DEFAULT_IDLE_TIMEOUT_S),
    )
    return tb2._provider(backend)
