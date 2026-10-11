"""Cloud layer: Modal sandbox backend for the reward environment's TB2 adapter.

Moved out of ``yeto/rl/harness/reward_env/tb2.py`` (neutral core must not
import ``modal``; spec rl-framework-neutral-core).  The neutral adapter only
knows ``tb2_provider.SandboxBackend`` / ``Tb2EnvironmentProvider``; this module
is the Modal implementation, injected at launch with
``YETO_HARNESS_ENVIRONMENT_PROVIDER=yeto.cloud.modal_reward_env:modal_provider``.
Behaviour is unchanged from the original location.
"""

from __future__ import annotations

import base64
import os
from typing import Any

from yeto.rl.harness.codex import tb2_provider as tb2
from yeto.rl.harness.reward_env import tb2 as rt
from yeto.rl.harness.reward_env.benchmark import PrebakePlan
from yeto.rl.rlf.modal import RlfCtfConfig, RlfModalProvider, RlfModalVmConfig



def modal_image(plan: PrebakePlan, *, prebake: bool = True) -> Any:
    from modal import Image

    image = Image.from_registry(plan.base_image)
    if prebake and not plan.empty:
        image = image.run_commands(prebake_run_command(plan))
    return image


def prebake_run_command(plan: PrebakePlan) -> str:
    """One Dockerfile ``RUN`` line for the multi-line prebake script.

    A Dockerfile instruction cannot span raw newlines (S17 G3: Modal rejected
    ``bash -lc '<multi-line>'`` with "could not parse Dockerfile"), so the
    script travels base64-encoded and runs as a login shell from a temp file.
    """
    b64 = base64.b64encode(rt.prebake_script(plan).encode()).decode()
    return (f"echo {b64} | base64 -d > /tmp/yeto-prebake.sh && bash -l /tmp/yeto-prebake.sh"
            " && rm -f /tmp/yeto-prebake.sh")


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
            **self._client_kwargs(),
            **self.network_policy.grant(task.task_id).modal_kwargs(),
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
        network_policy=tb2.NetworkPolicy.from_env(),
    )
    return tb2._provider(backend)


RLF_CTF_TASKS_DIR_ENV = "YETO_RLF_CTF_TASKS_DIR"
RLF_CTF_FLAGS_ENV = "YETO_RLF_CTF_FLAGS_JSON"  # {"task_id": "flag{...}"} (trusted only)
RLF_CTF_VM_IMAGE_ENV = "YETO_RLF_CTF_VM_IMAGE"
RLF_CTF_LEASE_SECONDS_ENV = "YETO_RLF_CTF_LEASE_SECONDS"


def rlf_ctf_config_from_env(environ: Any = None) -> RlfCtfConfig | None:
    """CTF packaging config, or None when the launch did not configure it.

    Fail closed: setting ``YETO_RLF_CTF_TASKS_DIR`` without flags (or with
    invalid JSON) raises instead of silently degrading to the VM-only lease.
    """
    import json
    from pathlib import Path

    environ = os.environ if environ is None else environ
    tasks_dir = environ.get(RLF_CTF_TASKS_DIR_ENV)
    if not tasks_dir:
        return None
    if not Path(tasks_dir).is_dir():
        raise RuntimeError(f"{RLF_CTF_TASKS_DIR_ENV}={tasks_dir}: not a directory")
    raw_flags = environ.get(RLF_CTF_FLAGS_ENV)
    if not raw_flags:
        raise RuntimeError(f"{RLF_CTF_TASKS_DIR_ENV} is set but {RLF_CTF_FLAGS_ENV} is not "
                           "(the trusted flags never live in the task package)")
    flags = json.loads(raw_flags)
    if not isinstance(flags, dict) or not all(isinstance(k, str) and isinstance(v, str) and v
                                              for k, v in flags.items()):
        raise RuntimeError(f"{RLF_CTF_FLAGS_ENV} must be a JSON object of task_id -> flag")
    lease = float(environ.get(RLF_CTF_LEASE_SECONDS_ENV) or 1800.0)
    return RlfCtfConfig(tasks_dir=tasks_dir, flags=flags,
                        vm_image=environ.get(RLF_CTF_VM_IMAGE_ENV) or "docker:27-dind",
                        lease_seconds=lease)


def rlf_modal_provider(miles_args: Any = None, *,
                       config: RlfModalVmConfig | None = None,
                       ctf: RlfCtfConfig | None = None) -> RlfModalProvider:
    """``YETO_HARNESS_ENVIRONMENT_PROVIDER=yeto.cloud.modal_reward_env:rlf_modal_provider``.

    Same factory contract as ``modal_provider`` (preflight calls
    ``factory(miles_args)`` and requires ``.acquire`` on the product), but each
    acquire opens one isolated Modal VM carrying RLF task/episode/trajectory
    identity instead of a TB2 sandbox.

    When CTF packaging is configured (``ctf=`` or ``YETO_RLF_CTF_TASKS_DIR`` +
    ``YETO_RLF_CTF_FLAGS_JSON``), the provider is a full ``EnvironmentProvider``:
    each acquire additionally runs the packaged nested-Docker CTF orchestration
    and the lease exposes env_url/env_token/verifier/destroy/describe/deadline
    (audit breakpoint 3).  Without it, the S20 VM-only lease is returned.
    """
    del miles_args
    return RlfModalProvider(config=config, ctf=ctf if ctf is not None else rlf_ctf_config_from_env())
