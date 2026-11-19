"""Harbor environment provider for AgentENV (E2B-compatible firecracker microVMs).

Load it without changing Harbor::

    EnvironmentConfig(import_path="yeto.rl.harness.harbor.agentenv_env:AgentenvEnvironment")

(``build_trial_config`` does this for ``AgentTask.provider == "agentenv"``.)
Written against the pinned Harbor (``harbor-miles-v0.20.0@8f88560c``,
``harbor/environments/base.py``).  The work is in
:class:`~.agentenv_sandbox.AgentenvSandbox`; this class maps Harbor's interface
onto it, mirroring :class:`~.boat_env.BoatEnvironment`.

Unlike boat (a VM that runs the task image as a Docker container), an AgentENV
microVM *is* the task environment: there is no Docker layer, so exec and file
transfer land directly in the guest.  The task's ``[environment].docker_image``
names the OCI image/template the microVM boots from (AgentENV converts an OCI
image to a rootfs).  Refused before any microVM starts: tasks with no image,
compose tasks, and -- via the base capabilities -- GPUs/TPUs, Windows and
non-public networks.

``stop`` always deletes the microVM, also for ``delete=False`` (a stopped
microVM keeps nothing worth paying for without a snapshot).  With ``pool`` on
(:mod:`.agentenv_pool`), ``stop`` instead recycles the microVM for the next
task, unless an exec or file transfer failed on it; then it is deleted.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable
from pathlib import Path
from typing import Any, TypeVar

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities

from .agentenv_client import AgentenvQuota, process_client, process_quota
from .agentenv_pool import POOL_ENV, AgentenvSandboxPool, process_pool
from .agentenv_sandbox import AgentenvSandbox, SandboxSpec

T = TypeVar("T")

AGENTENV_IMPORT_PATH = "yeto.rl.harness.harbor.agentenv_env:AgentenvEnvironment"


class AgentenvEnvironment(BaseEnvironment):
    """One AgentENV microVM booted from the task image/template.

    Extra kwargs (``EnvironmentConfig.kwargs``): ``ttl_s`` (microVM auto-stop, a
    backstop for a lost delete), ``client`` (an
    :class:`~.agentenv_client.AgentenvClient`) and ``quota`` (an
    :class:`~.agentenv_client.AgentenvQuota`), both for tests.  Without ``quota``
    the environment takes the process quota, so it must be built inside the
    running loop (Harbor builds it in ``Trial.create``).

    ``pool``: ``True`` takes the process
    :class:`~.agentenv_pool.AgentenvSandboxPool`, a pool object takes that one,
    ``None`` (default) follows ``YETO_AGENTENV_POOL=1``, ``False`` never pools.
    """

    def __init__(self, *args: Any, ttl_s: int | None = None, client: Any = None,
                 quota: AgentenvQuota | None = None,
                 pool: bool | AgentenvSandboxPool | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._sandbox_kwargs: dict[str, Any] = {}
        if ttl_s is not None:
            self._sandbox_kwargs["ttl_s"] = ttl_s
        if pool is None:
            pool = os.environ.get(POOL_ENV, "").strip() == "1"
        self._pool: AgentenvSandboxPool | None = process_pool() if pool is True else (pool or None)
        self._sandbox: AgentenvSandbox | None = None
        self._failed_io = False
        if self._pool is None:
            self._sandbox = AgentenvSandbox(
                client if client is not None else process_client(),
                quota if quota is not None else process_quota(),
                name=self.session_id, template=self._template(), **self._sandbox_kwargs,
            )

    @staticmethod
    def type() -> str:
        return "agentenv"

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        # Public network only, no GPUs, not mounted.  TODO(verify on hardware):
        # AgentENV supports egress allow/deny rules, so ``disable_internet`` /
        # ``network_allowlist`` could be enabled once confirmed on the cluster.
        return EnvironmentCapabilities()

    def _template(self) -> str:
        """The AgentENV template / OCI image the microVM boots from.

        TODO(verify on hardware): yeto maps it from ``[environment].docker_image``
        (AgentENV converts an OCI image to a rootfs).  Confirm whether a task
        instead names a dedicated template id/alias field.
        """
        return self.task_env_config.docker_image

    def _validate_definition(self) -> None:
        if not self.task_env_config.docker_image:
            raise ValueError("agentenv provider needs [environment].docker_image (the OCI image / "
                             "template the microVM boots from); building from a Dockerfile is not supported")
        if self.extra_docker_compose_paths:
            raise ValueError("agentenv provider does not run docker compose task environments")

    @property
    def sandbox_id(self) -> str | None:
        return None if self._sandbox is None else self._sandbox.sandbox_id

    @property
    def usage_seconds(self) -> float | None:
        """Billable seconds of this microVM, read just before the delete (``None`` while pooled)."""
        return None if self._sandbox is None else self._sandbox.usage_seconds

    @property
    def start_timings(self) -> dict[str, float]:
        """Seconds per start phase (``create_s``, ...)."""
        return {} if self._sandbox is None else dict(self._sandbox.timings)

    def _box(self) -> AgentenvSandbox:
        if self._sandbox is None:
            raise RuntimeError("agentenv environment is not started")
        return self._sandbox

    async def start(self, force_build: bool) -> None:
        spec = SandboxSpec(
            template=self._template(),
            workdir=self.task_env_config.workdir,
            cpus=self._effective_cpus,
            memory_mb=self._effective_memory_mb,
            env=self._startup_env(),
        )
        if self._pool is not None:
            self._sandbox = await self._pool.acquire(spec, name=self.session_id, **self._sandbox_kwargs)
        else:
            await self._box().start(spec)
        try:
            await self.ensure_dirs(self._mount_targets(writable_only=True))
            await self._network_check()
        except BaseException as exc:
            self._failed_io = not isinstance(exc, asyncio.CancelledError)
            await self.stop(delete=True)
            raise

    async def _network_check(self) -> None:
        """Reserved call point for AgentENV egress / reachability setup.

        AgentENV denies egress to private ranges by default, so a microVM may
        need egress rules or a relay before it can reach the judge task-server /
        records-proxy.  That logic lives in a separate module ``agentenv_network``
        (owned elsewhere; interface in docs/NETWORK_DESIGN.md).  This hook calls
        it when present and is a no-op until it ships -- this provider does not
        implement reachability itself.
        """
        try:
            from . import agentenv_network  # type: ignore[attr-defined]
        except ImportError:
            return
        ensure = getattr(agentenv_network, "ensure_reachable", None)
        if ensure is not None:
            await ensure(self._box())

    async def stop(self, delete: bool) -> None:
        if self._sandbox is None:
            return
        if self._pool is not None:
            await self._pool.release(self._sandbox, reusable=not self._failed_io)
        else:
            await self._sandbox.close()

    async def _io(self, work: Awaitable[T]) -> T:
        # A failed exec/transfer may leave the microVM in an unknown state: never
        # reuse it.  A cancelled one (the trial was aborted) goes back through
        # the recycle check.
        try:
            return await work
        except asyncio.CancelledError:
            raise
        except BaseException:
            self._failed_io = True
            raise

    async def exec(self, command: str, cwd: str | None = None, env: dict[str, str] | None = None,
                   timeout_sec: int | None = None, user: str | int | None = None) -> ExecResult:
        out = await self._io(self._box().exec(command, cwd=cwd, env=self._merge_env(env),
                                              timeout_s=timeout_sec, user=self._resolve_user(user)))
        return ExecResult(stdout=out.stdout, stderr=out.stderr, return_code=out.return_code)

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        await self._io(self._box().upload_bytes(Path(source_path).read_bytes(), target_path))

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        await self._io(self._box().upload_dir(Path(source_dir), target_dir))

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(await self._io(self._box().download_bytes(source_path)))

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        await self._io(self._box().download_dir(source_dir, Path(target_dir)))
