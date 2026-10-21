"""Harbor environment provider for boat.dev (S25 N10.8).

Load it without changing Harbor::

    EnvironmentConfig(import_path="yeto.rl.harness.harbor.boat_env:BoatEnvironment")

(``build_trial_config`` does this for ``AgentTask.provider == "boat"``.)
Written against the pinned Harbor (``harbor-miles-v0.20.0@8f88560c``,
``harbor/environments/base.py``).  The work is in
:class:`~.boat_sandbox.BoatSandbox`; this class maps Harbor's interface onto it.

Scope: single-container tasks with a prebuilt ``docker_image`` (all 89 local
TB2 tasks).  Refused before any VM starts: Dockerfile-only or compose tasks,
GPUs/TPUs, Windows, network modes other than public.  ``stop`` always
deletes the VM, also for ``delete=False`` (boat snapshots are off, so a
stopped VM keeps nothing worth paying for).  With ``pool`` on (S26 E8,
:mod:`.boat_pool`), ``stop`` instead recycles the VM for the next task,
unless an exec or file transfer failed on it; then it is deleted.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable
from pathlib import Path
from typing import Any, TypeVar

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities

from .boat_client import BoatQuota, process_client, process_quota
from .boat_pool import POOL_ENV, BoatSandboxPool, process_pool
from .boat_sandbox import BoatSandbox, ContainerSpec, machine_type_for

T = TypeVar("T")

BOAT_IMPORT_PATH = "yeto.rl.harness.harbor.boat_env:BoatEnvironment"


class BoatEnvironment(BaseEnvironment):
    """One boat VM running the task image as container ``main``.

    Extra kwargs (``EnvironmentConfig.kwargs``): ``machine_type`` (default: the
    smallest that fits the task), ``ttl_s`` (VM auto-stop, a backstop for a
    lost delete), ``client`` (a :class:`~.boat_client.BoatClient`) and
    ``quota`` (a :class:`~.boat_client.BoatQuota`), both for tests.  Without
    ``quota`` the environment takes the process quota, so it must be built
    inside the running loop (Harbor builds it in ``Trial.create``).

    ``pool`` (S26 E8): ``True`` takes the process
    :class:`~.boat_pool.BoatSandboxPool`, a pool object takes that one,
    ``None`` (default) follows ``YETO_BOAT_POOL=1``, ``False`` never pools.
    """

    def __init__(self, *args: Any, machine_type: str | None = None, ttl_s: int | None = None,
                 client: Any = None, quota: BoatQuota | None = None,
                 pool: bool | BoatSandboxPool | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._sandbox_kwargs: dict[str, Any] = {}
        if ttl_s is not None:
            self._sandbox_kwargs["ttl_s"] = ttl_s
        self._machine_type = machine_type or machine_type_for(self._effective_memory_mb, self._effective_cpus)
        if pool is None:
            pool = os.environ.get(POOL_ENV, "").strip() == "1"
        self._pool: BoatSandboxPool | None = process_pool() if pool is True else (pool or None)
        self._sandbox: BoatSandbox | None = None
        self._failed_io = False
        if self._pool is None:
            self._sandbox = BoatSandbox(
                client if client is not None else process_client(),
                quota if quota is not None else process_quota(),
                name=self.session_id, machine_type=self._machine_type, **self._sandbox_kwargs,
            )

    @staticmethod
    def type() -> str:
        return "boat"

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities()  # public network only, no GPUs, not mounted

    def _validate_definition(self) -> None:
        if not self.task_env_config.docker_image:
            raise ValueError("boat provider needs [environment].docker_image (prebuilt image); "
                             "building from a Dockerfile is not supported")
        if self.extra_docker_compose_paths:
            raise ValueError("boat provider does not run docker compose task environments")

    @property
    def sandbox_id(self) -> str | None:
        return None if self._sandbox is None else self._sandbox.sandbox_id

    @property
    def usage_seconds(self) -> float | None:
        """Billable boat seconds of this VM, read just before the delete (``None`` while pooled)."""
        return None if self._sandbox is None else self._sandbox.usage_seconds

    @property
    def start_timings(self) -> dict[str, float]:
        """Seconds per start phase (``create_s``, ``ready_s``, ``pull_s``, ``run_s``, ...)."""
        return {} if self._sandbox is None else dict(self._sandbox.timings)

    def _box(self) -> BoatSandbox:
        if self._sandbox is None:
            raise RuntimeError("boat environment is not started")
        return self._sandbox

    async def start(self, force_build: bool) -> None:
        spec = ContainerSpec(
            image=self.task_env_config.docker_image,
            workdir=self.task_env_config.workdir,
            cpus=self._effective_cpus,
            memory_mb=self._effective_memory_mb,
            env=self._startup_env(),
        )
        if self._pool is not None:
            self._sandbox = await self._pool.acquire(spec, name=self.session_id, machine_type=self._machine_type,
                                                     **self._sandbox_kwargs)
        else:
            await self._box().start(spec)
        try:
            await self.ensure_dirs(self._mount_targets(writable_only=True))
        except BaseException:
            self._failed_io = True
            await self.stop(delete=True)
            raise

    async def stop(self, delete: bool) -> None:
        if self._sandbox is None:
            return
        if self._pool is not None:
            await self._pool.release(self._sandbox, reusable=not self._failed_io)
        else:
            await self._sandbox.close()

    async def _io(self, work: Awaitable[T]) -> T:
        # A failed exec/transfer may leave the VM in an unknown state: never reuse it.
        try:
            return await work
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
