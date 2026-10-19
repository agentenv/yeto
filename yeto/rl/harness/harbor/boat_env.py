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
stopped VM keeps nothing worth paying for).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities

from .boat_client import BoatQuota, process_client, process_quota
from .boat_sandbox import BoatSandbox, ContainerSpec, machine_type_for

BOAT_IMPORT_PATH = "yeto.rl.harness.harbor.boat_env:BoatEnvironment"


class BoatEnvironment(BaseEnvironment):
    """One boat VM running the task image as container ``main``.

    Extra kwargs (``EnvironmentConfig.kwargs``): ``machine_type`` (default: the
    smallest that fits the task), ``ttl_s`` (VM auto-stop, a backstop for a
    lost delete), ``client`` (a :class:`~.boat_client.BoatClient`) and
    ``quota`` (a :class:`~.boat_client.BoatQuota`), both for tests.  Without
    ``quota`` the environment takes the process quota, so it must be built
    inside the running loop (Harbor builds it in ``Trial.create``).
    """

    def __init__(self, *args: Any, machine_type: str | None = None, ttl_s: int | None = None,
                 client: Any = None, quota: BoatQuota | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        sandbox_kwargs: dict[str, Any] = {}
        if ttl_s is not None:
            sandbox_kwargs["ttl_s"] = ttl_s
        self._sandbox = BoatSandbox(
            client if client is not None else process_client(),
            quota if quota is not None else process_quota(),
            name=self.session_id,
            machine_type=machine_type or machine_type_for(self._effective_memory_mb, self._effective_cpus),
            **sandbox_kwargs,
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
        return self._sandbox.sandbox_id

    @property
    def usage_seconds(self) -> float | None:
        """Billable boat seconds of this VM, read just before the delete."""
        return self._sandbox.usage_seconds

    async def start(self, force_build: bool) -> None:
        await self._sandbox.start(ContainerSpec(
            image=self.task_env_config.docker_image,
            workdir=self.task_env_config.workdir,
            cpus=self._effective_cpus,
            memory_mb=self._effective_memory_mb,
            env=self._startup_env(),
        ))
        try:
            await self.ensure_dirs(self._mount_targets(writable_only=True))
        except BaseException:
            await self.stop(delete=True)
            raise

    async def stop(self, delete: bool) -> None:
        await self._sandbox.close()

    async def exec(self, command: str, cwd: str | None = None, env: dict[str, str] | None = None,
                   timeout_sec: int | None = None, user: str | int | None = None) -> ExecResult:
        out = await self._sandbox.exec(command, cwd=cwd, env=self._merge_env(env),
                                       timeout_s=timeout_sec, user=self._resolve_user(user))
        return ExecResult(stdout=out.stdout, stderr=out.stderr, return_code=out.return_code)

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        await self._sandbox.upload_bytes(Path(source_path).read_bytes(), target_path)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        await self._sandbox.upload_dir(Path(source_dir), target_dir)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(await self._sandbox.download_bytes(source_path))

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        await self._sandbox.download_dir(source_dir, Path(target_dir))
