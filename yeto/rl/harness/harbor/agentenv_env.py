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
import logging
import os
from collections.abc import Awaitable, Mapping
from pathlib import Path
from typing import Any, TypeVar

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities

from . import breaker_settings
from .agentenv_client import AgentenvQuota, process_client, process_quota
from .agentenv_pool import POOL_ENV, AgentenvSandboxPool, process_pool
from .agentenv_sandbox import AgentenvSandbox, SandboxSpec

T = TypeVar("T")

logger = logging.getLogger(__name__)

AGENTENV_IMPORT_PATH = "yeto.rl.harness.harbor.agentenv_env:AgentenvEnvironment"

# Env-var names whose values are service URLs an AgentENV microVM must be able to
# reach before a trial can score, mapped to the logical sidecar name the
# reachability self-check reports under.  A microVM is a Firecracker guest, not a
# container on a docker bridge, so it reaches these only by egress to a routable
# address (:mod:`.agentenv_network`); a misconfigured or unreachable records /
# judge endpoint is the silent-zero (sbx-3) failure the self-check guards against.
#
#   TASK_SERVER_URL                 judge task-server, if the task declares one
#                                   (read from the task's ``[environment].env``)
#   YETO_RECORDS_PROXY_PUBLIC_BASE  the rollout records-proxy the microVM posts
#                                   trajectories to (set in the sampler process
#                                   env by the agentic launch, see agentic.env)
#
# NOTE: the per-trial sampler endpoint (``OPENAI_BASE_URL``) is injected into the
# *agent process* env by :func:`...trial.agent_env` from the session URL and is
# not visible to the environment object here, so it is not probed by this hook.
REACH_URL_ENV_VARS: tuple[tuple[str, str], ...] = (
    ("TASK_SERVER_URL", "task-server"),
    ("YETO_RECORDS_PROXY_PUBLIC_BASE", "records-proxy"),
)
# The self-check is the ``harness.reach_check`` breaker (:mod:`.breaker_settings`); these
# env vars are its aliases.  Skipping it (escape hatch only) re-exposes the
# silent-zero-burn failure mode.
SKIP_REACH_ENV = "YETO_AGENTENV_SKIP_REACH_CHECK"  # "1": enabled false
REACH_TRIES_ENV = "YETO_AGENTENV_REACH_TRIES"  # attempts_n
REACH_TIMEOUT_ENV = "YETO_AGENTENV_REACH_TIMEOUT_S"  # timeout_s


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
                 pool: bool | AgentenvSandboxPool | None = None,
                 reach_endpoints: Mapping[str, str] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Explicit ``{name: url}`` reachability targets (tests / advanced callers);
        # ``None`` discovers them from the trial + process env (:meth:`_reach_endpoints`).
        self._reach_override = dict(reach_endpoints) if reach_endpoints is not None else None
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

    def _reach_endpoints(self) -> dict[str, str]:
        """``{name: url}`` the microVM must reach, from the trial + process env.

        Explicit ``reach_endpoints`` (constructor) win; otherwise every
        :data:`REACH_URL_ENV_VARS` name that is **set and non-empty** -- in the
        task's ``[environment].env`` (:meth:`_startup_env`) or the sampler
        process env -- is collected.  A value that is set but malformed (e.g. a
        ``__TODO__`` placeholder) is kept on purpose: :func:`agentenv_network.
        self_check` validates the URL and fails loud rather than silently
        skipping a path the microVM needs.  An unset name is simply omitted (a
        self-contained task with a local verifier has nothing to reach).
        """
        if self._reach_override is not None:
            return dict(self._reach_override)
        startup = self._startup_env()
        out: dict[str, str] = {}
        for env_name, logical in REACH_URL_ENV_VARS:
            value = startup.get(env_name) or os.environ.get(env_name)
            if value:
                out[logical] = value
        return out

    async def _network_check(self) -> None:
        """Fail loud if the microVM cannot reach its judge / records endpoints.

        A microVM is not a container on a docker bridge, so it reaches the judge
        task-server and the rollout records-proxy only by egress to a routable
        address.  A misconfigured or unreachable endpoint makes every rollout
        score a silent zero while burning GPU time (the sbx-3 failure); this hook
        probes each endpoint with ``curl`` from *inside* the microVM via
        :mod:`.agentenv_network` and raises :class:`NetworkError` before any
        trial runs if a required one is unreachable.

        The probe uses ``curl``, else python, else a bash TCP connect, so task
        images without curl (several TB2 images) are probed too.
        """
        if not breaker_settings.enabled("harness.reach_check"):
            logger.warning("agentenv reachability self-check skipped (harness.reach_check off, %s): a "
                           "misconfigured judge/records endpoint will silently zero-score",
                           breaker_settings.source("harness.reach_check.enabled"))
            return
        try:
            from . import agentenv_network
        except ImportError:
            return
        mapping = self._reach_endpoints()
        if not mapping:
            logger.info("agentenv reachability self-check: no judge/records endpoint configured "
                        "(%s); nothing to probe", ", ".join(n for n, _ in REACH_URL_ENV_VARS))
            return

        timeout_s = float(breaker_settings.setting("harness.reach_check.timeout_s"))
        kwargs: dict[str, Any] = {"timeout_s": timeout_s,
                                  "tries": int(breaker_settings.setting("harness.reach_check.attempts_n")),
                                  "sleep_s": float(breaker_settings.setting("harness.reach_check.interval_s"))}

        box = self._box()

        async def probe(command: str) -> Any:
            # Raw guest exec (the SDK already wraps it in /bin/bash -c); allow the
            # curl its own --max-time plus margin for process start/stop overhead.
            return await box.vm_exec(command, timeout_s=timeout_s + 10.0)

        endpoints = agentenv_network.endpoints_from_mapping(mapping)
        await agentenv_network.self_check(probe, endpoints, **kwargs)
        logger.info("agentenv reachability self-check passed: %s", ", ".join(sorted(mapping)))

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
