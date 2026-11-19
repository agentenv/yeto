"""One task microVM on AgentENV; no Harbor import.

An AgentENV sandbox is a firecracker microVM whose rootfs is the task image
(E2B semantics: the sandbox *is* the execution environment).  So, unlike
:class:`~.boat_sandbox.BoatSandbox` (a VM that then runs Docker), this class
has no ``docker run`` / ``docker exec`` / ``docker cp`` layer: it

1. creates the microVM from the task template/image (quota slot + create,
   cancel-safe so a create that returns after a cancel is still deleted),
2. runs every exec directly in the guest through the client's background
   command API (polled; a cancelled or timed-out exec is stopped remotely),
3. moves files straight into / out of the guest filesystem (a directory
   travels as one tar archive so modes, symlinks and empty dirs survive),
4. deletes the microVM in :meth:`close` (idempotent, also on a failed start),
   so a failed or cancelled trial leaves no sandbox.

A started microVM can be reused for the next task (:meth:`recycle` then
:meth:`reuse`, driven by :class:`~.agentenv_pool.AgentenvSandboxPool`).  Because
there is no container to throw away, :meth:`recycle` wipes the work and staging
directories and checks them empty; a microVM that will not come back clean is
deleted instead of reused.  Strong cross-task isolation on reuse is better
served by AgentENV's snapshot ``/fork`` (a fresh clone of a clean microVM) than
by an in-place wipe -- see :class:`~.agentenv_pool.AgentenvSandboxPool`.

The client is synchronous; calls run in a worker thread.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shlex
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from .agentenv_client import DEFAULT_TTL_S, AgentenvClient, AgentenvError, AgentenvQuota
from .boat_sandbox import extract_dir, pack_dir  # generic tar helpers, no Harbor dep

logger = logging.getLogger(__name__)
T = TypeVar("T")

# The guest paths the task owns; recycle wipes these and checks them empty.
STAGING_DIR = "/tmp/yeto-xfer"
DEFAULT_WORKDIR = "/home/user"


@dataclass(frozen=True)
class SandboxSpec:
    """What a task needs from its microVM.

    ``template`` is the AgentENV template / OCI image the microVM boots from
    (Harbor carries it as ``[environment].docker_image``; AgentENV converts an
    OCI image to a rootfs).  TODO(verify on hardware): confirm whether a task
    names its microVM by ``docker_image`` or by a dedicated template field.
    """

    template: str
    workdir: str | None = None
    cpus: int | None = None
    memory_mb: int | None = None
    env: Mapping[str, str] | None = None


@dataclass(frozen=True)
class ExecOutput:
    stdout: str
    stderr: str
    return_code: int


class AgentenvSandbox:
    def __init__(self, client: AgentenvClient, quota: AgentenvQuota, *, name: str,
                 template: str | None = None, ttl_s: int = DEFAULT_TTL_S,
                 poll_interval_s: float = 1.0, sleep: Callable[[float], object] = asyncio.sleep) -> None:
        self._client = client
        self._quota = quota
        self._name = name
        self._ttl_s = ttl_s
        self._poll_s = poll_interval_s
        self._sleep = sleep
        self._holds_slot = False
        self._workdir = DEFAULT_WORKDIR
        self.sandbox_id: str | None = None
        self.usage_seconds: float | None = None
        # ``template`` labels the current/last microVM image for pool pairing.
        self.template: str | None = template
        self.image: str | None = template  # pool uses ``image`` for same-image preference
        self.timings: dict[str, float] = {}
        self.reuse_count = 0
        self.forked = False  # a pristine snapshot clone; the pool kills it rather than recycling

    @property
    def name(self) -> str:
        return self._name

    # -- lifecycle -----------------------------------------------------------

    async def start(self, spec: SandboxSpec) -> None:
        await self._start(spec, run=True)

    async def start_vm(self, spec: SandboxSpec | None = None) -> None:
        """Create the microVM without preparing a task workdir (pool prewarm)."""
        await self._start(spec, run=False)

    async def _start(self, spec: SandboxSpec | None, *, run: bool) -> None:
        if self._holds_slot:
            raise AgentenvError("agentenv sandbox already started")
        self.timings = {}
        await self._quota.acquire()
        self._holds_slot = True
        try:
            started = time.monotonic()
            template = spec.template if spec is not None else (self.template or "")
            if not template:
                raise AgentenvError("agentenv sandbox needs a template/image to create a microVM")
            await self._create(template, spec)
            self.timings["create_s"] = time.monotonic() - started
            logger.info("agentenv sandbox %s created for %s (%s)", self.sandbox_id, self._name, template)
            self.template = self.image = template
            if run:
                assert spec is not None
                await self._prepare_workdir(spec)
        except BaseException:
            await self.close()
            raise

    async def _create(self, template: str, spec: SandboxSpec | None) -> None:
        # The create runs in a thread a cancellation cannot stop.  On cancel,
        # wait for it so the sandbox id is known and close() deletes the microVM;
        # otherwise it would run unseen until its TTL.
        env = dict(spec.env) if spec is not None and spec.env else None
        create = asyncio.ensure_future(asyncio.to_thread(
            self._client.create, template=template, ttl_seconds=self._ttl_s,
            metadata=self._metadata(), env=env,
            idempotency_key=f"yeto-{self._name}-{uuid.uuid4().hex}"))
        try:
            self.sandbox_id = await asyncio.shield(create)
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                self.sandbox_id = await create
            raise

    def _metadata(self) -> dict[str, str] | None:
        """Run / island / incarnation labels for orphan cleanup by a successor.

        TODO(verify on hardware): wire ``agentenv_labels`` (mirroring
        ``boat_labels``) once AgentENV's list/metadata filtering is confirmed.
        """
        return {"yeto_name": self._name}

    async def adopt(self, sandbox_id: str, spec: SandboxSpec, *, name: str) -> None:
        """Take over an already-created microVM (a snapshot ``/fork`` clone) for ``name``.

        The clone is a pristine copy of a clean source, so no recycle is needed;
        it holds its own quota slot (it is a live sandbox) and only its workdir
        is prepared.  On failure it is deleted, as in :meth:`start`.
        """
        if self._holds_slot:
            raise AgentenvError("agentenv sandbox already started")
        await self._quota.acquire()
        self._holds_slot = True
        self.sandbox_id = sandbox_id
        self.forked = True
        self._name = name
        self.template = self.image = spec.template
        self.timings = {"create_s": 0.0, "fork_s": 0.0}
        try:
            await self._prepare_workdir(spec)
        except BaseException:
            await self.close()
            raise

    async def _prepare_workdir(self, spec: SandboxSpec) -> None:
        self._workdir = spec.workdir or DEFAULT_WORKDIR
        await self._checked(f"mkdir -p {shlex.quote(self._workdir)} {STAGING_DIR}", "prepare workdir")

    # -- reuse ---------------------------------------------------------------

    async def recycle(self) -> None:
        """Wipe every trace of the task from the microVM and check it is clean.

        Removes the work and staging directories' contents; keeps the booted
        microVM.  Raises :class:`AgentenvError` when a directory is not empty
        afterwards, so the caller deletes the microVM instead of reusing it.

        TODO(verify on hardware): an in-place wipe cannot guarantee that a
        background process the previous task started is gone.  For strong
        isolation the pool should instead hand out a snapshot ``/fork`` of a
        clean microVM; this wipe is the cold fallback when fork is unavailable.
        """
        self._require_started()
        work = shlex.quote(self._workdir)
        await self._checked(
            f"rm -rf {work} {STAGING_DIR} && mkdir -p {work} {STAGING_DIR}", "recycle", timeout_s=120.0)
        await self._verify_clean()

    async def _verify_clean(self) -> None:
        work = shlex.quote(self._workdir)
        result = await self.vm_exec(
            f"echo w=$(ls -A {work} 2>/dev/null | wc -l) s=$(ls -A {STAGING_DIR} 2>/dev/null | wc -l)",
            timeout_s=60.0)
        if result.return_code != 0 or result.stdout.split() != ["w=0", "s=0"]:
            raise AgentenvError(f"agentenv sandbox {self.sandbox_id} not clean after recycle: "
                                f"{(result.stdout or result.stderr)[-500:]!r}")

    async def reuse(self, spec: SandboxSpec, *, name: str) -> None:
        """Prepare this (recycled) microVM for the next task ``name``.

        The microVM must have passed :meth:`recycle`.  On failure it is deleted,
        as in :meth:`start`.
        """
        self._require_started()
        self._name = name
        self.timings = {"create_s": 0.0}
        try:
            await self._verify_clean()
            await self._prepare_workdir(spec)
        except BaseException:
            await self.close()
            raise
        self.reuse_count += 1

    @property
    def ttl_s(self) -> int:
        return self._ttl_s

    async def set_ttl(self, ttl_s: int) -> None:
        """Move the microVM's auto-stop deadline to now + ``ttl_s`` (no-op without one)."""
        if self.sandbox_id is not None:
            await asyncio.to_thread(self._client.set_ttl, self.sandbox_id, ttl_s)

    def delete_now(self) -> None:
        """Delete the microVM from a synchronous context (process exit); errors are logged."""
        sandbox_id, self.sandbox_id = self.sandbox_id, None
        if sandbox_id is None:
            return
        try:
            self._client.delete(sandbox_id)
        except Exception:
            logger.exception("agentenv sandbox %s delete at exit failed; its TTL stops it", sandbox_id)

    async def close(self) -> None:
        """Delete the microVM (idempotent).  Errors are logged: the caller is often unwinding."""
        sandbox_id, self.sandbox_id = self.sandbox_id, None
        try:
            if sandbox_id is not None:
                await asyncio.shield(self._read_usage_and_delete(sandbox_id))
        except Exception:
            logger.exception("agentenv sandbox %s delete failed; delete it by hand", sandbox_id)
        finally:
            if self._holds_slot:
                self._holds_slot = False
                self._quota.release()

    async def _read_usage_and_delete(self, sandbox_id: str) -> None:
        try:
            self.usage_seconds = await asyncio.to_thread(self._client.usage_seconds, sandbox_id)
        except Exception:  # usage is a metric; never block the delete
            logger.warning("agentenv usage read failed for %s", sandbox_id)
        await asyncio.to_thread(self._client.delete, sandbox_id)
        logger.info("agentenv sandbox %s deleted (%s billable s)", sandbox_id, self.usage_seconds)

    # -- exec ----------------------------------------------------------------

    async def vm_exec(self, command: str, *, timeout_s: float | None, cwd: str | None = None,
                      env: Mapping[str, str] | None = None, user: str | int | None = None) -> ExecOutput:
        """Run ``command`` in the guest; past ``timeout_s`` it is stopped and RuntimeError raised.

        A cancel (trial abort) stops the remote command too.
        """
        sandbox_id = self._require_started()
        env_dict = dict(env) if env else None
        start = asyncio.ensure_future(asyncio.to_thread(
            self._client.start_command, sandbox_id, command, cwd=cwd, env=env_dict, user=user))
        try:
            process_id = await asyncio.shield(start)
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await self._stop_command(sandbox_id, await start)
            raise
        loop = asyncio.get_running_loop()
        deadline = None if timeout_s is None else loop.time() + timeout_s
        try:
            while True:
                status = await asyncio.to_thread(self._client.command_status, sandbox_id, process_id)
                if not status.running:
                    code = status.exit_code if status.exit_code is not None else -1
                    return ExecOutput(stdout=status.stdout, stderr=status.stderr, return_code=code)
                if deadline is not None and loop.time() > deadline:
                    await self._stop_command(sandbox_id, process_id)
                    raise RuntimeError(f"Command timed out after {timeout_s} seconds")
                await self._sleep(self._poll_s)
        except asyncio.CancelledError:
            await asyncio.shield(self._stop_command(sandbox_id, process_id))
            raise

    async def _stop_command(self, sandbox_id: str, process_id: int) -> None:
        try:
            await asyncio.to_thread(self._client.stop_command, sandbox_id, process_id)
        except Exception:
            logger.warning("agentenv stop_command %s/%s failed", sandbox_id, process_id)

    async def exec(self, command: str, *, cwd: str | None = None, env: Mapping[str, str] | None = None,
                   timeout_s: float | None = None, user: str | int | None = None) -> ExecOutput:
        """Run ``command`` with bash in the guest (directly, no container)."""
        return await self.vm_exec(wrap_bash(command), timeout_s=timeout_s, cwd=cwd, env=env, user=user)

    # -- files ---------------------------------------------------------------

    def _staging(self) -> str:
        return f"{STAGING_DIR}/{uuid.uuid4().hex}"

    async def _checked(self, command: str, what: str, timeout_s: float = 600.0) -> None:
        result = await self.vm_exec(command, timeout_s=timeout_s)
        if result.return_code != 0:
            raise AgentenvError(f"{what} failed ({result.return_code}): "
                                f"{(result.stderr or result.stdout)[-2000:]}")

    async def upload_bytes(self, data: bytes, target_path: str) -> None:
        sandbox_id = self._require_started()
        parent = shlex.quote(str(Path(target_path).parent))
        await self._checked(f"mkdir -p {parent}", f"mkdir for {target_path}")
        await asyncio.to_thread(self._client.put_file, sandbox_id, target_path, data, user=None)

    async def upload_dir(self, source_dir: Path, target_dir: str) -> None:
        sandbox_id = self._require_started()
        archive = await asyncio.to_thread(pack_dir, source_dir)
        staged = f"{self._staging()}.tar.gz"
        await self._checked(f"mkdir -p {STAGING_DIR}", "mkdir staging")
        await asyncio.to_thread(self._client.put_file, sandbox_id, staged, archive, user=None)
        s, t = shlex.quote(staged), shlex.quote(target_dir)
        await self._checked(f"mkdir -p {t} && tar -xzf {s} -C {t}; rc=$?; rm -f {s}; exit $rc",
                            f"upload dir to {target_dir}")

    async def download_bytes(self, source_path: str) -> bytes:
        sandbox_id = self._require_started()
        return await asyncio.to_thread(self._client.get_file, sandbox_id, source_path, user=None)

    async def download_dir(self, source_dir: str, target_dir: Path) -> None:
        sandbox_id = self._require_started()
        staged = f"{self._staging()}.tar.gz"
        src = shlex.quote(source_dir.rstrip("/") or "/")
        s = shlex.quote(staged)
        await self._checked(
            f"mkdir -p {STAGING_DIR} && tar -czf {s} -C {src} .", f"pack dir {source_dir}")
        archive = await self._then_clean(
            asyncio.to_thread(self._client.get_file, sandbox_id, staged, user=None),
            f"rm -f {s}")
        await asyncio.to_thread(extract_dir, archive, target_dir)

    async def _then_clean(self, work: Awaitable[T], cleanup: str) -> T:
        """Await ``work``, then run the guest ``cleanup`` command.

        When ``work`` raises, a failing cleanup is logged and the original error
        propagates (a ``finally`` would replace it with the cleanup's).
        """
        try:
            result = await work
        except BaseException:
            try:
                await self.vm_exec(cleanup, timeout_s=60)
            except Exception:  # the work's error is the one to report
                logger.warning("agentenv staging cleanup failed after an error: %s", cleanup, exc_info=True)
            raise
        await self.vm_exec(cleanup, timeout_s=60)
        return result

    def _require_started(self) -> str:
        if self.sandbox_id is None:
            raise AgentenvError("agentenv sandbox is not started (or already deleted)")
        return self.sandbox_id


def wrap_bash(command: str) -> str:
    """Run ``command`` under ``bash -c`` in the guest (matches boat's exec shell)."""
    return f"bash -c {shlex.quote(command)}"
